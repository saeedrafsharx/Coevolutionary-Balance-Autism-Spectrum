"""ABIDE PCP acquisition helpers, embedded in the self-contained Kaggle notebook.

Downloads existing derivatives; does not preprocess, trim, harmonize, or calculate fALFF.
"""
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

S3_PREFIX = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENO_URL = f"{S3_PREFIX}/Phenotypic_V1_0b_preprocessed1.csv"
REPO_RAW = "https://raw.githubusercontent.com/preprocessed-connectomes-project/abide/master"
STRATEGIES = ("filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal")
SUCCESS = {"downloaded", "verified_existing"}
SOURCE_URLS = {
    "pcp_download.html": "https://preprocessed-connectomes-project.org/abide/download.html",
    "pcp_pipelines.html": "https://preprocessed-connectomes-project.org/abide/Pipelines.html",
    "pcp_cpac.html": "https://preprocessed-connectomes-project.org/abide/cpac.html",
    "pcp_quality_assessment.html": "https://preprocessed-connectomes-project.org/abide/quality_assessment.html",
    "pipeline_config_abide_rerun.yml": f"{REPO_RAW}/preprocessing/settings/pipeline_config_abide_rerun.yml",
    "abide_rois.txt": f"{REPO_RAW}/preprocessing/settings/abide_rois.txt",
}
ATLAS_URL = f"{REPO_RAW}/preprocessing/resources/abide_rois/CC200.nii.gz"
YEO_URL = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/Yeo_JNeurophysiol11_MNI152.zip"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nifti_header(header):
    """Read NIfTI-1 metadata without interpreting a 3D fALFF header as scan TR."""
    if len(header) < 348:
        raise ValueError("Truncated NIfTI-1 header")
    endian = next((e for e in ("<", ">") if struct.unpack(e + "i", header[:4])[0] == 348), None)
    if endian is None or header[344:348] != b"n+1\x00":
        raise ValueError("Expected single-file NIfTI-1 (.nii.gz)")
    dims = struct.unpack_from(endian + "8h", header, 40)
    if not 1 <= dims[0] <= 7 or any(v <= 0 for v in dims[1:dims[0] + 1]):
        raise ValueError("Invalid NIfTI dimensions")
    bitpix = struct.unpack_from(endian + "h", header, 72)[0]
    offset = struct.unpack_from(endian + "f", header, 108)[0]
    if bitpix not in (8, 16, 32, 64, 128, 256) or not math.isfinite(offset) or offset < 352 or offset != int(offset):
        raise ValueError("Invalid NIfTI bit depth or voxel offset")
    zooms = struct.unpack_from(endian + "8f", header, 76)
    unit = header[123] & 56
    tr = None
    if dims[0] >= 4 and unit in (8, 16, 24):
        tr = zooms[4] * {8: 1, 16: .001, 24: .000001}[unit]
        if not math.isfinite(tr) or tr <= 0:
            tr = None
    return {
        "shape": list(dims[1:dims[0] + 1]), "voxel_sizes": list(zooms[1:4]),
        "bitpix": bitpix, "vox_offset": int(offset), "TR_seconds": tr,
        "time_unit_code": unit, "raw_pixdim4": zooms[4],
    }


def validate_nifti(path, expected_ndim=3):
    # Read through EOF: gzip verifies both stream completeness and CRC.
    with gzip.open(path, "rb") as stream:
        header = stream.read(348)
        meta = nifti_header(header)
        total = 348
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
    if len(meta["shape"]) != expected_ndim:
        raise ValueError(f"Expected {expected_ndim}D NIfTI, got {meta['shape']}")
    expected = meta["vox_offset"] + math.prod(meta["shape"]) * meta["bitpix"] // 8
    if total < expected:
        raise ValueError(f"Truncated NIfTI voxel data: {total} < {expected}")
    return meta


def validate_roi(path):
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[0] < 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("ROI time series must be a finite time-by-region matrix")
    with Path(path).open() as stream:
        first = stream.readline().strip()
    labels = first.replace("#", " ").split() if first.startswith("#") else []
    return {
        "n_volumes": int(values.shape[0]), "n_rois": int(values.shape[1]),
        "roi_labels": labels, "roi_header": first if first.startswith("#") else "",
        "constant_roi_count": int(np.sum(np.ptp(values, axis=0) == 0)),
        "warning": "" if values.shape[1] == 200 else "ROI count differs from manuscript's 200; inspect atlas/header",
    }


def validate_yeo_archive(path):
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        names = archive.namelist()
        if corrupt or not any('7Networks' in name and name.endswith('.nii.gz') for name in names):
            raise ValueError("Corrupt Yeo archive or missing seven-network parcellation")
    # Preserve the original archive. Select thin/thick cortex and resample explicitly during analysis.
    return {"members": names}


def fetch_file(url, destination, validator=None, retries=3, timeout=60, offline=False):
    """Atomic, checked downloads. No strategy fallback; failure is a manifest row.

    Existing files require the matching URL receipt, stored hash and fresh validation.
    A bare filename is never sufficient evidence that a download succeeded.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_name(path.name + ".receipt.json")
    result = {"url": url, "path": str(path), "status": "failed", "error": "", "checked_utc": utc_now()}
    if path.exists() and receipt_path.exists():
        try:
            old = json.loads(receipt_path.read_text())
            if old["url"] == url and old["sha256"] == sha256(path):
                details = validator(path) if validator else {}
                return {**old, **result, "status": "verified_existing", "details": details}
        except Exception:
            pass  # A corrupt/unverifiable cache must be downloaded again.
    if offline:
        result.update(status='not_downloaded_offline', error='No verified cached file; network disabled for this run')
        return result
    partial = path.with_name(path.name + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ABIDE-PCP-revision-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, partial.open("wb") as output:
                length = response.headers.get("Content-Length")
                shutil.copyfileobj(response, output, length=1024 * 1024)
                headers = {k: response.headers.get(k) for k in ("ETag", "Last-Modified", "Content-Length")}
            if partial.stat().st_size == 0 or (length and partial.stat().st_size != int(length)):
                raise ValueError("Empty or incomplete HTTP response")
            try:
                details = validator(partial) if validator else {}
            except Exception:
                # Keep complete but rejected files as evidence for debugging legacy formats.
                rejected = path.with_name(path.name + '.rejected')
                shutil.copy2(partial, rejected)
                result['rejected_path'] = str(rejected)
                result['rejected_sha256'] = sha256(rejected)
                raise
            checksum = sha256(partial)
            size = partial.stat().st_size
            partial.replace(path)
            result.update(status="downloaded", sha256=checksum, size_bytes=size,
                          headers=headers, details=details, downloaded_utc=utc_now())
            write_json(receipt_path, result)
            return result
        except Exception as exc:
            partial.unlink(missing_ok=True)
            result["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                result["status"] = "unavailable"
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    # Previously cached but untrusted files are not marked successful.
    return result


def validate_pheno(path):
    frame = pd.read_csv(path)
    required = {"SUB_ID", "FILE_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ", "func_mean_fd"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Phenotype columns missing: {sorted(required - set(frame.columns))}")
    if frame["SUB_ID"].isna().any() or frame["SUB_ID"].duplicated().any():
        raise ValueError("Missing or duplicated phenotype SUB_ID")
    return {"n_rows": len(frame), "columns": frame.columns.tolist()}


def select_subjects(pheno, subject_ids):
    ids = [int(s) for s in subject_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("SUBJECT_IDS must be nonempty and unique")
    frame = pheno.copy()
    frame["SUB_ID"] = pd.to_numeric(frame["SUB_ID"], errors="raise").astype(int)
    if frame["SUB_ID"].duplicated().any():
        raise ValueError("Duplicated phenotype SUB_ID")
    selected = frame.set_index("SUB_ID").reindex(ids).reset_index()
    selected["metadata_found"] = selected["SUB_ID"].isin(frame["SUB_ID"])
    selected["downloadable_file_id"] = selected["FILE_ID"].fillna("").astype(str).str.match(r"^[A-Za-z0-9_-]+_\d+$")
    numeric = lambda name: pd.to_numeric(selected[name], errors="coerce").replace(-9999, np.nan)
    selected["flag_age_outside_18_30_or_missing"] = ~numeric("AGE_AT_SCAN").between(18, 30)
    selected["flag_not_male_or_missing"] = numeric("SEX").ne(1)
    selected["flag_FIQ_not_above_80_or_missing"] = ~(numeric("FIQ") > 80)
    selected["flag_mean_FD_above_0_5_or_missing"] = numeric("func_mean_fd").isna() | (numeric("func_mean_fd") > .5)
    # Audit flags only. Preserve legacy participants, including missing clinical data and small sites.
    return selected


def build_plan(selected, root):
    rows = []
    for subject in selected.to_dict("records"):
        for strategy in STRATEGIES:
            derivatives = ["rois_cc200", "falff"]
            if strategy.startswith("nofilt_"):
                derivatives.append("func_mask")
            for derivative in derivatives:
                extension = "1D" if derivative.startswith("rois_") else "nii.gz"
                file_id = subject.get("FILE_ID")
                valid = bool(subject["downloadable_file_id"])
                filename = f"{file_id}_{derivative}.{extension}" if valid else ""
                relative = f"cpac/{strategy}/{derivative}/{filename}" if valid else ""
                rows.append({
                    "SUB_ID": int(subject["SUB_ID"]), "FILE_ID": file_id if valid else "",
                    "SITE_ID": None if pd.isna(subject.get("SITE_ID")) else subject["SITE_ID"], "strategy": strategy,
                    "derivative": derivative,
                    "url": f"{S3_PREFIX}/Outputs/{relative}" if valid else "",
                    "relative_path": relative, "status": "pending" if valid else "missing_file_id",
                })
    return rows


def fetch_scan_header(file_id, destination, retries=3):
    """Fetch only the initial decompressed header from one 4D derivative.

    The server may ignore Range; the response is still closed after the header is read.
    No 4D image is retained, and a header digest is not a full-image checksum.
    """
    url = f"{S3_PREFIX}/Outputs/cpac/nofilt_noglobal/func_preproc/{file_id}_func_preproc.nii.gz"
    path = Path(destination)
    row = {"url": url, "status": "failed", "error": "", "checked_utc": utc_now()}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
            with urllib.request.urlopen(req, timeout=60) as response:
                with gzip.GzipFile(fileobj=response) as compressed:
                    header = compressed.read(348)
            meta = nifti_header(header)
            if len(meta["shape"]) != 4:
                raise ValueError("Expected 4D functional header")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(header)
            return {**row, **meta, "status": "downloaded", "header_path": str(path),
                    "header_sha256": hashlib.sha256(header).hexdigest()}
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404):
                row["status"] = "unavailable"
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    return row


def export_manifest(root, records):
    """Checkpoint during acquisition so interrupted runs retain their audit trail."""
    root = Path(root)
    write_json(root / "metadata/download_manifest.json", records)
    flat = [{**{k: v for k, v in row.items() if k not in ("details", "headers")},
             "details_json": json.dumps(row.get("details", {})),
             "headers_json": json.dumps(row.get("headers", {}))} for row in records]
    pd.DataFrame(flat).to_csv(root / "metadata/download_manifest.csv", index=False)


def subject_completeness(selected, records, scan_headers):
    rows = []
    for subject in selected.to_dict("records"):
        sub_id = int(subject["SUB_ID"])
        files = [r for r in records if r["SUB_ID"] == sub_id]
        row = {"SUB_ID": sub_id, "FILE_ID": subject.get("FILE_ID"), "SITE_ID": subject.get("SITE_ID")}
        for strategy in STRATEGIES:
            needed = [r for r in files if r["strategy"] == strategy and r["derivative"] in ("falff", "rois_cc200")]
            row[strategy + "_complete"] = len(needed) == 2 and all(r["status"] in SUCCESS for r in needed)
        row["all_four_strategies_complete"] = all(row[s + "_complete"] for s in STRATEGIES)
        row["recommended_pair_complete"] = all(
            r["status"] in SUCCESS for r in files
            if (r["derivative"] == "rois_cc200" and r["strategy"].startswith("filt_"))
            or (r["derivative"] == "falff" and r["strategy"].startswith("nofilt_"))
        ) and len(files) == 10
        row["all_requested_files_complete"] = len(files) == 10 and all(r["status"] in SUCCESS for r in files)
        h = scan_headers.get(sub_id, {})
        row["TR_seconds"] = h.get("TR_seconds")
        row["TR_source_url"] = h.get("url", "")
        row["TR_status"] = h.get("status", "not_attempted")
        row["header_n_volumes"] = h.get("shape", [None] * 4)[3]
        row["nyquist_Hz"] = .5 / row["TR_seconds"] if row["TR_seconds"] else None
        row["TR_at_least_2_5_seconds"] = row["TR_seconds"] >= 2.5 if row["TR_seconds"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def compare_strategies(records):
    """Hash/ROI comparisons are provenance diagnostics, not proof of preprocessing order."""
    comparisons = []
    by_key = {(r["SUB_ID"], r["strategy"], r["derivative"]): r for r in records}
    for sid in sorted(set(r["SUB_ID"] for r in records)):
        for derivative in ("falff", "rois_cc200"):
            for a, b in (("filt_global", "nofilt_global"), ("filt_noglobal", "nofilt_noglobal"),
                         ("nofilt_global", "nofilt_noglobal"), ("filt_global", "filt_noglobal")):
                left, right = by_key.get((sid, a, derivative)), by_key.get((sid, b, derivative))
                if left and right and left["status"] in SUCCESS and right["status"] in SUCCESS:
                    comparisons.append({"SUB_ID": sid, "derivative": derivative, "strategy_a": a,
                                        "strategy_b": b, "same_sha256": left["sha256"] == right["sha256"]})
    return pd.DataFrame(comparisons, columns=["SUB_ID", "derivative", "strategy_a", "strategy_b", "same_sha256"])


def run_acquisition(root, subject_ids, retries=3, download_scan_headers=True, create_zip=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata"
    metadata.mkdir(exist_ok=True)
    write_json(metadata / "run_config.json", {
        "started_utc": utc_now(), "subject_ids": list(subject_ids), "pipeline": "cpac",
        "strategies": list(STRATEGIES), "derivatives": ["falff", "rois_cc200", "func_mask (nofilt only)"],
        "selection": "explicit legacy IDs; no scientific exclusions at acquisition",
        "download_scan_headers": download_scan_headers, "create_zip": create_zip,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    })
    pheno_result = fetch_file(PHENO_URL, metadata / "Phenotypic_V1_0b_preprocessed1.csv", validate_pheno, retries)
    write_json(metadata / "phenotype_download.json", pheno_result)
    if pheno_result["status"] not in SUCCESS:
        raise RuntimeError(f"Cannot read required phenotype metadata: {pheno_result['error']}")
    pheno = pd.read_csv(pheno_result["path"])
    selected = select_subjects(pheno, subject_ids)
    selected.to_csv(metadata / "participants_requested.csv", index=False)
    flags = [c for c in selected if c.startswith("flag_")]
    print(f"Requested {len(selected)} subjects; metadata found for {selected.metadata_found.sum()}")
    print("Eligibility flags (reported, not excluded):", selected[flags].sum().to_dict())
    clean = selected.replace({-9999: np.nan, "-9999": np.nan, "-9999.0": np.nan})
    clean.to_csv(metadata / "participants_missing_normalized.csv", index=False)
    pd.DataFrame({"column": pheno.columns, "dtype_as_loaded": [str(pheno[c].dtype) for c in pheno],
                  "nonmissing_requested": [int(clean[c].notna().sum()) for c in pheno]}).to_csv(
                      metadata / "phenotype_columns.csv", index=False)
    selected.groupby(["SITE_ID", "DX_GROUP"], dropna=False).size().rename("n_requested").to_csv(
        metadata / "site_by_diagnosis_requested.csv")

    resources = []
    for name, url in SOURCE_URLS.items():
        result = fetch_file(url, root / "provenance" / name, retries=retries)
        resources.append({"resource": name, **result})
        print("Provenance:", name, result["status"])
    atlas = fetch_file(ATLAS_URL, root / "atlases/CC200.nii.gz", validate_nifti, retries)
    resources.append({"resource": "CC200.nii.gz", **atlas})
    yeo = fetch_file(YEO_URL, root / "atlases/Yeo_JNeurophysiol11_MNI152.zip", validate_yeo_archive, retries)
    resources.append({"resource": "Yeo_JNeurophysiol11_MNI152.zip", **yeo})
    write_json(metadata / "resource_manifest.json", resources)

    plan = build_plan(selected, root)
    pd.DataFrame(plan).to_csv(metadata / "download_plan.csv", index=False)
    records = []
    export_manifest(root, records)
    for i, item in enumerate(plan, 1):
        row = dict(item)
        if item["status"] == "pending":
            validator = validate_roi if item["derivative"] == "rois_cc200" else validate_nifti
            row.update(fetch_file(item["url"], root / item["relative_path"], validator, retries))
        records.append(row)
        if i % 25 == 0 or i == len(plan):
            export_manifest(root, records)
            count = sum(r["status"] in SUCCESS for r in records)
            print(f"Derivatives {i}/{len(plan)}; valid={count}; failed/missing={i-count}")

    scan_headers = {}
    if download_scan_headers:
        for i, subject in enumerate(selected.to_dict("records"), 1):
            if subject["downloadable_file_id"]:
                sid = int(subject["SUB_ID"])
                scan_headers[sid] = fetch_scan_header(
                    subject["FILE_ID"], metadata / "scan_headers" / f"{sid}_nifti_header.bin", retries)
                write_json(metadata / "scan_headers.json", scan_headers)
            if i % 25 == 0 or i == len(selected):
                print(f"Scan headers {i}/{len(selected)} (header only, not 4D download)")
    write_json(metadata / "scan_headers.json", scan_headers)
    completeness = subject_completeness(selected, records, scan_headers)
    completeness.to_csv(metadata / "subject_completeness.csv", index=False)
    compare_strategies(records).to_csv(metadata / "strategy_comparisons.csv", index=False)
    failures = [{"SUB_ID": r["SUB_ID"], "strategy": r["strategy"], "derivative": r["derivative"],
                 "status": r["status"], "url": r["url"], "error": r.get("error", "")} for r in records if r["status"] not in SUCCESS]
    pd.DataFrame(failures, columns=["SUB_ID", "strategy", "derivative", "status", "url", "error"]).to_csv(
        metadata / "download_failures.csv", index=False)
    roi_rows = []
    for r in records:
        if r["derivative"] == "rois_cc200" and r["status"] in SUCCESS:
            roi_rows.append({"SUB_ID": r["SUB_ID"], "strategy": r["strategy"], **r["details"]})
    pd.DataFrame(roi_rows).to_csv(metadata / "roi_timeseries_validation.csv", index=False)
    reference_ready = all(r["status"] in SUCCESS for r in resources if r["resource"] in (
        "CC200.nii.gz", "Yeo_JNeurophysiol11_MNI152.zip", "pipeline_config_abide_rerun.yml", "abide_rois.txt"))
    ready = bool(completeness.all_requested_files_complete.all()) and reference_ready and completeness.TR_seconds.notna().all()
    write_json(metadata / "completion_summary.json", {
        "finished_utc": utc_now(), "requested_subjects": len(selected), "requested_derivative_files": len(plan),
        "valid_derivative_files": sum(r["status"] in SUCCESS for r in records),
        "subjects_all_four_strategies": int(completeness.all_four_strategies_complete.sum()),
        "subjects_recommended_pair": int(completeness.recommended_pair_complete.sum()),
        "subjects_with_TR": int(completeness.TR_seconds.notna().sum()),
        "required_reference_resources_complete": reference_ready,
        "acquisition_complete": bool(ready), "scientific_eligibility_confirmed": False,
    })
    (root / "README.md").write_text(dataset_readme())
    # Create the optional archive only after all metadata and documentation exist.
    if create_zip:
        print("Creating archive (requires additional disk space)...")
        shutil.make_archive(str(root) + "_bundle", "zip", root)
    print("Output:", root)
    print("ACQUISITION COMPLETE" if ready else "INCOMPLETE: inspect metadata/completion_summary.json and failure manifests")
    print("Acquisition completeness does not imply scientific eligibility or verified fALFF processing.")
    return completeness


def dataset_readme():
    return """# ABIDE I PCP derivative acquisition for revision

These are downloaded, precomputed C-PAC derivatives from ABIDE I / PCP.
This notebook does not calculate time series, fALFF, FC, or harmonized features.
No participants are dropped for diagnosis, age, IQ, motion, site size, or energy.
The legacy ID list is preserved; audit flags are not exclusion decisions.

## Files and recommended starting pair

- `cpac/filt_global/rois_cc200`: band-pass-filtered, GSR ROI time series for FC.
- `cpac/filt_noglobal/rois_cc200`: filtered, no-GSR ROI time series for FC.
- `cpac/nofilt_global/falff` and `cpac/nofilt_noglobal/falff`: fALFF derivatives
  acquired from the corresponding unfiltered strategy paths for provenance review.
- Both fALFF and ROI time series are also retained for the other strategy paths,
  allowing comparisons. Never infer actual fALFF computation order from the folder
  name alone. Inspect upstream configuration and, where needed, historical code.
- `cpac/nofilt_*/func_mask`: masks for regional coverage/QC.
- `atlases/CC200.nii.gz`: upstream ROI atlas; match its labels/grid to time-series
  headers and fALFF maps during reanalysis. Do not assume exactly 200 usable ROIs.
- `atlases/Yeo_JNeurophysiol11_MNI152.zip`: upstream Yeo atlas and label resources;
  choose the cortical mask variant, resample, and compute ROI overlap explicitly
  during reanalysis. No CC200-to-network mapping is silently imposed here.

Preserve original scan lengths. ROI validation records volumes, regions, header
labels, and constant columns; no filtering, censoring, or truncation is performed.
TR is obtained from a small header-only request to a 4D nofilt_noglobal derivative,
not from a 3D fALFF image. Cross-check the header volume count against ROI lengths.
No full 4D scan is retained. Missing TR is explicit, never filled with a site guess.

## Metadata and provenance

`metadata/Phenotypic_V1_0b_preprocessed1.csv` preserves the source bytes and all
clinical, diagnostic, demographic, and motion/QC columns. Requested-subject tables
retain raw missing codes; a separately named table normalizes -9999 to missing.
Do not treat diagnostic/clinical missing values as zero or require complete clinical
scores for all imaging analyses. Handedness fields retain their source encoding.

`download_manifest.csv/json` gives subject, strategy, derivative, exact URL, local
path, validation, size, SHA-256, and HTTP metadata. Receipts permit validated reruns.
`subject_completeness.csv`, `download_failures.csv`, and `completion_summary.json`
distinguish download failure from usable pairs. Incomplete runs are not successes.
`strategy_comparisons.csv` reports exact file identity; identity alone cannot prove
the temporal processing used to calculate the upstream derivative.
`provenance/` snapshots upstream instructions/configuration with source URLs and
hashes in `metadata/resource_manifest.json`; these are retrieved snapshots, not a
claim that a particular historical pipeline version has been established.

## Issues to resolve before analysis

The published PCP configuration specifies fALFF cutoffs 0.01–0.1 Hz, whereas the
current manuscript says 0.01–0.08 Hz. Document the actual upstream computation,
including its input branch and sampled frequency denominator. Obtain upstream code
or additional data if the existing derivative cannot answer the reviewers' concern.
The original notebook listed 182 subjects, not the manuscript's initial 200; source
phenotypes must verify the claimed adult, male, age, IQ, and handedness criteria.
The phenotype's `func_perc_fd` field uses a 0.2-mm threshold per the PCP QA page,
not the manuscript's 0.5-mm threshold. Per-volume FD is not supplied here.
This package therefore cannot establish the >20%-of-volumes-at-0.5-mm QC rule.
It also cannot establish neurological-comorbidity exclusions from missing fields.

Use the source dataset's terms, acknowledgments, and citation requirements when
publishing a derived Kaggle dataset. Include this entire output directory, including
metadata, provenance, atlas, and receipts. ZIP creation is optional and off by default.
"""
