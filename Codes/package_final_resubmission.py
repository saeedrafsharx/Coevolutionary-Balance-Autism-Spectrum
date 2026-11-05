"""Validate final revision documents and assemble only the journal upload files."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,re,csv,io
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision/resubmission'
sha=lambda b:hashlib.sha256(b).hexdigest()
# Final PDFs, citations, references, layout and numbering.
logs={}
for name in ['manuscript_illustrated','manuscript_submission','supporting_information','response_to_reviewers','manuscript_marked']:
 s=(R/(name+'.log')).read_text();bad=[l for l in s.splitlines() if 'undefined' in l.lower() or 'Overfull' in l or l.startswith('!')]
 assert not bad,(name,bad)
 pages=int(re.search(r'Output written on .*?\((\d+) pages',s,re.S).group(1));logs[name]={'pages':pages,'unresolved_or_overflow_errors':bad}
s=(R/'manuscript.tex').read_text();assert re.findall(r'\\paperfigure\{Fig(\d+)\}',s)==list(map(str,range(1,12)))
assert len(re.findall(r'\\input\{tables/',s))==9
marked=(R/'manuscript_marked.tex').read_text()
assert len(re.findall(r'\\label\{deletedoriginal:',marked))==12
assert 'Short title:' in s
# The local analysis data are unchanged.
f=ROOT/'results/abide_stage2_repaired/features/all_designs_raw_untrimmed.csv'
feature_sha=sha(f.read_bytes());assert feature_sha=='25178d5b3f5b87ad10c4b2f60f0574fb5fe51b26d042ddac7255f9030894848b'
# Check exact page/line references against the clean submission aux.
aux=(R/'manuscript_submission.aux').read_text();valid={(int(p),int(l)) for l,p in re.findall(r'\\newlabel\{lin:[^}]+\}\{\{(\d+)\}\{(\d+)\}',aux)}
response=(R/'response_to_reviewers.md').read_text()
for p,l in re.findall(r'p\. (\d+), line (\d+)',response):assert (int(p),int(l)) in valid
assert len(re.findall(r'^\*\*Comment:\*\*',response,re.M))==14
assert not re.search('author review version|submission gate|critical provenance item open',response,re.I)
# Complete supporting data archive with per-file hashes.
with ZipFile(R/'S2_Data.zip') as z:
 assert z.testzip() is None
 rows=list(csv.DictReader(io.StringIO(z.read('INDEX_SHA256.csv').decode())))
 for row in rows:
  b=z.read(row['file']);assert len(b)==int(row['bytes']) and sha(b)==row['sha256']
# Scientific code archive excludes private decision correspondence and response.
code={}
excluded={'assemble_resubmission_manuscript.py','prepare_resubmission_support.py','prepare_resubmission_response.py','finalize_resubmission_text.py','build_resubmission_marked.py','render_resubmission_response.py','package_final_resubmission.py'}
for p in (ROOT/'Codes').glob('*'):
 if p.is_file() and p.suffix in {'.py','.sh','.ipynb','.md','.txt'} and p.name not in excluded:code['Codes/'+p.name]=p.read_bytes()
for p in (ROOT/'revision/analysis_plan').glob('*'):
 if p.is_file():code['revision/analysis_plan/'+p.name]=p.read_bytes()
code['README.md']=b'''# S3 Code: exact revision analysis release

The executable analysis, audit, plot and reporting source is in Codes/.
Frozen participant/ROI definitions and revision-stage analysis specifications are
in revision/analysis_plan/. Historical acquisition and analysis notebooks are
retained for audit; the revised feature/stats scripts define current results.

Sequence: stage-1 acquisition/coverage QC -> stage-1b review and frozen design ->
abide_stage2_run.py (or corresponding notebook) -> documented statistics repair
when needed -> complete_original_classifiers.py -> run_ml_feature_selection.py
-> report_ml_feature_selection.py. Read each script's top-level configuration
and command arguments before execution. Source imaging is not included here.

Input Kaggle datasets: abide-full-derivatives-dataset/abide_revision and
abide-metadata-patch/abide_metadata_patch_local_v2. ABIDE/PCP retains source images.
PCP/C-PAC computed the downloaded voxelwise fALFF maps. Local code extracts ROI
means and computes FC, graph features and statistics; it does not recompute the
voxelwise fALFF denominator. Exact historical denominator/filter-order details
remain unverified and are disclosed in the main Methods, not asserted as known.

The five-model classifier analysis is the baseline. Expanded feature selection
is a separate exploratory follow-up specified after baseline results were seen.
It must not overwrite the baseline or be treated as independent validation.
S2 Data supplies complete results, selection names, inner candidates, predictions,
covariates and provenance. Directory paths in scripts refer to the full project
layout; reconstruct those paths from S2 where rebuilding reports locally.
The frozen plans are historical records and do not supersede the final Methods.

Software: Python 3; NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4,
statsmodels 0.15.0, neuroCombat 0.2.12, scikit-learn 1.9.1,
XGBoost CPU 3.4.1, matplotlib 3.6.3. S2 contains audit version records.
Publication plots use Arial. build_resubmission_exhibits.py regenerates figures
and tables from the verified results. Manuscript LaTeX is supplied separately
as manuscript_source.zip with the submission, not mixed with private review letters.
'''
code['INDEX_SHA256.csv']=('file,bytes,sha256\n'+''.join(f'{n},{len(b)},{sha(b)}\n' for n,b in sorted(code.items()))).encode()
with ZipFile(R/'S3_Code.zip','w',ZIP_DEFLATED) as z:
 for n,b in sorted(code.items()):z.writestr(n,b)
# Self-contained clean LaTeX source with a local journal bibliography style.
source={'main.tex':('\\def\\SubmissionVersion{1}\n'+s.replace('\\bibliographystyle{../../plos2025}','\\bibliographystyle{plos2025}')).encode(), 'Bibliography.bib':(R/'Bibliography.bib').read_bytes(),'plos2025.bst':(ROOT/'plos2025.bst').read_bytes()}
for p in (R/'tables').glob('*.tex'):source['tables/'+p.name]=p.read_bytes()
source['README.txt']=b'Compile main.tex using pdflatex/bibtex or latexmk. Main figures are supplied separately as Fig1.tif through Fig11.tif. The clean manuscript includes inline captions and tables, without embedded main images.\n'
with ZipFile(R/'manuscript_source.zip','w',ZIP_DEFLATED) as z:
 for n,b in sorted(source.items()):z.writestr(n,b)
figs=[]
for p in sorted((R/'upload_figures').glob('*.tif')):
 with Image.open(p) as im:
  assert im.mode=='RGB' and im.info['compression']=='tiff_lzw'
  assert im.width/600<=7.5 and im.height/600<=8.75 and p.stat().st_size<10_000_000
  figs.append({'file':p.name,'sha256':sha(p.read_bytes())})
assert len(figs)==11
expected={'manuscript.tex':'1dff55b40ae5a9638e57b5bee68aed787e505c336e9790df8bc4c159ce8952c6','Bibliography.bib':'9ca106a98f44e7c85ee363c1634b7385c4307041316c983c42698c6c85f0b70e','AGENTS.md':'b28ace7504fa9dc009123bdcfb2ca8b784139c6ba125b32ac46c7a672ff4dfe0'}
for n,h in expected.items():assert sha((ROOT/n).read_bytes())==h
report={'documents':logs,'main_figures':11,'supporting_figures':13,'main_tables':9,'indexed_data_payloads':len(rows),'code_payloads':len(code)-1,'feature_table_sha256':feature_sha,'original_file_hashes':expected,'build_validation_passed':True,'submission_files_complete':True,'provider_falff_implementation_independently_verified':False,'provider_documentation_limitation_disclosed':True,'reviewer_upstream_question_claimed_fully_verified':False,'visual_review':'Final Methods page and response opening inspected; earlier complete-page montage and representative figures reviewed.','uploaded_or_published':False,'ready_for_author_upload':True,'deleted_original_main_tables_in_marked_copy':12,'journal_requirements_checked':'2026-11-05'}
(R/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
files={p.name:p.read_bytes() for p in [R/n for n in ['manuscript_submission.pdf','manuscript_marked.pdf','response_to_reviewers.pdf','supporting_information.pdf','S2_Data.zip','S3_Code.zip','manuscript_source.zip']]}
for p in (R/'upload_figures').glob('*.tif'):files[p.name]=p.read_bytes()
upload_rows=[('manuscript_submission.pdf','Manuscript','Required'),('manuscript_marked.pdf','Revised Manuscript with Track Changes / Revised Article with Changes Highlighted','Required'),('response_to_reviewers.pdf','Response to Reviewers','Required'),('supporting_information.pdf','Supporting Information: S1 Appendix','Required'),('S2_Data.zip','Supporting Information: S2 Data','Required'),('S3_Code.zip','Supporting Information: S3 Code','Required')]+[(f'Fig{i}.tif','Figure','Required') for i in range(1,12)]+[('manuscript_source.zip','LaTeX source','Only if requested by the portal')]
buf=io.StringIO();writer=csv.writer(buf);writer.writerow(['filename','upload_category','status']);writer.writerows(upload_rows)
files['UPLOAD_ORDER.csv']=buf.getvalue().encode()
(R/'UPLOAD_ORDER.csv').write_bytes(files['UPLOAD_ORDER.csv'])
files['UPLOAD_README.txt']=b'''PONE-D-26-34219: COMPLETE REVISION UPLOAD SET (checked November 5, 2026)

1. Extract this ZIP on your computer. Upload the enclosed files individually;
   the package ZIP itself is a container, not the manuscript upload.
2. In Submissions Needing Revision, open PONE-D-26-34219 and revise the submission.
3. Replace the manuscript with manuscript_submission.pdf (clean article).
4. Upload manuscript_marked.pdf under the marked/track-changes category and
   response_to_reviewers.pdf under Response to Reviewers. The response's
   page/line locations refer to the clean manuscript.
5. Replace ALL old main figure entries with Fig1.tif through Fig11.tif.
6. Replace the old supporting files with supporting_information.pdf (S1 Appendix),
   S2_Data.zip (S2 Data), and S3_Code.zip (S3 Code). Do not retain stale duplicates.
7. The clean article is compiled LaTeX PDF; no Word conversion is needed.
   manuscript_source.zip is available if the portal requests source files.
8. Retain accurate existing portal declarations and author details. Build the
   portal's combined PDF and check that it includes the revised figures and
   links to the replacement supporting files before approving the submission.

UPLOAD_ORDER.csv lists each file's category. Files are prepared for author upload;
nothing has been submitted automatically. The marked copy retains all 12 deleted
original main tables in a clearly labeled deletion appendix; only the new tables
appear in the clean article. Do not upload the illustrated reading PDF as well.

The fALFF response attributes voxelwise computation to downloaded PCP/C-PAC
maps, documents the provider's 0.01-0.1 Hz band, and discloses unverified historical
denominator/filter-order details. The editor will assess this explanation; no
provider confirmation or independent verification is invented.

PLOS guidance: https://journals.plos.org/plosone/s/revising-your-manuscript
LaTeX/PDF format: https://journals.plos.org/plosone/s/submission-guidelines
'''
(R/'UPLOAD_README.txt').write_bytes(files['UPLOAD_README.txt'])
files['FILES_SHA256.csv']=('file,bytes,sha256\n'+''.join(f'{n},{len(b)},{sha(b)}\n' for n,b in sorted(files.items()))).encode()
with ZipFile(R/'PLOS_revision_upload.zip','w',ZIP_DEFLATED) as z:
 for n,b in sorted(files.items()):z.writestr(n,b)
print(json.dumps(report,indent=2));print('Upload archive:',R/'PLOS_revision_upload.zip')
