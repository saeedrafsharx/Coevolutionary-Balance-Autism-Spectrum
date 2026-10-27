"""Run metadata repair from a normal terminal; no notebook or Kaggle required."""
import argparse
import os
import shutil
from pathlib import Path

from abide_metadata_repair import repair_metadata


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=repo / 'Data',
                        help='Dataset directory containing metadata/, or the flat Data directory supplied here')
    parser.add_argument('--output-root', type=Path, default=repo / 'Data/abide_metadata_patch_local')
    parser.add_argument('--cache-root', type=Path, default=repo / 'Data/abide_metadata_patch')
    parser.add_argument('--offline', action='store_true', help='Validate cached files without network requests')
    args = parser.parse_args()
    if args.cache_root.resolve() == args.output_root.resolve():
        parser.error('Use a distinct output directory to preserve the original repair evidence')
    # Reuse the already acquired index and method pages, including URL/hash receipts.
    for directory in ('provenance', 'sidecars', 'atlases'):
        source = args.cache_root / directory
        if source.is_dir():
            shutil.copytree(source, args.output_root / directory, dirs_exist_ok=True)
    offline = args.offline or os.environ.get('CODEX_SANDBOX_NETWORK_DISABLED') == '1'
    print('Mode:', 'offline cache validation; downloads remain pending' if offline else 'online metadata repair', flush=True)
    repair_metadata(args.dataset_root, args.output_root, offline=offline)


if __name__ == '__main__':
    main()
