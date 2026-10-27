#!/usr/bin/env bash
# Run in a normal terminal: installs only the missing XGBoost CPU package and resumes.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
classifier_python="$project_dir/.venv_stage2/bin/python"
if [[ ! -x "$classifier_python" ]]; then
  echo 'Missing .venv_stage2; first run Codes/finish_stage2.sh in a normal terminal.' >&2
  exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/coev_matplotlib}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if ! "$classifier_python" -c 'from xgboost import XGBClassifier' 2>/dev/null; then
  "$classifier_python" -m pip install 'xgboost-cpu>=3,<4'
fi
"$classifier_python" "$project_dir/Codes/complete_original_classifiers.py" --require-xgboost
"$classifier_python" "$project_dir/Codes/update_classifier_materials.py"
