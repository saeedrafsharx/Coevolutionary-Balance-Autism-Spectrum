#!/usr/bin/env bash
# Run from a normal local terminal. Installs missing statistics packages in a project venv.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage2_venv="$project_dir/.venv_stage2"
if [[ ! -x "$stage2_venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$stage2_venv"
fi
if ! "$stage2_venv/bin/python" -c 'import statsmodels.api, sklearn, neuroCombat' 2>/dev/null; then
  "$stage2_venv/bin/python" -m pip install 'statsmodels>=0.14,<1' 'scikit-learn>=1.3,<2' neuroCombat
fi
exec "$stage2_venv/bin/python" "$project_dir/Codes/repair_stage2_results.py" --complete "$@"
