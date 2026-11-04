#!/usr/bin/env bash
# Fill the Hugging Face cache with the experiment's safetensors weights, however often the connection drops.
#   bash scripts/fetch_models.sh CohereLabs/tiny-aya-global CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-base
# Plain HTTP (not Xet) so a partial file resumes. Retries until each snapshot is complete.
set -u
export HF_HUB_DISABLE_XET=1
python="${LANGSAFE_PYTHON:-.venv/Scripts/python.exe}"
[ -x "$python" ] || python=.venv/bin/python
for model in "$@"; do
  until "$python" - "$model" <<'PY' 2>/dev/null
import sys
from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"], max_workers=2)
PY
  do
    sleep 30
  done
  echo "complete $model"
done
