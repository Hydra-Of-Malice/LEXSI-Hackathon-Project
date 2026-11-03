#!/usr/bin/env bash
# Install the Lexsi kits from GitHub (pinned to the commits we read and tested against) plus langsafe's GPU extras.
# Colab/Kaggle: run in a cell with `!bash scripts/install_kits.sh`.
set -euo pipefail

pip uninstall -y torchao 2>/dev/null || true          # SafeTune README: torchao on Colab breaks imports
pip install -q -e ".[gpu,dev]"
pip install -q "safetune @ git+https://github.com/Lexsi-Labs/SafeTune@3901a9f"
pip install -q "circuitkit @ git+https://github.com/Lexsi-Labs/CircuitKIT@e71d5e1"
pip install -q "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da"
pip install -q "curatorkit @ git+https://github.com/Lexsi-Labs/CuratorKIT@dd9e835"
python - <<'PY'
import circuitkit, safetune, auditkit, curatorkit, transformers, torch
print("circuitkit", circuitkit.__version__, "| safetune", safetune.__version__, "| auditkit", auditkit.__version__,
      "| curatorkit", curatorkit.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
PY
