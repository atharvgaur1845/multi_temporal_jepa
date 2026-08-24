#!/bin/bash
# RUN ON THE LOGIN NODE. Builds the Python environment. ~5 minutes.
#   bash REO-2/slurm/00_env_setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "[setup] python: $(python3 -V)"
echo "[setup] if this is older than 3.10, run 'module avail python' and load a newer one first."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install torch FIRST and let pip pick the CUDA build matching the cluster driver.
# If the login node has no GPU (normal), pip cannot autodetect -- the default
# PyPI wheel bundles CUDA and works on any recent driver.
pip install torch
pip install -r requirements.txt

python -c "import torch, sys; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "[setup] NOTE: torch.cuda.is_available() is False on a login node. That is expected."
echo "[setup] Verify on a compute node:"
echo "        srun --partition=gpu_a100_8 --gres=gpu:1 --time=00:10:00 --pty \\"
echo "             bash -c 'source .venv/bin/activate && python -c \"import torch;print(torch.cuda.get_device_name(0))\"'"
echo "[setup] Then run the offline test suite (no GPU, no data needed):  pytest -q"
