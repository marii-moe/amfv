#!/usr/bin/env bash
#SBATCH --job-name=amfv-test-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --export=ALL
#SBATCH --output=test_vllm_%j.log
#SBATCH --error=test_vllm_%j.log

set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
echo "==> Testing container: ${VLLM_IMAGE}"

srun --ntasks=1 \
     --container-image="${VLLM_IMAGE}" \
     python3 -c "
import torch
import vllm
print('torch version:', torch.__version__)
print('CUDA version (torch):', torch.version.cuda)
print('vllm version:', vllm.__version__)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
