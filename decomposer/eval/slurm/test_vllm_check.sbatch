#!/usr/bin/env bash
#SBATCH --job-name=amfv-vllm-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --export=ALL
#SBATCH --output=test_vllm_check_%j.log
#SBATCH --error=test_vllm_check_%j.log

set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
echo "==> Checking container: ${VLLM_IMAGE}"

srun --ntasks=1 \
     --container-image="${VLLM_IMAGE}" \
     python3 -c "
import torch, vllm
print('torch:     ', torch.__version__)
print('CUDA:      ', torch.version.cuda)
print('vllm:      ', vllm.__version__)
print('GPU avail: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:       ', torch.cuda.get_device_name(0))
"
