#!/usr/bin/env bash
#SBATCH --job-name=amfv-vllm-import
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=c
#SBATCH --account=training
#SBATCH --export=ALL
#SBATCH --output=test_vllm_import_%j.log
#SBATCH --error=test_vllm_import_%j.log

set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
echo "==> Importing/caching container image: ${VLLM_IMAGE}"

srun --ntasks=1 \
     --container-image="${VLLM_IMAGE}" \
     echo "Image cached OK"

echo "==> Done. Image is now cached on this node."
