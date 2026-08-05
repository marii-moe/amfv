#!/usr/bin/env bash
# Source this file to start a vLLM server and export VLLM_BASE_URL.
#
# Usage (enroot/pyxis mode — recommended):
#   source _vllm_start.sh <model> <tensor_parallel> <container_image> [port]
#
# Usage (direct binary mode — fallback):
#   source _vllm_start.sh <model> <tensor_parallel> <vllm_bin_path> [port]
#
# The third argument is auto-detected: if it ends in .sqsh or starts with
# docker:// it is treated as a container image; otherwise as a binary path.
#
# Sets:
#   VLLM_BASE_URL  — OpenAI-compatible base URL for the started server
#   VLLM_PID       — PID of the background vllm process

_VLLM_MODEL="${1:?model argument required}"
_VLLM_TP="${2:-4}"
_VLLM_ARG3="${3:-vllm}"
_VLLM_PORT="${4:-8000}"
_VLLM_TIMEOUT="${VLLM_TIMEOUT:-600}"

# vllm logs go to a separate file next to the SLURM job log so they're always visible.
_VLLM_LOG="${SLURM_JOB_ID:+${SLURM_LOG_DIR:-/tmp}/vllm_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.log}"
_VLLM_LOG="${_VLLM_LOG:-/tmp/vllm_$$.log}"

echo "[vllm] Starting ${_VLLM_MODEL} (tp=${_VLLM_TP}, port=${_VLLM_PORT})…"
echo "[vllm] Server log: ${_VLLM_LOG}"

if [[ "${_VLLM_ARG3}" == *.sqsh || "${_VLLM_ARG3}" == docker://* ]]; then
    # enroot/pyxis mode: run vllm serve inside a container via srun.
    # Point enroot cache at a shared NFS path so nodes share the .sqsh.
    _ENROOT_CACHE="${ENROOT_CACHE:-${HOME}/.enroot/cache}"
    mkdir -p "${_ENROOT_CACHE}"
    export LABLESS_ENROOT_CACHE_PATH="${_ENROOT_CACHE}"

    echo "[vllm] Using container: ${_VLLM_ARG3}"
    srun --ntasks=1 \
         --container-image="${_VLLM_ARG3}" \
         --container-mounts="${HF_HOME}:${HF_HOME}" \
         --container-env="HF_HOME,HUGGING_FACE_HUB_TOKEN" \
         vllm serve "${_VLLM_MODEL}" \
             --tensor-parallel-size "${_VLLM_TP}" \
             --port "${_VLLM_PORT}" \
         >> "${_VLLM_LOG}" 2>&1 &
else
    # direct binary mode
    echo "[vllm] Using binary: ${_VLLM_ARG3}"
    "${_VLLM_ARG3}" serve "${_VLLM_MODEL}" \
        --tensor-parallel-size "${_VLLM_TP}" \
        --port "${_VLLM_PORT}" \
        >> "${_VLLM_LOG}" 2>&1 &
fi
VLLM_PID=$!

trap 'echo "[vllm] Stopping vLLM (pid ${VLLM_PID})…"; kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null' EXIT

export VLLM_BASE_URL="http://localhost:${_VLLM_PORT}/v1"

echo "[vllm] Waiting for server to be ready (timeout ${_VLLM_TIMEOUT}s)…"
for i in $(seq 1 "${_VLLM_TIMEOUT}"); do
    if curl -sf "${VLLM_BASE_URL}/health" > /dev/null 2>&1; then
        echo "[vllm] Ready after ${i}s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[vllm] Process died unexpectedly — last lines of ${_VLLM_LOG}:" >&2
        tail -20 "${_VLLM_LOG}" >&2
        exit 1
    fi
    if (( i % 30 == 0 )); then
        echo "[vllm] Still waiting… (${i}s)"
    fi
    sleep 1
done

if ! curl -sf "${VLLM_BASE_URL}/health" > /dev/null 2>&1; then
    echo "[vllm] Server did not become healthy within ${_VLLM_TIMEOUT}s — last lines of ${_VLLM_LOG}:" >&2
    tail -20 "${_VLLM_LOG}" >&2
    exit 1
fi
