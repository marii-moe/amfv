#!/usr/bin/env bash
# Source this file to start a vLLM server and export VLLM_BASE_URL.
# Usage: source _vllm_start.sh <model> <tensor_parallel> [port]
#
# Sets:
#   VLLM_BASE_URL  — OpenAI-compatible base URL for the started server
#   VLLM_PID       — PID of the background vllm process
#
# Registers an EXIT trap that kills vLLM when the sourcing script exits.

_VLLM_MODEL="${1:?model argument required}"
_VLLM_TP="${2:-4}"
_VLLM_PORT="${3:-8000}"

echo "[vllm] Starting ${_VLLM_MODEL} (tp=${_VLLM_TP}, port=${_VLLM_PORT})…"

vllm serve "${_VLLM_MODEL}" \
    --tensor-parallel-size "${_VLLM_TP}" \
    --port "${_VLLM_PORT}" \
    --disable-log-requests \
    &
VLLM_PID=$!

trap 'echo "[vllm] Stopping vLLM (pid ${VLLM_PID})…"; kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null' EXIT

export VLLM_BASE_URL="http://localhost:${_VLLM_PORT}/v1"

echo "[vllm] Waiting for server to be ready…"
for i in $(seq 1 180); do
    if curl -sf "${VLLM_BASE_URL}/health" > /dev/null 2>&1; then
        echo "[vllm] Ready after ${i}s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[vllm] Process died unexpectedly" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -sf "${VLLM_BASE_URL}/health" > /dev/null 2>&1; then
    echo "[vllm] Server did not become healthy within 180s" >&2
    exit 1
fi
