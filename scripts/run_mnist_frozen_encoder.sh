#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${BIO_ATTENTION_VENV:-.venv-bio-attention}"
TORCH_INSTALL_CMD="${TORCH_INSTALL_CMD:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ -n "${TORCH_INSTALL_CMD}" ]]; then
  "${VENV_DIR}/bin/python" -m ${TORCH_INSTALL_CMD}
else
  "${VENV_DIR}/bin/python" -m pip install torch torchvision
fi
"${VENV_DIR}/bin/python" -m pip install "numpy>=1.26" scipy Pillow matplotlib

"${VENV_DIR}/bin/python" scripts/mnist_scenarios.py frozen-encoder \
  --pretrained-dir "${PRETRAINED_DIR:-./pretrained/mnist_v2}" \
  --data-path "${DATA_PATH:-./data}" \
  --output-root "${OUTPUT_ROOT:-./results}" \
  --name "${RUN_NAME:-mnist_frozen_encoder}" \
  --encoder-epochs "${ENCODER_EPOCHS:-96}" \
  ${DECODER_EPOCHS:+--decoder-epochs "$DECODER_EPOCHS"} \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"}
