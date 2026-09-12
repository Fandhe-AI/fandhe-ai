#!/usr/bin/env bash
# DGX: Python 参照フレームワーク用 venv（PyTorch CPU/CUDA・TensorFlow CPU）
set -u
# pipefail: `| tail` で pip／cargo の失敗が隠れないようにする（PR #1654 レビュー後に追加。
# 記録済み実行時は `set -u` のみだったが、結果ログで成功を確認済み。README「レビュー後の修正」節）
set -o pipefail
cd "${HOME}/work"
echo "start $(date -u +%FT%TZ)"
python3 -m venv .venv-bench 2>&1 | tail -1
./.venv-bench/bin/pip install -q --upgrade pip 2>&1 | tail -1
./.venv-bench/bin/pip install -q numpy scipy 2>&1 | tail -1
# CUDA 13 系 wheel（aarch64）を優先し、無ければ既定 index（CPU/cu12x）へフォールバック
./.venv-bench/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2 || ./.venv-bench/bin/pip install -q torch 2>&1 | tail -2
./.venv-bench/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.version.cuda)" 2>&1 | tail -1
./.venv-bench/bin/pip install -q tensorflow 2>&1 | tail -2
./.venv-bench/bin/python -c "import tensorflow as tf;print('tf',tf.__version__, len(tf.config.list_physical_devices('GPU')))" 2>&1 | tail -1
echo "venv-done. $(date -u +%FT%TZ)"
