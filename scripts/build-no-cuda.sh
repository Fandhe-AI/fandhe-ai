#!/usr/bin/env bash
# CUDA toolkit 非搭載環境でのビルド・実行成立検証（make build-no-cuda。TASK-1.7d・#35／
# TASK-2.3・#56）。scripts/check-cuda-toolkit-absent.sh（ci.yml の build-no-cuda-toolkit
# ジョブと共用する単一ソース）で非搭載を検証したうえで cargo build・cargo test を実行する。
# CUDA 実機（DGX Spark GB10 等）で toolkit 搭載環境から `make ci` を実行しても壊れないよう
# probe でスキップ分岐する（CI ジョブ側は assert で fail-closed のまま。非搭載が確定した
# 環境での検証は `make docker-ci` を使う）。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "${DIR}/lib.sh"

reject_args "$(basename "$0")" "$@"

bash scripts/check-cuda-toolkit-absent.sh self-test
# ROCm 非搭載の検証（イシュー #3154）も同じ流れで行う。
bash scripts/check-rocm-absent.sh self-test
if bash scripts/check-cuda-toolkit-absent.sh probe && bash scripts/check-rocm-absent.sh probe; then
  bash scripts/check-cuda-toolkit-absent.sh assert
  bash scripts/check-rocm-absent.sh assert
  cargo build --workspace --locked
  cargo test --workspace --locked
else
  echo "skip: CUDA toolkit または ROCm 搭載環境のため非搭載検証をスキップ（make docker-ci で検証可能）"
fi
