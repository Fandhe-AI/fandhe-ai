#!/bin/bash
# GEMM 目標達成ゲート（#1031）CUDA 向け薄い wrapper（イシュー #1142）。
# 本体ロジックは device 汎用化された `run_gemm_gate.sh`（イシュー #1147）
# へ移設済み。既存呼び出し（`bash run_gemm_gate_cuda.sh <label>`）との CLI
# 互換のため本ファイルは残す。
set -u
exec bash "$(dirname "$0")/run_gemm_gate.sh" cuda "$@"
