#!/bin/bash
# GEMM 目標達成ゲート（#1037）Metal 向け薄い wrapper（イシュー #1147）。
# 本体ロジックは device 汎用化された `run_gemm_gate.sh`（cuda 向け
# `run_gemm_gate_cuda.sh` と共通）。
set -u
exec bash "$(dirname "$0")/run_gemm_gate.sh" metal "$@"
