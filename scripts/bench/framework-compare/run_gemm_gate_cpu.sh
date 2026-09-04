#!/bin/bash
# GEMM 目標達成ゲート（#1117）CPU 向け薄い wrapper（イシュー #1148）。
# 本体ロジックは device 汎用化された `run_gemm_gate.sh`（cuda 向け
# `run_gemm_gate_cuda.sh`・metal 向け `run_gemm_gate_metal.sh` と共通）。
set -u
exec bash "$(dirname "$0")/run_gemm_gate.sh" cpu "$@"
