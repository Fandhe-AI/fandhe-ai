#!/usr/bin/env bash
# イシュー #4149: PyTorch の CPU／CUDA が GEMM をどの入口・どのライブラリへ流すかの確認（計測外・時間は測らない）。<cuda-node> 上で実行する。
# 使い方: PY=<torch 2.14.0 入り python の絶対パス> bash run_probe.sh <出力ファイルの絶対パス>
# 役割: probe_shim_4149.c を mktemp 領域でビルドし、probe_driver_4149.py を shim 有り／無しで起動して JSON 行を集める。
#   train 1 step・infer 1 回（CPU）、infer 1 回（CUDA）、import のみ（基準）。shim 有り／無しの checksum 一致も同じ出力で確認できる。
# 派生元: docs/perf/logs/cpu-gemm-resident-reuse-3646/gb10-gcpuinf/blas_probe/run_probe.sh（#3646）。
# 計時腕とは別プロセス・別時刻（全 round の後に 1 回）で動かし、計時値の生成には関与しない。
# 外部入力はシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。一時領域は自分の mktemp -d だけを掃除する。
set -euo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2" >&2; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む" >&2; exit 2;; esac
}
PY="${PY:?torch 入り python の絶対パス}"
OUT="${1:?出力ファイルの絶対パス}"
validate_abs PY "${PY}"; validate_abs OUT "${OUT}"
[ ! -e "${OUT}" ] || { echo "${OUT} が既にある（上書き禁止）" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
gcc -O1 -shared -fPIC -o "${work}/libprobe.so" "${HERE}/probe_shim_4149.c" -ldl
DRV="${HERE}/probe_driver_4149.py"
{
  echo "# gcc: $(gcc --version | head -n 1)"
  echo "# --- shim なし（checksum の基準）"
  for d in cpu cuda; do
    for op in train infer; do
      [ "${d}" = "cuda" ] && [ "${op}" = "train" ] && continue
      "${PY}" "${DRV}" --device "${d}" --op "${op}"
    done
  done
  echo "# --- shim あり（per_step_calls が GEMM 入口の 1 step あたり回数。next_lib が転送先）"
  for d in cpu cuda; do
    for op in train infer; do
      [ "${d}" = "cuda" ] && [ "${op}" = "train" ] && continue
      LD_PRELOAD="${work}/libprobe.so" "${PY}" "${DRV}" --device "${d}" --op "${op}" --shim "${work}/libprobe.so"
    done
  done
  echo "# --- import torch のみ（基準。mm 以外の GEMM 入口呼び出しの有無）"
  LD_PRELOAD="${work}/libprobe.so" "${PY}" "${DRV}" --device cpu --op import-only --shim "${work}/libprobe.so"
} 2>&1 | sed -e "s#${HOME}#<home>#g" > "${OUT}"
echo "wrote ${OUT}"
