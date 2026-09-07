#!/usr/bin/env bash
# イシュー #1305: 専有環境での RAYON_NUM_THREADS スイープ駆動スクリプト。
#
# `scripts/bench/oss-gemm-compare`（独立 workspace。self_gemm_blis_parallel・
# matrixmultiply・gemm crate を同一プロトコルで計測するハーネス）に対し、
# 各スレッド数ごとに独立プロセスを 5 回起動し、run ヘッダ・終了コードを記録する。
# #1148 §8.2 の非単調性（3 回計測・共有負荷下）の再現有無を専有環境で
# 5 回計測中央値で確定するための計測本体（docs/perf/cpu-gemm-candle-gate-remeasurement.md §16）。
#
# ホスト名はこのスクリプトにハードコードしない（.claude/rules/security.md）。
# 呼び出し側（DGX では ssh 経由、M4 Max ではローカル）が環境変数で制御する。
#
# 環境変数:
#   BIN       oss-gemm-compare リリースバイナリの絶対パス（必須）
#   SIZES     "--sizes" に渡す値（既定 1024,2048,4096）
#   THREADS   スペース区切りのスレッド数列（既定は呼び出し側で指定必須）
#   RUNS      各スレッド数の独立プロセス起動回数（既定 5）
#   OUT       出力ログファイル（必須）
#   TASKSET   taskset -c 相当の cpu list を付けたい場合のみ設定（補助軸 A 用）
#
# 既知挙動: K>=1024 の形状では OSS 実装間の丸め差により output_match=false
# となり非 0 終了することがある（fail-closed 仕様。ハーネス側の設計）。
# 本スクリプトはその終了コードでスイープ全体を止めず run ヘッダへ記録して続行する。
set -uo pipefail

BIN="${BIN:?BIN (oss-gemm-compare バイナリパス) を指定すること}"
SIZES="${SIZES:-1024,2048,4096}"
THREADS="${THREADS:?THREADS (スペース区切りのスレッド数列) を指定すること}"
RUNS="${RUNS:-5}"
OUT="${OUT:?OUT (出力ログパス) を指定すること}"
TASKSET_MASK="${TASKSET:-}"

: > "$OUT"

for T in $THREADS; do
  for i in $(seq 1 "$RUNS"); do
    echo "== RAYON_NUM_THREADS=$T run=$i taskset=${TASKSET_MASK:-none} ==" >> "$OUT"
    if [ -n "$TASKSET_MASK" ]; then
      RAYON_NUM_THREADS="$T" taskset -c "$TASKSET_MASK" "$BIN" --sizes "$SIZES" >> "$OUT" 2>>"$OUT"
    else
      RAYON_NUM_THREADS="$T" "$BIN" --sizes "$SIZES" >> "$OUT" 2>>"$OUT"
    fi
    rc=$?
    echo "== RAYON_NUM_THREADS=$T run=$i rc=$rc ==" >> "$OUT"
  done
done

echo "done." >> "$OUT"
