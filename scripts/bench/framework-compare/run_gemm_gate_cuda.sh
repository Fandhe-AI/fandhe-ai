#!/bin/bash
# GEMM 目標達成ゲート（#1031）の 5 回計測スイープ（イシュー #1142）。
#
# `run_all_cuda.sh` は GEMM を N ごとに 1 回しか起動しないため、そのままでは
# coding-rust.md「ベンチは 5 回計測の中央値」を満たせない。本スクリプトは
# N=1024/2048/4096 それぞれについて `bench-fandhe gemm cuda <N> reuse` と
# `bench-candle gemm cuda <N> fresh`（candle は reuse 非対応。README「計測
# プロトコル」節）を交互に 5 回ずつ起動し、`compare_gemm_gate.py` が run 間
# 中央値を算出できる形で JSONL に記録する（fandhe/candle を run 内で交互に
# 実行し、熱・クロック状態の偏りを両フレームワークで揃える）。
#
# 呼び出し例（GB10 実機。ユーザー承認・別セッション）:
#   bash run_gemm_gate_cuda.sh 0.6.0
#   GEMM_GATE_SKIP_BUILD=1 bash run_gemm_gate_cuda.sh head-<short sha>
#
# GEMM_GATE_SKIP_BUILD=1 でビルドを省略する（参考系列: #1164 結線後 HEAD を
# `cargo build … --config 'patch.crates-io.fandhe-ai.path="<facade 絶対パス>"'`
# で事前ビルドしてから本スクリプトを走らせる用途。README「GEMM ゲート
# 5 回計測（#1142）」節参照。この `--config patch` はコマンドライン引数のみで
# 与え、`[patch]` セクション・`.cargo/config.toml` はコミットしない）。
#
# 出力は `run_all_cuda.sh` と同じ「失敗を捏造しない」方針（skipped ログへ
# 記録。security.md A08）。
set -u
cd "$(dirname "$0")"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する（run_ab_train_cuda.sh
# と同じ方針）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. 0.6.0 or head-abc1234)" >&2
  exit 1
fi

OUT="results/raw/results-dgx-gemm-gate-${LABEL}.jsonl"
SKIP="results/raw/skipped-dgx-gemm-gate-${LABEL}.log"
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

ANY_FAILED=0

run() { # run <binary> <task> <device> <size> [mode] [extra_flag]
  local bin=$1 task=$2 device=$3 size=$4 mode=${5:-fresh} extra_flag=${6:-}
  echo "== $bin $task $device size=$size mode=$mode extra=${extra_flag:-none} =="
  if ! "./target/release/$bin" --task "$task" --device "$device" --size "$size" --mode "$mode" ${extra_flag:+"$extra_flag"} --out "$OUT" 2>err.tmp; then
    echo "$bin task=$task device=$device size=$size mode=$mode extra=${extra_flag:-none} : $(cat err.tmp)" >> "$SKIP"
    echo "  -> FAILED (recorded in $SKIP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

if [[ "${GEMM_GATE_SKIP_BUILD:-0}" != "1" ]]; then
  echo "== build bench-fandhe =="
  if ! cargo build --release -p bench-fandhe 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
    rm -f build-err.tmp
    echo "done. results in $OUT ; failures (if any) in $SKIP"
    exit 1
  fi
  rm -f build-err.tmp

  echo "== build bench-candle (cuda) =="
  if ! cargo build --release -p bench-candle --no-default-features --features cuda 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-candle BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
    rm -f build-err.tmp
    echo "done. results in $OUT ; failures (if any) in $SKIP"
    exit 1
  fi
  rm -f build-err.tmp
else
  echo "== GEMM_GATE_SKIP_BUILD=1: ビルドを省略（既存 ./target/release バイナリを使用） =="
fi

# nvidia-smi のスナップショットを実行ログへ残す（GPU 競合検出用の参考情報。
# 判定はしない。競合が疑われる run は人間・エージェントが確認して破棄する）。
echo "== nvidia-smi (before) =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 || true

for run_i in 1 2 3 4 5; do
  for n in 1024 2048 4096; do
    run bench-fandhe gemm cuda "$n" reuse
    run bench-candle gemm cuda "$n" fresh
  done
done

echo "== nvidia-smi (after) =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 || true

echo "done. results in $OUT ; failures (if any) in $SKIP"

if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
