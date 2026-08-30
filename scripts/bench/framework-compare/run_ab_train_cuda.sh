#!/bin/bash
# 都度同期廃止（#1011）の実践規模 A/B 計測（イシュー #1083）。
#
# `run_all_cuda.sh` は `bench-fandhe train cuda 64` を fresh/reuse 各 1 回しか
# 起動しないため、そのままでは coding-rust.md の「ベンチは 5 回計測の中央値」
# を満たせない。本スクリプトはバイナリを fresh/reuse それぞれ 5 回起動し、
# `compare_ab.py` が before/after の 5 レコードから中央値・min/max を算出できる
# 形で JSONL に追記する（1 回の起動 = 1 レコードのため、比較対象ラベル
# 〈before/after〉ごとに本スクリプトを実行する運用）。
#
# 呼び出し例（Phase C・DGX Spark 実機。ユーザー承認・別セッション）:
#   bash run_ab_train_cuda.sh before-0.4.0
#   bash run_ab_train_cuda.sh after-0.5.0
#
# 出力は `run_all_cuda.sh` と同じ「失敗を捏造しない」方針（skipped ログへ記録。
# security.md A08）。`--phases` は診断用（#1009 の区間分解）で fresh/reuse
# 各 1 回のみ（`--phases` は必ず単発計測でありプロトコル比較の主対象ではない）。
set -u
cd "$(dirname "$0")"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名・パスに直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する（bench-common の
# `InvalidMode`/`InvalidPhaseName` と同じ「未知の値をそのまま下流へ流さず
# ここで fail-fast する」思想）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. before-0.4.0)" >&2
  exit 1
fi

OUT="results/raw/results-dgx-ab-${LABEL}.jsonl"
SKIP="results/raw/skipped-dgx-ab-${LABEL}.log"
mkdir -p results/raw
: > "$OUT"
: > "$SKIP"

# fail-closed（AGENTS.md）: run() 内の個々の起動失敗はログに記録しつつ計測を
# 継続する（1 回の失敗で残りの試行を打ち切らない）が、スクリプト全体としては
# 1 件でも失敗があれば呼び出し元へ非 0 終了で伝播させる。ANY_FAILED はその
# ための集計フラグ（真偽ではなく失敗件数を保持し、末尾の exit 判定材料と
# スクリプト内 "-> FAILED" 表示の裏付けに使う）。
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

echo "== build bench-fandhe =="
if ! cargo build --release -p bench-fandhe 2>build-err.tmp; then
  tail -40 build-err.tmp
  echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >> "$SKIP"
  echo "  -> BUILD FAILED (recorded in $SKIP)"
  rm -f build-err.tmp
  echo "done. results in $OUT ; failures (if any) in $SKIP"
  exit 1
fi
rm -f build-err.tmp

# (A) 1 step 総和の A/B 本体: fresh・reuse それぞれ 5 回起動（coding-rust.md
# 「ベンチは 5 回計測の中央値」。バイナリ内部の 100 step warmup/measure と
# 別の外側反復で、`compare_ab.py` がこの 5 レコードの median_s から中央値を
# 取る）。
for i in 1 2 3 4 5; do
  run bench-fandhe train cuda 64 fresh
done
for i in 1 2 3 4 5; do
  run bench-fandhe train cuda 64 reuse
done

# (B) 診断用フェーズ分解（イシュー #1009。§2 の同期点分析の裏付け）。
# `--phases` はプロトコル比較の主対象ではないため 1 回のみ。
run bench-fandhe train cuda 64 fresh --phases
run bench-fandhe train cuda 64 reuse --phases

echo "done. results in $OUT ; failures (if any) in $SKIP"

# fail-closed（AGENTS.md）: fresh/reuse や --phases の一部が失敗した不完全な
# 計測を、呼び出し元（実行者・CI 相当）が成功と誤判定しないよう非 0 終了する
# （github-actions[bot] レビュー指摘。PR #1088）。
if [[ "$ANY_FAILED" -gt 0 ]]; then
  echo "FAILED: $ANY_FAILED run(s) failed; see $SKIP" >&2
  exit 1
fi
