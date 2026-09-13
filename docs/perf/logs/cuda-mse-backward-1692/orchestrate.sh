#!/usr/bin/env bash
# イシュー #1692: CUDA `mse_loss_backward` ストリーム順序契約確認の
# GB10 実機ノイズ床・再現性計測オーケストレーション。
#
# `crates/facade/tests/mse_backward_bench.rs::mse_backward_cases`
# （`FANDHE_BENCH_DEVICE=cuda`）を before/after 2 ツリーで 5 round・
# run 単位で起動順を反転しながら実行し、`bench[...].median_s=`／
# `grad[...].fold_bits=` 行を回収する（`run_bitdump.sh`〈#1560〉と同型の
# 「同一テストを 2 つの独立ツリーで実行して出力を突き合わせる」方式）。
#
# 事前登録判定規則（`docs/perf/cuda-mse-backward-stream-contract.md`
# §2）: `crates/*/src` の機能差分がなければ ADOPT／REJECT の判定対象
# ではなくノイズ床・再現性の記録として扱う。
#
# 使い方（GB10 実機。ユーザー承認・別セッション）:
#   BEFORE_TREE=/home/<user>/work/rust-ai-library-run-1692-before \
#   AFTER_TREE=/home/<user>/work/rust-ai-library-run-1692-after \
#     ./orchestrate.sh 1692
#
# `--dry-run` で経路解決のみ検証できる（実機不要。Linux で自己検証可能）。
set -u
LABEL=${1:-1692}
ROUNDS=5

# A03 インジェクション対策: ラベルはファイル名に直接埋め込む
# （`run-${LABEL}-*.log`）ため allowlist で検証する（パストラバーサル・
# コマンド注入の防止。`--dry-run` はラベル検証より前に処理する）。
if [[ "$LABEL" != "--dry-run" && ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label> [--dry-run]  (label must match [A-Za-z0-9._-]+)" >&2
  exit 1
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SELF_DIR}/runs-${LABEL}"

if [[ "${2:-}" == "--dry-run" || "${1:-}" == "--dry-run" ]]; then
  echo "dry-run: SELF_DIR=$SELF_DIR"
  echo "dry-run: label=$LABEL rounds=$ROUNDS"
  echo "dry-run: BEFORE_TREE=${BEFORE_TREE:-<unset>} AFTER_TREE=${AFTER_TREE:-<unset>}"
  exit 0
fi

if [[ -z "${BEFORE_TREE:-}" || -z "${AFTER_TREE:-}" ]]; then
  echo "usage: BEFORE_TREE=<absolute path> AFTER_TREE=<absolute path> $0 <label>" >&2
  exit 1
fi
for v in BEFORE_TREE AFTER_TREE; do
  path="${!v}"
  if [[ "$path" != /* ]]; then
    echo "error: $v must be an absolute path (got: $path)" >&2
    exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]]; then
    echo "error: $v/Cargo.toml not found ($path)" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

run_bench() { # run_bench <tree_path> <out_file>
  local tree=$1
  local out_file=$2
  (cd "$tree" && FANDHE_BENCH_DEVICE=cuda cargo test -p fandhe-ai --release \
    --test mse_backward_bench -- --ignored --nocapture --exact mse_backward_cases) \
    >"$out_file" 2>&1
}

extract_lines() {
  grep -E '^(bench\[|grad\[)'
}

for round in $(seq 1 "$ROUNDS"); do
  echo "== round ${round}/${ROUNDS} ==" >&2
  if (( round % 2 == 1 )); then
    order="before_first"
    run_bench "$BEFORE_TREE" "${OUT_DIR}/before_round${round}.log"
    before_rc=$?
    run_bench "$AFTER_TREE" "${OUT_DIR}/after_round${round}.log"
    after_rc=$?
  else
    order="after_first"
    run_bench "$AFTER_TREE" "${OUT_DIR}/after_round${round}.log"
    after_rc=$?
    run_bench "$BEFORE_TREE" "${OUT_DIR}/before_round${round}.log"
    before_rc=$?
  fi
  echo "round=${round} order=${order} before_rc=${before_rc} after_rc=${after_rc}" >>"${OUT_DIR}/rounds.log"
  if [[ "$before_rc" -ne 0 || "$after_rc" -ne 0 ]]; then
    echo "エラー: round ${round} で非 0 終了（before_rc=${before_rc} after_rc=${after_rc}）" >&2
    exit 1
  fi
  extract_lines <"${OUT_DIR}/before_round${round}.log" >"${OUT_DIR}/before_round${round}_lines.txt"
  extract_lines <"${OUT_DIR}/after_round${round}.log" >"${OUT_DIR}/after_round${round}_lines.txt"
done

echo "done. logs in $OUT_DIR" >&2
