#!/bin/bash
# `gemm_auto_f16_mma_switch_bench`（#1156・PR #1177）の独立 5 回起動・
# run-median 集約 wrapper（codex-review PR #1177 P1 是正。2 回目）。
#
# `docs/dispatch-rules-design.md` §5.6・`.claude/rules/coding-rust.md`
# 「ベンチは 5 回計測の中央値を採用し」が求める「5 回計測中央値」は
# 独立した 5 回のプロセス起動それぞれの計測値を指す。バイナリ側
# （`crates/backend-cuda/examples/gemm_auto_f16_mma_switch_bench.rs`）は
# プロセス起動ごとに各形状 1 回だけ計測して終了する設計へ是正済みのため、
# 独立 5 回起動・形状ごとの run-median 集計は本スクリプトが外側から担う
# （同一プロセス内ループでは捕捉できないプロセス起動・クロック・熱条件の
# 独立実行間ばらつきを、実際にプロセスを 5 回起動することで反映する）。
#
# 実行例（実機・DGX Spark GB10 等。CUDA 実機なし環境では各プロセスが
# skip メッセージを出して終了する）:
#   bash scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh
#
# イシュー #1160 是正: `BIN` をリポジトリ標準の `target/` へ固定して
# いたため、実機手順（`docs/real-hardware-verification-env.md` §3/§4.1）
# が指示する `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai` 運用（複数
# セッション・複数 worktree でビルド成果物ディレクトリを共有・分離する
# ための外部変数）の下では `cargo build` の成果物が `CARGO_TARGET_DIR`
# 側に置かれ、本スクリプトが探す `target/release/...` には存在せず
# 「ビルド成果物が見つからない」で失敗していた。`cargo build` 自体は
# 環境変数 `CARGO_TARGET_DIR` を尊重するため、`BIN` の探索先も同じ変数
# （未設定時は cargo の既定 `target/` へフォールバック）へ揃える。
set -u
cd "$(dirname "$0")/../.." || exit 1

RUNS=5
SIZES=(512 1024 2048 4096)
BIN_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$BIN_LOG_DIR"' EXIT

echo "gemm_auto_f16_mma_switch_bench: 独立 ${RUNS} 回のプロセス起動で計測する。"

cargo build -p fandhe-ai-backend-cuda --example gemm_auto_f16_mma_switch_bench --release
build_status=$?
if [[ $build_status -ne 0 ]]; then
  echo "gemm_auto_f16_mma_switch_bench: cargo build に失敗（exit=${build_status}）。計測を中止する。" >&2
  exit "$build_status"
fi
BIN="${CARGO_TARGET_DIR:-target}/release/examples/gemm_auto_f16_mma_switch_bench"
if [[ ! -x "$BIN" ]]; then
  echo "gemm_auto_f16_mma_switch_bench: ビルド成果物 ${BIN} が見つからない。" >&2
  exit 1
fi

for run in $(seq 1 "$RUNS"); do
  log_file="${BIN_LOG_DIR}/run_${run}.log"
  # 各 run は独立した OS プロセスとしてバイナリを起動する（本スクリプト
  # のプロセスからの fork+exec。プロセス起動・クロック・熱条件が run 間で
  # 独立するのは、同一プロセス内ループではなくこの `"$BIN"` 呼び出しが
  # run のたびに新規プロセスを生成するため）。
  "$BIN" >"$log_file" 2>&1
  run_status=$?
  if [[ $run_status -ne 0 ]]; then
    echo "gemm_auto_f16_mma_switch_bench: run ${run} が異常終了（exit=${run_status}）。" >&2
    cat "$log_file" >&2
    exit "$run_status"
  fi
  if grep -q "skipping\." "$log_file"; then
    echo "gemm_auto_f16_mma_switch_bench: run ${run} は CUDA 実機なしのため skip。"
    cat "$log_file"
    exit 0
  fi
done

echo "gemm_auto_f16_mma_switch_bench: ${RUNS} 回の独立起動が完了。形状別 run-median を集計する。"

for size in "${SIZES[@]}"; do
  values=()
  for run in $(seq 1 "$RUNS"); do
    log_file="${BIN_LOG_DIR}/run_${run}.log"
    value=$(grep -oE "^size=${size} auto_f16_tflops=[0-9.]+" "$log_file" | sed -E "s/^size=${size} auto_f16_tflops=//")
    if [[ -z "$value" ]]; then
      echo "gemm_auto_f16_mma_switch_bench: run ${run} に size=${size} の出力が見つからない。" >&2
      exit 1
    fi
    values+=("$value")
  done
  values_csv=$(IFS=,; echo "${values[*]}")
  run_median=$(python3 -c "import statistics,sys; print(f'{statistics.median([float(v) for v in sys.argv[1:]]):.4f}')" "${values[@]}")
  echo "size=${size} auto_f16_tflops_runs=[${values_csv}] auto_f16_tflops_run_median=${run_median}"
done
