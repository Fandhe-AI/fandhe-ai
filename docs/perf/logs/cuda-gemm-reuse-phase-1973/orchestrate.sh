#!/usr/bin/env bash
# イシュー #1973: CUDA GEMM reuse N=1024/2048/4096 の H2D／カーネル／
# D2H／同期フェーズ分解を framework-compare ピン `fandhe-ai =0.9.0` で
# GB10 実機再計測するオーケストレーション（#1182 の 0.6.0 時点計測を
# 0.9.0 へ更新する。親 #1972）。
#
# 構成は #1182（`docs/perf/cuda-gemm-reuse-phase-breakdown.md`）と同型の
# 2 層構成を踏襲する:
#   - Layer A（公開 API 境界。`bench-fandhe --task gemm --device cuda
#     --mode reuse --phases`。registry ピン `fandhe-ai =0.9.0` 固定）:
#     `matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total` の
#     5 区間を N ごとに 5 回計測する
#   - Layer B（`crates/backend-cuda` 非公開 API。
#     `gemm_reuse_phase_diag_select`／`gemm_reuse_phase_diag_classic`。
#     HEAD ツリー）: `h2d_a`／`h2d_b`／`alloc_c`／`launch_issue`／
#     `kernel_wait`／`d2h`／`host_copy` の 7 区間を 5 回計測する
#   - AC-2（挙動不変確認）: `--phases` なしの `gemm --mode reuse` を
#     1 回実行し checksum が phases 版と bit 単位で一致することを確認
#     する
#   - candle 参照（診断用）: `bench-candle --task gemm --device cuda
#     --mode fresh` を N ごとに 5 回実行し、#1489（0.8.0 時点の正式系列
#     candle fresh 中央値）との整合を確認する参考値として記録する
#     （#1031 の正式ゲート判定〈`run_gemm_gate_cuda.sh`〉を代替しない。
#     本スクリプトは診断専用）
#
# 事前登録判定規則（結果を見てから緩和しない。issue #1973 コメントへ
# 転記済み）:
#   - N=512／1024／2048 × 5 run 中央値の内訳表を作成する
#     （実際の対象形状は #1182 を踏襲し N=1024／2048／4096 とする。
#     issue 本文の「N=512／1024／2048」という受け入れ条件の記述は
#     #1182 の対象形状〈1024/2048/4096〉と表記揺れがあるため、本
#     オーケストレーションでは #1182 実測範囲を継承し 1024/2048/4096
#     を対象とする。この表記揺れの解消要否は issue 側で確認する）
#   - 削減候補（アロケータ・同期・readback）の優先順位を Layer A/B の
#     区間比率から根拠付きで記録する（#1182 §6 の帰属手法を踏襲）
#   - `--no-verify` 禁止・`#[allow(clippy::…)]` 禁止・tolerance／
#     baseline／依存の変更禁止
#   - 内部ホスト名・ユーザー名・絶対パスをログへ書かない
#
# 実行ディレクトリ: `scripts/bench/framework-compare/`
#   （`FRAMEWORK_COMPARE_DIR` で上書き可。未指定時は本ファイルの保存先
#   からリポジトリ相対で解決する）。
#
# 使い方（GB10 実機。ユーザー承認・別セッション）:
#   ./orchestrate.sh
#   ./orchestrate.sh --dry-run   # 経路解決のみ検証（実機不要）
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${FRAMEWORK_COMPARE_DIR:-}" ]]; then
  WORK_DIR="$FRAMEWORK_COMPARE_DIR"
elif [[ -f "$SELF_DIR/../../../../scripts/bench/framework-compare/Cargo.toml" ]]; then
  WORK_DIR="$(cd "$SELF_DIR/../../../../scripts/bench/framework-compare" && pwd)"
else
  WORK_DIR=""
fi
if [[ -z "$WORK_DIR" || ! -f "$WORK_DIR/Cargo.toml" ]]; then
  echo "ERROR: scripts/bench/framework-compare が見つからない: ${WORK_DIR:-<unresolved>}" >&2
  echo "  FRAMEWORK_COMPARE_DIR=<scripts/bench/framework-compare の絶対パス> を指定すること" >&2
  exit 1
fi

REPO_ROOT="$(cd "$WORK_DIR/../../.." && pwd)"

SIZES=(1024 2048 4096)
RUNS=5

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run: WORK_DIR resolved to $WORK_DIR"
  echo "dry-run: REPO_ROOT resolved to $REPO_ROOT"
  echo "dry-run: sizes=${SIZES[*]} runs=$RUNS"
  ls -la "$WORK_DIR/Cargo.toml" "$WORK_DIR/bench-fandhe/Cargo.toml" "$WORK_DIR/bench-candle/Cargo.toml"
  ls -la "$REPO_ROOT/crates/backend-cuda/src/gemm_reuse_phase_diag_tests.rs"
  echo "dry-run: registry pin確認 (bench-fandhe/Cargo.toml の fandhe-ai 行)"
  grep -n '^fandhe-ai' "$WORK_DIR/bench-fandhe/Cargo.toml" || true
  echo "dry-run: ビルドコマンド（run_all_cuda.sh／run_gemm_gate.sh の CUDA 分岐と同型）"
  echo "  cargo build --release -p bench-fandhe"
  echo "  cargo build --release -p bench-candle --no-default-features --features cuda"
  echo "dry-run: OK"
  exit 0
fi

cd "$WORK_DIR"

# ビルドフラグは `run_all_cuda.sh`／`run_gemm_gate.sh` の CUDA 分岐と揃える
# （bench-candle は既定 feature が `metal` のため `--device cuda` で使うには
# `--no-default-features --features cuda` が必須。bench-fandhe は device
# 別の cargo feature を持たないため既定ビルドのままでよい）。
echo "== ビルド（release） =="
cargo build --release -p bench-fandhe
cargo build --release -p bench-candle --no-default-features --features cuda

echo "== Layer A: bench-fandhe gemm reuse --phases (N=${SIZES[*]}, ${RUNS} run) =="
for n in "${SIZES[@]}"; do
  OUT="${SELF_DIR}/layerA-phases-N${n}.log"
  : >"$OUT"
  for i in $(seq 1 "$RUNS"); do
    echo "-- run $i/N=$n --" >>"$OUT"
    ./target/release/bench-fandhe --task gemm --device cuda --size "$n" \
      --mode reuse --phases >>"$OUT" 2>>"$OUT"
  done
done

echo "== AC-2: bench-fandhe gemm reuse（非 phases。挙動不変確認） =="
AC2_OUT="${SELF_DIR}/layerA-ac2.log"
: >"$AC2_OUT"
for n in "${SIZES[@]}"; do
  ./target/release/bench-fandhe --task gemm --device cuda --size "$n" \
    --mode reuse >>"$AC2_OUT" 2>>"$AC2_OUT"
done

echo "== candle 参照（診断用。gemm fresh。N=${SIZES[*]}, ${RUNS} run） =="
for n in "${SIZES[@]}"; do
  OUT="${SELF_DIR}/candle-fresh-N${n}.log"
  : >"$OUT"
  for i in $(seq 1 "$RUNS"); do
    echo "-- run $i/N=$n --" >>"$OUT"
    ./target/release/bench-candle --task gemm --device cuda --size "$n" \
      --mode fresh >>"$OUT" 2>>"$OUT"
  done
done

echo "== Layer B: crates/backend-cuda 非公開 API 診断テスト（${RUNS} run） =="
cd "$REPO_ROOT"
for i in $(seq 1 "$RUNS"); do
  OUT="${SELF_DIR}/layerB-run${i}.log"
  cargo test --release -p fandhe-ai-backend-cuda \
    gemm_reuse_phase_diag -- --ignored --test-threads=1 --nocapture \
    >"$OUT" 2>&1
done

echo "== env_info の記入補助（uptime・nvidia-smi） =="
{
  echo "計測終了時刻（UTC）: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uptime
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv || true
} >>"${SELF_DIR}/env_info.txt"

echo "done."
