#!/bin/sh
# Apple M4 Max（本セッションのホスト自身）側オーケストレーション（イシュー
# #1521）。Metal GEMM candle 比ゲート（旧 #1037）の対照系列
# （`0.8.0-ctrl-1521`。registry `fandhe-ai =0.8.0`）と参考系列
# （`head-<short sha>-1521`。split-K 結線後 HEAD への `crates/facade`
# path patch）を同一セッション内で A→B の固定順に計測する。
#
# `docs/perf/logs/metal-gemm-splitk-framework-compare-1517/orchestrate.sh`
# の record_only 型（専有ゲートなし・二重起動防止ロック・バックグラウンド
# uptime サンプラー・watchlist プロセス件数記録・`--dry-run`）を踏襲し、
# `docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/orchestrate_m4max.sh`
# のプレビルド分離・pmset サーマル記録を組み合わせる。
# ルート #1509 のユーザー指示（専有ゲートは要件にしない）に従い、本
# スクリプトは専有ゲートを持たない（record_only。負荷推移は
# monitor ログへ記録するのみで合否判定はしない）。
#
# 使い方: ./orchestrate_m4max.sh <short-sha> [--dry-run]
#   <short-sha>: split-K 結線後 HEAD のコミット短縮 SHA（[0-9a-f]{7,40}。
#     参考系列のラベル `head-<short-sha>-1521` に埋め込む）
#   環境変数 GEMM_GATE_PATCH_FACADE_PATH: 参考系列がビルドする
#     `crates/facade` の絶対パス（通常は本 worktree 自身の
#     `crates/facade`）。未設定なら実行せず fail-closed で終了する。
#
# 実行対象ツリーは呼び出し元の cwd（scripts/bench/framework-compare）と
# する。ログ出力先は環境変数 LOG で上書きできる。
set -u

SHORT_SHA="${1:-}"
case "$SHORT_SHA" in
  '')
    echo "usage: $0 <short-sha> [--dry-run]" >&2
    echo "  env GEMM_GATE_PATCH_FACADE_PATH=<crates/facade 絶対パス> が必須（参考系列のビルド元）" >&2
    exit 1
    ;;
  *[!0-9a-f]*)
    echo "short-sha は 16 進小文字のみ許可する（got: ${SHORT_SHA}）" >&2
    exit 1
    ;;
esac
SHA_LEN=${#SHORT_SHA}
if [ "$SHA_LEN" -lt 7 ] || [ "$SHA_LEN" -gt 40 ]; then
  echo "short-sha は 7〜40 文字である必要がある（got length=${SHA_LEN}）" >&2
  exit 1
fi

DRY_RUN=0
shift || true
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *)
      echo "未知の引数: '$arg'（既知の引数: --dry-run）" >&2
      exit 1
      ;;
  esac
done

# `GEMM_GATE_PATCH_FACADE_PATH` の検証（A03・A08。`run_ab_splitk_metal.sh`
# の `validate_facade_path` と同一方針: 絶対パス・`"`／`\`／空白禁止・
# Cargo.toml 存在・crate 名 `fandhe-ai` 一致）。本スクリプト自身が
# 検証するのは、ログファイル名にラベルは埋め込むが値自体は埋め込まない
# ため二重にはならず、`run_gemm_gate.sh` 側の検証（引用符・バックスラッシュ
# のみ）より早期に誤設定を検出するため。
FACADE_PATH="${GEMM_GATE_PATCH_FACADE_PATH:-}"
if [ -z "$FACADE_PATH" ]; then
  echo "error: GEMM_GATE_PATCH_FACADE_PATH is required (absolute path to a crates/facade worktree; issue #1521)" >&2
  exit 1
fi
case "$FACADE_PATH" in
  /*) ;;
  *)
    echo "error: GEMM_GATE_PATCH_FACADE_PATH must be an absolute path (got: $FACADE_PATH)" >&2
    exit 1
    ;;
esac
case "$FACADE_PATH" in
  *'"'*|*'\'*|*' '*)
    echo "error: GEMM_GATE_PATCH_FACADE_PATH must not contain '\"', '\\', or a space (got: $FACADE_PATH)" >&2
    exit 1
    ;;
esac
if [ ! -f "$FACADE_PATH/Cargo.toml" ]; then
  echo "error: GEMM_GATE_PATCH_FACADE_PATH/Cargo.toml not found ($FACADE_PATH)" >&2
  exit 1
fi
if ! grep -qE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"fandhe-ai"[[:space:]]*$' "$FACADE_PATH/Cargo.toml"; then
  echo "error: GEMM_GATE_PATCH_FACADE_PATH/Cargo.toml does not declare name = \"fandhe-ai\" ($FACADE_PATH)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FC_DIR="$(cd "$SCRIPT_DIR/../../../../scripts/bench/framework-compare" && pwd)"

LABEL_A="0.8.0-ctrl-1521"
LABEL_B="head-${SHORT_SHA}-1521"

DIFF_HEAD="$SCRIPT_DIR/diff_v0.8.0_head_metal_path.txt"
DIFF_COMMITTED="$SCRIPT_DIR/diff_v0.8.0_origin-main_metal_path.txt"
PREBUILD_LOG="$SCRIPT_DIR/prebuild-1521.log"
UPTIME_BEFORE="$SCRIPT_DIR/uptime_before.txt"
PMSET_BEFORE="$SCRIPT_DIR/pmset_therm_before.txt"
PMSET_AFTER="$SCRIPT_DIR/pmset_therm_after.txt"
MONITOR_LOG="$SCRIPT_DIR/monitor.log"
PROCS_LOG="$SCRIPT_DIR/procs.txt"
RUN_A_LOG="$SCRIPT_DIR/run_${LABEL_A}.log"
RUN_B_LOG="$SCRIPT_DIR/run_${LABEL_B}.log"
ALL_DONE="$SCRIPT_DIR/ALL_DONE_m4max.marker"
FAILED_MARKER="$SCRIPT_DIR/MEASUREMENT_FAILED_m4max.marker"

WATCHLIST="python torch mlx cargo gemm_ bench"

record_procs() {
  out="$1"
  {
    echo "# 固定 watchlist（${WATCHLIST}）に一致するプロセス名の件数のみを記録する"
    echo "# （コマンドライン全文・絶対パスは含めない）。"
    for name in $WATCHLIST; do
      count=$(ps -axo comm= 2>/dev/null | grep -ic -- "$name" || true)
      echo "watchlist_proc name=${name} count=${count}"
    done
  } > "$out"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] label_a=$LABEL_A label_b=$LABEL_B"
  echo "[dry-run]   facade_path   -> $FACADE_PATH"
  echo "[dry-run]   diff_head     -> ${DIFF_HEAD}（実行時点の v0.8.0..HEAD diff を再取得）"
  echo "[dry-run]   prebuild_log  -> $PREBUILD_LOG"
  echo "[dry-run]   uptime_before -> $UPTIME_BEFORE"
  echo "[dry-run]   pmset_before  -> $PMSET_BEFORE"
  echo "[dry-run]   monitor_log   -> ${MONITOR_LOG}（バックグラウンド uptime サンプラー・10 秒間隔）"
  echo "[dry-run]   procs_log     -> ${PROCS_LOG}（watchlist: ${WATCHLIST}）"
  echo "[dry-run]   step A        -> cd \"$FC_DIR\" && cargo build --release -p bench-fandhe && cargo build --release -p bench-candle && env -u GEMM_GATE_PATCH_FACADE_PATH bash run_gemm_gate_metal.sh \"$LABEL_A\""
  echo "[dry-run]   run_a_log     -> $RUN_A_LOG"
  echo "[dry-run]   step B        -> GEMM_GATE_PATCH_FACADE_PATH=\"$FACADE_PATH\" bash run_gemm_gate_metal.sh \"$LABEL_B\""
  echo "[dry-run]   run_b_log     -> $RUN_B_LOG"
  echo "[dry-run]   pmset_after   -> $PMSET_AFTER"
  exit 0
fi

# 既存成果物の確認（run の差し替え禁止。#1517 と同方針）。
for artifact in "$UPTIME_BEFORE" "$PMSET_BEFORE" "$PMSET_AFTER" "$RUN_A_LOG" "$RUN_B_LOG" "$MONITOR_LOG" "$PROCS_LOG" "$ALL_DONE"; do
  if [ -e "$artifact" ]; then
    echo "既存の成果物が見つかった（${artifact}）。同 label の再実行は" \
         "成果物を手動で別名へ退避してから行う（run の差し替え禁止）" >&2
    exit 1
  fi
done

# ロックは framework-compare ディレクトリ単位で取る（#1517 と同じ理由:
# Cargo.lock・path patch・before/after バイナリ等 $FC_DIR 配下の共有
# ファイルを書き換えるため、他 issue の実行と衝突しうる）。
LOCK_DIR="$FC_DIR/.splitk-framework-compare-1517.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "framework-compare ディレクトリの実行が既に進行中" \
       "（ロック $LOCK_DIR が存在する。他 issue を含め同時実行しない）" >&2
  exit 1
fi

SAMPLER_PID=""
cleanup() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

fail_measurement() {
  echo "$1 failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$FAILED_MARKER"
  exit 1
}

# --- 帰属根拠の diff・ログ・SHORT_SHA との一致・未コミット変更の
#     有無は、必ず参考系列（B）が実際にビルドする `FACADE_PATH` 側の
#     リポジトリから取得・検証する（codex P2 指摘: `$FC_DIR` 側〈本
#     スクリプトが置かれた worktree〉から取得すると、`FACADE_PATH` に
#     別 worktree を指定した場合に計測対象と異なるコードの差分を記録
#     してしまい、対象側の変更を検知できない）。`SHORT_SHA` との不一致・
#     計測経路配下の未コミット変更は fail-closed で停止する（警告のみ
#     で継続していた従来挙動を変更）。 ---
FACADE_ROOT="$(cd "$FACADE_PATH" && git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$FACADE_ROOT" ]; then
  fail_measurement "FACADE_PATH is not inside a git repository ($FACADE_PATH)"
fi
FACADE_HEAD_SHA="$(git -C "$FACADE_ROOT" rev-parse HEAD 2>/dev/null)"
if [ -z "$FACADE_HEAD_SHA" ]; then
  fail_measurement "failed to resolve HEAD sha in FACADE_ROOT ($FACADE_ROOT)"
fi
case "$FACADE_HEAD_SHA" in
  "$SHORT_SHA"*) ;;
  *)
    echo "error: SHORT_SHA ($SHORT_SHA) does not match FACADE_ROOT HEAD"          "($FACADE_HEAD_SHA at $FACADE_ROOT)" >&2
    fail_measurement "short-sha mismatch against FACADE_ROOT HEAD"
    ;;
esac
FACADE_DIRTY="$(git -C "$FACADE_ROOT" status --porcelain --   crates/backend-metal/src crates/facade/src crates/autodiff/src   crates/tensor-core/src 2>&1)"
if [ -n "$FACADE_DIRTY" ]; then
  echo "error: FACADE_ROOT has uncommitted changes under the measured path"        "(計測経路に未コミット変更あり。帰属根拠を確定できない):" >&2
  echo "$FACADE_DIRTY" >&2
  fail_measurement "uncommitted changes under measured path in FACADE_ROOT"
fi
(
  cd "$FACADE_ROOT"
  {
    echo '$ git diff v0.8.0..HEAD --stat -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src'
    git diff v0.8.0..HEAD --stat -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src 2>&1
    echo
    echo '$ git log --oneline v0.8.0..HEAD -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src'
    git log --oneline v0.8.0..HEAD -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src 2>&1
  } > "$DIFF_HEAD" 2>&1 || true
)
if [ -f "$DIFF_COMMITTED" ] && [ -f "$DIFF_HEAD" ]; then
  # ヘッダ行（`$ git diff v0.8.0..origin/main ...`／`$ git diff v0.8.0..HEAD
  # ...` 等のコマンド表記行）は比較対象の参照名（origin/main／HEAD）が異なる
  # ため実行のたびに必ず不一致になる。ヘッダ行を除いた本文（--stat 集計
  # 行・log 行）のみを比較し、実差分の有無だけを検知する。
  DIFF_COMMITTED_BODY="$(mktemp)"
  DIFF_HEAD_BODY="$(mktemp)"
  grep -v '^\$ git ' "$DIFF_COMMITTED" > "$DIFF_COMMITTED_BODY"
  grep -v '^\$ git ' "$DIFF_HEAD" > "$DIFF_HEAD_BODY"
  if ! diff -q "$DIFF_COMMITTED_BODY" "$DIFF_HEAD_BODY" > /dev/null 2>&1; then
    echo "WARNING: v0.8.0..HEAD の Metal 計測経路 diff がコミット時点" \
         "（${DIFF_COMMITTED}）と異なる。帰属表（attribution.md）の再導出" \
         "が必要な可能性がある。差分: $DIFF_HEAD" >&2
  fi
  rm -f "$DIFF_COMMITTED_BODY" "$DIFF_HEAD_BODY"
fi

uptime > "$UPTIME_BEFORE"
pmset -g therm > "$PMSET_BEFORE" 2>&1 || true
record_procs "$PROCS_LOG"

: > "$MONITOR_LOG"
(
  while true; do
    TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    LOADS=$(uptime | sed -E 's/.*load averages?: *([0-9.]+)[, ]+([0-9.]+)[, ]+([0-9.]+).*/\1 \2 \3/')
    # shellcheck disable=SC2086
    set -- $LOADS
    L1="${1:-NA}"
    L5="${2:-NA}"
    L15="${3:-NA}"
    echo "${TS} load1=${L1} load5=${L5} load15=${L15}" >> "$MONITOR_LOG"
    sleep 10
  done
) &
SAMPLER_PID=$!

cd "$FC_DIR" || fail_measurement "cd FC_DIR"

# --- 系列 A（対照。registry `fandhe-ai =0.8.0`）: ビルドを計測から
#     分離する（#1490 と同じ設計判断。§16.3 のプレビルドを踏襲）。
#     各ステップの終了ステータスを個別に検査する（`set -e` を使わず、
#     `{ ...; } > log 2>&1` の `$?` はブロック内最後のコマンドの結果しか
#     反映しないため、途中の `cargo build` 失敗を握りつぶさないよう
#     明示的に確認する）。 ---
(
  echo "== prebuild A (registry) $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  cargo build --release -p bench-fandhe || exit 1
  cargo build --release -p bench-candle || exit 1
  echo "== prebuild A done $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
  git status --porcelain Cargo.lock
) > "$PREBUILD_LOG" 2>&1
if [ "$?" != "0" ]; then
  fail_measurement "prebuild A"
fi

# `GEMM_GATE_PATCH_FACADE_PATH` はこのオーケストレータ自身の起動時 env
# （README の `GEMM_GATE_PATCH_FACADE_PATH="$FACADE_PATH" ./orchestrate_m4max.sh ...`
# 起動）としてプロセス環境に残り続ける。`FACADE_PATH` へは既に退避済みのため、
# 系列 A（対照・registry 版）の起動では `env -u` で明示的に取り除き、
# `run_gemm_gate_metal.sh` の子プロセス（`run_gemm_gate.sh`）が
# `patch.crates-io.fandhe-ai.path` を誤って適用しないようにする
# （codex-review 指摘。放置すると A/B が同一ビルドになり得るが、
# `run_gemm_gate.sh` 側の manifest 整合性検査は env var の有無と実ビルド元の
# 自己整合性しか見ないため検知できない）。
if ! env -u GEMM_GATE_PATCH_FACADE_PATH \
     bash run_gemm_gate_metal.sh "$LABEL_A" > "$RUN_A_LOG" 2>&1; then
  fail_measurement "series A ($LABEL_A)"
fi

# --- 系列 B（参考。split-K 結線後 HEAD への crates/facade path patch）:
#     ビルドと計測は不可分（#1166 の設計。プレビルドしない）。 ---
if ! env GEMM_GATE_PATCH_FACADE_PATH="$FACADE_PATH" \
     bash run_gemm_gate_metal.sh "$LABEL_B" > "$RUN_B_LOG" 2>&1; then
  fail_measurement "series B ($LABEL_B)"
fi

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
SAMPLER_PID=""

pmset -g therm > "$PMSET_AFTER" 2>&1 || true

echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ) label_a=$LABEL_A label_b=$LABEL_B" > "$ALL_DONE"
echo "完了: A=$RUN_A_LOG B=${RUN_B_LOG}（監視ログ: ${MONITOR_LOG}）"
