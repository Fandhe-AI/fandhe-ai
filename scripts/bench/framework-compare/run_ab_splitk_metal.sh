#!/bin/bash
# イシュー #1545: Metal split-K opt-in 経路の runtime トグル
# （`fandhe_ai::set_metal_split_k_gemm_enabled`/`metal_split_k_gemm_enabled`。
# `#[cfg(target_os = "macos")]`。crates.io 公開版 `fandhe-ai =0.8.0` には
# 未収録のため `bench-fandhe` の `metal-split-k-toggle` feature〈既定無効〉
# 経由の path patch ビルド限定）を、同一バイナリで run 単位に interleave
# 計測する。
#
# **旧方式からの変更点（#1517 当時 → 本イシュー #1545 で置換）**: #1517
# 時点では `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` 定数（`crates/
# backend-metal/src/tile.rs`）の `false`/`true` を切り替えた 2 つの
# worktree（`AB_BEFORE_FACADE_PATH`/`AB_AFTER_FACADE_PATH`）を別々に
# ビルドして比較する方式だった（結線前後の実体がコンパイル時定数
# そのものだったため）。#1544 で当該定数が既定 `true` へ切り替わり
# split-K が本番経路として既定有効化された現在は、facade 公開 API
# （`set_metal_split_k_gemm_enabled`）による runtime on/off 切替が
# 可能になったため、`run_ab_readout_metal.sh`（`--readout legacy|
# borrowed` の同一バイナリ run 単位 interleave）と同型の**単一 facade
# path・単一バイナリ・`--metal-split-k on|off` の runtime 切替**方式へ
# 置換する（`docs/perf/metal-gemm-splitk-two-pass.md` §5.9 以降・
# `docs/backend-metal-splitk-decision.md` §5 参照）。
#
# 差分ガード（旧方式の「before==after で計測対象なし」再発防止・
# `docs/perf/train-step-phase-breakdown.md` §5.11 の教訓を踏襲）は、
# 2 バイナリ比較ではなくなったため以下の 2 点へ置き換える:
#   1. `AB_PATCH_FACADE_PATH/../backend-metal/src/split_k_runtime.rs` の
#      `SPLIT_K_RUNTIME_ENABLED`（実行時トグル本体）の既定値宣言行が
#      `true` であること（`--metal-split-k off` が「本番経路が元々 off
#      だから off に見える」だけの無意味な比較にならないことの確認。
#      #1547 でコンパイル時定数ゲート `tile::
#      SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` を撤去したため対象を
#      付け替えた）。
#   2. ビルド後、`--metal-split-k off` のドライラン 1 回（極小サイズの
#      gemm 1 回）が MEASURE_ERROR にならないこと（`metal-split-k-toggle`
#      feature が実際に有効化されていることの確認）。
#
# gemm 8 セル（N=512/1024/2048/4096 × fresh/reuse）に加え、train 2 セル
# （`--task train` × fresh/reuse。`docs/perf/metal-gemm-splitk-framework-
# compare-1517.md` §2 の帰属表参照）を計測する。gemm と train は
# `compare_gemm_ab.py --task <t>` の task 別 fail-closed 検証（他タスク
# の行を警告つきで除外し 1 件でもあれば判定不能にする。`load_rows`
# docstring）と整合させるため、タスク別の JSONL（`-gemm.jsonl`／
# `-train.jsonl`）へ出力を分離する（#1517 PR #1531 是正を踏襲）。
# `--phases`（train のみ・診断用・各腕 1 回）はさらに別ファイルへ出力し、
# 5 回計測の「ちょうど 5 件」契約を汚さない。
#
# 専有ゲートは要求しない（record_only。ルート #1509 運用方針。
# `run_ab_readout_metal.sh` の `AB_LOAD_GATE_MODE=record_only` と同型の
# 「待機なし・load average を記録するのみ」を既定にする）。
# `uptime`／`pmset -g therm` を各 run の前後で記録し負荷推移を残す。
#
# 呼び出し例（M4 Max 実機。Mac セッション）:
#   AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
#     bash run_ab_splitk_metal.sh splitk-toggle-1545
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ
# 一時ファイルを正規パスへ原子的に反映する。1 件でも失敗すれば正規パスは
# 変更せず、不完全な結果は `.failed-<UTC>` へ退避する。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する
# （`run_ab_readout_metal.sh` と同一方針）。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. splitk-toggle-1545)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to HEAD's crates/facade> is required" >&2
  exit 1
fi

DRY_RUN=0
if [[ "${AB_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08。`run_ab_readout_metal.sh` の
# `AB_PATCH_FACADE_PATH` 検証と同一方針）: 未設定・相対パス・`"`／`\`
# ／空白混入・Cargo.toml 不在・crate 名不一致のいずれかなら fail-closed
# で exit 1。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to HEAD's crates/facade; issue #1545)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" != /* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must be an absolute path (got: $AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" == *'"'* || "$AB_PATCH_FACADE_PATH" == *'\'* || "$AB_PATCH_FACADE_PATH" == *' '* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must not contain '\"', '\\', or a space" >&2
  exit 1
fi
if [[ ! -f "$AB_PATCH_FACADE_PATH/Cargo.toml" ]]; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml not found ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if ! grep -qE '^\s*name\s*=\s*"fandhe-ai"\s*$' "$AB_PATCH_FACADE_PATH/Cargo.toml"; then
  echo "error: AB_PATCH_FACADE_PATH/Cargo.toml does not declare name = \"fandhe-ai\" ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${AB_PATCH_FACADE_PATH}\""

# 差分ガード其の 1（上記コメント参照）: `SPLIT_K_RUNTIME_ENABLED`
# （`crates/backend-metal/src/split_k_runtime.rs`。実行時トグル本体の
# 既定値）が `true` であることを機械検証する（#1547 でコンパイル時定数
# ゲート `tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` を撤去した
# ため、本ガードの対象を実行時トグルの既定値宣言行へ付け替えた）。
# `grep -E` の alternation は BRE エスケープ不要（`(false|true)`）。
RUNTIME_RS="$(cd "$AB_PATCH_FACADE_PATH/../backend-metal" 2>/dev/null && pwd)/src/split_k_runtime.rs"
if [[ ! -f "$RUNTIME_RS" ]]; then
  echo "error: split_k_runtime.rs not found relative to AB_PATCH_FACADE_PATH ($AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
GATE_LINE="$(grep -E '^static SPLIT_K_RUNTIME_ENABLED: AtomicBool = AtomicBool::new\((false|true)\);$' "$RUNTIME_RS" || true)"
if [[ -z "$GATE_LINE" ]]; then
  echo "error: SPLIT_K_RUNTIME_ENABLED の宣言行を $RUNTIME_RS から特定できなかった（フォーマット変更の可能性。fail-closed）" >&2
  exit 1
fi
if [[ "$GATE_LINE" != *"(true);" ]]; then
  echo "error: SPLIT_K_RUNTIME_ENABLED の既定値が true ではない（${GATE_LINE}）。runtime トグルの on/off 比較が本番経路の on/off 比較にならないため fail-closed で停止する（issue #1545）。" >&2
  exit 1
fi
echo "gate check: SPLIT_K_RUNTIME_ENABLED=true ($RUNTIME_RS)"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "AB_DRY_RUN=1: バリデーションのみ完了（cargo/pmset/sysctl は実行しない）。"
  exit 0
fi

AB_ROUNDS=${AB_ROUNDS:-5}
if [[ ! "$AB_ROUNDS" =~ ^[0-9]+$ || "$AB_ROUNDS" -lt 1 ]]; then
  echo "error: AB_ROUNDS must be a positive integer (got: $AB_ROUNDS)" >&2
  exit 1
fi

SIZES=(512 1024 2048 4096)
MODES=(fresh reuse)

# タスク別 JSONL（`compare_gemm_ab.py --task <t>` の task 別 fail-closed
# 検証と整合させるため。#1517 PR #1531 是正を踏襲）。
OUT_OFF_GEMM="results/raw/results-m4max-splitk-ab-off-${LABEL}-gemm.jsonl"
OUT_ON_GEMM="results/raw/results-m4max-splitk-ab-on-${LABEL}-gemm.jsonl"
OUT_OFF_TRAIN="results/raw/results-m4max-splitk-ab-off-${LABEL}-train.jsonl"
OUT_ON_TRAIN="results/raw/results-m4max-splitk-ab-on-${LABEL}-train.jsonl"
OUT_OFF_PHASES="results/raw/results-m4max-splitk-ab-off-${LABEL}-phases.jsonl"
OUT_ON_PHASES="results/raw/results-m4max-splitk-ab-on-${LABEL}-phases.jsonl"
SKIP="results/raw/skipped-m4max-splitk-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-m4max-splitk-ab-${LABEL}.json"
mkdir -p results/raw

OUT_OFF_GEMM_TMP="${OUT_OFF_GEMM}.tmp"
OUT_ON_GEMM_TMP="${OUT_ON_GEMM}.tmp"
OUT_OFF_TRAIN_TMP="${OUT_OFF_TRAIN}.tmp"
OUT_ON_TRAIN_TMP="${OUT_ON_TRAIN}.tmp"
OUT_OFF_PHASES_TMP="${OUT_OFF_PHASES}.tmp"
OUT_ON_PHASES_TMP="${OUT_ON_PHASES}.tmp"
SKIP_TMP="${SKIP}.tmp"
: > "$OUT_OFF_GEMM_TMP"
: > "$OUT_ON_GEMM_TMP"
: > "$OUT_OFF_TRAIN_TMP"
: > "$OUT_ON_TRAIN_TMP"
: > "$OUT_OFF_PHASES_TMP"
: > "$OUT_ON_PHASES_TMP"
: > "$SKIP_TMP"

ANY_FAILED=0

sha256_of() {
  local f=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# `cargo tree -p bench-fandhe --depth 1` の `fandhe-ai` 行から解決元
# （path か registry か）を抽出する（`run_ab_readout_metal.sh` と同型の
# ハードゲート）。
fandhe_ai_source_desc() {
  local line path_part
  line=$(cargo tree -p bench-fandhe --depth 1 "$@" 2>/dev/null | grep -E "^[├└]── fandhe-ai " || true)
  if [[ -z "$line" ]]; then
    return 1
  elif [[ "$line" == *"("* ]]; then
    path_part=${line#*(}
    path_part=${path_part%)}
    echo "path:${path_part}"
  else
    echo "registry"
  fi
}

# `AB_LOAD_GATE_MODE=record_only`（既定。ルート #1509 運用方針）: 専有
# ゲートを要件にせず、現在の load average を 1 行記録するだけで即座に
# 計測を開始する。「共有負荷下であることと計測中の load average 推移を
# env_info に記録する」方針を、`gate-splitk-ab-<label>.log`／
# `uptime-splitk-ab-<label>.log` の記録先で満たす
# （`run_ab_readout_metal.sh` の `record_only` モードと同型。専有ゲート
# の `exclusive` モードは本スクリプトでは実装しない — 旧 `run_ab_
# splitk_metal.sh` も専有ゲートを要求しない前提のため、必要になれば
# `run_ab_readout_metal.sh` から `wait_for_exclusive_gate` を移植する）。
load1_now() {
  local raw parsed
  if ! raw="$(uptime 2>/dev/null)"; then
    return 0
  fi
  parsed="$(printf '%s\n' "$raw" | sed -E 's/.*load average[s]?: ([0-9.]+).*/\1/')"
  printf '%s' "$parsed"
}
load1_is_valid() {
  local v="$1"
  [[ -n "$v" ]] && awk -v l="$v" 'BEGIN{exit !(l ~ /^[0-9]+(\.[0-9]+)?$/ && l + 0 >= 0)}'
}
record_only_gate_note() {
  local gate_log="results/raw/gate-splitk-ab-${LABEL}.log"
  local l1
  l1="$(load1_now)"
  : > "$gate_log"
  if ! load1_is_valid "$l1"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1_invalid=${l1:-<empty>}（専有ゲート要件なし。issue #1545・ルート #1509）" | tee -a "$gate_log"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1=${l1}（専有ゲート要件なし。issue #1545・ルート #1509）" | tee -a "$gate_log"
  fi
}
record_only_gate_note

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (used to parse 'cargo build --message-format=json' artifact paths)" >&2
  exit 1
fi

# `Cargo.lock` の退避・復元（`run_ab_readout_metal.sh` と同一方針。deps-
# policy.md 第 9 区分「`[patch]` は CLI 引数のみで与え Cargo.lock は
# 変更しない」契約）。
LOCK_BACKUP="$(mktemp)"
if ! cp Cargo.lock "$LOCK_BACKUP"; then
  echo "error: cp Cargo.lock '$LOCK_BACKUP' (backup) に失敗した" >&2
  rm -f "$LOCK_BACKUP"
  exit 1
fi
if [[ "$(sha256_of Cargo.lock)" != "$(sha256_of "$LOCK_BACKUP")" ]]; then
  echo "error: Cargo.lock のバックアップ内容が元ファイルと一致しない（不完全な退避の可能性）" >&2
  rm -f "$LOCK_BACKUP"
  exit 1
fi
restore_lock() {
  if ! cp "$LOCK_BACKUP" Cargo.lock; then
    echo "error: Cargo.lock の復元に失敗した。バックアップを保持する: $LOCK_BACKUP" >&2
    return 1
  fi
  rm -f "$LOCK_BACKUP"
}
restore_lock_trap() {
  local code=$?
  if ! restore_lock; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
  fi
  exit "$code"
}
trap restore_lock_trap EXIT

# `build_bench_fandhe`（旧 `run_ab_splitk_metal.sh`〈#1517〉と同一方式。
# codex-review 指摘・PR #1546）: `target/release/bench-fandhe` を直接
# 決め打ちで参照すると、`CARGO_TARGET_DIR`／`.cargo/config.toml` の
# `build.target-dir`（Cargo が対象トリプルごとのサブディレクトリへ成果物
# を出す設定）が指定された環境では実際のビルド成果物とは異なる場所を
# 読むことになり、また同名の旧バイナリが残っていた場合はビルド失敗時
# でも古いバイナリを計測してしまう fail-open の危険がある。`--target-dir
# target`（本スクリプトの cwd 基準に固定）と `cargo build --message-
# format=json` の JSON Lines 出力から `compiler-artifact` の
# `target.name == "bench-fandhe"` かつ `target.kind` に `"bin"` を含む
# 最後のエントリの `executable` を `jq` で抽出し、そのパスを以後の
# sha256 記録・ドライラン・全計測で使う。
build_bench_fandhe() { # build_bench_fandhe <out_exe_pathvar> [追加の cargo build 引数...]
  local __out_var=$1
  shift
  local msg_file
  msg_file="$(mktemp)"
  if ! cargo build --release -p bench-fandhe --target-dir target --message-format=json "$@" >"$msg_file" 2>build-err.tmp; then
    tail -40 build-err.tmp
    echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >&2
    rm -f build-err.tmp "$msg_file"
    exit 1
  fi
  rm -f build-err.tmp
  local exe
  exe="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .target.name == "bench-fandhe" and (.target.kind[]? == "bin") and .executable != null)] | last | .executable // empty' "$msg_file")"
  rm -f "$msg_file"
  if [[ -z "$exe" || ! -f "$exe" ]]; then
    echo "error: bench-fandhe ビルド成果物のパスを 'cargo build --message-format=json' から特定できなかった（jq 抽出結果: '${exe:-<空>}')" >&2
    exit 1
  fi
  printf -v "$__out_var" '%s' "$exe"
}

echo "== build bench-fandhe (HEAD path patch, --features metal-split-k-toggle) =="
build_bench_fandhe BUILT_EXE --features metal-split-k-toggle --config "$PATCH_CONFIG"

SOURCE_DESC="$(fandhe_ai_source_desc --features metal-split-k-toggle --config "$PATCH_CONFIG" || true)"
if [[ "$SOURCE_DESC" != "path:${AB_PATCH_FACADE_PATH}" ]]; then
  echo "error: fandhe-ai が期待した path 解決ではない (expected=path:${AB_PATCH_FACADE_PATH} actual=${SOURCE_DESC:-<取得失敗>})" >&2
  exit 1
fi

# 抽出した成果物パスを、以後の全参照（sha256・ドライラン・run_gemm／
# run_train・`--phases`）が使う既知の固定パスへコピーする（旧 splitk
# スクリプトの before/after コピーと同型。ビルド成果物パス自体を都度
# jq で再取得せずに済み、計測中のバイナリ入れ替え検出〈`verify_binary`〉
# の対象を単純化する）。
mkdir -p target/release
if ! cp "$BUILT_EXE" target/release/bench-fandhe-splitk-toggle; then
  echo "error: cp '$BUILT_EXE' target/release/bench-fandhe-splitk-toggle に失敗した" >&2
  exit 1
fi
EXE=target/release/bench-fandhe-splitk-toggle

BIN_SHA="$(sha256_of "$EXE")"
echo "bench-fandhe sha256: $BIN_SHA (source: $SOURCE_DESC, exe: $BUILT_EXE)"

# 差分ガード其の 2（冒頭コメント参照）: `--metal-split-k off` のドライラン
# 1 回（極小サイズの gemm。計測対象の JSONL・SKIP には出力しない）が
# MEASURE_ERROR にならないことを確認する。`metal-split-k-toggle` feature
# が実際に有効化されビルドへ反映されていることの機械確認
# （feature フラグ自体は `--config` 経由の path patch と独立してビルド
# 時に固定されるため、ビルド成功だけでは runtime 分岐が意図どおり
# 通っているか判別できない）。
DRYRUN_OUT="$(mktemp)"
if ! "./$EXE" --task gemm --device metal --size 64 --mode fresh --metal-split-k off --out "$DRYRUN_OUT" 2>dryrun-err.tmp; then
  echo "error: --metal-split-k off のドライランが失敗した（metal-split-k-toggle feature が有効に反映されていない可能性）: $(cat dryrun-err.tmp)" >&2
  rm -f dryrun-err.tmp "$DRYRUN_OUT"
  exit 1
fi
rm -f dryrun-err.tmp "$DRYRUN_OUT"
echo "gate check: --metal-split-k off dry-run OK (metal-split-k-toggle feature is active)"

SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
FACADE_HEAD_SHA="$(git -C "$AB_PATCH_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
MANIFEST_TMP="${MANIFEST}.tmp"
cat > "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"metal","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","facade_head_sha":"${FACADE_HEAD_SHA}","bin_sha256":"${BIN_SHA}","bin_source":"${SOURCE_DESC}","split_k_arms":["off","on"],"gate_mode":"record_only","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
echo "== manifest（一時ファイル）記録: $MANIFEST_TMP =="
cat "$MANIFEST_TMP"

verify_binary() {
  local now
  now="$(sha256_of "$EXE")"
  if [[ "$now" != "$BIN_SHA" ]]; then
    echo "error: bench-fandhe のバイナリが計測中に変化した（sha256 不一致）" >&2
    exit 1
  fi
}

run_gemm() { # run_gemm <arm(off|on)> <out_tmp> <size> <mode>
  local arm=$1 out=$2 size=$3 mode=$4
  verify_binary
  echo "== bench-fandhe gemm metal size=$size mode=$mode metal-split-k=$arm =="
  if ! "./$EXE" --task gemm --device metal --size "$size" --mode "$mode" --metal-split-k "$arm" --out "$out" 2>err.tmp; then
    echo "gemm metal size=$size mode=$mode metal-split-k=$arm : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

run_train() { # run_train <arm(off|on)> <out_tmp> <mode>
  local arm=$1 out=$2 mode=$3
  verify_binary
  echo "== bench-fandhe train metal mode=$mode metal-split-k=$arm =="
  if ! "./$EXE" --task train --device metal --mode "$mode" --metal-split-k "$arm" --out "$out" 2>err.tmp; then
    echo "train metal mode=$mode metal-split-k=$arm : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

echo "== metal status (before loop) =="
sysctl -n machdep.cpu.brand_string 2>&1 || true
pmset -g therm 2>&1 || true
uptime 2>&1 || true

UPTIME_SAMPLER_LOG="results/raw/uptime-splitk-ab-${LABEL}.log"
: > "$UPTIME_SAMPLER_LOG"
(
  while true; do
    { date -u +%Y-%m-%dT%H:%M:%SZ; uptime; } >> "$UPTIME_SAMPLER_LOG" 2>&1
    sleep 30
  done
) &
UPTIME_SAMPLER_PID=$!
# `restore_lock_trap`（Cargo.lock 復元）を上書きせず、バックグラウンド
# サンプラーの kill も併せて行う合成 trap へ差し替える（元の trap を
# 上書きしたまま計測ループ中に exit すると Cargo.lock が復元されない
# 事故を防ぐ。`run_ab_readout_metal.sh` と同一方針）。
restore_lock_and_kill_sampler_trap() {
  local code=$?
  kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
  if ! restore_lock; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
  fi
  exit "$code"
}
trap restore_lock_and_kill_sampler_trap EXIT

# run 単位で off/on を交互起動する。奇数 run: off→on・偶数 run: on→off
# （起動順序自体の系統誤差を均す。`run_ab_readout_metal.sh`／旧
# `run_ab_splitk_metal.sh` と同型。判定規則は `docs/perf/metal-gemm-
# splitk-framework-compare-1517.md` を踏襲）。gemm 8 セル・train 2 セルは
# 同一 run ループ内で実行するが、出力先はタスク別ファイル
# （`OUT_*_GEMM`／`OUT_*_TRAIN`）へ分離する（#1517 PR #1531 是正を踏襲）。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
      if (( run_i % 2 == 1 )); then
        run_gemm off "$OUT_OFF_GEMM_TMP" "$size" "$mode"
        run_gemm on "$OUT_ON_GEMM_TMP" "$size" "$mode"
      else
        run_gemm on "$OUT_ON_GEMM_TMP" "$size" "$mode"
        run_gemm off "$OUT_OFF_GEMM_TMP" "$size" "$mode"
      fi
    done
  done
  for mode in "${MODES[@]}"; do
    if (( run_i % 2 == 1 )); then
      run_train off "$OUT_OFF_TRAIN_TMP" "$mode"
      run_train on "$OUT_ON_TRAIN_TMP" "$mode"
    else
      run_train on "$OUT_ON_TRAIN_TMP" "$mode"
      run_train off "$OUT_OFF_TRAIN_TMP" "$mode"
    fi
  done
  echo "== run $run_i/$AB_ROUNDS 完了時点の status =="
  pmset -g therm 2>&1 || true
  uptime 2>&1 || true
done

# `--phases`（train のみ・診断用・各腕 1 回。本体セルの「ちょうど 5 件」
# 契約を汚さないよう別ファイルへ出力する）。
for mode in "${MODES[@]}"; do
  verify_binary
  echo "== bench-fandhe train --phases mode=$mode metal-split-k=off =="
  if ! "./$EXE" --task train --device metal --mode "$mode" --metal-split-k off --phases --out "$OUT_OFF_PHASES_TMP" 2>err.tmp; then
    echo "train --phases mode=$mode metal-split-k=off : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in ${SKIP_TMP}。--phases は診断用のため ANY_FAILED には計上しない)"
  fi
  rm -f err.tmp
  echo "== bench-fandhe train --phases mode=$mode metal-split-k=on =="
  if ! "./$EXE" --task train --device metal --mode "$mode" --metal-split-k on --phases --out "$OUT_ON_PHASES_TMP" 2>err.tmp; then
    echo "train --phases mode=$mode metal-split-k=on : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in ${SKIP_TMP}。--phases は診断用のため ANY_FAILED には計上しない)"
  fi
  rm -f err.tmp
done

echo "== metal status (after loop) =="
pmset -g therm 2>&1 || true
uptime 2>&1 || true

kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
wait "$UPTIME_SAMPLER_PID" 2>/dev/null || true
trap restore_lock_trap EXIT

MV_FAILED=0
mv_checked() { # mv_checked <src> <dst>
  if ! mv -f "$1" "$2"; then
    echo "error: mv -f '$1' '$2' に失敗した" >&2
    MV_FAILED=$((MV_FAILED + 1))
  fi
}

if [[ "$ANY_FAILED" -eq 0 ]]; then
  mv_checked "$OUT_OFF_GEMM_TMP" "$OUT_OFF_GEMM"
  mv_checked "$OUT_ON_GEMM_TMP" "$OUT_ON_GEMM"
  mv_checked "$OUT_OFF_TRAIN_TMP" "$OUT_OFF_TRAIN"
  mv_checked "$OUT_ON_TRAIN_TMP" "$OUT_ON_TRAIN"
  mv_checked "$OUT_OFF_PHASES_TMP" "$OUT_OFF_PHASES"
  mv_checked "$OUT_ON_PHASES_TMP" "$OUT_ON_PHASES"
  mv_checked "$SKIP_TMP" "$SKIP"
  mv_checked "$MANIFEST_TMP" "$MANIFEST"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の mv が失敗した。新旧結果混在の可能性があるため、正規パスの内容を手動確認すること（fail-closed。security.md A08）。" >&2
    exit 1
  fi
  echo "done. gemm off/on results in $OUT_OFF_GEMM / $OUT_ON_GEMM ; train off/on results in $OUT_OFF_TRAIN / $OUT_ON_TRAIN ; phases (diagnostic) in $OUT_OFF_PHASES/$OUT_ON_PHASES ; failures (if any) in $SKIP ; manifest in $MANIFEST"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  mv_checked "$OUT_OFF_GEMM_TMP" "results/raw/results-m4max-splitk-ab-off-${LABEL}-gemm.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_ON_GEMM_TMP" "results/raw/results-m4max-splitk-ab-on-${LABEL}-gemm.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_OFF_TRAIN_TMP" "results/raw/results-m4max-splitk-ab-off-${LABEL}-train.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_ON_TRAIN_TMP" "results/raw/results-m4max-splitk-ab-on-${LABEL}-train.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_OFF_PHASES_TMP" "results/raw/results-m4max-splitk-ab-off-${LABEL}-phases.failed-${FAIL_TS}.jsonl"
  mv_checked "$OUT_ON_PHASES_TMP" "results/raw/results-m4max-splitk-ab-on-${LABEL}-phases.failed-${FAIL_TS}.jsonl"
  mv_checked "$SKIP_TMP" "results/raw/skipped-m4max-splitk-ab-${LABEL}.failed-${FAIL_TS}.log"
  mv_checked "$MANIFEST_TMP" "results/raw/manifest-m4max-splitk-ab-${LABEL}.failed-${FAIL_TS}.json"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の失敗結果退避 mv も失敗した（診断用データが一部欠落している可能性がある）。" >&2
  fi
  echo "FAILED: $ANY_FAILED run(s) failed; partial/unreliable data kept for diagnosis (${FAIL_TS}). $OUT_OFF_GEMM/$OUT_ON_GEMM/$OUT_OFF_TRAIN/$OUT_ON_TRAIN/$MANIFEST left untouched (fail-closed. security.md A08)." >&2
  exit 1
fi
