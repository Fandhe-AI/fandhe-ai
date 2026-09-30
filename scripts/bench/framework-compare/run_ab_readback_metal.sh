#!/bin/bash
# イシュー #2112: Metal readback 宛先ポリシー（`FANDHE_AI_METAL_READBACK_DEST`。
# `crates/backend-metal/src/readback_policy.rs`。既定 OFF）の fresh／parallel を
# 同一バイナリで run 単位に interleave 計測する A/B。`run_ab_readout_metal.sh`
# （#1477）の検証・専有ゲート・原子的出力・Cargo.lock 退避の型を踏襲する。
#
# 判定規則は実測前に固定した `docs/perf/logs/metal-reuse-readback-2112/RULE.txt`
# が正（本スクリプトは判定しない。後段の compare_gemm_ab.py 呼び出しは同 README）。
# 判定セルは gemm reuse N=1024・N=4096（既定 readout=legacy）と infer reuse。
# borrowed readout は #1520 の REJECT 再実行に当たるため計測しない。
#
# 呼び出し例（M4 Max 実機。Mac セッション）:
#   AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
#     bash run_ab_readback_metal.sh head-<short sha>-2112
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ一時ファイルを
# 正規パスへ排他的に（既存を置換せず）公開し、失敗時は `.failed-<UTC>` へ排他退避する。同一 label の既存
# 出力があれば fail-closed で停止する（上書き・run の差し替えをしない）。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. head-abc1234-1477)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to HEAD's crates/facade> is required" >&2
  exit 1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08。`run_ab_gemm_metal.sh` と
# 同一方針）: 未設定・相対パス・`"`／`\` 混入・Cargo.toml 不在・crate 名
# 不一致のいずれかなら fail-closed で exit 1。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to HEAD's crates/facade; issue #1477)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" != /* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must be an absolute path (got: $AB_PATCH_FACADE_PATH)" >&2
  exit 1
fi
if [[ "$AB_PATCH_FACADE_PATH" == *'"'* || "$AB_PATCH_FACADE_PATH" == *'\'* ]]; then
  echo "error: AB_PATCH_FACADE_PATH must not contain '\"' or '\\'" >&2
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

# イシュー #2112・codex-review 指摘（PR #2456）: RULE.txt は 5 round 固定で、
# 後段 compare_gemm_ab.py も各セル 5 件を要求する。5 以外は判定不能な出力を
# 生むため、計測開始前に fail-closed で拒否する（環境変数での上書きも 5 のみ許可）。
AB_ROUNDS=${AB_ROUNDS:-5}
if [[ "$AB_ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt の事前登録条件。got: $AB_ROUNDS）" >&2
  exit 1
fi

GEMM_SIZES=(1024 4096)

OUT_FRESH="results/raw/results-m4max-readback-ab-${LABEL}-fresh.jsonl"
OUT_PARALLEL="results/raw/results-m4max-readback-ab-${LABEL}-parallel.jsonl"
SKIP="results/raw/skipped-m4max-readback-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-m4max-readback-ab-${LABEL}.json"
UNDETERMINED="results/raw/readback-ab-${LABEL}.undetermined.txt"
if [[ -L results || -L results/raw ]]; then
  echo "error: results／results/raw がシンボリックリンク（出力先のすり替え防止のため拒否）" >&2
  exit 1
fi
mkdir -p results/raw

OUT_FRESH_TMP="${OUT_FRESH}.tmp"
OUT_PARALLEL_TMP="${OUT_PARALLEL}.tmp"
SKIP_TMP="${SKIP}.tmp"
MANIFEST_TMP="${MANIFEST}.tmp"
GATE_LOG="results/raw/gate-readback-ab-${LABEL}.log"
UPTIME_SAMPLER_LOG="results/raw/uptime-readback-ab-${LABEL}.log"

# イシュー #2112・codex-review 指摘（PR #2456 P0）: 全出力（正規パス・一時ファイル・
# 各ログ・undetermined マーカー）は既存パス・シンボリックリンク（dangling 含む）を
# 拒否し、排他作成（noclobber = O_EXCL）でのみ作る。`: >` は既存リンクをたどって
# リンク先を切り詰めうるため使わない。上書き禁止・run 差し替え禁止（RULE.txt）。
reject_existing() { # reject_existing <path>
  if [[ -e "$1" || -L "$1" ]]; then
    echo "error: 既存の出力またはシンボリックリンクがある（上書き禁止・run 差し替え禁止。RULE.txt）: $1" >&2
    return 1
  fi
}
create_excl() { # create_excl <path>: 空ファイルを排他作成
  reject_existing "$1" || return 1
  ( set -C; : > "$1" ) || { echo "error: 排他作成に失敗した: $1" >&2; return 1; }
}
write_excl() { # write_excl <path>: stdin を排他作成したファイルへ書く
  reject_existing "$1" || return 1
  ( set -C; cat > "$1" ) || { echo "error: 排他書き込みに失敗した: $1" >&2; return 1; }
}

# 計測開始前に全出力先を一括検査する（途中失敗で部分出力を残さない）。
for existing in "$OUT_FRESH" "$OUT_PARALLEL" "$SKIP" "$MANIFEST" "$UNDETERMINED" \
  "$OUT_FRESH_TMP" "$OUT_PARALLEL_TMP" "$SKIP_TMP" "$MANIFEST_TMP" \
  "$GATE_LOG" "$UPTIME_SAMPLER_LOG"; do
  reject_existing "$existing" || exit 1
done
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
# （path か registry か）を抽出する（`run_ab_gemm_metal.sh` と同型の
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

# ------------------------------------------------------------------
# 専有ゲート（イシュー #1477 判定規則 §2。#1309 `wait_gate.sh` と同一の
# パラメータ）: 1 分 load average < AB_LOAD_GATE_MAX_LOAD1 を 30 秒間隔で
# 2 回連続確認できるまで待機する（60 秒開始 × 1.5 倍・最大 10 回）。
# 不合格のまま試行回数を使い切った場合は計測を開始せず undetermined
# マーカーを書いて非ゼロ終了する（再試行ループで待たない。判定規則
# §2「undetermined」）。
# ------------------------------------------------------------------
AB_LOAD_GATE_MAX_LOAD1=${AB_LOAD_GATE_MAX_LOAD1:-4.0}
AB_LOAD_GATE_MAX_ATTEMPTS=${AB_LOAD_GATE_MAX_ATTEMPTS:-10}
AB_LOAD_GATE_INITIAL_WAIT=${AB_LOAD_GATE_INITIAL_WAIT:-60}

# イシュー #1520（ルート #1519）: ユーザー指示「Metal は専有ゲートを
# 要件にしない」を受け、専有ゲートを「必須」から「既定は現行どおり必須・
# record_only で明示的に要件を外せる」opt-out 方式にする。既定
# `exclusive` は #1477 までの挙動（本節冒頭のゲート）を一切変えない
# （後方互換。将来また専有環境で再計測する場合はそのまま使える）。
# `record_only` は待機・リトライを一切行わず、現在の load average を
# 1 行記録してから直ちに計測へ進む。「計測中の load average 推移」は
# 既存の UPTIME_SAMPLER_LOG（30 秒間隔）が無条件に取得するため
# record_only でも失われない。
AB_LOAD_GATE_MODE=${AB_LOAD_GATE_MODE:-exclusive}
if [[ "$AB_LOAD_GATE_MODE" != "exclusive" && "$AB_LOAD_GATE_MODE" != "record_only" ]]; then
  echo "error: AB_LOAD_GATE_MODE must be 'exclusive' or 'record_only' (got: $AB_LOAD_GATE_MODE)" >&2
  exit 1
fi

# `AB_LOAD_GATE_MAX_LOAD1`（環境変数から利用者が上書き可能な専有ゲート
# 閾値）が有限の正数であることを事前検証する（codex-review 指摘・PR
# #1493 スレッド 2: 未検証のまま awk の数値コンテキストへ渡すと不正値
# 〈空文字・非数値・負数〉が暗黙に 0 または文字列比較として扱われ、
# `l1 < t` が常に真になり load1=14.64 のような高負荷でも専有ゲートが
# 誤って通過しうる。`load1_is_valid` は 0 を許容するため使い回さず、
# 閾値専用に厳密な正数〈> 0〉検証を行う）。不正値では即座に fail-closed
# で終了する（再試行ループへは入らない）。
if ! awk -v t="$AB_LOAD_GATE_MAX_LOAD1" 'BEGIN{exit !(t ~ /^[0-9]+(\.[0-9]+)?$/ && t + 0 > 0)}'; then
  echo "error: AB_LOAD_GATE_MAX_LOAD1 が有限の正数ではありません: ${AB_LOAD_GATE_MAX_LOAD1}" >&2
  exit 1
fi

# イシュー #2112・codex-review 指摘（PR #2456）: RULE.txt の正式系列は
# 「load1 < 4.0 を 2 回連続」で固定されている。exclusive（正式系列）で閾値を
# 環境変数から変えると判定前提が崩れるため、4.0 以外は fail-closed で拒否する。
# 別閾値で試す場合は record_only（非正式系列＝ADOPT 不可・undetermined）を使い、
# 実効閾値は manifest の load_gate_max_load1 へ残して系列を区別する。
if [[ "$AB_LOAD_GATE_MODE" == "exclusive" ]] \
  && ! awk -v t="$AB_LOAD_GATE_MAX_LOAD1" 'BEGIN{exit !(t + 0 == 4.0)}'; then
  echo "error: exclusive（正式系列）の専有ゲート閾値は 4.0 固定（RULE.txt）。AB_LOAD_GATE_MAX_LOAD1=${AB_LOAD_GATE_MAX_LOAD1} は不可。別閾値は AB_LOAD_GATE_MODE=record_only（非正式）で使う" >&2
  exit 1
fi

load1_now() {
  # `uptime` の失敗（コマンド自体の異常終了）／出力形式の不一致は
  # 空文字を返す（呼び出し側 `wait_for_exclusive_gate` が非数値・空文字
  # を「取得失敗」として fail-closed に扱う契約。codex-review 指摘・
  # PR #1493 P1: 空文字を awk の数値コンテキストへそのまま渡すと 0 扱い
  # されて閾値未満と誤判定され、専有ゲートが誤通過しうる）。
  local raw parsed
  if ! raw="$(uptime 2>/dev/null)"; then
    return 0
  fi
  parsed="$(printf '%s\n' "$raw" | sed -E 's/.*load average[s]?: ([0-9.]+).*/\1/')"
  # sed が置換に成功しなかった場合（`uptime` の出力形式が想定外）、
  # `-n` を付けていないため置換前の行がそのまま出力される。それを
  # そのまま数値として扱わないよう、`load1_is_valid` 相当の判定は
  # 呼び出し側で行う（ここでは非数値も含めそのまま返す）。
  printf '%s' "$parsed"
}

# `load1_now` が返した文字列が非負の有限数値であることを検証する
# （codex-review 指摘・PR #1493 P1）。`awk` の数値コンテキストは非数値
# 文字列を暗黙に 0 として扱うため、専有ゲートの比較へ渡す前に本関数で
# 明示的に弾く。
load1_is_valid() {
  local v="$1"
  [[ -n "$v" ]] && awk -v l="$v" 'BEGIN{exit !(l ~ /^[0-9]+(\.[0-9]+)?$/ && l + 0 >= 0)}'
}

wait_for_exclusive_gate() {
  local attempt=0 wait_s="$AB_LOAD_GATE_INITIAL_WAIT" consecutive_ok=0 l1
  local gate_log="$GATE_LOG"
  create_excl "$gate_log" || return 1
  while [[ "$attempt" -lt "$AB_LOAD_GATE_MAX_ATTEMPTS" ]]; do
    l1="$(load1_now)"
    if ! load1_is_valid "$l1"; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt load1_invalid=${l1:-<empty>} threshold=$AB_LOAD_GATE_MAX_LOAD1 consecutive_ok=$consecutive_ok" | tee -a "$gate_log"
      consecutive_ok=0
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt load1=$l1 threshold=$AB_LOAD_GATE_MAX_LOAD1 consecutive_ok=$consecutive_ok" | tee -a "$gate_log"
      if awk -v l="$l1" -v t="$AB_LOAD_GATE_MAX_LOAD1" 'BEGIN{exit !(l < t)}'; then
        consecutive_ok=$((consecutive_ok + 1))
        if [[ "$consecutive_ok" -ge 2 ]]; then
          echo "gate: ok (2 consecutive checks under threshold)" | tee -a "$gate_log"
          return 0
        fi
      else
        consecutive_ok=0
      fi
    fi
    attempt=$((attempt + 1))
    if [[ "$attempt" -ge "$AB_LOAD_GATE_MAX_ATTEMPTS" ]]; then
      break
    fi
    sleep 30
    # 2 回連続確認の 2 回目は 30 秒後に取るため、試行間バックオフ
    # （60 秒開始 x 1.5 倍）は 2 回連続確認が崩れた場合にのみ適用する。
    if [[ "$consecutive_ok" -eq 0 ]]; then
      sleep "$wait_s"
      wait_s=$(awk -v w="$wait_s" 'BEGIN{printf "%.0f", w * 1.5}')
    fi
  done
  {
    echo "verdict=undetermined"
    echo "reason=専有ゲート（load average < ${AB_LOAD_GATE_MAX_LOAD1} を 2 回連続）が ${AB_LOAD_GATE_MAX_ATTEMPTS} 試行以内に成立しなかった"
    echo "gate_log=$gate_log"
    date -u +%Y-%m-%dT%H:%M:%SZ
  } | write_excl "$UNDETERMINED"
  echo "undetermined: ${UNDETERMINED}（判定規則 §2 に従い再試行せず終了する）" >&2
  return 1
}

# 他セッションの計測プロセス並走を確認する（専有ゲートの補助チェック。
# load average だけでは検出できない、まだ CPU 負荷が立ち上がっていない
# 起動直後の並走プロセスを検出する）。
if pgrep -f 'bench-fandhe|bench-candle' >/dev/null 2>&1; then
  echo "warning: 他の bench-fandhe/bench-candle プロセスが実行中の可能性がある（pgrep 検出）" >&2
fi

# `AB_LOAD_GATE_MODE=record_only`（イシュー #1520）: 専有ゲートを要件に
# せず、現在の load average を 1 行記録するだけで即座に計測を開始する。
# #2112 では record_only は非正式系列（RULE.txt: ADOPT 不可・undetermined）。
# load average の推移は gate-readback-ab-<label>.log／uptime-readback-ab-<label>.log
# へ記録する（#1520 と同じ記録先）。
record_only_gate_note() {
  local gate_log="$GATE_LOG"
  local l1
  l1="$(load1_now)"
  create_excl "$gate_log" || return 1
  # `wait_for_exclusive_gate` と同じく `load1_is_valid` で数値妥当性を
  # 検証してから記録する（codex-review 指摘・PR #1493 P1 と同型の懸念:
  # `uptime` の出力形式が想定外の場合 `load1_now` が非数値をそのまま
  # 返しうるため、無検証でログへ書くと後続の集計・判定を誤誘導しうる）。
  if ! load1_is_valid "$l1"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1_invalid=${l1:-<empty>}（専有ゲート要件なし。イシュー #1520・ルート #1519）" | tee -a "$gate_log"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1=${l1}（専有ゲート要件なし。イシュー #1520・ルート #1519）" | tee -a "$gate_log"
  fi
  return 0
}

run_gate() {
  if [[ "$AB_LOAD_GATE_MODE" == "record_only" ]]; then
    record_only_gate_note
  else
    wait_for_exclusive_gate
  fi
}

if ! run_gate; then
  exit 1
fi

# `Cargo.lock` を退避し、異常終了含め終了時に必ず復元する（trap。
# `run_ab_gemm_metal.sh` と同一方針）。
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

# 計測前（ビルド・解決元検証）の失敗では、この実行が排他作成した専有ゲートログ
# だけを掃除する（計測が走っていないのに同一 LABEL の再試行が塞がれない。
# 専有ゲート不成立の undetermined マーカーとそのログは判定根拠のため残す）。
discard_pre_measure_gate_log() {
  rm -f "$GATE_LOG"
}

# stderr の退避先は mktemp（排他作成・推測不能名）。固定名のリダイレクトは
# 既存リンクをたどるため使わない（PR #2456 P0 と同型）。
BUILD_ERR="$(mktemp)"
RUN_ERR="$(mktemp)"
echo "== build bench-fandhe (HEAD path patch) =="
if ! cargo build --release -p bench-fandhe --config "$PATCH_CONFIG" 2>"$BUILD_ERR"; then
  tail -40 "$BUILD_ERR"
  echo "bench-fandhe BUILD FAILED: $(tail -3 "$BUILD_ERR" | tr '\n' ' ')" >&2
  rm -f "$BUILD_ERR"
  discard_pre_measure_gate_log
  exit 1
fi
rm -f "$BUILD_ERR"

SOURCE_DESC="$(fandhe_ai_source_desc --config "$PATCH_CONFIG" || true)"
if [[ "$SOURCE_DESC" != "path:${AB_PATCH_FACADE_PATH}" ]]; then
  echo "error: fandhe-ai が期待した path 解決ではない (expected=path:${AB_PATCH_FACADE_PATH} actual=${SOURCE_DESC:-<取得失敗>})" >&2
  discard_pre_measure_gate_log
  exit 1
fi

# イシュー #2112・codex-review／Bugbot 指摘（PR #2456）: 結果一時ファイルは
# 入力検証・専有ゲート・ビルド・解決元検証がすべて通った後（計測開始直前）に
# 排他作成する。検証・ビルド失敗で残骸を残し同一 LABEL の再試行が
# reject_existing で塞がれるのを防ぐ。
create_excl "$OUT_FRESH_TMP" || exit 1
create_excl "$OUT_PARALLEL_TMP" || exit 1
create_excl "$SKIP_TMP" || exit 1

BIN_SHA="$(sha256_of target/release/bench-fandhe)"
echo "bench-fandhe sha256: $BIN_SHA (source: $SOURCE_DESC)"

SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
FACADE_HEAD_SHA="$(git -C "$AB_PATCH_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
write_excl "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"metal","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","facade_head_sha":"${FACADE_HEAD_SHA}","bin_sha256":"${BIN_SHA}","bin_source":"${SOURCE_DESC}","readback_arms":["fresh","parallel"],"env":"FANDHE_AI_METAL_READBACK_DEST","gate_mode":"${AB_LOAD_GATE_MODE}","load_gate_max_load1":"${AB_LOAD_GATE_MAX_LOAD1}","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
echo "== manifest（一時ファイル）記録: $MANIFEST_TMP =="
cat "$MANIFEST_TMP"

verify_binary() {
  local now
  now="$(sha256_of target/release/bench-fandhe)"
  if [[ "$now" != "$BIN_SHA" ]]; then
    echo "error: bench-fandhe のバイナリが計測中に変化した（sha256 不一致）" >&2
    exit 1
  fi
}

run() { # run <arm: fresh|parallel> <task: gemm|infer> <size(gemm のみ)>
  local arm=$1 task=$2 size=${3:-} out_tmp
  verify_binary
  if [[ "$arm" == "fresh" ]]; then out_tmp="$OUT_FRESH_TMP"; else out_tmp="$OUT_PARALLEL_TMP"; fi
  echo "== bench-fandhe $task metal reuse size=${size:-n/a} readback=$arm =="
  local rc=0
  if [[ "$task" == "gemm" ]]; then
    FANDHE_AI_METAL_READBACK_DEST="$arm" ./target/release/bench-fandhe --task gemm --device metal --size "$size" --mode reuse --out "$out_tmp" 2>"$RUN_ERR" || rc=$?
  else
    FANDHE_AI_METAL_READBACK_DEST="$arm" ./target/release/bench-fandhe --task infer --device metal --mode reuse --out "$out_tmp" 2>"$RUN_ERR" || rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "$task metal size=${size:-n/a} readback=$arm : $(cat "$RUN_ERR")" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  : > "$RUN_ERR"
}

run_cell() { # run_cell <round> <task> <size>
  if (( $1 % 2 == 1 )); then
    run fresh "$2" "${3:-}"; run parallel "$2" "${3:-}"
  else
    run parallel "$2" "${3:-}"; run fresh "$2" "${3:-}"
  fi
}

echo "== metal status (before loop) =="
sysctl -n machdep.cpu.brand_string 2>&1 || true
pmset -g therm 2>&1 || true
uptime 2>&1 || true

create_excl "$UPTIME_SAMPLER_LOG" || exit 1
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
# 事故を防ぐ）。
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

# run 単位で fresh/parallel を交互起動する。奇数 round: fresh→parallel・偶数 round:
# parallel→fresh（起動順序の系統誤差を均す）。プロセスは run ごとに独立起動。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${GEMM_SIZES[@]}"; do
    run_cell "$run_i" gemm "$size"
  done
  run_cell "$run_i" infer
  echo "== round $run_i/$AB_ROUNDS 完了時点の status =="
  pmset -g therm 2>&1 || true
  uptime 2>&1 || true
done

kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
wait "$UPTIME_SAMPLER_PID" 2>/dev/null || true
trap restore_lock_trap EXIT

echo "== metal status (after loop) =="
pmset -g therm 2>&1 || true
uptime 2>&1 || true

# 一時ファイルを最終パスへ排他的に公開する。`ln`（ハードリンク）は宛先が
# 既存（dangling シンボリックリンク含む）なら EEXIST で失敗するため、計測中に
# 作られた同一 label の結果や既存の退避先を上書きしない（`mv -f` は無条件に
# 置換するため使わない。PR #2456 P0）。成功時のみ一時ファイルを削除する。
MV_FAILED=0
publish_excl() { # publish_excl <src> <dst>
  if [[ -e "$2" || -L "$2" ]] || ! ln "$1" "$2"; then
    echo "error: '$2' へ排他的に公開できない（既存あり。上書きしない）。結果は '$1' に残す" >&2
    MV_FAILED=$((MV_FAILED + 1))
    return 1
  fi
  rm -f "$1"
}

if [[ "$ANY_FAILED" -eq 0 ]]; then
  # 公開前に全宛先の不在を一括確認する（一部だけ公開された状態を避ける）。
  for dst in "$OUT_FRESH" "$OUT_PARALLEL" "$SKIP" "$MANIFEST"; do
    if [[ -e "$dst" || -L "$dst" ]]; then
      echo "error: 計測中に同一 label の出力が作られた（上書き禁止）: $dst。結果は *.tmp に残す" >&2
      exit 1
    fi
  done
  publish_excl "$OUT_FRESH_TMP" "$OUT_FRESH"
  publish_excl "$OUT_PARALLEL_TMP" "$OUT_PARALLEL"
  publish_excl "$SKIP_TMP" "$SKIP"
  publish_excl "$MANIFEST_TMP" "$MANIFEST"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の公開が失敗した（正規パスの反映が不完全な可能性がある）" >&2
    exit 1
  fi
  echo "done. results in $OUT_FRESH / $OUT_PARALLEL ; manifest in $MANIFEST"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  publish_excl "$OUT_FRESH_TMP" "results/raw/results-m4max-readback-ab-${LABEL}-fresh.failed-${FAIL_TS}.jsonl"
  publish_excl "$OUT_PARALLEL_TMP" "results/raw/results-m4max-readback-ab-${LABEL}-parallel.failed-${FAIL_TS}.jsonl"
  publish_excl "$SKIP_TMP" "results/raw/skipped-m4max-readback-ab-${LABEL}.failed-${FAIL_TS}.log"
  publish_excl "$MANIFEST_TMP" "results/raw/manifest-m4max-readback-ab-${LABEL}.failed-${FAIL_TS}.json"
  echo "FAILED: $ANY_FAILED run(s) failed; partial data kept (${FAIL_TS}). 正規パスは未変更（fail-closed。security.md A08）。" >&2
  exit 1
fi
