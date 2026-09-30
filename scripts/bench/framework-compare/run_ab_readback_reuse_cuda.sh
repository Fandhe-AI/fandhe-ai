#!/bin/bash
# イシュー #2108: CUDA readback 宛先ポリシー（`FANDHE_AI_CUDA_READBACK_DEST`。
# `crates/backend-cuda/src/readback_policy.rs`。既定 OFF）の pretouched（before・env 未設定）／
# pinned-reuse（after）を同一バイナリで run 単位に interleave 計測する A/B。
# `run_ab_readback_metal.sh`（Metal 版 #2112）の検証・専有ゲート・原子的出力・Cargo.lock 退避の型を踏襲する。
#
# 判定規則は実測前に固定した `docs/perf/logs/cuda-gemm-readback-reuse-2108/RULE.txt` が正
# （本スクリプトは計測と compare_gemm_ab.py の実行のみ。ADOPT／結線判断は RULE.txt と別 PR）。
# 判定セルは gemm cuda reuse N=1024/2048/4096。fresh は参考行としても計測しない（計測量抑制）。
#
# 順序制約（RULE.txt）: #2107 の `orchestrate.sh gb10` は本 PR より前のコミットで先に実行済みである
# こと。本スクリプトは HEAD（本 PR 以降）の facade で実行する。
#
# 呼び出し例（GB10 実機。実機セッション）:
#   AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
#     bash run_ab_readback_reuse_cuda.sh head-<short sha>-2108
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ一時ファイルを
# 正規パスへ排他的に公開し、失敗時は `.failed-<UTC>` へ排他退避する。同一 label の既存出力が
# あれば fail-closed で停止する。
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LABEL=${1:-}

# A03 インジェクション対策: ラベルはファイル名へ直接埋め込むため、
# 英数字・`._-` のみを許可する allowlist で検証する。
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label must match [A-Za-z0-9._-]+, e.g. head-abc1234-2108)" >&2
  echo "  env AB_PATCH_FACADE_PATH=<absolute path to HEAD's crates/facade> is required" >&2
  exit 1
fi

# `AB_PATCH_FACADE_PATH` の検証（A03・A08。`run_ab_gemm_metal.sh` と
# 同一方針）: 未設定・相対パス・`"`／`\` 混入・Cargo.toml 不在・crate 名
# 不一致のいずれかなら fail-closed で exit 1。
if [[ -z "${AB_PATCH_FACADE_PATH:-}" ]]; then
  echo "error: AB_PATCH_FACADE_PATH is required (absolute path to HEAD's crates/facade; issue #2108)" >&2
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

# Metal 版（#2112）由来の codex-review 指摘（PR #2456 P1）: 計測対象 facade は「本スクリプトを含む
# PR head の同一リポジトリの crates/facade」でなければならない（RULE.txt・AGENTS.md の
# 同一バイナリ・同一 HEAD 契約）。別所に checkout した facade が通ると、HEAD 計測として
# 公開される結果の出所が崩れる。パス（正規化後）と HEAD sha の双方を計測前に fail-closed で照合する。
SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
FACADE_HEAD_SHA="$(git -C "$AB_PATCH_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
EXPECTED_FACADE_PATH="$(cd "$SCRIPT_DIR/../../../crates/facade" 2>/dev/null && pwd -P || echo unknown)"
ACTUAL_FACADE_PATH="$(cd "$AB_PATCH_FACADE_PATH" 2>/dev/null && pwd -P || echo unknown)"
if [[ "$SCRIPT_REPO_HEAD_SHA" == "unknown" || "$FACADE_HEAD_SHA" == "unknown" \
  || "$SCRIPT_REPO_HEAD_SHA" != "$FACADE_HEAD_SHA" ]]; then
  echo "error: facade の HEAD sha が本スクリプトのリポジトリ HEAD と一致しない（script=${SCRIPT_REPO_HEAD_SHA} facade=${FACADE_HEAD_SHA}）。PR head の facade のみ計測できる" >&2
  exit 1
fi
if [[ "$EXPECTED_FACADE_PATH" == "unknown" || "$ACTUAL_FACADE_PATH" != "$EXPECTED_FACADE_PATH" ]]; then
  echo "error: AB_PATCH_FACADE_PATH が本リポジトリの crates/facade ではない（expected=${EXPECTED_FACADE_PATH} actual=${ACTUAL_FACADE_PATH}）" >&2
  exit 1
fi

# Metal 版（#2112）由来の codex-review 指摘（PR #2456）: RULE.txt は 5 round 固定で、
# 後段 compare_gemm_ab.py も各セル 5 件を要求する。5 以外は判定不能な出力を
# 生むため、計測開始前に fail-closed で拒否する（環境変数での上書きも 5 のみ許可）。
AB_ROUNDS=${AB_ROUNDS:-5}
if [[ "$AB_ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt の事前登録条件。got: $AB_ROUNDS）" >&2
  exit 1
fi

# before 腕は env 未設定が契約（RULE.txt）。呼び出し元シェルで設定済みだと before 腕が汚染される
# ため fail-closed で拒否する（値はエコーしない）。
if [[ -n "${FANDHE_AI_CUDA_READBACK_DEST+x}" ]]; then
  echo "error: FANDHE_AI_CUDA_READBACK_DEST が呼び出し元で設定済み（before 腕は未設定が契約）。unset して再実行する" >&2
  exit 1
fi

GEMM_SIZES=(1024 2048 4096)

OUT_BEFORE="results/raw/results-dgx-readback-reuse-ab-${LABEL}-before.jsonl"
OUT_AFTER="results/raw/results-dgx-readback-reuse-ab-${LABEL}-after.jsonl"
SKIP="results/raw/skipped-dgx-readback-reuse-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-dgx-readback-reuse-ab-${LABEL}.json"
UNDETERMINED="results/raw/readback-reuse-ab-${LABEL}.undetermined.txt"
if [[ -L results || -L results/raw ]]; then
  echo "error: results／results/raw がシンボリックリンク（出力先のすり替え防止のため拒否）" >&2
  exit 1
fi
mkdir -p results/raw

OUT_BEFORE_TMP="${OUT_BEFORE}.tmp"
OUT_AFTER_TMP="${OUT_AFTER}.tmp"
SKIP_TMP="${SKIP}.tmp"
MANIFEST_TMP="${MANIFEST}.tmp"
GATE_LOG="results/raw/gate-readback-reuse-ab-${LABEL}.log"
UPTIME_SAMPLER_LOG="results/raw/uptime-readback-reuse-ab-${LABEL}.log"

# Metal 版（#2112）由来の codex-review 指摘（PR #2456 P0）: 全出力（正規パス・一時ファイル・
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
for existing in "$OUT_BEFORE" "$OUT_AFTER" "$SKIP" "$MANIFEST" "$UNDETERMINED" \
  "$OUT_BEFORE_TMP" "$OUT_AFTER_TMP" "$SKIP_TMP" "$MANIFEST_TMP" \
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
AB_LOAD_GATE_MAX_LOAD1=${AB_LOAD_GATE_MAX_LOAD1:-1.0}
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

# Metal 版（#2112）由来の codex-review 指摘（PR #2456）: RULE.txt の正式系列は
# 「load1 < 1.0 かつ GPU 使用率 0% を 3 サンプル連続」で固定されている。exclusive（正式系列）で
# 閾値を環境変数から変えると判定前提が崩れるため、1.0 以外は fail-closed で拒否する。
# 別閾値で試す場合は record_only（非正式系列＝ADOPT 不可・undetermined）を使い、
# 実効閾値は manifest の load_gate_max_load1 へ残して系列を区別する。
if [[ "$AB_LOAD_GATE_MODE" == "exclusive" ]] \
  && ! awk -v t="$AB_LOAD_GATE_MAX_LOAD1" 'BEGIN{exit !(t + 0 == 1.0)}'; then
  echo "error: exclusive（正式系列）の専有ゲート閾値は 1.0 固定（RULE.txt）。AB_LOAD_GATE_MAX_LOAD1=${AB_LOAD_GATE_MAX_LOAD1} は不可。別閾値は AB_LOAD_GATE_MODE=record_only（非正式）で使う" >&2
  exit 1
fi

# codex-review 指摘（PR #2463）: RULE.txt の正式系列は試行条件（最大 10 試行・初回待機 60 秒
# ×1.5 倍）も事前固定である。環境変数で試行回数・待機を変えると事前固定を超えて待てるため、
# exclusive では固定値以外を fail-closed で拒否する。別値は record_only（非正式）で使う
# （record_only は待機・リトライ自体を行わないため値は参考記録のみ）。
if [[ "$AB_LOAD_GATE_MODE" == "exclusive" ]] \
  && [[ "$AB_LOAD_GATE_MAX_ATTEMPTS" != "10" || "$AB_LOAD_GATE_INITIAL_WAIT" != "60" ]]; then
  echo "error: exclusive（正式系列）の試行条件は AB_LOAD_GATE_MAX_ATTEMPTS=10・AB_LOAD_GATE_INITIAL_WAIT=60 固定（RULE.txt）。got: MAX_ATTEMPTS=${AB_LOAD_GATE_MAX_ATTEMPTS} INITIAL_WAIT=${AB_LOAD_GATE_INITIAL_WAIT}。別値は AB_LOAD_GATE_MODE=record_only（非正式）で使う" >&2
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
# GPU 使用率（%）を返す。取得失敗・非数値は空文字（呼び出し側が fail-closed に扱う）。
gpu_util_now() {
  local u
  u="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ "$u" =~ ^[0-9]+$ ]]; then printf '%s' "$u"; fi
}

load1_is_valid() {
  local v="$1"
  [[ -n "$v" ]] && awk -v l="$v" 'BEGIN{exit !(l ~ /^[0-9]+(\.[0-9]+)?$/ && l + 0 >= 0)}'
}

wait_for_exclusive_gate() {
  local attempt=0 wait_s="$AB_LOAD_GATE_INITIAL_WAIT" consecutive_ok=0 l1 gu
  local gate_log="$GATE_LOG"
  create_excl "$gate_log" || return 1
  while [[ "$attempt" -lt "$AB_LOAD_GATE_MAX_ATTEMPTS" ]]; do
    l1="$(load1_now)"
    gu="$(gpu_util_now)"
    if ! load1_is_valid "$l1" || [[ -z "$gu" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt load1_invalid_or_gpu_util_unavailable load1=${l1:-<empty>} gpu_util=${gu:-<empty>} threshold=$AB_LOAD_GATE_MAX_LOAD1 consecutive_ok=$consecutive_ok" | tee -a "$gate_log"
      consecutive_ok=0
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=$attempt load1=$l1 gpu_util=${gu}% threshold=$AB_LOAD_GATE_MAX_LOAD1 consecutive_ok=$consecutive_ok" | tee -a "$gate_log"
      if awk -v l="$l1" -v t="$AB_LOAD_GATE_MAX_LOAD1" 'BEGIN{exit !(l < t)}' && [[ "$gu" == "0" ]]; then
        consecutive_ok=$((consecutive_ok + 1))
        if [[ "$consecutive_ok" -ge 3 ]]; then
          echo "gate: ok (3 consecutive checks: load1 < threshold and GPU util 0%)" | tee -a "$gate_log"
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
    echo "reason=専有ゲート（load average < ${AB_LOAD_GATE_MAX_LOAD1} かつ GPU 使用率 0% を 3 回連続）が ${AB_LOAD_GATE_MAX_ATTEMPTS} 試行以内に成立しなかった"
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
# #2108 では record_only は非正式系列（RULE.txt: ADOPT 不可・undetermined）。
# load average の推移は gate-readback-reuse-ab-<label>.log／uptime-readback-reuse-ab-<label>.log
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
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) mode=record_only load1=${l1} gpu_util=$(gpu_util_now)%（専有ゲート要件なし。非正式系列＝ADOPT 不可。RULE.txt）" | tee -a "$gate_log"
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
  rm -f "${BUILD_ERR:-}" "${RUN_ERR:-}"
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

# Metal 版（#2112）由来の codex-review／Bugbot 指摘（PR #2456）: 結果一時ファイルは
# 入力検証・専有ゲート・ビルド・解決元検証がすべて通った後（計測開始直前）に
# 排他作成する。検証・ビルド失敗で残骸を残し同一 LABEL の再試行が
# reject_existing で塞がれるのを防ぐ。
create_excl "$OUT_BEFORE_TMP" || exit 1
create_excl "$OUT_AFTER_TMP" || exit 1
create_excl "$SKIP_TMP" || exit 1

BIN_SHA="$(sha256_of target/release/bench-fandhe)"
# manifest には絶対パスを載せない（RULE.txt の規則）。path 解決の記述は固定の相対表記へ正規化する。
BIN_SOURCE_REL="path:crates/facade"
echo "bench-fandhe sha256: $BIN_SHA (source: $BIN_SOURCE_REL)"

# manifest の書き込み失敗・内容不備は計測前に fail-closed で停止する（PR #2456 P1）。
# この時点で排他作成済みの一時ファイルと専有ゲートログは、計測が走っていないため掃除する
# （同一 LABEL の再試行を塞がない）。
abort_before_measure() { # abort_before_measure <理由>
  echo "error: $1" >&2
  rm -f "$OUT_BEFORE_TMP" "$OUT_AFTER_TMP" "$SKIP_TMP" "$MANIFEST_TMP"
  discard_pre_measure_gate_log
  exit 1
}
# manifest が 1 つの JSON オブジェクトで、公開前提の必須キーが非空であることを検証する。
validate_manifest() { # validate_manifest <path>
  python3 - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        m = json.load(f)
except Exception as e:
    sys.exit("manifest が JSON として不正: %s" % e)
if not isinstance(m, dict):
    sys.exit("manifest が JSON オブジェクトではない")
for k in ("label", "device", "script_repo_head_sha", "facade_head_sha", "bin_sha256",
          "bin_source", "readback_arms", "env", "gate_mode", "load_gate_max_load1", "recorded_at"):
    if k not in m or m[k] in ("", None, []):
        sys.exit("manifest の必須キーが欠落または空: %s" % k)
PY
}
if ! write_excl "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"cuda","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","facade_head_sha":"${FACADE_HEAD_SHA}","bin_sha256":"${BIN_SHA}","bin_source":"${BIN_SOURCE_REL}","readback_arms":["pretouched","pinned-reuse"],"env":"FANDHE_AI_CUDA_READBACK_DEST","gate_mode":"${AB_LOAD_GATE_MODE}","load_gate_max_load1":"${AB_LOAD_GATE_MAX_LOAD1}","recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
then
  abort_before_measure "manifest の書き込みに失敗した（不完全な manifest を公開しないため計測前に停止）: $MANIFEST_TMP"
fi
if ! validate_manifest "$MANIFEST_TMP"; then
  abort_before_measure "manifest の内容検証に失敗した（計測前に停止）: $MANIFEST_TMP"
fi
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

run() { # run <arm: before|after> <size>
  local arm=$1 size=$2 out_tmp
  verify_binary
  if [[ "$arm" == "before" ]]; then out_tmp="$OUT_BEFORE_TMP"; else out_tmp="$OUT_AFTER_TMP"; fi
  echo "== bench-fandhe gemm cuda reuse size=${size} arm=$arm =="
  local rc=0
  if [[ "$arm" == "before" ]]; then
    # before 腕: env 未設定（既定 PretouchedFresh）。上で呼び出し元の設定は拒否済み。
    ./target/release/bench-fandhe --task gemm --device cuda --size "$size" --mode reuse --out "$out_tmp" 2>"$RUN_ERR" || rc=$?
  else
    FANDHE_AI_CUDA_READBACK_DEST="pinned-reuse" ./target/release/bench-fandhe --task gemm --device cuda --size "$size" --mode reuse --out "$out_tmp" 2>"$RUN_ERR" || rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    # 生 stderr は絶対パス・内部ホスト名を含みうるため公開する失敗記録へは書かない（RULE.txt
    # 「生成物に含めない」。codex-review 指摘 PR #2463）。定型のエラー分類（終了コードのみ）を
    # 記録し、詳細は端末（非永続）へだけ出す。
    tail -20 "$RUN_ERR" >&2 || true
    echo "gemm cuda size=${size} arm=$arm : bench-fandhe failed (exit_code=${rc})" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  : > "$RUN_ERR"
}

run_cell() { # run_cell <round> <size>
  if (( $1 % 2 == 1 )); then
    run before "$2"; run after "$2"
  else
    run after "$2"; run before "$2"
  fi
}

echo "== cuda status (before loop) =="
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,utilization.gpu --format=csv 2>&1 || true
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
  rm -f "${BUILD_ERR:-}" "${RUN_ERR:-}"
  kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
  if ! restore_lock; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
  fi
  exit "$code"
}
trap restore_lock_and_kill_sampler_trap EXIT

# run 単位で before/after を交互起動する。奇数 round: before→after・偶数 round:
# after→before（起動順序の系統誤差を均す）。プロセスは run ごとに独立起動。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${GEMM_SIZES[@]}"; do
    run_cell "$run_i" "$size"
  done
  echo "== round $run_i/$AB_ROUNDS 完了時点の status =="
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader 2>&1 || true
  uptime 2>&1 || true
done

kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
wait "$UPTIME_SAMPLER_PID" 2>/dev/null || true
trap restore_lock_trap EXIT

echo "== cuda status (after loop) =="
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,utilization.gpu --format=csv,noheader 2>&1 || true
uptime 2>&1 || true

# 一時ファイルを最終パスへ排他的に公開する。`ln`（ハードリンク）は宛先が
# 既存（dangling シンボリックリンク含む）なら EEXIST で失敗するため、計測中に
# 作られた同一 label の結果や既存の退避先を上書きしない（`mv -f` は無条件に
# 置換するため使わない。PR #2456 P0）。
# PR #2456 P1: 4 ファイルは「全部公開できたときだけ」成立させる。途中の ln 失敗時は
# この実行が作成した正規パスだけを巻き戻し（既存ファイルは触らない）、結果は *.tmp に残す。
# 成功時のみ全 ln 完了後に一時ファイルを削除する。manifest は最後に公開し完了印とする。
publish_set() { # publish_set <src1> <dst1> [<src2> <dst2> ...]
  local created=() srcs=() src dst d
  # 公開前に全宛先の不在を一括確認する。
  local i
  for ((i = 1; i < $#; i += 2)); do
    dst="${@:i+1:1}"
    if [[ -e "$dst" || -L "$dst" ]]; then
      echo "error: 計測中に同一 label の出力が作られた（上書き禁止）: $dst。結果は *.tmp に残す" >&2
      return 1
    fi
  done
  while [[ $# -ge 2 ]]; do
    src=$1 dst=$2
    shift 2
    if ln "$src" "$dst"; then
      created+=("$dst")
      srcs+=("$src")
    else
      echo "error: '$dst' へ排他的に公開できない。この実行が公開済みの正規パスを巻き戻す。結果は '$src' などの *.tmp に残す" >&2
      for d in ${created[@]+"${created[@]}"}; do rm -f "$d"; done
      return 1
    fi
  done
  rm -f "${srcs[@]}"
}

if [[ "$ANY_FAILED" -eq 0 ]]; then
  if ! validate_manifest "$MANIFEST_TMP"; then
    echo "error: 公開前の manifest 検証に失敗した。結果は *.tmp に残す" >&2
    exit 1
  fi
  if ! publish_set "$OUT_BEFORE_TMP" "$OUT_BEFORE" "$OUT_AFTER_TMP" "$OUT_AFTER" \
    "$SKIP_TMP" "$SKIP" "$MANIFEST_TMP" "$MANIFEST"; then
    echo "error: 公開に失敗した（正規パスは巻き戻し済み。部分公開なし）" >&2
    exit 1
  fi
  echo "done. results in $OUT_BEFORE / $OUT_AFTER ; manifest in $MANIFEST"
  # 判定（RULE.txt）: 5 run 中央値の ratio と checksum 完全一致（fail-closed）。ADOPT 判断は
  # RULE.txt の前提ゲート（#2107 の帰属判定）と合わせて人間が行う。
  REPORT="results/raw/compare-readback-reuse-ab-${LABEL}.md"
  if [[ -e "$REPORT" || -L "$REPORT" ]]; then
    echo "error: 判定レポートの出力先が既に存在する（上書き禁止）: $REPORT" >&2
    exit 1
  fi
  python3 compare_gemm_ab.py "$OUT_BEFORE" "$OUT_AFTER" --device cuda --sizes large --modes reuse --threshold 1.00 \
    --require-checksum-exact | write_excl "$REPORT"
  # 比較器と書き込み先の両方の終了状態を検査する（片方だけ見ると、write_excl 失敗で
  # レポートが無いのに exit 0 になる。fail-closed。security.md A08）。
  pipe_rc=("${PIPESTATUS[@]}")
  cmp_rc=${pipe_rc[0]}
  write_rc=${pipe_rc[1]}
  echo "compare_gemm_ab.py exit=$cmp_rc; report write exit=$write_rc; report: $REPORT"
  if [[ "$write_rc" -ne 0 ]]; then
    echo "error: 判定レポートの書き込みに失敗した（レポートなし。fail-closed）" >&2
    exit 1
  fi
  # record_only は非正式系列（RULE.txt §判定: ADOPT 不可・undetermined）。比較値は上のレポートへ
  # 記録したうえで、比較器の結果に関わらず最終状態を undetermined として明示し非 0 終了する。
  if [[ "$AB_LOAD_GATE_MODE" == "record_only" ]]; then
    {
      echo "verdict=undetermined"
      echo "reason=AB_LOAD_GATE_MODE=record_only は非正式系列（専有ゲート要件なし。RULE.txt: ADOPT 不可）。比較器 exit=${cmp_rc} は参考値"
      echo "report=$REPORT"
      date -u +%Y-%m-%dT%H:%M:%SZ
    } | write_excl "$UNDETERMINED" || exit 1
    echo "undetermined: ${UNDETERMINED}（record_only。比較値は ${REPORT} に記録済み）" >&2
    exit 1
  fi
  exit "$cmp_rc"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  if ! publish_set \
    "$OUT_BEFORE_TMP" "results/raw/results-dgx-readback-reuse-ab-${LABEL}-before.failed-${FAIL_TS}.jsonl" \
    "$OUT_AFTER_TMP" "results/raw/results-dgx-readback-reuse-ab-${LABEL}-after.failed-${FAIL_TS}.jsonl" \
    "$SKIP_TMP" "results/raw/skipped-dgx-readback-reuse-ab-${LABEL}.failed-${FAIL_TS}.log" \
    "$MANIFEST_TMP" "results/raw/manifest-dgx-readback-reuse-ab-${LABEL}.failed-${FAIL_TS}.json"; then
    echo "error: 失敗退避の公開にも失敗した。部分データは *.tmp に残る" >&2
  fi
  echo "FAILED: $ANY_FAILED run(s) failed; partial data kept (${FAIL_TS}). 正規パスは未変更（fail-closed。security.md A08）。" >&2
  exit 1
fi
