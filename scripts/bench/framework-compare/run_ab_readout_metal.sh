#!/bin/bash
# イシュー #1477: Metal 借用ビュー readout（legacy フォールバック）の
# legacy/borrowed override（`--readout`）を同一バイナリで run 単位に
# interleave 計測する。
#
# `docs/perf/metal-gemm-candle-gate-remeasurement.md` §13.5 の教訓（before
# → after の連続実行では負荷変動と切替効果を分離できない）を踏まえ、
# `run_ab_managed_cuda.sh`（同一バイナリのフラグ切替専用 A/B）の設計を
# 引き継ぎつつ、`run_ab_gemm_metal.sh` の run 単位順序反転（奇数 run:
# legacy→borrowed・偶数 run: borrowed→legacy）・fail-closed の一時ファイル
# →原子的 mv・`pmset -g therm`／`uptime` 記録・専有ゲート（load average）
# を合成したもの。
#
# `--readout` override（`bench-fandhe` 側実装）は `Var::host_view` 等
# （#1335。crates.io 公開版 `fandhe-ai =0.8.0` に #1487 でピン更新済みの
# ため収録済み）自体は要求せず（`readout_uses_borrowed_view` の device
# 文字列 1 個の runtime 分岐のみ）、`AB_PATCH_FACADE_PATH` は本スクリプト
# の主目的（HEAD ソースでの計測）のため必須とする（`run_ab_gemm_metal.sh`
# の after 腕と同型。registry pin 限定の before 腕は本スクリプトには無い
# — 両腕〈legacy／borrowed〉は同一バイナリの `--readout` 値切替であり、
# 「結線前後」の 2 バイナリ比較ではないため）。
#
# 呼び出し例（M4 Max 実機。ユーザー承認・別セッション。低負荷時間帯に
# `uptime` を確認してから実行する）:
#   AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
#     bash run_ab_readout_metal.sh head-<short sha>-1477
#
# 出力は「失敗を捏造しない」方針（security.md A08）: 全 run 成功時のみ
# 一時ファイルを正規パスへ原子的に反映する。1 件でも失敗すれば正規パスは
# 変更せず、不完全な結果は `.failed-<UTC>` へ退避する。
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

AB_ROUNDS=${AB_ROUNDS:-5}
if [[ ! "$AB_ROUNDS" =~ ^[0-9]+$ || "$AB_ROUNDS" -lt 1 ]]; then
  echo "error: AB_ROUNDS must be a positive integer (got: $AB_ROUNDS)" >&2
  exit 1
fi

SIZES=(1024 2048 4096)
MODES=(fresh reuse)

OUT="results/raw/results-m4max-readout-ab-${LABEL}.jsonl"
SKIP="results/raw/skipped-m4max-readout-ab-${LABEL}.log"
MANIFEST="results/raw/manifest-m4max-readout-ab-${LABEL}.json"
UNDETERMINED="results/raw/readout-ab-${LABEL}.undetermined.txt"
mkdir -p results/raw

OUT_TMP="${OUT}.tmp"
SKIP_TMP="${SKIP}.tmp"
: > "$OUT_TMP"
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
  local gate_log="results/raw/gate-readout-ab-${LABEL}.log"
  : > "$gate_log"
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
  } > "$UNDETERMINED"
  echo "undetermined: $UNDETERMINED（判定規則 §2 に従い再試行せず終了する）" >&2
  return 1
}

# 他セッションの計測プロセス並走を確認する（専有ゲートの補助チェック。
# load average だけでは検出できない、まだ CPU 負荷が立ち上がっていない
# 起動直後の並走プロセスを検出する）。
if pgrep -f 'bench-fandhe|bench-candle' >/dev/null 2>&1; then
  echo "warning: 他の bench-fandhe/bench-candle プロセスが実行中の可能性がある（pgrep 検出）" >&2
fi

if ! wait_for_exclusive_gate; then
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

echo "== build bench-fandhe (HEAD path patch) =="
if ! cargo build --release -p bench-fandhe --config "$PATCH_CONFIG" 2>build-err.tmp; then
  tail -40 build-err.tmp
  echo "bench-fandhe BUILD FAILED: $(tail -3 build-err.tmp | tr '\n' ' ')" >&2
  rm -f build-err.tmp
  exit 1
fi
rm -f build-err.tmp

SOURCE_DESC="$(fandhe_ai_source_desc --config "$PATCH_CONFIG" || true)"
if [[ "$SOURCE_DESC" != "path:${AB_PATCH_FACADE_PATH}" ]]; then
  echo "error: fandhe-ai が期待した path 解決ではない (expected=path:${AB_PATCH_FACADE_PATH} actual=${SOURCE_DESC:-<取得失敗>})" >&2
  exit 1
fi

BIN_SHA="$(sha256_of target/release/bench-fandhe)"
echo "bench-fandhe sha256: $BIN_SHA (source: $SOURCE_DESC)"

SCRIPT_REPO_HEAD_SHA="$(git -C "$SCRIPT_DIR/../../.." rev-parse HEAD 2>/dev/null || echo unknown)"
FACADE_HEAD_SHA="$(git -C "$AB_PATCH_FACADE_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
MANIFEST_TMP="${MANIFEST}.tmp"
cat > "$MANIFEST_TMP" <<JSON
{"label":"${LABEL}","device":"metal","script_repo_head_sha":"${SCRIPT_REPO_HEAD_SHA}","facade_head_sha":"${FACADE_HEAD_SHA}","bin_sha256":"${BIN_SHA}","bin_source":"${SOURCE_DESC}","readout_arms":["legacy","borrowed"],"recorded_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
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

run() { # run <readout> <size> <mode>
  local readout=$1 size=$2 mode=$3
  verify_binary
  echo "== bench-fandhe gemm metal size=$size mode=$mode readout=$readout =="
  if ! ./target/release/bench-fandhe --task gemm --device metal --size "$size" --mode "$mode" --readout "$readout" --out "$OUT_TMP" 2>err.tmp; then
    echo "gemm metal size=$size mode=$mode readout=$readout : $(cat err.tmp)" >> "$SKIP_TMP"
    echo "  -> FAILED (recorded in $SKIP_TMP)"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f err.tmp
}

echo "== metal status (before loop) =="
sysctl -n machdep.cpu.brand_string 2>&1 || true
pmset -g therm 2>&1 || true
uptime 2>&1 || true

UPTIME_SAMPLER_LOG="results/raw/uptime-readout-ab-${LABEL}.log"
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

# run 単位で legacy/borrowed を交互起動する。奇数 run: legacy→borrowed・
# 偶数 run: borrowed→legacy（起動順序自体の系統誤差を均す。
# `run_ab_gemm_metal.sh` と同型。判定規則 §2）。
for run_i in $(seq 1 "$AB_ROUNDS"); do
  for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
      if (( run_i % 2 == 1 )); then
        run legacy "$size" "$mode"
        run borrowed "$size" "$mode"
      else
        run borrowed "$size" "$mode"
        run legacy "$size" "$mode"
      fi
    done
  done
  echo "== run $run_i/$AB_ROUNDS 完了時点の status =="
  pmset -g therm 2>&1 || true
  uptime 2>&1 || true
done

kill "$UPTIME_SAMPLER_PID" 2>/dev/null || true
wait "$UPTIME_SAMPLER_PID" 2>/dev/null || true
trap restore_lock_trap EXIT

echo "== metal status (after loop) =="
pmset -g therm 2>&1 || true
uptime 2>&1 || true

MV_FAILED=0
mv_checked() { # mv_checked <src> <dst>
  if ! mv -f "$1" "$2"; then
    echo "error: mv -f '$1' '$2' に失敗した" >&2
    MV_FAILED=$((MV_FAILED + 1))
  fi
}

if [[ "$ANY_FAILED" -eq 0 ]]; then
  mv_checked "$OUT_TMP" "$OUT"
  mv_checked "$SKIP_TMP" "$SKIP"
  mv_checked "$MANIFEST_TMP" "$MANIFEST"
  if [[ "$MV_FAILED" -ne 0 ]]; then
    echo "error: $MV_FAILED 件の mv が失敗した（正規パスの反映が不完全な可能性がある）" >&2
    exit 1
  fi
  echo "done. results in $OUT ; failures (if any) in $SKIP ; manifest in $MANIFEST"
else
  FAIL_TS=$(date -u +%Y%m%dT%H%M%SZ)
  mv_checked "$OUT_TMP" "results/raw/results-m4max-readout-ab-${LABEL}.failed-${FAIL_TS}.jsonl"
  mv_checked "$SKIP_TMP" "results/raw/skipped-m4max-readout-ab-${LABEL}.failed-${FAIL_TS}.log"
  mv_checked "$MANIFEST_TMP" "results/raw/manifest-m4max-readout-ab-${LABEL}.failed-${FAIL_TS}.json"
  echo "FAILED: $ANY_FAILED run(s) failed; partial/unreliable data kept for diagnosis (${FAIL_TS}). $OUT/$MANIFEST left untouched (fail-closed. security.md A08)." >&2
  exit 1
fi
