#!/usr/bin/env bash
# イシュー #4299: G-CUDA-TRN（CUDA の MLP 学習・size 64・reuse）の after が #4168 で 5/5 遅かった原因を、
#   stream 種別の 4 腕 A/B で切り分ける GB10（aarch64 Linux + CUDA）専用ランナー。
# 役割: 計測（ビルド 2 つ・事前確認・ゲート・16 ラウンド・記録）だけを担う。判定は judge_diag_4299.py、
#   ロックの取得・収録（マスク・漏れ検査・SHA256SUMS）は docs/perf/logs/diag-4299-cudatrn-stream/collect.sh が行う。
# 規則の正本: docs/perf/logs/diag-4299-cudatrn-stream/RULE.txt（計測より前に固定する。事後に緩めない）。
# 派生元（いずれも変えない。#4168 の道具は RULE で sha256 固定済み）:
#   run_ab_prepost_0150_4168.sh（path patch・cargo tree の事前確認・Cargo.lock の退避と復元・GB10 のゲート・環境変数の拒否）、
#   docs/perf/logs/prepost-0150-4168/orchestrate.sh（起動役の作法）。
# 腕（同じ BASE〈RULE の BASE SHA〉の同じ bench-fandhe ソース。ビルドは 2 つだけ）:
#   A = before のバイナリ（registry の fandhe-ai =0.14.0・--locked）。環境変数なし
#   B = after のバイナリ（`--config patch.crates-io.fandhe-ai.path="<BASE>/crates/facade"`）。環境変数なし
#   C = B と同じバイナリ + そのプロセスにだけ FANDHE_AI_CUDA_GRAPH_INFER=0
#   D = A と同じバイナリ + そのプロセスにだけ FANDHE_AI_CUDA_GRAPH_INFER=1
#   腕と stream 種別（legacy／created）の対応はコードを読んだ結果で、実行時には観測していない（RULE.txt [腕]）。
# 事前確認（fail-closed）: RULE.txt の [道具の sha256]・[ラテン方格] ブロックと実体の完全一致、BASE の .rev-stamp と HEAD が
#   RULE の BASE SHA と一致、BASE の追跡ファイルに変更なし、ランナー側 clone と BASE の crates/・bench-fandhe の差分 0、
#   cargo tree による腕の解決、FANDHE_AI_*／RUST_AI_* の不在（RAYON_NUM_THREADS は unset）、実機ロックの owner 一致。
# 使い方（起動役 collect.sh が呼ぶ。直接呼ぶときもロックは collect.sh 経由で取る）:
#   AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> AB_HWLOCK_OWNER=<owner> bash run_diag_4299_cudatrn_stream.sh [--dry-run] <label>
#   AB_RUNNER_ROOT=<abs> bash run_diag_4299_cudatrn_stream.sh --print-tool-sha256
#   --dry-run: ビルドと事前確認だけを行い、ゲート・計測はしない（出力は終了時に削除）。
# 出力: scripts/bench/framework-compare/results/raw/diag4299-<label>/roundN/<腕>-<mode>.jsonl（1 ファイル 1 行）と
#   roundN/launch.tsv（起動ごとに 1 行。渡した環境変数を行ごとに記録）・env_info.txt・gate.log・compute_apps.log・skipped.log。
# bash 3.2 互換（mapfile・連想配列・eval・jq 不使用）。変数の直後に全角文字を置かない（${VAR} と書く）。
set -uo pipefail

DRY=0
PRINT=""
case "${1:-}" in
  --dry-run) DRY=1; shift ;;
  --print-tool-sha256) PRINT=tools; shift ;;
esac

# 腕の既定は環境変数で黙って変わる（FANDHE_AI_CUDA_GRAPH_INFER で A・B の stream が変わる）。
# unset で黙って直さず、接頭辞 FANDHE_AI_／RUST_AI_ の変数が 1 つでもあれば拒否する（print モード・dry-run でも同じ）。
PREFIXED_ENV="$(env | cut -d= -f1 | grep -E '^(FANDHE_AI_|RUST_AI_)' || true)"
if [[ -n "${PREFIXED_ENV}" ]]; then
  echo "error: 環境変数 FANDHE_AI_*／RUST_AI_* が設定されている（腕の既定が変わるため拒否。名前: $(echo "${PREFIXED_ENV}" | tr '\n' ' ')）" >&2; exit 1
fi
RAYON_PRIOR='unset'
if [[ -n "${RAYON_NUM_THREADS+x}" ]]; then RAYON_PRIOR='set'; fi
unset RAYON_NUM_THREADS

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-16}
NROUNDS=16
GRAPH_ENV=FANDHE_AI_CUDA_GRAPH_INFER

# 外部入力のパスは絶対パスに限り、引用符・バックスラッシュ・改行を拒否する（A03。cargo の --config へ埋め込むため厳格に）。
validate_abs() { # <var_name> <path>
  if [[ -z "$2" || "$2" != /* || "$2" == *\"* || "$2" == *\\* || "$2" == *$'\n'* || "$2" == *\'* ]]; then
    echo "error: ${1} は引用符・バックスラッシュ・改行を含まない絶対パス必須" >&2; exit 1
  fi
}
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
FC_REL=scripts/bench/framework-compare
LOGS_REL=docs/perf/logs/diag-4299-cudatrn-stream
RUNNER_REL=$FC_REL/run_diag_4299_cudatrn_stream.sh

# RULE が固定する道具（RUNNER_ROOT 相対。この順で出力する）。判定器側の同じ一覧は judge_diag_4299.py の TOOL_FILES。
TOOL_FILES="$RUNNER_REL
$FC_REL/judge_diag_4299.py
$LOGS_REL/collect.sh"
tool_sha_lines() { # <root> -> 'sha256  相対パス' の行（RULE の [道具の sha256] ブロックと完全一致させる）
  local rel
  while IFS= read -r rel; do
    [[ -f "$1/$rel" ]] || { echo "error: 道具ファイルが無い: ${rel}" >&2; return 1; }
    printf '%s  %s\n' "$(sha256_of "$1/$rel")" "$rel"
  done <<EOF
$TOOL_FILES
EOF
}

# 腕の順番（4×4 のラテン方格。Williams 型で、どの腕の直前にもほかの 3 腕が 1 回ずつ来る）。
# round r は (r-1) mod 4 + 1 行目を使い、16 ラウンドで方格を 4 回繰り返す。RULE の [ラテン方格] と完全一致させる
# （判定器の LATIN と同じ。launch.tsv の起動順を判定器が照合する）。
LATIN="A B D C
B C A D
C D B A
D A C B"
latin_row() { # <round> -> その round の腕の並び（空白区切り）
  local idx=$(( ($1 - 1) % 4 + 1 ))
  echo "$LATIN" | sed -n "${idx}p"
}

# RULE の固定ブロック（`# >>> <名前>` と `# <<< <名前>` の間の行をそのまま）を取り出す（#4168 のランナーと同じ）。
rule_block() { # <rule-file> <名前>
  awk -v b="# >>> $2" -v e="# <<< $2" '$0 == e { on = 0 } on { print } $0 == b { on = 1 }' "$1"
}

if [[ "$PRINT" == "tools" ]]; then
  validate_abs AB_RUNNER_ROOT "${AB_RUNNER_ROOT:-}"
  tool_sha_lines "$AB_RUNNER_ROOT"; exit $?
fi

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> AB_HWLOCK_OWNER=<owner> $0 [--dry-run] <label>  (label: [A-Za-z0-9._-]+)" >&2
  echo "       AB_RUNNER_ROOT=<abs> $0 --print-tool-sha256" >&2; exit 1
fi
if [[ "$ROUNDS" != "$NROUNDS" ]]; then echo "error: AB_ROUNDS は ${NROUNDS} 固定（RULE.txt。got: ${ROUNDS}）" >&2; exit 1; fi
validate_abs AB_BASE_ROOT "${AB_BASE_ROOT:-}"
validate_abs AB_RUNNER_ROOT "${AB_RUNNER_ROOT:-}"
BASE_ROOT=$AB_BASE_ROOT; RUNNER_ROOT=$AB_RUNNER_ROOT
# GB10 は対話しない ssh から起動されると PATH に cargo・nvidia-smi が無い。cargo tree（腕の解決確認）より前に足す。
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
LOAD_GATE=1.0

# 機体の同定と実機ロック（dry-run もビルドで機体を使うため、ロックの owner 一致を要求する。取得・解放は collect.sh）。
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then echo "error: GB10（aarch64 Linux）専用" >&2; exit 1; fi
command -v nvidia-smi >/dev/null 2>&1 || { echo "error: nvidia-smi が無い（GB10 専用）" >&2; exit 1; }
OWNER="${AB_HWLOCK_OWNER:-}"
[[ -n "$OWNER" && "$OWNER" != *$'\n'* ]] || { echo "error: AB_HWLOCK_OWNER 必須（collect.sh 経由で起動する）" >&2; exit 1; }
[[ "$(cat "${HOME}/work/.fandhe-hw-lock/owner" 2>/dev/null)" == "$OWNER" ]] || { echo "error: 実機ロックの owner が一致しない（LOCK-NOT-MINE）" >&2; exit 1; }

RULE_FILE="$RUNNER_ROOT/$LOGS_REL/RULE.txt"
FACADE_PATH="$BASE_ROOT/crates/facade"
validate_abs FACADE_PATH "$FACADE_PATH"

if [[ "$DRY" == "1" ]]; then
  OUT="$(mktemp -d)"
else
  OUT="$RUNNER_ROOT/$FC_REL/results/raw/diag4299-${LABEL}"
  if [[ -e "$OUT" ]]; then echo "error: ラベル ${LABEL} の出力が既に存在する（差し替え禁止）: ${OUT}" >&2; exit 1; fi
  mkdir -p "$OUT"
fi
SKIP="$OUT/skipped.log"; : >"$SKIP"
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }
# dry-run の出力は Cargo.lock を戻すまで消さない（下の cleanup_exit が後始末する）。ここでは早期失敗用の仮の trap。
if [[ "$DRY" == "1" ]]; then trap 'rm -rf "$OUT"' EXIT; fi

# ---- 事前確認 1: RULE.txt（未記入の拒否・BASE SHA・道具の sha256・ラテン方格）
stamp_of() { if [[ -f "$1/.rev-stamp" ]]; then head -n1 "$1/.rev-stamp"; fi; }
[[ -f "$RULE_FILE" ]] || precheck_fail "RULE.txt が無い（計測より前に固定してコミットする）"
if grep -q '<要記入' "$RULE_FILE"; then precheck_fail "RULE.txt に <要記入 が残っている"; fi
RULE_SHA="$(sha256_of "$RULE_FILE")"
BASE_SHA="$(grep -E '^- BASE = [0-9a-f]{40}$' "$RULE_FILE" | head -n1 | grep -oE '[0-9a-f]{40}')"
[[ -n "$BASE_SHA" ]] || precheck_fail "RULE.txt から BASE SHA（'- BASE = <40hex>'）を読めない"
# ブロックは `$(...)` で末尾の改行が落ちるため、空行を先に拒否して判定器と読みを揃える。
if rule_block "$RULE_FILE" tool-sha256 | grep -q '^$'; then precheck_fail "[道具の sha256] ブロックに空行がある"; fi
EXP_TOOLS="$(rule_block "$RULE_FILE" tool-sha256)"
[[ -n "$EXP_TOOLS" ]] || precheck_fail "RULE.txt に [道具の sha256] ブロック（# >>> tool-sha256）が無い"
ACT_TOOLS="$(tool_sha_lines "$RUNNER_ROOT")" || precheck_fail "道具ファイルを読めない"
[[ "$ACT_TOOLS" == "$EXP_TOOLS" ]] || precheck_fail "道具の sha256 が RULE と不一致（--print-tool-sha256 の出力と RULE のブロックを比べる）"
# 実行中のこのスクリプト自身も RUNNER_ROOT 側と同一であること（別の場所の改変コピーを実行しても通さない）。
[[ "$(sha256_of "${BASH_SOURCE[0]}")" == "$(sha256_of "$RUNNER_ROOT/$RUNNER_REL")" ]] \
  || precheck_fail "実行中のランナーが AB_RUNNER_ROOT 側のランナーと一致しない"
echo "tool-sha256=ok ($(echo "$EXP_TOOLS" | wc -l | tr -d ' ') 件)"
[[ "$(rule_block "$RULE_FILE" latin-square)" == "$LATIN" ]] || precheck_fail "RULE の [ラテン方格] がランナーの LATIN と不一致"
echo "latin-square=ok"

# ---- 事前確認 2: BASE（.rev-stamp と HEAD が BASE SHA・追跡ファイルの変更なし）とランナー側 clone の差分
[[ "$(stamp_of "$BASE_ROOT")" == "$BASE_SHA" ]] || precheck_fail "BASE の .rev-stamp が RULE の BASE SHA と不一致"
[[ "$(git -C "$BASE_ROOT" rev-parse HEAD 2>/dev/null)" == "$BASE_SHA" ]] || precheck_fail "BASE の HEAD が RULE の BASE SHA と不一致（git archive ではなく clone を使う）"
git -C "$BASE_ROOT" diff --quiet HEAD -- crates "$FC_REL" || precheck_fail "BASE の crates/・framework-compare に未コミットの変更がある"
STAMP_RUNNER="$(stamp_of "$RUNNER_ROOT")"
[[ -n "$STAMP_RUNNER" ]] || precheck_fail "runner の .rev-stamp が無い"
git -C "$RUNNER_ROOT" cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null || precheck_fail "ランナー側 clone に BASE SHA が無い"
# ランナー側（RULE を含むコミット）と BASE の差は crates/・bench-fandhe で 0 であること（RULE [BASE]）。
git -C "$RUNNER_ROOT" diff --quiet "$BASE_SHA" HEAD -- crates "$FC_REL/bench-fandhe" \
  || precheck_fail "ランナー側 clone と BASE の間に crates/・bench-fandhe の差分がある"
# B が created stream を既定で選ぶ前提（#4167。RULE [腕]）。BASE SHA で固定されるため確認だけ。
[[ "$(grep -cE '^pub\(crate\) const INFER_GRAPH_DEFAULT_ENABLED: bool = true;$' "$BASE_ROOT/crates/backend-cuda/src/graph.rs")" == "1" ]] \
  || precheck_fail "BASE の INFER_GRAPH_DEFAULT_ENABLED が true でない"
echo "base=ok (${BASE_SHA})"

# ---- ビルド・腕の解決確認（Cargo.lock は path patch で書き換わるため退避し、終了時に復元して差分 0 を確かめる）
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${FACADE_PATH}\""
LOCK_CLEAN_REL="$FC_REL/Cargo.lock"
LOCK_COPY="$(mktemp)"
cp "$BASE_ROOT/$LOCK_CLEAN_REL" "$LOCK_COPY" || { echo "error: Cargo.lock の退避に失敗" >&2; exit 1; }
# 復元に失敗したら退避ファイルを残して保存先とエラーを示し非 0 を返す（成功時のみ trap が退避を削除する）。
restore_lock() {
  cp "$LOCK_COPY" "$BASE_ROOT/$LOCK_CLEAN_REL" || { echo "error: Cargo.lock の復元に失敗。退避ファイルを残した: ${LOCK_COPY}（手動で ${BASE_ROOT}/${LOCK_CLEAN_REL} へ戻す）" >&2; return 1; }
}
cleanup_exit() {
  if restore_lock; then rm -f "$LOCK_COPY"; fi
  if [[ "$DRY" == "1" ]]; then rm -rf "$OUT"; fi
}
trap cleanup_exit EXIT
FC_DIR="$BASE_ROOT/$FC_REL"

tree_checks() { # A（before）／B（after）の fandhe-ai 解決を検査
  local t need
  t="$(cd "$FC_DIR" && cargo tree --locked -p bench-fandhe --depth 2 --target all 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-A.txt"
  echo "$t" | grep -qE 'fandhe-ai v0\.14\.0$' || precheck_fail "A の fandhe-ai が registry 0.14.0 に解決されない"
  if echo "$t" | grep -E 'fandhe-ai[a-z-]* v[0-9.]+ \(' >/dev/null; then precheck_fail "A に path へ解決された fandhe-ai 系クレートがある"; fi
  t="$(cd "$FC_DIR" && cargo tree -p bench-fandhe --depth 2 --target all --config "$PATCH_CONFIG" 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-B.txt"
  # B: fandhe-ai と fandhe-ai-* のすべてが path に解決される（registry 形 'fandhe-ai... vX.Y.Z' で行が終わる行が 0 件）
  if echo "$t" | grep -E 'fandhe-ai[a-z-]* v[0-9.]+$' >/dev/null; then precheck_fail "B に registry へ解決された fandhe-ai 系クレートが残る（比較が無意味になる）"; fi
  # 今回測る差（stream の選択）は backend-cuda にあるため、backend-cuda も BASE の crates/ へ解決されることを要求する。
  for need in fandhe-ai fandhe-ai-tensor-core fandhe-ai-autodiff fandhe-ai-backend-cpu fandhe-ai-backend-cuda; do
    echo "$t" | grep -qE "${need} v[0-9.]+ \(${BASE_ROOT}/crates/" || precheck_fail "B で ${need} が BASE の crates/ へ解決されない"
  done
  return 0
}
tc=0; tree_checks || tc=$?
# B 側の cargo tree（--locked なし）は Cargo.lock を patched グラフへ書き換える。A ビルド（--locked）の前に必ず戻す。
restore_lock || exit 1
[[ "$tc" -eq 0 ]] || precheck_fail "cargo tree の実行に失敗（rc=${tc}）"
echo "tree-checks=ok"

# ビルドは計測の前にすべて済ませる（2 つだけ。計測中はビルドを並走させない）。
( cd "$FC_DIR" && cargo build --release --locked -p bench-fandhe --target-dir "$OUT/target-A" ) 2>"$OUT/build-A.err" \
  || { tail -40 "$OUT/build-A.err"; echo "BUILD FAILED A" >>"$SKIP"; exit 1; }
# B は --locked を付けない（path patch の解決で Cargo.lock が更新されるため。直後と終了時に復元する）
( cd "$FC_DIR" && cargo build --release -p bench-fandhe --target-dir "$OUT/target-B" --config "$PATCH_CONFIG" ) 2>"$OUT/build-B.err" \
  || { tail -40 "$OUT/build-B.err"; echo "BUILD FAILED B" >>"$SKIP"; exit 1; }
restore_lock || exit 1
git -C "$BASE_ROOT" diff --quiet -- "$LOCK_CLEAN_REL" || precheck_fail "ビルド後に Cargo.lock の差分が残った"
BIN_BEFORE="$OUT/target-A/release/bench-fandhe"; BIN_AFTER="$OUT/target-B/release/bench-fandhe"
[[ -x "$BIN_BEFORE" && -x "$BIN_AFTER" ]] || precheck_fail "bench-fandhe のバイナリが無い"
SHA_BEFORE="$(sha256_of "$BIN_BEFORE")"; SHA_AFTER="$(sha256_of "$BIN_AFTER")"
[[ "$SHA_BEFORE" != "$SHA_AFTER" ]] || precheck_fail "before と after のバイナリが同一（腕が分かれていない）"
echo "precheck=ok"

{
  echo "profile=gb10"; echo "label=${LABEL}"; echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  echo "dry_run=${DRY}"; echo "rule_sha256=${RULE_SHA}"; echo "tool_sha256_check=ok"; echo "latin_square_check=ok"
  echo "fandhe_ai_rust_ai_env_vars=0"; echo "rayon_num_threads=unset"; echo "rayon_num_threads_prior=${RAYON_PRIOR}"
  echo "base_sha=${BASE_SHA}"; echo "rev_stamp_base=$(stamp_of "$BASE_ROOT")"; echo "rev_stamp_runner=${STAMP_RUNNER}"
  echo "runner_base_diff_crates_bench_fandhe=0"
  echo "cargo_lock_sha256=$(sha256_of "$LOCK_COPY")"
  echo "bin_sha256_A=${SHA_BEFORE}"; echo "bin_sha256_B=${SHA_AFTER}"
  echo "bin_sha256_C=${SHA_AFTER}"; echo "bin_sha256_D=${SHA_BEFORE}"
  echo "env_A=none"; echo "env_B=none"; echo "env_C=${GRAPH_ENV}=0"; echo "env_D=${GRAPH_ENV}=1"
  echo "gate=1"; echo "load_gate=${LOAD_GATE}"; echo "rounds=${ROUNDS}"; echo "utc=$(date -u +%FT%TZ)"
} >"$OUT/env_info.txt"

if [[ "$DRY" == "1" ]]; then
  echo "dry-run: ビルドと事前確認まで済んだ。ゲート・計測はしない（出力は終了時に削除する）。計画:"
  echo "  rounds=${NROUNDS}（各ラウンドの前に compute プロセス 0・load1<${LOAD_GATE}・GPU 使用率 0% を 3 サンプル連続）"
  echo "  各ラウンド: reuse を方格の行の順に 4 腕 → fresh を同じ順に 4 腕（1 プロセス 1 行）"
  r=1
  while [[ "$r" -le 4 ]]; do echo "  方格 ${r} 行目: $(latin_row "$r")"; r=$((r + 1)); done
  echo "DRY-RUN-OK"
  exit 0
fi

load1() {
  local l=""
  if [[ -r /proc/loadavg ]]; then l=$(awk '{print $1}' /proc/loadavg); fi
  printf '%s' "$l"
}
gpu_idle3() {
  local n=0 u
  while [[ "$n" -lt 3 ]]; do
    u="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    [[ "$u" == "0" ]] || return 1
    n=$((n + 1)); if [[ "$n" -lt 3 ]]; then sleep 2; fi
  done
  return 0
}
# GB10 の専有確認（#4168 と同じ fail-closed）: 取得失敗は EXCLUSIVE-INVALID、1 件でもあれば専有未達。
compute_apps_zero() { # <記録先>
  local apps rc
  apps="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>&1)"; rc=$?
  printf '%s rc=%s\n%s\n' "$(date -u +%FT%TZ)" "$rc" "$apps" >>"$1"
  [[ "$rc" -eq 0 ]] || return 2
  [[ -z "$apps" ]] || return 1
  return 0
}
# ゲート（#4168 の GB10 と同じ）: compute プロセス 0 を確かめ、load1 < 1.0 かつ GPU 使用率 0% 3 サンプル連続を 30 秒間隔・最大 30 分待つ。
gate_wait() { # <round>
  local waited=0 l ok rc
  rc=0; compute_apps_zero "$OUT/compute_apps.log" || rc=$?
  if [[ "$rc" -eq 2 ]]; then echo "round$1 EXCLUSIVE-INVALID" >>"$OUT/gate.log"; return 2; fi
  if [[ "$rc" -eq 1 ]]; then echo "round$1 EXCLUSIVE-UNMET compute-process-present" >>"$OUT/gate.log"; return 3; fi
  while :; do
    l="$(load1)"
    if [[ ! "$l" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "round$1 GATE-INVALID load1-not-numeric" >>"$OUT/gate.log"; return 2; fi
    ok="$(awk -v a="$l" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')"
    if [[ "$ok" == "1" ]] && ! gpu_idle3; then ok=0; fi
    if [[ "$ok" == "1" ]]; then
      echo "round$1 gate=pass threshold=${LOAD_GATE} waited_s=${waited} load1=${l}" >>"$OUT/gate.log"; return 0
    fi
    if [[ "$waited" -ge 1800 ]]; then echo "round$1 GATE-TIMEOUT load1=${l}" >>"$OUT/gate.log"; return 1; fi
    sleep 30; waited=$((waited + 30))
  done
}
: >"$OUT/gate.log"

arm_bin() { case "$1" in A|D) echo "$BIN_BEFORE" ;; B|C) echo "$BIN_AFTER" ;; esac; }
arm_build() { case "$1" in A|D) echo before ;; B|C) echo after ;; esac; }
arm_env() { case "$1" in A|B) echo none ;; C) echo "${GRAPH_ENV}=0" ;; D) echo "${GRAPH_ENV}=1" ;; esac; }
# 1 プロセス 1 行。環境変数は env でそのプロセスにだけ渡す（A・B は何も渡さない）。--diag-*・--graph・--phases は渡さない。
# 失敗（rc≠0・行数≠1）は skipped へ記録して系列を打ち切る（差し替え・追加起動はしない）。
launch() { # <round> <pos> <arm> <mode>
  local rd="$OUT/round$1" arm=$3 mode=$4 bin envs out errf t0 t1 crc=0 rows
  bin="$(arm_bin "$arm")"; envs="$(arm_env "$arm")"
  out="$rd/${arm}-${mode}.jsonl"; errf="$rd/${arm}-${mode}.err"
  t0="$(date -u +%FT%TZ)"
  if [[ "$envs" == "none" ]]; then
    env "$bin" --task train --device cuda --size 64 --mode "$mode" --out "$out" 2>"$errf" || crc=$?
  else
    env "$envs" "$bin" --task train --device cuda --size 64 --mode "$mode" --out "$out" 2>"$errf" || crc=$?
  fi
  t1="$(date -u +%FT%TZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$arm" "$mode" "$envs" "$(arm_build "$arm")" "$crc" "$t0" "$t1" >>"$rd/launch.tsv"
  rows=0; if [[ -f "$out" ]]; then rows=$(wc -l <"$out" | tr -d ' '); fi
  if [[ "$crc" -ne 0 || "$rows" != "1" ]]; then
    echo "round$1 ${arm} ${mode}: rc=${crc} rows=${rows} $(tr '\n' ' ' <"$errf")" >>"$SKIP"
    return 1
  fi
  return 0
}

R=1
while [[ "$R" -le "$NROUNDS" ]]; do
  RD="$OUT/round$R"; mkdir -p "$RD"; : >"$RD/launch.tsv"
  grc=0; gate_wait "$R" || grc=$?
  if [[ "$grc" -ne 0 ]]; then echo "SERIES-ABORTED completed=$((R - 1))/${NROUNDS} (gate rc=${grc})"; exit 1; fi
  uptime >"$RD/uptime_before.txt"
  nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_before.txt" 2>&1 || true
  ORDER="$(latin_row "$R")"
  pos=0
  # reuse（判定行）を方格の行の順に 4 腕 → fresh（参考）を同じ順に 4 腕
  for mode in reuse fresh; do
    for arm in $ORDER; do
      pos=$((pos + 1))
      if ! launch "$R" "$pos" "$arm" "$mode"; then echo "SERIES-ABORTED completed=$((R - 1))/${NROUNDS} (round${R} ${arm} ${mode} が失敗)"; exit 1; fi
    done
  done
  uptime >"$RD/uptime_after.txt"
  nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_after.txt" 2>&1 || true
  echo "round${R}: order=$(echo "$ORDER" | tr -d ' ') launches=${pos}"
  R=$((R + 1))
done
restore_lock || exit 1
if ! git -C "$BASE_ROOT" diff --quiet -- "$LOCK_CLEAN_REL"; then echo "error: Cargo.lock に差分が残った" >&2; exit 1; fi
echo "ALL-DONE completed=${NROUNDS}/${NROUNDS}"
