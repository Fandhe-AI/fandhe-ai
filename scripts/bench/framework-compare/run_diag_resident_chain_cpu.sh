#!/usr/bin/env bash
# イシュー #3611: 常駐チェーン CPU train セルの REGRESS（#3257: Mac mini +0.4%・GB10 +5.9%）の原因を切り分ける診断ランナー。
# 両機体共用（AB_PROFILE=macmini|gb10。機体間の並走は可、同一機体では計測とビルドを並走させない）。
# 手順の正本: docs/perf/logs/resident-chain-cpu-train-diag-3611/DIAG-r2.txt（r2。r1 の DIAG.txt は precheck の mangling 正規化不備で計測 0 本。
# 計測前に固定。DIAG に runner／helper の sha256 を持たせ、一致しなければ停止する。本スクリプトは sha256 を env_info へ記録し、
# summarize_diag_resident_chain_cpu.py が DIAG と突合する）。判定ではなく分類を出す診断であり、judge ツールの入力ではない。
# run_ab_resident_chain.sh は 1 ツリー・4 腕固定で B0／B1／B2 の 3 ツリーと A/A 腕を扱えないため流用せず、同じ検証
# （facade パス検証・.rev-stamp・一時 index による完全一致）を 7 腕向けに持つ。既存ランナーは変更しない。
# 書式: scripts/bench/framework-compare/ の兄弟スクリプトに合わせ、汎用 scripts/ の `set -euo pipefail` ではなく
# `set -u`（失敗は skipped へ記録して継続し最後に非 0 終了。bash 3.2 互換・mapfile／連想配列／eval 不使用）。
#
# ツリー: B0 = DIAG の oldcode、B1 = DIAG の b1、B2 = B1 から作る detached worktree + layout-shift.patch（終了時に必ず削除）。
# 腕: aa0／aa1（B1・reuse の同一バイナリ）、ch1（B1・chain）、rb0／rb1（B2・reuse／chain）、o0／o1（B0・reuse／chain）。
# 呼び出し例:
#   AB_PROFILE=macmini AB_B0_FACADE_PATH=<abs>/b0/crates/facade AB_B1_FACADE_PATH=<abs>/b1/crates/facade \
#   AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 bash run_diag_resident_chain_cpu.sh 3611-macmini
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./ab_round_cooldown.sh
source ./ab_round_cooldown.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
PROFILE=${AB_PROFILE:-}
ARMS=(aa0 aa1 ch1 rb0 rb1 o0 o1)
NARMS=7

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: AB_PROFILE=macmini|gb10 $0 <label>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（DIAG.txt。got: ${ROUNDS}）" >&2
  exit 1
fi
case "$PROFILE" in
  macmini) LOAD_GATE=3.0 ;;
  gb10) LOAD_GATE=1.0 ;;
  *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
esac
# 継承環境の漏れ拒否（配置パディング関数を起動する環境変数は設定させない）。
if [[ -n "${BENCH_LAYOUT_PAD_3611+x}" ]]; then
  echo "error: 継承環境に BENCH_LAYOUT_PAD_3611 が設定されている。unset してから実行すること" >&2; exit 1
fi
# Mac mini は round 前に ab_round_cooldown.sh を 3.0 で通す（DIAG [負荷ゲート]）。未設定なら 3.0、異値は拒否。
if [[ "$PROFILE" == "macmini" ]]; then
  if [[ -z "${AB_ROUND_COOLDOWN_MAX_LOAD1:-}" ]]; then
    AB_ROUND_COOLDOWN_MAX_LOAD1=3.0
  elif [[ "$AB_ROUND_COOLDOWN_MAX_LOAD1" != "3.0" ]]; then
    echo "error: macmini の AB_ROUND_COOLDOWN_MAX_LOAD1 は 3.0 固定（DIAG.txt。got: ${AB_ROUND_COOLDOWN_MAX_LOAD1}）" >&2; exit 1
  fi
  export AB_ROUND_COOLDOWN_MAX_LOAD1
fi

validate_facade_path() { # <var_name> <path>
  local var_name=$1 path=$2
  if [[ -z "$path" || "$path" != /* ]]; then
    echo "error: ${var_name} は絶対パス必須（got: ${path:-unset}）" >&2; exit 1
  fi
  # --config 文字列へ埋め込むため引用符・バックスラッシュ・改行を拒否する（A03）。
  if [[ "$path" == *\"* || "$path" == *\\* || "$path" == *$'\n'* ]]; then
    echo "error: ${var_name} に使えない文字を含む" >&2; exit 1
  fi
  if [[ ! -f "$path/Cargo.toml" ]] || ! grep -qE '^[[:space:]]*name[[:space:]]*=[[:space:]]*"fandhe-ai"[[:space:]]*$' "$path/Cargo.toml"; then
    echo "error: ${var_name}/Cargo.toml が name = \"fandhe-ai\" を宣言していない" >&2; exit 1
  fi
}
validate_facade_path AB_B0_FACADE_PATH "${AB_B0_FACADE_PATH:-}"
validate_facade_path AB_B1_FACADE_PATH "${AB_B1_FACADE_PATH:-}"
B0_ROOT="$(cd "${AB_B0_FACADE_PATH}/../.." && pwd)"
B1_ROOT="$(cd "${AB_B1_FACADE_PATH}/../.." && pwd)"
if [[ "$B0_ROOT" == "$B1_ROOT" ]]; then
  echo "error: B0 と B1 が同一ツリー" >&2; exit 1
fi
RUNNER_ROOT="$(cd ../../.. && pwd)"
LOGDIR="$RUNNER_ROOT/docs/perf/logs/resident-chain-cpu-train-diag-3611"
# DIAG ファイル名は固定の正規表現のみ受け、固定ディレクトリ配下でのみ解決する（パス走査防止。A03）。
DIAG_NAME="${DIAG_NAME-DIAG-r2.txt}"
if [[ ! "$DIAG_NAME" =~ ^DIAG(-[a-z0-9]+)?\.txt$ ]]; then
  echo "error: DIAG_NAME は DIAG(-<英小文字数字>).txt のファイル名のみ（got: ${DIAG_NAME}）" >&2; exit 1
fi
DIAG_FILE="$LOGDIR/${DIAG_NAME}"
PATCH_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"
LAYOUT_FILE="$LOGDIR/layout-shift.patch"

# 機体同定（fail-closed）。Mac mini は M4 完全一致のみ（M4 Max 開発機の値を混ぜない）。
if [[ "$PROFILE" == "macmini" ]]; then
  brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  if [[ "$brand" != "Apple M4" ]]; then
    echo "error: macmini プロファイルは 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1
  fi
else
  if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: gb10 は aarch64 のみ" >&2; exit 1; fi
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  if [[ "$GPU_NAME" != *GB10* ]]; then
    echo "error: nvidia-smi の GPU 名に GB10 を含まない（got: ${GPU_NAME:-unknown}）" >&2; exit 1
  fi
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
[[ -f "$DIAG_FILE" && -f "$PATCH_FILE" && -f "$LAYOUT_FILE" ]] || { echo "error: DIAG / patch / layout-shift.patch が無い" >&2; exit 1; }
DIAG_SHA="$(sha256_of "$DIAG_FILE")"
PATCH_SHA="$(sha256_of "$PATCH_FILE")"
LAYOUT_SHA="$(sha256_of "$LAYOUT_FILE")"
# DIAG に値で固定した値の読み出しは行頭アンカー（コメント行・本文中の言及では素通りさせない）。
DIAG_PATCH_SHA="$(sed -n 's/^  patch_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
DIAG_LAYOUT_SHA="$(sed -n 's/^  layout_shift_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
EXPECT_B1="$(sed -n 's/^- b1 = \([0-9a-f]\{40\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
EXPECT_B0="$(sed -n 's/^- oldcode = \([0-9a-f]\{40\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
if [[ -z "$DIAG_PATCH_SHA" || "$DIAG_PATCH_SHA" != "$PATCH_SHA" ]]; then
  echo "error: DIAG の patch_sha256 と実パッチの sha256 が一致しない" >&2; exit 1
fi
if [[ -z "$DIAG_LAYOUT_SHA" || "$DIAG_LAYOUT_SHA" != "$LAYOUT_SHA" ]]; then
  echo "error: DIAG の layout_shift_sha256 と実パッチの sha256 が一致しない" >&2; exit 1
fi
if [[ -z "$EXPECT_B1" || -z "$EXPECT_B0" ]]; then
  echo "error: DIAG から b1／oldcode の SHA を読めない" >&2; exit 1
fi
# r2（#3611）: runner 本体と正規化 helper の sha256 を DIAG に固定し、計測前登録後の差し替えを fail-closed で検出する。
# sha フィールドを持たない DIAG.txt（r1）はここで拒否される。
HELPER_FILE="$(pwd)/diag_layout_symbols.py"
[[ -f "$HELPER_FILE" ]] || { echo "error: diag_layout_symbols.py が無い" >&2; exit 1; }
RUNNER_SHA="$(sha256_of "$(pwd)/$(basename "$0")")"
HELPER_SHA="$(sha256_of "$HELPER_FILE")"
DIAG_RUNNER_SHA="$(sed -n 's/^  runner_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
DIAG_HELPER_SHA="$(sed -n 's/^  layout_helper_sha256: \([0-9a-f]\{64\}\)$/\1/p' "$DIAG_FILE" | head -n1)"
if [[ -z "$DIAG_RUNNER_SHA" || "$DIAG_RUNNER_SHA" != "$RUNNER_SHA" ]]; then
  echo "error: DIAG の runner_sha256 と実ランナーの sha256 が一致しない（DIAG にフィールドが無い場合を含む）" >&2; exit 1
fi
if [[ -z "$DIAG_HELPER_SHA" || "$DIAG_HELPER_SHA" != "$HELPER_SHA" ]]; then
  echo "error: DIAG の layout_helper_sha256 と実 helper の sha256 が一致しない" >&2; exit 1
fi

OUT="results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-diag-resident-chain-cpu-${LABEL}.log"
ENV_INFO="$OUT/env_info-diag-${LABEL}.txt"
# 同じラベルの既存出力があれば停止（run の差し替え・追記禁止。A08）。
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0
unset RAYON_NUM_THREADS

precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

stamp_head() { # <root>
  local stamp real dirty
  stamp="$(head -n1 "$1/.rev-stamp" 2>/dev/null || true)"
  dirty="$(sed -n 2p "$1/.rev-stamp" 2>/dev/null || true)"
  real="$(git -C "$1" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "$stamp" || "$stamp" != "$real" || "$dirty" != "dirty=0" ]]; then
    echo "error: $1 の .rev-stamp が無い・HEAD と不一致・dirty=0 でない（stamp=${stamp:-none} head=${real:-none}）" >&2; exit 1
  fi
  echo "$stamp"
}
STAMP_B0="$(stamp_head "$B0_ROOT")" || exit 1
STAMP_B1="$(stamp_head "$B1_ROOT")" || exit 1
[[ "$STAMP_B0" == "$EXPECT_B0" ]] || precheck_fail "B0 の HEAD が DIAG の oldcode と不一致（head=${STAMP_B0} expected=${EXPECT_B0}）"
[[ "$STAMP_B1" == "$EXPECT_B1" ]] || precheck_fail "B1 の HEAD が DIAG の b1 と不一致（head=${STAMP_B1} expected=${EXPECT_B1}）"

# ツリーが「HEAD + bench パッチ（+ layout パッチ）」だけであることを、一時 index による完全一致で検証する。
check_patched_tree() { # <root> [<extra_patch>]
  local root=$1 extra=${2:-} tmp_idx
  # extra（layout パッチ）適用済みのツリーでは bench パッチのハンクが layout パッチの挿入で
  # 文脈ずれし逆適用チェックが常に失敗するため、extra 無しのときだけ行う。
  # extra 有りの場合は下の一時 index との完全一致（HEAD + bench + extra）が同じ保証を担う。
  if [[ -z "$extra" ]]; then
    git -C "$root" apply --reverse --check "$PATCH_FILE" 2>/dev/null || precheck_fail "bench patch が ${root} と一致しない"
  fi
  tmp_idx="$(mktemp "${TMPDIR:-/tmp}/diag-idx.XXXXXX")" || precheck_fail "一時 index を作れない"
  rm -f "$tmp_idx"
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" read-tree HEAD \
    || ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" apply --cached "$PATCH_FILE"; then
    rm -f "$tmp_idx"; precheck_fail "${root} の HEAD へ bench パッチを適用できない"
  fi
  if [[ -n "$extra" ]] && ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" apply --cached "$extra"; then
    rm -f "$tmp_idx"; precheck_fail "${root} の HEAD へ ${extra} を適用できない"
  fi
  # 追跡ファイル全体が一時 index と完全一致すること。
  if ! GIT_INDEX_FILE="$tmp_idx" git -C "$root" diff --quiet; then
    rm -f "$tmp_idx"; precheck_fail "${root} の作業ツリーが HEAD + 所定パッチと完全一致しない"
  fi
  rm -f "$tmp_idx"
  grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$root/scripts/bench/framework-compare/bench-fandhe/src/main.rs" \
    || precheck_fail "${root} の main.rs に (train, chain) の分岐が無い"
}
check_patched_tree "$B0_ROOT"
check_patched_tree "$B1_ROOT"
echo "precheck=assert ok"

# B0・B1 の Cargo.lock は path patch 付き cargo が書き換えうる。独立に退避し、終了時に復元する。
# B2 の worktree は終了時に必ず片付ける（配列保持。空配列は ${arr[@]+...} で展開して bash 3.2 の unbound を避ける）。
B0_LOCK="$B0_ROOT/scripts/bench/framework-compare/Cargo.lock"
B1_LOCK="$B1_ROOT/scripts/bench/framework-compare/Cargo.lock"
B0_BACKUP="$(mktemp)" || { echo "error: 退避用の一時ファイルを作れない" >&2; exit 1; }
B1_BACKUP="$(mktemp)" || { rm -f "$B0_BACKUP"; echo "error: 退避用の一時ファイルを作れない" >&2; exit 1; }
if ! cp "$B0_LOCK" "$B0_BACKUP" || ! cmp -s "$B0_LOCK" "$B0_BACKUP" \
  || ! cp "$B1_LOCK" "$B1_BACKUP" || ! cmp -s "$B1_LOCK" "$B1_BACKUP"; then
  rm -f "$B0_BACKUP" "$B1_BACKUP"
  echo "error: Cargo.lock の退避に失敗した" >&2; exit 1
fi
WT_LIST=()
cleanup_on_exit() {
  local code=$? wt
  if cp "$B0_BACKUP" "$B0_LOCK"; then rm -f "$B0_BACKUP"; else
    echo "error: B0 の Cargo.lock の復元に失敗した。バックアップを保持する: $B0_BACKUP" >&2; [[ "$code" -eq 0 ]] && code=1
  fi
  if cp "$B1_BACKUP" "$B1_LOCK"; then rm -f "$B1_BACKUP"; else
    echo "error: B1 の Cargo.lock の復元に失敗した。バックアップを保持する: $B1_BACKUP" >&2; [[ "$code" -eq 0 ]] && code=1
  fi
  for wt in ${WT_LIST[@]+"${WT_LIST[@]}"}; do git -C "$B1_ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || true; done
  exit "$code"
}
trap cleanup_on_exit EXIT

# <arm> <root>: root の framework-compare で path patch 付きビルドし、成果物を $OUT へ回収する。
build_in_tree() {
  local name=$1 root=$2 tdir="target-diag-${LABEL}-$1" cfg tree fc out_abs
  fc="$root/scripts/bench/framework-compare"
  out_abs="$(pwd)/$OUT"
  cfg="patch.crates-io.fandhe-ai.path=\"${root}/crates/facade\""
  if ! ( cd "$fc" && cargo build --release -p bench-fandhe --target-dir "$tdir" --config "$cfg" ) 2>"$OUT/build-diag-${name}-${LABEL}.err"; then
    tail -40 "$OUT/build-diag-${name}-${LABEL}.err"
    echo "bench-fandhe BUILD FAILED (${name})" >>"$SKIP"; exit 1
  fi
  tree="$(cd "$fc" && cargo tree -p bench-fandhe --depth 1 --config "$cfg" 2>&1)"
  if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
    echo "error: fandhe-ai が path patch に解決されていない (${name})" >&2; echo "$tree" >&2; exit 1
  fi
  echo "$tree" >"$OUT/tree-diag-${name}-${LABEL}.txt"
  cp "$fc/$tdir/release/bench-fandhe" "$out_abs/bench-fandhe-diag-${name}-${LABEL}"
  echo "$(sha256_of "$out_abs/bench-fandhe-diag-${name}-${LABEL}")  bench-fandhe-diag-${name}-${LABEL}" >"$OUT/sha-diag-${name}-${LABEL}.txt"
}

# B2: B1 の HEAD から detached worktree を作り、bench パッチ + layout パッチを当てて完全一致を検証する。
B2_ROOT="$(dirname "$B1_ROOT")/diag-wt-${LABEL}-b2"
[[ ! -e "$B2_ROOT" ]] || precheck_fail "worktree ${B2_ROOT} が既に存在する"
git -C "$B1_ROOT" worktree add --detach "$B2_ROOT" HEAD >/dev/null 2>&1 || precheck_fail "worktree ${B2_ROOT} を作れない"
WT_LIST+=("$B2_ROOT")
git -C "$B2_ROOT" apply "$PATCH_FILE" || precheck_fail "B2: bench patch を適用できない"
git -C "$B2_ROOT" apply "$LAYOUT_FILE" || precheck_fail "B2: layout-shift patch を適用できない"
check_patched_tree "$B2_ROOT" "$LAYOUT_FILE"

# 全ビルドを計測の前に済ませる（計測中のビルド並走禁止）。
build_in_tree b0 "$B0_ROOT"
build_in_tree b1 "$B1_ROOT"
build_in_tree b2 "$B2_ROOT"

bin_for() { # <arm>
  case "$1" in
    aa0|aa1|ch1) echo "$OUT/bench-fandhe-diag-b1-${LABEL}" ;;
    rb0|rb1) echo "$OUT/bench-fandhe-diag-b2-${LABEL}" ;;
    o0|o1) echo "$OUT/bench-fandhe-diag-b0-${LABEL}" ;;
    *) echo "error: 未知の腕 $1" >&2; exit 1 ;;
  esac
}
mode_for() { # <arm>
  case "$1" in
    aa0|aa1|rb0|o0) echo reuse ;;
    ch1|rb1|o1) echo chain ;;
    *) echo "error: 未知の腕 $1" >&2; exit 1 ;;
  esac
}
# 3 本のバイナリが互いに異なること（同じなら配置の対照にならない）。
S_B0="$(sha256_of "$(bin_for o0)")"; S_B1="$(sha256_of "$(bin_for aa0)")"; S_B2="$(sha256_of "$(bin_for rb0)")"
if [[ "$S_B0" == "$S_B1" || "$S_B1" == "$S_B2" || "$S_B0" == "$S_B2" ]]; then
  precheck_fail "B0・B1・B2 のバイナリ sha256 が互いに異なっていない"
fi
# B1／B2 の主要シンボル（ハッシュ接尾辞を除いた名前）のアドレスを記録し、少なくとも 1 つが異なることを確かめる。
# r2: 突き合わせは diag_layout_symbols.py（legacy／v0 mangling の正規化）が行う。nm 生出力は証拠として残す。
nm "$(bin_for aa0)" >"$OUT/nm-raw-diag-b1-${LABEL}.txt" 2>/dev/null
nm "$(bin_for rb0)" >"$OUT/nm-raw-diag-b2-${LABEL}.txt" 2>/dev/null
SYM_OUT="$(python3 -I "$HELPER_FILE" --b1 "$OUT/nm-raw-diag-b1-${LABEL}.txt" --b2 "$OUT/nm-raw-diag-b2-${LABEL}.txt" \
  --out-b1 "$OUT/nm-diag-b1-${LABEL}.txt" --out-b2 "$OUT/nm-diag-b2-${LABEL}.txt" 2>>"$SKIP")" \
  || precheck_fail "diag_layout_symbols.py が失敗した（主要シンボルを突き合わせられない）"
LAYOUT_SYM_DIFF="$(echo "$SYM_OUT" | sed -n 's/^layout_symbol_diff=\([0-9]\{1,\}\)$/\1/p' | head -n1)"
LAYOUT_SYM_KEYS="$(echo "$SYM_OUT" | sed -n 's/^layout_symbol_keys=\([0-9]\{1,\}\)$/\1/p' | head -n1)"
if [[ -z "$LAYOUT_SYM_DIFF" || -z "$LAYOUT_SYM_KEYS" || "$LAYOUT_SYM_DIFF" -lt 1 ]]; then
  precheck_fail "B1／B2 の主要シンボルのアドレス差を確認できない（layout_symbol_diff=${LAYOUT_SYM_DIFF:-unset}）"
fi

{
  echo "profile=${PROFILE}"
  echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=${GPU_NAME}"; fi
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "diag_sha256=${DIAG_SHA}"
  echo "patch_sha256=${PATCH_SHA}"
  echo "layout_shift_sha256=${LAYOUT_SHA}"
  echo "rev_stamp_head_b0=${STAMP_B0}"
  echo "rev_stamp_head_b1=${STAMP_B1}"
  for a in "${ARMS[@]}"; do echo "bin_sha256_${a}=$(sha256_of "$(bin_for "$a")")"; done
  echo "runner_sha256=${RUNNER_SHA}"
  echo "layout_helper_sha256=${HELPER_SHA}"
  echo "layout_symbol_keys=${LAYOUT_SYM_KEYS}"
  echo "layout_symbol_diff=${LAYOUT_SYM_DIFF}"
  echo "precheck=assert"
  echo "rounds=${ROUNDS}"
  echo "tf32=off"
} >"$ENV_INFO"

# ---- 計測 ----
gate_ok_once() {
  local l1
  l1="$(ab_cooldown_read_load1)"
  [[ "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1
  [[ "$(awk -v a="$l1" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')" == "1" ]]
}
wait_gate() { # <round> <logfile>。通過なら 0。不通過（30 分待っても load が下がらない）は 1。
  local waited=0 ok=0
  while :; do
    if gate_ok_once; then ok=1; break; fi
    if [[ "$waited" -ge 1800 ]]; then break; fi
    sleep 30; waited=$((waited + 30))
  done
  if [[ "$ok" == "1" ]]; then
    echo "round$1 dev=cpu gate=pass gate_ok=1 threshold=${LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$2"
  else
    echo "round$1 dev=cpu gate=fail gate_ok=0 threshold=${LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$2"
    return 1
  fi
}

run_arm() { # <arm> <round>
  local arm=$1 r=$2 errf res bin mode
  res="$OUT/results-diag-${arm}-${LABEL}-cpu-train.jsonl"
  errf="$OUT/digest-diag-${arm}-${LABEL}-cpu-train-r${r}.log"
  bin="$(bin_for "$arm")"; mode="$(mode_for "$arm")"
  echo "== diag train cpu arm=${arm} mode=${mode} round=${r} =="
  BENCH_PARAM_DIGEST=1 "$bin" --task train --device cpu --mode "$mode" --out "$res" 2>"$errf"
  # shellcheck disable=SC2181
  if [[ $? -ne 0 ]]; then
    echo "task=train dev=cpu arm=${arm} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

# 起動順の回転（DIAG.txt 固定。基準順 L の位置 (r-1) mod 7 から巡回）。
order_for_round() {
  local r=$1 start i out=""
  start=$(( (r - 1) % NARMS ))
  i=0
  while [[ "$i" -lt "$NARMS" ]]; do
    out="${out}${ARMS[$(( (start + i) % NARMS ))]} "
    i=$((i + 1))
  done
  echo "$out"
}

pm_log() { if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$OUT/pmset_therm_$1-diag-${LABEL}.txt" 2>&1 || true; fi; }
for arm in "${ARMS[@]}"; do : >"$OUT/results-diag-${arm}-${LABEL}-cpu-train.jsonl"; done
GATE_LOG="$OUT/load_gate-diag-${LABEL}-cpu.log"
echo "series=official threshold=${LOAD_GATE} rule_threshold=${LOAD_GATE} precheck=assert" >"$GATE_LOG"
ab_round_cooldown_init "$OUT/cooldown-diag-${LABEL}-cpu.log"
pm_log before
for r in 1 2 3 4 5; do
  ab_round_cooldown "$r"
  # 負荷ゲート不通過は判定不能（DIAG の契約）。負荷下で計測を続けず、記録して停止し非 0 終了する。
  if ! wait_gate "$r" "$GATE_LOG"; then
    echo "round${r}: load gate failed; diagnosis impossible, stopping measurement" >>"$SKIP"
    echo "round${r}: load gate failed; stopping" >&2
    ANY_FAILED=$((ANY_FAILED + 1))
    break
  fi
  for arm in $(order_for_round "$r"); do run_arm "$arm" "$r"; done
  echo "run ${r}: $(uptime)" | tee -a "$OUT/uptime-diag-${LABEL}.log"
done
pm_log after
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
