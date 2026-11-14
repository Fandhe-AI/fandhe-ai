#!/usr/bin/env bash
# イシュー #4168: 結線後（Phase 4 後）の main HEAD と v0.14.0 の前後比較を、相手フレームワーク同席で併記計測する両機体共用ランナー。
# 役割: 計測（ビルド・ゲート・5 ラウンド・記録）だけを担う。判定は judge_prepost_0150_4168.py が行う。
# 規則の正本: docs/perf/logs/prepost-0150-4168/RULE.txt（RULE.template.txt を実体化して計測前に単独コミットする。
#   sha256 を env_info へ記録し、判定器が現 RULE と突合する。`<要記入` が残っていれば本スクリプトは拒否する）。
# 派生元: run_ab_bench_fandhe_symmetry.sh（#3266。凍結。セル構成・期待行数・ゲート・ラウンド手順）、
#   run_ab_sme_default_on_4156_cpu.sh（path patch 腕のビルド作法）、bench_fandhe_lock_restore.sh（Cargo.lock の退避・復元）、
#   docs/perf/logs/framework-compare-0.14.0-remeasure/scripts/dgx-run-0140.sh（GB10 の compute プロセス 0 の専有確認）。
# 腕（両腕とも同じ BASE ツリーの同じ bench-fandhe ソース。差はライブラリだけ）:
#   before = registry の fandhe-ai =0.14.0（path patch なし・--locked）
#   after  = `--config patch.crates-io.fandhe-ai.path="<BASE>/crates/facade"`（BASE の main HEAD ソース）
#   相手（candle・burn・Python）は BASE からビルドして両腕で共用する。
# 事前確認（すべて fail-closed。RULE.txt の固定ブロックと実体の完全一致）:
#   - [道具の sha256]: このランナー・判定器・orchestrate.sh・foreign_job_gate.py・ab_foreign_gate.sh（実行中のこのファイル自身も照合）
#   - [期待するゲート定数]: crates/ の `const *ENABLED*: bool` の全定義（継続行・cfg! 等の式も含む）。1 行でも差があれば停止
#     （結線は Phase 4 で入っているため baseline との値の差は consts-diff.txt へ記録するだけ。baseline の全キーの存在は検査する）
#   - 環境変数 FANDHE_AI_*／RUST_AI_* が 1 つでもあれば拒否（腕の既定が黙って変わるため）。RAYON_NUM_THREADS は unset して env_info へ記録
#   - 2 つのブロックは RULE の実体化時に `--print-expected-consts`／`--print-tool-sha256` で BASE から機械的に生成する（ランナーの検査と同じ関数）
# Mac mini のみ: 各ラウンドの前後に他ジョブゲート（#4166 と同じ。ab_foreign_gate.sh 経由の foreign_job_gate.py G1〜G4）。
#   前は FOREIGN でなくなるまで 30 秒間隔・最大 30 分待ち GATE-TIMEOUT で打ち切り、後が FOREIGN なら系列を打ち切る（5/5 完走だけを採る）。GB10 のゲートは従来どおり。
# --smoke（RULE.txt の実体化前の行数スモーク）: RULE.txt を読まず（部分的に埋めた RULE も受け付けない）、固定部分はこのスクリプトの定数
#   （bench_py の sha256・baseline コミット・期待行数）を使う。実ラウンドと同じ手順を 1 ラウンドだけ流す（ビルド〔before は --locked・after は path patch〕・
#   相手・セルのコマンド・環境変数の拒否・ハードロックの owner 確認・機体の同定・負荷／他ジョブ〔Mac mini〕／専有〔GB10〕のゲート・Cargo.lock の復元と差分 0 の確認）。
#   出力は行数（rows／pyrows／skipped）と rc だけ。計測値は表示せず、結果ファイルは成功・失敗を問わず終了時に削除する（trap）。
#   AB_SMOKE_SUMMARY（絶対パス・未存在）に件数・rc・UTC 時刻・機体プロファイルだけの smoke-summary を残す。実ラウンドの出力ディレクトリ
#   （ab4168-<label>）は使わず（ab4168-smoke-<label>）、既存なら拒否する。計測の代わりにはならず、判定器は env_info の smoke=1 を拒否する。
# 呼び出し例:
#   AB_PROFILE=macmini AB_BASE_ROOT=<abs>/base AB_RUNNER_ROOT=<abs>/base AB_HWLOCK_OWNER=<owner> \
#     AB_PY_TORCH=<abs>/venv-torch/bin/python AB_PY_TF=<abs>/venv-tf/bin/python bash run_ab_prepost_0150_4168.sh series-b
#   AB_PROFILE=gb10 ... AB_PY=<abs>/python bash run_ab_prepost_0150_4168.sh gb10
#   行数スモーク: AB_PROFILE=macmini|gb10 ... AB_HWLOCK_OWNER=<owner> AB_SMOKE_SUMMARY=<abs> bash run_ab_prepost_0150_4168.sh --smoke <label>
#     （起動役 orchestrate.sh --smoke がロックの取得・解放つきで呼ぶ。README の手順 2）
#   Linux の事前確認（ビルド・計測なし）: AB_PROFILE=macmini AB_BASE_ROOT=<repo> AB_RUNNER_ROOT=<repo> bash run_ab_prepost_0150_4168.sh --dry-run <label>
# bash 3.2 互換（mapfile・連想配列・eval・jq 不使用）。失敗は skipped ログへ記録し非 0 終了。
set -uo pipefail

DRY=0
SMOKE=0
PRINT=""
case "${1:-}" in
  --dry-run) DRY=1; shift ;;
  --smoke) SMOKE=1; shift ;;
  --print-expected-consts) PRINT=consts; shift ;;
  --print-tool-sha256) PRINT=tools; shift ;;
esac

# 腕の既定は環境変数で黙って変わる（FANDHE_AI_CUDA_GRAPH_INFER=0 なら after が OFF、=1 なら before が ON）。
# unset で黙って直さず、接頭辞 FANDHE_AI_／RUST_AI_ の変数が 1 つでもあれば拒否する（print モード・dry-run でも同じ）。
PREFIXED_ENV="$(env | cut -d= -f1 | grep -E '^(FANDHE_AI_|RUST_AI_)' || true)"
if [[ -n "$PREFIXED_ENV" ]]; then
  echo "error: 環境変数 FANDHE_AI_*／RUST_AI_* が設定されている（腕の既定が変わるため拒否。名前: $(echo "$PREFIXED_ENV" | tr '\n' ' ')）" >&2; exit 1
fi
RAYON_PRIOR='unset'
if [[ -n "${RAYON_NUM_THREADS+x}" ]]; then RAYON_PRIOR='set'; fi
unset RAYON_NUM_THREADS

LABEL=${1:-}
PROFILE=${AB_PROFILE:-}
ROUNDS=${AB_ROUNDS:-5}

# 外部入力のパスは絶対パスに限り、引用符・バックスラッシュ・改行を拒否する（A03。cargo の --config へ埋め込むため厳格に）。
validate_abs() { # <var_name> <path>
  if [[ -z "$2" || "$2" != /* || "$2" == *\"* || "$2" == *\\* || "$2" == *$'\n'* || "$2" == *\'* ]]; then
    echo "error: $1 は引用符・バックスラッシュ・改行を含まない絶対パス必須" >&2; exit 1
  fi
}
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
FC_REL=scripts/bench/framework-compare
LOGS_REL=docs/perf/logs/prepost-0150-4168

# RULE が固定する道具（RUNNER_ROOT 相対。この順で出力する）。ランナーが実行・source するものと、判定器・起動役・他ジョブゲート。
# 判定器側の同じ一覧は judge_prepost_0150_4168.py の TOOL_FILES。変えるときは両方とテンプレートの手順を揃える。
TOOL_FILES="$FC_REL/run_ab_prepost_0150_4168.sh
$FC_REL/judge_prepost_0150_4168.py
$LOGS_REL/orchestrate.sh
docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py
$FC_REL/ab_foreign_gate.sh"
tool_sha_lines() { # <root> -> 'sha256  相対パス' の行（RULE の [道具の sha256] ブロックと完全一致させる）
  local rel
  while IFS= read -r rel; do
    [[ -f "$1/$rel" ]] || { echo "error: 道具ファイルが無い: $rel" >&2; return 1; }
    printf '%s  %s\n' "$(sha256_of "$1/$rel")" "$rel"
  done <<EOF
$TOOL_FILES
EOF
}

# ゲート定数の抽出（crates/ 配下の `const *ENABLED*: bool` 定義を 'path:NAME=value' で整列出力する）。
# 入力: 1 ファイル分のソース（標準入力）と -v P=<path>。ベース側（git show）・作業ツリー側とも同じ awk を通す。
#  - `;` まで継続行をつなぐ。値は空白を全部詰めて式のまま記録する（`cfg!(...)` 等。true／false 以外も同様）。
#  - 行頭が `//` のコメント行は無視する（定義を引用するコメントを拾わない）。行末の `//` コメントは切り落とす。
#  - 検出は緩く（`const <名前に ENABLED を含む>`）、構文解析は厳しく: 型が bool なのに解析できない定義・`;` に届かない定義は
#    `UNPARSABLE` 行を出す（呼び出し側が precheck_fail にする。取りこぼしを fail-closed にする）。bool 以外の型は対象外。
# POSIX awk のみ（macOS の BSD awk でも動く）。
# shellcheck disable=SC2016  # awk プログラム。$ は awk 側のもの（シェルでは展開しない）
CONST_AWK='
function trim(x) { gsub(/^[ \t]+|[ \t]+$/, "", x); return x }
function flush(a,   st, rest, nm, val, i) {
  st = a
  sub(/;.*$/, "", st)
  gsub(/[ \t]+/, " ", st)
  st = trim(st)
  i = index(st, "const ")
  st = substr(st, i + 6)
  nm = st
  sub(/[^A-Za-z0-9_].*$/, "", nm)
  rest = substr(st, length(nm) + 1)
  if (rest ~ /^ ?: ?bool ?= /) {
    val = rest
    sub(/^ ?: ?bool ?= /, "", val)
    gsub(/[ \t]/, "", val)
    if (val == "") { print "UNPARSABLE " P ":" nm; return }
    print P ":" nm "=" val
  } else if (rest ~ /:[ ]*bool/ || rest !~ /:/) {
    print "UNPARSABLE " P ":" nm
  }
}
{
  line = $0
  t = line; sub(/^[ \t]+/, "", t)
  if (t ~ /^\/\//) next
  if (acc == "") {
    line = " " line
    if (!match(line, /[^A-Za-z0-9_]const[ \t]+[A-Za-z0-9_]*ENABLED/)) next
    line = substr(line, RSTART)
  }
  sub(/[ \t]*\/\/.*$/, "", line)
  acc = (acc == "") ? line : acc " " line
  if (index(acc, ";") > 0) { flush(acc); acc = "" }
}
END { if (acc != "") print "UNPARSABLE " P ":(; に届かない)" }
'
consts_of() { # <git-rev|WORKTREE>  -> 'path:NAME=value' を整列して出力。解析不能があれば UNPARSABLE 行を含めて返す
  local f
  {
    if [[ "$1" == "WORKTREE" ]]; then
      ( cd "$BASE_ROOT" && grep -rlE 'const[[:space:]]+[A-Za-z0-9_]*ENABLED' crates --include='*.rs' ) | while IFS= read -r f; do
        awk -v P="$f" "$CONST_AWK" <"$BASE_ROOT/$f"
      done
    else
      # pathspec は glob を使わずディレクトリ crates だけを渡し、.rs の絞り込みは後段の grep で行う
      # （GIT_GLOB_PATHSPECS・GIT_LITERAL_PATHSPECS などの環境で pathspec の解釈が変わり、baseline 側が空になるのを避ける）。
      git -C "$BASE_ROOT" grep -lE 'const[[:space:]]+[A-Za-z0-9_]*ENABLED' "$1" -- crates | sed -e "s/^$1://" | grep -E '\.rs$' | while IFS= read -r f; do
        git -C "$BASE_ROOT" show "$1:$f" | awk -v P="$f" "$CONST_AWK"
      done
    fi
  } | LC_ALL=C sort
}
# RULE の固定ブロック（`# >>> <名前>` と `# <<< <名前>` の間の行をそのまま）を取り出す。
rule_block() { # <rule-file> <名前>
  awk -v b="# >>> $2" -v e="# <<< $2" '$0 == e { on = 0 } on { print } $0 == b { on = 1 }' "$1"
}

if [[ -z "$PRINT" ]]; then
  if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "usage: AB_PROFILE=macmini|gb10 $0 [--dry-run] <label>  (label: [A-Za-z0-9._-]+)" >&2
    echo "       AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> $0 --print-expected-consts | --print-tool-sha256" >&2; exit 1
  fi
  if [[ "$ROUNDS" != "5" ]]; then echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2; exit 1; fi
  case "$PROFILE" in
    macmini) RULE_LOAD_GATE=3.0 ;;
    # gb10 は対話しない ssh から起動されると PATH に cargo・nvcc が無い。cargo tree（腕の解決確認）より前に足す。
    gb10) RULE_LOAD_GATE=1.0; export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}" ;;
    *) echo "error: AB_PROFILE は macmini か gb10（got: ${PROFILE:-unset}）" >&2; exit 1 ;;
  esac
fi
validate_abs AB_BASE_ROOT "${AB_BASE_ROOT:-}"
validate_abs AB_RUNNER_ROOT "${AB_RUNNER_ROOT:-}"
BASE_ROOT=$AB_BASE_ROOT; RUNNER_ROOT=$AB_RUNNER_ROOT
if [[ "$SMOKE" == "1" ]]; then
  validate_abs AB_SMOKE_SUMMARY "${AB_SMOKE_SUMMARY:-}"
  if [[ -e "$AB_SMOKE_SUMMARY" ]]; then echo "error: AB_SMOKE_SUMMARY が既に存在する（上書き禁止）" >&2; exit 1; fi
  [[ -d "$(dirname "$AB_SMOKE_SUMMARY")" ]] || { echo "error: AB_SMOKE_SUMMARY の親ディレクトリが無い" >&2; exit 1; }
fi

# RULE の [期待するゲート定数]・[道具の sha256] ブロックを機械的に生成する入口（RULE.txt の実体化時に BASE から使う。ランナーの検査と同じ関数）。
if [[ "$PRINT" == "consts" ]]; then
  out="$(consts_of WORKTREE)"
  if [[ -z "$out" ]] || echo "$out" | grep -q '^UNPARSABLE'; then
    echo "error: ゲート定数を抽出できない／解析不能の定義がある" >&2; echo "$out" | grep '^UNPARSABLE' >&2; exit 1
  fi
  echo "$out"; exit 0
fi
if [[ "$PRINT" == "tools" ]]; then tool_sha_lines "$RUNNER_ROOT"; exit $?; fi

if [[ "$DRY" == "0" ]]; then
  if [[ "$PROFILE" == "macmini" ]]; then
    validate_abs AB_PY_TORCH "${AB_PY_TORCH:-}"; validate_abs AB_PY_TF "${AB_PY_TF:-}"
    PY_PT=$AB_PY_TORCH; PY_TF=$AB_PY_TF
    brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    if [[ "$brand" != "Apple M4" ]]; then echo "error: macmini プロファイルは 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1; fi
  else
    validate_abs AB_PY "${AB_PY:-}"; PY_PT=$AB_PY; PY_TF=$AB_PY
    if [[ "$(uname -m)" != "aarch64" ]]; then echo "error: gb10 は aarch64 のみ" >&2; exit 1; fi
  fi
  # 実機ロック（owner 文字列が一致すること。取得・解放は orchestrate.sh）。ロックなしの値は採らない。
  OWNER="${AB_HWLOCK_OWNER:-}"
  [[ -n "$OWNER" && "$OWNER" != *$'\n'* ]] || { echo "error: AB_HWLOCK_OWNER 必須" >&2; exit 1; }
  [[ "$(cat "${HOME}/work/.fandhe-hw-lock/owner" 2>/dev/null)" == "$OWNER" ]] || { echo "error: 実機ロックの owner が一致しない（LOCK-NOT-MINE）" >&2; exit 1; }
fi

RULE_FILE="$RUNNER_ROOT/$LOGS_REL/RULE.txt"
BENCH_PY_REL=docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py
BENCH_PY_SHA=9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b
BASELINE_COMMIT=18a7f4b87fe61029a179cb1922cc82928440f2c9   # 0.14.0 スコアボードの計測腕（#4063 マージ）
FACADE_PATH="$BASE_ROOT/crates/facade"
validate_abs FACADE_PATH "$FACADE_PATH"

# smoke の要約（件数・rc・UTC 時刻・機体プロファイルだけ。計測値は入れない）。結果ファイルは出力ディレクトリごと削除する。
SM_B=NA; SM_A=NA; SM_O=NA; SM_P=NA; SM_UNEXP=NA; SMOKE_BAD=0
smoke_finish() { # <rc>
  local n=NA
  if [[ -n "${SKIP:-}" && -f "$SKIP" ]]; then n=$(wc -l <"$SKIP" | tr -d ' '); fi
  printf 'smoke=1\nprofile=%s\nutc=%s\nrows_before=%s\nrows_after=%s\nrows_opp=%s\npyrows=%s\nskipped=%s\nunexpected_failures=%s\nrc=%s\n' \
    "$PROFILE" "$(date -u +%FT%TZ)" "$SM_B" "$SM_A" "$SM_O" "$SM_P" "$n" "$SM_UNEXP" "$1" >"$AB_SMOKE_SUMMARY"
  echo "SMOKE-SUMMARY profile=${PROFILE} rows(before/after/opp)=${SM_B}/${SM_A}/${SM_O} pyrows=${SM_P} skipped=${n} unexpected=${SM_UNEXP} rc=$1"
  rm -rf "$OUT"
}
if [[ "$DRY" == "1" ]]; then
  OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
elif [[ "$SMOKE" == "1" ]]; then
  OUT="$RUNNER_ROOT/$FC_REL/results/raw/ab4168-smoke-${LABEL}"
  if [[ -e "$OUT" ]]; then echo "error: smoke の出力が既に存在する（差し替え禁止）: $OUT" >&2; exit 1; fi
  if [[ -e "$RUNNER_ROOT/$FC_REL/results/raw/ab4168-${LABEL}" ]]; then echo "error: 実ラウンドの出力ディレクトリ ab4168-${LABEL} が存在する（smoke は同名ラベルを使わない）" >&2; exit 1; fi
  case "$AB_SMOKE_SUMMARY" in "$OUT"/*) echo "error: AB_SMOKE_SUMMARY は smoke の出力ディレクトリの外に置く" >&2; exit 1 ;; esac
  mkdir -p "$OUT"
  trap 'smoke_finish "$?"' EXIT
else
  OUT="$RUNNER_ROOT/$FC_REL/results/raw/ab4168-${LABEL}"
  if [[ -e "$OUT" ]]; then echo "error: ラベル ${LABEL} の出力が既に存在する（差し替え禁止）: $OUT" >&2; exit 1; fi
  mkdir -p "$OUT"
fi
SKIP="$OUT/skipped.log"; : >"$SKIP"
precheck_fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

# ---- 事前確認 1: RULE.txt（未記入の拒否・after SHA と .rev-stamp の照合・道具の sha256・期待するゲート定数）
stamp_of() { if [[ -f "$1/.rev-stamp" ]]; then head -n1 "$1/.rev-stamp"; fi; }
RULE_SHA=none; EXP_CONSTS_SHA=none; TOOL_CHECK=ok; STAMP_RUNNER=""
if [[ "$SMOKE" == "1" ]]; then
  # RULE.txt は読まない（未実体化・部分記入の RULE は受け付けない）。固定部分はこのスクリプトの定数を使う。
  TOOL_CHECK=skipped
  STAMP_RUNNER="$(stamp_of "$RUNNER_ROOT")"
  [[ "$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")" == "$BENCH_PY_SHA" ]] || precheck_fail "bench_py.py の sha256 が登録値と不一致"
  [[ "$(sha256_of "${BASH_SOURCE[0]}")" == "$(sha256_of "$RUNNER_ROOT/$FC_REL/run_ab_prepost_0150_4168.sh")" ]] \
    || precheck_fail "実行中のランナーが AB_RUNNER_ROOT 側のランナーと一致しない"
  echo "smoke: RULE.txt・after SHA・道具の sha256 の照合は行わない（行数スモーク。計測の代わりにならない）"
else
[[ -f "$RULE_FILE" ]] || precheck_fail "RULE.txt が無い（RULE.template.txt の実体化と単独コミットが先）"
if grep -q '<要記入' "$RULE_FILE"; then precheck_fail "RULE.txt に <要記入 が残っている"; fi
RULE_SHA="$(sha256_of "$RULE_FILE")"
rule_sha_of() { grep -E "^- $1[[:space:]]*=[[:space:]]*[0-9a-f]{40}([[:space:]]|\$)" "$RULE_FILE" | head -n1 | grep -oE '[0-9a-f]{40}' | head -n1; }
EXP_AFTER="$(rule_sha_of after)"
[[ -n "$EXP_AFTER" ]] || precheck_fail "RULE.txt から after SHA（'- after = <40hex>'）を読めない"
if [[ "$DRY" == "0" ]]; then
  [[ "$(stamp_of "$BASE_ROOT")" == "$EXP_AFTER" ]] || precheck_fail "BASE の .rev-stamp が RULE の after SHA と不一致"
  STAMP_RUNNER="$(stamp_of "$RUNNER_ROOT")"
  [[ -n "$STAMP_RUNNER" ]] || precheck_fail "runner の .rev-stamp が無い"
  [[ "$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")" == "$BENCH_PY_SHA" ]] || precheck_fail "bench_py.py の sha256 が登録値と不一致"
else
  echo "dry-run: .rev-stamp・bench_py.py の照合は実機でのみ行う（未検証）"
fi
# 道具の sha256: RULE のブロックと実ファイルの完全一致（1 行でも違えば拒否）。実行中のこのスクリプト自身も RUNNER_ROOT 側と同一であること
# （別の場所の改変コピーを実行しても通さない）。
# ブロックは `$(...)` で末尾の改行が落ちるため、空行（判定器は行をそのまま数える）を先に拒否して両者の読みを揃える。
if rule_block "$RULE_FILE" tool-sha256 | grep -q '^$'; then precheck_fail "[道具の sha256] ブロックに空行がある"; fi
if rule_block "$RULE_FILE" expected-consts | grep -q '^$'; then precheck_fail "[期待するゲート定数] ブロックに空行がある"; fi
EXP_TOOLS="$(rule_block "$RULE_FILE" tool-sha256)"
[[ -n "$EXP_TOOLS" ]] || precheck_fail "RULE.txt に [道具の sha256] ブロック（# >>> tool-sha256）が無い"
ACT_TOOLS="$(tool_sha_lines "$RUNNER_ROOT")" || precheck_fail "道具ファイルを読めない"
[[ "$ACT_TOOLS" == "$EXP_TOOLS" ]] || precheck_fail "道具の sha256 が RULE と不一致（--print-tool-sha256 の出力と RULE のブロックを比べる）"
[[ "$(sha256_of "${BASH_SOURCE[0]}")" == "$(sha256_of "$RUNNER_ROOT/$FC_REL/run_ab_prepost_0150_4168.sh")" ]] \
  || precheck_fail "実行中のランナーが AB_RUNNER_ROOT 側のランナーと一致しない"
echo "tool-sha256=ok ($(echo "$EXP_TOOLS" | wc -l | tr -d ' ') 件)"

fi

# ---- 事前確認 2: ゲート定数（RULE の [期待するゲート定数] と完全一致。増減・値の違いのどれでも停止）
# 結線は Phase 4 で入った（STEEL・SME・推論 Graph・jpw1）。「結線あり」は想定内なので、baseline との値の差は consts-diff.txt へ記録するだけで止めない。
CONSTS_RAW="$(consts_of WORKTREE)"
if echo "$CONSTS_RAW" | grep -q '^UNPARSABLE'; then precheck_fail "解析できない const *ENABLED*: bool の定義がある（取りこぼし）: $(echo "$CONSTS_RAW" | grep '^UNPARSABLE' | tr '\n' ' ')"; fi
[[ -n "$CONSTS_RAW" ]] || precheck_fail "ゲート定数を抽出できない"
printf '%s\n' "$CONSTS_RAW" >"$OUT/consts.txt"
if [[ "$SMOKE" == "1" ]]; then
  echo "gate-consts=extracted ($(echo "$CONSTS_RAW" | wc -l | tr -d ' ') 件。smoke のため RULE との照合なし)"
else
EXP_CONSTS="$(rule_block "$RULE_FILE" expected-consts)"
[[ -n "$EXP_CONSTS" ]] || precheck_fail "RULE.txt に [期待するゲート定数] ブロック（# >>> expected-consts）が無い"
if [[ "$CONSTS_RAW" != "$EXP_CONSTS" ]]; then
  printf '%s\n' "$EXP_CONSTS" >"$OUT/consts-expected.txt"
  precheck_fail "ゲート定数が RULE の期待と不一致（差: $(diff "$OUT/consts-expected.txt" "$OUT/consts.txt" | grep '^[<>]' | tr '\n' ' ')）"
fi
EXP_CONSTS_SHA="$(printf '%s\n' "$EXP_CONSTS" | { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; } | awk '{print $1}')"
echo "gate-consts=ok ($(echo "$CONSTS_RAW" | wc -l | tr -d ' ') 件・RULE と完全一致)"
fi
CONSTS_DIFF_N=0
if git -C "$BASE_ROOT" cat-file -e "$BASELINE_COMMIT^{commit}" 2>/dev/null; then
  CB_RAW="$(consts_of "$BASELINE_COMMIT")"
  if echo "$CB_RAW" | grep -q '^UNPARSABLE'; then precheck_fail "baseline 側に解析できない定義がある: $(echo "$CB_RAW" | grep '^UNPARSABLE' | tr '\n' ' ')"; fi
  printf '%s\n' "$CB_RAW" >"$OUT/consts-baseline.txt"
  [[ -s "$OUT/consts-baseline.txt" ]] || precheck_fail "baseline のゲート定数を抽出できない"
  val_of() { awk -v k="$2" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$1"; } # <file> <path:NAME>
  # baseline の定数が現在側で削除・抽出対象外化されても見逃さない（baseline の全キーが現在側に在ることを検査）
  while IFS= read -r bline; do
    bkey=${bline%%=*}
    [[ -n "$(val_of "$OUT/consts.txt" "$bkey")" ]] || precheck_fail "baseline のゲート定数が現在側に無い（削除・抽出対象外化）: $bkey"
  done <"$OUT/consts-baseline.txt"
  # baseline からの値の変化・追加は記録のみ（情報。止める理由にしない）
  : >"$OUT/consts-diff.txt"
  while IFS= read -r line; do
    key=${line%%=*}; val=${line#*=}; bval="$(val_of "$OUT/consts-baseline.txt" "$key")"
    if [[ -z "$bval" ]]; then echo "$key (baseline に無い) -> $val" >>"$OUT/consts-diff.txt"
    elif [[ "$bval" != "$val" ]]; then echo "$key $bval -> $val" >>"$OUT/consts-diff.txt"; fi
  done <"$OUT/consts.txt"
  CONSTS_DIFF_N=$(wc -l <"$OUT/consts-diff.txt" | tr -d ' ')
  echo "consts-diff=${CONSTS_DIFF_N} 件（baseline からの変化・追加。記録のみ）"
  # bench-fandhe の差分は記録する（機械的に断定はしない。README「事前確認」で人が読む）。両腕は同じソースを使う。
  git -C "$BASE_ROOT" diff "$BASELINE_COMMIT" HEAD --stat -- "$FC_REL/bench-fandhe" >"$OUT/bench-fandhe-diff-stat.txt" 2>&1 || true
else
  [[ "$DRY" == "1" ]] || precheck_fail "BASE に baseline コミット ${BASELINE_COMMIT} が無い（git archive ではなく clone を使う）"
  echo "dry-run: baseline コミットが無いため baseline との差の記録を省略（RULE との完全一致は検査済み）"
fi

# ---- ビルド・腕の解決確認（Cargo.lock は path patch で書き換わるため退避し、終了時に復元して差分 0 を確かめる）
PATCH_CONFIG="patch.crates-io.fandhe-ai.path=\"${FACADE_PATH}\""
LOCK_CLEAN_REL="$FC_REL/Cargo.lock"
LOCK_COPY="$(mktemp)"
cp "$BASE_ROOT/$LOCK_CLEAN_REL" "$LOCK_COPY" || { echo "error: Cargo.lock の退避に失敗" >&2; exit 1; }
# 復元に失敗したら退避ファイルを残して保存先とエラーを示し非 0 を返す（成功時のみ trap が退避を削除する）。
restore_lock() {
  cp "$LOCK_COPY" "$BASE_ROOT/$LOCK_CLEAN_REL" || { echo "error: Cargo.lock の復元に失敗。退避ファイルを残した: $LOCK_COPY（手動で $BASE_ROOT/$LOCK_CLEAN_REL へ戻す）" >&2; return 1; }
}
cleanup_exit() {
  local rc=$?
  if restore_lock; then rm -f "$LOCK_COPY"; fi
  if [[ "$DRY" == "1" ]]; then rm -rf "$OUT"; fi
  if [[ "$SMOKE" == "1" ]]; then smoke_finish "$rc"; fi
}
trap cleanup_exit EXIT
FC_DIR="$BASE_ROOT/$FC_REL"

tree_checks() { # before/after の fandhe-ai 解決を検査
  local t
  t="$(cd "$FC_DIR" && cargo tree --locked -p bench-fandhe --depth 2 --target all 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-before.txt"
  echo "$t" | grep -qE 'fandhe-ai v0\.14\.0$' || precheck_fail "before の fandhe-ai が registry 0.14.0 に解決されない"
  t="$(cd "$FC_DIR" && cargo tree -p bench-fandhe --depth 2 --target all --config "$PATCH_CONFIG" 2>&1)" || { echo "$t" | tail -5; return 2; }
  echo "$t" >"$OUT/tree-after.txt"
  # after: fandhe-ai と fandhe-ai-* のすべてが path に解決される（registry 形 'fandhe-ai... vX.Y.Z' で行が終わる行が 0 件）
  if echo "$t" | grep -E 'fandhe-ai[a-z-]* v[0-9.]+$' >/dev/null; then precheck_fail "after に registry へ解決された fandhe-ai 系クレートが残る（比較が無意味になる）"; fi
  local need
  for need in fandhe-ai fandhe-ai-tensor-core fandhe-ai-autodiff fandhe-ai-backend-cpu; do
    echo "$t" | grep -qE "${need} v[0-9.]+ \(${BASE_ROOT}/crates/" || precheck_fail "after で ${need} が BASE の crates/ へ解決されない"
  done
  return 0
}
tc=0; tree_checks || tc=$?
# after 側の cargo tree（--locked なし）は Cargo.lock を patched グラフへ書き換える。before ビルド（--locked）の前に必ず戻す。
restore_lock || exit 1
if [[ "$tc" -ne 0 ]]; then
  if [[ "$DRY" == "1" ]]; then echo "dry-run: cargo tree を実行できない環境（rc=$tc）。腕の解決確認は未検証"; else precheck_fail "cargo tree の実行に失敗"; fi
else
  echo "tree-checks=ok"
fi

if [[ "$DRY" == "1" ]]; then
  echo "dry-run: 以降（ビルド・ゲート・計測）は実機でのみ実行する。計画:"
  echo "  build before: cargo build --release --locked -p bench-fandhe"
  echo "  build opp   : bench-candle, bench-burn（gb10 は --no-default-features --features cuda）"
  echo "  build after : cargo build --release -p bench-fandhe --config '${PATCH_CONFIG}'（Cargo.lock は trap で復元）"
  echo "  rounds=5 ゲート(${PROFILE}) load1<${RULE_LOAD_GATE}（macmini は各ラウンドの前後に他ジョブゲート foreign_job_gate.py〈ab_foreign_gate.sh 経由〉。待ちは最大 30 分）"
  echo "DRY-RUN-OK"
  exit 0
fi

build() { # <target-dir-name> <crate> [cargo args...]（BASE の framework-compare で --locked ビルド）
  local arm=$1 crate=$2 tdir="$OUT/target-$1"; shift 2
  ( cd "$FC_DIR" && cargo build --release --locked -p "$crate" --target-dir "$tdir" "$@" ) 2>"$OUT/build-${arm}-${crate}.err" \
    || { tail -40 "$OUT/build-${arm}-${crate}.err"; echo "BUILD FAILED ${arm} ${crate}" >>"$SKIP"; exit 1; }
}
build before bench-fandhe
if [[ "$PROFILE" == "gb10" ]]; then
  build opp-cuda bench-candle --no-default-features --features cuda
  build opp-cuda bench-burn --no-default-features --features cuda
  OPP_DIR="$OUT/target-opp-cuda/release"
else
  build opp bench-candle
  build opp bench-burn
  OPP_DIR="$OUT/target-opp/release"
fi
# after は --locked を付けない（path patch の解決で Cargo.lock が更新されるため。trap が終了時に復元する）
( cd "$FC_DIR" && cargo build --release -p bench-fandhe --target-dir "$OUT/target-after" --config "$PATCH_CONFIG" ) 2>"$OUT/build-after-bench-fandhe.err" \
  || { tail -40 "$OUT/build-after-bench-fandhe.err"; echo "BUILD FAILED after bench-fandhe" >>"$SKIP"; exit 1; }
BIN_BEFORE="$OUT/target-before/release/bench-fandhe"; BIN_AFTER="$OUT/target-after/release/bench-fandhe"
[[ -x "$BIN_BEFORE" && -x "$BIN_AFTER" ]] || precheck_fail "bench-fandhe のバイナリが無い"
[[ "$(sha256_of "$BIN_BEFORE")" != "$(sha256_of "$BIN_AFTER")" ]] || precheck_fail "before と after のバイナリが同一（腕が分かれていない）"
echo "precheck=ok"

{
  echo "profile=${PROFILE}"; echo "label=${LABEL}"; echo "host=masked"
  echo "uname=$(uname -srm)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "cpu_brand=$(sysctl -n machdep.cpu.brand_string)"; else echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"; fi
  echo "smoke=${SMOKE}"; echo "rule_sha256=${RULE_SHA}"; echo "tool_sha256_check=${TOOL_CHECK}"; echo "expected_consts_sha256=${EXP_CONSTS_SHA}"; echo "consts_diff_lines=${CONSTS_DIFF_N}"
  echo "fandhe_ai_rust_ai_env_vars=0"; echo "rayon_num_threads=unset"; echo "rayon_num_threads_prior=${RAYON_PRIOR}"
  echo "bench_py_sha256=$(sha256_of "$RUNNER_ROOT/$BENCH_PY_REL")"
  echo "rev_stamp_after=$(stamp_of "$BASE_ROOT")"; echo "rev_stamp_runner=${STAMP_RUNNER}"; echo "baseline_commit=${BASELINE_COMMIT}"
  echo "cargo_lock_sha256=$(sha256_of "$LOCK_COPY")"
  echo "bin_sha256_before=$(sha256_of "$BIN_BEFORE")"; echo "bin_sha256_after=$(sha256_of "$BIN_AFTER")"
  echo "bin_sha256_candle=$(sha256_of "$OPP_DIR/bench-candle")"; echo "bin_sha256_burn=$(sha256_of "$OPP_DIR/bench-burn")"
  echo "python_pt=$("$PY_PT" -c 'import sys,torch;print(sys.version.split()[0],"torch",torch.__version__)' 2>&1 | tail -1)"
  echo "python_tf=$("$PY_TF" -c 'import tensorflow as t;print("tf",t.__version__)' 2>&1 | tail -1)"
  echo "python_scipy=$("$PY_PT" -c 'import scipy;print("scipy",scipy.__version__)' 2>&1 | tail -1)"
  if [[ "$PROFILE" == "macmini" ]]; then echo "foreign_gate=on"; else echo "foreign_gate=off"; fi
  echo "gate=1"; echo "rounds=${ROUNDS}"; echo "utc=$(date -u +%FT%TZ)"
} >"$OUT/env_info.txt"

load1() {
  local l
  l=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
  if [[ -z "$l" && -r /proc/loadavg ]]; then l=$(awk '{print $1}' /proc/loadavg); fi
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
# GB10 の専有確認（0.14.0 RULE と同じ fail-closed）: 取得失敗は EXCLUSIVE-INVALID、1 件でもあれば専有未達。
compute_apps_zero() { # <記録先>
  local apps rc
  apps="$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>&1)"; rc=$?
  printf '%s rc=%s\n%s\n' "$(date -u +%FT%TZ)" "$rc" "$apps" >>"$1"
  [[ "$rc" -eq 0 ]] || return 2
  [[ -z "$apps" ]] || return 1
  return 0
}
# ゲート: load1 < 閾値を 30 秒間隔・最大 30 分待つ（GB10 は GPU 使用率 0% 3 サンプルと compute プロセス 0 も要求）。
gate_wait() { # <round>
  local waited=0 l ok rc
  if [[ "$PROFILE" == "macmini" ]]; then
    # load1 < 閾値と他ジョブ CLEAN を同じループで待つ（30 秒間隔・最大 30 分）。gate.log の文法: 通過は従来の `gate=pass` 行 + `foreign-before=clean` 行。
    # 打ち切りは GATE-TIMEOUT、ゲートの失敗は GATE-INVALID（どちらも判定器が系列を採らない）。
    fg_wait_before "r$1" "$RULE_LOAD_GATE"
    case "$FG_STATUS" in
      pass)
        echo "round$1 gate=pass threshold=${RULE_LOAD_GATE} waited_s=${FG_WAITED} load1=${FG_L1}" >>"$OUT/gate.log"
        echo "round$1 foreign-before=clean" >>"$OUT/gate.log"; return 0 ;;
      timeout) echo "round$1 GATE-TIMEOUT reason=${FG_REASON} load1=${FG_L1}" >>"$OUT/gate.log"; return 1 ;;
      *) echo "round$1 GATE-INVALID reason=${FG_REASON}" >>"$OUT/gate.log"; return 2 ;;
    esac
  fi
  if [[ "$PROFILE" == "gb10" ]]; then
    rc=0; compute_apps_zero "$OUT/compute_apps.log" || rc=$?
    if [[ "$rc" -eq 2 ]]; then echo "round$1 EXCLUSIVE-INVALID" >>"$OUT/gate.log"; return 2; fi
    if [[ "$rc" -eq 1 ]]; then echo "round$1 EXCLUSIVE-UNMET compute-process-present" >>"$OUT/gate.log"; return 3; fi
  fi
  while :; do
    l="$(load1)"
    if [[ ! "$l" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then echo "round$1 GATE-INVALID load1-not-numeric" >>"$OUT/gate.log"; return 2; fi
    ok="$(awk -v a="$l" -v t="$RULE_LOAD_GATE" 'BEGIN{print (a<t)?1:0}')"
    if [[ "$ok" == "1" && "$PROFILE" == "gb10" ]] && ! gpu_idle3; then ok=0; fi
    if [[ "$ok" == "1" ]]; then
      echo "round$1 gate=pass threshold=${RULE_LOAD_GATE} waited_s=${waited} load1=${l}" >>"$OUT/gate.log"; return 0
    fi
    if [[ "$waited" -ge 1800 ]]; then echo "round$1 GATE-TIMEOUT load1=${l}" >>"$OUT/gate.log"; return 1; fi
    sleep 30; waited=$((waited + 30))
  done
}
: >"$OUT/gate.log"
if [[ "$PROFILE" == "macmini" ]]; then
  # 他ジョブの前提ゲート（#4166 と同じ。ab_foreign_gate.sh は foreign_job_gate.py を呼ぶだけで中身は変えない。どちらも sha256 を RULE が固定する）。
  # shellcheck source=/dev/null
  . "$RUNNER_ROOT/$FC_REL/ab_foreign_gate.sh"
  FOREIGN_GATE="$RUNNER_ROOT/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py"
  fg_require "$FOREIGN_GATE" "$OUT/foreign-gate" || precheck_fail "他ジョブゲートの準備に失敗"
fi

if [[ "$PROFILE" == "macmini" ]]; then
  GEMM_DEV="cpu:256,512,1024,2048 metal:256,512,1024,2048,4096"; MDEVS="cpu metal"
  EXP_ARM=26; EXP_OPP=22; EXP_PY=32
else
  GEMM_DEV="cuda:256,512,1024,2048,4096 cpu:256,512,1024,2048,4096"; MDEVS="cuda cpu"
  EXP_ARM=28; EXP_OPP=28; EXP_PY=28
fi
ANY_FAILED=0
PYF=""
cell() { # <bin> <out> <task> <dev> <size> <mode>   失敗は skipped へ記録（捏造しない）。--diag-* は渡さない。
  local errf="$OUT/err.tmp"
  local crc=0
  if [[ "$SMOKE" == "1" ]]; then "$1" --task "$3" --device "$4" --size "$5" --mode "$6" --out "$2" >/dev/null 2>"$errf" || crc=$?
  else "$1" --task "$3" --device "$4" --size "$5" --mode "$6" --out "$2" 2>"$errf" || crc=$?; fi
  if [[ "$crc" -ne 0 ]]; then
    echo "$(basename "$1") $3 $4 $5 $6: $(tr '\n' ' ' <"$errf")" >>"$SKIP"; ANY_FAILED=$((ANY_FAILED + 1))
  fi
}
each_cell() { # <fn>: 判定セルを (task dev size) で列挙して fn を呼ぶ
  local spec dev sizes n
  for spec in $GEMM_DEV; do
    dev=${spec%%:*}; sizes=${spec#*:}
    for n in ${sizes//,/ }; do "$1" gemm "$dev" "$n"; done
  done
  for dev in $MDEVS; do "$1" train "$dev" 64; "$1" infer "$dev" 64; done
}
pyrun() { # <python> args...
  local py=$1; shift
  if ! "$py" "$RUNNER_ROOT/$BENCH_PY_REL" "$@" --out "$PYF" >/dev/null 2>"$OUT/err.tmp"; then
    echo "py $*: $(tail -3 "$OUT/err.tmp" | tr '\n' ' ')" >>"$SKIP"; ANY_FAILED=$((ANY_FAILED + 1))
  fi
}
RD=""
opp_cell() { local b; for b in bench-candle bench-burn; do cell "$OPP_DIR/$b" "$RD/results-opp.jsonl" "$1" "$2" "$3" fresh; done; }
fandhe_cell() {
  local mode
  for mode in fresh reuse; do
    if (( R % 2 == 1 )); then
      cell "$BIN_BEFORE" "$RD/results-before.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_AFTER" "$RD/results-after.jsonl" "$1" "$2" "$3" "$mode"
    else
      cell "$BIN_AFTER" "$RD/results-after.jsonl" "$1" "$2" "$3" "$mode"; cell "$BIN_BEFORE" "$RD/results-before.jsonl" "$1" "$2" "$3" "$mode"
    fi
  done
}
py_cell() { # task dev size
  local fw
  if [[ "$PROFILE" == "macmini" ]]; then
    pyrun "$PY_PT" --framework pytorch --task "$1" --device "$2" --size "$3"
    pyrun "$PY_TF" --framework tensorflow --task "$1" --device "$2" --size "$3"
    if [[ "$2" == "cpu" ]]; then pyrun "$PY_PT" --framework scipy --task "$1" --device cpu --size "$3"; fi
  elif [[ "$2" == "cpu" ]]; then
    for fw in pytorch tensorflow scipy; do pyrun "$PY_PT" --framework "$fw" --task "$1" --device cpu --size "$3"; done
  else
    pyrun "$PY_PT" --framework pytorch --task "$1" --device cuda --size "$3"
  fi
}

R=0
ROUND_LIST="1 2 3 4 5"
if [[ "$SMOKE" == "1" ]]; then ROUND_LIST="1"; fi
for R in $ROUND_LIST; do
  RD="$OUT/run$R"; mkdir -p "$RD"
  : >"$RD/results-before.jsonl"; : >"$RD/results-after.jsonl"; : >"$RD/results-opp.jsonl"; : >"$RD/results-py.jsonl"
  PYF="$RD/results-py.jsonl"
  grc=0; gate_wait "$R" || grc=$?
  if [[ "$grc" -ne 0 ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (gate rc=${grc})"; exit 1; fi
  uptime >"$RD/uptime_before.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_before.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_before.txt" 2>&1 || true; fi
  # 順序: Rust の相手 → fandhe 両腕（セルごとに before/after を続け、先行腕はラウンドの奇偶で反転）→ Python
  each_cell opp_cell
  each_cell fandhe_cell
  each_cell py_cell
  if [[ "$PROFILE" == "macmini" ]]; then
    # ラウンド直後に 1 回。FOREIGN（またはゲートの失敗）ならこのラウンドは無効で系列を打ち切る（5/5 完走だけを採る）。
    fg_after "r$R"
    if [[ "$FG_RC" -eq 0 ]]; then echo "round$R foreign-after=clean" >>"$OUT/gate.log"
    else
      echo "round$R GATE-INVALID reason=$(fg_after_reason)" >>"$OUT/gate.log"
      echo "SERIES-ABORTED completed=$((R - 1))/5 (round${R} は無効: $(fg_after_reason))"; exit 1
    fi
  fi
  uptime >"$RD/uptime_after.txt"
  if [[ "$PROFILE" == "macmini" ]]; then pmset -g therm >"$RD/pmset_after.txt" 2>&1 || true
  else nvidia-smi --query-compute-apps=pid,name --format=csv >"$RD/gpu_apps_after.txt" 2>&1 || true; fi
  rows_b=$(wc -l <"$RD/results-before.jsonl" | tr -d ' '); rows_a=$(wc -l <"$RD/results-after.jsonl" | tr -d ' ')
  rows_opp=$(wc -l <"$RD/results-opp.jsonl" | tr -d ' '); rows_py=$(wc -l <"$RD/results-py.jsonl" | tr -d ' ')
  echo "run${R}: before=${rows_b} after=${rows_a} opp=${rows_opp} py=${rows_py}"
  unexpected=$(grep -vcE '^bench-burn gemm metal (512|1024|2048|4096) fresh: MEASURE_ERROR: gemm checksum is degenerate' "$SKIP")
  if [[ "$SMOKE" == "1" ]]; then
    # 行数だけを要約へ渡す。期待との不一致・想定外の失敗は rc=1 にするが、打ち切らず件数を報告する。
    SM_B=$rows_b; SM_A=$rows_a; SM_O=$rows_opp; SM_P=$rows_py; SM_UNEXP=$unexpected
    if [[ "$rows_b" != "$EXP_ARM" || "$rows_a" != "$EXP_ARM" || "$rows_opp" != "$EXP_OPP" || "$rows_py" != "$EXP_PY" || "$unexpected" != "0" ]]; then
      echo "smoke: 期待行数 ${EXP_ARM}/${EXP_ARM}/${EXP_OPP}/${EXP_PY} と不一致、または想定外の失敗がある"; SMOKE_BAD=1
    fi
  else
    if [[ "$rows_b" != "$EXP_ARM" || "$rows_a" != "$EXP_ARM" || "$rows_opp" != "$EXP_OPP" || "$rows_py" != "$EXP_PY" ]]; then
      echo "SERIES-ABORTED completed=$((R - 1))/5 (期待行数 ${EXP_ARM}/${EXP_ARM}/${EXP_OPP}/${EXP_PY} と不一致)"; exit 1
    fi
    if [[ "$unexpected" != "0" ]]; then echo "SERIES-ABORTED completed=$((R - 1))/5 (想定外の失敗)"; exit 1; fi
  fi
done
rm -f "$OUT/err.tmp"
restore_lock || exit 1
if ! git -C "$BASE_ROOT" diff --quiet -- "$LOCK_CLEAN_REL"; then echo "error: Cargo.lock に差分が残った" >&2; exit 1; fi
if [[ "$SMOKE" == "1" ]]; then echo "SMOKE-DONE（結果ファイルは終了時に削除。rc=${SMOKE_BAD}）"; exit "$SMOKE_BAD"; fi
echo "ALL-DONE completed=5/5"
