#!/usr/bin/env bash
# #4153 の 1 round（<metal-node> = Mac mini M4。計測キット。PR-1 時点では未実行・Linux では動かない）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/*} を作る。
#   順序は precheck → gate → (a) trace なしの窓（4 本すべて）→ (b) xctrace の腕（2 本）→ (c) 診断テスト（route → diag）。
#   奇偶の入れ替えは (a)・(b) の各段の中の framework の順だけ（奇数 round は fandhe → candle、偶数は candle → fandhe）。
#   段をまたいで framework ごとにまとめない（trace 下と trace なしの窓の熱状態をそろえるため）。(c) は入れ替えない。
#   各ステップの間に 30 秒の待ちを入れ、実行したステップ名と開始時刻（UTC）を raw/order.txt に追記する。
#   ビルドは本スクリプトの前に済ませ、ゲート通過後は cargo を呼ばない（ビルド済みバイナリを直接起動する）。
# 派生元: diag-0140-cuda-inf-wait-4150/run_round.sh（precheck・ゲート・絶対パス検証の作法）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA。framework-compare は ROOT 配下でビルド済み）
#           EXPECT_SHA=RULE.txt の tree_sha
#           DIAG_BIN=backend-metal のテストバイナリ（`cargo test -p fandhe-ai-backend-metal --release --no-run` が出力する絶対パス）
#           XCTRACE_MODE=RULE.txt の AUX xctrace（available|unavailable。unavailable なら xctrace の腕・trace 下の窓を取らない F2 のみの経路）
#           XCTRACE_SCHEMA=RULE.txt の AUX xctrace_schema（XCTRACE_MODE=available のとき必須）
#           HW_LOCK_OWNER_FILE=実機ロックの owner ファイル（空でないこと）  HW_LOCK_HELD=1（実機ロック取得済みの宣言）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# マスク: raw/ を作った後は成功・ゲート不通過・腕の失敗・シグナルのどの経路でも EXIT trap（finish）が
#   trace バンドルを消し、`to_values_4153.py --mask-dir` で round ディレクトリ（gate.txt・raw/）の全ファイルを置換する
#   （対象と限界は同スクリプト冒頭）。マスクに失敗（識別子が残る等）したら非 0 で終わる。
# bash 3.2／5 互換。run の差し替え・上書きは禁止。外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
TV="${HERE}/to_values_4153.py"
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"; DIAG_BIN="${DIAG_BIN:?診断テストのバイナリ}"
XMODE="${XCTRACE_MODE:?RULE の AUX xctrace（available|unavailable）}"; SCHEMA="${XCTRACE_SCHEMA:-}"; LOCKF="${HW_LOCK_OWNER_FILE:?実機ロックの owner ファイル}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs DIAG_BIN "${DIAG_BIN}"; validate_abs LOCKF "${LOCKF}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${SHA}" in *[!0-9a-f]*|'') echo "EXPECT_SHA は hex のみ"; exit 2;; esac
case "${XMODE}" in available|unavailable) ;; *) echo "XCTRACE_MODE は available|unavailable"; exit 2;; esac
if [ "${XMODE}" = "available" ]; then
  case "${SCHEMA}" in *[!A-Za-z0-9_-]*|'') echo "XCTRACE_SCHEMA は英数字・_・- のみ"; exit 2;; esac
fi
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
FC="${ROOT}/scripts/bench/framework-compare/target/release"
BF="${FC}/bench-fandhe"; BC="${FC}/bench-candle"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }

# ---- precheck（fail-closed）
[ -s "${LOCKF}" ] || { echo "実機ロックの owner が空（ロックなしの値は採らない）"; exit 1; }
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
for b in "${BF}" "${BC}" "${DIAG_BIN}"; do [ -x "${b}" ] || { echo "ビルド済みバイナリがない: ${b}"; exit 1; }; done
if [ "${XMODE}" = "available" ]; then
  xcrun xctrace version > /dev/null 2>&1 || { echo "xctrace が使えない"; exit 1; }
fi
# 計測の後でマスクが動かないと分かるのを避けるため、ゲートの前に確かめる
python3 -I "${TV}" --mask-ready > /dev/null || { echo "マスクの準備に失敗（to_values_4153.py --mask-ready）"; exit 1; }
mkdir -p "${RD}/raw" || exit 1
finish() {
  local rc=$?
  trap - EXIT
  rm -rf "${RD}/raw/"*.trace  # 未マスクのバイナリ（パスを含む）。失敗経路でも raw/ に残さない
  python3 -I "${TV}" --mask-dir "${RD}" || { echo "マスク失敗（round を公開しない）"; rc=3; }
  exit "${rc}"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

# ---- ゲート: load1 < 3.0 を 30 秒間隔で最大 30 分（hw-lock は上で確認済み）。
gate() {
  local i=0 l
  while [ "${i}" -le 60 ]; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${RD}/gate.txt"; return 2; fi
    if awk -v x="${l}" 'BEGIN{exit !(x < 3.0)}'; then
      echo "PASS load1=${l} hw-lock=held $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 0
    fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 1
}
gate || { echo "ゲート不通過: $(cat "${RD}/gate.txt")"; exit 1; }

RAW="${RD}/raw"
# --out は CWD 依存の既定パス（results/raw/results.jsonl）へ追記させないため、書き込み可能な raw 内を必ず明示する。
run_f()  { "${BF}" --task gemm --device metal --size 4096 --mode "$1" --out "${RAW}/$2.out" > "${RAW}/$2" 2> "${RAW}/$2.err" || { echo "bench-fandhe 失敗: $1"; exit 1; }; }
run_cg() { "${BC}" --task gemm --device metal --size 4096 --out "${RAW}/candle_gemm.out" > "${RAW}/candle_gemm.jsonl" 2> "${RAW}/candle_gemm.err" || { echo "bench-candle gemm 失敗"; exit 1; }; }
run_cs() { "${BC}" --task gemm-transfer-split --device metal --size 4096 --out "${RAW}/$1.out" > "${RAW}/$1" 2> "${RAW}/$1.err" || { echo "bench-candle split 失敗"; exit 1; }; }
# xctrace: $1=バイナリ $2=出力名の語幹 $3..=バイナリ引数。trace バンドルは export 後に自分で消す。
trace() {
  local bin="$1" stem="$2"; shift 2
  xcrun xctrace record --template 'Metal System Trace' --output "${RAW}/${stem}.trace" \
    --target-stdout "${RAW}/${stem}_traced.jsonl" --launch -- "${bin}" "$@" --out "${RAW}/${stem}.out" > "${RAW}/${stem}.xctrace.log" 2>&1 \
    || { echo "xctrace record 失敗: ${stem}"; exit 1; }
  xcrun xctrace export --input "${RAW}/${stem}.trace" \
    --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"${SCHEMA}\"]" --output "${RAW}/${stem}.trace.xml" \
    || { echo "xctrace export 失敗: ${stem}"; exit 1; }
  rm -rf "${RAW}/${stem}.trace"
}
# 1 ステップの実行。診断テストはビルド済みバイナリ（--test-threads=1 必須。--exact はモジュール修飾名が要るため使わず
# 部分一致で絞る。複数ヒットは to_values が停止する）。
step() {
  echo "$1 $(date -u +%FT%TZ)" >> "${RAW}/order.txt"
  case "$1" in
    fandhe_reuse) run_f reuse fandhe_reuse.jsonl ;;
    fandhe_fresh) run_f fresh fandhe_fresh.jsonl ;;
    candle_gemm) run_cg ;;
    candle_split) run_cs candle_split.jsonl ;;
    trace_fandhe) trace "${BF}" fandhe_reuse --task gemm --device metal --size 4096 --mode reuse ;;
    trace_candle) trace "${BC}" candle_split --task gemm-transfer-split --device metal --size 4096 ;;
    route) "${DIAG_BIN}" gemm_window_route_precheck --ignored --test-threads=1 --nocapture > "${RAW}/route.txt" 2> "${RAW}/route.err" || { echo "route precheck 失敗"; exit 1; } ;;
    diag) "${DIAG_BIN}" gemm_reuse_phase_diag_production_batch --ignored --test-threads=1 --nocapture > "${RAW}/diag.txt" 2> "${RAW}/diag.err" || { echo "phase diag 失敗"; exit 1; } ;;
    *) echo "未知のステップ: $1"; exit 1 ;;
  esac
}
# RULE 項 3 の順: (a) trace なしの窓 → (b) xctrace の腕（unavailable なら取らない）→ (c) 診断テスト。奇偶は (a)・(b) の中の framework の順。
if [ $((RN % 2)) -eq 1 ]; then
  PH_A="fandhe_reuse fandhe_fresh candle_gemm candle_split"; PH_B="trace_fandhe trace_candle"
else
  PH_A="candle_gemm candle_split fandhe_reuse fandhe_fresh"; PH_B="trace_candle trace_fandhe"
fi
[ "${XMODE}" = "available" ] || PH_B=""
first=1
for s in ${PH_A} ${PH_B} route diag; do
  [ "${first}" = "1" ] || sleep 30  # 腕の間の 30 秒（最初のステップの前と最後の後には入れない）
  first=0
  step "${s}"
done

# マスク（ユーザー名・ホスト名・ホームパスを公開面へ出さない）は finish（EXIT trap）が全経路で行う
echo "round ${RN} done $(date -u +%FT%TZ)"
