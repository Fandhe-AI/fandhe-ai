#!/bin/sh
# イシュー #4160: Mac mini で N>=1024 限定の steel 候補 T0U-LU-FB opt-in（tile::STEEL_LARGE_N_ENABLED）の
# 到達追跡（trace）と純カーネル時間（kern）、candle の転送除外境界（gemm_transfer_excl）を 5 run 取る
# 計測本体。規則の正は同ディレクトリの RULE.txt（RULE.template.txt を埋めて単独コミットで固定したもの）。
# 雛形は metal-frag-interleave-ab-macmini-3212/orchestrate_macmini.sh（.rev-stamp・差し替え禁止・
# ビルド失敗の即中止・負荷ゲート）。gemm セル A/B は別プロセス
# （scripts/bench/framework-compare/run_ab_steel_large_n_gemm_metal.sh）で、本スクリプトと同時に走らせない
# （RULE 項 12）。ビルドはすべて最初に済ませ、負荷ゲート付きの計測区間の外で行う（RULE 項 6）。
# 使い方（Mac mini 上。展開ツリーの docs/perf/logs/metal-steel-large-n-ab-macmini-4160/ から）:
#   FOREIGN_GATE=<foreign_job_gate.py の絶対パス> CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
#     OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
#   FOREIGN_GATE は必須（他ジョブの前提ゲート。docs/perf/logs/macmini-foreign-job-gate/。未指定なら起動しない）。
#   sh orchestrate_macmini.sh --dry-run   # 何も作らず計画だけ表示する
# bash 3.2／sh 互換のため変数は必ず ${VAR} で参照する。外部入力を eval・展開しない（テスト名・N は固定）。
# ホスト名（uname -n）・ユーザー名・$HOME は出力しない（RULE 項 2 のマスク方針）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
FC="${ROOT}/scripts/bench/framework-compare"
# 他ジョブの前提ゲートの関数群（run_ab_steel_large_n_gemm_metal.sh と共用。記録の文法はそちらの冒頭コメントが正）
# shellcheck source=../../../../scripts/bench/framework-compare/ab_foreign_gate.sh
. "${FC}/ab_foreign_gate.sh" || exit 1
GATE_LOAD=3.0
MOD="gemm_steel_large_n_diag_tests"
PREREQ_MOD="gemm_steel_candidate_diag_tests"

if [ "$#" -gt 1 ]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [ "$#" -eq 1 ]; then
  if [ "$1" != "--dry-run" ]; then
    echo "不明な引数: $1（usage: $0 [--dry-run]）" >&2
    exit 2
  fi
  echo "dry-run: 以下を実行する（ログは作らない）"
  echo "  前提: .rev-stamp・RULE.txt（記入済み）必須・OUT の既存ログは上書き禁止"
  echo "  ビルド: backend-metal テスト（release）・bench-candle"
  echo "  前提ゲート: tile:: 単体・${PREREQ_MOD} 全件・${MOD}::steel_large_n_bit_identity・steel_large_n_prereq_parity"
  echo "  5 run x {trace(${MOD}::steel_large_n_reach_trace), kern(${MOD}::steel_large_n_kernel_gpu)} と candle excl（N=256 512 1024 2048 4096）"
  echo "  奇数 run は fandhe -> candle、偶数 run は candle -> fandhe。各プロセス前に hw-lock と load1<${GATE_LOAD}"
  echo "  他ジョブの前提ゲート: FOREIGN_GATE 必須。各プロセスの前は他ジョブ CLEAN かつ load1<${GATE_LOAD} を同じループで待ち（30 秒間隔・最大 60 回）、後に 1 回確認（FOREIGN は gate=invalid reason=foreign-after）"
  exit 0
fi

: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定（target は 1 つに保つ）}"
export CARGO_TARGET_DIR
: "${FOREIGN_GATE:?FOREIGN_GATE（foreign_job_gate.py の絶対パス）が未指定}"
# framework-compare は独自の [workspace]・Cargo.lock を持つため target を分ける
FC_TARGET="${CARGO_TARGET_DIR}-fc"

[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
# 事前登録規則 RULE.txt（雛形を埋めて単独コミットで固定したもの）が無い・未記入のまま計測しない。
# aggregate.py は OUT/RULE.txt を要求するため、計測開始前に確認して OUT へコピーする。
[ -f "${HERE}/RULE.txt" ] || { echo "RULE.txt が無い（RULE.template.txt を埋めて固定してから計測する）" >&2; exit 1; }
if grep -E '^[[:space:]]*(fixed-at-utc|tree_sha|tree_fingerprint|aggregate_script_sha256)[[:space:]]*[:=].*<[^<>]*>' "${HERE}/RULE.txt" > /dev/null; then
  echo "RULE.txt に未記入欄（<...>）が残っている" >&2; exit 1
fi
mkdir -p "${OUT}" || exit 1
# 相対指定の OUT は以降の cd で別基準になりログ上書き禁止検査を迂回しうるため、絶対パスへ正規化する。
OUT=$(cd "${OUT}" && pwd) || exit 1
if [ -e "${OUT}/foreign-gate" ]; then
  echo "foreign-gate が既に存在する（差し替え禁止。再実行は別の OUT で行う）" >&2
  exit 1
fi
for f in prereq_tile.log prereq_steel.log bitident.log parity.log load_gate.log RULE.txt rev-stamp.txt env_info.txt build_tests.log build_candle.log; do
  if [ -e "${OUT}/${f}" ]; then
    echo "${f} が既に存在する（差し替え禁止。再実行は別の OUT で行う）" >&2
    exit 1
  fi
done
for i in 1 2 3 4 5; do
  for f in trace_run${i}.log kern_run${i}.log candle_run${i}.jsonl; do
    if [ -e "${OUT}/${f}" ]; then
      echo "${f} が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
done
fg_require "${FOREIGN_GATE}" "${OUT}/foreign-gate" || exit 1
cp "${HERE}/RULE.txt" "${OUT}/RULE.txt"
cp "${ROOT}/.rev-stamp" "${OUT}/rev-stamp.txt"
{
  echo "host_kind=macmini"
  date -u +%Y-%m-%dT%H:%M:%SZ
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu hw.memsize
  rustc --version
  uptime
  echo "foreign_job_gate_sha256=$(fg_sha256 "${FOREIGN_GATE}")"
} > "${OUT}/env_info.txt" 2>&1

# ---- ビルド（計測より前に全て終える。失敗なら即中止） ----
cd "${ROOT}" || exit 1
if ! cargo test --release -p fandhe-ai-backend-metal --lib --no-run > "${OUT}/build_tests.log" 2>&1; then
  echo "BUILD FAILED（backend-metal テストバイナリ）" >&2; tail -20 "${OUT}/build_tests.log" >&2; exit 1
fi
cd "${FC}" || exit 1
if ! CARGO_TARGET_DIR="${FC_TARGET}" cargo build --release -p bench-candle > "${OUT}/build_candle.log" 2>&1; then
  echo "BUILD FAILED（bench-candle）" >&2; tail -20 "${OUT}/build_candle.log" >&2; exit 1
fi
CARGO_TARGET_DIR="${FC_TARGET}" cargo tree -p bench-candle --depth 1 | grep -E 'candle-core v' > "${OUT}/candle_tree.txt"
if [ "$(wc -l < "${OUT}/candle_tree.txt" | tr -d ' ')" != 1 ]; then
  echo "candle-core の版を 1 行で確定できない" >&2; exit 1
fi
CANDLE="${FC_TARGET}/release/bench-candle"
[ -x "${CANDLE}" ] || { echo "bench-candle が無い" >&2; exit 1; }
cd "${ROOT}" || exit 1
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

# ---- ゲート（hw-lock の保持確認と、他ジョブ CLEAN かつ load1 < 3.0。30 秒間隔・最大 30 分＝60 回。$1 = ラベル） ----
# 通過以外（GATE-TIMEOUT／GATE-INVALID）は gate= に記録し（理由は reason=foreign-job|load1|foreign-gate-error|load1-unavailable）、
# aggregate.py が正式系列から外す。通過の行の書式は従来のまま。各回のゲート出力は ${OUT}/foreign-gate/ に残る。
# hw-lock は docs/real-hardware-verification-env.md の運用に従い、呼び出し側が保持してから起動する。
gate() {
  fg_wait_before "$1" "${GATE_LOAD}"
  extra=""
  if [ "${FG_STATUS}" != "pass" ]; then
    extra=" reason=${FG_REASON}"
    [ "${FG_REASON}" = "foreign-gate-error" ] && extra="${extra} phase=before"
  fi
  echo "$1 gate=${FG_STATUS} load1=${FG_L1} waited_s=${FG_WAITED}${extra} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}

# 計測プロセスの直後に他ジョブを 1 回確かめる（$1 = gate() と同じラベル）。FOREIGN／ゲート失敗は同じラベルの
# gate=invalid を足す（aggregate.py は重複・非 pass として正式系列から外す。続行は既存の GATE-INVALID と同じ）。
gate_after() {
  fg_after "$1"
  if [ "${FG_RC}" -ne 0 ]; then
    echo "$1 gate=invalid reason=$(fg_after_reason) phase=after at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
  fi
}

run_diag() { # $1 = モジュール、$2 = テスト名（空なら全件）、$3 = 出力ファイル
  if [ -n "$2" ]; then
    cargo test --release -p fandhe-ai-backend-metal --lib "$1::$2" \
      -- --ignored --nocapture --exact --test-threads=1 > "$3" 2>&1
  else
    cargo test --release -p fandhe-ai-backend-metal --lib "$1" \
      -- --ignored --nocapture --test-threads=1 > "$3" 2>&1
  fi
}

run_fandhe() { # $1 = run 番号
  i="$1"
  gate "run${i}-trace"
  run_diag "${MOD}" steel_large_n_reach_trace "${OUT}/trace_run${i}.log" \
    || echo "run${i} FAILED trace" >> "${OUT}/load_gate.log"
  gate_after "run${i}-trace"
  gate "run${i}-kern"
  run_diag "${MOD}" steel_large_n_kernel_gpu "${OUT}/kern_run${i}.log" \
    || echo "run${i} FAILED kern" >> "${OUT}/load_gate.log"
  gate_after "run${i}-kern"
}

run_candle() { # $1 = run 番号
  i="$1"
  for n in 256 512 1024 2048 4096; do
    gate "run${i}-candle-N${n}"
    "${CANDLE}" --task gemm-transfer-split --device metal --size "${n}" \
      --out "${OUT}/candle_run${i}.jsonl" 2>> "${OUT}/candle_run${i}.err" > /dev/null \
      || echo "run${i} FAILED candle N=${n}" >> "${OUT}/load_gate.log"
    gate_after "run${i}-candle-N${n}"
  done
}

# ---- 前提ゲート（RULE 項 5。性能 run の前に 1 回。1 件でも落ちたら系列を開始しない） ----
gate "prereq-tile"
if ! cargo test --release -p fandhe-ai-backend-metal --lib tile:: > "${OUT}/prereq_tile.log" 2>&1; then
  echo "prereq_tile FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PREREQ TILE FAILED" >&2; tail -20 "${OUT}/prereq_tile.log" >&2; exit 1
fi
gate "prereq-steel"
if ! run_diag "${PREREQ_MOD}" "" "${OUT}/prereq_steel.log"; then
  echo "prereq_steel FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PREREQ STEEL FAILED" >&2; tail -20 "${OUT}/prereq_steel.log" >&2; exit 1
fi
gate "prereq-bitident"
if ! run_diag "${MOD}" steel_large_n_bit_identity "${OUT}/bitident.log"; then
  echo "bitident FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "BITIDENT FAILED" >&2; tail -20 "${OUT}/bitident.log" >&2; exit 1
fi
gate "prereq-parity"
if ! run_diag "${MOD}" steel_large_n_prereq_parity "${OUT}/parity.log"; then
  echo "parity FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PARITY FAILED" >&2; tail -20 "${OUT}/parity.log" >&2; exit 1
fi

# ---- 5 run（奇数 run は fandhe -> candle、偶数 run は candle -> fandhe） ----
for i in 1 2 3 4 5; do
  if [ $((i % 2)) -eq 1 ]; then
    run_fandhe "${i}"
    run_candle "${i}"
  else
    run_candle "${i}"
    run_fandhe "${i}"
  fi
  echo "run${i} end_load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load_gate.log"
done
echo "series done" >> "${OUT}/load_gate.log"
