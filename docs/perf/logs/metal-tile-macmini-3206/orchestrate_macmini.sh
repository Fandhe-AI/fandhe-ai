#!/bin/sh
# イシュー #3206: Mac mini 向けタイル表（PR #3525）の追加前後（before=CANDIDATES[3] 強制／after=Mini 表）の
# GEMM 時間と candle の転送除外境界（gemm_transfer_excl）を 5 run 取る計測本体。規則の正は同ディレクトリの
# RULE.txt。雛形は metal-tile-macmini-3181/orchestrate_macmini.sh（.rev-stamp・差し替え禁止・ビルド失敗の即中止・
# 負荷ゲート）。診断テスト本体は crates/backend-metal/src/gemm_tile_macmini_table_diag_tests.rs（#[ignore]）。
# 使い方（Mac mini 上。展開ツリーの docs/perf/logs/metal-tile-macmini-3206/ から）:
#   CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
# bash 3.2／sh 互換のため変数は必ず ${VAR} で参照する。外部入力を eval・展開しない（N・テスト名は固定）。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/../../../.." && pwd)
FC="${ROOT}/scripts/bench/framework-compare"
: "${OUT:?OUT（出力ディレクトリ）が未指定}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR が未指定（target は 1 つに保つ）}"
export CARGO_TARGET_DIR
# framework-compare は独自の [workspace]・Cargo.lock を持つため target を分ける
FC_TARGET="${CARGO_TARGET_DIR}-fc"
GATE_LOAD=3.0
MOD="gemm_tile_macmini_table_diag_tests"

[ -f "${ROOT}/.rev-stamp" ] || { echo ".rev-stamp が無い（腕の同定不能）" >&2; exit 1; }
mkdir -p "${OUT}"
for f in precheck.log bitgate.log load_gate.log rev-stamp.txt env_info.txt build_tests.log build_candle.log \
  candle_tree.txt binaries.sha256; do
  if [ -e "${OUT}/${f}" ]; then
    echo "${f} が既に存在する（差し替え禁止）" >&2
    exit 1
  fi
done
for i in 1 2 3 4 5; do
  for f in select_run${i}.log bit_run${i}.log ab_run${i}.log candle_run${i}.jsonl candle_run${i}.err; do
    if [ -e "${OUT}/${f}" ]; then
      echo "${f} が既に存在する（差し替え禁止）" >&2
      exit 1
    fi
  done
done
cp "${ROOT}/.rev-stamp" "${OUT}/rev-stamp.txt"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  sw_vers
  sysctl -n machdep.cpu.brand_string
  sysctl hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu hw.memsize
  rustc --version
  uptime
} > "${OUT}/env_info.txt" 2>&1

# ---- ビルド（計測より前に全て終える。失敗なら即中止） ----
cd "${ROOT}" || exit 1
BUILD_JSON="${OUT}/build_tests.json.tmp"
if ! cargo test --release -p fandhe-ai-backend-metal --lib --no-run --message-format=json \
  > "${BUILD_JSON}" 2> "${OUT}/build_tests.log"; then
  echo "BUILD FAILED（backend-metal テストバイナリ）" >&2; tail -20 "${OUT}/build_tests.log" >&2; exit 1
fi
# lib テストバイナリ（deps/fandhe_ai_backend_metal-<hash>）を 1 つに確定する
grep -o '"executable":"[^"]*/fandhe_ai_backend_metal-[0-9a-f]*"' "${BUILD_JSON}" | sort -u \
  | sed 's/^"executable":"//; s/"$//' > "${OUT}/testbin_path.tmp"
if [ "$(wc -l < "${OUT}/testbin_path.tmp" | tr -d ' ')" != 1 ]; then
  echo "lib テストバイナリを 1 つに確定できない" >&2; exit 1
fi
SRC_TESTBIN=$(cat "${OUT}/testbin_path.tmp")
rm -f "${BUILD_JSON}" "${OUT}/testbin_path.tmp"
[ -x "${SRC_TESTBIN}" ] || { echo "テストバイナリが無い" >&2; exit 1; }
cd "${FC}" || exit 1
if ! CARGO_TARGET_DIR="${FC_TARGET}" cargo build --release -p bench-candle > "${OUT}/build_candle.log" 2>&1; then
  echo "BUILD FAILED（bench-candle）" >&2; tail -20 "${OUT}/build_candle.log" >&2; exit 1
fi
CARGO_TARGET_DIR="${FC_TARGET}" cargo tree -p bench-candle --depth 1 | grep -E 'candle-core v' > "${OUT}/candle_tree.txt"
if [ "$(wc -l < "${OUT}/candle_tree.txt" | tr -d ' ')" != 1 ]; then
  echo "candle-core の版を 1 行で確定できない" >&2; exit 1
fi
SRC_CANDLE="${FC_TARGET}/release/bench-candle"
[ -x "${SRC_CANDLE}" ] || { echo "bench-candle が無い" >&2; exit 1; }
# 他セッションが共有 target を再ビルドしても計測対象が変わらないよう run 専用ディレクトリへコピーして直接実行する
RUNBIN="${OUT}/.runbin"
mkdir -p "${RUNBIN}"
cp "${SRC_TESTBIN}" "${RUNBIN}/metal-tests" || exit 1
cp "${SRC_CANDLE}" "${RUNBIN}/bench-candle" || exit 1
TESTBIN="${RUNBIN}/metal-tests"
CANDLE="${RUNBIN}/bench-candle"
shasum -a 256 "${TESTBIN}" "${CANDLE}" | sed "s#${RUNBIN}/##" > "${OUT}/binaries.sha256"
cd "${ROOT}/crates/backend-metal" || exit 1
echo "build-done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"

# ---- 負荷ゲート（load1 < 3.0 を 30 秒間隔・最大 30 分。$1 = ラベル） ----
gate() {
  waited=0
  status="timeout"
  while [ "${waited}" -lt 1800 ]; do
    l1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    case "${l1}" in
      ''|*[!0-9.]*) status="unavailable"; l1="NA"; break ;;
    esac
    ok=$(awk -v a="${l1}" -v t="${GATE_LOAD}" 'BEGIN{print (a<t)?1:0}')
    if [ "${ok}" = "1" ]; then status="pass"; break; fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "$1 gate=${status} load1=${l1} waited_s=${waited} at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT}/load_gate.log"
}

run_diag() { # $1 = テスト名、$2 = 出力ファイル
  "${TESTBIN}" "${MOD}::$1" --ignored --nocapture --exact --test-threads=1 > "$2" 2>&1
}

run_fandhe() { # $1 = run 番号
  i="$1"
  gate "run${i}-select"
  run_diag m4_base_table_selected_on_macmini "${OUT}/select_run${i}.log" \
    || echo "run${i} FAILED select" >> "${OUT}/load_gate.log"
  gate "run${i}-bit"
  run_diag m4_base_table_bit_match_and_parity "${OUT}/bit_run${i}.log" \
    || echo "run${i} FAILED bit" >> "${OUT}/load_gate.log"
  gate "run${i}-ab"
  run_diag m4_base_table_ab_timing "${OUT}/ab_run${i}.log" \
    || echo "run${i} FAILED ab" >> "${OUT}/load_gate.log"
}

run_candle() { # $1 = run 番号
  i="$1"
  for n in 256 512 1024 2048 4096; do
    gate "run${i}-candle-N${n}"
    "${CANDLE}" --task gemm-transfer-split --device metal --size "${n}" \
      --out "${OUT}/candle_run${i}.jsonl" 2>> "${OUT}/candle_run${i}.err" > /dev/null \
      || echo "run${i} FAILED candle N=${n}" >> "${OUT}/load_gate.log"
  done
}

# ---- 前提ゲート 1: AC2 の tile:: 単体テスト（落ちたら系列を開始しない） ----
gate "precheck"
if ! "${TESTBIN}" tile:: --test-threads=1 > "${OUT}/precheck.log" 2>&1; then
  echo "precheck FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "PRECHECK FAILED" >&2; tail -20 "${OUT}/precheck.log" >&2; exit 1
fi
# ---- 前提ゲート 2: bit 一致・CPU 参照 parity（1 回だけ。落ちたら系列を開始しない） ----
gate "bitgate"
if ! run_diag m4_base_table_bit_match_and_parity "${OUT}/bitgate.log"; then
  echo "bitgate FAILED（系列を開始しない）" >> "${OUT}/load_gate.log"
  echo "BITGATE FAILED" >&2; tail -20 "${OUT}/bitgate.log" >&2; exit 1
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
