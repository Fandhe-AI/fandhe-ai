#!/usr/bin/env bash
# orchestrate.sh — framework-compare 両機体再計測（イシュー #2120・Phase 3 採否反映）の 5 run オーケストレーション。
#
# 役割: 同一ノード上で fandhe-ai の 3 腕（A: registry =0.9.0 / B: Phase 3 直前 65035979 / C: 計測時点 HEAD）と
#   candle・burn を、専有ゲート付き・5 run・プロセス独立で計測し run{1..5}/{A,B,C,others}.jsonl を LOGD へ書く。
#   判定規則は同ディレクトリ RULE.txt（実測前に固定）。集計は aggregate.py、スコアボードは scoreboard/gen_2120.py。
# 呼び出し元: orchestrate_gb10.sh／orchestrate_m4max.sh（MACHINE 引数を付けて exec する薄いラッパー）。
#   ノード上で実行する（Mac から HEAD_TREE／PRE_TREE を転送後）。
# 使い方: HEAD_TREE=<main HEAD ツリー> PRE_TREE=<65035979 ツリー> LOGD=<出力先> orchestrate.sh <gb10|m4max>
#   HEAD_TREE／PRE_TREE は各々ルートに .rev-stamp（空不可）が必要。PRE_TREE の stamp は 65035979 で始まること。
#   SMOKE=1: 1 run・N=256 の gemm のみ・ゲート省略・candle／burn なしの疎通確認（LOGD 必須。結果は判定に使わない。
#   env_info.txt の smoke=1・gate.log の smoke-skip により aggregate.py の前提ゲートが正式集計を拒否する）。
# 専有ゲート（RULE.txt「計測」）: GB10 は正式計測の前提条件で、20 回とも不通過ならその run を計測せず非 0 で停止する
#   （不通過のまま計測した run を作らない）。M4 Max は record_only で、不通過でも計測し gate.log に記録する。
# 完了記録: env_info.txt は全 run 完了・Cargo.lock 突合・build.log マスク後に最後に書く（aggregate.py は無ければ未完走として拒否）。
# 参照元の方針: 承認ピン（fandhe-ai =0.9.0）と Cargo.lock は変更しない。腕 B／C は invocation 限定の --config patch
#   （run_all.sh／run_all_cuda.sh の GEMM_GATE_PATCH_FACADE_PATH と同一機構）でビルドし、
#   bench_fandhe_lock_restore.sh の退避・復元 trap と sha256 突合で Cargo.lock 不変を保証する。
set -uo pipefail

MACHINE="${1:?usage: orchestrate.sh <gb10|m4max>}"
case "${MACHINE}" in gb10|m4max) ;; *) echo "ERROR: MACHINE は gb10 か m4max" >&2; exit 1 ;; esac
SMOKE="${SMOKE:-0}"
HEAD_TREE="${HEAD_TREE:?HEAD_TREE を指定（main HEAD ツリー）}"
PRE_TREE="${PRE_TREE:?PRE_TREE を指定（65035979 ツリー）}"
LOGD="${LOGD:?LOGD を指定（出力先ディレクトリ）}"

# A03 インジェクション対策: パスは TOML 文字列（--config）へ埋め込むため '"' と '\' を含む値を拒否する。
for v in "${HEAD_TREE}" "${PRE_TREE}" "${LOGD}"; do
  if [[ "${v}" == *'"'* || "${v}" == *'\'* ]]; then
    echo "ERROR: パスに '\"' または '\\' を含めることはできない" >&2; exit 1
  fi
done

# opt-in 環境変数の暗黙混入を防ぐ（RULE.txt: FANDHE_AI_* は一切設定しない）。
if env | grep -q '^FANDHE_AI_'; then
  echo "ERROR: FANDHE_AI_* 環境変数が設定されている。既定値のみを計測するため停止" >&2
  env | grep '^FANDHE_AI_' | cut -d= -f1 >&2
  exit 1
fi

for t in "${HEAD_TREE}" "${PRE_TREE}"; do
  [[ -s "${t}/.rev-stamp" ]] || { echo "ERROR: ${t}/.rev-stamp が無い／空（転送後に書き込むこと）" >&2; exit 1; }
done
SHA_C="$(cat "${HEAD_TREE}/.rev-stamp")"; SHA_B="$(cat "${PRE_TREE}/.rev-stamp")"
[[ "${SHA_B}" == 65035979* ]] || { echo "ERROR: PRE_TREE の rev-stamp が 65035979 で始まらない: ${SHA_B}" >&2; exit 1; }

FC="${HEAD_TREE}/scripts/bench/framework-compare"
[[ -d "${FC}" ]] || { echo "ERROR: ${FC} が無い" >&2; exit 1; }
mkdir -p "${LOGD}/bin" || exit 1
# 完了記録は最後に書き直す（途中停止したとき前回の完走記録が新しい gate.log 等と並んで残らないよう、最初に消す）
rm -f "${LOGD}/env_info.txt" || exit 1
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
# 置換順: 具体的なパス（LOGD・HEAD_TREE・PRE_TREE）を先に、最後に HOME（HOME 配下に置く実機ノードで <head-tree> 等が潰れないように）
# sed の置換式へパスを埋め込むと `|`・改行・`&` で式を脱出でき（GNU sed の e コマンドでシェル実行に至る）、
# 正規表現メタ文字も誤解釈されるため、python3 の固定文字列置換へ渡す（値は環境変数経由でコード・式に連結しない）。
mask() {
  MASK_LOGD="${LOGD}" MASK_HEAD="${HEAD_TREE}" MASK_PRE="${PRE_TREE}" MASK_HOME="${HOME}" python3 -c '
import os, sys
pairs = [(os.environ["MASK_LOGD"], "<logd>"), (os.environ["MASK_HEAD"], "<head-tree>"),
         (os.environ["MASK_PRE"], "<pre-tree>"), (os.environ["MASK_HOME"], "<home>")]
for line in sys.stdin:
    for src, dst in pairs:
        if src:
            line = line.replace(src, dst)
    sys.stdout.write(line)
'
}

if [[ "${SMOKE}" == "1" ]]; then RUNS=1; else RUNS=5; fi
BUILD_LOG="${LOGD}/build.log"; GATE_LOG="${LOGD}/gate.log"; : > "${BUILD_LOG}" || exit 1; : > "${GATE_LOG}" || exit 1

cd "${FC}" || exit 1
# Cargo.lock の退避・復元 trap（EXIT で必ず元へ戻す）＋各腕ビルド前の手動リセット用コピー。
source ./bench_fandhe_lock_restore.sh
bench_fandhe_setup_lock_restore_trap
LOCK_SHA0="$(sha256_of Cargo.lock)"
LOCK_COPY="$(mktemp)" || exit 1
cp Cargo.lock "${LOCK_COPY}" || exit 1
# EXIT trap は 1 スロットのため、Cargo.lock 復元 trap を包んで一時コピー削除も行う（build 失敗の exit 1 でも残さない）。
cleanup_all() { local c=$?; rm -f "${LOCK_COPY}"; (exit "$c"); bench_fandhe_restore_lock_trap; }
trap cleanup_all EXIT
reset_lock() { cp "${LOCK_COPY}" Cargo.lock; }
assert_lock() {
  [[ "$(sha256_of Cargo.lock)" == "${LOCK_SHA0}" ]] || { echo "ERROR: Cargo.lock の sha256 が変化した ($1)" | tee -a "${BUILD_LOG}" >&2; return 1; }
}
echo "Cargo.lock sha256(before)=${LOCK_SHA0}" >> "${BUILD_LOG}"

# --- スイッチ既定値スナップショット（腕 B は Phase 3 導入前のため欠落許容）---
SW="$(dirname "$0")/switches.sh"
[[ -x "${SW}" ]] || SW="${HEAD_TREE}/docs/perf/logs/framework-compare-phase3-remeasure-2120/switches.sh"
bash "${SW}" "${PRE_TREE}" --allow-missing > "${LOGD}/switches-B.txt" || { echo "ERROR: switches.sh(B) 失敗" >&2; exit 1; }
bash "${SW}" "${HEAD_TREE}" > "${LOGD}/switches-C.txt" || { echo "ERROR: switches.sh(C) 失敗。計測を開始しない" >&2; exit 1; }

# --- ビルド（失敗したら計測前に停止。古いバイナリで計測しない）---
build_arm() { # build_arm <A|B|C> <patch-facade-path or "">
  local arm=$1 patch=$2; local args=()
  reset_lock
  [[ -n "${patch}" ]] && args+=(--config "patch.crates-io.fandhe-ai.path=\"${patch}/crates/facade\"")
  echo "== build bench-fandhe arm=${arm} patch=${patch:-none} ($(date -u +%FT%TZ))" >> "${BUILD_LOG}"
  cargo build --release -p bench-fandhe ${args[@]+"${args[@]}"} >> "${BUILD_LOG}" 2>&1 || { echo "ERROR: build arm ${arm} 失敗 -> 計測を停止" >&2; exit 1; }
  echo "-- cargo tree arm=${arm}" >> "${BUILD_LOG}"
  local tree_line
  tree_line="$(cargo tree -p bench-fandhe --depth 1 ${args[@]+"${args[@]}"} 2>/dev/null | grep -E 'fandhe-ai v')"
  echo "${tree_line}" >> "${BUILD_LOG}"
  # 腕 B／C は指定ツリーの facade path に解決されたことを計測前に確認する（patch 不発で registry 版を計測しない）。
  if [[ -n "${patch}" && "${tree_line}" != *"(${patch}/crates/facade)"* ]]; then
    echo "ERROR: 腕 ${arm} が ${patch}/crates/facade に解決されていない: ${tree_line}" >&2; exit 1
  fi
  cp target/release/bench-fandhe "${LOGD}/bin/bench-fandhe-${arm}" || exit 1
}
build_arm A ""
grep -q 'fandhe-ai v0.9.0$' <(sed -n '/cargo tree arm=A/,$p' "${BUILD_LOG}" | head -3) || { echo "ERROR: 腕 A が registry =0.9.0 に解決されていない" >&2; exit 1; }
build_arm B "${PRE_TREE}"
build_arm C "${HEAD_TREE}"
reset_lock
if [[ "${SMOKE}" != "1" ]]; then
  if [[ "${MACHINE}" == "gb10" ]]; then OF=(--no-default-features --features cuda); else OF=(); fi
  for crate in bench-candle bench-burn; do
    echo "== build ${crate} ${OF[*]:-}" >> "${BUILD_LOG}"
    cargo build --release -p "${crate}" ${OF[@]+"${OF[@]}"} >> "${BUILD_LOG}" 2>&1 || { echo "ERROR: build ${crate} 失敗 -> 計測を停止" >&2; exit 1; }
    cp "target/release/${crate}" "${LOGD}/bin/${crate}" || exit 1
  done
fi
assert_lock "after builds" || exit 1
echo "Cargo.lock sha256(after)=$(sha256_of Cargo.lock)" >> "${BUILD_LOG}"

# --- 専有ゲート ---
# 取得値が数値でない（コマンド失敗・[N/A] 等）ときは通過扱いにしない（awk の文字列比較で "" < 8.0 が真になるのを防ぐ）。
is_num() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
gate() { # gate <run>
  local r=$1 i l1 gu=-
  if [[ "${SMOKE}" == "1" ]]; then echo "run=${r} attempt=0 load1=- gpu_util=- smoke-skip" >> "${GATE_LOG}"; return 0; fi
  if [[ "${MACHINE}" == "gb10" ]]; then
    for i in $(seq 1 20); do
      l1="$(cut -d' ' -f1 /proc/loadavg)"
      gu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
      is_num "${l1}" || l1="NA"; [[ "${gu}" =~ ^[0-9]+$ ]] || gu="NA"
      if [[ "${l1}" != "NA" && "${gu}" != "NA" ]] && awk -v l="${l1}" -v g="${gu}" 'BEGIN{exit !(l<1.0 && g==0)}'; then
        echo "run=${r} attempt=${i} load1=${l1} gpu_util=${gu} pass" >> "${GATE_LOG}"; return 0
      fi
      echo "run=${r} attempt=${i} load1=${l1} gpu_util=${gu} wait" >> "${GATE_LOG}"; sleep 30
    done
    echo "run=${r} attempt=20 load1=${l1} gpu_util=${gu} fail" >> "${GATE_LOG}"; return 1
  fi
  # M4 Max（record_only）: load1 < 8.0 を最大 30 分（60 秒間隔）待つ。不通過でも計測は実行し記録する。
  for i in $(seq 1 30); do
    l1="$(sysctl -n vm.loadavg | awk '{print $2}')"
    is_num "${l1}" || l1="NA"
    if [[ "${l1}" != "NA" ]] && awk -v l="${l1}" 'BEGIN{exit !(l<8.0)}'; then
      echo "run=${r} attempt=${i} load1=${l1} gpu_util=- pass" >> "${GATE_LOG}"; return 0
    fi
    echo "run=${r} attempt=${i} load1=${l1} gpu_util=- wait" >> "${GATE_LOG}"; sleep 60
  done
  echo "run=${r} attempt=30 load1=${l1} gpu_util=- fail" >> "${GATE_LOG}"; return 1
}

# --- セル定義（RULE.txt「セル範囲」。task device size mode flag を 1 行 1 セル）---
fandhe_cells() {
  local n dev m
  if [[ "${MACHINE}" == "gb10" ]]; then local GDEV=cuda GSZ="256 512 1024 2048 4096" CSZ="256 512 1024 2048 4096" DEVS="cuda cpu"
  else local GDEV=metal GSZ="256 512 1024 2048 4096" CSZ="256 512 1024 2048" DEVS="cpu metal"; fi
  if [[ "${SMOKE}" == "1" ]]; then GSZ="256"; CSZ="256"; fi
  for n in ${GSZ}; do for m in fresh reuse; do echo "gemm ${GDEV} ${n} ${m} -"; done; done
  for n in ${CSZ}; do for m in fresh reuse; do echo "gemm cpu ${n} ${m} -"; done; done
  [[ "${SMOKE}" == "1" ]] && return 0
  for dev in ${DEVS}; do
    for m in fresh reuse; do echo "train ${dev} 64 ${m} -"; echo "infer ${dev} 64 ${m} -"; done
    for m in fresh reuse; do echo "train ${dev} 64 ${m} --phases"; echo "infer ${dev} 64 ${m} --phases"; done
  done
}
others_cells() {
  local n dev
  if [[ "${MACHINE}" == "gb10" ]]; then local GDEV=cuda GSZ="256 512 1024 2048 4096" CSZ="256 512 1024 2048 4096" DEVS="cuda cpu"
  else local GDEV=metal GSZ="256 512 1024 2048 4096" CSZ="256 512 1024 2048" DEVS="cpu metal"; fi
  for n in ${GSZ}; do echo "gemm ${GDEV} ${n} fresh -"; done
  for n in ${CSZ}; do echo "gemm cpu ${n} fresh -"; done
  for dev in ${DEVS}; do echo "train ${dev} 64 fresh -"; echo "infer ${dev} 64 fresh -"; done
}

run_cells() { # run_cells <bin> <out> <err> <skiplog> < cells
  local bin=$1 out=$2 err=$3 skip=$4 task dev size mode flag
  while read -r task dev size mode flag; do
    if [[ "${flag}" == "-" ]]; then
      "${bin}" --task "${task}" --device "${dev}" --size "${size}" --mode "${mode}" --out "${out}" 2>>"${err}" \
        || echo "$(basename "${bin}") ${task} ${dev} ${size} ${mode}: FAIL" >> "${skip}"
    else
      "${bin}" --task "${task}" --device "${dev}" --size "${size}" --mode "${mode}" "${flag}" --out "${out}" 2>>"${err}" \
        || echo "$(basename "${bin}") ${task} ${dev} ${size} ${mode} ${flag}: FAIL" >> "${skip}"
    fi
  done
}

uptime > "${LOGD}/uptime_before.txt" || exit 1
echo "start $(date -u +%FT%TZ) machine=${MACHINE} B=${SHA_B} C=${SHA_C} smoke=${SMOKE}"
for r in $(seq 1 "${RUNS}"); do
  RD="${LOGD}/run${r}"; mkdir -p "${RD}" || exit 1
  ERR="${RD}/err.log"; SKIPLOG="${RD}/skipped.log"; : > "${ERR}" || exit 1; : > "${SKIPLOG}" || exit 1
  if ! gate "${r}"; then
    if [[ "${MACHINE}" == "gb10" ]]; then
      # GB10 の専有ゲートは前提条件（RULE.txt）。不通過のまま計測した run を作らず、ここで正式計測を停止する。
      echo "ERROR: run${r}: 専有ゲート不通過（GB10 は前提条件）。計測を停止する。負荷を解消して最初から再実行すること" >&2
      exit 1
    fi
    echo "run${r}: 専有ゲート不通過（M4 Max は record_only。除外せず gate.log に記録して続行。RULE.txt）"
  fi
  if (( r % 2 == 1 )); then ORDER="A B C"; else ORDER="C B A"; fi
  for arm in ${ORDER}; do
    : > "${RD}/${arm}.jsonl" || exit 1
    echo "-- run${r} arm=${arm} ($(date -u +%FT%TZ))"
    fandhe_cells | run_cells "${LOGD}/bin/bench-fandhe-${arm}" "${RD}/${arm}.jsonl" "${ERR}" "${SKIPLOG}"
  done
  if [[ "${SMOKE}" != "1" ]]; then
    : > "${RD}/others.jsonl" || exit 1
    for bin in bench-candle bench-burn; do
      others_cells | run_cells "${LOGD}/bin/${bin}" "${RD}/others.jsonl" "${ERR}" "${SKIPLOG}"
    done
  fi
  echo "run${r} rows: A=$(wc -l < "${RD}/A.jsonl") B=$(wc -l < "${RD}/B.jsonl") C=$(wc -l < "${RD}/C.jsonl") skipped=$(wc -l < "${SKIPLOG}")"
done
assert_lock "after runs" || exit 1
if ! { mask < "${BUILD_LOG}" > "${BUILD_LOG}.masked" && mv "${BUILD_LOG}.masked" "${BUILD_LOG}"; }; then
  echo "ERROR: build.log のマスクに失敗" >&2; exit 1
fi
uptime > "${LOGD}/uptime_after.txt" || exit 1

# 完了記録（最後に書く）。runs・smoke は aggregate.py の前提ゲートが正式計測か否かの判定に使う。
write_env_info() {
  {
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "machine=${MACHINE}"
    echo "rev_A=registry fandhe-ai =0.9.0"
    echo "rev_B=${SHA_B}"
    echo "rev_C=${SHA_C}"
    echo "runs=${RUNS}"
    echo "smoke=${SMOKE}"
    echo "uname=$(uname -srm)"
    echo "rustc=$(rustc -V)"; echo "cargo=$(cargo -V)"
    if [[ "${MACHINE}" == "gb10" ]]; then
      echo "nvcc=$(nvcc --version 2>/dev/null | tail -1)"
      nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv 2>/dev/null
      echo "nproc=$(nproc)"
      echo "load_gate=load1<1.0 && gpu_util==0 (see gate.log)"
    else
      echo "sw_vers=$(sw_vers -productVersion 2>/dev/null)"
      echo "cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
      echo "ncpu=$(sysctl -n hw.ncpu 2>/dev/null)"
      echo "therm=$( (pmset -g therm 2>/dev/null | tr '\n' ' ') || true)"
      [[ -n "$(pmset -g therm 2>/dev/null)" ]] || echo "therm=取得不能"
      echo "load_gate=record_only: load1<8.0 を最大 30 分待機（see gate.log）"
    fi
    echo "FANDHE_AI_*=未設定（起動時に検査）"
    echo "Cargo.lock_sha256=${LOCK_SHA0}（前後一致を確認）"
    echo "head_tree=<head-tree>  pre_tree=<pre-tree>"
  } 2>&1 | mask > "${LOGD}/env_info.txt.tmp"
}
if ! { write_env_info && mv "${LOGD}/env_info.txt.tmp" "${LOGD}/env_info.txt"; }; then
  echo "ERROR: env_info.txt の書き出しに失敗（完了記録なし＝aggregate.py は集計しない）" >&2; exit 1
fi
rm -f "${LOCK_COPY}"
echo "done. $(date -u +%FT%TZ)"
