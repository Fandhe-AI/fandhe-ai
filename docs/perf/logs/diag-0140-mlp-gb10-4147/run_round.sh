#!/usr/bin/env bash
# #4147 の 1 round（<cuda-node> = DGX GB10。計測キット。この PR 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。values.tsv は
#   to_values_4147.py が raw/ から作る（計測と変換を分け、計測中に余計な処理を走らせない）。
# 派生元: diag-0140-mlp-macmini-4146/run_round.sh（腕の構成・上書き禁止・絶対パス検証）と
#   framework-compare-0.14.0-remeasure/scripts/dgx-run-0140.sh・dgx-0140-loop.sh（GB10 のゲートと実機ロックの作法）。
# 環境変数: ROOT=計測ツリー（tree_sha の展開先。.rev-stamp に完全 SHA）
#           PY=torch 入りの python（torch 2.14.0）  BP=bench_py.py の絶対パス
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_OWNER_FILE=実機ロックの owner ファイル（空でないこと）
#           HW_LOCK_HELD=1（実機ロックを取得済みであることの宣言）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# 順序: 奇数 round は Rust → Python、偶数 round は Python → Rust。ビルドは系列前に済ませる（ここでは実行しない）。
# 計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止（既存ファイルがあれば停止）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
export PATH="${HOME}/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ROOT="${ROOT:?計測ツリー}"; PY="${PY:?torch 入り python}"; BP="${BP:?bench_py.py}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"
LOCKF="${HW_LOCK_OWNER_FILE:?実機ロックの owner ファイル}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs PY "${PY}"; validate_abs BP "${BP}"; validate_abs LOCKF "${LOCKF}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
[ -z "${GEMM_GATE_PATCH_FACADE_PATH:-}" ] || { echo "GEMM_GATE_PATCH_FACADE_PATH が非空（registry 計測では拒否）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ -s "${LOCKF}" ] || { echo "実機ロックの owner が空（ロックなしの値は採らない）"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'GB10' || { echo "機体が GB10 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
echo "${BENCH_PY_SHA256}  ${BP}" | sha256sum -c - > /dev/null || { echo "bench_py.py の sha256 不一致"; exit 1; }
"${PY}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") else 1)' || { echo "torch の版が 0.14.0 計測（2.14.0）と違う"; exit 1; }
[ -x "${FC}/target/release/bench-fandhe" ] || { echo "bench-fandhe が未ビルド"; exit 1; }
[ -x "${ROOT}/target/release/examples/cpu_jobgrid_bench" ] || { echo "cpu_jobgrid_bench が未ビルド"; exit 1; }
[ -x "${ROOT}/target/release/examples/cuda_mlp_event_bench_4147" ] || { echo "cuda_mlp_event_bench_4147 が未ビルド（--features internal-diagnostics）"; exit 1; }

# ---- ゲート: compute プロセス 0 件・GPU 使用率 0 が 3 サンプル連続・load1 < 1.0 を 30 秒間隔で最大 30 分。
#   nvidia-smi の失敗は専有未確認として停止（INVALID）。
gate() {
  local i=0 apps util l zero=0
  while [ "${i}" -le 60 ]; do
    if ! apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); then echo "INVALID nvidia-smi-apps-failed" > "${RD}/gate.txt"; return 2; fi
    if ! util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1); then echo "INVALID nvidia-smi-util-failed" > "${RD}/gate.txt"; return 2; fi
    l=$(awk '{print $1}' /proc/loadavg)
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${RD}/gate.txt"; return 2; fi
    if [ -z "${apps}" ] && [ "$(printf '%s' "${util}" | tr -d ' ')" = "0" ] && awk -v x="${l}" 'BEGIN{exit !(x < 1.0)}'; then
      zero=$((zero + 1))
    else
      zero=0
    fi
    if [ "${zero}" -ge 3 ]; then echo "PASS load1=${l} compute-apps=0 gpu-util=0x3 hw-lock=held $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 0; fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 1
}
gate || { echo "ゲート不通過: $(cat "${RD}/gate.txt")"; exit 1; }

rust_arm() {
  : > "${RD}/raw/rust.jsonl"; : > "${RD}/raw/jobgrid.txt"
  local t m
  for t in train infer; do for m in fresh reuse; do
    "${FC}/target/release/bench-fandhe" --task "${t}" --device cpu --size 64 --mode "${m}" --out "${RD}/raw/rust.jsonl" 2>> "${RD}/raw/rust.err" \
      || { echo "bench-fandhe cpu ${t} ${m} 失敗"; return 1; }
  done; done
  "${ROOT}/target/release/examples/cpu_jobgrid_bench" "r${RN}" threads > "${RD}/raw/jobgrid.txt" 2> "${RD}/raw/jobgrid.err" || { echo "cpu_jobgrid_bench 失敗"; return 1; }
  for m in fresh reuse; do
    "${FC}/target/release/bench-fandhe" --task infer --device cuda --size 64 --mode "${m}" --out "${RD}/raw/rust.jsonl" 2>> "${RD}/raw/rust.err" \
      || { echo "bench-fandhe cuda infer ${m} 失敗"; return 1; }
  done
  "${ROOT}/target/release/examples/cuda_mlp_event_bench_4147" > "${RD}/raw/fandhe_event.txt" 2> "${RD}/raw/fandhe_event.err" || { echo "cuda_mlp_event_bench_4147 失敗"; return 1; }
}
py_arm() {
  : > "${RD}/raw/py.jsonl"
  local t
  for t in train infer; do
    "${PY}" "${BP}" --framework pytorch --task "${t}" --device cpu --size 64 --out "${RD}/raw/py.jsonl" > /dev/null 2>> "${RD}/raw/py.err" \
      || { echo "bench_py cpu ${t} 失敗"; return 1; }
  done
  "${PY}" "${FC}/opp_kernel_torch_diag.py" measure --bench-py "${BP}" --device cpu --shapes mlp --arm base --run "${RN}" --out "${RD}/raw/opp_kernel.jsonl" 2> "${RD}/raw/opp_kernel.err" \
    || { echo "opp_kernel_torch_diag 失敗"; return 1; }
  "${PY}" "${BP}" --framework pytorch --task infer --device cuda --size 64 --out "${RD}/raw/py.jsonl" > /dev/null 2>> "${RD}/raw/py.err" \
    || { echo "bench_py cuda infer 失敗"; return 1; }
  "${PY}" "${HERE}/opp_kernel_cuda_mlp_diag_4147.py" measure --bench-py "${BP}" --fc-dir "${FC}" --run "${RN}" --out "${RD}/raw/opp_event.jsonl" 2> "${RD}/raw/opp_event.err" \
    || { echo "opp_kernel_cuda_mlp_diag_4147 失敗"; return 1; }
}
if [ $((RN % 2)) -eq 1 ]; then rust_arm && py_arm; else py_arm && rust_arm; fi || { echo "round ${RN} FAILED"; exit 1; }

# record_only（分類には使わない）: Tape 経路の区間内訳。失敗しても round は止めない（欠損は README に記す）
"${FC}/target/release/bench-fandhe" --task train --device cpu --size 64 --mode reuse --phases --out "${RD}/raw/phases-train.jsonl" 2> "${RD}/raw/phases-train.err" || echo "phases-train 失敗（record_only）"
"${FC}/target/release/bench-fandhe" --task infer --device cpu --size 64 --mode reuse --phases --out "${RD}/raw/phases-infer.jsonl" 2> "${RD}/raw/phases-infer.err" || echo "phases-infer 失敗（record_only）"
"${FC}/target/release/bench-fandhe" --task infer --device cuda --size 64 --mode reuse --phases --out "${RD}/raw/phases-infer-cuda.jsonl" 2> "${RD}/raw/phases-infer-cuda.err" || echo "phases-infer-cuda 失敗（record_only）"
echo "round ${RN} done $(date -u +%FT%TZ)"
