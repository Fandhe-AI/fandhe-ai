#!/usr/bin/env bash
# #4146 の 1 round（<metal-node> = Mac mini・Apple M4。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。values.tsv は
#   to_values_4146.py が raw/ から作る（計測と変換を分け、計測中に余計な処理を走らせない）。
# 派生元: docs/perf/logs/framework-compare-0.14.0-macmini-remeasure/scripts/macmini-run-0140.sh と同じ不変条件
#   （機体 Apple M4・.rev-stamp 照合・bench_py.py の sha256・絶対パス検証）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA）
#           D=0.14.0 計測と同じ作業ディレクトリ（py/bench_py.py と venv-torch を持つ）
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_HELD=1（hw-lock を取得済みであることの宣言）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# 順序: 奇数 round は Rust → Python、偶数 round は Python → Rust。ビルドは系列前に済ませる（SKIP しない場合はここで実行しない）。
# bash 3.2 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止（既存ファイルがあれば停止）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; D="${D:?作業ディレクトリ}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs D "${D}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "hw-lock 未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
PT="${D}/venv-torch/bin/python"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
[ "$(shasum -a 256 "${D}/py/bench_py.py" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
"${PT}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") else 1)' || { echo "torch の版が 0.14.0 計測（2.14.0）と違う"; exit 1; }
[ -x "${FC}/target/release/bench-fandhe" ] || { echo "bench-fandhe が未ビルド"; exit 1; }
[ -x "${ROOT}/target/release/examples/cpu_jobgrid_bench" ] || { echo "cpu_jobgrid_bench が未ビルド（workspace ルートで cargo build --release -p fandhe-ai-backend-cpu --example cpu_jobgrid_bench）"; exit 1; }

# ---- ゲート: load1 < 3.0 を 30 秒間隔で最大 30 分（hw-lock は上で宣言確認）
gate() {
  local i=0 raw l
  while [ "${i}" -le 60 ]; do
    if ! raw=$(sysctl -n vm.loadavg); then echo "INVALID sysctl-failed" > "${RD}/gate.txt"; return 2; fi
    l=$(printf '%s' "${raw}" | tr -d '{}' | awk '{print $1}')
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then echo "INVALID load1-not-numeric" > "${RD}/gate.txt"; return 2; fi
    if awk -v x="${l}" 'BEGIN{exit !(x < 3.0)}'; then echo "PASS load1=${l} hw-lock=held $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 0; fi
    sleep 30; i=$((i + 1))
  done
  echo "TIMEOUT load1=${l} $(date -u +%FT%TZ)" > "${RD}/gate.txt"; return 1
}
gate || { echo "ゲート不通過: $(cat "${RD}/gate.txt")"; exit 1; }

rust_arm() {
  : > "${RD}/raw/rust.jsonl"; : > "${RD}/raw/jobgrid.txt"
  local t m
  for t in train infer; do for m in fresh reuse; do
    "${FC}/target/release/bench-fandhe" --task "${t}" --device cpu --size 64 --mode "${m}" --out "${RD}/raw/rust.jsonl" 2> "${RD}/raw/rust.err" \
      || { echo "bench-fandhe ${t} ${m} 失敗"; return 1; }
  done; done
  "${ROOT}/target/release/examples/cpu_jobgrid_bench" "r${RN}" threads > "${RD}/raw/jobgrid.txt" 2> "${RD}/raw/jobgrid.err" || { echo "cpu_jobgrid_bench 失敗"; return 1; }
}
py_arm() {
  : > "${RD}/raw/py.jsonl"
  local t
  for t in train infer; do
    "${PT}" "${D}/py/bench_py.py" --framework pytorch --task "${t}" --device cpu --size 64 --out "${RD}/raw/py.jsonl" > /dev/null 2> "${RD}/raw/py.err" \
      || { echo "bench_py ${t} 失敗"; return 1; }
  done
  "${PT}" "${FC}/opp_kernel_torch_diag.py" measure --bench-py "${D}/py/bench_py.py" --device cpu --shapes mlp --arm base --run "${RN}" --out "${RD}/raw/opp_kernel.jsonl" 2> "${RD}/raw/opp_kernel.err" \
    || { echo "opp_kernel_torch_diag 失敗"; return 1; }
}
if [ $((RN % 2)) -eq 1 ]; then rust_arm && py_arm; else py_arm && rust_arm; fi || { echo "round ${RN} FAILED"; exit 1; }

# record_only（分類には使わない）: Tape 経路の区間内訳。失敗しても round は止めない（欠損は README に記す）
"${FC}/target/release/bench-fandhe" --task train --device cpu --size 64 --mode reuse --phases --out "${RD}/raw/phases-train.jsonl" 2> "${RD}/raw/phases-train.err" || echo "phases-train 失敗（record_only）"
"${FC}/target/release/bench-fandhe" --task infer --device cpu --size 64 --mode reuse --phases --out "${RD}/raw/phases-infer.jsonl" 2> "${RD}/raw/phases-infer.err" || echo "phases-infer 失敗（record_only）"
echo "round ${RN} done $(date -u +%FT%TZ)"
