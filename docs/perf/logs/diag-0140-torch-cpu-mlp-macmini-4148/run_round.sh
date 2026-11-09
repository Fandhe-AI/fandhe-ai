#!/usr/bin/env bash
# #4148 の 1 round（<metal-node> = Mac mini・Apple M4。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。
#   順序は gate → env → measure（train・infer・gemm256・shapes）→ profile（計時から分離した別腕・最後）。
# 派生元: docs/perf/logs/diag-0140-mlp-macmini-4146/run_round.sh と同じ不変条件
#   （機体 Apple M4・.rev-stamp 照合・bench_py.py の sha256・絶対パス検証・hw-lock 宣言・既存 raw/ の上書き拒否）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA）
#           D=作業ディレクトリ（py/bench_py.py と venv-torch を持つ）
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_HELD=1（hw-lock を取得済みであることの宣言）
#           PROFILE_STEPS=RULE.txt の profile_steps（既定 50）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# bash 3.2 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止。
# 外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; D="${D:?作業ディレクトリ}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
STEPS="${PROFILE_STEPS:-50}"
validate_abs ROOT "${ROOT}"; validate_abs D "${D}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${STEPS}" in ''|*[!0-9]*) echo "PROFILE_STEPS は数字のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "hw-lock 未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
PT="${D}/venv-torch/bin/python"
BP="${D}/py/bench_py.py"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
[ "$(shasum -a 256 "${BP}" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
"${PT}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") else 1)' || { echo "torch の版が 2.14.0 ではない"; exit 1; }

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

DIAG="${FC}/torch_cpu_mlp_diag.py"
"${PT}" "${DIAG}" env --bench-py "${BP}" --out "${RD}/raw/env.json" 2> "${RD}/raw/env.err" || { echo "env 失敗"; exit 1; }
for s in train infer gemm256 shapes; do
  "${PT}" "${DIAG}" measure --bench-py "${BP}" --section "${s}" --run "${RN}" --out "${RD}/raw/measure_${s}.jsonl" 2> "${RD}/raw/measure_${s}.err" \
    || { echo "measure ${s} 失敗"; exit 1; }
done
# profiler 腕は計時腕のあと。時間値は計時腕と混ぜない（op 数のみ使う）。
for s in train infer gemm256; do
  "${PT}" "${DIAG}" profile --bench-py "${BP}" --section "${s}" --run "${RN}" --steps "${STEPS}" --out "${RD}/raw/profile_${s}.json" 2> "${RD}/raw/profile_${s}.err" \
    || { echo "profile ${s} 失敗"; exit 1; }
done
echo "round ${RN} done $(date -u +%FT%TZ)"
