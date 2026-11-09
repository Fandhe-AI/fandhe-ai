#!/usr/bin/env bash
# #4149 の 1 round（<cuda-node> = DGX GB10。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。
#   順序は gate → env → CPU measure（cpu_train・cpu_infer・cpu_shapes）→ CUDA measure（cuda_infer・cuda_fixed・cuda_shapes）
#   → profile（cpu_train・cpu_infer・cuda_infer。計時から分離した別腕・最後）。観測専用 shim（probe/）は全 round の後に別途 1 回。
# 派生元: diag-0140-torch-cpu-mlp-macmini-4148/run_round.sh（腕の構成）と diag-0140-mlp-gb10-4147/run_round.sh（GB10 のゲート・実機ロックの作法）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA）
#           PY=torch 2.14.0 入りの python  BP=bench_py.py の絶対パス
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_OWNER_FILE=実機ロックの owner ファイル（空でないこと）
#           HW_LOCK_HELD=1（実機ロックを取得済みであることの宣言）  PROFILE_STEPS=RULE.txt の profile_steps（既定 50）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# bash 3.2／5 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止。
# 外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; PY="${PY:?torch 入り python}"; BP="${BP:?bench_py.py}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"
LOCKF="${HW_LOCK_OWNER_FILE:?実機ロックの owner ファイル}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
STEPS="${PROFILE_STEPS:-50}"
validate_abs ROOT "${ROOT}"; validate_abs PY "${PY}"; validate_abs BP "${BP}"; validate_abs LOCKF "${LOCKF}"; validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${STEPS}" in ''|*[!0-9]*) echo "PROFILE_STEPS は数字のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
# 計時腕は shim・preload を一切載せない（観測専用 shim は別腕）。
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD が非空（計時腕では拒否）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
DIAG="${FC}/torch_gb10_mlp_diag.py"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ -s "${LOCKF}" ] || { echo "実機ロックの owner が空（ロックなしの値は採らない）"; exit 1; }
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'GB10' || { echo "機体が GB10 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
echo "${BENCH_PY_SHA256}  ${BP}" | sha256sum -c - > /dev/null || { echo "bench_py.py の sha256 不一致"; exit 1; }
[ -f "${DIAG}" ] || { echo "torch_gb10_mlp_diag.py がない"; exit 1; }
"${PY}" -c 'import torch,sys; sys.exit(0 if (torch.__version__.startswith("2.14.0") and torch.cuda.is_available()) else 1)' \
  || { echo "torch が 2.14.0 でない、または CUDA が使えない"; exit 1; }

# ---- ゲート: compute プロセス 0 件・GPU 使用率 0 が 3 サンプル連続・load1 < 1.0 を 30 秒間隔で最大 30 分（#4147 と同一）。
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

"${PY}" "${DIAG}" env --bench-py "${BP}" --out "${RD}/raw/env.json" 2> "${RD}/raw/env.err" || { echo "env 失敗"; exit 1; }
for s in cpu_train cpu_infer cpu_shapes cuda_infer cuda_fixed cuda_shapes; do
  "${PY}" "${DIAG}" measure --bench-py "${BP}" --section "${s}" --run "${RN}" --out "${RD}/raw/measure_${s}.jsonl" 2> "${RD}/raw/measure_${s}.err" \
    || { echo "measure ${s} 失敗"; exit 1; }
done
# profiler 腕は計時腕のあと。時間値は計時腕と混ぜない（op 数・kernel／API 回数のみ使う）。
for s in cpu_train cpu_infer cuda_infer; do
  "${PY}" "${DIAG}" profile --bench-py "${BP}" --section "${s}" --run "${RN}" --steps "${STEPS}" --out "${RD}/raw/profile_${s}.json" 2> "${RD}/raw/profile_${s}.err" \
    || { echo "profile ${s} 失敗"; exit 1; }
done
echo "round ${RN} done $(date -u +%FT%TZ)"
