#!/usr/bin/env bash
# #4154 の 1 round（<metal-node> = Mac mini・Apple M4。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。values.tsv は
#   aggregate_4154.py --write-values が raw/ から作る（計測と変換を分け、計測中に余計な処理を走らせない）。
# 派生元: docs/perf/logs/diag-0140-torch-cpu-mlp-macmini-4148/run_round.sh と同じ不変条件
#   （機体 Apple M4・.rev-stamp 照合・bench_py.py の sha256・絶対パス検証・hw-lock 宣言・既存 raw/ の上書き拒否）。
# 環境変数: ROOT=計測ツリー（tree_sha の git archive 展開先。.rev-stamp に完全 SHA）
#           D=作業ディレクトリ（py/bench_py.py と venv-torch を持つ）
#           EXPECT_SHA=RULE.txt の tree_sha  HW_LOCK_HELD=1（hw-lock を取得済みであることの宣言）
#           TESTBIN=prebuild.sh が記録した test-bin.path の中身（backend-metal の lib テストバイナリ。絶対パス）
# 引数: $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス。round{i}）
# 順序（RULE 項 3）: 奇数 round は A0→B0→B1→TW→TS→TE→TF、偶数 round は TW→TS→TE→TF→A0→B0→B1。
# bash 3.2 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止。
# 外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す）。
set -uo pipefail
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*$'\n'*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
ROOT="${ROOT:?計測ツリー}"; D="${D:?作業ディレクトリ}"; SHA="${EXPECT_SHA:?RULE の tree_sha}"; TESTBIN="${TESTBIN:?B0/B1 のテストバイナリ}"
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
validate_abs ROOT "${ROOT}"; validate_abs D "${D}"; validate_abs RD "${RD}"; validate_abs TESTBIN "${TESTBIN}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "hw-lock 未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
FC="${ROOT}/scripts/bench/framework-compare"
PT="${D}/venv-torch/bin/python"
BP="${D}/py/bench_py.py"
DIAG="${FC}/torch_mps_train_diag.py"
[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
[ "$(cat "${ROOT}/.rev-stamp")" = "${SHA}" ] || { echo ".rev-stamp が RULE と不一致"; exit 1; }
[ "$(shasum -a 256 "${BP}" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
"${PT}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") and torch.backends.mps.is_available() else 1)' \
  || { echo "torch が 2.14.0 でない、または MPS が使えない"; exit 1; }
[ -x "${FC}/target/release/bench-fandhe" ] || { echo "bench-fandhe が未ビルド（prebuild.sh）"; exit 1; }
[ -x "${TESTBIN}" ] || { echo "TESTBIN が実行できない（prebuild.sh）"; exit 1; }

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

"${PT}" "${DIAG}" env --bench-py "${BP}" --out "${RD}/raw/env.json" 2> "${RD}/raw/env.err" || { echo "env 失敗"; exit 1; }

arm_a0() {
  "${FC}/target/release/bench-fandhe" --task train --device metal --size 64 --mode reuse --out "${RD}/raw/a0.jsonl" 2> "${RD}/raw/a0.err" \
    || { echo "A0 失敗"; return 1; }
}
arm_b() { # $1=b0|b1
  BREAKDOWN_3612_OUT="${RD}/raw/$1.jsonl" "${TESTBIN}" --ignored --nocapture --test-threads=1 --exact \
    "resident_chain_breakdown_diag_tests_3612::resident_chain_breakdown_3612_train_$1" > "${RD}/raw/$1.out" 2> "${RD}/raw/$1.err" \
    || { echo "$1 失敗"; return 1; }
}
arm_tw() {
  "${PT}" "${BP}" --framework pytorch --task train --device metal --size 64 --out "${RD}/raw/tw.jsonl" > /dev/null 2> "${RD}/raw/tw.err" \
    || { echo "TW 失敗"; return 1; }
}
arm_t() { # $1=split|event|floor $2=出力名
  "${PT}" "${DIAG}" measure --bench-py "${BP}" --arm "$1" --run "${RN}" --out "${RD}/raw/$2.jsonl" 2> "${RD}/raw/$2.err" \
    || { echo "$1 失敗"; return 1; }
}
fandhe_arms() { arm_a0 && arm_b b0 && arm_b b1; }
torch_arms() { arm_tw && arm_t split ts && arm_t event te && arm_t floor tf; }
if [ $((RN % 2)) -eq 1 ]; then fandhe_arms && torch_arms; else torch_arms && fandhe_arms; fi || { echo "round ${RN} FAILED"; exit 1; }
echo "round ${RN} done $(date -u +%FT%TZ)"
