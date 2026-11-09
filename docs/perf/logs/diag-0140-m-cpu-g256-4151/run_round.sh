#!/usr/bin/env bash
# #4151 の 1 round（<metal-node> = Mac mini・Apple M4。計測キット。PR-1 時点では未実行）。
# 役割: RULE.txt 項 3・4 の手順を 1 round 分実行し、round{i}/{gate.txt, raw/} を作る。
#   順序は precheck → gate → 計測（奇数 round は Rust → Python、偶数 round は Python → Rust）。
#   values.tsv は raw/ から別手順で作る（計測と変換を分け、計測中に余計な処理を走らせない。手順は README）。
# 派生元: docs/perf/logs/diag-0140-mlp-macmini-4146/run_round.sh と同じ不変条件
#   （機体 Apple M4・.rev-stamp 照合・bench_py.py の sha256・絶対パス検証・hw-lock 宣言・既存 raw/ の上書き拒否）。
# 環境変数（すべて絶対パス。ビルドはこのスクリプトの外で済ませる。README の「ビルド」節）:
#   ROOT_OFF_BASE／ROOT_ON_BASE／ROOT_OFF_TIMED／ROOT_ON_TIMED = 4 腕のツリー（.rev-stamp に完全 SHA。
#       target/release/examples/cpu_gemm_n256_fixed_cost_4151 ビルド済み。ROOT_OFF_BASE は --lib のテストバイナリも持つ）
#   BENCH_FANDHE_OFF／BENCH_FANDHE_ON = off-base／on-base ツリーに向けて作った bench-fandhe の実行ファイル
#   D = 作業ディレクトリ（py/bench_py.py と venv-torch を持つ）  FC_PY = torch_cpu_mlp_diag.py のあるディレクトリ
#   EXPECT_SHA = RULE.txt の tree_sha  HW_LOCK_HELD=1（hw-lock を取得済みであることの宣言）
# 引数: [--dry-run] $1=round 番号（1 始まり）$2=記録先 round ディレクトリ（絶対パス）
#   --dry-run は検証と実行予定のコマンドの表示だけで、何も実行・作成しない。
# bash 3.2 互換。計測中はビルド・集計・転送を並走させない。run の差し替え・上書きは禁止。
# 外部入力をシェル文字列へ展開しない（変数は常に引用符つきで引数として渡す。eval は使わない。変数名の間接参照は ${!v}）。
set -uo pipefail
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
validate_abs() { # $1=変数名 $2=値。絶対パスのみ。引用符・バックスラッシュ・改行・空白を拒否（A03）
  case "$2" in /*) ;; *) echo "$1 は絶対パスのみ: $2"; exit 2;; esac
  case "$2" in *\"*|*\'*|*\\*|*\$*|*\`*|*$'\n'*|*' '*) echo "$1 に禁止文字を含む"; exit 2;; esac
}
for v in ROOT_OFF_BASE ROOT_ON_BASE ROOT_OFF_TIMED ROOT_ON_TIMED BENCH_FANDHE_OFF BENCH_FANDHE_ON D FC_PY EXPECT_SHA; do
  [ -n "${!v:-}" ] || { echo "$v が未設定"; exit 2; }
done
RN="${1:?round 番号}"; RD="${2:?round ディレクトリ}"
for v in ROOT_OFF_BASE ROOT_ON_BASE ROOT_OFF_TIMED ROOT_ON_TIMED BENCH_FANDHE_OFF BENCH_FANDHE_ON D FC_PY; do
  validate_abs "$v" "${!v}"
done
validate_abs RD "${RD}"
case "${RN}" in ''|*[!0-9]*) echo "round 番号は数字のみ"; exit 2;; esac
case "${EXPECT_SHA}" in *[!0-9a-f]*|'') echo "EXPECT_SHA は小文字 hex のみ"; exit 2;; esac
[ "${#EXPECT_SHA}" -eq 40 ] || { echo "EXPECT_SHA は 40 桁"; exit 2; }
[ "${RAYON_NUM_THREADS:-}" = "" ] || { echo "RAYON_NUM_THREADS は設定しない（RULE 項 4）"; exit 2; }
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "hw-lock 未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
BENCH_PY_SHA256="9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b"
PT="${D}/venv-torch/bin/python"
BP="${D}/py/bench_py.py"
EXN="target/release/examples/cpu_gemm_n256_fixed_cost_4151"
PROBE_FILTER="gemm_blis::tests::sme_microcall_fixed_cost_probe"

if [ "${DRY}" = "1" ]; then
  echo "[dry-run] round ${RN} -> ${RD}"
  echo "[dry-run] precheck: Apple M4・.rev-stamp×4=${EXPECT_SHA}・bench_py.py sha256・torch 2.14.0・各ビルド成果物"
  echo "[dry-run] gate: hw-lock 宣言済み・load1 < 3.0 を 30 秒間隔で最大 30 分 -> ${RD}/gate.txt"
  for a in off-base on-base off-timed on-timed; do echo "[dry-run] example ${a}: <ROOT_$(echo "${a}" | tr 'a-z-' 'A-Z_')>/${EXN} ${a} > ${RD}/raw/example-${a}.txt"; done
  echo "[dry-run] bench-fandhe: --task gemm --device cpu --size 256 --mode fresh|reuse（off／on）"
  echo "[dry-run] bench_py.py: --framework pytorch --task gemm --device cpu --size 256"
  echo "[dry-run] torch_cpu_mlp_diag.py measure --section gemm256 --run ${RN}"
  echo "[dry-run] cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored ${PROBE_FILTER} --exact --nocapture --test-threads=1（off-base ツリーだけ）"
  exit 0
fi

[ ! -e "${RD}/raw" ] || { echo "${RD}/raw が既にある（上書き禁止）"; exit 2; }
mkdir -p "${RD}/raw"

# ---- precheck（fail-closed）
[ "$(sysctl -n machdep.cpu.brand_string)" = "Apple M4" ] || { echo "機体が Apple M4 ではない"; exit 1; }
for r in "${ROOT_OFF_BASE}" "${ROOT_ON_BASE}" "${ROOT_OFF_TIMED}" "${ROOT_ON_TIMED}"; do
  [ "$(cat "${r}/.rev-stamp")" = "${EXPECT_SHA}" ] || { echo ".rev-stamp が RULE と不一致: ${r}"; exit 1; }
  [ -x "${r}/${EXN}" ] || { echo "example が未ビルド: ${r}"; exit 1; }
done
[ -x "${BENCH_FANDHE_OFF}" ] && [ -x "${BENCH_FANDHE_ON}" ] || { echo "bench-fandhe が未ビルド"; exit 1; }
[ "$(shasum -a 256 "${BP}" | cut -d' ' -f1)" = "${BENCH_PY_SHA256}" ] || { echo "bench_py.py の sha256 不一致"; exit 1; }
"${PT}" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.14.0") else 1)' || { echo "torch の版が 0.14.0 計測（2.14.0）と違う"; exit 1; }

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

# 腕の順序は奇数 round と偶数 round で入れ替える（腕の順序による位置効果を相殺する）。
if [ $((RN % 2)) -eq 1 ]; then ORDER="off-base on-base off-timed on-timed"; else ORDER="on-timed off-timed on-base off-base"; fi
run_example() {
  local a="$1" root
  case "${a}" in
    off-base) root="${ROOT_OFF_BASE}";; on-base) root="${ROOT_ON_BASE}";;
    off-timed) root="${ROOT_OFF_TIMED}";; on-timed) root="${ROOT_ON_TIMED}";;
    *) echo "未知の腕"; return 1;;
  esac
  "${root}/${EXN}" "${a}" > "${RD}/raw/example-${a}.txt" 2> "${RD}/raw/example-${a}.err" || { echo "example ${a} 失敗"; return 1; }
}
rust_arm() {
  : > "${RD}/raw/rust.jsonl"
  local a m
  for a in ${ORDER}; do run_example "${a}" || return 1; done
  for m in fresh reuse; do
    "${BENCH_FANDHE_OFF}" --task gemm --device cpu --size 256 --mode "${m}" --out "${RD}/raw/rust.jsonl" 2>> "${RD}/raw/rust-off.err" \
      || { echo "bench-fandhe off ${m} 失敗"; return 1; }
  done
  : > "${RD}/raw/rust-on.jsonl"
  for m in fresh reuse; do
    "${BENCH_FANDHE_ON}" --task gemm --device cpu --size 256 --mode "${m}" --out "${RD}/raw/rust-on.jsonl" 2>> "${RD}/raw/rust-on.err" \
      || { echo "bench-fandhe on ${m} 失敗"; return 1; }
  done
}
# 1 呼び出し固定費の上限の参照用 probe。cargo test 起動が CPU 負荷になるため、GEMM・PyTorch の計測窓が両方終わった後に走らせる
# （分類に使う O・K_opp を汚さない）。失敗は非致命（c_sme を values.tsv に書かず、ON の上限欄は unavailable）。
sme_probe() {
  # 1 呼び出し固定費の上限（off-base ツリーでだけ。ON ツリーの --lib はドリフト検出テストが落ちる）
  ( cd "${ROOT_OFF_BASE}" && cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored "${PROBE_FILTER}" --exact --nocapture --test-threads=1 ) \
    > "${RD}/raw/sme-probe.txt" 2> "${RD}/raw/sme-probe.err" \
    && grep -q "probe=microcall" "${RD}/raw/sme-probe.txt" \
    || echo "sme probe 失敗またはレコード未出力（c_sme が欠損。ON の上限欄は unavailable）"
}
py_arm() {
  : > "${RD}/raw/py.jsonl"
  "${PT}" "${BP}" --framework pytorch --task gemm --device cpu --size 256 --out "${RD}/raw/py.jsonl" > /dev/null 2> "${RD}/raw/py.err" \
    || { echo "bench_py 失敗"; return 1; }
  "${PT}" "${FC_PY}/torch_cpu_mlp_diag.py" measure --bench-py "${BP}" --section gemm256 --run "${RN}" --out "${RD}/raw/measure_gemm256.jsonl" 2> "${RD}/raw/measure_gemm256.err" \
    || { echo "torch_cpu_mlp_diag 失敗"; return 1; }
}
if [ $((RN % 2)) -eq 1 ]; then rust_arm && py_arm; else py_arm && rust_arm; fi || { echo "round ${RN} FAILED"; exit 1; }
sme_probe

# 生出力のホスト名・ホームの絶対パスをマスクする（公開面。RULE 項 8）
for f in "${RD}"/raw/*; do
  [ -f "${f}" ] || continue
  sed -e "s#${HOME}#<home>#g" -e "s#$(hostname)#<metal-node>#g" "${f}" > "${f}.masked" && mv "${f}.masked" "${f}"
done
echo "round ${RN} done $(date -u +%FT%TZ)"
