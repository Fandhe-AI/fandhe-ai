#!/usr/bin/env bash
# #4150 の nsys 補助（record_only。PR-1 時点では未実行）。
# 役割: 5 round とは別の 1 回、example を nsys 配下で実行し、cuda_gpu_trace から L1 終了〜L2 開始の間隙と
#   起動〜開始の遅延を抽出する。nsys は時間を汚すため、時間値は判定に使わない（RULE 項 3）。
#   nsys 不在・不通・CPU サンプリング不可の警告（#2109 の前例）は unavailable＋理由として残す。
# 環境変数: ROOT=計測ツリー（run_round.sh が ROOT/target に固定してビルド済み。BIN は run_round.sh と同じ）  HW_LOCK_HELD=1  OUT=出力ディレクトリの絶対パス（新規）
# 引数なし。外部入力をシェル文字列へ展開しない。
set -uo pipefail
ROOT="${ROOT:?計測ツリー}"; OUT="${OUT:?出力ディレクトリ}"
case "${ROOT}${OUT}" in *\"*|*\'*|*\\*|*$'\n'*) echo "禁止文字を含む"; exit 2;; esac
case "${ROOT}" in /*) ;; *) echo "ROOT は絶対パスのみ"; exit 2;; esac
case "${OUT}" in /*) ;; *) echo "OUT は絶対パスのみ"; exit 2;; esac
[ "${HW_LOCK_HELD:-}" = "1" ] || { echo "実機ロック未取得（HW_LOCK_HELD=1 が必要）"; exit 2; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD が非空"; exit 2; }
[ ! -e "${OUT}" ] || { echo "${OUT} が既にある（上書き禁止）"; exit 2; }
mkdir -p "${OUT}"
if ! command -v nsys > /dev/null 2>&1; then
  echo "unavailable: nsys not found" > "${OUT}/nsys.txt"; exit 0
fi
BIN="${ROOT}/target/release/examples/cuda_infer_chain_event_diag_4150"
[ -x "${BIN}" ] || { echo "unavailable: example binary not built" > "${OUT}/nsys.txt"; exit 0; }
if ! nsys profile -t cuda -o "${OUT}/trace" --force-overwrite=true "${BIN}" --round 1 > "${OUT}/chain.txt" 2> "${OUT}/nsys.err"; then
  echo "unavailable: nsys profile failed" > "${OUT}/nsys.txt"; exit 0
fi
if nsys stats --report cuda_gpu_trace --format csv "${OUT}/trace.nsys-rep" > "${OUT}/cuda_gpu_trace.csv" 2> "${OUT}/stats.err"; then
  echo "ok: cuda_gpu_trace.csv（時間値は判定に使わない。間隙・起動遅延は手で読む）" > "${OUT}/nsys.txt"
else
  echo "unavailable: nsys stats failed" > "${OUT}/nsys.txt"
fi
