#!/bin/bash
# イシュー #3215: GB10 の gemm CUDA セル（N=256〜4096 x fresh/reuse）を 5 round 測る（現行のみ・record_only）。
# 役割: 既存の `run_gemm_gate.sh` は cuda の N=1024/2048/4096 かつ burn なしで固定のため、本スクリプトが
#   N=256〜4096 の 5 点・bench-fandhe（fresh/reuse）・bench-candle（fresh）・bench-burn（fresh。TF32 固定）を
#   同一 round で測る。判定ロジックは持たない（集計は docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/aggregate.py）。
# 継承する安全策（run_gemm_gate.sh・run_ab_frag_interleave_gemm_metal.sh と同一方針。重複実装ではなく同じ規則の再適用）:
#   - ラベルは [A-Za-z0-9._-]+ の allowlist、facade パスは絶対パスかつ引用符・バックスラッシュ禁止（A03）
#   - `bench_fandhe_lock_restore.sh`（path patch で書き換わる Cargo.lock の退避・復元）
#   - ビルド直後に 3 バイナリの sha256 と依存解決元を manifest に記録し、計測直前に再照合（fail-closed。A08）
#   - ROUNDS=5 固定。round ごとに起動順（fandhe 先／相手先）を反転する
#   - 各 round 先頭で load1 < 1.0 かつ GPU 利用率 0% の専有ゲート（30 秒間隔・最大 30 分。不通過は GATE-TIMEOUT で停止）
# 使い方（GB10 上）:
#   CELL_OUT=<出力ディレクトリ(未作成の絶対パス)> bash run_cell_gemm_cuda_3215.sh <label> <crates/facade の絶対パス>
# 出力: ${CELL_OUT}/results.jsonl（bench の JSONL）・manifest.json・skipped.log・load_gate.log・uptime_*.txt・done 行
set -euo pipefail
export LC_ALL=C
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=./bench_fandhe_lock_restore.sh
source ./bench_fandhe_lock_restore.sh

LABEL=${1:-}
FACADE_PATH=${2:-}
OUT=${CELL_OUT:-}
SIZES=(256 512 1024 2048 4096)
ROUNDS=5
GATE_LOAD=1.0

if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: CELL_OUT=<dir> $0 <label [A-Za-z0-9._-]+> <facade abs path>" >&2
  exit 1
fi
if [[ -z "$FACADE_PATH" || "$FACADE_PATH" != /* || "$FACADE_PATH" == *'"'* || "$FACADE_PATH" == *'\'* ]]; then
  echo "ERROR: facade パスは絶対パスで、二重引用符・バックスラッシュを含められない" >&2
  exit 1
fi
if [[ ! -f "${FACADE_PATH}/Cargo.toml" ]]; then
  echo "ERROR: ${FACADE_PATH}/Cargo.toml が無い" >&2
  exit 1
fi
if [[ -z "$OUT" || "$OUT" != /* || -e "$OUT" ]]; then
  echo "ERROR: CELL_OUT は未作成の絶対パスで指定する（上書き禁止）" >&2
  exit 1
fi
mkdir -p "$OUT"
RESULTS="${OUT}/results.jsonl"
SKIPPED="${OUT}/skipped.log"
GATE_LOG="${OUT}/load_gate.log"
MANIFEST="${OUT}/manifest.json"
: >"$RESULTS"
: >"$SKIPPED"

bench_fandhe_setup_lock_restore_trap

sha256_of() { bench_fandhe_sha256_of "$1"; }

dep_source() { # dep_source <package> <crate> [cargo args...]
  local pkg=$1 crate=$2 line path_part
  shift 2
  line=$(cargo tree -p "$pkg" --depth 1 "$@" 2>/dev/null | grep -E "^[├└]── ${crate} " || true)
  if [[ -z "$line" ]]; then
    return 1
  elif [[ "$line" == *"("* ]]; then
    path_part=${line#*(}
    echo "path:${path_part%)}"
  else
    echo "registry"
  fi
}

CARGO_CONFIG_ARGS=(--config "patch.crates-io.fandhe-ai.path=\"${FACADE_PATH}\"")

echo "== build bench-fandhe (path patch) =="
cargo build --release -p bench-fandhe "${CARGO_CONFIG_ARGS[@]}"
echo "== build bench-candle (cuda) =="
cargo build --release -p bench-candle --no-default-features --features cuda
echo "== build bench-burn (cuda) =="
cargo build --release -p bench-burn --no-default-features --features cuda

fandhe_src=$(dep_source bench-fandhe fandhe-ai "${CARGO_CONFIG_ARGS[@]}") || { echo "ERROR: fandhe-ai の依存元を取得できない" >&2; exit 1; }
if [[ "$fandhe_src" != "path:${FACADE_PATH}" ]]; then
  echo "ERROR: fandhe-ai の依存元が path:${FACADE_PATH} でない（actual=${fandhe_src}）" >&2
  exit 1
fi
candle_src=$(dep_source bench-candle candle-core) || { echo "ERROR: candle-core の依存元を取得できない" >&2; exit 1; }
burn_src=$(dep_source bench-burn burn) || { echo "ERROR: burn の依存元を取得できない" >&2; exit 1; }
if [[ "$candle_src" != "registry" || "$burn_src" != "registry" ]]; then
  echo "ERROR: 相手側の依存元が registry でない（candle=${candle_src} burn=${burn_src}）" >&2
  exit 1
fi
SHA_FANDHE=$(sha256_of ./target/release/bench-fandhe)
SHA_CANDLE=$(sha256_of ./target/release/bench-candle)
SHA_BURN=$(sha256_of ./target/release/bench-burn)
printf '{"label":"%s","bench_fandhe_sha256":"%s","bench_candle_sha256":"%s","bench_burn_sha256":"%s","fandhe_ai_source":"%s","candle_core_source":"%s","burn_source":"%s"}\n' \
  "$LABEL" "$SHA_FANDHE" "$SHA_CANDLE" "$SHA_BURN" "path:<facade>" "$candle_src" "$burn_src" >"$MANIFEST"

verify_binaries() { # 計測直前の再照合（ビルド〜計測の間のすり替えを検出）
  [[ "$(sha256_of ./target/release/bench-fandhe)" == "$SHA_FANDHE" ]] || { echo "ERROR: bench-fandhe の sha256 が manifest と不一致" >&2; exit 1; }
  [[ "$(sha256_of ./target/release/bench-candle)" == "$SHA_CANDLE" ]] || { echo "ERROR: bench-candle の sha256 が manifest と不一致" >&2; exit 1; }
  [[ "$(sha256_of ./target/release/bench-burn)" == "$SHA_BURN" ]] || { echo "ERROR: bench-burn の sha256 が manifest と不一致" >&2; exit 1; }
  echo "== バイナリ sha256 照合 OK =="
}

wait_gate() { # $1=ラベル
  local i=0 load util
  while [[ "$i" -le 60 ]]; do
    load="$(uptime | sed -E 's/.*load averages?: *([0-9.]+).*/\1/')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    case "$load" in '' | *[!0-9.]*) echo "GATE-INVALID $1 load=${load}" >>"$GATE_LOG"; exit 5 ;; esac
    case "$util" in '' | *[!0-9]*) echo "GATE-INVALID $1 util=${util}" >>"$GATE_LOG"; exit 5 ;; esac
    if awk -v l="$load" -v t="$GATE_LOAD" 'BEGIN{exit !(l<t)}' && [[ "$util" == 0 ]]; then
      echo "GATE-PASS $1 load1=${load} gpu_util=${util}" >>"$GATE_LOG"
      return 0
    fi
    i=$((i + 1))
    sleep 30
  done
  echo "GATE-TIMEOUT $1" >>"$GATE_LOG"
  exit 6
}

ANY_FAILED=0
run() { # run <binary> <size> <mode>
  local bin=$1 size=$2 mode=$3
  echo "== $bin gemm cuda size=$size mode=$mode =="
  if ! "./target/release/$bin" --task gemm --device cuda --size "$size" --mode "$mode" --out "$RESULTS" 2>"${OUT}/err.tmp"; then
    echo "$bin size=$size mode=$mode : $(cat "${OUT}/err.tmp")" >>"$SKIPPED"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "${OUT}/err.tmp"
}

verify_binaries
for round in $(seq 1 "$ROUNDS"); do
  wait_gate "cell-round${round}"
  uptime >"${OUT}/uptime_before_round${round}.txt"
  for n in "${SIZES[@]}"; do
    if [[ $((round % 2)) -eq 1 ]]; then
      run bench-fandhe "$n" reuse
      run bench-fandhe "$n" fresh
      run bench-candle "$n" fresh
      run bench-burn "$n" fresh
    else
      run bench-burn "$n" fresh
      run bench-candle "$n" fresh
      run bench-fandhe "$n" fresh
      run bench-fandhe "$n" reuse
    fi
  done
done

if [[ "$ANY_FAILED" -ne 0 ]]; then
  echo "FAILED: ${ANY_FAILED} run(s) failed (see ${SKIPPED}). 結果は破棄せず保持するが完走扱いにしない" >&2
  exit 1
fi
echo "done."
