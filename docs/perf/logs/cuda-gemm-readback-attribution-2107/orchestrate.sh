#!/usr/bin/env bash
# イシュー #2107: CUDA GEMM reuse readback 宛先確保の帰属検証オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。
# `crates/backend-cuda` のテストバイナリを 1 回ビルドし、(N, 腕) ごとに独立
# プロセスで 5 run 起動して run{k}/n{N}_{arm}.jsonl を残す。同一セッションで
# Layer A（bench-fandhe gemm reuse --phases。fandhe-ai =0.9.0 ピン）も
# 5 run 実行する（#1973 の orchestrate.sh と同型の構成）。
# 使い方: orchestrate.sh <gb10|x86> [--dry-run] [--out <dir>]
# 集計は `python3 aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
# 入力は固定の選択肢のみ（eval・未検証入力のコマンド展開は使わない）。
set -euo pipefail

usage() { echo "usage: $0 <gb10|x86> [--dry-run] [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "$host_kind" ] || usage
shift
case "$host_kind" in gb10 | x86) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
out_dir="$script_dir/$host_kind"
dry_run=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --out)
      [ $# -ge 2 ] || usage
      out_dir="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

sizes=(1024 2048 4096)
arms=(
  clone_dtoh_legacy_to_vec
  clone_dtoh_borrowed_keep_alive
  clone_dtoh_borrowed_dummy_alloc_free
  pretouched_fresh_split
  pretouched_fresh_production
  prod_order_fresh
  prod_order_reused
)
runs=5
test_prefix="readback_attribution_diag_tests_2107::readback_attribution_2107"
fc_dir="$repo_root/scripts/bench/framework-compare"

# run ごとに腕の順序を巡回させる（k-1 個ずつ先頭をずらす）。
rotated_arms() {
  local k="$1" i
  local n=${#arms[@]}
  for ((i = 0; i < n; i++)); do
    echo "${arms[$(((i + k - 1) % n))]}"
  done
}

if [ "$dry_run" = "1" ]; then
  echo "dry-run: host_kind=$host_kind out_dir=$out_dir"
  echo "dry-run: build: cargo test --release --locked -p fandhe-ai-backend-cuda --lib --no-run"
  echo "dry-run: build: (cd $fc_dir && cargo build --release -p bench-fandhe)"
  for ((k = 1; k <= runs; k++)); do
    echo "dry-run: run$k arm order: $(rotated_arms "$k" | tr '\n' ' ')"
    for n in "${sizes[@]}"; do
      while read -r arm; do
        echo "dry-run:   <bin> --ignored --exact ${test_prefix}_n${n}_${arm} --nocapture --test-threads=1 -> run$k/n${n}_${arm}.jsonl"
      done < <(rotated_arms "$k")
    done
    for n in "${sizes[@]}"; do
      echo "dry-run:   Layer A: bench-fandhe --task gemm --device cuda --size $n --mode reuse --phases -> layerA-phases-N${n}.log"
    done
  done
  echo "dry-run: OK"
  exit 0
fi

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for ((k = 1; k <= runs; k++)); do
  if [ -d "$out_dir/run$k" ] && [ -n "$(ls -A "$out_dir/run$k")" ]; then
    echo "既存の run$k があるため中止する（差し替え禁止）" >&2
    exit 1
  fi
done
for n in "${sizes[@]}"; do
  if [ -e "$out_dir/layerA-phases-N${n}.log" ]; then
    echo "既存の layerA-phases-N${n}.log があるため中止する（差し替え禁止）" >&2
    exit 1
  fi
done

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

cd "$repo_root"
bin="$(cargo test --release --locked -p fandhe-ai-backend-cuda --lib --no-run \
  --message-format=json 2>/dev/null |
  python3 -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if (m.get("reason") == "compiler-artifact" and m.get("executable")
            and m.get("target", {}).get("name") == "fandhe_ai_backend_cuda"
            and m.get("profile", {}).get("test")):
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
')"
[ -x "$bin" ] || { echo "テストバイナリを特定できない" >&2; exit 1; }
(cd "$fc_dir" && cargo build --release -p bench-fandhe)
bench_fandhe="$fc_dir/target/release/bench-fandhe"
[ -x "$bench_fandhe" ] || { echo "bench-fandhe を特定できない" >&2; exit 1; }

# Layer A（fandhe-ai =0.9.0 crates.io 版）と Layer B（HEAD）の計測経路が同一か。
# crates/backend-cuda のファイル単位 diff は #2299 の feature gate・ドキュメント・
# 診断専用機能追加で常に差分が出て判定が恒常的に無効化されるため、Layer A の
# 計測経路（GEMM tile 選択・起動選択・tiled カーネルソース・D2H readback 宛先確保・
# 生成／起動経路）の項目に限定し、コメントと診断 feature ゲート項目を除いた正規化テキストで比較する
# （RULE.txt「同一コード確認」節。差分・抽出不能なら判定は「無効（参考扱い）」）。
same_code=unknown
path_identity_detail=""
if git rev-parse -q --verify "refs/tags/v0.9.0" >/dev/null 2>&1; then
  path_identity_out="$(python3 "$script_dir/check_layer_a_path_identity.py" v0.9.0 HEAD || true)"
  same_code="$(printf '%s\n' "$path_identity_out" | sed -n 's/^layerA_same_code: //p' | head -1)"
  case "$same_code" in yes | no | unknown) ;; *) same_code=unknown ;; esac
  path_identity_detail="$(printf '%s\n' "$path_identity_out" | grep '^path_item:' || true)"
fi

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "host_kind: $host_kind"
  echo "layerA_same_code: $same_code"
  [ -z "$path_identity_detail" ] || echo "$path_identity_detail"
  echo "os_release: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi: unavailable"
  nvcc --version 2>/dev/null | tail -1 || echo "nvcc: unavailable"
  echo "load_avg: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"
  echo "gpu_util_start: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || echo unknown)"
} 2>&1 | mask >"$out_dir/env_info.txt"

load1() { cut -d' ' -f1 /proc/loadavg; }
gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

: >"$out_dir/load_gate_status.txt"
: >"$out_dir/load_gate.log"
for ((k = 1; k <= runs; k++)); do
  if [ "$host_kind" = "gb10" ]; then
    gate_status=pass
    waited=0
    while :; do
      l="$(load1)"
      g="$(gpu_util)"
      echo "run$k load1=$l gpu_util=$g waited=${waited}s" >>"$out_dir/load_gate.log"
      # 空・非数値は通過扱いにしない。
      if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "$g" =~ ^[0-9]+$ ]] &&
        awk -v l="$l" 'BEGIN{exit !(l < 1.0)}' && [ "$g" -eq 0 ]; then
        break
      fi
      if [ "$waited" -ge 1800 ]; then
        echo "run$k: 専有ゲート未通過（参考扱い）" >>"$out_dir/load_gate.log"
        gate_status=unpassed
        break
      fi
      sleep 30
      waited=$((waited + 30))
    done
  else
    gate_status=record_only
    echo "run$k load1=$(load1) (record_only)" >>"$out_dir/load_gate.log"
  fi
  echo "run$k gate=$gate_status" >>"$out_dir/load_gate_status.txt"

  mkdir -p "$out_dir/run$k"
  for n in "${sizes[@]}"; do
    while read -r arm; do
      name="${test_prefix}_n${n}_${arm}"
      "$bin" --ignored --exact "$name" --nocapture --test-threads=1 \
        >"$out_dir/run$k/n${n}_${arm}.raw" 2>"$out_dir/run$k/n${n}_${arm}.err.raw"
      grep -o 'DIAG_JSON .*' "$out_dir/run$k/n${n}_${arm}.raw" |
        sed -e 's/^DIAG_JSON //' >"$out_dir/run$k/n${n}_${arm}.jsonl"
      mask <"$out_dir/run$k/n${n}_${arm}.err.raw" >"$out_dir/run$k/n${n}_${arm}.err"
      rm -f "$out_dir/run$k/n${n}_${arm}.raw" "$out_dir/run$k/n${n}_${arm}.err.raw"
    done < <(rotated_arms "$k")
  done

  # 同一セッションの Layer A（各 N を run 内で 1 回ずつ。log は 5 run 連結）。
  for n in "${sizes[@]}"; do
    {
      echo "-- run $k/N=$n --"
      "$bench_fandhe" --task gemm --device cuda --size "$n" --mode reuse --phases 2>&1
    } | mask >>"$out_dir/layerA-phases-N${n}.log"
  done
done

{
  echo "計測終了時刻（UTC）: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "load_avg_end: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"
} >>"$out_dir/env_info.txt"
echo "完了: $out_dir（次: python3 $script_dir/aggregate.py $out_dir）"
