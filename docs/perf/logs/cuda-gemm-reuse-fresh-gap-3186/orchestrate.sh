#!/usr/bin/env bash
# イシュー #3186: CUDA GEMM の reuse が fresh より遅い原因の切り分けオーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。#2107 の orchestrate.sh と
# 同型で、テストバイナリを 1 回ビルドし (N, 腕) ごとに独立プロセスで 5 run 起動して
# run{k}/n{N}_{arm}.jsonl を残す。同一セッションで Layer A（bench-fandhe gemm の
# fresh／reuse。HEAD の crates/facade を path patch で差し込む）も実行する。
# 使い方: orchestrate.sh <gb10|x86> [--dry-run] [--out <dir>]
# 集計は `python3 aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
# 入力は固定の選択肢のみ（eval・未検証入力のコマンド展開は使わない）。
# ノードのグローバル状態（THP・sysctl・常駐サービス）は変更しない。
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

# 出力先: 英数字・. _ - ・/ のみ、'..' 禁止（A03）。
case "$out_dir" in
  *[!A-Za-z0-9._/-]*) echo "不正な --out（許容文字外）" >&2; exit 2 ;;
  *..*) echo "不正な --out（'..' 禁止）" >&2; exit 2 ;;
esac

sizes=(512 1024 2048 4096)
arms=(
  drop_pretouched
  keep_pretouched
  drop_untouched
  keep_untouched
  drop_reused
  drop_pinned
  keep_pinned
  drop_pretouched_split
  keep_pretouched_split
)
runs=5
test_prefix="fresh_reuse_window_diag_tests_3186::window_gap_3186"
fc_dir="$repo_root/scripts/bench/framework-compare"

# run ごとに腕の順序を巡回させる（k-1 個ずつ先頭をずらす）。
rotated_arms() {
  local k="$1" i
  local n=${#arms[@]}
  for ((i = 0; i < n; i++)); do
    echo "${arms[$(((i + k - 1) % n))]}"
  done
}

if [ "${FANDHE_AI_CUDA_READBACK_DEST:-}" != "" ]; then
  echo "FANDHE_AI_CUDA_READBACK_DEST が設定されている（既定計測の契約違反）" >&2
  exit 1
fi

if [ "$dry_run" = "1" ]; then
  echo "dry-run: host_kind=$host_kind out_dir=$out_dir"
  echo "dry-run: build: cargo test --release --locked -p fandhe-ai-backend-cuda --lib --no-run"
  echo "dry-run: build: (cd $fc_dir && cargo build --release -p bench-fandhe --config patch.crates-io.fandhe-ai.path=\"$repo_root/crates/facade\")"
  for ((k = 1; k <= runs; k++)); do
    echo "dry-run: run$k arm order: $(rotated_arms "$k" | tr '\n' ' ')"
    for n in "${sizes[@]}"; do
      while read -r arm; do
        echo "dry-run:   <bin> --ignored --exact ${test_prefix}_n${n}_${arm} --nocapture --test-threads=1 -> run$k/n${n}_${arm}.jsonl"
      done < <(rotated_arms "$k")
      for mode in fresh reuse; do
        echo "dry-run:   Layer A: bench-fandhe --task gemm --device cuda --size $n --mode $mode --out run$k/layerA_${mode}_n${n}.jsonl"
      done
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

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

cd "$repo_root"

# 計測ツリーの同定: HEAD と .rev-stamp の一致を要求する（A08）。
head_sha="$(git rev-parse HEAD)"
if [ -f "$repo_root/.rev-stamp" ]; then
  stamp="$(tr -d '[:space:]' <"$repo_root/.rev-stamp")"
  if [ "$stamp" != "$head_sha" ]; then
    echo ".rev-stamp ($stamp) と HEAD ($head_sha) が不一致のため中止する" >&2
    exit 1
  fi
else
  echo ".rev-stamp が無いため中止する（計測ツリーを同定できない）" >&2
  exit 1
fi

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

facade_path="$repo_root/crates/facade"
case "$facade_path" in
  *[\"\\]*) echo "facade パスに不正な文字" >&2; exit 1 ;;
esac
patch_config="patch.crates-io.fandhe-ai.path=\"${facade_path}\""
(cd "$fc_dir" && cargo build --release -p bench-fandhe --config "$patch_config")
bench_fandhe="${CARGO_TARGET_DIR:-$fc_dir/target}/release/bench-fandhe"
[ -x "$bench_fandhe" ] || { echo "bench-fandhe を特定できない" >&2; exit 1; }

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "git_head: $head_sha"
  echo "rev_stamp: $stamp"
  echo "host_kind: $host_kind"
  echo "os_release: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi: unavailable"
  nvcc --version 2>/dev/null | tail -1 || echo "nvcc: unavailable"
  echo "load_avg: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"
  echo "gpu_util_start: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "thp_enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unknown)"
  echo "thp_defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo unknown)"
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
        >"$out_dir/run$k/n${n}_${arm}.raw" 2>"$out_dir/run$k/n${n}_${arm}.err.raw" ||
        echo "run$k n${n} ${arm}: テスト失敗" >>"$out_dir/run$k/FAILED"
      grep -o 'DIAG_JSON .*' "$out_dir/run$k/n${n}_${arm}.raw" |
        sed -e 's/^DIAG_JSON //' >"$out_dir/run$k/n${n}_${arm}.jsonl" || true
      mask <"$out_dir/run$k/n${n}_${arm}.err.raw" >"$out_dir/run$k/n${n}_${arm}.err"
      rm -f "$out_dir/run$k/n${n}_${arm}.raw" "$out_dir/run$k/n${n}_${arm}.err.raw"
      [ -s "$out_dir/run$k/n${n}_${arm}.err" ] || rm -f "$out_dir/run$k/n${n}_${arm}.err"
    done < <(rotated_arms "$k")
    for mode in fresh reuse; do
      "$bench_fandhe" --task gemm --device cuda --size "$n" --mode "$mode" \
        --out "$out_dir/run$k/layerA_${mode}_n${n}.jsonl" >/dev/null 2>&1 ||
        echo "run$k LayerA ${mode} n${n}: 失敗" >>"$out_dir/run$k/FAILED"
    done
  done
done

{
  echo "計測終了時刻（UTC）: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "load_avg_end: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unknown)"
} >>"$out_dir/env_info.txt"
echo "完了: $out_dir（次: python3 $script_dir/aggregate.py $out_dir）"
