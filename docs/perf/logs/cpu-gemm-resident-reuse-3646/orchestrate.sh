#!/usr/bin/env bash
# イシュー #3646: CPU GEMM 常駐入力経路の reuse>fresh 逆転の帰属計測オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。構成は
# docs/perf/logs/cpu-gemm-reuse-fresh-3187/orchestrate.sh と同型。
# テストバイナリを 1 回ビルドし、独立 5 プロセスで cpu_gemm_resident_reuse_phases を
# 実行して run{1..5}.jsonl を残す（--aux minflt のみ run1 の 1 run）。
# 使い方: orchestrate.sh <m4max|gb10> [--aux <minflt|tunables|taskset>] [--out <dir>]
#   --aux 無し: 正式系列（series_kind=main）。--aux は判定外の補助系列で、値は RULE.txt の
#   固定定数だけを使う（外部入力から組み立てない）。
# 集計は `python3 aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
# 計測ツリーのルートに .rev-stamp（1 行目 HEAD・2 行目 dirty=<件数>）が必須。
set -euo pipefail

usage() { echo "usage: $0 <m4max|gb10> [--aux <minflt|tunables|taskset>] [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "$host_kind" ] || usage
shift
case "$host_kind" in m4max | gb10) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
aux="main"
out_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --aux)
      [ $# -ge 2 ] || usage
      case "$2" in minflt | tunables | taskset) aux="$2" ;; *) usage ;; esac
      shift 2
      ;;
    --out)
      [ $# -ge 2 ] || usage
      out_dir="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
if [ -z "$out_dir" ]; then
  if [ "$aux" = "main" ]; then out_dir="$script_dir/$host_kind"; else out_dir="$script_dir/$host_kind-$aux"; fi
fi

# RULE.txt 固定定数（補助系列）。変更するときは RULE.v2 を別コミットで作る。
GLIBC_TUNABLES_VALUE="glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824"
TASKSET_CPUS="5-9,15-19"

# RAYON_NUM_THREADS スイープは対象外（RULE.txt）。設定されていれば拒否する。
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
  echo "RAYON_NUM_THREADS が設定されている（対象外のため拒否）" >&2
  exit 1
fi
if [ ! -f "$repo_root/.rev-stamp" ]; then
  echo ".rev-stamp が無い（計測ツリーの同定に必須）" >&2
  exit 1
fi

if [ "$aux" = "minflt" ]; then run_count=1; else run_count=5; fi

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for n in $(seq 1 "$run_count"); do
  if [ -e "$out_dir/run$n.jsonl" ]; then
    echo "既存の run$n.jsonl があるため中止する（差し替え禁止）" >&2
    exit 1
  fi
done

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g" -e "s|$(id -un)|<user>|g"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

cd "$repo_root"
bin="$(cargo test --release --locked -p fandhe-ai --test cpu_gemm_resident_reuse_diag \
  --no-run --message-format=json 2>/dev/null |
  python3 -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") \
            and m.get("target", {}).get("name") == "cpu_gemm_resident_reuse_diag":
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
')"
[ -x "$bin" ] || { echo "テストバイナリを特定できない" >&2; exit 1; }

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Architecture' || true
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu 2>/dev/null || true
  fi
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "pagesize: $(getconf PAGESIZE 2>/dev/null || echo unknown)"
  echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
  echo "glibc: $( (ldd --version 2>/dev/null || true) | head -n 1)"
  echo "rev_stamp_head: $(sed -n 1p "$repo_root/.rev-stamp")"
  echo "rev_stamp_dirty: $(sed -n 2p "$repo_root/.rev-stamp")"
  echo "rule_sha256: $(sha256_of "$script_dir/RULE.txt")"
  echo "RAYON_NUM_THREADS: unset"
  echo "host_kind: $host_kind"
  echo "series_kind: $aux"
  [ "$aux" != "tunables" ] || echo "GLIBC_TUNABLES: $GLIBC_TUNABLES_VALUE"
  [ "$aux" != "taskset" ] || echo "taskset_cpus: $TASKSET_CPUS"
} 2>&1 | mask >"$out_dir/env_info.txt"
echo "$aux" >"$out_dir/series_kind.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

: >"$out_dir/load_gate_status.txt"
for n in $(seq 1 "$run_count"); do
  if [ "$host_kind" = "gb10" ]; then
    gate_status=pass
    waited=0
    while :; do
      l="$(load1)"
      echo "run$n load1=$l waited=${waited}s" >>"$out_dir/load_gate.log"
      # 空・非数値は通過扱いにしない（awk では 0 とみなされ l<1.0 が成立してしまうため）。
      if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v l="$l" 'BEGIN{exit !(l < 1.0)}'; then break; fi
      if [ "$waited" -ge 1800 ]; then
        echo "run$n: 負荷ゲート未通過（参考扱い）" >>"$out_dir/load_gate.log"
        gate_status=unpassed
        break
      fi
      sleep 30
      waited=$((waited + 30))
    done
  else
    gate_status=record_only
    echo "run$n load1=$(load1) (record_only)" >>"$out_dir/load_gate.log"
  fi
  echo "run$n gate=$gate_status" >>"$out_dir/load_gate_status.txt"
  case "$aux" in
    main) runner=("$bin") ; minflt_env=0 ;;
    minflt) runner=("$bin") ; minflt_env=1 ;;
    tunables) runner=(env "GLIBC_TUNABLES=$GLIBC_TUNABLES_VALUE" "$bin") ; minflt_env=0 ;;
    taskset) runner=(taskset -c "$TASKSET_CPUS" "$bin") ; minflt_env=0 ;;
  esac
  GEMM_REUSE_MINFLT="$minflt_env" "${runner[@]}" --ignored --exact cpu_gemm_resident_reuse_phases \
    --nocapture --test-threads=1 >"$out_dir/run$n.raw" 2>"$out_dir/run$n.err.raw"
  grep -o 'DIAG_JSON .*' "$out_dir/run$n.raw" | sed -e 's/^DIAG_JSON //' >"$out_dir/run$n.jsonl"
  mask <"$out_dir/run$n.err.raw" >"$out_dir/run$n.err"
  rm -f "$out_dir/run$n.raw" "$out_dir/run$n.err.raw"
done
echo "完了: $out_dir（次: python3 $script_dir/aggregate.py $out_dir）"
