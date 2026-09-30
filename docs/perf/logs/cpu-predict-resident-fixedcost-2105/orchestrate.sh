#!/usr/bin/env bash
# イシュー #2105: CPU predict_resident 固定費の実機計測オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。
# テストバイナリを 1 回ビルドし、独立 5 プロセスで
# cpu_predict_resident_fixedcost_phases を実行して run{1..5}.jsonl を残す。
# 使い方: orchestrate.sh <m4max|gb10> [--out <dir>]
# 集計は `python3 aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
set -euo pipefail

usage() { echo "usage: $0 <m4max|gb10> [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "$host_kind" ] || usage
shift
case "$host_kind" in m4max | gb10) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
out_dir="$script_dir/$host_kind"
while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      [ $# -ge 2 ] || usage
      out_dir="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
# cd 後も同じ場所を指すよう絶対パス化する（相対 --out はリポジトリルート基準にずれるため）。
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for n in 1 2 3 4 5; do
  if [ -e "$out_dir/run$n.jsonl" ]; then
    echo "既存の run$n.jsonl があるため中止する（差し替え禁止）" >&2
    exit 1
  fi
done

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

cd "$repo_root"
bin="$(cargo test --release --locked -p fandhe-ai --test cpu_predict_resident_fixedcost_diag \
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
            and m.get("target", {}).get("name") == "cpu_predict_resident_fixedcost_diag":
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
  echo "git_head: $(git rev-parse HEAD)"
  echo "RAYON_NUM_THREADS: ${RAYON_NUM_THREADS:-unset}"
  echo "host_kind: $host_kind"
} 2>&1 | mask >"$out_dir/env_info.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

# 集計器へ渡す負荷ゲート状態（aggregate.py が読み、未通過 run があれば参考扱いを出力へ反映する）
: >"$out_dir/load_gate_status.txt"
for n in 1 2 3 4 5; do
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
  "$bin" --ignored --exact cpu_predict_resident_fixedcost_phases --nocapture --test-threads=1 \
    >"$out_dir/run$n.raw" 2>"$out_dir/run$n.err.raw"
  grep -o 'DIAG_JSON .*' "$out_dir/run$n.raw" | sed -e 's/^DIAG_JSON //' >"$out_dir/run$n.jsonl"
  mask <"$out_dir/run$n.err.raw" >"$out_dir/run$n.err"
  rm -f "$out_dir/run$n.raw" "$out_dir/run$n.err.raw"
done
echo "完了: $out_dir（次: python3 $script_dir/aggregate.py $out_dir）"
