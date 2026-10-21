#!/usr/bin/env bash
# イシュー #3188（再オープン分）: H4（device_update の prologue + 残差）分解診断の GB10 オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。2106 の orchestrate.sh と同型。
# テストバイナリ cpu_reuse_device_update_h4_diag を 1 回だけビルドし（計測中はビルドしない）、
# multi（RAYON_NUM_THREADS 未設定）と single（RAYON_NUM_THREADS=1）の 2 系列を
# run ごとに交互に独立プロセスで起動して <out>/h4-multi/run{1..5}.jsonl・h4-single/run{1..5}.jsonl を残す。
# 使い方: orchestrate_h4.sh --out <dir>    （集計は aggregate_h4.py <dir>/h4-multi <dir>/h4-single）
# 既存 run ファイルの上書きは拒否する。実行は GB10 上のみを想定（負荷ゲート load1 < 1.0 を適用）。
set -euo pipefail

usage() { echo "usage: $0 --out <dir>" >&2; exit 2; }
[ "${1:-}" = "--out" ] && [ $# -eq 2 ] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../../.." && pwd)"
mkdir -p "$2"
out_base="$(cd "$2" && pwd)"
series="multi single"

for s in $series; do
  mkdir -p "$out_base/h4-$s"
  for n in 1 2 3 4 5; do
    if [ -e "$out_base/h4-$s/run$n.jsonl" ]; then
      echo "既存の h4-$s/run$n.jsonl があるため中止する（差し替え禁止）" >&2
      exit 1
    fi
  done
done

# 未マスクの生ログ（ホスト名・ホームパスを含み得る）は mktemp -d の一時領域へ置き、trap で必ず削除する。
raw_dir="$(mktemp -d)"
trap 'rm -rf "$raw_dir"' EXIT

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

cd "$repo_root"
bin="$(cargo test --release --locked -p fandhe-ai --test cpu_reuse_device_update_h4_diag \
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
            and m.get("target", {}).get("name") == "cpu_reuse_device_update_h4_diag":
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
')"
[ -x "$bin" ] || { echo "テストバイナリを特定できない" >&2; exit 1; }

for s in $series; do
  rt="unset"
  [ "$s" = single ] && rt="1"
  {
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname: $(uname -srm)"
    if command -v lscpu >/dev/null 2>&1; then
      lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Architecture' || true
      echo "lscpu_e (CPU, CORE, MAXMHZ):"
      lscpu -e=CPU,CORE,SOCKET,MAXMHZ 2>/dev/null || true
    fi
    echo "rustc: $(rustc --version)"
    echo "cargo: $(cargo --version)"
    echo "git_head: $(git rev-parse HEAD 2>/dev/null || true)"
    echo "series: $s"
    echo "RAYON_NUM_THREADS: $rt"
  } 2>&1 | mask >"$out_base/h4-$s/env_info.txt"
  : >"$out_base/h4-$s/load_gate_status.txt"
done

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

# 負荷ゲート（load1 < 1.0・30 秒間隔・最大 30 分）。結果は pass / unpassed を標準出力へ返す。
gate() { # gate <label> <logfile>
  local label="$1" log="$2" waited=0 l
  while :; do
    l="$(load1)"
    echo "$label load1=$l waited=${waited}s" >>"$log"
    # 空・非数値は通過扱いにしない。
    if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v l="$l" 'BEGIN{exit !(l < 1.0)}'; then
      echo pass
      return
    fi
    if [ "$waited" -ge 1800 ]; then
      echo "$label: 負荷ゲート未通過（参考扱い）" >>"$log"
      echo unpassed
      return
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

for n in 1 2 3 4 5; do
  for s in $series; do
    d="$out_base/h4-$s"
    status="$(gate "run$n" "$d/load_gate.log")"
    echo "run$n gate=$status" >>"$d/load_gate_status.txt"
    if [ "$s" = single ]; then
      env RAYON_NUM_THREADS=1 "$bin" --ignored --exact cpu_reuse_device_update_h4_phases \
        --nocapture --test-threads=1 >"$raw_dir/$s.$n.raw" 2>"$raw_dir/$s.$n.err.raw"
    else
      env -u RAYON_NUM_THREADS "$bin" --ignored --exact cpu_reuse_device_update_h4_phases \
        --nocapture --test-threads=1 >"$raw_dir/$s.$n.raw" 2>"$raw_dir/$s.$n.err.raw"
    fi
    grep -o 'DIAG_JSON .*' "$raw_dir/$s.$n.raw" | sed -e 's/^DIAG_JSON //' >"$d/run$n.jsonl"
    mask <"$raw_dir/$s.$n.err.raw" >"$d/run$n.err"
    rm -f "$raw_dir/$s.$n.raw" "$raw_dir/$s.$n.err.raw"
  done
done
echo "完了: $out_base（次: python3 -I $script_dir/aggregate_h4.py $out_base/h4-multi $out_base/h4-single）"
