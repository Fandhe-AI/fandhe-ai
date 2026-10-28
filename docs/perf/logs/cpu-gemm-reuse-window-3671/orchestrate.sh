#!/usr/bin/env bash
# イシュー #3671: N=256〜2048 の bench reuse 窓と診断の差の切り分け（診断側の腕）オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt ＋ RULE.v2.txt（実測前に固定）が正。構成は
# docs/perf/logs/cpu-gemm-resident-reuse-3646/orchestrate.sh と同型（ビルド 1 回・独立 5 プロセス・
# .rev-stamp 必須・既存 run の上書き禁止・load ゲート・ホスト名／$HOME／ユーザー名のマスク）。
# 使い方: orchestrate.sh <m4max|gb10> [--only-arm resident_public] [--aux minflt] [--out <dir>]
#   既定: 診断の正式系列（series_kind=main。cpu_gemm_reuse_window_phases を 5 run）
#   --only-arm: rp_solo 系列（series_kind=solo。cpu_gemm_reuse_window_solo を 5 run）。値は allowlist のみ
#   --aux minflt: 判定外の補助 1 run（/usr/bin/time -v 配下。series_kind=minflt）
# bench 側の腕（bench_base／bench_tunables／bench_headpatch）は session-scripts/session-3671-excl.sh が回す。
# 集計は `python3 -I aggregate.py <dir>`。計測ツリーのルートに .rev-stamp（1 行目 HEAD・2 行目 dirty=<件数>）が必須。
set -euo pipefail

usage() { echo "usage: $0 <m4max|gb10> [--only-arm resident_public] [--aux minflt] [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "$host_kind" ] || usage
shift
case "$host_kind" in m4max | gb10) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../.." && pwd)"
kind="main"
only_arm=""
out_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only-arm)
      [ $# -ge 2 ] || usage
      # allowlist（RULE.txt の rp_solo。許可外は拒否）。テスト側の SOLO_ALLOWED と同じ集合。
      case "$2" in resident_public) only_arm="$2" ;; *) usage ;; esac
      kind="solo"
      shift 2
      ;;
    --aux)
      [ $# -ge 2 ] || usage
      case "$2" in minflt) kind="minflt" ;; *) usage ;; esac
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
  if [ "$kind" = "main" ]; then out_dir="$script_dir/$host_kind"; else out_dir="$script_dir/$host_kind-$kind"; fi
fi

# RAYON_NUM_THREADS スイープは対象外（RULE.txt）。設定されていれば拒否する。
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
  echo "RAYON_NUM_THREADS が設定されている（対象外のため拒否）" >&2
  exit 1
fi
if [ ! -f "$repo_root/.rev-stamp" ]; then
  echo ".rev-stamp が無い（計測ツリーの同定に必須）" >&2
  exit 1
fi
if [ "$kind" = "minflt" ] && ! command -v /usr/bin/time >/dev/null 2>&1; then
  echo "/usr/bin/time が無い（minflt 補助系列は未実測とする）" >&2
  exit 1
fi

if [ "$kind" = "minflt" ]; then run_count=1; else run_count=5; fi
if [ "$kind" = "solo" ]; then test_name="cpu_gemm_reuse_window_solo"; else test_name="cpu_gemm_reuse_window_phases"; fi

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
bin="$(cargo test --release --locked -p fandhe-ai --test cpu_gemm_reuse_window_diag \
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
            and m.get("target", {}).get("name") == "cpu_gemm_reuse_window_diag":
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
  echo "rustc -vV: $(rustc -vV | tr '\n' ';')"
  echo "cargo: $(cargo --version)"
  echo "pagesize: $(getconf PAGESIZE 2>/dev/null || echo unknown)"
  echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
  echo "glibc: $( (ldd --version 2>/dev/null || true) | head -n 1)"
  echo "rev_stamp_head: $(sed -n 1p "$repo_root/.rev-stamp")"
  echo "rev_stamp_dirty: $(sed -n 2p "$repo_root/.rev-stamp")"
  echo "rule_sha256: $(sha256_of "$script_dir/RULE.txt")"
  echo "rule_v2_sha256: $(sha256_of "$script_dir/RULE.v2.txt")"
  echo "RAYON_NUM_THREADS: unset"
  echo "host_kind: $host_kind"
  echo "series_kind: $kind"
  [ -z "$only_arm" ] || echo "GEMM_WINDOW_ONLY_ARM: $only_arm"
  # 同定の記録（RULE.txt: v0.12.0 から HEAD までの本体差分。タグが無ければ未実測を明記する）
  if git rev-parse --verify --quiet v0.12.0 >/dev/null 2>&1; then
    echo "diff_stat_v0.12.0_head:"
    git diff --stat v0.12.0 HEAD -- crates/tensor-core crates/autodiff crates/backend-cpu crates/facade/src || true
  else
    echo "diff_stat_v0.12.0_head: 未実測（v0.12.0 タグが計測ツリーに無い）"
  fi
} 2>&1 | mask >"$out_dir/env_info.txt"
echo "$kind" >"$out_dir/series_kind.txt"

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
  case "$kind" in
    main) runner=("$bin") ;;
    solo) runner=(env "GEMM_WINDOW_ONLY_ARM=$only_arm" "$bin") ;;
    minflt) runner=(/usr/bin/time -v -o "$out_dir/run$n.time.raw" "$bin") ;;
  esac
  "${runner[@]}" --ignored --exact "$test_name" \
    --nocapture --test-threads=1 >"$out_dir/run$n.raw" 2>"$out_dir/run$n.err.raw"
  grep -o 'DIAG_JSON .*' "$out_dir/run$n.raw" | sed -e 's/^DIAG_JSON //' >"$out_dir/run$n.jsonl"
  mask <"$out_dir/run$n.err.raw" >"$out_dir/run$n.err"
  if [ "$kind" = "minflt" ]; then
    grep -E 'Minor|context switches|Maximum resident' "$out_dir/run$n.time.raw" | mask >"$out_dir/run$n.time"
    rm -f "$out_dir/run$n.time.raw"
  fi
  rm -f "$out_dir/run$n.raw" "$out_dir/run$n.err.raw"
done
echo "完了: $out_dir（次: python3 -I $script_dir/aggregate.py $out_dir）"
