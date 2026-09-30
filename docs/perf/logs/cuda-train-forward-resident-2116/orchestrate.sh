#!/usr/bin/env bash
# イシュー #2116: CUDA train forward_resident 内訳・param_readout 診断の実機計測オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。
# 上位（facade）・下位（backend-cuda lib）の 2 テストバイナリを 1 回ずつビルドし、
# 独立 5 回ずつ実行して run{1..5}.facade.jsonl / run{1..5}.backend.jsonl を残す。
# 使い方: orchestrate.sh <gb10|rtx3060> [--out <dir>]
# 集計は `python3 aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
# 構成は docs/perf/logs/cpu-reuse-device-update-2106/orchestrate.sh と同型。
set -euo pipefail

usage() { echo "usage: $0 <gb10|rtx3060> [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "$host_kind" ] || usage
shift
case "$host_kind" in gb10 | rtx3060) ;; *) usage ;; esac

# CPU 診断モードの混入拒否: TRAIN_FWD_DIAG_KIND が cuda 以外だと上位診断だけ CPU で走り
# 下位（CUDA）と混合した系列になる。未設定または cuda のみ許可する（出力の backend 識別子は
# aggregate.py も照合する二重ガード）。
case "${TRAIN_FWD_DIAG_KIND:-cuda}" in
  cuda) ;;
  *)
    echo "TRAIN_FWD_DIAG_KIND=${TRAIN_FWD_DIAG_KIND} は不可（CUDA 実測のみ。unset して再実行）" >&2
    exit 2
    ;;
esac
export TRAIN_FWD_DIAG_KIND=cuda

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
# cd 後も同じ場所を指すよう絶対パス化する。
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for n in 1 2 3 4 5; do
  for part in facade backend; do
    if [ -e "$out_dir/run$n.$part.jsonl" ]; then
      echo "既存の run$n.$part.jsonl があるため中止する（差し替え禁止）" >&2
      exit 1
    fi
  done
done

# 未マスクの生ログは mktemp -d の一時領域へ置き、trap で必ず削除する。
raw_dir="$(mktemp -d)"
trap 'rm -rf "$raw_dir"' EXIT

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

# テストバイナリのパスを cargo の JSON 出力から特定する（第 1 引数: facade|backend、残りは cargo test の引数）。
find_bin() {
  local kind="$1"
  shift
  cargo test --release --locked "$@" --no-run --message-format=json 2>/dev/null |
    python3 -c '
import json, sys
kind = sys.argv[1]
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    t = m.get("target", {})
    if kind == "facade" and t.get("name") == "cuda_train_forward_resident_diag":
        exe = m["executable"]
    if kind == "backend" and t.get("name") in ("fandhe-ai-backend-cuda", "fandhe_ai_backend_cuda") \
            and m.get("profile", {}).get("test") and "lib" in t.get("kind", []):
        exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
' "$kind"
}

cd "$repo_root"
facade_bin="$(find_bin facade -p fandhe-ai --test cuda_train_forward_resident_diag)"
backend_bin="$(find_bin backend -p fandhe-ai-backend-cuda --lib)"
[ -x "$facade_bin" ] || { echo "facade テストバイナリを特定できない" >&2; exit 1; }
[ -x "$backend_bin" ] || { echo "backend テストバイナリを特定できない" >&2; exit 1; }

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Architecture' || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # GPU 名・driver・compute capability・CUDA・プロセス数に絞る（それ以外は採取しない）。
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true
    nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true
    echo "compute_apps_count: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  else
    echo "nvidia-smi: unavailable"
  fi
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "host_kind: $host_kind"
} 2>&1 | mask >"$out_dir/env_info.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
}

# 集計器へ渡す負荷ゲート状態（aggregate.py が読み、未通過 run があれば参考扱いを出力へ反映する）
: >"$out_dir/load_gate_status.txt"
for n in 1 2 3 4 5; do
  if [ "$host_kind" = "gb10" ]; then
    gate_status=pass
    waited=0
    while :; do
      l="$(load1)"
      u="$(gpu_util || true)"
      echo "run$n load1=$l gpu_util=${u:-na} waited=${waited}s" >>"$out_dir/load_gate.log"
      # 空・非数値は通過扱いにしない（awk では 0 とみなされ l<1.0 が成立してしまうため）。
      if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "$u" =~ ^[0-9]+$ ]] &&
        awk -v l="$l" 'BEGIN{exit !(l < 1.0)}' && [ "$u" -eq 0 ]; then
        break
      fi
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
    echo "run$n load1=$(load1) gpu_util=$(gpu_util || echo na) (record_only)" >>"$out_dir/load_gate.log"
  fi
  echo "run$n gate=$gate_status" >>"$out_dir/load_gate_status.txt"

  "$facade_bin" --ignored --exact cuda_train_forward_resident_phases --nocapture --test-threads=1 \
    >"$raw_dir/run$n.facade.raw" 2>"$raw_dir/run$n.facade.err.raw"
  grep -o 'DIAG_JSON .*' "$raw_dir/run$n.facade.raw" | sed -e 's/^DIAG_JSON //' >"$out_dir/run$n.facade.jsonl"
  mask <"$raw_dir/run$n.facade.err.raw" >"$out_dir/run$n.facade.err"

  "$backend_bin" --ignored --exact train_forward_resident_diag_tests::cuda_train_forward_resident_backend_phases \
    --nocapture --test-threads=1 \
    >"$raw_dir/run$n.backend.raw" 2>"$raw_dir/run$n.backend.err.raw"
  grep -o 'DIAG_JSON .*' "$raw_dir/run$n.backend.raw" | sed -e 's/^DIAG_JSON //' >"$out_dir/run$n.backend.jsonl"
  mask <"$raw_dir/run$n.backend.err.raw" >"$out_dir/run$n.backend.err"

  rm -f "$raw_dir/run$n.facade.raw" "$raw_dir/run$n.facade.err.raw" "$raw_dir/run$n.backend.raw" "$raw_dir/run$n.backend.err.raw"
done
echo "完了: $out_dir（次: python3 $script_dir/aggregate.py $out_dir）"
