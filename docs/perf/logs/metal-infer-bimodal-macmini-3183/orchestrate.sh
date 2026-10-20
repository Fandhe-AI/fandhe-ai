#!/usr/bin/env bash
# イシュー #3183: Mac mini の infer Metal 二峰性診断の実機計測オーケストレータ。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）が正。
# `docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh` と同型で、
# テストバイナリ 2 本（facade 用: 腕 R/D/Don、backend-metal 用: 腕 G）を 1 回だけビルドし、
# run ごとに腕の起動順をローテーションして独立 5 回実行する。
# 使い方: orchestrate.sh macmini [--out <dir>]
# 集計は `python3 -I aggregate.py <dir>`。既存 run ファイルの上書きは拒否する。
# bash 3.2（macOS 標準）互換: 変数展開は ${VAR} 形式、連想配列は使わない。
set -euo pipefail

usage() { echo "usage: $0 macmini [--out <dir>]" >&2; exit 2; }

host_kind="${1:-}"
[ -n "${host_kind}" ] || usage
shift
case "${host_kind}" in macmini) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
out_dir="${script_dir}/${host_kind}"
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
mkdir -p "${out_dir}"
out_dir="$(cd "${out_dir}" && pwd)"

# 差し替え禁止: 既存の run ファイルがあれば中止する。
for n in 1 2 3 4 5; do
  if [ -e "${out_dir}/run${n}.jsonl" ]; then
    echo "既存の run${n}.jsonl があるため中止する（差し替え禁止）" >&2
    exit 1
  fi
done

mask() {
  sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"
}

# cargo の JSON メッセージからテストバイナリ（実行可能ファイル）の絶対パスを 1 つ取り出す。
# $1 = target.name、$2 = 種別 (test|lib)。
find_bin() {
  python3 -I -c '
import json, sys
name, kind = sys.argv[1], sys.argv[2]
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    t = m.get("target", {})
    if t.get("name") != name:
        continue
    if kind == "lib" and not (m.get("profile", {}).get("test") and "lib" in t.get("kind", [])):
        continue
    exe = m["executable"]
if exe is None:
    sys.exit(1)
print(exe)
' "$1" "$2"
}

cd "${repo_root}"
facade_bin="$(cargo test --release --locked -p fandhe-ai --test metal_infer_bimodal_diag \
  --no-run --message-format=json 2>/dev/null | find_bin metal_infer_bimodal_diag test)"
backend_bin="$(cargo test --release --locked -p fandhe-ai-backend-metal --lib \
  --no-run --message-format=json 2>/dev/null | find_bin fandhe_ai_backend_metal lib)"
[ -x "${facade_bin}" ] || { echo "facade テストバイナリを特定できない" >&2; exit 1; }
[ -x "${backend_bin}" ] || { echo "backend-metal テストバイナリを特定できない" >&2; exit 1; }

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu 2>/dev/null || true
  echo "rustc: $(rustc --version)"
  echo "cargo: $(cargo --version)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "host_kind: ${host_kind}"
} 2>&1 | mask >"${out_dir}/env_info.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

# 負荷ゲート（RULE.txt ゲート節。load1 < 3.0 を 60 秒間隔で最大 20 回待つ）
: >"${out_dir}/load_gate_status.txt"
: >"${out_dir}/load_gate.log"
gate_wait() {
  local n="$1" i=0 l
  while [ "${i}" -lt 20 ]; do
    l="$(load1)"
    if ! printf '%s' "${l}" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
      echo "run${n} load1=${l} GATE-INVALID" >>"${out_dir}/load_gate.log"
      echo "run${n} gate=GATE-INVALID" >>"${out_dir}/load_gate_status.txt"
      return 1
    fi
    echo "run${n} load1=${l} try=${i}" >>"${out_dir}/load_gate.log"
    if awk -v l="${l}" 'BEGIN { exit !(l < 3.0) }'; then
      echo "run${n} gate=pass load1=${l}" >>"${out_dir}/load_gate_status.txt"
      return 0
    fi
    i=$((i + 1))
    sleep 60
  done
  echo "run${n} gate=GATE-TIMEOUT" >>"${out_dir}/load_gate_status.txt"
  return 1
}

# 腕の起動順（RULE.txt: run ごとにローテーション）
order_for() {
  case "$1" in
    1 | 5) echo "R D Don G" ;;
    2) echo "D Don G R" ;;
    3) echo "Don G R D" ;;
    4) echo "G R D Don" ;;
  esac
}

run_arm() {
  local arm="$1" n="$2" raw
  raw="$(mktemp "${TMPDIR:-/tmp}/bimodal3183.XXXXXX")"
  case "${arm}" in
    R) "${facade_bin}" --ignored --exact metal_infer_bimodal_arm_r --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    D) "${facade_bin}" --ignored --exact metal_infer_bimodal_arm_d --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    Don) "${facade_bin}" --ignored --exact metal_infer_bimodal_arm_d_on --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
    G) "${backend_bin}" --ignored infer_bimodal_diag_per_iter_gpu_timestamps --nocapture --test-threads=1 >"${raw}" 2>>"${out_dir}/run${n}.err.raw" ;;
  esac
  grep -o 'DIAG_JSON {.*' "${raw}" | sed -e 's/^DIAG_JSON //' >>"${out_dir}/run${n}.jsonl"
  rm -f "${raw}"
}

for n in 1 2 3 4 5; do
  gate_wait "${n}" || { echo "負荷ゲート失敗（run${n}）。系列は無効" >&2; exit 3; }
  {
    echo "== run${n} before =="
    uptime
    pmset -g therm 2>&1 || true
    pmset -g 2>&1 | grep -i 'lowpowermode' || true
  } 2>&1 | mask >"${out_dir}/env_run${n}.txt"
  : >"${out_dir}/run${n}.err.raw"
  : >"${out_dir}/run${n}.jsonl"
  for arm in $(order_for "${n}"); do
    run_arm "${arm}" "${n}"
  done
  {
    echo "== run${n} after =="
    uptime
    pmset -g therm 2>&1 || true
  } 2>&1 | mask >>"${out_dir}/env_run${n}.txt"
  # powermetrics は sudo なしで通る場合のみ（sudo は使わない。RULE.txt）
  if sudo -n true 2>/dev/null; then
    sudo -n powermetrics --samplers gpu_power -n 1 2>&1 | mask >"${out_dir}/power_run${n}.txt" || echo "未取得" >"${out_dir}/power_run${n}.txt"
  else
    echo "未取得（sudo -n 不可）" >"${out_dir}/power_run${n}.txt"
  fi
  mask <"${out_dir}/run${n}.err.raw" >"${out_dir}/run${n}.err"
  rm -f "${out_dir}/run${n}.err.raw"
done
echo "完了: ${out_dir}（次: python3 -I ${script_dir}/aggregate.py ${out_dir}）"
