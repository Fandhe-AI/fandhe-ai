#!/usr/bin/env bash
# イシュー #3701: S1 内側分解診断の before／after 診断のみの 1 系列オーケストレータ（1 機体・1 系列分）。
# 判定規則は同ディレクトリの RULE.txt（実測前に単独コミットで固定）が正。構成は
# #3693 の orchestrate.sh と同型で、bench は測らず診断（5 診断腕。内側の区間を含む）だけを
# 5 run 回す。腕は build_split_arms.sh が作った事前ビルド済みの diag-before／diag-after（計測中にビルドしない）。
# run ごとに腕の起動順を反転する（ABBA）。--first は run1 の先頭腕で、系列間で交互にする（RULE.txt）。
# 出力: <out>/<arm>/run<r>.jsonl（診断 DIAG_JSON 行）・<out>/{env_info.txt,series_kind.txt,load_gate.log,
#       load_gate_status.txt,first_arm.txt}。集計は split_inner_judge.py（呼び出し元は session-scripts/session-3701.sh）。
# 使い方: orchestrate.sh <gb10|macmini> --bindir <build_split_arms.sh の出力> --out <新規ディレクトリ> --first <before|after>
# 既存の出力があれば中止する（差し替え禁止）。診断の反復数は既定のまま（縮めない）。
# bash 3.2（macOS 標準）で動くよう ${VAR} 形式・mapfile／連想配列なしで書く。
set -euo pipefail

usage() { echo "usage: $0 <gb10|macmini> --bindir <dir> --out <dir> --first <before|after>" >&2; exit 2; }

host_kind="${1:-}"
[ -n "${host_kind}" ] || usage
shift
case "${host_kind}" in gb10 | macmini) ;; *) usage ;; esac
bindir=""
out_dir=""
first_arm=""
while [ $# -gt 0 ]; do
  case "$1" in
    --bindir) [ $# -ge 2 ] || usage; bindir="$2"; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; out_dir="$2"; shift 2 ;;
    --first) [ $# -ge 2 ] || usage; first_arm="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "${bindir}" ] && [ -n "${out_dir}" ] && [ -n "${first_arm}" ] || usage
case "${first_arm}" in before | after) ;; *) usage ;; esac
for p in "${bindir}" "${out_dir}"; do
  case "${p}" in /*) ;; *) echo "絶対パスのみ許可: ${p}" >&2; exit 2 ;; esac
  case "${p}" in *[!A-Za-z0-9._/-]*) echo "許可外の文字を含むパス: ${p}" >&2; exit 2 ;; esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="1 2 3 4 5"

# RAYON_NUM_THREADS スイープは対象外。設定されていれば拒否する。
if [ -n "${RAYON_NUM_THREADS:-}" ]; then
  echo "RAYON_NUM_THREADS が設定されている（対象外のため拒否）" >&2
  exit 1
fi
unset GEMM_REUSE_MINFLT GLIBC_TUNABLES CARGO_TARGET_DIR

if [ "${host_kind}" = "gb10" ]; then
  load_max="1.0"
  node_placeholder="<cuda-node>"
else
  load_max="3.0"
  node_placeholder="<metal-node>"
fi

[ -f "${bindir}/arms.txt" ] || { echo "arms.txt が無い: ${bindir}" >&2; exit 1; }
for b in diag-before diag-after; do
  [ -x "${bindir}/${b}" ] || { echo "バイナリが無い: ${b}" >&2; exit 1; }
done

if [ -e "${out_dir}" ]; then
  echo "出力ディレクトリが既に存在する（差し替え禁止）: ${out_dir}" >&2
  exit 5
fi
mkdir -p "${out_dir}/before" "${out_dir}/after"
out_dir="$(cd "${out_dir}" && pwd)"

# 公開物に実ホスト名・HOME・ユーザー名を残さない（回収後 grep 0 件の前提）。
mask() {
  sed -e "s|local\\.fandhe\\.[A-Za-z0-9-]*|${node_placeholder}|g" \
    -e "s|${HOME}|<home>|g" -e "s|$(hostname)|${node_placeholder}|g" -e "s|$(id -un)|<user>|g"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# 起動前にバイナリの sha256 が arms.txt と一致することを確かめる（腕の取り違え防止）。
verify_bin() { # $1=バイナリ名
  local want have
  want="$(grep "^$1_sha256: " "${bindir}/arms.txt" | awk '{print $2}')"
  have="$(sha256_of "${bindir}/$1")"
  [ -n "${want}" ] && [ "${want}" = "${have}" ] || { echo "sha256 不一致: $1" >&2; exit 1; }
}

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
  echo "rule_sha256: $(sha256_of "${script_dir}/RULE.txt")"
  echo "RAYON_NUM_THREADS: unset"
  echo "host_kind: ${host_kind}"
  echo "series_kind: main"
  echo "first_arm: ${first_arm}"
  echo "load_max: ${load_max}"
  cat "${bindir}/arms.txt"
} 2>&1 | mask >"${out_dir}/env_info.txt"
echo main >"${out_dir}/series_kind.txt"
echo "${first_arm}" >"${out_dir}/first_arm.txt"

load1() {
  uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'
}

: >"${out_dir}/load_gate_status.txt"
: >"${out_dir}/load_gate.log"

run_arm() { # $1=before|after $2=run 番号
  local arm="$1" r="$2"
  verify_bin "diag-${arm}"
  "${bindir}/diag-${arm}" --ignored --exact cpu_gemm_s1_inner_split_phases \
    --nocapture --test-threads=1 >"${out_dir}/${arm}/run${r}.raw" 2>"${out_dir}/${arm}/run${r}.err.raw"
  grep -o 'DIAG_JSON .*' "${out_dir}/${arm}/run${r}.raw" | sed -e 's/^DIAG_JSON //' >"${out_dir}/${arm}/run${r}.jsonl"
  mask <"${out_dir}/${arm}/run${r}.err.raw" >"${out_dir}/${arm}/run${r}.err"
  rm -f "${out_dir}/${arm}/run${r}.raw" "${out_dir}/${arm}/run${r}.err.raw"
}

if [ "${first_arm}" = "before" ]; then other_arm=after; else other_arm=before; fi

for r in ${RUNS}; do
  gate_status=pass
  waited=0
  while :; do
    l="$(load1)"
    echo "run${r} load1=${l} waited=${waited}s" >>"${out_dir}/load_gate.log"
    # 空・非数値は通過扱いにしない（awk では 0 とみなされ閾値未満が成立してしまうため）。
    if [[ "${l}" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v l="${l}" -v m="${load_max}" 'BEGIN{exit !(l < m)}'; then break; fi
    if [ "${waited}" -ge 1800 ]; then
      echo "run${r}: 負荷ゲート未通過（record_only）" >>"${out_dir}/load_gate.log"
      gate_status=unpassed
      break
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "run${r} gate=${gate_status}" >>"${out_dir}/load_gate_status.txt"
  # 奇数 run は --first の腕が先、偶数 run は反転（ABBA）。
  if [ $((r % 2)) = 1 ]; then a="${first_arm}"; b="${other_arm}"; else a="${other_arm}"; b="${first_arm}"; fi
  echo "run${r} order=${a},${b}" >>"${out_dir}/load_gate.log"
  run_arm "${a}" "${r}"
  run_arm "${b}" "${r}"
done
echo "完了: ${out_dir}（次: python3 -I ${script_dir}/split_inner_judge.py --host ${host_kind}=<ラベル順の系列ディレクトリ>）"
