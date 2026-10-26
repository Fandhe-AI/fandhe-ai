#!/usr/bin/env bash
# イシュー #3612: Mac mini の Metal 常駐チェーン（train・infer）の区間別内訳ランナー。
# 手順の正本: docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt（計測前に固定。sha256 を env_info へ記録する）。
# 腕 B0／B1（backend-metal の #[ignore] 診断テスト resident_chain_breakdown_diag_tests_3612）を 5 round、
# round ごとに独立プロセスで起動する。A0（全体窓）は #3257 の bench-resident-chain.patch を当てた bench-fandhe の
# 別ランで取り、summarize_breakdown_resident_chain_metal.py の --a0-*-us で渡す（本ランナーの範囲外）。
# 書式: 兄弟スクリプトに合わせ `set -u`（bash 3.2 互換・mapfile／連想配列／eval 不使用）。
# 呼び出し例: bash run_breakdown_resident_chain_metal.sh 3612-macmini
set -u
cd "$(dirname "$0")" || exit 1
# shellcheck source=./ab_round_cooldown.sh
source ./ab_round_cooldown.sh

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
LOAD_GATE=3.0
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 <label>  (label: [A-Za-z0-9._-]+)" >&2
  exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2
  exit 1
fi
if [[ -z "${AB_ROUND_COOLDOWN_MAX_LOAD1:-}" ]]; then
  AB_ROUND_COOLDOWN_MAX_LOAD1=3.0
elif [[ "$AB_ROUND_COOLDOWN_MAX_LOAD1" != "3.0" ]]; then
  echo "error: AB_ROUND_COOLDOWN_MAX_LOAD1 は 3.0 固定（RULE.txt。got: ${AB_ROUND_COOLDOWN_MAX_LOAD1}）" >&2
  exit 1
fi
export AB_ROUND_COOLDOWN_MAX_LOAD1

# 機体同定（fail-closed）。Mac mini は M4 完全一致のみ（M4 Max 開発機の値を混ぜない）。
brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
if [[ "$brand" != "Apple M4" ]]; then
  echo "error: 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
RUNNER_ROOT="$(cd ../../.. && pwd)"
RULE_FILE="$RUNNER_ROOT/docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt"
[[ -f "$RULE_FILE" ]] || { echo "error: RULE.txt が無い" >&2; exit 1; }
RULE_SHA="$(sha256_of "$RULE_FILE")"

# .rev-stamp（1 行目 HEAD・2 行目 dirty=0）の検証。計測ツリーは HEAD と完全一致であること。
stamp="$(head -n1 "$RUNNER_ROOT/.rev-stamp" 2>/dev/null || true)"
dirty="$(sed -n 2p "$RUNNER_ROOT/.rev-stamp" 2>/dev/null || true)"
real="$(git -C "$RUNNER_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$stamp" || "$stamp" != "$real" || "$dirty" != "dirty=0" ]]; then
  echo "error: .rev-stamp が無い・HEAD と不一致・dirty=0 でない" >&2; exit 1
fi
if ! git -C "$RUNNER_ROOT" diff --quiet HEAD --; then
  echo "error: 追跡ファイルに未コミットの差分がある" >&2; exit 1
fi

OUT="$PWD/results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-breakdown-3612-${LABEL}.log"
ENV_INFO="$OUT/env_info-3612-${LABEL}.txt"
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
ANY_FAILED=0

# ビルドは計測の前に済ませる（計測中のビルド並走禁止）。テストバイナリのパスは JSON から抽出する。
BUILD_JSON="$(cd "$RUNNER_ROOT" && cargo test --release -p fandhe-ai-backend-metal --lib --no-run --message-format=json 2>"$OUT/build-3612-${LABEL}.err")" || {
  tail -40 "$OUT/build-3612-${LABEL}.err"; echo "BUILD FAILED" >>"$SKIP"; exit 1; }
BIN="$(printf '%s\n' "$BUILD_JSON" | python3 -I -c '
import json, sys
exe = None
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m.get("target", {}).get("name") == "fandhe_ai_backend_metal" and m.get("profile", {}).get("test"):
        exe = m["executable"]
print(exe or "")
')"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "error: テストバイナリを特定できない" >&2; echo "no test binary" >>"$SKIP"; exit 1
fi

{
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "cpu_brand=${brand}"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "rule_sha256=${RULE_SHA}"
  echo "rev_stamp_head=${stamp}"
  echo "bin_sha256=$(sha256_of "$BIN")"
  echo "rounds=${ROUNDS}"
} >"$ENV_INFO"

GATE_LOG="$OUT/load_gate-3612-${LABEL}.log"
echo "series=official threshold=${LOAD_GATE}" >"$GATE_LOG"
ab_round_cooldown_init "$OUT/cooldown-3612-${LABEL}.log"

gate_ok_once() {
  local l1
  l1="$(ab_cooldown_read_load1)"
  [[ "$l1" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1
  [[ "$(awk -v a="$l1" -v t="$LOAD_GATE" 'BEGIN{print (a<t)?1:0}')" == "1" ]]
}
wait_gate() { # <round>
  local waited=0 ok=0
  while :; do
    if gate_ok_once; then ok=1; break; fi
    if [[ "$waited" -ge 1800 ]]; then break; fi
    sleep 30; waited=$((waited + 30))
  done
  if [[ "$ok" == "1" ]]; then
    echo "round$1 dev=metal gate=pass gate_ok=1 threshold=${LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$GATE_LOG"
  else
    echo "round$1 dev=metal gate=fail gate_ok=0 threshold=${LOAD_GATE} waited_s=${waited} load1=$(ab_cooldown_read_load1)" >>"$GATE_LOG"
    return 1
  fi
}

# テスト名 → 出力ファイル（task／arm 別。1 run 1 行追記）。
run_cell() { # <test_name> <task> <arm> <round>
  local t=$1 task=$2 arm=$3 r=$4 res
  res="$OUT/results-3612-${task}-${arm}-${LABEL}.jsonl"
  echo "== 3612 ${task} ${arm} round=${r} =="
  if ! BREAKDOWN_3612_OUT="$res" "$BIN" --ignored --exact --nocapture --test-threads=1 \
    "resident_chain_breakdown_diag_tests_3612::${t}" >"$OUT/run-3612-${task}-${arm}-${LABEL}-r${r}.log" 2>&1; then
    echo "${task} ${arm} round=${r}: $(tail -3 "$OUT/run-3612-${task}-${arm}-${LABEL}-r${r}.log" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
}

pm_log() { pmset -g therm >"$OUT/pmset_therm_$1-3612-${LABEL}.txt" 2>&1 || true; }
pm_log before
for r in 1 2 3 4 5; do
  ab_round_cooldown "$r"
  if ! wait_gate "$r"; then
    echo "round${r}: load gate failed; stopping" >>"$SKIP"
    echo "round${r}: load gate failed; stopping" >&2
    ANY_FAILED=$((ANY_FAILED + 1))
    break
  fi
  # round ごとに起動順を入れ替える（RULE.txt: 奇数 round は B0 先、偶数 round は B1 先）。
  if [[ $((r % 2)) -eq 1 ]]; then
    run_cell resident_chain_breakdown_3612_train_b0 train b0 "$r"
    run_cell resident_chain_breakdown_3612_train_b1 train b1 "$r"
  else
    run_cell resident_chain_breakdown_3612_train_b1 train b1 "$r"
    run_cell resident_chain_breakdown_3612_train_b0 train b0 "$r"
  fi
  # infer は 1 テストが b0／b1 の 2 行を出す。行を task／arm 別ファイルへ分ける。
  tmp="$(mktemp)" || { echo "error: mktemp" >&2; exit 1; }
  # 推論も同じ規定: 偶数 round は B1 先（テスト側が BREAKDOWN_3612_B1_FIRST=1 で計測順を反転する）。
  b1_first=0
  if [[ $((r % 2)) -eq 0 ]]; then b1_first=1; fi
  if BREAKDOWN_3612_B1_FIRST="$b1_first" BREAKDOWN_3612_OUT="$tmp" "$BIN" --ignored --exact --nocapture --test-threads=1 \
    "resident_chain_breakdown_diag_tests_3612::resident_chain_breakdown_3612_infer" >"$OUT/run-3612-infer-${LABEL}-r${r}.log" 2>&1; then
    grep -E '^\{"task":"infer","arm":"b0"' "$tmp" >>"$OUT/results-3612-infer-b0-${LABEL}.jsonl"
    grep -E '^\{"task":"infer","arm":"b1"' "$tmp" >>"$OUT/results-3612-infer-b1-${LABEL}.jsonl"
  else
    echo "infer round=${r}: $(tail -3 "$OUT/run-3612-infer-${LABEL}-r${r}.log" | tr '\n' ' ')" >>"$SKIP"
    ANY_FAILED=$((ANY_FAILED + 1))
  fi
  rm -f "$tmp"
  echo "run ${r}: $(uptime)" | tee -a "$OUT/uptime-3612-${LABEL}.log"
done
pm_log after
echo "done. results in ${OUT}; skipped log: ${SKIP}"
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED} run(s)" >&2; exit 1; fi
