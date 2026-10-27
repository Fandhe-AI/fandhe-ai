#!/usr/bin/env bash
# イシュー #3612: A0（全体窓）専用ランナー。bench-fandhe（#3257 の bench-resident-chain.patch を当てたツリー）の
# `--device metal --mode chain` を Mac mini で 5 round 取り、step 中央値 5 値（us）を出力する。
# 手順の正本: docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt の [腕] A0 と [限界]
# （HEAD の fandhe-ai ピンが合わない場合、A0 は a8e91f0c9 のツリーで取る）。本スクリプトは RULE の A0 定義を
# 実装するだけで、条件は足さない。値は run_breakdown_resident_chain_metal.sh の B0/B1 とは別ツリー・別ランで取り、
# summarize_breakdown_resident_chain_metal.py の --a0-train-us／--a0-infer-us へ渡す。
# 兄弟 run_ab_resident_chain.sh の after 腕（BENCH_PARAM_DIGEST=1 ... --mode chain）と同じ呼び出し。
# 書式: `set -u`・bash 3.2 互換（mapfile／連想配列／eval 不使用）。失敗は捏造せず skipped へ記録し非 0 終了。
# 呼び出し例: A0_TREE=<abs>/a0 bash run_a0_resident_chain_metal_3612.sh 3612-macmini-a0
set -u

LABEL=${1:-}
ROUNDS=${AB_ROUNDS:-5}
LOAD_GATE=3.0
A0_HEAD_FULL=a8e91f0c99fa40b0bbdb26a8ebd7f05f08ecf9d0
if [[ -z "$LABEL" || ! "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: A0_TREE=<abs> $0 <label>  (label: [A-Za-z0-9._-]+)" >&2; exit 1
fi
if [[ "$ROUNDS" != "5" ]]; then
  echo "error: AB_ROUNDS は 5 固定（RULE.txt。got: ${ROUNDS}）" >&2; exit 1
fi
if [[ -z "${AB_ROUND_COOLDOWN_MAX_LOAD1:-}" ]]; then
  AB_ROUND_COOLDOWN_MAX_LOAD1=3.0
elif [[ "$AB_ROUND_COOLDOWN_MAX_LOAD1" != "3.0" ]]; then
  echo "error: AB_ROUND_COOLDOWN_MAX_LOAD1 は 3.0 固定（got: ${AB_ROUND_COOLDOWN_MAX_LOAD1}）" >&2; exit 1
fi
export AB_ROUND_COOLDOWN_MAX_LOAD1

# A03: --config 文字列へ埋め込むため、絶対パス・引用符／バックスラッシュ／改行の拒否。
A0_TREE=${A0_TREE:-}
if [[ -z "$A0_TREE" || "$A0_TREE" != /* || "$A0_TREE" == *\"* || "$A0_TREE" == *\\* || "$A0_TREE" == *$'\n'* ]]; then
  echo "error: A0_TREE は絶対パス必須で、引用符・バックスラッシュ・改行を含まない（got: ${A0_TREE:-unset}）" >&2; exit 1
fi

# 機体同定（fail-closed）。
brand="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
if [[ "$brand" != "Apple M4" ]]; then
  echo "error: 'Apple M4' のみ（got: ${brand:-unknown}）" >&2; exit 1
fi

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
SELF_SHA="$(sha256_of "$SELF_DIR/$(basename "$0")")"

# shellcheck source=./ab_round_cooldown.sh
source "$A0_TREE/scripts/bench/framework-compare/ab_round_cooldown.sh" || { echo "error: ab_round_cooldown.sh を読めない" >&2; exit 1; }

FC="$A0_TREE/scripts/bench/framework-compare"
OUT="$FC/results/raw"
mkdir -p "$OUT"
SKIP="$OUT/skipped-a0-3612-${LABEL}.log"
ENV_INFO="$OUT/a0-env_info-3612-${LABEL}.txt"
if [[ -e "$SKIP" || -e "$ENV_INFO" ]]; then
  echo "error: ラベル ${LABEL} の出力が既に存在する。新しいラベルを使うこと" >&2; exit 1
fi
: >"$SKIP"
fail() { echo "precheck FAILED: $1" | tee -a "$SKIP" >&2; exit 1; }

# ツリー検証（HEAD・.rev-stamp・変更が所定パッチの 2 ファイルだけ・パッチ完全一致）。
real="$(git -C "$A0_TREE" rev-parse HEAD 2>/dev/null || true)"
[[ "$real" == "$A0_HEAD_FULL" ]] || fail "A0 ツリーの HEAD が ${A0_HEAD_FULL} でない（got: ${real:-none}）"
stamp="$(head -n1 "$A0_TREE/.rev-stamp" 2>/dev/null || true)"
dirty="$(sed -n 2p "$A0_TREE/.rev-stamp" 2>/dev/null || true)"
[[ "$stamp" == "$real" && "$dirty" == "dirty=0" ]] || fail ".rev-stamp が無い・HEAD と不一致・dirty=0 でない"
PATCH_FILE="$A0_TREE/docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch"
RULE_FILE_REL="docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt"
[[ -f "$PATCH_FILE" ]] || fail "bench-resident-chain.patch が無い"
status_lines="$(git -C "$A0_TREE" status --porcelain | grep -v '^??' | sort || true)"
want_status=" M scripts/bench/framework-compare/bench-common/src/lib.rs
 M scripts/bench/framework-compare/bench-fandhe/src/main.rs"
[[ "$status_lines" == "$want_status" ]] || fail "変更が bench-common/lib.rs・bench-fandhe/main.rs の 2 件だけでない（${status_lines}）"
git -C "$A0_TREE" apply --reverse --check "$PATCH_FILE" || fail "patch が A0 ツリーと一致しない"
TMP_IDX="$(mktemp "${TMPDIR:-/tmp}/a0-idx.XXXXXX")" || fail "一時 index を作れない"
rm -f "$TMP_IDX"
if ! GIT_INDEX_FILE="$TMP_IDX" git -C "$A0_TREE" read-tree HEAD \
  || ! GIT_INDEX_FILE="$TMP_IDX" git -C "$A0_TREE" apply --cached "$PATCH_FILE"; then
  rm -f "$TMP_IDX"; fail "HEAD へ所定パッチを適用できない"
fi
if ! GIT_INDEX_FILE="$TMP_IDX" git -C "$A0_TREE" diff --quiet -- \
  scripts/bench/framework-compare/bench-common/src/lib.rs \
  scripts/bench/framework-compare/bench-fandhe/src/main.rs; then
  rm -f "$TMP_IDX"; fail "対象 2 ファイルが HEAD + 所定パッチと完全一致しない"
fi
rm -f "$TMP_IDX"
MAIN_RS="$FC/bench-fandhe/src/main.rs"
grep -qE '^[[:space:]]*\("train", "chain", false\) =>' "$MAIN_RS" || fail "main.rs に (train, chain) の分岐が無い"
grep -qE '^[[:space:]]*\("infer", "chain", false\) =>' "$MAIN_RS" || fail "main.rs に (infer, chain) の分岐が無い"
echo "precheck=assert ok"

# ビルド（計測の前に 1 回。path patch は --config 引数のみ）。
PATCH_CFG="patch.crates-io.fandhe-ai.path=\"${A0_TREE}/crates/facade\""
TDIR="$A0_TREE/target-a0-3612"
cd "$FC" || exit 1
if ! cargo build --release -p bench-fandhe --target-dir "$TDIR" --config "$PATCH_CFG" 2>"$OUT/a0-build-3612-${LABEL}.err"; then
  tail -40 "$OUT/a0-build-3612-${LABEL}.err"; echo "bench-fandhe BUILD FAILED" >>"$SKIP"; exit 1
fi
tree="$(cargo tree -p bench-fandhe --depth 1 --config "$PATCH_CFG" 2>&1)"
if ! echo "$tree" | grep -qE 'fandhe-ai v[0-9.]+ \(.*crates/facade\)'; then
  echo "$tree" >&2; fail "fandhe-ai が path patch に解決されていない"
fi
echo "$tree" >"$OUT/a0-tree-3612-${LABEL}.txt"
BIN="$OUT/a0-bench-fandhe-3612-${LABEL}"
cp "$TDIR/release/bench-fandhe" "$BIN"
echo "$(sha256_of "$BIN")  a0-bench-fandhe-3612-${LABEL}" >"$OUT/a0-sha-3612-${LABEL}.txt"

{
  echo "host=masked"
  echo "uname=$(uname -srm)"
  echo "cpu_brand=${brand}"
  echo "rustc=$(rustc --version 2>/dev/null)"
  echo "a0_tree_head=${real}"
  echo "patch_sha256=$(sha256_of "$PATCH_FILE")"
  echo "rule_sha256=$(sha256_of "$A0_TREE/$RULE_FILE_REL" 2>/dev/null || echo none)"
  echo "script_sha256=${SELF_SHA}"
  echo "bin_sha256=$(awk '{print $1}' "$OUT/a0-sha-3612-${LABEL}.txt")"
  echo "rounds=${ROUNDS}"
} >"$ENV_INFO"

GATE_LOG="$OUT/a0-load_gate-3612-${LABEL}.log"
echo "series=official threshold=${LOAD_GATE}" >"$GATE_LOG"
ab_round_cooldown_init "$OUT/a0-cooldown-3612-${LABEL}.log"
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

unset RAYON_NUM_THREADS
for task in train infer; do : >"$OUT/a0-3612-${LABEL}-${task}.jsonl"; done
ANY_FAILED=0
pm_log() { pmset -g therm >"$OUT/a0-pmset_therm_$1-3612-${LABEL}.txt" 2>&1 || true; }
pm_log before
for r in 1 2 3 4 5; do
  ab_round_cooldown "$r"
  if ! wait_gate "$r"; then
    echo "round${r}: load gate failed; stopping" >>"$SKIP"; ANY_FAILED=$((ANY_FAILED + 1)); break
  fi
  for task in train infer; do
    errf="$OUT/a0-digest-3612-${LABEL}-${task}-r${r}.log"
    echo "== a0 ${task} metal round=${r} =="
    if ! BENCH_PARAM_DIGEST=1 "$BIN" --task "$task" --device metal --mode chain \
      --out "$OUT/a0-3612-${LABEL}-${task}.jsonl" 2>"$errf"; then
      echo "task=${task} round=${r}: $(tail -3 "$errf" | tr '\n' ' ')" >>"$SKIP"
      ANY_FAILED=$((ANY_FAILED + 1))
    fi
  done
  echo "run ${r}: $(uptime)" | tee -a "$OUT/a0-uptime-3612-${LABEL}.log"
done
pm_log after
if [[ "$ANY_FAILED" -gt 0 ]]; then echo "FAILED: ${ANY_FAILED}" >&2; exit 1; fi

# 変換: 各 JSONL がちょうど 5 行・median_s が有限の正数であることを検証し、us（小数 3 桁）で出力する。
if ! python3 -I - "$OUT/a0-3612-${LABEL}-train.jsonl" "$OUT/a0-3612-${LABEL}-infer.jsonl" >"$OUT/a0-3612-${LABEL}-us.txt" 2>>"$SKIP" <<'PY'
import json, math, sys
for name, path in (("train", sys.argv[1]), ("infer", sys.argv[2])):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if len(rows) != 5:
        raise SystemExit(f"{name}: 行数が 5 でない ({len(rows)})")
    vals = []
    for row in rows:
        v = row["median_s"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise SystemExit(f"{name}: median_s が有限の正数でない ({v!r})")
        vals.append(f"{v * 1e6:.3f}")
    print(f"{name}=" + ",".join(vals))
PY
then
  echo "convert FAILED" >&2; exit 1
fi
echo "done. results in ${OUT}"
