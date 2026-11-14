#!/usr/bin/env bash
# イシュー #4168: 実機セッション用の起動役（Mac mini と GB10 で共用。AB_PROFILE=macmini|gb10）。
# 役割: 実機ロックの取得・解放（時刻を hwlock.log へ記録）→ ランナー（scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh）
#   → 判定器 → 成果物の収録（マスク・マスクの grep 検査・SHA256SUMS）を 1 回で行う。計測中はビルドも別負荷も並走させない。
# 前提: RULE.txt は実体化して単独コミット済み（RULE.template.txt の手順）。AB_BASE_ROOT は RULE の after SHA の clone（.rev-stamp つき）。
# 公開面には機体名を <metal-node>・<cuda-node> とだけ書く。ホスト名・ユーザー名・ホームの絶対パスは収録前に置換し、
# 残りが 1 件でもあれば収録しない（fail-closed）。実名は docs/real-hardware-verification-env.local.md を見るだけで転記しない。
# 行数スモーク（RULE.txt の実体化前。行数と rc だけを報告する。値は見ない）:
#   AB_PROFILE=macmini|gb10 AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> AB_SMOKE_SUMMARY=<abs・未存在> (macmini: AB_PY_TORCH・AB_PY_TF / gb10: AB_PY) bash orchestrate.sh --smoke
# 使い方: AB_PROFILE=macmini AB_BASE_ROOT=<abs> AB_RUNNER_ROOT=<abs> AB_PY_TORCH=<abs> AB_PY_TF=<abs> bash orchestrate.sh [--dry-run]
#   gb10 は AB_PY=<abs>。--dry-run はロックを取らず、ランナーの --dry-run（事前確認だけ）を呼んで終わる。
# bash 3.2 互換。
set -uo pipefail

DRY=0
SMOKE=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1
[[ "${1:-}" == "--smoke" ]] && SMOKE=1
PROFILE=${AB_PROFILE:-}
case "$PROFILE" in
  macmini) LABEL=series-b ;;
  gb10) LABEL=gb10 ;;
  *) echo "error: AB_PROFILE は macmini か gb10" >&2; exit 1 ;;
esac
for v in AB_BASE_ROOT AB_RUNNER_ROOT; do
  p=${!v:-}
  if [[ -z "$p" || "$p" != /* || "$p" == *\"* || "$p" == *\\* || "$p" == *\'* || "$p" == *$'\n'* ]]; then
    echo "error: $v は引用符・バックスラッシュ・改行を含まない絶対パス必須" >&2; exit 1
  fi
done
RUNNER_ROOT=$AB_RUNNER_ROOT
RUNNER="$RUNNER_ROOT/scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh"
JUDGE="$RUNNER_ROOT/scripts/bench/framework-compare/judge_prepost_0150_4168.py"
LOGS="$RUNNER_ROOT/docs/perf/logs/prepost-0150-4168"
RULE="$LOGS/RULE.txt"
if [[ "$SMOKE" == "1" ]]; then
  # 行数スモーク（RULE.txt の実体化前）: RULE.txt は要らず、読みもしない。AB_SMOKE_SUMMARY（絶対パス・未存在）に件数・rc だけを残す。
  [[ -f "$RUNNER" ]] || { echo "error: ランナーが無い" >&2; exit 1; }
else
  [[ -f "$RUNNER" && -f "$JUDGE" && -f "$RULE" ]] || { echo "error: ランナー・判定器・RULE.txt のいずれかが無い" >&2; exit 1; }
fi

if [[ "$DRY" == "1" ]]; then
  exec bash "$RUNNER" --dry-run "$LABEL"
fi

LOCK_DIR="${HOME}/work/.fandhe-hw-lock"
OWNER="#4168 prepost-0150 $(date -u +%FT%TZ)"
mkdir -p "${HOME}/work"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "error: 実機ロックを取得できない（他の計測が保持中）" >&2; exit 1; }
printf '%s\n' "$OWNER" >"$LOCK_DIR/owner"
LOCK_T0="$(date -u +%FT%TZ)"
release() {
  if [[ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" == "$OWNER" ]]; then rm -rf "$LOCK_DIR"; echo "lock released $(date -u +%FT%TZ)"; else echo "LOCK-LOST: 解放しない"; fi
}
trap release EXIT
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM

export AB_HWLOCK_OWNER="$OWNER"
if [[ "$SMOKE" == "1" ]]; then
  # 実ラウンドと同じ手順・同じゲートを 1 ラウンドだけ。計測値は出さず、結果ファイルは削除される（ランナーの trap）。判定・収録はしない。
  bash "$RUNNER" --smoke "smoke-${PROFILE}"; rc=$?
  echo "smoke rc=${rc}"
  exit "$rc"
fi
bash "$RUNNER" "$LABEL"; rc=$?
RAW="$RUNNER_ROOT/scripts/bench/framework-compare/results/raw/ab4168-${LABEL}"
echo "runner rc=${rc}"
[[ -d "$RAW" ]] || { echo "error: 出力が無い: $RAW" >&2; exit 1; }

DEST="$LOGS/$PROFILE"
[[ ! -e "$DEST" ]] || { echo "error: 収録先が既に存在する（差し替え禁止）: $DEST" >&2; exit 1; }
mkdir -p "$DEST"
printf 'acquired=%s\nreleased=pending\n' "$LOCK_T0" >"$DEST/hwlock.log"

# 判定器（生成物は手で書き換えない）。ゲート・RULE の整合も判定器が検査する。
JARGS=(--profile "$PROFILE" --label "$LABEL" --series-dir "$RAW" --env-info "$RAW/env_info.txt" --rule "$RULE" --gate-log "$RAW/gate.log")
if [[ "$PROFILE" == "gb10" ]]; then JARGS+=(--compute-log "$RAW/compute_apps.log"); fi
python3 -I "$JUDGE" "${JARGS[@]}" --ratios-out "$DEST/ratios.tsv" >"$DEST/judge-4168-${PROFILE}.md"; jrc=$?
echo "$jrc" >"$DEST/judge-4168-${PROFILE}.exit"
echo "judge rc=${jrc}"

# 収録: ログと JSONL だけ（target-*・バイナリ・build-*.err は収録しない）
for f in env_info.txt gate.log skipped.log tree-before.txt tree-after.txt consts.txt consts-baseline.txt consts-diff.txt bench-fandhe-diff-stat.txt compute_apps.log; do
  [[ -f "$RAW/$f" ]] && cp "$RAW/$f" "$DEST/$f"
done
# Mac mini の他ジョブゲートの出力（ゲート側がマスク済み。収録後に下のマスク検査をもう一度通す）
if [[ -d "$RAW/foreign-gate" ]]; then mkdir -p "$DEST/foreign-gate"; cp "$RAW/foreign-gate/"* "$DEST/foreign-gate/"; fi
i=1
while [[ "$i" -le 5 ]]; do
  if [[ -d "$RAW/run$i" ]]; then mkdir -p "$DEST/run$i"; cp "$RAW/run$i/"* "$DEST/run$i/"; fi
  i=$((i + 1))
done
# マスク: ホームの絶対パス・ホスト名・ユーザー名を置換（JSONL の数値は触らない）
H="${HOME}"; HN="$(hostname 2>/dev/null || true)"; HNS="$(hostname -s 2>/dev/null || true)"; UN="$(id -un)"
mask_file() {
  local f=$1 t
  t="$(mktemp)"
  LC_ALL=C sed -e "s#${H}#<home>#g" ${HN:+-e "s#${HN}#<host>#g"} ${HNS:+-e "s#${HNS}#<host>#g"} ${UN:+-e "s#${UN}#<user>#g"} "$f" >"$t" && cat "$t" >"$f"
  rm -f "$t"
}
find "$DEST" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.md' -o -name '*.tsv' \) | while IFS= read -r f; do mask_file "$f"; done
leaks=0
for needle in "$H" "$HN" "$HNS" "$UN"; do
  [[ -n "$needle" ]] || continue
  if grep -rFq -- "$needle" "$DEST"; then echo "LEAK: '${needle:0:2}…' が収録物に残る" >&2; leaks=$((leaks + 1)); fi
done
if grep -rEq '([0-9]{1,3}\.){3}[0-9]{1,3}' "$DEST" --include='*.txt' --include='*.log' --include='*.md'; then echo "LEAK: IP アドレスらしき文字列が残る" >&2; leaks=$((leaks + 1)); fi
if [[ "$leaks" -ne 0 ]]; then echo "error: マスク検査に失敗。収録しない（$DEST を確認して削除する）" >&2; exit 1; fi
printf 'acquired=%s\nreleased=%s\n' "$LOCK_T0" "$(date -u +%FT%TZ)" >"$DEST/hwlock.log"
SUMS_TMP="$(mktemp)"
( cd "$DEST" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$f"; else shasum -a 256 "$f"; fi
  done ) >"$SUMS_TMP" && cat "$SUMS_TMP" >"$DEST/SHA256SUMS"
rm -f "$SUMS_TMP"
echo "収録完了: $DEST（rc=${rc} judge=${jrc}）"
[[ "$rc" -eq 0 && "$jrc" -eq 0 ]]
