#!/bin/bash
# #4160 RULE.v2 の系列 1 本: 項 4 (b) の確認 → （通れば）純カーネル orchestrate → セル A/B → 全ゲート通過の判定。値は見ない。
# 使い方: run4160v2.sh <展開先 W> <k> <試みの番号 n> <run|dry>
#   dry は確認の手順だけを通す（hw-lock は取って外す。計測は起動しない。出力は W/precheck-dry-<n>.txt）。
set -u
W=$1; K=$2; N=$3; MODE=$4
R=docs/perf/logs/metal-steel-large-n-ab-macmini-4160
FG="$W/tree/docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py"
FC="$W/tree/scripts/bench/framework-compare"
LOG="$W/formal-v2.drive.log"
EXP_AB=$(cat "$W/fp-deploy-ab.txt")
SER="formal-v2-$K"; LABEL="fv2-$K"
utc() { date -u +%FT%TZ; }
case "$MODE" in run) A="$W/attempt-v2-$K-$N.txt";; dry) A="$W/precheck-dry-$N.txt";; *) echo "mode?" >&2; exit 2;; esac
[ -e "$A" ] && { echo "exists: $A" >&2; exit 2; }
[ "$MODE" = run ] && [ -e "$W/$SER" ] && { echo "exists: $W/$SER" >&2; exit 2; }
: > "$A"
say() { echo "$*" >> "$A"; }
LOCKED=0; LOCK="$HOME/work/.fandhe-hw-lock"
release() { if [ $LOCKED = 1 ]; then rm -f "$LOCK/owner"; rmdir "$LOCK"; LOCKED=0; say "lock released $(utc)"; fi; }
notstart() { say "result=not-started reason=$1 $(utc)"; release
  if [ "$MODE" = run ]; then mv "$A" "$W/prestart-v2-$K-$N.txt"; echo "$SER attempt $N NOT-STARTED reason=$1 $(utc)" >> "$LOG"; fi
  exit 5; }
proc_check() { # $1=回目
  local g="$W/.pc-gate-$K-$N-$1.txt" grc hits app
  rm -f "$g"
  python3 -I "$FG" --phase before --out "$g" > /dev/null 2>&1; grc=$?
  app=$(ps -A -o comm= | awk -F/ '$NF=="Safari"' | wc -l | tr -d ' ')
  if [ -f "$g" ]; then hits=$(grep -cE '^hit G[13]-' "$g"); else hits=missing; fi
  say "proc-check$1 $(utc) safari_app=$app gate_rc=$grc g1g3_hits=$hits"
  say "  safari-named (pcpu basename):"
  ps -A -o pcpu=,comm= | awk '{p=$1; $1=""; sub(/^ +/,""); n=split($0,a,"/"); if (a[n] ~ /Safari/) print "    " p " " a[n]}' | LC_ALL=C sort -k2 >> "$A"
  say "  gate output:"; [ -f "$g" ] && sed 's/^/    /' "$g" >> "$A"; rm -f "$g"
  [ "$app" = 0 ] && [ "$hits" = 0 ] && [ "$grc" != 2 ]
}
say "$SER attempt=$N mode=$MODE begin $(utc)"
# 1. プロセスの確認（1 回目）
proc_check 1 || notstart proc-check1
# 2. mediaanalysisd（2 分間隔で 3 回 %CPU 5 未満）
ok=1
for i in 1 2 3; do
  v=$(ps -A -o pcpu=,comm= | awk '/mediaanalysisd$/{print $1; f=1} END{if(!f) print "absent"}')
  say "mediaanalysisd check$i $(utc) pcpu=$v"
  case "$v" in absent) ;; *) awk -v x="$v" 'BEGIN{exit !(x < 5)}' || ok=0;; esac
  [ $i -lt 3 ] && sleep 120
done
n=$(log show --last 10m --style compact --predicate 'process == "spotlightknowledged" AND eventMessage CONTAINS "Submitting text processing"' 2>/dev/null | awk 'NR>1' | wc -l | tr -d ' ')
say "spotlightknowledged_submitting_last10m=$n idle_ok=$ok"
[ $ok = 1 ] || notstart mediaanalysisd
# 3. hw-lock と load1
if ! mkdir "$LOCK" 2>/dev/null; then say "LOCK-BUSY owner=$(cat "$LOCK/owner" 2>/dev/null)"; notstart lock-busy; fi
LOCKED=1; trap release EXIT
printf '#4160 measure %s %s\n' "$SER-$MODE" "$(utc)" > "$LOCK/owner"
say "lock-acquired $(utc) owner=$(cat "$LOCK/owner")"
t0=$(date +%s); k=0
while :; do k=$((k+1)); l=$(sysctl -n vm.loadavg | awk '{print $2}'); say "load1 sample $k $(utc) load1=$l"
  if awk -v a="$l" 'BEGIN{exit !(a<1.5)}'; then say "load1-wait done waited_s=$(( $(date +%s)-t0 )) load1=$l reached_below_1.5=yes"; break; fi
  if [ $k -ge 40 ]; then say "load1-wait done waited_s=$(( $(date +%s)-t0 )) load1=$l reached_below_1.5=no"; break; fi; sleep 30; done
# 4. プロセスの確認（2 回目。起動の直前・hw-lock の下）
proc_check 2 || notstart proc-check2
# 5. ツリーの照合（results/ と target-ab-4160-* を除く指紋 + 配置時 manifest）
fpab=$(bash "$W/fp4160.sh" "$W/tree" ab)
( cd "$W/tree" && shasum -a 256 -c --quiet "$W/manifest-deploy.txt" ) > "$W/.mf-$K-$N.txt" 2>&1; mrc=$?
say "tree-check $(utc) fingerprint_ab_excluded: $fpab expected: $EXP_AB manifest_check_rc=$mrc"
rm -f "$W/.mf-$K-$N.txt"
[ "$fpab" = "$EXP_AB" ] && [ $mrc = 0 ] || notstart tree-mismatch
if [ "$MODE" = dry ]; then say "result=dry-ok (no measurement started) $(utc)"; exit 0; fi
# 起動
mkdir "$W/$SER" || notstart outdir-exists
say "series-start $(utc)"
mv "$A" "$W/$SER/series-start.txt"; A="$W/$SER/series-start.txt"
ST="$W/$SER.status"
echo "$SER begin $(utc)" >> "$LOG"
echo "start $(utc) owner=$(cat "$LOCK/owner")" >> "$ST"
( cd "$W/tree/$R" && env -u RAYON_NUM_THREADS FOREIGN_GATE="$FG" CARGO_TARGET_DIR="$W/target" OUT="$W/$SER/kern" sh orchestrate_macmini.sh ) > "$W/$SER/orchestrate.out" 2>&1
echo "orchestrate rc=$? $(utc)" >> "$ST"
( cd "$FC" && env -u RAYON_NUM_THREADS -u CARGO_TARGET_DIR FOREIGN_GATE="$FG" AB_BEFORE_FACADE_PATH="$W/before/crates/facade" AB_AFTER_FACADE_PATH="$W/after/crates/facade" bash run_ab_steel_large_n_gemm_metal.sh "$LABEL" ) > "$W/$SER/ab.out" 2>&1
echo "ab rc=$? $(utc)" >> "$ST"
echo "all done $(utc)" >> "$ST"
# 全ゲート通過の判定（ゲートの行と終了コードだけ。RULE.v2 項 3）
KG="$W/$SER/kern/load_gate.log"; AG="$FC/results/raw/load-gate-4160-metal-$LABEL.log"
orc=$(grep -c '^orchestrate rc=0 ' "$ST"); abc=$(grep -c '^ab rc=0 ' "$ST")
kn=$(grep -c 'gate=' "$KG" 2>/dev/null); kp=$(grep -c 'gate=pass' "$KG" 2>/dev/null); kf=$(grep -c 'FAILED' "$KG" 2>/dev/null); kd=$(grep -c '^series done' "$KG" 2>/dev/null)
an=$(grep -c 'gate=' "$AG" 2>/dev/null); ap=$(grep -c 'gate=pass' "$AG" 2>/dev/null)
v=notpass
if [ "$orc" = 1 ] && [ "$abc" = 1 ] && [ "${kn:-0}" -gt 0 ] && [ "$kn" = "$kp" ] && [ "${kf:-1}" = 0 ] && [ "${kd:-0}" -ge 1 ] && [ "${an:-0}" -gt 0 ] && [ "$an" = "$ap" ]; then v=pass; fi
echo "$SER verdict=$v orchestrate_rc0=$orc ab_rc0=$abc kern_gate=$kp/$kn kern_failed=$kf kern_series_done=$kd ab_gate=$ap/$an $(utc)" >> "$LOG"
