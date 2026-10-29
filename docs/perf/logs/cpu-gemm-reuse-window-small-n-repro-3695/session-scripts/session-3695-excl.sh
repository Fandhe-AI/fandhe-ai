#!/usr/bin/env bash
# イシュー #3695 GB10 専有系列セッション（1 系列 = 1 セッション）。RULE.txt が正（不変）。
# 使い方: session-3695-excl.sh <系列ラベル s1..s5>
# session-3680-excl.sh（#3680 の専有系列）の hw-lock・30 秒 poller・系列ラベル（CUR）・出力先の存在で中止する骨格を写し、
# 腕を 4 本（base／vafter／tunmmap／vafter_tunmmap）へ差し替え、系列ラベルで出力先を分けた。
# 1 本の切り離しセッションで直列に実行する:
#   (1) diag   診断の正式系列 5 run（#3671 の orchestrate.sh。cpu_gemm_reuse_window_phases）
#   (2) bench  bench-fandhe（計測木 HEAD への path patch ビルド 1 本。全系列で同一バイナリ）を 4 腕 x N x 3 モード x 5 run
#   (3) excl_check.py で専有判定（excl_status.txt）と env_info を書く
# ビルドは事前に完了済み（診断テストバイナリは README の手順、bench バイナリは #3680 の build_bench_head.sh）。
# 外部入力は系列ラベルだけで、s1〜s5 の allowlist で照合する。値（GLIBC_TUNABLES のリテラル・腕名・N）は固定定数で、
# 外部入力から組み立てない。ホスト名・ユーザー名のマスクは回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
# 系列ラベルの allowlist（パス組み立てに使う唯一の外部入力）。
LABEL="${1:-}"
case "$LABEL" in
  s1|s2|s3|s4|s5) ;;
  *) echo "usage: $0 <s1|s2|s3|s4|s5>" >&2; exit 2 ;;
esac
# RAYON_NUM_THREADS は無条件に unset せず、設定済みなら（計測条件の混入として）先に拒否して終了する。
if [ -n "${RAYON_NUM_THREADS+x}" ]; then echo "RAYON_NUM_THREADS が設定済みのため中止する" >&2; exit 7; fi
unset CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES GEMM_WINDOW_ONLY_ARM
R=$HOME/work/r3695-excl
T=$R/runner
O=$R/out-${LABEL}
BINDIR=$R/bench-bin
BIN=$BINDIR/bench-fandhe-head
LOCK_DIR=$HOME/work/.fandhe-hw-lock
SS=$O/session-scripts
LOG=$SS/steps.log
GATE=$SS/load_gate_bench.log
GPU=$SS/gpu_procs.txt
POLL=$SS/gpu_poll.log
CUR=$SS/current_series
POLL_INTERVAL=30
RES3671=$T/docs/perf/logs/cpu-gemm-reuse-window-3671
RES3646=$T/docs/perf/logs/cpu-gemm-resident-reuse-3646
RES=$T/docs/perf/logs/cpu-gemm-reuse-window-small-n-repro-3695
SIZES="256 512 1024 2048 4096"
# RULE.txt 固定定数
TUN_MMAP="glibc.malloc.mmap_threshold=33554432"
ARMS_ODD="base vafter tunmmap vafter_tunmmap"
ARMS_EVEN="vafter_tunmmap tunmmap vafter base"

# 既存の出力があれば中止（専有系列は新しい作業ディレクトリでだけ取る。上書き・追記をしない）。
if [ -e "$O" ]; then echo "output dir exists: $O" >&2; exit 5; fi
[ -x "$BIN" ] || { echo "bench バイナリが無い: $BIN（build_bench_head.sh を先に実行）" >&2; exit 6; }
# bench バイナリのビルド元と計測木の一致を保証する。arms.txt を必須とし、
# ビルド時 rev-stamp・バイナリ sha256 を現在の計測木（.rev-stamp）・実行ファイルと照合する（不一致は中止）。
ARMS=$BINDIR/arms.txt
[ -f "$ARMS" ] || { echo "arms.txt が無い: $ARMS（build_bench_head.sh の出力が必須）" >&2; exit 6; }
[ -f "$T/.rev-stamp" ] || { echo ".rev-stamp が無い: $T" >&2; exit 6; }
sha256_pre() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}
want_rev="rev_stamp: $(sed -n 1p "$T/.rev-stamp") $(sed -n 2p "$T/.rev-stamp")"
[ "$(sed -n 's/^rev_stamp: .*/&/p' "$ARMS" | head -n 1)" = "$want_rev" ] \
  || { echo "arms.txt の rev_stamp が計測木と不一致（bench バイナリのビルド元が異なる）" >&2; exit 8; }
[ "$(sed -n 2p "$T/.rev-stamp")" = "dirty=0" ] || { echo "計測木が dirty" >&2; exit 8; }
want_sha="$(sed -n 's/^bench-fandhe-head_sha256: //p' "$ARMS" | head -n 1)"
[ -n "$want_sha" ] && [ "$want_sha" = "$(sha256_pre "$BIN")" ] \
  || { echo "bench バイナリの sha256 が arms.txt と不一致" >&2; exit 8; }
mkdir -p "$SS" || exit 1
echo idle >"$CUR"
step() { echo "$(date -u +%FT%TZ) $*" >>"$LOG"; }

# ---- 実機ロック（取れるまで 60 秒おき。最大 3 時間）----
got=0
i=0
while [ "$i" -lt 180 ]; do
  if mkdir "$LOCK_DIR" 2>/dev/null; then got=1; break; fi
  sleep 60
  i=$((i + 1))
done
if [ "$got" != 1 ]; then step "hwlock timeout"; echo "done. rc=lock" >>"$LOG"; exit 3; fi
echo "#3695 ${LABEL} $(date -u +%FT%TZ)" >"$LOCK_DIR/owner"

# ---- compute-apps poller（30 秒固定間隔。停止も変更もしない観測のみ。pid が出たら ps を添える）----
poller() {
  local c rc out n pids
  while :; do
    c="$(cat "$CUR" 2>/dev/null || echo unknown)"
    out="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"
    rc=$?
    if [ "$rc" = 0 ]; then
      if [ -z "$out" ]; then n=0; else n="$(printf '%s\n' "$out" | wc -l | tr -d ' ')"; fi
    else
      n=NA
    fi
    echo "$(date -u +%FT%TZ) series=$c rc=$rc apps=$n $(printf '%s' "$out" | tr '\n' ';')" >>"$POLL"
    if [ "$rc" = 0 ] && [ "$n" != 0 ]; then
      pids="$(printf '%s\n' "$out" | cut -d, -f1 | tr -d ' ' | paste -sd, -)"
      ps -o pid,ppid,comm -p "$pids" >>"$POLL" 2>&1
    fi
    sleep "$POLL_INTERVAL"
  done
}
poller &
POLLER_PID=$!
trap 'kill "$POLLER_PID" 2>/dev/null; rm -rf "$LOCK_DIR"; step "hwlock released"' EXIT
step "hwlock acquired"

snap() { # $1=ラベル:pre|post。常駐 GPU プロセスと該当 PID の ps を記録する（停止・変更はしない）。
  {
    echo "== $1 $(date -u +%FT%TZ)"
    local o r pids
    o="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"
    r=$?
    echo "rc=$r"
    printf '%s\n' "$o"
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -)"
    [ -z "$pids" ] || ps -o pid,%cpu,etime,comm -p "$pids"
    uptime
  } >>"$GPU" 2>&1
}

gate() { # $1=ラベル。load1 < 1.0 を 30 秒間隔・最大 30 分待つ（RULE.txt ゲート）。通過 0／未通過 1。
  local w=0 l
  while :; do
    l="$(uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}')"
    echo "$1 load1=$l waited=${w}s" >>"$GATE"
    if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v l="$l" 'BEGIN{exit !(l < 1.0)}'; then
      echo "$1 gate=pass" >>"$GATE"
      return 0
    fi
    if [ "$w" -ge 1800 ]; then echo "$1 gate=unpassed" >>"$GATE"; return 1; fi
    sleep 30
    w=$((w + 30))
  done
}

abort_gate() { step "gate unpassed: $1 -> abort"; echo "done. rc=gate-unpassed" >>"$LOG"; exit 4; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# 1 つの (腕, N, run) で 3 モードを実行する。$1=腕 $2=N $3=run。
# 腕ごとの要因（環境変数・フラグ）はここだけで決まる。環境変数は 3 モードすべて、フラグは reuse／phases のみ。
run_cell() {
  local arm=$1 n=$2 r=$3 modes m tun="" extra=""
  case "$arm" in
    base) ;;
    vafter) extra="--diag-verify-after-loop" ;;
    tunmmap) tun="$TUN_MMAP" ;;
    vafter_tunmmap) tun="$TUN_MMAP"; extra="--diag-verify-after-loop" ;;
    *) echo "unknown arm" >&2; return 1 ;;
  esac
  if [ $((r % 2)) = 1 ]; then modes="fresh reuse phases"; else modes="phases reuse fresh"; fi
  for m in $modes; do
    # 負荷ゲートは 3 モード（fresh・reuse・phases）それぞれの起動前に通す（モードごとの通過記録を残す）。
    gate "bench_$arm/n${n}/r${r}/${m}" || abort_gate "bench_$arm/n${n}/r${r}/${m}"
    local f="$O/bench/$arm/gemm_${m}_n${n}_r${r}"
    local args flag=""
    case "$m" in
      fresh) args="--mode fresh" ;;
      reuse) args="--mode reuse"; flag="$extra" ;;
      phases) args="--mode reuse --phases"; flag="$extra" ;;
    esac
    # shellcheck disable=SC2086
    if [ -n "$tun" ]; then
      env "GLIBC_TUNABLES=$tun" "$BIN" --task gemm --device cpu $args $flag --size "$n" \
        --out "$f.jsonl" >"$f.stdout.log" 2>&1
    else
      "$BIN" --task gemm --device cpu $args $flag --size "$n" \
        --out "$f.jsonl" >"$f.stdout.log" 2>&1
    fi
    echo "$m n=$n r=$r rc=$?" >>"$O/bench/$arm/rc.log"
  done
}

cd "$T" || exit 1
step "start series=${LABEL} tree=$(sed -n 1p "$T/.rev-stamp")"

# ---- (1) 診断の正式系列 ----
snap "diag:pre"
echo diag >"$CUR"
# 診断系列も起動前に負荷ゲートを通す。未通過なら起動せず中止する。
gate "diag:pre" || abort_gate "diag:pre"
# --abort-on-gate-unpassed: run ごとのゲート（30 分）未通過ならそのラン以降を起動せず rc=4 で止める（RULE.txt「各起動前・未通過は abort」）。
bash "$RES3671/orchestrate.sh" gb10 --abort-on-gate-unpassed --out "$O/gb10" >"$SS/orch-gb10.log" 2>&1
orc=$?
step "orchestrate-diag rc=$orc"
if [ "$orc" = 4 ]; then abort_gate "diag orchestrate 内の run で未通過"; fi
# 念のための二重確認（未通過記録が残っていれば bench 系列も中止する）。
if grep -q 'gate=unpassed' "$O/gb10/load_gate_status.txt" 2>/dev/null; then
  abort_gate "diag orchestrate 内の run で未通過"
fi
echo idle >"$CUR"
snap "diag:post"

# ---- (2) bench 系列（腕を run ごとに巡回。--out を必ず明示。既定の results/ へ追記させない）----
for a in $ARMS_ODD; do mkdir -p "$O/bench/$a"; done
snap "bench:pre"
echo bench >"$CUR"
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then arms="$ARMS_ODD"; else arms="$ARMS_EVEN"; fi
  for a in $arms; do
    for n in $SIZES; do
      run_cell "$a" "$n" "$r"
    done
  done
done
step "bench done"
echo idle >"$CUR"
snap "bench:post"

# ---- (3) env_info（judge のドリフトガード用）と専有判定 ----
{
  echo "series_label: ${LABEL}"
  echo "rev_stamp_head: $(sed -n 1p "$T/.rev-stamp")"
  echo "rev_stamp_dirty: $(sed -n 2p "$T/.rev-stamp")"
  echo "rule_sha256: $(sha256_of "$RES/RULE.txt")"
  echo "bench_binary_sha256: $(sha256_of "$BIN")"
  [ ! -f "$BINDIR/arms.txt" ] || sed 's/^/  /' "$BINDIR/arms.txt"
} >"$SS/env_info_3695.txt"
cp "$T/.rev-stamp" "$O/.rev-stamp"
# excl_check.py が $SS/excl_status.txt を自分で書く（標準出力は別ファイルへ）。
# 失敗しても rc を記録して先へ進み、終端行に反映する（#3679 の作法）。
python3 -I "$RES3646/excl_check.py" "$SS" --series diag,bench >"$SS/excl_check.out" 2>"$SS/excl_check.err"
erc=$?
step "excl_check rc=$erc"

step "session body done"
if [ "$erc" != 0 ]; then echo "done. rc=excl-check-$erc" >>"$LOG"; else echo "done. rc=0" >>"$LOG"; fi
