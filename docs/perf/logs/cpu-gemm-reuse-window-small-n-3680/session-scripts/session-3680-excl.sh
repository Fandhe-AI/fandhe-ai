#!/usr/bin/env bash
# イシュー #3680 GB10 専有系列セッション。RULE.txt が正（不変）。
# session-3671-excl.sh（#3671 の専有系列）の hw-lock・30 秒 poller・系列ラベル（CUR）・出力先の存在で中止する骨格を写し、
# 計測本体だけを #3680 用に差し替えた。1 本の切り離しセッションで直列に実行する:
#   (1) diag   診断の正式系列 5 run（#3671 の orchestrate.sh。cpu_gemm_reuse_window_phases）
#   (2) bench  bench-fandhe（計測木 HEAD への path patch ビルド 1 本）を 5 腕 x N x 3 モード x 5 run
#              腕: base／vafter／tunboth／tunmmap／tuntrim（run ごとに巡回順を反転）
# ビルドは事前に完了済み（診断テストバイナリは README の手順で事前ビルド、bench バイナリは build_bench_head.sh）。
# 値（GLIBC_TUNABLES のリテラル・腕名・N）はすべてこのスクリプト内の固定定数で、外部入力から組み立てない。
# ホスト名・ユーザー名のマスクは回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES GEMM_WINDOW_ONLY_ARM
R=$HOME/work/r3680-excl
T=$R/runner
O=$R/out
BINDIR=$R/bench-bin
BIN=$BINDIR/bench-fandhe-head
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log
GATE=$O/load_gate_bench.log
GPU=$O/gpu_procs.txt
POLL=$O/gpu_poll.log
CUR=$O/current_series
POLL_INTERVAL=30
RES3671=$T/docs/perf/logs/cpu-gemm-reuse-window-3671
RES=$T/docs/perf/logs/cpu-gemm-reuse-window-small-n-3680
SIZES="256 512 1024 2048 4096"
# RULE.txt 固定定数
TUN_BOTH="glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824"
TUN_MMAP="glibc.malloc.mmap_threshold=33554432"
TUN_TRIM="glibc.malloc.trim_threshold=1073741824"
ARMS_ODD="base vafter tunboth tunmmap tuntrim"
ARMS_EVEN="tuntrim tunmmap tunboth vafter base"

# 既存の出力があれば中止（専有系列は新しい作業ディレクトリでだけ取る。上書き・追記をしない）。
if [ -e "$O" ]; then echo "output dir exists: $O" >&2; exit 5; fi
[ -x "$BIN" ] || { echo "bench バイナリが無い: $BIN（build_bench_head.sh を先に実行）" >&2; exit 6; }
mkdir -p "$O" || exit 1
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
echo "#3680 $(date -u +%FT%TZ)" >"$LOCK_DIR/owner"

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
# 腕ごとの 1 要因（環境変数 or フラグ）はここだけで決まる。
run_cell() {
  local arm=$1 n=$2 r=$3 modes m tun="" extra=""
  case "$arm" in
    base) ;;
    vafter) extra="--diag-verify-after-loop" ;;
    tunboth) tun="$TUN_BOTH" ;;
    tunmmap) tun="$TUN_MMAP" ;;
    tuntrim) tun="$TUN_TRIM" ;;
    *) echo "unknown arm" >&2; return 1 ;;
  esac
  if [ $((r % 2)) = 1 ]; then modes="fresh reuse phases"; else modes="phases reuse fresh"; fi
  for m in $modes; do
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
step "start tree=$(sed -n 1p "$T/.rev-stamp")"

# ---- (1) 診断の正式系列 ----
snap "diag:pre"
echo diag >"$CUR"
bash "$RES3671/orchestrate.sh" gb10 --out "$O/gb10" >"$O/orch-gb10.log" 2>&1
step "orchestrate-diag rc=$?"
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
      gate "bench_$a/n${n}/r${r}" || abort_gate "bench_$a/n${n}/r${r}"
      run_cell "$a" "$n" "$r"
    done
  done
done
step "bench done"
echo idle >"$CUR"
snap "bench:post"

# ---- env_info（judge のドリフトガード用）----
{
  echo "rev_stamp_head: $(sed -n 1p "$T/.rev-stamp")"
  echo "rev_stamp_dirty: $(sed -n 2p "$T/.rev-stamp")"
  echo "rule_sha256: $(sha256_of "$RES/RULE.txt")"
  echo "bench_binary_sha256: $(sha256_of "$BIN")"
  [ ! -f "$BINDIR/arms.txt" ] || sed 's/^/  /' "$BINDIR/arms.txt"
} >"$O/env_info_3680.txt"

step "session body done"
echo "done." >>"$LOG"
