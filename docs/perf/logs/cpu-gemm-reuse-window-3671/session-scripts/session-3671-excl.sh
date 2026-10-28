#!/usr/bin/env bash
# イシュー #3671 GB10 専有系列セッション。RULE.txt ＋ RULE.v2.txt が正（不変）。
# session-3646-excl.sh（#3646 の専有系列）の hw-lock・30 秒 poller・系列ラベル（CUR）・出力先の存在で中止する骨格を写し、
# 計測本体だけを #3671 用に差し替えた。1 本の切り離しセッションで直列に実行する:
#   (1) diag    診断の正式系列 5 run（cpu_gemm_reuse_window_phases）
#   (2) solo    rp_solo 5 run（cpu_gemm_reuse_window_solo）
#   (3) base    bench_base: gemm cpu の fresh・reuse・reuse --phases を N 別に 5 run
#   (4) tunables bench_tunables: (3) と同じ構成で GLIBC_TUNABLES=（固定リテラル）
#   (5) headpatch bench_headpatch: HEAD の path ビルドで (3) と同じ構成
#   (6) minflt  補助 1 run（/usr/bin/time -v。判定外）
# ビルドは事前に完了済み（診断テストバイナリは README の `cargo test --release ... --no-run` を事前に実行して
# キャッシュ済みにし、orchestrate.sh 内のビルドを no-op にする。bench バイナリは build_bench_arms.sh）。
# 各ステップの rc を steps.log へ追記し、最後に終端行を書く。
# 値はすべてこのスクリプト内の固定定数で、外部入力から組み立てない。ホスト名・ユーザー名のマスクは回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES GEMM_WINDOW_ONLY_ARM
R=$HOME/work/r3671-excl
T=$R/runner
O=$R/out
BINDIR=$R/bench-bin
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log
GATE=$O/load_gate_bench.log
GPU=$O/gpu_procs.txt
POLL=$O/gpu_poll.log
CUR=$O/current_series
POLL_INTERVAL=30
RES=$T/docs/perf/logs/cpu-gemm-reuse-window-3671
SIZES="256 512 1024 2048 4096"
# RULE.txt 固定定数（bench_tunables。#3646 の補助系列と同じリテラル）
TUNABLES_VALUE="glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824"

# 既存の出力があれば中止（専有系列は新しい作業ディレクトリでだけ取る。上書き・追記をしない）。
if [ -e "$O" ]; then echo "output dir exists: $O" >&2; exit 5; fi
for b in "$BINDIR/bench-fandhe-base" "$BINDIR/bench-fandhe-head"; do
  [ -x "$b" ] || { echo "bench バイナリが無い: $b（build_bench_arms.sh を先に実行）" >&2; exit 6; }
done
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
echo "#3671 $(date -u +%FT%TZ)" >"$LOCK_DIR/owner"

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

# 負荷ゲート未通過は RULE.txt [ゲート・専有・同定] により系列を参考扱いにし判定しない。
abort_gate() { step "gate unpassed: $1 -> abort"; echo "done. rc=gate-unpassed" >>"$LOG"; exit 4; }

# bench 系列 1 本。$1=腕ラベル（base|tunables|headpatch）$2=バイナリ $3...=env 前置き（空でもよい）
bench_series() {
  local label=$1 bin=$2 r n m modes args
  shift 2
  mkdir -p "$O/bench/$label"
  snap "$label:pre"
  echo "$label" >"$CUR"
  for r in 1 2 3 4 5; do
    for n in $SIZES; do
      gate "bench_$label/n${n}/r${r}" || abort_gate "bench_$label/n${n}/r${r}"
      # run の偶奇で fresh と reuse の先後を入れ替える（順序効果の相殺。#3646 ブリッジ v2 と同じ）。
      if [ $((r % 2)) = 1 ]; then modes="fresh reuse phases"; else modes="phases reuse fresh"; fi
      for m in $modes; do
        case "$m" in
          fresh) args=(--mode fresh) ;;
          reuse) args=(--mode reuse) ;;
          phases) args=(--mode reuse --phases) ;;
        esac
        "$@" "$bin" --task gemm --device cpu "${args[@]}" --size "$n" \
          --out "$O/bench/$label/gemm_${m}_n${n}_r${r}.jsonl" >"$O/bench/$label/gemm_${m}_n${n}_r${r}.stdout.log" 2>&1
        echo "$m n=$n r=$r rc=$?" >>"$O/bench/$label/rc.log"
      done
    done
  done
  step "bench $label done"
  echo idle >"$CUR"
  snap "$label:post"
}

cd "$T" || exit 1
step "start tree=$(sed -n 1p "$T/.rev-stamp")"

# ---- (1) 診断の正式系列 ----
snap "diag:pre"
echo diag >"$CUR"
bash "$RES/orchestrate.sh" gb10 --out "$O/gb10" >"$O/orch-gb10.log" 2>&1
step "orchestrate-diag rc=$?"
echo idle >"$CUR"
snap "diag:post"

# ---- (2) rp_solo ----
snap "solo:pre"
echo solo >"$CUR"
bash "$RES/orchestrate.sh" gb10 --only-arm resident_public --out "$O/gb10-solo" >"$O/orch-solo.log" 2>&1
step "orchestrate-solo rc=$?"
echo idle >"$CUR"
snap "solo:post"

# ---- (3)〜(5) bench 側の腕（--out を必ず明示。既定の results/ へ追記させない）----
bench_series base "$BINDIR/bench-fandhe-base" env
bench_series tunables "$BINDIR/bench-fandhe-base" env "GLIBC_TUNABLES=$TUNABLES_VALUE"
bench_series headpatch "$BINDIR/bench-fandhe-head" env

# ---- (6) 補助 minflt（判定外）----
snap "minflt:pre"
echo minflt >"$CUR"
bash "$RES/orchestrate.sh" gb10 --aux minflt --out "$O/gb10-minflt" >"$O/orch-minflt.log" 2>&1
step "orchestrate-minflt rc=$?"
echo idle >"$CUR"
snap "minflt:post"

step "session body done"
echo "done." >>"$LOG"
