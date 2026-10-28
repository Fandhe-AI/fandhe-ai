#!/usr/bin/env bash
# イシュー #3646 GB10 専有系列セッション（3 回目の reopen・所有者判断 2026-10-28）。RULE.txt ＋ RULE.v2.txt が正（不変）。
# session-3646.sh（PR #3669。常駐 GPU プロセスのある機体で取った参考記録）と計測本体は同一で、次の 3 点だけ加える:
#   (a) 系列ラベル CUR（current_series）を各ステップで更新する  (b) 30 秒間隔の compute-apps poller をロック取得後に常駐させる
#   (c) 作業ディレクトリ・出力先を専有系列専用（r3646-excl・gb10-excl 相当）にし、既存の出力があれば開始前に中止する
# 専有の合否は excl_check.py が poller ログと境界スナップショットから系列ごとに判定する（1 件でも出た系列は参考記録へ降格）。
# 以下は session-3646.sh の記述を引き継ぐ。
# イシュー #3646 GB10 セッション（RULE.txt ＋ RULE.v2.txt が正）。1 本の切り離しセッションで直列に実行する。
#   (1) 診断 5 run 正式系列 (2) minflt 1 run (3) tunables 5 run (4) taskset 5 run
#   (5) ブリッジ v2: bench-fandhe gemm の fresh 通常行・reuse 通常行・reuse --phases を N 別に 5 run
#   (6) G-CPU-INF: (i) PyTorch CPU 純カーネル＋PyTorch 窓 (ii) fandhe 窓＋層別区間 (iii) BLAS 経路の実行時確認
# ビルドは事前に別スクリプトで完了済み（計測中のビルド並走は禁止）。各ステップの rc を steps.log へ追記し、最後に終端行を書く。
# 実機ロックは外部の mkdir 方式（他イシューの計測と排他。共有ノードのため）。値は RULE の固定定数だけを使い、
# 外部入力から組み立てない。ホスト名・ユーザー名のマスクは回収側で行う。
export PATH=$HOME/.cargo/bin:$PATH
export LC_ALL=C
unset RAYON_NUM_THREADS CARGO_TARGET_DIR GEMM_REUSE_MINFLT GLIBC_TUNABLES
R=$HOME/work/r3646-excl
T=$R/runner
O=$R/out
LOCK_DIR=$HOME/work/.fandhe-hw-lock
LOG=$O/steps.log
GATE=$O/load_gate_bench.log
GPU=$O/gpu_procs.txt
POLL=$O/gpu_poll.log
CUR=$O/current_series
POLL_INTERVAL=30
PY=$HOME/work/.venv-bench/bin/python
BENCH=$T/scripts/bench/framework-compare/target/release/bench-fandhe
DOCS=$T/docs/perf/logs
RES=$DOCS/cpu-gemm-resident-reuse-3646
SIZES="256 512 1024 2048 4096"

# 既存の出力があれば中止（専有系列は新しい作業ディレクトリでだけ取る。上書き・追記をしない）。
if [ -e "$O" ]; then echo "output dir exists: $O" >&2; exit 5; fi
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
echo "#3646 $(date -u +%FT%TZ)" >"$LOCK_DIR/owner"
trap 'rm -rf "$LOCK_DIR"; step "hwlock released"' EXIT
step "hwlock acquired"

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

snap() { # $1=ラベル。常駐 GPU プロセスと該当 PID の ps を記録する（停止・変更はしない）。
  {
    echo "== $1 $(date -u +%FT%TZ)"
    local o r
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

# 負荷ゲート未通過は RULE.txt [ゲート] により系列を参考扱いにし判定しない。以降の計測を中止して終端行へ記録する
# （未通過の系列から通常の判定結果を作らない。判定器も load_gate_bench.log の gate=unpassed を拒否する）。
abort_gate() { step "gate unpassed: $1 -> abort"; echo "done. rc=gate-unpassed" >>"$LOG"; exit 4; }

cd "$T" || exit 1
step "start tree=$(sed -n 1p "$T/.rev-stamp")"
{ "$PY" --version; "$PY" -c 'import torch,numpy; print(torch.__version__, numpy.__version__); print(torch.get_num_threads())'; } >"$O/versions.txt" 2>&1

# ---- (1)〜(4) 診断系列 ----
snap "main:pre"
echo main >"$CUR"
bash "$RES/orchestrate.sh" gb10 --out "$O/gb10" >"$O/orch-gb10.log" 2>&1
step "orchestrate-main rc=$?"
echo idle >"$CUR"
snap "main:post"
snap "minflt:pre"
echo minflt >"$CUR"
bash "$RES/orchestrate.sh" gb10 --aux minflt --out "$O/gb10-minflt" >"$O/orch-minflt.log" 2>&1
step "orchestrate-minflt rc=$?"
echo idle >"$CUR"
snap "minflt:post"
snap "tunables:pre"
echo tunables >"$CUR"
bash "$RES/orchestrate.sh" gb10 --aux tunables --out "$O/gb10-tunables" >"$O/orch-tunables.log" 2>&1
step "orchestrate-tunables rc=$?"
echo idle >"$CUR"
snap "tunables:post"
snap "taskset:pre"
echo taskset >"$CUR"
bash "$RES/orchestrate.sh" gb10 --aux taskset --out "$O/gb10-taskset" >"$O/orch-taskset.log" 2>&1
step "orchestrate-taskset rc=$?"
echo idle >"$CUR"
snap "taskset:post"

# ---- (5) ブリッジ v2（bench-fandhe。--out を必ず明示。既定の results/ へ追記させない）----
mkdir -p "$O/gb10-bridge"
snap "bridge:pre"
echo bridge >"$CUR"
for r in 1 2 3 4 5; do
  for n in $SIZES; do
    gate "bridge/gemm/n${n}/r${r}" || abort_gate "bridge/gemm/n${n}/r${r}"
    # run の偶奇で fresh と reuse の先後を入れ替える（順序効果の相殺）。
    if [ $((r % 2)) = 1 ]; then modes="fresh reuse phases"; else modes="phases reuse fresh"; fi
    for m in $modes; do
      case "$m" in
        fresh) args=(--mode fresh) ;;
        reuse) args=(--mode reuse) ;;
        phases) args=(--mode reuse --phases) ;;
      esac
      "$BENCH" --task gemm --device cpu "${args[@]}" --size "$n" \
        --out "$O/gb10-bridge/gemm_${m}_n${n}_r${r}.jsonl" >"$O/gb10-bridge/gemm_${m}_n${n}_r${r}.stdout.log" 2>&1
      echo "bridge gemm $m n=$n r=$r rc=$?" >>"$O/gb10-bridge/rc.log"
    done
  done
done
step "bridge done"
echo idle >"$CUR"
snap "bridge:post"

# ---- (6) G-CPU-INF ----
mkdir -p "$O/gb10-gcpuinf" "$O/pybench"
cp "$DOCS/lowlayer-diagnosis-2026-09-12/bench_py.py" "$O/pybench/bench_py.py"
sha256sum "$O/pybench/bench_py.py" >"$O/gb10-gcpuinf/bench_py.sha256"
snap "gcpuinf:pre"
echo gcpuinf >"$CUR"
# (i) PyTorch 純カーネル（MLP 2 形状を含む 6 形状・base 腕）と PyTorch の infer 窓（bench_py と同一プロトコル）
for r in 1 2 3 4 5; do
  gate "gcpuinf/torch-kernel/r${r}" || abort_gate "gcpuinf/torch-kernel/r${r}"
  "$PY" "$T/scripts/bench/framework-compare/opp_kernel_torch_diag.py" measure --bench-py "$O/pybench/bench_py.py" \
    --device cpu --shapes mlp --arm base --run "$r" --out "$O/gb10-gcpuinf/torch_kernel_r${r}.jsonl" \
    >"$O/gb10-gcpuinf/torch_kernel_r${r}.log" 2>&1
  echo "torch-kernel r=$r rc=$?" >>"$O/gb10-gcpuinf/rc.log"
  gate "gcpuinf/torch-infer-window/r${r}" || abort_gate "gcpuinf/torch-infer-window/r${r}"
  "$PY" "$O/pybench/bench_py.py" --framework pytorch --task infer --device cpu \
    --out "$O/gb10-gcpuinf/torch_infer_r${r}.jsonl" >"$O/gb10-gcpuinf/torch_infer_r${r}.log" 2>&1
  echo "torch-infer-window r=$r rc=$?" >>"$O/gb10-gcpuinf/rc.log"
done
# (ii) fandhe の infer 窓（fresh 通常行・reuse 通常行・reuse --phases）
for r in 1 2 3 4 5; do
  gate "gcpuinf/fandhe-infer/r${r}" || abort_gate "gcpuinf/fandhe-infer/r${r}"
  if [ $((r % 2)) = 1 ]; then modes="fresh reuse phases"; else modes="phases reuse fresh"; fi
  for m in $modes; do
    case "$m" in
      fresh) args=(--mode fresh) ;;
      reuse) args=(--mode reuse) ;;
      phases) args=(--mode reuse --phases) ;;
    esac
    "$BENCH" --task infer --device cpu "${args[@]}" \
      --out "$O/gb10-gcpuinf/fandhe_infer_${m}_r${r}.jsonl" >"$O/gb10-gcpuinf/fandhe_infer_${m}_r${r}.stdout.log" 2>&1
    echo "fandhe-infer $m r=$r rc=$?" >>"$O/gb10-gcpuinf/rc.log"
  done
done
# (ii') 層別区間（#2105 診断。別系列として 5 run）
bash "$DOCS/cpu-predict-resident-fixedcost-2105/orchestrate.sh" gb10 --out "$O/gb10-gcpuinf/fixedcost-2105" >"$O/orch-2105.log" 2>&1
step "orchestrate-2105 rc=$?"
# (iii) PyTorch CPU BLAS 経路の実行時確認（観測のみ。perf が拒否されても sudo は使わない）
blas_rc=0
{
  echo "== torch.__config__.show()"
  "$PY" -c 'import torch; print(torch.__config__.show())' || blas_rc=1
  echo "== LD_DEBUG=bindings: mm 実行時に解決された BLAS 系シンボルの解決先"
  LD_DEBUG=bindings "$PY" -c 'import torch; a=torch.ones(64,784); b=torch.ones(784,256); c=a@b; d=torch.mm(torch.ones(64,256), torch.ones(256,10)); print(float(c.sum()), float(d.sum()))' 2>&1 \
    | grep -i -E 'gemm|blas|dnnl|mkldnn|acl|arm_compute' | head -n 80
  echo "== ldd libtorch_cpu.so"
  ldd "$("$PY" -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib","libtorch_cpu.so"))')" 2>&1 | grep -i -E 'blas|mkl|dnnl|arm_compute|acl' || echo "(blas 系の NEEDED なし)"
  echo "== perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
} >"$O/gb10-gcpuinf/blas_path.log" 2>&1
step "blas-path rc=$blas_rc"
( cd "$O/gb10-gcpuinf" && perf record -o "$O/gb10-gcpuinf/perf.data" -- "$PY" -c 'import torch; a=torch.ones(64,784); b=torch.ones(784,256)
for _ in range(2000): a@b' >"$O/gb10-gcpuinf/perf_record.log" 2>&1 )
echo "perf-record rc=$?" >>"$O/gb10-gcpuinf/rc.log"
rm -f "$O/gb10-gcpuinf/perf.data"
echo idle >"$CUR"
snap "gcpuinf:post"
step "session body done"
echo "done." >>"$LOG"
