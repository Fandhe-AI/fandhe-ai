#!/usr/bin/env bash
# 役割: イシュー #2997 の Mac mini（Apple M4・<metal-node>）CPU 参考行で「1 腕 × 1 run = 1 プロセス」を実行する。
# 呼び出し文脈: 開発機から `ssh <metal-node> 'zsh -lc "./docs/perf/logs/int8-quant-baseline-2609/orchestrate_mac.sh <arm> <run>"'`
# の形でノード側同期ツリー上で呼ぶ（docs/real-hardware-verification-env.md §7）。規則の正は RULE.txt。
# 変数は ${VAR} 形式（macOS の bash 3.2 は $VAR 直後の全角文字を変数名に取り込むため）。完了マーカーは行頭 "done." 。
set -euo pipefail

ARM="${1:-}"
RUN="${2:-}"
case "${ARM}" in rust-cpu|pytorch-cpu|facade-metal|pytorch-mps|parity-metal) ;; *) echo "bad arm: ${ARM}" >&2; exit 2 ;; esac
case "${RUN}" in 1|2|3|4|5) ;; *) echo "bad run: ${RUN}" >&2; exit 2 ;; esac

ROOT="${ROOT:-${HOME}/work/rust-ai-library-run}"
OUT="${OUT:-${HOME}/work/int8-2997-logs}"  # #2998 は OUT=~/work/int8-2998-logs で呼ぶ
PYVENV="${PYVENV:-${HOME}/work/venv-torch-cpu}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${HOME}/work/target-fandhe-ai}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
mkdir -p "${OUT}"
LOG="${OUT}/${ARM}-run${RUN}.log"
JSON="${OUT}/${ARM}-run${RUN}.json"
if [ -e "${LOG}" ]; then echo "既存ログがあるため停止: ${LOG}" >&2; exit 3; fi

# 開始前ゲート: load1 < 2.0 を最大 30 分待つ。満たさなければ「参考扱い」と記録して実行する（RULE.txt DISCARD_MAC）。
waited=0
while :; do
  load1="$(sysctl -n vm.loadavg | awk '{print $2}')"
  if awk -v l="${load1}" 'BEGIN{exit !(l < 2.0)}'; then note="load1_ok"; break; fi
  if [ "${waited}" -ge 1800 ]; then note="REFERENCE_ONLY_load1_high"; break; fi
  sleep 20; waited=$((waited + 20))
done
echo "pre load1=${load1} waited=${waited}s note=${note}" > "${OUT}/load-${ARM}-run${RUN}.txt"

# macOS には coreutils の timeout が無いため perl で外部タイムアウトを掛ける。
# 子を専用プロセスグループ（setpgrp）で起動し、タイムアウト時はグループ全体へ TERM→KILL を送り、
# グループ内の全プロセスが消えたこと（kill 0 が ESRCH）を確認してから戻る。
# alarm+exec では cargo だけが終了し cargo が起動した計測バイナリが残って後続 run と並走する（RULE.txt DISCARD_MAC に反する）ため。
# INT/HUP/TERM 受信時も同じ停止経路を通り、終了コードは 128+シグナル番号。
# 終了コード: タイムアウト 124（coreutils timeout 準拠）／通常終了後にグループへ残存があり KILL した場合 125／停止確認不能 126。
with_timeout() {
  local secs="$1"; shift
  perl -e '
    my $secs = shift @ARGV;
    my $pid = fork();
    die "fork: $!" unless defined $pid;
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV or die "exec: $!"; }
    setpgrp($pid, $pid);
    my $timedout = 0;
    my $sig = 0;
    my $st;
    # SSH 切断（HUP）・中断（INT）・TERM を受けても子グループを孤児化させない。
    # ハンドラは waitpid を中断して下のグループ停止経路（TERM→KILL→消滅確認）へ合流させる。
    my %signo = (HUP => 1, INT => 2, TERM => 15);
    for my $n (keys %signo) { $SIG{$n} = sub { $sig = $signo{$n}; die "signal\n" }; }
    eval {
      local $SIG{ALRM} = sub { die "timeout\n" };
      alarm $secs;
      my $r = waitpid($pid, 0);
      $st = $? if $r == $pid;
      alarm 0;
    };
    alarm 0;
    $timedout = 1 if defined $@ && $@ eq "timeout\n";
    my $leftover = 0;
    if (!defined $st || kill(0, -$pid)) {
      kill "TERM", -$pid;
      for (1 .. 10) { last unless kill(0, -$pid); select(undef, undef, undef, 0.5); }
      if (kill(0, -$pid)) {
        kill "KILL", -$pid;
        for (1 .. 20) { last unless kill(0, -$pid); select(undef, undef, undef, 0.5); }
      }
      $leftover = 1;
    }
    waitpid($pid, 0) unless defined $st;
    if (kill(0, -$pid)) { print STDERR "with_timeout: process group still alive\n"; exit 126; }
    exit(128 + $sig) if $sig;
    exit 124 if $timedout;
    exit 125 if $leftover;
    exit(($st & 127) ? 128 + ($st & 127) : ($st >> 8));
  ' "${secs}" "$@"
}

cd "${ROOT}"
set +e
case "${ARM}" in
  rust-cpu)
    BENCH_TRANSFORMER_REPORT="${JSON}" with_timeout "${TIMEOUT_SECS}" \
      cargo test -p bench-harness --release --test transformer_workload \
      -- transformer_block_forward_bench_cpu --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  facade-metal)
    # イシュー #2998（案 ii）: facade TransformerEncoderLayer の Metal 代替分母。規則は RULE-metal.txt。
    BENCH_TRANSFORMER_METAL_REPORT="${JSON}" with_timeout "${TIMEOUT_SECS}" \
      cargo test -p fandhe-ai --release --test transformer_encoder_metal_denominator \
      -- transformer_encoder_facade_forward_bench_metal_relu --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  parity-metal)
    with_timeout "${TIMEOUT_SECS}" \
      cargo test -p fandhe-ai --release --test compat_sequential_layers_backend_parity \
      -- metal_transformer_encoder_matches_cpu --exact --ignored --nocapture > "${LOG}" 2>&1 ;;
  pytorch-mps)
    with_timeout "${TIMEOUT_SECS}" "${PYVENV}/bin/python" scripts/bench-transformer-pytorch.py \
      --device mps --output "${JSON}" > "${LOG}" 2>&1 ;;
  pytorch-cpu)
    with_timeout "${TIMEOUT_SECS}" "${PYVENV}/bin/python" scripts/bench-transformer-pytorch.py \
      --device cpu --output "${JSON}" > "${LOG}" 2>&1 ;;
esac
status=$?
set -e
# 終了コードを保存し、非ゼロ（失敗・タイムアウト・残存プロセス検出 125 等）の run は成果物を退避する
# （通常名の JSON が残って aggregate.py に正常 run として集計されるのを防ぐ。取り直し時の既存ログ停止も回避）。
echo "${status}" > "${OUT}/${ARM}-run${RUN}.status"
echo "post load1=$(sysctl -n vm.loadavg | awk '{print $2}')" >> "${OUT}/load-${ARM}-run${RUN}.txt"
if [ "${status}" -ne 0 ]; then
  stamp="$(date +%s)"
  for f in "${LOG}" "${JSON}" "${OUT}/${ARM}-run${RUN}.status"; do
    if [ -e "${f}" ]; then mv "${f}" "${f}.discarded-${stamp}"; fi
  done
  echo "done. arm=${ARM} run=${RUN} status=${status} FAILED (discarded)"
  exit "${status}"
fi
echo "done. arm=${ARM} run=${RUN} status=${status}"
exit "${status}"
