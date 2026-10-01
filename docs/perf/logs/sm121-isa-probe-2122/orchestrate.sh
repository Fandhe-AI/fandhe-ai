#!/usr/bin/env bash
# イシュー #2122: sm_121 ISA プローブの実測オーケストレーション（GB10 実機）。
# 判定規則は同ディレクトリの RULE.txt（実測前に固定）。構成は #2109
# （`../cuda-gemm-small-launch-cost-2109/orchestrate.sh`）と同型で、次を踏襲する:
#   - set -euo pipefail・LC_ALL=C・noclobber・既存ログがあれば開始前に停止（上書き禁止）
#   - パスは最初の `cd` より前に絶対パスへ正規化
#   - 収録時にホスト名を masked、$HOME と `/home/<name>` を <home> へ置換
#   - `--dry-run`（実機・cargo・出力先の作成なし）
#
# 起動するプロセスの一覧は RULE.txt の `PROCESS:`／`PROBE:`／`TARGET:`／`DEVTARGET:` 行から
# 導出する（本ファイルに一覧を持たない）。`--dry-run` が `dry-run: process <名前>` の形で
# 展開した一覧を出し、aggregate.py の self-test が RULE.txt と突き合わせる。
#
# 使い方:
#   ./orchestrate.sh                      # GB10 実機（正式 target。LOG_DIR 既定は本ディレクトリ）
#   LOG_DIR=/path/to/new-empty-dir ./orchestrate.sh
#   ./orchestrate.sh --dry-run            # 起動列の表示と事前検査のみ
#   LOG_DIR=/scratch/dir ./orchestrate.sh --dev-smoke   # 開発機スモーク（compute_86。LOG_DIR 必須・
#                                                       # リポジトリ外に限る。結果は G0 不成立になり
#                                                       # 全セルが判定不能になる。コミットしない）
#
# 共有 GB10 の注意: 意図的に ILLEGAL_INSTRUCTION 等を起こすプローブを含む。プロセスごとに
# 分離して外部 timeout を付けるが、ノードが空いているときに実行すること。
#
# 環境変数（任意）: CARGO_TARGET_DIR・LD_LIBRARY_PATH（libnvrtc の場所。開発機スモーク用）・
# CARGO_FLAGS（cargo へ渡す追加フラグ。例 `--offline`。空白区切り）。
set -euo pipefail
set -o noclobber
export LC_ALL=C

DRY_RUN=0
DEV_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --dev-smoke) DEV_SMOKE=1 ;;
    *) echo "ERROR: 未知の引数: ${arg}（許容: --dry-run --dev-smoke）" >&2; exit 2 ;;
  esac
done

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../../../.." && pwd)"
RULE="$SELF_DIR/RULE.txt"
# パスは最初の `cd` より前に絶対パスへ正規化する（後段で `cd "$REPO_ROOT"` するため、相対のままだと
# ログが別ディレクトリへ分散する）。ディレクトリ未作成でも解決できるよう先頭が `/` でなければ cwd を前置する。
LOG_DIR_GIVEN="${LOG_DIR:-}"
LOG_DIR="${LOG_DIR:-$SELF_DIR}"
case "$LOG_DIR" in /*) ;; *) LOG_DIR="$PWD/$LOG_DIR" ;; esac
# `..` と symlink を解決して正規化する（リポジトリ外判定を迂回させない）。未作成でも解決できる -m。
LOG_DIR="$(realpath -m -- "$LOG_DIR")"
CARGO_FLAGS="${CARGO_FLAGS:-}"
# shellcheck disable=SC2206  # 意図的な単語分割（空白区切りのフラグ列）
CARGO_FLAG_ARR=($CARGO_FLAGS)

# timeout の秒数は RULE.txt の PROCESS 行の `timeout=<秒>` が正（本ファイルに定数を持たない）。
KILL_AFTER=10

if [[ ! -f "$RULE" ]]; then
  echo "ERROR: RULE.txt が見つからない: $RULE" >&2
  exit 1
fi

# RULE.txt の機械可読行から起動列を導出する（唯一の契約）。
mapfile -t PROBES < <(sed -n 's/^PROBE: \([^ ]*\) .*$/\1/p' "$RULE")
mapfile -t TARGETS < <(sed -n 's/^TARGET: \([^ ]*\) .*$/\1/p' "$RULE")
mapfile -t DEVTARGETS < <(sed -n 's/^DEVTARGET: \([^ ]*\) .*$/\1/p' "$RULE")
mapfile -t PROCESS_LINES < <(sed -n 's/^PROCESS: \(.*\)$/\1/p' "$RULE")
if (( ${#PROBES[@]} == 0 || ${#TARGETS[@]} == 0 || ${#DEVTARGETS[@]} == 0 || ${#PROCESS_LINES[@]} == 0 )); then
  echo "ERROR: RULE.txt から PROBE／TARGET／DEVTARGET／PROCESS 行を読めない" >&2
  exit 1
fi
for p in "${PROCESS_LINES[@]}"; do
  if [[ "$p" != "env_info" && ! "$p" =~ \ timeout=[1-9][0-9]*$ ]]; then
    echo "ERROR: RULE.txt の PROCESS 行に正の timeout=<秒> が無い: $p" >&2
    exit 1
  fi
done
if [[ "$DEV_SMOKE" == "1" ]]; then
  EXEC_TARGETS=("${DEVTARGETS[@]}")
  TARGET_SET=dev
  MODE=dev_smoke
else
  EXEC_TARGETS=("${TARGETS[@]}")
  TARGET_SET=official
  MODE=gb10
fi

# 展開した起動プロセス名の一覧（`exec_matrix` は PROBE × target）。
expand_processes() {
  local p probe target
  for p in "${PROCESS_LINES[@]}"; do
    if [[ "$p" == exec_matrix\ * ]]; then
      for probe in "${PROBES[@]}"; do
        for target in "${EXEC_TARGETS[@]}"; do
          echo "exec $probe $target ${p#exec_matrix }"
        done
      done
    elif [[ "$DEV_SMOKE" == "1" && "$p" == legacy\ * ]]; then
      : # 開発機スモークでは legacy（compute_121 前提の既存プローブ）を回さない（RULE.txt 14）
    else
      echo "$p"
    fi
  done
}

# 開発機スモークの出力先はリポジトリ外の呼び出し側指定ディレクトリに限る（コミット防止）。
if [[ "$DEV_SMOKE" == "1" ]]; then
  if [[ -z "$LOG_DIR_GIVEN" ]]; then
    echo "ERROR: --dev-smoke は LOG_DIR（リポジトリ外の scratch ディレクトリ）の明示指定が必須" >&2
    exit 2
  fi
  case "$LOG_DIR/" in
    "$(realpath -m -- "$REPO_ROOT")"/*)
      echo "ERROR: --dev-smoke の出力先 $LOG_DIR はリポジトリ内（コミット防止のためリポジトリ外のみ許容）" >&2
      exit 2 ;;
  esac
fi

# 既存の生成物ログが 1 つでもあれば開始前に停止する（上書き・追加起動をしない）。
shopt -s nullglob
existing=()
for f in env_info.txt compile.log device_attributes_dump.log; do
  [[ -e "$LOG_DIR/$f" ]] && existing+=("$f")
done
for f in "$LOG_DIR"/exec/*.log "$LOG_DIR"/legacy-*.log; do
  existing+=("${f#"$LOG_DIR"/}")
done
shopt -u nullglob
if (( ${#existing[@]} > 0 )); then
  echo "ERROR: 出力先 $LOG_DIR に既存のログがある。上書きは禁止（RULE.txt 14）: ${existing[*]}" >&2
  echo "ERROR: 再計測は空の別ディレクトリを LOG_DIR に指定すること" >&2
  exit 1
fi

TEST_PKG=(-p fandhe-ai-backend-cuda --all-features)
CARGO_TEST_PROBES=(cargo test "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry-run: LOG_DIR=$LOG_DIR"
  echo "dry-run: REPO_ROOT=$REPO_ROOT"
  echo "dry-run: mode=$MODE targets=${EXEC_TARGETS[*]}"
  echo "dry-run: cargo test ${TEST_PKG[*]} --no-run（sm121_isa_probe_compile・sm121_isa_probe_exec_real_device・setmaxnreg_probe_* ×4・tma_probe_real_device）"
  expand_processes | while IFS= read -r line; do
    echo "dry-run: process $line"
  done
  echo "dry-run: OK"
  exit 0
fi

# ---------------------------------------------------------------- 実行

cd "$REPO_ROOT"

# provenance（git HEAD と clean tree）。出力先・作業ディレクトリを作る前に、プロセスを 1 つも
# 起動しない段階で確定する。
# GB10 ノードの作業ツリーは rsync 転送で `.git` を持たない（docs/real-hardware-verification-env.md §3）。
# その場合は転送元が書いた `.rev-stamp`（1 行目: HEAD の 40 桁 16 進・2 行目: `dirty=<件数>`。
# 作り方は docs/cuda-sm121-isa-probe.md §6.1）から読む。どちらも得られなければ停止する。
GIT_SOURCE=none
GIT_HEAD=unknown
GIT_CLEAN=0
if [[ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || true)" == "$REPO_ROOT" ]]; then
  GIT_SOURCE=git
  GIT_HEAD="$(git rev-parse HEAD)"
  if [[ -z "$(git status --porcelain --untracked-files=normal)" ]]; then
    GIT_CLEAN=1
  fi
elif [[ -f "$REPO_ROOT/.rev-stamp" ]]; then
  GIT_SOURCE=rev-stamp
  GIT_HEAD="$(sed -n 1p "$REPO_ROOT/.rev-stamp")"
  if [[ "$(sed -n 2p "$REPO_ROOT/.rev-stamp")" == "dirty=0" ]]; then
    GIT_CLEAN=1
  fi
else
  echo "ERROR: git 作業ツリーでも .rev-stamp でもない。転送元の HEAD を .rev-stamp に書いて転送すること" >&2
  echo "ERROR: （docs/cuda-sm121-isa-probe.md §6.1）" >&2
  exit 1
fi
if [[ ! "$GIT_HEAD" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: HEAD が 40 桁 16 進でない（$GIT_SOURCE）: $GIT_HEAD" >&2
  exit 1
fi
# 正式実行は clean な provenance を必須とする（dirty なら G0 が不成立になり結果が使えないため、
# 実測を始める前に止める）。submodule ポインタの変更（`M docs/spec` 等）も dirty に数える。
if [[ "$MODE" == "gb10" && "$GIT_CLEAN" != "1" ]]; then
  echo "ERROR: 作業ツリーが clean でない（git_source=$GIT_SOURCE）。正式実行は clean な状態からのみ許容" >&2
  echo "ERROR: .rev-stamp の場合は 2 行目 dirty=0 が必要（docs/cuda-sm121-isa-probe.md §6.1）" >&2
  exit 1
fi

HOST_NAME="$(hostname)"
# ホスト名のマスクは JSON の構造を壊さないよう、SM121_PROBE_ 行（JSON）には素置換しない。
# ホスト名が短い・JSON のキー／語彙と部分一致する場合は、素置換が JSON を壊すか漏れを見逃すので
# fail-closed で停止する。JSON 行にホスト名が残っていたら run_proc が停止する。
if (( ${#HOST_NAME} < 4 )); then
  echo "ERROR: ホスト名が短すぎる（4 文字未満）ためマスクの安全性を確認できない" >&2
  exit 1
fi
JSON_TOKENS=(
  "v" "kind" "cell" "env" "done" "proc" "phase" "probe" "target" "arch" "stage" "status" "code" "detail" "device" "cc" "nvrtc" "exit" "cells" "mismatch" "compile" "exec" "legacy" "dump" "ok" "error" "rejected" "timeout" "unavailable" "true" "false" "compute" "sm" "mode" "git_head" "git_clean" "git_source" "start_utc" "end_utc" "uname" "gpu_at_start" "load1_at_start" "nvrtc_ptx" "nvrtc_cubin" "module_load" "launch" "sync" "verify" "ctl" "home" "hopper"
)
for t in "${JSON_TOKENS[@]}"; do
  if [[ "$t" == *"$HOST_NAME"* || "$HOST_NAME" == *"$t"* && ${#t} -ge 4 ]]; then
    echo "ERROR: ホスト名 $HOST_NAME が JSON のキー・語彙 $t と部分一致する。マスクが JSON を壊しうるため停止" >&2
    exit 1
  fi
done
sed_escape() { printf '%s' "$1" | sed 's/[][\.*^$+?(){}|\/]/\\&/g'; }
HOST_RE="$(sed_escape "$HOST_NAME")"
HOME_RE="$(sed_escape "$HOME")"
# 収録時マスク（$HOME と /home/<name> → <home>。ホスト名 → masked は非 JSON 行のみ）。aggregate.py は
# /home/<name> の残存を検出して拒否する（ホスト名の残存は検出しない。run_proc が JSON 行を検査する）。
mask() {
  sed -E -e "s|${HOME_RE}|<home>|g" -e "s|/home/[A-Za-z0-9_.-]+|<home>|g" \
    -e "/^SM121_PROBE_/!s|${HOST_RE}|masked|g"
}

mkdir -p "$LOG_DIR/exec"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

{
  echo "mode=$MODE"
  echo "git_head=$GIT_HEAD"
  echo "git_clean=$GIT_CLEAN"
  echo "git_source=$GIT_SOURCE"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname=$(uname -srm)"
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv,noheader 2>/dev/null \
    | head -n1 | sed 's/^/gpu_at_start=/' || echo "gpu_at_start=unavailable"
  echo "load1_at_start=$(cut -d' ' -f1 /proc/loadavg)"
} 2>&1 | mask >"$LOG_DIR/env_info.txt"

# 1 プロセスを外部 timeout 付きで実行し、マスクしたログと proc 記録（終了コード）を残す。
# 引数: <ログパス> <phase> <probe> <target> <timeout 秒> <コマンド...>
run_proc() {
  local out="$1" phase="$2" probe="$3" target="$4" tmo="$5"
  shift 5
  local raw="$WORK/raw.log" rc start
  start=$SECONDS
  set +e
  timeout -k "$KILL_AFTER" "$tmo" "$@" >"$raw" 2>&1
  rc=$?
  set -e
  # KILL まで必要だった timeout は 137 で返る。経過時間が timeout 以上なら 124 へ正規化する。
  if (( rc == 137 && SECONDS - start >= tmo )); then
    rc=124
  fi
  mask <"$raw" >"$out"
  if grep '^SM121_PROBE_' "$out" | grep -F -q -- "$HOST_NAME"; then
    echo "ERROR: JSON 行にホスト名が残っている（$out）。マスクできないため停止" >&2
    exit 1
  fi
  printf 'SM121_PROBE_PROC {"v":1,"kind":"proc","phase":"%s","probe":"%s","target":"%s","exit":%d}\n' \
    "$phase" "$probe" "$target" "$rc" >>"$out"
  rm -f "$raw"
}

echo "== ビルド（テストバイナリ） =="
cargo test "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --no-run \
  --test sm121_isa_probe_compile --test sm121_isa_probe_exec_real_device \
  --test setmaxnreg_probe_dec_base_real_device --test setmaxnreg_probe_dec_accel_real_device \
  --test setmaxnreg_probe_incdec_base_real_device --test setmaxnreg_probe_incdec_accel_real_device \
  --test tma_probe_real_device
cargo build "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --example device_attributes_dump

while IFS= read -r proc; do
  # 末尾の `timeout=<秒>` を取り出す（env_info は timeout を持たない）。秒数は RULE.txt の PROCESS 行が正。
  tmo=""
  desc="$proc"
  if [[ "$proc" =~ ^(.*)\ timeout=([1-9][0-9]*)$ ]]; then
    desc="${BASH_REMATCH[1]}"
    tmo="${BASH_REMATCH[2]}"
  fi
  case "$desc" in
    env_info)
      : # env_info.txt は上で記録済み（シェルのみ）
      ;;
    device_attributes_dump)
      echo "== device_attributes_dump =="
      run_proc "$LOG_DIR/device_attributes_dump.log" dump - - "$tmo" \
        cargo run "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --example device_attributes_dump
      ;;
    compile)
      echo "== compile（S1／S2／home／hopper） =="
      run_proc "$LOG_DIR/compile.log" compile - - "$tmo" \
        env "SM121_PROBE_TARGET_SET=$TARGET_SET" \
        "${CARGO_TEST_PROBES[@]}" --test sm121_isa_probe_compile -- --ignored --nocapture --exact \
        sm121_isa_probe_compile_matrix
      ;;
    exec\ *)
      read -r _ probe target <<<"$desc"
      echo "== exec $probe $target =="
      run_proc "$LOG_DIR/exec/${probe}@${target}.log" exec "$probe" "$target" "$tmo" \
        env "SM121_PROBE_ID=$probe" "SM121_PROBE_TARGET=$target" \
        "${CARGO_TEST_PROBES[@]}" --test sm121_isa_probe_exec_real_device -- --ignored --nocapture --exact \
        sm121_isa_probe_exec_selected
      ;;
    legacy\ *)
      read -r _ name <<<"$desc"
      echo "== legacy $name =="
      # 名前は <テストバイナリ> または <テストバイナリ>@<テスト関数>（後者は 1 テストだけを
      # 別プロセスで実行する。sticky なエラーを他のテストへ波及させない）。
      bin="${name%%@*}"
      test_filter=()
      if [[ "$name" == *@* ]]; then
        test_filter=(--exact "${name#*@}")
      fi
      run_proc "$LOG_DIR/legacy-${name}.log" legacy "$name" - "$tmo" \
        "${CARGO_TEST_PROBES[@]}" --test "$bin" -- --ignored --nocapture "${test_filter[@]+"${test_filter[@]}"}"
      ;;
    *)
      echo "ERROR: 未知の PROCESS 行: $proc" >&2
      exit 1
      ;;
  esac
done < <(expand_processes)

{
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | mask >>"$LOG_DIR/env_info.txt"

echo "done. 次: python3 $SELF_DIR/aggregate.py --log-dir $LOG_DIR"
