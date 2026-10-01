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
CARGO_FLAGS="${CARGO_FLAGS:-}"
# shellcheck disable=SC2206  # 意図的な単語分割（空白区切りのフラグ列）
CARGO_FLAG_ARR=($CARGO_FLAGS)

EXEC_TIMEOUT=120
COMPILE_TIMEOUT=900
LEGACY_TIMEOUT=180
DUMP_TIMEOUT=60
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
    if [[ "$p" == "exec_matrix" ]]; then
      for probe in "${PROBES[@]}"; do
        for target in "${EXEC_TARGETS[@]}"; do
          echo "exec $probe $target"
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
    "$REPO_ROOT"/*)
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
  echo "dry-run: timeout 秒 exec=$EXEC_TIMEOUT compile=$COMPILE_TIMEOUT legacy=$LEGACY_TIMEOUT dump=$DUMP_TIMEOUT"
  echo "dry-run: cargo test ${TEST_PKG[*]} --no-run（sm121_isa_probe_compile・sm121_isa_probe_exec_real_device・setmaxnreg_probe_* ×4）"
  expand_processes | while IFS= read -r line; do
    echo "dry-run: process $line"
  done
  echo "dry-run: OK"
  exit 0
fi

# ---------------------------------------------------------------- 実行

cd "$REPO_ROOT"
HOST_NAME="$(hostname)"
# 収録時マスク（ホスト名 → masked、$HOME と /home/<name> → <home>）。aggregate.py は /home/<name> の
# 残存を検出して拒否する。
mask() {
  sed -E -e "s|${HOME}|<home>|g" -e "s|/home/[A-Za-z0-9_.-]+|<home>|g" -e "s|${HOST_NAME}|masked|g"
}

mkdir -p "$LOG_DIR/exec"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# provenance（git HEAD と clean tree。出力先を作る前＝生成物が dirty を作る前の状態を記録する）。
# GB10 ノードの作業ツリーは rsync 転送で `.git` を持たない（docs/real-hardware-verification-env.md §3）。
# その場合は転送元が書いた `.rev-stamp`（1 行目: HEAD の 40 桁 16 進・2 行目: `dirty=<件数>`。
# 作り方は docs/cuda-sm121-isa-probe.md §6.1）から読む。どちらも得られなければ停止する
# （黙って続行すると G0 が常に不成立になり、正式な集計を作れない）。
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
  printf 'SM121_PROBE_PROC {"v":1,"kind":"proc","phase":"%s","probe":"%s","target":"%s","exit":%d}\n' \
    "$phase" "$probe" "$target" "$rc" >>"$out"
  rm -f "$raw"
}

echo "== ビルド（テストバイナリ） =="
cargo test "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --no-run \
  --test sm121_isa_probe_compile --test sm121_isa_probe_exec_real_device \
  --test setmaxnreg_probe_dec_base_real_device --test setmaxnreg_probe_dec_accel_real_device \
  --test setmaxnreg_probe_incdec_base_real_device --test setmaxnreg_probe_incdec_accel_real_device
cargo build "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --example device_attributes_dump

while IFS= read -r proc; do
  case "$proc" in
    env_info)
      : # env_info.txt は上で記録済み（シェルのみ）
      ;;
    device_attributes_dump)
      echo "== device_attributes_dump =="
      run_proc "$LOG_DIR/device_attributes_dump.log" dump - - "$DUMP_TIMEOUT" \
        cargo run "${CARGO_FLAG_ARR[@]}" "${TEST_PKG[@]}" --example device_attributes_dump
      ;;
    compile)
      echo "== compile（S1／S2／home／hopper） =="
      run_proc "$LOG_DIR/compile.log" compile - - "$COMPILE_TIMEOUT" \
        env "SM121_PROBE_TARGET_SET=$TARGET_SET" \
        "${CARGO_TEST_PROBES[@]}" --test sm121_isa_probe_compile -- --ignored --nocapture --exact \
        sm121_isa_probe_compile_matrix
      ;;
    exec\ *)
      read -r _ probe target <<<"$proc"
      echo "== exec $probe $target =="
      run_proc "$LOG_DIR/exec/${probe}@${target}.log" exec "$probe" "$target" "$EXEC_TIMEOUT" \
        env "SM121_PROBE_ID=$probe" "SM121_PROBE_TARGET=$target" \
        "${CARGO_TEST_PROBES[@]}" --test sm121_isa_probe_exec_real_device -- --ignored --nocapture --exact \
        sm121_isa_probe_exec_selected
      ;;
    legacy\ *)
      read -r _ name <<<"$proc"
      echo "== legacy $name =="
      run_proc "$LOG_DIR/legacy-${name}.log" legacy "$name" - "$LEGACY_TIMEOUT" \
        "${CARGO_TEST_PROBES[@]}" --test "$name" -- --ignored --nocapture
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
