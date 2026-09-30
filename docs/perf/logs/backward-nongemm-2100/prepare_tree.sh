#!/usr/bin/env bash
# イシュー #2100: backward 非 GEMM 内訳の診断計装パッチを、指定 ref の
# スクラッチ worktree にだけ当てる準備スクリプト。
#
# 役割: `diag-instrumentation-head.patch`（crates/ 配下のみ。Cargo.lock の
# hunk は含まない）を本体ツリーへ残さず、使い捨ての worktree にだけ適用する
# （計装は本番コードへ残さない。イシュー #2100 スコープ外）。
# 呼び出し元: なし（手動実行。計装あり／なしの 2 本のツリーを作る）。実測側の
# orchestrate_backward_diag.sh（M4 Max・GB10 共用）は本スクリプトが作った
# ツリーを引数で受け取る。実行はリポジトリ内の任意の
# ディレクトリから可能（git worktree はそのリポジトリへ作られる）。
#
# 使い方:
#   bash prepare_tree.sh <instr|plain> <ref> <dest-abs-path>
#     instr: パッチを apply --check → apply する（計装あり）
#     plain: パッチを当てない（計装なし対照。checksum 同一性検査用）
#   例: bash prepare_tree.sh instr origin/main /tmp/wt-diag-instr
# 後片付け: `git worktree remove --force <dest>`。
set -eu

MODE=${1:-}
REF=${2:-}
DEST=${3:-}

# A03 インジェクション対策: 引数は allowlist・絶対パス検査で検証する。
if [[ "$MODE" != "instr" && "$MODE" != "plain" ]]; then
  echo "usage: $0 <instr|plain> <ref> <dest-abs-path>" >&2
  exit 1
fi
if [[ -z "$REF" || ! "$REF" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "error: ref must match [A-Za-z0-9._/-]+ (got: '$REF')" >&2
  exit 1
fi
if [[ -z "$DEST" || "$DEST" != /* ]]; then
  echo "error: dest must be an absolute path (got: '$DEST')" >&2
  exit 1
fi
if [[ -e "$DEST" ]]; then
  echo "error: dest already exists: $DEST" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$HERE/diag-instrumentation-head.patch"
if [[ ! -f "$PATCH" ]]; then
  echo "error: patch not found: $PATCH" >&2
  exit 1
fi

git worktree add --detach "$DEST" "$REF"
if [[ "$MODE" == "instr" ]]; then
  if ! git -C "$DEST" apply --check "$PATCH"; then
    echo "error: patch does not apply cleanly to $REF (要手作業での再移植)" >&2
    git worktree remove --force "$DEST"
    exit 1
  fi
  git -C "$DEST" apply "$PATCH"
fi
echo "prepared: mode=$MODE ref=$REF rev=$(git -C "$DEST" rev-parse --short HEAD) dest=$DEST"
