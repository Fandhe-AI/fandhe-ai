#!/usr/bin/env bash
# イシュー #3188（再オープン分）: bench-fandhe（framework-compare）の pre／head 前後比較（GB10・CPU）。
# 判定規則・フォールバックは同ディレクトリの RULE.txt（実測前に固定）が正。参考値のみで判定ゲートは置かない。
# 使い方: bench_prepost.sh <out_dir> [a|own]
#   a   : ハーネス = 計測ツリー A（$HOME/work/fandhe-3188-reopen-A）の framework-compare を pre／head 両方に使う（既定）
#   own : ハーネス = 各ツリー（pre／head）自身の framework-compare（RULE.txt のフォールバック。a で pre が
#         ビルドできない場合のみ）
# 前提: $HOME/work/fandhe-3188-{reopen-A,pre,head} に git archive 展開済み（.git なし）で .rev-stamp がある。
# ラベル・期待 SHA は固定表から引く（外部入力からパスを組み立てない）。
# 5 round（奇数: pre→head、偶数: head→pre）x 6 セルを独立プロセスで実行し、round の前に負荷ゲート
# （load1 < 1.0・30 秒間隔・最大 30 分。未通過 round は参考扱い）を置く。
set -euo pipefail

usage() { echo "usage: $0 <out_dir> [a|own]" >&2; exit 2; }
[ $# -ge 1 ] && [ $# -le 2 ] || usage
mode="${2:-a}"
case "$mode" in a | own) ;; *) usage ;; esac

export PATH="$HOME/.cargo/bin:$PATH"
work="$HOME/work"
declare_sha() {
  case "$1" in
    reopen-A) ;; # A の SHA は RULE.txt に記録（.rev-stamp と突合は呼び出し側 verify で行う）
    pre) echo "17c7767b588a8c4c52eccccc39ba880519962d53" ;;
    head) echo "4f6716f638a6997559847480ddf93cf6924a96e3" ;;
    *) return 1 ;;
  esac
}

out="$1"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

mask() { sed -e "s|${HOME}|<home>|g" -e "s|$(hostname)|masked|g"; }

harness_dir() { # harness_dir <label>
  if [ "$mode" = a ]; then echo "$work/fandhe-3188-reopen-A/scripts/bench/framework-compare"
  else echo "$work/fandhe-3188-$1/scripts/bench/framework-compare"; fi
}
target_dir() { echo "$work/target-fc-3188-$mode-$1"; }

fingerprint() { # fingerprint <tree>
  (cd "$1" && find Cargo.toml Cargo.lock rust-toolchain.toml crates scripts/bench/framework-compare \
    -type f ! -path '*/target/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
}

for label in pre head; do
  tree="$work/fandhe-3188-$label"
  [ "$(cat "$tree/.rev-stamp")" = "$(declare_sha "$label")" ] || { echo "$label: .rev-stamp 不一致" >&2; exit 1; }
  mkdir -p "$out/$label"
  fingerprint "$tree" >"$out/$label/tree_sha256_before.txt"
  cp "$tree/.rev-stamp" "$out/$label/.rev-stamp"
done
cp "$work/fandhe-3188-reopen-A/.rev-stamp" "$out/harness-A.rev-stamp"
fingerprint "$work/fandhe-3188-reopen-A" >"$out/harness-A.tree_sha256_before.txt"

# ビルド（計測前に 2 本とも済ませる。計測中はビルドしない）。Cargo.lock は共有ヘルパーの trap で復元する。
for label in pre head; do
  h="$(harness_dir "$label")"
  facade="$work/fandhe-3188-$label/crates/facade"
  grep -q '^name = "fandhe-ai"$' "$facade/Cargo.toml" || { echo "$label: facade の name 不一致" >&2; exit 1; }
  if [[ "$facade" == *'"'* || "$facade" == *'\'* ]]; then echo "不正なパス" >&2; exit 1; fi
  (
    cd "$h"
    # shellcheck disable=SC1091
    source ./bench_fandhe_lock_restore.sh
    bench_fandhe_setup_lock_restore_trap
    export CARGO_TARGET_DIR
    CARGO_TARGET_DIR="$(target_dir "$label")"
    cargo build --release -p bench-fandhe --config "patch.crates-io.fandhe-ai.path=\"$facade\"" \
      2>&1 | tail -n 3
    cargo tree -p bench-fandhe --depth 1 --config "patch.crates-io.fandhe-ai.path=\"$facade\"" \
      2>&1 | mask >"$out/$label/cargo_tree.txt"
  )
  grep -q "crates/facade" "$out/$label/cargo_tree.txt" || { echo "$label: path patch が解決されていない" >&2; exit 1; }
  echo "$mode" >"$out/harness_mode.txt"
done

load1() { uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{print $1}'; }
gate() { # gate <label>
  local waited=0 l
  while :; do
    l="$(load1)"
    echo "$1 load1=$l waited=${waited}s" >>"$out/load_gate.log"
    if [[ "$l" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v l="$l" 'BEGIN{exit !(l < 1.0)}'; then echo pass; return; fi
    if [ "$waited" -ge 1800 ]; then echo unpassed; return; fi
    sleep 30
    waited=$((waited + 30))
  done
}

{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -srm)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Architecture' || true
  echo "rustc: $(rustc --version)"
  echo "harness_mode: $mode"
} 2>&1 | mask >"$out/env_info.txt"

cells="train_reuse_total train_reuse_phases infer_reuse_total infer_reuse_phases infer_fresh_total infer_fresh_phases"
cell_args() { # cell_args <cell> -> task mode flag
  case "$1" in
    train_reuse_total) echo "train reuse" ;;
    train_reuse_phases) echo "train reuse --phases" ;;
    infer_reuse_total) echo "infer reuse" ;;
    infer_reuse_phases) echo "infer reuse --phases" ;;
    infer_fresh_total) echo "infer fresh" ;;
    infer_fresh_phases) echo "infer fresh --phases" ;;
    *) return 1 ;;
  esac
}

: >"$out/load_gate_status.txt"
failed=0
for r in 1 2 3 4 5; do
  status="$(gate "round$r")"
  echo "round$r gate=$status" >>"$out/load_gate_status.txt"
  if [ $((r % 2)) -eq 1 ]; then order="pre head"; else order="head pre"; fi
  for label in $order; do
    d="$out/$label/round$r"
    mkdir -p "$d"
    for c in $cells; do
      # shellcheck disable=SC2046
      set -- $(cell_args "$c")
      args=(--task "$1" --device cpu --size 64 --mode "$2")
      [ "${3:-}" = "--phases" ] && args+=(--phases)
      if [ -e "$d/$c.jsonl" ]; then echo "既存 $d/$c.jsonl（差し替え禁止）" >&2; exit 1; fi
      if ! "$(target_dir "$label")/release/bench-fandhe" "${args[@]}" --out "$d/$c.jsonl" 2> >(mask >"$d/$c.err"); then
        echo "$label round$r $c: 失敗" >>"$out/failures.log"
        failed=$((failed + 1))
      fi
    done
  done
done

for label in pre head; do
  fingerprint "$work/fandhe-3188-$label" >"$out/$label/tree_sha256_after.txt"
done
fingerprint "$work/fandhe-3188-reopen-A" >"$out/harness-A.tree_sha256_after.txt"
echo "完了: $out（失敗 $failed 件。次: python3 -I aggregate_bench_prepost.py $out）"
[ "$failed" -eq 0 ]
