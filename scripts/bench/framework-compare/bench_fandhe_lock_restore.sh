# bench_fandhe_lock_restore.sh
#
# 役割: `scripts/bench/framework-compare/` 配下で `[patch.crates-io.
# fandhe-ai]`（`GEMM_GATE_PATCH_FACADE_PATH` 等。参考系列 HEAD ソース計測
# 用の任意指定）を併用する `cargo build` を実行するスクリプトが source
# する共有ヘルパー。`cargo build` はこの invocation-only patch を解決する
# 過程でカレントディレクトリの `Cargo.lock`（本 workspace の承認済みピン
# 固定。deps-policy.md 第 9 区分）を patch 後の依存グラフへ書き換えて
# しまい、スクリプト終了後もその書き換えが残る（`run_ab_gemm_metal.sh`／
# `run_ab_graph_cuda.sh`／`run_ab_managed_cuda.sh` が individually 対処
# 済みの既知の問題と同型。PR #1452 codex-review P2 指摘
# 〈PRRT_kwDOTuUCJc6gKI3z〉）。
#
# もとは `bench_fandhe_pin_guard.sh`（借用ビュー readout API のピン未収録
# ガード。#1438 で導入）に同居していたが、承認ピンが `fandhe-ai =0.8.0`
# （#1487）へ更新され当該ガードが不要になったのに対し、この Cargo.lock
# 退避・復元機能は path patch を伴う計測（参考系列）が存在する限り引き
# 続き必要であるため、#1487 で本ファイルへ分離した（関数名は変更せず
# 逐語移設）。
#
# 呼び出し元: run_all.sh・run_all_cuda.sh・run_ab_train_cuda.sh（いずれも
# cargo build 実行前・cd 済みの `scripts/bench/framework-compare/`
# ディレクトリ内で `bench_fandhe_setup_lock_restore_trap` を 1 回呼ぶ）。
# `run_ab_gemm_metal.sh`／`run_ab_graph_cuda.sh`／`run_ab_managed_cuda.sh`
# は同型の trap を自前で持つため本ファイルの呼び出しは不要。

bench_fandhe_sha256_of() {
  local f=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# Cargo.lock を退避する。退避に失敗（cp 失敗・sha256 不一致による不完全な
# 退避の疑い）した場合は fail-closed で exit 1 する（`run_ab_gemm_metal.sh`
# と同一方針。バックアップが信頼できないまま計測を進めると、path patch で
# 書き換わった Cargo.lock を最終状態のまま残しうる）。
bench_fandhe_backup_lock() {
  BENCH_FANDHE_LOCK_BACKUP="$(mktemp)"
  if ! cp Cargo.lock "$BENCH_FANDHE_LOCK_BACKUP"; then
    echo "error: cp Cargo.lock '$BENCH_FANDHE_LOCK_BACKUP' (backup) に失敗した" >&2
    rm -f "$BENCH_FANDHE_LOCK_BACKUP"
    exit 1
  fi
  if [[ "$(bench_fandhe_sha256_of Cargo.lock)" != "$(bench_fandhe_sha256_of "$BENCH_FANDHE_LOCK_BACKUP")" ]]; then
    echo "error: Cargo.lock のバックアップ内容が元ファイルと一致しない（不完全な退避の可能性）" >&2
    rm -f "$BENCH_FANDHE_LOCK_BACKUP"
    exit 1
  fi
}

# 退避した Cargo.lock を復元する。復元（cp）自体に失敗した場合はバックアップを
# 保持したまま呼び出し元へ失敗を返す（削除すると再試行手段が失われるため）。
bench_fandhe_restore_lock() {
  if [[ -z "${BENCH_FANDHE_LOCK_BACKUP:-}" ]]; then
    return 0
  fi
  if ! cp "$BENCH_FANDHE_LOCK_BACKUP" Cargo.lock; then
    echo "error: Cargo.lock の復元に失敗した。バックアップを保持する: $BENCH_FANDHE_LOCK_BACKUP" >&2
    return 1
  fi
  rm -f "$BENCH_FANDHE_LOCK_BACKUP"
}

# EXIT trap ハンドラ本体。`run_ab_gemm_metal.sh` の `restore_lock_trap` と
# 同型: trap ハンドラ内の `return` はスクリプト全体の終了コードへ伝播しない
# ため、元の終了コード（`$?`）を保持しつつ復元失敗時のみ非 0 へ強制する。
bench_fandhe_restore_lock_trap() {
  local code=$?
  if ! bench_fandhe_restore_lock; then
    if [[ "$code" -eq 0 ]]; then
      code=1
    fi
  fi
  exit "$code"
}

# 呼び出し元（cargo build 実行前・cd 済みディレクトリ内）から 1 回呼ぶ。
# 退避 + EXIT trap 登録をまとめて行う。
bench_fandhe_setup_lock_restore_trap() {
  bench_fandhe_backup_lock
  trap bench_fandhe_restore_lock_trap EXIT
}
