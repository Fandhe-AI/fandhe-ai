# bench_fandhe_pin_guard.sh
#
# 役割: `scripts/bench/framework-compare/` 配下の各スクリプトから
# `source` される共有ガード。イシュー #1438 で bench-fandhe の借用ビュー
# readout（`Var::host_view`／`Tensor::host_slice`。#1335・#1336・#1337）を
# 既定経路へ取り込み、旧計測専用 cargo feature（既定 OFF・#1337 で導入）を
# 撤去した結果、bench-fandhe は借用ビュー API を常に要求するようになった。
# しかし crates.io 公開版ピン `fandhe-ai =0.7.0`（deps-policy.md 第 9 区分）
# にはこの API が未収録のため、`GEMM_GATE_PATCH_FACADE_PATH` 等の path
# patch を併用しない registry 解決ビルドは構造的にコンパイル不能になった。
#
# 本ガードは「性能値を捏造しない」fail-closed 方針（security.md A08。
# `run_gemm_gate.sh` の manifest 検証群と同型）に従い、ビルドを実際に
# 起動して分かりにくい cargo エラーへ落とすのではなく、ビルド直前に
# 明示エラーで早期停止する。crates.io 次回公開でピンが借用ビュー API を
# 収録した版へ更新されたら、更新 PR が本ファイルと各呼び出し箇所を削除
# する運用とする（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §12.7
# の「ピン更新後に正式判定を確定する」運用と対になる）。
#
# 呼び出し元: run_gemm_gate.sh・run_all.sh・run_all_cuda.sh・
# run_ab_train_cuda.sh・run_ab_gemm_metal.sh（いずれも registry 解決で
# bench-fandhe をビルドしうる箇所の直前）。run_ab_managed_cuda.sh・
# run_ab_graph_cuda.sh は常に path patch を併用するため呼び出し不要。

# ピン `fandhe-ai =0.7.0` に借用ビュー API が未収録である事実を表す定数。
# ピン更新 PR がこのファイルごと削除する（値を推測して次版番号を
# ハードコードしない）。
BENCH_FANDHE_REGISTRY_PIN_LACKS_HOST_VIEW=1

# 引数:
#   $1 - ctx: エラーメッセージに出す呼び出し文脈（スクリプト名等）
#   $2 - patch_path: `GEMM_GATE_PATCH_FACADE_PATH` 等の path patch 値
#        （空文字なら registry 解決を意図した invocation とみなす）
#
# path patch が空かつ定数が 1 の場合、bench-fandhe/Cargo.toml から現在の
# ピン値を読み取って理由とともに stderr へ出し exit 1 する（呼び出し元の
# シェルを終了させる。source される前提のため exit で呼び出し元プロセス
# 自体を止める）。
#   $3 - note（任意）: 既定の「GEMM_GATE_PATCH_FACADE_PATH を指定すること」
#        という案内が誤りになる呼び出し元（例: registry 解決そのものを
#        意図する「before」腕。path patch では本質的に解消できない）向けの
#        差し替え文面。PR #1452 codex-review P1 指摘（PRRT_kwDOTuUCJc6gIbMM）:
#        `run_ab_gemm_metal.sh` の before 腕は bench-fandhe ソース自体
#        （現行 HEAD）が借用ビュー readout API を無条件に要求するため、
#        facade 側だけを path patch しても bench-fandhe のビルドは解決
#        しない（`fandhe_ai_source_desc` の "registry" 検証が後続で
#        必ず失敗する）。この腕に汎用の GEMM_GATE_PATCH_FACADE_PATH 案内を
#        出すと「渡せば解決する」という誤った期待を与えるため、専用の
#        note で正しい対処（ピン更新を待つ、または #1438 以前のコミットを
#        別 worktree にチェックアウトして実行する）を案内する。
bench_fandhe_require_facade_patch() {
  local ctx="$1"
  local patch_path="${2:-}"
  local note="${3:-}"

  if [[ "$BENCH_FANDHE_REGISTRY_PIN_LACKS_HOST_VIEW" != "1" ]]; then
    return 0
  fi
  if [[ -n "$patch_path" ]]; then
    return 0
  fi

  local pin
  pin=$(grep -o 'fandhe-ai = "=[^"]*"' "$(dirname "${BASH_SOURCE[0]}")/bench-fandhe/Cargo.toml" \
    | head -1 | sed -E 's/.*"=(.*)"/\1/')
  echo "ERROR: [${ctx}] crates.io ピン 'fandhe-ai =${pin:-<不明>}' には" >&2
  echo "  借用ビュー readout API（Var::host_view/Tensor::host_slice）が" >&2
  echo "  未収録のため、registry 解決のままの bench-fandhe ビルドは実行不可" >&2
  echo "  （#1438。旧計測専用 cargo feature〈#1337 導入〉は既に撤去済みで" >&2
  echo "  条件分岐そのものが存在しない）。" >&2
  if [[ -n "$note" ]]; then
    echo "  $note" >&2
  else
    echo "  GEMM_GATE_PATCH_FACADE_PATH=<crates/facade 絶対パス> を指定して" >&2
    echo "  crates/facade（HEAD ツリー）への path patch を併用すること。" >&2
    echo "  正式系列（registry ピン）の再計測はピン更新後にのみ可能" >&2
    echo "  （README「既定経路（#1438）」節参照）。" >&2
  fi
  exit 1
}

# PR #1452 codex-review P2 指摘（PRRT_kwDOTuUCJc6gKI3z）: `run_all.sh`／
# `run_ab_train_cuda.sh` は上記ガードにより常に `GEMM_GATE_PATCH_FACADE_PATH`
# 経由の `--config patch.crates-io.fandhe-ai.path=...` でビルドするが、
# `cargo build` はこの invocation-only patch を解決する過程でカレント
# ディレクトリの `Cargo.lock`（本 workspace の承認済みピン固定。
# deps-policy.md 第 9 区分）を patch 後の依存グラフへ書き換えてしまい、
# スクリプト終了後もその書き換えが残る（`run_ab_gemm_metal.sh`／
# `run_ab_graph_cuda.sh`／`run_ab_managed_cuda.sh` が individually 対処
# 済みの既知の問題と同型）。本関数群は `run_ab_gemm_metal.sh` の
# backup/restore trap（sha256 による退避内容確認・復元失敗時の非 0 終了
# 強制）と同一設計を共有化し、`run_all.sh`／`run_ab_train_cuda.sh` からも
# 呼び出せるようにする。呼び出し元は cargo build 実行前（かつ cd 済みの
# `scripts/bench/framework-compare/` ディレクトリ内）で
# `bench_fandhe_setup_lock_restore_trap` を 1 回呼ぶだけでよい。

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
