//! イシュー #1349（親 #1348・ルート #1341 → #1269）受け入れ条件 (a):
//! 「学習 step を CUDA Graph capture 経路（opt-in ON）と非 capture 経路
//! （opt-in OFF）で実行し、10 step の損失・勾配・パラメータが bit 同一
//! であること」を facade 公開 API 経由で検証する実機テスト。
//!
//! `crates/facade/tests/device_param_store_train.rs`（CPU・
//! `fandhe_ai::tape()`）と同じ学習ループ構成を `fandhe_ai::tape_for(
//! Device::Cuda(0))` へ差し替えたもの（共有ヘルパーは
//! `tests/cuda_graph_step_common/mod.rs`）。capture できるのは update
//! 区間（`sgd_step_device_tracked`）のみであり forward／backward は対象外
//! （`docs/backend-cuda-graph-step-capture-design.md` §1・§3.2）ため、
//! bit 同一性は「同じ update カーネル・同じポインタへ同じ引数で
//! launch される」という構成上の保証（同 doc §4.5）の直接検証となる。
//!
//! 2 GPU 構成の機械比較テスト（`graph_capture_matches_eager_baseline_
//! bit_identical_across_two_gpus`）は**別ファイル**
//! （`tests/cuda_graph_step_two_gpu_bit_identity.rs`）に分離している
//! （codex-review P2 指摘対応・PR #1390。理由は分離先ファイル・
//! `cuda_graph_step_common/mod.rs` 冒頭コメント参照）。
//!
//! **opt-in はプロセス内で最初の CUDA デバイス初期化より前に設定する
//! 必要がある**（`fandhe_ai::set_cuda_graph_step_enabled` doc 参照）ため、
//! 本ファイルは 2 プロセス構成にはせず、**先に非 capture 経路（opt-in
//! OFF）を完走させ、その後で capture 経路（opt-in ON）用に別 ordinal
//! を使う**——ただし DGX Spark GB10 等の単一 GPU 構成では ordinal 1 が
//! 存在しないため、この構成では成立しない。そのため実運用は「本ファイル
//! を 2 回、環境変数 `FANDHE_AI_CUDA_GRAPH_STEP` の有無で切り替えて
//! それぞれ 1 プロセスとして実行し、出力 JSON を突合する」形を取る
//! （`--nocapture` で loss ログを標準出力へ出す）。
//!
//! **各テストは必ず `--exact` を付けて単独実行すること（codex-review
//! P2 指摘対応・PR #1390）**: `eager_baseline`・`graph_capture`・
//! `graph_capture_completes_training_loop_without_error` の 3 関数名は
//! いずれも文字列 `graph_capture` を含むため、`--exact` なしの部分一致
//! フィルタ（旧稿の `-- --ignored --nocapture graph_capture` 等）は
//! 複数関数を同一プロセス・並行スレッドで選んでしまう。各テストは
//! プロセスワイドな opt-in フラグ（`fandhe_ai::
//! set_cuda_graph_step_enabled`）と「最初のデバイス初期化より前に
//! 固定される」前提を持つため、複数が並行実行されると
//! （a）ある関数が観測する opt-in の値が他スレッドの変更で不定になる、
//! （b）「最初のデバイス初期化」がどのテストの呼び出しで発生するか
//! 実行順序に依存する、という形で検証契約が崩れる（フィルタの部分
//! 一致によりテストランナー自身が複数関数を選んでしまう構造的な問題
//! であり、`--test-threads=1` だけでは実行順序が非決定的なままのため
//! 解決しない）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`docs/real-hardware-
//! verification-env.md` の手順に従う）:
//!
//! ```sh
//! # 非 capture 経路（比較の基準値）
//! cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
//!   -- --ignored --nocapture --exact eager_baseline
//!
//! # capture 経路（opt-in ON。同一プロセス内で最初のデバイス初期化前に
//! # 設定されるよう、テスト関数の先頭で `set_cuda_graph_step_enabled(true)`
//! # を呼ぶ）
//! FANDHE_AI_CUDA_GRAPH_STEP=1 \
//! cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
//!   -- --ignored --nocapture --exact graph_capture
//!
//! # 弱い受け入れ条件（opt-in ON で学習ループが最後まで通る）の単独確認
//! cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
//!   -- --ignored --nocapture --exact graph_capture_completes_training_loop_without_error
//! ```
//!
//! 両者の標準出力（loss 列・各 step 完了直後のパラメータ・最終
//! パラメータのビット表現）を比較し、完全一致することを確認する。
//!
//! **単一 GPU 環境での自動比較（codex-review P2 指摘対応・PR #1390
//! 再修正）**: 上記 2 コマンドを手動で実行し目視・手動 diff するのは
//! 本ファイル自体（2 プロセス構成の制約は変わらない）では検出漏れの
//! リスクがあるため、`scripts/verify-cuda-graph-step-bit-identity.sh`
//! （CUDA 実機限定・通常 CI では実行しない）が両プロセスを順に実行し
//! `step[...].loss.bits`／`step[...].param[...][...].bits`／
//! `final.param[...][...].bits` の全行を機械的に diff する。不一致が
//! あれば非ゼロ終了する（fail-closed）。2 GPU 搭載機での機械比較は
//! 別途 `cuda_graph_step_two_gpu_bit_identity.rs`（`assert_eq!` に
//! よる同一プロセス内比較）が担う。

#[path = "cuda_graph_step_common/mod.rs"]
mod cuda_graph_step_common;

use cuda_graph_step_common::{LR, STEPS, print_bit_identity_report, train_on_cuda};

/// 非 capture 経路（opt-in OFF。既定）の基準値を出力する。
///
/// `FANDHE_AI_CUDA_GRAPH_STEP` が未設定であれば opt-in は既定 OFF の
/// ままのはず（`--exact` 単独実行が前提。ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。docs/real-hardware-verification-env.md 参照。--exact 単独実行必須"]
fn eager_baseline() {
    assert!(
        !fandhe_ai::cuda_graph_step_enabled(),
        "本テストは opt-in OFF（既定）の基準値を記録する。環境変数 \
         FANDHE_AI_CUDA_GRAPH_STEP を設定せずに実行すること"
    );
    let (log, per_step_params, final_params) = train_on_cuda(0, STEPS, LR);
    print_bit_identity_report("eager (opt-in OFF)", &log, &per_step_params, &final_params);
}

/// capture 経路（opt-in ON）を実行する。`FANDHE_AI_CUDA_GRAPH_STEP=1`
/// を設定した別プロセスとして実行することを想定する
/// （`fandhe_ai::set_cuda_graph_step_enabled` は「最初のデバイス初期化
/// より前」の制約があるため、同一プロセス内で `eager_baseline` の後に
/// 実行しても capture は成立しない。`--exact` 単独実行が前提。ファイル
/// 冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_STEP=1 の別プロセスで --exact 単独実行すること"]
fn graph_capture() {
    assert!(
        fandhe_ai::cuda_graph_step_enabled(),
        "本テストは opt-in ON（環境変数 FANDHE_AI_CUDA_GRAPH_STEP=1）の \
         別プロセスとして実行すること"
    );
    let (log, per_step_params, final_params) = train_on_cuda(0, STEPS, LR);
    print_bit_identity_report(
        "graph capture (opt-in ON)",
        &log,
        &per_step_params,
        &final_params,
    );
}

/// 同一プロセス内でも検証できる範囲の簡易チェック: opt-in を
/// 明示的に ON へ設定してから学習ループを走らせ、少なくとも
/// `BackendError::Unsupported`（opt-in ON だが legacy stream のまま等
/// の設定順序ミス）を起こさずに完走することを確認する（bit 同一性
/// そのものはプロセスを跨いだ突合〈`eager_baseline`／`graph_capture`〉
/// または 2 GPU 機械比較〈`cuda_graph_step_two_gpu_bit_identity.rs`〉に
/// 委ねる。本テストは「opt-in ON の状態で学習ループが最後まで通る」
/// という弱い受け入れ条件のみを 1 プロセスで検証する）。
///
/// **opt-in を明示的に OFF へ戻す処理は行わない（codex-review P2
/// 指摘対応・PR #1390 是正）**: 旧稿は末尾で
/// `set_cuda_graph_step_enabled(false)` を呼んでいたが、これは
/// プロセスワイドなグローバル状態を書き換える操作であり、他テストが
/// 同一プロセス・並行スレッドで選ばれた場合（フィルタの部分一致。
/// ファイル冒頭コメント参照）に競合を引き起こす副作用がある一方、
/// 本ファイルは各テストを `--exact` で単独実行する契約（同コメント
/// 参照）のため、プロセス終了時に暗黙に破棄される opt-in フラグを
/// 明示的にリセットする必要自体がない。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。単独プロセスで --exact 単独実行すること（opt-in をプロセスワイドに変更するため）"]
fn graph_capture_completes_training_loop_without_error() {
    fandhe_ai::set_cuda_graph_step_enabled(true);
    let (log, _per_step_params, _final_params) = train_on_cuda(0, STEPS, LR);
    assert_eq!(log.len(), STEPS);
    for loss in &log {
        assert!(loss.is_finite(), "loss must remain finite: {loss}");
    }
}
