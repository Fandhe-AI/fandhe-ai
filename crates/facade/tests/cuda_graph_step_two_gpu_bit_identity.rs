//! イシュー #1349 受け入れ条件 (a) の**機械比較版**（codex-review P2
//! 指摘対応。`cuda_graph_step_bit_identity.rs` から分離。PR #1390）:
//! `eager_baseline`／`graph_capture`（`cuda_graph_step_bit_identity.rs`）
//! は出力を印字するのみで自動比較しない（目視・スクリプト任せ）ため、
//! 2 GPU 搭載機では本テスト単独で「損失・最終パラメータが bit 同一」を
//! `assert_eq!` により機械検証する。
//!
//! **`cuda_graph_step_bit_identity.rs` から別ファイル（＝別テスト
//! バイナリ・別プロセス）へ分離した理由**: 本テストは「ordinal 0 の
//! eager baseline は opt-in OFF のまま初期化する」という前提を持つが、
//! 同じファイルにあった `graph_capture`／
//! `graph_capture_completes_training_loop_without_error` は opt-in を
//! ON のまま・またはプロセスワイドに変更する前提を持つ。3 関数とも
//! 名前に `graph_capture` を含むため、`cargo test graph_capture` の
//! ような部分一致フィルタで同一プロセス・並行スレッドに選ばれると、
//! 本テストの「opt-in OFF で開始する」前提が他スレッドの操作で崩れうる
//! （分離元・共有ヘルパー `cuda_graph_step_common/mod.rs` 冒頭コメント
//! 参照）。ファイルを分ければ `cargo test` のフィルタ挙動に関わらず
//! 構造的に同一プロセスへ混在しない。
//!
//! **前提**: opt-in はプロセス内最初の CUDA デバイス初期化より前に固定
//! される必要がある（`fandhe_ai::set_cuda_graph_step_enabled` doc・
//! `cuda_graph_step_bit_identity.rs` 冒頭コメント参照）ため、同一
//! プロセス内で両経路を機械比較するには異なる ordinal が要る:
//! ordinal 0 で opt-in OFF（eager）を先に完走させたあと opt-in を
//! ON にし、ordinal 1（この時点で初めて初期化される）で capture
//! 経路を走らせる。
//!
//! **単一 GPU 環境（DGX Spark GB10 等）では成立しない**（`CudaDevice::
//! device_count()` が 2 未満なら早期 return する。単一 GPU 環境の
//! 受け入れ検証は既存の 2 プロセス比較
//! （`eager_baseline`／`graph_capture`。`cuda_graph_step_bit_identity.rs`）
//! に委ねる）。
//!
//! 実行コマンド（2 GPU 搭載機。`--exact` は他ファイルとの一貫性のため
//! 付けるが、本テストは元々単一ファイルにつき単一テストのため部分
//! 一致による衝突リスクはない）:
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test cuda_graph_step_two_gpu_bit_identity \
//!   -- --ignored --nocapture --exact graph_capture_matches_eager_baseline_bit_identical_across_two_gpus
//! ```

#[path = "cuda_graph_step_common/mod.rs"]
mod cuda_graph_step_common;

use cuda_graph_step_common::{LR, STEPS, print_bit_identity_report, train_on_cuda};
use fandhe_ai_backend_cuda::device::CudaDevice;

#[test]
#[ignore = "CUDA 実機（2 GPU 構成）必須。単独プロセスで実行すること（opt-in をプロセスワイドに変更するため）。単一 GPU 環境では device_count() < 2 のため早期 return する"]
fn graph_capture_matches_eager_baseline_bit_identical_across_two_gpus() {
    let device_count = CudaDevice::device_count().unwrap_or(0);
    if device_count < 2 {
        eprintln!(
            "device_count()={device_count} < 2: 単一 GPU 環境のため本テストの機械比較は成立しない\
             （2 プロセス構成の eager_baseline／graph_capture に委ねる）。早期 return する。"
        );
        return;
    }

    assert!(
        !fandhe_ai::cuda_graph_step_enabled(),
        "ordinal 0 の eager baseline は opt-in OFF のまま初期化する必要がある"
    );
    let (eager_log, eager_params) = train_on_cuda(0, STEPS, LR);

    fandhe_ai::set_cuda_graph_step_enabled(true);
    let (graph_log, graph_params) = train_on_cuda(1, STEPS, LR);
    fandhe_ai::set_cuda_graph_step_enabled(false);

    print_bit_identity_report("eager (opt-in OFF, ordinal 0)", &eager_log, &eager_params);
    print_bit_identity_report(
        "graph capture (opt-in ON, ordinal 1)",
        &graph_log,
        &graph_params,
    );

    assert_eq!(
        eager_log.len(),
        graph_log.len(),
        "loss 列の長さが一致しないはず（STEPS は共通の定数）"
    );
    for (i, (e, g)) in eager_log.iter().zip(graph_log.iter()).enumerate() {
        assert_eq!(
            e.to_bits(),
            g.to_bits(),
            "step[{i}] の loss が bit 同一でない: eager={e:#010x?}／graph={g:#010x?}"
        );
    }

    assert_eq!(
        eager_params.len(),
        graph_params.len(),
        "パラメータ列の個数が一致しないはず（同一モデル構成）"
    );
    for (p, (ep, gp)) in eager_params.iter().zip(graph_params.iter()).enumerate() {
        let e_contig = ep.contiguous();
        let g_contig = gp.contiguous();
        let e_slice = e_contig.as_slice().unwrap_or(&[]);
        let g_slice = g_contig.as_slice().unwrap_or(&[]);
        assert_eq!(
            e_slice.len(),
            g_slice.len(),
            "param[{p}] の要素数が一致しないはず"
        );
        for (i, (ev, gv)) in e_slice.iter().zip(g_slice.iter()).enumerate() {
            assert_eq!(
                ev.to_bits(),
                gv.to_bits(),
                "param[{p}][{i}] が bit 同一でない: eager={ev:#010x?}／graph={gv:#010x?}"
            );
        }
    }
}
