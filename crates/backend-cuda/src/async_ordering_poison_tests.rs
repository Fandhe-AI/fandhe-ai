//! イシュー #1014（設計文書 `docs/backend-cuda-async-execution-design.md`
//! §8 T3i）: `invalidate_with` の drain・ストリーム完了同期（b'）を実 CUDA
//! クロージャで検証する。`context_cache.rs` の子モジュール
//! （`#[path]` 宣言。`jit_cache_regression_tests.rs` と同じ配置方式）と
//! して置く理由は、[`super::OrdinalRegistry`]・[`super::invalidate_with`]・
//! [`super::ProbeFailure`] がモジュール非公開（`context_cache` 内部限定）
//! であり、`tests/` 配下（別クレート扱いの統合テスト）からはアクセス
//! できないため。
//!
//! T3b（呼び出し側の拒否経路。GPU 不要・CI 常時実行）は
//! `crates/backend-cuda/src/ops.rs` の `#[cfg(test)] mod tests` へ追加した
//! （`begin_driver_call`／`observe_cuda_result` は `pub(crate)` のため
//! `ops.rs` から直接アクセスでき、既存の `gemm_rejects_on_poisoned_
//! ordinal_before_device_handle_is_attempted` 等と同じ場所へ置く方が
//! ヘルパー〈`unique_test_ordinal`・`EmptyHandle`・`sticky_driver_error`〉を
//! 再利用でき二重管理にならない）。本ファイルには置かない。

use std::sync::Arc;

use super::{OrdinalRegistry, Phase, ProbeFailure, invalidate_with};
use crate::device::CudaDevice;
use crate::sgd::{CudaSgd, SgdKernelParams};

/// **本テストが検証できること／できないこと（advisor レビューを経た
/// 方針決定・#1014 実装計画からの逸脱）**:
///
/// 当初計画（イシュー #1014 §5 手順 6）は「長時間カーネルを投入した直後に
/// 別スレッドから `invalidate_with` を呼び、復帰が host 復帰ではなく
/// デバイス側完了までブロックされることを完了マーカー値＋経過時間下限で
/// 検証する」ものだったが、本バックエンドは ordinal ごとに単一の
/// **in-order** ストリームのみを持つ（設計文書 §2.1・§3）。この構成では
/// `invalidate_with` の復帰後に発行する D2H は、`sync` クロージャが
/// 実際に `stream.synchronize()` を呼んだかどうかに関わらず、同一
/// ストリーム上の先行カーネルの後に実行される（ストリーム順序保証）。
/// したがって「値が正しい」ことは `sync` クロージャの呼び出し有無を
/// **判別しない**（`sync` を呼ばない実装へ差し替えても、続く D2H 自体が
/// 依然としてストリーム順序で先行カーネルの後に来るため、多くの場合は
/// 依然として正しい値が読める）。経過時間の下限判定も、閾値をどこに
/// 置いても環境依存でフレーキーになるため採用しない（advisor 指摘）。
///
/// `sync` クロージャが「呼ばれること」「`probe` より前に呼ばれること」
/// 「失敗時は `probe` を一切呼ばず恒久 poison へ倒すこと」という契約
/// そのものは、`context_cache.rs` の `poison_state_tests`
/// （`invalidate_with_sync_failure_poisons_unrecoverably` 等。GPU 非依存
/// モック）が既に検証している。本テストが追加する価値は
/// **配線（wiring）の実機確認**に限定する: 実際の
/// `device.stream().synchronize()` を `sync` クロージャとして渡した
/// `invalidate_with` 呼び出しが、実ストリームの残作業（多数の launch-only
/// SGD カーネル）を抱えた状態でも `Ok(())` を返し、`generation` が
/// ちょうど 1 進み `phase` が `Active` へ復帰し、かつ最終的な計算結果
/// （決定的な期待値）が正しく読み出せることを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn invalidate_with_real_stream_sync_drains_pending_launches_and_reactivates() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let sgd = CudaSgd::new(&device).expect("CudaSgd::new must succeed on real hardware");
    let stream = device.stream().clone();

    // 実行時間を確保するため大きめの numel を使う（数百万要素）。
    const NUMEL: usize = 8 * 1024 * 1024;
    const STEPS: usize = 100;

    let mut param = stream
        .clone_htod(&vec![0.0f32; NUMEL])
        .expect("alloc param");
    let grad = stream.clone_htod(&vec![1.0f32; NUMEL]).expect("alloc grad");

    let params = SgdKernelParams {
        lr: 1.0,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };

    // launch-only（同期なし）で連鎖投入する。momentum == 0.0 のため
    // velocity は不要（`sgd.rs::CudaSgd::run` 参照）。lr=1.0・grad=1.0・
    // weight_decay=0.0 なので各ステップで param -= 1.0（決定的）。
    for _ in 0..STEPS {
        sgd.run(&mut param, &grad, None, &params)
            .expect("cuda sgd launch must succeed on real hardware");
    }

    // 独立レジストリ（グローバル `ordinal_registry()` とは無関係。並走
    // する他の実機テスト・並行イシュー実装の ordinal 0/1 の状態機械へ
    // 干渉しない。イシュー #1014 実装計画 §3 方針 3 の判断）。
    let registry = OrdinalRegistry::new();
    let ordinal = 0usize; // registry 内で閉じたキーであり他の ordinal とは無関係。

    let stream_for_sync = Arc::clone(&stream);
    let result = invalidate_with(
        &registry,
        ordinal,
        // b'. ストリーム完了同期（実 CUDA）。
        move || {
            stream_for_sync
                .synchronize()
                .map_err(|_| ProbeFailure::Sticky)
        },
        // c. 実処理プローブ（本テストでは検証しない部分のため常に成功）。
        || Ok(()),
    );
    assert!(
        result.is_ok(),
        "実ストリームの sync クロージャが成功する限り invalidate_with は Ok を         返すはず: {result:?}"
    );

    {
        let cell = registry.entry(ordinal).expect("entry must succeed");
        let state = cell.0.lock().expect("mutex must not be poisoned");
        assert_eq!(
            state.generation, 1,
            "sync・probe の双方が成功した invalidate_with は generation を             ちょうど 1 進めるはず"
        );
        assert_eq!(
            state.phase,
            Phase::Active,
            "sync・probe の双方が成功した invalidate_with は phase を Active へ             復帰させるはず"
        );
    }

    // `invalidate_with` 復帰後に読み出す（上記の限界注記のとおり、この
    // 値の正しさ自体は sync クロージャの呼び出し有無を判別しないが、
    // 「実 CUDA クロージャを渡した呼び出しが少なくとも壊れていない」
    // ことの確認として残す）。
    let host: Vec<f32> = stream.clone_dtoh(&param).expect("readback must succeed");
    let expected = -(STEPS as f32);
    for (i, &v) in host.iter().enumerate().take(8) {
        assert!(
            (v - expected).abs() < 1e-3,
            "param[{i}] = {v}, expected {expected} after {STEPS} steps"
        );
    }
}
