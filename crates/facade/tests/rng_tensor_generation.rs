//! facade 経由の乱数テンソル生成（`fandhe_ai::{randn, rand, randint}`。
//! イシュー #1725）の統合テスト。
//!
//! ホスト生成した [`Tensor`] を [`Tape::var`] でデバイスへアップロード
//! する経路（`docs/rng-global-contract-design.md` の設計方針どおり
//! `BackendOps` を経由しない）を確認する。CPU（`fandhe_ai::tape()`）は
//! ホスト生成データと bit 完全一致することを直接主張できる（生成自体が
//! ホスト側完結のため、アップロード先が CPU であれば往復コピーのみで
//! 数値変化が起きない）。CUDA／Metal は `#[ignore]` に分離し、
//! `crates/facade/tests/shape_ops_backend_parity.rs` の
//! `VarSource` 方式に倣って本エージェント実行環境に実機がある場合の
//! みローカル実行する（本番 CI では実行されない。
//! `.claude/rules/coding-rust.md` 実機依存テストの分離方針）。
//!
//! グローバル RNG 状態を書き換えるため、他の統合テストファイルとの
//! プロセス間競合はないが（`cargo test` は統合テストファイルごとに
//! 別プロセス）、本ファイル内のテスト同士は `cargo test` の既定並列
//! 実行で競合しうるため、ファイル局所 `Mutex` で直列化する
//! （`crates/tensor-core/src/rng.rs::global_rng_test_lock` と同型）。

use std::sync::Mutex;

use fandhe_ai::{Device, RngError, Tensor};

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

#[test]
fn manual_seed_then_randn_rand_randint_are_deterministic_via_facade() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai::manual_seed(7);
    let n1 = fandhe_ai::randn(&[2, 3]).unwrap();
    let u1 = fandhe_ai::rand(&[2, 3]).unwrap();
    let i1 = fandhe_ai::randint(-5, 5, &[4]).unwrap();

    fandhe_ai::manual_seed(7);
    let n2 = fandhe_ai::randn(&[2, 3]).unwrap();
    let u2 = fandhe_ai::rand(&[2, 3]).unwrap();
    let i2 = fandhe_ai::randint(-5, 5, &[4]).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(n1.get(&[i, j]), n2.get(&[i, j]));
            assert_eq!(u1.get(&[i, j]), u2.get(&[i, j]));
        }
    }
    for i in 0..4 {
        assert_eq!(i1.get(&[i]), i2.get(&[i]));
    }
}

#[test]
fn randint_rejects_invalid_range_via_facade() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let err = fandhe_ai::randint(3, 3, &[1]).unwrap_err();
    assert_eq!(err, RngError::InvalidRange { low: 3, high: 3 });
}

/// ホスト生成 → CPU デバイスへの `Tape::var` アップロードが bit 完全
/// 一致であることを確認する（CPU は往復コピーのみで数値変化がない
/// ため、REQ-2 複合判定ではなく bit 一致を直接主張できる）。
#[test]
fn randn_upload_to_cpu_tape_is_bit_identical_and_sum_matches_host_computation() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai::manual_seed(11);
    let host = fandhe_ai::randn(&[2, 3]).unwrap();

    let tape = fandhe_ai::tape();
    let v = tape.var(&host);
    let round_tripped = v.to_tensor();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(host.get(&[i, j]), round_tripped.get(&[i, j]));
        }
    }

    let host_sum: f32 = (0..2)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| host.get(&[i, j]).unwrap())
        .sum();
    let sum_var = v.sum(None).unwrap();
    let sum_tensor = sum_var.to_tensor();
    assert_eq!(sum_tensor.get(&[]).unwrap(), host_sum);
}

/// `randint` の出力を `Var::index_select` の index 引数としてそのまま
/// 使えることを確認する（`Tensor<i32>` の dtype 整合スモーク。
/// `docs/rng-global-contract-design.md`「dtype 判断」参照）。
#[test]
fn randint_output_is_usable_as_index_select_index() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai::manual_seed(3);
    let data = Tensor::<f32>::new((0..12).map(|v| v as f32).collect(), &[4, 3]).unwrap();
    let idx = fandhe_ai::randint(0, 4, &[2]).unwrap();

    let tape = fandhe_ai::tape();
    let v = tape.var(&data);
    let selected = v.index_select(0, &idx).unwrap();
    let out = selected.to_tensor();
    assert_eq!(out.shape(), &[2, 3]);
}

#[test]
#[ignore = "実機（CUDA）依存。DGX Spark GB10 等で手動実行する"]
fn randn_upload_to_cuda_tape_round_trips() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai::manual_seed(21);
    let host = fandhe_ai::randn(&[2, 3]).unwrap();

    let tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA デバイスが利用可能であるはず");
    let v = tape.var(&host);
    let round_tripped = v.to_tensor();
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(host.get(&[i, j]), round_tripped.get(&[i, j]));
        }
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "実機（Metal）依存。Apple Silicon 実機で手動実行する"]
fn randn_upload_to_metal_tape_round_trips() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    fandhe_ai::manual_seed(22);
    let host = fandhe_ai::randn(&[2, 3]).unwrap();

    let tape = fandhe_ai::tape_for(Device::Metal).expect("Metal デバイスが利用可能であるはず");
    let v = tape.var(&host);
    let round_tripped = v.to_tensor();
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(host.get(&[i, j]), round_tripped.get(&[i, j]));
        }
    }
}
