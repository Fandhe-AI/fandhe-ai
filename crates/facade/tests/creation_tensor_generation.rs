//! facade 経由の決定的テンソル生成（`fandhe_ai::{arange, linspace, eye,
//! zeros_like, ones_like}`。イシュー #1726）の統合テスト。
//!
//! ホスト生成した [`Tensor`] を [`fandhe_ai::tape`]／[`Tape::var`] で
//! デバイスへアップロードする経路（`docs/rng-global-contract-design.md`
//! §11 の設計方針どおり `BackendOps` を経由しない）を確認する。CPU
//! （`fandhe_ai::tape()`）はホスト生成データと bit 完全一致することを
//! 直接主張できる（生成自体がホスト側完結のため、アップロード先が
//! CPU であれば往復コピーのみで数値変化が起きない）。CUDA／Metal は
//! `#[ignore]` に分離し、`crates/facade/tests/rng_tensor_generation.rs`
//! と同じ方式で本エージェント実行環境に実機がある場合のみローカル
//! 実行する（本番 CI では実行されない。`.claude/rules/coding-rust.md`
//! 実機依存テストの分離方針）。
//!
//! 本モジュールが生成する値はグローバル状態を書き換えないため
//! （`rng.rs` と異なり `Mutex` 直列化は不要）、テスト同士の競合はない。

use fandhe_ai::{CreationError, Device, ShapeError, Tensor};

#[test]
fn arange_linspace_eye_upload_to_cpu_tape_are_bit_identical() {
    let arange_host = fandhe_ai::arange(0.0, 6.0, 1.0).unwrap();
    let linspace_host = fandhe_ai::linspace(-1.0, 1.0, 5).unwrap();
    let eye_host = fandhe_ai::eye(3).unwrap();

    let tape = fandhe_ai::tape();

    let arange_rt = tape.var(&arange_host).to_tensor();
    for i in 0..6 {
        assert_eq!(arange_host.get(&[i]), arange_rt.get(&[i]));
    }

    let linspace_rt = tape.var(&linspace_host).to_tensor();
    for i in 0..5 {
        assert_eq!(linspace_host.get(&[i]), linspace_rt.get(&[i]));
    }

    let eye_rt = tape.var(&eye_host).to_tensor();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(eye_host.get(&[i, j]), eye_rt.get(&[i, j]));
        }
    }
}

/// `x.matmul(eye(n))` が `x` と bit 一致することを確認する（対角 1 項の
/// `fma(x, 1, acc)` のみが寄与し他項は 0 のため正確。`eye` の機能検証を
/// 兼ねる実質的なスモーク。イシュー #1726）。
#[test]
fn matmul_with_eye_is_bit_exact_on_cpu() {
    // `arange` は 1 次元しか生成しないため、`[3, 4]` 形状は
    // `Tensor::new` で直接組み立てる（値自体は `arange` と同じ連番）。
    let x_host =
        Tensor::<f32>::new((0..12).map(|v| v as f32).collect::<Vec<f32>>(), &[3, 4]).unwrap();
    let eye_host = fandhe_ai::eye(4).unwrap();

    let tape = fandhe_ai::tape();
    let x = tape.var(&x_host);
    let e = tape.var(&eye_host);
    let out = x.matmul(&e).unwrap().to_tensor();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(out.get(&[i, j]), x_host.get(&[i, j]));
        }
    }
}

#[test]
fn zeros_like_ones_like_via_facade_follow_var_shape() {
    let base = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let tape = fandhe_ai::tape();
    let v = tape.var(&base);
    let v_tensor = v.to_tensor();

    let z = fandhe_ai::zeros_like(&v_tensor).unwrap();
    assert_eq!(z.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(z.get(&[i, j]).unwrap(), 0.0);
        }
    }

    let o = fandhe_ai::ones_like(&v_tensor).unwrap();
    assert_eq!(o.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(o.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

#[test]
fn arange_rejects_invalid_step_via_facade() {
    let err = fandhe_ai::arange(0.0, 5.0, 0.0).unwrap_err();
    assert_eq!(err, CreationError::InvalidStep { step: 0.0 });
}

#[test]
fn eye_rejects_overflow_via_facade() {
    let err = fandhe_ai::eye(usize::MAX).unwrap_err();
    assert_eq!(err, ShapeError::ElementCountOverflow);
}

#[test]
#[ignore = "実機（CUDA）依存。DGX Spark GB10 等で手動実行する"]
fn arange_upload_to_cuda_tape_round_trips() {
    let host = fandhe_ai::arange(0.0, 6.0, 1.0).unwrap();

    let tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA デバイスが利用可能であるはず");
    let v = tape.var(&host);
    let round_tripped = v.to_tensor();
    for i in 0..6 {
        assert_eq!(host.get(&[i]), round_tripped.get(&[i]));
    }
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "実機（Metal）依存。Apple Silicon 実機で手動実行する"]
fn arange_upload_to_metal_tape_round_trips() {
    let host = fandhe_ai::arange(0.0, 6.0, 1.0).unwrap();

    let tape = fandhe_ai::tape_for(Device::Metal).expect("Metal デバイスが利用可能であるはず");
    let v = tape.var(&host);
    let round_tripped = v.to_tensor();
    for i in 0..6 {
        assert_eq!(host.get(&[i]), round_tripped.get(&[i]));
    }
}
