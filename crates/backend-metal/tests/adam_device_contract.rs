//! `MetalBackendOps::adam_step_device`（イシュー #2070・in-place デバイス
//! 常駐 Adam・AdamW 更新）の Linux 実行可能な契約テスト。
//!
//! `MetalBackendOps`（`crate::ops`）自体は macOS 限定
//! （`cfg(target_os = "macos")`。`objc2` 系 FFI に触れるため）のため、
//! `crates/backend-cuda/tests/adam_device_contract.rs` と異なり本ファイル
//! は `MetalBackendOps::adam_step_device` を直接呼べない。代わりに
//! `fandhe_ai_backend_metal::adam_model`（cfg なし・`ops.rs::
//! adam_step_device_impl` が実際に呼ぶ純関数群）を直接検証することで、
//! device 検査を除く shape・`AdamStepKind` 分岐の契約を Linux 上で
//! ロックする（`ops.rs` 側の device 検査自体・実際の GPU 起動は
//! `adam_device_parity.rs`〈`#[ignore]`・macOS 実機限定〉へ引き継ぐ。
//! `adam_model.rs` モジュール doc「`Device::Metal` を扱わない理由」
//! 参照）。

use fandhe_ai_backend_metal::adam_model::{adam_kernel_flags, validate_adam_step_shapes};
use fandhe_ai_tensor_core::{AdamStepKind, ShapeError};

/// `grad` の shape が `param` と一致しない場合、`ShapeMismatch` を返す。
#[test]
fn grad_shape_mismatch_rejected() {
    let err = validate_adam_step_shapes(&[4], &[3], &[4], &[4]).unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

/// `m` の shape が `param` と一致しない場合、`ShapeMismatch` を返す。
#[test]
fn m_shape_mismatch_rejected() {
    let err = validate_adam_step_shapes(&[4], &[4], &[3], &[4]).unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

/// `v` の shape が `param` と一致しない場合、`ShapeMismatch` を返す。
#[test]
fn v_shape_mismatch_rejected() {
    let err = validate_adam_step_shapes(&[4], &[4], &[4], &[3]).unwrap_err();
    assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
}

/// 一致する shape は受理される。
#[test]
fn matching_shapes_accepted() {
    assert!(validate_adam_step_shapes(&[2, 3], &[2, 3], &[2, 3], &[2, 3]).is_ok());
}

/// `AdamStepKind::Coupled` かつ `weight_decay == 0.0` では
/// `use_coupled_wd` が偽になる（decay 項の演算自体を skip する CPU
/// 参照実装〈`backend-cpu::ops::adam_step_device`〉と同じ分岐）。
#[test]
fn coupled_weight_decay_zero_disables_coupled_wd_flag() {
    let flags = adam_kernel_flags(AdamStepKind::Coupled, 0.0).unwrap();
    assert!(!flags.use_coupled_wd);
    assert!(!flags.decoupled);
}

/// `AdamStepKind::Coupled` かつ `weight_decay != 0.0` では
/// `use_coupled_wd` が真になる。
#[test]
fn coupled_weight_decay_nonzero_enables_coupled_wd_flag() {
    let flags = adam_kernel_flags(AdamStepKind::Coupled, 0.01).unwrap();
    assert!(flags.use_coupled_wd);
    assert!(!flags.decoupled);
}

/// `AdamStepKind::Decoupled` では `weight_decay` の値に関わらず
/// `decoupled` が真・`use_coupled_wd` が偽になる（`p_eff = p *
/// decay_factor` を無条件適用する `AdamW` 契約）。
#[test]
fn decoupled_sets_decoupled_flag_regardless_of_weight_decay() {
    for weight_decay in [0.0f32, 0.01f32] {
        let flags = adam_kernel_flags(AdamStepKind::Decoupled, weight_decay).unwrap();
        assert!(flags.decoupled);
        assert!(!flags.use_coupled_wd);
    }
}
