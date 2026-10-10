//! `fandhe_ai::models::{Mlp, LeNet}`（イシュー #2974・親 #2541 の Phase 11-1）の統合テスト。
//!
//! 公開パス（`fandhe_ai::models`・`fandhe_ai::compat`）と `std` だけを使い、次を固定する。
//! - 既存の `save_model`／`load_model` 経由の保存往復が bit 一致する（新しい保存経路は無い。
//!   `docs/reference-models-decision.md` §11）
//! - 読み込んだ状態を同構成の別シードのモデルへ `load_state_dict` で戻すと、公開型の
//!   `predict` が元と bit 一致する
//! - `Mlp::new` の既定シードが examples 実装から変わっていない
//! - 外部 `Tape` の `forward` と `predict` が eval モードで一致し、不正引数は型付きエラーになる
//!
//! 学習判定は `example_mlp_mnist.rs`／`example_lenet_mnist.rs` が担う。新しいカーネルは無いため
//! 実機（CUDA／Metal）依存の `#[ignore]` テストは追加しない。

use fandhe_ai::compat::Sequential;
use fandhe_ai::models::{LeNet, Mlp};
use fandhe_ai::{AutodiffError, Tensor};

#[cfg(unix)]
use fandhe_ai::compat::{load_model, save_model};

#[cfg(unix)]
mod common;
#[cfg(unix)]
use common::temp_dir::TempDirGuard;

/// `Mlp::new` の既定シード（`models/mlp.rs` の `DEFAULT_SEED`。移設でずれていないことの固定用）。
const MLP_DEFAULT_SEED: u64 = 0x4D4C505F53454544;

fn input(shape: &[usize], base: f32) -> Tensor<f32> {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.37 + base).sin()).collect();
    Tensor::new(data, shape).expect("テンソルを作れるはず")
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let (ca, cb) = (a.contiguous(), b.contiguous());
    let (sa, sb) = (
        ca.as_slice().expect("連続のはず"),
        cb.as_slice().expect("連続のはず"),
    );
    assert!(
        sa.iter()
            .zip(sb.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{what}: bit 不一致"
    );
}

fn assert_state_dicts_identical(a: &Sequential, b: &Sequential) {
    let (sa, sb) = (a.state_dict(), b.state_dict());
    assert_eq!(sa.len(), sb.len(), "state_dict のキー数");
    for (k, v) in &sa {
        let other = sb.get(k).unwrap_or_else(|| panic!("キー {k} が欠落"));
        assert_bit_identical(v, other, k);
    }
}

#[cfg(unix)]
#[test]
fn mlp_save_load_roundtrip_is_bit_identical() {
    let guard = TempDirGuard::new("models-mlp");
    let dir = guard.path().join("model");
    let mut mlp = Mlp::new(12, &[16, 8], 4, 0.2).unwrap();
    mlp.sequential_mut().eval();
    let x = input(&[3, 12], 0.5);

    save_model(mlp.sequential(), &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");

    assert_state_dicts_identical(mlp.sequential(), &loaded);
    assert_eq!(mlp.sequential().training(), loaded.training());
    assert_bit_identical(
        &mlp.predict(&x).unwrap(),
        &loaded.predict(&x).unwrap(),
        "Mlp predict（復元した Sequential）",
    );

    // 公開型へ戻す経路: 別シードの同構成モデルへ state_dict を読み戻す。
    let mut restored = Mlp::with_seed(12, &[16, 8], 4, 0.2, 999).unwrap();
    restored.sequential_mut().eval();
    restored
        .sequential_mut()
        .load_state_dict(loaded.state_dict())
        .expect("読み戻せるはず");
    assert_bit_identical(
        &mlp.predict(&x).unwrap(),
        &restored.predict(&x).unwrap(),
        "Mlp predict（state_dict 読み戻し後）",
    );
}

#[cfg(unix)]
#[test]
fn lenet_save_load_roundtrip_is_bit_identical() {
    let guard = TempDirGuard::new("models-lenet");
    let dir = guard.path().join("model");
    let mut lenet = LeNet::new(10, 7).unwrap();
    lenet.sequential_mut().eval();
    let x = input(&[2, 1, 28, 28], 0.25);

    save_model(lenet.sequential(), &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");

    assert_state_dicts_identical(lenet.sequential(), &loaded);
    assert_eq!(lenet.sequential().training(), loaded.training());
    assert_bit_identical(
        &lenet.predict(&x).unwrap(),
        &loaded.predict(&x).unwrap(),
        "LeNet predict（復元した Sequential）",
    );

    let mut restored = LeNet::new(10, 999).unwrap();
    restored.sequential_mut().eval();
    restored
        .sequential_mut()
        .load_state_dict(loaded.state_dict())
        .expect("読み戻せるはず");
    assert_bit_identical(
        &lenet.predict(&x).unwrap(),
        &restored.predict(&x).unwrap(),
        "LeNet predict（state_dict 読み戻し後）",
    );
}

#[test]
fn mlp_default_seed_matches_with_seed() {
    let a = Mlp::new(6, &[5], 3, 0.1).unwrap();
    let b = Mlp::with_seed(6, &[5], 3, 0.1, MLP_DEFAULT_SEED).unwrap();
    assert_state_dicts_identical(a.sequential(), b.sequential());
}

#[test]
fn forward_on_external_tape_matches_predict_in_eval_mode() {
    let mut mlp = Mlp::with_seed(8, &[6], 3, 0.3, 1).unwrap();
    mlp.sequential_mut().eval();
    let x = input(&[2, 8], 0.1);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = mlp.forward(&tape, &xv).unwrap().to_tensor();
    assert_bit_identical(
        &via_forward,
        &mlp.predict(&x).unwrap(),
        "Mlp forward/predict",
    );

    let mut lenet = LeNet::new(5, 2).unwrap();
    lenet.sequential_mut().eval();
    let x = input(&[1, 1, 28, 28], 0.3);
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let via_forward = lenet.forward(&tape, &xv).unwrap().to_tensor();
    assert_bit_identical(
        &via_forward,
        &lenet.predict(&x).unwrap(),
        "LeNet forward/predict",
    );
}

#[test]
fn invalid_arguments_return_typed_errors() {
    for r in [
        Mlp::new(0, &[4], 2, 0.1),
        Mlp::new(4, &[0], 2, 0.1),
        Mlp::new(4, &[4], 0, 0.1),
        Mlp::new(4, &[], 2, 1.5),
        Mlp::new(4, &[], 2, f32::NAN),
        Mlp::new(4, &[4], 2, -0.1),
    ] {
        assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
    }
    assert!(matches!(
        LeNet::new(0, 1),
        Err(AutodiffError::InvalidArgument(_))
    ));
}
