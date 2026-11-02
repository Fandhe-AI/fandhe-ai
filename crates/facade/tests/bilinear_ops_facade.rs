//! facade 公開面だけで `Var::bilinear` に到達できることの固定（イシュー #3818・親 #3813）。
//! `fandhe_ai::*` のみを use する。層型 `Bilinear`／`BilinearVars` は facade に無く、2 入力の独自構造体は
//! `Var::bilinear` と自前のパラメータ（`nn::init::uniform` の出力）で組む。

use fandhe_ai::{AutodiffError, Tensor, Var};

/// シグネチャを関数ポインタ型の注釈で固定する（承認形からの逸脱をコンパイルエラーにする）。
#[test]
fn approved_signatures_are_pinned() {
    fn pin<'a, 't>(
        _: fn(
            &'a Var<'t>,
            &'a Var<'t>,
            &'a Var<'t>,
            Option<&'a Var<'t>>,
        ) -> Result<Var<'t>, AutodiffError>,
    ) {
    }
    pin(Var::bilinear);
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

#[test]
fn forward_backward_hand_computed_via_facade() {
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
    let x2 = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
    let w = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]));
    let b = tape.var(&t(vec![0.5], &[1]));
    let y = x1.bilinear(&x2, &w, Some(&b)).unwrap();
    assert_eq!(y.to_tensor().host_slice().to_vec(), vec![61.5]);
    let loss = y.sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    assert_eq!(
        g.get(&x1).unwrap().unwrap().host_slice().to_vec(),
        vec![11.0, 25.0]
    );
    assert_eq!(
        g.get(&x2).unwrap().unwrap().host_slice().to_vec(),
        vec![7.0, 10.0]
    );
    assert_eq!(
        g.get(&w).unwrap().unwrap().host_slice().to_vec(),
        vec![3.0, 4.0, 6.0, 8.0]
    );
    assert_eq!(g.get(&b).unwrap().unwrap().host_slice().to_vec(), vec![1.0]);
}

/// 2 入力の独自構造体の例（層型を使わずに `Var::bilinear` へ委譲する）。
struct MyBilinear {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl MyBilinear {
    fn new(in1: usize, in2: usize, out: usize) -> Result<Self, AutodiffError> {
        let bound = 1.0 / (in1 as f32).sqrt();
        Ok(Self {
            weight: fandhe_ai::nn::init::uniform(&[out, in1, in2], -bound, bound)?,
            bias: fandhe_ai::nn::init::uniform(&[out], -bound, bound)?,
        })
    }

    fn forward<'t>(
        &self,
        tape: &'t fandhe_ai::Tape,
        x1: &Var<'t>,
        x2: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        let w = tape.var(&self.weight);
        let b = tape.var(&self.bias);
        x1.bilinear(x2, &w, Some(&b))
    }
}

#[test]
fn custom_two_input_struct_example() {
    let layer = MyBilinear::new(3, 2, 4).unwrap();
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6], &[2, 3]));
    let x2 = tape.var(&t(vec![1.0, -1.0, 0.5, 2.0], &[2, 2]));
    let y = layer.forward(&tape, &x1, &x2).unwrap();
    assert_eq!(y.to_tensor().shape(), [2, 4]);
    let loss = y.sum(None).unwrap();
    let g = tape.backward(&loss).unwrap();
    assert!(g.get(&x1).unwrap().is_some());
    assert!(g.get(&x2).unwrap().is_some());
}

#[test]
fn invalid_inputs_are_rejected_via_facade() {
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&t(vec![1.0; 4], &[2, 2]));
    let x2 = tape.var(&t(vec![1.0; 6], &[2, 3]));
    let w = tape.var(&t(vec![1.0; 12], &[2, 2, 3]));
    // weight が rank 2。
    let w2 = tape.var(&t(vec![1.0; 6], &[2, 3]));
    assert!(matches!(
        x1.bilinear(&x2, &w2, None),
        Err(AutodiffError::Shape(_))
    ));
    // x1 が rank 1。
    let v = tape.var(&t(vec![1.0; 2], &[2]));
    assert!(matches!(
        v.bilinear(&x2, &w, None),
        Err(AutodiffError::Shape(_))
    ));
    // in2 の不一致。
    let bad = tape.var(&t(vec![1.0; 4], &[2, 2]));
    assert!(matches!(
        x1.bilinear(&bad, &w, None),
        Err(AutodiffError::Shape(_))
    ));
    // bias の shape 不一致。
    let b = tape.var(&t(vec![0.0; 3], &[3]));
    assert!(matches!(
        x1.bilinear(&x2, &w, Some(&b)),
        Err(AutodiffError::Shape(_))
    ));
    // 別 tape。
    let other = fandhe_ai::tape();
    let ox2 = other.var(&t(vec![1.0; 6], &[2, 3]));
    assert!(matches!(
        x1.bilinear(&ox2, &w, None),
        Err(AutodiffError::TapeMismatch)
    ));
}
