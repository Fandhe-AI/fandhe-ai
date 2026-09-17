//! `Tape::custom`／`CustomFunction`（イシュー #1946・案 B。
//! `docs/autodiff-custom-function-decision.md` §12.4）の統合テスト。
//!
//! `crates/autodiff/tests/*.rs` は `autodiff` の公開 API のみを経由する
//! 別クレート扱いのため、ここでは `fandhe_ai_autodiff::{Tape,
//! CustomFunction}` 経由でのみ検証する（`Op::Custom`／`CustomFn` 自体は
//! クレート非公開）。`ResidentLeaf` 入力拒否・`is_checkpoint_eligible`
//! ・`for_each_input` の網羅性は `tape.rs` 内の単体テスト
//! （`pub(crate)` API を直接使う必要があるため）で検証する。

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fandhe_ai_autodiff::{AutodiffError, CustomFunction, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

/// 組み込み `Var::relu` と同じ意味論（forward: `nan_propagating_max(x,
/// 0)`・backward: 入力 `x > 0.0` マスク。`grad.rs::Op::Relu` の VJP と
/// 同一述語）を独立実装した `CustomFunction`。受入基準 2（組み込みと
/// bit 一致）の検証対象。
struct CustomRelu;

impl CustomFunction for CustomRelu {
    fn name(&self) -> &str {
        "custom_relu"
    }

    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }

    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let x = inputs[0].contiguous();
        let data: Vec<f32> = x
            .as_slice()
            .expect("test fixture: contiguous 直後は必ず Some")
            .iter()
            .map(|&v| if v.is_nan() { f32::NAN } else { v.max(0.0) })
            .collect();
        Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)
    }

    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        let x = dense(inputs[0]);
        let u = dense(upstream);
        let data: Vec<f32> = x
            .iter()
            .zip(u.iter())
            .map(|(&xv, &uv)| if xv > 0.0 { uv } else { 0.0 })
            .collect();
        let g = Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)?;
        Ok(vec![Some(g)])
    }
}

/// 受入基準 2: 自作 `CustomRelu` を `w` との積・`sum` で非一様な
/// upstream を作った上で、組み込み `Var::relu` の同じグラフと forward
/// 値・`x`／`w` への勾配が **bit 完全一致**することを確認する（負・0・
/// -0.0・NaN を含む入力で `to_bits` 全要素比較）。
#[test]
fn custom_relu_matches_builtin_relu_bit_exact() {
    let x_data = vec![-2.0, 0.0, -0.0, 3.5, f32::NAN, -1e-3];
    let w_data = vec![1.5, -2.0, 0.5, 4.0, 2.0, -3.0];
    let shape = [2usize, 3];

    // 組み込み経路
    let tape_builtin = Tape::new_with_ops(common::naive_ops());
    let x_b = tape_builtin.var(&t(x_data.clone(), &shape));
    let w_b = tape_builtin.var(&t(w_data.clone(), &shape));
    let loss_b = x_b
        .relu()
        .mul(&w_b)
        .expect("同 shape の要素積")
        .sum(None)
        .expect("全縮約");
    let forward_b = loss_b.to_tensor();
    let grads_b = tape_builtin.backward(&loss_b).expect("backward 成功");
    let dx_b = grads_b
        .get(&x_b)
        .expect("get 成功")
        .expect("x は loss に寄与する")
        .clone();
    let dw_b = grads_b
        .get(&w_b)
        .expect("get 成功")
        .expect("w は loss に寄与する")
        .clone();

    // 自作 `CustomRelu` 経路
    let tape_custom = Tape::new_with_ops(common::naive_ops());
    let x_c = tape_custom.var(&t(x_data, &shape));
    let w_c = tape_custom.var(&t(w_data, &shape));
    let relu_c = tape_custom
        .custom(Arc::new(CustomRelu), &[x_c])
        .expect("Tape::custom 登録成功");
    let loss_c = relu_c
        .mul(&w_c)
        .expect("同 shape の要素積")
        .sum(None)
        .expect("全縮約");
    let forward_c = loss_c.to_tensor();
    let grads_c = tape_custom.backward(&loss_c).expect("backward 成功");
    let dx_c = grads_c
        .get(&x_c)
        .expect("get 成功")
        .expect("x は loss に寄与する")
        .clone();
    let dw_c = grads_c
        .get(&w_c)
        .expect("get 成功")
        .expect("w は loss に寄与する")
        .clone();

    let bits_eq = |a: &Tensor<f32>, b: &Tensor<f32>| {
        let da = dense(a);
        let db = dense(b);
        assert_eq!(da.len(), db.len());
        for (i, (&av, &bv)) in da.iter().zip(db.iter()).enumerate() {
            assert_eq!(
                av.to_bits(),
                bv.to_bits(),
                "index {i}: builtin={av} custom={bv}"
            );
        }
    };
    bits_eq(&forward_b, &forward_c);
    bits_eq(&dx_b, &dx_c);
    bits_eq(&dw_b, &dw_c);
}

/// 2 入力の解析的 `CustomFunction`（`a * b` 相当）。複数入力・
/// 正常系の検証対象。
struct CustomMul;

impl CustomFunction for CustomMul {
    fn name(&self) -> &str {
        "custom_mul"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let a = dense(inputs[0]);
        let b = dense(inputs[1]);
        let data: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
        Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        let a = dense(inputs[0]);
        let b = dense(inputs[1]);
        let u = dense(upstream);
        let da = if requires_grad[0] {
            let data: Vec<f32> = u.iter().zip(b.iter()).map(|(&uv, &bv)| uv * bv).collect();
            Some(Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)?)
        } else {
            None
        };
        let db = if requires_grad[1] {
            let data: Vec<f32> = u.iter().zip(a.iter()).map(|(&uv, &av)| uv * av).collect();
            Some(Tensor::new(data, inputs[1].shape()).map_err(AutodiffError::Shape)?)
        } else {
            None
        };
        Ok(vec![da, db])
    }
}

/// 受入基準 3・複数入力の正常系: `custom_mul(a, b)` の解析勾配が
/// `a.mul(&b)`（組み込み）と bit 一致することを確認する。
#[test]
fn custom_function_with_two_inputs_matches_analytic_mul() {
    let a_data = vec![2.0, -3.0, 0.5, 4.0];
    let b_data = vec![-1.0, 2.0, 3.0, -0.5];
    let shape = [2usize, 2];

    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(a_data.clone(), &shape));
    let b = tape.var(&t(b_data.clone(), &shape));
    let custom_out = tape
        .custom(Arc::new(CustomMul), &[a, b])
        .expect("2 入力 custom 登録成功");
    let loss = custom_out.sum(None).expect("全縮約");
    let grads = tape.backward(&loss).expect("backward 成功");
    let da = grads.get(&a).unwrap().unwrap().clone();
    let db = grads.get(&b).unwrap().unwrap().clone();

    assert_eq!(dense(&da), b_data);
    assert_eq!(dense(&db), a_data);
}

/// `output_shape` が宣言した shape と実際の `forward` 出力 shape が
/// 食い違う場合、`Tape::custom` は `AutodiffError::Shape` を返し、
/// テープにノードを増やさない（§6「shape 検証」）。
struct WrongShapeFn;

impl CustomFunction for WrongShapeFn {
    fn name(&self) -> &str {
        "wrong_shape"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        // 宣言（入力と同じ shape）と異なる shape を返す契約違反。
        let n: usize = inputs[0].shape().iter().product();
        Tensor::new(vec![0.0; n], &[n]).map_err(AutodiffError::Shape)
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        unreachable!("shape 検証で forward 直後に拒否されるため backward には到達しない")
    }
}

#[test]
fn tape_custom_rejects_forward_output_shape_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let result = tape.custom(Arc::new(WrongShapeFn), &[x]);
    assert!(matches!(result, Err(AutodiffError::Shape(_))));
    // 失敗後もテープは正常に使い続けられる（`Tape::custom` が shape
    // 不一致検出時にノードを追加しないことの間接確認: 破損した状態が
    // 残っていれば後続の演算も失敗するはず）。
    let y = x.sum(None).expect("失敗後もテープは正常なまま");
    let _ = tape.backward(&y).expect("失敗後も backward が正常に動く");
}

/// backward の戻り値長が入力数と一致しない場合は `AutodiffError::
/// Backward`（fail-closed）。
struct WrongBackwardLenFn;

impl CustomFunction for WrongBackwardLenFn {
    fn name(&self) -> &str {
        "wrong_backward_len"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok((*inputs[0]).clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![]) // 入力は 1 個なのに空を返す契約違反
    }
}

#[test]
fn backward_wrong_length_is_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let out = tape
        .custom(Arc::new(WrongBackwardLenFn), &[x])
        .expect("forward は成功する");
    let loss = out.sum(None).expect("全縮約");
    let result = tape.backward(&loss);
    assert!(matches!(result, Err(AutodiffError::Backward(_))));
}

/// backward が返す勾配の shape が入力 shape と食い違う場合は
/// `AutodiffError::Shape`。
struct WrongBackwardShapeFn;

impl CustomFunction for WrongBackwardShapeFn {
    fn name(&self) -> &str {
        "wrong_backward_shape"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok((*inputs[0]).clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(t(vec![1.0, 2.0, 3.0], &[3]))]) // 入力 shape [2] と不一致
    }
}

#[test]
fn backward_wrong_shape_is_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let out = tape
        .custom(Arc::new(WrongBackwardShapeFn), &[x])
        .expect("forward は成功する");
    let loss = out.sum(None).expect("全縮約");
    let result = tape.backward(&loss);
    assert!(matches!(result, Err(AutodiffError::Shape(_))));
}

/// `requires_grad[i] == true` の入力に `None` を返すと勾配欠落として
/// fail-closed に拒否する（§12.4「backward の戻り値」）。
struct MissingGradFn;

impl CustomFunction for MissingGradFn {
    fn name(&self) -> &str {
        "missing_grad"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok((*inputs[0]).clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        _upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![None])
    }
}

#[test]
fn backward_none_for_required_grad_is_fail_closed() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let out = tape
        .custom(Arc::new(MissingGradFn), &[x])
        .expect("forward は成功する");
    let loss = out.sum(None).expect("全縮約");
    let result = tape.backward(&loss);
    assert!(matches!(result, Err(AutodiffError::Backward(_))));
}

/// `requires_grad` 前方伝播（イシュー #1748）: `Tape::var_no_grad` の
/// 入力に対しては `requires_grad[i] == false` が正しく `backward` へ
/// 渡り、`None` を返しても成功する。記録用の `Mutex` で実際に受領した
/// `requires_grad` を検証する。
struct RecordingFn {
    seen_requires_grad: std::sync::Mutex<Vec<Vec<bool>>>,
}

impl CustomFunction for RecordingFn {
    fn name(&self) -> &str {
        "recording"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let a = dense(inputs[0]);
        let b = dense(inputs[1]);
        let data: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        Tensor::new(data, inputs[0].shape()).map_err(AutodiffError::Shape)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        self.seen_requires_grad
            .lock()
            .unwrap()
            .push(requires_grad.to_vec());
        let out = if requires_grad[0] {
            Some(upstream.clone())
        } else {
            None
        };
        // 2 番目の入力（requires_grad == false）は `Some` を返しても
        // 呼び出し元が破棄することを確認する（§12.4「`Some` を返す
        // こと自体は禁止しない」）。
        let _ = inputs;
        Ok(vec![out, Some(upstream.clone())])
    }
}

#[test]
fn requires_grad_is_forwarded_correctly_with_mixed_inputs() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let a = tape.var(&t(vec![1.0, 2.0], &[2]));
    let b_tensor = t(vec![3.0, 4.0], &[2]);
    let b = tape.var_no_grad(&b_tensor);
    let func = Arc::new(RecordingFn {
        seen_requires_grad: std::sync::Mutex::new(Vec::new()),
    });
    let out = tape
        .custom(func.clone(), &[a, b])
        .expect("forward は成功する");
    let loss = out.sum(None).expect("全縮約");
    let grads = tape
        .backward(&loss)
        .expect("backward 成功（None は寄与破棄）");
    assert_eq!(
        func.seen_requires_grad.lock().unwrap().as_slice(),
        &[vec![true, false]]
    );
    assert!(grads.get(&a).unwrap().is_some());
    // `b`（var_no_grad）は構造的に勾配を持たないため `GradientTrackingDisabled`。
    assert!(matches!(
        grads.get(&b),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
}

/// `Tape::backward_accumulate`（#1749）で同一 `Op::Custom` ノードの
/// `backward` を 2 回呼んだ場合、勾配が単純に 2 倍蓄積される（1 回分の
/// 2 回合算と bit 一致）。
#[test]
fn backward_accumulate_doubles_gradient_for_custom_node() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let relu_c = tape
        .custom(Arc::new(CustomRelu), &[x])
        .expect("Tape::custom 登録成功");
    let loss = relu_c.sum(None).expect("全縮約");

    let mut grads = tape.backward(&loss).expect("初回 backward 成功");
    tape.backward_accumulate(&loss, &mut grads)
        .expect("2 回目の蓄積成功");

    let single = tape
        .backward(&loss)
        .expect("単発 backward")
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    let accumulated = grads.get(&x).unwrap().unwrap().clone();

    let single_d = dense(&single);
    let acc_d = dense(&accumulated);
    for (i, (&s, &a)) in single_d.iter().zip(acc_d.iter()).enumerate() {
        assert_eq!((s * 2.0).to_bits(), a.to_bits(), "index {i}");
    }
}

/// backward が実際に 2 回呼ばれることも確認する（呼び出し回数の
/// カウンタで検証。§12.6「backward_accumulate で 2 回呼ばれる」）。
struct CountingReluFn {
    calls: Arc<AtomicUsize>,
}

impl CustomFunction for CountingReluFn {
    fn name(&self) -> &str {
        "counting_relu"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        CustomRelu.forward(inputs)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        CustomRelu.backward(inputs, out_value, upstream, requires_grad)
    }
}

#[test]
fn backward_accumulate_calls_user_backward_twice() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
    let calls = Arc::new(AtomicUsize::new(0));
    let func = Arc::new(CountingReluFn {
        calls: calls.clone(),
    });
    let out = tape.custom(func, &[x]).expect("登録成功");
    let loss = out.sum(None).expect("全縮約");

    let mut grads = tape.backward(&loss).expect("初回 backward");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    tape.backward_accumulate(&loss, &mut grads)
        .expect("2 回目の蓄積");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

/// 別 `Tape` に属する `Var` を混在させると `AutodiffError::
/// TapeMismatch`（`Var::cat` と同型の検査規律）。
#[test]
fn tape_custom_rejects_cross_tape_inputs() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let a = tape_a.var(&t(vec![1.0], &[]));
    let b = tape_b.var(&t(vec![2.0], &[]));
    let result = tape_a.custom(Arc::new(CustomMul), &[a, b]);
    assert!(matches!(result, Err(AutodiffError::TapeMismatch)));
}

/// `Tape: Send` 静的アサーション（`Op::Custom` 追加後もコンパイルを
/// 通ることの確認。§12.6）。`Arc<dyn CustomFunction>: Send` の条件は
/// `CustomFunction: Send + Sync` を trait 自体に課すことで保証される
/// （`fusion_backend_integration.rs::tape_is_send` が既に検証済みの
/// 既存アサーションと同じ形を、本ファイルでも独立に固定する）。
#[test]
fn custom_function_trait_object_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<dyn CustomFunction>>();
}
