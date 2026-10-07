//! forward／backward hooks の facade 公開（イシュー #2587・親 #2584。ルート #2499 の承認・
//! `docs/autodiff-forward-backward-hooks-design.md` §14.4・§18）の利用例・挙動テスト。
//!
//! 役割: `Tape::register_backward_hook`／`remove_hook`・`HookHandle`・`nn::ForwardHooked`・
//! `nn::ForwardHookCtx` が **facade の import だけ**で使えること、hook が観察専用で勾配・出力を
//! 変えないこと（CPU の本番 ops で hook 有無の bit 一致）、登録の fail-closed 拒否、
//! `ForwardHooked` が inner の `Module` 契約を接頭辞なしで透過することを固定する。
//!
//! **import 契約**: `fandhe_ai` と `std` 以外は import しない（`fandhe_ai_autodiff` を使わない）。
//! tolerance は使わず bit 一致・等値比較のみ。CPU で完結するため `#[ignore]` は付けない。

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fandhe_ai::compat::Sequential as CompatSequential;
use fandhe_ai::nn::{ForwardHookCtx, ForwardHooked, Module, Sequential};
use fandhe_ai::{AutodiffError, HookHandle, Tape, TapeRef, Tensor, Var, tape};

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_slice(data, shape).expect("テスト入力の構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("host tensor")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `x @ weight + bias`（3→2）。`requires_grad` で葉登録を切り替える凍結対応のパラメータ持ち層。
struct Affine {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    training: bool,
    requires_grad: bool,
}

impl Affine {
    fn new() -> Self {
        Self {
            weight: t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]),
            bias: t32(&[0.05, -0.05], &[2]),
            training: true,
            requires_grad: true,
        }
    }
}

impl Module for Affine {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let (w, b) = if self.requires_grad {
            (tape.var(&self.weight), tape.var(&self.bias))
        } else {
            (tape.var_no_grad(&self.weight), tape.var_no_grad(&self.bias))
        };
        input.matmul(&w)?.add(&b)
    }
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".into(), &self.weight), ("bias".into(), &self.bias)]
    }
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let slot = match name {
            "weight" => &mut self.weight,
            "bias" => &mut self.bias,
            _ => return Err(AutodiffError::InvalidArgument(format!("unknown `{name}`"))),
        };
        if slot.shape() != value.shape() {
            return Err(AutodiffError::InvalidArgument("shape".into()));
        }
        *slot = value;
        Ok(())
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
    fn training(&self) -> bool {
        self.training
    }
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.requires_grad = requires_grad;
        Ok(())
    }
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

/// 無状態の tanh 層。
struct CustomTanh;
impl Module for CustomTanh {
    fn forward<'t>(&self, _tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(input.tanh())
    }
}

/// 常に `Err` を返す層（inner が失敗したとき hook が不発であることの確認用）。
struct Failing;
impl Module for Failing {
    fn forward<'t>(&self, _tape: TapeRef<'t>, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(AutodiffError::InvalidArgument("inner failure".into()))
    }
}

fn input() -> Tensor<f32> {
    t32(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], &[2, 3])
}

/// 発火回数を数えるだけの forward hook 付き `Affine`。
fn counted_affine(calls: &Arc<AtomicUsize>) -> ForwardHooked<Affine> {
    let seen = Arc::clone(calls);
    ForwardHooked::new(Affine::new(), move |_ctx: &ForwardHookCtx<'_>| {
        seen.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
}

// ---------------------------------------------------------------------
// backward hook
// ---------------------------------------------------------------------

#[test]
fn backward_hook_observes_final_gradient_bit_identical() {
    let t = tape();
    let x = t.var(&t32(&[1.0, 2.0, -3.0], &[3]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    let seen: Arc<Mutex<Vec<Vec<u32>>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let _h: HookHandle = t
        .register_backward_hook(&x, move |g: &Tensor<f32>| {
            sink.lock().unwrap().push(bits(g));
            Ok(())
        })
        .unwrap();
    let grads = t.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x は loss に寄与する");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0], bits(dx), "hook が見る勾配は Gradients と bit 一致");
}

#[test]
fn backward_hooks_on_same_node_fire_in_registration_order() {
    let t = tape();
    let x = t.var(&t32(&[1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    let order: Arc<Mutex<Vec<u8>>> = Arc::default();
    for id in 1u8..=3 {
        let sink = Arc::clone(&order);
        t.register_backward_hook(&x, move |_g| {
            sink.lock().unwrap().push(id);
            Ok(())
        })
        .unwrap();
    }
    t.backward(&loss).unwrap();
    assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
}

#[test]
fn removed_hook_does_not_fire() {
    let t = tape();
    let x = t.var(&t32(&[1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let (a, b) = (Arc::clone(&calls), Arc::clone(&calls));
    let removed = t
        .register_backward_hook(&x, move |_g| {
            a.fetch_add(10, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
    t.register_backward_hook(&x, move |_g| {
        b.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    t.remove_hook(removed).unwrap();
    t.backward(&loss).unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "残った hook だけが発火する"
    );
}

#[test]
fn backward_hook_error_propagates_as_backward_error() {
    let t = tape();
    let x = t.var(&t32(&[1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    t.register_backward_hook(&x, |_g| Err(AutodiffError::InvalidArgument("stop".into())))
        .unwrap();
    match t.backward(&loss) {
        Err(AutodiffError::InvalidArgument(m)) => assert_eq!(m, "stop"),
        other => panic!(
            "hook の Err が backward の Err になるはず: {:?}",
            other.err()
        ),
    }
}

#[test]
fn registration_is_rejected_fail_closed() {
    let t = tape();
    let other = tape();
    let x = t.var(&t32(&[1.0], &[1]));
    let frozen = t.var_no_grad(&t32(&[1.0], &[1]));
    let foreign = other.var(&t32(&[1.0], &[1]));
    assert!(matches!(
        t.register_backward_hook(&foreign, |_g| Ok(())),
        Err(AutodiffError::TapeMismatch)
    ));
    assert!(matches!(
        t.register_backward_hook(&frozen, |_g| Ok(())),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    // 別テープのハンドルでの解除は TapeMismatch（登録簿は変わらない）。
    let h = other.register_backward_hook(&foreign, |_g| Ok(())).unwrap();
    assert!(matches!(t.remove_hook(h), Err(AutodiffError::TapeMismatch)));
    // 正常系は通る。
    let ok = t.register_backward_hook(&x, |_g| Ok(())).unwrap();
    t.remove_hook(ok).unwrap();
}

#[test]
fn reset_clears_hooks_and_invalidates_old_handles() {
    let mut t = tape();
    let calls = Arc::new(AtomicUsize::new(0));
    let handle = {
        let x = t.var(&t32(&[1.0, 2.0], &[2]));
        let loss = x.mul(&x).unwrap().sum(None).unwrap();
        let seen = Arc::clone(&calls);
        let handle = t
            .register_backward_hook(&x, move |_g| {
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap();
        t.backward(&loss).unwrap();
        handle
    };
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    t.reset();
    assert!(matches!(
        t.remove_hook(handle),
        Err(AutodiffError::TapeMismatch)
    ));
    // reset 後の再計算で旧 hook が誤発火しない（葉は保持され NodeId が再利用されうる）。
    let x = t.leaf(0).expect("葉は reset 後も保持される");
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    t.backward(&loss).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1, "旧 hook は発火しない");
}

/// CPU の本番 ops（matmul・tanh・mul・sum）で、hook あり／なしの勾配が bit 一致する。
#[test]
fn backward_hooks_do_not_change_gradients_bitwise() {
    fn grads_of(with_hooks: bool) -> (Vec<u32>, Vec<u32>) {
        let t = tape();
        let x = t.var(&input());
        let w = t.var(&t32(&[0.1, 0.2, -0.3, 0.4, 0.5, -0.6], &[3, 2]));
        let h = x.matmul(&w).unwrap().tanh();
        let loss = h.mul(&h).unwrap().sum(None).unwrap();
        if with_hooks {
            for v in [&x, &w, &h] {
                t.register_backward_hook(v, |_g| Ok(())).unwrap();
            }
        }
        let grads = t.backward(&loss).unwrap();
        (
            bits(grads.get(&x).unwrap().unwrap()),
            bits(grads.get(&w).unwrap().unwrap()),
        )
    }
    assert_eq!(grads_of(false), grads_of(true));
}

// ---------------------------------------------------------------------
// forward hook
// ---------------------------------------------------------------------

#[test]
fn forward_hook_fires_once_and_ctx_matches_output() {
    type Observed = (Vec<usize>, Vec<usize>, Vec<u32>);
    let seen: Arc<Mutex<Option<Observed>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let hooked = ForwardHooked::new(Affine::new(), move |ctx: &ForwardHookCtx<'_>| {
        let value = ctx.output_value()?;
        let mut slot = sink.lock().unwrap();
        assert!(slot.is_none(), "hook は 1 forward につき 1 回");
        *slot = Some((
            ctx.input_shape().to_vec(),
            ctx.output_shape().to_vec(),
            bits(&value),
        ));
        Ok(())
    });
    let t = tape();
    let x = t.var(&input());
    let y = hooked.forward(TapeRef::from(&t), &x).unwrap();
    let (ishape, oshape, value) = seen.lock().unwrap().clone().expect("hook が発火する");
    assert_eq!(ishape, vec![2, 3]);
    assert_eq!(oshape, vec![2, 2]);
    assert_eq!(
        value,
        bits(&y.to_tensor()),
        "output_value は出力と bit 一致"
    );
}

#[test]
fn forward_hook_without_value_read_keeps_output_bitwise() {
    let t = tape();
    let plain = Affine::new()
        .forward(TapeRef::from(&t), &t.var(&input()))
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let t2 = tape();
    let hooked = counted_affine(&calls)
        .forward(TapeRef::from(&t2), &t2.var(&input()))
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(bits(&plain.to_tensor()), bits(&hooked.to_tensor()));
}

#[test]
fn forward_hook_error_propagates_and_inner_error_skips_hook() {
    let t = tape();
    let x = t.var(&input());
    let failing_hook = ForwardHooked::new(Affine::new(), |_ctx: &ForwardHookCtx<'_>| {
        Err(AutodiffError::InvalidArgument("hook stop".into()))
    });
    match failing_hook.forward(TapeRef::from(&t), &x) {
        Err(AutodiffError::InvalidArgument(m)) => assert_eq!(m, "hook stop"),
        other => panic!(
            "hook の Err が forward の Err になるはず: {:?}",
            other.err()
        ),
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let inner_fails = ForwardHooked::new(Failing, move |_ctx: &ForwardHookCtx<'_>| {
        seen.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    assert!(inner_fails.forward(TapeRef::from(&t), &x).is_err());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "inner が Err なら hook は不発"
    );
}

#[test]
fn forward_hook_inside_nn_sequential_fires_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seq = Sequential::new()
        .add(counted_affine(&calls))
        .add(CustomTanh);
    let t = tape();
    let y = seq.forward(TapeRef::from(&t), &t.var(&input())).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // 包まない構成と出力が bit 一致する。
    let plain = Sequential::new().add(Affine::new()).add(CustomTanh);
    let t2 = tape();
    let y2 = plain
        .forward(TapeRef::from(&t2), &t2.var(&input()))
        .unwrap();
    assert_eq!(bits(&y.to_tensor()), bits(&y2.to_tensor()));
}

#[test]
fn forward_hook_via_compat_sequential_fires_once_per_forward_and_predict() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = CompatSequential::new().add_module(counted_affine(&calls));
    let t = tape();
    let y = model.forward(&t, &t.var(&input())).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // facade `Module` はホスト推論経路を持たないため predict も tape 経路で 1 回発火する。
    let p = model.predict(&input()).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(bits(&y.to_tensor()), bits(&p));
}

// ---------------------------------------------------------------------
// ForwardHooked の透過性
// ---------------------------------------------------------------------

#[test]
fn forward_hooked_is_transparent_for_module_contract() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut hooked = counted_affine(&calls);
    let plain = Affine::new();

    // 名前に接頭辞が付かない・parameter_count／type_name は inner と一致。
    let names: Vec<String> = hooked
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, vec!["weight".to_string(), "bias".to_string()]);
    assert_eq!(hooked.parameter_count(), plain.parameter_count());
    assert_eq!(hooked.type_name(), plain.type_name());
    assert!(hooked.children().is_empty());
    assert!(hooked.named_modules().is_empty());

    // state_dict／load_state_dict の往復。
    let mut state: HashMap<String, Tensor<f32>> = hooked.state_dict();
    assert_eq!(state.len(), 2);
    state.insert("bias".into(), t32(&[9.0, 8.0], &[2]));
    hooked.load_state_dict(state).unwrap();
    assert_eq!(bits(&hooked.inner().bias), bits(&t32(&[9.0, 8.0], &[2])));
    assert!(matches!(
        hooked.set_parameter("nope", t32(&[0.0], &[1])),
        Err(AutodiffError::InvalidArgument(_))
    ));

    // モード・凍結が inner へ届く。
    hooked.set_training(false);
    assert!(!hooked.training());
    assert!(!hooked.inner().training());
    hooked.freeze().unwrap();
    assert!(!hooked.requires_grad());
    hooked.set_requires_grad(true).unwrap();
    assert!(hooked.inner().requires_grad());

    // inner／inner_mut／into_inner。
    hooked.inner_mut().training = true;
    assert!(hooked.training());
    let inner: Affine = hooked.into_inner();
    assert_eq!(bits(&inner.bias), bits(&t32(&[9.0, 8.0], &[2])));
}

#[test]
fn forward_hooked_inside_nn_sequential_keeps_container_semantics() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut seq = Sequential::new()
        .add(counted_affine(&calls))
        .add(CustomTanh);
    let keys: Vec<String> = seq.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(keys, vec!["0.weight".to_string(), "0.bias".to_string()]);
    seq.set_requires_grad(false).unwrap();
    assert!(!seq.requires_grad());
    seq.set_training(false);
    assert!(!seq.training());
    let sd = seq.state_dict();
    seq.load_state_dict(sd).unwrap();
}

/// 利用例（doctest と同型）: backward hook で勾配を観察し、解除する。
#[test]
fn usage_example_backward_hook_then_remove() {
    let t: Tape = tape();
    let x = t.var(&t32(&[1.0, 2.0], &[2]));
    let loss = x.mul(&x).unwrap().sum(None).unwrap();
    let max_abs = Arc::new(Mutex::new(0.0f32));
    let sink = Arc::clone(&max_abs);
    let handle = t
        .register_backward_hook(&x, move |g: &Tensor<f32>| {
            let m = g
                .host_slice()
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()));
            *sink.lock().unwrap() = m;
            Ok(())
        })
        .unwrap();
    t.backward(&loss).unwrap();
    assert_eq!(*max_abs.lock().unwrap(), 4.0);
    t.remove_hook(handle).unwrap();
}
