//! forward／backward hooks（イシュー #2586）の統合テスト。
//!
//! `autodiff` の公開 API（`Tape::register_backward_hook`／`remove_hook`・
//! `HookHandle`・`nn::ForwardHooked`／`ForwardHookCtx`）のみを経由して、設計記録
//! `docs/autodiff-forward-backward-hooks-design.md` §5〜§8・§14 の契約を固定する。
//! resident 葉への登録拒否のみ `Var` を外部へ露出できないためクレート内
//! テスト（`src/hooks.rs`）で検証する。hook は数値経路を追加しないため
//! CUDA／Metal の実機 parity は対象外（同 §6・§14.7）。

mod common;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fandhe_ai_autodiff::nn::activation::{Relu, Sigmoid};
use fandhe_ai_autodiff::nn::{ForwardHookCtx, ForwardHooked, Identity, Linear, Module, Sequential};
use fandhe_ai_autodiff::{AutodiffError, CustomFunction, HookHandle, Tape, Var};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn dense(x: &Tensor<f32>) -> Vec<f32> {
    let c = x.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    dense(x).into_iter().map(f32::to_bits).collect()
}

type Log = Arc<Mutex<Vec<String>>>;

fn new_log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

fn snapshot(log: &Log) -> Vec<String> {
    log.lock().expect("test log").clone()
}

/// ラベルを記録するだけの hook を作る。
fn recorder(
    log: &Log,
    label: &str,
) -> impl Fn(&Tensor<f32>) -> Result<(), AutodiffError> + Send + Sync + 'static {
    let log = Arc::clone(log);
    let label = label.to_string();
    move |_g| {
        log.lock().expect("test log").push(label.clone());
        Ok(())
    }
}

struct Graph<'t> {
    x: Var<'t>,
    w: Var<'t>,
    h: Var<'t>,
    r: Var<'t>,
    loss: Var<'t>,
}

/// x → matmul(w)=h → (h*h + h) → relu=r → sum=loss。h は fan-out、
/// mul／add は lazy elementwise 連鎖になる。
fn build<'t>(tape: &'t Tape) -> Graph<'t> {
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.75], &[2, 3]));
    let w = tape.var(&t(vec![0.3, -0.2, 0.8, 0.1, -0.5, 0.9], &[3, 2]));
    let h = x.matmul(&w).expect("matmul");
    let a = h.mul(&h).expect("mul");
    let b = a.add(&h).expect("add");
    let r = b.relu();
    let loss = r.sum(None).expect("sum");
    Graph { x, w, h, r, loss }
}

fn grad_bits(tape: &Tape, g: &Graph<'_>) -> (Vec<u32>, Vec<u32>) {
    let grads = tape.backward(&g.loss).expect("backward");
    let gx = grads.get(&g.x).expect("get").expect("x grad");
    let gw = grads.get(&g.w).expect("get").expect("w grad");
    (bits(gx), bits(gw))
}

#[test]
fn fifo_within_node_and_descending_node_order_across_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let log = new_log();
    // 登録順をわざと NodeId 順と逆にする。
    tape.register_backward_hook(&g.x, recorder(&log, "x"))
        .unwrap();
    tape.register_backward_hook(&g.h, recorder(&log, "h1"))
        .unwrap();
    tape.register_backward_hook(&g.r, recorder(&log, "r"))
        .unwrap();
    tape.register_backward_hook(&g.h, recorder(&log, "h2"))
        .unwrap();
    tape.backward(&g.loss).unwrap();
    assert_eq!(snapshot(&log), ["r", "h1", "h2", "x"]);
}

#[test]
fn hook_receives_the_final_gradient_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let seen: Arc<Mutex<Option<Vec<u32>>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&seen);
    tape.register_backward_hook(&g.h, move |gr| {
        *sink.lock().unwrap() = Some(bits(gr));
        Ok(())
    })
    .unwrap();
    let grads = tape.backward(&g.loss).unwrap();
    let expected = bits(grads.get(&g.h).unwrap().expect("h grad"));
    assert_eq!(seen.lock().unwrap().clone(), Some(expected));
}

#[test]
fn err_aborts_remaining_hooks_and_propagates_same_error() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let log = new_log();
    tape.register_backward_hook(&g.r, recorder(&log, "r1"))
        .unwrap();
    tape.register_backward_hook(&g.r, |_| Err(AutodiffError::InvalidArgument("boom".into())))
        .unwrap();
    tape.register_backward_hook(&g.r, recorder(&log, "r3"))
        .unwrap();
    tape.register_backward_hook(&g.h, recorder(&log, "h"))
        .unwrap();
    let err = tape.backward(&g.loss).expect_err("hook の Err が伝播する");
    assert!(
        matches!(&err, AutodiffError::InvalidArgument(m) if m == "boom"),
        "{err:?}"
    );
    assert_eq!(snapshot(&log), ["r1"]);
}

#[test]
fn gradients_are_bit_identical_with_and_without_hooks() {
    let plain = Tape::new_with_ops(common::naive_ops());
    let gp = build(&plain);
    let expected = grad_bits(&plain, &gp);

    let hooked = Tape::new_with_ops(common::naive_ops());
    let gh = build(&hooked);
    let log = new_log();
    for (v, l) in [
        (&gh.x, "x"),
        (&gh.w, "w"),
        (&gh.h, "h"),
        (&gh.r, "r"),
        (&gh.loss, "loss"),
    ] {
        hooked.register_backward_hook(v, recorder(&log, l)).unwrap();
    }
    assert_eq!(grad_bits(&hooked, &gh), expected);
    assert_eq!(snapshot(&log).len(), 5);
}

#[test]
fn backward_accumulate_fires_each_call_and_err_leaves_into_untouched() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let count = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&count);
    let handle = tape
        .register_backward_hook(&g.h, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
    let mut into = tape.backward(&g.loss).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    tape.backward_accumulate(&g.loss, &mut into).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2);
    let before = bits(into.get(&g.x).unwrap().unwrap());

    tape.remove_hook(handle).unwrap();
    tape.register_backward_hook(&g.h, |_| Err(AutodiffError::InvalidArgument("stop".into())))
        .unwrap();
    assert!(tape.backward_accumulate(&g.loss, &mut into).is_err());
    assert_eq!(bits(into.get(&g.x).unwrap().unwrap()), before);
}

#[test]
fn backward_create_graph_fires_first_order_once() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let count = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&count);
    tape.register_backward_hook(&g.h, move |_| {
        c.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    let child = Tape::new_with_ops(common::naive_ops());
    let _ = tape.backward_create_graph(&g.loss, &child).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn unreached_node_does_not_fire() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let side = g.x.mul(&g.x).unwrap(); // loss へ寄与しない枝
    let log = new_log();
    tape.register_backward_hook(&side, recorder(&log, "side"))
        .unwrap();
    tape.register_backward_hook(&g.h, recorder(&log, "h"))
        .unwrap();
    tape.backward(&g.loss).unwrap();
    assert_eq!(snapshot(&log), ["h"]);
}

struct Double;

impl CustomFunction for Double {
    fn name(&self) -> &str {
        "double"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        let d: Vec<f32> = dense(inputs[0]).iter().map(|v| v * 2.0).collect();
        Tensor::new(d, inputs[0].shape()).map_err(AutodiffError::Shape)
    }
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        if !requires_grad[0] {
            return Ok(vec![None]);
        }
        let d: Vec<f32> = dense(upstream).iter().map(|v| v * 2.0).collect();
        Ok(vec![Some(
            Tensor::new(d, inputs[0].shape()).map_err(AutodiffError::Shape)?,
        )])
    }
}

#[test]
fn custom_op_output_node_can_be_hooked() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let y = tape.custom(Arc::new(Double), &[x]).unwrap();
    let loss = y.sum(None).unwrap();
    let log = new_log();
    tape.register_backward_hook(&y, recorder(&log, "custom"))
        .unwrap();
    tape.backward(&loss).unwrap();
    assert_eq!(snapshot(&log), ["custom"]);
}

#[test]
fn registration_is_rejected_for_untracked_and_foreign_vars() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let nograd = tape.var_no_grad(&t(vec![1.0, 2.0], &[2]));
    assert!(matches!(
        tape.register_backward_hook(&nograd, |_| Ok(())),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let detached = x.detach().unwrap();
    assert!(matches!(
        tape.register_backward_hook(&detached, |_| Ok(())),
        Err(AutodiffError::GradientTrackingDisabled)
    ));

    // 別テープの Var は拒否され、相手テープの同 index ノードで誤発火しない。
    let other = Tape::new_with_ops(common::naive_ops());
    let ox = other.var(&t(vec![1.0, 2.0], &[2]));
    let log = new_log();
    assert!(matches!(
        tape.register_backward_hook(&ox, recorder(&log, "foreign")),
        Err(AutodiffError::TapeMismatch)
    ));
    let oloss = ox.sum(None).unwrap();
    other.backward(&oloss).unwrap();
    let loss = x.sum(None).unwrap();
    tape.backward(&loss).unwrap();
    assert!(snapshot(&log).is_empty());
}

#[test]
fn detach_source_hook_still_fires_through_other_paths() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let h = x.mul(&x).unwrap();
    let cut = h.detach().unwrap(); // 勾配は流れない
    let loss = h.add(&cut).unwrap().sum(None).unwrap();
    let log = new_log();
    tape.register_backward_hook(&h, recorder(&log, "h"))
        .unwrap();
    tape.backward(&loss).unwrap();
    assert_eq!(snapshot(&log), ["h"]);
}

#[test]
fn remove_hook_stops_firing_and_keeps_order_of_others() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let log = new_log();
    tape.register_backward_hook(&g.h, recorder(&log, "a"))
        .unwrap();
    let hb = tape
        .register_backward_hook(&g.h, recorder(&log, "b"))
        .unwrap();
    tape.register_backward_hook(&g.h, recorder(&log, "c"))
        .unwrap();
    tape.remove_hook(hb).unwrap();
    tape.backward(&g.loss).unwrap();
    assert_eq!(snapshot(&log), ["a", "c"]);
}

#[test]
fn stale_and_foreign_handles_are_rejected() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0], &[2]));
    let stale: HookHandle = tape.register_backward_hook(&x, |_| Ok(())).unwrap();
    let live: HookHandle = tape.register_backward_hook(&x, |_| Ok(())).unwrap();

    let other = Tape::new_with_ops(common::naive_ops());
    assert!(matches!(
        other.remove_hook(live),
        Err(AutodiffError::TapeMismatch)
    ));

    tape.reset();
    assert!(matches!(
        tape.remove_hook(stale),
        Err(AutodiffError::TapeMismatch)
    ));
}

#[test]
fn reset_drops_all_hooks_even_when_node_ids_are_reused() {
    let mut tape = Tape::new_with_ops(common::naive_ops());
    let count = Arc::new(AtomicUsize::new(0));
    {
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        let h = x.mul(&x).unwrap();
        for v in [x, h] {
            let c = Arc::clone(&count);
            tape.register_backward_hook(&v, move |_| {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap();
        }
    }
    tape.reset();
    // 葉プレフィックス（x）は残り、h と同じ NodeId に別ノードが再び載る。
    let x = tape.leaf(0).expect("leaf prefix");
    let h2 = x.mul(&x).unwrap();
    let loss = h2.sum(None).unwrap();
    tape.backward(&loss).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[test]
fn tape_stays_usable_after_a_hook_panics() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let g = build(&tape);
    let handle = tape
        .register_backward_hook(&g.h, |_| panic!("hook panic (expected in test)"))
        .unwrap();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = tape.backward(&g.loss);
    }));
    assert!(res.is_err());
    tape.remove_hook(handle).unwrap();
    tape.register_backward_hook(&g.x, |_| Ok(())).unwrap();
    tape.backward(&g.loss).expect("panic 後も Tape は一貫状態");
}

#[test]
fn send_bounds_are_preserved() {
    fn assert_send<T: Send>() {}
    assert_send::<Tape>();
    assert_send::<HookHandle>();
    assert_send::<ForwardHooked<Linear>>();
}

// ---------------------------------------------------------------------
// forward hook
// ---------------------------------------------------------------------

fn linear() -> Linear {
    Linear::new(3, 2, true, 11).expect("linear")
}

#[test]
fn forward_hook_fires_once_with_shapes_and_output_value() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let calls = Arc::new(AtomicUsize::new(0));
    let observed: Arc<Mutex<Option<Vec<u32>>>> = Arc::new(Mutex::new(None));
    let (c, o) = (Arc::clone(&calls), Arc::clone(&observed));
    let hooked = ForwardHooked::new(linear(), move |ctx: &ForwardHookCtx<'_>| {
        assert_eq!(ctx.input_shape(), &[2, 3]);
        assert_eq!(ctx.output_shape(), &[2, 2]);
        c.fetch_add(1, Ordering::SeqCst);
        *o.lock().unwrap() = Some(bits(&ctx.output_value()?));
        Ok(())
    });
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.75], &[2, 3]));
    let y = hooked.forward(&tape, &x).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(observed.lock().unwrap().clone(), Some(bits(&y.to_tensor())));
}

/// 出力 Var が引数 tape に属さない場合は hook を呼ばず `TapeMismatch`（panic しない）。
#[test]
fn forward_hook_rejects_output_from_foreign_tape() {
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let hooked = ForwardHooked::new(Identity, move |ctx: &ForwardHookCtx<'_>| {
        c.fetch_add(1, Ordering::SeqCst);
        ctx.output_value().map(|_| ())
    });
    let xb = tape_b.var(&t(vec![1.0, 2.0], &[1, 2]));
    let r = hooked.forward(&tape_a, &xb);
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

type ForwardHookBox = Box<dyn Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync>;

fn sigmoid_loss_grad_bits(wrap: Option<ForwardHookBox>) -> (Vec<u32>, Vec<u32>) {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.75], &[2, 3]));
    let y = match wrap {
        Some(hook) => ForwardHooked::new(Sigmoid, move |c| hook(c))
            .forward(&tape, &x)
            .unwrap(),
        None => Module::forward(&Sigmoid, &tape, &x).unwrap(),
    };
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let out = bits(&loss.to_tensor());
    let grads = tape.backward(&loss).unwrap();
    (out, bits(grads.get(&x).unwrap().unwrap()))
}

#[test]
fn forward_hook_that_does_not_read_values_is_bit_identical() {
    let plain = sigmoid_loss_grad_bits(None);
    let hooked = sigmoid_loss_grad_bits(Some(Box::new(|ctx| {
        let _ = (ctx.input_shape(), ctx.output_shape());
        Ok(())
    })));
    assert_eq!(plain, hooked);
}

#[test]
fn forward_hook_that_reads_values_stays_within_composite_tolerance() {
    let plain = sigmoid_loss_grad_bits(None);
    let hooked = sigmoid_loss_grad_bits(Some(Box::new(|ctx| ctx.output_value().map(|_| ()))));
    // REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）。同一演算列の
    // 実体化時期が変わるだけなので実際には bit 一致のはずだが契約は複合判定。
    for (a, b) in [(&plain.0, &hooked.0), (&plain.1, &hooked.1)] {
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
            let abs = (x - y).abs();
            assert!(
                abs < 1e-5 || abs / x.abs().max(f32::MIN_POSITIVE) < 1e-3,
                "{x} vs {y}"
            );
        }
    }
}

#[test]
fn forward_hook_error_propagates_from_forward() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let hooked = ForwardHooked::new(Relu, |_| Err(AutodiffError::InvalidArgument("fwd".into())));
    let x = tape.var(&t(vec![1.0, -1.0], &[2]));
    let err = hooked.forward(&tape, &x).expect_err("hook の Err");
    assert!(matches!(err, AutodiffError::InvalidArgument(m) if m == "fwd"));
}

#[test]
fn forward_host_fires_with_host_ctx_and_matches_inner_bit_exact() {
    let ops = common::naive_ops();
    let input = t(vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.75], &[2, 3]);
    let inner = linear();
    let expected = inner.forward_host(ops.as_ref(), &input).unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let seen: Arc<Mutex<Option<Vec<u32>>>> = Arc::new(Mutex::new(None));
    let (c, s) = (Arc::clone(&calls), Arc::clone(&seen));
    let hooked = ForwardHooked::new(inner, move |ctx| {
        assert_eq!(ctx.input_shape(), &[2, 3]);
        assert_eq!(ctx.output_shape(), &[2, 2]);
        c.fetch_add(1, Ordering::SeqCst);
        *s.lock().unwrap() = Some(bits(&ctx.output_value()?));
        Ok(())
    });
    let out = hooked.forward_host(ops.as_ref(), &input).unwrap();
    assert_eq!(bits(&out), bits(&expected));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(seen.lock().unwrap().clone(), Some(bits(&expected)));
}

/// `forward_host` を実装しない層（既定の `Unsupported`）。
struct NoHost;

impl Module for NoHost {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(*input)
    }
}

#[test]
fn forward_host_unsupported_inner_does_not_fire_hook() {
    let ops = common::naive_ops();
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let hooked = ForwardHooked::new(NoHost, move |_| {
        c.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let err = hooked
        .forward_host(ops.as_ref(), &t(vec![1.0], &[1]))
        .expect_err("Unsupported");
    assert!(matches!(err, AutodiffError::Backend(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn wrapped_layer_inside_sequential_still_fires_despite_fusion_hints() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let calls = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&calls);
    let seq = Sequential::new()
        .add(ForwardHooked::new(linear(), move |_| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }))
        .add(Relu);
    let x = tape.var(&t(vec![0.5, -1.0, 2.0, 1.5, 0.25, -0.75], &[2, 3]));
    let _ = seq.forward(&tape, &x).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn nested_wrappers_fire_inner_then_outer() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let log = new_log();
    let (l1, l2) = (Arc::clone(&log), Arc::clone(&log));
    let inner = ForwardHooked::new(Relu, move |_| {
        l1.lock().unwrap().push("inner".into());
        Ok(())
    });
    let outer = ForwardHooked::new(inner, move |_| {
        l2.lock().unwrap().push("outer".into());
        Ok(())
    });
    let x = tape.var(&t(vec![1.0, -1.0], &[2]));
    outer.forward(&tape, &x).unwrap();
    assert_eq!(snapshot(&log), ["inner", "outer"]);
}

#[test]
fn delegated_behaviour_matches_inner() {
    let mut plain = linear();
    let mut hooked = ForwardHooked::new(linear(), |_| Ok(()));

    assert_eq!(hooked.parameter_count(), plain.parameter_count());
    assert_eq!(hooked.type_name(), plain.type_name());
    assert_eq!(
        hooked.supports_forward_host(),
        plain.supports_forward_host()
    );
    assert_eq!(hooked.training(), plain.training());
    assert_eq!(hooked.children().len(), plain.children().len());
    assert_eq!(hooked.named_modules().len(), plain.named_modules().len());

    let names = |m: &dyn Module| -> Vec<String> {
        let mut v: Vec<String> = m.named_parameters().into_iter().map(|(n, _)| n).collect();
        v.sort();
        v
    };
    assert_eq!(names(&hooked), names(&plain));

    plain.set_training(false);
    hooked.set_training(false);
    assert_eq!(hooked.training(), plain.training());

    // state_dict 往復と set_parameter（接頭辞なしの透過委譲）。
    let sd: HashMap<String, Tensor<f32>> = hooked.state_dict();
    assert_eq!(
        sd.keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        plain.state_dict().keys().cloned().collect()
    );
    hooked.load_state_dict(plain.state_dict()).unwrap();
    let zeros = t(vec![0.0; 6], &[3, 2]);
    hooked.set_parameter("weight", zeros.clone()).unwrap();
    assert_eq!(bits(hooked.inner().weight()), bits(&zeros));

    // requires_grad／freeze／snapshot。
    assert_eq!(hooked.requires_grad(), plain.requires_grad());
    let snap = hooked.requires_grad_snapshot().unwrap();
    hooked.freeze().unwrap();
    assert!(!hooked.requires_grad());
    hooked.restore_requires_grad_snapshot(&snap).unwrap();
    assert!(hooked.requires_grad());
    hooked.set_requires_grad(false).unwrap();
    assert!(!hooked.requires_grad());

    // inner_mut／into_inner。
    hooked.inner_mut().set_requires_grad(true).unwrap();
    assert!(hooked.into_inner().requires_grad());
}

// ---------------------------------------------------------------------
// 委譲網羅性のソース走査（Module にメソッドが増えたら委譲か非委譲かの判断を強制する）
// ---------------------------------------------------------------------

/// `{` に対応する `}` までの本文（開き括弧の次から閉じ括弧の手前）を返す。
fn brace_body(src: &str, header: &str) -> String {
    let start = src
        .find(header)
        .unwrap_or_else(|| panic!("header not found: {header}"));
    let open = start + src[start..].find('{').expect("open brace");
    let mut depth = 0i32;
    for (i, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return src[open + 1..open + i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces after {header}");
}

/// 本文の深さ 0 に現れる `fn <name>` の名前集合（コメント行は除外）。
fn top_level_fn_names(body: &str) -> std::collections::BTreeSet<String> {
    let mut depth = 0i32;
    let mut out = std::collections::BTreeSet::new();
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        if depth == 0
            && let Some(rest) = trimmed.strip_prefix("fn ")
        {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            out.insert(name);
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
    }
    out
}

fn is_non_delegated_by_design(name: &str) -> bool {
    name.starts_with("as_") || name == "is_pooling"
}

#[test]
fn forward_hooked_delegates_every_module_method_except_fusion_hints() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let module_src = std::fs::read_to_string(root.join("src/nn/module.rs")).expect("module.rs");
    let hook_src =
        std::fs::read_to_string(root.join("src/nn/forward_hook.rs")).expect("forward_hook.rs");

    let trait_fns = top_level_fn_names(&brace_body(&module_src, "pub trait Module"));
    let impl_fns = top_level_fn_names(&brace_body(
        &hook_src,
        "impl<M: Module> Module for ForwardHooked<M>",
    ));

    // 空合格の防止: 既知のメソッドが双方で検出できていること。
    assert!(trait_fns.contains("forward") && trait_fns.contains("as_linear"));
    assert!(impl_fns.contains("forward") && impl_fns.contains("named_parameters"));
    assert!(
        trait_fns.len() > 30,
        "trait の走査が空に近い: {trait_fns:?}"
    );

    let missing: Vec<&String> = trait_fns
        .iter()
        .filter(|n| !impl_fns.contains(*n) && !is_non_delegated_by_design(n))
        .collect();
    assert!(
        missing.is_empty(),
        "Module のメソッドが ForwardHooked で委譲も明示の非委譲もされていない: {missing:?}"
    );
    let extra: Vec<&String> = impl_fns
        .iter()
        .filter(|n| !trait_fns.contains(*n))
        .collect();
    assert!(
        extra.is_empty(),
        "trait に無いメソッドを実装している: {extra:?}"
    );
    // 融合ヒントを誤って委譲していない（委譲すると Sequential の融合で hook が不発になる）。
    let delegated_hints: Vec<&String> = impl_fns
        .iter()
        .filter(|n| is_non_delegated_by_design(n))
        .collect();
    assert!(
        delegated_hints.is_empty(),
        "融合ヒントを委譲している: {delegated_hints:?}"
    );
}

#[test]
fn fn_name_scanner_self_test_detects_added_and_nested_fns() {
    let body = brace_body(
        "pub trait T {\n fn a(&self) {\n  fn inner() {}\n }\n // fn commented()\n fn b();\n}",
        "pub trait T",
    );
    let names = top_level_fn_names(&body);
    assert_eq!(names.into_iter().collect::<Vec<_>>(), ["a", "b"]);
}
