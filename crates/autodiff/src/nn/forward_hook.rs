//! forward hook（Module ラッパー `ForwardHooked` と観察 ctx `ForwardHookCtx`）。
//!
//! 役割: 層（[`Module`]）の forward 完了後に、観察専用の hook を 1 回呼ぶ
//! 透過ラッパーを提供する（イシュー #2586・PyTorch `register_forward_hook`
//! 相当の Module 単位版。`Var` 単位の forward hook は不採用。設計記録
//! `docs/autodiff-forward-backward-hooks-design.md` §5.3・§14.2・§14.4 P6）。
//! backward 側の hook は `Tape::register_backward_hook`（`crate::hooks`）が担い、
//! 本ファイルは forward 側のみを持つ。
//!
//! # 契約
//!
//! - hook は `Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync + 'static`
//!   の観察専用。ctx は入出力 shape（構築時保持値・実体化しない）と
//!   `output_value()`（出力値の複製。lazy 出力の場合はここで初めて実体化される
//!   observer effect があり、値を読まない hook は出力・勾配に bit 一致の影響を
//!   与えない。§6）だけを公開し、`Var` も入力値も露出しない。
//! - ctx の構築子は `pub(crate)`（P5′: facade ラッパーが autodiff の単一実装を
//!   再利用する前提で、hook 呼び出しと ctx 構築を autodiff 内に閉じる）。
//! - **融合ヒント（`as_*`・`as_relu`・`as_gelu`・`as_softmax`・`is_pooling`）は
//!   委譲しない**（§5.3 の精密化。実装記録 §17 参照）。`Sequential::forward`
//!   （`container.rs`）は `as_linear`／`as_relu` を見て層の `forward` を呼ばず
//!   融合経路へ直結するため、委譲すると包んだ層の hook が無音で不発になる。
//!   既定値（`None`／`false`）のままにして常に [`Module::forward`] 経由で実行し、
//!   hook が必ず発火することを優先する（先例: facade `FacadeModuleAdapter` も
//!   `as_*` を委譲しない）。
//! - パラメータ・状態・子の列挙は inner へ透過委譲する（名前に接頭辞を付けない）。

use std::collections::HashMap;

use fandhe_ai_tensor_core::{BackendOps, Tensor};

use crate::error::AutodiffError;
use crate::nn::module::{Module, RequiresGradSnapshot};
use crate::tape::{NodeId, Op, Tape};
use crate::var::Var;

/// forward hook の本体型（単一所有）。
type ForwardHookFn = dyn Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync;

/// ctx が出力値を得る経路（公開しない 2 経路の区別）。
enum OutputSource<'a> {
    /// tape 経路: 出力ノードを保持し、`output_value` が fallible 実体化で複製する。
    Tape { tape: &'a Tape, node: NodeId },
    /// host 経路（`forward_host`）: 確定済みの出力を借用する。
    Host(&'a Tensor<f32>),
}

/// [`ForwardHooked`] の hook へ渡す観察専用コンテキスト。
///
/// 入出力 shape と出力値の複製のみを公開する（`Var`・入力値は露出しない。
/// 設計記録 §5.5・§14.4 P6）。構築は autodiff 内部（`ForwardHooked` の
/// `Module` 実装）に限る。
pub struct ForwardHookCtx<'a> {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    source: OutputSource<'a>,
}

impl<'a> ForwardHookCtx<'a> {
    /// tape 経路の ctx（`ForwardHooked::forward` から）。
    pub(crate) fn from_tape(
        tape: &'a Tape,
        output: &Var<'a>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        Self {
            input_shape,
            output_shape,
            source: OutputSource::Tape {
                tape,
                node: output.node_id(),
            },
        }
    }

    /// host 経路の ctx（`ForwardHooked::forward_host` から）。
    pub(crate) fn from_host(input_shape: Vec<usize>, output: &'a Tensor<f32>) -> Self {
        Self {
            input_shape,
            output_shape: output.shape().to_vec(),
            source: OutputSource::Host(output),
        }
    }

    /// 入力の shape（構築時に保持した値。実体化しない）。
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// 出力の shape（構築時に保持した値。実体化しない）。
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// 出力値の複製を返す。tape 経路で出力が未実体化（lazy elementwise の末端等）
    /// の場合はここで fallible に実体化する（以後の backward はその値を再利用する。
    /// 値は変わらない）。実体化に失敗した場合は `Err`。
    pub fn output_value(&self) -> Result<Tensor<f32>, AutodiffError> {
        match &self.source {
            OutputSource::Host(t) => Ok((*t).clone()),
            OutputSource::Tape { tape, node } => {
                let nodes = tape.nodes.borrow();
                if matches!(nodes[node.0].op, Op::ResidentLeaf { .. }) {
                    return Err(AutodiffError::InvalidArgument(
                        "ForwardHookCtx::output_value: 出力がデバイス常駐葉でホスト値を持たない"
                            .into(),
                    ));
                }
                Ok(crate::tape::materialize_fallible(&nodes, tape.ops(), *node)?.clone())
            }
        }
    }
}

/// 層 `M` を包み、forward 完了後に観察専用 hook を呼ぶ透過ラッパー。
///
/// `Sequential` 等へ `M` の代わりに積める。hook は構築時に 1 つだけ受け取り
/// （単一所有。後付け・複数登録は設けない。設計記録 §14.4 P6）、`forward`／
/// `forward_host` が inner の結果を得た直後に 1 回呼ぶ。hook の `Err` は
/// そのまま forward の `Err` として伝播する。inner が `Err`（`Unsupported` 含む）
/// の場合 hook は呼ばれない。
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::sync::Arc;
/// use fandhe_ai_autodiff::nn::{ForwardHooked, Linear, Module};
/// use fandhe_ai_autodiff::Tape;
/// use fandhe_ai_tensor_core::Tensor;
///
/// let calls = Arc::new(AtomicUsize::new(0));
/// let c = Arc::clone(&calls);
/// let layer = Linear::new(2, 2, true, 7).unwrap();
/// let hooked = ForwardHooked::new(layer, move |ctx| {
///     assert_eq!(ctx.output_shape(), &[1, 2]);
///     c.fetch_add(1, Ordering::SeqCst);
///     Ok(())
/// });
/// let tape = Tape::new();
/// let x = tape.var(&Tensor::new(vec![3.0f32, 4.0], &[1, 2]).unwrap());
/// let _y = hooked.forward(&tape, &x).unwrap();
/// assert_eq!(calls.load(Ordering::SeqCst), 1);
/// ```
pub struct ForwardHooked<M: Module> {
    inner: M,
    hook: Box<ForwardHookFn>,
}

impl<M: Module> ForwardHooked<M> {
    /// `inner` を包み、forward ごとに `hook` を呼ぶラッパーを構築する。
    pub fn new<F>(inner: M, hook: F) -> Self
    where
        F: Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync + 'static,
    {
        Self {
            inner,
            hook: Box::new(hook),
        }
    }

    /// 包んでいる層への共有参照。
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// 包んでいる層への可変参照。
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// ラッパーを外して層を返す（hook は破棄される）。
    pub fn into_inner(self) -> M {
        self.inner
    }
}

impl<M: Module> Module for ForwardHooked<M> {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let input_shape = input.shape();
        let out = self.inner.forward(tape, input)?;
        let ctx = ForwardHookCtx::from_tape(tape, &out, input_shape, out.shape());
        (self.hook)(&ctx)?;
        Ok(out)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out = self.inner.forward_host(ops, input)?;
        let ctx = ForwardHookCtx::from_host(input.shape().to_vec(), &out);
        (self.hook)(&ctx)?;
        Ok(out)
    }

    fn supports_forward_host(&self) -> bool {
        self.inner.supports_forward_host()
    }

    fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }

    fn training(&self) -> bool {
        self.inner.training()
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.inner.set_requires_grad(requires_grad)
    }

    fn freeze(&mut self) -> Result<(), AutodiffError> {
        self.inner.freeze()
    }

    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn requires_grad_snapshot(&mut self) -> Result<RequiresGradSnapshot, AutodiffError> {
        self.inner.requires_grad_snapshot()
    }

    fn restore_requires_grad_snapshot(
        &mut self,
        snapshot: &RequiresGradSnapshot,
    ) -> Result<(), AutodiffError> {
        self.inner.restore_requires_grad_snapshot(snapshot)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.inner.named_parameters()
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.inner.set_parameter(name, value)
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.inner.state_dict()
    }

    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        self.inner.load_state_dict(state)
    }

    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.inner.children()
    }

    fn named_modules(&self) -> Vec<(String, &dyn Module)> {
        self.inner.named_modules()
    }

    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }
}
