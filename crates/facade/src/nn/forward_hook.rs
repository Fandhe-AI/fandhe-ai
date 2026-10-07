//! facade 版 forward hook ラッパー `ForwardHooked`（イシュー #2587・親 #2584）。
//!
//! 役割: facade 独自の [`crate::nn::Module`] 実装を包み、`forward` 完了直後に
//! 観察専用 hook（PyTorch `register_forward_hook` 相当）を 1 回呼ぶ。autodiff 側
//! `nn::ForwardHooked`（#2586）は autodiff `Module` 専用で、facade 利用者の層は
//! 生の `fandhe_ai_autodiff::Tape` を名指せず直接は包めないため、本型が橋渡しをする
//! （`docs/autodiff-forward-backward-hooks-design.md` §14.4 P5／P5′・§18）。
//!
//! 単一実装の再利用（P5′）: 内部では crate 内専用アダプタ `FacadeModuleAdapter` で
//! inner を autodiff の `Module` へ包み、autodiff 側 `ForwardHooked` に hook の呼び出し・
//! `ForwardHookCtx` の構築を任せる。facade 側は hook を直接呼ばず ctx も構築しない
//! （ドリフトを避ける。`tests/api_surface.rs` が固定する）。`forward` 以外の `Module`
//! メソッド（`named_parameters`・`state_dict` 等）は inner の facade `Module` メソッドへ
//! 名前の接頭辞なしで透過委譲する。
//!
//! 契約:
//! - hook は観察専用（戻り値は `Result<(), AutodiffError>`）。出力を書き換えられず、
//!   `Err` は `forward` の `Err` として伝播する。inner の `forward` が `Err` なら hook は不発。
//! - hook は構築時に 1 つだけ指定する（登録・解除 API は持たない）。
//! - facade `Module` はホスト推論経路を持たない（`FacadeModuleAdapter` は
//!   `supports_forward_host() == false`）ため、`compat::Sequential::predict` 等でも tape 経路で
//!   1 forward につき 1 回発火する。
//! - [`ForwardHookCtx::output_value`] は出力ノードを実体化するため、遅延融合の分割が変わりうる
//!   （observer effect。値を読まない hook では出力は bit 一致。§5.3・§17.3）。
//! - 包めるのは facade `nn::Module` 実装（利用者定義層・`nn::{ModuleList, Sequential,
//!   ModuleDict}`）。autodiff 組み込み層の直接ラップは承認形の範囲外。
//!
//! 公開は `crate::nn::ForwardHooked` と `crate::nn::ForwardHookCtx`（autodiff からの純
//! 再エクスポート。`nn/mod.rs`）のみで、`pub mod hooks` は設けない（P4）。

use std::collections::HashMap;

use fandhe_ai_autodiff::nn::ForwardHooked as AutodiffForwardHooked;
use fandhe_ai_autodiff::nn::Module as AutodiffModule;

use super::ForwardHookCtx;
use crate::nn::module::{FacadeModuleAdapter, Module};
use crate::{AutodiffError, TapeRef, Tensor, Var};

/// facade `Module` を包み、`forward` 後に観察専用 hook を呼ぶラッパー。
///
/// 構築は [`Self::new`]、取り出しは [`Self::inner`]・[`Self::inner_mut`]・
/// [`Self::into_inner`]。`Module` としては inner と同じ振る舞いを透過し、`forward` にだけ
/// hook が加わる。hook の ctx（[`ForwardHookCtx`]）は入力・出力 shape と、必要時のみ出力値
/// （`output_value`）を提供する。
///
/// ```
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use fandhe_ai::nn::{ForwardHooked, Module};
/// use fandhe_ai::{AutodiffError, Tensor, TapeRef, Var};
///
/// struct Double;
/// impl Module for Double {
///     fn forward<'t>(
///         &self,
///         _tape: TapeRef<'t>,
///         input: &Var<'t>,
///     ) -> Result<Var<'t>, AutodiffError> {
///         input.add(input)
///     }
/// }
///
/// let calls = Arc::new(AtomicUsize::new(0));
/// let seen = Arc::clone(&calls);
/// let hooked = ForwardHooked::new(Double, move |ctx| {
///     assert_eq!(ctx.input_shape(), ctx.output_shape());
///     seen.fetch_add(1, Ordering::SeqCst);
///     Ok(())
/// });
/// let tape = fandhe_ai::tape();
/// let x = tape.var(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
/// let y = hooked.forward(TapeRef::from(&tape), &x).unwrap();
/// assert_eq!(y.to_tensor().host_slice().as_ref(), &[2.0, 4.0]);
/// assert_eq!(calls.load(Ordering::SeqCst), 1);
/// ```
pub struct ForwardHooked<M: Module> {
    inner: AutodiffForwardHooked<FacadeModuleAdapter<Box<M>>>,
}

impl<M: Module> ForwardHooked<M> {
    /// `inner` を包み、`forward` 完了直後に `hook` を 1 回呼ぶラッパーを作る。
    pub fn new<F>(inner: M, hook: F) -> Self
    where
        F: Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync + 'static,
    {
        Self {
            inner: AutodiffForwardHooked::new(FacadeModuleAdapter(Box::new(inner)), hook),
        }
    }

    /// 内側の層への参照。
    pub fn inner(&self) -> &M {
        &self.inner.inner().0
    }

    /// 内側の層への可変参照。
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner.inner_mut().0
    }

    /// 包みを外して内側の層を取り出す（hook は破棄される）。
    pub fn into_inner(self) -> M {
        *self.inner.into_inner().0
    }
}

impl<M: Module> Module for ForwardHooked<M> {
    fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AutodiffModule::forward(&self.inner, tape.0, input)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.inner().named_parameters()
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.inner_mut().set_parameter(name, value)
    }

    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.inner().state_dict()
    }

    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        self.inner_mut().load_state_dict(state)
    }

    fn set_training(&mut self, training: bool) {
        self.inner_mut().set_training(training);
    }

    fn training(&self) -> bool {
        self.inner().training()
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        self.inner_mut().set_requires_grad(requires_grad)
    }

    fn freeze(&mut self) -> Result<(), AutodiffError> {
        self.inner_mut().freeze()
    }

    fn requires_grad(&self) -> bool {
        self.inner().requires_grad()
    }

    fn children(&self) -> Vec<(String, &dyn Module)> {
        self.inner().children()
    }

    fn children_mut(&mut self) -> Vec<(String, &mut dyn Module)> {
        self.inner_mut().children_mut()
    }

    fn named_modules(&self) -> Vec<(String, &dyn Module)> {
        self.inner().named_modules()
    }

    fn parameter_count(&self) -> usize {
        self.inner().parameter_count()
    }

    fn type_name(&self) -> &'static str {
        self.inner().type_name()
    }
}
