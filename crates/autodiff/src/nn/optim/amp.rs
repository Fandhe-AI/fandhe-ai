//! 損失スケーリング（loss scaling）・unscale・inf/nan 検出のコア関数
//! （AMP: Automatic Mixed Precision。親イシュー #1625・本イシュー #1721。
//! `docs/spec/04-requirements.md` REQ-9 2026-09-12 追記の Tier 2「AMP」）。
//!
//! f16 forward で勾配が underflow するのを防ぐため、loss を大きな
//! `scale` 倍してから backward し、得られた勾配を optimizer step の
//! 直前に `scale` で割り戻す（PyTorch `torch.cuda.amp.GradScaler` と
//! 同一の考え方）。本モジュールは **`Tensor<f32>` へ実体化済みの勾配**
//! に対する後処理のみを扱う純関数・純データ構造の集合であり、新規
//! `Op`／`BackendOps` メソッド／VJP は一切追加しない（`scale_loss` は
//! 既存 [`Var::mul`] の合成のみで実装する）。
//!
//! **真の混合精度（f16 forward・f32 master weight）は対象外**
//! （`Var`／`Tape` の dtype 一般化は `docs/backend-dtype-dispatch-design.md`
//! §8 で明示的にスコープ外とされている別軸の変更のため）。本モジュールは
//! f32 勾配列に対するスケーリング／unscale／非有限検出のみを提供する。
//!
//! `nn/optim/mod.rs` の適用順序契約（backward → unscale → clip →
//! optimizer step）における「unscale」ステップの実体がこのモジュール
//! （[`unscale_grads`]／[`GradScaler`]）である。facade（`fandhe_ai::optim`）
//! への公開・識別子の再エクスポートはイシュー #1722 で完了済み（純
//! 再エクスポート。`crates/facade/src/optim.rs` 参照）。本モジュールは
//! クレート内実装（`crate::nn::optim` 経由）に留まる。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::var::Var;

/// `scale` の共通検証（有限かつ正）。`scale_loss`／`scale_grads`／
/// `unscale_grads` のいずれも「0 除算・非有限伝播を未然に防ぐ」
/// fail-closed 方針を共有するため 1 箇所へ集約する。
fn validate_scale(scale: f32, caller: &str) -> Result<(), AutodiffError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{caller}: scale must be finite and > 0.0, got {scale}"
        )));
    }
    Ok(())
}

/// 損失を `scale` 倍する（AMP の順伝播側ステップ）。
///
/// 実装は既存 [`Var::mul`] とスカラー葉の合成のみ（`loss.mul(&scale_var)`）
/// であり、`Var::mul` の既存 VJP がそのまま `d(loss*scale)/dloss = scale`
/// を逆伝播へ伝えるため、新規 `Op`／VJP は不要（本モジュール冒頭 doc
/// 参照）。
///
/// **スカラー葉は `Tape::reset` をまたいで蓄積しない**: 本関数が
/// `loss.tape().var(&Tensor::scalar(scale))` で登録する葉は、必ず forward の
/// 他の演算（`matmul`／`mse_loss` 等）より後に呼ばれる（`loss` が既に
/// 計算済みでなければ呼びようがないため）。`Tape::reset` は「最初の
/// 非葉ノードが記録される *前* に登録した葉」のみを保持し、それ以降に
/// 登録された葉は reset のたびに破棄する（`tape.rs::Tape::reset` doc）
/// ため、本関数の葉は step ごとに無限蓄積しない。
///
/// `loss` の shape は強制しない（`Var::mul` の NumPy 互換ブロード
/// キャストにより非スカラーの `loss` でも動作するが、通常は
/// `mse_loss` 等の縮約後スカラーを渡す想定）。
///
/// # Errors
///
/// `scale` が非有限または 0 以下の場合は `AutodiffError::InvalidArgument`
/// を返す（fail-closed。非有限・非正のスケールは以後の逆伝播・unscale
/// を無意味にするため、演算グラフへ記録する前に弾く）。
pub fn scale_loss<'t>(loss: &Var<'t>, scale: f32) -> Result<Var<'t>, AutodiffError> {
    validate_scale(scale, "scale_loss")?;
    let scale_var = loss.tape().var(&Tensor::scalar(scale));
    loss.mul(&scale_var)
}

/// 勾配列（`Gradients::get` 等で取り出した `&Tensor<f32>` 列）の各要素を
/// `scale` 倍する。`unscale_grads` の逆写像として使う想定（両者は
/// `scale` が 2 のべき乗のとき `f32` 乗算・除算の性質上ほぼ厳密な
/// 逆写像になる。`unscale_grads` doc 参照）。
///
/// # Errors
///
/// `scale` が非有限または 0 以下の場合は `AutodiffError::InvalidArgument`
/// を返す。
pub fn scale_grads(
    tensors: &[&Tensor<f32>],
    scale: f32,
) -> Result<Vec<Tensor<f32>>, AutodiffError> {
    validate_scale(scale, "scale_grads")?;
    let mut out = Vec::with_capacity(tensors.len());
    for t in tensors {
        let values = t.host_slice();
        let scaled: Vec<f32> = values.iter().map(|&v| v * scale).collect();
        out.push(Tensor::from_slice(&scaled, t.shape())?);
    }
    Ok(out)
}

/// [`unscale_grads`] の結果。
pub struct UnscaleResult {
    /// unscale 後の勾配（`grads` と同じ順序・shape）。
    pub grads: Vec<Tensor<f32>>,
    /// `grads`（unscale 後）のいずれかの要素に非有限値（NaN/Inf）が
    /// 含まれていたかどうか。unscale 前の生の勾配（overflow していた
    /// 可能性がある値）を走査した結果であり、unscale の除算自体が新たに
    /// 非有限を生む場合（`scale` が非有限に近いほど小さい等）も含めて
    /// 検出する。
    pub found_non_finite: bool,
}

impl UnscaleResult {
    /// このステップの optimizer step（clip も含む）をスキップすべきか。
    /// PyTorch `GradScaler` が非有限勾配の step を internally skip する
    /// のと同じ判断を明示的に呼び出し元へ返す（`found_non_finite` の
    /// 別名アクセサ）。
    pub fn should_skip_step(&self) -> bool {
        self.found_non_finite
    }
}

/// 勾配列を `scale` で割り戻しつつ（AMP の逆伝播側ステップ）、1 パスで
/// 非有限値の有無を検出する（PyTorch
/// `torch._amp_foreach_non_finite_check_and_unscale_` と同型の複合演算）。
///
/// unscale の式は `v / scale` に固定する（`v * (1.0 / scale)` は
/// 採用しない）。`scale` が 2 のべき乗のときは [`scale_grads`] の
/// 厳密な逆写像になる（over/underflow を除き `f32` 除算は 2 のべき乗
/// による乗算の丸めと同一の指数部シフトのみで、仮数部の丸めが生じ
/// ないため）。2 のべき乗以外の `scale` では丸め差が生じうる。
///
/// **非有限値を含んでいても `Err` にしない**（[`crate::nn::optim::clip::global_grad_norm`]
/// が非有限勾配で `Err` を返す既存規約とは意図的に異なる）。AMP では
/// overflow による非有限勾配の出現は正常な運用パスであり、呼び出し側
/// は [`UnscaleResult::should_skip_step`] で判定してこの step の
/// clip／optimizer step をスキップし、`GradScaler` 側のスケールを
/// backoff させる（`GradScaler::update` 参照）——これが「skip 判定は
/// clip より前に行う」適用順序契約（`nn/optim/mod.rs` doc）の理由。
///
/// # Errors
///
/// `scale` が非有限または 0 以下の場合は `AutodiffError::InvalidArgument`
/// を返す（fail-closed。この検証は非有限勾配の検出とは独立で、
/// 呼び出し側の `scale` 引数そのものの誤りを早期に弾く）。
pub fn unscale_grads(grads: &[&Tensor<f32>], scale: f32) -> Result<UnscaleResult, AutodiffError> {
    validate_scale(scale, "unscale_grads")?;
    let mut out = Vec::with_capacity(grads.len());
    let mut found_non_finite = false;
    for grad in grads {
        let values = grad.host_slice();
        let mut unscaled = Vec::with_capacity(values.len());
        for &v in values.iter() {
            let u = v / scale;
            if !u.is_finite() {
                found_non_finite = true;
            }
            unscaled.push(u);
        }
        out.push(Tensor::from_slice(&unscaled, grad.shape())?);
    }
    Ok(UnscaleResult {
        grads: out,
        found_non_finite,
    })
}

/// 複数テンソルを横断して非有限値（NaN/Inf）が含まれるかを検出する
/// （[`unscale_grads`] と独立に単体で使える版。空スライスは `false`）。
pub fn has_non_finite(tensors: &[&Tensor<f32>]) -> bool {
    tensors
        .iter()
        .any(|t| t.host_slice().iter().any(|v| !v.is_finite()))
}

/// [`GradScaler`] のハイパーパラメータ。既定値は PyTorch
/// `torch.cuda.amp.GradScaler` の既定と同一
/// （`init_scale=2**16`・`growth_factor=2.0`・`backoff_factor=0.5`・
/// `growth_interval=2000`）。いずれも 2 のべき乗を維持する値であり、
/// [`scale_grads`]／[`unscale_grads`] の丸め誤差（doc 参照）を最小化する。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradScalerConfig {
    pub init_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u64,
}

impl Default for GradScalerConfig {
    fn default() -> GradScalerConfig {
        GradScalerConfig {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }
}

/// 損失スケーリングのスケール値更新契約を保持する状態機械
/// （PyTorch `torch.cuda.amp.GradScaler` 相当）。
///
/// **1 step の使い方**:
/// `scaler.scale_loss(&loss)` → `Tape::backward` →
/// `scaler.unscale(&grads)` で `UnscaleResult` を得る →
/// `should_skip_step()` が `true` ならこの step の clip・optimizer step
/// を **両方スキップ**する（`clip_grad_norm` は非有限勾配で `Err` を
/// 返すため、非有限検出より前に clip を呼ぶと学習ループ全体が失敗する。
/// `nn/optim/mod.rs` doc「適用順序契約」参照）→ `false` なら
/// `clip_grad_norm` → optimizer step の順に進める → 最後に必ず
/// `scaler.update(found_non_finite)` を呼びスケールを更新する。
pub struct GradScaler {
    config: GradScalerConfig,
    scale: f32,
    growth_tracker: u64,
}

impl GradScaler {
    /// [`GradScalerConfig`] を検証して構築する（`AdamW::new` と同様、
    /// テンソル生成に進む前に構築不可能な引数を弾く。
    /// `error.rs::AutodiffError::InvalidArgument` doc 参照）。
    ///
    /// # Errors
    ///
    /// - `init_scale` が有限かつ正でない
    /// - `growth_factor` が有限かつ `1.0` より大きくない
    ///   （`1.0` 以下では成長条件が意味を持たない）
    /// - `backoff_factor` が `(0.0, 1.0)` の範囲外
    ///   （`1.0` 以上では backoff が縮小にならず、`0.0` 以下では
    ///   scale が即座に潰れる）
    /// - `growth_interval` が `0`（`0` 回連続 clean で成長する定義は
    ///   意味を持たない）
    pub fn new(config: GradScalerConfig) -> Result<GradScaler, AutodiffError> {
        if !config.init_scale.is_finite() || config.init_scale <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "GradScaler::new: init_scale must be finite and > 0.0, got {}",
                config.init_scale
            )));
        }
        if !config.growth_factor.is_finite() || config.growth_factor <= 1.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "GradScaler::new: growth_factor must be finite and > 1.0, got {}",
                config.growth_factor
            )));
        }
        if !(config.backoff_factor.is_finite()
            && config.backoff_factor > 0.0
            && config.backoff_factor < 1.0)
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "GradScaler::new: backoff_factor must be finite and in (0.0, 1.0), got {}",
                config.backoff_factor
            )));
        }
        if config.growth_interval == 0 {
            return Err(AutodiffError::InvalidArgument(
                "GradScaler::new: growth_interval must be >= 1, got 0".to_string(),
            ));
        }
        let scale = config.init_scale;
        Ok(GradScaler {
            config,
            scale,
            growth_tracker: 0,
        })
    }

    /// 現在のスケール値。
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// 直近の `backoff`／`growth` からの連続 clean step 数
    /// （`growth_interval` に到達すると growth し 0 へリセットされる）。
    pub fn growth_tracker(&self) -> u64 {
        self.growth_tracker
    }

    /// [`scale_loss`] へ現在のスケール値を渡す薄いラッパー。
    pub fn scale_loss<'t>(&self, loss: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        scale_loss(loss, self.scale)
    }

    /// [`unscale_grads`] へ現在のスケール値を渡す薄いラッパー。
    pub fn unscale(&self, grads: &[&Tensor<f32>]) -> Result<UnscaleResult, AutodiffError> {
        unscale_grads(grads, self.scale)
    }

    /// この step の unscale 結果（[`UnscaleResult::found_non_finite`]）に
    /// 基づき、次 step のスケール値を更新する（PyTorch `GradScaler.update`
    /// 相当）。
    ///
    /// - `found_non_finite == true`: `scale *= backoff_factor`・
    ///   `growth_tracker` を `0` へリセットする。backoff 後の scale が
    ///   `0.0`・非正規化数・非有限になる場合は以後の unscale が
    ///   0 除算・無意味な値になるため `Err` を返す（fail-closed）。
    /// - `found_non_finite == false`: `growth_tracker` を 1 増やし、
    ///   `growth_interval` に達したら `scale *= growth_factor` して
    ///   `growth_tracker` を `0` へリセットする。ただし成長後の scale が
    ///   非有限になる場合は **成長をスキップして scale を据え置く**
    ///   （`growth_tracker` は `0` へリセットする。無限に近い scale へ
    ///   増殖させて次 step で即座に backoff させるより、据え置いて
    ///   様子を見るほうが安全側の挙動であるため）。
    ///
    /// # Errors
    ///
    /// backoff の結果 `scale` が `0.0`・非正規化数・非有限になった場合。
    pub fn update(&mut self, found_non_finite: bool) -> Result<(), AutodiffError> {
        if found_non_finite {
            let next = self.scale * self.config.backoff_factor;
            if !next.is_finite() || next <= 0.0 || next.is_subnormal() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "GradScaler::update: backoff により scale が不正な値になった: {next}"
                )));
            }
            self.scale = next;
            self.growth_tracker = 0;
            return Ok(());
        }

        self.growth_tracker += 1;
        if self.growth_tracker >= self.config.growth_interval {
            let grown = self.scale * self.config.growth_factor;
            if grown.is_finite() {
                self.scale = grown;
            }
            // 非有限になる場合は成長をスキップし scale を据え置く
            // （doc 参照）。tracker はどちらの場合もリセットする。
            self.growth_tracker = 0;
        }
        Ok(())
    }
}
