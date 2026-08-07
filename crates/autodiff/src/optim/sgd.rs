//! SGD（momentum・dampening・weight decay・nesterov）optimizer。
//! `optim` モジュール（#193）の第 1 分割。
//!
//! `torch.optim.SGD` の更新則（PyTorch ドキュメントの擬似コード。
//! https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
//! 「Algorithm」節。以下 `g`=grad、`p`=param、`b`=momentum buffer、
//! `μ`=momentum、`τ`=dampening、`λ`=weight_decay）を厳密に踏襲する:
//!
//! ```text
//! g ← ∇f(p) （呼び出し元が渡す grads[i]）
//! if λ ≠ 0: g ← g + λ·p
//! if μ ≠ 0:
//!     if t = 1: b ← g                      （初回 step）
//!     else:     b ← μ·b + (1−τ)·g          （2 回目以降）
//!     if nesterov: g ← g + μ·b
//!     else:        g ← b
//! p ← p − lr·g
//! ```
//!
//! `nn::Linear` はパラメータ本体を不変な `Tensor<f32>` として保持し、
//! ステップごとに `Tensor<f32>` を丸ごと差し替える運用
//! （`nn/linear.rs::Linear::from_parameters` 参照）のため、[`Sgd::step`]
//! も同じ形（更新後の新規 `Tensor<f32>` 列を返す関数型 API）に合わせる。

use tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;

/// SGD のハイパーパラメータ。フィールド名・既定値は `torch.optim.SGD`
/// と揃える（fixture 突合時に対応関係を読み違えないため）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdConfig {
    /// 学習率。負値・非有限値は [`Sgd::new`] で拒否する。
    pub lr: f32,
    /// momentum 係数 `μ`。既定 `0.0`（無効）。
    pub momentum: f32,
    /// dampening `τ`。既定 `0.0`。
    pub dampening: f32,
    /// weight decay `λ`（L2 正則化。decoupled ではない。`torch.optim.SGD`
    /// と同じく `p` に係数を乗じて `g` へ加算する）。既定 `0.0`。
    pub weight_decay: f32,
    /// nesterov momentum を使うか。既定 `false`。
    pub nesterov: bool,
}

impl SgdConfig {
    /// `lr` のみを指定し、他は PyTorch と同じ既定値（momentum 無効・
    /// dampening 0・weight_decay 0・nesterov 無効）で構築する。
    pub fn new(lr: f32) -> SgdConfig {
        SgdConfig {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// momentum `μ` を設定して返す（ビルダー）。
    pub fn with_momentum(mut self, momentum: f32) -> SgdConfig {
        self.momentum = momentum;
        self
    }

    /// dampening `τ` を設定して返す（ビルダー）。
    pub fn with_dampening(mut self, dampening: f32) -> SgdConfig {
        self.dampening = dampening;
        self
    }

    /// weight decay `λ` を設定して返す（ビルダー）。
    pub fn with_weight_decay(mut self, weight_decay: f32) -> SgdConfig {
        self.weight_decay = weight_decay;
        self
    }

    /// nesterov momentum を有効化して返す（ビルダー）。
    pub fn with_nesterov(mut self, nesterov: bool) -> SgdConfig {
        self.nesterov = nesterov;
        self
    }
}

/// SGD optimizer 本体。`step()` をまたいで momentum バッファ
/// （`velocity`）を保持する（`torch.optim.SGD` の `state['momentum_buffer']`
/// 相当）。
///
/// **位置対応契約**: `velocity[i]` は「呼び出し元が `step()` に渡す
/// `params`/`grads` の i 番目」に紐づく。呼び出し元は毎 step 同一の
/// 順序・件数・shape でパラメータを渡す契約とする（`nn::Linear` の
/// `weight`/`bias` のような固定順の呼び出しを想定）。契約違反（件数・
/// shape の変化）は黙って続行せず [`AutodiffError`] を返す（`step()`
/// doc 参照）。
#[derive(Debug)]
pub struct Sgd {
    config: SgdConfig,
    /// momentum バッファ。`momentum == 0.0` の場合は使わないため
    /// `None` のまま（無駄なゼロ初期化を避ける）。初回 `step()` で
    /// `Some(params と同じ shape の Vec)` に遅延初期化する
    /// （PyTorch の「初回は `b ← g`」と同じタイミング）。
    velocity: Option<Vec<Tensor<f32>>>,
}

impl Sgd {
    /// ハイパーパラメータを検証してから構築する。
    ///
    /// 検証基準（PyTorch `torch.optim.SGD.__init__` の `ValueError` と
    /// 同基準 + 自作コアの追加検証）:
    /// - `lr < 0.0` → `InvalidArgument`（PyTorch: `Invalid learning rate`）
    /// - `momentum < 0.0` → `InvalidArgument`（PyTorch: `Invalid momentum value`）
    /// - `weight_decay < 0.0` → `InvalidArgument`（PyTorch: `Invalid weight_decay value`）
    /// - `nesterov && (momentum == 0.0 || dampening != 0.0)` →
    ///   `InvalidArgument`（PyTorch: `Nesterov momentum requires a momentum
    ///   and zero dampening`）
    /// - `lr`/`momentum`/`dampening`/`weight_decay` のいずれかが非有限
    ///   （NaN/inf）→ `InvalidArgument`（PyTorch には無い自作コア側の
    ///   追加検証。非有限なハイパーパラメータは更新式のどこかで必ず
    ///   NaN/inf を伝播させ、実行時に気づきにくい形で学習を破壊する
    ///   ため、構築時に安全側で拒否する）
    pub fn new(config: SgdConfig) -> Result<Sgd, AutodiffError> {
        if !config.lr.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: lr must be finite, got {}",
                config.lr
            )));
        }
        if config.lr < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: invalid lr: {} (must be >= 0)",
                config.lr
            )));
        }
        if !config.momentum.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: momentum must be finite, got {}",
                config.momentum
            )));
        }
        if config.momentum < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: invalid momentum value: {} (must be >= 0)",
                config.momentum
            )));
        }
        if !config.dampening.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: dampening must be finite, got {}",
                config.dampening
            )));
        }
        if !config.weight_decay.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: weight_decay must be finite, got {}",
                config.weight_decay
            )));
        }
        if config.weight_decay < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::new: invalid weight_decay value: {} (must be >= 0)",
                config.weight_decay
            )));
        }
        if config.nesterov && (config.momentum == 0.0 || config.dampening != 0.0) {
            return Err(AutodiffError::InvalidArgument(
                "Sgd::new: Nesterov momentum requires a momentum and zero dampening".to_string(),
            ));
        }
        Ok(Sgd {
            config,
            velocity: None,
        })
    }

    /// `params[i]` と `grads[i]` を対応づけて更新後テンソル列を返す
    /// （`params`/`grads` 自体は変更しない。呼び出し元が
    /// `nn::Linear::from_parameters` 等で戻り値を用いて差し替える）。
    ///
    /// # エラー
    /// - `params.len() != grads.len()` → `InvalidArgument`
    /// - `params[i].shape() != grads[i].shape()` → `Shape`
    ///   （`ShapeError::ShapeMismatch`）
    /// - momentum 有効時、2 回目以降の呼び出しで件数・各要素の shape が
    ///   前回と変化した → `InvalidArgument`（「位置対応契約」違反。
    ///   momentum バッファとの対応が取れなくなるため黙って続行しない）
    pub fn step(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        if params.len() != grads.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Sgd::step: params.len() ({}) != grads.len() ({})",
                params.len(),
                grads.len()
            )));
        }
        for (param, grad) in params.iter().zip(grads.iter()) {
            if param.shape() != grad.shape() {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: param.shape().to_vec(),
                    rhs: grad.shape().to_vec(),
                }));
            }
        }

        let use_momentum = self.config.momentum != 0.0;
        // `use_momentum` ではなく `self.velocity.is_some()` をゲートにする:
        // 現状 `SgdConfig` は構築後不変で `velocity` は `use_momentum` の
        // 場合のみ `Some` になるため両者は常に一致するが、下流のループが
        // `self.velocity.as_ref().map(|v| ... v[i] ...)` で件数検証なしに
        // 添字アクセスするため、ここでの検証は velocity の有無だけで
        // 判定し `use_momentum` の値に依存させない。将来 config が可変に
        // なる等で両者が乖離しても、添字アクセス前に必ず件数・shape が
        // 検証された状態を保つ（Review 指摘: #193 momentum PR）。
        if let Some(velocity) = &self.velocity {
            if velocity.len() != params.len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Sgd::step: params.len() ({}) changed from previous step ({}); \
                     momentum buffer の位置対応契約に違反する",
                    params.len(),
                    velocity.len()
                )));
            }
            for (v, param) in velocity.iter().zip(params.iter()) {
                if v.shape() != param.shape() {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Sgd::step: param shape {:?} changed from previous step's \
                         momentum buffer shape {:?}; 位置対応契約に違反する",
                        param.shape(),
                        v.shape()
                    )));
                }
            }
        }

        let mut new_velocity: Option<Vec<Tensor<f32>>> = if use_momentum {
            Some(Vec::with_capacity(params.len()))
        } else {
            None
        };
        let mut updated = Vec::with_capacity(params.len());

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // `as_slice()` は非 contiguous view で `None` を返す（`tensor.rs`）
            // ため、要素順の走査を保証できる稠密化が要る。`eval::dense_vec`
            // （`crate::eval`）は同じ正規化（`contiguous()` → `as_slice()`、
            // 理論上到達しない `None` 経路は `debug_assert!` 付きフォール
            // バックで吸収）を forward/backward 双方の値計算で既に使って
            // いるため、ここでも同じ稠密化ロジックを再利用し 2 か所に
            // 別実装しない（`eval.rs` doc・coding-rust.md「本番経路で
            // unwrap()/expect() を使わない」参照）。
            let param_data = crate::eval::dense_vec(param);
            let grad_data = crate::eval::dense_vec(grad);

            let prev_v = self
                .velocity
                .as_ref()
                .map(|v| crate::eval::dense_vec(&v[i]));

            let mut out = Vec::with_capacity(param_data.len());
            let mut v_out: Vec<f32> = if use_momentum {
                Vec::with_capacity(param_data.len())
            } else {
                Vec::new()
            };

            for j in 0..param_data.len() {
                let p = param_data[j];
                // weight decay: g ← g + λ·p（`torch.optim.SGD` の
                // 「Algorithm」節 L2 正則化ステップ）。`f32::mul_add` は
                // 使わない: coding-rust.md の FMA 契約統一は GPU カーネル
                // との丸め一致を要求する GEMM 系バックエンド間契約であり、
                // ここは PyTorch CPU 実装（fixture の生成元）の素の乗加算
                // 演算順に揃えて parity を取ることが目的のため対象外
                // （実装計画 §3.3 参照）。
                let mut g = grad_data[j];
                if self.config.weight_decay != 0.0 {
                    g += self.config.weight_decay * p;
                }

                if use_momentum {
                    let b = match &prev_v {
                        Some(prev) => {
                            self.config.momentum * prev[j] + (1.0 - self.config.dampening) * g
                        }
                        None => g,
                    };
                    g = if self.config.nesterov {
                        g + self.config.momentum * b
                    } else {
                        b
                    };
                    v_out.push(b);
                }

                out.push(p - self.config.lr * g);
            }

            // `out`/`v_out` の要素数は `param_data.len()`（= `param` の
            // numel）と常に一致するため `Tensor::new` は理論上失敗しない。
            // それでも本番経路で `unwrap()`/`expect()` は使わない方針
            // （coding-rust.md）のため、`ShapeError` を型付きエラーへ
            // ラップして呼び出し元へ伝播する（到達すれば呼び出し元の
            // 契約違反であり、握り潰さず可視化する）。
            let shape = param.shape().to_vec();
            updated.push(Tensor::new(out, &shape)?);
            if let Some(nv) = &mut new_velocity {
                nv.push(Tensor::new(v_out, &shape)?);
            }
        }

        if use_momentum {
            self.velocity = new_velocity;
        }

        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    #[test]
    fn rejects_negative_lr() {
        let err = Sgd::new(SgdConfig::new(-0.1)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_non_finite_lr() {
        let err = Sgd::new(SgdConfig::new(f32::NAN)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = Sgd::new(SgdConfig::new(f32::INFINITY)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_negative_momentum() {
        let err = Sgd::new(SgdConfig::new(0.1).with_momentum(-0.5)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_non_finite_momentum() {
        let err = Sgd::new(SgdConfig::new(0.1).with_momentum(f32::NAN)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = Sgd::new(SgdConfig::new(0.1).with_momentum(f32::INFINITY)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_non_finite_dampening() {
        let err = Sgd::new(SgdConfig::new(0.1).with_dampening(f32::NAN)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = Sgd::new(SgdConfig::new(0.1).with_dampening(f32::INFINITY)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_negative_weight_decay() {
        let err = Sgd::new(SgdConfig::new(0.1).with_weight_decay(-0.01)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_non_finite_weight_decay() {
        let err = Sgd::new(SgdConfig::new(0.1).with_weight_decay(f32::NAN)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        let err = Sgd::new(SgdConfig::new(0.1).with_weight_decay(f32::INFINITY)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_nesterov_without_momentum() {
        let err = Sgd::new(SgdConfig::new(0.1).with_nesterov(true)).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_nesterov_with_nonzero_dampening() {
        let err = Sgd::new(
            SgdConfig::new(0.1)
                .with_momentum(0.9)
                .with_dampening(0.1)
                .with_nesterov(true),
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn accepts_nesterov_with_momentum_and_zero_dampening() {
        assert!(Sgd::new(SgdConfig::new(0.1).with_momentum(0.9).with_nesterov(true)).is_ok());
    }

    #[test]
    fn rejects_param_grad_count_mismatch() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        let p = tensor(vec![1.0], &[1]);
        let g1 = tensor(vec![0.1], &[1]);
        let g2 = tensor(vec![0.2], &[1]);
        let err = sgd.step(&[&p], &[&g1, &g2]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_param_grad_shape_mismatch() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        let p = tensor(vec![1.0, 2.0], &[2]);
        let g = tensor(vec![0.1], &[1]);
        let err = sgd.step(&[&p], &[&g]).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn rejects_shape_change_between_steps_with_momentum() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
        let p1 = tensor(vec![1.0], &[1]);
        let g1 = tensor(vec![0.1], &[1]);
        sgd.step(&[&p1], &[&g1]).unwrap();

        let p2 = tensor(vec![1.0, 2.0], &[2]);
        let g2 = tensor(vec![0.1, 0.1], &[2]);
        let err = sgd.step(&[&p2], &[&g2]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_param_count_change_between_steps_with_momentum() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
        let p1 = tensor(vec![1.0], &[1]);
        let g1 = tensor(vec![0.1], &[1]);
        sgd.step(&[&p1], &[&g1]).unwrap();

        let p2 = tensor(vec![1.0], &[1]);
        let g2 = tensor(vec![0.1], &[1]);
        let err = sgd.step(&[&p1, &p2], &[&g1, &g2]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn vanilla_sgd_matches_hand_computed_values() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        let p = tensor(vec![1.0], &[1]);
        let g = tensor(vec![0.5], &[1]);
        let out = sgd.step(&[&p], &[&g]).unwrap();
        // p ← p - lr*g = 1.0 - 0.1*0.5 = 0.95
        assert!((out[0].get(&[0]).unwrap() - 0.95).abs() < 1e-6);
    }

    #[test]
    fn momentum_buffer_matches_recursive_hand_computed_sequence() {
        // 実装計画 §5 記載の手計算列: p0=1.0, lr=0.1, μ=0.9,
        // g=[0.5,0.25,0.125] → p=[0.95, 0.88, 0.8045]
        let mut sgd = Sgd::new(SgdConfig::new(0.1).with_momentum(0.9)).unwrap();
        let mut p = tensor(vec![1.0], &[1]);
        let grads = [0.5f32, 0.25, 0.125];
        let expected = [0.95f32, 0.88, 0.8045];

        for (g, exp) in grads.iter().zip(expected.iter()) {
            let grad = tensor(vec![*g], &[1]);
            let out = sgd.step(&[&p], &[&grad]).unwrap();
            let got = out[0].get(&[0]).unwrap();
            assert!((got - exp).abs() < 1e-5, "got={got} expected={exp}");
            p = out.into_iter().next().unwrap();
        }
    }

    #[test]
    fn empty_tensor_step_returns_empty_tensor() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        let p = tensor(vec![], &[0]);
        let g = tensor(vec![], &[0]);
        let out = sgd.step(&[&p], &[&g]).unwrap();
        assert_eq!(out[0].numel(), 0);
    }

    #[test]
    fn non_contiguous_input_is_handled_via_contiguous_normalization() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        // transpose は非 contiguous view を返す（`tensor.rs::transpose`）。
        let p = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .transpose_2d()
            .unwrap();
        // grad は要素ごとに異なる値にする: 全要素同値だと
        // param↔grad の要素対応がずれても出力が偶然一致してしまい、
        // テストが対応関係の正しさを検証できない（Review 指摘: #193
        // momentum PR）。
        let g = tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
        assert!(!p.is_contiguous());
        let out = sgd.step(&[&p], &[&g]).unwrap();
        // p.contiguous() の行優先データは transpose 後の並び [1,3,2,4]。
        // g は contiguous のため元の並び [0.1,0.2,0.3,0.4] のまま
        // （`p[0,0]`↔`g[0,0]`=0.1、`p[0,1]`↔`g[0,1]`=0.2、…）で、
        // 対応がずれていれば期待値と食い違い検出できる。
        assert!((out[0].get(&[0, 0]).unwrap() - (1.0 - 0.1 * 0.1)).abs() < 1e-6);
        assert!((out[0].get(&[0, 1]).unwrap() - (3.0 - 0.1 * 0.2)).abs() < 1e-6);
        assert!((out[0].get(&[1, 0]).unwrap() - (2.0 - 0.1 * 0.3)).abs() < 1e-6);
        assert!((out[0].get(&[1, 1]).unwrap() - (4.0 - 0.1 * 0.4)).abs() < 1e-6);
    }

    #[test]
    fn non_contiguous_grad_is_handled_via_contiguous_normalization() {
        let mut sgd = Sgd::new(SgdConfig::new(0.1)).unwrap();
        // param 側は contiguous のまま、grad 側のみ非 contiguous
        // （transpose view）にする。上のテストが param 側のみを非
        // contiguous にしていたため、grad 側の正規化経路は未検証
        // だった（Review 指摘: #193 momentum PR）。
        let p = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let g = tensor(vec![0.1, 0.2, 0.3, 0.4], &[2, 2])
            .transpose_2d()
            .unwrap();
        assert!(!g.is_contiguous());
        let out = sgd.step(&[&p], &[&g]).unwrap();
        // g.contiguous() の行優先データは transpose 後の並び
        // [0.1,0.3,0.2,0.4]。
        assert!((out[0].get(&[0, 0]).unwrap() - (1.0 - 0.1 * 0.1)).abs() < 1e-6);
        assert!((out[0].get(&[0, 1]).unwrap() - (2.0 - 0.1 * 0.3)).abs() < 1e-6);
        assert!((out[0].get(&[1, 0]).unwrap() - (3.0 - 0.1 * 0.2)).abs() < 1e-6);
        assert!((out[0].get(&[1, 1]).unwrap() - (4.0 - 0.1 * 0.4)).abs() < 1e-6);
    }
}
