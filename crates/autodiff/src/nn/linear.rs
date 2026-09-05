//! 全結合層（Linear/Dense）。`nn` モジュール（TASK-9.1a・#91）の第 1 分割。
//!
//! `docs/spec/05-tasks.md` TASK-9.1（REQ-9・M3）に基づき、自作テンソル
//! （`tensor-core`）・自作 autodiff（`Tape`/`Var`）の上に PyTorch
//! `nn.Linear` 相当の層を再実装する。参照実装は PoC-v2-2 の MLP
//! （`docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/mlp.rs`）の
//! `x.matmul(w) → add(bias)` 構成。
//!
//! **`Tape` ライフサイクルとの関係**（`tape.rs` の「学習ループでの運用」
//! 節参照）: `Tape` はステップごとに生成・破棄される前提のため、
//! パラメータの永続的な値（`Tensor<f32>`）を保持する `Linear` 本体と、
//! 特定ステップのテープへ登録した後の `Var` を保持する `LinearVars` を
//! 分離する。呼び出し元は毎ステップ `Linear::bind(&tape)` で
//! `LinearVars` を作り直し、`forward` を呼ぶ。

use fandhe_ai_tensor_core::{
    Activation, BackendOps, ShapeError, Tensor, broadcast_shape, matmul_out_shape,
};

use crate::error::AutodiffError;
use crate::nn::init::{BIAS_SEED_SALT, WEIGHT_SEED_SALT, derive_seed, uniform_init};
use crate::tape::Tape;
use crate::var::Var;

/// 全結合層のパラメータ本体。`weight` は `[in_features, out_features]`
/// （PoC-v2-2 の `x.matmul(w)` 慣習。転置は持たない）、`bias` は
/// `Some` の場合 `[out_features]`。
pub struct Linear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
}

impl Linear {
    /// 決定的シードで `U(-1/√in_features, 1/√in_features)` の一様初期化
    /// を行う（PyTorch `nn.Linear` 既定初期化と同じ有効範囲。
    /// `nn/init.rs` 参照）。`bias: true` で `[out_features]` の bias を
    /// 同じ範囲・独立した導出シードで初期化する（weight/bias のシード
    /// 導出は `nn/init.rs::derive_seed` を参照。「同じ呼び出しシードから
    /// 2 系統を作る」設計上、単純な線形オフセットでは連番呼び出しシード
    /// で層を重ねる使い方の際に系列が衝突しうるため、ビットミキシング
    /// で独立させている）。
    ///
    /// `out_features == 0` は実行時に構築できる空テンソル（`tensor-core`
    /// はサイズ 0 軸を妥当な shape として扱う。`ops_shape.rs` の
    /// `matmul_zero_size_axis` 参照）としてそのまま受理する。一方
    /// `in_features == 0` は `bound = 1/√in_features` が非有限（inf）に
    /// なるため、テンソル生成に進む前に引数として弾く。この失敗は
    /// 「生成済み・生成中のテンソルの shape 不整合」ではなく「コンス
    /// トラクタ引数がそもそも構築不可能」という性質のため、
    /// `tensor-core::ShapeError` の既存 variant（`RankMismatch` 等）は
    /// いずれも意味的に適合せず、`AutodiffError::InvalidArgument` で
    /// 表現する（`error.rs` の doc 参照。review 指摘 #91: 当初
    /// `ShapeError::AxisOutOfRange` へ転用していたが撤回した）。
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Linear, AutodiffError> {
        if in_features == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::new: in_features must be > 0 (1/sqrt(in_features) would be non-finite)"
                    .to_string(),
            ));
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_data = uniform_init(in_features * out_features, bound, weight_seed);
        let weight = Tensor::new(weight_data, &[in_features, out_features])?;
        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data = uniform_init(out_features, bound, bias_seed);
            Some(Tensor::new(bias_data, &[out_features])?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }

    /// 明示的な重み・バイアスから構築する（テスト・将来の safetensors
    /// ロード経路（REQ-7 系。本イシューではスコープ外）向けの入口）。
    /// `weight` は rank 2、`bias` を渡す場合は rank 1 かつ
    /// `bias.shape() == [weight.shape()[1]]` を要求する（A03: 外部由来
    /// パラメータを計算前に検証する契約。`.claude/rules/security.md`）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
    ) -> Result<Linear, AutodiffError> {
        if weight.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            }));
        }
        // `weight.shape()[0]`（in_features）が 0 の場合、`tensor-core::ops_shape::matmul_out_shape`
        // は `lhs[1]==rhs[0]==0` を zero-K パスとして妥当な shape 扱いで許容してしまうため、
        // forward はエラーにならず全要素 0.0 の出力を静かに返す（review 指摘 #91）。
        // `Linear::new` が同条件を `AutodiffError::InvalidArgument` で明示的に拒否しているのに
        // 対し、safetensors ロード等の外部由来パラメータ入口である `from_parameters` がこの
        // 検証を欠くと、壊れた／欠損した checkpoint（shape が `[0, N]` に縮退したもの）を
        // エラーにせず読み込み、学習・推論が常時ゼロ出力のまま進行しうる（A03: 外部由来
        // パラメータを計算前に検証する契約。`.claude/rules/security.md`）。`out_features == 0`
        // は妥当な shape として引き続き許容する（`new` と対称。docstring 参照）。
        if weight.shape()[0] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::from_parameters: weight.shape()[0] (in_features) must be > 0 \
                 (zero-K matmul would silently produce an all-zero output)"
                    .to_string(),
            ));
        }
        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            let out_features = weight.shape()[1];
            if b.shape() != [out_features] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_features],
                }));
            }
        }
        Ok(Linear { weight, bias })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録し、
    /// `forward` を呼べる `LinearVars` を返す。`Tape::var`（`tape.rs`）を
    /// 経由するため、返る `Var` はこの `tape` に属する（クロステープ
    /// 検査の対象になる）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> LinearVars<'t> {
        let weight = tape.var(&self.weight);
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        LinearVars { weight, bias }
    }

    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// [`crate::nn::module::Module::forward_host`]（`Linear` 実装。
    /// `matmul_out_shape` → `ops.gemm` → `ops.add` の非融合合成。
    /// `nn/module.rs` 参照）の epilogue 融合版（イシュー #1218・
    /// `docs/perf/cpu-infer-predict-profile.md`）。[`LinearVars::
    /// forward_with_activation`]（tape 経路。`Var::linear_act` 経由）と
    /// 対をなす tape 不要版で、`y = act(input.matmul(weight) (+ bias))`
    /// を `ops.gemm_bias_act` へ 1 呼び出しで委ねる。
    ///
    /// **呼び出し元が守る契約**: `ops.gemm_bias_act` の融合オーバーライドが
    /// 非融合合成（`gemm` → `add` → `act`）と bit 完全一致するバックエンド
    /// でのみ使うこと。CPU（CPU バックエンドクレートの `CpuBackendOps`）は
    /// `crates/backend-cpu/tests/gemm_epilogue_parity.rs` が MR/NR/MC/KC/NC
    /// 境界を跨ぐ形状グリッドで bit 完全一致を hard assert 済み
    /// （`docs/perf/cpu-gemm-epilogue-fusion.md`「数値一致」節）。CUDA／
    /// Metal の融合オーバーライドはこの一致が未保証のため（`docs/
    /// inference-forward-fixed-cost-design.md` §3.1「段階 A」の
    /// bit-exactness 契約は `Module::forward_host`〈trait レベル・汎用
    /// バックエンド向け〉に適用され続ける）、汎用 `&dyn BackendOps` を
    /// 受け取る `Module::forward_host` 自体はこのメソッドを使わず非融合
    /// のまま維持する。呼び出し元は現状 CPU 固定経路
    /// （`fandhe_ai_facade::compat::sequential::Sequential::
    /// predict_tape_free_with_ops`。Linear→ReLU の先読み結線）に限定する。
    ///
    /// **エラー型の一致契約**: `forward_host` と同じ理由（同ファイル
    /// `forward_host` doc 参照）で、`ops.gemm_bias_act` の `?` に検査を
    /// 任せず `matmul_out_shape`／`broadcast_shape` を先に呼び、shape
    /// 不整合を `forward_host` と同じ `AutodiffError::Shape` として返す
    /// （`AutodiffError::Backend` へ variant が変わらないようにする）。
    pub fn forward_host_with_activation(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
        act: Activation,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out_shape = matmul_out_shape(input.shape(), self.weight.shape())?;
        if let Some(ref bias) = self.bias {
            broadcast_shape(&out_shape, bias.shape())?;
        }
        Ok(ops.gemm_bias_act(input, &self.weight, self.bias.as_ref(), act)?)
    }
}

/// `Linear::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
/// `weight`/`bias` を公開する理由: `Tape::backward` 後に
/// `Gradients::get(&vars.weight)` で勾配を取り出すのは呼び出し側
/// （optimizer。#192・本イシューではスコープ外）の責務であり、
/// `LinearVars` 自身は勾配更新 API を持たない。
pub struct LinearVars<'t> {
    pub weight: Var<'t>,
    pub bias: Option<Var<'t>>,
}

impl<'t> LinearVars<'t> {
    /// `y = input.matmul(weight) (+ bias)`。`input` は `[batch,
    /// in_features]`（2 次元。`Var::matmul` の rank 制約に従う）を
    /// 想定し、出力は `[batch, out_features]`。bias 加算は
    /// `Var::add` の broadcast（`[batch, out_features]` + `[out_features]`）
    /// に委ねるため、bias 勾配の batch 軸縮約は既存の `reduce_to_shape`
    /// 機構（`grad.rs`）でそのまま成立する（TASK-9.1a 計画 §2 参照）。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let y = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    /// [`Self::forward`] の epilogue 融合版（イシュー #1044・`docs/
    /// kernel-fusion.md` §2.2「学習経路への結線」）。`y = act(input.matmul(
    /// weight) (+ bias))` を `Var::linear_act`（`var.rs`）経由で 1 ノード
    /// （`crate::tape::Op::LinearAct`。非公開のためコードスパン表記で
    /// 参照しリンク化しない）として記録し、`BackendOps::
    /// gemm_bias_act`（epilogue 融合カーネル。CPU／CUDA／Metal とも
    /// オーバーライド済み）へ直接委ねる。
    ///
    /// 唯一の呼び出し元は `fandhe_ai_facade::compat::sequential::
    /// Sequential`（次層が `ReLU` かを先読みし、その場合のみ
    /// `Activation::Relu` を渡して `ReLU` 層自体のノード追加をスキップ
    /// する。次層が `ReLU` でなければ `Activation::None` で bias のみ
    /// 融合する）。`Var::linear_act` が bias の broadcast 可否を
    /// `Var::add` と同じ判定で検査するため、本メソッド自体は追加の
    /// shape 検査を持たない。
    pub fn forward_with_activation(
        &self,
        input: &Var<'t>,
        act: fandhe_ai_tensor_core::Activation,
    ) -> Result<Var<'t>, AutodiffError> {
        input.linear_act(&self.weight, self.bias.as_ref(), act)
    }
}

#[cfg(test)]
mod tests {
    //! [`Linear::forward_host_with_activation`]（イシュー #1218・`docs/
    //! perf/cpu-infer-predict-profile.md`）の単体テスト。ここでは
    //! `BackendOps::gemm_bias_act` の**トレイト既定実装**（`gemm` →
    //! `add` → `act` の合成。`tensor-core/src/backend_ops.rs` 参照）を
    //! 経由させるため、`gemm`/`add`/`relu` を素朴に実装するだけの
    //! テスト用 `BackendOps`（CPU バックエンドクレートへの新規依存を
    //! 避ける。同クレートの `gemm_bias_act` 単体テストが同型の
    //! `ComputingMockOps` を使う先例と同じ方式）を使う。

    use super::*;
    use crate::nn::module::Module;
    use fandhe_ai_tensor_core::{Activation, BackendError, BackendOps, device::Device};

    struct ComputingMockOps;

    impl BackendOps for ComputingMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let n = a_shape[1];
            let out: Vec<f32> = a_data
                .iter()
                .enumerate()
                .map(|(idx, x)| x + b_data[idx % n])
                .collect();
            Tensor::new(out, &a_shape).map_err(BackendError::ShapeMismatch)
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out: Vec<f32> = data.iter().map(|x| x.max(0.0)).collect();
            Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
        t.as_slice()
            .expect("test: expected contiguous tensor")
            .to_vec()
    }

    #[test]
    fn forward_host_with_activation_relu_matches_forward_host_then_relu() {
        let ops = ComputingMockOps;
        let linear = Linear::new(4, 6, true, 7).unwrap();
        let input = Tensor::new(
            vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            &[2, 4],
        )
        .unwrap();

        let fused = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap();
        let non_fused = {
            let y = linear.forward_host(&ops, &input).unwrap();
            ops.relu(&y).unwrap()
        };

        assert_eq!(dense_vec(&fused), dense_vec(&non_fused));
    }

    #[test]
    fn forward_host_with_activation_none_matches_forward_host() {
        let ops = ComputingMockOps;
        let linear = Linear::new(3, 5, false, 11).unwrap();
        let input = Tensor::new(vec![0.2_f32, -0.1, 0.4, 0.3, -0.5, 0.1], &[2, 3]).unwrap();

        let fused = linear
            .forward_host_with_activation(&ops, &input, Activation::None)
            .unwrap();
        let non_fused = linear.forward_host(&ops, &input).unwrap();

        assert_eq!(dense_vec(&fused), dense_vec(&non_fused));
    }

    #[test]
    fn forward_host_with_activation_rejects_matmul_shape_mismatch_as_shape_error() {
        // `forward_host` と同じエラー variant 一致契約（`nn/module.rs`
        // doc 参照）: `ops.gemm_bias_act` の `?` に検査を任せず
        // `matmul_out_shape` を先に呼ぶため、shape 不整合は
        // `AutodiffError::Shape` として返る（`AutodiffError::Backend`
        // ではない）。
        let ops = ComputingMockOps;
        let linear = Linear::new(4, 6, true, 7).unwrap();
        // in_features=4 の重みに対し in_features=5 の入力を渡す。
        let input = Tensor::new(vec![0.0_f32; 2 * 5], &[2, 5]).unwrap();

        let err = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));

        // 同じ入力で forward_host も同じ variant を返すことを併せて確認する
        // （新旧 2 メソッド間でエラー variant が食い違わないことの回帰）。
        let err2 = linear.forward_host(&ops, &input).unwrap_err();
        assert!(matches!(err2, AutodiffError::Shape(_)));
    }

    #[test]
    fn forward_host_with_activation_rejects_bias_broadcast_mismatch_as_shape_error() {
        let ops = ComputingMockOps;
        // bias が weight の out_features（6）と食い違う shape になる
        // ケースを直接構築する（`Linear::new` は bias を自動導出するため
        // `from_parameters` で意図的に壊れた bias を渡す）。
        let weight = Tensor::new(vec![0.1_f32; 4 * 6], &[4, 6]).unwrap();
        let bad_bias = Tensor::new(vec![0.0_f32; 3], &[3]).unwrap();
        let linear = Linear {
            weight,
            bias: Some(bad_bias),
        };
        let input = Tensor::new(vec![0.0_f32; 2 * 4], &[2, 4]).unwrap();

        let err = linear
            .forward_host_with_activation(&ops, &input, Activation::Relu)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }
}
