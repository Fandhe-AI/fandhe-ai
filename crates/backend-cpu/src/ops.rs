//! CPU バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `tensor_core::backend_ops::BackendOps` の CPU 実装。既存カーネル
//! （`gemm_blis::gemm_blis_parallel`・`elementwise::{add,mul,relu,exp,tanh}`・
//! `reduction::{sum,max}`）への薄い委譲に徹し、カーネル本体・許容誤差・
//! 境界検査には一切触れない（`.claude/rules/delegation-impl.md` の
//! 実装フロー標準どおり、本ファイルはディスパッチ層のみを追加する）。
//! CPU は常に利用可能なため（`device::CpuDeviceProvider` と同じ位置付け）
//! 全 8 演算とも `Unsupported` を返す経路は持たない（TASK-1.9c の受け入れ
//! 条件「3 バックエンドが呼び分けられる」の参照実装として、CPU は常に
//! 実カーネルを実行できることを保証する）。

use tensor_core::device::{BackendError, Device};
use tensor_core::{Activation, BackendOps, DType, FusionPlan, ShapeError, Tensor};

use crate::gemm_blis::{gemm_blis_bias_act_parallel, gemm_blis_parallel};
use crate::rmsnorm::{self, match_rmsnorm_plan};
use crate::softmax::{self, match_softmax_plan};
use crate::{elementwise, fused_elementwise, reduction};

/// CPU バックエンドの `BackendOps` 実装。状態を持たないゼロサイズ型
/// （CPU カーネルはホストメモリのみを扱い、CUDA `CudaDevice`／Metal
/// `MetalContext` のようなデバイスハンドルを必要としないため）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackendOps;

impl CpuBackendOps {
    /// 新規 `CpuBackendOps` を構築する。
    pub fn new() -> Self {
        Self
    }
}

/// `Tensor::contiguous()` 実体化後もなお `as_slice()` が `None` を返す
/// （契約上到達しないはずだが、`Tensor` 実装のバグに対する fail-safe と
/// して型付きエラーで受ける）場合の変換ヘルパー。shape 不一致ではなく
/// 実行時の契約違反であるため `BackendError::KernelLaunchFailed` を返す
/// （命名を実際のエラー種別に合わせ `gemm_shape_mismatch` から改名。
/// Review 指摘対応）。
fn gemm_contiguity_fail_safe(msg: impl std::fmt::Display) -> BackendError {
    BackendError::KernelLaunchFailed(msg.to_string())
}

impl BackendOps for CpuBackendOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // `gemm_blis_parallel` は contiguous な `&[f32]` を要求する。
        // 非 contiguous（view・broadcast 由来）な入力は `contiguous()` で
        // 実体化してから渡す（`Tensor::as_slice` は非 contiguous では
        // `None` を返す契約。`crates/tensor-core/src/tensor.rs` 参照）。
        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm: lhs not contiguous after contiguous()")
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm: rhs not contiguous after contiguous()")
        })?;

        let mut out = vec![0.0f32; m * n];
        gemm_blis_parallel(a_slice, b_slice, &mut out, m, n, k)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// [`tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、CPU カーネル内で epilogue を融合
    /// する [`gemm_blis_bias_act_parallel`] へ差し替える（TASK-12.1f・
    /// #203）。CUDA は同型のオーバーライド（`backend-cuda::ops::
    /// CudaBackendOps::gemm_bias_act`）をイシュー #599 で追加済み。Metal
    /// はこのオーバーライドを持たずデフォルト実装（非融合合成）を使う
    /// （elementwise 未実装により `bias`／`act` 指定時は `Unsupported` を
    /// 透過的に返す。モジュールドキュメント冒頭・`tensor_core::
    /// backend_ops` のコメント参照）。
    ///
    /// 融合カーネル（[`gemm_blis_bias_act_parallel`]）は bias の行方向
    /// 複製（shape が厳密に `[n]`）のみ対応する。`bias.shape() == [1]` の
    /// ようなブロードキャスト可能だが `[n]` ちょうどでない shape は、
    /// デフォルト実装と同じ `gemm` → `add`（NumPy 互換ブロードキャスト。
    /// `crate::elementwise::add` 経由）→ act の非融合パスへフォールバック
    /// する。こうしないと `BackendOps::gemm_bias_act` の同一メソッドが
    /// CPU では拒否し CUDA／Metal のデフォルト実装では成功するという
    /// バックエンド依存の挙動差が生じる（Issue #203 Review 指摘）。
    /// 非融合パスへ落ちる場合も `gemm_blis_bias_act_parallel` の
    /// `BiasLenMismatch` 検証（カーネル本体アクセス前に検証。REQ-8・
    /// OWASP A03）と同じ順序契約を保つため、`self.gemm` を実行する前に
    /// `tensor_core::broadcast_shape` でブロードキャスト可否のみ先に
    /// 検証する（m×n×k の GEMM 本体を実行してから失敗が判明する、という
    /// 順序にしない）。エラーは `broadcast_shape` のものをそのまま返す
    /// （誤った `ShapeError` variant を独自に組み立てて診断精度を
    /// 落とさない）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out_shape = tensor_core::matmul_out_shape(a.shape(), b.shape())
            .map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        if let Some(bias) = bias
            && bias.shape() != [n]
        {
            // 融合カーネルの対応範囲外（行方向複製の厳密一致ではない
            // shape）。デフォルト実装と同じ 3 段合成へフォールバックする。
            // GEMM 本体を実行する前にブロードキャスト可否を検証する
            // （REQ-8・OWASP A03。`gemm_blis` の `BiasLenMismatch` と
            // 同じ「カーネル本体アクセス前に検証」の順序契約）。
            tensor_core::broadcast_shape(&out_shape, bias.shape())
                .map_err(BackendError::ShapeMismatch)?;
            let mut out = self.gemm(a, b)?;
            out = self.add(&out, bias)?;
            out = match act {
                Activation::None => out,
                Activation::Relu => self.relu(&out)?,
                // `Activation` は `#[non_exhaustive]`（`tensor-core` 側で
                // 将来 variant 追加を見込む）。融合カーネル側
                // （`gemm_blis_bias_act_parallel` 内 `apply_epilogue`）も
                // `_ =>` で未知 variant を静かに無視せず拒否する方針
                // （同ファイル該当コメント参照）と合わせ、ここでも黙って
                // 恒等関数として扱わず明示的に拒否する。
                _ => {
                    return Err(BackendError::Unsupported(format!(
                        "gemm_bias_act: unsupported activation {act:?} in non-fused fallback path"
                    )));
                }
            };
            return Ok(out);
        }

        let a_owned = a.contiguous();
        let b_owned = b.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_bias_act: lhs not contiguous after contiguous()")
        })?;
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_bias_act: rhs not contiguous after contiguous()")
        })?;

        // ここに到達するのは bias が `None`、または shape が厳密に `[n]`
        // の場合のみ（上の早期リターンで他ケースは処理済み）。
        let bias_owned;
        let bias_slice = match bias {
            Some(bias) => {
                bias_owned = bias.contiguous();
                Some(bias_owned.as_slice().ok_or_else(|| {
                    gemm_contiguity_fail_safe(
                        "gemm_bias_act: bias not contiguous after contiguous()",
                    )
                })?)
            }
            None => None,
        };

        let mut out = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(a_slice, b_slice, &mut out, m, n, k, bias_slice, act)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::add(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::mul(a, b).map_err(BackendError::ShapeMismatch)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::relu(a).map_err(BackendError::ShapeMismatch)
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::exp(a).map_err(BackendError::ShapeMismatch)
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        elementwise::tanh(a).map_err(BackendError::ShapeMismatch)
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::sum(a, dim).map_err(reduce_error_to_backend_error)
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        reduction::max(a, dim).map_err(reduce_error_to_backend_error)
    }

    /// [`tensor_core::BackendOps::run_fused`] のデフォルト実装（`Unsupported`
    /// fail-safe）を、CPU 単一パス融合カーネル
    /// [`fused_elementwise::run_fused_elementwise`] へ差し替える。
    ///
    /// # 結線ギャップの経緯（#167 実装時に発見）
    /// TASK-12.1 系列は融合 IR（#163・PR #400）と CPU カーネル本体
    /// （#164・PR #403）の双方をマージ済みだったが、#400 は「CPU 融合実行
    /// への結線は #164 のスコープ」、#403 は「`run_fused` オーバーライドの
    /// 提供元は backend-cpu 側（#163 のスコープ）」としており、双方が
    /// 相手に委ねた結果、本オーバーライドが一度も追加されず
    /// `CpuBackendOps` は常にデフォルト `Unsupported` を返して per-op
    /// フォールバックに倒れていた（融合カーネルが実行系上で一度も起動
    /// しない状態）。TASK-12.2a（本イシュー #167）の受け入れ条件は
    /// 「融合効果の実測記録」であり、この状態では融合条件と非融合条件が
    /// 区別不能で実測が構造的に不可能なため、本イシューの前提ステップと
    /// してここで結線する（新規設計を含まない・両先行イシューの設計文書
    /// が明示的に予定していた結線であることが安全側判断の根拠）。
    ///
    /// # 3 分岐ルーティング（イシュー #607 で拡張）
    /// `match_rmsnorm_plan`／`match_softmax_plan`（いずれも純関数。プランの
    /// op 列・leaf 数・`row_fusion()` の形状を厳密照合する。
    /// `backend-cuda::rmsnorm`／`backend-cuda::softmax` と同一契約の CPU
    /// 側ミラー）で canonical RMSNorm／softmax 融合プランを検出した場合は
    /// それぞれ [`rmsnorm::run_rmsnorm_f32_raw`]／[`softmax::run_softmax_f32`]
    /// へルーティングする。どちらにも一致しない場合（elementwise-only・
    /// 中間軸 softmax 等）は従来どおり [`fused_elementwise::
    /// run_fused_elementwise`] の allowlist 検査へ委ねる（既存 elementwise
    /// 融合経路の挙動は不変）。RMSNorm 判定を先に試す理由は CUDA 側と同じ
    /// （op 列長〈6 vs 8〉が異なるため両方に一致するプランは存在しない）。
    ///
    /// # 呼び出し元
    /// `autodiff::tape` の遅延評価 2 層（`materialize_fallible`／
    /// `materialize_non_fallible`）から `BackendOps::run_fused` 経由で
    /// 呼ばれる。CUDA／Metal は独自の融合カーネル実装（#592/#594/#604）を
    /// 持つため本オーバーライドとは独立。
    ///
    /// # 数値契約
    /// `run_fused_elementwise` は per-op 逐次合成と同一スカラー演算
    /// （`f32::mul_add` を用いない単純四則・超越関数）で構成されるため
    /// 数値は不変。REQ-2 複合判定（相対誤差 1e-3 未満 または絶対誤差
    /// 1e-5 未満）での一致は
    /// `crates/backend-cpu/tests/fused_elementwise_parity.rs` で検証済み。
    /// RMSNorm／softmax 経路の数値一致は `tests/rmsnorm_parity.rs`・
    /// `tests/softmax_parity.rs` で検証する。
    fn run_fused(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        if let Some(hidden) = match_rmsnorm_plan(plan) {
            return self.run_fused_rmsnorm(plan, leaves, hidden);
        }
        if let Some((rows, cols)) = match_softmax_plan(plan) {
            return self.run_fused_softmax(plan, leaves, rows, cols);
        }
        fused_elementwise::run_fused_elementwise(plan, leaves)
    }
}

impl CpuBackendOps {
    /// [`BackendOps::run_fused`] の RMSNorm 一致経路（イシュー #607）。
    /// `match_rmsnorm_plan` が一致した後の dtype／leaf 数／leaf shape の
    /// 起動前 fail-closed 検証（`backend-cuda::ops::CudaBackendOps::
    /// run_fused_rmsnorm` と同じ検査順序）と、
    /// [`rmsnorm::run_rmsnorm_f32_raw`]（`inv_n = 1.0`・`eps = 0.0`・
    /// `w = None`）への委譲を行う。
    fn run_fused_rmsnorm(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        hidden: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: unsupported dtype {:?} (canonical RMSNorm fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: canonical RMSNorm プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("run_fused: rmsnorm input not contiguous"))?;

        let out = rmsnorm::run_rmsnorm_f32_raw(x_slice, None, 0.0, 1.0, 1, hidden)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`BackendOps::run_fused`] の softmax 一致経路（イシュー #607）。
    /// `run_fused_rmsnorm` と同じ起動前 fail-closed 検証パターンを踏襲し、
    /// [`softmax::run_softmax_f32`] を直接呼ぶ。
    fn run_fused_softmax(
        &self,
        plan: &FusionPlan,
        leaves: &[&Tensor<f32>],
        rows: usize,
        cols: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        if plan.dtype() != DType::F32 {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: unsupported dtype {:?} (canonical softmax fusion \
                 kernel supports F32 only)",
                plan.dtype()
            )));
        }
        let [x] = leaves else {
            return Err(BackendError::Unsupported(format!(
                "CpuBackendOps::run_fused: canonical softmax プランは leaf 1 個を要求するが \
                 {} 個が渡された",
                leaves.len()
            )));
        };
        if x.shape() != plan.output_shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: plan.output_shape().to_vec(),
                rhs: x.shape().to_vec(),
            }));
        }

        let x_owned = x.contiguous();
        let x_slice = x_owned
            .as_slice()
            .ok_or_else(|| gemm_contiguity_fail_safe("run_fused: softmax input not contiguous"))?;

        let out = softmax::run_softmax_f32(x_slice, rows, cols)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, plan.output_shape()).map_err(BackendError::ShapeMismatch)
    }
}

/// `reduction::ReduceError`（`Shape`／`EmptyReduction` の 2 variant）を
/// `BackendError` へ写像する。`EmptyReduction` は shape 由来ではない
/// 実行時失敗のため `KernelLaunchFailed` に寄せる（`BackendError` に
/// reduction 専用 variant は設けない。§4.4 の 5 variant + TASK-1.9a/1.9c
/// 拡張の範囲に収める）。
fn reduce_error_to_backend_error(err: reduction::ReduceError) -> BackendError {
    match err {
        reduction::ReduceError::Shape(shape_err) => BackendError::ShapeMismatch(shape_err),
        reduction::ReduceError::EmptyReduction { op } => {
            BackendError::KernelLaunchFailed(format!("empty reduction for op \"{op}\""))
        }
    }
}
