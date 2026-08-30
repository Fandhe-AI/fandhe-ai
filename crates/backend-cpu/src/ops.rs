//! CPU バックエンドの `BackendOps` 実装（TASK-1.9c・#46）。
//!
//! `fandhe_ai_tensor_core::backend_ops::BackendOps` の CPU 実装。既存カーネル
//! （`gemm_blis::gemm_blis_parallel`・`elementwise::{add,mul,relu,exp,tanh}`・
//! `reduction::{sum,max}`）への薄い委譲に徹し、カーネル本体・許容誤差・
//! 境界検査には一切触れない（`.claude/rules/delegation-impl.md` の
//! 実装フロー標準どおり、本ファイルはディスパッチ層のみを追加する）。
//! CPU は常に利用可能なため（`device::CpuDeviceProvider` と同じ位置付け）
//! 全 8 演算とも `Unsupported` を返す経路は持たない（TASK-1.9c の受け入れ
//! 条件「3 バックエンドが呼び分けられる」の参照実装として、CPU は常に
//! 実カーネルを実行できることを保証する）。

use std::sync::OnceLock;

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{
    Activation, BackendOps, DType, FusionPlan, MseReduction, SgdStepConfig, ShapeError, Tensor,
    require_same_shape,
};

use crate::gemm_blis::{gemm_blis_bias_act_parallel, gemm_blis_parallel};
use crate::memory::{CpuBufferHandle, CpuMemory};
use crate::rmsnorm::{self, match_rmsnorm_plan};
use crate::softmax::{self, match_softmax_plan};
use crate::{elementwise, fused_elementwise, mse, reduction};

/// `CpuBackendOps` が `MemoryOps` を実装するための、プロセスワイドに共有
/// する単一 `CpuMemory`（イシュー #935・`docs/device-resident-update-design.md`
/// §3.3d「`AllocationTracker` の計測系列単一化」）。
///
/// `CpuBackendOps` は `#[derive(Debug, Default, Clone, Copy)] pub struct
/// CpuBackendOps;`（unit struct）であり、`fandhe-ai-backend-cpu` として
/// crates.io へ公開済みのためフィールド追加は破壊的変更になりうる
/// （実装計画 §3.2「CPU は `CpuBackendOps` が unit struct で公開済みの
/// ためフィールド追加不可」）。そのためプロセスワイド `static` で
/// `AllocationTracker` の計測系列を共有する（`CpuMemory::new()` を毎回
/// 呼ぶと `Arc<AllocationTracker>` が呼び出しごとに新規生成され、
/// `sgd_step_device` の一連の `alloc_zeroed`／`upload`／`download` 呼び出し
/// 間でピーク計測が繋がらなくなるため）。
fn shared_cpu_memory() -> &'static CpuMemory {
    static SHARED: OnceLock<CpuMemory> = OnceLock::new();
    SHARED.get_or_init(CpuMemory::new)
}

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

/// [`MemoryOps`] の CPU 実装（イシュー #935）。`shared_cpu_memory()`
/// （プロセスワイド共有 `CpuMemory`）へ委譲する薄いラッパー。
impl MemoryOps for CpuBackendOps {
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
        shared_cpu_memory().alloc_zeroed(shape)
    }

    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
        shared_cpu_memory().upload(tensor)
    }

    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
        shared_cpu_memory().download(buffer)
    }
}

impl BackendOps for CpuBackendOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    /// `CpuBackendOps` 自身が [`MemoryOps`] を実装する（上記 `impl
    /// MemoryOps for CpuBackendOps`）ため、`self` をそのまま返す
    /// （イシュー #935）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        Some(self)
    }

    /// SGD の 1 パラメータ分の更新を in-place で実行する（イシュー #935・
    /// `docs/device-resident-update-design.md` §3.2・§5.2）。CPU は
    /// 「デバイス」がホストメモリそのものであるため、`downcast_handle_mut`
    /// で取り出した `Vec<f32>` を直接書き換えるだけで完結する（転送コスト
    /// ゼロ）。
    ///
    /// 更新式の項順序は `fandhe_ai_autodiff::optim::sgd::Sgd::step`（ホスト
    /// 参照実装）と同一（weight_decay → momentum〈`is_first_step` で
    /// `b ← g` 分岐〉→ nesterov → 減算）。丸えは `.claude/rules/
    /// coding-rust.md` の FMA 契約統一方針に従い `f32::mul_add` を使う
    /// （GEMM 系 CPU 参照実装と同じく、CUDA `fmaf`／Metal
    /// `fma`〈`shaders/sgd.metal`〉と丸めを揃えるため。`Sgd::step` 自身は
    /// PyTorch 参照 fixture との parity を優先し `mul_add` を使わない別の
    /// 契約を持つ〈`sgd.rs` 該当コメント参照〉が、本メソッドは 3
    /// バックエンド間一致が目的のため対象が異なる）。
    fn sgd_step_device(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        if param.device() != Device::Cpu || grad.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        if param.shape() != grad.shape() {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: param.shape().to_vec(),
                rhs: grad.shape().to_vec(),
            }));
        }
        let grad_handle = grad
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;

        let use_momentum = config.momentum != 0.0;
        let mut velocity_handle = match velocity {
            Some(v) => {
                if v.device() != Device::Cpu {
                    return Err(BackendError::DeviceMismatch);
                }
                if v.shape() != param.shape() {
                    return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                        lhs: param.shape().to_vec(),
                        rhs: v.shape().to_vec(),
                    }));
                }
                Some(
                    v.downcast_handle_mut::<CpuBufferHandle>()
                        .ok_or(BackendError::DeviceMismatch)?,
                )
            }
            None => {
                if use_momentum {
                    return Err(BackendError::Unsupported(
                        "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
                    ));
                }
                None
            }
        };
        let param_handle = param
            .downcast_handle_mut::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;

        for j in 0..param_handle.data.len() {
            let p = param_handle.data[j];
            let mut g = grad_handle.data[j];
            if config.weight_decay != 0.0 {
                g = config.weight_decay.mul_add(p, g);
            }
            if use_momentum {
                // 直前の分岐（`velocity.is_none() && use_momentum` の
                // 早期 return）により、ここへ到達する時点で
                // `velocity_handle` は必ず `Some` である。
                let Some(velocity_handle) = velocity_handle.as_deref_mut() else {
                    return Err(BackendError::Unsupported(
                        "sgd_step_device: momentum enabled but no velocity buffer provided".into(),
                    ));
                };
                let prev = velocity_handle.data[j];
                let b = if config.is_first_step {
                    g
                } else {
                    config.momentum.mul_add(prev, (1.0 - config.dampening) * g)
                };
                velocity_handle.data[j] = b;
                g = if config.nesterov {
                    config.momentum.mul_add(b, g)
                } else {
                    b
                };
            }
            param_handle.data[j] = p - config.lr * g;
        }
        Ok(())
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
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

    /// [`fandhe_ai_tensor_core::BackendOps::gemm_bias_act`] のデフォルト実装（非融合
    /// `gemm` → `add` → `relu` 合成）を、CPU カーネル内で epilogue を融合
    /// する [`gemm_blis_bias_act_parallel`] へ差し替える（TASK-12.1f・
    /// #203）。CUDA は同型のオーバーライド（`backend-cuda::ops::
    /// CudaBackendOps::gemm_bias_act`）をイシュー #599 で追加済み。Metal
    /// はこのオーバーライドを持たずデフォルト実装（非融合合成）を使う
    /// （elementwise 未実装により `bias`／`act` 指定時は `Unsupported` を
    /// 透過的に返す。モジュールドキュメント冒頭・`fandhe_ai_tensor_core::
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
    /// `fandhe_ai_tensor_core::broadcast_shape` でブロードキャスト可否のみ先に
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
        let out_shape = fandhe_ai_tensor_core::matmul_out_shape(a.shape(), b.shape())
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
            fandhe_ai_tensor_core::broadcast_shape(&out_shape, bias.shape())
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

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」）。CPU は「デバイス」がホストメモリ
    /// そのもの（`CpuBufferHandle.data: Vec<f32>`）であるため、
    /// `downcast_handle` で連結バッファを直接読み、[`DeviceBufferView`]
    /// の `offset()..offset()+numel()` 範囲をスライスするだけでゼロ
    /// コピーに `gemm_blis_bias_act_parallel` へ渡せる（`sgd_step_device`
    /// と同じ「転送コストゼロ」契約）。`bias` は `[n]`（`w` の列数）への
    /// 厳密一致のみ対応する（`gemm_bias_act` の融合カーネル契約と同じ）。
    /// カーネル本体（`gemm_blis_bias_act_parallel`）へ触れる前に shape を
    /// 検証する（REQ-8・OWASP A03）。範囲自体の検査は
    /// `DeviceBufferView::new` が構築時に済ませているため、ここでは
    /// スライスの長さ（`w.numel()`）が shape 由来の期待値と一致する前提で
    /// 直接インデックスする。
    fn gemm_resident_rhs(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];
        let w_handle = w
            .buffer()
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

        let bias_handle = match bias {
            Some(b) => {
                if b.device() != Device::Cpu {
                    return Err(BackendError::DeviceMismatch);
                }
                if b.shape() != [n] {
                    return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                        lhs: b.shape().to_vec(),
                        rhs: vec![n],
                    }));
                }
                let handle = b
                    .buffer()
                    .downcast_handle::<CpuBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(&handle.data[b.offset()..b.offset() + b.numel()])
            }
            None => None,
        };

        let a_owned = a.contiguous();
        let a_slice = a_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_resident_rhs: lhs not contiguous after contiguous()")
        })?;

        let mut out = vec![0.0f32; m * n];
        gemm_blis_bias_act_parallel(
            a_slice,
            w_slice,
            &mut out,
            m,
            n,
            k,
            bias_handle,
            Activation::None,
        )
        .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。`Op::LinearResident` の VJP（`fandhe_ai_autodiff::
    /// grad`）が `d_input^T = w @ g^T` を計算するために使う。[`Self::
    /// gemm_resident_rhs`] と同じくゼロコピー（`downcast_handle` 直読み
    /// + オフセット範囲スライス）。
    fn gemm_resident_lhs(
        &self,
        w: DeviceBufferView<'_>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        if w.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let w_shape = w.shape();
        if w_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: w_shape.len(),
            }));
        }
        let (p, q) = (w_shape[0], w_shape[1]);
        let b_shape = b.shape();
        if b_shape.len() != 2 || b_shape[0] != q {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: w_shape.to_vec(),
                rhs: b_shape.to_vec(),
            }));
        }
        let r = b_shape[1];
        let w_handle = w
            .buffer()
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

        let b_owned = b.contiguous();
        let b_slice = b_owned.as_slice().ok_or_else(|| {
            gemm_contiguity_fail_safe("gemm_resident_lhs: rhs not contiguous after contiguous()")
        })?;

        let mut out = vec![0.0f32; p * r];
        gemm_blis_parallel(w_slice, b_slice, &mut out, p, r, q)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &[p, r]).map_err(BackendError::ShapeMismatch)
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入出力ともホストへ実体化
    /// せずに計算する（イシュー #1028・`docs/inference-forward-fixed-
    /// cost-design.md` §3.2）。CPU は「デバイス」がホストメモリその
    /// ものであるため `downcast_handle` の直読みだけでゼロコピーに
    /// なる（`gemm_resident_rhs`／`sgd_step_device` と同じモデル）。
    ///
    /// **bit-exactness 契約**（`docs/inference-forward-fixed-cost-
    /// design.md` §3.3 (b)）: 旧経路（`Sequential::predict` 等が
    /// `tape.ops()` 経由で呼ぶ非融合 `gemm` → `add`（bias 行方向複製）
    /// → `relu` の 3 段合成）と**同一の累積順序**を保つため、本メソッドは
    /// `gemm_bias_act`／`gemm_resident_rhs` が使う融合カーネル
    /// （[`gemm_blis_bias_act_parallel`]。bias／act をカーネル内
    /// epilogue で適用するため tiling 次第で加算順序が変わりうる）を
    /// 使わず、`gemm`（[`gemm_blis_parallel`]）→ bias 行方向複製加算
    /// （単一の `a + b` はグルーピングに依らず IEEE 754 で一意に定まる
    /// ため、ループ構造が異なっても `elementwise::add` と bit-exact）→
    /// `relu`（`max(x, 0.0)`。`elementwise::relu_slice` と同一定義）の
    /// 3 段を明示的に合成する。将来 CPU 側に融合カーネル版を追加する
    /// 場合は、本 doc の bit-exactness 契約ごと見直すこと。
    fn linear_forward_device(
        &self,
        a: &DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        if a.device() != Device::Cpu || w.device() != Device::Cpu {
            return Err(BackendError::DeviceMismatch);
        }
        let a_shape = a.shape();
        if a_shape.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a_shape.len(),
            }));
        }
        let (m, k) = (a_shape[0], a_shape[1]);
        let w_shape = w.shape();
        if w_shape.len() != 2 || w_shape[0] != k {
            return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                lhs: a_shape.to_vec(),
                rhs: w_shape.to_vec(),
            }));
        }
        let n = w_shape[1];

        let a_handle = a
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        if a_handle.data.len() != a.numel() {
            // `DeviceBuffer::new` 経由で構築される限り到達しないはずだが、
            // shape とハンドル実体のずれを本番経路で `unwrap`/`expect` に
            // 頼らず検出する（REQ-8・OWASP A03 と同種の防御）。
            return Err(BackendError::ShapeMismatch(
                ShapeError::ElementCountMismatch {
                    expected: a.numel(),
                    actual: a_handle.data.len(),
                },
            ));
        }
        let a_slice = &a_handle.data[..];

        let w_handle = w
            .buffer()
            .downcast_handle::<CpuBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let w_slice = &w_handle.data[w.offset()..w.offset() + w.numel()];

        let bias_handle_slice = match bias {
            Some(b) => {
                if b.device() != Device::Cpu {
                    return Err(BackendError::DeviceMismatch);
                }
                if b.shape() != [n] {
                    return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
                        lhs: b.shape().to_vec(),
                        rhs: vec![n],
                    }));
                }
                let handle = b
                    .buffer()
                    .downcast_handle::<CpuBufferHandle>()
                    .ok_or(BackendError::DeviceMismatch)?;
                Some(&handle.data[b.offset()..b.offset() + b.numel()])
            }
            None => None,
        };

        // 1 段目: 非融合 `gemm`（`self.gemm` と同一カーネル）。
        // `m * n` はホスト入力由来の shape 積であり、アロケーション前に
        // オーバーフロー検査を行う（REQ-8・OWASP A03。`reduction.rs` の
        // `ElementCountOverflow` 使用箇所と同一方針）。
        let out_len = m.checked_mul(n).ok_or(BackendError::ShapeMismatch(
            ShapeError::ElementCountOverflow,
        ))?;
        let mut out = vec![0.0f32; out_len];
        gemm_blis_parallel(a_slice, w_slice, &mut out, m, n, k)
            .map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;

        // 2 段目: bias の行方向複製加算（`elementwise::add` の broadcast
        // と同じ結果を単一ループで生成。単一の浮動小数点加算はグルー
        // ピングに依らず一意に定まるため bit-exact）。
        if let Some(bias_slice) = bias_handle_slice {
            for i in 0..m {
                let row = &mut out[i * n..(i + 1) * n];
                for (c, b) in row.iter_mut().zip(bias_slice.iter()) {
                    *c += b;
                }
            }
        }

        // 3 段目: activation（`elementwise::relu_slice` と同一定義）。
        match act {
            Activation::None => {}
            Activation::Relu => {
                for v in out.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "linear_forward_device: unsupported activation {act:?}"
                )));
            }
        }

        shared_cpu_memory().wrap_vec(out, vec![m, n])
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

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss`] の CPU 実装
    /// （イシュー #1045）。shape 検証・contiguous 化・`mse::
    /// mse_sum_sq_f32` への委譲・`reduction` に応じた最終変換（`Mean`/
    /// `Sum`）・スカラー `Tensor` への詰め直しを行う（`ops.rs` の既存
    /// 方針。モジュール冒頭コメント参照）。
    ///
    /// `reduction` 分岐をここで解決する理由: `MseReduction` は
    /// `#[non_exhaustive]` であり、`f32` を返す `mse::mse_sum_sq_f32`
    /// 側では未知 variant に対する「安全な既定値」が存在しない
    /// （`mse.rs` モジュール doc 参照）。`BackendError` を返せる本メソッド
    /// でのみ、未知 variant を `Unsupported` として型付きに拒否できる。
    fn mse_loss(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_c = pred.contiguous();
        let target_c = target.contiguous();
        // `contiguous()` の戻り値は常に `as_slice()` が `Some` を返す
        // （`Tensor::contiguous` の契約。`tensor.rs`）ため、`unwrap_or`
        // で空スライスへ後退することはない（shape 一致検証済みで両者
        // 同じ要素数のため、以降のスライス長も一致する）。
        let pred_slice = pred_c.as_slice().unwrap_or(&[]);
        let target_slice = target_c.as_slice().unwrap_or(&[]);
        let numel = pred_slice.len();
        let sum_sq = mse::mse_sum_sq_f32(pred_slice, target_slice);
        let value = match reduction {
            MseReduction::Mean => {
                if numel == 0 {
                    0.0
                } else {
                    sum_sq / numel as f32
                }
            }
            MseReduction::Sum => sum_sq,
            _ => {
                return Err(BackendError::Unsupported(format!(
                    "mse_loss: unsupported MseReduction variant {reduction:?}"
                )));
            }
        };
        Tensor::new(vec![value], &[]).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::mse_loss_backward`] の CPU
    /// 実装（イシュー #1045）。`dTarget = −dPred` は呼び出し元
    /// （`fandhe_ai_autodiff::grad::vjp`）がホスト側で符号反転して得る
    /// 契約のため、本メソッドは `dPred` のみを計算して返す
    /// （`backend_ops.rs::BackendOps::mse_loss_backward` doc 参照）。
    fn mse_loss_backward(
        &self,
        pred: &Tensor<f32>,
        target: &Tensor<f32>,
        scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        require_same_shape(pred.shape(), target.shape()).map_err(BackendError::ShapeMismatch)?;
        let pred_c = pred.contiguous();
        let target_c = target.contiguous();
        let pred_slice = pred_c.as_slice().unwrap_or(&[]);
        let target_slice = target_c.as_slice().unwrap_or(&[]);
        let mut dpred = vec![0.0f32; pred_slice.len()];
        mse::mse_loss_backward_f32(pred_slice, target_slice, scale, &mut dpred);
        Tensor::new(dpred, pred.shape()).map_err(BackendError::ShapeMismatch)
    }

    /// [`fandhe_ai_tensor_core::BackendOps::run_fused`] のデフォルト実装（`Unsupported`
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
    /// それぞれ `rmsnorm::run_rmsnorm_f32_raw`／[`softmax::run_softmax_f32`]
    /// へルーティングする。どちらにも一致しない場合（elementwise-only・
    /// 中間軸 softmax 等）は従来どおり [`fused_elementwise::
    /// run_fused_elementwise`] の allowlist 検査へ委ねる（既存 elementwise
    /// 融合経路の挙動は不変）。RMSNorm 判定を先に試す理由は CUDA 側と同じ
    /// （op 列長〈6 vs 8〉が異なるため両方に一致するプランは存在しない）。
    ///
    /// # 呼び出し元
    /// `fandhe_ai_autodiff::tape` の遅延評価 2 層（`materialize_fallible`／
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
