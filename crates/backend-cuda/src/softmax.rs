//! online softmax 順伝播カーネルの起動 API（NVRTC コンパイル・保持・実行。
//! イシュー #594）。
//!
//! `rmsnorm.rs::CudaRmsNorm`（#592）と同じ構成方針を踏襲する:
//! [`CudaSoftmax::new`] が `CudaDevice` から 2 カーネル（1 パス／2 パス。
//! `kernels_softmax.rs`）を一括 NVRTC コンパイルして保持し、同時にデバイス
//! 属性（SMEM 予算・SM 数）を 1 回だけ取得してキャッシュする。以降は
//! [`CudaSoftmax::run_softmax_f32`] へホスト側スライスを渡すだけで
//! 経路選択・persistent grid 導出・H2D → 起動 → 同期 → D2H を内部で完結
//! できる。経路選択（[`crate::rmsnorm::rmsnorm_route`]）・persistent grid
//! 導出（[`crate::rmsnorm::derive_persistent_grid_one_pass`]／
//! [`crate::rmsnorm::derive_persistent_grid_two_pass`]）は RMSNorm と
//! 同一の判定式（f32 1 行 = `cols * 4` バイト）のため RMSNorm 側の
//! `pub(crate)` ヘルパをそのまま再利用する（重複実装しない）。
//!
//! `ops.rs::CudaBackendOps::run_fused` から canonical softmax プラン
//! （`exp(x - max(x)) / sum(exp(x - max(x)))`。最終軸 softmax または
//! 全軸縮約）検出時に呼ばれる（[`match_softmax_plan`] 参照）。
//!
//! # 意味論注記（プランの `Exp`／`Div` とカーネルの `exp2`／online 更新）
//!
//! canonical プランは自然指数 `Exp`（`FusedOpKind::Exp`）で表現されるが、
//! 本カーネルは `exp2(x * log2(e))` を計算する（`kernels_softmax.rs`
//! 冒頭コメント「log2(e) 事前スケール + exp2f のみ使用」）。数学的には
//! 恒等（`e^x = 2^(x*log2(e))`）だが丸めは per-op 経路（`onnx-interop`
//! 素朴実装・CPU 参照実装の `expf`／`f32::exp`）とは異なるため、一致
//! 判定は常に REQ-2 複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。
//! 相対誤差 1e-3 未満または絶対誤差 1e-5 未満）に依る。tolerance は変更
//! しない（`.claude/rules/coding-rust.md`）。

use std::sync::Arc;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use fandhe_ai_tensor_core::{FusedOpKind, FusionPlan, RowFusionMeta};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm_auto::read_clamped_smem_budget_bytes;
use crate::kernels_softmax::{self, SOFTMAX_BLOCK_DIM};
use crate::nvrtc::compile_ptx;
use crate::rmsnorm::{
    RmsNormRoute, derive_persistent_grid_one_pass, derive_persistent_grid_two_pass, rmsnorm_route,
};

/// exp2 ドメインの境界マスク値（有限。`f32` 版）。
///
/// `kernels_softmax.rs` の `SOFTMAX_MASK_E2` マクロ（`-__FLT_MAX__`）と
/// 同一の値であり、ホスト・デバイス間でビット同一の値になる
/// （`kernels_softmax.rs` 冒頭コメント「境界マスク定数」参照）。
/// `f32` が表現できる最小の有限値（`f32::MIN == -f32::MAX`）を使う理由・
/// 数値的な妥当性は本モジュール末尾の `mask_value_e2_f32` 系テストで検証
/// する（実装計画 §3.3。参照実装〈metal-flash-attention〉の値を無検証で
/// 採用しない）。旧実装（`-0.875 * f32::MAX` のマージン値）は「行の全要素
/// がこのマスク値未満の正規の有限入力」（例: 全要素 `-f32::MAX`）で `m`
/// が一度も更新されず最終出力が NaN になる欠陥があった（イシュー #594
/// PR #712 codex-review 指摘・Cursor Bugbot 指摘の P1 修正。
/// `kernels_softmax.rs` 冒頭コメント「境界マスク定数」参照）。カーネル
/// 起動経路（`run_softmax_f32_raw` 等）はマスク値を一切参照しない（マスク
/// はカーネルソース文字列内に完結する定数であり、ホスト側はその妥当性を
/// 検証するためだけにこの値を保持する）ため `#[cfg(test)]` に限定する。
#[cfg(test)]
const SOFTMAX_MASK_E2_F32: f32 = -f32::MAX;

/// exp2 ドメインの境界マスク値（有限。`half::f16` の動的レンジ版）。
///
/// f16 **カーネル**経路自体は本イシュー対象外（backend-cuda の演算面が
/// f32 で統一されている現状に整合。実装計画 §8「スコープ外」）だが、
/// 将来の f16 経路が無検証採用にならないよう定数の妥当性のみ先行検証する
/// （実装計画 §3.3）。f16 が表現できる最小の有限値（`-half::f16::MAX`。
/// `half::f16::MAX` ≈ 65504）を使う。上記 `SOFTMAX_MASK_E2_F32` と同じ
/// 理由で `#[cfg(test)]` に限定する。
#[cfg(test)]
fn softmax_mask_e2_f16() -> half::f16 {
    // `half::f16` の演算は Rust 側でソフトウェア実装される（ハードウェア
    // f16 演算命令は使わない）ため、デバイス側 f16 カーネル（スコープ外）
    // とのビット同一性契約は持たない。あくまで定数の有限性・妥当性のみを
    // 検証する目的の値である（本ファイル冒頭コメント参照）。
    -half::f16::MAX
}

/// [`softmax_route`] が返す経路選択。[`RmsNormRoute`] をそのまま再利用する
/// （f32 1 行 = `cols * 4` バイトの判定式が RMSNorm と同一のため、独立した
/// enum を新設せず型を共有する）。
pub(crate) type SoftmaxRoute = RmsNormRoute;

/// `row_len`（行長。softmax では `cols`）が SMEM 予算に収まるかを
/// [`crate::rmsnorm::rmsnorm_route`] へそのまま委譲する（判定式が
/// RMSNorm と同一。モジュール冒頭コメント参照）。
pub(crate) fn softmax_route(cols: usize, per_block_smem_budget_bytes: u64) -> SoftmaxRoute {
    rmsnorm_route(cols, per_block_smem_budget_bytes)
}

/// ホスト側検証（起動前・fail-closed）: `rows * cols == x_len`（checked
/// 乗算）・`rows`／`cols`／`numel` が `i32::MAX`（カーネル引数 `int rows`／
/// `int cols` 契約）に収まることを検証する（`rmsnorm.rs::
/// validate_rmsnorm_launch` と同型。OWASP A03・`.claude/rules/security.md`）。
pub(crate) fn validate_softmax_launch(
    rows: usize,
    cols: usize,
    x_len: usize,
) -> Result<(), CudaError> {
    let numel = rows
        .checked_mul(cols)
        .ok_or_else(|| CudaError::InvalidSoftmaxShape {
            detail: format!("softmax rows*cols overflowed usize: rows={rows}, cols={cols}"),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidSoftmaxShape {
            detail: format!("softmax x length mismatch: rows*cols={numel}, x.len()={x_len}"),
        });
    }
    if rows > i32::MAX as usize || cols > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidSoftmaxShape {
            detail: format!(
                "softmax dims must fit in i32 (kernel argument type): rows={rows}, cols={cols}, \
                 numel={numel}"
            ),
        });
    }
    Ok(())
}

/// canonical softmax 融合プラン（`exp(x - max(x)) / sum(exp(x - max(x)))`）
/// に厳密一致する `plan` から、起動に必要な `(rows, cols)` を取り出す。
///
/// プラン形状は `fandhe_ai_tensor_core::fusion::plan` の softmax パターン（`leaf` 1
/// 個・8 op 列: `Input → Max{axis} → Broadcast{axis} → Sub{lhs:0,rhs:2} →
/// Exp{input:3} → Sum{axis} → Broadcast{axis} → Div{lhs:4,rhs:6}`。
/// `plan.rs::from_ops_builds_softmax_plan_with_row_fusion_metadata`
/// 参照）に厳密一致する場合のみ `Some` を返す。一致しない場合（RMSNorm
/// 型・elementwise-only・中間軸 softmax 等）は `None` を返し、呼び出し元
/// （`ops.rs::CudaBackendOps::run_fused`）はデフォルトの `Unsupported` へ
/// フォールバックする（`rmsnorm.rs::match_rmsnorm_plan` と同じ fail-closed
/// allowlist 方針。`.claude/rules/security.md` A08「判定の迂回経路を
/// 作らない」）。
///
/// 受理する `axis`:
/// - `axis: None`（rank-1・全軸縮約）→ `rows = 1`・`cols = row_len`
/// - `axis: Some(a)` かつ `a == rank - 1`（**最終軸**）→
///   `rows = output_shape[..a] の積`・`cols = output_shape[a]`
///   （`row_fusion().row_len()` と一致検証）
///
/// それ以外（中間軸 softmax 等）は `None`（per-op フォールバックへ
/// fail-closed 委譲。実装計画 §3.2）。
pub(crate) fn match_softmax_plan(plan: &FusionPlan) -> Option<(usize, usize)> {
    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 8 {
        return None;
    }
    if !matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }) {
        return None;
    }
    let axis = match ops[1] {
        FusedOpKind::Max { input: 0, axis } => axis,
        _ => return None,
    };
    let expect = [
        matches!(ops[2], FusedOpKind::Broadcast { input: 1, axis: a } if a == axis),
        matches!(ops[3], FusedOpKind::Sub { lhs: 0, rhs: 2 }),
        matches!(ops[4], FusedOpKind::Exp { input: 3 }),
        matches!(ops[5], FusedOpKind::Sum { input: 4, axis: a } if a == axis),
        matches!(ops[6], FusedOpKind::Broadcast { input: 5, axis: a } if a == axis),
        matches!(ops[7], FusedOpKind::Div { lhs: 4, rhs: 6 }),
    ];
    if expect.iter().any(|ok| !ok) {
        return None;
    }

    let output_shape = plan.output_shape();
    let rank = output_shape.len();
    if let Some(a) = axis
        && (rank == 0 || a != rank - 1)
    {
        // 最終軸以外（中間軸 softmax）は対象外（実装計画 §3.2）。
        return None;
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis() != axis {
        return None;
    }
    let cols = row_fusion.row_len();

    let rows = match axis {
        None => 1,
        Some(a) => {
            // `row_fusion().row_len()` は `output_shape[a]` と一致する
            // はずだが、`match_rmsnorm_plan` と同じ理由（要素数一致だけの
            // 取り違えを許さない）で明示的に照合する。
            if output_shape[a] != cols {
                return None;
            }
            output_shape[..a].iter().product()
        }
    };
    Some((rows, cols))
}

/// online softmax 順伝播カーネル（1 パス／2 パスの 2 エントリ）のコンパイル
/// 済みハンドルと、経路選択・persistent grid 導出に使うデバイス属性
/// （SMEM 予算・SM 数）を保持する。
pub struct CudaSoftmax {
    stream: Arc<CudaStream>,
    onepass_f32: CudaFunction,
    twopass_f32: CudaFunction,
    /// per-block SMEM 予算（[`softmax_route`] の分岐に使う）。
    smem_per_block_budget_bytes: u64,
    /// per-SM SMEM 予算（[`derive_persistent_grid_one_pass`] の
    /// `blocks_per_sm` 導出に使う）。
    smem_per_sm_budget_bytes: u64,
    /// SM（マルチプロセッサ）数（`derive_persistent_grid_*` の共通入力）。
    sm_count: u32,
}

impl CudaSoftmax {
    /// `device` 上で softmax 2 カーネルを NVRTC コンパイルし、経路選択・
    /// persistent grid 導出に使うデバイス属性を 1 回だけ取得して保持する
    /// ハンドルを構築する（`rmsnorm.rs::CudaRmsNorm::new` と同型）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let onepass_ptx = compile_ptx(kernels_softmax::SOFTMAX_F32_ONEPASS, arch)?;
        let twopass_ptx = compile_ptx(kernels_softmax::SOFTMAX_F32_TWOPASS, arch)?;

        let onepass_f32 = device
            .context()
            .load_module(onepass_ptx)?
            .load_function("softmax_f32_onepass")?;
        let twopass_f32 = device
            .context()
            .load_module(twopass_ptx)?
            .load_function("softmax_f32_twopass")?;

        let smem_per_block_budget_bytes = read_clamped_smem_budget_bytes(device)?;

        let raw_smem_per_sm = device.context().attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        )?;
        // `rmsnorm.rs::CudaRmsNorm::new` と同じ理由（`InvalidRmsNormShape`
        // 相当の入力起因エラーではなく `InvalidKernelDescriptor` を使う。
        // `CudaError::Display` の意味論的な誤表示を避ける）。
        let smem_per_sm_budget_bytes =
            u64::try_from(raw_smem_per_sm).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaSoftmax::new: CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR \
                     returned a negative value ({raw_smem_per_sm}), which cannot be a valid SMEM \
                     budget"
                ),
            })?;

        let raw_sm_count = device
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let sm_count =
            u32::try_from(raw_sm_count).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaSoftmax::new: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT returned a \
                     negative value ({raw_sm_count}), which cannot be a valid SM count"
                ),
            })?;

        Ok(Self {
            stream: device.stream().clone(),
            onepass_f32,
            twopass_f32,
            smem_per_block_budget_bytes,
            smem_per_sm_budget_bytes,
            sm_count,
        })
    }

    /// `out[r, :] = softmax(x[r, :])`（行方向・最終軸 softmax）を実行する
    /// 公開エントリ。`x` は `[rows, cols]` の行優先 1 次元化済みバッファ。
    /// `scale = log2(e)`（`std::f32::consts::LOG2_E`）を内部合成して
    /// `Self::run_softmax_f32_raw` へ委譲する。
    pub fn run_softmax_f32(
        &self,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, CudaError> {
        self.run_softmax_f32_raw(x, std::f32::consts::LOG2_E, rows, cols)
    }

    /// `out[r, :] = exp2((x[r, :] - m_r) * scale) / l_r`（`m_r`／`l_r` は行
    /// `r` の online softmax 統計。`m_r` は生ドメイン〈スケール未適用〉。
    /// `kernels_softmax.rs` 冒頭コメント「`log2(e)` 事前スケール」参照）を
    /// 実行する内部エントリ。`scale` を
    /// 呼び出し元が明示するため、標準公開 API（[`Self::run_softmax_f32`]・
    /// `scale = log2(e)`）と将来の attention 融合（`scale = log2(e)/sqrt(d)`
    /// 合成。実装計画 §3.1「将来の attention 合成を見込んだ引数化」）の
    /// 両方の起動元になれる。`ops.rs::CudaBackendOps::run_fused` も
    /// `scale = log2(e)` で本メソッドを直接呼ぶ（プランの意味論
    /// `exp(x - max(x)) / sum(...)` に厳密一致させるため。標準公開 API を
    /// 経由しない理由は `rmsnorm.rs::CudaRmsNorm::run_rmsnorm_f32` の
    /// ドキュメンテーションコメントと同じ整理）。
    ///
    /// `rows == 0 || cols == 0` は空結果の早期 return（`rmsnorm.rs::
    /// CudaRmsNorm::run_rmsnorm_f32_raw` の 0 要素契約と同じ）。経路選択は
    /// [`softmax_route`]、persistent grid は [`derive_persistent_grid_one_pass`]／
    /// [`derive_persistent_grid_two_pass`] が導出する。
    pub(crate) fn run_softmax_f32_raw(
        &self,
        x: &[f32],
        scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_softmax_launch(rows, cols, x.len())?;

        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }

        let route = softmax_route(cols, self.smem_per_block_budget_bytes);

        let x_dev = self.stream.clone_htod(x)?;
        let mut out_dev = self.stream.alloc_zeros::<f32>(x.len())?;

        let rows_i = rows as i32;
        let cols_i = cols as i32;

        let (func, cfg): (&CudaFunction, LaunchConfig) = match route {
            RmsNormRoute::OnePassSmem => {
                // `derive_persistent_grid_one_pass` の `smem_bytes_per_block`
                // 契約は「1 ブロックが実際に確保する SMEM バイト数」
                // （`rmsnorm.rs::CudaRmsNorm::run_rmsnorm_f32_raw` の同種
                // コメント参照。予算上限〈通常 48KiB〉ではなく実バイト数
                // を渡さないと `blocks_per_sm` が過小評価される）。
                let smem_bytes_per_block = (cols as u64).saturating_mul(4);
                let grid = derive_persistent_grid_one_pass(
                    self.smem_per_sm_budget_bytes,
                    self.sm_count,
                    smem_bytes_per_block,
                    rows_i as u32,
                );
                let shared_mem_bytes = smem_bytes_per_block.min(u32::MAX as u64) as u32;
                (
                    &self.onepass_f32,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (SOFTMAX_BLOCK_DIM, 1, 1),
                        shared_mem_bytes,
                    },
                )
            }
            RmsNormRoute::TwoPass => {
                let grid = derive_persistent_grid_two_pass(self.sm_count, rows_i as u32);
                (
                    &self.twopass_f32,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (SOFTMAX_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
        };

        // SAFETY: カーネル引数（x_dev/out_dev・rows_i/cols_i・scale）は
        // `validate_softmax_launch` で検証済みの形状と 1:1 対応する
        // デバイスバッファ長・値であり、カーネル内の手動境界チェック
        // （`if (base+3 < cols)`／グリッドストライド `row < rows`・
        // REQ-8）と合わせて OOB 読み書きが起きない根拠とする。1 パス
        // 経路の `shared_mem_bytes` は `cols * 4`（実際に確保する SMEM
        // バイト数）であり、`softmax_route` が判定した
        // `smem_per_block_budget_bytes` 以下であることを既に確認済み
        // （`rmsnorm.rs::CudaRmsNorm::run_rmsnorm_f32_raw` の同種
        // SAFETY コメントと同じ不変条件）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&x_dev)
                .arg(&mut out_dev)
                .arg(&rows_i)
                .arg(&cols_i)
                .arg(&scale)
                .launch(cfg)?;
        }
        // 同期点は readback ヘルパーへ集約（#1013）。
        crate::memory::readback(&self.stream, &out_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_softmax_launch ---

    #[test]
    fn validate_softmax_launch_accepts_matching_dims() {
        assert!(validate_softmax_launch(3, 8, 24).is_ok());
    }

    #[test]
    fn validate_softmax_launch_rejects_x_len_mismatch() {
        let err = validate_softmax_launch(3, 8, 23).unwrap_err();
        assert!(matches!(err, CudaError::InvalidSoftmaxShape { .. }));
    }

    #[test]
    fn validate_softmax_launch_rejects_dims_exceeding_i32_max() {
        let err =
            validate_softmax_launch(i32::MAX as usize + 1, 1, i32::MAX as usize + 1).unwrap_err();
        assert!(matches!(err, CudaError::InvalidSoftmaxShape { .. }));
    }

    #[test]
    fn validate_softmax_launch_accepts_cols_at_i32_max_boundary() {
        assert!(validate_softmax_launch(1, i32::MAX as usize, i32::MAX as usize).is_ok());
    }

    #[test]
    fn validate_softmax_launch_accepts_zero_dims() {
        // `rows == 0 || cols == 0` は `run_softmax_f32_raw` が空 Vec を
        // 早期 return する契約（`numel == 0 == x_len`）。
        assert!(validate_softmax_launch(0, 8, 0).is_ok());
        assert!(validate_softmax_launch(8, 0, 0).is_ok());
    }

    // --- softmax_route（rmsnorm_route への委譲確認） ---

    #[test]
    fn softmax_route_matches_rmsnorm_route_for_same_inputs() {
        assert_eq!(softmax_route(1024, 4096), rmsnorm_route(1024, 4096));
        assert_eq!(softmax_route(16384, 49152), rmsnorm_route(16384, 49152));
    }

    // --- match_softmax_plan ---
    //
    // `fandhe_ai_tensor_core::fusion::graph`／`detect` は `tensor-core` 内部限定の
    // `pub(crate)` のため、`rmsnorm.rs::tests::build_canonical_rmsnorm_plan`
    // と同じ構築経路（`FusionPlan::from_ops`）でプランを直接組み立てる。
    // 8 op 列は `tensor-core` 側の受け入れ済みテスト
    // （`plan.rs::from_ops_builds_softmax_plan_with_row_fusion_metadata`）
    // と厳密同一の並びを使う（手書きの op 列と `match_softmax_plan` の
    // 期待値がズレていないかをこの構築経路自体が検証する。advisor
    // 指摘: マッチャが「一度も一致しない」まま緑になる回帰を防ぐ）。

    fn build_canonical_softmax_plan(axis: Option<usize>, output_shape: Vec<usize>) -> FusionPlan {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max { input: 0, axis },
            FusedOpKind::Broadcast { input: 1, axis },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum { input: 4, axis },
            FusedOpKind::Broadcast { input: 5, axis },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        FusionPlan::from_ops(ops, output_shape, fandhe_ai_tensor_core::DType::F32, 1).unwrap()
    }

    #[test]
    fn match_softmax_plan_accepts_canonical_last_axis_plan() {
        // `plan.rs::from_ops_builds_softmax_plan_with_row_fusion_metadata`
        // と同一の shape（`[2, 8]`・`axis: Some(1)`）。
        let plan = build_canonical_softmax_plan(Some(1), vec![2, 8]);
        assert_eq!(match_softmax_plan(&plan), Some((2, 8)));
    }

    #[test]
    fn match_softmax_plan_accepts_canonical_full_reduction_plan() {
        // `axis: None`（rank-1・全軸縮約）→ rows=1。
        let plan = build_canonical_softmax_plan(None, vec![8]);
        assert_eq!(match_softmax_plan(&plan), Some((1, 8)));
    }

    #[test]
    fn match_softmax_plan_rejects_non_final_axis() {
        // `axis: Some(0)` は 3 次元 `[2, 8, 4]` の最終軸（index 2）ではない
        // ため対象外（実装計画 §3.2「中間軸は None」）。
        let plan = build_canonical_softmax_plan(Some(0), vec![2, 8, 4]);
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[test]
    fn match_softmax_plan_rejects_rmsnorm_shaped_plan() {
        // RMSNorm: mul(0,0) → sum → rsqrt → broadcast → mul（softmax と
        // 異なる op 列・op 数）。
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: None,
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: None,
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        let plan =
            FusionPlan::from_ops(ops, vec![8], fandhe_ai_tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[test]
    fn match_softmax_plan_rejects_elementwise_only_plan() {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
        ];
        let plan =
            FusionPlan::from_ops(ops, vec![4], fandhe_ai_tensor_core::DType::F32, 2).unwrap();
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[test]
    fn match_softmax_plan_rejects_partial_op_sequence() {
        // 末尾の Div を欠いた 7 op 列（部分変形）は拒否する。
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max {
                input: 0,
                axis: Some(1),
            },
            FusedOpKind::Broadcast {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum {
                input: 4,
                axis: Some(1),
            },
            FusedOpKind::Broadcast {
                input: 5,
                axis: Some(1),
            },
        ];
        let plan =
            FusionPlan::from_ops(ops, vec![2, 8], fandhe_ai_tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_softmax_plan(&plan), None);
    }

    // --- 境界マスク値の数値検証（実装計画 §3.3。実機不要のホスト単体
    // テスト。無検証採用しない） ---

    #[test]
    fn mask_value_e2_f32_is_finite() {
        assert!(SOFTMAX_MASK_E2_F32.is_finite());
    }

    /// マスク値どうしの差分が有限（`0.0`）であることを検証する。
    /// `-INFINITY` を使った場合は `(-INF) - (-INF) = NaN` になり、warp
    /// 結合で「担当要素ゼロの lane 同士」が組み合わさった際に NaN が
    /// 伝播しうる（`kernels_softmax.rs` 冒頭コメント「境界マスク定数」
    /// 参照）。有限値（`-f32::MAX`）であればこの性質を持たないことを
    /// 実証する。
    #[test]
    fn mask_value_e2_f32_self_difference_is_finite_unlike_neg_infinity() {
        let diff = SOFTMAX_MASK_E2_F32 - SOFTMAX_MASK_E2_F32;
        assert_eq!(diff, 0.0, "有限マスク値どうしの差分は 0.0 になるはず");
        assert!(diff.is_finite());

        // 対比: -INFINITY だとこの性質が崩れる（NaN になる）ことを併せて
        // 実証し、有限値方式の必要性自体を検証する（実装計画 §3.3 (b)）。
        let neg_inf_diff = f32::NEG_INFINITY - f32::NEG_INFINITY;
        assert!(
            neg_inf_diff.is_nan(),
            "-INFINITY 同士の差分が NaN になることを対比確認する（有限値方式の必要性の根拠）"
        );
    }

    /// イシュー #594 PR #712 codex-review 指摘・Cursor Bugbot 指摘
    /// （P1）の直接的な回帰テスト: 行の全要素がマスク値そのもの
    /// （`-f32::MAX`）である場合でも、online スキャン（本テストは
    /// `kernels_softmax.rs` の 1 パス経路スカラーループと同じ更新式を
    /// ホスト側でシミュレートする）が `l` を正しく要素数まで積算し、
    /// `l == 0`（→ `inv_l == Inf` → 出力が `0 * Inf == NaN`）にならない
    /// ことを検証する。旧実装（`-0.875 * f32::MAX` のマージン値）では
    /// `raw < mask` となり `m` が一度も更新されず `l` が 0 のまま
    /// アンダーフローしていた（`kernels_softmax.rs` 冒頭コメント
    /// 「境界マスク定数」参照）。`-f32::MAX` は `f32` の値域下限その
    /// ものであるため、この入力でも `raw == m`（等値）となり `l` が
    /// 正しく積算される。
    #[test]
    fn mask_value_e2_f32_all_elements_at_mask_value_scan_is_correct() {
        let row = [SOFTMAX_MASK_E2_F32; 4];
        let scale = std::f32::consts::LOG2_E;

        let mut m = SOFTMAX_MASK_E2_F32;
        let mut l = 0.0f32;
        for &raw in &row {
            let m_new = m.max(raw);
            if m_new > m {
                l *= ((m - m_new) * scale).exp2();
            }
            l += ((raw - m_new) * scale).exp2();
            m = m_new;
        }

        assert!(l.is_finite() && !l.is_nan(), "l が NaN/Inf になった: {l}");
        assert_eq!(
            l,
            row.len() as f32,
            "全要素がマスク値の行では l は要素数と一致するはず（各要素の寄与が 1.0）"
        );

        let inv_l = 1.0f32 / l;
        assert!(
            inv_l.is_finite(),
            "l != 0 のため inv_l は有限になるはず: {inv_l}"
        );
        for &raw in &row {
            let out = ((raw - m) * scale).exp2() * inv_l;
            assert!(!out.is_nan(), "出力が NaN になった: {out}");
            assert!(
                (out - 1.0 / row.len() as f32).abs() < 1e-6,
                "全要素同値の行は一様分布になるはず: got {out}"
            );
        }
    }

    /// `exp2f(mask - m)` が代表的な `m`（0・大きな正負値）で NaN を生まず
    /// 0 へアンダーフローすることを検証する（実装計画 §3.3 (c)）。
    #[test]
    fn mask_value_e2_f32_exp2_underflows_to_zero_for_representative_m() {
        for &m in &[0.0f32, 1.0e10, -1.0e10, f32::MAX / 2.0] {
            let diff = SOFTMAX_MASK_E2_F32 - m;
            let result = diff.exp2();
            assert!(!result.is_nan(), "exp2f(mask - {m}) が NaN になった");
            assert_eq!(
                result, 0.0,
                "exp2f(mask - {m}) は 0 へアンダーフローするはず"
            );
        }
    }

    /// 空 lane（`m = MASK, l = 0`）を含む (m, l) 結合シミュレーション
    /// （カーネル内 butterfly 結合の 1 ステップ相当）が結果を変えない
    /// ことを検証する（実装計画 §3.3 (d)）。
    #[test]
    fn mask_value_e2_f32_empty_lane_combine_is_identity() {
        // 実データを持つ lane（m=2.0, l=3.0）と、担当要素ゼロの lane
        // （m=MASK, l=0）を結合する。`kernels_softmax.rs` の butterfly
        // 結合ロジックと同じ式をホスト側でシミュレートする。
        let (m_data, l_data) = (2.0f32, 3.0f32);
        let (m_empty, l_empty) = (SOFTMAX_MASK_E2_F32, 0.0f32);

        let m_t = m_data.max(m_empty);
        let l_self = if m_t > m_data {
            l_data * (m_data - m_t).exp2()
        } else {
            l_data
        };
        let l_peer = if m_t > m_empty {
            l_empty * (m_empty - m_t).exp2()
        } else {
            l_empty
        };
        let combined_l = l_self + l_peer;

        assert_eq!(
            m_t, m_data,
            "実データ側の最大値がそのまま結合後の最大値になるはず"
        );
        assert!(!combined_l.is_nan(), "空 lane との結合で NaN が発生した");
        assert_eq!(combined_l, l_data, "空 lane との結合は l を変えないはず");
    }

    /// `mask_value_e2_f32_empty_lane_combine_is_identity` の対比ケース:
    /// 「担当要素ゼロの空 lane」ではなく「実データを持つが、行の最大値が
    /// たまたまマスク値そのもの（`m == SOFTMAX_MASK_E2_F32`。例えば行の
    /// 最大値が `-f32::MAX`）である lane」を、別の空 lane（`m = MASK,
    /// l = 0`）と結合するシミュレーション。`kernels_softmax.rs` の
    /// butterfly 結合ロジック（PR #712 Cursor Bugbot 指摘の L190-194
    /// 相当箇所）が、この「マスク値との衝突」ケースでも NaN を生まず
    /// `l` を正しく保つことを検証する（本ファイル冒頭コメント
    /// 「境界マスク定数」参照。等値の場合は分岐で減算をスキップする
    /// 構造が「空 lane との結合」に限らず適用されることの直接的な根拠）。
    #[test]
    fn mask_value_e2_f32_data_lane_at_mask_value_combine_with_empty_lane_is_identity() {
        // 実データを持つが m がマスク値と一致する lane（行の最大値自体が
        // マスク値だったケースに相当）。
        let (m_data, l_data) = (SOFTMAX_MASK_E2_F32, 4.0f32);
        // 担当要素ゼロの空 lane。
        let (m_empty, l_empty) = (SOFTMAX_MASK_E2_F32, 0.0f32);

        let m_t = m_data.max(m_empty);
        let l_self = if m_t > m_data {
            l_data * (m_data - m_t).exp2()
        } else {
            l_data
        };
        let l_peer = if m_t > m_empty {
            l_empty * (m_empty - m_t).exp2()
        } else {
            l_empty
        };
        let combined_l = l_self + l_peer;

        assert_eq!(
            m_t, SOFTMAX_MASK_E2_F32,
            "両者ともマスク値のため結合後の最大値もマスク値のまま"
        );
        assert!(
            !combined_l.is_nan(),
            "マスク値どうしの結合で NaN が発生した"
        );
        assert_eq!(
            combined_l, l_data,
            "空 lane との結合は実データ側の l を変えないはず"
        );
    }

    // --- f16 マージン値の数値検証（実装計画 §3.3。カーネル経路自体は
    // スコープ外だが定数の妥当性のみ先行検証する） ---

    #[test]
    fn mask_value_e2_f16_is_finite() {
        assert!(softmax_mask_e2_f16().is_finite());
    }

    #[test]
    fn mask_value_e2_f16_self_difference_is_finite() {
        let mask = softmax_mask_e2_f16();
        let diff = mask.to_f32() - mask.to_f32();
        assert_eq!(diff, 0.0);
        assert!(diff.is_finite());
    }

    #[test]
    fn mask_value_e2_f16_exp2_underflows_to_zero_for_representative_m() {
        let mask = softmax_mask_e2_f16().to_f32();
        for &m in &[0.0f32, 100.0, -100.0, half::f16::MAX.to_f32() / 2.0] {
            let result = (mask - m).exp2();
            assert!(!result.is_nan());
            assert_eq!(result, 0.0);
        }
    }
}
