//! 融合 RMSNorm 順伝播カーネルの起動 API（NVRTC コンパイル・保持・実行。
//! イシュー #592）。
//!
//! `elementwise.rs::CudaElementwise` と同じ構成方針を踏襲する:
//! [`CudaRmsNorm::new`] が `CudaDevice` から 2 カーネル（1 パス／2 パス。
//! `kernels_rmsnorm.rs`）を一括 NVRTC コンパイルして保持し、同時にデバイス
//! 属性（SMEM 予算・SM 数）を 1 回だけ取得してキャッシュする
//! （`gemm_auto.rs::CudaGemmAuto` の `DeviceCaps` キャッシュと同じ設計
//! 判断: 呼び出しごとに driver API 照会を繰り返さない）。以降は
//! [`CudaRmsNorm::run_rmsnorm_f32`] へホスト側スライスを渡すだけで
//! 経路選択（[`rmsnorm_route`]）・persistent grid 導出
//! （[`derive_persistent_grid_one_pass`]／[`derive_persistent_grid_two_pass`]）・
//! H2D → 起動 → 同期 → D2H を内部で完結できる。
//!
//! `ops.rs::CudaBackendOps::run_fused` から canonical RMSNorm プラン
//! （`x * rsqrt(sum(x^2))`。`mean`／`eps`／`weight` を含まない）検出時に
//! 呼ばれる（[`match_rmsnorm_plan`] 参照）。

use std::sync::Arc;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use tensor_core::{FusedOpKind, FusionPlan, RowFusionMeta};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm_auto::read_clamped_smem_budget_bytes;
use crate::kernels_rmsnorm::{self, RMSNORM_BLOCK_DIM};
use crate::nvrtc::compile_ptx;

/// [`rmsnorm_route`] が返す経路選択。行長（`RowFusionMeta::row_len`／
/// 直接 API の `hidden`）が 1 パス経路（動的 SMEM 常駐）の予算に収まるか
/// どうかで分岐する（実装計画 §4.4）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RmsNormRoute {
    /// [`kernels_rmsnorm::RMSNORM_F32_ONEPASS`]（動的 SMEM 常駐・`x` の
    /// HBM 読みは 1 回のみ）。
    OnePassSmem,
    /// [`kernels_rmsnorm::RMSNORM_F32_TWOPASS`]（global 再読・SMEM 非使用）。
    TwoPass,
}

/// `row_len`（行長）が `per_block_smem_budget_bytes`（[`crate::gemm_auto::
/// read_clamped_smem_budget_bytes`] でクランプ済みの per-block SMEM 予算）
/// に `f32` 換算で収まるかを純関数で判定する（実機なしで単体テスト可能）。
///
/// **1 パス／2 パス判定は本関数（バックエンド側）の責務**である
/// （`tensor_core::fusion::RowFusionMeta` ドキュメンテーションコメント
/// 「1 パス／2 パス判定は各バックエンドの責務」参照。#588 codex-review
/// P2 是正で `tensor-core` 側から閾値を切り離した設計を、本関数が CUDA
/// バックエンド側の具体的な予算判定として実装する）。opt-in SMEM
/// （`cuFuncSetAttribute`）は使わないため、予算は常に既定 per-block 上限
/// （[`crate::gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプ済み）
/// を超えない（`read_clamped_smem_budget_bytes` 呼び出し元が保証する）。
pub(crate) fn rmsnorm_route(row_len: usize, per_block_smem_budget_bytes: u64) -> RmsNormRoute {
    let needed_bytes = (row_len as u64).saturating_mul(4);
    if needed_bytes <= per_block_smem_budget_bytes {
        RmsNormRoute::OnePassSmem
    } else {
        RmsNormRoute::TwoPass
    }
}

/// 1 パス経路（動的 SMEM 常駐）の persistent grid（block 数）を導出する
/// 純関数（実装計画 §4.4）。
///
/// `blocks_per_sm = min(smem_per_sm_bytes / smem_bytes_per_block, 16)`
/// （16 は参照実装〈TileKernels engram gate カーネル〉由来のレジスタ圧
/// キャップであり、実測値ではなくコード上の定数である。`smem_bytes_per_block
/// == 0` は `hidden == 0` の縮退ケースを表し、SMEM 制約が事実上存在しない
/// ため 2 パス経路と同じ上限 16 を用いる）。`grid = clamp(sm_count *
/// blocks_per_sm, 1, rows)`。`rows == 0` は呼び出し元（[`CudaRmsNorm::
/// run_rmsnorm_f32`]）が早期 return するため本関数へは渡らない契約
/// （`rows == 0` で呼ばれた場合は `clamp` パニックを避けるため 1 を返す
/// フェイルセーフのみ持つ）。
pub(crate) fn derive_persistent_grid_one_pass(
    smem_per_sm_bytes: u64,
    sm_count: u32,
    smem_bytes_per_block: u64,
    rows: u32,
) -> u32 {
    if rows == 0 {
        return 1;
    }
    let blocks_per_sm: u64 = smem_per_sm_bytes
        .checked_div(smem_bytes_per_block)
        .map_or(16, |v| v.clamp(1, 16));
    let grid = (sm_count as u64).saturating_mul(blocks_per_sm);
    grid.clamp(1, rows as u64) as u32
}

/// 2 パス経路（SMEM 制約なし）の persistent grid を導出する純関数。
/// [`derive_persistent_grid_one_pass`] と同じ `rows == 0` フェイルセーフを
/// 持つ。
pub(crate) fn derive_persistent_grid_two_pass(sm_count: u32, rows: u32) -> u32 {
    if rows == 0 {
        return 1;
    }
    let grid = (sm_count as u64).saturating_mul(16);
    grid.clamp(1, rows as u64) as u32
}

/// ホスト側検証（起動前・fail-closed）: `rows * hidden == x_len`（checked
/// 乗算）・`w_len == Some(hidden)`（`w` 指定時のみ）・`rows`／`hidden`／
/// `numel` が `i32::MAX`（カーネル引数 `int rows`／`int hidden` 契約）に
/// 収まること・`eps` が有限かつ非負（`is_finite() && eps >= 0.0`。`0.0`
/// は許容する——`run_fused` 経由の canonical プラン起動は `eps = 0.0` を
/// 渡す契約——`ops.rs` ドキュメンテーションコメント参照。負の `eps` を
/// 許すと `sum(x^2) * inv_n + eps` が負化しうる〈有限入力から `sqrt` が
/// NaN を生成する経路。codex-review 指摘・PR #706 レビュー
/// r3793473250〉ため fail-closed で拒否する）を検証する
/// （`elementwise.rs::validate_elementwise_len` と同じ OWASP A03 対策。
/// `.claude/rules/security.md`）。
pub(crate) fn validate_rmsnorm_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    eps: f32,
) -> Result<(), CudaError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm eps must be finite and non-negative: eps={eps}"),
        });
    }

    let numel = rows
        .checked_mul(hidden)
        .ok_or_else(|| CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm rows*hidden overflowed usize: rows={rows}, hidden={hidden}"),
        })?;
    if numel != x_len {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm x length mismatch: rows*hidden={numel}, x.len()={x_len}"),
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!("rmsnorm w length mismatch: hidden={hidden}, w.len()={wl}"),
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(CudaError::InvalidRmsNormShape {
            detail: format!(
                "rmsnorm dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        });
    }
    Ok(())
}

/// canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`。mean 化・eps・
/// weight を含まない）に厳密一致する `plan` から、起動に必要な行長
/// （`row_fusion().row_len()`）を取り出す。
///
/// プラン形状は `plan.rs::from_segment_builds_rmsnorm_plan_with_row_fusion_metadata`
/// が構築する 6 op 列（leaf 1 個・`Mul(0,0) → Sum{axis:None} → Rsqrt →
/// Broadcast{axis:None} → Mul(bc, x)`）に厳密一致する場合のみ `Some` を
/// 返す。一致しない場合（softmax 型・elementwise-only 等）は `None` を
/// 返し、呼び出し元（`ops.rs::CudaBackendOps::run_fused`）はデフォルトの
/// `Unsupported` へフォールバックする（`backend-cpu::fused_elementwise::
/// run_fused_elementwise` の allowlist 拒否方針と同じ fail-closed。
/// 実装計画 §5「Step 4」）。
///
/// `axis: None`（全軸縮約）のみを受理する: canonical プランは 1 次元
/// テンソル入力（`row_fusion().axis() == None`・`rows == 1` 相当）を
/// 対象とし、行方向（`axis: Some(a)`）の RMSNorm 融合プランは対象外
/// （現時点の融合 IR〈#588〉が生成する形状と一致させる。実装計画 §5
/// 「Step 4」の「厳密一致」方針）。
pub(crate) fn match_rmsnorm_plan(plan: &FusionPlan) -> Option<usize> {
    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 6 {
        return None;
    }
    let expect = [
        matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }),
        matches!(ops[1], FusedOpKind::Mul { lhs: 0, rhs: 0 }),
        matches!(
            ops[2],
            FusedOpKind::Sum {
                input: 1,
                axis: None
            }
        ),
        matches!(ops[3], FusedOpKind::Rsqrt { input: 2 }),
        matches!(
            ops[4],
            FusedOpKind::Broadcast {
                input: 3,
                axis: None
            }
        ),
        matches!(ops[5], FusedOpKind::Mul { lhs: 4, rhs: 0 }),
    ];
    if expect.iter().any(|ok| !ok) {
        return None;
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis().is_some() {
        return None;
    }
    Some(row_fusion.row_len())
}

/// 融合 RMSNorm 順伝播カーネル（1 パス／2 パスの 2 エントリ）のコンパイル
/// 済みハンドルと、経路選択・persistent grid 導出に使うデバイス属性
/// （SMEM 予算・SM 数）を保持する。
pub struct CudaRmsNorm {
    stream: Arc<CudaStream>,
    onepass_f32: CudaFunction,
    twopass_f32: CudaFunction,
    /// per-block SMEM 予算（[`crate::gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES`]
    /// でクランプ済み。[`rmsnorm_route`] の分岐に使う）。
    smem_per_block_budget_bytes: u64,
    /// per-SM SMEM 予算（[`derive_persistent_grid_one_pass`] の
    /// `blocks_per_sm` 導出に使う）。
    smem_per_sm_budget_bytes: u64,
    /// SM（マルチプロセッサ）数（`derive_persistent_grid_*` の共通入力）。
    sm_count: u32,
}

impl CudaRmsNorm {
    /// `device` 上で RMSNorm 2 カーネルを NVRTC コンパイルし、経路選択・
    /// persistent grid 導出に使うデバイス属性を 1 回だけ取得して保持する
    /// ハンドルを構築する。
    ///
    /// 属性取得の失敗（driver API エラー・負値応答）は fail-closed で
    /// `CudaError` を返す（推定値へフォールバックしない。`gemm_auto.rs::
    /// read_clamped_smem_budget_bytes`／`select_tile_config_for_device`
    /// と同じ方針）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        let onepass_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_F32_ONEPASS, arch)?;
        let twopass_ptx = compile_ptx(kernels_rmsnorm::RMSNORM_F32_TWOPASS, arch)?;

        let onepass_f32 = device
            .context()
            .load_module(onepass_ptx)?
            .load_function("rmsnorm_f32_onepass")?;
        let twopass_f32 = device
            .context()
            .load_module(twopass_ptx)?
            .load_function("rmsnorm_f32_twopass")?;

        let smem_per_block_budget_bytes = read_clamped_smem_budget_bytes(device)?;

        let raw_smem_per_sm = device.context().attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        )?;
        // 属性取得失敗（`TryFrom` の負値検出）は `InvalidRmsNormShape`
        // （入力テンソルの shape/eps 起因のホスト側検証エラー）ではなく
        // `InvalidKernelDescriptor` を使う。`gemm_auto.rs::
        // read_clamped_smem_budget_bytes` が同種のデバイス属性負値検出に
        // 使う variant と揃え、`CudaError::Display` の意味論的な誤表示を
        // 避ける（cursor[bot] 指摘・PR #706 レビュー r3793478993）。
        let smem_per_sm_budget_bytes =
            u64::try_from(raw_smem_per_sm).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaRmsNorm::new: CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR \
                     returned a negative value ({raw_smem_per_sm}), which cannot be a valid \
                     SMEM budget"
                ),
            })?;

        let raw_sm_count = device
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let sm_count =
            u32::try_from(raw_sm_count).map_err(|_| CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "CudaRmsNorm::new: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT returned a \
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

    /// 標準 RMSNorm（mean 正規化あり）: `out = x * rsqrt(mean(x^2, axis=-1)
    /// + eps) * w`（`w` が `None` の場合は乗算をスキップ）を実行する。
    ///
    /// `inv_n = 1/hidden` を内部で導出し [`Self::run_rmsnorm_f32_raw`]
    /// （`hidden == 0` の早期 return より後に呼ぶため `1/hidden` のゼロ
    /// 除算は起きない）へ委譲する。`ops.rs::CudaBackendOps::run_fused` は
    /// canonical プランの意味論（mean 化しない・`x * rsqrt(sum(x^2))`）に
    /// 厳密一致させるため本メソッドを経由せず [`Self::run_rmsnorm_f32_raw`]
    /// を `inv_n = 1.0` で直接呼ぶ（`ops.rs` ドキュメンテーションコメント
    /// 参照）。
    pub fn run_rmsnorm_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, CudaError> {
        if rows == 0 || hidden == 0 {
            // `validate_rmsnorm_launch` 経由の検証は `run_rmsnorm_f32_raw`
            // 側で行う（0 要素契約の早期 return も含め、二重の入口を
            // 一本化する）。
            return self.run_rmsnorm_f32_raw(x, w, eps, 1.0, rows, hidden);
        }
        let inv_n = 1.0f32 / hidden as f32;
        self.run_rmsnorm_f32_raw(x, w, eps, inv_n, rows, hidden)
    }

    /// `out = x * rsqrt(sum(x^2, axis=-1) * inv_n + eps) * w`（`w` が
    /// `None` の場合は `has_weight = 0` で乗算をスキップ）を実行する
    /// 内部エントリ。`inv_n` を呼び出し元が明示するため、標準 RMSNorm
    /// （[`Self::run_rmsnorm_f32`]・`inv_n = 1/hidden`）と canonical
    /// 融合プラン（`ops.rs::CudaBackendOps::run_fused`・`inv_n = 1.0`。
    /// mean 化しない）の両方の起動元になれる。
    ///
    /// `x` は `[rows, hidden]` の行優先 1 次元化済みバッファ。`rows == 0 ||
    /// hidden == 0` は空結果の早期 return（`elementwise.rs::run_binary` の
    /// 0 要素契約と同じ）。経路選択は [`rmsnorm_route`]（構築時に保持した
    /// `smem_per_block_budget_bytes` を使用）、persistent grid は
    /// [`derive_persistent_grid_one_pass`]／[`derive_persistent_grid_two_pass`]
    /// が導出する。
    pub(crate) fn run_rmsnorm_f32_raw(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        inv_n: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, CudaError> {
        validate_rmsnorm_launch(rows, hidden, x.len(), w.map(|s| s.len()), eps)?;

        if rows == 0 || hidden == 0 {
            return Ok(Vec::new());
        }

        let route = rmsnorm_route(hidden, self.smem_per_block_budget_bytes);

        let x_dev = self.stream.clone_htod(x)?;
        // `w` が `None` の場合もカーネル引数としてポインタは必要だが
        // `has_weight == 0` により決してデリファレンスされない
        // （`kernels_rmsnorm.rs` 参照）。ダミーとして 1 要素のゼロ初期化
        // バッファを渡す（`0` 要素バッファの確保を一部 CUDA driver が
        // 拒否しうる問題を避ける。`elementwise.rs` の `numel == 0` 早期
        // return と同じ理由）。
        let (w_dev, has_weight) = match w {
            Some(w_slice) => (self.stream.clone_htod(w_slice)?, 1i32),
            None => (self.stream.alloc_zeros::<f32>(1)?, 0i32),
        };
        let mut out_dev = self.stream.alloc_zeros::<f32>(x.len())?;

        let rows_i = rows as i32;
        let hidden_i = hidden as i32;

        let (func, cfg): (&CudaFunction, LaunchConfig) = match route {
            RmsNormRoute::OnePassSmem => {
                // `derive_persistent_grid_one_pass` の `smem_bytes_per_block`
                // 引数は「1 ブロックが実際に確保する SMEM バイト数」を
                // 受け取る契約（同関数のドキュメンテーションコメント
                // 参照）。予算上限（`self.smem_per_block_budget_bytes`。
                // 通常 48KiB）をそのまま渡すと、行サイズが予算より小さい
                // 一般的な hidden で `blocks_per_sm` が過小評価され、
                // 意図した persistent occupancy 上限（16）を大きく下回る
                // （cursor[bot] 指摘・PR #706 レビュー r3793478990）。
                // 経路判定〈`rmsnorm_route`〉は予算上限との比較のままで
                // よいが、grid 導出とカーネル起動の `shared_mem_bytes` は
                // 実際に確保する `hidden * 4` を単一の真実源として揃える。
                let smem_bytes_per_block = (hidden as u64).saturating_mul(4);
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
                        block_dim: (RMSNORM_BLOCK_DIM, 1, 1),
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
                        block_dim: (RMSNORM_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    },
                )
            }
        };

        // SAFETY: カーネル引数（x_dev/w_dev/out_dev・rows_i/hidden_i・
        // eps/inv_n/has_weight）は `validate_rmsnorm_launch` で検証済みの
        // 形状と 1:1 対応するデバイスバッファ長・値であり、カーネル内の
        // 手動境界チェック（`if (base+3 < hidden)`／グリッドストライド
        // `row < rows`・REQ-8）と合わせて OOB 読み書きが起きない根拠と
        // する。1 パス経路の `shared_mem_bytes` は `hidden * 4`（実際に
        // 確保する SMEM バイト数）であり、`rmsnorm_route` が判定した
        // `smem_per_block_budget_bytes` 以下であることを既に確認済み
        // （経路判定は予算上限との比較、起動は実バイト数という異なる
        // 量を扱うが、`hidden * 4 <= 予算上限` の不変条件により smem
        // 予算超過による起動失敗は起きない）。
        unsafe {
            self.stream
                .launch_builder(func)
                .arg(&x_dev)
                .arg(&w_dev)
                .arg(&mut out_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&eps)
                .arg(&inv_n)
                .arg(&has_weight)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;

        Ok(self.stream.clone_dtoh(&out_dev)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- rmsnorm_route ---

    #[test]
    fn rmsnorm_route_selects_onepass_when_exactly_at_budget() {
        // hidden=1024 の f32 行 = 4096 バイト。予算ちょうど。
        assert_eq!(rmsnorm_route(1024, 4096), RmsNormRoute::OnePassSmem);
    }

    #[test]
    fn rmsnorm_route_selects_twopass_when_one_byte_over_budget() {
        assert_eq!(rmsnorm_route(1024, 4095), RmsNormRoute::TwoPass);
    }

    #[test]
    fn rmsnorm_route_selects_onepass_when_one_byte_under_budget() {
        assert_eq!(rmsnorm_route(1023, 4096), RmsNormRoute::OnePassSmem);
    }

    #[test]
    fn rmsnorm_route_selects_twopass_for_large_hidden() {
        // 16384 * 4 = 65536 > 49152（既定 per-block 上限）。
        assert_eq!(rmsnorm_route(16384, 49152), RmsNormRoute::TwoPass);
    }

    #[test]
    fn rmsnorm_route_selects_onepass_for_zero_row_len() {
        assert_eq!(rmsnorm_route(0, 0), RmsNormRoute::OnePassSmem);
    }

    // --- derive_persistent_grid_one_pass ---

    #[test]
    fn derive_persistent_grid_one_pass_clamps_blocks_per_sm_at_16() {
        // smem_per_sm=1MiB, smem_per_block=1B → blocks_per_sm は
        // 極端に大きい値になるはずだが 16 でクランプされる。
        let grid = derive_persistent_grid_one_pass(1024 * 1024, 4, 1, 1_000_000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_one_pass_clamps_grid_at_rows() {
        let grid = derive_persistent_grid_one_pass(1024 * 1024, 100, 1, 10);
        assert_eq!(grid, 10);
    }

    #[test]
    fn derive_persistent_grid_one_pass_returns_at_least_one() {
        let grid = derive_persistent_grid_one_pass(0, 0, 4096, 5);
        assert_eq!(grid, 1);
    }

    #[test]
    fn derive_persistent_grid_one_pass_zero_smem_per_block_uses_cap() {
        // hidden==0（縮退）で smem_bytes_per_block==0 の場合、SMEM
        // 制約が事実上存在しないため 2 パスと同じ上限 16 を使う。
        let grid = derive_persistent_grid_one_pass(0, 4, 0, 1000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_one_pass_rows_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid_one_pass(1024, 4, 4, 0), 1);
    }

    // --- derive_persistent_grid_two_pass ---

    #[test]
    fn derive_persistent_grid_two_pass_uses_16_per_sm() {
        assert_eq!(derive_persistent_grid_two_pass(4, 1_000_000), 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_two_pass_clamps_at_rows() {
        assert_eq!(derive_persistent_grid_two_pass(100, 3), 3);
    }

    #[test]
    fn derive_persistent_grid_two_pass_rows_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid_two_pass(4, 0), 1);
    }

    // --- validate_rmsnorm_launch ---

    #[test]
    fn validate_rmsnorm_launch_accepts_matching_dims() {
        assert!(validate_rmsnorm_launch(3, 8, 24, Some(8), 1e-5).is_ok());
        assert!(validate_rmsnorm_launch(3, 8, 24, None, 0.0).is_ok());
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_x_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 23, None, 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_w_len_mismatch() {
        let err = validate_rmsnorm_launch(3, 8, 24, Some(7), 1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_non_finite_eps() {
        let err = validate_rmsnorm_launch(3, 8, 24, None, f32::NAN).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
        let err = validate_rmsnorm_launch(3, 8, 24, None, f32::INFINITY).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_negative_eps() {
        // 負の eps は `sum(x^2) * inv_n + eps` を負化しうるため fail-closed
        // で拒否する（codex-review 指摘・PR #706 レビュー r3793473250）。
        let err = validate_rmsnorm_launch(3, 8, 24, None, -1e-5).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
        let err = validate_rmsnorm_launch(3, 8, 24, None, -0.0001).unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_accepts_zero_eps() {
        // `run_fused` 経由の canonical プラン起動は eps=0.0 を渡す契約
        // （`ops.rs` ドキュメンテーションコメント参照）。0.0 は有限値。
        assert!(validate_rmsnorm_launch(3, 8, 24, None, 0.0).is_ok());
    }

    #[test]
    fn validate_rmsnorm_launch_rejects_dims_exceeding_i32_max() {
        let err =
            validate_rmsnorm_launch(i32::MAX as usize + 1, 1, i32::MAX as usize + 1, None, 1e-5)
                .unwrap_err();
        assert!(matches!(err, CudaError::InvalidRmsNormShape { .. }));
    }

    #[test]
    fn validate_rmsnorm_launch_accepts_hidden_at_i32_max_boundary() {
        // `hidden == i32::MAX`（許容上限ちょうど）は受理される契約
        // （`kernels_rmsnorm.rs` 冒頭コメント「ループ添字のオーバーフロー
        // 安全性」参照）。実機でこの `hidden` を実行するには行あたり
        // 約 8 GiB のバッファが要るため実行までは検証しないが、ホスト側
        // 検証がこの境界を「拒否しすぎていない」ことのみ回帰確認する
        // （`validate_rmsnorm_launch_rejects_dims_exceeding_i32_max` の
        // 対となる境界値ケース）。
        assert!(
            validate_rmsnorm_launch(1, i32::MAX as usize, i32::MAX as usize, None, 1e-5).is_ok()
        );
    }

    // --- match_rmsnorm_plan ---
    //
    // `tensor_core::fusion::graph`／`detect` は `tensor-core` 内部限定の
    // `pub(crate)`（`fusion/mod.rs` 冒頭コメント「配置は `tensor-core` の
    // 1 か所に閉じる」参照）で `backend-cuda` からは参照できないため、
    // ここでは `autodiff` と同じ構築経路（[`FusionPlan::from_ops`]。`pub` +
    // `#[doc(hidden)]`）で直接 canonical プランを組み立てる
    // （`plan.rs` モジュール冒頭コメント「`FusionPlan::from_ops`
    // （`pub` + `#[doc(hidden)]`）: `autodiff` クレート専用の構築経路」
    // 参照）。

    fn build_canonical_rmsnorm_plan(hidden: usize) -> FusionPlan {
        // leaf 0=x, 1=sq(Mul(0,0)), 2=sum(Sum{axis:None}(1)), 3=rsqrt(2),
        // 4=bc(Broadcast{axis:None}(3)), 5=out(Mul(4,0))
        // （`plan.rs::from_segment_builds_rmsnorm_plan_with_row_fusion_metadata`
        // と同型の op 列）。
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
        FusionPlan::from_ops(ops, vec![hidden], tensor_core::DType::F32, 1).unwrap()
    }

    #[test]
    fn match_rmsnorm_plan_accepts_canonical_plan() {
        let plan = build_canonical_rmsnorm_plan(8);
        assert_eq!(match_rmsnorm_plan(&plan), Some(8));
    }

    #[test]
    fn match_rmsnorm_plan_rejects_softmax_shaped_plan() {
        // softmax: max → sub(broadcast) → exp → sum → div(broadcast)
        // （RMSNorm と異なる op 列・leaf_count）。
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
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![2, 8], tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[test]
    fn match_rmsnorm_plan_rejects_elementwise_only_plan() {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![4], tensor_core::DType::F32, 2).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[test]
    fn match_rmsnorm_plan_rejects_row_axis_variant() {
        // 行方向（axis: Some(a)）の RMSNorm 型プランは対象外
        // （`match_rmsnorm_plan` ドキュメンテーションコメント参照）。
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: Some(1),
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![2, 8], tensor_core::DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }
}
