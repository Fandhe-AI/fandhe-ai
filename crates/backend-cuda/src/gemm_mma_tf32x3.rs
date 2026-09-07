//! 3×TF32（split-single 法）`mma.sync`(m16n8k8)/`ldmatrix`/`cp.async` GEMM
//! の起動 API（イシュー #1355。親ツリー #1354・承認元 #1338）。
//!
//! `CudaMmaTf32x3Gemm` は `kernels_mma_tf32x3::mma_tf32x3_source()` を
//! NVRTC コンパイル・保持し、以降はホスト側スライスを渡すだけで GPU
//! 実行できる境界を担う（`gemm_mma_tf32.rs::CudaMmaTf32Gemm` と同じ
//! 責務分割・同じ API 形状）。
//!
//! ホスト側形状検証は `gemm.rs::validate_gemm_dims`（`pub(crate)`）を
//! そのまま再利用し判定ロジックを複製しない。`gemm_mma_tf32.rs` が
//! 定義する本経路固有の `cp.async` 16 バイト（f32 4 要素）整列制約・
//! グリッド上限・K タイル添字オーバーフロー上限検証（`validate_mma_
//! tf32_alignment`／`validate_mma_tf32_grid_bounds`／`validate_mma_
//! tf32_k_bound`）・グリッド計算（`mma_tf32_launch_config`）・カーネル
//! 起動本体（`launch_mma_tf32_family`）を、タイル構成の完全エイリアス
//! （`kernels_mma_tf32x3.rs` 冒頭コメント「タイル構成のエイリアス」）を
//! 根拠にそのまま再利用する（複製しない）。
//!
//! compute capability ゲートは `gemm_mma.rs::check_min_compute_capability`
//! （`pub(crate)`）を再利用する（単発 TF32 経路と同じ `cp.async`/
//! `ldmatrix` 命令セットが要求する下限のため）。
//!
//! **本番結線**: `ops.rs::CudaBackendOps::gemm` が
//! `crate::precision::CudaGemmPrecision::Tf32x3` opt-in 時にのみ本型を
//! 経由する（`context_cache::cached_mma_tf32x3` 経由でキャッシュされる。
//! 既定 `Fp32Strict`・単発 `Tf32` の経路には一切影響しない）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::gemm_mma::check_min_compute_capability;
use crate::gemm_mma_tf32::{
    launch_mma_tf32_family, validate_mma_tf32_alignment, validate_mma_tf32_grid_bounds,
    validate_mma_tf32_k_bound,
};
use crate::kernels_mma_tf32x3;
use crate::nvrtc::compile_ptx;

/// A・B に非有限値（`NaN`／`±inf`）が含まれていないか起動前にホスト側で
/// 検査する（`CudaError::NonFiniteInput` ドキュメンテーションコメント
/// 参照。codex-review 指摘・PR #1400）。
///
/// `kernels_mma_tf32x3.rs::MMA_TF32X3_SPLIT` の hi/lo 分割
/// （`lo = round_tf32(v - hi)`）は `v` が非有限だと `v - hi` が
/// `inf - inf = NaN` 等の不定形になり、その汚染が `mma.sync` の乗算
/// （`inf * 0 = NaN`）を通じて出力全体へ伝播しうる。CPU・単発 TF32
/// 経路が `±inf` をそのまま返す契約と食い違うため、本経路のみ非有限
/// 入力を未対応として明示的に拒否する（fail-closed。黙って誤った
/// 数値を返さない）。
fn validate_tf32x3_finite_input(a: &[f32], b: &[f32]) -> Result<(), CudaError> {
    if let Some((idx, v)) = a.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(CudaError::NonFiniteInput {
            detail: format!(
                "lhs (a) contains a non-finite value at flat index {idx}: {v} \
                 (3xTF32 split-single decomposition cannot represent non-finite \
                 operands without producing spurious NaN; use Fp32Strict or Tf32 \
                 precision mode for non-finite inputs)"
            ),
        });
    }
    if let Some((idx, v)) = b.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(CudaError::NonFiniteInput {
            detail: format!(
                "rhs (b) contains a non-finite value at flat index {idx}: {v} \
                 (3xTF32 split-single decomposition cannot represent non-finite \
                 operands without producing spurious NaN; use Fp32Strict or Tf32 \
                 precision mode for non-finite inputs)"
            ),
        });
    }
    Ok(())
}

/// 3×TF32 `mma.sync`(m16n8k8) GEMM カーネルのコンパイル済みハンドルを
/// 保持する。`stream` は `CudaDevice` から `Arc` クローンで受け取る
/// （`gemm_mma_tf32.rs::CudaMmaTf32Gemm` と同じ共有契約）。
pub struct CudaMmaTf32x3Gemm {
    stream: Arc<CudaStream>,
    mma_tf32x3: CudaFunction,
}

impl CudaMmaTf32x3Gemm {
    /// `device` 上で 3×TF32 `mma.sync`(m16n8k8) GEMM カーネルを NVRTC
    /// コンパイルし保持するハンドルを構築する。
    ///
    /// 手順: (1) `check_min_compute_capability`（cc>=8.0。単発 TF32 経路
    /// と共有するゲート）→ (2) `kernels_mma_tf32x3::mma_tf32x3_source()`
    /// を `device.arch()` 向けに `nvrtc::compile_ptx` でコンパイル → (3)
    /// `device.context().load_module()` → `load_function("gemm_mma_
    /// tf32x3")`。`libnvrtc` 不在時は `CudaError::NvrtcUnavailable` を
    /// 返す（`compile_ptx` のプローブゲート経由。panic しない。
    /// `gemm_mma_tf32.rs::CudaMmaTf32Gemm::new` と同一契約）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        check_min_compute_capability(device)?;

        let arch = device.arch();
        let ptx = compile_ptx(kernels_mma_tf32x3::mma_tf32x3_source(), arch)?;
        let mma_tf32x3 = device
            .context()
            .load_module(ptx)?
            .load_function("gemm_mma_tf32x3")?;

        Ok(Self {
            stream: device.stream().clone(),
            mma_tf32x3,
        })
    }

    /// A・B（f32）を渡すだけで GPU 実行し C（f32）を得る一括 API
    /// （`gemm_mma_tf32.rs::CudaMmaTf32Gemm::run_tf32` と同一の検証順序・
    /// 同一のゼロ次元形状契約。ドキュメンテーションコメント参照）。
    ///
    /// 形状検証群に加え、内部で呼ぶ `upload_f32` が `validate_tf32x3_
    /// finite_input` で A・B の非有限値（`NaN`／`±inf`）を起動前に検査
    /// し、含まれていれば [`crate::error::CudaError::NonFiniteInput`]
    /// で fail-closed に拒否する（本経路固有。分離 API 経路
    /// `upload_f32` → `launch_tf32x3` → `download_f32` でも同じ検証を
    /// 通る。`CudaError::NonFiniteInput` ドキュメンテーションコメント
    /// 参照）。
    pub fn run_tf32x3(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_gemm_dims(a.len(), b.len(), m, n, k)?;

        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        if k == 0 {
            return Ok(vec![0.0f32; (m as usize) * (n as usize)]);
        }

        validate_mma_tf32_alignment(n, k)?;
        validate_mma_tf32_grid_bounds(m)?;
        validate_mma_tf32_k_bound(k)?;

        let (a_dev, b_dev) = self.upload_f32(a, b)?;
        let mut c_dev = self.alloc_output_f32(m, n)?;
        self.launch_tf32x3(&a_dev, &b_dev, &mut c_dev, m, n, k)?;
        self.download_f32(&c_dev)
    }

    /// A・B をホスト→デバイスへ転送する（`run_tf32x3` の H2D 部分の
    /// 切り出し。#1356 のベンチマークが GPU 実行時間のみを計測できる
    /// よう、転送とカーネル実行を分離する）。
    ///
    /// 非有限入力の拒否（`validate_tf32x3_finite_input`）はここで行う
    /// （`run_tf32x3` 単体ではなく、分離された公開 API 経路
    /// `upload_f32` → `launch_tf32x3` → `download_f32` を含む全公開
    /// 起動経路が唯一のホスト→デバイス取り込み点である本関数を必ず通る
    /// ため。`launch_tf32x3` はデバイス常駐スライスしか受け取らずホスト
    /// 側で有限性を再検査できないため、ここでの検証を迂回できない
    /// fail-closed 境界とする。codex-review 指摘・PR #1400）。
    pub fn upload_f32(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), CudaError> {
        validate_tf32x3_finite_input(a, b)?;
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        Ok((a_dev, b_dev))
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（`run_tf32x3` のバッファ
    /// 確保部分の切り出し）。
    pub fn alloc_output_f32(&self, m: u32, n: u32) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self
            .stream
            .alloc_zeros::<f32>((m as usize) * (n as usize))?)
    }

    /// デバイス常駐済みの A/B/C バッファに対してカーネルをストリームへ
    /// 非同期投入する（H2D/D2H を含まない「GPU 実行のみ」の区間）。
    /// 形状検証・no-op/`k==0` 契約・起動引数の組み立ては
    /// `gemm_mma_tf32.rs::launch_mma_tf32_family` へ委譲する（複製しない。
    /// 両カーネルはシグネチャ・タイル・境界検査契約が完全に同一である
    /// ことが根拠。`launch_mma_tf32_family` ドキュメンテーションコメント
    /// 参照）。
    ///
    /// 非同期投入契約（イシュー #1013 と同型）: 本関数は完了を待たない。
    /// 完了保証は呼び出し元の次の同期点（`download_f32` 等）に委ねる。
    pub fn launch_tf32x3(
        &self,
        a_dev: &CudaSlice<f32>,
        b_dev: &CudaSlice<f32>,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        launch_mma_tf32_family(&self.stream, &self.mma_tf32x3, a_dev, b_dev, c_dev, m, n, k)
    }

    /// C をデバイス→ホストへ転送する（`run_tf32x3` の D2H 部分の切り出
    /// し）。
    pub fn download_f32(&self, c_dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        crate::memory::readback(&self.stream, c_dev)
    }

    /// ストリームの完了を明示的に待つ（`gemm_mma_tf32.rs::CudaMmaTf32Gemm
    /// ::synchronize` と同じ理由の公開 API）。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.stream.synchronize()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gemm_mma_tf32x3` 経路が単発 TF32 経路と同一のグリッド計算・
    /// 整列検証を再利用していることを、境界値の往復で確認する
    /// （`gemm_mma_tf32.rs::tests` の同名テストと同一の判定基準）。
    #[test]
    fn validate_mma_tf32_alignment_accepts_multiples_of_four() {
        assert!(validate_mma_tf32_alignment(64, 32).is_ok());
        assert!(validate_mma_tf32_alignment(4, 4).is_ok());
    }

    #[test]
    fn validate_mma_tf32_alignment_rejects_non_multiple_n() {
        let err = validate_mma_tf32_alignment(9, 32).expect_err("n=9 is not a multiple of 4");
        assert!(matches!(err, CudaError::InvalidShape { .. }));
    }

    #[test]
    fn validate_mma_tf32_grid_bounds_accepts_shapes_within_limit() {
        assert!(validate_mma_tf32_grid_bounds(65_535 * kernels_mma_tf32x3::MMA_TF32X3_BM).is_ok());
    }

    #[test]
    fn validate_mma_tf32_k_bound_accepts_ordinary_k() {
        assert!(validate_mma_tf32_k_bound(0).is_ok());
        assert!(validate_mma_tf32_k_bound(4096).is_ok());
    }

    /// 全要素が有限な A・B を受理する（`validate_tf32x3_finite_input`
    /// の非破壊契約。codex-review 指摘・PR #1400）。
    #[test]
    fn validate_tf32x3_finite_input_accepts_all_finite() {
        let a = vec![1.0f32, -2.0, 0.0, f32::MAX, f32::MIN, 1e-30];
        let b = vec![3.0f32, 4.0, -5.0, 6.0, 7.0, 8.0];
        assert!(validate_tf32x3_finite_input(&a, &b).is_ok());
    }

    /// codex-review 指摘の再現ケース: A が全要素 `+inf`・B が全要素
    /// `1.0` の場合、`+inf` が拒否される（hi/lo 分割の `v - hi =
    /// inf - inf = NaN` 汚染を未然に防ぐ。詳細は
    /// `CudaError::NonFiniteInput` ドキュメンテーションコメント）。
    #[test]
    fn validate_tf32x3_finite_input_rejects_positive_infinity_in_lhs() {
        let a = vec![f32::INFINITY; 16];
        let b = vec![1.0f32; 16];
        let err = validate_tf32x3_finite_input(&a, &b)
            .expect_err("+inf lhs は非有限入力として拒否されるべき");
        match err {
            CudaError::NonFiniteInput { detail } => {
                assert!(
                    detail.contains("lhs (a)"),
                    "detail should identify the lhs operand: {detail}"
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }

    #[test]
    fn validate_tf32x3_finite_input_rejects_negative_infinity_in_rhs() {
        let a = vec![1.0f32; 4];
        let b = vec![f32::NEG_INFINITY; 4];
        let err = validate_tf32x3_finite_input(&a, &b)
            .expect_err("-inf rhs は非有限入力として拒否されるべき");
        match err {
            CudaError::NonFiniteInput { detail } => {
                assert!(
                    detail.contains("rhs (b)"),
                    "detail should identify the rhs operand: {detail}"
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }

    #[test]
    fn validate_tf32x3_finite_input_rejects_nan() {
        let a = vec![1.0f32, f32::NAN, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let err = validate_tf32x3_finite_input(&a, &b).expect_err("NaN lhs は拒否されるべき");
        assert!(matches!(err, CudaError::NonFiniteInput { .. }));
    }
}
