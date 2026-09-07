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

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::validate_gemm_dims;
use crate::gemm_mma::check_min_compute_capability;
use crate::gemm_mma_tf32::{
    launch_mma_tf32_family, validate_mma_tf32_alignment, validate_mma_tf32_grid_bounds,
    validate_mma_tf32_k_bound,
};
use crate::kernels_mma_tf32x3;
use crate::memory::GuardedSlice;
use crate::nvrtc::compile_ptx;

/// `v`（有限値のみを対象とする呼び出し前提。非有限値は
/// `validate_tf32x3_finite_input` の別チェックが先に拒否する）を TF32
/// 精度（8 bit 指数・10 bit 仮数）へ丸めた際、仮数の丸め上げが指数へ
/// 繰り上がって `±inf` へオーバーフローするかどうかを判定する
/// （codex-review 指摘・PR #1400 スレッド PRRT_kwDOTuUCJc6f01d6）。
///
/// TF32 は f32 と指数フィールド（8 bit・バイアス 127）を共有し仮数のみ
/// 23 bit → 10 bit（round-nearest, ties away from zero。PTX ISA
/// `cvt.rna.tf32.f32` 命令節）へ丸める。仮数の保持側上位 10 bit が
/// 既に全 1（`0x3FF`）かつ指数が f32 で表現可能な最大値（biased
/// exponent `0xFE`。unbiased 127）の場合に限り、切り捨てられる下位
/// 13 bit の丸め上げが仮数へキャリーし、指数を無限大／NaN のビット
/// パターン（`0xFF`）へ押し上げる。`f32::MAX`（仮数が全 1 のため
/// 必ず丸め上げが発生する）が典型例で、`hi = round_tf32(v)` が
/// `+inf` になり、残差 `v - hi` の再丸め（`lo`）も非有限へ汚染される
/// （`kernels_mma_tf32x3.rs` 冒頭コメント「分割式」節）。`v.is_finite()`
/// のみの検査では `v` 自体は有限のため検出できない。
fn tf32_round_overflows(v: f32) -> bool {
    debug_assert!(v.is_finite(), "caller must pre-filter non-finite values");
    let bits = v.to_bits();
    let exp = (bits >> 23) & 0xFF;
    if exp != 0xFE {
        // 指数が最大未満なら仮数の繰り上がりで指数が高々 1 増えるのみ
        // で、依然として f32 の有限範囲（biased exponent <= 0xFE）に
        // 収まる。
        return false;
    }
    let mantissa = bits & 0x007F_FFFF;
    let kept = mantissa >> 13; // TF32 が保持する上位 10 bit
    if kept != 0x3FF {
        // 保持部が全 1 でなければ丸め上げが発生しても仮数内で桁上げが
        // 収まり指数は変化しない。
        return false;
    }
    let dropped = mantissa & 0x1FFF; // 丸めで切り捨てられる下位 13 bit
    // ties away from zero: 半分（2^12）以上で丸め上げが発生し、
    // 全 1 の保持部がキャリーして指数が 0xFF（±inf）へ繰り上がる。
    dropped >= 0x1000
}

/// A・B に (1) 非有限値（`NaN`／`±inf`）が含まれていないか、
/// (2) 有限値だが TF32 丸め（[`tf32_round_overflows`]）で `±inf` へ
/// オーバーフローする値が含まれていないかを起動前にホスト側で検査する
/// （`CudaError::NonFiniteInput` ドキュメンテーションコメント参照。
/// (1) は codex-review 指摘・PR #1400 スレッド PRRT_kwDOTuUCJc6f0YV_、
/// (2) は同 PR スレッド PRRT_kwDOTuUCJc6f01d6）。
///
/// `kernels_mma_tf32x3.rs::MMA_TF32X3_SPLIT` の hi/lo 分割
/// （`lo = round_tf32(v - hi)`）は `v` が非有限、または丸め後の `hi` が
/// 非有限になると `v - hi` が `inf - inf = NaN` 等の不定形になり、
/// その汚染が `mma.sync` の乗算（`inf * 0 = NaN`）を通じて出力全体へ
/// 伝播しうる。CPU・単発 TF32 経路が `±inf` をそのまま返す契約と
/// 食い違うため、本経路のみ両方のケースを未対応として明示的に拒否する
/// （fail-closed。黙って誤った数値を返さない）。
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
    if let Some((idx, v)) = a
        .iter()
        .enumerate()
        .find(|(_, v)| tf32_round_overflows(**v))
    {
        return Err(CudaError::NonFiniteInput {
            detail: format!(
                "lhs (a) contains a finite value at flat index {idx}: {v} that \
                 overflows to +/-inf when rounded to TF32 precision (10-bit \
                 mantissa); 3xTF32 split-single decomposition cannot represent \
                 this without producing spurious NaN; use Fp32Strict or Tf32 \
                 precision mode for values this close to f32::MAX"
            ),
        });
    }
    if let Some((idx, v)) = b
        .iter()
        .enumerate()
        .find(|(_, v)| tf32_round_overflows(**v))
    {
        return Err(CudaError::NonFiniteInput {
            detail: format!(
                "rhs (b) contains a finite value at flat index {idx}: {v} that \
                 overflows to +/-inf when rounded to TF32 precision (10-bit \
                 mantissa); 3xTF32 split-single decomposition cannot represent \
                 this without producing spurious NaN; use Fp32Strict or Tf32 \
                 precision mode for values this close to f32::MAX"
            ),
        });
    }
    Ok(())
}

/// `CudaMmaTf32x3Gemm::upload_f32` でのみ構築できる、非有限入力の
/// 拒否（`validate_tf32x3_finite_input`）を通過済みの A/B デバイス
/// バッファのペア。フィールドは非公開（`super`〈本モジュール〉限定）
/// のため、外部クレートはもちろん本クレート内でも本モジュール外から
/// 未検証の `CudaSlice<f32>` を差し込んで構築することはできない。
/// `CudaMmaTf32x3Gemm::launch_tf32x3` はこの型のみを受け取り、生の
/// `CudaSlice<f32>` を受け付けないことで、公開されている
/// `device.stream().clone_htod()` 等で作った未検証バッファが起動境界
/// まで到達する経路自体を型で排除する（codex-review 指摘・PR #1400
/// スレッド PRRT_kwDOTuUCJc6f0YV_。`upload_f32` ドキュメンテーション
/// コメント参照）。
///
/// フィールドは [`GuardedSlice`]（codex-review P0 指摘対応・PR #1390
/// 再々修正。理由は `gemm.rs::CudaGemm::upload_f32` ドキュメンテーション
/// コメント参照。生の `CudaSlice` のままだと、本型を保持したまま
/// 複数呼び出しをまたぐベンチ・診断コードが最終的に drop する際、その
/// 解放が capture 排他機構を経由しない）。
#[derive(Debug)]
pub struct ValidatedTf32x3Inputs {
    a: GuardedSlice<f32>,
    b: GuardedSlice<f32>,
}

/// 3×TF32 `mma.sync`(m16n8k8) GEMM カーネルのコンパイル済みハンドルを
/// 保持する。`stream` は `CudaDevice` から `Arc` クローンで受け取る
/// （`gemm_mma_tf32.rs::CudaMmaTf32Gemm` と同じ共有契約）。
pub struct CudaMmaTf32x3Gemm {
    stream: Arc<CudaStream>,
    /// capture 排他（`with_driver_call`）に使うデバイス ordinal
    /// （codex-review P0 指摘対応・PR #1390 再々修正。`gemm.rs::
    /// CudaGemm::ordinal` と同じ役割）。
    ordinal: usize,
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
            ordinal: device.ordinal(),
            mma_tf32x3,
        })
    }

    /// `Self` の公開 driver 呼び出し系メソッド（H2D 転送・確保・
    /// カーネル起動・D2H readback）を CUDA Graph capture 排他へ参加
    /// させる（codex-review P0 指摘対応・PR #1390 再々修正。実体は
    /// `context_cache::with_driver_call` へ委譲。`gemm.rs::CudaGemm::
    /// with_driver_call` doc コメント参照）。`ops.rs::CudaBackendOps::gemm`
    /// の `Tf32x3` 分岐は既に自身の `with_driver_call` 区間の内側から
    /// `run_tf32x3` を呼ぶため、本関数によるネストは `context_cache::
    /// begin_driver_call` の `DRIVER_CALL_DEPTH` 機構（Cursor Bugbot
    /// 指摘対応）により正しく処理される。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
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

        let inputs = self.upload_f32(a, b)?;
        let mut c_dev = self.alloc_output_f32(m, n)?;
        self.launch_tf32x3(&inputs, &mut c_dev, m, n, k)?;
        self.download_f32(&c_dev)
    }

    /// A・B をホスト→デバイスへ転送する（`run_tf32x3` の H2D 部分の
    /// 切り出し。#1356 のベンチマークが GPU 実行時間のみを計測できる
    /// よう、転送とカーネル実行を分離する）。
    ///
    /// 非有限入力の拒否（`validate_tf32x3_finite_input`）はここで行い、
    /// 検証済みであることを型で保証する [`ValidatedTf32x3Inputs`] を返す
    /// （`run_tf32x3` 単体ではなく、分離された公開 API 経路 `upload_f32`
    /// → `launch_tf32x3` → `download_f32` を含む全公開起動経路が唯一の
    /// ホスト→デバイス取り込み点である本関数を必ず通るため）。
    ///
    /// `launch_tf32x3` は生の `CudaSlice<f32>` ではなく本関数が返す
    /// `ValidatedTf32x3Inputs` のみを受け取る。同型はフィールド非公開で
    /// 本関数以外に構築手段がないため、呼び出し元が公開されている
    /// `device.stream().clone_htod()` 等で未検証バッファを作って
    /// `launch_tf32x3` へ直接差し込む経路が存在しない（codex-review
    /// 指摘・PR #1400 スレッド PRRT_kwDOTuUCJc6f0YV_。以前の実装は
    /// `launch_tf32x3` が生スライスを受け取っており、この経路で
    /// `upload_f32` の検証を迂回できた）。
    pub fn upload_f32(&self, a: &[f32], b: &[f32]) -> Result<ValidatedTf32x3Inputs, CudaError> {
        validate_tf32x3_finite_input(a, b)?;
        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let b_dev = self.stream.clone_htod(b)?;
            Ok(ValidatedTf32x3Inputs {
                a: GuardedSlice::new(self.ordinal, a_dev),
                b: GuardedSlice::new(self.ordinal, b_dev),
            })
        })
    }

    /// C 用のゼロ初期化デバイスバッファを確保する（`run_tf32x3` のバッファ
    /// 確保部分の切り出し）。戻り値の型については [`Self::upload_f32`]
    /// ドキュメンテーションコメント参照（codex-review P0 指摘対応・
    /// PR #1390 再々修正で [`GuardedSlice`] へ変更）。
    pub fn alloc_output_f32(&self, m: u32, n: u32) -> Result<GuardedSlice<f32>, CudaError> {
        self.with_driver_call(|| {
            let c_dev = self
                .stream
                .alloc_zeros::<f32>((m as usize) * (n as usize))?;
            Ok(GuardedSlice::new(self.ordinal, c_dev))
        })
    }

    /// 検証済みの A/B（[`ValidatedTf32x3Inputs`]。`upload_f32` でのみ
    /// 構築できる）と C バッファに対してカーネルをストリームへ非同期
    /// 投入する（H2D/D2H を含まない「GPU 実行のみ」の区間）。形状検証・
    /// no-op/`k==0` 契約・起動引数の組み立ては
    /// `gemm_mma_tf32.rs::launch_mma_tf32_family` へ委譲する（複製しない。
    /// 両カーネルはシグネチャ・タイル・境界検査契約が完全に同一である
    /// ことが根拠。`launch_mma_tf32_family` ドキュメンテーションコメント
    /// 参照）。
    ///
    /// 引数を生の `&CudaSlice<f32>` ではなく `&ValidatedTf32x3Inputs` に
    /// することで、`upload_f32` の非有限値検査を経ていないバッファが
    /// 本関数へ到達しえない（`upload_f32` ドキュメンテーションコメント
    /// 参照。codex-review 指摘・PR #1400）。
    ///
    /// 非同期投入契約（イシュー #1013 と同型）: 本関数は完了を待たない。
    /// 完了保証は呼び出し元の次の同期点（`download_f32` 等）に委ねる。
    pub fn launch_tf32x3(
        &self,
        inputs: &ValidatedTf32x3Inputs,
        c_dev: &mut CudaSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), CudaError> {
        // codex-review P0 指摘対応（PR #1390 再々修正）: `Self::
        // with_driver_call` で `launch_mma_tf32_family` 呼び出しを
        // capture 排他へ参加させる。`&inputs.a`／`&inputs.b`
        // （`GuardedSlice<f32>`）は `Deref` により `&CudaSlice<f32>` を
        // 要求する `launch_mma_tf32_family` へそのまま渡せる。
        self.with_driver_call(|| {
            launch_mma_tf32_family(
                &self.stream,
                &self.mma_tf32x3,
                &inputs.a,
                &inputs.b,
                c_dev,
                m,
                n,
                k,
            )
        })
    }

    /// C をデバイス→ホストへ転送する（`run_tf32x3` の D2H 部分の切り出
    /// し）。codex-review P0 指摘対応（PR #1390 再々修正）: `Self::
    /// with_driver_call` で capture 排他へ参加させる。
    pub fn download_f32(&self, c_dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        self.with_driver_call(|| crate::memory::readback(&self.stream, c_dev))
    }

    /// ストリームの完了を明示的に待つ（`gemm_mma_tf32.rs::CudaMmaTf32Gemm
    /// ::synchronize` と同じ理由の公開 API）。codex-review P0 指摘対応
    /// （PR #1390 再々修正）: `Self::with_driver_call` で capture 排他へ
    /// 参加させる。
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.with_driver_call(|| Ok(self.stream.synchronize()?))
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

    /// 全要素が有限かつ TF32 丸めでオーバーフローしない A・B を受理する
    /// （`validate_tf32x3_finite_input` の非破壊契約。codex-review 指摘・
    /// PR #1400）。`f32::MAX`／`f32::MIN` は仮数が全 1 のため TF32 丸めで
    /// 必ずオーバーフローするので、ここでは含めない
    /// （`validate_tf32x3_finite_input_rejects_tf32_rounding_overflow_positive`
    /// 参照）。
    #[test]
    fn validate_tf32x3_finite_input_accepts_all_finite() {
        let a = vec![1.0f32, -2.0, 0.0, 1e30, -1e30, 1e-30];
        let b = vec![3.0f32, 4.0, -5.0, 6.0, 7.0, 8.0];
        assert!(validate_tf32x3_finite_input(&a, &b).is_ok());
    }

    /// [`tf32_round_overflows`] の境界値判定を単体で確認する
    /// （codex-review 指摘・PR #1400 スレッド PRRT_kwDOTuUCJc6f01d6）。
    /// `f32::MAX` は仮数が全 1 のため必ず丸め上げが発生しオーバーフロー
    /// する。通常の有限値（`0.0`・`1.0`・`f32::MIN_POSITIVE` 等）や
    /// 非有限値（本関数の呼び出し前提の対象外だが false を返す）は
    /// 該当しない。
    #[test]
    fn tf32_round_overflows_detects_boundary_values() {
        assert!(tf32_round_overflows(f32::MAX));
        assert!(tf32_round_overflows(f32::MIN));
        assert!(!tf32_round_overflows(0.0));
        assert!(!tf32_round_overflows(1.0));
        assert!(!tf32_round_overflows(-1.0));
        assert!(!tf32_round_overflows(1e30));
        assert!(!tf32_round_overflows(-1e30));
        assert!(!tf32_round_overflows(f32::MIN_POSITIVE));
    }

    /// codex-review 指摘の再現ケース: A に `f32::MAX` が含まれる場合、
    /// TF32 丸めオーバーフローとして拒否される（`hi = round_tf32(v)` が
    /// `+inf` になり `v - hi` が `-inf` へ汚染されるのを未然に防ぐ。
    /// PR #1400 スレッド PRRT_kwDOTuUCJc6f01d6・
    /// `CudaError::NonFiniteInput` ドキュメンテーションコメント）。
    #[test]
    fn validate_tf32x3_finite_input_rejects_tf32_rounding_overflow_positive() {
        let a = vec![1.0f32, 0.0, 0.0, f32::MAX];
        let b = vec![0.0f32; 16];
        let err = validate_tf32x3_finite_input(&a, &b)
            .expect_err("f32::MAX lhs は TF32 丸めオーバーフローとして拒否されるべき");
        match err {
            CudaError::NonFiniteInput { detail } => {
                assert!(
                    detail.contains("lhs (a)") && detail.contains("TF32"),
                    "detail should identify the lhs operand and TF32 overflow: {detail}"
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
    }

    /// 上記の RHS 側・負符号側の再現ケース（`f32::MIN` は `-f32::MAX`
    /// で同様に仮数が全 1 のためオーバーフローする）。
    #[test]
    fn validate_tf32x3_finite_input_rejects_tf32_rounding_overflow_negative_rhs() {
        let a = vec![1.0f32; 4];
        let b = vec![f32::MIN, 0.0, 0.0, 0.0];
        let err = validate_tf32x3_finite_input(&a, &b)
            .expect_err("f32::MIN rhs は TF32 丸めオーバーフローとして拒否されるべき");
        match err {
            CudaError::NonFiniteInput { detail } => {
                assert!(
                    detail.contains("rhs (b)") && detail.contains("TF32"),
                    "detail should identify the rhs operand and TF32 overflow: {detail}"
                );
            }
            other => panic!("unexpected error variant: {other}"),
        }
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
