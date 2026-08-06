//! GEMM 自動経路選択の入口（TASK-11.2b・#68）。
//!
//! `tensor_core::dispatch::select_gemm_kernel`（規則の純関数実装。
//! `docs/dispatch-rules-design.md` §5 の決定表）を `backend-cuda` の
//! 既存 GEMM 実装（`gemm.rs::CudaGemm`〈naive／tiled〉・
//! `gemm_wmma.rs::CudaWmmaGemm`〈WMMA f16 Tensor Core〉）へ結線する。
//! `BackendOps` trait 実装（TASK-1.9c・#46）はこの `CudaGemmAuto` を
//! `BackendOps::gemm` から呼ぶだけの構成にできる（実装計画 §3.2
//! 「#46 との境界」）。
//!
//! `DeviceCaps` は [`CudaGemmAuto::new`] 実行時（デバイス初期化）に 1 回
//! だけ `CudaDevice::compute_capability()` から構築し、以降の
//! `run_f32`／`run_f16` 呼び出しでは FFI 照会を繰り返さない
//! （`docs/dispatch-rules-design.md` §2.1「判定タイミング」）。
//!
//! フォールバック連鎖は `MatrixUnit → Tiled → Naive`（§5.2）。
//! `CudaWmmaGemm::new` が cc ゲート非対応・NVRTC コンパイル失敗のいずれか
//! で `Err` を返した場合は WMMA を候補から外し、以降 `run_f16` は常に
//! tiled 経路を使う（fail-safe。`panic!`／`unwrap()` を使わない）。

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::CudaGemm;
use crate::gemm_wmma::CudaWmmaGemm;
use half::f16;
use tensor_core::dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};

/// naive／tiled／WMMA の全 GEMM カーネルを保持し、`select_gemm_kernel`
/// の判定結果に従って呼び分ける自動経路選択の入口。
///
/// `wmma` は `Option` とし、cc ゲート非対応・NVRTC コンパイル失敗時は
/// `None` のまま保持する（fail-safe。上記モジュールコメント参照）。
/// `caps` は `wmma` の有無とは独立に「cc ゲートを満たすか」だけを表す
/// （`select_gemm_kernel` の判定材料は cc のみであり、コンパイル成否は
/// 別軸のフォールバックとして扱う。`run_f16` 内のコメント参照）。
pub struct CudaGemmAuto {
    gemm: CudaGemm,
    wmma: Option<CudaWmmaGemm>,
    caps: DeviceCaps,
}

impl CudaGemmAuto {
    /// `device` 上で naive／tiled GEMM（[`CudaGemm`]）を構築し、cc ゲート
    /// を満たす場合のみ WMMA f16 GEMM（[`CudaWmmaGemm`]）の構築を試みる。
    ///
    /// naive／tiled の構築失敗（NVRTC 不在等）はそのまま `Err` として
    /// 呼び出し元へ伝播する（naive／tiled は最終フォールバックであり
    /// 失敗を握り潰すべきではない）。WMMA の構築失敗（cc ゲート非対応の
    /// `TensorCoreUnsupported`・NVRTC コンパイル失敗のいずれも）は
    /// `wmma = None` として握り潰し、`run_f16` を tiled へ倒す
    /// （fail-safe。`docs/dispatch-rules-design.md` §2.2）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let gemm = CudaGemm::new(device)?;
        let wmma = CudaWmmaGemm::new(device).ok();
        let caps = DeviceCaps::cuda(device.compute_capability());
        Ok(Self { gemm, wmma, caps })
    }

    /// f32 GEMM を自動経路選択で実行する。
    ///
    /// 現時点の決定表（`select_gemm_kernel`）では f32 は常に
    /// [`KernelKind::Tiled`] を返す（TF32 経路は #62／#186 の実測・承認
    /// まで既定採用を保留。§4）。将来 TF32 経路が有効化された際に
    /// `KernelKind::MatrixUnit` 分岐が生きるよう、match 式で明示的に
    /// 経路を切り分ける構造にしておく（未実装分岐は tiled へフォール
    /// バックし `unreachable!` は使わない。fail-safe）。
    pub fn run_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f32>, CudaError> {
        let shape = GemmShape::new(m, n, k);
        match select_gemm_kernel(&self.caps, shape, DType::F32) {
            // TF32/f32 Tensor Core 経路は未実装（#62）のため、決定表が
            // 万一 MatrixUnit を返しても tiled へ倒す（現状の決定表は
            // f32 に対し常に Tiled を返すため、この分岐は到達しない
            // 契約だが、`select_gemm_kernel` 側の将来変更に対する
            // fail-safe として保持する）。
            KernelKind::MatrixUnit | KernelKind::Tiled => self.gemm.run_tiled_f32(a, b, m, n, k),
            KernelKind::Naive => self.gemm.run_naive_f32(a, b, m, n, k),
        }
    }

    /// f16 GEMM を自動経路選択で実行する。
    ///
    /// `select_gemm_kernel` が `MatrixUnit` を返しても `wmma` が `None`
    /// （構築失敗。[`Self::new`] 参照）なら tiled へフォールバックする
    /// （§5.2 のフォールバック連鎖 `MatrixUnit → Tiled → Naive` の
    /// 中間段。`self.gemm`〈naive／tiled〉は `new` 成功時点で必ず存在
    /// するため、tiled 自体が失敗するケースはカーネル起動時エラー
    /// （`CudaError`）としてそのまま呼び出し元へ返る）。
    pub fn run_f16(
        &self,
        a: &[f16],
        b: &[f16],
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<Vec<f16>, CudaError> {
        let shape = GemmShape::new(m, n, k);
        match select_gemm_kernel(&self.caps, shape, DType::F16) {
            KernelKind::MatrixUnit => match &self.wmma {
                Some(wmma) => wmma.run_f16(a, b, m, n, k),
                // cc ゲートは満たすが WMMA 構築が失敗していた場合の
                // fail-safe フォールバック（NVRTC コンパイル失敗等）。
                None => self.gemm.run_tiled_f16(a, b, m, n, k),
            },
            KernelKind::Tiled => self.gemm.run_tiled_f16(a, b, m, n, k),
            KernelKind::Naive => self.gemm.run_naive_f16(a, b, m, n, k),
        }
    }
}
