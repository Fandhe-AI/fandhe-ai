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

use std::num::NonZeroU32;

use cudarc::driver::sys::CUdevice_attribute;
use half::f16;
use tensor_core::dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};

use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::gemm::CudaGemm;
use crate::gemm_wmma::CudaWmmaGemm;
use crate::kernels_mma::{MMA_SHARED_MEM_BYTES, MMA_STATIC_SMEM_LIMIT_BYTES};
use crate::nvrtc::derive_pipeline_stages;

/// 静的 `__shared__`（動的 SMEM opt-in 非使用）構成のカーネルが従う
/// per-block SMEM 上限。[`MMA_STATIC_SMEM_LIMIT_BYTES`] から
/// 直接導出し、49,152（48KiB）という値をここで独立にハードコードしない
/// （以前は `gemm_auto.rs` と `kernels_mma.rs` の双方が同じリテラルを
/// 個別に持っており、どちらか一方だけが変更されても検出されない静かな
/// 乖離リスクがあった。#521 レビュー指摘）。[`derive_stages_for_device`]
/// はデバイス実測値がこれを上回っても静的上限でクランプする（本モジュール
/// 冒頭コメント 3.1 節の安全側判断: NVRTC コンパイル自体がこの上限超過で
/// 失敗するため、予算側で先に fail-closed に遮断する）。
pub const STATIC_SMEM_BUDGET_CAP_BYTES: u64 = MMA_STATIC_SMEM_LIMIT_BYTES as u64;

// 実使用量が予算上限に収まっていることのコンパイル時契約検査（上記の
// 「唯一の真実源」導出そのものは値の重複を構造的に排除しているが、これは
// 別の検査: 現在のタイル構成〈MMA_SHARED_MEM_BYTES〉が STATIC_SMEM_BUDGET_
// CAP_BYTES を超えていないかを機械検出する使用量ガードである。実機
// コンパイルできない環境でも `cargo build` の時点で検出できる代替チェック
// （`kernels_mma.rs` 側の同種 assert と対になる二重チェック）。
const _: () = assert!(
    MMA_SHARED_MEM_BYTES as u64 <= STATIC_SMEM_BUDGET_CAP_BYTES,
    "gemm_auto::STATIC_SMEM_BUDGET_CAP_BYTES に対して \
     kernels_mma::MMA_SHARED_MEM_BYTES の実使用量が超過している"
);

/// f16 の要素バイト幅（`kernels_mma.rs` の f16 * 2B と同じ根拠）。
/// [`derive_stages_for_device`] の `bytes_per_element` 導出で使う
/// コンパイル時定数。`NonZeroU32::new` は `Option` を返すため `match` で
/// 分解するが、リテラルが非ゼロであることはコンパイル時に確定しており
/// 実行時に `None` 分岐へ到達することはない（#521 レビュー指摘: 旧実装は
/// これを実行時エラーとして扱っており、到達不能なエラーパスを持っていた）。
const BYTES_PER_ELEMENT_F16: NonZeroU32 = match NonZeroU32::new(2) {
    Some(v) => v,
    None => panic!("BYTES_PER_ELEMENT_F16: 2 must be non-zero"),
};

/// F32 の要素バイト幅（将来の mma/tf32 経路を見越した 4 バイト）。
/// [`BYTES_PER_ELEMENT_F16`] と同じ理由でコンパイル時定数とする。
const BYTES_PER_ELEMENT_F32: NonZeroU32 = match NonZeroU32::new(4) {
    Some(v) => v,
    None => panic!("BYTES_PER_ELEMENT_F32: 4 must be non-zero"),
};

/// `device` の `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を実行時
/// に取得し、[`STATIC_SMEM_BUDGET_CAP_BYTES`] でクランプしたうえで
/// [`derive_pipeline_stages`]（`nvrtc.rs`・C-8）へ渡してパイプライン段数を
/// 導出する結線ヘルパー（イシュー #521 実装計画 §4）。
///
/// SMEM 容量をハードコード定数として持たない理由: `docs/perf/
/// sm121-device-attributes.md` の sm_121 実機実測は「未実測・要実機実行」
/// のまま安全側クローズされており（A-2・#482）、存在しない実測値を推定で
/// 定数化することは同ドキュメントの方針に反する。CUTLASS
/// `StageCountAutoCarveout` と同じ「実デバイスの残量から自動算出・固定値
/// ハードコードなし」の思想を採る。
///
/// 属性取得の失敗（例: デバイス問い合わせ自体の driver API エラー）は
/// `CudaError::Driver` として呼び出し元へそのまま伝播する（fail-closed。
/// `From<DriverError>` は `error.rs` に既存）。
///
/// 本関数は導出ロジックと B-1（#492）可変化カーネルとの接続を実証する
/// ためのものであり、既定の本番 GEMM 経路（[`CudaGemmAuto::run_f16`]・
/// `gemm_mma.rs::CudaMmaGemm::run_f16` の `MMA_STAGES=3` 固定）へ導出値を
/// 適用する判断は本イシューのスコープ外（C-10・#527 最良構成選定に委ねる。
/// カーネルソース自体は一切変更していないため、この関数は既定経路の
/// 実行結果・parity ベースラインに影響しない）。
pub fn derive_stages_for_device(
    device: &CudaDevice,
    block_m: NonZeroU32,
    block_n: NonZeroU32,
    block_k: NonZeroU32,
    dtype: DType,
) -> Result<NonZeroU32, CudaError> {
    let raw_attr = device
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
    // `attribute()` は driver API 契約上 `i32` を返す（cudarc 0.19.8
    // `CudaContext::attribute`）。`MAX_SHARED_MEMORY_PER_BLOCK` は物理的に
    // 負値になり得ないが、ドライバが不正値を返すケースを暗黙の 0 丸め
    // （`unwrap_or(0)`）で握り潰すと、後続の `derive_pipeline_stages` が
    // 返す `min_required` 未達エラーが「予算 0」という誤解を招く診断に
    // なる。`TryFrom` 失敗を明示的な `InvalidKernelDescriptor` として
    // 伝播し、fail-closed のまま原因を追跡可能にする。
    let raw_attr_u64 = u64::try_from(raw_attr).map_err(|_| CudaError::InvalidKernelDescriptor {
        detail: format!(
            "derive_stages_for_device: CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK \
             returned a negative value ({raw_attr}), which cannot be a valid SMEM budget"
        ),
    })?;
    let smem_budget_bytes = raw_attr_u64.min(STATIC_SMEM_BUDGET_CAP_BYTES);

    // `bytes_per_element` はカーネル入出力 dtype のバイト幅（`kernels_mma.rs`
    // の f16 * 2B と同じ根拠）。F32 は将来の mma/tf32 経路を見越して
    // 4 バイトとする。`DType` は非網羅的でない列挙（`tensor_core::dispatch`）
    // であるため match は両変種を明示し、新変種追加時はコンパイルエラーで
    // 見落としを検出する（`_ =>` フォールバックは持たない）。
    //
    // 両アームともコンパイル時定数（`BYTES_PER_ELEMENT_F16`／`_F32`）を返す
    // ため、`NonZeroU32::new` の失敗分岐はコンパイル時に確定して排除される
    // （旧実装は実行時 `ok_or_else` で検査しており、両アームとも非ゼロの
    // リテラルであるため到達不能な `Err` 分岐を持つだけだった。#521 レビュー
    // 指摘）。
    let bytes_per_element = match dtype {
        DType::F16 => BYTES_PER_ELEMENT_F16,
        DType::F32 => BYTES_PER_ELEMENT_F32,
    };

    derive_pipeline_stages(
        block_m,
        block_n,
        block_k,
        bytes_per_element,
        smem_budget_bytes,
    )
}

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
