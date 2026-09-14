//! `TypedOps<half::bf16>` の Metal 実装（イシュー #1706・親 #1651・
//! `docs/backend-dtype-dispatch-design.md` §5「bf16 × Metal」・§13）。
//!
//! # (a) と (b) の分離（本イシューの核心。設計文書 §13.1 参照）
//!
//! 設計文書 §5 は「bf16 × Metal」を「未検証（MSL `bfloat` 型・
//! `simdgroup_matrix` 対応をコンパイルプローブで確認する必要がある）」
//! としていたが、これは 2 つの独立した事実を混同していた:
//!
//! - **(a) `TypedOps<bf16>` の実装可否**: 兄弟イシュー #1704（CUDA
//!   bf16）・#1699（CPU bf16）はいずれもホスト側 bf16⇔f32 変換＋既存
//!   f32 `BackendOps` カーネルへの委譲で実装しており、これは MSL
//!   `bfloat` 型の可用性に**依存しない**。設計 §6 の数値契約（f16／bf16
//!   は f32 昇格・f32 累算・最後に 1 回丸め）自体が最適化ではなく契約
//!   であるため、本モジュールも同じ 3 段ラッパー方式で **実装済み**
//!   （本イシューで解決）
//! - **(b) MSL `bfloat`／`simdgroup_bfloat8x8` の実機可用性**: これが
//!   決めるのは将来のデバイス常駐ネイティブ bf16 経路（bf16 のまま
//!   H2D・`simdgroup_bfloat8x8` GEMM 等）の実現可能性であり、
//!   `crate::typed_bf16_probe_diag_tests` のコンパイルプローブへ切り出す
//!   （**未実測**。本段階のスコープ外〈設計文書 §8「段階 B」〉）
//!
//! `TypedOps<bf16>` の accessor が `None` から `Some(self)` へ変わる／
//! 変わらないの分岐は (b) の結果に依存しない。
//!
//! # 実装方式: ホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの
//! 委譲
//!
//! `crates/backend-cuda/src/typed_bf16.rs`（#1704）・
//! `crates/backend-cpu/src/typed_bf16.rs`（#1699）と同型の 3 段構成を
//! 採る。bf16 デバイス常駐で widen/narrow する専用 MSL カーネルは
//! 追加せず（性能最適化はスコープ外）、`gemm`／`add`／`mul`／`relu`／
//! `exp`／`tanh`／`sum`／`max` の 8 演算とも既存 Metal f32 経路
//! （`gemm.rs`／`elementwise.rs`。H2D〈`MetalBuffer::new_with_data`〉・
//! MSL カーネル・readback を含む）へホスト側変換（[`crate::
//! typed_bf16_convert::upcast_bf16`]／[`downcast_f32`]）の前後で委譲する:
//!
//! 1. **昇格**: [`crate::typed_bf16_convert::upcast_bf16`]
//! 2. **演算**: 既存 `<MetalBackendOps as BackendOps>::{add, mul, relu,
//!    exp, tanh, sum, max}` および `gemm_fp32_strict`（`gemm` ではない。
//!    下記「`gemm` は `gemm_fp32_strict` へ委譲する」参照）
//! 3. **丸め**: [`crate::typed_bf16_convert::downcast_f32`]
//!
//! この構成により、`TypedOps<bf16>` の各メソッドは構造的に
//! `bf16::from_f32(BackendOps::op(upcast(x)))` という不変条件を満たす
//! （`tests/typed_ops_bf16_parity.rs` の層 1 で確認）。既存 f32 経路
//! （`gemm.rs`／`elementwise.rs`／`shaders/**`／`pipeline.rs`／
//! `tensor-core/**`）は本ファイル追加によって一切変更されない
//! （`git diff --stat` で構造的に確認可能）。
//!
//! # `gemm` は `BackendOps::gemm_fp32_strict` へ委譲する（`gemm` ではない）
//!
//! Metal の `BackendOps::gemm`（`ops.rs` の公開経路）は
//! `gemm::MetalGemm::dispatch_auto`（split-K opt-in 経路。既定 ON。
//! `docs/backend-metal-splitk-decision.md`）へ委譲する。`gemm_fp32_strict`
//! の Metal オーバーライドは既定実装（`self.gemm(a, b)` へ委譲。
//! `crates/tensor-core/src/backend_ops.rs:665`）のままのため、現時点
//! では `gemm`／`gemm_fp32_strict` の呼び出しは Metal では実質同一経路
//! だが、CUDA `typed_bf16.rs`（TF32 opt-in を明示的に避ける）との
//! 対称性・「opt-in 精度モードを参照しない」意図を明示するため、
//! CUDA 版と同じく `gemm_fp32_strict` を選ぶ（将来 Metal 側に TF32 相当の
//! opt-in 精度モードが追加された場合でも本ファイルの変更が不要になる）。
//!
//! # `sum`／`max` は `Unsupported` をそのまま伝播する
//!
//! Metal f32 `BackendOps::sum`／`max`（`ops.rs`）は GPU カーネル未実装の
//! ため `Err(BackendError::Unsupported(_))` を返す（out-of-scope-
//! tracking.md 対象。`tests/backend_ops_real_device.rs::
//! reduction_remains_unsupported_without_device_init` が固定済み）。
//! 本モジュールはこれを bf16 へ「昇格」せず（f32 版より高機能にしない）、
//! `upcast_bf16` 後にそのまま `BackendOps::sum`／`max` を呼び
//! `Unsupported` エラーをそのまま返す（f32 契約と同一の fail-closed）。
//!
//! # REQ-8（カーネル境界検査）の適用状況
//!
//! 本ファイルは新規 MSL カーネルを追加しない（既存 f32 カーネルへの
//! ホスト側委譲のみ）ため、新規に境界検査を要する対象は生じない。
//! 既存カーネル（`gemm_simdgroup_tiled`・`elementwise` 系）の境界検査
//! 契約は無変更のまま適用される。
//!
//! # accessor `typed_ops_bf16` は無条件に `Some(self)` を返す
//!
//! `MetalBackendOps::memory_ops` は `static_metal_memory()`（プロセス内
//! シングルトン初期化）を経由するため、デバイス非対応等で `None` へ
//! 縮退しうるが（`ops.rs::memory_ops` doc 参照）、`TypedOps<bf16>` の
//! 実体は `self`（ZST）自身であり、いかなるデバイス初期化にも触れない。
//! 実行時の Metal 実機不在・演算失敗は各演算メソッド内部
//! （`BackendOps::add` 等・`gemm_fp32_strict`）が型付きエラーで返す
//! （他の Metal `BackendOps` メソッドと同じ契約。CUDA `typed_bf16.rs` の
//! accessor 設計と同型）。
//!
//! # 数値・意味論上の明文化事項
//!
//! `crates/backend-cuda/src/typed_bf16.rs`（#1704）と同一（バックエンド
//! 差異はカーネル本体のみで、本ファイルが行うホスト側変換・委譲方式は
//! 共通）:
//!
//! - 丸めは `half::bf16::from_f32`（IEEE 754 最近接偶数丸め）
//! - `sum`（全縮約・軸縮約）は `Unsupported`（上記「`sum`／`max` は
//!   `Unsupported` をそのまま伝播する」参照。CPU／CUDA と異なり Metal
//!   はそもそも f32 reduction カーネルが存在しない）
//! - `add`／`mul` のブロードキャスト規則は f32 版
//!   （`fandhe_ai_tensor_core::elementwise_out_shape`。NumPy 互換）と同一
//! - 非 contiguous view は `host_slice()`（[`crate::typed_bf16_convert`]
//!   内）で実体化してから渡すため、f32 版の stride 読み高速経路は
//!   経由しない（性能最適化は本イシューのスコープ外）
//!
//! # スコープ外
//!
//! MSL `bfloat`／`simdgroup_bfloat8x8` を用いるデバイス常駐ネイティブ
//! bf16 経路（[`crate::typed_bf16_probe_diag_tests`] のプローブ結果が
//! 可の場合の後続候補。設計文書 §8「段階 B」）・Metal f32 `sum`／`max`
//! reduction カーネル自体（未実装）・Metal `TypedOps<f64>`（恒久
//! `Unsupported`）／`TypedOps<f16>`（#1705）・`Var`／`Tape`／VJP・facade
//! 公開面への昇格・M4 Max 実機での `#[ignore]` テスト実測（本
//! エージェント実行環境に到達手段がなく未実測。設計文書 §13.6 記入欄・
//! Mac セッションへ申し送り）。

use half::bf16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

use crate::ops::MetalBackendOps;
use crate::typed_bf16_convert::{downcast_f32, upcast_bf16};

impl TypedOps<bf16> for MetalBackendOps {
    /// bf16 入力を f32 へ昇格し、`BackendOps::gemm_fp32_strict`
    /// （モジュール doc「`gemm` は `gemm_fp32_strict` へ委譲する」参照）
    /// へ委譲してから bf16 へ 1 回丸める。
    fn gemm(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let b32 = upcast_bf16(b)?;
        let out32 = BackendOps::gemm_fp32_strict(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// bf16 入力を f32 へ昇格し、既存 `BackendOps::add`（ブロードキャスト
    /// 規則は f32 版と同一）へ委譲してから bf16 へ丸める。
    fn add(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let b32 = upcast_bf16(b)?;
        let out32 = BackendOps::add(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// `add` と同型。既存 `BackendOps::mul` へ委譲する。
    fn mul(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let b32 = upcast_bf16(b)?;
        let out32 = BackendOps::mul(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::relu` へ委譲する。
    fn relu(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::relu(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::exp` へ委譲する。bf16 表現範囲を超える結果は
    /// `bf16::from_f32` により `bf16::INFINITY`／`bf16::NEG_INFINITY`
    /// へ丸められる（IEEE 754 挙動。モジュール doc「数値・意味論上の
    /// 明文化事項」参照）。
    fn exp(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::exp(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::tanh` へ委譲する。
    fn tanh(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::tanh(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::sum` へ委譲する。Metal f32 `sum` は GPU
    /// カーネル未実装のため常に `Err(BackendError::Unsupported(_))`
    /// を返し、本メソッドもそれをそのまま伝播する（モジュール doc
    /// 「`sum`／`max` は `Unsupported` をそのまま伝播する」参照。bf16
    /// だけ f32 より高機能にしない）。
    fn sum(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::sum(self, &a32, dim)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::max` へ委譲する。`sum` と同じ理由で常に
    /// `Unsupported` を伝播する。
    fn max(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::max(self, &a32, dim)?;
        downcast_f32(&out32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// accessor 経由の到達性（デバイス初期化不要）。`MetalBackendOps`
    /// は ZST・`new()` は常時成功するため実機なしでも検証できる。
    #[test]
    fn typed_ops_bf16_accessor_returns_some_without_device_init() {
        let metal = MetalBackendOps::new();
        let ops: &dyn BackendOps = &metal;
        assert!(
            ops.typed_ops_bf16().is_some(),
            "typed_ops_bf16 accessor must always return Some for MetalBackendOps"
        );
    }

    /// `sum`／`max` がデバイス初期化なしで `Unsupported` を返すことを
    /// 確認する（`tests/backend_ops_real_device.rs::
    /// reduction_remains_unsupported_without_device_init` の bf16 版。
    /// f32 版と同じく `MetalContext::new` を呼ばないため実機不要）。
    #[test]
    fn sum_and_max_remain_unsupported_without_device_init() {
        let metal = MetalBackendOps::new();
        let a = Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(-2.0)], &[1, 2]).unwrap();

        assert!(matches!(
            TypedOps::<bf16>::sum(&metal, &a, None),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            TypedOps::<bf16>::max(&metal, &a, None),
            Err(BackendError::Unsupported(_))
        ));
    }
}
