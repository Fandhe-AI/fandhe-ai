//! `TypedOps<half::f16>` の Metal 実装（イシュー #1705・親 #1651・
//! `docs/backend-dtype-dispatch-design.md` §4.2「最小集合 8 演算」）。
//!
//! # `gemm`: 既存 `dispatch_f16_auto_unverified`（動的タイル選択）への内部結線
//!
//! `crate::gemm::MetalGemm::dispatch_f16_auto_unverified`（`tile::
//! select_for_device` によるタイル構成の動的選択 →
//! `dispatch_f16_tiled_unverified`。`_unverified` suffix・
//! `#[doc(hidden)]` は #798 以来の「精度未検証カーネルを検証済み
//! production 入口へ直接結線しない」安全境界。イシュー #1651 の
//! 承認事項 §7-4 は本イシューでは未承認のため suffix・属性とも維持する）
//! は、REQ-2 統一複合判定の検証を担う `tests/gemm_f16_auto_parity.rs`
//! （`#[ignore]`。#799 実機セッションで M4 Max 実機 8/8 PASS 記録済み。
//! `docs/perf/metal-f16-vs-mps-f16.md:378`）を持ちながら、本イシュー
//! 以前は `facade`／`ops::MetalBackendOps`（f32 `BackendOps::gemm`）／
//! `dispatch_auto`（真の production 自動経路）のいずれからも到達
//! 不能だった（`lib.rs` モジュール doc「#798」節参照）。
//!
//! 本ファイルの `gemm` は `context_cache::cached_context`／
//! `cached_gemm`（プロセス内キャッシュ。イシュー #930）経由で
//! `MetalGemm` を取得し `dispatch_f16_auto_unverified` へそのまま
//! 委譲する薄いパススルーであり、`tile::select_for_device` のタイル
//! 選択ロジック自体は一切複製しない。**`Tensor<f16>` という型でのみ
//! 到達する経路**（`TypedOps<f16>::gemm` を呼ぶには `Tensor<f16>` を
//! 構築する必要がある。REQ-11 整合）であり、`f32` 側
//! `BackendOps::gemm`（`ops::MetalBackendOps::gemm`。
//! `dispatch_auto`／`dispatch_strided_bias_act_prepared` 経由）から
//! `dispatch_f16_auto_unverified` へ暗黙に迂回する経路は存在しない
//! （fail-closed。`.claude/rules/security.md` A08。「未検証カーネルを
//! 検証済み production 入口へ直接結線しない」既存の安全境界とは
//! 別次元の話であり矛盾しない）。`f32` への暗黙フォールバックも
//! しない。
//!
//! # 残り 7 演算: f32 昇格 → 既存 Metal `BackendOps` カーネル → 1 回丸め
//!
//! `add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` は
//! `crates/backend-cpu/src/typed_f16.rs`（#1698）・
//! `crates/backend-cuda/src/typed_f16.rs`（#1703）と同じ合成方式を
//! 採る:
//!
//! 1. **昇格**: 入力 `Tensor<f16>` を [`Tensor::host_slice`]（contiguous
//!    なら借用・非 contiguous な view はここで 1 回だけ実体化）で
//!    読み出し `f16::to_f32` で `Tensor<f32>` を構築する
//! 2. **演算**: 既存 `<MetalBackendOps as BackendOps>::{add, mul, relu,
//!    exp, tanh, sum, max}` へそのまま委譲する
//!    （`elementwise::MetalElementwise` の f32 カーネル本体・
//!    ブロードキャスト規則・エラー変換をすべて継承する）
//! 3. **丸め**: 出力 `Tensor<f32>` を `f16::from_f32`（IEEE 754 最近接
//!    偶数丸め）で `Tensor<f16>` へ 1 回だけ丸める
//!
//! この構成により、各メソッドは構造的に
//! `f16::from_f32(BackendOps::op(upcast(x)))`（要素ごと bit 一致）
//! という不変条件を満たす。`elementwise.rs`／`gemm.rs`（関数本体）の
//! f32 実装は本ファイル追加によって一切変更されない。
//!
//! # `sum`／`max` は `Unsupported` を継承する（ホストで代替しない）
//!
//! `ops::MetalBackendOps::sum`／`max`（f32）は reduction カーネル
//! 未実装のため常に [`BackendError::Unsupported`] を返す
//! （`ops.rs` 冒頭コメント「汎用 reduction（`sum`／`max`）は未実装の
//! まま `Unsupported` を返す」参照）。本ファイルの `sum`／`max` は
//! 上記 3 段構成（昇格 → `BackendOps::sum`／`max` へ委譲 → 丸め）を
//! そのまま適用するだけで、委譲先が常に `Unsupported` を返すため
//! 構造的に同じ結果になる——ホスト側で reduction を計算して
//! `Unsupported` を偽装しない（バックエンド内部で CPU 計算を隠す
//! silent fallback を避ける。レイヤリング上、ホストフォールバックは
//! `autodiff` 側の責務であり `backend-metal` の責務ではない）。
//! Metal f32 reduction カーネルが将来実装されれば、本ファイルは
//! 変更なしでそのまま有効になる（out-of-scope-tracking.md 対象。
//! 後続イシュー起票の要否はユーザー承認後に判断する）。
//!
//! # スコープ外
//!
//! Metal `half` 専用 elementwise／reduction カーネル（H2D 転送量削減の
//! 性能最適化）・f16 NT/TN strided 入口（非 contiguous view の性能
//! 最適化）・`_unverified`／`#[doc(hidden)]` の解除（§7-4 の別途承認
//! 事項）・bf16（#1706）・`Var`／`Tape`／VJP・facade 公開面への昇格・
//! `MemoryOps`／`DeviceBuffer<f16>` 常駐経路（段階 B）は対象外
//! （`docs/backend-dtype-dispatch-design.md` §7〜§8・親 #1651 の
//! 後続イシュー分担）。

use half::f16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps, gemm_out_shape};

use crate::context_cache;
use crate::error::MetalError;
use crate::memory::map_metal_error;
use crate::ops::MetalBackendOps;

/// `Tensor<f16>` を `Tensor<f32>` へソフトウェア変換で昇格する
/// （`crates/backend-cpu/src/typed_f16.rs::upcast_f16` と同型）。
///
/// [`Tensor::host_slice`] で読み出す（contiguous なら借用・非
/// contiguous な view は 1 回だけ実体化）ため、呼び出し元の非
/// contiguous view（`transpose_2d`／`narrow`／`broadcast_to` 等）を
/// そのまま渡してよい。shape 自体は `f16`→`f32` で変わらないため
/// `Tensor::new` の失敗は契約上到達しないはずだが、`Tensor` 実装の
/// 不変条件違反に対する fail-safe として型付きエラーで受ける。
fn upcast_f16(t: &Tensor<f16>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|v| v.to_f32()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_f16::upcast_f16: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `Tensor<f32>` を `Tensor<f16>` へ最近接偶数丸めで降格する
/// （[`upcast_f16`] と対になるヘルパー）。
fn downcast_f32(t: &Tensor<f32>) -> Result<Tensor<f16>, BackendError> {
    let rounded: Vec<f16> = t.host_slice().iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_f16::downcast_f32: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

impl TypedOps<f16> for MetalBackendOps {
    /// `MetalGemm::dispatch_f16_auto_unverified`（動的タイル選択の
    /// f16 自動経路）への薄いパススルー（モジュール doc 参照）。
    ///
    /// デバイスに触れる前に shape 検証を行い、失敗は
    /// `ShapeMismatch` として即座に返す（`ops::MetalBackendOps::gemm`
    /// と同じ「デバイス呼び出し前に事前検証」契約）。
    fn gemm(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let out_shape =
            gemm_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        let a_owned = a.host_slice();
        let b_owned = b.host_slice();

        let ctx = context_cache::cached_context().map_err(map_metal_error)?;
        let gemm = context_cache::cached_gemm(&ctx)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        let out = gemm
            .dispatch_f16_auto_unverified(&ctx, &a_owned, &b_owned, m, n, k)
            .map_err(|e: MetalError| BackendError::KernelLaunchFailed(e.to_string()))?;
        Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
    }

    /// f16 入力を f32 へ昇格し、既存 `BackendOps::add`（ブロードキャスト
    /// 規則は f32 版と同一）へ委譲してから f16 へ丸める。
    fn add(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let b32 = upcast_f16(b)?;
        let out32 = BackendOps::add(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// `add` と同型。既存 `BackendOps::mul` へ委譲する。
    fn mul(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let b32 = upcast_f16(b)?;
        let out32 = BackendOps::mul(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::relu` へ委譲する。
    fn relu(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::relu(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::exp` へ委譲する。f16 表現範囲
    /// （`|x| <= 65504`）を超える結果は `f16::from_f32` により
    /// `f16::INFINITY`／`f16::NEG_INFINITY` へ丸められる（IEEE 754
    /// 挙動。CPU／CUDA 版と同じ既知事項）。
    fn exp(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::exp(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::tanh` へ委譲する。
    fn tanh(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::tanh(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::sum`（f32。reduction カーネル未実装のため
    /// 常に `Unsupported`）へ委譲する。モジュール doc「`sum`／`max` は
    /// `Unsupported` を継承する」参照。
    fn sum(&self, a: &Tensor<f16>, dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::sum(self, &a32, dim)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::max`（f32。同上）へ委譲する。
    fn max(&self, a: &Tensor<f16>, dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::max(self, &a32, dim)?;
        downcast_f32(&out32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f16> {
        let d: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        Tensor::new(d, shape).unwrap()
    }

    /// `typed_ops_f16()` accessor が `Some` を返し、`typed_ops_f64()`
    /// は既定 `None`（本ファイルが `impl TypedOps<f64>` を追加しない
    /// ため）のままであることを確認する（イシュー #1705 の中核契約。
    /// デバイス非接触）。
    #[test]
    fn typed_ops_f16_is_some_and_typed_ops_f64_is_none() {
        let ops = MetalBackendOps::new();
        assert!(BackendOps::typed_ops_f16(&ops).is_some());
        assert!(BackendOps::typed_ops_f64(&ops).is_none());
    }

    /// shape 不一致は `cached_context`（デバイス取得）に一切触れる前に
    /// `ShapeMismatch` を返す。
    #[test]
    fn gemm_rejects_shape_mismatch_before_touching_device() {
        let ops = MetalBackendOps::new();
        let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let b = t(&[1.0, 2.0], &[2, 1]);
        let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    /// `add`／`mul` はブロードキャスト不能な shape 不一致を、デバイスに
    /// 触れる前に `elementwise_binary`（`ops.rs`）の
    /// `broadcast_with` が `ShapeMismatch` として返す
    /// （`crates/backend-cuda/src/typed_f16.rs` と同型の是正——戻り値の
    /// `Result` を実際に検査し、shape 検証欠落や常時エラー化を検出
    /// できるようにする）。
    #[test]
    fn add_mul_reject_shape_mismatch_before_touching_device() {
        let ops = MetalBackendOps::new();
        let a = t(&[1.0, -2.0, 3.0], &[3]);
        let b = t(&[3.0, 4.0], &[2]);
        let add_err = TypedOps::<f16>::add(&ops, &a, &b).unwrap_err();
        assert!(matches!(add_err, BackendError::ShapeMismatch(_)));
        let mul_err = TypedOps::<f16>::mul(&ops, &a, &b).unwrap_err();
        assert!(matches!(mul_err, BackendError::ShapeMismatch(_)));
    }

    /// `sum`／`max` は Metal f32 reduction 未実装のため、有効な入力・
    /// 範囲外 `dim` のいずれでも `BackendOps::sum`／`max`（f32）と
    /// 同じ結果クラス（`Unsupported`）を返すことを確認する（モジュール
    /// doc「`sum`／`max` は `Unsupported` を継承する」の直接検証。
    /// Metal f32 reduction が将来実装された場合にそのまま有効な契約
    /// として書く: f32／f16 の結果クラスが一致することのみを検査する）。
    #[test]
    fn sum_max_inherit_f32_backend_ops_result_class() {
        let ops = MetalBackendOps::new();
        let a16 = t(&[1.0, 5.0, 3.0, 2.0], &[2, 2]);
        let a32 = upcast_f16(&a16).unwrap();

        for dim in [None, Some(0), Some(5)] {
            let sum16 = TypedOps::<f16>::sum(&ops, &a16, dim);
            let sum32 = BackendOps::sum(&ops, &a32, dim);
            assert_eq!(
                sum16.is_ok(),
                sum32.is_ok(),
                "sum(dim={dim:?}) の結果クラスが f16/f32 で不一致: f16={sum16:?} f32={sum32:?}"
            );
            if let (Ok(v16), Ok(v32)) = (&sum16, &sum32) {
                let rounded32: Vec<f32> = v32
                    .host_slice()
                    .iter()
                    .map(|&x| f16::from_f32(x).to_f32())
                    .collect();
                assert_eq!(
                    v16.host_slice()
                        .iter()
                        .map(|v| v.to_f32())
                        .collect::<Vec<_>>(),
                    rounded32
                );
            }

            let max16 = TypedOps::<f16>::max(&ops, &a16, dim);
            let max32 = BackendOps::max(&ops, &a32, dim);
            assert_eq!(
                max16.is_ok(),
                max32.is_ok(),
                "max(dim={dim:?}) の結果クラスが f16/f32 で不一致: f16={max16:?} f32={max32:?}"
            );
        }
    }

    /// [`upcast_f16`]／[`downcast_f32`] の往復が f16 表現可能な既知値で
    /// bit 一致することを確認する（デバイス不要）。
    #[test]
    fn upcast_downcast_roundtrip_matches_known_values() {
        let a = t(&[1.0, -2.5, 0.0, 4.0], &[4]);
        let a32 = upcast_f16(&a).unwrap();
        assert_eq!(a32.host_slice().to_vec(), vec![1.0, -2.5, 0.0, 4.0]);
        let back = downcast_f32(&a32).unwrap();
        assert_eq!(
            back.host_slice()
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            vec![1.0, -2.5, 0.0, 4.0]
        );
    }
}
