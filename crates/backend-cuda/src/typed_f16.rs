//! `TypedOps<half::f16>` の CUDA 実装（イシュー #1703・親 #1650・
//! `docs/backend-dtype-dispatch-design.md` §4.2「最小集合 8 演算」）。
//!
//! # `gemm`: 既存 Tensor Core 自動選択経路（`CudaGemmAuto::run_f16`）への結線
//!
//! `crate::gemm_auto::CudaGemmAuto::run_f16`（`mma.sync` 優先 →
//! WMMA → tiled → naive のフォールバック連鎖。`docs/dispatch-rules-
//! design.md` §5.6）は本イシュー以前、`facade`／`backend-cuda::ops`／
//! `bench-harness` のいずれからも到達不能だった（`gemm_auto.rs`
//! モジュール doc「`CudaGemmAuto` を構築する本番経路は現状存在せず」
//! 参照）。本ファイルの `gemm` は、`context_cache::cached_gemm_auto`
//! （プロセス内キャッシュ。初回呼び出し時にのみ NVRTC コンパイルを
//! 支払う遅延構築）経由でこのスイートを取得し `run_f16` へそのまま
//! 委譲する薄いパススルーであり、カーネル選択ロジック自体（形状ゲート・
//! mma/wmma/tiled/naive の優先順位）は一切複製しない。`f32` 側
//! `BackendOps::gemm`（`context_cache::cached_gemm` 経由の tiled 固定
//! 経路）には一切影響しない（別スイート・別キャッシュキー）。
//!
//! `f32` への暗黙フォールバックはしない（fail-closed。他の演算と同じ
//! 契約。`.claude/rules/security.md` A08）。
//!
//! # 残り 7 演算: f32 昇格 → 既存 CUDA `BackendOps` カーネル → 1 回丸め
//!
//! `add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` は
//! `crates/backend-cpu/src/typed_f16.rs`（#1698）と同じ合成方式を採る:
//!
//! 1. **昇格**: 入力 `Tensor<f16>` を [`Tensor::host_slice`]（contiguous
//!    なら借用・非 contiguous な view はここで 1 回だけ実体化）で読み出し
//!    `f16::to_f32` で `Tensor<f32>` を構築する
//! 2. **演算**: 既存 `<CudaBackendOps as BackendOps>::{add, mul, relu, exp,
//!    tanh, sum, max}` へそのまま委譲する（`elementwise::CudaElementwise`／
//!    `reduce::CudaReduce` の f32 カーネル本体・ブロードキャスト規則・
//!    エラー変換をすべて継承する）
//! 3. **丸め**: 出力 `Tensor<f32>` を `f16::from_f32`（IEEE 754 最近接
//!    偶数丸め）で `Tensor<f16>` へ 1 回だけ丸める
//!
//! この構成により、各メソッドは構造的に
//! `f16::from_f32(BackendOps::op(upcast(x)))`（要素ごと bit 一致）という
//! 不変条件を満たす。`elementwise.rs`／`reduce.rs`／`gemm.rs`／
//! `gemm_auto.rs` の f32 実装本体は本ファイル追加によって一切変更されない。
//!
//! # スコープ外
//!
//! CUDA `__half` 専用 elementwise／reduction カーネル（H2D 転送量削減の
//! 性能最適化）・bf16（#1704）・Metal（#1705）・`Var`／`Tape`／VJP・
//! facade 公開面への昇格は対象外（`docs/backend-dtype-dispatch-
//! design.md` §7〜§8・親 #1650 の後続イシュー分担）。

use half::f16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps, matmul_out_shape};

use crate::context_cache;
use crate::ops::CudaBackendOps;

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

impl TypedOps<f16> for CudaBackendOps {
    /// `CudaGemmAuto::run_f16`（`mma.sync` 優先の Tensor Core 自動選択
    /// 経路）への薄いパススルー（モジュール doc 参照）。
    ///
    /// driver に触れる前に shape 検証・`u32` 変換を行い、失敗は
    /// `ShapeMismatch`／`KernelLaunchFailed` として即座に返す（他の
    /// `CudaBackendOps` メソッドと同じ「driver 呼び出し前に事前検証」
    /// 契約。`self.with_driver_call` の外側で完結する）。
    fn gemm(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let out_shape =
            matmul_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;

        let m = u32::try_from(a.shape()[0]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: m exceeds u32 range".to_string())
        })?;
        let k = u32::try_from(a.shape()[1]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: k exceeds u32 range".to_string())
        })?;
        let n = u32::try_from(b.shape()[1]).map_err(|_| {
            BackendError::KernelLaunchFailed("gemm: n exceeds u32 range".to_string())
        })?;

        let a_owned = a.host_slice();
        let b_owned = b.host_slice();

        let auto = self.with_driver_call(
            &[],
            |e| BackendError::CudaUnavailable(e.to_string()),
            || {
                let device = self.device_handle_raw()?;
                context_cache::cached_gemm_auto(&device)
            },
        )?;
        let out = self.with_driver_call(
            &[],
            |e| BackendError::KernelLaunchFailed(e.to_string()),
            || auto.run_f16(&a_owned, &b_owned, m, n, k),
        )?;
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
    /// 挙動。CPU 版と同じ既知事項）。
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

    /// 既存 `BackendOps::sum`（`reduce::CudaReduce`。f64 アキュムレータ
    /// 契約）へ委譲する。
    fn sum(&self, a: &Tensor<f16>, dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::sum(self, &a32, dim)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::max`（`reduce::CudaReduce`。厳密選択
    /// `fmaxf` 契約）へ委譲する。
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

    /// `typed_ops_f16()` accessor が `Some` を返し、`typed_ops_bf16()`
    /// は `None` のまま（#1704 の範囲を侵さない契約）であることを
    /// 確認する。
    #[test]
    fn typed_ops_f16_is_some_and_bf16_stays_none() {
        let ops = CudaBackendOps::new(0);
        assert!(BackendOps::typed_ops_f16(&ops).is_some());
        assert!(BackendOps::typed_ops_bf16(&ops).is_none());
    }

    /// shape 不一致は driver に一切触れる前に `ShapeMismatch` を返す
    /// （`CudaUnavailable` にはならない。ordinal 0 が driver 不在環境
    /// でも本テストは成立する）。
    #[test]
    fn gemm_rejects_shape_mismatch_before_touching_driver() {
        let ops = CudaBackendOps::new(0);
        let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let b = t(&[1.0, 2.0], &[2, 1]);
        let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    /// [`upcast_f16`]／[`downcast_f32`] の往復が f16 表現可能な既知値で
    /// bit 一致することを確認する（driver 不要）。
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

    /// `add`／`mul` は要素ごとの二項演算のため、shape 不一致（かつ
    /// ブロードキャスト不能）な入力を渡すと `elementwise_binary`
    /// （`ops.rs::CudaBackendOps::elementwise_binary`）の
    /// `a.broadcast_with(b)` が driver に一切触れる前に
    /// `BackendError::ShapeMismatch` を返す（`gemm` の shape 検証と
    /// 同じ「driver 呼び出し前に事前検証」契約）。ここでは戻り値の
    /// `Result` を実際に検査し、shape 検証が構造的に効いていることを
    /// 確認する（以前は 7 演算すべての戻り値を `_ =` で握り潰しており、
    /// shape 検証欠落や常時エラー化を検出できなかった。#1797 codex-review
    /// 指摘の是正）。
    #[test]
    fn add_mul_reject_shape_mismatch_before_touching_driver() {
        let ops = CudaBackendOps::new(0);
        let a = t(&[1.0, -2.0, 3.0], &[3]);
        let b = t(&[3.0, 4.0], &[2]);
        let add_err = TypedOps::<f16>::add(&ops, &a, &b).unwrap_err();
        assert!(matches!(add_err, BackendError::ShapeMismatch(_)));
        let mul_err = TypedOps::<f16>::mul(&ops, &a, &b).unwrap_err();
        assert!(matches!(mul_err, BackendError::ShapeMismatch(_)));
    }

    /// `sum`／`max` は縮約軸 `dim` の範囲検査（`ops_shape::reduce_out_shape`）
    /// が driver に触れる前に行われるため、rank を超える `dim` は
    /// `BackendError::ShapeMismatch` を返す。戻り値を実際に検査する
    /// （上記と同じ是正）。
    #[test]
    fn sum_max_reject_out_of_range_dim_before_touching_driver() {
        let ops = CudaBackendOps::new(0);
        let a = t(&[1.0, -2.0], &[2]);
        let sum_err = TypedOps::<f16>::sum(&ops, &a, Some(5)).unwrap_err();
        assert!(matches!(sum_err, BackendError::ShapeMismatch(_)));
        let max_err = TypedOps::<f16>::max(&ops, &a, Some(5)).unwrap_err();
        assert!(matches!(max_err, BackendError::ShapeMismatch(_)));
    }

    /// `relu`／`exp`／`tanh` は単項演算で shape 検証点を持たないため、
    /// 有効な入力に対する呼び出しが型シグネチャ通りにコンパイル・実行
    /// できることのみを確認する（driver 不在環境では
    /// `BackendError::CudaUnavailable` を返す契約。それ以外のエラー
    /// 種別が返らないことを検査し、実際の数値〈driver 必須〉は GB10
    /// 実機 `#[ignore]` テストへ引き継ぐ）。
    #[test]
    fn relu_exp_tanh_accept_valid_input_without_unexpected_error_kind() {
        let ops = CudaBackendOps::new(0);
        let a = t(&[1.0, -2.0], &[2]);
        for result in [
            TypedOps::<f16>::relu(&ops, &a),
            TypedOps::<f16>::exp(&ops, &a),
            TypedOps::<f16>::tanh(&ops, &a),
        ] {
            if let Err(e) = result {
                assert!(
                    matches!(
                        e,
                        BackendError::CudaUnavailable(_) | BackendError::KernelLaunchFailed(_)
                    ),
                    "unexpected error variant for valid-shape input: {e:?}"
                );
            }
        }
    }
}
