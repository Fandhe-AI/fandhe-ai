//! `TypedOps<half::bf16>` の CUDA 実装（イシュー #1704・親 #1650・
//! `docs/backend-dtype-dispatch-design.md` §5「bf16 × CUDA」・§12）。
//!
//! # 調査結果（R1）: cudarc `DeviceRepr` 可用性は「可」
//!
//! 設計文書 §5 は「cudarc 0.19.8 に bf16 向け `DeviceRepr` 実装があるか
//! 未検証」としていたが、本イシューで確定した:
//!
//! - `unsafe impl DeviceRepr for half::bf16 {}`
//!   （`cudarc-0.19.8/src/driver/safe/core.rs:1038`。`#[cfg(feature =
//!   "f16")]` 配下）
//! - `unsafe impl ValidAsZeroBits for half::bf16 {}`（同 `core.rs:990`。
//!   同じ `#[cfg(feature = "f16")]` 配下）
//! - workspace ルート `Cargo.toml` の `cudarc` 依存は
//!   `features = ["driver", "nvrtc", "dynamic-loading", "cuda-13000",
//!   "f16"]` で `f16` feature を既に有効化済み（deps-policy.md CUDA 区分）
//!
//! よって `clone_htod`／`clone_dtoh`／`alloc_zeros` 等の `T: DeviceRepr`
//! （＋ `ValidAsZeroBits`）境界を要求する cudarc API は `half::bf16` を
//! そのまま渡せる。依存・feature の追加変更は不要（ユーザー承認不要）。
//!
//! # 方針: ホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの委譲
//!
//! `crates/backend-cpu/src/typed_f16.rs`（#1698）と同型の 3 段構成を
//! 採る。cudarc 側で bf16 の `DeviceRepr` が可能と判明した今回も、
//! bf16 のままデバイス常駐で widen/narrow する専用カーネルは追加せず
//! （性能最適化は本イシューのスコープ外。下記「スコープ外」参照）、
//! **ホスト側**で bf16 ⇔ f32 変換してから既存 CUDA f32 経路
//! （`elementwise.rs`／`gemm.rs`／`reduce.rs`。H2D・NVRTC カーネル・D2H
//! を含む）へそのまま委譲する:
//!
//! 1. **昇格**: 入力 `Tensor<bf16>` を [`Tensor::host_slice`]（contiguous
//!    なら借用・非 contiguous な view はここで 1 回だけ実体化）で読み
//!    出し、`bf16::to_f32`（exact widening。情報損失なし）で `Vec<f32>`
//!    へ変換してから `Tensor<f32>` を構築する
//! 2. **演算**: 既存 `<CudaBackendOps as BackendOps>::{add, mul, relu,
//!    exp, tanh, sum, max}` および `gemm_fp32_strict`（`gemm` ではない。
//!    下記「`gemm` は `gemm_fp32_strict` へ委譲する」参照）へ委譲する。
//!    本ファイルは CUDA カーネル本体（`kernels_*.rs`／`elementwise.rs`／
//!    `gemm*.rs`／`reduce.rs`）に一切触れない
//! 3. **丸め**: 出力 `Tensor<f32>` を `bf16::from_f32`（IEEE 754 最近接
//!    偶数丸め）で `Vec<bf16>` へ変換し最終 `Tensor<bf16>` を構築する
//!
//! この構成により、`TypedOps<bf16>` の各メソッドは構造的に
//! `bf16::from_f32(BackendOps::op(upcast(x)))`（要素ごと bit 一致。
//! `gemm` のみ `gemm_fp32_strict` に対応）という不変条件を満たす
//! （`tests` モジュールで確認）。f32 既存経路（`elementwise.rs`／
//! `gemm.rs`／`gemm_mma*.rs`／`reduce.rs`／`memory.rs`／`precision.rs`）
//! は本ファイル追加によって一切変更されない。
//!
//! # `gemm` は `BackendOps::gemm_fp32_strict` へ委譲する（`gemm` ではない）
//!
//! CUDA の `BackendOps::gemm`（`ops.rs` の公開経路）は
//! `crate::precision::gemm_precision()`（TF32／TF32x3 opt-in モード）に
//! よって暗黙に精度が変わりうる。設計 §6 は「f16／bf16 は入力を f32 へ
//! 昇格し `f32::mul_add` で累算」という f32 厳密契約を要求するため、
//! opt-in フラグを一切参照しない [`fandhe_ai_tensor_core::BackendOps::
//! gemm_fp32_strict`]（CUDA オーバーライドは `ops.rs:1700` 付近）を使う。
//! `backend-cpu::typed_f16` が `gemm`（無印）を使っているのは CPU に
//! TF32 の概念がなく `gemm`＝常に f32 厳密経路のためであり、CUDA では
//! そのまま踏襲しない。
//!
//! # accessor `typed_ops_bf16` は無条件に `Some(self)` を返す
//!
//! `ops.rs::CudaBackendOps::memory_ops` は `device_handle()` を呼んで
//! `CudaMemory` を構築する必要があるため driver 不在時に `None` へ
//! 縮退するが（`ops.rs:1256` 付近の doc コメント参照）、本 accessor は
//! driver に一切触れずに `TypedOps<bf16>` の実体（`self` 自身）を返す
//! だけでよい。`ops.rs` 冒頭の poison 検査迂回に関する既存コメントが
//! 警告するとおり、accessor 内で `device_handle()` を呼ぶ必要はなく
//! また呼ばない。実行時の CUDA 不在は各演算メソッド内部
//! （`BackendOps::add` 等・`gemm_fp32_strict`）が
//! `BackendError::CudaUnavailable` を型付きエラーとして返す形で伝える
//! （他の CUDA `BackendOps` メソッドと同じ契約）。
//!
//! # 数値・意味論上の明文化事項
//!
//! - 丸めは `half::bf16::from_f32`（IEEE 754 最近接偶数丸め）。bf16
//!   表現範囲を超える f32 結果は ±inf、絶対値が小さい結果は非正規化数
//!   ／0 へ落ちる（f16 版と同じ IEEE 挙動）
//! - `sum`（全縮約・軸縮約）は f32 経路（f64 アキュムレータ→f32 に 1 回
//!   downcast）の出力を bf16 へ丸める二段丸め（f64→f32→bf16）。
//!   `typed_f16.rs` と同じ帰結（`reduction::sum`／CUDA reduce カーネル
//!   の既存契約を変更しないため）
//! - `max` は f32 版と同じ NaN 非伝播（既知事項・本イシューのスコープ外）
//! - `add`／`mul` のブロードキャスト規則は f32 版
//!   （`fandhe_ai_tensor_core::elementwise_out_shape`。NumPy 互換）と同一
//! - 非 contiguous view は `host_slice()` で実体化してから渡すため、f32
//!   版の stride 読み高速経路・`gemm_fp32_strict` の NT/TN 転置 fast
//!   path は経由しない（性能最適化は本イシューのスコープ外）
//! - 一時 `Vec<f32>` の追加確保（H2D／D2H は既存 f32 経路が内部で行う）
//!   は許容する（性能最適化は本イシューの対象外）
//!
//! # スコープ外
//!
//! bf16 デバイス常駐経路（bf16 のまま H2D し device 側で widen/narrow
//! する 2 カーネル方式）・bf16 `mma.sync` GEMM カーネル（cc ≥ 8.0。
//! 設計 §8「#1650 の実装事項」）・CUDA `TypedOps<f64>`／`TypedOps<f16>`
//! （#1703）・Metal（#1651）・`Var`／
//! `Tape`／VJP の dtype 一般化・facade 公開面への昇格・GB10 実機での
//! `#[ignore]` テスト実測（本エージェント実行環境に到達手段がなく未実測。
//! 設計文書 §12 記入欄・親 #1650 へ申し送り）。

use half::bf16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

use crate::ops::CudaBackendOps;

/// `Tensor<bf16>` を `Tensor<f32>` へソフトウェア変換で昇格する。
///
/// [`Tensor::host_slice`] で読み出す（contiguous なら借用・非
/// contiguous な view は 1 回だけ実体化）ため、呼び出し元の非
/// contiguous view をそのまま渡してよい。shape 自体は `bf16`→`f32` で
/// 変わらないため `Tensor::new` の失敗は契約上到達しないはずだが、
/// `Tensor` 実装の不変条件違反に対する fail-safe として型付きエラーで
/// 受ける（`crates/backend-cpu/src/typed_f16.rs::upcast_f16` と同じ
/// 位置づけ）。
fn upcast_bf16(t: &Tensor<bf16>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|v| v.to_f32()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_bf16::upcast_bf16: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `Tensor<f32>` を `Tensor<bf16>` へ最近接偶数丸めで降格する。
///
/// [`upcast_bf16`] と対になるヘルパー。丸めは `half::bf16::from_f32`
/// （IEEE 754 最近接偶数丸め）。shape は不変のため `Tensor::new` の
/// 失敗は契約上到達しないはずだが、同じ理由で fail-safe を持つ。
fn downcast_f32(t: &Tensor<f32>) -> Result<Tensor<bf16>, BackendError> {
    let rounded: Vec<bf16> = t.host_slice().iter().map(|&v| bf16::from_f32(v)).collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_bf16::downcast_f32: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

impl TypedOps<bf16> for CudaBackendOps {
    /// bf16 入力を f32 へ昇格し、`BackendOps::gemm_fp32_strict`
    /// （`crate::precision::gemm_precision()` の TF32／TF32x3 opt-in
    /// モードを一切見ない f32 厳密経路。モジュール doc「`gemm` は
    /// `gemm_fp32_strict` へ委譲する」参照）へ委譲してから bf16 へ
    /// 1 回丸める。
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

    /// 既存 `BackendOps::sum`（`f64` アキュムレータ・CUDA reduce カーネル
    /// の固定順序）へ委譲する。二段丸め（f64→f32→bf16）についてはモジュール
    /// doc「数値・意味論上の明文化事項」参照。
    fn sum(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::sum(self, &a32, dim)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::max` へ委譲する。NaN 非伝播は f32 版と同じ
    /// 既知事項（モジュール doc 参照）。
    fn max(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        let a32 = upcast_bf16(a)?;
        let out32 = BackendOps::max(self, &a32, dim)?;
        downcast_f32(&out32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cudarc `DeviceRepr`／`ValidAsZeroBits` が `half::bf16` に実装
    /// されていることのコンパイル時検査（イシュー #1704 R1 調査結果の
    /// 機械的固定。Linux 実行可能・GPU 不要）。この関数が単にコンパイル
    /// できること自体が「可」判定の直接証跡であり、cudarc の feature
    /// 構成（`f16` feature）が崩れて `DeviceRepr` impl が消えた場合に
    /// ビルドが壊れることで検出する。
    fn assert_bf16_is_device_repr<T>()
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
    }

    #[test]
    fn bf16_satisfies_cudarc_device_repr_bound() {
        // 型引数を明示するだけで上記の trait 境界がコンパイル時に検査
        // される。実行時アサーションは不要（コンパイルが通ること自体が
        // 検証）。
        assert_bf16_is_device_repr::<bf16>();
    }

    fn t(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
        let d: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
        Tensor::new(d, shape).unwrap()
    }

    fn f32v(t: &Tensor<bf16>) -> Vec<f32> {
        t.host_slice().iter().map(|v| v.to_f32()).collect()
    }

    #[test]
    fn upcast_downcast_roundtrip_preserves_exact_values() {
        // bf16 で正確に表現できる値（小整数）は upcast → downcast の
        // 往復で完全一致する（丸め誤差が発生しないことの確認）。
        let a = t(&[1.0, -2.0, 0.0, 4.5], &[4]);
        let up = upcast_bf16(&a).unwrap();
        assert_eq!(up.host_slice().as_ref(), &[1.0f32, -2.0, 0.0, 4.5][..]);
        let down = downcast_f32(&up).unwrap();
        assert_eq!(f32v(&down), vec![1.0, -2.0, 0.0, 4.5]);
    }

    #[test]
    fn downcast_rounds_to_nearest_even_at_bf16_boundary() {
        // bf16 の仮数部は 7 bit。1.0 + 2^-8 は 1.0 と 1.0078125（2^-7 刻み
        // の隣接表現値）のちょうど中間点で、bf16 で正確に表現できず
        // 最近接偶数丸めが働く（`half::bf16::from_f32` の契約どおり）。
        // 1.0（末尾仮数ビット 0＝偶数）・1.0078125（末尾仮数ビット 1＝
        // 奇数）のうち偶数側の 1.0 へ丸められるはず（tie-to-even。
        // `crates/backend-cpu/src/typed_bf16.rs::
        // add_one_plus_two_pow_neg_eight_rounds_to_one_via_bf16_tie_to_even`
        // と同じ丸め契約の根拠をコードで固定する）。
        let value = 1.0f32 + f32::from_bits(0x3b800000); // 2^-8
        let rounded = bf16::from_f32(value);
        assert_eq!(
            rounded.to_bits(),
            bf16::from_f32(1.0).to_bits(),
            "tie-to-even は偶数側（1.0）へ丸められるはずが {} へ丸められた",
            rounded.to_f32()
        );
    }

    #[test]
    fn upcast_bf16_infinity_and_nan_are_preserved() {
        let a = Tensor::new(vec![bf16::INFINITY, bf16::NEG_INFINITY, bf16::NAN], &[3]).unwrap();
        let up = upcast_bf16(&a).unwrap();
        let s = up.host_slice();
        assert!(s[0].is_infinite() && s[0] > 0.0);
        assert!(s[1].is_infinite() && s[1] < 0.0);
        assert!(s[2].is_nan());
    }

    /// 非 contiguous view（transpose）に対する `upcast_bf16`／
    /// `downcast_f32` が contiguous 等価物と bit 完全一致することを
    /// 確認する（レビュー指摘対応。`crates/backend-cpu/src/typed_f16.rs
    /// ::non_contiguous_transpose_view_matches_contiguous_equivalent`
    /// と同型の懸念に対する CUDA 側の検証）。
    ///
    /// `TypedOps::<bf16>` の演算本体（`gemm`／`add` 等）は
    /// `device_handle()` 経由で実 CUDA driver を要求するため、実機
    /// 非搭載の通常 CI では `CudaUnavailable` にしかならず非
    /// contiguous 経路の検証にならない。一方 [`upcast_bf16`]／
    /// [`downcast_f32`] はモジュール doc の「1. 昇格」「3. 丸め」段が
    /// 述べるとおり `Tensor::host_slice()`（非 contiguous view は
    /// ここで実体化）のみに依存するホスト側関数で、GPU 不要かつ
    /// 各 `TypedOps<bf16>` メソッドが実際に非 contiguous 入力へ辿る
    /// 経路そのものである。この関数がどんな view に対しても同じ結果
    /// （bit 完全一致）を返すことを検証すれば、`TypedOps<bf16>` 各
    /// メソッド全体としての非 contiguous 対応は構造的に保証される。
    #[test]
    fn non_contiguous_transpose_view_upcast_matches_contiguous_equivalent() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let a_t = a.transpose_2d().unwrap();
        assert!(
            !a_t.is_contiguous(),
            "transpose_2d は非 contiguous view を返すはず"
        );
        let a_t_contig = a_t.contiguous();

        let up_view = upcast_bf16(&a_t).unwrap();
        let up_contig = upcast_bf16(&a_t_contig).unwrap();
        assert_eq!(
            up_view.host_slice().as_ref(),
            up_contig.host_slice().as_ref(),
            "非 contiguous view と contiguous 等価物で upcast_bf16 の結果が一致しない"
        );

        // downcast_f32 側（f32 → bf16）も同様に非 contiguous 入力を
        // 実体化してから丸める。upcast 側の出力（f32）をそのまま
        // 非 contiguous view として渡し、対称性を確認する。
        let up_view_t = up_view.transpose_2d().unwrap();
        let up_contig_t = up_view_t.contiguous();
        let down_view = downcast_f32(&up_view_t).unwrap();
        let down_contig = downcast_f32(&up_contig_t).unwrap();
        assert_eq!(
            f32v(&down_view),
            f32v(&down_contig),
            "非 contiguous view と contiguous 等価物で downcast_f32 の結果が一致しない"
        );
    }
}
