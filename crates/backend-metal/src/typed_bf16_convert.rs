//! `Tensor<half::bf16>` ⇔ `Tensor<f32>` の変換ヘルパー（イシュー #1706・
//! 親 #1651・`docs/backend-dtype-dispatch-design.md` §5「bf16 × Metal」・
//! §13）。
//!
//! # cfg を付けない理由
//!
//! [`crate::typed_bf16`]（`impl TypedOps<half::bf16> for MetalBackendOps`）
//! は `objc2` 系 FFI に触れるため `cfg(target_os = "macos")` 限定だが、
//! 本モジュールは `half`／`fandhe_ai_tensor_core` のみに依存し `objc2`
//! 系には一切触れない。[`crate::pad`]／[`crate::layout`] と同じ判断で
//! cfg を付けず、Linux（CI・本実装環境）でも単体テストが回るようにする
//! （丸め契約〈tie-to-even・非 contiguous view 対応〉を実機なしで
//! 固定できる）。
//!
//! macOS 側からしか参照されない場合に Linux の
//! `clippy --all-targets --all-features -- -D warnings` が dead_code を
//! 検出しうるため公開関数は **`pub`** にする（`crate::pad`／`crate::layout`
//! と同じ扱い）。
//!
//! # 方針: bf16⇔f32 のホスト側変換のみ（既存 f32 カーネルへの委譲は
//! [`crate::typed_bf16`] 側が行う）
//!
//! - [`upcast_bf16`]: bf16→f32 は完全表現可能・損失なし
//!   （`bf16::to_f32()`）
//! - [`downcast_f32`]: `bf16::from_f32`（IEEE 754 最近接偶数丸め）

use half::bf16;

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::device::BackendError;

/// `Tensor<bf16>` を `Tensor<f32>` へソフトウェア変換で昇格する。
///
/// [`Tensor::host_slice`] で読み出す（contiguous なら借用・非
/// contiguous な view は 1 回だけ実体化）ため、呼び出し元の非
/// contiguous view をそのまま渡してよい。shape 自体は `bf16`→`f32` で
/// 変わらないため `Tensor::new` の失敗は契約上到達しないはずだが、
/// `Tensor` 実装の不変条件違反に対する fail-safe として型付きエラーで
/// 受ける（`crates/backend-cuda/src/typed_bf16.rs::upcast_bf16` と同じ
/// 位置づけ）。
pub fn upcast_bf16(t: &Tensor<bf16>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|v| v.to_f32()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_bf16_convert::upcast_bf16: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `Tensor<f32>` を `Tensor<bf16>` へ最近接偶数丸めで降格する。
///
/// [`upcast_bf16`] と対になるヘルパー。丸めは `half::bf16::from_f32`
/// （IEEE 754 最近接偶数丸め）。shape は不変のため `Tensor::new` の
/// 失敗は契約上到達しないはずだが、同じ理由で fail-safe を持つ。
pub fn downcast_f32(t: &Tensor<f32>) -> Result<Tensor<bf16>, BackendError> {
    let rounded: Vec<bf16> = t.host_slice().iter().map(|&v| bf16::from_f32(v)).collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_bf16_convert::downcast_f32: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // 偶数側（1.0）へ丸められるはず（tie-to-even。
        // `crates/backend-cpu/src/typed_bf16.rs`・
        // `crates/backend-cuda/src/typed_bf16.rs` と同じ丸め契約の根拠を
        // コードで固定する）。
        let value = 1.0f32 + f32::from_bits(0x3b80_0000); // 2^-8
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
    /// 確認する（`crates/backend-cuda/src/typed_bf16.rs::
    /// non_contiguous_transpose_view_upcast_matches_contiguous_equivalent`
    /// と同型の懸念に対する Metal 側の検証）。`impl TypedOps<bf16> for
    /// MetalBackendOps` の各メソッドはここで検証した [`upcast_bf16`]／
    /// [`downcast_f32`] のみに依存するホスト側変換を経由するため、この
    /// 関数がどんな view に対しても同じ結果（bit 完全一致）を返せば、
    /// `TypedOps<bf16>` 各メソッド全体としての非 contiguous 対応は
    /// 構造的に保証される。
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

    #[test]
    fn shape_is_preserved_across_roundtrip() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let up = upcast_bf16(&a).unwrap();
        assert_eq!(up.shape(), &[2, 3]);
        let down = downcast_f32(&up).unwrap();
        assert_eq!(down.shape(), &[2, 3]);
    }
}
