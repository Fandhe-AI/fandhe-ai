//! ONNX 8 オペ（`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`。
//! TASK-7.2c・#79）に加え、MVP 算術オペ（`Add`／`Mul`／`Div`／`Mod`／`Sqrt`／`Constant`。
//! TASK-7.3a・#82）・MVP 形状操作オペ（`Cast`／`Reshape`／`Squeeze`／`Transpose`。
//! TASK-7.3b・#83）・Attention 系オペ（`MatMul`／`Softmax`／`Erf`。TASK-7.3c・#84）・
//! `LayerNormalization`（TASK-7.3d・#85）・`Conv`（イシュー #2076）・
//! `GlobalAveragePool`／`BatchNormalization`／`Flatten`（イシュー #2200。
//! CNN 系モデル対応）を提供する。算術・活性化・正規化系
//! （`arith`／`activation`／`gemm`／`matmul`／`softmax`／`layer_norm`／
//! `batch_norm`／`global_average_pool`）は `tensor-core::Tensor<f32>` 専用の
//! 純粋関数のまま、形状系（`shape_ops`／`shape_transform`／`gather`／
//! `concat`／`slice`）は要素コピーのみで算術を伴わないため `T: Element` で
//! ジェネリック化し（`shape_transform::flatten` も同様）、`Cast`（`cast`）は
//! dtype ごとに型安全な変換関数を個別に提供する（イシュー #274）。
//!
//! イシュー #2076（親 #2034）で `Conv`（2 次元畳み込み）を追加し、イシュー #2199
//! （親 #2185）で `MaxPool`／`AveragePool`（`pool.rs`）を追加・`Conv` に 1D
//! （`[N,C,L]`）対応を追加した（`conv.rs`）。`MaxPool`／`AveragePool` は import
//! 専用オペで、export allowlist（`export_ops::SUPPORTED_OP_TYPES`。23 op で不変）
//! には含まれない（`docs/onnx-model-zoo-parity.md` §5）。
//!
//! 各関数は「入力テンソル＋属性 → 出力テンソル」の単体演算に限定し、ONNX proto デコード
//! （TASK-7.2a）やグラフ実行順序の解決には関与しない。属性は proto 由来の型に依存しない
//! プレーンな Rust 構造体・スライスで受け取るため、decode 層（`AttributeProto` 等）の
//! 実装順序に依存せず本モジュール単体でテスト・使用できる。インタープリタのディスパッチ
//! （op 名 → 本モジュール関数の解決）は [`crate::onnx::interp`]（TASK-7.2b・#78、
//! TASK-7.3 系 14 オペの結線は #274・`Conv` の結線は #2076・`GlobalAveragePool`／
//! `BatchNormalization`／`Flatten` の結線は #2200・`MaxPool`／`AveragePool` の
//! 結線は #2199 で実装）が担う。これらに加え、イシュー #2186 で追加した 8 op
//! （`Clip`／`Tanh`／`Gelu`／`Where`／`Expand`／`ReduceMean`／`Pad`／`Resize`）
//! は本モジュールへは追加せず `fandhe_ai_autodiff::Var` の同名演算へ委譲する形で
//! `interp` のディスパッチ表から到達可能である（`crate::onnx::interp_ext` 冒頭
//! コメント参照。本モジュールの対応 op 数を絶対数で記述せず、ディスパッチ表
//! 〈`interp::run_impl`〉を正とする。import 対応全 36 オペがグラフ実行から
//! 到達可能）。

mod activation;
mod arith;
mod batch_norm;
mod cast;
mod concat;
mod constant;
mod conv;
mod error;
mod gather;
mod gemm;
mod global_average_pool;
mod layer_norm;
mod matmul;
mod pool;
mod shape_ops;
mod shape_transform;
mod slice;
mod softmax;

pub use activation::{erf, relu, sigmoid};
pub use arith::{add, add_i64, div, div_i64, mod_i64, modulo, mul, mul_i64, sqrt};
pub use batch_norm::{BatchNormAttrs, batch_normalization};
pub use cast::{
    cast_bool_to_float, cast_f16_to_float, cast_to_bool, cast_to_f16, cast_to_float, cast_to_int64,
    check_supported_cast_target,
};
pub use concat::concat;
pub use constant::{ConstantValue, constant};
pub use conv::{ConvAttrs, conv};
pub use error::OpError;
pub use gather::gather;
pub use gemm::{GemmAttrs, gemm};
pub use global_average_pool::global_average_pool;
pub use layer_norm::{LayerNormAttrs, layer_normalization};
pub use matmul::matmul;
pub use pool::{PoolAttrs, average_pool, max_pool};
pub use shape_ops::{shape, unsqueeze};
pub use shape_transform::{flatten, reshape, squeeze, transpose};
pub use slice::{SliceParams, slice};
pub use softmax::softmax;

/// 非信頼な ONNX 属性（`pads`／`kernel_shape` 等の整数属性で、対応する
/// 実データテンソルによる自然な上限を持たないもの）に由来する出力バッファの
/// 総要素数の実用上限（イシュー #2199 codex-review 指摘。PR #2314 レビュー
/// 指摘 P0: `MaxPool`／`AveragePool`〈`pool.rs`〉は導入当初 `h_out * w_out`
/// のみを上限検査しており、`n * c` を掛けた実際の確保サイズは無検査だった
/// ため、`[1, 64, 1, 1]` のような小さい入力でも `n * c` 倍〈約 64 倍〉の
/// メモリを要求できた。本定数は `pool.rs::MAX_POOL_SPATIAL_OUT_ELEMENTS`
/// として導入された値をそのまま流用し（新規の閾値を発明しない。閾値変更は
/// ユーザー承認事項。`.claude/rules/deps-policy.md` 相当の運用方針）、
/// 確保対象の**全軸の積**（`n * c * h_out * w_out` 等）を検査する対象へ
/// 一般化する。`Conv`（`conv.rs`）の `pads` も `Conv2dParams::new` が
/// 「padding は上限なし（pooling と異なり `padding <= kernel/2` を要求
/// しない）」ため同種の増幅が可能であり、同じ定数で `n * cout * hout *
/// wout` を検査する（`.claude/rules/security.md` A03）。
pub(crate) const MAX_UNTRUSTED_OUTPUT_ELEMENTS: usize = 1 << 26;

/// `dims` の積（出力バッファの総要素数）を `checked_mul` の連鎖で求め、
/// [`MAX_UNTRUSTED_OUTPUT_ELEMENTS`] 以下かどうかを判定する。オーバー
/// フロー（各軸の積が `usize` の範囲を超える）も上限超過として扱う
/// （fail-closed。`pool.rs::ensure_pool_out_bound`／`conv.rs` から呼ばれる）。
pub(crate) fn output_elements_within_bound(dims: &[usize]) -> bool {
    match dims.iter().copied().try_fold(1usize, usize::checked_mul) {
        Some(total) => total <= MAX_UNTRUSTED_OUTPUT_ELEMENTS,
        None => false,
    }
}

/// ONNX の負軸表記（`axis < 0` の場合 `axis + rank`）を正規化し、`[0, rank)` の範囲を
/// 検査する。範囲外の場合は `None`（呼び出し元が `op` 名を添えて `OpError::AxisOutOfRange`
/// を構築する）。全オペ（`Gather`／`Unsqueeze`／`Concat`／`Slice`）が共有する規則
/// （ONNX Operators スキーマの axis 属性の共通仕様）。
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank_i = rank as i64;
    let normalized = if axis < 0 { axis + rank_i } else { axis };
    if normalized < 0 || normalized >= rank_i {
        None
    } else {
        Some(normalized as usize)
    }
}

#[cfg(test)]
mod output_elements_within_bound_tests {
    use super::{MAX_UNTRUSTED_OUTPUT_ELEMENTS, output_elements_within_bound};

    #[test]
    fn accepts_at_cap_and_rejects_over_cap() {
        assert!(output_elements_within_bound(&[
            1,
            MAX_UNTRUSTED_OUTPUT_ELEMENTS
        ]));
        assert!(!output_elements_within_bound(&[
            1,
            MAX_UNTRUSTED_OUTPUT_ELEMENTS + 1
        ]));
    }

    #[test]
    fn checks_the_product_of_all_axes_not_just_a_subset() {
        // `n * c` を含めた全軸の積を検査する（単一軸のみ・部分軸のみの
        // 検査では見逃す組み合わせ。PR #2314 レビュー指摘 P0 の再現条件）。
        let n = 1;
        let c = 64;
        let h_out = 1;
        let w_out = MAX_UNTRUSTED_OUTPUT_ELEMENTS / 32; // h_out*w_out 単体は上限未満
        assert!(output_elements_within_bound(&[n, 1, h_out, w_out]));
        assert!(!output_elements_within_bound(&[n, c, h_out, w_out]));
    }

    #[test]
    fn overflow_is_rejected() {
        assert!(!output_elements_within_bound(&[usize::MAX, 2]));
    }
}

#[cfg(test)]
mod normalize_axis_tests {
    use super::normalize_axis;

    #[test]
    fn positive_within_range() {
        assert_eq!(normalize_axis(1, 3), Some(1));
    }

    #[test]
    fn negative_wraps_from_end() {
        assert_eq!(normalize_axis(-1, 3), Some(2));
        assert_eq!(normalize_axis(-3, 3), Some(0));
    }

    #[test]
    fn out_of_range_returns_none() {
        assert_eq!(normalize_axis(3, 3), None);
        assert_eq!(normalize_axis(-4, 3), None);
    }
}
