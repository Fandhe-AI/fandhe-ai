//! `fandhe_ai_tensor_core::layout` の再エクスポート（イシュー #1046 で
//! tensor-core へ移設。詳細な設計コメント・テストは移設先を参照）。
//!
//! `autodiff::eval::matmul`（VJP のホスト側転置コピー除去）が
//! `backend-metal`（#1040）と同じ「2 次元 view の転置分類」ロジックを
//! 必要としたため、実体を `tensor-core::layout` へ移し `backend-metal`
//! 側は再エクスポートのみに縮約した。`crate::gemm`・`crate::ops`・
//! 既存テストは `crate::layout::…` のパスのまま参照できる
//! （呼び出し元の変更不要）。

pub use fandhe_ai_tensor_core::layout::{
    MatrixLayout, TransposePattern, classify_2d, collapse_leading_dims, required_span,
};
