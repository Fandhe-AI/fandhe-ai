//! dtype 別カーネル演算本体（イシュー #1687・親 #1649）。
//!
//! [`crate::BackendOps`] の `typed_ops_f64`／`typed_ops_f16`／
//! `typed_ops_bf16` capability accessor 経由でのみ取得する dtype 別
//! 演算集合。演算×dtype の組合せごとに `BackendOps` へメソッドを追加する
//! 案（`docs/backend-dtype-dispatch-design.md` §4 案 B）ではなく、
//! 型パラメータ trait 1 本（案 D）に集約することで、`BackendOps` 自体の
//! メソッド数を増やさずに dtype を追加できるようにする。
//!
//! 対象は最小集合 8 演算（既存 `BackendOps` v1 の `gemm`／`add`／`mul`／
//! `relu`／`exp`／`tanh`／`sum`／`max` と同一）に固定する（設計 §4.2・
//! 本イシューの承認事項 6）。`gemm_bias_act`・`mse_loss`・softmax 系・
//! resident 系・カーネル融合対応は対象外（別イシュー）。
//!
//! **本イシューでは dtype ごとの実装（`impl TypedOps<f64/f16/bf16> for
//! CpuBackendOps` 等）を追加しない**。実装は後続 sub（CPU: #1697／#1698／
//! #1699、CUDA: #1650、Metal: #1651）が本 trait 定義に従って行う。

use crate::Tensor;
use crate::device::BackendError;
use crate::element::Scalar;

/// dtype 別の演算本体（イシュー #1687）。
///
/// `T: Scalar` を具象型（`f64`／`half::f16`／`half::bf16` 等）に固定した
/// 状態でのみ `dyn` 化する（`&dyn TypedOps<f64>` のように使う）。
/// `Scalar` 自体が object-safe である必要はない
/// （`docs/backend-dtype-dispatch-design.md` §4 案 D）。
///
/// 各バックエンド実装は既存 `BackendOps`（f32 固定）の対応メソッドと
/// 同じ数値契約（バックエンド構成の丸め方針・境界検査省略禁止。
/// `.claude/rules/coding-rust.md`）に従うこと。
pub trait TypedOps<T: Scalar> {
    /// 行列積（`m×k` × `k×n` → `m×n`）。
    fn gemm(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// 要素ごとの加算（ブロードキャスト規則は実装依存）。
    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// 要素ごとの乗算。
    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// ReLU 活性化。
    fn relu(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// 要素ごとの指数関数。
    fn exp(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// 要素ごとの双曲線正接。
    fn tanh(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    /// 総和縮約（`dim` が `None` の場合は全要素縮約）。
    fn sum(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
    /// 最大値縮約（`dim` が `None` の場合は全要素縮約）。
    fn max(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TypedOps<f64>` が dyn-compatible であることのコンパイル時検査
    /// （`&dyn TypedOps<f64>` を関数引数として受け取れる。
    /// `crate::backend_ops::tests::assert_object_safe` と同型のガード）。
    fn assert_dyn_compatible_f64(_ops: &dyn TypedOps<f64>) {}

    /// `half::f16` 版。同上。
    fn assert_dyn_compatible_f16(_ops: &dyn TypedOps<half::f16>) {}

    /// `half::bf16` 版。同上。
    fn assert_dyn_compatible_bf16(_ops: &dyn TypedOps<half::bf16>) {}

    /// 上記 3 関数を実際に呼び出し、コンパイルが通ることをテスト実行
    /// ログ上でも確認できるようにする（関数自体は未使用なら `dead_code`
    /// lint で警告されるため、ここで参照する）。
    #[test]
    fn typed_ops_dyn_compatible_for_f64_f16_bf16() {
        fn accepts_f64(_f: fn(&dyn TypedOps<f64>)) {}
        fn accepts_f16(_f: fn(&dyn TypedOps<half::f16>)) {}
        fn accepts_bf16(_f: fn(&dyn TypedOps<half::bf16>)) {}
        accepts_f64(assert_dyn_compatible_f64);
        accepts_f16(assert_dyn_compatible_f16);
        accepts_bf16(assert_dyn_compatible_bf16);
    }
}
