//! `CpuBackendOps::sort`／`topk`（イシュー #1733）の受け入れ条件テスト。
//!
//! `fandhe_ai_autodiff::eval::sort`／`topk`（ホスト参照実装。`autodiff`
//! クレート非公開のため `backend-cpu` から直接は呼べない）と CPU
//! ネイティブ実装（本クレート `sort_topk` モジュール）が同一アルゴリズム
//! （`BackendOps::sort` doc の順序契約 1〜4）であることを、
//! `Var::sort`／`argsort`／`topk` の 2 系統実行経路を間接的に突き合わせて
//! 確認する: `CpuBackendOps` をそのまま渡した `Tape` は本クレートの
//! ネイティブ実装を経由し、`sort`／`topk` だけを強制的に `Unsupported`
//! にする [`ForceEvalFallback`] を渡した `Tape` は `autodiff` のホスト
//! フォールバック（`eval::sort`／`topk`）を経由する
//! （`gather_scatter_parity.rs` と同型の「1 メソッドだけ意図的に
//! override しない」ラッパー方針）。両経路の `values` の出力が
//! `f32::to_bits()` で bit 完全一致し、`index` が完全一致することを
//! 固定し、将来どちらかの実装が並列化された際の決定性回帰を検知する。
//!
//! 本ファイルは §2.2「順序契約」（同値の元添字昇順・NaN・±0・
//! run-to-run 決定性）を CPU ネイティブ経路（`Var` 経由）で個別に
//! 固定する契約テストでもある——後続の CUDA／Metal 実装（イシュー
//! #1741）が同じ観測結果を再現する際の基準として同ファイルを
//! 横展開できる形にする。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `CpuBackendOps` の必須メソッド（デフォルト実装を持たない 9 個）へ
/// 委譲しつつ、`sort`／`topk` だけは意図的に override せずデフォルト
/// （`Unsupported`）のまま残すラッパー（`gather_scatter_parity.rs::
/// ForceEvalFallback` と同型）。`Var::sort`／`argsort`／`topk` を
/// `autodiff::eval::sort`／`topk`（ホストフォールバック経路）へ
/// 強制的に迂回させるための唯一の差分点。
struct ForceEvalFallback {
    inner: CpuBackendOps,
}

impl BackendOps for ForceEvalFallback {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    // `sort`／`topk` はデフォルト実装（`Unsupported`）のまま override
    // しない。これが `eval::` フォールバック経路の唯一の実現手段。
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn dense_vec_i32(tensor: &Tensor<i32>) -> Vec<i32> {
    let c = tensor.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn assert_bit_exact(label: &str, native: &Tensor<f32>, fallback: &Tensor<f32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: shape が一致しない"
    );
    let a = dense_vec(native);
    let b = dense_vec(fallback);
    assert_eq!(a.len(), b.len(), "{label}: 要素数が一致しない");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: 要素 {i} が bit 一致しない（native={x}, fallback={y}）"
        );
    }
}

fn assert_index_exact(label: &str, native: &Tensor<i32>, fallback: &Tensor<i32>) {
    assert_eq!(
        native.shape(),
        fallback.shape(),
        "{label}: index の shape が一致しない"
    );
    assert_eq!(
        dense_vec_i32(native),
        dense_vec_i32(fallback),
        "{label}: index が一致しない"
    );
}

/// `sort`（同値・NaN・±0 を含む）の CPU ネイティブ実装と `eval::sort`
/// フォールバックが `values`（bit 完全一致）・`index`（完全一致）とも
/// 一致することを確認する。
#[test]
fn sort_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![2.0, 1.0, f32::NAN, -0.0, 0.0, 1.0, f32::NAN], &[1, 7]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, native_index) = native_x.sort(1, false).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, fallback_index) = fallback_x.sort(1, false).unwrap();

    assert_bit_exact("sort", &native_out.to_tensor(), &fallback_out.to_tensor());
    assert_index_exact("sort", &native_index, &fallback_index);
}

/// `sort`（降順）の CPU ネイティブ実装と `eval::sort` フォールバックが
/// bit 完全一致することを確認する（同値タイブレークが両方向で一致
/// することの回帰確認を兼ねる）。
#[test]
fn sort_descending_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![2.0, 1.0, 1.0, 2.0, 3.0], &[1, 5]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, native_index) = native_x.sort(1, true).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, fallback_index) = fallback_x.sort(1, true).unwrap();

    assert_bit_exact(
        "sort(descending)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_index_exact("sort(descending)", &native_index, &fallback_index);
}

/// `sort` を transpose 後の非 contiguous view に適用しても両経路が
/// 一致することを確認する（strided 入力の扱いの回帰確認）。
#[test]
fn sort_non_contiguous_view_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![3.0, 1.0, 4.0, 1.5], &[2, 2]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let native_xt = native_x.transpose(0, 1).unwrap();
    let (native_out, native_index) = native_xt.sort(1, false).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let fallback_xt = fallback_x.transpose(0, 1).unwrap();
    let (fallback_out, fallback_index) = fallback_xt.sort(1, false).unwrap();

    assert_bit_exact(
        "sort(transposed view)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_index_exact("sort(transposed view)", &native_index, &fallback_index);
}

/// `topk`（`largest=true`・`k < n`）の CPU ネイティブ実装と `eval::topk`
/// フォールバックが一致することを確認する。
#[test]
fn topk_largest_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![3.0, 1.0, 4.0, 1.5, 2.0], &[1, 5]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, native_index) = native_x.topk(3, 1, true).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, fallback_index) = fallback_x.topk(3, 1, true).unwrap();

    assert_bit_exact(
        "topk(largest)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_index_exact("topk(largest)", &native_index, &fallback_index);
}

/// `topk`（`largest=false`）の CPU ネイティブ実装と `eval::topk`
/// フォールバックが一致することを確認する。
#[test]
fn topk_smallest_native_matches_eval_fallback_bit_exact() {
    let x = t(vec![3.0, 1.0, 4.0, 1.5, 2.0], &[1, 5]);

    let native_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let native_x = native_tape.var(&x);
    let (native_out, native_index) = native_x.topk(2, 1, false).unwrap();

    let fallback_tape = Tape::new_with_ops(Box::new(ForceEvalFallback {
        inner: CpuBackendOps::new(),
    }));
    let fallback_x = fallback_tape.var(&x);
    let (fallback_out, fallback_index) = fallback_x.topk(2, 1, false).unwrap();

    assert_bit_exact(
        "topk(smallest)",
        &native_out.to_tensor(),
        &fallback_out.to_tensor(),
    );
    assert_index_exact("topk(smallest)", &native_index, &fallback_index);
}

/// `argsort` が `Var::sort` と同一の `index` を返すことを CPU ネイティブ
/// 経路で確認する（`Var::argsort` doc「§2.1」参照）。
#[test]
fn argsort_matches_sort_index_native() {
    let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let xv = tape.var(&x);
    let (_out, sort_index) = xv.sort(1, false).unwrap();
    let argsort_index = xv.argsort(1, false).unwrap();
    assert_eq!(dense_vec_i32(&sort_index), dense_vec_i32(&argsort_index));
}

/// run-to-run 決定性（§2.2 契約 4）: 同一入力で 5 回実行しても
/// `values`（bit 完全一致）・`index`（完全一致）とも変化しないことを
/// CPU ネイティブ実装で確認する。
#[test]
fn sort_and_topk_are_run_to_run_bit_identical() {
    let ops = CpuBackendOps::new();
    let x = t(vec![3.0, 1.0, 4.0, 1.5, 2.0, 2.0], &[1, 6]);

    let (first_sort_vals, first_sort_idx) = ops.sort(&x, 1, false).unwrap();
    let (first_topk_vals, first_topk_idx) = ops.topk(&x, 1, 3, true).unwrap();
    for _ in 0..5 {
        let (sort_vals, sort_idx) = ops.sort(&x, 1, false).unwrap();
        assert_bit_exact("sort run-to-run", &first_sort_vals, &sort_vals);
        assert_index_exact("sort run-to-run", &first_sort_idx, &sort_idx);

        let (topk_vals, topk_idx) = ops.topk(&x, 1, 3, true).unwrap();
        assert_bit_exact("topk run-to-run", &first_topk_vals, &topk_vals);
        assert_index_exact("topk run-to-run", &first_topk_idx, &topk_idx);
    }
}

/// `CpuBackendOps::sort` が `ops_shape::sort_out_shape` による shape
/// 再検査を実装側でも行い、不一致を `BackendError::ShapeMismatch` と
/// して fail-closed に拒否することを確認する（`.claude/rules/
/// security.md` A08。トレイトメソッドを直接呼び `Var` 側の検査を
/// 経由しない経路を対象とする）。
#[test]
fn backend_ops_sort_rejects_axis_out_of_range() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
    let err = ops.sort(&input, 5, false).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `CpuBackendOps::topk` が `k > shape[dim]` を fail-closed に拒否
/// することを確認する。
#[test]
fn backend_ops_topk_rejects_k_exceeding_dim_size() {
    let ops = CpuBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let err = ops.topk(&input, 1, 5, true).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
