//! `MetalBackendOps::gemm_batched`／`gemm_batched_fp32_strict` の受け入れ
//! 基準対応テスト（イシュー #1717・親 #1600・#1715 の Metal 版）。
//!
//! `crates/backend-cpu/tests/gemm_batched_parity.rs`（#1715）と同型の
//! 構成に加え、Metal 固有の契約（split-K 実行時トグル非干渉・
//! run-to-run 決定性・NT/TN 転置 view 入力）を検証する:
//!
//! 1. 各バッチの結果が `matmul_reference_fma`（スカラー `mul_add`
//!    参照実装）と REQ-2 統一複合判定（`assert_parity`）を満たすこと
//!    （Metal は分類不能形状〈`normalize_batched_operand` 後は常に
//!    NN〉のため classic strided カーネル `gemm_tiled_bias_act` を経由
//!    し、per-batch `dispatch_auto`〈`gemm_simdgroup_tiled`〉とは
//!    bit 同一を主張しない。`gemm_strided_nt_tn`〈#1215〉と同じ契約）。
//! 2. rank 2 同士は `gemm` への直接委譲で bit 同一になること。
//! 3. lhs／rhs バッチ次元 1（NumPy 互換ブロードキャスト）を正しく
//!    処理すること。
//! 4. shape 不整合が `BackendError::ShapeMismatch` を返すこと。
//! 5. `batch_len == 0`／`m == 0`／`n == 0`／`k == 0` の退化形状を
//!    正しく処理すること（GPU 起動なし）。
//! 6. run-to-run 決定性（同入力 2 回で bit 完全一致）。
//! 7. `gemm_batched_fp32_strict` が `gemm_batched` と bit 同一になる
//!    こと（Metal は TF32 の概念を持たない）。
//! 8. split-K 実行時トグル（`crate::split_k_runtime`）の on/off で
//!    出力が変わらないこと（本経路は `dispatch_auto` を経由しない
//!    ため split-K に非到達）。
//! 9. VJP と同型の転置 view 入力（`transpose(1, 2)` 後の rank 3 view）
//!    でも正しく正規化されて結果が一致すること。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される。実機実行は本エージェント実行環境に Apple Silicon 実機が
//! ないため未実施のまま Mac セッションへ申し送る。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_batched_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::assert_parity;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_backend_metal::split_k_runtime::{set_split_k_enabled, split_k_enabled};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};
use std::sync::Mutex;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// スカラー `f32::mul_add` 参照実装（REQ-2 の FMA 契約突合対象。
/// `crates/backend-cpu/tests/gemm_batched_parity.rs` と同一実装）。
fn matmul_reference_fma(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = a[i * k + kk].mul_add(b[kk * n + j], acc);
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// `split_k_runtime` はプロセスグローバルなため、本ファイル内のテスト
/// （同一プロセスで並列実行されうる）を直列化・原状復帰する RAII
/// ガード（`tests/gemm_splitk_runtime_toggle.rs::RuntimeFlagGuard` と
/// 同型。本経路は split-K に到達しないためトグルの値そのものを検証
/// する必要はなく、「値を変えても出力が変わらない」ことのみを確認する）。
struct RuntimeFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl RuntimeFlagGuard {
    fn acquire() -> Self {
        static LOCK: Mutex<()> = Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = split_k_enabled();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for RuntimeFlagGuard {
    fn drop(&mut self) {
        set_split_k_enabled(self.original);
    }
}

/// (T1)/(T2) バッチ `B=3`・複数形状（1×1×1 の退化形状・非正方・大きめ）
/// で `gemm_batched` の各バッチが `matmul_reference_fma`・per-batch
/// `gemm` の両方と REQ-2 統一複合判定を満たすことを確認する
/// （Metal は classic strided カーネルを経由するため per-batch
/// `dispatch_auto` と bit 同一は主張しない）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_matches_reference_and_per_batch_gemm() {
    let ops = MetalBackendOps::new();

    for &(b, m, k, n) in &[
        (3usize, 1usize, 1usize, 1usize),
        (3, 7, 5, 9),
        (2, 64, 32, 48),
        (3, 129, 65, 33),
    ] {
        let a = tensor(
            random_matrix(0x3000 + (b * m * k) as u64, b * m * k),
            &[b, m, k],
        );
        let bb = tensor(
            random_matrix(0x4000 + (b * k * n) as u64, b * k * n),
            &[b, k, n],
        );

        let batched = ops.gemm_batched(&a, &bb).unwrap();
        assert_eq!(batched.shape(), &[b, m, n]);

        let a_c = a.contiguous();
        let b_c = bb.contiguous();
        let a_slice = a_c.as_slice().unwrap();
        let b_slice = b_c.as_slice().unwrap();
        let batched_c = batched.contiguous();
        let batched_slice = batched_c.as_slice().unwrap();

        for i in 0..b {
            // (T2) per-batch `gemm`（同じ Metal オペレータの単発呼び
            // 出し）との複合判定。
            let a_i = a.narrow(0, i, 1).unwrap().reshape(&[m, k]).unwrap();
            let b_i = bb.narrow(0, i, 1).unwrap().reshape(&[k, n]).unwrap();
            let expected_gemm = ops.gemm(&a_i, &b_i).unwrap();
            let got = &batched_slice[i * m * n..(i + 1) * m * n];
            assert_parity(
                "gemm_batched vs per-batch gemm",
                got,
                expected_gemm.contiguous().as_slice().unwrap(),
            );

            // (T1) スカラー FMA 参照実装との複合判定。
            let a_i_slice = &a_slice[i * m * k..(i + 1) * m * k];
            let b_i_slice = &b_slice[i * k * n..(i + 1) * k * n];
            let expected_ref = matmul_reference_fma(a_i_slice, b_i_slice, m, k, n);
            assert_parity("gemm_batched vs scalar FMA reference", got, &expected_ref);
        }
    }
}

/// (T3) rank 2 同士は `gemm` への直接委譲で bit 同一になる。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_rank2_delegates_to_gemm_bit_exact() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0x10, 37 * 41), &[37, 41]);
    let b = tensor(random_matrix(0x20, 41 * 29), &[41, 29]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    let direct = ops.gemm(&a, &b).unwrap();
    assert_eq!(batched.shape(), direct.shape());
    assert_eq!(
        batched.contiguous().as_slice().unwrap(),
        direct.contiguous().as_slice().unwrap()
    );
}

/// (T4) lhs バッチ次元 1（NumPy 互換ブロードキャスト。`[1, m, k]`）が
/// 出力バッチ数ぶん複製されて使われることを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_broadcasts_lhs_batch_dim_one() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0x30, 4 * 6), &[1, 4, 6]);
    let b = tensor(random_matrix(0x40, 5 * 6 * 8), &[5, 6, 8]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(batched.shape(), &[5, 4, 8]);

    let a_2d = a.reshape(&[4, 6]).unwrap();
    let batched_c = batched.contiguous();
    let batched_slice = batched_c.as_slice().unwrap();
    for i in 0..5 {
        let b_i = b.narrow(0, i, 1).unwrap().reshape(&[6, 8]).unwrap();
        let expected = ops.gemm(&a_2d, &b_i).unwrap();
        let got = &batched_slice[i * 4 * 8..(i + 1) * 4 * 8];
        assert_parity(
            "gemm_batched lhs broadcast vs per-batch gemm",
            got,
            expected.contiguous().as_slice().unwrap(),
        );
    }
}

/// 中間軸を含む broadcast（`[2,1,m,k]` × `[1,3,k,n]`）も正しく処理する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_broadcasts_middle_batch_axis() {
    let ops = MetalBackendOps::new();
    let (m, k, n) = (3usize, 4usize, 5usize);
    let a = tensor(random_matrix(0x35, 2 * m * k), &[2, 1, m, k]);
    let b = tensor(random_matrix(0x45, 3 * k * n), &[1, 3, k, n]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(batched.shape(), &[2, 3, m, n]);

    let a_c = a.contiguous();
    let batched_c = batched.contiguous();
    let batched_slice = batched_c.as_slice().unwrap();
    for i in 0..2 {
        for j in 0..3 {
            let a_ij = a_c
                .narrow(0, i, 1)
                .unwrap()
                .reshape(&[1, m, k])
                .unwrap()
                .narrow(0, 0, 1)
                .unwrap()
                .reshape(&[m, k])
                .unwrap();
            let b_j = b
                .narrow(1, j, 1)
                .unwrap()
                .reshape(&[1, k, n])
                .unwrap()
                .narrow(0, 0, 1)
                .unwrap()
                .reshape(&[k, n])
                .unwrap();
            let expected = ops.gemm(&a_ij, &b_j).unwrap();
            let got = &batched_slice[(i * 3 + j) * m * n..(i * 3 + j + 1) * m * n];
            assert_parity(
                "gemm_batched middle-axis broadcast vs per-batch gemm",
                got,
                expected.contiguous().as_slice().unwrap(),
            );
        }
    }
}

/// (T5) 形状不整合（内部次元不一致）は `BackendError::ShapeMismatch`。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_inner_dim_mismatch_is_shape_mismatch() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0xB0, 2 * 3 * 4), &[2, 3, 4]);
    let b = tensor(random_matrix(0xC0, 2 * 5 * 6), &[2, 5, 6]);
    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// (T5) バッチ broadcast 不可は `BackendError::ShapeMismatch`。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_batch_broadcast_incompatible_is_shape_mismatch() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0xD0, 2 * 3 * 4), &[2, 3, 4]);
    let b = tensor(random_matrix(0xE0, 3 * 4 * 5), &[3, 4, 5]);
    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// (T6) バッチ数 0 は要素数 0 の出力を返す（GPU 起動なし）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_zero_batch_returns_empty_output() {
    let ops = MetalBackendOps::new();
    let a = tensor(Vec::new(), &[0, 3, 4]);
    let b = tensor(Vec::new(), &[0, 4, 5]);
    let out = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(out.shape(), &[0, 3, 5]);
    assert_eq!(out.numel(), 0);
}

/// (T6) `m == 0`／`n == 0` は要素数 0 の出力を返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_zero_m_or_n_returns_empty_output() {
    let ops = MetalBackendOps::new();
    let a = tensor(Vec::new(), &[2, 0, 4]);
    let b = tensor(random_matrix(0x50, 2 * 4 * 5), &[2, 4, 5]);
    let out = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 0, 5]);
    assert_eq!(out.numel(), 0);
}

/// (T6) `k == 0`（`m, n > 0`）は GPU 起動なしの全 0 テンソルを返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_zero_k_returns_zero_filled_output() {
    let ops = MetalBackendOps::new();
    let a = tensor(Vec::new(), &[2, 3, 0]);
    let b = tensor(Vec::new(), &[2, 0, 4]);
    let out = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 3, 4]);
    let out_c = out.contiguous();
    assert!(out_c.as_slice().unwrap().iter().all(|&x| x == 0.0));
}

/// (T7) `gemm_batched_fp32_strict` は Metal（TF32 の概念を持たない）
/// では `gemm_batched` と bit 同一。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_fp32_strict_matches_gemm_batched_bit_exact() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0xF0, 2 * 6 * 7), &[2, 6, 7]);
    let b = tensor(random_matrix(0x100, 2 * 7 * 8), &[2, 7, 8]);

    let standard = ops.gemm_batched(&a, &b).unwrap();
    let strict = ops.gemm_batched_fp32_strict(&a, &b).unwrap();
    assert_eq!(
        standard.contiguous().as_slice().unwrap(),
        strict.contiguous().as_slice().unwrap()
    );
}

/// (T8) run-to-run 決定性: 同入力を 2 回呼んでも bit 完全一致する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_is_deterministic_across_runs() {
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0x110, 4 * 33 * 17), &[4, 33, 17]);
    let b = tensor(random_matrix(0x120, 4 * 17 * 29), &[4, 17, 29]);

    let run1 = ops.gemm_batched(&a, &b).unwrap();
    let run2 = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(
        run1.contiguous().as_slice().unwrap(),
        run2.contiguous().as_slice().unwrap(),
        "gemm_batched: run-to-run で出力がビット単位で一致しなかった"
    );
}

/// (T9) split-K 実行時トグル非干渉: `gemm_batched` は `dispatch_auto`
/// （split-K 経路を持つ）を経由しないため、トグルの on/off で出力が
/// 変わらないことを確認する（`docs/backend-metal-splitk-decision.md`
/// §5「実行時トグル」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_output_unaffected_by_split_k_runtime_toggle() {
    let _guard = RuntimeFlagGuard::acquire();
    let ops = MetalBackendOps::new();
    let a = tensor(random_matrix(0x130, 3 * 64 * 64), &[3, 64, 64]);
    let b = tensor(random_matrix(0x140, 3 * 64 * 64), &[3, 64, 64]);

    set_split_k_enabled(false);
    let off = ops.gemm_batched(&a, &b).unwrap();

    set_split_k_enabled(true);
    let on = ops.gemm_batched(&a, &b).unwrap();

    assert_eq!(
        off.contiguous().as_slice().unwrap(),
        on.contiguous().as_slice().unwrap(),
        "gemm_batched: split-K 実行時トグルの on/off で出力が変わった\
         （本経路が誤って dispatch_auto を経由している可能性がある）"
    );
}

/// (T10) VJP と同型の転置 view 入力（`transpose(1, 2)` 後の rank 3
/// view。`autodiff::grad::matmul_vjp` の `transpose_last2` が渡す形状と
/// 同型）でも `normalize_batched_operand` が正しく再パックし、CPU 参照
/// 実装と一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_batched_handles_transposed_view_input() {
    let ops = MetalBackendOps::new();
    let (b, m, k, n) = (3usize, 5usize, 7usize, 4usize);
    // `raw` は `[b, k, m]`。`transpose(1, 2)` で論理形状 `[b, m, k]`
    // （非 contiguous）へ変換して `gemm_batched` の lhs として渡す。
    let raw = tensor(random_matrix(0x150, b * k * m), &[b, k, m]);
    let a = raw.transpose(1, 2).unwrap();
    assert!(!a.is_contiguous());
    let bb = tensor(random_matrix(0x160, b * k * n), &[b, k, n]);

    let batched = ops.gemm_batched(&a, &bb).unwrap();
    assert_eq!(batched.shape(), &[b, m, n]);

    let a_c = a.contiguous();
    let batched_c = batched.contiguous();
    let batched_slice = batched_c.as_slice().unwrap();
    for i in 0..b {
        let a_i = a_c.narrow(0, i, 1).unwrap().reshape(&[m, k]).unwrap();
        let b_i = bb.narrow(0, i, 1).unwrap().reshape(&[k, n]).unwrap();
        let expected = ops.gemm(&a_i, &b_i).unwrap();
        let got = &batched_slice[i * m * n..(i + 1) * m * n];
        assert_parity(
            "gemm_batched transposed view vs per-batch gemm",
            got,
            expected.contiguous().as_slice().unwrap(),
        );
    }
}
