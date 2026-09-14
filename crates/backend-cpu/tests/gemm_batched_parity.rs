//! `CpuBackendOps::gemm_batched`／`gemm_batched_fp32_strict` の受け入れ
//! 基準対応テスト（イシュー #1715。親 #1600）。
//!
//! `fandhe_ai_tensor_core::BackendOps` に非破壊追加したバッチ行列積
//! （`gemm_batched`／`gemm_batched_fp32_strict`。既定は per-batch
//! `gemm`/`gemm_fp32_strict` への合成）の CPU オーバーライドが、
//!
//! 1. 各バッチの計算結果が同 shape の 2 次元 `gemm`（バッチをほどいて
//!    個別に呼んだもの）と **bit 完全一致**すること、
//! 2. `matmul_reference_fma`（スカラー `mul_add` 参照実装。REQ-2）との
//!    複合判定（`assert_parity`）を満たすこと、
//! 3. lhs／rhs いずれかのバッチ次元 1（NumPy 互換ブロードキャスト）を
//!    正しく処理すること、
//! 4. rank 2 同士は `gemm` への直接委譲で bit 同一になること、
//! 5. shape 不整合（rank<2・内部次元不一致・バッチ broadcast 不可）が
//!    `BackendError::ShapeMismatch` を返すこと、
//! 6. バッチ数 0 が空出力を返すこと
//!
//! を検証する。CUDA／Metal 専用バッチカーネルは後続イシュー
//! （#1716／#1717）の対象であり、本ファイルでは扱わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::assert_parity;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// スカラー `f32::mul_add` 参照実装（REQ-2 の FMA 契約突合対象）。
/// `a: [m, k]`・`b: [k, n]` の 2 次元のみを受け付ける。
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

/// バッチ `B=3`・複数形状（1×1×1 の退化形状・非正方・大きめ）で
/// `gemm_batched` の各バッチが同 shape の 2 次元 `gemm`（バッチを
/// ほどいて個別呼び出ししたもの）と bit 完全一致することを確認する。
#[test]
fn gemm_batched_matches_per_batch_gemm_bit_exact() {
    let ops = CpuBackendOps::new();

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

        for i in 0..b {
            let a_i = a.narrow(0, i, 1).unwrap().reshape(&[m, k]).unwrap();
            let b_i = bb.narrow(0, i, 1).unwrap().reshape(&[k, n]).unwrap();
            let expected = ops.gemm(&a_i, &b_i).unwrap();
            let got = batched.narrow(0, i, 1).unwrap().reshape(&[m, n]).unwrap();
            assert_eq!(
                got.as_slice().unwrap(),
                expected.as_slice().unwrap(),
                "batch {i} mismatch for shape ({b},{m},{k},{n})"
            );
        }

        // REQ-2: スカラー FMA 参照実装との複合判定（各バッチ独立に検査）。
        let a_c = a.contiguous();
        let b_c = bb.contiguous();
        let a_slice = a_c.as_slice().unwrap();
        let b_slice = b_c.as_slice().unwrap();
        let batched_c = batched.contiguous();
        let batched_slice = batched_c.as_slice().unwrap();
        for i in 0..b {
            let a_i = &a_slice[i * m * k..(i + 1) * m * k];
            let b_i = &b_slice[i * k * n..(i + 1) * k * n];
            let expected = matmul_reference_fma(a_i, b_i, m, k, n);
            let got = &batched_slice[i * m * n..(i + 1) * m * n];
            assert_parity("gemm_batched vs scalar FMA reference", got, &expected);
        }
    }
}

/// rank 2 同士は `gemm` への直接委譲で bit 同一になる
/// （既定合成実装・CPU オーバーライド共通の契約）。
#[test]
fn gemm_batched_rank2_delegates_to_gemm_bit_exact() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0x10, 37 * 41), &[37, 41]);
    let b = tensor(random_matrix(0x20, 41 * 29), &[41, 29]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    let direct = ops.gemm(&a, &b).unwrap();
    assert_eq!(batched.shape(), direct.shape());
    assert_eq!(batched.as_slice().unwrap(), direct.as_slice().unwrap());
}

/// lhs バッチ次元 1（NumPy 互換ブロードキャスト。`[1, m, k]`）が
/// 出力バッチ数ぶん複製されて使われることを確認する。
#[test]
fn gemm_batched_broadcasts_lhs_batch_dim_one() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0x30, 4 * 6), &[1, 4, 6]);
    let b = tensor(random_matrix(0x40, 5 * 6 * 8), &[5, 6, 8]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(batched.shape(), &[5, 4, 8]);

    let a_2d = a.reshape(&[4, 6]).unwrap();
    for i in 0..5 {
        let b_i = b.narrow(0, i, 1).unwrap().reshape(&[6, 8]).unwrap();
        let expected = ops.gemm(&a_2d, &b_i).unwrap();
        let got = batched.narrow(0, i, 1).unwrap().reshape(&[4, 8]).unwrap();
        assert_eq!(got.as_slice().unwrap(), expected.as_slice().unwrap());
    }
}

/// rhs が 2 次元（暗黙のバッチ rank 0。全バッチへブロードキャスト）の
/// ケース。
#[test]
fn gemm_batched_broadcasts_rhs_2d() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0x50, 4 * 3 * 5), &[4, 3, 5]);
    let b = tensor(random_matrix(0x60, 5 * 7), &[5, 7]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(batched.shape(), &[4, 3, 7]);

    for i in 0..4 {
        let a_i = a.narrow(0, i, 1).unwrap().reshape(&[3, 5]).unwrap();
        let expected = ops.gemm(&a_i, &b).unwrap();
        let got = batched.narrow(0, i, 1).unwrap().reshape(&[3, 7]).unwrap();
        assert_eq!(got.as_slice().unwrap(), expected.as_slice().unwrap());
    }
}

/// 非 contiguous 入力（`permute` 後の view）でも正しく正規化されて
/// 結果が一致することを確認する。
#[test]
fn gemm_batched_non_contiguous_input() {
    let ops = CpuBackendOps::new();
    // [3, 4, 5] を作ってから軸 0/1 を permute し、論理形状 [4, 3, 5]
    // （非 contiguous）を `gemm_batched` に渡す。
    let raw = tensor(random_matrix(0x70, 3 * 4 * 5), &[3, 4, 5]);
    let a = raw.permute(&[1, 0, 2]).unwrap();
    assert!(!a.is_contiguous());
    let b = tensor(random_matrix(0x80, 4 * 5 * 6), &[4, 5, 6]);

    let batched = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(batched.shape(), &[4, 3, 6]);

    // `a` は非 contiguous（permute 後）のため、期待値の算出では
    // 先に `contiguous()` してから `narrow`/`reshape` する
    // （production 側の正規化と同じ手順。`reshape` は contiguous
    // 前提のため、非 contiguous のまま `narrow`/`reshape` すると
    // `NonContiguousReshape` になりうる）。
    let a_c = a.contiguous();
    for i in 0..4 {
        let a_i = a_c.narrow(0, i, 1).unwrap().reshape(&[3, 5]).unwrap();
        let b_i = b.narrow(0, i, 1).unwrap().reshape(&[5, 6]).unwrap();
        let expected = ops.gemm(&a_i, &b_i).unwrap();
        let got = batched.narrow(0, i, 1).unwrap().reshape(&[3, 6]).unwrap();
        assert_eq!(got.as_slice().unwrap(), expected.as_slice().unwrap());
    }
}

/// rank 1 の入力は `RankMismatch` として `ShapeMismatch` へ包まれる。
#[test]
fn gemm_batched_rank_below_2_is_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0x90, 5), &[5]);
    let b = tensor(random_matrix(0xA0, 5 * 4), &[5, 4]);
    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// 内部次元不一致は `MatmulDimMismatch` として `ShapeMismatch` へ包まれる。
#[test]
fn gemm_batched_inner_dim_mismatch_is_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0xB0, 2 * 3 * 4), &[2, 3, 4]);
    let b = tensor(random_matrix(0xC0, 2 * 5 * 6), &[2, 5, 6]);
    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// バッチ次元同士が（1 でもなく）不一致な場合は
/// `BroadcastIncompatible` として `ShapeMismatch` へ包まれる。
#[test]
fn gemm_batched_batch_broadcast_incompatible_is_shape_mismatch() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0xD0, 2 * 3 * 4), &[2, 3, 4]);
    let b = tensor(random_matrix(0xE0, 3 * 4 * 5), &[3, 4, 5]);
    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// バッチ数 0 は要素数 0 の出力を返す（fail-closed だが panic しない）。
#[test]
fn gemm_batched_zero_batch_returns_empty_output() {
    let ops = CpuBackendOps::new();
    let a = tensor(Vec::new(), &[0, 3, 4]);
    let b = tensor(Vec::new(), &[0, 4, 5]);
    let out = ops.gemm_batched(&a, &b).unwrap();
    assert_eq!(out.shape(), &[0, 3, 5]);
    assert_eq!(out.numel(), 0);
}

/// 出力要素数のバイトサイズが `Vec` の allocation 上限（`isize::MAX`
/// バイト）を超える巨大なバッチ次元は、実際に確保を試みてパニックする
/// のではなく `BackendError::ShapeMismatch` を返す（PR #1810
/// codex-review P1 是正の回帰テスト）。
///
/// `B = isize::MAX as usize / 4 + 1` を選ぶと `B * size_of::<f32>()`
/// （`m = n = 1` なので出力要素数は `B` に一致）がちょうど
/// `isize::MAX` を 1 バイト超える。入力オペランドは `k = 0`（内部次元
/// 0）の shape `[B, 1, 0]`／`[B, 0, 1]` を用いることで、要素数積が
/// 0（`B * 1 * 0 = 0`）になり `Tensor::new` 自体は巨大な `B` でも
/// 実データを 1 バイトも確保せずに構築できる（テスト自体がメモリを
/// 消費しないことを保証する）。
#[test]
fn gemm_batched_huge_batch_output_bytes_overflow_is_shape_mismatch_not_panic() {
    let huge_batch = isize::MAX as usize / 4 + 1;
    let ops = CpuBackendOps::new();
    let a = tensor(Vec::new(), &[huge_batch, 1, 0]);
    let b = tensor(Vec::new(), &[huge_batch, 0, 1]);

    let err = ops.gemm_batched(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));

    let err = ops.gemm_batched_fp32_strict(&a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

/// `gemm_batched_fp32_strict` は CPU（TF32 の概念を持たない）では
/// `gemm_batched` と bit 同一。
#[test]
fn gemm_batched_fp32_strict_matches_gemm_batched_bit_exact() {
    let ops = CpuBackendOps::new();
    let a = tensor(random_matrix(0xF0, 2 * 6 * 7), &[2, 6, 7]);
    let b = tensor(random_matrix(0x100, 2 * 7 * 8), &[2, 7, 8]);

    let standard = ops.gemm_batched(&a, &b).unwrap();
    let strict = ops.gemm_batched_fp32_strict(&a, &b).unwrap();
    assert_eq!(standard.as_slice().unwrap(), strict.as_slice().unwrap());
}
