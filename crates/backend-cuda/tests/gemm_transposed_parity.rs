//! `CudaBackendOps::gemm_fp32_strict`（`BackendOps` トレイト経由）／
//! `gemm_resident_lhs` の VJP 専用 NT/TN 2 パターン入口（イシュー #1214）
//! の受け入れ基準対応テスト。`backend-cpu::tests::gemm_transposed_parity`
//! （#1213）と同じ構成方針を CUDA 実機向けに移植する。
//!
//! 本イシューの設計（`docs/matmul-vjp-zero-copy-decision.md` §4.3）は
//! 「転置元 storage を GPU 側 smem 転置カーネルで転置してから既存 NN
//! GEMM カーネルへ渡す」ため、転置カーネル・GEMM カーネルいずれも
//! 決定的な純データ移動・積和のみで丸めを追加しない。よって NT/TN 入口
//! を経由した結果は `contiguous()` してから渡した結果と **bit 完全一致**
//! する契約（CPU 版 #1213 と同じ判断基準）。加えて CPU 参照実装との
//! REQ-2 統一複合判定（`fandhe_ai_backend_cpu::assert_parity`）も確認する
//! （受け入れ基準 C）。TT（両方転置）・`narrow` 後の一般 stride は本
//! イシューのスコープ外で `contiguous()` フォールバックのまま結果が
//! 正しいことのみを確認する。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    // 実機依存テストのため決定的な軽量疑似乱数生成（外部依存追加を避ける。
    // `.claude/rules/deps-policy.md`）。xorshift 系の最小実装。
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// `gemm_fp32_strict(g, w.transpose_2d())`（NT: 元の B `[n,k]` を転置した
/// `[k,n]` view）が `gemm_fp32_strict(g, w.transpose_2d().contiguous())`
/// と bit 完全一致し、かつ CPU 参照実装と REQ-2 複合判定内で一致する
/// ことを、整列形状（cp.async pipeline 経路。`n%4==0 && k%4==0`）・非整列
/// 形状（classic 経路）・32 の非倍数を跨ぐ複数形状で確認する
/// （`matmul_vjp` の d_input `g @ Wᵀ` が該当）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_nt_transposed_view_matches_contiguous_and_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (64, 256, 784),
        (128, 96, 128),
    ] {
        let g = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
        // w: 論理形状 [n,k]（Linear.weight 相当）。transpose_2d() で [k,n] view。
        let w = tensor(random_matrix(0x2000 + n as u64, n * k), &[n, k]);
        let w_t = w.transpose_2d().unwrap();

        let actual = cuda_ops.gemm_fp32_strict(&g, &w_t).unwrap();
        let expected = cuda_ops
            .gemm_fp32_strict(&g, &tensor(contiguous_slice(&w_t), &[k, n]))
            .unwrap();
        let cpu_expected = cpu_ops
            .gemm(&g, &tensor(contiguous_slice(&w_t), &[k, n]))
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "NT 入口は contiguous() 経路と bit 完全一致するはず（m={m} k={k} n={n}）"
        );
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm NT vs CPU reference m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &contiguous_slice(&cpu_expected),
        );
    }
}

/// [`gemm_nt_transposed_view_matches_contiguous_and_cpu_reference_on_real_device`]
/// の TN 版（`x.transpose_2d()`。`matmul_vjp` の d_weight `Aᵀ @ g` が
/// 該当）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_tn_transposed_view_matches_contiguous_and_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (256, 64, 784),
        (96, 128, 128),
    ] {
        let x = tensor(random_matrix(0x3000 + m as u64, m * k), &[m, k]);
        let x_t = x.transpose_2d().unwrap();
        let g = tensor(random_matrix(0x4000 + n as u64, m * n), &[m, n]);

        let actual = cuda_ops.gemm_fp32_strict(&x_t, &g).unwrap();
        let expected = cuda_ops
            .gemm_fp32_strict(&tensor(contiguous_slice(&x_t), &[k, m]), &g)
            .unwrap();
        let cpu_expected = cpu_ops
            .gemm(&tensor(contiguous_slice(&x_t), &[k, m]), &g)
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "m={m} k={k} n={n}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "TN 入口は contiguous() 経路と bit 完全一致するはず（m={m} k={k} n={n}）"
        );
        fandhe_ai_backend_cpu::assert_parity(
            &format!("gemm TN vs CPU reference m={m} k={k} n={n}"),
            &contiguous_slice(&actual),
            &contiguous_slice(&cpu_expected),
        );
    }
}

/// 両方転置（TT）は本イシューのスコープ外（従来どおり `contiguous()`
/// フォールバック）だが、計算結果自体は正しいことを確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_tt_both_transposed_falls_back_but_still_correct_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let (big_m, big_k, big_n) = (17usize, 23usize, 19usize);
    let orig_a = tensor(random_matrix(0x5000, big_k * big_m), &[big_k, big_m]);
    let a_t = orig_a.transpose_2d().unwrap(); // [big_m, big_k]
    let orig_b = tensor(random_matrix(0x6000, big_n * big_k), &[big_n, big_k]);
    let b_t = orig_b.transpose_2d().unwrap(); // [big_k, big_n]

    let actual = cuda_ops.gemm_fp32_strict(&a_t, &b_t).unwrap();
    let expected = cuda_ops
        .gemm_fp32_strict(
            &tensor(contiguous_slice(&a_t), &[big_m, big_k]),
            &tensor(contiguous_slice(&b_t), &[big_k, big_n]),
        )
        .unwrap();

    assert_eq!(contiguous_slice(&actual), contiguous_slice(&expected));
}

/// `narrow` 後の転置（一般 stride）は `contiguous()` フォールバック経由
/// でも結果自体は正しいことを確認する（一般 stride 化は本イシューの
/// スコープ外。`docs/matmul-vjp-zero-copy-decision.md` §3.2）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_narrow_then_transpose_general_stride_falls_back_but_still_correct_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let (m, k, n) = (6usize, 5usize, 4usize);
    let w0 = tensor(random_matrix(0x7000, n * (k + 2)), &[n, k + 2]);
    let w_narrowed = w0.narrow(1, 1, k).unwrap();
    let w_t = w_narrowed.transpose_2d().unwrap();
    assert!(!w_t.is_contiguous());

    let g = tensor(random_matrix(0x8000, m * k), &[m, k]);
    let actual = cuda_ops.gemm_fp32_strict(&g, &w_t).unwrap();
    let expected = cuda_ops
        .gemm_fp32_strict(&g, &tensor(contiguous_slice(&w_t), &[k, n]))
        .unwrap();

    assert_eq!(contiguous_slice(&actual), contiguous_slice(&expected));
}

/// `gemm_resident_lhs(w_dev, g.transpose_2d())`（NT: `g` が転置格納）が、
/// `gemm_resident_lhs(w_dev, g.transpose_2d().contiguous())` と bit 完全
/// 一致することを確認する（`Op::LinearResident` の d_input `W @ gᵀ` が
/// 該当。イシュー #1214）。`gemm_resident_real_device.rs` の
/// `DeviceBufferView` 構築手順を踏襲する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_resident_lhs_nt_transposed_view_matches_contiguous_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    for &(p, q, r) in &[(1usize, 1usize, 1usize), (4, 8, 4), (37, 65, 33)] {
        let w = tensor(random_matrix(0x9000 + p as u64, p * q), &[p, q]);
        // g0: 論理形状 [r,q]。transpose_2d() で [q,r] view
        // （gemm_resident_lhs の第 2 引数 `b` の期待形状）。
        let g0 = tensor(random_matrix(0xa000 + r as u64, r * q), &[r, q]);
        let g_t = g0.transpose_2d().unwrap();

        let w_dev = cuda_mem.upload(&w).unwrap();
        let w_shape = [p, q];
        let w_view = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let actual = cuda_ops.gemm_resident_lhs(w_view, &g_t).unwrap();

        let w_view2 = DeviceBufferView::new(&w_dev, 0, &w_shape).unwrap();
        let expected = cuda_ops
            .gemm_resident_lhs(w_view2, &tensor(contiguous_slice(&g_t), &[q, r]))
            .unwrap();

        assert_eq!(actual.shape(), expected.shape(), "p={p} q={q} r={r}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "gemm_resident_lhs の NT 入口は contiguous() 経路と bit 完全一致するはず \
             （p={p} q={q} r={r}）"
        );
    }
}
