//! `CudaBackendOps::gemm_fp32_strict_into`（イシュー #1559）の受け入れ
//! 基準対応テスト。`gemm_transposed_parity.rs`（#1214）と同じ疑似乱数・
//! 構成方針を踏襲し、device-resident 直書き込み経路
//! （`gemm::CudaGemm::launch_tiled_f32_nt_into`／`_tn_into`）が
//! `gemm_fp32_strict`（ホスト戻り値経路）と bit 完全一致することを
//! 確認する。
//!
//! `docs/train-resident-grad-device-update.md`・`docs/device-resident-
//! update-design.md` が定める契約（`out[out_offset..out_offset+m*n]` を
//! 上書きし、範囲外は変化させない）も併せて検証する。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_fp32_strict_into_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    // 実機依存テストのため決定的な軽量疑似乱数生成（外部依存追加を避ける。
    // `.claude/rules/deps-policy.md`）。xorshift 系の最小実装
    // （`gemm_transposed_parity.rs` と同一）。
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

/// NT（`b` が転置格納）・TN（`a` が転置格納）双方について、整列形状
/// （cp.async pipeline 経路。`n%4==0 && k%4==0`）・非整列形状（classic
/// 経路）を含む複数形状で `gemm_fp32_strict_into` の結果（`download` で
/// 読み戻し）が `gemm_fp32_strict` の結果と bit 完全一致することを
/// 確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_nt_and_tn_match_gemm_fp32_strict_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (64, 256, 784),
        (128, 96, 128),
    ] {
        // NT: g @ w_t（w は論理形状 [n,k]。Linear.weight 相当）。
        let g = tensor(random_matrix(0x1000 + m as u64, m * k), &[m, k]);
        let w = tensor(random_matrix(0x2000 + n as u64, n * k), &[n, k]);
        let w_t = w.transpose_2d().unwrap();

        let expected = cuda_ops.gemm_fp32_strict(&g, &w_t).unwrap();
        let mut out = mem.alloc_zeroed(&[m, n]).unwrap();
        cuda_ops
            .gemm_fp32_strict_into(&g, &w_t, &mut out, 0)
            .unwrap();
        let actual = mem.download(&out).unwrap();

        assert_eq!(actual.shape(), expected.shape(), "NT m={m} k={k} n={n}");
        assert_eq!(
            contiguous_slice(&actual),
            contiguous_slice(&expected),
            "gemm_fp32_strict_into の NT 入口は gemm_fp32_strict と bit 完全一致するはず \
             （m={m} k={k} n={n}）"
        );

        // TN: x_t @ g2（x は論理形状 [m,k] → 転置元は [k,m]）。
        let x = tensor(random_matrix(0x3000 + m as u64, m * k), &[m, k]);
        let x_t = x.transpose_2d().unwrap();
        let g2 = tensor(random_matrix(0x4000 + n as u64, m * n), &[m, n]);

        let expected_tn = cuda_ops.gemm_fp32_strict(&x_t, &g2).unwrap();
        let mut out_tn = mem.alloc_zeroed(&[k, n]).unwrap();
        cuda_ops
            .gemm_fp32_strict_into(&x_t, &g2, &mut out_tn, 0)
            .unwrap();
        let actual_tn = mem.download(&out_tn).unwrap();

        assert_eq!(
            actual_tn.shape(),
            expected_tn.shape(),
            "TN m={m} k={k} n={n}"
        );
        assert_eq!(
            contiguous_slice(&actual_tn),
            contiguous_slice(&expected_tn),
            "gemm_fp32_strict_into の TN 入口は gemm_fp32_strict と bit 完全一致するはず \
             （m={m} k={k} n={n}）"
        );
    }
}

/// NN（分類不能・フォールバック経路）でも `gemm_fp32_strict` と bit 完全
/// 一致することを確認する（`gemm_fp32_strict_into_impl` のフォールバック
/// 分岐 `self.gemm_fp32_strict(a, b)` → `upload_into` の正しさ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_nn_fallback_matches_gemm_fp32_strict_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let (m, k, n) = (37usize, 65usize, 33usize);
    let a = tensor(random_matrix(0x5000, m * k), &[m, k]);
    let b = tensor(random_matrix(0x6000, k * n), &[k, n]);

    let expected = cuda_ops.gemm_fp32_strict(&a, &b).unwrap();
    let mut out = mem.alloc_zeroed(&[m, n]).unwrap();
    cuda_ops.gemm_fp32_strict_into(&a, &b, &mut out, 0).unwrap();
    let actual = mem.download(&out).unwrap();

    assert_eq!(
        contiguous_slice(&actual),
        contiguous_slice(&expected),
        "NN フォールバックは gemm_fp32_strict と bit 完全一致するはず"
    );
}

/// `out_offset` に非ゼロを指定した場合、書き込み範囲より前の領域が
/// 変化しないこと（範囲外不変契約）・書き込み範囲は正しい結果になる
/// ことを確認する（`DeviceParamStore` の連結バッファへの直接書き込みを
/// 想定した契約）。偶数・奇数双方のオフセットで確認し、アラインメント
/// 制約がないこと（設計判断 B。`gemm.rs::CudaGemm::
/// launch_tiled_f32_nt_into` ドキュメンテーションコメント参照）も
/// 実機的に裏付ける。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_respects_nonzero_offset_and_leaves_out_of_range_untouched_on_real_device()
{
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let (m, k, n) = (4usize, 8usize, 4usize);
    let mn = m * n;
    let g = tensor(random_matrix(0x7000, m * k), &[m, k]);
    let w = tensor(random_matrix(0x8000, n * k), &[n, k]);
    let w_t = w.transpose_2d().unwrap();
    let expected = cuda_ops.gemm_fp32_strict(&g, &w_t).unwrap();
    let expected_slice = contiguous_slice(&expected);

    for &offset in &[0usize, 1, 3] {
        let total = offset + mn + 5; // 書き込み範囲の前後にマージンを持たせる。
        let sentinel = tensor(vec![f32::NAN; total], &[total]);
        let mut out = mem.upload(&sentinel).unwrap();

        cuda_ops
            .gemm_fp32_strict_into(&g, &w_t, &mut out, offset)
            .unwrap();
        let actual = mem.download(&out).unwrap();
        let actual_slice = contiguous_slice(&actual);

        for v in &actual_slice[..offset] {
            assert!(
                v.is_nan(),
                "offset={offset}: 書き込み範囲より前の領域は変化しないはず"
            );
        }
        assert_eq!(
            &actual_slice[offset..offset + mn],
            expected_slice.as_slice(),
            "offset={offset}: 書き込み範囲は gemm_fp32_strict と bit 完全一致するはず"
        );
        for v in &actual_slice[offset + mn..] {
            assert!(
                v.is_nan(),
                "offset={offset}: 書き込み範囲より後ろの領域は変化しないはず"
            );
        }
    }
}

/// 誤った ordinal の `out`（`Device::Cuda(other)`）を渡すと
/// `DeviceMismatch` になることを確認する（`gemm_fp32_strict_into_impl`
/// の host-only チェック。GPU 非依存ユニットテストと同型の内容だが、
/// 実機を要する `#[ignore]` テストファイルにも同居させることで実機
/// バイナリ単体でも境界検査を確認できるようにする）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_rejects_out_on_different_ordinal_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");

    let (m, k, n) = (2usize, 2usize, 2usize);
    let a = tensor(random_matrix(0x9000, m * k), &[m, k]);
    let b = tensor(random_matrix(0xa000, k * n), &[k, n]);
    let mut out = mem.alloc_zeroed(&[m, n]).unwrap();
    // 実在しない別 ordinal を偽装するため、ハンドルの中身は変えず
    // `DeviceBuffer` の device タグだけを差し替えられないため、代わりに
    // 現在の ordinal + 1（この呼び出し元 `cuda_ops` からは常に不一致）を
    // 使い `CudaBackendOps::new` 側で検証させる。
    let other_ordinal = device.ordinal() + 1;
    let other_ops = CudaBackendOps::new(other_ordinal);
    // `out` は ordinal 0 のバッファのまま、other_ops（ordinal+1）から
    // 呼び出すことで DeviceMismatch を発生させる。
    let result = other_ops.gemm_fp32_strict_into(&a, &b, &mut out, 0);
    assert!(
        matches!(result, Err(BackendError::DeviceMismatch)),
        "別 ordinal からの呼び出しは DeviceMismatch で拒否されるはず: {result:?}"
    );
    assert_eq!(out.device(), Device::Cuda(device.ordinal()));
}
