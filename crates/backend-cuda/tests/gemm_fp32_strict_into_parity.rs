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

// ---------------------------------------------------------------------
// bias 勾配のデバイス縮約（イシュー #3204）
//
// `gemm_fp32_strict_into_with_bias_reduce_tracked` が weight を
// `gemm_fp32_strict` と bit 一致で、bias を「`f64` アキュムレータで行昇順に
// 加算し最後に 1 回 `f32`（縮約軸長 1 は直接コピー）」のホスト参照と bit 一致
// で書くことを確認する。NaN のみ payload がハードウェア依存のためクラス一致。
// ---------------------------------------------------------------------

/// `autodiff::eval::reduce_bias_grad_rows` と同一契約のホスト参照
/// （`backend-cuda` は `autodiff` に依存できないため独立に再実装する）。
fn host_bias_reference(b: &Tensor<f32>) -> Vec<f32> {
    let data = contiguous_slice(b);
    let (rows, cols) = (b.shape()[0], b.shape()[1]);
    if rows == 1 {
        return data[..cols].to_vec();
    }
    let mut acc = vec![0f64; cols];
    for r in 0..rows {
        for (c, a) in acc.iter_mut().enumerate() {
            *a += f64::from(data[r * cols + c]);
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
}

/// NaN 同士はクラス一致、それ以外は bit 完全一致（`-0.0` と `+0.0` も区別）。
fn assert_bits_or_nan_class_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: 長さ不一致");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{what}[{i}]: NaN クラス不一致 {a} vs {e}"
            );
        } else {
            assert_eq!(a.to_bits(), e.to_bits(), "{what}[{i}]: {a} vs {e}");
        }
    }
}

/// weight（`[m, n]`）の前後どちらかに bias（`n`）を置いて検証する。
/// `bias_first == true` なら bias を weight より前に置く。
fn check_bias_reduce(
    cuda_ops: &CudaBackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    bias_first: bool,
    label: &str,
) {
    use fandhe_ai_tensor_core::DispatchFailureCell;
    let mem = cuda_ops.memory_ops().expect("MemoryOps");
    let (m, n) = (a.shape()[0], b.shape()[1]);
    let expected_w = cuda_ops.gemm_fp32_strict(a, b).unwrap();
    let expected_b = host_bias_reference(b);

    let total = m * n + n + 8;
    let (w_off, b_off) = if bias_first {
        (n + 4, 2)
    } else {
        (3, m * n + 3 + 2)
    };
    let mut out = mem.alloc_zeroed(&[total]).unwrap();
    // 範囲外を NaN で事前に埋め、書き込み範囲外が不変であることを確認する。
    let sentinel = tensor(vec![f32::NAN; total], &[total]);
    mem.upload_into(&sentinel, &mut out, 0).unwrap();

    let token = DispatchFailureCell::default();
    let filled = cuda_ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            a,
            b,
            &mut out,
            w_off,
            Some((b_off, n)),
            &token,
        )
        .unwrap();
    assert!(filled, "{label}: bias を書いたら Ok(true)");
    let all = contiguous_slice(&mem.download(&out).unwrap());
    assert_bits_or_nan_class_eq(
        &all[w_off..w_off + m * n],
        &contiguous_slice(&expected_w),
        &format!("{label} weight"),
    );
    assert_bits_or_nan_class_eq(
        &all[b_off..b_off + n],
        &expected_b,
        &format!("{label} bias"),
    );
    for (i, v) in all.iter().enumerate() {
        let inside = (w_off..w_off + m * n).contains(&i) || (b_off..b_off + n).contains(&i);
        if !inside {
            assert!(v.is_nan(), "{label}: 範囲外 [{i}] が書き換えられた: {v}");
        }
    }
}

/// TN・NT・NN フォールバックで bias が `gemm_fp32_strict`／ホスト参照と bit
/// 一致し、weight の前後どちらの配置でも範囲外が不変であること。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_bias_reduce_matches_host_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    // (m, k, n): TN は x_t[k, m] @ g[m, n]、NT は g[m, k] @ w_t[k, n]。
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (64, 256, 784),
        (128, 96, 128),
        (5, 1, 7), // NT・NN の縮約軸長 1（直接コピー・-0.0 保持の分岐）
        (6, 9, 1), // n == 1
    ] {
        for bias_first in [false, true] {
            // TN: 縮約軸は g の行（m 行）。
            let x = tensor(random_matrix(0x7000 + m as u64, m * k), &[m, k]);
            let x_t = x.transpose_2d().unwrap();
            let g = tensor(random_matrix(0x7100 + n as u64, m * n), &[m, n]);
            check_bias_reduce(
                &cuda_ops,
                &x_t,
                &g,
                bias_first,
                &format!("TN m={m} k={k} n={n}"),
            );

            // NT: 縮約軸は w_t の行（k 行）。
            let g2 = tensor(random_matrix(0x7200 + m as u64, m * k), &[m, k]);
            let w = tensor(random_matrix(0x7300 + n as u64, n * k), &[n, k]);
            let w_t = w.transpose_2d().unwrap();
            check_bias_reduce(
                &cuda_ops,
                &g2,
                &w_t,
                bias_first,
                &format!("NT m={m} k={k} n={n}"),
            );

            // NN（フォールバック）。
            let a = tensor(random_matrix(0x7400 + m as u64, m * k), &[m, k]);
            let b = tensor(random_matrix(0x7500 + n as u64, k * n), &[k, n]);
            check_bias_reduce(
                &cuda_ops,
                &a,
                &b,
                bias_first,
                &format!("NN m={m} k={k} n={n}"),
            );
        }
    }
}

/// 相殺列・`-0.0`・NaN・±inf を含む入力でホスト参照（f64 逐次和）と一致する
/// こと。f32 のみの縮約では `[1e8, 1.0, -1e8]` で 1.0 が失われる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_bias_reduce_special_values_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());

    // TN: 縮約軸 3 行の g（x_t は [2, 3]）。
    let g = tensor(
        vec![
            1e8,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            1.0,
            -0.0,
            1.0,
            f32::NEG_INFINITY,
            -1e8,
            -0.0,
            2.0,
            1.0,
        ],
        &[3, 4],
    );
    let x = tensor(random_matrix(0x8000, 3 * 2), &[3, 2]);
    let x_t = x.transpose_2d().unwrap();
    check_bias_reduce(&cuda_ops, &x_t, &g, false, "special TN");

    // 縮約軸長 1 の `-0.0` は直接コピーで保たれる。
    let g1 = tensor(vec![-0.0, 0.0, -0.0, 1.5], &[1, 4]);
    let x1 = tensor(random_matrix(0x8100, 2), &[1, 2]);
    let x1_t = x1.transpose_2d().unwrap();
    check_bias_reduce(&cuda_ops, &x1_t, &g1, true, "special TN k=1");
}

/// `bias = None` は従来どおり weight のみを書き `Ok(false)` を返し、範囲外
/// bias オフセットは weight も書かずに `InvalidArgument` で拒否すること。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_fp32_strict_into_bias_none_and_invalid_range_on_real_device() {
    use fandhe_ai_tensor_core::DispatchFailureCell;
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let mem = cuda_ops.memory_ops().expect("MemoryOps");
    let (m, k, n) = (4usize, 6usize, 5usize);
    let x = tensor(random_matrix(0x9000, m * k), &[m, k]);
    let x_t = x.transpose_2d().unwrap();
    let g = tensor(random_matrix(0x9100, m * n), &[m, n]);
    let token = DispatchFailureCell::default();

    let expected = cuda_ops.gemm_fp32_strict(&x_t, &g).unwrap();
    let mut out = mem.alloc_zeroed(&[k * n]).unwrap();
    let filled = cuda_ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(&x_t, &g, &mut out, 0, None, &token)
        .unwrap();
    assert!(!filled, "bias=None は Ok(false)");
    assert_eq!(
        contiguous_slice(&mem.download(&out).unwrap()),
        contiguous_slice(&expected)
    );

    let mut out2 = mem.alloc_zeroed(&[k * n + 2]).unwrap();
    let err = cuda_ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &x_t,
            &g,
            &mut out2,
            0,
            Some((k * n, n)), // bias 範囲が out を超える
            &token,
        )
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidArgument(_)), "{err:?}");
    assert!(
        contiguous_slice(&mem.download(&out2).unwrap())
            .iter()
            .all(|v| *v == 0.0),
        "Err のとき weight も書かれていないこと"
    );
}
