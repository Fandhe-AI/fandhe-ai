//! `MetalBackendOps::gemm_fp32_strict_into`（`BackendOps::
//! gemm_fp32_strict_into` の Metal 実装。イシュー #1555・
//! `docs/device-resident-update-design.md` 追補）の実機テスト。
//!
//! `crates/backend-cpu/tests/gemm_into_parity.rs`（イシュー #1212）と同型
//! の受け入れ基準（`gemm_fp32_strict` との bit 完全一致・NaN 事前充填
//! 領域の非破壊・範囲外オフセットの拒否）に加え、Metal 実装固有の契約
//! （NT/TN は encode-only 直接書き込み・NN/TT/分類不能形状はホスト経路
//! フォールバックで成功〈codex-review 指摘・PR #1556〉・デバイス不一致の
//! 拒否）を検証する。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_fp32_strict_into_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::buffer::MemoryOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// `Op::LinearResident` の VJP が実際に渡す形（`x_t = x.transpose(0, 1)`・
/// `g` はそのまま）を再現する transposed-operand ペアを作る。`x_t` は
/// `layout::classify_2d` が `transposed: true` と分類する view（TN 側）、
/// `g` は行優先 contiguous（`transposed: false`。NT 側）であり、
/// `MetalBackendOps::gemm`（イシュー #1215）と同じ NT/TN 判定条件を満たす。
fn transposed_operand_pair(
    seed_x: u64,
    seed_g: u64,
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> (Tensor<f32>, Tensor<f32>) {
    let x = tensor(random_matrix(seed_x, batch * d_in), &[batch, d_in]);
    let x_t = x.transpose(0, 1).unwrap();
    let g = tensor(random_matrix(seed_g, batch * d_out), &[batch, d_out]);
    (x_t, g)
}

/// `gemm_fp32_strict_into` が `out_offset` から書き込む結果は、同じ
/// `x_t`／`g`（`d_weight = x_t @ g` を模した NT/TN 形状）に対する
/// `gemm_fp32_strict` と bit 完全一致する。NaN で事前充填した永続バッファ
/// に対しても、対象範囲以外は変更されない（`docs/perf/
/// train-resident-grad-device-update.md` の NaN 事前充填契約と同型）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_matches_gemm_fp32_strict_with_offset() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");

    for &(batch, d_in, d_out, prefix, suffix) in &[
        // `batch=d_in=d_out=1` は `layout::classify_2d` が「行優先」と
        // 「列優先」を区別できない縮退ケース（両方とも `sc == 1 && sr >=
        // cols` の行優先分岐で一致してしまい、`transposed` が常に
        // `false` になる）ため対象外とする（NT/TN 判定不成立で本経路に
        // 到達しない。実装のバグではなく `classify_2d` の既存契約
        // 〈`layout.rs` doc コメント〉どおりの挙動）。
        (2usize, 3usize, 2usize, 0usize, 0usize),
        (4, 8, 4, 3, 5),
        (37, 65, 33, 11, 0),
        (64, 129, 96, 0, 7),
        (5, 1, 130, 1, 1),
    ] {
        let (x_t, g) = transposed_operand_pair(
            0x1000 + d_in as u64,
            0x2000 + d_out as u64,
            batch,
            d_in,
            d_out,
        );

        let expected = ops
            .gemm_fp32_strict(&x_t, &g)
            .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
        let expected_c = expected.contiguous();
        let expected_data = expected_c.as_slice().unwrap();

        let mn = d_in * d_out;
        let total = prefix + mn + suffix;
        let seed = tensor(vec![f32::NAN; total], &[total]);
        let mut staging = mem.upload(&seed).unwrap();

        ops.gemm_fp32_strict_into(&x_t, &g, &mut staging, prefix)
            .expect("gemm_fp32_strict_into must succeed for the NT/TN transposed-operand path");

        let readback = mem.download(&staging).unwrap();
        let readback_c = readback.contiguous();
        let readback_data = readback_c.as_slice().unwrap();

        assert_eq!(
            &readback_data[prefix..prefix + mn],
            expected_data,
            "gemm_fp32_strict_into は gemm_fp32_strict と bit 完全一致するはず（batch={batch} \
             d_in={d_in} d_out={d_out} prefix={prefix}）"
        );
        assert!(
            readback_data[..prefix].iter().all(|v| v.is_nan()),
            "対象範囲より前の NaN 事前充填領域が変更された（batch={batch} d_in={d_in} \
             d_out={d_out}）"
        );
        assert!(
            readback_data[prefix + mn..].iter().all(|v| v.is_nan()),
            "対象範囲より後の NaN 事前充填領域が変更された（batch={batch} d_in={d_in} \
             d_out={d_out}）"
        );
    }
}

/// 範囲外書き込み（`out_offset + m*n > out.numel()`）は `InvalidArgument`
/// で拒否される（REQ-8「カーネル側の手動境界チェックを省略しない」・
/// OWASP A03）。NT/TN 判定より前に検査されることを確認する
/// （`ops::MetalBackendOps::gemm_fp32_strict_into` doc「境界検査の順序」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_rejects_out_of_range_offset() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let (x_t, g) = transposed_operand_pair(11, 22, 2, 2, 2);

    // 出力は 2x2=4 要素分ちょうどしかないため offset=1 は範囲外になる。
    let seed = tensor(vec![0.0f32; 4], &[4]);
    let mut staging = mem.upload(&seed).unwrap();

    let err = ops
        .gemm_fp32_strict_into(&x_t, &g, &mut staging, 1)
        .unwrap_err();
    assert!(
        matches!(err, BackendError::InvalidArgument(_)),
        "範囲外オフセットは InvalidArgument であるべき: {err:?}"
    );
}

/// `out.device()` が Metal でない場合は `DeviceMismatch` で拒否される。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_rejects_device_mismatch() {
    let ops = MetalBackendOps::new();
    let (x_t, g) = transposed_operand_pair(33, 44, 2, 2, 2);

    let cpu_mem = fandhe_ai_backend_cpu::CpuMemory::new();
    let seed = tensor(vec![0.0f32; 4], &[4]);
    let mut cpu_staging = cpu_mem.upload(&seed).unwrap();

    let err = ops
        .gemm_fp32_strict_into(&x_t, &g, &mut cpu_staging, 0)
        .unwrap_err();
    assert!(
        matches!(err, BackendError::DeviceMismatch),
        "Metal 以外のデバイスバッファは DeviceMismatch であるべき: {err:?}"
    );
}

/// NN（両オペランドとも行優先 contiguous。`layout::classify_2d` の
/// NT/TN 判定条件を満たさない形状）は `Unsupported` を返さず、ホスト経路
/// `gemm_fp32_strict` の結果を [`MemoryOps::upload_into`] で書き込む
/// フォールバックにより成功する（codex-review 指摘・PR #1556。
/// `ops::MetalBackendOps::gemm_fp32_strict_into` doc「`Unsupported` を
/// 返さない理由」）。`Unsupported` を返すと呼び出し元
/// `DeviceParamStore::fill_resident_weight_grad` がストア全体の
/// `resident_grad_capability` を `Some(false)` に確定させてしまい、
/// 同一 backward 内で先に成功済みの resident slot を読めなくする
/// （`crates/facade/tests/device_param_store_metal_mixed_shape_grad.rs`
/// の回帰テストが実例を検証する）。NT/TN 経路と同じく `gemm_fp32_strict`
/// と bit 完全一致し、範囲外の NaN 事前充填領域も変更されない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_falls_back_to_host_path_for_nn_shape() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");

    let a = tensor(random_matrix(55, 2 * 3), &[2, 3]);
    let b = tensor(random_matrix(66, 3 * 4), &[3, 4]);

    let expected = ops
        .gemm_fp32_strict(&a, &b)
        .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
    let expected_c = expected.contiguous();
    let expected_data = expected_c.as_slice().unwrap();

    let prefix = 2usize;
    let mn = 2 * 4;
    let suffix = 3usize;
    let total = prefix + mn + suffix;
    let seed = tensor(vec![f32::NAN; total], &[total]);
    let mut staging = mem.upload(&seed).unwrap();

    ops.gemm_fp32_strict_into(&a, &b, &mut staging, prefix)
        .expect("gemm_fp32_strict_into must fall back to the host path for NN shapes");

    let readback = mem.download(&staging).unwrap();
    let readback_c = readback.contiguous();
    let readback_data = readback_c.as_slice().unwrap();

    assert_eq!(
        &readback_data[prefix..prefix + mn],
        expected_data,
        "NN フォールバック経路は gemm_fp32_strict と bit 完全一致するはず"
    );
    assert!(
        readback_data[..prefix].iter().all(|v| v.is_nan()),
        "対象範囲より前の NaN 事前充填領域が変更された"
    );
    assert!(
        readback_data[prefix + mn..].iter().all(|v| v.is_nan()),
        "対象範囲より後の NaN 事前充填領域が変更された"
    );
}

/// `x_t` の strides が `[1, 1]` になる `in_features=1` の縮退ケース
/// （codex-review 指摘の実例そのもの。`Linear(1, 8)` の d_weight）は
/// `layout::classify_2d` が NN と分類し本メソッド旧実装では
/// `Unsupported` になっていたが、NN フォールバックにより成功する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_falls_back_for_in_features_one_transpose() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");

    // `Linear(1, 8)` の d_weight = x_t @ g 相当: x: [batch, 1] → x_t:
    // [1, batch] だが元の stride は [1, 1]（`in_features=1`）であり、
    // `classify_2d` は「行優先」と「列優先」を区別できず NN 扱いになる
    // （既存の `[batch=1, d_in=1, ...]` 除外ケースと同種の縮退）。
    let batch = 4usize;
    let d_out = 8usize;
    let x_t = tensor(random_matrix(77, batch), &[1, batch]);
    let g = tensor(random_matrix(88, batch * d_out), &[batch, d_out]);

    let expected = ops
        .gemm_fp32_strict(&x_t, &g)
        .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
    let expected_c = expected.contiguous();
    let expected_data = expected_c.as_slice().unwrap();

    let mn = d_out;
    let seed = tensor(vec![f32::NAN; mn], &[mn]);
    let mut staging = mem.upload(&seed).unwrap();

    ops.gemm_fp32_strict_into(&x_t, &g, &mut staging, 0)
        .expect("gemm_fp32_strict_into must fall back to the host path for this shape");

    let readback = mem.download(&staging).unwrap();
    let readback_c = readback.contiguous();
    let readback_data = readback_c.as_slice().unwrap();

    assert_eq!(
        readback_data, expected_data,
        "in_features=1 の縮退ケースも gemm_fp32_strict と bit 完全一致するはず"
    );
}
