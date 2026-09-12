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
//! **イシュー #1566 追加**: `MetalBackendOps::gemm_fp32_strict_into_
//! with_bias_reduce_tracked`（bias 勾配も同一ディスパッチで同時に
//! 計算する拡張版）の bit 完全一致・NaN 事前充填非破壊・境界検査を
//! 本ファイル末尾のテスト群で検証する。
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
use fandhe_ai_tensor_core::{BackendOps, DispatchFailureCell, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// 要素ごとの `to_bits()` 比較（bit 完全一致契約の検証。codex-review
/// 指摘・PR #1556: `assert_eq!(&[f32], &[f32])` は `+0.0`／`-0.0` を
/// 同一値として扱い区別できないため、符号付きゼロの取り違えを見逃す
/// おそれがある。`f32::to_bits()` はビット表現をそのまま比較するため
/// `+0.0`（`0x0000_0000`）と `-0.0`（`0x8000_0000`）を確実に区別する）。
fn assert_bits_eq(actual: &[f32], expected: &[f32], ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{ctx}: length mismatch (actual={} expected={})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{ctx}: element {i} bit mismatch (actual={a:?} bits={:#010x}, expected={e:?} \
             bits={:#010x})",
            a.to_bits(),
            e.to_bits(),
        );
    }
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

        assert_bits_eq(
            &readback_data[prefix..prefix + mn],
            expected_data,
            &format!(
                "gemm_fp32_strict_into は gemm_fp32_strict と bit 完全一致するはず \
                 （batch={batch} d_in={d_in} d_out={d_out} prefix={prefix}）"
            ),
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

    assert_bits_eq(
        &readback_data[prefix..prefix + mn],
        expected_data,
        "NN フォールバック経路は gemm_fp32_strict と bit 完全一致するはず",
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

    assert_bits_eq(
        readback_data,
        expected_data,
        "in_features=1 の縮退ケースも gemm_fp32_strict と bit 完全一致するはず",
    );
}

/// 符号付きゼロ（`-0.0`）を含むオペランドでも `gemm_fp32_strict_into` が
/// `gemm_fp32_strict` と bit 完全一致することを確認する（codex-review
/// 指摘・PR #1556）。`assert_eq!(&[f32], &[f32])` は `-0.0 == 0.0`
/// （IEEE 754 の等価性）のため符号の取り違えを検出できず、旧版の
/// テストは実際には符号ビットを検証していなかった。`assert_bits_eq`
/// （`to_bits()` 比較）へ置き換えたことで、本テストが実際に符号ビット
/// まで検証していることを NT/TN 経路・NN フォールバック経路の双方で
/// 確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_bit_matches_with_negative_zero_operands() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");

    // NT/TN 経路: `x`（batch=2, d_in=2）の一部要素を `-0.0` にしてから
    // 転置して `x_t` を作る。`g`（batch=2, d_out=2）にも `-0.0` を混ぜる。
    let x = tensor(vec![-0.0f32, 1.5, 2.5, -0.0], &[2, 2]);
    let x_t = x.transpose(0, 1).unwrap();
    let g = tensor(vec![0.0f32, -0.0, -3.0, 4.0], &[2, 2]);

    let expected_nt_tn = ops
        .gemm_fp32_strict(&x_t, &g)
        .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
    let expected_nt_tn_c = expected_nt_tn.contiguous();
    let expected_nt_tn_data = expected_nt_tn_c.as_slice().unwrap();

    let mn_nt_tn = 2 * 2;
    let seed_nt_tn = tensor(vec![f32::NAN; mn_nt_tn], &[mn_nt_tn]);
    let mut staging_nt_tn = mem.upload(&seed_nt_tn).unwrap();
    ops.gemm_fp32_strict_into(&x_t, &g, &mut staging_nt_tn, 0)
        .expect("gemm_fp32_strict_into must succeed for the NT/TN transposed-operand path");
    let readback_nt_tn = mem.download(&staging_nt_tn).unwrap();
    let readback_nt_tn_c = readback_nt_tn.contiguous();
    let readback_nt_tn_data = readback_nt_tn_c.as_slice().unwrap();
    assert_bits_eq(
        readback_nt_tn_data,
        expected_nt_tn_data,
        "NT/TN 経路（符号付きゼロ入力）は gemm_fp32_strict と bit 完全一致するはず",
    );

    // NN フォールバック経路: `a`（2x3）・`b`（3x4）の一部要素を `-0.0` に
    // する（NN は `layout::classify_2d` が両方 contiguous と分類する
    // ため `Unsupported` を返さずホスト経路フォールバックへ落ちる。
    // `ops::MetalBackendOps::gemm_fp32_strict_into` doc 参照）。
    let a = tensor(vec![-0.0f32, 1.0, -0.0, 2.0, -0.0, 3.0], &[2, 3]);
    let b = tensor(
        vec![
            0.0f32, -0.0, 1.0, -1.0, -0.0, 0.0, -0.0, 2.0, 1.0, -0.0, -2.0, 0.0,
        ],
        &[3, 4],
    );

    let expected_nn = ops
        .gemm_fp32_strict(&a, &b)
        .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
    let expected_nn_c = expected_nn.contiguous();
    let expected_nn_data = expected_nn_c.as_slice().unwrap();

    let mn_nn = 2 * 4;
    let seed_nn = tensor(vec![f32::NAN; mn_nn], &[mn_nn]);
    let mut staging_nn = mem.upload(&seed_nn).unwrap();
    ops.gemm_fp32_strict_into(&a, &b, &mut staging_nn, 0)
        .expect("gemm_fp32_strict_into must fall back to the host path for NN shapes");
    let readback_nn = mem.download(&staging_nn).unwrap();
    let readback_nn_c = readback_nn.contiguous();
    let readback_nn_data = readback_nn_c.as_slice().unwrap();
    assert_bits_eq(
        readback_nn_data,
        expected_nn_data,
        "NN フォールバック経路（符号付きゼロ入力）は gemm_fp32_strict と bit 完全一致するはず",
    );
}

// === イシュー #1566: gemm_fp32_strict_into_with_bias_reduce_tracked ===

/// NT/TN 経路（同一 `ctx.encode` 呼び出し内で weight・bias を同時に
/// 書く）で、weight は `gemm_fp32_strict_into` と、bias は
/// `layout::reduce_bias_grad_rows_host`（ホスト参照実装）と、それぞれ
/// bit 完全一致することを確認する。戻り値は `bias` を渡した場合
/// `Ok(true)` になるはず（`ops::MetalBackendOps::gemm_fp32_strict_into_
/// with_bias_reduce_tracked` doc「NT/TN」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_with_bias_reduce_tracked_matches_reference_for_nt_tn() {
    use fandhe_ai_backend_metal::layout::{MatrixLayout, reduce_bias_grad_rows_host};

    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let token = DispatchFailureCell::new();

    for &(batch, d_in, d_out) in &[(4usize, 8usize, 4usize), (37, 65, 33), (64, 129, 96)] {
        let (x_t, g) = transposed_operand_pair(
            0x3000 + d_in as u64,
            0x4000 + d_out as u64,
            batch,
            d_in,
            d_out,
        );

        let expected_weight = ops
            .gemm_fp32_strict(&x_t, &g)
            .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
        let expected_weight_c = expected_weight.contiguous();
        let expected_weight_data = expected_weight_c.as_slice().unwrap();

        let g_contiguous = g.contiguous();
        let g_slice = g_contiguous.as_slice().unwrap();
        let g_layout = MatrixLayout {
            rows: batch,
            cols: d_out,
            ld: d_out,
            transposed: false,
        };
        let expected_bias =
            reduce_bias_grad_rows_host(g_slice, &g_layout).expect("valid layout/data in test");

        let weight_mn = d_in * d_out;
        let bias_offset = weight_mn + 2; // gap を空け範囲混同がないことも確認する
        let total = bias_offset + d_out;
        let seed = tensor(vec![f32::NAN; total], &[total]);
        let mut staging = mem.upload(&seed).unwrap();

        let bias_written = ops
            .gemm_fp32_strict_into_with_bias_reduce_tracked(
                &x_t,
                &g,
                &mut staging,
                0,
                Some((bias_offset, d_out)),
                &token,
            )
            .expect("gemm_fp32_strict_into_with_bias_reduce_tracked must succeed for NT/TN");
        assert!(
            bias_written,
            "NT/TN 経路は bias を渡すと常に Ok(true) を返すはず（batch={batch} \
             d_in={d_in} d_out={d_out}）"
        );

        let readback = mem.download(&staging).unwrap();
        let readback_c = readback.contiguous();
        let readback_data = readback_c.as_slice().unwrap();

        assert_bits_eq(
            &readback_data[0..weight_mn],
            expected_weight_data,
            &format!(
                "weight 部分は gemm_fp32_strict と bit 完全一致するはず（batch={batch} \
                 d_in={d_in} d_out={d_out}）"
            ),
        );
        assert!(
            readback_data[weight_mn..bias_offset]
                .iter()
                .all(|v| v.is_nan()),
            "weight と bias の間の未使用領域（NaN 事前充填）が変更された（batch={batch} \
             d_in={d_in} d_out={d_out}）"
        );
        assert_bits_eq(
            &readback_data[bias_offset..bias_offset + d_out],
            &expected_bias,
            &format!(
                "bias 部分は reduce_bias_grad_rows_host（ホスト参照実装）と bit 完全一致 \
                 するはず（batch={batch} d_in={d_in} d_out={d_out}）"
            ),
        );
    }
}

/// NN フォールバック経路（`layout::classify_2d` が NT/TN と判定しない
/// 形状）でも、weight・bias とも成功しホスト参照実装と bit 完全一致する
/// ことを確認する（`ops::MetalBackendOps::gemm_fp32_strict_into_with_
/// bias_reduce_tracked` doc「NN/TT・分類不能形状」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_with_bias_reduce_tracked_matches_reference_for_nn_fallback() {
    use fandhe_ai_backend_metal::layout::{MatrixLayout, reduce_bias_grad_rows_host};

    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let token = DispatchFailureCell::new();

    let a = tensor(random_matrix(155, 2 * 3), &[2, 3]);
    let b = tensor(random_matrix(166, 3 * 4), &[3, 4]);

    let expected_weight = ops
        .gemm_fp32_strict(&a, &b)
        .expect("gemm_fp32_strict must succeed on a Metal-equipped test runner");
    let expected_weight_c = expected_weight.contiguous();
    let expected_weight_data = expected_weight_c.as_slice().unwrap();

    let b_contiguous = b.contiguous();
    let b_slice = b_contiguous.as_slice().unwrap();
    let b_layout = MatrixLayout {
        rows: 3,
        cols: 4,
        ld: 4,
        transposed: false,
    };
    let expected_bias =
        reduce_bias_grad_rows_host(b_slice, &b_layout).expect("valid layout/data in test");

    let weight_mn = 2 * 4;
    let bias_offset = weight_mn;
    let total = bias_offset + 4;
    let seed = tensor(vec![f32::NAN; total], &[total]);
    let mut staging = mem.upload(&seed).unwrap();

    let bias_written = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &a,
            &b,
            &mut staging,
            0,
            Some((bias_offset, 4)),
            &token,
        )
        .expect("gemm_fp32_strict_into_with_bias_reduce_tracked must fall back for NN shapes");
    assert!(
        bias_written,
        "NN フォールバック経路も bias を渡すと Ok(true) を返すはず（常に成功する契約）"
    );

    let readback = mem.download(&staging).unwrap();
    let readback_c = readback.contiguous();
    let readback_data = readback_c.as_slice().unwrap();

    assert_bits_eq(
        &readback_data[0..weight_mn],
        expected_weight_data,
        "NN フォールバック経路の weight は gemm_fp32_strict と bit 完全一致するはず",
    );
    assert_bits_eq(
        &readback_data[bias_offset..bias_offset + 4],
        &expected_bias,
        "NN フォールバック経路の bias は reduce_bias_grad_rows_host と bit 完全一致するはず",
    );
}

/// `bias` に `None` を渡した場合は既存の `gemm_fp32_strict_into_tracked`
/// と完全に同じ（weight のみ・`Ok(false)`）ことを確認する
/// （非破壊拡張契約）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_with_bias_reduce_tracked_none_bias_matches_weight_only_entry() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let token = DispatchFailureCell::new();

    let (x_t, g) = transposed_operand_pair(0x5000, 0x6000, 4, 8, 4);
    let mn = 8 * 4;

    let seed_a = tensor(vec![f32::NAN; mn], &[mn]);
    let mut staging_a = mem.upload(&seed_a).unwrap();
    ops.gemm_fp32_strict_into_tracked(&x_t, &g, &mut staging_a, 0, &token)
        .expect("gemm_fp32_strict_into_tracked must succeed");

    let seed_b = tensor(vec![f32::NAN; mn], &[mn]);
    let mut staging_b = mem.upload(&seed_b).unwrap();
    let bias_written = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(&x_t, &g, &mut staging_b, 0, None, &token)
        .expect("gemm_fp32_strict_into_with_bias_reduce_tracked must succeed with bias=None");
    assert!(!bias_written, "bias=None のときは常に Ok(false) のはず");

    let readback_a = mem.download(&staging_a).unwrap();
    let readback_a_c = readback_a.contiguous();
    let readback_b = mem.download(&staging_b).unwrap();
    let readback_b_c = readback_b.contiguous();
    assert_bits_eq(
        readback_b_c.as_slice().unwrap(),
        readback_a_c.as_slice().unwrap(),
        "bias=None の場合、gemm_fp32_strict_into_with_bias_reduce_tracked は \
         gemm_fp32_strict_into_tracked と bit 完全一致するはず",
    );
}

/// bias の書き込み範囲が `out` を超える場合は `InvalidArgument` で
/// 拒否される（REQ-8・OWASP A03。weight 側の範囲検査と同じ規約）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_fp32_strict_into_with_bias_reduce_tracked_rejects_out_of_range_bias_offset() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let token = DispatchFailureCell::new();

    let (x_t, g) = transposed_operand_pair(0x7000, 0x8000, 2, 2, 2);
    // weight は 4 要素・bias は 2 要素だが、バッファは weight 分しか
    // 確保しない（bias_offset=4 は範囲外）。
    let seed = tensor(vec![0.0f32; 4], &[4]);
    let mut staging = mem.upload(&seed).unwrap();

    let err = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &x_t,
            &g,
            &mut staging,
            0,
            Some((4, 2)),
            &token,
        )
        .unwrap_err();
    assert!(
        matches!(err, BackendError::InvalidArgument(_)),
        "範囲外の bias_offset は InvalidArgument であるべき: {err:?}"
    );
}
