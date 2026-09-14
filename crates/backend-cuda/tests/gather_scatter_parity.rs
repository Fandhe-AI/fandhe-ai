//! イシュー #1777: `BackendOps::gather`／`scatter`（`torch.gather`／
//! `torch.scatter`／`torch.scatter_add` 相当）の CPU-CUDA 数値一致検証。
//!
//! `where_masked_fill_parity.rs`（#1637）と同じ構成方針を踏襲する:
//! 環境適応スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ検証。
//! デバイス初期化より前に返る shape 検査経路は GPU 有無に依らず検証する）
//! と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を分離する。
//!
//! **契約は bit 同一**（`docs/spec` REQ-2 の丸め誤差許容ではなく、
//! `fandhe_ai_tensor_core::ScatterReduce` doc が定める決定的集約契約
//! そのものの検証）: gather は選択演算（丸めなし）、scatter_add は
//! `f64` アキュムレータで CPU と同一の走査順・同一の丸めを行うため、
//! CPU 参照実装（[`fandhe_ai_backend_cpu::CpuBackendOps`]）と CUDA 実装
//! の出力は run-to-run のみならず互いにも bit 完全一致する契約
//! （`docs/backend-cuda-gather-scatter` 系設計記録・`.claude/rules/
//! coding-rust.md`「勾配の長軸縮約」節）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gather_scatter_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, Tensor};

mod common;

/// `BackendError` は `PartialEq` を実装しないため、`ShapeMismatch` の
/// 内側 `ShapeError`（`PartialEq` 実装済み）だけを取り出して比較する。
fn expect_shape_mismatch(err: BackendError) -> fandhe_ai_tensor_core::ShapeError {
    match err {
        BackendError::ShapeMismatch(inner) => inner,
        other => panic!("expected BackendError::ShapeMismatch, got {other}"),
    }
}

/// `[0, dim_size)` の一様乱数 `i32` 添字列を決定的シードで作る。
fn index_vec(seed: u64, numel: usize, dim_size: usize) -> Vec<i32> {
    Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| {
            // `v` は `[-1, 1)`。`[0, dim_size)` へ写像する。
            let unit = (v + 1.0) / 2.0;
            ((unit * dim_size as f32) as i64)
                .clamp(0, dim_size as i64 - 1)
                .max(0) as i32
        })
        .collect()
}

fn assert_gather_parity(
    seed_in: u64,
    seed_idx: u64,
    in_shape: &[usize],
    dim: usize,
    out_shape: &[usize],
) {
    let numel_in: usize = in_shape.iter().product();
    let numel_out: usize = out_shape.iter().product();
    let dim_size = in_shape[dim];

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let input = Tensor::new(Xorshift64Star::new(seed_in).fill_vec(numel_in), in_shape)
        .expect("valid tensor");
    let index = Tensor::<i32>::new(index_vec(seed_idx, numel_out, dim_size), out_shape)
        .expect("valid tensor");

    let cpu_out = cpu
        .gather(&input, dim, &index)
        .expect("cpu gather always succeeds for valid input");
    let cuda_out = cuda
        .gather(&input, dim, &index)
        .expect("cuda gather must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cuda_slice, cpu_slice,
        "gather: 選択演算は丸めを伴わないため bit 同一のはず \
         (in_shape={in_shape:?}, dim={dim}, out_shape={out_shape:?})"
    );
    // run-to-run 決定性: 同一入力で 2 回起動しても bit 同一。
    let cuda_out2 = cuda.gather(&input, dim, &index).expect("cuda gather rerun");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "gather: run-to-run で bit 同一のはず"
    );
}

fn assert_scatter_parity(
    seed_in: u64,
    seed_idx: u64,
    seed_src: u64,
    in_shape: &[usize],
    dim: usize,
    index_shape: &[usize],
    reduce: ScatterReduce,
) {
    let numel_in: usize = in_shape.iter().product();
    let numel_index: usize = index_shape.iter().product();
    let dim_size = in_shape[dim];

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let input = Tensor::new(Xorshift64Star::new(seed_in).fill_vec(numel_in), in_shape)
        .expect("valid tensor");
    let index = Tensor::<i32>::new(index_vec(seed_idx, numel_index, dim_size), index_shape)
        .expect("valid tensor");
    let src = Tensor::new(
        Xorshift64Star::new(seed_src).fill_vec(numel_index),
        index_shape,
    )
    .expect("valid tensor");

    let cpu_out = cpu
        .scatter(&input, dim, &index, &src, reduce)
        .expect("cpu scatter always succeeds for valid input");
    let cuda_out = cuda
        .scatter(&input, dim, &index, &src, reduce)
        .expect("cuda scatter must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cuda_slice, cpu_slice,
        "scatter({reduce:?}): CPU と bit 同一のはず \
         (in_shape={in_shape:?}, dim={dim}, index_shape={index_shape:?})"
    );
    let cuda_out2 = cuda
        .scatter(&input, dim, &index, &src, reduce)
        .expect("cuda scatter rerun");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "scatter({reduce:?}): run-to-run で bit 同一のはず"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`where_masked_fill_parity_smoke_env_adaptive` と同じ分岐パターン）。
/// デバイス初期化より前に返る shape 検査経路（`gather_out_shape`／
/// `scatter_out_shape` の再検査・index 値の範囲外検査）は GPU 有無に
/// 依らず検証する。
#[test]
fn gather_scatter_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");
    let index = Tensor::<i32>::new(vec![0, 2, 2, 1], &[2, 2]).expect("valid tensor");

    match cuda.gather(&input, 1, &index) {
        Ok(_) => {
            assert_gather_parity(9001, 9002, &[2, 3], 1, &[2, 2]);
            assert_gather_parity(9003, 9004, &[3, 4], 0, &[2, 4]);
            assert_scatter_parity(
                9005,
                9006,
                9007,
                &[2, 3],
                1,
                &[2, 2],
                ScatterReduce::Overwrite,
            );
            assert_scatter_parity(9008, 9009, 9010, &[2, 3], 1, &[2, 2], ScatterReduce::Add);

            // 重複集中: 全 index が同一位置を指す（Add の逐次加算契約を
            // 実機でも確認する）。
            let input2 = Tensor::new(vec![10.0, 0.0, 0.0], &[1, 3]).expect("valid tensor");
            let index2 = Tensor::<i32>::new(vec![0, 0, 0], &[1, 3]).expect("valid tensor");
            let src2 = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).expect("valid tensor");
            let cpu_dup = cpu
                .scatter(&input2, 1, &index2, &src2, ScatterReduce::Add)
                .expect("cpu scatter_add");
            let cuda_dup = cuda
                .scatter(&input2, 1, &index2, &src2, ScatterReduce::Add)
                .expect("cuda scatter_add");
            assert_eq!(
                cuda_dup.as_slice().expect("contiguous"),
                cpu_dup.as_slice().expect("contiguous")
            );

            // shape 不一致は `BackendError::ShapeMismatch` を返す
            // （実装側の再検査。`.claude/rules/security.md` A08）。
            let bad_index = Tensor::<i32>::new(vec![0, 1], &[2]).expect("valid tensor");
            let err = cuda
                .gather(&input, 1, &bad_index)
                .expect_err("rank mismatch must be rejected");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            // 範囲外添字は `BackendError::ShapeMismatch(ShapeError::
            // IndexOutOfRange)` を返す（CPU と同一 variant・フィールド。
            // `Var` を経由しない直接呼び出しでも独立に検査する）。
            let bad_scatter_index =
                Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let src = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
            let cpu_err = expect_shape_mismatch(
                cpu.scatter(
                    &input,
                    1,
                    &bad_scatter_index,
                    &src,
                    ScatterReduce::Overwrite,
                )
                .expect_err("cpu must reject out-of-range index"),
            );
            let cuda_err = expect_shape_mismatch(
                cuda.scatter(
                    &input,
                    1,
                    &bad_scatter_index,
                    &src,
                    ScatterReduce::Overwrite,
                )
                .expect_err("cuda must reject out-of-range index"),
            );
            assert_eq!(
                cpu_err, cuda_err,
                "CPU と CUDA は同一の IndexOutOfRange を返す契約"
            );

            // gather も scatter と同じ独立検査を持つ（イシュー #1777
            // codex-review 指摘: 当初 `gather` にはこの検査が欠落しており
            // 範囲外 index がカーネル側フォールバックで `0.0` を書いて
            // `Ok` を返す silent data corruption になっていた）。
            let bad_gather_index =
                Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let cpu_gather_err = expect_shape_mismatch(
                cpu.gather(&input, 1, &bad_gather_index)
                    .expect_err("cpu must reject out-of-range index"),
            );
            let cuda_gather_err = expect_shape_mismatch(
                cuda.gather(&input, 1, &bad_gather_index)
                    .expect_err("cuda gather must reject out-of-range index"),
            );
            assert_eq!(
                cpu_gather_err, cuda_gather_err,
                "gather も CPU と CUDA で同一の IndexOutOfRange を返す契約"
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");

            // shape 検査はデバイス初期化より前に走るため CUDA 非搭載
            // 環境でも検証できる（CPU の返す値と一致することも確認）。
            let bad_index = Tensor::<i32>::new(vec![0, 1], &[2]).expect("valid tensor");
            let err = cuda
                .gather(&input, 1, &bad_index)
                .expect_err("rank mismatch must be rejected even without CUDA");
            assert!(matches!(err, BackendError::ShapeMismatch(_)));

            let bad_scatter_index =
                Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let src = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
            let cpu_err = expect_shape_mismatch(
                cpu.scatter(
                    &input,
                    1,
                    &bad_scatter_index,
                    &src,
                    ScatterReduce::Overwrite,
                )
                .expect_err("cpu must reject out-of-range index"),
            );
            let cuda_err = expect_shape_mismatch(
                cuda.scatter(
                    &input,
                    1,
                    &bad_scatter_index,
                    &src,
                    ScatterReduce::Overwrite,
                )
                .expect_err("cuda must reject out-of-range index even without CUDA device"),
            );
            assert_eq!(cpu_err, cuda_err);

            // gather の範囲外 index 検査もデバイス初期化より前に走るため
            // CUDA 非搭載環境でも検証できる（イシュー #1777 codex-review
            // 指摘の是正）。
            let bad_gather_index =
                Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let cpu_gather_err = expect_shape_mismatch(
                cpu.gather(&input, 1, &bad_gather_index)
                    .expect_err("cpu must reject out-of-range index"),
            );
            let cuda_gather_err =
                expect_shape_mismatch(cuda.gather(&input, 1, &bad_gather_index).expect_err(
                    "cuda gather must reject out-of-range index even without CUDA device",
                ));
            assert_eq!(cpu_gather_err, cuda_gather_err);
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::gather: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gather_matches_cpu_across_shapes() {
    let shapes_dims: &[(&[usize], usize)] = &[
        (&[4], 0),
        (&[2, 3], 0),
        (&[2, 3], 1),
        (&[2, 3, 4], 1),
        (&[2, 3, 4], 2),
        (&[1 << 12, 3], 1), // ブロック境界をまたぐ大きさ
    ];
    let mut seed = 10_000u64;
    for &(in_shape, dim) in shapes_dims {
        seed += 5;
        // index_shape[dim] が input より大きい（重複読み出し）／小さい
        // 両方を網羅する。
        for &grow in &[0i64, 3, -1] {
            let mut out_shape: Vec<usize> = in_shape.to_vec();
            let new_dim = (in_shape[dim] as i64 + grow).max(1) as usize;
            out_shape[dim] = new_dim;
            seed += 3;
            assert_gather_parity(seed, seed + 1, in_shape, dim, &out_shape);
        }
    }

    // 非 contiguous な input（transpose view）を渡しても contiguous 化後
    // に一致する。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let base = Tensor::new((0..12).map(|v| v as f32).collect(), &[3, 4]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("valid transpose");
    // `transposed` の shape は `[4, 3]`（`base` の `[3, 4]` を転置）のため、
    // dim=1 の有効添字範囲は `0..=2`（PR #1795 codex-review 指摘の是正・
    // イシュー #1777）。添字 `3` は範囲外で CPU 側が `IndexOutOfRange` を
    // 返してしまい、意図していた非 contiguous 入力での CUDA/CPU 比較・
    // 特殊値検証に到達できていなかった。
    let index = Tensor::<i32>::new(vec![0, 1, 2, 0, 1, 2, 0, 1], &[4, 2]).expect("valid tensor");
    let cpu_out = cpu.gather(&transposed, 1, &index).expect("cpu gather");
    let cuda_out = cuda.gather(&transposed, 1, &index).expect("cuda gather");
    assert_eq!(
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );

    // NaN／±inf の通過。
    let input_special = Tensor::new(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[2, 2],
    )
    .expect("valid tensor");
    let index_special = Tensor::<i32>::new(vec![0, 1, 1, 0], &[2, 2]).expect("valid tensor");
    let cpu_special = cpu
        .gather(&input_special, 1, &index_special)
        .expect("cpu gather");
    let cuda_special = cuda
        .gather(&input_special, 1, &index_special)
        .expect("cuda gather");
    let cpu_special_slice = cpu_special.as_slice().expect("contiguous");
    let cuda_special_slice = cuda_special.as_slice().expect("contiguous");
    for (c, g) in cpu_special_slice.iter().zip(cuda_special_slice.iter()) {
        if c.is_nan() {
            assert!(g.is_nan(), "NaN must pass through gather unchanged");
        } else {
            assert_eq!(c, g);
        }
    }
}

/// 実機必須の形状網羅（scatter/scatter_add。受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn scatter_matches_cpu_across_shapes_and_reduce() {
    let cases: &[(&[usize], usize, &[usize])] = &[
        (&[4], 0, &[4]),
        (&[2, 3], 0, &[2, 3]),
        (&[2, 3], 1, &[2, 3]),
        (&[2, 3, 4], 1, &[2, 3, 4]),
        // 非 dim 軸で index_shape < input_shape（未走査位置は input のまま
        // 残る）。
        (&[3, 4], 1, &[2, 4]),
        (&[3, 4], 0, &[2, 4]),
        (&[1 << 12, 3], 0, &[1 << 12, 3]), // ブロック境界をまたぐ大きさ
    ];
    let mut seed = 20_000u64;
    for &(in_shape, dim, index_shape) in cases {
        for &reduce in &[ScatterReduce::Overwrite, ScatterReduce::Add] {
            seed += 7;
            assert_scatter_parity(seed, seed + 1, seed + 2, in_shape, dim, index_shape, reduce);
        }
    }

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    // 相殺しやすい値列（f32 逐次和と f64 逐次和が実際に異なる系列）。
    let input = Tensor::new(vec![0.0, 0.0, 0.0], &[1, 3]).expect("valid tensor");
    let index = Tensor::<i32>::new(vec![0, 0, 0, 0, 0], &[1, 5]).expect("valid tensor");
    let src = Tensor::new(vec![1e8, 1.0, -1e8, 1.0, 1.0], &[1, 5]).expect("valid tensor");
    let cpu_out = cpu
        .scatter(&input, 1, &index, &src, ScatterReduce::Add)
        .expect("cpu scatter_add");
    let cuda_out = cuda
        .scatter(&input, 1, &index, &src, ScatterReduce::Add)
        .expect("cuda scatter_add");
    assert_eq!(
        cuda_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous"),
        "相殺列は f64 アキュムレータでなければ CPU と一致しないはず"
    );

    // NaN／±inf の通過（Overwrite）。
    let input_special = Tensor::new(vec![0.0, 0.0], &[1, 2]).expect("valid tensor");
    let index_special = Tensor::<i32>::new(vec![0, 1], &[1, 2]).expect("valid tensor");
    let src_special = Tensor::new(vec![f32::NAN, f32::INFINITY], &[1, 2]).expect("valid tensor");
    let cpu_special = cpu
        .scatter(
            &input_special,
            1,
            &index_special,
            &src_special,
            ScatterReduce::Overwrite,
        )
        .expect("cpu scatter overwrite");
    let cuda_special = cuda
        .scatter(
            &input_special,
            1,
            &index_special,
            &src_special,
            ScatterReduce::Overwrite,
        )
        .expect("cuda scatter overwrite");
    let cpu_slice = cpu_special.as_slice().expect("contiguous");
    let cuda_slice = cuda_special.as_slice().expect("contiguous");
    assert!(cpu_slice[0].is_nan() && cuda_slice[0].is_nan());
    assert_eq!(cpu_slice[1], cuda_slice[1]);

    // 空ケース: index が空（= input のコピー）。
    let input_empty = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).expect("valid tensor");
    let index_empty = Tensor::<i32>::new(Vec::new(), &[1, 0]).expect("valid tensor");
    let src_empty = Tensor::new(Vec::new(), &[1, 0]).expect("valid tensor");
    let cpu_empty = cpu
        .scatter(
            &input_empty,
            1,
            &index_empty,
            &src_empty,
            ScatterReduce::Overwrite,
        )
        .expect("cpu scatter empty index");
    let cuda_empty = cuda
        .scatter(
            &input_empty,
            1,
            &index_empty,
            &src_empty,
            ScatterReduce::Overwrite,
        )
        .expect("cuda scatter empty index");
    assert_eq!(
        cuda_empty.as_slice().expect("contiguous"),
        cpu_empty.as_slice().expect("contiguous")
    );
}

// --- one_hot（非微分演算。イシュー #1755） ---

/// `[0, num_classes)` の一様乱数 `i32` クラス id 列を決定的シードで作る
/// （`index_vec` と同じ写像だが `dim_size` を `num_classes` と呼び直した
/// だけの独立関数。呼び出し意図を明確にするため複製する）。
fn class_id_vec(seed: u64, numel: usize, num_classes: usize) -> Vec<i32> {
    index_vec(seed, numel, num_classes)
}

/// `CPU`（[`fandhe_ai_backend_cpu::CpuBackendOps`]）と `CUDA` の
/// `one_hot` 出力が bit 完全一致し、run-to-run でも bit 同一
/// （決定的）であることを確認する。
fn assert_one_hot_parity(seed: u64, index_shape: &[usize], num_classes: usize) {
    let numel: usize = index_shape.iter().product();
    let index = Tensor::<i32>::new(class_id_vec(seed, numel, num_classes), index_shape)
        .expect("valid tensor");

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let cpu_out = cpu
        .one_hot(&index, num_classes)
        .expect("cpu one_hot must succeed");
    let cuda_out = cuda
        .one_hot(&index, num_classes)
        .expect("cuda one_hot must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cuda_slice, cpu_slice,
        "one_hot: CPU と bit 同一のはず (index_shape={index_shape:?}, num_classes={num_classes})"
    );

    let cuda_out2 = cuda
        .one_hot(&index, num_classes)
        .expect("cuda one_hot rerun");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "one_hot: run-to-run で bit 同一のはず"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`gather_scatter_
/// parity_smoke_env_adaptive` と同じ分岐パターン: CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する。範囲外
/// クラス id 検査（`checked_shape_numel`・値域検査）はデバイス初期化
/// より前に走るため GPU 有無に依らず検証する。
#[test]
fn one_hot_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let index = Tensor::<i32>::new(vec![0, 2, 1, 1], &[2, 2]).expect("valid tensor");

    match cuda.one_hot(&index, 3) {
        Ok(_) => {
            assert_one_hot_parity(20001, &[2, 2], 3);
            assert_one_hot_parity(20002, &[5], 4);

            // 範囲外クラス id は `BackendError::ShapeMismatch
            // (ShapeError::IndexOutOfRange)` を返す（CPU と同一
            // variant・フィールド）。
            let bad_index = Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let cpu_err = expect_shape_mismatch(
                cpu.one_hot(&bad_index, 3)
                    .expect_err("cpu must reject out-of-range class id"),
            );
            let cuda_err = expect_shape_mismatch(
                cuda.one_hot(&bad_index, 3)
                    .expect_err("cuda must reject out-of-range class id"),
            );
            assert_eq!(
                cpu_err, cuda_err,
                "CPU と CUDA は同一の IndexOutOfRange を返す契約"
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");

            // 範囲外検査はデバイス初期化より前に走るため CUDA 非搭載
            // 環境でも検証できる。
            let bad_index = Tensor::<i32>::new(vec![0, 9, 2, 1], &[2, 2]).expect("valid tensor");
            let cpu_err = expect_shape_mismatch(
                cpu.one_hot(&bad_index, 3)
                    .expect_err("cpu must reject out-of-range class id"),
            );
            let cuda_err = expect_shape_mismatch(
                cuda.one_hot(&bad_index, 3)
                    .expect_err("cuda must reject out-of-range class id even without CUDA device"),
            );
            assert_eq!(cpu_err, cuda_err);
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::one_hot: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn one_hot_matches_cpu_across_shapes() {
    let index_shapes_classes: &[(&[usize], usize)] = &[
        (&[4], 3),
        (&[2, 3], 5),
        (&[2, 3, 4], 2),
        (&[1 << 12], 8), // ブロック境界をまたぐ大きさ
    ];
    let mut seed = 30_000u64;
    for &(index_shape, num_classes) in index_shapes_classes {
        seed += 7;
        assert_one_hot_parity(seed, index_shape, num_classes);
    }

    // 空ケース: index が空（出力も空）。
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let index_empty = Tensor::<i32>::new(Vec::new(), &[0]).expect("valid tensor");
    let cpu_empty = cpu
        .one_hot(&index_empty, 3)
        .expect("cpu one_hot empty index");
    let cuda_empty = cuda
        .one_hot(&index_empty, 3)
        .expect("cuda one_hot empty index");
    assert_eq!(
        cuda_empty.as_slice().expect("contiguous"),
        cpu_empty.as_slice().expect("contiguous")
    );
}
