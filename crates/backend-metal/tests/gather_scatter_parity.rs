//! イシュー #1778: `BackendOps::gather`／`scatter`（`torch.gather`／
//! `torch.scatter`／`torch.scatter_add` 相当）の CPU-Metal 数値一致検証
//! （CUDA 側 #1777 の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`tests/where_masked_fill_parity.rs` と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される）。
//!
//! gather・scatter(Overwrite) は丸めを伴わないため CPU-Metal 間で
//! **bit 完全一致**を検証する。scatter(Add) は soft-f64 経路（本ファイル
//! 冒頭の設計に基づく `.claude/rules/coding-rust.md`「勾配の長軸縮約」節
//! と同じ精度規律）のためやはり bit 完全一致（NaN のみクラス一致）を
//! 検証する（`fandhe_ai_backend_metal::soft_f64::f32_bits_match`）。
//! Tape レベルの VJP 経路（`gather` の `d_input` が `scatter(Add)` 経由・
//! `scatter` の `d_src` が `gather` 経由）も CPU テープと bit 一致する
//! ことを確認する。
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
//! cargo test -p fandhe-ai-backend-metal --release --test gather_scatter_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_backend_metal::soft_f64::f32_bits_match;
use fandhe_ai_backend_metal::{MetalContext, MetalError, MetalGatherScatter};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScatterReduce, ShapeError, Tensor};

fn i32_index(seed: u64, numel: usize, dim_size: usize) -> Vec<i32> {
    Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| {
            let unit = (v + 1.0) / 2.0;
            let idx = (unit * dim_size as f32) as usize;
            idx.min(dim_size.saturating_sub(1)) as i32
        })
        .collect()
}

fn assert_gather_parity(in_shape: &[usize], index_shape: &[usize], dim: usize, seed: u64) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let in_numel: usize = in_shape.iter().product();
    let idx_numel: usize = index_shape.iter().product();
    let input =
        Tensor::new(Xorshift64Star::new(seed).fill_vec(in_numel), in_shape).expect("valid tensor");
    let index = Tensor::<i32>::new(
        i32_index(seed.wrapping_add(1), idx_numel, in_shape[dim]),
        index_shape,
    )
    .expect("valid index tensor");

    let cpu_out = cpu
        .gather(&input, dim, &index)
        .expect("cpu gather succeeds");
    let metal_out = metal
        .gather(&input, dim, &index)
        .expect("metal gather must succeed on Metal-equipped test runner");

    assert_eq!(cpu_out.shape(), metal_out.shape());
    for (a, b) in cpu_out
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(metal_out.as_slice().expect("contiguous").iter())
    {
        assert_eq!(a.to_bits(), b.to_bits(), "gather: CPU/Metal bit 不一致");
    }
}

fn assert_scatter_parity(
    out_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    reduce: ScatterReduce,
    seed: u64,
) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let out_numel: usize = out_shape.iter().product();
    let idx_numel: usize = index_shape.iter().product();
    let input = Tensor::new(Xorshift64Star::new(seed).fill_vec(out_numel), out_shape)
        .expect("valid tensor");
    let index = Tensor::<i32>::new(
        i32_index(seed.wrapping_add(1), idx_numel, out_shape[dim]),
        index_shape,
    )
    .expect("valid index tensor");
    let src = Tensor::new(
        Xorshift64Star::new(seed.wrapping_add(2)).fill_vec(idx_numel),
        index_shape,
    )
    .expect("valid src tensor");

    let cpu_out = cpu
        .scatter(&input, dim, &index, &src, reduce)
        .expect("cpu scatter succeeds");
    let metal_out = metal
        .scatter(&input, dim, &index, &src, reduce)
        .expect("metal scatter must succeed on Metal-equipped test runner");

    assert_eq!(cpu_out.shape(), metal_out.shape());
    for (a, b) in cpu_out
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(metal_out.as_slice().expect("contiguous").iter())
    {
        match reduce {
            ScatterReduce::Add => assert!(
                f32_bits_match(*b, *a),
                "scatter(Add): CPU/Metal 不一致: cpu={a} metal={b}"
            ),
            _ => assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "scatter(Overwrite): CPU/Metal bit 不一致"
            ),
        }
    }
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_parity_1d() {
    assert_gather_parity(&[5], &[8], 0, 1);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_parity_2d_each_dim() {
    assert_gather_parity(&[3, 4], &[3, 6], 1, 2);
    assert_gather_parity(&[3, 4], &[6, 4], 0, 3);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_parity_3d() {
    assert_gather_parity(&[2, 3, 4], &[2, 3, 7], 2, 4);
}

/// index_select（`Var::index_select` が委譲する gather の広義用例。
/// dim を除く各軸が 1 で broadcast された index を模した最小形）。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_parity_index_select_like() {
    assert_gather_parity(&[4, 5], &[4, 2], 1, 42);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn scatter_overwrite_parity_2d() {
    assert_scatter_parity(&[3, 4], &[3, 2], 1, ScatterReduce::Overwrite, 10);
    assert_scatter_parity(&[3, 4], &[2, 4], 0, ScatterReduce::Overwrite, 11);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn scatter_add_parity_2d() {
    assert_scatter_parity(&[3, 4], &[3, 2], 1, ScatterReduce::Add, 12);
    assert_scatter_parity(&[3, 4], &[2, 4], 0, ScatterReduce::Add, 13);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn scatter_parity_3d_add() {
    assert_scatter_parity(&[2, 3, 4], &[2, 3, 4], 2, ScatterReduce::Add, 14);
}

#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn scatter_parity_reduced_index_axis() {
    assert_scatter_parity(&[4, 4], &[2, 3], 1, ScatterReduce::Overwrite, 20);
    assert_scatter_parity(&[4, 4], &[2, 3], 1, ScatterReduce::Add, 21);
}

/// 相殺列（`.claude/rules/coding-rust.md`「勾配の長軸縮約」節・
/// `soft_f64.rs` モジュール doc 参照）を同一出力スロットへ集約した
/// scatter_add で CPU/Metal が bit 完全一致することを実機で確認する。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn scatter_add_parity_cancelling_sequence() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let out_shape = [1usize];
    let index_shape = [5usize];
    let input = Tensor::new(vec![0.0f32], &out_shape).unwrap();
    let index = Tensor::<i32>::new(vec![0i32; 5], &index_shape).unwrap();
    let src = Tensor::new(
        vec![
            2f32.powi(48),
            2f32.powi(24),
            1.0,
            -(2f32.powi(48)),
            -(2f32.powi(24)),
        ],
        &index_shape,
    )
    .unwrap();

    let cpu_out = cpu
        .scatter(&input, 0, &index, &src, ScatterReduce::Add)
        .unwrap();
    let metal_out = metal
        .scatter(&input, 0, &index, &src, ScatterReduce::Add)
        .unwrap();
    assert_eq!(
        cpu_out.as_slice().unwrap()[0].to_bits(),
        metal_out.as_slice().unwrap()[0].to_bits()
    );
    assert_eq!(metal_out.as_slice().unwrap()[0], 1.0);
}

/// エラー経路: shape 不一致は `ShapeMismatch`、範囲外 index も
/// `ShapeMismatch(IndexOutOfRange)`（`Unsupported` を返さずネイティブ
/// 経路へ到達していることの確認）。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_scatter_error_paths_reach_native_kernel() {
    let metal = MetalBackendOps::new();

    let input = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let bad_index = Tensor::<i32>::new(vec![0, 5, 2, 1], &[2, 2]).unwrap();
    let err = metal.gather(&input, 1, &bad_index).unwrap_err();
    assert!(
        matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::IndexOutOfRange { .. })
        ),
        "範囲外 index は IndexOutOfRange であるべき: {err:?}"
    );

    let mismatched_index = Tensor::<i32>::new(vec![0, 1, 2], &[3]).unwrap();
    let err = metal.gather(&input, 1, &mismatched_index).unwrap_err();
    assert!(
        matches!(err, BackendError::ShapeMismatch(_)),
        "shape 不一致は ShapeMismatch であるべき: {err:?}"
    );
}

/// Tape レベル: `Var::gather` → `backward()` の勾配（`scatter_add` 経由）
/// が CPU テープと bit 一致する。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_backward_matches_cpu_tape() {
    use fandhe_ai_autodiff::Tape;

    let in_shape = [3usize, 4usize];
    let index_shape = [3usize, 2usize];
    let input_data = Xorshift64Star::new(100).fill_vec(12);
    let index_data = i32_index(101, 6, 4);

    // `Var::gather` の VJP は `scatter(Add)` 経由（`crates/autodiff/src/
    // grad.rs::gather_with_fallback`／`scatter_with_fallback`）のため、
    // 本テストは gather forward・scatter_add backward の両カーネルを
    // 同時に検証する（`crates/autodiff/tests/backward.rs::
    // gather_backward_matches_numeric_with_duplicate_indices` と同型の
    // Tape 構成）。
    let run = |ops: Box<dyn BackendOps + Send>| -> Vec<f32> {
        let tape = Tape::new_with_ops(ops);
        let input = Tensor::new(input_data.clone(), &in_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), &index_shape).unwrap();
        let x = tape.var(&input);
        let out = x.gather(1, &index).expect("gather forward");
        let loss = out.sum(None).expect("sum");
        let grads = tape.backward(&loss).expect("backward succeeds");
        grads
            .get(&x)
            .expect("no error")
            .expect("x reaches loss")
            .as_slice()
            .expect("contiguous")
            .to_vec()
    };

    let cpu_grad = run(Box::new(CpuBackendOps::new()));
    let metal_grad = run(Box::new(MetalBackendOps::new()));

    assert_eq!(cpu_grad.len(), metal_grad.len());
    for (a, b) in cpu_grad.iter().zip(metal_grad.iter()) {
        assert!(
            f32_bits_match(*b, *a),
            "gather backward: CPU/Metal 勾配不一致: cpu={a} metal={b}"
        );
    }
}

/// イシュー #1799（codex-review・Cursor Bugbot 指摘）の回帰テスト:
/// `MetalGatherScatter::run_gather_f32`／`run_scatter_f32` は `pub` で
/// `ops.rs` を経由せず直接呼べるため、`ops.rs` 側の検査（`gather_out_shape`
/// ／`scatter_out_shape`・`validate_shapes_fit_u32`・`validate_index_range`）
/// を迂回した不正な入力（rank 不一致・`dim` 範囲外）を本関数自身が独立に
/// 拒否することを確認する（P0: shape 不一致による GPU バッファ範囲外
/// アクセス防止）。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_scatter_direct_call_rejects_rank_mismatch_and_bad_dim() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gs = MetalGatherScatter::new(&ctx).expect("gather/scatter カーネルのコンパイルに失敗した");

    // gather: in_shape と index_shape の rank が不一致（ops.rs の
    // `gather_out_shape` が通常拒否するが、直接呼び出しでは迂回できる）。
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let index = vec![0i32, 1, 0];
    let err = gs
        .run_gather_f32(&ctx, &input, &[2, 3], &index, &[3], 1)
        .expect_err("rank 不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "rank 不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );

    // gather: dim が rank 範囲外（カーネルが `shapes` バッファを
    // `in_shape[dim]` で読む前提が崩れ GPU 側範囲外読み出しになりうる）。
    let err = gs
        .run_gather_f32(&ctx, &input, &[2, 3], &index, &[2, 3], 5)
        .expect_err("dim 範囲外は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "dim 範囲外は InvalidGatherScatterShape であるべき: {err:?}"
    );

    // scatter も同じ独立検査を持つ（out_shape 側）。
    let src = vec![9.0f32, 9.0, 9.0];
    let err = gs
        .run_scatter_f32(
            &ctx,
            &input,
            &[2, 3],
            &index,
            &[3],
            &src,
            1,
            ScatterReduce::Overwrite,
        )
        .expect_err("scatter も rank 不一致を拒否するべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "scatter の rank 不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );
}

/// イシュー #1799（advisor 指摘）の回帰テスト: `in_shape`／`index_shape`
/// の rank・`dim` 自体は正しくても、非 `dim` 軸の次元が食い違う
/// （`in_shape=[2,3]`・`index_shape=[5,3]`・`dim=1`）gather を拒否する
/// ことを確認する。`rank`／`dim` のみの検査では見逃され、カーネルの
/// `gs_ravel(coords, in_shape, rank)` が `in_shape` の実バッファ長
/// （6 要素）を超えるオフセット（最大 `4*3+2=14`）を計算し GPU 側
/// バッファ範囲外読み出しになりうる。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_direct_call_rejects_non_dim_axis_mismatch() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gs = MetalGatherScatter::new(&ctx).expect("gather/scatter カーネルのコンパイルに失敗した");

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // index_shape[0]=5 > in_shape[0]=2（dim=1 のため axis 0 は非 dim）。
    let index = vec![0i32; 15];
    let err = gs
        .run_gather_f32(&ctx, &input, &[2, 3], &index, &[5, 3], 1)
        .expect_err("非 dim 軸の shape 不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "非 dim 軸の shape 不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );
}

/// イシュー #1799（codex-review P0 指摘・「スライス長」検証）の回帰
/// テスト: `index_shape` の要素数積（`numel`）と実際の `index` スライス
/// 長が食い違う直接呼び出しを拒否することを確認する（一致していれば
/// `MetalIndexBuffer` が渡されたスライス実長でバッファを確保する一方
/// カーネルは `numel` から導出した添字で読むため、実長の方が短い場合
/// GPU 側バッファ範囲外読み出しになりうる）。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_scatter_direct_call_rejects_slice_length_mismatch() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gs = MetalGatherScatter::new(&ctx).expect("gather/scatter カーネルのコンパイルに失敗した");

    // gather: index_shape=[2, 3]（numel=6）に対し index の実長が 3 のみ。
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let short_index = vec![0i32, 1, 0];
    let err = gs
        .run_gather_f32(&ctx, &input, &[2, 3], &short_index, &[2, 3], 1)
        .expect_err("index スライス長不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "index スライス長不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );

    // gather: in_shape=[2, 3]（in_numel=6）に対し input の実長が 3 のみ。
    let short_input = vec![1.0f32, 2.0, 3.0];
    let full_index = vec![0i32, 1, 2, 0, 1, 2];
    let err = gs
        .run_gather_f32(&ctx, &short_input, &[2, 3], &full_index, &[2, 3], 1)
        .expect_err("input スライス長不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "input スライス長不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );

    // scatter: out_shape=[2, 3]（numel_out=6）に対し input の実長が 3 のみ。
    let src = vec![9.0f32, 9.0, 9.0, 9.0, 9.0, 9.0];
    let err = gs
        .run_scatter_f32(
            &ctx,
            &short_input,
            &[2, 3],
            &full_index,
            &[2, 3],
            &src,
            1,
            ScatterReduce::Overwrite,
        )
        .expect_err("scatter の input スライス長不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "scatter の input スライス長不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );

    // scatter: index_shape=[2, 3]（idx_numel=6）に対し src の実長が 3 のみ。
    let short_src = vec![9.0f32, 9.0, 9.0];
    let err = gs
        .run_scatter_f32(
            &ctx,
            &input,
            &[2, 3],
            &full_index,
            &[2, 3],
            &short_src,
            1,
            ScatterReduce::Overwrite,
        )
        .expect_err("scatter の src スライス長不一致は拒否されるべき");
    assert!(
        matches!(err, MetalError::InvalidGatherScatterShape { .. }),
        "scatter の src スライス長不一致は InvalidGatherScatterShape であるべき: {err:?}"
    );
}

/// イシュー #1799（Cursor Bugbot Medium／codex-review P2 指摘）の回帰
/// テスト: `input` が非空でも `index`／`src`（`index_shape` の要素数積が
/// 0）が空の scatter は、`MetalIndexBuffer`／`MetalBuffer` の 0 バイト
/// 確保拒否（`ZeroLengthAllocation`）に落ちず、CPU 参照実装・ホスト
/// モデル（`gather_scatter_model::scatter_model`）と同じ `input` の
/// 完全なパススルーを返すことを確認する。
#[test]
#[ignore = "Apple Silicon 実機（Metal）が必要"]
fn gather_scatter_direct_call_empty_index_scatter_passes_through_input() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gs = MetalGatherScatter::new(&ctx).expect("gather/scatter カーネルのコンパイルに失敗した");

    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out_shape = [2usize, 3usize];
    // index_shape の非 dim 軸（axis 0）が 0 のため idx_numel = 0 だが
    // out_shape（＝ input.len()）は非空。
    let index_shape = [0usize, 3usize];
    let empty_index: Vec<i32> = Vec::new();
    let empty_src: Vec<f32> = Vec::new();

    for reduce in [ScatterReduce::Overwrite, ScatterReduce::Add] {
        let out = gs
            .run_scatter_f32(
                &ctx,
                &input,
                &out_shape,
                &empty_index,
                &index_shape,
                &empty_src,
                1,
                reduce,
            )
            .expect("空 index の scatter は ZeroLengthAllocation で失敗してはならない");
        assert_eq!(
            out, input,
            "空 index の scatter は input の完全なパススルーであるべき（reduce={reduce:?}）"
        );
    }
}
