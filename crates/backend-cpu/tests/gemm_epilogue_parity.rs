//! `backend-cpu::gemm_blis::gemm_blis_bias_act_parallel` の受け入れ基準
//! 対応テスト（TASK-12.1f・#203）。
//!
//! 受け入れ条件は「Linear+bias+ReLU 相当で非融合比の性能向上を実測
//! （性能実測は `tests/gemm_epilogue_perf.rs`）・数値一致維持」。本ファイルは
//! 数値一致側を担当する。
//!
//! 1. 手計算の既知値で正しさの基準を確認し、
//! 2. 融合版（`gemm_blis_bias_act_parallel`）と非融合合成
//!    （`gemm_blis_parallel` → bias 行加算 → `relu`。逐次参照実装）が
//!    MR/NR/MC/KC/NC 境界を跨ぐ形状グリッドで **bit 完全一致**することを
//!    確認し（epilogue は要素ごとに独立な演算で演算順序に依存しないため。
//!    `src/gemm_blis/mod.rs` の `gemm_blis_bias_act_parallel` ドキュメント
//!    コメント参照）、
//! 3. REQ-2 統一複合判定（`backend_cpu::parity::assert_parity`）でも
//!    重ねて確認し、
//! 4. エッジケース（`bias=None`・`act=None`・`m/n/k=0`・bias 長不一致
//!    エラー・非 contiguous 入力〈`BackendOps::gemm_bias_act` 経由〉）を
//!    検証する。

use backend_cpu::{CpuBackendOps, GemmError, gemm_blis_bias_act_parallel, gemm_blis_parallel};
use bench_harness::rng::Xorshift64Star;
use tensor_core::{Activation, BackendOps, Tensor};

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// 非融合合成の逐次参照実装（`gemm_blis_parallel` → bias 行加算 → `relu`
/// 相当の activation 適用）。`gemm_blis_bias_act_parallel` の bit 完全
/// 一致テストの基準点として使う。
fn compose_reference(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    gemm_blis_parallel(a, b, &mut c, m, n, k).unwrap();
    if let Some(bias) = bias {
        for row in c.chunks_mut(n) {
            for (x, bv) in row.iter_mut().zip(bias.iter()) {
                *x += *bv;
            }
        }
    }
    match act {
        Activation::None => {}
        Activation::Relu => {
            for x in c.iter_mut() {
                *x = x.max(0.0);
            }
        }
        _ => unreachable!("test only exercises known Activation variants"),
    }
    c
}

// --- 1. 既知値 ---

#[test]
fn gemm_bias_act_matches_hand_computed_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let bias = vec![-100.0, 1.0];
    let mut c = vec![0.0; 4];
    gemm_blis_bias_act_parallel(&a, &b, &mut c, 2, 2, 2, Some(&bias), Activation::Relu).unwrap();
    // A@B = [[19, 22], [43, 50]] → +bias[-100, 1] → [[-81, 23], [-57, 51]]
    // → relu → [[0, 23], [0, 51]]
    assert_eq!(c, vec![0.0, 23.0, 0.0, 51.0]);
}

#[test]
fn gemm_bias_act_none_bias_none_act_matches_plain_gemm() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c_fused = vec![0.0; 4];
    gemm_blis_bias_act_parallel(&a, &b, &mut c_fused, 2, 2, 2, None, Activation::None).unwrap();

    let mut c_plain = vec![0.0; 4];
    gemm_blis_parallel(&a, &b, &mut c_plain, 2, 2, 2).unwrap();
    assert_eq!(c_fused, c_plain);
}

#[test]
fn gemm_bias_act_no_bias_with_relu_matches_composed_reference() {
    // `bias=None` かつ `act=Relu` の組合せは上記 2 ケース（両方指定・両方
    // 無指定）と別に検証する必要がある（`apply_epilogue` の bias 分岐と
    // activation 分岐が独立して機能することの確認）。
    let a = vec![1.0, -2.0, -3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c_fused = vec![0.0; 4];
    gemm_blis_bias_act_parallel(&a, &b, &mut c_fused, 2, 2, 2, None, Activation::Relu).unwrap();

    let c_ref = compose_reference(&a, &b, 2, 2, 2, None, Activation::Relu);
    assert_eq!(c_fused, c_ref);
}

// --- 2. 融合版 vs 非融合合成: bit 完全一致（形状グリッド） ---

const SHAPE_GRID_M: [usize; 6] = [1, 5, 15, 19, 51, 200];
const SHAPE_GRID_N: [usize; 6] = [1, 5, 17, 19, 512, 600];
const SHAPE_GRID_K: [usize; 5] = [1, 3, 255, 257, 700];

#[test]
fn gemm_bias_act_matches_composed_reference_bit_exact_shape_grid() {
    let mut seed = 500u64;
    for &m in &SHAPE_GRID_M {
        for &n in &SHAPE_GRID_N {
            for &k in &SHAPE_GRID_K {
                let a = random_matrix(seed, m * k);
                seed += 1;
                let b = random_matrix(seed, k * n);
                seed += 1;
                let bias = random_matrix(seed, n);
                seed += 1;

                for act in [Activation::None, Activation::Relu] {
                    let mut c_fused = vec![0.0f32; m * n];
                    gemm_blis_bias_act_parallel(&a, &b, &mut c_fused, m, n, k, Some(&bias), act)
                        .unwrap();

                    let c_ref = compose_reference(&a, &b, m, n, k, Some(&bias), act);

                    assert_eq!(
                        c_fused, c_ref,
                        "融合版と非融合合成が bit 一致しない（m={m}, n={n}, k={k}, act={act:?}）"
                    );
                }
            }
        }
    }
}

// --- 3. REQ-2 統一複合判定でも重ねて確認 ---

#[test]
fn gemm_bias_act_passes_unified_parity_check() {
    let (m, n, k) = (129, 130, 131);
    let a = random_matrix(0xAAAA_AAAA, m * k);
    let b = random_matrix(0xBBBB_BBBB, k * n);
    let bias = random_matrix(0xCCCC_CCCC, n);

    let mut c_fused = vec![0.0f32; m * n];
    gemm_blis_bias_act_parallel(&a, &b, &mut c_fused, m, n, k, Some(&bias), Activation::Relu)
        .unwrap();

    let c_ref = compose_reference(&a, &b, m, n, k, Some(&bias), Activation::Relu);

    backend_cpu::assert_parity("gemm_bias_act vs 非融合合成", &c_fused, &c_ref);
}

// --- 4. エッジケース ---

#[test]
fn gemm_bias_act_handles_zero_dims() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    assert!(gemm_blis_bias_act_parallel(&a, &b, &mut c, 0, 0, 0, None, Activation::None).is_ok());
}

#[test]
fn gemm_bias_act_handles_zero_n_as_noop() {
    let a = vec![1.0f32; 4];
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    let bias: Vec<f32> = vec![];
    assert!(
        gemm_blis_bias_act_parallel(&a, &b, &mut c, 2, 0, 2, Some(&bias), Activation::Relu).is_ok()
    );
}

#[test]
fn gemm_bias_act_rejects_bias_len_mismatch() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let bias = vec![1.0, 2.0, 3.0]; // n=2 のはずが 3
    let mut c = vec![0.0; 4];
    let err = gemm_blis_bias_act_parallel(&a, &b, &mut c, 2, 2, 2, Some(&bias), Activation::Relu)
        .unwrap_err();
    assert!(matches!(
        err,
        GemmError::BiasLenMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

/// `BackendOps::gemm_bias_act`（`CpuBackendOps` オーバーライド）経由での
/// 非 contiguous 入力（transpose view）の検証。`gemm`（既存経路）と同じ
/// `contiguous()` 実体化を経由することを確認する。
#[test]
fn cpu_backend_ops_gemm_bias_act_handles_non_contiguous_input() {
    let ops = CpuBackendOps::new();
    // 3x2 を作って transpose し、2x3 の非 contiguous view として使う。
    let base = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let a = base
        .transpose(0, 1)
        .expect("2 次元 transpose は成功するはず"); // [2, 3]、非 contiguous
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2]).unwrap();
    let bias = Tensor::new(vec![10.0, -10.0], &[2]).unwrap();

    let fused = ops
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("非 contiguous 入力でも gemm_bias_act は成功するはず");

    let plain = ops.gemm(&a, &b).expect("gemm 経路との比較用");
    let plain_data = plain.as_slice().unwrap();
    let expected: Vec<f32> = plain_data
        .chunks(2)
        .flat_map(|row| [(row[0] + 10.0).max(0.0), (row[1] - 10.0).max(0.0)])
        .collect();

    assert_eq!(fused.as_slice().unwrap(), expected.as_slice());
}

/// bias が `[n]` ちょうどではない shape でも、`out_shape`（`[m, n]`）へ
/// ブロードキャスト可能な場合は非融合パス（`gemm` → `add` → act）へ
/// フォールバックして成功する（デフォルト実装と同一の意味論。
/// Issue #203 Review 指摘: 融合パス専用の独自厳格検証が `[n]` ちょうど
/// でない broadcast 可能 shape〈本ケースは `out_shape` と同一の `[2, 2]`〉を
/// 無条件拒否し、デフォルト実装〈CUDA／Metal がフォールバックする経路〉
/// と挙動が食い違っていたことの回帰防止）。
#[test]
fn cpu_backend_ops_gemm_bias_act_falls_back_for_broadcastable_non_row_bias() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    // bias は `[n]`（= `[2]`）ちょうどではないが `out_shape` と同一の
    // `[2, 2]`。`add` の同一 shape 加算として成功するはず。
    let bias = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let fused = ops
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("out_shape と同一 shape の bias は非融合フォールバックで成功するはず");

    let plain = ops.gemm(&a, &b).expect("gemm 経路との比較用");
    let plain_data = plain.as_slice().unwrap();
    let bias_data = bias.as_slice().unwrap();
    let expected: Vec<f32> = plain_data
        .iter()
        .zip(bias_data)
        .map(|(c, bi)| (c + bi).max(0.0))
        .collect();

    assert_eq!(fused.as_slice().unwrap(), expected.as_slice());
}

/// bias が `[1]`（スカラー相当）や `[1, n]` のような、レビューで名指しで
/// 指摘された broadcast 可能 shape（`[n]` ちょうどではない）でも成功する
/// ことを確認する（`m != n` の非正方 shape で `[n]` との取り違えを排除）。
#[test]
fn cpu_backend_ops_gemm_bias_act_falls_back_for_scalar_and_row_vector_bias() {
    let ops = CpuBackendOps::new();
    // a: [2, 2]・b: [2, 3] -> out: [2, 3]（n = 3）。
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]).unwrap();
    let plain = ops.gemm(&a, &b).expect("gemm 経路との比較用");
    let plain_data = plain.as_slice().unwrap().to_vec();

    // bias: [1]（スカラー。全要素へ同一値を加算）。
    let bias_scalar = Tensor::new(vec![10.0], &[1]).unwrap();
    let fused_scalar = ops
        .gemm_bias_act(&a, &b, Some(&bias_scalar), Activation::None)
        .expect("[1] は broadcast 可能なので成功するはず");
    let expected_scalar: Vec<f32> = plain_data.iter().map(|c| c + 10.0).collect();
    assert_eq!(fused_scalar.as_slice().unwrap(), expected_scalar.as_slice());

    // bias: [1, n]（行ベクトルを rank 2 で表現。`[n]` とは shape が異なる）。
    let bias_row = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let fused_row = ops
        .gemm_bias_act(&a, &b, Some(&bias_row), Activation::None)
        .expect("[1, n] は broadcast 可能なので成功するはず");
    let bias_row_data = [1.0f32, 2.0, 3.0];
    let expected_row: Vec<f32> = plain_data
        .chunks(3)
        .flat_map(|row| {
            row.iter()
                .zip(bias_row_data)
                .map(|(c, bi)| c + bi)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(fused_row.as_slice().unwrap(), expected_row.as_slice());
}

/// broadcast 不能な shape（`out_shape` の末尾軸と一致せず・1 でもない）は
/// `ShapeMismatch` を返す（`add` のブロードキャスト判定への委譲経路）。
#[test]
fn cpu_backend_ops_gemm_bias_act_rejects_non_broadcastable_bias() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    // n = 2 に対し bias 長 3（`[1]`・`[2]`・`[2, 2]` のいずれとも
    // 一致せず broadcast 不能）。
    let bias = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

    let err = ops
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .unwrap_err();
    assert!(matches!(
        err,
        tensor_core::device::BackendError::ShapeMismatch(_)
    ));
}
