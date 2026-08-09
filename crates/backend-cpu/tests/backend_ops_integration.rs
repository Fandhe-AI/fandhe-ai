//! TASK-1.9d（#47）: 3 バックエンド（CPU／CUDA／Metal）統合テストの本体。
//!
//! `tensor_core::BackendOps`（TASK-1.9c・#46）経由の演算が、抽象層を
//! 介してもなお各バックエンドの実カーネルと同じ結果を返すことを検証する
//! （受け入れ条件: 抽象層経由の演算が全バックエンドで期待値と一致する）。
//!
//! **`backend_ops_dispatch.rs`（TASK-1.9c・#46）との役割分担**: 同ファイルは
//! 「同一コードで 3 バックエンドのカーネルが呼び分けられる」という
//! ディスパッチ機構そのものの受け入れ条件を検証する最小限のテストに
//! 留めている（同ファイル冒頭コメント参照）。本ファイルはその上で、
//! CPU 全 8 演算の期待値一致・非 contiguous 入力・エラー経路・境界形状・
//! 3 バックエンド横断のエンドツーエンド経路までを網羅する「本格的な統合
//! テスト」（同ファイルが引き継ぎ先として明示）を担う。判定式・許容誤差の
//! 独自定義は行わず、`backend_cpu::parity::{assert_parity,
//! matmul_reference_fma}`（REQ-2 統一複合判定・FMA 契約の唯一の参照点）を
//! 再利用する（`.claude/rules/coding-rust.md`）。
//!
//! 入力生成は `bench_harness::rng::Xorshift64Star`（決定的シード。
//! `.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード設定
//! ユーティリティを使う」）。

use backend_cpu::CpuBackendOps;
use backend_cpu::parity::{assert_parity, matmul_reference_fma};
use backend_cuda::CudaBackendOps;
use bench_harness::rng::Xorshift64Star;
use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor, ops_for};

#[cfg(target_os = "macos")]
use backend_metal::MetalBackendOps;

// ---------------------------------------------------------------------
// 1. CPU 全 8 演算の期待値一致
// ---------------------------------------------------------------------

/// gemm: `ops_for(&ops, Device::Cpu)` 経由の結果を `matmul_reference_fma`
/// （FMA 契約の唯一の参照点）と `assert_parity`（複合判定の唯一の実体）で
/// 突き合わせる。形状は既存テスト（`backend_ops_dispatch.rs` の 2x2 既知値・
/// `gemm_blis_parity.rs` 等）と重複しない境界含みの組を選ぶ。
#[test]
fn cpu_gemm_via_backend_ops_matches_reference_across_boundary_shapes() {
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];

    // (seed_a, seed_b, m, n, k)
    let cases: [(u64, u64, usize, usize, usize); 5] = [
        (101, 102, 1, 1, 1),    // 最小境界（スカラー積相当）
        (103, 104, 1, 7, 5),    // 行ベクトル（m=1）
        (105, 106, 9, 1, 4),    // 列ベクトル（n=1）
        (107, 108, 33, 17, 65), // 非正方・素数近傍
        (109, 110, 4, 4, 1),    // k=1（外積相当）
    ];

    for (seed_a, seed_b, m, n, k) in cases {
        let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
        let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);

        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a_data, &b_data, &mut expected, m, n, k)
            .expect("well-formed test shapes must pass matmul_reference_fma validation");

        let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
        let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");

        let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");
        let result = selected.gemm(&a, &b).expect("cpu gemm always succeeds");
        assert_eq!(result.shape(), &[m, n]);

        let actual = result.as_slice().expect("contiguous result");
        assert_parity(&format!("cpu gemm m={m} n={n} k={k}"), actual, &expected);
    }
}

/// add／mul／relu／exp／tanh: テスト内の逐次スカラー参照計算
/// （同一 `f32` std 関数）との一致を検証する。
#[test]
fn cpu_elementwise_ops_via_backend_ops_match_scalar_reference() {
    let cpu = CpuBackendOps::new();

    let a_data: Vec<f32> = vec![-3.5, -1.0, 0.0, 0.5, 2.25, 10.0];
    let b_data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let a = Tensor::new(a_data.clone(), &[2, 3]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[2, 3]).expect("valid tensor");

    let add_expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();
    let add_result = cpu.add(&a, &b).expect("cpu add succeeds");
    assert_parity(
        "cpu add",
        add_result.as_slice().expect("contiguous"),
        &add_expected,
    );

    let mul_expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x * y).collect();
    let mul_result = cpu.mul(&a, &b).expect("cpu mul succeeds");
    assert_parity(
        "cpu mul",
        mul_result.as_slice().expect("contiguous"),
        &mul_expected,
    );

    let relu_expected: Vec<f32> = a_data.iter().map(|x| x.max(0.0)).collect();
    let relu_result = cpu.relu(&a).expect("cpu relu succeeds");
    assert_parity(
        "cpu relu",
        relu_result.as_slice().expect("contiguous"),
        &relu_expected,
    );

    let exp_expected: Vec<f32> = a_data.iter().map(|x| x.exp()).collect();
    let exp_result = cpu.exp(&a).expect("cpu exp succeeds");
    assert_parity(
        "cpu exp",
        exp_result.as_slice().expect("contiguous"),
        &exp_expected,
    );

    let tanh_expected: Vec<f32> = a_data.iter().map(|x| x.tanh()).collect();
    let tanh_result = cpu.tanh(&a).expect("cpu tanh succeeds");
    assert_parity(
        "cpu tanh",
        tanh_result.as_slice().expect("contiguous"),
        &tanh_expected,
    );
}

/// sum／max: `dim=None`（全縮約）と `dim=Some(d)`（各軸）の双方を
/// 手計算参照と突合する。
#[test]
fn cpu_reduction_ops_via_backend_ops_match_hand_computed_reference() {
    let cpu = CpuBackendOps::new();

    // A = [[1, 2, 3], [4, 5, 6]] (2x3)
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");

    // 全縮約
    let sum_all = cpu.sum(&a, None).expect("cpu sum(None) succeeds");
    assert_parity(
        "cpu sum all",
        sum_all.as_slice().expect("contiguous"),
        &[21.0],
    );

    let max_all = cpu.max(&a, None).expect("cpu max(None) succeeds");
    assert_parity(
        "cpu max all",
        max_all.as_slice().expect("contiguous"),
        &[6.0],
    );

    // 軸 0（行方向縮約）: 列ごとの和/最大 = [1+4, 2+5, 3+6] = [5, 7, 9]
    let sum_dim0 = cpu.sum(&a, Some(0)).expect("cpu sum(dim=0) succeeds");
    assert_parity(
        "cpu sum dim=0",
        sum_dim0.as_slice().expect("contiguous"),
        &[5.0, 7.0, 9.0],
    );
    let max_dim0 = cpu.max(&a, Some(0)).expect("cpu max(dim=0) succeeds");
    assert_parity(
        "cpu max dim=0",
        max_dim0.as_slice().expect("contiguous"),
        &[4.0, 5.0, 6.0],
    );

    // 軸 1（列方向縮約）: 行ごとの和/最大 = [1+2+3, 4+5+6] = [6, 15]
    let sum_dim1 = cpu.sum(&a, Some(1)).expect("cpu sum(dim=1) succeeds");
    assert_parity(
        "cpu sum dim=1",
        sum_dim1.as_slice().expect("contiguous"),
        &[6.0, 15.0],
    );
    let max_dim1 = cpu.max(&a, Some(1)).expect("cpu max(dim=1) succeeds");
    assert_parity(
        "cpu max dim=1",
        max_dim1.as_slice().expect("contiguous"),
        &[3.0, 6.0],
    );
}

// ---------------------------------------------------------------------
// 2. 非 contiguous 入力（transpose ビュー）
// ---------------------------------------------------------------------

/// elementwise・reduction・gemm 各カテゴリで `transpose` ビューを通し、
/// `contiguous()` 実体化経路の正しさを検証する（既存 `backend_ops_dispatch.rs`
/// は gemm の非 contiguous のみカバーしていた）。
#[test]
fn cpu_backend_ops_handle_non_contiguous_transpose_input_across_categories() {
    let cpu = CpuBackendOps::new();

    // A = [[1, 2, 3], [4, 5, 6]] (2x3) -> A^T = [[1, 4], [2, 5], [3, 6]] (3x2)
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");
    let a_t = a.transpose(0, 1).expect("valid transpose");
    assert!(
        !a_t.is_contiguous(),
        "transpose ビューは非 contiguous なはず"
    );

    // --- elementwise: A^T + A^T (同形状同士の add) ---
    let add_result = cpu
        .add(&a_t, &a_t)
        .expect("non-contiguous add must succeed via contiguous() realization");
    assert_parity(
        "cpu add non-contiguous",
        add_result.as_slice().expect("contiguous"),
        &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0],
    );

    // --- reduction: A^T の全縮約（値集合は転置で不変なので合計は 21） ---
    let sum_result = cpu
        .sum(&a_t, None)
        .expect("non-contiguous sum must succeed via contiguous() realization");
    assert_parity(
        "cpu sum non-contiguous",
        sum_result.as_slice().expect("contiguous"),
        &[21.0],
    );

    // --- gemm: A^T (3x2) @ B (2x2) ---
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("valid tensor"); // 単位行列
    let ops: Vec<&dyn BackendOps> = vec![&cpu];
    let gemm_result = ops_for(&ops, Device::Cpu)
        .expect("cpu ops registered")
        .gemm(&a_t, &b)
        .expect("non-contiguous gemm must succeed via contiguous() realization");
    assert_eq!(gemm_result.shape(), &[3, 2]);
    // A^T @ I = A^T
    assert_parity(
        "cpu gemm non-contiguous",
        gemm_result.as_slice().expect("contiguous"),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
}

// ---------------------------------------------------------------------
// 3. エラー経路（panic しない契約）
// ---------------------------------------------------------------------

#[test]
fn shape_mismatch_returns_typed_error_not_panic() {
    let cpu = CpuBackendOps::new();
    // 2x3 と 2x3 の add は shape 一致だが、gemm には shape 不整合
    // （2x3 @ 2x3 は k 不一致で `ShapeMismatch`）。
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");

    let ops: Vec<&dyn BackendOps> = vec![&cpu];
    let result = ops_for(&ops, Device::Cpu)
        .expect("cpu ops registered")
        .gemm(&a, &b);
    assert!(matches!(result, Err(BackendError::ShapeMismatch(_))));
}

#[test]
fn unregistered_device_returns_device_unavailable_not_panic() {
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");

    let result = ops_for(&ops, Device::Cuda(0)).map(|selected| selected.gemm(&a, &b));
    assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
}

#[test]
fn cuda_add_returns_typed_unsupported_not_panic() {
    // CUDA は driver 非搭載環境でも `add` は「未実装カーネル」の
    // `Unsupported` を driver 呼び出し前に返す（`backend-cuda/src/ops.rs`）。
    let cuda = CudaBackendOps::new(0);
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");
    let b = a.clone();

    assert!(matches!(
        cuda.add(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
}

// ---------------------------------------------------------------------
// 4. 端点（空テンソル・単一要素）
// ---------------------------------------------------------------------

#[test]
fn cpu_backend_ops_handle_single_element_tensors() {
    let cpu = CpuBackendOps::new();
    let a = Tensor::new(vec![3.0f32], &[1, 1]).expect("valid tensor");
    let b = Tensor::new(vec![4.0f32], &[1, 1]).expect("valid tensor");

    let add_result = cpu.add(&a, &b).expect("single-element add succeeds");
    assert_parity(
        "cpu add single",
        add_result.as_slice().expect("contiguous"),
        &[7.0],
    );

    let ops: Vec<&dyn BackendOps> = vec![&cpu];
    let gemm_result = ops_for(&ops, Device::Cpu)
        .expect("cpu ops registered")
        .gemm(&a, &b)
        .expect("single-element gemm succeeds");
    assert_parity(
        "cpu gemm single",
        gemm_result.as_slice().expect("contiguous"),
        &[12.0],
    );

    let sum_result = cpu.sum(&a, None).expect("single-element sum succeeds");
    assert_parity(
        "cpu sum single",
        sum_result.as_slice().expect("contiguous"),
        &[3.0],
    );
}

/// 空テンソル（numel==0）の挙動固定。`sum` は単位元 0.0 を返す
/// （`reduction.rs` の「空縮約の意味論」契約）。`add`／`relu` 等の
/// elementwise は要素なしで正常終了する。
#[test]
fn cpu_backend_ops_handle_empty_tensors() {
    let cpu = CpuBackendOps::new();
    let empty_a = Tensor::<f32>::new(vec![], &[0]).expect("valid empty tensor");
    let empty_b = Tensor::<f32>::new(vec![], &[0]).expect("valid empty tensor");

    let add_result = cpu
        .add(&empty_a, &empty_b)
        .expect("empty add succeeds (no elements)");
    assert_eq!(add_result.numel(), 0);

    let relu_result = cpu
        .relu(&empty_a)
        .expect("empty relu succeeds (no elements)");
    assert_eq!(relu_result.numel(), 0);

    // 空縮約の意味論（`reduction.rs`）: sum は単位元 0.0 を返す。
    let sum_result = cpu
        .sum(&empty_a, None)
        .expect("empty sum returns identity 0.0");
    assert_parity(
        "cpu sum empty",
        sum_result.as_slice().expect("contiguous"),
        &[0.0],
    );

    // max は単位元を持たないため `KernelLaunchFailed`（`EmptyReduction` の
    // 写像先。`ops.rs::reduce_error_to_backend_error`）を返す（panic しない）。
    let max_result = cpu.max(&empty_a, None);
    assert!(matches!(
        max_result,
        Err(BackendError::KernelLaunchFailed(_))
    ));
}

// ---------------------------------------------------------------------
// 5. 3 バックエンド横断のエンドツーエンド
// ---------------------------------------------------------------------

/// `enumerate_all`／`select_from`（1.9a）→ `ops_for`（1.9c）→ 演算実行
/// までを単一の共通関数で記述し、`Device` を差し替えるだけで CPU は実行
/// 成功・CUDA は環境適応（利用可なら CPU 結果と `assert_parity`、不可なら
/// `CudaUnavailable`）・Metal は同様（macOS のみ）、を検証する
/// （`cpu_cuda_parity.rs` の環境適応スモークの分岐パターンを踏襲）。
fn run_gemm_end_to_end(
    ops: &[&dyn BackendOps],
    device: Device,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    let selected = ops_for(ops, device)?;
    selected.gemm(a, b)
}

#[test]
fn end_to_end_dispatch_cpu_executes_successfully() {
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];

    let a_data = Xorshift64Star::new(201).fill_vec(4 * 3);
    let b_data = Xorshift64Star::new(202).fill_vec(3 * 5);
    let mut expected = vec![0.0f32; 4 * 5];
    matmul_reference_fma(&a_data, &b_data, &mut expected, 4, 5, 3).expect("valid shapes");

    let a = Tensor::new(a_data, &[4, 3]).expect("valid tensor");
    let b = Tensor::new(b_data, &[3, 5]).expect("valid tensor");

    let result = run_gemm_end_to_end(&ops, Device::Cpu, &a, &b).expect("cpu always succeeds");
    assert_parity(
        "end-to-end cpu gemm",
        result.as_slice().expect("contiguous"),
        &expected,
    );
}

#[test]
fn end_to_end_dispatch_cuda_matches_cpu_when_available_or_returns_typed_error() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

    let a_data = Xorshift64Star::new(203).fill_vec(4 * 3);
    let b_data = Xorshift64Star::new(204).fill_vec(3 * 5);
    let a = Tensor::new(a_data.clone(), &[4, 3]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[3, 5]).expect("valid tensor");

    let cpu_result = run_gemm_end_to_end(&ops, Device::Cpu, &a, &b).expect("cpu always succeeds");

    match run_gemm_end_to_end(&ops, Device::Cuda(0), &a, &b) {
        Ok(cuda_result) => {
            // 実機（CUDA 搭載 CI ランナー）: CPU 結果と複合判定で一致する
            // ことまで確認する（本テストが「環境適応」であるゆえん）。
            assert_parity(
                "end-to-end cuda gemm vs cpu",
                cuda_result.as_slice().expect("contiguous"),
                cpu_result.as_slice().expect("contiguous"),
            );
        }
        Err(BackendError::CudaUnavailable(_)) => {
            // 非搭載環境（本 CI）での期待経路（panic しない）。
        }
        Err(other) => panic!("unexpected error variant for CUDA end-to-end dispatch: {other}"),
    }
}

#[cfg(target_os = "macos")]
#[test]
fn end_to_end_dispatch_metal_matches_cpu_when_available_or_returns_typed_error() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &metal];

    let a_data = Xorshift64Star::new(205).fill_vec(4 * 3);
    let b_data = Xorshift64Star::new(206).fill_vec(3 * 5);
    let a = Tensor::new(a_data.clone(), &[4, 3]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[3, 5]).expect("valid tensor");

    let cpu_result = run_gemm_end_to_end(&ops, Device::Cpu, &a, &b).expect("cpu always succeeds");

    match run_gemm_end_to_end(&ops, Device::Metal, &a, &b) {
        Ok(metal_result) => {
            assert_parity(
                "end-to-end metal gemm vs cpu",
                metal_result.as_slice().expect("contiguous"),
                cpu_result.as_slice().expect("contiguous"),
            );
        }
        Err(BackendError::DeviceUnavailable(_))
        | Err(BackendError::DeviceAllocationFailed(_))
        | Err(BackendError::KernelLaunchFailed(_)) => {
            // Metal 非対応環境での期待経路（panic しない。
            // `backend_ops_dispatch.rs` と同じ許容 variant 集合。PR #262
            // Bugbot 指摘対応を踏襲）。
        }
        Err(other) => panic!("unexpected error variant for Metal end-to-end dispatch: {other}"),
    }
}

// ---------------------------------------------------------------------
// 4. `run_fused` の結線ガード（#167 実装時に発見した結線ギャップの回帰
//    テスト。TASK-12.1 系列〈#163・#164〉が「相手のスコープ」と委ね合った
//    結果 `CpuBackendOps` は `run_fused` をオーバーライドしておらず、
//    デフォルト実装〈`Unsupported` fail-safe〉のまま融合カーネルが
//    一度も起動しない状態だった。本テストはマージ前は
//    `BackendError::Unsupported` で失敗する＝結線漏れの回帰を検知する。
// ---------------------------------------------------------------------

/// `&dyn BackendOps` 経由の `run_fused` が `Ok` を返し（デフォルト
/// `Unsupported` へフォールバックしていないこと）、結果が per-op 逐次
/// 合成（非融合基準）と REQ-2 複合判定で一致することを固定する。
/// プランは `fused_elementwise_parity.rs` の ew4（add→relu→exp→tanh）と
/// 同型（`tensor_core::FusionPlan::from_ops` はクレート間構築経路。
/// `docs/kernel-fusion.md` §3.4 根拠）。
#[test]
fn cpu_run_fused_via_backend_ops_is_wired_and_matches_sequential_composition() {
    let cpu = CpuBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu];
    let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");

    let shape = vec![1usize << 10];
    let x_data = Xorshift64Star::new(301).fill_vec(shape[0]);
    let y_data = Xorshift64Star::new(302).fill_vec(shape[0]);
    let x = Tensor::new(x_data, &shape).expect("valid tensor");
    let y = Tensor::new(y_data, &shape).expect("valid tensor");

    // ops: 0=Input(x) 1=Input(y) 2=Add(0,1) 3=Relu(2) 4=Exp(3) 5=Tanh(4)
    let plan = FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ],
        shape,
        DType::F32,
        2,
    )
    .expect("well-formed fusion plan");

    let fused = selected
        .run_fused(&plan, &[&x, &y])
        .expect("run_fused must be wired to the fusion kernel (not the Unsupported default)");

    let a = selected.add(&x, &y).expect("cpu add always succeeds");
    let b = selected.relu(&a).expect("cpu relu always succeeds");
    let c = selected.exp(&b).expect("cpu exp always succeeds");
    let expected = selected.tanh(&c).expect("cpu tanh always succeeds");

    assert_parity(
        "run_fused via BackendOps vs sequential per-op composition",
        fused.as_slice().expect("contiguous fused result"),
        expected.as_slice().expect("contiguous sequential result"),
    );
}
