//! 非同期化後（イシュー #1011 ツリー・PR #1064）の順序依存・数値一致・
//! 前提条件回帰テスト。`docs/backend-cuda-async-execution-design.md` §8
//! が #1014 へ引き渡す実機依存テスト（T1・T2・T4）を実装する。GPU 非依存の
//! T3 系モックテストは `crates/backend-cuda/src/context_cache.rs` の
//! `poison_state_tests`・`crates/backend-cuda/src/ops.rs` の `#[cfg(test)]`
//! モジュール（poison 拒否テスト群）・
//! `crates/backend-cuda/src/async_ordering_poison_tests.rs`（T3i）に分離
//! 済み（本ファイルには置かない）。
//!
//! `crates/backend-cuda/tests/sgd_device_real_device.rs`・
//! `gemm_resident_real_device.rs` と同じ構成方針（`#[ignore]` 分離・
//! 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --all-features \
//!     --test async_ordering_real_device -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::DeviceBufferView;
use fandhe_ai_tensor_core::{BackendOps, SgdStepConfig, Tensor};

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

/// T1（順序依存）: `docs/backend-cuda-async-execution-design.md` I1/I2 の
/// 実機検証。
///
/// **設計上の注記（advisor レビューを経た方針決定・#1014）**: 当初計画
/// （イシュー #1014 の実装計画 §5 手順 2）は「async 系」「各段の後に
/// `download` を挟む staged-sync 系」「CPU 参照系」の 3 系統比較を想定
/// していたが、`gemm_resident_rhs` は戻り値がホスト `Tensor` である以上
/// `with_driver_call` の観測点（D2H）を経由し、`docs/
/// backend-cuda-async-execution-design.md` §4 の同期点契約どおり毎回
/// ストリームを排出する（single-stream 構成。§2.1）。そのため
/// 「staged-sync 系」を追加しても「async 系」と全く同一のスケジュールに
/// なり、両者は判別力を持たない（advisor 指摘）。よって本テストは
/// 「async 系 vs CPU 参照系」の 1 対比較へ簡略化する。
///
/// I1 の本質的な検証点は **`sgd_step_device`（W の in-place 更新。D2H を
/// 伴わない唯一の常駐経路）と直後の `gemm_resident_rhs`（W を読む forward）
/// との間に明示同期を挟まない**ことであり、この間で `download`／
/// `sync_to_host`／`snapshot_resident_leaves` 等を一切呼ばないことが
/// 不変条件（I1 の再現に必須）。もしこの間に `download` を挿入すると
/// ストリームが強制的に排出され、非同期実行下の順序保証（同一ストリーム
/// 上の投入順序のみに依拠する契約）を検証しないテストに縮退してしまう。
/// **将来の変更でこの箇所に同期呼び出しを追加しないこと**（テストの意図が
/// 無効化される）。
///
/// 損失を `L = 0.5 * sum(y^2)`（`y = X @ W`）と定義し
/// `dL/dW = X^T @ y` とすることで、各ステップの forward 出力 `y` が
/// 直後の backward（勾配計算）・update（`sgd_step_device`）に伝播し、
/// さらに次ステップの forward が「直近の update 後の W」を読む、という
/// 依存の連鎖（forward → backward → update → forward → ...）を作る。
/// もし non-同期化のどこかで投入順序が崩れる（例: sgd_step_device の
/// カーネルが次の forward より後に実行される）と、CPU 参照実装との
/// 数値が発散し検出できる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn forward_backward_update_chain_matches_cpu_reference_without_explicit_sync() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let (b, k, n) = (6usize, 8usize, 4usize);
    let x_data: Vec<f32> = (0..b * k)
        .map(|i| ((i as f32) * 0.037).sin() * 0.5)
        .collect();
    let x = tensor(x_data, &[b, k]);
    let w_init: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.021).cos() * 0.3)
        .collect();

    // CUDA 側: W をデバイス常駐のまま保持し、以降のループでは一度も
    // download しない（最終ステップ後に 1 回だけ download する）。
    let mut w_cuda = cuda_mem.upload(&tensor(w_init.clone(), &[k, n])).unwrap();
    let mut velocity_cuda = cuda_mem.alloc_zeroed(&[k, n]).unwrap();

    // CPU 参照側: `sgd_step_device` は `DeviceBuffer`（`CpuMemory` 経由）を
    // 要求するため、CUDA 側と同じ `MemoryOps` 抽象で保持する（forward
    // 自体は `CpuBackendOps::gemm` がホスト `Tensor` を直接読み書きする）。
    let mut w_cpu_dev = cpu_mem.upload(&tensor(w_init.clone(), &[k, n])).unwrap();
    let mut velocity_cpu_dev = cpu_mem.alloc_zeroed(&[k, n]).unwrap();

    const STEPS: usize = 12;
    for step in 0..STEPS {
        let w_shape = [k, n];

        // --- forward（CUDA: gemm_resident_rhs。W をデバイス常駐のまま
        // 読む。戻り値はホスト Tensor のため D2H を 1 回伴う）。CPU 参照側
        // は `CpuMemory::download` がホスト内メモリの単純な読み出しに
        // すぎない（実デバイス往復を伴わない）ため、ここで毎回 download
        // しても CUDA 側で検証したい非同期順序保証（I1）とは無関係
        // （I1 の不変条件は CUDA 側にのみ課す。上記ドキュメンテーション
        // コメント参照）。 ---
        let w_view_cuda = DeviceBufferView::new(&w_cuda, 0, &w_shape).unwrap();
        let y_cuda = cuda_ops.gemm_resident_rhs(&x, w_view_cuda, None).unwrap();
        let w_cpu = cpu_mem.download(&w_cpu_dev).unwrap();
        let y_cpu = cpu_ops.gemm(&x, &w_cpu).unwrap();

        // --- backward（ホスト側で解析的に勾配を計算。L = 0.5*sum(y^2)
        // なので dL/dy = y、dL/dW = X^T @ y） ---
        let grad_w_cuda = cpu_ops.gemm(&x.transpose_2d().unwrap(), &y_cuda).unwrap();
        let grad_w_cpu = cpu_ops.gemm(&x.transpose_2d().unwrap(), &y_cpu).unwrap();

        let config = SgdStepConfig {
            lr: 0.05,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.01,
            nesterov: true,
            is_first_step: step == 0,
        };

        // --- update（CUDA: sgd_step_device。in-place・D2H なし。
        // ここから次ループの forward までの間、W には一切触れない） ---
        let grad_w_cuda_dev = cuda_mem.upload(&grad_w_cuda).unwrap();
        cuda_ops
            .sgd_step_device(
                &mut w_cuda,
                &grad_w_cuda_dev,
                Some(&mut velocity_cuda),
                &config,
            )
            .expect("cuda sgd_step_device must succeed on real hardware");
        let grad_w_cpu_dev = cpu_mem.upload(&grad_w_cpu).unwrap();
        cpu_ops
            .sgd_step_device(
                &mut w_cpu_dev,
                &grad_w_cpu_dev,
                Some(&mut velocity_cpu_dev),
                &config,
            )
            .unwrap();
    }

    // 最終ステップ後、初めて W を download する。
    let w_cuda_final = cuda_mem.download(&w_cuda).unwrap().contiguous();
    let w_cpu_final = cpu_mem.download(&w_cpu_dev).unwrap().contiguous();
    let w_cuda_slice = w_cuda_final.as_slice().unwrap();
    let w_cpu_slice = w_cpu_final.as_slice().unwrap();
    for i in 0..k * n {
        assert_close(
            w_cuda_slice[i],
            w_cpu_slice[i],
            &format!("W[{i}] after {STEPS} steps"),
        );
    }
}

/// T2（早期 D2H の検出）: 大形状の常駐バッファへ launch-only の
/// `sgd_step_device` を多数回連鎖投入した直後に `download` し、CPU
/// 参照実装（同一系列）と一致することを確認する。
///
/// **スコープの誠実な記述（advisor 指摘・#1014）**: 本バックエンドは
/// ordinal ごとに単一の非同期ストリームのみを持つ（設計文書 §2.1）ため、
/// 本テストで検出できるのは「D2H が別ストリーム／null ストリームへ
/// 誤って発行され、投入順序の外側で実行されてしまう」種類の回帰であり、
/// 「本来必要な明示同期が抜けている」種類の回帰そのものではない
/// （同一ストリーム上では D2H も投入順に並ぶため、明示同期の有無に
/// 依存しない）。この限定は設計文書 §8 T2 の「負の対照は任意」という
/// 記述とも整合する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn many_chained_sgd_steps_then_immediate_download_matches_cpu_reference() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda_ops = CudaBackendOps::new(device.ordinal());
    let cuda_mem = cuda_ops
        .memory_ops()
        .expect("CudaBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    // 数百万要素規模（実行時間の長いカーネルを作るため）。
    const NUMEL: usize = 4 * 1024 * 1024;
    const STEPS: usize = 50;

    let init: Vec<f32> = (0..NUMEL).map(|i| ((i as f32) * 1e-6).sin()).collect();
    let mut cuda_param = cuda_mem.upload(&tensor(init.clone(), &[NUMEL])).unwrap();
    let mut cuda_velocity = cuda_mem.alloc_zeroed(&[NUMEL]).unwrap();
    let mut cpu_param = cpu_mem.upload(&tensor(init, &[NUMEL])).unwrap();
    let mut cpu_velocity = cpu_mem.alloc_zeroed(&[NUMEL]).unwrap();

    for step in 0..STEPS {
        // 勾配は決定的な擬似データ（ステップごとに変化させ、SGD の状態
        // 〈momentum 項〉が毎回異なる値で更新されるようにする）。
        let grad_data: Vec<f32> = (0..NUMEL)
            .map(|i| 0.001 * (step as f32 + 1.0) + 1e-7 * i as f32)
            .collect();
        let grad_tensor = tensor(grad_data, &[NUMEL]);
        let cuda_grad = cuda_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();

        let config = SgdStepConfig {
            lr: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: step == 0,
        };

        // launch-only（結果を読まない）で連鎖投入する。
        cuda_ops
            .sgd_step_device(
                &mut cuda_param,
                &cuda_grad,
                Some(&mut cuda_velocity),
                &config,
            )
            .expect("cuda sgd_step_device must succeed on real hardware");
        cpu_ops
            .sgd_step_device(&mut cpu_param, &cpu_grad, Some(&mut cpu_velocity), &config)
            .unwrap();
    }

    // 連鎖投入直後（明示同期なし）に download する。
    let cuda_result = cuda_mem.download(&cuda_param).unwrap();
    let cuda_slice = cuda_result.as_slice().unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    let cpu_slice = cpu_result.contiguous();
    let cpu_slice = cpu_slice.as_slice().unwrap();

    // 全要素検査は時間がかかるため、代表的な位置（先頭・末尾・中央付近・
    // ブロック境界近傍）のみ検査する（`SGD_BLOCK_DIM` 境界に起因する
    // 潜在的なオフバイワンを拾うため、境界付近を含める）。
    let sample_indices: Vec<usize> = [0, 1, 255, 256, 257, NUMEL / 2, NUMEL - 2, NUMEL - 1]
        .into_iter()
        .filter(|&i| i < NUMEL)
        .collect();
    for i in sample_indices {
        assert_close(
            cuda_slice[i],
            cpu_slice[i],
            &format!("param[{i}] after {STEPS} chained steps, numel={NUMEL}"),
        );
    }
}

/// T4（前提確認プローブ）: `docs/backend-cuda-async-execution-design.md`
/// I4 が要求する `CudaContext::has_async_alloc()` の実測値を記録する。
///
/// 本テストはアサーションを行わない（プローブ専用。`--nocapture` の
/// 標準出力が実測値そのもの）。I4 の安全性契約（`CudaSlice::drop` が
/// 明示同期なしに use-after-free を起こさないこと）は
/// `has_async_alloc` が真の場合にのみ non-blocking な
/// `cuMemFreeAsync` 経路を通る（`pool.rs` 参照）。実測値をこのテストの
/// `--nocapture` 出力から `docs/backend-cuda-async-execution-design.md`
/// I4 の注記へ転記する運用とする（GB10 実機は未実測のまま。ローカル
/// RTX 3060 の実測値は同 doc 実装記録節を参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn probe_has_async_alloc_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let has_async_alloc = device.context().has_async_alloc();
    println!(
        "[T4 probe] CudaContext::has_async_alloc() = {has_async_alloc} (ordinal={})",
        device.ordinal()
    );
}
