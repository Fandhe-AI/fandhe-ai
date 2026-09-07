//! イシュー #1349（親 #1348・ルート #1341 → #1269）受け入れ条件 (a):
//! 「学習 step を CUDA Graph capture 経路（opt-in ON）と非 capture 経路
//! （opt-in OFF）で実行し、10 step の損失・勾配・パラメータが bit 同一
//! であること」を facade 公開 API 経由で検証する実機テスト。
//!
//! `crates/facade/tests/device_param_store_train.rs`（CPU・
//! `fandhe_ai::tape()`）と同じ学習ループ構成を `fandhe_ai::tape_for(
//! Device::Cuda(0))` へ差し替えたもの。capture できるのは update 区間
//! （`sgd_step_device_tracked`）のみであり forward／backward は対象外
//! （`docs/backend-cuda-graph-step-capture-design.md` §1・§3.2）ため、
//! bit 同一性は「同じ update カーネル・同じポインタへ同じ引数で
//! launch される」という構成上の保証（同 doc §4.5）の直接検証となる。
//!
//! **opt-in はプロセス内で最初の CUDA デバイス初期化より前に設定する
//! 必要がある**（`fandhe_ai::set_cuda_graph_step_enabled` doc 参照）ため、
//! 本ファイルは 2 プロセス構成にはせず、**先に非 capture 経路（opt-in
//! OFF）を完走させ、その後で capture 経路（opt-in ON）用に別 ordinal
//! を使う**——ただし DGX Spark GB10 等の単一 GPU 構成では ordinal 1 が
//! 存在しないため、この構成では成立しない。そのため実運用は「本ファイル
//! を 2 回、環境変数 `FANDHE_AI_CUDA_GRAPH_STEP` の有無で切り替えて
//! それぞれ 1 プロセスとして実行し、出力 JSON を突合する」形を取る
//! （`--nocapture` で loss ログを標準出力へ出す）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`docs/real-hardware-
//! verification-env.md` の手順に従う）:
//!
//! ```sh
//! # 非 capture 経路（比較の基準値）
//! cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
//!   -- --ignored --nocapture eager_baseline
//!
//! # capture 経路（opt-in ON。同一プロセス内で最初のデバイス初期化前に
//! # 設定されるよう、テスト関数の先頭で `set_cuda_graph_step_enabled(true)`
//! # を呼ぶ）
//! FANDHE_AI_CUDA_GRAPH_STEP=1 \
//! cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
//!   -- --ignored --nocapture graph_capture
//! ```
//!
//! 両者の標準出力（loss 列・最終パラメータのビット表現）を比較し、
//! 完全一致することを目視・スクリプトで確認する（本ファイル自体は
//! 環境変数の有無に応じて同じ手順を実行するだけであり、プロセスを
//! 跨いだ比較の自動化はハーネス側〈#1350 等の後続〉に委ねる）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_backend_cuda::device::CudaDevice;
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const STEPS: usize = 10;
const LR: f32 = 0.05;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `device_param_store_train.rs::train_with_device_param_store` の CUDA
/// 版。各 step の loss（`f32` そのまま。ビット比較は呼び出し元が
/// `to_bits()` で行う）と、最終的にホストへ同期したパラメータ列を返す。
///
/// `ordinal` を引数化している理由（codex-review P2 指摘対応。イシュー
/// #1349）: opt-in（`FANDHE_AI_CUDA_GRAPH_STEP`／
/// `set_cuda_graph_step_enabled`）はプロセス内最初の CUDA デバイス
/// 初期化より前に固定される必要があるため（モジュール冒頭コメント）、
/// 同一プロセス内で「opt-in OFF の基準値」と「opt-in ON の capture 経路」
/// を機械比較するには**異なる ordinal**を使う必要がある
/// （[`graph_capture_matches_eager_baseline_bit_identical_across_two_gpus`]
/// 参照）。
fn train_on_cuda(ordinal: usize, steps: usize, lr: f32) -> (Vec<f32>, Vec<Tensor<f32>>) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr);
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape =
            fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    let final_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let synced = final_tape.sync_device_param_store_to_host(&store).unwrap();
    (log, synced)
}

/// loss 列・最終パラメータを `to_bits()` の 16 進表現で標準出力へ出す
/// （プロセス間比較のための決定的なテキスト表現。浮動小数点の表示
/// 誤差を避けるため `{:?}`／`{}` ではなくビット表現を使う）。
fn print_bit_identity_report(label: &str, log: &[f32], params: &[Tensor<f32>]) {
    println!("=== cuda_graph_step_bit_identity: {label} ===");
    for (i, loss) in log.iter().enumerate() {
        println!("step[{i}].loss.bits = {:#010x}", loss.to_bits());
    }
    for (p, tensor) in params.iter().enumerate() {
        let contiguous = tensor.contiguous();
        let slice = contiguous.as_slice().unwrap_or(&[]);
        for (i, v) in slice.iter().enumerate() {
            println!("param[{p}][{i}].bits = {:#010x}", v.to_bits());
        }
    }
}

/// 非 capture 経路（opt-in OFF。既定）の基準値を出力する。
///
/// `FANDHE_AI_CUDA_GRAPH_STEP` が未設定・`graph_capable_matches_eager_
/// baseline_within_same_process` 実行前であれば opt-in は既定 OFF の
/// ままのはず。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。docs/real-hardware-verification-env.md 参照"]
fn eager_baseline() {
    assert!(
        !fandhe_ai::cuda_graph_step_enabled(),
        "本テストは opt-in OFF（既定）の基準値を記録する。環境変数 \
         FANDHE_AI_CUDA_GRAPH_STEP を設定せずに実行すること"
    );
    let (log, params) = train_on_cuda(0, STEPS, LR);
    print_bit_identity_report("eager (opt-in OFF)", &log, &params);
}

/// capture 経路（opt-in ON）を実行する。`FANDHE_AI_CUDA_GRAPH_STEP=1`
/// を設定した別プロセスとして実行することを想定する
/// （`fandhe_ai::set_cuda_graph_step_enabled` は「最初のデバイス初期化
/// より前」の制約があるため、同一プロセス内で `eager_baseline` の後に
/// 実行しても capture は成立しない）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_STEP=1 の別プロセスで実行すること"]
fn graph_capture() {
    assert!(
        fandhe_ai::cuda_graph_step_enabled(),
        "本テストは opt-in ON（環境変数 FANDHE_AI_CUDA_GRAPH_STEP=1）の \
         別プロセスとして実行すること"
    );
    let (log, params) = train_on_cuda(0, STEPS, LR);
    print_bit_identity_report("graph capture (opt-in ON)", &log, &params);
}

/// 同一プロセス内でも検証できる範囲の簡易チェック: opt-in を
/// 明示的に ON へ設定してから**新規** ordinal 上で学習ループを走らせ、
/// 少なくとも `BackendError::Unsupported`（opt-in ON だが legacy stream
/// のまま等の設定順序ミス）を起こさずに完走することを確認する
/// （bit 同一性そのものはプロセスを跨いだ突合が必要なため
/// `eager_baseline`／`graph_capture` の 2 プロセス比較に委ねる。本テスト
/// は「opt-in ON の状態で学習ループが最後まで通る」という弱い受け入れ
/// 条件のみを 1 プロセスで検証する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。単独プロセスで実行すること（opt-in をプロセスワイドに変更するため）"]
fn graph_capture_completes_training_loop_without_error() {
    fandhe_ai::set_cuda_graph_step_enabled(true);
    let (log, _params) = train_on_cuda(0, STEPS, LR);
    assert_eq!(log.len(), STEPS);
    for loss in &log {
        assert!(loss.is_finite(), "loss must remain finite: {loss}");
    }
    fandhe_ai::set_cuda_graph_step_enabled(false);
}

/// 受け入れ条件 (a) の**機械比較版**（codex-review P2 指摘対応。イシュー
/// #1349）: `eager_baseline`／`graph_capture` は出力を印字するのみで
/// 自動比較しない（目視・スクリプト任せ）ため、2 GPU 搭載機では本テスト
/// 単独で「損失・最終パラメータが bit 同一」を `assert_eq!` により機械
/// 検証する。
///
/// **前提**: opt-in はプロセス内最初の CUDA デバイス初期化より前に固定
/// される必要がある（`fandhe_ai::set_cuda_graph_step_enabled` doc・本
/// ファイル冒頭コメント参照）ため、同一プロセス内で両経路を機械比較
/// するには異なる ordinal が要る: ordinal 0 で opt-in OFF（eager）を
/// 先に完走させたあと opt-in を ON にし、ordinal 1（この時点で初めて
/// 初期化される）で capture 経路を走らせる。
///
/// **単一 GPU 環境（DGX Spark GB10 等）では成立しない**（`CudaDevice::
/// device_count()` が 2 未満なら、コンテキスト非構築の軽量プローブ〈
/// `tape_cuda_cache_bench.rs` と同じ理由で選定〉で検出し早期 return
/// する。単一 GPU 環境の受け入れ検証は既存の 2 プロセス比較
/// （`eager_baseline`／`graph_capture`。モジュール冒頭コメント）に委ねる）。
#[test]
#[ignore = "CUDA 実機（2 GPU 構成）必須。単独プロセスで実行すること（opt-in をプロセスワイドに             変更するため）。単一 GPU 環境では device_count() < 2 のため早期 return する"]
fn graph_capture_matches_eager_baseline_bit_identical_across_two_gpus() {
    let device_count = CudaDevice::device_count().unwrap_or(0);
    if device_count < 2 {
        eprintln!(
            "device_count()={device_count} < 2: 単一 GPU 環境のため本テストの機械比較は成立しない              （2 プロセス構成の eager_baseline／graph_capture に委ねる）。早期 return する。"
        );
        return;
    }

    assert!(
        !fandhe_ai::cuda_graph_step_enabled(),
        "ordinal 0 の eager baseline は opt-in OFF のまま初期化する必要がある"
    );
    let (eager_log, eager_params) = train_on_cuda(0, STEPS, LR);

    fandhe_ai::set_cuda_graph_step_enabled(true);
    let (graph_log, graph_params) = train_on_cuda(1, STEPS, LR);
    fandhe_ai::set_cuda_graph_step_enabled(false);

    print_bit_identity_report("eager (opt-in OFF, ordinal 0)", &eager_log, &eager_params);
    print_bit_identity_report(
        "graph capture (opt-in ON, ordinal 1)",
        &graph_log,
        &graph_params,
    );

    assert_eq!(
        eager_log.len(),
        graph_log.len(),
        "loss 列の長さが一致しないはず（STEPS は共通の定数）"
    );
    for (i, (e, g)) in eager_log.iter().zip(graph_log.iter()).enumerate() {
        assert_eq!(
            e.to_bits(),
            g.to_bits(),
            "step[{i}] の loss が bit 同一でない: eager={e:#010x?}／graph={g:#010x?}"
        );
    }

    assert_eq!(
        eager_params.len(),
        graph_params.len(),
        "パラメータ列の個数が一致しないはず（同一モデル構成）"
    );
    for (p, (ep, gp)) in eager_params.iter().zip(graph_params.iter()).enumerate() {
        let e_contig = ep.contiguous();
        let g_contig = gp.contiguous();
        let e_slice = e_contig.as_slice().unwrap_or(&[]);
        let g_slice = g_contig.as_slice().unwrap_or(&[]);
        assert_eq!(
            e_slice.len(),
            g_slice.len(),
            "param[{p}] の要素数が一致しないはず"
        );
        for (i, (ev, gv)) in e_slice.iter().zip(g_slice.iter()).enumerate() {
            assert_eq!(
                ev.to_bits(),
                gv.to_bits(),
                "param[{p}][{i}] が bit 同一でない: eager={ev:#010x?}／graph={gv:#010x?}"
            );
        }
    }
}
