//! `cuda_graph_step_bit_identity.rs`・`cuda_graph_step_two_gpu_bit_identity.rs`
//! （イシュー #1349）が共有する学習ループ・出力ヘルパー。
//!
//! **分離した理由（codex-review P2 指摘対応・PR #1390）**: 2 GPU 構成の
//! 機械比較テスト（`graph_capture_matches_eager_baseline_bit_identical_
//! across_two_gpus`）は「opt-in OFF で開始する」前提を持つが、同じ
//! バイナリ内の他テスト（`graph_capture`・
//! `graph_capture_completes_training_loop_without_error`）は opt-in を
//! ON のまま／プロセスワイドに変更する前提を持つ。`cargo test
//! graph_capture` のような部分一致フィルタで 3 つとも同一プロセス・
//! 並行スレッドで選ばれてしまうと、いずれかのテストの前提
//! （「opt-in はプロセス内最初の CUDA デバイス初期化より前に固定」・
//! 「プロセスワイドな opt-in フラグが実行中に他スレッドから変わらない」）
//! が崩れる。2 GPU テストを別ファイル（＝別テストバイナリ・別プロセス）
//! へ分離することで、フィルタの部分一致に関わらず両者が同一プロセスで
//! 選ばれることを構造的になくす（`crates/backend-cuda/tests/
//! graph_capture_real_device.rs`／`graph_capture_real_device_optin_off.rs`
//! を opt-in ON／OFF で別ファイルに分けた既存パターンと同じ方針。
//! `docs/backend-cuda-graph-step-capture-design.md` 9 節参照）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

pub const BATCH: usize = 4;
pub const D_IN: usize = 8;
pub const D_HIDDEN: usize = 16;
pub const D_OUT: usize = 4;
pub const STEPS: usize = 10;
pub const LR: f32 = 0.05;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

pub fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

pub fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

pub fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// [`train_on_cuda`] の戻り値型（clippy `type_complexity` 回避。
/// `(loss 列, 各 step の入力勾配（d(loss)/d(x)）列, 各 step の重み勾配
/// （`Tape::param_grads_to_host` の戻り値そのもの。パラメータごとの
/// `Vec<Tensor<f32>>`）列, 各 step 完了直後のパラメータ列, 最終
/// パラメータ列)`。フィールドの意味は [`train_on_cuda`] doc コメント
/// 参照）。
pub type TrainOnCudaResult = (
    Vec<f32>,
    Vec<Tensor<f32>>,
    Vec<Vec<Tensor<f32>>>,
    Vec<Vec<Tensor<f32>>>,
    Vec<Tensor<f32>>,
);

/// `device_param_store_train.rs::train_with_device_param_store` の CUDA
/// 版。各 step の loss（`f32` そのまま。ビット比較は呼び出し元が
/// `to_bits()` で行う）・各 step の入力勾配（`per_step_dinput`）・各
/// step の重み勾配（`per_step_grads`。#1480）・各 step の完了直後に
/// ホストへ同期したパラメータ列（`per_step_params`）・最終的にホストへ
/// 同期したパラメータ列（`final_params`）の 5 つを返す。
///
/// `ordinal` を引数化している理由（codex-review P2 指摘対応。イシュー
/// #1349）: opt-in（`FANDHE_AI_CUDA_GRAPH_STEP`／
/// `set_cuda_graph_step_enabled`）はプロセス内最初の CUDA デバイス
/// 初期化より前に固定される必要があるため（両呼び出し元ファイルの
/// 冒頭コメント参照）、同一プロセス内で「opt-in OFF の基準値」と
/// 「opt-in ON の capture 経路」を機械比較するには**異なる ordinal**を
/// 使う必要がある（`cuda_graph_step_two_gpu_bit_identity.rs` 参照）。
///
/// **各 step のパラメータをホストへ同期する理由（codex-review P2
/// 指摘対応・PR #1390 再々修正）**: 旧稿は最終 step 完了後のパラメータ
/// のみを返しており、途中の step で一時的に発生し最終値には現れない
/// ビット差異（例えば SGD 更新の中間丸め誤差が後続 step で偶然打ち消し
/// 合う場合）を検出できなかった。`step_device_param_store` の直後に毎回
/// `sync_device_param_store_to_host` を呼ぶことで、`STEPS` 回すべての
/// パラメータ状態を比較対象にする。
///
/// **各 step の入力勾配 `d(loss)/d(x)` を比較対象に加える理由
/// （codex-review P2 指摘対応・PR #1390 是正）**: `x`（学習データ。
/// `tape.var(&x_data)`）は `Op::ResidentLeaf` ではない通常の `Var` の
/// ため、`Gradients::get(&x)` で公開 API から直接取得できる。逆伝播は
/// MSE backward（`CudaMse`）→ `Linear` 2 層分の d_input（`ops.
/// gemm_resident_lhs` 経由。resident weight を用いる）→ ReLU backward
/// を経て `x` まで届くため、本比較は「勾配計算そのもの」（elementwise・
/// MSE・rmsnorm 融合等、本 PR で capture 排他へ新たに参加させた
/// `CudaElementwise`／`CudaMse`／`CudaRmsNorm`／`CudaSoftmax` の起動
/// 経路を含む）が capture 経路・非 capture 経路で bit 同一であることを
/// 直接検証する。
///
/// **重み自体の勾配を per-step で比較対象へ加える（イシュー #1480。
/// 旧稿は「引き続き含まない」としていたが、依存イシュー #1479 が
/// 読み出し手段を追加したため本節を全面的に書き換える）**: `weight`
/// は `Op::ResidentLeaf`（`optim::device_store::ResidentLeaf`）であり、
/// このハンドルは意図的に `node_id` を公開しない（`ResidentLeaf` の
/// ドキュメンテーションコメント参照）ため、`Gradients::get()` を呼べる
/// `Var` を外部（`facade` 利用者側）から構築する手段は依然として
/// 存在しない。#1479 はこの制約を回避するのではなく、
/// `DeviceParamStore` 自身が `pending`（`sync_to_host` と同じ登録順）
/// を経由して直接読み出す新規公開 API（`facade::Tape::
/// resident_grads_to_host`〈strict 版。resident staging 限定〉／
/// `Tape::param_grads_to_host`〈統合版。resident 未充填 slot は
/// `grads.get(...)` へフォールバック〉）を追加した。
///
/// **CUDA では統合版 `param_grads_to_host` を使う理由**: CUDA は
/// `gemm_fp32_strict_into` 未実装のため `Op::LinearResident` の重み
/// 勾配は resident staging（`GradStaging`）へ書き込まれず、backward が
/// ホスト経路（`fill_resident_weight_grad` が `Ok(false)` を返し
/// `ops.gemm_fp32_strict` へ委譲。`fandhe_ai_autodiff::grad::vjp` の
/// `Op::LinearResident` 分岐コメント参照）で `Gradients` へ寄与を
/// 書き込む。strict 版 `resident_grads_to_host` を CUDA で呼ぶと
/// `resident_grad_capability` が `Some(false)` のため必ず
/// `BackendError::Unsupported` を返す（`crates/facade/tests/
/// device_param_store_backend_parity.rs::assert_grad_readout_contract`
/// が別途検証）。統合版 `param_grads_to_host` は全 slot がこの
/// `grads.get(...)` フォールバック経路を通ることで `Ok` を返す。
///
/// **呼び出し窓**: `backward_device_param_store`（backward 実行）の
/// 直後・`step_device_param_store`（`pending` を消費する SGD 更新）の
/// 前に限る（`DeviceParamStore::param_grads_to_host` doc の「呼び出し
/// 窓」契約）。この窓を外すと `BackendError::InvalidArgument` を返す。
///
/// **CUDA での本比較が検証する内容**: CUDA は resident 経路
/// （`GradStaging` へのデバイス直接書き込み）に到達しないため、本比較
/// は「backward が計算した重み勾配そのもの（ホスト経路で `Gradients`
/// へ書き込まれた値）が capture 経路・非 capture 経路で bit 同一」で
/// あることの直接検証であり、`GradStaging` の D2H 検証ではない（CPU
/// のみ resident 経路が成立する。`docs/device-resident-update-design.md`
/// 追補 #1479 参照）。既存の `dinput`（d(loss)/d(x)）比較が「backward
/// の入力側端点」を見るのに対し、本比較は「backward のパラメータ側
/// 端点（SGD 更新の入力そのもの）」を見る点で相補的である。上記の
/// per-step パラメータ比較（更新後の重み）と合わせて、SGD 更新の
/// 入力・出力の両端が capture 経路・非 capture 経路で bit 同一である
/// ことを揃って検証する。
pub fn train_on_cuda(ordinal: usize, steps: usize, lr: f32) -> TrainOnCudaResult {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr);
    let mut log = Vec::with_capacity(steps);
    let mut per_step_dinput = Vec::with_capacity(steps);
    let mut per_step_grads = Vec::with_capacity(steps);
    let mut per_step_params = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape =
            fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();

        // codex-review P2 指摘対応（PR #1390 是正）: `x` は `Op::
        // ResidentLeaf` ではない通常の `Var` のため `Gradients::get`
        // で d(loss)/d(x) を公開 API から直接取得できる（`train_on_cuda`
        // doc コメント「各 step の入力勾配 d(loss)/d(x) を比較対象に
        // 加える理由」参照）。`step_device_param_store`（SGD 更新）より
        // 前に取得する: 逆伝播直後の勾配値そのものを記録するためで
        // あり、SGD 更新自体は `x` に触れないため順序を入れ替えても
        // 値は変わらないが、「勾配計算の結果」であることを本文脈で
        // 明確にするためこの位置に置く。
        let dinput = grads
            .get(&x)
            .expect("x は同一 tape・同一 epoch 内の Var のため TapeMismatch は起きないはず")
            .expect(
                "x は loss へ到達する経路（MSE→Linear→ReLU→Linear）上にあるため \
                 d(loss)/d(x) は必ず計算されるはず",
            )
            .clone();
        per_step_dinput.push(dinput);

        // イシュー #1480: backward 直後・`step_device_param_store`
        // （`pending` を消費する SGD 更新）の前という呼び出し窓内で
        // 各 step の重み勾配を読み出す（`train_on_cuda` doc コメント
        // 「重み自体の勾配を per-step で比較対象へ加える」参照）。
        // strict 版 `resident_grads_to_host` は CUDA で必ず
        // `Unsupported` を返す設計（`gemm_fp32_strict_into` 未実装）
        // のため、統合版 `param_grads_to_host`（resident 未充填 slot は
        // `grads.get(...)` フォールバック）を使う。
        let step_grads = tape
            .param_grads_to_host(&store, &grads)
            .expect("param_grads_to_host: backward 直後・step 前の呼び出し窓内で呼んでいるはず");
        per_step_grads.push(step_grads);

        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();

        // codex-review P2 指摘対応（PR #1390 再々修正）: この step の
        // 完了直後にパラメータをホストへ同期し記録する（doc コメント
        // 「各 step のパラメータをホストへ同期する理由」参照）。
        let step_synced = tape.sync_device_param_store_to_host(&store).unwrap();
        per_step_params.push(step_synced);
    }

    let final_tape =
        fandhe_ai::tape_for(Device::Cuda(ordinal)).expect("CUDA device must be available");
    let final_params = final_tape.sync_device_param_store_to_host(&store).unwrap();
    (
        log,
        per_step_dinput,
        per_step_grads,
        per_step_params,
        final_params,
    )
}

/// loss 列・各 step の入力勾配（d(loss)/d(x)）列・各 step 完了直後の
/// パラメータ列・最終パラメータを `to_bits()` の 16 進表現で標準出力へ
/// 出す（プロセス間比較のための決定的なテキスト表現。浮動小数点の
/// 表示誤差を避けるため `{:?}`／`{}` ではなくビット表現を使う）。
///
/// `per_step_params` 引数の追加（codex-review P2 指摘対応・PR #1390
/// 再々修正）: `train_on_cuda` doc コメント「各 step のパラメータを
/// ホストへ同期する理由」参照。最終値のみでは検出できない中間 step の
/// ビット差異を目視比較でも追えるようにする。
///
/// `per_step_dinput` 引数の追加（codex-review P2 指摘対応・PR #1390
/// 是正）: `train_on_cuda` doc コメント「各 step の入力勾配
/// d(loss)/d(x) を比較対象に加える理由」参照。`step[{i}].dinput[{j}]`
/// ラベルで出力する（`param` と衝突しない専用プレフィックス）。
///
/// `per_step_grads` 引数の追加（イシュー #1480）: `train_on_cuda` doc
/// コメント「重み自体の勾配を per-step で比較対象へ加える」参照。
/// `step[{step}].grad[{p}][{j}]` ラベルで出力する（`dinput`／`param`
/// いずれとも衝突しない専用プレフィックス）。
pub fn print_bit_identity_report(
    label: &str,
    log: &[f32],
    per_step_dinput: &[Tensor<f32>],
    per_step_grads: &[Vec<Tensor<f32>>],
    per_step_params: &[Vec<Tensor<f32>>],
    final_params: &[Tensor<f32>],
) {
    println!("=== cuda_graph_step_bit_identity: {label} ===");
    for (i, loss) in log.iter().enumerate() {
        println!("step[{i}].loss.bits = {:#010x}", loss.to_bits());
    }
    for (step, dinput) in per_step_dinput.iter().enumerate() {
        let contiguous = dinput.contiguous();
        let slice = contiguous.as_slice().unwrap_or(&[]);
        for (j, v) in slice.iter().enumerate() {
            println!("step[{step}].dinput[{j}].bits = {:#010x}", v.to_bits());
        }
    }
    for (step, grads) in per_step_grads.iter().enumerate() {
        for (p, tensor) in grads.iter().enumerate() {
            let contiguous = tensor.contiguous();
            let slice = contiguous.as_slice().unwrap_or(&[]);
            for (j, v) in slice.iter().enumerate() {
                println!("step[{step}].grad[{p}][{j}].bits = {:#010x}", v.to_bits());
            }
        }
    }
    for (step, params) in per_step_params.iter().enumerate() {
        for (p, tensor) in params.iter().enumerate() {
            let contiguous = tensor.contiguous();
            let slice = contiguous.as_slice().unwrap_or(&[]);
            for (i, v) in slice.iter().enumerate() {
                println!("step[{step}].param[{p}][{i}].bits = {:#010x}", v.to_bits());
            }
        }
    }
    for (p, tensor) in final_params.iter().enumerate() {
        let contiguous = tensor.contiguous();
        let slice = contiguous.as_slice().unwrap_or(&[]);
        for (i, v) in slice.iter().enumerate() {
            println!("final.param[{p}][{i}].bits = {:#010x}", v.to_bits());
        }
    }
}
