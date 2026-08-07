//! 学習途中状態のチェックポイント検証（イシュー #198・親イシュー #196・REQ-7）。
//!
//! 親イシュー #196 の受け入れ条件「save→load ラウンドトリップで bit 一致」を
//! 学習文脈で検証する。`autodiff::nn::Linear`（`Tape` 経由の forward/backward）・
//! `autodiff::optim::Sgd` を使い、学習途中の重みを `onnx_interop::st_save` で
//! 書き出し・`onnx_interop::st_load` で読み戻す。
//!
//! **optimizer 状態はスコープ外**: イシュー #198 本文の明記事項として、
//! momentum バッファ（`Sgd` の `velocity`）・AdamW の m/v のチェックポイント
//! 保存・復元は初期スコープ外とする（`.claude/rules/out-of-scope-tracking.md`）。
//! そのため「再開が中断なし学習と等価」を主張するテスト
//! （`resume_from_checkpoint_matches_uninterrupted_training`）は
//! **momentum なしの stateless SGD**（`SgdConfig::new` の既定値 `momentum:
//! 0.0` → `crates/autodiff/src/optim/sgd.rs::Sgd::velocity` が `None` の
//! まま＝optimizer が一切の状態を保持しない）で構成する。この構成では
//! 「10 step 学習して保存 → 新規 `Sgd` インスタンスで再開」と「20 step
//! 連続学習」が数学的に完全に等価であり、momentum/AdamW の状態込み再開の
//! 主張はしない（`weights_round_trip_bit_exact_even_under_momentum_training`
//! が momentum ありの学習でも重み往復自体は bit 一致することを別途確認する
//! が、そちらも optimizer 状態の保存・再開等価性は主張しない）。
//!
//! **キー命名・shape**: PyTorch 慣習キー（`fc1.weight` 等）を使うが、
//! `st_save`/`st_load` の「暗黙アダプタを設けない」契約（`st_save.rs`
//! モジュール冒頭ドキュメント参照）により転置は行わない。shape は
//! 自作コアの `Linear` が保持するそのままの形（`[in_features,
//! out_features]`）で往復する。PyTorch 実ファイルとの互換往復（転置込み）
//! の検証は `tests/st_save.rs`（#197）が fixture で担当済みであり、本
//! ファイルでは扱わない。
//!
//! **決定的シード**: 重み初期化（`Linear::new` の `seed` 引数）・データ生成
//! （`bench_harness::rng::Xorshift64Star`）を固定シードで駆動する
//! （`coding-rust.md`「学習系回帰テストには決定的シード設定ユーティリティを
//! 使う」）。`crates/autodiff/tests/nn_train_convergence.rs` と同じ
//! 「重み初期化用シードとデータ生成用シードを分離する」方針を踏襲する。
//!
//! **数値判定の規律**: 本ファイルの全 assert は `to_bits()` の bit 一致
//! であり、tolerance（許容誤差）は新設・緩和しない
//! （`coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で
//! 緩和しない」の精神を学習チェックポイント検証にも適用する）。
//!
//! 全テストは CPU のみで完結するため実機（CUDA/Metal）非依存であり
//! `#[ignore]` 分離は行わない。

use std::collections::HashMap;

use autodiff::Tape;
use autodiff::nn::Linear;
use autodiff::nn::activation::Relu;
use autodiff::optim::{Sgd, SgdConfig};
use onnx_interop::st_load::load_safetensors_f32;
use onnx_interop::st_save::save_safetensors_f32;
use tensor_core::Tensor;

use bench_harness::rng::Xorshift64Star;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

// `nn_train_convergence.rs` と同じく、重み初期化用シード（SEED_L1/SEED_L2）
// とデータ生成用シード（SEED_DATA）を分離する（同一シード使い回しによる
// 相関を避ける）。
const SEED_DATA: u64 = 0xBEEF_CAFE;
const SEED_L1: u64 = 0xAAAA_AAAA;
const SEED_L2: u64 = 0xBBBB_BBBB;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn gen_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// テスト内で使い回す一時ディレクトリ（プロセス ID・スレッド ID・
/// 呼び出し元指定タグで分離。`tests/st_save.rs::save_to_file_produces_loadable_file`
/// の先例を踏襲し、CWD 変更は行わない）。呼び出し側で後始末する。
fn make_tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "onnx-interop-st-checkpoint-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("test fixture: 一時ディレクトリの作成に失敗した");
    dir
}

/// 2 層 MLP（`Linear(D_IN→D_HIDDEN)` → `ReLU` → `Linear(D_HIDDEN→D_OUT)`）を
/// `steps` 回フルバッチ SGD で学習し、各 step の loss（`(値,
/// to_bits())`）を記録しつつ、最終的な `l1`/`l2` を返す。
///
/// `sgd`（momentum の有無を含むハイパーパラメータ）は呼び出し元が構築した
/// インスタンスを渡す（`sgd.step()` は同一インスタンスをまたいで momentum
/// バッファを保持するため、再開シナリオでは呼び出し元が「新規インスタンス
/// を渡す」か「既存インスタンスを渡し続ける」かを選べる設計にしてある）。
fn train_steps(
    l1: Linear,
    l2: Linear,
    x_data: &Tensor<f32>,
    y_data: &Tensor<f32>,
    sgd: &mut Sgd,
    steps: usize,
) -> (Linear, Linear, Vec<(f32, u32)>) {
    let relu = Relu;
    let mut l1 = l1;
    let mut l2 = l2;
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape = Tape::new();
        let x = tape.var(x_data);
        let y = tape.var(y_data);

        let l1v = l1.bind(&tape);
        let l2v = l2.bind(&tape);

        let h1 = l1v.forward(&x).unwrap();
        let a1 = relu.forward(&h1);
        let h2 = l2v.forward(&a1).unwrap();
        let loss = h2.mse_loss(&y).unwrap();

        let loss_value = scalar(&loss.to_tensor());
        let grads = tape.backward(&loss).unwrap();

        let l1_weight_grad = grads.get(&l1v.weight).unwrap().unwrap();
        let l1_bias_grad = grads
            .get(l1v.bias.as_ref().expect("test fixture: bias=true で構築"))
            .unwrap()
            .unwrap();
        let l2_weight_grad = grads.get(&l2v.weight).unwrap().unwrap();
        let l2_bias_grad = grads
            .get(l2v.bias.as_ref().expect("test fixture: bias=true で構築"))
            .unwrap()
            .unwrap();

        let l1_bias = l1.bias().expect("test fixture: bias=true で構築").clone();
        let l2_bias = l2.bias().expect("test fixture: bias=true で構築").clone();

        let params = [l1.weight(), &l1_bias, l2.weight(), &l2_bias];
        let grads_slice = [l1_weight_grad, l1_bias_grad, l2_weight_grad, l2_bias_grad];
        let updated = sgd.step(&params, &grads_slice).unwrap();

        l1 = Linear::from_parameters(updated[0].clone(), Some(updated[1].clone()))
            .expect("test fixture: shape は sgd.step() で保存されている");
        l2 = Linear::from_parameters(updated[2].clone(), Some(updated[3].clone()))
            .expect("test fixture: shape は sgd.step() で保存されている");

        log.push((loss_value, loss_value.to_bits()));
    }

    (l1, l2, log)
}

/// `l1`/`l2` の 4 パラメータを PyTorch 慣習キーで集約する
/// （`fc1.weight`/`fc1.bias`/`fc2.weight`/`fc2.bias`）。
fn params_to_map(l1: &Linear, l2: &Linear) -> HashMap<String, Tensor<f32>> {
    let mut map = HashMap::new();
    map.insert("fc1.weight".to_string(), l1.weight().clone());
    map.insert(
        "fc1.bias".to_string(),
        l1.bias().expect("test fixture: bias=true で構築").clone(),
    );
    map.insert("fc2.weight".to_string(), l2.weight().clone());
    map.insert(
        "fc2.bias".to_string(),
        l2.bias().expect("test fixture: bias=true で構築").clone(),
    );
    map
}

fn assert_bits_eq(key: &str, expected: &Tensor<f32>, actual: &Tensor<f32>) {
    assert_eq!(
        expected.shape(),
        actual.shape(),
        "shape mismatch for key {key}"
    );
    let expected_dense = expected.contiguous();
    let actual_dense = actual.contiguous();
    let expected_slice = expected_dense
        .as_slice()
        .expect("test fixture: contiguous() 後は as_slice() が Some のはず");
    let actual_slice = actual_dense
        .as_slice()
        .expect("test fixture: contiguous() 後は as_slice() が Some のはず");
    for (i, (a, b)) in expected_slice.iter().zip(actual_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch for key {key} at index {i}"
        );
    }
}

// --- ケース 1: 学習途中チェックポイントの save→load bit 一致 ---

#[test]
fn mid_training_checkpoint_round_trips_bit_exact() {
    let (x_data, y_data) = gen_data(SEED_DATA);
    let l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1).expect("test fixture: shape は事前に妥当");
    let l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).expect("test fixture: shape は事前に妥当");
    let mut sgd = Sgd::new(SgdConfig::new(0.05)).expect("test fixture: config は事前に妥当");

    let (l1, l2, _log) = train_steps(l1, l2, &x_data, &y_data, &mut sgd, 10);
    let params = params_to_map(&l1, &l2);

    let tmp_dir = make_tmp_dir("mid-training");
    let out_path = tmp_dir.join("checkpoint.safetensors");
    save_safetensors_f32(&out_path, &params).expect("チェックポイント書き出しに失敗した");
    let reloaded = load_safetensors_f32(&out_path).expect("チェックポイント再ロードに失敗した");

    for key in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"] {
        assert_bits_eq(key, &params[key], &reloaded[key]);
    }

    std::fs::remove_dir_all(&tmp_dir).ok();
}

// --- ケース 2: チェックポイントからの再開が中断なし学習と等価（stateless SGD） ---

/// 経路 A（20 step 連続学習）と経路 B（10 step 学習 → save → load →
/// 新規 `Sgd` で残り 10 step）の step 11〜20 の loss・最終重みが bit 一致
/// することを確認する。momentum なし（stateless）の SGD であるため、
/// 「新規 `Sgd` インスタンスで再開」しても数学的に等価となる
/// （モジュール冒頭ドキュメント参照）。
#[test]
fn resume_from_checkpoint_matches_uninterrupted_training() {
    const LR: f32 = 0.05;

    // 経路 A: 20 step 連続学習。
    let (x_data, y_data) = gen_data(SEED_DATA);
    let l1_a =
        Linear::new(D_IN, D_HIDDEN, true, SEED_L1).expect("test fixture: shape は事前に妥当");
    let l2_a =
        Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).expect("test fixture: shape は事前に妥当");
    let mut sgd_a = Sgd::new(SgdConfig::new(LR)).expect("test fixture: config は事前に妥当");
    let (l1_a, l2_a, log_a) = train_steps(l1_a, l2_a, &x_data, &y_data, &mut sgd_a, 20);

    // 経路 B: 10 step 学習 → save → load → Linear::from_parameters で再構築
    // → 新規 Sgd（momentum なし＝状態を持たないため数学的に等価）で残り
    // 10 step。
    let l1_b =
        Linear::new(D_IN, D_HIDDEN, true, SEED_L1).expect("test fixture: shape は事前に妥当");
    let l2_b =
        Linear::new(D_HIDDEN, D_OUT, true, SEED_L2).expect("test fixture: shape は事前に妥当");
    let mut sgd_b_phase1 = Sgd::new(SgdConfig::new(LR)).expect("test fixture: config は事前に妥当");
    let (l1_b, l2_b, log_b_phase1) =
        train_steps(l1_b, l2_b, &x_data, &y_data, &mut sgd_b_phase1, 10);
    assert_eq!(log_b_phase1.len(), 10);

    let params_b = params_to_map(&l1_b, &l2_b);
    let tmp_dir = make_tmp_dir("resume");
    let out_path = tmp_dir.join("checkpoint.safetensors");
    save_safetensors_f32(&out_path, &params_b).expect("チェックポイント書き出しに失敗した");
    let reloaded = load_safetensors_f32(&out_path).expect("チェックポイント再ロードに失敗した");

    let l1_resumed = Linear::from_parameters(
        reloaded["fc1.weight"].clone(),
        Some(reloaded["fc1.bias"].clone()),
    )
    .expect("test fixture: shape は往復で保存されている");
    let l2_resumed = Linear::from_parameters(
        reloaded["fc2.weight"].clone(),
        Some(reloaded["fc2.bias"].clone()),
    )
    .expect("test fixture: shape は往復で保存されている");

    // stateless（momentum なし）のため、再開時に新規 Sgd インスタンスを
    // 使っても経路 A の 20 step 連続学習と数学的に等価になる。
    let mut sgd_b_phase2 = Sgd::new(SgdConfig::new(LR)).expect("test fixture: config は事前に妥当");
    let (l1_resumed, l2_resumed, log_b_phase2) = train_steps(
        l1_resumed,
        l2_resumed,
        &x_data,
        &y_data,
        &mut sgd_b_phase2,
        10,
    );

    let log_a_tail_bits: Vec<u32> = log_a[10..].iter().map(|(_, bits)| *bits).collect();
    let log_b_phase2_bits: Vec<u32> = log_b_phase2.iter().map(|(_, bits)| *bits).collect();
    assert_eq!(
        log_a_tail_bits, log_b_phase2_bits,
        "経路 A（step 11-20）と経路 B（再開後の 10 step）の loss 系列が bit 一致しない\
         （チェックポイントからの再開が中断なし学習と等価であることの検証に失敗）"
    );

    let params_a = params_to_map(&l1_a, &l2_a);
    let params_resumed = params_to_map(&l1_resumed, &l2_resumed);
    for key in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"] {
        assert_bits_eq(key, &params_a[key], &params_resumed[key]);
    }

    std::fs::remove_dir_all(&tmp_dir).ok();
}

// --- ケース 3: momentum あり学習でも重みの save→load 自体は bit 一致 ---

/// momentum あり SGD で学習した重みも save→load で bit 一致することを
/// 確認する（重みの往復のみを主張する。optimizer 状態〈momentum
/// バッファ〉の保存・再開等価性は主張しない。モジュール冒頭ドキュメント
/// 参照）。
#[test]
fn weights_round_trip_bit_exact_even_under_momentum_training() {
    let (x_data, y_data) = gen_data(SEED_DATA.wrapping_add(1));
    let l1 = Linear::new(D_IN, D_HIDDEN, true, SEED_L1.wrapping_add(1))
        .expect("test fixture: shape は事前に妥当");
    let l2 = Linear::new(D_HIDDEN, D_OUT, true, SEED_L2.wrapping_add(1))
        .expect("test fixture: shape は事前に妥当");
    let mut sgd = Sgd::new(SgdConfig::new(0.05).with_momentum(0.9))
        .expect("test fixture: config は事前に妥当");

    let (l1, l2, _log) = train_steps(l1, l2, &x_data, &y_data, &mut sgd, 10);
    let params = params_to_map(&l1, &l2);

    let tmp_dir = make_tmp_dir("momentum-weights");
    let out_path = tmp_dir.join("checkpoint.safetensors");
    save_safetensors_f32(&out_path, &params).expect("チェックポイント書き出しに失敗した");
    let reloaded = load_safetensors_f32(&out_path).expect("チェックポイント再ロードに失敗した");

    for key in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"] {
        assert_bits_eq(key, &params[key], &reloaded[key]);
    }

    std::fs::remove_dir_all(&tmp_dir).ok();
}
