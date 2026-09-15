//! `compat::Sequential` の train／eval モード（[`Sequential::set_training`]／
//! [`Sequential::train`]／[`Sequential::eval`]／[`Sequential::training`]）・
//! `named_parameters`（イシュー #1758）の公開 API 契約テストを、
//! `fandhe_ai`（facade）経由でのみ検証する。
//!
//! #1758 時点では本ファイルの検証はモード切替が数値経路に一切影響
//! しないことの固定（Linear・活性化関数はいずれもモード非依存）
//! だった。イシュー #1603 で [`Sequential::add_dropout`] が追加され、
//! `Dropout` が本クレート内実装で唯一 `training` フラグを実際に
//! 保持する層になったため、末尾に `add_dropout` 固有のモード契約
//! テスト（eval は恒等・train は決定的・`dyn Module` 経由の伝播が
//! `bind().forward()` にも及ぶ・`p` 範囲外は `Err`）を追加した。
//! それ以外の検証は CPU 上の bit 完全一致に限定する（新規 `Op`／
//! `BackendOps`／GPU カーネルは追加していないため）。CUDA／Metal への
//! 申し送りは不要（数値経路に一切触れないため）。

use std::sync::Mutex;

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tensor, tape};

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)
        .unwrap()
}

/// 初期値 `training()==true`・`eval()`→`false`・`train()`→`true`・
/// `set_training(b)` 往復。
#[test]
fn training_mode_toggle() {
    let mut model = build_model();
    assert!(model.training(), "既定は PyTorch の初期値と揃え true");

    model.eval();
    assert!(!model.training());

    model.train();
    assert!(model.training());

    model.set_training(false);
    assert!(!model.training());
    model.set_training(true);
    assert!(model.training());
}

/// index 接頭辞契約: `Linear→ReLU→Linear` で
/// `["0.weight","0.bias","2.weight","2.bias"]`（活性化を含む index。
/// PyTorch `nn.Sequential` と同じ規約）。
#[test]
fn named_parameters_index_prefix_contract() {
    let model = build_model();
    let params = model.named_parameters();
    let names: Vec<&str> = params.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["0.weight", "0.bias", "2.weight", "2.bias"]);
}

/// 順序契約: `named_parameters()` の tensor 列と
/// `trainable_parameters()` を要素ごとに `std::ptr::eq` で一致確認する
/// （`Sequential::named_parameters` doc「順序契約」参照）。
#[test]
fn named_parameters_order_matches_trainable_parameters() {
    let model = build_model();
    let named = model.named_parameters();
    let trainable = model.trainable_parameters();
    assert_eq!(named.len(), trainable.len());
    for ((_, named_tensor), trainable_tensor) in named.iter().zip(trainable.iter()) {
        assert!(std::ptr::eq(*named_tensor, *trainable_tensor));
    }
}

/// bit 同一（parity の代替）: 同一入力に対し `eval()` 前後・`train()`
/// 前後で `predict`（tape 不要経路）の出力が bit 完全一致する
/// （本 issue はモード依存層を一切導入しないため、`set_training` は
/// 数値経路に影響しない契約を固定する）。
#[test]
fn predict_output_unaffected_by_training_mode() {
    let mut model = build_model();
    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();

    let out_train = model.predict(&x).unwrap();

    model.eval();
    let out_eval = model.predict(&x).unwrap();

    model.train();
    let out_train_again = model.predict(&x).unwrap();

    assert_eq!(out_train.as_slice(), out_eval.as_slice());
    assert_eq!(out_train.as_slice(), out_train_again.as_slice());
}

/// bit 同一（parity の代替）: `bind → forward → mse_loss → backward`
/// の勾配が `eval()` 前後で bit 完全一致する。
#[test]
fn backward_gradients_unaffected_by_training_mode() {
    let mut model = build_model();
    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();
    let y = Tensor::new(vec![0.0_f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap();

    let grads_train = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let yv = t.var(&y);
        let pred = bound.forward(&t, &xv).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let grads = t.backward(&loss).unwrap();
        let refs = bound.trainable_grads(&grads).unwrap();
        refs.into_iter().cloned().collect::<Vec<_>>()
    };

    model.eval();

    let grads_eval = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let yv = t.var(&y);
        let pred = bound.forward(&t, &xv).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let grads = t.backward(&loss).unwrap();
        let refs = bound.trainable_grads(&grads).unwrap();
        refs.into_iter().cloned().collect::<Vec<_>>()
    };

    assert_eq!(grads_train.len(), grads_eval.len());
    for (a, b) in grads_train.iter().zip(grads_eval.iter()) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
}

// --- `add_dropout`（イシュー #1603）のモード契約 -------------------------

/// グローバル RNG 状態を書き換えるテストを直列化する
/// （`crates/facade/tests/rng_tensor_generation.rs` と同型）。
fn dropout_test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn build_model_with_dropout(p: f32) -> Result<Sequential, AutodiffError> {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_dropout(p)?
        .add_linear(8, 2, /* seed = */ 43)
}

/// (a) eval モードの `add_dropout` は恒等写像のため、`Dropout` 層なし
/// の同重みモデル（`build_model`。`add_linear` の seed を揃えている）
/// の `predict` と bit 完全一致する。
#[test]
fn dropout_model_eval_predict_matches_model_without_dropout() {
    let mut with_dropout = build_model_with_dropout(0.5).unwrap();
    with_dropout.eval();
    let baseline = build_model();

    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();
    let out_with_dropout = with_dropout.predict(&x).unwrap();
    let out_baseline = baseline.predict(&x).unwrap();

    assert_eq!(out_with_dropout.as_slice(), out_baseline.as_slice());
}

/// (b) train モードの `add_dropout` は `manual_seed` で同一マスクを
/// 再現すれば run-to-run bit 完全一致し、tape 不要経路（`predict`）と
/// tape 経路（`bind().forward()`）も bit 完全一致する（`predict_tape_free
/// ≡ predict_via_tape` 不変条件が train モードでも成立することの確認。
/// `nn::Dropout` モジュール doc 参照）。
#[test]
fn dropout_model_train_predict_is_deterministic_and_matches_tape_path() {
    let _guard = dropout_test_lock()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let model = build_model_with_dropout(0.5).unwrap();
    assert!(model.training());
    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();

    fandhe_ai::manual_seed(31415);
    let out1 = model.predict(&x).unwrap();

    fandhe_ai::manual_seed(31415);
    let out2 = model.predict(&x).unwrap();
    assert_eq!(
        out1.as_slice(),
        out2.as_slice(),
        "同一 seed なら run-to-run bit 同一"
    );

    fandhe_ai::manual_seed(31415);
    let via_tape = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let pred = bound.forward(&t, &xv).unwrap();
        pred.to_tensor()
    };
    assert_eq!(
        out1.as_slice(),
        via_tape.as_slice(),
        "tape 不要経路〈predict〉と tape 経路〈bind().forward()〉は同一 seed で bit 一致"
    );
}

/// (c) `set_training(false)` がコンテナ（`nn::Sequential::inner`。
/// `dyn Module` 経由）から `Dropout` へ実際に伝播し、`predict`
/// （tape 不要経路）だけでなく `bind().forward()`（tape 経路）でも
/// 恒等写像へ切り替わることを確認する。
#[test]
fn dropout_model_set_training_false_propagates_to_bind_forward_too() {
    let mut model = build_model_with_dropout(0.9).unwrap();
    model.set_training(false);
    assert!(!model.training());

    let x = Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[2, 4]).unwrap();
    let baseline = build_model();
    let out_baseline = baseline.predict(&x).unwrap();

    let via_tape = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let pred = bound.forward(&t, &xv).unwrap();
        pred.to_tensor()
    };
    assert_eq!(
        via_tape.as_slice(),
        out_baseline.as_slice(),
        "set_training(false) 伝播後は Dropout なしモデルの forward と bit 一致するはず"
    );
}

/// (d) `add_dropout` は `p` の範囲外（`[0, 1]` 外）を `Err` で拒否する
/// （`Dropout::new` の検査を層構築時点で早期化する契約）。
#[test]
fn add_dropout_rejects_out_of_range_p() {
    // `Sequential` は `Debug` を実装しない（数値ロジックを持たないビル
    // ダーのため。`unwrap_err()`／`{:?}` での `Result` 全体表示は `Ok`
    // 側 `T: Debug` を要求するため使えない）ので `if let` で `Err` 側の
    // variant のみを検査する。
    if let Err(err) = Sequential::new().add_dropout(1.5) {
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    } else {
        panic!("p=1.5 は InvalidArgument で拒否されるはず");
    }
    if let Err(err) = Sequential::new().add_dropout(-0.1) {
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    } else {
        panic!("p=-0.1 は InvalidArgument で拒否されるはず");
    }
}

// --- イシュー #1760・Cursor Bugbot 指摘是正: predict の tape 不要経路
//     フォールバックが副作用を二重発生させないことの回帰テスト ---

/// `predict` の tape 不要経路（`predict_tape_free`）が
/// `MultiheadAttention`（`forward_host` 未実装で常に `Unsupported`）に
/// 到達する前に `Dropout`（RNG を消費する副作用付き層）を実行して
/// しまうと、`Unsupported` を受けて全層を旧経路（`predict_via_tape`）
/// で再実行する際に `Dropout` の RNG 消費が二重に発生し、同一 seed
/// から `predict` を 1 回呼んだ結果が「`Dropout` を 1 回だけ適用した
/// 場合」の期待値と食い違ってしまう（是正前は本テストが失敗する）。
///
/// [`Module::supports_forward_host`] による事前判定
/// （`compat::Sequential::predict` 冒頭）で、`MultiheadAttention` を
/// 含むモデルは tape 不要経路を一切実行せず最初から旧経路のみを使う
/// ことにより、RNG は 1 回しか消費されない。
#[test]
fn predict_fallback_to_via_tape_does_not_double_apply_dropout_side_effects() {
    let _guard = dropout_test_lock()
        .lock()
        .unwrap_or_else(|p| p.into_inner());

    // `Dropout` → `MultiheadAttention`。`Dropout` は shape を変えない
    // ため、そのまま `MultiheadAttention` の `[batch, seq, embed_dim]`
    // 契約へ連鎖できる（`multihead_attention_predict_matches_forward_bit_exact`
    // と同じ形状・seed 方針）。
    let model = Sequential::new()
        .add_dropout(0.5)
        .unwrap()
        .add_multihead_attention(4, 2, /* seed = */ 7)
        .unwrap();
    assert!(
        model.training(),
        "既定は train モード（Dropout が作用する）"
    );

    let x = Tensor::new(
        (0..2 * 3 * 4).map(|i| (i as f32) * 0.03 - 0.4).collect(),
        &[2, 3, 4],
    )
    .unwrap();

    fandhe_ai::manual_seed(2024);
    let out_predict = model.predict(&x).unwrap();

    // 「`Dropout` の RNG 消費が 1 回だけ」であることの基準値: 同じ
    // seed から直接 `bind().forward()`（tape 経路）を 1 回呼んだ結果
    // （`predict` が内部で `predict_via_tape` のみへ委譲する場合の
    // 期待値と一致するはず）。
    fandhe_ai::manual_seed(2024);
    let out_reference = {
        let t = tape();
        let bound = model.bind(&t);
        let xv = t.var(&x);
        let pred = bound.forward(&t, &xv).unwrap();
        pred.to_tensor()
    };

    assert_eq!(
        out_predict.as_slice(),
        out_reference.as_slice(),
        "predict は Dropout の RNG 消費を 1 回だけ行うはず\
         （tape 不要経路での部分実行 → 旧経路への全体フォールバックで\
         二重消費してはならない）"
    );
}
