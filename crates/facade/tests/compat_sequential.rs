//! 受け入れ条件「Sequential でのモデル構築・推論が動作する」
//! （TASK-9.2a・イシュー #95。TASK-9.4・#411 で `fandhe_ai_autodiff::compat` から
//! `fandhe_ai::compat` へ移設）を公開 API（`fandhe_ai::compat`）経由で検証する
//! 統合テスト。ユニットテスト（`src/compat/sequential.rs`・
//! `src/compat/array.rs` 内）はクレート内部の中間状態を扱うのに対し、
//! 本ファイルは crate 外部から見える公開 API のみを使う点が異なる。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。
//!
//! **TASK-9.4 での構成変更**: 旧 `fandhe_ai_autodiff::compat` 版は無引数版
//! `predict`／`predict_with_ops`（naive ops 明示指定）の同値性、および
//! `Tape::default()` の独立した確認を含んでいた。`predict_with_ops` は
//! 本移設で公開面から撤去した（`src/compat/sequential.rs` モジュール doc
//! 参照）ため、`predict_default_matches_predict_with_ops_naive` は
//! 「`predict`（既定 CPU・融合経路）と `fandhe_ai::tape()` 上の手動 `forward`
//! のビット一致」＋「既定 CPU 経路と naive 参照実装（`fandhe_ai_autodiff::Tape::new()`）
//! との REQ-2 複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）」の 2 段
//! 判定に再構成し、意図（既定 `predict` が期待どおりの経路へ委譲して
//! いることの確認）を保った。`Tape::default()` 単体の確認
//! （`tape_default_records_and_evaluates_ops`）は compat 非依存のため
//! `crates/autodiff/tests/tape_recording.rs` に残置している。
//!
//! **`fandhe_ai::Tape`（newtype）と `fandhe_ai_autodiff::Tape`（生の型）の使い分け
//! （codex-review PR #424 P1 是正）**: `fandhe_ai::compat::Sequential::
//! forward`／`bind` は公開シグネチャに `fandhe_ai::Tape` を取る（`src/lib.rs`
//! モジュール doc「`Tape`（composition root が構築する値）の扱い」参照）。
//! 本ファイルは「naive 参照実装」（融合を経ない per-op 逐次実装との
//! REQ-2 複合判定）を構築するために `fandhe_ai_autodiff::nn::Linear`／
//! `fandhe_ai_autodiff::nn::activation` を直接呼ぶ手動 forward を使う箇所があり、
//! そこでは `fandhe_ai_autodiff::Tape::new_with_ops(Box::new(CpuBackendOps::new()))`
//! （`fandhe_ai::tape()` と同じ ops 構成を明示的に組み立てたもの。テスト
//! フィクスチャのみで許容される内部 API 直接利用。`Sequential::forward`
//! 自体には渡さない）を使う。

use fandhe_ai::compat::{Sequential, array};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::activation::Sigmoid;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `fandhe_ai::tape()` と同じ ops 構成（既定 CPU・`CpuBackendOps`・融合
/// 有効）の生 `fandhe_ai_autodiff::Tape` を構築する（テストフィクスチャ専用。
/// モジュール冒頭 doc 参照）。
fn raw_facade_equivalent_tape() -> Tape {
    Tape::new_with_ops(Box::new(CpuBackendOps::new()))
}

const SEED1: u64 = 7001;
const SEED2: u64 = 7002;

/// 受け入れ条件本体: `Sequential::new().add_linear(..).add_relu()...` の
/// メソッドチェーンで 2 層 MLP を構築し、`predict` が期待 shape の
/// `Tensor<f32>` を返すこと。
#[test]
fn sequential_builds_two_layer_mlp_and_predicts() {
    let model = Sequential::new()
        .add_linear(8, 16, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(16, 4, SEED2)
        .unwrap();

    let batch = 5;
    let input = array(vec![vec![0.25_f32; 8]; batch]).unwrap();
    let output = model.predict(&input).unwrap();

    assert_eq!(output.shape(), &[batch, 4]);
}

/// `Sequential::predict`（既定 CPU・[`fandhe_ai::tape`] 経由の融合経路）が
/// (a) `fandhe_ai::tape()` 上で組んだ手動 `forward` とビット一致し、
/// (b) naive 参照実装（`fandhe_ai_autodiff::Tape::new()`。融合を経ない per-op
/// 逐次実装）と REQ-2 統一複合判定（相対誤差 1e-3 未満 または絶対誤差
/// 1e-5 未満。`crates/facade/tests/fusion_default_parity.rs` と同じ判定
/// 様式）で一致することを確認する。(a) は同一経路（fandhe_ai::tape()）の
/// ため tolerance を新設せずビット一致で判定し、(b) のみ許容誤差を要する
/// 既存定数を使う（`.claude/rules/coding-rust.md`「許容誤差を単独で
/// 緩和しない」）。
#[test]
fn predict_matches_manual_forward_and_naive_reference() {
    let model = Sequential::new()
        .add_linear(8, 16, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(16, 4, SEED2)
        .unwrap();

    let input = array(vec![vec![0.25_f32; 8]; 3]).unwrap();
    let via_predict = model.predict(&input).unwrap();

    // (a) fandhe_ai::tape() 上の手動 forward とビット一致。
    let manual_tape = fandhe_ai::tape();
    let manual_input = manual_tape.var(&input);
    let manual_output = model.forward(&manual_tape, &manual_input).unwrap();
    assert_eq!(
        output_dense(&via_predict),
        output_dense(&manual_output.to_tensor())
    );

    // (b) naive 参照実装（fandhe_ai_autodiff::Tape::new()）との REQ-2 複合判定。
    // `fandhe_ai::compat::Sequential::forward` は `fandhe_ai::Tape` のみを
    // 受け取る（本ファイル冒頭 doc 参照）ため、naive 側は `model` と
    // 同一シード・同一層構成（Linear(8,16,SEED1)→ReLU→Linear(16,4,SEED2)）
    // を `fandhe_ai_autodiff::nn::Linear`／`Var::relu` で直接組み立てる（モデル
    // 構造は `sequential_forward_matches_manual_nn_forward_bit_exact` の
    // 手動経路と同型）。
    let naive_tape = Tape::new();
    let naive_input = naive_tape.var(&input);
    let naive_l1 = Linear::new(8, 16, true, SEED1).unwrap();
    let naive_l2 = Linear::new(16, 4, true, SEED2).unwrap();
    let naive_h = naive_l1.bind(&naive_tape).forward(&naive_input).unwrap();
    let naive_h = naive_h.relu();
    let naive_output = naive_l2.bind(&naive_tape).forward(&naive_h).unwrap();
    assert_parity(
        "Sequential::predict（既定 CPU・融合経路）vs fandhe_ai_autodiff::Tape::new()（NaiveOps 非融合参照）",
        &output_dense(&via_predict),
        &output_dense(&naive_output.to_tensor()),
    );
}

/// `compat::array` から生成したテンソルをそのまま `Sequential::predict`
/// に渡す経路（numpy 慣習の入口 → Keras 慣習の入口、という互換 API 層
/// 2 系統の連携）を確認する。
#[test]
fn array_output_feeds_directly_into_sequential_predict() {
    let model = Sequential::new()
        .add_linear(3, 2, SEED1)
        .unwrap()
        .add_tanh();

    let input = array([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
    let output = model.predict(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // tanh の出力は (-1, 1) に収まる。
    for v in output_dense(&output) {
        assert!((-1.0..=1.0).contains(&v));
    }
}

/// 数値一致（同一経路の完全一致）: 同一シードで `nn::Linear` +
/// `nn::activation` を直接組んだ手動 forward と `Sequential::forward` の
/// 出力がビット一致すること。同一演算列のため tolerance は新設しない
/// （`.claude/rules/coding-rust.md` の許容誤差を単独で緩和しない方針に
/// 沿い、ここでは緩和ではなく完全一致で判定する）。
#[test]
fn sequential_forward_matches_manual_nn_forward_bit_exact() {
    let linear1 = Linear::new(4, 6, true, SEED1).unwrap();
    let linear2 = Linear::new(6, 3, true, SEED2).unwrap();

    let input_tensor = array(vec![vec![0.1_f32, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]]).unwrap();

    // 手動経路。`Linear::bind` は `&fandhe_ai_autodiff::Tape` を要求するため
    // `fandhe_ai::tape()` ではなく生の `fandhe_ai_autodiff::Tape` を使う（本ファイル
    // 冒頭 doc「fandhe_ai::Tape（newtype）と fandhe_ai_autodiff::Tape（生の型）の
    // 使い分け」参照。ops 構成は fandhe_ai::tape() と同一）。
    let manual_tape = raw_facade_equivalent_tape();
    let manual_input = manual_tape.var(&input_tensor);
    let h = linear1.bind(&manual_tape).forward(&manual_input).unwrap();
    let h = Sigmoid.forward(&h);
    let manual_output = linear2.bind(&manual_tape).forward(&h).unwrap();

    // Sequential 経路（別インスタンスだが同一シードのため同一パラメータ）。
    let model = Sequential::new()
        .add_linear(4, 6, SEED1)
        .unwrap()
        .add_sigmoid()
        .add_linear(6, 3, SEED2)
        .unwrap();
    let seq_tape = fandhe_ai::tape();
    let seq_input = seq_tape.var(&input_tensor);
    let seq_output = model.forward(&seq_tape, &seq_input).unwrap();

    assert_eq!(
        output_dense(&manual_output.to_tensor()),
        output_dense(&seq_output.to_tensor())
    );
}

/// Tape 経由 forward: `Sequential::forward` を外部 `Tape` 上で実行し
/// `Tape::backward` まで通ることを確認する（グラフ記録の整合確認）。
#[test]
fn sequential_forward_on_external_tape_reaches_backward() {
    let model = Sequential::new()
        .add_linear(4, 3, SEED1)
        .unwrap()
        .add_relu();

    let tape = fandhe_ai::tape();
    let input_tensor = array([[0.1_f32, -0.2, 0.3, -0.4]]).unwrap();
    let input = tape.var(&input_tensor);
    let output = model.forward(&tape, &input).unwrap();
    let loss = output.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let input_grad = grads
        .get(&input)
        .unwrap()
        .expect("入力ノードは loss に寄与している");
    assert_eq!(input_grad.shape(), input_tensor.shape());
}

/// `array` の jagged 入力エラーは `Sequential` 側の構築失敗と混同されず、
/// 独立して伝播すること（外部入力検証は forward 到達前に完了する契約。
/// `.claude/rules/security.md` A03）。
#[test]
fn jagged_array_input_is_rejected_before_sequential_predict() {
    let jagged = vec![vec![1.0_f32, 2.0], vec![3.0]];
    assert!(array(jagged).is_err());
}

fn output_dense(t: &Tensor<f32>) -> Vec<f32> {
    // `tests/backward.rs` 等と同じく、非連続 stride を考慮せず単純な
    // 添字アクセスで足りる shape のみを扱うテストのため、`Tensor` の
    // 公開 API（`shape`/インデックス相当）を使わず素朴に検証したい
    // 箇所は呼び出し側で直接比較する。ここでは全要素が連続レイアウト
    // で確定している `to_tensor()` 直後の値のみを対象にするため、
    // `storage` を経由しない単純な複製で十分。
    (0..t.numel())
        .map(|i| {
            let shape = t.shape();
            let mut idx = vec![0usize; shape.len()];
            let mut rem = i;
            for d in (0..shape.len()).rev() {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            t.get(&idx).expect("計算済み添字は shape 内に収まる")
        })
        .collect()
}
