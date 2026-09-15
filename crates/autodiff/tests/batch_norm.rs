//! BatchNorm1d／2d（イシュー #1732・親 #1608）の tape 経路統合テスト。
//!
//! ユニットレベルの検証（`eval::batch_norm_train_channels`／
//! `batch_norm_infer_channels` の手計算突合、`grad::
//! batch_norm_vjp_channels` の数値微分突合、`nn::batch_norm` の
//! running stats 更新・rank 限定契約）は `crates/autodiff/src/eval.rs`・
//! `grad.rs`・`nn/batch_norm.rs` のクレート内テストで完結している。
//! 本ファイルは facade を経由しない公開 API（`fandhe_ai_autodiff::
//! {Tape, nn}`）のみを使う統合テストに限定する。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{BatchNorm1d, Module, ModuleList, Sequential};
use fandhe_ai_tensor_core::Tensor;

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("test: expected contiguous tensor")
        .to_vec()
}

/// train モードの出力は「チャネルごとの正規化」であるという統計
/// オラクルを、独立に実装した `Var::mean_dims`／`Var::var(dim,
/// correction=0)`（`#1601`。rank 2 入力に限り単一軸 `dim=0` で
/// チャネルごとの N 方向縮約と一致する）で突き合わせる（REQ-2 複合
/// 判定。実装計画 §2.4「バッチ統計のテストオラクル」）。
#[test]
fn batch_norm_train_output_matches_independent_mean_var_oracle() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = Tensor::new(vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0, 2.0, -2.0], &[4, 2]).unwrap();
    let xv = tape.var(&x);
    let eps = 0.0f32;

    let bn = BatchNorm1d::without_affine(2, eps, 0.1).unwrap();
    let out = bn.bind(&tape).forward(&xv).unwrap().to_tensor();

    // 独立実装: 各チャネル方向 (dim=0) の mean／var(biased) を
    // Var::mean_dims／Var::var で求め、手作業で正規化した値と比較。
    let mean_v = xv.mean_dims(&[0], true).unwrap().to_tensor();
    let var_v = xv.var(Some(0), 0).unwrap().to_tensor();
    let mean = dense(&mean_v);
    let var = dense(&var_v);

    let x_data = dense(&x);
    let out_data = dense(&out);
    let n = 4usize;
    let c = 2usize;
    for row in 0..n {
        for ch in 0..c {
            let idx = row * c + ch;
            let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
            let expected = ((x_data[idx] as f64 - mean[ch] as f64) * rstd) as f32;
            let diff = (out_data[idx] - expected).abs();
            assert!(
                diff < 1e-4,
                "row={row} ch={ch} out={} expected={expected}",
                out_data[idx]
            );
        }
    }
}

/// rank 3（`[N, C, L]`。`BatchNorm1d` の空間入力）で train モードの
/// 出力がチャネル方向で概ね正規化される（平均 ≈ 0・biased 分散 ≈ 1）
/// ことをスモーク確認する。
#[test]
fn batch_norm_1d_rank3_train_normalizes_per_channel() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = Tensor::new(
        vec![
            1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.0, 3.0, 1.0, -0.5, 0.0,
        ],
        &[2, 2, 3],
    )
    .unwrap();
    let xv = tape.var(&x);
    let bn = BatchNorm1d::without_affine(2, 1e-8, 0.1).unwrap();
    let out = bn.bind(&tape).forward(&xv).unwrap().to_tensor();
    let data = dense(&out);

    // M = n*spatial = 2*3 = 6 要素/チャネル。
    for ch in 0..2 {
        let mut vals = Vec::new();
        for batch in 0..2 {
            for sp in 0..3 {
                vals.push(data[batch * (2 * 3) + ch * 3 + sp] as f64);
            }
        }
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 1e-3, "ch={ch} mean={mean}");
        assert!((var - 1.0).abs() < 1e-2, "ch={ch} var={var}");
    }
}

/// `nn::Sequential`（`container.rs`。イシュー #1759）経由で
/// `set_training(false)` が子 `BatchNorm1d` へ伝播し、以後の forward
/// が running stats（固定統計）経路へ切り替わることを確認する
/// （`Module::set_training` trait doc の「モードの正はコンテナが保持
/// するフラグ」契約）。
///
/// 修正前（PR #1874 codex-review P2・イシュー #1732 fix ループ）は
/// eval モードでの forward を同一入力で 2 回呼び「2 回とも bit 一致」
/// だけを検証していたが、BatchNorm の train モード出力自体が「その
/// バッチの統計で正規化する」決定的関数であるため、
/// `set_training(false)` の伝播が壊れて `seq` が実質 train モードの
/// ままでも、同一入力 `x2` を 2 回 forward すればそのつど `x2` 自身の
/// バッチ統計で正規化され結局 2 回とも bit 一致してしまい、この
/// テストは伝播漏れを検出できない設計だった。本テストは代わりに、
/// train forward（`momentum=1.0` により running stats が x1 の
/// バッチ統計そのもの — biased mean／unbiased var — へ完全に
/// 上書きされる）で得られる**固定統計から独立に算出した期待値**と
/// eval forward の出力を突き合わせる。伝播が壊れていれば eval
/// forward は実際には x2 自身のバッチ統計で正規化されるため、
/// この期待値（x1 由来の固定統計での正規化）とは一致せず検出できる。
#[test]
fn batch_norm_mode_propagates_through_sequential_container() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let bn = BatchNorm1d::without_affine(2, 0.0, 1.0).unwrap();
    let mut seq = Sequential::from(ModuleList::from_iter([Box::new(bn) as Box<dyn Module>]));

    // 1 回 train forward してから eval へ切り替える。momentum=1.0 に
    // より running_mean は x1 の biased channel mean、running_var は
    // x1 の unbiased channel var（biased var · M/(M−1)）へちょうど
    // 上書きされる（モジュール doc comment「running stats 更新契約」
    // 参照）。
    let x1 = Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap();
    let n = 2usize;
    // 独立実装: x1 のチャネル方向 (dim=0) の biased mean／var を
    // `Var::mean_dims`／`Var::var` で算出し、running_var の unbiased
    // 補正（M/(M−1)。M = n·spatial = n = 2）を手作業で適用する
    // （最初のテスト `batch_norm_train_output_matches_independent_
    // mean_var_oracle` と同じオラクル方式）。
    let running_mean;
    let running_var_unbiased;
    {
        let xv1 = tape.var(&x1);
        let mean_v = xv1.mean_dims(&[0], true).unwrap().to_tensor();
        let var_v_biased = xv1.var(Some(0), 0).unwrap().to_tensor();
        let m = n as f64;
        running_var_unbiased = dense(&var_v_biased)
            .iter()
            .map(|&v| (v as f64 * m / (m - 1.0)) as f32)
            .collect::<Vec<f32>>();
        running_mean = dense(&mean_v);
        seq.forward(&tape, &xv1).unwrap();
    }
    assert!(seq.training());

    seq.set_training(false);
    assert!(!seq.training());

    // eval モードでは running stats（固定統計＝上記で算出した x1 由来
    // の running_mean／running_var_unbiased）を使うため、x2 自身の
    // バッチ統計は一切使われない。
    let x2 = Tensor::new(vec![100.0, 200.0, -100.0, 50.0], &[2, 2]).unwrap();
    let xv2 = tape.var(&x2);
    let out = seq.forward(&tape, &xv2).unwrap().to_tensor();
    let out_data = dense(&out);
    let x2_data = dense(&x2);
    let c = 2usize;
    for row in 0..n {
        for ch in 0..c {
            let idx = row * c + ch;
            let rstd = 1.0f64 / (running_var_unbiased[ch] as f64).sqrt();
            let expected = ((x2_data[idx] as f64 - running_mean[ch] as f64) * rstd) as f32;
            let diff = (out_data[idx] - expected).abs();
            assert!(
                diff < 1e-3,
                "row={row} ch={ch} out={} expected={expected} \
                 （eval モードで固定統計を使わず x2 自身のバッチ統計で \
                 正規化された場合、set_training(false) の伝播漏れを \
                 示唆する）",
                out_data[idx]
            );
        }
    }

    // eval モードは running stats を更新しないため、同一入力を再度
    // forward しても bit 一致する（決定性の追加確認）。
    let out_again = seq.forward(&tape, &xv2).unwrap().to_tensor();
    assert_eq!(dense(&out), dense(&out_again));
}
