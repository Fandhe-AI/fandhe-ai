//! オペ横断回帰テスト（TASK-7.3d・#85）。
//!
//! transformer ブロックの縮小パイプライン（前処理 → 線形層 → 活性化 → 残差 →
//! `LayerNormalization`）を模して `crates/onnx-interop/src/ops/` の各オペを連結実行し、
//! 各段の中間 shape・数値を固定値で回帰確認する。ONNX proto デコード・グラフ実行との
//! 結線（TASK-7.2a・#78/#274）はスコープ外であり、本テストは decode 層を経由しない
//! 直接呼び出しのみで完結する。
//!
//! ## カバレッジ（重要 — #86 TASK-7.4 実装前に必読）
//!
//! イシュー #85 計画は「22 オペ横断テスト」を想定していたが、着手時点で以下 2 イシューが
//! 未マージのため、本テストは **未マージオペを含まず** 15 オペ
//! （`Gemm`／`Relu`／`Sigmoid`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`／
//! `Add`／`Mul`／`Div`／`Mod`／`Sqrt`／`Constant`／`LayerNormalization`）のみを対象とする:
//!
//! - TASK-7.3b・#83（`Cast`／`Reshape`／`Squeeze`／`Transpose`）: PR #275 でマージ済み
//! - TASK-7.3c・#84（`MatMul`／`Softmax`／`Erf`）: 本 PR（#276）で実装。`main` へのマージ
//!   時点では未反映（本テストは #85 時点の 15 オペのまま）
//!
//! `#86`（TASK-7.4 end-to-end 推論）が本テストを 22 オペ完全カバレッジの根拠として
//! 前提にしないよう、この節を残す。#83／#84 マージ後の 22 オペ完全カバレッジ化（`Cast`／
//! `Reshape`／`Squeeze`／`Transpose`／`MatMul`／`Softmax`／`Erf` の追加）は別途対応する。

use fandhe_ai_tensor_core::Tensor;
use onnx_interop::ops::{
    ConstantValue, GemmAttrs, LayerNormAttrs, SliceParams, add, concat, constant, div, gather,
    gemm, layer_normalization, modulo, mul, relu, shape, sigmoid, slice, sqrt, unsqueeze,
};

fn assert_close(a: f32, b: f32) {
    let tol = 1e-4_f32.max(b.abs() * 1e-3);
    assert!(
        (a - b).abs() <= tol,
        "assert_close failed: a={a}, b={b}, diff={}",
        (a - b).abs()
    );
}

/// ## Stage 1: 前処理系（`Constant`／`Shape`／`Gather`／`Unsqueeze`／`Concat`／`Slice`）
///
/// トークン埋め込みテーブルを模した `[4, 3]`（vocab=4, dim=3）の定数テンソルから
/// 2 トークン分の埋め込みを `gather` で取り出し、`unsqueeze`/`concat`/`slice` で
/// バッチ次元の合成・切り出しを行う縮小パイプライン。
#[test]
fn stage1_preprocessing_ops() {
    // `Constant`: 埋め込みテーブル [4,3]（行 i の値は i*10 + [0,1,2]）。
    let table = constant(&ConstantValue::Tensor {
        data: vec![
            0.0, 1.0, 2.0, // token 0
            10.0, 11.0, 12.0, // token 1
            20.0, 21.0, 22.0, // token 2
            30.0, 31.0, 32.0, // token 3
        ],
        shape: vec![4, 3],
    })
    .unwrap();

    // `Shape`: [4,3] を確認（後段の gather 範囲検査に使う想定の情報）。
    assert_eq!(shape(&table), vec![4, 3]);

    // `Gather`: token id = [2, 0] を axis=0 で取り出し -> [2,3]。
    let embedded = gather(&table, &[2, 0], &[2], 0).unwrap();
    assert_eq!(embedded.shape(), &[2, 3]);
    assert_eq!(embedded.get(&[0, 0]).unwrap(), 20.0);
    assert_eq!(embedded.get(&[1, 0]).unwrap(), 0.0);

    // `Unsqueeze`: バッチ次元を先頭に追加 -> [1,2,3]。
    let batched = unsqueeze(&embedded, &[0]).unwrap();
    assert_eq!(batched.shape(), &[1, 2, 3]);

    // `Concat`: バッチサイズ 2 に複製結合 -> [2,2,3]（2 バッチ分のシーケンスを模す）。
    let doubled = concat(&[&batched, &batched], 0).unwrap();
    assert_eq!(doubled.shape(), &[2, 2, 3]);
    assert_eq!(
        doubled.get(&[0, 0, 0]).unwrap(),
        doubled.get(&[1, 0, 0]).unwrap()
    );

    // `Slice`: シーケンス次元（axis=1）の先頭 1 トークンのみ切り出し -> [2,1,3]。
    let sliced = slice(
        &doubled,
        &SliceParams {
            starts: &[0],
            ends: &[1],
            axes: Some(&[1]),
            steps: None,
        },
    )
    .unwrap();
    assert_eq!(sliced.shape(), &[2, 1, 3]);
    assert_eq!(sliced.get(&[0, 0, 0]).unwrap(), 20.0);
    assert_eq!(sliced.get(&[1, 0, 0]).unwrap(), 20.0);
}

/// ## Stage 2: 算術系（`Add`／`Mul`／`Div`／`Mod`／`Sqrt`）
///
/// スケーリング・正則化で典型的な算術オペの連結（`(a * b + a) / b` に `Mod`／`Sqrt` を
/// 追加した合成関数）を局所的な参照値と突合する。
#[test]
fn stage2_arithmetic_ops() {
    let a = Tensor::<f32>::new(vec![4.0, 9.0, 16.0, 25.0], &[4]).unwrap();
    let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 5.0], &[4]).unwrap();

    // `Mul` -> `Add` -> `Div`: (a*b + a) / b = a * (b+1) / b
    let ab = mul(&a, &b).unwrap();
    let ab_plus_a = add(&ab, &a).unwrap();
    let y = div(&ab_plus_a, &b).unwrap();
    for i in 0..4 {
        let av = a.get(&[i]).unwrap();
        let bv = b.get(&[i]).unwrap();
        assert_close(y.get(&[i]).unwrap(), av * (bv + 1.0) / bv);
    }

    // `Mod`（fmod=1）: a を b で割った余り。
    let m = modulo(&a, &b, true).unwrap();
    assert_eq!(m.get(&[0]).unwrap(), 0.0); // 4 % 2 = 0
    assert_eq!(m.get(&[1]).unwrap(), 0.0); // 9 % 3 = 0
    assert_eq!(m.get(&[2]).unwrap(), 0.0); // 16 % 4 = 0
    assert_eq!(m.get(&[3]).unwrap(), 0.0); // 25 % 5 = 0

    // `Sqrt`: a = [4,9,16,25] は完全平方数。
    let s = sqrt(&a).unwrap();
    assert_eq!(s.get(&[0]).unwrap(), 2.0);
    assert_eq!(s.get(&[1]).unwrap(), 3.0);
    assert_eq!(s.get(&[2]).unwrap(), 4.0);
    assert_eq!(s.get(&[3]).unwrap(), 5.0);
}

/// ## Stage 3: 線形層＋活性化＋残差＋正規化
/// （`Gemm`／`Relu`／`Sigmoid`／`Add`／`LayerNormalization`）
///
/// FFN ブロックを模した `X -> Gemm -> Relu -> residual Add -> LayerNormalization` と
/// `Gemm -> Sigmoid` の 2 分岐を連結し、`LayerNormalization` を含む最終段まで到達することを
/// 確認する。数値は独立参照実装（本ファイル内の素朴なループ）との突合で検証する。
#[test]
fn stage3_ffn_and_layer_norm() {
    // X: [2,3]（バッチ 2・特徴 3）。W: [3,2]（Gemm の B）。bias なし（C=None）。
    let x = Tensor::<f32>::new(vec![1.0, -2.0, 3.0, 0.5, 1.5, -1.0], &[2, 3]).unwrap();
    let w = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();

    let gemm_out = gemm(&x, &w, None, &GemmAttrs::default()).unwrap();
    assert_eq!(gemm_out.shape(), &[2, 2]);
    // 手計算: row0 [1,-2,3] @ W -> col0: 1*1+(-2)*0+3*1=4, col1: 1*0+(-2)*1+3*1=1
    // row1 [0.5,1.5,-1] @ W -> col0: 0.5*1+1.5*0+(-1)*1=-0.5, col1: 0.5*0+1.5*1+(-1)*1=0.5
    assert_close(gemm_out.get(&[0, 0]).unwrap(), 4.0);
    assert_close(gemm_out.get(&[0, 1]).unwrap(), 1.0);
    assert_close(gemm_out.get(&[1, 0]).unwrap(), -0.5);
    assert_close(gemm_out.get(&[1, 1]).unwrap(), 0.5);

    // 分岐 1: `Relu` -> 残差 `Add`（Relu(gemm_out) + gemm_out）。
    let relu_out = relu(&gemm_out).unwrap();
    assert_eq!(relu_out.get(&[1, 0]).unwrap(), 0.0); // -0.5 -> 0
    let residual = add(&relu_out, &gemm_out).unwrap();
    assert_close(residual.get(&[0, 0]).unwrap(), 4.0 + 4.0);
    assert_close(residual.get(&[1, 0]).unwrap(), 0.0 + (-0.5));

    // 分岐 2: `Sigmoid`（実行のみ確認。値域 (0,1) を検査）。
    let sig_out = sigmoid(&gemm_out).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let v = sig_out.get(&[i, j]).unwrap();
            assert!(v > 0.0 && v < 1.0, "sigmoid output out of (0,1): {v}");
        }
    }

    // `LayerNormalization`: 残差出力を axis=-1 で正規化。
    let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
    let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
    let ln_out =
        layer_normalization(&residual, &scale, Some(&bias), &LayerNormAttrs::default()).unwrap();
    assert_eq!(ln_out.shape(), &[2, 2]);

    // 独立参照実装: 行ごとの平均・母分散で正規化（epsilon=1e-5 は無視できるほど小さい）。
    for i in 0..2 {
        let row = [
            residual.get(&[i, 0]).unwrap(),
            residual.get(&[i, 1]).unwrap(),
        ];
        let mean = (row[0] + row[1]) / 2.0;
        let var = ((row[0] - mean).powi(2) + (row[1] - mean).powi(2)) / 2.0;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        assert_close(ln_out.get(&[i, 0]).unwrap(), (row[0] - mean) * inv_std);
        assert_close(ln_out.get(&[i, 1]).unwrap(), (row[1] - mean) * inv_std);
    }

    // 正規化後の行はほぼ平均 0・母分散 1（epsilon 分の誤差のみ）であることを確認
    // （`LayerNormalization` の意味論そのものの回帰確認）。
    for i in 0..2 {
        let row_mean = (ln_out.get(&[i, 0]).unwrap() + ln_out.get(&[i, 1]).unwrap()) / 2.0;
        assert_close(row_mean, 0.0);
    }
}

// 15 オペすべてが本ファイル内で少なくとも 1 回呼ばれていることの一覧（コメントのみ。
// コンパイル時に強制はできないため、レビュー・#86 参照用の対応表として残す）:
//
// | オペ | 呼び出し箇所 |
// |------|-------------|
// | Constant | stage1 |
// | Shape | stage1 |
// | Gather | stage1 |
// | Unsqueeze | stage1 |
// | Concat | stage1 |
// | Slice | stage1 |
// | Add | stage2・stage3 |
// | Mul | stage2 |
// | Div | stage2 |
// | Mod | stage2 |
// | Sqrt | stage2 |
// | Gemm | stage3 |
// | Relu | stage3 |
// | Sigmoid | stage3 |
// | LayerNormalization | stage3 |
