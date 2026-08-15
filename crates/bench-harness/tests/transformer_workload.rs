//! TASK-8.3a（イシュー #155。親 #154・TASK-8.3）: Transformer 複合ワークロードの実測。
//!
//! REQ-8（`docs/spec/04-requirements.md`）は複合ワークロード系（Transformer 推論等、
//! attention／softmax／LayerNorm を含む複合演算）について「v2 自作カーネルでは未実測の
//! ため下限を設定しない。自作カーネルでの Transformer ブロック実測後、本要件の丸め規則で
//! 下限を設定する」を受け入れ基準として残している。本ファイルはその未実測領域を埋める
//! ための計測ワークロード（小型 Transformer ブロック 1 層の forward）を提供する。
//!
//! ## ワークロード形状（PoC-8 定義。PoC-5 流用）
//!
//! `d_model=512, n_heads=8, d_ff=2048, batch=8, seq_len=128, num_layers=1`・activation=GELU
//! （`0.5*x*(1+erf(x/sqrt(2)))`）・post-norm（`Add → LayerNormalization` の順序。
//! `transformer.onnx` フィクスチャの `norm_first=false` と整合）。
//!
//! ## 経路選択（性能解釈に影響するため明記する）
//!
//! - QKV／出力射影・FFN の 2D GEMM（`[batch*seq, d_model] @ [d_model, d_model]` 等）は
//!   [`backend_cpu::CpuBackendOps::gemm`]（BLIS 型・rayon 並列の最適化済み自作カーネル）を
//!   経由する。REQ-8 が求める「自作カーネルでの Transformer ブロック実測」の主対象。
//! - attention 内のバッチ行列積（`Q @ K^T`・`softmax(...) @ V`）は
//!   [`onnx_interop::ops::matmul`]（`numpy.matmul` 準拠のバッチ対応 naive 実装）を使う。
//!   ヘッド単位 2D GEMM ループへの分解は行っていない（naive 経路であることが計測値に
//!   含まれる。過小評価方向であり、下限確定〈#158〉の判断材料として
//!   `docs/perf/transformer-workload-measurement.md` に明記する）。
//! - softmax・LayerNormalization・Erf（GELU 合成）・残差 Add は
//!   [`onnx_interop::ops`] の公開関数（naive 実装）をそのまま使う。
//!
//! ## スコープ境界
//!
//! - 丸め規則の適用・段階的下限表の合否判定は #152／#153（TASK-8.2）
//! - 下限値の確定・REQ-8 表への反映は #158（TASK-8.3d・人間判断）
//! - CUDA 最適化後下限の再実測は #157、Metal f16 実測は #156
//! - ワークロード形状定数・決定的シードは #589（Phase G-4）で
//!   [`bench_harness::transformer_workload`] へ単一真実源化済み（挙動不変。本ファイルは
//!   その定数を参照するのみで、計測ロジック・ベンチ名・`#[ignore]` 分離は変更しない）

use std::hint::black_box;

use backend_cpu::CpuBackendOps;
use bench_harness::rng::Xorshift64Star;
use bench_harness::transformer_workload::baseline_spec;
use bench_harness::{BenchError, BenchReport, MeasurementConfig, run};
use onnx_interop::ops::{
    LayerNormAttrs, add, erf, layer_normalization, matmul, mul, reshape, softmax, transpose,
};
use tensor_core::{BackendOps, Tensor};

/// ワークロード形状（PoC-8 定義。単一真実源は [`bench_harness::transformer_workload::baseline_spec`]。
/// 本ファイルはローカル `const` へ束縛して既存コードの参照箇所を変えずに済ませる
/// （#589 での単一真実源化に伴う挙動不変のリファクタリング）。
const D_MODEL: usize = baseline_spec().d_model;
const N_HEADS: usize = baseline_spec().n_heads;
const D_FF: usize = baseline_spec().d_ff;
const BATCH: usize = baseline_spec().batch;
const SEQ_LEN: usize = baseline_spec().seq_len;
/// ヘッド次元。単一真実源は [`baseline_spec`] の [`TransformerWorkloadSpec::head_dim`]
/// （codex-review 指摘・PR #647 P1 修正後の型付き失敗 API）。`baseline_spec()` の値は
/// 不変条件（`n_heads > 0` かつ `d_model % n_heads == 0`）を満たす確定値のため
/// `const` 文脈で `panic!` する分岐は到達しない（違反時はコンパイルエラーとして表面化する）。
const HEAD_DIM: usize = match baseline_spec().head_dim() {
    Some(v) => v,
    None => panic!("baseline_spec() は head_dim の不変条件を満たす確定値のはず"),
};
/// LayerNormalization の epsilon。単一真実源は [`baseline_spec`] の
/// `layer_norm_eps` フィールド（Bugbot 指摘・PR #647: 従来 `1e-5` をハードコードしており、
/// SSOT を変更しても実ワークロードが追従しないドリフトを許していた）。
const LAYER_NORM_EPS: f32 = baseline_spec().layer_norm_eps;
/// Transformer ブロックの層数。単一真実源は [`baseline_spec`] の `num_layers` フィールド
/// （codex-review 指摘・PR #647 P2: 従来この値を参照せず常に 1 層固定で実行しており、
/// SSOT を変更しても実測が追従しないドリフトを許していた）。`generate_layers`／
/// `full_forward` がこの値だけ層を積み重ねる。
const NUM_LAYERS: usize = baseline_spec().num_layers;
/// Multi-Head Self-Attention サブレイヤーを含むかどうか。単一真実源は [`baseline_spec`] の
/// `has_attention` フィールド（同上 P2 指摘）。`transformer_block_forward` がこのフラグで
/// attention サブレイヤーの有無を分岐する。
const HAS_ATTENTION: bool = baseline_spec().has_attention;

/// 決定的シード（REQ-8「決定的シードを用いること」）。入力・重み生成の双方に使う。
/// 単一真実源は [`bench_harness::transformer_workload::SEED`]（#589）。
const SEED: u64 = bench_harness::transformer_workload::SEED;

/// `[shape]` の要素数を持つテンソルを決定的 RNG から生成する。
///
/// `Xorshift64Star::fill_vec`（TASK-8.1b）が返す `Vec<f32>` をそのまま `Tensor::new` に
/// 渡すだけの薄いヘルパーであり、本ファイル内の重み・入力生成箇所で重複する
/// 「生成 → shape 検証」パターンを 1 箇所に集約する。
fn gen_tensor(rng: &mut Xorshift64Star, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = rng.fill_vec(numel);
    Tensor::new(data, shape).expect("固定 shape のテンソル生成に失敗するはずがない")
}

/// 単一値を持つ形状 `[1]` のテンソルを構築する（`onnx_interop::ops::{add,mul}` の
/// multidirectional broadcasting を利用したスカラー演算に使う。`gemm.rs` の `C` 引数と
/// 同じブロードキャスト委譲パターン）。
fn scalar(value: f32) -> Tensor<f32> {
    Tensor::new(vec![value], &[1]).expect("スカラー shape [1] のテンソル生成に失敗するはずがない")
}

/// Transformer ブロック 1 層分の重み一式。post-norm 構成（`Add → LayerNormalization`）の
/// Multi-Head Attention + FFN サブレイヤーが必要とする全パラメータを決定的シードから
/// 生成し保持する。
struct TransformerWeights {
    w_q: Tensor<f32>,
    b_q: Tensor<f32>,
    w_k: Tensor<f32>,
    b_k: Tensor<f32>,
    w_v: Tensor<f32>,
    b_v: Tensor<f32>,
    w_o: Tensor<f32>,
    b_o: Tensor<f32>,
    ln1_scale: Tensor<f32>,
    ln1_bias: Tensor<f32>,
    w1: Tensor<f32>,
    b1: Tensor<f32>,
    w2: Tensor<f32>,
    b2: Tensor<f32>,
    ln2_scale: Tensor<f32>,
    ln2_bias: Tensor<f32>,
}

impl TransformerWeights {
    fn generate(rng: &mut Xorshift64Star) -> Self {
        Self {
            w_q: gen_tensor(rng, &[D_MODEL, D_MODEL]),
            b_q: gen_tensor(rng, &[D_MODEL]),
            w_k: gen_tensor(rng, &[D_MODEL, D_MODEL]),
            b_k: gen_tensor(rng, &[D_MODEL]),
            w_v: gen_tensor(rng, &[D_MODEL, D_MODEL]),
            b_v: gen_tensor(rng, &[D_MODEL]),
            w_o: gen_tensor(rng, &[D_MODEL, D_MODEL]),
            b_o: gen_tensor(rng, &[D_MODEL]),
            ln1_scale: gen_tensor(rng, &[D_MODEL]),
            ln1_bias: gen_tensor(rng, &[D_MODEL]),
            w1: gen_tensor(rng, &[D_MODEL, D_FF]),
            b1: gen_tensor(rng, &[D_FF]),
            w2: gen_tensor(rng, &[D_FF, D_MODEL]),
            b2: gen_tensor(rng, &[D_MODEL]),
            ln2_scale: gen_tensor(rng, &[D_MODEL]),
            ln2_bias: gen_tensor(rng, &[D_MODEL]),
        }
    }
}

/// [`NUM_LAYERS`] 層分の重みを決定的シードから連続生成する（`baseline_spec().num_layers` を
/// 実際のループ回数へ反映する側。#589・P2 修正: 従来 [`TransformerWeights::generate`] を
/// 1 回だけ呼ぶ形で暗黙に 1 層固定になっており `num_layers` フィールドが未参照だった）。
fn generate_layers(rng: &mut Xorshift64Star) -> Vec<TransformerWeights> {
    (0..NUM_LAYERS)
        .map(|_| TransformerWeights::generate(rng))
        .collect()
}

/// [`transformer_block_forward`] を [`NUM_LAYERS`] 層分連鎖適用する（前層の出力を
/// 次層の入力とする標準的な Transformer スタック構成）。
fn full_forward(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    layers: &[TransformerWeights],
) -> Tensor<f32> {
    let mut y = x.clone();
    for w in layers {
        y = transformer_block_forward(ops, &y, w);
    }
    y
}

/// `x2d @ w + b` を計算する（`ops.gemm` が最適化済み自作カーネル、`add` が bias の
/// multidirectional broadcasting を担う）。QKV・出力射影・FFN 2 層すべてが使う
/// 共通の「射影＋バイアス」パターン。
fn linear(
    ops: &dyn BackendOps,
    x2d: &Tensor<f32>,
    w: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Tensor<f32> {
    let proj = ops
        .gemm(x2d, w)
        .expect("固定 shape の GEMM 実行に失敗するはずがない");
    add(&proj, b).expect("bias の broadcast add に失敗するはずがない")
}

/// Multi-Head Self-Attention サブレイヤー（`x: [batch, seq, d_model] -> [batch, seq, d_model]`）。
///
/// QKV・出力射影は `ops.gemm`（最適化済み自作カーネル）、ヘッド分割後のバッチ行列積・
/// softmax は `onnx_interop::ops`（naive 実装）を使う（モジュール冒頭コメント「経路選択」節）。
fn multi_head_attention(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    w: &TransformerWeights,
) -> Tensor<f32> {
    let x2d = reshape(x, &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64], false)
        .expect("x の [batch*seq, d_model] への reshape に失敗するはずがない");

    let q2d = linear(ops, &x2d, &w.w_q, &w.b_q);
    let k2d = linear(ops, &x2d, &w.w_k, &w.b_k);
    let v2d = linear(ops, &x2d, &w.w_v, &w.b_v);

    // [batch*seq, d_model] -> [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
    let to_heads = |t2d: &Tensor<f32>| -> Tensor<f32> {
        let t4d = reshape(
            t2d,
            &[
                BATCH as i64,
                SEQ_LEN as i64,
                N_HEADS as i64,
                HEAD_DIM as i64,
            ],
            false,
        )
        .expect("[batch, seq, n_heads, head_dim] への reshape に失敗するはずがない");
        transpose(&t4d, Some(&[0, 2, 1, 3]))
            .expect("[batch, n_heads, seq, head_dim] への transpose に失敗するはずがない")
    };
    let q = to_heads(&q2d);
    let k = to_heads(&k2d);
    let v = to_heads(&v2d);

    // scores = (Q @ K^T) / sqrt(head_dim)
    let k_t =
        transpose(&k, Some(&[0, 1, 3, 2])).expect("K の末尾 2 軸 transpose に失敗するはずがない");
    let raw_scores = matmul(&q, &k_t).expect("Q @ K^T のバッチ行列積に失敗するはずがない");
    let scale = scalar(1.0 / (HEAD_DIM as f32).sqrt());
    let scores =
        mul(&raw_scores, &scale).expect("attention scores のスケーリングに失敗するはずがない");

    let weights = softmax(&scores, -1).expect("attention softmax に失敗するはずがない");
    let attn = matmul(&weights, &v).expect("softmax(...) @ V のバッチ行列積に失敗するはずがない");

    // [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim] -> [batch*seq, d_model]
    let attn_back = transpose(&attn, Some(&[0, 2, 1, 3]))
        .expect("[batch, seq, n_heads, head_dim] への transpose に失敗するはずがない");
    let attn2d = reshape(
        &attn_back,
        &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64],
        false,
    )
    .expect("attention 出力の [batch*seq, d_model] への reshape に失敗するはずがない");

    let out2d = linear(ops, &attn2d, &w.w_o, &w.b_o);
    reshape(
        &out2d,
        &[BATCH as i64, SEQ_LEN as i64, D_MODEL as i64],
        false,
    )
    .expect("attention 出力の [batch, seq, d_model] への reshape に失敗するはずがない")
}

/// GELU（`0.5*x*(1+erf(x/sqrt(2)))`）を `onnx_interop::ops`（`erf`／`add`／`mul`）の合成で計算する。
/// ONNX に `Gelu` 単体オペは無く（`transformer.onnx` も `Erf` 合成で表現する。REQ-7 準拠）、
/// 本ワークロードも同じ合成方式を採る。
fn gelu(x: &Tensor<f32>) -> Tensor<f32> {
    let inv_sqrt2 = scalar(std::f32::consts::FRAC_1_SQRT_2);
    let half = scalar(0.5);
    let one = scalar(1.0);

    let scaled = mul(x, &inv_sqrt2).expect("GELU: x / sqrt(2) の計算に失敗するはずがない");
    let erf_part = erf(&scaled).expect("GELU: erf の計算に失敗するはずがない");
    let one_plus_erf = add(&erf_part, &one).expect("GELU: 1 + erf(...) の計算に失敗するはずがない");
    let half_x = mul(x, &half).expect("GELU: 0.5 * x の計算に失敗するはずがない");
    mul(&half_x, &one_plus_erf).expect("GELU: 最終積の計算に失敗するはずがない")
}

/// Position-wise Feed-Forward サブレイヤー（`x: [batch, seq, d_model] -> [batch, seq, d_model]`）。
/// `d_model -> d_ff -> d_model` の 2 層 GEMM は `ops.gemm`（最適化済み自作カーネル）を使う。
fn feed_forward(ops: &dyn BackendOps, x: &Tensor<f32>, w: &TransformerWeights) -> Tensor<f32> {
    let x2d = reshape(x, &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64], false)
        .expect("FFN 入力の [batch*seq, d_model] への reshape に失敗するはずがない");
    let hidden = linear(ops, &x2d, &w.w1, &w.b1);
    let activated = gelu(&hidden);
    let out2d = linear(ops, &activated, &w.w2, &w.b2);
    reshape(
        &out2d,
        &[BATCH as i64, SEQ_LEN as i64, D_MODEL as i64],
        false,
    )
    .expect("FFN 出力の [batch, seq, d_model] への reshape に失敗するはずがない")
}

/// Transformer ブロック 1 層の forward（post-norm）。
///
/// `x -> MHA -> Add(residual) -> LayerNorm -> FFN -> Add(residual) -> LayerNorm` の順で、
/// `transformer.onnx` フィクスチャの `norm_first=false` 構成と整合させる
/// （`docs/spec/03-poc/poc-8-*` 系実測に対応する REQ-8 の代表ワークロード定義）。
///
/// [`HAS_ATTENTION`]（単一真実源は [`baseline_spec`] の `has_attention`）が `false` の場合は
/// MHA サブレイヤー・その残差接続／LayerNorm1 を丸ごとスキップし FFN サブレイヤーのみを
/// 適用する（#589・P2 修正: 従来はこのフィールドを一切参照せず常に attention ありの
/// 経路を実行しており、SSOT を変更しても実測が追従しないドリフトを許していた）。
fn transformer_block_forward(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    w: &TransformerWeights,
) -> Tensor<f32> {
    let ln_attrs = LayerNormAttrs {
        axis: -1,
        epsilon: LAYER_NORM_EPS,
    };

    let post_attn = if HAS_ATTENTION {
        let attn_out = multi_head_attention(ops, x, w);
        let residual1 = add(x, &attn_out).expect("MHA 残差接続の add に失敗するはずがない");
        layer_normalization(&residual1, &w.ln1_scale, Some(&w.ln1_bias), &ln_attrs)
            .expect("LayerNorm1 の計算に失敗するはずがない")
    } else {
        x.clone()
    };

    let ffn_out = feed_forward(ops, &post_attn, w);
    let residual2 = add(&post_attn, &ffn_out).expect("FFN 残差接続の add に失敗するはずがない");
    layer_normalization(&residual2, &w.ln2_scale, Some(&w.ln2_bias), &ln_attrs)
        .expect("LayerNorm2 の計算に失敗するはずがない")
}

/// 受け入れ基準対応の軽量テスト（非 ignore。通常 CI で実行される）。
///
/// - forward 計算 1 回分の出力 shape が `[batch, seq, d_model]` であることを検証する
///   （下記 ignored ベンチテストが計測する forward 実装の shape 整合の直接検証）。
/// - `MeasurementConfig::new` が TASK-8.1 プロトコル下限（20/20）未満を
///   `BenchError::ProtocolViolation` で拒否することを確認する（既存 API の性質を利用。
///   計測用ワークロード自体を軽量版で二重実装しない）。
#[test]
fn transformer_block_forward_produces_expected_shape() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(SEED);
    let layers = generate_layers(&mut rng);
    let x = gen_tensor(&mut rng, &[BATCH, SEQ_LEN, D_MODEL]);

    let y = full_forward(&ops, &x, &layers);
    assert_eq!(y.shape(), &[BATCH, SEQ_LEN, D_MODEL]);
}

/// TASK-8.1 計測プロトコル（20/20 下限）が本ワークロードの計測経路でも遵守されることを
/// 確認する（`MeasurementConfig::new` の下限検証を回避する API を追加していないことの
/// 回帰確認。`.claude/rules/security.md` A08）。
#[test]
fn measurement_config_rejects_below_protocol_minimum() {
    let err = MeasurementConfig::new(1, 1)
        .expect_err("warmup/iters 1 は TASK-8.1 下限（20）未満のため拒否されるはず");
    assert!(matches!(err, BenchError::ProtocolViolation(_)));
}

/// TASK-8.3a 本体: Transformer ブロック 1 層 forward（CPU・最適化済み自作カーネル経路）の
/// 実測（`docs/spec/05-tasks.md` TASK-8.1 プロトコル準拠）。
///
/// 実行時間が長い（1 回の forward が GEMM 4 回＋バッチ行列積 2 回＋softmax／LayerNorm／GELU
/// を含む）ため通常 CI から `#[ignore]` 分離する（`.claude/rules/coding-rust.md`）。
/// `cargo test -p bench-harness --release -- --ignored transformer --nocapture` で明示実行し、
/// 実測結果（中央値・Q1/Q3）は `docs/perf/transformer-workload-measurement.md` に記録する
/// （受け入れ条件「実測記録が残されている」）。
#[test]
#[ignore = "計測用（実行時間長）。実機・計測環境で明示実行する"]
fn transformer_block_forward_bench_cpu() {
    let ops = CpuBackendOps::new();
    let mut rng = Xorshift64Star::new(SEED);
    let layers = generate_layers(&mut rng);
    let x = gen_tensor(&mut rng, &[BATCH, SEQ_LEN, D_MODEL]);

    let config = MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず");
    let measurement = run(&config, || {
        let y = full_forward(&ops, black_box(&x), black_box(&layers));
        black_box(y);
    })
    .expect("Transformer ブロック forward の計測に失敗しました");

    // CPU は `sync::CpuSync`（no-op）が契約上の同期方式（REQ-8「ホスト転送を伴わない
    // 完了待ち」への統一）に該当する。`ops.gemm` 等はホスト常駐 `Tensor<f32>` を
    // 同期的に返す契約（`tensor_core::backend_ops::BackendOps` ドキュメンテーション
    // コメント参照）のため、`workload` クロージャの戻り時点で計測対象処理は完了しており
    // 追加の `wait_idle()` 呼び出しは不要（`protocol::run` の前提を満たす）。

    let report =
        BenchReport::from_measurement("transformer-block-forward-cpu-blis", "cpu", &measurement)
            .expect("BenchReport 構築に失敗しました");
    let json = report
        .to_json()
        .expect("BenchReport の JSON エンコードに失敗しました");
    println!("{json}");

    // `docs/perf/transformer-workload-measurement.md` への転記を自動化する場合の出力先。
    // 未設定時は stdout 出力のみで完結する（実測記録は手動転記が既定運用。
    // `docs/perf/cuda-tensor-core-measurement.md` の先例と同じ運用）。
    if let Ok(path) = std::env::var("BENCH_TRANSFORMER_REPORT") {
        std::fs::write(&path, &json).unwrap_or_else(|e| {
            panic!("BENCH_TRANSFORMER_REPORT={path} への書き込みに失敗しました: {e}")
        });
    }
}
