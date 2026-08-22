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
//!   [`fandhe_ai_backend_cpu::CpuBackendOps::gemm`]（BLIS 型・rayon 並列の最適化済み自作カーネル）を
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
//!
//! ## Phase G 適用前後の CUDA 経路（#602・G-12。追記分）
//!
//! 親 #582（Phase G）で CUDA 側にマージ済みの融合カーネル
//! （`gemm_bias_act` epilogue 融合・#599／online softmax・#594）を、上記
//! 経路選択の QKV・出力射影・FFN の GEMM＋bias（`linear` → `linear_fused`）と
//! attention softmax（`onnx_interop::ops::softmax` → `run_fused` 経由の
//! canonical softmax プラン）にのみ適用した「改善後」経路
//! （[`linear_fused`]／[`multi_head_attention_fused`]／[`feed_forward_fused`]／
//! [`transformer_block_forward_fused`]／[`full_forward_fused`]）を追加する。
//! attention 内バッチ行列積（`Q @ K^T`・`softmax(...) @ V`）・LayerNormalization・
//! GELU・残差 `Add` は「改善前」経路と同一のホスト naive 実装のまま据え置く
//! （Phase G がこれらに融合カーネルを提供していないため。差分を epilogue 融合＋
//! online softmax の効果に閉じる設計判断は #602 実装計画 §3 参照）。
//!
//! **計測解釈上の重要な非対称性**: linear 側の差分は「GPU `gemm` → 非融合
//! `add`（ホスト naive）」対「GPU 融合 `gemm_bias_act` epilogue」という
//! 同一デバイス内の epilogue 融合効果だが、softmax 側の差分は「ホスト
//! naive softmax（GPU 未使用）」対「`run_fused` 経由の GPU online softmax
//! カーネル（H2D → 起動 → 同期 → D2H を含む）」であり、**非融合側がそもそも
//! GPU に一度も触れていない**。したがって softmax 部分の速度差は「online
//! softmax カーネルの高速性」ではなく「ホスト計算から GPU 計算への変更」を
//! 反映する。両者を単純合算した「相対改善率」を Phase G（融合カーネル）の
//! 効果として報告する際は、linear 側（真の融合効果）と softmax 側
//! （ホスト→GPU 移行効果）を区別して記録する（`docs/perf/
//! transformer-workload-baseline.md` §7 追記時の注意点）。

use std::hint::black_box;

use bench_harness::rng::Xorshift64Star;
use bench_harness::transformer_workload::{baseline_spec, report_name, report_name_fused};
use bench_harness::{BenchError, BenchReport, MeasurementConfig, run};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::{Activation, BackendOps, DType, FusedOpKind, FusionPlan, Tensor};
use onnx_interop::ops::{
    LayerNormAttrs, add, erf, layer_normalization, matmul, mul, reshape, softmax, transpose,
};

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
    // 同期的に返す契約（`fandhe_ai_tensor_core::backend_ops::BackendOps` ドキュメンテーション
    // コメント参照）のため、`workload` クロージャの戻り時点で計測対象処理は完了しており
    // 追加の `wait_idle()` 呼び出しは不要（`protocol::run` の前提を満たす）。

    // ベンチ名の単一真実源は [`bench_harness::transformer_workload::report_name`]
    // （codex-review 指摘・PR #647: 従来 `"transformer-block-forward-cpu-blis"` を
    // 直書きしており `BENCH_NAME_PREFIX` 変更が実測経路に反映されなかった。
    // `report_name("cpu")` は CPU 経路が常に `gemm_blis` 系カーネルへディスパッチ
    // する契約に基づき `"-blis"` サフィックスを含む値を返す）。
    let report = BenchReport::from_measurement(report_name("cpu"), "cpu", &measurement)
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

// ============================================================================
// Phase G 適用後（融合あり）経路（#602・G-12。モジュール冒頭コメント参照）
// ============================================================================

/// [`linear`] の「改善後」版。`x2d @ w + b` を CUDA 融合 epilogue
/// （`ops.gemm_bias_act`。イシュー #599・`Activation::None`）経由で計算する。
/// 本ワークロードの bias（`w_q`/`w_k`/`w_v`/`w_o`/`w1`/`w2` の各 `b_*`）は
/// 常に `[n]` ちょうどの 1 階形状のため、`gemm_bias_act_route`
/// （`crates/backend-cuda/src/ops.rs`）は必ず `Fused` 分岐を選び、
/// 非融合フォールバック（`self.gemm` → `self.add`）へは落ちない。
fn linear_fused(
    ops: &dyn BackendOps,
    x2d: &Tensor<f32>,
    w: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Tensor<f32> {
    ops.gemm_bias_act(x2d, w, Some(b), Activation::None)
        .expect("固定 shape の融合 GEMM+bias 実行に失敗するはずがない")
}

/// attention softmax（最終軸縮約）の「改善後」版。`ops.run_fused` 経由の
/// canonical softmax 融合プラン（イシュー #594）へルーティングする。
///
/// `scores` は `[batch, n_heads, seq, seq]`（4 階）を想定する。
/// `match_softmax_plan`（`crates/backend-cuda/src/softmax.rs`）が受理する
/// canonical 形状は「leaf 1 個・2 階 `[rows, cols]`・最終軸縮約」のみのため、
/// 先頭 3 軸を `rows` へ畳んだ 2 階形状へ reshape してからプランを構築し、
/// 融合カーネル適用後に元の 4 階形状へ reshape し直す。プランの op 列は
/// `crates/backend-cuda/tests/softmax_parity.rs::
/// softmax_run_fused_matches_cpu_composed_env_adaptive` の構築先例をそのまま
/// 踏襲し、`axis`（`None`＝全軸縮約）を最終軸縮約の `Some(1)` に変更したもの
/// （`plan.rs::from_ops_builds_softmax_plan_with_row_fusion_metadata` が
/// 検証する 2 階・`axis: Some(1)` 形式と同型）。
///
/// 呼び出し元（[`multi_head_attention_fused`]）が計測する速度差は「CUDA
/// 融合 online softmax カーネルの高速性」ではなく「ホスト naive 計算から
/// GPU 計算への移行」であることに注意（モジュール冒頭コメント「計測解釈上の
/// 重要な非対称性」参照。非融合側の [`softmax`]（`onnx_interop::ops`）は
/// GPU に一切触れないため）。
fn softmax_fused(ops: &dyn BackendOps, scores: &Tensor<f32>) -> Tensor<f32> {
    let shape = scores.shape().to_vec();
    let cols = *shape.last().expect("scores は少なくとも 1 階のはず");
    let rows: usize = shape[..shape.len() - 1].iter().product();

    let flat = reshape(scores, &[rows as i64, cols as i64], false)
        .expect("softmax 融合プラン適用前の [rows, cols] への reshape に失敗するはずがない");

    // canonical softmax プラン（axis: Some(1)・最終軸縮約。leaf 0=x,
    // 1=Max{Some(1)}(0), 2=Broadcast{Some(1)}(1), 3=Sub(0,2), 4=Exp(3),
    // 5=Sum{Some(1)}(4), 6=Broadcast{Some(1)}(5), 7=Div(4,6)）。
    let plan_ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Max {
            input: 0,
            axis: Some(1),
        },
        FusedOpKind::Broadcast {
            input: 1,
            axis: Some(1),
        },
        FusedOpKind::Sub { lhs: 0, rhs: 2 },
        FusedOpKind::Exp { input: 3 },
        FusedOpKind::Sum {
            input: 4,
            axis: Some(1),
        },
        FusedOpKind::Broadcast {
            input: 5,
            axis: Some(1),
        },
        FusedOpKind::Div { lhs: 4, rhs: 6 },
    ];
    let plan = FusionPlan::from_ops(plan_ops, vec![rows, cols], DType::F32, 1)
        .expect("canonical softmax プランの構築に失敗するはずがない");

    let flat_out = ops
        .run_fused(&plan, &[&flat])
        .expect("run_fused 経由の canonical softmax 実行に失敗するはずがない");

    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    reshape(&flat_out, &shape_i64, false)
        .expect("softmax 融合プラン適用後の元 shape への reshape に失敗するはずがない")
}

/// [`multi_head_attention`] の「改善後」版。QKV・出力射影を
/// [`linear_fused`]、softmax を [`softmax_fused`] に差し替える。attention
/// 内バッチ行列積（`Q @ K^T`・`softmax(...) @ V`）・スケーリングは
/// 「改善前」経路と同一のホスト naive 実装のまま据え置く
/// （モジュール冒頭コメント「Phase G 適用前後の CUDA 経路」参照）。
fn multi_head_attention_fused(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    w: &TransformerWeights,
) -> Tensor<f32> {
    let x2d = reshape(x, &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64], false)
        .expect("x の [batch*seq, d_model] への reshape に失敗するはずがない");

    let q2d = linear_fused(ops, &x2d, &w.w_q, &w.b_q);
    let k2d = linear_fused(ops, &x2d, &w.w_k, &w.b_k);
    let v2d = linear_fused(ops, &x2d, &w.w_v, &w.b_v);

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

    let k_t =
        transpose(&k, Some(&[0, 1, 3, 2])).expect("K の末尾 2 軸 transpose に失敗するはずがない");
    let raw_scores = matmul(&q, &k_t).expect("Q @ K^T のバッチ行列積に失敗するはずがない");
    let scale = scalar(1.0 / (HEAD_DIM as f32).sqrt());
    let scores =
        mul(&raw_scores, &scale).expect("attention scores のスケーリングに失敗するはずがない");

    let weights = softmax_fused(ops, &scores);
    let attn = matmul(&weights, &v).expect("softmax(...) @ V のバッチ行列積に失敗するはずがない");

    let attn_back = transpose(&attn, Some(&[0, 2, 1, 3]))
        .expect("[batch, seq, n_heads, head_dim] への transpose に失敗するはずがない");
    let attn2d = reshape(
        &attn_back,
        &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64],
        false,
    )
    .expect("attention 出力の [batch*seq, d_model] への reshape に失敗するはずがない");

    let out2d = linear_fused(ops, &attn2d, &w.w_o, &w.b_o);
    reshape(
        &out2d,
        &[BATCH as i64, SEQ_LEN as i64, D_MODEL as i64],
        false,
    )
    .expect("attention 出力の [batch, seq, d_model] への reshape に失敗するはずがない")
}

/// [`feed_forward`] の「改善後」版。2 層 GEMM＋bias を [`linear_fused`] に
/// 差し替える。GELU（`erf` 合成）は「改善前」経路と同一のホスト naive
/// 実装のまま据え置く（Phase G が GELU 融合カーネルを提供していないため）。
fn feed_forward_fused(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    w: &TransformerWeights,
) -> Tensor<f32> {
    let x2d = reshape(x, &[(BATCH * SEQ_LEN) as i64, D_MODEL as i64], false)
        .expect("FFN 入力の [batch*seq, d_model] への reshape に失敗するはずがない");
    let hidden = linear_fused(ops, &x2d, &w.w1, &w.b1);
    let activated = gelu(&hidden);
    let out2d = linear_fused(ops, &activated, &w.w2, &w.b2);
    reshape(
        &out2d,
        &[BATCH as i64, SEQ_LEN as i64, D_MODEL as i64],
        false,
    )
    .expect("FFN 出力の [batch, seq, d_model] への reshape に失敗するはずがない")
}

/// [`transformer_block_forward`] の「改善後」版。MHA・FFN サブレイヤーを
/// [`multi_head_attention_fused`]／[`feed_forward_fused`] に差し替える。
/// 残差 `Add`・LayerNormalization は「改善前」経路と同一のホスト naive
/// 実装のまま据え置く（Phase G が LayerNorm 融合カーネルを提供していない
/// ため。本ワークロードは post-norm の `LayerNormalization` であり、
/// Phase G の融合 RMSNorm・#592 は mean 化・bias を持たない別演算のため
/// 適用対象外。#602 実装計画 §2「適用可能な Phase G 機能」参照）。
fn transformer_block_forward_fused(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    w: &TransformerWeights,
) -> Tensor<f32> {
    let ln_attrs = LayerNormAttrs {
        axis: -1,
        epsilon: LAYER_NORM_EPS,
    };

    let post_attn = if HAS_ATTENTION {
        let attn_out = multi_head_attention_fused(ops, x, w);
        let residual1 = add(x, &attn_out).expect("MHA 残差接続の add に失敗するはずがない");
        layer_normalization(&residual1, &w.ln1_scale, Some(&w.ln1_bias), &ln_attrs)
            .expect("LayerNorm1 の計算に失敗するはずがない")
    } else {
        x.clone()
    };

    let ffn_out = feed_forward_fused(ops, &post_attn, w);
    let residual2 = add(&post_attn, &ffn_out).expect("FFN 残差接続の add に失敗するはずがない");
    layer_normalization(&residual2, &w.ln2_scale, Some(&w.ln2_bias), &ln_attrs)
        .expect("LayerNorm2 の計算に失敗するはずがない")
}

/// [`full_forward`] の「改善後」版。[`NUM_LAYERS`] 層分
/// [`transformer_block_forward_fused`] を連鎖適用する。
fn full_forward_fused(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    layers: &[TransformerWeights],
) -> Tensor<f32> {
    let mut y = x.clone();
    for w in layers {
        y = transformer_block_forward_fused(ops, &y, w);
    }
    y
}

/// #602（G-12）本体・改善前: Transformer ブロック 1 層 forward の
/// CUDA 実測（融合カーネル未適用。[`full_forward`] をそのまま
/// `CudaBackendOps` へ適用する。既存の `linear`／`multi_head_attention`
/// 等は `ops: &dyn BackendOps` へジェネリック化済みのため CPU 版
/// （[`transformer_block_forward_bench_cpu`]）と実装を共有する）。
///
/// `CudaBackendOps::new(0)` はドライバ初期化を行わないため常に成功する
/// （`crates/backend-cuda/src/ops.rs::CudaBackendOps::new` ドキュメンテーション
/// コメント参照）。実際の CUDA 呼び出しは各 `ops.gemm`／`add`
/// （`onnx_interop::ops`。ホスト naive）の実行時点で発生し、CUDA 非搭載
/// 環境では `BackendError::CudaUnavailable` で `panic` する（`#[ignore]`
/// 分離により通常 CI では実行されないため許容する。実機・計測環境で
/// 明示実行する契約は本ファイル冒頭コメント「経路選択」節と同じ）。
///
/// 同期方式: `CudaBackendOps` の各演算はホスト常駐 `Tensor<f32>` を
/// 同期的に返す契約（`crates/backend-cuda/src/ops.rs` の各メソッドが
/// `stream.synchronize()` 相当を内包して D2H 転送済みの `Vec<f32>` を
/// 返す。`docs/perf/transformer-workload-baseline.md` §3「同期方式」の
/// CUDA 行と同じ）のため、`workload` クロージャの戻り時点で計測対象処理は
/// 完了しており追加の同期呼び出しは不要。**per-op（GEMM／bias-add／
/// softmax／LayerNorm 等 演算ごと）にホスト転送が発生し、その転送コストが
/// 計測値へそのまま含まれる**点に注意（`CudaBackendOps` はデバイス常駐
/// バッファを保持しないため。#602 実装計画 §3「同期方式」参照）。
#[test]
#[ignore = "計測用（実行時間長・CUDA 実機必須）。実機・計測環境で明示実行する"]
fn transformer_block_forward_bench_cuda_prefusion() {
    let ops = CudaBackendOps::new(0);
    let mut rng = Xorshift64Star::new(SEED);
    let layers = generate_layers(&mut rng);
    let x = gen_tensor(&mut rng, &[BATCH, SEQ_LEN, D_MODEL]);

    let config = MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず");
    let measurement = run(&config, || {
        let y = full_forward(&ops, black_box(&x), black_box(&layers));
        black_box(y);
    })
    .expect("Transformer ブロック forward（CUDA・改善前）の計測に失敗しました");

    let report = BenchReport::from_measurement(report_name("cuda"), "cuda", &measurement)
        .expect("BenchReport 構築に失敗しました");
    let json = report
        .to_json()
        .expect("BenchReport の JSON エンコードに失敗しました");
    println!("{json}");

    if let Ok(path) = std::env::var("BENCH_TRANSFORMER_REPORT_CUDA_PREFUSION") {
        std::fs::write(&path, &json).unwrap_or_else(|e| {
            panic!("BENCH_TRANSFORMER_REPORT_CUDA_PREFUSION={path} への書き込みに失敗しました: {e}")
        });
    }
}

/// #602（G-12）本体・改善後: Phase G 融合カーネル（`gemm_bias_act`
/// epilogue 融合・online softmax）適用後の CUDA 実測
/// （[`full_forward_fused`]。他条件は
/// [`transformer_block_forward_bench_cuda_prefusion`] と同一）。
#[test]
#[ignore = "計測用（実行時間長・CUDA 実機必須）。実機・計測環境で明示実行する"]
fn transformer_block_forward_bench_cuda_fused() {
    let ops = CudaBackendOps::new(0);
    let mut rng = Xorshift64Star::new(SEED);
    let layers = generate_layers(&mut rng);
    let x = gen_tensor(&mut rng, &[BATCH, SEQ_LEN, D_MODEL]);

    let config = MeasurementConfig::new(20, 20).expect("20/20 は下限ちょうどのため成功するはず");
    let measurement = run(&config, || {
        let y = full_forward_fused(&ops, black_box(&x), black_box(&layers));
        black_box(y);
    })
    .expect("Transformer ブロック forward（CUDA・改善後）の計測に失敗しました");

    let report = BenchReport::from_measurement(report_name_fused("cuda"), "cuda", &measurement)
        .expect("BenchReport 構築に失敗しました");
    let json = report
        .to_json()
        .expect("BenchReport の JSON エンコードに失敗しました");
    println!("{json}");

    if let Ok(path) = std::env::var("BENCH_TRANSFORMER_REPORT_CUDA_FUSED") {
        std::fs::write(&path, &json).unwrap_or_else(|e| {
            panic!("BENCH_TRANSFORMER_REPORT_CUDA_FUSED={path} への書き込みに失敗しました: {e}")
        });
    }
}

/// #602（G-12）受け入れ基準「改善前後の数値一致」: [`full_forward`]（改善前）
/// と [`full_forward_fused`]（改善後）の出力が REQ-2 複合判定（相対誤差
/// 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::parity::assert_parity`）
/// で一致することを検証する。tolerance は既存値のまま変更しない
/// （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を
/// 単独で緩和しない」）。CUDA 非搭載環境では `#[ignore]` 分離のため通常 CI
/// で実行されない（実機・計測環境で明示実行する）。
#[test]
#[ignore = "CUDA 実機必須。実機・計測環境で明示実行する"]
fn transformer_block_forward_cuda_fused_parity() {
    let ops = CudaBackendOps::new(0);
    let mut rng = Xorshift64Star::new(SEED);
    let layers = generate_layers(&mut rng);
    let x = gen_tensor(&mut rng, &[BATCH, SEQ_LEN, D_MODEL]);

    let prefusion = full_forward(&ops, &x, &layers);
    let fused = full_forward_fused(&ops, &x, &layers);

    assert_eq!(prefusion.shape(), fused.shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        "transformer block forward: CUDA prefusion vs fused (Phase G)",
        prefusion
            .contiguous()
            .as_slice()
            .expect("prefusion 出力は contiguous のはず"),
        fused
            .contiguous()
            .as_slice()
            .expect("fused 出力は contiguous のはず"),
    );
}
