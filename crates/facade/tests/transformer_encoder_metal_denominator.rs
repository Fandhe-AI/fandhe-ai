//! イシュー #2998: int8 格上げ条件 (c) の Metal 分母（案 ii: facade 代替分母）の計測。
//!
//! `docs/int8-quant-grade-up-verification-plan.md` §4 の条件 (c) は Transformer 複合ワークロードの
//! 対 PyTorch 比（量子化効果を測る分母）を要求する。Metal には `baseline_spec()` 同一定義の
//! 専用実行経路がなく、所有者判断（#2995・2026-10-11）で「facade `TransformerEncoderLayer`
//! （`Sequential::add_transformer_encoder`）による代替」（案 ii）に決まった。本ファイルは
//! その経路を `bench_harness::run` の計測プロトコル（warmup 20／iters 20）で測る。
//!
//! ## `baseline_spec()` 系列との差分（正は `docs/perf/logs/int8-quant-baseline-2609/README.md` §2.2）
//!
//! - FFN 活性化は ReLU 固定（baseline_spec・PyTorch 基準は GELU）。facade は GELU を公開しない
//! - autodiff tape 経由で、各演算はホスト常駐 `Tensor` を返す（per-op 転送を含む）
//! - 重み生成は autodiff `nn::init` のシード導出（`Xorshift64Star` 直接生成ではない）
//! - 反復ごとに `tape_for(Device::Metal)` を作り直す（`predict` と同じ運用）
//!
//! よって本計測値は `baseline_spec()` 同一定義の分母ではなく**代替分母**であり、
//! 合否判定・REQ-8 下限判定には使わない（記録のみ）。実機（Mac mini〈Apple M4〉）必須のため
//! `#[ignore]` で分離し、macOS 限定とする。

#![cfg(target_os = "macos")]

use std::hint::black_box;

use bench_harness::rng::Xorshift64Star;
use bench_harness::transformer_workload::{SEED, baseline_spec};
use bench_harness::{BenchReport, MeasurementConfig, run};
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tensor};

/// 代替分母の腕名（`baseline_spec()` 系列の `report_name("metal")` とは意図的に別名にする）。
const ARM_FACADE_RELU: &str = "transformer-encoder-facade-metal-relu";
/// `tape_for(Device::Metal)` 単体（存在検証の固定費）の補助計測名。
const ARM_TAPE_FOR: &str = "transformer-encoder-facade-metal-tape-for";
/// JSON レポートの任意出力先（`BENCH_TRANSFORMER_REPORT` と同運用。未設定なら stdout のみ）。
const REPORT_ENV: &str = "BENCH_TRANSFORMER_METAL_REPORT";

/// 壊れた出力を計測値として扱わないための fail-closed 検査（shape と有限性）。
fn assert_output_sane(out: &Tensor<f32>, expected_shape: &[usize]) {
    assert_eq!(out.shape(), expected_shape, "出力 shape が入力と一致しない");
    let dense = out.contiguous();
    let data = dense
        .as_slice()
        .expect("contiguous() 後は as_slice が Some");
    assert!(
        data.iter().all(|v| v.is_finite()),
        "出力に非有限値が含まれる"
    );
}

#[test]
#[ignore = "Metal 実機（Mac mini〈Apple M4〉）必須。#2998 の代替分母計測"]
fn transformer_encoder_facade_forward_bench_metal_relu() {
    let spec = baseline_spec();
    spec.head_dim()
        .expect("baseline_spec の d_model は n_heads で割り切れる");
    let shape = [spec.batch, spec.seq_len, spec.d_model];
    let x = Tensor::new(
        Xorshift64Star::new(SEED).fill_vec(shape.iter().product()),
        &shape,
    )
    .expect("入力データ長は shape と一致");
    let model = Sequential::new()
        .add_transformer_encoder(spec.d_model, spec.n_heads, spec.d_ff, SEED)
        .expect("モデル構築");

    let forward = || {
        let tape = fandhe_ai::tape_for(Device::Metal).expect("Metal 実機必須");
        let xv = tape.var(&x);
        model.forward(&tape, &xv).expect("forward").to_tensor()
    };

    // 計測前の健全性検査（壊れた出力で分母を記録しない）。
    assert_output_sane(&forward(), &shape);

    let cfg = MeasurementConfig::new(20, 20).expect("20/20 は TASK-8.1 下限を満たす");
    let m = run(&cfg, || {
        black_box(forward());
    })
    .expect("計測");
    let json = BenchReport::from_measurement(ARM_FACADE_RELU, "metal", &m)
        .expect("レポート構築")
        .to_json()
        .expect("JSON 化");
    println!("int8_c_metal_denominator arm=facade_relu report={json}");
    if let Ok(path) = std::env::var(REPORT_ENV) {
        std::fs::write(&path, &json)
            .unwrap_or_else(|e| panic!("{REPORT_ENV}={path} への書き込みに失敗しました: {e}"));
    }

    // 補助計測: tape_for 単体（デバイス存在検証）の固定費を分離して読めるようにする。
    let mt = run(&cfg, || {
        black_box(fandhe_ai::tape_for(Device::Metal).expect("Metal 実機必須"));
    })
    .expect("補助計測");
    let tjson = BenchReport::from_measurement(ARM_TAPE_FOR, "metal", &mt)
        .expect("レポート構築")
        .to_json()
        .expect("JSON 化");
    println!("int8_c_metal_denominator arm=facade_tape_for report={tjson}");
}
