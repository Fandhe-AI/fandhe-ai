//! 推論スコープ内 jpw1 の opt-in（イシュー #4161）の結線テスト。
//!
//! 役割: `backend-cpu` の `override_infer_scope_jpw1_for_scope` で機構を ON にして、
//! 推論本体（`predict`・`predict_resident`・`predict_resident_input`）だけが jpw を下げ、
//! 学習経路（tape の forward／backward・`train_step_resident`）は下げないこと、および出力が
//! OFF と bit 一致することを CI（CPU）で確かめる。計測用の `--ignored` report テストは
//! A/B の腕判別（`docs/perf/logs/cpu-infer-jpw1-gb10-ab-4161/`）が使う。
//! applied は呼び出しスレッドごとのカウンタ。rayon ワーカー上の GEMM は数えない。

use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{Device, Tensor, compat::Sequential, tape_for};
use fandhe_ai_backend_cpu::{
    gb10_affinity_report, infer_scope_jpw1_report, override_infer_scope_jpw1_for_scope,
};

fn tensor(seed: u32, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let data = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Tensor::new(data, shape).unwrap()
}

/// bench-fandhe と同じ MLP（784→256→10・ReLU）。
fn model() -> Sequential {
    Sequential::new()
        .add_linear(784, 256, 11)
        .unwrap()
        .add_relu()
        .add_linear(256, 10, 12)
        .unwrap()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn applied() -> u64 {
    infer_scope_jpw1_report().applied
}

#[test]
fn inference_paths_apply_and_training_paths_do_not() {
    let m = model();
    let x = tensor(1, &[64, 784]);
    let y = tensor(2, &[64, 10]);
    let _on = override_infer_scope_jpw1_for_scope(true);

    let a0 = applied();
    let _ = m.predict(&x).unwrap();
    assert!(applied() > a0, "predict");

    let tape = tape_for(Device::Cpu).unwrap();
    let mut store = m.init_device_param_store(&tape).unwrap();
    let a1 = applied();
    let _ = m.predict_resident(&store, &x).unwrap();
    assert!(applied() > a1, "predict_resident");

    let rx = tape.upload_resident(&store, &x).unwrap();
    let ry = tape.upload_resident(&store, &y).unwrap();
    let a2 = applied();
    let _ = m.predict_resident_input(&store, &rx).unwrap();
    assert!(applied() > a2, "predict_resident_input");

    // 学習: tape の forward + backward
    let a3 = applied();
    {
        let t = fandhe_ai::tape();
        let out = m.forward(&t, &t.var(&x)).unwrap();
        let loss = out.sum(None).unwrap();
        let _ = t.backward(&loss).unwrap();
    }
    assert_eq!(applied(), a3, "tape forward/backward");

    // 学習: train_step_resident
    let a4 = applied();
    let config = SgdConfig::new(0.01);
    let loss = m
        .train_step_resident(&mut store, &rx, &ry, &config)
        .unwrap();
    let _ = loss.read(&mut store).unwrap();
    assert_eq!(applied(), a4, "train_step_resident");
}

#[test]
fn outputs_are_bit_identical_with_and_without_override() {
    let m = model();
    let x = tensor(3, &[64, 784]);
    let tape = tape_for(Device::Cpu).unwrap();
    let store = m.init_device_param_store(&tape).unwrap();
    let rx = tape.upload_resident(&store, &x).unwrap();

    let run = || {
        (
            bits(&m.predict(&x).unwrap()),
            bits(&m.predict_resident(&store, &x).unwrap()),
            bits(&m.predict_resident_input(&store, &rx).unwrap()),
        )
    };
    let off = {
        let _g = override_infer_scope_jpw1_for_scope(false);
        run()
    };
    let on = {
        let _g = override_infer_scope_jpw1_for_scope(true);
        run()
    };
    assert_eq!(off, on);
}

#[test]
fn default_gate_does_not_apply() {
    // override なし（定数 OFF）では推論でも applied は増えない。
    // ゲートを反転した計測ツリーでは意味が変わるため、定数 ON のときは検査しない。
    if infer_scope_jpw1_report().gate_default {
        return;
    }
    let m = model();
    let x = tensor(4, &[64, 784]);
    let a0 = applied();
    let _ = m.predict(&x).unwrap();
    assert_eq!(applied(), a0);
}

/// A/B の腕判別用 report（`--ignored --exact` で 1 件だけ実行する）。override は使わず、
/// 定数の状態をそのまま報告する。
#[test]
#[ignore = "A/B 計測の前提確認用 report（docs/perf/logs/cpu-infer-jpw1-gb10-ab-4161/）"]
fn report_infer_jpw1_probe() {
    let m = model();
    let x = tensor(5, &[64, 784]);
    let y = tensor(6, &[64, 10]);
    let gate = infer_scope_jpw1_report().gate_default;
    let tape = tape_for(Device::Cpu).unwrap();
    let store = m.init_device_param_store(&tape).unwrap();
    let rx = tape.upload_resident(&store, &x).unwrap();

    let probe = |path: &str, f: &mut dyn FnMut()| {
        let a0 = applied();
        f();
        println!(
            "probe=infer_jpw1 path={path} gate={gate} applied={}",
            applied() - a0
        );
    };
    probe("infer_fresh", &mut || {
        let _ = m.predict(&x).unwrap();
    });
    probe("infer_reuse", &mut || {
        let _ = m.predict_resident_input(&store, &rx).unwrap();
    });
    probe("train_fresh", &mut || {
        let t = fandhe_ai::tape();
        let out = m.forward(&t, &t.var(&x)).unwrap();
        let loss = out.sum(None).unwrap();
        let _ = t.backward(&loss).unwrap();
    });
    let config = SgdConfig::new(0.01);
    let mut store2 = m.init_device_param_store(&tape).unwrap();
    // store_id が store に束縛された rx/ry は store2 では使えないため専用ハンドルを作る。
    let rx2 = tape.upload_resident(&store2, &x).unwrap();
    let ry2 = tape.upload_resident(&store2, &y).unwrap();
    probe("train_reuse", &mut || {
        let l = m
            .train_step_resident(&mut store2, &rx2, &ry2, &config)
            .unwrap();
        let _ = l.read(&mut store2).unwrap();
    });
    println!(
        "probe=gb10_affinity enabled={}",
        gb10_affinity_report().enabled
    );
}
