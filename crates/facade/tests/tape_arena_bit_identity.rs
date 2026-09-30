//! host arena（`fandhe_ai_tensor_core::alloc`・イシュー #2104・opt-in 既定 OFF）の
//! bit 同一回帰テスト。
//!
//! arena は「確保したバッファの再利用」だけで数値を変えない契約
//! （`docs/tape-arena-reuse-design.md` の bit 同一の節）。small batch の多層 MLP
//! （Linear+ReLU 複数層・MSE・SGD）を arena OFF／ON で同一シード・同一 step 数
//! 学習し、各 step の loss・最終パラメータ・推論出力を `to_bits()` で完全一致比較する。
//! 使い捨て Tape（step ごとに新規）・同一 Tape の `reset` 反復・checkpoint 併用の
//! 3 経路を CI（実機不要・非 ignore）で検証する。ON 側では回収・再利用
//! （`recycled > 0`・`hit > 0`）も確認し、素通し経路だけを見て偽陽性にならないようにする。
//! facade の公開面は広げない（`alloc` は内部クレートを直接参照する）。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::alloc;

const STEPS: usize = 6;
const DIMS: [usize; 4] = [8, 16, 16, 4];
const BATCH: usize = 4;

/// 決定的な擬似乱数（LCG）。グローバル RNG に依存しない。
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*seed >> 33) as u32) as f32 / (u32::MAX >> 1) as f32) - 0.5
}

fn rand_tensor(seed: &mut u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| lcg(seed) * 0.5).collect();
    Tensor::new(data, shape).expect("shape と長さは一致")
}

fn floats(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .map(<[f32]>::to_vec)
        .unwrap_or_default()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    floats(t).iter().map(|x| x.to_bits()).collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    FreshTapePerStep,
    ResetSameTape,
    Checkpoint,
}

fn forward<'t>(
    tape: &'t fandhe_ai::Tape,
    params: &[Tensor<f32>],
    x: &Tensor<f32>,
    grad: bool,
    checkpoint: bool,
) -> (Vec<fandhe_ai::Var<'t>>, fandhe_ai::Var<'t>) {
    let vars: Vec<_> = params
        .iter()
        .map(|p| {
            if grad {
                tape.var(p)
            } else {
                tape.var_no_grad(p)
            }
        })
        .collect();
    let mut h = tape.var_no_grad(x);
    let layers = vars.len() / 2;
    for l in 0..layers {
        let z = h
            .matmul(&vars[2 * l])
            .and_then(|m| m.add(&vars[2 * l + 1]))
            .expect("forward");
        h = if l + 1 < layers { z.relu() } else { z };
        if checkpoint && l == 1 {
            let refs: Vec<&fandhe_ai::Var<'_>> = vars.iter().collect();
            h = h.checkpoint_from(&refs).expect("checkpoint");
        }
    }
    (vars, h)
}

/// 1 step の学習（forward・MSE・backward・ホスト SGD）。loss の bit 列と更新後パラメータを返す。
fn train_step(
    tape: &fandhe_ai::Tape,
    params: &[Tensor<f32>],
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    checkpoint: bool,
) -> (Vec<u32>, Vec<Tensor<f32>>) {
    let (vars, out) = forward(tape, params, x, true, checkpoint);
    let yv = tape.var_no_grad(y);
    let loss = out.mse_loss(&yv).expect("loss");
    let loss_bits = bits(&loss.value());
    let grads = tape.backward(&loss).expect("backward");
    let next = params
        .iter()
        .zip(vars.iter())
        .map(|(p, v)| {
            let g = grads.get(v).expect("get").expect("grad 到達");
            let upd: Vec<f32> = floats(p)
                .iter()
                .zip(floats(g).iter())
                .map(|(w, g)| w - 0.05 * g)
                .collect();
            Tensor::new(upd, p.shape()).expect("shape")
        })
        .collect();
    (loss_bits, next)
}

/// 学習 STEPS 回 + 推論 1 回を行い、全 loss・推論出力・最終パラメータの bit 列を返す。
fn run(mode: Mode) -> Vec<u32> {
    let mut seed = 7u64;
    let mut params: Vec<Tensor<f32>> = Vec::new();
    for w in DIMS.windows(2) {
        params.push(rand_tensor(&mut seed, &[w[0], w[1]]));
        params.push(rand_tensor(&mut seed, &[1, w[1]]));
    }
    let x = rand_tensor(&mut seed, &[BATCH, DIMS[0]]);
    let y = rand_tensor(&mut seed, &[BATCH, DIMS[3]]);
    let mut out_bits: Vec<u32> = Vec::new();
    let mut shared = fandhe_ai::tape();

    for _ in 0..STEPS {
        let (lb, next) = if mode == Mode::ResetSameTape {
            let r = train_step(&shared, &params, &x, &y, false);
            shared.reset();
            r
        } else {
            let tape = fandhe_ai::tape();
            train_step(&tape, &params, &x, &y, mode == Mode::Checkpoint)
        };
        out_bits.extend(lb);
        params = next;
    }

    // 推論（使い捨て Tape。Sequential::predict と同型の使い捨て構造）。
    {
        let tape = fandhe_ai::tape();
        let (_vars, h) = forward(&tape, &params, &x, false, false);
        out_bits.extend(bits(&h.value()));
    }
    for p in &params {
        out_bits.extend(bits(p));
    }
    out_bits
}

fn assert_off_on_identical(mode: Mode) {
    alloc::clear_thread_arena();
    let off = {
        let _g = alloc::override_enabled_for_scope(false);
        run(mode)
    };
    alloc::clear_thread_arena();
    alloc::reset_stats();
    let on = {
        let _g = alloc::override_enabled_for_scope(true);
        run(mode)
    };
    let s = alloc::stats();
    alloc::clear_thread_arena();
    assert!(
        s.recycled > 0,
        "ON で回収が 0 件（arena が働いていない）: {s:?}"
    );
    assert!(
        s.hit > 0,
        "ON で再利用が 0 件（arena が働いていない）: {s:?}"
    );
    assert_eq!(off.len(), on.len());
    assert!(off == on, "arena OFF/ON で bit 不一致");
}

#[test]
fn fresh_tape_per_step_is_bit_identical() {
    assert_off_on_identical(Mode::FreshTapePerStep);
}

#[test]
fn reset_same_tape_is_bit_identical() {
    assert_off_on_identical(Mode::ResetSameTape);
}

#[test]
fn checkpoint_is_bit_identical() {
    assert_off_on_identical(Mode::Checkpoint);
}

/// 既定は OFF（結線は両機体 A/B ADOPT 後の別 PR）。
#[test]
fn arena_is_default_off() {
    assert!(!alloc::is_enabled());
}
