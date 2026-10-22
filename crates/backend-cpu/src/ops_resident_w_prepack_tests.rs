//! 常駐 W の pack 済み B panel キャッシュ（#3200）の ops 層テスト（`ops.rs` の子モジュール）。
//!
//! ゲート ON／OFF の bit 一致、書き込み経路ごとの無効化、同一連結バッファ上の別 offset・
//! 複数ストアでのキー分離を固定する。キャッシュの実装は `gemm_blis/prepack.rs`、持ち主は
//! `memory::CpuBufferHandle`。決定記録は `docs/cpu-gemm-small-m-job-merge-prepack-decision.md`。

use super::*;
use fandhe_ai_tensor_core::pool::PoolZeroFill;

const L1: (usize, usize) = (784, 256);
const L2: (usize, usize) = (256, 10);

const OFF_L1: usize = 0;
const OFF_L2: usize = L1.0 * L1.1;
const OFF_W3: usize = L1.0 * L1.1 + L2.0 * L2.1;

fn rnd(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / 16777216.0) - 0.5
        })
        .collect()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice().unwrap().iter().map(|v| v.to_bits()).collect()
}

fn upload_1d(ops: &CpuBackendOps, v: Vec<f32>) -> DeviceBuffer<f32> {
    let n = v.len();
    ops.upload(&Tensor::new(v, &[n]).unwrap()).unwrap()
}

/// L1 W・L2 W・L1 と同形状の W3 を 1 本に連結した params_buf を作る。
fn params_buf(ops: &CpuBackendOps, seed: u32) -> DeviceBuffer<f32> {
    let mut v = rnd(L1.0 * L1.1, seed);
    v.extend(rnd(L2.0 * L2.1, seed + 1));
    v.extend(rnd(L1.0 * L1.1, seed + 2));
    upload_1d(ops, v)
}

fn view(buf: &DeviceBuffer<f32>, off: usize, kn: (usize, usize)) -> DeviceBufferView<'_> {
    // 形状は 'static に昇格できるリテラルで渡す（view が shape を借用するため）。
    let shape: &'static [usize] = if kn == L1 { &[784, 256] } else { &[256, 10] };
    DeviceBufferView::new(buf, off, shape).unwrap()
}

fn cache_len(buf: &DeviceBuffer<f32>) -> usize {
    buf.downcast_handle::<CpuBufferHandle>()
        .unwrap()
        .prepack_cache()
        .len()
}

fn rhs(
    a: &Tensor<f32>,
    buf: &DeviceBuffer<f32>,
    off: usize,
    kn: (usize, usize),
    enabled: bool,
) -> Tensor<f32> {
    gemm_resident_rhs_impl_with(a, view(buf, off, kn), None, Activation::None, enabled).unwrap()
}

fn sgd_cfg() -> SgdStepConfig {
    SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: false,
    }
}

#[test]
fn gate_off_never_touches_the_cache() {
    let ops = CpuBackendOps::new();
    let buf = params_buf(&ops, 1);
    let a = Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap();
    let _ = rhs(&a, &buf, OFF_L1, L1, false);
    assert_eq!(cache_len(&buf), 0);
}

#[test]
fn on_off_are_bit_identical_and_keys_do_not_collide() {
    let ops = CpuBackendOps::new();
    let buf = params_buf(&ops, 1);
    let a1 = Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap();
    let a2 = Tensor::new(rnd(64 * L2.0, 10), &[64, L2.0]).unwrap();
    for (a, off, kn) in [(&a1, OFF_L1, L1), (&a2, OFF_L2, L2), (&a1, OFF_W3, L1)] {
        for _ in 0..2 {
            let on = rhs(a, &buf, off, kn, true);
            let offv = rhs(a, &buf, off, kn, false);
            assert_eq!(bits(&on), bits(&offv), "off={off}");
        }
    }
    // L1 と同形状の W3（offset 違い）・L2（形状違い）が別エントリになる。
    assert_eq!(cache_len(&buf), 3);
    // L1 と W3 は同形状でも中身が違うので出力も違う（キー衝突の検出）。
    let l1 = rhs(&a1, &buf, OFF_L1, L1, true);
    let w3 = rhs(&a1, &buf, OFF_W3, L1, true);
    assert_ne!(bits(&l1), bits(&w3));
}

#[test]
fn bias_and_relu_match_with_gate_on() {
    let ops = CpuBackendOps::new();
    let buf = params_buf(&ops, 3);
    let bias = upload_1d(&ops, rnd(L1.1, 77));
    let a = Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap();
    for act in [Activation::None, Activation::Relu] {
        let run = |enabled: bool| {
            let bv = DeviceBufferView::new(&bias, 0, &[L1.1]).unwrap();
            gemm_resident_rhs_impl_with(&a, view(&buf, OFF_L1, L1), Some(bv), act, enabled).unwrap()
        };
        assert_eq!(bits(&run(true)), bits(&run(false)));
    }
}

#[test]
fn linear_forward_device_on_off_are_bit_identical() {
    let ops = CpuBackendOps::new();
    let buf = params_buf(&ops, 5);
    let a = ops
        .upload(&Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap())
        .unwrap();
    for act in [Activation::None, Activation::Relu] {
        let on = linear_forward_device_with(&a, view(&buf, OFF_L1, L1), None, act, true).unwrap();
        let off = linear_forward_device_with(&a, view(&buf, OFF_L1, L1), None, act, false).unwrap();
        assert_eq!(
            bits(&ops.download(&on).unwrap()),
            bits(&ops.download(&off).unwrap())
        );
    }
    assert_eq!(cache_len(&buf), 1);
}

/// train の流れを模す: forward（panel 構築）→ 更新 → 古い panel を使わない、を 3 step。
#[test]
fn weight_update_invalidates_panel_for_three_steps() {
    let ops = CpuBackendOps::new();
    let mut buf = params_buf(&ops, 11);
    let grad = upload_1d(&ops, rnd(buf.numel(), 99));
    let a = Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap();
    let mut prev = rhs(&a, &buf, OFF_L1, L1, true);
    assert_eq!(cache_len(&buf), 1);
    for step in 0..3 {
        ops.sgd_step_device(&mut buf, &grad, None, &sgd_cfg())
            .unwrap();
        assert_eq!(
            cache_len(&buf),
            0,
            "step {step}: 更新直後にキャッシュが残っている"
        );
        let on = rhs(&a, &buf, OFF_L1, L1, true);
        let off = rhs(&a, &buf, OFF_L1, L1, false);
        assert_eq!(bits(&on), bits(&off), "step {step}");
        assert_ne!(bits(&on), bits(&prev), "step {step}: 更新前の出力のまま");
        assert_eq!(cache_len(&buf), 1);
        prev = on;
    }
}

#[test]
fn every_write_path_clears_the_cache() {
    let ops = CpuBackendOps::new();
    let a = Tensor::new(rnd(64 * L2.0, 10), &[64, L2.0]).unwrap();

    // upload_into
    let mut buf = params_buf(&ops, 21);
    let _ = rhs(&a, &buf, OFF_L2, L2, true);
    assert_eq!(cache_len(&buf), 1);
    let patch = Tensor::new(vec![1.0f32; 4], &[4]).unwrap();
    ops.upload_into(&patch, &mut buf, OFF_L2).unwrap();
    assert_eq!(cache_len(&buf), 0);
    let on = rhs(&a, &buf, OFF_L2, L2, true);
    let off = rhs(&a, &buf, OFF_L2, L2, false);
    assert_eq!(bits(&on), bits(&off));

    // PoolZeroFill（pool 再利用時のゼロ初期化）
    assert_eq!(cache_len(&buf), 1);
    let mem = CpuMemory::new();
    let h = buf.downcast_handle_mut::<CpuBufferHandle>().unwrap();
    mem.zero_fill(h).unwrap();
    assert_eq!(cache_len(&buf), 0);
    let on = rhs(&a, &buf, OFF_L2, L2, true);
    assert!(on.as_slice().unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn separate_stores_do_not_share_or_leak_panels() {
    let ops = CpuBackendOps::new();
    let mut s1 = params_buf(&ops, 31);
    let s2 = params_buf(&ops, 31); // 同形状・同 offset・同内容の別ストア
    let grad = upload_1d(&ops, rnd(s1.numel(), 5));
    let a = Tensor::new(rnd(64 * L1.0, 9), &[64, L1.0]).unwrap();
    let before = rhs(&a, &s2, OFF_L1, L1, true);
    let _ = rhs(&a, &s1, OFF_L1, L1, true);
    ops.sgd_step_device(&mut s1, &grad, None, &sgd_cfg())
        .unwrap();
    // s1 の更新は s2 のキャッシュにも出力にも影響しない。
    assert_eq!(cache_len(&s1), 0);
    assert_eq!(cache_len(&s2), 1);
    assert_eq!(bits(&rhs(&a, &s2, OFF_L1, L1, true)), bits(&before));
    assert_ne!(bits(&rhs(&a, &s1, OFF_L1, L1, true)), bits(&before));
}

#[test]
fn memory_footprint_matches_formula() {
    // 追加メモリ = k * align_up(n, NR) * 4 バイト（決定記録 §12 の表の裏付け）。
    use crate::gemm_blis::prepack::panel_words;
    for (kn, nr, words) in [
        (L1, 12usize, 206_976usize),
        (L1, 16, 200_704),
        (L1, 32, 200_704),
        (L1, 4, 200_704),
        (L2, 12, 3_072),
        (L2, 16, 4_096),
        (L2, 32, 8_192),
        (L2, 4, 3_072),
    ] {
        assert_eq!(panel_words(kn.0, kn.1, nr), Some(words), "{kn:?} NR={nr}");
    }
}
