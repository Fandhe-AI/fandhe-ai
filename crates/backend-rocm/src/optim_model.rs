//! `kernels_optim.rs` の 2 カーネルを CPU 上で (block, thread) ごとに再現するテスト専用モデル
//! （イシュー #3440・R9）。
//!
//! 役割: AMD 実機がない環境で、1 次元グリッド分解・`gid >= numel` ガード・各要素がちょうど 1 回
//! 書かれること・`use_momentum == 0` で velocity を読まず書かないこと・カーネル式（FMA は
//! `__builtin_fmaf` 相当の `f32::mul_add`、それ以外は素の演算）を構造的に検証する。
//! `RocmBackendOps::with_host_model` の経路（`ops/optim.rs`）が呼ぶ。
//!
//! 限界: `sqrt` と `/` は Rust の IEEE 754 演算で、**GPU の `sqrtf`・除算を模したものではない**
//! （実機での bit 一致は #3159 で測る）。HIPRTC のコンパイル結果そのものの検証ではない。

use crate::launch::for_each_thread_gid;
use crate::optim::{AdamKernelParams, SgdKernelParams};

/// `sgd_step_f32`。`velocity` は `use_momentum` のときだけ触れる（無効時は読み書きしない）。
pub(crate) fn sgd_step(
    param: &mut [f32],
    grad: &[f32],
    mut velocity: Option<&mut [f32]>,
    p: &SgdKernelParams,
) {
    let numel = param.len();
    assert_eq!(grad.len(), numel);
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let i = gid as usize;
        let pv = param[i];
        let mut g = grad[i];
        if p.weight_decay != 0.0 {
            g = p.weight_decay.mul_add(pv, g);
        }
        if p.use_momentum {
            let v = velocity
                .as_deref_mut()
                .expect("use_momentum なら velocity が必要（呼び出し側が検証済み）");
            let prev = v[i];
            let b = if p.is_first_step {
                g
            } else {
                p.momentum.mul_add(prev, (1.0 - p.dampening) * g)
            };
            v[i] = b;
            g = if p.nesterov {
                p.momentum.mul_add(b, g)
            } else {
                b
            };
        }
        param[i] = pv - p.lr * g;
        writes[i] += 1;
    });
    assert!(
        writes.iter().all(|&w| w == 1),
        "各要素はちょうど 1 回書かれる"
    );
}

/// `adam_step_f32`。
pub(crate) fn adam_step(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    p: &AdamKernelParams,
) {
    let numel = param.len();
    assert!(grad.len() == numel && m.len() == numel && v.len() == numel);
    let mut writes = vec![0u8; numel];
    for_each_thread_gid(numel, |gid| {
        if gid >= numel as i64 {
            return;
        }
        let i = gid as usize;
        let pv = param[i];
        let g = grad[i];
        let g_eff = if p.use_coupled_wd {
            p.weight_decay.mul_add(pv, g)
        } else {
            g
        };
        let p_eff = if p.decoupled { pv * p.decay_factor } else { pv };
        let m_new = p.beta1.mul_add(m[i], (1.0 - p.beta1) * g_eff);
        let v_new = p.beta2.mul_add(v[i], ((1.0 - p.beta2) * g_eff) * g_eff);
        m[i] = m_new;
        v[i] = v_new;
        let denom = v_new.sqrt() / p.bias_correction2_sqrt + p.eps;
        param[i] = p_eff - (p.step_size * m_new) / denom;
        writes[i] += 1;
    });
    assert!(
        writes.iter().all(|&w| w == 1),
        "各要素はちょうど 1 回書かれる"
    );
}
