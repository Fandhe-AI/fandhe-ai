//! `shaders/linalg.metal` 4 カーネルの f32 ホスト逐語モデル（イシュー #2676・
//! 設計記録 `docs/backend-gpu-linalg-decision.md` §3・§5）。
//!
//! # 目的
//!
//! カーネル起動（新規 `unsafe`）がユーザー承認待ちの間（Stage A）も、
//! 「カーネルに写像した演算列が CPU 参照実装（f64）と統一複合判定で一致する」
//! ことを実機なし（Linux CI）で検証できるようにする。MSL は
//! `#pragma clang fp contract(off)`・`precise::sqrt`・`precise::divide` で FMA
//! 縮約なしの IEEE f32 を使うため、Rust の `*`／`-`／`/`／`f32::sqrt`（いずれも
//! 縮約せず正しく丸められる）と同じ結果になる。各出力要素の算術順序（k 昇順の
//! 逐次ループ）はカーネルと同一で、並列化は要素間のみ（decision §3）。
//!
//! 呼び出し元は `tests/linalg_model_parity.rs`（`CpuBackendOps` との突合）で、
//! Stage B では実機出力との bit 比較（診断）にも使う。`objc2` 系 FFI に触れない
//! ため `cfg(target_os = "macos")` を付けない（`adam_model.rs` と同じ判断）。

/// LU 分解の結果（MSL `linalg_lu_f32` の `a`・`perm`・`status` に対応）。
#[derive(Debug, Clone, PartialEq)]
pub struct LuModelOut {
    /// L（単位下三角・対角以下に乗数）と U を詰めた `n*n`（行優先）。
    pub lu: Vec<f32>,
    /// 行置換（`perm[i]` = 元の行番号）。
    pub perm: Vec<u32>,
    /// ピボットが厳密 0 で特異と判定された場合 `true`。
    pub singular: bool,
    /// 行入替の回数（det の置換符号用）。
    pub swap_count: u32,
}

/// `shaders/linalg.metal::linalg_lu_f32` の逐語モデル（部分ピボット LU）。
pub fn lu_factor_f32(a: &[f32], n: usize) -> LuModelOut {
    let mut lu = a.to_vec();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut swap_count = 0u32;
    for k in 0..n {
        let mut max_row = k;
        let mut max_val = lu[k * n + k].abs();
        for i in (k + 1)..n {
            let v = lu[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_row != k {
            for c in 0..n {
                lu.swap(k * n + c, max_row * n + c);
            }
            perm.swap(k, max_row);
            swap_count += 1;
        }
        let pivot = lu[k * n + k];
        if pivot == 0.0 {
            return LuModelOut {
                lu,
                perm,
                singular: true,
                swap_count,
            };
        }
        // フェーズ 1: 乗数（行ごと独立）。
        for i in (k + 1)..n {
            lu[i * n + k] /= pivot;
        }
        // フェーズ 2: 更新（要素ごと独立。乗数 0 の行は飛ばす）。
        for i in (k + 1)..n {
            let factor = lu[i * n + k];
            if factor == 0.0 {
                continue;
            }
            for c in (k + 1)..n {
                lu[i * n + c] -= factor * lu[k * n + c];
            }
        }
    }
    LuModelOut {
        lu,
        perm,
        singular: false,
        swap_count,
    }
}

/// `shaders/linalg.metal::linalg_lu_solve_f32` の逐語モデル。`b` は `n*k`
/// （行優先）で、右辺 1 列ごとに前進・後退代入する。`lu.singular` なら `None`
/// （カーネルは何も書かない）。
pub fn lu_solve_f32(lu: &LuModelOut, b: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
    if lu.singular {
        return None;
    }
    let mut x = vec![0.0f32; n * k];
    for c in 0..k {
        for i in 0..n {
            let mut sum = b[lu.perm[i] as usize * k + c];
            for j in 0..i {
                sum -= lu.lu[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = x[i * k + c];
            for j in (i + 1)..n {
                sum -= lu.lu[i * n + j] * x[j * k + c];
            }
            x[i * k + c] = sum / lu.lu[i * n + i];
        }
    }
    Some(x)
}

/// `shaders/linalg.metal::linalg_cholesky_f32` の逐語モデル（下三角のみ読む）。
/// 非正定値（対角が非有限または非正）は `None`。
pub fn cholesky_f32(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; n * n];
    for j in 0..n {
        let mut sum = a[j * n + j];
        for k in 0..j {
            sum -= l[j * n + k] * l[j * n + k];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return None;
        }
        let ljj = sum.sqrt();
        l[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Some(l)
}

/// `shaders/linalg.metal::linalg_qr_f32` の逐語モデル。戻り値は `(r_work, q_work)`
/// （`r_work`: `m*n` の上三角化済み作業行列、`q_work`: `m*k`）。対角符号の正規化は
/// `crate::linalg::qr_finalize`（ホスト）が行う。
pub fn qr_f32(a: &[f32], m: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let k = m.min(n);
    let mut r = a.to_vec();
    // 反射ベクトル保存領域（列 col の v は `v[col*m ..]` に長さ m-col）。
    let mut v = vec![0.0f32; m * k];
    let mut applied = vec![false; k];
    for col in 0..k {
        let len = m - col;
        let x: Vec<f32> = (0..len).map(|i| r[(col + i) * n + col]).collect();
        let norm_x = crate::linalg::scale_ssq_norm_f32(&x);
        if norm_x == 0.0 {
            continue;
        }
        let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
        let mut vv = x.clone();
        vv[0] -= alpha;
        let norm_v = crate::linalg::scale_ssq_norm_f32(&vv);
        if norm_v == 0.0 {
            continue;
        }
        for e in vv.iter_mut() {
            *e /= norm_v;
        }
        for c in col..n {
            let mut dot = 0.0f32;
            for (i, &vi) in vv.iter().enumerate() {
                dot += vi * r[(col + i) * n + c];
            }
            if dot == 0.0 {
                continue;
            }
            for (i, &vi) in vv.iter().enumerate() {
                let idx = (col + i) * n + c;
                r[idx] -= (2.0f32 * vi) * dot;
            }
        }
        v[col * m..col * m + len].copy_from_slice(&vv);
        applied[col] = true;
    }
    let mut q = vec![0.0f32; m * k];
    for i in 0..k {
        q[i * k + i] = 1.0;
    }
    for col in (0..k).rev() {
        if !applied[col] {
            continue;
        }
        let len = m - col;
        for c in 0..k {
            let mut dot = 0.0f32;
            for i in 0..len {
                dot += v[col * m + i] * q[(col + i) * k + c];
            }
            if dot == 0.0 {
                continue;
            }
            for i in 0..len {
                let idx = (col + i) * k + c;
                q[idx] -= (2.0f32 * v[col * m + i]) * dot;
            }
        }
    }
    (r, q)
}
