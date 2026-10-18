//! `kernels_reduce.rs` のカーネルを CPU 上で (block, thread, lane) ごとに逐語再現するテスト専用モデル
//! （イシュー #3434・R3）。
//!
//! 役割: AMD 実機がない環境で、wave 幅（32／64）から導いた `warp_sums` のバッファ長・2 段目の
//! 読み出し境界・grid-stride ループ・butterfly 結合・`f64` アキュムレータ・`fmaxf`／`fminf` の
//! 厳密選択を検証する。`__shfl_xor` は「全 lane が旧値を同時に読む」意味論で再現する。未初期化の
//! `warp_sums` 読みと範囲外アクセスは `ModelError` で検出する（`warp_model.rs` の
//! 「CUDA の 8 warp 固定は wave64 で破綻する」確認を、実カーネルの構造に対して行う）。
//! `RocmBackendOps` の HostModel 経路（`ops.rs`）もこのモデルを呼ぶ。HIPRTC のコンパイル結果や
//! 実機の挙動の検証ではない（実機確認は #3159）。

use crate::kernels_reduce::REDUCE_BLOCK_DIM;
use crate::reduce::{ReduceKind, reduce_num_blocks};
use crate::warp_geometry::WarpGeometry;

/// モデル内で検出した、実機なら未定義動作になる読み書き。
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ModelError {
    /// `warp_sums[i]` を書かれる前に読んだ。
    UninitRead(usize),
    /// `warp_sums[i]` がバッファ外。
    OutOfBounds(usize),
}

/// `warp_sums` のバッファ長と 2 段目の読み出し境界。通常は `WARPS_PER_BLOCK` から導くが、
/// CUDA 由来の 8 固定を再現するテスト用に差し替えられる。
#[derive(Clone, Copy)]
pub(crate) struct Cfg {
    pub(crate) g: WarpGeometry,
    pub(crate) buf_len: usize,
    pub(crate) read_bound: usize,
}

impl Cfg {
    /// `WARPS_PER_BLOCK = 256 / wave 幅` から導いた正しい構成。
    pub(crate) fn new(g: WarpGeometry) -> Self {
        let n = g
            .warps_per_block(REDUCE_BLOCK_DIM)
            .expect("REDUCE_BLOCK_DIM は有効な block_dim") as usize;
        Self {
            g,
            buf_len: n,
            read_bound: n,
        }
    }
}

/// `__shfl_xor(v, off, WARP_SIZE)` の butterfly 縮約（wave ごと・全 lane が旧値を同時に読む）。
fn butterfly<T: Copy>(vals: &mut [T], width: usize, half: usize, comb: &impl Fn(T, T) -> T) {
    for wave in vals.chunks_mut(width) {
        let mut off = half;
        while off > 0 {
            let old = wave.to_vec();
            for i in 0..wave.len() {
                let src = i ^ off;
                let other = if src < old.len() { old[src] } else { old[i] };
                wave[i] = comb(old[i], other);
            }
            off >>= 1;
        }
    }
}

/// カーネルの block 内 2 段縮約（wave 内 butterfly → `warp_sums` → wave 0 が再縮約）。
/// 戻りは wave 0 の lane 0 の値。`vals` は各スレッドの 1 段目の累積値（長さ `blockDim.x`）。
fn block_reduce<T: Copy>(
    cfg: Cfg,
    mut vals: Vec<T>,
    id: T,
    comb: &impl Fn(T, T) -> T,
) -> Result<T, ModelError> {
    let w = cfg.g.width() as usize;
    let half = cfg.g.half_width() as usize;
    butterfly(&mut vals, w, half, comb);
    // lane == 0 のスレッドが warp_sums[warp_id] を書く。
    let mut warp_sums: Vec<Option<T>> = vec![None; cfg.buf_len];
    for (warp_id, wave) in vals.chunks(w).enumerate() {
        *warp_sums
            .get_mut(warp_id)
            .ok_or(ModelError::OutOfBounds(warp_id))? = Some(wave[0]);
    }
    // __syncthreads() の後、wave 0 の各 lane が `lane < WARPS_PER_BLOCK` なら warp_sums[lane]。
    let mut second = vec![id; w];
    for (lane, slot) in second.iter_mut().enumerate() {
        if lane < cfg.read_bound {
            *slot = warp_sums
                .get(lane)
                .ok_or(ModelError::OutOfBounds(lane))?
                .ok_or(ModelError::UninitRead(lane))?;
        }
    }
    butterfly(&mut second, w, half, comb);
    Ok(second[0])
}

/// 全軸 2 段縮約の骨格（partial → finalize）。`x` は非空。
fn all_two_stage<T: Copy>(
    cfg: Cfg,
    x: &[f32],
    id: T,
    load: impl Fn(f32) -> T,
    comb: impl Fn(T, T) -> T,
    finish: impl Fn(T) -> f32,
) -> Result<f32, ModelError> {
    assert!(
        !x.is_empty(),
        "全軸縮約の入力は非空（空は ops 層が CPU へ委譲）"
    );
    let bd = REDUCE_BLOCK_DIM as usize;
    let nb = reduce_num_blocks(x.len()) as usize;
    let stride = nb * bd;
    // 1 段目: grid = nb ブロック。各スレッドは grid-stride ループで `idx < numel` を守る。
    let mut partial: Vec<Option<T>> = vec![None; nb];
    for (b, slot) in partial.iter_mut().enumerate() {
        let vals: Vec<T> = (0..bd)
            .map(|t| {
                let mut acc = id;
                let mut idx = b * bd + t;
                while idx < x.len() {
                    acc = comb(acc, load(x[idx]));
                    idx += stride;
                }
                acc
            })
            .collect();
        *slot = Some(block_reduce(cfg, vals, id, &comb)?);
    }
    // 2 段目: 1 ブロック（256 スレッド）が `idx < num_partials` を守って partial を読む。
    let vals: Vec<T> = (0..bd)
        .map(|t| {
            let mut acc = id;
            let mut idx = t;
            while idx < nb {
                acc = comb(acc, partial[idx].expect("partial は 1 段目で全て書かれる"));
                idx += bd;
            }
            acc
        })
        .collect();
    Ok(finish(block_reduce(cfg, vals, id, &comb)?))
}

/// 全軸縮約。`cfg` で `warp_sums` 構成を差し替えられる。
pub(crate) fn simulate_all_with(cfg: Cfg, kind: ReduceKind, x: &[f32]) -> Result<f32, ModelError> {
    match kind {
        ReduceKind::Sum => all_two_stage(cfg, x, 0.0f64, f64::from, |a, b| a + b, |s| s as f32),
        ReduceKind::Max => all_two_stage(cfg, x, f32::NEG_INFINITY, |v| v, f32::max, |s| s),
        ReduceKind::Min => all_two_stage(cfg, x, f32::INFINITY, |v| v, f32::min, |s| s),
    }
}

/// 全軸縮約（`g` から導いた正しい構成）。
pub(crate) fn simulate_all(
    g: WarpGeometry,
    kind: ReduceKind,
    x: &[f32],
) -> Result<f32, ModelError> {
    simulate_all_with(Cfg::new(g), kind, x)
}

/// 軸縮約（1 スレッド = 1 出力・軸を昇順に逐次）。wave 幅に依存しない。
/// 各出力がちょうど 1 回書かれることを検査する。
pub(crate) fn simulate_axis(
    kind: ReduceKind,
    x: &[f32],
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Vec<f32> {
    let total = outer * inner;
    assert_eq!(x.len(), total * axis_len);
    let mut out: Vec<Option<f32>> = vec![None; total];
    if total == 0 {
        return Vec::new();
    }
    let bd = REDUCE_BLOCK_DIM as usize;
    let grid = total.div_ceil(bd);
    for b in 0..grid {
        for t in 0..bd {
            let idx = b * bd + t;
            if idx >= total {
                continue;
            }
            let (o, i) = (idx / inner, idx % inner);
            let at = |a: usize| x[(o * axis_len + a) * inner + i];
            let v = match kind {
                ReduceKind::Sum => {
                    let mut acc = 0.0f64;
                    for a in 0..axis_len {
                        acc += f64::from(at(a));
                    }
                    acc as f32
                }
                ReduceKind::Max => {
                    let mut acc = f32::NEG_INFINITY;
                    for a in 0..axis_len {
                        acc = f32::max(acc, at(a));
                    }
                    acc
                }
                ReduceKind::Min => {
                    let mut acc = f32::INFINITY;
                    for a in 0..axis_len {
                        acc = f32::min(acc, at(a));
                    }
                    acc
                }
            };
            assert!(out[idx].is_none(), "出力 {idx} は 1 回だけ書かれる");
            out[idx] = Some(v);
        }
    }
    out.into_iter()
        .map(|v| v.expect("全出力が書かれる"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // 0 を含まない値（符号付きゼロの扱いは f32::max/fmaxf で未規定のため bit 比較から外す）。
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0 + 1e-3
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn cpu_all(kind: ReduceKind, x: &[f32]) -> f32 {
        let t = Tensor::new(x.to_vec(), &[x.len()]).unwrap();
        let ops = CpuBackendOps::new();
        let r = match kind {
            ReduceKind::Sum => ops.sum(&t, None),
            ReduceKind::Max => ops.max(&t, None),
            ReduceKind::Min => ops.min(&t, None),
        }
        .unwrap();
        r.as_slice().unwrap()[0]
    }

    fn cpu_axis(kind: ReduceKind, x: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), shape).unwrap();
        let ops = CpuBackendOps::new();
        let r = match kind {
            ReduceKind::Sum => ops.sum(&t, Some(axis)),
            ReduceKind::Max => ops.max(&t, Some(axis)),
            ReduceKind::Min => ops.min(&t, Some(axis)),
        }
        .unwrap();
        r.contiguous().as_slice().unwrap().to_vec()
    }

    const KINDS: [ReduceKind; 3] = [ReduceKind::Sum, ReduceKind::Max, ReduceKind::Min];
    const WAVES: [u32; 2] = [32, 64];

    /// 全軸: numel が 1・255・256・257・ブロック上限超（grid-stride が複数周する）。
    const ALL_SIZES: [usize; 7] = [1, 2, 255, 256, 257, 5000, 256 * 1024 + 1];

    #[test]
    fn all_reduce_matches_cpu_for_both_wave_widths() {
        for w in WAVES {
            let g = WarpGeometry::new(w).unwrap();
            for &n in &ALL_SIZES {
                let x = lcg_data(n, 7 + n as u64);
                for kind in KINDS {
                    let got = simulate_all(g, kind, &x).unwrap();
                    let want = cpu_all(kind, &x);
                    match kind {
                        // CPU は 4096 要素チャンクの f64 partial、GPU は block/lane の butterfly で
                        // 結合順序が異なるため bit 一致は期待せず統一複合判定で照合する。
                        ReduceKind::Sum => {
                            assert_parity(&format!("sum all w{w} n{n}"), &[got], &[want])
                        }
                        _ => assert_eq!(got.to_bits(), want.to_bits(), "{kind:?} w{w} n{n}"),
                    }
                }
            }
        }
    }

    #[test]
    fn integer_valued_sum_is_exact() {
        for w in WAVES {
            let g = WarpGeometry::new(w).unwrap();
            for &n in &ALL_SIZES {
                let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
                let want: f64 = x.iter().map(|&v| f64::from(v)).sum();
                assert_eq!(
                    simulate_all(g, ReduceKind::Sum, &x).unwrap(),
                    want as f32,
                    "w{w} n{n}"
                );
            }
        }
    }

    #[test]
    fn max_min_with_nan_and_infinities_match_cpu_bitwise() {
        let cases: Vec<Vec<f32>> = vec![
            vec![1.0, f32::NAN, 3.0, -2.0],
            vec![f32::NAN; 300],
            vec![f32::INFINITY, 1.0, f32::NEG_INFINITY],
            vec![f32::NEG_INFINITY; 5],
            vec![f32::INFINITY; 5],
            {
                let mut v = lcg_data(1000, 3);
                v[17] = f32::NAN;
                v[500] = f32::INFINITY;
                v[999] = f32::NEG_INFINITY;
                v
            },
        ];
        for w in WAVES {
            let g = WarpGeometry::new(w).unwrap();
            for x in &cases {
                for kind in [ReduceKind::Max, ReduceKind::Min] {
                    assert_eq!(
                        simulate_all(g, kind, x).unwrap().to_bits(),
                        cpu_all(kind, x).to_bits(),
                        "{kind:?} w{w} n{}",
                        x.len()
                    );
                }
            }
        }
    }

    #[test]
    fn sum_f64_accumulator_keeps_cancellation_case() {
        // f32 アキュムレータなら [1e8, 1, -1e8] の総和は 0 になるが、f64 なら 1 を保つ。
        let x = [1e8f32, 1.0, -1e8];
        for w in WAVES {
            let g = WarpGeometry::new(w).unwrap();
            assert_eq!(simulate_all(g, ReduceKind::Sum, &x).unwrap(), 1.0);
        }
        assert_eq!(simulate_axis(ReduceKind::Sum, &x, 1, 3, 1), vec![1.0]);
    }

    #[test]
    fn axis_reduce_matches_cpu_bitwise() {
        // [3,4] の軸 0/1、[2,3,5] の軸 0/1/2、[1,n]・[n,1]、ブロック境界をまたぐ出力数。
        let shapes: Vec<Vec<usize>> = vec![
            vec![3, 4],
            vec![2, 3, 5],
            vec![1, 300],
            vec![300, 1],
            vec![1, 257],
            vec![257, 3],
            vec![4, 1, 7],
        ];
        for (n, shape) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let x = lcg_data(numel, 40 + n as u64);
            for axis in 0..shape.len() {
                let (outer, axis_len, inner) =
                    crate::reduce::reduce_axis_layout(shape, axis).unwrap();
                for kind in KINDS {
                    assert_eq!(
                        bits(&simulate_axis(kind, &x, outer, axis_len, inner)),
                        bits(&cpu_axis(kind, &x, shape, axis)),
                        "{kind:?} shape={shape:?} axis={axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn axis_reduce_special_values_match_cpu_bitwise() {
        let mut x = lcg_data(6 * 5, 9);
        x[3] = f32::NAN;
        x[12] = f32::INFINITY;
        x[20] = f32::NEG_INFINITY;
        for axis in 0..2 {
            let (outer, axis_len, inner) =
                crate::reduce::reduce_axis_layout(&[6, 5], axis).unwrap();
            for kind in [ReduceKind::Max, ReduceKind::Min] {
                assert_eq!(
                    bits(&simulate_axis(kind, &x, outer, axis_len, inner)),
                    bits(&cpu_axis(kind, &x, &[6, 5], axis)),
                    "{kind:?} axis={axis}"
                );
            }
        }
    }

    /// CUDA 由来の `warp_sums[8]`・`lane < 8` 固定は wave64 で未初期化読みになる（実カーネルの
    /// 構造に対する再確認。`warp_model::cuda_fixed_eight_warps_breaks_on_wave64` と同型）。
    #[test]
    fn cuda_fixed_eight_warps_breaks_on_wave64_kernel_structure() {
        let x = vec![1.0f32; 1024];
        let g64 = WarpGeometry::new(64).unwrap();
        let fixed8 = Cfg {
            g: g64,
            buf_len: 8,
            read_bound: 8,
        };
        for kind in KINDS {
            assert_eq!(
                simulate_all_with(fixed8, kind, &x),
                Err(ModelError::UninitRead(4)),
                "{kind:?}"
            );
        }
        assert_eq!(simulate_all(g64, ReduceKind::Sum, &x).unwrap(), 1024.0);
        // wave32 では 8 固定が偶然成立する（CUDA の前提）。
        let g32 = WarpGeometry::new(32).unwrap();
        let fixed8_w32 = Cfg {
            g: g32,
            buf_len: 8,
            read_bound: 8,
        };
        assert_eq!(
            simulate_all_with(fixed8_w32, ReduceKind::Sum, &x).unwrap(),
            1024.0
        );
        // バッファが小さすぎれば範囲外。
        let small = Cfg {
            g: g32,
            buf_len: 4,
            read_bound: 4,
        };
        assert_eq!(
            simulate_all_with(small, ReduceKind::Max, &x),
            Err(ModelError::OutOfBounds(4))
        );
    }

    #[test]
    fn derived_cfg_matches_warps_per_block() {
        assert_eq!(Cfg::new(WarpGeometry::new(64).unwrap()).buf_len, 4);
        assert_eq!(Cfg::new(WarpGeometry::new(32).unwrap()).buf_len, 8);
    }

    #[test]
    fn empty_axis_shapes_return_empty() {
        assert!(simulate_axis(ReduceKind::Sum, &[], 2, 0, 0).is_empty());
        assert!(simulate_axis(ReduceKind::Max, &[], 0, 4, 3).is_empty());
    }
}
