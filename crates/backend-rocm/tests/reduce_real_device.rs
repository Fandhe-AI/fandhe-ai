//! ROCm 実機（MI300X・gfx942・wave64）での縮約 5 演算を CPU 参照と突き合わせる実機 parity テスト
//! （イシュー #3434。実測は #3159 段 B）。
//!
//! - 役割: `RocmReduce`（sum/max/min）・`RocmArgReduce`（argmax/argmin）の出力を `CpuBackendOps` と
//!   `fandhe_ai_backend_cpu::compare`（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5
//!   未満。定数は変更しない）で判定する。max/min/argmax/argmin と軸指定 sum は CPU と完全一致を
//!   assert する。全軸 sum は結合順序（block/lane の butterfly）が CPU と異なるため複合判定のみ
//!   assert し、bit 不一致は観測値（`bit_mismatch`）として出力するだけにする。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。`run_rocm_parity.sh` はクレートの `--ignored`
//!   テストを全件実行するため本ファイルも自動で含まれ、**本ファイルの失敗は run の終了コードに影響する**。
//!   記録は `docs/perf/logs/rocm-parity-3159/README.md` の「reduce（#3434）」節。
//! - ログ行は GEMM の `ROCM_PARITY` 行と同じ形（`ROCM_PARITY_REDUCE` で始まる）。
//! - 初の実機実行では wave64 での `__shfl_xor`（double 版を含む）・`__uint_as_float`・
//!   `#pragma clang fp contract` の HIPRTC コンパイル可否も同時に確認される。

use fandhe_ai_backend_cpu::{CpuBackendOps, compare};
use fandhe_ai_backend_rocm::{GfxTarget, RocmArgReduce, RocmReduce, is_available};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0 + 1e-3
        })
        .collect()
}

fn t(d: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(d.to_vec(), shape).unwrap()
}

/// 1 件ぶんの記録行を出力し、複合判定の fail 数を返す。
fn record(name: &str, got: &[f32], want: &[f32]) -> usize {
    let r = compare(got, want).unwrap_or_else(|e| panic!("compare {name}: {e:?}"));
    let bits = got
        .iter()
        .zip(want)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    eprintln!(
        "ROCM_PARITY_REDUCE\t{name}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bits}",
        r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
    );
    r.fail_count
}

/// 全要素のビット一致を要求する（max/min/軸 sum）。観測行も出す。
fn record_exact(name: &str, got: &[f32], want: &[f32]) {
    let fails = record(name, got, want);
    assert_eq!(fails, 0, "{name}: 複合判定 fail");
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{name}: CPU と bit 不一致"
    );
}

/// numel: 1・255・256・257・ブロック上限（256*1024）超（grid-stride が複数周）。
const ALL_SIZES: [usize; 6] = [1, 255, 256, 257, 5000, 256 * 1024 + 1];

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn reduce_all_matches_cpu_reference() {
    if !is_available() {
        eprintln!("ROCm 非搭載のためスキップ");
        return;
    }
    let r = RocmReduce::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let cpu = CpuBackendOps::new();
    let mut fails = 0;
    for &n in &ALL_SIZES {
        let x = data(n, 100 + n as u64);
        let tx = t(&x, &[n]);
        let want_sum = cpu.sum(&tx, None).unwrap().as_slice().unwrap()[0];
        fails += record(
            &format!("sum all n={n}"),
            &[r.run_sum_all_f32(&x).unwrap()],
            &[want_sum],
        );
        let want_max = cpu.max(&tx, None).unwrap().as_slice().unwrap()[0];
        record_exact(
            &format!("max all n={n}"),
            &[r.run_max_all_f32(&x).unwrap()],
            &[want_max],
        );
        let want_min = cpu.min(&tx, None).unwrap().as_slice().unwrap()[0];
        record_exact(
            &format!("min all n={n}"),
            &[r.run_min_all_f32(&x).unwrap()],
            &[want_min],
        );
    }
    // f64 アキュムレータ: [1e8, 1, -1e8] の総和は 1（f32 なら 0）。
    assert_eq!(r.run_sum_all_f32(&[1e8, 1.0, -1e8]).unwrap(), 1.0);
    // NaN・±inf（max/min は NaN を伝播しない。CPU と一致）。
    let mut x = data(1000, 3);
    x[17] = f32::NAN;
    x[500] = f32::INFINITY;
    x[999] = f32::NEG_INFINITY;
    let tx = t(&x, &[1000]);
    record_exact(
        "max all special",
        &[r.run_max_all_f32(&x).unwrap()],
        cpu.max(&tx, None).unwrap().as_slice().unwrap(),
    );
    record_exact(
        "min all special",
        &[r.run_min_all_f32(&x).unwrap()],
        cpu.min(&tx, None).unwrap().as_slice().unwrap(),
    );
    assert_eq!(fails, 0, "全軸 sum の複合判定 fail 合計");
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn reduce_axis_matches_cpu_reference() {
    if !is_available() {
        eprintln!("ROCm 非搭載のためスキップ");
        return;
    }
    let r = RocmReduce::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let cpu = CpuBackendOps::new();
    let shapes: [&[usize]; 6] = [
        &[3, 4],
        &[2, 3, 5],
        &[1, 300],
        &[300, 1],
        &[257, 3],
        &[64, 784],
    ];
    for (n, shape) in shapes.iter().enumerate() {
        let numel: usize = shape.iter().product();
        let x = data(numel, 40 + n as u64);
        let tx = t(&x, shape);
        for axis in 0..shape.len() {
            let outer: usize = shape[..axis].iter().product();
            let inner: usize = shape[axis + 1..].iter().product();
            let axis_len = shape[axis];
            let name = format!("shape={shape:?} axis={axis}");
            let v = |t: Tensor<f32>| t.contiguous().as_slice().unwrap().to_vec();
            record_exact(
                &format!("sum axis {name}"),
                &r.run_sum_axis_f32(&x, outer, axis_len, inner).unwrap(),
                &v(cpu.sum(&tx, Some(axis)).unwrap()),
            );
            record_exact(
                &format!("max axis {name}"),
                &r.run_max_axis_f32(&x, outer, axis_len, inner).unwrap(),
                &v(cpu.max(&tx, Some(axis)).unwrap()),
            );
            record_exact(
                &format!("min axis {name}"),
                &r.run_min_axis_f32(&x, outer, axis_len, inner).unwrap(),
                &v(cpu.min(&tx, Some(axis)).unwrap()),
            );
        }
    }
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn arg_reduce_matches_cpu_reference() {
    if !is_available() {
        eprintln!("ROCm 非搭載のためスキップ");
        return;
    }
    let a = RocmArgReduce::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let cpu = CpuBackendOps::new();
    let idx = |t: Tensor<i32>| t.contiguous().as_slice().unwrap().to_vec();
    for &n in &[1usize, 255, 4095, 4096, 4097, 12_295, 256 * 1024 + 1] {
        let mut x = data(n, 200 + n as u64);
        if n > 8 {
            // タイ（最初の添字が勝つ）と NaN を混ぜる。
            x[1] = x[0];
            x[n / 2] = f32::NAN;
            x[n - 1] = x[n - 2];
        }
        let tx = t(&x, &[n]);
        assert_eq!(
            a.run_argmax_all_f32(&x).unwrap(),
            idx(cpu.argmax(&tx, None).unwrap())[0],
            "argmax all n={n}"
        );
        assert_eq!(
            a.run_argmin_all_f32(&x).unwrap(),
            idx(cpu.argmin(&tx, None).unwrap())[0],
            "argmin all n={n}"
        );
        eprintln!("ROCM_PARITY_REDUCE\targ all n={n}\texact=true");
    }
    // 全 NaN は 0。
    let nan = vec![f32::NAN; 10];
    assert_eq!(a.run_argmax_all_f32(&nan).unwrap(), 0);
    assert_eq!(a.run_argmin_all_f32(&nan).unwrap(), 0);
    for shape in [&[3usize, 4][..], &[2, 3, 5], &[257, 3], &[1, 300]] {
        let numel: usize = shape.iter().product();
        let mut x = data(numel, 9);
        x[1] = x[0];
        x[4] = f32::NAN;
        let tx = t(&x, shape);
        for axis in 0..shape.len() {
            let outer: usize = shape[..axis].iter().product();
            let inner: usize = shape[axis + 1..].iter().product();
            assert_eq!(
                a.run_argmax_axis_f32(&x, outer, shape[axis], inner)
                    .unwrap(),
                idx(cpu.argmax(&tx, Some(axis)).unwrap()),
                "argmax shape={shape:?} axis={axis}"
            );
            assert_eq!(
                a.run_argmin_axis_f32(&x, outer, shape[axis], inner)
                    .unwrap(),
                idx(cpu.argmin(&tx, Some(axis)).unwrap()),
                "argmin shape={shape:?} axis={axis}"
            );
            eprintln!("ROCM_PARITY_REDUCE\targ axis shape={shape:?} axis={axis}\texact=true");
        }
    }
}
