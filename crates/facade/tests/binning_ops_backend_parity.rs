//! `fandhe_ai_autodiff::binning_ops`（イシュー #2638・`histc`／`bincount`／
//! `searchsorted`／`bucketize`。facade 非公開のため `fandhe_ai_autodiff::binning_ops::*` を
//! 直接 use する。`crates/autodiff/src/binning_ops.rs` モジュール doc 参照）の
//! バックエンド間 parity テスト（`stat_reduce_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`CpuBackendOps::binning_*` を結線した tape と `Tape::new()`〈`NaiveOps`＝既定
//! `Unsupported` → 共有ホストカーネルへフォールバック〉の突き合わせ）: 索引・整数カウント・
//! `histc` のカウントは完全一致、重み付き `bincount` は REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`）で検証する。両経路とも同じ共有カーネルを
//! 呼ぶため、ここで確認するのは「`BackendOps` 経由の配線（形状検証・フォールバック）が結果を
//! 変えないこと」である。
//!
//! `bincount` 系は `&fandhe_ai_autodiff::Tape` を要する（整数入力に `Var` の受け手がなく
//! `BackendOps` へ到達する `tape` を明示引数にしているため）が、facade の `Tape` newtype からは
//! 取り出せない。そのため `fandhe_ai::tape()` と同じ結線
//! （`Tape::new_with_ops(Box::new(CpuBackendOps::new()))`。先例
//! `cpu_predict_resident_fixedcost_diag.rs`）で自前の tape を作って全演算に使う。
//!
//! `#[ignore]`（CUDA／Metal〈`cfg(target_os = "macos")` 限定〉の `BackendOps` を結線した tape
//! を CPU tape と比較）: 実機への到達手段が本エージェント実行環境にないため未実施のまま
//! GB10／Mac セッションへ申し送る（`docs/perf/logs/binning-ops-2638/README.md`）。CUDA／Metal は
//! いずれも本 4 演算の GPU カーネルを持たないため（既定 `Unsupported`）、この比較は「ホストへの
//! フォールバック経路が CPU tape と同じ結果になること」を確認するものであり、GPU カーネル自体の
//! parity ではない。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::binning_ops::{
    bincount, bincount_weighted, bucketize, histc, searchsorted,
};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

/// 決定的な `[2, 3, 4]` 入力（NaN・範囲外を含まない）。
fn input() -> Tensor<f32> {
    let data: Vec<f32> = (0..24)
        .map(|i| (((i * 13 + 5) % 29) as f32) * 0.21 - 3.0)
        .collect();
    t(data, &[2, 3, 4])
}

fn sorted_boundaries() -> Tensor<f32> {
    t(vec![-2.0, -1.0, 0.0, 0.5, 1.5, 2.5], &[6])
}

fn bin_indices() -> Tensor<i32> {
    Tensor::new((0..24).map(|i| (i * 7) % 11).collect(), &[24]).expect("shape 一致")
}

struct Outs {
    histc: Vec<f32>,
    histc_default: Vec<f32>,
    bincount: Vec<i32>,
    weighted: Tensor<f32>,
    search_left: Vec<i32>,
    search_right: Vec<i32>,
    search_batched: Vec<i32>,
    bucket_left: Vec<i32>,
    bucket_right: Vec<i32>,
}

fn compute(tape: &Tape) -> Outs {
    let x = tape.var(&input());
    let bounds = tape.var(&sorted_boundaries());
    let idx = bin_indices();
    let w = tape.var(&t(
        (0..24).map(|i| 0.25 * (i as f32) - 2.0).collect(),
        &[24],
    ));
    let batch_seq = tape.var(&t(vec![-1.0, 0.0, 1.0, 2.0, -2.0, -0.5, 0.5, 3.0], &[2, 4]));
    let batch_vals = tape.var(&t(vec![0.0, 1.5, -3.0, 0.5, 2.5, -0.5, 3.0, 1.0], &[2, 4]));
    let host = |t: Tensor<i32>| t.contiguous().host_slice().into_owned();
    Outs {
        histc: histc(&x, 7, -2.0, 2.0).unwrap().host_slice().into_owned(),
        histc_default: histc(&x, 5, 0.0, 0.0).unwrap().host_slice().into_owned(),
        bincount: host(bincount(tape, &idx, 14).unwrap()),
        weighted: bincount_weighted(tape, &idx, &w, 0).unwrap(),
        search_left: host(searchsorted(&bounds, &x, false).unwrap()),
        search_right: host(searchsorted(&bounds, &x, true).unwrap()),
        search_batched: host(searchsorted(&batch_seq, &batch_vals, false).unwrap()),
        bucket_left: host(bucketize(&x, &bounds, false).unwrap()),
        bucket_right: host(bucketize(&x, &bounds, true).unwrap()),
    }
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
}

fn assert_outs_match(label: &str, a: &Outs, b: &Outs) {
    assert_eq!(a.histc, b.histc, "{label}: histc");
    assert_eq!(a.histc_default, b.histc_default, "{label}: histc 既定範囲");
    assert_eq!(a.bincount, b.bincount, "{label}: bincount");
    assert_eq!(a.search_left, b.search_left, "{label}: searchsorted left");
    assert_eq!(
        a.search_right, b.search_right,
        "{label}: searchsorted right"
    );
    assert_eq!(
        a.search_batched, b.search_batched,
        "{label}: searchsorted batched"
    );
    assert_eq!(a.bucket_left, b.bucket_left, "{label}: bucketize left");
    assert_eq!(a.bucket_right, b.bucket_right, "{label}: bucketize right");
    assert_eq!(
        a.weighted.shape(),
        b.weighted.shape(),
        "{label}: weighted shape"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("{label}: bincount_weighted"),
        a.weighted.host_slice().as_ref(),
        b.weighted.host_slice().as_ref(),
    );
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = compute(&cpu_tape());
    let naive = compute(&Tape::new());
    assert_outs_match("cpu vs naive", &cpu, &naive);

    // 期待値の健全性（両者が同じ誤りで一致していないこと）。
    let data = input().host_slice().into_owned();
    // histc: カウント総和 = 範囲 [-2, 2] 内の要素数。
    let in_range = data.iter().filter(|&&v| (-2.0..=2.0).contains(&v)).count();
    assert_eq!(cpu.histc.iter().sum::<f32>() as usize, in_range);
    assert_eq!(cpu.histc.len(), 7);
    // 既定範囲は全要素を数える。
    assert_eq!(cpu.histc_default.iter().sum::<f32>() as usize, data.len());
    // bincount: 総和 = 入力長・長さ 14（minlength が最大値 + 1 = 11 より大きい）。
    assert_eq!(cpu.bincount.len(), 14);
    assert_eq!(cpu.bincount.iter().sum::<i32>(), 24);
    // 重み付き: 全ビンの総和 = 重みの総和（f64 アキュムレータで丸め誤差は微小）。
    let wsum: f32 = (0..24).map(|i| 0.25 * (i as f32) - 2.0).sum();
    let got: f32 = cpu.weighted.host_slice().iter().sum();
    assert!((got - wsum).abs() < 1e-3, "{got} vs {wsum}");
    // searchsorted: 素朴な線形走査と一致し、upper >= lower。
    let bounds = sorted_boundaries().host_slice().into_owned();
    for (i, &v) in data.iter().enumerate() {
        let lower = bounds.iter().filter(|&&b| b < v).count() as i32;
        let upper = bounds.iter().filter(|&&b| b <= v).count() as i32;
        assert_eq!(cpu.search_left[i], lower, "left[{i}]");
        assert_eq!(cpu.search_right[i], upper, "right[{i}]");
    }
    assert_eq!(cpu.bucket_left, cpu.search_left);
    assert_eq!(cpu.bucket_right, cpu.search_right);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/binning-ops-2638/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(ops: Box<dyn BackendOps + Send>, label: &str) {
    let device_tape = Tape::new_with_ops(ops);
    let cpu = compute(&cpu_tape());
    let dev = compute(&device_tape);
    assert_outs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 4 演算の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/binning-ops-2638/README.md 参照"]
fn metal_binning_ops_match_cpu_reference() {
    assert_device_matches_cpu(
        Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()),
        "metal",
    );
}

/// 4 演算の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/binning-ops-2638/README.md 参照"]
fn cuda_binning_ops_match_cpu_reference() {
    assert_device_matches_cpu(
        Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)),
        "cuda",
    );
}
