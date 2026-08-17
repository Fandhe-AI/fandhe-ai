//! イシュー #594: online softmax 順伝播カーネル（NVRTC・log2(e) 事前
//! スケール + exp2f・補正係数スキップ・境界マスク）の CPU-CUDA 数値一致
//! 検証。
//!
//! `rmsnorm_parity.rs`（#592）と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `backend_cuda::CudaError::DriverUnavailable`／`NvrtcUnavailable` を
//! 確認して panic しないことのみ検証）と、実機必須の形状網羅・極値入力
//! （`#[ignore]`。DGX Spark GB10 等）を分離する。判定式・許容誤差は
//! 再定義せず `backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（自然指数 `f32::exp` を
//! 使う素朴な max-sub-exp-sum-div 実装）である。`onnx-interop` 側の素朴
//! softmax 実装との突き合わせは、`backend-cuda` の dev-dependencies へ
//! 新規の外部依存（advisor 指摘: workspace 内クレートであっても
//! `Cargo.toml`/`Cargo.lock` の変更は本イシューの「依存追加なし」契約と
//! 衝突しうるため安全側に見送る）を追加しないスコープ判断により本イシュー
//! では対象外とする（out-of-scope-tracking.md 対象。実装計画 §8 相当）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p backend-cuda --release --test softmax_parity -- --ignored --nocapture
//! ```

use backend_cuda::{CudaDevice, CudaError, CudaSoftmax};
use bench_harness::rng::Xorshift64Star;

mod common;

/// テスト専用 CPU 参照実装（自然指数 `f32::exp` を使う素朴な 3 パス
/// 実装。`out = exp(x - max(x)) / sum(exp(x - max(x)))`）。カーネルは
/// `exp2(x*log2(e))` を計算するため（`softmax.rs` モジュール冒頭コメント
/// 「意味論注記」参照）丸めは異なるが、一致判定は REQ-2 複合判定
/// （`backend_cpu::parity::assert_parity`）に依るため問題ない。
fn cpu_softmax_reference(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if cols == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out_row = &mut out[r * cols..(r + 1) * cols];
        for i in 0..cols {
            out_row[i] = exps[i] / sum;
        }
    }
    out
}

fn assert_softmax_parity(softmax: &CudaSoftmax, seed: u64, rows: usize, cols: usize) {
    let x_data = Xorshift64Star::new(seed).fill_vec(rows * cols);

    let gpu_out = softmax
        .run_softmax_f32(&x_data, rows, cols)
        .expect("CudaSoftmax::run_softmax_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_softmax_reference(&x_data, rows, cols);

    assert_eq!(gpu_out.len(), cpu_out.len());
    backend_cpu::parity::assert_parity(
        &format!("softmax cpu-cuda parity rows={rows} cols={cols}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`rmsnorm_parity.rs::
/// rmsnorm_parity_smoke_env_adaptive` と同じ分岐パターン: 環境不在を表す
/// 既知の variant（`DriverUnavailable`／`NvrtcUnavailable`）のみを早期
/// return の対象とし、それ以外は `panic!` する（CUDA/NVRTC が利用可能な
/// CI 環境で実際のバグが握りつぶされないようにする。codex-review 指摘・
/// PR #706 レビュー r3793473253 と同じ方針）。
#[test]
fn softmax_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaSoftmax::new(&device) {
        Ok(softmax) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_softmax_parity(&softmax, 701, 1, 8);
            assert_softmax_parity(&softmax, 703, 3, 1024);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（driver はあるが nvrtc が無い）。panic
            // しないことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaSoftmax::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
///
/// cols の網羅: 1（行長 1。出力は恒等的に 1.0）・8（極小）・1024（1 パス
/// 中位）・4096（1 パス上位）・4097（vec4 端要素・`cols % 4 != 0`）・
/// 8192（1 パス上限付近）・16384（2 パス強制）。rows の網羅: 1・3・33
/// （persistent 行ループを grid 超で回す）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn softmax_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    let cols_cases: &[usize] = &[1, 8, 1024, 4096, 4097, 8192, 16384];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 2000u64;
    for &cols in cols_cases {
        for &rows in rows_cases {
            seed += 1;
            assert_softmax_parity(&softmax, seed, rows, cols);
        }
    }
}

/// 極値入力の数値安定性（実装計画 §5「Step 5」）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn softmax_numerically_stable_for_extreme_inputs() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    // 全要素同値 → 一様分布 1/cols。
    let cols = 256usize;
    let uniform_x = vec![3.5f32; cols];
    let out = softmax
        .run_softmax_f32(&uniform_x, 1, cols)
        .expect("softmax must succeed for uniform input");
    let expected = 1.0f32 / cols as f32;
    for &v in &out {
        assert!(
            (v - expected).abs() < 1e-6,
            "uniform softmax output should be ~1/cols={expected}, got {v}"
        );
    }
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax output must sum to ~1.0, got {sum}"
    );

    // 大きな正値（±1e37 級）。
    let large_pos_x: Vec<f32> = vec![1.0e37, 1.0000001e37, 0.9999999e37, -1.0e37];
    let out = softmax
        .run_softmax_f32(&large_pos_x, 1, 4)
        .expect("softmax must succeed for large-magnitude input");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output must not contain NaN/Inf: {out:?}"
    );
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "softmax output must sum to ~1.0, got {sum}"
    );

    // 大きな負値のみ（最大値も非常に負）。
    let large_neg_x: Vec<f32> = vec![-1.0e30, -2.0e30, -1.5e30];
    let out = softmax
        .run_softmax_f32(&large_neg_x, 1, 3)
        .expect("softmax must succeed for large-negative input");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output must not contain NaN/Inf: {out:?}"
    );
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "softmax output must sum to ~1.0, got {sum}"
    );

    // `±f32::MAX` 付近（イシュー #594 PR #712 codex-review 指摘・P1
    // 修正の直接的な回帰テスト）。旧実装は `raw * scale`（`scale =
    // log2(e)`）を最大値減算より先に行っていたため、`f32::MAX` のような
    // 有限な極値が `+Inf` へオーバーフローし、続く `exp2f(Inf - Inf) =
    // NaN` が発生していた（`kernels_softmax.rs` 冒頭コメント「`log2(e)`
    // 事前スケール」参照）。
    let f32_max_x: Vec<f32> = vec![f32::MAX, f32::MAX, 0.0, -f32::MAX];
    let out = softmax
        .run_softmax_f32(&f32_max_x, 1, 4)
        .expect("softmax must succeed for ±f32::MAX input");
    assert!(
        out.iter().all(|v| v.is_finite() && !v.is_nan()),
        "output must not contain NaN/Inf for ±f32::MAX input: {out:?}"
    );
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "softmax output must sum to ~1.0 for ±f32::MAX input, got {sum}"
    );
    // 2 要素が `f32::MAX` で並ぶため、そのペアがおよそ 0.5 ずつを分け合う
    // （`-f32::MAX`・`0.0` は寄与がアンダーフローして無視できる）。
    assert!(
        (out[0] - 0.5).abs() < 1e-3 && (out[1] - 0.5).abs() < 1e-3,
        "±f32::MAX input: tied maxima should each get ~0.5, got {out:?}"
    );

    // 全要素が `-f32::MAX`（境界マスク値そのもの。イシュー #594 PR #712
    // codex-review 指摘・Cursor Bugbot 指摘・P1 の直接的な回帰テスト）。
    // 旧実装は `m` の初期値を有限マージン値（`-0.875 * f32::MAX`）に
    // していたため、行の全要素がこのマージン値未満（`-f32::MAX` は
    // `-0.875 * f32::MAX` より小さい）の場合に `m` が一度も更新されず
    // `l == 0` のまま `inv_l == Inf`・出力が `0 * Inf == NaN` になって
    // いた（`kernels_softmax.rs` 冒頭コメント「境界マスク定数」参照）。
    // 一様分布（`1/cols`）になるはずの正規の入力である。
    let all_min_x: Vec<f32> = vec![-f32::MAX; 8];
    let out = softmax
        .run_softmax_f32(&all_min_x, 1, 8)
        .expect("softmax must succeed for all-elements-at-mask-value input");
    assert!(
        out.iter().all(|v| v.is_finite() && !v.is_nan()),
        "output must not contain NaN/Inf for all -f32::MAX input: {out:?}"
    );
    let expected = 1.0f32 / 8.0;
    for &v in &out {
        assert!(
            (v - expected).abs() < 1e-6,
            "all -f32::MAX row should be uniform ~1/8={expected}, got {v}"
        );
    }

    // 行長 1 → 出力は恒等的に 1.0。
    let single_x = vec![42.0f32];
    let out = softmax
        .run_softmax_f32(&single_x, 1, 1)
        .expect("softmax must succeed for row length 1");
    assert_eq!(out, vec![1.0f32]);

    // 多行（grid をまたぐ行ループ + 極値混在）。
    let multi_row_x = vec![
        1.0, 2.0, 3.0, // 通常行
        5.0, 5.0, 5.0, // 一様行
        1.0e30, -1.0e30, 0.0, // 極値混在行
    ];
    let out = softmax
        .run_softmax_f32(&multi_row_x, 3, 3)
        .expect("softmax must succeed for multi-row extreme input");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output must not contain NaN/Inf: {out:?}"
    );
    for r in 0..3 {
        let row_sum: f32 = out[r * 3..(r + 1) * 3].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "row {r} softmax output must sum to ~1.0, got {row_sum}"
        );
    }
}

/// `run_fused`（`ops.rs::CudaBackendOps::run_fused`）経由の canonical
/// softmax プラン実行を CPU per-op 合成（`max → sub(broadcast) → exp →
/// sum → div(broadcast)`）と突き合わせる。CUDA 非搭載環境では
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// env-adaptive 分岐（`rmsnorm_parity.rs::
/// rmsnorm_run_fused_matches_cpu_composed_env_adaptive` と同じパターン）。
#[test]
fn softmax_run_fused_matches_cpu_composed_env_adaptive() {
    use tensor_core::device::BackendError;
    use tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

    let cols = 16usize;
    let x_data = Xorshift64Star::new(9201).fill_vec(cols);
    let x = Tensor::new(x_data.clone(), &[cols]).expect("valid tensor");

    // canonical softmax プラン（axis: None・全軸縮約。leaf 0=x,
    // 1=Max{None}(0), 2=Broadcast{None}(1), 3=Sub(0,2), 4=Exp(3),
    // 5=Sum{None}(4), 6=Broadcast{None}(5), 7=Div(4,6)）。
    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Max {
            input: 0,
            axis: None,
        },
        FusedOpKind::Broadcast {
            input: 1,
            axis: None,
        },
        FusedOpKind::Sub { lhs: 0, rhs: 2 },
        FusedOpKind::Exp { input: 3 },
        FusedOpKind::Sum {
            input: 4,
            axis: None,
        },
        FusedOpKind::Broadcast {
            input: 5,
            axis: None,
        },
        FusedOpKind::Div { lhs: 4, rhs: 6 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![cols], DType::F32, 1)
        .expect("canonical softmax plan must construct");

    let cuda = backend_cuda::CudaBackendOps::new(0);
    match cuda.run_fused(&plan, &[&x]) {
        Ok(fused_out) => {
            let composed = cpu_softmax_reference(&x_data, 1, cols);

            assert_eq!(fused_out.shape(), &[cols]);
            backend_cpu::parity::assert_parity(
                "softmax run_fused vs cpu composed (canonical plan)",
                fused_out.as_slice().expect("contiguous"),
                &composed,
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::run_fused: {other}"),
    }
}

/// CPU-CUDA 直接突合（イシュー #607）: `backend_cpu::softmax::
/// run_softmax_f32`（NEON/rayon 参照実装。`f32::exp` ベース）を GPU 出力
/// （`exp2` ベース）と直接比較する。実機必須（`#[ignore]`。CI では
/// コンパイルのみ）。丸めが異なるため一致判定は REQ-2 複合判定に依る
/// （モジュール冒頭コメント参照。実装計画 §4「Step 5」）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn softmax_matches_backend_cpu_directly() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let softmax = CudaSoftmax::new(&device).expect("softmax kernel compile must succeed");

    let rows = 3usize;
    let cols = 4097usize; // NEON 端要素を含む。
    let x_data = Xorshift64Star::new(32_001).fill_vec(rows * cols);

    let gpu_out = softmax
        .run_softmax_f32(&x_data, rows, cols)
        .expect("CudaSoftmax::run_softmax_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = backend_cpu::softmax::run_softmax_f32(&x_data, rows, cols)
        .expect("backend_cpu::softmax::run_softmax_f32 must succeed");

    backend_cpu::parity::assert_parity(
        "softmax cpu(backend_cpu)-cuda direct parity",
        &gpu_out,
        &cpu_out,
    );
}
