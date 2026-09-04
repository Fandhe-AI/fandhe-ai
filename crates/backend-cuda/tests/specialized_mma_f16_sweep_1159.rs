//! イシュー #1159 用の実機診断テスト（GB10 sweep 証跡再現用。git 管理下）。
//!
//! `specialized_mma_parity.rs::
//! specialized_mma_f16_matches_default_and_reference_across_shapes` は
//! `check_cpu_reference=true` の形状で `fandhe_ai_backend_cpu::assert_parity`
//! （厳密ゼロ fail 判定・fail で panic）を呼ぶため、1 形状でも FAIL する
//! とそこでテストが止まり、残りの (形状, プリセット) の組が未評価のまま
//! 残る（#1134 で `(256,512,1024)` の `DYNAMIC_ALL` が FAIL し発覚）。
//!
//! 本テストは `assert_parity` の代わりに
//! `fandhe_ai_backend_cpu::compare`（panic しない `Result` 版。
//! `assert_parity` 自体は `compare` を呼んで fail 時に panic するだけの
//! 薄いラッパーであり、`compare` を直接使えば同一の複合判定ロジックを
//! 複製せずに「1 形状 fail してもテストを止めず全組を評価する」動作を
//! 実現できる）を直接呼び、`specialized_mma_parity.rs` と同一の形状表・
//! 同一のシード生成規則（`gen_ab`）・同一の 3 プリセットを全網羅する。
//!
//! `docs/perf/logs/specialized-mma-f16-sweep-1159/README.md` に記録した
//! GB10 実機 sweep 証跡（2 回実行・完全一致確認済み）は、本テストの
//! **前身**（`crates/backend-cuda/tests/specialized_mma_f16_sweep_1159.rs`
//! として実行したが git 管理外のまま破棄され、実行ログのみが証跡として
//! 残った一時ファイル）から得たものである。codex-review（本 PR #1181）
//! P2 指摘「診断テストのソースがリポジトリ内に存在せず将来再現できない」
//! を受け、前身ファイルと同名パスへ本テストを追加することで自己完結
//! 再現可能にした。**前身ファイルのソースは保存されておらずバイト単位
//! の再現ではない**が、`specialized_mma_parity.rs` の形状表・生成規則・
//! 複合判定を一切変更せず再利用しているため、同一環境（GB10・同一
//! tolerance 定数・同一カーネル）で再実行すれば同一の
//! `(形状, プリセット)` 別 fail/pass 判定・bit 一致判定が再現される
//! （tolerance 定数・カーネル・`specialized_mma_parity.rs` 本体・
//! `ParityBaseline` は本テスト追加でも一切変更していない）。
//!
//! 受け入れ判定には使わない（`.claude/rules/coding-rust.md`「TF32/f16
//! Tensor Core 経路の parity テスト判定方式」の対象外。
//! `specialized_mma_f16_triage.rs`・`gemm_mma_tf32_triage.rs` と同じ
//! 診断専用の位置づけ）。CUDA 実機（compute capability 8.0 以上・NVRTC
//! 搭載）必須のため `#[ignore]` で分離する。

use fandhe_ai_backend_cpu::compare;
use fandhe_ai_backend_cuda::{CompiledDims, CudaDevice, CudaMmaGemm, run_specialized_mma_f16};
use half::f16;

/// `specialized_mma_parity.rs::gen_ab` と同一の決定的生成規則（乱数生成
/// ロジックを複製せず、同一入力を再現するためだけに同型の呼び出しを
/// 持つ）。
fn gen_ab(seed: u64, m: u32, n: u32, k: u32) -> (Vec<f16>, Vec<f16>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    (a, b)
}

/// `specialized_mma_parity.rs::cpu_reference_f32` と同一手順（f16→f32→
/// `matmul_reference_fma`→f16 丸め→f32）。
fn cpu_reference_f32(a: &[f16], b: &[f16], m: u32, n: u32, k: u32) -> Vec<f32> {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect()
}

/// `specialized_mma_parity.rs::
/// specialized_mma_f16_matches_default_and_reference_across_shapes` と
/// 完全に同一の形状表（10 形状。CPU 参照検査可否のフラグも同一）。
/// 形状表自体を複製変更しないよう、値は同ファイルと 1 対 1 で対応させて
/// いる。
const CASES: &[(u32, u32, u32, bool)] = &[
    (4096, 4096, 4096, false),
    (64, 128, 32, true),
    (128, 256, 128, true),
    (256, 512, 1024, true),
    (40, 24, 72, true),
    (65, 136, 40, true),
    (63, 120, 24, true),
    (200, 264, 104, true),
    (1, 136, 40, true),
    (512, 64, 4096, false),
];

/// 全 10 形状 × 3 プリセット（`CompiledDims::{DYNAMIC_ALL, STATIC_NK,
/// STATIC_MNK}`）を、1 組が FAIL してもテストを止めずに評価し、機械可読
/// な行を `--nocapture` 出力へ残す（`docs/perf/logs/
/// specialized-mma-f16-sweep-1159/README.md` が参照する
/// `SWEEP_BITMATCH`/`SWEEP_ROW`/`SWEEP_SUMMARY` プレフィックス）。
///
/// - `SWEEP_BITMATCH`: 既定カーネルとの bit 一致検査（全 10 形状 × 3
///   プリセット。CPU 参照の有無に関わらず全組で実行）
/// - `SWEEP_ROW`: `check_cpu_reference=true` の形状のみ、CPU 参照との
///   複合判定統計（`assert_parity` が panic する内容と同一の
///   `CompareReport` を `compare()` から直接取得し、panic させずに
///   1 行で記録する）
/// - `SWEEP_SUMMARY`: 全体の pass/fail 組数の集計
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_f16_sweep_1159_all_shapes_and_presets() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let default_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let mut bit_match_ok = 0usize;
    let mut bit_match_total = 0usize;
    let mut cpu_pass = 0usize;
    let mut cpu_fail = 0usize;

    for (idx, &(m, n, k, check_cpu_reference)) in CASES.iter().enumerate() {
        let seed = 4000 + idx as u64;
        let (a, b) = gen_ab(seed, m, n, k);

        let default_c = default_gemm.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
            panic!("shape (m={m}, n={n}, k={k}): default run_f16 failed: {err}")
        });

        let c_ref_rounded = if check_cpu_reference {
            Some(cpu_reference_f32(&a, &b, m, n, k))
        } else {
            None
        };

        for compiled in [
            CompiledDims::DYNAMIC_ALL,
            CompiledDims::STATIC_NK,
            CompiledDims::STATIC_MNK,
        ] {
            let specialized_c = run_specialized_mma_f16(&device, compiled, &a, &b, m, n, k)
                .unwrap_or_else(|err| {
                    panic!(
                        "shape (m={m}, n={n}, k={k}) compiled={compiled:?}: \
                         run_specialized_mma_f16 failed: {err}"
                    )
                });

            let bit_match = default_c == specialized_c;
            bit_match_total += 1;
            if bit_match {
                bit_match_ok += 1;
            }
            println!(
                "SWEEP_BITMATCH shape={m}x{n}x{k} compiled={compiled:?} bit_match={bit_match}"
            );

            if let Some(c_ref_rounded) = &c_ref_rounded {
                let c_specialized_f32: Vec<f32> =
                    specialized_c.iter().map(|x| x.to_f32()).collect();
                match compare(&c_specialized_f32, c_ref_rounded) {
                    Ok(report) => {
                        if report.passes() {
                            cpu_pass += 1;
                        } else {
                            cpu_fail += 1;
                        }
                        println!(
                            "SWEEP_ROW shape={m}x{n}x{k} compiled={compiled:?} \
                             fail_count={}/{} max_abs_diff={:.6e} max_rel_err={:.6e} \
                             mean_abs_diff={:.6e} max_fail_abs_diff={:.6e} \
                             p999_abs_diff={:.6e}",
                            report.fail_count,
                            report.total,
                            report.max_abs_diff,
                            report.max_rel_err,
                            report.mean_abs_diff,
                            report.max_fail_abs_diff,
                            report.p999_abs_diff,
                        );
                    }
                    Err(err) => {
                        cpu_fail += 1;
                        println!(
                            "SWEEP_ROW shape={m}x{n}x{k} compiled={compiled:?} compare_error={err}"
                        );
                    }
                }
            }
        }
    }

    println!(
        "SWEEP_SUMMARY bit_match={bit_match_ok}/{bit_match_total} \
         cpu_reference_pass={cpu_pass} cpu_reference_fail={cpu_fail}"
    );

    // 本テストは診断専用（受け入れ判定には使わない。冒頭コメント参照）
    // のため、CPU 参照 FAIL があっても panic させない。bit 一致のみ
    // 契約違反として扱う（特化カーネルが既定カーネルと演算命令列・
    // アキュムレート順序を変えない契約は `kernels_mma.rs` 冒頭コメント
    // で不変条件として定義されているため、ここが崩れたら診断以前に
    // 検出する）。
    assert_eq!(
        bit_match_ok, bit_match_total,
        "特化カーネルが既定カーネルと bit 一致しない組があります（診断以前の契約違反）"
    );
}
