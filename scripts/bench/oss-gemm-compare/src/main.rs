//! OSS 直接比較ハーネス（イシュー #755）の CPU 比較バイナリ。
//!
//! GEMM 第 2 次最適化ツリー（#735。Phase 4 親 #754）の各最適化完了時に、
//! 本体の CPU 最適経路（`backend_cpu::gemm_blis_parallel`。BLIS 5-loop +
//! rayon 並列）を、workspace 外部の OSS 実装（matrixmultiply・gemm crate）と
//! 同一プロトコル（`bench_harness::protocol::run`。warmup 20 回・計測 20 回・
//! 中央値/Q1/Q3）で計測し、機械可読な JSON Lines として出力する。
//!
//! 本パッケージはリポジトリ内・本体 workspace 外の独立 Cargo プロジェクトである
//! （`Cargo.toml` の `[workspace]` 空テーブル）。matrixmultiply・gemm crate は
//! 許容依存 8 区分（`.claude/rules/deps-policy.md`）の対象外の外部依存だが、
//! 本体 workspace の依存グラフには一切現れないためユーザー承認対象の「依存追加」に
//! 当たらないと判断している（設計判断の詳細は `docs/oss-comparison-harness-decision.md`）。
//!
//! ## 計測境界
//!
//! 3 実装とも「デバイス内」（ホスト側で確保済みの `Vec<f32>` を直接読み書きする、
//! GPU 計測でいう prepared 境界に相当。CPU のためホスト⇔デバイス転送区間はそもそも
//! 存在しない）で計測する。詳細な境界定義・Metal 側（MLX・PyTorch MPS）の対応する
//! 2 境界の定義は `docs/perf/oss-gemm-comparison-baseline.md` を参照。
//!
//! ## 出力突合（fail-closed）
//!
//! 自作実装（`gemm_blis_parallel`）の出力 C を基準に、matrixmultiply・gemm crate の
//! 出力 C を統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
//! `.claude/rules/coding-rust.md`）で照合する。縮約順序・FMA 契約が実装間で
//! 異なるため bit 一致は要求しない。不一致が 1 要素でもあれば非 0 終了する
//! （性能比較の前提となる正しさの検証を、性能実測の達成を理由に緩和しない。REQ-8）。
//!
//! ## スレッド構成（比較の公平性軸）
//!
//! `gemm_blis_parallel` は rayon 並列（既定スレッド数）、gemm crate は
//! `Parallelism::Rayon(0)`（既定スレッド数・rayon 共有スレッドプール）を明示指定する。
//! matrixmultiply は `threading` feature を有効化していないため常に単一スレッドで
//! 動作する（crates.io の既定 feature 構成。多くの利用者が採用する構成をそのまま
//! 比較対象とするための意図的な選択であり、単一スレッド版との比較であることを
//! 出力の `impl_threads` フィールドと `docs/perf/oss-gemm-comparison-baseline.md` の
//! 計測境界節に明記する）。各実装が実際に使ったスレッド数を JSON Lines の
//! `impl_threads` フィールドに記録し、再計測キャンペーン間の比較可能性を保つ。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{Measurement, MeasurementConfig};
use std::process::Command;

/// 本ハーネスの決定的シード。既存 `backend-cpu/examples/gemm_bench.rs::SEED` と
/// 同一値を用い、本体 CPU ベンチ・PoC-v2 系と同じ入力分布に揃える。
const SEED: u64 = 0xC0FFEE;

/// 既定の計測対象サイズ（正方行列 M=N=K）。
/// `docs/spec/03-poc` 系・既存 `gemm_bench.rs` と同一の系列に揃える。
const DEFAULT_SIZES: &[usize] = &[512, 1024, 2048, 4096];

/// 統一複合判定の許容誤差（`.claude/rules/coding-rust.md`）。
const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-5;

/// CLI 引数を検証して計測対象サイズ一覧を返す（OWASP A03: 外部入力の検証を
/// 数値パースの成否のみに委ねず、正の整数であることを明示的に確認する）。
///
/// 引数なしの場合は [`DEFAULT_SIZES`] を使う。`--sizes 512,1024` 形式で
/// カンマ区切りの正整数列を受理する。パース失敗・0 以下の値は即座に
/// エラーメッセージを表示して非 0 終了する（fail-closed。シェル展開・eval は
/// 一切使わない）。
fn parse_sizes() -> Vec<usize> {
    let args: Vec<String> = std::env::args().collect();
    let sizes_arg = args
        .iter()
        .position(|a| a == "--sizes")
        .and_then(|idx| args.get(idx + 1));

    let Some(raw) = sizes_arg else {
        return DEFAULT_SIZES.to_vec();
    };

    let mut sizes = Vec::new();
    for token in raw.split(',') {
        match token.trim().parse::<usize>() {
            Ok(n) if n > 0 => sizes.push(n),
            _ => {
                eprintln!(
                    "error: --sizes は正整数のカンマ区切りで指定する（不正な値: \"{token}\"）"
                );
                std::process::exit(2);
            }
        }
    }
    if sizes.is_empty() {
        eprintln!("error: --sizes に有効なサイズが 1 つも指定されなかった");
        std::process::exit(2);
    }
    sizes
}

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// `git rev-parse --short HEAD` を実行してコミット SHA を取得する。
/// 取得できない場合（.git 不在・git 未導入等）は計測自体は継続し `"unknown"` を返す
/// （JSON Lines の再現性メタデータは best-effort。計測失敗の理由にはしない）。
fn git_commit_short() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で 2 つの C 行列を照合する。
/// 不一致要素があれば、最初に見つかった不一致の情報を含むメッセージを返す。
fn compare_outputs(reference: &[f32], candidate: &[f32], label: &str) -> Result<(), String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "{label}: 出力長不一致（reference={}, candidate={}）",
            reference.len(),
            candidate.len()
        ));
    }
    for (i, (&r, &c)) in reference.iter().zip(candidate.iter()).enumerate() {
        let abs_diff = (r - c).abs();
        let rel_diff = abs_diff / r.abs().max(f32::EPSILON);
        if rel_diff >= REL_TOL && abs_diff >= ABS_TOL {
            return Err(format!(
                "{label}: index={i} reference={r} candidate={c} abs_diff={abs_diff} rel_diff={rel_diff}"
            ));
        }
    }
    Ok(())
}

/// 1 実装の 1 サイズぶんの計測結果（JSON Lines 1 行に対応）。
struct Record {
    impl_name: &'static str,
    lib_version: &'static str,
    impl_threads: usize,
    size: usize,
    measurement: Measurement,
}

fn print_record(commit: &str, hw: &str, boundary: &str, r: &Record) {
    let tflops_median = tflops(r.size, r.measurement.median_secs);
    // Measurement の q1/q3 は所要時間（秒）の四分位のため、TFLOPS（時間の逆数）へ
    // 変換すると大小関係が反転する（`backend-cpu/examples/gemm_bench.rs::measure` と
    // 同じ理由・同じ対処）。表示ラベルどおり tflops_q1 <= tflops_q3 になるよう
    // 変換後に入れ替える。
    let tflops_q1 = tflops(r.size, r.measurement.q3_secs);
    let tflops_q3 = tflops(r.size, r.measurement.q1_secs);

    // 手書き JSON Lines（`serde_json` は本パッケージの依存に含めていない。
    // フィールド構成が固定・小規模で、値はすべて内部生成の数値・固定文字列
    // （利用者入力の埋め込みなし）のため、エスケープ不要な最小実装で足りる
    // という判断。OWASP A03: 外部入力をそのまま JSON へ埋め込む経路がないため
    // インジェクションの余地がない）。
    println!(
        "{{\"date\":\"{date}\",\"commit\":\"{commit}\",\"hw\":\"{hw}\",\"boundary\":\"{boundary}\",\"impl\":\"{impl_name}\",\"lib_version\":\"{lib_version}\",\"impl_threads\":{impl_threads},\"size\":{size},\"warmup\":{warmup},\"iters\":{iters},\"tflops_median\":{tflops_median:.4},\"tflops_q1\":{tflops_q1:.4},\"tflops_q3\":{tflops_q3:.4}}}",
        date = chrono_like_utc_date(),
        impl_name = r.impl_name,
        lib_version = r.lib_version,
        impl_threads = r.impl_threads,
        size = r.size,
        warmup = r.measurement.warmup,
        iters = r.measurement.iters,
    );
}

/// UTC 日付（YYYY-MM-DD）を `SystemTime` から手計算する。`chrono` は許容依存
/// 8 区分外のため使わない（JSON Lines のメタデータ生成のみに外部クレートを
/// 追加する必要はない）。
fn chrono_like_utc_date() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    // civil_from_days（Howard Hinnant 方式）: エポック（1970-01-01）からの経過日数を
    // グレゴリオ暦の年月日へ変換する標準的な整数演算アルゴリズム。
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn main() {
    let sizes = parse_sizes();
    let commit = git_commit_short();
    let hw = format!(
        "{}/{} ({} logical cores)",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    let config = MeasurementConfig::default();

    eprintln!(
        "oss-gemm-compare: commit={commit} hw={hw} rayon_threads={}",
        rayon::current_num_threads()
    );

    let mut had_mismatch = false;

    for &size in &sizes {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        // 基準実装（自作・現行最適 CPU 経路）。突合の reference にも計測にも使う。
        let mut c_ref = vec![0.0f32; size * size];
        backend_cpu::gemm_blis_parallel(&a, &b, &mut c_ref, size, size, size)
            .expect("gemm_blis_parallel: 形状検証は固定サイズの正方行列で常に成立する");

        // matrixmultiply（既定 feature。threading 未有効のため単一スレッド）。
        let mut c_mm = vec![0.0f32; size * size];
        unsafe {
            // SAFETY: a・b・c_mm は全て size*size 要素の連続 `Vec<f32>` で、
            // 行主導（row-major）レイアウトのため rs=size・cs=1 で strides を渡す。
            // matrixmultiply crate は安全な高レベル API を提供せず（`sgemm` が
            // 唯一のエントリポイント）、C 側は書き込み専用（beta=0.0 のため
            // 未初期化読み出しはない。上記 `vec![0.0f32; ...]` で初期化済みだが
            // beta=0.0 なら本来読み出されない）。
            matrixmultiply::sgemm(
                size,
                size,
                size,
                1.0,
                a.as_ptr(),
                size as isize,
                1,
                b.as_ptr(),
                size as isize,
                1,
                0.0,
                c_mm.as_mut_ptr(),
                size as isize,
                1,
            );
        }
        if let Err(msg) = compare_outputs(&c_ref, &c_mm, "matrixmultiply") {
            eprintln!("::error::size={size} 出力不一致（matrixmultiply）: {msg}");
            had_mismatch = true;
        }

        // gemm crate（Parallelism::Rayon(0) = 既定スレッド数で rayon 共有プールを使う）。
        let mut c_gemm = vec![0.0f32; size * size];
        unsafe {
            // SAFETY: matrixmultiply と同じ row-major ストライド規約
            // （`gemm::gemm` は `*_rs`/`*_cs` を要素単位ストライドとして扱う）。
            // `read_dst=false` のため C の初期値は使われず、未初期化読み出しは
            // 発生しない。gemm crate は安全な高レベル API を提供しない
            // （エントリポイントが `unsafe fn` のみ）ため、この呼び出し 1 箇所に
            // unsafe を閉じる。
            //
            // gemm crate の契約は `dst := alpha*dst + beta*lhs*rhs`
            // （matrixmultiply・自作カーネルの一般的な「alpha が積、beta が
            // 既存 dst」という慣例とは alpha/beta の役割が逆）。ここでは
            // `read_dst=false` のため alpha は無視されるが、積の係数は beta
            // 側（1.0）に置く必要がある（`gemm-0.19.0/src/gemm.rs`
            // `gemm_fallback` の `accum = accum * beta; if read_dst { accum
            // += alpha * dst }` 実装で確認済み）。
            gemm::gemm(
                size,
                size,
                size,
                c_gemm.as_mut_ptr(),
                1,
                size as isize,
                false,
                a.as_ptr(),
                1,
                size as isize,
                b.as_ptr(),
                1,
                size as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        }
        if let Err(msg) = compare_outputs(&c_ref, &c_gemm, "gemm") {
            eprintln!("::error::size={size} 出力不一致（gemm crate）: {msg}");
            had_mismatch = true;
        }

        // 計測本体（3 実装とも同一プロトコル・同一入力）。
        let mut c_buf = vec![0.0f32; size * size];
        let m_ref = bench_harness::run(&config, || {
            backend_cpu::gemm_blis_parallel(&a, &b, &mut c_buf, size, size, size)
                .expect("固定サイズの正方行列で形状検証は常に成立する");
        })
        .expect("MeasurementConfig::default は下限を満たすため失敗しない");

        let m_mm = bench_harness::run(&config, || unsafe {
            matrixmultiply::sgemm(
                size,
                size,
                size,
                1.0,
                a.as_ptr(),
                size as isize,
                1,
                b.as_ptr(),
                size as isize,
                1,
                0.0,
                c_buf.as_mut_ptr(),
                size as isize,
                1,
            );
        })
        .expect("MeasurementConfig::default は下限を満たすため失敗しない");

        let m_gemm = bench_harness::run(&config, || unsafe {
            gemm::gemm(
                size,
                size,
                size,
                c_buf.as_mut_ptr(),
                1,
                size as isize,
                false,
                a.as_ptr(),
                1,
                size as isize,
                b.as_ptr(),
                1,
                size as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(0),
            );
        })
        .expect("MeasurementConfig::default は下限を満たすため失敗しない");

        let records = [
            Record {
                impl_name: "self_gemm_blis_parallel",
                lib_version: env!("CARGO_PKG_VERSION"),
                impl_threads: rayon::current_num_threads(),
                size,
                measurement: m_ref,
            },
            Record {
                impl_name: "matrixmultiply",
                lib_version: "0.3.11",
                impl_threads: 1,
                size,
                measurement: m_mm,
            },
            Record {
                impl_name: "gemm",
                lib_version: "0.19.0",
                impl_threads: rayon::current_num_threads(),
                size,
                measurement: m_gemm,
            },
        ];
        for record in &records {
            print_record(&commit, &hw, "device_resident", record);
        }
    }

    if had_mismatch {
        eprintln!(
            "::error::出力突合 NG（統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）"
        );
        std::process::exit(1);
    }
}
