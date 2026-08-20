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
//! ## 出力突合（既定は非 fatal・`--strict-compare` で fail-closed に切替）
//!
//! 自作実装（`gemm_blis_parallel`）の出力 C を基準に、matrixmultiply・gemm crate の
//! 出力 C を統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
//! `.claude/rules/coding-rust.md`）で照合する。縮約順序・FMA 契約が実装間で
//! 異なるため bit 一致は要求しない。**この許容誤差の値自体は変更しない**
//! （coding-rust.md「バックエンド間数値一致テストの許容誤差を単独で緩和しない」の
//! 保護対象。本ハーネスは CPU/CUDA/Metal 自作バックエンド間比較ではなく OSS
//! 実装との比較のため同ルールの直接の適用対象ではないが、許容誤差の数値自体は
//! 予防的に据え置く）。
//!
//! `docs/oss-comparison-harness-decision.md`「出力突合とその限界」節に実測記録の
//! とおり、K が大きいサイズ（1024〜4096）では OSS 実装間の縮約順序差に由来する
//! 丸め誤差の蓄積により、複合判定をわずかに超える不一致が実装バグなしに発生しうる
//! ことが分かっている。本ハーネスの主目的（#735 各 Phase 完了時の素朴な再実行による
//! 性能再計測）を既定引数のまま成立させるため、既定では不一致を fatal にしない:
//! 各レコードの `output_match` フィールドに突合結果を記録し（不一致時は
//! `mismatch_detail` に詳細を併記）、標準エラー出力に警告を出したうえで性能計測は
//! 継続し、プロセスは 0 終了する。`--strict-compare` を指定した場合のみ、不一致を
//! 検出した時点で非 0 終了する（性能比較の前提となる正しさの検証自体は変更しない。
//! 単に「検証結果をどう扱うか」を既定と opt-in で分離する。REQ-8）。
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
/// `--sizes` フラグ自体が引数列に存在しない場合のみ [`DEFAULT_SIZES`] を使う。
/// `--sizes 512,1024` 形式でカンマ区切りの正整数列を受理する。パース失敗・
/// 0 以下の値・**`--sizes` はあるのに値トークンが続かない場合**（末尾に
/// `--sizes` だけが置かれた等）は、いずれも即座にエラーメッセージを表示して
/// 非 0 終了する（fail-closed。シェル展開・eval は一切使わない）。
///
/// レビュー指摘対応（イシュー #755）: 従来実装は `--sizes` の直後の値の
/// 有無を `Option` の `and_then` で素通しし、値が取れなければ「`--sizes`
/// 自体が未指定」の場合と区別せずに [`DEFAULT_SIZES`] へフォールバックして
/// いた。これは呼び出し側が明示的にサイズを指定しようとして誤って値を
/// 書き忘れた場合に、黙って既定サイズで実行してしまう fail-open な挙動
/// だったため、「フラグ不在」と「フラグはあるが値なし」を明示的に区別し、
/// 後者はエラー終了させる。
///
/// 各 `size` について `size * size` 要素の `Vec<f32>`（`main` の `a`/`b`/`c_*`
/// 系バッファ）を確保するため、`size * size` がバイト換算（`* 4`）を含め
/// `usize` の範囲に収まることを [`checked_mul`](usize::checked_mul) で明示検証する
/// （レビュー指摘対応。イシュー #755）。オーバーフローする巨大値は、
/// `unsafe` ブロック（`matrixmultiply::sgemm`・`gemm::gemm` 呼び出し）が
/// 前提とする「`size*size` 要素の連続確保」という SAFETY コメントの前提が
/// 実際のアロケーションで破綻しうる（`Vec` 確保自体は容量超過で panic するため
/// 未定義動作には至らないが、意図しない巨大確保・panic による異常終了を
/// 事前に型付きエラーで防ぐ。fail-closed）。
fn parse_sizes() -> Vec<usize> {
    let args: Vec<String> = std::env::args().collect();
    let flag_pos = args.iter().position(|a| a == "--sizes");

    let Some(idx) = flag_pos else {
        // `--sizes` フラグ自体が指定されていない: 既定サイズを使う。
        return DEFAULT_SIZES.to_vec();
    };

    let Some(raw) = args.get(idx + 1) else {
        // `--sizes` は指定されたが値トークンが続かない: 既定へフォールバック
        // せず fail-closed でエラー終了する（レビュー指摘対応。イシュー #755）。
        eprintln!("error: --sizes に値が指定されていない（例: --sizes 512,1024）");
        std::process::exit(2);
    };

    let mut sizes = Vec::new();
    for token in raw.split(',') {
        match token.trim().parse::<usize>() {
            Ok(n) if n > 0 => match validate_size_no_overflow(n) {
                Ok(()) => sizes.push(n),
                Err(msg) => {
                    eprintln!("error: --sizes の値 \"{token}\" が不正: {msg}");
                    std::process::exit(2);
                }
            },
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

/// `size` から確保する `size * size` 要素の `Vec<f32>`（4 バイト要素）が
/// `usize` 範囲でオーバーフローしないことを検証する。
///
/// `main` の各行列バッファ（`a`・`b`・`c_ref`・`c_mm`・`c_gemm`・`c_buf`）は
/// すべて `size * size` 要素で確保するため、要素数計算（`size * size`）と
/// バイト数換算（`* 4`）の両方を [`usize::checked_mul`] で検証する
/// （`unsafe` ブロックの SAFETY コメントが前提とする
/// 「size*size 要素の連続確保」を常に成立させるための事前検証）。
fn validate_size_no_overflow(size: usize) -> Result<(), String> {
    let elems = size
        .checked_mul(size)
        .ok_or_else(|| format!("size={size} は size*size で usize をオーバーフローする"))?;
    elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("size={size} は size*size*4 バイトで usize をオーバーフローする"))?;
    Ok(())
}

/// `--strict-compare` の指定有無を返す（レビュー指摘対応。イシュー #755）。
///
/// 指定時は出力突合 NG を検出した時点で非 0 終了する（従来の既定挙動）。
/// 未指定（既定）では、K が大きいサイズで実装バグなしに複合判定をわずかに
/// 超えうる既知の限界（`docs/oss-comparison-harness-decision.md` 参照）により
/// 本ハーネスの主目的（既定引数での素朴な再実行）が阻害されないよう、
/// 突合結果を JSON Lines の情報項目として記録するに留め非 fatal とする。
fn parse_strict_compare() -> bool {
    std::env::args().any(|a| a == "--strict-compare")
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
///
/// 判定式は `crates/backend-cpu/src/parity.rs::compare` と同方針に揃える
/// （レビュー指摘対応。イシュー #755）:
///
/// - 合格条件を肯定形（`pass = rel < REL_TOL || abs < ABS_TOL`）で先に判定し、
///   その否定を fail とする。旧実装は否定形（`rel >= REL_TOL && abs >= ABS_TOL`
///   を fail とする）を直接書いており、`reference`/`candidate` に NaN・Inf が
///   混入すると `rel`/`abs` も NaN になり、IEEE 754 上 `>=` 比較は NaN を含むと
///   常に false になるため両辺 false（fail 条件不成立）＝誤って「合格」と
///   判定してしまう欠陥があった（parity.rs 同コメント・Cursor Bugbot 指摘・
///   PR #239 で判明した既知のバグパターン）。肯定形の `pass` 判定なら `rel < TOL`
///   は NaN で必ず false になり `pass` も false（＝ fail）に倒れるため
///   fail-closed になる
/// - 相対誤差の分母（scale）を `reference` のみでなく
///   `max(|reference|, |candidate|, 1e-12)` とする（parity.rs と同一）。
///   `reference` が 0 近傍のとき分母を `reference` 単独に頼ると、`candidate`
///   側の実際のスケールを無視した過大な相対誤差になりうる。`abs_diff` が
///   `ABS_TOL` 未満なら（0 近傍の絶対誤差救済）相対誤差の値によらず合格になる
///   点は従来どおり（このため実質的な挙動差は「NaN/Inf を fail-closed で
///   検出できるようになった」点が主）
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
        let scale = r.abs().max(c.abs()).max(1e-12);
        let rel_diff = abs_diff / scale;
        let pass = rel_diff < REL_TOL || abs_diff < ABS_TOL;
        if !pass {
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
    /// 基準実装（`gemm_blis_parallel`）との出力突合結果（[`compare_outputs`] の
    /// 戻り値をそのまま保持する）。基準実装自身のレコードは自明に `Ok(())`。
    output_compare: Result<(), String>,
}

fn print_record(commit: &str, hw: &str, boundary: &str, r: &Record) {
    let tflops_median = tflops(r.size, r.measurement.median_secs);
    // Measurement の q1/q3 は所要時間（秒）の四分位のため、TFLOPS（時間の逆数）へ
    // 変換すると大小関係が反転する（`backend-cpu/examples/gemm_bench.rs::measure` と
    // 同じ理由・同じ対処）。表示ラベルどおり tflops_q1 <= tflops_q3 になるよう
    // 変換後に入れ替える。
    let tflops_q1 = tflops(r.size, r.measurement.q3_secs);
    let tflops_q3 = tflops(r.size, r.measurement.q1_secs);

    let output_match = r.output_compare.is_ok();
    // mismatch_detail は不一致時のみ内容を持つ文字列、一致時は JSON null。
    // detail の内容は compare_outputs が生成する内部固定文字列（数値のみを
    // 埋め込み、利用者入力は含まない）だが、念のため `"` のみエスケープする
    // （手書き JSON のため `serde_json` 相当のフルエスケープは行わないが、
    // 埋め込み文字列は内部生成のためこれで壊れない。OWASP A03: 経路上に
    // 外部入力なし）。
    let mismatch_detail_json = match &r.output_compare {
        Ok(()) => "null".to_string(),
        Err(detail) => format!("\"{}\"", detail.replace('"', "'")),
    };

    // 手書き JSON Lines（`serde_json` は本パッケージの依存に含めていない。
    // フィールド構成が固定・小規模で、値はすべて内部生成の数値・固定文字列
    // （利用者入力の埋め込みなし）のため、エスケープ不要な最小実装で足りる
    // という判断。OWASP A03: 外部入力をそのまま JSON へ埋め込む経路がないため
    // インジェクションの余地がない）。
    println!(
        "{{\"date\":\"{date}\",\"commit\":\"{commit}\",\"hw\":\"{hw}\",\"boundary\":\"{boundary}\",\"impl\":\"{impl_name}\",\"lib_version\":\"{lib_version}\",\"impl_threads\":{impl_threads},\"size\":{size},\"warmup\":{warmup},\"iters\":{iters},\"tflops_median\":{tflops_median:.4},\"tflops_q1\":{tflops_q1:.4},\"tflops_q3\":{tflops_q3:.4},\"output_match\":{output_match},\"mismatch_detail\":{mismatch_detail_json}}}",
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

/// 本番経路（テスト・examples 以外）で `.expect()` を使わない方針
/// （`.claude/rules/coding-rust.md`「コード品質」節）に従い、`main` は
/// `Result` を返して型付きエラーを呼び出し元（プロセス終了処理）へ伝播する。
/// `bench_harness::run`・`backend_cpu::gemm_blis_parallel` はいずれも
/// `std::error::Error` 実装済みのエラー型を返すため `Box<dyn Error>` に集約する
/// （レビュー指摘対応。イシュー #755）。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes = parse_sizes();
    let strict_compare = parse_strict_compare();
    let commit = git_commit_short();
    let hw = format!(
        "{}/{} ({} logical cores)",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    let config = MeasurementConfig::default();

    eprintln!(
        "oss-gemm-compare: commit={commit} hw={hw} rayon_threads={} strict_compare={strict_compare}",
        rayon::current_num_threads()
    );

    let mut had_mismatch = false;

    for &size in &sizes {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        // 基準実装（自作・現行最適 CPU 経路）。突合の reference にも計測にも使う。
        let mut c_ref = vec![0.0f32; size * size];
        // 形状検証（`m`・`n`・`k` と `a`/`b`/`c` の長さ整合）は固定サイズの正方行列
        // （`size * size` 要素で確保した `a`・`b`・`c_ref`）で常に成立するため、
        // `GemmError` は実運用上発生しない想定だが、本番経路での `.expect()` 禁止
        // 方針（coding-rust.md）に従い `?` で型付きに伝播する。
        backend_cpu::gemm_blis_parallel(&a, &b, &mut c_ref, size, size, size)?;

        // matrixmultiply（既定 feature。threading 未有効のため単一スレッド）。
        let mut c_mm = vec![0.0f32; size * size];
        // SAFETY: a・b・c_mm は全て size*size 要素の連続 `Vec<f32>` で、
        // 行主導（row-major）レイアウトのため rs=size・cs=1 で strides を渡す。
        // matrixmultiply crate は安全な高レベル API を提供せず（`sgemm` が
        // 唯一のエントリポイント）、C 側は書き込み専用（beta=0.0 のため
        // 未初期化読み出しはない。上記 `vec![0.0f32; ...]` で初期化済みだが
        // beta=0.0 なら本来読み出されない）。
        unsafe {
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
        let compare_mm = compare_outputs(&c_ref, &c_mm, "matrixmultiply");
        if let Err(msg) = &compare_mm {
            // 既定では警告に留める（既知の限界。上記モジュールドキュメント参照）。
            // `--strict-compare` 指定時のみ非 0 終了の対象として集計する。
            eprintln!("::warning::size={size} 出力不一致（matrixmultiply）: {msg}");
            had_mismatch = true;
        }

        // gemm crate（Parallelism::Rayon(0) = 既定スレッド数で rayon 共有プールを使う）。
        let mut c_gemm = vec![0.0f32; size * size];
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
        unsafe {
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
        let compare_gemm = compare_outputs(&c_ref, &c_gemm, "gemm");
        if let Err(msg) = &compare_gemm {
            eprintln!("::warning::size={size} 出力不一致（gemm crate）: {msg}");
            had_mismatch = true;
        }

        // 計測本体（3 実装とも同一プロトコル・同一入力）。
        let mut c_buf = vec![0.0f32; size * size];
        // `bench_harness::run` の `ProtocolViolation` は `config`
        // （`MeasurementConfig::default()`。warmup=20・iters=20 で下限を満たす）が
        // 固定のため実運用上発生しない想定だが、`.expect()` 禁止方針
        // （coding-rust.md）に従い `?` で型付きに伝播する（レビュー指摘対応。
        // イシュー #755）。
        // `bench_harness::run` の計測クロージャは `FnMut()`（戻り値 `()`）の
        // ため `?` を直接使えない。計測クロージャ内で panic
        // （`.expect()`／`unreachable!` 等）を起こすと本番 CLI 経路（本
        // ハーネスの主目的である既定引数での再計測実行）に panic が漏れる
        // ため（レビュー指摘対応。イシュー #755）、エラーはループ外の
        // `Option` に捕捉するのみに留め、`bench_harness::run` 呼び出し後に
        // `?` で型付きに伝播する。
        let mut ref_gemm_err = None;
        let m_ref = bench_harness::run(&config, || {
            if let Err(e) = backend_cpu::gemm_blis_parallel(&a, &b, &mut c_buf, size, size, size) {
                ref_gemm_err = Some(e);
            }
        })?;
        if let Some(e) = ref_gemm_err {
            return Err(Box::new(e));
        }

        let m_mm = bench_harness::run(&config, || {
            // SAFETY: a・b・c_buf は全て size*size 要素の連続 `Vec<f32>`（上記の
            // `c_ref`/`c_mm` 計測時と同一バッファ形状・同一 row-major ストライド
            // 規約。rs=size・cs=1）。matrixmultiply crate は安全な高レベル API を
            // 提供せず `sgemm` が唯一のエントリポイントであり、計測ループ内で
            // 同一呼び出しを反復するだけで確保・借用の状態は変化しないため、
            // 上の一度目の呼び出しと同じ安全性根拠がそのまま成立する。beta=0.0
            // のため c_buf の未初期化読み出しはない。
            unsafe {
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
            }
        })?;

        let m_gemm = bench_harness::run(&config, || {
            // SAFETY: 上記 c_gemm 計測時と同一の row-major ストライド規約・同一
            // バッファ形状（a・b・c_buf は size*size 要素の連続 `Vec<f32>`）。
            // `read_dst=false` のため C の未初期化読み出しはなく、gemm crate は
            // 安全な高レベル API を提供しないためこの呼び出し 1 箇所に unsafe を
            // 閉じる。alpha/beta の役割は matrixmultiply と逆（`dst := alpha*dst +
            // beta*lhs*rhs`）なため積の係数は beta 側（1.0）に置く
            // （`gemm-0.19.0/src/gemm.rs` `gemm_fallback` で確認済み。上記
            // c_gemm 計測時のコメント参照）。
            unsafe {
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
            }
        })?;

        let records = [
            Record {
                impl_name: "self_gemm_blis_parallel",
                lib_version: env!("CARGO_PKG_VERSION"),
                impl_threads: rayon::current_num_threads(),
                size,
                measurement: m_ref,
                output_compare: Ok(()),
            },
            Record {
                impl_name: "matrixmultiply",
                lib_version: "0.3.11",
                impl_threads: 1,
                size,
                measurement: m_mm,
                output_compare: compare_mm,
            },
            Record {
                impl_name: "gemm",
                lib_version: "0.19.0",
                impl_threads: rayon::current_num_threads(),
                size,
                measurement: m_gemm,
                output_compare: compare_gemm,
            },
        ];
        for record in &records {
            print_record(&commit, &hw, "device_resident", record);
        }
    }

    if had_mismatch {
        eprintln!(
            "::warning::出力突合 NG（統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を検出した。\
             各 JSON Lines レコードの output_match / mismatch_detail を参照。\
             既知の限界は docs/oss-comparison-harness-decision.md「出力突合とその限界」節を参照。"
        );
        if strict_compare {
            eprintln!(
                "::error::--strict-compare 指定のため出力突合 NG を fatal として扱い非 0 終了する"
            );
            std::process::exit(1);
        }
    }

    Ok(())
}
