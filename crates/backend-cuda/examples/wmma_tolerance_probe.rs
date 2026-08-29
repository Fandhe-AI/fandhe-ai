//! Tensor Core（WMMA）経路の誤差分布実測ハーネス（TASK-11.1g・#186）。
//!
//! REQ-2 受け入れ基準「tensor core（WMMA/mma）化で TF32／f16 累算経路を
//! 導入する際は当該経路の数値一致閾値を実測に基づき再評価する」の実測
//! ステップを担う。TF32 経路（[`fandhe_ai_backend_cuda::CudaGemm::run_wmma_tf32`]。
//! `tests/gemm_wmma_tf32.rs`）・f16 WMMA 経路
//! （[`fandhe_ai_backend_cuda::CudaWmmaGemm::run_f16`]。`tests/cpu_cuda_wmma_parity.rs`）
//! それぞれについて、形状×シードごとに `fandhe_ai_backend_cpu::compare` の
//! `CompareReport`（誤差分布統計）を収集し、REQ-2 統一複合判定の閾値
//! （`fandhe_ai_backend_cpu::RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）に
//! 対する閾値マージンを算出して Markdown 表形式で stdout へ出力する
//! （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` へ転記する
//! 想定）。
//!
//! **本ハーネスは閾値・判定式を一切変更しない**。`fandhe_ai_backend_cpu::parity`
//! の定数・`compare` 関数をそのまま import して使い、ローカル複製・
//! 緩和は行わない（`.claude/rules/coding-rust.md`「バックエンド間数値
//! 一致テストの許容誤差を単独で緩和しない」・`delegation-impl.md`
//! 「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」）。
//!
//! `examples/` に置くのは `gemm_bench.rs`（`crates/backend-cpu/examples/`）
//! と同じ理由: `dev-dependencies`（`bench-harness`・`backend-cpu`）を
//! 利用しつつ通常の `cargo test`／CI では実行されず、ビルド検証のみが
//! CI を通過させるためである（self-hosted runner をベンチ実行で占有
//! しない。`ci.md`）。
//!
//! # 使い方
//!
//! ```text
//! cargo build --release -p fandhe-ai-backend-cuda --example wmma_tolerance_probe
//!
//! # 既定モード（s=1 単一。出力は #993 以前と同一形式）
//! ./target/release/examples/wmma_tolerance_probe
//!
//! # スケールスイープモード（イシュー #993。CLI 優先、未指定なら env）
//! ./target/release/examples/wmma_tolerance_probe --scales 0.1,1,10,100
//! WMMA_TOLERANCE_PROBE_SCALES=0.1,1,10,100 ./target/release/examples/wmma_tolerance_probe
//!
//! ./target/release/examples/wmma_tolerance_probe --help
//! ```
//!
//! CUDA driver／NVRTC 非搭載環境、または Tensor Core 非対応（compute
//! capability 7.0/8.0 未満）環境では、経路ごとに理由を表示して正常終了
//! する（`tests/gemm_wmma_tf32.rs`・`tests/cpu_cuda_wmma_parity.rs` と
//! 同じ環境適応の分岐パターン）。一方、shape 検証失敗・`compare` の長さ
//! 不一致・WMMA 起動時エラー等の**意図しない**計測エラーは行を出力した
//! うえで exit code 1 を返す（環境非対応の意図的スキップと区別する。
//! #186・PR #257 Codex Review 指摘対応）。`--scales` の不正値（数値以外・
//! 0 以下・非有限・重複・要素数上限超過・未知の引数）も使い方を表示した
//! うえで exit code 1 を返す（fail-closed。#993・`security.md` A03）。
//!
//! CUDA toolkit 非搭載環境で NVRTC をプロセス限定プロビジョニングした
//! 場合、`LD_LIBRARY_PATH`・`CUDA_INCLUDE_PATH` は **`cargo build` 時だけ
//! でなく実行時（本バイナリの起動時）にも必要**である。NVRTC は
//! `CudaGemm::new`／`CudaWmmaGemm::new` 呼び出し時に実行時コンパイルを
//! 行う（動的ロードした `libnvrtc` を使う）ため、ビルド後に環境変数が
//! 失われた状態で実行すると NVRTC 利用不可と誤報告される
//! （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §1 の手順参照。
//! #186・PR #257 Codex Review 指摘対応）。
//!
//! ## スケールスイープ（#993）
//!
//! `docs/perf/cuda-tensor-core-tolerance-evaluation.md` §3.3 は、絶対誤差
//! 救済閾値候補（TF32 2.535e-2・f16 4.883e-4）が `Xorshift64Star` の
//! `[-1, 1)` 入力限定の暫定値であり、GEMM 絶対誤差が入力スケール `s` に
//! 対し概ね `s²` で変化するため代表スケールでの実測スイープが必要と
//! 記録している。`--scales`／`WMMA_TOLERANCE_PROBE_SCALES` で指定した
//! 各スケール `s` について、`SHAPES`・`SEEDS` は変えずに入力（`[-1,1)`
//! の一様分布）を `s` 倍して計測する。スイープモードの出力表には
//! `scale`・`kernel`（実際に実行されたカーネル種別。TF32 は 3 段選択
//! `staged`／`staged-swizzle`／`opt`／`basic`、f16 は `wmma_f16` 固定）・
//! `ref_nonfinite`（CPU 参照出力の非有限要素数。f16 は `s` が大きいと
//! 出力が f16 表現範囲〈最大 65504〉を超えて ±Inf になりうるため、該当
//! 行は `s²` 比例性の分析から除外する目安として付す）の列を追加する。
//! スケール未指定（CLI・env とも未指定）の既定モードは現行どおり
//! `s = 1` 単一・現行 13 列のままで、`x * 1.0` は IEEE 754 上 bit 単位で
//! 恒等（f16 側は「乱数取得後にスケールを乗じてから丸める」順序を
//! 取り、`s = 1.0` は `next_f32() * 1.0` の丸めと同一）のため出力は
//! #993 以前と完全に一致する（A2）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, RELATIVE_TOLERANCE, compare, matmul_reference_fma,
};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaWmmaGemm};
use half::f16;

/// 形状セット（`tests/gemm_wmma_tf32.rs`・`tests/cpu_cuda_wmma_parity.rs` の
/// `#[ignore]` 形状網羅テストと同じ形状に、K スイープ（M=N=256 固定で
/// K=256/512/1024/4096）を追加し、桁落ち蓄積と K の関係を見る）。
///
/// **両 ignored parity suite の全形状を含む**（#186・PR #257 Codex Review
/// 指摘「probe の shape リストが TF32 の (1,1,1) と f16 の (17,19,23) を
/// 欠いており、両 ignored parity suite を網羅していると主張しているが実際は
/// 網羅していない」対応）:
/// - `tests/gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes` の
///   `(1, 1, 1)`（K タイル未満の極小形状）を追加した。
/// - `tests/cpu_cuda_wmma_parity.rs::wmma_f16_matches_reference_across_shapes`
///   の `(17, 19, 23)`（m=17, n=19, k=23）を追加した。TF32 側の
///   `(17, 23, 19)`（m=17, n=23, k=19）とは n・k が入れ替わっており別形状
///   のため、いずれも残している。
///
/// **意図的な重複**: 「256x256x256 (block tile x8)」と「256x256x256
/// (K sweep base)」は m=n=k=256 で完全に同一の形状であり、シード導出式
/// （形状インデックスに依存しない `m` 由来のシード）により同一入力・同一
/// 結果になる。前者は既存 `#[ignore]` テストの形状網羅網（block tile 系列
/// の完成）としての意味、後者は K スイープ（256/512/1024/4096）の起点として
/// の意味を持たせるため、重複を承知のうえで両方残している
/// （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §2 の実測表では
/// 重複分をマージして 1 行に記録する）。
///
/// **K=512 スイープ点**（#186・PR #257 Codex Review 指摘「K 依存の単調傾向の
/// 主張が 256/1024/4096 の sweep と 512x512x512（M/N・シードも変わる）の
/// 結果を混在させている」対応）: 「256x256x512 (K sweep)」を追加し、
/// M=N=256 固定のまま K のみを 256→512→1024→4096 と揃えて比較できるように
/// した。既存の「512x512x512」行は M/N も変わる別条件（シード導出も異なる）
/// のため、K 単調性の主張には流用しない。
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("32x32x32 (block tile)", 32, 32, 32),
    ("64x64x64 (block tile x2)", 64, 64, 64),
    ("128x128x128 (block tile x4)", 128, 128, 128),
    ("256x256x256 (block tile x8)", 256, 256, 256),
    ("512x512x512 (block tile x16)", 512, 512, 512),
    ("1x1x1 (sub-K-tile, TF32 suite)", 1, 1, 1),
    ("17x23x19 (non-multiple edge, TF32 suite)", 17, 23, 19),
    ("17x19x23 (non-multiple edge, f16 suite)", 17, 19, 23),
    ("33x31x65 (non-multiple edge)", 33, 31, 65),
    ("100x100x100 (non-multiple edge)", 100, 100, 100),
    ("130x70x90 (non-multiple edge)", 130, 70, 90),
    ("64x96x128 (non-square)", 64, 96, 128),
    ("256x256x256 (K sweep base)", 256, 256, 256),
    ("256x256x512 (K sweep)", 256, 256, 512),
    ("256x256x1024 (K sweep)", 256, 256, 1024),
    ("256x256x4096 (K sweep, PoC-v2-5 stress)", 256, 256, 4096),
];

/// 各形状 5 シード（5 回計測の中央値方針〈coding-rust.md〉に整合させ、
/// 単一シードの偶然の一致・不一致に結論を左右されないようにする）。
const SEEDS: &[u64] = &[1, 2, 3, 4, 5];

/// `--scales`／`WMMA_TOLERANCE_PROBE_SCALES` で受理する要素数の上限
/// （fail-closed の一環。形状×シード×スケールの計測時間が無制限に
/// 膨らむのを防ぐ。#993）。
const MAX_SCALES: usize = 16;

/// 入力スケールの指定状態（#993）。
///
/// `Default`（CLI・env とも未指定）は `s = 1` 固定・現行 13 列の出力
/// （A2: #993 以前と完全一致）。`Sweep` は明示指定（内容が `1` 単独でも
/// スイープ出力モードへ切り替わる。R2 のとおり「未指定」のみが legacy
/// 扱い）で、スケール列・カーネル種別列・`ref_nonfinite` 列を追加した
/// 16 列の表を出力する。
#[derive(Debug, Clone, PartialEq)]
enum ScaleConfig {
    Default,
    Sweep(Vec<f64>),
}

impl ScaleConfig {
    /// 計測ループが走査するスケール列。`Default` は `s = 1.0` の単一要素
    /// （`scaled_f32_inputs`／`scaled_f16_inputs` の `s = 1.0` は無変更の
    /// `fill_vec`／`fill_vec_f16` と bit 単位で一致するため、レガシー
    /// モードの出力互換〈A2〉はこの単一要素ループで自然に満たされる）。
    fn scales(&self) -> &[f64] {
        match self {
            ScaleConfig::Default => &[1.0],
            ScaleConfig::Sweep(v) => v,
        }
    }

    fn is_sweep(&self) -> bool {
        matches!(self, ScaleConfig::Sweep(_))
    }
}

/// `--scales`／env の値（カンマ区切り文字列）をパースする。
///
/// fail-closed（`security.md` A03: 外部入力の検証を先に行う）: 各要素は
/// `trim()` 後に `f64` パースでき、かつ有限・正・非重複でなければならず、
/// 要素数は [`MAX_SCALES`] 以下に制限する。不正入力はシェル・ファイル
/// パス・カーネルソースへ渡らず、ここで `Err` として弾く。
fn parse_scales(raw: &str) -> Result<Vec<f64>, String> {
    let mut scales: Vec<f64> = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(format!("invalid --scales value: empty element in '{raw}'"));
        }
        let value: f64 = trimmed
            .parse()
            .map_err(|_| format!("invalid --scales value: '{trimmed}' is not a number"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "invalid --scales value: '{trimmed}' must be a positive finite number"
            ));
        }
        if scales.contains(&value) {
            return Err(format!(
                "invalid --scales value: duplicate scale '{trimmed}'"
            ));
        }
        scales.push(value);
    }
    if scales.is_empty() {
        return Err("invalid --scales value: no scales given".to_string());
    }
    if scales.len() > MAX_SCALES {
        return Err(format!(
            "invalid --scales value: too many scales ({} > {MAX_SCALES})",
            scales.len()
        ));
    }
    Ok(scales)
}

/// CLI 引数（`--help`／`-h` は呼び出し元 `main` が先に処理済みの前提）と
/// 環境変数 `WMMA_TOLERANCE_PROBE_SCALES` から [`ScaleConfig`] を決定する。
/// CLI が優先し、いずれも未指定なら `ScaleConfig::Default`（R1/R2）。
///
/// 未知の引数は fail-closed でエラーにする（無音無視は「指定したはずの
/// `--scales` が効いていない」を気づかせないため。`cuda_floor_bench.rs`
/// の `env_override` と同じ「無音フォールバックを避ける」方針）。
fn resolve_scale_config(
    args: &[String],
    env_scales: Option<String>,
) -> Result<ScaleConfig, String> {
    let mut cli_value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--scales" {
            let value = args
                .get(i + 1)
                .ok_or_else(|| "--scales requires a value".to_string())?;
            cli_value = Some(value.clone());
            i += 2;
        } else if let Some(value) = arg.strip_prefix("--scales=") {
            cli_value = Some(value.to_string());
            i += 1;
        } else {
            return Err(format!("unknown argument: '{arg}'"));
        }
    }
    match cli_value.or(env_scales) {
        Some(raw) => Ok(ScaleConfig::Sweep(parse_scales(&raw)?)),
        None => Ok(ScaleConfig::Default),
    }
}

fn print_usage() {
    println!(
        "使い方: wmma_tolerance_probe [--scales <s1,s2,...>] [--help]\n\
         \n\
         --scales <カンマ区切りの正の有限値>\n\
         \u{20}\u{20}各要素をこの倍率で入力データにスケールしてスイープ計測する\n\
         \u{20}\u{20}（例: --scales 0.1,1,10,100）。未指定時は環境変数\n\
         \u{20}\u{20}WMMA_TOLERANCE_PROBE_SCALES を確認し、それも未指定なら\n\
         \u{20}\u{20}s=1 単一の既定モード（#993 以前と同一形式の出力）になる。\n\
         --help, -h\n\
         \u{20}\u{20}この使い方を表示して終了する（exit 0）。"
    );
}

/// `observed` が非有限（NaN・Inf。s が大きいスケールで f16 出力が
/// オーバーフローした場合等）のときは誤読を招く `0.00x` を出さず
/// `\"n/a\"` を返す（#993。レガシーモードでは `max_abs_diff` が非有限に
/// なる実測ケースがなく、この分岐追加は出力互換〈A2〉に影響しない）。
fn margin(threshold: f64, observed: f64) -> String {
    if !observed.is_finite() {
        "n/a".to_string()
    } else if observed <= 0.0 {
        "inf".to_string()
    } else {
        format!("{:.2}x", threshold / observed)
    }
}

/// `report` の全項目を Markdown 表 1 行として出力する。
///
/// `CompareReport` の全フィールド（`p50`/`p99`/`p999_abs_diff`・
/// `max_fail_abs_diff` を含む）を出力に含める（#186・PR #257 Codex Review
/// 指摘「出力が p50_abs_diff / p99_abs_diff / p999_abs_diff を破棄しており、
/// 再実行で CompareReport 全フィールドが再現できるという説明と矛盾する」
/// 対応。ファイル冒頭の doc コメントが謳う「`CompareReport`（誤差分布統計）を
/// 収集」という説明を出力自体で満たす）。
fn report_row(context: &str, seed: u64, report: &CompareReport) {
    println!(
        "| {context} | {seed} | {}/{} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} | {} |",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.mean_abs_diff,
        report.max_rel_err,
        report.mean_rel_err,
        report.p50_abs_diff,
        report.p99_abs_diff,
        report.p999_abs_diff,
        report.max_fail_abs_diff,
        margin(ABSOLUTE_RESCUE_THRESHOLD, report.max_abs_diff),
        margin(RELATIVE_TOLERANCE, report.max_rel_err),
    );
}

fn table_header() {
    println!(
        "| shape | seed | fail/total | max_abs_diff | mean_abs_diff | max_rel_err | mean_rel_err | p50_abs_diff | p99_abs_diff | p999_abs_diff | max_fail_abs_diff | abs margin (1e-5/max) | rel margin (1e-3/max) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
}

/// エラー行を Markdown 表フォーマット（列数を `report_row` と揃える）で
/// 出力する。空欄列数は `table_header` の列数（13 列）に合わせている。
fn error_row(label: &str, seed: u64, reason: &str) {
    println!("| {label} | {seed} | ({reason}) | - | - | - | - | - | - | - | - | - | - |");
}

/// スイープモード用の表ヘッダ（16 列。#993）。レガシー 13 列に加え、
/// 先頭の `scale` 列と末尾の `kernel`・`ref_nonfinite` 列を持つ。
fn table_header_sweep() {
    println!(
        "| scale | shape | seed | fail/total | max_abs_diff | mean_abs_diff | max_rel_err | mean_rel_err | p50_abs_diff | p99_abs_diff | p999_abs_diff | max_fail_abs_diff | abs margin (1e-5/max) | rel margin (1e-3/max) | kernel | ref_nonfinite |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
}

/// スイープモード用の計測行（#993）。`kernel` は実際に実行されたカーネル
/// 種別（[`tf32_kernel_kind`] 参照。f16 は呼び出し元が `"wmma_f16"` 固定
/// で渡す）、`ref_nonfinite` は CPU 参照出力の非有限要素数
/// （[`count_nonfinite`]）。
fn report_row_sweep(
    scale: f64,
    context: &str,
    seed: u64,
    report: &CompareReport,
    kernel: &str,
    ref_nonfinite: usize,
) {
    println!(
        "| {scale} | {context} | {seed} | {}/{} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} | {} | {kernel} | {ref_nonfinite} |",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.mean_abs_diff,
        report.max_rel_err,
        report.mean_rel_err,
        report.p50_abs_diff,
        report.p99_abs_diff,
        report.p999_abs_diff,
        report.max_fail_abs_diff,
        margin(ABSOLUTE_RESCUE_THRESHOLD, report.max_abs_diff),
        margin(RELATIVE_TOLERANCE, report.max_rel_err),
    );
}

/// スイープモード用のエラー行（列数を [`report_row_sweep`] と揃える。16 列）。
fn error_row_sweep(scale: f64, label: &str, seed: u64, reason: &str) {
    println!(
        "| {scale} | {label} | {seed} | ({reason}) | - | - | - | - | - | - | - | - | - | - | - | - |"
    );
}

/// `v` の非有限（NaN・Inf）要素数を数える。CPU 参照出力に対して呼び、
/// f16 経路で `s` が大きいスケールが入力範囲（f16 最大値 65504）を
/// 超える場合の可視化に使う（#993）。
fn count_nonfinite(v: &[f32]) -> usize {
    v.iter().filter(|x| !x.is_finite()).count()
}

/// `Xorshift64Star::fill_vec` の各要素に `scale` を乗じた f32 入力を返す
/// （#993）。`scale = 1.0` は IEEE 754 上 `x * 1.0` が bit 単位の恒等
/// 演算（±0・subnormal を含む）であるため、レガシーモード（`s = 1.0` の
/// みで呼ばれる）の出力は `rng.fill_vec(len)` を直接使った場合と完全に
/// 一致する（A2 を満たす設計根拠）。
fn scaled_f32_inputs(rng: &mut Xorshift64Star, len: usize, scale: f64) -> Vec<f32> {
    let s = scale as f32;
    rng.fill_vec(len).into_iter().map(|x| x * s).collect()
}

/// `next_f32()` を `scale` 倍してから f16 へ丸めた入力を返す（スケール
/// 後丸め。#993）。GPU 入力（本関数の戻り値）と CPU 参照
/// （[`probe_f16`] が同じ値を f32 へ拡張して使う）が同じ f16 値を参照する
/// ため parity 比較の意味は変わらず、各スケールで入力が f16 表現格子上に
/// 乗る。`scale = 1.0` は `next_f32() * 1.0` の丸めであり
/// `Xorshift64Star::fill_vec_f16` と bit 単位で一致する（A2）。
fn scaled_f16_inputs(rng: &mut Xorshift64Star, len: usize, scale: f64) -> Vec<f16> {
    let s = scale as f32;
    (0..len)
        .map(|_| f16::from_f32(rng.next_f32() * s))
        .collect()
}

/// 指定した形状 `(m, n, k)` で `CudaGemm::run_wmma_tf32` が実際に選ぶ
/// カーネル種別を返す（#993。R5: #994 の前提となる「実行カーネル種別の
/// 明示」）。
///
/// `gemm.rs::CudaGemm::run_wmma_tf32` の 3 段選択ロジック（staged
/// （`n%4==0 && k%4==0` 整列形状限定。さらにサイズ条件付き swizzle 変種）
/// → opt → basic）と**同一真実源の公開アクセサ**
/// （[`CudaGemm::wmma_tf32_routed_path_is_staged`]・
/// [`CudaGemm::wmma_tf32_staged_swizzle_applies`]・
/// [`CudaGemm::wmma_tf32_opt_available`]）から導出する（独自の推定
/// ロジックを持たない。`security.md` A08「取り込み判断の根拠を追跡可能に
/// する」に相当する設計判断: ここで判定を再実装すると `run_wmma_tf32`
/// 本体の変更に追従できず誤ったカーネル種別を報告しうる）。
fn tf32_kernel_kind(gemm: &CudaGemm, m: u32, n: u32, k: u32) -> &'static str {
    if gemm.wmma_tf32_routed_path_is_staged(n, k) {
        if gemm.wmma_tf32_staged_swizzle_applies(m, n, k) {
            "staged-swizzle"
        } else {
            "staged"
        }
    } else if gemm.wmma_tf32_opt_available() {
        "opt"
    } else {
        "basic"
    }
}

/// TF32 経路のカーネル可用性ヘッダをスイープモードの表直前に出力する
/// （R5。#993）。`gemm` の公開アクセサ（[`CudaGemm::wmma_tf32_staged_available`]・
/// [`CudaGemm::wmma_tf32_staged_unavailable_reason`]・
/// [`CudaGemm::wmma_tf32_staged_swizzle_group_width`]・
/// [`CudaGemm::wmma_tf32_opt_available`]・
/// [`CudaGemm::wmma_tf32_opt_unavailable_reason`]）をそのまま読むだけで、
/// `CudaGemm::new` 時点の状態を独自判定なしに可視化する。
fn tf32_kernel_availability_header(gemm: &CudaGemm) {
    let staged = if gemm.wmma_tf32_staged_available() {
        "yes".to_string()
    } else {
        format!(
            "no ({})",
            gemm.wmma_tf32_staged_unavailable_reason()
                .unwrap_or("unknown reason")
        )
    };
    let swizzle_width = gemm
        .wmma_tf32_staged_swizzle_group_width()
        .map(|w| w.to_string())
        .unwrap_or_else(|| "none".to_string());
    let opt = if gemm.wmma_tf32_opt_available() {
        "yes".to_string()
    } else {
        format!(
            "no ({})",
            gemm.wmma_tf32_opt_unavailable_reason()
                .unwrap_or("unknown reason")
        )
    };
    println!(
        "kernel availability: staged={staged}, staged-swizzle group width={swizzle_width}, opt={opt}, basic=yes"
    );
    println!(
        "routing rule: staged if n%4==0 && k%4==0 (swizzle variant if size condition) -> opt -> basic\n"
    );
}

/// f16 経路のカーネル注記をスイープモードの表直前に出力する（R5。#993）。
/// f16 は `CudaWmmaGemm::run_f16` 一択（TF32 のような多段選択を持たない）
/// のためカーネル種別は固定だが、f16 最大値（65504）を超えるスケールで
/// 出力が ±Inf になりうる旨を明示し `ref_nonfinite` 列の読み方を示す。
fn f16_kernel_note() {
    println!("kernel: CudaWmmaGemm::run_f16 (WMMA f16)");
    println!(
        "f16 max = 65504; outputs beyond this overflow to \u{00b1}Inf in both GPU and \
         reference; such cells are counted in ref_nonfinite and compare reports them as fail \
         (max_fail_abs_diff = inf)\n"
    );
}

/// TF32 経路（[`CudaGemm::run_wmma_tf32`]）の誤差分布を形状×シード×
/// スケールごとに計測する。CPU 参照は `matmul_reference_fma`（FMA 契約の
/// 唯一の参照点。`tests/gemm_wmma_tf32.rs::assert_wmma_tf32_parity` と
/// 同じ比較方法）。
///
/// 戻り値 `true` は「意図しない計測エラー（shape 検証失敗・`compare` の
/// 長さ不一致・WMMA 起動時エラー）が 1 件以上発生した」ことを示す。
///
/// `CudaError::WmmaUnavailable` の扱いは `device` の compute capability で
/// 分岐する（PR #257 Codex Review 再指摘対応。`gemm.rs::CudaGemm::new` の
/// ドキュメンテーションコメントが明記するとおり、TF32 WMMA 経路は
/// `TensorCoreUnsupported` のような事前ゲートを持たず、NVRTC コンパイル結果
/// のみで可否を判定する事後判定方式である。そのため `WmmaUnavailable` は
/// 「cc<8.0 でカーネルが拒否された（意図的スキップ）」と「cc≥8.0 なのに
/// `<mma.h>` 解決失敗等でコンパイル・ロードが実際に失敗した（意図しない
/// 計測エラー）」の両方を表しうる。`compute_capability() < (8, 0)` の場合
/// のみ意図的スキップとして `false` のまま `return` し、それ以外
/// （cc≥8.0 での失敗）は想定外エラーとして扱い `had_unexpected_error` を
/// 立てたうえで計測を打ち切る）。
///
/// `scales` が [`ScaleConfig::Default`]（s=1 単一）のときは既存の
/// `table_header`／`report_row`／`error_row`（現行 13 列）を無変更で使い、
/// [`ScaleConfig::Sweep`] のときはカーネル可用性ヘッダ
/// （[`tf32_kernel_availability_header`]）を出力したうえで
/// `table_header_sweep`／`report_row_sweep`／`error_row_sweep`（16 列）を
/// 使う（#993）。
fn probe_tf32(device: &CudaDevice, gemm: &CudaGemm, scales: &ScaleConfig) -> bool {
    println!("\n## TF32 (`CudaGemm::run_wmma_tf32`)\n");
    if scales.is_sweep() {
        tf32_kernel_availability_header(gemm);
        table_header_sweep();
    } else {
        table_header();
    }
    let mut had_unexpected_error = false;
    for &(label, m, n, k) in SHAPES {
        for &seed in SEEDS {
            for &scale in scales.scales() {
                let mut rng = Xorshift64Star::new(seed.wrapping_mul(1000).wrapping_add(m as u64));
                let a = scaled_f32_inputs(&mut rng, (m as usize) * (k as usize), scale);
                let b = scaled_f32_inputs(&mut rng, (k as usize) * (n as usize), scale);

                let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
                if matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
                    .is_err()
                {
                    // SHAPES は固定の well-formed 形状のみのため、ここに到達する
                    // のは想定外（shape 定義のバグ）である。
                    if scales.is_sweep() {
                        error_row_sweep(scale, label, seed, "unexpected: shape validation error");
                    } else {
                        error_row(label, seed, "unexpected: shape validation error");
                    }
                    had_unexpected_error = true;
                    continue;
                }

                match gemm.run_wmma_tf32(&a, &b, m, n, k) {
                    Ok(c_gpu) => match compare(&c_gpu, &c_ref) {
                        Ok(report) => {
                            if scales.is_sweep() {
                                let kernel = tf32_kernel_kind(gemm, m, n, k);
                                let ref_nonfinite = count_nonfinite(&c_ref);
                                report_row_sweep(
                                    scale,
                                    label,
                                    seed,
                                    &report,
                                    kernel,
                                    ref_nonfinite,
                                );
                            } else {
                                report_row(label, seed, &report);
                            }
                        }
                        Err(err) => {
                            let reason = format!("unexpected: compare error: {err}");
                            if scales.is_sweep() {
                                error_row_sweep(scale, label, seed, &reason);
                            } else {
                                error_row(label, seed, &reason);
                            }
                            had_unexpected_error = true;
                        }
                    },
                    Err(CudaError::WmmaUnavailable { detail }) => {
                        let (major, minor) = device.compute_capability();
                        if (major, minor) < (8, 0) {
                            println!(
                                "\n(TF32 WMMA unavailable: compute capability {major}.{minor} \
                                 < 8.0（意図的スキップ）: {detail})\n"
                            );
                            return had_unexpected_error;
                        }
                        // cc≥8.0 なのにコンパイル・ロードが失敗している。
                        // Tensor Core 非対応環境ではなく実際のコンパイル・ロード
                        // 失敗（`<mma.h>` 解決失敗等）であり、部分計測を完了と
                        // 誤認させないため想定外エラーとして扱う（Codex Review
                        // 再指摘対応）。
                        println!(
                            "\n(TF32 WMMA unavailable despite compute capability {major}.{minor} \
                             >= 8.0 — unexpected compile/load failure: {detail})\n"
                        );
                        return true;
                    }
                    Err(other) => {
                        let reason = format!("unexpected: run error: {other}");
                        if scales.is_sweep() {
                            error_row_sweep(scale, label, seed, &reason);
                        } else {
                            error_row(label, seed, &reason);
                        }
                        had_unexpected_error = true;
                    }
                }
            }
        }
    }
    had_unexpected_error
}

/// f16 WMMA 経路（[`CudaWmmaGemm::run_f16`]）の誤差分布を形状×シード×
/// スケールごとに計測する。参照方法は `tests/cpu_cuda_wmma_parity.rs`
/// 冒頭コメントの 3 手順（f16→f32→参照 matmul→f16 丸め→f32）をそのまま
/// 踏襲する。
///
/// 戻り値の意味は [`probe_tf32`] と同じ（意図しないエラー発生の有無）。
/// レガシー／スイープの出力切り替えも [`probe_tf32`] と同方針（#993）。
fn probe_f16(gemm: &CudaWmmaGemm, scales: &ScaleConfig) -> bool {
    println!("\n## f16 WMMA (`CudaWmmaGemm::run_f16`)\n");
    if scales.is_sweep() {
        f16_kernel_note();
        table_header_sweep();
    } else {
        table_header();
    }
    let mut had_unexpected_error = false;
    for &(label, m, n, k) in SHAPES {
        for &seed in SEEDS {
            for &scale in scales.scales() {
                let mut rng = Xorshift64Star::new(seed.wrapping_mul(2000).wrapping_add(m as u64));
                let a_f16: Vec<f16> =
                    scaled_f16_inputs(&mut rng, (m as usize) * (k as usize), scale);
                let b_f16: Vec<f16> =
                    scaled_f16_inputs(&mut rng, (k as usize) * (n as usize), scale);
                let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
                let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();

                let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
                if matmul_reference_fma(
                    &a_f32,
                    &b_f32,
                    &mut c_ref_f32,
                    m as usize,
                    n as usize,
                    k as usize,
                )
                .is_err()
                {
                    if scales.is_sweep() {
                        error_row_sweep(scale, label, seed, "unexpected: shape validation error");
                    } else {
                        error_row(label, seed, "unexpected: shape validation error");
                    }
                    had_unexpected_error = true;
                    continue;
                }
                let c_ref_rounded: Vec<f32> = c_ref_f32
                    .iter()
                    .map(|&x| f16::from_f32(x).to_f32())
                    .collect();

                match gemm.run_f16(&a_f16, &b_f16, m, n, k) {
                    Ok(c_gpu_f16) => {
                        let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();
                        match compare(&c_gpu_f32, &c_ref_rounded) {
                            Ok(report) => {
                                if scales.is_sweep() {
                                    let ref_nonfinite = count_nonfinite(&c_ref_rounded);
                                    report_row_sweep(
                                        scale,
                                        label,
                                        seed,
                                        &report,
                                        "wmma_f16",
                                        ref_nonfinite,
                                    );
                                } else {
                                    report_row(label, seed, &report);
                                }
                            }
                            Err(err) => {
                                let reason = format!("unexpected: compare error: {err}");
                                if scales.is_sweep() {
                                    error_row_sweep(scale, label, seed, &reason);
                                } else {
                                    error_row(label, seed, &reason);
                                }
                                had_unexpected_error = true;
                            }
                        }
                    }
                    Err(other) => {
                        let reason = format!("unexpected: run error: {other}");
                        if scales.is_sweep() {
                            error_row_sweep(scale, label, seed, &reason);
                        } else {
                            error_row(label, seed, &reason);
                        }
                        had_unexpected_error = true;
                    }
                }
            }
        }
    }
    had_unexpected_error
}

/// `main` の終了コード。CUDA driver／NVRTC 非搭載・Tensor Core 非対応
/// 環境での意図的スキップは exit 0 のまま維持し、それ以外の想定外エラー
/// （`CudaDevice::new`／`CudaGemm::new`／`CudaWmmaGemm::new` の想定外エラー、
/// または [`probe_tf32`]・[`probe_f16`] 内の想定外エラー）は exit 1 を返す
/// （#186・PR #257 Codex Review 指摘「stress shape で
/// allocation/launch/execution エラーが起きても行を出力するだけでプログラム
/// は exit 0 のままになり、部分計測を完了と誤認しうる」対応）。`--scales`
/// の解決（[`resolve_scale_config`]）に失敗した場合も使い方を表示して
/// exit 1 を返す（#993。CUDA 初期化より前に判定するため driver 非搭載
/// 環境でも入力検証エラーを検出できる）。
fn main() -> std::process::ExitCode {
    use std::process::ExitCode;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return ExitCode::SUCCESS;
    }
    let env_scales = std::env::var("WMMA_TOLERANCE_PROBE_SCALES").ok();
    let scale_config = match resolve_scale_config(&args, env_scales) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("error: {msg}\n");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    println!("# WMMA Tensor Core 経路 誤差分布実測（TASK-11.1g・#186・#993）\n");
    println!(
        "閾値（REQ-2 統一複合判定・変更対象外）: RELATIVE_TOLERANCE={RELATIVE_TOLERANCE:e}, \
         ABSOLUTE_RESCUE_THRESHOLD={ABSOLUTE_RESCUE_THRESHOLD:e}\n"
    );
    if let ScaleConfig::Sweep(scales) = &scale_config {
        println!("scales: {scales:?}\n");
    }

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!("CUDA driver 非搭載環境のため計測をスキップします: {detail}");
            return ExitCode::SUCCESS;
        }
        Err(other) => {
            // `CudaError::Driver`（`libcuda` は存在するが `cuInit`／
            // コンテキスト生成／デバイスメタデータ取得等の driver API
            // 呼び出し自体が失敗したケース。`error.rs` のドキュメンテーション
            // コメント参照）は「環境非対応」ではなく実際の driver API 失敗
            // であり、`DriverUnavailable` と同列にスキップ扱いすると計測ゼロ
            // のまま exit 0 を返し完了と区別できなくなる（Codex Review
            // 再指摘対応）。`DriverUnavailable` 以外は全て想定外エラーとして
            // exit 1 を返す。
            println!("CudaDevice::new が想定外のエラーを返しました: {other}");
            return ExitCode::FAILURE;
        }
    };
    println!("device compute capability: {}", device.arch());

    let mut had_unexpected_error = false;

    match CudaGemm::new(&device) {
        Ok(gemm) => {
            if probe_tf32(&device, &gemm, &scale_config) {
                had_unexpected_error = true;
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("\nNVRTC 非搭載環境のため TF32 経路の計測をスキップします: {detail}");
        }
        Err(other) => {
            println!("\nCudaGemm::new が想定外のエラーを返しました: {other}");
            had_unexpected_error = true;
        }
    }

    match CudaWmmaGemm::new(&device) {
        Ok(gemm) => {
            if probe_f16(&gemm, &scale_config) {
                had_unexpected_error = true;
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("\nNVRTC 非搭載環境のため f16 WMMA 経路の計測をスキップします: {detail}");
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            println!(
                "\ncompute capability 7.0 未満のため f16 WMMA 経路の計測をスキップします: {detail}"
            );
        }
        Err(other) => {
            println!("\nCudaWmmaGemm::new が想定外のエラーを返しました: {other}");
            had_unexpected_error = true;
        }
    }

    if had_unexpected_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scales_accepts_valid_csv() {
        assert_eq!(
            parse_scales("0.1,1,10,100").unwrap(),
            vec![0.1, 1.0, 10.0, 100.0]
        );
    }

    #[test]
    fn parse_scales_trims_whitespace() {
        assert_eq!(parse_scales(" 0.1 , 1 ").unwrap(), vec![0.1, 1.0]);
    }

    #[test]
    fn parse_scales_rejects_empty_element() {
        assert!(parse_scales("1,,2").is_err());
    }

    #[test]
    fn parse_scales_rejects_non_numeric() {
        assert!(parse_scales("1,abc").is_err());
    }

    #[test]
    fn parse_scales_rejects_zero_and_negative() {
        assert!(parse_scales("0").is_err());
        assert!(parse_scales("-1").is_err());
    }

    #[test]
    fn parse_scales_rejects_non_finite() {
        assert!(parse_scales("inf").is_err());
        assert!(parse_scales("nan").is_err());
    }

    #[test]
    fn parse_scales_rejects_duplicates() {
        assert!(parse_scales("1,2,1").is_err());
    }

    #[test]
    fn parse_scales_rejects_too_many() {
        let many = (1..=(MAX_SCALES + 1))
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert!(parse_scales(&many).is_err());
    }

    #[test]
    fn parse_scales_rejects_empty_string() {
        assert!(parse_scales("").is_err());
    }

    #[test]
    fn resolve_scale_config_defaults_when_unspecified() {
        assert_eq!(
            resolve_scale_config(&[], None).unwrap(),
            ScaleConfig::Default
        );
    }

    #[test]
    fn resolve_scale_config_prefers_cli_over_env() {
        let args = vec!["--scales".to_string(), "1,2".to_string()];
        let cfg = resolve_scale_config(&args, Some("9,9.5".to_string())).unwrap();
        assert_eq!(cfg, ScaleConfig::Sweep(vec![1.0, 2.0]));
    }

    #[test]
    fn resolve_scale_config_falls_back_to_env() {
        let cfg = resolve_scale_config(&[], Some("0.1,1".to_string())).unwrap();
        assert_eq!(cfg, ScaleConfig::Sweep(vec![0.1, 1.0]));
    }

    #[test]
    fn resolve_scale_config_supports_equals_form() {
        let args = vec!["--scales=1,10".to_string()];
        let cfg = resolve_scale_config(&args, None).unwrap();
        assert_eq!(cfg, ScaleConfig::Sweep(vec![1.0, 10.0]));
    }

    #[test]
    fn resolve_scale_config_rejects_unknown_argument() {
        let args = vec!["--bogus".to_string()];
        assert!(resolve_scale_config(&args, None).is_err());
    }

    #[test]
    fn resolve_scale_config_rejects_missing_value() {
        let args = vec!["--scales".to_string()];
        assert!(resolve_scale_config(&args, None).is_err());
    }

    #[test]
    fn scale_config_explicit_one_is_sweep_mode() {
        // R2: 「未指定」のみがレガシー扱いで、明示指定は内容が 1 単独でも
        // sweep 出力モードへ切り替わる。
        let cfg = ScaleConfig::Sweep(vec![1.0]);
        assert!(cfg.is_sweep());
        assert_eq!(cfg.scales(), &[1.0]);
    }

    #[test]
    fn scaled_f32_inputs_scale_one_matches_unscaled_bit_for_bit() {
        // s=1.0 は `x * 1.0` の bit 恒等演算のため、`fill_vec` 直接呼び出しと
        // 完全一致する必要がある（A2 の根拠）。
        let mut a = Xorshift64Star::new(42);
        let mut b = Xorshift64Star::new(42);
        let scaled = scaled_f32_inputs(&mut a, 64, 1.0);
        let plain = b.fill_vec(64);
        assert_eq!(scaled, plain);
    }

    #[test]
    fn scaled_f32_inputs_applies_scale() {
        let mut rng = Xorshift64Star::new(7);
        let scaled = scaled_f32_inputs(&mut rng, 16, 10.0);
        for v in &scaled {
            assert!((-10.0..10.0).contains(v), "out of expected range: {v}");
        }
    }

    #[test]
    fn scaled_f16_inputs_scale_one_matches_unscaled_bit_for_bit() {
        let mut a = Xorshift64Star::new(99);
        let mut b = Xorshift64Star::new(99);
        let scaled = scaled_f16_inputs(&mut a, 32, 1.0);
        let plain = b.fill_vec_f16(32);
        assert_eq!(scaled, plain);
    }

    #[test]
    fn scaled_f16_inputs_scale_after_round_order() {
        // 「next_f32() * scale をそれから丸める」順序であることを確認する
        // （設計注記どおり: 先に丸めてからスケールするのではない）。
        let mut a = Xorshift64Star::new(3);
        let mut b = Xorshift64Star::new(3);
        let scaled = scaled_f16_inputs(&mut a, 8, 100.0);
        let expected: Vec<f16> = (0..8)
            .map(|_| f16::from_f32(b.next_f32() * 100.0))
            .collect();
        assert_eq!(scaled, expected);
    }

    #[test]
    fn count_nonfinite_counts_nan_and_inf_only() {
        let v = [1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        assert_eq!(count_nonfinite(&v), 3);
    }

    #[test]
    fn count_nonfinite_zero_when_all_finite() {
        let v = [1.0f32, -2.5, 0.0, 100.0];
        assert_eq!(count_nonfinite(&v), 0);
    }

    #[test]
    fn margin_reports_n_a_for_non_finite_observed() {
        assert_eq!(margin(1e-3, f64::NAN), "n/a");
        assert_eq!(margin(1e-3, f64::INFINITY), "n/a");
    }

    #[test]
    fn margin_reports_inf_for_zero_observed() {
        assert_eq!(margin(1e-3, 0.0), "inf");
    }

    #[test]
    fn margin_computes_ratio_for_finite_positive_observed() {
        assert_eq!(margin(1e-3, 1e-4), "10.00x");
    }
}
