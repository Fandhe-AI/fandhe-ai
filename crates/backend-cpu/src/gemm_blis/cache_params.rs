//! MC/KC/NC の実行時キャッシュ検出（イシュー #753・§3.1）。
//!
//! [`super::default_blocks`] は Apple M4 Max 実測（#749・#481 §3）に基づく
//! コンパイル時定数だが、実行環境が変われば L1D／L2 実容量も変わるため、
//! 固定値は実機ごとの最適点から外れうる。本モジュールは `sysctl`
//! （macOS 限定）で実測した L1D／L2 サイズから BLIS 解析モデル系の技法
//! （L1 に A/B マイクロパネルが追い出し合わずに共存する条件で KC を、
//! L2 に A パネルが収まる条件で MC を、というキャッシュ階層ブロッキングの
//! 一般的な導出方針。gemm crate `cache.rs` の技法を**参照**するのみで
//! コードは転記しない）で MC/KC/NC を算出する。
//!
//! ## デッドコード化を避ける設計（PR #766 の教訓）
//!
//! PR #766 では「常に不活性な sysctl FFI」（機種識別子ベースの判定分岐が
//! 実測未確定のため常に旧経路へ落ちる構成）が codex-review の P0/P1 指摘で
//! 撤去された。本モジュールは同じ轍を踏まないよう、以下の 3 点で構成する:
//!
//! 1. **純関数化**（[`compute_blocks`]）: 機種判定ではなくキャッシュ
//!    サイズからの算出式のため、macOS 以外を含む全プラットフォームで
//!    単体テスト可能（`mod tests` 参照。CI（GitHub ホステッド
//!    `ubuntu-latest`）上でも実行され続ける）。
//! 2. **`cargo test` 経由でのコンパイル検証**: 本モジュール自体
//!    （[`super`] の `mod cache_params;` 宣言）は本番未結線の間
//!    `#[cfg(test)]` のため、CI の通常ビルドジョブ
//!    （`cargo build (linux / aarch64-apple-darwin)`。`.claude/rules/ci.md`）
//!    には**含まれない**（`cfg(test)` 無効時は構造的にコンパイル対象外）。
//!    純関数（[`compute_blocks`] 等・`cfg(target_os = "macos")` 非依存）の
//!    型・借用検査は `rust-ci` の test ジョブ（`cargo test`。`cfg(test)` 有効）
//!    が Linux ホスト上で担うが、同ジョブは `target_os != "macos"` のため
//!    sysctl FFI 部（[`sysctl_ffi`]）自体はコンパイル対象に含まれない
//!    （**レビュー指摘・#753**: 通常の macOS クロスビルドジョブも
//!    `cfg(test)` 無効のため到達せず、結果として `sysctl_ffi` を継続的に
//!    型・借用検査するジョブが存在しない状態だった）。この空白を埋めるため
//!    `ci.yml` build ジョブに `cargo check -p backend-cpu --tests --target
//!    aarch64-apple-darwin`（`cfg(test)` 有効かつ `target_os = "macos"`
//!    クロスターゲット。backend-metal 向け同型ステップと同じ手法。
//!    `Makefile` の `check-cross-cpu-tests` と同一コマンド）を追加し、
//!    `sysctl_ffi` を継続的コンパイル検証の対象に含めている（詳細は
//!    `docs/perf/cpu-gemm-runtime-cache-detect.md` §3）。
//! 3. **常に到達可能な公開入口**（[`detected_blocks`]）: `#[cfg(test)]`
//!    ではなく通常ビルドから到達可能な `pub(crate)` 関数とし、
//!    非 macOS・sysctl 失敗時は [`super::default_blocks`] へ
//!    フォールバックする（本モジュール `mod tests` の
//!    `detected_blocks_returns_valid_block_sizes_on_any_platform` が
//!    Linux CI 上でこのフォールバック経路を実行し続ける）。
//!
//! ## 本番未結線
//!
//! [`detected_blocks`] は本番 3 公開関数（[`super::gemm_blis`]／
//! [`super::gemm_blis_parallel`]／[`super::gemm_blis_bias_act_parallel`]）
//! からは呼ばれない（受け入れ条件 2＝実機 5 回中央値での非劣化確認が
//! 本 PR のスコープ外のため。#750・#758 と同型の判断。
//! `docs/perf/cpu-gemm-runtime-cache-detect.md` 参照）。テスト専用の
//! パラメータ化入口（[`super::gemm_blis_parallel_with_blocks`] 等）・
//! 実機 A/B ハーネス（`mod tests` の `#[ignore]` テスト）から到達する。

use std::sync::OnceLock;

use crate::gemm::BlockSizes;

/// f32 要素 1 つのバイト数（GEMM の A/B/C は全て f32。`.claude/rules/coding-rust.md`
/// は本クレートの数値型を f32 に統一しており、本モジュールの容量計算も
/// これに合わせる）。
const F32_BYTES: usize = 4;

/// KC のクランプ下限・上限。下限は極小 L1D（組込み・仮想化環境等）でも
/// マイクロカーネル 1 反復あたりの演算密度を落としすぎないための保守的な
/// 最小値、上限は #749 実測（`docs/perf/cpu-gemm-blocking-sweep.md` §7）の
/// 候補グリッドで検証済みの最大値（KC=4096・firestorm 参照値）に合わせる。
const KC_MIN: usize = 128;
const KC_MAX: usize = 4096;
/// MC のクランプ下限・上限。上限は同じく #749 実測グリッドの firestorm
/// 参照値（MC=480）を上回る余裕を持たせつつ、L2 実効容量の非現実的な
/// 過大評価を防ぐ。
const MC_MIN: usize = 64;
const MC_MAX: usize = 1024;
/// NC のクランプ下限・上限。上限は #749 実測（NC=9600 は n>=4096 でのみ
/// 改善・n=2048 では劣化）を踏まえ、firestorm 参照値（NC=9600）を包含
/// しつつ無制限の拡大を防ぐ値とする。
const NC_MIN: usize = 256;
const NC_MAX: usize = 16384;

/// L1D サイズの正当性検査範囲（バイト）。sysctl の戻り値は外部入力として
/// 扱い、明らかに非現実的な値（0・極小・極大）は算出前に拒否する
/// （OWASP A03・`.claude/rules/security.md`）。下限 4KiB は実在する最小級
/// L1D の下限を大きく下回らない値、上限 8MiB は実在する L1D 実装を
/// 大きく超える安全側の値。
const L1D_SANE_MIN: usize = 4 * 1024;
const L1D_SANE_MAX: usize = 8 * 1024 * 1024;
/// L2 サイズの正当性検査範囲（バイト）。理由は L1D と同じ。
const L2_SANE_MIN: usize = 128 * 1024;
const L2_SANE_MAX: usize = 256 * 1024 * 1024;

/// `x` を `multiple` の倍数へ切り上げる（`multiple == 0` は `x` を素通し）。
///
/// レビュー指摘（#753）: `mr`／`nr`（[`compute_blocks`] 冒頭で 0 のみ検査
/// 済みで上限は検査していない）を `multiple` に渡すと `div_ceil(...) *
/// multiple` が大きな値で `usize` オーバーフローしうる（debug では
/// panic、release ではラップして誤ったブロックサイズを返す）。`checked_mul`
/// で防御し、オーバーフロー時は `None` を返す（呼び出し元は
/// [`compute_blocks`] を経由し最終的に [`super::default_blocks`] への
/// フォールバックへ落ちる。受け入れ条件 3 と同じ fail-closed 方針）。
fn round_up_to_multiple(x: usize, multiple: usize) -> Option<usize> {
    if multiple == 0 {
        return Some(x);
    }
    x.div_ceil(multiple).checked_mul(multiple)
}

/// `raw` を `[min, max]` へクランプしたうえで `multiple` の倍数へ丸める
/// （レビュー指摘 #753: 従来は「切り上げしてからクランプ」の順だったため、
/// クランプが効いた場合に結果が `multiple` の倍数であるという契約が
/// 破れていた。`MC_MIN=64`／`NC_MIN=256` は NEON `NR=12` 等では倍数に
/// ならない値のため、先にクランプしてから倍数へ丸める必要がある）。
///
/// 切り上げた結果が `max` を超える場合（`max` 自体が `multiple` の倍数で
/// ない場合に起こりうる）は、`max` 以下で `multiple` の倍数になる最大値
/// へ切り下げる。これにより返り値は常に `multiple` の倍数であり、かつ
/// `max` を超えない（`min` をわずかに下回りうるが、`multiple` が `min`
/// 以下である通常のマイクロカーネル構成では起こらない）。
///
/// [`round_up_to_multiple`] のオーバーフロー検出（`multiple` に極端に
/// 大きな `mr`／`nr` が渡された場合）を `None` として伝播する。
fn clamp_to_multiple(raw: usize, min: usize, max: usize, multiple: usize) -> Option<usize> {
    let clamped = raw.clamp(min, max);
    let rounded = round_up_to_multiple(clamped, multiple)?;
    Some(if rounded <= max {
        rounded
    } else {
        (max / multiple) * multiple
    })
}

/// L1D／L2 実測値（バイト）と、対象マイクロカーネルの `MR`／`NR`
/// （[`super::microkernel::Microkernel`] の型定数）から MC/KC/NC を算出する
/// 純関数（全プラットフォームで単体テスト可能。#753 §3.1）。
///
/// - **KC**: A マイクロパネル（`MR × KC` 要素）と B マイクロパネル
///   （`KC × NR` 要素）が L1D に共存し追い出し合わない条件から算出する。
///   L1D の連想度は `sysctl` から取得できないため、保守的に「L1D 実容量の
///   半分」を予算とする（残り半分は C アキュムレータタイル・他の常駐
///   データ・連想度に由来する実効容量低下の余裕分。gemm crate `cache.rs`
///   の「L1 連想度と A パネルの追い出し関係を整合させる」技法を、
///   連想度非取得という制約下での保守的な固定仮定として反映したもの）。
/// - **MC**: A パネル（`MC × KC × 4B`）が L2 実容量の一定割合に収まる
///   条件から算出する（予算は同じく半分。同時に B パネルもコアクラスタ
///   共有の L2 に常駐するため）。`MR` の倍数へ切り上げる。
/// - **NC**: L2 の残余容量から B パネル（`KC × NC × 4B`）が収まる上限を
///   算出する。`NR` の倍数へ切り上げる。#749 実測（NC=9600 は n>=4096
///   でのみ改善・n=2048 では劣化）と矛盾しないよう [`NC_MAX`] でクランプ
///   する。
///
/// 各値は [`KC_MIN`]〜[`NC_MAX`] の範囲へクランプする。`l1d_bytes`／
/// `l2_bytes` が正当性検査範囲外、`mr`／`nr` が 0 の場合、または
/// `mr`／`nr` に起因する中間計算（`mr + nr`・`F32_BYTES * (mr + nr)`・
/// [`clamp_to_multiple`] 内の丸め）が `usize` オーバーフローする場合は
/// `None`（呼び出し元は [`super::default_blocks`] へフォールバックする。
/// 受け入れ条件 3。オーバーフロー検出はレビュー指摘 #753: `mr`／`nr` は
/// 0 のみ検査しており上限を検査していなかったため、大きな値で
/// debug では panic・release では誤った丸め値が生じうる状態だった）。
pub(crate) fn compute_blocks(
    l1d_bytes: usize,
    l2_bytes: usize,
    mr: usize,
    nr: usize,
) -> Option<BlockSizes> {
    if mr == 0 || nr == 0 {
        return None;
    }
    if !(L1D_SANE_MIN..=L1D_SANE_MAX).contains(&l1d_bytes) {
        return None;
    }
    if !(L2_SANE_MIN..=L2_SANE_MAX).contains(&l2_bytes) {
        return None;
    }

    let l1_budget = l1d_bytes / 2;
    // `checked_add`／`checked_mul` で防御する（`mr`／`nr` の 0 排除だけでは
    // 上限未検証のため大きな値でオーバーフローしうる。オーバーフロー時は
    // `None` を返し呼び出し元のフォールバックへ委ねる）。
    let per_k_bytes = F32_BYTES.checked_mul(mr.checked_add(nr)?)?;
    if per_k_bytes == 0 {
        // `mr`／`nr` を 0 排除済みのため理論上到達しないが、`compute_blocks`
        // 全体の fail-closed 方針（0 除算防止）に合わせ明示的に拒否する。
        return None;
    }
    let kc = (l1_budget / per_k_bytes).clamp(KC_MIN, KC_MAX);

    let l2_budget = l2_bytes / 2;
    let kc_bytes = F32_BYTES.checked_mul(kc)?;
    if kc_bytes == 0 {
        return None;
    }
    let mc_raw = l2_budget / kc_bytes;
    let mc = clamp_to_multiple(mc_raw, MC_MIN, MC_MAX, mr)?;

    // NC は B パネルが占有できる L2 残余（A パネル分を除いた残り）から
    // 算出する。`l2_budget` は A パネルに割り当てた予算のため、B 側は
    // 残りの半分（`l2_bytes - l2_budget == l2_budget`）を使う。
    let nc_raw = l2_budget / kc_bytes;
    let nc = clamp_to_multiple(nc_raw, NC_MIN, NC_MAX, nr)?;

    Some(BlockSizes { mc, kc, nc })
}

/// `sysctlbyname`（macOS/BSD 系 libSystem が提供する標準 API）による
/// L1D／L2 実測値の読み取り（`cfg(target_os = "macos")` 限定）。
///
/// 追加クレート依存を使わない理由: `libc` は許容 9 区分
/// （`.claude/rules/deps-policy.md`）外であり、`sysctlbyname` は macOS
/// 実行環境に常にリンクされている libSystem の C ABI 関数のため、
/// `objc2` 系（`cfg(target_os = "macos")` 限定の許容依存）と同様に
/// `extern "C"` の自前宣言で足りる（#749 時点の実装〈PR #766・撤去済み〉
/// と同方式）。
#[cfg(target_os = "macos")]
mod sysctl_ffi {
    use std::ffi::{CString, c_char, c_int, c_void};

    // SAFETY: この `extern "C"` 宣言は macOS/BSD 系 libSystem が公開する
    // 標準 API `sysctlbyname`（`<sys/sysctl.h>`、`man 3 sysctlbyname`）の
    // シグネチャと一致させている: 戻り値は `c_int`（0 は成功、非 0 はエラー。
    // errno 相当）、引数は `name: *const c_char`（NUL 終端文字列）・
    // `oldp: *mut c_void`／`oldlenp: *mut usize`（読み取り先バッファと
    // その長さ、in/out）・`newp: *mut c_void`／`newlen: usize`（書き込み
    // 値、本モジュールでは常に null／0 で読み取り専用呼び出しに限定）で、
    // C ABI 上の型幅・呼び出し規約（`extern "C"`）は libSystem のヘッダ
    // 定義と 1:1 対応する。シンボルは全 macOS 実行環境に常にリンクされる
    // libSystem が提供するため動的ロード不要で解決可能（`cfg(target_os =
    // "macos")` 限定でのみコンパイルされ、他 OS では宣言自体が存在しない）。
    // 個々の呼び出し引数の安全性（ポインタ有効性・長さ整合）は呼び出し側
    // `read_usize` の SAFETY コメントを参照。
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// 指定した sysctl 名の `usize` 値を読む。取得失敗（戻り値 != 0）・
    /// 長さ不一致・値 0 は `None`（外部入力の検証。OWASP A03・
    /// `.claude/rules/security.md`）。
    pub(super) fn read_usize(name: &str) -> Option<usize> {
        // `sysctlbyname` は NUL 終端 C 文字列を要求する契約
        // （`man 3 sysctlbyname`）。`name` に埋め込み NUL が含まれる
        // 呼び出しは本モジュール内の固定文字列のみのため到達しないが、
        // `CString::new` は防御的に検査してから変換する。
        let cname = CString::new(name).ok()?;
        let mut value: usize = 0;
        let mut len = size_of::<usize>();
        // SAFETY: `cname` はこの呼び出しの生存期間中有効な NUL 終端 C
        // 文字列。`oldp` は `size_of::<usize>()` ちょうどの有効な書き込み
        // 先（ローカル変数 `value` のアドレス）で、`oldlenp` にその長さを
        // 渡す（`sysctlbyname` の「呼び出し前に `*oldlenp` へバッファ長を
        // 設定する」契約）。`newp` は null・`newlen` は 0 とし「値を変更
        // しない読み取り専用呼び出し」の契約を満たす。戻り値・書き戻された
        // `len` は呼び出し直後に検査する。FFI 境界の `unsafe` はこの 1
        // 箇所に限定する（`.claude/rules/coding-rust.md`「`unsafe` は FFI
        // 境界等の必要最小限に留め、理由をコメントで明記」）。
        let ret = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                &mut value as *mut usize as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret != 0 || len != size_of::<usize>() || value == 0 {
            return None;
        }
        Some(value)
    }
}

/// Apple Silicon の P コア（`hw.perflevel0`。高性能クラスタ）の L1D／L2
/// 実測値（バイト）を読む。E コア（`hw.perflevel1`）は対象外
/// （`gemm_blis` の実行時 ISA ディスパッチ〈#185〉・マイクロカーネル選定は
/// P コア想定でチューニングされている。#481 §3 と同じ前提）。
#[cfg(target_os = "macos")]
fn read_cache_sizes() -> Option<(usize, usize)> {
    let l1d = sysctl_ffi::read_usize("hw.perflevel0.l1dcachesize")?;
    let l2 = sysctl_ffi::read_usize("hw.perflevel0.l2cachesize")?;
    Some((l1d, l2))
}

/// macOS 以外は sysctl 経路自体を持たないため常に `None`
/// （[`detected_blocks`] が [`super::default_blocks`] へフォールバックする）。
#[cfg(not(target_os = "macos"))]
fn read_cache_sizes() -> Option<(usize, usize)> {
    None
}

/// プロセス内で 1 回だけ [`read_cache_sizes`] を評価しキャッシュする
/// （`sysctl` はプロセス生存期間中不変のため呼び出しごとの再取得は不要）。
static CACHE_SIZES: OnceLock<Option<(usize, usize)>> = OnceLock::new();

fn detected_cache_sizes() -> Option<(usize, usize)> {
    *CACHE_SIZES.get_or_init(read_cache_sizes)
}

/// 実行時検出した L1D／L2 実測値から [`compute_blocks`] で MC/KC/NC を
/// 算出する（非 macOS・sysctl 失敗・算出不能時は [`super::default_blocks`]
/// へフォールバック。受け入れ条件 3）。
///
/// 本番未結線（モジュールドキュメント参照）。テスト専用パラメータ化入口・
/// 実機 A/B ハーネスから `mr`／`nr` に対象マイクロカーネルの
/// [`super::microkernel::Microkernel::MR`]／`NR` を渡して呼ぶ想定。
pub(crate) fn detected_blocks(mr: usize, nr: usize) -> BlockSizes {
    detected_cache_sizes()
        .and_then(|(l1d, l2)| compute_blocks(l1d, l2, mr, nr))
        .unwrap_or_else(super::default_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_blocks_apple_m4_max_like_values_stays_within_clamp_bounds() {
        // Apple M4 Max P コア相当の代表値（L1D=192KiB・L2=16MiB。#481 §3）。
        // NEON 既定カーネル（MR=8・NR=12。#559）で算出する。
        let blocks = compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 12)
            .expect("正当な範囲の値は Some を返すはず");
        assert!((KC_MIN..=KC_MAX).contains(&blocks.kc));
        assert!((MC_MIN..=MC_MAX).contains(&blocks.mc));
        assert!((NC_MIN..=NC_MAX).contains(&blocks.nc));
        // MC は MR（8）の倍数へ丸めている。
        assert_eq!(blocks.mc % 8, 0);
        assert_eq!(blocks.nc % 12, 0);
    }

    #[test]
    fn compute_blocks_rejects_zero_mr_or_nr() {
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 0, 12).is_none());
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 0).is_none());
    }

    #[test]
    fn compute_blocks_rejects_zero_cache_sizes() {
        assert!(compute_blocks(0, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 0, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_rejects_implausibly_small_cache_sizes() {
        // 正当性検査範囲（[`L1D_SANE_MIN`]／[`L2_SANE_MIN`]）を下回る値は
        // sysctl の異常値（破損・不正取得）とみなし拒否する。
        assert!(compute_blocks(64, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 4096, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_rejects_implausibly_large_cache_sizes() {
        // 正当性検査範囲（[`L1D_SANE_MAX`]／[`L2_SANE_MAX`]）を超える値は
        // 非現実的（sysctl の異常値・将来の桁違いなハードウェア変化）と
        // みなし拒否する（0 値検査だけでは捉えられない fail-closed 契約）。
        assert!(compute_blocks(64 * 1024 * 1024, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 1024 * 1024 * 1024, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_clamped_mc_nc_stay_multiples_for_non_divisor_mr_nr() {
        // レビュー指摘（#753）: `MC_MIN`／`NC_MIN`（64／256）は `mr`／`nr`
        // が 7 のような非自明な値のとき倍数にならない。クランプが効く
        // 状況（L2 実容量を極端に大きくし `mc_raw`／`nc_raw` を
        // `MC_MAX`／`NC_MAX` へ張り付かせる）でも `mc`／`nc` が `mr`／`nr`
        // の倍数であるというドキュメント契約（本ファイル冒頭・
        // `docs/perf/cpu-gemm-runtime-cache-detect.md` §2）が成立し続ける
        // ことを検証する（M4 Max 相当値〈MR=8・NR=12〉は `MC_MAX=1024`・
        // `NC_MAX=16384` がたまたま倍数のため、この回帰は非自明な
        // `mr`／`nr` でなければ検出できない）。
        let blocks = compute_blocks(192 * 1024, 1024 * 1024 * 1024 / 8, 7, 13)
            .or_else(|| compute_blocks(192 * 1024, L2_SANE_MAX, 7, 13))
            .expect("正当な範囲の値は Some を返すはず");
        assert_eq!(blocks.mc % 7, 0, "mc={} は mr=7 の倍数ではない", blocks.mc);
        assert_eq!(
            blocks.nc % 13,
            0,
            "nc={} は nr=13 の倍数ではない",
            blocks.nc
        );
        assert!((MC_MIN..=MC_MAX).contains(&blocks.mc));
        assert!((NC_MIN..=NC_MAX).contains(&blocks.nc));
    }

    #[test]
    fn compute_blocks_small_l1d_clamps_kc_to_minimum() {
        // L1D 実容量が小さいほど算出 KC は縮むが、KC_MIN を下回らない
        // （マイクロカーネル 1 反復あたりの演算密度を落としすぎない下限）。
        let blocks = compute_blocks(L1D_SANE_MIN, 16 * 1024 * 1024, 8, 12).unwrap();
        assert_eq!(blocks.kc, KC_MIN);
    }

    #[test]
    fn compute_blocks_rejects_mr_nr_overflowing_intermediate_arithmetic() {
        // レビュー指摘（#753）: `mr`／`nr` は 0 のみ検査しており上限を
        // 検査していなかったため、`mr + nr`（`checked_add`）・
        // `F32_BYTES * (mr + nr)`（`checked_mul`）が `usize` オーバーフロー
        // する組み合わせで debug では panic・release では誤った丸め値が
        // 生じうる状態だった。`usize::MAX` に近い `mr`／`nr` で `None`
        // （fail-closed。呼び出し元は `default_blocks()` へフォールバック）
        // を返すことを検証する。
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, usize::MAX, 1).is_none());
        assert!(
            compute_blocks(
                192 * 1024,
                16 * 1024 * 1024,
                usize::MAX / 2,
                usize::MAX / 2 + 2
            )
            .is_none()
        );
        // `clamp_to_multiple` 側（`round_up_to_multiple` の `checked_mul`）
        // も同様に防御されていることを、`mc_raw`／`nc_raw` が確実にクランプ
        // される極端に大きな L2 実容量と、それ自体は `mr + nr` を
        // オーバーフローさせない範囲の巨大な `mr` の組み合わせで検証する。
        assert!(compute_blocks(192 * 1024, L2_SANE_MAX, usize::MAX / 4, 12).is_none());
    }

    /// [`detected_blocks`] は非 macOS・sysctl 失敗時に必ず
    /// [`super::default_blocks`] 相当の非ゼロ値を返す（受け入れ条件 3。
    /// Linux CI 上でもこのフォールバック経路を実行し続けることで、
    /// 本モジュールが「常に不活性なデッドコード」（PR #766 で撤去された
    /// 構成）に陥っていないことを検証する）。
    #[test]
    fn detected_blocks_returns_valid_block_sizes_on_any_platform() {
        let blocks = detected_blocks(8, 12);
        assert!(blocks.mc > 0 && blocks.kc > 0 && blocks.nc > 0);
    }
}
