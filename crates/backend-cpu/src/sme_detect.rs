//! Arm SME（Scalable Matrix Extension）の実行時検出（イシュー #1587）。
//!
//! [`gemm_blis::microkernel::SmeKernel::try_new`] から呼ばれ、「実行 CPU が
//! SME の非拡張 FP32 外積命令（`fmopa`）を安全に実行できるか」を判定する。
//! `std::arch::is_aarch64_feature_detected!("sme")` は本リポジトリの
//! rustc（stable 1.96 系）では `stdarch_aarch64_feature_detection` が
//! unstable のため使用できない（E0658。計画セッションで実測確認済み）ため、
//! OS 側の機能フラグを直接読む fail-closed 方式を採る（[`Avx2Kernel::try_new`]
//! 等と同じ「検出済みの場合のみ構築可能なトークン」パターンをここでも踏襲
//! する。`gemm_blis::microkernel` モジュール冒頭ドキュメント参照）。
//!
//! ## fail-closed 方針（OWASP A03・`.claude/rules/security.md`）
//!
//! - OS フラグ・`/proc/cpuinfo` の読み取り・parse に失敗した場合はすべて
//!   「非対応」（`false`）として扱う。環境変数による上書き機構は設けない
//!   （`gemm_blis::microkernel` モジュール冒頭「環境変数等による dispatch
//!   上書き機構は設けない」方針と同じ理由）。
//! - macOS の OS フラグ照会は `crate::thread_limit::read_sysctl`
//!   （`/usr/sbin/sysctl` 固定引数・`env_clear()`）を再利用する（新しい
//!   子プロセス起動経路を増やさない）。
//! - OS フラグが SME 対応を報告した場合に限り、`rdsvl`（SVL 読み取りの
//!   専用命令。ストリーミングモードへ一時的に遷移するのみで演算は行わない
//!   読み取り専用命令）を実行して SVL=64 バイト（512 bit。本番マイクロ
//!   カーネル `gemm_blis::microkernel::sme` が前提とするベクトル長）を
//!   確認する。OS フラグ判定を経ずに `rdsvl` を実行すると非対応 CPU で
//!   SIGILL になりうるため、必ず OS 判定の後にのみ実行する。

use std::sync::OnceLock;

/// `sme_report()` が返す診断結果（`ThreadLimitReport` 等と同型。facade へは
/// 昇格しない内部診断 API。env_info 記録・実機実測ログ用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmeReport {
    /// OS が SME＋非拡張 FP32 外積（`SME_F32F32`／`smef32f32`）対応を
    /// 報告したか。
    pub os_flag: bool,
    /// `rdsvl` で読み取った SVL（バイト単位）。`os_flag` が `false` の
    /// 場合は `rdsvl` 自体を実行しないため常に `None`。
    pub svl_bytes: Option<usize>,
    /// 本番マイクロカーネル（[`crate::gemm_blis::microkernel::SmeKernel`]）
    /// を構築可能かどうか（`os_flag && svl_bytes == Some(64)`）。
    pub kernel_enabled: bool,
}

/// `/proc/cpuinfo` の最初の `Features` 行から `sme`・`smef32f32` の
/// 両方を含むかを判定する純関数（単体テスト可能。実際のファイル I/O は
/// [`linux_os_flag`] が担う）。
///
/// 本番経路では Linux の [`linux_os_flag`] からのみ呼ばれるため
/// `cfg(any(target_os = "linux", test))` で条件付きコンパイルする
/// （`thread_limit::parse_sysctl_stdout` と同じ理由。macOS 単体ビルドでは
/// 未使用になり `dead_code` を誤検出するため）。
#[cfg(any(target_os = "linux", test))]
fn parse_cpuinfo_features(cpuinfo: &str) -> bool {
    for line in cpuinfo.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() != "Features" {
            continue;
        }
        let tokens: Vec<&str> = value.split_whitespace().collect();
        return tokens.contains(&"sme") && tokens.contains(&"smef32f32");
    }
    false
}

#[cfg(target_os = "linux")]
fn linux_os_flag() -> bool {
    match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(text) => parse_cpuinfo_features(&text),
        Err(_) => false,
    }
}

#[cfg(target_os = "macos")]
fn macos_os_flag() -> bool {
    // `read_sysctl` は `0` を `None` へ倒す（`parse_sysctl_stdout` 参照）ため
    // `Some(1)` と比較するだけで「1 以外はすべて非対応」を fail-closed に
    // 判定できる。
    crate::thread_limit::read_sysctl("hw.optional.arm.FEAT_SME") == Some(1)
        && crate::thread_limit::read_sysctl("hw.optional.arm.SME_F32F32") == Some(1)
}

/// OS フラグ判定（macOS／Linux 以外は常に非対応）。
fn os_flag() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos_os_flag()
    }
    #[cfg(target_os = "linux")]
    {
        linux_os_flag()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// SVL（Streaming Vector Length。バイト単位）を読み取る。呼び出し元
/// （[`decide`]）が [`os_flag`] を確認した後にのみ呼ぶ契約（非対応 CPU
/// では SIGILL になりうる）。
///
/// # Safety
///
/// 呼び出し元が [`os_flag`] で OS 側の SME 対応を確認済みであることが
/// 前提（`rdsvl` はストリーミングモードへの遷移を伴う SME 命令のため、
/// 非対応 CPU での実行は未定義動作〈実態は SIGILL〉になりうる）。
#[cfg(target_arch = "aarch64")]
unsafe fn rdsvl_bytes() -> usize {
    let mut svl: u64;
    // SAFETY（呼び出し元契約は本関数ドキュメント参照）:
    // - `.arch_extension sme` はアセンブラへ SME 命令の使用を許可する
    //   ディレクティブ（コンパイル時のみに影響し実行時状態を変えない）。
    // - `rdsvl {0}, #1` は現在の SVL をバイト単位で汎用レジスタへ読み出す
    //   読み取り専用命令（メモリアクセスなし・レジスタ状態を変更しない）。
    //   `options(nomem, nostack, preserves_flags)` はこの性質をそのまま
    //   宣言する（計画セッションでの事前検証プローブと同一の宣言）。
    unsafe {
        std::arch::asm!(
            ".arch_extension sme",
            "rdsvl {0}, #1",
            out(reg) svl,
            options(nomem, nostack, preserves_flags)
        );
    }
    svl as usize
}

/// 本番マイクロカーネルが前提とする SVL（512 bit = 64 バイト。
/// `gemm_blis::microkernel::sme` モジュール参照）。
const REQUIRED_SVL_BYTES: usize = 64;

/// OS フラグ・SVL 実測から [`SmeReport`] を組み立てる純ロジック
/// （`rdsvl` 呼び出し自体は [`sme_report`] 側に閉じ込め、ここでは
/// 「フラグと SVL から kernel_enabled を導く」判定のみを行う。単体
/// テスト可能にする分離）。
fn decide(os_flag: bool, svl_bytes: Option<usize>) -> SmeReport {
    let kernel_enabled = os_flag && svl_bytes == Some(REQUIRED_SVL_BYTES);
    SmeReport {
        os_flag,
        svl_bytes,
        kernel_enabled,
    }
}

/// プロセス内で 1 回だけ検出を行い、以降は結果をキャッシュする
/// （[`Isa::detect`] と同じ `OnceLock` パターン）。
pub fn sme_report() -> SmeReport {
    static REPORT: OnceLock<SmeReport> = OnceLock::new();
    *REPORT.get_or_init(|| {
        let flag = os_flag();
        if !flag {
            return decide(false, None);
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `flag` が `true`（OS が SME 対応を報告済み）の場合に
            // 限り `rdsvl_bytes` を呼ぶ（本関数ドキュメント・
            // `rdsvl_bytes` の Safety 契約参照）。
            let svl = unsafe { rdsvl_bytes() };
            decide(true, Some(svl))
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // os_flag() は aarch64 以外では常に false を返すためここへは
            // 到達しないが、cfg 分岐の網羅性のため明示する。
            decide(false, None)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpuinfo_features_detects_both_tokens() {
        let cpuinfo = "processor\t: 0\nFeatures\t: fp asimd sme smef32f32 sve\n";
        assert!(parse_cpuinfo_features(cpuinfo));
    }

    #[test]
    fn parse_cpuinfo_features_rejects_missing_smef32f32() {
        let cpuinfo = "processor\t: 0\nFeatures\t: fp asimd sme sve\n";
        assert!(!parse_cpuinfo_features(cpuinfo));
    }

    #[test]
    fn parse_cpuinfo_features_rejects_missing_sme() {
        let cpuinfo = "processor\t: 0\nFeatures\t: fp asimd smef32f32 sve\n";
        assert!(!parse_cpuinfo_features(cpuinfo));
    }

    #[test]
    fn parse_cpuinfo_features_handles_empty_input() {
        assert!(!parse_cpuinfo_features(""));
    }

    #[test]
    fn parse_cpuinfo_features_uses_first_features_line_only() {
        // 複数コアぶんの Features 行が並ぶ実際の /proc/cpuinfo を模す。
        // 最初の行のみを見る契約（全 CPU が同一機能集合を持つ前提。
        // 異種コア構成でも「1 コアでも対応」を過大評価しない fail-closed
        // 側の単純化）。
        let cpuinfo = "processor\t: 0\nFeatures\t: fp asimd sme smef32f32\nprocessor\t: 1\nFeatures\t: fp asimd\n";
        assert!(parse_cpuinfo_features(cpuinfo));
    }

    #[test]
    fn decide_requires_both_os_flag_and_exact_svl() {
        assert!(!decide(false, None).kernel_enabled);
        assert!(!decide(false, Some(64)).kernel_enabled);
        assert!(!decide(true, None).kernel_enabled);
        assert!(!decide(true, Some(32)).kernel_enabled);
        assert!(decide(true, Some(64)).kernel_enabled);
    }

    #[test]
    fn sme_report_is_cached_and_consistent() {
        let first = sme_report();
        let second = sme_report();
        assert_eq!(first, second);
    }
}
