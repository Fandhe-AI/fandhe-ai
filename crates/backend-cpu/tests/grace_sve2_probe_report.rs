//! イシュー #2123（親 #2121）: Grace CPU（GB10）の SVE／SVE2 検出プローブ。
//!
//! ## 役割
//!
//! `docs/gb10-unified-memory-grace-cpu-consideration.md` §5 の「SVE2 を `/proc/cpuinfo` と
//! getauxval 系（AT_HWCAP／AT_HWCAP2）で検出できるか」を確定する材料を出す。
//! 検出は std のみ（`/proc/cpuinfo`・`/proc/self/auxv`・`is_aarch64_feature_detected!`）で行い、
//! `unsafe`・新規依存（`libc` は deps-policy 第 10 区分で onnx-interop 用途限定のため使わない）は無い。
//! 判定規則は `docs/perf/logs/gb10-unified-memory-grace-2123/RULE.txt`（実測前に固定）。
//!
//! ## 構成
//!
//! - pure parser 群（fail-closed: 形式不正は `None`）と、その単体テスト（CI で実行）。
//! - `#[ignore]` の実機レポート。env `EXPECT_GRACE_SVE2`: 未設定=出力のみ／`1`=3 経路一致と VL
//!   取得を要求／`0`=SVE2 非検出を要求／他は panic で拒否（値はエコーしない）。

/// auxv のキー（Linux `AT_HWCAP`／`AT_HWCAP2`）。
const AT_HWCAP: u64 = 16;
const AT_HWCAP2: u64 = 26;
/// aarch64 の `HWCAP_SVE`（AT_HWCAP bit22）・`HWCAP2_SVE2`（AT_HWCAP2 bit1）。
const HWCAP_SVE: u64 = 1 << 22;
const HWCAP2_SVE2: u64 = 1 << 1;

/// `/proc/cpuinfo` 相当の文字列から、最初の `Features` 行のトークン集合を返す。無ければ `None`。
fn parse_cpuinfo_features(text: &str) -> Option<Vec<String>> {
    for line in text.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case("features") {
            return Some(v.split_whitespace().map(str::to_owned).collect());
        }
    }
    None
}

/// auxv のバイト列（native endian の u64 ペア列）を `(key, value)` へ分解する。
/// 長さが 16 の倍数でなければ `None`（fail-closed）。AT_NULL(0) で打ち切る。
fn parse_auxv(bytes: &[u8]) -> Option<Vec<(u64, u64)>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(16) {
        return None;
    }
    let mut out = Vec::new();
    let (chunks, _) = bytes.as_chunks::<16>();
    for chunk in chunks {
        let key = u64::from_ne_bytes(chunk[0..8].try_into().ok()?);
        let val = u64::from_ne_bytes(chunk[8..16].try_into().ok()?);
        if key == 0 {
            break;
        }
        out.push((key, val));
    }
    Some(out)
}

fn auxv_lookup(pairs: &[(u64, u64)], key: u64) -> Option<u64> {
    pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// `/proc/sys/abi/sve_default_vector_length`（バイト数の 10 進文字列）を parse する。0・非数値は `None`。
fn parse_vl_bytes(text: &str) -> Option<u32> {
    text.trim().parse::<u32>().ok().filter(|v| *v > 0)
}

/// GB10 実機ログ（`docs/perf/logs/cpu-gemm-b-laneq-vec-ab-1318/lscpu-dgx.txt`）の Features 行相当の固定 fixture。
const GB10_FEATURES_LINE: &str = "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svebf16 i8mm bf16 dgh bti";

#[test]
fn cpuinfo_features_detects_sve2_on_gb10_fixture() {
    let t = parse_cpuinfo_features(GB10_FEATURES_LINE).expect("fixture");
    assert!(t.iter().any(|x| x == "sve"));
    assert!(t.iter().any(|x| x == "sve2"));
}

#[test]
fn cpuinfo_features_negative_and_missing() {
    let x86 = "flags\t: fpu sse sse2 avx2\nmodel name : x";
    assert!(parse_cpuinfo_features(x86).is_none());
    assert!(parse_cpuinfo_features("").is_none());
    let t = parse_cpuinfo_features("Features : fp asimd").expect("features");
    assert!(!t.iter().any(|x| x == "sve2"));
}

fn auxv_bytes(pairs: &[(u64, u64)]) -> Vec<u8> {
    let mut b = Vec::new();
    for (k, v) in pairs {
        b.extend_from_slice(&k.to_ne_bytes());
        b.extend_from_slice(&v.to_ne_bytes());
    }
    b
}

#[test]
fn auxv_parses_and_finds_hwcaps() {
    let b = auxv_bytes(&[
        (AT_HWCAP, HWCAP_SVE | 1),
        (AT_HWCAP2, HWCAP2_SVE2),
        (0, 0),
        (AT_HWCAP, 0),
    ]);
    let p = parse_auxv(&b).expect("parse");
    assert_eq!(p.len(), 2, "AT_NULL で打ち切る");
    assert_ne!(auxv_lookup(&p, AT_HWCAP).expect("hwcap") & HWCAP_SVE, 0);
    assert_ne!(auxv_lookup(&p, AT_HWCAP2).expect("hwcap2") & HWCAP2_SVE2, 0);
}

#[test]
fn auxv_rejects_malformed_length() {
    assert!(parse_auxv(&[]).is_none());
    assert!(parse_auxv(&[0u8; 15]).is_none());
    assert!(parse_auxv(&[0u8; 17]).is_none());
}

#[test]
fn vl_parser_is_fail_closed() {
    assert_eq!(parse_vl_bytes("16\n"), Some(16));
    assert_eq!(parse_vl_bytes("0"), None);
    assert_eq!(parse_vl_bytes("abc"), None);
    assert_eq!(parse_vl_bytes("-1"), None);
    assert_eq!(parse_vl_bytes(""), None);
}

fn std_detect() -> (Option<bool>, Option<bool>) {
    #[cfg(target_arch = "aarch64")]
    {
        (
            Some(std::arch::is_aarch64_feature_detected!("sve")),
            Some(std::arch::is_aarch64_feature_detected!("sve2")),
        )
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        (None, None)
    }
}

fn show(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "true",
        Some(false) => "false",
        None => "na",
    }
}

#[test]
#[ignore = "GB10 実機依存（イシュー #2123）。EXPECT_GRACE_SVE2=1|0 で assert、未設定は出力のみ"]
fn grace_sve2_probe_dump() {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| parse_cpuinfo_features(&t));
    let cpuinfo_sve = cpuinfo.as_ref().map(|t| t.iter().any(|x| x == "sve"));
    let cpuinfo_sve2 = cpuinfo.as_ref().map(|t| t.iter().any(|x| x == "sve2"));
    let auxv = std::fs::read("/proc/self/auxv")
        .ok()
        .and_then(|b| parse_auxv(&b));
    // HWCAP のビット意味は arch 依存（x86 の AT_HWCAP2 bit1 は SVE2 ではない）。aarch64 以外は na。
    let is_arm = cfg!(target_arch = "aarch64");
    let aux_sve = auxv
        .as_ref()
        .filter(|_| is_arm)
        .and_then(|p| auxv_lookup(p, AT_HWCAP))
        .map(|v| v & HWCAP_SVE != 0);
    let aux_sve2 = auxv
        .as_ref()
        .filter(|_| is_arm)
        .and_then(|p| auxv_lookup(p, AT_HWCAP2))
        .map(|v| v & HWCAP2_SVE2 != 0);
    let (std_sve, std_sve2) = std_detect();
    let vl = std::fs::read_to_string("/proc/sys/abi/sve_default_vector_length")
        .ok()
        .and_then(|t| parse_vl_bytes(&t));
    eprintln!(
        "grace_sve2_probe cpuinfo_sve={} cpuinfo_sve2={} auxv_hwcap_sve={} auxv_hwcap2_sve2={} std_detect_sve={} std_detect_sve2={} sve_default_vl_bytes={}",
        show(cpuinfo_sve),
        show(cpuinfo_sve2),
        show(aux_sve),
        show(aux_sve2),
        show(std_sve),
        show(std_sve2),
        vl.map_or_else(|| "na".to_owned(), |v| v.to_string()),
    );
    match std::env::var("EXPECT_GRACE_SVE2").ok().as_deref() {
        None => {}
        Some("1") => {
            assert_eq!(cpuinfo_sve2, Some(true), "cpuinfo 経路が sve2 を検出しない");
            assert_eq!(aux_sve2, Some(true), "auxv HWCAP2 経路が sve2 を検出しない");
            assert_eq!(std_sve2, Some(true), "std_detect 経路が sve2 を検出しない");
            assert!(vl.is_some(), "SVE ベクトル長を取得できない");
        }
        Some("0") => {
            assert_ne!(cpuinfo_sve2, Some(true));
            assert_ne!(aux_sve2, Some(true));
            assert_ne!(std_sve2, Some(true));
        }
        Some(_) => panic!("EXPECT_GRACE_SVE2 は 0/1 のみ"),
    }
}
