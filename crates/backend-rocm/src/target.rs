//! HIPRTC の `--gpu-architecture` に渡す gfx target の許可リスト。
//!
//! 役割: オプション文字列の唯一の生成元。外部入力（デバイス照会値を含む）は
//! [`parse_gcn_arch_name`] の完全一致 allowlist を通った enum 値としてのみ使い、
//! 文字列連結でオプションを作らない（A03。docs/backend-rocm-hiprtc-gemm-decision.md §3）。
//! `hiprtc.rs` の `compile_code_object` から呼ばれる。

use crate::error::RocmError;

/// コンパイル対象の gfx target。手配済み実機（MI300X = gfx942）から始め、他は実測後に追加する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GfxTarget {
    /// CDNA3（MI300X。wave64）。
    Gfx942,
}

impl GfxTarget {
    /// `--gpu-architecture=` の値（固定リテラル）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
        }
    }
}

/// `gcnArchName`（例 `gfx942:sramecc+:xnack-`）を allowlist で照合する純関数。
///
/// 先頭の `gfxNNN` 部分のみを完全一致で照合し、`:feature±` 部分はオプションに使わず捨てる。
pub fn parse_gcn_arch_name(name: &str) -> Result<GfxTarget, RocmError> {
    let head = name.split(':').next().unwrap_or("");
    match head {
        "gfx942" => Ok(GfxTarget::Gfx942),
        _ => Err(RocmError::UnsupportedTarget {
            detail: format!("許可リストにありません: {head:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_feature_suffix() {
        assert_eq!(parse_gcn_arch_name("gfx942"), Ok(GfxTarget::Gfx942));
        assert_eq!(
            parse_gcn_arch_name("gfx942:sramecc+:xnack-"),
            Ok(GfxTarget::Gfx942)
        );
    }

    #[test]
    fn rejects_unknown_and_injection() {
        for s in [
            "",
            "gfx90a",
            "gfx9420",
            " gfx942",
            "gfx942 --foo",
            "GFX942",
            "x;gfx942",
        ] {
            assert!(parse_gcn_arch_name(s).is_err(), "{s:?}");
        }
    }

    #[test]
    fn as_str_is_fixed() {
        assert_eq!(GfxTarget::Gfx942.as_str(), "gfx942");
    }
}
