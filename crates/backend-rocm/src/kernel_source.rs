//! HIPRTC へ渡すカーネルソース型の共通基盤（イシュー #3439・親 #3150・分解案 R8）。
//!
//! 役割: `hiprtc::compile_code_object` が受け取れるソースを封印 trait [`CompileSource`] を
//! 実装する型に限る（任意の `&str` はコンパイルできない。A03）。GEMM は従来どおり
//! `kernels_gemm::GemmSource`、gather/scatter・scan・sort 等の添字系カーネルは本ファイルの
//! [`KernelSource`] を使う。`KernelSource` は「`&'static str` の静的テンプレート本体」と
//! 「名前を検証した `#define NAME <u32>`」だけから作る。shape・dim・k など外部由来の値は
//! ソースへ入れず、デバイスバッファかカーネル引数で渡す（`backend-cuda::warp_geometry::
//! render_defines` と同方針）。呼び出し元は各 `kernels_*.rs` の `render_*_source`。

use crate::error::RocmError;
use crate::kernels_gemm::GemmSource;
use crate::kernels_linalg::LinalgSource;
use crate::warp_geometry::WarpGeometry;
use std::fmt::Write as _;

/// `compile_code_object` が受け付けるソース。実装は本クレート内の検証済み型だけ。
pub(crate) trait CompileSource {
    /// ソース文字列。
    fn as_str(&self) -> &str;
    /// `hiprtcCreateProgram` へ渡す固定のプログラム名（外部入力から組み立てない）。
    fn program_name(&self) -> &'static str;
}

impl CompileSource for GemmSource {
    fn as_str(&self) -> &str {
        GemmSource::as_str(self)
    }
    fn program_name(&self) -> &'static str {
        "fandhe_gemm_f32.hip"
    }
}

impl CompileSource for LinalgSource {
    fn as_str(&self) -> &str {
        LinalgSource::as_str(self)
    }
    fn program_name(&self) -> &'static str {
        LinalgSource::program_name(self)
    }
}

/// 静的テンプレートと検証済み数値の `#define` だけで作るコンパイル対象ソース。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelSource(String);

impl KernelSource {
    /// `defines`（名前は `[A-Z_][A-Z0-9_]*`・値は `u32` を 10 進整形）を前置して本体を連結する。
    ///
    /// 名前が不正、または本体に NUL を含む場合は `InvalidKernelConfig`（fail-closed）。
    pub(crate) fn render(
        defines: &[(&'static str, u32)],
        body: &'static str,
    ) -> Result<Self, RocmError> {
        if body.contains('\0') {
            return Err(RocmError::InvalidKernelConfig {
                detail: "カーネル本体に NUL を含みます".into(),
            });
        }
        let mut s = String::with_capacity(body.len() + defines.len() * 32);
        for (name, value) in defines {
            if !is_valid_define_name(name) {
                return Err(RocmError::InvalidKernelConfig {
                    detail: format!("#define 名が不正です: {name:?}"),
                });
            }
            // 値は u32 の 10 進表記（数字のみ）で、文字列の連結入力にならない。
            let _ = writeln!(s, "#define {name} {value}");
        }
        s.push_str(body);
        Ok(Self(s))
    }

    /// [`Self::render`] に加え、wave 幅の `#define` 群（`WARP_SIZE`・`WARP_HALF`・
    /// `WARP_SHFL_XOR` 等。[`WarpGeometry::render_hip_defines`] が検証済みの数値と固定テンプレートだけで
    /// 作る）を先頭へ前置する。縮約系（R6 正規化・R4 softmax 等）の render 関数が使う入口で、
    /// 任意の文字列を前置する口は作らない（A03）。`WARPS_PER_BLOCK` は出さない
    /// （block を 1 wave に固定する系は不要。必要な系は別途 `defines` で渡す）。
    pub(crate) fn render_with_warp(
        geom: WarpGeometry,
        defines: &[(&'static str, u32)],
        body: &'static str,
    ) -> Result<Self, RocmError> {
        let warp = geom.render_hip_defines(None)?;
        let rest = Self::render(defines, body)?;
        let mut s = String::with_capacity(warp.as_str().len() + rest.0.len());
        s.push_str(warp.as_str());
        s.push_str(&rest.0);
        Ok(Self(s))
    }
}

impl CompileSource for KernelSource {
    fn as_str(&self) -> &str {
        &self.0
    }
    fn program_name(&self) -> &'static str {
        "fandhe_kernel.hip"
    }
}

/// `[A-Z_][A-Z0-9_]*`（空は不可）。
fn is_valid_define_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// テスト用: カーネル `name` の宣言から引数の種別列を取り出す（`launch::ArgKind` との一致検査用）。
///
/// ポインタ（`*` を含む）→ `Ptr`、`int` → `I32`、`long long` → `I64`、`float` → `F32`。
/// 起動側の `*_ARGS` 定数とソース署名のドリフトを機械的に検出するために使う。
#[cfg(test)]
pub(crate) fn kernel_param_kinds(body: &str, name: &str) -> Vec<crate::launch::ArgKind> {
    use crate::launch::ArgKind;
    let marker = format!("void {name}(");
    let start = body
        .find(&marker)
        .unwrap_or_else(|| panic!("kernel {name} not found"))
        + marker.len();
    let end = start + body[start..].find(')').expect("closing paren");
    body[start..end]
        .split(',')
        .map(|p| {
            let p = p.trim();
            if p.contains('*') {
                ArgKind::Ptr
            } else if p.starts_with("long long ") {
                ArgKind::I64
            } else if p.starts_with("int ") {
                ArgKind::I32
            } else if p.starts_with("float ") {
                ArgKind::F32
            } else {
                panic!("unsupported param type: {p:?}")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_define_names() {
        for bad in ["", "block", "1A", "A-B", "A B", "A;B", "Ab"] {
            assert!(
                KernelSource::render(&[(bad, 1)], "x").is_err(),
                "{bad:?} は拒否される"
            );
        }
        assert!(KernelSource::render(&[("BLOCK_DIM", 256), ("_X1", 2)], "x").is_ok());
    }

    #[test]
    fn render_is_deterministic_and_defines_are_digits_only() {
        let a = KernelSource::render(&[("BLOCK_DIM", 256)], "body").unwrap();
        let b = KernelSource::render(&[("BLOCK_DIM", 256)], "body").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.as_str(), "#define BLOCK_DIM 256\nbody");
        for line in a.as_str().lines().filter(|l| l.starts_with("#define")) {
            let value = line.rsplit(' ').next().unwrap();
            assert!(value.chars().all(|c| c.is_ascii_digit()), "{line}");
        }
    }

    #[test]
    fn render_with_warp_prepends_only_validated_defines() {
        let g = WarpGeometry::new(64).unwrap();
        let s = KernelSource::render_with_warp(g, &[("BLOCK_DIM", 256)], "body").unwrap();
        let want = format!(
            "{}#define BLOCK_DIM 256\nbody",
            g.render_hip_defines(None).unwrap().as_str()
        );
        assert_eq!(s.as_str(), want);
        assert!(!s.as_str().contains("WARPS_PER_BLOCK"));
        // `#define` の値は数字（または固定テンプレートのマクロ定義）だけ。
        for line in s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define") && !l.contains('('))
        {
            let value = line.rsplit(' ').next().unwrap();
            assert!(value.chars().all(|c| c.is_ascii_alphanumeric()), "{line}");
        }
        assert!(KernelSource::render_with_warp(g, &[("bad", 1)], "x").is_err());
        assert_eq!(
            s,
            KernelSource::render_with_warp(g, &[("BLOCK_DIM", 256)], "body").unwrap()
        );
    }

    #[test]
    fn rendered_source_has_no_nul() {
        let s = KernelSource::render(&[("BLOCK_DIM", u32::MAX)], "extern \"C\" {}").unwrap();
        assert!(!s.as_str().contains('\0'));
    }

    #[test]
    fn param_kind_parser_classifies_types() {
        use crate::launch::ArgKind::*;
        let body = "extern \"C\" __global__ void k(\n const float* __restrict__ a,\n int n,\n long long m,\n float s,\n unsigned long long* __restrict__ keys)\n{}";
        // unsigned long long* はポインタとして先に分類される。
        assert_eq!(kernel_param_kinds(body, "k"), vec![Ptr, I32, I64, F32, Ptr]);
    }
}
