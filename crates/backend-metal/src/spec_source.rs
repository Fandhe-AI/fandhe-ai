//! `gemm_simdgroup_tiled` のソーステキスト特殊化経路（イシュー #1288。
//! E2 試作）。
//!
//! `crate::pipeline` の function constant 経路（`MTLFunctionConstantValues`
//! 経由でコンパイル済みライブラリを構成別に「特殊化」する。#188/#538/
//! #540/#809/#1138/#1282）に対し、本モジュールは候補（[`crate::tile::
//! TileConfig`]）ごとに厳密なアキュムレータ配列サイズ（`ACC_ROWS_CAP`/
//! `ACC_COLS_CAP`）を `#define` でリテラル埋め込みしたソーステキストを
//! 生成し、`crate::pipeline::compile_source` で**候補ごとに再コンパイル**
//! する opt-in 経路の入口を提供する。`docs/perf/
//! metal-gemm-n4096-kernel-gap.md` §2（H1 レジスタ圧仮説）が残した未実施
//! 候補（E2: function constant 特殊化後もコンパイラが厳密サイズでレジスタ
//! 割付を最適化しない可能性）を検証するための機構であり、性能実測・
//! 本番結線判断は行わない（`crate::tile::SOURCE_SPECIALIZATION_ENABLED`
//! 〈既定 `false`〉・後続イシュー #1289／#1302 のスコープ）。
//!
//! `objc2` 系 FFI に一切触れない純粋な文字列生成ロジックのみで構成する
//! ため、`crate::pad`/`crate::tile`/`crate::layout` と同じ設計判断で
//! `cfg(any(test, target_os = "macos"))` を付け、Linux（本実装環境・CI）
//! でも `cargo test -p fandhe-ai-backend-metal` で生成結果を単体テスト
//! できるようにしてある（`crate::tile::TileConfig::acc_rows`/`acc_cols`
//! と同一の cfg 条件。dead_code 誤検知回避も同じ理由）。

use crate::layout::TransposePattern;
use crate::tile::TileConfig;

/// `shaders/gemm.metal` の全文（naive/tiled/simdgroup/f16 等の全カーネルを
/// 含む）。[`specialized_gemm_source`] が `#define GEMM_SPEC_*` ヘッダを
/// 前置する対象。`crate::pipeline::compile_gemm_library`（本番既定の
/// function constant 経路）も本定数を参照する（従来 `pipeline.rs` 内の
/// private const だったものを本モジュールへ移設し、2 経路で共有する）。
pub(crate) const GEMM_MSL_SRC: &str = include_str!("shaders/gemm.metal");

/// 生成ソースの先頭に必ず現れるマーカー行。`spec_source.rs` の単体テスト・
/// `tests/shader_source_evidence.rs` の証跡検査が参照する。
#[cfg(any(test, target_os = "macos"))]
pub(crate) const SPEC_HEADER_MARKER: &str = "#define GEMM_SPEC_ENABLED 1";

/// [`specialized_gemm_source`] へ渡す特殊化パラメータ。`gemm.metal` の
/// `#ifdef GEMM_SPEC_ENABLED` ブロック（12 個の `#define GEMM_SPEC_*`）と
/// 1:1 対応する。`crate::pipeline::GemmGateConstants`（function constant
/// 経路の 3 ゲート束ね）と同じ「`clippy::too_many_arguments` 回避のため
/// 構造体へ束ねる」設計判断（`.claude/rules/coding-rust.md`）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecializationParams {
    /// タイル構成（BM/BN/BK/WM/WN/staged）。`GEMM_SPEC_BM`〜`GEMM_SPEC_WN`・
    /// `GEMM_SPEC_USE_TGP_STAGING`・`GEMM_SPEC_TGP_PAD`（`cfg.pad()`）・
    /// `GEMM_SPEC_ACC_ROWS`/`GEMM_SPEC_ACC_COLS`（`cfg.acc_rows()`/
    /// `acc_cols()`）の由来。
    pub(crate) cfg: TileConfig,
    /// `GEMM_SPEC_SWIZZLE_ENABLED`。`crate::pipeline::GemmGateConstants::
    /// swizzle_enabled` と同じ意味（本番既定 `crate::tile::
    /// SWIZZLE_ENABLED`）。
    pub(crate) swizzle_enabled: bool,
    /// `GEMM_SPEC_FINE_BARRIER_ENABLED`。`GemmGateConstants::
    /// fine_barrier_enabled` と同じ意味。
    pub(crate) fine_barrier_enabled: bool,
    /// `GEMM_SPEC_UNROLL_ACC_ENABLED`。`GemmGateConstants::
    /// unroll_acc_enabled` と同じ意味（呼び出し元が候補ごとに
    /// `crate::tile::unroll_acc_loops_for` で導出した実効値を渡す契約は
    /// function constant 経路と同一）。
    pub(crate) unroll_acc_enabled: bool,
    /// `GEMM_SPEC_TRANS_A`。`crate::layout::TransposePattern` から導出する
    /// （呼び出し元が `TransposePattern::from_flags` 相当の分解を行う）。
    pub(crate) trans_a: bool,
    /// `GEMM_SPEC_TRANS_B`。
    pub(crate) trans_b: bool,
    /// `GEMM_SPEC_FRAG_LOAD_DEVICE_HOISTED`（イシュー #1293）。
    /// `crate::pipeline::GemmGateConstants::frag_load_device_hoisted` と
    /// 同じ意味。
    pub(crate) frag_load_device_hoisted: bool,
    /// `GEMM_SPEC_FRAG_LOAD_KSTEPS`（イシュー #1293。値は 1 または 2）。
    /// `crate::pipeline::GemmGateConstants::frag_load_ksteps` と同じ意味。
    pub(crate) frag_load_ksteps: u32,
    /// `GEMM_SPEC_TGP_PAD`（イシュー #1298）。従来は [`new`](Self::new) 内で
    /// `cfg.pad()` を直接埋め込んでいたが、`crate::pipeline::
    /// GemmGateConstants::tgp_pad_elems` と同じ「呼び出し元が実効値を渡す」
    /// 設計へ揃えた。[`new`](Self::new) は既定値として `cfg.pad()` を渡す
    /// （挙動は無変更）。
    pub(crate) tgp_pad_elems: u32,
    /// `GEMM_SPEC_COOP_LOAD_LAYOUT`（イシュー #1298）。
    /// `crate::pipeline::GemmGateConstants::coop_load_layout` と同じ意味。
    /// [`new`](Self::new) は本番既定 `0`（`crate::tile::CoopLoadLayout::
    /// RowLinear`）を渡す。
    pub(crate) coop_load_layout: u32,
}

impl SpecializationParams {
    /// `pattern` から `trans_a`/`trans_b` を導出して構築する。
    /// `crate::gemm::MetalGemm::pipeline_for_tile` が
    /// `TransposePattern`（呼び出し元から受け取る）を保持しているため、
    /// 変換の手間を本モジュール側に閉じる。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        cfg: TileConfig,
        swizzle_enabled: bool,
        fine_barrier_enabled: bool,
        unroll_acc_enabled: bool,
        frag_load_device_hoisted: bool,
        frag_load_ksteps: u32,
        pattern: TransposePattern,
    ) -> Self {
        let (trans_a, trans_b) = match pattern {
            TransposePattern::Nn => (false, false),
            TransposePattern::Nt => (false, true),
            TransposePattern::Tn => (true, false),
            TransposePattern::Tt => (true, true),
        };
        Self {
            cfg,
            swizzle_enabled,
            fine_barrier_enabled,
            unroll_acc_enabled,
            trans_a,
            trans_b,
            frag_load_device_hoisted,
            frag_load_ksteps,
            // イシュー #1298: 本コンストラクタは従来の 7 引数のまま据え置き
            // （既存呼び出し元を破壊しない）、協調ロード軸は本番既定値
            // （`cfg.pad()`／`0`＝`CoopLoadLayout::RowLinear`）で埋める。
            tgp_pad_elems: cfg.pad(),
            coop_load_layout: 0,
        }
    }
}

/// `true`/`false` を MSL の bool リテラルへ変換する（Rust の
/// `{}` フォーマットは既に `true`/`false` を出力するが、C++/MSL の
/// bool リテラルと同一表記であることを明示するための薄いヘルパ）。
fn msl_bool(b: bool) -> &'static str {
    if b { "true" } else { "false" }
}

/// `params` から候補固有の `#define GEMM_SPEC_*` ヘッダを組み立て、
/// [`GEMM_MSL_SRC`]（`gemm.metal` 全文）をそのまま連結して返す。
///
/// 生成される文字列は
/// `#define GEMM_SPEC_ENABLED 1\n#define GEMM_SPEC_BM <bm>\n...` の後に
/// `gemm.metal` 全文が続く形になる。`GEMM_SPEC_ENABLED` を定義した状態で
/// コンパイルすると、`shaders/gemm.metal` の `#ifdef GEMM_SPEC_ENABLED`
/// ブロックがリテラル定数側へ分岐し、`gemm_simdgroup_tiled` のアキュム
/// レータ配列が `ACC_ROWS_CAP`/`ACC_COLS_CAP`（候補の厳密サイズ）で
/// 確保される。
///
/// **A03（インジェクション）対策**: 埋め込む値は [`SpecializationParams`]
/// の `u32`/`bool` フィールドのみで、外部入力・環境変数・ファイル内容を
/// 一切含めない。数値は `u32` の `Display`、真偽値は [`msl_bool`] の
/// 固定リテラルでのみ出力するため、生成文字列に任意の文字列断片が
/// 混入する経路は存在しない（`.claude/rules/security.md` A03）。
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn specialized_gemm_source(params: &SpecializationParams) -> String {
    let cfg = params.cfg;
    let acc_rows = cfg.acc_rows();
    let acc_cols = cfg.acc_cols();
    let mut header = String::with_capacity(GEMM_MSL_SRC.len() + 512);
    header.push_str(SPEC_HEADER_MARKER);
    header.push('\n');
    header.push_str(&format!("#define GEMM_SPEC_BM {}\n", cfg.bm));
    header.push_str(&format!("#define GEMM_SPEC_BN {}\n", cfg.bn));
    header.push_str(&format!("#define GEMM_SPEC_BK {}\n", cfg.bk));
    header.push_str(&format!("#define GEMM_SPEC_WM {}\n", cfg.wm));
    header.push_str(&format!("#define GEMM_SPEC_WN {}\n", cfg.wn));
    header.push_str(&format!(
        "#define GEMM_SPEC_USE_TGP_STAGING {}\n",
        msl_bool(cfg.staged)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_TGP_PAD {}\n",
        params.tgp_pad_elems
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_SWIZZLE_ENABLED {}\n",
        msl_bool(params.swizzle_enabled)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_FINE_BARRIER_ENABLED {}\n",
        msl_bool(params.fine_barrier_enabled)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_TRANS_A {}\n",
        msl_bool(params.trans_a)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_TRANS_B {}\n",
        msl_bool(params.trans_b)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_UNROLL_ACC_ENABLED {}\n",
        msl_bool(params.unroll_acc_enabled)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_FRAG_LOAD_DEVICE_HOISTED {}\n",
        msl_bool(params.frag_load_device_hoisted)
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_FRAG_LOAD_KSTEPS {}\n",
        params.frag_load_ksteps
    ));
    header.push_str(&format!(
        "#define GEMM_SPEC_COOP_LOAD_LAYOUT {}\n",
        params.coop_load_layout
    ));
    header.push_str(&format!("#define GEMM_SPEC_ACC_ROWS {acc_rows}\n"));
    header.push_str(&format!("#define GEMM_SPEC_ACC_COLS {acc_cols}\n"));
    header.push_str(GEMM_MSL_SRC);
    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::CANDIDATES;

    /// 生成ヘッダの `GEMM_SPEC_*` が `TileConfig` 各フィールド・
    /// `pad()`・`acc_rows()`/`acc_cols()` と一致することを、全 `CANDIDATES`
    /// × 全 `TransposePattern` で確認する（Linux 実行可能。実機非依存）。
    #[test]
    fn generated_header_matches_config_fields_for_all_candidates() {
        for cfg in CANDIDATES.iter().copied() {
            for pattern in [
                TransposePattern::Nn,
                TransposePattern::Nt,
                TransposePattern::Tn,
                TransposePattern::Tt,
            ] {
                let params = SpecializationParams::new(cfg, false, false, false, false, 1, pattern);
                let src = specialized_gemm_source(&params);
                assert!(src.starts_with(SPEC_HEADER_MARKER));
                assert!(src.contains(&format!("#define GEMM_SPEC_BM {}\n", cfg.bm)));
                assert!(src.contains(&format!("#define GEMM_SPEC_BN {}\n", cfg.bn)));
                assert!(src.contains(&format!("#define GEMM_SPEC_BK {}\n", cfg.bk)));
                assert!(src.contains(&format!("#define GEMM_SPEC_WM {}\n", cfg.wm)));
                assert!(src.contains(&format!("#define GEMM_SPEC_WN {}\n", cfg.wn)));
                assert!(src.contains(&format!(
                    "#define GEMM_SPEC_USE_TGP_STAGING {}\n",
                    msl_bool(cfg.staged)
                )));
                assert!(src.contains(&format!("#define GEMM_SPEC_TGP_PAD {}\n", cfg.pad())));
                assert!(src.contains("#define GEMM_SPEC_FRAG_LOAD_DEVICE_HOISTED false\n"));
                assert!(src.contains("#define GEMM_SPEC_FRAG_LOAD_KSTEPS 1\n"));
                assert!(src.contains("#define GEMM_SPEC_COOP_LOAD_LAYOUT 0\n"));
                assert!(src.contains(&format!("#define GEMM_SPEC_ACC_ROWS {}\n", cfg.acc_rows())));
                assert!(src.contains(&format!("#define GEMM_SPEC_ACC_COLS {}\n", cfg.acc_cols())));
            }
        }
    }

    /// 生成文字列は必ず `GEMM_MSL_SRC`（`gemm.metal` 全文）で終わる
    /// （ヘッダを前置するだけで本文を一切改変しない契約）。
    #[test]
    fn generated_source_ends_with_original_gemm_metal_source() {
        let cfg = CANDIDATES[0];
        let params =
            SpecializationParams::new(cfg, false, false, false, false, 1, TransposePattern::Nn);
        let src = specialized_gemm_source(&params);
        assert!(src.ends_with(GEMM_MSL_SRC));
    }

    /// 候補・パターン・ゲート値が異なれば生成文字列も相異なることを
    /// 確認する（`crate::gemm::MetalGemm` のキャッシュキー粒度
    /// `(TileConfig, TransposePattern)` が実際にソース差分へ反映される
    /// ことの裏付け）。
    #[test]
    fn distinct_inputs_produce_distinct_sources() {
        let a = specialized_gemm_source(&SpecializationParams::new(
            CANDIDATES[0],
            false,
            false,
            false,
            false,
            1,
            TransposePattern::Nn,
        ));
        let b = specialized_gemm_source(&SpecializationParams::new(
            CANDIDATES[1],
            false,
            false,
            false,
            false,
            1,
            TransposePattern::Nn,
        ));
        let c = specialized_gemm_source(&SpecializationParams::new(
            CANDIDATES[0],
            false,
            false,
            false,
            false,
            1,
            TransposePattern::Nt,
        ));
        let d = specialized_gemm_source(&SpecializationParams::new(
            CANDIDATES[0],
            false,
            false,
            true,
            false,
            1,
            TransposePattern::Nn,
        ));
        let e = specialized_gemm_source(&SpecializationParams::new(
            CANDIDATES[0],
            false,
            false,
            false,
            true,
            2,
            TransposePattern::Nn,
        ));
        assert_ne!(a, b, "候補が異なれば生成文字列も異なるべき");
        assert_ne!(a, c, "パターンが異なれば生成文字列も異なるべき");
        assert_ne!(a, d, "ゲートが異なれば生成文字列も異なるべき");
        assert_ne!(
            a, e,
            "フラグメントロード方式候補ゲート（イシュー #1293）が異なれば生成文字列も異なるべき"
        );
    }

    /// bool フィールドが `1`/`0` ではなく MSL の `true`/`false` リテラルで
    /// 出力されることを固定する（A03 対策コメントの「固定リテラルのみ」
    /// 契約の機械検証）。
    #[test]
    fn bool_fields_render_as_true_false_literals() {
        let params = SpecializationParams::new(
            CANDIDATES[3],
            true,
            true,
            true,
            true,
            2,
            TransposePattern::Tt,
        );
        let src = specialized_gemm_source(&params);
        assert!(src.contains("#define GEMM_SPEC_SWIZZLE_ENABLED true\n"));
        assert!(src.contains("#define GEMM_SPEC_FINE_BARRIER_ENABLED true\n"));
        assert!(src.contains("#define GEMM_SPEC_UNROLL_ACC_ENABLED true\n"));
        assert!(src.contains("#define GEMM_SPEC_TRANS_A true\n"));
        assert!(src.contains("#define GEMM_SPEC_TRANS_B true\n"));
        assert!(src.contains("#define GEMM_SPEC_FRAG_LOAD_DEVICE_HOISTED true\n"));
        assert!(src.contains("#define GEMM_SPEC_FRAG_LOAD_KSTEPS 2\n"));
    }

    /// イシュー #1298: 協調ロード軸（`tgp_pad_elems`/`coop_load_layout`）を
    /// `new()` の既定値から変更した場合、生成される `#define` が実際に
    /// 追従することを固定する（struct update 構文で直接フィールドを
    /// 上書きして構築。`new()` 自体は 7 引数のまま不変）。
    #[test]
    fn coop_load_axis_overrides_are_reflected_in_generated_defines() {
        let base = SpecializationParams::new(
            CANDIDATES[3],
            false,
            false,
            false,
            false,
            1,
            TransposePattern::Nn,
        );
        let overridden = SpecializationParams {
            tgp_pad_elems: 8,
            coop_load_layout: 1,
            ..base
        };
        let src = specialized_gemm_source(&overridden);
        assert!(src.contains("#define GEMM_SPEC_TGP_PAD 8\n"));
        assert!(src.contains("#define GEMM_SPEC_COOP_LOAD_LAYOUT 1\n"));
        assert_ne!(
            specialized_gemm_source(&base),
            src,
            "pad/layout 差分が生成文字列へ反映されていない"
        );
    }
}
