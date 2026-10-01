//! sm_121 ISA プローブ（イシュー #2122）のレジストリ・参照モデルの整合検査。
//! GPU・NVRTC を使わないため CI（GitHub ホステッド）で走る（非 `#[ignore]`）。
//!
//! 検査対象:
//! - レジストリの内部整合（ID 一意・シンボル・ASCII・`#include` 不使用・opcode・
//!   方針と検証方法の整合・期待値が sentinel と衝突しない）
//! - 参照モデル（fragment のレイアウトが全単射で、`D = A*B + C` を再現する）
//! - JSONL のエスケープ
//! - **RULE.txt との突き合わせ**（`PROBE:`・`CLAUSE:`・`STAGE:`・`STATUS:`・
//!   `TARGET:` 行がレジストリと一致）と `guide_claims.tsv` の測定項目の存在
//! - `Cargo.toml` の `[[test]] required-features` 指定
//!
//! RULE.txt などはワークスペースの `docs/` から `CARGO_MANIFEST_DIR/../..` 起点で
//! 読む。パッケージ化した crate 単体（`cargo package` の tarball）からは実行できないが、
//! 本テストは `required-features = ["internal-diagnostics"]` の開発・CI 専用テストで
//! あり配布対象ではない。ファイルが無い場合は失敗する（黙ってスキップしない）。

#[path = "sm121_isa_probe_common/mod.rs"]
mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use common::jsonl::{self, Record};
use common::kernels_arch::MACRO_CANDIDATES;
use common::model;
use common::registry::{self, CLAUSES, Check, attr_table, probes};
use common::runner::SENTINEL;
use common::types::{
    ALL_STAGES, ALL_STATUSES, Kind, Layout, Policy, REAL_ARCH_WHITELIST, TARGETS,
    VIRT_ARCH_WHITELIST,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_repo(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} を読めない（必須ファイル）: {e}", path.display()))
}

const RULE: &str = "docs/perf/logs/sm121-isa-probe-2122/RULE.txt";
const CLAIMS: &str = "docs/perf/logs/sm121-isa-probe-2122/guide_claims.tsv";

/// RULE.txt の `KEY: value` 行を（key → 行リスト）で返す。
fn rule_lines(prefix: &str) -> Vec<String> {
    read_repo(RULE)
        .lines()
        .filter(|l| l.starts_with(prefix))
        .map(|l| l.trim_end().to_string())
        .collect()
}

fn assert_unique(items: &[String], what: &str) {
    let set: BTreeSet<&String> = items.iter().collect();
    assert_eq!(set.len(), items.len(), "{what} に重複がある: {items:?}");
}

// ---------------------------------------------------------------- レジストリ

#[test]
fn ids_are_unique_and_symbols_follow_the_id() {
    let all = probes();
    let ids: Vec<String> = all.iter().map(|p| p.id.to_string()).collect();
    assert_unique(&ids, "プローブ ID");
    for p in &all {
        if p.kind == Kind::Kernel {
            // 共有ソース（同じカーネルを別 spec・別起動で使う）の例外は明示列挙する。
            let want = match p.id {
                "tc5.cross" => "tc5_alloc".to_string(),
                "tma.base_cta" | "tma.coord" | "tma.oob_none" | "tma.oob_nan" | "tma.oob_neg"
                | "tma.swz32" | "tma.swz64" | "tma.swz128" => "tma_load_cta".to_string(),
                "tma.base_cluster" => "tma_load_cluster".to_string(),
                "tma.store" => "tma_store_cta".to_string(),
                id => id.replace('.', "_"),
            };
            assert_eq!(p.symbol, want, "{}: シンボル名が ID と対応しない", p.id);
            assert!(
                p.src.contains("extern \"C\" __global__")
                    && p.src.contains(&format!(" {}(", p.symbol)),
                "{}: extern \"C\" のシンボル {} がソースに無い",
                p.id,
                p.symbol
            );
        } else {
            assert!(
                p.src.is_empty() && p.symbol.is_empty(),
                "{}: attr はソースを持たない",
                p.id
            );
        }
    }
}

/// 全プローブに、そのソースが含むべき opcode のトークンを事前登録する（新しい
/// プローブを足したらここへも足す。足し忘れは失敗する）。
fn opcode_token(id: &str) -> &'static str {
    match id {
        "ctl.copy" => "ST(i, LD(i))",
        "macro.arch" => "__CUDA_ARCH__",
        "tc5.alloc" | "tc5.cross" => "tcgen05.alloc",
        "tc5.ld" => "tcgen05.ld",
        "mma.tf32.m16n8k8" => "m16n8k8.row.col.f32.tf32",
        "mma.tf32.m16n8k4" => "m16n8k4.row.col.f32.tf32",
        "mma.f16.m16n8k16.f32" => "m16n8k16.row.col.f32.f16",
        "mma.f16.m16n8k8.f32" => "m16n8k8.row.col.f32.f16",
        "mma.f16.m16n8k16.f16" => "m16n8k16.row.col.f16.f16",
        "mma.bf16.m16n8k16.f32" => "m16n8k16.row.col.f32.bf16",
        "mma.bf16.m16n8k8.f32" => "m16n8k8.row.col.f32.bf16",
        "mma.f64.m8n8k4" => "m8n8k4.row.col.f64",
        "mma.f16.m8n8k4" => "m8n8k4.row.col.f32.f16",
        "mma.f64.m16n8k4" => "m16n8k4.row.col.f64",
        "mma.f64.m16n8k8" => "m16n8k8.row.col.f64",
        "mma.f64.m16n8k16" => "m16n8k16.row.col.f64",
        "mma.s8.m16n8k32" => "m16n8k32.row.col.s32.s8",
        "mma.e4m3.m16n8k32" => "m16n8k32.row.col.f32.e4m3",
        "mma.e5m2.m16n8k32" => "m16n8k32.row.col.f32.e5m2",
        "mma.f8f6f4.m16n8k32" => "kind::f8f6f4",
        "mma.block_scale.m16n8k64" => "block_scale",
        "mma.ldmatrix.x1" => "ldmatrix.sync.aligned.m8n8.x1",
        "mma.ldmatrix.x2" => "ldmatrix.sync.aligned.m8n8.x2",
        "mma.ldmatrix.x4" => "ldmatrix.sync.aligned.m8n8.x4.shared",
        "mma.ldmatrix.x4_trans" => "x4.trans",
        "mma.stmatrix.x4" => "stmatrix.sync.aligned.m8n8.x4",
        "simt.fma_f32" => "fma.rn.f32",
        "simt.fma_f64" => "fma.rn.f64",
        "simt.fma_f16x2" => "fma.rn.f16x2",
        "simt.fma_bf16x2" => "fma.rn.bf16x2",
        "simt.f32x2_add" => "add.rn.f32x2",
        "simt.f32x2_mul" => "mul.rn.f32x2",
        "simt.f32x2_fma" => "fma.rn.f32x2",
        "simt.cvt_tf32" => "cvt.rna.tf32.f32",
        "simt.elect_sync" => "elect.sync",
        "simt.redux_u32" => "redux.sync.add.u32",
        "simt.redux_f32" => "redux.sync.max.f32",
        "wgmma.m64n8k16" => "wgmma.mma_async",
        "hop.griddepcontrol" => "griddepcontrol",
        "hop.fence_proxy_async" => "fence.proxy.async",
        "snr.dec" => "setmaxnreg.dec",
        "snr.incdec" => "setmaxnreg.inc",
        "clu.dims1" | "clu.dims2" | "clu.dims4" | "clu.dims8" | "clu.dims16" => "__cluster_dims__",
        "clu.dsmem" => "mapa.shared::cluster",
        "clu.rt2" | "clu.rt4" => "%cluster_ctarank",
        "ctl.raw" => "ST(i, LD(i))",
        "ctl.rawmap" => "tm.opaque",
        "tma.base_cta" | "tma.coord" | "tma.oob_none" | "tma.oob_nan" | "tma.oob_neg"
        | "tma.swz32" | "tma.swz64" | "tma.swz128" => "cp.async.bulk.tensor.2d.shared::cta.global",
        "tma.base_cluster" => "cp.async.bulk.tensor.2d.shared::cluster.global",
        "tma.store" => "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group",
        "tma.prefetch" => "cp.async.bulk.prefetch.tensor.2d.L2",
        "tma.multicast" => ".multicast::cluster",
        "tma.bulk_cta" => "cp.async.bulk.shared::cta.global",
        "tma.bulk_cluster" => "cp.async.bulk.shared::cluster.global",
        "attr.limits" | "attr.cluster" | "attr.misc" => "",
        other => panic!("{other}: opcode トークンが未登録（registry テストへ追加すること）"),
    }
}

#[test]
fn sources_are_ascii_without_include_and_contain_their_opcode() {
    for p in probes() {
        let token = opcode_token(p.id);
        if p.kind == Kind::Attr {
            continue;
        }
        assert!(p.src.is_ascii(), "{}: ソースに非 ASCII がある", p.id);
        assert!(
            !p.src.contains("#include"),
            "{}: #include を使わない契約",
            p.id
        );
        // マクロ連結の取り違え（raw 文字列の区切りの誤り等）でソースへ Rust の記法が混入していない。
        for bad in ["$space", "$sym", "r#\"", "\"#"] {
            assert!(
                !p.src.contains(bad),
                "{}: ソースに連結の取り違え {bad:?} が混入",
                p.id
            );
        }
        assert_eq!(
            p.src.matches('{').count(),
            p.src.matches('}').count(),
            "{}: 波括弧の数が合わない",
            p.id
        );
        assert!(
            p.src.contains(token),
            "{}: opcode トークン {token:?} がソースに無い",
            p.id
        );
        // 境界チェック（REQ-8）: ストアは ST、ロードは LD マクロ経由。生の `out[`／`in[`
        // の添字アクセスはマクロ定義内（`#define`）以外に書かない。
        for line in p.src.lines().filter(|l| !l.starts_with("#define")) {
            assert!(
                !has_raw_index(line, "out") && !has_raw_index(line, "in"),
                "{}: 境界チェックのない生アクセス: {line}",
                p.id
            );
        }
    }
}

/// `name[`（直前が識別子文字でない）の生の添字アクセスを検出する。空白を挟む `in [`、
/// `(in[`・`*in[`・`=in[` のような書き方も捕まえる（`n_in[` のような別識別子は対象外）。
fn has_raw_index(line: &str, name: &str) -> bool {
    let mut norm = String::with_capacity(line.len());
    for ch in line.chars() {
        if ch == '[' {
            while norm.ends_with(' ') || norm.ends_with('\t') {
                norm.pop();
            }
        }
        norm.push(ch);
    }
    let needle = format!("{name}[");
    let mut start = 0;
    while let Some(pos) = norm[start..].find(&needle) {
        let at = start + pos;
        let prev = norm[..at].chars().next_back();
        if !prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            return true;
        }
        start = at + needle.len();
    }
    false
}

#[test]
fn raw_index_detector_catches_each_spelling() {
    for bad in [
        "x = in[i];",
        "f(in[i])",
        "*(in[0])",
        "a=out[k]",
        "out [k] = 1;",
        "(out[0])",
        "in\t[2]",
    ] {
        let name = if bad.contains("out") { "out" } else { "in" };
        assert!(has_raw_index(bad, name), "検出できない: {bad}");
    }
    for ok in [
        "n_in[0]",
        "my_out[1]",
        "ST(i, v)",
        "LD(i)",
        "int n_in, unsigned* out, int n",
    ] {
        assert!(
            !has_raw_index(ok, "in") && !has_raw_index(ok, "out"),
            "誤検出: {ok}"
        );
    }
}

/// ソース中の `ST(`／`ST_F64(` の第 1 引数が lane 添字の `l`・`l * A`・`l * A + B`（`u` 接尾辞可）
/// の形だけでできているとき、`(stride A, 書く word 集合)` を返す。`ST_F64` は B と B+1 の 2 語。
/// 他の形（ループ添字・定数添字など）が 1 つでもあれば `None`（検査対象外）。
fn lane_strided_stores(src: &str) -> Option<(u32, BTreeSet<u32>)> {
    let mut stride: Option<u32> = None;
    let mut words = BTreeSet::new();
    let mut seen = false;
    for line in src.lines().filter(|l| !l.starts_with("#define")) {
        let mut rest = line;
        while let Some(pos) = rest.find("ST") {
            let tail = &rest[pos..];
            let (width, after) = if let Some(a) = tail.strip_prefix("ST_F64(") {
                (2, a)
            } else if let Some(a) = tail.strip_prefix("ST(") {
                (1, a)
            } else {
                rest = &rest[pos + 2..];
                continue;
            };
            let prev = rest[..pos].chars().next_back();
            if prev.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                rest = &rest[pos + 2..];
                continue;
            }
            let arg: String = after
                .split(',')
                .next()?
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            let num = |s: &str| s.trim_end_matches('u').parse::<u32>().ok();
            let (a, b) = if arg == "l" {
                (1, 0)
            } else {
                let r = arg.strip_prefix("l*")?;
                match r.split_once('+') {
                    Some((a, b)) => (num(a)?, num(b)?),
                    None => (num(r)?, 0),
                }
            };
            if stride.is_some_and(|s| s != a) {
                return None;
            }
            stride = Some(a);
            seen = true;
            words.extend((0..width).map(|w| b + w));
            rest = after;
        }
    }
    seen.then(|| (stride.unwrap_or(1), words))
}

/// 宣言した出力 word 数と、カーネルが書く word 数の静的な突き合わせ。lane 添字の
/// `l * stride + offset` 形のストアだけでできているカーネルについて、書く word の集合が
/// `0..stride` ちょうどで、`out_words == 32 * stride` であることを要求する（上位語と下位語の
/// 取り違えや、結果レジスタの書き漏らしを検出する。ループ添字など他の形のカーネルは対象外で、
/// 値の正しさそのものは S6 のビット一致が担う）。
#[test]
fn declared_out_words_match_the_words_each_kernel_stores() {
    let mut checked = 0;
    for p in probes()
        .iter()
        .filter(|p| p.kind == Kind::Kernel && p.block == 32 && p.cluster == 0)
    {
        let Some((stride, words)) = lane_strided_stores(p.src) else {
            continue;
        };
        let want: BTreeSet<u32> = (0..stride).collect();
        assert_eq!(
            words, want,
            "{}: lane あたりの書き込み word が 0..{stride} と一致しない",
            p.id
        );
        assert_eq!(
            p.out_words,
            32 * stride as usize,
            "{}: out_words が 32 lane x {stride} と不一致",
            p.id
        );
        checked += 1;
    }
    assert!(
        checked >= 30,
        "静的検査の対象が少なすぎる（{checked} 件）。解析が空振りしている疑い"
    );
}

#[test]
fn store_layout_analysis_detects_the_f64_half_word_bug() {
    // Bugbot 指摘の型: stride 8 なのに d0/d2 の下位と d1/d3 の上位だけを書く（4 語）。
    let buggy = "ST(l * 8u, a); ST(l * 8u + 1u, b);\nST(l * 8u + 2u, c); ST(l * 8u + 3u, d);";
    let (stride, words) = lane_strided_stores(buggy).expect("解析できる");
    assert_eq!(stride, 8);
    assert_ne!(words, (0..8).collect::<BTreeSet<u32>>());
    let fixed = "ST_F64(l * 8u, a); ST_F64(l * 8u + 2u, b); ST_F64(l * 8u + 4u, c); ST_F64(l * 8u + 6u, d);";
    let (stride, words) = lane_strided_stores(fixed).expect("解析できる");
    assert_eq!((stride, words), (8, (0..8).collect::<BTreeSet<u32>>()));
    assert!(lane_strided_stores("ST(i, sm[i]);").is_none());
}

#[test]
fn symbol_table_covers_every_probe_and_rejects_unknown_ids() {
    assert!(registry::symbol_of("no.such.probe").is_none());
    for p in probes() {
        assert!(registry::symbol_of(p.id).is_some(), "{}", p.id);
    }
}

#[test]
fn policy_and_check_are_consistent() {
    for p in probes() {
        match p.policy {
            Policy::Verify => {
                assert!(
                    !matches!(p.check, Check::Skip),
                    "{}: verify は検証方法が要る",
                    p.id
                );
                assert!(p.out_words > 0, "{}", p.id);
            }
            Policy::RecordOnly => {
                assert!(matches!(p.check, Check::Custom(_)), "{}", p.id);
            }
            Policy::AcceptOnly => {
                assert!(
                    matches!(p.check, Check::Skip),
                    "{}: accept_only は検証しない",
                    p.id
                );
            }
            Policy::Attr => {
                assert_eq!(p.kind, Kind::Attr, "{}", p.id);
                assert!(!attr_table(p.id).is_empty(), "{}: 属性表が空", p.id);
            }
        }
        if p.layout == Layout::Verified {
            assert!(
                p.runs_on_sm86(),
                "{}: layout=verified は sm_86 で実行できる形状のみ",
                p.id
            );
        }
        if p.layout == Layout::Unverified {
            assert_eq!(p.policy, Policy::Verify, "{}", p.id);
            assert!(
                !p.runs_on_sm86(),
                "{}: sm_86 で実行できるものを未検証にしない",
                p.id
            );
        }
        assert!(
            CLAUSES.contains(&p.clause),
            "{}: 未知の条項 {}",
            p.id,
            p.clause
        );
        assert!(
            REAL_ARCH_WHITELIST.contains(&p.home_arch),
            "{}: home がホワイトリスト外",
            p.id
        );
        if let Some((virt, real)) = p.fixed_arch {
            assert!(VIRT_ARCH_WHITELIST.contains(&virt) && REAL_ARCH_WHITELIST.contains(&real));
        }
        if p.cluster > 0 {
            assert_eq!(p.grid % p.cluster, 0, "{}: grid は cluster の倍数", p.id);
            assert!(
                p.src
                    .contains(&format!("__cluster_dims__({},1,1)", p.cluster)),
                "{}: cluster 次元とソースが不一致",
                p.id
            );
        }
    }
}

#[test]
fn expected_outputs_have_the_declared_length_and_never_collide_with_the_sentinel() {
    for p in probes() {
        let Check::Exact(f) = p.check else { continue };
        let input = (p.make_input)();
        let expected = f(&input);
        assert_eq!(
            expected.len(),
            p.out_words,
            "{}: 期待値の長さが out_words と違う",
            p.id
        );
        assert!(
            !expected.contains(&SENTINEL),
            "{}: 期待値が sentinel と衝突",
            p.id
        );
        // 自己一致（比較関数の健全性）。
        assert!(matches!(
            registry::compare_words(&expected, &expected),
            registry::Outcome::Match(_)
        ));
    }
}

#[test]
fn compare_words_detects_each_kind_of_difference() {
    use registry::{Outcome, compare_words};
    assert!(matches!(compare_words(&[1, 2], &[1, 2]), Outcome::Match(_)));
    match compare_words(&[1, 2, 3], &[1, 9, 9]) {
        Outcome::Mismatch { first, count, .. } => assert_eq!((first, count), (1, 2)),
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        compare_words(&[1, 2], &[1]),
        Outcome::Mismatch { .. }
    ));
    assert!(matches!(
        compare_words(&[1], &[1, 2]),
        Outcome::Mismatch { .. }
    ));
}

#[test]
fn macro_candidates_appear_in_the_kernel_source() {
    let p = registry::probe_by_id("macro.arch").expect("macro.arch");
    for name in MACRO_CANDIDATES {
        assert!(p.src.contains(name), "{name} がカーネルソースに無い");
    }
    assert_eq!(p.out_words, MACRO_CANDIDATES.len());
}

#[test]
fn targets_and_arch_whitelists_are_well_formed() {
    let names: Vec<String> = TARGETS.iter().map(|t| t.name.to_string()).collect();
    assert_unique(&names, "target 名");
    for t in TARGETS {
        assert!(VIRT_ARCH_WHITELIST.contains(&t.virt), "{}", t.name);
        assert!(t.real.starts_with("sm_") && t.virt.starts_with("compute_"));
    }
    assert!(TARGETS.iter().filter(|t| t.official).count() == 3);
}

// ---------------------------------------------------------------- 参照モデル

fn covers_exactly_once(coords: impl Iterator<Item = (usize, usize)>, rows: usize, cols: usize) {
    let mut seen = BTreeMap::new();
    for c in coords {
        *seen.entry(c).or_insert(0usize) += 1;
    }
    assert_eq!(seen.len(), rows * cols, "要素の被覆数が違う");
    assert!(seen.values().all(|&n| n == 1), "重複被覆がある");
    assert!(seen.keys().all(|&(r, c)| r < rows && c < cols), "範囲外");
}

#[test]
fn fragment_layouts_are_bijections() {
    // f16/bf16 k16: A 16x16、B 16x8、C/D 16x8。
    covers_exactly_once(
        (0..32).flat_map(|l| (0..4).flat_map(move |i| model::f16_a_coords(l, i))),
        16,
        16,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..2).flat_map(move |j| model::f16_b_coords(l, j))),
        16,
        8,
    );
    // k8: A 16x8、B 8x8。
    covers_exactly_once(
        (0..32).flat_map(|l| (0..2).flat_map(move |i| model::f16_a_coords(l, i))),
        16,
        8,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..1).flat_map(move |j| model::f16_b_coords(l, j))),
        8,
        8,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..4).map(move |i| model::acc32_coords(l, i))),
        16,
        8,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..2).flat_map(move |i| model::acc16_coords(l, i))),
        16,
        8,
    );
    // tf32 k8: A 16x8（4 reg）、B 8x8（2 reg）。k4: A 16x4（2 reg）、B 4x8（1 reg）。
    covers_exactly_once(
        (0..32).flat_map(|l| (0..4).map(move |i| model::tf32_a_coords(l, i))),
        16,
        8,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..2).map(move |j| model::tf32_b_coords(l, j))),
        8,
        8,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..2).map(move |i| model::tf32_a_coords(l, i))),
        16,
        4,
    );
    covers_exactly_once(
        (0..32).flat_map(|l| (0..1).map(move |j| model::tf32_b_coords(l, j))),
        4,
        8,
    );
}

/// モデルの入力 fragment を逆引きして密行列 `A*B + C` を再計算し、期待出力と
/// 突き合わせる（レイアウトの取り違えが入力側と期待値側で同時に起きても、
/// 密行列の積とは一致しないことで検出する）。
#[test]
fn f16_family_expected_equals_dense_product_and_input_roundtrips() {
    for (k, bf, acc16) in [
        (16, false, false),
        (8, false, false),
        (16, false, true),
        (16, true, false),
        (8, true, false),
    ] {
        let input = model::f16_family_input(k, bf, acc16);
        let stride = k / 4 + k / 8 + if acc16 { 2 } else { 4 };
        assert_eq!(input.len(), 32 * stride, "k={k}");
        let expected = model::f16_family_expected(k, acc16);
        // 期待出力の各語を座標へ戻し、密行列の値と一致することを確認。
        let per_lane = if acc16 { 2 } else { 4 };
        assert_eq!(expected.len(), 32 * per_lane);
        for l in 0..32 {
            for i in 0..per_lane {
                let w = expected[l * per_lane + i];
                if acc16 {
                    let [(r0, n0), (r1, n1)] = model::acc16_coords(l, i);
                    let lo = half::f16::from_bits(w as u16).to_f32();
                    let hi = half::f16::from_bits((w >> 16) as u16).to_f32();
                    assert_eq!(lo, model::dense_d(r0, n0, k) as f32);
                    assert_eq!(hi, model::dense_d(r1, n1, k) as f32);
                } else {
                    let (r, n) = model::acc32_coords(l, i);
                    assert_eq!(f32::from_bits(w), model::dense_d(r, n, k) as f32);
                }
            }
        }
        // A・B fragment の語を座標へ戻して元の行列値と一致（pack/unpack の往復）。
        for l in 0..32 {
            let base = l * stride;
            for i in 0..k / 4 {
                let w = input[base + i];
                let [(r0, c0), (r1, c1)] = model::f16_a_coords(l, i);
                let dec = |bits: u16| {
                    if bf {
                        half::bf16::from_bits(bits).to_f32()
                    } else {
                        half::f16::from_bits(bits).to_f32()
                    }
                };
                assert_eq!(dec(w as u16), model::a_val(r0, c0) as f32);
                assert_eq!(dec((w >> 16) as u16), model::a_val(r1, c1) as f32);
            }
        }
    }
}

#[test]
fn tf32_expected_equals_dense_product() {
    for k in [8, 4] {
        let expected = model::tf32_expected(k);
        for l in 0..32 {
            for i in 0..4 {
                let (r, n) = model::acc32_coords(l, i);
                assert_eq!(
                    f32::from_bits(expected[l * 4 + i]),
                    model::dense_d(r, n, k) as f32
                );
            }
        }
        let input = model::tf32_input(k);
        let stride = k / 2 + k / 4 + 4;
        assert_eq!(input.len(), 32 * stride);
    }
}

#[test]
fn ldmatrix_and_stmatrix_models_are_inverse_of_each_other() {
    // stmatrix の期待 smem 画像を ldmatrix（非 trans）の入力画像として読むと、
    // 各 lane は元のレジスタ値を取り戻す。
    let regs = model::stmatrix_input();
    let image = model::stmatrix_expected(&regs);
    assert_eq!(image.len(), 128);
    for l in 0..32 {
        let (g, t) = (l / 4, l % 4);
        for m in 0..4 {
            assert_eq!(
                image[m * 32 + g * 4 + t],
                regs[l * 4 + m],
                "lane {l} reg {m}"
            );
        }
    }
    // ldmatrix の smem 入力は半語が全て異なり 0 を含まない。
    let smem = model::ldmatrix_smem_input();
    let mut halves: Vec<u16> = smem
        .iter()
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    assert!(!halves.contains(&0));
    halves.sort_unstable();
    halves.dedup();
    assert_eq!(halves.len(), 256);
}

#[test]
fn simt_models_match_known_values() {
    // cvt.rna.tf32: 1.0 + (下位 13 bit が 0x1000 以上) は切り上げ、未満は切り捨て。
    let up = model::cvt_tf32_expected(&[0x3F80_1000, 0x3F80_0FFF]);
    assert_eq!(up, vec![0x3F80_2000, 0x3F80_0000]);
    let input = model::cvt_tf32_input();
    assert!(
        input
            .iter()
            .all(|&b| (0x3F80_0000..0x4000_0000).contains(&b))
    );
    // redux: 全 lane 同値の [add, min, max, or]。
    let rin = model::redux_u32_input();
    let r = model::redux_u32_expected(&rin);
    assert_eq!(r.len(), 128);
    assert_eq!(r[0..4], r[4..8]);
    // fma（f32）は融合積和。丸め 2 回の `a*b+c` と一致しない入力が少なくとも 1 つある。
    let fin = model::fma_f32_input();
    let fused = model::fma_f32_expected(&fin);
    let unfused: Vec<u32> = fin
        .chunks(3)
        .map(|c| (f32::from_bits(c[0]) * f32::from_bits(c[1]) + f32::from_bits(c[2])).to_bits())
        .collect();
    assert_ne!(fused, unfused, "FMA 契約を検出できない入力になっている");
}

// ---------------------------------------------------------------- JSONL

#[test]
fn json_escape_covers_quotes_backslashes_controls_and_non_ascii() {
    assert_eq!(jsonl::escape("a\"b\\c"), "a\\\"b\\\\c");
    assert_eq!(jsonl::escape("l1\nl2\r\t"), "l1\\nl2\\r\\t");
    assert_eq!(jsonl::escape("\u{1}\u{1f}\u{7f}"), "\\u0001\\u001f\\u007f");
    assert_eq!(jsonl::escape("あ"), "\\u3042");
    assert_eq!(jsonl::escape("\u{1F600}"), "\\ud83d\\ude00");
    assert!(jsonl::escape("ptxas error: 'x' \"y\"\n\u{0}").is_ascii());
    let rec = Record::new().str("k", "v\"").int("n", 3).boolean("b", true);
    assert_eq!(rec.to_json(), "{\"k\":\"v\\\"\",\"n\":3,\"b\":true}");
}

// ---------------------------------------------------------------- RULE.txt との突き合わせ

#[test]
fn rule_probe_lines_match_the_registry_exactly() {
    let want: Vec<String> = probes().iter().map(|p| p.rule_line()).collect();
    let got = rule_lines("PROBE: ");
    assert_unique(&got, "RULE.txt の PROBE 行");
    let want_set: BTreeSet<&String> = want.iter().collect();
    let got_set: BTreeSet<&String> = got.iter().collect();
    assert_eq!(
        got_set, want_set,
        "RULE.txt の PROBE 行とレジストリが一致しない"
    );
}

#[test]
fn rule_clause_stage_status_and_target_lines_match() {
    let clauses = rule_lines("CLAUSE: ");
    assert_unique(&clauses, "CLAUSE 行");
    let want: BTreeSet<String> = CLAUSES.iter().map(|c| format!("CLAUSE: {c}")).collect();
    assert_eq!(clauses.iter().cloned().collect::<BTreeSet<_>>(), want);

    let stages = rule_lines("STAGE: ");
    let want: Vec<String> = ALL_STAGES
        .iter()
        .map(|s| format!("STAGE: {}", s.name()))
        .collect();
    assert_eq!(stages, want, "STAGE 行（順序を含む）");

    let statuses = rule_lines("STATUS: ");
    let want: Vec<String> = ALL_STATUSES
        .iter()
        .map(|s| format!("STATUS: {}", s.as_str()))
        .collect();
    assert_eq!(statuses, want);

    let official: Vec<String> = TARGETS
        .iter()
        .filter(|t| t.official)
        .map(|t| format!("TARGET: {} {}", t.name, t.real))
        .collect();
    assert_eq!(rule_lines("TARGET: "), official);
    let dev: Vec<String> = TARGETS
        .iter()
        .filter(|t| !t.official)
        .map(|t| format!("DEVTARGET: {} {}", t.name, t.real))
        .collect();
    assert_eq!(rule_lines("DEVTARGET: "), dev);
}

#[test]
fn rule_process_lines_carry_timeouts_and_fixed_arch_matches_the_registry() {
    // PROCESS 行は env_info 以外すべて正の `timeout=<秒>` を持つ（orchestrate.sh はそこから読む）。
    let lines = rule_lines("PROCESS: ");
    assert!(lines.iter().any(|l| l == "PROCESS: env_info"));
    assert!(lines.iter().any(|l| l.starts_with("PROCESS: exec_matrix ")));
    assert_eq!(
        lines
            .iter()
            .filter(|l| l.starts_with("PROCESS: legacy "))
            .count(),
        7
    );
    for l in lines.iter().filter(|l| *l != "PROCESS: env_info") {
        let secs = l.rsplit_once(" timeout=").map(|(_, n)| n);
        assert!(
            secs.is_some_and(|n| n.parse::<u64>().is_ok_and(|v| v > 0)),
            "PROCESS 行に正の timeout= が無い: {l}"
        );
    }
    // orchestrate.sh は秒数の定数を持たない（RULE.txt の単一ソース）。
    let sh = read_repo("docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh");
    for needle in [
        "EXEC_TIMEOUT=",
        "COMPILE_TIMEOUT=",
        "LEGACY_TIMEOUT=",
        "DUMP_TIMEOUT=",
    ] {
        assert!(
            !sh.contains(needle),
            "orchestrate.sh に timeout 定数 {needle} が残っている"
        );
    }
    // 固定アーキ（tc5.cross）は FIXEDARCH 行と一致する。
    let want: Vec<String> = probes()
        .iter()
        .filter_map(|p| {
            p.fixed_arch
                .map(|(v, r)| format!("FIXEDARCH: {} {v} {r}", p.id))
        })
        .collect();
    assert_eq!(rule_lines("FIXEDARCH: "), want);
}

#[test]
fn guide_claims_reference_existing_probes_and_attributes() {
    let text = read_repo(CLAIMS);
    let mut rows = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 6, "guide_claims.tsv は 6 列: {line}");
        rows += 1;
        let probe = cols[3];
        let key = cols[4];
        if probe == "-" {
            continue;
        }
        let spec = registry::probe_by_id(probe)
            .unwrap_or_else(|| panic!("guide_claims.tsv が未知のプローブを参照: {probe}"));
        if spec.kind == Kind::Attr {
            assert!(
                attr_table(probe).iter().any(|(name, _)| *name == key),
                "{probe} に属性 {key} が無い"
            );
        }
    }
    assert!(rows > 0, "guide_claims.tsv が空");
}

#[test]
fn cargo_toml_requires_internal_diagnostics_for_the_three_tests() {
    let manifest =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("Cargo.toml");
    for name in [
        "sm121_isa_probe_registry",
        "sm121_isa_probe_compile",
        "sm121_isa_probe_exec_real_device",
    ] {
        let needle = format!("name = \"{name}\"\nrequired-features = [\"internal-diagnostics\"]");
        assert!(
            manifest.contains(&needle),
            "{name}: required-features の指定が無い（#1390 の先例）"
        );
    }
}

// ---------------------------------------------------------------- TMA（PR-B）のホスト側モデル・encoder 引数

use common::model_tma::{self as mt, Swz, TmaSpec};
use common::registry::Outcome;
use common::registry_tma::{
    SPEC_BASE, SPEC_COORD, SPEC_MULTICAST, SPEC_OOB_NAN, SPEC_OOB_NEG, SPEC_OOB_NONE, SPEC_STORE,
    SPEC_SWZ32, SPEC_SWZ64, SPEC_SWZ128, all_tma_specs,
};
use common::types::Launch;

fn dump_of(spec: &TmaSpec, elems: &[Option<u32>], oob: u32) -> Vec<u32> {
    let mut out = vec![7u32];
    out.extend(elems.iter().map(|e| e.unwrap_or(oob)));
    assert_eq!(out.len(), mt::HDR + spec.dump_words as usize);
    out
}

fn record(o: Outcome) -> String {
    match o {
        Outcome::Record(d) => d,
        other => panic!("Record を期待: {other:?}"),
    }
}

#[test]
fn every_tma_spec_satisfies_the_encoder_constraints() {
    for (name, spec) in all_tma_specs() {
        spec.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    // 制約違反は検出できる（陰性）。
    let mut bad = SPEC_SWZ64;
    bad.box_cols = 32; // 128 B > 64 B の swizzle 幅
    assert!(bad.validate().is_err());
    let mut bad = SPEC_BASE;
    bad.expect_tx = 513;
    assert!(bad.validate().is_err());
    let mut bad = SPEC_BASE;
    bad.global_cols = 95; // 行ストライド 380 B は 16 の倍数でない
    assert!(bad.validate().is_err());
}

#[test]
fn tma_probe_wiring_is_consistent() {
    for p in probes().iter().filter(|p| {
        p.id.starts_with("tma.") || p.id.starts_with("clu.rt") || p.id.starts_with("ctl.raw")
    }) {
        let Launch::Raw { cluster } = p.launch else {
            panic!("{}: raw 起動であること", p.id)
        };
        if cluster > 0 {
            assert_eq!(p.grid % cluster, 0, "{}: grid は cluster の倍数", p.id);
        }
        if let Some(spec) = p.tma {
            // tensor map 系: 入力（global）は spec どおりの語数、出力語数はカーネルの書き込み規約どおり。
            assert_eq!(
                (p.make_input)().len(),
                spec.global_words() as usize,
                "{}",
                p.id
            );
            let want_out = if spec.readback_global {
                1 + spec.global_words() as usize
            } else if p.id == "tma.prefetch" {
                1
            } else if p.id == "ctl.rawmap" {
                5
            } else {
                (mt::HDR + spec.dump_words as usize) * p.grid as usize
            };
            assert_eq!(p.out_words, want_out, "{}", p.id);
        }
    }
    // PR-B の AC3 の全プローブが存在する（欠落の検出）。
    for id in [
        "tma.base_cta",
        "tma.coord",
        "tma.oob_none",
        "tma.oob_nan",
        "tma.oob_neg",
        "tma.swz32",
        "tma.swz64",
        "tma.swz128",
        "tma.store",
        "tma.bulk_cta",
        "tma.bulk_cluster",
        "tma.prefetch",
        "tma.multicast",
    ] {
        assert!(registry::probe_by_id(id).is_some(), "{id}");
    }
}

#[test]
fn swizzle_candidate_models_are_bijections_and_the_src_b64_model_matches_the_xor_model() {
    for (swz, words) in [(Swz::B32, 64u32), (Swz::B64, 128), (Swz::B128, 256)] {
        let phys: BTreeSet<u32> = (0..words)
            .map(|w| mt::swizzle_xor_phys_word(swz, w))
            .collect();
        assert_eq!(phys.len() as u32, words, "{swz:?}: 全単射でない");
        assert!(phys.iter().all(|&p| p < words), "{swz:?}: 範囲外");
    }
    // src の B64 仮説（`tma_swizzled_chunk_a`）は標準の 64B XOR モデルと全語で一致する。
    for w in 0..128u32 {
        assert_eq!(
            mt::swizzle_src_b64_phys_word(w),
            mt::swizzle_xor_phys_word(Swz::B64, w),
            "w={w}"
        );
    }
    // 恒等ではない（行によっては並べ替えが起きる）ので、線形との区別が付く。
    assert!((0..128u32).any(|w| mt::swizzle_xor_phys_word(Swz::B64, w) != w));
    assert!((0..64u32).any(|w| mt::swizzle_xor_phys_word(Swz::B32, w) != w));
    assert!((0..256u32).any(|w| mt::swizzle_xor_phys_word(Swz::B128, w) != w));
}

/// `[polls]` ヘッダと smem ダンプを出力する TMA カーネルか（tensor 系 load・multicast・bulk。
/// それ以外〈store・prefetch・ctl・clu〉はヘッダを持たない）。
fn has_polls_header(id: &str) -> bool {
    matches!(
        id,
        "tma.bulk_cta"
            | "tma.bulk_cluster"
            | "tma.base_cta"
            | "tma.base_cluster"
            | "tma.coord"
            | "tma.oob_none"
            | "tma.oob_nan"
            | "tma.oob_neg"
            | "tma.swz32"
            | "tma.swz64"
            | "tma.swz128"
            | "tma.multicast"
    )
}

/// `ST(<式>, smem[i])`（smem のダンプ）の式から、ダンプの先頭オフセット（`Nu + i` の N）を取り出す。
fn dump_base_offset(src: &str) -> Option<usize> {
    let end = src.find(", smem[i])")?;
    let call = &src[..end];
    let expr = &call[call.rfind("ST(")? + 3..];
    let expr: String = expr.chars().filter(|c| !c.is_whitespace()).collect();
    let tail = expr.strip_suffix("+i")?;
    let last = tail.rsplit('+').next()?;
    last.trim_end_matches('u').parse().ok()
}

/// カーネルのヘッダ長（出力へのダンプの先頭オフセット・multicast のスロット幅・ヘッダ語の格納数）が、
/// checker が解釈するヘッダ長 `mt::HDR`（`[polls]` の 1 語）と一致する（出力を別のヘッダ長として
/// 解釈する取り違えの検出）。
#[test]
fn kernel_header_length_matches_the_checker_interpretation() {
    let mut checked = 0;
    for p in probes().iter().filter(|p| has_polls_header(p.id)) {
        assert_eq!(
            dump_base_offset(p.src),
            Some(mt::HDR),
            "{}: smem ダンプの先頭オフセットがヘッダ長 {} と不一致",
            p.id,
            mt::HDR
        );
        if p.id == "tma.multicast" {
            assert!(
                p.src
                    .contains(&format!("blockIdx.x * ({}u + dump_words)", mt::HDR)),
                "{}: スロット幅がヘッダ長と不一致",
                p.id
            );
            assert!(p.src.contains("ST(slot, p1)"), "{}", p.id);
        } else {
            assert!(
                p.src.contains("ST(0u, p"),
                "{}: ヘッダ語（polls）の格納が無い",
                p.id
            );
        }
        checked += 1;
    }
    assert_eq!(
        checked, 12,
        "対象のカーネル数（load 9・multicast 1・bulk 2）"
    );
    // 出力語数の宣言（registry）とヘッダ・ダンプ長の整合（bulk は 1 + 64 語）。
    let bulk = registry::probe_by_id("tma.bulk_cta").expect("bulk");
    assert_eq!(bulk.out_words, mt::HDR + 64);
}

/// tx-count の整合の静的検査（`mbarrier.arrive.expect_tx` の総量が同じ phase の実転送量と一致し、待つ
/// phase に必ず転送が来る。実転送量より少ない expect_tx はアンダーフロー＝未定義動作のため計測しない）。
/// 検査するのはソース構造と宣言値の一致までで、実行時の転送量そのものは保証しない:
/// - load・multicast の `TmaSpec.expect_tx` は box 全体のバイト数（OOB を含む `box_words * 4`）
/// - bulk のカーネル定数 256 は転送サイズ（`cp.async.bulk … 256`）かつ 64 語 x 4 B
/// - mbarrier は count 1 で init され、arrive.expect_tx は tid 0 の 1 か所だけ（multicast は各 CTA の
///   tid 0 が自分の mbarrier へ。発行は rank 0 のみで mask 0b11 のため両 CTA が box 全体を受信する）
/// - 待ちの呼び出しは 1 回で phase parity 0（転送の来ない phase を待たない）
#[test]
fn expect_tx_matches_the_transfer_size_and_every_waited_phase_receives_a_transfer() {
    for p in probes().iter().filter(|p| p.src.contains("mbarrier.init")) {
        assert_eq!(p.src.matches("mbarrier.init").count(), 1, "{}", p.id);
        assert!(
            p.src.contains("mbarrier.init.shared::cta.b64 [%0], 1;"),
            "{}: init の count は 1",
            p.id
        );
        assert_eq!(
            p.src.matches("mbarrier.arrive.expect_tx").count(),
            1,
            "{}: arrive は 1 か所",
            p.id
        );
        let waits: Vec<&str> = p
            .src
            .match_indices("tma_wait_or_hang(mb, ")
            .map(|(i, _)| &p.src[i..i + 28])
            .collect();
        // ヘルパ定義（unsigned* out を取る宣言）を除いた呼び出しは 1 回で parity 0。
        let calls: Vec<&&str> = waits
            .iter()
            .filter(|w| w.starts_with("tma_wait_or_hang(mb, 0u,"))
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "{}: 待ちの呼び出しは parity 0 の 1 回",
            p.id
        );
        assert!(
            !p.src.contains("tma_wait_or_hang(mb, 1u"),
            "{}: 転送の来ない phase 1 を待っている",
            p.id
        );
        if let Some(spec) = p.tma {
            assert_eq!(
                spec.expect_tx,
                spec.box_words() * 4,
                "{}: expect_tx は box 全体（OOB を含む）",
                p.id
            );
            assert!(
                p.src.contains("\"r\"(expect_tx)"),
                "{}: expect_tx は spec の値をそのまま使う",
                p.id
            );
        } else {
            // bulk: 定数 256 が expect_tx と転送サイズの両方に現れ、64 語 x 4 B と一致する。
            assert!(
                p.src.contains("[%0], 256;") && p.src.contains("[%1], 256, [%2]"),
                "{}",
                p.id
            );
            assert_eq!(256, 64 * 4);
            assert_eq!(p.out_words, mt::HDR + 64);
        }
    }
    // multicast: 発行は rank 0 のみ・mask は両 CTA、expect_tx は rank に依らず各 CTA の tid 0 が自分の mbarrier へ。
    let mc = registry::probe_by_id("tma.multicast").expect("multicast");
    let arrive = mc.src.find("mbarrier.arrive.expect_tx").expect("arrive");
    let issue = mc.src.find("if (rank == 0u)").expect("issue");
    assert!(
        arrive < issue,
        "multicast: peer も含め arrive.expect_tx は発行（rank 0 のみ）より前に全 CTA が実行する"
    );
    assert!(
        mc.src.contains("unsigned short mask = 3;"),
        "宛先は 2 CTA（mask 0b11）"
    );
    assert_eq!(mc.grid, 2);
    assert_eq!(
        mc.out_words,
        2 * (mt::HDR + SPEC_MULTICAST.dump_words as usize)
    );
}

/// 上限付きの待ちが未完了を返した経路で転送先の smem を読まないことの静的検査（限界: ソース構造の
/// 検査であり、実行時の挙動は保証しない）: 共通ヘルパ `tma_wait_or_hang` は smem を参照せず、
/// `__threadfence_system()` で到達の事実を出してから上限なしの待ちに移る。カーネルは
/// `tma_try` を直接呼ばず、smem のダンプ（`ST(…, smem[i])`）はすべて最後の待ちの呼び出しより後にある。
#[test]
fn bounded_waits_never_read_the_destination_smem_before_completion() {
    let mut checked = 0;
    for p in probes()
        .iter()
        .filter(|p| p.src.contains("tma_wait_or_hang("))
    {
        let helper_start = p
            .src
            .find("__device__ __forceinline__ unsigned tma_wait_or_hang(")
            .expect("helper");
        let helper_end = helper_start + p.src[helper_start..].find("\n}\n").expect("helper end");
        let helper = &p.src[helper_start..helper_end];
        assert!(
            !helper.contains("smem"),
            "{}: 待ちのヘルパが smem を参照している",
            p.id
        );
        assert!(
            helper.contains("__threadfence_system()") && helper.contains("TMA_LIMIT_MARK"),
            "{}",
            p.id
        );
        // 未完了のときは上限なしの待ちへ移る（ここで return しない）。
        let hang = helper.find("while (!tma_try(").expect("hang loop");
        assert!(
            helper.find("return polls;").is_some_and(|r| r > hang),
            "{}: 未完了で戻る経路がある",
            p.id
        );
        let body = &p.src[helper_end..];
        if !body.contains("tma_wait_or_hang(") {
            continue; // ヘルパを含むだけで待ちを呼ばないカーネル（store・prefetch・ctl.rawmap）
        }
        assert!(
            !body.contains("tma_try("),
            "{}: カーネルが上限なしでない待ちを直接呼んでいる",
            p.id
        );
        let last_wait = body.rfind("tma_wait_or_hang(").expect("wait");
        // smem の参照は、宣言（`__shared__ … smem[512]`）と番兵の初期化（`smem[i] = TMA_SENTINEL`。書き込み）を除いてすべて最後の待ちの
        // 呼び出しより後にある（`ST(…, smem[0])` のような待ち前の読みも検出する）。
        for (pos, _) in body.match_indices("smem[") {
            let is_decl = body[body[..pos].rfind('\n').unwrap_or(0)..pos].contains("__shared__");
            let is_sentinel_init = body[pos..]
                .split(';')
                .next()
                .is_some_and(|stmt| stmt.contains("= TMA_SENTINEL"));
            assert!(
                is_decl || is_sentinel_init || pos > last_wait,
                "{}: 待ちの完了前に smem を読んでいる",
                p.id
            );
        }
        checked += 1;
    }
    assert_eq!(
        checked, 12,
        "mbarrier の待ちを持つカーネル数（load 9・multicast 1・bulk 2）"
    );
    // wait_group を使う store は、smem を読むのは TMA 自身で、カーネルは待ち（上限なし・外部 timeout）の後に
    // 目印だけを書いて終了する。
    let store = registry::probe_by_id("tma.store").expect("store");
    let (wait, magic) = (
        store.src.find("cp.async.bulk.wait_group 0"),
        store.src.find("ST(0u, 0x57025E5Eu)"),
    );
    assert!(wait.zip(magic).is_some_and(|(w, m)| w < m));
}

#[test]
fn tma_match_details_are_all_key_value_tokens() {
    // XFER／BASE の成立の detail も k=v のみ（aggregate.py がキー集合を固定して検査する）。
    let data = mt::global_data(64, 96);
    let ok = dump_of(&SPEC_BASE, &mt::box_elems(&SPEC_BASE, &data, false), 0);
    let both: Vec<u32> = ok.iter().chain(ok.iter()).copied().collect();
    for o in [
        mt::check_base(&SPEC_BASE, &data, &ok),
        mt::check_multicast(&SPEC_BASE, &data, &both),
        mt::check_prefetch(&SPEC_BASE, &[], &[mt::PREFETCH_MAGIC]),
    ] {
        let Outcome::Match(d) = o else {
            panic!("Match を期待: {o:?}")
        };
        assert!(d.split_whitespace().all(|t| t.contains('=')), "{d}");
    }
}

#[test]
fn bulk_output_is_not_interpreted_as_a_four_word_load_header() {
    // データ語がヘッダ語として表示されない（Codex P2/Bugbot の指摘の再発防止）。
    let input: Vec<u32> = (0..64).map(|i| 0xB000_0000 | (i * 7 + 1)).collect();
    let mut out = vec![5u32];
    out.extend(&input);
    let Outcome::Match(d) = mt::check_bulk(&input, &out) else {
        panic!("Match を期待")
    };
    assert_eq!(d, "polls=5 words=64 bit_exact=true");
    // 4 語ヘッダ前提の長さ（66 語）は長さ不一致として拒否される。
    let mut old_layout = vec![0u32, 5];
    old_layout.extend(&input);
    assert!(matches!(
        mt::check_bulk(&input, &old_layout),
        Outcome::Mismatch { .. }
    ));
}

/// mbarrier を使うカーネルの初期化順序（CUTLASS の `fence_barrier_init` 相当）の静的検査: `mbarrier.init`
/// の後に `fence.mbarrier_init.release.cluster` があり、cluster 同期（`barrier.cluster.arrive`）はその後に
/// 来る。smem を generic proxy で初期化するカーネルは、最初の `__syncthreads` より前に
/// `fence.proxy.async.shared::cta` を持つ。順序の欠落（multicast で peer の mbarrier へ complete_tx が
/// 届く前に init が見えない等）が「結果不一致」という誤ったハードウェア結論になるのを防ぐ。
#[test]
fn mbarrier_kernels_fence_the_init_before_any_cross_proxy_or_cross_cta_use() {
    let mut checked = 0;
    for p in probes()
        .iter()
        .filter(|p| p.kind == Kind::Kernel && p.src.contains("mbarrier.init"))
    {
        let init = p.src.find("mbarrier.init").expect("init");
        let fence = p.src.find("fence.mbarrier_init.release.cluster");
        assert!(
            fence.is_some_and(|f| f > init),
            "{}: init の後に fence.mbarrier_init.release.cluster が無い",
            p.id
        );
        if let Some(arrive) = p.src.find("barrier.cluster.arrive") {
            assert!(
                fence.is_some_and(|f| f < arrive),
                "{}: cluster 同期より前に fence が必要",
                p.id
            );
        }
        let first_sync = p.src.find("__syncthreads();").expect("sync");
        let proxy = p.src.find("fence.proxy.async.shared::cta");
        assert!(
            proxy.is_some_and(|f| f < first_sync),
            "{}: 最初の __syncthreads より前に fence.proxy.async が必要",
            p.id
        );
        // 待ちは上限付きのヘルパ経由（生の無限ループを書かない）。
        assert!(
            !p.src.contains("while (1)") && !p.src.contains("while(1)"),
            "{}",
            p.id
        );
        checked += 1;
    }
    assert!(checked >= 12, "対象が少なすぎる（{checked} 件）");
    // store は mbarrier を使わず、smem の書き込みの後に fence.proxy.async を出してから同期する。
    let store = registry::probe_by_id("tma.store").expect("store");
    let (fence, sync) = (
        store.src.find("fence.proxy.async.shared::cta"),
        store.src.find("__syncthreads();"),
    );
    assert!(fence.zip(sync).is_some_and(|(f, s)| f < s));
}

/// NVIDIA の swizzle 定義（物理バイトアドレスのビット [7,7+B) をビット [4,4+B) へ XOR。B は 32B=1・
/// 64B=2・128B=3）から**手で計算した**既知の値。実装の式を写したものではない（循環の回避）。
#[test]
fn xor_swizzle_model_matches_hand_computed_values_from_the_nvidia_definition() {
    // (幅, 線形バイト, 物理バイト)。語添字は 1/4。
    let cases: [(Swz, u32, u32); 14] = [
        // 128B（B=3）: 行 = 128 B。行 r のチャンク c(16 B) は c ^ r。
        (Swz::B128, 0, 0),
        (Swz::B128, 128, 144),  // 行 1 のチャンク 0 → チャンク 1
        (Swz::B128, 896, 1008), // 行 7 のチャンク 0 → チャンク 7
        (Swz::B128, 144, 128),  // 行 1 のチャンク 1 → チャンク 0
        // 64B（B=2）: 行 = 64 B。ビット [7,9) は 2 行ごとに +1。
        (Swz::B64, 64, 64),   // 行 1: ビット 7 以上が 0 → 不変
        (Swz::B64, 128, 144), // 行 2 のチャンク 0 → チャンク 1
        (Swz::B64, 384, 432), // 行 6 のチャンク 0 → チャンク 3（384 ^ 48）
        (Swz::B64, 448, 496), // 行 7 のチャンク 0 → 448 ^ 48
        // 32B（B=1）: 行 = 32 B。ビット 7 が 4 行ごとに 1 → 1 チャンク分だけずれる。
        (Swz::B32, 32, 32),
        (Swz::B32, 128, 144), // 行 4 のチャンク 0 → チャンク 1
        (Swz::B32, 160, 176), // 行 5 のチャンク 0 → チャンク 1（160 ^ 16）
        (Swz::B32, 240, 224), // 行 7 のチャンク 1（240）→ 240 ^ 16 = 224
        (Swz::B32, 256, 256), // 行 8: ビット 7 は 0（256 = 0b1_0000_0000）→ 不変
        (Swz::B32, 384, 400), // 行 12: ビット 7 が 1 → 384 ^ 16
    ];
    for (swz, linear, phys) in cases {
        assert_eq!(
            mt::swizzle_xor_phys_word(swz, linear / 4) * 4,
            phys,
            "{swz:?}: 線形 {linear} B の物理バイトは {phys} B のはず"
        );
    }
}

/// 本 box では src の B64 仮説（`tma_swizzled_chunk_a`。行番号のみに依存）と標準の XOR モデル
/// （絶対アドレスのビットに依存）を区別できない: タイル先頭が 1024 B 整列で行ストライドが 64 B なら
/// 両者は全語で同一になる。区別するにはタイル先頭を 64 B ずらす必要があるが、TMA は swizzle 使用時に
/// smem 先頭の整列を要求するため、区別できる box は作れない。分類が `XOR_ADDR_BITS+SRC_B64_MODEL` の
/// 両方を返すのはこのため（RULE.txt 10b・doc に明記）。
#[test]
fn src_b64_model_and_xor_model_are_indistinguishable_on_the_aligned_64b_box() {
    let data = mt::global_data(64, 96);
    let elems = mt::box_elems(&SPEC_SWZ64, &data, false);
    let mut dump = vec![1u32];
    dump.extend(std::iter::repeat_n(0, 128));
    for (w, e) in elems.iter().enumerate() {
        dump[mt::HDR + mt::swizzle_src_b64_phys_word(w as u32) as usize] = e.expect("範囲内");
    }
    let c = record(mt::classify_swizzle(&SPEC_SWZ64, &data, &dump));
    assert!(c.contains("XOR_ADDR_BITS+SRC_B64_MODEL"), "{c}");
}

#[test]
fn oob_fill_classes_are_deduplicated_preserving_first_seen_order() {
    // 値を昇順に並べると種別が ZERO, OTHER, NAN, OTHER と並び、OTHER が連続しない重複になる。
    let spec = &SPEC_OOB_NONE;
    let data = mt::global_data(64, 96);
    let elems = mt::box_elems(spec, &data, false);
    let mut dump = dump_of(spec, &elems, 0);
    let oob_idx: Vec<usize> = elems
        .iter()
        .enumerate()
        .filter(|(_, e)| e.is_none())
        .map(|(i, _)| mt::HDR + i)
        .collect();
    let values = [0u32, 0x1234_5678, 0x7fc0_0000, 0x8000_0001];
    for (k, i) in oob_idx.iter().enumerate() {
        dump[*i] = values[k % values.len()];
    }
    let d = record(mt::classify_oob(spec, &data, &dump));
    assert!(d.contains("oob_fill=ZERO+OTHER+NAN "), "{d}");
}

#[test]
fn coord_classification_distinguishes_the_hypotheses() {
    let data = mt::global_data(64, 96);
    let normal = mt::box_elems(&SPEC_COORD, &data, false);
    let transposed = mt::box_elems(&SPEC_COORD, &data, true);
    assert_ne!(
        normal, transposed,
        "座標が仮説を区別できない位置になっていない"
    );
    assert!(normal.iter().all(Option::is_some) && transposed.iter().all(Option::is_some));
    let c = record(mt::classify_coord(
        &SPEC_COORD,
        &data,
        &dump_of(&SPEC_COORD, &normal, 0),
    ));
    assert!(
        c.contains("class=ELEM_INNER_FIRST ") || c.ends_with("class=ELEM_INNER_FIRST"),
        "{c}"
    );
    assert!(!c.contains("TRANSPOSED"), "{c}");
    let c = record(mt::classify_coord(
        &SPEC_COORD,
        &data,
        &dump_of(&SPEC_COORD, &transposed, 0),
    ));
    assert!(c.contains("class=TRANSPOSED"), "{c}");
    let garbage = dump_of(&SPEC_COORD, &vec![Some(1); 128], 0);
    let c = record(mt::classify_coord(&SPEC_COORD, &data, &garbage));
    assert!(
        c.contains("class=NONE") && c.contains("dump=0x"),
        "NONE ではダンプ全文を記録する: {c}"
    );
}

#[test]
fn oob_classification_records_the_fill_kind_and_counts() {
    let data = mt::global_data(64, 96);
    for spec in [&SPEC_OOB_NONE, &SPEC_OOB_NAN] {
        let elems = mt::box_elems(spec, &data, false);
        let inside = elems.iter().filter(|e| e.is_some()).count();
        assert_eq!(inside, 32, "範囲内 8 列 x 4 行");
        // expect_tx は OOB を含む box 全体のバイト数（部分 expect_tx は未定義動作のため計測しない）。
        assert_eq!(spec.expect_tx, spec.box_words() * 4);
        for (oob, want) in [
            (0u32, "oob_fill=ZERO"),
            (0x7fc0_0000, "oob_fill=NAN"),
            (mt::SENTINEL, "oob_fill=SENTINEL"),
            (0x1234_5678, "oob_fill=OTHER"),
        ] {
            let d = record(mt::classify_oob(spec, &data, &dump_of(spec, &elems, oob)));
            assert!(
                d.contains("inrange=MATCH") && d.contains(want) && d.contains("oob_elems=96"),
                "{d}"
            );
        }
    }
    let elems = mt::box_elems(&SPEC_OOB_NEG, &data, false);
    assert_eq!(
        elems.iter().filter(|e| e.is_some()).count(),
        32,
        "負座標: 行 0〜3 x 列 0〜7"
    );
    let mut bad = dump_of(
        &SPEC_OOB_NONE,
        &mt::box_elems(&SPEC_OOB_NONE, &data, false),
        0,
    );
    bad[mt::HDR] ^= 1;
    assert!(record(mt::classify_oob(&SPEC_OOB_NONE, &data, &bad)).contains("inrange=MISMATCH"));
    // OOB が無い box は NO_OOB_ELEMENTS。
    let d = record(mt::classify_oob(
        &SPEC_BASE,
        &data,
        &dump_of(&SPEC_BASE, &mt::box_elems(&SPEC_BASE, &data, false), 0),
    ));
    assert!(
        d.contains("oob_fill=NO_OOB_ELEMENTS") && d.contains("oob_distinct=-"),
        "{d}"
    );
}

#[test]
fn swizzle_classification_names_each_matching_candidate() {
    for (spec, swz) in [
        (&SPEC_SWZ32, Swz::B32),
        (&SPEC_SWZ64, Swz::B64),
        (&SPEC_SWZ128, Swz::B128),
    ] {
        let data = mt::global_data(64, 96);
        let elems = mt::box_elems(spec, &data, false);
        let linear = dump_of(spec, &elems, 0);
        let c = record(mt::classify_swizzle(spec, &data, &linear));
        assert!(c.contains("LINEAR"), "{swz:?}: {c}");
        // XOR モデルの配置（線形の語 w が物理 phys(w) にある）。
        let mut swz_dump = linear.clone();
        for (w, e) in elems.iter().enumerate() {
            swz_dump[mt::HDR + mt::swizzle_xor_phys_word(swz, w as u32) as usize] =
                e.expect("範囲内");
        }
        let c = record(mt::classify_swizzle(spec, &data, &swz_dump));
        assert!(
            c.contains("XOR_ADDR_BITS") && !c.contains("LINEAR"),
            "{swz:?}: {c}"
        );
        assert_eq!(
            c.contains("SRC_B64_MODEL"),
            swz == Swz::B64,
            "{swz:?}: src の B64 仮説は 64B のみ: {c}"
        );
        // どれとも一致しない配置は NONE＋ダンプ全文。
        let mut bad = swz_dump.clone();
        bad.swap(mt::HDR, mt::HDR + 1);
        let c = record(mt::classify_swizzle(spec, &data, &bad));
        assert!(
            c.contains("class=NONE") && c.contains("dump=0x"),
            "{swz:?}: {c}"
        );
    }
}

#[test]
fn transfer_checks_distinguish_success_from_each_failure_kind() {
    let data = mt::global_data(64, 96);
    // base: ビット一致 → Match。1 語不一致・長さ違いは Mismatch（待ちの失敗はカーネルが終了せず、外部 timeout で扱う）。
    let ok = dump_of(&SPEC_BASE, &mt::box_elems(&SPEC_BASE, &data, false), 0);
    assert!(matches!(
        mt::check_base(&SPEC_BASE, &data, &ok),
        Outcome::Match(_)
    ));
    let mut one_off = ok.clone();
    one_off[mt::HDR + 5] ^= 1;
    assert!(matches!(
        mt::check_base(&SPEC_BASE, &data, &one_off),
        Outcome::Mismatch { count: 1, .. }
    ));
    assert!(matches!(
        mt::check_base(&SPEC_BASE, &data, &ok[..10]),
        Outcome::Mismatch { .. }
    ));
    // store: global の box 領域が既知パターン・他は番兵 → Match。
    let spec = &SPEC_STORE;
    let mut out = vec![mt::STORE_MAGIC];
    out.extend(std::iter::repeat_n(
        mt::SENTINEL,
        spec.global_words() as usize,
    ));
    for r in 0..spec.box_rows {
        for c in 0..spec.box_cols {
            let idx = ((spec.cy as u32 + r) * spec.global_cols + spec.cx as u32 + c) as usize;
            out[1 + idx] = 0xC0DE_0000 | (r * spec.box_cols + c);
        }
    }
    assert!(matches!(
        mt::check_store(spec, &[], &out),
        Outcome::Match(_)
    ));
    let mut bad = out.clone();
    bad[1] = 0; // box の外が書き換わった
    assert!(matches!(
        mt::check_store(spec, &[], &bad),
        Outcome::Mismatch { .. }
    ));
    let mut no_magic = out.clone();
    no_magic[0] = 0;
    assert!(matches!(
        mt::check_store(spec, &[], &no_magic),
        Outcome::Mismatch { .. }
    ));
    // bulk・prefetch・multicast。
    let input: Vec<u32> = (0..64).map(|i| 0xB000_0000 | (i * 7 + 1)).collect();
    let mut bulk = vec![3u32];
    bulk.extend(&input);
    assert!(matches!(mt::check_bulk(&input, &bulk), Outcome::Match(_)));
    bulk[10] ^= 1;
    assert!(matches!(
        mt::check_bulk(&input, &bulk),
        Outcome::Mismatch { .. }
    ));
    assert!(matches!(
        mt::check_prefetch(&SPEC_BASE, &[], &[mt::PREFETCH_MAGIC]),
        Outcome::Match(_)
    ));
    assert!(matches!(
        mt::check_prefetch(&SPEC_BASE, &[], &[0]),
        Outcome::Mismatch { .. }
    ));
    let slot = ok.clone();
    let mut both: Vec<u32> = slot.iter().chain(slot.iter()).copied().collect();
    assert!(matches!(
        mt::check_multicast(&SPEC_BASE, &data, &both),
        Outcome::Match(_)
    ));
    both[slot.len() + mt::HDR + 3] ^= 1; // 2 つ目の CTA だけ不一致
    assert!(matches!(
        mt::check_multicast(&SPEC_BASE, &data, &both),
        Outcome::Mismatch { count: 1, .. }
    ));
}

#[test]
fn recorded_tma_details_are_machine_parsable_key_value_tokens() {
    let data = mt::global_data(64, 96);
    let elems = mt::box_elems(&SPEC_OOB_NONE, &data, false);
    for d in [
        record(mt::classify_oob(
            &SPEC_OOB_NONE,
            &data,
            &dump_of(&SPEC_OOB_NONE, &elems, 0),
        )),
        record(mt::classify_coord(
            &SPEC_COORD,
            &data,
            &dump_of(&SPEC_COORD, &vec![Some(1); 128], 0),
        )),
    ] {
        assert!(
            d.split_whitespace().all(|t| t.contains('=')),
            "k=v 以外のトークンがある: {d}"
        );
    }
}
