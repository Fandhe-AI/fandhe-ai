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
            let want = if p.id == "tc5.cross" {
                "tc5_alloc".to_string()
            } else {
                p.id.replace('.', "_")
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
        4
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
