//! イシュー #1517: framework-compare `train`（`bench-fandhe`。
//! `BATCH=64・D_IN=784・D_HIDDEN=256・D_OUT=10`）の各 GEMM 形状が
//! `tile::should_split_k`（split-K 対象判定の純関数）の観点で
//! Some/None のどちらになるかを機械的に確定する「帰属表」テスト。
//!
//! # 位置づけ
//!
//! 本テストは `should_split_k` 単体の Some/None・[`tile::SplitKPlan`]
//! の値を確定するのみであり、「実際にその形状が `MetalGemm::
//! dispatch_auto`（split-K 判定を行う唯一の本番入口。イシュー #1516）
//! へ到達するか」（=入口条件）は別問題として扱う。`docs/perf/
//! metal-gemm-splitk-framework-compare-1517.md` §2 の帰属表は、本テスト
//! の出力（形状条件）と、コード読み（`crates/autodiff/src/nn/linear.rs`・
//! `crates/autodiff/src/grad.rs`・`crates/backend-metal/src/ops.rs`）で
//! 裏取りした入口条件を併記する。
//!
//! `should_split_k` は `pub mod tile;`（`crates/backend-metal/src/lib.rs`）
//! ・`pub use tile::{..., should_split_k}`（クレートルート）のいずれも
//! `cfg(target_os = "macos")` を持たないため、本統合テストは Linux
//! （CI・本実装環境）でもコンパイル・実行できる（`pad`／`layout` と同じ
//! 設計判断。`tile.rs` モジュール冒頭 doc 参照）。
//!
//! 実行例（帰属表を標準出力へ表示する）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test splitk_train_shape_attribution -- --nocapture
//! ```

use fandhe_ai_backend_metal::SplitKPlan;
use fandhe_ai_backend_metal::tile;

/// `scripts/bench/framework-compare/bench-fandhe/src/main.rs` の train
/// タスク定数（`const BATCH: usize = 64;` 等）を複製する。出典は同ファイル
/// `BATCH`/`D_IN`/`D_HIDDEN`/`D_OUT` 定義（2026-09-10 時点）。本テストは
/// これらの値が変わった場合に追従が必要である旨を明示するため、値を
/// 直接埋め込まず定数として名前を付ける（実装計画 §3.3 手順 6 参照）。
const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

/// 1 GEMM 形状の帰属表 1 行分。`shape` は `(m, n, k)`（`should_split_k`
/// の引数順）。
struct ShapeRow {
    label: &'static str,
    shape: (usize, usize, usize),
}

/// forward 2 本（L1・L2）＋ backward 4 本（L1/L2 の d_input／d_weight）。
/// `docs/perf/metal-gemm-splitk-framework-compare-1517.md` §2 の帰属表
/// 元データ。
///
/// - forward L1: `(BATCH, D_IN) x (D_IN, D_HIDDEN)` → `(m,n,k) =
///   (BATCH, D_HIDDEN, D_IN)`
/// - forward L2: `(BATCH, D_HIDDEN) x (D_HIDDEN, D_OUT)` → `(m,n,k) =
///   (BATCH, D_OUT, D_HIDDEN)`
/// - backward は `matmul_vjp`（`crates/autodiff/src/grad.rs`）の
///   `da = gemm(g, b^T)`・`db = gemm(a^T, g)` に対応する形状。
///   L1 の入力 `a`＝`(BATCH, D_IN)`・重み `b`＝`(D_IN, D_HIDDEN)`・
///   上流勾配 `g`＝`(BATCH, D_HIDDEN)` とすると:
///   - L1 d_input（`da = g @ b^T`）: `(m,n,k) = (BATCH, D_IN, D_HIDDEN)`
///   - L1 d_weight（`db = a^T @ g`）: `(m,n,k) = (D_IN, D_HIDDEN, BATCH)`
///   - L2 d_input（`da = g @ b^T`。`g`＝`(BATCH, D_OUT)`・
///     `b`＝`(D_HIDDEN, D_OUT)`）: `(m,n,k) = (BATCH, D_HIDDEN, D_OUT)`
///   - L2 d_weight（`db = a^T @ g`。`a`＝`(BATCH, D_HIDDEN)`）:
///     `(m,n,k) = (D_HIDDEN, D_OUT, BATCH)`
///
///   注（実装計画 §3.3 手順 5）: L1 d_input は `docs/autodiff-nograd-
///   leaf-dinput-skip-decision.md`（非学習葉への d_input 伝播スキップ）
///   の対象になりうる（入力 `x` は学習対象でない葉のため）。本表では
///   その場合でも形状条件自体の記録として行を残す（計算されない場合が
///   あることを注記するのみ）。
const SHAPES: &[ShapeRow] = &[
    ShapeRow {
        label: "forward L1 (64,784)x(784,256)",
        shape: (BATCH, D_HIDDEN, D_IN),
    },
    ShapeRow {
        label: "forward L2 (64,256)x(256,10)",
        shape: (BATCH, D_OUT, D_HIDDEN),
    },
    ShapeRow {
        label: "backward L1 d_input (非学習葉スキップ対象の可能性あり)",
        shape: (BATCH, D_IN, D_HIDDEN),
    },
    ShapeRow {
        label: "backward L1 d_weight",
        shape: (D_IN, D_HIDDEN, BATCH),
    },
    ShapeRow {
        label: "backward L2 d_input",
        shape: (BATCH, D_HIDDEN, D_OUT),
    },
    ShapeRow {
        label: "backward L2 d_weight",
        shape: (D_HIDDEN, D_OUT, BATCH),
    },
];

fn print_row(label: &str, shape: (usize, usize, usize), plan: Option<SplitKPlan>) {
    let (m, n, k) = shape;
    match plan {
        Some(p) => println!(
            "| {label} | ({m},{n},{k}) | Some | partitions={} k_per_partition={} bm={} bn={} bk={} |",
            p.partitions, p.k_per_partition, p.tile.bm, p.tile.bn, p.tile.bk
        ),
        None => println!("| {label} | ({m},{n},{k}) | None | - |"),
    }
}

/// 帰属表の機械生成（`--nocapture` で標準出力へ Markdown 表を出す）。
/// アサーション自体は行わない（`assert_*` は各形状ごとの個別テストで
/// 行う。本テストは表の可視化専用）。
#[test]
fn print_attribution_table() {
    println!("| shape | (m,n,k) | should_split_k | plan |");
    println!("|---|---|---|---|");
    for row in SHAPES {
        let (m, n, k) = row.shape;
        print_row(row.label, row.shape, tile::should_split_k(m, n, k));
    }
}

/// forward L1 `(64,256,784)` が split-K 対象形状（`should_split_k` が
/// `Some`）であることを固定する（`docs/perf/metal-gemm-splitk-
/// framework-compare-1517.md` §2 に転記した帰属表の前提。実測値は
/// `--nocapture` 出力〈`print_attribution_table`〉を正とする）。
#[test]
fn forward_l1_shape() {
    let plan = tile::should_split_k(BATCH, D_HIDDEN, D_IN);
    assert!(
        plan.is_some(),
        "forward L1 (64,256,784) の should_split_k が None だった（帰属表の前提が崩れている）"
    );
}

/// forward L2 `(64,10,256)` が split-K 対象形状であることを固定する。
#[test]
fn forward_l2_shape() {
    let plan = tile::should_split_k(BATCH, D_OUT, D_HIDDEN);
    assert!(
        plan.is_some(),
        "forward L2 (64,10,256) の should_split_k が None だった（帰属表の前提が崩れている）"
    );
}

/// backward 4 形状はいずれも `k < max(m,n)`（K が支配的でない）ため
/// MLX Case 1 の手順 1 で弾かれ `None` になることを固定する。
#[test]
fn backward_shapes_are_none() {
    let backward = [
        ("L1 d_input", (BATCH, D_IN, D_HIDDEN)),
        ("L1 d_weight", (D_IN, D_HIDDEN, BATCH)),
        ("L2 d_input", (BATCH, D_HIDDEN, D_OUT)),
        ("L2 d_weight", (D_HIDDEN, D_OUT, BATCH)),
    ];
    for (label, (m, n, k)) in backward {
        let plan = tile::should_split_k(m, n, k);
        assert!(
            plan.is_none(),
            "backward {label} ({m},{n},{k}) の should_split_k が Some だった\
             （帰属表の前提『backward は形状条件でも非到達』が崩れている。\
             もっとも本番経路は入口条件〈gemm_strided_nt_tn〉により\
             どのみち dispatch_auto 非経由のため split-K には到達しない）"
        );
    }
}
