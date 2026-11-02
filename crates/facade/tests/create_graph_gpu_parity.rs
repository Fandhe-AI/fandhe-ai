//! 二階微分（`create_graph`・孫テープ再生）の CUDA 実機 parity テストの雛形
//! （イシュー #3776・親 #3719。測り方は `docs/autodiff-higher-order-grad-decision.md`
//! §48〈#3739〉）。Phase 5 の各族の実装 issue は [`CASES`] に 1 行足し、
//! Phase 10 が GB10（`<cuda-node>`）で測る。Metal 版（イシュー #3777）は同じ
//! ファイルの `create_graph_<族>_metal_matches_cpu`（`cfg(target_os = "macos")`
//! ・`#[ignore]`）で、Phase 10 が `<metal-node>` で測る。
//!
//! 置き場が facade の理由: `crates/autodiff` は GPU バックエンドへ届かず、facade は
//! 全バックエンドを合成する公開経路で、既存の `create_graph_facade.rs`
//! （#3168・#3169 で実測済み）と同じ実行経路になるため（§48.4）。既存テスト・
//! そのログは変えない。
//!
//! 測る量（§48.4）: Q1 `first_order()` が同デバイスの素の `backward` と bit 一致、
//! Q2 子テープ上の `cg.grad`、Q3 二階 `H v`（ゼロでないことも assert）、
//! Q4 孫テープ再生（任意）は GPU と CPU の間を REQ-2 統一複合判定で比べる。
//! Q5 拒否の一致は GPU でも `Err(Backward)` かつ子テープが空であること。
//!
//! 判定: 比較は `fandhe_ai_backend_cpu::parity::compare`。tolerance 定数・baseline の
//! 値は変えない。`Baseline` 方式は `tests/common/create_graph_gpu_parity_baseline.rs`
//! の空の置き場を引き、行が無ければ undetermined で止まる。
//!
//! S3（§48.4）: Metal の `gemm_fp32_strict` は `gemm` に委譲され split-K を通りうる。
//! 各族は [`SplitKShapes`] で子テープ構造内の MatMul 形状を宣言し、macOS では
//! `Judge::Strict` のケースは `should_split_k` が `None` であることを厳密に検査する。
//! `Judge::Baseline` のケースは非発火検査を課さず、宣言の形式検査だけを行う。Strict で
//! 発火する形状は主表から外す（論点 8＝スコープ外）。結合順序の異なる経路が避けられない族だけが
//! `Judge::Baseline` の Metal 行（`BaselineDevice::Metal`）に回る。Linux では
//! Metal の行が cfg で外れ、宣言の形式検査だけが CI で走る。
//!
//! 実行: `#[ignore]` の GPU テストは CI で走らず、`cargo test -p fandhe-ai --test
//! create_graph_gpu_parity --no-run` でビルドだけ通る。実機手順は
//! `docs/real-hardware-verification-env.md` 4.9 節（Metal は 7.3 節）。RULE.txt の雛形は
//! `docs/perf/logs/create-graph-gpu-parity-template/RULE.txt`。
//! 登録表自体の健全性は CPU 同士の非 `#[ignore]` 自己検査で CI が守る。

#[path = "common/create_graph_gpu_parity_baseline.rs"]
mod baseline;

use std::sync::Arc;

use fandhe_ai::{AutodiffError, CustomFunction, Device, Tape, Tensor, Var};
use fandhe_ai_backend_cpu::parity::compare;

/// 経路区分（§48.4）。`H` の pass は「経路の parity」でカーネル同士ではない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // 各族の実装 issue が使う語彙。雛形では K だけ使う。
enum Route {
    /// デバイスカーネル経路。
    K,
    /// デバイスからホストへの実体化を含むフォールバック経路。
    H,
    /// opt-in ゲートの後ろ（既定 OFF なので H として測る）。
    G,
}

/// 判定方式。既定は `Strict`（fail_count == 0）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Baseline は所有者承認後に使う語彙。雛形は空の置き場だけ。
enum Judge {
    Strict,
    /// 結合順序の異なる経路を含む族。baseline に行が無ければ undetermined。
    Baseline,
}

/// 親・子テープ（同一デバイス）。Q4 の孫テープ再生が要る族は `grand` を使う。
struct Tapes {
    parent: Tape,
    child: Tape,
    /// Q4 用の別の空の子テープ（`jvp_create_graph` が受け取る）。
    jvp_child: Tape,
    grand: Tape,
}

/// 1 ケースが返す測定量。
struct Outputs {
    /// `cg.first_order()` の勾配（Q1）。
    first_order: Vec<f32>,
    /// 同デバイスの素の `Tape::backward` の勾配（Q1 の bit 比較相手）。
    plain: Vec<f32>,
    /// `cg.grad` の値（Q2）。
    grad: Vec<f32>,
    /// 固定方向 `probe_dir` の二階 `H v`（Q3。ゼロでないことも assert）。
    hv: Vec<f32>,
    /// 孫テープ再生の出力（Q4。測らないなら `None`）。
    replay: Option<Vec<f32>>,
}

/// S3（§48.4）の宣言。各族が明示する（宣言なしで Metal の主表に入れない）。
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // `Listed`／`Rule` は MatMul を含む族の実装 issue が使う。自己検査が構築する。
enum SplitKShapes {
    /// 子テープの構造に MatMul を含まない。
    NoMatMul,
    /// 構造に含まれる MatMul の `(m, n, k)` を列挙できる族。
    Listed(&'static [(usize, usize, usize)]),
    /// 列挙できない族。`k < max(m, n)` となるよう形状を選んだ規則の文（RULE.txt に引用）。
    Rule(&'static str),
}

/// 宣言の形式検査（全プラットフォーム）。曖昧な宣言は fail-closed で弾く。
fn validate_split_k_decl(d: &SplitKShapes) -> Result<(), String> {
    match d {
        SplitKShapes::NoMatMul => Ok(()),
        SplitKShapes::Listed([]) => {
            Err("Listed が空: MatMul が無いなら NoMatMul と書く".to_string())
        }
        SplitKShapes::Listed(shapes) => {
            if shapes.iter().any(|&(m, n, k)| m == 0 || n == 0 || k == 0) {
                Err("Listed に 0 を含む次元がある".to_string())
            } else {
                Ok(())
            }
        }
        SplitKShapes::Rule(s) if s.trim().is_empty() => Err("Rule の文が空".to_string()),
        SplitKShapes::Rule(_) => Ok(()),
    }
}

/// macOS 限定の検査。宣言の形式検査は全ケース、`Listed` の split-K 非発火は `Strict` のみ。
/// `should_split_k` は `backend-metal` が `pub use` 済みの既存項目（公開面は不変）。
#[cfg(target_os = "macos")]
fn check_split_k_not_fired(case: &Case) -> Result<(), String> {
    validate_split_k_decl(&case.split_k).map_err(|e| format!("{}:S3: {e}", case.name))?;
    // 非発火の必須検査は Strict 対象に限る。Baseline 対象は split-K が避けられない形状を
    // Listed に登録でき、承認済み baseline の judge へ到達させる（宣言の形式検査は上で維持）。
    if !matches!(case.judge, Judge::Strict) {
        return Ok(());
    }
    if let SplitKShapes::Listed(shapes) = case.split_k {
        for &(m, n, k) in shapes {
            if fandhe_ai_backend_metal::should_split_k(m, n, k).is_some() {
                return Err(format!(
                    "{}:S3 違反: ({m},{n},{k}) で split-K が発火する。主表から外す（論点 8）",
                    case.name
                ));
            }
        }
    }
    Ok(())
}

/// `device` を baseline の軸へ写す。CPU・未指定は baseline 対象外（undetermined）。
fn baseline_device(device: Option<Device>) -> Option<baseline::BaselineDevice> {
    match device {
        Some(Device::Cuda(_)) => Some(baseline::BaselineDevice::Cuda),
        #[cfg(target_os = "macos")]
        Some(Device::Metal) => Some(baseline::BaselineDevice::Metal),
        _ => None,
    }
}

/// 登録表の 1 行。**Op を足すときはここに 1 行足す**（テスト関数は族単位で別に置く）。
struct Case {
    /// ケース名。baseline の `case` と逐語一致させる。
    name: &'static str,
    /// 族名（`create_graph_<族>_cuda_matches_cpu` の `<族>`。§48.4 の命名表）。
    family: &'static str,
    route: Route,
    judge: Judge,
    /// S3: 子テープ構造に含まれる MatMul の split-K 宣言（Metal 用。§48.4）。
    split_k: SplitKShapes,
    /// 決定的な入力でテープを走らせ測定量を返す。形状・入力はこの関数内で固定する
    /// （S1: CI の小形状 + S2: 2 のべき乗でない中形状、S4: 決定的、S5: しきい値回避）。
    run: fn(&Tapes) -> Outputs,
    /// Q5。拒否入力で `Err(Backward)` かつ子が空（`leaf_count() == 0`）なら `true`。不要なら `None`。
    reject: Option<fn(&Tapes) -> bool>,
}

/// 登録表。雛形の自己検査用に、実装済みの `Mul`・`Sum` だけの 1 行を置く。
const CASES: &[Case] = &[
    Case {
        name: "scaffold_cubic",
        family: "scaffold",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scaffold_cubic,
        reject: None,
    },
    // 選択縮約 4 Op（イシュー #3789・決定記録 §34）。forward はデバイスのカーネル
    // （CUDA `crates/backend-cuda/src/ops.rs` の max／min 縮約、Metal
    // `crates/backend-metal/src/ops.rs` の同縮約）。`Amax`／`Amin` の V3 値（均等分配）は
    // 1 階と同じホスト関数で、`Max`／`Min` の cgrad は親記録値から作る先勝ちマスクと
    // `where_cond`（ホスト計算を含む）。構造に MatMul は無い。
    Case {
        name: "select_reduce_minmax_max",
        family: "select_reduce_minmax",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_max,
        reject: Some(reject_select_reduce_minmax_with_custom),
    },
    Case {
        name: "select_reduce_minmax_min",
        family: "select_reduce_minmax",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_min,
        reject: Some(reject_select_reduce_minmax_with_custom),
    },
    Case {
        name: "select_reduce_minmax_amax",
        family: "select_reduce_minmax",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_amax,
        reject: Some(reject_select_reduce_minmax_with_custom),
    },
    Case {
        name: "select_reduce_minmax_amin",
        family: "select_reduce_minmax",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_amin,
        reject: Some(reject_select_reduce_minmax_with_custom),
    },
    // 選択縮約（イシュー #3790・決定記録 §34）。`BackendOps::stat_*` は CPU のみ実装で、
    // CUDA／Metal ではホストへフォールバックする（経路 H）。実機は未実測（#3856・#3858）。
    Case {
        name: "select_reduce_order_kthvalue",
        family: "select_reduce_order",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_kthvalue,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "select_reduce_order_median_all",
        family: "select_reduce_order",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_median_all,
        reject: None,
    },
    Case {
        name: "select_reduce_order_quantile_linear",
        family: "select_reduce_order",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_select_reduce_quantile,
        reject: None,
    },
    // 統計縮約 4 Op（イシュー #3791・決定記録 §33・§52）。`BackendOps::var`／`vector_norm`／
    // `vector_norm_p` を実装しているのは `backend-cpu` だけで、CUDA／Metal ではホストへ
    // フォールバックする。`Std` は対応する `BackendOps` メソッドが無く常にホスト実装。cgrad の
    // V3 値は 1 階と同じホスト関数で、構造は f32 の `Var` 演算（MatMul は無い）。経路 H の pass は
    // 「経路の parity」でカーネル同士ではない。実機は未実測（Phase 10 の #3856・#3858）。
    Case {
        name: "stat_varnorm_var",
        family: "stat_varnorm",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_varnorm_var,
        reject: Some(reject_stat_varnorm_with_custom),
    },
    Case {
        name: "stat_varnorm_std",
        family: "stat_varnorm",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_varnorm_std,
        reject: Some(reject_stat_varnorm_with_custom),
    },
    Case {
        name: "stat_varnorm_norm_l1",
        family: "stat_varnorm",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_varnorm_l1,
        reject: Some(reject_stat_varnorm_with_custom),
    },
    Case {
        name: "stat_varnorm_norm_l2",
        family: "stat_varnorm",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_varnorm_l2,
        reject: Some(reject_stat_varnorm_with_custom),
    },
    Case {
        name: "stat_varnorm_norm_p",
        family: "stat_varnorm",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_varnorm_pnorm,
        reject: Some(reject_stat_varnorm_with_custom),
    },
    // 統計縮約 3 Op（イシュー #3792・決定記録 §33/§51）。`logsumexp`／`stat_nansum`／`stat_nanmean`
    // は backend-cpu にだけ実装があり、CUDA／Metal ではホストへフォールバックする（経路 H）。
    // 実機は未実測（#3856・#3858）。構造に MatMul は無い。
    Case {
        name: "stat_reduce_lse_nan_logsumexp",
        family: "stat_reduce_lse_nan",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_reduce_logsumexp,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "stat_reduce_lse_nan_nansum",
        family: "stat_reduce_lse_nan",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_reduce_nansum,
        reject: None,
    },
    Case {
        name: "stat_reduce_lse_nan_nanmean",
        family: "stat_reduce_lse_nan",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stat_reduce_nanmean,
        reject: None,
    }, // 損失の二値・マージン系 6 variant（イシュー #3799・決定記録 §39/§54）。Bce は 1 階が融合カーネル
    // `bce_loss_backward` 経路（K）、残る 5 つは 1 階がホスト関数（H）。構造に MatMul は無い。
    // 実機は未実測（#3857・#3859）。
    Case {
        name: "loss_binary_margin_bce",
        family: "loss_binary_margin",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_bce,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "loss_binary_margin_bce_pos_weight",
        family: "loss_binary_margin",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_bce_pos_weight,
        reject: None,
    },
    Case {
        name: "loss_binary_margin_soft_margin",
        family: "loss_binary_margin",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_soft_margin,
        reject: None,
    },
    Case {
        name: "loss_binary_margin_hinge_embedding",
        family: "loss_binary_margin",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_hinge_embedding,
        reject: None,
    },
    Case {
        name: "loss_binary_margin_margin_ranking",
        family: "loss_binary_margin",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_margin_ranking,
        reject: None,
    },
    Case {
        name: "loss_binary_margin_sigmoid_focal",
        family: "loss_binary_margin",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_binary_margin_sigmoid_focal,
        reject: None,
    },
    // 構造変換 3 Op（イシュー #3793・決定記録 §35/§56）。forward がデバイスカーネルなのは `interpolate` の
    // `Nearest`・`Bilinear`（CUDA `ops.rs` の `interpolate`。ほかの mode は `Unsupported` でホストへ落ちる）と、
    // `scatter`（Overwrite を GPU で処理する `max_unpool`）・`mul`（`dropout`）。1 階の scatter_add・mul も
    // デバイスで走るので K。`Bicubic`・`Area` はホストフォールバックを含むので H。構造に MatMul は無い。
    // Dropout のマスクはホスト RNG で、CPU と GPU で同じマスクを引くよう `manual_seed` で固定する。
    // 実機は未実測（#3856・#3858）。
    Case {
        name: "structural_interpolate_nearest",
        family: "structural",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_interpolate_nearest,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "structural_interpolate_bilinear",
        family: "structural",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_interpolate_bilinear,
        reject: None,
    },
    Case {
        name: "structural_interpolate_bicubic",
        family: "structural",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_interpolate_bicubic,
        reject: None,
    },
    Case {
        name: "structural_interpolate_area",
        family: "structural",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_interpolate_area,
        reject: None,
    },
    Case {
        name: "structural_max_unpool",
        family: "structural",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_max_unpool,
        reject: None,
    },
    Case {
        name: "structural_dropout",
        family: "structural",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_structural_dropout,
        reject: None,
    },
    // RNN セル 2 Op（イシュー #3795・決定記録 §37/§58）。forward の `gemm_bias_act`・`gru_pointwise` と 1 階の
    // pointwise は CPU・CUDA・Metal ともデバイスカーネルでホストへ落ちない（K）。子テープの構造は MatMul・
    // pointwise・Narrow・Concat で、微分は `h0`（公開 API の `x` は定数）。形状は B=3・D=4・H=5（3H=15）に固定し、
    // 構造とその二階の GEMM の k がすべて 15 以下になるので split-K は発火しない。実機は未実測（#3856・#3858）。
    Case {
        name: "rnn_gru_cell_rnn",
        family: "rnn_gru_cell",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule("k ∈ {B=3, D=4, H=5, 3H=15} ≤ 15 < 16 のため split-K 非発火"),
        run: run_rnn_gru_cell_rnn,
        reject: Some(reject_rnn_gru_cell_with_custom),
    },
    Case {
        name: "rnn_gru_cell_gru",
        family: "rnn_gru_cell",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule("k ∈ {B=3, D=4, H=5, 3H=15} ≤ 15 < 16 のため split-K 非発火"),
        run: run_rnn_gru_cell_gru,
        reject: None,
    },
    // 損失のクラス軸系 6 variant（イシュー #3800・決定記録 §40/§55）。6 つとも 1 階がホスト関数（H）で、
    // 構造の Softmax・pointwise・Sum がデバイスで走る。構造に MatMul は無い。実機は未実測（#3857・#3859）。
    Case {
        name: "loss_class_cewo",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_cewo,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "loss_class_multi_margin",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_multi_margin,
        reject: None,
    },
    Case {
        name: "loss_class_multilabel_margin",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_multilabel_margin,
        reject: None,
    },
    Case {
        name: "loss_class_multilabel_soft_margin",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_multilabel_soft_margin,
        reject: None,
    },
    Case {
        name: "loss_class_cosine_embedding",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_cosine_embedding,
        reject: None,
    },
    Case {
        name: "loss_class_triplet_margin",
        family: "loss_class",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_class_triplet_margin,
        reject: None,
    },
    // 損失の回帰系 3 variant（イシュー #3798・決定記録 §39/§53）。KlDiv は 1 階が融合カーネル
    // `kl_div_loss_backward` 経路（K）、Poisson・Gaussian は 1 階がホスト関数（H）。構造に MatMul は無い。
    // 実機は未実測（#3857・#3859）。
    Case {
        name: "loss_regression_kl_div",
        family: "loss_regression",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_regression_kl_div,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "loss_regression_poisson_nll",
        family: "loss_regression",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_regression_poisson_nll,
        reject: None,
    },
    Case {
        name: "loss_regression_gaussian_nll",
        family: "loss_regression",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_loss_regression_gaussian_nll,
        reject: None,
    },
];

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::<f32>::new(data, shape).expect("test fixture: tensor")
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

/// `loss = sum(x^3)`。二階が非自明（Hessian 対角 6x）な最小例。
fn run_scaffold_cubic(tp: &Tapes) -> Outputs {
    let x0: Vec<f32> = (0..17).map(|i| 0.25 + 0.125 * i as f32).collect();
    let dir: Vec<f32> = (0..17).map(|i| 1.0 - 0.0625 * i as f32).collect();

    let x = tp.parent.var(&t(x0, &[17]));
    let loss = x.mul(&x).unwrap().mul(&x).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&x).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    let g = cg.grad(&x).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[17]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let hv = vals(grads2.get(&xc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

/// 選択縮約ケースの共通本体（S2: 2 のべき乗でない中形状 `[17, 33]`・軸 1 縮約）。
/// 入力は決定的で各 lane の値が重複せず（最小間隔 0.01。S4・S5）、argmax／argmin が
/// タイにならない。損失は `sum(reduce(x)² ⊙ c)`（行ごとの非一様な重み `c`）で、`H v` は
/// 選択位置の対角 `2·c·v` となり全ゼロにならない。
fn run_select_reduce_minmax_body(
    tp: &Tapes,
    reduce: for<'a> fn(&fandhe_ai::Var<'a>) -> fandhe_ai::Var<'a>,
) -> Outputs {
    let (rows, cols) = (17usize, 33usize);
    let n = rows * cols;
    let x0: Vec<f32> = (0..n)
        .map(|i| ((i * 389 + 17) % 1009) as f32 * 0.01 + 1.0)
        .collect();
    let dir: Vec<f32> = (0..n).map(|i| 0.5 + 0.01 * (i % 7) as f32).collect();
    let c: Vec<f32> = (0..rows).map(|i| 0.5 + 0.1 * i as f32).collect();

    let x = tp.parent.var(&t(x0, &[rows, cols]));
    let cv = tp.parent.var_no_grad(&t(c, &[rows]));
    let y = reduce(&x);
    let loss = y.mul(&y).unwrap().mul(&cv).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&x).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    let g = cg.grad(&x).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[rows, cols]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let hv = vals(grads2.get(&xc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。同じ損失の微分可能 jvp（方向 `dir`）の出力を孫テープで
    // `x` について逆伝播し、その勾配を GPU と CPU で比べる（子は上と別の空テープ）。
    let tangent = t(
        (0..n).map(|i| 0.5 + 0.01 * (i % 7) as f32).collect(),
        &[rows, cols],
    );
    let res = tp
        .parent
        .jvp_create_graph(&loss, &x, &tangent, &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rx = res.child_var(&x).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rx).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

fn run_select_reduce_max(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.max(Some(1)).unwrap())
}

fn run_select_reduce_min(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.min(Some(1)).unwrap())
}

fn run_select_reduce_amax(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.amax(Some(1)).unwrap())
}

fn run_select_reduce_amin(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.amin(Some(1)).unwrap())
}

/// 恒久対象外の `Op::Custom`（決定記録 §21.2 A 区分）用の恒等関数。
struct IdentityFn;

impl CustomFunction for IdentityFn {
    fn name(&self) -> &str {
        "identity"
    }
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
        Ok(input_shapes[0].to_vec())
    }
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
        Ok(inputs[0].clone())
    }
    fn backward(
        &self,
        _inputs: &[&Tensor<f32>],
        _out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        _requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
        Ok(vec![Some(upstream.clone())])
    }
}

/// Q5: 選択縮約の祖先に `Op::Custom` があるグラフは GPU でも拒否され、子テープが空のまま。
fn reject_select_reduce_minmax_with_custom(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![0.5, 1.5, 1.0, 2.0], &[2, 2]));
    let Ok(id) = tp.parent.custom(Arc::new(IdentityFn), &[x]) else {
        return false;
    };
    let loss = id.max(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// 選択縮約の共通走査。`f(x)` が選択 Op を含む出力を返し、損失は `sum(y ⊙ y)`。
/// 入力は順位が切り替わらない（タイなし・最小間隔 0.07 以上）値に限る（S5）。
fn run_select_reduce(tp: &Tapes, f: for<'t> fn(&Var<'t>) -> Var<'t>) -> Outputs {
    let x0: Vec<f32> = (0..15)
        .map(|i| 0.3 + 0.07 * ((i * 7) % 15) as f32)
        .collect();
    let dir: Vec<f32> = (0..15).map(|i| 1.0 - 0.0625 * i as f32).collect();
    let x = tp.parent.var(&t(x0, &[3, 5]));
    let y = f(&x);
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&x).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    let g = cg.grad(&x).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[3, 5]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let hv = vals(grads2.get(&xc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_select_reduce_kthvalue(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| x.mul(x).unwrap().kthvalue(2, 1).unwrap().0)
}

fn run_select_reduce_median_all(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| x.mul(x).unwrap().median(None).unwrap())
}

fn run_select_reduce_quantile(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| {
        x.mul(x)
            .unwrap()
            .quantile(0.4, Some(1), fandhe_ai::QuantileInterpolation::Linear)
            .unwrap()
    })
}

/// 統計縮約（#3792）。入力は有限値（`run_select_reduce` の固定入力）、損失は `sum(y ⊙ y)`。
fn run_stat_reduce_logsumexp(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| x.mul(x).unwrap().logsumexp(Some(1)).unwrap())
}

fn run_stat_reduce_nansum(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| x.mul(x).unwrap().nansum(Some(1)).unwrap())
}

fn run_stat_reduce_nanmean(tp: &Tapes) -> Outputs {
    run_select_reduce(tp, |x| x.mul(x).unwrap().nanmean(Some(1)).unwrap())
}

/// 損失の二値・マージン系（#3799）。葉 `z`（長さ 12）から損失を作り、`loss²` で `H v ≠ 0` にする。
/// 区分定数の損失（Hinge・MarginRanking）も `loss²` の `g` 経由で二階が非零になる。
fn run_loss_binary_margin(tp: &Tapes, f: for<'t> fn(&Var<'t>) -> Var<'t>) -> Outputs {
    let z0: Vec<f32> = vec![
        0.4, 0.55, 0.7, 0.85, 0.9, 0.1, 0.6, 0.3, 0.2, 0.5, 0.8, 0.45,
    ];
    // 符号付き勾配との内積が偶然 0 にならない非対称な方向。
    let dir: Vec<f32> = (0..12)
        .map(|i| 1.0 - 0.0625 * i as f32 + 0.03 * (i * i) as f32)
        .collect();
    let z = tp.parent.var(&t(z0, &[12]));
    let l = f(&z);
    let loss = l.mul(&l).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&z).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&z).unwrap().expect("first_order"));
    let g = cg.grad(&z).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[12]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let zc = cg.child_var(&z).unwrap().expect("mirror");
    let hv = vals(grads2.get(&zc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_loss_binary_margin_bce(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let (x, tg) = (z.narrow(0, 0, 4).unwrap(), z.narrow(0, 4, 4).unwrap());
        x.bce_loss(&tg, fandhe_ai::nn::loss::Reduction::Mean)
            .unwrap()
    })
}

fn run_loss_binary_margin_bce_pos_weight(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let (x, tg) = (z.narrow(0, 0, 4).unwrap(), z.narrow(0, 4, 4).unwrap());
        let opts = fandhe_ai::nn::loss::BceWithLogitsOptions::default()
            .pos_weight(Tensor::<f32>::new(vec![2.0, 0.5, 3.0, 1.0], &[4]).unwrap());
        x.bce_with_logits_loss_with(&tg, fandhe_ai::nn::loss::Reduction::Mean, &opts)
            .unwrap()
    })
}

fn run_loss_binary_margin_soft_margin(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let x = z.narrow(0, 0, 4).unwrap();
        x.soft_margin_loss(
            &Tensor::<f32>::new(vec![1.0, -1.0, -1.0, 1.0], &[4]).unwrap(),
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

fn run_loss_binary_margin_hinge_embedding(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let x = z.narrow(0, 0, 4).unwrap();
        x.hinge_embedding_loss(
            &Tensor::<f32>::new(vec![1.0, -1.0, -1.0, 1.0], &[4]).unwrap(),
            1.0,
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

fn run_loss_binary_margin_margin_ranking(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let (x1, x2) = (z.narrow(0, 0, 4).unwrap(), z.narrow(0, 4, 4).unwrap());
        x1.margin_ranking_loss(
            &x2,
            &Tensor::<f32>::new(vec![1.0, -1.0, 1.0, -1.0], &[4]).unwrap(),
            0.5,
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

fn run_loss_binary_margin_sigmoid_focal(tp: &Tapes) -> Outputs {
    run_loss_binary_margin(tp, |z| {
        let x = z.narrow(0, 0, 4).unwrap();
        let opts = fandhe_ai::nn::loss::SigmoidFocalLossOptions::default()
            .alpha(Some(0.25))
            .gamma(2.0);
        x.sigmoid_focal_loss(
            &Tensor::<f32>::new(vec![1.0, 0.0, 0.7, 0.2], &[4]).unwrap(),
            &opts,
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

/// 損失のクラス軸系（#3800）。葉 `z`（長さ 12）から損失を作り、`loss²` で `H v ≠ 0` にする。
fn run_loss_class(tp: &Tapes, f: for<'t> fn(&Var<'t>) -> Var<'t>) -> Outputs {
    let z0: Vec<f32> = vec![
        0.4, 0.55, 0.7, 0.85, 0.9, 0.1, 0.6, 0.3, 0.2, 0.5, 0.8, 0.45,
    ];
    // 符号付き勾配との内積が偶然 0 にならない非対称な方向。
    let dir: Vec<f32> = (0..12)
        .map(|i| 1.0 - 0.0625 * i as f32 + 0.03 * (i * i) as f32)
        .collect();
    let z = tp.parent.var(&t(z0, &[12]));
    let l = f(&z);
    let loss = l.mul(&l).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&z).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&z).unwrap().expect("first_order"));
    let g = cg.grad(&z).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[12]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let zc = cg.child_var(&z).unwrap().expect("mirror");
    let hv = vals(grads2.get(&zc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_loss_class_cewo(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let opts = fandhe_ai::nn::loss::CrossEntropyOptions::default()
            .label_smoothing(0.1)
            .class_weight(Tensor::<f32>::new(vec![1.0, 2.0, 0.5, 1.5], &[4]).unwrap());
        z.reshape(&[3, 4])
            .unwrap()
            .cross_entropy_loss_with(
                &Tensor::<i32>::new(vec![0, 2, 3], &[3]).unwrap(),
                1,
                fandhe_ai::nn::loss::Reduction::Mean,
                &opts,
            )
            .unwrap()
    })
}

fn run_loss_class_multi_margin(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let opts = fandhe_ai::nn::loss::MultiMarginOptions::default().p(2);
        z.reshape(&[3, 4])
            .unwrap()
            .multi_margin_loss(
                &Tensor::<i32>::new(vec![1, 0, 3], &[3]).unwrap(),
                &opts,
                fandhe_ai::nn::loss::Reduction::Mean,
            )
            .unwrap()
    })
}

fn run_loss_class_multilabel_margin(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        z.reshape(&[3, 4])
            .unwrap()
            .multilabel_margin_loss(
                &Tensor::<i32>::new(vec![3, 0, -1, -1, 1, -1, 0, 0, 2, 1, -1, -1], &[3, 4])
                    .unwrap(),
                fandhe_ai::nn::loss::Reduction::Mean,
            )
            .unwrap()
    })
}

fn run_loss_class_multilabel_soft_margin(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        z.reshape(&[3, 4])
            .unwrap()
            .multilabel_soft_margin_loss(
                &Tensor::<f32>::new(
                    vec![1.0, 0.0, 0.3, 1.0, 0.0, 1.0, 0.5, 0.0, 0.2, 0.9, 1.0, 0.0],
                    &[3, 4],
                )
                .unwrap(),
                &fandhe_ai::nn::loss::MultiLabelSoftMarginOptions::default(),
                fandhe_ai::nn::loss::Reduction::Mean,
            )
            .unwrap()
    })
}

fn run_loss_class_cosine_embedding(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let (x1, x2) = (z.narrow(0, 0, 6).unwrap(), z.narrow(0, 6, 6).unwrap());
        x1.reshape(&[2, 3])
            .unwrap()
            .cosine_embedding_loss(
                &x2.reshape(&[2, 3]).unwrap(),
                &Tensor::<f32>::new(vec![1.0, -1.0], &[2]).unwrap(),
                0.2,
                fandhe_ai::nn::loss::Reduction::Mean,
            )
            .unwrap()
    })
}

fn run_loss_class_triplet_margin(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let p = |i: usize| z.narrow(0, 4 * i, 4).unwrap().reshape(&[2, 2]).unwrap();
        p(0).triplet_margin_loss(
            &p(1),
            &p(2),
            &fandhe_ai::nn::loss::TripletMarginOptions::default().margin(2.0),
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}
/// 構造変換（#3793）。葉 `z`（長さ 12。形状は `shape`）に線形 Op `f` を適用し、`Σ y²·c` で `H v ≠ 0` にする。
fn run_structural(tp: &Tapes, shape: &[usize], f: for<'t> fn(&Var<'t>) -> Var<'t>) -> Outputs {
    let z0: Vec<f32> = vec![
        0.4, 0.55, 0.7, 0.85, 0.9, 0.1, 0.6, 0.3, 0.2, 0.5, 0.8, 0.45,
    ];
    // 符号付き勾配との内積が偶然 0 にならない非対称な方向。
    let dir: Vec<f32> = (0..12)
        .map(|i| 1.0 - 0.0625 * i as f32 + 0.03 * (i * i) as f32)
        .collect();
    let z = tp.parent.var(&t(z0, shape));
    let y = f(&z);
    let ys = y.value().shape().to_vec();
    let c: Vec<f32> = (0..ys.iter().product::<usize>())
        .map(|i| 0.4 + 0.37 * i as f32 - 0.05 * (i * i) as f32)
        .collect();
    let c = tp.parent.var_no_grad(&t(c, &ys));
    let loss = y.mul(&y).unwrap().mul(&c).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&z).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&z).unwrap().expect("first_order"));
    let g = cg.grad(&z).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, shape));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let zc = cg.child_var(&z).unwrap().expect("mirror");
    let hv = vals(grads2.get(&zc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_structural_interpolate_nearest(tp: &Tapes) -> Outputs {
    run_structural(tp, &[1, 2, 6], |z| {
        z.interpolate(&[9], fandhe_ai::InterpolateMode::Nearest)
            .unwrap()
    })
}

fn run_structural_interpolate_bilinear(tp: &Tapes) -> Outputs {
    run_structural(tp, &[1, 2, 2, 3], |z| {
        z.interpolate(
            &[3, 4],
            fandhe_ai::InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .unwrap()
    })
}

fn run_structural_interpolate_bicubic(tp: &Tapes) -> Outputs {
    run_structural(tp, &[1, 1, 3, 4], |z| {
        z.interpolate(
            &[4, 5],
            fandhe_ai::InterpolateMode::Bicubic {
                align_corners: false,
            },
        )
        .unwrap()
    })
}

fn run_structural_interpolate_area(tp: &Tapes) -> Outputs {
    run_structural(tp, &[1, 1, 3, 4], |z| {
        z.interpolate(&[2, 3], fandhe_ai::InterpolateMode::Area)
            .unwrap()
    })
}

fn run_structural_max_unpool(tp: &Tapes) -> Outputs {
    run_structural(tp, &[1, 1, 3, 4], |z| {
        let (p, idx) = z
            .max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1], false)
            .unwrap();
        p.max_unpool2d(&idx, [2, 2], Some([1, 1]), [0, 0], None)
            .unwrap()
    })
}

fn run_structural_dropout(tp: &Tapes) -> Outputs {
    // マスクはホスト RNG。CPU と GPU の双方で同じマスクを引くよう固定し、並行実行を直列化する。
    static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    fandhe_ai::manual_seed(20_261_101);
    run_structural(tp, &[3, 4], |z| z.dropout(0.4, true).unwrap())
}

/// RNN セル（#3795）。公開 API の `x` は定数なので、葉は `h0`（`[3, 5]`）。`Σ h_n²·c` で `H v ≠ 0` にする。
/// `gru` が真なら `gru_forward_seq`、偽なら `rnn_forward_seq`。決定的な重み（`Rnn::new`／`Gru::new` のシード）。
fn run_rnn_gru_cell(tp: &Tapes, gru: bool) -> Outputs {
    let (b, d, h, steps) = (3usize, 4usize, 5usize, 2usize);
    let h0v: Vec<f32> = (0..b * h).map(|i| 0.3 * ((i as f32) * 0.9).sin()).collect();
    let xv: Vec<f32> = (0..steps * b * d)
        .map(|i| 0.5 * ((i as f32) * 0.37 + 0.2).cos())
        .collect();
    let x = t(xv, &[steps, b, d]);
    let rnn = fandhe_ai::nn::rnn::Rnn::new(d, h, true, 7).expect("Rnn::new");
    let grucell = fandhe_ai::nn::rnn::Gru::new(d, h, true, 7).expect("Gru::new");
    let dir: Vec<f32> = (0..b * h)
        .map(|i| 1.0 - 0.0625 * i as f32 + 0.03 * (i * i) as f32 / 4.0)
        .collect();
    let z = tp.parent.var(&t(h0v, &[b, h]));
    let h_n = if gru {
        tp.parent
            .gru_forward_seq(&grucell, &x, Some(&z))
            .expect("gru_forward_seq")
            .h_n
    } else {
        tp.parent
            .rnn_forward_seq(&rnn, &x, Some(&z))
            .expect("rnn_forward_seq")
            .h_n
    };
    let c: Vec<f32> = (0..b * h)
        .map(|i| 0.4 + 0.37 * i as f32 - 0.05 * (i * i) as f32 / 4.0)
        .collect();
    let c = tp.parent.var_no_grad(&t(c, &[b, h]));
    let loss = h_n.mul(&h_n).unwrap().mul(&c).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&z).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&z).unwrap().expect("first_order"));
    let g = cg.grad(&z).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[b, h]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let zc = cg.child_var(&z).unwrap().expect("mirror");
    let hv = vals(grads2.get(&zc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_rnn_gru_cell_rnn(tp: &Tapes) -> Outputs {
    run_rnn_gru_cell(tp, false)
}

fn run_rnn_gru_cell_gru(tp: &Tapes) -> Outputs {
    run_rnn_gru_cell(tp, true)
}

/// Q5（#3795）: RNN セルの下流に恒久対象外の `Op::Custom` があれば `Err(Backward)` で、子テープは空のまま。
fn reject_rnn_gru_cell_with_custom(tp: &Tapes) -> bool {
    struct Identity;
    impl fandhe_ai::CustomFunction for Identity {
        fn name(&self) -> &str {
            "identity"
        }
        fn output_shape(&self, s: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
            Ok(s[0].to_vec())
        }
        fn forward(&self, i: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
            Ok(i[0].contiguous())
        }
        fn backward(
            &self,
            _i: &[&Tensor<f32>],
            _o: &Tensor<f32>,
            up: &Tensor<f32>,
            _r: &[bool],
        ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
            Ok(vec![Some(up.contiguous())])
        }
    }
    let rnn = fandhe_ai::nn::rnn::Rnn::new(2, 3, true, 7).expect("Rnn::new");
    let x = t(vec![0.1, 0.2, 0.3, 0.4], &[2, 1, 2]);
    let h0 = tp.parent.var(&t(vec![0.2, -0.3, 0.5], &[1, 3]));
    let h_n = tp
        .parent
        .rnn_forward_seq(&rnn, &x, Some(&h0))
        .expect("rnn_forward_seq")
        .h_n;
    let custom = tp
        .parent
        .custom(Arc::new(Identity), &[h_n])
        .expect("custom");
    let loss = custom.sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// 損失の回帰系（#3798）。葉 `z`（長さ 12 = `[x; target; var]`）から損失を作り、`loss²` で `H v ≠ 0` にする。
fn run_loss_regression(tp: &Tapes, f: for<'t> fn(&Var<'t>) -> Var<'t>) -> Outputs {
    let z0: Vec<f32> = vec![
        0.3, -0.8, 0.5, 0.2, 0.55, 0.7, 0.85, 0.65, 0.4, 0.8, 1.2, 0.6,
    ];
    let dir: Vec<f32> = (0..12).map(|i| 1.0 - 0.0625 * i as f32).collect();
    let z = tp.parent.var(&t(z0, &[12]));
    let l = f(&z);
    let loss = l.mul(&l).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&z).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&z).unwrap().expect("first_order"));
    let g = cg.grad(&z).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[12]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let zc = cg.child_var(&z).unwrap().expect("mirror");
    let hv = vals(grads2.get(&zc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn run_loss_regression_kl_div(tp: &Tapes) -> Outputs {
    run_loss_regression(tp, |z| {
        let (x, tg) = (z.narrow(0, 0, 4).unwrap(), z.narrow(0, 4, 4).unwrap());
        x.kl_div_loss(&tg, fandhe_ai::nn::loss::Reduction::Sum)
            .unwrap()
    })
}

fn run_loss_regression_poisson_nll(tp: &Tapes) -> Outputs {
    run_loss_regression(tp, |z| {
        let (x, tg) = (z.narrow(0, 0, 4).unwrap(), z.narrow(0, 4, 4).unwrap());
        x.poisson_nll_loss(
            &tg,
            &fandhe_ai::nn::loss::PoissonNllOptions::default(),
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

fn run_loss_regression_gaussian_nll(tp: &Tapes) -> Outputs {
    run_loss_regression(tp, |z| {
        let (x, tg, v) = (
            z.narrow(0, 0, 4).unwrap(),
            z.narrow(0, 4, 4).unwrap(),
            z.narrow(0, 8, 4).unwrap(),
        );
        x.gaussian_nll_loss(
            &tg,
            &v,
            &fandhe_ai::nn::loss::GaussianNllOptions::default(),
            fandhe_ai::nn::loss::Reduction::Mean,
        )
        .unwrap()
    })
}

/// Q5: `quantile` に `Op::Custom`（恒久的に対象外）を合成すると fail-closed で拒否され、
/// 子テープは空のまま。
fn reject_select_reduce_with_custom(tp: &Tapes) -> bool {
    struct Identity;
    impl fandhe_ai::CustomFunction for Identity {
        fn name(&self) -> &str {
            "identity"
        }
        fn output_shape(&self, s: &[&[usize]]) -> Result<Vec<usize>, AutodiffError> {
            Ok(s[0].to_vec())
        }
        fn forward(&self, i: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError> {
            Ok(i[0].contiguous())
        }
        fn backward(
            &self,
            _i: &[&Tensor<f32>],
            _o: &Tensor<f32>,
            up: &Tensor<f32>,
            _r: &[bool],
        ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError> {
            Ok(vec![Some(up.contiguous())])
        }
    }
    let x = tp.parent.var(&t(vec![0.3, 0.9, 0.5, 0.7], &[4]));
    let q = x
        .quantile(0.4, None, fandhe_ai::QuantileInterpolation::Linear)
        .unwrap()
        .reshape(&[1])
        .unwrap();
    let c = tp
        .parent
        .custom(std::sync::Arc::new(Identity), &[q])
        .unwrap();
    let loss = c.sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

fn run_stat_varnorm_var(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.var(Some(1), 1).unwrap())
}

fn run_stat_varnorm_std(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.std(Some(1), 1).unwrap())
}

fn run_stat_varnorm_l1(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.norm_l1(Some(1)).unwrap())
}

fn run_stat_varnorm_l2(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.norm_l2(Some(1)).unwrap())
}

fn run_stat_varnorm_pnorm(tp: &Tapes) -> Outputs {
    run_select_reduce_minmax_body(tp, |x| x.norm_p(1.5, Some(1)).unwrap())
}

/// Q5: 統計縮約の祖先に `Op::Custom`（恒久的に対象外）があるグラフは GPU でも拒否され、
/// 子テープは空のまま。
fn reject_stat_varnorm_with_custom(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![0.5, 1.5, 1.0, 2.0], &[2, 2]));
    let Ok(id) = tp.parent.custom(Arc::new(IdentityFn), &[x]) else {
        return false;
    };
    let loss = id.std(None, 1).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

fn tapes_for(device: Option<Device>) -> Tapes {
    let mk = || match device {
        None => fandhe_ai::tape(),
        Some(d) => fandhe_ai::tape_for(d).expect("実機が利用可能な前提のテスト"),
    };
    Tapes {
        parent: mk(),
        child: mk(),
        jvp_child: mk(),
        grand: mk(),
    }
}

/// REQ-2 統一複合判定（`Baseline` 方式は空の置き場を引く）。fail 時は RULE.txt から
/// 引用できるよう fail_count・max_abs_diff・max_rel_err を出す。
fn judge(
    case: &Case,
    device: Option<Device>,
    quantity: &str,
    gpu: &[f32],
    cpu: &[f32],
) -> Result<(), String> {
    let report = compare(gpu, cpu).map_err(|e| format!("{}:{quantity}: {e:?}", case.name))?;
    match case.judge {
        Judge::Strict if report.passes() => Ok(()),
        Judge::Strict => Err(format!(
            "{}:{quantity}: REQ-2 fail (route={:?}, fail_count={}/{}, max_abs_diff={:.6e}, \
             max_rel_err={:.6e})",
            case.name,
            case.route,
            report.fail_count,
            report.total,
            report.max_abs_diff,
            report.max_rel_err
        )),
        Judge::Baseline => {
            let dev = baseline_device(device).ok_or_else(|| {
                format!(
                    "undetermined: {} は baseline 対象外のデバイス {device:?}",
                    case.name
                )
            })?;
            baseline::check_no_regression(baseline::lookup(case.name, dev)?, &report)
        }
    }
}

/// 1 ケースを `device`（None は CPU）と CPU 参照で突き合わせる。
fn check_case(case: &Case, device: Option<Device>) -> Result<(), String> {
    // S3: Metal では split-K が発火しない形状だけを主表に置く（§48.4）。
    #[cfg(target_os = "macos")]
    if matches!(device, Some(Device::Metal)) {
        check_split_k_not_fired(case)?;
    }
    let gpu = (case.run)(&tapes_for(device));
    let cpu = (case.run)(&tapes_for(None));
    // Q1: 同デバイスの素の backward と bit 一致（構造的な性質）。
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&gpu.first_order) != bits(&gpu.plain) {
        return Err(format!(
            "{}:Q1: first_order が backward と bit 不一致",
            case.name
        ));
    }
    // Q1: GPU・CPU 間も REQ-2 複合判定で突き合わせる（§48.4。tolerance は不変）。
    // 同デバイス bit 一致だけでは backward と first_order が同じ誤値を返しても通るため。
    judge(
        case,
        device,
        "Q1 first_order",
        &gpu.first_order,
        &cpu.first_order,
    )?;
    judge(case, device, "Q2 grad", &gpu.grad, &cpu.grad)?;
    // 線形な Op で二階の検査が自明に通るのを防ぐ。
    if gpu.hv.iter().all(|&v| v == 0.0) {
        return Err(format!("{}:Q3: H v が全ゼロ", case.name));
    }
    judge(case, device, "Q3 hv", &gpu.hv, &cpu.hv)?;
    match (&gpu.replay, &cpu.replay) {
        (Some(g), Some(c)) => judge(case, device, "Q4 replay", g, c)?,
        (None, None) => {}
        _ => return Err(format!("{}:Q4: 片側だけ replay を返した", case.name)),
    }
    if let Some(reject) = case.reject
        && !reject(&tapes_for(device))
    {
        return Err(format!("{}:Q5: 拒否が CPU と一致しない", case.name));
    }
    Ok(())
}

/// Q5 の共通判定。`r` が `Err(Backward)` かつ `child` が空なら `true`。
fn is_clean_rejection<T>(r: Result<T, AutodiffError>, child: &Tape) -> bool {
    matches!(r, Err(AutodiffError::Backward(_))) && child.leaf_count() == 0
}

/// 族 `family` の全ケースを `device` で測る。panic メッセージは RULE.txt 引用用。
fn run_family(family: &str, device: Option<Device>) {
    let cases: Vec<&Case> = CASES.iter().filter(|c| c.family == family).collect();
    assert!(
        !cases.is_empty(),
        "undetermined: 族 `{family}` の登録が 0 件"
    );
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|c| check_case(c, device).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 実装 issue ごとに 1 本、`#[ignore]` の CUDA テストを足すマクロ
/// （`--exact` で 1 本ずつ実行するため。名前は `create_graph_<族>_cuda_matches_cpu`）。
macro_rules! cuda_family_test {
    ($name:ident, $family:literal) => {
        #[test]
        #[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
        fn $name() {
            run_family($family, Some(Device::Cuda(0)));
        }
    };
}

cuda_family_test!(create_graph_scaffold_cuda_matches_cpu, "scaffold");
cuda_family_test!(
    create_graph_select_reduce_minmax_cuda_matches_cpu,
    "select_reduce_minmax"
);
cuda_family_test!(
    create_graph_select_reduce_order_cuda_matches_cpu,
    "select_reduce_order"
);
cuda_family_test!(create_graph_stat_varnorm_cuda_matches_cpu, "stat_varnorm");
cuda_family_test!(
    create_graph_stat_reduce_lse_nan_cuda_matches_cpu,
    "stat_reduce_lse_nan"
);
cuda_family_test!(
    create_graph_loss_binary_margin_cuda_matches_cpu,
    "loss_binary_margin"
);
cuda_family_test!(create_graph_structural_cuda_matches_cpu, "structural");
cuda_family_test!(create_graph_rnn_gru_cell_cuda_matches_cpu, "rnn_gru_cell");
cuda_family_test!(create_graph_loss_class_cuda_matches_cpu, "loss_class");
cuda_family_test!(
    create_graph_loss_regression_cuda_matches_cpu,
    "loss_regression"
);

/// Metal 版（#3777）。マクロ定義は cfg 無しで、展開される `fn` だけを macOS に限る
/// （Linux で `unused_macros` にならない）。名前は `create_graph_<族>_metal_matches_cpu`。
macro_rules! metal_family_test {
    ($name:ident, $family:literal) => {
        #[cfg(target_os = "macos")]
        #[test]
        #[ignore = "Metal 実機が必要"]
        fn $name() {
            run_family($family, Some(Device::Metal));
        }
    };
}

metal_family_test!(create_graph_scaffold_metal_matches_cpu, "scaffold");
metal_family_test!(
    create_graph_select_reduce_minmax_metal_matches_cpu,
    "select_reduce_minmax"
);
metal_family_test!(
    create_graph_select_reduce_order_metal_matches_cpu,
    "select_reduce_order"
);
metal_family_test!(create_graph_stat_varnorm_metal_matches_cpu, "stat_varnorm");
metal_family_test!(
    create_graph_stat_reduce_lse_nan_metal_matches_cpu,
    "stat_reduce_lse_nan"
);
metal_family_test!(
    create_graph_loss_binary_margin_metal_matches_cpu,
    "loss_binary_margin"
);
metal_family_test!(create_graph_structural_metal_matches_cpu, "structural");
metal_family_test!(create_graph_rnn_gru_cell_metal_matches_cpu, "rnn_gru_cell");
metal_family_test!(create_graph_loss_class_metal_matches_cpu, "loss_class");
metal_family_test!(
    create_graph_loss_regression_metal_matches_cpu,
    "loss_regression"
);

// --- CI で走る自己検査（GPU 不要。登録表と置き場の健全性だけを見る） ---

#[test]
fn registry_names_are_unique() {
    for (i, a) in CASES.iter().enumerate() {
        assert!(
            CASES[i + 1..].iter().all(|b| b.name != a.name),
            "{}",
            a.name
        );
    }
}

#[test]
fn scaffold_case_passes_cpu_against_cpu() {
    run_family("scaffold", None);
}

/// 選択縮約族の登録（`run`・`reject`・`H v ≠ 0` の assert）を CI で一度は実行する。
#[test]
fn select_reduce_minmax_cases_pass_cpu_against_cpu() {
    run_family("select_reduce_minmax", None);
}

#[test]
fn select_reduce_order_cases_pass_cpu_against_cpu() {
    run_family("select_reduce_order", None);
}

/// 統計縮約族の登録（`run`・`reject`・`H v ≠ 0` の assert）を CI で一度は実行する。
#[test]
fn stat_varnorm_cases_pass_cpu_against_cpu() {
    run_family("stat_varnorm", None);
}

#[test]
fn stat_reduce_lse_nan_cases_pass_cpu_against_cpu() {
    run_family("stat_reduce_lse_nan", None);
}

#[test]
fn loss_binary_margin_cases_pass_cpu_against_cpu() {
    run_family("loss_binary_margin", None);
}

#[test]
fn structural_cases_pass_cpu_against_cpu() {
    run_family("structural", None);
}

#[test]
fn rnn_gru_cell_cases_pass_cpu_against_cpu() {
    run_family("rnn_gru_cell", None);
}

#[test]
fn loss_class_cases_pass_cpu_against_cpu() {
    run_family("loss_class", None);
}

#[test]
fn loss_regression_cases_pass_cpu_against_cpu() {
    run_family("loss_regression", None);
}

#[test]
fn baseline_is_an_empty_placeholder_and_fails_closed() {
    assert!(
        baseline::BASELINES.is_empty(),
        "値は実機実測と所有者承認まで入れない"
    );
    assert!(baseline::lookup("scaffold_cubic", baseline::BaselineDevice::Cuda).is_err());
    assert!(baseline::lookup("scaffold_cubic", baseline::BaselineDevice::Metal).is_err());
}

#[test]
fn registry_split_k_declarations_are_well_formed() {
    for c in CASES {
        validate_split_k_decl(&c.split_k).unwrap_or_else(|e| panic!("{}: {e}", c.name));
    }
}

#[test]
fn split_k_declaration_validator_rejects_malformed() {
    assert!(validate_split_k_decl(&SplitKShapes::Listed(&[])).is_err());
    assert!(validate_split_k_decl(&SplitKShapes::Listed(&[(0, 1, 1)])).is_err());
    assert!(validate_split_k_decl(&SplitKShapes::Rule("")).is_err());
    assert!(validate_split_k_decl(&SplitKShapes::Listed(&[(33, 17, 8)])).is_ok());
    assert!(validate_split_k_decl(&SplitKShapes::Rule("k < max(m, n) の形状だけを使う")).is_ok());
    assert!(validate_split_k_decl(&SplitKShapes::NoMatMul).is_ok());
}

/// GPU 不要の純関数検査。形状規則の検査であり測定値ではない（macOS 開発機で走る）。
#[cfg(target_os = "macos")]
#[test]
fn registry_shapes_do_not_fire_split_k() {
    for c in CASES {
        check_split_k_not_fired(c).unwrap_or_else(|e| panic!("{e}"));
    }
}
