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
    // LSTM 2 Op（イシュー #3796・決定記録 §37/§61）。forward の `gemm_bias_act`・`lstm_pointwise` と 1 階の
    // pointwise は CPU・CUDA・Metal ともデバイスカーネルでホストへ落ちない（K）。子テープの構造は MatMul・
    // pointwise・Narrow・Concat で、微分は `h0`・`c0`（公開 API の `x` は定数なので葉 `z: [3, 10]` から切り出す）。
    // 形状は B=3・D=4・H=5（3H=15・4H=20）に固定する。構造とその二階の GEMM の k は最大 20 で、
    // `should_split_k` が要する `k/16 >= min_k_tiles16 (=8)`（k >= 128）に届かないので split-K は発火しない。
    // `lstm_cell_h` は `LstmHidden`→`LstmCell` の連鎖を、`lstm_cell_c` は最後の step で `LstmHidden` の寄与が
    // 無い経路を測る。実機は未実測（#3856・#3858）。
    Case {
        name: "lstm_cell_h",
        family: "lstm_cell",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule(
            "k ∈ {B=3, D=4, H=5, 3H=15, 4H=20} ≤ 20、k/16 = 1 < min_k_tiles16 = 8 のため split-K 非発火",
        ),
        run: run_lstm_cell_h,
        reject: Some(reject_lstm_cell_with_custom),
    },
    Case {
        name: "lstm_cell_c",
        family: "lstm_cell",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule(
            "k ∈ {B=3, D=4, H=5, 3H=15, 4H=20} ≤ 20、k/16 = 1 < min_k_tiles16 = 8 のため split-K 非発火",
        ),
        run: run_lstm_cell_c,
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
    // 正規化・再パラメータ 3 Op（イシュー #3794・決定記録 §36/§57）。3 つとも forward・1 階がホスト関数
    // （H。`BackendOps::{lrn_forward, weight_norm_forward, spectral_norm_forward}` は CPU のみ実装）で、
    // 構造の Pad・Narrow・Mul・Div・Sum・Sqrt・PowScalar がデバイスで走る。構造に MatMul は無い。
    // SpectralNorm は u・v を定数として扱い状態を進めない。実機は未実測（#3856・#3858）。
    Case {
        name: "norm_reparam_lrn",
        family: "norm_reparam",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_norm_reparam_lrn,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "norm_reparam_weight_norm",
        family: "norm_reparam",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_norm_reparam_weight_norm,
        reject: None,
    },
    Case {
        name: "norm_reparam_spectral_norm",
        family: "norm_reparam",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_norm_reparam_spectral_norm,
        reject: None,
    },
    // 3D 畳み込み族（イシュー #3797・決定記録 §38/§59）。`im2col3d`／`col2im3d` は CUDA・Metal・ROCm で
    // override されておらずホストへフォールバックする（GEMM のみデバイスで走る）ので H。d_weight・d_bias の
    // 値は 1 階と同じホスト関数。構造に rank 2 MatMul を含むため `Listed` で子テープ構造の
    // `(m, n, k)`（と、その VJP が積む MatMul）を宣言する。どの形状も縮約長 `k` が 128 未満で
    // `should_split_k` は `None`（`k/16 < 8`）。実機は未実測（#3857・#3859）。
    Case {
        name: "conv3d_family_conv3d",
        family: "conv3d_family",
        route: Route::H,
        judge: Judge::Strict,
        // N=2・Cin=3・Cout=5・kvol=8・P=24: K_g=24・N·P=48。d_input (24,48,5)、d_weight (5,24,48)、
        // およびそれぞれの VJP が積む MatMul。
        split_k: SplitKShapes::Listed(&[
            (24, 48, 5),
            (24, 5, 48),
            (5, 48, 24),
            (5, 24, 48),
            (48, 24, 5),
        ]),
        run: run_conv3d_family_conv3d,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "conv3d_family_conv_transpose3d",
        family: "conv3d_family",
        route: Route::H,
        judge: Judge::Strict,
        // N=2・Cin=3・Cout=4・kvol=8・P_x=18: K_g=32・N·P_x=36。d_input (3,36,32)、d_weight (3,32,36)、
        // およびそれぞれの VJP が積む MatMul。
        split_k: SplitKShapes::Listed(&[(3, 36, 32), (3, 32, 36), (32, 36, 3), (36, 32, 3)]),
        run: run_conv3d_family_conv_transpose3d,
        reject: None,
    },
    // 累積 B 区分（イシュー #3803・決定記録 §41/§64）。`cumprod` は CUDA・Metal にカーネルがあり K、
    // `logcumsumexp` はデバイスカーネルが無くホストへフォールバックする H。構造は narrow／concat／
    // 要素ごとの演算だけ（MatMul なし）で、値ノードは 1 階と同じホスト関数。実機は未実測（#3857・#3859）。
    Case {
        name: "cumulative_b_cumprod",
        family: "cumulative_b",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_cumulative_b_cumprod,
        reject: Some(reject_cumulative_b_cumprod_inf_input),
    },
    Case {
        name: "cumulative_b_logcumsumexp",
        family: "cumulative_b",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_cumulative_b_logcumsumexp,
        reject: Some(reject_cumulative_b_logcumsumexp_inf_input),
    },
    // 索引付き縮約族（イシュー #3802・決定記録 §41/§62）。`indexed_scatter_reduce` は CUDA・Metal・ROCm で
    // override されておらずホストへフォールバックする（`Gather`／`Where`／`Mul` はデバイスで走る）ので H。
    // 構造に MatMul を含まない。値は 1 階と同じホスト関数 `scatter_reduce_vjp_host`。実機は未実測（#3857・#3859）。
    Case {
        name: "scatter_reduce_sum",
        family: "scatter_reduce",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scatter_reduce_sum,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "scatter_reduce_mean",
        family: "scatter_reduce",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scatter_reduce_mean,
        reject: None,
    },
    Case {
        name: "scatter_reduce_amax",
        family: "scatter_reduce",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scatter_reduce_amax,
        reject: None,
    },
    Case {
        name: "scatter_reduce_amin",
        family: "scatter_reduce",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scatter_reduce_amin,
        reject: None,
    },
    // `Prod`（イシュー #3977・決定記録 §74）。区分つき scan 補助の構造（`Gather`／`Narrow`／`Concat`／`Mul`／
    // `Add`／`Where`）で、値は 1 階と同じホスト関数。経路は同じく H。実機は未実測（#3857・#3859）。
    Case {
        name: "scatter_reduce_prod",
        family: "scatter_reduce",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_scatter_reduce_prod,
        reject: None,
    },
    // 短時間フーリエ変換族（イシュー #3804・決定記録 §42/§63）。`fft_stft`／`fft_istft` は CUDA・Metal・ROCm で
    // override されておらずホストへフォールバックし、随伴 Op（`StftAdjoint`／`IstftAdjoint`）もホスト関数
    // なので H。構造に MatMul は無い。実機は未実測（#3857・#3859）。
    Case {
        name: "stft_family_stft",
        family: "stft_family",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stft_family_stft,
        reject: Some(reject_select_reduce_with_custom),
    },
    Case {
        name: "stft_family_istft",
        family: "stft_family",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_stft_family_istft,
        reject: None,
    },
    // 線形代数族（イシュー #3805・決定記録 §43/§65）。`linalg_cholesky`／`linalg_pinv`／`linalg_lstsq` は CUDA・Metal・
    // ROCm のうち override されない経路でホストの `eval::linalg` へフォールバックし（値は 1 階と同じホスト関数の V3）、
    // 構造の `MatMul`／`Solve` 等はデバイスで走るので H。構造の `matmul_fp32_strict` の形状は小さく（k < 128）、
    // Metal の split-K を発火させない。`pinv`・`lstsq` は打ち切りなしの入力だけ（Q5 は打ち切りありの拒否）。
    // 実機は未実測（#3857・#3859）。
    Case {
        name: "linalg_cholesky",
        family: "linalg_cholesky_pinv_lstsq",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(3, 3, 3)]),
        run: run_linalg_cpl_cholesky,
        reject: Some(reject_pinv_truncated),
    },
    Case {
        name: "linalg_pinv_tall",
        family: "linalg_cholesky_pinv_lstsq",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(2, 2, 4), (2, 4, 2), (4, 2, 2)]),
        run: run_linalg_cpl_pinv_tall,
        reject: None,
    },
    Case {
        name: "linalg_pinv_wide",
        family: "linalg_cholesky_pinv_lstsq",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(2, 2, 4), (2, 4, 2), (4, 2, 2)]),
        run: run_linalg_cpl_pinv_wide,
        reject: None,
    },
    Case {
        name: "linalg_lstsq",
        family: "linalg_cholesky_pinv_lstsq",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[
            (2, 2, 4),
            (2, 4, 2),
            (4, 2, 2),
            (2, 4, 1),
            (4, 2, 1),
            (4, 1, 2),
            (2, 1, 4),
            (1, 2, 4),
            (1, 4, 2),
        ]),
        run: run_linalg_cpl_lstsq,
        reject: None,
    },
    // 分解 QR 族（イシュー #3807・決定記録 §44/§67）。`linalg_qr` は CUDA・Metal のデバイスカーネルが
    // opt-in ゲートの後ろ（既定 OFF）で、既定ではホストの `eval::linalg::qr` へフォールバックする（値は 1 階と
    // 同じホスト関数の V3）。構造の `MatMul`／`Solve` 等はデバイスで走るので H。構造の `matmul_fp32_strict` の
    // 形状は小さく（k < 128）、Metal の split-K を発火させない。実機は未実測（#3857・#3859）。
    Case {
        name: "linalg_qr_tall",
        family: "linalg_qr",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(2, 2, 4), (2, 4, 2), (4, 2, 2), (2, 2, 2)]),
        run: run_linalg_qr_tall,
        reject: Some(reject_qr_wide),
    },
    Case {
        name: "linalg_qr_square",
        family: "linalg_qr",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(3, 3, 3)]),
        run: run_linalg_qr_square,
        reject: None,
    },
    // 分解 SVD・Eigh 族（イシュー #3808・決定記録 §44/§68）。`linalg_svd` は CUDA・Metal のデバイスカーネルが
    // 無く、`linalg_eigh` は CUDA のゲートが既定 OFF のため、どちらも既定ではホストの `eval::linalg::{svd,eigh}`
    // へフォールバックする（値は 1 階と同じホスト関数の V3）。構造の `MatMul` 等はデバイスで走るので H。構造の
    // `matmul_fp32_strict` の形状は小さく（k < 128）、Metal の split-K を発火させない。新しいカーネルは作らない。
    // 実機は未実測（#3857・#3859）。
    Case {
        name: "linalg_svd_tall",
        family: "linalg_svd",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(3, 3, 4), (4, 3, 3), (3, 3, 3)]),
        run: run_linalg_svd_tall,
        reject: None,
    },
    Case {
        name: "linalg_svd_wide",
        family: "linalg_svd",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(2, 2, 2), (2, 3, 2), (2, 2, 3), (3, 2, 2)]),
        run: run_linalg_svd_wide,
        reject: None,
    },
    Case {
        name: "linalg_eigh_sym3",
        family: "linalg_eigh",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Listed(&[(3, 3, 3)]),
        run: run_linalg_eigh_sym3,
        reject: Some(reject_eigh_degenerate),
    },
    // 正規化の S0 前スケーリング族（イシュー #3809・決定記録 §45/§69）。`layer_norm` の 1 階 VJP は CUDA・Metal の
    // 正規化逆伝播カーネルを通るので K。子テープの構造（`Div`・`Mean`・`Sub` 等の elementwise と行方向の縮約）は
    // デバイスで走り、`MatMul` を含まない。入力は `[3e38, -3e38, …]` の幅広の行（入口の (b) に当たり S0 が発動する）。
    // 新しいカーネルは作らない。実機は未実測（#3857・#3859）。
    Case {
        name: "norm_prescale_layer_norm_wide",
        family: "norm_prescale",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_norm_prescale_layer_norm,
        reject: Some(reject_layer_norm_const_row_eps0),
    },
    // 行列ノルム・slogdet.sign・matrix_rank 族（イシュー #3806・決定記録 §43/§66）。`linalg_*` は CUDA・Metal・
    // ROCm のうち override されない経路でホストの `eval::linalg` へフォールバックし（値は 1 階と同じホスト関数の V3／
    // ゼロ定数）、構造は `Mul`／`Sum` 等の elementwise のみで MatMul を含まない。実機は未実測（#3857・#3859）。
    Case {
        name: "linalg_matrix_norm_fro",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_linalg_nr_fro,
        reject: Some(reject_matrix_norm_nuc_degenerate),
    },
    Case {
        name: "linalg_matrix_norm_one",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_linalg_nr_one,
        reject: None,
    },
    Case {
        name: "linalg_matrix_norm_inf",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_linalg_nr_inf,
        reject: None,
    },
    // `Nuc`・`Spectral` の値の経路（イシュー #4052・決定記録 §76）。ホストの SVD（`eval::linalg::svd`）へフォールバックし
    // （値は 1 階と同じ `matrix_norm_vjp` の V3）、構造の `MatMul` 等はデバイスで走るので H（`linalg_svd_*` と同じ区分）。
    // 入力は縦長 `[4, 3]`（特異値が約 3・2・1 で互いに離れ、σ == 0 でも縮退でもない）。構造とその二階の GEMM は
    // すべて各次元 4 以下で k/16 = 0 < min_k_tiles16 のため split-K は発火しない。実機は未実測（#4052 で測る）。
    Case {
        name: "linalg_matrix_norm_nuc",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule(
            "全 MatMul の次元が 4 以下（k ≤ 4）で k/16 = 0 < min_k_tiles16 = 8 のため split-K 非発火",
        ),
        run: run_linalg_nr_nuc,
        reject: None,
    },
    Case {
        name: "linalg_matrix_norm_spectral",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::Rule(
            "全 MatMul の次元が 4 以下（k ≤ 4）で k/16 = 0 < min_k_tiles16 = 8 のため split-K 非発火",
        ),
        run: run_linalg_nr_spectral,
        reject: None,
    },
    Case {
        name: "linalg_slogdet_sign",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_linalg_nr_sign,
        reject: None,
    },
    Case {
        name: "linalg_matrix_rank",
        family: "linalg_norm_rank",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_linalg_nr_rank,
        reject: None,
    },
    // Det の行別スケール族（イシュー #3810・決定記録 §46/§70。名前は §48 の表）。`det` の 1 階 VJP は
    // ホストの `eval::linalg::det_vjp`（V3 の値）で、デバイス経路は既定 OFF のためホストへフォールバックする
    // （H）。構造は `Mul`／`Det`／`Inv` 等で MatMul を含まない。孫テープ再生（Q4）も比較する
    // （CPU 側は `det_row_scaled_grandchild_replay_is_finite` でも検査済み。実機は未実測）
    // （§70。#3857・#3859）。
    Case {
        name: "det_family_rowscale_diag_large",
        family: "det_family_rowscale",
        route: Route::H,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_det_family_rowscale,
        reject: Some(reject_det_identity_huge_g),
    },
    // Embedding の `padding_idx` の二階（イシュー #3811・決定記録 §47/§71。名前は §48 の表）。写しを
    // `Op::Embedding` で再生し、子テープ上の 1 階 VJP が行 `p` をマスクする。`gather`／`scatter` は
    // CUDA・Metal の両方で override されているので K。Embedding に固有の create_graph 拒否は無いので reject は
    // なし。孫テープ再生（Q4）も比較する。実機は未実測（#3857・#3859。baseline は作らない）。
    Case {
        name: "embedding_padding_dup_ids",
        family: "embedding_padding",
        route: Route::K,
        judge: Judge::Strict,
        split_k: SplitKShapes::NoMatMul,
        run: run_embedding_padding,
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

/// 正規化・再パラメータ（#3794）。葉 `z`（長さ 12）から出力の二乗和を作り、`run_loss_class` の
/// `loss²` 包みで `H v ≠ 0` にする。
fn run_norm_reparam_lrn(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let y = z
            .reshape(&[1, 4, 3])
            .unwrap()
            .local_response_norm(3, 0.5, 0.75, 1.0)
            .unwrap();
        y.mul(&y).unwrap().sum(None).unwrap()
    })
}

fn run_norm_reparam_weight_norm(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let v = z.narrow(0, 0, 8).unwrap().reshape(&[2, 4]).unwrap();
        let g = z.narrow(0, 8, 2).unwrap().reshape(&[2, 1]).unwrap();
        // `Σ w²` は `g` だけで決まり上流が `w` に平行になって相殺するため `tanh` を挟む。
        let w = v.weight_norm(&g, Some(0)).unwrap();
        w.tanh().sum(None).unwrap()
    })
}

fn run_norm_reparam_spectral_norm(tp: &Tapes) -> Outputs {
    run_loss_class(tp, |z| {
        let w = z.narrow(0, 0, 6).unwrap().reshape(&[2, 3]).unwrap();
        let mut st = fandhe_ai::SpectralNormState::from_vectors(
            &[2, 3],
            0,
            &[0.6, 0.8],
            &[0.5, 0.3, 0.8],
            1,
            1e-12,
        )
        .unwrap();
        // `Σ y²` は上流が `W` に平行で要素ごとの相殺が起きうるため `tanh` を挟む。
        let y = w.spectral_norm(&mut st, false).unwrap();
        y.tanh().sum(None).unwrap()
    })
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

/// LSTM（#3796）。公開 API の `x` は定数なので、葉は `z: [3, 10]`（`h0 = z[:, :5]`・`c0 = z[:, 5:]`）。
/// `Σ y²·c` で `H v ≠ 0` にする。`use_c` が真なら `y = c_n`、偽なら `y = h_n`。決定的な重み（`Lstm::new` のシード）。
fn run_lstm_cell(tp: &Tapes, use_c: bool) -> Outputs {
    let (b, d, h, steps) = (3usize, 4usize, 5usize, 2usize);
    let zv: Vec<f32> = (0..b * 2 * h)
        .map(|i| 0.3 * ((i as f32) * 0.9).sin())
        .collect();
    let xv: Vec<f32> = (0..steps * b * d)
        .map(|i| 0.5 * ((i as f32) * 0.37 + 0.2).cos())
        .collect();
    let x = t(xv, &[steps, b, d]);
    let lstm = fandhe_ai::nn::rnn::Lstm::new(d, h, true, 7).expect("Lstm::new");
    let dir: Vec<f32> = (0..b * 2 * h)
        .map(|i| 1.0 - 0.0625 * i as f32 + 0.03 * (i * i) as f32 / 4.0)
        .collect();
    let z = tp.parent.var(&t(zv, &[b, 2 * h]));
    let h0 = z.narrow(1, 0, h).expect("h0");
    let c0 = z.narrow(1, h, h).expect("c0");
    let out = tp
        .parent
        .lstm_forward_seq(&lstm, &x, Some(&h0), Some(&c0))
        .expect("lstm_forward_seq");
    let y = if use_c { out.c_n } else { out.h_n };
    let c: Vec<f32> = (0..b * h)
        .map(|i| 0.4 + 0.37 * i as f32 - 0.05 * (i * i) as f32 / 4.0)
        .collect();
    let c = tp.parent.var_no_grad(&t(c, &[b, h]));
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
    let v = tp.child.var_no_grad(&t(dir, &[b, 2 * h]));
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

fn run_lstm_cell_h(tp: &Tapes) -> Outputs {
    run_lstm_cell(tp, false)
}

fn run_lstm_cell_c(tp: &Tapes) -> Outputs {
    run_lstm_cell(tp, true)
}

/// Q5（#3796）: LSTM の下流に恒久対象外の `Op::Custom` があれば `Err(Backward)` で、子テープは空のまま。
fn reject_lstm_cell_with_custom(tp: &Tapes) -> bool {
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
    let lstm = fandhe_ai::nn::rnn::Lstm::new(2, 3, true, 7).expect("Lstm::new");
    let x = t(vec![0.1, 0.2, 0.3, 0.4], &[2, 1, 2]);
    let h0 = tp.parent.var(&t(vec![0.2, -0.3, 0.5], &[1, 3]));
    let c0 = tp.parent.var(&t(vec![0.1, 0.4, -0.2], &[1, 3]));
    let h_n = tp
        .parent
        .lstm_forward_seq(&lstm, &x, Some(&h0), Some(&c0))
        .expect("lstm_forward_seq")
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

/// [`run_conv3d_family_body`] の損失 `sum(y ⊙ y ⊙ c)`。葉 `p` から x・w・b を切り出す。
fn conv3d_family_loss<'a>(
    tape: &'a Tape,
    p: &Var<'a>,
    transposed: bool,
    x_shape: [usize; 5],
    w_shape: [usize; 5],
    cout: usize,
) -> Result<Var<'a>, AutodiffError> {
    let (nx, nw) = (
        x_shape.iter().product::<usize>(),
        w_shape.iter().product::<usize>(),
    );
    let x = p.narrow(0, 0, nx)?.reshape(&x_shape)?;
    let w = p.narrow(0, nx, nw)?.reshape(&w_shape)?;
    let b = p.narrow(0, nx + nw, cout)?;
    let y = if transposed {
        x.conv_transpose3d(&w, Some(&b), [1; 3], [0; 3], [0; 3], [1; 3], 1)?
    } else {
        x.conv3d(&w, Some(&b), [1; 3], [0; 3], [1; 3], 1)?
    };
    let ys = y.to_tensor().shape().to_vec();
    let coef: Vec<f32> = (0..ys.iter().product::<usize>())
        .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &ys));
    y.mul(&y)?.mul(&c)?.sum(None)
}

/// 3D 畳み込み族（#3797）の共通本体。x・w・b を 1 本の葉 `p` から切り出し（d_input・d_weight・d_bias を
/// 同時に測る）、損失は `sum(y ⊙ y ⊙ c)`。Q4 は同じ損失の `jvp_create_graph` の出力を孫テープで逆伝播する。
fn run_conv3d_family_body(
    tp: &Tapes,
    transposed: bool,
    x_shape: [usize; 5],
    w_shape: [usize; 5],
    cout: usize,
) -> Outputs {
    let (nx, nw) = (
        x_shape.iter().product::<usize>(),
        w_shape.iter().product::<usize>(),
    );
    let n = nx + nw + cout;
    let p0: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + 11) % 17) as f32 - 8.0) * 0.1)
        .collect();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, &[n]));
    let loss =
        conv3d_family_loss(&tp.parent, &p, transposed, x_shape, w_shape, cout).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &[n]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &[n]), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

/// N=2・Cin=3・Cout=5・入力 `[3, 4, 5]`・kernel `[2, 2, 2]`（2 のべき乗でない中形状）。
fn run_conv3d_family_conv3d(tp: &Tapes) -> Outputs {
    run_conv3d_family_body(tp, false, [2, 3, 3, 4, 5], [5, 3, 2, 2, 2], 5)
}

/// N=2・Cin=3・Cout=4・入力 `[2, 3, 3]`・kernel `[2, 2, 2]`。
fn run_conv3d_family_conv_transpose3d(tp: &Tapes) -> Outputs {
    run_conv3d_family_body(tp, true, [2, 3, 2, 3, 3], [3, 4, 2, 2, 2], 4)
}

/// 累積 B 区分（#3803）の共通本体（S2: 2 のべき乗でない中形状 `[17, 33]`・軸 1 の累積）。
/// 入力は決定的で、`cumprod` が overflow／underflow しない範囲 `[0.8, 1.2]`・`logcumsumexp` が
/// 係数の丸めで外れない範囲に収める（S5）。損失は `sum(y² ⊙ c)`（行ごとの非一様な重み）。
fn run_cumulative_b_body(
    tp: &Tapes,
    scan: for<'a> fn(&fandhe_ai::Var<'a>) -> fandhe_ai::Var<'a>,
    lo: f32,
    span: f32,
) -> Outputs {
    let (rows, cols) = (17usize, 33usize);
    let n = rows * cols;
    let x0: Vec<f32> = (0..n)
        .map(|i| lo + span * (((i * 389 + 17) % 1009) as f32 / 1009.0))
        .collect();
    let dir: Vec<f32> = (0..n).map(|i| 0.5 + 0.01 * (i % 7) as f32).collect();
    let c: Vec<f32> = (0..rows).map(|i| 0.5 + 0.1 * i as f32).collect();

    let x = tp.parent.var(&t(x0, &[rows, cols]));
    let cv = tp.parent.var_no_grad(&t(c, &[rows, 1]));
    let y = scan(&x);
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
    let v = tp.child.var_no_grad(&t(dir.clone(), &[rows, cols]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let hv = vals(grads2.get(&xc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &x, &t(dir, &[rows, cols]), &tp.jvp_child, &tp.grand)
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

fn run_cumulative_b_cumprod(tp: &Tapes) -> Outputs {
    run_cumulative_b_body(tp, |x| x.cumprod(1).unwrap(), 0.8, 0.4)
}

fn run_cumulative_b_logcumsumexp(tp: &Tapes) -> Outputs {
    run_cumulative_b_body(tp, |x| x.logcumsumexp(1).unwrap(), -1.0, 2.0)
}

/// Q5: ±inf の入力は型付きエラーで拒否され、子テープは空のまま（両デバイスで同じ結果にする）。
/// `lse` が真なら Logcumsumexp、偽なら Cumprod を対象にする（族の各 Case が自分の演算を検査する）。
fn reject_cumulative_b_inf_input_op(tp: &Tapes, lse: bool) -> bool {
    let x = tp
        .parent
        .var(&t(vec![0.5, f32::INFINITY, 0.7, 0.9], &[1, 4]));
    let y = if lse { x.logcumsumexp(1) } else { x.cumprod(1) };
    let loss = y.unwrap().sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

fn reject_cumulative_b_cumprod_inf_input(tp: &Tapes) -> bool {
    reject_cumulative_b_inf_input_op(tp, false)
}

fn reject_cumulative_b_logcumsumexp_inf_input(tp: &Tapes) -> bool {
    reject_cumulative_b_inf_input_op(tp, true)
}

/// 索引付き縮約族（#3802）の共通本体。input `[3, 4]` と src `[2, 3]` を 1 本の葉 `p` から切り出し
/// （d_input・d_src を同時に測る）、損失は `sum(y ⊙ y ⊙ c)`。Q4 は同じ損失の `jvp_create_graph` を孫テープで逆伝播する。
fn scatter_reduce_family_loss<'a>(
    tape: &'a Tape,
    p: &Var<'a>,
    mode: fandhe_ai::ScatterReduceMode,
) -> Result<Var<'a>, AutodiffError> {
    let x = p.narrow(0, 0, 12)?.reshape(&[3, 4])?;
    let s = p.narrow(0, 12, 6)?.reshape(&[2, 3])?;
    let idx = Tensor::<i32>::new(vec![1, 1, 3, 0, 2, 2], &[2, 3]).expect("index fixture");
    let y = x.scatter_reduce(1, &idx, &s, mode, true)?;
    let coef: Vec<f32> = (0..12)
        .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &[3, 4]));
    y.mul(&y)?.mul(&c)?.sum(None)
}

fn run_scatter_reduce_body(tp: &Tapes, mode: fandhe_ai::ScatterReduceMode) -> Outputs {
    let n = 18;
    let p0: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + 11) % 17) as f32 - 8.0) * 0.1)
        .collect();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, &[n]));
    let loss = scatter_reduce_family_loss(&tp.parent, &p, mode).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &[n]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &[n]), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

fn run_scatter_reduce_sum(tp: &Tapes) -> Outputs {
    run_scatter_reduce_body(tp, fandhe_ai::ScatterReduceMode::Sum)
}

fn run_scatter_reduce_mean(tp: &Tapes) -> Outputs {
    run_scatter_reduce_body(tp, fandhe_ai::ScatterReduceMode::Mean)
}

fn run_scatter_reduce_amax(tp: &Tapes) -> Outputs {
    run_scatter_reduce_body(tp, fandhe_ai::ScatterReduceMode::Amax)
}

fn run_scatter_reduce_amin(tp: &Tapes) -> Outputs {
    run_scatter_reduce_body(tp, fandhe_ai::ScatterReduceMode::Amin)
}

fn run_scatter_reduce_prod(tp: &Tapes) -> Outputs {
    run_scatter_reduce_body(tp, fandhe_ai::ScatterReduceMode::Prod)
}

/// STFT 族（#3804）の損失 `sum(y ⊙ y ⊙ c)`。`istft` は `[5, 5, 2]` のスペクトル、`stft` は長さ 16 の信号。
fn stft_family_loss<'a>(
    tape: &'a Tape,
    p: &Var<'a>,
    inverse: bool,
) -> Result<Var<'a>, AutodiffError> {
    let y = if inverse {
        let x = p.reshape(&[5, 5, 2])?;
        x.istft(
            8,
            None,
            &fandhe_ai::IstftOptions::default().with_hop_length(2),
        )?
    } else {
        p.stft(
            8,
            None,
            &fandhe_ai::StftOptions::default().with_hop_length(2),
        )?
    };
    let ys = y.to_tensor().shape().to_vec();
    let coef: Vec<f32> = (0..ys.iter().product::<usize>())
        .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &ys));
    y.mul(&y)?.mul(&c)?.sum(None)
}

/// STFT 族（#3804）の共通本体。Q4 は同じ損失の `jvp_create_graph` の出力を孫テープで逆伝播する。
fn run_stft_family_body(tp: &Tapes, inverse: bool, n: usize) -> Outputs {
    let p0: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + 11) % 17) as f32 - 8.0) * 0.1)
        .collect();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, &[n]));
    let loss = stft_family_loss(&tp.parent, &p, inverse).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &[n]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &[n]), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

/// 長さ 16 の信号（`n_fft = 8`・`hop = 2`・`center`）。
fn run_stft_family_stft(tp: &Tapes) -> Outputs {
    run_stft_family_body(tp, false, 16)
}

/// `[5, 5, 2]` のスペクトル（`n_fft = 8`・`hop = 2`・`center`）。
fn run_stft_family_istft(tp: &Tapes) -> Outputs {
    run_stft_family_body(tp, true, 50)
}

/// 線形代数族（#3805）の損失 `sum(y ⊙ y ⊙ c)`。`kind` は 0 = cholesky（`[3,3]`）、1 = pinv 縦長（`[4,2]`）、
/// 2 = pinv 横長（`[2,4]`）、3 = lstsq（A `[4,2]` と B `[4,1]` を 1 本の長さ 12 のベクトルにまとめる）。
fn linalg_cpl_loss<'a>(tape: &'a Tape, p: &Var<'a>, kind: usize) -> Result<Var<'a>, AutodiffError> {
    let y = match kind {
        0 => {
            let half = tape.var_no_grad(&t(vec![0.5; 9], &[3, 3]));
            let mut eye = vec![0.0f32; 9];
            for i in 0..3 {
                eye[i * 3 + i] = 4.0;
            }
            let eye = tape.var_no_grad(&t(eye, &[3, 3]));
            p.add(&p.transpose(0, 1)?)?
                .mul(&half)?
                .add(&eye)?
                .cholesky()?
        }
        1 | 2 => p.pinv(None)?,
        _ => {
            let a = p.narrow(0, 0, 8)?.reshape(&[4, 2])?;
            let b = p.narrow(0, 8, 4)?.reshape(&[4, 1])?;
            a.lstsq(&b, None)?
        }
    };
    let ys = y.to_tensor().shape().to_vec();
    let coef: Vec<f32> = (0..ys.iter().product::<usize>())
        .map(|i| 0.5 + ((i * 13 + 5) % 11) as f32 * 0.1)
        .collect();
    let c = tape.var_no_grad(&t(coef, &ys));
    y.mul(&y)?.mul(&c)?.sum(None)
}

/// 線形代数族（#3805）の共通本体。入力は条件の良い（打ち切りなしの）値。Q4 は同じ損失の
/// `jvp_create_graph` の出力を孫テープで逆伝播する。
fn run_linalg_cpl_body(tp: &Tapes, kind: usize) -> Outputs {
    let (shape, p0): (Vec<usize>, Vec<f32>) = match kind {
        0 => (
            vec![3, 3],
            vec![0.3, -0.5, 0.2, 0.1, 0.4, -0.2, 0.6, 0.0, -0.3],
        ),
        1 => (vec![4, 2], vec![2.0, 0.3, -0.4, 1.5, 0.2, -0.6, 0.5, 0.9]),
        2 => (vec![2, 4], vec![2.0, 0.3, -0.4, 0.5, 0.2, -0.6, 1.5, 0.9]),
        _ => (
            vec![12],
            vec![
                2.0, 0.3, -0.4, 1.5, 0.2, -0.6, 0.5, 0.9, 0.7, -0.2, 0.5, 0.1,
            ],
        ),
    };
    let n: usize = shape.iter().product();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, &shape));
    let loss = linalg_cpl_loss(&tp.parent, &p, kind).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &shape));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &shape), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

fn run_linalg_cpl_cholesky(tp: &Tapes) -> Outputs {
    run_linalg_cpl_body(tp, 0)
}

fn run_linalg_cpl_pinv_tall(tp: &Tapes) -> Outputs {
    run_linalg_cpl_body(tp, 1)
}

fn run_linalg_cpl_pinv_wide(tp: &Tapes) -> Outputs {
    run_linalg_cpl_body(tp, 2)
}

fn run_linalg_cpl_lstsq(tp: &Tapes) -> Outputs {
    run_linalg_cpl_body(tp, 3)
}

/// 分解 QR 族（#3807）の損失 `sum(tanh(Q) ⊙ c1) + sum(R ⊙ R ⊙ c2)`（`Q`・`R` の両ノードを使う）。
fn linalg_qr_loss<'a>(tape: &'a Tape, p: &Var<'a>) -> Result<Var<'a>, AutodiffError> {
    let f = p.qr()?;
    let weights = |shape: &[usize], phase: usize| {
        let coef: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| 0.5 + ((i * 13 + phase) % 11) as f32 * 0.1)
            .collect();
        tape.var_no_grad(&t(coef, shape))
    };
    let qs = f.q.to_tensor().shape().to_vec();
    let rs = f.r.to_tensor().shape().to_vec();
    let lq = f.q.tanh().mul(&weights(&qs, 5))?.sum(None)?;
    let lr = f.r.mul(&f.r)?.mul(&weights(&rs, 3))?.sum(None)?;
    lq.add(&lr)
}

/// 分解 QR 族（#3807）の共通本体。入力は条件の良いフルランクの縦長／正方。Q4 は同じ損失の
/// `jvp_create_graph` の出力を孫テープで逆伝播する。
fn run_linalg_qr_body(tp: &Tapes, shape: &[usize], p0: Vec<f32>) -> Outputs {
    run_linalg_decomp_body(tp, shape, p0, linalg_qr_loss)
}

type DecompLoss = for<'a> fn(&'a Tape, &Var<'a>) -> Result<Var<'a>, AutodiffError>;

/// 分解族（#3807・#3808）の共通本体。損失だけを差し替える。
fn run_linalg_decomp_body(
    tp: &Tapes,
    shape: &[usize],
    p0: Vec<f32>,
    loss_fn: DecompLoss,
) -> Outputs {
    let n: usize = shape.iter().product();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, shape));
    let loss = loss_fn(&tp.parent, &p).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), shape));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, shape), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

fn run_linalg_qr_tall(tp: &Tapes) -> Outputs {
    run_linalg_qr_body(tp, &[4, 2], vec![2.0, 0.3, -0.4, 1.5, 0.2, -0.6, 0.5, 0.9])
}

fn run_linalg_qr_square(tp: &Tapes) -> Outputs {
    run_linalg_qr_body(
        tp,
        &[3, 3],
        vec![2.0, 0.3, -0.4, 0.1, 1.5, 0.2, -0.6, 0.5, 1.8],
    )
}

/// 分解 SVD 族（#3808）の損失 `sum(tanh(U) ⊙ c1) + sum(s ⊙ s) + sum(tanh(Vh) ⊙ c2)`（3 ノード全てを使う）。
fn linalg_svd_loss<'a>(tape: &'a Tape, p: &Var<'a>) -> Result<Var<'a>, AutodiffError> {
    let f = p.svd()?;
    let weights = |v: &Var<'a>, phase: usize| {
        let shape = v.to_tensor().shape().to_vec();
        let coef: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| 0.5 + ((i * 13 + phase) % 11) as f32 * 0.1)
            .collect();
        tape.var_no_grad(&t(coef, &shape))
    };
    let lu = f.u.tanh().mul(&weights(&f.u, 5))?.sum(None)?;
    let ls = f.s.mul(&f.s)?.sum(None)?;
    let lv = f.vh.tanh().mul(&weights(&f.vh, 3))?.sum(None)?;
    lu.add(&ls)?.add(&lv)
}

/// 分解 Eigh 族（#3808）の損失 `sum(tanh(λ) ⊙ c1) + sum(V ⊙ V ⊙ c2)`（2 ノードを使う）。
fn linalg_eigh_loss<'a>(tape: &'a Tape, p: &Var<'a>) -> Result<Var<'a>, AutodiffError> {
    let f = p.eigh()?;
    let weights = |v: &Var<'a>, phase: usize| {
        let shape = v.to_tensor().shape().to_vec();
        let coef: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| 0.5 + ((i * 13 + phase) % 11) as f32 * 0.1)
            .collect();
        tape.var_no_grad(&t(coef, &shape))
    };
    let ll = f
        .eigenvalues
        .tanh()
        .mul(&weights(&f.eigenvalues, 5))?
        .sum(None)?;
    let lv = f
        .eigenvectors
        .mul(&f.eigenvectors)?
        .mul(&weights(&f.eigenvectors, 3))?
        .sum(None)?;
    ll.add(&lv)
}

fn run_linalg_svd_tall(tp: &Tapes) -> Outputs {
    run_linalg_decomp_body(
        tp,
        &[4, 3],
        vec![
            3.0, 0.3, -0.4, 0.1, 2.0, 0.2, -0.6, 0.5, 1.0, 0.2, -0.1, 0.3,
        ],
        linalg_svd_loss,
    )
}

fn run_linalg_svd_wide(tp: &Tapes) -> Outputs {
    run_linalg_decomp_body(
        tp,
        &[2, 3],
        vec![3.0, 0.3, -0.4, 0.1, 2.0, 0.2],
        linalg_svd_loss,
    )
}

fn run_linalg_eigh_sym3(tp: &Tapes) -> Outputs {
    run_linalg_decomp_body(
        tp,
        &[3, 3],
        vec![4.0, 0.3, 0.2, 0.3, 2.5, 0.1, 0.2, 0.1, 1.0],
        linalg_eigh_loss,
    )
}

/// Q5: 縮退した固有値（`eigh(I2)`）は固有値だけの損失でも fail-closed で拒否され、子テープは空のまま。
fn reject_eigh_degenerate(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x.eigh().unwrap().eigenvalues.sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// 正規化 S0 族（#3809）の損失 `sum(layer_norm(x; w, b) ⊙ c)`（`w`・`b`・`c` は定数）。
fn norm_prescale_loss<'a>(tape: &'a Tape, p: &Var<'a>) -> Result<Var<'a>, AutodiffError> {
    let w = tape.var_no_grad(&t(vec![1.2, 0.7, -0.4, 0.9], &[4]));
    let b = tape.var_no_grad(&t(vec![0.1, -0.2, 0.3, 0.0], &[4]));
    let c = tape.var_no_grad(&t(
        (0..12)
            .map(|i| 0.5 + ((i * 7 + 3) % 5) as f32 * 0.3)
            .collect(),
        &[3, 4],
    ));
    p.layer_norm(Some(&w), Some(&b), 1e-5)?.mul(&c)?.sum(None)
}

/// 行 0 が (b)（`x_i − x_0` が f32 で非有限）、行 1 が (c)（部分和 overflow）に当たる `[3, 4]`。行 2 は
/// 通常の値で、幅広の行の Hessian は f32 では 0 へ潰れる（`~1/x²`）ため、二階の値の検出力はここが担う。
fn run_norm_prescale_layer_norm(tp: &Tapes) -> Outputs {
    run_linalg_decomp_body(
        tp,
        &[3, 4],
        vec![
            3e38, -3e38, 3e38, -3e38, -2e38, 1e38, 1e38, 0.0, 0.8, -0.3, 1.4, 0.2,
        ],
        norm_prescale_loss,
    )
}

/// Det 行別スケール族（#3810）。`diag(1e20, 1e20)` の `det`（f32 を超える行列式）。Hv は
/// `d²det/dA²` の方向微分で、非対角の成分がゼロにならない方向を使う。
fn run_det_family_rowscale(tp: &Tapes) -> Outputs {
    let shape = [2usize, 2];
    let dir = vec![0.6f32, -0.23, 0.41, 0.3];
    let p = tp.parent.var(&t(vec![1e20, 0.0, 0.0, 1e20], &shape));
    let loss = p.det().expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));
    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &shape));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。行別スケール入力でも有限に受理される（§70.3 の逸脱 1）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &shape), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

/// Embedding の `padding_idx` 族（#3811）。重複 id と `padding_idx` 行を含む `[4, 2]` の重み。
fn embedding_padding_loss<'a>(_tape: &'a Tape, p: &Var<'a>) -> Result<Var<'a>, AutodiffError> {
    let w = p.reshape(&[4, 2])?;
    let ids = Tensor::<i32>::new(vec![1, 1, 3, 0, 2], &[5])?;
    let y = w.embedding(&ids, Some(1))?;
    let s = y.sum(None)?;
    y.mul(&y)?.mul(&y)?.sum(None)?.add(&s.mul(&s)?)
}

fn run_embedding_padding(tp: &Tapes) -> Outputs {
    run_linalg_decomp_body(
        tp,
        &[8],
        (0..8)
            .map(|i| 0.45 * ((i * 5 + 2) % 8) as f32 - 0.7)
            .collect(),
        embedding_padding_loss,
    )
}

/// Q5: 真の 1 階値が f32 を超える `det`（`A = I`・`g = 3e38`）は拒否され、子テープは空のまま。
fn reject_det_identity_huge_g(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let big = tp.parent.var_no_grad(&t(vec![3e38], &[]));
    let loss = x.det().unwrap().mul(&big).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// Q5: `eps = 0` の定数行は S0 でも fail-closed で拒否され、子テープは空のまま。
fn reject_layer_norm_const_row_eps0(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![2.0, 2.0, 2.0], &[1, 3]));
    let loss = x.layer_norm(None, None, 0.0).unwrap().sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// 行列ノルム族（#3806）の損失。`kind` は 0 = Fro・1 = One・2 = Inf（いずれも `[3,2]`。One／Inf はタイから離れた入力）、
/// 3 = `sign · exp(logabsdet)` の二乗（`[3,3]`）、4 = `rank · Σx²`（`[3,3]`）、
/// 5 = Nuc・6 = Spectral（#4052。いずれも縦長 `[4,3]` で特異値が約 3・2・1 と離れた縮退しない入力）。
fn linalg_nr_loss<'a>(p: &Var<'a>, kind: usize) -> Result<Var<'a>, AutodiffError> {
    let y = match kind {
        0 => p.matrix_norm(fandhe_ai::MatrixNormOrd::Fro)?,
        1 => p.matrix_norm(fandhe_ai::MatrixNormOrd::One)?,
        2 => p.matrix_norm(fandhe_ai::MatrixNormOrd::Inf)?,
        5 => p.matrix_norm(fandhe_ai::MatrixNormOrd::Nuc)?,
        6 => p.matrix_norm(fandhe_ai::MatrixNormOrd::Spectral)?,
        3 => {
            let s = p.slogdet()?;
            s.sign.mul(&s.logabsdet.exp())?
        }
        _ => {
            let r = p.matrix_rank(None)?;
            return r.mul(&p.mul(p)?.sum(None)?)?.sum(None);
        }
    };
    y.mul(&y)?.sum(None)
}

/// 行列ノルム族（#3806）の共通本体。Q4 は同じ損失の `jvp_create_graph` の出力を孫テープで逆伝播する。
fn run_linalg_nr_body(tp: &Tapes, kind: usize) -> Outputs {
    let (shape, p0): (Vec<usize>, Vec<f32>) = match kind {
        0 => (vec![3, 2], vec![2.0, 0.3, -0.4, 1.5, 0.2, -0.6]),
        1 | 2 => (vec![3, 2], vec![1.0, -2.0, 3.0, 0.5, -0.7, 1.1]),
        5 | 6 => (
            vec![4, 3],
            vec![
                3.0, 0.3, -0.4, 0.1, 2.0, 0.2, -0.6, 0.5, 1.0, 0.2, -0.1, 0.3,
            ],
        ),
        _ => (
            vec![3, 3],
            vec![2.0, 0.3, -0.4, 0.1, 1.5, 0.2, -0.6, 0.5, 1.8],
        ),
    };
    let n: usize = shape.iter().product();
    let dir: Vec<f32> = (0..n)
        .map(|i| 0.6 - 0.23 * (i % 13) as f32 + 0.04 * (i % 5) as f32)
        .collect();

    let p = tp.parent.var(&t(p0, &shape));
    let loss = linalg_nr_loss(&p, kind).expect("forward");
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&p).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&p).unwrap().expect("first_order"));
    let g = cg.grad(&p).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir.clone(), &shape));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let pc = cg.child_var(&p).unwrap().expect("mirror");
    let hv = vals(grads2.get(&pc).unwrap().expect("second order"));
    // Q4: 孫テープ再生（§48.4）。
    let res = tp
        .parent
        .jvp_create_graph(&loss, &p, &t(dir, &shape), &tp.jvp_child, &tp.grand)
        .expect("jvp_create_graph");
    let rgrads = tp
        .grand
        .backward(&res.output().sum(None).unwrap())
        .expect("grand backward");
    let rp = res.child_var(&p).unwrap().expect("mirror");
    let replay = vals(rgrads.get(&rp).unwrap().expect("replay grad"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: Some(replay),
    }
}

fn run_linalg_nr_fro(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 0)
}

fn run_linalg_nr_one(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 1)
}

fn run_linalg_nr_inf(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 2)
}

fn run_linalg_nr_nuc(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 5)
}

fn run_linalg_nr_spectral(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 6)
}

fn run_linalg_nr_sign(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 3)
}

fn run_linalg_nr_rank(tp: &Tapes) -> Outputs {
    run_linalg_nr_body(tp, 4)
}

/// Q5: `matrix_norm` の `Nuc`（#3991 で対象化。特異値が縮退する単位行列は拒否）は GPU でも `Err(Backward)` で拒否され、
/// 子テープは空のまま。
fn reject_matrix_norm_nuc_degenerate(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x
        .matrix_norm(fandhe_ai::MatrixNormOrd::Nuc)
        .unwrap()
        .sum(None)
        .unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// Q5: 打ち切りあり（`A = diag(2, 1)`・`rcond = 0.75`）の `pinv` は GPU でも `Err(Backward)` で拒否され、
/// 子テープは空のまま。
fn reject_pinv_truncated(tp: &Tapes) -> bool {
    let x = tp.parent.var(&t(vec![2.0, 0.0, 0.0, 1.0], &[2, 2]));
    let loss = x.pinv(Some(0.75)).unwrap().sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
}

/// Q5: 横長（`m < n`）の `qr` は GPU でも `Err(Backward)` で拒否され、子テープは空のまま。
fn reject_qr_wide(tp: &Tapes) -> bool {
    let x = tp
        .parent
        .var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[2, 3]));
    let loss = x.qr().unwrap().r.sum(None).unwrap();
    is_clean_rejection(tp.parent.backward_create_graph(&loss, &tp.child), &tp.child)
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
cuda_family_test!(create_graph_lstm_cell_cuda_matches_cpu, "lstm_cell");
cuda_family_test!(create_graph_loss_class_cuda_matches_cpu, "loss_class");
cuda_family_test!(
    create_graph_loss_regression_cuda_matches_cpu,
    "loss_regression"
);
cuda_family_test!(create_graph_norm_reparam_cuda_matches_cpu, "norm_reparam");
cuda_family_test!(create_graph_conv3d_family_cuda_matches_cpu, "conv3d_family");
cuda_family_test!(create_graph_cumulative_b_cuda_matches_cpu, "cumulative_b");
cuda_family_test!(
    create_graph_scatter_reduce_cuda_matches_cpu,
    "scatter_reduce"
);
cuda_family_test!(create_graph_stft_family_cuda_matches_cpu, "stft_family");
cuda_family_test!(
    create_graph_linalg_cholesky_pinv_lstsq_cuda_matches_cpu,
    "linalg_cholesky_pinv_lstsq"
);
cuda_family_test!(
    create_graph_linalg_norm_rank_cuda_matches_cpu,
    "linalg_norm_rank"
);
cuda_family_test!(create_graph_linalg_qr_cuda_matches_cpu, "linalg_qr");
cuda_family_test!(create_graph_linalg_svd_cuda_matches_cpu, "linalg_svd");
cuda_family_test!(create_graph_linalg_eigh_cuda_matches_cpu, "linalg_eigh");
cuda_family_test!(create_graph_norm_prescale_cuda_matches_cpu, "norm_prescale");
cuda_family_test!(
    create_graph_det_family_rowscale_cuda_matches_cpu,
    "det_family_rowscale"
);
cuda_family_test!(
    create_graph_embedding_padding_cuda_matches_cpu,
    "embedding_padding"
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
metal_family_test!(create_graph_lstm_cell_metal_matches_cpu, "lstm_cell");
metal_family_test!(create_graph_loss_class_metal_matches_cpu, "loss_class");
metal_family_test!(
    create_graph_loss_regression_metal_matches_cpu,
    "loss_regression"
);
metal_family_test!(create_graph_norm_reparam_metal_matches_cpu, "norm_reparam");
metal_family_test!(
    create_graph_conv3d_family_metal_matches_cpu,
    "conv3d_family"
);
metal_family_test!(create_graph_cumulative_b_metal_matches_cpu, "cumulative_b");
metal_family_test!(
    create_graph_scatter_reduce_metal_matches_cpu,
    "scatter_reduce"
);
metal_family_test!(create_graph_stft_family_metal_matches_cpu, "stft_family");
metal_family_test!(
    create_graph_linalg_cholesky_pinv_lstsq_metal_matches_cpu,
    "linalg_cholesky_pinv_lstsq"
);
metal_family_test!(
    create_graph_linalg_norm_rank_metal_matches_cpu,
    "linalg_norm_rank"
);
metal_family_test!(create_graph_linalg_qr_metal_matches_cpu, "linalg_qr");
metal_family_test!(create_graph_linalg_svd_metal_matches_cpu, "linalg_svd");
metal_family_test!(create_graph_linalg_eigh_metal_matches_cpu, "linalg_eigh");
metal_family_test!(
    create_graph_norm_prescale_metal_matches_cpu,
    "norm_prescale"
);
metal_family_test!(
    create_graph_det_family_rowscale_metal_matches_cpu,
    "det_family_rowscale"
);
metal_family_test!(
    create_graph_embedding_padding_metal_matches_cpu,
    "embedding_padding"
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
fn lstm_cell_cases_pass_cpu_against_cpu() {
    run_family("lstm_cell", None);
}

#[test]
fn loss_class_cases_pass_cpu_against_cpu() {
    run_family("loss_class", None);
}

#[test]
fn norm_reparam_cases_pass_cpu_against_cpu() {
    run_family("norm_reparam", None);
}

#[test]
fn loss_regression_cases_pass_cpu_against_cpu() {
    run_family("loss_regression", None);
}

#[test]
fn conv3d_family_cases_pass_cpu_against_cpu() {
    run_family("conv3d_family", None);
}

#[test]
fn cumulative_b_cases_pass_cpu_against_cpu() {
    run_family("cumulative_b", None);
}

#[test]
fn scatter_reduce_cases_pass_cpu_against_cpu() {
    run_family("scatter_reduce", None);
}

#[test]
fn stft_family_cases_pass_cpu_against_cpu() {
    run_family("stft_family", None);
}

#[test]
fn linalg_cholesky_pinv_lstsq_cases_pass_cpu_against_cpu() {
    run_family("linalg_cholesky_pinv_lstsq", None);
}

#[test]
fn linalg_norm_rank_cases_pass_cpu_against_cpu() {
    run_family("linalg_norm_rank", None);
}

#[test]
fn linalg_qr_cases_pass_cpu_against_cpu() {
    run_family("linalg_qr", None);
}

#[test]
fn linalg_svd_cases_pass_cpu_against_cpu() {
    run_family("linalg_svd", None);
}

#[test]
fn linalg_eigh_cases_pass_cpu_against_cpu() {
    run_family("linalg_eigh", None);
}

#[test]
fn norm_prescale_cases_pass_cpu_against_cpu() {
    run_family("norm_prescale", None);
}

#[test]
fn det_family_rowscale_cases_pass_cpu_against_cpu() {
    run_family("det_family_rowscale", None);
}

#[test]
fn embedding_padding_cases_pass_cpu_against_cpu() {
    run_family("embedding_padding", None);
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
