//! Transformer 複合ワークロードのベンチ定義（単一真実源。TASK-8.4/Phase G-4・イシュー #589）。
//!
//! 親 #582（Phase G「融合 RMSNorm / online softmax / 量子化キャストによる Transformer 複合
//! ワークロード改善」）は REQ-8 の GEMM 単体 5 行（[`crate::threshold::floor_spec`] が判定する
//! CPU／CUDA f32・f16／Metal f32・f16 の各行）とは**別行**である「Transformer 複合ワークロード」
//! を評価軸とする。本モジュールはその評価軸の分母（ワークロード形状・決定的シード）を
//! バックエンド非依存のデータとして 1 箇所に集約し、既存の実測実装
//! （`crates/bench-harness/tests/transformer_workload.rs`。#155・TASK-8.3a）と、
//! ベースライン文書（`docs/perf/transformer-workload-baseline.md`。#589）の双方から参照される
//! 単一真実源とする。
//!
//! ## 系列の分離（重要）
//!
//! 本モジュールが保持する形状・ベンチ名は [`crate::threshold::floor_spec`]／[`crate::threshold::judge`]
//! の合否判定対象**ではない**。`docs/performance-targets.md` §2 の「Transformer 複合ワークロード」行は
//! 「初期リリース: 下限を設定しない／最適化後: 下限を設定しない」（REQ-8 の受け入れ基準）であり、
//! この行に対して本モジュールが下限値を持つことはない。Phase G の評価は実機ベースライン
//! （G-12・#602・CUDA 実機／G-14・#605・Metal 実機）からの**相対改善**で行う
//! （`docs/perf/transformer-workload-baseline.md` §5「評価方式」参照）。
//!
//! ## 定数の出典
//!
//! 形状定数（`d_model`・`n_heads`・`d_ff`・`batch`・`seq_len`）と `SEED` は PoC-8 定義
//! （PoC-5 流用）を踏襲し、#155 実装当時に `crates/bench-harness/tests/transformer_workload.rs`
//! がテストファイル内ローカル定数として保持していた値と同値である（挙動不変の単一真実源化）。
//! 決定的シード値にイシュー番号（`155_083`）を含めるのは、他計測（`determinism.rs` の `2026`）
//! との衝突を避けるため（#155 由来の判断を継承）。
//!
//! ## 計測プロトコルとの関係
//!
//! 本モジュールは形状・シードのみを保持し、計測プロトコル（warmup／iters 下限 20/20・
//! 中央値・Q1/Q3・`BenchReport` への構造化。`docs/performance-targets.md` §4）は
//! [`crate::protocol::MeasurementConfig`]／[`crate::protocol::run`]／[`crate::report::BenchReport`]
//! （変更しない既存実装）が引き続き担う。

/// Transformer ブロック 1 層 forward のワークロード形状。
///
/// `crates/bench-harness/tests/transformer_workload.rs` の実測実装（attention 込み・post-norm・
/// GELU）が本構造体の各フィールドと 1 対 1 で対応する。バックエンド（CPU／CUDA／Metal）・
/// 精度（f32 のみ。本ワークロードは f16 版を持たない）に依存しない共通形状のため、
/// `backend-cpu`／`onnx-interop` 等への依存は持たない（純粋データ）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformerWorkloadSpec {
    /// Transformer ブロックの層数。#589 時点の確定値は 1 層（PoC-8 定義）。
    pub num_layers: usize,
    /// 隠れ層次元。
    pub d_model: usize,
    /// Multi-Head Attention のヘッド数。`d_model % n_heads == 0` を満たす。
    pub n_heads: usize,
    /// FFN 中間層次元。
    pub d_ff: usize,
    /// バッチサイズ。
    pub batch: usize,
    /// 系列長。
    pub seq_len: usize,
    /// LayerNormalization の epsilon（`onnx_interop::ops::LayerNormAttrs::epsilon` に渡す値）。
    pub layer_norm_eps: f32,
    /// Multi-Head Self-Attention サブレイヤーを含むかどうか。
    /// #589 時点の確定値は `true`（既存実測実装が Q/K/V・softmax・スケーリングを含む構成のため）。
    pub has_attention: bool,
}

impl TransformerWorkloadSpec {
    /// `n_heads` あたりのヘッド次元（`d_model / n_heads`）。
    ///
    /// `multi_head_attention`（実測実装側）の `to_heads` クロージャが
    /// `[batch, seq, n_heads, head_dim]` への reshape に使う値と同一の導出式。
    ///
    /// 全フィールドが公開されているため利用者が `n_heads == 0` や
    /// `d_model % n_heads != 0` の値を構築できる（コンストラクタ非経由）。
    /// 本番経路で `panic` させない方針（`.claude/rules/coding-rust.md`）に従い、
    /// 両不変条件（ゼロ除算・割り切れ）を満たさない場合は `None` を返す
    /// 型付き失敗にする（codex-review 指摘・PR #647 P1）。
    pub const fn head_dim(&self) -> Option<usize> {
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return None;
        }
        Some(self.d_model / self.n_heads)
    }
}

/// 確定済みベースライン形状（#589・Phase G-4 で確定。`docs/perf/transformer-workload-baseline.md` §2 と同値）。
///
/// `d_model=512, n_heads=8 (head_dim=64), d_ff=2048, batch=8, seq_len=128, num_layers=1`・
/// f32・attention あり・post-norm（`Add → LayerNormalization`、eps=1e-5）・GELU（erf 合成）。
/// PoC-8 定義（PoC-5 流用）を単一真実源としてそのまま踏襲する（値の変更は本イシューのスコープ外）。
pub const fn baseline_spec() -> TransformerWorkloadSpec {
    TransformerWorkloadSpec {
        num_layers: 1,
        d_model: 512,
        n_heads: 8,
        d_ff: 2048,
        batch: 8,
        seq_len: 128,
        layer_norm_eps: 1e-5,
        has_attention: true,
    }
}

/// 決定的シード（#155 由来。入力・重み生成の双方に使う。`crate::rng::Xorshift64Star::new` に渡す値）。
///
/// イシュー番号を値に含め、他計測（`determinism.rs` の `2026`）との衝突を避ける
/// （`crates/bench-harness/tests/transformer_workload.rs` 冒頭コメントを踏襲）。
pub const SEED: u64 = 155_083;

/// ベンチ名の接頭辞。`report_name` がバックエンド名と組み合わせて
/// [`crate::BenchReport::from_measurement`] の `name` 引数を構築する際に使う
/// （既存の `"transformer-block-forward-cpu-blis"` 系ベンチ名との命名系列の単一真実源化）。
pub const BENCH_NAME_PREFIX: &str = "transformer-block-forward";

/// バックエンド名（例: `"cpu"`・`"cuda"`・`"metal"`）からベンチ名を構築する。
///
/// GEMM 単体ベンチ（[`crate::threshold::floor_spec`] の判定対象）とは別系列のベンチ名空間である
/// ことを明示するため、接頭辞を [`BENCH_NAME_PREFIX`] に統一する薄いヘルパー。
///
/// `"cpu"` バックエンドのみ `"-blis"` サフィックスを付与する。CPU 経路
/// （[`crate::transformer_workload`] の実測が使う `CpuBackendOps`）は
/// `gemm`・`gemm_bias_act` の両方とも常に `gemm_blis` 系カーネル
/// （`gemm_blis_parallel`・`gemm_blis_bias_act_parallel`）へディスパッチする
/// 契約（`crates/backend-cpu/src/ops.rs` 冒頭コメント）のため、他バックエンド
/// 追加時に選択肢が増えうる汎用パラメータではなく、CPU バックエンドの実装
/// 系列そのものを表す固定サフィックスとして扱う（codex-review 指摘・PR #647:
/// 従来 `report_name("cpu")` の結果に `"-blis"` が欠落し、実測経路がハード
/// コードの `"transformer-block-forward-cpu-blis"` を直書きしていた設計不整合
/// の解消）。CUDA／Metal 側のカーネル実装が確定した際に同様の命名要否を
/// 個別に判断する（PoC-v2-5 の cfg ベースバックエンド構成では実装が単一の
/// ため、現時点で `cuda`／`metal` に対応する固定サフィックスは設けない）。
pub fn report_name(backend: &str) -> String {
    match backend {
        "cpu" => format!("{BENCH_NAME_PREFIX}-cpu-blis"),
        _ => format!("{BENCH_NAME_PREFIX}-{backend}"),
    }
}

/// [`report_name`] の「Phase G 適用後（融合あり経路）」版ベンチ名。
///
/// イシュー #602（G-12）が Phase G（融合 RMSNorm・online softmax・
/// `gemm_bias_act` epilogue 融合。親 #582）適用前後の相対改善を計測する
/// ため、CUDA 実測経路（`crates/bench-harness/tests/transformer_workload.rs`）
/// は同一ワークロード形状に対し [`report_name`]（改善前＝非融合経路）と
/// 本関数（改善後＝融合経路）の 2 種類のベンチ名を使い分ける。単純に
/// `"-fused"` サフィックスを付与するだけの薄いヘルパーだが、実測経路の
/// 直書きを避け [`BENCH_NAME_PREFIX`] 変更への追従を保つ単一真実源として
/// [`report_name`] と同じ理由で切り出す。
pub fn report_name_fused(backend: &str) -> String {
    format!("{}-fused", report_name(backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ドリフト防止テスト: `baseline_spec()` の全フィールドが
    /// `docs/perf/transformer-workload-baseline.md` §2 記載の確定値と一致することを検査する。
    /// 文書とコードのどちらか一方だけが変更された場合にこのテストが fail することで、
    /// 単一真実源としての一致を保つ（`.claude/rules/code-comment-style.md`: コードと
    /// 二重管理しない一方、値の整合自体は自動検査で担保する）。
    #[test]
    fn baseline_spec_matches_documented_values() {
        let spec = baseline_spec();
        assert_eq!(spec.num_layers, 1);
        assert_eq!(spec.d_model, 512);
        assert_eq!(spec.n_heads, 8);
        assert_eq!(spec.d_ff, 2048);
        assert_eq!(spec.batch, 8);
        assert_eq!(spec.seq_len, 128);
        assert_eq!(spec.layer_norm_eps, 1e-5);
        assert!(spec.has_attention);
    }

    /// `head_dim * n_heads == d_model` の整合検査（Multi-Head Attention のヘッド分割が
    /// 割り切れることを保証する。実測実装側の reshape が失敗しないための前提条件）。
    #[test]
    fn baseline_spec_head_dim_divides_evenly() {
        let spec = baseline_spec();
        let head_dim = spec
            .head_dim()
            .expect("baseline_spec() は不変条件を満たす確定値のはず");
        assert_eq!(head_dim * spec.n_heads, spec.d_model);
        assert_eq!(head_dim, 64);
    }

    /// `n_heads == 0`（ゼロ除算になりうる値）を構築しても `head_dim()` は `panic` せず
    /// `None` を返すことを検査する（codex-review 指摘・PR #647 P1 の回帰防止）。
    #[test]
    fn head_dim_returns_none_for_zero_n_heads() {
        let spec = TransformerWorkloadSpec {
            n_heads: 0,
            ..baseline_spec()
        };
        assert_eq!(spec.head_dim(), None);
    }

    /// `d_model % n_heads != 0`（割り切れない値）でも `head_dim()` が `None` を返すことを
    /// 検査する（`d_model` フィールドのドキュメンテーションコメントが掲げる不変条件の検査）。
    #[test]
    fn head_dim_returns_none_when_not_evenly_divisible() {
        let spec = TransformerWorkloadSpec {
            n_heads: 3,
            ..baseline_spec()
        };
        assert_eq!(spec.head_dim(), None);
    }

    #[test]
    fn report_name_uses_shared_prefix() {
        assert_eq!(report_name("cpu"), "transformer-block-forward-cpu-blis");
        assert_eq!(report_name("cuda"), "transformer-block-forward-cuda");
    }

    /// #602（G-12）: 融合あり経路のベンチ名は非融合経路（[`report_name`]）に
    /// `"-fused"` サフィックスを付与した値になる。
    #[test]
    fn report_name_fused_appends_suffix() {
        assert_eq!(
            report_name_fused("cuda"),
            "transformer-block-forward-cuda-fused"
        );
        assert_eq!(
            report_name_fused("cpu"),
            "transformer-block-forward-cpu-blis-fused"
        );
    }
}
