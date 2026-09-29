//! reduction カーネル（`sum`・`max`・`mean`。TASK-1.6c・#23）。
//!
//! `backend-cpu` は `backend-cuda`/`backend-metal` との数値一致検証（REQ-2
//! 複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）の**参照点**であり、
//! PoC-v2-5 実測（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md:24,147`）
//! が示すとおり「演算順序を固定した決定的な reduction」であることが後続タスク
//! （TASK-2.2 数値一致回帰テスト）の前提になる。本モジュールの全 API は
//! **スレッド数に依存しない固定順序の累積**を必須契約とする（下記「決定性契約」参照）。
//!
//! `docs/public-api-design.md` §4.2 の `BackendOps` トレイト（`sum`/`max` の
//! `dim: Option<usize>` シグネチャ）と対称な自由関数として実装する。トレイト
//! 実装（`BackendOps` そのもの）・`DeviceBuffer` 対応は TASK-1.9 のスコープであり
//! 本イシューには含めない。
//!
//! ## 決定性契約
//!
//! - **軸指定（`dim=Some(axis)`）**: 出力を `outer × inner` に分解し、rayon は
//!   出力要素側のみ並列化する。各出力要素は縮約軸を昇順に逐次累積するため、
//!   共有アキュムレータへの並列書き込みが発生せず、スレッド数に依らず bit
//!   決定的になる。
//! - **全縮約（`dim=None`）**: 固定チャンクサイズ（`CHUNK`）で分割し、
//!   各チャンク内は逐次累積・チャンク間は rayon で並列処理する。
//!   `rayon::slice::ParallelSlice::par_chunks` は `IndexedParallelIterator`
//!   であり、そこからの `.collect::<Vec<_>>()` は実行スレッド数・
//!   スケジューリング順に依らず**入力順を保持する**契約を rayon が保証する
//!   （<https://docs.rs/rayon/latest/rayon/iter/trait.IndexedParallelIterator.html>）。
//!   本モジュールはこの保証を用いてチャンク部分和をチャンク番号順に逐次結合し、
//!   PoC-v2-5 の「逐次固定順序で bit 一致」前提を踏襲する。
//!   **例外（`logsumexp`／`vector_norm_p`。イシュー #2147・PR #2263
//!   codex-review P2 是正）**: この 2 演算の全縮約経路（`logsumexp_slice`・
//!   `vector_norm_p_slice`。いずれも非公開関数）は `autodiff::eval` の
//!   ホスト参照実装（`logsumexp_along`／`vector_norm_p_along`）と **bit 完全一致**
//!   させる契約（`docs/autodiff-reduce-ops-decision.md` §2.4）を持つ。
//!   eval 側は `dim=None` を「単一 lane・逐次 `f64` fold」として計算する
//!   ため、CHUNK 単位でチャンク内逐次 → チャンク間結合という 2 段の
//!   結合順序（本節上記の一般契約）とは異なり、要素数が `CHUNK`
//!   （4096）を超えると丸め結果が食い違う。そのためこの 2 演算のみ
//!   `par_chunks` を使わず、eval と同一の単一逐次 `f64` fold をそのまま
//!   用いる（全軸縮約の rayon 並列性を犠牲にする。軸指定側は他演算と
//!   同じ lane 間並列化のままで問題は生じない）。
//!
//! ## `sum`/`mean` の `f64` アキュムレータ契約（イシュー #1675）
//!
//! `sum`（`sum_slice`／`axis_reduce_sum`）は `.claude/rules/
//! coding-rust.md`「正規化統計・勾配の長軸縮約は `f64` アキュムレータで
//! 統一する」契約に合わせ、`f64` で累積し**最後に 1 回だけ** `f32` へ
//! downcast する（`backend-cuda::kernels_reduce` の sum カーネル
//! （`REDUCE_SUM_ALL_PARTIAL_F32` 等）と同じ精度契約。`mean` は `sum` の
//! 結果を除算するため同じ契約を継承する）。チャンク分割・出力要素の並列化
//! 軸・累積順序自体は変更しない（本節冒頭の決定性契約はそのまま維持）。
//! `max`（`max_slice`／`axis_reduce` 経由）は丸めを伴わない厳密選択の
//! ため `f32` のまま（対象外）。
//!
//! ## 小サイズ直列フォールバック（機構導入済み・既定 OFF・イシュー #2101）
//!
//! 全 rayon サイトは 2 つのヘルパー（`chunk_partials`・`map_outputs`）経由で、
//! ゲート `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED`（既定 `false`）が `true`
//! かつ入力要素数（`numel`）が `REDUCTION_PARALLEL_MIN_ELEMS` 未満のとき
//! だけ逐次腕へ落ちる。ゲート `false` の間は変更前と完全に同一（常に rayon）。
//! 逐次腕もチャンク内 fold → チャンク番号順 fold の 2 段構造・出力要素ごとの
//! 昇順累積を保つため、どちらの腕も bit 同一（上記「決定性契約」を壊さない）。
//!
//! - しきい値の値は未実測の暫定候補。M4 Max・GB10 での実機スイープ（#2102）
//!   で決める。`crate::elementwise::PARALLEL_THRESHOLD` は累積を伴わない
//!   契約向けの値であり流用しない（`docs/perf/cpu-parallel-threshold-sweep.md`）。
//! - `mse_loss_backward` への適用は #1578 で REJECT 確定（対象外）。
//! - 設計・事前登録規則は `docs/perf/cpu-reduction-sequential-threshold.md`。
//! - `logsumexp`／`vector_norm_p` の全縮約側はもともと逐次のため対象外。
//!
//! ## 空縮約の意味論
//!
//! - `sum` は単位元 `0.0`（NumPy 互換）を返す。
//! - `max`/`mean` は単位元を持たないため [`ReduceError::EmptyReduction`] を
//!   返す（NaN を黙って返さない安全側の設計）。
//!
//! ## 境界検査（REQ-8）
//!
//! 性能下限・最適化の達成を理由に手動境界チェックを省略しない。`Tensor::get`
//! （境界チェック付き安全アクセス）のみを用い、`get_unchecked` 等は使わない。

use std::fmt;

use fandhe_ai_tensor_core::{ShapeError, Tensor, VectorNormOrd, reduce_out_shape};
use rayon::prelude::*;

/// 全縮約（`dim=None`）の決定的チャンク結合に用いる固定チャンクサイズ。
///
/// チャンク境界を跨ぐ演算順序の違いが bit 差を生まないよう、値は実装内で
/// 固定する（呼び出し側からの変更点を持たない。ガードレール閾値ではないが、
/// 数値一致回帰テストの前提となるため安易に変更しない）。
pub(crate) const CHUNK: usize = 4096;

/// 全縮約サイトの rayon fork-join を要素数で逐次へ落とす機構の**ゲート**
/// （イシュー #2101。低レイヤー診断 `docs/perf/lowlayer-diagnosis-2026-09-12.md`
/// §4 の小形状 fork-join 固定費対策）。`false`（既定）の間は全サイトが従来
/// どおり常に rayon 経由で、挙動は変更前と完全に同一。`true` への切替は
/// #2102 の事前登録判定（`docs/perf/logs/elemental-reduction-threshold-2101/
/// RULE.txt`）を経た場合のみ。`mse_loss_backward` は #1578 で REJECT
/// 確定のため対象外。
pub(crate) const REDUCTION_SEQUENTIAL_FALLBACK_ENABLED: bool = false;

/// 逐次へ落とす入力要素数（`numel`）の**未実測の暫定候補**。ゲートが
/// `false` の間は効かない。値は #2102 の実測で決める（#1578 Phase 0 で両機体
/// とも `1 << 18` まで逐次が優位だったことのみを参考根拠とする）。
/// `elementwise::PARALLEL_THRESHOLD` は流用しない（モジュール doc 参照）。
pub(crate) const REDUCTION_PARALLEL_MIN_ELEMS: usize = 1 << 18;

/// 逐次・並列の選択規則。判定尺度は入力要素数（`numel`）で、`dim=None`・
/// `dim=Some(axis)`（`outer*axis_len*inner`）とも同じ量を使う。
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeqPolicy {
    pub(crate) enabled: bool,
    pub(crate) min_elems: usize,
}

impl SeqPolicy {
    /// 本番既定（上記 2 定数から構成）。
    pub(crate) const DEFAULT: SeqPolicy = SeqPolicy {
        enabled: REDUCTION_SEQUENTIAL_FALLBACK_ENABLED,
        min_elems: REDUCTION_PARALLEL_MIN_ELEMS,
    };

    /// `numel` で逐次腕を選ぶ述語（fork-join 入口の唯一の判定点）。
    pub(crate) fn run_sequential(self, numel: usize) -> bool {
        self.enabled && numel < self.min_elems
    }
}

#[cfg(test)]
thread_local! {
    /// テスト専用の強制ポリシー。ヘルパーは呼び出しスレッド上で判定する
    /// （rayon ワーカー側では参照しない）ため thread_local で足りる。
    /// 公開面を増やさないための `#[cfg(test)]` 限定機構。
    static POLICY_OVERRIDE: std::cell::Cell<Option<SeqPolicy>> =
        const { std::cell::Cell::new(None) };
}

/// 現在有効なポリシー。本番ビルドでは常に [`SeqPolicy::DEFAULT`]。
#[cfg(not(test))]
fn current_policy() -> SeqPolicy {
    SeqPolicy::DEFAULT
}

#[cfg(test)]
fn current_policy() -> SeqPolicy {
    POLICY_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(SeqPolicy::DEFAULT)
}

/// `data` を [`CHUNK`] 単位に分割し各チャンクへ `f` を適用した部分結果を
/// チャンク番号順に返す。逐次腕・並列腕とも**チャンク内 fold → チャンク
/// 番号順 fold の 2 段構造は同一**で、結合順序は変わらず bit 同一
/// （全体を 1 本で fold する形にはしない）。`sum_slice_f64` 等から呼ばれる。
fn chunk_partials<T, F>(data: &[f32], f: F) -> Vec<T>
where
    T: Send,
    F: Fn(&[f32]) -> T + Sync + Send,
{
    if current_policy().run_sequential(data.len()) {
        data.chunks(CHUNK).map(f).collect()
    } else {
        data.par_chunks(CHUNK).map(f).collect()
    }
}

/// 出力要素 `0..total_out` へ `compute` を適用して順序どおり集める。各出力
/// は縮約軸を昇順に逐次累積するため逐次腕・並列腕は bit 同一。`numel` は
/// 入力要素数（しきい値判定用）。軸指定縮約から呼ばれる。
fn map_outputs<T, F>(numel: usize, total_out: usize, compute: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if current_policy().run_sequential(numel) {
        (0..total_out).map(compute).collect()
    } else {
        (0..total_out).into_par_iter().map(compute).collect()
    }
}

/// reduction カーネル固有のエラー。`BackendError`（TASK-1.9 で導入予定）への
/// ラップは `BackendOps` 実装時に行う想定であり、本モジュールでは行わない。
#[non_exhaustive]
#[derive(Debug)]
pub enum ReduceError {
    /// shape・軸検査の失敗（`fandhe_ai_tensor_core::reduce_out_shape` に委譲する検査の
    /// 失敗をそのまま透過する）。
    Shape(ShapeError),
    /// 縮約対象の要素数が 0（`max`・`mean` は単位元を持たないためエラーとする。
    /// `op` は失敗した演算名 `"max"`/`"mean"`/`"min"`/`"argmax"`/`"argmin"`／`"var"`/`"norm"`。
    /// `min`／`argmax`／`argmin` はイシュー #1720 で追加）。`argmax`／
    /// `argmin` の添字が `i32::MAX` を超える場合は
    /// `Shape(ShapeError::IndexRangeOverflow { .. })`
    /// （`sort_topk.rs`／`gather_scatter.rs` と同じ契約）を使う。
    EmptyReduction { op: &'static str },
    /// `var` の自由度不足（`n <= correction`。イシュー #1723）。`n` は
    /// 縮約対象の要素数（`dim=Some(axis)` は `shape[axis]`・`dim=None`
    /// は全要素数）、`correction` は呼び出し元が指定した自由度補正
    /// （`torch.var` の `correction` 引数と同じ意味）。`NaN`／`inf` を
    /// 黙って返さない安全側の判断（`fandhe_ai_tensor_core::BackendOps::
    /// var` doc「エラー契約」参照）。
    InsufficientDegreesOfFreedom { n: usize, correction: usize },
    /// `vector_norm` へ未知の [`VectorNormOrd`] variant（同型は
    /// `#[non_exhaustive]`。`crates/backend-cpu/src/linalg.rs::
    /// matrix_norm` が `MatrixNormOrd` に対して行う fail-closed 拒否と
    /// 同方針。イシュー #1723）が渡された。
    UnsupportedOrd(String),
    /// `vector_norm_p` へ渡された `p` が無効（有限かつ正の実数ではない。
    /// イシュー #2147）。`NaN`／`±inf`／`0`／負の値を fail-closed に
    /// 拒否する（`fandhe_ai_tensor_core::BackendOps::vector_norm_p`
    /// doc「エラー契約」参照）。
    InvalidOrder(f32),
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceError::Shape(err) => write!(f, "reduction shape error: {err}"),
            ReduceError::EmptyReduction { op } => {
                write!(f, "cannot compute {op} of an empty reduction")
            }
            ReduceError::InsufficientDegreesOfFreedom { n, correction } => write!(
                f,
                "cannot compute var: n ({n}) <= correction ({correction}) (degrees of freedom \
                 would be non-positive)"
            ),
            ReduceError::UnsupportedOrd(desc) => {
                write!(f, "unsupported VectorNormOrd variant: {desc}")
            }
            ReduceError::InvalidOrder(p) => {
                write!(f, "vector_norm_p: p must be finite and positive, got {p}")
            }
        }
    }
}

impl std::error::Error for ReduceError {}

/// 線形インデックス空間（`0..outer*inner`）を行優先で `dims` の多次元
/// インデックスへ展開する。`row_major_strides`（tensor-core）と対の関係にある
/// 展開処理で、`axis_reduce`（出力側の外側・内側インデックス復元）と
/// `gather_elements`（非 contiguous 全縮約時の走査順再現）の両方から使う。
pub(crate) fn unravel(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; dims.len()];
    for (axis, &d) in dims.iter().enumerate().rev() {
        if d == 0 {
            out[axis] = 0;
            continue;
        }
        out[axis] = idx % d;
        idx /= d;
    }
    out
}

/// 非 contiguous な入力を行優先順（`Tensor::contiguous()` が実体化する順序と
/// 同一）で走査し `Vec<f32>` へ収集する。`as_slice()` が使えない（`None` を
/// 返す）場合の全縮約（`dim=None`）専用フォールバック。
///
/// `Tensor::get` は範囲内アクセスであれば必ず `Some` を返す契約だが、走査
/// ロジック自体にバグがあった場合に備え `tensor-core::Tensor::contiguous()`
/// と同じ方針（debug ビルドで `debug_assert!` により早期検知、release
/// ビルドは安全側フォールバック）を踏襲し `unwrap`/`expect` は使わない
/// （`.claude/rules/coding-rust.md`）。
fn gather_elements(a: &Tensor<f32>) -> Vec<f32> {
    let shape = a.shape();
    let numel = a.numel();
    let mut out = Vec::with_capacity(numel);
    for flat in 0..numel {
        let idx = unravel(flat, shape);
        let value = a.get(&idx);
        debug_assert!(
            value.is_some(),
            "gather_elements: 走査ロジックのバグにより index {idx:?} が範囲外になった"
        );
        out.push(value.unwrap_or(0.0));
    }
    out
}

/// `data` を [`CHUNK`] 単位に分割し、決定性契約（モジュール doc 参照）に
/// 従って `sum` を計算する。逐次フォールバック機構（イシュー #2101）は
/// 導入済みだが既定 OFF のため、現状は常に rayon 経由（`ThreadPoolBuilder::
/// num_threads(1)` 下でも `par_chunks` の順序保持契約により逐次実行と
/// bit 完全一致する）で計算する。
///
/// アキュムレータは `f64`（チャンク内・チャンク間結合とも）で、**最後に
/// 1 回だけ** `f32` へ downcast する（`.claude/rules/coding-rust.md`
/// 「正規化統計・勾配の長軸縮約は `f64` アキュムレータで統一する」契約。
/// `backend-cuda::kernels_reduce`（`REDUCE_SUM_ALL_PARTIAL_F32`／
/// `REDUCE_SUM_ALL_FINALIZE_F32`。イシュー #1584）と同じ精度契約を CPU
/// 参照実装にも揃える。チャンク分割・結合順序自体は変更しない — 変更は
/// 各要素の畳み込み精度のみであり、REQ-2 の許容誤差（統一複合判定）を
/// 緩和するものではない。イシュー #1675 codex-review 指摘）。
fn sum_slice(data: &[f32]) -> f32 {
    sum_slice_f64(data) as f32
}

/// [`sum_slice`] の `f64` 版（downcast 前の値をそのまま返す）。
/// `var`（[`var_slice`]）が平均を `f64` のまま使う（`sum_slice` を経由
/// すると 1 回余計に `f32` へ downcast してしまい `.claude/rules/
/// coding-rust.md` の「最後に 1 回だけ downcast する」契約に反する）
/// ため、チャンク分割・結合ロジックそのものを共有する目的で切り出した
/// （イシュー #1723。`sum_slice` の挙動・決定性契約は不変）。
fn sum_slice_f64(data: &[f32]) -> f64 {
    chunk_partials(data, |chunk| {
        chunk.iter().fold(0.0f64, |acc, &v| acc + v as f64)
    })
    .into_iter()
    .fold(0.0f64, |acc, v| acc + v)
}

/// `data` を [`CHUNK`] 単位に分割し、決定性契約（モジュール doc 参照）に
/// 従って `max` を計算する。`data` が空の場合は `None` を返す（呼び出し元が
/// [`ReduceError::EmptyReduction`] に変換する）。直列フォールバック
/// （既定 OFF。イシュー #2101）の扱いは [`sum_slice`] と同じ（モジュール
/// doc「小サイズ直列フォールバック」参照）。
///
/// 単位元として `f32::NEG_INFINITY` を用いる（`max(x, -inf) == x` が任意の
/// 有限値 `x` で成立するため、`unwrap`/`expect` なしで畳み込みの初期値に
/// 使える）。`f32::max` は NaN 非伝播（`NaN` を無視して他方を返す）
/// セマンティクスを持つ（PyTorch の NaN 伝播 `max` とは意味論が異なる。
/// スコープ外事項として記録。実装計画 §7 参照）。
fn max_slice(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let result = chunk_partials(data, |chunk| {
        chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    })
    .into_iter()
    .fold(f32::NEG_INFINITY, f32::max);
    Some(result)
}

/// [`max_slice`] の最小値版（イシュー #1720）。単位元は
/// `f32::INFINITY`。**NaN 非伝播**（`f32::min`。`fminf` と同じ）で、
/// [`max_slice`] の `f32::max` と対称の意味論とする（`autodiff::eval::
/// min` の意図的複製先——`crate::backend_ops::BackendOps::min` doc
/// 参照）。
fn min_slice(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let result = chunk_partials(data, |chunk| {
        chunk.iter().copied().fold(f32::INFINITY, f32::min)
    })
    .into_iter()
    .fold(f32::INFINITY, f32::min);
    Some(result)
}

/// `outer`/`inner`（軸を除いた前後の次元積）を `checked_mul` で計算する。
/// アロケーション前の要素数計算にオーバーフローが混入すると過小確保・
/// 境界不整合を招く（OWASP A03 相当。`.claude/rules/security.md`）ため、
/// `tensor-core::checked_numel` と同方針でオーバーフローを検出する。
pub(crate) fn checked_product(dims: &[usize]) -> Result<usize, ReduceError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))
}

/// [`checked_product`] に加え、`f32` 換算のバイトサイズが `Vec` の
/// allocation 上限（`isize::MAX` バイト）に収まるかも検査する
/// （`backend-cpu::ops::checked_alloc_numel_f32`／`backend-cuda::
/// ops::checked_bytes_for`／`backend-metal::ops::checked_bytes_for`／
/// `autodiff::bool_ops::checked_bytes_for` と同型の独立複製。可視性の
/// 意味論が異なるモジュール・クレートを跨ぐため個別に持つ方針は本
/// リポジトリの既存パターン〈`backend-cpu::ops.rs` の同名関数
/// doc〉を踏襲する）。
///
/// **動機**: `logsumexp`／`vector_norm_p`／`vector_norm`（イシュー
/// #2287 で追加）・`var`（イシュー #2288 で追加）は小さなストレージ
/// を巨大な shape へ broadcast した view を受け取りうる。`outer *
/// inner`（軸指定側の出力要素数）や `a.numel()`（全縮約・非
/// contiguous 側の入力要素数）は `usize` の積としては収まっても、
/// `f32` 換算のバイト数が `isize::MAX` を超えることがあり、その
/// まま `Vec::with_capacity`／`.collect()` へ進むと型付きエラー
/// ではなく capacity overflow panic になる（本番経路 panic 禁止
/// 規約 `.claude/rules/coding-rust.md`。codex-review P1 指摘の是正・
/// イシュー #2147・PR #2263）。
pub(crate) fn checked_alloc_numel_f32(shape: &[usize]) -> Result<usize, ReduceError> {
    let numel = checked_product(shape)?;
    let bytes = numel
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
    if bytes > isize::MAX as usize {
        return Err(ReduceError::Shape(ShapeError::ElementCountOverflow));
    }
    Ok(numel)
}

/// 軸指定 reduction（`dim=Some(axis)`）の出力要素ごとの畳み込みを行う共通
/// 駆動関数。`axis` は呼び出し元（`sum`/`max`/`mean`）が `reduce_out_shape`
/// で事前検査済みであることを前提とする（本関数自体は範囲検査を行わない）。
///
/// 出力要素（`outer × inner` 個）側のみ rayon で並列化し、各要素内では
/// 縮約軸を `0..axis_len` の昇順で `op` により逐次累積する（決定性契約は
/// モジュール doc 参照）。`Range<usize>` は `IndexedParallelIterator` であり
/// `.collect()` が出力順を保持するため、`flat` 昇順の出力ベクタが得られる。
/// 小サイズ直列フォールバックは `map_outputs` 経由の既定 OFF 機構
/// （モジュール doc「小サイズ直列フォールバック」参照）。
fn axis_reduce<F>(a: &Tensor<f32>, axis: usize, identity: f32, op: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    // 出力要素ごとの計算（`flat in 0..total_out` を前提とする契約は
    // クロージャ内コメント参照）。
    let compute = |flat: usize| -> f32 {
        // このクロージャは `flat in 0..total_out`（`total_out = outer *
        // inner`）でのみ呼ばれる。`total_out > 0` は `inner > 0` を含意する
        // ため（`inner == 0` なら `total_out == 0` で range が空になり
        // 到達しない）、`inner` によるゼロ除算は発生しない。
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut acc = identity;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_reduce: 走査ロジックのバグにより index {full_idx:?} が範囲外になった"
            );
            acc = op(acc, value.unwrap_or(identity));
        }
        acc
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// [`axis_reduce`] の `sum` 専用版。走査構造は完全に同一（並列化軸・
/// 縮約順序とも）だが、アキュムレータを `f64` にし出力要素ごとに
/// **最後に 1 回だけ** `f32` へ downcast する（[`sum_slice`] と同じ
/// `.claude/rules/coding-rust.md` 契約。`mean` の軸指定経路も本関数の
/// 結果を除算するため同じ精度契約を継承する）。`max`（[`axis_reduce`]
/// 経由。丸めなしの厳密選択）は本関数の対象外（イシュー #1675
/// codex-review 指摘）。
fn axis_reduce_sum(a: &Tensor<f32>, axis: usize) -> Vec<f32> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    // `axis_reduce` と同じ契約（クロージャは `flat in 0..total_out` の
    // みで呼ばれ、`inner` によるゼロ除算は発生しない）。
    let compute = |flat: usize| -> f32 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_reduce_sum: 走査ロジックのバグにより index {full_idx:?} が範囲外になった"
            );
            acc += value.unwrap_or(0.0) as f64;
        }
        acc as f32
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// `data`（`dim=None` の全縮約対象。空でないこと・`n > correction` は
/// 呼び出し元 [`var`] が事前検査済みの前提）から分散を計算する。
/// [`sum_slice_f64`] と同じチャンク並列・決定性契約（モジュール doc
/// 参照）で ①平均 ②二乗和 の 2 パスを `f64` で計算し、最後に 1 回だけ
/// `f32` へ downcast する（`.claude/rules/coding-rust.md`「正規化統計は
/// 要素を先に `f64` へ昇格してから二乗し、最後に 1 回だけ `f32` へ
/// downcast する」契約。イシュー #1723）。
fn var_slice(data: &[f32], correction: usize) -> f32 {
    let n = data.len() as f64;
    let mean = sum_slice_f64(data) / n;
    let sq_sum: f64 = chunk_partials(data, |chunk| {
        chunk.iter().fold(0.0f64, |acc, &v| {
            let d = v as f64 - mean;
            acc + d * d
        })
    })
    .into_iter()
    .fold(0.0f64, |acc, v| acc + v);
    (sq_sum / (n - correction as f64)) as f32
}

/// [`var_slice`] の軸指定（`dim=Some(axis)`）版。`axis_reduce_sum` と
/// 同じ「出力側 rayon 並列・各要素内は縮約軸を昇順に逐次」決定性契約
/// （モジュール doc 参照）で、出力要素ごとに ①平均 ②二乗和 の 2 パスを
/// `f64` で計算する。`axis_len > correction` は呼び出し元 [`var`] が
/// 事前検査済みの前提。
fn axis_reduce_var(a: &Tensor<f32>, axis: usize, correction: usize) -> Vec<f32> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;
    let n = axis_len as f64;
    let denom = n - correction as f64;

    // `axis_reduce_sum` と同じ契約（クロージャは `flat in 0..total_out`
    // のみで呼ばれ、`inner` によるゼロ除算は発生しない）。
    let compute = |flat: usize| -> f32 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut mean_acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_reduce_var: 走査ロジックのバグにより index {full_idx:?} が範囲外になった"
            );
            mean_acc += value.unwrap_or(0.0) as f64;
        }
        let mean = mean_acc / n;
        let mut sq_acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx).unwrap_or(0.0) as f64;
            let d = value - mean;
            sq_acc += d * d;
        }
        (sq_acc / denom) as f32
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// [`VectorNormOrd`] を検査済みの内部表現へ変換したもの。`VectorNormOrd`
/// は `#[non_exhaustive]`（`tensor-core`）のため、これを直接 match する
/// 箇所（[`vector_norm_slice`]／[`axis_reduce_vector_norm`] 双方）は
/// 将来 variant に備えた `_` 分岐を必要とする。変換点を [`vector_norm`]
/// の入口 1 箇所へ集約し、それ以外の内部関数は本 2 variant の閉じた
/// enum（fail-closed 検査済み）だけを扱えばよいようにする（`crates/
/// backend-cpu/src/linalg.rs::matrix_norm` の `MatrixNormOrd` 拒否と
/// 同方針。イシュー #1723）。
#[derive(Clone, Copy)]
enum NormKind {
    L1,
    L2,
}

impl NormKind {
    fn from_ord(ord: VectorNormOrd) -> Result<Self, ReduceError> {
        match ord {
            VectorNormOrd::L1 => Ok(NormKind::L1),
            VectorNormOrd::L2 => Ok(NormKind::L2),
            _ => Err(ReduceError::UnsupportedOrd(format!("{ord:?}"))),
        }
    }
}

/// `data`（`dim=None` の全縮約対象。空でないことは呼び出し元 [`vector_norm`]
/// が事前検査済みの前提）から L1／L2 ノルムを計算する。[`sum_slice_f64`]
/// と同じチャンク並列・決定性契約で `f64` 累積し、L2 のみ最後に `sqrt`
/// してから 1 回だけ `f32` へ downcast する（イシュー #1723）。
fn vector_norm_slice(data: &[f32], kind: NormKind) -> f32 {
    let acc: f64 = chunk_partials(data, |chunk| {
        chunk.iter().fold(0.0f64, |acc, &v| {
            let v = v as f64;
            match kind {
                NormKind::L1 => acc + v.abs(),
                NormKind::L2 => acc + v * v,
            }
        })
    })
    .into_iter()
    .fold(0.0f64, |acc, v| acc + v);
    match kind {
        NormKind::L1 => acc as f32,
        NormKind::L2 => acc.sqrt() as f32,
    }
}

/// [`vector_norm_slice`] の軸指定（`dim=Some(axis)`）版。`axis_reduce_sum`
/// と同じ決定性契約で、出力要素ごとに `f64` 累積後 L2 のみ `sqrt` して
/// 1 回だけ `f32` へ downcast する。
fn axis_reduce_vector_norm(a: &Tensor<f32>, axis: usize, kind: NormKind) -> Vec<f32> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    let compute = |flat: usize| -> f32 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_reduce_vector_norm: 走査ロジックのバグにより index {full_idx:?} が範囲外に\
                 なった"
            );
            let v = value.unwrap_or(0.0) as f64;
            acc += match kind {
                NormKind::L1 => v.abs(),
                NormKind::L2 => v * v,
            };
        }
        match kind {
            NormKind::L1 => acc as f32,
            NormKind::L2 => acc.sqrt() as f32,
        }
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// 軸指定・全縮約いずれにも対応する `var`（`torch.var(dim, correction)`
/// 相当。イシュー #1723）。`dim=None` は rank 0（スカラー）テンソルを
/// 返す。数値契約は [`fandhe_ai_tensor_core::BackendOps::var`] doc を
/// 正とする。
///
/// 縮約対象の要素数 `n`（`dim=Some(axis)` は `shape[axis]`・`dim=None`
/// は全要素数）が `0` の場合は [`ReduceError::EmptyReduction`]、
/// `n <= correction` の場合は [`ReduceError::InsufficientDegreesOfFreedom`]
/// を返す。
///
/// **確保前のバイト数上限検査（`checked_alloc_numel_f32`。イシュー
/// #2288）**: `vector_norm`（イシュー #2287）と同じ理由・同じ位置で、
/// 小さなストレージを巨大な shape へ broadcast した view に対する
/// `gather_elements`（非 contiguous 全縮約）・`axis_reduce_var`
/// の `.collect()`（軸指定）の確保前に、要素数積の `usize`
/// オーバーフロー・`Vec` allocation 上限（`isize::MAX` バイト）超過を
/// 型付きエラーで拒否する（本番経路 panic 禁止規約
/// `.claude/rules/coding-rust.md`）。本関数の中間バッファ（`var_slice`
/// ／`axis_reduce_var` の `f64` アキュムレータ）はいずれも出力要素
/// ごとのスタックローカル変数であり `Vec` として確保しないため、
/// `f32` 換算の検査のみで足りる（`autodiff::var.rs::
/// var_std_out_shape_checked` が入口で `f64` 出力検査も行う eval 側
/// 〈`Vec<f64>` 中間バッファを持つ〉との違い）。
pub fn var(
    a: &Tensor<f32>,
    dim: Option<usize>,
    correction: usize,
) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let n = match dim {
        None => a.numel(),
        Some(axis) => a.shape()[axis],
    };
    if n == 0 {
        return Err(ReduceError::EmptyReduction { op: "var" });
    }
    if n <= correction {
        return Err(ReduceError::InsufficientDegreesOfFreedom { n, correction });
    }
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => var_slice(slice, correction),
                None => {
                    // 非 contiguous（`gather_elements` が実体化する）
                    // 経路のみ確保前検査する。`as_slice()` が `Some` の
                    // 場合は既に実体化済みのスライスを走査するだけで
                    // 新規確保がないため検査不要（`vector_norm` の
                    // `None` 分岐と同じ理由。
                    // [`checked_alloc_numel_f32`] doc「動機」参照）。
                    checked_alloc_numel_f32(a.shape())?;
                    var_slice(&gather_elements(a), correction)
                }
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            // `axis_reduce_var` の `.collect()` は `out_shape`
            // （`outer * inner` 要素）と同じサイズの `Vec<f32>` を確保
            // する。直上の `checked_mul` は要素数積のオーバーフロー
            // のみを検査するため、確保可能バイト数（`isize::MAX`
            // 上限）は別途検査する（`vector_norm` の `Some(axis)`
            // 分岐と同じ理由。[`checked_alloc_numel_f32`] doc「動機」
            // 参照）。
            checked_alloc_numel_f32(&out_shape)?;
            axis_reduce_var(a, axis, correction)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// 軸指定・全縮約いずれにも対応する `vector_norm`（`torch.norm`／
/// `tf.norm` の L1／L2 相当。イシュー #1723）。`dim=None` は rank 0
/// （スカラー）テンソルを返す。数値契約は
/// [`fandhe_ai_tensor_core::BackendOps::vector_norm`] doc を正とする。
///
/// 縮約対象の要素数が `0` の場合は [`ReduceError::EmptyReduction`] を
/// 返す（`var` と対称な「空縮約は明示エラー」の方針）。
///
/// **確保前のバイト数上限検査（`checked_alloc_numel_f32`。イシュー
/// #2287）**: `logsumexp`／`vector_norm_p` と同じ理由・同じ位置で、
/// 小さなストレージを巨大な shape へ broadcast した view に対する
/// `gather_elements`（非 contiguous 全縮約）・`axis_reduce_vector_norm`
/// の `.collect()`（軸指定）の確保前に、要素数積の `usize`
/// オーバーフロー・`Vec` allocation 上限（`isize::MAX` バイト）超過を
/// 型付きエラーで拒否する（本番経路 panic 禁止規約
/// `.claude/rules/coding-rust.md`）。
pub fn vector_norm(
    a: &Tensor<f32>,
    ord: VectorNormOrd,
    dim: Option<usize>,
) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let kind = NormKind::from_ord(ord)?;
    let n = match dim {
        None => a.numel(),
        Some(axis) => a.shape()[axis],
    };
    if n == 0 {
        return Err(ReduceError::EmptyReduction { op: "norm" });
    }
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => vector_norm_slice(slice, kind),
                None => {
                    // 非 contiguous（`gather_elements` が実体化する）
                    // 経路のみ確保前検査する。`as_slice()` が `Some` の
                    // 場合は既に実体化済みのスライスを走査するだけで
                    // 新規確保がないため検査不要（`logsumexp` の
                    // `None` 分岐と同じ理由。
                    // [`checked_alloc_numel_f32`] doc「動機」参照）。
                    checked_alloc_numel_f32(a.shape())?;
                    vector_norm_slice(&gather_elements(a), kind)
                }
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            // `axis_reduce_vector_norm` の `.collect()` は `out_shape`
            // （`outer * inner` 要素）と同じサイズの `Vec<f32>` を確保
            // する。直上の `checked_mul` は要素数積のオーバーフロー
            // のみを検査するため、確保可能バイト数（`isize::MAX`
            // 上限）は別途検査する（`logsumexp` の `Some(axis)` 分岐と
            // 同じ理由。[`checked_alloc_numel_f32`] doc「動機」参照）。
            checked_alloc_numel_f32(&out_shape)?;
            axis_reduce_vector_norm(a, axis, kind)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// `f64` の NaN 伝播 2 項最大値（`eval::nan_propagating_max` の `f64`
/// 版。イシュー #2147）。`f64::max` は非 `NaN` 側を返してしまうため、
/// `logsumexp`／`vector_norm_p` の `m = max(x)`／`mx = max|x_i|` が
/// `NaN` 入力を静かに消さないよう本関数で置き換える。
fn nan_propagating_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `logsumexp`（`dim=None` の全縮約対象。空でないことは呼び出し元
/// [`logsumexp`] が事前検査済みの前提）を `f64` で計算する（イシュー
/// #2147・PR #2263 codex-review P2 是正）。`m = max(x)`（非有限なら
/// 安定化シフトを `0` に切り替える）→`Σ exp(x_i − m)` を `f64` で蓄積
/// → `ln(acc) + m` を計算し、最後に 1 回だけ `f32` へ downcast する
/// （`fandhe_ai_tensor_core::BackendOps::logsumexp` doc「数値契約」
/// 参照）。
///
/// **`par_chunks` を使わない理由（他の全縮約 slice 関数〈`sum_slice`
/// 等〉との違い）**: この関数は `eval::logsumexp_along`（`dim=None`
/// 時の `outer=1, axis_len=n, inner=1` 分解）の**逐次 `f64` 蓄積**
/// （`0..n` を単一の `fold` で左から右へ加算）と bit 完全一致させる
/// 契約を持つ（`docs/autodiff-reduce-ops-decision.md` §2.4）。CHUNK
/// 単位でチャンク内を逐次累積してからチャンク結果をチャンク番号順に
/// 結合する方式（`sum_slice` 等が使う）は、浮動小数点加算が結合則を
/// 満たさないため要素数が `CHUNK`（4096）を超えると eval の単一逐次
/// fold と異なる丸め結果になり bit 一致が崩れる（PR #2263
/// codex-review 指摘。回帰テストは `crates/facade/tests/
/// reduce_ops_backend_parity.rs::
/// cpu_logsumexp_vector_norm_p_forward_bit_matches_naive_reference_across_chunk_boundary`）。
/// そのためこの全縮約経路は rayon 並列化を諦め、eval と同一の演算列
/// （逐次 `f64` fold）をそのまま踏襲する。軸指定側
/// （[`axis_reduce_logsumexp`]）は出力要素（lane）間のみ rayon で
/// 並列化し、各 lane 内は元々逐次走査のため本問題は生じない。
fn logsumexp_slice(data: &[f32]) -> f32 {
    let m = data.iter().fold(f64::NEG_INFINITY, |acc, &v| {
        nan_propagating_max_f64(acc, v as f64)
    });
    let shift = if m.is_finite() { m } else { 0.0 };
    let mut acc = 0.0f64;
    for &v in data {
        acc += ((v as f64) - shift).exp();
    }
    (acc.ln() + shift) as f32
}

/// [`logsumexp_slice`] の軸指定（`dim=Some(axis)`）版。`axis_reduce_
/// vector_norm` と同じ決定性契約（出力要素側を rayon で並列化・縮約軸は
/// 逐次走査）。
fn axis_reduce_logsumexp(a: &Tensor<f32>, axis: usize) -> Vec<f32> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    let compute = |flat: usize| -> f32 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut m = f64::NEG_INFINITY;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let v = a.get(&full_idx).unwrap_or(0.0) as f64;
            m = nan_propagating_max_f64(m, v);
        }
        let shift = if m.is_finite() { m } else { 0.0 };
        let mut acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let v = a.get(&full_idx).unwrap_or(0.0) as f64;
            acc += (v - shift).exp();
        }
        (acc.ln() + shift) as f32
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// 軸指定・全縮約いずれにも対応する `logsumexp`（`torch.logsumexp(dim)`
/// 相当。イシュー #2147）。`dim=None` は rank 0（スカラー）テンソルを
/// 返す。数値契約は [`fandhe_ai_tensor_core::BackendOps::logsumexp`]
/// doc を正とする。
///
/// 縮約対象の要素数が `0` の場合は [`ReduceError::EmptyReduction`]
/// （`op: "logsumexp"`）を返す（`-inf` を黙って返さない安全側の判断。
/// `var`／`norm` と対称）。
pub fn logsumexp(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let n = match dim {
        None => a.numel(),
        Some(axis) => a.shape()[axis],
    };
    if n == 0 {
        return Err(ReduceError::EmptyReduction { op: "logsumexp" });
    }
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => logsumexp_slice(slice),
                None => {
                    // 非 contiguous（`gather_elements` が実体化する）
                    // 経路のみ確保前検査する。`as_slice()` が `Some` の
                    // 場合は既に実体化済みのスライスを走査するだけで
                    // 新規確保がないため検査不要（[`checked_alloc_numel_f32`]
                    // doc「動機」参照）。
                    checked_alloc_numel_f32(a.shape())?;
                    logsumexp_slice(&gather_elements(a))
                }
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            // `axis_reduce_logsumexp` の `.collect()` は `out_shape`
            // （`outer * inner` 要素）と同じサイズの `Vec<f32>` を
            // 確保する。直上の `checked_mul` は要素数積のオーバー
            // フローのみを検査するため、確保可能バイト数
            // （`isize::MAX` 上限）は別途検査する
            // （[`checked_alloc_numel_f32`] doc「動機」参照）。
            checked_alloc_numel_f32(&out_shape)?;
            axis_reduce_logsumexp(a, axis)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// p-ノルム（`data`。`dim=None` の全縮約対象。空でないこと・`p` が有限
/// かつ正であることは呼び出し元 [`vector_norm_p`] が事前検査済みの
/// 前提）を `f64` で計算する（イシュー #2147・PR #2263 codex-review
/// P2 是正）。`mx = max|x_i|` を括り出してから
/// `norm = mx · (Σ (|x_i|/mx)^p)^(1/p)` を計算する overflow-safe な
/// スケール形（`fandhe_ai_tensor_core::BackendOps::vector_norm_p` doc
/// 「数値契約」参照）。
///
/// **`par_chunks` を使わない理由**: [`logsumexp_slice`] と同じ理由
/// （同関数 doc 参照）で、`eval::vector_norm_p_along`（`dim=None` 時の
/// 逐次 `f64` fold）と bit 完全一致させる契約を持つため、この全縮約
/// 経路は rayon 並列化を用いない。
fn vector_norm_p_slice(data: &[f32], p: f64) -> f32 {
    let mx = data.iter().fold(0.0f64, |acc, &v| {
        nan_propagating_max_f64(acc, (v as f64).abs())
    });
    if mx.is_nan() {
        return f32::NAN;
    }
    if mx == 0.0 {
        return 0.0;
    }
    if mx.is_infinite() {
        return f32::INFINITY;
    }
    let mut acc = 0.0f64;
    for &v in data {
        let ratio = (v as f64).abs() / mx;
        acc += ratio.powf(p);
    }
    (mx * acc.powf(1.0 / p)) as f32
}

/// [`vector_norm_p_slice`] の軸指定（`dim=Some(axis)`）版。
fn axis_reduce_vector_norm_p(a: &Tensor<f32>, axis: usize, p: f64) -> Vec<f32> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    let compute = |flat: usize| -> f32 {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut mx = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let v = a.get(&full_idx).unwrap_or(0.0) as f64;
            mx = nan_propagating_max_f64(mx, v.abs());
        }
        if mx.is_nan() {
            return f32::NAN;
        }
        if mx == 0.0 {
            return 0.0;
        }
        if mx.is_infinite() {
            return f32::INFINITY;
        }
        let mut acc = 0.0f64;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let v = a.get(&full_idx).unwrap_or(0.0) as f64;
            let ratio = v.abs() / mx;
            acc += ratio.powf(p);
        }
        (mx * acc.powf(1.0 / p)) as f32
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// 軸指定・全縮約いずれにも対応する `vector_norm_p`（`torch.linalg.
/// vector_norm(ord=p)` 相当。イシュー #2147）。`dim=None` は rank 0
/// （スカラー）テンソルを返す。数値契約は [`fandhe_ai_tensor_core::
/// BackendOps::vector_norm_p`] doc を正とする。
///
/// `p` が有限かつ正でなければ [`ReduceError::InvalidOrder`]。縮約対象の
/// 要素数が `0` の場合は [`ReduceError::EmptyReduction`]（`op:
/// "norm_p"`）を返す。
pub fn vector_norm_p(
    a: &Tensor<f32>,
    p: f32,
    dim: Option<usize>,
) -> Result<Tensor<f32>, ReduceError> {
    if !p.is_finite() || p <= 0.0 {
        return Err(ReduceError::InvalidOrder(p));
    }
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let n = match dim {
        None => a.numel(),
        Some(axis) => a.shape()[axis],
    };
    if n == 0 {
        return Err(ReduceError::EmptyReduction { op: "norm_p" });
    }
    let p64 = p as f64;
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => vector_norm_p_slice(slice, p64),
                None => {
                    // `logsumexp` の `None` 分岐と同じ理由（非 contiguous
                    // 経路のみ確保前検査。[`checked_alloc_numel_f32`]
                    // doc「動機」参照）。
                    checked_alloc_numel_f32(a.shape())?;
                    vector_norm_p_slice(&gather_elements(a), p64)
                }
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            // `logsumexp` の `Some(axis)` 分岐と同じ理由（`axis_reduce_
            // vector_norm_p` の `.collect()` 確保前にバイト数上限も
            // 検査する。[`checked_alloc_numel_f32`] doc「動機」参照）。
            checked_alloc_numel_f32(&out_shape)?;
            axis_reduce_vector_norm_p(a, axis, p64)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// 軸指定・全縮約いずれにも対応する `sum`。
///
/// `dim=None` は rank 0（スカラー）テンソルを返す。空テンソルの `sum` は
/// 単位元 `0.0`（NumPy 互換。モジュール doc「空縮約の意味論」参照）。
pub fn sum(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let data = match dim {
        None => {
            let total = match a.as_slice() {
                Some(slice) => sum_slice(slice),
                None => sum_slice(&gather_elements(a)),
            };
            vec![total]
        }
        Some(axis) => {
            let shape = a.shape();
            // `outer*inner`（`axis_reduce` 内部の `total_out` 計算）は無検査の
            // `*` だと 0 次元を挟む shape でオーバーフローしうる（max/mean と
            // 同じ理由。Review 指摘 #23）。呼び出し前に checked_mul で検査し、
            // オーバーフロー時は panic ではなく型付きエラーを返す
            // （`.claude/rules/coding-rust.md`「本番経路で unwrap/expect を使わない」）。
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            axis_reduce_sum(a, axis)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// 軸指定・全縮約いずれにも対応する `max`。
///
/// 縮約対象の要素数が 0 の場合は [`ReduceError::EmptyReduction`] を返す
/// （単位元を持たないため。モジュール doc「空縮約の意味論」参照）。
pub fn max(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let data = match dim {
        None => {
            let result = match a.as_slice() {
                Some(slice) => max_slice(slice),
                None => max_slice(&gather_elements(a)),
            };
            match result {
                Some(v) => vec![v],
                None => return Err(ReduceError::EmptyReduction { op: "max" }),
            }
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "max" });
            }
            axis_reduce(a, axis, f32::NEG_INFINITY, f32::max)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// 軸指定・全縮約いずれにも対応する `min`（イシュー #1720）。[`max`]
/// と対称だが単位元 `f32::INFINITY`・`f32::min`（NaN 非伝播）を使う。
/// 縮約対象の要素数が 0 の場合は [`ReduceError::EmptyReduction`] を
/// 返す（単位元を持たないため。モジュール doc「空縮約の意味論」参照）。
pub fn min(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let data = match dim {
        None => {
            let result = match a.as_slice() {
                Some(slice) => min_slice(slice),
                None => min_slice(&gather_elements(a)),
            };
            match result {
                Some(v) => vec![v],
                None => return Err(ReduceError::EmptyReduction { op: "min" }),
            }
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "min" });
            }
            axis_reduce(a, axis, f32::INFINITY, f32::min)
        }
    };
    Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
}

/// `dim` 軸に沿った添字を線形走査で求める `argmax`／`argmin` 共通の
/// 出力要素ごとの畳み込み（イシュー #1720）。[`axis_reduce`] と異なり
/// 添字（`usize`）を返し、`better(v, best)` が `true` のときのみ更新
/// する（[`fandhe_ai_tensor_core::BackendOps::argmax`] doc の走査契約
/// 1〜2 と同型。呼び出し元〈本モジュール〉は決定性契約を保つため
/// [`axis_reduce`] と同じ出力要素側のみ並列化する構造を踏襲する）。
fn axis_arg_reduce(
    a: &Tensor<f32>,
    axis: usize,
    better: impl Fn(f32, f32) -> bool + Sync,
) -> Vec<usize> {
    let shape = a.shape();
    let outer_dims = &shape[..axis];
    let inner_dims = &shape[axis + 1..];
    let axis_len = shape[axis];
    let outer: usize = outer_dims.iter().product();
    let inner: usize = inner_dims.iter().product();
    let total_out = outer * inner;

    // `axis_reduce` と同じ契約（クロージャは `flat in 0..total_out` の
    // みで呼ばれ、`inner` によるゼロ除算は発生しない）。
    let compute = |flat: usize| -> usize {
        let (o, i) = (flat / inner, flat % inner);
        let outer_idx = unravel(o, outer_dims);
        let inner_idx = unravel(i, inner_dims);
        let mut full_idx = Vec::with_capacity(shape.len());
        full_idx.extend_from_slice(&outer_idx);
        full_idx.push(0);
        full_idx.extend_from_slice(&inner_idx);
        let mut best_idx = 0usize;
        let mut best_val = f32::NAN;
        for k in 0..axis_len {
            full_idx[axis] = k;
            let value = a.get(&full_idx);
            debug_assert!(
                value.is_some(),
                "axis_arg_reduce: 走査ロジックのバグにより index {full_idx:?} が範囲外になった"
            );
            let v = value.unwrap_or(f32::NAN);
            if v.is_nan() {
                continue;
            }
            if best_val.is_nan() || better(v, best_val) {
                best_val = v;
                best_idx = k;
            }
        }
        best_idx
    };

    map_outputs(total_out.saturating_mul(axis_len), total_out, compute)
}

/// `data`（全縮約対象。`gather_elements` 済みまたは `as_slice()` の
/// 借用）から [`axis_arg_reduce`] と同じ走査契約で単一の添字を求める
/// （`dim: None` の `argmax`／`argmin` が使う。空入力は `None`）。
fn arg_slice(data: &[f32], better: impl Fn(f32, f32) -> bool) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NAN;
    for (idx, &v) in data.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if best_val.is_nan() || better(v, best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    Some(best_idx)
}

/// 添字（`Vec<usize>`）を `i32` の `Tensor` へ変換する（`argmax`／
/// `argmin` 共通。`i32::MAX` を超える添字は `sort_topk.rs`／
/// `gather_scatter.rs` と同じ契約で
/// `ShapeError::IndexRangeOverflow` を返す）。
fn build_index_tensor(
    indices: Vec<usize>,
    out_shape: &[usize],
) -> Result<Tensor<i32>, ReduceError> {
    let mut out = Vec::with_capacity(indices.len());
    for idx in indices {
        out.push(
            i32::try_from(idx)
                .map_err(|_| ReduceError::Shape(ShapeError::IndexRangeOverflow { index: idx }))?,
        );
    }
    Tensor::new(out, out_shape).map_err(ReduceError::Shape)
}

/// 軸指定・全縮約いずれにも対応する `argmax`（イシュー #1720）。
/// 縮約対象の要素数が 0 の場合は [`ReduceError::EmptyReduction`] を
/// 返す（`min`／`max` と同じ方針。単位元を持たないため）。
pub fn argmax(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<i32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let indices = match dim {
        None => {
            let result = match a.as_slice() {
                Some(slice) => arg_slice(slice, |v, best| v > best),
                None => arg_slice(&gather_elements(a), |v, best| v > best),
            };
            match result {
                Some(idx) => vec![idx],
                None => return Err(ReduceError::EmptyReduction { op: "argmax" }),
            }
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "argmax" });
            }
            axis_arg_reduce(a, axis, |v, best| v > best)
        }
    };
    build_index_tensor(indices, &out_shape)
}

/// [`argmax`] の最小値版（イシュー #1720）。走査・空縮約・オーバー
/// フロー契約は [`argmax`] と対称（`v < best` のときのみ更新）。
pub fn argmin(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<i32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    let indices = match dim {
        None => {
            let result = match a.as_slice() {
                Some(slice) => arg_slice(slice, |v, best| v < best),
                None => arg_slice(&gather_elements(a), |v, best| v < best),
            };
            match result {
                Some(idx) => vec![idx],
                None => return Err(ReduceError::EmptyReduction { op: "argmin" }),
            }
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "argmin" });
            }
            axis_arg_reduce(a, axis, |v, best| v < best)
        }
    };
    build_index_tensor(indices, &out_shape)
}

/// 軸指定・全縮約いずれにも対応する `mean`。`sum` の結果を要素数で **1 回
/// だけ除算**する（丸め 1 回で決定性を維持する契約）。
///
/// 縮約対象の要素数が 0 の場合は [`ReduceError::EmptyReduction`] を返す
/// （`max` と同様、単位元を持たないため）。
pub fn mean(a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, ReduceError> {
    let out_shape = reduce_out_shape(a.shape(), dim).map_err(ReduceError::Shape)?;
    match dim {
        None => {
            let numel = a.numel();
            if numel == 0 {
                return Err(ReduceError::EmptyReduction { op: "mean" });
            }
            let total = match a.as_slice() {
                Some(slice) => sum_slice(slice),
                None => sum_slice(&gather_elements(a)),
            };
            Tensor::new(vec![total / numel as f32], &out_shape).map_err(ReduceError::Shape)
        }
        Some(axis) => {
            let shape = a.shape();
            let axis_len = shape[axis];
            let outer = checked_product(&shape[..axis])?;
            let inner = checked_product(&shape[axis + 1..])?;
            let total_out = outer
                .checked_mul(inner)
                .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))?;
            if axis_len == 0 && total_out > 0 {
                return Err(ReduceError::EmptyReduction { op: "mean" });
            }
            let sums = axis_reduce_sum(a, axis);
            let divisor = axis_len as f32;
            let data: Vec<f32> = sums.into_iter().map(|s| s / divisor).collect();
            Tensor::new(data, &out_shape).map_err(ReduceError::Shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_full_matches_naive() {
        let t = Tensor::<f32>::new((0..24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap();
        let out = sum(&t, None).unwrap();
        // 型注釈が必要な理由: 単体テストビルドは dev-dependency の
        // `bench_harness`（→ `serde_json`）を参照するため、`serde_json` の
        // `impl PartialEq<Value> for usize` が候補に載り空スライスリテラルの
        // 要素型を推論できない（elementwise.rs の `bench_internal` 移設で
        // 顕在化。PR #1066）。
        assert_eq!(out.shape(), &[] as &[usize]);
        let expected: f32 = (0..24).map(|v| v as f32).sum();
        assert_eq!(out.get(&[]).unwrap(), expected);
    }

    #[test]
    fn sum_axis_matches_expected() {
        // shape [2, 3]: [[0,1,2],[3,4,5]]
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let out0 = sum(&t, Some(0)).unwrap();
        assert_eq!(out0.shape(), &[3]);
        assert_eq!(out0.get(&[0]).unwrap(), 3.0); // 0+3
        assert_eq!(out0.get(&[1]).unwrap(), 5.0); // 1+4
        assert_eq!(out0.get(&[2]).unwrap(), 7.0); // 2+5

        let out1 = sum(&t, Some(1)).unwrap();
        assert_eq!(out1.shape(), &[2]);
        assert_eq!(out1.get(&[0]).unwrap(), 3.0); // 0+1+2
        assert_eq!(out1.get(&[1]).unwrap(), 12.0); // 3+4+5
    }

    #[test]
    fn max_axis_matches_expected() {
        let t = Tensor::<f32>::new(vec![1.0, 5.0, 3.0, 9.0, 2.0, 0.0], &[2, 3]).unwrap();
        let out = max(&t, Some(1)).unwrap();
        assert_eq!(out.get(&[0]).unwrap(), 5.0);
        assert_eq!(out.get(&[1]).unwrap(), 9.0);
    }

    // --- min／argmax／argmin（イシュー #1720） ---

    #[test]
    fn min_axis_matches_expected() {
        let t = Tensor::<f32>::new(vec![1.0, 5.0, 3.0, 9.0, 2.0, 0.0], &[2, 3]).unwrap();
        let out = min(&t, Some(1)).unwrap();
        assert_eq!(out.get(&[0]).unwrap(), 1.0);
        assert_eq!(out.get(&[1]).unwrap(), 0.0);
    }

    #[test]
    fn min_full_matches_naive() {
        let t = Tensor::<f32>::new(vec![3.0, -1.0, 5.0, -7.0, 2.0], &[5]).unwrap();
        let out = min(&t, None).unwrap();
        assert_eq!(out.get(&[]).unwrap(), -7.0);
    }

    #[test]
    fn min_empty_is_error() {
        let t = Tensor::<f32>::zeros(&[0]).unwrap();
        assert!(matches!(
            min(&t, None).unwrap_err(),
            ReduceError::EmptyReduction { op: "min" }
        ));
    }

    #[test]
    fn min_nan_is_non_propagating() {
        // `f32::min` は NaN 非伝播（`max` の `f32::max` と同じ流儀。
        // `reduction` モジュール doc・`BackendOps::min` doc の契約）。
        let t = Tensor::<f32>::new(vec![f32::NAN, 3.0, -1.0], &[3]).unwrap();
        let out = min(&t, None).unwrap();
        assert_eq!(out.get(&[]).unwrap(), -1.0);
    }

    #[test]
    fn argmax_and_argmin_full_and_axis_match_expected() {
        let t = Tensor::<f32>::new(vec![1.0, 5.0, 3.0, 9.0, 2.0, 0.0], &[2, 3]).unwrap();

        let amax_all = argmax(&t, None).unwrap();
        assert_eq!(amax_all.get(&[]).unwrap(), 3); // 9.0 at flat index 3
        let amin_all = argmin(&t, None).unwrap();
        assert_eq!(amin_all.get(&[]).unwrap(), 5); // 0.0 at flat index 5

        let amax_axis1 = argmax(&t, Some(1)).unwrap();
        assert_eq!(amax_axis1.get(&[0]).unwrap(), 1); // row0 max=5.0 at col1
        assert_eq!(amax_axis1.get(&[1]).unwrap(), 0); // row1 max=9.0 at col0
        let amin_axis1 = argmin(&t, Some(1)).unwrap();
        assert_eq!(amin_axis1.get(&[0]).unwrap(), 0); // row0 min=1.0 at col0
        assert_eq!(amin_axis1.get(&[1]).unwrap(), 2); // row1 min=0.0 at col2
    }

    #[test]
    fn argmax_tie_returns_first_index() {
        let t = Tensor::<f32>::new(vec![1.0, 5.0, 3.0, 5.0], &[4]).unwrap();
        let amax = argmax(&t, None).unwrap();
        assert_eq!(amax.get(&[]).unwrap(), 1, "同値タイは最初の添字を返す");
    }

    #[test]
    fn argmin_tie_returns_first_index() {
        let t = Tensor::<f32>::new(vec![3.0, -2.0, 1.0, -2.0], &[4]).unwrap();
        let amin = argmin(&t, None).unwrap();
        assert_eq!(amin.get(&[]).unwrap(), 1, "同値タイは最初の添字を返す");
    }

    #[test]
    fn argmax_ignores_nan_and_picks_valid_max() {
        // 先頭 NaN は無視し、後続の非 NaN 値のうち最大を選ぶ
        // （`BackendOps::argmax` doc の NaN 規約）。
        let t = Tensor::<f32>::new(vec![f32::NAN, 2.0, f32::NAN, 7.0, 1.0], &[5]).unwrap();
        let amax = argmax(&t, None).unwrap();
        assert_eq!(amax.get(&[]).unwrap(), 3);
    }

    #[test]
    fn argmax_all_nan_returns_index_zero() {
        let t = Tensor::<f32>::new(vec![f32::NAN, f32::NAN, f32::NAN], &[3]).unwrap();
        let amax = argmax(&t, None).unwrap();
        assert_eq!(amax.get(&[]).unwrap(), 0);
    }

    #[test]
    fn argmax_and_argmin_empty_are_errors() {
        let t = Tensor::<f32>::zeros(&[0]).unwrap();
        assert!(matches!(
            argmax(&t, None).unwrap_err(),
            ReduceError::EmptyReduction { op: "argmax" }
        ));
        assert!(matches!(
            argmin(&t, None).unwrap_err(),
            ReduceError::EmptyReduction { op: "argmin" }
        ));
    }

    #[test]
    fn min_chunk_boundary_deterministic() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build single-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        for n in [CHUNK - 1, CHUNK, CHUNK + 1, CHUNK * 2 + 1] {
            let data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.5 - 3.0).collect();
            let t = Tensor::<f32>::new(data, &[n]).unwrap();

            let min_a = single.install(|| min(&t, None).unwrap());
            let min_b = multi.install(|| min(&t, None).unwrap());
            assert_eq!(
                min_a.get(&[]).unwrap().to_bits(),
                min_b.get(&[]).unwrap().to_bits(),
                "min が n={n} でスレッド数間に不一致"
            );
        }
    }

    #[test]
    fn mean_matches_expected() {
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let out = mean(&t, None).unwrap();
        assert_eq!(out.get(&[]).unwrap(), 2.5);
    }

    #[test]
    fn axis_out_of_range_errors() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = sum(&t, Some(5)).unwrap_err();
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::AxisOutOfRange { axis: 5, rank: 2 })
        ));
    }

    #[test]
    fn empty_axis_sum_is_zero_but_max_mean_error() {
        // shape [2, 0, 4]: axis=1 is size 0, but outer*inner = 2*4 = 8 > 0.
        let t = Tensor::<f32>::zeros(&[2, 0, 4]).unwrap();
        let out = sum(&t, Some(1)).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        for v in 0..2 {
            for w in 0..4 {
                assert_eq!(out.get(&[v, w]).unwrap(), 0.0);
            }
        }
        assert!(matches!(
            max(&t, Some(1)).unwrap_err(),
            ReduceError::EmptyReduction { op: "max" }
        ));
        assert!(matches!(
            mean(&t, Some(1)).unwrap_err(),
            ReduceError::EmptyReduction { op: "mean" }
        ));
    }

    #[test]
    fn fully_empty_tensor_sum_zero_max_mean_error() {
        let t = Tensor::<f32>::zeros(&[0]).unwrap();
        let out = sum(&t, None).unwrap();
        assert_eq!(out.get(&[]).unwrap(), 0.0);
        assert!(matches!(
            max(&t, None).unwrap_err(),
            ReduceError::EmptyReduction { op: "max" }
        ));
        assert!(matches!(
            mean(&t, None).unwrap_err(),
            ReduceError::EmptyReduction { op: "mean" }
        ));
    }

    #[test]
    fn sum_axis_overflow_returns_error_not_panic() {
        // Review 指摘（#23）の再現ケース: shape [1<<40, 0, 1<<40], axis=1 は
        // `checked_numel`（tensor-core）が 0 次元で早期に 0 を経由するため
        // `Tensor::zeros` 自体は成功するが、`outer * inner`
        // （= (1<<40) * (1<<40)）は usize 上でオーバーフローする。
        // max/mean と同様、sum も panic ではなく型付きエラーを返す契約を検証する
        // （axis_reduce 呼び出し前の checked_mul 検査。reduction.rs:226 付近）。
        let t = Tensor::<f32>::zeros(&[1usize << 40, 0, 1usize << 40]).unwrap();
        let err = sum(&t, Some(1)).unwrap_err();
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    // --- `logsumexp`／`vector_norm_p` の確保前バイト数上限検査
    // （codex-review P1 是正・イシュー #2147・PR #2263）: 小さな
    // ストレージを巨大な shape へ broadcast した view は、要素数積が
    // `usize` に収まっても `f32` 換算のバイト数が `isize::MAX` を
    // 超えうる。`checked_alloc_numel_f32` が `Vec::with_capacity`／
    // `.collect()` 前に型付きエラーで拒否し panic しないことを検証する
    // （本質は panic しないこと。エラー内容は `ElementCountOverflow`
    // で共通）。

    #[test]
    fn logsumexp_vector_norm_p_axis_reduce_rejects_huge_broadcast_output_without_panicking() {
        // base shape [1, 4] を broadcast して [1usize << 61, 4] にする
        // （軸 1 は実軸〈長さ 4・非空縮約〉、軸 0 は broadcast で巨大）。
        // dim=Some(1) で縮約すると out_shape=[1usize << 61] となり、
        // 要素数積（2^61）自体は usize に収まるが f32 換算バイト数
        // （2^61 * 4 = 2^63）が isize::MAX（2^63 - 1）を 1 超える。
        let base = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61, 4]).unwrap();

        let err = logsumexp(&huge, Some(1)).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));

        let err = vector_norm_p(&huge, 3.0, Some(1)).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn logsumexp_vector_norm_p_full_reduce_rejects_huge_broadcast_input_without_panicking() {
        // base shape [1] を broadcast して [1usize << 61] にする
        // （非 contiguous・全縮約〈dim=None〉。`as_slice()` が `None`
        // を返すため `gather_elements` 経路に入る）。
        let base = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61]).unwrap();
        assert!(huge.as_slice().is_none(), "fixture は非 contiguous のはず");

        let err = logsumexp(&huge, None).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));

        let err = vector_norm_p(&huge, 3.0, None).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    // --- `vector_norm`（L1／L2）の確保前バイト数上限検査（イシュー
    // #2287。`logsumexp`／`vector_norm_p` と同型の fixture・同じ理由）。

    #[test]
    fn vector_norm_axis_reduce_rejects_huge_broadcast_output_without_panicking() {
        // base shape [1, 4] を broadcast して [1usize << 61, 4] にする
        // （軸 1 は実軸〈長さ 4・非空縮約〉、軸 0 は broadcast で巨大）。
        // dim=Some(1) で縮約すると out_shape=[1usize << 61] となり、
        // 要素数積（2^61）自体は usize に収まるが f32 換算バイト数
        // （2^61 * 4 = 2^63）が isize::MAX（2^63 - 1）を 1 超える。
        let base = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61, 4]).unwrap();

        for ord in [VectorNormOrd::L1, VectorNormOrd::L2] {
            let err = vector_norm(&huge, ord, Some(1)).expect_err("確保前に拒否されるはず");
            assert!(matches!(
                err,
                ReduceError::Shape(ShapeError::ElementCountOverflow)
            ));
        }
    }

    #[test]
    fn vector_norm_full_reduce_rejects_huge_broadcast_input_without_panicking() {
        // base shape [1] を broadcast して [1usize << 61] にする
        // （非 contiguous・全縮約〈dim=None〉。`as_slice()` が `None`
        // を返すため `gather_elements` 経路に入る）。
        let base = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61]).unwrap();
        assert!(huge.as_slice().is_none(), "fixture は非 contiguous のはず");

        for ord in [VectorNormOrd::L1, VectorNormOrd::L2] {
            let err = vector_norm(&huge, ord, None).expect_err("確保前に拒否されるはず");
            assert!(matches!(
                err,
                ReduceError::Shape(ShapeError::ElementCountOverflow)
            ));
        }
    }

    // --- `var` の確保前バイト数上限検査（イシュー #2288。
    // `vector_norm` と同型の fixture・同じ理由）。

    #[test]
    fn var_axis_reduce_rejects_huge_broadcast_output_without_panicking() {
        // base shape [1, 4] を broadcast して [1usize << 61, 4] にする
        // （軸 1 は実軸〈長さ 4・自由度十分〉、軸 0 は broadcast で
        // 巨大）。dim=Some(1) で縮約すると out_shape=[1usize << 61]
        // となり、`vector_norm` と同じ理由で f32 換算バイト数が
        // isize::MAX を超える。
        let base = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61, 4]).unwrap();

        let err = var(&huge, Some(1), 1).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn var_full_reduce_rejects_huge_broadcast_input_without_panicking() {
        // base shape [1] を broadcast して [1usize << 61] にする
        // （非 contiguous・全縮約〈dim=None〉。`as_slice()` が `None`
        // を返すため `gather_elements` 経路に入る）。
        let base = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61]).unwrap();
        assert!(huge.as_slice().is_none(), "fixture は非 contiguous のはず");

        let err = var(&huge, None, 1).expect_err("確保前に拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn var_axis_reduce_empty_reduction_with_huge_broadcast_out_shape_is_empty_reduction() {
        // 縮約対象軸自体は空（n == 0）だが out_shape が broadcast 由来
        // で巨大というケース。空縮約判定が確保前検査より先であること
        // を確認する（`n == 0`／`n <= correction` の判定順序は元々
        // 確保前検査より先にあるため、本テストは既存順序の回帰防止）。
        let base = Tensor::<f32>::new(Vec::new(), &[1, 0]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61, 0]).unwrap();

        let err = var(&huge, Some(1), 1).expect_err("空縮約で拒否されるはず");
        assert!(matches!(err, ReduceError::EmptyReduction { op: "var" }));
    }

    #[test]
    fn var_insufficient_dof_with_huge_broadcast_shape_is_insufficient_degrees_of_freedom() {
        // 縮約対象軸自体は自由度不足（n <= correction）だが out_shape
        // が broadcast 由来で巨大というケース。
        let base = Tensor::<f32>::new(vec![1.0], &[1, 1]).unwrap();
        let huge = base.broadcast_to(&[1usize << 61, 1]).unwrap();

        let err = var(&huge, Some(1), 1).expect_err("自由度不足で拒否されるはず");
        assert!(matches!(
            err,
            ReduceError::InsufficientDegreesOfFreedom {
                n: 1,
                correction: 1
            }
        ));
    }

    #[test]
    fn empty_axis_with_zero_outer_inner_no_error() {
        // shape [0, 0]: axis=1 の縮約対象は空だが outer*inner (= 0) も 0 の
        // ため出力自体が空 Tensor になり、EmptyReduction は発生しない。
        let t = Tensor::<f32>::zeros(&[0, 0]).unwrap();
        let out = max(&t, Some(1)).unwrap();
        assert_eq!(out.shape(), &[0]);
        assert!(out.is_empty());
    }

    #[test]
    fn chunk_boundary_deterministic_sum() {
        // CHUNK（4096）境界をまたぐ要素数で、シングルスレッド／マルチスレッド
        // プール双方の実行結果が to_bits() で完全一致することを検証する。
        let n = CHUNK * 3 + 17;
        let data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.5 - 3.0).collect();
        let t = Tensor::<f32>::new(data.clone(), &[n]).unwrap();

        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build single-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        let a = single.install(|| sum(&t, None).unwrap());
        let b = multi.install(|| sum(&t, None).unwrap());
        assert_eq!(a.get(&[]).unwrap().to_bits(), b.get(&[]).unwrap().to_bits());

        // 参照実装（同一累積順序の逐次 naive 実装）との bit 一致。浮動小数点
        // 加算は結合則を満たさないため、単純な左から右への fold ではなく
        // 本実装と同一の「CHUNK 単位でチャンク内を逐次累積 → チャンク結果を
        // 番号順に逐次結合」という累積順序を naive 側でも再現する。
        // `sum_slice` は `f64` アキュムレータ契約（イシュー #1675 是正）の
        // ため naive 側も `f64` で累積し最後に 1 回だけ `f32` へ downcast
        // する。
        let naive: f32 = data
            .chunks(CHUNK)
            .map(|chunk| chunk.iter().fold(0.0f64, |acc, &v| acc + v as f64))
            .fold(0.0f64, |acc, v| acc + v) as f32;
        assert_eq!(a.get(&[]).unwrap().to_bits(), naive.to_bits());
    }

    /// `max`／`mean` の CHUNK（4096）境界決定性（#25 棚卸しで特定した
    /// ギャップ）: 既存の `chunk_boundary_deterministic_sum` は `sum` のみを
    /// 検証しており、`max_slice` の `par_chunks(CHUNK)` 分割・`mean` の
    /// 「sum を経由し分母で 1 回だけ除算する」経路は未検証だった。
    /// `n ∈ {CHUNK-1, CHUNK, CHUNK+1, CHUNK*2+1}` でシングル／マルチスレッド
    /// プール間の to_bits() 完全一致を確認する。
    #[test]
    fn chunk_boundary_deterministic_max_and_mean() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build single-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        for n in [CHUNK - 1, CHUNK, CHUNK + 1, CHUNK * 2 + 1] {
            let data: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.5 - 3.0).collect();
            let t = Tensor::<f32>::new(data, &[n]).unwrap();

            let max_a = single.install(|| max(&t, None).unwrap());
            let max_b = multi.install(|| max(&t, None).unwrap());
            assert_eq!(
                max_a.get(&[]).unwrap().to_bits(),
                max_b.get(&[]).unwrap().to_bits(),
                "max が n={n} でスレッド数間に不一致"
            );

            let mean_a = single.install(|| mean(&t, None).unwrap());
            let mean_b = multi.install(|| mean(&t, None).unwrap());
            assert_eq!(
                mean_a.get(&[]).unwrap().to_bits(),
                mean_b.get(&[]).unwrap().to_bits(),
                "mean が n={n} でスレッド数間に不一致"
            );
        }
    }

    /// サイズ 1 軸の軸指定 reduction（#25 棚卸しで特定したギャップ）:
    /// `dim=Some(axis)` で当該軸長が 1 の場合、`sum`/`max` は恒等値
    /// （縮約対象が要素そのもの）、`mean` は除算 1 回（分母 1）で入力値と
    /// 一致することを確認する。
    #[test]
    fn axis_reduction_with_size_one_axis_is_identity() {
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

        let out_sum = sum(&t, Some(0)).unwrap();
        assert_eq!(out_sum.shape(), &[4]);
        for i in 0..4 {
            assert_eq!(out_sum.get(&[i]).unwrap(), t.get(&[0, i]).unwrap());
        }

        let out_max = max(&t, Some(0)).unwrap();
        for i in 0..4 {
            assert_eq!(out_max.get(&[i]).unwrap(), t.get(&[0, i]).unwrap());
        }

        let out_mean = mean(&t, Some(0)).unwrap();
        for i in 0..4 {
            assert_eq!(out_mean.get(&[i]).unwrap(), t.get(&[0, i]).unwrap());
        }
    }

    #[test]
    fn non_contiguous_axis_reduction_matches_contiguous() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap(); // shape [3, 2], non-contiguous
        assert!(tt.as_slice().is_none());
        let out = sum(&tt, Some(0)).unwrap();
        let expected = tt.contiguous();
        let out_c = sum(&expected, Some(0)).unwrap();
        assert_eq!(out.shape(), out_c.shape());
        for i in 0..out.shape()[0] {
            assert_eq!(out.get(&[i]).unwrap(), out_c.get(&[i]).unwrap());
        }
    }

    /// `crate::elementwise::PARALLEL_THRESHOLD` の直下・直上というサイズ
    /// （reduction 自体はこの閾値で分岐しない。イシュー #811・#1027・
    /// モジュール doc「小サイズ直列フォールバック」参照。他モジュールと
    /// 揃えた代表サイズとして流用するのみ）で、全縮約（`dim=None`）の
    /// `sum`/`max` がシングルスレッド／マルチスレッドプール間で
    /// to_bits() 完全一致することを確認する（`chunk_boundary_deterministic_sum`／
    /// `chunk_boundary_deterministic_max_and_mean` と同じ検証方針）。
    #[test]
    fn parallel_threshold_boundary_deterministic_full_reduction() {
        let threshold = crate::elementwise::PARALLEL_THRESHOLD;
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build single-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        for n in [threshold - 1, threshold, threshold + 1] {
            let data: Vec<f32> = (0..n).map(|i| ((i % 131) as f32) * 0.25 - 7.0).collect();
            let t = Tensor::<f32>::new(data.clone(), &[n]).unwrap();

            let sum_a = single.install(|| sum(&t, None).unwrap());
            let sum_b = multi.install(|| sum(&t, None).unwrap());
            assert_eq!(
                sum_a.get(&[]).unwrap().to_bits(),
                sum_b.get(&[]).unwrap().to_bits(),
                "sum が PARALLEL_THRESHOLD 境界 n={n} でスレッド数間に不一致"
            );

            let max_a = single.install(|| max(&t, None).unwrap());
            let max_b = multi.install(|| max(&t, None).unwrap());
            assert_eq!(
                max_a.get(&[]).unwrap().to_bits(),
                max_b.get(&[]).unwrap().to_bits(),
                "max が PARALLEL_THRESHOLD 境界 n={n} でスレッド数間に不一致"
            );

            // sum_slice は既定（ゲート OFF）で常に par_chunks 経由（モジュール
            // doc「小サイズ直列フォールバック」参照）で
            // CHUNK 単位の逐次 fold をチャンク番号順に結合する構造のため、
            // `chunk_boundary_deterministic_sum` と同じ naive 実装
            // （本実装と同一の累積順序。`f64` アキュムレータ契約）との bit
            // 一致で当該構造の正しさも確認する。
            let naive: f32 = data
                .chunks(CHUNK)
                .map(|chunk| chunk.iter().fold(0.0f64, |acc, &v| acc + v as f64))
                .fold(0.0f64, |acc, v| acc + v) as f32;
            assert_eq!(sum_a.get(&[]).unwrap().to_bits(), naive.to_bits());
        }
    }

    /// `crate::elementwise::PARALLEL_THRESHOLD` 相当のサイズ（reduction 自体
    /// はこの閾値で分岐しない。上記テスト同様、代表サイズとして流用する
    /// のみ）で、軸指定 reduction（`axis_reduce`）がシングルスレッド／
    /// マルチスレッドプール間で bit 完全一致することを確認する。
    /// shape は `[outer, axis_len]`（`axis=1`）とし、`outer` は各
    /// `total_in` の約数（`32,767 = 7 × 4,681`・`32,768 = 4 × 8,192`・
    /// `32,769 = 3 × 10,923`）を取って丸めなしで正確に `total_in` 要素の
    /// 入力を構成する（当初の `outer = 4` 固定 + `div_ceil` 丸めでは
    /// 閾値直下 32,767 が 32,768 要素へ丸まり閾値ちょうどと同一 shape に
    /// なっていた。PR #1066 codex-review P2 対応）。
    #[test]
    fn parallel_threshold_boundary_deterministic_axis_reduction() {
        let threshold = crate::elementwise::PARALLEL_THRESHOLD;
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build single-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        for (total_in, outer) in [(threshold - 1, 7usize), (threshold, 4), (threshold + 1, 3)] {
            assert_eq!(
                total_in % outer,
                0,
                "テスト前提の破れ: outer={outer} が total_in={total_in} の約数でない"
            );
            let axis_len = total_in / outer;
            let data: Vec<f32> = (0..total_in)
                .map(|i| ((i % 97) as f32) * 0.5 - 3.0)
                .collect();
            let t = Tensor::<f32>::new(data, &[outer, axis_len]).unwrap();

            let sum_a = single.install(|| sum(&t, Some(1)).unwrap());
            let sum_b = multi.install(|| sum(&t, Some(1)).unwrap());
            for i in 0..outer {
                assert_eq!(
                    sum_a.get(&[i]).unwrap().to_bits(),
                    sum_b.get(&[i]).unwrap().to_bits(),
                    "axis sum が total_in={total_in}（outer={outer}, axis_len={axis_len}）\
                     でスレッド数間に不一致（i={i}）"
                );
            }
        }
    }

    /// `var(dim=None, correction=1)`（不偏分散）が既知値と一致することを
    /// 確認する（イシュー #1723）。`[1,2,3,4]` の不偏分散は 5/3。
    #[test]
    fn var_full_matches_known_value() {
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let out = var(&t, None, 1).unwrap();
        assert_eq!(out.shape(), &[] as &[usize]);
        let expected = 5.0 / 3.0;
        assert!((out.get(&[]).unwrap() - expected).abs() < 1e-6);
    }

    /// `var(dim=None, correction=0)`（母分散）が既知値と一致することを
    /// 確認する。`[1,2,3,4]` の母分散は 1.25。
    #[test]
    fn var_full_correction_zero_matches_known_value() {
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let out = var(&t, None, 0).unwrap();
        let expected = 1.25;
        assert!((out.get(&[]).unwrap() - expected).abs() < 1e-6);
    }

    /// `var(dim=Some(axis))` が軸ごとに独立して計算されることを確認する。
    #[test]
    fn var_axis_matches_expected() {
        // shape [2, 3]: [[1,2,3],[4,4,4]]（行 1 は定数 = 分散 0）
        let t = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 4.0, 4.0], &[2, 3]).unwrap();
        let out = var(&t, Some(1), 0).unwrap();
        assert_eq!(out.shape(), &[2]);
        // 母分散: [1,2,3] の平均 2・二乗和 (1+0+1)=2・/3 = 2/3
        assert!((out.get(&[0]).unwrap() - (2.0 / 3.0)).abs() < 1e-6);
        assert!((out.get(&[1]).unwrap() - 0.0).abs() < 1e-6);
    }

    /// `n == 0`（空縮約）が `EmptyReduction` を返すことを確認する。
    #[test]
    fn var_empty_reduction_is_error() {
        let t = Tensor::<f32>::new(Vec::<f32>::new(), &[0]).unwrap();
        let err = var(&t, None, 1).unwrap_err();
        assert!(matches!(err, ReduceError::EmptyReduction { op: "var" }));
    }

    /// `n <= correction`（自由度不足）が `InsufficientDegreesOfFreedom` を
    /// 返すことを確認する（`n=1, correction=1` の境界値）。
    #[test]
    fn var_insufficient_degrees_of_freedom_is_error() {
        let t = Tensor::<f32>::new(vec![1.0], &[1]).unwrap();
        let err = var(&t, None, 1).unwrap_err();
        assert!(matches!(
            err,
            ReduceError::InsufficientDegreesOfFreedom {
                n: 1,
                correction: 1
            }
        ));
    }

    /// `vector_norm(L1)` が既知値と一致することを確認する。
    #[test]
    fn vector_norm_l1_full_matches_known_value() {
        let t = Tensor::<f32>::new(vec![-1.0, 2.0, -3.0, 4.0], &[4]).unwrap();
        let out = vector_norm(&t, VectorNormOrd::L1, None).unwrap();
        assert!((out.get(&[]).unwrap() - 10.0).abs() < 1e-6);
    }

    /// `vector_norm(L2)` が既知値と一致することを確認する（3-4-5 の直角三角形）。
    #[test]
    fn vector_norm_l2_full_matches_known_value() {
        let t = Tensor::<f32>::new(vec![3.0, 4.0], &[2]).unwrap();
        let out = vector_norm(&t, VectorNormOrd::L2, None).unwrap();
        assert!((out.get(&[]).unwrap() - 5.0).abs() < 1e-6);
    }

    /// `vector_norm(dim=Some(axis))` が軸ごとに独立して計算されることを
    /// 確認する。
    #[test]
    fn vector_norm_axis_matches_expected() {
        // shape [2, 2]: [[3,4],[-1,-1]]
        let t = Tensor::<f32>::new(vec![3.0, 4.0, -1.0, -1.0], &[2, 2]).unwrap();
        let out_l2 = vector_norm(&t, VectorNormOrd::L2, Some(1)).unwrap();
        assert!((out_l2.get(&[0]).unwrap() - 5.0).abs() < 1e-6);
        assert!((out_l2.get(&[1]).unwrap() - 2.0f32.sqrt()).abs() < 1e-6);

        let out_l1 = vector_norm(&t, VectorNormOrd::L1, Some(1)).unwrap();
        assert!((out_l1.get(&[0]).unwrap() - 7.0).abs() < 1e-6);
        assert!((out_l1.get(&[1]).unwrap() - 2.0).abs() < 1e-6);
    }

    /// `vector_norm` の空縮約が `EmptyReduction` を返すことを確認する。
    #[test]
    fn vector_norm_empty_reduction_is_error() {
        let t = Tensor::<f32>::new(Vec::<f32>::new(), &[0]).unwrap();
        let err = vector_norm(&t, VectorNormOrd::L1, None).unwrap_err();
        assert!(matches!(err, ReduceError::EmptyReduction { op: "norm" }));
    }

    /// `var`（軸指定）が single/multi スレッドで bit 完全一致する
    /// （決定性契約。既存 `chunk_boundary_deterministic_sum` と同型）。
    #[test]
    fn var_axis_deterministic_across_thread_counts() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("failed to build 1-thread rayon pool for determinism test");
        let multi = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("failed to build 4-thread rayon pool for determinism test");

        let outer = 5usize;
        let axis_len = 37usize;
        let data: Vec<f32> = (0..outer * axis_len)
            .map(|i| ((i % 53) as f32) * 0.37 - 9.0)
            .collect();
        let t = Tensor::<f32>::new(data, &[outer, axis_len]).unwrap();

        let a = single.install(|| var(&t, Some(1), 1).unwrap());
        let b = multi.install(|| var(&t, Some(1), 1).unwrap());
        for i in 0..outer {
            assert_eq!(
                a.get(&[i]).unwrap().to_bits(),
                b.get(&[i]).unwrap().to_bits(),
                "var axis がスレッド数間に不一致（i={i}）"
            );
        }
    }

    // ---- 逐次フォールバック機構（イシュー #2101）の回帰テスト ----

    const FORCE_SEQ: SeqPolicy = SeqPolicy {
        enabled: true,
        min_elems: usize::MAX,
    };
    const FORCE_PAR: SeqPolicy = SeqPolicy {
        enabled: false,
        min_elems: 0,
    };

    /// `policy` を呼び出しスレッドへ強制して `f` を実行する。
    fn with_policy<R>(policy: SeqPolicy, f: impl FnOnce() -> R) -> R {
        POLICY_OVERRIDE.with(|c| c.set(Some(policy)));
        let r = f();
        POLICY_OVERRIDE.with(|c| c.set(None));
        r
    }

    fn bits(t: &Tensor<f32>) -> Vec<u32> {
        let n: usize = t.shape().iter().product();
        let mut out = Vec::with_capacity(n);
        for flat in 0..n {
            let idx = unravel(flat, t.shape());
            out.push(t.get(&idx).unwrap().to_bits());
        }
        out
    }

    fn seeded(n: usize, salt: u64) -> Vec<f32> {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
            })
            .collect()
    }

    #[test]
    fn sequential_fallback_gate_is_off_by_default() {
        assert!(!std::hint::black_box(REDUCTION_SEQUENTIAL_FALLBACK_ENABLED));
        for n in [
            0usize,
            1,
            640,
            4096,
            REDUCTION_PARALLEL_MIN_ELEMS - 1,
            usize::MAX,
        ] {
            assert!(!SeqPolicy::DEFAULT.run_sequential(n));
        }
        assert!(FORCE_SEQ.run_sequential(usize::MAX - 1));
        assert!(!FORCE_PAR.run_sequential(0));
    }

    fn check_all_arms<F>(t: &Tensor<f32>, dim: Option<usize>, ctx: &str, f: F)
    where
        F: Fn(&Tensor<f32>, Option<usize>) -> Result<Tensor<f32>, ReduceError>,
    {
        let seq = with_policy(FORCE_SEQ, || f(t, dim));
        let par = with_policy(FORCE_PAR, || f(t, dim));
        let def = f(t, dim);
        match (seq, par, def) {
            (Ok(s), Ok(p), Ok(d)) => {
                assert_eq!(bits(&s), bits(&p), "seq/par 不一致 {ctx}");
                assert_eq!(bits(&s), bits(&d), "seq/default 不一致 {ctx}");
            }
            (Err(_), Err(_), Err(_)) => {}
            _ => panic!("結果種別が腕間で異なる {ctx}"),
        }
    }

    #[test]
    fn forced_seq_and_par_are_bit_identical_full_reduction() {
        let sizes = [
            0usize, 1, 2, 639, 640, 641, 4095, 4096, 4097, 8193, 32767, 32768, 32769, 65537,
            262143, 262144, 262145,
        ];
        for &n in &sizes {
            let mut inputs = vec![seeded(n, n as u64)];
            if n >= 3 {
                let mut special = seeded(n, 7);
                special[0] = 1e8;
                special[1] = 1.0;
                special[2] = -1e8;
                inputs.push(special.clone());
                let mut sp = special;
                sp[n / 2] = f32::NAN;
                sp[n - 1] = -0.0;
                inputs.push(sp);
                let mut inf = seeded(n, 9);
                inf[0] = f32::INFINITY;
                inf[n - 1] = f32::MIN_POSITIVE / 4.0;
                inputs.push(inf);
            }
            for data in inputs {
                let t = Tensor::<f32>::new(data, &[n]).unwrap();
                let ctx = format!("n={n}");
                check_all_arms(&t, None, &ctx, sum);
                check_all_arms(&t, None, &ctx, max);
                check_all_arms(&t, None, &ctx, min);
                check_all_arms(&t, None, &ctx, mean);
                check_all_arms(&t, None, &ctx, |a, d| var(a, d, 1));
                check_all_arms(&t, None, &ctx, |a, d| vector_norm(a, VectorNormOrd::L2, d));
                check_all_arms(&t, None, &ctx, |a, d| vector_norm(a, VectorNormOrd::L1, d));
            }
        }
    }

    #[test]
    fn forced_seq_and_par_are_bit_identical_axis_reduction() {
        for shape in [
            vec![7usize, 4681],
            vec![64, 10],
            vec![3, 5, 8],
            vec![512, 513],
            vec![2, 3, 4, 5],
        ] {
            let n: usize = shape.iter().product();
            let t = Tensor::<f32>::new(seeded(n, 3), &shape).unwrap();
            for axis in 0..shape.len() {
                let ctx = format!("shape={shape:?} axis={axis}");
                let d = Some(axis);
                check_all_arms(&t, d, &ctx, sum);
                check_all_arms(&t, d, &ctx, max);
                check_all_arms(&t, d, &ctx, min);
                check_all_arms(&t, d, &ctx, mean);
                check_all_arms(&t, d, &ctx, |a, d| var(a, d, 1));
                check_all_arms(&t, d, &ctx, |a, d| vector_norm(a, VectorNormOrd::L2, d));
                check_all_arms(&t, d, &ctx, logsumexp);
                let s = with_policy(FORCE_SEQ, || argmax(&t, d).unwrap());
                let p = with_policy(FORCE_PAR, || argmax(&t, d).unwrap());
                let m = with_policy(FORCE_SEQ, || argmin(&t, d).unwrap());
                let q = with_policy(FORCE_PAR, || argmin(&t, d).unwrap());
                assert_eq!(s.shape(), p.shape());
                let cnt: usize = s.shape().iter().product();
                for flat in 0..cnt {
                    let idx = unravel(flat, s.shape());
                    assert_eq!(s.get(&idx), p.get(&idx), "argmax {ctx}");
                    assert_eq!(m.get(&idx), q.get(&idx), "argmin {ctx}");
                }
            }
        }
    }

    #[test]
    fn forced_seq_and_par_are_bit_identical_non_contiguous_views() {
        let t = Tensor::<f32>::new(seeded(6 * 5000, 11), &[6, 5000]).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        check_all_arms(&tr, None, "transpose full", sum);
        check_all_arms(&tr, Some(0), "transpose axis0", sum);
        check_all_arms(&tr, Some(1), "transpose axis1", mean);
        let small = Tensor::<f32>::new(seeded(5000, 13), &[1, 5000]).unwrap();
        let bc = small.broadcast_to(&[6, 5000]).unwrap();
        check_all_arms(&bc, None, "broadcast full", sum);
        check_all_arms(&bc, Some(0), "broadcast axis0", max);
        check_all_arms(&bc, Some(1), "broadcast axis1", sum);
    }

    #[test]
    fn forced_policies_are_thread_count_independent() {
        let t = Tensor::<f32>::new(seeded(20000, 5), &[20000]).unwrap();
        let base = bits(&sum(&t, None).unwrap());
        for threads in [1usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for policy in [FORCE_SEQ, FORCE_PAR] {
                let got = pool.install(|| with_policy(policy, || bits(&sum(&t, None).unwrap())));
                assert_eq!(base, got, "threads={threads} policy={policy:?}");
            }
        }
    }

    /// 実機（M4 Max・GB10。CI 非対象）でのしきい値候補スイープ。手順・判定
    /// 規則は `docs/perf/logs/elemental-reduction-threshold-2101/RULE.txt`
    /// （事前登録。判定は #2102）。
    #[test]
    #[ignore]
    fn reduction_threshold_sweep() {
        use std::hint::black_box;
        use std::time::Instant;

        const SIZES: &[usize] = &[640, 2560, 4096, 8192, 16384, 32768, 65536, 131072, 262144];
        const WARMUP: usize = 50;
        const ITERS: usize = 1000;
        const ROWS: usize = 64;

        eprintln!("threads={}", rayon::current_num_threads());
        for &n in SIZES {
            let flat = Tensor::<f32>::new(seeded(n, 1), &[n]).unwrap();
            let mat = Tensor::<f32>::new(seeded(ROWS * (n / ROWS), 2), &[ROWS, n / ROWS]).unwrap();
            let cases: [(&str, &Tensor<f32>, Option<usize>, u8); 5] = [
                ("sum_all", &flat, None, 0),
                ("sum_axis0", &mat, Some(0), 0),
                ("sum_axis1", &mat, Some(1), 0),
                ("max_all", &flat, None, 1),
                ("mean_all", &flat, None, 2),
            ];
            for (name, t, dim, op) in cases {
                for (arm, policy) in [("seq", FORCE_SEQ), ("par", FORCE_PAR)] {
                    let run = || {
                        match op {
                            0 => sum(t, dim),
                            1 => max(t, dim),
                            _ => mean(t, dim),
                        }
                        .unwrap()
                    };
                    with_policy(policy, || {
                        for _ in 0..WARMUP {
                            black_box(run());
                        }
                        let mut samples = Vec::with_capacity(ITERS);
                        let mut out = run();
                        for _ in 0..ITERS {
                            let start = Instant::now();
                            out = black_box(run());
                            samples.push(start.elapsed().as_nanos());
                        }
                        samples.sort_unstable();
                        let checksum: u64 = bits(&out)
                            .iter()
                            .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(*b as u64));
                        eprintln!(
                            "op={name} n={n} arm={arm} median_ns={} threads={} checksum={checksum:#x}",
                            samples[samples.len() / 2],
                            rayon::current_num_threads()
                        );
                    });
                }
            }
        }
    }
}
