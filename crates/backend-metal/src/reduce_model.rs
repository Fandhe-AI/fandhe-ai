//! `shaders/reduce.metal`（f32 `sum` reduction。全要素・単一軸。
//! イシュー #1895・親 #1894）のホスト側逐語モデル。
//!
//! `crate::scan_model`・`crate::batch_norm_model` と同じ設計判断:
//! `crate::soft_f64` の binary64 ソフトウェアエミュレーション
//! （[`crate::soft_f64::widen_f32_bits`]／[`crate::soft_f64::
//! add_f64_bits`]／[`crate::soft_f64::narrow_f64_bits`]）を
//! `reduce.metal::red_f64_*` と同じ演算列で呼び出すことで、GPU 側
//! カーネルが正しい binary64 逐次演算列を実行することを Mac 実機に
//! 到達できない環境（本実装環境。Linux・CI）でも機械的に裏付ける
//! （`objc2` 系 FFI に触れないため `cfg(target_os = "macos")` を付け
//! ない）。
//!
//! # CPU 参照実装との演算順序の対応（重要）
//!
//! `fandhe_ai_backend_cpu::reduction::sum` は 2 つの異なる構造を持つ
//! （`crates/backend-cpu/src/reduction.rs` モジュール doc「決定性
//! 契約」参照）。本モジュールはその両方を逐語再現する:
//!
//! - **全要素（`dim=None`）**: `sum_slice_f64` は `data` を
//!   [`REDUCE_SUM_CHUNK`]（4096）単位のチャンクへ分割し、各チャンク
//!   内を `0.0f64` から index 順に逐次加算したうえで、チャンク部分和
//!   （まだ `f32` へ downcast しない `f64` のまま）を**チャンク番号順に
//!   `0.0f64` から逐次加算**し、最後に 1 回だけ `f32` へ downcast
//!   する。単純な平坦逐次和（1 スレッドで先頭から末尾まで通しで
//!   加算する方式）とは `numel > CHUNK` で一般に bit が一致しない
//!   （[`sum_all_soft_f64`] のチャンク境界感度テストが機械的に
//!   区別する）。
//! - **単一軸（`dim=Some(axis)`）**: `axis_reduce_sum` は出力要素
//!   （`outer × inner` 個）ごとに独立して `0.0f64` から縮約軸を昇順
//!   に逐次加算し、`f32` へ downcast する（平坦逐次和と同一構造。
//!   [`sum_axis_lane_soft_f64`] は [`crate::soft_f64::
//!   sequential_sum_f32_bits`] と同値だが、`reduce.metal::
//!   reduce_sum_axis_f32` との 1 対 1 対応を明示するために本モジュール
//!   でも別名で定義する）。
//!
//! 空縮約の意味論（`numel==0` → `0.0`・`axis_len==0` → 各出力
//! `0.0`）・`-0.0` のみの入力（`+0.0` になる）は CPU 側の契約を
//! そのまま継承する。
//!
//! # 結線について
//!
//! 本モジュールは `MetalReduce`（`crate::reduce`）の正しさを裏付ける
//! ホスト側参照実装・`plan_reduce_all`／`plan_reduce_axis`（`ops.rs::
//! MetalBackendOps::sum` が起動前に呼ぶ事前検証ヘルパー）を提供する。
//! `MetalBackendOps::sum` への結線（`context_cache::cached_reduce`・
//! `ops.rs` からの呼び出し）はイシュー #1896 で完了済み。

use crate::soft_f64::{add_f64_bits, narrow_f64_bits, sequential_sum_f32_bits, widen_f32_bits};

/// `argmax`／`argmin`（イシュー #1951）の走査対象（`max`／`min`）。
/// `reduce.metal::reduce_arg_*` の `constant uint& mode`（0=Max・
/// 1=Min）と 1 対 1 対応する（カーネル数を抑えるための切替引数。
/// モジュール doc §2「数値契約の核心」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgExtKind {
    Max,
    Min,
}

/// `v`（`f32` の bit 表現）を「NaN は除外（`None`）・±0 は同値化」した
/// `u32` の totalOrder 風キーへ変換する。`crate::sort_model::value_key`
/// と同じ正規化式（`±0` 同値化・符号ビット反転による単調写像）を使うが、
/// NaN の扱いが異なる（`sort_model::value_key` は NaN を最大キーへ
/// 写像するのに対し、本関数は argmax／argmin の「NaN は無視」契約
/// （`fandhe_ai_tensor_core::BackendOps::argmax` doc の走査契約 2）に
/// 合わせて `None` を返す）ため、`sort_model` の関数を再利用せず独立に
/// 定義する（`docs/backend-metal-reduce-sum-design.md` §11 参照）。
///
/// 返すキーは非 NaN の `f32` 全順序（`partial_cmp`）と同型（単調）で
/// あることを本モジュール下部の単体テストで検証する。GPU 側（`reduce.
/// metal::red_arg_key`）はこの関数のビット演算のみを逐語再現し、
/// `f32` の `<`／`>`・`isnan(`（GPU 上で非正規化数が flush されうる
/// ため float 比較そのものは使わない）は用いない。
pub fn arg_key(bits: u32) -> Option<u32> {
    let v = f32::from_bits(bits);
    if v.is_nan() {
        return None;
    }
    let normalized = if v == 0.0 { 0.0f32 } else { v };
    let nbits = normalized.to_bits();
    Some(if nbits >> 31 == 1 {
        !nbits
    } else {
        nbits | 0x8000_0000
    })
}

/// `kind` に応じて `challenger` が `incumbent`（現在の最良値。`None`
/// は「まだ何も採用していない」）より優れているかを判定する
/// （`axis_arg_reduce`／`arg_slice`〈`crates/backend-cpu/src/
/// reduction.rs`〉の `best_val.is_nan() || better(v, best_val)` と同じ
/// 「未確定または strict に優れる場合のみ更新」契約）。
fn arg_better(kind: ArgExtKind, challenger: u32, incumbent: Option<u32>) -> bool {
    match incumbent {
        None => true,
        Some(b) => match kind {
            ArgExtKind::Max => challenger > b,
            ArgExtKind::Min => challenger < b,
        },
    }
}

/// `reduce.metal::reduce_arg_axis_f32` の 1 lane 分のループ本体の逐語
/// モデル（`fandhe_ai_backend_cpu::reduction::axis_arg_reduce` の
/// 出力要素ごとの畳み込みと同一構造）。`xs` はライン内を軸順に並べた
/// スライス。NaN はスキップし、`best` が未確定または strict に優れる
/// 場合のみ更新する。全要素 NaN の場合は添字 0 を返す（`best_idx` の
/// 初期値のまま。CPU `axis_arg_reduce` と同じ契約）。
pub fn argext_axis_lane(xs: &[f32], kind: ArgExtKind) -> usize {
    let mut best_idx = 0usize;
    let mut best_key: Option<u32> = None;
    for (k, &v) in xs.iter().enumerate() {
        let Some(key) = arg_key(v.to_bits()) else {
            continue;
        };
        if arg_better(kind, key, best_key) {
            best_key = Some(key);
            best_idx = k;
        }
    }
    best_idx
}

/// 全要素縮約（`dim=None`）の 2 段構成モデル（`reduce.metal::
/// reduce_arg_all_chunk_f32`／`reduce_arg_all_finalize_f32` の逐語
/// モデル）。
///
/// CPU 参照実装 `fandhe_ai_backend_cpu::reduction::arg_slice` は
/// **平坦な逐次走査**（`sum` の `sum_slice_f64` と異なりチャンク分割
/// しない）だが、本関数は GPU 側の並列度確保のため
/// [`REDUCE_SUM_CHUNK`] 単位のチャンクへ分割し、
/// (1) 各チャンク内を昇順走査して「チャンク内で最初に現れる極値」の
/// (キー, グローバル添字) を求め（チャンクが全要素 NaN なら
/// 「無効」）、
/// (2) チャンク番号昇順に走査し、無効チャンクはスキップ、`best` が
/// 未確定または strict に優れる場合のみ置換する（同値なら先行
/// チャンクを保持）
/// という 2 段構成で結果を求める。
///
/// **等価性の証明**（`docs/backend-metal-reduce-sum-design.md` §11
/// 参照）: 平坦走査の結果は「値が最良でその中で最小添字を持つ要素」
/// である。(1) はチャンク内で最小添字（先勝ち）を選ぶ。(2) の strict
/// 置換は同値のとき先行チャンク（＝より小さい添字）を保持するため、
/// チャンクをまたいでも「先勝ち」が保たれる。よって 2 段構成の結果は
/// 平坦走査と常に一致する（両者は演算順序が異なるが比較は丸めを
/// 伴わない厳密演算のため結合順序に依存しない。下記テスト
/// `argext_all_chunked_matches_flat_scan` が全域で機械的に検証する）。
pub fn argext_all_chunked(x: &[f32], kind: ArgExtKind) -> usize {
    let mut global_best_idx = 0usize;
    let mut global_best_key: Option<u32> = None;
    for (chunk_idx, chunk) in x.chunks(REDUCE_SUM_CHUNK).enumerate() {
        let base = chunk_idx * REDUCE_SUM_CHUNK;
        let mut local_best: Option<(u32, usize)> = None;
        for (k, &v) in chunk.iter().enumerate() {
            let Some(key) = arg_key(v.to_bits()) else {
                continue;
            };
            let better = match local_best {
                None => true,
                Some((bk, _)) => arg_better(kind, key, Some(bk)),
            };
            if better {
                local_best = Some((key, k));
            }
        }
        if let Some((key, k)) = local_best {
            let global_idx = base + k;
            if arg_better(kind, key, global_best_key) {
                global_best_key = Some(key);
                global_best_idx = global_idx;
            }
        }
    }
    global_best_idx
}

/// [`argext_all_chunked`] の平坦走査版参照実装（CPU `arg_slice` と
/// 同一構造。`argext_all_chunked` との一致を検証するためのテスト
/// 専用ヘルパー）。
#[cfg(test)]
fn argext_all_flat(x: &[f32], kind: ArgExtKind) -> usize {
    let mut best_idx = 0usize;
    let mut best_key: Option<u32> = None;
    for (k, &v) in x.iter().enumerate() {
        let Some(key) = arg_key(v.to_bits()) else {
            continue;
        };
        if arg_better(kind, key, best_key) {
            best_key = Some(key);
            best_idx = k;
        }
    }
    best_idx
}

// [`plan_argext_all`]／[`plan_argext_axis`] の失敗理由。[`argext_all_chunked`]
// のグローバル添字は `numel` 未満、[`argext_axis_lane`] の添字は
// `axis_len` 未満であり、CPU `build_index_tensor` の `IndexRangeOverflow`
// と同じくデータに依存せず「形状だけから」 `i32` 範囲超過の可能性を
// 判定できる（実際に選ばれる添字がその範囲に収まるかはデータ依存だが、
// 形状上その可能性がある場合は `sum` と同じ `Unsupported` 経由の
// ホストフォールバックへ委ねる。`docs/backend-metal-reduce-sum-design.md`
// §11「サイズ上限・エラー契約」参照）。エラー型自体は既存の
// `ReducePrepareError` をそのまま再利用する（専用型を新設しない）。

/// 全要素縮約（`dim=None`）の起動計画。[`plan_reduce_all`] に加えて
/// `numel > i32::MAX` を検査する（候補添字が `i32` 範囲を超えうる
/// ため。`ops.rs::metal_argext` が `Unsupported` へ写像しホスト
/// フォールバックへ委ねる）。
pub fn plan_argext_all(numel: usize) -> Result<ReduceAllPlan, ReducePrepareError> {
    let plan = plan_reduce_all(numel)?;
    if numel > i32::MAX as usize {
        return Err(ReducePrepareError::SizeLimitExceeded {
            what: "numel (i32 index range)",
            value: numel,
            limit: i32::MAX as usize,
        });
    }
    Ok(plan)
}

/// 単一軸縮約（`dim=Some(axis)`）の起動計画。[`plan_reduce_axis`] に
/// 加えて `axis_len > i32::MAX` を検査する（候補添字は軸内添字
/// `0..axis_len` のため）。
pub fn plan_argext_axis(shape: &[usize], dim: usize) -> Result<ReduceAxisPlan, ReducePrepareError> {
    let plan = plan_reduce_axis(shape, dim)?;
    if plan.axis_len > i32::MAX as usize {
        return Err(ReducePrepareError::SizeLimitExceeded {
            what: "axis_len (i32 index range)",
            value: plan.axis_len,
            limit: i32::MAX as usize,
        });
    }
    Ok(plan)
}

/// 全要素縮約（`dim=None`）のチャンク分割サイズ。`fandhe_ai_backend_cpu::
/// reduction::CHUNK` と同値（同モジュールは `pub(crate)` のため本
/// クレートから直接参照できない。ドリフトは
/// `crates/backend-metal/tests/reduce_source_evidence.rs`（MSL 側
/// リテラルとの一致検査）と `reduce_model` 単体テスト（チャンク境界
/// 感度テスト。本モジュール下部）の両方で検出する）。
pub const REDUCE_SUM_CHUNK: usize = 4096;

/// reduction カーネルの `uint` 引数（`numel`／`num_chunks`／`lanes`／
/// `axis_len`／`inner`）が収まるバックエンド固有上限（`u32::MAX`）。
/// `crate::scan_model::SCAN_KERNEL_ARG_LIMIT` と同じ判断根拠。
pub const REDUCE_KERNEL_ARG_LIMIT: usize = u32::MAX as usize;

/// `reduce.metal::reduce_sum_all_chunk_f32`／`reduce_sum_all_finalize_f32`
/// の 2 段構成を逐語再現する: `x` を [`REDUCE_SUM_CHUNK`] 単位のチャンク
/// へ分割し、各チャンク内を `0.0` から [`crate::soft_f64::add_f64_bits`]
/// で逐次加算（`narrow` しない・`f64` bit のまま保持）したうえで、
/// チャンク部分和をチャンク番号順に `0.0` から逐次加算し、最後に 1 回
/// だけ [`crate::soft_f64::narrow_f64_bits`] で `f32` へ downcast する。
///
/// `fandhe_ai_backend_cpu::reduction::sum(a, None)`（内部の `sum_slice_f64`）
/// と bit 完全一致する契約（NaN のみクラス一致。本モジュール下部の
/// 単体テストで検証）。
pub fn sum_all_soft_f64(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut outer_acc: u64 = 0; // widen_f32_bits(0.0) == 0 bit（+0.0）。
    for chunk in x.chunks(REDUCE_SUM_CHUNK) {
        let mut chunk_acc: u64 = 0;
        for &v in chunk {
            chunk_acc = add_f64_bits(chunk_acc, widen_f32_bits(v.to_bits()));
        }
        outer_acc = add_f64_bits(outer_acc, chunk_acc);
    }
    f32::from_bits(narrow_f64_bits(outer_acc))
}

/// `reduce.metal::reduce_sum_axis_f32` の 1 lane 分のループ本体の逐語
/// モデル（`0.0` から昇順に [`crate::soft_f64::add_f64_bits`] で逐次
/// 加算し最後に 1 回 [`crate::soft_f64::narrow_f64_bits`]）。
/// [`crate::soft_f64::sequential_sum_f32_bits`] と数学的に同値だが、
/// `reduce.metal` のループ構造との 1 対 1 対応を明示するために独立に
/// 定義する（`crate::scan_model::cumsum_lane_soft_f64` と同じ設計
/// 判断）。
pub fn sum_axis_lane_soft_f64(xs: &[f32]) -> f32 {
    f32::from_bits(sequential_sum_f32_bits(xs.iter().map(|x| x.to_bits())))
}

/// `x`（`shape` 形状・`dim` 軸で走査）全体へ [`sum_axis_lane_soft_f64`]
/// を lane（`outer × inner` 個）ごとに適用する（`fandhe_ai_backend_cpu::
/// reduction::axis_reduce_sum` と同一の `outer`／`axis_len`／`inner`
/// 分解。テスト専用——本体経路は `crate::reduce::MetalReduce::
/// run_sum_axis_f32` が GPU 側で直接計算するため、本関数は突合用の
/// Rust 側参照実装としてのみ使う。`crate::scan_model::scan_over_shape`
/// と同型）。
#[cfg(test)]
pub fn sum_axis_soft_f64(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let lane: Vec<f32> = (0..axis_len)
                .map(|a| x[(o * axis_len + a) * inner + i])
                .collect();
            out[o * inner + i] = sum_axis_lane_soft_f64(&lane);
        }
    }
    out
}

/// [`plan_reduce_all`]／[`plan_reduce_axis`] の失敗理由。
/// `crate::scan_model::ScanPrepareError` と同型のホスト側純関数エラー
/// （`objc2` 非依存。Linux でも単体テスト可能）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducePrepareError {
    /// `numel`／`num_chunks`（全要素）または `lanes`／`axis_len`／
    /// `inner`（単一軸）のいずれか（`what`）がカーネル `uint` 引数の
    /// 範囲（[`REDUCE_KERNEL_ARG_LIMIT`]）を超えた、または中間積が
    /// `usize` をオーバーフローした（`value == usize::MAX` で表す）。
    /// `ops.rs::map_reduce_prepare_error`（イシュー #1896 で追加済み）
    /// は本 variant を `BackendError::Unsupported` へ写像し
    /// ホストフォールバックへ委ねる（`crate::scan_model::
    /// ScanPrepareError` と同じ判断。`Var::sum` 自体はホスト
    /// フォールバックを持たないため、呼び出し元へそのまま伝播する。
    /// `docs/backend-metal-reduce-sum-design.md` §10）。
    SizeLimitExceeded {
        what: &'static str,
        value: usize,
        limit: usize,
    },
}

impl std::fmt::Display for ReducePrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReducePrepareError::SizeLimitExceeded { what, value, limit } => write!(
                f,
                "sum reduction size limit exceeded: {what}={value} exceeds limit={limit}"
            ),
        }
    }
}

/// 全要素縮約（`dim=None`）の起動計画。`numel`（要素数）・
/// `num_chunks`（`numel.div_ceil(REDUCE_SUM_CHUNK)`）を保持する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceAllPlan {
    pub numel: usize,
    pub num_chunks: usize,
}

/// `numel` から [`ReduceAllPlan`] を構築し、カーネル `uint` 引数
/// （`numel`／`num_chunks`）が [`REDUCE_KERNEL_ARG_LIMIT`] に収まる
/// ことを検証する。`crate::reduce::MetalReduce::run_sum_all_f32` が
/// ディスパッチ前に呼ぶ（呼び出し元の検査結果を信頼しない二重検査
/// 方針・`crate::scan_model::plan_scan` と同型）。
pub fn plan_reduce_all(numel: usize) -> Result<ReduceAllPlan, ReducePrepareError> {
    if numel > REDUCE_KERNEL_ARG_LIMIT {
        return Err(ReducePrepareError::SizeLimitExceeded {
            what: "numel",
            value: numel,
            limit: REDUCE_KERNEL_ARG_LIMIT,
        });
    }
    let num_chunks = numel.div_ceil(REDUCE_SUM_CHUNK);
    if num_chunks > REDUCE_KERNEL_ARG_LIMIT {
        return Err(ReducePrepareError::SizeLimitExceeded {
            what: "num_chunks",
            value: num_chunks,
            limit: REDUCE_KERNEL_ARG_LIMIT,
        });
    }
    Ok(ReduceAllPlan { numel, num_chunks })
}

/// 単一軸縮約（`dim=Some(axis)`）の起動計画。`crate::sort_model::
/// line_layout` を再利用して `outer`／`axis_len`／`inner` を導出する
/// （`crate::scan_model::plan_scan` と同じ再利用方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceAxisPlan {
    pub outer: usize,
    pub axis_len: usize,
    pub inner: usize,
    pub lanes: usize,
}

/// `shape`／`dim` から [`ReduceAxisPlan`] を構築し、カーネル `uint`
/// 引数（`lanes`／`axis_len`／`inner`）が [`REDUCE_KERNEL_ARG_LIMIT`]
/// に収まることを検証する。`dim >= shape.len()`（軸範囲外）・lane
/// 分解の中間積 `usize` オーバーフローは `crate::sort_model::
/// line_layout` が `None` を返すため panic せず
/// `SizeLimitExceeded { value: usize::MAX, .. }` へ写像する
/// （`crate::scan_model::plan_scan` と同じ防御）。
pub fn plan_reduce_axis(shape: &[usize], dim: usize) -> Result<ReduceAxisPlan, ReducePrepareError> {
    let overflow = |what: &'static str| ReducePrepareError::SizeLimitExceeded {
        what,
        value: usize::MAX,
        limit: REDUCE_KERNEL_ARG_LIMIT,
    };
    let (outer, axis_len, inner) =
        crate::sort_model::line_layout(shape, dim).ok_or(overflow("lane layout"))?;
    let lanes = outer.checked_mul(inner).ok_or(overflow("lanes"))?;
    for (what, value) in [("lanes", lanes), ("axis_len", axis_len), ("inner", inner)] {
        if value > REDUCE_KERNEL_ARG_LIMIT {
            return Err(ReducePrepareError::SizeLimitExceeded {
                what,
                value,
                limit: REDUCE_KERNEL_ARG_LIMIT,
            });
        }
    }
    Ok(ReduceAxisPlan {
        outer,
        axis_len,
        inner,
        lanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::reduction::sum as cpu_sum;
    use fandhe_ai_tensor_core::Tensor;

    /// xorshift64* 決定的疑似乱数（`crate::soft_f64::tests::Rng` と
    /// 同型の独立実装。テスト専用のため重複を許容する）。
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f32(&mut self) -> f32 {
            // [-1024, 1024) 程度の小数混在値（bit 一致テストとしては
            // 値域自体は本質的でない。指数域を分散させるため上位ビット
            // から符号・指数相当を取り出す）。
            let bits = self.next_u64();
            let scaled = ((bits >> 32) as i32 as f64) / (i32::MAX as f64) * 1024.0;
            scaled as f32
        }
    }

    fn assert_bits_match(actual: f32, expected: f32) {
        assert!(
            crate::soft_f64::f32_bits_match(actual, expected),
            "bit mismatch: actual={actual:?} (0x{:08x}), expected={expected:?} (0x{:08x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }

    /// 全要素縮約: [`sum_all_soft_f64`] が `fandhe_ai_backend_cpu::
    /// reduction::sum(a, None)` と各種サイズで bit 完全一致する
    /// （チャンク境界前後・チャンク複数個を横断する規模を含む）。
    #[test]
    fn sum_all_matches_cpu_reference_across_sizes() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for &n in &[0usize, 1, 2, 4095, 4096, 4097, 8192, 3 * 4096 + 1] {
            let data: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let model = sum_all_soft_f64(&data);
            let tensor = Tensor::new(data.clone(), &[n]).unwrap();
            let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
            assert_bits_match(model, expected);
        }
    }

    /// チャンク境界感度（複数チャンク・端数チャンクを跨ぐ規模）:
    /// `CHUNK`（4096）境界を跨ぐ相殺列を含む大きめの配列で、モデルが
    /// CPU 参照実装（チャンク内逐次 → チャンク間逐次の 2 段結合）と
    /// bit 一致することを確認する。**注意**: 本テストの相殺値
    /// （`2^24`／`2^20`）は f64 の仮数精度（52 bit）内に十分収まる
    /// ため、実際には平坦逐次和でも同じ結果になり（binary64 では
    /// 結合順序による丸め差が生じない）、`REDUCE_SUM_CHUNK` の値の
    /// 取り違えやチャンク数の計算誤りは検出するが、2 段結合という
    /// 構造そのもの（結合順序）は検出しない。結合順序依存の丸め差は
    /// 下の `sum_all_flat_sequential_diverges_from_chunked_at_large_n`
    /// が別途固定する（codex-review 指摘。イシュー #1895 PR #1925）。
    #[test]
    fn sum_all_chunk_boundary_cancelling_sequence_matches_cpu() {
        // 2^24 は f32 の仮数精度限界（`2^24 + 1` は `2^24` へ丸められる）
        // で、相殺列の順序依存性を顕在化させる典型値。
        let mut data = vec![0.0f32; 4097 * 3];
        data[4094] = 2f32.powi(24);
        data[4095] = 1.0;
        data[4096] = -2f32.powi(24); // チャンク境界（4096）を跨ぐ。
        data[4097] = -1.0;
        data[8192] = 2f32.powi(20);
        data[8193] = -2f32.powi(20);
        let model = sum_all_soft_f64(&data);
        let tensor = Tensor::new(data, &[4097 * 3]).unwrap();
        let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
        assert_bits_match(model, expected);
    }

    /// チャンク境界の結合順序依存性（binary64 の丸め差を実際に顕在化
    /// させる入力）: `2^53`（f64 仮数精度限界。`2^53 + 1` は最近接
    /// 偶数丸めで `2^53` へ丸められる）をチャンク 0 の末尾（index
    /// `CHUNK - 1`）に、`1.0` と `-2^53` をチャンク 1 の先頭 2 要素
    /// （index `CHUNK`・`CHUNK + 1`）に配置する。
    ///
    /// 平坦逐次和（単一アキュムレータで先頭から通しで加算）は
    /// `(2^53 + 1) + (-2^53)` の順で評価され、`2^53 + 1` が `2^53`
    /// へ丸められた結果 `-2^53` と相殺して `0.0` になる。一方
    /// [`sum_all_soft_f64`] の 2 段結合（チャンク 1 を `0.0` から
    /// 独立に `1.0 + (-2^53)` として先に評価してから `2^53` へ加算）
    /// は `1.0 + (-2^53) = -(2^53 - 1)`（`2^53 - 1` は 53 bit 仮数で
    /// 丸めなしに正確に表現できる）を経て `2^53 + (-(2^53 - 1)) =
    /// 1.0` になる。両者は `0.0` と `1.0` という異なる結果になり
    /// （Python `float`〈binary64〉で事前計算し固定した値。本テストは
    /// この相違自体を主張とする）、CPU 参照実装の 2 段結合構造との
    /// bit 一致とあわせて、モデルが単純な平坦逐次和ではなくチャンク
    /// 境界での結合順序を正しく再現していることを検証する
    /// （codex-review 指摘・イシュー #1895 PR #1925）。
    #[test]
    fn sum_all_flat_sequential_diverges_from_chunked_at_large_n() {
        let mut data = vec![0.0f32; REDUCE_SUM_CHUNK + 2];
        data[REDUCE_SUM_CHUNK - 1] = 2f32.powi(53); // チャンク 0 末尾。
        data[REDUCE_SUM_CHUNK] = 1.0; // チャンク 1 先頭。
        data[REDUCE_SUM_CHUNK + 1] = -(2f32.powi(53)); // チャンク 1 2 番目。

        let flat = crate::soft_f64::sequential_sum_f32(&data);
        let chunked = sum_all_soft_f64(&data);

        // 事前に Python（binary64）で計算し固定した値: 平坦逐次和は
        // `0.0`・2 段チャンク結合は `1.0` になり、実際に異なる。
        assert_eq!(flat.to_bits(), 0.0f32.to_bits(), "flat sequential sum");
        assert_eq!(chunked.to_bits(), 1.0f32.to_bits(), "chunked sum");
        assert_ne!(
            flat.to_bits(),
            chunked.to_bits(),
            "平坦逐次和と 2 段チャンク結合が一致してしまっている（構造を検証できていない）"
        );

        // モデルは CPU 参照実装（同じ 2 段結合構造）と bit 一致する。
        let tensor = Tensor::new(data, &[REDUCE_SUM_CHUNK + 2]).unwrap();
        let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
        assert_bits_match(chunked, expected);
    }

    /// `-0.0` のみの入力は `+0.0` になる（fold の初期値 `+0.0` により
    /// 自然に成立。CPU 参照実装との一致で確認）。
    #[test]
    fn sum_all_negative_zero_only_yields_positive_zero() {
        let data = vec![-0.0f32; 10];
        let model = sum_all_soft_f64(&data);
        assert_eq!(model.to_bits(), 0.0f32.to_bits());
        let tensor = Tensor::new(data, &[10]).unwrap();
        let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
        assert_bits_match(model, expected);
    }

    /// NaN 伝播（1 要素 NaN・±inf 混在で `inf + -inf = NaN`）はクラス
    /// 一致で比較する（payload はハードウェア依存のため）。
    #[test]
    fn sum_all_nan_and_inf_propagation() {
        let mut data = vec![1.0f32, 2.0, f32::NAN, 3.0];
        let model = sum_all_soft_f64(&data);
        assert!(model.is_nan());
        let tensor = Tensor::new(data.clone(), &[data.len()]).unwrap();
        let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
        assert_bits_match(model, expected);

        data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let model = sum_all_soft_f64(&data);
        assert!(model.is_nan());
        let tensor = Tensor::new(data.clone(), &[data.len()]).unwrap();
        let expected = cpu_sum(&tensor, None).unwrap().as_slice().unwrap()[0];
        assert_bits_match(model, expected);
    }

    /// `numel <= CHUNK` では [`sum_all_soft_f64`] が
    /// [`crate::soft_f64::sequential_sum_f32`]（単純な平坦逐次和）と
    /// 一致する（構造の健全性: チャンクが 1 個のみの場合は
    /// チャンク間結合が恒等写像になるはず）。
    #[test]
    fn sum_all_matches_sequential_when_single_chunk() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let data: Vec<f32> = (0..REDUCE_SUM_CHUNK).map(|_| rng.f32()).collect();
        let model = sum_all_soft_f64(&data);
        let seq = crate::soft_f64::sequential_sum_f32(&data);
        assert_bits_match(model, seq);
    }

    /// 単一軸縮約: 複数 rank・複数 `dim` 位置で CPU 参照実装
    /// （`axis_reduce_sum`）と bit 完全一致する。
    #[test]
    fn sum_axis_matches_cpu_reference_across_shapes_and_dims() {
        let mut rng = Rng(0x0ff1_ce0f_f1ce_0ff1);
        let cases: &[(&[usize], usize)] = &[
            (&[5], 0),
            (&[3, 4], 0),
            (&[3, 4], 1),
            (&[2, 3, 5], 0),
            (&[2, 3, 5], 1),
            (&[2, 3, 5], 2),
            (&[2, 3, 4, 2], 2),
            (&[1, 4097], 1),
            (&[4097, 1], 0),
        ];
        for &(shape, dim) in cases {
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|_| rng.f32()).collect();
            let model = sum_axis_soft_f64(&data, shape, dim);
            let tensor = Tensor::new(data, shape).unwrap();
            let expected_tensor = cpu_sum(&tensor, Some(dim)).unwrap();
            let expected = expected_tensor.as_slice().unwrap();
            assert_eq!(model.len(), expected.len());
            for (a, e) in model.iter().zip(expected.iter()) {
                assert_bits_match(*a, *e);
            }
        }
    }

    /// 単一軸縮約の `axis_len == 0`（空出力ではなく空縮約）は
    /// 各出力要素が `0.0` になる（CPU 参照実装は `EmptyReduction`
    /// を返さない——`sum` は単位元 `0.0` を持つ。モジュール doc
    /// 「空縮約の意味論」参照）。`outer`／`inner` に 0 を含む形状
    /// （空出力）は出力ベクタが空になることも併せて確認する。
    #[test]
    fn sum_axis_empty_axis_len_yields_zero_outputs() {
        // axis_len=0・outer=1・inner=1 → 1 出力要素、値 0.0。
        let model = sum_axis_soft_f64(&[], &[0, 3], 0);
        assert_eq!(model, vec![0.0f32; 3]);

        // outer=0（空出力）。
        let model_empty_outer = sum_axis_soft_f64(&[], &[0, 3, 5], 1);
        assert!(model_empty_outer.is_empty());
    }

    /// [`plan_reduce_all`]: 通常形状。
    #[test]
    fn plan_reduce_all_boundaries() {
        assert_eq!(
            plan_reduce_all(0).unwrap(),
            ReduceAllPlan {
                numel: 0,
                num_chunks: 0
            }
        );
        assert_eq!(
            plan_reduce_all(REDUCE_SUM_CHUNK).unwrap(),
            ReduceAllPlan {
                numel: REDUCE_SUM_CHUNK,
                num_chunks: 1
            }
        );
        assert_eq!(
            plan_reduce_all(REDUCE_SUM_CHUNK + 1).unwrap(),
            ReduceAllPlan {
                numel: REDUCE_SUM_CHUNK + 1,
                num_chunks: 2
            }
        );
    }

    /// [`plan_reduce_all`]: `u32::MAX` 境界超過（`numel`）。
    /// `REDUCE_KERNEL_ARG_LIMIT + 1` は 32bit `usize` では計算自体が
    /// オーバーフローするため `scan_model::plan_scan_u32_limit_boundary`
    /// と同じく 64bit 限定にする。
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn plan_reduce_all_u32_limit_boundary() {
        let over = REDUCE_KERNEL_ARG_LIMIT + 1;
        assert_eq!(
            plan_reduce_all(over),
            Err(ReducePrepareError::SizeLimitExceeded {
                what: "numel",
                value: over,
                limit: REDUCE_KERNEL_ARG_LIMIT,
            })
        );
    }

    /// [`plan_reduce_axis`]: 通常形状・`dim` 範囲外・`usize`
    /// オーバーフロー相当の防御（`line_layout` の `None` 経由）。
    #[test]
    fn plan_reduce_axis_normal_and_out_of_range() {
        let plan = plan_reduce_axis(&[2, 3, 4], 1).unwrap();
        assert_eq!(
            plan,
            ReduceAxisPlan {
                outer: 2,
                axis_len: 3,
                inner: 4,
                lanes: 8,
            }
        );

        let err = plan_reduce_axis(&[2, 3], 2).unwrap_err();
        assert!(matches!(
            err,
            ReducePrepareError::SizeLimitExceeded {
                what: "lane layout",
                value: usize::MAX,
                ..
            }
        ));
    }

    /// [`plan_reduce_axis`]: `u32::MAX` 境界（`crate::scan_model::
    /// plan_scan_u32_limit_boundary` と同じ形状パターンで `axis_len`／
    /// `lanes`／`inner` 単独超過を確認する。64bit 限定の理由は
    /// [`plan_reduce_all_u32_limit_boundary`] と同じ）。
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn plan_reduce_axis_u32_limit_boundary() {
        let over = REDUCE_KERNEL_ARG_LIMIT + 1;
        // axis_len 単独超過（outer=1, inner=1）。
        assert!(matches!(
            plan_reduce_axis(&[1, over], 1),
            Err(ReducePrepareError::SizeLimitExceeded {
                what: "axis_len",
                ..
            })
        ));
        // inner 超過（outer=1 のため lanes（= inner）も同時に超過し、
        // 検査順序どおり lanes を先に報告する。`crate::scan_model::
        // plan_scan_u32_limit_boundary` の同型ケースと同じ挙動）。
        assert!(matches!(
            plan_reduce_axis(&[1, 1, over], 1),
            Err(ReducePrepareError::SizeLimitExceeded { what: "lanes", .. })
        ));
        // lanes 超過（outer 側）。
        assert!(matches!(
            plan_reduce_axis(&[over, 2], 1),
            Err(ReducePrepareError::SizeLimitExceeded { what: "lanes", .. })
        ));
    }

    // ---- argmax／argmin（イシュー #1951）----

    use fandhe_ai_backend_cpu::reduction::{argmax as cpu_argmax, argmin as cpu_argmin};

    fn cpu_argext(data: &[f32], shape: &[usize], dim: Option<usize>, kind: ArgExtKind) -> Vec<i32> {
        let tensor = Tensor::new(data.to_vec(), shape).unwrap();
        let out = match kind {
            ArgExtKind::Max => cpu_argmax(&tensor, dim).unwrap(),
            ArgExtKind::Min => cpu_argmin(&tensor, dim).unwrap(),
        };
        out.as_slice().unwrap().to_vec()
    }

    /// [`arg_key`]: 非 NaN の代表値集合の全ペアで、キーの大小が
    /// `f32::partial_cmp`（NaN を除く）と一致することを検証する
    /// （モジュール doc「§2.3 比較は整数ドメインで行う」の裏付け）。
    #[test]
    fn arg_key_total_order_matches_partial_cmp() {
        let values: &[f32] = &[
            f32::NEG_INFINITY,
            -1e30,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            1e30,
            f32::INFINITY,
            f32::MIN_POSITIVE, // 非正規化数近傍。
            -f32::MIN_POSITIVE,
        ];
        for &a in values {
            for &b in values {
                let ka = arg_key(a.to_bits()).unwrap();
                let kb = arg_key(b.to_bits()).unwrap();
                let expected = a.partial_cmp(&b).unwrap();
                assert_eq!(
                    ka.cmp(&kb),
                    expected,
                    "arg_key order mismatch: a={a:?} b={b:?}"
                );
            }
        }
        // ±0 は同値キー。
        assert_eq!(arg_key(0.0f32.to_bits()), arg_key((-0.0f32).to_bits()));
        // NaN は None（除外）。
        assert_eq!(arg_key(f32::NAN.to_bits()), None);
    }

    /// [`argext_all_chunked`]: CPU `argmax`／`argmin`（`dim=None`）と
    /// 各種サイズ（チャンク境界前後を含む）で完全一致する。
    #[test]
    fn argext_all_matches_cpu_reference_across_sizes() {
        let mut rng = Rng(0x2222_4444_6666_8888);
        for &n in &[1usize, 2, 4095, 4096, 4097, 2 * 4096 + 1, 3 * 4096 + 17] {
            let data: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            for kind in [ArgExtKind::Max, ArgExtKind::Min] {
                let model = argext_all_chunked(&data, kind);
                let expected = cpu_argext(&data, &[n], None, kind);
                assert_eq!(model as i32, expected[0], "n={n} kind={kind:?}");
            }
        }
    }

    /// [`argext_all_chunked`] と平坦走査（[`argext_all_flat`]）が全域で
    /// 一致する（モジュール doc §2.2「等価性の証明」の機械的裏付け）。
    #[test]
    fn argext_all_chunked_matches_flat_scan() {
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for &n in &[1usize, 4095, 4096, 4097, 8192, 3 * 4096 + 17] {
            let data: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            for kind in [ArgExtKind::Max, ArgExtKind::Min] {
                assert_eq!(
                    argext_all_chunked(&data, kind),
                    argext_all_flat(&data, kind),
                    "n={n} kind={kind:?}"
                );
            }
        }
    }

    /// 全要素同値・チャンク境界をまたぐ同値極値は最小添字を返す
    /// （先勝ちタイ契約）。
    #[test]
    fn argext_all_tie_returns_first_index() {
        let data = vec![1.0f32; 4097 * 2];
        assert_eq!(argext_all_chunked(&data, ArgExtKind::Max), 0);
        assert_eq!(argext_all_chunked(&data, ArgExtKind::Min), 0);

        // チャンク境界（4096）をまたぐ同値極値: index 4095 と 4096 が
        // 同じ最大値・4095 が先勝ち。
        let mut data2 = vec![0.0f32; 4098];
        data2[4095] = 5.0;
        data2[4096] = 5.0;
        assert_eq!(argext_all_chunked(&data2, ArgExtKind::Max), 4095);
    }

    /// NaN 混入時の挙動: NaN はスキップし有効な極値の添字を返す。
    /// 全要素 NaN の場合は添字 0（`fandhe_ai_tensor_core::BackendOps::
    /// argmax` doc の走査契約 1〜3 参照）。
    #[test]
    fn argext_all_nan_handling() {
        let data = vec![f32::NAN, 3.0, f32::NAN, 1.0, f32::NAN];
        assert_eq!(argext_all_chunked(&data, ArgExtKind::Max), 1);
        assert_eq!(argext_all_chunked(&data, ArgExtKind::Min), 3);

        // 全要素 NaN（先頭・中間・末尾の NaN を含むチャンク）。
        let all_nan = vec![f32::NAN; 4097];
        assert_eq!(argext_all_chunked(&all_nan, ArgExtKind::Max), 0);
        assert_eq!(argext_all_chunked(&all_nan, ArgExtKind::Min), 0);

        // CPU 参照実装と一致することも確認する。
        for kind in [ArgExtKind::Max, ArgExtKind::Min] {
            let expected = cpu_argext(&data, &[data.len()], None, kind);
            assert_eq!(argext_all_chunked(&data, kind) as i32, expected[0]);
        }
    }

    /// [`argext_axis_lane`]: CPU `argmax`／`argmin`（`dim=Some`）と
    /// 複数 shape・複数 dim・タイ・NaN で一致する。
    #[test]
    fn argext_axis_matches_cpu_reference_across_shapes_and_dims() {
        let mut rng = Rng(0xabab_cdcd_efef_0101);
        let cases: &[(&[usize], usize)] = &[
            (&[5], 0),
            (&[3, 4], 0),
            (&[3, 4], 1),
            (&[2, 3, 5], 0),
            (&[2, 3, 5], 1),
            (&[2, 3, 5], 2),
            (&[2, 3, 4, 2], 2),
        ];
        for &(shape, dim) in cases {
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|_| rng.f32()).collect();
            for kind in [ArgExtKind::Max, ArgExtKind::Min] {
                let expected = cpu_argext(&data, shape, Some(dim), kind);
                let outer: usize = shape[..dim].iter().product();
                let axis_len = shape[dim];
                let inner: usize = shape[dim + 1..].iter().product();
                for o in 0..outer {
                    for i in 0..inner {
                        let lane: Vec<f32> = (0..axis_len)
                            .map(|a| data[(o * axis_len + a) * inner + i])
                            .collect();
                        let model = argext_axis_lane(&lane, kind);
                        assert_eq!(
                            model as i32,
                            expected[o * inner + i],
                            "shape={shape:?} dim={dim} kind={kind:?} o={o} i={i}"
                        );
                    }
                }
            }
        }
    }

    /// [`argext_axis_lane`]: タイ（先勝ち）・NaN 混入時に添字 0。
    #[test]
    fn argext_axis_lane_tie_and_nan() {
        assert_eq!(
            argext_axis_lane(&[2.0, 2.0, 2.0], ArgExtKind::Max),
            0,
            "全同値は先勝ちで添字 0"
        );
        assert_eq!(
            argext_axis_lane(&[f32::NAN, f32::NAN], ArgExtKind::Max),
            0,
            "全 NaN は添字 0"
        );
        assert_eq!(argext_axis_lane(&[f32::NAN, 3.0, 1.0], ArgExtKind::Max), 1);
        assert_eq!(argext_axis_lane(&[f32::NAN, 3.0, 1.0], ArgExtKind::Min), 2);
    }

    /// [`plan_argext_all`]: `numel <= i32::MAX` は既存 `plan_reduce_all`
    /// と同じ結果、`i32::MAX` 超過は新規の `i32` 上限違反を返す。
    #[test]
    fn plan_argext_all_i32_limit() {
        assert!(plan_argext_all(REDUCE_SUM_CHUNK).is_ok());
        let over = i32::MAX as usize + 1;
        assert!(matches!(
            plan_argext_all(over),
            Err(ReducePrepareError::SizeLimitExceeded {
                what: "numel (i32 index range)",
                value,
                ..
            }) if value == over
        ));
    }

    /// [`plan_argext_axis`]: `axis_len <= i32::MAX` は既存
    /// `plan_reduce_axis` と同じ結果、超過は新規の `i32` 上限違反を
    /// 返す。
    #[test]
    fn plan_argext_axis_i32_limit() {
        assert_eq!(
            plan_argext_axis(&[2, 3, 4], 1).unwrap(),
            ReduceAxisPlan {
                outer: 2,
                axis_len: 3,
                inner: 4,
                lanes: 8,
            }
        );
        let over = i32::MAX as usize + 1;
        assert!(matches!(
            plan_argext_axis(&[1, over], 1),
            Err(ReducePrepareError::SizeLimitExceeded {
                what: "axis_len (i32 index range)",
                ..
            })
        ));
    }
}
