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
//! ホスト側参照実装・`plan_reduce_all`／`plan_reduce_axis`（`ops.rs`
//! からの呼び出しを見越した事前検証ヘルパー）を提供するのみであり、
//! `MetalBackendOps::sum` への結線（`context_cache::cached_reduce`・
//! `ops.rs` からの呼び出し）は行わない（#1896 のスコープ）。

use crate::soft_f64::{add_f64_bits, narrow_f64_bits, sequential_sum_f32_bits, widen_f32_bits};

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
    /// `ops.rs::map_reduce_prepare_error`（#1896 で追加予定）は本
    /// variant を `BackendError::Unsupported` へ写像しホスト
    /// フォールバックへ委ねる設計を想定する（`crate::scan_model::
    /// ScanPrepareError` と同じ判断）。
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
}
