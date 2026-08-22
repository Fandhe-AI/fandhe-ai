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

use fandhe_ai_tensor_core::{ShapeError, Tensor, reduce_out_shape};
use rayon::prelude::*;

/// 全縮約（`dim=None`）の決定的チャンク結合に用いる固定チャンクサイズ。
///
/// チャンク境界を跨ぐ演算順序の違いが bit 差を生まないよう、値は実装内で
/// 固定する（呼び出し側からの変更点を持たない。ガードレール閾値ではないが、
/// 数値一致回帰テストの前提となるため安易に変更しない）。
const CHUNK: usize = 4096;

/// reduction カーネル固有のエラー。`BackendError`（TASK-1.9 で導入予定）への
/// ラップは `BackendOps` 実装時に行う想定であり、本モジュールでは行わない。
#[non_exhaustive]
#[derive(Debug)]
pub enum ReduceError {
    /// shape・軸検査の失敗（`fandhe_ai_tensor_core::reduce_out_shape` に委譲する検査の
    /// 失敗をそのまま透過する）。
    Shape(ShapeError),
    /// 縮約対象の要素数が 0（`max`・`mean` は単位元を持たないためエラーとする。
    /// `op` は失敗した演算名 `"max"`/`"mean"`）。
    EmptyReduction { op: &'static str },
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceError::Shape(err) => write!(f, "reduction shape error: {err}"),
            ReduceError::EmptyReduction { op } => {
                write!(f, "cannot compute {op} of an empty reduction")
            }
        }
    }
}

impl std::error::Error for ReduceError {}

/// 線形インデックス空間（`0..outer*inner`）を行優先で `dims` の多次元
/// インデックスへ展開する。`row_major_strides`（tensor-core）と対の関係にある
/// 展開処理で、`axis_reduce`（出力側の外側・内側インデックス復元）と
/// `gather_elements`（非 contiguous 全縮約時の走査順再現）の両方から使う。
fn unravel(mut idx: usize, dims: &[usize]) -> Vec<usize> {
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
/// 従って `sum` を計算する。
fn sum_slice(data: &[f32]) -> f32 {
    data.par_chunks(CHUNK)
        .map(|chunk| chunk.iter().copied().fold(0.0f32, |acc, v| acc + v))
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(0.0f32, |acc, v| acc + v)
}

/// `data` を [`CHUNK`] 単位に分割し、決定性契約（モジュール doc 参照）に
/// 従って `max` を計算する。`data` が空の場合は `None` を返す（呼び出し元が
/// [`ReduceError::EmptyReduction`] に変換する）。
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
    let result = data
        .par_chunks(CHUNK)
        .map(|chunk| chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect::<Vec<f32>>()
        .into_iter()
        .fold(f32::NEG_INFINITY, f32::max);
    Some(result)
}

/// `outer`/`inner`（軸を除いた前後の次元積）を `checked_mul` で計算する。
/// アロケーション前の要素数計算にオーバーフローが混入すると過小確保・
/// 境界不整合を招く（OWASP A03 相当。`.claude/rules/security.md`）ため、
/// `tensor-core::checked_numel` と同方針でオーバーフローを検出する。
fn checked_product(dims: &[usize]) -> Result<usize, ReduceError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ReduceError::Shape(ShapeError::ElementCountOverflow))
}

/// 軸指定 reduction（`dim=Some(axis)`）の出力要素ごとの畳み込みを行う共通
/// 駆動関数。`axis` は呼び出し元（`sum`/`max`/`mean`）が `reduce_out_shape`
/// で事前検査済みであることを前提とする（本関数自体は範囲検査を行わない）。
///
/// 出力要素（`outer × inner` 個）側のみ rayon で並列化し、各要素内では
/// 縮約軸を `0..axis_len` の昇順で `op` により逐次累積する（決定性契約は
/// モジュール doc 参照）。`Range<usize>` は `IndexedParallelIterator` であり
/// `.collect()` が出力順を保持するため、`flat` 昇順の出力ベクタが得られる。
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

    (0..total_out)
        .into_par_iter()
        .map(|flat| {
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
        })
        .collect()
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
            axis_reduce(a, axis, 0.0, |acc, v| acc + v)
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
            let sums = axis_reduce(a, axis, 0.0, |acc, v| acc + v);
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
        assert_eq!(out.shape(), &[]);
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
        let naive = data
            .chunks(CHUNK)
            .map(|chunk| chunk.iter().fold(0.0f32, |acc, &v| acc + v))
            .fold(0.0f32, |acc, v| acc + v);
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
}
