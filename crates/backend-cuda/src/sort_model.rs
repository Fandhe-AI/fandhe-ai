//! `sort`／`topk`（イシュー #1741）の GPU 非依存な純関数群（キー設計・
//! ライン分解・起動前検証・ホストモデル）。
//!
//! [`crate::kernels_sort`]（NVRTC カーネルソース）は本モジュールの
//! [`value_key`]／[`composite_key`]／[`line_layout`] と同一のアルゴリズム
//! を GPU 側で逐語的に再実装したものであり、本モジュールの
//! `#[cfg(test)]` ホストモデル（[`build_keys_host`]／
//! [`bitonic_step_u64_host`]／[`finalize_host`]／
//! [`sort_lines_host_model`]）が両者の一致を検証する足場になる
//! （`unique_model.rs` と同じ「意図的複製・ホストモデルによる検証」
//! 方針）。
//!
//! # 鍵設計（`fandhe_ai_tensor_core::BackendOps::sort` doc の順序契約
//! 1〜4 を GPU 非安定ソート〈ビットニックソート〉でも機械的に満たす
//! ための構成）
//!
//! 各要素へ 64bit 合成キー `key = (hi << 32) | lo` を与える:
//! - `lo`: ライン内の元添字（`u32`）。ライン内で一意なので、非安定
//!   ソートでも `key` 自体がライン内で一意になり、同値（ties）が
//!   あっても最終順序が構造的に一意に定まる（非決定性の余地がない）。
//! - `hi`: [`value_key`] で得た「NaN を最大・±0 を同値化」した
//!   totalOrder 風キー（[`crate::unique_model::total_order_key`] を
//!   出発点に契約用へ正規化したもの。本モジュール独自実装——クレート
//!   境界を跨いで `pub(crate)` 関数を共有できないため意図的に複製
//!   する必要はないが、正規化の要件〈NaN 最大化・±0 同値化〉が
//!   `unique` 側の totalOrder とは異なるため別関数として定義する）。
//!   `descending` のときは `hi` のみを反転する（`lo` は反転しない）。
//!   これにより「値降順・同値内は添字昇順」が成立する——単純な
//!   昇順ソート結果の `reverse()` は同値の添字順を反転させてしまう
//!   （CPU 参照実装 `backend-cpu::sort_topk` モジュール doc と同じ罠の
//!   回避）。
//!
//! 昇順ビットニックソート後の `key & 0xFFFF_FFFF`（＝`lo`）がそのまま
//! `index` になる。`values` は鍵から復元せず、`index` で元入力を
//! gather する（NaN の payload・±0 の符号は正規化で失われるため。
//! `fandhe_ai_tensor_core::BackendOps::sort` doc の
//! `values == gather(input, dim, index)` 恒等式と同一）。
//!
//! # ライン分解（rank 上限なし）
//!
//! `input`（行優先・contiguous 済み）を `dim` 軸で `outer * dim_size *
//! inner` へ分解する（[`line_layout`]）。ライン `l`（`l < outer*inner`）
//! の先頭 flat 位置は `(l / inner) * dim_size * inner + (l % inner)`、
//! ライン内ストライドは `inner`。出力（`dim` 軸のみ `out_len` に置換
//! した shape）も同じ式の `dim_size` を `out_len` へ置換した形で位置が
//! 決まる。rank ごとの固定長配列・座標展開を使わないため rank に
//! 上限を設けない設計（`gather_scatter` 系のような `GS_MAX_RANK` は
//! 不要）。
//!
//! # ビットニックソートのパディング
//!
//! `padded = dim_size.next_power_of_two()`。パディング要素には
//! `u64::MAX` を書く（実キーは `hi <= 0xFFFF_FFFF`・`lo < dim_size <=
//! i32::MAX` のため、`u64::MAX` は厳密に全実キーより大きい——各ラインの
//! 末尾へ自然に集まる）。

/// `f32` を「NaN は最大・NaN 同士は同値・±0 は同値」に正規化した
/// `u32` の totalOrder 風キーへ変換する（モジュール doc「鍵設計」
/// 参照）。`crate::unique_model::total_order_key` と異なり、NaN は
/// 符号・payload に関わらず常に `0xFFFF_FFFF`（最大）、`-0.0` は
/// `+0.0` と同一キーになる（`sort`／`topk` の順序契約 2・3 の要求）。
#[cfg(test)]
pub fn value_key(v: f32) -> u32 {
    if v.is_nan() {
        return 0xFFFF_FFFF;
    }
    // `v == 0.0` は `-0.0 == 0.0` が `true` を返す IEEE 754 の等価性を
    // 利用して ±0 を `+0.0` の bit パターンへ同値化する。
    let normalized = if v == 0.0 { 0.0f32 } else { v };
    let bits = normalized.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// 64bit 合成キー（モジュール doc「鍵設計」参照）。`idx` はライン内の
/// 元添字（`dim_size` 未満であることを呼び出し元が保証する）。
#[cfg(test)]
pub fn composite_key(v: f32, idx: u32, descending: bool) -> u64 {
    let vk = value_key(v);
    let hi = if descending { !vk } else { vk };
    ((hi as u64) << 32) | (idx as u64)
}

/// パディング要素に書く値（モジュール doc「ビットニックソートの
/// パディング」参照）。
#[cfg(test)]
pub const PADDING_KEY: u64 = u64::MAX;

/// `dim` 軸に沿ったライン分解（モジュール doc「ライン分解」参照）。
/// `shape` は空要素を含まないこと（呼び出し元が事前に検査する契約。
/// `sort.rs::CudaSort::run_sort_f32` は空 shape を GPU 起動なしで
/// 早期処理する）。
pub fn line_layout(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

/// [`plan_sort`] の失敗理由。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortPrepareError {
    /// `dim_size`（ソート対象軸の長さ）がカーネル引数型の範囲
    /// （`i32::MAX`）を超えた。CPU 参照実装（`backend-cpu::sort_topk`）
    /// の `i32::try_from` 失敗と同じ分類——`fandhe_ai_tensor_core::
    /// ShapeError::IndexRangeOverflow` へ写像される（`ops.rs`）。
    DimSizeTooLarge { dim_size: usize },
    /// 合成キー配列長（`lines * padded`）がバックエンド固有上限
    /// （`i32::MAX`）を超えた。`ops.rs` は本 variant のみを
    /// `BackendError::Unsupported` へ写像しホストフォールバックへ
    /// 委ねる。
    SizeLimitExceeded { total: usize, limit: usize },
}

impl std::fmt::Display for SortPrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortPrepareError::DimSizeTooLarge { dim_size } => {
                write!(
                    f,
                    "sort/topk dim_size too large for kernel argument: {dim_size}"
                )
            }
            SortPrepareError::SizeLimitExceeded { total, limit } => {
                write!(
                    f,
                    "sort/topk size limit exceeded: total={total} exceeds limit={limit}"
                )
            }
        }
    }
}

/// [`line_layout`] の結果と派生量（`padded`・`lines`）をまとめた起動
/// 計画（CUDA／Metal 双方の起動 API が `dispatch_sync`／GPU 起動へ入る
/// 前にホスト側で完結させる検証の結果）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortPlan {
    pub outer: usize,
    pub dim_size: usize,
    pub inner: usize,
    pub lines: usize,
    pub padded: usize,
}

/// `shape`／`dim` から [`SortPlan`] を構築し、カーネル引数・合成キー
/// 配列長がバックエンド固有上限（`i32::MAX`）に収まることを検証する
/// （`unique.rs::CudaUnique::run_unique_f32` の `padded` 検証と同型の
/// 「サイズ上限超過のみ `Unsupported` へ、それ以外は伝播」方針。呼び
/// 出し元は本関数を GPU 起動・バッファ確保の前に呼び、`Result` を
/// 評価してから初めて `unsafe` な起動処理へ入ること）。
pub fn plan_sort(shape: &[usize], dim: usize) -> Result<SortPlan, SortPrepareError> {
    let (outer, dim_size, inner) = line_layout(shape, dim);
    if dim_size > i32::MAX as usize {
        return Err(SortPrepareError::DimSizeTooLarge { dim_size });
    }
    let padded =
        dim_size
            .checked_next_power_of_two()
            .ok_or(SortPrepareError::SizeLimitExceeded {
                total: usize::MAX,
                limit: i32::MAX as usize,
            })?;
    let lines = outer
        .checked_mul(inner)
        .ok_or(SortPrepareError::SizeLimitExceeded {
            total: usize::MAX,
            limit: i32::MAX as usize,
        })?;
    let total_keys = lines
        .checked_mul(padded)
        .ok_or(SortPrepareError::SizeLimitExceeded {
            total: usize::MAX,
            limit: i32::MAX as usize,
        })?;
    if total_keys > i32::MAX as usize {
        return Err(SortPrepareError::SizeLimitExceeded {
            total: total_keys,
            limit: i32::MAX as usize,
        });
    }
    Ok(SortPlan {
        outer,
        dim_size,
        inner,
        lines,
        padded,
    })
}

/// ライン `line`（`< outer*inner`）・ライン内添字 `i`（`axis_size`
/// 未満のときのみ有効）から flat 位置を求める（モジュール doc「ライン
/// 分解」参照）。`axis_size` に `dim_size` を渡せば入力位置、`out_len`
/// を渡せば出力位置になる（[`finalize_host`] が両方に本関数を使う）。
#[cfg(test)]
fn line_pos(line: usize, i: usize, axis_size: usize, inner: usize) -> usize {
    (line / inner) * axis_size * inner + (line % inner) + i * inner
}

/// [`crate::kernels_sort::SORT_BUILD_KEYS_U64`] の逐語ホストモデル
/// （テスト専用）。`input`（行優先 contiguous）から `plan` に従って
/// 合成キー配列（`lines * padded` 長）を構築する。
#[cfg(test)]
pub fn build_keys_host(input: &[f32], plan: &SortPlan, descending: bool) -> Vec<u64> {
    let SortPlan {
        dim_size,
        inner,
        lines,
        padded,
        ..
    } = *plan;
    let mut keys = vec![PADDING_KEY; lines * padded];
    for line in 0..lines {
        for i in 0..dim_size {
            let in_pos = line_pos(line, i, dim_size, inner);
            let v = input[in_pos];
            keys[line * padded + i] = composite_key(v, i as u32, descending);
        }
    }
    keys
}

/// [`crate::kernels_sort::BITONIC_STEP_U64`] の逐語ホストモデル（1
/// ライン分。テスト専用）。
#[cfg(test)]
pub fn bitonic_step_u64_host(keys: &mut [u64], line: usize, j: usize, k: usize, padded: usize) {
    let base = line * padded;
    for i in 0..padded {
        let ixj = i ^ j;
        if ixj <= i || ixj >= padded {
            continue;
        }
        let a = keys[base + i];
        let b = keys[base + ixj];
        let ascending = (i & k) == 0;
        let should_swap = if ascending { a > b } else { a < b };
        if should_swap {
            keys[base + i] = b;
            keys[base + ixj] = a;
        }
    }
}

/// `keys`（`lines` 本・各 `padded` 長）の全ラインへ標準的なビットニック
/// ソート（昇順）を適用する（テスト専用）。
#[cfg(test)]
pub fn bitonic_sort_all_lines_host(keys: &mut [u64], lines: usize, padded: usize) {
    if padded <= 1 {
        return;
    }
    for line in 0..lines {
        let mut k = 2usize;
        while k <= padded {
            let mut j = k / 2;
            while j >= 1 {
                bitonic_step_u64_host(keys, line, j, k, padded);
                j /= 2;
            }
            k *= 2;
        }
    }
}

/// [`crate::kernels_sort::SORT_FINALIZE_F32`] の逐語ホストモデル
/// （テスト専用）。`out_len <= dim_size` を要求する（`sort` は
/// `out_len == dim_size`、`topk` は `out_len == k`）。
#[cfg(test)]
pub fn finalize_host(
    input: &[f32],
    keys: &[u64],
    plan: &SortPlan,
    out_len: usize,
) -> (Vec<f32>, Vec<i32>) {
    let SortPlan {
        dim_size,
        inner,
        lines,
        padded,
        ..
    } = *plan;
    let out_numel = lines * out_len;
    let mut values = vec![0f32; out_numel];
    let mut index = vec![0i32; out_numel];
    for line in 0..lines {
        for o in 0..out_len {
            let key = keys[line * padded + o];
            let idx = (key & 0xFFFF_FFFF) as usize;
            debug_assert!(
                idx < dim_size,
                "finalize_host: idx out of range (contract violation)"
            );
            let in_pos = line_pos(line, idx, dim_size, inner);
            let out_pos = line_pos(line, o, out_len, inner);
            values[out_pos] = input[in_pos];
            index[out_pos] = idx as i32;
        }
    }
    (values, index)
}

/// build → sort → finalize を通しで実行するホストモデル（テスト専用。
/// `sort.rs::CudaSort::run_sort_f32` の GPU 側実装と同一のアルゴリズム
/// をホスト上で再現し、CPU 参照実装〈`fandhe_ai_backend_cpu::
/// sort_topk`〉との bit 一致を検証する主ゲートとして使う）。
#[cfg(test)]
pub fn sort_lines_host_model(
    input: &[f32],
    shape: &[usize],
    dim: usize,
    descending: bool,
    out_len: usize,
) -> Result<(Vec<f32>, Vec<i32>), SortPrepareError> {
    let plan = plan_sort(shape, dim)?;
    let mut keys = build_keys_host(input, &plan, descending);
    bitonic_sort_all_lines_host(&mut keys, plan.lines, plan.padded);
    Ok(finalize_host(input, &keys, &plan, out_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_key_normalizes_nan_and_zero() {
        let nan1 = f32::NAN;
        let nan2 = -f32::NAN;
        let nan3 = f32::from_bits(f32::NAN.to_bits() | 1);
        assert_eq!(value_key(nan1), 0xFFFF_FFFF);
        assert_eq!(value_key(nan2), 0xFFFF_FFFF);
        assert_eq!(value_key(nan3), 0xFFFF_FFFF);
        assert_eq!(value_key(0.0), value_key(-0.0));
    }

    #[test]
    fn value_key_matches_total_cmp_ordering() {
        let mut values = [
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f32::MAX,
            f32::INFINITY,
        ];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let keys: Vec<u32> = values.iter().map(|&v| value_key(v)).collect();
        for w in keys.windows(2) {
            assert!(
                w[0] <= w[1],
                "value_key ordering diverges: {:#x} > {:#x}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn plan_sort_basic_shapes() {
        let plan = plan_sort(&[3, 4], 1).unwrap();
        assert_eq!(plan.outer, 3);
        assert_eq!(plan.dim_size, 4);
        assert_eq!(plan.inner, 1);
        assert_eq!(plan.lines, 3);
        assert_eq!(plan.padded, 4);

        let plan0 = plan_sort(&[3, 4], 0).unwrap();
        assert_eq!(plan0.outer, 1);
        assert_eq!(plan0.dim_size, 3);
        assert_eq!(plan0.inner, 4);
        assert_eq!(plan0.lines, 4);
        assert_eq!(plan0.padded, 4);
    }

    #[test]
    fn plan_sort_rejects_dim_size_overflow() {
        let err = plan_sort(&[(i32::MAX as usize) + 1], 0).unwrap_err();
        assert!(matches!(err, SortPrepareError::DimSizeTooLarge { .. }));
    }

    /// [`sort_lines_host_model`] と CPU 参照実装
    /// （`fandhe_ai_backend_cpu::sort_topk`）の bit 一致を網羅的に確認
    /// する主ゲート（実機なしでアルゴリズムの正しさを固定する）。
    #[test]
    fn sort_lines_host_model_matches_cpu_reference_bit_exact() {
        use bench_harness::rng::Xorshift64Star;
        use fandhe_ai_tensor_core::{BackendOps, Tensor};

        let shapes: &[(&[usize], usize)] = &[
            (&[4], 0),
            (&[1, 4], 1),
            (&[3, 4], 1),
            (&[4, 3], 0),
            (&[2, 3, 4], 1),
            (&[2, 3, 4], 2),
            (&[2, 3, 4], 0),
            (&[1, 7], 1),
            (&[1, 8], 1),
            (&[1, 255], 1),
            (&[1, 256], 1),
            (&[1, 257], 1),
        ];

        for (seed, &(shape, dim)) in shapes.iter().enumerate() {
            let numel: usize = shape.iter().product();
            let mut rng = Xorshift64Star::new(30_000 + seed as u64);
            let mut data = rng.fill_vec(numel);
            // NaN（両符号）・±0 を注入し同値・特殊値の扱いも検証する。
            if numel >= 4 {
                data[0] = f32::NAN;
                data[1] = -f32::NAN;
                data[2] = 0.0;
                data[3] = -0.0;
            }

            for &descending in &[false, true] {
                let dim_size = shape[dim];
                for &out_len in &[0usize, 1, dim_size / 2, dim_size] {
                    if out_len > dim_size {
                        continue;
                    }
                    let (values, index) =
                        sort_lines_host_model(&data, shape, dim, descending, out_len).unwrap();

                    let cpu_input = Tensor::new(data.clone(), shape).unwrap();
                    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
                    let (cpu_values, cpu_index) = if out_len == dim_size {
                        BackendOps::sort(&cpu_ops, &cpu_input, dim, descending).unwrap()
                    } else {
                        BackendOps::topk(&cpu_ops, &cpu_input, dim, out_len, descending).unwrap()
                    };
                    let cpu_values: Vec<f32> = cpu_values.contiguous().as_slice().unwrap().to_vec();
                    let cpu_index: Vec<i32> = cpu_index.contiguous().as_slice().unwrap().to_vec();

                    assert_eq!(
                        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        cpu_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "values mismatch: shape={shape:?} dim={dim} descending={descending} out_len={out_len}"
                    );
                    assert_eq!(
                        index, cpu_index,
                        "index mismatch: shape={shape:?} dim={dim} descending={descending} out_len={out_len}"
                    );
                }
            }
        }
    }

    #[test]
    fn sort_lines_host_model_explicit_descending_case() {
        // `[-0.0, 0.0, NaN, -NaN, +inf]` を descending でソートすると、
        // NaN 2 個が添字順で先頭、次に +inf、最後に ±0 が添字順で並ぶ
        // （NaN は正規化後最大キー・descending で反転すると最小キーに
        // なるため先頭へ来る。計画書の明示ケース）。
        let data = vec![-0.0f32, 0.0, f32::NAN, -f32::NAN, f32::INFINITY];
        let (_, index) = sort_lines_host_model(&data, &[5], 0, true, 5).unwrap();
        assert_eq!(index, vec![2, 3, 4, 0, 1]);
    }
}
