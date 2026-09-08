//! 2 次元タイルジョブ分配（イシュー #753・§3.2）の純関数群。
//!
//! 現行の並列経路（[`super::gemm_blis_parallel`]）は C を行方向へ
//! `m.div_ceil(num_threads)` 行ずつの静的パネルへ分割する
//! （`par_chunks_mut(panel_rows * n)`）。この方式は MC タイル境界と
//! パネル境界が一致しないため、MC タイル数が `num_threads` で割り切れ
//! ない形状では端数タイルが特定 worker へ偏る（#753 実装計画 §3.2）。
//!
//! 本モジュールはこの偏りを是正するための 2 段階の純関数を提供する:
//!
//! 1. [`tile_grid`]: M×N を MC×NC 単位のミニタイルへ区切った 2 次元
//!    ジョブ空間そのもの（イシューが指す「2 次元タイルジョブ分配」の
//!    対象空間）。重複なし・被覆完全であることを `mod tests` で検証する。
//! 2. [`row_ranges_for_workers`]: MC タイル**数**を [`split_evenly`]
//!    （gemm crate `gemm.rs` の n_jobs 分配方式を参照した均等割り）で
//!    worker 数へ分配してから、行範囲（連続区間）へ変換する。
//!
//! ## unsafe を使わない設計判断（PR #766 の教訓の反映）
//!
//! [`tile_grid`] が表す (行タイル, 列タイル) の完全な 2 次元ジョブ空間を
//! worker へ非連続に分配する実装（gemm crate 本来の方式）は、C への
//! 書き込みが行方向にも列方向にも入り組むため、`&mut [f32]` の借用検査を
//! 素朴には満たせず生ポインタ経由の `unsafe` ラッパーが必要になる。
//! PR #766 で「常に不活性な sysctl FFI」が P0/P1 指摘で撤去された経緯を
//! 踏まえ、`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の
//! 必要最小限に留める」方針との整合を優先し、#753 では **列方向は各
//! worker が担当行範囲の全列を内部で処理する（既存の
//! [`super::gemm_blis_ic_loop`]／[`super::gemm_blis_region`] がそのまま
//! 対応する）行方向のみの安全な分配**を採用する（[`row_ranges_for_workers`]）。
//! [`tile_grid`] は「偏りなく被覆する 2 次元ジョブ空間」という設計上の
//! 前提を独立に検証するための純関数として残し、実行時分配（安全な
//! `split_at_mut` 連鎖。[`super::gemm_blis_parallel_2d_with_blocks`]）とは
//! 別に単体テストする。設計判断の詳細は
//! `docs/perf/cpu-gemm-runtime-cache-detect.md` を参照。

use std::ops::Range;

/// M×N のミニタイル格子上の 1 タイル（行範囲・列範囲の組）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tile {
    pub row: Range<usize>,
    pub col: Range<usize>,
}

/// `[0, total)` を `size` 幅の連続区間へ分割する（最終区間のみ端数で
/// 短くなる）。`total == 0` または `size == 0` は空の結果を返す全域関数
/// （境界値であっても panic しない。REQ-8 境界検査の精神を純関数側にも
/// 適用する）。区間終端は `start + size` を素朴に評価せず
/// `saturating_add` してから `total` へ丸める（`size` に `usize::MAX`
/// 近傍の値が渡された場合でもオーバーフロー panic（debug）・折り返し
/// （release）せず `total` に飽和させる。codex-review 指摘・PR #773）。
pub(crate) fn bands(total: usize, size: usize) -> Vec<Range<usize>> {
    if total == 0 || size == 0 {
        return Vec::new();
    }
    (0..total)
        .step_by(size)
        .map(|start| start..start.saturating_add(size).min(total))
        .collect()
}

/// M×N を `mc`×`nc` 単位のミニタイルへ row-major で区切った一覧
/// （#753 §3.2「ジョブ空間」）。`mod tests` で「重複なし・被覆完全」を
/// 検証する（実行時分配ロジックそのものではなく、分配の前提となる
/// ジョブ空間の定義を独立に固定するための純関数。モジュールドキュメント
/// 「unsafe を使わない設計判断」参照）。
pub(crate) fn tile_grid(m: usize, n: usize, mc: usize, nc: usize) -> Vec<Tile> {
    let mut tiles = Vec::new();
    for row in bands(m, mc) {
        for col in bands(n, nc) {
            tiles.push(Tile {
                row: row.clone(),
                col,
            });
        }
    }
    tiles
}

/// `[0, total)` を `workers` 個の区間へできるだけ均等に分割する（gemm
/// crate `gemm.rs` の n_jobs 分配方式を参照した端数タイル均等化。余り
/// `total % workers` 個の worker が 1 つ多く受け取るため、区間長の差は
/// 常に高々 1）。`workers == 0` または `total == 0` は空の結果を返す
/// 全域関数。
pub(crate) fn split_evenly(total: usize, workers: usize) -> Vec<Range<usize>> {
    if workers == 0 || total == 0 {
        return Vec::new();
    }
    let base = total / workers;
    let extra = total % workers;
    let mut ranges = Vec::with_capacity(workers);
    let mut start = 0;
    for w in 0..workers {
        let len = base + usize::from(w < extra);
        if len == 0 {
            continue;
        }
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}

/// `blocks.mc` 単位の行タイル**数**を [`split_evenly`] で `workers` 個へ
/// 均等分配し、各 worker が担当する C の行範囲（連続・disjoint・`[0, m)`
/// を隙間なく被覆）を返す（#753）。
///
/// 従来の `m.div_ceil(num_threads)` による静的パネル分割はタイル境界を
/// 考慮せず行**数**のみを均等化するため、MC タイル数が `num_threads` で
/// 割り切れない形状ではタイル**数**が特定 worker（境界を跨ぐパネルを
/// 担当する worker）へ偏りうる。本関数はタイル数を先に均等化してから
/// 行範囲へ変換することで、この偏りを ±1 タイルへ抑える（モジュール
/// ドキュメント「unsafe を使わない設計判断」参照。実行側は
/// [`super::gemm_blis_parallel_2d_with_blocks`] が本関数の結果を
/// `split_at_mut` 連鎖で安全に disjoint 分割する）。
pub(crate) fn row_ranges_for_workers(m: usize, mc: usize, workers: usize) -> Vec<Range<usize>> {
    let row_bands = bands(m, mc);
    if row_bands.is_empty() {
        return Vec::new();
    }
    split_evenly(row_bands.len(), workers)
        .into_iter()
        .filter_map(|tile_idx_range| {
            let first = row_bands.get(tile_idx_range.start)?;
            let last = row_bands.get(tile_idx_range.end - 1)?;
            Some(first.start..last.end)
        })
        .collect()
}

/// (mc, nc) 2D job 格子（イシュー #1311・設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`
/// §5）。[`super::GemmDriverVariant::TwoDDynamic`] が [`job_grid`] の出力を使って
/// C を worker 数より多い job へ分割し、rayon work stealing で動的分配する
/// （`super::gemm_blis_two_d_dynamic_region` 参照）。`tiles` は
/// [`tile_grid`]`(m, n, mc_job, nc_job)` と同一集合であることを契約とする
/// （設計 §3 条件 3・`partition::tests::job_grid_tiles_equal_tile_grid_and_cover_exactly_once`
/// で検証）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JobGrid {
    /// 1 job が担当する最大行数（`mr` の倍数。端の行帯はこれより短い）。
    pub mc_job: usize,
    /// 1 job が担当する最大列数（`nr` の倍数。端の列帯はこれより短い）。
    pub nc_job: usize,
    /// `tile_grid(m, n, mc_job, nc_job)` と同一の job 一覧（被覆完全・
    /// 互いに素。row-major ではなく [`super::split_c_into_jobs`] が
    /// column-band-major へ並べ替える。設計 §6）。
    pub tiles: Vec<Tile>,
    /// `align_up` 後に実際に生成される行帯数（`ceil(m/mc_job)`）。
    pub row_bands: usize,
    /// `align_up` 後に実際に生成される列帯数（`ceil(n/nc_job)`）。
    pub col_bands: usize,
}

/// `x` を `align` の倍数へ切り上げる（`align == 0` は `x` をそのまま返す
/// 全域関数。`job_grid` は `mr`／`nr` に 0 を渡さない前提だが、境界検査の
/// 精神〈REQ-8〉に合わせ 0 除算を発生させない）。
fn align_up(x: usize, align: usize) -> usize {
    if align == 0 {
        return x;
    }
    x.div_ceil(align).saturating_mul(align)
}

/// [`job_grid`] の本体（`reached_fallback` を追加で返す内部版）。
///
/// `reached_fallback == true` は「§5.2 で証明されている探索の受理条件
/// （`cb = ceil(n/nr)` は collapse しないため必ず受理される）が実際には
/// 満たされなかった」ことを意味し、本来到達しないはずの防御分岐（設計
/// §5.2「フォールバック」節）を通ったことを示す。呼び出し元
/// [`job_grid`] はこのフラグを捨てるが、`partition::tests::
/// job_grid_all_candidates_rejected_is_unreachable` がこの内部関数を
/// 直接呼び、フラグが常に `false` であることを実行時検査する
/// （契約違反の検出。`unreachable!()` によるパニック変換は行わず、
/// 到達した場合も被覆完全・互いに素な最大分割を返すため正しさ自体は
/// 保たれる。設計 §5.2 末尾参照）。
fn job_grid_with_trace(
    m: usize,
    n: usize,
    mr: usize,
    nr: usize,
    blocks: &crate::gemm::BlockSizes,
    num_threads: usize,
    jobs_per_worker: usize,
) -> Result<(JobGrid, bool), crate::gemm::GemmError> {
    use crate::gemm::GemmError;

    // `m==0` または `n==0`: `num_threads` の値に関わらず空の JobGrid
    // （設計 §5.1「判定順序」。`num_threads==1` 特例より必ず先に判定
    // する。§5.2 の証明参照）。
    if m == 0 || n == 0 {
        return Ok((
            JobGrid {
                mc_job: align_up(m, mr),
                nc_job: align_up(n, nr),
                tiles: Vec::new(),
                row_bands: 0,
                col_bands: 0,
            },
            false,
        ));
    }

    // `num_threads <= 1`: job 1 個に固定し、以下のコスト最小化探索は
    // 経由しない（設計 §5.1・§5.2「num_threads == 1 の特例」。
    // `num_threads == 0` はスレッドが存在しない不正入力だが、直列扱いへ
    // fail-closed にフォールバックすることで 0 除算・空区間を避ける）。
    if num_threads <= 1 {
        let mc_job = align_up(m, mr);
        let nc_job = align_up(n, nr);
        let tiles = tile_grid(m, n, mc_job, nc_job);
        return Ok((
            JobGrid {
                mc_job,
                nc_job,
                tiles,
                row_bands: 1,
                col_bands: 1,
            },
            false,
        ));
    }

    let cap = m.div_ceil(mr.max(1));
    let max_col = n.div_ceil(nr.max(1));
    let max_jobs = cap
        .checked_mul(max_col)
        .ok_or(GemmError::DimProductOverflow)?;
    let target_jobs = jobs_per_worker
        .checked_mul(num_threads)
        .ok_or(GemmError::DimProductOverflow)?;
    let bound = target_jobs.min(max_jobs);

    // `real_rb(rb)`（設計 §5.2）: `rb` について単調非減少
    // （`ceil(m/rb)` が非増加・`align_up` が非減少写像のため
    // `mc_job(rb)` は非増加、よって `real_rb(rb) = ceil(m/mc_job(rb))`
    // は非減少）。この単調性を使い、`real_rb(rb) >= t` を満たす最小の
    // `rb` を二分探索で求める（`rb ∈ [1, cap]`）。
    let real_rb = |rb: usize| -> usize {
        let mc_job = align_up(m.div_ceil(rb.max(1)), mr);
        m.div_ceil(mc_job.max(1))
    };

    // best: (cost, tie_break_nc_job 降順のための符号反転しない生値, rb, cb, mc_job, nc_job)
    // コスト最小・同コストは nc_job が大きい方を採る（設計 §5.2）。
    let mut best: Option<(u128, usize, usize, usize, usize, usize)> = None;

    for cb in 1..=max_col {
        let nc_job = align_up(n.div_ceil(cb), nr);
        let real_cb = n.div_ceil(nc_job.max(1));
        let t = bound.div_ceil(real_cb.max(1));

        if real_rb(cap) < t {
            // この cb 候補は cap まで rb を引き上げても下限を満たせない
            // ため棄却する（設計 §5.2「具体例 1」と同型の棄却）。
            continue;
        }

        // `real_rb(rb) >= t` を満たす最小の rb を二分探索する
        // （単調非減少性より一意に定まる）。
        let mut lo = 1usize;
        let mut hi = cap;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if real_rb(mid) >= t {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let rb = lo;
        let mc_job = align_up(m.div_ceil(rb), mr);
        let real_rb_val = m.div_ceil(mc_job.max(1));

        // cost = real_cb_eff * m + real_rb * n（k は共通因子のため省略。
        // 設計 §5.2「残った候補について pack 総量モデル」）。
        let nc_blocks = nc_job.div_ceil(blocks.nc.max(1));
        let real_cb_eff = real_cb
            .checked_mul(nc_blocks)
            .ok_or(GemmError::DimProductOverflow)?;
        let cost = (real_cb_eff as u128)
            .checked_mul(m as u128)
            .and_then(|v| v.checked_add((real_rb_val as u128).checked_mul(n as u128)?))
            .ok_or(GemmError::DimProductOverflow)?;

        let candidate = (cost, nc_job, rb, cb, mc_job, nc_job);
        best = Some(match best {
            None => candidate,
            Some(cur) => {
                // 小さいコストを優先。同コストは nc_job が大きい方を採る
                // （タイブレーク。設計 §5.2）。
                if candidate.0 < cur.0 || (candidate.0 == cur.0 && candidate.1 > cur.1) {
                    candidate
                } else {
                    cur
                }
            }
        });
    }

    let (reached_fallback, rb, cb, mc_job, nc_job) = match best {
        Some((_, _, rb, cb, mc_job, nc_job)) => (false, rb, cb, mc_job, nc_job),
        None => {
            // 設計 §5.2 の証明により理論上到達しない防御分岐（`cb =
            // max_col` は collapse しないため常に受理されるはず）。
            // 到達しても被覆完全・互いに素な最大分割を返すため正しさは
            // 保たれる（`unreachable!()` によるパニック変換はしない。
            // `.claude/rules/coding-rust.md` 本番経路の panic 禁止）。
            let cb = max_col.max(1);
            let nc_job = align_up(n.div_ceil(cb), nr);
            let rb = cap.max(1);
            let mc_job = align_up(m.div_ceil(rb), mr);
            (true, rb, cb, mc_job, nc_job)
        }
    };
    let _ = (rb, cb); // 意図した rb/cb は診断用途のみ（実帯数は下で再計算）。

    let tiles = tile_grid(m, n, mc_job, nc_job);
    let row_bands = m.div_ceil(mc_job.max(1));
    let col_bands = n.div_ceil(nc_job.max(1));

    Ok((
        JobGrid {
            mc_job,
            nc_job,
            tiles,
            row_bands,
            col_bands,
        },
        reached_fallback,
    ))
}

/// (mc, nc) 2D job 格子を算出する純関数（イシュー #1311・設計 §5.1〜§5.2）。
///
/// `m`／`n`（C の寸法）・`mr`／`nr`（[`super::microkernel::Microkernel`]
/// の完全タイル寸法）・`blocks`（[`crate::gemm::BlockSizes`]。コスト
/// モデルの `nc` のみ使用）・`num_threads`（並列度）・`jobs_per_worker`
/// （目標 job 数 = `jobs_per_worker * num_threads` の係数）から、
/// worker 数より多い job 数へ分割した [`JobGrid`] を返す。
///
/// 判定順序は「`m==0`／`n==0`」→「`num_threads==1`」→「コスト最小化
/// 探索」の順に固定（設計 §5.1）。返す `tiles` は
/// `tile_grid(m, n, mc_job, nc_job)` と同一集合（被覆完全・互いに素。
/// [`tile_grid`] が既に検証済みの性質をそのまま継承する）。
///
/// 寸法計算はすべて `checked_mul`／`div_ceil`（0 除算しない `.max(1)`
/// ガード込み）で行い、乗算オーバーフローは
/// [`crate::gemm::GemmError::DimProductOverflow`] へ変換する
/// （OWASP A03・`.claude/rules/security.md`）。
#[cfg(test)]
pub(crate) fn job_grid(
    m: usize,
    n: usize,
    mr: usize,
    nr: usize,
    blocks: &crate::gemm::BlockSizes,
    num_threads: usize,
    jobs_per_worker: usize,
) -> Result<JobGrid, crate::gemm::GemmError> {
    let (grid, _) = job_grid_with_trace(m, n, mr, nr, blocks, num_threads, jobs_per_worker)?;
    Ok(grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_covers_total_without_overlap_with_remainder_tail() {
        let b = bands(23, 8);
        assert_eq!(b, vec![0..8, 8..16, 16..23]);
    }

    #[test]
    fn bands_returns_empty_for_zero_total_or_zero_size() {
        assert!(bands(0, 8).is_empty());
        assert!(bands(23, 0).is_empty());
    }

    #[test]
    fn bands_exact_multiple_has_no_short_tail() {
        assert_eq!(bands(16, 8), vec![0..8, 8..16]);
    }

    /// `size` が `usize::MAX` 近傍でも `start + size` の素朴な加算で
    /// オーバーフロー panic（debug）・折り返し（release）しないことを
    /// 固定する（codex-review 指摘・PR #773。`saturating_add` により
    /// 終端は `total` へ飽和する）。
    #[test]
    fn bands_saturates_end_when_size_is_near_usize_max() {
        let b = bands(23, usize::MAX);
        assert_eq!(b, vec![0..23]);
    }

    /// [`tile_grid`] が表す 2 次元ジョブ空間の「重複なし・被覆完全」不変
    /// 条件を、境界を跨ぐ非整除な形状で全点走査により検証する（#753 §3.2
    /// のジョブ空間定義そのものの正しさを固定するテスト）。
    #[test]
    fn tile_grid_covers_every_point_exactly_once() {
        let (m, n, mc, nc) = (257usize, 193usize, 64usize, 48usize);
        let tiles = tile_grid(m, n, mc, nc);

        let mut coverage = vec![0u8; m * n];
        for tile in &tiles {
            for i in tile.row.clone() {
                for j in tile.col.clone() {
                    coverage[i * n + j] += 1;
                }
            }
        }
        assert!(
            coverage.iter().all(|&c| c == 1),
            "tile_grid はすべての (i,j) をちょうど 1 回ずつ被覆するはず"
        );
    }

    #[test]
    fn tile_grid_is_empty_for_zero_m_or_n() {
        assert!(tile_grid(0, 100, 64, 48).is_empty());
        assert!(tile_grid(100, 0, 64, 48).is_empty());
    }

    #[test]
    fn split_evenly_covers_total_contiguously_with_balanced_lengths() {
        let ranges = split_evenly(23, 5);
        // 連続・被覆完全（[0, 23) を隙間なく分割）。
        let mut expected_start = 0;
        for r in &ranges {
            assert_eq!(r.start, expected_start);
            expected_start = r.end;
        }
        assert_eq!(expected_start, 23);

        // 区間長の差は高々 1（n_jobs 均等分配の性質）。
        let lens: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let min_len = *lens.iter().min().unwrap();
        let max_len = *lens.iter().max().unwrap();
        assert!(max_len - min_len <= 1);
    }

    #[test]
    fn split_evenly_returns_empty_for_zero_total_or_zero_workers() {
        assert!(split_evenly(0, 5).is_empty());
        assert!(split_evenly(23, 0).is_empty());
    }

    #[test]
    fn split_evenly_workers_exceeding_total_yields_at_most_total_ranges() {
        // workers > total の場合、余った worker には長さ 0 の区間が
        // 割り当たるはずのものを除外するため、区間数は total を超えない。
        let ranges = split_evenly(3, 10);
        assert_eq!(ranges.len(), 3);
        for r in &ranges {
            assert_eq!(r.end - r.start, 1);
        }
    }

    #[test]
    fn row_ranges_for_workers_covers_m_contiguously_and_disjointly() {
        let (m, mc, workers) = (523usize, 64usize, 5usize);
        let ranges = row_ranges_for_workers(m, mc, workers);

        let mut expected_start = 0;
        for r in &ranges {
            assert_eq!(r.start, expected_start, "行範囲は連続で隙間がないはず");
            assert!(r.end > r.start, "空の行範囲は含まれないはず");
            expected_start = r.end;
        }
        assert_eq!(expected_start, m, "行範囲は m を過不足なく被覆するはず");
    }

    #[test]
    fn row_ranges_for_workers_balances_tile_count_within_one() {
        // MC タイル数（523.div_ceil(64) = 9）が worker 数（4）で割り切れ
        // ない形状。タイル数の割り当て差が高々 1 であることを、各 worker
        // の担当行数から逆算したタイル数で検証する（境界を跨ぐ worker が
        // いないため `(end-start).div_ceil(mc)` で一致するタイル数が
        // 求まる）。
        let (m, mc, workers) = (523usize, 64usize, 4usize);
        let ranges = row_ranges_for_workers(m, mc, workers);
        let tile_counts: Vec<usize> = ranges
            .iter()
            .map(|r| (r.end - r.start).div_ceil(mc))
            .collect();
        let min_tiles = *tile_counts.iter().min().unwrap();
        let max_tiles = *tile_counts.iter().max().unwrap();
        assert!(
            max_tiles - min_tiles <= 1,
            "MC タイル数の worker 間の偏りは高々 1 のはず（実際: {tile_counts:?}）"
        );
    }

    #[test]
    fn row_ranges_for_workers_returns_empty_for_zero_m() {
        assert!(row_ranges_for_workers(0, 64, 4).is_empty());
    }

    /// [`job_grid`] が返す `tiles` が `tile_grid(m, n, mc_job, nc_job)`
    /// と同一集合（被覆完全・互いに素）であることを固定する（設計 §3
    /// 条件 3・§10「`job_grid_*`」）。複数形状・スレッド数・
    /// `jobs_per_worker` で検証する。
    #[test]
    fn job_grid_tiles_equal_tile_grid_and_cover_exactly_once() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let cases = [
            (257usize, 193usize, 8usize, 12usize, 4usize, 2usize),
            (1024, 1024, 8, 12, 10, 2),
            (523, 700, 6, 16, 8, 4),
            (16, 108, 8, 12, 8, 2),
            (48, 36, 8, 12, 2, 5),
        ];
        for (m, n, mr, nr, num_threads, jpw) in cases {
            let grid = job_grid(m, n, mr, nr, &blocks, num_threads, jpw).unwrap();
            let expected = tile_grid(m, n, grid.mc_job, grid.nc_job);
            assert_eq!(
                grid.tiles, expected,
                "shape=({m},{n}) mr={mr} nr={nr} T={num_threads} jpw={jpw}: \
                 tiles は tile_grid(m,n,mc_job,nc_job) と一致するはず"
            );

            let mut coverage = vec![0u8; m * n];
            for tile in &grid.tiles {
                for i in tile.row.clone() {
                    for j in tile.col.clone() {
                        coverage[i * n + j] += 1;
                    }
                }
            }
            assert!(
                coverage.iter().all(|&c| c == 1),
                "shape=({m},{n}): job_grid の tiles は被覆完全・互いに素のはず"
            );
        }
    }

    /// `mc_job` は `mr` の倍数・`nc_job` は `nr` の倍数（設計 §5.1 契約）
    /// であることを固定する。
    #[test]
    fn job_grid_aligns_mc_to_mr_and_nc_to_nr() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        for (m, n, mr, nr, num_threads, jpw) in [
            (257usize, 193usize, 8usize, 12usize, 4usize, 2usize),
            (1024, 1024, 6, 16, 20, 4),
            (523, 700, 8, 32, 8, 1),
        ] {
            let grid = job_grid(m, n, mr, nr, &blocks, num_threads, jpw).unwrap();
            assert_eq!(grid.mc_job % mr, 0, "mc_job は mr の倍数のはず");
            assert_eq!(grid.nc_job % nr, 0, "nc_job は nr の倍数のはず");
        }
    }

    /// job 数の下限（設計 §5.1「`到達可能最大 job 数`」を伴う `bound`）が
    /// `align_up` 後の**実帯数の積**（`row_bands * col_bands`）で満たされる
    /// ことを、ランダム形状 × スレッド数 × `jobs_per_worker` で検証する
    /// （設計 §5.1「選択規則はこの実帯数の積が bound 以上になる候補のみを
    /// 対象とする」契約。codex-review #1431／Cursor Bugbot 指摘の反例
    /// 固定化に対応する不変条件）。
    #[test]
    fn job_grid_meets_lower_bound_with_real_band_product() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            // xorshift64（決定的疑似乱数。テストの再現性を保つ）。
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..200 {
            let m = 1 + (next() % 4000) as usize;
            let n = 1 + (next() % 4000) as usize;
            let mr = [6usize, 8, 12][(next() % 3) as usize];
            let nr = [12usize, 16, 32][(next() % 3) as usize];
            let num_threads = [2usize, 3, 8, 10, 16, 20][(next() % 6) as usize];
            let jobs_per_worker = [1usize, 2, 4, 8][(next() % 4) as usize];

            let grid = job_grid(m, n, mr, nr, &blocks, num_threads, jobs_per_worker).unwrap();
            let cap = m.div_ceil(mr);
            let max_col = n.div_ceil(nr);
            let max_jobs = cap * max_col;
            let target = jobs_per_worker * num_threads;
            let bound = target.min(max_jobs);
            assert!(
                grid.row_bands * grid.col_bands >= bound,
                "m={m} n={n} mr={mr} nr={nr} T={num_threads} jpw={jobs_per_worker}: \
                 row_bands*col_bands={} は bound={bound} 以上のはず",
                grid.row_bands * grid.col_bands
            );
        }
    }

    /// `m==0`／`n==0`／`m<mr`／`n<nr`／`num_threads==1`／`usize::MAX` 近傍
    /// （オーバーフロー検出）の全域性を固定する（設計 §5.1・§10）。
    #[test]
    fn job_grid_handles_degenerate_inputs() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };

        // num_threads == 1: 直列扱い（job 1 個）。
        let grid = job_grid(1024, 1024, 8, 12, &blocks, 1, 4).unwrap();
        assert_eq!(grid.row_bands, 1);
        assert_eq!(grid.col_bands, 1);
        assert_eq!(grid.tiles.len(), 1);

        // m == 0 / n == 0: num_threads に関わらず空の JobGrid。
        for num_threads in [1usize, 8] {
            let grid = job_grid(0, 1024, 8, 12, &blocks, num_threads, 2).unwrap();
            assert_eq!(grid.row_bands, 0);
            assert_eq!(grid.col_bands, 0);
            assert!(grid.tiles.is_empty());

            let grid = job_grid(1024, 0, 8, 12, &blocks, num_threads, 2).unwrap();
            assert_eq!(grid.row_bands, 0);
            assert_eq!(grid.col_bands, 0);
            assert!(grid.tiles.is_empty());
        }

        // m < mr / n < nr: 1 タイルへ丸め込まれる。
        let grid = job_grid(5, 700, 8, 12, &blocks, 8, 2).unwrap();
        assert_eq!(grid.mc_job % 8, 0);
        let expected = tile_grid(5, 700, grid.mc_job, grid.nc_job);
        assert_eq!(grid.tiles, expected);

        let grid = job_grid(700, 5, 8, 12, &blocks, 8, 2).unwrap();
        assert_eq!(grid.nc_job % 12, 0);
        let expected = tile_grid(700, 5, grid.mc_job, grid.nc_job);
        assert_eq!(grid.tiles, expected);

        // usize::MAX 近傍: jobs_per_worker * num_threads の乗算が
        // オーバーフローし DimProductOverflow を返すはず。
        let err = job_grid(1024, 1024, 8, 12, &blocks, usize::MAX, usize::MAX).unwrap_err();
        assert!(matches!(err, crate::gemm::GemmError::DimProductOverflow));
    }

    /// `real_rb(rb)`／`real_cb(cb)` の単調非減少性（設計 §5.2 の探索が
    /// 依拠する性質）を、align_up の性質から独立に再現して検証する。
    #[test]
    fn job_grid_real_band_counts_are_monotone_non_decreasing() {
        let real_band = |total: usize, divisor: usize, align: usize| -> usize {
            let job = align_up(total.div_ceil(divisor), align);
            total.div_ceil(job)
        };
        for (total, align) in [(523usize, 64usize), (16, 8), (108, 12), (48, 8), (36, 12)] {
            let cap = total.div_ceil(align.max(1)).max(1);
            let mut prev = real_band(total, 1, align);
            for d in 2..=cap {
                let cur = real_band(total, d, align);
                assert!(
                    cur >= prev,
                    "total={total} align={align}: real_band は d について単調非減少のはず\
                     （d={d} で {cur} < {prev}）"
                );
                prev = cur;
            }
        }
    }

    /// 設計 §5.2「具体例 2」（alignment collapse が実際に発生する反例。
    /// codex-review #1431／Cursor Bugbot 指摘）を固定入力として再現し、
    /// `row_bands * col_bands >= bound` が成立することを検証する。
    #[test]
    fn job_grid_lower_bound_survives_alignment_collapse() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let (m, n, mr, nr, num_threads, jpw) = (16usize, 108usize, 8usize, 12usize, 8usize, 2usize);
        let grid = job_grid(m, n, mr, nr, &blocks, num_threads, jpw).unwrap();
        let bound = (jpw * num_threads).min(m.div_ceil(mr) * n.div_ceil(nr));
        assert_eq!(bound, 16);
        assert!(
            grid.row_bands * grid.col_bands >= bound,
            "具体例 2: row_bands*col_bands={} は bound={bound} 以上のはず",
            grid.row_bands * grid.col_bands
        );
        // 設計の解析どおり (rb,cb)=(2,9) が選ばれ 18 job になるはず。
        assert_eq!(grid.row_bands, 2);
        assert_eq!(grid.col_bands, 9);
    }

    /// 設計 §5.2「フォールバック到達不能」の証明（`cb = ceil(n/nr)` は
    /// collapse しないため必ず受理される）を、ランダム形状・スレッド数・
    /// `jobs_per_worker` の組で実行時検査する（`job_grid_with_trace` の
    /// `reached_fallback` が常に `false` であることを固定。イシュー
    /// #1431 の旧版で発生していた誤ったフォールバック転落の再発防止）。
    #[test]
    fn job_grid_all_candidates_rejected_is_unreachable() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let m = 1 + (next() % 5000) as usize;
            let n = 1 + (next() % 5000) as usize;
            let mr = [6usize, 8, 12][(next() % 3) as usize];
            let nr = [12usize, 16, 32][(next() % 3) as usize];
            let num_threads = [2usize, 3, 8, 10, 16, 20][(next() % 6) as usize];
            let jobs_per_worker = [1usize, 2, 4, 8][(next() % 4) as usize];

            let (_, reached_fallback) =
                job_grid_with_trace(m, n, mr, nr, &blocks, num_threads, jobs_per_worker).unwrap();
            assert!(
                !reached_fallback,
                "m={m} n={n} mr={mr} nr={nr} T={num_threads} jpw={jobs_per_worker}: \
                 フォールバック分岐は理論上到達不能のはず（設計 §5.2）"
            );
        }
    }

    /// 設計 §5.2「具体例 3」（`jobs_per_worker` を 5→6 に増やすと job 数が
    /// 18→12 に逆転していた codex-review #1431 指摘の反例）を固定入力
    /// として再現し、修正後は両方とも `(rb,cb)=(6,2)`・12 job になる
    /// ことを検証する。加えて `jobs_per_worker ∈ {1,2,4,8}`（#1312 の
    /// `{2,4}` を含む）でのランダム形状に対する非減少性を経験的に
    /// 検査する（設計 §5.2「この修正で証明できる単調性の範囲」のとおり
    /// 全域の数学的証明ではなく、このスイープ範囲内での回帰検証）。
    #[test]
    fn job_grid_jobs_per_worker_product_non_decreasing() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };

        // 固定入力（具体例 3）。
        let (m, n, mr, nr, num_threads) = (48usize, 36usize, 8usize, 12usize, 2usize);
        let grid5 = job_grid(m, n, mr, nr, &blocks, num_threads, 5).unwrap();
        let grid6 = job_grid(m, n, mr, nr, &blocks, num_threads, 6).unwrap();
        assert_eq!((grid5.row_bands, grid5.col_bands), (6, 2));
        assert_eq!((grid6.row_bands, grid6.col_bands), (6, 2));
        assert_eq!(grid5.tiles.len(), 12);
        assert_eq!(grid6.tiles.len(), 12);

        // 経験的な非減少性検査（ランダム形状 × jobs_per_worker
        // ∈ {1,2,4,8} のスイープ範囲内。全域の証明ではない）。
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..100 {
            let m = 1 + (next() % 3000) as usize;
            let n = 1 + (next() % 3000) as usize;
            let mr = [6usize, 8, 12][(next() % 3) as usize];
            let nr = [12usize, 16, 32][(next() % 3) as usize];
            let num_threads = [2usize, 3, 8, 10][(next() % 4) as usize];

            let mut prev_jobs = 0usize;
            for jpw in [1usize, 2, 4, 8] {
                let grid = job_grid(m, n, mr, nr, &blocks, num_threads, jpw).unwrap();
                let jobs = grid.row_bands * grid.col_bands;
                assert!(
                    jobs >= prev_jobs,
                    "m={m} n={n} mr={mr} nr={nr} T={num_threads} jpw={jpw}: \
                     job 数は jobs_per_worker について単調非減少のはず（{jobs} < {prev_jobs}）"
                );
                prev_jobs = jobs;
            }
        }
    }

    /// 設計 §5.3「解析 pack 表」の 6 行（N=1024/2048/4096 × T=10/20/8。
    /// NEON MR=8/NR=12/NC=512 前提）を `job_grid` の実装出力と突合する
    /// （`jobs_per_worker=2`）。食い違う場合は §5.3 表を実装出力へ更新
    /// する契約（設計 §5.3「値が食い違えば本表を更新する」）だが、本
    /// テストは #1311 実装時点での実装出力を固定する回帰として機能する。
    #[test]
    fn job_grid_reproduces_design_pack_table_rows() {
        let blocks = crate::gemm::BlockSizes {
            mc: 128,
            kc: 256,
            nc: 512,
        };
        let (mr, nr, jpw) = (8usize, 12usize, 2usize);
        // (N, T, expected row_bands, expected col_bands, expected mc_job, expected nc_job)
        let rows = [
            (1024usize, 10usize, 5usize, 4usize, 208usize, 264usize),
            (1024, 20, 8, 5, 128, 216),
            (2048, 10, 4, 5, 512, 420),
            (2048, 20, 8, 5, 256, 420),
            (4096, 8, 2, 9, 2048, 456),
            (4096, 20, 5, 9, 824, 456),
        ];
        for (n_dim, t, exp_rb, exp_cb, exp_mc, exp_nc) in rows {
            let grid = job_grid(n_dim, n_dim, mr, nr, &blocks, t, jpw).unwrap();
            assert_eq!(
                (grid.row_bands, grid.col_bands, grid.mc_job, grid.nc_job),
                (exp_rb, exp_cb, exp_mc, exp_nc),
                "N={n_dim} T={t}: 設計 §5.3 表との突合"
            );
        }
    }
}
