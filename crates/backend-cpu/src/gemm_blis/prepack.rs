//! 常駐重み W の pack 済み B panel キャッシュ（イシュー #3200・既定 OFF）。
//!
//! 役割: `TwoDDynamic` 経路は job ごとに B（= 常駐重み W）の列帯を毎回 pack する。推論では W が
//! 変わらないため、この pack は呼び出しのたびに繰り返される無駄である。本モジュールは W 全幅の
//! pack 済み panel を W のバッファハンドル（`memory::CpuBufferHandle`）に持たせ、forward の B を
//! そこから切り出して読めるようにする。ゲート [`RESIDENT_W_PREPACK_ENABLED`] は既定 OFF で、
//! OFF のとき呼び出し元（`ops.rs`）は [`PrepackSource`] を渡さず、現行の job 内 pack と同一になる。
//!
//! 呼び出し元: `ops.rs` の `gemm_resident_rhs_impl`（train forward）・`linear_forward_device`
//! （infer チェーン）が `gemm_blis_bias_act_parallel_prepacked` へ [`PrepackSource`] を渡し、
//! `gemm_blis_two_d_dynamic_region` が適格性を判定して panel を引き当て・構築する。
//!
//! 無効化: キャッシュはハンドルが所有し、ハンドルの `data` は private で書き込み口
//! `data_mut()` が必ずキャッシュを破棄する。ストア破棄・別 Tape（別ハンドル）・ABA は構造的に
//! 起きない。
//!
//! bit 一致の根拠（決定記録 `docs/cpu-gemm-small-m-job-merge-prepack-decision.md` §7）: pack は
//! 純粋な配置換えで、全幅 panel の各区画は job 内 `pack_b` の出力と同一のバイト列になる。
//! 微小カーネルへ渡す `kc_len`・ap／bp の内容・pc 昇順 p 昇順の FMA 連鎖は不変（FMA 契約・
//! tolerance は変更しない）。
//!
//! 決定性: キャッシュの排他制御は引き当て専用で、数値アキュムレータではない。panel の内容は
//! (W の内容, キー) の純関数のため、構築の競合（後勝ち）は結果に影響しない。構築は逐次で、
//! 並列ライブラリを使わない（determinism_inventory の allowlist を動かさない）。

use std::sync::{Arc, Mutex, PoisonError};

use super::pack::{BPackTile, pack_b};

/// 常駐 W の pack 済み panel キャッシュの本番ゲート（既定 OFF。反転は #3201 の A/B 後に
/// #3227 が判断する）。
pub(crate) const RESIDENT_W_PREPACK_ENABLED: bool = false;

/// 1 エントリの panel 語数の上限（`k * align_up(n, nr)`。4 MiB）。超える形状は通常経路。
pub(crate) const RESIDENT_W_PREPACK_MAX_WORDS: usize = 1 << 20;

/// 1 ハンドルが保持するエントリ数の上限。任意の view を大量に渡されたときの際限ない
/// メモリ増加を防ぐ。上限に達したら構築した panel をその呼び出しだけで使い、挿入しない。
pub(crate) const RESIDENT_W_PREPACK_MAX_ENTRIES: usize = 16;

/// キャッシュのキー。`offset` は view の offset（連結した 1 本の `params_buf` 上の L1 W と
/// L2 W を区別する）。転置は Nn 固定のためキーに含めない。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrepackKey {
    pub(crate) offset: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) nr: usize,
    pub(crate) kc: usize,
}

#[derive(Default)]
struct PrepackState {
    entries: Vec<(PrepackKey, Arc<[f32]>)>,
    #[cfg(test)]
    builds: usize,
    #[cfg(test)]
    hits: usize,
}

/// ハンドル付随の pack 済み panel キャッシュ。ハンドルと一緒に drop される。
#[derive(Default)]
pub(crate) struct PrepackCache {
    state: Mutex<PrepackState>,
}

impl std::fmt::Debug for PrepackCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // panel の全要素を出力しない（件数のみ）。
        f.debug_struct("PrepackCache")
            .field("entries", &self.len())
            .finish()
    }
}

impl PrepackCache {
    /// 現在のエントリ数。
    pub(crate) fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .len()
    }

    /// `key` の panel を返す。無ければロックの外で `build` を呼んで構築する。
    /// 同時構築の内容は同一なので後勝ちでよく、挿入時に同一キーが既にあれば既存を使う。
    /// エントリ数が上限なら挿入せず、構築した panel をそのまま返す。
    pub(crate) fn get_or_build(
        &self,
        key: PrepackKey,
        build: impl FnOnce() -> Vec<f32>,
    ) -> Arc<[f32]> {
        {
            #[cfg(test)]
            let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            #[cfg(not(test))]
            let st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((_, panel)) = st.entries.iter().find(|(k, _)| *k == key) {
                let panel = Arc::clone(panel);
                #[cfg(test)]
                {
                    st.hits += 1;
                }
                return panel;
            }
        }
        let built: Arc<[f32]> = Arc::from(build());
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, panel)) = st.entries.iter().find(|(k, _)| *k == key) {
            return Arc::clone(panel);
        }
        #[cfg(test)]
        {
            st.builds += 1;
        }
        if st.entries.len() < RESIDENT_W_PREPACK_MAX_ENTRIES {
            st.entries.push((key, Arc::clone(&built)));
        }
        built
    }

    /// 全エントリを破棄する。`&mut self` を要求するため競合せず、ロックを取らない。
    pub(crate) fn clear_mut(&mut self) {
        self.state
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .entries
            .clear();
    }

    /// テスト用: (構築回数, 引き当て回数)。
    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, usize) {
        let st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        (st.builds, st.hits)
    }
}

/// `gemm_blis_bias_act_parallel_prepacked` が受け取る引き当て元（`ops.rs` が W のハンドルから作る）。
#[derive(Clone, Copy)]
pub(crate) struct PrepackSource<'a> {
    pub(crate) cache: &'a PrepackCache,
    pub(crate) offset: usize,
}

/// job 本体へ渡す pack 済み全幅 panel の読み取りビュー。
#[derive(Clone, Copy)]
pub(crate) struct PrepackedB<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) n_padded: usize,
}

impl<'a> PrepackedB<'a> {
    /// pc ブロック（先頭 k 位置 `pc`・長さ `kc_len`）の大域 nr ブロック `jb0` から
    /// `nr_blocks` 個ぶんを連続スライスで返す。job 内 `pack_b` が `b_panel` へ書く内容と同一。
    pub(crate) fn slice(
        &self,
        nr: usize,
        pc: usize,
        kc_len: usize,
        jb0: usize,
        nr_blocks: usize,
    ) -> &'a [f32] {
        let start = pc * self.n_padded + jb0 * kc_len * nr;
        &self.data[start..start + nr_blocks * kc_len * nr]
    }
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// panel の語数 `k * align_up(n, nr)`。溢れたら `None`。
pub(crate) fn panel_words(k: usize, n: usize, nr: usize) -> Option<usize> {
    if nr == 0 {
        return None;
    }
    n.checked_next_multiple_of(nr)?.checked_mul(k)
}

/// 事前 pack を使える条件（純関数）。すべて満たすとき true:
/// Nn・`k * align_up(n, nr) <= RESIDENT_W_PREPACK_MAX_WORDS`（溢れは false）・job 内の jc 境界が
/// nr 格子と一致する（`nc_job <= blocks_nc` または `blocks_nc % nr == 0`。job 列帯の先頭は
/// `nc_job`（nr の倍数）の倍数なので、これで job 側の nr ブロック境界が大域の格子と揃う）。
/// 満たさないときは呼び出し元が job 内 pack へフォールバックする。
pub(crate) fn select_prepack(
    is_nn: bool,
    k: usize,
    n: usize,
    nr: usize,
    nc_job: usize,
    blocks_nc: usize,
) -> bool {
    if !is_nn || nr == 0 || n == 0 || k == 0 {
        return false;
    }
    match panel_words(k, n, nr) {
        Some(w) if w <= RESIDENT_W_PREPACK_MAX_WORDS => {}
        _ => return false,
    }
    nc_job <= blocks_nc || blocks_nc.is_multiple_of(nr)
}

/// W 全幅の pack 済み panel を構築する。配置は pc ブロックごとに全幅の大域 nr ブロックを
/// 順に並べる: 区画 (pc, jb) の先頭は `pc * align_up(n, nr) + jb * kc_len * nr`
/// （それ以前の pc ブロックの kc_len の総和が pc に等しい）。各区画は
/// `pack_b(col_start = jb * nr, nr_eff = min(nr, n - jb * nr))` の出力と同一。
pub(crate) fn build_full_width_panel(
    b: &[f32],
    k: usize,
    n: usize,
    nr: usize,
    kc: usize,
) -> Vec<f32> {
    let n_padded = align_up(n, nr);
    let mut out = vec![0.0f32; k * n_padded];
    for pc in (0..k).step_by(kc) {
        let kc_len = kc.min(k - pc);
        let base = pc * n_padded;
        for jb in 0..n_padded / nr {
            let col_start = jb * nr;
            let nr_eff = nr.min(n - col_start);
            let off = base + jb * kc_len * nr;
            pack_b(
                &mut out[off..off + kc_len * nr],
                b,
                BPackTile {
                    n_total: n,
                    kc_start: pc,
                    kc_len,
                    col_start,
                    nr,
                    nr_eff,
                },
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 16777216.0) - 0.5
            })
            .collect()
    }

    #[test]
    fn gate_is_off_by_default() {
        // 実行時 assert にする理由は small_m.rs の同名テストと同じ（#3201）。
        assert!(!std::hint::black_box(RESIDENT_W_PREPACK_ENABLED));
    }

    #[test]
    fn select_prepack_boundaries() {
        // Nt/Tn は不可。
        assert!(!select_prepack(false, 8, 16, 16, 16, 512));
        // 語数の境界（nr=16, n=16 → k=1<<16 で 1<<20 ちょうど）。
        assert!(select_prepack(true, 1 << 16, 16, 16, 16, 512));
        assert!(!select_prepack(true, (1 << 16) + 1, 16, 16, 16, 512));
        // 乗算の溢れは false。
        assert!(!select_prepack(true, usize::MAX, 16, 16, 16, 512));
        assert!(!select_prepack(true, 2, usize::MAX - 1, 16, 16, 512));
        // jc 境界: NR=12・NC=512 は 512 % 12 != 0。
        assert!(select_prepack(true, 8, 256, 12, 264, 512));
        assert!(!select_prepack(true, 8, 1024, 12, 1032, 512));
        // NR=16 は常に通る。
        assert!(select_prepack(true, 8, 1024, 16, 1024, 512));
        // 退化形状。
        assert!(!select_prepack(true, 0, 8, 16, 16, 512));
        assert!(!select_prepack(true, 8, 0, 16, 16, 512));
    }

    #[test]
    fn full_width_panel_matches_per_block_pack_b() {
        for &(k, n, nr, kc) in &[
            (5usize, 13usize, 4usize, 3usize),
            (9, 10, 12, 4),
            (784, 256, 12, 256),
            (300, 257, 16, 256),
            (256, 10, 32, 256),
        ] {
            let b = rnd(k * n, 7);
            let panel = build_full_width_panel(&b, k, n, nr, kc);
            assert_eq!(panel.len(), panel_words(k, n, nr).unwrap());
            let n_padded = align_up(n, nr);
            let view = PrepackedB {
                data: &panel,
                n_padded,
            };
            for pc in (0..k).step_by(kc) {
                let kc_len = kc.min(k - pc);
                for jb in 0..n_padded / nr {
                    let nr_eff = nr.min(n - jb * nr);
                    let mut want = vec![f32::NAN; kc_len * nr];
                    pack_b(
                        &mut want,
                        &b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jb * nr,
                            nr,
                            nr_eff,
                        },
                    );
                    let got = view.slice(nr, pc, kc_len, jb, 1);
                    let wb: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
                    let gb: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(wb, gb, "k={k} n={n} nr={nr} kc={kc} pc={pc} jb={jb}");
                }
            }
        }
    }

    fn key(offset: usize) -> PrepackKey {
        PrepackKey {
            offset,
            k: 4,
            n: 4,
            nr: 4,
            kc: 4,
        }
    }

    #[test]
    fn cache_hits_and_distinct_keys() {
        let c = PrepackCache::default();
        let a = c.get_or_build(key(0), || vec![1.0; 4]);
        let b = c.get_or_build(key(0), || unreachable!("hit のはず"));
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(c.counts(), (1, 1));
        let mut k2 = key(0);
        k2.k = 5;
        let mut k3 = key(0);
        k3.nr = 8;
        let _ = c.get_or_build(key(7), || vec![2.0; 4]);
        let _ = c.get_or_build(k2, || vec![3.0; 4]);
        let _ = c.get_or_build(k3, || vec![4.0; 4]);
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn clear_mut_empties_cache() {
        let mut c = PrepackCache::default();
        let _ = c.get_or_build(key(0), || vec![1.0; 4]);
        assert_eq!(c.len(), 1);
        c.clear_mut();
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn entry_cap_does_not_insert_but_returns_panel() {
        let c = PrepackCache::default();
        for i in 0..RESIDENT_W_PREPACK_MAX_ENTRIES {
            let _ = c.get_or_build(key(i), || vec![i as f32; 4]);
        }
        let extra = c.get_or_build(key(1000), || vec![9.0; 4]);
        assert_eq!(&extra[..], &[9.0; 4]);
        assert_eq!(c.len(), RESIDENT_W_PREPACK_MAX_ENTRIES);
    }

    /// #3201 の A/B 用の語数表示（メモリ記録 = `k * align_up(n, NR) * 4` バイト）。
    #[test]
    #[ignore = "報告専用（メモリ記録）"]
    fn resident_w_prepack_report() {
        for &(k, n) in &[(784usize, 256usize), (256, 10)] {
            for nr in [4usize, 12, 16, 32] {
                let w = panel_words(k, n, nr).unwrap();
                println!("k={k} n={n} NR={nr} words={w} bytes={}", w * 4);
            }
        }
    }

    // ---- gemm レベルの bit 一致（強制入口 vs job 内 pack の現行経路） ----

    use super::super::{
        Activation, GemmTranspose, default_blocks, gemm_blis_bias_act_parallel,
        gemm_blis_bias_act_parallel_prepacked,
        gemm_blis_parallel_two_d_dynamic_prepacked_with_params,
        gemm_blis_parallel_two_d_dynamic_with_params, install_in_pool_for_test,
    };

    fn special_vec(seed: u32, len: usize, special: bool) -> Vec<f32> {
        let mut v = rnd(len, seed);
        if special {
            for (i, x) in v.iter_mut().enumerate() {
                if i % 97 == 5 {
                    *x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][(i / 97) % 3];
                }
            }
        }
        v
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_gemm_bit_equal(m: usize, n: usize, k: usize, special: bool) {
        let a = special_vec(0x1234 ^ m as u32, m * k, special);
        let b = special_vec(0x9abc ^ n as u32, k * n, special);
        for threads in [1usize, 10, 20] {
            for (jpw, share_a) in [(2usize, false), (1, false), (1, true)] {
                let (want, got, builds) = install_in_pool_for_test(threads, || {
                    let mut want = vec![0.0f32; m * n];
                    gemm_blis_parallel_two_d_dynamic_with_params(
                        &a,
                        &b,
                        &mut want,
                        m,
                        n,
                        k,
                        default_blocks(),
                        jpw,
                        GemmTranspose::Nn,
                    )
                    .unwrap();
                    let cache = PrepackCache::default();
                    let mut got = vec![0.0f32; m * n];
                    gemm_blis_parallel_two_d_dynamic_prepacked_with_params(
                        &a,
                        &b,
                        &mut got,
                        m,
                        n,
                        k,
                        default_blocks(),
                        jpw,
                        share_a,
                        PrepackSource {
                            cache: &cache,
                            offset: 0,
                        },
                    )
                    .unwrap();
                    (want, got, cache.counts().0)
                });
                assert_eq!(
                    bits(&want),
                    bits(&got),
                    "{m}x{n}x{k} T={threads} jpw={jpw} share_a={share_a} special={special} builds={builds}"
                );
            }
        }
    }

    #[test]
    fn prepacked_is_bit_identical_on_forward_shapes() {
        for &(m, n, k) in &[(64usize, 256usize, 784usize), (64, 10, 256)] {
            assert_gemm_bit_equal(m, n, k, false);
            assert_gemm_bit_equal(m, n, k, true);
        }
    }

    #[test]
    fn prepacked_is_bit_identical_on_remainder_and_boundary_shapes() {
        // 端数・KC をまたぐ k・フォールバックが発火しうる広い n。
        for &(m, n, k) in &[
            (63usize, 257usize, 785usize),
            (65, 10, 9),
            (2, 13, 7),
            (64, 256, 1000),
            (64, 1024, 256),
        ] {
            assert_gemm_bit_equal(m, n, k, false);
        }
    }

    #[test]
    fn prepacked_builds_once_per_cache_and_hits_afterwards() {
        let (m, n, k) = (64usize, 256usize, 784usize);
        let a = rnd(m * k, 1);
        let b = rnd(k * n, 2);
        let cache = PrepackCache::default();
        install_in_pool_for_test(4, || {
            for _ in 0..3 {
                let mut c = vec![0.0f32; m * n];
                gemm_blis_bias_act_parallel_prepacked(
                    &a,
                    &b,
                    &mut c,
                    m,
                    n,
                    k,
                    None,
                    Activation::None,
                    Some(PrepackSource {
                        cache: &cache,
                        offset: 0,
                    }),
                )
                .unwrap();
            }
        });
        // n=256 は全 ISA で適格（nc_job <= NC）。1 回構築・2 回引き当て。
        assert_eq!(cache.counts(), (1, 2));
    }

    #[test]
    fn bias_act_prepacked_matches_public_entry_bit_for_bit() {
        for &(m, n, k) in &[(64usize, 256usize, 784usize), (64, 10, 256), (7, 33, 40)] {
            let a = rnd(m * k, 3);
            let b = rnd(k * n, 4);
            let bias = rnd(n, 5);
            for act in [Activation::None, Activation::Relu] {
                for bias_opt in [None, Some(&bias[..])] {
                    let cache = PrepackCache::default();
                    install_in_pool_for_test(10, || {
                        let mut want = vec![0.0f32; m * n];
                        gemm_blis_bias_act_parallel(&a, &b, &mut want, m, n, k, bias_opt, act)
                            .unwrap();
                        let mut got = vec![0.0f32; m * n];
                        gemm_blis_bias_act_parallel_prepacked(
                            &a,
                            &b,
                            &mut got,
                            m,
                            n,
                            k,
                            bias_opt,
                            act,
                            Some(PrepackSource {
                                cache: &cache,
                                offset: 0,
                            }),
                        )
                        .unwrap();
                        assert_eq!(bits(&want), bits(&got), "{m}x{n}x{k} {act:?}");
                    });
                }
            }
        }
    }
}
