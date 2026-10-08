//! KV キャッシュ巻き戻しの内部部品（イシュー #2885。親 #2499 系列。設計正本
//! `docs/facade-speculative-decoding-batching-design.md` §4.1 案 (i)・§7）。
//!
//! speculative decoding（#2886）が target の検証 forward（draft の K トークン
//! を 1 回で流す）の前に [`KvSnapshot::capture`] で `KvCache` 列を保存し、棄却時に
//! [`KvSnapshot::restore`] で検証前の状態へ戻す。その後の受理分の再 forward
//! （`forward_step`）は #2886 の責務で、本モジュールは差し替えのみを担う。
//!
//! # 不変条件
//!
//! - `KvCache` の書き手は `MultiheadAttentionVars::forward_with_cache` のみ、
//!   という不変条件を緩めない。本モジュールは `KvCache` に公開メソッドを足さず、
//!   型全体の代入（`*cache = saved.clone()`）だけを行う。算術は一切行わない
//!   （案 (ii) の `pub(crate)` 切り詰め・案 (iii) の `pub` 追加は採らない）
//! - `KvCache::clone()` は `Tensor` の `Arc` 共有で O(1) であり、
//!   `forward_with_cache` は `Var::cat` で作った新テンソルで丸ごと差し替える
//!   ため、保存分の中身は後続 forward で変化しない（単体テストで固定）。
//!   保存分を保持している間は旧テンソルのメモリが解放されない
//!
//! # 対象外
//!
//! `num_kv_layers() == 0` のモデルには巻き戻す対象がなく、空の
//! [`KvSnapshot`] の `restore` は no-op で `Ok` になる。内部状態保持型
//! （`RefCell<StatefulAttention>` 等。`docs/facade-generate-decision.md`
//! §17.4）は `num_kv_layers() == 0` で状態なし型と区別できず、本部品は内部状態
//! を巻き戻せない。その型の speculative での扱いは設計記録 §10 の論点 3・8
//! （未承認）であり本実装の対象外。

use crate::error::AutodiffError;
use crate::nn::KvCache;

/// 検証 forward 前の KV 状態の保存分（案 (i)）。
#[derive(Debug, Clone)]
pub(super) struct KvSnapshot {
    layers: Vec<KvCache>,
}

impl KvSnapshot {
    /// `caches` を層ごとに clone して保存する。`Arc` 共有のためデータは複製
    /// されない。#2886 が検証 forward の直前に呼ぶ。
    pub(super) fn capture(caches: &[KvCache]) -> KvSnapshot {
        KvSnapshot {
            layers: caches.to_vec(),
        }
    }

    /// 保存分で `caches` を復元する。#2886 が棄却時に呼ぶ。
    ///
    /// 全層の検査を済ませてから書き込む（原子的）。いずれかの検査が失敗した
    /// 場合は `caches` を一切変更せず `Err` を返す。**`Err` を受けた呼び出し側は
    /// その `caches` で処理を続けてはならない**（握りつぶし禁止。fail-closed）。
    /// 書き込み後の自己確認が失敗した場合も `Err` であり、同様に続行不可。
    ///
    /// # Errors
    ///
    /// - 層数が保存時と異なる、または現在の `seq_len` が保存時より短い
    ///   （追記のみの履歴を前提とするため、`clear` 済み・別履歴の誤用）:
    ///   `AutodiffError::InvalidArgument`
    /// - 両方が非空で `batch`／`embed_dim` が保存時と異なる:
    ///   `AutodiffError::InvalidArgument`
    pub(super) fn restore(&self, caches: &mut [KvCache]) -> Result<(), AutodiffError> {
        if caches.len() != self.layers.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "KvSnapshot::restore: 層数が保存時と異なる（保存 {} / 現在 {}）",
                self.layers.len(),
                caches.len()
            )));
        }
        for (i, (saved, cur)) in self.layers.iter().zip(caches.iter()).enumerate() {
            if cur.seq_len() < saved.seq_len() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "KvSnapshot::restore: 層 {i} の現在の seq_len {} が保存時の {} より短い\
                     （追記のみの履歴でない）",
                    cur.seq_len(),
                    saved.seq_len()
                )));
            }
            if !saved.is_empty()
                && !cur.is_empty()
                && (saved.batch() != cur.batch() || saved.embed_dim() != cur.embed_dim())
            {
                return Err(AutodiffError::InvalidArgument(format!(
                    "KvSnapshot::restore: 層 {i} の batch／embed_dim が保存時と異なる"
                )));
            }
        }
        for (saved, cur) in self.layers.iter().zip(caches.iter_mut()) {
            *cur = saved.clone();
        }
        // 書き込み後の自己確認（fail-closed）。
        for (i, (saved, cur)) in self.layers.iter().zip(caches.iter()).enumerate() {
            if cur.is_empty() != saved.is_empty()
                || cur.seq_len() != saved.seq_len()
                || cur.batch() != saved.batch()
                || cur.embed_dim() != saved.embed_dim()
            {
                return Err(AutodiffError::InvalidArgument(format!(
                    "KvSnapshot::restore: 層 {i} の復元後の状態が保存分と一致しない"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use crate::nn::MultiheadAttention;
    use fandhe_ai_tensor_core::Tensor;

    const E: usize = 4;
    const H: usize = 2;

    fn seq(b: usize, len: usize, phase: f32) -> Tensor<f32> {
        let data: Vec<f32> = (0..b * len * E)
            .map(|i| (i as f32 * 0.37 + phase).sin())
            .collect();
        Tensor::new(data, &[b, len, E]).expect("test fixture: 形状とデータ長は一致")
    }

    fn mha(seed: u64) -> MultiheadAttention {
        MultiheadAttention::new(E, H, true, seed).expect("test fixture")
    }

    /// 実際の `forward_with_cache` で `cache` を伸ばし、出力を返す。
    fn step(m: &MultiheadAttention, x: &Tensor<f32>, cache: &mut KvCache) -> Tensor<f32> {
        let tape = Tape::new();
        let xv = tape.var(x);
        m.bind(&tape)
            .forward_with_cache(&xv, &xv, &xv, cache)
            .expect("test fixture: forward_with_cache")
            .to_tensor()
    }

    fn flat(t: &Tensor<f32>) -> (Vec<usize>, Vec<f32>) {
        (t.shape().to_vec(), t.host_slice().to_vec())
    }

    fn assert_same(a: &KvCache, b: &KvCache) {
        assert_eq!(a.is_empty(), b.is_empty());
        assert_eq!(a.seq_len(), b.seq_len());
        assert_eq!(a.batch(), b.batch());
        assert_eq!(a.embed_dim(), b.embed_dim());
        assert_eq!(a.k().map(flat), b.k().map(flat));
        assert_eq!(a.v().map(flat), b.v().map(flat));
    }

    /// REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）。
    fn close(a: f32, b: f32) -> bool {
        let d = (a as f64 - b as f64).abs();
        d < 1e-5 || d / (b as f64).abs().max(f64::MIN_POSITIVE) < 1e-3
    }

    #[test]
    fn restore_returns_cache_identical_to_pre_verify_state() {
        let m = mha(11);
        let mut caches = vec![KvCache::new()];
        step(&m, &seq(1, 3, 0.0), &mut caches[0]);
        let snap = KvSnapshot::capture(&caches);
        let before = caches[0].clone();
        step(&m, &seq(1, 2, 1.0), &mut caches[0]); // 検証 forward（L_new=2）
        assert_eq!(caches[0].seq_len(), 5);
        snap.restore(&mut caches).unwrap();
        assert_eq!(caches[0].seq_len(), 3);
        assert_same(&caches[0], &before);
    }

    #[test]
    fn snapshot_is_unaffected_by_later_forward() {
        let m = mha(11);
        let mut caches = vec![KvCache::new()];
        step(&m, &seq(1, 3, 0.0), &mut caches[0]);
        let snap = KvSnapshot::capture(&caches);
        let k_before = flat(snap.layers[0].k().unwrap());
        let v_before = flat(snap.layers[0].v().unwrap());
        step(&m, &seq(1, 1, 2.0), &mut caches[0]);
        step(&m, &seq(1, 2, 3.0), &mut caches[0]);
        assert_eq!(snap.layers[0].seq_len(), 3);
        assert_eq!(flat(snap.layers[0].k().unwrap()), k_before);
        assert_eq!(flat(snap.layers[0].v().unwrap()), v_before);
    }

    #[test]
    fn replay_after_restore_matches_straight_run() {
        let m = mha(11);
        let accepted = seq(1, 1, 4.0);

        let mut rewound = vec![KvCache::new()];
        step(&m, &seq(1, 3, 0.0), &mut rewound[0]);
        let snap = KvSnapshot::capture(&rewound);
        step(&m, &seq(1, 2, 1.0), &mut rewound[0]);
        snap.restore(&mut rewound).unwrap();
        let out_rewound = step(&m, &accepted, &mut rewound[0]);

        let mut straight = [KvCache::new()];
        step(&m, &seq(1, 3, 0.0), &mut straight[0]);
        let out_straight = step(&m, &accepted, &mut straight[0]);

        let (sa, da) = flat(&out_rewound);
        let (sb, db) = flat(&out_straight);
        assert_eq!(sa, sb);
        assert!(da.iter().zip(&db).all(|(a, b)| close(*a, *b)));
        assert_eq!(rewound[0].seq_len(), straight[0].seq_len());
    }

    #[test]
    fn restore_empty_snapshot_for_zero_layers_is_noop() {
        let snap = KvSnapshot::capture(&[]);
        let mut caches: Vec<KvCache> = Vec::new();
        snap.restore(&mut caches).unwrap();
        assert!(caches.is_empty());
    }

    #[test]
    fn restore_rejects_layer_count_mismatch_without_mutation() {
        let m = mha(11);
        let mut caches = vec![KvCache::new(), KvCache::new()];
        step(&m, &seq(1, 2, 0.0), &mut caches[0]);
        step(&m, &seq(1, 2, 0.5), &mut caches[1]);
        let snap = KvSnapshot::capture(&caches);
        let mut fewer = vec![caches[0].clone()];
        let orig = fewer[0].clone();
        assert!(snap.restore(&mut fewer).is_err());
        assert_same(&fewer[0], &orig);
    }

    #[test]
    fn restore_rejects_shorter_current_cache_without_mutation() {
        let m = mha(11);
        let mut caches = vec![KvCache::new()];
        step(&m, &seq(1, 3, 0.0), &mut caches[0]);
        let snap = KvSnapshot::capture(&caches);
        caches[0].clear();
        assert!(snap.restore(&mut caches).is_err());
        assert!(caches[0].is_empty());
    }

    #[test]
    fn restore_rejects_batch_mismatch_without_mutation() {
        let m = mha(11);
        let mut caches = vec![KvCache::new()];
        step(&m, &seq(1, 2, 0.0), &mut caches[0]);
        let snap = KvSnapshot::capture(&caches);
        let mut other = vec![KvCache::new()];
        step(&m, &seq(2, 3, 0.0), &mut other[0]);
        let orig = other[0].clone();
        assert!(snap.restore(&mut other).is_err());
        assert_same(&other[0], &orig);
    }

    #[test]
    fn multi_layer_restore_is_atomic_and_complete() {
        let m = mha(11);
        let mut caches = vec![KvCache::new(), KvCache::new()];
        step(&m, &seq(1, 2, 0.0), &mut caches[0]);
        step(&m, &seq(1, 2, 0.5), &mut caches[1]);
        let snap = KvSnapshot::capture(&caches);
        let before: Vec<KvCache> = caches.clone();

        // 全層が伸びた状態から一括復元される。
        step(&m, &seq(1, 2, 1.0), &mut caches[0]);
        step(&m, &seq(1, 2, 1.5), &mut caches[1]);
        snap.restore(&mut caches).unwrap();
        assert_same(&caches[0], &before[0]);
        assert_same(&caches[1], &before[1]);

        // 後ろの層だけ検査失敗: 先頭層も書き換わらない。
        step(&m, &seq(1, 2, 2.0), &mut caches[0]);
        caches[1].clear();
        let grown0 = caches[0].clone();
        assert!(snap.restore(&mut caches).is_err());
        assert_same(&caches[0], &grown0);
        assert!(caches[1].is_empty());
    }
}
