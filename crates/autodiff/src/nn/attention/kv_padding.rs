//! `KvCache` の padding 付き（可変長）キャッシュを束ねる・分ける crate 内部
//! ヘルパー（イシュー #3105・親 #3103。`docs/facade-speculative-decoding-
//! batching-design.md` §21.4・§21.5・§21.9）。
//!
//! 本イシュー時点で非テストの呼び出し元がない（#3106 の `step_batched` が使う
//! 予定）ため `attention` の `#[cfg(test)]` 子モジュールとし、`#[allow]` を
//! 避ける。**#3106 で `cfg(test)` を外し、§21.9 の `max_active`／
//! `max_length` 検査を加える**。`lens` は親モジュールの非公開フィールドで、
//! 子モジュールからのみ直接読み書きする（公開面を増やさない）。

use super::KvCache;
use crate::error::AutodiffError;
use fandhe_ai_tensor_core::Tensor;

impl KvCache {
    /// テスト専用: 非公開 `lens` の参照（公開メソッド集合に入れない）。
    pub(crate) fn lens_for_test(&self) -> Option<&[usize]> {
        self.lens.as_deref()
    }

    /// バッチ 1 のキャッシュ列を左詰め padding で `[B, S_max, E]` へ束ねる。
    /// 入力検査・上限検査（§21.9）は確保の前に行い、算術はしない。
    /// 全行が同じ長さなら `lens = None`（正規化）。
    pub(crate) fn stack_left_padded(rows: &[KvCache]) -> Result<KvCache, AutodiffError> {
        let bad =
            |m: &str| AutodiffError::InvalidArgument(format!("KvCache::stack_left_padded: {m}"));
        let first = rows.first().ok_or_else(|| bad("rows が空"))?;
        let e = first
            .embed_dim()
            .ok_or_else(|| bad("空キャッシュの行がある"))?;
        let mut s_max = 0usize;
        for r in rows {
            if r.is_empty() {
                return Err(bad("空キャッシュの行がある"));
            }
            if r.batch() != Some(1) || r.lens.is_some() {
                return Err(bad("各行は B=1 かつ padding なしである必要がある"));
            }
            if r.embed_dim() != Some(e) {
                return Err(bad("embed_dim が不一致"));
            }
            s_max = s_max.max(r.seq_len());
        }
        let b = rows.len();
        let total = b
            .checked_mul(s_max)
            .and_then(|x| x.checked_mul(e))
            .filter(|n| {
                n.checked_mul(std::mem::size_of::<f32>())
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| bad("B x S_max x E x 4 が上限 (isize::MAX バイト) を超える"))?;
        let mut kd: Vec<f32> = Vec::with_capacity(total);
        let mut vd: Vec<f32> = Vec::with_capacity(total);
        let mut lens = Vec::with_capacity(b);
        for r in rows {
            let s = r.seq_len();
            let pad = (s_max - s) * e;
            kd.resize(kd.len() + pad, 0.0);
            vd.resize(vd.len() + pad, 0.0);
            let (Some(k), Some(v)) = (r.k.as_ref(), r.v.as_ref()) else {
                return Err(bad("k/v 不整合"));
            };
            let (kc, vc) = (k.contiguous(), v.contiguous());
            kd.extend_from_slice(kc.as_slice().ok_or_else(|| bad("k が host slice でない"))?);
            vd.extend_from_slice(vc.as_slice().ok_or_else(|| bad("v が host slice でない"))?);
            lens.push(s);
        }
        let k = Tensor::new(kd, &[b, s_max, e]).map_err(AutodiffError::Shape)?;
        let v = Tensor::new(vd, &[b, s_max, e]).map_err(AutodiffError::Shape)?;
        let lens = if lens.iter().all(|&l| l == s_max) {
            None
        } else {
            Some(lens)
        };
        Ok(KvCache {
            k: Some(k),
            v: Some(v),
            lens,
        })
    }

    /// [`KvCache::stack_left_padded`] の逆。各行の有効区間 `[S - len_r, S)` を
    /// `[1, len_r, E]`（padding なし）として切り出す（bit を変えないコピー）。
    pub(crate) fn split_rows(&self) -> Result<Vec<KvCache>, AutodiffError> {
        let bad = |m: &str| AutodiffError::InvalidArgument(format!("KvCache::split_rows: {m}"));
        let (Some(k), Some(v)) = (self.k.as_ref(), self.v.as_ref()) else {
            return Err(bad("空キャッシュは分割できない"));
        };
        let (b, s, e) = (k.shape()[0], k.shape()[1], k.shape()[2]);
        let (kc, vc) = (k.contiguous(), v.contiguous());
        let ks = kc.as_slice().ok_or_else(|| bad("k が host slice でない"))?;
        let vs = vc.as_slice().ok_or_else(|| bad("v が host slice でない"))?;
        let mut out = Vec::with_capacity(b);
        for r in 0..b {
            let len = self.valid_len(r).ok_or_else(|| bad("valid_len 不整合"))?;
            let start = r * s * e + (s - len) * e;
            let end = (r + 1) * s * e;
            let kt =
                Tensor::new(ks[start..end].to_vec(), &[1, len, e]).map_err(AutodiffError::Shape)?;
            let vt =
                Tensor::new(vs[start..end].to_vec(), &[1, len, e]).map_err(AutodiffError::Shape)?;
            out.push(KvCache {
                k: Some(kt),
                v: Some(vt),
                lens: None,
            });
        }
        Ok(out)
    }
}
