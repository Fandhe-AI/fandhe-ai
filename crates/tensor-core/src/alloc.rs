//! ホスト側 `Tensor<f32>` バッファの thread-local arena（イシュー #2104・opt-in・既定 OFF）。
//!
//! `autodiff` の `Tape`／`Gradients` が破棄・reset される際に回収した `Vec<f32>` を、
//! 次の tape の CPU カーネル（`backend-cpu` の出力確保）が再利用するための内部機構。
//! CPU の train／infer は step ごとに新しい `Tape` を作るため、寿命が Tape 1 個分の
//! プールでは step をまたぐ再利用が起きない。このため free list は thread-local とし
//! `Tape` より長く生かす（設計判断は `docs/tape-arena-reuse-design.md`）。
//!
//! - `pool.rs`（`MemoryOps`／`BufferHandle` 層のデバイスバッファ）・`DeviceParamStore` とは
//!   別層で、ホスト `Vec<f32>` のみを扱う。互いに干渉しない
//! - 既定 OFF（`HOST_ARENA_DEFAULT_ENABLED`）。OFF の間は全関数が素通し（`take_*` は
//!   従来どおり新規確保、`recycle_*` は単に drop）。ON でも出力は bit 同一
//!   （`take_zeroed_f32` は必ず全要素ゼロ埋めして返す）
//! - 参照中のメモリは再利用しない: 回収は `Arc` が一意所有のときだけ（`unsafe` なし）
//! - facade からは再公開しない（`#[doc(hidden)] pub` は crate 間結線のための内部面）

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::tensor::Tensor;

/// arena の既定有効フラグ。**既定 OFF**。両機体 A/B（`RULE.txt`）が ADOPT となった後の
/// 後続 PR でのみ `true` に切り替える。A/B の after 側 worktree ではこの定数だけを反転する。
pub(crate) const HOST_ARENA_DEFAULT_ENABLED: bool = false;

/// スレッドごとのアイドル保持上限（バイト）。REQ-14 14-3（メモリ係数）の安全側として
/// `pool.rs` の既定 128 MiB の半分とした。ガードレール閾値ではない。
pub(crate) const HOST_ARENA_MAX_BYTES: usize = 64 * 1024 * 1024;

/// arena の統計スナップショット（テスト・診断用）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostArenaStats {
    /// `take_*` が free list から再利用した回数。
    pub hit: u64,
    /// `take_*` が新規確保に落ちた回数（有効時のみ計上）。
    pub miss: u64,
    /// `recycle_*` が free list へ保持した回数。
    pub recycled: u64,
    /// 上限超過・長さ 0 等で保持せず解放した回数。
    pub rejected: u64,
    /// 現在アイドル保持しているバイト数。
    pub pooled_bytes: usize,
}

#[derive(Default)]
struct Arena {
    /// 要素数（`Vec::len` 相当。完全一致のみ再利用）→ 回収済みバッファ。
    free: HashMap<usize, Vec<Vec<f32>>>,
    stats: HostArenaStats,
}

thread_local! {
    static ARENA: RefCell<Arena> = RefCell::new(Arena::default());
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// スコープ限定の有効／無効上書きを元に戻す RAII ガード。
#[must_use = "ガードを drop すると上書きが元に戻る"]
pub struct HostArenaOverrideGuard {
    prev: Option<bool>,
    /// `!Send` 化。OVERRIDE は thread_local のため、別スレッドで drop されると作成元
    /// スレッドの上書きが復元されない（PR #2448 codex 指摘）。
    _not_send: PhantomData<*const ()>,
}

impl Drop for HostArenaOverrideGuard {
    fn drop(&mut self) {
        OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// 現スレッドの有効／無効をスコープ限定で上書きする（テスト・診断用。ネスト可）。
pub fn override_enabled_for_scope(enabled: bool) -> HostArenaOverrideGuard {
    let prev = OVERRIDE.with(|c| c.replace(Some(enabled)));
    HostArenaOverrideGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// 現スレッドで arena が有効か。
pub fn is_enabled() -> bool {
    OVERRIDE
        .with(Cell::get)
        .unwrap_or(HOST_ARENA_DEFAULT_ENABLED)
}

/// 長さ `len`・全要素 `0.0f32` の `Vec` を返す。有効時は回収済みバッファを再利用し、
/// 必ず全要素をゼロ埋めする（前回データの残留・bit 差を防ぐ）。
pub fn take_zeroed_f32(len: usize) -> Vec<f32> {
    if !is_enabled() || len == 0 {
        return vec![0.0f32; len];
    }
    match pop(len) {
        Some(mut v) => {
            v.fill(0.0);
            v
        }
        None => vec![0.0f32; len],
    }
}

/// 長さ 0・容量 `len` 以上の `Vec` を返す。呼び出し側が全要素を書き込む
/// `collect`／`push` 系の確保箇所専用（残留データは長さ 0 のため見えない）。
pub fn take_cleared_f32(len: usize) -> Vec<f32> {
    if !is_enabled() || len == 0 {
        return Vec::with_capacity(len);
    }
    match pop(len) {
        Some(mut v) => {
            v.clear();
            v
        }
        None => Vec::with_capacity(len),
    }
}

fn pop(len: usize) -> Option<Vec<f32>> {
    ARENA
        .try_with(|a| {
            let mut a = a.borrow_mut();
            let got = a.free.get_mut(&len).and_then(Vec::pop);
            match got {
                Some(v) => {
                    a.stats.hit += 1;
                    a.stats.pooled_bytes -= v.len() * 4;
                    Some(v)
                }
                None => {
                    a.stats.miss += 1;
                    None
                }
            }
        })
        .ok()
        .flatten()
}

/// `Vec<f32>` を回収する。無効時・長さ 0・上限超過は単に drop（解放）する。
pub fn recycle_f32(v: Vec<f32>) {
    if !is_enabled() || v.is_empty() {
        return;
    }
    // TLS 破棄後の呼び出しでは try_with が Err になり、v はそのまま drop される。
    let _ = ARENA.try_with(|a| {
        let mut a = a.borrow_mut();
        let bytes = v.len() * 4;
        if bytes > HOST_ARENA_MAX_BYTES || a.stats.pooled_bytes + bytes > HOST_ARENA_MAX_BYTES {
            a.stats.rejected += 1;
            return;
        }
        a.stats.pooled_bytes += bytes;
        a.stats.recycled += 1;
        a.free.entry(v.len()).or_default().push(v);
    });
}

/// `Tensor<f32>` が一意所有・offset 0・contiguous・`data.len() == numel` のときだけ
/// 内部 `Vec` を回収する。共有中の view 等は取り出さず drop する（参照中メモリの
/// 再利用を型で排除）。
pub fn recycle_tensor_f32(t: Tensor<f32>) {
    if !is_enabled() {
        return;
    }
    if let Ok(v) = t.try_into_unique_vec() {
        recycle_f32(v);
    }
}

/// 現スレッドの free list を全解放する（統計は保持）。
pub fn clear_thread_arena() {
    let _ = ARENA.try_with(|a| {
        let mut a = a.borrow_mut();
        a.free.clear();
        a.stats.pooled_bytes = 0;
    });
}

/// 現スレッドの統計を返す。
pub fn stats() -> HostArenaStats {
    ARENA.try_with(|a| a.borrow().stats).unwrap_or_default()
}

/// 現スレッドの統計を 0 に戻す（保持中バッファは維持し `pooled_bytes` のみ実値へ合わせる）。
pub fn reset_stats() {
    let _ = ARENA.try_with(|a| {
        let mut a = a.borrow_mut();
        let pooled = a.stats.pooled_bytes;
        a.stats = HostArenaStats {
            pooled_bytes: pooled,
            ..HostArenaStats::default()
        };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() {
        clear_thread_arena();
        reset_stats();
    }

    #[test]
    fn disabled_is_passthrough() {
        let _g = override_enabled_for_scope(false);
        fresh();
        recycle_f32(vec![1.0; 8]);
        assert_eq!(stats().recycled, 0);
        assert_eq!(take_zeroed_f32(8), vec![0.0; 8]);
        assert_eq!(stats().hit, 0);
    }

    #[test]
    fn reuse_exact_len_and_zero_fill() {
        let _g = override_enabled_for_scope(true);
        fresh();
        recycle_f32(vec![f32::NAN; 16]);
        assert_eq!(stats().recycled, 1);
        let v = take_zeroed_f32(16);
        assert_eq!(stats().hit, 1);
        assert!(v.iter().all(|x| x.to_bits() == 0));
        // 長さ不一致は再利用しない
        recycle_f32(vec![1.0; 16]);
        let w = take_zeroed_f32(15);
        assert_eq!(w.len(), 15);
        assert_eq!(stats().hit, 1);
    }

    #[test]
    fn cleared_returns_empty_with_capacity() {
        let _g = override_enabled_for_scope(true);
        fresh();
        recycle_f32(vec![3.0; 32]);
        let v = take_cleared_f32(32);
        assert!(v.is_empty());
        assert!(v.capacity() >= 32);
        assert_eq!(stats().hit, 1);
    }

    #[test]
    fn cap_rejects_and_bounds_pooled_bytes() {
        let _g = override_enabled_for_scope(true);
        fresh();
        recycle_f32(vec![0.0; HOST_ARENA_MAX_BYTES / 4 + 1]);
        assert_eq!(stats().rejected, 1);
        assert_eq!(stats().pooled_bytes, 0);
        let n = HOST_ARENA_MAX_BYTES / 4 / 2;
        recycle_f32(vec![0.0; n]);
        recycle_f32(vec![0.0; n]);
        recycle_f32(vec![0.0; n - 1]);
        assert!(stats().pooled_bytes <= HOST_ARENA_MAX_BYTES);
        assert_eq!(stats().rejected, 2);
        clear_thread_arena();
        assert_eq!(stats().pooled_bytes, 0);
    }

    #[test]
    fn shared_and_view_tensors_are_not_recycled() {
        let _g = override_enabled_for_scope(true);
        fresh();
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let keep = t.clone();
        recycle_tensor_f32(t);
        assert_eq!(stats().recycled, 0);
        assert_eq!(keep.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        let tt = keep.transpose(0, 1).unwrap();
        drop(keep);
        recycle_tensor_f32(tt);
        assert_eq!(stats().recycled, 0);
        let u = Tensor::new(vec![1.0f32; 4], &[2, 2]).unwrap();
        recycle_tensor_f32(u);
        assert_eq!(stats().recycled, 1);
    }

    #[test]
    fn override_guard_nests_and_restores() {
        assert!(!is_enabled());
        {
            let _a = override_enabled_for_scope(true);
            assert!(is_enabled());
            {
                let _b = override_enabled_for_scope(false);
                assert!(!is_enabled());
            }
            assert!(is_enabled());
        }
        assert!(!is_enabled());
    }
}
