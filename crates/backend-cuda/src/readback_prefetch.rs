//! readback 宛先の「背景事前タッチ済みバッファ」プール（イシュー #3225。既定 OFF）。
//!
//! # 位置づけ
//! - 呼び出し元: `memory::ReadbackSentinel::readback_prefetched`（f32 のみ override）。
//!   `ReadbackDest::PrefetchedPretouched`（env `FANDHE_AI_CUDA_READBACK_DEST=pretouched-prefetch`
//!   でのみ選ばれる。`crate::readback_policy`）が選ばれたときだけ本モジュールが動く。
//! - 動機（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §14）: reuse モードは tape が readout の
//!   宛先 `Vec` を保持し続けるため、毎回まだタッチしていない新規ページへ事前タッチと D2H をする。
//!   fresh モードは宛先が毎回 free されるので glibc がタッチ済みヒープを再利用できる。
//!   本プールは「次回の同形状の readback 用に、事前タッチ済みの `Vec<f32>` を背景スレッドで
//!   先に用意しておく」ことで事前タッチを D2H 区間の外（後続の checksum・H2D・カーネルと重なる位置）へ逃がす。
//!
//! # 安全性
//! - **新規 `unsafe` なし**。worker は CUDA に一切触れず、ホストメモリの確保と事前タッチ
//!   （`Vec::try_reserve_exact` + `resize`）だけを行い、`Vec<f32>`（`Send`）を受け渡す。
//! - 保持量は `cap_bytes`（Ready と Pending の合計）・`max_entries`・1 バッファ上限で頭打ちにする。
//!   同形状が `repeat_trigger` 回以上来たときだけ補充するため、1 回きりの形状ではメモリが増えない。
//! - 待ちには期限があり、期限切れ・poison・worker 不在・OOM はすべて呼び出し側のインライン確保
//!   （`PretouchedFresh` 相当）へ倒れる。OOM でプロセスは abort しない。
//!
//! # 数値契約
//! D2H が全要素を上書きするため返す `Vec` は `PretouchedFresh` と bit 同一。tolerance・REQ-2 判定は不変。

use std::collections::HashMap;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// 事前登録規則（`docs/perf/logs/cuda-readback-reuse-fix-3225/RULE.txt`）で凍結する定数群。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    /// これ未満のバッファは対象外（従来経路。小形状の挙動を変えない）。
    pub(crate) min_bytes: usize,
    /// 1 バッファの上限。
    pub(crate) max_buffer_bytes: usize,
    /// Ready と Pending の合計保持上限。
    pub(crate) cap_bytes: usize,
    /// 保持するエントリ（形状）数の上限。
    pub(crate) max_entries: usize,
    /// 同じ numel がこの回数使われてから補充を始める。
    pub(crate) repeat_trigger: u32,
    /// Pending の補充を待つ上限。
    pub(crate) wait_timeout: Duration,
}

pub(crate) const PREFETCH_LIMITS: Limits = Limits {
    min_bytes: 1 << 20,
    max_buffer_bytes: 128 << 20,
    cap_bytes: 256 << 20,
    max_entries: 8,
    repeat_trigger: 2,
    wait_timeout: Duration::from_millis(20),
};

enum SlotState {
    Empty,
    Pending,
    Ready(Vec<f32>),
}

struct Slot {
    seen: u32,
    state: SlotState,
}

#[derive(Default)]
struct State {
    slots: HashMap<usize, Slot>,
    /// 世代番号。`release_all` で加算し、補充要求に載せた世代と完了時に突き合わせる。
    /// 解放後に完了した古い補充結果を破棄し、解放直後に Ready が再登録されて
    /// 最大 `cap_bytes` が居座るのを防ぐ（REQ-14 の明示解放経路。PR #3540 指摘）。
    epoch: u64,
    hits: u64,
    waits: u64,
    misses: u64,
}

impl State {
    fn held_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter(|(_, s)| !matches!(s.state, SlotState::Empty))
            .map(|(n, _)| n * 4)
            .sum()
    }
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
}

fn lock(shared: &Shared) -> MutexGuard<'_, State> {
    // poison 後も回復する（保持するのは Vec と整数だけで、不整合になっても miss へ倒れるだけ）。
    shared.state.lock().unwrap_or_else(|p| p.into_inner())
}

/// 背景補充プール。worker スレッドを 1 本持つ。
pub(crate) struct Prefetcher {
    limits: Limits,
    shared: Arc<Shared>,
    tx: Mutex<Sender<(usize, u64)>>,
}

impl Prefetcher {
    /// worker を起動する。spawn 失敗時は `None`（呼び出し側が常にインライン確保へ倒す）。
    pub(crate) fn new(limits: Limits) -> Option<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            cv: Condvar::new(),
        });
        let (tx, rx) = channel::<(usize, u64)>();
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("fandhe-cuda-readback-prefetch".into())
            .spawn(move || {
                // tx が全て drop されると recv が Err を返して終了する。
                while let Ok((numel, epoch)) = rx.recv() {
                    // 確保前に世代と Pending 状態を確認し、解放済みの古い要求は確保・事前タッチせず破棄する
                    // （release_all と readback の反復でキューに溜まった古い要求が、最大 128 MiB の確保で
                    // メモリ帯域を消費し現世代の補充を遅らせるのを防ぐ。PR #3540 指摘）。
                    {
                        let g = lock(&worker_shared);
                        let live = g.epoch == epoch
                            && g.slots
                                .get(&numel)
                                .is_some_and(|s| matches!(s.state, SlotState::Pending));
                        if !live {
                            continue;
                        }
                    }
                    let buf = alloc_pretouched(numel);
                    let mut g = lock(&worker_shared);
                    // 世代不一致（要求後に release_all された）なら結果を破棄する（buf は drop）。
                    let current = g.epoch == epoch;
                    if let Some(slot) = g.slots.get_mut(&numel)
                        && current
                        && matches!(slot.state, SlotState::Pending)
                    {
                        slot.state = match buf {
                            Some(v) => SlotState::Ready(v),
                            None => SlotState::Empty,
                        };
                    }
                    drop(g);
                    worker_shared.cv.notify_all();
                }
            })
            .ok()?;
        Some(Self {
            limits,
            shared,
            tx: Mutex::new(tx),
        })
    }

    /// 対象サイズか（小形状・巨大形状は従来経路）。
    pub(crate) fn eligible(&self, numel: usize) -> bool {
        let bytes = numel.saturating_mul(4);
        bytes >= self.limits.min_bytes && bytes <= self.limits.max_buffer_bytes
    }

    /// 事前タッチ済み `Vec`（長さ `numel`・全要素 sentinel）を取り出す。Ready なら即時、Pending なら
    /// 期限まで待ち、それ以外は `None`（呼び出し側がインライン確保する）。
    pub(crate) fn acquire(&self, numel: usize) -> Option<Vec<f32>> {
        let deadline = Instant::now() + self.limits.wait_timeout;
        let mut g = lock(&self.shared);
        let mut waited = false;
        loop {
            match g.slots.get_mut(&numel).map(|s| &mut s.state) {
                Some(st @ SlotState::Ready(_)) => {
                    // Ready 判定済みのため else は到達しないが、panic せずインライン確保へ倒す。
                    if let SlotState::Ready(v) = std::mem::replace(st, SlotState::Empty) {
                        g.hits += 1;
                        if waited {
                            g.waits += 1;
                        }
                        return Some(v);
                    }
                    g.misses += 1;
                    return None;
                }
                Some(SlotState::Pending) => {
                    let now = Instant::now();
                    if now >= deadline {
                        g.misses += 1;
                        return None;
                    }
                    waited = true;
                    let (ng, _) = self
                        .shared
                        .cv
                        .wait_timeout(g, deadline - now)
                        .unwrap_or_else(|p| p.into_inner());
                    g = ng;
                }
                _ => {
                    g.misses += 1;
                    return None;
                }
            }
        }
    }

    /// D2H の同期が終わった後に呼ぶ。使用回数を数え、閾値以上かつ上限内のときだけ補充を要求する。
    pub(crate) fn note_used(&self, numel: usize) {
        if !self.eligible(numel) {
            return;
        }
        let mut g = lock(&self.shared);
        if !g.slots.contains_key(&numel) {
            if g.slots.len() >= self.limits.max_entries {
                // 補充中でも保持中でもない（Empty）エントリを 1 つ捨てて場所を空ける。
                let victim = g
                    .slots
                    .iter()
                    .find(|(_, s)| matches!(s.state, SlotState::Empty))
                    .map(|(n, _)| *n);
                match victim {
                    Some(v) => {
                        g.slots.remove(&v);
                    }
                    None => return,
                }
            }
            g.slots.insert(
                numel,
                Slot {
                    seen: 0,
                    state: SlotState::Empty,
                },
            );
        }
        let held = g.held_bytes();
        let Some(slot) = g.slots.get_mut(&numel) else {
            return;
        };
        slot.seen = slot.seen.saturating_add(1);
        if slot.seen < self.limits.repeat_trigger || !matches!(slot.state, SlotState::Empty) {
            return;
        }
        if held + numel * 4 > self.limits.cap_bytes {
            return;
        }
        slot.state = SlotState::Pending;
        let epoch = g.epoch;
        drop(g);
        let sent = self
            .tx
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .send((numel, epoch))
            .is_ok();
        if !sent {
            // worker が不在: Pending のまま残さず Empty へ戻す。
            if let Some(slot) = lock(&self.shared).slots.get_mut(&numel) {
                slot.state = SlotState::Empty;
            }
        }
    }

    /// Ready のバッファを破棄し、解放したバイト数を返す（REQ-14 の明示解放経路）。
    /// Pending（補充中）も Empty へ戻し世代を進めるため、worker が解放後に完了しても Ready は残らない。
    pub(crate) fn release_all(&self) -> u64 {
        let mut g = lock(&self.shared);
        g.epoch = g.epoch.wrapping_add(1);
        let mut freed = 0u64;
        for slot in g.slots.values_mut() {
            if let SlotState::Ready(v) = &slot.state {
                freed += (v.len() * 4) as u64;
            }
            slot.state = SlotState::Empty;
        }
        drop(g);
        // Pending を待っていた acquire を起こす（Empty なので即 miss へ倒れる）。
        self.shared.cv.notify_all();
        freed
    }

    /// `(hits, waits, misses, cached_bytes)`。
    #[cfg(any(test, feature = "internal-diagnostics"))]
    pub(crate) fn stats(&self) -> (u64, u64, u64, u64) {
        let g = lock(&self.shared);
        let cached = g
            .slots
            .values()
            .filter_map(|s| match &s.state {
                SlotState::Ready(v) => Some((v.len() * 4) as u64),
                _ => None,
            })
            .sum();
        (g.hits, g.waits, g.misses, cached)
    }
}

/// 非ゼロ sentinel で埋めて物理ページまで確定した `Vec<f32>`（`calloc` の罠を避ける）。
/// 確保失敗は `None`（abort しない）。
fn alloc_pretouched(numel: usize) -> Option<Vec<f32>> {
    let mut v: Vec<f32> = Vec::new();
    v.try_reserve_exact(numel).ok()?;
    v.resize(numel, <f32 as crate::memory::ReadbackSentinel>::SENTINEL);
    Some(v)
}

fn global_cell() -> &'static OnceLock<Option<Prefetcher>> {
    static P: OnceLock<Option<Prefetcher>> = OnceLock::new();
    &P
}

/// プロセスワイドのプール。opt-in された最初の readback で遅延 spawn する。
/// spawn 失敗なら `None` を保持し、以後は常に従来経路。
pub(crate) fn global() -> Option<&'static Prefetcher> {
    global_cell()
        .get_or_init(|| Prefetcher::new(PREFETCH_LIMITS))
        .as_ref()
}

/// 解放のためだけに worker を起こさない（未生成なら 0）。
pub(crate) fn release_all_if_active() -> u64 {
    global_cell()
        .get()
        .and_then(|p| p.as_ref())
        .map_or(0, |p| p.release_all())
}

/// 診断用の統計（未生成なら 0）。
#[cfg(feature = "internal-diagnostics")]
pub(crate) fn global_stats() -> (u64, u64, u64, u64) {
    global_cell()
        .get()
        .and_then(|p| p.as_ref())
        .map_or((0, 0, 0, 0), |p| p.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Limits {
        Limits {
            min_bytes: 64,
            max_buffer_bytes: 4096,
            cap_bytes: 8192,
            max_entries: 2,
            repeat_trigger: 2,
            wait_timeout: Duration::from_millis(500),
        }
    }

    #[test]
    fn below_min_or_above_max_not_eligible() {
        let p = Prefetcher::new(small()).unwrap();
        assert!(!p.eligible(8));
        assert!(p.eligible(16));
        assert!(p.eligible(1024));
        assert!(!p.eligible(1025));
        p.note_used(8);
        assert!(p.acquire(8).is_none());
    }

    #[test]
    fn no_refill_before_repeat_trigger() {
        let p = Prefetcher::new(small()).unwrap();
        p.note_used(256);
        assert!(p.acquire(256).is_none());
        assert_eq!(p.stats().3, 0);
    }

    #[test]
    fn refill_after_trigger_gives_full_sentinel_vec() {
        let p = Prefetcher::new(small()).unwrap();
        p.note_used(256);
        p.note_used(256);
        let v = p.acquire(256).expect("補充が完了して取り出せる");
        assert_eq!(v.len(), 256);
        let s = <f32 as crate::memory::ReadbackSentinel>::SENTINEL.to_bits();
        assert!(v.iter().all(|x| x.to_bits() == s));
        assert_eq!(p.stats().0, 1);
        // 取り出し後は Empty。再度 note_used で補充される。
        p.note_used(256);
        assert!(p.acquire(256).is_some());
    }

    #[test]
    fn cap_limits_held_bytes() {
        // cap 8192 B。1000 要素（4000 B）は 2 つまでしか保持できない。
        let mut l = small();
        l.max_entries = 4;
        let p = Prefetcher::new(l).unwrap();
        for n in [1000usize, 1001, 1002] {
            p.note_used(n);
            p.note_used(n);
        }
        std::thread::sleep(Duration::from_millis(100));
        let g = lock(&p.shared);
        assert!(g.held_bytes() <= l.cap_bytes, "held={}", g.held_bytes());
    }

    #[test]
    fn entries_limit_is_enforced() {
        let p = Prefetcher::new(small()).unwrap();
        for n in [100usize, 200, 300] {
            p.note_used(n);
            p.note_used(n);
        }
        std::thread::sleep(Duration::from_millis(100));
        assert!(lock(&p.shared).slots.len() <= 2);
    }

    #[test]
    fn release_all_returns_bytes_and_zeroes_cache() {
        let p = Prefetcher::new(small()).unwrap();
        p.note_used(256);
        p.note_used(256);
        // Ready になるまで待つ（acquire は取り出してしまうため stats をポーリングする）。
        let t0 = Instant::now();
        while p.stats().3 == 0 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(p.stats().3, 1024);
        assert_eq!(p.release_all(), 1024);
        assert_eq!(p.stats().3, 0);
    }

    #[test]
    fn release_during_pending_discards_late_refill() {
        let p = Prefetcher::new(small()).unwrap();
        // worker を介さず、世代 0 の Pending を作る。
        lock(&p.shared).slots.insert(
            256,
            Slot {
                seen: 2,
                state: SlotState::Pending,
            },
        );
        assert_eq!(p.release_all(), 0);
        // 解放前の世代 0 の要求が解放後に完了する状況を worker へ直接再現する。
        p.tx.lock().unwrap().send((256, 0)).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(p.stats().3, 0, "解放後に Ready が再登録されてはならない");
        assert!(matches!(
            lock(&p.shared).slots[&256].state,
            SlotState::Empty
        ));
        // 現世代の補充は従来どおり成立する。
        p.note_used(256);
        assert!(p.acquire(256).is_some());
    }

    #[test]
    fn poisoned_mutex_recovers() {
        let p = Prefetcher::new(small()).unwrap();
        let sh = Arc::clone(&p.shared);
        let _ = std::thread::spawn(move || {
            let _g = sh.state.lock().unwrap();
            panic!("poison");
        })
        .join();
        p.note_used(256);
        p.note_used(256);
        assert!(p.acquire(256).is_some());
    }

    #[test]
    fn oom_alloc_returns_none_without_abort() {
        assert!(alloc_pretouched(usize::MAX / 8).is_none());
    }

    #[test]
    fn pending_timeout_falls_back_to_none() {
        let mut l = small();
        l.wait_timeout = Duration::from_millis(1);
        let p = Prefetcher::new(l).unwrap();
        // worker を介さず Pending を直接作り、期限切れで None になることを確認する。
        lock(&p.shared).slots.insert(
            256,
            Slot {
                seen: 2,
                state: SlotState::Pending,
            },
        );
        assert!(p.acquire(256).is_none());
        assert_eq!(p.stats().2, 1);
    }
}
