//! readback 宛先ポリシー（イシュー #2112）。`MetalBuffer::read_to_vec` が返す
//! ホスト `Vec<f32>` を「単一スレッドの `to_vec()`」で作るか「分割並列コピー」で作るかを
//! 選ぶ純ロジックと、その env 切替をまとめる。
//!
//! # 位置づけ
//! - 呼び出し元: `buffer.rs::MetalBuffer::read_to_vec`（macOS 限定）。gemm.rs・memory.rs
//!   （`download_inner`）・各 op が共通に通る readback の 1 箇所に集約する。
//! - 本モジュールは `objc2` 系 FFI に触れず `unsafe` も持たないため、`pad`／`batch_state`
//!   と同じ判断で cfg 非依存とし、Linux（CI）でも単体テストが回る。
//!
//! # 根拠と期待値
//! `docs/perf/metal-readout-legacy-regression-four-arm-diag.md`（#1696）は、keep-alive の
//! 宛先 `Vec` が first-touch ページのとき readback が伸びることを観測した（M1。機構の
//! 帰属は仮説段階）。本ポリシーの `ParallelChunked` は「first-touch fault 処理と memcpy を
//! 複数コアへ分散する」派生仮説 H-par に基づく opt-in であり、#1696 が検証済みの結論では
//! ない。legacy 既定の判定セルで見込める改善は小さい
//! （`docs/perf/metal-reuse-readback-2112.md`）。
//!
//! # 既定 OFF・数値契約
//! 既定は `Fresh`（`src.to_vec()` と完全に同一）。`ParallelChunked` も全要素を上書きする
//! ため出力は bit 同一で、tolerance・REQ-2 判定には影響しない。宛先の再利用はしない
//! （古い内容が露出する経路を作らない）。

// macOS 以外ではテスト以外から呼ばれない（`buffer.rs` が macOS 限定のため）。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::{Mutex, OnceLock};

/// readback 宛先ポリシーを選ぶ env 変数名。許容値は `fresh`／`parallel`（完全一致）のみ。
pub(crate) const READBACK_DEST_ENV: &str = "FANDHE_AI_METAL_READBACK_DEST";

/// 宛先の作り方。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadbackDest {
    /// 単一スレッドの `to_vec()`（従来挙動）。
    Fresh,
    /// しきい値以上のとき、fresh 宛先を chunk に分けスコープ付きスレッドで並列コピーする。
    ParallelChunked,
}

/// 既定ポリシー（既定 OFF）。ADOPT 判定後の結線 PR でのみ変更する（RULE.txt）。
pub(crate) const READBACK_DEST_DEFAULT: ReadbackDest = ReadbackDest::Fresh;

/// 並列コピーの対象にする最小バイト数（8 MiB。N=1024 の f32 4 MiB は対象外、N=2048 以上）。
pub(crate) const PARALLEL_READBACK_MIN_BYTES: usize = 8 * 1024 * 1024;

/// 並列コピーの最大スレッド数（`available_parallelism` との min をとる）。
pub(crate) const PARALLEL_READBACK_MAX_THREADS: usize = 8;

/// env 値の解釈。完全一致の allowlist のみ受理し、未知値は `None`（呼び出し側が既定へ倒す）。
/// 値はログ・エラーへエコーしない。
pub(crate) fn parse_env_value(raw: &str) -> Option<ReadbackDest> {
    match raw {
        "fresh" => Some(ReadbackDest::Fresh),
        "parallel" => Some(ReadbackDest::ParallelChunked),
        _ => None,
    }
}

fn env_mode() -> ReadbackDest {
    static MODE: OnceLock<ReadbackDest> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var(READBACK_DEST_ENV)
            .ok()
            .and_then(|v| parse_env_value(&v))
            .unwrap_or(READBACK_DEST_DEFAULT)
    })
}

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<ReadbackDest>> =
        const { std::cell::Cell::new(None) };
}

/// テスト用 scoped override（thread-local・RAII。drop で直前値へ復元する）。
#[cfg(test)]
pub(crate) struct DestOverrideGuard {
    prev: Option<ReadbackDest>,
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(test)]
impl DestOverrideGuard {
    pub(crate) fn new(dest: ReadbackDest) -> Self {
        let prev = TEST_OVERRIDE.with(|c| c.replace(Some(dest)));
        Self {
            prev,
            _not_send: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for DestOverrideGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// 現在有効なポリシー（テスト override > env > 既定）。
pub(crate) fn current_dest() -> ReadbackDest {
    #[cfg(test)]
    if let Some(d) = TEST_OVERRIDE.with(|c| c.get()) {
        return d;
    }
    env_mode()
}

/// `n` 要素を `threads` 個へ重なり・隙間なく分ける `[start, end)` 列を返す。
/// `n == 0` は空列。`threads == 0` は 1 として扱う。空 chunk は返さない。
pub(crate) fn chunk_plan(n: usize, threads: usize) -> Vec<(usize, usize)> {
    if n == 0 {
        return Vec::new();
    }
    let t = threads.max(1).min(n);
    let base = n / t;
    let rem = n % t;
    let mut out = Vec::with_capacity(t);
    let mut start = 0;
    for i in 0..t {
        let len = base + usize::from(i < rem);
        out.push((start, start + len));
        start += len;
    }
    out
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(PARALLEL_READBACK_MAX_THREADS)
}

/// 分割並列コピー本体。`min_bytes` 未満・スレッド 1 以下なら `to_vec()`。
/// spawn 失敗の chunk は呼び出し元スレッドで逐次コピーする（`unwrap`／`expect` なし）。
fn parallel_copy(src: &[f32], min_bytes: usize, threads: usize) -> Vec<f32> {
    let n = src.len();
    if n.saturating_mul(std::mem::size_of::<f32>()) < min_bytes || threads <= 1 {
        return src.to_vec();
    }
    // calloc（大きいサイズではゼロページ）。全要素を下で上書きするため古い値も残らない。
    let mut dst = vec![0.0f32; n];
    let plan = chunk_plan(n, threads);
    {
        // 各 chunk の可変借用を Mutex<Option<..>> に包んで scope の外へ置く。spawn 失敗時は
        // closure が捨てられるため、借用をここから取り戻して呼び出し元で書く。
        let mut cells: Vec<Mutex<Option<&mut [f32]>>> = Vec::with_capacity(plan.len());
        let mut rest: &mut [f32] = &mut dst;
        let mut consumed = 0;
        for &(_, end) in &plan {
            let (head, tail) = rest.split_at_mut(end - consumed);
            rest = tail;
            consumed = end;
            cells.push(Mutex::new(Some(head)));
        }
        std::thread::scope(|scope| {
            for (cell, &(start, end)) in cells.iter().zip(&plan) {
                let s = &src[start..end];
                let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                    if let Some(h) = cell.lock().ok().and_then(|mut g| g.take()) {
                        h.copy_from_slice(s);
                    }
                });
                if spawned.is_err()
                    && let Some(h) = cell.lock().ok().and_then(|mut g| g.take())
                {
                    h.copy_from_slice(s);
                }
            }
        });
    }
    dst
}

/// ポリシーに従い `src` を新規 `Vec<f32>` へコピーする（`read_to_vec` の実体）。
pub(crate) fn copy_to_vec(src: &[f32]) -> Vec<f32> {
    copy_to_vec_with(current_dest(), src)
}

/// ポリシーを明示して `copy_to_vec` する（テスト・診断用に分離）。
pub(crate) fn copy_to_vec_with(dest: ReadbackDest, src: &[f32]) -> Vec<f32> {
    match dest {
        ReadbackDest::Fresh => src.to_vec(),
        ReadbackDest::ParallelChunked => {
            parallel_copy(src, PARALLEL_READBACK_MIN_BYTES, default_threads())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| f32::from_bits((i as u32).wrapping_mul(2654435761) ^ 0x5bd1e995))
            .collect();
        let specials = [
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
        ];
        for (i, s) in specials.iter().enumerate() {
            if i < n {
                v[i] = *s;
            }
        }
        v
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn env_values_are_exact_allowlist() {
        assert_eq!(parse_env_value("fresh"), Some(ReadbackDest::Fresh));
        assert_eq!(
            parse_env_value("parallel"),
            Some(ReadbackDest::ParallelChunked)
        );
        for bad in ["", "Fresh", "PARALLEL", " parallel", "parallel ", "x", "1"] {
            assert_eq!(parse_env_value(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn default_is_fresh() {
        assert_eq!(READBACK_DEST_DEFAULT, ReadbackDest::Fresh);
    }

    #[test]
    fn chunk_plan_covers_without_overlap() {
        for n in [0usize, 1, 2, 7, 8, 9, 1000, 1_048_577] {
            for t in [0usize, 1, 2, 3, 8, 64] {
                let plan = chunk_plan(n, t);
                let mut pos = 0;
                for &(s, e) in &plan {
                    assert_eq!(s, pos);
                    assert!(e > s);
                    pos = e;
                }
                assert_eq!(pos, n, "n={n} t={t}");
                assert!(plan.len() <= t.max(1));
            }
        }
    }

    #[test]
    fn parallel_copy_is_bit_identical_around_threshold() {
        let min_bytes = 4096;
        let min_elems = min_bytes / 4;
        for n in [0, 1, 5, min_elems - 1, min_elems, min_elems + 1, 100_003] {
            let src = sample(n);
            for threads in [1, 2, 3, 8] {
                let out = parallel_copy(&src, min_bytes, threads);
                assert_eq!(bits(&out), bits(&src), "n={n} threads={threads}");
            }
        }
    }

    #[test]
    fn copy_to_vec_with_matches_to_vec() {
        let src = sample(3_000_001);
        for d in [ReadbackDest::Fresh, ReadbackDest::ParallelChunked] {
            assert_eq!(bits(&copy_to_vec_with(d, &src)), bits(&src));
        }
    }

    #[test]
    fn scoped_override_nests_and_restores() {
        let base = current_dest();
        {
            let _a = DestOverrideGuard::new(ReadbackDest::ParallelChunked);
            assert_eq!(current_dest(), ReadbackDest::ParallelChunked);
            {
                let _b = DestOverrideGuard::new(ReadbackDest::Fresh);
                assert_eq!(current_dest(), ReadbackDest::Fresh);
            }
            assert_eq!(current_dest(), ReadbackDest::ParallelChunked);
        }
        assert_eq!(current_dest(), base);
    }

    /// macOS 実機: 実 `MetalBuffer` の readback が Fresh／ParallelChunked で bit 同一
    /// （N=1024・2048・4096 相当・非整列 numel を含む）。
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_buffer_read_to_vec_bit_identical_across_policies() {
        use crate::buffer::MetalBuffer;
        use crate::context::MetalContext;
        let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
        for n in [1024usize * 1024, 2048 * 2048, 4096 * 4096, 3_000_001] {
            let data = sample(n);
            let buf = MetalBuffer::new_with_data(&ctx, &data).expect("確保に失敗した");
            let fresh = {
                let _g = DestOverrideGuard::new(ReadbackDest::Fresh);
                buf.read_to_vec()
            };
            let par = {
                let _g = DestOverrideGuard::new(ReadbackDest::ParallelChunked);
                buf.read_to_vec()
            };
            assert_eq!(bits(&fresh), bits(&data), "fresh n={n}");
            assert_eq!(bits(&par), bits(&data), "parallel n={n}");
        }
    }
}
