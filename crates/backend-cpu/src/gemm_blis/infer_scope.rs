//! 推論経路の中だけ `jobs_per_worker = 1`（jpw1）を効かせる opt-in ゲート（イシュー #4161）。
//!
//! 役割: `facade` の推論本体（`Sequential::predict` 系・`predict_resident`・
//! `predict_resident_input`）が [`enter_inference_scope`] でスコープに入っている間だけ、
//! `small_m::jobs_per_worker_for` が仕事量の小さい形状の `jobs_per_worker` を 1 へ下げる。
//! 学習（tape の forward／backward・`train_step_resident`）はスコープの外なので機構が効かない。
//! 背景は #3177 の単体 A/B（GB10 で jpw1 が一部形状を 0.82〜0.94 に縮める一方、正方 512 は悪化）と、
//! 全経路適用が train セルの後退で REJECT された #3201。
//!
//! ゲート `INFER_SCOPE_JPW1_ENABLED` は aarch64 Linux（GB10 で実測。#4161 → #4266 ADOPT・
//! 記録 PR #4269）でだけ既定 ON、ほかは既定 OFF（実測範囲外へ広げない。所有者判断 2026-11-12・
//! #4165）。OFF のとき挙動は現行と同一（常に既定の `TWO_D_JOBS_PER_WORKER`）。
//!
//! bit 一致の根拠は `small_m.rs` と同じ。jpw は job の大きさだけを決め、各 C 要素の FMA 連鎖は
//! 変わらない。tolerance と FMA 契約は不変。A の pack 共有（#3199 の G1）は含めない。
//!
//! 制約: スコープ判定は thread-local のため、呼び出し元スレッドで読む必要がある。GEMM 入口
//! （`mod.rs`）は affinity／cap のプールへ入る前に jpw を決める。rayon ワーカー上から呼ばれる
//! GEMM はスコープを継がず、既定の jpw（安全側）になる。

use std::cell::Cell;
use std::marker::PhantomData;

/// 推論スコープ内 jpw1 の既定有効フラグ。aarch64 Linux だけ ON、ほかは OFF（#4165）。
/// #4266 の A/B after 腕はこの定数を true にした状態で、GB10 上の既定と同じ挙動になる。
pub(super) const INFER_SCOPE_JPW1_ENABLED: bool =
    cfg!(all(target_arch = "aarch64", target_os = "linux"));

thread_local! {
    static SCOPE_DEPTH: Cell<u32> = const { Cell::new(0) };
    static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static APPLIED: Cell<u64> = const { Cell::new(0) };
}

/// 推論スコープの RAII ガード（`!Send`。drop で深さを戻す）。
#[must_use = "ガードを drop するとスコープを抜ける"]
pub struct InferenceScopeGuard {
    _not_send: PhantomData<*const ()>,
}

impl Drop for InferenceScopeGuard {
    fn drop(&mut self) {
        // TLS 破棄後の drop でも panic しない。
        let _ = SCOPE_DEPTH.try_with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// 現スレッドで推論スコープに入る（ネスト可）。`facade` の推論本体の先頭から呼ばれる。
#[doc(hidden)]
pub fn enter_inference_scope() -> InferenceScopeGuard {
    let _ = SCOPE_DEPTH.try_with(|d| d.set(d.get().saturating_add(1)));
    InferenceScopeGuard {
        _not_send: PhantomData,
    }
}

/// スコープ限定の有効／無効の上書きを戻す RAII ガード（`!Send`）。
#[must_use = "ガードを drop すると上書きが元に戻る"]
pub struct InferScopeJpw1OverrideGuard {
    prev: Option<bool>,
    _not_send: PhantomData<*const ()>,
}

impl Drop for InferScopeJpw1OverrideGuard {
    fn drop(&mut self) {
        let _ = OVERRIDE.try_with(|c| c.set(self.prev));
    }
}

/// 現スレッドのゲートをスコープ限定で上書きする（テスト・診断用。ネスト可）。
#[doc(hidden)]
pub fn override_infer_scope_jpw1_for_scope(enabled: bool) -> InferScopeJpw1OverrideGuard {
    let prev = OVERRIDE
        .try_with(|c| c.replace(Some(enabled)))
        .unwrap_or(None);
    InferScopeJpw1OverrideGuard {
        prev,
        _not_send: PhantomData,
    }
}

/// 現スレッドが推論スコープ内か。
pub(super) fn in_inference_scope() -> bool {
    SCOPE_DEPTH.try_with(|d| d.get() > 0).unwrap_or(false)
}

/// 上書きがあればその値、なければ定数。TLS 破棄後は定数へ倒す。
pub(super) fn infer_jpw1_effective() -> bool {
    OVERRIDE
        .try_with(|c| c.get())
        .ok()
        .flatten()
        .unwrap_or(INFER_SCOPE_JPW1_ENABLED)
}

/// jpw を 1 へ下げたことを数える（値が下がったときだけ呼ぶ）。
pub(super) fn note_applied() {
    let _ = APPLIED.try_with(|a| a.set(a.get().saturating_add(1)));
}

/// 診断用レポート（A/B の機構発火の前提確認に使う。facade では再公開しない）。
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferScopeJpw1Report {
    /// 定数 [`INFER_SCOPE_JPW1_ENABLED`] の値（上書きは含まない）。
    pub gate_default: bool,
    /// 現スレッドで jpw を 1 に下げた回数。
    pub applied: u64,
}

/// 現スレッドのレポートを返す。
#[doc(hidden)]
pub fn infer_scope_jpw1_report() -> InferScopeJpw1Report {
    InferScopeJpw1Report {
        gate_default: INFER_SCOPE_JPW1_ENABLED,
        applied: APPLIED.try_with(|a| a.get()).unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_default_matches_platform() {
        // 実行時 assert（#3201 の教訓: 反転ツリーでもテストをコンパイルできるようにする）。
        let expected = cfg!(all(target_arch = "aarch64", target_os = "linux"));
        assert_eq!(std::hint::black_box(INFER_SCOPE_JPW1_ENABLED), expected);
    }

    #[test]
    fn nested_scope_depth_restores() {
        assert!(!in_inference_scope());
        let g1 = enter_inference_scope();
        let g2 = enter_inference_scope();
        assert!(in_inference_scope());
        drop(g2);
        assert!(in_inference_scope());
        drop(g1);
        assert!(!in_inference_scope());
    }

    #[test]
    fn override_nests_and_restores() {
        assert!(!infer_jpw1_effective());
        let a = override_infer_scope_jpw1_for_scope(true);
        assert!(infer_jpw1_effective());
        {
            let _b = override_infer_scope_jpw1_for_scope(false);
            assert!(!infer_jpw1_effective());
        }
        assert!(infer_jpw1_effective());
        drop(a);
        assert!(!infer_jpw1_effective());
    }
}
