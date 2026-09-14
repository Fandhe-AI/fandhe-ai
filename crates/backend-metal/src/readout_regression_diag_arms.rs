//! `readout_regression_diag_tests_1695.rs`（Metal 実機依存の診断テスト
//! 本体）が使う 4 腕の定義・純関数ヘルパ（イシュー #1695。CUDA 側
//! `crates/backend-cuda/src/readout_regression_diag_tests_1436.rs`
//! （イシュー #1436）と同型の腕構成を Metal へ移植する）。
//!
//! # `objc2` 系 FFI に触れない理由（`gather_scatter_model`／`soft_f64`
//! と同じ配置判断）
//!
//! 本モジュールが定義する型・関数はいずれも `Vec<f32>` 等の純粋な
//! ホスト側データのみを扱い、Metal デバイス・バッファへは一切触れない
//! （実機呼び出しは `readout_regression_diag_tests_1695.rs` 側が担う）。
//! そのため `cfg(target_os = "macos")` を付けず、Linux（本実装環境・CI）
//! でも単体テストが回る。`pub` にしない理由: crates.io 公開クレートの
//! 公開 API 面へ診断専用の型を増やさないため（`compat-api-scope.md`
//! §0「`facade` が唯一のサポートされる公開 API 面」）。`#[cfg(test)]`
//! にする理由: 非 test ビルドで `dead_code` にならないようにするため
//! （本モジュールの全項目は Linux 側 `mod tests` と macOS 側の診断
//! テスト本体からのみ参照される）。
//!
//! # 4 腕の定義（#1436 からの移植・Metal 向け再解釈）
//!
//! Metal は UMA（統合メモリ）のため CUDA の「D2H（`clone_dtoh`／
//! `memcpy_dtoh`）」に相当する明示転送は存在せず、`MetalBuffer::
//! read_to_vec`（`contents()` からの memcpy。`buffer.rs` 参照）が
//! その対応物になる（`docs/perf/metal-gemm-reuse-phase-breakdown.md`
//! の `readback` 区間定義と同じ整理）。
//!
//! - `LegacyToVec`（`bench-fandhe` legacy readout の再現）: `read_to_vec`
//!   で 1 本目を受け取り、`to_vec()` で 2 本目を確保して読み出し、
//!   2 本目を drop する。1 本目は `keep_alive` で保持する
//! - `BorrowedKeepAlive`（`bench-fandhe` borrowed readout の再現）:
//!   `read_to_vec` が返す `Vec` を直接読み出し、追加確保・free なしで
//!   `keep_alive` へ積む
//! - `BorrowedWithDummyAllocFree`: `BorrowedKeepAlive` と同じ読み出しに
//!   加え、読み出し後に同サイズ `Vec` を確保・全要素へ書き込み・即 drop
//!   する（「2 本目のコピー」と「アロケータへの free 副作用」を分離する
//!   対照腕）
//! - `PretouchedReusedDest`: 計測外で 1 回だけ確保し全要素を明示書き込み
//!   （事前タッチ）した宛先 `Vec` へ `MetalBuffer::read_into_slice`
//!   （`buffer.rs` に本イシューで追加した `#[cfg(test)]` 限定ヘルパ。
//!   `read_to_vec` と異なり確保を伴わない memcpy）し、読み出し後にその
//!   内容を `keep_alive` 用の別 `Vec` へ copy-out する
//!
//! CUDA 版の 5 腕目 `PretouchedFreshDest`（CUDA 本番の是正
//! `ReadbackDest::PretouchedFresh`〈#1437〉に対応する腕）は、Metal 側に
//! 対応する本番機構が存在しないため本イシューでは移植しない（意図的な
//! スコープ判断。イシュー本文の 4 腕限定に従う）。
//!
//! # 機構仮説は中立表現に留める
//!
//! CUDA 側モジュール冒頭コメントの機構説明（glibc の `M_MMAP_THRESHOLD`
//! 動的適応・`MALLOC_MMAP_THRESHOLD_` 確証実験）は glibc 固有であり、
//! M4 Max（macOS・libmalloc）にそのまま適用できる保証はない。本モジュール
//! およびそれを使う診断テストの doc comment は「keep-alive により
//! `read_to_vec` の宛先が毎回 first-touch ページになるか、legacy の
//! free-and-reuse で既タッチページが再利用されるか」という中立な仮説に
//! 留め、機構確認・アロケータ環境変数実験は実機実測（イシュー #1696）へ
//! 委ねる。

/// 読み出し方式 4 腕（本ファイル冒頭「4 腕の定義」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadoutArm {
    LegacyToVec,
    BorrowedKeepAlive,
    BorrowedWithDummyAllocFree,
    PretouchedReusedDest,
}

impl ReadoutArm {
    pub(crate) const ALL: [ReadoutArm; 4] = [
        ReadoutArm::LegacyToVec,
        ReadoutArm::BorrowedKeepAlive,
        ReadoutArm::BorrowedWithDummyAllocFree,
        ReadoutArm::PretouchedReusedDest,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            ReadoutArm::LegacyToVec => "LegacyToVec",
            ReadoutArm::BorrowedKeepAlive => "BorrowedKeepAlive",
            ReadoutArm::BorrowedWithDummyAllocFree => "BorrowedWithDummyAllocFree",
            ReadoutArm::PretouchedReusedDest => "PretouchedReusedDest",
        }
    }
}

/// `f64` 逐次和（CUDA 側 `checksum_f64` と同一。腕間・参照値との sanity
/// 比較にのみ使い、REQ-2 判定契約とは無関係）。
pub(crate) fn checksum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum()
}

/// `PretouchedReusedDest` 腕専用の非ゼロ sentinel（`vec![0.0; n]` 単独は
/// glibc/libmalloc の calloc 経由確保でゼロページ（COW）を返しうるため
/// 実ページがコミットされない懸念がある。全要素を明示的に非ゼロへ
/// 書き込むことで実際のページフォールト・コミットを発生させる。CUDA 側
/// `run_size_arm` の `pretouched_dest` 初期化と同じ判断）。
pub(crate) const PRETOUCH_SENTINEL: f32 = 1.0;

/// 事前タッチ済みのホスト `Vec<f32>` を確保する（`PretouchedReusedDest`
/// 腕の宛先。呼び出し元がループ外で 1 回だけ呼ぶ）。
pub(crate) fn pretouched_host_vec_f32(numel: usize) -> Vec<f32> {
    vec![PRETOUCH_SENTINEL; numel]
}

/// `BorrowedWithDummyAllocFree` 腕専用: 同サイズの `Vec` を確保し全要素へ
/// 非ゼロ値を書き込んでから即 drop する（アロケータへ free を発生させる
/// 副作用のみを追加した対照腕。CUDA 側 `BorrowedWithDummyAllocFree` 分岐
/// と同型）。
///
/// 書き込み結果を一度も読み出さない dead store はコンパイラ最適化で除去
/// されうるため、確保・書き込みの両方を [`std::hint::black_box`] で観測
/// 境界に包み除去を防ぐ（CUDA 側 #1442 レビュー指摘対応を踏襲）。
pub(crate) fn dummy_alloc_touch_free(numel: usize) {
    let mut dummy: Vec<f32> = std::hint::black_box(vec![0.0f32; numel]);
    for (i, x) in dummy.iter_mut().enumerate() {
        *x = std::hint::black_box(i as f32 + 1.0);
    }
    std::hint::black_box(&dummy);
    drop(dummy);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_labels_are_distinct() {
        let labels: Vec<&str> = ReadoutArm::ALL.iter().map(|a| a.label()).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            labels.len(),
            sorted.len(),
            "arm labels must be pairwise distinct for log parsing"
        );
    }

    #[test]
    fn all_has_exactly_four_arms() {
        assert_eq!(ReadoutArm::ALL.len(), 4);
    }

    #[test]
    fn checksum_f64_matches_manual_sum_for_small_vector() {
        let v = vec![1.0f32, 2.0f32, 3.5f32];
        assert!((checksum_f64(&v) - 6.5).abs() < 1e-9);
    }

    #[test]
    fn checksum_f64_empty_is_zero() {
        assert_eq!(checksum_f64(&[]), 0.0);
    }

    #[test]
    fn pretouched_host_vec_f32_has_requested_len_and_nonzero_sentinel() {
        let v = pretouched_host_vec_f32(1024);
        assert_eq!(v.len(), 1024);
        assert!(v.iter().all(|&x| x == PRETOUCH_SENTINEL));
        assert_ne!(PRETOUCH_SENTINEL, 0.0);
    }

    #[test]
    fn pretouched_host_vec_f32_zero_len_is_empty() {
        let v = pretouched_host_vec_f32(0);
        assert!(v.is_empty());
    }

    #[test]
    fn dummy_alloc_touch_free_does_not_panic_for_typical_sizes() {
        dummy_alloc_touch_free(1024 * 1024);
    }

    #[test]
    fn dummy_alloc_touch_free_does_not_panic_for_zero_len() {
        dummy_alloc_touch_free(0);
    }
}
