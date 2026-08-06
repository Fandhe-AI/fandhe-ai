//! マイクロカーネル契約と ISA ごとの実装への配線。
//!
//! [`super`] の 5-loop ドライバから呼ばれる。契約は共通:
//! packed A（`ap`、`MR * kc_len` 要素、p-major）・packed B（`bp`、
//! `kc_len * NR` 要素、p-major）から MR×NR の C タイル（`c_tile`、
//! row-major・ld=NR）へ `kc_len` ぶんの寄与を p 昇順の `mul_add`（または
//! 対応する SIMD FMA）で加算する。C タイルの現在値ロード・書き戻しは
//! 呼び出し元（ドライバ）の責務であり、本モジュール以下の関数は
//! `c_tile` の中身のみを扱う。
//!
//! ## ISA 選択（TASK-1.6f のスコープ: コンパイル時 cfg のみ）
//!
//! - aarch64: NEON は常時有効の baseline ISA のため無条件で [`neon`] を選ぶ
//! - x86_64: コンパイル時に `target_feature = "avx2"` かつ `"fma"` が
//!   有効（`RUSTFLAGS="-C target-feature=+avx2,+fma"` 等）な場合のみ
//!   [`avx2`] を選び、それ以外は [`scalar`] にフォールバックする
//! - 実行時 CPU 機能検出（`is_x86_feature_detected!`）による本番経路の
//!   ディスパッチは #185（TASK-1.6g）のスコープ。本モジュールはテスト
//!   限定で `avx2::kernel_unchecked` を実行時検出ガード付きに直接呼ぶ
//!   ことで、既定ビルド（RUSTFLAGS なし）の CI でもカーネル本体を
//!   検証できるようにする（`avx2` モジュールのテスト参照）。
//!
//! `avx2` モジュールは `cfg(target_arch = "x86_64")` のみでコンパイルし
//! `target_feature` ではゲートしない（レビュー指摘: モジュール単位で
//! ゲートすると既定ビルドで本体が一切コンパイルされず上記のテスト限定
//! 直接検証が不可能になるため。`avx2` モジュールドキュメント参照）。

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

// デフォルト経路の選択（コンパイル時 cfg。この 3 分岐は互いに排他的かつ
// 網羅的: aarch64 → neon、x86_64 で avx2+fma がコンパイル時有効 → avx2、
// それ以外（x86_64 で avx2+fma 無効・その他 arch）→ scalar）。

#[cfg(target_arch = "aarch64")]
pub use neon::{MR, NR, kernel};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
pub use avx2::{MR, NR, kernel};

#[cfg(not(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )
)))]
pub use scalar::{MR, NR, kernel};
