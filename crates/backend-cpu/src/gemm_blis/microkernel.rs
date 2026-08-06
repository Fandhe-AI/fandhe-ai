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
//! ## ISA 選択（#185・TASK-1.6g で実行時ディスパッチへ移行）
//!
//! TASK-1.6f（#184）まではコンパイル時 cfg のみで ISA を固定していたが、
//! それでは x86_64 の既定ビルド（`RUSTFLAGS` 指定なし）で実行 CPU が
//! AVX2/AVX-512 を持っていてもスカラーへ落ちてしまい REQ-8 の CPU 性能
//! 下限（対 PyTorch CPU 比 20%）達成を妨げる。本モジュールは faer の
//! `pulp` 方式（検出済みトークン型による dispatch）を依存追加なしで
//! 自前実装し、[`Isa::detect`] による実行時 CPU 機能検出でマイクロ
//! カーネルを選択する:
//!
//! - aarch64: NEON は常時有効の baseline ISA のため無条件で [`neon`] を選ぶ
//!   （実行時検出は不要。[`Isa::detect`] は常に `Isa::Neon` を返す）
//! - x86_64: [`Isa::detect`] が `is_x86_feature_detected!("avx512f")` →
//!   `"avx2"` かつ `"fma"` → 非対応の順に検出し、対応する [`Avx512Kernel`] /
//!   [`Avx2Kernel`] / [`ScalarKernel`] トークンを選ぶ
//! - その他 arch: 常に [`ScalarKernel`]
//!
//! ### 健全性契約（トークン型による dispatch）
//!
//! `Avx2Kernel`／`Avx512Kernel` は生成経路を検出済みの場合に限定する
//! （`try_new` が検出成功時のみ `Some` を返す非公開コンストラクタ）。
//! トークンのインスタンスが存在すること自体が「実行 CPU が当該 ISA を
//! サポートする」証明となり、[`Microkernel::run`] 内部の
//! `unsafe { kernel_unchecked(...) }` 呼び出しの SAFETY 根拠になる
//! （Safety 契約の履行責務をコンストラクタに集約することで、`run` 呼び
//! 出し側は `unsafe` を意識せず安全に dispatch できる）。
//!
//! 環境変数等による dispatch 上書き機構は設けない（外部入力が `unsafe`
//! カーネル選択を制御できると SIGILL・未定義動作の攻撃面になるため。
//! OWASP A03・`.claude/rules/security.md`）。
//!
//! ### 公開 API 非破壊
//!
//! TASK-1.6f で公開していたコンパイル時 cfg 選択の `pub use
//! {neon,avx2,scalar}::{MR, NR, kernel}` はそのまま残す（既存呼び出し元
//! 互換のため）。ただし [`super::gemm_blis`]／[`super::gemm_blis_parallel`]
//! の駆動経路は本モジュールの実行時ディスパッチ（[`Isa::detect`] 経由）
//! へ切り替わっており、この `pub use` 経路は駆動経路から外れている点に
//! 注意する。
//!
//! `avx2`／`avx512` モジュールは `cfg(target_arch = "x86_64")` のみで
//! コンパイルし `target_feature` ではゲートしない（レビュー指摘: モジュール
//! 単位でゲートすると既定ビルドで本体が一切コンパイルされず、テスト
//! 限定の実行時検出ガード付き直接検証が不可能になるため）。

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

use std::sync::OnceLock;

// デフォルト経路の選択（コンパイル時 cfg。公開 API 非破壊のため残すが、
// [`gemm_blis`]／[`gemm_blis_parallel`] の駆動経路は本モジュールの
// 実行時ディスパッチへ切り替わっている。モジュールドキュメント参照）。

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

/// 実行時 ISA ディスパッチが選択可能なマイクロカーネルの共通契約。
///
/// [`super::gemm_blis_region`] はこの trait をジェネリック境界に取り、
/// `K::MR`／`K::NR` でタイル形状を、[`Microkernel::run`] で累積計算を
/// 行う。各実装（[`ScalarKernel`]／[`NeonKernel`]／[`Avx2Kernel`]／
/// [`Avx512Kernel`]）は `Copy + Sync` な ZST（サイズ 0 の型）であり、
/// rayon のクロージャへそのまま値渡しできる。
pub trait Microkernel: Copy + Sync {
    /// マイクロカーネルタイルの行数。
    const MR: usize;
    /// マイクロカーネルタイルの列数。
    const NR: usize;

    /// `ap`（packed A）・`bp`（packed B）から `c_tile`（MR×NR、row-major・
    /// ld=NR）へ `kc_len` ぶんの寄与を加算する。安全な呼び出し専用の入口
    /// であり、内部の `unsafe`（intrinsics 呼び出し）はトークンの生成
    /// 経路（検出済みの場合のみ構築可能）によって健全性が保証される。
    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize);
}

/// 全 arch 共通のスカラーフォールバックトークン。検出不要のため
/// `ScalarKernel` は常に構築可能（`Default` 相当の unit struct）。
#[derive(Clone, Copy)]
pub struct ScalarKernel;

impl Microkernel for ScalarKernel {
    const MR: usize = scalar::MR;
    const NR: usize = scalar::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        scalar::kernel(ap, bp, c_tile, kc_len);
    }
}

/// aarch64 NEON トークン。NEON は baseline ISA のため実行時検出不要
/// （[`neon`] モジュールドキュメント参照）。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl Microkernel for NeonKernel {
    const MR: usize = neon::MR;
    const NR: usize = neon::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        neon::kernel(ap, bp, c_tile, kc_len);
    }
}

/// x86_64 AVX2+FMA トークン。[`Avx2Kernel::try_new`] 経由でのみ構築でき、
/// これが実行 CPU の AVX2+FMA 対応を保証する（[`Microkernel::run`] 内部の
/// `unsafe { avx2::kernel_unchecked(...) }` の SAFETY 根拠）。
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Avx2Kernel {
    /// 外部からの直接構築を禁止する非公開フィールド（[`Avx2Kernel::try_new`]
    /// 経由の検出済み構築のみを許可するための封止）。
    _private: (),
}

#[cfg(target_arch = "x86_64")]
impl Avx2Kernel {
    /// 実行 CPU が AVX2+FMA をサポートする場合のみ `Some` を返す。この
    /// 判定こそが [`Microkernel::run`] 内 `unsafe` 呼び出しの安全根拠。
    pub(crate) fn try_new() -> Option<Self> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Some(Self { _private: () })
        } else {
            None
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Microkernel for Avx2Kernel {
    const MR: usize = avx2::MR;
    const NR: usize = avx2::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx2") && ("fma") を確認済み
        // （avx2::kernel_unchecked の `# Safety` 契約を満たす）。
        unsafe { avx2::kernel_unchecked(ap, bp, c_tile, kc_len) }
    }
}

/// x86_64 AVX-512F トークン。[`Avx512Kernel::try_new`] 経由でのみ構築でき、
/// これが実行 CPU の AVX-512F 対応を保証する。
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Avx512Kernel {
    /// [`Avx2Kernel`] と同じ封止パターン。
    _private: (),
}

#[cfg(target_arch = "x86_64")]
impl Avx512Kernel {
    /// 実行 CPU が AVX-512F をサポートする場合のみ `Some` を返す。
    pub(crate) fn try_new() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") {
            Some(Self { _private: () })
        } else {
            None
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Microkernel for Avx512Kernel {
    const MR: usize = avx512::MR;
    const NR: usize = avx512::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx512f") を確認済み
        // （avx512::kernel_unchecked の `# Safety` 契約を満たす）。
        unsafe { avx512::kernel_unchecked(ap, bp, c_tile, kc_len) }
    }
}

/// 実行時に選択された ISA を表す列挙型。[`super::gemm_blis`]／
/// [`super::gemm_blis_parallel`] の公開入口が [`Isa::detect`] の結果で
/// 1 回だけ match し、モノモーフィック化された `gemm_blis_region::<K>`
/// へ分岐する（`K` は対応する ISA トークン型）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Neon,
    Avx2,
    Avx512,
}

/// 検出結果（bool）から選択する ISA を決める純関数。優先順位は
/// Avx512 > Avx2 > Scalar（x86_64）。aarch64 は常に Neon、その他 arch は
/// 常に Scalar（呼び出し元の cfg 分岐が担保する。[`select_isa`] 自体は
/// arch に依存しない純ロジックとして単体テスト可能にする）。
///
/// x86_64 の [`Isa::detect_uncached`] からのみ本番経路で呼ばれる。他 arch
/// では駆動経路から外れるため `cfg(any(target_arch = "x86_64", test))` で
/// dead_code 警告を避けつつ、単体テストは arch に依らず実行できるように
/// 残す（`cargo check --target aarch64-unknown-linux-gnu` のクロス検証で
/// 未使用警告が出ないようにするための cfg）。
#[cfg(any(target_arch = "x86_64", test))]
fn select_isa(has_avx2_fma: bool, has_avx512f: bool) -> Isa {
    if has_avx512f {
        Isa::Avx512
    } else if has_avx2_fma {
        Isa::Avx2
    } else {
        Isa::Scalar
    }
}

impl Isa {
    /// プロセス内で 1 回だけ実行 CPU の機能検出を行い、以降は結果を
    /// キャッシュする（`is_x86_feature_detected!` 自体も std 内部で
    /// キャッシュされるが、dispatch 判定を 1 箇所に固定する意図で
    /// `OnceLock` を用いる）。
    pub fn detect() -> Isa {
        static ISA: OnceLock<Isa> = OnceLock::new();
        *ISA.get_or_init(Self::detect_uncached)
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_uncached() -> Isa {
        Isa::Neon
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_uncached() -> Isa {
        let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let has_avx512f = is_x86_feature_detected!("avx512f");
        select_isa(has_avx2_fma, has_avx512f)
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    fn detect_uncached() -> Isa {
        Isa::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_isa_prefers_avx512_over_avx2() {
        assert_eq!(select_isa(true, true), Isa::Avx512);
    }

    #[test]
    fn select_isa_prefers_avx2_over_scalar() {
        assert_eq!(select_isa(true, false), Isa::Avx2);
    }

    #[test]
    fn select_isa_falls_back_to_scalar() {
        assert_eq!(select_isa(false, false), Isa::Scalar);
    }

    #[test]
    fn select_isa_avx512_alone_selects_avx512() {
        assert_eq!(select_isa(false, true), Isa::Avx512);
    }

    /// [`Isa::detect`] は実行環境に依らず必ず何らかの ISA を返す
    /// （panic しないこと自体が契約。実測値は環境依存のため固定しない）。
    #[test]
    fn isa_detect_returns_consistent_result() {
        let first = Isa::detect();
        let second = Isa::detect();
        assert_eq!(
            first, second,
            "OnceLock キャッシュにより結果は不変であるはず"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_try_new_matches_feature_detection() {
        let expected = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        assert_eq!(Avx2Kernel::try_new().is_some(), expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_kernel_try_new_matches_feature_detection() {
        let expected = is_x86_feature_detected!("avx512f");
        assert_eq!(Avx512Kernel::try_new().is_some(), expected);
    }
}
