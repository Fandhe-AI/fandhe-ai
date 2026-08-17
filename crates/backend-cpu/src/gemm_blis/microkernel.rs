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
//!
//! ただし `avx512` モジュールのみ、上記に加えて `avx512_stable` cfg
//! （`backend-cpu` クレートルートの `build.rs` が、AVX-512F intrinsics と
//! `#[target_feature(enable = "avx512f")]` を実際にコンパイルする probe を
//! 実行して発行。バージョン番号の決め打ちではなく実測判定である理由は
//! `build.rs` のコメント参照。PR #337 の CI 実測: self-hosted runner の
//! rustc 1.88.0 では `_mm512_*` intrinsics が `stdarch_x86_avx512` unstable
//! ライブラリ機能のため E0658 でビルド不能）でもゲートする。AVX2 はこの
//! 制約を受けない（AVX2 intrinsics は長らく stable）。

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", avx512_stable))]
pub mod avx512;

use std::sync::OnceLock;

/// `ldc`／`c` 長の契約検査（scalar・neon・avx2・avx512 の各 ISA 実装間で
/// 重複していたロジックを集約。#691 レビュー指摘: 本番経路で `.expect()`
/// を使わず、オーバーフロー検査を明示的な `match` で分岐する
/// （`.claude/rules/coding-rust.md`「本番経路で unwrap()/expect() を
/// 使わない」）。
///
/// `mr == 0`・`ldc < NR`・`(MR-1)*ldc+NR` のオーバーフロー・`c` の長さ
/// 不足のいずれかであれば panic する（呼び出し元契約違反〈呼び出し元
/// バグ〉を早期検出する REQ-8 境界検査規約に基づく検証であり、実行時
/// 外部入力の検証ではない。各 ISA の `kernel`／`kernel_unchecked` 入口
/// から呼ばれる）。
///
/// ## `mr > 0` の明示検査（#691 レビュー再指摘への対応）
///
/// [`Microkernel`] trait は `MR > 0` を型・ドキュメントいずれでも契約と
/// して課していなかった（本対応で trait 側にも明記。[`Microkernel::MR`]
/// 参照）。外部実装が `MR = 0` の状態で（`ldc != NR` の）
/// [`Microkernel::run_with_ldc`] 既定実装から本関数を呼ぶと、`mr - 1` の
/// 減算が本来の境界検査より先に発生してしまう（debug ビルドでは減算
/// オーバーフローで panic、release ビルド〈overflow-checks 無効〉では
/// `usize::MAX` へラップし後続の `checked_mul`/`checked_add` がほぼ全ての
/// `ldc`/`nr` 組み合わせで `None` または長さ不足の assert 失敗に倒れる
/// ため実害は無いが、意図した「MR must be positive」という診断ではなく
/// 偶発的なオーバーフロー起因の panic になってしまう）。`mr.checked_sub(1)`
/// を計算チェーンへ含め、`mr == 0` を明示的に検出して意図の分かる panic
/// メッセージを出す。
pub(crate) fn check_c_tile_bounds(mr: usize, nr: usize, ldc: usize, c_len: usize) {
    assert!(ldc >= nr, "ldc must be at least NR");
    match mr
        .checked_sub(1)
        .and_then(|mr_minus_1| mr_minus_1.checked_mul(ldc))
        .and_then(|v| v.checked_add(nr))
    {
        Some(required) => assert!(
            c_len >= required,
            "C tile buffer too small for MR*ldc access pattern"
        ),
        None => panic!("MR must be positive, or ldc*MR overflow"),
    }
}

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
    /// マイクロカーネルタイルの行数。**契約: 1 以上でなければならない**
    /// （#691 レビュー指摘。`MR == 0` は [`check_c_tile_bounds`] の
    /// `mr.checked_sub(1)` 経由で明示的に panic するが、`0` を許容する
    /// 設計ではない。実装がこの契約を破ると
    /// [`Microkernel::run_with_ldc`] 既定実装・各 ISA の
    /// `kernel`／`kernel_unchecked` は正しい結果を返さない）。
    const MR: usize;
    /// マイクロカーネルタイルの列数。**契約: 1 以上でなければならない**
    /// （[`Self::MR`] 同様。`NR == 0` は本 trait 側では明示検査していない
    /// が、`ldc >= NR` の `assert` および呼び出し元のタイル分割ロジック
    /// が非ゼロを前提とするため、実装は 0 を返してはならない）。
    const NR: usize;

    /// `ap`（packed A）・`bp`（packed B）から `c_tile`（MR×NR、row-major・
    /// ld=NR）へ `kc_len` ぶんの寄与を加算する。安全な呼び出し専用の入口
    /// であり、内部の `unsafe`（intrinsics 呼び出し）はトークンの生成
    /// 経路（検出済みの場合のみ構築可能）によって健全性が保証される。
    ///
    /// ## 公開 API 非破壊（#691 レビュー指摘への再対応）
    ///
    /// 当初 #691 対応として本メソッドへデフォルト実装（[`Self::run_with_ldc`]
    /// への委譲）を与え `run_with_ldc` を必須メソッドとしたが、これは
    /// 「従来 `run` のみを実装するクレート外部の `Microkernel` 実装」が
    /// 新設の必須メソッド `run_with_ldc` 未実装により `E0046` でコンパイル
    /// 不能になる別の破壊的変更を生んでいた（codex-review・Cursor Bugbot
    /// 双方の再指摘）。そのため本メソッドを**従来どおり必須メソッド
    /// （デフォルト実装なし）として維持**し、`ldc` 拡張は
    /// [`Self::run_with_ldc`] 側にのみデフォルト実装を持たせる非対称な形へ
    /// 修正した。既存の外部実装（`run` のみをオーバーライド）は無変更で
    /// コンパイル可能になる（`run_with_ldc` のデフォルト実装が `run` へ
    /// フォールバックする。[`Self::run_with_ldc`] のドキュメント参照）。
    /// 組み込み実装（[`ScalarKernel`] 等）は `run` の実装を
    /// `run_with_ldc(..., Self::NR, ...)` への委譲として与えつつ、
    /// `run_with_ldc` 自体は ISA ごとの直接 C 経路（#557）を実装する。
    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize);

    /// `ldc` 契約版（#557: 完全タイルの C 直接ロード/ストア）。`c` は要素
    /// `c[i * ldc + j]`（`i in 0..MR`・`j in 0..NR`）のみを読み書きする
    /// 対象とし、それ以外のインデックスへは触れない。この契約により、
    /// 呼び出し元（[`super::gemm_blis_region`]）は 2 通りの呼び出し方が
    /// できる:
    ///
    /// - 完全タイル（`mr_eff == MR && nr_eff == NR`）: C の実バッファから
    ///   タイル原点起点のサブスライスを直接渡し `ldc = n`（C の列数）と
    ///   する。コピーイン/コピーアウトが不要になる（#557 の主目的）
    /// - 端タイル: 従来どおり `MAX_TILE` スタックバッファの先頭
    ///   `MR*NR` 要素を渡し `ldc = NR` とする（現行の密パッキング契約は
    ///   `ldc = NR` の特殊ケースとして包含される）
    ///
    /// ## デフォルト実装（#691 再指摘への対応。公開 API 非破壊）
    ///
    /// 本メソッドを必須にすると [`Self::run`] のみを実装する既存の外部
    /// `Microkernel` 実装を破壊するため、デフォルト実装を設けて
    /// オーバーライド不要にする:
    ///
    /// - `ldc == Self::NR`（密パッキング契約）: 追加コピーなしで
    ///   [`Self::run`] へそのまま委譲する
    /// - `ldc != Self::NR`（#557 の直接 C 経路が使われるケース）:
    ///   [`Self::run`] は `ldc = NR` の密パッキングしか扱えないため、
    ///   ヒープ確保した `MR*NR` 要素のスクラッチタイルへ `c` の現在値を
    ///   `ldc` ストライドでギャザーし、[`Self::run`] を密パッキング契約で
    ///   呼んだ後、結果を `ldc` ストライドで `c` へスキャッタし直す
    ///   （正しさ優先のフォールバック。#557 が狙うコピー往復削減の効果は
    ///   本フォールバック経路には及ばないが、組み込みカーネル
    ///   （[`ScalarKernel`]／[`NeonKernel`]／[`Avx2Kernel`]／
    ///   [`Avx512Kernel`]）は全て本メソッドを直接オーバーライドしており
    ///   本番の駆動経路（[`super::gemm_blis_region`]）はこのフォールバック
    ///   を通らない）
    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        if ldc == Self::NR {
            self.run(ap, bp, c, kc_len);
            return;
        }
        check_c_tile_bounds(Self::MR, Self::NR, ldc, c.len());
        let mut tile = vec![0.0f32; Self::MR * Self::NR];
        for i in 0..Self::MR {
            tile[i * Self::NR..(i + 1) * Self::NR].copy_from_slice(&c[i * ldc..i * ldc + Self::NR]);
        }
        self.run(ap, bp, &mut tile, kc_len);
        for i in 0..Self::MR {
            c[i * ldc..i * ldc + Self::NR].copy_from_slice(&tile[i * Self::NR..(i + 1) * Self::NR]);
        }
    }
}

/// 全 arch 共通のスカラーフォールバックトークン。検出不要のため
/// `ScalarKernel` は常に構築可能（`Default` 相当の unit struct）。
#[derive(Clone, Copy)]
pub struct ScalarKernel;

impl Microkernel for ScalarKernel {
    const MR: usize = scalar::MR;
    const NR: usize = scalar::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        self.run_with_ldc(ap, bp, c_tile, Self::NR, kc_len);
    }

    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        scalar::kernel_with_ldc(ap, bp, c, ldc, kc_len);
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
        self.run_with_ldc(ap, bp, c_tile, Self::NR, kc_len);
    }

    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        neon::kernel_with_ldc(ap, bp, c, ldc, kc_len);
    }
}

/// aarch64 NEON 12×8（firestorm 型）A/B 対抗トークン。[`neon::kernel_12x8`]
/// と同じく NEON は baseline ISA のため実行時検出不要。`super::dispatch_region`
/// の駆動経路には接続せず、`gemm_blis::mod` の `#[cfg(test)]` A/B 計測
/// テスト（`super::gemm_blis_with_kernel` 経由）専用のトークン（#559）。
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct Neon12x8Kernel;

#[cfg(target_arch = "aarch64")]
impl Microkernel for Neon12x8Kernel {
    const MR: usize = neon::MR_12X8;
    const NR: usize = neon::NR_12X8;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        neon::kernel_12x8(ap, bp, c_tile, kc_len);
    }

    /// `run_with_ldc` の override（Cursor Bugbot 指摘・review 4947832636・
    /// thread PRRT_kwDOTuUCJc6Zq5PH への対応）。
    ///
    /// このトークンはオーバーライドせず [`Microkernel::run_with_ldc`] の
    /// デフォルト実装（`ldc != NR` でヒープ確保 `Vec` によるギャザー/
    /// スキャッタ）に頼っていたが、`super::gemm_blis_region` の完全タイル
    /// 経路（#557）は `ldc = n`（C の実列数）で常に `run_with_ldc` を呼ぶ
    /// ため、[`NeonKernel`]（8×12 側）が `run_with_ldc` を直接オーバー
    /// ライドして直接ロード/ストアで応じるのに対し、`Neon12x8Kernel`
    /// （12×8 側）のみ full-tile 呼び出しのたびに `MR*NR`（=96 要素）タイル
    /// を毎回ヒープ確保するという非対称が生じていた。本トークンは
    /// `super::dispatch_region` の駆動経路には接続せず `gemm_blis::mod` の
    /// `#[cfg(test)]` A/B 計測テスト（8×12 vs 12×8 のスループット比較。
    /// #559）専用のため、正当性への影響はなかったが、比較対象の
    /// `NeonKernel` 側だけコピー往復が省かれヒープ確保も無い一方
    /// `Neon12x8Kernel` 側は毎呼び出しヒープ確保が乗るため、A/B 比較の
    /// 公平性が損なわれていた。
    ///
    /// [`neon::kernel_12x8`] 自体は `ldc` 一般化（[`neon::kernel_with_ldc`]
    /// 相当の strided ロード/ストア）を持たないため、ここではスタック
    /// 固定長バッファ（ヒープ確保なし）へのギャザー/スキャッタで
    /// デフォルト実装と同じ正当性を保ちつつ `Vec` 確保のみを除去する
    /// （A/B 計測の対称性回復が目的であり、`NeonKernel` と同水準の
    /// strided 直接アクセスへ揃えるほどの追加実装コストは、本番駆動
    /// 経路に接続しないテスト専用トークンには見合わないと判断した）。
    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        if ldc == Self::NR {
            self.run(ap, bp, c, kc_len);
            return;
        }
        check_c_tile_bounds(Self::MR, Self::NR, ldc, c.len());
        // ヒープ確保（`Vec`）を避けるため MR_12X8*NR_12X8（=96）固定長の
        // スタック配列を使う（`super::MAX_TILE`〈256〉以内。デフォルト
        // 実装との唯一の差分はここのみで、ギャザー/スキャッタのロジック
        // 自体は同一）。
        let mut tile = [0.0f32; neon::MR_12X8 * neon::NR_12X8];
        for i in 0..Self::MR {
            tile[i * Self::NR..(i + 1) * Self::NR].copy_from_slice(&c[i * ldc..i * ldc + Self::NR]);
        }
        neon::kernel_12x8(ap, bp, &mut tile, kc_len);
        for i in 0..Self::MR {
            c[i * ldc..i * ldc + Self::NR].copy_from_slice(&tile[i * Self::NR..(i + 1) * Self::NR]);
        }
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
        self.run_with_ldc(ap, bp, c_tile, Self::NR, kc_len);
    }

    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx2") && ("fma") を確認済み
        // （avx2::kernel_unchecked_with_ldc の `# Safety` 契約を満たす）。
        unsafe { avx2::kernel_unchecked_with_ldc(ap, bp, c, ldc, kc_len) }
    }
}

/// x86_64 AVX-512F トークン。[`Avx512Kernel::try_new`] 経由でのみ構築でき、
/// これが実行 CPU の AVX-512F 対応を保証する。`avx512_stable` cfg
/// （[`avx512`] モジュールドキュメント参照）が立っている rustc（AVX-512F
/// intrinsics が stable 化済みと `build.rs` の probe が確認できた場合）
/// でのみコンパイル対象となる。
#[cfg(all(target_arch = "x86_64", avx512_stable))]
#[derive(Clone, Copy)]
pub struct Avx512Kernel {
    /// [`Avx2Kernel`] と同じ封止パターン。
    _private: (),
}

#[cfg(all(target_arch = "x86_64", avx512_stable))]
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

#[cfg(all(target_arch = "x86_64", avx512_stable))]
impl Microkernel for Avx512Kernel {
    const MR: usize = avx512::MR;
    const NR: usize = avx512::NR;

    fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
        self.run_with_ldc(ap, bp, c_tile, Self::NR, kc_len);
    }

    fn run_with_ldc(&self, ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
        // SAFETY: Self は try_new() 経由でのみ構築可能であり、構築時点で
        // is_x86_feature_detected!("avx512f") を確認済み
        // （avx512::kernel_unchecked_with_ldc の `# Safety` 契約を満たす）。
        unsafe { avx512::kernel_unchecked_with_ldc(ap, bp, c, ldc, kc_len) }
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
        // `avx512_stable` cfg が立っていない rustc では `Avx512Kernel` 自体が
        // コンパイル対象外となり `dispatch_region`（`super` モジュール）は
        // AVX-512F 実行 CPU 上でも AVX2/scalar へフォールバックする
        // （`build.rs` のコメント・本モジュール冒頭のドキュメント参照）。
        // ここで実行 CPU の avx512f 対応を無条件採用すると、この
        // introspection API（`Isa::detect`）が実際の dispatch 結果と
        // 食い違う（Bugbot 指摘: PR #337 review 4886262265・comment
        // 3738511491）。`avx512_stable` 未設定時は has_avx512f を常に
        // false とし、実際の dispatch 経路と一致させる。
        #[cfg(avx512_stable)]
        let has_avx512f = is_x86_feature_detected!("avx512f");
        #[cfg(not(avx512_stable))]
        let has_avx512f = false;
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

    /// `run` のみをオーバーライドする「クレート外部実装」を模したトークン
    /// （#691 再指摘への回帰テスト）。`Microkernel::run_with_ldc` は
    /// デフォルト実装のみに頼り、MR=2・NR=2 の単純な mul_add 累積を
    /// 密パッキング（`ldc == NR`）契約でのみ行う。
    #[derive(Clone, Copy)]
    struct LegacyRunOnlyKernel;

    impl Microkernel for LegacyRunOnlyKernel {
        const MR: usize = 2;
        const NR: usize = 2;

        fn run(&self, ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
            for i in 0..Self::MR {
                for j in 0..Self::NR {
                    let mut acc = c_tile[i * Self::NR + j];
                    for p in 0..kc_len {
                        acc = ap[p * Self::MR + i].mul_add(bp[p * Self::NR + j], acc);
                    }
                    c_tile[i * Self::NR + j] = acc;
                }
            }
        }
    }

    /// `run` のみを実装するレガシー実装は、密パッキング契約（`ldc == NR`）
    /// では追加コピーなしで `run_with_ldc` のデフォルト実装から呼び出せる
    /// （#691 再指摘: 公開 API 非破壊の核心シナリオ）。
    #[test]
    fn run_with_ldc_default_impl_delegates_to_run_when_ldc_equals_nr() {
        let k = LegacyRunOnlyKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0]; // kc_len=2, MR=2 (p-major)
        let bp = [5.0f32, 6.0, 7.0, 8.0]; // kc_len=2, NR=2 (p-major)
        let mut via_run = [0.0f32; 4];
        k.run(&ap, &bp, &mut via_run, 2);

        let mut via_default_ldc = [0.0f32; 4];
        k.run_with_ldc(&ap, &bp, &mut via_default_ldc, 2, 2);

        assert_eq!(
            via_run, via_default_ldc,
            "ldc == NR では run へ委譲するはず"
        );
    }

    /// `run` のみを実装するレガシー実装でも、`ldc != NR`（#557 の直接 C
    /// 経路が使うストライド）で呼ばれた場合はギャザー/スキャッタの
    /// フォールバックにより正しい結果を返す（正しさ優先。性能は
    /// 組み込みカーネルほど出ないがコンパイル不能にはならない）。
    #[test]
    fn run_with_ldc_default_impl_gather_scatter_fallback_matches_dense_result() {
        let k = LegacyRunOnlyKernel;
        let ap = [1.0f32, 2.0, 3.0, 4.0];
        let bp = [5.0f32, 6.0, 7.0, 8.0];

        // 密パッキング（ldc = NR = 2）で得られる期待値。
        let mut dense = [0.0f32; 4];
        k.run_with_ldc(&ap, &bp, &mut dense, 2, 2);

        // ldc = 3 の広い C バッファ（行間に 1 要素のギャップ）へ同じ演算を行う。
        // 初期値は 0 埋めなので dense と同じ結果になるはず。
        let mut strided = [0.0f32; 6];
        k.run_with_ldc(&ap, &bp, &mut strided, 3, 2);

        assert_eq!(strided[0], dense[0]);
        assert_eq!(strided[1], dense[1]);
        assert_eq!(strided[3], dense[2]);
        assert_eq!(strided[4], dense[3]);
        // ギャップ列（各行末尾）には触れない契約であるはず。
        assert_eq!(strided[2], 0.0);
        assert_eq!(strided[5], 0.0);
    }

    /// `mr == 0`（[`Microkernel::MR`] の契約違反）を [`check_c_tile_bounds`]
    /// が意図の分かる panic メッセージで検出することの回帰テスト（PR #691
    /// レビュー指摘 `PRRT_kwDOTuUCJc6Zq7vw`）。修正前は `mr - 1` の減算
    /// オーバーフロー由来の panic（debug ビルド）だったが、
    /// `mr.checked_sub(1)` により意図した「MR must be positive」診断へ
    /// 変わったことを確認する。
    #[test]
    #[should_panic(expected = "MR must be positive")]
    fn check_c_tile_bounds_rejects_mr_zero_with_explicit_message() {
        check_c_tile_bounds(0, 2, 2, 4);
    }

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

    #[cfg(all(target_arch = "x86_64", avx512_stable))]
    #[test]
    fn avx512_kernel_try_new_matches_feature_detection() {
        let expected = is_x86_feature_detected!("avx512f");
        assert_eq!(Avx512Kernel::try_new().is_some(), expected);
    }

    /// `avx512_stable` cfg 未設定の rustc では `Avx512Kernel` 自体が
    /// コンパイル対象外となり `super::dispatch_region` は実行 CPU の
    /// avx512f 対応に関わらず AVX2/scalar のみを試す。この条件下で
    /// `Isa::detect` が `Isa::Avx512` を返すと、実際の dispatch と
    /// introspection API（`Isa::detect`）が食い違う（PR #337 review
    /// 4886262265・comment 3738511491 の Bugbot 指摘）。本テストは
    /// その食い違いへの回帰を検知する。
    #[cfg(all(target_arch = "x86_64", not(avx512_stable)))]
    #[test]
    fn isa_detect_never_reports_avx512_when_not_stable() {
        assert_ne!(Isa::detect(), Isa::Avx512);
    }
}
