//! NVRTC コンパイルヘルパ（カーネルソース非依存の基盤部分）。
//!
//! PoC-v2-3 の `compile`（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:84-117`）
//! を productize したもの。カーネルソース文字列・起動 API は #33/#34 の
//! スコープであり、本モジュールは「ソース文字列と arch を渡すと `Ptx` を
//! 返す」境界のみを担う。
//!
//! # A03（インジェクション）対応
//!
//! `CUDA_INCLUDE_PATH` 環境変数はコンパイルオプションの include パス
//! 文字列としてのみ `CompileOptions::include_paths` へ渡し、シェル展開・
//! コマンド実行には一切使わない（`.claude/rules/security.md`）。
//!
//! `src` 引数の契約（イシュー #516 で更新）: 従来は「コンパイル時定数
//! （`&'static str`）であること」を呼び出し元に要求していたが、
//! `kernels_mma::render_mma_f16`／`kernels_wmma_opt::render_wmma_tf32_opt`・
//! `render_wmma_f16_opt`（shape／タイル／段数のテンプレート文字列展開）
//! 導入に伴い、実行時に組み立てた `String`（`&str` として渡す）も許容
//! する。許容の条件は「静的テンプレート（本クレート内の `const *_BODY`
//! リテラル）＋ fail-closed 検証済みの型付き数値・enum パラメータのみ
//! から組み立てられていること」であり、外部入力文字列（ユーザー・
//! ファイル・環境変数由来の文字列）をソースへ連結する経路は引き続き
//! 禁止する（A03 対策の実体は不変。`kernels_mma.rs`・
//! `kernels_wmma_opt.rs` の `render_*` 関数のドキュメンテーションコメント
//! 参照）。
//!
//! # サポート対象 OS（非 unix のビルドを拒否する。イシュー #509 PR #677
//! codex-review P0 再指摘対応）
//!
//! 本モジュールの NVRTC キャッシュ I/O（[`ensure_cache_root`]・
//! [`store_cache_entry`]・[`load_cache_entry`] とその内部実装）は
//! symlink 脱出・TOCTOU 対策として `O_NOFOLLOW`・fd 相対解決
//! （`/proc/self/fd/<fd>` 経由・`openat`/`mkdirat`/`renameat`/`unlinkat`
//! 相当の自前 FFI）を用いており、いずれも unix 系 API（`<fcntl.h>`・
//! `std::os::unix::fs::OpenOptionsExt`）にのみ依存する。以前は
//! `#[cfg(not(unix))]` に「検証してから読み書きする」という構造的に
//! TOCTOU を閉じられないパスベースのフォールバック実装を維持していたが、
//! `.claude/rules/deps-policy.md`（`libc`／`rustix` はユーザー承認なしに
//! 追加できない）の制約下ではこのフォールバックを同水準まで強化できず、
//! 本クレートのサポート対象（Linux／macOS。`backend-switching-design.md`）
//! では到達しないコードでもあったため、`.claude/rules/security.md` の
//! fail-closed 方針に従いフォールバックごと削除した。非 unix
//! ターゲットでのビルドはコンパイルエラーで明示的に拒否する。

#[cfg(not(unix))]
compile_error!(
    "backend-cuda の NVRTC キャッシュ（crates/backend-cuda/src/nvrtc.rs）は \
     unix（Linux/macOS）のみサポートする。fd pin による TOCTOU 対策が \
     O_NOFOLLOW・/proc/self/fd・openat 等の unix 系 API に依存するため、 \
     非 unix 向けの同水準フォールバックは提供しない \
     （イシュー #509 PR #677 codex-review P0 再指摘対応）。"
);

use std::ffi::OsStr;
use std::fs;
use std::hash::Hash;
use std::num::NonZeroU32;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::error::CudaError;

/// 次元（M／N／K）ごとの compile-time 定数化選択（イシュー #519・C-7）。
///
/// DeepGEMM `runtime_utils.hpp` の `get_compiled_dim`（`compiled_dims`
/// 文字列、例 `"nk"`）の型付き版。DeepGEMM 設計の意図（M はバッチ×系列長
/// で可変なのでカーネル数を抑えるため動的化し、N/K は重み形状で
/// 固定的なので定数化して最適化を最大化する）をそのまま踏襲するが、
/// **文字列パースは行わない**（A03: 外部入力文字列がカーネルソースへ
/// 混入する経路を型で遮断する。`kernel_name: &'static str` の判断
/// 〈本ファイル冒頭〉と同型）。
///
/// `kernels_mma::DimSpec`（M/N/K 各次元を `Dynamic`／`Static(u32)` の
/// どちらでカーネルソースへ焼き込むかを表す機構）に対して、「どの次元を
/// 焼き込むか」という選択ポリシー側の型であり、[`CudaKernelDescriptor`]
/// のキャッシュキー構成要素としても使う（[`Self::cache_shape`] 参照）。
///
/// フィールドは private + getter とし、`new`（任意組合せ）と 3 つの
/// プリセット定数（[`Self::DYNAMIC_ALL`]／[`Self::STATIC_NK`]／
/// [`Self::STATIC_MNK`]）以外の経路で構築できないようにする（既存
/// descriptor 型と同じ不変条件維持方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompiledDims {
    m: bool,
    n: bool,
    k: bool,
}

impl CompiledDims {
    /// 全次元を実行時引数のまま扱う（`kernels_mma::DimSpec::Dynamic` 相当）。
    /// shape が変わってもカーネルは 1 種類のみで済むため、キャッシュ
    /// エントリ数を最小化したい既定探索・フォールバック用途に使う。
    pub const DYNAMIC_ALL: Self = Self {
        m: false,
        n: false,
        k: false,
    };

    /// N/K のみ定数化し M は動的のまま扱う（DeepGEMM `get_compiled_dim`
    /// の既定意図: M はバッチ×系列長で可変・N/K は重み形状で固定的）。
    /// `Default` 実装もこのプリセットに揃える。
    pub const STATIC_NK: Self = Self {
        m: false,
        n: true,
        k: true,
    };

    /// 全次元を定数化する（最適化を最大化するが、shape の組合せごとに
    /// 別カーネル・別キャッシュエントリが必要になる）。
    pub const STATIC_MNK: Self = Self {
        m: true,
        n: true,
        k: true,
    };

    /// 任意の次元別組合せで構築する（プリセット以外の組合せが必要な
    /// 呼び出し元向け）。
    pub const fn new(m: bool, n: bool, k: bool) -> Self {
        Self { m, n, k }
    }

    /// M 次元を定数化するか。
    pub fn m(&self) -> bool {
        self.m
    }

    /// N 次元を定数化するか。
    pub fn n(&self) -> bool {
        self.n
    }

    /// K 次元を定数化するか。
    pub fn k(&self) -> bool {
        self.k
    }

    /// `shape` を「キャッシュキー用 shape」へ正規化する。定数化しない
    /// （`Dynamic` のまま扱う）次元は sentinel `0` に落とす（DeepGEMM
    /// `get_compiled_dim` が非選択次元に対し `0` を返す設計の踏襲）。
    ///
    /// 非選択次元の実際の値が異なっても同一カーネル（同一キャッシュ
    /// エントリ）を再利用できることが本メソッドの目的であり、
    /// [`CudaKernelDescriptor::new`] は本メソッドの戻り値を `shape` として
    /// 保持することでこれを構造的に強制する（呼び出し元の規律に頼らない。
    /// PR #643 レビューの教訓と同方針）。
    pub fn cache_shape(
        &self,
        shape: tensor_core::dispatch::GemmShape,
    ) -> tensor_core::dispatch::GemmShape {
        tensor_core::dispatch::GemmShape::new(
            if self.m { shape.m } else { 0 },
            if self.n { shape.n } else { 0 },
            if self.k { shape.k } else { 0 },
        )
    }
}

impl Default for CompiledDims {
    /// DeepGEMM `get_compiled_dim` の既定意図（N/K 定数化・M 動的）に
    /// 揃える（[`Self::STATIC_NK`] 参照）。
    fn default() -> Self {
        Self::STATIC_NK
    }
}

/// カーネル特化パラメータ記述子（Phase C-1・イシュー #504）。
///
/// 親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・
/// 静的タイル選定）の先頭タスクであり、後続タスクが共通に使う
/// 「キャッシュキーの単位」を定義する。C-2（自作ハッシュ・ディレクトリ
/// 命名・#506）・C-4（プロセス内 LRU・#511）・C-6（テンプレート展開・
/// #516）から利用される想定であり、本モジュールではキー型の定義のみを
/// 担う（コンパイル・キャッシュ本体は持たない）。
///
/// 設計は 2 つの参照実装を踏まえる:
/// - DeepGEMM: キャッシュキーに「カーネル名・コンパイラ種別＋バージョン
///   （例 NVRTC12.9）・コンパイルフラグ・ソース」を含め、GPU アーキ違い
///   （`--gpu-architecture=sm_XX`）が自動的に別エントリになる設計。
/// - metal-flash-attention: descriptor 全パラメータを `Optional` で持ち
///   未確定時は `fatalError` する設計 + `Equatable + Hashable` なキー型。
///
/// 本型は後者の「Optional + 実行時 panic」を Rust の型システムで排除する
/// （受け入れ基準）。全フィールドを [`CudaKernelDescriptor::new`] の必須
/// 引数として要求し、ブロックタイル寸法・パイプライン段数は
/// [`NonZeroU32`] でゼロ値を型レベルで排除するため、「未確定パラメータの
/// まま生成できる」状態が構築できない。
///
/// フィールドは private + getter とし、構築後にゼロ値へ書き換えられない
/// （不変条件を型で維持する）。
///
/// `PartialEq`／`Eq`／`Hash` は `#[derive]` ではなく手動実装する
/// （[`Self::shape`]・[`Self::cache_key_shape`] 参照）。`shape`
/// フィールドは呼び出し元が渡した実値をそのまま保持し正規化しないため
/// （`shape()` の契約維持。codex-review 指摘・PR #674・P1）、これを
/// キャッシュキー相当の判定へ含めると `cache_key_shape` による動的次元の
/// 正規化（同一カーネルへの再利用）が `shape` フィールドの実値差分で
/// 無効化されてしまう（例: `STATIC_NK` で M のみ異なる 2 shape は
/// `cache_key_shape` は同一だが `shape` は異なるため、`shape` を含めると
/// 誤って別キャッシュエントリになる）。キャッシュキーとしての同一性判定
/// は `cache_key_shape` のみで行い、`shape` は除外する。
#[derive(Debug, Clone)]
pub struct CudaKernelDescriptor {
    /// カーネル名。C-2（#506）のキャッシュディレクトリ命名
    /// （`kernel.<name>.<hash>`）で使われる想定であり、`&'static str`
    /// （コンパイル時定数）に限定して外部入力の混入を型で遮断する
    /// （A03 対策。`compile_ptx` の `src` 引数契約〈本ファイル冒頭〉と
    /// 同型の判断）。
    kernel_name: &'static str,
    /// GEMM 形状（M/N/K）。`tensor_core::dispatch` の既存 `Hash + Eq` 型を
    /// 再利用し、ディスパッチ規則（#67/#68）が扱う形状表現と揃える。
    ///
    /// 呼び出し元が渡した実値をそのまま保持する（正規化しない）。
    /// キャッシュキーとして使う正規化済み shape は [`Self::cache_key_shape`]
    /// を参照（codex-review 指摘・PR #674・P1: `shape()` の既存契約
    /// 〈実 shape をそのまま返す〉を壊さないため、正規化後の値は別
    /// フィールド・別 getter に分離する）。
    shape: tensor_core::dispatch::GemmShape,
    /// ブロックタイル M 次元。
    block_m: NonZeroU32,
    /// ブロックタイル N 次元。
    block_n: NonZeroU32,
    /// ブロックタイル K 次元。
    block_k: NonZeroU32,
    /// パイプライン段数。SMEM 予算からの逆算ロジックは
    /// [`derive_pipeline_stages`]（C-8・#521 で実装）が担い、本型は
    /// 導出済みの値をデータとして保持するのみ。
    stages: NonZeroU32,
    /// 入出力 dtype。`tensor_core::dispatch::DType` を再利用する。
    dtype: tensor_core::dispatch::DType,
    /// 次元ごとの compile-time 定数化選択（イシュー #519）。
    /// [`Self::new`]（従来コンストラクタ）経由では `None`（次元特化なし。
    /// 導入前の契約を維持）、[`Self::new_with_compiled_dims`] 経由では
    /// `Some`。`Hash + Eq` derive が本フィールドを自動的に含むため、
    /// 同一 shape でも定数化組合せが違えば別キャッシュエントリになる。
    compiled_dims: Option<CompiledDims>,
    /// キャッシュキーとして使う正規化済み shape。`compiled_dims` が
    /// `None`（従来コンストラクタ経由）のときは `shape` と同一値
    /// （導入前の挙動を維持）。`Some` のときは
    /// [`CompiledDims::cache_shape`] で非選択（動的扱い）次元を sentinel
    /// `0` に正規化した値（イシュー #519）。`shape()` の意味は変えず、
    /// キャッシュ用の正規化後表現はこの専用フィールド・専用 getter
    /// （[`Self::cache_key_shape`]）に閉じ込める。
    cache_key_shape: tensor_core::dispatch::GemmShape,
}

impl CudaKernelDescriptor {
    /// `CudaKernelDescriptor` を構築する（次元特化〈`compiled_dims`〉
    /// なしの従来コンストラクタ）。
    ///
    /// イシュー #519（C-7）導入前の契約をそのまま維持する: `shape` は
    /// 正規化されず渡した実値のまま保持され、`shape()`／`cache_key_shape()`
    /// はいずれも同一値を返す。次元ごとの compile-time 定数化選択が
    /// 必要な呼び出し元は [`Self::new_with_compiled_dims`] を使う
    /// （codex-review 指摘・PR #674・P1: `new` への必須引数追加・`shape()`
    /// のセマンティクス変更は既存利用者を壊す破壊的変更のため、別
    /// コンストラクタへ分離した）。
    ///
    /// `block_m`／`block_n`／`block_k`／`stages` がゼロの場合、および
    /// `kernel_name` がパス走査文字（`/`・`\`・`..`）・空文字列の場合の
    /// 検証は [`Self::new_with_compiled_dims`] のドキュメンテーション
    /// コメント参照（検証本体は共通の内部ヘルパへ集約している）。
    pub fn new(
        kernel_name: &'static str,
        shape: tensor_core::dispatch::GemmShape,
        block_m: u32,
        block_n: u32,
        block_k: u32,
        stages: u32,
        dtype: tensor_core::dispatch::DType,
    ) -> Result<Self, CudaError> {
        Self::build(
            kernel_name,
            shape,
            block_m,
            block_n,
            block_k,
            stages,
            dtype,
            None,
        )
    }

    /// `CudaKernelDescriptor` を、次元（M／N／K）ごとの compile-time
    /// 定数化選択（[`CompiledDims`]）付きで構築する（イシュー #519・
    /// C-7）。
    ///
    /// `block_m`／`block_n`／`block_k`／`stages` がゼロの場合は
    /// `CudaError::InvalidKernelDescriptor` を返す（panic 経路なし。
    /// `unwrap`/`expect` は使わない。`.claude/rules/coding-rust.md`）。
    ///
    /// `kernel_name` はパス走査文字（`/`・`\`・`..`）と空文字列を拒否する
    /// （codex-review 指摘・PR #641・P2）。`kernel_name` は `&'static str`
    /// でも `"../escape"` 等のリテラルは書けてしまい、C-2（#506）の
    /// キャッシュディレクトリ命名（`kernel.<name>.<hash>`）で
    /// そのままパスセグメントへ使われる想定のため、型（`&'static str`）
    /// だけでは意図しないパス脱出を防げない。値の妥当性は構築時に
    /// ここで検証し、C-2 側は「検証済み文字列である」ことを前提にできる。
    ///
    /// `compiled_dims` で定数化対象とした次元の `shape` 実値が `0` の
    /// 場合は `CudaError::InvalidKernelDescriptor` を返す（イシュー #519。
    /// [`CompiledDims::cache_shape`] が動的次元の sentinel として `0` を
    /// 使うため、定数化対象の実値が `0` だと「値 0 で定数化」なのか
    /// 「動的次元の sentinel」なのかが構築時点で曖昧になる。この検査は
    /// 曖昧性を型・構造の両面で排除する）。
    ///
    /// `shape()` は渡した実値をそのまま返す（正規化しない）。正規化済み
    /// のキャッシュキー用 shape は [`Self::cache_key_shape`] を使う
    /// （codex-review 指摘・PR #674・P1）。
    ///
    /// 引数 8 個は `clippy::too_many_arguments`（既定閾値 7）に抵触するが、
    /// 全フィールドを構築時必須引数として要求し「未確定パラメータのまま
    /// 構築できない」不変条件を保つ設計（本型ドキュメンテーションコメント
    /// 参照）自体が目的のため、ビルダーパターン等への分割は行わず
    /// `#[allow]` する（`gemm.rs`／`gemm_wmma.rs`／`kernels_mma.rs` の
    /// 同種カーネル起動関数と同じ判断）。
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_compiled_dims(
        kernel_name: &'static str,
        shape: tensor_core::dispatch::GemmShape,
        block_m: u32,
        block_n: u32,
        block_k: u32,
        stages: u32,
        dtype: tensor_core::dispatch::DType,
        compiled_dims: CompiledDims,
    ) -> Result<Self, CudaError> {
        Self::build(
            kernel_name,
            shape,
            block_m,
            block_n,
            block_k,
            stages,
            dtype,
            Some(compiled_dims),
        )
    }

    /// [`Self::new`]／[`Self::new_with_compiled_dims`] 共通の構築ロジック。
    /// `compiled_dims` が `None` の場合は次元特化なし（従来コンストラクタ
    /// 経由）として扱い、`cache_key_shape` は `shape` と同一値になる。
    #[allow(clippy::too_many_arguments)]
    fn build(
        kernel_name: &'static str,
        shape: tensor_core::dispatch::GemmShape,
        block_m: u32,
        block_n: u32,
        block_k: u32,
        stages: u32,
        dtype: tensor_core::dispatch::DType,
        compiled_dims: Option<CompiledDims>,
    ) -> Result<Self, CudaError> {
        // イシュー #506（Phase C-2）レビュー指摘: 先頭・末尾がドット
        // （例 `".foo"`・`"foo."`）の `kernel_name` は、当初この検証を
        // 通過したうえで `cache_entry_dir_name()`（イシュー #506）が
        // 生成する `"kernel..foo.<hash>"`/`"kernel.foo..<hash>"` に
        // `".."` を出現させ、同関数の縦深防御検査で構築後になって初めて
        // 拒否される（fail-closed だが、構築時点では有効に見える
        // descriptor が消費時点まで使えないことが分かる、という早期検知の
        // 弱さがあった）。ここで前倒しして拒否し、「構築に成功した
        // `kernel_name` は消費側で必ず使える」という契約を保つ。
        if kernel_name.is_empty()
            || kernel_name.contains('/')
            || kernel_name.contains('\\')
            || kernel_name.contains("..")
            || kernel_name.starts_with('.')
            || kernel_name.ends_with('.')
        {
            return Err(CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "kernel_name must be a non-empty path-safe segment \
                     (no '/', '\\\\', \"..\", and no leading/trailing '.'), \
                     got {kernel_name:?}"
                ),
            });
        }
        if let Some(compiled_dims) = compiled_dims {
            for (name, is_compiled, value) in [
                ("shape.m", compiled_dims.m(), shape.m),
                ("shape.n", compiled_dims.n(), shape.n),
                ("shape.k", compiled_dims.k(), shape.k),
            ] {
                if is_compiled && value == 0 {
                    return Err(CudaError::InvalidKernelDescriptor {
                        detail: format!(
                            "{name} is selected for compile-time constant folding \
                             (compiled_dims) but is 0, which is indistinguishable from \
                             the dynamic-dimension cache sentinel"
                        ),
                    });
                }
            }
        }
        let non_zero = |value: u32, field: &str| {
            NonZeroU32::new(value).ok_or_else(|| CudaError::InvalidKernelDescriptor {
                detail: format!("{field} must be non-zero (got 0)"),
            })
        };
        let cache_key_shape = match compiled_dims {
            Some(compiled_dims) => compiled_dims.cache_shape(shape),
            None => shape,
        };
        Ok(Self {
            kernel_name,
            shape,
            block_m: non_zero(block_m, "block_m")?,
            block_n: non_zero(block_n, "block_n")?,
            block_k: non_zero(block_k, "block_k")?,
            stages: non_zero(stages, "stages")?,
            dtype,
            compiled_dims,
            cache_key_shape,
        })
    }

    /// カーネル名。
    pub fn kernel_name(&self) -> &'static str {
        self.kernel_name
    }

    /// GEMM 形状。呼び出し元が渡した実値をそのまま返す（正規化しない。
    /// `new`／`new_with_compiled_dims` いずれの経路でも同じ契約）。
    pub fn shape(&self) -> tensor_core::dispatch::GemmShape {
        self.shape
    }

    /// キャッシュキーとして使う正規化済み shape。[`Self::new`] 経由
    /// （次元特化なし）では [`Self::shape`] と同一値、
    /// [`Self::new_with_compiled_dims`] 経由では動的次元を sentinel `0`
    /// に正規化した値（[`CompiledDims::cache_shape`] 契約）。
    pub fn cache_key_shape(&self) -> tensor_core::dispatch::GemmShape {
        self.cache_key_shape
    }

    /// 次元ごとの compile-time 定数化選択。[`Self::new`] 経由（次元特化
    /// なし）では `None`。
    pub fn compiled_dims(&self) -> Option<CompiledDims> {
        self.compiled_dims
    }

    /// ブロックタイル M 次元。
    pub fn block_m(&self) -> NonZeroU32 {
        self.block_m
    }

    /// ブロックタイル N 次元。
    pub fn block_n(&self) -> NonZeroU32 {
        self.block_n
    }

    /// ブロックタイル K 次元。
    pub fn block_k(&self) -> NonZeroU32 {
        self.block_k
    }

    /// パイプライン段数。
    pub fn stages(&self) -> NonZeroU32 {
        self.stages
    }

    /// 入出力 dtype。
    pub fn dtype(&self) -> tensor_core::dispatch::DType {
        self.dtype
    }
}

/// キャッシュキーとしての同一性は `cache_key_shape`（正規化済み）のみで
/// 判定し、呼び出し元が渡した実値そのままの `shape` は含めない（本型
/// ドキュメンテーションコメント参照。codex-review 指摘・PR #674・P1）。
impl PartialEq for CudaKernelDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.kernel_name == other.kernel_name
            && self.cache_key_shape == other.cache_key_shape
            && self.block_m == other.block_m
            && self.block_n == other.block_n
            && self.block_k == other.block_k
            && self.stages == other.stages
            && self.dtype == other.dtype
            && self.compiled_dims == other.compiled_dims
    }
}

impl Eq for CudaKernelDescriptor {}

impl Hash for CudaKernelDescriptor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kernel_name.hash(state);
        self.cache_key_shape.hash(state);
        self.block_m.hash(state);
        self.block_n.hash(state);
        self.block_k.hash(state);
        self.stages.hash(state);
        self.dtype.hash(state);
        self.compiled_dims.hash(state);
    }
}

/// コンパイルキャッシュのキー（Phase C-1・イシュー #504）。
///
/// [`CudaKernelDescriptor`] に加え、環境依存パラメータ（compute
/// capability・NVRTC バージョン・コンパイルフラグ）を含める。DeepGEMM の
/// 設計（キャッシュキーに compiler signature を含め GPU アーキ違いが
/// 自動的に別エントリになる）を踏襲し、環境が変わった際に古いキャッシュ
/// エントリが誤って再利用されない性質を担保する（OWASP A08 整合性。
/// `.claude/rules/security.md`）。
///
/// ソースコード断片は `source` フィールドとして本キーに含める
/// （C-5・#514）。
///
/// # C-5（#514）の必要性判断: 「再帰ハッシュ」ではなく最終ソース全体を
/// 取り込む方式を採る理由
///
/// イシュー #514 が要求する DeepGEMM 型の対応物（生成コードが `#include`
/// する外部ヘッダファイルを正規表現抽出して再帰的にハッシュへ取り込み、
/// フルプリプロセスなしにヘッダ変更を検知する機構）は、本リポジトリの
/// カーネルソースには構造的に不要である。DeepGEMM がヘッダの再帰 include
/// パースを必要とするのは、生成コードがリポジトリ内ヘッダファイルを
/// `#include` で参照するため。一方、本クレートのカーネルソースは
/// `kernels_mma::render_mma_f16`／`render_mma_f16_unchecked` のような
/// `render_*` 関数が Rust の `const` 文字列断片（`MMA_F16_BODY` 等）と
/// 型付きパラメータ（shape・タイル寸法・段数）のみからプロセス内で
/// 最終 `String` を組み立てて確定させる契約（本ファイル冒頭 A03 節・
/// `kernels_mma.rs` ドキュメンテーションコメント参照）であり、リポジトリ
/// 内ヘッダファイルへの `#include` 参照が存在しない（NVRTC へ渡す
/// `#include <cuda_fp16.h>` 等は toolkit 標準ヘッダのみで、その ABI 変更は
/// 本フィールドではなく `nvrtc_version` フィールドが追従する）。
///
/// したがって、最終レンダー済みソース文字列そのものをキーへ含めれば、
/// 断片（`*_BODY` 定数・展開ヘッダ・将来の断片合成のいずれ）を編集しても
/// 推移的に本フィールドの値が変わり、DeepGEMM の再帰ハッシュと同じ
/// 「ソースの何がどう変わっても確実にキャッシュミスする」性質を、
/// ファイルパース・fs I/O を一切伴わない単純な方式で得られる。これは
/// DeepGEMM 自身がキーに `code`（最終展開済みソース全体）を含める設計
/// （`compiler.hpp` の `{name}$${signature}$${flags}$${code}`）とも
/// 対応関係にある。
///
/// ディレクトリ命名（`kernel.<name>.<hash>`）とハッシュ関数自体は C-2
/// （#506）のスコープであり、本型は「ハッシュ化される前のキーの単位」を
/// 定義するのみ。
///
/// # 破壊的変更の意図的な受容（C-5・#514・codex-review P1 是正）
///
/// [`Self::new`]／[`Self::from_device`] は本フィールド（`source`）追加に
/// 伴い必須引数が増える破壊的シグネチャ変更を受けている。旧シグネチャを
/// 互換維持したまま `source` なしのコンストラクタを併存させる代替案は
/// 採らない: `source` を含まないキーは `canonical_bytes()`／派生
/// `Hash`/`Eq` がソース変更を検知できず、C-5 が解消対象とする「ソースを
/// 編集してもキャッシュがヒットし続ける」問題（stale cache reuse。OWASP
/// A08 整合性）をまさに再導入してしまうため、互換コンストラクタの追加は
/// 安全側ではなく危険側の選択となる。
///
/// 移行契約: 本型は crate root（`lib.rs`）から `pub use` で再公開されて
/// おり形式上は公開 API だが、`backend-cuda` はこのリポジトリの「唯一の
/// サポートされる公開 API 面」ではない内部クレートであり（`facade` が
/// 公開面。CLAUDE.md「想定クレート 10 個」節）、かつ workspace は
/// `publish = false`（crates.io 非公開）のため crate 外・リポジトリ外の
/// SemVer 契約下の利用者は存在しない（`grep -rn CudaKernelCacheKey .` で
/// 呼び出し元は本ファイル自身のみと確認済み）。既存呼び出し元
/// （本ファイル内の実装・テスト）は本 PR で全て新シグネチャへ移行済み。
/// 将来 crate 外から利用する場合は最終レンダー済みソース文字列
/// （`kernels_mma::render_mma_f16` 等の戻り値）を渡すこと。
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CudaKernelCacheKey {
    descriptor: CudaKernelDescriptor,
    /// `CudaDevice::compute_capability()` と同型（`device.rs`）。
    compute_capability: (i32, i32),
    /// NVRTC バージョン（[`nvrtc_version`] が返す値。DeepGEMM の
    /// compiler signature「NVRTC12.9」相当）。
    nvrtc_version: (i32, i32),
    /// `--gpu-architecture` 等のコンパイルフラグ。アーキ違いが自動的に
    /// 別キャッシュエントリになる性質はこのフィールドが担う。
    ///
    /// 順序契約: `Vec<String>` は要素順序を `Hash`/`Eq` に含めるため、
    /// 呼び出し元は意味的に同一なフラグ集合を常に決定的な順序で
    /// 構築すること（`HashSet` 等の非決定順コレクションから直接
    /// 構築しない）。契約違反時の影響は誤ヒット（安全性違反）方向では
    /// なく無駄なキャッシュミス方向のみだが、C-2（#506）でのキー消費側
    /// 実装時にも本契約を維持する。
    compile_flags: Vec<String>,
    /// 最終レンダー済みカーネルソース全体（C-5・#514）。上記ドキュメン
    /// テーションコメントの必要性判断を参照。`derive(Hash)`/`derive(Eq)`
    /// へ自然に含まれるため、プロセス内キー比較（将来の C-4 LRU・#511）は
    /// ハッシュ縮退なしの厳密一致になる。
    ///
    /// 外部への公開 getter は設けない（`pub(crate)` に留める）:
    /// `RenderedMmaKernel` がソース文字列を外部へ返さない設計（PR #643
    /// P0 対応）を採っており、キー経由でソース全文が漏出する新たな公開
    /// 経路を作らないため。
    source: String,
}

impl CudaKernelCacheKey {
    /// `descriptor`（検証済み）と環境パラメータ、および最終レンダー済み
    /// カーネルソース（`source`。C-5・#514）からキーを構築する。
    /// `descriptor` は [`CudaKernelDescriptor::new`] を経由済みのため
    /// infallible（`Result` を返さない）。
    pub fn new(
        descriptor: CudaKernelDescriptor,
        compute_capability: (i32, i32),
        nvrtc_version: (i32, i32),
        compile_flags: Vec<String>,
        source: String,
    ) -> Self {
        Self {
            descriptor,
            compute_capability,
            nvrtc_version,
            compile_flags,
            source,
        }
    }

    /// `device`（`CudaDevice`）から compute capability を取得し、
    /// `nvrtc_version()` で NVRTC バージョンを取得してキーを構築する
    /// 実機導線。CI では GPU が無いため `new`（literal 構築）でテストする
    /// （ユニットテスト参照）。
    pub fn from_device(
        descriptor: CudaKernelDescriptor,
        device: &crate::device::CudaDevice,
        compile_flags: Vec<String>,
        source: String,
    ) -> Result<Self, CudaError> {
        Ok(Self::new(
            descriptor,
            device.compute_capability(),
            nvrtc_version()?,
            compile_flags,
            source,
        ))
    }

    /// カーネル特化パラメータ記述子。
    pub fn descriptor(&self) -> &CudaKernelDescriptor {
        &self.descriptor
    }

    /// compute capability（`major`, `minor`）。
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// NVRTC バージョン（`major`, `minor`）。
    pub fn nvrtc_version(&self) -> (i32, i32) {
        self.nvrtc_version
    }

    /// コンパイルフラグ。
    pub fn compile_flags(&self) -> &[String] {
        &self.compile_flags
    }

    /// 処理系非依存の正準バイト列表現（イシュー #506・Phase C-2）。
    ///
    /// `derive(Hash)` の `Hash` 実装は Rust 処理系バージョン間で安定性が
    /// 保証されない（`std::hash::Hash` のドキュメント契約）ため、ディスク
    /// 永続キー（[`Self::cache_entry_dir_name`]）には使えない。本メソッドは
    /// DeepGEMM が `{name}$${signature}$${flags}$${code}` の明示的な文字列
    /// 連結をハッシュする方式（`compiler.hpp`）に倣い、全フィールドを
    /// 曖昧さのない形式（可変長フィールドは長さプレフィクス付き、数値は
    /// フィールド順固定の固定長 LE バイト列）で連結する。区切り文字方式
    /// ではなく長さプレフィクス方式を採るのは、コンパイルフラグ文字列
    /// 自体に区切り文字が混入すると異なる論理キーが同一バイト列に
    /// 縮退しうるため。
    ///
    /// 先頭 1 バイトはエンコーディングバージョン。将来このバイト列表現を
    /// 変更する際はこのバージョン番号を上げ、互換性の切り替え点として使う。
    /// **このバイト列表現の変更は既存キャッシュエントリを全て無効化する**
    /// （ハッシュ値が変わるため）契約であることに注意。
    ///
    /// v2（C-5・#514）: 末尾に [`Self::source`] を長さプレフィクス付きで
    /// 追記した。v1 → v2 の切り替えにより、v1 時点で存在しうるディスク
    /// キャッシュエントリ（C-3・#509 実装後に実体化）は本フィールド追加
    /// と無関係に全て無効化される（意図どおり。C-2 実装時点のコメントで
    /// 「C-5 でソース取り込み時に ENCODING_VERSION を上げる」ことが
    /// 予告済み）。
    ///
    /// v3（イシュー #519・cursor[bot] 指摘・PR #674）: shape 部分を
    /// 生の [`CudaKernelDescriptor::shape`] から
    /// [`CudaKernelDescriptor::cache_key_shape`]（動的次元を sentinel
    /// `0` に正規化済み）へ差し替え、[`CudaKernelDescriptor::compiled_dims`]
    /// を末尾（`source` の直後）に追記した。`PartialEq`／`Hash`
    /// （本ファイル `impl PartialEq for CudaKernelDescriptor` 参照）は
    /// 既に `cache_key_shape`／`compiled_dims` をキーにするよう手動実装
    /// 済みだったが、本メソッド（ディスク永続キー用の正準バイト列）は
    /// 追従しておらず、プロセス内キー（メモリ上の `HashMap` 等）とディスク
    /// キーが乖離していた: 動的次元がディスク側では正規化されず、`source`
    /// が一致するだけで異なる specialization policy（例: `STATIC_NK` と
    /// `STATIC_MNK`）が同一の `stable_hash` を共有しうる（キャッシュ汚染・
    /// 誤ヒットのリスク。OWASP A08 整合性）。v2 → v3 の切り替えにより
    /// v2 時点のディスクキャッシュエントリは全て無効化される（意図どおり）。
    fn canonical_bytes(&self) -> Vec<u8> {
        const ENCODING_VERSION: u8 = 3;

        let mut buf = Vec::new();
        buf.push(ENCODING_VERSION);

        push_len_prefixed_str(&mut buf, self.descriptor.kernel_name);

        let cache_key_shape = self.descriptor.cache_key_shape();
        buf.extend_from_slice(&cache_key_shape.m.to_le_bytes());
        buf.extend_from_slice(&cache_key_shape.n.to_le_bytes());
        buf.extend_from_slice(&cache_key_shape.k.to_le_bytes());

        buf.extend_from_slice(&self.descriptor.block_m.get().to_le_bytes());
        buf.extend_from_slice(&self.descriptor.block_n.get().to_le_bytes());
        buf.extend_from_slice(&self.descriptor.block_k.get().to_le_bytes());
        buf.extend_from_slice(&self.descriptor.stages.get().to_le_bytes());

        // `DType` は non-exhaustive ではない自クレート型ではなく
        // `tensor_core::dispatch::DType` だが、キー用途では判別子のみが
        // 意味を持つため 1 バイトの手書き判別子へ写像する（derive(Hash)
        // に依存しない方針をここでも一貫させる）。
        let dtype_tag: u8 = match self.descriptor.dtype {
            tensor_core::dispatch::DType::F32 => 0,
            tensor_core::dispatch::DType::F16 => 1,
        };
        buf.push(dtype_tag);

        buf.extend_from_slice(&self.compute_capability.0.to_le_bytes());
        buf.extend_from_slice(&self.compute_capability.1.to_le_bytes());
        buf.extend_from_slice(&self.nvrtc_version.0.to_le_bytes());
        buf.extend_from_slice(&self.nvrtc_version.1.to_le_bytes());

        buf.extend_from_slice(&(self.compile_flags.len() as u32).to_le_bytes());
        for flag in &self.compile_flags {
            push_len_prefixed_str(&mut buf, flag);
        }

        // v2（C-5・#514）: 最終レンダー済みカーネルソース全体。フラグ列の
        // 末尾（可変長プレフィクス方式）の直後に置くため、長さプレフィクス
        // 方式のまま追記すれば既存フィールドとの境界曖昧化は起きない
        // （`push_len_prefixed_str` のドキュメンテーションコメント参照）。
        push_len_prefixed_str(&mut buf, &self.source);

        // v3（イシュー #519・cursor[bot] 指摘・PR #674）: 次元ごとの
        // compile-time 定数化選択。`cache_key_shape` は非選択次元を
        // sentinel `0` に正規化するため、`compiled_dims` 自体を含めない
        // と異なる選択ポリシー（例: 全次元 `0` の `DYNAMIC_ALL` shape と
        // `STATIC_NK` で N=K=0 の shape）が同一バイト列に縮退しうる。
        // `Option<CompiledDims>` は固定長 1 バイトの判別子 + 3 bool で
        // 十分表現できるため（`None`／`Some` を含め最大 4 バイト）、長さ
        // プレフィクスは使わず固定長のまま追記する。
        match self.descriptor.compiled_dims {
            None => buf.push(0),
            Some(dims) => {
                buf.push(1);
                buf.push(dims.m() as u8);
                buf.push(dims.n() as u8);
                buf.push(dims.k() as u8);
            }
        }

        buf
    }

    /// [`Self::canonical_bytes`] を FNV-1a 64bit（[`fnv1a_64`]）でハッシュ
    /// した値。[`Self::cache_entry_dir_name`] のハッシュ部として使う
    /// ディスク永続キー本体（イシュー #506）。
    ///
    /// `pub(crate)` に留める（crate ルートからは再公開しない）: FNV-1a・
    /// `canonical_bytes` のフィールド順序といった内部ハッシュ表現の選択を
    /// 外部利用者との SemVer 契約にしない。[`cache_root`]／
    /// [`cache_entry_path`] と同じ理由（PR #659 レビュー指摘）。
    pub(crate) fn stable_hash(&self) -> u64 {
        fnv1a_64(&self.canonical_bytes())
    }

    /// キャッシュエントリのディレクトリ名を返す（`kernel.<name>.<hash>`。
    /// DeepGEMM の `~/.deep_gemm/cache/kernel.<name>.<hex16>` 命名に倣う。
    /// イシュー #506・Phase C-2）。
    ///
    /// ディスクへの書き込み・アトミック rename は C-3（#509）のスコープで
    /// あり、本メソッドは純粋なパス計算（fs I/O なし）に留める。
    ///
    /// `kernel_name` は [`CudaKernelDescriptor::new`] が構築時に検証済み
    /// （パス走査文字・空文字列を拒否）だが、構築後の不変条件破壊
    /// （型不変条件はあくまで実行時検査であり、将来のリファクタでこの
    /// 保証が崩れる可能性はゼロではない）への縦深防御として、生成した
    /// ディレクトリ名自体にもパスセパレータ・`..` が含まれないことを
    /// ここで再検査する（fail-closed。A03 対策）。
    ///
    /// `pub(crate)` に留める（crate ルートからは再公開しない）:
    /// `kernel.<name>.<hash>` というディレクトリ命名規則自体が内部
    /// キャッシュ形式であり、外部利用者との SemVer 契約にしない。
    /// [`stable_hash`](Self::stable_hash)・[`cache_root`]・
    /// [`cache_entry_path`] と同じ理由（PR #659 レビュー指摘）。
    pub(crate) fn cache_entry_dir_name(&self) -> Result<String, CudaError> {
        let name = format!(
            "kernel.{}.{:016x}",
            self.descriptor.kernel_name,
            self.stable_hash()
        );
        if name.contains('/') || name.contains('\\') || name.contains("..") {
            return Err(CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "generated cache entry dir name unexpectedly contains \
                     path-traversal characters: {name:?}"
                ),
            });
        }
        Ok(name)
    }
}

/// 手動 `Debug` 実装（C-5・#514・codex-review P1 是正）。`derive(Debug)`
/// のままだと [`CudaKernelCacheKey::source`]（数十 KB になりうる展開済み
/// カーネルソース全文）がそのままログ・パニックメッセージへ出力されて
/// しまう（情報露出。`.claude/rules/security.md` セキュリティ考慮）。
///
/// 当初案（先頭 40 文字の平文要約 `source_summary`）は codex-review で
/// P1 指摘を受けた: ソース先頭 40 文字はカーネル名・シグネチャ等の識別
/// 情報を含みうる平文断片であり、「公開 getter を設けずソース全文漏出を
/// 防止する」（`source` フィールドのドキュメンテーションコメント参照）
/// という設計方針に反する部分的漏出だった。本実装は `source` の内容を
/// 一切表示せず、長さと非可逆な変更検知用フィンガープリント
/// （[`fnv1a_64`]。[`Self::stable_hash`] と同一アルゴリズム）のみを出力
/// する。`fnv1a_64` は非暗号ハッシュのため、このフィンガープリントは
/// 「同一ソースかどうかの変更検知」用途に限られ、改竄検知・完全性保証
/// （OWASP A08）の根拠にはしない（[`fnv1a_64`] のドキュメンテーション
/// コメント参照）。他フィールドは derive 相当の表示を保つ。
impl std::fmt::Debug for CudaKernelCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKernelCacheKey")
            .field("descriptor", &self.descriptor)
            .field("compute_capability", &self.compute_capability)
            .field("nvrtc_version", &self.nvrtc_version)
            .field("compile_flags", &self.compile_flags)
            .field("source_len", &self.source.len())
            .field(
                "source_fnv1a64",
                &format_args!("{:016x}", fnv1a_64(self.source.as_bytes())),
            )
            .finish()
    }
}

/// `s` の UTF-8 バイト長（`u32` LE）＋バイト列そのものを `buf` へ追記する
/// ([`CudaKernelCacheKey::canonical_bytes`] のみが使う長さプレフィクス
/// エンコーディングヘルパ)。区切り文字ではなく長さプレフィクスを使う
/// 理由は当該メソッドのドキュメンテーションコメントを参照。
fn push_len_prefixed_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// FNV-1a 64bit ハッシュ（自作・非暗号）。
///
/// イシュー #506（Phase C-2）: ディスク上のコンパイルキャッシュ
/// エントリ名（[`CudaKernelCacheKey::cache_entry_dir_name`]）を短く
/// 決定的に導出する目的専用であり、改竄検知・完全性保証（OWASP A08）
/// には使わない（非暗号ハッシュのため衝突耐性の暗号学的保証はない。
/// 破損検出・整合性検証は C-3/C-10（#509・#529）のスコープで別途扱う）。
/// 依存クレート追加なしで実装する（deps-policy.md の許容 8 区分に
/// ハッシュ関数区分はなく、std のみで完結させる判断）。
///
/// アルゴリズムは標準の FNV-1a（32bit ではなく 64bit 版）。オフセット
/// ベーシスと素数は FNV の公開仕様値。
///
/// 既知テストベクタ（`""` → `0xcbf29ce484222325`・`"a"` →
/// `0xaf63dc4c8601ec8c`・`"foobar"` → `0x85944171f73967e8`）でユニット
/// テスト済み（下記 `tests` モジュール）。
///
/// `pub(crate)` に留める（crate ルートからは再公開しない）: FNV-1a という
/// 内部ハッシュ表現の選択を外部利用者との SemVer 契約にしない
/// （PR #659 レビュー指摘）。
pub(crate) const fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// コンパイルキャッシュのルートディレクトリを解決する内部純関数
/// （イシュー #506・Phase C-2）。
///
/// 優先順位: (1) `override_dir`（`RUST_AI_CUDA_CACHE_DIR` 相当。
/// DeepGEMM の `DG_JIT_CACHE_DIR` に相当する本リポ命名として新規決定）
/// → (2) `xdg_cache_home`（`XDG_CACHE_HOME`）→ (3) `home`（`$HOME`）。
/// (2)(3) には `rust-ai-library/cuda` を付加する。全て `None`（環境変数
/// 全欠落）なら [`CudaError::CacheDirUnavailable`] を返す（panic 経路
/// なし）。
///
/// 実環境変数を直接 `std::env::var_os` するのではなく引数として受け取る
/// のは `self-repair::isolation::resolve_toolchain_home_reinjections`
/// （`crates/self-repair/src/isolation.rs:242`）と同じ「注入で決定化」
/// パターン。edition 2024 では `std::env::set_var` が `unsafe` になり
/// （プロセス環境変数はテストバイナリ全体で共有されるグローバル状態）、
/// フォールバック分岐を書き換えなしで決定的に再現するテストが書けない
/// ため、本関数を純関数として切り出し公開ラッパー（[`resolve_cache_root`]）
/// が実環境変数を読んで委譲する形にする。
///
/// **安全側の検証**: `override_dir`・`xdg_cache_home`・`home` はいずれも
/// 外部環境変数由来の信頼できない入力であり、空文字列・相対パスの場合は
/// 三者とも同様に `CudaError::CacheDirUnavailable` で拒否する（PR #659
/// レビュー指摘。`override_dir` のみを検証し `xdg_cache_home`／`home` を
/// 未検証のまま `Path::join` すると `XDG_CACHE_HOME=.` 等でフォールバック
/// を迂回できたため三者を揃えた。空文字列は `Some("")` として渡ってきても
/// 「未設定扱い」でフォールバックせず明示的に拒否する。PR #659 codex-review
/// P0 指摘: 以前の実装は `!xdg.is_empty()` 相当のガードが `if let` の外に
/// なく、空文字列の `XDG_CACHE_HOME` が `HOME` へ素通りしていた）。
///
/// **`workspace_root` containment 検証（PR #659 codex-review P0 再指摘への
/// 対応。2 回目の設計変更）**: 1 回目の修正（P1 指摘対応）ではコンパイル時
/// `CARGO_MANIFEST_DIR` から導出した「ビルド時ワークスペースルート」との
/// 比較を削除し、containment 検証自体を持たない許可リスト方式に置き換えた。
/// しかしこれは `override_dir`（`RUST_AI_CUDA_CACHE_DIR`）がリポジトリ
/// ツリー内を指す絶対パス（例 `/workspace/repository/cache`）をそのまま
/// 受理してしまう回帰であり、codex-review に P0 として再指摘された。
/// 「ビルド時定数のハードコードは実行環境が変わると素通りする」という
/// 1 回目の指摘の要点は正しいが、対策は「containment 検証を削除する」
/// ことではなく「**信頼できる境界を呼び出し元から注入で受け取る**」こと
/// だった。本関数はこの反省を踏まえ `workspace_root: &Path`（`Option` では
/// なく必須引数）を受け取り、`override_dir`・`xdg_cache_home`・`home` の
/// 3 分岐すべてで解決結果を [`path_lexically_within`] により
/// `workspace_root` 配下でないことを検証する（3 分岐のうち `override_dir`
/// だけを検証対象外にする例外は設けない。それが今回の P0 指摘の核心の
/// ため）。`workspace_root` を `Option` にして呼び出し元が `None` を渡せる
/// 余地を残すと「検証を迂回できる」構造が復活するため、必須引数として
/// 契約に組み込む（呼び出し元は [`cache_root`] も参照）。
///
/// **C-3（#509）への委譲は残る範囲のみ**: 実際にディレクトリを作成・
/// オープンする時点での `canonicalize` 済みパスによる symlink 解決込みの
/// 再検証（本関数は fs I/O なしの字句正規化のみで、パスの実在を要求する
/// symlink 解決は原理的に実行できない）は引き続き C-3 のスコープとする
/// （`docs/cuda-jit-cache-design.md` 検証条件節も参照）。`workspace_root`
/// 自体をどう決定するか（呼び出し元がどの値を渡すか）は [`cache_root`] の
/// doc を参照。
///
/// **`workspace_root` は絶対パス必須（PR #659 codex-review Bugbot 指摘
/// 対応）**: [`path_lexically_within`] はコンポーネント単位の
/// `starts_with` 比較のため、`workspace_root` が相対パスだと絶対パスの
/// 候補（`override_dir`・`XDG_CACHE_HOME`／`HOME` 由来の候補はいずれも
/// 絶対パス必須で既に検証済み）との比較が常に不一致になり、containment
/// 判定が fail-open 側へ倒れる（本来ブロックすべきリポジトリ内キャッシュ
/// ルートを再び受理してしまう）。3 分岐へ分岐する前に本関数の入口で
/// `workspace_root` の絶対パス性を検証し、相対パスなら 3 分岐いずれも
/// 実行せず fail-closed で拒否する（迂回できる分岐を残さない）。
fn resolve_cache_root(
    workspace_root: &Path,
    override_dir: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf, CudaError> {
    if workspace_root.is_relative() {
        return Err(CudaError::CacheDirUnavailable {
            detail: format!(
                "workspace_root must be an absolute path for containment checks \
                 to be meaningful, got {workspace_root:?}"
            ),
        });
    }

    if let Some(dir) = override_dir {
        let path = Path::new(dir);
        if dir.is_empty() {
            return Err(CudaError::CacheDirUnavailable {
                detail: "RUST_AI_CUDA_CACHE_DIR is set but empty".to_string(),
            });
        }
        if path.is_relative() {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!("RUST_AI_CUDA_CACHE_DIR must be an absolute path, got {path:?}"),
            });
        }
        if path_lexically_within(path, workspace_root) {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!(
                    "RUST_AI_CUDA_CACHE_DIR must not resolve within the workspace root \
                     {workspace_root:?}, got {path:?}"
                ),
            });
        }
        return Ok(path.to_path_buf());
    }

    if let Some(xdg) = xdg_cache_home {
        if xdg.is_empty() {
            return Err(CudaError::CacheDirUnavailable {
                detail: "XDG_CACHE_HOME is set but empty".to_string(),
            });
        }
        let path = Path::new(xdg);
        if path.is_relative() {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!("XDG_CACHE_HOME must be an absolute path, got {path:?}"),
            });
        }
        let candidate = path.join("rust-ai-library").join("cuda");
        if path_lexically_within(&candidate, workspace_root) {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!(
                    "XDG_CACHE_HOME-derived cache root must not resolve within the workspace \
                     root {workspace_root:?}, got {candidate:?}"
                ),
            });
        }
        return Ok(candidate);
    }

    if let Some(home_dir) = home {
        if home_dir.is_empty() {
            return Err(CudaError::CacheDirUnavailable {
                detail: "HOME is set but empty".to_string(),
            });
        }
        let path = Path::new(home_dir);
        if path.is_relative() {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!("HOME must be an absolute path, got {path:?}"),
            });
        }
        let candidate = path.join(".cache").join("rust-ai-library").join("cuda");
        if path_lexically_within(&candidate, workspace_root) {
            return Err(CudaError::CacheDirUnavailable {
                detail: format!(
                    "HOME-derived cache root must not resolve within the workspace root \
                     {workspace_root:?}, got {candidate:?}"
                ),
            });
        }
        return Ok(candidate);
    }

    Err(CudaError::CacheDirUnavailable {
        detail: "none of RUST_AI_CUDA_CACHE_DIR, XDG_CACHE_HOME, HOME is set; \
                 cannot determine a cache root"
            .to_string(),
    })
}

/// `candidate` が `root` 配下（`root` 自身を含む）に字句上収まるかを
/// 判定する（[`resolve_cache_root`] 用。fs I/O なしの純比較。
/// [`Path::starts_with`] はコンポーネント単位の比較のため、
/// `/repo-extra` を `/repo` の配下と誤判定しない）。
///
/// 比較前に両パスを [`lexically_normalize`] で正規化する: 正規化なしでは
/// `..` コンポーネントを含む絶対パス（例 `/outside/../repo/cache`）が
/// `Path::starts_with` のコンポーネント単位比較を素通りしてしまう
/// （先頭コンポーネントが `root` と一致しないため）が、実際のファイル
/// 操作は `..` 解決後のパス（`root` 配下）を指してしまう。symlink 解決
/// までは行わない（fs I/O なしの字句正規化のみ）。
///
/// [`resolve_cache_root`] が `workspace_root` containment 検証（3 分岐
/// 共通）に使う（PR #659 codex-review P0 再指摘への対応で、1 回目の修正
/// 時に外していた呼び出しを復元した。詳細は [`resolve_cache_root`] doc
/// 参照）。`..` 折り畳み込みの字句正規化プリミティブは、C-3（#509）が実際に
/// ディレクトリを作成・オープンする時点で行う `canonicalize` 済みパスでの
/// symlink 解決込み再検証にも転用できる。
///
/// **前提: `root` は絶対パスであること（PR #659 codex-review Bugbot
/// 指摘対応）**: `starts_with` はコンポーネント単位の比較のため、
/// `root` が相対パスだと絶対パスの `candidate` とは先頭コンポーネント
/// （`RootDir`／`Prefix` の有無）が食い違い、実際には包含関係にあって
/// も必ず `false`（fail-open）を返す。本関数自身は `root` の絶対パス
/// 性を検証しない（呼び出し元の責務）。[`resolve_cache_root`] は入口で
/// `workspace_root.is_relative()` を検査してから本関数へ委譲すること
/// でこの前提を満たす。
fn path_lexically_within(candidate: &Path, root: &Path) -> bool {
    lexically_normalize(candidate).starts_with(lexically_normalize(root))
}

/// パスのコンポーネントを fs I/O なしで字句上（lexically）正規化する
/// （[`path_lexically_within`] 用）。`..`（[`Component::ParentDir`]）が
/// 現れるたびに直前の通常コンポーネント（[`Component::Normal`]）を
/// 取り除く（realpath 相当だがシンボリックリンク解決は行わない）。
///
/// ルート／プレフィックス（`/`・Windows ドライブ文字等）を越える `..`
/// は OS のパス解決（ルートの親はルート自身）と同様に無視する。相対
/// パスの先頭に現れる `..`（遡り先の通常コンポーネントが存在しない
/// 場合）のみ、そのまま保持する。
fn lexically_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => match normalized.components().next_back() {
                Some(Component::Normal(_)) => {
                    normalized.pop();
                }
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {
                    // ルート／プレフィックス直下からはこれ以上遡れない
                    // （OS のパス解決と同様、ルートの親はルート自身）
                    // ため `..` を読み捨てる。
                }
                _ => {
                    // 正規化結果が空、または直前も `..`: 遡り先の通常
                    // コンポーネントがまだないため、相対パスの先頭
                    // `..` として保持する。
                    normalized.push(component);
                }
            },
            Component::CurDir => {}
            other => normalized.push(other),
        }
    }
    normalized
}

/// [`resolve_cache_root`] の crate 内ラッパー。実プロセス環境変数
/// （`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME`）を読んで委譲する
/// （イシュー #506・Phase C-2。C-3（#509）・C-4（#511）から呼ばれる想定）。
///
/// `pub(crate)` に留める（crate ルートからは再公開しない）: 環境変数名・
/// キャッシュ配置規則は C-3/C-4/C-6（同一 crate 内の後続実装）が使う内部
/// 実装詳細であり、外部公開すると SemVer 契約になってしまう
/// （PR #659 レビュー指摘）。外部向けキャッシュ設定 API が要件になった
/// 場合は `docs/public-api-design.md` に契約・安定性を明記した専用 API
/// として別途設計する。
///
/// 呼び出し元は C-3（#509）・C-4（#511）で追加予定のため、本タスク
/// （C-2・#506）時点では crate 内に呼び出し元がまだなく `dead_code` 警告
/// が出る。テスト側は実環境変数への依存を避けるため注入可能な
/// [`resolve_cache_root`] を直接呼ぶ（本関数の doc 参照）ので、本関数
/// 自体は非テストコードから未参照のままになる。
///
/// **`workspace_root` を必須引数として受け取る（PR #659 codex-review P0
/// 再指摘への対応）**: 1 回目の修正（P1 指摘対応）ではコンパイル時
/// `CARGO_MANIFEST_DIR` から導出する「ビルド時ワークスペースルート」を
/// 引数に渡す処理を削除し、containment 検証自体をなくしてしまっていた
/// （ビルド環境と実行環境が異なる配布シナリオでは当該ワークスペースルート
/// が実行時のリポジトリ配置と一致せず検証が無条件で素通りするという
/// 1 回目の指摘自体は正しかったが、対策として検証を丸ごと削除したのが
/// 誤りだった）。本関数は `workspace_root: &Path` を必須引数として受け取り
/// [`resolve_cache_root`] へそのまま渡す。呼び出し元（C-3・C-4）は、
/// ビルド時定数のハードコードではなく実行時に確定する信頼できる境界
/// （例: 実行時に明示設定される runtime workspace 設定値。プロセスの
/// カレントディレクトリ `std::env::current_dir()` は「プロセス起動時の
/// 作業ディレクトリ」であり「リポジトリツリーの境界」ではないため
/// **使わない**: `XDG_CACHE_HOME`／`HOME` 未設定でカレントディレクトリが
/// たまたまホームディレクトリ配下だった場合、`~/.cache/...` という正当な
/// フォールバック結果が誤って拒否される）を C-3 設計時に決定して渡す。
///
/// **`workspace_root` は絶対パスであること（PR #659 codex-review Bugbot
/// 指摘対応）**: [`resolve_cache_root`] は相対パスを fail-closed で拒否
/// する（[`path_lexically_within`] の `starts_with` 比較は相対パスの
/// `root` に対して常に `false` を返し containment 判定が意味を失うため）。
/// C-3・C-4 が渡す境界値は必ず絶対パスへ解決してから渡すこと。
#[allow(
    dead_code,
    reason = "C-3(#509)/C-4(#511) の crate 内呼び出し元が実装されるまでの \
              意図的な先行スキャフォールディング（PR #659 レビュー指摘）"
)]
pub(crate) fn cache_root(workspace_root: &Path) -> Result<PathBuf, CudaError> {
    resolve_cache_root(
        workspace_root,
        std::env::var_os("RUST_AI_CUDA_CACHE_DIR").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// `root` と [`CudaKernelCacheKey::cache_entry_dir_name`] を合成し、
/// キャッシュエントリの完全パスを返す内部純関数（イシュー #506・
/// Phase C-2）。fs I/O は行わない純粋なパス組み立てのみ（`create_dir_all`
/// 等は C-3（#509）のスコープ）。`root` は呼び出し元
/// （[`cache_entry_path`] 経由の [`cache_root`]）で既に `workspace_root`
/// containment 検証を経ている前提であり、本関数自体は単純な結合のみを
/// 行う（結果が常に `root` 配下に収まることは `Path::join` の性質として
/// 自明であり、下記ユニットテストで回帰確認する）。
///
/// [`resolve_cache_root`]／[`cache_root`] と同じ「注入で決定化」パターン
/// （`self-repair::isolation::resolve_toolchain_home_reinjections`と同型。
/// `crates/self-repair/src/isolation.rs:242`）で `root` を引数化する。
/// 実 `cache_root()`（環境変数依存）を経由すると「結果が必ず `root` 配下
/// （`starts_with(root)`）」というトラバーサル防御（A03）の受け入れ基準を
/// テスト実行環境の実 `HOME` に依存させずには検証できないため、公開
/// ラッパー（[`cache_entry_path`]）と分離する。
fn cache_entry_path_in(root: &Path, key: &CudaKernelCacheKey) -> Result<PathBuf, CudaError> {
    let entry_name = key.cache_entry_dir_name()?;
    Ok(root.join(entry_name))
}

/// [`cache_root`] と [`CudaKernelCacheKey::cache_entry_dir_name`] を
/// 合成し、キャッシュエントリの完全パスを返す（イシュー #506・
/// Phase C-2。C-3（#509）・C-4（#511）から呼ばれる想定）。
///
/// 結果が必ず `root` 配下（`starts_with(root)`）に収まることを
/// [`cache_entry_path_in`] のユニットテストで検証し、トラバーサル防御
/// （A03）の第 3 層とする（第 1 層: `CudaKernelDescriptor::new` の構築時
/// 検証、第 2 層: `cache_entry_dir_name` 内の縦深防御検査）。`root` 自体が
/// リポジトリツリー外であることの検証（第 0 層）は [`cache_root`] 経由の
/// [`resolve_cache_root`] が `workspace_root` containment 検証で担う
/// （PR #659 codex-review P0 再指摘: `cache_entry_path_in` 自体にも
/// containment がないと指摘されたが、それは `cache_root` が渡す `root` に
/// 対して検証すべき事項であり、本関数の責務は `root` 配下への結合のみに
/// 留める設計とした）。
///
/// [`cache_root`] と同じ理由で `pub(crate)` に留める（crate ルートからは
/// 再公開しない。PR #659 レビュー指摘）。呼び出し元も [`cache_root`] と
/// 同じく C-3（#509）・C-4（#511）で追加予定のため、本タスク時点では
/// 先行スキャフォールディングとして `dead_code` を許容する。
#[allow(
    dead_code,
    reason = "C-3(#509)/C-4(#511) の crate 内呼び出し元が実装されるまでの \
              意図的な先行スキャフォールディング（PR #659 レビュー指摘）"
)]
pub(crate) fn cache_entry_path(
    workspace_root: &Path,
    key: &CudaKernelCacheKey,
) -> Result<PathBuf, CudaError> {
    cache_entry_path_in(&cache_root(workspace_root)?, key)
}

// ============================================================================
// キャッシュ I/O（イシュー #509・Phase C-3）
//
// 上記まで（C-2・#506）はキャッシュキーの正規化・ハッシュ化・ルート／
// エントリパスの「解決」のみを担う純関数群だった。以下は解決済みパスに
// 対する実際の fs I/O（一時ディレクトリコンパイル → fsync → アトミック
// rename）を追加する。参照実装は DeepGEMM の compiler（一時ディレクトリで
// ビルド後 `std::filesystem::rename`、先着プロセスがいた場合は rename
// 失敗を正常系として吸収。rename 前にボトムアップ再帰 fsync）。
//
// crate 内呼び出し元（GEMM 経路への結線・プロセス内 LRU）は C-4（#511）の
// スコープであり、本タスク時点では未結線（先行スキャフォールディング。
// C-2 の `cache_root`／`cache_entry_path` と同じ判断）。
// ============================================================================

/// キャッシュエントリ内のソースファイル名（NVRTC へ渡した `.cu` ソース
/// 全文）。[`validate_cache_entry`]・[`store_cache_entry_in`]・
/// [`load_cache_entry_in`] が共用する。
const CACHE_ENTRY_SOURCE_FILE: &str = "kernel.cu";

/// キャッシュエントリ内の成果物ファイル名（NVRTC コンパイル結果の PTX
/// アセンブリ全文）。定数化の理由は [`CACHE_ENTRY_SOURCE_FILE`] と同じ。
const CACHE_ENTRY_PTX_FILE: &str = "kernel.ptx";

/// コンパイルキャッシュから読み出したカーネルの実体（イシュー #509・
/// Phase C-3）。
///
/// エントリディレクトリ直下の [`CACHE_ENTRY_SOURCE_FILE`]／
/// [`CACHE_ENTRY_PTX_FILE`] の内容をそのまま保持する薄いデータ型。
/// NVRTC 呼び出し（`compile_ptx`）との結線・プロセス内 LRU での保持は
/// C-4（#511）のスコープであり、本 struct はディスクとメモリの間で
/// バイト列（UTF-8 テキスト）を運ぶだけの役割に留める。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CachedKernel {
    /// NVRTC へ渡したソース全文（[`CACHE_ENTRY_SOURCE_FILE`] の内容）。
    pub(crate) kernel_cu: String,
    /// NVRTC コンパイル結果の PTX アセンブリ全文
    /// （[`CACHE_ENTRY_PTX_FILE`] の内容）。
    pub(crate) kernel_ptx: String,
}

/// キャッシュエントリディレクトリが不変条件（[`CACHE_ENTRY_SOURCE_FILE`]・
/// [`CACHE_ENTRY_PTX_FILE`] の両方が存在する）を満たすかを判定する
/// （イシュー #509・Phase C-3。実装計画 §3.1・§3.3）。
///
/// [`store_cache_entry_in`]（rename 失敗時の「他プロセス先着」判定・
/// 破損置換判定）と [`load_cache_entry_in`]（読み出し前のミス判定）で
/// 共用する。存在確認のみを行う軽量チェックであり、内容の妥当性（PTX と
/// して有効か等）までは検証しない（DeepGEMM の kernel_runtime 検証と
/// 同水準）。ディレクトリ rename はアトミックなため通常運用で破損
/// エントリは生まれず、これはクラッシュ残骸・外部破壊への縦深防御
/// （A08 整合性）である。
///
/// [`Path::is_file`] ではなく [`is_plain_dir`]／[`is_plain_file`]（symlink
/// を追跡しない [`fs::symlink_metadata`] ベース）を使う（イシュー #509
/// codex-review P0 指摘対応の名残。`entry_dir` 自体または
/// [`CACHE_ENTRY_SOURCE_FILE`]／[`CACHE_ENTRY_PTX_FILE`] が symlink に
/// 置換されている場合、`is_file()`（symlink を追跡する）だと「有効な
/// キャッシュエントリ」と誤判定しうるため、その種のテストアサーション
/// でも symlink を追跡しない判定にしておく）。
///
/// 本関数自体は fs I/O の本番経路（[`load_cache_entry_in`]・
/// [`store_cache_entry_in`]）からは呼ばれない: それらの Unix 版は
/// パスの再解決を避けるため fd 相対版 [`validate_cache_entry_at`] を
/// 使い、検証と読み書きを同一 fd に結合することで TOCTOU を構造的に
/// 閉じている（本関数のようなパス経由の再検証は、検証と読み取りの間に
/// 別の TOCTOU 窓を開くため本番経路では使わない）。本関数はテスト
/// （symlink 差し替え等の外部観測用アサーション）専用として
/// `#[cfg(test)]` で残す（イシュー #509 PR #677 codex-review P0 再指摘
/// 対応。旧非 Unix フォールバックは検証と読み取りが別ステップで
/// TOCTOU を構造的に閉じられなかったため削除済み。crate ルート／
/// `nvrtc` モジュール冒頭の `compile_error!` 参照）。
#[cfg(test)]
fn validate_cache_entry(entry_dir: &Path) -> bool {
    is_plain_dir(entry_dir)
        && is_plain_file(&entry_dir.join(CACHE_ENTRY_SOURCE_FILE))
        && is_plain_file(&entry_dir.join(CACHE_ENTRY_PTX_FILE))
}

/// `path` が symlink ではない・空でない通常ファイルであるかを、symlink
/// を追跡しない [`fs::symlink_metadata`] で判定する（[`validate_cache_entry`]
/// 用。イシュー #509 codex-review P0 指摘対応・PR #677 Bugbot 指摘
/// 〈Empty entries block replacement〉対応）。
///
/// 非空検査（`meta.len() > 0`）を含めるのは、[`read_verified_cache_entry_file`]
/// の非空検査（実装計画 §3.1 の不変条件「両ファイルが存在し、いずれも
/// 空でない」）と判定基準を一致させるため。この検査を省くと、クラッシュ
/// 直後の 0 バイト残骸（`create` 直後・書き込み前にプロセスが落ちた場合
/// 等）を本関数（`validate_cache_entry` 経由で [`store_cache_entry_in`]
/// の破損判定に使われる）は「有効」と誤判定する一方、読み込み側
/// （[`read_verified_cache_entry_file`]）は同じ残骸を「ミス」として
/// `Ok(None)` にする。この不一致があると、rename 衝突時に
/// `store_cache_entry_in` が 0 バイト残骸を正常な先着エントリとみなして
/// 自分の新規書き込みを破棄してしまい、そのキーは以降ずっと「ミスと
/// 判定されて再コンパイルされる」空回りに陥る（正常書き込みが永久に
/// キャッシュへ反映されない）。
#[cfg(test)]
fn is_plain_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file() && meta.len() > 0)
        .unwrap_or(false)
}

/// `path` が symlink ではない通常ディレクトリであるかを、symlink を
/// 追跡しない [`fs::symlink_metadata`] で判定する（[`is_plain_file`] と
/// 同じ理由。エントリディレクトリ自体が symlink に置換されているケース
/// を拒否する）。
#[cfg(test)]
fn is_plain_dir(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_dir())
        .unwrap_or(false)
}

/// Linux の `O_NOFOLLOW`（`<fcntl.h>`。`asm-generic/bits/fcntl-linux.h`
/// では 8 進数 `0400000`）。[`open_nofollow`] 用（イシュー #509
/// codex-review P0 指摘対応）。
///
/// `libc`／`rustix` クレートは許容依存 8 区分（`.claude/rules/
/// deps-policy.md`）外でユーザー承認なしに追加できないため、
/// `std::os::unix::fs::OpenOptionsExt::custom_flags`（std 標準機能）へ
/// 渡すフラグ値を自前で定義する。値はターゲット OS ごとに異なるため
/// `target_os` で分岐する（本クレートのビルド対象は Linux/macOS のみ。
/// `backend-switching-design.md`。非 unix は crate ルート／`nvrtc`
/// モジュール冒頭の `compile_error!` でビルド自体を拒否する）。
#[cfg(target_os = "linux")]
const O_NOFOLLOW: i32 = 0o400000;

/// macOS（Darwin）の `O_NOFOLLOW`（`<fcntl.h>` では `0x0100`）。上記
/// Linux 版と同じ理由・同じ用途。
#[cfg(target_os = "macos")]
const O_NOFOLLOW: i32 = 0x0100;

/// Linux の `ENOTDIR`（`<asm-generic/errno-base.h>` では `20`）。
/// [`is_confirmed_non_directory_open_error`] 用（イシュー #509 PR #677
/// Cursor Bugbot 再指摘対応: `open_dir_nofollow`／`opendirat_nofollow`
/// の失敗原因を「確実に非ディレクトリ」と「それ以外（fd 枯渇・権限
/// 不足等）」に区別するため、errno の生値を見る必要がある。
/// `std::io::ErrorKind::NotADirectory`（`io_error_more`）は本ワーク
/// スペースの stable toolchain では未安定化のため使えない）。
#[cfg(target_os = "linux")]
const ENOTDIR: i32 = 20;

/// Linux の `ELOOP`（`<asm-generic/errno.h>` では `40`）。上記
/// `ENOTDIR` と同じ理由・同じ用途（symlink を `O_NOFOLLOW` で拒否
/// した場合の errno）。
#[cfg(target_os = "linux")]
const ELOOP: i32 = 40;

/// macOS（Darwin）の `ENOTDIR`（`<sys/errno.h>` では `20`）。上記 Linux
/// 版 `ENOTDIR` と同じ理由・同じ用途。
#[cfg(target_os = "macos")]
const ENOTDIR: i32 = 20;

/// macOS（Darwin）の `ELOOP`（`<sys/errno.h>` では `62`）。上記 Linux
/// 版 `ELOOP` と同じ理由・同じ用途。値は Linux と異なる（プラット
/// フォームごとに errno 番号が異なるため `target_os` で分岐する。
/// 上記 `O_NOFOLLOW` 等と同じ判断）。
#[cfg(target_os = "macos")]
const ELOOP: i32 = 62;

/// Linux の `O_NONBLOCK`（`<fcntl.h>`。8 進数 `04000`）。[`open_nofollow`]
/// ／[`openat_nofollow`] 用（イシュー #509 codex-review P0 指摘対応:
/// キャッシュエントリの `kernel.cu`／`kernel.ptx` として FIFO 等の特殊
/// ファイルを配置されると、`O_NONBLOCK` なしの `open` は writer 待ちで
/// ハングしうる〈外部由来キャッシュエントリの種類検証欠落による DoS〉）。
/// 正規ファイルに対しては `O_NONBLOCK` は open／read の挙動を変えない
/// （POSIX `open(2)`）ため、通常のキャッシュ読み出しには影響しない。
#[cfg(target_os = "linux")]
const O_NONBLOCK: i32 = 0o4000;

/// macOS（Darwin）の `O_NONBLOCK`（`<fcntl.h>` では `0x0004`）。上記
/// Linux 版と同じ理由・同じ用途。
#[cfg(target_os = "macos")]
const O_NONBLOCK: i32 = 0x0004;

/// macOS（Darwin）の `O_CLOEXEC`（`<fcntl.h>` では `0x01000000`）。
/// [`openat_nofollow`] 専用（イシュー #509 PR #677 codex-review P2
/// 指摘対応）。
///
/// [`open_nofollow`]／[`open_dir_nofollow`] は `std::fs::OpenOptions`
/// 経由で開くため、std が全 Unix open 系呼び出しへ既定で付与する
/// close-on-exec（`sys::pal::unix::fs` 実装。fork/exec 先へ fd を漏らさない
/// ための std 既定動作）を自動的に受け取る。一方 `openat_nofollow` は
/// std を経由しない `extern "C"` の `openat(2)` 直接呼び出しのため
/// std の既定付与が及ばず、明示的に `O_CLOEXEC` を渡さない限り生成
/// fd に close-on-exec が設定されない（`self-repair` 等がサブ
/// プロセスを fork/exec する経路と将来結合した場合、この fd がキャッシュ
/// エントリの中身を子プロセスへ意図せず漏出させうる）。
#[cfg(target_os = "macos")]
const O_CLOEXEC: i32 = 0x0100_0000;

/// macOS（Darwin）の `O_CREAT`（`<fcntl.h>` では `0x0200`）。
/// [`create_file_pinned`]・[`create_dir_all_verified`] macOS 版用（イシュー
/// #509 PR #677 codex-review P0 再指摘対応: 旧パスベース実装が呼んでいた
/// [`fs::write`]（既存ファイルを暗黙に truncate する）相当の書き込みを、
/// fd 起点かつ排他生成に置き換えるためのフラグ）。
#[cfg(target_os = "macos")]
const O_CREAT: i32 = 0x0200;

/// macOS（Darwin）の `O_EXCL`（`<fcntl.h>` では `0x0800`）。`O_CREAT` と
/// 併用し、対象が symlink を含め既に存在すれば `open` 自体を `EEXIST` で
/// 拒否する（新規作成のみを許可し、既存ファイル・symlink 先への
/// 上書きを構造的に防ぐ）。[`create_file_pinned`] 用。
#[cfg(target_os = "macos")]
const O_EXCL: i32 = 0x0800;

/// macOS（Darwin）の `O_WRONLY`（`<fcntl.h>` では `0x0001`）。
/// [`create_file_pinned`] 用。
#[cfg(target_os = "macos")]
const O_WRONLY: i32 = 0x0001;

/// macOS（Darwin）の `unlinkat(2)` `AT_REMOVEDIR`（`<fcntl.h>` では
/// `0x0080`）。[`unlinkat_raw`] 用（イシュー #509 PR #677 codex-review P0
/// 指摘対応）。指定時は対象を空ディレクトリとして `rmdir(2)` 相当で削除
/// し、非指定時は通常ファイル・symlink 自体を `unlink(2)` 相当で削除する。
#[cfg(target_os = "macos")]
const AT_REMOVEDIR: i32 = 0x0080;

/// macOS（Darwin）の `fcntl(2)` `F_GETPATH`（`<fcntl.h>` では `50`）。
/// [`real_path_of_fd`] 用（イシュー #509 PR #677 codex-review P0 再指摘
/// 対応: [`create_dir_all_verified`] macOS 版のコンテインメント再検証で、
/// Linux 版の `/proc/self/fd/<fd>`（[`proc_fd_path`]）に相当する
/// 「fd が指す実体の現在の絶対パス」を取得する唯一の標準手段として使う）。
#[cfg(target_os = "macos")]
const F_GETPATH: std::os::raw::c_int = 50;

/// `path` を `O_NOFOLLOW | O_NONBLOCK` 付きで開く（[`load_cache_entry_in`]
/// 用。イシュー #509 codex-review P0 指摘対応）。
///
/// 最終コンポーネントが symlink であれば `open(2)` 自体をカーネルが
/// 拒否するため、「symlink でないことを確認してから読む」という 2 手順
/// （`validate_cache_entry` → `read_to_string`）の間に symlink へ差し替え
/// られる TOCTOU を、検証と読み取りを 1 回のシステムコールへ結合する
/// ことで構造的に排除できる（返した [`fs::File`] ハンドルからそのまま
/// 読み取ること。パスへ再度アクセスしない）。`O_NONBLOCK` は FIFO 等の
/// 特殊ファイルに対する open のハングを防ぐ（[`O_NONBLOCK`] 定数の
/// コメント参照）。返した fd は必ず [`read_verified_cache_entry_file`]
/// へ渡し、そこで fd 経由の `fstat`（symlink を追跡しない）により
/// 「実際に通常ファイルか」を再検証してから読むこと（`O_NOFOLLOW` は
/// symlink のみを拒否し FIFO・デバイス等の他の特殊ファイル種別は拒否
/// しないため）。`custom_flags` は [`std::os::unix::fs::OpenOptionsExt`]
/// 経由の std 標準機能であり追加依存を要しない（[`O_NOFOLLOW`] 定数の
/// コメント参照）。
#[cfg(unix)]
// macOS 版 `load_cache_entry_in`（`#[cfg(all(unix, not(target_os =
// "linux")))]`）は `openat_nofollow` を使うため、macOS の通常ビルド
// （`#[cfg(test)]` を含まない lib コンパイル単位）では本関数を直接
// 呼ばない。`O_NOFOLLOW` 定数値そのものの直接回帰テスト
// （`open_nofollow_rejects_a_symlinked_path`。全 unix 共通で実行）専用の
// 到達経路として維持するため、Linux 以外では dead_code を抑止する
// （イシュー #509・`aarch64-apple-darwin` クロスチェックで判明）。
#[cfg_attr(
    not(target_os = "linux"),
    allow(
        dead_code,
        reason = "macOS では openat_nofollow に一本化されテスト専用の到達経路になるため"
    )
)]
fn open_nofollow(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(path)
}

/// Linux の `O_DIRECTORY`（`<fcntl.h>`。8 進数 `0200000`）。
/// [`open_dir_nofollow`] 用（イシュー #509 codex-review P0 再指摘対応:
/// [`create_dir_all_verified`]／[`load_cache_entry_in`] の fd 起点実装が
/// 対象を確実にディレクトリへ限定するために使う）。[`O_NOFOLLOW`] 定数と
/// 同じ理由で `libc`／`rustix` を使わず自前定義する。
#[cfg(target_os = "linux")]
const O_DIRECTORY: i32 = 0o200000;

/// macOS（Darwin）の `O_DIRECTORY`（`<fcntl.h>` では `0x100000`）。上記
/// Linux 版と同じ理由・同じ用途。
#[cfg(target_os = "macos")]
const O_DIRECTORY: i32 = 0x100000;

/// `path` を `O_NOFOLLOW | O_DIRECTORY` 付きで開く（[`create_dir_all_verified`]
/// ／[`load_cache_entry_in`] の Linux 版 fd pin 用。イシュー #509
/// codex-review P0 再指摘対応）。
///
/// 最終コンポーネントが symlink または非ディレクトリであれば `open(2)`
/// 自体が `ELOOP`／`ENOTDIR` で拒否する。返した [`fs::File`] は
/// [`proc_fd_path`] 経由でのみ以降アクセスし、元のパス文字列を再度
/// ルートから辿り直さない（`openat`／`mkdirat` 相当の効果を得るため。
/// [`proc_fd_path`] のドキュメンテーションコメント参照）。
#[cfg(unix)]
fn open_dir_nofollow(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_DIRECTORY)
        .open(path)
}

/// pin 済みディレクトリ fd を指す `/proc/self/fd/<fd>` magic path
/// （`proc(5)`）を返す（Linux 限定。イシュー #509 codex-review P0
/// 再指摘対応）。
///
/// `libc`／`rustix` の `openat`／`mkdirat`（fd 起点でパス再解決を避ける
/// システムコール）は許容依存 8 区分（`.claude/rules/deps-policy.md`）
/// 外のためユーザー承認なしに追加できない。Linux の `/proc/self/fd/<fd>`
/// は fd が指す実体（inode）への magic symlink で、その配下
/// （`/proc/self/fd/<fd>/name`）へのパス解決は「fd が指すディレクトリを
/// 起点に `name` の 1 コンポーネントだけを解決する」ため、`openat` と
/// 等価な効果を std のみ（`fs::File`／`PathBuf` の文字列組み立てのみ）
/// で得られる（`man 5 proc` の `/proc/[pid]/fd/` 節）。macOS の
/// `/dev/fd/<fd>`（`fdesc` ファイルシステム）は `<fd>` 自体をノードとして
/// 提供するのみで配下へのパス継続解決をサポートしないため、本手法は
/// `target_os = "linux"` に閉じる（[`create_dir_all_verified`]／
/// [`load_cache_entry_in`] の macOS 版は本手法を使わない別実装を持つ）。
#[cfg(target_os = "linux")]
fn proc_fd_path(dir: &fs::File) -> PathBuf {
    use std::os::unix::io::AsRawFd;
    PathBuf::from(format!("/proc/self/fd/{}", dir.as_raw_fd()))
}

// POSIX `openat(2)` の FFI 宣言（macOS 版 `load_cache_entry_in` 用。
// イシュー #509 codex-review P0 再指摘〈TOCTOU〉対応）。
//
// `libc`／`rustix` クレートは許容依存 8 区分（`.claude/rules/
// deps-policy.md`）外のためユーザー承認なしに追加できない（`O_NOFOLLOW`
// 定数のコメントと同じ制約）。一方 macOS の `/dev/fd/<fd>` は配下への
// パス継続解決をサポートしないため（`proc_fd_path` のドキュメン
// テーションコメント参照）、Linux 版と同じ「fd 経由でパス再解決を
// 避ける」手法を適用できない。そこで std のみでは到達できない `openat`
// システムコールを、新規クレートを追加せず `extern "C"` 宣言（libSystem
// が提供する libc シンボルへリンクする FFI 境界）で直接呼ぶ。`unsafe`
// は FFI 境界の必要最小限に留める方針（`.claude/rules/security.md`）に
// 沿い、本関数は `openat_nofollow` の内部でのみ使う（rustdoc は extern
// ブロックへの `///` ドキュメンテーションコメントを認識しないため
// `//` の通常コメントとする。`aarch64-apple-darwin` クロスチェックで
// unused-doc-comments として検出）。
//
// POSIX の実 C シグネチャは `int openat(int, const char *, int, ...)`
// という variadic 関数（`mode_t mode` は `O_CREAT`／`O_TMPFILE` 指定時
// のみ渡される可変引数）である。`...` を落とし固定 3 引数として宣言
// すると、可変引数を 1 つも渡さない呼び出しであっても Rust コンパイラ
// が非 variadic 呼び出し規約でコード生成しうる（イシュー #509 PR #677
// codex-review P0 再指摘: `aarch64-apple-darwin` の Apple 独自 ABI は
// variadic 呼び出しに対し固定引数と異なるスタック渡し規約を要求する
// ため、宣言と実体の不一致は未定義動作になりうる）。以下は `...` を
// 明示し、呼び出し側（[`openat_nofollow`]）も可変引数を 1 つも渡さず
// `mode` 相当の引数を渡さない（`O_CREAT` を指定しないため mode は
// 意味を持たない）ことで、実体の variadic 関数へ ABI 上安全に対応
// させる。
// 同ブロックへ `mkdirat`／`renameat`（イシュー #509 PR #677 codex-review
// P0 再指摘対応: [`create_subdir_pinned`]・[`rename_pinned`] macOS 版・
// [`create_dir_all_verified`] macOS 版が使う）を追加する。両者とも POSIX
// では固定引数の非 variadic 関数（`mkdirat` の `mode_t mode` は常に
// 必須引数、`renameat` に可変引数はない）であり、`openat` のような ABI
// 上の注意点は生じない。
//
// SAFETY: 3 関数とも libSystem（macOS の libc 実体）が提供する POSIX
// 標準シンボルであり、リンク時に解決される（`unsafe extern` の安全性は
// 宣言したシグネチャが実体の C ABI と一致することに懸かる）。`openat` は
// 上記コメントのとおり実体が `int openat(int, const char *, int, ...)` の
// variadic 関数であるため `...` を明示し、呼び出し側（[`openat_raw`]）は
// 可変引数の数を `O_CREAT` 有無で切り替えて ABI 不一致を避ける。
// `mkdirat`／`renameat` は POSIX で固定引数（可変引数なし）と規定される
// ため宣言どおりの固定引数で一致する。各関数のポインタ引数
// （`pathname`／`oldpath`／`newpath`）は呼び出し側が呼び出し中生存する
// NUL 終端 `CString` から取るため、宣言側で追加の生存期間契約は生じない
// （呼び出し側の SAFETY コメント参照: [`openat_raw`]・[`mkdirat_raw`]・
// [`renameat_raw`]）。
#[cfg(all(unix, not(target_os = "linux")))]
unsafe extern "C" {
    fn openat(
        dirfd: std::os::raw::c_int,
        pathname: *const std::os::raw::c_char,
        flags: std::os::raw::c_int,
        ...
    ) -> std::os::raw::c_int;

    fn mkdirat(
        dirfd: std::os::raw::c_int,
        pathname: *const std::os::raw::c_char,
        mode: ModeT,
    ) -> std::os::raw::c_int;

    fn renameat(
        olddirfd: std::os::raw::c_int,
        oldpath: *const std::os::raw::c_char,
        newdirfd: std::os::raw::c_int,
        newpath: *const std::os::raw::c_char,
    ) -> std::os::raw::c_int;

    // `unlinkat`（イシュー #509 PR #677 codex-review P0 指摘対応:
    // [`remove_child_pinned`] macOS 版が使う。`AT_REMOVEDIR` フラグ指定時は
    // 空ディレクトリの削除〈`rmdir(2)` 相当〉、非指定時は通常ファイル・
    // symlink 自体の削除〈`unlink(2)` 相当〉になる。POSIX で固定 3 引数の
    // 非 variadic 関数）。
    fn unlinkat(
        dirfd: std::os::raw::c_int,
        pathname: *const std::os::raw::c_char,
        flags: std::os::raw::c_int,
    ) -> std::os::raw::c_int;
}

/// macOS（Darwin）の `mode_t`（`<sys/_types/_mode_t.h>` では
/// `__uint16_t`）。[`mkdirat`]／[`openat`]（`O_CREAT` 使用時）の第 3 引数
/// 型として使う（イシュー #509 PR #677 codex-review P0 再指摘対応）。
#[cfg(all(unix, not(target_os = "linux")))]
type ModeT = u16;

/// pin 済みディレクトリ fd を起点に `openat(2)` を呼ぶ共通実装（イシュー
/// #509 PR #677 codex-review P0 再指摘対応。[`openat_nofollow`]（ファイル
/// 読み取り用）・[`opendirat_nofollow`]（ディレクトリ descent 用）・
/// [`create_file_pinned`] macOS 版（新規作成用）が共用する）。
///
/// `mode` は `flags` に `O_CREAT` を含む場合のみ `Some` を渡す（`openat`
/// は宣言上 variadic〈[`openat`] 宣言のコメント参照〉であり、実際に
/// `mode` を渡すかどうかで呼び出し側の可変引数の数を変える。`O_CREAT`
/// 非指定時に `mode` を渡すと POSIX 契約〈`mode` は `O_CREAT`／
/// `O_TMPFILE` 指定時のみ意味を持つ可変引数〉から外れるため、
/// `None`／`Some` で呼び出し形自体を分ける）。
#[cfg(all(unix, not(target_os = "linux")))]
fn openat_raw(
    dir: &fs::File,
    name: &str,
    flags: std::os::raw::c_int,
    mode: Option<ModeT>,
) -> std::io::Result<fs::File> {
    use std::ffi::CString;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    let c_name = CString::new(name).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry file name must not contain a NUL byte",
        )
    })?;

    // SAFETY: `dir` は呼び出し元が pin 済みの有効なディレクトリ fd で
    // 本呼び出しの間生存する。`c_name` は直前に構築した NUL 終端の有効な
    // C 文字列で `openat` 呼び出し中生存する（`CString` はこのスコープを
    // 抜けるまでドロップされない）。`openat` は POSIX 標準のシステム
    // コールで、`dirfd`（`dir` の fd 値）を起点に `pathname`（`name`）の
    // 1 コンポーネントのみを解決する（`man 2 openat`）。戻り値は新規 fd
    // （成功時 `>= 0`）または `-1`（失敗時。`errno` は
    // `std::io::Error::last_os_error()` で読む）であり、いずれも呼び出し
    // 直後にこの Rust コードへ制御を戻す前提を満たす。`mode` を渡す
    // 呼び出し（`Some`）は `c_uint` へ昇格済みの値を可変引数として渡す
    // （`O_CREAT` 指定時の ABI 上正しい形。[`openat`] 宣言のコメント
    // 参照）。`mode` なしの呼び出し（`None`）は可変引数を 1 つも渡さない。
    let fd = unsafe {
        match mode {
            Some(m) => openat(
                dir.as_raw_fd(),
                c_name.as_ptr(),
                flags,
                m as std::os::raw::c_uint,
            ),
            None => openat(dir.as_raw_fd(), c_name.as_ptr(), flags),
        }
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }

    // SAFETY: `fd` は直前の `openat` 呼び出しが正常終了時（`fd >= 0` を
    // 確認済み）に返した新規オープン fd で、他のどのハンドルにも所有
    // されていない。`fs::File::from_raw_fd` はその所有権を一度だけ
    // `fs::File` へ移す（以降のクローズ責務は返した `File` の Drop に
    // 委ねる）。
    Ok(unsafe { fs::File::from_raw_fd(fd) })
}

/// pin 済みディレクトリ fd（`open_dir_nofollow` で `O_NOFOLLOW |
/// O_DIRECTORY` 付きで開いたもの）を起点に、`name`（1 コンポーネントの
/// ファイル名。エントリ内は [`CACHE_ENTRY_SOURCE_FILE`]／
/// [`CACHE_ENTRY_PTX_FILE`] の固定文字列のみを渡す）を読み取り用に開く
/// （macOS 版 [`load_cache_entry_in`] 用。イシュー #509 codex-review P0
/// 再指摘対応）。
///
/// 呼び出し元は `dir` を一度だけ pin し、以降は本関数を介して `dir` を
/// dirfd（起点）とした 1 コンポーネントの解決のみを行う。`entry_dir` を
/// 表すパス文字列をルートから再度辿り直す経路が存在しないため、事前
/// 検証（`open_dir_nofollow` 成功）後に `entry_dir` 自体を別 symlink へ
/// 差し替える TOCTOU（旧実装は open 前後の `dev`/`ino` 比較という事後
/// 検出に留まっていた）が構造的に発生しない。`O_NOFOLLOW`（最終
/// コンポーネントの symlink を拒否）・`O_NONBLOCK`（FIFO 等特殊ファイルの
/// open ハング防止。[`O_NONBLOCK`] 定数参照）・`O_CLOEXEC`（std 経由の
/// open が既定で得る close-on-exec と同じ保証を揃える。[`O_CLOEXEC`]
/// 定数のコメント参照）を付与する。
#[cfg(all(unix, not(target_os = "linux")))]
fn openat_nofollow(dir: &fs::File, name: &str) -> std::io::Result<fs::File> {
    openat_raw(dir, name, O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, None)
}

/// pin 済みディレクトリ fd を起点に、`name` の 1 コンポーネントを
/// `O_NOFOLLOW | O_DIRECTORY` で開く（[`open_dir_nofollow`] の macOS
/// dirfd 起点版。イシュー #509 PR #677 codex-review P0 再指摘対応:
/// [`create_subdir_pinned`]・[`create_dir_all_verified`] macOS 版が、
/// ディレクトリ descent の各階層で「元のパス文字列をルートから再度
/// 辿り直さない」ために使う）。最終コンポーネントが symlink または
/// 非ディレクトリであれば `openat(2)` 自体が `ELOOP`／`ENOTDIR` で拒否
/// する。
#[cfg(all(unix, not(target_os = "linux")))]
fn opendirat_nofollow(dir: &fs::File, name: &str) -> std::io::Result<fs::File> {
    openat_raw(dir, name, O_NOFOLLOW | O_DIRECTORY | O_CLOEXEC, None)
}

/// pin 済みディレクトリ fd を起点に、`name` を新規ファイルとして排他
/// 生成する（`O_CREAT | O_EXCL`。[`create_subdir_pinned`]・
/// [`write_child_file_pinned`] 用。イシュー #509 PR #677 codex-review P0
/// 再指摘対応）。
///
/// `O_EXCL` により、対象が symlink を含め既に存在すれば `open` 自体が
/// `EEXIST` で失敗する（symlink 先への意図しない書き込みを構造的に防ぐ。
/// Linux 版 [`create_file_pinned`] の `OpenOptions::create_new` と同じ
/// 保証）。作成モードは `0o600`（所有者のみ読み書き。キャッシュエントリは
/// プロセス実行ユーザー以外が読む必要はない）。
#[cfg(all(unix, not(target_os = "linux")))]
fn create_file_pinned_at(dir: &fs::File, name: &str) -> std::io::Result<fs::File> {
    openat_raw(
        dir,
        name,
        O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC,
        Some(0o600),
    )
}

/// pin 済みディレクトリ fd を起点に `name` のサブディレクトリを作成する
/// （`mkdirat(2)` の FFI 直接呼び出し。イシュー #509 PR #677 codex-review
/// P0 再指摘対応。[`create_subdir_pinned`] macOS 版・[`create_dir_all_verified`]
/// macOS 版が使う）。
///
/// `mkdirat` は POSIX の固定引数関数（可変引数なし）であり [`openat`]
/// と異なり ABI 上の variadic 注意点はない。モードは `0o700`（所有者の
/// み読み書き実行。既存の `fs::create_dir`〈umask 適用前は既定で
/// パーミッションを渡さない〉と同水準の意図）。
#[cfg(all(unix, not(target_os = "linux")))]
fn mkdirat_raw(dir: &fs::File, name: &str) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::io::AsRawFd;

    let c_name = CString::new(name).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry directory name must not contain a NUL byte",
        )
    })?;

    // SAFETY: `dir` は呼び出し元が pin 済みの有効なディレクトリ fd で
    // 本呼び出しの間生存する。`c_name` は直前に構築した NUL 終端の有効な
    // C 文字列で呼び出し中生存する。`mkdirat` は POSIX 標準のシステム
    // コールで `dirfd` を起点に `pathname` の 1 コンポーネントのみを
    // 解決する（`man 2 mkdirat`）。戻り値は成功時 `0`、失敗時 `-1`
    // （`errno` は `std::io::Error::last_os_error()` で読む）。
    let ret = unsafe { mkdirat(dir.as_raw_fd(), c_name.as_ptr(), 0o700) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// pin 済みディレクトリ fd を起点に `renameat(2)` を呼ぶ（イシュー #509
/// PR #677 codex-review P0 再指摘対応。[`rename_pinned`] macOS 版が使う）。
///
/// `from_dir`／`to_dir` はいずれも呼び出し元が pin 済みの dirfd
/// （本モジュールでは常にキャッシュルート fd）。`renameat` は POSIX
/// 標準のシステムコールで、両方のパスをそれぞれの dirfd を起点に
/// 1 コンポーネントだけ解決するため、Linux 版
/// （[`proc_fd_path`] 経由の `fs::rename`）と同様にキャッシュルートの
/// パス文字列を再度ルートから辿り直すことがない。
#[cfg(all(unix, not(target_os = "linux")))]
fn renameat_raw(
    from_dir: &fs::File,
    from_name: &str,
    to_dir: &fs::File,
    to_name: &str,
) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::io::AsRawFd;

    let c_from = CString::new(from_name).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry directory name must not contain a NUL byte",
        )
    })?;
    let c_to = CString::new(to_name).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry directory name must not contain a NUL byte",
        )
    })?;

    // SAFETY: `from_dir`／`to_dir` は呼び出し元が pin 済みの有効な
    // ディレクトリ fd で本呼び出しの間生存する。`c_from`／`c_to` は
    // 直前に構築した NUL 終端の有効な C 文字列で呼び出し中生存する。
    // `renameat` は POSIX 標準のシステムコールで、戻り値は成功時 `0`、
    // 失敗時 `-1`（`errno` は `std::io::Error::last_os_error()` で読む）。
    let ret = unsafe {
        renameat(
            from_dir.as_raw_fd(),
            c_from.as_ptr(),
            to_dir.as_raw_fd(),
            c_to.as_ptr(),
        )
    };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// pin 済みディレクトリ fd を起点に `unlinkat(2)` を呼ぶ（イシュー #509
/// PR #677 codex-review P0 指摘対応。[`remove_child_pinned`] macOS 版が
/// 使う）。
///
/// `dir` は呼び出し元が pin 済みの dirfd（本モジュールでは常にキャッシュ
/// ルート fd、またはそこから 1 階層だけ辿ったエントリディレクトリ fd）。
/// `unlinkat` は POSIX 標準のシステムコールで `dirfd` を起点に `name` の
/// 1 コンポーネントのみを解決するため、Linux 版（[`proc_fd_path`] 経由の
/// `fs::remove_file`／`fs::remove_dir`）と同様にキャッシュルートのパス
/// 文字列を再度ルートから辿り直すことがない。`remove_dir` が真なら
/// `AT_REMOVEDIR` を付与し空ディレクトリの削除として扱う。
#[cfg(all(unix, not(target_os = "linux")))]
fn unlinkat_raw(dir: &fs::File, name: &str, remove_dir: bool) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::io::AsRawFd;

    let c_name = CString::new(name).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry name must not contain a NUL byte",
        )
    })?;

    let flags = if remove_dir { AT_REMOVEDIR } else { 0 };

    // SAFETY: `dir` は呼び出し元が pin 済みの有効なディレクトリ fd で
    // 本呼び出しの間生存する。`c_name` は直前に構築した NUL 終端の有効な
    // C 文字列で呼び出し中生存する。`unlinkat` は POSIX 標準のシステム
    // コールで、戻り値は成功時 `0`、失敗時 `-1`（`errno` は
    // `std::io::Error::last_os_error()` で読む）。
    let ret = unsafe { unlinkat(dir.as_raw_fd(), c_name.as_ptr(), flags) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

// macOS `fcntl(2)` `F_GETPATH` の FFI 宣言（[`real_path_of_fd`] 用。
// イシュー #509 PR #677 codex-review P0 再指摘対応）。`fcntl` は POSIX
// では variadic（`int fcntl(int, int, ...)`）だが、`F_GETPATH` の第 3
// 引数はポインタ型でありデフォルト引数昇格の対象外（整数・浮動小数点の
// 昇格ルールはポインタには適用されない）のため、`openat` の `mode`
// 引数のような ABI 上の追加注意点は生じない。
//
// SAFETY: libSystem が提供する POSIX 標準シンボルであり、リンク時に解決
// される。`fcntl` の実体は `int fcntl(int, int, ...)`（variadic）で宣言も
// `...` を明示済み。呼び出し側（[`real_path_of_fd`]）は `F_GETPATH` の
// 第 3 引数としてポインタ（バッファ先頭アドレス）のみを渡し、上記の
// とおりポインタは可変引数のデフォルト昇格対象外のため宣言・実体間の
// ABI 齟齬は生じない。
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn fcntl(fd: std::os::raw::c_int, cmd: std::os::raw::c_int, ...) -> std::os::raw::c_int;
}

/// `dir` fd が指す実体の現在の絶対パスを `F_GETPATH` で取得する（macOS
/// 限定。イシュー #509 PR #677 codex-review P0 再指摘対応:
/// [`create_dir_all_verified`] macOS 版のコンテインメント再検証用。
/// Linux 版が使う `/proc/self/fd/<fd>`〈[`proc_fd_path`]〉の symlink
/// 解決込み絶対パス取得に相当する、macOS で fd から実パスを得る唯一の
/// 標準的手段）。
///
/// バッファは `PATH_MAX`（macOS では `1024`。`<sys/syslimits.h>`）を
/// 確保する。`F_GETPATH` は成功時に NUL 終端済みの絶対パスをバッファへ
/// 書き込む。
#[cfg(target_os = "macos")]
fn real_path_of_fd(dir: &fs::File) -> std::io::Result<PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::io::AsRawFd;

    const PATH_MAX: usize = 1024;
    let mut buf = vec![0u8; PATH_MAX];

    // SAFETY: `dir` は呼び出し元が所有する有効なオープン fd。`buf` は
    // `PATH_MAX` バイト確保済みの書き込み可能バッファで、`fcntl` は
    // `F_GETPATH` 使用時に高々 `PATH_MAX` バイトしか書き込まない
    // （`man 2 fcntl` の `F_GETPATH` 節）。戻り値は成功時 `0`、失敗時
    // `-1`（`errno` は `std::io::Error::last_os_error()` で読む）。
    let ret = unsafe { fcntl(dir.as_raw_fd(), F_GETPATH, buf.as_mut_ptr()) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }

    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    buf.truncate(nul);
    Ok(PathBuf::from(std::ffi::OsString::from_vec(buf)))
}

/// キャッシュエントリ 1 ファイルあたりの読み込み上限（バイト数。イシュー
/// #509 codex-review P0 指摘対応）。
///
/// 書き込み可能なキャッシュディレクトリへ巨大ファイルを配置されるだけで
/// `read_to_string`／`read_to_end` が EOF まで無制限に読み込み OOM に
/// なりうる問題への対策。NVRTC が実際に生成する PTX・その変換元ソースは
/// 通常数百 KiB 程度に収まるため、想定される最大カーネルサイズに十分な
/// 余裕を持たせた値として 64 MiB を採用する。
#[cfg(unix)]
const MAX_CACHE_ENTRY_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// [`open_nofollow`]／[`openat_nofollow`] で開いた fd から、通常ファイル
/// であることと読み込みサイズ上限（[`MAX_CACHE_ENTRY_FILE_BYTES`]）を
/// 検証したうえでバイト列を読み込む（[`load_cache_entry_in`]（Linux／
/// macOS 版）共用。イシュー #509 codex-review P0 指摘 2 件対応）。
///
/// 1. **特殊ファイル拒否**: `file.metadata()`（fd 経由の `fstat`。パスを
///    再解決しないため symlink 差し替え TOCTOU の対象にならない）で
///    `file_type().is_file()` を検証する。`O_NOFOLLOW` は symlink のみを
///    拒否し FIFO・ソケット・デバイス等の他の特殊ファイル種別は拒否
///    しないため、`kernel.cu`／`kernel.ptx` として FIFO を配置され
///    `open` 自体は `O_NONBLOCK` により即座に成功してしまうケースを
///    ここで検出し拒否する。
/// 2. **サイズ上限**: 読み込み前に `metadata().len()` で事前検査した
///    うえで、実際の読み込みも `Read::take` で上限 + 1 バイトに制限する
///    （事前検査後にファイルが伸長される競合〈他プロセスが追記する等〉
///    への縦深防御）。
/// 3. **非空検査**: 読み込んだバイト列が空なら `Ok(None)`（ミス）とする
///    （実装計画 §3.1 の不変条件「両ファイルが存在し、いずれも空でない」。
///    クラッシュ直後の 0 バイト残骸〈`create` 直後・書き込み前に
///    プロセスが落ちた場合等〉を有効なエントリとして誤読しないための
///    fail-closed 判定。fd 経由で読み込み済みのバイト列そのものを検査
///    する〈別途 `metadata()` を取り直さない〉ため、この検査自体が新規の
///    TOCTOU 窓を開かない）。
///
/// いずれの拒否条件も「壊れたエントリはキャッシュミス扱いにする」という
/// 本モジュールの既存方針（[`load_cache_entry_in`] のドキュメンテーション
/// コメント参照）に合わせて `Ok(None)` を返す（エラーにはしない）。
/// 一方 I/O 自体の失敗（読み込み中のエラー等）は `Err` として呼び出し元
/// へ伝える（既存の `read_to_string` 呼び出しと同じ扱い）。
#[cfg(unix)]
fn read_verified_cache_entry_file(mut file: fs::File) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read;

    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Ok(None);
    }
    if meta.len() > MAX_CACHE_ENTRY_FILE_BYTES {
        return Ok(None);
    }

    let mut buf = Vec::new();
    (&mut file)
        .take(MAX_CACHE_ENTRY_FILE_BYTES + 1)
        .read_to_end(&mut buf)?;
    if buf.len() as u64 > MAX_CACHE_ENTRY_FILE_BYTES {
        return Ok(None);
    }
    if buf.is_empty() {
        return Ok(None);
    }

    Ok(Some(buf))
}

/// 一時ディレクトリ名のシーケンス番号（プロセス内一意性の担保。
/// [`temp_entry_dir_name`] が使う）。
static TEMP_ENTRY_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

/// [`store_cache_entry_in`] が使う一時ディレクトリ名を生成する。
///
/// 命名は `.tmp.<final_entry_name>.<pid>.<seq>`。先頭 `.` により
/// `kernel.*` エントリ名前空間（[`CudaKernelCacheKey::cache_entry_dir_name`]）
/// と衝突しない。`pid`（`std::process::id()`。プロセス間一意性）と
/// `seq`（[`TEMP_ENTRY_DIR_SEQ`]。プロセス内一意性）の組み合わせで、
/// 同一ルートへの並行書き込み（複数プロセス・同一プロセス内の複数
/// スレッド）が一時ディレクトリ名で衝突しないことを保証する（DeepGEMM
/// compiler の一時ディレクトリ方式に倣う。実装計画 §3.2）。
/// `final_entry_name` はキー検証済み（[`CudaKernelCacheKey::cache_entry_dir_name`]
/// の A03 トラバーサル防御を経由済み）の文字列のみを渡すこと。
fn temp_entry_dir_name(final_entry_name: &str) -> String {
    let seq = TEMP_ENTRY_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    format!(".tmp.{final_entry_name}.{}.{seq}", std::process::id())
}

/// キャッシュルート fd（[`open_dir_nofollow`] で pin 済み）を起点に
/// `name` の 1 コンポーネントのサブディレクトリを作成し、そのディレクトリ
/// fd を返す（[`write_entry_files_pinned`]・[`create_dir_all_verified`]
/// macOS 版が共用。イシュー #509 PR #677 codex-review P0 再指摘対応）。
///
/// Linux 版は [`proc_fd_path`]（`/proc/self/fd/<fd>/<name>`）経由で
/// [`fs::create_dir`] → [`open_dir_nofollow`] する（`load_cache_entry_in`
/// Linux 版・[`create_dir_all_verified`] Linux 版と同じ手法）。macOS 版は
/// [`mkdirat_raw`] → [`opendirat_nofollow`]（`openat(2)` FFI 直接呼び出し）
/// する。いずれも `parent` を起点とした fd 相対操作のみで完結し、`parent`
/// が指す実体を表すパス文字列をルートから再度辿り直すことがないため、
/// `parent` の pin 後にその祖先が symlink へ差し替えられても書き込み先が
/// 追従しない（`mkdirat`／`openat` と同等の効果。[`proc_fd_path`]・
/// [`openat_raw`] のドキュメンテーションコメント参照）。
#[cfg(unix)]
fn create_subdir_pinned(parent: &fs::File, name: &str) -> std::io::Result<fs::File> {
    #[cfg(target_os = "linux")]
    {
        let child_path = proc_fd_path(parent).join(name);
        fs::create_dir(&child_path)?;
        open_dir_nofollow(&child_path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        mkdirat_raw(parent, name)?;
        opendirat_nofollow(parent, name)
    }
}

/// 親ディレクトリ fd（[`create_subdir_pinned`] が返した一時ディレクトリ
/// fd）を起点に `name` を新規ファイルとして排他生成する（[`create_subdir_pinned`]
/// と対になる書き込み用プリミティブ。イシュー #509 PR #677 codex-review
/// P0 再指摘対応）。
///
/// Linux 版は [`proc_fd_path`] 経由で `OpenOptions::create_new`
/// （`O_CREAT | O_EXCL` 相当。既存の symlink・ファイルがあれば `EEXIST`
/// で拒否）＋ `O_NOFOLLOW`（最終コンポーネント自体が symlink の場合の
/// 追加拒否。`create_new` と合わせた縦深防御）で開く。macOS 版は
/// [`create_file_pinned_at`]（`openat(2)` FFI 直接呼び出し。同じ
/// `O_CREAT | O_EXCL | O_NOFOLLOW`）を使う。いずれも `parent.write(true)`
/// 相当の書き込み専用ハンドルを返す。
#[cfg(unix)]
fn create_file_pinned(parent: &fs::File, name: &str) -> std::io::Result<fs::File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let path = proc_fd_path(parent).join(name);
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(O_NOFOLLOW | O_NONBLOCK)
            .open(&path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        create_file_pinned_at(parent, name)
    }
}

/// `dir_fd` を起点に `name` の 1 コンポーネントを読み取り専用で開く
/// （[`validate_cache_entry_at`]・[`entry_exists_at`] 用。イシュー #509
/// PR #677 codex-review P0 再指摘対応）。
///
/// Linux 版は [`open_nofollow`]（[`proc_fd_path`] 経由）、macOS 版は
/// 既存の [`openat_nofollow`]（`load_cache_entry_in` macOS 版と同一実装）
/// をそのまま再利用する。
#[cfg(unix)]
fn open_file_child_nofollow(dir_fd: &fs::File, name: &str) -> std::io::Result<fs::File> {
    #[cfg(target_os = "linux")]
    {
        open_nofollow(&proc_fd_path(dir_fd).join(name))
    }
    #[cfg(not(target_os = "linux"))]
    {
        openat_nofollow(dir_fd, name)
    }
}

/// `dir_fd` を起点に `name` の 1 コンポーネントのサブディレクトリを
/// `O_NOFOLLOW | O_DIRECTORY` で開く（[`validate_cache_entry_at`]・
/// [`entry_exists_at`]・[`store_cache_entry_at`] 用。イシュー #509 PR
/// #677 codex-review P0 再指摘対応）。
#[cfg(unix)]
fn open_dir_child_nofollow(dir_fd: &fs::File, name: &str) -> std::io::Result<fs::File> {
    #[cfg(target_os = "linux")]
    {
        open_dir_nofollow(&proc_fd_path(dir_fd).join(name))
    }
    #[cfg(not(target_os = "linux"))]
    {
        opendirat_nofollow(dir_fd, name)
    }
}

/// `root_fd` 起点で `entry_name` に何らかの実体（ディレクトリ・通常
/// ファイル・symlink 等、種別を問わず）が存在するかを、パスを再解決せず
/// fd 相対の open のみで判定する（[`store_cache_entry_at`] 用。パス
/// ベース版 [`validate_cache_entry`] の存在確認に相当するが、`root_fd`
/// を pin した後にキャッシュルート自体が symlink へ差し替えられても、
/// `root_fd` が指す元の実体だけを見るため誤判定しない）。
///
/// [`open_file_child_nofollow`]（`O_DIRECTORY` を要求しない）で判定する
/// ため、ディレクトリ・通常ファイルいずれも `Ok` になる。symlink は
/// 最終コンポーネントの `O_NOFOLLOW` 拒否で `Err`（`ELOOP` 相当）になる
/// が「symlink が存在する」こと自体は事実のため、`NotFound` 以外の
/// エラーはすべて「占有中」とみなす（イシュー #509 PR #677 Bugbot 指摘
/// 〈Non-dir blocks cache replacement〉対応: 旧実装は [`open_dir_child_nofollow`]
/// 〈`O_DIRECTORY` 必須〉のみで判定していたため、最終エントリ名が
/// プレーンファイルや symlink で占有されている場合に「存在しない」と
/// 誤判定し、`store_cache_entry_at` の破損エントリ置換分岐〈受け入れ
/// 基準 2〉を素通りして常に `CudaError::CacheIo`〈恒久失敗〉に陥って
/// いた。fail-safe のため、判定不能なケースを「空いている」と誤認しない
/// 方向〈占有中扱い〉に倒す）。
#[cfg(unix)]
fn entry_exists_at(root_fd: &fs::File, entry_name: &str) -> bool {
    // まずディレクトリとして判定する（`O_DIRECTORY` 指定。旧実装から
    // そのまま踏襲する経路で、両プラットフォームで実測済みの安定した
    // 判定）。これにより「正規のキャッシュエントリ（常にディレクトリ）」
    // の既存の呼び出し元（[`validate_cache_entry_at`] 等）への挙動を
    // 変えない。
    if open_dir_child_nofollow(root_fd, entry_name).is_ok() {
        return true;
    }
    // ディレクトリでなければ、`O_DIRECTORY` を要求しない
    // [`open_file_child_nofollow`] で通常ファイル・symlink 等を含めた
    // 占有を検出する（`NotFound` のみ「空いている」とみなす。上記の
    // ドキュメンテーションコメント参照）。
    match open_file_child_nofollow(root_fd, entry_name) {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// `root_fd` 起点で `entry_name` のキャッシュエントリが不変条件
/// （[`CACHE_ENTRY_SOURCE_FILE`]／[`CACHE_ENTRY_PTX_FILE`] の両方が
/// 存在し、いずれも symlink でない非空の通常ファイルであること）を
/// 満たすかを、パスを再解決せず fd 相対の open のみで判定する
/// （[`store_cache_entry_at`] 用。イシュー #509 PR #677 codex-review P0
/// 再指摘対応。パスベース版 [`validate_cache_entry`] の fd 版）。
#[cfg(unix)]
fn validate_cache_entry_at(root_fd: &fs::File, entry_name: &str) -> bool {
    let dir_fd = match open_dir_child_nofollow(root_fd, entry_name) {
        Ok(f) => f,
        Err(_) => return false,
    };
    is_plain_file_at(&dir_fd, CACHE_ENTRY_SOURCE_FILE)
        && is_plain_file_at(&dir_fd, CACHE_ENTRY_PTX_FILE)
}

/// `dir_fd` 起点で `name` が symlink ではない非空の通常ファイルかを、
/// fd 経由の `fstat`（`open_file_child_nofollow` が成功した時点で
/// symlink・特殊ファイルは既に拒否済み。[`open_nofollow`]・
/// [`openat_nofollow`] のドキュメンテーションコメント参照）で判定する
/// （[`validate_cache_entry_at`] 用）。
#[cfg(unix)]
fn is_plain_file_at(dir_fd: &fs::File, name: &str) -> bool {
    match open_file_child_nofollow(dir_fd, name) {
        Ok(f) => f
            .metadata()
            .map(|m| m.file_type().is_file() && m.len() > 0)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// `from_dir_fd`／`to_dir_fd` を起点に `from_name` から `to_name` へ
/// アトミックに rename する（[`store_cache_entry_at`] 用。イシュー #509
/// PR #677 codex-review P0 再指摘対応。両 fd が同一キャッシュルートを
/// 指す場合、Linux/macOS いずれも通常の同一ファイルシステム内 rename と
/// 同じくアトミック）。
///
/// Linux 版は [`proc_fd_path`] 経由で両端を magic path 化した
/// [`fs::rename`]（`rename(2)` はどちらの引数もパス解決するが、
/// `/proc/self/fd/<fd>/<name>` は fd が指す実体を直接指すため、キャッシュ
/// ルートのパス文字列を再度ルートから辿り直すことがない）。macOS 版は
/// [`renameat_raw`]（`renameat(2)` FFI 直接呼び出し）を使う。
#[cfg(unix)]
fn rename_pinned(
    from_dir_fd: &fs::File,
    from_name: &str,
    to_dir_fd: &fs::File,
    to_name: &str,
) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        fs::rename(
            proc_fd_path(from_dir_fd).join(from_name),
            proc_fd_path(to_dir_fd).join(to_name),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        renameat_raw(from_dir_fd, from_name, to_dir_fd, to_name)
    }
}

/// `dir_fd` を起点に `name` の 1 コンポーネントを削除する（[`remove_cache_entry_pinned`]
/// が使う削除プリミティブ。イシュー #509 PR #677 codex-review P0 指摘
/// 対応: 後始末〈一時ディレクトリ・退避ディレクトリの削除〉を `root_fd`
/// 起点の fd 相対操作へ統一するための最小単位）。
///
/// Linux 版は [`proc_fd_path`] 経由の [`fs::remove_file`]／[`fs::remove_dir`]
/// （`/proc/self/fd/<fd>/<name>` は fd が指す実体を起点に `name` の 1
/// コンポーネントだけを解決するため、[`rename_pinned`]・
/// [`create_subdir_pinned`] 等の既存 fd 相対プリミティブと同じ手法。
/// [`proc_fd_path`] のドキュメンテーションコメント参照）。macOS 版は
/// [`unlinkat_raw`]（`unlinkat(2)` FFI 直接呼び出し）を使う。`is_dir` が
/// 真の場合は空ディレクトリの削除（`rmdir(2)` 相当）、偽の場合は通常
/// ファイル・symlink 自体の削除（`unlink(2)` 相当）として扱う。
#[cfg(unix)]
fn remove_child_pinned(dir_fd: &fs::File, name: &str, is_dir: bool) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let path = proc_fd_path(dir_fd).join(name);
        if is_dir {
            fs::remove_dir(&path)
        } else {
            fs::remove_file(&path)
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        unlinkat_raw(dir_fd, name, is_dir)
    }
}

/// `root_fd` 起点で `name`（[`store_cache_entry_at`] の `tmp_name`／
/// `stale_name`）が指す実体を best-effort に削除する（イシュー #509 PR
/// #677 codex-review P0 指摘対応:
/// [`fs::remove_dir_all`]（`root.join(name)` によるパス再解決）・
/// 旧 `remove_cache_entry_best_effort` の置き換え。`root_fd` を pin した
/// 後に `root` 自体が別ディレクトリへの symlink に差し替えられても、
/// `root_fd` から辿った実体だけを削除対象にするため、攻撃者が用意した
/// symlink 先のディレクトリ・ファイルを誤って削除しない）。
///
/// [`store_cache_entry_at`] が扱う削除対象は常にディレクトリ（中身は高々
/// [`CACHE_ENTRY_SOURCE_FILE`]／[`CACHE_ENTRY_PTX_FILE`] の 2 ファイルのみ。
/// [`write_entry_files_pinned`] の書き込み契約、または元は正常なキャッシュ
/// エントリだった退避コピー）か、非ディレクトリ（イシュー #509 PR #677
/// Bugbot 指摘〈Non-dir blocks cache replacement〉対応で置換対象になった
/// 通常ファイル・symlink）のいずれかである。ディレクトリの場合は既知の 2
/// ファイル名のみを対象に子を削除してから親ディレクトリ自体を削除する
/// （未知のファイルが残っていれば `remove_dir` が `ENOTEMPTY` 相当で失敗
/// し当該ディレクトリは残置されるが、削除失敗はいずれも呼び出し元が
/// best-effort として無視する契約であり、無条件の再帰削除〈子孫を limit
/// なく辿る〉より安全側に倒す）。
#[cfg(unix)]
fn remove_cache_entry_pinned(root_fd: &fs::File, name: &str) {
    match open_dir_child_nofollow(root_fd, name) {
        Ok(dir_fd) => {
            let _ = remove_child_pinned(&dir_fd, CACHE_ENTRY_SOURCE_FILE, false);
            let _ = remove_child_pinned(&dir_fd, CACHE_ENTRY_PTX_FILE, false);
            drop(dir_fd);
            let _ = remove_child_pinned(root_fd, name, true);
        }
        Err(_) => {
            // ディレクトリでなければ通常ファイル・symlink として直接
            // 削除する（[`entry_exists_at`] の非ディレクトリ占有判定と
            // 対称の扱い）。存在しない場合（そもそも作られなかった等）も
            // ここへ来るが、削除失敗は best-effort として無視される。
            let _ = remove_child_pinned(root_fd, name, false);
        }
    }
}

/// [`store_cache_entry_at`] の手順 1〜3（一時ディレクトリ作成・
/// `kernel.cu`／`kernel.ptx` 書き込み・各ファイルの fsync・一時
/// ディレクトリ自体の fsync）を、キャッシュルート fd（`root_fd`）起点の
/// fd 相対操作のみで担う（旧パスベース実装からの fd 相対化。イシュー
/// #509 PR #677 codex-review P0 指摘対応: パスベース版は `root_fd` を
/// pin した後でも `tmp_dir`（`root.join(..)`）という文字列パスを経由
/// して `fs::create_dir`／`fs::write` するため、pin 後にキャッシュ
/// ルート自体が symlink へ差し替えられると containment 検証済みの
/// 実体の外へ書き込みうる TOCTOU が残っていた。
/// 本関数は一時ディレクトリの作成〈[`create_subdir_pinned`]〉・
/// 2 ファイルの生成〈[`create_file_pinned`]〉・fsync（書き込みに使った
/// fd から直接 `sync_all`。パス経由の再オープンをしない）まで、すべて
/// `root_fd` から辿った fd のみで行う）。
///
/// `tmp_name` は [`temp_entry_dir_name`] が返す一意な名前
/// （[`create_subdir_pinned`] は排他的作成のため、一意性が壊れていれば
/// 本関数が `Err` を返す。サイレントな上書きを避ける）。
#[cfg(unix)]
fn write_entry_files_pinned(
    root_fd: &fs::File,
    tmp_name: &str,
    kernel_cu: &str,
    kernel_ptx: &str,
) -> Result<(), CudaError> {
    let tmp_dir_fd = create_subdir_pinned(root_fd, tmp_name).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to create temp cache directory {tmp_name}: {e}"),
    })?;

    write_child_file_pinned(&tmp_dir_fd, CACHE_ENTRY_SOURCE_FILE, kernel_cu)?;
    write_child_file_pinned(&tmp_dir_fd, CACHE_ENTRY_PTX_FILE, kernel_ptx)?;

    tmp_dir_fd.sync_all().map_err(|e| CudaError::CacheIo {
        detail: format!("failed to fsync temp cache directory {tmp_name}: {e}"),
    })
}

/// [`write_entry_files_pinned`] の内側で `name` の 1 ファイルを作成・
/// 書き込み・fsync する（イシュー #509 PR #677 codex-review P0 再指摘
/// 対応）。書き込みに使った [`fs::File`] ハンドルから直接 `sync_all` する
/// ため、パス経由の再オープン（symlink 差し替え TOCTOU の窓）が生じない。
#[cfg(unix)]
fn write_child_file_pinned(dir_fd: &fs::File, name: &str, content: &str) -> Result<(), CudaError> {
    use std::io::Write;

    let mut file = create_file_pinned(dir_fd, name).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to create {name}: {e}"),
    })?;
    file.write_all(content.as_bytes())
        .map_err(|e| CudaError::CacheIo {
            detail: format!("failed to write {name}: {e}"),
        })?;
    file.sync_all().map_err(|e| CudaError::CacheIo {
        detail: format!("failed to fsync {name}: {e}"),
    })
}

/// [`cache_root`] が解決した候補パスを実際に fs 上へ実体化し、symlink
/// 解決込みの containment 再検証を行う（イシュー #509・Phase C-3。
/// C-2 が `docs/cuda-jit-cache-design.md` 「残課題」節で明示的に委譲した
/// 「symlink 解決込みの再検証」を実装する）。
///
/// `candidate_root` は [`cache_root`]（環境変数解決＋字句正規化のみの
/// containment 検証）の戻り値を渡す想定。C-2 時点の字句正規化のみの
/// 検証は、`candidate_root` の祖先ディレクトリが symlink 経由で
/// `workspace_root` 配下を指すケース（例: `~/.cache` が
/// `<workspace_root>/evil` への symlink）を見逃す（字句上は
/// `workspace_root` 配下に見えないため）。本関数はまず fs 上に実在する
/// 最長祖先（[`longest_existing_ancestor`]）を [`Path::canonicalize`]
/// （symlink 解決込み）して containment を検証してから、未作成の
/// 残りコンポーネントを [`create_dir_all_verified`]（fd pin 起点で
/// 1 コンポーネントずつ作成・検証を結合する。イシュー #509 codex-review
/// P0 再指摘対応: 「作成してから検証」の順序では拒否確定前に workspace
/// 内へ書き込みが発生しうるため、検証と作成を同じディレクトリハンドル
/// へ結び付ける）で実体化する。ディレクトリを実際に作成した後でなければ
/// 最終的な symlink 解決ができない（存在しないパスは `canonicalize` が
/// `Err` を返す）ため、この再検証は fs I/O を行わない C-2 の純関数群では
/// 実行できず C-3（本関数）のスコープとされていた。
///
/// テスト容易性のため `candidate_root`（実体化対象）と `workspace_root`
/// （containment 検証の基準）を分離した「注入で決定化」パターン
/// （[`cache_entry_path_in`] と同型）を採る。公開ラッパー
/// [`ensure_cache_root`] は `candidate_root` を [`cache_root`]（実環境変数
/// 依存）から求めるため、実 `HOME`／`XDG_CACHE_HOME` に依存させず・
/// 実プロセス環境変数を書き換えず（並行テスト実行時の競合を避ける）に
/// containment 再検証（特に symlink 経由の拒否）をテストするには本関数を
/// 直接呼ぶ。
fn ensure_cache_root_in(
    candidate_root: &Path,
    workspace_root: &Path,
) -> Result<PathBuf, CudaError> {
    // `workspace_root` 自体が存在しない（呼び出し元がまだ何も書き込んで
    // いないビルドディレクトリ等）場合は canonicalize できないため、
    // その場合のみ字句正規化した値へフォールバックする（C-2 の
    // `resolve_cache_root` と同じ判断: containment 検証の基準側が
    // 存在しない場合、字句比較で近似する）。
    let canonical_workspace_root = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| lexically_normalize(workspace_root));

    // 実体化（`fs::create_dir_all`）より前の containment 事前検証
    // （イシュー #509 codex-review P0 指摘対応）: `candidate_root` 自身は
    // まだ存在しない場合が多く canonicalize できないため、fs 上に実在
    // する最長の祖先（[`longest_existing_ancestor`]）を symlink 解決込み
    // で先に検証する。ここで拒否すれば `fs::create_dir_all` は一切呼ばれ
    // ない。旧実装は「作成してから canonicalize して検証」という順序
    // だったため、workspace 外の祖先 symlink が workspace 内を指す
    // ケース（例: `~/.cache` が `<workspace_root>/evil` への symlink）で
    // 「実際に workspace 内へディレクトリを作成した後」に初めて拒否して
    // おり、containment 契約を副作用の時点で破っていた。
    let existing_ancestor =
        longest_existing_ancestor(candidate_root).ok_or_else(|| CudaError::CacheIo {
            detail: format!(
                "failed to resolve any existing ancestor of cache root candidate {}",
                candidate_root.display()
            ),
        })?;
    let canonical_existing_ancestor =
        existing_ancestor
            .canonicalize()
            .map_err(|e| CudaError::CacheIo {
                detail: format!(
                    "failed to canonicalize existing ancestor {} of cache root candidate {}: {e}",
                    existing_ancestor.display(),
                    candidate_root.display()
                ),
            })?;
    if path_lexically_within(&canonical_existing_ancestor, &canonical_workspace_root) {
        return Err(CudaError::CacheDirUnavailable {
            detail: format!(
                "cache root candidate {} has an existing ancestor {} that resolves (after \
                 symlink resolution) within workspace root {}; refusing to create any \
                 directory under it",
                candidate_root.display(),
                canonical_existing_ancestor.display(),
                canonical_workspace_root.display()
            ),
        });
    }

    create_dir_all_verified(
        &existing_ancestor,
        &canonical_existing_ancestor,
        candidate_root,
        &canonical_workspace_root,
    )?;

    let canonical_root = candidate_root
        .canonicalize()
        .map_err(|e| CudaError::CacheIo {
            detail: format!(
                "failed to canonicalize cache root directory {}: {e}",
                candidate_root.display()
            ),
        })?;

    // 事後の再検証（TOCTOU に対する縦深防御）: 上記の事前検証と
    // `fs::create_dir_all` の間で祖先が symlink に差し替えられる競合が
    // 万一あっても、最終的に返す `canonical_root` 自体を再検証すること
    // で fail-closed に倒す（事前検証だけに頼らない多層防御）。
    if path_lexically_within(&canonical_root, &canonical_workspace_root) {
        return Err(CudaError::CacheDirUnavailable {
            detail: format!(
                "cache root {} resolves (after symlink resolution) within workspace root {}",
                canonical_root.display(),
                canonical_workspace_root.display()
            ),
        });
    }

    Ok(canonical_root)
}

/// `path` 自身または祖先のうち、fs 上に実在する最も長い（`path` に最も
/// 近い）パスを返す（[`ensure_cache_root_in`] の実体化前 containment
/// 事前検証用。イシュー #509 codex-review P0 指摘対応）。`path` が絶対
/// パスであれば必ずルート（`/`）は実在するため、絶対パスに対しては
/// `None` を返さない。相対パスかつどの祖先も存在しない極端なケースの
/// みフォールバックとして `None` を返す（呼び出し元は `CacheIo` で
/// fail-closed に扱う）。
fn longest_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    loop {
        if current.exists() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

/// [`open_dir_nofollow`]／[`opendirat_nofollow`] の失敗が「対象が確実に
/// plain directory ではない」（symlink または非ディレクトリ）ことに
/// 起因するかを判定する（イシュー #509 PR #677 Cursor Bugbot 再指摘
/// 対応: `create_dir_all_verified` の Linux 版・macOS 版いずれも、旧実装
/// は open 失敗の原因種別を区別せず一律に「非ディレクトリ」とみなして
/// `remove_dir` していた。しかし `EMFILE`〈fd 枯渇〉・`EACCES`〈権限
/// 不足〉のような一時的・権限系エラーでも `open`／`openat` は失敗し
/// うる一方、`rmdir` は新規 fd を必要としないため成功してしまい、
/// 正当な peer プロセスが作成した空ディレクトリを誤って削除しうる）。
///
/// `O_NOFOLLOW` は symlink を `ELOOP` で、`O_DIRECTORY` は非ディレクトリ
/// を `ENOTDIR` で拒否する（`open(2)`）。この 2 種類の errno のみを
/// 「確実に非ディレクトリ」と判断し、それ以外（fd 枯渇・権限不足・
/// シグナル割り込み等）は削除せず [`CudaError::CacheIo`] として呼び
/// 出し元へそのまま伝播する（削除は「確実に非ディレクトリと判明した
/// 場合」のみに限定する fail-safe 方針）。`std::io::ErrorKind::
/// FilesystemLoop`／`NotADirectory`（`io_error_more`）は本ワークスペース
/// の stable toolchain では未安定化のため使えず、[`ELOOP`]／[`ENOTDIR`]
/// （`target_os` ごとの errno 生値）と `raw_os_error()` を直接比較する。
#[cfg(unix)]
fn is_confirmed_non_directory_open_error(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(ENOTDIR) | Some(ELOOP))
}

/// `existing_ancestor`（[`longest_existing_ancestor`] で求めた、事前検証
/// 済みの実在祖先）から `candidate_root` まで、パスコンポーネントを
/// 1 階層ずつ fd 起点（`/proc/self/fd/<fd>`）で作成し、作成直後の fd
/// そのものから containment を再検証する（[`ensure_cache_root_in`] 用。
/// イシュー #509 codex-review P0 再指摘対応。Linux 版）。
///
/// 旧実装（[`fs::create_dir_all`] で一括作成 → 事後検証、および本ファイル
/// `cfg(not(target_os = "linux"))` 分岐に残る「1 コンポーネントずつ
/// [`fs::create_dir`] → `fs::symlink_metadata` → `canonicalize` で検証」
/// 版）は、いずれもパス文字列を毎回ルートから再解決するため、直前の
/// コンポーネント検証と次のコンポーネント作成の間に祖先が symlink へ
/// 差し替えられる TOCTOU が残っていた（検出は事後にしかできず「拒否対象
/// への書き込み自体が起きない構造」にはなっていなかった。イシュー #509
/// codex-review 再指摘）。
///
/// 本関数は `open_dir_nofollow`（`O_NOFOLLOW | O_DIRECTORY`）で開いた
/// 直前のコンポーネントの fd を「現在位置」として保持し、次のコンポー
/// ネントは常に [`proc_fd_path`]（`/proc/self/fd/<fd>/<name>`）経由で
/// のみ [`fs::create_dir`]・オープンする。`/proc/self/fd/<fd>` は fd が
/// 指す実体を直接指すため、元のパス文字列を再度ルートから辿り直すこと
/// がなく、途中のコンポーネントが symlink へ差し替えられても「1 つ前に
/// 自分が pin した fd」の配下という参照だけが使われる。`mkdir` は対象が
/// 既に symlink であれば `EEXIST` で失敗し symlink を辿って中身を作る
/// ことはなく、続く `open_dir_nofollow` も symlink／非ディレクトリを
/// `ELOOP`／`ENOTDIR` で拒否するため、**拒否時点で次の階層への書き込み
/// は一切発生していない**（`openat`／`mkdirat` と同等の効果を std のみ
/// で得る。[`proc_fd_path`] のドキュメンテーションコメント参照）。
///
/// `libc`／`rustix` の `openat`／`mkdirat` は許容依存 8 区分
/// （`.claude/rules/deps-policy.md`）外のためユーザー承認なしに追加
/// できない。
///
/// `existing_ancestor`（fd pin の起点として使う `canonical_existing_ancestor`
/// とは別に、`candidate_root` からの相対コンポーネントを求める字句上の
/// 基準として渡す）自身が正当な symlink（例: `~/.cache` や
/// `XDG_CACHE_HOME` 自体が workspace 外の実ディレクトリを指す一般的な
/// 構成）である場合、fd pin を `existing_ancestor` に対して直接
/// `open_dir_nofollow`（`O_NOFOLLOW`）で行うと、最終コンポーネントが
/// symlink であること自体を理由に `ELOOP` で拒否してしまい、正当な
/// 構成を利用不能にする（イシュー #509 codex-review P1 再指摘対応。
/// Linux のみ・macOS 版は元々 `existing_ancestor` を文字列連結の起点
/// として使うだけで `O_NOFOLLOW` open を行わないため影響しない）。
#[cfg(target_os = "linux")]
fn create_dir_all_verified(
    existing_ancestor: &Path,
    canonical_existing_ancestor: &Path,
    candidate_root: &Path,
    canonical_workspace_root: &Path,
) -> Result<(), CudaError> {
    let relative = candidate_root
        .strip_prefix(existing_ancestor)
        .map_err(|_| CudaError::CacheIo {
            detail: format!(
                "cache root candidate {} is not nested under its existing ancestor {}",
                candidate_root.display(),
                existing_ancestor.display()
            ),
        })?;

    // fd pin の起点には `existing_ancestor`（字句上のパス。symlink で
    // あり得る）ではなく `canonical_existing_ancestor`（呼び出し元
    // [`ensure_cache_root_in`] が `Path::canonicalize` で symlink 解決
    // 済み・containment 検証済みの実在ディレクトリ）を使う。canonicalize
    // 済みパスは構成コンポーネントに symlink を含まないため
    // `open_dir_nofollow` の `O_NOFOLLOW` が正当な symlink 祖先を誤って
    // 拒否することがない（上記ドキュメンテーションコメント参照）。
    let mut current_dir =
        open_dir_nofollow(canonical_existing_ancestor).map_err(|e| CudaError::CacheIo {
            detail: format!(
                "failed to open existing cache directory ancestor {} (canonical: {}): {e}",
                existing_ancestor.display(),
                canonical_existing_ancestor.display()
            ),
        })?;
    // 表示・削除用の人間可読パス（セキュリティ判断には使わない。
    // 実際のオープン・作成は常に `current_dir` 起点の magic path で行う）。
    let mut display_path = existing_ancestor.to_path_buf();

    for component in relative.components() {
        let Component::Normal(name) = component else {
            // `existing_ancestor` は fs 上に実在するパスの祖先探索結果
            // であり、`candidate_root` は環境変数由来の絶対パスを結合
            // したもの（[`resolve_cache_root`]）のため、ここに `..`／`.`／
            // プレフィックス等の非正規コンポーネントが現れることは想定
            // しない。防御的に fail-closed で拒否する。
            return Err(CudaError::CacheIo {
                detail: format!(
                    "cache root candidate {} contains a non-normal path component",
                    candidate_root.display()
                ),
            });
        };
        display_path.push(name);

        let child_magic_path = proc_fd_path(&current_dir).join(name);

        if let Err(e) = fs::create_dir(&child_magic_path)
            && e.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(CudaError::CacheIo {
                detail: format!(
                    "failed to create cache directory component {}: {e}",
                    display_path.display()
                ),
            });
        }
        // `AlreadyExists`（並行する他プロセスが同じ祖先を先に作成した等）
        // は下記の再検証へフォールスルーし、既存の中身が正規ディレクト
        // リ・workspace 外であることを確認する。

        let child_dir = match open_dir_nofollow(&child_magic_path) {
            Ok(f) => f,
            Err(e) if is_confirmed_non_directory_open_error(&e) => {
                let _ = fs::remove_dir(&display_path);
                return Err(CudaError::CacheDirUnavailable {
                    detail: format!(
                        "cache directory component {} is not a plain directory (symlink or \
                         other non-directory found where a directory was expected)",
                        display_path.display()
                    ),
                });
            }
            Err(e) => {
                // `EMFILE`／`EACCES` 等、対象が非ディレクトリだと確定
                // していないエラー。`remove_dir` を実行せずそのまま
                // 伝播する（[`is_confirmed_non_directory_open_error`]
                // のドキュメンテーションコメント参照）。
                return Err(CudaError::CacheIo {
                    detail: format!(
                        "failed to open cache directory component {} after creation: {e}",
                        display_path.display()
                    ),
                });
            }
        };

        // containment 再検証: pin した `child_dir` fd が実際に指す実体の
        // 絶対パスを `/proc/self/fd/<fd>` の symlink 解決（`read_link`）
        // で取得する。fd は既に pin 済みのため、この `read_link` 自体は
        // 「今開いている実体」を報告するだけで再解決の余地はない。
        let canonical_current =
            fs::read_link(proc_fd_path(&child_dir)).map_err(|e| CudaError::CacheIo {
                detail: format!(
                    "failed to resolve real path of cache directory component {} via /proc: {e}",
                    display_path.display()
                ),
            })?;
        if path_lexically_within(&canonical_current, canonical_workspace_root) {
            let _ = fs::remove_dir(&display_path);
            return Err(CudaError::CacheDirUnavailable {
                detail: format!(
                    "cache directory component {} resolves (after symlink resolution) within \
                     workspace root {}; refusing to continue creating directories under it",
                    canonical_current.display(),
                    canonical_workspace_root.display()
                ),
            });
        }

        current_dir = child_dir;
    }

    Ok(())
}

/// [`create_dir_all_verified`]（Linux 版）の macOS 向け fd 起点実装
/// （イシュー #509 PR #677 codex-review P0 再指摘対応）。
///
/// macOS の `/dev/fd/<fd>`（`fdesc` ファイルシステム）は `<fd>` 自体を
/// ノードとして提供するのみで配下（`/dev/fd/<fd>/name`）へのパス継続
/// 解決をサポートしないため、Linux 版が使う `/proc/self/fd/<fd>` 相当の
/// magic path 手法は適用できない（[`proc_fd_path`] のドキュメンテーション
/// コメント参照）。旧実装（1 コンポーネントずつ [`fs::create_dir`]／
/// [`fs::symlink_metadata`]／[`Path::canonicalize`] で扱う検出型の縦深
/// 防御）は、直前のコンポーネント検証と次のコンポーネント作成の間に
/// 祖先が symlink へ差し替えられる TOCTOU が残っていた（Linux 版の旧
/// 実装が抱えていたのと同じ問題。イシュー #509 PR #677 codex-review P0
/// 再指摘）。
///
/// 本実装は [`mkdirat_raw`]／[`opendirat_nofollow`]（`mkdirat(2)`／
/// `openat(2)` FFI 直接呼び出し）で「現在位置」のディレクトリ fd
/// （`current_dir`）を起点とした dirfd 相対操作のみを行い、次の
/// コンポーネントへは常に直前に pin した fd 経由でのみ進む。`mkdirat`
/// は対象が既に symlink であれば `EEXIST` で失敗し symlink を辿って
/// 中身を作ることはなく、続く `opendirat_nofollow` も symlink／非
/// ディレクトリを `ELOOP`／`ENOTDIR` で拒否するため、Linux 版
/// （[`proc_fd_path`] 経由）と同様に**拒否時点で次の階層への書き込みは
/// 一切発生していない**。
///
/// containment 再検証は各階層で [`real_path_of_fd`]（`fcntl(2)`
/// `F_GETPATH`）を使う。Linux 版が使う [`proc_fd_path`] の
/// `fs::read_link`（symlink 解決込みの絶対パス取得）に相当する、fd から
/// 実パスを得る唯一の標準的手段である（[`real_path_of_fd`] の
/// ドキュメンテーションコメント参照）。パスベースの `canonicalize` を
/// 使わない理由: `current_dir` を pin した後に別プロセスが途中の祖先を
/// symlink へ差し替えても、`display_path`（表示専用の文字列パス）を
/// `canonicalize` すると差し替え後の偽の実体を検証してしまい、実際に
/// 書き込んだ fd 起点の実体（`workspace_root` 外に留まっている）とは
/// 別のものを見てしまう。`F_GETPATH` は fd が指す実体そのものの現在の
/// 絶対パスを返すため、この乖離が生じない。
///
/// `existing_ancestor`（Linux 版の P1 再指摘と同じ理由で、正当な
/// symlink〈`~/.cache`・`XDG_CACHE_HOME` 自体が workspace 外の実
/// ディレクトリを指す一般的な構成〉でありうる）は `O_NOFOLLOW` なしの
/// 通常 `fs::File::open` で最初の 1 段だけ開く（`opendirat_nofollow`
/// 〈`O_NOFOLLOW` 付き〉を最終コンポーネントへ直接使うと正当な構成を
/// `ELOOP` で誤って拒否してしまうため）。`canonical_existing_ancestor`
/// は Linux 版が fd pin の起点として使う symlink 解決済みパスだが、
/// 本実装は `existing_ancestor` を `O_NOFOLLOW` で開かないため未使用
/// （呼び出し元 [`ensure_cache_root_in`] を cfg 分岐させずに共有する
/// ため引数だけ受け取る）。
///
/// 拒否時に [`fs::remove_dir`] で削除するのはベストエフォートの
/// パスベースクリーンアップに留める（削除失敗はそれ自体をエラーに
/// せず元エラーを優先する。[`store_cache_entry_at`] の一時ディレクトリ
/// 削除と同じ方針: 削除は containment 突破の経路ではなく、失敗しても
/// 高々空ディレクトリが残るだけであるため path ベースのままでよい）。
#[cfg(all(unix, not(target_os = "linux")))]
fn create_dir_all_verified(
    existing_ancestor: &Path,
    _canonical_existing_ancestor: &Path,
    candidate_root: &Path,
    canonical_workspace_root: &Path,
) -> Result<(), CudaError> {
    let relative = candidate_root
        .strip_prefix(existing_ancestor)
        .map_err(|_| CudaError::CacheIo {
            detail: format!(
                "cache root candidate {} is not nested under its existing ancestor {}",
                candidate_root.display(),
                existing_ancestor.display()
            ),
        })?;

    let mut current_dir = fs::File::open(existing_ancestor).map_err(|e| CudaError::CacheIo {
        detail: format!(
            "failed to open existing cache directory ancestor {}: {e}",
            existing_ancestor.display()
        ),
    })?;
    // 表示・削除用の人間可読パス（セキュリティ判断には使わない。
    // 実際のオープン・作成は常に `current_dir` 起点の dirfd 相対操作で
    // 行う）。
    let mut display_path = existing_ancestor.to_path_buf();

    for component in relative.components() {
        let Component::Normal(name) = component else {
            // `existing_ancestor` は fs 上に実在するパスの祖先探索結果
            // であり、`candidate_root` は環境変数由来の絶対パスを結合
            // したもの（[`resolve_cache_root`]）のため、ここに `..`／`.`／
            // プレフィックス等の非正規コンポーネントが現れることは想定
            // しない。防御的に fail-closed で拒否する。
            return Err(CudaError::CacheIo {
                detail: format!(
                    "cache root candidate {} contains a non-normal path component",
                    candidate_root.display()
                ),
            });
        };
        display_path.push(name);
        let name_str = name.to_str().ok_or_else(|| CudaError::CacheIo {
            detail: format!(
                "cache directory component {} is not valid UTF-8",
                display_path.display()
            ),
        })?;

        if let Err(e) = mkdirat_raw(&current_dir, name_str)
            && e.kind() != std::io::ErrorKind::AlreadyExists
        {
            return Err(CudaError::CacheIo {
                detail: format!(
                    "failed to create cache directory component {}: {e}",
                    display_path.display()
                ),
            });
        }
        // `AlreadyExists`（並行する他プロセスが同じ祖先を先に作成した等）
        // は下記の再検証へフォールスルーし、既存の中身が正規ディレクト
        // リ・workspace 外であることを確認する。

        let child_dir = match opendirat_nofollow(&current_dir, name_str) {
            Ok(f) => f,
            Err(e) if is_confirmed_non_directory_open_error(&e) => {
                let _ = fs::remove_dir(&display_path);
                return Err(CudaError::CacheDirUnavailable {
                    detail: format!(
                        "cache directory component {} is not a plain directory (symlink or \
                         other non-directory found where a directory was expected)",
                        display_path.display()
                    ),
                });
            }
            Err(e) => {
                // `EMFILE`／`EACCES` 等、対象が非ディレクトリだと確定
                // していないエラー。`remove_dir` を実行せずそのまま
                // 伝播する（[`is_confirmed_non_directory_open_error`]
                // のドキュメンテーションコメント参照）。
                return Err(CudaError::CacheIo {
                    detail: format!(
                        "failed to open cache directory component {} after creation: {e}",
                        display_path.display()
                    ),
                });
            }
        };

        let canonical_current = real_path_of_fd(&child_dir).map_err(|e| CudaError::CacheIo {
            detail: format!(
                "failed to resolve real path of cache directory component {} via F_GETPATH: {e}",
                display_path.display()
            ),
        })?;
        if path_lexically_within(&canonical_current, canonical_workspace_root) {
            let _ = fs::remove_dir(&display_path);
            return Err(CudaError::CacheDirUnavailable {
                detail: format!(
                    "cache directory component {} resolves (after symlink resolution) within \
                     workspace root {}; refusing to continue creating directories under it",
                    canonical_current.display(),
                    canonical_workspace_root.display()
                ),
            });
        }

        current_dir = child_dir;
    }

    Ok(())
}

/// [`cache_root`] の解決結果を実際に fs 上へ実体化する公開ラッパー
/// （イシュー #509・Phase C-3。[`store_cache_entry`]／[`load_cache_entry`]
/// から呼ばれる）。実処理は [`ensure_cache_root_in`] に委譲する（同関数
/// ドキュメンテーションコメント参照）。
#[allow(
    dead_code,
    reason = "C-4(#511) の crate 内呼び出し元が実装されるまでの意図的な \
              先行スキャフォールディング（PR #659 の cache_root と同じ判断）"
)]
pub(crate) fn ensure_cache_root(workspace_root: &Path) -> Result<PathBuf, CudaError> {
    ensure_cache_root_in(&cache_root(workspace_root)?, workspace_root)
}

/// 一時ディレクトリコンパイル → アトミック rename でキャッシュエントリを
/// 書き込む（イシュー #509・Phase C-3。実装計画 §3.2）。
///
/// `root` は既に [`ensure_cache_root_in`]／[`ensure_cache_root`] で実体化・
/// containment 検証済みのキャッシュルートを渡す想定（本関数自体は
/// containment 再検証を行わない。責務分離は [`cache_entry_path_in`] と
/// 同じ設計判断）。
///
/// # 並行競合の吸収（受け入れ基準 1）
///
/// [`fs::rename`] が失敗した場合、最終パスに [`validate_cache_entry`] を
/// 満たす既存エントリがあれば「他プロセス（他スレッド）が同一キーへ
/// 先着した」**正常系**とみなし、自分の一時ディレクトリを削除して最終
/// パスを返す（`Ok`）。rename のアトミック性により、途中状態の破損
/// エントリが他プロセスから観測されることはない（A08 整合性）。
///
/// # 破損エントリの置換（受け入れ基準 2）
///
/// 最終パスに既存エントリがあるが [`validate_cache_entry`] を満たさない
/// （破損。クラッシュ残骸・外部破壊）場合、削除対象を「検証したその
/// 実体」に固定するため、まず一意な退避名へ [`fs::rename`]（atomic）
/// してから退避コピー側を再検証する（イシュー #509 codex-review P2
/// 再指摘対応。直前の判定と削除の間に別 writer が正常なエントリへ
/// 置き換えていた場合、パス文字列への `fs::remove_dir_all` だけでは
/// その正常なエントリを誤って消しうるため、rename で捕まえた実体を
/// 単位に判定・処理する）。退避コピーが破損なら削除し、rename を
/// **一度だけ**再試行する。退避コピーが実は正常だった場合は元の位置へ
/// 戻す（戻せなければ既に別 writer が先着したとみなし退避コピーを
/// 破棄する）。再試行後も失敗する場合（別プロセスが同時に置き直した
/// 等）は再度有効性を確認し、有効なら正常系として吸収する。それでも
/// 無効なら無限リトライせず [`CudaError::CacheIo`] で fail-closed に
/// 失敗させる（DoS 耐性。`.claude/rules/security.md` A08）。
///
/// いずれのエラー経路でも一時ディレクトリの削除を試みる（best-effort。
/// 削除失敗はそれ自体をエラーにせず元エラーを優先する）。
///
/// Unix 版は `root` を一度だけ [`open_dir_nofollow`] で fd pin し、以降の
/// 全操作（一時ディレクトリ作成・書き込み・rename・破損判定・退避）を
/// [`store_cache_entry_at`] へ委譲して fd 相対操作のみで行う（イシュー
/// #509 PR #677 codex-review P0 指摘対応: `root` を pin した後で
/// キャッシュルート自体が symlink へ差し替えられても、以降の全操作が
/// pin 済みの元の実体だけを見るため追従しない）。本クレートのビルド
/// 対象は Linux/macOS（unix）のみであり、`O_NOFOLLOW` 相当の std API を
/// 持たない非 unix 向けの検出型フォールバックは維持しない（fd pin による
/// TOCTOU 対策が unix 系 API に依存するため。crate ルート／`nvrtc`
/// モジュール冒頭の `compile_error!` 参照）。
#[cfg(unix)]
fn store_cache_entry_in(
    root: &Path,
    key: &CudaKernelCacheKey,
    kernel_cu: &str,
    kernel_ptx: &str,
) -> Result<PathBuf, CudaError> {
    let final_dir = cache_entry_path_in(root, key)?;
    let entry_name = key.cache_entry_dir_name()?;

    let root_fd = open_dir_nofollow(root).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to pin cache root {}: {e}", root.display()),
    })?;

    store_cache_entry_at(&root_fd, &final_dir, &entry_name, kernel_cu, kernel_ptx)
}

/// [`store_cache_entry_in`]（Unix 版）が pin 済みの `root_fd` を渡して
/// 呼ぶ、fd 相対操作のみで完結する実処理本体（イシュー #509 PR #677
/// codex-review P0 指摘対応）。
///
/// `final_dir` は成功時の戻り値組み立て専用（表示・呼び出し元への通知用）
/// のパスとしてのみ使う。作成・書き込み・rename・検証・**後始末（一時
/// ディレクトリ・退避ディレクトリの削除）を含む全操作**は `root_fd`／
/// `entry_name`／`tmp_name`／`stale_name` の fd 相対操作（[`create_subdir_pinned`]・
/// [`rename_pinned`]・[`validate_cache_entry_at`]・[`entry_exists_at`]・
/// [`remove_cache_entry_pinned`]）のみで行う（イシュー #509 PR #677
/// codex-review P0 再指摘対応: 旧実装は後始末のみ `root.join(..)` という
/// 文字列パスを経由して [`fs::remove_dir_all`] するため、`root_fd` を pin
/// した後で `root` 自体が別ディレクトリへの symlink に差し替えられると
/// pin 済みの実体ではなく symlink 先を再解決してしまい、攻撃者が用意した
/// 同名ディレクトリ・ファイルを削除しうる TOCTOU が残っていた。作成・
/// rename・検証系は既に `root_fd` 起点だったため、削除系のみを同じ手法へ
/// 揃える）。テスト容易性のため `root_fd` を注入で決定化する構造は
/// [`ensure_cache_root_in`] と同型。
///
/// # 並行競合の吸収（受け入れ基準 1）
///
/// [`rename_pinned`] が失敗した場合、最終エントリに
/// [`validate_cache_entry_at`] を満たす既存エントリがあれば「他プロセス
/// （他スレッド）が同一キーへ先着した」**正常系**とみなし、自分の一時
/// ディレクトリを削除して最終パスを返す（`Ok`）。rename のアトミック性
/// により、途中状態の破損エントリが他プロセスから観測されることはない
/// （A08 整合性）。
///
/// # 破損エントリの置換（受け入れ基準 2）
///
/// 最終エントリが存在するが [`validate_cache_entry_at`] を満たさない
/// （破損。クラッシュ残骸・外部破壊）場合、削除対象を「検証したその
/// 実体」に固定するため、まず一意な退避名へ [`rename_pinned`]（atomic）
/// してから退避コピー側を再検証する（パスベース版と同じ判断。イシュー
/// #509 codex-review P2 再指摘対応の踏襲）。退避コピーが破損なら削除し、
/// rename を**一度だけ**再試行する。退避コピーが実は正常だった場合は
/// 元の位置へ戻す（戻せなければ既に別 writer が先着したとみなし退避
/// コピーを破棄する）。再試行後も失敗する場合（別プロセスが同時に置き
/// 直した等）は再度有効性を確認し、有効なら正常系として吸収する。
/// それでも無効なら無限リトライせず [`CudaError::CacheIo`] で
/// fail-closed に失敗させる（DoS 耐性。`.claude/rules/security.md` A08）。
/// 「最終エントリが存在するか」の判定（[`entry_exists_at`]）はディレクト
/// リ・通常ファイル・symlink のいずれで占有されていても検出する（イシュー
/// #509 PR #677 Bugbot 指摘〈Non-dir blocks cache replacement〉対応。
/// [`entry_exists_at`] のドキュメンテーションコメント参照）。旧実装は
/// ディレクトリのみを検出したため、非ディレクトリ占有時は本節の置換
/// 分岐が素通りされ常に `CudaError::CacheIo` の恒久失敗に陥っていた。
#[cfg(unix)]
fn store_cache_entry_at(
    root_fd: &fs::File,
    final_dir: &Path,
    entry_name: &str,
    kernel_cu: &str,
    kernel_ptx: &str,
) -> Result<PathBuf, CudaError> {
    let tmp_name = temp_entry_dir_name(entry_name);

    if let Err(e) = write_entry_files_pinned(root_fd, &tmp_name, kernel_cu, kernel_ptx) {
        remove_cache_entry_pinned(root_fd, &tmp_name);
        return Err(e);
    }

    if rename_pinned(root_fd, &tmp_name, root_fd, entry_name).is_ok() {
        // ルートディレクトリ自体の fsync は best-effort（rename の
        // アトミック性・可視性自体は fsync に依存しない。エントリ追加が
        // ディスクへ確実に反映されることを高めるための追加防御に留める）。
        let _ = root_fd.sync_all();
        return Ok(final_dir.to_path_buf());
    }

    // rename 失敗経路: 他プロセス先着（正常系）／破損エントリ置換／
    // その他 fs エラーのいずれかを、最終エントリの状態から判別する。
    if validate_cache_entry_at(root_fd, entry_name) {
        remove_cache_entry_pinned(root_fd, &tmp_name);
        return Ok(final_dir.to_path_buf());
    }

    if entry_exists_at(root_fd, entry_name) {
        // 破損エントリの可能性がある経路。ただし直前の
        // `validate_cache_entry_at` 判定から本チェックまでの間に、別
        // writer（他プロセス・他スレッド）が正常なエントリを先着配置
        // した可能性がある（パスベース版と同じ判断。イシュー #509
        // codex-review P2 指摘 `PRRT_kwDOTuUCJc6ZkpxM` 対応の踏襲）。
        // 削除直前に再検証し、既に有効なら削除せず正常系として吸収
        // する: 再検証なしに削除すると、その別 writer の先着エントリを
        // 「破損」扱いで消してしまい first-writer-wins 契約を破りうる。
        if validate_cache_entry_at(root_fd, entry_name) {
            remove_cache_entry_pinned(root_fd, &tmp_name);
            return Ok(final_dir.to_path_buf());
        }

        // 破損エントリを削除する前に、直前の `validate_cache_entry_at`
        // 判定と削除の間に別 writer（他プロセス・他スレッド）が破損
        // エントリを正常なエントリへ atomic rename で置き換え得る
        // （パスベース版と同じ判断の踏襲）。fd 相対の再検証だけでは
        // 「検証したその実体」と「削除するその実体」が同一である保証が
        // ない（検証と削除の間に再度差し替えられ得る）ため、削除対象を
        // まず一意な退避名へ atomic rename して固定してから検証・削除
        // する（[`rename_pinned`] は POSIX ではアトミックなので、この
        // 1 手で「rename 時点で最終エントリにあった実体」を確実に
        // 捕まえられる）。
        let stale_name = temp_entry_dir_name(entry_name);
        match rename_pinned(root_fd, entry_name, root_fd, &stale_name) {
            Ok(()) => {
                if validate_cache_entry_at(root_fd, &stale_name) {
                    // 捕まえた実体は実は正常だった（別 writer が直前の
                    // 判定後・本 rename 前に先着していた）。
                    // first-writer-wins 契約を守るため、可能なら元の
                    // 位置へ戻す。戻せない場合（さらに別 writer が
                    // 既に最終エントリを埋めた場合）は自分の退避
                    // コピーを破棄する（いずれにせよ最終エントリには
                    // 有効なエントリが残る）。
                    if rename_pinned(root_fd, &stale_name, root_fd, entry_name).is_ok() {
                        remove_cache_entry_pinned(root_fd, &tmp_name);
                        return Ok(final_dir.to_path_buf());
                    }
                    remove_cache_entry_pinned(root_fd, &stale_name);
                    if validate_cache_entry_at(root_fd, entry_name) {
                        remove_cache_entry_pinned(root_fd, &tmp_name);
                        return Ok(final_dir.to_path_buf());
                    }
                } else {
                    // 退避コピーは検証済みで真に破損している。安全に削除
                    // できる（最終エントリからは既に rename 済みのため、
                    // これを削除しても他 writer の実体を破壊しない）。
                    remove_cache_entry_pinned(root_fd, &stale_name);
                }
            }
            Err(_) => {
                // 最終エントリが既に別 writer によって削除・置換された
                // 等。下の再試行（最終エントリの現状に応じて分岐）へ
                // フォールスルーする。
            }
        }

        // 破損エントリを置換して一度だけ再試行する（無限リトライしない）。
        // 上の分岐で既に有効なエントリを復元・確認できていれば最終
        // エントリは存在するため、ここでは何もせず下の最終判定へ
        // フォールスルーする。
        if !entry_exists_at(root_fd, entry_name)
            && rename_pinned(root_fd, &tmp_name, root_fd, entry_name).is_ok()
        {
            let _ = root_fd.sync_all();
            return Ok(final_dir.to_path_buf());
        }
    }

    remove_cache_entry_pinned(root_fd, &tmp_name);
    if validate_cache_entry_at(root_fd, entry_name) {
        // 再試行の間に別プロセスが有効なエントリを置いた（正常系）。
        Ok(final_dir.to_path_buf())
    } else {
        Err(CudaError::CacheIo {
            detail: format!(
                "failed to atomically rename cache entry into place: {}",
                final_dir.display()
            ),
        })
    }
}

/// [`store_cache_entry_in`] の公開ラッパー（イシュー #509・Phase C-3）。
/// `workspace_root` から [`ensure_cache_root`] でルートを実体化した上で
/// 委譲する。NVRTC コンパイル（`compile_ptx`）との結線（コンパイル成功後
/// に本関数を呼ぶ導線）は C-4（#511）のスコープ（実装計画 §3.5: store は
/// バイト列を受け渡す純 I/O プリミティブに留める）。
#[allow(
    dead_code,
    reason = "C-4(#511) の crate 内呼び出し元が実装されるまでの意図的な \
              先行スキャフォールディング（PR #659 の cache_root と同じ判断）"
)]
pub(crate) fn store_cache_entry(
    workspace_root: &Path,
    key: &CudaKernelCacheKey,
    kernel_cu: &str,
    kernel_ptx: &str,
) -> Result<PathBuf, CudaError> {
    let root = ensure_cache_root(workspace_root)?;
    store_cache_entry_in(&root, key, kernel_cu, kernel_ptx)
}

/// キャッシュエントリを読み出す（イシュー #509・Phase C-3。実装計画
/// §3.3）。`root` は [`store_cache_entry_in`] と同じく呼び出し元で
/// 実体化・containment 検証済みのキャッシュルートを渡す想定。
///
/// - エントリディレクトリ不在 → `Ok(None)`（ミス）
/// - 両ファイル存在・読み取り成功 → `Ok(Some(..))`
/// - 片方欠落（破損） → `Ok(None)`（ミス扱い）
///
/// 破損エントリを検出しても**削除は行わない**: 読み手が消すと並行する
/// 書き手・他の読み手との競合面が増えるため、置き換えは
/// [`store_cache_entry_in`] 側（rename 失敗判定）に一元化する（実装計画
/// §3.3 の設計判断）。
// `root` 自体を fd pin してから、以降は `root_fd` を起点とした fd 相対
// 操作（[`open_dir_child_nofollow`]・[`open_file_child_nofollow`]）のみで
// エントリディレクトリ・2 ファイルを解決する（イシュー #509 PR #677
// Bugbot 指摘〈Load skips cache root pin〉対応）。旧実装は `root`
// （呼び出し元検証済みのキャッシュルート）と `key` から組み立てた
// パス文字列（[`cache_entry_path_in`]）をそのまま `open_dir_nofollow` に
// 渡していたため、[`ensure_cache_root`] での containment 検証後に
// `root` 自体が symlink へ差し替えられると、検証済みルート外の細工
// された `kernel.cu`／`kernel.ptx` がキャッシュヒットとして読み込まれ
// 得た（現状はデコード先が `CachedKernel`〈構造体フィールドへコピー
// されるだけ〉のため実害は限定的だが、NVRTC 結線後は当該フィールドが
// PTX として GPU 上で実行されうる導線になるため、検証と読み取りを
// 原子的に結合しておく）。[`store_cache_entry_in`]（[`store_cache_entry_at`]
// のドキュメンテーションコメント参照）は既に `root_fd` を pin してから
// fd 相対操作のみで完結しており、本関数もそれと対称な構造へ揃える。
// `root_fd` pin 後は `entry_name`（1 コンポーネント）・
// [`CACHE_ENTRY_SOURCE_FILE`]／[`CACHE_ENTRY_PTX_FILE`]（各 1 コンポー
// ネント）のみを fd 相対で解決するため、パス文字列を再度ルートから
// 辿り直す経路が存在しない。[`open_dir_child_nofollow`]・
// [`open_file_child_nofollow`] は Linux では [`proc_fd_path`]
// （`/proc/self/fd/<fd>`）経由、macOS では [`opendirat_nofollow`]・
// [`openat_nofollow`]（`openat(2)` の FFI 直接呼び出し）経由でそれぞれ
// fd 起点解決を実現する（[`store_cache_entry_at`] が使う fd 相対
// プリミティブと共用。新規クレート追加なし。`libc`／`rustix` は許容
// 依存 8 区分〈`.claude/rules/deps-policy.md`〉外のためユーザー承認
// なしに追加できない）。
//
// [`read_verified_cache_entry_file`] で fd 経由の `fstat` による通常
// ファイル種別検証（FIFO 等の特殊ファイル拒否。イシュー #509 codex-review
// P0 指摘対応）と読み込みサイズ上限（[`MAX_CACHE_ENTRY_FILE_BYTES`]。同
// P0 指摘対応）を適用する。非 UTF-8 デコード失敗は他の破損種別（欠落・
// 空・過大・特殊ファイル・ソース不一致）と同じくミス扱い（`Ok(None)`）
// とする（イシュー #509 PR #677 Bugbot 指摘〈Invalid UTF-8 treated as
// hard error〉対応）。旧実装は非 UTF-8 のみ `CudaError::CacheIo` の
// ハードエラーとして扱っており、本関数冒頭のドキュメンテーション
// コメントが定める「破損エントリはミス扱いにする」契約と矛盾していた。
// 他の破損検出と同様、呼び出し元は再コンパイルへフォールバックすれば
// よい。
#[cfg(unix)]
fn load_cache_entry_in(
    root: &Path,
    key: &CudaKernelCacheKey,
    expected_src: &str,
) -> Result<Option<CachedKernel>, CudaError> {
    let entry_name = key.cache_entry_dir_name()?;

    // キャッシュルート自体を `O_NOFOLLOW | O_DIRECTORY` で一度だけ開き
    // fd を pin する（[`store_cache_entry_in`] Unix 版と同じ手順）。
    let root_fd = match open_dir_nofollow(root) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    // `root_fd` 起点で `entry_name`（1 コンポーネント）のみを解決する。
    // symlink・非ディレクトリであれば拒否されるため「symlink でないこと
    // を確認してから中身を読む」という 2 手順の間の TOCTOU が生じない。
    let dir = match open_dir_child_nofollow(&root_fd, &entry_name) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let cu_file = match open_file_child_nofollow(&dir, CACHE_ENTRY_SOURCE_FILE) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let ptx_file = match open_file_child_nofollow(&dir, CACHE_ENTRY_PTX_FILE) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let kernel_cu =
        match read_verified_cache_entry_file(cu_file).map_err(|e| CudaError::CacheIo {
            detail: format!("failed to read cached kernel source: {e}"),
        })? {
            Some(bytes) => match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            },
            None => return Ok(None),
        };
    // ハッシュ衝突安全弁（実装計画 §3.1・§7。FNV-1a 64bit は非暗号ハッシュ
    // のため、異なるソースが同一エントリ名〈`kernel.<name>.<hash16>`〉に
    // 衝突した場合でも誤った PTX を返さないよう、保存済みソース全文を
    // 要求元ソースとバイト単位で照合する。不一致はミス扱い〈`Ok(None)`〉
    // とし、C-4〈#511〉配線後は素直に再コンパイルへフォールバックする）。
    if kernel_cu != expected_src {
        return Ok(None);
    }
    let kernel_ptx =
        match read_verified_cache_entry_file(ptx_file).map_err(|e| CudaError::CacheIo {
            detail: format!("failed to read cached kernel ptx: {e}"),
        })? {
            Some(bytes) => match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Ok(None),
            },
            None => return Ok(None),
        };

    Ok(Some(CachedKernel {
        kernel_cu,
        kernel_ptx,
    }))
}

/// [`load_cache_entry_in`] の公開ラッパー（イシュー #509・Phase C-3）。
/// `workspace_root` から [`ensure_cache_root`] でルートを実体化した上で
/// 委譲する（store 側と対称。ルート実体化は冪等なため読み出し専用の
/// 呼び出しでも安全）。
///
/// `expected_src` は呼び出し元がこれからコンパイルしようとしている
/// カーネルソース全文。`key` の 64bit ハッシュ（非暗号・FNV-1a）が
/// 衝突した場合に誤った PTX を返さないための安全弁として、保存済み
/// `kernel.cu` とバイト単位で照合する（実装計画 §3.1・§7）。この検査を
/// `load_cache_entry_in` 側に閉じ込めるのは、呼び出し元が省略できる
/// `Option` 引数にしないという C-2 以来の「注入で決定化・迂回不能」
/// 方針を fs I/O 層まで一貫させるため。
#[allow(
    dead_code,
    reason = "C-4(#511) の crate 内呼び出し元が実装されるまでの意図的な \
              先行スキャフォールディング（PR #659 の cache_root と同じ判断）"
)]
pub(crate) fn load_cache_entry(
    workspace_root: &Path,
    key: &CudaKernelCacheKey,
    expected_src: &str,
) -> Result<Option<CachedKernel>, CudaError> {
    let root = ensure_cache_root(workspace_root)?;
    load_cache_entry_in(&root, key, expected_src)
}

/// リンクされている NVRTC のバージョンを `(major, minor)` で返す。
///
/// [`CudaKernelCacheKey::from_device`] から呼ばれ、DeepGEMM の
/// compiler signature（例 NVRTC12.9）相当をキャッシュキーへ含めるために
/// 使う。`compile_ptx` と同じく `is_culib_present()` で先にプローブし、
/// 不在なら `CudaError::NvrtcUnavailable` を返す（dynamic-loading の
/// panic 経路回避。CUDA toolkit 非搭載環境でも panic しない契約は本関数
/// でも維持する。`build-no-cuda-toolkit` CI ジョブと整合）。
///
/// # Safety（`unsafe` 使用箇所）
///
/// `cudarc::nvrtc::sys::nvrtcVersion` は cudarc 0.19.8 に safe wrapper が
/// ないため FFI 境界の `unsafe` を必要最小限で使う（`.claude/rules/
/// security.md`）。呼び出し前に `is_culib_present()` で `libnvrtc` の
/// 存在を確認済みであり、`major`/`minor` はスタック上のローカル変数への
/// 有効なポインタを渡す（NVRTC API 契約上、両ポインタは非 null かつ
/// 呼び出し中のみ書き込まれる）。
pub fn nvrtc_version() -> Result<(i32, i32), CudaError> {
    // SAFETY: `cudarc::nvrtc::sys::is_culib_present()` は `libnvrtc` の
    // 存在確認のみを行う `dlopen` ベースのプローブで、引数を取らず
    // ポインタ・共有可変状態も扱わない（cudarc 0.19.8 に safe wrapper が
    // ないため FFI 境界の `unsafe` を必要最小限で使う。`.claude/rules/
    // security.md`）。本呼び出し自体が「`libnvrtc` が存在するか」を
    // 判定する目的であり、後続の `nvrtcVersion` 呼び出し（252 行目付近）
    // の前提確認としてここで先に行う。
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(CudaError::NvrtcUnavailable {
            detail: "libnvrtc dynamic library not found (dlopen failed); \
                     CUDA toolkit is not installed or not on the library search path"
                .to_string(),
        });
    }

    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    // SAFETY: `is_culib_present()`（上記ブロック）で `libnvrtc` の存在を
    // 確認済みのうえで呼ぶ。`major`/`minor` はこの直前で初期化した
    // スタックローカル変数への `&mut` であり、非 null かつ呼び出し中の
    // 書き込みに対して有効な唯一の可変参照を渡す（NVRTC API 契約上、
    // 両ポインタは `nvrtcVersion` 呼び出し中のみ書き込まれる）。
    // 1 番目のブロック（`is_culib_present()`）とは前提条件（引数なし・
    // 副作用なしの存在確認 vs 出力ポインタへの書き込みを伴うバージョン
    // 取得）が異なるため、ここで独立して根拠を明示する。
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(CudaError::NvrtcUnavailable {
            detail: format!("nvrtcVersion() failed: {result:?}"),
        });
    }
    Ok((major, minor))
}

/// `derive_pipeline_stages` が受け付ける段数の絶対上限（DeepGEMM
/// `kNumMaxStages`（`csrc/jit_kernels/heuristics/sm90.hpp` 付近）を踏襲）。
/// SMEM 予算がどれだけ大きくても、これ以上の段数はレジスタ圧・命令数
/// 増加に見合わないため候補から除外する。
pub const MAX_PIPELINE_STAGES: u32 = 16;

// カーネル契約上の絶対下限は `STAGES >= 2`（`kernels_mma.rs` の
// `cp.async.wait_group "n"(STAGES - 2)` が前提とする。同ファイル 194 行目
// 付近のコメント参照）。`derive_pipeline_stages` の `min_required`
// （DeepGEMM 由来の 3／4 段要求）はこの絶対下限より常に大きい値を返すため
// 通常この下限が直接効くことはないが、タイル構成が変わった場合の安全側の
// 前提として本コメントに明示する（値そのものを保持するだけの未使用定数は
// 置かない）。

/// SMEM 予算から `cp.async` パイプライン段数を逆算する（C-8・#521）。
///
/// # 導出式（DeepGEMM `sm90.hpp` の `get_smem_config` 系ヒューリスティクス
/// を踏襲）
///
/// ```text
/// smem_per_stage = (block_m * block_k + block_k * block_n) * bytes_per_element
/// derived        = min(smem_budget / smem_per_stage, MAX_PIPELINE_STAGES)
/// min_required   = if block_m * block_n < 128 * 192 { 4 } else { 3 }
/// ```
///
/// `derived < min_required` の場合は候補不成立として `Err` を返す
/// （DeepGEMM が対応する構成を棄却するのと同義。カーネル契約自体の絶対
/// 下限は `STAGES >= 2` だが、DeepGEMM 由来の `min_required` は常にそれ
/// 以上のため、本関数が返す `Err` はより保守的な「性能上望ましくない
/// 段数」を弾く判定になる）。
///
/// # 呼び出し文脈
///
/// SMEM 予算（`smem_budget_bytes`）は呼び出し元（`gemm_auto.rs` の
/// 結線ヘルパー想定。C-8 実装計画 §4）がデバイスの
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を実行時に取得し、
/// 本カーネル群が動的 SMEM opt-in（`cudaFuncSetAttribute`）を行わない
/// 静的 `__shared__` 構成であることを踏まえ 49,152 バイト（48KiB）で
/// クランプしたうえで渡す想定である（ハードコード定数を持たない設計。
/// `docs/perf/sm121-device-attributes.md` の SMEM 容量行が「未実測」の
/// ままであるため、実測値を推定で定数化しない安全側判断）。本関数自体は
/// 予算値の出所を問わない純関数として、その判断とは独立にテスト可能に
/// する。
///
/// # オーバーフロー安全性
///
/// `block_m`／`block_n`／`block_k` は `u32`（`CudaKernelDescriptor` の
/// フィールド型と同じ）で受け取り、乗算は `u64` へ拡張した
/// `checked_mul`／`checked_add` で行う。巨大な block 値を渡されても
/// panic せず `Err` を返す（本番経路で `unwrap`/`expect` を使わない
/// 契約。`.claude/rules/coding-rust.md`）。
pub fn derive_pipeline_stages(
    block_m: NonZeroU32,
    block_n: NonZeroU32,
    block_k: NonZeroU32,
    bytes_per_element: NonZeroU32,
    smem_budget_bytes: u64,
) -> Result<NonZeroU32, CudaError> {
    let (bm, bn, bk, bpe) = (
        u64::from(block_m.get()),
        u64::from(block_n.get()),
        u64::from(block_k.get()),
        u64::from(bytes_per_element.get()),
    );

    let a_tile_elems = bm
        .checked_mul(bk)
        .ok_or_else(|| overflow_err("block_m * block_k"))?;
    let b_tile_elems = bk
        .checked_mul(bn)
        .ok_or_else(|| overflow_err("block_k * block_n"))?;
    let tile_elems = a_tile_elems
        .checked_add(b_tile_elems)
        .ok_or_else(|| overflow_err("block_m*block_k + block_k*block_n"))?;
    let smem_per_stage = tile_elems
        .checked_mul(bpe)
        .ok_or_else(|| overflow_err("tile_elems * bytes_per_element"))?;

    if smem_per_stage == 0 {
        return Err(CudaError::InvalidKernelDescriptor {
            detail: "derive_pipeline_stages: smem_per_stage must be non-zero \
                     (block_m/block_n/block_k/bytes_per_element are all \
                     NonZeroU32, so this indicates an internal invariant \
                     violation)"
                .to_string(),
        });
    }

    let derived_raw = smem_budget_bytes / smem_per_stage;
    let derived = derived_raw.min(u64::from(MAX_PIPELINE_STAGES));

    // DeepGEMM 由来の最小段数要求: 小タイル（BM*BN < 128*192）はレジスタ
    // 圧が相対的に軽いぶんレイテンシ隠蔽により多くの段数を要求し（4 段）、
    // それ以外は 3 段を要求する（`docs/spec` には記載のない実装判断で
    // あり、根拠は本関数冒頭ドキュメンテーションコメントの参照実装）。
    let block_area = bm
        .checked_mul(bn)
        .ok_or_else(|| overflow_err("block_m * block_n"))?;
    let min_required: u64 = if block_area < 128 * 192 { 4 } else { 3 };

    if derived < min_required {
        return Err(CudaError::InvalidKernelDescriptor {
            detail: format!(
                "derive_pipeline_stages: derived stage count {derived} \
                 (smem_budget_bytes={smem_budget_bytes}, \
                 smem_per_stage={smem_per_stage}) is below the DeepGEMM-derived \
                 minimum requirement {min_required} (block_m={block_m}, \
                 block_n={block_n}, tile area={block_area})",
                block_m = block_m.get(),
                block_n = block_n.get(),
            ),
        });
    }

    // `derived >= min_required >= 3 > 0` かつ `u32` 範囲内（`MAX_PIPELINE_STAGES`
    // による上限クランプ済み）であるため `NonZeroU32::new` は必ず `Some` を
    // 返すが、`unwrap`/`expect` を使わず型付きエラーへ倒す
    // （`.claude/rules/coding-rust.md`）。
    let derived_u32 = u32::try_from(derived).map_err(|_| overflow_err("derived stage count"))?;
    NonZeroU32::new(derived_u32).ok_or_else(|| CudaError::InvalidKernelDescriptor {
        detail: format!(
            "derive_pipeline_stages: internal invariant violation, derived \
             stage count {derived_u32} was zero despite min_required={min_required}"
        ),
    })
}

/// [`derive_pipeline_stages`] のオーバーフロー検出箇所を集約するヘルパ。
/// 巨大な block 寸法が渡された場合に `u64` 中間値でも表現できない旨を
/// `CudaError::InvalidKernelDescriptor` として返す（panic 経路を作らない）。
fn overflow_err(step: &str) -> CudaError {
    CudaError::InvalidKernelDescriptor {
        detail: format!("derive_pipeline_stages: overflow computing {step}"),
    }
}

/// `src` を `arch`（`compute_XY` 形式。`CudaDevice::arch()` が返す値）
/// 向けに NVRTC でコンパイルし `Ptx` を返す。
///
/// 手順:
/// 1. `cudarc::nvrtc::sys::is_culib_present()` で `libnvrtc` の存在を
///    確認する（driver 側と同様、NVRTC 側にも `dlopen` 失敗時の panic
///    経路があるため。cudarc-0.19.8/src/nvrtc/sys/mod.rs:529）。
///    不在なら `CudaError::NvrtcUnavailable` を返す。
/// 2. include_paths なしでコンパイルを試みる。
/// 3. 失敗した場合のみ、`CUDA_INCLUDE_PATH` 環境変数または既知の候補
///    パス（CUDA 13.0 標準インストール先）で順に再試行する
///    （`cuda_fp16.h` 等が NVRTC 組み込みで解決できない環境向け。
///    PoC-v2-3 の 2 段構えを踏襲）。
///
/// `fmad`（NVRTC の FMA 契約フラグ）は明示せず NVRTC 既定のまま扱う。
/// CPU 参照実装（`f32::mul_add`）との丸め方針統一（PoC-v2-5 の K=4096
/// ストレスケースで実測確認済み。`.claude/rules/coding-rust.md`）は
/// この既定 FMA 契約が前提であり、呼び出し元が独自に上書きしないこと。
pub fn compile_ptx(src: &str, arch: &str) -> Result<Ptx, CudaError> {
    // SAFETY: `cudarc::nvrtc::sys::is_culib_present()` は `libnvrtc` の
    // 存在確認のみを行う `dlopen` ベースのプローブで、引数を取らず
    // ポインタ・共有可変状態も扱わない（cudarc 0.19.8 に safe wrapper が
    // ないため FFI 境界の `unsafe` を必要最小限で使う。`.claude/rules/
    // security.md`）。`nvrtc_version()` と同じく、`libnvrtc` 不在時に
    // dynamic-loading 側の panic 経路を踏まないための事前プローブとして
    // 呼び出し本体（NVRTC コンパイル）の前に行う。
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(CudaError::NvrtcUnavailable {
            detail: "libnvrtc dynamic library not found (dlopen failed); \
                     CUDA toolkit is not installed or not on the library search path"
                .to_string(),
        });
    }

    // `CompileOptions::arch` は `Option<&'static str>`（cudarc 0.19.8
    // src/nvrtc/safe.rs:240）のため呼び出しごとの `String` をそのまま
    // 渡せず `Box::leak` が必要になる。デバイス初期化・カーネル
    // コンパイルは #33/#34 でも通常デバイスあたり定数回（カーネル種別数）
    // に限られるため、リーク量は限定的であることをここに契約として
    // 明記する（無制限ループでの呼び出しは避けること）。
    let arch_static: &'static str = Box::leak(arch.to_string().into_boxed_str());
    let base = CompileOptions {
        arch: Some(arch_static),
        ..Default::default()
    };

    if let Ok(ptx) = compile_ptx_with_opts(src, base.clone()) {
        return Ok(ptx);
    }

    let candidates = [
        std::env::var("CUDA_INCLUDE_PATH").ok(),
        Some("/usr/local/cuda/include".to_string()),
        Some("/usr/local/cuda-13.0/targets/sbsa-linux/include".to_string()),
        Some("/usr/local/cuda-13.0/targets/x86_64-linux/include".to_string()),
    ];
    for path in candidates.into_iter().flatten() {
        let opts = CompileOptions {
            include_paths: vec![path],
            ..base.clone()
        };
        if let Ok(ptx) = compile_ptx_with_opts(src, opts) {
            return Ok(ptx);
        }
    }

    // 最終的にすべて失敗した場合は、include_paths なしの試行のエラーを
    // 呼び出し元に返す（フォールバック群のどれが失敗したかより、
    // 素の失敗理由の方が診断に有用。PoC-v2-3 と同じ判断）。
    compile_ptx_with_opts(src, base).map_err(CudaError::from)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use tensor_core::dispatch::{DType, GemmShape};

    use super::*;

    fn sample_descriptor() -> CudaKernelDescriptor {
        CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            2,
            DType::F32,
            CompiledDims::STATIC_MNK,
        )
        .expect("valid descriptor parameters must not fail")
    }

    // C-5（#514）: 実テンプレート（`kernels_mma::mma_f16_source`。既定
    // 構成で展開済みの実カーネルソース）を使う。合成のダミー文字列ではなく
    // 実際にコンパイル対象となるソースでテストすることで、
    // `source_changes_produce_distinct_key_and_hash` が「テンプレート
    // 文字列を編集した際に必ずキャッシュミスする」という受け入れ基準を
    // 実物に即して検証する。
    fn sample_source() -> String {
        crate::kernels_mma::mma_f16_source().to_string()
    }

    fn sample_key() -> CudaKernelCacheKey {
        CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        )
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // 受け入れ基準 1: 未確定パラメータ（ゼロ値）のまま構築できず、
    // panic ではなく型付きエラーを返すこと。
    #[test]
    fn new_rejects_zero_block_m() {
        let result = CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            0,
            64,
            32,
            2,
            DType::F32,
            CompiledDims::STATIC_MNK,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_stages() {
        let result = CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            0,
            DType::F32,
            CompiledDims::STATIC_MNK,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // codex-review 指摘（PR #641・P2）: `new_rejects_zero_block_m` /
    // `new_rejects_zero_stages` のみでは `block_n`・`block_k` の
    // ゼロ値拒否が未検証だった。`non_zero` ヘルパはフィールドごとに
    // 独立して呼ばれるため、各フィールドを個別に検証する。
    #[test]
    fn new_rejects_zero_block_n() {
        let result = CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            0,
            32,
            2,
            DType::F32,
            CompiledDims::STATIC_MNK,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_block_k() {
        let result = CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            0,
            2,
            DType::F32,
            CompiledDims::STATIC_MNK,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // codex-review 指摘（PR #641・P2）: `kernel_name: &'static str` は
    // 型だけでは `"../escape"` 等のパス走査文字列を排除できない。C-2
    // （#506）のディレクトリ命名（`kernel.<name>.<hash>`）で使われる前に
    // ここで拒否し、後続タスクが「検証済み文字列」を前提にできるように
    // する。
    #[test]
    fn new_rejects_path_traversal_kernel_name() {
        // イシュー #506（Phase C-2）レビュー指摘: 先頭・末尾ドット
        // （`.foo`・`foo.`）は素朴なチェックでは通過してしまうが、
        // `cache_entry_dir_name()` が生成する `"kernel..foo.<hash>"` 等に
        // `".."` を出現させるため、構築時点で前倒しして拒否する
        // （`CudaKernelDescriptor::new` 内コメント参照）。検証本体は
        // `new`／`new_with_compiled_dims` 共通の `build()` にあるため
        // （イシュー #519・C-7）、ここでは `new_with_compiled_dims` 経由で
        // 検査する。
        for bad_name in ["../escape", "a/b", "a\\b", "..", "", ".foo", "foo.", "."] {
            let result = CudaKernelDescriptor::new_with_compiled_dims(
                bad_name,
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            );
            assert!(
                matches!(result, Err(CudaError::InvalidKernelDescriptor { .. })),
                "kernel_name {bad_name:?} must be rejected"
            );
        }
    }

    // 受け入れ基準: 同一パラメータから構築した 2 キーは `==` かつ
    // 同一ハッシュになること（`HashMap` キーとして使うための前提）。
    #[test]
    fn identical_parameters_produce_equal_keys_and_hashes() {
        let a = sample_key();
        let b = sample_key();
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    // 受け入れ基準: shape・BM/BN/BK・stages・dtype・compute capability・
    // NVRTC バージョン・compile_flags のいずれか 1 つを変えると不一致に
    // なること（DeepGEMM のアーキ違い＝別エントリの性質を各フィールドへ
    // 一般化して検証する）。
    #[test]
    fn changing_any_field_produces_distinct_key() {
        let base = sample_key();

        let different_shape = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(2048, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_shape);

        let different_block_m = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                32,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_block_m);

        // codex-review 指摘（PR #641・P2）: `block_n`・`block_k`・
        // `stages`・`kernel_name` は `different_block_m` までのケースでは
        // 未検証だった。各フィールドを単独で変更し、`Hash`/`Eq` の
        // derive がフィールド漏れなく全メンバを含んでいることを検証する。
        let different_block_n = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                32,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_block_n);

        let different_block_k = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                16,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_block_k);

        let different_stages = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                3,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_stages);

        let different_kernel_name = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32_v2",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_kernel_name);

        // `kernel_name` は固定したまま `dtype` のみ変更する。
        // `kernel_name` を同時に変えると、万一 `dtype` が `Hash`/`Eq`
        // から欠落しても `kernel_name` 側の差分だけで
        // `assert_ne!` が通ってしまい検出できない
        // （codex-review 指摘・PR #641・P2）。
        let different_dtype = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F16,
                CompiledDims::STATIC_MNK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_dtype);

        // compute capability 違い（DeepGEMM のアーキ違い＝別エントリの
        // 性質そのもの）。
        let different_cc = CudaKernelCacheKey::new(
            sample_descriptor(),
            (9, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_cc);

        let different_nvrtc_version = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (13, 0),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_nvrtc_version);

        let different_flags = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_90".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_flags);

        // イシュー #519 受け入れ基準 3: `compiled_dims` のみを変えても
        // 別エントリになること（他フィールド一式は `sample_descriptor`
        // と同一のまま `compiled_dims` だけ `STATIC_NK` へ切り替える）。
        let different_compiled_dims = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_NK,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        assert_ne!(base, different_compiled_dims);

        // 受け入れ基準の中核（C-5・#514）: テンプレート展開元の descriptor・
        // 環境パラメータが全て同一でも、ソース文字列自体（例: 1 文字の
        // コメント追記に相当する変更）が異なれば別キーになること。
        // カーネルテンプレート（`kernels_mma.rs` の `MMA_F16_BODY` 等）を
        // 編集してもタイル定数・カーネル名が不変なら以前は同一キーに
        // 縮退していた欠陥（本タスクの動機）をここで直接検証する。
        let different_source = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            format!("{}\n// edited\n", sample_source()),
        );
        assert_ne!(base, different_source);
        // 実装計画 §4-5 の中核受け入れ基準そのもの: ソース断片編集が
        // ディスクキャッシュのディレクトリ名（`kernel.<name>.<hash>`）を
        // 変え、C-3（#509）実装後に陳腐化エントリへ誤ヒットしないこと。
        assert_ne!(
            base.cache_entry_dir_name().expect("must succeed"),
            different_source
                .cache_entry_dir_name()
                .expect("must succeed"),
        );
    }

    // イシュー #519（C-7）: `CompiledDims` プリセットが意図した次元集合を
    // 表すこと（`DYNAMIC_ALL` は全次元動的・`STATIC_NK` は N/K のみ定数化
    // ・M 動的・`STATIC_MNK` は全次元定数化）。`Default` は DeepGEMM
    // `get_compiled_dim` の既定意図（N/K 定数化）に揃えて `STATIC_NK` と
    // 一致すること。
    #[test]
    fn compiled_dims_presets_have_expected_flags() {
        assert_eq!(
            CompiledDims::DYNAMIC_ALL,
            CompiledDims::new(false, false, false)
        );
        assert_eq!(
            CompiledDims::STATIC_NK,
            CompiledDims::new(false, true, true)
        );
        assert_eq!(
            CompiledDims::STATIC_MNK,
            CompiledDims::new(true, true, true)
        );
        assert_eq!(CompiledDims::default(), CompiledDims::STATIC_NK);

        assert!(!CompiledDims::DYNAMIC_ALL.m());
        assert!(!CompiledDims::DYNAMIC_ALL.n());
        assert!(!CompiledDims::DYNAMIC_ALL.k());
        assert!(!CompiledDims::STATIC_NK.m());
        assert!(CompiledDims::STATIC_NK.n());
        assert!(CompiledDims::STATIC_NK.k());
        assert!(CompiledDims::STATIC_MNK.m());
        assert!(CompiledDims::STATIC_MNK.n());
        assert!(CompiledDims::STATIC_MNK.k());
    }

    // 受け入れ基準 3（キャッシュキーの正規化）の基礎となる純関数の検証:
    // 非選択次元は sentinel `0` に、選択次元は実値のまま保たれること。
    #[test]
    fn cache_shape_normalizes_only_non_selected_dims_to_zero() {
        let shape = GemmShape::new(64, 4096, 2048);

        assert_eq!(
            CompiledDims::DYNAMIC_ALL.cache_shape(shape),
            GemmShape::new(0, 0, 0)
        );
        assert_eq!(
            CompiledDims::STATIC_NK.cache_shape(shape),
            GemmShape::new(0, 4096, 2048)
        );
        assert_eq!(CompiledDims::STATIC_MNK.cache_shape(shape), shape);
    }

    // イシュー #519 受け入れ基準 3（エントリ数抑制）の基礎:
    // `new_with_compiled_dims` が `compiled_dims` に従って
    // `cache_key_shape()` を正規化して保持するため、動的次元の実値違いは
    // `descriptor.cache_key_shape()` に現れないこと。一方 `shape()` は
    // codex-review 指摘（PR #674・P1）対応により、渡した実値をそのまま
    // 返す契約を維持する（正規化しない）。
    #[test]
    fn new_with_compiled_dims_normalizes_cache_key_shape_only() {
        let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
            "mma_f16",
            GemmShape::new(64, 4096, 2048),
            64,
            128,
            32,
            3,
            DType::F16,
            CompiledDims::STATIC_NK,
        )
        .expect("valid descriptor parameters must not fail");
        assert_eq!(descriptor.shape(), GemmShape::new(64, 4096, 2048));
        assert_eq!(descriptor.cache_key_shape(), GemmShape::new(0, 4096, 2048));
        assert_eq!(descriptor.compiled_dims(), Some(CompiledDims::STATIC_NK));
    }

    // codex-review 指摘（PR #674・P1）対応の回帰テスト: `new`（従来
    // コンストラクタ）は次元特化なしで、`shape()`／`cache_key_shape()`
    // が同一値を返し、`compiled_dims()` は `None` を返すこと（イシュー
    // #519 導入前の契約と同じ）。
    #[test]
    fn new_preserves_legacy_contract_without_compiled_dims() {
        let descriptor = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            2,
            DType::F32,
        )
        .expect("valid descriptor parameters must not fail");
        assert_eq!(descriptor.shape(), GemmShape::new(4096, 4096, 4096));
        assert_eq!(
            descriptor.cache_key_shape(),
            GemmShape::new(4096, 4096, 4096)
        );
        assert_eq!(descriptor.compiled_dims(), None);
    }

    // イシュー #519 fail-closed 契約: 定数化対象次元の実値が 0 だと
    // sentinel（動的次元の正規化値）と曖昧になるため構築時点で拒否する。
    #[test]
    fn new_rejects_zero_value_for_a_compiled_dim() {
        for (shape, compiled_dims) in [
            (GemmShape::new(0, 4096, 4096), CompiledDims::STATIC_MNK),
            (GemmShape::new(4096, 0, 4096), CompiledDims::STATIC_NK),
            (GemmShape::new(4096, 4096, 0), CompiledDims::STATIC_NK),
        ] {
            let result = CudaKernelDescriptor::new_with_compiled_dims(
                "mma_f16",
                shape,
                64,
                128,
                32,
                3,
                DType::F16,
                compiled_dims,
            );
            assert!(
                matches!(result, Err(CudaError::InvalidKernelDescriptor { .. })),
                "shape={shape:?} compiled_dims={compiled_dims:?} must be rejected"
            );
        }
    }

    // 動的次元（`compiled_dims` 非選択）の実値が 0 でも sentinel と同じ
    // 表現になるだけであり拒否されないこと（fail-closed 検査は「定数化
    // 対象かつ 0」の組合せのみを拒否する）。
    #[test]
    fn new_allows_zero_value_for_a_dynamic_dim() {
        let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
            "mma_f16",
            GemmShape::new(0, 4096, 4096),
            64,
            128,
            32,
            3,
            DType::F16,
            CompiledDims::STATIC_NK,
        )
        .expect("dynamic dim with value 0 must not be rejected");
        assert_eq!(descriptor.shape(), GemmShape::new(0, 4096, 4096));
    }

    // 受け入れ基準: `HashMap` でのヒット/ミス動作のスモークテスト。
    #[test]
    fn cache_key_works_as_hashmap_key() {
        let mut cache: HashMap<CudaKernelCacheKey, &str> = HashMap::new();
        cache.insert(sample_key(), "compiled-ptx-placeholder");

        assert_eq!(cache.get(&sample_key()), Some(&"compiled-ptx-placeholder"));

        let miss_key = CudaKernelCacheKey::new(
            sample_descriptor(),
            (9, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_90".to_string()],
            sample_source(),
        );
        assert_eq!(cache.get(&miss_key), None);
    }

    // codex-review P1 是正（C-5・#514）: `Debug` 出力にカーネルソース内容
    // （先頭 40 文字の平文要約を含む旧実装）が漏出しないことを保証する
    // 回帰テスト。`sample_source()` は `mma_f16_source()`（数十行の実
    // カーネルソース）を返すため、その内容の一部（先頭のプリプロセッサ
    // 指令等の識別可能な断片）が `{:?}` 出力に一切含まれないことを検査
    // する。長さ・非可逆フィンガープリントのみが出力される契約
    // （`impl Debug for CudaKernelCacheKey` ドキュメンテーションコメント
    // 参照）。
    #[test]
    fn debug_output_does_not_leak_source_content() {
        let key = sample_key();
        let debug_output = format!("{key:?}");
        let source = sample_source();

        // ソース全文はもちろん、旧実装が漏出させていた先頭 40 文字断片も
        // 含め、ソースからの任意の非自明な部分文字列が出力に現れない
        // ことを確認する。
        let leading_fragment: String = source.chars().take(40).collect();
        assert!(
            !debug_output.contains(&leading_fragment),
            "Debug 出力にソース先頭断片が含まれている（情報露出）: {debug_output}"
        );
        assert!(
            !debug_output.contains(&source),
            "Debug 出力にソース全文が含まれている（情報露出）: {debug_output}"
        );

        // 代わりに長さと非可逆フィンガープリントは出力される契約。
        assert!(debug_output.contains(&source.len().to_string()));
        assert!(debug_output.contains("source_fnv1a64"));
    }

    // イシュー #506（Phase C-2）: FNV-1a 64bit の既知テストベクタ
    // （公開仕様値。実装が標準アルゴリズムと一致することを保証する）。
    #[test]
    fn fnv1a_64_matches_known_test_vectors() {
        assert_eq!(fnv1a_64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
    }

    // 受け入れ基準: ハッシュ関数の決定性（同一入力 → 複数回・別呼び出しで
    // 同一値）。
    #[test]
    fn stable_hash_is_deterministic() {
        let key = sample_key();
        assert_eq!(key.stable_hash(), key.stable_hash());
        assert_eq!(sample_key().stable_hash(), key.stable_hash());
    }

    // 受け入れ基準: 代表ケースの非衝突。`changing_any_field_produces_
    // distinct_key`（既存・derive Hash/Eq 用）と同型のフィールド網羅を
    // `stable_hash()` に対しても行う（DeepGEMM のアーキ違い＝別エントリの
    // 性質を自作ハッシュ側でも担保する）。
    #[test]
    fn stable_hash_changing_any_field_produces_distinct_hash() {
        let base = sample_key();
        let base_hash = base.stable_hash();

        let variants = [
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(2048, 4096, 4096),
                    64,
                    64,
                    32,
                    2,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32_v2",
                    GemmShape::new(4096, 4096, 4096),
                    64,
                    64,
                    32,
                    2,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (9, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (8, 0),
                (13, 0),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_90".to_string()],
                sample_source(),
            ),
            // Cursor Bugbot 指摘（PR #659）: 本テストは
            // `changing_any_field_produces_distinct_key`
            // と同等の全フィールド網羅を謳っていたが、実際には
            // `block_n`/`block_k`/`stages`/`dtype` を変えるケースが
            // 欠けていた。とくに `dtype` は `canonical_bytes` 内の
            // 手書きタグであり、このケースがないと将来タグを誤って
            // 省略しても F32/F16 がディスクキャッシュエントリを
            // 共有する不具合を検出できない。ここで全フィールドを
            // 単独変更するケースを追加する。
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(4096, 4096, 4096),
                    32,
                    64,
                    32,
                    2,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(4096, 4096, 4096),
                    64,
                    32,
                    32,
                    2,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(4096, 4096, 4096),
                    64,
                    64,
                    16,
                    2,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(4096, 4096, 4096),
                    64,
                    64,
                    32,
                    3,
                    DType::F32,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new(
                    "wmma_tf32_f32",
                    GemmShape::new(4096, 4096, 4096),
                    64,
                    64,
                    32,
                    2,
                    DType::F16,
                )
                .unwrap(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                sample_source(),
            ),
            // C-5（#514）: `stable_hash` 側でも `source` フィールド単独の
            // 変更が別ハッシュを生むことを検証する
            // （`changing_any_field_produces_distinct_key` の
            // `different_source` ケースと同型）。
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                format!("{}\n// edited\n", sample_source()),
            ),
        ];

        for variant in &variants {
            assert_ne!(
                base_hash,
                variant.stable_hash(),
                "variant unexpectedly produced the same stable_hash as base"
            );
        }
    }

    // 正準エンコーディング非曖昧性: フラグ境界の移動（["ab","c"] vs
    // ["a","bc"]）が長さプレフィクス方式のため異なるバイト列・異なる
    // ハッシュになること（区切り文字方式だと `"ab" + "c"` == `"a" + "bc"`
    // に縮退しうる、というリスクをここで否定する）。
    #[test]
    fn canonical_encoding_disambiguates_flag_boundaries() {
        let key_ab_c = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["ab".to_string(), "c".to_string()],
            sample_source(),
        );
        let key_a_bc = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["a".to_string(), "bc".to_string()],
            sample_source(),
        );
        assert_ne!(key_ab_c.canonical_bytes(), key_a_bc.canonical_bytes());
        assert_ne!(key_ab_c.stable_hash(), key_a_bc.stable_hash());
    }

    // 正準エンコーディング非曖昧性（C-5・#514）: `compile_flags` 末尾と
    // `source` 先頭の境界も、`canonical_encoding_disambiguates_flag_
    // boundaries` と同じ長さプレフィクス方式の性質により曖昧化しない
    // ことを検証する。区切り文字方式であれば
    // `flags=["x"], source="yz"` と `flags=["x","y"], source="z"`
    // が同一バイト列へ縮退しうる、というリスクをここで否定する。
    #[test]
    fn canonical_encoding_disambiguates_flags_source_boundary() {
        // `compile_flags` の要素数（u32 カウント）は両者とも 1 で揃える。
        // 要素数まで変えると `canonical_bytes` はその時点で既に異なる
        // バイト列になり、区切り文字方式であれば
        // `"x" + "yz"` == `"xy" + "z"` に縮退しうる、という本テストが
        // 否定したい境界曖昧化を実際には検証しないまま通過してしまう
        // （advisor 指摘）。
        let flags_x_source_yz = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["x".to_string()],
            "yz".to_string(),
        );
        let flags_xy_source_z = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["xy".to_string()],
            "z".to_string(),
        );
        assert_ne!(
            flags_x_source_yz.canonical_bytes(),
            flags_xy_source_z.canonical_bytes()
        );
        assert_ne!(
            flags_x_source_yz.stable_hash(),
            flags_xy_source_z.stable_hash()
        );
    }

    // 回帰テスト（イシュー #519・cursor[bot] 指摘・PR #674・High
    // Severity）: `canonical_bytes`／`stable_hash`（ディスク永続キー側）が
    // `compiled_dims` の選択差を検知できることを検証する。修正前は
    // `canonical_bytes` が正規化前の生 `shape` をハッシュし
    // `compiled_dims` を含めていなかったため、同一の生 `shape` を持ち
    // `compiled_dims` のみが異なる（= `cache_key_shape` も異なる）2 つの
    // descriptor が `source` 一致のみで同一 `stable_hash` を共有しえた
    // （メモリ上キー〈`PartialEq`/`Hash` は `cache_key_shape`・
    // `compiled_dims` 済みで正しかった〉とディスクキーの乖離。キャッシュ
    // 汚染・誤ヒットのリスク）。
    #[test]
    fn stable_hash_distinguishes_compiled_dims_with_same_raw_shape() {
        let same_raw_shape = GemmShape::new(4096, 4096, 4096);

        let static_nk = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                same_raw_shape,
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_NK,
            )
            .expect("valid descriptor parameters must not fail"),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );
        let static_mnk = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                same_raw_shape,
                64,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .expect("valid descriptor parameters must not fail"),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
            sample_source(),
        );

        // 前提: 生 `shape` は同一だが `cache_key_shape` は異なる
        // （`STATIC_NK` は M を sentinel `0` に正規化、`STATIC_MNK` は
        // しない）。本テストが実際に狙った境界を検証していることを
        // 明示する。
        assert_eq!(
            static_nk.descriptor().shape(),
            static_mnk.descriptor().shape()
        );
        assert_ne!(
            static_nk.descriptor().cache_key_shape(),
            static_mnk.descriptor().cache_key_shape()
        );

        assert_ne!(
            static_nk.canonical_bytes(),
            static_mnk.canonical_bytes(),
            "compiled_dims 違いが canonical_bytes に反映されていない（ディスクキャッシュ汚染のリスク）"
        );
        assert_ne!(
            static_nk.stable_hash(),
            static_mnk.stable_hash(),
            "compiled_dims 違いが stable_hash に反映されていない（ディスクキャッシュ汚染のリスク）"
        );
    }

    // 命名規則: `cache_entry_dir_name()` が `kernel.<name>.` + 16 桁小文字
    // hex の形式であり、パスセパレータを含まないこと（DeepGEMM
    // `compiler.hpp:102` の hex16 に対応。イシュー #506）。
    #[test]
    fn cache_entry_dir_name_has_expected_format() {
        let key = sample_key();
        let name = key.cache_entry_dir_name().expect("must succeed");

        let expected_prefix = format!("kernel.{}.", key.descriptor().kernel_name());
        assert!(
            name.starts_with(&expected_prefix),
            "{name:?} must start with {expected_prefix:?}"
        );

        let hash_part = &name[expected_prefix.len()..];
        assert_eq!(hash_part.len(), 16, "hash part must be 16 hex digits");
        assert!(
            hash_part
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "hash part must be lowercase hex: {hash_part:?}"
        );
        assert!(!name.contains('/') && !name.contains('\\') && !name.contains(".."));
    }

    // `resolve_cache_root` テスト共通の `workspace_root`（イシュー #506・
    // PR #659 codex-review P0 再指摘対応）: 既存の全テストケース
    // （`/opt/rust-ai-cache`・`/home/user/...`）とは無関係なパスにして
    // おき、containment 検証が既存のフォールバック挙動を壊さないことを
    // 既存テスト自体で回帰確認する（下記の専用テストは逆に workspace_root
    // 配下を指すケースを検証する）。
    fn unrelated_workspace_root() -> PathBuf {
        PathBuf::from("/workspace/repository")
    }

    // キャッシュルート解決: env 上書き（override）が XDG_CACHE_HOME・
    // HOME より優先されること。
    #[test]
    fn resolve_cache_root_prefers_override() {
        let root = resolve_cache_root(
            &unrelated_workspace_root(),
            Some(OsStr::new("/opt/rust-ai-cache")),
            Some(OsStr::new("/home/user/.cache")),
            Some(OsStr::new("/home/user")),
        )
        .expect("must succeed");
        assert_eq!(root, PathBuf::from("/opt/rust-ai-cache"));
    }

    // キャッシュルート解決: override 欠落時は XDG_CACHE_HOME にフォール
    // バックし、`rust-ai-library/cuda` サブパスを付加すること。
    #[test]
    fn resolve_cache_root_falls_back_to_xdg_cache_home() {
        let root = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            Some(OsStr::new("/home/user/.cache")),
            Some(OsStr::new("/home/user")),
        )
        .expect("must succeed");
        assert_eq!(
            root,
            PathBuf::from("/home/user/.cache/rust-ai-library/cuda")
        );
    }

    // キャッシュルート解決: override・XDG_CACHE_HOME 欠落時は HOME に
    // フォールバックし `.cache/rust-ai-library/cuda` を付加すること。
    #[test]
    fn resolve_cache_root_falls_back_to_home() {
        let root = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            None,
            Some(OsStr::new("/home/user")),
        )
        .expect("must succeed");
        assert_eq!(
            root,
            PathBuf::from("/home/user/.cache/rust-ai-library/cuda")
        );
    }

    // キャッシュルート解決: 全欠落時は `CacheDirUnavailable`（panic なし）。
    #[test]
    fn resolve_cache_root_errs_when_all_missing() {
        let result = resolve_cache_root(&unrelated_workspace_root(), None, None, None);
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: 空文字列の override は拒否する。
    #[test]
    fn resolve_cache_root_rejects_empty_override() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            Some(OsStr::new("")),
            Some(OsStr::new("/home/user/.cache")),
            Some(OsStr::new("/home/user")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: 相対パスの override は拒否する（リポジトリツリー内へ
    // キャッシュが落ちるのを防ぐ。イシュー #506 §4.4）。
    #[test]
    fn resolve_cache_root_rejects_relative_override() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            Some(OsStr::new("relative/cache/dir")),
            Some(OsStr::new("/home/user/.cache")),
            Some(OsStr::new("/home/user")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: 相対パスの XDG_CACHE_HOME は拒否する（PR #659 レビュー
    // 指摘。`XDG_CACHE_HOME=.` のようにカレントディレクトリ（呼び出し
    // コンテキストによってはリポジトリツリー内）を指す相対パスを未検証で
    // `Path::join` すると、override 検証を回避してキャッシュがリポジトリ
    // ツリー内へ落ちてしまうため、override と同じ fail-closed 検証を課す）。
    #[test]
    fn resolve_cache_root_rejects_relative_xdg_cache_home() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            Some(OsStr::new("relative/xdg/cache")),
            Some(OsStr::new("/home/user")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: 相対パスの HOME は拒否する（PR #659 レビュー指摘。
    // `HOME=.` 等の相対パスフォールバックを未検証で許すと override・
    // XDG_CACHE_HOME と同じくリポジトリツリー内へキャッシュが落ちうる）。
    #[test]
    fn resolve_cache_root_rejects_relative_home() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            None,
            Some(OsStr::new("relative/home")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証（回帰テスト）: 空文字列の XDG_CACHE_HOME は「未設定」
    // 扱いで HOME へフォールバックせず拒否する（PR #659 codex-review P0
    // 指摘。旧実装は `if let Some(xdg) = xdg_cache_home && !xdg.is_empty()`
    // というガード構造のため、`Some("")`（環境変数は設定済みだが値が空）が
    // 分岐を素通りして有効な HOME があれば HOME 側の解決結果を返してしまう
    // fail-closed 契約違反があった。`docs/cuda-jit-cache-design.md:19-22`
    // の「三者とも空文字列を CacheDirUnavailable として拒否する」方針との
    // 整合を検証する）。
    #[test]
    fn resolve_cache_root_rejects_empty_xdg_cache_home_even_with_valid_home() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("/home/user")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証（回帰テスト）: 空文字列の HOME も同様に拒否する
    // （override・XDG_CACHE_HOME 双方が未設定かつ HOME のみ空文字列で
    // 設定されているケース。上記 XDG のケースと対称に fail-closed である
    // ことを確認する。PR #659 codex-review P0 指摘）。
    #[test]
    fn resolve_cache_root_rejects_empty_home() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            None,
            None,
            Some(OsStr::new("")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証（PR #659 codex-review P0 再指摘そのものの再現テスト）:
    // `RUST_AI_CUDA_CACHE_DIR` が `workspace_root` 配下を指す絶対パスの
    // 場合は拒否する。codex-review が指摘した具体例
    // `/workspace/repository/cache` をそのまま使う。
    #[test]
    fn resolve_cache_root_rejects_override_within_workspace_root() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            Some(OsStr::new("/workspace/repository/cache")),
            None,
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証（PR #659 codex-review P0 再指摘。`..` を含む絶対パスに
    // よる字句上の回避を防ぐ）: `/tmp/../workspace/repository/cache` は
    // `Path::starts_with` の素朴なコンポーネント比較なら
    // `workspace_root`（`/workspace/repository`）と先頭コンポーネントが
    // 一致せず素通りしてしまうが、`..` 折り畳み後は `workspace_root` 配下
    // になるため [`path_lexically_within`] の正規化込み比較で拒否される
    // ことを確認する（codex-review 指摘の具体例そのもの）。
    #[test]
    fn resolve_cache_root_rejects_override_within_workspace_root_via_parent_dir_traversal() {
        let result = resolve_cache_root(
            &unrelated_workspace_root(),
            Some(OsStr::new("/tmp/../workspace/repository/cache")),
            None,
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: `XDG_CACHE_HOME` から導出したキャッシュルートが
    // `workspace_root` 配下になる場合も override と同様に拒否する
    // （PR #659 codex-review P0 再指摘: 3 分岐すべてを検証対象にする）。
    #[test]
    fn resolve_cache_root_rejects_xdg_cache_home_within_workspace_root() {
        let result = resolve_cache_root(
            Path::new("/workspace/repository"),
            None,
            Some(OsStr::new("/workspace/repository/.cache")),
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: `HOME` から導出したキャッシュルートが `workspace_root`
    // 配下になる場合も同様に拒否する（3 分岐目。上記 2 テストと対称）。
    #[test]
    fn resolve_cache_root_rejects_home_within_workspace_root() {
        let result = resolve_cache_root(
            Path::new("/workspace/repository"),
            None,
            None,
            Some(OsStr::new("/workspace/repository")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 回帰ガード（advisor 指摘）: `workspace_root` containment 検証の追加が
    // 「カレントディレクトリを境界に使う」という誤った実装に陥っていない
    // ことを確認する。`workspace_root` が解決結果と無関係な場所を指す限り、
    // XDG_CACHE_HOME／HOME フォールバックは引き続き成功しなければならない
    // （`resolve_cache_root_falls_back_to_home` 等の既存テストと合わせ、
    // workspace_root がたまたまホームディレクトリ等と重ならない限り誤検知
    // しないことの明示的な回帰確認）。
    #[test]
    fn resolve_cache_root_succeeds_when_workspace_root_is_elsewhere() {
        let root = resolve_cache_root(
            Path::new("/some/other/workspace"),
            None,
            None,
            Some(OsStr::new("/home/user")),
        )
        .expect("must succeed: workspace_root is unrelated to the resolved cache root");
        assert_eq!(
            root,
            PathBuf::from("/home/user/.cache/rust-ai-library/cuda")
        );
    }

    // PR #659 codex-review Bugbot 指摘の再現テスト: `workspace_root` が
    // 相対パスだと `path_lexically_within` の `starts_with` 比較が絶対
    // パスの候補と食い違い containment 判定が fail-open になる（本来
    // ブロックすべきリポジトリ内キャッシュルートを受理してしまう）。
    // `resolve_cache_root` は 3 分岐へ入る前に `workspace_root` の絶対
    // パス性を検証して fail-closed で拒否しなければならない。
    #[test]
    fn resolve_cache_root_rejects_relative_workspace_root_even_when_override_is_within_it() {
        let result = resolve_cache_root(
            Path::new("workspace/repository"),
            Some(OsStr::new("/workspace/repository/cache")),
            None,
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 空文字列の `workspace_root`（`Path::new("")` は相対パス扱い）も
    // 同じ fail-closed 経路で拒否されることを確認する。
    #[test]
    fn resolve_cache_root_rejects_empty_workspace_root() {
        let result = resolve_cache_root(
            Path::new(""),
            None,
            Some(OsStr::new("/home/user/.cache")),
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 相対 `workspace_root` の拒否が特定の分岐（override）だけでなく
    // 入口の共通ガードで行われていることを、XDG_CACHE_HOME 分岐でも
    // 確認する（分岐ごとに個別実装していないことの回帰確認）。
    #[test]
    fn resolve_cache_root_rejects_relative_workspace_root_via_xdg_branch() {
        let result = resolve_cache_root(
            Path::new("relative/workspace"),
            None,
            Some(OsStr::new("/home/user/.cache")),
            None,
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // `path_lexically_within` の単体テスト（[`resolve_cache_root`] の
    // `workspace_root` containment 検証で使う。PR #659 codex-review P0
    // 再指摘対応で呼び出しを復元した）。
    #[test]
    fn path_lexically_within_detects_containment_after_normalizing_parent_dir() {
        // `..` 折り畳み後は `/path/to/repository/cache` となり
        // `/path/to/repository` 配下に収まる。
        assert!(path_lexically_within(
            Path::new("/path/to/outside/../repository/cache"),
            Path::new("/path/to/repository"),
        ));
    }

    #[test]
    fn path_lexically_within_rejects_sibling_directory() {
        // 兄弟ディレクトリ（`/path/to/repository-extra`）は
        // `Path::starts_with` がコンポーネント単位の比較であり文字列前方
        // 一致でないため、containment と誤判定されない。
        assert!(!path_lexically_within(
            Path::new("/path/to/repository-extra/cache"),
            Path::new("/path/to/repository"),
        ));
    }

    // `lexically_normalize` の単体テスト: `..` の畳み込み・ルートを
    // 越える `..` の読み捨て・`.` の除去を直接検証する（fs I/O なしの
    // 純関数であるため実ファイルシステムに依存せずテストできる）。
    #[test]
    fn lexically_normalize_collapses_parent_dir_components() {
        assert_eq!(
            lexically_normalize(Path::new("/path/to/outside/../repository/cache")),
            PathBuf::from("/path/to/repository/cache")
        );
    }

    #[test]
    fn lexically_normalize_ignores_parent_dir_past_root() {
        // ルート直下からの `..` は OS のパス解決と同様にルート自身へ
        // とどまる（それ以上遡らない）。
        assert_eq!(
            lexically_normalize(Path::new("/../repository")),
            PathBuf::from("/repository")
        );
    }

    #[test]
    fn lexically_normalize_removes_cur_dir_components() {
        assert_eq!(
            lexically_normalize(Path::new("/path/./to/./repository")),
            PathBuf::from("/path/to/repository")
        );
    }

    #[test]
    fn lexically_normalize_preserves_leading_parent_dir_in_relative_path() {
        // 絶対パス前提の呼び出し元（`resolve_cache_root`）では通常
        // 到達しない経路だが、純関数としての境界挙動を明示する。
        assert_eq!(
            lexically_normalize(Path::new("../repository")),
            PathBuf::from("../repository")
        );
    }

    // トラバーサル防御（A03）: `cache_entry_path`（実体は
    // `cache_entry_path_in`）の組み立て結果が常に `root` 配下
    // （`starts_with(root)`）であること。既存の
    // `new_rejects_path_traversal_kernel_name` と合わせ二層で担保する
    // （イシュー #506 §6）。`cache_entry_path` 自体ではなく
    // `cache_entry_path_in` を直接呼ぶのは、テスト実行環境の実 `HOME` に
    // 依存させずに `root` を注入で決定化するため（`resolve_cache_root` と
    // 同じパターン）。
    #[test]
    fn cache_entry_path_stays_within_cache_root() {
        let root = PathBuf::from("/opt/rust-ai-cache");
        let key = sample_key();

        let entry_path = cache_entry_path_in(&root, &key).expect("must succeed");

        assert!(entry_path.starts_with(&root));
        let entry_name = key.cache_entry_dir_name().expect("must succeed");
        assert_eq!(entry_path, root.join(entry_name));
    }

    // ------------------------------------------------------------------
    // キャッシュ I/O（イシュー #509・Phase C-3）のユニットテスト。
    //
    // 実プロセス環境変数（`RUST_AI_CUDA_CACHE_DIR` 等）を書き換えると
    // 並行テスト実行時に競合するため、`*_in` 系（`store_cache_entry_in`・
    // `load_cache_entry_in`・`ensure_cache_root_in`）を直接呼び、`root`
    // を `std::env::temp_dir()` 配下の一意なディレクトリで注入する
    // （`cache_entry_path_in` と同じ「注入で決定化」パターン）。網羅的な
    // ヒット/ミス・並行競合・破損検出の回帰テスト拡充は C-10（#529）の
    // スコープであり、ここでは受け入れ基準を直接検証する最小限に留める
    // （実装計画 §1 スコープ境界節）。
    // ------------------------------------------------------------------

    /// テスト用に一意な一時ディレクトリを払い出す（プロセス内
    /// `AtomicU64` カウンタ＋PID で並行テスト実行時の衝突を避ける）。
    /// 呼び出し元がテスト末尾で `remove_dir_all` して片付ける。
    fn fresh_temp_dir(label: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "rust-ai-library-cache-test.{label}.{}.{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("failed to create test temp dir");
        dir
    }

    // 受け入れ基準（実装計画 §5）: store → load ラウンドトリップ。両
    // ファイルが存在し内容が一致すること。
    #[test]
    fn store_then_load_roundtrips_entry_contents() {
        let root = fresh_temp_dir("roundtrip");
        let key = sample_key();

        let stored = store_cache_entry_in(&root, &key, "// kernel.cu source", "// kernel.ptx body")
            .expect("store must succeed");
        assert!(validate_cache_entry(&stored));

        let loaded = load_cache_entry_in(&root, &key, "// kernel.cu source")
            .expect("load must succeed")
            .expect("entry must be a hit after store");
        assert_eq!(loaded.kernel_cu, "// kernel.cu source");
        assert_eq!(loaded.kernel_ptx, "// kernel.ptx body");

        // エラー経路・正常経路とも一時ディレクトリが残存しないこと。
        let leftover_tmp_dirs = fs::read_dir(&root)
            .expect("root must be readable")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp."))
            .count();
        assert_eq!(leftover_tmp_dirs, 0, "temp directories must not remain");

        let _ = fs::remove_dir_all(&root);
    }

    // symlink 脱出防御（イシュー #509 codex-review P0 指摘対応。
    // `cfg(unix)`）: エントリディレクトリ自体は正規に store されたものだが、
    // その中の `kernel.ptx` をルート外の任意ファイルへの symlink に
    // 差し替えた場合、`validate_cache_entry`（ひいては `load_cache_entry_in`）
    // が symlink を追跡せず「無効なエントリ」として拒否すること
    // （symlink 先の任意ファイル内容がキャッシュヒットとして読み出されない
    // ことの回帰確認）。
    #[cfg(unix)]
    #[test]
    fn load_rejects_entry_whose_ptx_file_is_replaced_with_a_symlink() {
        let root = fresh_temp_dir("symlink-entry-ptx");
        let key = sample_key();

        let entry_dir = store_cache_entry_in(&root, &key, "kernel source", "legit ptx")
            .expect("store must succeed");

        // ルート外に「秘密ファイル」を用意し、キャッシュルート外の
        // 任意ファイルを指す想定を再現する。
        let outside_dir = fresh_temp_dir("symlink-entry-secret");
        let secret_path = outside_dir.join("secret.ptx");
        fs::write(&secret_path, "leaked-outside-root-ptx").expect("must write secret file");

        let ptx_path = entry_dir.join(CACHE_ENTRY_PTX_FILE);
        fs::remove_file(&ptx_path).expect("must remove legit ptx file before replacing");
        std::os::unix::fs::symlink(&secret_path, &ptx_path)
            .expect("must replace kernel.ptx with a symlink pointing outside the cache root");

        assert!(
            !validate_cache_entry(&entry_dir),
            "entry with a symlinked kernel.ptx must be rejected, not treated as a valid entry"
        );

        let loaded = load_cache_entry_in(&root, &key, "kernel source")
            .expect("load must not error, only miss");
        assert!(
            loaded.is_none(),
            "load must treat a symlink-replaced entry as a cache miss, never follow the symlink"
        );

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // symlink 脱出防御（`cfg(unix)`）: エントリディレクトリ自体
    // （`kernel.<hash>`）が symlink に置換されているケースも拒否すること。
    #[cfg(unix)]
    #[test]
    fn load_rejects_entry_directory_replaced_with_a_symlink() {
        let root = fresh_temp_dir("symlink-entry-dir");
        let key = sample_key();

        let outside_dir = fresh_temp_dir("symlink-entry-dir-target");
        fs::write(outside_dir.join(CACHE_ENTRY_SOURCE_FILE), "outside cu")
            .expect("must write outside kernel.cu");
        fs::write(outside_dir.join(CACHE_ENTRY_PTX_FILE), "outside ptx")
            .expect("must write outside kernel.ptx");

        let entry_name = key.cache_entry_dir_name().expect("must succeed");
        let entry_path = root.join(entry_name);
        std::os::unix::fs::symlink(&outside_dir, &entry_path)
            .expect("must create entry dir as a symlink pointing outside the cache root");

        assert!(
            !validate_cache_entry(&entry_path),
            "entry directory replaced with a symlink must be rejected"
        );
        let loaded = load_cache_entry_in(&root, &key, "").expect("load must not error, only miss");
        assert!(loaded.is_none());

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // 特殊ファイル拒否（イシュー #509 codex-review P0 指摘対応。
    // `cfg(unix)`）: `kernel.ptx` を FIFO（named pipe）に差し替えると、
    // `O_NONBLOCK` なしの旧実装では writer 不在のまま `open` がハングし
    // うる（DoS）。本テストは `load_cache_entry_in` の呼び出しを別
    // スレッドで実行し、有界時間内に `Ok(None)`（ミス扱い。ハングせず
    // 拒否）で返ることを検証する。ハングした場合はスレッド join が
    // タイムアウトしテストを fail させる（無限ハングでテストプロセス
    // 自体が停止するのを防ぐ）。
    #[cfg(unix)]
    #[test]
    fn load_rejects_fifo_cache_entry_file_without_hanging() {
        let root = fresh_temp_dir("fifo-entry-ptx");
        let key = sample_key();

        let entry_dir = store_cache_entry_in(&root, &key, "kernel source", "legit ptx")
            .expect("store must succeed");

        let ptx_path = entry_dir.join(CACHE_ENTRY_PTX_FILE);
        fs::remove_file(&ptx_path).expect("must remove legit ptx file before replacing");
        let status = std::process::Command::new("mkfifo")
            .arg(&ptx_path)
            .status()
            .expect("mkfifo command must be spawnable in the test environment");
        assert!(status.success(), "mkfifo must succeed");

        let (tx, rx) = std::sync::mpsc::channel();
        let root_for_thread = root.clone();
        std::thread::spawn(move || {
            let result = load_cache_entry_in(&root_for_thread, &sample_key(), "kernel source");
            // reader が存在しないためテスト終了後も FIFO の open を試みる
            // 別プロセス/スレッドが残らないよう、成否に関わらず結果だけ
            // 送る（本スレッド自体は fn 終了で自然に終わる）。
            let _ = tx.send(result);
        });

        let result = rx.recv_timeout(std::time::Duration::from_secs(10)).expect(
            "load_cache_entry_in must return within a bounded time \
                 (a FIFO cache entry must never hang the open call)",
        );
        let loaded = result.expect("load must not error, only miss");
        assert!(
            loaded.is_none(),
            "load must treat a FIFO-replaced entry as a cache miss, never read from it"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // 読み込みサイズ上限（イシュー #509 codex-review P0 指摘対応。
    // `cfg(unix)`）: `kernel.ptx` を [`MAX_CACHE_ENTRY_FILE_BYTES`] 超の
    // 巨大ファイルへ差し替えても、`load_cache_entry_in` が無制限に読み
    // 込まず `Ok(None)`（ミス扱い）で拒否すること。
    #[cfg(unix)]
    #[test]
    fn load_rejects_oversized_cache_entry_file() {
        let root = fresh_temp_dir("oversized-entry-ptx");
        let key = sample_key();

        let entry_dir = store_cache_entry_in(&root, &key, "kernel source", "legit ptx")
            .expect("store must succeed");

        let ptx_path = entry_dir.join(CACHE_ENTRY_PTX_FILE);
        let oversized = vec![b'A'; (MAX_CACHE_ENTRY_FILE_BYTES + 1) as usize];
        fs::write(&ptx_path, &oversized).expect("must write oversized ptx file");

        let loaded = load_cache_entry_in(&root, &key, "kernel source")
            .expect("load must not error, only miss");
        assert!(
            loaded.is_none(),
            "load must treat an oversized entry file as a cache miss, never read it unbounded"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // [`open_nofollow`]（[`O_NOFOLLOW`]）が実行プラットフォーム上で実際に
    // symlink を拒否することの直接回帰確認（イシュー #509 codex-review
    // P0 指摘対応）。`O_NOFOLLOW` の数値定数は `target_os` ごとに手書き
    // 定義しているため（`libc`／`rustix` は許容依存 8 区分外。
    // [`O_NOFOLLOW`] のコメント参照）、値の取り違えを本テストで検知する。
    // errno の具体値までは断定せず `is_err()` のみを検査する（Linux は
    // `ELOOP`、実装によっては別の errno になりうるため。プラットフォーム
    // 間の errno 差異に依存しない検証にする）。
    #[cfg(unix)]
    #[test]
    fn open_nofollow_rejects_a_symlinked_path() {
        let root = fresh_temp_dir("open-nofollow");
        let real_file = root.join("real.txt");
        fs::write(&real_file, "real content").expect("must write real file");
        let symlink_path = root.join("via-symlink.txt");
        std::os::unix::fs::symlink(&real_file, &symlink_path)
            .expect("must create symlink to real file");

        assert!(
            open_nofollow(&symlink_path).is_err(),
            "open_nofollow must refuse to open a path whose final component is a symlink"
        );
        // 対照実験: symlink でない通常ファイルは開けること（定数の値自体が
        // 誤って全 open を拒否する側に倒れていないことの検算）。
        assert!(
            open_nofollow(&real_file).is_ok(),
            "open_nofollow must still open a plain, non-symlink file"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // [`open_dir_nofollow`]（`O_NOFOLLOW | O_DIRECTORY`）が symlink・
    // 非ディレクトリのいずれも拒否し、通常ディレクトリは開けることの
    // 直接回帰確認（イシュー #509 codex-review P0 再指摘対応）。
    // [`O_DIRECTORY`] の数値定数は `target_os` ごとに手書き定義している
    // ため（[`O_NOFOLLOW`] と同じ理由）、値の取り違えを本テストで検知
    // する。errno の具体値までは断定せず `is_err()` のみを検査する
    // （`open_nofollow_rejects_a_symlinked_path` と同じ方針）。
    #[cfg(unix)]
    #[test]
    fn open_dir_nofollow_rejects_symlink_and_plain_file_but_accepts_a_plain_dir() {
        let root = fresh_temp_dir("open-dir-nofollow");

        let real_dir = root.join("real-dir");
        fs::create_dir(&real_dir).expect("must create real directory");
        assert!(
            open_dir_nofollow(&real_dir).is_ok(),
            "open_dir_nofollow must open a plain, non-symlink directory"
        );

        let symlink_to_dir = root.join("via-symlink-dir");
        std::os::unix::fs::symlink(&real_dir, &symlink_to_dir)
            .expect("must create symlink to real directory");
        assert!(
            open_dir_nofollow(&symlink_to_dir).is_err(),
            "open_dir_nofollow must refuse to open a path whose final component is a symlink, \
             even when it resolves to a plain directory"
        );

        let real_file = root.join("real-file");
        fs::write(&real_file, "not a directory").expect("must write real file");
        assert!(
            open_dir_nofollow(&real_file).is_err(),
            "open_dir_nofollow must refuse a path whose final component is a plain file \
             (O_DIRECTORY must reject non-directories)"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // [`is_confirmed_non_directory_open_error`] が「確実に非ディレクトリ」
    // （`ELOOP`／`ENOTDIR`）と「一時的・権限系エラー」（`EACCES`／`EMFILE`
    // 等）を区別することの回帰確認（イシュー #509 PR #677 Cursor Bugbot
    // 再指摘対応: 旧実装は open 失敗の原因種別を区別せず一律に
    // `remove_dir` していたため、fd 枯渇・権限不足時に peer プロセスが
    // 作成した正当な空ディレクトリを誤って削除しうる問題があった）。
    #[cfg(unix)]
    #[test]
    fn is_confirmed_non_directory_open_error_distinguishes_eloop_enotdir_from_transient_errors() {
        let root = fresh_temp_dir("confirmed-non-dir-classification");

        let real_dir = root.join("real-dir");
        fs::create_dir(&real_dir).expect("must create real directory");
        let symlink_to_dir = root.join("via-symlink-dir");
        std::os::unix::fs::symlink(&real_dir, &symlink_to_dir)
            .expect("must create symlink to real directory");
        let eloop_err =
            open_dir_nofollow(&symlink_to_dir).expect_err("O_NOFOLLOW must reject the symlink");
        assert!(
            is_confirmed_non_directory_open_error(&eloop_err),
            "an O_NOFOLLOW rejection of a symlink (ELOOP) must be classified as confirmed \
             non-directory: {eloop_err}"
        );

        let real_file = root.join("real-file");
        fs::write(&real_file, "not a directory").expect("must write real file");
        let enotdir_err =
            open_dir_nofollow(&real_file).expect_err("O_DIRECTORY must reject the plain file");
        assert!(
            is_confirmed_non_directory_open_error(&enotdir_err),
            "an O_DIRECTORY rejection of a plain file (ENOTDIR) must be classified as confirmed \
             non-directory: {enotdir_err}"
        );

        // `EACCES`（権限不足）・`EMFILE`（fd 枯渇）は対象が非ディレクトリ
        // だと確定させない。実際のシステムコールを再現困難な状況（fd
        // 枯渇等）まで踏み込まず、`raw_os_error` のみを見る本関数の契約を
        // 直接検証する（[`is_confirmed_non_directory_open_error`] は
        // `std::io::Error` の中身のみを見て判定するため、合成した
        // `Error::from_raw_os_error` で十分）。
        let eacces_err = std::io::Error::from_raw_os_error(libc_like_errno::EACCES);
        assert!(
            !is_confirmed_non_directory_open_error(&eacces_err),
            "EACCES must not be classified as confirmed non-directory (would wrongly delete a \
             directory that could not be opened due to a permission error)"
        );

        let emfile_err = std::io::Error::from_raw_os_error(libc_like_errno::EMFILE);
        assert!(
            !is_confirmed_non_directory_open_error(&emfile_err),
            "EMFILE must not be classified as confirmed non-directory (would wrongly delete a \
             peer-created directory during fd exhaustion)"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // `EACCES`／`EMFILE` の errno 生値（Linux・macOS で共通の値。POSIX
    // 標準の基本エラー番号のため、両プラットフォームで一致する）。
    // `libc`／`rustix` クレートを追加せずテスト専用の合成エラーを組み
    // 立てるための最小限の定数（許容依存 8 区分外のため追加不可。
    // [`O_NOFOLLOW`] 定数のコメントと同じ制約）。
    #[cfg(unix)]
    mod libc_like_errno {
        pub(super) const EACCES: i32 = 13;
        pub(super) const EMFILE: i32 = 24;
    }

    // 未書き込みキーの load はミス（`Ok(None)`）を返すこと。
    #[test]
    fn load_returns_none_when_entry_absent() {
        let root = fresh_temp_dir("miss");
        let key = sample_key();

        let loaded = load_cache_entry_in(&root, &key, "irrelevant").expect("load must succeed");
        assert!(loaded.is_none());

        let _ = fs::remove_dir_all(&root);
    }

    // 受け入れ基準 1: 並行競合（先着吸収）。同一キーで 2 回 store しても
    // 両方 `Ok`、エントリは 1 つ・内容は 1 回目のもののまま破壊されない
    // （2 回目の rename は失敗し「他プロセス先着」として吸収される）。
    #[test]
    fn store_twice_absorbs_second_writer_as_success() {
        let root = fresh_temp_dir("double-store");
        let key = sample_key();

        let first = store_cache_entry_in(&root, &key, "first.cu", "first.ptx")
            .expect("first store must succeed");
        let second = store_cache_entry_in(&root, &key, "second.cu", "second.ptx")
            .expect("second store must be absorbed as success, not fail");

        assert_eq!(first, second);
        let loaded = load_cache_entry_in(&root, &key, "first.cu")
            .expect("load must succeed")
            .expect("entry must exist");
        // 先着（1 回目）の内容が保たれ、2 回目の書き込みで上書きされない
        // こと（rename 失敗時に自分の一時ディレクトリを捨てる契約）。
        assert_eq!(loaded.kernel_cu, "first.cu");
        assert_eq!(loaded.kernel_ptx, "first.ptx");

        let _ = fs::remove_dir_all(&root);
    }

    // イシュー #509 PR #677 codex-review P0 再指摘の直接回帰テスト:
    // 「`ensure_cache_root` が返した canonical path が fd として固定
    // されず、後続の書き込みがパスを再解決している。検証後にキャッシュ
    // ルートを rename して同名の symlink に差し替えられると、一時
    // ディレクトリと `kernel.cu`／`kernel.ptx` が containment 検証外へ
    // 作成される」を、`store_cache_entry_in` が内部で行う手順
    // （`open_dir_nofollow(root)` による pin → `store_cache_entry_at` へ
    // 委譲）を直接呼ぶことでタイミング競合なしに決定的に再現する。
    //
    // `root_fd` を pin した**後**に元のパス（`real_root`）にあった実体を
    // 別名（`moved_aside`）へ rename でどかし、元のパスへは無関係な
    // `attacker_target` を指す symlink を差し替えで置く（攻撃者が root
    // pin と書き込みの間で祖先を入れ替えるシナリオを模す。`rmdir` では
    // なく `rename` でどかすのは、ディレクトリを完全にリンクされた状態
    // のまま維持するため。`rmdir` 後の unlink 済みディレクトリへの子
    // 作成可否はファイルシステム依存の未規定動作であり、本テストの
    // 意図する検証〈pin 済み fd が指す実体が正しい場所であること〉とは
    // 無関係な不確実性を持ち込むため避ける）。pin 済み fd はこの
    // 差し替えの影響を受けない（POSIX の `openat`／`mkdirat`／`rename` は
    // dirfd が指す実体を直接操作し、パス文字列を再解決しない）ため、
    // `store_cache_entry_at` が fd 相対操作のみで書き込みを行っていれば、
    // エントリは常に元の実体（`moved_aside`。pin 時点で `real_root` に
    // あったディレクトリそのもの）側に作られ、`attacker_target` 側には
    // 一切作られないはずである。
    #[cfg(unix)]
    #[test]
    fn store_at_writes_through_pinned_root_fd_even_after_root_path_is_swapped_to_symlink() {
        let real_root = fresh_temp_dir("pin-real-root");
        let moved_aside = fresh_temp_dir("pin-moved-aside-placeholder");
        // `fresh_temp_dir` は作成まで行うため、rename の宛先として使う
        // 前に一旦空にしておく（`fs::rename` はディレクトリ同士でも
        // 宛先が空ディレクトリなら置換できるが、ここでは単純に削除して
        // から rename する）。
        fs::remove_dir(&moved_aside).expect("must clear rename destination placeholder");
        let attacker_target = fresh_temp_dir("pin-attacker-target");
        let key = sample_key();

        // `store_cache_entry_in` 冒頭と同じ手順: 実ディレクトリを一度だけ
        // pin する。
        let root_fd = open_dir_nofollow(&real_root).expect("must pin real cache root");

        // pin 後に実ディレクトリを別名へどかし、元のパスへ無関係な
        // ディレクトリへの symlink を差し替える。
        fs::rename(&real_root, &moved_aside).expect("must move the real root directory aside");
        std::os::unix::fs::symlink(&attacker_target, &real_root)
            .expect("must plant a symlink at the original root path");

        let final_dir = cache_entry_path_in(&real_root, &key).expect("must compute final dir");
        let entry_name = key.cache_entry_dir_name().expect("must compute entry name");

        store_cache_entry_at(&root_fd, &final_dir, &entry_name, "pinned.cu", "pinned.ptx")
            .expect("store must still succeed via the pinned fd, unaffected by the symlink swap");

        // 実体は pin 済み fd（差し替え前の実ディレクトリ、現在は
        // `moved_aside` の位置）側に作られていること。
        assert!(
            validate_cache_entry_at(&root_fd, &entry_name),
            "entry must land in the directory that was pinned before the symlink swap"
        );
        assert!(
            moved_aside
                .join(&entry_name)
                .join(CACHE_ENTRY_SOURCE_FILE)
                .is_file(),
            "entry must be observable by path at the directory's post-rename location"
        );

        // symlink 先（攻撃者が用意した無関係なディレクトリ）には一切
        // 作られていないこと（containment 突破の直接検証）。
        assert!(
            fs::read_dir(&attacker_target)
                .expect("attacker_target must still be readable")
                .next()
                .is_none(),
            "entry must not leak into the symlink target planted after pinning"
        );

        drop(root_fd);
        let _ = fs::remove_dir_all(&moved_aside);
        let _ = fs::remove_file(&real_root);
        let _ = fs::remove_dir_all(&attacker_target);
    }

    // 回帰テスト（イシュー #509 PR #677 codex-review P0 指摘対応）:
    // `store_cache_entry_at` の後始末（一時ディレクトリ・退避ディレクトリ
    // の削除）が `root_fd` 起点の fd 相対操作であり、`root_fd` を pin した
    // 後に `root` 自体が symlink へ差し替えられても、削除対象を pin 済み
    // の実体（`moved_aside`）だけに限定し symlink 先（`attacker_target`）
    // には一切触れないことを検証する。
    //
    // 上のテスト（`store_at_writes_through_pinned_root_fd_even_after_root_path_is_swapped_to_symlink`）
    // は成功系（rename が一発で成功する経路）のみを通るため、後始末の
    // 削除コード（[`remove_cache_entry_pinned`]）を一切経由しない。本
    // テストは事前に破損エントリを置いて「破損エントリ置換」分岐（受け
    // 入れ基準 2）を強制的に踏ませ、削除コードそのものを symlink 差し
    // 替え下で実行させる。
    #[cfg(unix)]
    #[test]
    fn store_at_cleanup_stays_within_pinned_root_fd_even_after_root_path_is_swapped_to_symlink() {
        let real_root = fresh_temp_dir("pin-cleanup-real-root");
        let moved_aside = fresh_temp_dir("pin-cleanup-moved-aside-placeholder");
        fs::remove_dir(&moved_aside).expect("must clear rename destination placeholder");
        let attacker_target = fresh_temp_dir("pin-cleanup-attacker-target");
        // 攻撃者が用意した無関係なディレクトリに「監視用」の目印ファイルを
        // 置いておく。後始末が symlink 先を誤って再帰削除すれば、この目印
        // ごと消え去るはずである。
        fs::write(attacker_target.join("sentinel"), b"do-not-delete")
            .expect("must plant a sentinel file in the attacker-controlled directory");
        let key = sample_key();
        let entry_name = key.cache_entry_dir_name().expect("must compute entry name");

        // 破損エントリ（`kernel.ptx` 欠如）を pin 前に実ディレクトリへ
        // 用意し、「破損エントリ置換」分岐（受け入れ基準 2）を強制する。
        let entry_dir = real_root.join(&entry_name);
        fs::create_dir(&entry_dir).expect("must create corrupt entry directory");
        fs::write(entry_dir.join(CACHE_ENTRY_SOURCE_FILE), b"stale-cu")
            .expect("must write stale kernel.cu");
        assert!(
            !validate_cache_entry(&entry_dir),
            "entry must be corrupt (missing kernel.ptx) before replacement"
        );

        let root_fd = open_dir_nofollow(&real_root).expect("must pin real cache root");

        // pin 後に実ディレクトリを別名へどかし、元のパスへ無関係な
        // ディレクトリへの symlink を差し替える。
        fs::rename(&real_root, &moved_aside).expect("must move the real root directory aside");
        std::os::unix::fs::symlink(&attacker_target, &real_root)
            .expect("must plant a symlink at the original root path");

        let final_dir = cache_entry_path_in(&real_root, &key).expect("must compute final dir");

        store_cache_entry_at(&root_fd, &final_dir, &entry_name, "fresh-cu", "fresh-ptx")
            .expect("store must replace the corrupt entry via the pinned fd");

        // 置換後のエントリは pin 済み fd（`moved_aside`）側で有効である
        // こと。
        assert!(
            validate_cache_entry_at(&root_fd, &entry_name),
            "replaced entry must land in the directory that was pinned before the symlink swap"
        );

        // 後始末（一時ディレクトリ・退避ディレクトリの削除）が pin 済み
        // 実体側で完了し、`.tmp.` プレフィックスの残骸が残っていないこと。
        let leftover_tmp_dirs = fs::read_dir(&moved_aside)
            .expect("moved_aside must be readable")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp."))
            .count();
        assert_eq!(
            leftover_tmp_dirs, 0,
            "temp/stale cleanup must happen via the pinned root fd, not leave debris behind"
        );

        // symlink 先（攻撃者が用意した無関係なディレクトリ）は後始末に
        // よって一切変更されていないこと（削除の containment 突破が
        // ないことの直接検証。旧実装は `root.join(name)` を再解決した
        // ため、削除処理が symlink 先を辿り攻撃者のディレクトリ・
        // ファイルを消しうる P0 指摘があった）。
        assert!(
            attacker_target.join("sentinel").is_file(),
            "cleanup must never delete anything inside the symlink target planted after pinning"
        );
        let attacker_target_entries: Vec<_> = fs::read_dir(&attacker_target)
            .expect("attacker_target must still be readable")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert_eq!(
            attacker_target_entries.len(),
            1,
            "cleanup must not create or delete anything inside the symlink target"
        );

        drop(root_fd);
        let _ = fs::remove_dir_all(&moved_aside);
        let _ = fs::remove_file(&real_root);
        let _ = fs::remove_dir_all(&attacker_target);
    }

    // 受け入れ基準 1: 並行競合（複数スレッド）。同一注入ルート・同一キー
    // へ複数スレッドが同時 store しても全スレッド `Ok` を返し、最終
    // エントリが不変条件（`validate_cache_entry`）を満たすこと。
    #[test]
    fn concurrent_store_from_multiple_threads_all_succeed() {
        use std::sync::Arc;
        use std::thread;

        let root = Arc::new(fresh_temp_dir("concurrent"));
        let key = Arc::new(sample_key());

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let root = Arc::clone(&root);
                let key = Arc::clone(&key);
                thread::spawn(move || {
                    store_cache_entry_in(
                        &root,
                        &key,
                        &format!("thread-{i}.cu"),
                        &format!("thread-{i}.ptx"),
                    )
                })
            })
            .collect();

        for handle in handles {
            handle
                .join()
                .expect("thread must not panic")
                .expect("every concurrent store must succeed (Ok)");
        }

        let entry_path = cache_entry_path_in(&root, &key).expect("must succeed");
        assert!(validate_cache_entry(&entry_path));

        let leftover_tmp_dirs = fs::read_dir(root.as_path())
            .expect("root must be readable")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp."))
            .count();
        assert_eq!(
            leftover_tmp_dirs, 0,
            "no thread's temp directory may remain after the race"
        );

        let _ = fs::remove_dir_all(root.as_path());
    }

    // 受け入れ基準 2: 破損検出。`kernel.ptx` を削除した破損エントリで
    // load がミス（`Ok(None)`）を返すこと。
    #[test]
    fn load_treats_missing_ptx_file_as_miss() {
        let root = fresh_temp_dir("corrupt-load");
        let key = sample_key();

        let entry_dir =
            store_cache_entry_in(&root, &key, "cu-body", "ptx-body").expect("store must succeed");
        fs::remove_file(entry_dir.join(CACHE_ENTRY_PTX_FILE)).expect("must remove ptx file");

        let loaded = load_cache_entry_in(&root, &key, "cu-body").expect("load must succeed");
        assert!(loaded.is_none(), "entry missing kernel.ptx must be a miss");

        let _ = fs::remove_dir_all(&root);
    }

    // 受け入れ基準 2: 破損エントリ存在下で store が置換に成功すること。
    #[test]
    fn store_replaces_corrupt_existing_entry() {
        let root = fresh_temp_dir("corrupt-replace");
        let key = sample_key();

        let entry_dir = store_cache_entry_in(&root, &key, "stale-cu", "stale-ptx")
            .expect("initial store must succeed");
        fs::remove_file(entry_dir.join(CACHE_ENTRY_PTX_FILE)).expect("must remove ptx file");
        assert!(
            !validate_cache_entry(&entry_dir),
            "entry must be corrupt (missing kernel.ptx) before replacement"
        );

        let replaced = store_cache_entry_in(&root, &key, "fresh-cu", "fresh-ptx")
            .expect("store must replace the corrupt entry");
        assert_eq!(replaced, entry_dir);

        let loaded = load_cache_entry_in(&root, &key, "fresh-cu")
            .expect("load must succeed")
            .expect("replaced entry must be a hit");
        assert_eq!(loaded.kernel_cu, "fresh-cu");
        assert_eq!(loaded.kernel_ptx, "fresh-ptx");

        let _ = fs::remove_dir_all(&root);
    }

    // 受け入れ基準（実装計画 §3.1・§8）: 非空検査。`kernel.cu` をクラッシュ
    // 残骸想定の 0 バイトファイルへ差し替えると、`is_file()` は真だが
    // `read_verified_cache_entry_file` の非空チェックでミス扱いになること。
    #[test]
    fn load_treats_empty_source_file_as_miss() {
        let root = fresh_temp_dir("empty-cu");
        let key = sample_key();

        let entry_dir =
            store_cache_entry_in(&root, &key, "cu-body", "ptx-body").expect("store must succeed");
        fs::write(entry_dir.join(CACHE_ENTRY_SOURCE_FILE), b"")
            .expect("must truncate kernel.cu to zero bytes");

        let loaded =
            load_cache_entry_in(&root, &key, "cu-body").expect("load must not error, only miss");
        assert!(
            loaded.is_none(),
            "entry with an empty kernel.cu must be a miss, not a crash-remnant hit"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // 受け入れ基準（実装計画 §3.1・§8）: PR #677 Bugbot 指摘〈Empty
    // entries block replacement〉の回帰テスト。`validate_cache_entry`
    // （`is_plain_file` 経由）が 0 バイト残骸を「有効」と誤判定すると、
    // `store_cache_entry_in` の rename 衝突時の破損判定
    // （`!validate_cache_entry(&final_dir)`）を通過してしまい、新規
    // 書き込みが破棄されて空回りキャッシュミスが恒久化する（`is_plain_file`
    // の非空検査追加前は本テストが失敗していたはず）。0 バイト残骸を
    // `store_cache_entry_in` が「破損」として検出・置換することを検証する。
    #[test]
    fn store_replaces_zero_byte_remnant_entry() {
        let root = fresh_temp_dir("zero-byte-remnant");
        let key = sample_key();

        let entry_dir = store_cache_entry_in(&root, &key, "stale-cu", "stale-ptx")
            .expect("initial store must succeed");
        // クラッシュ直後の 0 バイト残骸を模す（`create` 直後・書き込み前に
        // プロセスが落ちた場合等）。ファイル自体は存在し `is_file()` は
        // 真だが、内容は空。
        fs::write(entry_dir.join(CACHE_ENTRY_SOURCE_FILE), b"")
            .expect("must truncate kernel.cu to zero bytes");
        assert!(
            !validate_cache_entry(&entry_dir),
            "zero-byte kernel.cu must make the entry invalid (corrupt), not merely a load-time miss"
        );

        let replaced = store_cache_entry_in(&root, &key, "fresh-cu", "fresh-ptx").expect(
            "store must replace the zero-byte remnant instead of absorbing it as first-writer-wins",
        );
        assert_eq!(replaced, entry_dir);

        let loaded = load_cache_entry_in(&root, &key, "fresh-cu")
            .expect("load must succeed")
            .expect("replaced entry must be a hit");
        assert_eq!(loaded.kernel_cu, "fresh-cu");
        assert_eq!(loaded.kernel_ptx, "fresh-ptx");

        let _ = fs::remove_dir_all(&root);
    }

    // 受け入れ基準 2（イシュー #509 PR #677 Bugbot 指摘〈Non-dir blocks
    // cache replacement〉の回帰テスト）: 最終エントリ名がディレクトリで
    // はなくプレーンファイルで占有されている場合でも、`entry_exists_at`
    // の型非依存化により破損エントリ置換分岐に到達し、store が
    // `CudaError::CacheIo`（恒久失敗）に陥らず正常に置換できること。
    #[cfg(unix)]
    #[test]
    fn store_replaces_non_directory_occupant_at_final_entry_path() {
        let root = fresh_temp_dir("non-dir-occupant");
        let key = sample_key();

        // まだキャッシュエントリを一度も書き込んでいない状態で、最終
        // エントリ名の位置にプレーンファイル（ディレクトリではない）を
        // 直接置く（外部プロセスによる破壊・想定外の残骸を模す）。
        let entry_dir = cache_entry_path_in(&root, &key).expect("must build entry path");
        fs::write(&entry_dir, b"not a directory").expect("must create plain-file occupant");

        let replaced = store_cache_entry_in(&root, &key, "fresh-cu", "fresh-ptx")
            .expect("store must replace a non-directory occupant instead of failing permanently");
        assert_eq!(replaced, entry_dir);

        let loaded = load_cache_entry_in(&root, &key, "fresh-cu")
            .expect("load must succeed")
            .expect("replaced entry must be a hit");
        assert_eq!(loaded.kernel_cu, "fresh-cu");
        assert_eq!(loaded.kernel_ptx, "fresh-ptx");

        let _ = fs::remove_dir_all(&root);
    }

    // イシュー #509 PR #677 Bugbot 指摘〈Invalid UTF-8 treated as hard
    // error〉の回帰テスト: 他の破損種別（欠落・空・過大・特殊ファイル・
    // ソース不一致）と同じく、非 UTF-8 の `kernel.cu` はハードエラー
    // （`CudaError::CacheIo`）ではなくミス（`Ok(None)`）として扱われる
    // こと。
    #[test]
    fn load_treats_invalid_utf8_source_as_miss() {
        let root = fresh_temp_dir("invalid-utf8");
        let key = sample_key();

        let entry_dir =
            store_cache_entry_in(&root, &key, "cu-body", "ptx-body").expect("store must succeed");
        // 有効な UTF-8 にならないバイト列（単独の継続バイト）で
        // `kernel.cu` を上書きする。
        fs::write(
            entry_dir.join(CACHE_ENTRY_SOURCE_FILE),
            [0xFFu8, 0xFE, 0xFD],
        )
        .expect("must overwrite kernel.cu with invalid UTF-8 bytes");

        let loaded = load_cache_entry_in(&root, &key, "cu-body");
        assert!(
            matches!(loaded, Ok(None)),
            "invalid UTF-8 kernel.cu must be a miss (Ok(None)), not a hard error: {loaded:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // イシュー #509 PR #677 Bugbot 指摘〈Load skips cache root pin〉の
    // 回帰テスト: `ensure_cache_root` 相当の検証後にキャッシュルート
    // 自体が symlink へ差し替えられても（旧実装は `root.join(entry_name)`
    // というパス文字列を組み立てて開いていたため、`root` は非最終
    // コンポーネントとして symlink を追跡されてしまっていた）、`load`
    // が攻撃者制御ディレクトリ配下のエントリをヒット扱いしないこと。
    // `root_fd` を pin してから fd 相対操作のみで解決する現行実装では、
    // `root` 自体が symlink であれば `open_dir_nofollow` 自体が
    // `O_NOFOLLOW` で拒否するためミス（`Ok(None)`）になる。
    #[cfg(unix)]
    #[test]
    fn load_rejects_cache_root_replaced_with_symlink() {
        let base = fresh_temp_dir("root-symlink-toctou");
        let legit_root = base.join("legit-root");
        let attacker_root = base.join("attacker-root");
        fs::create_dir_all(&legit_root).expect("must create legit root");
        fs::create_dir_all(&attacker_root).expect("must create attacker root");
        let key = sample_key();

        // 正規ルートに正規のエントリを保存する（`ensure_cache_root` 相当
        // の検証を経た状態を模す）。
        store_cache_entry_in(&legit_root, &key, "legit-cu", "legit-ptx")
            .expect("initial legit store must succeed");

        // 攻撃者制御ディレクトリに、同じキーで `expected_src` に一致する
        // 細工エントリを用意する（TOCTOU が成立すれば `load` がこちらを
        // ヒットとして返してしまう）。
        store_cache_entry_in(&attacker_root, &key, "legit-cu", "malicious-ptx")
            .expect("attacker store must succeed");

        // 検証済みの正規ルートを、攻撃者ディレクトリへの symlink へ
        // 差し替える（`ensure_cache_root` 後の TOCTOU を模す）。
        fs::remove_dir_all(&legit_root).expect("must remove legit root before swapping");
        std::os::unix::fs::symlink(&attacker_root, &legit_root)
            .expect("must replace legit root with a symlink to the attacker root");

        let loaded = load_cache_entry_in(&legit_root, &key, "legit-cu");
        assert!(
            matches!(loaded, Ok(None)),
            "a cache root replaced with a symlink must be treated as a miss, \
             never read through to the attacker-controlled directory: {loaded:?}"
        );

        let _ = fs::remove_dir_all(&attacker_root);
        let _ = fs::remove_file(&legit_root);
        let _ = fs::remove_dir_all(&base);
    }

    // 受け入れ基準（実装計画 §3.1・§7）: ハッシュ衝突安全弁。保存済み
    // `kernel.cu` の内容が呼び出し元の要求ソース（`expected_src`）と
    // バイト不一致であれば、64bit FNV-1a ハッシュの衝突によって別ソースの
    // エントリを誤ってヒット扱いしない（誤った PTX を GPU へ渡さない
    // fail-closed）。
    #[test]
    fn load_treats_source_mismatch_as_miss() {
        let root = fresh_temp_dir("source-mismatch");
        let key = sample_key();

        store_cache_entry_in(&root, &key, "stored-source.cu", "stored-source.ptx")
            .expect("store must succeed");

        let loaded = load_cache_entry_in(&root, &key, "different-source.cu")
            .expect("load must not error, only miss");
        assert!(
            loaded.is_none(),
            "a stored source that differs from the caller's expected source must be a miss \
             (hash-collision safety valve), even though the entry itself is otherwise valid"
        );

        // 対照実験: 一致する場合は通常どおりヒットすること。
        let hit = load_cache_entry_in(&root, &key, "stored-source.cu")
            .expect("load must succeed")
            .expect("matching source must still be a hit");
        assert_eq!(hit.kernel_cu, "stored-source.cu");

        let _ = fs::remove_dir_all(&root);
    }

    // `ensure_cache_root_in`: 通常ケース（symlink なし）で `candidate_root`
    // が実体化され、containment 検証（`workspace_root` 配下でない）を
    // 通過した canonical パスが返ること。
    #[test]
    fn ensure_cache_root_in_creates_and_returns_canonical_root() {
        let workspace_root = fresh_temp_dir("ensure-root-workspace");
        let candidate = fresh_temp_dir("ensure-root-candidate").join("nested/cache/dir");

        let resolved = ensure_cache_root_in(&candidate, &workspace_root)
            .expect("candidate outside workspace_root must be accepted");

        assert!(resolved.is_dir());
        assert_eq!(
            resolved,
            candidate
                .canonicalize()
                .expect("candidate must exist and be canonicalizable after ensure_cache_root_in")
        );

        let _ = fs::remove_dir_all(&workspace_root);
        // `candidate` は `candidate.parent()`（`fresh_temp_dir` が返した
        // ディレクトリ）を消せば `nested/...` ごと片付く。
        let _ = fs::remove_dir_all(
            candidate
                .ancestors()
                .nth(2)
                .expect("candidate must have a fresh_temp_dir ancestor"),
        );
    }

    // symlink 再検証（cfg(unix)）: `candidate_root` が symlink 経由で
    // `workspace_root` 配下を指す場合、字句上は `workspace_root` 外に
    // 見えても `ensure_cache_root_in` が `canonicalize` 済みパスで
    // containment を再検証し拒否すること（実装計画 §3.4・受け入れ
    // 基準相当）。
    #[cfg(unix)]
    #[test]
    fn ensure_cache_root_in_rejects_symlink_into_workspace_root() {
        let workspace_root = fresh_temp_dir("symlink-workspace");
        let real_target = workspace_root.join("evil-cache-inside-workspace");
        fs::create_dir_all(&real_target).expect("must create symlink target inside workspace");

        // symlink 自体は `workspace_root` の外に置く（字句比較だけでは
        // containment 違反に見えないことを保証するため）。
        let outside_dir = fresh_temp_dir("symlink-outside");
        let candidate = outside_dir.join("cache-symlink");
        std::os::unix::fs::symlink(&real_target, &candidate)
            .expect("must create symlink pointing into workspace_root");

        // 字句上は `workspace_root` の配下ではないことを前提として確認
        // する（symlink 解決なしでは検出できないケースであることの検算）。
        assert!(!path_lexically_within(&candidate, &workspace_root));

        let result = ensure_cache_root_in(&candidate, &workspace_root);
        assert!(
            matches!(result, Err(CudaError::CacheDirUnavailable { .. })),
            "symlink resolving into workspace_root must be rejected after canonicalization"
        );

        let _ = fs::remove_dir_all(&workspace_root);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // symlink 再検証（cfg(unix)）・実体化前チェック: `candidate_root`
    // 自身ではなく**祖先**が symlink 経由で `workspace_root` 配下を指し、
    // かつ `candidate_root` の末端（`nested/cache`）がまだ存在しない場合
    // （イシュー #509 codex-review P0 指摘の具体的な再現ケース）でも、
    // `fs::create_dir_all` を一切呼ばずに拒否すること（symlink 先＝
    // workspace 内へ実際にディレクトリが作成されてしまう副作用がないこと
    // を確認する）。
    #[cfg(unix)]
    #[test]
    fn ensure_cache_root_in_rejects_ancestor_symlink_without_creating_anything_inside_workspace() {
        let workspace_root = fresh_temp_dir("ancestor-symlink-workspace");
        let real_target = workspace_root.join("evil-cache-inside-workspace");
        fs::create_dir_all(&real_target).expect("must create symlink target inside workspace");

        // symlink 自体は `workspace_root` の外に置く。
        let outside_dir = fresh_temp_dir("ancestor-symlink-outside");
        let ancestor_symlink = outside_dir.join("cache-symlink");
        std::os::unix::fs::symlink(&real_target, &ancestor_symlink)
            .expect("must create ancestor symlink pointing into workspace_root");

        // `candidate_root` はこの symlink のさらに下（まだ存在しない）。
        let candidate = ancestor_symlink.join("nested/cache/dir");
        assert!(
            !candidate.exists(),
            "candidate must not exist yet to reproduce the pre-existing-ancestor scenario"
        );

        let result = ensure_cache_root_in(&candidate, &workspace_root);
        assert!(
            matches!(result, Err(CudaError::CacheDirUnavailable { .. })),
            "ancestor symlink resolving into workspace_root must be rejected before any \
             directory is created under it"
        );

        // 副作用がないこと: symlink 先（workspace 内）に `nested` 等が
        // 実際に作られていないことを確認する。
        assert!(
            !real_target.join("nested").exists(),
            "no directory must have been created inside workspace_root as a side effect of \
             the rejected containment check"
        );

        let _ = fs::remove_dir_all(&workspace_root);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // symlink 再検証（cfg(unix)）・正当な外部 symlink 祖先の許容: 祖先
    // （`~/.cache`／`XDG_CACHE_HOME` 相当）自体が symlink であっても、
    // その解決先が `workspace_root` 外の正当なディレクトリであれば
    // 拒否せずディレクトリを実体化できること（イシュー #509 codex-review
    // P1 再指摘対応の回帰テスト。修正前は Linux 版が
    // `open_dir_nofollow`（`O_NOFOLLOW`）を symlink である祖先そのものへ
    // 適用していたため `CacheIo` で失敗していた）。
    #[cfg(unix)]
    #[test]
    fn ensure_cache_root_in_accepts_legitimate_ancestor_symlink_outside_workspace() {
        let workspace_root = fresh_temp_dir("legit-ancestor-symlink-workspace");

        // symlink とその解決先はいずれも `workspace_root` 外に置く
        // （`~/.cache` が別ディスク上の実ディレクトリを指す一般的な
        // 構成を模す）。
        let real_target = fresh_temp_dir("legit-ancestor-symlink-real-target");
        let outside_dir = fresh_temp_dir("legit-ancestor-symlink-outside");
        let ancestor_symlink = outside_dir.join("cache-symlink");
        std::os::unix::fs::symlink(&real_target, &ancestor_symlink)
            .expect("must create ancestor symlink pointing outside workspace_root");

        // `candidate_root` はこの symlink のさらに下（まだ存在しない）。
        let candidate = ancestor_symlink.join("nested/cache/dir");
        assert!(
            !candidate.exists(),
            "candidate must not exist yet to reproduce the pre-existing-ancestor scenario"
        );

        let resolved = ensure_cache_root_in(&candidate, &workspace_root).expect(
            "a legitimate ancestor symlink resolving outside workspace_root must be accepted",
        );

        assert!(resolved.is_dir());
        assert_eq!(
            resolved,
            real_target
                .join("nested/cache/dir")
                .canonicalize()
                .expect("resolved cache root must exist under the symlink's real target")
        );

        let _ = fs::remove_dir_all(&workspace_root);
        let _ = fs::remove_dir_all(&real_target);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // 実機（DGX Spark GB10 等）依存: `libnvrtc` の実バージョンを問い合わせる。
    // CUDA toolkit 非搭載環境（通常 CI）では `NvrtcUnavailable` を返す想定
    // であり panic しないため `#[ignore]` で分離する（`make
    // test-ignored-cuda` 導線。`.claude/rules/coding-rust.md` 実機分離規約）。
    #[test]
    #[ignore]
    fn nvrtc_version_returns_ok_on_real_device() {
        let version = nvrtc_version().expect("NVRTC must be present on the real-device runner");
        assert!(version.0 > 0, "NVRTC major version must be positive");
    }

    // `derive_pipeline_stages`（C-8・#521）のユニットテスト群。GPU 不要
    // （純関数）のため通常 CI で実行される（`#[ignore]` なし）。

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("test helper: value must be non-zero")
    }

    // 現行タイル構成（#494。`kernels_mma.rs::MMA_BM`=64/`MMA_BN`=128/
    // `MMA_BK`=32・f16=2 バイト）で、静的 SMEM 上限 49,152 バイト
    // （48KiB）を予算として渡すと 4 段が導出されること（実装計画 §3.2
    // の検算: 49152 / 12288 = 4、タイル面積 64*128=8192 < 128*192=24576
    // のため min_required=4 も満たす）。
    #[test]
    fn derive_pipeline_stages_matches_current_tile_configuration() {
        let stages = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), 49_152)
            .expect("current tile configuration must derive a valid stage count");
        assert_eq!(stages.get(), 4);
    }

    // 予算が 1 段分にも満たない場合は `min_required` 未達で `Err`。
    #[test]
    fn derive_pipeline_stages_rejects_insufficient_budget() {
        let result = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), 1_024);
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // 予算が極端に大きい場合でも `MAX_PIPELINE_STAGES`（16）でクランプ
    // されること。
    #[test]
    fn derive_pipeline_stages_clamps_at_max_pipeline_stages() {
        let stages = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), u64::MAX)
            .expect("huge budget must still derive a valid (clamped) stage count");
        assert_eq!(stages.get(), MAX_PIPELINE_STAGES);
    }

    // 小タイル（block_m * block_n < 128*192）は最小 4 段要求。3 段相当の
    // 予算では `Err` になり、4 段分の予算でちょうど成立することを確認する。
    #[test]
    fn derive_pipeline_stages_small_tile_requires_four_stages() {
        // block_m=32, block_n=64 → 面積 2048 < 24576（小タイル）。
        // 1 段あたり (32*32 + 32*64) * 2 = 6144 バイト。3 段分の予算
        // （18432）では 3 段しか導出できず min_required=4 未達で Err。
        let insufficient = derive_pipeline_stages(nz(32), nz(64), nz(32), nz(2), 18_432);
        assert!(matches!(
            insufficient,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));

        // 4 段分ちょうど（24576）なら成立する。
        let stages = derive_pipeline_stages(nz(32), nz(64), nz(32), nz(2), 24_576)
            .expect("4-stage budget must satisfy the small-tile minimum requirement");
        assert_eq!(stages.get(), 4);
    }

    // 大タイル（block_m * block_n >= 128*192）は最小 3 段要求。
    #[test]
    fn derive_pipeline_stages_large_tile_requires_three_stages() {
        // block_m=128, block_n=192 → 面積 24576（境界値。`< 128*192` は
        // false なので大タイル扱い）。1 段あたり (128*32 + 32*192) * 2 =
        // 20480 バイト。3 段分ちょうど（61440）で成立する。
        let stages = derive_pipeline_stages(nz(128), nz(192), nz(32), nz(2), 61_440)
            .expect("3-stage budget must satisfy the large-tile minimum requirement");
        assert_eq!(stages.get(), 3);
    }

    // オーバーフロー安全性: 巨大な block 値を渡しても panic せず `Err`
    // を返す（`u64` 中間値でも `checked_mul` が `None` を返すケース）。
    #[test]
    fn derive_pipeline_stages_rejects_overflowing_block_dimensions() {
        let result =
            derive_pipeline_stages(nz(u32::MAX), nz(u32::MAX), nz(u32::MAX), nz(4), 49_152);
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }
}
