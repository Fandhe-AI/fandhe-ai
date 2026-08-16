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

use std::hash::Hash;
use std::num::NonZeroU32;

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
        if kernel_name.is_empty()
            || kernel_name.contains('/')
            || kernel_name.contains('\\')
            || kernel_name.contains("..")
        {
            return Err(CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "kernel_name must be a non-empty path-safe segment \
                     (no '/', '\\\\', or \"..\"), got {kernel_name:?}"
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
/// ソースコード断片のハッシュ（DeepGEMM の `code` 相当）は本キーには
/// 含めない（C-5・#514 のスコープ。ソース断片によるキー拡張・キャッシュ
/// 無効化はそちらで扱う）。ディレクトリ命名（`kernel.<name>.<hash>`）と
/// ハッシュ関数自体は C-2（#506）のスコープであり、本型は「ハッシュ化
/// される前のキーの単位」を定義するのみ。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
}

impl CudaKernelCacheKey {
    /// `descriptor`（検証済み）と環境パラメータからキーを構築する。
    /// `descriptor` は [`CudaKernelDescriptor::new`] を経由済みのため
    /// infallible（`Result` を返さない）。
    pub fn new(
        descriptor: CudaKernelDescriptor,
        compute_capability: (i32, i32),
        nvrtc_version: (i32, i32),
        compile_flags: Vec<String>,
    ) -> Self {
        Self {
            descriptor,
            compute_capability,
            nvrtc_version,
            compile_flags,
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
    ) -> Result<Self, CudaError> {
        Ok(Self::new(
            descriptor,
            device.compute_capability(),
            nvrtc_version()?,
            compile_flags,
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

    fn sample_key() -> CudaKernelCacheKey {
        CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
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
        for bad_name in ["../escape", "a/b", "a\\b", "..", ""] {
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
        );
        assert_ne!(base, different_dtype);

        // compute capability 違い（DeepGEMM のアーキ違い＝別エントリの
        // 性質そのもの）。
        let different_cc = CudaKernelCacheKey::new(
            sample_descriptor(),
            (9, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_cc);

        let different_nvrtc_version = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (13, 0),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_nvrtc_version);

        let different_flags = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_90".to_string()],
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
        );
        assert_ne!(base, different_compiled_dims);
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
        );
        assert_eq!(cache.get(&miss_key), None);
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
