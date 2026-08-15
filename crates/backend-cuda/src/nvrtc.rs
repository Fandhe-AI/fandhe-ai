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
//! カーネルソース（`src` 引数）はコンパイル時定数（`&'static str`）
//! であることを呼び出し元（#33/#34）に要求し、外部入力をソース文字列へ
//! 連結しない方針とする。

use std::ffi::OsStr;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::error::CudaError;

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaKernelDescriptor {
    /// カーネル名。C-2（#506）のキャッシュディレクトリ命名
    /// （`kernel.<name>.<hash>`）で使われる想定であり、`&'static str`
    /// （コンパイル時定数）に限定して外部入力の混入を型で遮断する
    /// （A03 対策。`compile_ptx` の `src` 引数契約〈本ファイル冒頭〉と
    /// 同型の判断）。
    kernel_name: &'static str,
    /// GEMM 形状（M/N/K）。`tensor_core::dispatch` の既存 `Hash + Eq` 型を
    /// 再利用し、ディスパッチ規則（#67/#68）が扱う形状表現と揃える。
    shape: tensor_core::dispatch::GemmShape,
    /// ブロックタイル M 次元。
    block_m: NonZeroU32,
    /// ブロックタイル N 次元。
    block_n: NonZeroU32,
    /// ブロックタイル K 次元。
    block_k: NonZeroU32,
    /// パイプライン段数。SMEM 予算からの逆算ロジック（`stages` の
    /// 決定方法）は C-8（#521）のスコープであり、本型はデータとして
    /// 保持するのみ。
    stages: NonZeroU32,
    /// 入出力 dtype。`tensor_core::dispatch::DType` を再利用する。
    dtype: tensor_core::dispatch::DType,
}

impl CudaKernelDescriptor {
    /// `CudaKernelDescriptor` を構築する。
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
    pub fn new(
        kernel_name: &'static str,
        shape: tensor_core::dispatch::GemmShape,
        block_m: u32,
        block_n: u32,
        block_k: u32,
        stages: u32,
        dtype: tensor_core::dispatch::DType,
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
        let non_zero = |value: u32, field: &str| {
            NonZeroU32::new(value).ok_or_else(|| CudaError::InvalidKernelDescriptor {
                detail: format!("{field} must be non-zero (got 0)"),
            })
        };
        Ok(Self {
            kernel_name,
            shape,
            block_m: non_zero(block_m, "block_m")?,
            block_n: non_zero(block_n, "block_n")?,
            block_k: non_zero(block_k, "block_k")?,
            stages: non_zero(stages, "stages")?,
            dtype,
        })
    }

    /// カーネル名。
    pub fn kernel_name(&self) -> &'static str {
        self.kernel_name
    }

    /// GEMM 形状。
    pub fn shape(&self) -> tensor_core::dispatch::GemmShape {
        self.shape
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
    /// 先頭 1 バイトはエンコーディングバージョン。C-5（#514）でソース
    /// 断片ハッシュをキーへ拡張する等、将来このバイト列表現を変更する際は
    /// このバージョン番号を上げ、互換性の切り替え点として使う。
    /// **このバイト列表現の変更は既存キャッシュエントリを全て無効化する**
    /// （ハッシュ値が変わるため）契約であることに注意。
    fn canonical_bytes(&self) -> Vec<u8> {
        const ENCODING_VERSION: u8 = 1;

        let mut buf = Vec::new();
        buf.push(ENCODING_VERSION);

        push_len_prefixed_str(&mut buf, self.descriptor.kernel_name);

        buf.extend_from_slice(&self.descriptor.shape.m.to_le_bytes());
        buf.extend_from_slice(&self.descriptor.shape.n.to_le_bytes());
        buf.extend_from_slice(&self.descriptor.shape.k.to_le_bytes());

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

        buf
    }

    /// [`Self::canonical_bytes`] を FNV-1a 64bit（[`fnv1a_64`]）でハッシュ
    /// した値。[`Self::cache_entry_dir_name`] のハッシュ部として使う
    /// ディスク永続キー本体（イシュー #506）。
    pub fn stable_hash(&self) -> u64 {
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
    pub fn cache_entry_dir_name(&self) -> Result<String, CudaError> {
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
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
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
/// **安全側の検証**: `override_dir` が空文字列・相対パスの場合は
/// `CudaError::CacheDirUnavailable` で拒否する。相対パスを許すと
/// カレントディレクトリ（呼び出しコンテキストによってはリポジトリ
/// ツリー内）配下にキャッシュが作られ、「キャッシュルートはリポジトリ
/// ツリー外」要件（runner workspace に成果物を残さない方針。
/// `.claude/rules/security.md`）と矛盾するため fail-closed で拒否する。
fn resolve_cache_root(
    override_dir: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Result<PathBuf, CudaError> {
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
        return Ok(path.to_path_buf());
    }

    if let Some(xdg) = xdg_cache_home
        && !xdg.is_empty()
    {
        return Ok(Path::new(xdg).join("rust-ai-library").join("cuda"));
    }

    if let Some(home_dir) = home
        && !home_dir.is_empty()
    {
        return Ok(Path::new(home_dir)
            .join(".cache")
            .join("rust-ai-library")
            .join("cuda"));
    }

    Err(CudaError::CacheDirUnavailable {
        detail: "none of RUST_AI_CUDA_CACHE_DIR, XDG_CACHE_HOME, HOME is set; \
                 cannot determine a cache root outside the repository tree"
            .to_string(),
    })
}

/// [`resolve_cache_root`] の公開ラッパー。実プロセス環境変数
/// （`RUST_AI_CUDA_CACHE_DIR`・`XDG_CACHE_HOME`・`HOME`）を読んで委譲する
/// （イシュー #506・Phase C-2。C-3（#509）・C-4（#511）から呼ばれる想定）。
pub fn cache_root() -> Result<PathBuf, CudaError> {
    resolve_cache_root(
        std::env::var_os("RUST_AI_CUDA_CACHE_DIR").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// `root` と [`CudaKernelCacheKey::cache_entry_dir_name`] を合成し、
/// キャッシュエントリの完全パスを返す内部純関数（イシュー #506・
/// Phase C-2）。fs I/O は行わない純粋なパス組み立てのみ（`create_dir_all`
/// 等は C-3（#509）のスコープ）。
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
/// 検証、第 2 層: `cache_entry_dir_name` 内の縦深防御検査）。
pub fn cache_entry_path(key: &CudaKernelCacheKey) -> Result<PathBuf, CudaError> {
    cache_entry_path_in(&cache_root()?, key)
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
        CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            2,
            DType::F32,
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
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            0,
            64,
            32,
            2,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_stages() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            0,
            DType::F32,
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
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            0,
            32,
            2,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_block_k() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            0,
            2,
            DType::F32,
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
        // （`CudaKernelDescriptor::new` 内コメント参照）。
        for bad_name in ["../escape", "a/b", "a\\b", "..", "", ".foo", "foo.", "."] {
            let result = CudaKernelDescriptor::new(
                bad_name,
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
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
        );
        assert_ne!(base, different_shape);

        let different_block_m = CudaKernelCacheKey::new(
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
        );
        assert_ne!(base, different_block_m);

        // codex-review 指摘（PR #641・P2）: `block_n`・`block_k`・
        // `stages`・`kernel_name` は `different_block_m` までのケースでは
        // 未検証だった。各フィールドを単独で変更し、`Hash`/`Eq` の
        // derive がフィールド漏れなく全メンバを含んでいることを検証する。
        let different_block_n = CudaKernelCacheKey::new(
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
        );
        assert_ne!(base, different_block_n);

        let different_block_k = CudaKernelCacheKey::new(
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
        );
        assert_ne!(base, different_block_k);

        let different_stages = CudaKernelCacheKey::new(
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
        );
        assert_ne!(base, different_stages);

        let different_kernel_name = CudaKernelCacheKey::new(
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
        );
        assert_ne!(base, different_kernel_name);

        // `kernel_name` は固定したまま `dtype` のみ変更する。
        // `kernel_name` を同時に変えると、万一 `dtype` が `Hash`/`Eq`
        // から欠落しても `kernel_name` 側の差分だけで
        // `assert_ne!` が通ってしまい検出できない
        // （codex-review 指摘・PR #641・P2）。
        let different_dtype = CudaKernelCacheKey::new(
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
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (9, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (8, 0),
                (13, 0),
                vec!["--gpu-architecture=compute_80".to_string()],
            ),
            CudaKernelCacheKey::new(
                sample_descriptor(),
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_90".to_string()],
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
        );
        let key_a_bc = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["a".to_string(), "bc".to_string()],
        );
        assert_ne!(key_ab_c.canonical_bytes(), key_a_bc.canonical_bytes());
        assert_ne!(key_ab_c.stable_hash(), key_a_bc.stable_hash());
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

    // キャッシュルート解決: env 上書き優先。
    #[test]
    fn resolve_cache_root_prefers_override() {
        let root = resolve_cache_root(
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
        let root =
            resolve_cache_root(None, None, Some(OsStr::new("/home/user"))).expect("must succeed");
        assert_eq!(
            root,
            PathBuf::from("/home/user/.cache/rust-ai-library/cuda")
        );
    }

    // キャッシュルート解決: 全欠落時は `CacheDirUnavailable`（panic なし）。
    #[test]
    fn resolve_cache_root_errs_when_all_missing() {
        let result = resolve_cache_root(None, None, None);
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
    }

    // 安全側の検証: 空文字列の override は拒否する。
    #[test]
    fn resolve_cache_root_rejects_empty_override() {
        let result = resolve_cache_root(
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
            Some(OsStr::new("relative/cache/dir")),
            Some(OsStr::new("/home/user/.cache")),
            Some(OsStr::new("/home/user")),
        );
        assert!(matches!(result, Err(CudaError::CacheDirUnavailable { .. })));
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
}
