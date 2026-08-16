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

use std::ffi::OsStr;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// パイプライン段数。SMEM 予算からの逆算ロジックは
    /// [`derive_pipeline_stages`]（C-8・#521 で実装）が担い、本型は
    /// 導出済みの値をデータとして保持するのみ。
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
/// codex-review P0 指摘対応）: `entry_dir` 自体または
/// [`CACHE_ENTRY_SOURCE_FILE`]／[`CACHE_ENTRY_PTX_FILE`] を symlink に
/// 置換されると、`is_file()`（symlink を追跡する）はキャッシュルート外の
/// 任意ファイルでも「有効なキャッシュエントリ」と誤判定してしまう
/// （A03 symlink 脱出。後続で PTX として任意ファイルを読み込み GPU 上で
/// 実行する経路に繋がりうる）。本関数は [`load_cache_entry_in`] が実際に
/// 読み出す直前に呼ばれるため、ここで symlink を拒否すれば
/// `fs::read_to_string` が symlink 先の内容を読むことはない。
fn validate_cache_entry(entry_dir: &Path) -> bool {
    is_plain_dir(entry_dir)
        && is_plain_file(&entry_dir.join(CACHE_ENTRY_SOURCE_FILE))
        && is_plain_file(&entry_dir.join(CACHE_ENTRY_PTX_FILE))
}

/// `path` が symlink ではない通常ファイルであるかを、symlink を追跡
/// しない [`fs::symlink_metadata`] で判定する（[`validate_cache_entry`]
/// 用。イシュー #509 codex-review P0 指摘対応）。
fn is_plain_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_file())
        .unwrap_or(false)
}

/// `path` が symlink ではない通常ディレクトリであるかを、symlink を
/// 追跡しない [`fs::symlink_metadata`] で判定する（[`is_plain_file`] と
/// 同じ理由。エントリディレクトリ自体が symlink に置換されているケース
/// を拒否する）。
fn is_plain_dir(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|meta| meta.file_type().is_dir())
        .unwrap_or(false)
}

/// ファイルを fsync する（[`write_entry_files`] が書き込み直後に呼ぶ）。
///
/// rename 前に内容がディスクへ確実に反映されていることを保証する
/// （A08 整合性: rename のアトミック性だけでは「rename 前にクラッシュし
/// 部分書き込みのまま OS バッファに留まる」ケースを防げないため。
/// DeepGEMM compiler の書き込みプロトコルに倣う）。
fn fsync_file(path: &Path) -> std::io::Result<()> {
    fs::File::open(path)?.sync_all()
}

/// ディレクトリを fsync する（[`write_entry_files`]・[`store_cache_entry_in`]
/// が一時ディレクトリ／ルートディレクトリに対して呼ぶ）。
///
/// POSIX 準拠 OS ではディレクトリエントリの追加（ファイル作成・rename）も
/// fsync が必要な変更として扱われるため、`kernel.cu`／`kernel.ptx` の
/// 書き込み後に一時ディレクトリ自体も fsync する（DeepGEMM `fsync_dir`
/// 相当のボトムアップ fsync）。非 Unix（本クレートのビルド対象は
/// `backend-switching-design.md` の cfg ベース分岐上 Linux/macOS/CUDA
/// toolkit 非搭載環境のみを想定し Windows は対象外）では
/// `File::open` でディレクトリを開くこと自体が失敗しうるため no-op と
/// する（fsync できないことを理由に I/O 全体を失敗させない。rename の
/// アトミック性自体は非 Unix でも成立する）。
#[cfg(unix)]
fn fsync_dir(path: &Path) -> std::io::Result<()> {
    fs::File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn fsync_dir(_path: &Path) -> std::io::Result<()> {
    Ok(())
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

/// [`store_cache_entry_in`] の手順 1〜3（一時ディレクトリ作成・
/// `kernel.cu`／`kernel.ptx` 書き込み・各ファイルの fsync・一時
/// ディレクトリ自体の fsync）を担う。
///
/// `tmp_dir` は [`fs::create_dir`]（排他的作成。既存パスなら `Err`）で
/// 確保するため、[`temp_entry_dir_name`] の一意性が壊れていれば本関数が
/// `Err` を返す（サイレントな上書きを避ける）。
fn write_entry_files(tmp_dir: &Path, kernel_cu: &str, kernel_ptx: &str) -> Result<(), CudaError> {
    fs::create_dir(tmp_dir).map_err(|e| CudaError::CacheIo {
        detail: format!(
            "failed to create temp cache directory {}: {e}",
            tmp_dir.display()
        ),
    })?;

    let cu_path = tmp_dir.join(CACHE_ENTRY_SOURCE_FILE);
    fs::write(&cu_path, kernel_cu).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to write {}: {e}", cu_path.display()),
    })?;
    fsync_file(&cu_path).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to fsync {}: {e}", cu_path.display()),
    })?;

    let ptx_path = tmp_dir.join(CACHE_ENTRY_PTX_FILE);
    fs::write(&ptx_path, kernel_ptx).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to write {}: {e}", ptx_path.display()),
    })?;
    fsync_file(&ptx_path).map_err(|e| CudaError::CacheIo {
        detail: format!("failed to fsync {}: {e}", ptx_path.display()),
    })?;

    fsync_dir(tmp_dir).map_err(|e| CudaError::CacheIo {
        detail: format!(
            "failed to fsync temp cache directory {}: {e}",
            tmp_dir.display()
        ),
    })
}

/// [`cache_root`] が解決した候補パスを実際に fs 上へ実体化し、symlink
/// 解決込みの containment 再検証を行う（イシュー #509・Phase C-3。
/// C-2 が `docs/cuda-jit-cache-design.md` 「残課題」節で明示的に委譲した
/// 「symlink 解決込みの再検証」を実装する）。
///
/// `candidate_root` は [`cache_root`]（環境変数解決＋字句正規化のみの
/// containment 検証）の戻り値を渡す想定。本関数はそれを
/// [`fs::create_dir_all`] で実体化した後、[`Path::canonicalize`]
/// （symlink 解決込み）した実在パスで [`path_lexically_within`] による
/// containment を**再検証**する。C-2 時点の字句正規化のみの検証は、
/// `candidate_root` の祖先ディレクトリが symlink 経由で `workspace_root`
/// 配下を指すケース（例: `~/.cache` が `<workspace_root>/evil` への
/// symlink）を見逃す（字句上は `workspace_root` 配下に見えないため）。
/// ディレクトリを実際に作成した後でなければ symlink 解決ができない
/// （存在しないパスは `canonicalize` が `Err` を返す）ため、この再検証は
/// fs I/O を行わない C-2 の純関数群では実行できず C-3（本関数）のスコープ
/// とされていた。
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

    fs::create_dir_all(candidate_root).map_err(|e| CudaError::CacheIo {
        detail: format!(
            "failed to create cache root directory {}: {e}",
            candidate_root.display()
        ),
    })?;

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
/// （破損。クラッシュ残骸・外部破壊）場合、そのエントリを
/// [`fs::remove_dir_all`] で削除して rename を**一度だけ**再試行する。
/// 再試行後も失敗する場合（別プロセスが同時に置き直した等）は再度
/// 有効性を確認し、有効なら正常系として吸収する。それでも無効なら
/// 無限リトライせず [`CudaError::CacheIo`] で fail-closed に失敗させる
/// （DoS 耐性。`.claude/rules/security.md` A08）。
///
/// いずれのエラー経路でも一時ディレクトリの削除を試みる（best-effort。
/// 削除失敗はそれ自体をエラーにせず元エラーを優先する）。
fn store_cache_entry_in(
    root: &Path,
    key: &CudaKernelCacheKey,
    kernel_cu: &str,
    kernel_ptx: &str,
) -> Result<PathBuf, CudaError> {
    let final_dir = cache_entry_path_in(root, key)?;
    let entry_name = key.cache_entry_dir_name()?;
    let tmp_dir = root.join(temp_entry_dir_name(&entry_name));

    if let Err(e) = write_entry_files(&tmp_dir, kernel_cu, kernel_ptx) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err(e);
    }

    if fs::rename(&tmp_dir, &final_dir).is_ok() {
        // ルートディレクトリ自体の fsync は best-effort（rename の
        // アトミック性・可視性自体は fsync に依存しない。エントリ追加が
        // ディスクへ確実に反映されることを高めるための追加防御に留める）。
        let _ = fsync_dir(root);
        return Ok(final_dir);
    }

    // rename 失敗経路: 他プロセス先着（正常系）／破損エントリ置換／
    // その他 fs エラーのいずれかを、最終パスの状態から判別する。
    if validate_cache_entry(&final_dir) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Ok(final_dir);
    }

    if final_dir.exists() {
        // 破損エントリ。置換して一度だけ再試行する（無限リトライしない）。
        let _ = fs::remove_dir_all(&final_dir);
        if fs::rename(&tmp_dir, &final_dir).is_ok() {
            let _ = fsync_dir(root);
            return Ok(final_dir);
        }
    }

    let _ = fs::remove_dir_all(&tmp_dir);
    if validate_cache_entry(&final_dir) {
        // 再試行の間に別プロセスが有効なエントリを置いた（正常系）。
        Ok(final_dir)
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
fn load_cache_entry_in(
    root: &Path,
    key: &CudaKernelCacheKey,
) -> Result<Option<CachedKernel>, CudaError> {
    let entry_dir = cache_entry_path_in(root, key)?;
    if !validate_cache_entry(&entry_dir) {
        return Ok(None);
    }

    let kernel_cu = fs::read_to_string(entry_dir.join(CACHE_ENTRY_SOURCE_FILE)).map_err(|e| {
        CudaError::CacheIo {
            detail: format!("failed to read cached kernel source: {e}"),
        }
    })?;
    let kernel_ptx = fs::read_to_string(entry_dir.join(CACHE_ENTRY_PTX_FILE)).map_err(|e| {
        CudaError::CacheIo {
            detail: format!("failed to read cached kernel ptx: {e}"),
        }
    })?;

    Ok(Some(CachedKernel {
        kernel_cu,
        kernel_ptx,
    }))
}

/// [`load_cache_entry_in`] の公開ラッパー（イシュー #509・Phase C-3）。
/// `workspace_root` から [`ensure_cache_root`] でルートを実体化した上で
/// 委譲する（store 側と対称。ルート実体化は冪等なため読み出し専用の
/// 呼び出しでも安全）。
#[allow(
    dead_code,
    reason = "C-4(#511) の crate 内呼び出し元が実装されるまでの意図的な \
              先行スキャフォールディング（PR #659 の cache_root と同じ判断）"
)]
pub(crate) fn load_cache_entry(
    workspace_root: &Path,
    key: &CudaKernelCacheKey,
) -> Result<Option<CachedKernel>, CudaError> {
    let root = ensure_cache_root(workspace_root)?;
    load_cache_entry_in(&root, key)
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

        let loaded = load_cache_entry_in(&root, &key)
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

        let loaded = load_cache_entry_in(&root, &key).expect("load must not error, only miss");
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
        let loaded = load_cache_entry_in(&root, &key).expect("load must not error, only miss");
        assert!(loaded.is_none());

        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    // 未書き込みキーの load はミス（`Ok(None)`）を返すこと。
    #[test]
    fn load_returns_none_when_entry_absent() {
        let root = fresh_temp_dir("miss");
        let key = sample_key();

        let loaded = load_cache_entry_in(&root, &key).expect("load must succeed");
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
        let loaded = load_cache_entry_in(&root, &key)
            .expect("load must succeed")
            .expect("entry must exist");
        // 先着（1 回目）の内容が保たれ、2 回目の書き込みで上書きされない
        // こと（rename 失敗時に自分の一時ディレクトリを捨てる契約）。
        assert_eq!(loaded.kernel_cu, "first.cu");
        assert_eq!(loaded.kernel_ptx, "first.ptx");

        let _ = fs::remove_dir_all(&root);
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

        let loaded = load_cache_entry_in(&root, &key).expect("load must succeed");
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

        let loaded = load_cache_entry_in(&root, &key)
            .expect("load must succeed")
            .expect("replaced entry must be a hit");
        assert_eq!(loaded.kernel_cu, "fresh-cu");
        assert_eq!(loaded.kernel_ptx, "fresh-ptx");

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
