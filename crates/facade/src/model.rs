//! ローカルモデルレジストリ公開面（イシュー #2087・親 #2082）。
//!
//! 利用者がホームディレクトリ配下に手動配置したプリトレイン重みを、
//! 名前・バージョンで一元管理し同期ロードする入口 `ModelRegistry` を
//! 提供する。数値経路（`Op`／`BackendOps`／VJP／カーネル）には一切
//! 触れず、ホスト側のパス管理と既存の
//! [`crate::interop::safetensors::load_safetensors_f32`]（#2019）への
//! 委譲のみで完結する（REQ-12「任意 `BackendOps` 注入の公開 API を
//! 設けない」と矛盾しない）。
//!
//! # ディレクトリレイアウト（規定）
//!
//! ```text
//! <cache_dir>/                      既定 $HOME/.fandhe-ai/models
//!                                    （Windows は $USERPROFILE。
//!                                    [`ModelRegistry::new`] 参照）
//!   <name>/                         `[A-Za-z0-9._-]+`（先頭 '.' 不可）
//!     <version>/                    同上。semver 等の記法は解釈しない
//!                                    不透明な文字列として扱う
//!       model.safetensors           F32 テンソルのみ・キー = state_dict
//!                                    キー（転置・リネームなし。REQ-7）
//! ```
//!
//! 配置は利用者（または将来のダウンロード側。#2088 リモート取得。本
//! イシューのスコープ外）が行う。`ModelRegistry` はディレクトリの
//! 作成・削除を一切行わない**読み取り専用**のレジストリである。
//!
//! # `load` の戻り値型（受入条件からの逸脱・承認事項）
//!
//! イシュー受入条件の字面は `load` の戻り値要素型を単一 `Tensor` とする
//! が、本実装は [`std::collections::HashMap<String, crate::Tensor<f32>>`]
//! （state dict）を返す。safetensors ファイルは複数テンソルを保持し、
//! 消費先の [`crate::compat::Sequential::load_state_dict`] も map を
//! 取るため、単一 `Tensor` 化には恣意的な規則（どのキーを選ぶか）が
//! 必要になる。詳細・代替案（`load_tensor(name, version, key)` の追加）
//! は `docs/facade-model-registry-decision.md` を参照。
//!
//! # 非信頼入力の扱い（OWASP A01／A03。`.claude/rules/security.md`）
//!
//! `name`・`version` は利用者から渡される非信頼入力であり、パス
//! トラバーサル（`..`・絶対パス区切り混入等）を防ぐため
//! `validate_component` が許可文字集合（`[A-Za-z0-9._-]+`・先頭 `.`
//! 不可）でファイルシステムへ触れる前に fail-closed 検証する。
//! safetensors ファイル自体の検証（ヘッダ・dtype・shape）は
//! `crate::interop::safetensors` に一元化されており本モジュールは
//! 複製・迂回しない。
//!
//! 文字集合検証だけでは、レジストリ配下に事前配置されたシンボリック
//! リンク経由のキャッシュルート脱出（`<root>/<name>`・`<version>`・
//! `model.safetensors` のいずれかがリンクである場合）を防げない
//! （codex-review 指摘・PR #2226）。`load`・`available_models` は
//! ともに内部の `resolve_model_file` を経由し、次の多層防御で
//! シンボリックリンク経由の脱出・検査後の差し替え（TOCTOU）・非通常
//! ファイル（FIFO・ソケット・デバイス等）の受理をすべて防ぐ（同じく
//! codex-review 指摘・PR #2226。当初案は「検査後の open までの TOCTOU
//! は許容依存 9 区分に無い `libc` 直接依存が要るため対象外」としていた
//! が、`libc` クレートを追加せずとも `std::os::unix::fs::OpenOptionsExt`
//! （`custom_flags`）で `O_NOFOLLOW`／`O_NONBLOCK` の生値を渡せるため、
//! この判断は撤回し下記の対策へ差し替えた）:
//!
//! 1. `name`・`version`・葉ファイルの各段を [`std::fs::symlink_metadata`]
//!    （リンクを辿らない）で検査し、シンボリックリンクを拒否する。葉は
//!    `is_dir() == false` ではなく **`is_file() == true`** を明示要求し、
//!    FIFO・Unix ソケット・デバイスファイル等を拒否する。
//! 2. 葉パスを canonicalize し正規化済みキャッシュルート配下である
//!    ことを多層防御として再確認する（この canonicalize 自体は検査時点
//!    のスナップショットであり単独では TOCTOU を閉じない）。
//! 3. 葉を [`open_leaf_no_follow`] で開く。Linux／macOS は
//!    `O_NOFOLLOW`（最終コンポーネントのシンボリックリンク追跡を
//!    カーネルレベルで拒否）と `O_NONBLOCK`（FIFO への差し替えで
//!    `open` が無期限ブロックするのを防ぐ。通常ファイルには無効）を
//!    生の flag 値で付与する。それ以外の OS はプレーンな `open` に
//!    フォールバックする。
//! 4. 開いたハンドルの `fstat`（[`std::fs::File::metadata`]）で
//!    `is_file()` を再確認したうえで、Unix では手順 1 で取得した
//!    `symlink_metadata` の `(dev, ino)` と一致することを検証する
//!    （「検査と open のハンドル一体化」）。手順 1〜3 の間に
//!    `name`／`version`／葉のいずれかを別ファイルへ差し替えられても、
//!    差し替え後に実際に開かれたファイルの実体識別子（デバイス番号＋
//!    inode 番号）は検査時点のものと一致しないため、この不一致で
//!    確実に検出できる（パスの再解決ではなく実体の同一性で判定するため、
//!    中間ディレクトリの差し替えも同じ仕組みで捕捉する）。
//! 5. 以降は同じ [`std::fs::File`] ハンドルからバイト列を読み取り
//!    [`load_safetensors_f32_from_bytes`](crate::interop::safetensors::load_safetensors_f32_from_bytes)
//!    へ渡す（パスを使って再度 open し直すと手順 3〜4 で閉じた TOCTOU
//!    窓が復活するため、ハンドルの使い回しは必須）。
//!
//! 対象外として残る経路: レジストリ内に事前配置されたハードリンク
//! （攻撃者が任意タイミングで作成できるのは同一ファイルシステム上の
//! 既存ファイルへのリンクのみであり、所有者・権限チェックを伴わない
//! 本レジストリの脅威モデル外）。
//!
//! # 対象外
//!
//! リモート取得・HF hub 連携（#2088）、`docs/model-distribution-
//! design.md` の作成（親 #2082 の成果物）、version 記法の解釈
//! （semver・hash 等）・最新版解決、`compat::Sequential` への直結
//! ラッパー（`Sequential::from_registry` 等）、非 F32 dtype、ファイル
//! サイズ上限・改竄検知（値の決定にはユーザー承認が要るため）。

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Tensor;
use crate::interop::safetensors::{LoadError, load_safetensors_f32_from_bytes};

/// レイアウト規定上のファイル名（モジュール doc 参照）。
const MODEL_FILE_NAME: &str = "model.safetensors";

/// `open_leaf_no_follow` が付与する生の `open(2)` flag 値（Linux／macOS
/// のみ。許容依存 9 区分に `libc` が無いため、カーネル UAPI ヘッダ
/// 由来の固定値を直接埋め込む。`std::os::unix::fs::OpenOptionsExt::
/// custom_flags` はこの生値をそのまま `open` システムコールへ渡す）。
#[cfg(target_os = "linux")]
mod open_flags {
    /// `include/uapi/asm-generic/fcntl.h`（x86_64・aarch64 Linux 共通。
    /// alpha／parisc／sparc／mips 等の非対応アーキテクチャは本リポジトリの
    /// 対象外）。
    pub(crate) const O_NONBLOCK: i32 = 0o4_000;
    pub(crate) const O_NOFOLLOW: i32 = 0o400_000;
    /// `include/uapi/asm-generic/errno.h`。`O_NOFOLLOW` がシンボリック
    /// リンクを検出した際に `open(2)` が返す errno（ELOOP）。
    /// `std::io::ErrorKind::FilesystemLoop` は本リポジトリの pin toolchain
    /// （`rust-toolchain.toml`）でも `#![feature(io_error_more)]` 相当の
    /// unstable のため使えず、`raw_os_error()` の生値で判定する。
    pub(crate) const ELOOP: i32 = 40;
}
#[cfg(target_os = "macos")]
mod open_flags {
    /// `<sys/fcntl.h>`（Darwin／macOS）。
    pub(crate) const O_NONBLOCK: i32 = 0x0004;
    pub(crate) const O_NOFOLLOW: i32 = 0x0100;
    /// `<sys/errno.h>`（Darwin／macOS）の ELOOP。上記 Linux 側コメント参照。
    pub(crate) const ELOOP: i32 = 62;
}

/// 葉ファイル（`model.safetensors`）をシンボリックリンク追跡なし・
/// 非ブロッキングで開く（モジュール doc「非信頼入力の扱い」節の手順
/// 3 参照）。Linux／macOS は `O_NOFOLLOW`（最終コンポーネントの
/// シンボリックリンクを拒否）・`O_NONBLOCK`（FIFO への差し替えによる
/// 無期限ブロックを防ぐ。通常ファイルの読み取りには影響しない）を
/// 付与する。それ以外の OS はプレーンな `File::open` にフォールバック
/// する（対応する生 flag 値を持たないため。呼び出し元の fstat 識別子
/// 一致検査は OS に依らず TOCTOU を捕捉する）。
fn open_leaf_no_follow(leaf: &Path) -> std::io::Result<File> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(open_flags::O_NOFOLLOW | open_flags::O_NONBLOCK)
            .open(leaf)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        File::open(leaf)
    }
}

/// ローカルモデルレジストリの失敗を表す型付きエラー。`#[non_exhaustive]`:
/// 将来のフィールド追加（例: 改竄検知失敗）を非破壊にするため。
#[non_exhaustive]
#[derive(Debug)]
pub enum ModelError {
    /// 既定のキャッシュディレクトリを解決できない（[`ModelRegistry::new`]。
    /// Unix は `HOME`、Windows は `USERPROFILE` のいずれも未設定）。
    CacheDirUnavailable,
    /// `name`・`version` のいずれかが許可文字集合
    /// （`[A-Za-z0-9._-]+`・先頭 `.` 不可）に適合しない
    /// （パストラバーサル対策。ファイルシステムへ触れる前に拒否する）。
    InvalidComponent { kind: &'static str, value: String },
    /// `<cache_dir>/<name>/<version>/model.safetensors` が通常ファイル
    /// として存在しない。
    NotFound { name: String, version: String },
    /// safetensors デコード失敗（`crate::interop::safetensors::LoadError`
    /// を連鎖。ヘッダ不整合・未対応 dtype・shape 不整合等）。
    Load(LoadError),
    /// 上記以外の I/O 失敗（存在確認時の権限エラー等）。
    Io(std::io::Error),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::CacheDirUnavailable => {
                write!(
                    f,
                    "既定のモデルキャッシュディレクトリを解決できません（HOME／USERPROFILE 未設定）"
                )
            }
            ModelError::InvalidComponent { kind, value } => {
                write!(f, "不正な {kind}: {value:?}")
            }
            ModelError::NotFound { name, version } => {
                write!(
                    f,
                    "モデルが見つかりません（name={name}・version={version}）"
                )
            }
            ModelError::Load(e) => write!(f, "モデルのロードに失敗しました: {e}"),
            ModelError::Io(e) => write!(f, "モデルレジストリの I/O に失敗しました: {e}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Load(e) => Some(e),
            ModelError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// `name`・`version` の許可文字集合検証（OWASP A03 パストラバーサル
/// 対策）。`load`・[`ModelRegistry::available_models`] の両方が同一の
/// この関数を使う。`kind` はエラーメッセージ用の "name" / "version"。
fn validate_component(kind: &'static str, value: &str) -> Result<(), ModelError> {
    let ok = !value.is_empty()
        && !value.starts_with('.')
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(ModelError::InvalidComponent {
            kind,
            value: value.to_string(),
        })
    }
}

/// 既定キャッシュディレクトリを `home` から純粋に導出する（環境変数の
/// 読み取りは [`ModelRegistry::new`] 側で行い、本関数は環境非依存の
/// 単体テストを可能にするために切り出した private 純関数）。
fn default_cache_dir_from(home: Option<OsString>) -> Option<PathBuf> {
    home.map(|h| PathBuf::from(h).join(".fandhe-ai").join("models"))
}

/// `HOME`／`USERPROFILE` のどちらを優先するかを OS 別に決める純関数
/// （環境変数の読み取り自体は [`ModelRegistry::new`] 側で行い、本関数は
/// 選択ロジックのみを環境非依存の単体テストで検証できるように切り出した。
/// `is_windows` を引数化しているのも同じ理由で、CI が Linux 環境でも
/// Windows 分岐をテストできるようにするため）。
///
/// Windows は `USERPROFILE` を優先する（公開ドキュメント規定の既定
/// ルート `$USERPROFILE/.fandhe-ai/models` と一致させる。codex-review
/// 指摘・PR #2226。`HOME` は両 OS で設定されうるため単純な `HOME`
/// 優先だと Windows で乖離する）。それ以外の OS は `HOME` を優先する。
fn select_home_var(
    is_windows: bool,
    home: Option<OsString>,
    userprofile: Option<OsString>,
) -> Option<OsString> {
    if is_windows {
        userprofile.or(home)
    } else {
        home.or(userprofile)
    }
}

/// ホームディレクトリ配下のキャッシュディレクトリを基盤とする
/// **ローカル限定**のモデルレジストリ。ディレクトリレイアウトは
/// モジュール doc の規定を参照。本レジストリはディレクトリの作成・
/// 削除を一切行わない読み取り専用の入口である。
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    /// 既定のキャッシュルート（Unix: `$HOME/.fandhe-ai/models`、
    /// Windows: `$USERPROFILE/.fandhe-ai/models`）で
    /// [`ModelRegistry`] を構築する。ディレクトリは作成しない
    /// （読み取り専用のレジストリ）。`HOME`／`USERPROFILE` のいずれも
    /// 未設定な場合は [`ModelError::CacheDirUnavailable`] を返す。
    pub fn new() -> Result<Self, ModelError> {
        let home = std::env::var_os("HOME");
        let userprofile = std::env::var_os("USERPROFILE");
        let selected = select_home_var(cfg!(windows), home, userprofile);
        let root = default_cache_dir_from(selected).ok_or(ModelError::CacheDirUnavailable)?;
        Ok(Self { root })
    }

    /// キャッシュルートを明示指定して [`ModelRegistry`] を構築する
    /// （統合テスト・CI・非標準配置向け。実 `$HOME` に触れずに
    /// テストするための入口。環境変数の書き換えは `cargo test` の
    /// 並列実行でプロセスグローバル競合を起こすため採用しない）。
    pub fn with_cache_dir(dir: impl Into<PathBuf>) -> Self {
        Self { root: dir.into() }
    }

    /// キャッシュルートのパスを返す。
    pub fn cache_dir(&self) -> &Path {
        &self.root
    }

    /// `<cache_dir>/<name>/<version>/model.safetensors` をシンボリック
    /// リンク経由のキャッシュルート脱出・TOCTOU・非通常ファイルの
    /// 受理を許さずに解決し、開いた [`File`] ハンドルを返す（OWASP
    /// A03。codex-review 指摘・PR #2226。手順の詳細はモジュール doc
    /// 「非信頼入力の扱い」節を参照）。`load`・
    /// [`available_models`](Self::available_models) の両方が同一の
    /// この関数を経由するため、列挙結果は必ず `load` が受理するパス
    /// のみを含む。
    ///
    /// 権限エラー等それ以外の I/O 失敗は [`ModelError::Io`] として
    /// 伝える。
    fn resolve_model_file(&self, name: &str, version: &str) -> Result<File, ModelError> {
        validate_component("name", name)?;
        validate_component("version", version)?;

        let not_found = || ModelError::NotFound {
            name: name.to_string(),
            version: version.to_string(),
        };

        let canonical_root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(not_found()),
            Err(e) => return Err(ModelError::Io(e)),
        };

        let name_dir = self.root.join(name);
        let version_dir = name_dir.join(version);
        let leaf = version_dir.join(MODEL_FILE_NAME);

        // 中間ディレクトリ（name・version）: シンボリックリンク拒否＋
        // ディレクトリ型必須。
        for component in [&name_dir, &version_dir] {
            let meta = match std::fs::symlink_metadata(component) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(not_found()),
                Err(e) => return Err(ModelError::Io(e)),
            };
            let file_type = meta.file_type();
            if file_type.is_symlink() || !file_type.is_dir() {
                return Err(not_found());
            }
        }

        // 葉ファイル: シンボリックリンク拒否＋**通常ファイルであることを
        // 明示要求**（`!is_dir()` ではなく `is_file()`。FIFO・Unix
        // ソケット・デバイスファイル等は通常ファイルではないため拒否
        // される。codex-review 指摘・PR #2226）。`leaf_meta` は手順 4
        // の識別子一致検査で使うため保持する。
        let leaf_meta = match std::fs::symlink_metadata(&leaf) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(not_found()),
            Err(e) => return Err(ModelError::Io(e)),
        };
        if leaf_meta.file_type().is_symlink() || !leaf_meta.is_file() {
            return Err(not_found());
        }

        // 多層防御: canonicalize 後も正規化済みルート配下であることを
        // 確認する（この時点はまだ検査のスナップショットであり単独では
        // TOCTOU を閉じない。手順 3〜4 で最終的に閉じる）。
        let canonical_leaf = leaf.canonicalize().map_err(ModelError::Io)?;
        if !canonical_leaf.starts_with(&canonical_root) {
            return Err(not_found());
        }

        // シンボリックリンク追跡なし・非ブロッキングで葉を開く
        // （`open_leaf_no_follow`）。ELOOP（O_NOFOLLOW がシンボリック
        // リンクへ差し替えられた葉を検出した場合）は
        // `ErrorKind::FilesystemLoop` として報告される。
        let file = match open_leaf_no_follow(&leaf) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(not_found()),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            Err(e) if e.raw_os_error() == Some(open_flags::ELOOP) => return Err(not_found()),
            Err(e) => return Err(ModelError::Io(e)),
        };

        // 検査と open のハンドル一体化: 開いたハンドルの fstat が
        // 通常ファイルであること、かつ Unix では手順 2 で lstat した
        // 葉と同一の実体（デバイス番号＋inode 番号）であることを
        // 確認する。手順 2〜ここまでの間に name／version／葉のいずれか
        // が別ファイルへ差し替えられていた場合、開かれた実体の識別子は
        // 検査時点のものと一致しないためここで確実に検出できる
        // （TOCTOU 対策。モジュール doc「非信頼入力の扱い」節参照）。
        let open_meta = file.metadata().map_err(ModelError::Io)?;
        if !open_meta.is_file() {
            return Err(not_found());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if open_meta.dev() != leaf_meta.dev() || open_meta.ino() != leaf_meta.ino() {
                return Err(not_found());
            }
        }

        Ok(file)
    }

    /// `<cache_dir>/<name>/<version>/model.safetensors` を同期ロードし、
    /// state dict（キー = テンソル名）を返す。
    ///
    /// パス解決・オープンは `resolve_model_file`（シンボリックリンク
    /// 経由のキャッシュルート脱出・TOCTOU・非通常ファイル対策。
    /// モジュール doc「非信頼入力の扱い」節参照）に委譲する。存在
    /// しない、通常ファイルでない、または脱出と判定された場合は
    /// [`ModelError::NotFound`] に丸める。権限エラー等それ以外の
    /// I/O 失敗は [`ModelError::Io`]。開いた同一ハンドルから読み
    /// 取ったバイト列を
    /// [`load_safetensors_f32_from_bytes`](crate::interop::safetensors::load_safetensors_f32_from_bytes)
    /// （`crate::interop::safetensors`。#2019）へ委譲する（パスで
    /// 再度 open し直すと `resolve_model_file` が閉じた TOCTOU 窓が
    /// 復活するため、ハンドルの使い回しは必須。#2019）。転置・
    /// キーリネーム等の暗黙アダプタは一切行わない（REQ-7 契約は
    /// `interop::safetensors` に一元化済みでロジックを複製しない）。
    pub fn load(
        &self,
        name: &str,
        version: &str,
    ) -> Result<HashMap<String, Tensor<f32>>, ModelError> {
        let mut file = self.resolve_model_file(name, version)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(ModelError::Io)?;
        load_safetensors_f32_from_bytes(&bytes).map_err(ModelError::Load)
    }

    /// キャッシュルート配下に存在するモデルを列挙する。ルートが
    /// 存在しない、または読み取れない場合はレジストリが空とみなし
    /// 空 `Vec` を返す（エラー型を返さないのは戻り値型の契約による。
    /// [`load`](Self::load) 側の I/O エラー区別とは異なる扱い）。
    ///
    /// 列挙規則:
    /// - ルート直下の**ディレクトリ**（[`std::fs::DirEntry::file_type`]
    ///   はリンクを辿らないため、シンボリックリンクは対象外）で名前が
    ///   `validate_component` を通るものを `name` 候補とする
    ///   （ファイル・検証不合格名はスキップ）
    /// - `name` 直下のディレクトリで名前が検証を通り、かつ
    ///   `resolve_model_file` が受理するものだけを `version` として
    ///   列挙する（`load` が受理するパスのみを列挙する一貫性保証。
    ///   モジュール doc「非信頼入力の扱い」節参照）
    /// - version が 1 件もない `name` は結果に含めない
    /// - `name`・`versions` とも文字列昇順ソート（決定的出力）
    /// - ディレクトリ名は UTF-8 変換できるもののみ対象
    ///   （非 UTF-8 名はスキップ）
    pub fn available_models(&self) -> Vec<(String, Vec<String>)> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let mut result: Vec<(String, Vec<String>)> = Vec::new();
        for entry in entries.flatten() {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if validate_component("name", &name).is_err() {
                continue;
            }
            // `file_type()` はリンクを辿らない（`Path::is_dir` は辿る
            // ため使わない）。シンボリックリンクの `name` ディレクトリは
            // 列挙対象外とする。
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let name_dir = entry.path();
            let Ok(version_entries) = std::fs::read_dir(&name_dir) else {
                continue;
            };
            let mut versions: Vec<String> = Vec::new();
            for version_entry in version_entries.flatten() {
                let Ok(version) = version_entry.file_name().into_string() else {
                    continue;
                };
                if validate_component("version", &version).is_err() {
                    continue;
                }
                if self.resolve_model_file(&name, &version).is_ok() {
                    versions.push(version);
                }
            }
            if versions.is_empty() {
                continue;
            }
            versions.sort();
            result.push((name, versions));
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_component_accepts_ascii_allowlist() {
        for value in ["a", "mlp", "v1", "1.2.3", "a_b-c.d", "A9"] {
            assert!(
                validate_component("name", value).is_ok(),
                "許可されるべき値が拒否された: {value:?}"
            );
        }
    }

    #[test]
    fn validate_component_rejects_traversal_and_unsafe_values() {
        for value in [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            ".hidden",
            "a b",
            "a\0b",
            "/etc/passwd",
        ] {
            assert!(
                validate_component("name", value).is_err(),
                "拒否されるべき値が許可された: {value:?}"
            );
        }
    }

    #[test]
    fn default_cache_dir_from_home_appends_fixed_suffix() {
        let home = Some(OsString::from("/home/x"));
        let dir = default_cache_dir_from(home).unwrap();
        assert_eq!(dir, PathBuf::from("/home/x/.fandhe-ai/models"));
    }

    #[test]
    fn default_cache_dir_from_none_is_none() {
        assert!(default_cache_dir_from(None).is_none());
    }

    #[test]
    fn select_home_var_prefers_userprofile_on_windows() {
        let home = Some(OsString::from("/home/x"));
        let userprofile = Some(OsString::from(r"C:\Users\x"));
        assert_eq!(
            select_home_var(true, home.clone(), userprofile.clone()),
            userprofile
        );
        // Windows でも `USERPROFILE` 未設定なら `HOME` へフォールバックする。
        assert_eq!(
            select_home_var(true, home, None),
            Some(OsString::from("/home/x"))
        );
    }

    #[test]
    fn select_home_var_prefers_home_on_non_windows() {
        let home = Some(OsString::from("/home/x"));
        let userprofile = Some(OsString::from(r"C:\Users\x"));
        assert_eq!(
            select_home_var(false, home.clone(), userprofile.clone()),
            home
        );
        // 非 Windows でも `HOME` 未設定なら `USERPROFILE` へフォールバックする。
        assert_eq!(
            select_home_var(false, None, userprofile.clone()),
            userprofile
        );
    }

    #[test]
    fn select_home_var_none_when_both_unset() {
        assert_eq!(select_home_var(true, None, None), None);
        assert_eq!(select_home_var(false, None, None), None);
    }
}
