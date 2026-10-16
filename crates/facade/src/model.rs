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
//!                                    （Windows のキャッシュルート解決は
//!                                    $USERPROFILE。ただし `load`／
//!                                    `available_models` は Windows で
//!                                    fail-closed 非対応。
//!                                    [`ModelRegistry::new`] 参照）
//!   <name>/                         `[A-Za-z0-9._-]+`（先頭 '.' 不可）
//!     <version>/                    同上。semver 等の記法は解釈しない
//!                                    不透明な文字列として扱う
//!       model.safetensors           F32 テンソルのみ・キー = state_dict
//!                                    キー（転置・リネームなし。REQ-7）
//! ```
//!
//! 配置は利用者（または将来のダウンロード側。#2088 リモート取得。本
//! イシューのスコープ外）が行うか、[`crate::model::ModelRegistry::save`]（#3119。
//! `docs/facade-model-registry-decision.md` §16）で書き込む。`save` 以外の
//! メソッド（`new`・`load`・`available_models` 等）はディレクトリの
//! 作成・削除を一切行わない読み取り専用で、`save` も削除は自分の一時
//! ファイルに限り、作成するのはルートと `<name>`・`<version>` だけである。
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
//! ファイル（FIFO・ソケット・デバイス等）の受理・上限なしのメモリ
//! 確保（手順 5〜6）をすべて防ぐ（同じく
//! codex-review 指摘・PR #2226。当初案は「検査後の open までの TOCTOU
//! は当時の許容依存区分に無い `libc` 直接依存が要るため対象外」としていた
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
//! 3. 葉を `open_leaf_no_follow` で開く。Linux（x86_64／aarch64 限定。
//!    `O_NOFOLLOW` の生値はアーキ間で異なるため。`crate::fs_guard` の `open_flags`
//!    モジュール doc 参照）／macOS は `O_NOFOLLOW`（最終コンポーネントの
//!    シンボリックリンク追跡をカーネルレベルで拒否）と `O_NONBLOCK`
//!    （FIFO への差し替えで `open` が無期限ブロックするのを防ぐ。通常
//!    ファイルには無効）を生の flag 値で付与する。それ以外の OS・
//!    アーキテクチャ（Windows・上記以外の Linux アーキ等）は同等の
//!    no-follow 実装を持たない、または値の出典を確認していないため、
//!    安全性を仮定せず **fail-closed でオープン自体を拒否**する
//!    （`ErrorKind::Unsupported` を返し `ModelError::Io` へ
//!    丸める）。キャッシュディレクトリ解決（`USERPROFILE`）自体は
//!    Windows でも動作するが、`resolve_model_file` を経由する `load`
//!    は常に失敗し（`ModelError::Io`）、`available_models` は各
//!    バージョンの存在確認に `resolve_model_file(..).is_ok()` を使う
//!    ため、失敗を伝播せず対象バージョンを一覧から除外する（結果的に
//!    Windows では常に空の一覧を返す。エラーにはならない）。
//! 4. 開いたハンドルの `fstat`（[`std::fs::File::metadata`]）で
//!    `is_file()` を再確認したうえで、Unix では手順 1 で取得した
//!    `symlink_metadata` の `(dev, ino)` と一致することを検証する
//!    （「検査と open のハンドル一体化」）。手順 1〜3 の間に
//!    `name`／`version`／葉のいずれかを別ファイルへ差し替えられても、
//!    差し替え後に実際に開かれたファイルの実体識別子（デバイス番号＋
//!    inode 番号）は検査時点のものと一致しないため、この不一致で
//!    確実に検出できる（パスの再解決ではなく実体の同一性で判定するため、
//!    中間ディレクトリの差し替えも同じ仕組みで捕捉する）。
//! 5. 開いたハンドルの `fstat` で取得済みのファイルサイズ
//!    （`std::fs::Metadata::len`）を `MAX_MODEL_FILE_BYTES` と比較し、
//!    上回る場合は読み取りに入る前に `ModelError::TooLarge` で拒否する
//!    （`model.safetensors` は非信頼な外部フォーマット入力であり、
//!    サイズ検証なしに丸ごと `Vec` へ確保するとメモリ枯渇を招く。
//!    codex-review 指摘・PR #2226・P0。AGENTS.md「外部フォーマットの
//!    パース検証（P0）」長さ事前検証要件。当初 8 GiB としていたが、
//!    一般的な実行環境（GitHub ホステッド runner の既定 7 GiB RAM 等）
//!    では単一ファイルの 8 GiB 確保自体が実質的な OOM 防止にならない
//!    という codex-review 再指摘を受け 1 GiB へ引き下げた。値の
//!    導出根拠は `MAX_MODEL_FILE_BYTES` のドキュメントを参照。
//!    `crate::fs_guard` の crate 内限定 const のため intra-doc link にはせずコードスパンで
//!    参照する〈rustdoc は非公開項目のリンク先を解決できず
//!    `--no-deps` ビルドで `broken_intra_doc_links` になるため〉）。
//! 6. 以降は同じ [`std::fs::File`] ハンドルからバイト列を読み取る
//!    （パスを使って再度 open し直すと手順 3〜4 で閉じた TOCTOU 窓が
//!    復活するため、ハンドルの使い回しは必須）。手順 5 の fstat 後に
//!    ファイルが差し替え・追記されて増大する TOCTOU にも備え、
//!    `load` 側は `std::io::Read::take` で `MAX_MODEL_FILE_BYTES + 1`
//!    バイトを上限に読み取り、実際に読めたバイト数が
//!    `MAX_MODEL_FILE_BYTES` を超えていれば同じく
//!    `ModelError::TooLarge` で拒否する（許容量ちょうどで打ち切る
//!    と超過を検出できないため `+ 1` バイト分だけ多く読む）。読み
//!    取ったバイト列は
//!    [`load_safetensors_f32_from_bytes`](crate::interop::safetensors::load_safetensors_f32_from_bytes)
//!    へ渡す。
//!
//! 対象外として残る経路: レジストリ内に事前配置されたハードリンク
//! （攻撃者が任意タイミングで作成できるのは同一ファイルシステム上の
//! 既存ファイルへのリンクのみであり、所有者・権限チェックを伴わない
//! 本レジストリの脅威モデル外）。
//!
//! # Windows 対応状況
//!
//! 公開ドキュメントとの整合（codex-review 指摘・PR #2226・P1）。
//!
//! [`crate::model::ModelRegistry::new`] のキャッシュルート解決
//! （`$USERPROFILE` 優先）自体は Windows でも動作するが、
//! [`crate::model::ModelRegistry::load`]・
//! [`crate::model::ModelRegistry::available_models`] が経由する `resolve_model_file`
//! の葉オープン（`open_leaf_no_follow`）は Linux／macOS 限定実装のため、
//! Windows では常に失敗する（`load` は必ず `Err`、`available_models`
//! は常に空の一覧を返す。上記「非信頼入力の扱い」節・
//! `open_leaf_no_follow` のドキュメント参照）。安全な Windows 実装
//! （`file_index`／
//! `volume_serial_number` によるハンドル識別子照合を含む）の追加は
//! スコープ外として別イシューで追跡する（詳細は
//! `docs/facade-model-registry-decision.md`）。
//!
//! facade 自体は x86_64-pc-windows-msvc でビルドでき、CI の Windows クロス
//! clippy で継続検査している（#2390・#2391）。ただし `ModelRegistry::load`／
//! `available_models` は上記のとおり `Unsupported` の fail-closed のままで、
//! Windows 実機での動作は未検証（#2393）。
//!
//! # 対象外
//!
//! リモート取得・HF hub 連携（#2088）、`docs/model-distribution-
//! design.md` の作成（親 #2082 の成果物）、version 記法の解釈
//! （semver・hash 等）・最新版解決、`compat::Sequential` への直結
//! ラッパー（`Sequential::from_registry` 等）、非 F32 dtype、改竄検知、
//! Windows 向け安全な no-follow オープン実装。

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Tensor;
use crate::fs_guard::{MAX_MODEL_FILE_BYTES, open_leaf_no_follow};
use crate::interop::safetensors::{LoadError, SaveError, load_safetensors_f32_from_bytes};

/// レイアウト規定上のファイル名（モジュール doc 参照）。
const MODEL_FILE_NAME: &str = "model.safetensors";

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
    /// `model.safetensors` のファイルサイズが固定上限（内部定数
    /// `MAX_MODEL_FILE_BYTES`）を超過している（メモリ枯渇対策。
    /// モジュール doc「非信頼入力の扱い」節手順 5〜6 参照）。`len` は
    /// 検出時点で判明していた実際のバイト数（`fstat` 実測値、または
    /// TOCTOU 検出時は `max + 1` 超）、`max` はその固定上限値。
    TooLarge {
        name: String,
        version: String,
        len: u64,
        max: u64,
    },
    /// 保存対象のテンソル集合が空（[`ModelRegistry::save`]。空の
    /// safetensors は `load` で何も復元できないため書き込み前に拒否する。
    /// `name`・`version` は許可文字集合検証済みの値で、パスは含まない）。
    EmptyModel { name: String, version: String },
    /// safetensors エンコード失敗（[`ModelRegistry::save`]。
    /// `crate::interop::safetensors::SaveError` を連鎖）。
    Save(SaveError),
    /// safetensors デコード失敗（`crate::interop::safetensors::LoadError`
    /// を連鎖。ヘッダ不整合・未対応 dtype・shape 不整合等）。
    Load(LoadError),
    /// 上記以外の I/O 失敗（存在確認時の権限エラー等）。読み込み
    /// バッファの確保失敗（[`std::io::ErrorKind::OutOfMemory`]。
    /// `reserve_read_buffer`）はその一例。
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
            ModelError::TooLarge {
                name,
                version,
                len,
                max,
            } => write!(
                f,
                "モデルファイルが上限を超えています（name={name}・version={version}・{len} バイト > 上限 {max} バイト）"
            ),
            ModelError::EmptyModel { name, version } => write!(
                f,
                "保存するテンソルが空です（name={name}・version={version}）"
            ),
            ModelError::Save(e) => write!(f, "モデルの保存に失敗しました: {e}"),
            ModelError::Load(e) => write!(f, "モデルのロードに失敗しました: {e}"),
            ModelError::Io(e) => write!(f, "モデルレジストリの I/O に失敗しました: {e}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Save(e) => Some(e),
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

/// 本番の一時ファイル名（`compat::model_io::tmp_manifest_name` と同型。先頭 `.` のため
/// `validate_component` を通らず、`load`／`available_models` からは見えない）。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn production_tmp_name() -> String {
    let (pid, nanos, counter) = crate::fs_write::name_entropy();
    format!(".{MODEL_FILE_NAME}.tmp-{pid}-{counter}-{nanos}")
}

/// `/proc/self/fd/<fd>` 形式のパス（保持中のディレクトリ fd を基準にした相対解決用）。
/// `save` はこのパスの下にだけ作成・`rename` する。`<fd>` が指す inode は
/// 開いた時点で固定されるため、以後 `<name>`・`<version>` のパス文字列を
/// シンボリックリンクへ差し替えられても書き込み先は動かない。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn proc_fd_path(dir: &File) -> PathBuf {
    use std::os::unix::io::AsRawFd;
    PathBuf::from(format!("/proc/self/fd/{}", dir.as_raw_fd()))
}

/// `/proc/self/fd` が使えるか（ディレクトリとして開けるか）を確認する。
/// `save` がルートを作る前に呼び、`/proc` 未マウントの環境では副作用（ルート・
/// 親ディレクトリの作成）なしで `Err` を返すための fail-closed 前提検査
/// （設計記録 §16.9。codex-review 指摘・PR #3377 P2）。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn ensure_proc_fd_usable(proc_fd_dir: &Path) -> std::io::Result<()> {
    std::fs::read_dir(proc_fd_dir).map(|_| ())
}

/// `save` のキャッシュルートをディレクトリ fd として開く（ルート自体が
/// シンボリックリンクでも許容する。利用者が選んだ配置で、`load` 側と同じ扱い）。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn open_root_dir(root: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(crate::fs_guard::open_flags::O_DIRECTORY)
        .open(root)
}

/// `parent`（保持中のディレクトリ fd）の直下に `component`（検証済みの 1 段）を
/// ディレクトリとして用意し、その fd を返す。存在しなければ `create_dir` で 1 段だけ
/// 作る（`create_dir_all` は使わない）。開く際は `O_NOFOLLOW|O_DIRECTORY` を付け、
/// 最終コンポーネントがシンボリックリンク・非ディレクトリなら拒否する
/// （`InvalidInput`）。パス検査と使用の間の差し替え窓を作らないよう、検査ではなく
/// 「開いた fd」を以後の基準にする。エラーメッセージにパスを含めない。
/// `/proc` が使えない環境では開けず `Err`（書き込み前の fail-closed）。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn open_dir_component(parent: &File, component: &str) -> std::io::Result<File> {
    use crate::fs_guard::open_flags::{ELOOP, O_DIRECTORY, O_NOFOLLOW};
    use std::io::{Error as IoError, ErrorKind};
    use std::os::unix::fs::OpenOptionsExt;
    // ENOTDIR（asm-generic errno。x86_64・aarch64 共通）
    const ENOTDIR: i32 = 20;
    let path = proc_fd_path(parent).join(component);
    match std::fs::create_dir(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECTORY | O_NOFOLLOW)
        .open(&path)
        .map_err(|e| match e.raw_os_error() {
            Some(code) if code == ELOOP || code == ENOTDIR => IoError::new(
                ErrorKind::InvalidInput,
                "保存先の中間パスがシンボリックリンクまたはディレクトリではありません",
            ),
            _ => e,
        })
}

/// `len` が `max` を超えていないか検査する純関数（環境非依存の単体
/// テストを可能にするため `resolve_model_file`・[`ModelRegistry::load`]
/// 双方の判定ロジックを切り出した。モジュール doc「非信頼入力の扱い」
/// 節手順 5〜6・`MAX_MODEL_FILE_BYTES`〈`crate::fs_guard` の crate 内限定 const のため
/// intra-doc link にはせずコードスパンで参照〉参照）。
fn enforce_size_limit(name: &str, version: &str, len: u64, max: u64) -> Result<(), ModelError> {
    if len > max {
        Err(ModelError::TooLarge {
            name: name.to_string(),
            version: version.to_string(),
            len,
            max,
        })
    } else {
        Ok(())
    }
}

/// 読み込みバッファ（[`ModelRegistry::load_with_limit`] が
/// safetensors ファイル全体を読み込む先）を `capacity_hint` バイトで
/// 事前確保する。本番では `capacity_hint` に
/// `usize::try_from(..).unwrap_or(usize::MAX)` 由来の値（`u64` が
/// `usize` に収まらない場合の飽和）も渡りうる。
///
/// 確保失敗（`Vec::try_reserve` の `Err`）は
/// [`ModelError::Io`]`(`[`std::io::Error::from`]`(`
/// [`std::io::ErrorKind::OutOfMemory`]`))` で返す。
/// `std::io::Error::other` は元エラーを `Box<dyn Error>` として
/// ヒープに確保する（Custom 表現）ため、実メモリ枯渇時にはエラーを
/// 構築すること自体が確保失敗しうる（「確保失敗は panic／abort では
/// なく `Err` で伝播する」契約を報告経路自体が破ってしまう）。対して
/// `io::Error::from(ErrorKind)` は Simple 表現でヒープ確保なしに
/// 構築できるため、確保失敗の報告経路として安全に使える。要求
/// サイズ等の診断情報は失われるが、`ErrorKind::OutOfMemory` だけで
/// 失敗理由は呼び出し元に伝わり、確保失敗の報告で再確保しないことを
/// 優先する。autodiff 側の `nn::init::alloc_failed()`／
/// `eval::linalg::alloc_failed()`（PR #2239）と同じ類型（#2250）。
fn reserve_read_buffer(capacity_hint: usize) -> Result<Vec<u8>, ModelError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve(capacity_hint)
        .map_err(|_| ModelError::Io(std::io::Error::from(std::io::ErrorKind::OutOfMemory)))?;
    Ok(bytes)
}

/// 既定キャッシュディレクトリを `home` から純粋に導出する（環境変数の
/// 読み取りは [`ModelRegistry::new`] 側で行い、本関数は環境非依存の
/// 単体テストを可能にするために切り出した private 純関数）。
fn default_cache_dir_from(home: Option<OsString>) -> Option<PathBuf> {
    home.map(|h| PathBuf::from(h).join(".fandhe-ai").join("models"))
}

/// 環境変数から読み取った値が空文字列の場合は「未設定」として扱う
/// （`HOME=""` ／ `USERPROFILE=""` を有効なホームパスとして受理すると
/// `PathBuf::from("")` はカレントディレクトリ相対の空パスになり、
/// [`default_cache_dir_from`] が `.fandhe-ai/models`
/// というカレントディレクトリ相対のキャッシュルートを構築してしまう
/// ため。本来は [`ModelError::CacheDirUnavailable`] になるべき
/// （Cursor Bugbot 指摘・PR #2226）。
fn non_empty_os_string(value: Option<OsString>) -> Option<OsString> {
    value.filter(|v| !v.is_empty())
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
///
/// 空文字列の値は [`non_empty_os_string`] で「未設定」に正規化した
/// うえで優先順位判定する（Cursor Bugbot 指摘・PR #2226。呼び出し元
/// [`ModelRegistry::new`] 側でフィルタせず本関数側で行うのは、判定
/// ロジックと正規化を分離すると将来どちらかだけが更新され再発しうる
/// ため）。例えば `HOME=""`・`USERPROFILE="/x"` の非 Windows では
/// `HOME` を空文字列のまま優先せず `USERPROFILE` へフォールバックする。
fn select_home_var(
    is_windows: bool,
    home: Option<OsString>,
    userprofile: Option<OsString>,
) -> Option<OsString> {
    let home = non_empty_os_string(home);
    let userprofile = non_empty_os_string(userprofile);
    if is_windows {
        userprofile.or(home)
    } else {
        home.or(userprofile)
    }
}

/// ホームディレクトリ配下のキャッシュディレクトリを基盤とする
/// **ローカル限定**のモデルレジストリ。ディレクトリレイアウトは
/// モジュール doc の規定を参照。読み取り系（`load`・`available_models`）は
/// ディレクトリの作成・削除を行わず、書き込みは [`ModelRegistry::save`] だけが行う。
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    /// 既定のキャッシュルート（Unix: `$HOME/.fandhe-ai/models`、
    /// Windows: `$USERPROFILE/.fandhe-ai/models`）で
    /// [`ModelRegistry`] を構築する。ディレクトリは作成しない
    /// （構築は副作用なし。作成は [`save`](Self::save) のみ）。`HOME`／`USERPROFILE` のいずれも
    /// 未設定な場合は [`ModelError::CacheDirUnavailable`] を返す。
    ///
    /// **Windows での対応状況**: 本関数によるキャッシュルート解決
    /// 自体は Windows でも動作するが、[`load`](Self::load)・
    /// [`available_models`](Self::available_models) は Windows では
    /// 常に失敗（もしくは空の一覧）を返す fail-closed 非対応である
    /// （モジュール doc「Windows 対応状況」節参照）。
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
    ///
    /// サイズ上限は本番経路では常に `MAX_MODEL_FILE_BYTES`
    /// （[`Self::resolve_model_file`] 経由）だが、`max` を引数化する
    /// ことで単体テストが実際の 1 GiB ファイルを用意せずに fstat 経路の
    /// 上限超過拒否を検証できるようにする（`crate::model::tests::
    /// resolve_model_file_with_limit_rejects_over_bound` 参照）。
    fn resolve_model_file_with_limit(
        &self,
        name: &str,
        version: &str,
        max: u64,
    ) -> Result<File, ModelError> {
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
            #[cfg(any(
                all(
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                ),
                target_os = "macos"
            ))]
            Err(e) if e.raw_os_error() == Some(crate::fs_guard::open_flags::ELOOP) => {
                return Err(not_found());
            }
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

        // ファイルサイズ上限検査（モジュール doc「非信頼入力の扱い」節
        // 手順 5。メモリ枯渇対策・codex-review 指摘・PR #2226・P0）。
        // 開いたハンドルの fstat 実測値で判定するため、手順 1〜4 の
        // 検査対象と同一の実体（TOCTOU 対策で確認済み）のサイズを見る。
        // ここを通過しても手順 6（`load` 側の `take` 二段構え）が
        // fstat 後の追記・差し替えによる増大を別途検出する。
        enforce_size_limit(name, version, open_meta.len(), max)?;

        Ok(file)
    }

    /// [`Self::resolve_model_file_with_limit`] を本番経路の固定上限
    /// `MAX_MODEL_FILE_BYTES` で呼び出す薄いラッパー。
    fn resolve_model_file(&self, name: &str, version: &str) -> Result<File, ModelError> {
        self.resolve_model_file_with_limit(name, version, MAX_MODEL_FILE_BYTES)
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
    /// [`load_safetensors_f32_from_bytes`]
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
        self.load_with_limit(name, version, MAX_MODEL_FILE_BYTES)
    }

    /// [`Self::load`] の本体。サイズ上限 `max` を引数化することで、
    /// 単体テストが実際の `MAX_MODEL_FILE_BYTES`（1 GiB）ファイルを
    /// 用意せずに「fstat 通過後の読み取り自体が上限超過を検出する」
    /// 手順 6 の TOCTOU 対策（`Read::take` 二段構え）を検証できる
    /// （`#[cfg(test)]` `load_with_limit_rejects_read_exceeding_bound`
    /// 参照。本番経路は `load` から `MAX_MODEL_FILE_BYTES` 固定で
    /// 呼ばれる）。
    fn load_with_limit(
        &self,
        name: &str,
        version: &str,
        max: u64,
    ) -> Result<HashMap<String, Tensor<f32>>, ModelError> {
        let mut file = self.resolve_model_file_with_limit(name, version, max)?;
        // `resolve_model_file_with_limit` の fstat 時点では `max`
        // 以内であることを確認済みだが、その後ファイルが差し替え・
        // 追記されて増大する TOCTOU に備え、読み取り自体も
        // `max + 1` バイトで打ち切る（モジュール doc「非信頼入力の
        // 扱い」節手順 6。ちょうど上限バイト数で打ち切ると超過を
        // 検出できないため `+ 1` 分だけ多く読む）。
        let limit = max.saturating_add(1);
        // 事前確保サイズは実測ファイルサイズ（`limit` 未満なら実測値、
        // 以上なら `limit`）を使う。`limit`（本番経路では 1 GiB + 1）
        // を毎回丸ごと事前確保すると小さい正規ファイルの読み込みでも
        // 無駄に大きな割り当てが発生するため、実測値を優先しつつ
        // `try_reserve` で割り当て失敗を panic ではなく型付きエラーへ
        // 変換する（実体は `reserve_read_buffer`）。
        let observed_len = file.metadata().map_err(ModelError::Io)?.len();
        let capacity_hint = usize::try_from(observed_len.min(limit)).unwrap_or(usize::MAX);
        let mut bytes = reserve_read_buffer(capacity_hint)?;
        file.by_ref()
            .take(limit)
            .read_to_end(&mut bytes)
            .map_err(ModelError::Io)?;
        enforce_size_limit(name, version, bytes.len() as u64, max)?;
        load_safetensors_f32_from_bytes(&bytes).map_err(ModelError::Load)
    }

    /// `tensors`（state dict）を `<cache_dir>/<name>/<version>/model.safetensors`
    /// へ保存する（[`load`](Self::load) の書き込み側の対。イシュー #3119・
    /// `docs/facade-model-registry-decision.md` §16）。既存ファイルは原子的に
    /// 置き換える（上書きフラグは持たない）。
    ///
    /// # 手順（順序固定）
    ///
    /// 1. 対応プラットフォーム判定（範囲外は副作用の前に `Unsupported`）
    /// 2. `name`・`version` の許可文字集合検証、空集合・サイズ上限の検査、
    ///    safetensors へのエンコードと読み戻しによる整合確認（ここまで
    ///    ファイルシステムには触れない）
    /// 3. ルートを開き、`<name>`・`<version>` ディレクトリを 1 段ずつ用意して
    ///    ディレクトリ fd を保持する（シンボリックリンク・非ディレクトリは拒否。
    ///    `create_dir_all` は使わない）
    /// 4. 保持した `<version>` の fd 配下の一時ファイル（`.model.safetensors.tmp-*`・
    ///    `create_new`）へ書き込み、実体識別子を再確認してから同じ fd 配下で
    ///    `rename`（唯一のコミット点）
    /// 5. `load` と同じ経路で開き直し、確定したファイルが書き込んだ実体で
    ///    あることを確認する
    ///
    /// 失敗時は自分が作った一時ファイルだけを（`(dev, ino)` で所有を確かめて）
    /// 削除し、既存の `model.safetensors` には触れない。作成済みの
    /// ディレクトリは削除しない。
    ///
    /// # `load` との往復契約
    ///
    /// `Ok(())` なら同じ `name`・`version` の [`load`](Self::load) が同じキーと
    /// 値（F32 の bit 一致）を返す。F32 限定は型 `Tensor<f32>` で保証される。
    ///
    /// # プラットフォーム
    ///
    /// Linux の x86_64／aarch64 のみ。保存先はディレクトリ fd で固定する
    /// （`/proc/self/fd/<fd>` 経由。std だけで `openat` 相当を得る手段）ため、
    /// `load` の対応範囲のうち macOS は対象外で、それ以外（Windows 等）と併せて
    /// 最初に `Io(Unsupported)` を返し何も作らない（範囲外では不正な `name`・
    /// `version` も `InvalidComponent` ではなく `Unsupported` になるが、いずれも
    /// 書き込み前の拒否である）。`/proc` が使えない Linux でも書き込み前に `Err`
    ///
    /// # 保存先の固定・並行性・耐久性
    ///
    /// - キャッシュルートを開いて得たディレクトリ fd を基準に `<name>`・`<version>` を
    ///   1 段ずつ `O_NOFOLLOW|O_DIRECTORY` で開き、一時ファイルの作成と `rename` は
    ///   その fd 配下だけで行う。検査後に `<name>`・`<version>` のパス文字列が
    ///   シンボリックリンクへ差し替えられても、書き込みは開いた時点の実体に
    ///   留まりルート外へ出ない（差し替えを検出した場合は確定後の検証が `Err` を返す。
    ///   `docs/facade-model-registry-decision.md` §16.6・§16.9）。ルート自体の
    ///   シンボリックリンクは利用者の選択として許容する
    /// - 並行する `save` は「最後に `rename` した側が勝つ」。調停は
    ///   サポート対象外
    /// - `fsync` はしない（電源断に対する耐久性は保証しない。原子性は
    ///   `rename` による）
    ///
    /// # 例
    ///
    /// 一時ディレクトリをキャッシュルートにして保存・再読込する（`$HOME` には
    /// 触れない）。対応外プラットフォーム（macOS・Windows 等）では `Unsupported`
    /// になるため何もしない。
    ///
    /// ```
    /// use fandhe_ai::Tensor;
    /// use fandhe_ai::model::{ModelError, ModelRegistry};
    /// use std::collections::HashMap;
    ///
    /// let nanos = std::time::SystemTime::now()
    ///     .duration_since(std::time::UNIX_EPOCH)
    ///     .map(|d| d.as_nanos())
    ///     .unwrap_or(0);
    /// let root = std::env::temp_dir().join(format!(
    ///     "fandhe-ai-model-save-doc-{}-{nanos}",
    ///     std::process::id()
    /// ));
    /// let registry = ModelRegistry::with_cache_dir(root.clone());
    /// let mut state = HashMap::new();
    /// state.insert("w".to_string(), Tensor::new(vec![1.0_f32, 2.0], &[2])?);
    /// match registry.save("mlp", "v1", &state) {
    ///     Ok(()) => {
    ///         let loaded = registry.load("mlp", "v1")?;
    ///         assert_eq!(loaded.len(), 1);
    ///         assert_eq!(
    ///             registry.available_models(),
    ///             vec![("mlp".to_string(), vec!["v1".to_string()])]
    ///         );
    ///     }
    ///     Err(ModelError::Io(e)) if e.kind() == std::io::ErrorKind::Unsupported => {}
    ///     Err(e) => return Err(e.into()),
    /// }
    /// let _ = std::fs::remove_dir_all(&root);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - [`ModelError::InvalidComponent`] — `name`・`version` が許可文字集合外
    /// - [`ModelError::EmptyModel`] — `tensors` が空
    /// - [`ModelError::TooLarge`] — 概算またはエンコード後のサイズが上限（1 GiB）超過
    /// - [`ModelError::Save`] — safetensors エンコード失敗
    /// - [`ModelError::Load`] — 読み戻し時のデコード失敗
    /// - [`ModelError::Io`] — 上記以外。読み戻しの不一致・確定後の実体不一致は
    ///   `InvalidData`、中間段がシンボリックリンクまたは非ディレクトリなら
    ///   `InvalidInput`、非対応プラットフォームは `Unsupported`。メッセージに
    ///   パスは含めない
    pub fn save(
        &self,
        name: &str,
        version: &str,
        tensors: &HashMap<String, Tensor<f32>>,
    ) -> Result<(), ModelError> {
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        {
            self.save_with(
                name,
                version,
                tensors,
                MAX_MODEL_FILE_BYTES,
                production_tmp_name,
                |_: &Path| Ok(()),
            )
        }
        #[cfg(not(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )))]
        {
            let _ = (name, version, tensors);
            Err(ModelError::Io(std::io::Error::from(
                std::io::ErrorKind::Unsupported,
            )))
        }
    }

    /// [`Self::save`] の本体。サイズ上限・一時名生成器・確定直前フックを
    /// 引数化し、単体テストが 1 GiB 級のデータなしに上限超過・名前衝突・
    /// 途中失敗を再現できるようにする（`compat::model_io` の
    /// `write_prepared_with` と同じ差し込み方）。本番は `save` から
    /// `MAX_MODEL_FILE_BYTES`・`production_tmp_name`・何もしないフックで呼ぶ。
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn save_with(
        &self,
        name: &str,
        version: &str,
        tensors: &HashMap<String, Tensor<f32>>,
        max: u64,
        mut next_tmp: impl FnMut() -> String,
        before_commit: impl FnOnce(&Path) -> std::io::Result<()>,
    ) -> Result<(), ModelError> {
        use crate::fs_write::{MAX_TMP_NAME_ATTEMPTS, create_new_with_retry, remove_own_tmp};
        use crate::interop::safetensors::save_safetensors_f32_to_bytes;
        use std::io::{Error as IoError, ErrorKind, Write};
        use std::os::unix::fs::MetadataExt;

        validate_component("name", name)?;
        validate_component("version", version)?;
        if tensors.is_empty() {
            return Err(ModelError::EmptyModel {
                name: name.to_string(),
                version: version.to_string(),
            });
        }

        // エンコード前の概算（F32 データ部のみ）。巨大入力のエンコードで
        // メモリを確保する前に拒否する。overflow は上限超過として扱う。
        let mut estimate: u64 = 0;
        for t in tensors.values() {
            match (t.numel() as u64)
                .checked_mul(4)
                .and_then(|b| estimate.checked_add(b))
            {
                Some(v) => estimate = v,
                None => {
                    return Err(ModelError::TooLarge {
                        name: name.to_string(),
                        version: version.to_string(),
                        len: u64::MAX,
                        max,
                    });
                }
            }
        }
        enforce_size_limit(name, version, estimate, max)?;

        let bytes = save_safetensors_f32_to_bytes(tensors, None).map_err(ModelError::Save)?;
        enforce_size_limit(name, version, bytes.len() as u64, max)?;

        // 読み戻し検証: 保存できたのに `load` で読めないファイルを作らない。
        {
            let back = load_safetensors_f32_from_bytes(&bytes).map_err(ModelError::Load)?;
            let consistent = back.len() == tensors.len()
                && tensors
                    .iter()
                    .all(|(k, t)| back.get(k).is_some_and(|b| b.shape() == t.shape()));
            if !consistent {
                return Err(ModelError::Io(IoError::new(
                    ErrorKind::InvalidData,
                    "エンコード結果の読み戻しがキー集合または shape と一致しません",
                )));
            }
        }

        // ここからファイルシステムに触れる。ルート自体がシンボリックリンク
        // でも許容する（利用者が選んだ配置。`load` 側と同じ扱い）。
        ensure_proc_fd_usable(Path::new("/proc/self/fd")).map_err(ModelError::Io)?;
        std::fs::create_dir_all(&self.root).map_err(ModelError::Io)?;
        // ディレクトリ fd を 1 段ずつ固定し、以後の一時ファイル作成・`rename` は
        // `/proc/self/fd/<fd>` 配下だけで行う（パス文字列の再解決をしない。
        // codex-review 指摘・PR #3377 P0。差し替えられても書き込みはルート外へ出ない）。
        let root_dir = open_root_dir(&self.root).map_err(ModelError::Io)?;
        let name_fd = open_dir_component(&root_dir, name).map_err(ModelError::Io)?;
        let version_fd = open_dir_component(&name_fd, version).map_err(ModelError::Io)?;
        let version_dir = proc_fd_path(&version_fd);

        let (mut tmp_file, tmp_name) =
            create_new_with_retry(&version_dir, &mut next_tmp, MAX_TMP_NAME_ATTEMPTS)
                .map_err(ModelError::Io)?;
        let tmp_path = version_dir.join(&tmp_name);
        let leaf = version_dir.join(MODEL_FILE_NAME);

        let committed: std::io::Result<()> = (|| {
            tmp_file.write_all(&bytes)?;
            // 確定直前の再確認: 一時ファイルが保持ハンドルと同一の実体であること。
            let held = tmp_file.metadata()?;
            let on_disk = std::fs::symlink_metadata(&tmp_path)?;
            if !on_disk.file_type().is_file()
                || on_disk.dev() != held.dev()
                || on_disk.ino() != held.ino()
            {
                return Err(IoError::new(
                    ErrorKind::InvalidData,
                    "一時ファイルが書き込み中に差し替えられました",
                ));
            }
            before_commit(&tmp_path)?;
            std::fs::rename(&tmp_path, &leaf)
        })();
        if let Err(e) = committed {
            // 削除してよいのはこの呼び出しが作って保持している一時ファイルだけ。
            remove_own_tmp(&tmp_file, &tmp_path);
            return Err(ModelError::Io(e));
        }

        // 確定後の検証（`rename` 後なので `model.safetensors` は消さない）。
        let held = tmp_file.metadata().map_err(ModelError::Io)?;
        let committed_file = match self.resolve_model_file_with_limit(name, version, max) {
            Ok(f) => f,
            Err(ModelError::NotFound { .. }) => {
                return Err(ModelError::Io(IoError::new(
                    ErrorKind::InvalidData,
                    "保存直後のモデルファイルを解決できません",
                )));
            }
            Err(e) => return Err(e),
        };
        let now = committed_file.metadata().map_err(ModelError::Io)?;
        if now.dev() != held.dev() || now.ino() != held.ino() {
            return Err(ModelError::Io(IoError::new(
                ErrorKind::InvalidData,
                "確定したファイルが書き込んだ実体と一致しません",
            )));
        }
        Ok(())
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

    /// 確保失敗（`TryReserveError`）が `ModelError::Io` の
    /// `ErrorKind::OutOfMemory` へヒープ確保なしで写ることを検証する
    /// （#2250）。`usize::MAX` は本番の
    /// `usize::try_from(..).unwrap_or(usize::MAX)` 由来の飽和値と同じ
    /// であり、合成的でない実経路の入力。`Vec<u8>` は要素サイズ 1
    /// バイトのため `usize::MAX` バイトの確保要求は `isize::MAX` を
    /// 超え、`try_reserve` はアロケータへ実際に触れる前に
    /// `TryReserveErrorKind::CapacityOverflow` を返す（32／64bit・
    /// 実メモリ量に依存しない決定的な失敗経路）。
    #[test]
    fn reserve_read_buffer_maps_alloc_failure_to_out_of_memory() {
        let err = reserve_read_buffer(usize::MAX).unwrap_err();
        match err {
            ModelError::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::OutOfMemory);
                // `io::Error::other` はペイロードを `Some(..)` で
                // 保持する（Custom 表現）。ここが `None` であることが
                // 「ヒープ確保なしで構築した」Simple 表現である証拠
                // （旧実装〈`io::Error::other`〉との弁別）。
                assert!(e.get_ref().is_none());
            }
            other => panic!("ModelError::Io を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn reserve_read_buffer_succeeds_for_small_capacity() {
        let bytes = reserve_read_buffer(16).expect("小さい確保要求は成功するべき");
        assert!(bytes.capacity() >= 16);
        assert_eq!(bytes.len(), 0);
    }

    #[test]
    fn enforce_size_limit_accepts_within_bound() {
        assert!(enforce_size_limit("mlp", "v1", 0, 100).is_ok());
        assert!(enforce_size_limit("mlp", "v1", 100, 100).is_ok());
    }

    #[test]
    fn enforce_size_limit_rejects_over_bound() {
        let err = enforce_size_limit("mlp", "v1", 101, 100).unwrap_err();
        match err {
            ModelError::TooLarge {
                name,
                version,
                len,
                max,
            } => {
                assert_eq!(name, "mlp");
                assert_eq!(version, "v1");
                assert_eq!(len, 101);
                assert_eq!(max, 100);
            }
            other => panic!("ModelError::TooLarge を期待したが {other:?} だった"),
        }
    }

    #[test]
    fn too_large_error_display_includes_sizes() {
        let err = enforce_size_limit("mlp", "v1", 101, 100).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("101"), "rendered={rendered:?}");
        assert!(rendered.contains("100"), "rendered={rendered:?}");
    }

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

    #[test]
    fn non_empty_os_string_filters_empty_value() {
        // 空文字列は「未設定」として扱う（Cursor Bugbot 指摘・PR #2226。
        // `HOME=""` をそのまま受理すると `PathBuf::from("")` がカレント
        // ディレクトリ相対のキャッシュルートを構築してしまう）。
        assert_eq!(non_empty_os_string(Some(OsString::from(""))), None);
    }

    #[test]
    fn non_empty_os_string_keeps_non_empty_value() {
        let value = Some(OsString::from("/home/x"));
        assert_eq!(non_empty_os_string(value.clone()), value);
    }

    #[test]
    fn non_empty_os_string_none_stays_none() {
        assert_eq!(non_empty_os_string(None), None);
    }

    #[test]
    fn select_home_var_none_when_both_empty_strings() {
        // 空文字列の HOME／USERPROFILE を有効値扱いすると
        // `default_cache_dir_from` が cwd 相対パスを構築してしまう
        // 回帰の直接検証（Cursor Bugbot 指摘・PR #2226）。フィルタは
        // `select_home_var` 内部で行われるため、ここで直接呼び出して
        // 検証する（`ModelRegistry::new` の内部ロジックを再現するの
        // ではなく、実際に呼ばれる関数そのものを検証する）。
        let empty = || Some(OsString::from(""));
        assert_eq!(select_home_var(true, empty(), empty()), None);
        assert_eq!(select_home_var(false, empty(), empty()), None);
        assert!(default_cache_dir_from(select_home_var(true, empty(), empty())).is_none());
    }

    #[test]
    fn select_home_var_falls_back_when_preferred_is_empty_string() {
        // 空文字列は「未設定」として扱われ、もう一方の値へフォールバック
        // する（`or` の優先順位ロジック自体は不変）。
        let value = Some(OsString::from("/x"));
        assert_eq!(
            select_home_var(false, Some(OsString::from("")), value.clone()),
            value
        );
        assert_eq!(
            select_home_var(true, value.clone(), Some(OsString::from(""))),
            value
        );
    }

    /// テスト用一時ディレクトリの RAII ガード（イシュー #2378・親 #2363）。
    /// `tests/common/temp_dir.rs::TempDirGuard` と同方式（一意名の `create_dir`
    /// 排他作成・事前削除なし・drop 時に作成分のみ削除）。`tests/common` は
    /// src から見えないため複製している。実際の 1 GiB ファイルを用意する
    /// 非現実的な手段は取らず、`resolve_model_file_with_limit`／
    /// `load_with_limit` の `max` 引数を小さく指定して上限超過を再現する。
    struct TempDirGuard {
        path: std::path::PathBuf,
    }

    impl TempDirGuard {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let pid = std::process::id();
            for _ in 0..64 {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
                let candidate = std::env::temp_dir().join(format!(
                    "fandhe-ai-model-registry-unit-{pid}-{nanos}-{seq}-{label}"
                ));
                match std::fs::create_dir(&candidate) {
                    Ok(()) => return Self { path: candidate },
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("一時ディレクトリを作成できない: {e}"),
                }
            }
            panic!("一意な一時ディレクトリ名を 64 回試行しても確保できない");
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_leaf(dir: &Path, contents: &[u8]) {
        let version_dir = dir.join("mlp").join("v1");
        std::fs::create_dir_all(&version_dir).unwrap();
        std::fs::write(version_dir.join(MODEL_FILE_NAME), contents).unwrap();
    }

    /// `resolve_model_file_with_limit`（モジュール doc「非信頼入力の
    /// 扱い」節手順 5・fstat 時点の上限検査）が実ファイル経由で上限
    /// 超過を fail-closed に拒否することを検証する（PR #2226
    /// codex-review P0 是正。`enforce_size_limit_rejects_over_bound` は
    /// 純関数の境界値のみを検証するため、実際の fstat 経路が同じ判定
    /// へ到達することは別途要検証）。`max` を引数化しているため、
    /// `MAX_MODEL_FILE_BYTES`〈1 GiB〉相当の実ファイルを用意せずに
    /// 9 バイトのファイル・上限 8 バイトで同じ経路を再現できる。
    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn resolve_model_file_with_limit_rejects_over_bound() {
        let dir = TempDirGuard::new("resolve-over-bound");
        write_leaf(dir.path(), b"123456789");

        let registry = ModelRegistry::with_cache_dir(dir.path());
        let err = registry
            .resolve_model_file_with_limit("mlp", "v1", 8)
            .unwrap_err();
        match err {
            ModelError::TooLarge {
                name,
                version,
                len,
                max,
            } => {
                assert_eq!(name, "mlp");
                assert_eq!(version, "v1");
                assert_eq!(len, 9);
                assert_eq!(max, 8);
            }
            other => panic!("ModelError::TooLarge を期待したが {other:?} だった"),
        }
    }

    /// `resolve_model_file_with_limit` は `len == max`（境界値）を
    /// 許可する（fstat 経路でも `enforce_size_limit_accepts_within_
    /// bound` と同じ境界を保つことの確認。安全側検証のため意図的に
    /// safetensors として不正な中身を使い、この後段の
    /// `load_safetensors_f32_from_bytes` 側のエラーとは切り分ける）。
    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn resolve_model_file_with_limit_accepts_at_bound() {
        let dir = TempDirGuard::new("resolve-at-bound");
        write_leaf(dir.path(), b"12345678");

        let registry = ModelRegistry::with_cache_dir(dir.path());
        assert!(
            registry
                .resolve_model_file_with_limit("mlp", "v1", 8)
                .is_ok()
        );
    }

    /// `load_with_limit`（[`ModelRegistry::load`] 本体）が `max` を
    /// 引数化した小さな実ファイルで上限超過を fail-closed に拒否する
    /// ことを、公開 `load` と同じ配線（`resolve_model_file_with_limit`
    /// → `Read::take` 二段構え）を通して検証する（モジュール doc
    /// 「非信頼入力の扱い」節手順 5〜6）。
    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn load_with_limit_rejects_over_bound() {
        let dir = TempDirGuard::new("load-over-bound");
        write_leaf(dir.path(), b"123456789");

        let registry = ModelRegistry::with_cache_dir(dir.path());
        let err = registry.load_with_limit("mlp", "v1", 8).unwrap_err();
        assert!(
            matches!(err, ModelError::TooLarge { len: 9, max: 8, .. }),
            "ModelError::TooLarge {{ len: 9, max: 8, .. }} を期待したが {err:?} だった"
        );
    }

    /// 非対応プラットフォーム（Windows 等）の fail-closed 契約の回帰検査
    /// （モジュール doc の「load は `ModelError::Io`」記述を固定する）。
    /// open がサイズ判定より先に拒否するため、上限超過でも `TooLarge` ではなく
    /// `Io(Unsupported)` になる。述語の正は `fs_guard::open_leaf_no_follow`。
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    )))]
    #[test]
    fn load_is_unsupported_on_non_supported_platform() {
        let dir = TempDirGuard::new("unsupported");
        write_leaf(dir.path(), b"12345678");
        let registry = ModelRegistry::with_cache_dir(dir.path());
        let err = registry
            .resolve_model_file_with_limit("mlp", "v1", 8)
            .unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
            "Io(Unsupported) を期待したが {err:?} だった"
        );
        write_leaf(dir.path(), b"123456789");
        let err = registry
            .resolve_model_file_with_limit("mlp", "v1", 8)
            .unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
            "TooLarge ではなく Io(Unsupported) を期待したが {err:?} だった"
        );
        let err = registry.load_with_limit("mlp", "v1", 8).unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
            "Io(Unsupported) を期待したが {err:?} だった"
        );
    }

    // ---- `save`（イシュー #3119）の単体テスト。結合テストは #3120。----

    fn sample_state() -> HashMap<String, Tensor<f32>> {
        let mut m = HashMap::new();
        m.insert(
            "fc.weight".to_string(),
            Tensor::new(
                vec![0.1_f32, -0.2, 0.3, -0.4, f32::MIN_POSITIVE, 1e30],
                &[2, 3],
            )
            .unwrap(),
        );
        m.insert(
            "fc.bias".to_string(),
            Tensor::new(vec![-0.0_f32, 7.5], &[2]).unwrap(),
        );
        m
    }

    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    fn bits(t: &Tensor<f32>) -> Vec<u32> {
        t.contiguous()
            .as_slice()
            .expect("contiguous は連続")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    fn tmp_leftovers(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".model.safetensors.tmp-"))
            .collect()
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_then_load_roundtrips_bit_exact() {
        let dir = TempDirGuard::new("save-roundtrip");
        let registry = ModelRegistry::with_cache_dir(dir.path().join("root"));
        let state = sample_state();
        registry.save("mlp", "v1", &state).unwrap();
        let back = registry.load("mlp", "v1").unwrap();
        assert_eq!(back.len(), state.len());
        for (k, t) in &state {
            let b = back.get(k).expect("キーが往復するはず");
            assert_eq!(b.shape(), t.shape());
            assert_eq!(bits(b), bits(t), "{k}");
        }
        assert!(tmp_leftovers(&dir.path().join("root/mlp/v1")).is_empty());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_rejects_invalid_components_before_fs_access() {
        let dir = TempDirGuard::new("save-invalid");
        let root = dir.path().join("root");
        let registry = ModelRegistry::with_cache_dir(&root);
        let state = sample_state();
        for bad in ["", ".", "..", ".hidden", "a/b", "a\\b", "a\0b", "a\nb"] {
            assert!(
                matches!(
                    registry.save(bad, "v1", &state),
                    Err(ModelError::InvalidComponent { kind: "name", .. })
                ),
                "name={bad:?}"
            );
            assert!(
                matches!(
                    registry.save("mlp", bad, &state),
                    Err(ModelError::InvalidComponent {
                        kind: "version",
                        ..
                    })
                ),
                "version={bad:?}"
            );
        }
        assert!(!root.exists(), "検証失敗でルートを作ってはならない");
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_rejects_empty_model_without_creating_root() {
        let dir = TempDirGuard::new("save-empty");
        let root = dir.path().join("root");
        let registry = ModelRegistry::with_cache_dir(&root);
        let err = registry.save("mlp", "v1", &HashMap::new()).unwrap_err();
        assert!(matches!(err, ModelError::EmptyModel { .. }), "{err:?}");
        assert!(!root.exists());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_with_limit_rejects_over_bound_without_creating_root() {
        let dir = TempDirGuard::new("save-limit");
        let root = dir.path().join("root");
        let registry = ModelRegistry::with_cache_dir(&root);
        let err = registry
            .save_with("mlp", "v1", &sample_state(), 8, production_tmp_name, |_| {
                Ok(())
            })
            .unwrap_err();
        assert!(
            matches!(err, ModelError::TooLarge { max: 8, .. }),
            "{err:?}"
        );
        assert!(!root.exists());
        // エンコード後サイズだけが超える境界: 概算（F32 データ部）は通る上限。
        let data_bytes = 8 * 4;
        let err = registry
            .save_with(
                "mlp",
                "v1",
                &sample_state(),
                data_bytes,
                production_tmp_name,
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(matches!(err, ModelError::TooLarge { .. }), "{err:?}");
        assert!(!root.exists());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_failure_before_commit_keeps_existing_and_leaves_no_tmp() {
        let dir = TempDirGuard::new("save-fail");
        let registry = ModelRegistry::with_cache_dir(dir.path().join("root"));
        let state = sample_state();
        registry.save("mlp", "v1", &state).unwrap();
        let mut other = HashMap::new();
        other.insert("x".to_string(), Tensor::new(vec![1.0_f32], &[1]).unwrap());
        let err = registry
            .save_with(
                "mlp",
                "v1",
                &other,
                MAX_MODEL_FILE_BYTES,
                production_tmp_name,
                |_| Err(std::io::Error::other("注入した失敗")),
            )
            .unwrap_err();
        assert!(matches!(err, ModelError::Io(_)), "{err:?}");
        let back = registry.load("mlp", "v1").unwrap();
        assert_eq!(back.len(), state.len());
        for (k, t) in &state {
            assert_eq!(bits(&back[k]), bits(t));
        }
        assert!(tmp_leftovers(&dir.path().join("root/mlp/v1")).is_empty());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_gives_up_after_tmp_name_collisions() {
        let dir = TempDirGuard::new("save-collide");
        let registry = ModelRegistry::with_cache_dir(dir.path().join("root"));
        let state = sample_state();
        registry.save("mlp", "v1", &state).unwrap();
        let vdir = dir.path().join("root/mlp/v1");
        std::fs::write(vdir.join(".collide"), b"keep").unwrap();
        let err = registry
            .save_with(
                "mlp",
                "v1",
                &state,
                MAX_MODEL_FILE_BYTES,
                || ".collide".to_string(),
                |_| Ok(()),
            )
            .unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists),
            "{err:?}"
        );
        assert_eq!(std::fs::read(vdir.join(".collide")).unwrap(), b"keep");
        assert_eq!(registry.load("mlp", "v1").unwrap().len(), state.len());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_overwrites_existing_atomically() {
        let dir = TempDirGuard::new("save-overwrite");
        let registry = ModelRegistry::with_cache_dir(dir.path().join("root"));
        registry.save("mlp", "v1", &sample_state()).unwrap();
        let mut next = HashMap::new();
        next.insert(
            "w".to_string(),
            Tensor::new(vec![3.0_f32, 4.0], &[2]).unwrap(),
        );
        registry.save("mlp", "v1", &next).unwrap();
        let back = registry.load("mlp", "v1").unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(bits(&back["w"]), bits(&next["w"]));
        assert_eq!(
            registry.available_models(),
            vec![("mlp".to_string(), vec!["v1".to_string()])]
        );
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn proc_fd_precheck_rejects_missing_proc_and_accepts_real_proc() {
        let dir = TempDirGuard::new("save-noproc");
        assert!(ensure_proc_fd_usable(&dir.path().join("no-such-proc")).is_err());
        assert!(ensure_proc_fd_usable(Path::new("/proc/self/fd")).is_ok());
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_rejects_symlinked_name_dir_without_writing_outside() {
        let dir = TempDirGuard::new("save-symlink");
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("mlp")).unwrap();
        let registry = ModelRegistry::with_cache_dir(&root);
        let err = registry.save("mlp", "v1", &sample_state()).unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::InvalidInput),
            "{err:?}"
        );
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
    }

    /// 検査後（一時ファイル書き込み後・`rename` 前）に `<version>` を退避して
    /// シンボリックリンクへ差し替えても、ルート外へ書かれないこと（codex-review
    /// 指摘・PR #3377 P0。ディレクトリ fd 固定の回帰テスト）。
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_does_not_write_outside_when_version_dir_swapped_to_symlink() {
        let dir = TempDirGuard::new("save-swap");
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let registry = ModelRegistry::with_cache_dir(&root);
        let vdir = root.join("mlp").join("v1");
        let moved = root.join("mlp").join("v1-moved");
        let outside_c = outside.clone();
        let (vdir_c, moved_c) = (vdir.clone(), moved.clone());
        let result = registry.save_with(
            "mlp",
            "v1",
            &sample_state(),
            MAX_MODEL_FILE_BYTES,
            production_tmp_name,
            move |_| {
                std::fs::rename(&vdir_c, &moved_c)?;
                std::os::unix::fs::symlink(&outside_c, &vdir_c)
            },
        );
        assert!(result.is_err(), "差し替えを検出して Err のはず");
        assert_eq!(
            std::fs::read_dir(&outside).unwrap().count(),
            0,
            "ルート外へ書き込んではならない"
        );
    }

    /// 同上。`<name>` 段を差し替えるケース。
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    #[test]
    fn save_does_not_write_outside_when_name_dir_swapped_to_symlink() {
        let dir = TempDirGuard::new("save-swap-name");
        let root = dir.path().join("root");
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(outside.join("v1")).unwrap();
        let registry = ModelRegistry::with_cache_dir(&root);
        let ndir = root.join("mlp");
        let moved = root.join("mlp-moved");
        let (outside_c, ndir_c, moved_c) = (outside.clone(), ndir.clone(), moved.clone());
        let result = registry.save_with(
            "mlp",
            "v1",
            &sample_state(),
            MAX_MODEL_FILE_BYTES,
            production_tmp_name,
            move |_| {
                std::fs::rename(&ndir_c, &moved_c)?;
                std::os::unix::fs::symlink(&outside_c, &ndir_c)
            },
        );
        assert!(result.is_err(), "差し替えを検出して Err のはず");
        assert_eq!(std::fs::read_dir(outside.join("v1")).unwrap().count(), 0);
    }

    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    #[test]
    fn save_is_unsupported_on_non_supported_platform() {
        let registry = ModelRegistry::with_cache_dir("unused-root-for-unsupported-save");
        let err = registry.save("mlp", "v1", &sample_state()).unwrap_err();
        assert!(
            matches!(&err, ModelError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
            "{err:?}"
        );
    }

    #[test]
    fn new_error_variants_display_without_path() {
        let e = ModelError::EmptyModel {
            name: "mlp".into(),
            version: "v1".into(),
        };
        let text = format!("{e} {e:?}");
        assert!(text.contains("mlp") && text.contains("v1"));
        assert!(!text.contains('/'));
        use std::error::Error as _;
        assert!(e.source().is_none());
    }
}
