//! facade 内部で非信頼なファイルシステム入力を読むための共有ハードニング
//! ヘルパー（no-follow・非ブロッキングの葉オープンとファイルサイズ上限）。
//! 公開面ではなく `pub(crate)` のみを持つ（`lib.rs` で素の `mod` 宣言）。
//!
//! 呼び出し元: `crate::model`（`ModelRegistry::resolve_model_file`／`load`）と、
//! `compat::model_io`（#2369。読み取り側は `open_leaf_checked` で同じ手順を使い、
//! 独自の `O_NOFOLLOW` 定数を持たない）。`model.rs` の読み取り手順を本ヘルパーへ
//! 寄せる整理は将来の課題（挙動を変えない純粋な移設として別途行う）。
//!
//! cfg の前提: Linux の x86_64／aarch64 と macOS だけが実オープンを行い、
//! それ以外（Windows・他の Linux アーキ等）は `ErrorKind::Unsupported` で
//! fail-closed にする。Linux では `O_NOFOLLOW` の値がアーキ間で異なる。
//! `libc` は許容依存第 10 区分で `onnx-interop` の external data 用途限定の
//! ため facade は直接依存に持たず、生 flag 値を `mod open_flags` に埋め込む
//! （`.claude/rules/deps-policy.md`）。
//!
//! 出典: `docs/compat-model-io-decision.md` §13.2・§13.4、
//! `docs/facade-model-registry-decision.md`、イシュー #2364。
//! 対象外: Windows 実装（#2389〜#2392）。書き込み側（`create_new`・一時ファイルの
//! 所有権確認削除）は `compat::model_io` が std のみで持つ（このモジュールには置かない）。

use std::fs::File;
use std::path::Path;

/// `model.safetensors` を読み込む際のファイルサイズ上限（バイト数）。
/// 1 GiB。`model.safetensors` は非信頼な外部フォーマット入力であり、
/// 上限なしに `read_to_end` で丸ごと `Vec` へ確保するとメモリ枯渇を
/// 招く（codex-review 指摘・PR #2226・P0）。
///
/// # 値の導出根拠（8 GiB からの引き下げ。codex-review 再指摘・P0）
///
/// 当初 8 GiB としていたが、一般的な実行環境（GitHub ホステッド
/// runner の既定 7 GiB RAM・開発者のノート PC 等）では単一ファイル
/// の 8 GiB 確保自体がその環境の RAM 総量に匹敵・超過し、実質的な
/// OOM 防止になっていなかった。加えて
/// [`crate::interop::safetensors::load_safetensors_f32_from_bytes`] は読み込んだバイト列
/// （`Vec<u8>`。本ファイルサイズ相当）と、デコード後の
/// `Tensor<f32>` 群（safetensors の F32 データ部とほぼ同サイズ）を
/// `bytes` が drop されるまで同時に保持するため、ピークメモリは
/// おおよそ**ファイルサイズの 2 倍**になる（`crate::interop::
/// safetensors::load_safetensors_f32_from_bytes` 実装参照）。
///
/// 想定する最低限のホスト RAM を 4 GiB、単一モデルロードに許容する
/// 割合をその半分（2 GiB）とし、上記ピーク倍率 2 で割った
/// `2 GiB ÷ 2 = 1 GiB` をファイルサイズ上限とした。本レジストリが
/// 対象とする F32 のみの `compat::Sequential` 向けローカル重み
/// （数百 MB 級を主に想定。1 GiB 超のモデルは対象外）に対しては
/// 引き続き十分な余裕を持ち、攻撃者が用意した巨大ファイルによる
/// 無制限確保を防ぐ（詳細は `docs/facade-model-registry-decision.md`
/// 参照）。将来より大きなモデルを扱う必要が生じた場合の値の見直しは
/// 別途ユーザー承認を経る。
pub(crate) const MAX_MODEL_FILE_BYTES: u64 = 1024 * 1024 * 1024;

/// `open_leaf_no_follow` が付与する生の `open(2)` flag 値（Linux
/// （x86_64／aarch64 限定。下記 `O_NOFOLLOW` 節参照）／macOS のみ。
/// `facade` は `libc` を直接依存に持たない（許容依存第 10 区分の `libc` は
/// `onnx-interop` の external data 用途限定。`.claude/rules/deps-policy.md`）
/// ため、カーネル UAPI ヘッダ由来の固定値を直接埋め込む。`std::os::unix::fs::OpenOptionsExt::
/// custom_flags` はこの生値をそのまま `open` システムコールへ渡す）。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) mod open_flags {
    /// `include/uapi/asm-generic/fcntl.h`。x86_64・aarch64 Linux で
    /// 共通の値（alpha／parisc／sparc／mips 等の非対応アーキテクチャは
    /// 本 cfg のガードにより到達しない。`crates/self-repair/src/
    /// fd_walk.rs` と同方針）。
    pub(crate) const O_NONBLOCK: i32 = 0o4_000;
    /// `O_NOFOLLOW` は Linux ではアーキ間で値が異なる（`O_DIRECTORY`
    /// と同様。`crates/self-repair/src/fd_walk.rs` の `mod raw` 参照）。
    /// asm-generic（x86_64 が使う値）をそのまま aarch64 にも共用すると
    /// aarch64 の実際の `O_NOFOLLOW`（`arch/arm64/include/uapi/asm/
    /// fcntl.h` 由来）ではなく `O_LARGEFILE`（no-op）に化けてしまい、
    /// `open_leaf_no_follow` が aarch64 Linux でシンボリックリンクを
    /// 黙って追跡してしまう（Cursor Bugbot 指摘・PR #2226。過去に
    /// `nvrtc.rs`・`fd_walk.rs` で修正済みの同種不具合）。
    /// 出典（裏付け）: `libc` crate v0.2.174 の
    /// `src/unix/linux_like/linux/gnu/b64/aarch64/mod.rs` は
    /// `O_NOFOLLOW = 0x8000`（=0o100000）を、
    /// `b64/x86_64/mod.rs` は `O_NOFOLLOW = 0x20000`（=0o400000）を
    /// それぞれ定義する。
    #[cfg(target_arch = "x86_64")]
    pub(crate) const O_NOFOLLOW: i32 = 0o400_000;
    #[cfg(target_arch = "aarch64")]
    pub(crate) const O_NOFOLLOW: i32 = 0o100_000;
    /// `include/uapi/asm-generic/errno.h`。`O_NOFOLLOW` がシンボリック
    /// リンクを検出した際に `open(2)` が返す errno（ELOOP）。値自体は
    /// x86_64・aarch64 Linux で共通（asm-generic errno はアーキ別
    /// 再定義対象に含まれない）。
    /// `std::io::ErrorKind::FilesystemLoop` は本リポジトリの pin toolchain
    /// （`rust-toolchain.toml`）でも `#![feature(io_error_more)]` 相当の
    /// unstable のため使えず、`raw_os_error()` の生値で判定する。
    pub(crate) const ELOOP: i32 = 40;
}
#[cfg(target_os = "macos")]
pub(crate) mod open_flags {
    /// `<sys/fcntl.h>`（Darwin／macOS）。
    pub(crate) const O_NONBLOCK: i32 = 0x0004;
    pub(crate) const O_NOFOLLOW: i32 = 0x0100;
    /// `<sys/errno.h>`（Darwin／macOS）の ELOOP。上記 Linux 側コメント参照。
    pub(crate) const ELOOP: i32 = 62;
}

/// 葉ファイル（`model.safetensors`）をシンボリックリンク追跡なし・
/// 非ブロッキングで開く（`crate::model` モジュール doc「非信頼入力の扱い」節の手順
/// 3 参照）。Linux／macOS は `O_NOFOLLOW`（最終コンポーネントの
/// シンボリックリンクを拒否）・`O_NONBLOCK`（FIFO への差し替えによる
/// 無期限ブロックを防ぐ。通常ファイルの読み取りには影響しない）を
/// 付与する。
///
/// それ以外の OS（Windows 等）向けの安全な no-follow 実装は未導入
/// （codex-review 指摘・PR #2226。`facade` は `libc`（許容依存第 10 区分は
/// `onnx-interop` の external data 用途限定）／`windows-sys` を直接依存に
/// 持たないため、`FILE_FLAG_OPEN_REPARSE_POINT`
/// 相当の生 flag 値を安全に組み立てる手段が確立していない）。
/// フォールバックとして無防備な `File::open` を使うと、呼び出し元の
/// 識別子一致検査（`dev`／`ino`）が `#[cfg(unix)]` 限定で Windows では
/// 実施されないため、検査〜open 間の差し替え（TOCTOU）をキャッシュ
/// ルート外のファイル読み取りへ悪用できてしまう。代わりに
/// **fail-closed で `ErrorKind::Unsupported` を返しオープン自体を
/// 拒否**する（呼び出し元の `model.rs` が型付きエラー `ModelError::Io` へ丸める）。安全な
/// Windows 実装（`file_index`／`volume_serial_number` によるハンドル
/// 識別子照合を含む）の追加はスコープ外として別イシューで追跡する。
pub(crate) fn open_leaf_no_follow(leaf: &Path) -> std::io::Result<File> {
    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(open_flags::O_NOFOLLOW | open_flags::O_NONBLOCK)
            .open(leaf)
    }
    #[cfg(not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    )))]
    {
        let _ = leaf;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "このプラットフォームでは葉ファイルのシンボリックリンク追跡なし \
             オープンを実装していないため、TOCTOU 対策として fail-closed で \
             オープンを拒否します（Linux／macOS のみ対応）",
        ))
    }
}
/// [`open_leaf_checked`]／[`OpenedLeaf::read_exact_len`] の失敗種別。
/// 呼び出し元（`compat::model_io`）が型付きエラー（`Io`／`TooLarge`）へ写す。
#[derive(Debug)]
pub(crate) enum LeafError {
    /// OS 由来の I/O 失敗、または no-follow 手順の拒否（シンボリックリンク・
    /// 非通常ファイル・実体識別子の不一致は `ErrorKind::InvalidInput`、読み取り中の
    /// 増大・短縮は `ErrorKind::InvalidData`、非対応プラットフォームは
    /// `ErrorKind::Unsupported`）。`O_NOFOLLOW` の ELOOP は元の OS エラーのまま。
    Io(std::io::Error),
    /// `fstat` の実長が固定上限 `limit` を超えている（読み取りには入らない）。
    TooLarge { limit: u64 },
}

/// no-follow 手順（決定記録 §13.2 手順 1〜4）を通過した葉ファイルのハンドル。
/// 実長は `fstat` 由来（固定上限以下と確認済み）で、呼び出し元は読む前に
/// この値と非信頼値（manifest の `safetensors_bytes` 等）との**一致**だけを
/// 確認できる（上限判定に非信頼値を使わない。§13.2 手順 4）。
pub(crate) struct OpenedLeaf {
    file: File,
    len: u64,
}

impl OpenedLeaf {
    /// `fstat` 由来の実長（固定上限以下と確認済みの信頼できる値）。
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// 同じハンドルから `take(実長 + 1)` で読み、読めた長さが実長とちょうど
    /// 一致することを確認する（§13.2 手順 5。読み取り中の増大・短縮の検出）。
    /// 確保は `try_reserve_exact` で行い、失敗は panic ではなく `Io(OutOfMemory)`。
    pub(crate) fn read_exact_len(mut self) -> Result<Vec<u8>, LeafError> {
        use std::io::Read;
        let expected = usize::try_from(self.len).map_err(|_| {
            LeafError::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "ファイル実長が usize に収まりません",
            ))
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(expected.saturating_add(1))
            .map_err(|_| {
                LeafError::Io(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "読み取りバッファを確保できません",
                ))
            })?;
        self.file
            .by_ref()
            .take(self.len.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(LeafError::Io)?;
        if bytes.len() as u64 != self.len {
            return Err(LeafError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "読み取り中にファイルサイズが変化しました（fstat 実長と不一致）",
            )));
        }
        Ok(bytes)
    }
}

/// 葉ファイルを no-follow 手順で開き、実長を固定上限 `max` と比べる
/// （決定記録 §13.2 手順 1〜4。`compat::model_io::load_model` が
/// `manifest.json`・`model.<gen>.safetensors` の双方で使う）。
///
/// 1. `symlink_metadata` でシンボリックリンク・非通常ファイルを拒否
/// 2. [`open_leaf_no_follow`]（非対応プラットフォームは `Unsupported`）
/// 3. `fstat` で通常ファイルを再確認し、unix では手順 1 と `(dev, ino)` を照合
/// 4. `fstat` 実長を `max` と比較し、超過なら読まずに [`LeafError::TooLarge`]
pub(crate) fn open_leaf_checked(leaf: &Path, max: u64) -> Result<OpenedLeaf, LeafError> {
    open_leaf_checked_with(leaf, max, || {})
}

/// [`open_leaf_checked`] の本体。`after_check` は手順 1（`symlink_metadata` 検査）の直後・
/// 手順 2（open）の直前に呼ばれ、単体テストが検査〜open 間の差し替え（TOCTOU。
/// 決定記録 §13.3・イシュー #2376）を再現するための注入点。本番は no-op を渡す。
fn open_leaf_checked_with(
    leaf: &Path,
    max: u64,
    after_check: impl FnOnce(),
) -> Result<OpenedLeaf, LeafError> {
    let invalid = |msg: &'static str| {
        LeafError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, msg))
    };
    let leaf_meta = std::fs::symlink_metadata(leaf).map_err(LeafError::Io)?;
    if leaf_meta.file_type().is_symlink() || !leaf_meta.is_file() {
        return Err(invalid(
            "シンボリックリンクまたは通常ファイルでないため読み取りを拒否しました",
        ));
    }
    after_check();
    let file = open_leaf_no_follow(leaf).map_err(LeafError::Io)?;
    let open_meta = file.metadata().map_err(LeafError::Io)?;
    if !open_meta.is_file() {
        return Err(invalid("開いたハンドルが通常ファイルではありません"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if open_meta.dev() != leaf_meta.dev() || open_meta.ino() != leaf_meta.ino() {
            return Err(invalid(
                "検査後にファイルが差し替えられました（dev/ino 不一致）",
            ));
        }
    }
    let len = open_meta.len();
    if len > max {
        return Err(LeafError::TooLarge { limit: max });
    }
    Ok(OpenedLeaf { file, len })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fandhe-ai-fs-guard-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir(&dir).expect("一時ディレクトリを作成できるはず");
        dir
    }

    #[test]
    fn open_leaf_checked_reads_regular_file_exactly() {
        let dir = temp_dir("ok");
        let path = dir.join("a.bin");
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(b"hello"))
            .expect("書き込めるはず");
        let opened = open_leaf_checked(&path, 16).expect("開けるはず");
        assert_eq!(opened.len(), 5);
        assert_eq!(opened.read_exact_len().expect("読めるはず"), b"hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_leaf_checked_rejects_over_bound_without_reading() {
        let dir = temp_dir("big");
        let path = dir.join("a.bin");
        std::fs::write(&path, vec![0u8; 17]).expect("書き込めるはず");
        assert!(matches!(
            open_leaf_checked(&path, 16),
            Err(LeafError::TooLarge { limit: 16 })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn open_leaf_checked_rejects_symlink_and_directory() {
        let dir = temp_dir("sym");
        let target = dir.join("target.bin");
        std::fs::write(&target, b"x").expect("書き込めるはず");
        let link = dir.join("link.bin");
        std::os::unix::fs::symlink(&target, &link).expect("symlink を作れるはず");
        assert!(matches!(
            open_leaf_checked(&link, 16),
            Err(LeafError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput
        ));
        assert!(matches!(
            open_leaf_checked(&dir, 16),
            Err(LeafError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn read_exact_len_detects_growth_after_open() {
        let dir = temp_dir("grow");
        let path = dir.join("a.bin");
        std::fs::write(&path, b"abcd").expect("書き込めるはず");
        let opened = open_leaf_checked(&path, 16).expect("開けるはず");
        // open 後（fstat 後）に追記して読み取り中の増大を再現する。
        let mut appender = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("追記できるはず");
        appender.write_all(b"efgh").expect("追記できるはず");
        assert!(matches!(
            opened.read_exact_len(),
            Err(LeafError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidData
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 検査〜open 間の差し替え（TOCTOU）の注入テスト。`symlink_metadata` の事前拒否を
    /// 通過させた後に差し替えるため、`O_NOFOLLOW`（ELOOP）と `(dev, ino)` 照合の分岐へ届く。
    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    fn swapped_leaf_setup(label: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = temp_dir(label);
        let path = dir.join("leaf.bin");
        std::fs::write(&path, b"original").expect("書き込めるはず");
        (dir, path)
    }

    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn open_leaf_checked_rejects_symlink_swapped_in_after_check() {
        let (dir, path) = swapped_leaf_setup("swap-sym");
        let outside = dir.join("outside.bin");
        std::fs::write(&outside, b"original").expect("書き込めるはず");
        let result = open_leaf_checked_with(&path, 16, || {
            std::fs::remove_file(&path).expect("消せるはず");
            std::os::unix::fs::symlink(&outside, &path).expect("symlink を作れるはず");
        });
        assert!(matches!(
            result,
            Err(LeafError::Io(e)) if e.raw_os_error() == Some(open_flags::ELOOP)
        ));
        assert_eq!(std::fs::read(&outside).expect("読めるはず"), b"original");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn open_leaf_checked_rejects_regular_file_swapped_in_after_check() {
        let (dir, path) = swapped_leaf_setup("swap-reg");
        // 検査時点の inode を保持したまま別 inode の通常ファイルへ差し替える
        // （解放済み inode の再利用による偶然の一致を避ける）。
        let keep = dir.join("keep.bin");
        let result = open_leaf_checked_with(&path, 16, || {
            std::fs::rename(&path, &keep).expect("退避できるはず");
            std::fs::write(&path, b"replaced").expect("書き込めるはず");
        });
        assert!(matches!(
            result,
            Err(LeafError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    #[test]
    fn open_leaf_checked_rejects_fifo_swapped_in_after_check_without_hanging() {
        let (dir, path) = swapped_leaf_setup("swap-fifo");
        let worker_path = path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        // O_NONBLOCK が効かない退行でも、テストがハングせず失敗するよう別スレッドで有界化する。
        std::thread::spawn(move || {
            let result = open_leaf_checked_with(&worker_path, 16, || {
                std::fs::remove_file(&worker_path).expect("消せるはず");
                let status = std::process::Command::new("mkfifo")
                    .arg(&worker_path)
                    .status()
                    .expect("mkfifo を起動できるはず");
                assert!(status.success(), "mkfifo が成功するはず");
            });
            let _ = tx.send(result.map(|_| ()));
        });
        let result = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("FIFO でもハングせず結果を返すはず");
        assert!(matches!(
            result,
            Err(LeafError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
