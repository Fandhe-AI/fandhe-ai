//! facade 内部で非信頼なファイルシステム入力を読むための共有ハードニング
//! ヘルパー（no-follow・非ブロッキングの葉オープンとファイルサイズ上限）。
//! 公開面ではなく `pub(crate)` のみを持つ（`lib.rs` で素の `mod` 宣言）。
//!
//! 呼び出し元: `crate::model`（`ModelRegistry::resolve_model_file`／`load`）と、
//! 将来の `compat::model_io`（親 #2362 配下。読み取り側で同じヘルパーを使い、
//! 独自の `O_NOFOLLOW` 定数を持たない方針）。
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
//! 対象外: 書き込み側ヘルパー（#2369）・Windows 実装（#2389〜#2392）。

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
