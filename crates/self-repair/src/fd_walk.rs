//! パスコンポーネントを `openat`（`O_NOFOLLOW`）で 1 段ずつ辿り、検証済み
//! fd に対して直接読み書きするユーティリティ（PR #361 codex-review 第 4 波
//! P0 指摘: [`crate::candidate`] の途中ディレクトリ symlink 差し替え TOCTOU
//! の構造的排除）。
//!
//! # 旧実装（`reject_symlink_escape` + `fs::write`/`fs::read_to_string`）の限界
//! 旧実装は「検査」（`fs::symlink_metadata` によるコンポーネント逐次検査）と
//! 「利用」（パス文字列を再解決する `fs::write`/`fs::OpenOptions::open`）が
//! 別々の syscall 列に分かれていた。両者の間（TOCTOU window）に途中
//! ディレクトリが symlink へ差し替えられると、「利用」側がその symlink を
//! 追跡してしまう余地が残った（末尾コンポーネントは `O_NOFOLLOW` 付き open で
//! 塞いでいたが、途中ディレクトリは字句検査後の `stat` に依存していた）。
//!
//! # 本モジュールの設計
//! `workspace` を dir-fd として開き、そこを起点に相対パスの各コンポーネントを
//! `openat(dirfd, name, O_NOFOLLOW | O_DIRECTORY, ...)` で 1 段ずつ辿る。
//! 各コンポーネントの「symlink でないことの検証」と「そのディレクトリを開いて
//! 次段の起点にすること」が単一の syscall に統合されるため、検査後に別の
//! syscall で再解決する隙（TOCTOU window）が構造的に存在しない。パス文字列
//! ベースの API（`fs::write`・`fs::read_to_string`・`Path::join` を経由した
//! 全体パス解決）は一切使わない。
//!
//! # プラットフォーム
//! Linux（`x86_64`/`aarch64` に限定）・macOS のみ対応する。
//! `openat`/`O_NOFOLLOW`/`O_DIRECTORY` 等のフラグ値は各 OS の `fcntl.h` を
//! 出典として本ファイル内にローカル複製する（`libc` 依存は禁止。
//! deps-policy.md 許容依存 8 区分外）。self-repair CLI の実行環境は
//! self-hosted の Linux（x86_64/aarch64）/macOS runner のみであり、
//! それ以外のプラットフォーム・アーキテクチャでは値の出典を保証できないため、
//! 限定的なフォールバック（旧: 一時ファイル + rename 方式）は用意せず
//! `io::Error` で fail-closed に候補適用そのものを拒否する（ユーザー指示。
//! 安全側に倒す。下記 `unsupported_platform_error` 参照）。
//!
//! Linux 側は `target_os = "linux"` だけでなく `target_arch` も
//! `x86_64`/`aarch64` に限定する（PR #361 codex-review P0 指摘: `target_os =
//! "linux"` のみの判定は mips/parisc/sparc/alpha 等、`O_NOFOLLOW`/
//! `O_DIRECTORY`/`O_CLOEXEC` の値が `asm-generic/fcntl.h` と異なりうる
//! アーキテクチャも含んでしまい、それらの実行環境では誤った定数値で
//! `openat` を呼び出し symlink 追跡が意図せず有効化されうる〈fail-open〉。
//! self-repair CLI の実行対象は self-hosted runner の x86_64/aarch64 のみ。
//! `x86_64-linux-gnu` の `/usr/include/asm-generic/fcntl.h` を実機で確認し、
//! `O_DIRECTORY`（0200000）・`O_NOFOLLOW`（0400000）・`O_CLOEXEC`
//! （02000000）が下記 `raw` モジュールの定数と一致すること、および
//! `/usr/include/x86_64-linux-gnu/asm/fcntl.h` がアーキ固有の再定義なしに
//! `asm-generic/fcntl.h` を `#include` するだけであることを確認済み。
//!
//! **`aarch64` は asm-generic を継承しない**（PR #361 codex-review P0 指摘:
//! 2026-08-08 修正前は上記 x86_64（asm-generic）値を aarch64 にも共用しており、
//! 誤りだった）。Linux カーネルの `arch/arm64/include/uapi/asm/fcntl.h` は
//! 32-bit ARM 由来の値で `O_DIRECTORY`/`O_NOFOLLOW`/`O_DIRECT`/`O_LARGEFILE`
//! の 4 つを明示的に再定義してから `asm-generic/fcntl.h` を `#include` する
//! （出典: 上記ヘッダ。`O_DIRECTORY = 040000`・`O_NOFOLLOW = 0100000`・
//! `O_DIRECT = 0200000`・`O_LARGEFILE = 0400000`）。修正前の値
//! （asm-generic の `O_DIRECTORY = 0200000`・`O_NOFOLLOW = 0400000`）は
//! aarch64 の実際の `O_DIRECTORY`/`O_NOFOLLOW` の値ではなく、むしろ aarch64
//! 側の `O_DIRECT`/`O_LARGEFILE` の値と一致してしまっていた。すなわち
//! aarch64 上では `openat` に `O_DIRECT | O_LARGEFILE` が渡り、カーネルは
//! これを無視できるフラグ（少なくとも `O_NOFOLLOW`/`O_DIRECTORY` としては
//! 無効）として黙って受理するため、symlink 追跡防御が静かに無効化されていた
//! （fail-open）。`O_RDONLY`/`O_WRONLY`/`O_CREAT`/`O_TRUNC`/`O_CLOEXEC` は
//! `arch/arm64` 側の再定義対象に含まれておらず asm-generic の値のまま
//! （同ヘッダで再定義されているのは上記 4 つのみ）。この非対称性を反映し、
//! 下記 `raw` モジュールはアーキ間で共通の 5 定数（`O_RDONLY`/`O_WRONLY`/
//! `O_CREAT`/`O_TRUNC`/`O_CLOEXEC`）を `target_arch` cfg の外側に 1 箇所だけ
//! 定義し、実際にアーキ間で値が異なる `O_DIRECTORY`/`O_NOFOLLOW` の 2 定数
//! のみを `target_arch` ごとに個別定義する（同じ定数を x86_64/aarch64 双方に
//! 重複定義すると、x86_64 CI では aarch64 側の定義ミスがコンパイルされず
//! 検出できない。advisor 指摘）。
//!
//! それ以外のアーキテクチャは値を個別確認していないため対象に含めず、
//! 下記 `unsupported` サブモジュール側の fail-closed 経路へ送る。
//! 対応判定・実装・フォールバックの整合を単一箇所で保つため、Linux/macOS
//! 専用実装は本ファイル内の `supported` サブモジュール 1 つにまとめる
//! （下記 `cfg` 境界を参照）。

/// Linux（`x86_64`/`aarch64` に限定。モジュール冒頭 doc 参照）・macOS
/// 専用実装。`openat`/`O_NOFOLLOW` 等の値の出典が保証できるプラットフォーム・
/// アーキテクチャの組のみをこのモジュールの cfg 境界に含める。対応判定は
/// この 1 箇所にのみ書き、末尾の `unsupported` サブモジュール側は本 cfg の
/// 否定（`not(...)`）を使うことで実装・フォールバックの cfg 不一致を
/// 構造的に防ぐ。
#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
))]
mod supported {
    use std::ffi::CString;
    use std::fs;
    use std::io;
    use std::io::{Read as _, Write as _};
    use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};
    use std::os::raw::{c_char, c_int, c_uint};
    use std::os::unix::ffi::OsStrExt as _;
    use std::path::Component;
    use std::path::Path;

    /// 各 OS の `fcntl.h` 定義値のローカル複製（`libc` 依存禁止のため）。
    ///
    /// # Linux: アーキ共通 5 定数（`O_DIRECTORY`/`O_NOFOLLOW` を除く）
    /// 出典: Linux `include/uapi/asm-generic/fcntl.h`。`arch/arm64/include/
    /// uapi/asm/fcntl.h` はこの 5 定数を再定義しないため、本ファイル冒頭 doc の
    /// 対応アーキテクチャ〈x86_64/aarch64〉双方で asm-generic の値のまま
    /// 一致する（それ以外のアーキテクチャはこのモジュール自体がコンパイル
    /// されないため到達しない）。
    #[cfg(target_os = "linux")]
    mod raw {
        pub const O_RDONLY: i32 = 0o0;
        pub const O_WRONLY: i32 = 0o1;
        pub const O_CREAT: i32 = 0o100;
        pub const O_TRUNC: i32 = 0o1000;
        /// fork/exec で子プロセスへ fd を継承させない（`execve` 時に自動
        /// close）。security-auditor 監査 Low 指摘対応: `openat` で得た fd
        /// はいずれも self-repair プロセス内でのみ使う一時 fd であり、
        /// `SelfRepairLoop` が起動する `cargo build`/`test`/`clippy` 子
        /// プロセス（`verify_gates.rs`）へ意図せず継承されると fd リーク・
        /// 情報露出につながるため付与する。`O_CLOEXEC` も `arch/arm64` 側の
        /// 再定義対象に含まれない（本モジュール冒頭 doc 参照）。
        pub const O_CLOEXEC: i32 = 0o2000000;

        /// `O_DIRECTORY`/`O_NOFOLLOW` はアーキ間で値が異なる（本ファイル冒頭
        /// doc「`aarch64` は asm-generic を継承しない」参照）。出典: Linux
        /// `include/uapi/asm-generic/fcntl.h`（x86_64 が使う値）。
        #[cfg(target_arch = "x86_64")]
        pub const O_DIRECTORY: i32 = 0o200000;
        #[cfg(target_arch = "x86_64")]
        pub const O_NOFOLLOW: i32 = 0o400000;

        /// 出典: Linux `arch/arm64/include/uapi/asm/fcntl.h`（32-bit ARM 由来の
        /// 値。asm-generic の値〈0o200000/0o400000〉とは異なる。本ファイル
        /// 冒頭 doc 参照）。
        #[cfg(target_arch = "aarch64")]
        pub const O_DIRECTORY: i32 = 0o40000;
        #[cfg(target_arch = "aarch64")]
        pub const O_NOFOLLOW: i32 = 0o100000;
    }

    /// 出典: macOS `sys/fcntl.h`。
    #[cfg(target_os = "macos")]
    mod raw {
        pub const O_RDONLY: i32 = 0x0000;
        pub const O_WRONLY: i32 = 0x0001;
        pub const O_CREAT: i32 = 0x0200;
        pub const O_TRUNC: i32 = 0x0400;
        pub const O_DIRECTORY: i32 = 0x100000;
        pub const O_NOFOLLOW: i32 = 0x0100;
        /// 上記 Linux 側 `O_CLOEXEC` と同じ理由で付与する。
        pub const O_CLOEXEC: i32 = 0x1000000;
    }

    // `openat(2)` の FFI 宣言。C 側のシグネチャは
    // `int openat(int dirfd, const char *pathname, int flags, ...)` で
    // `mode_t` 引数（`O_CREAT` 使用時のみ意味を持つ）は可変長引数側にある。
    // 固定 4 引数として宣言すると呼び出し規約が可変長引数と異なる ABI
    // （例: `aarch64-apple-darwin` は可変長引数をスタック経由で渡す）で
    // `mode` が読めない不整合を起こすため、Rust 側も `...` 可変長引数として
    // 宣言し、`O_CREAT` を渡す呼び出しでのみ `mode` 実引数を追加する
    // （下記 openat_dir/openat_final 参照）。
    unsafe extern "C" {
        fn openat(dirfd: c_int, pathname: *const c_char, flags: c_int, ...) -> c_int;
    }

    /// 相対パスをコンポーネント単位の NUL 終端文字列列へ分解する。
    ///
    /// 呼び出し元は事前に [`crate::candidate::validate_relative_path`] を
    /// 経由させる契約のため、ここに到達するのは通常 `Normal`（1 コンポーネント
    /// あたり 1 段のディレクトリ／ファイル名）のみである。それ以外
    /// （`RootDir`・`Prefix`・`ParentDir`）が渡された場合は契約違反として
    /// fail-closed に拒否する（`openat` へ `..` をそのまま渡すと親ディレクトリへ
    /// 遡ってしまうため、ここでの防御は多重化の意味を持つ）。
    fn split_components(relative_path: &Path) -> io::Result<Vec<CString>> {
        let mut parts = Vec::new();
        for component in relative_path.components() {
            match component {
                Component::Normal(part) => {
                    let cstring = CString::new(part.as_bytes()).map_err(|_| {
                        io::Error::other(format!(
                            "候補修正のパスに NUL バイトを含むコンポーネントがあります: {}",
                            relative_path.display()
                        ))
                    })?;
                    parts.push(cstring);
                }
                Component::CurDir => continue,
                other => {
                    return Err(io::Error::other(format!(
                        "候補修正のパスに許可されないコンポーネント（{other:?}）が含まれます: {}",
                        relative_path.display()
                    )));
                }
            }
        }
        if parts.is_empty() {
            return Err(io::Error::other(format!(
                "候補修正のパスが空です: {}",
                relative_path.display()
            )));
        }
        Ok(parts)
    }

    /// `workspace` を dir-fd として開く（fd 走査チェーンの起点）。
    ///
    /// `workspace` 自体は sandbox.rs が管理する信頼済みルート（self-repair が
    /// 自ら作成した一時ディレクトリ）であり、途中ディレクトリの symlink
    /// 差し替え防御の対象は `workspace` からの相対パス側（[`walk_to_final`]）
    /// である。ルート自体は std の安全な API（`fs::OpenOptions`）で開けるため
    /// `unsafe` を使わない（`O_DIRECTORY` を `custom_flags` で付与し、
    /// ディレクトリでなければ open 自体を失敗させる）。
    fn open_root_dir(workspace: &Path) -> io::Result<OwnedFd> {
        use std::os::unix::fs::OpenOptionsExt as _;
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(raw::O_DIRECTORY)
            .open(workspace)?;
        Ok(OwnedFd::from(file))
    }

    /// 途中ディレクトリを 1 段開く（`O_NOFOLLOW | O_DIRECTORY`）。
    ///
    /// `name` が symlink または非ディレクトリであれば `openat` 自体が
    /// `ELOOP`/`ENOTDIR` で失敗する。「symlink でないことの検証」と
    /// 「次段の起点として開くこと」が単一の syscall に統合されるため、
    /// モジュール冒頭 doc が説明する TOCTOU window がここには存在しない。
    fn openat_dir(parent: RawFd, name: &CString) -> io::Result<OwnedFd> {
        // `O_CLOEXEC` を付与し、この dir-fd が子プロセス（cargo build/test/clippy
        // 等）へ継承されないようにする（security-auditor 監査 Low 指摘対応。
        // `raw::O_CLOEXEC` doc 参照）。
        let flags = raw::O_DIRECTORY | raw::O_NOFOLLOW | raw::O_RDONLY | raw::O_CLOEXEC;
        // SAFETY: FFI 境界（TOCTOU 排除のための openat 呼び出し。モジュール冒頭
        // doc 参照）。`parent` は呼び出し元（[`walk_to_final`]）が直前の
        // `open_root_dir`/`openat_dir` 呼び出しで取得した有効な dir-fd であり、
        // `name` は [`split_components`] が生成した NUL 終端済み `CString` から
        // 得た有効なポインタである。`O_CREAT` を含まない固定 3 引数呼び出しの
        // ため可変長引数側の `mode` は渡さない（[`openat`] doc 参照）。
        let fd = unsafe { openat(parent, name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `fd` は直前の `openat(2)` 呼び出しが返した新規かつ有効な
        // 所有 fd（他に所有者はいない）。`OwnedFd` でラップし、以後の close(2)
        // を std の `Drop` 実装に委譲することで close 漏れ（fd リーク）を防ぐ。
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// 末尾コンポーネント（ファイル本体）を開く（`O_NOFOLLOW` 必須）。
    ///
    /// `extra_flags` に `O_CREAT` を含む場合のみ `mode`（`0o600`）を可変長引数
    /// として渡す（[`openat`] doc 参照。C 側で `mode_t` は `O_CREAT` 未指定時は
    /// 未評価だが、可変長引数の有無自体が呼び出し規約に影響するため実引数の
    /// 個数を一致させる）。
    fn openat_final(parent: RawFd, name: &CString, extra_flags: i32) -> io::Result<OwnedFd> {
        // `O_CLOEXEC` を付与する理由は `openat_dir` と同じ（`raw::O_CLOEXEC` doc
        // 参照）。末尾コンポーネントの fd は `fs::File` へ変換後 `write_all`/
        // `read_to_string` にのみ使うが、経路の途中で早期 return する可能性が
        // あるため無条件に付与する。
        let flags = extra_flags | raw::O_NOFOLLOW | raw::O_CLOEXEC;
        let fd = if extra_flags & raw::O_CREAT != 0 {
            let mode: c_uint = 0o600;
            // SAFETY: openat_dir と同じ契約（parent は検証済み dir-fd・name は
            // NUL 終端済み）。O_CREAT を渡すため mode を可変長引数側に追加する。
            unsafe { openat(parent, name.as_ptr(), flags, mode) }
        } else {
            // SAFETY: 同上。O_CREAT を含まないため mode は渡さない。
            unsafe { openat(parent, name.as_ptr(), flags) }
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat_dir と同じ契約（新規かつ有効な所有 fd を OwnedFd で
        // ラップし close を Drop に委譲する）。
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// `workspace` から `relative_path` を fd 走査で辿り、末尾コンポーネントを
    /// `final_flags`（`O_RDONLY` または `O_WRONLY | O_CREAT | O_TRUNC`）で開く。
    ///
    /// [`read_via_fd_walk`]・[`write_via_fd_walk`]・[`probe`] の共通実体。
    fn walk_to_final(
        workspace: &Path,
        relative_path: &Path,
        final_flags: i32,
    ) -> io::Result<OwnedFd> {
        let parts = split_components(relative_path)?;
        // `split_last` は `(末尾要素, 残り全部)` を返す。ここでの末尾要素が
        // ファイル本体（`file_part`）、残りが先頭からの中間ディレクトリ列
        // （`dir_parts`）である。`split_components` は末尾で空 `Vec` を明示的に
        // 拒否しているため `None` はここでは到達しないはずだが、本番経路で
        // `expect()` を使わない（coding-rust.md「本番経路で unwrap()/expect() を
        // 使わない」）ため型付きエラーへ変換する（security-auditor 監査 Low
        // 指摘対応）。
        let (file_part, dir_parts) = parts.split_last().ok_or_else(|| {
            io::Error::other(format!(
                "内部不整合: split_components が空の候補パスを返しました（{}）。\
                 split_components の空チェックを確認してください",
                relative_path.display()
            ))
        })?;
        let mut current = open_root_dir(workspace)?;
        for part in dir_parts {
            current = openat_dir(current.as_raw_fd(), part)?;
        }
        openat_final(current.as_raw_fd(), file_part, final_flags)
    }

    /// baseline 復元・候補適用が使う書き込み経路。
    ///
    /// `O_WRONLY | O_CREAT | O_TRUNC` で開く。生成 AI 候補は既存モジュール内の
    /// 合成実装に限定され（`feature_addition.rs` の新規ファイル拒否）、
    /// baseline はいずれのコンストラクタも「読み込みに成功した」パスからのみ
    /// 構築される（＝親ディレクトリは常に実在する）ため、本番経路で `O_CREAT`
    /// が実際に新規ファイルを作る場面はない。新規作成の動作自体は
    /// [`crate::candidate`] レベルの `apply_candidate` を直接呼ぶ経路（`pub`
    /// API）でも成立しうるため、汎用性のためのサポートとして残している
    /// （`mkdirat` による中間ディレクトリの新規作成はスコープ外 — 現状の呼び出し
    /// 元にその要求がない）。
    pub(crate) fn write_via_fd_walk(
        workspace: &Path,
        relative_path: &Path,
        content: &str,
    ) -> io::Result<()> {
        let owned = walk_to_final(
            workspace,
            relative_path,
            raw::O_WRONLY | raw::O_CREAT | raw::O_TRUNC,
        )?;
        let mut file = fs::File::from(owned);
        file.write_all(content.as_bytes())
    }

    /// baseline スナップショット取得が使う読み込み経路。
    pub(crate) fn read_via_fd_walk(workspace: &Path, relative_path: &Path) -> io::Result<String> {
        let owned = walk_to_final(workspace, relative_path, raw::O_RDONLY)?;
        let mut file = fs::File::from(owned);
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }

    /// `apply_candidate` の upfront 一括検証専用の「歩くだけ」プローブ。
    ///
    /// [`crate::candidate::apply_candidate`] は「候補が複数ファイルにまたがる
    /// 場合、一部だけ書き換わった状態で `Err` を返さない」契約を持つ（doc
    /// 参照）。これを実現するには実際の書き込みより前に全パスを検証し尽くす
    /// 必要がある。本関数は [`walk_to_final`] と同じ fd 走査を行うが、開いた
    /// fd は即座に破棄し内容の読み書きは行わない（検証のみ）。
    ///
    /// 旧 `reject_symlink_escape` と同じ許容規則を踏襲する: 未実在
    /// （`NotFound`）の中間ディレクトリ・末端ファイルは「まだ作成されていない
    /// だけ」として許容し、実在する symlink（`ELOOP`）や非ディレクトリ
    /// （`ENOTDIR`）等の異常のみを拒否する。
    ///
    /// 本関数の判定と実際の書き込み（[`write_via_fd_walk`]）呼び出しとの間には
    /// 検査後の再解決を挟む余地（TOCTOU window）が理論上存在するが、
    /// [`write_via_fd_walk`] 自身が呼び出し時に改めて `O_NOFOLLOW` 付きで
    /// fd 走査するため、この window を悪用しても最終的な書き込みは拒否される
    /// （本関数は「早期失敗によるファイル書き換えの部分適用防止」が目的であり、
    /// symlink 拒否そのものの安全性は保証しない・保証する必要もない）。
    pub(crate) fn probe(workspace: &Path, relative_path: &Path) -> io::Result<()> {
        match walk_to_final(workspace, relative_path, raw::O_RDONLY) {
            Ok(_owned) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
))]
pub(crate) use supported::{probe, read_via_fd_walk, write_via_fd_walk};

/// 上記 `supported` モジュールの cfg 境界に含まれないプラットフォーム・
/// アーキテクチャ向けの fail-closed フォールバック（モジュール冒頭 doc
/// 参照）。`O_NOFOLLOW`/`O_DIRECTORY` の値の出典を持たない組み合わせでは
/// 候補適用そのものを一律拒否する。cfg は `supported` 側の否定であるため、
/// 両者は必ず排他的かつ網羅的になる（片方にのみ実装がある・両方に実装が
/// ないという不一致が構造的に起こらない）。
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
)))]
mod unsupported {
    use std::io;
    use std::path::Path;

    fn unsupported_platform_error() -> io::Error {
        io::Error::other(
            "候補修正の適用は Linux（x86_64/aarch64）・macOS のみサポートします \
             （fd 走査ベースの symlink TOCTOU 対策〈openat の O_NOFOLLOW〉が \
             定義された値を持つプラットフォーム・アーキテクチャに限定されるため、\
             それ以外では fail-closed に拒否します）",
        )
    }

    pub(crate) fn write_via_fd_walk(
        _workspace: &Path,
        _relative_path: &Path,
        _content: &str,
    ) -> io::Result<()> {
        Err(unsupported_platform_error())
    }

    pub(crate) fn read_via_fd_walk(_workspace: &Path, _relative_path: &Path) -> io::Result<String> {
        Err(unsupported_platform_error())
    }

    pub(crate) fn probe(_workspace: &Path, _relative_path: &Path) -> io::Result<()> {
        Err(unsupported_platform_error())
    }
}

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "macos"
)))]
pub(crate) use unsupported::{probe, read_via_fd_walk, write_via_fd_walk};

#[cfg(all(
    test,
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    )
))]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn temp_workspace(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "self-repair-fd-walk-test-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create_dir_all should succeed in test setup");
        dir
    }

    #[test]
    fn write_then_read_plain_path_round_trips() {
        let dir = temp_workspace("plain-round-trip");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");
        let relative = Path::new("src/lib.rs");

        write_via_fd_walk(&dir, relative, "original")
            .expect("write_via_fd_walk should succeed for a plain existing parent dir");
        let content =
            read_via_fd_walk(&dir, relative).expect("read_via_fd_walk should succeed after write");
        assert_eq!(content, "original");

        write_via_fd_walk(&dir, relative, "updated")
            .expect("write_via_fd_walk should overwrite existing file content");
        let content = read_via_fd_walk(&dir, relative)
            .expect("read_via_fd_walk should see overwritten content");
        assert_eq!(content, "updated");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_creates_new_file_via_existing_parent_dir() {
        let dir = temp_workspace("new-file-via-parent");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");
        let relative = Path::new("src/new_module.rs");
        assert!(!dir.join(relative).exists());

        write_via_fd_walk(&dir, relative, "pub fn added() {}")
            .expect("write_via_fd_walk should create a new file under an existing parent dir");
        let content = read_via_fd_walk(&dir, relative)
            .expect("read_via_fd_walk should read the newly created file");
        assert_eq!(content, "pub fn added() {}");

        // `openat_final` の `O_CREAT` 呼び出しは可変長引数側に `mode = 0o600`
        // を渡す（[`openat`] doc 参照）。ここで実際に owner rw のみが立ち
        // group/other は 0 であることを確認することで、固定 4 引数宣言に
        // よる ABI 不整合（advisor 指摘: `aarch64-apple-darwin` では可変長
        // 引数がスタック経由になり、固定引数宣言だと `mode` が読めず不定値に
        // なりうる）が紛れ込んでいないことをこのテストで検出できるようにする。
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(dir.join(relative))
                .expect("metadata should succeed")
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(
                mode, 0o600,
                "作成されたファイルの権限が想定外です: {mode:o}"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_leaf_symlink_without_touching_outside_target() {
        let dir = temp_workspace("leaf-symlink-reject");
        let outside_dir = temp_workspace("leaf-symlink-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "do-not-overwrite").expect("write should succeed in test setup");

        let relative = Path::new("target.txt");
        std::os::unix::fs::symlink(&outside_file, dir.join(relative))
            .expect("symlink creation should succeed in test setup");

        let write_result = write_via_fd_walk(&dir, relative, "pwned");
        assert!(write_result.is_err());
        assert_eq!(
            fs::read_to_string(&outside_file).expect("read should succeed"),
            "do-not-overwrite"
        );

        let read_result = read_via_fd_walk(&dir, relative);
        assert!(read_result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_intermediate_directory_symlink_without_touching_outside_target() {
        let dir = temp_workspace("dir-symlink-reject");
        let outside_dir = temp_workspace("dir-symlink-reject-outside");
        fs::create_dir_all(&outside_dir).expect("create_dir_all should succeed in test setup");

        std::os::unix::fs::symlink(&outside_dir, dir.join("sub"))
            .expect("symlink creation should succeed in test setup");
        let relative = Path::new("sub/target.txt");

        let write_result = write_via_fd_walk(&dir, relative, "pwned");
        assert!(write_result.is_err());
        assert!(!outside_dir.join("target.txt").exists());

        let read_result = read_via_fd_walk(&dir, relative);
        assert!(read_result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }

    #[test]
    fn probe_allows_not_yet_existing_path() {
        let dir = temp_workspace("probe-not-found");
        fs::create_dir_all(dir.join("src")).expect("create_dir_all should succeed in test setup");

        let result = probe(&dir, Path::new("src/not_created_yet.rs"));
        assert!(result.is_ok());

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn probe_rejects_existing_symlink() {
        let dir = temp_workspace("probe-symlink-reject");
        let outside_dir = temp_workspace("probe-symlink-reject-outside");
        let outside_file = outside_dir.join("secret.txt");
        fs::write(&outside_file, "secret").expect("write should succeed in test setup");

        let relative = Path::new("target.txt");
        std::os::unix::fs::symlink(&outside_file, dir.join(relative))
            .expect("symlink creation should succeed in test setup");

        let result = probe(&dir, relative);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&outside_dir);
    }
}
