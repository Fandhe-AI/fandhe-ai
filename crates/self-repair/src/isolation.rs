//! 候補実行の縦深防御（イシュー #414・A08 ソフトウェア・データ整合性）。
//!
//! # 背景（信頼境界の残余リスク）
//! `self-repair run` は `--candidates` の候補コード（`build.rs`・`#[test]`・
//! proc-macro）を、隔離 sandbox（`git clone --local`。[`crate::sandbox`]）内で
//! `cargo build`／`cargo test --release`／`cargo clippy` を起動して検証する
//! （[`crate::exec::SystemCommandRunner`] 経由。[`crate::verify_gates::CargoVerificationGate`]・
//! `bug_fix.rs`／`feature_addition.rs` の `cargo test --release` 起動元）。
//! `RunSandbox` の隔離は**ファイルシステム上の作業分離のみ**であり、候補
//! コードはホストと同一の OS ユーザー権限・環境変数・ネットワーク到達性の
//! まま任意コード実行できる（`docs/guardrail-self-repair-cli.md` 3.7 節
//! 「候補実行の信頼境界」・`docs/self-repair-candidate-isolation.md` 参照）。
//!
//! 本モジュールは調査記録（`docs/self-repair-candidate-isolation.md`）が
//! 採用した縦深防御（依存クレート追加なし。`.claude/rules/deps-policy.md`）を
//! 実装する:
//! - 環境変数の**遮断**（`Command::env_clear` + 許可リスト再注入。既定有効）
//! - `HOME`／`TMPDIR` の**書き込み先制限**（sandbox 配下へ付け替え。既定有効）
//! - **ネットワーク遮断**（`unshare --user --map-root-user --net` による
//!   network namespace 分離。`--isolate-network` 指定時のみの opt-in。
//!   fail-closed: `unshare` が使えない環境では黙って劣化させず拒否する）
//!
//! これらはプロセス・OS ユーザー権限自体の分離ではない（seccomp／Landlock／
//! コンテナ等は依存追加または syscall 直接呼び出しが必要なため本イシューでは
//! 不採用。`docs/self-repair-candidate-isolation.md` 2 節「将来課題」参照）。

use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 候補実行の子プロセスへ継承を許可する環境変数の allowlist。
///
/// deny-list 方式（[`crate::exec::SystemCommandRunner::run`] の
/// `CARGO_TARGET_DIR`／`GIT_*` 等の個別 `env_remove`）は列挙漏れに弱く、
/// 祖先プロセス（CI・lefthook フック・開発シェル）が設定した未知の秘密情報
/// （API キー・トークン）を遮断できない。allowlist 方式は「明示的に許可した
/// ものだけを通す」fail-closed な構成であり、候補コード（`build.rs`・テスト）
/// が祖先プロセスの秘密情報を観測する経路を遮断する
/// （`.claude/rules/security.md`「秘密情報の混入防止」の実行時版）。
///
/// `CARGO_HOME`／`RUSTUP_HOME` はレジストリキャッシュ・toolchain 参照のため
/// ホスト実パスを明示的に再注入する（[`ExecIsolation::apply`] 参照。付け替える
/// とビルド不能になる。キャッシュ汚染リスクは残余リスクとして
/// `docs/self-repair-candidate-isolation.md` に記録済み）。`RUSTUP_TOOLCHAIN`
/// は含めない（sandbox clone 内の `rust-toolchain.toml` が単一真実源。
/// `.claude/rules/ci.md` 「前提: リポジトリルートの `rust-toolchain.toml`」）。
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "TERM",
    "LANG",
    "LC_ALL",
];

/// `unshare` バイナリ名（PATH 解決に委ねる。絶対パス固定にしないのは
/// `/usr/bin/unshare` 以外に配置される環境〈distro 依存〉を排除しないため）。
const UNSHARE_PROGRAM: &str = "unshare";

/// ネットワーク隔離の方針。既定は [`NetworkIsolation::Inherit`]（従来どおり
/// ホストのネットワーク到達性をそのまま候補実行へ渡す）。`--isolate-network`
/// 指定時のみ [`NetworkIsolation::UnshareNet`] を選び、`unshare` による
/// network namespace 分離でラップする（`cli.rs::RunArgs::isolate_network`
/// 参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NetworkIsolation {
    /// 従来どおりホストのネットワーク到達性を継承する（既定）。
    #[default]
    Inherit,
    /// `unshare --user --map-root-user --net` で network namespace を分離する
    /// （opt-in。`ExecIsolation::probe_unshare_net` が事前に可用性を確認して
    /// いる前提で使う。probe を経ずに使うと、可用性のない環境で実行時に
    /// 初めて失敗し診断が難しくなるため、`main.rs::run_run` は必ず probe を
    /// 先に呼ぶ契約とする）。
    UnshareNet,
}

/// 候補実行の縦深防御設定。[`crate::exec::SystemCommandRunner::isolated`] が
/// 保持し、`run` のたびに [`ExecIsolation::apply`] で `Command` へ適用する。
#[derive(Debug, Clone)]
pub struct ExecIsolation {
    /// 候補実行の `HOME` として使う sandbox 配下のディレクトリ（存在は
    /// 呼び出し元が保証する。本型は付け替え先パスを保持するのみで作成しない。
    /// `main.rs::run_run` が sandbox 構築後に作成する）。
    home: PathBuf,
    /// 候補実行の `TMPDIR` として使う sandbox 配下のディレクトリ。`HOME` と
    /// 同一ディレクトリでも別ディレクトリでもよい（呼び出し元の選択）。
    tmpdir: PathBuf,
    network: NetworkIsolation,
}

impl ExecIsolation {
    /// `home`／`tmpdir` を候補実行の書き込み先制限（sandbox 配下への付け替え）
    /// として使う設定を構築する。ネットワークは既定で [`NetworkIsolation::Inherit`]。
    pub fn new(home: PathBuf, tmpdir: PathBuf) -> Self {
        ExecIsolation {
            home,
            tmpdir,
            network: NetworkIsolation::Inherit,
        }
    }

    /// ネットワーク遮断を有効化した設定を返す（builder パターン）。
    /// 呼び出し前に [`ExecIsolation::probe_unshare_net`] で可用性を
    /// 確認しておくこと（fail-closed。`main.rs::run_run` の呼び出し契約）。
    pub fn with_network_isolation(mut self, network: NetworkIsolation) -> Self {
        self.network = network;
        self
    }

    /// `command` へ環境変数遮断・書き込み先制限を適用する。
    /// [`crate::exec::SystemCommandRunner::isolated`] の `run` 実装から
    /// 呼ばれる（`env_clear` 後の allowlist 再注入。祖先プロセスの環境変数を
    /// 一切継承しない fail-closed 構成）。
    ///
    /// ネットワーク隔離（`unshare` ラップ）は `Command` の構築そのもの
    /// （program/args の差し替え）を伴うため、この関数では扱わない
    /// （[`ExecIsolation::wrap_argv_for_network_isolation`] 参照）。
    pub fn apply(&self, command: &mut Command) {
        command.env_clear();
        for key in ENV_ALLOWLIST {
            if let Some(value) = env::var_os(key) {
                command.env(key, value);
            }
        }
        // `HOME`／`TMPDIR` は allowlist 経由で祖先の値を継承させず、
        // sandbox 配下の専用ディレクトリを明示的に注入する（候補の
        // `build.rs`／テストが `$HOME`・`/tmp` の実体へ書き込むのを sandbox
        // 配下〈使い捨て・削除対象〉へ誘導する。モジュール冒頭ドキュメント
        // 参照）。
        command.env("HOME", &self.home);
        command.env("TMPDIR", &self.tmpdir);
    }

    /// `program`／`args` を、[`NetworkIsolation::UnshareNet`] が指定されている
    /// 場合のみ `unshare` ラップした argv（`(新 program, 新 args)`）へ変換する
    /// 純関数。[`Inherit`][NetworkIsolation::Inherit] の場合は変換なしで
    /// そのまま返す。
    ///
    /// `Command` 構築から切り離した純関数にする理由: ユニットテストで
    /// 実際に `unshare` を起動せず argv 合成のみを検証できるようにするため
    /// （実装計画 §4 ステップ 5「network ラップ純関数の argv 合成」）。
    ///
    /// # A03（インジェクション）対応
    /// `--` 区切りで `program`／`args` を渡すため、候補側の引数が `unshare`
    /// 自身のフラグとして再解釈されることはない（シェルを経由しない配列
    /// 構築。`.claude/rules/security.md` A03 と同じ契約）。
    pub fn wrap_argv_for_network_isolation<'a>(
        &self,
        program: &'a str,
        args: &'a [&'a str],
    ) -> (&'a str, Vec<&'a str>) {
        match self.network {
            NetworkIsolation::Inherit => (program, args.to_vec()),
            NetworkIsolation::UnshareNet => {
                let mut wrapped = vec!["--user", "--map-root-user", "--net", "--", program];
                wrapped.extend_from_slice(args);
                (UNSHARE_PROGRAM, wrapped)
            }
        }
    }

    /// `network` が [`NetworkIsolation::UnshareNet`] かを返す
    /// （[`crate::exec::SystemCommandRunner`] が argv ラップの要否を判定する
    /// ために使う）。
    pub fn is_network_isolated(&self) -> bool {
        matches!(self.network, NetworkIsolation::UnshareNet)
    }

    /// `unshare --user --map-root-user --net true` を実行環境で 1 回試行し、
    /// network namespace 分離が可能かを確認する（fail-closed probe）。
    ///
    /// user namespace は root 不要で使える一方、container／CI 環境では
    /// カーネル・seccomp policy により禁止されている場合がある
    /// （`docs/self-repair-candidate-isolation.md` 2 節参照）。probe が失敗
    /// した場合、呼び出し元（`main.rs::run_run`）は `--isolate-network`
    /// 指定時に黙ってネットワーク隔離なしへ劣化させず、内部エラー区分
    /// （exit 1）として明確に拒否する（security.md A05「セキュリティ設定
    /// ミス」を招かないため）。
    pub fn probe_unshare_net() -> Result<(), String> {
        probe_unshare_net_with_program(UNSHARE_PROGRAM)
    }
}

/// [`ExecIsolation::probe_unshare_net`] の本体。probe に使うプログラム名を
/// 注入できる形にし、存在しないバイナリ名を渡すことで「probe 失敗」を
/// 決定的に再現するユニットテストを書けるようにする
/// （`sandbox.rs::create_at` と同じ「注入で決定化」パターン。実装計画 §4
/// ステップ 5「probe 失敗時に `Err`」参照）。
fn probe_unshare_net_with_program(program: impl AsRef<OsStr>) -> Result<(), String> {
    let output = Command::new(program.as_ref())
        .args(["--user", "--map-root-user", "--net", "true"])
        .env_clear()
        .output()
        .map_err(|error| {
            format!(
                "{} の起動に失敗しました: {error}（user namespace によるネットワーク隔離が \
                 この実行環境では利用できない可能性があります）",
                program.as_ref().to_string_lossy()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "{} --user --map-root-user --net true が失敗しました（exit={:?}）: {}。\
             この実行環境は user namespace によるネットワーク隔離を許可していない可能性が \
             あります（container/CI 環境で無効化されている場合があります。\
             docs/self-repair-candidate-isolation.md 2 節参照）",
            program.as_ref().to_string_lossy(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// `--home`/`--tmpdir` に使う sandbox 配下パスを組み立てるヘルパー。
/// `main.rs::run_run` が sandbox 構築後に呼び、返した 2 パスを
/// `fs::create_dir_all` してから [`ExecIsolation::new`] へ渡す（本関数自体は
/// ディレクトリを作成しない。副作用を持たない純粋なパス組み立てに留める）。
pub fn candidate_home_dirs(sandbox_root: &Path) -> (PathBuf, PathBuf) {
    let base = sandbox_root.join(".self-repair-isolation");
    (base.join("home"), base.join("tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_clears_ancestor_environment_and_reinjects_allowlist_only() {
        // マーカー環境変数（allowlist 外）を親プロセス側に設定し、`apply` 後の
        // `Command` がそれを引き継がないことを確認する。`Command` は
        // 実行前の環境設定を外部から直接観測できないため、実際に子プロセス
        // （シェル非経由の `printenv` 直接起動）を起動して確認する
        // （実装計画 §4 ステップ 5 unit テスト方針）。
        unsafe {
            env::set_var("SELF_REPAIR_TEST_SECRET", "must-not-leak");
        }
        let home = env::temp_dir().join("self-repair-isolation-test-home");
        let tmp = env::temp_dir().join("self-repair-isolation-test-tmp");
        let isolation = ExecIsolation::new(home, tmp);

        let mut command = Command::new("printenv");
        command.arg("SELF_REPAIR_TEST_SECRET");
        isolation.apply(&mut command);
        let output = command.output().expect("printenv の起動に失敗");

        unsafe {
            env::remove_var("SELF_REPAIR_TEST_SECRET");
        }

        assert!(
            !output.status.success(),
            "allowlist 外の環境変数は継承されず printenv が非 0 終了するはず"
        );
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("must-not-leak"),
            "マーカー環境変数の値が子プロセスへ漏れてはならない"
        );
    }

    #[test]
    fn apply_redirects_home_and_tmpdir_to_injected_paths() {
        let home = PathBuf::from("/sandbox/isolation-home");
        let tmp = PathBuf::from("/sandbox/isolation-tmp");
        let isolation = ExecIsolation::new(home.clone(), tmp.clone());

        let mut command = Command::new("sh");
        command.args(["-c", "printf '%s|%s' \"$HOME\" \"$TMPDIR\""]);
        isolation.apply(&mut command);
        let output = command.output().expect("sh の起動に失敗");
        let stdout = String::from_utf8_lossy(&output.stdout);

        assert_eq!(
            stdout,
            format!("{}|{}", home.display(), tmp.display()),
            "HOME/TMPDIR は sandbox 配下パスへ付け替わるはず"
        );
    }

    #[test]
    fn apply_reinjects_host_path_cargo_home_and_rustup_home() {
        // PATH/CARGO_HOME/RUSTUP_HOME はホスト値を再注入する契約（付け替えると
        // ビルド不能になるため）。`env::var_os` が Some を返す環境変数について
        // 再注入を確認する。CI/開発コンテナのいずれも PATH は必ず設定される。
        let home = env::temp_dir().join("self-repair-isolation-test-home2");
        let tmp = env::temp_dir().join("self-repair-isolation-test-tmp2");
        let isolation = ExecIsolation::new(home, tmp);

        let mut command = Command::new("sh");
        command.args(["-c", "printf '%s' \"$PATH\""]);
        isolation.apply(&mut command);
        let output = command.output().expect("sh の起動に失敗");
        let stdout = String::from_utf8_lossy(&output.stdout);

        let expected_path = env::var_os("PATH").expect("テスト実行環境に PATH が必要");
        assert_eq!(
            stdout,
            expected_path.to_string_lossy(),
            "PATH はホスト値がそのまま再注入されるはず"
        );
    }

    #[test]
    fn wrap_argv_is_noop_when_network_inherit() {
        let isolation = ExecIsolation::new(PathBuf::from("/h"), PathBuf::from("/t"));
        let (program, args) = isolation.wrap_argv_for_network_isolation("cargo", &["build"]);
        assert_eq!(program, "cargo");
        assert_eq!(args, vec!["build"]);
    }

    #[test]
    fn wrap_argv_prefixes_unshare_flags_with_double_dash_separator() {
        let isolation = ExecIsolation::new(PathBuf::from("/h"), PathBuf::from("/t"))
            .with_network_isolation(NetworkIsolation::UnshareNet);
        let (program, args) =
            isolation.wrap_argv_for_network_isolation("cargo", &["build", "--release"]);
        assert_eq!(program, UNSHARE_PROGRAM);
        assert_eq!(
            args,
            vec![
                "--user",
                "--map-root-user",
                "--net",
                "--",
                "cargo",
                "build",
                "--release"
            ]
        );
    }

    #[test]
    fn wrap_argv_does_not_let_candidate_args_be_reinterpreted_as_unshare_flags() {
        // 候補側の引数が `--net` のような unshare 自身のフラグと同名でも、
        // `--` 区切りより後ろに置かれるため unshare 側のフラグとして
        // 再解釈されないことを確認する（A03 対応。モジュール冒頭ドキュメント）。
        let isolation = ExecIsolation::new(PathBuf::from("/h"), PathBuf::from("/t"))
            .with_network_isolation(NetworkIsolation::UnshareNet);
        let (_, args) = isolation.wrap_argv_for_network_isolation("cargo", &["--net", "test"]);
        let separator_index = args
            .iter()
            .position(|arg| *arg == "--")
            .expect("-- 区切りが存在するはず");
        // `--` より後ろの要素にのみ候補側の引数が来る（program も含む）。
        assert_eq!(&args[separator_index + 1..], &["cargo", "--net", "test"]);
    }

    #[test]
    fn probe_unshare_net_fails_closed_for_nonexistent_binary() {
        // 存在しないバイナリ名を注入し、probe 失敗を決定的に再現する
        // （`sandbox.rs::create_at` と同じ「注入で決定化」パターン）。
        let result =
            probe_unshare_net_with_program("self_repair_definitely_not_a_real_unshare_binary_xyz");
        assert!(
            result.is_err(),
            "存在しないバイナリの probe は Err を返すはず"
        );
    }

    #[test]
    fn candidate_home_dirs_are_distinct_and_nested_under_sandbox_root() {
        let sandbox_root = PathBuf::from("/sandbox/root");
        let (home, tmp) = candidate_home_dirs(&sandbox_root);
        assert!(home.starts_with(&sandbox_root));
        assert!(tmp.starts_with(&sandbox_root));
        assert_ne!(home, tmp, "HOME と TMPDIR は別ディレクトリのはず");
    }
}
