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
//! - 環境変数の**遮断**（`Command::env_clear` + 許可リスト再注入。既定有効。
//!   `CARGO_HOME`／`RUSTUP_HOME` はホスト実パスへ明示再注入し、未設定環境
//!   でも rustup の既定パスへフォールバックする）
//! - `HOME`／`TMPDIR` の**書き込み先制限**（`RunSandbox::root()` の**外側**
//!   〈兄弟ディレクトリ〉の専用ディレクトリへ付け替え。既定有効。
//!   [`candidate_home_dirs`] 参照）
//! - **ネットワーク遮断**（`unshare --user --map-current-user --net` による
//!   network namespace 分離。`--isolate-network` 指定時のみの opt-in。
//!   fail-closed: `unshare` が使えない環境では黙って劣化させず拒否する）
//!
//! これらはプロセス・OS ユーザー権限自体の分離ではない（seccomp／Landlock／
//! コンテナ等は依存追加または syscall 直接呼び出しが必要なため本イシューでは
//! 不採用。`docs/self-repair-candidate-isolation.md` 2 節「将来課題」参照）。

use std::env;
use std::ffi::{OsStr, OsString};
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
/// `docs/self-repair-candidate-isolation.md` に記録済み）。両変数が
/// `env::var_os` で `None`（未設定。rustup の既定値をそのまま使っている
/// 環境）の場合も、[`resolve_toolchain_home_reinjections`] が rustup の
/// 既定パス（`$HOME/.cargo`／`$HOME/.rustup`）へフォールバックする。
/// 再注入自体を省略すると、`HOME` だけが sandbox 配下の空ディレクトリへ
/// 付け替わり、rustup が `$HOME/.cargo`・`$HOME/.rustup` を空と誤認して
/// toolchain 一式をネットワーク経由で毎回再ダウンロードする（レビュー
/// 指摘 #414 実測。`--isolate-network` と併用すると遮断下で取得できず
/// 検証ゲートが偽陰性で全滅する）。`RUSTUP_TOOLCHAIN`
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

/// network namespace 分離に使う `unshare` フラグ（`--net` 手前まで）。
/// [`ExecIsolation::wrap_argv_for_network_isolation`]（実行時の argv 合成）と
/// [`probe_unshare_net_with_program`]（事前 probe）の両方から参照する単一の
/// 真実源（レビュー指摘 #414: 2 箇所で文字列リテラルが重複していると、
/// 一方だけ変更した場合に probe が「実行時に本当に使う argv」と異なる
/// コマンドを検証してしまい、probe が成功を返しても実行時に失敗しうる）。
///
/// `--map-current-user`（現在の uid をそのまま namespace 内 uid へ 1 対 1
/// マップ）を使う。`--map-root-user`（namespace 内で uid 0＝擬似 root へ
/// マップ）ではなく本フラグを選ぶのは、ネットワーク遮断という目的に対し
/// namespace 内で不要な root 権限（関連 capability）を候補コードへ与える
/// のは縦深防御の趣旨に反するため（レビュー指摘 #414 Medium）。いずれの
/// マッピングも「現在の euid を単一行で namespace 内の 1 uid へ写す」操作
/// であり `CAP_SETUID` を要さず、user namespace 作成の可否
/// （`unprivileged_userns_clone`／container の seccomp policy）とは独立
/// （`docs/self-repair-candidate-isolation.md` 2 節）。`--map-current-user`
/// は util-linux 2.38 以降が前提（`unshare --help` で確認可能）。より古い
/// util-linux の runner では probe が失敗し fail-closed（exit 1）で拒否する
/// （劣化させず拒否するのが本モジュールの既定方針。上位 doc 参照）。
const UNSHARE_NET_FLAGS: [&str; 3] = ["--user", "--map-current-user", "--net"];

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
    /// `unshare --user --map-current-user --net`（`UNSHARE_NET_FLAGS`）で
    /// network namespace を分離する（opt-in。`ExecIsolation::probe_unshare_net`
    /// が事前に可用性を確認して
    /// いる前提で使う。probe を経ずに使うと、可用性のない環境で実行時に
    /// 初めて失敗し診断が難しくなるため、`main.rs::run_run` は必ず probe を
    /// 先に呼ぶ契約とする）。
    UnshareNet,
}

/// 候補実行の縦深防御設定。[`crate::exec::SystemCommandRunner::isolated`] が
/// 保持し、`run` のたびに [`ExecIsolation::apply`] で `Command` へ適用する。
#[derive(Debug, Clone)]
pub struct ExecIsolation {
    /// 候補実行の `HOME` として使う専用ディレクトリ（`RunSandbox::root()` の
    /// **外側**〈兄弟ディレクトリ〉。[`candidate_home_dirs`] 参照。存在は
    /// 呼び出し元が保証する。本型は付け替え先パスを保持するのみで作成しない。
    /// `main.rs::run_run` が構築する）。
    home: PathBuf,
    /// 候補実行の `TMPDIR` として使う専用ディレクトリ（`home` と同じく
    /// sandbox_root の外側）。`HOME` と同一ディレクトリでも別ディレクトリ
    /// でもよい（呼び出し元の選択）。
    tmpdir: PathBuf,
    network: NetworkIsolation,
}

impl ExecIsolation {
    /// `home`／`tmpdir` を候補実行の書き込み先制限（sandbox_root 外の専用
    /// ディレクトリへの付け替え）として使う設定を構築する。ネットワークは
    /// 既定で [`NetworkIsolation::Inherit`]。
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
        // `CARGO_HOME`／`RUSTUP_HOME` が未設定（`env::var_os` が `None`）の
        // 環境では、上のループでは何も注入されない。そのまま `HOME` だけ
        // 下で sandbox 配下へ付け替えると、rustup が `$HOME/.cargo`・
        // `$HOME/.rustup` を空ディレクトリと誤認し、候補実行のたび
        // toolchain 一式をネットワーク経由で再ダウンロードする（レビュー
        // 指摘 #414 実測。`ENV_ALLOWLIST` doc 参照）。ホストの実 `HOME` を
        // 基準に rustup の既定パスへフォールバックする。
        for (key, value) in resolve_toolchain_home_reinjections(
            env::var_os("CARGO_HOME"),
            env::var_os("RUSTUP_HOME"),
            env::var_os("HOME"),
        ) {
            command.env(key, value);
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
                let mut wrapped: Vec<&str> = UNSHARE_NET_FLAGS.to_vec();
                wrapped.push("--");
                wrapped.push(program);
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

    /// `unshare --user --map-current-user --net true`（`UNSHARE_NET_FLAGS`）
    /// を実行環境で 1 回試行し、
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

/// `CARGO_HOME`／`RUSTUP_HOME` が未設定の場合に注入すべき `(key, value)` を
/// 返す純関数（[`ExecIsolation::apply`] から呼ばれる）。祖先プロセス側の
/// 実測値を直接 `env::var_os` するのではなく引数として受け取る形にしたのは、
/// [`probe_unshare_net_with_program`] と同じ「注入で決定化」パターンで、
/// 実プロセス環境変数を書き換えずに未設定ケース（フォールバック経路）を
/// ユニットテストで決定的に再現するため（プロセス環境変数はテストバイナリ
/// 全体で共有されるグローバル状態であり、`cargo` を実際に起動する他のテスト
/// 〈`verify_gates.rs` 等〉と競合しうるため書き換えない）。
///
/// フォールバック値は rustup の既定パス（`$HOME/.cargo`・`$HOME/.rustup`）。
/// `host_home` も `None`（`HOME` 自体が未設定）の場合は何も注入しない
/// （フォールバック先がないため。`env_clear` 済みの `Command` はその変数を
/// 持たないまま起動され、cargo/rustup 自体の未設定時エラーに委ねる）。
fn resolve_toolchain_home_reinjections(
    cargo_home: Option<OsString>,
    rustup_home: Option<OsString>,
    host_home: Option<OsString>,
) -> Vec<(&'static str, PathBuf)> {
    let mut reinjections = Vec::new();
    if cargo_home.is_none()
        && let Some(home) = host_home.as_ref()
    {
        reinjections.push(("CARGO_HOME", PathBuf::from(home).join(".cargo")));
    }
    if rustup_home.is_none()
        && let Some(home) = host_home.as_ref()
    {
        reinjections.push(("RUSTUP_HOME", PathBuf::from(home).join(".rustup")));
    }
    reinjections
}

/// [`ExecIsolation::probe_unshare_net`] の本体。probe に使うプログラム名を
/// 注入できる形にし、存在しないバイナリ名を渡すことで「probe 失敗」を
/// 決定的に再現するユニットテストを書けるようにする
/// （`sandbox.rs::create_at` と同じ「注入で決定化」パターン。実装計画 §4
/// ステップ 5「probe 失敗時に `Err`」参照）。[`UNSHARE_NET_FLAGS`] を
/// [`ExecIsolation::wrap_argv_for_network_isolation`] と共有し、probe が
/// 実行時に本当に使う argv と異なるコマンドを検証してしまう desync を防ぐ
/// （レビュー指摘 #414）。
fn probe_unshare_net_with_program(program: impl AsRef<OsStr>) -> Result<(), String> {
    let mut probe_args: Vec<&str> = UNSHARE_NET_FLAGS.to_vec();
    probe_args.push("true");
    let output = Command::new(program.as_ref())
        .args(&probe_args)
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
            "{} {} が失敗しました（exit={:?}）: {}。\
             この実行環境は user namespace によるネットワーク隔離を許可していない可能性が \
             あります（container/CI 環境で無効化されている場合や、`--map-current-user` \
             未対応の古い util-linux〈2.38 未満〉である可能性があります。\
             docs/self-repair-candidate-isolation.md 2 節参照）",
            program.as_ref().to_string_lossy(),
            probe_args.join(" "),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// `--home`/`--tmpdir` に使うディレクトリを組み立てるヘルパー。
/// `main.rs::run_run` が sandbox 構築後に呼び、返した 2 パスを
/// `fs::create_dir_all` してから [`ExecIsolation::new`] へ渡す（本関数自体は
/// ディレクトリを作成しない。副作用を持たない純粋なパス組み立てに留める）。
///
/// # sandbox_root の**外側**（兄弟ディレクトリ）に置く理由（レビュー指摘 #414）
/// `sandbox_root`（`RunSandbox::root()`）は
/// `crate::verify_direct_composite::RepairCompositeGate::verify`
/// （→ [`crate::diff_signals::measure_diff_signals`]／`verify_bench_direct.rs`）
/// が検証のたび `git add -A -- .` で diff を計測する git worktree そのもの。
/// 隔離ディレクトリを `sandbox_root` の内側に置くと、候補の `build.rs`／
/// テストが `$HOME`／`$TMPDIR` へ書き込んだファイルがすべて `git add -A`
/// に拾われ、`lines_changed`／`api_broken` 等の diff シグナルを汚染する
/// （`Adopted` 判定時は `sandbox.rs::reflect_adopted_diff` がこの汚染された
/// diff を `--repo` へ反映しうる。`.claude/rules/security.md` A08「判定の
/// 迂回経路を作らない」に抵触しうる）。`.gitignore` による除外ではなく
/// 物理的に外へ出すのは、`.gitignore` 自体を sandbox 内へ書き込むと
/// それ自体が `git add -A` に拾われる未追跡ファイルになるため。
pub fn candidate_home_dirs(sandbox_root: &Path) -> (PathBuf, PathBuf) {
    let base = isolation_sibling_dir(sandbox_root);
    (base.join("home"), base.join("tmp"))
}

/// [`candidate_home_dirs`] が使う、`sandbox_root` の親ディレクトリ配下に
/// 兄弟パスとして隔離ディレクトリ名を組み立てる（`sandbox_root` 自体には
/// 触れない）。`sandbox_root` に親がない異常系（`RunSandbox::create` は
/// 常に `env::temp_dir()` 配下の一意パスを生成するため通常発生しない）では
/// `env::temp_dir()` 直下へフォールバックする（`sandbox_root` 配下へは
/// 戻さない。内側配置を避けるという本関数の契約を異常系でも破らないため）。
fn isolation_sibling_dir(sandbox_root: &Path) -> PathBuf {
    let dir_name = sandbox_root
        .file_name()
        .map(|name| format!("{}-isolation", name.to_string_lossy()))
        .unwrap_or_else(|| "self-repair-isolation".to_string());
    match sandbox_root.parent() {
        Some(parent) => parent.join(dir_name),
        None => env::temp_dir().join(dir_name),
    }
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
        // SAFETY: `env::set_var` / `env::remove_var` はプロセス全体の環境変数
        // テーブルを書き換えるため、他スレッドが同時に環境変数を読み書きしている
        // と data race になりうる。本クレート（`crates/self-repair`）の
        // テストバイナリ内でプロセス環境変数を読み書きするのはこの 1 テスト
        // （`apply_clears_ancestor_environment_and_reinjects_allowlist_only`）
        // のみであり（`grep -rn "set_var\|remove_var\|env::var" crates/self-repair/src`
        // で確認済み。他テストは `std::env::temp_dir`／`Command::env` 等
        // プロセス環境変数を経由しない API のみを使う）、他スレッドから
        // 並行して環境変数へアクセスされることはない。
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

        // SAFETY: 上記 set_var と同一テスト内の対のクリーンアップ。他スレッドが
        // 並行して環境変数へアクセスしないことの根拠は set_var 側のコメントに
        // 記載の通り（本クレートで環境変数を読み書きするのはこのテストのみ）。
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
    fn resolve_toolchain_home_reinjections_falls_back_to_host_home_when_unset() {
        // CARGO_HOME/RUSTUP_HOME が未設定の環境（レビュー指摘 #414 実測環境）
        // を、プロセス環境変数を書き換えずに `None` 注入で再現する
        // （実プロセス側の書き換えは `verify_gates.rs` 等 cargo を実起動する
        // 他テストと競合しうるため避ける。関数 doc 参照）。
        let host_home = OsString::from("/home/example");
        let reinjections = resolve_toolchain_home_reinjections(None, None, Some(host_home));
        assert_eq!(
            reinjections,
            vec![
                ("CARGO_HOME", PathBuf::from("/home/example/.cargo")),
                ("RUSTUP_HOME", PathBuf::from("/home/example/.rustup")),
            ],
            "CARGO_HOME/RUSTUP_HOME 未設定時は rustup の既定パスへフォールバックするはず"
        );
    }

    #[test]
    fn resolve_toolchain_home_reinjections_is_noop_when_already_set() {
        // 既に CARGO_HOME/RUSTUP_HOME が設定されている環境では `apply` の
        // allowlist ループが既にホスト値を再注入済みのため、本関数からは
        // 何も追加注入しない（二重注入で `Command::env` の後勝ち上書きに
        // 依存しない）。
        let reinjections = resolve_toolchain_home_reinjections(
            Some(OsString::from("/custom/cargo")),
            Some(OsString::from("/custom/rustup")),
            Some(OsString::from("/home/example")),
        );
        assert!(
            reinjections.is_empty(),
            "既に設定済みの CARGO_HOME/RUSTUP_HOME は上書きしないはず"
        );
    }

    #[test]
    fn resolve_toolchain_home_reinjections_is_noop_when_host_home_also_unset() {
        // HOME 自体が未設定でフォールバック先がない場合は何も注入しない
        // （フォールバックできないケースを黙って誤ったパスへ誘導しない）。
        let reinjections = resolve_toolchain_home_reinjections(None, None, None);
        assert!(
            reinjections.is_empty(),
            "HOME 未設定時はフォールバック先がなく何も注入しないはず"
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
                "--map-current-user",
                "--net",
                "--",
                "cargo",
                "build",
                "--release"
            ]
        );
    }

    #[test]
    fn wrap_argv_flags_match_probe_flags() {
        // `wrap_argv_for_network_isolation`（実行時）と
        // `probe_unshare_net_with_program`（事前 probe）が同一の
        // `UNSHARE_NET_FLAGS` を参照していることを確認する（レビュー指摘
        // #414: 2 箇所で文字列リテラルが重複し desync すると、probe 成功が
        // 実行時成功を保証しなくなる）。
        let isolation = ExecIsolation::new(PathBuf::from("/h"), PathBuf::from("/t"))
            .with_network_isolation(NetworkIsolation::UnshareNet);
        let (_, args) = isolation.wrap_argv_for_network_isolation("cargo", &["build"]);
        assert_eq!(
            &args[..UNSHARE_NET_FLAGS.len()],
            UNSHARE_NET_FLAGS.as_slice(),
            "wrap_argv の先頭フラグは UNSHARE_NET_FLAGS と一致するはず"
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
    fn candidate_home_dirs_are_distinct_and_outside_sandbox_root() {
        // レビュー指摘 #414: sandbox_root の内側に置くと `git add -A` が
        // 候補の $HOME/$TMPDIR 書き込みを diff シグナルとして拾ってしまう
        // ため、sandbox_root の外側（兄弟ディレクトリ）に置く契約を検証する。
        let sandbox_root = PathBuf::from("/tmp/self-repair-run-sandbox-123-456");
        let (home, tmp) = candidate_home_dirs(&sandbox_root);
        assert!(
            !home.starts_with(&sandbox_root),
            "HOME は sandbox_root の内側に置いてはならない: {home:?}"
        );
        assert!(
            !tmp.starts_with(&sandbox_root),
            "TMPDIR は sandbox_root の内側に置いてはならない: {tmp:?}"
        );
        assert_eq!(
            home.parent(),
            tmp.parent(),
            "HOME/TMPDIR は同一の隔離ベースディレクトリの子であるはず"
        );
        assert_eq!(
            home.parent().and_then(Path::parent),
            sandbox_root.parent(),
            "隔離ベースディレクトリは sandbox_root と同じ親ディレクトリの直下（兄弟）にあるはず"
        );
        assert_ne!(home, tmp, "HOME と TMPDIR は別ディレクトリのはず");
    }

    #[test]
    fn candidate_home_dirs_falls_back_to_temp_dir_when_sandbox_root_has_no_parent() {
        // `sandbox_root` に親がない異常系でも、フォールバック先が
        // `sandbox_root` 自体の配下に戻らないことを確認する（内側配置を
        // 避けるという契約を異常系でも破らない）。
        let sandbox_root = PathBuf::from("/");
        let (home, tmp) = candidate_home_dirs(&sandbox_root);
        assert!(!home.starts_with("/self-repair-isolation"));
        assert!(home.starts_with(env::temp_dir()));
        assert!(tmp.starts_with(env::temp_dir()));
    }
}
