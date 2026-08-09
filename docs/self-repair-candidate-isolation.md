# 候補実行の OS レベル sandbox 化: 調査結果・採否判断（イシュー #414）

## 1. 背景

`self-repair run` は `--candidates` の候補コードを、隔離 sandbox
（`git clone --local`。`crates/self-repair/src/sandbox.rs`）内で
`cargo build`／`cargo test --release`／`cargo clippy` を起動して検証する
（`crates/self-repair/src/verify_gates.rs`・`verify_direct_composite.rs`・
`bug_fix.rs`／`feature_addition.rs` の検出器）。この隔離は**ファイルシステム上の
作業分離のみ**であり、候補コード（`build.rs`・`#[test]`・proc-macro）は
ホストと同一の OS ユーザー権限・環境変数・ネットワーク到達性のまま任意コード
実行できる（PR #361 の「対象外」節・`docs/guardrail-self-repair-cli.md` 3.7 節
「候補実行の信頼境界」で追跡済み）。

現状の緩和策は `--allow-candidate-exec`（明示的承認・既定拒否。
`cli.rs::parse_run`）のみで、承認後の実行自体には縦深防御がなかった。

本イシューは A08（ソフトウェア・データ整合性）観点の縦深防御として、
**外部クレートを追加せずに**実現可能な範囲（環境変数の遮断・書き込み先の
制限・ネットワーク遮断オプション）を調査し、採用範囲を段階的に実装した
記録である。

## 2. 調査結果（依存追加なしで可能な隔離強化の選択肢）

| # | 手段 | 依存 | 採否 | 理由 |
|---|------|------|------|------|
| a | **環境変数の遮断**（`Command::env_clear` + 許可リスト再注入） | std のみ | **採用（候補実行経路で既定有効）** | 祖先プロセスの環境変数（トークン・API キー・プロキシ設定等）が候補コードへ継承されるのを遮断する。現行の deny-list 方式（`exec.rs::SystemCommandRunner::new` の `CARGO_TARGET_DIR`／`GIT_*` 等の個別 `env_remove`）は列挙漏れに弱く、allowlist 方式が fail-closed |
| b | **書き込み先の制限**（`HOME`・`TMPDIR` を専用ディレクトリへ付け替え） | std のみ | **採用（候補実行経路で既定有効）** | 候補の `build.rs`／テストが `$HOME`・`/tmp` の実体へ書き込むのを専用ディレクトリ（削除対象）へ誘導する。専用ディレクトリは `RunSandbox::root()`（診断用 git worktree）の**外側**（兄弟ディレクトリ）に置く。内側に置くと `git add -A`（`diff_signals.rs`／`verify_bench_direct.rs`）が候補の `$HOME`／`$TMPDIR` 書き込みを diff シグナルとして拾ってしまうため（レビュー指摘 #414。`isolation.rs::candidate_home_dirs` doc 参照）。`CARGO_HOME`／`RUSTUP_HOME` はレジストリキャッシュ・toolchain 参照のためホスト実パスを明示再注入する（付け替えるとビルド不能。両変数が未設定の環境でも rustup の既定パス〈`$HOME/.cargo`・`$HOME/.rustup`〉へフォールバックする〈レビュー指摘 #414 実測: フォールバックがないと候補実行のたび toolchain を再ダウンロードする〉。キャッシュ汚染リスクは 3 節「残余リスク」参照） |
| c | **ネットワーク遮断**（util-linux `unshare --user --map-current-user --net` による network namespace 分離でラップ） | 外部**バイナリ**のみ（クレート依存なし） | **採用（opt-in フラグ `--isolate-network`・fail-closed）** | Linux で root 不要・依存クレート不要。`--map-current-user`（現在の uid を namespace 内でも同じ uid へ写す）を使い、`--map-root-user`（namespace 内で uid 0＝擬似 root へ写す）を避けることで、ネットワーク遮断という目的に対し不要な特権を候補コードへ付与しない（レビュー指摘 #414）。`--map-current-user` は util-linux 2.38 以降が前提。ただし container/CI 環境では user namespace が禁止されている場合があり、また古い util-linux では `--map-current-user` 自体が使えない場合があるため既定 off の opt-in とし、指定時に事前 probe が失敗したら黙って劣化させず exit 1 で拒否する |
| d | seccomp／Landlock／`prctl(PR_SET_NO_NEW_PRIVS)` の直接呼び出し | `libc` 等の新規クレート or 生 syscall unsafe | **不採用** | 依存追加はユーザー承認必須（`.claude/rules/deps-policy.md`）。生 syscall の unsafe 自作は「unsafe は FFI 境界等の必要最小限」（`.claude/rules/coding-rust.md`）に反する。将来課題として記録 |
| e | 低権限コンテナ（docker/podman）・`systemd-run`・`bwrap` | 重い外部ツール | **不採用** | 実行環境（self-hosted runner・開発コンテナ）での可用性が保証できず、CLI の可搬性を損なう。将来課題として記録 |
| f | macOS `sandbox-exec`（seatbelt） | OS ツール | **不採用** | 本 CLI の主実行環境は Linux（self-hosted CI・開発コンテナ）。macOS 実機経路は現状 `#[ignore]` 分離対象であり優先度が低い。将来課題として記録 |

適用範囲: a・b は `self-repair run` の候補実行経路（検出器の `cargo test`・
4 ゲート検証・ベンチゲートが共有する `SystemCommandRunner`）へ一律適用する。
git 操作（`diff_signals` 等）も同一 runner を経由するため同じ遮断下に置く
（`GIT_*` 除去の既存契約は allowlist 方式で自然に包含される）。

## 3. 実装

- `crates/self-repair/src/isolation.rs`: `ExecIsolation` 設定型（環境変数
  allowlist・`HOME`/`TMPDIR` リダイレクト先・`NetworkIsolation`）と
  `unshare` 可用性 probe（`ExecIsolation::probe_unshare_net`）、argv ラップ
  純関数（`wrap_argv_for_network_isolation`。`--` 区切りでシェル非経由・
  A03 準拠）を提供する。
- `crates/self-repair/src/exec.rs`: `SystemCommandRunner::isolated(..)` を
  追加。`SystemCommandRunner::new()` の現行挙動（deny-list）は不変のまま
  既存呼び出し元・テストとの互換を保つ。
- `crates/self-repair/src/cli.rs`: `--isolate-network`（既定 false・値なし）
  を追加。
- `crates/self-repair/src/main.rs::run_run`: sandbox 構築後に
  `ExecIsolation` を構築する。`--isolate-network` 指定時は probe を実行し、
  失敗したら内部エラー区分（exit 1）で fail-closed に拒否する。
  `gate_spec.runner`・`BugFixDetector`・`FeatureAdditionDetector` の
  3 箇所すべてで `SystemCommandRunner::isolated(..)` を使う。

### 環境変数 allowlist

`PATH`・`CARGO_HOME`・`RUSTUP_HOME`・`TERM`・`LANG`・`LC_ALL` のみを
再注入する。`RUSTUP_TOOLCHAIN` は含めない（sandbox clone 内の
`rust-toolchain.toml` が単一真実源。`.claude/rules/ci.md` 「前提: リポジトリ
ルートの `rust-toolchain.toml`」）。

### 残余リスク（記録対象）

- **プロセス・OS ユーザー権限自体は非分離のまま**: seccomp／Landlock／
  コンテナ等（2 節 d・e・f）は依存追加・可搬性の理由で本イシューでは
  不採用。将来課題として残る
- **`CARGO_HOME` はホスト共有**: レジストリキャッシュ・toolchain 参照のため
  ホスト実パスを再注入する契約上、キャッシュ汚染の理論的余地が残る
  （候補コードが `$CARGO_HOME` 配下へ書き込みうる）
- **user namespace 不可環境ではネットワーク遮断そのものが使えない**:
  `--isolate-network` は opt-in のため既定動作には影響しないが、この制約が
  ある環境では選択肢として使えない。`--map-current-user` 自体が util-linux
  2.38 未満では使えない環境も同様（probe が fail-closed で拒否する）

これらの追加起票要否は `.claude/rules/out-of-scope-tracking.md` に沿って
別途ユーザー承認を経て判断する。

## 4. 検証

- `crates/self-repair/src/isolation.rs` の unit テスト: allowlist 外の
  環境変数が子プロセスへ継承されないこと（シェル非経由の `printenv` 直接
  起動）・`HOME`/`TMPDIR` の付け替え・専用ディレクトリが `sandbox_root` の
  外側に置かれること・`PATH`/`CARGO_HOME`/`RUSTUP_HOME` のホスト値再注入
  （未設定時の rustup 既定パスへのフォールバックを含む）・argv ラップの
  `--` 区切り・argv ラップと probe が同一の `unshare` フラグを参照すること
  ・probe の fail-closed 挙動
- `crates/self-repair/src/cli.rs` の unit テスト: `--isolate-network` の
  パース・既定 false
- `crates/self-repair/tests/isolate_network_probe.rs`: probe-guarded 統合
  テスト。probe が失敗する環境（本リポジトリの CI・開発コンテナが想定する
  既定）では `--isolate-network` が fail-closed（exit 1）で拒否されること、
  probe が成功する環境では early-return で skip することを実バイナリ起動で
  確認する
- 既存の統合テスト（`cli_run.rs`・`revalidation_bug_fix.rs`・
  `feature_addition_loop_completion_task_3_3c.rs` 等）は無変更で通ることを
  確認済み（`SystemCommandRunner::new()` の互換維持）
