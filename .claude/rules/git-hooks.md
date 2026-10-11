---
paths:
    - "lefthook.yml"
---

# Git hooks（lefthook）編集ルール

## 目的

フックはコミットのたびに実行されるため、遅いフックはそのまま開発速度の低下になる。
フックはローカルでの早期検知専用であり、CI の代替ではない。**開発速度を落とさないことを最優先**し、
重い検査・網羅的な検査は GitHub Actions（`.github/workflows/ci.yml`）に任せる。

本ルールは PR #2985（pre-push の clippy・test 廃止。1 回の push に 7〜13 分かかっていた。2026-10-11 所有者指示）で
`lefthook.yml` 冒頭コメントに記した方針を正式化したものである。

## 時間予算

| フック | 予算 | 判定方法 |
| --- | --- | --- |
| pre-commit | 通常のコミット（staged のみ）で全ジョブ合計 1 秒未満 | 代表的なコミットのファイル集合で `lefthook run pre-commit` の `summary: (done in N seconds)` を確認する |
| commit-msg | 即時（外部コマンド・ネットワーク不可） | 実コミット時に表示される commit-msg の summary |
| pre-push・post-checkout・post-merge 等 | 原則追加しない | 追加する場合は理由と実測値を PR 本文に記載する |

`--all-files` は全追跡ファイル（`.rs` 約 1,500 件）を対象にする最悪ケースで、rustfmt-check により 6 秒前後かかる。
実際のコミットは staged ファイルのみが対象のため、判定には代表的なコミットのファイル集合を使う。

```bash
# 直近コミットのファイル集合で計測する（NUL 区切りで渡す）
git show --name-only --format= HEAD | tr '\n' '\0' | lefthook run pre-commit --files-from-stdin --no-tty
# 最悪ケースの参考値
time lefthook run pre-commit --all-files
```

## フックに置かないもの（→ `ci.yml` へ）

- テスト・ビルド・clippy・rustdoc・cargo-deny（`ci.yml` の `rust-ci` と固有ジョブが PR ごとに必ず実行する）
- workspace 全体を走査する検査（`cargo fmt --all --check`・`make ci` / `make lint` 等の呼び出しを含む）
- 依存クレート・ツールのインストールやネットワークアクセスを伴う処理
- `ci.yml` のジョブと同じコマンドを同じ範囲で実行するもの

手元で CI 相当を確認したい場合は `make ci`（`ci.yml` と同一チェックの一括実行）を使う。
`coding-rust.md` の品質基準（fmt・clippy `-D warnings` を通す）自体は変わらず、フックで強制しないだけである。

## フックに置いてよいもの

- `{staged_files}` に限定し、`glob` / `exclude` で対象を絞った 1 秒未満の検査
- CI ではコミット単位で検知できない、または push 後では遅いもの
    - secrets-guard（秘密情報ファイルの stage 拒否）: push 後では漏えいを取り消せない。CI に同等の検査はない。
        判定は create-commit スキルの `RE_CREDFILE` / `RE_CREDFILE_OK` をベースにした正規表現（サブディレクトリの
        `.env.*`・`id_rsa` / `id_ed25519` を拒否し、`.example` / `.sample` / `.template` / `.pub` は許可）で、
        Fandhe-AI/fandhe-container と同一実装。パターンを変える場合は create-commit スキル側との整合を確認する
    - commit-msg（Conventional Commits 検証）: CI ではコミットメッセージ・PR タイトルを検査していない

### 明示的な例外

- pre-commit の rustfmt-check（staged の `.rs` に対する `rustfmt --check`）は CI の `rust-ci / cargo fmt --check` と
    重複するが、fmt 漏れによる CI 往復の差し戻しを避ける軽量ガードとして維持する（2026-10-11 計測・決定）。
    実測: 1 ファイル 0.10 秒、直近の実コミット 2 件のファイル集合で約 0.5 秒・約 0.8 秒、`--all-files` で 6.0〜6.4 秒
- rustfmt は引数のファイルから `mod` 宣言をたどって子モジュールも検査するため、`lib.rs` / `mod.rs` を多数
    stage すると 1 秒を超えうる（全クレートの `lib.rs` 同時 stage で 1.84 秒）。頻度が低いため許容する

## 書き方の規約

- `pre-commit` の `parallel: true` を維持する（直列化しない）
- `{staged_files}` と `glob` / `exclude` で対象を絞る
- cargo 系ジョブは `Cargo.toml` 不在時・ツール未導入時に skip する既存の書き方（CI の `detect` ステップと同一方針）に合わせる
- rustfmt は `Cargo.toml` の `edition` を明示指定する（未指定だと edition 2015 で検査してしまう。既存ジョブのコメント参照）
- ファイル単位の判定はファイル数に比例してプロセスを起動しない（`for` ループ内で grep 等を起動せず、awk 1 本で一括判定する）。
    判定コマンド自体が失敗した場合は fail-closed で拒否する（secrets-guard の実装を参照）
- 条件付き実行が必要な場合は lefthook の `skip` / `only` を使ってよい
    （`.claude/skills/lefthook/references/examples/skip.md`）

## 検査を追加するときの判断手順

1. その検査は `ci.yml` で既に実施されているか
    - 実施済みで、staged 限定にしても 1 秒以上かかる → フックに入れない
2. staged ファイルに限定できるか
    - できない → CI へ置く
3. CI へ置く場合は `ci.yml` にジョブを追加し、集約ジョブ `ci-complete` の `needs` と ruleset の required contexts を
    `ci.md`「ワークフロー設計」節に従って更新する
4. 編集後に「時間予算」節のコマンドで計測し、summary の合計時間を PR 本文に記載する（通常のコミットで 1 秒以上なら構成を見直す）

## バイパス禁止

遅いからといって `--no-verify` や `LEFTHOOK=0` で回避しない（`conventional-commits.md`）。遅いフックはフック側を直す
（検査を CI へ移す・staged 限定にする・対象 glob を絞る）。
