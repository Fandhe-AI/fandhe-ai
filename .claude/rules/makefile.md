---
paths:
    - "Makefile"
    - "scripts/**"
---

# Makefile と scripts/ の編集ルール

## 目的

`Makefile` は開発・CI 共通コマンドの**薄い入口**であり、処理の定義は `scripts/` に置く。
ターゲットの数が多く（`make help` で一覧）、CI（`ci.yml`）・lefthook・README・docs が呼び出し元になっているため、
追加・改名・削除は呼び出し元と一緒に更新する。詳細な設計指針は make skill（`.claude/skills/make`）を参照する。
`scripts/hooks/**` のような git hooks 専用スクリプトを追加する場合は `git-hooks.md`（`lefthook.yml` 編集時に読み込まれる）に従う。

## 基本方針

- レシピは `@scripts/<name>.sh [args]` の **1 行呼び出し**に限定する。複数行のシェル・`if` / `for` / `case`・`curl`・環境依存の処理は `scripts/` へ移す
- `cargo fmt --all` のように言語ネイティブの 1 行コマンドで済むものだけは scripts/ に包まず Makefile に直接書いてよい。無理に全部を包まない
- 空ターゲット（`@true` 等）は置かない。呼び出し元が終了コードを信用してしまうため
- GNU Make 3.81（macOS 標準）で動くこと。`.ONESHELL`・`.RECIPEPREFIX`・`.WAIT` 等の 3.82+ 専用機能を使わない
- `.PHONY` に宣言し、`.DEFAULT_GOAL := help` を維持する。ユーザー入力（`BACKEND` 等）を `$(VAR)` でレシピへ展開しない（シェル注入になる）。`export` して scripts 側で検証する（`scripts/bench-harness.sh` 参照）

## ターゲットの追加・改名・削除の手順

1. `Makefile` に `.PHONY` を追加し、`## 説明` コメントを付ける（`make help` は `## ` コメントから生成されるため、無いと表示されない）
2. `scripts/help.sh`（Make 非依存の一覧）を同じ変更で更新する。両者の自動同期は無い
3. 呼び出し元を洗い出して更新する: `grep -rn "make " .github .claude docs README.md CLAUDE.md AGENTS.md lefthook.yml scripts`
4. README の make ターゲット表と `CLAUDE.md` のリポジトリ構造（scripts/ 一覧）を更新する
5. `make help` と `make -n <target>` で確認し、`shellcheck -x scripts/*.sh` を通す
6. **既存ターゲット名・引数・環境変数・終了コードは維持する**（CI・lefthook・docs が参照）。改名する場合は呼び出し元を同じ PR で更新する

## scripts/ の規約

- 先頭は `#!/usr/bin/env bash` と `set -euo pipefail`。共通ヘルパーは `scripts/lib.sh`（source するとリポジトリルートへ `cd` する）を使う
- 実行権限を付与する（`chmod +x`）。`shellcheck` を通す
- 引数を取らないスクリプトは `reject_args` で未知の入力を終了コード 2 で拒否する
- エラー・診断メッセージは stderr に出す。出力言語は既存スクリプトと `AGENTS.md` の規約に従う
- ci.yml のジョブと共用する検査は、ロジックを `scripts/` に 1 つだけ置き、CI と Makefile の双方から同じコマンドで呼ぶ（二重定義しない）

## ターゲットの契約

- `fmt-check` / `lint` / `deny` / `deps-forbidden` / `runner-policy` 等の検査系は**ソースを変更しない**。整形は `fmt` だけが行う
- 実機専用の `test-ignored*` は CI では実行しない（Linux ホステッドは実機非搭載）。`--all-features` を外さない（required-features ゲートのテストが暗黙にスキップされる false-green を避ける）
- `ci` は `.github/workflows/ci.yml` と同一チェックの一括実行であり、ci.yml にチェックを足したら `ci` の依存にも足す
- 導入系（`setup` / `rustup` / `hooks`）は冪等にする。再実行しても壊れない
