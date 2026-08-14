#!/usr/bin/env python3
# self-hosted runner 逆戻り防止の fail-closed 契約検査の本体（イシュー #472・PR #626）。
#
# 呼び出し元: scripts/check-workflow-runner-policy.sh（ci.yml の runner-policy ジョブと
# Makefile の runner-policy ターゲットが共用する薄い wrapper）。直接起動は想定しない。
#
# grep/パターン照合ではなく実 YAML パーサー（PyYAML `yaml.safe_load_all`）方式を採る理由:
# PR #626 の codex-review で、テキスト照合ベースの旧実装は YAML 表記トリックで迂回可能
# （エスケープ済みキー `"runs-on"`・複数行 double-quoted キー・フローマッピング内キー・
# 明示キー `?`・引用文字列内 `#` によるコメント切り捨て誤動作・行末の緩い値一致）と
# P0/P1 指摘された。デコード後の構造に対してキー名・値を比較すれば、これらの表記揺れは
# 全てパーサーが正規化するため、パターン増築なしに構造的に遮断できる。
#
# 検査方針（fail-closed。許可リスト方式ではなく「唯一の許容形に一致しなければ違反」）:
#   1. 禁止トークン検査: デコード後の全スカラー文字列（キー・値とも）に `self-hosted` が
#      現れたら違反とする（コメントはパース時点で消えるため、コメント内の言及は
#      自然に検査対象外になる）
#   2. 唯一の許容形検査: キー名が runner 宣言（runs-on / runner-label /
#      post-feedback-runner-label。前後空白を除去して照合）である全マッピングエントリを
#      再帰走査で収集し、値が文字列 `ubuntu-latest` の完全一致でなければ違反とする
#      （larger runner 名・式展開 `${{ ... }}`・配列・group 指定・null はすべて違反側）
#   3. 構造の fail-closed: YAML パース失敗・ドキュメントがマッピングでない・対象
#      ディレクトリ不在・対象ファイル 0 件は、スキップせず全て失敗として扱う
#      （検査の空振りで green にならない設計）
#
# 例外の扱い（.claude/rules/ci.md「codex-review」節）: codex-review wrapper の codex
# 実行ジョブは runner-label を非指定とし reusable workflow の既定値 `codex`
# （self-hosted な codex 専用 runner）へ委譲する設計のため、本リポ側ファイルに runner
# 宣言が現れず本検査の対象外となる（ファイル名の除外ハードコードは持たない）。将来
# `codex` 等のラベルを明示する場合は、本スクリプトの許容形の意図的な拡張（レビュー必須）
# を要する。実機（CUDA/Metal）ジョブの将来的な例外追加（ci.md「実機依存」節）も同様。
#
# 終了コード: 0 = 適合、1 = 違反検出、2 = 使用方法・実行環境の異常（いずれも非 0 で
# CI は fail する）。
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # fail-closed: PyYAML 不在で検査をスキップして成功にしない。GitHub ホステッド
    # ランナー（ubuntu-latest）には標準搭載のため、これはローカル環境の導入漏れ。
    print(
        "::error::PyYAML が見つかりません。runner 契約検査は実行できないため失敗とします"
        "（fail-closed）。`pip3 install pyyaml` で導入してください。",
        file=sys.stderr,
    )
    sys.exit(2)

# 唯一の許容形: runner 宣言の値はこの文字列との完全一致のみ許す
# （`ubuntu-latest-8-cores` 等の larger runner は前方一致でも不許可。ci.md「runner」節）。
ALLOWED_RUNNER_VALUE = "ubuntu-latest"
# runner 宣言と見なすキー名（runs-on に加え、reusable workflow への runner 指定入力
# runner-label / post-feedback-runner-label 経由の逆戻りも検知対象とする）。
RUNNER_KEYS = {"runs-on", "runner-label", "post-feedback-runner-label"}
# 非コメント位置（デコード後のスカラー）に現れてはならない禁止トークン。
BANNED_TOKEN = "self-hosted"


def walk(node, path, violations, visited):
    """デコード後の YAML 構造を再帰走査し、違反を violations へ収集する。

    キー・値の双方を走査対象とする（複合キー・キー内文字列の禁止トークンも検知）。
    アンカー/エイリアスで循環構造が作られても無限再帰しないよう、走査済み
    コンテナは id で記録してスキップする。
    """
    if isinstance(node, (dict, list)):
        if id(node) in visited:
            return
        visited.add(id(node))

    if isinstance(node, dict):
        for key, value in node.items():
            key_label = key if isinstance(key, str) else repr(key)
            child_path = f"{path}.{key_label}" if path else str(key_label)
            if isinstance(key, str) and key.strip() in RUNNER_KEYS:
                if not (isinstance(value, str) and value == ALLOWED_RUNNER_VALUE):
                    violations.append(
                        f"{child_path}: runner 宣言の値が唯一の許容形"
                        f"（文字列 {ALLOWED_RUNNER_VALUE!r} 完全一致）ではありません: {value!r}"
                    )
            # キー自体も走査する（文字列キー内の禁止トークン・複合キーの検知）。
            walk(key, f"{child_path}(key)", violations, visited)
            walk(value, child_path, violations, visited)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            walk(item, f"{path}[{index}]", violations, visited)
    elif isinstance(node, str):
        if BANNED_TOKEN in node:
            violations.append(
                f"{path}: 禁止トークン {BANNED_TOKEN!r} を含む文字列です: {node!r}"
            )


def check_file(file_path):
    """1 ファイルを検査し、違反メッセージのリストを返す。パース不能・想定外構造も
    違反として返す（fail-closed）。"""
    violations = []
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except OSError as error:
        return [f"読み取りに失敗しました（fail-closed）: {error}"]

    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError as error:
        return [f"YAML としてパースできません（fail-closed）: {error}"]

    if not documents:
        return ["YAML ドキュメントが 0 件です（fail-closed: 空ファイルを違反として扱う）"]

    for doc_index, document in enumerate(documents):
        if not isinstance(document, dict):
            violations.append(
                f"doc[{doc_index}]: トップレベルがマッピングではありません"
                f"（fail-closed: 想定外構造を違反として扱う）: {type(document).__name__}"
            )
            continue
        walk(document, f"doc[{doc_index}]", violations, set())
    return violations


def report(label, violations):
    """1 ファイル分の判定結果を出力し、適合なら True を返す。"""
    if violations:
        print(
            f"::error::{label} が runner 契約"
            f"（{ALLOWED_RUNNER_VALUE} 完全一致・{BANNED_TOKEN} 不在。"
            ".claude/rules/ci.md「runner」節）に違反しています:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return False
    print(f"OK: {label} は runner 契約（{ALLOWED_RUNNER_VALUE} 完全一致・{BANNED_TOKEN} 不在）に適合")
    return True


def cmd_check(workflow_dir):
    """`.github/workflows/` 配下の *.yml・*.yaml を全件検査する（check サブコマンド本体）。"""
    directory = Path(workflow_dir)
    if not directory.is_dir():
        print(
            f"::error::{workflow_dir} が見つかりません"
            "（fail-closed: 検査対象ディレクトリの消失を違反として扱う）",
            file=sys.stderr,
        )
        return 1

    # BSD/GNU の sort 差異（macOS の sort -z 非対応で make ci が落ちる問題）を避け、
    # 列挙・整列とも Python 側で完結させる。
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in (".yml", ".yaml")
    )
    if not files:
        print(
            f"::error::{workflow_dir} に *.yml/*.yaml が見つかりません"
            "（fail-closed: 対象 0 件を違反として扱う）",
            file=sys.stderr,
        )
        return 1

    failed = False
    for path in files:
        if not report(str(path), check_file(path)):
            failed = True
    if failed:
        return 1
    print(f"OK: {len(files)} 件の workflow ファイルすべてが runner 契約に適合")
    return 0


def cmd_check_file(file_path):
    """単一ファイルを検査する（self-test が fixture 1 件ずつの判定に使う）。"""
    return 0 if report(file_path, check_file(file_path)) else 1


def main(argv):
    if len(argv) == 2 and argv[1] == "check":
        return cmd_check(".github/workflows")
    if len(argv) == 3 and argv[1] == "check-file":
        return cmd_check_file(argv[2])
    print(f"usage: {argv[0]} {{check|check-file <path>}}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
