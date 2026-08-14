#!/usr/bin/env python3
# self-hosted runner 逆戻り防止の fail-closed 契約検査の本体（イシュー #472・PR #626）。
#
# 呼び出し元: scripts/check-workflow-runner-policy.sh（ci.yml の runner-policy ジョブと
# Makefile の runner-policy ターゲットが共用する薄い wrapper）。直接起動は想定しない。
#
# grep/パターン照合ではなく実 YAML パーサー方式を採る理由:
# PR #626 の codex-review で、テキスト照合ベースの旧実装は YAML 表記トリックで迂回可能
# （エスケープ済みキー `"runs-on"`・複数行 double-quoted キー・フローマッピング内キー・
# 明示キー `?`・引用文字列内 `#` によるコメント切り捨て誤動作・行末の緩い値一致）と
# P0/P1 指摘された。デコード後の構造に対してキー名・値を比較すれば、これらの表記揺れは
# 全てパーサーが正規化するため、パターン増築なしに構造的に遮断できる。
#
# パーサー実装を PyYAML ではなく Python 標準ライブラリのみの自前実装にした理由（#472
# 再開時の追加指摘）: 当初 PyYAML 採用で実装したが、その後の codex-review で
# 「runner-policy ジョブ・Makefile が PyYAML を必須とするのにバージョン固定・導入
# ステップ・docs/license-matrix.md 更新・ユーザー承認の記録がない」と P1 指摘された。
# 依存の追加・更新はユーザー承認必須（.claude/rules/deps-policy.md・CLAUDE.md
# Conventions）だが、本検査は自律運転（ユーザー承認待ち不可）の中断作業を引き継ぐ形で
# 完成させる必要があり、承認を得られない以上は新規依存を前提にできない。そのため
# 「許容形リストに一致しなければ違反」という fail-closed 方針をパーサー自体にも適用し、
# 対応できない構文（アンカー・エイリアス・タグ・tab インデント・複合キー・複数
# ドキュメント区切り `---` 等）は例外を送出して違反側へ倒す設計にすることで、
# 依存を増やさずに PR #626 で指摘された表記トリック（下記 scripts/testdata/
# workflow-runner-trick-*.yml fixture）を構造的に遮断する。GitHub Actions workflow の
# YAML はブロックマッピング・ブロックシーケンス・フローマッピング／シーケンス・
# 単一／二重引用スカラー・ブロックスカラー（`|`／`>`）・明示キー（`?`）の範囲に収まる
# ため、この範囲に閉じたパーサーで実用上十分と判断した（本ファイル冒頭コメントの
# 対応範囲を参照）。
#
# 検査方針（fail-closed。許可リスト方式ではなく「許容形リストに一致しなければ違反」）:
#   1. 禁止トークン検査: デコード後の全スカラー文字列（キー・値とも）に `self-hosted` が
#      現れたら違反とする（コメントはパース時点で消えるため、コメント内の言及は
#      自然に検査対象外になる）
#   2. 許容形検査: キー名が runner 宣言（runs-on / runner-label /
#      post-feedback-runner-label。前後空白を除去して照合）である全マッピングエントリを
#      再帰走査で収集し、値が ALLOWED_RUNNER_VALUES（標準 GitHub ホステッドランナーの
#      `-latest` ラベル集合。下記定数コメント参照）のいずれかとの完全一致でなければ
#      違反とする（larger runner 名・式展開 `${{ ... }}`・配列・group 指定・null は
#      すべて違反側）
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


class YamlSubsetError(Exception):
    """本パーサーが対応する YAML 部分集合の範囲外・構文エラーを表す（fail-closed）。

    「パースできない・対応範囲外」は握り潰さず必ずこの例外にして呼び出し元へ伝播させる。
    検査対象は信頼できない外部入力（workflow ファイル）ではなく本リポ自身の
    .github/workflows/ だが、パーサーの取りこぼしが違反の見逃しに直結するため、
    未対応構文は常に「違反として扱う」側に倒す（許可リスト方式の裏返し）。
    """


class _Parser:
    """GitHub Actions workflow YAML の範囲に閉じた再帰下降パーサー。

    対応: ブロックマッピング／シーケンス・フローマッピング／シーケンス・単一／二重
    引用スカラー（エスケープ・複数行折り畳み・行継続 `\\` 込み）・平坦スカラー・
    ブロックスカラー（`|`／`>`。フォールド規則は簡略化し改行を保持する。禁止トークン
    検出という用途では折り畳みの正確な再現は不要なため）・明示キー（`?` / `:`）。
    非対応（fail-closed で例外にする）: アンカー／エイリアス（`&`／`*`）・タグ（`!`）・
    複合キー（マッピング／シーケンスをキーにする）・複数ドキュメント区切り `---`・
    tab によるインデント・ネストしたインラインシーケンス（`- - x`）。
    いずれも本リポの .github/workflows/*.yml では使用されていないことを実装時に
    grep で確認済み（trick fixture 群・実 workflow 双方で self-test 済み）。
    """

    def __init__(self, text):
        self.text = text
        self.n = len(text)
        self.pos = 0
        self.line = 1
        self.line_start = 0

    # ---- 文字レベルの基本操作 ----

    def eof(self):
        return self.pos >= self.n

    def peekc(self, offset=0):
        idx = self.pos + offset
        return self.text[idx] if 0 <= idx < self.n else ""

    def getc(self):
        if self.pos >= self.n:
            raise YamlSubsetError("予期しない EOF です（fail-closed）")
        c = self.text[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.line_start = self.pos
        return c

    @property
    def col(self):
        return self.pos - self.line_start

    def skip_inline_spaces(self):
        while self.peekc() in (" ", "\t"):
            self.getc()

    def skip_to_eol(self):
        while self.peekc() not in ("", "\n"):
            self.getc()
        if self.peekc() == "\n":
            self.getc()

    def peek_indent(self):
        """現在位置から続く行頭スペース数を消費せず数える（tab 混入は fail-closed）。"""
        j = 0
        while self.peekc(j) == " ":
            j += 1
        if self.peekc(j) == "\t":
            raise YamlSubsetError(f"インデントに tab は使用できません（行 {self.line}）")
        return j

    def consume_indent(self, n):
        for _ in range(n):
            self.getc()

    def expect_end_of_line(self):
        """値・キーを読み終えた直後に呼ぶ。残りはコメントか改行/EOF のみを許す。

        これを全スカラー読み取りの終端で徹底することで、次に
        skip_blank_lines_and_comments を呼ぶ時点で常に「行頭（列 0）」に居ることが
        保証される（インデント判定の前提）。
        """
        self.skip_inline_spaces()
        if self.eof():
            return
        c = self.peekc()
        if c == "#":
            self.skip_to_eol()
            return
        if c == "\n":
            self.getc()
            return
        raise YamlSubsetError(f"行末に想定外の文字があります（行 {self.line}）: {c!r}")

    def skip_blank_lines_and_comments(self):
        """空行・コメント専用行を読み飛ばし、内容のある行の行頭（列 0）で止まる。"""
        while True:
            if self.eof():
                return
            j = 0
            while self.peekc(j) == " ":
                j += 1
            c = self.peekc(j)
            if c == "\t":
                raise YamlSubsetError(f"インデントに tab は使用できません（行 {self.line}）")
            if c == "":
                self.consume_indent(j)
                return
            if c == "\n":
                self.consume_indent(j)
                self.getc()
                continue
            if c == "#":
                self.consume_indent(j)
                self.skip_to_eol()
                continue
            return

    # ---- スカラー読み取り ----

    def _read_hex_escape(self, width):
        digits = []
        for _ in range(width):
            c = self.peekc()
            if c == "" or c not in "0123456789abcdefABCDEF":
                raise YamlSubsetError(f"16 進エスケープが不正です（行 {self.line}。fail-closed）")
            digits.append(c)
            self.getc()
        return chr(int("".join(digits), 16))

    def _fold_line_break(self):
        """引用スカラー内の（バックスラッシュなし）改行の折り畳み処理。

        直前で改行 1 個を消費済みの状態で呼ばれる。YAML の flow scalar line folding
        規則（連続する空行の数 N に対し N==0 は空白 1 個・N>=1 は改行 N 個）を簡略に
        再現する。禁止トークン検出用途では正確なチョンピング規則までは不要なため、
        「改行を保持する」側に倒し空白への丸め込みで文字列が偶然結合し検出漏れが
        起きないようにする（fail-closed 志向の簡略化）。
        """
        extra = 0
        while True:
            skip_count = 0
            while self.peekc(skip_count) in (" ", "\t"):
                skip_count += 1
            if self.peekc(skip_count) == "\n":
                self.consume_indent(skip_count)
                self.getc()
                extra += 1
                continue
            self.consume_indent(skip_count)
            break
        return "\n" if extra >= 1 else " "

    def parse_double_quoted(self):
        assert self.peekc() == '"'
        self.getc()
        buf = []
        simple_escapes = {
            '"': '"', "\\": "\\", "/": "/", "0": "\0", "a": "\a", "b": "\b",
            "t": "\t", "n": "\n", "v": "\v", "f": "\f", "r": "\r", "e": "\x1b",
            " ": " ", "N": "", "_": "\xa0", "L": " ", "P": " ",
        }
        while True:
            c = self.peekc()
            if c == "":
                raise YamlSubsetError(f"二重引用符が閉じられていません（行 {self.line}）")
            if c == '"':
                self.getc()
                break
            if c == "\\":
                self.getc()
                e = self.peekc()
                if e == "":
                    raise YamlSubsetError("エスケープシーケンスが不完全です（EOF。fail-closed）")
                if e == "\n":
                    # 行継続（バックスラッシュ + 改行）: 両方を捨て、継続行の行頭空白も
                    # 読み飛ばす（trick-multiline-key fixture が要求する挙動）。
                    self.getc()
                    while self.peekc() in (" ", "\t"):
                        self.getc()
                    continue
                if e in simple_escapes:
                    buf.append(simple_escapes[e])
                    self.getc()
                    continue
                if e == "x":
                    self.getc()
                    buf.append(self._read_hex_escape(2))
                    continue
                if e == "u":
                    self.getc()
                    buf.append(self._read_hex_escape(4))
                    continue
                if e == "U":
                    self.getc()
                    buf.append(self._read_hex_escape(8))
                    continue
                raise YamlSubsetError(f"未対応のエスケープです（行 {self.line}）: \\{e}（fail-closed）")
            if c == "\n":
                self.getc()
                buf.append(self._fold_line_break())
                continue
            buf.append(c)
            self.getc()
        return "".join(buf)

    def parse_single_quoted(self):
        assert self.peekc() == "'"
        self.getc()
        buf = []
        while True:
            c = self.peekc()
            if c == "":
                raise YamlSubsetError(f"単一引用符が閉じられていません（行 {self.line}）")
            if c == "'":
                self.getc()
                if self.peekc() == "'":
                    buf.append("'")
                    self.getc()
                    continue
                break
            if c == "\n":
                self.getc()
                buf.append(self._fold_line_break())
                continue
            buf.append(c)
            self.getc()
        return "".join(buf)

    def parse_plain_scalar_key(self):
        """ブロック／フロー双方で使うキー用の平坦スカラー読み取り。

        `:` の直後が空白・タブ・改行・EOF（またはフロー終端記号）であれば、その
        `:` をキーの終端とみなす（YAML のマッピングキー終端規則）。
        """
        buf = []
        while True:
            c = self.peekc()
            if c in ("", "\n"):
                break
            if c == ":":
                nxt = self.peekc(1)
                if nxt in (" ", "\t", "\n", "", ",", "}", "]"):
                    break
            if c in (",", "}", "]"):
                break
            if c == "#" and buf and buf[-1] in (" ", "\t"):
                break
            buf.append(c)
            self.getc()
        key = "".join(buf).strip()
        if key == "":
            raise YamlSubsetError(f"空のキーです（行 {self.line}）")
        return key

    def parse_plain_scalar_value_block(self):
        """ブロックコンテキストでの値の平坦スカラー読み取り（行末まで。内部の `:` は
        終端にしない。空白 + `#` はコメント開始として扱う。実 YAML と同一挙動で、
        引用符で囲われていない値中の `#` は文字列内でもコメント化される
        （clean fixture の `run: echo "prefix # not-a-comment"` が意図的に検証する
        挙動。PyYAML でも同様に truncate されることを実装時に確認済み）。"""
        buf = []
        while True:
            c = self.peekc()
            if c in ("", "\n"):
                break
            if c in (" ", "\t"):
                j = 0
                while self.peekc(j) in (" ", "\t"):
                    j += 1
                if self.peekc(j) == "#":
                    break
                buf.append(c)
                self.getc()
                continue
            buf.append(c)
            self.getc()
        return "".join(buf).rstrip()

    def parse_plain_scalar_value_flow(self):
        """フローコンテキスト（`{}`／`[]` 内）での値の平坦スカラー読み取り。
        `,`／`}`／`]` およびコメントで終端し、改行は折り畳んで継続する。"""
        buf = []
        while True:
            c = self.peekc()
            if c in ("", ",", "}", "]"):
                break
            if c == "\n":
                self.getc()
                buf.append(self._fold_line_break())
                continue
            if c in (" ", "\t"):
                j = 0
                while self.peekc(j) in (" ", "\t"):
                    j += 1
                nxt = self.peekc(j)
                if nxt == "#":
                    self.consume_indent(j)
                    self.skip_to_eol()
                    break
                if nxt in ("", ",", "}", "]", "\n"):
                    self.consume_indent(j)
                    continue
                buf.append(c)
                self.getc()
                continue
            buf.append(c)
            self.getc()
        return "".join(buf).strip()

    def _looks_like_mapping_key(self):
        """先読みのみ（呼び出し前後で位置を復元する）。`- key: value` 圧縮マッピングと
        単なる平坦スカラー項目を判別する。

        プレーンスカラーは quote-blind な走査で十分（プレーンキーの中に
        `: `（コロン＋空白等）が現れた時点で、それ自体が YAML 上そこでキーが
        終端する規則のため）。一方クォート済み項目（`- "hello: world"` 等）は
        文字列本体に `: ` を含みうるため、quote-blind 走査のままだと
        シーケンス項目を圧縮マッピングと誤認し、後続の `expect_end_of_line` が
        `"key"` 直後の余り文字列を拒否する fail-closed 誤検知を起こす
        （cursor[bot] review #4936090383 指摘・PR #626）。クォート開始の場合は
        既存の `parse_key_token`（`parse_double_quoted`／`parse_single_quoted`。
        エスケープ・行継続・折り畳みを正しく扱う）で実際に 1 トークン読み切って
        から直後が `: ` 終端かを確認し、判定後に位置を必ず復元する。"""
        c = self.peekc()
        if c in ('"', "'"):
            saved_pos, saved_line, saved_line_start = self.pos, self.line, self.line_start
            try:
                try:
                    self.parse_key_token()
                except YamlSubsetError:
                    return False
                self.skip_inline_spaces()
                return self.peekc() == ":" and self.peekc(1) in (" ", "\t", "\n", "")
            finally:
                self.pos, self.line, self.line_start = saved_pos, saved_line, saved_line_start
        i = self.pos
        while True:
            c = self.text[i] if i < self.n else ""
            if c in ("", "\n"):
                return False
            if c == ":":
                nxt = self.text[i + 1] if i + 1 < self.n else ""
                if nxt in (" ", "\t", "\n", ""):
                    return True
            if c == "#" and i > self.pos and self.text[i - 1] in (" ", "\t"):
                return False
            i += 1

    def parse_key_token(self):
        c = self.peekc()
        if c == '"':
            return self.parse_double_quoted()
        if c == "'":
            return self.parse_single_quoted()
        if c in ("{", "["):
            raise YamlSubsetError(
                f"複合キー（マッピング/シーケンス）は非対応です（行 {self.line}。fail-closed）"
            )
        return self.parse_plain_scalar_key()

    # ---- ブロックスカラー（`|`／`>`） ----

    def parse_block_scalar(self, indent):
        """ブロックスカラーのヘッダ行（`|`／`>` + チョンピング/明示インデント指示子）
        を消費した後、`indent` より深いインデントの行を本文として集約する。
        フォールド（`>`）と リテラル（`|`）の折り畳み差は区別せず改行を保持する
        （本ファイルクラス docstring 参照。禁止トークン検出には過剰厳密さは不要）。
        """
        self.getc()  # '|' または '>'
        while self.peekc() in ("-", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
            self.getc()
        self.skip_inline_spaces()
        if self.peekc() == "#":
            self.skip_to_eol()
        elif self.peekc() == "\n":
            self.getc()
        elif self.peekc() != "":
            raise YamlSubsetError(
                f"ブロックスカラーのヘッダ後に想定外の文字があります（行 {self.line}）"
            )

        content_indent = None
        lines = []
        while not self.eof():
            j = 0
            while self.peekc(j) == " ":
                j += 1
            nxt = self.peekc(j)
            is_blank = nxt in ("\n", "")
            if not is_blank:
                if content_indent is None:
                    if j <= indent:
                        break
                    content_indent = j
                elif j < content_indent:
                    break
            strip = content_indent if content_indent is not None else j
            line_end = self.text.find("\n", self.pos)
            raw = self.text[self.pos : line_end if line_end != -1 else self.n]
            content = raw[strip:] if not is_blank else ""
            lines.append(content)
            if line_end == -1:
                self.pos = self.n
                break
            self.pos = line_end + 1
            self.line += 1
            self.line_start = self.pos
        return "\n".join(lines)

    # ---- マッピングへの格納（フロー形式・ブロック形式共用） ----

    def store_mapping_entry(self, result, key, value):
        """マッピングへ 1 エントリを格納する。キー重複は fail-closed で拒否する。

        重複キーを後勝ち（`result[key] = value` の素朴な上書き）で格納すると、
        同一マッピングに `runs-on: self-hosted` と `runs-on: ubuntu-latest` を
        この順で書くだけで禁止宣言が消え、検査が許容値しか見ない fail-open に
        なる（PR #626 codex-review P0 指摘）。GitHub Actions 側がどちらの値を
        採用するかに依存せず、重複そのものを YamlSubsetError で違反側へ倒す。
        引用・エスケープ表記のキー（`"runs-on"` 等）は parse_key_token の時点で
        同一文字列へデコード済みのため、ここでの照合は walk の RUNNER_KEYS 照合と
        同じ前後空白除去後の文字列比較で行う（正規化後に衝突する表記揺れも拒否）。
        """
        normalized = key.strip()
        for existing_key in result:
            if existing_key.strip() == normalized:
                raise YamlSubsetError(
                    f"キーが重複しています（行 {self.line}。fail-closed: "
                    f"後勝ち格納による禁止宣言の上書き迂回を拒否）: {key!r}"
                )
        result[key] = value

    # ---- フローコレクション ----

    def skip_flow_ws_and_comments(self):
        while True:
            c = self.peekc()
            if c in (" ", "\t", "\n"):
                self.getc()
                continue
            if c == "#":
                self.skip_to_eol()
                continue
            break

    def parse_flow_value(self):
        c = self.peekc()
        if c == "{":
            return self.parse_flow_mapping()
        if c == "[":
            return self.parse_flow_sequence()
        if c == '"':
            return self.parse_double_quoted()
        if c == "'":
            return self.parse_single_quoted()
        return self.parse_plain_scalar_value_flow()

    def parse_flow_mapping(self):
        assert self.peekc() == "{"
        self.getc()
        result = {}
        self.skip_flow_ws_and_comments()
        if self.peekc() == "}":
            self.getc()
            return result
        while True:
            self.skip_flow_ws_and_comments()
            if self.peekc() == "?":
                self.getc()
                self.skip_flow_ws_and_comments()
                key = self.parse_key_token()
                self.skip_flow_ws_and_comments()
                if self.peekc() == ":":
                    self.getc()
                    self.skip_flow_ws_and_comments()
                    value = self.parse_flow_value()
                else:
                    value = None
            else:
                key = self.parse_key_token()
                self.skip_flow_ws_and_comments()
                if self.peekc() != ":":
                    raise YamlSubsetError(
                        f"フローマッピングで ':' が見つかりません（行 {self.line}）"
                    )
                self.getc()
                self.skip_flow_ws_and_comments()
                value = self.parse_flow_value()
            self.store_mapping_entry(result, key, value)
            self.skip_flow_ws_and_comments()
            c = self.peekc()
            if c == ",":
                self.getc()
                self.skip_flow_ws_and_comments()
                if self.peekc() == "}":
                    self.getc()
                    return result
                continue
            if c == "}":
                self.getc()
                return result
            raise YamlSubsetError(f"フローマッピングの構文が不正です（行 {self.line}）: {c!r}")

    def parse_flow_sequence(self):
        assert self.peekc() == "["
        self.getc()
        result = []
        self.skip_flow_ws_and_comments()
        if self.peekc() == "]":
            self.getc()
            return result
        while True:
            self.skip_flow_ws_and_comments()
            result.append(self.parse_flow_value())
            self.skip_flow_ws_and_comments()
            c = self.peekc()
            if c == ",":
                self.getc()
                self.skip_flow_ws_and_comments()
                if self.peekc() == "]":
                    self.getc()
                    return result
                continue
            if c == "]":
                self.getc()
                return result
            raise YamlSubsetError(f"フローシーケンスの構文が不正です（行 {self.line}）: {c!r}")

    # ---- ブロックマッピング／シーケンス ----

    def parse_mapping_entry_into(self, result, indent):
        """カーソルがキー開始位置にある状態で 1 エントリを読み、result へ格納する。
        `indent` はこのマッピングの列位置（`?`/`:` 対応行の照合・圧縮記法双方で使う
        「論理的な列」であり、必ずしも物理行の先頭ではない。呼び出し元
        parse_block_mapping の docstring を参照。"""
        if self.peekc() == "?":
            self.getc()
            self.skip_inline_spaces()
            key = self.parse_key_token()
            self.expect_end_of_line()
            self.skip_blank_lines_and_comments()
            if self.eof():
                raise YamlSubsetError("明示キーに対応する ':' がありません（fail-closed）")
            cur_indent = self.peek_indent()
            if cur_indent != indent:
                raise YamlSubsetError(
                    f"明示キーに対応する ':' のインデントが不正です（行 {self.line}）"
                )
            self.consume_indent(cur_indent)
            if self.peekc() != ":":
                raise YamlSubsetError(f"明示キーの後に ':' が必要です（行 {self.line}）")
            self.getc()
            value = self.parse_value_after_colon(indent)
        else:
            key = self.parse_key_token()
            self.skip_inline_spaces()
            if self.peekc() != ":":
                raise YamlSubsetError(f"':' が見つかりません（行 {self.line}）")
            self.getc()
            value = self.parse_value_after_colon(indent)
        self.store_mapping_entry(result, key, value)

    def parse_block_mapping(self, indent):
        """`indent` 列でのブロックマッピングを読む。呼び出し元は 2 種類ある:
        (a) parse_node_body から、物理行の先頭（列 0 からインデント消費済み）で呼ばれる
            通常のケース、
        (b) parse_inline_node から、`- key: value` 圧縮記法の 1 行目の途中（`- ` の
            直後の列）で呼ばれるケース。
        いずれも「現在位置が 1 エントリ目のキー開始位置であり、以降のエントリは
        列 `indent` から始まる」という不変条件だけに依存するため、同一実装で両対応する。
        """
        result = {}
        self.parse_mapping_entry_into(result, indent)
        while True:
            self.skip_blank_lines_and_comments()
            if self.eof():
                break
            cur_indent = self.peek_indent()
            if cur_indent != indent:
                break
            self.consume_indent(cur_indent)
            self.parse_mapping_entry_into(result, indent)
        return result

    def parse_block_sequence(self, indent):
        """`indent` 列でのブロックシーケンスを読む。呼び出し元 parse_node_body は
        ディスパッチ判定のために既に `indent` 分のインデントを消費済みでカーソルは
        1 項目目の `-` の位置にある。parse_block_mapping と同じ理由（不変条件は
        「現在位置が 1 項目目の開始位置であり、以降の項目は列 indent から始まる」）
        により、1 項目目だけは skip_blank_lines_and_comments／インデント再検査を
        経ずに直接読む。"""
        result = [self.parse_sequence_item(indent)]
        while True:
            self.skip_blank_lines_and_comments()
            if self.eof():
                break
            cur_indent = self.peek_indent()
            if cur_indent != indent:
                break
            self.consume_indent(cur_indent)
            result.append(self.parse_sequence_item(indent))
        return result

    def parse_sequence_item(self, indent):
        """カーソルが `-` の位置にある状態で 1 項目を読む。

        `- ` の直後がインデント付きネストブロック（マッピング／シーケンス）で
        終わる場合、`-` 単独行（末尾が改行/EOF/コメント）と `- ` 空白付き行
        （空白の後に改行/EOF/コメント）は同義でなければならない（YAML の
        ブロックシーケンス仕様）。旧実装は `-` 単独行を無条件に null item と
        みなし `parse_optional_nested_block` を呼ばずに return していたため、
        `-` 単独行の直後にインデントされたマッピングが続く正当な workflow
        YAML を誤って fail-closed で拒否していた（Cursor Bugbot 指摘
        876ad41b-460f-4686-8abc-0706148f93e4・PR #626）。`skip_inline_spaces`
        は空白が無くても no-op のため、両ケースを同一分岐に統合して対称に
        扱う。
        """
        if not (self.peekc() == "-" and self.peekc(1) in (" ", "\n", "")):
            raise YamlSubsetError(f"シーケンス項目 '-' が見つかりません（行 {self.line}）")
        self.getc()
        self.skip_inline_spaces()
        c = self.peekc()
        if c in ("", "\n") or c == "#":
            self.expect_end_of_line()
            return self.parse_optional_nested_block(indent)
        start_col = self.col
        return self.parse_inline_node(start_col)

    def parse_inline_node(self, col):
        """`- ` の直後、同一行に続くノードを読む（圧縮マッピング記法が主用途）。"""
        c = self.peekc()
        if c == "-" and self.peekc(1) in (" ", "\n", ""):
            raise YamlSubsetError(
                f"ネストしたインラインシーケンス（- - ...）は非対応です（行 {self.line}。fail-closed）"
            )
        if c == "{":
            v = self.parse_flow_mapping()
            self.expect_end_of_line()
            return v
        if c == "[":
            v = self.parse_flow_sequence()
            self.expect_end_of_line()
            return v
        if c in ("|", ">"):
            return self.parse_block_scalar(col)
        # `- "key": value` のようなクォート済みキーの圧縮マッピングを、単なる
        # クォート済みスカラー値より先に判定する。クォート分岐より後段に置くと
        # `"key"` の時点で値として消費されてしまい、後続の `: value` が
        # `expect_end_of_line` で拒否される（fail-closed 誤検知。
        # cursor[bot] review #4935626935 指摘・PR #626）。`_looks_like_mapping_key`
        # 自体はクォート済みトークンを実際に読み切ってから判定するため、
        # `- "hello: world"` のような非マッピングのクォート済みシーケンス項目を
        # 誤ってマッピングと判定することはない（cursor[bot] review #4936090383
        # 指摘・同 PR で修正）。
        if c == "?" or self._looks_like_mapping_key():
            return self.parse_block_mapping(col)
        if c == '"':
            v = self.parse_double_quoted()
            self.expect_end_of_line()
            return v
        if c == "'":
            v = self.parse_single_quoted()
            self.expect_end_of_line()
            return v
        v = self.parse_plain_scalar_value_block()
        self.expect_end_of_line()
        return v

    def parse_value_after_colon(self, indent):
        """`key:` の直後（同一行）を見て値を判定する。何もなければ次行以降の
        ネストしたブロック（マッピング／シーケンス）か null。"""
        self.skip_inline_spaces()
        c = self.peekc()
        if c in ("", "\n", "#"):
            self.expect_end_of_line()
            return self.parse_optional_nested_block(indent)
        if c == '"':
            v = self.parse_double_quoted()
            self.expect_end_of_line()
            return v
        if c == "'":
            v = self.parse_single_quoted()
            self.expect_end_of_line()
            return v
        if c == "{":
            v = self.parse_flow_mapping()
            self.expect_end_of_line()
            return v
        if c == "[":
            v = self.parse_flow_sequence()
            self.expect_end_of_line()
            return v
        if c in ("|", ">"):
            return self.parse_block_scalar(indent)
        v = self.parse_plain_scalar_value_block()
        self.expect_end_of_line()
        return v

    def parse_optional_nested_block(self, indent):
        self.skip_blank_lines_and_comments()
        if self.eof():
            return None
        nxt_indent = self.peek_indent()
        if nxt_indent <= indent:
            return None
        self.consume_indent(nxt_indent)
        return self.parse_node_body(nxt_indent)

    def parse_node_body(self, indent):
        """カーソルが列 `indent` のノード先頭にある状態でディスパッチする。

        `key:` の次行に、ブロックシーケンス／フローコレクション／ブロックスカラー
        いずれでもないインデント付き内容が続く場合、それはネストしたブロック
        マッピングとは限らない。YAML はインデント継続だけで複数行にまたがる
        プレーンスカラー（明示的な `|`／`>` インジケータなし）を許すため、
        `key:` の値が単なるプレーン／クォート済みスカラーであるケースが存在する
        （例: `runs-on:` の次行に単独で `ubuntu-latest`）。旧実装は `-`／`{`／
        `[`／`|`／`>` 以外を無条件に `parse_block_mapping` へディスパッチして
        おり、`_looks_like_mapping_key`（`- key: value` 圧縮記法の判別で既に
        使っている「次に `:` + 空白/改行/EOF が続くか」の先読み）を経由しない
        ため、この形のスカラー値が `parse_mapping_entry_into` 内で `':' が
        見つかりません` として fail-closed 誤検知していた（cursor[bot] 指摘
        review 4936370341・PR #626。`- ` 圧縮記法側は既に同じ先読みで判別済み
        だったため非対称だった）。`parse_inline_node` と同じ先読みロジックを
        ここでも使い、マッピングかスカラーかを事前に判別することで対称にする。
        """
        c = self.peekc()
        if c == "":
            raise YamlSubsetError("ノードの内容がありません（fail-closed）")
        if c == "-" and self.peekc(1) in (" ", "\n", ""):
            return self.parse_block_sequence(indent)
        if c == "{":
            v = self.parse_flow_mapping()
            self.expect_end_of_line()
            return v
        if c == "[":
            v = self.parse_flow_sequence()
            self.expect_end_of_line()
            return v
        if c in ("|", ">"):
            return self.parse_block_scalar(indent)
        if c == "?" or self._looks_like_mapping_key():
            return self.parse_block_mapping(indent)
        if c == '"':
            v = self.parse_double_quoted()
            self.expect_end_of_line()
            return v
        if c == "'":
            v = self.parse_single_quoted()
            self.expect_end_of_line()
            return v
        v = self.parse_plain_scalar_value_block()
        self.expect_end_of_line()
        return v

    def parse_document(self):
        self.skip_blank_lines_and_comments()
        if self.eof():
            return None
        indent = self.peek_indent()
        self.consume_indent(indent)
        node = self.parse_node_body(indent)
        self.skip_blank_lines_and_comments()
        if not self.eof():
            raise YamlSubsetError(
                f"未消費のデータが残っています（行 {self.line}。fail-closed: "
                "複数ドキュメント区切り `---` 等の非対応構文の可能性があります）"
            )
        return node


def load_all(text):
    """`yaml.safe_load_all` 相当の最小インターフェース。本パーサーは単一ドキュメントの
    みサポートするため、内容があれば長さ 1 のリスト、空なら空リストを返す
    （呼び出し元 check_file の「ドキュメント 0 件」判定はそのまま機能する）。"""
    doc = _Parser(text).parse_document()
    return [] if doc is None else [doc]


# 許容形: runner 宣言の値はこの集合のいずれかとの完全一致のみ許す
# （`ubuntu-latest-8-cores` 等の larger runner は前方一致でも不許可。ci.md「runner」節）。
# 標準 GitHub ホステッドランナーの `-latest` ラベルに限定する（PR #626 の codex-review
# P1 指摘: 旧実装は `ubuntu-latest` 単独の完全一致に限定しており、ci.md 冒頭「runner」節
# が許可する「ubuntu-latest 等の標準スペック」〈例: macOS 標準ランナーを使う Metal 関連
# ジョブの将来追加〉まで一律に違反として遮断していた）。バージョン固定ラベル
# （`ubuntu-24.04`・`macos-14` 等）は本リポでの実使用実績がなく、GitHub 側の対応
# バージョン変遷を記憶だけで列挙すると一覧が陳腐化するため意図的に含めない。追加が
# 必要になった時点で本定数への明示的な拡張（レビュー必須）を行う設計とし、実機
# self-hosted ジョブの将来追加（ci.md「実機依存」節）と同じ「許容形の意図的な拡張」
# パターンに揃える。
ALLOWED_RUNNER_VALUES = frozenset({"ubuntu-latest", "macos-latest", "windows-latest"})
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
                if not (isinstance(value, str) and value in ALLOWED_RUNNER_VALUES):
                    allowed_repr = ", ".join(sorted(repr(v) for v in ALLOWED_RUNNER_VALUES))
                    violations.append(
                        f"{child_path}: runner 宣言の値が許容形"
                        f"（{allowed_repr} のいずれかとの完全一致）ではありません: {value!r}"
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
        documents = load_all(text)
    except YamlSubsetError as error:
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
    allowed_summary = "/".join(sorted(ALLOWED_RUNNER_VALUES))
    if violations:
        print(
            f"::error::{label} が runner 契約"
            f"（{allowed_summary} のいずれかとの完全一致・{BANNED_TOKEN} 不在。"
            ".claude/rules/ci.md「runner」節）に違反しています:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return False
    print(f"OK: {label} は runner 契約（{allowed_summary} のいずれかとの完全一致・{BANNED_TOKEN} 不在）に適合")
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
