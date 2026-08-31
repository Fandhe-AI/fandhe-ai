#!/usr/bin/env python3
"""framework-compare の数値一致契約（checksum 複合判定）の単一真実源。

`summarize.py`（横並び集計・目標達成ゲート）と `compare_ab.py`（都度同期廃止
#1011 の A/B 比較。イシュー #1083）の双方がここから tolerance 定数と
`checksums_match` を参照する（codex-review P1 指摘・PR #1088: 同値の定数を
2 ファイルへ分散定義すると片方だけ更新された際に同一計測結果が集計と A/B で
異なる判定になり、AGENTS.md「数値契約の統一」「閾値の分散定義による単一
真実源の破壊」に違反する）。

値は本体の数値一致契約（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`.claude/rules/coding-rust.md`・REQ-2）と同一。**この tolerance の緩和は
ユーザー承認必須**（coding-rust.md「テスト・ベンチ」節）。

利用側は sys.path に依存しないファイルパス指定 import（`importlib`）で
本モジュールを読み込む（`summarize_test.py`・`compare_ab_test.py` と同じ
方式。スクリプト直接実行・`python3 -m unittest <path>` のどちらでも同一
ファイルを解決できる）。
"""

CHECKSUM_ABS_TOL = 1e-5
CHECKSUM_REL_TOL = 1e-3


def checksums_match(a, b):
    """本体の数値一致契約と同一の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。

    分母は `max(abs(a), abs(b), 1e-12)` とし、引数順序（a, b のどちらを参照値と
    するか）に依らず対称な判定にする（`composite_close` 相当。イシュー #965
    codex-review P1 指摘: 旧実装の `diff / abs(b)` は非対称で、境界付近では
    a/b の順序次第で判定結果が変わりうる問題があった）。
    """
    diff = abs(a - b)
    if diff < CHECKSUM_ABS_TOL:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return diff / denom < CHECKSUM_REL_TOL
