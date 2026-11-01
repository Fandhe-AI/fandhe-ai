#!/usr/bin/env python3
"""追跡表 docs/compat-matrix-ari-tracker.md の形式検査と行別残件の集計（手動実行）。

役割: 対応表「部分的」29 項目の追跡表（§1）を fail-closed で解析し、
  (1) 形式（列数・連番・語彙・「対象外」の承認 URL）を検査し、
  (2) 0.12.0 の行別「未公開」「保留」を数えて §8.5 確定件数表（FROZEN_0_12_0）および
      追跡表 §2 の手書き件数と照合し、
  (3) 0.14.0 の区分を行別に集計して Markdown 表で出力する。
呼び出し文脈: 追跡表を更新する PR の手元確認と、#3869（0.14.0 事前監査）の件数突き合わせで使う。
正本は追跡表と 0.12.0 監査 §8.5 のままで、本スクリプトは読むだけ（書き換えない）。行判定（「ある」）は出さない。
CI には載せない（#3781 の受け入れ条件。ジョブ追加は ruleset の required contexts 更新＝所有者承認が要る）。
依存は python3 標準ライブラリのみ（ubuntu-latest の 3.12 で動く書き方に限る）。
終了コード: 0 = 適合、1 = 検査違反、2 = 使い方・入力パスの異常。
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC = REPO_ROOT / "docs" / "compat-matrix-ari-tracker.md"
TESTDATA = REPO_ROOT / "scripts" / "testdata"

# 追跡表 §1 / §2 のヘッダ（完全一致。列を変える PR は本定数と fixture も更新する）
TRACKER_HEADER = (
    "| # | 項目 | 対応表の行 | 0.12.0 の区分 | 設計記録 | 承認 | 実装 issue | 正ガード | 実機 parity | 0.14.0 の区分 |"
)
REMAINING_HEADER = "| 行 | 0.12.0 未公開 | 0.12.0 保留 | 追跡表の該当 # |"
SECTION1 = "## 1. 追跡表"
SECTION2 = "## 2. 行別の残件数"

# 対応表の 9 行（並び順固定）
ROW_NAMES = [
    "dtype", "自動微分", "演算の範囲", "NN 層・モデル構築", "最適化・学習ループ",
    "バックエンド・ハード", "相互運用", "事前学習済みモデル", "推論・サービング",
]
# 0.12.0 監査 §8.5 の行別 (未公開, 保留)。過去値で固定（追跡表 §3）
FROZEN_0_12_0 = {
    "dtype": (1, 1), "自動微分": (0, 4), "演算の範囲": (1, 0), "NN 層・モデル構築": (9, 0),
    "最適化・学習ループ": (1, 7), "バックエンド・ハード": (0, 1), "相互運用": (2, 1),
    "事前学習済みモデル": (1, 0), "推論・サービング": (1, 0),
}
EXPECTED_ITEMS = 29
VOCAB_0120 = ("未公開", "保留")
VOCAB_0140 = ("未確定", "公開", "対象外", "未公開", "保留")
PARITY_PREFIXES = ("未実測", "pass", "fail", "該当 issue なし")
APPROVAL_URL = re.compile(r"https://github\.com/Fandhe-AI/fandhe-ai/issues/[0-9]+#issuecomment-[0-9]+")
MULTI_ROW_SEP = "＋"  # 項目 2 のみ「dtype＋自動微分」


class TrackerError(Exception):
    """検査違反。曖昧な入力は読み飛ばさずすべてこれで止める。"""


@dataclass
class Row:
    line: int
    number: int
    item: str
    rows: list[str]
    cls_0120: str
    approval: str
    parity: str
    cls_0140: str


@dataclass
class Tally:
    n_items: int
    count_0120: dict[str, tuple[int, int]]
    count_0140: dict[str, dict[str, int]]
    items_0140: dict[str, int]


def _section_table(text: str, heading: str, header: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == heading]
    if len(starts) != 1:
        raise TrackerError(f"見出し '{heading}' が {len(starts)} 回ある（1 回だけ必要）")
    blocks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for i in range(starts[0] + 1, len(lines)):
        ln = lines[i]
        if ln.startswith("## "):
            break
        if ln.startswith("|"):
            cur.append((i + 1, ln.rstrip()))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if len(blocks) != 1:
        raise TrackerError(f"'{heading}' 節の表が {len(blocks)} 個ある（1 個だけ必要）")
    blk = blocks[0]
    if blk[0][1] != header:
        raise TrackerError(f"{blk[0][0]} 行: ヘッダが定義と一致しない")
    if len(blk) < 3 or not re.fullmatch(r"\|(-+\|)+", blk[1][1]):
        raise TrackerError(f"{blk[0][0] + 1} 行: 区切り行がない")
    return blk[2:]


def _cells(line_no: int, ln: str, n: int) -> list[str]:
    if not ln.endswith("|"):
        raise TrackerError(f"{line_no} 行: 行末が '|' でない")
    cells = [c.strip() for c in ln[1:-1].split("|")]
    if len(cells) != n:
        raise TrackerError(f"{line_no} 行: セル数 {len(cells)}（{n} 必要）")
    for k, c in enumerate(cells, 1):
        if not c:
            raise TrackerError(f"{line_no} 行: {k} 列目が空")
    return cells


def parse_tracker(text: str, expected_items: int) -> list[Row]:
    out: list[Row] = []
    for line_no, ln in _section_table(text, SECTION1, TRACKER_HEADER):
        c = _cells(line_no, ln, 10)
        if not re.fullmatch(r"[0-9]+", c[0]):
            raise TrackerError(f"{line_no} 行: # が整数でない: {c[0]}")
        rows = c[2].split(MULTI_ROW_SEP)
        for r in rows:
            if r not in ROW_NAMES:
                raise TrackerError(f"{line_no} 行: 対応表の行が語彙外: {r}")
        if len(set(rows)) != len(rows):
            raise TrackerError(f"{line_no} 行: 対応表の行が重複")
        if c[3] not in VOCAB_0120:
            raise TrackerError(f"{line_no} 行: 0.12.0 の区分が語彙外: {c[3]}")
        if not c[8].startswith(PARITY_PREFIXES):
            raise TrackerError(f"{line_no} 行: 実機 parity が語彙外: {c[8]}")
        if c[9] not in VOCAB_0140:
            raise TrackerError(f"{line_no} 行: 0.14.0 の区分が語彙外: {c[9]}")
        # 追跡表 §4 規則 3: 「対象外」へは所有者の承認コメント URL が必要。
        # 承認欄の語彙は自由記述（項目 6 のように「公開形は…」で始まる行がある）ため、
        # 前方一致の語彙検査はせず、この不変条件だけを機械で守る。
        if c[9] == "対象外" and not APPROVAL_URL.search(c[5]):
            raise TrackerError(f"{line_no} 行: 0.14.0 が「対象外」だが承認欄に承認コメント URL がない")
        out.append(Row(line_no, int(c[0]), c[1], rows, c[3], c[5], c[8], c[9]))
    nums = [r.number for r in out]
    if nums != list(range(1, expected_items + 1)):
        raise TrackerError(f"# が 1..{expected_items} の連番でない: {nums}")
    return out


def parse_remaining(text: str) -> dict[str, tuple[int, int]]:
    body = _section_table(text, SECTION2, REMAINING_HEADER)
    if len(body) != len(ROW_NAMES) + 1:
        raise TrackerError(f"§2 の行数 {len(body)}（{len(ROW_NAMES) + 1} 必要）")
    got: dict[str, tuple[int, int]] = {}
    for (line_no, ln), name in zip(body, ROW_NAMES + ["合計"]):
        c = _cells(line_no, ln, 4)
        if c[0] != name:
            raise TrackerError(f"{line_no} 行: 行名 '{c[0]}'（'{name}' が必要）")
        for v in (c[1], c[2]):
            if not re.fullmatch(r"[0-9]+", v):
                raise TrackerError(f"{line_no} 行: 件数が整数でない: {v}")
        got[name] = (int(c[1]), int(c[2]))
    return got


def _sum(frozen: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return (sum(u for u, _ in frozen.values()), sum(h for _, h in frozen.values()))


def _diff(label: str, want: dict[str, tuple[int, int]], got: dict[str, tuple[int, int]]) -> None:
    bad = [f"{k}: 期待 {want[k]} / 実際 {got.get(k)}（未公開, 保留）" for k in want if got.get(k) != want[k]]
    if bad:
        raise TrackerError(f"{label} が §8.5 固定値と不一致: " + "; ".join(bad))


def check(text: str, expected_items: int, frozen: dict[str, tuple[int, int]]) -> Tally:
    rows = parse_tracker(text, expected_items)
    c0120 = {n: (0, 0) for n in ROW_NAMES}
    c0140 = {n: {v: 0 for v in VOCAB_0140} for n in ROW_NAMES}
    items_0140 = {v: 0 for v in VOCAB_0140}
    for r in rows:
        items_0140[r.cls_0140] += 1
        for name in r.rows:
            u, h = c0120[name]
            c0120[name] = (u + (r.cls_0120 == "未公開"), h + (r.cls_0120 == "保留"))
            c0140[name][r.cls_0140] += 1
    want = {**frozen, "合計": _sum(frozen)}
    _diff("追跡表 §1 の集計", want, {**c0120, "合計": _sum(c0120)})
    _diff("追跡表 §2 の手書き件数", want, parse_remaining(text))
    return Tally(len(rows), c0120, c0140, items_0140)


def render(t: Tally) -> str:
    u, h = _sum(t.count_0120)
    out = [
        "0.12.0 の照合: OK（§1 の集計 == §2 == §8.5 固定値）",
        f"- 行別合計 {u + h} 件（未公開 {u}・保留 {h}）、重複を除いた項目数 {t.n_items}",
        "",
        "0.14.0 の区分（行別。複数行の項目は各行に数える）",
        "",
        "| 行 | 未確定 | 未公開 | 保留 | 残件 | 公開 | 対象外 |",
        "|---|---|---|---|---|---|---|",
    ]
    tot = {v: 0 for v in VOCAB_0140}
    for n in ROW_NAMES:
        d = t.count_0140[n]
        for v in VOCAB_0140:
            tot[v] += d[v]
        rem = d["未確定"] + d["未公開"] + d["保留"]
        out.append(f"| {n} | {d['未確定']} | {d['未公開']} | {d['保留']} | {rem} | {d['公開']} | {d['対象外']} |")
    rem = tot["未確定"] + tot["未公開"] + tot["保留"]
    out.append(f"| 合計 | {tot['未確定']} | {tot['未公開']} | {tot['保留']} | {rem} | {tot['公開']} | {tot['対象外']} |")
    i = t.items_0140
    irem = i["未確定"] + i["未公開"] + i["保留"]
    out += [
        "",
        f"重複を除いた項目数: 残件 {irem}（未確定 {i['未確定']}・未公開 {i['未公開']}・保留 {i['保留']}）"
        f"、公開 {i['公開']}、対象外 {i['対象外']}",
        "",
        "注: 出すのは「追跡している残件が 0 か」まで。行判定（「ある」）は公開・対象外の基準件数（§8.5）と合わせて監査側が判断する。",
    ]
    return "\n".join(out)


# self-test: clean fixture（5 項目）の 0.12.0 期待値
_CLEAN_FROZEN = {n: (0, 0) for n in ROW_NAMES}
_CLEAN_FROZEN.update(
    {"dtype": (1, 1), "自動微分": (0, 1), "演算の範囲": (1, 0), "NN 層・モデル構築": (1, 0), "相互運用": (0, 1)}
)
_NEGATIVES = [
    "missing-cell", "extra-cell", "bad-vocab-0120", "bad-vocab-0140", "unknown-row", "duplicate-number",
    "gap-number", "header-mismatch", "out-of-scope-without-url", "count-drift", "remaining-drift",
    "empty-cell", "missing-section",
]


def self_test() -> int:
    ok = True

    def report(name: str, good: bool, why: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"{'OK' if good else 'NG'}: {name}{' - ' + why if why else ''}")

    def load(name: str) -> str | None:
        p = TESTDATA / name
        if not p.is_file():
            report(name, False, "fixture が見つからない")
            return None
        return p.read_text(encoding="utf-8")

    t = load("compat-tracker-clean.md")
    if t is not None:
        try:
            tally = check(t, 5, _CLEAN_FROZEN)
            rendered = render(tally)
            good = (
                tally.items_0140 == {"未確定": 3, "公開": 1, "対象外": 1, "未公開": 0, "保留": 0}
                and tally.count_0140["dtype"]["未確定"] == 2
                and tally.count_0140["自動微分"]["未確定"] == 1
                and tally.count_0140["演算の範囲"]["公開"] == 1
                and tally.count_0140["相互運用"]["対象外"] == 1
                and "行別合計 6 件" in rendered
                and "項目数 5" in rendered
            )
            report("compat-tracker-clean.md", good, "" if good else "集計値が期待と不一致")
        except TrackerError as e:
            report("compat-tracker-clean.md", False, str(e))
    for neg in _NEGATIVES:
        name = f"compat-tracker-{neg}.md"
        t = load(name)
        if t is None:
            continue
        try:
            check(t, 5, _CLEAN_FROZEN)
            report(name, False, "違反を検出できなかった")
        except TrackerError:
            report(name, True)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="追跡表の形式検査と行別残件の集計")
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="対象の追跡表（既定: docs/compat-matrix-ari-tracker.md）")
    ap.add_argument("--self-test", action="store_true", help="scripts/testdata の fixture で自己テストする")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.doc.is_file():
        print(f"error: 通常ファイルではない: {args.doc}", file=sys.stderr)
        return 2
    try:
        text = args.doc.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: 読み込めない: {e}", file=sys.stderr)
        return 2
    try:
        print(render(check(text, EXPECTED_ITEMS, FROZEN_0_12_0)))
    except TrackerError as e:
        print(f"違反: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
