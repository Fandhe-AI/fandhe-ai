#!/usr/bin/env python3
"""#4150 の相手側入力（--opp-tsv）を #4149 の集計結果から作る変換（stdlib のみ）。

役割: #4149 の aggregate_4149.py が出した aggregate.tsv（単位 ms）の CUDA 9 項目を、
  aggregate_4150.py の --opp-tsv が受け付ける形式（ヘッダ 'item\\tmedian_us'、単位 µs）へ写す。
  値は中央値（median_ms）だけを 1000 倍して µs にする。推定値・0 で埋めない。
  #4149 側で未測定（tag=未測定・median_ms 空欄）の項目は出力しない（aggregate_4150.py の対応表で unavailable になる）。
呼び出し文脈: #4149 の実測 PR（aggregate.tsv・aggregate.md）が main に入った後、#4150 の集計前に 1 回実行する。
  aggregate_4150.py には手を入れない（RULE に sha256 を固定するため）。本スクリプトの入出力形式は
  aggregate_4149.py::render の tsv 列と aggregate_4150.py::load_opp_text の受理形式に合わせてある。
厳格さ（fail-closed）: ヘッダ・列数の違い、9 項目の section の食い違い、重複、未知の tag、
  数値でない・非有限・正でない中央値、実測なのに空欄、未測定なのに値あり、は例外で止める。
  #4149 の aggregate.md の表題が「（正式）」でない（ゲート不通過 round を含む参考集計）場合も止める。
  出力先が既にある場合は上書きしない。

使い方（python3 -I で実行する）:
  opp_from_4149.py --self-test
  opp_from_4149.py --src <#4149 の記録先ディレクトリ> --out <opp.tsv>
    （<記録先>/aggregate.tsv と <記録先>/aggregate.md を読む。入力 tsv の sha256 を標準エラーへ出す）
"""

import hashlib
import math
import os
import sys
import tempfile

# aggregate_4150.py::OPP_ITEMS と同じ 9 項目と、#4149 の aggregate.tsv 上の section（aggregate_4149.py::derive）。
ITEM_SECTION = (
    ("issue", "cuda_infer"),
    ("readout_sync", "cuda_infer"),
    ("gpu_span_event", "cuda_infer"),
    ("mm_event_l1_fwd", "cuda_shapes"),
    ("mm_event_l2_fwd", "cuda_shapes"),
    ("sync_idle", "cuda_fixed"),
    ("tiny_roundtrip", "cuda_fixed"),
    ("launch_floor_add", "cuda_fixed"),
    ("event_record_pair", "cuda_fixed"),
)
SRC_HEADER = "section\tquantity\ttag\tmedian_ms\tmin_ms\tmax_ms\tshare_median\tshare_min\tshare_max"
FORMAL_TITLE = "# 区間内訳（正式） #4149"
TAGS = ("実測", "推定", "未測定")


class ConvError(Exception):
    pass


def convert(tsv_text, md_text):
    """#4149 の aggregate.tsv・aggregate.md の本文から opp.tsv の本文を返す。"""
    md_lines = md_text.splitlines()
    if not md_lines or md_lines[0] != FORMAL_TITLE:
        raise ConvError("#4149 aggregate.md の表題が「（正式）」でない（参考集計は相手側の値として使わない）")
    lines = tsv_text.splitlines()
    if not lines or lines[0] != SRC_HEADER:
        raise ConvError("#4149 aggregate.tsv のヘッダが想定と違う")
    want = dict(ITEM_SECTION)
    seen = {}
    for ln in lines[1:]:
        if not ln:
            raise ConvError("#4149 aggregate.tsv に空行がある")
        parts = ln.split("\t")
        if len(parts) != 9:
            raise ConvError(f"列数が 9 でない: {ln!r}")
        sec, q, tag, med = parts[0], parts[1], parts[2], parts[3]
        if tag not in TAGS:
            raise ConvError(f"未知の tag: {ln!r}")
        if q not in want:
            continue
        if sec != want[q]:
            # 同名の量が別 section にある場合は対象外。対象 section の行だけを数える。
            continue
        if q in seen:
            raise ConvError(f"重複: {sec}/{q}")
        if tag == "未測定":
            if med != "":
                raise ConvError(f"未測定なのに値がある: {ln!r}")
            seen[q] = None
            continue
        if tag != "実測":
            raise ConvError(f"相手側の項目が実測でない: {ln!r}")
        try:
            v = float(med)
        except ValueError:
            raise ConvError(f"中央値が数値でない: {ln!r}")
        if not math.isfinite(v) or v <= 0:
            raise ConvError(f"中央値が有限の正値でない: {ln!r}")
        seen[q] = v * 1000.0
    missing = [q for q, _ in ITEM_SECTION if q not in seen]
    if missing:
        raise ConvError(f"項目が欠けている: {missing}")
    out = ["item\tmedian_us"]
    for q, _ in ITEM_SECTION:
        if seen[q] is not None:
            out.append(f"{q}\t{seen[q]:.3f}")
    return "\n".join(out) + "\n"


def run(src, out):
    if os.path.exists(out):
        raise ConvError(f"出力が既にある（上書きしない）: {out}")
    with open(os.path.join(src, "aggregate.tsv"), "rb") as f:
        raw = f.read()
    with open(os.path.join(src, "aggregate.md"), encoding="utf-8") as f:
        md = f.read()
    text = convert(raw.decode("utf-8"), md)
    with open(out, "x", encoding="utf-8") as f:
        f.write(text)
    print(f"source aggregate.tsv sha256={hashlib.sha256(raw).hexdigest()}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)


def _src(rows, extra=()):
    """self-test 用の #4149 aggregate.tsv 本文。rows は (section, quantity, tag, median_ms)。"""
    body = [SRC_HEADER]
    for s, q, t, m in list(rows) + list(extra):
        body.append("\t".join([s, q, t, m, m, m, "", "", ""]))
    return "\n".join(body) + "\n"


def self_test():
    base = [("cuda_infer", "window", "実測", "0.04000"), ("cuda_infer", "issue", "実測", "0.02000"),
            ("cuda_infer", "readout_sync", "実測", "0.01500"), ("cuda_infer", "checksum", "実測", "0.00100"),
            ("cuda_infer", "gpu_span_event", "実測", "0.01000"), ("cuda_infer", "residual", "推定", "0.00400"),
            ("cuda_fixed", "sync_idle", "実測", "0.00300"), ("cuda_fixed", "tiny_roundtrip", "実測", "0.01200"),
            ("cuda_fixed", "launch_floor_add", "実測", "0.00250"), ("cuda_fixed", "event_record_pair", "実測", "0.00050"),
            ("cuda_shapes", "mm_event_l1_fwd", "実測", "0.00800"), ("cuda_shapes", "mm_event_l2_fwd", "実測", "0.00400"),
            ("cuda_shapes", "mm_event_l1_dw", "実測", "0.00900")]
    md = FORMAL_TITLE + "\n\n- x\n"
    got = convert(_src(base), md)
    lines = got.splitlines()
    assert lines[0] == "item\tmedian_us", got
    assert "issue\t20.000" in lines and "mm_event_l2_fwd\t4.000" in lines and "event_record_pair\t0.500" in lines, got
    assert len(lines) == 10, got
    # 未測定は出力しない（0・推定で埋めない）
    unav = [r if r[1] != "gpu_span_event" else (r[0], r[1], "未測定", "") for r in base]
    got2 = convert(_src(unav), md)
    assert "gpu_span_event" not in got2 and len(got2.splitlines()) == 9, got2
    # 同名の量が別 section にあっても対象 section だけを使う
    convert(_src(base, [("cpu_train", "issue", "実測", "9.00000")]), md)
    bad = [
        ("参考集計", _src(base), "# 区間内訳（参考: ゲート不通過 round あり。確度を書かない） #4149\n"),
        ("ヘッダ", _src(base).replace("median_ms", "median_s", 1), md),
        ("重複", _src(base, [("cuda_fixed", "sync_idle", "実測", "0.00300")]), md),
        ("欠損", _src([r for r in base if r[1] != "tiny_roundtrip"]), md),
        ("未知 tag", _src(base, [("cuda_fixed", "x", "?", "1")]), md),
        ("推定 tag", _src([r if r[1] != "issue" else (r[0], r[1], "推定", r[3]) for r in base]), md),
        ("非数値", _src([r if r[1] != "issue" else (r[0], r[1], r[2], "abc") for r in base]), md),
        ("非有限", _src([r if r[1] != "issue" else (r[0], r[1], r[2], "nan") for r in base]), md),
        ("非正", _src([r if r[1] != "issue" else (r[0], r[1], r[2], "-0.00100") for r in base]), md),
        ("実測で空欄", _src([r if r[1] != "issue" else (r[0], r[1], r[2], "") for r in base]), md),
        ("未測定で値", _src([r if r[1] != "issue" else (r[0], r[1], "未測定", "0.00100") for r in base]), md),
        ("列数", _src(base) + "cuda_fixed\tx\t実測\n", md),
        ("空行", _src(base) + "\n", md),
    ]
    for name, t, m in bad:
        try:
            convert(t, m)
        except ConvError:
            continue
        raise AssertionError(f"拒否されなかった: {name}")
    # 出力の上書き拒否と、aggregate_4150.py の受理形式との一致
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "aggregate.tsv"), "w", encoding="utf-8").write(_src(base))
        open(os.path.join(d, "aggregate.md"), "w", encoding="utf-8").write(md)
        out = os.path.join(d, "opp.tsv")
        run(d, out)
        try:
            run(d, out)
        except ConvError:
            pass
        else:
            raise AssertionError("上書きが拒否されなかった")
        here = os.path.dirname(os.path.abspath(__file__))
        sys.dont_write_bytecode = True  # 記録ディレクトリに __pycache__ を残さない（ツリー指紋を汚さない）
        sys.path.insert(0, here)
        try:
            import aggregate_4150
        finally:
            sys.path.pop(0)
        opp = aggregate_4150.load_opp(out)
        assert set(opp) == {q for q, _ in ITEM_SECTION} and set(aggregate_4150.OPP_ITEMS) == set(opp), opp
    print("self-test OK")


def main(argv):
    def opt(name):
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None
    try:
        if "--self-test" in argv:
            self_test()
            return 0
        src, out = opt("--src"), opt("--out")
        if not src or not out:
            print(__doc__, file=sys.stderr)
            return 2
        run(src, out)
        return 0
    except (ConvError, OSError, UnicodeDecodeError) as e:
        print(f"CONVERT-ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
