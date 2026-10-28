#!/usr/bin/env python3
"""イシュー #3611 r2: B1／B2 バイナリの主要シンボルを mangling 方式に依らず突き合わせ、配置のずれを数える（stdlib のみ・fail-closed）。

役割: `run_diag_resident_chain_cpu.sh` が precheck で呼ぶ。`nm` の生出力（各行 `addr type name`）を B1／B2 について受け取り、
`train_step_resident`・`forward_resident` のシンボル名を正規化したキーで突き合わせて、アドレスが異なるキーの数
（layout_symbol_diff）と共通キー数（layout_symbol_keys）を stdout へ出す。DIAG-r2.txt の [対象] が規則の正本。
経緯: r1（DIAG.txt）の runner は legacy mangling の末尾ハッシュ `17h<16hex>E` だけを落としていた。実機の両機体は v0 mangling で、
名前の途中のクレート識別子 `Cs<base62>_` がビルドパスで変わるため共通キーが 0 件になり、計測前に停止した（§17.5）。
正規化: legacy は末尾ハッシュを除く。v0 は先頭の `_+R` を `_R` に揃え（macOS は `_` が 1 つ多い）、`Cs<base62>_` を `C` へ潰す
（直後が識別子の長さ接頭辞の数字か punycode の `u` の位置に限る）。正規表現はすべてアンカーまたは位置条件付きで、部分一致で素通りさせない。
fail-closed（終了コード 1）: 同一バイナリ内のキー重複・必須シンボルの欠落（B1／B2 のどちらでも）・共通キー 0 件・入力不正。
"""
import argparse
import re
import sys

REQUIRED_SUFFIXES = ("19train_step_resident", "16forward_resident")
TARGET = re.compile(r"train_step_resident|forward_resident")
LEGACY_HASH = re.compile(r"17h[0-9a-f]{16}E$")
V0_PREFIX = re.compile(r"^_+R")
V0_CRATE = re.compile(r"Cs[0-9A-Za-z]*_(?=[0-9u])")


class Bad(Exception):
    """入力不正・突き合わせ不能（終了コード 1）。"""


def normalize(name):
    """シンボル名をビルドパス非依存のキーへ正規化する。"""
    if V0_PREFIX.match(name):
        return V0_CRATE.sub("C", V0_PREFIX.sub("_R", name))
    return LEGACY_HASH.sub("", name)


def parse_nm(text, what):
    """`addr type name` 行から対象シンボルだけを {key: (addr, rawname)} にする。キー重複は Bad。"""
    out = {}
    for ln in text.splitlines():
        f = ln.split()
        if len(f) < 3 or not TARGET.search(f[2]):
            continue
        if not re.fullmatch(r"[0-9a-fA-F]+", f[0]):
            raise Bad(f"{what}: アドレスが 16 進でない: {ln!r}")
        key = normalize(f[2])
        if key in out:
            raise Bad(f"{what}: 正規化キーが衝突した: {key}")
        out[key] = (f[0].lower(), f[2])
    return out


def compare(t1, t2):
    """(layout_symbol_diff, layout_symbol_keys, 共通キー一覧) を返す。必須シンボルが両側に無ければ Bad。"""
    common = sorted(set(t1) & set(t2))
    if not common:
        raise Bad("共通の正規化キーが 0 件")
    for suf in REQUIRED_SUFFIXES:
        for what, t in (("B1", t1), ("B2", t2)):
            if not any(k.endswith(suf) for k in t):
                raise Bad(f"{what} に {suf} で終わるシンボルが無い")
        if not any(k.endswith(suf) for k in common):
            raise Bad(f"B1 と B2 で {suf} で終わるキーが共通でない")
    diff = sum(1 for k in common if int(t1[k][0], 16) != int(t2[k][0], 16))
    return diff, len(common), common


def dump(path, table):
    with open(path, "w", encoding="utf-8") as fh:
        for k in sorted(table):
            fh.write(f"{k} {table[k][0]} {table[k][1]}\n")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--out-b1", required=True)
    ap.add_argument("--out-b2", required=True)
    a = ap.parse_args(argv)
    try:
        tabs = []
        for p, what in ((a.b1, "B1"), (a.b2, "B2")):
            with open(p, encoding="utf-8", errors="replace") as fh:
                tabs.append(parse_nm(fh.read(), what))
        diff, keys, _ = compare(*tabs)
    except (Bad, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    dump(a.out_b1, tabs[0])
    dump(a.out_b2, tabs[1])
    print(f"layout_symbol_diff={diff}")
    print(f"layout_symbol_keys={keys}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
