#!/usr/bin/env python3
"""イシュー #2085（区分 B-1）マイクロベンチ A/B 集計（python3 標準ライブラリ
のみ）。`docs/perf/logs/elementwise-vjp-backend-ops-1583/aggregate.py`
（#1583）を本イシュー向けに改変したもの（期待セル集合・`out[...]`
フィールドの追加検査・`--self-test` を追加）。

before_run{1..5}.log / after_run{1..5}.log を読み、
bench[<case>][<numel>].median_s= 行から各 run の値を集め、
5 run の中央値同士の比 (after/before) を計算する。
out[<case>][<numel>].fold_bits=／grad[<case>][<numel>].fold_bits= 行は
全 run・before/after で完全一致することを検査する。

集計前に、事前登録した全セル（5 ケース × 3 形状 = 15 セル。
EXPECTED_KEYS）× 全 run（既定 5 run）× before/after の
bench・out・grad 記録がすべて過不足なく揃っていることを検証する
（#1583 aggregate.py の教訓をそのまま踏襲）。
"""
import io
import re
import sys
import statistics
import tempfile
import os


EXPECTED_CASES = (
    "chain4",
    "chain6",
    "fan_out",
    "fan_in",
    "matmul_boundary",
)
EXPECTED_SHAPES = (16384, 65536, 1048576)
EXPECTED_KEYS = sorted((case, shape) for case in EXPECTED_CASES for shape in EXPECTED_SHAPES)


class DuplicateKeyError(Exception):
    """同一ログ内で同じ (case, numel) キーの bench/out/grad 行が複数回出現した場合に送出する。"""


def parse(path):
    bench = {}
    out = {}
    grad = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            m = re.match(r"bench\[([^\]]+)\]\[(\d+)\]\.median_s=([0-9.eE+-]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in bench:
                    raise DuplicateKeyError(
                        f"{path}:{lineno}: bench{list(key)} が同一ログ内で複数回出現しました"
                        f"（先行値={bench[key]!r}・後続値={m.group(3)!r}）"
                    )
                bench[key] = float(m.group(3))
                continue
            m = re.match(r"out\[([^\]]+)\]\[(\d+)\]\.fold_bits=(0x[0-9a-f]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in out:
                    raise DuplicateKeyError(
                        f"{path}:{lineno}: out{list(key)} が同一ログ内で複数回出現しました"
                    )
                out[key] = m.group(3)
                continue
            m = re.match(r"grad\[([^\]]+)\]\[(\d+)\]\.fold_bits=(0x[0-9a-f]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in grad:
                    raise DuplicateKeyError(
                        f"{path}:{lineno}: grad{list(key)} が同一ログ内で複数回出現しました"
                    )
                grad[key] = m.group(3)
    return bench, out, grad


def main(logdir, n_runs=5, out_stream=None):
    out_stream = out_stream or sys.stdout
    before_samples = {}
    after_samples = {}
    before_dump = {"out": {}, "grad": {}}
    after_dump = {"out": {}, "grad": {}}

    expected_keys = EXPECTED_KEYS
    missing = []

    for run in range(1, n_runs + 1):
        before_path = f"{logdir}/before_run{run}.log"
        after_path = f"{logdir}/after_run{run}.log"
        try:
            b, bo, bg = parse(before_path)
            a, ao, ag = parse(after_path)
        except DuplicateKeyError as e:
            print(f"ERROR: {e}", file=out_stream)
            return False

        for label, path, keys_present in (
            ("before bench", before_path, set(b.keys())),
            ("after bench", after_path, set(a.keys())),
            ("before out", before_path, set(bo.keys())),
            ("after out", after_path, set(ao.keys())),
            ("before grad", before_path, set(bg.keys())),
            ("after grad", after_path, set(ag.keys())),
        ):
            lacking = set(expected_keys) - keys_present
            extra = keys_present - set(expected_keys)
            if lacking or extra:
                missing.append((run, label, path, sorted(lacking), sorted(extra)))

        for k in expected_keys:
            if k in b:
                before_samples.setdefault(k, []).append(b[k])
            if k in a:
                after_samples.setdefault(k, []).append(a[k])
            if k in bo:
                before_dump["out"].setdefault(k, []).append(bo[k])
            if k in ao:
                after_dump["out"].setdefault(k, []).append(ao[k])
            if k in bg:
                before_dump["grad"].setdefault(k, []).append(bg[k])
            if k in ag:
                after_dump["grad"].setdefault(k, []).append(ag[k])

    if missing:
        print(
            f"ERROR: 集計前検証に失敗しました"
            f"（事前登録セル数={len(expected_keys)}・期待 run 数={n_runs}）",
            file=out_stream,
        )
        for run, label, path, lacking, extra in missing:
            if lacking:
                print(f"  run{run} {label} ({path}): 欠落セル {lacking}", file=out_stream)
            if extra:
                print(f"  run{run} {label} ({path}): 未登録セル {extra}", file=out_stream)
        return False

    print(
        f"{'case':20s} {'numel':>9s} {'before_med_s':>14s} {'after_med_s':>14s} "
        f"{'ratio':>8s} {'out_ok':>8s} {'grad_ok':>8s}",
        file=out_stream,
    )
    all_ok = True
    for k in expected_keys:
        bench_samples_b = before_samples.get(k, [])
        bench_samples_a = after_samples.get(k, [])
        if len(bench_samples_b) != n_runs or len(bench_samples_a) != n_runs:
            print(
                f"ERROR: {k} の bench サンプル数が n_runs={n_runs} と一致しません "
                f"(before={len(bench_samples_b)}, after={len(bench_samples_a)})",
                file=out_stream,
            )
            return False

        bmed = statistics.median(bench_samples_b)
        amed = statistics.median(bench_samples_a)
        ratio = amed / bmed if bmed > 0 else float("nan")

        def _ok(dump, side):
            vals = dump[side].get(k, [])
            return len(vals) == n_runs and len(set(vals)) == 1

        out_ok = _ok(before_dump, "out") and _ok(after_dump, "out") and (
            before_dump["out"].get(k, [None])[0] == after_dump["out"].get(k, [None])[0]
        )
        grad_ok = _ok(before_dump, "grad") and _ok(after_dump, "grad") and (
            before_dump["grad"].get(k, [None])[0] == after_dump["grad"].get(k, [None])[0]
        )
        if not (out_ok and grad_ok):
            all_ok = False
        over = " <=1.00" if ratio <= 1.00 else " >1.00"
        print(
            f"{k[0]:20s} {k[1]:9d} {bmed:14.9f} {amed:14.9f} {ratio:8.4f} "
            f"{'OK' if out_ok else 'MISMATCH':>8s} {'OK' if grad_ok else 'MISMATCH':>8s}{over}",
            file=out_stream,
        )

    print(file=out_stream)
    print("out/grad bit-fold all match:", all_ok, file=out_stream)
    return all_ok


def _write_fixture_logs(dirpath, ratio_over_one=False):
    """`--self-test` 用の固定フィクスチャログを生成する（全セル揃い・
    out/grad 完全一致・ratio<=1.00 のケース。`ratio_over_one=True` で
    1 セルだけ ratio>1.00 に壊した検証にも使える）。"""
    for side, base_s, mult in (("before", 0.001, 1.0), ("after", 0.0009, 1.0)):
        for run in range(1, 6):
            lines = []
            for case in EXPECTED_CASES:
                for numel in EXPECTED_SHAPES:
                    s = base_s * mult
                    if ratio_over_one and side == "after" and case == EXPECTED_CASES[0] and numel == EXPECTED_SHAPES[0]:
                        s = base_s * 10  # 意図的に after > before にする
                    lines.append(f"bench[{case}][{numel}].median_s={s:.9f}")
                    lines.append(f"out[{case}][{numel}].fold_bits=0xdeadbeefcafebabe")
                    lines.append(f"grad[{case}][{numel}].fold_bits=0xfeedfacefeedface")
            with open(os.path.join(dirpath, f"{side}_run{run}.log"), "w") as f:
                f.write("\n".join(lines) + "\n")


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        _write_fixture_logs(d)
        buf = io.StringIO()
        result = main(d, 5, out_stream=buf)
        if not result:
            print("self-test NG: 正常フィクスチャで判定が False になった", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
            ok = False
        else:
            print("self-test OK: 正常フィクスチャは pass する")

    with tempfile.TemporaryDirectory() as d:
        _write_fixture_logs(d, ratio_over_one=True)
        buf = io.StringIO()
        result = main(d, 5, out_stream=buf)
        # ratio>1.00 は「集計自体は成立する（missing なし・grad_ok）」が、
        # 呼び出し元（人間）が判定規則に従い REJECT と判断する材料になる
        # 行を出力する契約であり、集計スクリプト自体は False を返さない
        # （集計失敗〈missing／件数不一致〉とは意味が異なるため）。
        if not result:
            print("self-test NG: ratio>1.00 フィクスチャで集計自体が失敗した", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
            ok = False
        elif " >1.00" not in buf.getvalue():
            print("self-test NG: ratio>1.00 の行が出力に含まれない", file=sys.stderr)
            ok = False
        else:
            print("self-test OK: ratio>1.00 セルが出力に現れる")

    with tempfile.TemporaryDirectory() as d:
        _write_fixture_logs(d)
        # 1 ケース分の行を欠落させて missing 検出を確認する。
        path = os.path.join(d, "before_run1.log")
        with open(path) as f:
            content = f.read()
        with open(path, "w") as f:
            f.write("\n".join(l for l in content.splitlines() if EXPECTED_CASES[0] not in l) + "\n")
        buf = io.StringIO()
        result = main(d, 5, out_stream=buf)
        if result:
            print("self-test NG: 欠落セルがあるのに pass 判定になった", file=sys.stderr)
            ok = False
        else:
            print("self-test OK: 欠落セルフィクスチャは fail する")

    if not ok:
        sys.exit(1)
    print("OK: self-test すべて pass")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        logdir = sys.argv[1] if len(sys.argv) > 1 else "."
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        ok = main(logdir, n_runs)
        sys.exit(0 if ok else 1)
