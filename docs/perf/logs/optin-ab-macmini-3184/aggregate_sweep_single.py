#!/usr/bin/env python3
"""縮約しきい値スイープの Mac mini 単機集計（イシュー #3184。補遺 RULE.txt 5）。

役割: 継承の `elemental-reduction-ab-2102/aggregate_sweep.py` は `--m4max` と `--gb10` の両方を
必須にする（2 機体判定のため）。Mac mini 単機の T_macmini を出すために、
同スクリプトの `load_machine`／`evaluate`／`render` を import して 1 機体だけで評価する薄い補助。
判定ロジックは継承側が正で、本ファイルは再実装しない（継承ファイルは編集しない）。
最終 T は将来 min(T_macmini, T_gb10)。stdlib のみ。`python3 -I` で実行する。

終了コード: 0=Stage 1 へ進める / 2=入力不正 / 4=checksum 不一致（FAIL）
            5=REJECT（T_macmini <= 640）/ 6=undetermined（利得が立たない）
"""
import argparse
import os
import sys
import tempfile

# `-I`（isolated）では script dir が sys.path に入らないため、継承側ディレクトリを明示追加する。
_INHERITED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "elemental-reduction-ab-2102"
)
sys.path.insert(0, os.path.normpath(_INHERITED))
import aggregate_sweep as base  # noqa: E402


def run_single(directory):
    """directory の run1..5.log から Mac mini 単機の判定結果 dict を返す。入力不正は InputError。"""
    return base.evaluate({"macmini": base.load_machine(directory)})


def self_test():
    def gain(op, n, arm):
        if arm == "par":
            return 1000
        return 500 if n < 8192 else 2000

    def no_gain(op, n, arm):
        return 1000 if arm == "par" else 3000

    with tempfile.TemporaryDirectory() as tmp:

        def mk(name, fn):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            base._machine(d, fn)
            return d

        res = run_single(mk("gain", gain))
        assert res["verdict"] == "PROCEED" and res["T"] == 8192, res["verdict"]
        res = run_single(mk("rej", no_gain))
        assert res["verdict"] == "REJECT" and res["T"] == 640
        # 欠損ファイルは入力不正（fail-closed）
        d = mk("miss", gain)
        os.remove(os.path.join(d, "run3.log"))
        try:
            run_single(d)
        except (base.InputError, OSError):
            pass
        else:
            raise AssertionError("missing run must fail")
        # checksum 不一致は FAIL
        d = mk("ck", gain)
        p = os.path.join(d, "run2.log")
        with open(p, encoding="utf-8") as f:
            t = f.read()
        t = t.replace(
            "arm=par median_ns=1000 threads=8 checksum=0x1",
            "arm=par median_ns=1000 threads=8 checksum=0x2",
            1,
        )
        with open(p, "w", encoding="utf-8") as f:
            f.write(t)
        assert run_single(d)["verdict"] == "FAIL"
    print("self-test OK")


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--macmini", help="Mac mini の run1..5.log のあるディレクトリ")
    ap.add_argument("--out", help="aggregate.md の出力先（絶対パス）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.self_test:
        self_test()
        return 0
    if not args.macmini:
        print("error: --macmini が必要", file=sys.stderr)
        return 2
    try:
        result = run_single(args.macmini)
    except (base.InputError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    text = base.render(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return base.EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
