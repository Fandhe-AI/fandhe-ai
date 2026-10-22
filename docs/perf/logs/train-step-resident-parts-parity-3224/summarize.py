#!/usr/bin/env python3
"""イシュー #3224: run_parity.sh が出したログを厳密に検査して summary.md を作る。

役割: <name>.run<N>.log（N = 1..5）がすべて揃っていること、末尾 rc=0、
`running K tests` が期待件数と一致、`test result: ok. K passed; 0 failed` を検査する。
欠損・不一致・パース不能は例外にして非 0 終了する（寛容なパースはしない。RULE.txt）。
fail したログは「fail」として表に出す（選別しない）。invalid マーカーがあれば表に出す。

使い方: python3 -I summarize.py <metal|cuda> <ディレクトリ>   /   python3 -I summarize.py --self-test
標準ライブラリのみ。
"""
import os
import re
import sys
import tempfile

RUNS = 5
# 名前 -> 期待テスト件数（running K tests の K）。RULE.txt の M1〜M5／C1〜C5 に対応する。
EXPECTED = {
    "metal": {
        "relu_mask_device_parity": 6,
        "mse_device_parity": 5,
        "gemm_device_input_parity": 6,
        "train_step_parts_no_wait_3224": 2,
        "gemm_device_tracked_token": 1,
    },
    "cuda": {
        "relu_mask_device_parity": 5,
        "mse_device_parity": 5,
        "gemm_device_input_parity": 7,
        "gemm_device_input_diag_tests_3223": 1,
        "train_step_parts_sync_diag_tests_3224": 4,
    },
}


class SummaryError(Exception):
    pass


def parse_log(path, expected):
    """ログ 1 本を検査して (status, detail) を返す。構造が崩れていれば SummaryError。"""
    if not os.path.isfile(path):
        raise SummaryError("missing log: %s" % path)
    text = open(path, encoding="utf-8", errors="strict").read()
    lines = text.rstrip("\n").split("\n")
    m = re.fullmatch(r"rc=(\d+)", lines[-1]) if lines else None
    if not m:
        raise SummaryError("no trailing rc= line: %s" % path)
    rc = int(m.group(1))
    running = re.findall(r"^running (\d+) tests?$", text, re.M)
    if len(running) != 1:
        raise SummaryError("expected exactly one 'running N tests' line: %s" % path)
    k = int(running[0])
    if expected is not None and k != expected:
        raise SummaryError("running %d tests, expected %d: %s" % (k, expected, path))
    res = re.findall(
        r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;", text, re.M
    )
    if len(res) != 1:
        raise SummaryError("expected exactly one 'test result' line: %s" % path)
    verdict, passed, failed = res[0][0], int(res[0][1]), int(res[0][2])
    if passed + failed != k:
        raise SummaryError("passed+failed != running: %s" % path)
    if rc == 0 and (verdict != "ok" or failed != 0):
        raise SummaryError("rc=0 but test result is not ok: %s" % path)
    if rc != 0 and verdict == "ok":
        raise SummaryError("rc!=0 but test result is ok: %s" % path)
    status = "pass" if (rc == 0 and verdict == "ok" and failed == 0) else "fail"
    return status, "%d passed / %d failed (rc=%d)" % (passed, failed, rc)


def summarize(platform, directory):
    rows = []
    overall_fail = False
    for name, expected in EXPECTED[platform].items():
        cells = []
        for n in range(1, RUNS + 1):
            path = os.path.join(directory, "%s.run%d.log" % (name, n))
            status, detail = parse_log(path, expected)
            if os.path.exists(os.path.join(directory, "%s.run%d.invalid" % (name, n))):
                status += " (INVALID)"
                overall_fail = True
            if not status.startswith("pass"):
                overall_fail = True
            cells.append("%s: %s" % (status, detail))
        rows.append((name, cells))
    out = ["# %s 集計（自動生成）" % platform, "", "| テスト | " + " | ".join("run%d" % i for i in range(1, RUNS + 1)) + " |",
           "|---|" + "---|" * RUNS]
    for name, cells in rows:
        out.append("| %s | %s |" % (name, " | ".join(cells)))
    out.append("")
    # 取り直しで attempts/ へ退避された invalid 試行（件数と理由をすべて出す）。
    adir = os.path.join(directory, "attempts")
    inv = sorted(f for f in os.listdir(adir) if f.endswith(".invalid")) if os.path.isdir(adir) else []
    out.append("退避済み invalid 試行: %d 件" % len(inv))
    for f in inv:
        with open(os.path.join(adir, f), encoding="utf-8") as fh:
            out.append("- %s: %s" % (f, fh.read().strip()))
    out.append("")
    out.append("総合: %s" % ("fail（または invalid を含む）" if overall_fail else "pass（全テスト x 5 run）"))
    return "\n".join(out) + "\n", overall_fail


def _write(directory, name, n, body):
    with open(os.path.join(directory, "%s.run%d.log" % (name, n)), "w", encoding="utf-8") as f:
        f.write(body)


GOOD = "running 2 tests\ntest a ... ok\ntest result: ok. 2 passed; 0 failed; 0 ignored\nrc=0\n"


def self_test():
    def expect_error(mutate, label):
        with tempfile.TemporaryDirectory() as d:
            for name in EXPECTED["metal"]:
                for n in range(1, RUNS + 1):
                    _write(d, name, n, GOOD.replace("running 2", "running %d" % (EXPECTED["metal"][name])).replace("2 passed", "%d passed" % EXPECTED["metal"][name]))
            mutate(d)
            try:
                summarize("metal", d)
            except SummaryError:
                return
            raise AssertionError("expected SummaryError: " + label)

    with tempfile.TemporaryDirectory() as d:
        for name, k in EXPECTED["metal"].items():
            for n in range(1, RUNS + 1):
                _write(d, name, n, "running %d tests\ntest result: ok. %d passed; 0 failed;\nrc=0\n" % (k, k))
        text, fail = summarize("metal", d)
        assert not fail and "pass（全テスト" in text
    expect_error(lambda d: os.remove(os.path.join(d, "mse_device_parity.run3.log")), "missing log")
    expect_error(lambda d: _write(d, "mse_device_parity", 2, "running 5 tests\ntest result: ok. 5 passed; 0 failed;\n"), "no rc")
    expect_error(lambda d: _write(d, "mse_device_parity", 2, "running 4 tests\ntest result: ok. 4 passed; 0 failed;\nrc=0\n"), "running count")
    expect_error(lambda d: _write(d, "mse_device_parity", 2, "running 5 tests\ntest result: ok. 5 passed; 0 failed;\nrc=101\n"), "rc!=0 with ok")
    expect_error(lambda d: _write(d, "mse_device_parity", 2, "garbage\nrc=0\n"), "unparseable")
    # fail は例外ではなく fail として集計される。
    with tempfile.TemporaryDirectory() as d:
        for name, k in EXPECTED["metal"].items():
            for n in range(1, RUNS + 1):
                _write(d, name, n, "running %d tests\ntest result: ok. %d passed; 0 failed;\nrc=0\n" % (k, k))
        _write(d, "mse_device_parity", 1, "running 5 tests\ntest result: FAILED. 4 passed; 1 failed;\nrc=101\n")
        text, fail = summarize("metal", d)
        assert fail and "fail:" in text
    print("self-test ok")


def main(argv):
    if argv == ["--self-test"]:
        self_test()
        return 0
    if len(argv) != 2 or argv[0] not in EXPECTED:
        print(__doc__)
        return 2
    try:
        text, fail = summarize(argv[0], argv[1])
    except SummaryError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
    with open(os.path.join(argv[1], "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
