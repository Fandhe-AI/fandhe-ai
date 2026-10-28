#!/usr/bin/env python3
"""イシュー #3646 専有系列の合否判定器（標準ライブラリのみ・fail-closed）。

役割: session-scripts/session-3646-excl.sh が書く `gpu_poll.log`（30 秒間隔 poller）と
`gpu_procs.txt`（系列ごとの pre/post 境界スナップショット）を読み、系列ごとに
`<label> excl=pass|fail` を `excl_status.txt` へ書く。RULE.txt「GB10 専有（compute プロセス 0）」と
3 回目 reopen の所有者条件（途中で 1 件でも現れた系列は参考記録へ降格）の機械適用であり、
判定閾値・aggregate.py・各 judge は変更しない。自プロセス由来かどうかでの除外はしない
（pid と comm は記録のみ。除外可否は所有者判断）。

使い方: excl_check.py <session-scripts ディレクトリ> / excl_check.py --self-test
pass 条件（系列ごと）: pre・post が各 1 件、途中の poll が 1 件以上、全記録が rc=0 かつ apps=0。
fail 条件: nvidia-smi の rc 非 0・出力が解析できない（0 件とは扱わない）・記録の欠落・ラベル重複・解析不能な行。
いずれかの系列が fail なら終了コード 1。
"""
import os
import re
import sys
import tempfile

SERIES = ["main", "minflt", "tunables", "taskset", "bridge", "gcpuinf"]
POLL_RE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ series=([a-z]+) rc=(\d+) apps=(\d+|NA)(?: (.*))?$"
)
HEAD_RE = re.compile(r"^== ([a-z]+):(pre|post) \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
UTIL_RE = re.compile(r"^\d+ %$")


def parse_poll(text, errors):
    """poll ログを {series: [(rc, apps, detail)]} へ。ps 継続行は apps>0 の記録の直後だけ許す。"""
    recs = {}
    prev_nonzero = False
    for ln in text.splitlines():
        m = POLL_RE.match(ln)
        if m:
            label, rc, apps, detail = m.group(1), int(m.group(2)), m.group(3), m.group(4) or ""
            recs.setdefault(label, []).append((rc, apps, detail))
            prev_nonzero = apps != "0"
            continue
        if prev_nonzero and ln.strip():
            continue  # apps>0 の記録に添えた ps -o pid,ppid,comm の出力
        if ln.strip():
            errors.append("poll: 解析不能な行: %r" % ln[:80])
    return recs


def parse_snaps(text, errors):
    """境界スナップショットを {(series, pre|post): [(rc, apps 件数 or None)]} へ。"""
    snaps = {}
    cur = None
    state = None
    rc = None
    apps = []

    def close():
        if cur is not None:
            if rc is None or state != "done":
                errors.append("snap %s:%s: rc 行または util 行が無い（解析不能）" % cur)
            snaps.setdefault(cur, []).append((rc, len(apps) if state == "done" else None))

    for ln in text.splitlines():
        m = HEAD_RE.match(ln)
        if m:
            close()
            cur, state, rc, apps = (m.group(1), m.group(2)), "rc", None, []
            continue
        if ln.startswith("== "):
            errors.append("snap: 解析不能なヘッダ: %r" % ln[:80])
            continue
        if cur is None:
            continue
        if state == "rc":
            mm = re.match(r"^rc=(\d+)$", ln)
            if not mm:
                errors.append("snap %s:%s: rc 行が無い: %r" % (cur + (ln[:60],)))
                state = "bad"
                continue
            rc = int(mm.group(1))
            state = "apps"
        elif state == "apps":
            if UTIL_RE.match(ln):
                state = "done"
            elif ln.strip():
                apps.append(ln.strip())
    close()
    return snaps


def judge(poll_text, snap_text):
    """({label: bool}, [理由]) を返す。"""
    errors = []
    poll = parse_poll(poll_text, errors)
    snaps = parse_snaps(snap_text, errors)
    status, reasons = {}, []
    for s in SERIES:
        why = []
        for edge in ("pre", "post"):
            lst = snaps.get((s, edge), [])
            if len(lst) != 1:
                why.append("%s:%s のスナップショットが %d 件（欠落または重複）" % (s, edge, len(lst)))
                continue
            rc, n = lst[0]
            if rc != 0:
                why.append("%s:%s nvidia-smi rc=%s" % (s, edge, rc))
            elif n is None:
                why.append("%s:%s 解析不能" % (s, edge))
            elif n != 0:
                why.append("%s:%s compute-apps が %d 件" % (s, edge, n))
        pl = poll.get(s, [])
        if not pl:
            why.append("%s: 途中の poll が 0 件（欠落）" % s)
        for rc, apps, detail in pl:
            if rc != 0 or apps == "NA":
                why.append("%s: poll で nvidia-smi rc=%d（0 件とは扱わない）" % (s, rc))
            elif apps != "0":
                why.append("%s: poll で compute-apps %s 件: %s" % (s, apps, detail.strip()[:120]))
        why.extend(errors)
        status[s] = not why
        reasons.extend(sorted(set(why)))
    # idle 区間（系列の外）の出現は系列を降格しないが、参考情報として報告する。
    for rc, apps, detail in poll.get("idle", []):
        if rc != 0 or apps != "0":
            reasons.append("(参考) idle 区間で rc=%d apps=%s %s" % (rc, apps, detail.strip()[:80]))
    return status, sorted(set(reasons))


def run(dirpath):
    try:
        with open(os.path.join(dirpath, "gpu_poll.log"), encoding="utf-8") as f:
            poll_text = f.read()
        with open(os.path.join(dirpath, "gpu_procs.txt"), encoding="utf-8") as f:
            snap_text = f.read()
    except OSError as e:
        print("入力を読めない: %s" % e, file=sys.stderr)
        return 2
    status, reasons = judge(poll_text, snap_text)
    with open(os.path.join(dirpath, "excl_status.txt"), "w", encoding="utf-8") as f:
        for s in SERIES:
            f.write("%s excl=%s\n" % (s, "pass" if status[s] else "fail"))
    for r in reasons:
        print(r)
    for s in SERIES:
        print("%s excl=%s" % (s, "pass" if status[s] else "fail"))
    return 0 if all(status.values()) else 1


def _fixture(series_over=None, drop_poll=None, dup_pre=False):
    series_over = series_over or {}
    poll, snap = [], []
    for s in SERIES:
        pre = "== %s:pre 2026-10-28T00:00:00Z\nrc=0\n0 %%\n" % s
        snap.append(pre)
        if dup_pre and s == "main":
            snap.append(pre)
        if s != drop_poll:
            rc, apps, det = series_over.get(s, (0, "0", ""))
            poll.append("2026-10-28T00:00:30Z series=%s rc=%d apps=%s %s" % (s, rc, apps, det))
            if apps not in ("0", "NA"):
                poll.append("    PID    PPID COMMAND\n 123 1 python")
        snap.append("== %s:post 2026-10-28T00:01:00Z\nrc=0\n0 %%\n" % s)
    return "\n".join(poll) + "\n", "".join(snap)


def self_test():
    ok = True

    def expect(name, args, only=None):
        nonlocal ok
        st, _ = judge(*_fixture(**args))
        got = [s for s in SERIES if st[s]]
        exp = [s for s in SERIES if s != only]
        if got != exp:
            print("SELF-TEST FAIL %s: got=%s exp=%s" % (name, got, exp))
            ok = False

    expect("all-empty", {})
    expect("appear-mid", {"series_over": {"bridge": (0, "1", "123, python;")}}, only="bridge")
    expect("rc-nonzero", {"series_over": {"taskset": (9, "NA", "err")}}, only="taskset")
    expect("poll-missing", {"drop_poll": "minflt"}, only="minflt")
    expect("dup-label", {"dup_pre": True}, only="main")
    p, s = _fixture()
    st, _ = judge(p + "garbage line\n", s)  # 解析不能な行は全系列 fail
    if any(st.values()):
        print("SELF-TEST FAIL garbage")
        ok = False
    st, _ = judge(p, s.replace("0 %\n", "", 1))  # util 行欠落は 0 件と扱わず fail
    if st["main"]:
        print("SELF-TEST FAIL util-missing")
        ok = False
    with tempfile.TemporaryDirectory() as d:
        p, s = _fixture()
        with open(os.path.join(d, "gpu_poll.log"), "w") as f:
            f.write(p)
        with open(os.path.join(d, "gpu_procs.txt"), "w") as f:
            f.write(s)
        rc = run(d)
        with open(os.path.join(d, "excl_status.txt")) as f:
            n = f.read().count("excl=pass")
        if rc != 0 or n != 6:
            print("SELF-TEST FAIL run")
            ok = False
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
