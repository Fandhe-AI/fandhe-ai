#!/usr/bin/env python3
"""イシュー #3646 専有系列の合否判定器（標準ライブラリのみ・fail-closed）。

役割: session-scripts/session-3646-excl.sh が書く `gpu_poll.log`（30 秒間隔 poller）と
`gpu_procs.txt`（系列ごとの pre/post 境界スナップショット）を読み、系列ごとに
`<label> excl=pass|fail` を `excl_status.txt` へ書く。RULE.txt「GB10 専有（compute プロセス 0）」と
3 回目 reopen の所有者条件（途中で 1 件でも現れた系列は参考記録へ降格）の機械適用であり、
判定閾値・aggregate.py・各 judge は変更しない。自プロセス由来かどうかでの除外はしない
（pid と comm は記録のみ。除外可否は所有者判断）。

使い方: excl_check.py <session-scripts ディレクトリ> [--series a,b,c] / excl_check.py --self-test
--series は #3671 で追加（系列ラベルを小文字英字のカンマ区切りで指定。省略時は #3646 の既定 SERIES で挙動不変）。
pass 条件（系列ごと）: pre・post が各 1 件、途中の poll が 1 件以上、全記録が rc=0 かつ apps=0、
かつ pre → 各 poll → post の隣接間隔がすべて MAX_GAP_S（60 秒）以下（poller 停止・ログの部分欠落の検出）。
fail 条件: 観測間隔の超過（時刻は poll 行と境界ヘッダのタイムスタンプ）・ nvidia-smi の rc 非 0・出力が解析できない（0 件とは扱わない）・記録の欠落・ラベル重複・解析不能な行。
いずれかの系列が fail なら終了コード 1。
"""
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

# poller は 30 秒間隔（session-3646-excl.sh の POLL_INTERVAL）。実測の最大間隔は 31 秒（173 件）。
# 契約（30 秒間隔で途中の出現を監視する）を満たすかの許容上限は 2 倍の 60 秒とし、超えたら欠落扱いで fail。
POLL_INTERVAL_S = 30
MAX_GAP_S = 2 * POLL_INTERVAL_S

SERIES = ["main", "minflt", "tunables", "taskset", "bridge", "gcpuinf"]
POLL_RE = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) series=([a-z]+) rc=(\d+) apps=(\d+|NA)(?: (.*))?$"
)
HEAD_RE = re.compile(r"^== ([a-z]+):(pre|post) (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$")
UTIL_RE = re.compile(r"^\d+ %$")


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def parse_poll(text, errors):
    """poll ログを {series: [(rc, apps, detail, ts)]} へ。ps 継続行は apps>0 の記録の直後だけ許す。"""
    recs = {}
    prev_nonzero = False
    for ln in text.splitlines():
        m = POLL_RE.match(ln)
        if m:
            label, rc, apps, detail = m.group(2), int(m.group(3)), m.group(4), m.group(5) or ""
            recs.setdefault(label, []).append((rc, apps, detail, parse_ts(m.group(1))))
            prev_nonzero = apps != "0"
            continue
        if prev_nonzero and ln.strip():
            continue  # apps>0 の記録に添えた ps -o pid,ppid,comm の出力
        if ln.strip():
            errors.append("poll: 解析不能な行: %r" % ln[:80])
    return recs


def parse_snaps(text, errors):
    """境界スナップショットを {(series, pre|post): [(rc, apps 件数 or None, ts)]} へ。"""
    snaps = {}
    cur = None
    state = None
    rc = None
    apps = []
    ts = None

    def close():
        if cur is not None:
            if rc is None or state != "done":
                errors.append("snap %s:%s: rc 行または util 行が無い（解析不能）" % cur)
            snaps.setdefault(cur, []).append((rc, len(apps) if state == "done" else None, ts))

    for ln in text.splitlines():
        m = HEAD_RE.match(ln)
        if m:
            close()
            cur, state, rc, apps = (m.group(1), m.group(2)), "rc", None, []
            ts = parse_ts(m.group(3))
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


def gap_failures(s, snaps, pl):
    """pre → poll 群（時刻順）→ post の隣接間隔が MAX_GAP_S を超える区間を fail 理由で返す。
    pre・post のどちらかが欠落（別途 fail）または poll が 0 件（別途 fail）なら判定を省く。"""
    pre, post = snaps.get((s, "pre"), []), snaps.get((s, "post"), [])
    if len(pre) != 1 or len(post) != 1 or not pl:
        return []
    marks = [pre[0][2]] + sorted(r[3] for r in pl) + [post[0][2]]
    why = []
    for a, b in zip(marks, marks[1:]):
        gap = (b - a).total_seconds()
        if gap < 0:
            why.append("%s: 観測時刻が pre/post の範囲外または逆順（%s → %s）" % (s, a.strftime("%FT%TZ"), b.strftime("%FT%TZ")))
        elif gap > MAX_GAP_S:
            why.append("%s: 観測間隔 %d 秒が上限 %d 秒を超過（%s → %s。poller 停止またはログ欠落）" % (s, gap, MAX_GAP_S, a.strftime("%FT%TZ"), b.strftime("%FT%TZ")))
    return why


def judge(poll_text, snap_text, series=None):
    """({label: bool}, [理由]) を返す。series 省略時は既定 SERIES。"""
    series = SERIES if series is None else series
    errors = []
    poll = parse_poll(poll_text, errors)
    snaps = parse_snaps(snap_text, errors)
    status, reasons = {}, []
    for s in series:
        why = []
        for edge in ("pre", "post"):
            lst = snaps.get((s, edge), [])
            if len(lst) != 1:
                why.append("%s:%s のスナップショットが %d 件（欠落または重複）" % (s, edge, len(lst)))
                continue
            rc, n, _ = lst[0]
            if rc != 0:
                why.append("%s:%s nvidia-smi rc=%s" % (s, edge, rc))
            elif n is None:
                why.append("%s:%s 解析不能" % (s, edge))
            elif n != 0:
                why.append("%s:%s compute-apps が %d 件" % (s, edge, n))
        pl = poll.get(s, [])
        if not pl:
            why.append("%s: 途中の poll が 0 件（欠落）" % s)
        for rc, apps, detail, _ in pl:
            if rc != 0 or apps == "NA":
                why.append("%s: poll で nvidia-smi rc=%d（0 件とは扱わない）" % (s, rc))
            elif apps != "0":
                why.append("%s: poll で compute-apps %s 件: %s" % (s, apps, detail.strip()[:120]))
        why.extend(gap_failures(s, snaps, pl))
        why.extend(errors)
        status[s] = not why
        reasons.extend(sorted(set(why)))
    # idle 区間（系列の外）の出現は系列を降格しないが、参考情報として報告する。
    for rc, apps, detail, _ in poll.get("idle", []):
        if rc != 0 or apps != "0":
            reasons.append("(参考) idle 区間で rc=%d apps=%s %s" % (rc, apps, detail.strip()[:80]))
    return status, sorted(set(reasons))


def run(dirpath, series=None):
    series = SERIES if series is None else series
    try:
        with open(os.path.join(dirpath, "gpu_poll.log"), encoding="utf-8") as f:
            poll_text = f.read()
        with open(os.path.join(dirpath, "gpu_procs.txt"), encoding="utf-8") as f:
            snap_text = f.read()
    except OSError as e:
        print("入力を読めない: %s" % e, file=sys.stderr)
        return 2
    status, reasons = judge(poll_text, snap_text, series)
    with open(os.path.join(dirpath, "excl_status.txt"), "w", encoding="utf-8") as f:
        for s in series:
            f.write("%s excl=%s\n" % (s, "pass" if status[s] else "fail"))
    for r in reasons:
        print(r)
    for s in series:
        print("%s excl=%s" % (s, "pass" if status[s] else "fail"))
    return 0 if all(status.values()) else 1


def _fixture(series_over=None, drop_poll=None, dup_pre=False, poll_times=None, series=None):
    series = SERIES if series is None else series
    series_over = series_over or {}
    poll, snap = [], []
    for s in series:
        pre = "== %s:pre 2026-10-28T00:00:00Z\nrc=0\n0 %%\n" % s
        snap.append(pre)
        if dup_pre and s == series[0]:
            snap.append(pre)
        if s != drop_poll:
            rc, apps, det = series_over.get(s, (0, "0", ""))
            for t in poll_times if poll_times is not None else ("00:00:30",):
                poll.append("2026-10-28T%sZ series=%s rc=%d apps=%s %s" % (t, s, rc, apps, det))
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
    # 時間的欠落: 30 秒間隔で観測した系列は pass、最初の 1 件だけに間引いた系列は fail。
    st, _ = judge(*_fixture(poll_times=("00:00:30", "00:01:00")))
    if not all(st.values()):
        print("SELF-TEST FAIL dense-poll")
        ok = False
    p, s = _fixture(poll_times=("00:00:30",))
    s = s.replace("00:01:00Z", "00:02:30Z")  # post を遅らせ、最初の 1 件だけの間引きを再現
    st, _ = judge(p, s)
    if any(st.values()):
        print("SELF-TEST FAIL thinned-poll")
        ok = False
    st, _ = judge(*_fixture(poll_times=("00:00:30", "00:02:00")))  # 途中 90 秒の欠落
    if any(st.values()):
        print("SELF-TEST FAIL mid-gap")
        ok = False
    st, _ = judge(*_fixture(poll_times=("00:01:01",)))  # pre 直後 61 秒の欠落
    if any(st.values()):
        print("SELF-TEST FAIL head-gap")
        ok = False
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
    # --series: 任意ラベルの系列でも同じ判定が働く（#3671）。
    custom = ["diag", "solo"]
    st, _ = judge(*_fixture(series=custom), custom)
    if list(st) != custom or not all(st.values()):
        print("SELF-TEST FAIL custom-series-pass")
        ok = False
    st, _ = judge(*_fixture(series=custom, series_over={"solo": (0, "1", "9, x;")}), custom)
    if st["diag"] is not True or st["solo"]:
        print("SELF-TEST FAIL custom-series-fail")
        ok = False
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    argv = sys.argv[1:]
    series_arg = None
    if len(argv) == 3 and argv[1] == "--series":
        series_arg = argv[2].split(",")
        if not all(re.fullmatch(r"[a-z]+", x) for x in series_arg) or len(set(series_arg)) != len(series_arg):
            print("--series は重複しない小文字英字のカンマ区切り", file=sys.stderr)
            sys.exit(2)
        argv = argv[:1]
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(run(argv[0], series_arg))
