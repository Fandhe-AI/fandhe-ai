#!/usr/bin/env python3
"""Mac mini（<metal-node>）の計測 round の前後で、計測対象以外の重いジョブがいないことを 1 回確かめる前提ゲート。

役割: 2026-11-10 に hw-lock を取らない他ジョブ（wgpu の GPU GEMM・CPU 100%・caffeinate 付き）が
#4151・#4153・#4154 の計測と重なり、load1 < 3.0 のゲートでは検出できなかった（所有者判断: 各 issue の
2026-11-10 のコメント「他ジョブと重なった系列は測り直す」）。本スクリプトはその再発を防ぐため、
`round_with_foreign_gate.sh` から round の開始前と終了後に呼ばれ、ps と GPU 使用率の 1 回の読み取りで判定する。
閾値・分類式・確度の規則には触れない（ゲートを厳しくする方向だけの追加）。

判定（どれか 1 つでも当たれば FOREIGN。基準は RULE と README に同じ文言で固定する）:
  G1 caffeinate   実行ファイルの basename が `caffeinate` のプロセスがある（ユーザーを問わない）
  G2 cpu          %CPU（ps の pcpu）が CPU_THRESHOLD 以上のプロセスがある（ユーザーを問わない）
  G3 name         basename（小文字化）が HEAVY_NAME_PATTERNS のどれかに当たるプロセスがある（ユーザー・%CPU を問わない）
  G4 gpu          IOAccelerator の "Device Utilization %" を GPU_SAMPLES 回（GPU_INTERVAL_S 秒間隔）読み、
                  GPU_THRESHOLD 以上の回が GPU_MIN_HITS 回以上（1 回だけの揺れ・直前の自分の GPU 処理の残りでは当たらず、
                  続いている GPU 負荷が当たる。他ジョブの GPU 使用は断続的で、2026-11-10 の実機の 3 回読みは 97・0・0 だった）
除外: 自分自身・自分の祖先（呼び出し元の wrapper・driver を含む）・自分の子孫（ps・ioreg 自身）と、pid 0（kernel_task）・pid 1（launchd）。
  名前で除外しない（自分の計測が python や cargo を動かしても、round の前後には生きていない。生きていれば他ジョブと同じく当てる）。
限界: ps の %CPU は減衰平均で、起動直後のジョブは低く出る（G3・G4 で補う）。round の途中で始まって途中で終わったジョブは検出できない
  （前後の 2 時点しか見ない）。GPU 使用率は機体全体の値で、どのプロセスかは分からない。

出力（--out。マスク済み）: 1 行目 `CLEAN|FOREIGN <phase> <UTC>`、続けて判定ごとの当たり・GPU の各回の値・
  走査したプロセス数・%CPU 上位 5 件（basename と pcpu）。ユーザー名・$HOME・ホスト名は置換し、自分以外の uid は数値のまま書く。
終了コード: 0 CLEAN／1 FOREIGN／2 読み取り・解釈の失敗（fail-closed。呼び出し側は round を始めない・INVALID にする）。
使い方: python3 -I foreign_job_gate.py --phase before|after --out <file>
        python3 -I foreign_job_gate.py --self-test
標準ライブラリのみ。
"""
import argparse
import os
import pwd
import re
import socket
import subprocess
import sys
import time

CPU_THRESHOLD = 25.0
GPU_THRESHOLD = 10
GPU_SAMPLES = 5
GPU_MIN_HITS = 2
GPU_INTERVAL_S = 1.0
HEAVY_NAME_PATTERNS = ("wgpu", "python", "cargo", "rustc", "xctrace", "torch", "bench", "caffeinate")
PS_ARGS = ["ps", "-A", "-o", "pid=,ppid=,uid=,pcpu=,comm="]
IOREG_ARGS = ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]
PS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\S.*?)\s*$")
GPU_RE = re.compile(r'"Device Utilization %"=(\d+)')
SYSTEM_PIDS = (0, 1)


class GateError(Exception):
    pass


def parse_ps(text):
    """ps の出力を [(pid, ppid, uid, pcpu, comm)] にする。1 行でも形が違えば止まる。"""
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        m = PS_RE.match(line)
        if not m:
            raise GateError(f"ps {ln} 行目の形式が想定外")
        rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)), m.group(5)))
    if not rows:
        raise GateError("ps の出力が空")
    pids = [r[0] for r in rows]
    if len(set(pids)) != len(pids):
        raise GateError("ps に pid の重複がある")
    return rows


def parse_gpu(text):
    """ioreg の出力から Device Utilization % を 1 つだけ取る（0 個・2 個以上は止まる）。"""
    got = GPU_RE.findall(text)
    if len(got) != 1:
        raise GateError(f'"Device Utilization %" が {len(got)} 個')
    v = int(got[0])
    if not 0 <= v <= 100:
        raise GateError(f"Device Utilization % が範囲外（{v}）")
    return v


def basename(comm):
    return comm.rstrip("/").rsplit("/", 1)[-1]


def display_name(comm):
    """記録に書く名前。basename のうち最初の `:` より前だけ（`sshd-session: <名前> [priv]` のようなプロセスタイトルが
    他ユーザー名を含みうるため、`:` 以降は書かない）。判定（G1・G3）は basename 全体で行う。"""
    return basename(comm).split(":", 1)[0].strip() or "?"


def lineage(rows, self_pid):
    """自分の祖先（pid 1 まで）と子孫の pid 集合。自分が ps に居なければ止まる。"""
    parent = {r[0]: r[1] for r in rows}
    if self_pid not in parent:
        raise GateError("自分の pid が ps に無い")
    keep = {self_pid}
    p = self_pid
    seen = set()
    while p in parent and p not in seen:
        seen.add(p)
        p = parent[p]
        keep.add(p)
    children = {}
    for pid, ppid, *_ in rows:
        children.setdefault(ppid, []).append(pid)
    stack = [self_pid]
    while stack:
        c = stack.pop()
        for k in children.get(c, []):
            if k not in keep:
                keep.add(k)
                stack.append(k)
    return keep


def judge(rows, self_pid, gpu_samples):
    """判定の本体（副作用なし）。(foreign: bool, hits: [(rule, pid, uid, pcpu, base)], gpu_hit_count)。"""
    excl = lineage(rows, self_pid) | set(SYSTEM_PIDS)
    hits = []
    for pid, _ppid, uid, pcpu, comm in rows:
        if pid in excl:
            continue
        b = basename(comm)
        low = b.lower()
        if low == "caffeinate":
            hits.append(("G1-caffeinate", pid, uid, pcpu, display_name(comm)))
        if pcpu >= CPU_THRESHOLD:
            hits.append(("G2-cpu", pid, uid, pcpu, display_name(comm)))
        if any(p in low for p in HEAVY_NAME_PATTERNS) and low != "caffeinate":
            hits.append(("G3-name", pid, uid, pcpu, display_name(comm)))
    if len(gpu_samples) != GPU_SAMPLES:
        raise GateError(f"GPU の読み取りが {len(gpu_samples)} 回（期待 {GPU_SAMPLES}）")
    ghits = sum(1 for v in gpu_samples if v >= GPU_THRESHOLD)
    if ghits >= GPU_MIN_HITS:
        hits.append(("G4-gpu", -1, -1, float(ghits), "Device Utilization %"))
    return bool(hits), hits, ghits


class Masker:
    def __init__(self, user, home, hosts):
        self.pairs = [(home, "<home>")] if home else []
        self.pairs += [(h, "<metal-node>") for h in hosts if h]
        if user:
            self.pairs.append((user, "<user>"))

    def __call__(self, s):
        for a, b in self.pairs:
            s = s.replace(a, b)
        return s


def render(phase, when, foreign, hits, gpu_samples, rows, self_uid, mask):
    def who(uid):
        return "<user>" if uid == self_uid else f"uid={uid}"
    out = [f"{'FOREIGN' if foreign else 'CLEAN'} {phase} {when}"]
    out.append(f"criteria cpu>={CPU_THRESHOLD} gpu>={GPU_THRESHOLD}x{GPU_MIN_HITS}of{GPU_SAMPLES} "
               f"names={','.join(HEAVY_NAME_PATTERNS)} caffeinate excl=self+ancestors+descendants+pid0,1")
    out.append("gpu_device_utilization " + " ".join(str(v) for v in gpu_samples))
    out.append(f"processes_scanned {len(rows)}")
    for rule, pid, uid, pcpu, b in hits:
        if rule == "G4-gpu":
            out.append(f"hit {rule} samples_at_or_above={int(pcpu)}")
        else:
            out.append(f"hit {rule} pid={pid} {who(uid)} pcpu={pcpu:.1f} name={b}")
    top = sorted(rows, key=lambda r: (-r[3], r[0]))[:5]
    for pid, _pp, uid, pcpu, comm in top:
        out.append(f"top pid={pid} {who(uid)} pcpu={pcpu:.1f} name={display_name(comm)}")
    return mask("\n".join(out) + "\n")


def run_cmd(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise GateError(f"{args[0]} が rc={r.returncode}")
    return r.stdout


def live(phase, out_path):
    if os.path.exists(out_path):
        raise GateError("出力先が既にある（上書きしない）")
    gpu = []
    for i in range(GPU_SAMPLES):
        if i:
            time.sleep(GPU_INTERVAL_S)
        gpu.append(parse_gpu(run_cmd(IOREG_ARGS)))
    rows = parse_ps(run_cmd(PS_ARGS))
    foreign, hits, _ = judge(rows, os.getpid(), gpu)
    pw = pwd.getpwuid(os.getuid())
    hosts = sorted({socket.gethostname(), socket.gethostname().split(".")[0]}, key=len, reverse=True)
    text = render(phase, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), foreign, hits, gpu, rows,
                  os.getuid(), Masker(pw.pw_name, os.path.expanduser("~"), hosts))
    tmp = out_path + ".tmp"
    with open(tmp, "x", encoding="utf-8") as f:
        f.write(text)
    os.rename(tmp, out_path)
    return 1 if foreign else 0


def self_test():
    # 合成 fixture（実在のバイナリ名・パスは使わない）。自分 = pid 500（親 400 = wrapper、400 の親 300 = driver、300 の親 1）
    base = "\n".join([
        "    0     0     0  4.0 kernel_task",
        "    1     0     0 30.0 /sbin/launchd",
        "  300     1   501  0.0 /bin/bash",
        "  400   300   501  0.0 /bin/bash",
        "  500   400   501 99.0 /usr/bin/python3",
        "  510   500   501 50.0 /bin/ps",
        "  600     1    88  1.0 /System/Library/WindowServer",
        "  700     1   501 15.9 /Applications/Some.app/Contents/MacOS/some-backend",
        "  800     1     0 11.7 sshd-session: someone [priv]",
    ]) + "\n"
    rows = parse_ps(base)
    assert lineage(rows, 500) == {500, 400, 300, 1, 0, 510}
    f, h, g = judge(rows, 500, [0, 0, 0, 0, 0])
    assert not f and h == [] and g == 0, h  # 平常: 自分の系譜・pid 0/1 の高 CPU は当てない、15.9% は閾値未満

    def j(extra, gpu=(0, 0, 0, 0, 0)):
        return judge(parse_ps(base + extra), 500, list(gpu))

    f, h, _ = j("  900     1   501  0.0 /usr/bin/caffeinate\n")
    assert f and [x[0] for x in h] == ["G1-caffeinate"], h
    f, h, _ = j("  901     1   501 25.0 /opt/x/synthetic-solver\n")
    assert f and [x[0] for x in h] == ["G2-cpu"], h  # 境界: 25.0 は当たる
    f, h, _ = j("  902     1   501 24.9 /opt/x/synthetic-solver\n")
    assert not f, h  # 境界: 24.9 は当たらない
    f, h, _ = j("  903     1   502  0.0 /opt/x/Fake-WGPU-gate\n")
    assert f and [x[0] for x in h] == ["G3-name"], h  # 名前は大文字小文字を問わず、別ユーザーでも当てる
    f, h, _ = j("  904     1   501  0.1 /usr/local/bin/python3.12\n")
    assert f and [x[0] for x in h] == ["G3-name"], h
    f, h, _ = j("  905     1   501 100.0 /opt/x/synthetic-gemm-bench\n")
    assert f and sorted(x[0] for x in h) == ["G2-cpu", "G3-name"], h
    f, h, _ = j("  906   400   501 100.0 /opt/x/synthetic-gemm-bench\n")
    assert f, h  # wrapper（祖先）の別の子は自分の子孫ではないので当てる（取り残された計測プロセスも他ジョブ扱い）
    f, h, _ = j("  907   510   501 100.0 /opt/x/cargo\n")
    assert not f, h  # 自分の子孫は除外
    f, h, g = j("", gpu=(88, 87, 90, 0, 91))
    assert f and [x[0] for x in h] == ["G4-gpu"] and g == 4, h
    f, h, _ = j("", gpu=(97, 0, 0, 0, 0))
    assert not f, h  # 1 回だけ（2026-11-10 の実機の読み取りの形）は当てない
    f, h, _ = j("", gpu=(10, 0, 0, 9, 10))
    assert f, h  # 境界: 10 が 2 回で当たる
    f, h, _ = j("", gpu=(9, 9, 9, 9, 9))
    assert not f, h  # 境界: 9 は当たらない
    for bad in ("", "abc def\n", "  1 0 0 x launchd\n", base + "  500   1 501 0.0 dup\n"):
        try:
            parse_ps(bad)
        except GateError:
            pass
        else:
            raise AssertionError(f"parse_ps が止まらない: {bad!r}")
    try:
        judge(rows, 4242, [0, 0, 0])
    except GateError:
        pass
    else:
        raise AssertionError("自分の pid が無いのに止まらない")
    try:
        judge(rows, 500, [0, 0, 0, 0])
    except GateError:
        pass
    else:
        raise AssertionError("GPU の回数不足で止まらない")
    for bad in ("", '"Device Utilization %"=5 "Device Utilization %"=6', '"Device Utilization %"=101'):
        try:
            parse_gpu(bad)
        except GateError:
            pass
        else:
            raise AssertionError(f"parse_gpu が止まらない: {bad!r}")
    assert parse_gpu('x "Tiler Utilization %"=2,"Device Utilization %"=88,') == 88
    # マスクと出力形式
    f, h, _ = j("  903     1   502  0.0 /opt/x/Fake-WGPU-gate\n  904     1   501  0.1 /h/alice/py/python3\n")
    txt = render("after", "2026-01-01T00:00:00Z", f, h, [0, 1, 2, 3, 4], parse_ps(base), 501,
                 Masker("alice", "/h/alice", ["node-a.local", "node-a"]))
    lines = txt.splitlines()
    assert lines[0] == "FOREIGN after 2026-01-01T00:00:00Z", lines[0]
    assert "hit G3-name pid=903 uid=502 pcpu=0.0 name=Fake-WGPU-gate" in lines
    assert "hit G3-name pid=904 <user> pcpu=0.1 name=python3" in lines
    assert "gpu_device_utilization 0 1 2 3 4" in lines
    # プロセスタイトルの他ユーザー名（someone）は `:` 以降を落として書かない
    f, h, _ = j("  950     1     0 30.0 sshd-session: someone [priv]\n")
    assert [x[4] for x in h] == ["sshd-session"], h
    txt2 = render("before", "2026-01-01T00:00:00Z", f, h, [0, 0, 0, 0, 0], parse_ps(base + "  950     1     0 30.0 sshd-session: someone [priv]\n"),
                  501, Masker("alice", "/h/alice", ["node-a"]))
    assert "someone" not in txt2, txt2
    assert display_name("/x/y/python3") == "python3" and display_name(":") == "?"
    m = Masker("alice", "/h/alice", ["node-a.local", "node-a"])
    assert m("/h/alice/x node-a.local alice") == "<home>/x <metal-node> <user>"
    print("self-test ok")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--phase", choices=("before", "after"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.phase or not a.out:
        ap.error("--phase と --out が必要")
    try:
        return live(a.phase, a.out)
    except (GateError, OSError, subprocess.SubprocessError) as e:
        print(f"FOREIGN-GATE-ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
