#!/usr/bin/env python3
"""イシュー #3201 の実測ログ（macmini／gb10）を fail-closed で集計し Markdown を標準出力へ出す。

判定規則は同ディレクトリ RULE.txt・RULE-addendum.txt（計測前に固定済み）。本スクリプトは規則を機械的に
実装するだけで、閾値・セル・ガードセルの扱いを変えない（4 値: ADOPT／REJECT／undetermined／FAIL）。
欠損・数値でない値・run の不足／過剰・腕×round の重複・checksum 不一致（FAIL として出す）・
rayon_threads の腕内食い違い・必須成果物（.rev-stamp／tree_verify.txt／patch_sha256.txt／prereq／bitmatch.log／DONE）
の不在は例外にし exit 2。gate=unavailable と明記された (series, run, arm) だけは数値を補完せず
unavailable として出す（参考系列）。機体ディレクトリ自体が無い場合は「未計測」と出し判定は undetermined。
前例: ../cpu-jobgrid-3177/aggregate.py。標準ライブラリのみ。`python3 -I aggregate.py [--dir DIR] [--machines a,b]`・
`--self-test`。
"""
import json
import os
import re
import statistics
import sys
import tempfile

RUNS = [1, 2, 3, 4, 5]
SHAPES = {  # label: (trans, m, n, k)。ref_square_512 は参照用で判定に使わない
    "l1_fwd": ("nn", 64, 256, 784),
    "l1_dw": ("tn", 784, 256, 64),
    "l1_dx": ("nt", 64, 784, 256),
    "l2_fwd": ("nn", 64, 10, 256),
    "l2_dw": ("tn", 256, 10, 64),
    "l2_dx": ("nt", 64, 256, 10),
}
REF_SHAPE = ("ref_square_512", ("nn", 512, 512, 512))
FC_CELLS = [(t, m) for t in ("train", "infer") for m in ("fresh", "reuse")]
MACHINE_ARMS = {
    "macmini": ["off", "merge", "mergeprepack"],
    "gb10": ["off", "merge", "mergeprepack", "off-aff", "merge-aff", "mergeprepack-aff"],
}
# 腕 -> (G1, G2, aff) 期待ゲート（orchestrate.sh と同一）
ARM_GATES = {
    "off": (False, False, False), "merge": (True, False, False), "mergeprepack": (True, True, False),
    "off-aff": (False, False, True), "merge-aff": (True, False, True), "mergeprepack-aff": (True, True, True),
}
THRESHOLD = 1.00


class AggError(Exception):
    pass


def kv(line, path):
    d = dict(re.findall(r"(\w+)=(\S+)", line))
    if not d:
        raise AggError(f"{path}: key=value を含まない行: {line!r}")
    return d


def pos_float(v, what):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise AggError(f"{what}: 数値でない: {v!r}")
    if not (x == x) or x in (float("inf"), float("-inf")) or x <= 0:
        raise AggError(f"{what}: 正の有限値でない: {x}")
    return x


def need(path):
    if not os.path.isfile(path):
        raise AggError(f"欠損: {path}")
    return path


def read_text(path):
    with open(need(path), encoding="utf-8") as f:
        return f.read()


def parse_gate(root, arms):
    """load_gate.log -> {(series, run, arm): gate}。期待集合と厳密照合（欠落・重複・未知の組は例外）。"""
    text = read_text(os.path.join(root, "load_gate.log"))
    gate, orders = {}, {}
    for ln in text.splitlines():
        if ln.startswith("START "):
            d = kv(ln, "load_gate.log")
            key = (d.get("series"), int(d.get("run", "-1")) if d.get("run", "").isdigit() else -1, d.get("arm"))
            if d.get("gate") not in ("pass", "unavailable"):
                raise AggError(f"load_gate.log: gate 値が不正: {ln!r}")
            if key in gate:
                raise AggError(f"load_gate.log: START 重複: {key}")
            gate[key] = d["gate"]
        elif ln.startswith("ORDER "):
            d = kv(ln, "load_gate.log")
            if not d.get("run", "").isdigit() or int(d["run"]) in orders:
                raise AggError(f"load_gate.log: ORDER が不正または重複: {ln!r}")
            mo = re.search(r"order=(.*)$", ln)
            orders[int(d["run"])] = mo.group(1).split() if mo else []
    expect = {("ab", r, a) for r in RUNS for a in arms} | {("torch", r, "torch") for r in RUNS}
    if set(gate) != expect:
        raise AggError(f"load_gate.log: START の組が期待と不一致（欠落 {sorted(expect - set(gate))[:3]} / 余剰 {sorted(set(gate) - expect)[:3]}）")
    for r in RUNS:
        if r not in orders:
            raise AggError(f"load_gate.log: ORDER run={r} が無い")
        o = orders[r]
        if sorted(o) != sorted(arms):
            raise AggError(f"load_gate.log: ORDER run={r} の腕集合が不一致: {o}")
        n = len(arms)
        if o != arms[(r - 1) % n:] + arms[:(r - 1) % n]:
            raise AggError(f"load_gate.log: ORDER run={r} がローテーションでない: {o}")
    return gate


def parse_jobgrid(root, arms, gate):
    """-> {arm: {shape: {run: (median, checksum)}}}、{arm: rayon_threads}。"""
    data = {a: {s: {} for s in list(SHAPES) + [REF_SHAPE[0]]} for a in arms}
    threads = {}
    table = dict(SHAPES, **{REF_SHAPE[0]: REF_SHAPE[1]})
    for a in arms:
        for r in RUNS:
            p = os.path.join(root, "ab", f"run{r}-{a}.log")
            if gate[("ab", r, a)] == "unavailable":
                if os.path.exists(p):
                    raise AggError(f"{p}: gate=unavailable なのにログが存在する")
                continue
            need(p)
            seen = set()
            for ln in read_text(p).splitlines():
                if not ln or ln.startswith("#"):
                    continue
                if not ln.startswith("arm="):
                    raise AggError(f"{p}: 想定外の行: {ln!r}")
                d = kv(ln, p)
                if d.get("arm") != a or d.get("series") != "threads":
                    raise AggError(f"{p}: arm/series が不一致: {d.get('arm')}/{d.get('series')}")
                shape = d.get("shape")
                if shape not in table or shape in seen:
                    raise AggError(f"{p}: 未知または重複の shape: {shape}")
                seen.add(shape)
                trans, m, n, k = table[shape]
                if (d.get("trans"), d.get("m"), d.get("n"), d.get("k")) != (trans, str(m), str(n), str(k)):
                    raise AggError(f"{p}: {shape} の寸法・転置が期待と不一致")
                if not re.fullmatch(r"\d+", d.get("rayon_threads", "")):
                    raise AggError(f"{p}: rayon_threads が整数でない")
                rt = int(d["rayon_threads"])
                if threads.setdefault(a, rt) != rt:
                    raise AggError(f"{p}: rayon_threads が腕内で食い違う（{threads[a]} vs {rt}）")
                if not re.fullmatch(r"0x[0-9a-f]{16}", d.get("checksum", "")):
                    raise AggError(f"{p}: checksum が 0x16 桁でない: {d.get('checksum')!r}")
                data[a][shape][r] = (pos_float(d.get("median_secs"), f"{p} {shape} median_secs"), d["checksum"])
            if seen != set(table):
                raise AggError(f"{p}: shape が揃っていない: {sorted(set(table) - seen)}")
    return data, threads


def load_jsonl(path, optional=False):
    """optional=True（利用可能 run が 0 件の腕・系列）に限り、ファイル不在を空データとして扱う。"""
    rows = []
    if optional and not os.path.exists(path):
        return rows
    with open(need(path), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    raise AggError(f"{path}: JSON でない行")
    return rows


def parse_fc(root, arms, gate):
    """-> {arm: {(task,mode): {run: (median, checksum)}}}、{('torch', task): {run: median}}。"""
    data = {a: {c: {} for c in FC_CELLS} for a in arms}
    meta = set()
    for a in arms:
        avail = [r for r in RUNS if gate[("ab", r, a)] == "pass"]
        for task in ("train", "infer"):
            rows = load_jsonl(os.path.join(root, "fc", f"results-{a}-{task}.jsonl"), optional=not avail)
            per = {"fresh": [], "reuse": []}
            for o in rows:
                if o.get("framework") != "fandhe-ai" or o.get("task") != task or o.get("device") != "cpu" or o.get("size") != 64:
                    raise AggError(f"results-{a}-{task}.jsonl: framework/task/device/size が不一致: {o}")
                if o.get("mode") not in per:
                    raise AggError(f"results-{a}-{task}.jsonl: mode が不正: {o.get('mode')!r}")
                meta.add((o.get("version"), o.get("warmup"), o.get("iters"), task))
                per[o["mode"]].append(o)
            for mode, lst in per.items():
                if len(lst) != len(avail):
                    raise AggError(f"results-{a}-{task}.jsonl: mode={mode} の行数 {len(lst)} が利用可能 run 数 {len(avail)} と不一致")
                for r, o in zip(avail, lst):
                    if not isinstance(o.get("checksum"), (int, float)) or isinstance(o.get("checksum"), bool):
                        raise AggError(f"results-{a}-{task}.jsonl: checksum が数値でない")
                    data[a][(task, mode)][r] = (pos_float(o.get("median_s"), f"{a} {task} {mode} median_s"), o["checksum"])
    versions = {(v, w, i, t) for (v, w, i, t) in meta}
    for t in ("train", "infer"):
        if len({(v, w, i) for (v, w, i, tt) in versions if tt == t}) != 1:
            raise AggError(f"fc {t}: version/warmup/iters が腕間で不一致")
    torch = {}
    tavail = [r for r in RUNS if gate[("torch", r, "torch")] == "pass"]
    for task in ("train", "infer"):
        rows = load_jsonl(os.path.join(root, "fc", f"results-torch-{task}.jsonl"), optional=not tavail)
        if len(rows) != len(tavail):
            raise AggError(f"results-torch-{task}.jsonl: 行数 {len(rows)} が利用可能 run 数 {len(tavail)} と不一致")
        torch[task] = {}
        for r, o in zip(tavail, rows):
            if o.get("framework") != "pytorch" or o.get("task") != task or o.get("device") != "cpu" or o.get("size") != 64:
                raise AggError(f"results-torch-{task}.jsonl: framework/task/device/size が不一致: {o}")
            torch[task][r] = pos_float(o.get("median_s"), f"torch {task} median_s")
    return data, torch


def check_prereq(root, arms):
    """prereq/<arm>.log: 全ブロック exit=0 と probe 行のゲート値が期待どおり。成立しなければ理由を返す（None=成立）。"""
    bad = {}
    for a in arms:
        text = read_text(os.path.join(root, "prereq", f"{a}.log"))
        exits = re.findall(r"^exit=(\d+)$", text, re.M)
        g1, g2, aff = ARM_GATES[a]
        reasons = []
        if len(exits) < 2 or any(e != "0" for e in exits):
            reasons.append("report テストの exit != 0")
        l1 = [l for l in text.splitlines() if l.startswith("probe=small_m_shared_a ")]
        l2 = [l for l in text.splitlines() if l.startswith("probe=resident_w_prepack ")]
        if not l1 or any(f"gate={str(g1).lower()} " not in l for l in l1):
            reasons.append("G1 probe のゲート値が期待と不一致")
        elif g1 and not any(" eligible=true " in l for l in l1):
            reasons.append("G1 適格行なし")
        if len(l2) != 4 or any(f"gate={str(g2).lower()} " not in l for l in l2):
            reasons.append("G2 probe のゲート値が期待と不一致")
        elif g2 and any(not re.search(r" builds=1 hits=[1-9]", l) for l in l2):
            reasons.append("G2 の builds/hits が期待と不一致")
        elif not g2 and any(" cache_len=0 " not in l for l in l2):
            reasons.append("G2 OFF なのに cache_len != 0")
        if "gb10_affinity_report enabled=" in text:
            if f"gb10_affinity_report enabled={str(aff).lower()} " not in text:
                reasons.append("affinity enabled が期待と不一致")
        elif "-aff" in a or "### gb10_affinity_report_dump" in text:
            reasons.append("affinity report 行が無い")
        if reasons:
            bad[a] = "・".join(reasons)
    return bad


def check_bitmatch(root):
    text = read_text(os.path.join(root, "bitmatch.log"))
    m = re.search(r"test result: ok\. (\d+) passed; 0 failed", text)
    if not m or int(m.group(1)) < 1 or not re.search(r"^exit=0$", text, re.M):
        return None
    return int(m.group(1))


def med(vals):
    return statistics.median(vals)


def ratio_stats(arm_cell, off_cell):
    """cell: {run: (median, checksum)}。-> (arm 中央値, off 中央値, 比, 勝ち run 数, 比較 run 数)。"""
    am = med([v[0] for v in arm_cell.values()])
    om = med([v[0] for v in off_cell.values()])
    both = sorted(set(arm_cell) & set(off_cell))
    wins = sum(1 for r in both if arm_cell[r][0] <= off_cell[r][0])
    return am, om, am / om, wins, len(both)


def machine_report(root, machine):
    arms = MACHINE_ARMS[machine]
    out = [f"## {machine}", ""]
    if not os.path.isdir(root):
        out += ["**未計測**（機体ディレクトリが存在しない。数値は埋めない）。腕ごとの判定:", ""]
        for a in arms[1:]:
            out.append(f"- {a}: undetermined（未計測／unavailable）")
        out.append("")
        return "\n".join(out), True, {a: "undetermined" for a in arms[1:]}
    if os.path.exists(os.path.join(root, "FAILED")):
        raise AggError(f"{root}/FAILED が存在する（途中停止した系列。orchestrate の失敗理由: {read_text(os.path.join(root, 'FAILED')).strip()}）")
    for n in (".rev-stamp", "tree_verify.txt", "patch_sha256.txt", "DONE", "env_info.txt"):
        need(os.path.join(root, n))
    rev = read_text(os.path.join(root, ".rev-stamp")).split("\n")
    gate = parse_gate(root, arms)
    jg, threads = parse_jobgrid(root, arms, gate)
    fc, torch = parse_fc(root, arms, gate)
    prereq_bad = check_prereq(root, arms)
    bit_n = check_bitmatch(root)

    # checksum 完全一致（単体: 形状ごとに全腕・全 run で同一。fc: (task, mode) ごとに全腕・全 run で同一）
    cs_bad = []
    for s in jg[arms[0]]:
        vals = {v[1] for a in arms for v in jg[a][s].values()}
        if len(vals) > 1:
            cs_bad.append(f"jobgrid {s}")
    for c in FC_CELLS:
        vals = {v[1] for a in arms for v in fc[a][c].values()}
        if len(vals) > 1:
            cs_bad.append(f"fc {c[0]}/{c[1]}")

    def official(a):
        return all(gate[("ab", r, a)] == "pass" for r in RUNS)
    series_official = all(official(a) for a in arms)

    out += [f"- 系列: {'正式（全腕 5/5 round が負荷ゲート pass）' if series_official else '**参考系列**（負荷ゲート unavailable の round あり）'}",
            f"- 計測コミット: `.rev-stamp` 1 行目 `{rev[0]}`・ツリー指紋 `{rev[1] if len(rev) > 1 else '?'}`",
            f"- bit 一致テスト: {'pass（' + str(bit_n) + ' 件）' if bit_n else '**不成立**'}",
            f"- checksum 完全一致: {'一致' if not cs_bad else '**不一致** ' + ', '.join(cs_bad)}",
            "- rayon_threads（腕内一定を検査済み）: " + ", ".join(f"{a}={threads.get(a, 'NA')}" for a in arms),
            "- 機構発火の前提: " + ("全腕成立" if not prereq_bad else "; ".join(f"{a}={m}" for a, m in prereq_bad.items())), ""]

    shape_cells = list(SHAPES)
    all_cells = [("jg", s) for s in shape_cells] + [("fc", c) for c in FC_CELLS]

    def cell_of(a, kind, key):
        return jg[a][key] if kind == "jg" else fc[a][key]

    def label(kind, key):
        return key if kind == "jg" else f"{key[0]}/{key[1]}"

    verdicts, notes = {}, {}
    off = "off"
    for a in arms:
        if a == off:
            continue
        reasons = []
        if cs_bad:
            verdicts[a] = "FAIL"
            notes[a] = "checksum 不一致"
            continue
        if not official(a) or not official(off):
            reasons.append("負荷ゲート不成立 round あり")
        if a in prereq_bad or off in prereq_bad:
            reasons.append("機構発火の前提不成立")
        if not bit_n:
            reasons.append("bit 一致テスト不成立")
        if reasons:
            verdicts[a] = "undetermined"
            notes[a] = "・".join(reasons)
            continue
        over = []
        for kind, key in all_cells:
            _, _, rt, _, _ = ratio_stats(cell_of(a, kind, key), cell_of(off, kind, key))
            if rt > THRESHOLD:
                over.append(f"{label(kind, key)}={rt:.3f}")
        verdicts[a] = "REJECT" if over else "ADOPT"
        notes[a] = ("比が 1.00 超: " + ", ".join(over)) if over else "全セルの中央値比が 1.00 以下"

    # 表 1: セル × 腕 の 5 回中央値（µs）と比（腕/off）・勝ち run 数
    out.append("### 中央値（µs。5 round 中央値）と比（腕/off。1.00 以下で非後退）・勝ち run 数（腕<=off の round）")
    out.append("")
    out.append("| セル | " + " | ".join(arms) + " |")
    out.append("|---|" + "---|" * len(arms))
    for kind, key in all_cells + [("jg", REF_SHAPE[0])]:
        row = [label(kind, key) + ("（参照・判定外）" if key == REF_SHAPE[0] else "")]
        for a in arms:
            if not cell_of(a, kind, key) or not cell_of(off, kind, key):
                row.append("unavailable")
                continue
            am, om, rt, w, n = ratio_stats(cell_of(a, kind, key), cell_of(off, kind, key))
            row.append(f"{am * 1e6:.1f}" if a == off else f"{am * 1e6:.1f}（{rt:.3f}・{w}/{n}）")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # 表 2: 判定
    out += ["### 判定（腕ごと。RULE.txt の 4 値）", "", "| 腕 | 判定 | 根拠 |", "|---|---|---|"]
    for a in arms[1:]:
        out.append(f"| {a} | **{verdicts[a]}** | {notes[a]} |")
    out.append("")

    # 参考: merge+prepack / merge、affinity 相互作用
    out += ["### 参考（判定に使わない）", "",
            "#### merge+prepack／merge 比（#3200 の上乗せ分。1.00 未満で prepack が追加で速い）", ""]
    out += ["| セル | mergeprepack/merge |", "|---|---|"]
    for kind, key in all_cells:
        a, b = cell_of("mergeprepack", kind, key), cell_of("merge", kind, key)
        out.append(f"| {label(kind, key)} | " + (f"{ratio_stats(a, b)[2]:.3f}" if a and b else "unavailable") + " |")
    out.append("")
    if machine == "gb10":
        out += ["#### affinity 相互作用（X-aff／X 比。affinity 単独の再判定ではない）", "",
                "| セル | off-aff/off | merge-aff/merge | mergeprepack-aff/mergeprepack |", "|---|---|---|---|"]
        for kind, key in all_cells:
            row = [label(kind, key)]
            for x in ("off", "merge", "mergeprepack"):
                a, b = cell_of(x + "-aff", kind, key), cell_of(x, kind, key)
                row.append(f"{ratio_stats(a, b)[2]:.3f}" if a and b else "unavailable")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    out += ["#### PyTorch 比（train／infer 4 セル。PyTorch 中央値 ÷ fandhe 中央値。1.00 以上で fandhe が同等以上。"
            "PyTorch は fresh のみ・同一 round で計測。単体 6 形状は未計測）", "",
            "| セル | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for task, mode in FC_CELLS:
        tv = torch.get(task, {})
        row = [f"{task}/{mode}"]
        for a in arms:
            c = fc[a][(task, mode)]
            row.append(f"{med(tv.values()) / med([v[0] for v in c.values()]):.3f}" if tv and c else "unavailable")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return "\n".join(out), not cs_bad, verdicts


# ---------------------------------------------------------------- self-test
def _write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(s)


def _sub(p, old, new, *count):
    with open(p) as f:
        s = f.read()
    with open(p, "w") as f:
        f.write(s.replace(old, new, *count))


def _fixture(root, machine, mutate=None, scale=None):
    arms = MACHINE_ARMS[machine]
    scale = scale or {}
    n = len(arms)
    gate_lines = []
    for r in RUNS:
        gate_lines.append(f"ORDER run={r} order={' '.join(arms[(r - 1) % n:] + arms[:(r - 1) % n])}")
        for a in arms:
            gate_lines.append(f"START series=ab run={r} arm={a} gate=pass load1=0.1 waited_s=0 at=t")
        gate_lines.append(f"START series=torch run={r} arm=torch gate=pass load1=0.1 waited_s=0 at=t")
    _write(f"{root}/load_gate.log", "\n".join(gate_lines) + "\n")
    for n_ in (".rev-stamp",):
        _write(f"{root}/{n_}", "deadbeef\nfeed\n")
    for n_ in ("tree_verify.txt", "patch_sha256.txt", "DONE", "env_info.txt"):
        _write(f"{root}/{n_}", "x\n")
    _write(f"{root}/bitmatch.log", "test result: ok. 12 passed; 0 failed; 0 ignored\nexit=0\n")
    table = dict(SHAPES, **{REF_SHAPE[0]: REF_SHAPE[1]})
    for a in arms:
        g1, g2, aff = ARM_GATES[a]
        txt = "### G1\n" + "".join(
            f"probe=small_m_shared_a gate={str(g1).lower()} eligible={str(g1).lower()} shape={s} arm=native row_bands=2 a_shared_bands=2\n" for s in SHAPES)
        txt += "exit=0\n### G2\n" + "".join(
            f"probe=resident_w_prepack gate={str(g2).lower()} entry=rhs shape=1x1 cache_len={0 if not g2 else 1} builds={1 if g2 else 0} hits={1 if g2 else 0}\n" for _ in range(4))
        txt += "exit=0\n"
        if machine == "gb10":
            txt += f"### aff\ngb10_affinity_report enabled={str(aff).lower()} env_override=false\nexit=0\n"
        _write(f"{root}/prereq/{a}.log", txt)
        sc = scale.get(a, 1.0)
        for r in RUNS:
            lines = ["# header"]
            for s, (t, m, n_, k) in table.items():
                lines.append(f"arm={a} series=threads shape={s} trans={t} m={m} n={n_} k={k} rayon_threads=10 "
                             f"median_secs={0.001 * sc:.9f} gflops=1.0 checksum=0x{'ab' * 8}")
            _write(f"{root}/ab/run{r}-{a}.log", "\n".join(lines) + "\n")
        for task in ("train", "infer"):
            rows = []
            for r in RUNS:
                for mode in ("fresh", "reuse"):
                    rows.append(json.dumps(dict(framework="fandhe-ai", version="0.11.0", task=task, device="cpu", size=64,
                                                median_s=0.001 * sc, checksum=0.5, warmup=20, iters=80, mode=mode)))
            _write(f"{root}/fc/results-{a}-{task}.jsonl", "\n".join(rows) + "\n")
    for task in ("train", "infer"):
        rows = [json.dumps(dict(framework="pytorch", task=task, device="cpu", size=64, median_s=0.002, checksum=1.0)) for _ in RUNS]
        _write(f"{root}/fc/results-torch-{task}.jsonl", "\n".join(rows) + "\n")
    if mutate:
        mutate(root)


def self_test():
    def run(machine, mutate=None, scale=None):
        with tempfile.TemporaryDirectory() as t:
            _fixture(t, machine, mutate, scale)
            return machine_report(t, machine)

    def expect_raise(label, machine, mutate):
        try:
            run(machine, mutate)
        except AggError:
            return
        raise SystemExit(f"self-test 失敗: {label} が例外にならない")

    md, ok, v = run("gb10")
    assert ok and all(x == "ADOPT" for x in v.values()), f"正常 fixture が全 ADOPT でない: {v}"
    _, ok, v = run("macmini", scale={"merge": 1.2})
    assert v["merge"] == "REJECT" and v["mergeprepack"] == "ADOPT", v
    _, ok, v = run("macmini", scale={"merge": 0.5})
    assert v["merge"] == "ADOPT"

    def bad_cs(r):
        p = f"{r}/ab/run1-merge.log"
        _sub(p, "0x" + "ab" * 8, "0x" + "cd" * 8, 1)
    _, ok, v = run("macmini", bad_cs)
    assert not ok and v["merge"] == "FAIL" and v["mergeprepack"] == "FAIL", v

    def bad_fc_cs(r):
        p = f"{r}/fc/results-merge-train.jsonl"
        _sub(p, '"checksum": 0.5', '"checksum": 0.6', 1)
    _, ok, v = run("macmini", bad_fc_cs)
    assert not ok and v["merge"] == "FAIL"

    def bad_prereq(r):
        p = f"{r}/prereq/mergeprepack.log"
        _sub(p, "builds=1 hits=1", "builds=2 hits=1")
    _, ok, v = run("macmini", bad_prereq)
    assert v["mergeprepack"] == "undetermined" and v["merge"] == "ADOPT", v

    def unavail(r):
        p = f"{r}/load_gate.log"
        s = open(p).read().replace("START series=ab run=3 arm=merge gate=pass", "START series=ab run=3 arm=merge gate=unavailable")
        open(p, "w").write(s)
        os.remove(f"{r}/ab/run3-merge.log")
        for task in ("train", "infer"):
            q = f"{r}/fc/results-merge-{task}.jsonl"
            lines = open(q).read().splitlines()
            del lines[4:6]  # run3 の fresh/reuse（行 index 4,5）
            open(q, "w").write("\n".join(lines) + "\n")
    _, ok, v = run("macmini", unavail)
    assert v["merge"] == "undetermined" and v["mergeprepack"] == "ADOPT", v

    def all_unavail(r):
        p = f"{r}/load_gate.log"
        s = open(p).read()
        for n in RUNS:
            s = s.replace(f"START series=ab run={n} arm=merge gate=pass", f"START series=ab run={n} arm=merge gate=unavailable")
            os.remove(f"{r}/ab/run{n}-merge.log")
        open(p, "w").write(s)
        for task in ("train", "infer"):
            os.remove(f"{r}/fc/results-merge-{task}.jsonl")
    _, ok, v = run("macmini", all_unavail)
    assert v["merge"] == "undetermined", v

    def unavail_with_log(r):
        p = f"{r}/load_gate.log"
        s = open(p).read().replace("START series=ab run=3 arm=merge gate=pass", "START series=ab run=3 arm=merge gate=unavailable")
        open(p, "w").write(s)
    expect_raise("unavailable なのにログが存在", "macmini", unavail_with_log)
    expect_raise("ログ欠損", "macmini", lambda r: os.remove(f"{r}/ab/run2-off.log"))
    expect_raise("DONE 欠損", "macmini", lambda r: os.remove(f"{r}/DONE"))
    expect_raise("FAILED 存在", "macmini", lambda r: _write(f"{r}/FAILED", "x\n"))
    expect_raise("bitmatch 欠損", "macmini", lambda r: os.remove(f"{r}/bitmatch.log"))

    def dup_start(r):
        p = f"{r}/load_gate.log"
        open(p, "a").write("START series=ab run=1 arm=off gate=pass load1=0.1 waited_s=0 at=t\n")
    expect_raise("START 重複", "macmini", dup_start)

    def six_rows(r):
        p = f"{r}/fc/results-off-train.jsonl"
        lines = open(p).read().splitlines()
        open(p, "w").write("\n".join(lines + lines[:2]) + "\n")
    expect_raise("fc の run 過剰（6 件）", "macmini", six_rows)

    def short_rows(r):
        p = f"{r}/fc/results-off-infer.jsonl"
        lines = open(p).read().splitlines()
        open(p, "w").write("\n".join(lines[:-2]) + "\n")
    expect_raise("fc の run 不足（4 件）", "macmini", short_rows)

    def nan_med(r):
        p = f"{r}/ab/run1-off.log"
        _sub(p, "median_secs=0.001000000", "median_secs=nan", 1)
    expect_raise("数値でない median", "macmini", nan_med)

    def bad_threads(r):
        p = f"{r}/ab/run2-off.log"
        _sub(p, "rayon_threads=10", "rayon_threads=20", 1)
    expect_raise("rayon_threads の腕内食い違い", "macmini", bad_threads)

    def bad_dims(r):
        p = f"{r}/ab/run2-merge.log"
        _sub(p, " m=64 ", " m=65 ", 1)
    expect_raise("寸法不一致", "macmini", bad_dims)

    def bad_order(r):
        p = f"{r}/load_gate.log"
        _sub(p, "ORDER run=2 order=merge mergeprepack off", "ORDER run=2 order=off merge mergeprepack")
    expect_raise("ORDER がローテーションでない", "macmini", bad_order)

    def torch_short(r):
        p = f"{r}/fc/results-torch-train.jsonl"
        lines = open(p).read().splitlines()
        open(p, "w").write("\n".join(lines[:-1]) + "\n")
    expect_raise("torch の行不足", "macmini", torch_short)

    with tempfile.TemporaryDirectory() as t:
        md, ok, v = machine_report(os.path.join(t, "none"), "macmini")
        assert "未計測" in md and all(x == "undetermined" for x in v.values())
    print("self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    here = os.path.dirname(os.path.abspath(__file__))
    root, machines = here, ["macmini", "gb10"]
    if "--dir" in argv:
        root = argv[argv.index("--dir") + 1]
    if "--machines" in argv:
        machines = argv[argv.index("--machines") + 1].split(",")
    for m in machines:
        if m not in MACHINE_ARMS:
            raise AggError(f"未知の機体: {m}")
    parts = ["# #3201 集計（aggregate.py の出力。手で直さない）", ""]
    ok_all = True
    for m in machines:
        md, ok, _ = machine_report(os.path.join(root, m), m)
        parts.append(md)
        ok_all = ok_all and ok
    sys.stdout.write("\n".join(parts) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except AggError as e:
        sys.stderr.write(f"集計失敗（fail-closed）: {e}\n")
        sys.exit(2)
