#!/usr/bin/env python3
"""#4153 の窓の差（fandhe − candle）がカーネル側か窓側かを、測る前に固定した規則で機械集計する補助。

役割: aggregate_diag.py（F1／F2 の分類と確度）とは別に、RULE 項 4 の補助式を出す。
  ΔW = fandhe_window − opp_window、ΔK = fandhe_pure_kernel − opp_pure_kernel、kernel_share = ΔK ÷ ΔW
  trace_perturbation（xctrace 下と trace なしの窓の差）、cross_check_rel（xctrace と診断テストの kernel_gpu の差）
閾値は RULE の AUX（kernel_share_threshold・perturbation_max・cross_check_max）が唯一の正。実測値は含まない。

入力（<dir> 配下）: RULE.txt・round{1..N}/{gate.txt, values.tsv, aux.tsv}（to_values_4153.py の出力）。
出力: attrib.tsv（round 別と中央値）と標準出力の要約。厳格パース（欠損・重複・非有限は AttribError で停止）。
標準ライブラリのみ。`python3 -I` で実行すること。

使い方:
  python3 -I attrib_4153.py --dir <記録先> [--rule RULE.txt] [--out attrib.tsv]
  python3 -I attrib_4153.py --check-rule <RULE.txt>   # AUX のプレースホルダ 0 件・補助スクリプトの sha256 一致
  python3 -I attrib_4153.py --self-test
"""
import hashlib
import math
import os
import re
import statistics
import sys
import tempfile

CELL = "M-MTL-G4096"
HERE = os.path.dirname(os.path.abspath(__file__))
AUX_NUM = ["perturbation_max", "cross_check_max", "kernel_share_threshold"]
AUX_SHA = {"to_values_sha256": "to_values_4153.py", "attrib_sha256": "attrib_4153.py", "run_round_sha256": "run_round.sh"}


class AttribError(Exception):
    pass


def finite(s, what):
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise AttribError(f"{what}: not a number: {s!r}")
    if not math.isfinite(v):
        raise AttribError(f"{what}: not finite: {s!r}")
    return v


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        raise AttribError(f"cannot read {path}: {e}")


def parse_aux(text):
    lines = text.splitlines()
    try:
        a = next(i for i, l in enumerate(lines) if l.strip() == "## AUX")
    except StopIteration:
        raise AttribError("RULE: '## AUX' section required")
    kv = {}
    for l in lines[a + 1:]:
        s = l.strip()
        if s.startswith("## "):
            break
        if not s or s.startswith("（") or s.startswith("#"):
            continue
        if ":" not in s:
            raise AttribError(f"AUX: not key: value: {l!r}")
        k, v = s.split(":", 1)
        k = k.strip()
        if k in kv:
            raise AttribError(f"AUX: duplicate key {k!r}")
        kv[k] = v.strip()
    return kv


def parse_rounds(text):
    m = re.findall(r"^rounds:\s*(\d+)\s*$", text, re.M)
    if len(m) != 1 or int(m[0]) < 1:
        raise AttribError("PARAMS: rounds must be a single integer >= 1")
    return int(m[0])


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_rule(path, base=HERE):
    text = read(path)
    aux = parse_aux(text)
    for k, v in aux.items():
        if re.search(r"<[^<>]*>", v):
            raise AttribError(f"AUX: unreplaced placeholder in {k}")
    for k in AUX_NUM:
        if k not in aux:
            raise AttribError(f"AUX: missing {k}")
        finite(aux[k], k)
    for k, fn in AUX_SHA.items():
        if not re.fullmatch(r"[0-9a-f]{64}", aux.get(k, "")):
            raise AttribError(f"AUX: {k} must be 64 hex")
        if sha256_file(os.path.join(base, fn)) != aux[k]:
            raise AttribError(f"AUX: {k} mismatch ({fn} changed after RULE was fixed)")
    if re.search(r"<[^<>]+>", "\n".join(l for l in text.split("## PARAMS", 1)[-1].splitlines())):
        raise AttribError("PARAMS: unreplaced placeholder remains")
    return aux


def read_tsv(path, header, keycols):
    lines = read(path).splitlines()
    if not lines or lines[0].split("\t") != header:
        raise AttribError(f"{path}: bad header")
    out = {}
    for ln, l in enumerate(lines[1:], 2):
        if not l.strip():
            continue
        f = l.split("\t")
        if len(f) != len(header):
            raise AttribError(f"{path}:{ln}: need {len(header)} fields")
        key = tuple(f[:keycols])
        if key in out:
            raise AttribError(f"{path}:{ln}: duplicate {key}")
        out[key] = f
    return out


def round_metrics(rd, thr):
    vals = {k[1]: (finite(f[2], "value"), f[3]) for k, f in read_tsv(
        os.path.join(rd, "values.tsv"), ["cell", "quantity", "value", "clock"], 2).items() if k[0] == CELL}
    aux = {k[0]: finite(f[1], "aux") for k, f in read_tsv(os.path.join(rd, "aux.tsv"), ["quantity", "value"], 1).items()}
    for q in ("fandhe_window", "opp_window"):
        if q not in vals:
            raise AttribError(f"{rd}: missing {q}")
    for q in ("fandhe_kernel_gpu_ts", "opp_transfer_excl"):
        if q not in aux:
            raise AttribError(f"{rd}: missing aux {q}")
    fw, ow = vals["fandhe_window"][0], vals["opp_window"][0]
    m = {"delta_w": fw - ow}
    # trace 下の窓は xctrace=available のときだけ取る。unavailable（F2 のみ）では摂動を unavailable とし、値を作らない
    traced = [q in aux for q in ("opp_transfer_excl_traced", "fandhe_window_traced")]
    if any(traced) and not all(traced):
        raise AttribError(f"{rd}: aux traced values must be both present or both absent")
    if all(traced):
        m["perturbation_fandhe"] = abs(aux["fandhe_window_traced"] / fw - 1.0)
        m["perturbation_opp"] = abs(aux["opp_transfer_excl_traced"] / aux["opp_transfer_excl"] - 1.0)
        m["perturbation"] = max(m["perturbation_fandhe"], m["perturbation_opp"])
    if "fandhe_pure_kernel" in vals and "opp_pure_kernel" in vals:
        fk, ok = vals["fandhe_pure_kernel"][0], vals["opp_pure_kernel"][0]
        if vals["fandhe_pure_kernel"][1] != vals["opp_pure_kernel"][1]:
            raise AttribError(f"{rd}: pure_kernel clock tags differ")
        m["delta_k"] = fk - ok
        m["cross_check_rel"] = abs(fk - aux["fandhe_kernel_gpu_ts"]) / aux["fandhe_kernel_gpu_ts"]
        if m["delta_w"] <= 0:
            m["side"] = "判定不能(ΔW<=0)"
        else:
            m["kernel_share"] = m["delta_k"] / m["delta_w"]
            m["side"] = "カーネル側" if m["kernel_share"] >= thr else "窓側"
    else:
        m["side"] = "判定不能(純カーネル unavailable)"
    return m


def attribute(dirp, rule_name="RULE.txt", verify=True, base=HERE):
    """verify=True（既定）では集計の前に check_rule（プレースホルダ 0 件・補助スクリプトの sha256 一致）を必ず通す。"""
    if verify:
        check_rule(os.path.join(dirp, rule_name), base=base)
    text = read(os.path.join(dirp, rule_name))
    aux = parse_aux(text)
    for k in AUX_NUM:
        if k not in aux:
            raise AttribError(f"AUX: missing {k}")
    thr, pmax, cmax = (finite(aux[k], k) for k in ("kernel_share_threshold", "perturbation_max", "cross_check_max"))
    n = parse_rounds(text)
    rows, per = [], []
    for i in range(1, n + 1):
        rd = os.path.join(dirp, f"round{i}")
        g = read(os.path.join(rd, "gate.txt")).split()
        if not g or g[0] not in ("PASS", "TIMEOUT", "INVALID"):
            raise AttribError(f"{rd}/gate.txt: first word must be PASS|TIMEOUT|INVALID")
        if g[0] != "PASS":
            rows.append((i, "gate", g[0]))
            continue
        m = round_metrics(rd, thr)
        per.append(m)
        for k, v in m.items():
            rows.append((i, k, v if isinstance(v, str) else f"{v:.6f}"))
    if not per:
        raise AttribError("no PASS round")
    formal = len(per) == n
    sides = {m["side"] for m in per}
    med = lambda k: statistics.median([m[k] for m in per]) if all(k in m for m in per) else None
    low = []
    if not formal:
        low.append("PASS 以外の round を含む")
    if len(sides) != 1 or next(iter(sides)).startswith("判定不能"):
        low.append("round 間で帰属が割れる、または判定不能")
    if any(m.get("perturbation", 0) > pmax for m in per):
        low.append(f"trace_perturbation が {pmax} 超")
    if any(m.get("cross_check_rel", 0) > cmax for m in per):
        low.append(f"cross_check_rel が {cmax} 超")
    side = next(iter(sides)) if len(sides) == 1 else "割れ"
    conf = "高" if not low else "低"
    summary = [("median", "kernel_share", "" if med("kernel_share") is None else f"{med('kernel_share'):.6f}"),
               ("median", "delta_w", f"{med('delta_w'):.6f}"),
               ("median", "perturbation", "unavailable" if med("perturbation") is None else f"{med('perturbation'):.6f}"),
               ("summary", "side", side), ("summary", "confidence", conf),
               ("summary", "confidence_reasons", "; ".join(low) or "-")]
    return rows, summary


def run(dirp, rule_name, out):
    rows, summary = attribute(dirp, rule_name)  # 既定で check_rule（sha256 照合）を通す
    if os.path.exists(out):
        raise AttribError(f"{out} already exists (no overwrite)")
    body = "round\tmetric\tvalue\n" + "".join(f"{r}\t{k}\t{v}\n" for r, k, v in rows + summary)
    with open(out, "x", encoding="utf-8") as f:
        f.write(body)
    for _, k, v in summary:
        print(f"{k}: {v}")


# ---------------------------------------------------------------- self-test（合成データ。数値は架空）
def _mk(d, n, rows_by_round, aux_extra=""):
    open(os.path.join(d, "RULE.txt"), "w").write(
        "RULE fixed-at-utc 2000-01-01T00:00:00Z\n## AUX\nkernel_share_threshold: 0.5\nperturbation_max: 0.10\n"
        f"cross_check_max: 0.10\n{aux_extra}## PARAMS\nrounds: {n}\n")
    for i, (gate, vals, aux) in enumerate(rows_by_round, 1):
        rd = os.path.join(d, f"round{i}")
        os.makedirs(rd)
        open(os.path.join(rd, "gate.txt"), "w").write(gate + "\n")
        open(os.path.join(rd, "values.tsv"), "w").write(
            "cell\tquantity\tvalue\tclock\n" + "".join(f"{CELL}\t{q}\t{v}\t{c}\n" for q, v, c in vals))
        open(os.path.join(rd, "aux.tsv"), "w").write("quantity\tvalue\n" + "".join(f"{q}\t{v}\n" for q, v in aux.items()))


def _vals(fw=64.0, ow=56.0, fk=60.0, ok=54.0, pk=True):
    v = [("fandhe_window", fw, "-"), ("fandhe_fresh_window", fw, "-"), ("opp_window", ow, "-")]
    if pk:
        v += [("fandhe_pure_kernel", fk, "gpu_trace"), ("opp_pure_kernel", ok, "gpu_trace")]
    return v


def _aux(fw=64.0, osp=42.0, kg=60.0, tf=1.02, to=1.02):
    return {"fandhe_kernel_gpu_ts": kg, "opp_transfer_excl": osp, "opp_transfer_excl_traced": osp * to,
            "fandhe_window_traced": fw * tf}


def _expect(fn, label, needle):
    try:
        fn()
    except AttribError as e:
        assert needle in str(e), f"{label}: wrong error {e}"
        return
    raise AssertionError(f"{label}: expected AttribError")


def self_test():
    def sc(case):
        with tempfile.TemporaryDirectory() as t:
            n, rounds = case
            _mk(t, n, rounds)
            return attribute(t, verify=False)

    # 正常: ΔW=8, ΔK=6 → share 0.75 → カーネル側・確度高
    _, s = sc((2, [("PASS", _vals(), _aux())] * 2))
    sm = {k: v for _, k, v in s}
    assert sm["side"] == "カーネル側" and sm["confidence"] == "高" and sm["kernel_share"] == "0.750000", sm
    # 窓側: ΔK=2 → share 0.25
    _, s = sc((1, [("PASS", _vals(fk=56.0), _aux())]))
    assert {k: v for _, k, v in s}["side"] == "窓側"
    # ΔW<=0
    _, s = sc((1, [("PASS", _vals(fw=50.0, ow=56.0), _aux(fw=50.0))]))
    sm = {k: v for _, k, v in s}
    assert sm["side"].startswith("判定不能") and sm["confidence"] == "低"
    # 摂動超過 → 確度低
    _, s = sc((1, [("PASS", _vals(), _aux(tf=1.2))]))
    assert {k: v for _, k, v in s}["confidence"] == "低"
    # 照合超過 → 確度低
    _, s = sc((1, [("PASS", _vals(), _aux(kg=40.0))]))
    sm = {k: v for _, k, v in s}
    assert sm["confidence"] == "低" and "cross_check" in sm["confidence_reasons"]
    # フォールバック（純カーネルなし・trace 下の窓なし＝F2 のみ）
    a = {k: v for k, v in _aux().items() if "traced" not in k}
    _, s = sc((1, [("PASS", _vals(pk=False), a)]))
    sm = {k: v for _, k, v in s}
    assert "unavailable" in sm["side"] and sm["median_perturbation" if False else "perturbation"] == "unavailable", sm
    # round 間で割れる → 低
    _, s = sc((2, [("PASS", _vals(), _aux()), ("PASS", _vals(fk=56.0), _aux())]))
    sm = {k: v for _, k, v in s}
    assert sm["side"] == "割れ" and sm["confidence"] == "低"
    # TIMEOUT を含む → 低（partial）
    _, s = sc((2, [("PASS", _vals(), _aux()), ("TIMEOUT x", _vals(), _aux())]))
    assert {k: v for _, k, v in s}["confidence"] == "低"
    # 異常系
    _expect(lambda: sc((1, [("TIMEOUT", _vals(), _aux())])), "no PASS", "no PASS")
    _expect(lambda: sc((1, [("PASS", _vals() + [("opp_window", 1, "-")], _aux())])), "dup", "duplicate")
    _expect(lambda: sc((1, [("PASS", _vals(), {k: v for k, v in _aux().items() if k != "opp_transfer_excl"})])), "missing aux", "missing aux")
    _expect(lambda: sc((1, [("PASS", _vals(fw="nan"), _aux())])), "nan", "finite")
    _expect(lambda: sc((1, [("BOGUS", _vals(), _aux())])), "bad gate", "gate.txt")
    with tempfile.TemporaryDirectory() as t:  # clock 不一致
        v = _vals()
        v[-1] = ("opp_pure_kernel", 54.0, "wall")
        _mk(t, 1, [("PASS", v, _aux())])
        _expect(lambda: attribute(t, verify=False), "clock mismatch", "clock")
    with tempfile.TemporaryDirectory() as t:  # check_rule: プレースホルダ・sha 不一致
        for fn in AUX_SHA.values():
            open(os.path.join(t, fn), "w").write(fn)
        shas = {k: hashlib.sha256(fn.encode()).hexdigest() for k, fn in AUX_SHA.items()}
        good = ("RULE fixed-at-utc 2000-01-01T00:00:00Z\n## AUX\nkernel_share_threshold: 0.5\nperturbation_max: 0.10\n"
                "cross_check_max: 0.10\n" + "".join(f"{k}: {v}\n" for k, v in shas.items()) + "## PARAMS\nrounds: 5\n")
        rp = os.path.join(t, "RULE.txt")
        open(rp, "w").write(good)
        check_rule(rp, base=t)
        open(rp, "w").write(good.replace("perturbation_max: 0.10", "perturbation_max: <x>"))
        _expect(lambda: check_rule(rp, base=t), "placeholder", "placeholder")
        open(os.path.join(t, "run_round.sh"), "w").write("changed")
        open(rp, "w").write(good)
        _expect(lambda: check_rule(rp, base=t), "sha", "mismatch")
        # 通常の --dir 集計経路も sha256 を検証する（verify 既定 True）
        _expect(lambda: attribute(t, base=t), "attribute verifies sha", "mismatch")
    print("self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    if "--check-rule" in argv:
        check_rule(argv[argv.index("--check-rule") + 1])
        print("rule ok")
        return 0
    a = dict(zip(argv[0::2], argv[1::2]))
    if "--dir" not in a or not set(a) <= {"--dir", "--rule", "--out"}:
        print(__doc__)
        return 2
    run(a["--dir"], a.get("--rule", "RULE.txt"), a.get("--out", os.path.join(a["--dir"], "attrib.tsv")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (AttribError, AssertionError) as e:
        print(f"ATTRIB-ERROR: {e}", file=sys.stderr)
        sys.exit(1)
