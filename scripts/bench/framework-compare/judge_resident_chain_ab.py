#!/usr/bin/env python3
"""イシュー #3257: 常駐チェーン（train_step_resident／predict_resident_input）A/B の判定ツール（fail-closed）。

`docs/perf/logs/resident-chain-ab-3257/RULE.txt` の [hard 条件]・[比]・[判定] を機械的に適用する。
`run_ab_resident_chain.sh` が機体ごとに出す生ログ群を読み、セル（train／infer × device）ごとに
IMPROVE / REGRESS / NEUTRAL / INVALID / REJECT を返す。手書きの判定表を作らない単一ソース。
寛容パース禁止（欠損・不正・件数不足は例外にせず INVALID／REJECT へ明示的に落とす）。

腕: before = 現行経路（--mode reuse）、after = 常駐チェーン（--mode chain）、candle／torch = 相手（参考・判定外）。
ratio_ab = median(after)/median(before)、ratio_r = after_r/before_r（JSONL の行順 = round 順）。

- hard（判定より先。正しさの失敗を隠さない）: checksum が全 run・両腕で完全一致。train は
  PARAM_DIGEST も全 run・両腕で同値（欠落は不一致扱い）。違反は REJECT。
- IMPROVE: ratio_ab <= 0.95 かつ 5/5 round で ratio_r < 1.00。
- REGRESS: ratio_ab > 1.00 または 5/5 round で ratio_r > 1.00。
- NEUTRAL: 上のどちらでもない（RULE の「undetermined」）。
- INVALID: run 数不足・env_info／ゲート／RULE sha 不一致など、計測が規則を満たさない。

終了コード: 0 = 全セルが IMPROVE/REGRESS/NEUTRAL / 2 = INVALID あり（REJECT なし）/ 3 = REJECT あり。
参考系列なので、判定で既定値やゲートを切り替えない。相手比は記録のみ（判定外）。
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys

ROUNDS = 5
IMPROVE_RATIO = 0.95
REGRESS_RATIO = 1.00
TASKS = ("train", "infer")
PROFILES = {
    "macmini": {"devices": ("cpu", "metal"), "gate": 3.0},
    "gb10": {"devices": ("cpu", "cuda"), "gate": 1.0},
}
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
DIGEST_RE = re.compile(
    r"^PARAM_DIGEST task=train mode=(reuse|chain) device=([a-z0-9]+) "
    r"n_tensors=(\d+) n_elems=(\d+) fnv1a64=(0x[0-9a-f]{16})$"
)
ARM_MODE = {"before": "reuse", "after": "chain"}
OPPONENTS = ("candle", "torch")


class Bad(Exception):
    """データ不正。セルを INVALID にする。"""


def _read_lines(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [ln.rstrip("\n") for ln in fh]
    except OSError as e:
        raise Bad(f"{path}: 開けない ({e})")


def _num(o, k, where):
    v = o.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Bad(f"{where}: {k} が有限数でない ({v!r})")
    return v


def load_rows(path, task, dev, mode=None, fandhe=True):
    """JSONL から task・device が一致する行を出現順に返す。fandhe=True は fandhe-ai 行で mode 一致必須。"""
    rows = []
    for n, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        try:
            o = json.loads(line)
        except ValueError:
            raise Bad(f"{where}: JSON 不正")
        if not isinstance(o, dict):
            raise Bad(f"{where}: object でない")
        if o.get("task") != task:
            raise Bad(f"{where}: task が {task} でない ({o.get('task')!r})")
        if o.get("device") != dev:
            raise Bad(f"{where}: device が {dev} でない ({o.get('device')!r})")
        if isinstance(o.get("size"), bool) or o.get("size") != 64:
            raise Bad(f"{where}: size が 64 でない ({o.get('size')!r})")
        if fandhe:
            if o.get("framework") != "fandhe-ai":
                raise Bad(f"{where}: framework != fandhe-ai")
            if o.get("mode") != mode:
                raise Bad(f"{where}: mode が {mode} でない ({o.get('mode')!r})")
        elif o.get("framework") == "fandhe-ai":
            raise Bad(f"{where}: 相手ファイルに fandhe-ai 行")
        if _num(o, "median_s", where) <= 0:
            raise Bad(f"{where}: median_s <= 0")
        if fandhe:
            _num(o, "checksum", where)
        rows.append(o)
    return rows


def load_digest(path, mode, dev):
    hits = [ln for ln in _read_lines(path) if ln.startswith("PARAM_DIGEST")]
    if len(hits) != 1:
        raise Bad(f"{path}: PARAM_DIGEST 行が {len(hits)} 件（1 件必須）")
    m = DIGEST_RE.match(hits[0])
    if not m:
        raise Bad(f"{path}: PARAM_DIGEST 行の形式不正")
    g = m.groups()
    if g[0] != mode or g[1] != dev:
        raise Bad(f"{path}: digest の mode/device が要求 ({mode}/{dev}) と不一致 ({g[0]}/{g[1]})")
    return g[2:]


def parse_kv(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def patch_sha(rule_path):
    p = os.path.join(os.path.dirname(os.path.abspath(rule_path)), "bench-resident-chain.patch")
    try:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError as e:
        raise Bad(f"計測パッチを開けない ({e})")


def check_env(profile, env_path, rule_sha, psha):
    kv = {}
    for ln in _read_lines(env_path):
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k] = v
    if kv.get("profile") != profile:
        raise Bad(f"env_info の profile が {profile} でない ({kv.get('profile')!r})")
    if kv.get("rule_sha256") != rule_sha:
        raise Bad("env_info の rule_sha256 が現 RULE.txt と一致しない")
    if kv.get("patch_sha256") != psha:
        raise Bad("env_info の patch_sha256 が計測パッチの sha256 と一致しない")
    if kv.get("precheck") != "assert":
        raise Bad("env_info の precheck が assert でない")
    if profile == "macmini" and kv.get("cpu_brand") != "Apple M4":
        raise Bad(f"Mac mini の cpu_brand が Apple M4 でない ({kv.get('cpu_brand')!r})")
    if not re.match(r"^[0-9a-f]{40}$", kv.get("rev_stamp_head_runner", "")):
        raise Bad("env_info の rev_stamp_head_runner が欠落または不正")
    if not re.match(r"^[0-9a-f]{64}$", kv.get("bin_sha256", "")):
        raise Bad("env_info の bin_sha256 が欠落または不正")


def check_gate(profile, dev, path):
    lines = [ln.strip() for ln in _read_lines(path) if ln.strip()]
    if not lines:
        raise Bad("負荷ゲートログが空")
    head = parse_kv(lines[0])
    want = PROFILES[profile]["gate"]
    if head.get("series") != "official" or head.get("precheck") != "assert":
        raise Bad("負荷ゲートログが official/assert でない")
    for key in ("threshold", "rule_threshold"):
        try:
            got = float(head.get(key, ""))
        except ValueError:
            got = None
        if got is None or got != want:
            raise Bad(f"{key} が機体規則値 {want} と不一致 ({head.get(key)!r})")
    rounds = [ln for ln in lines[1:] if ln.startswith("round")]
    if [ln.split()[0] for ln in rounds] != [f"round{i}" for i in range(1, ROUNDS + 1)]:
        raise Bad("round 行が round1〜5 と一致しない")
    for ln in rounds:
        r = parse_kv(ln)
        if r.get("dev") != dev or r.get("gate") != "pass" or r.get("gate_ok") != "1":
            raise Bad(f"負荷ゲート不通過または dev 不一致: {ln.split()[0]}")


def opponent_ratio(d, label, dev, task, fandhe_rows):
    """参考: 同一 round 内の有効な相手最速 ÷ fandhe の 5 round 中央値。読めなければ n/a（判定外）。"""
    per_round = []
    try:
        series = []
        for opp in OPPONENTS:
            p = f"{d}/results-{opp}-{label}-{dev}-{task}.jsonl"
            if not os.path.exists(p):
                continue
            rows = load_rows(p, task, dev, fandhe=False)
            if len(rows) == ROUNDS:
                series.append(rows)
        if not series:
            return "n/a"
        for r in range(ROUNDS):
            best = min(s[r]["median_s"] for s in series)
            per_round.append(best / fandhe_rows[r]["median_s"])
    except Bad as e:
        return f"n/a (相手不正: {str(e)[:50]})".replace("|", "/")
    return f"{statistics.median(per_round):.2f}x"


def judge_cell(profile, dev, task, d, label, rule_sha, psha):
    base = f"{d}/results-%s-{label}-{dev}-{task}.jsonl"
    res = {"machine": profile, "dev": dev, "task": task, "verdict": "INVALID", "mb": None,
           "ma": None, "ratio": None, "per": [], "reason": "", "opp_before": "n/a", "opp_after": "n/a"}
    try:
        before = load_rows(base % "before", task, dev, ARM_MODE["before"])
        after = load_rows(base % "after", task, dev, ARM_MODE["after"])
        if len(before) != ROUNDS or len(after) != ROUNDS:
            raise Bad(f"run 数が 5 でない (before={len(before)} after={len(after)})")
        res["mb"] = statistics.median(r["median_s"] for r in before)
        res["ma"] = statistics.median(r["median_s"] for r in after)
        res["ratio"] = res["ma"] / res["mb"]
        res["per"] = [a["median_s"] / b["median_s"] for b, a in zip(before, after)]
        if len({r["checksum"] for r in before + after}) != 1:
            res["verdict"], res["reason"] = "REJECT", "正しさ: checksum が不一致"
            return res
        if task == "train":
            digests, err = set(), None
            for arm in ("before", "after"):
                for r in range(1, ROUNDS + 1):
                    try:
                        digests.add(load_digest(f"{d}/digest-{arm}-{label}-{dev}-train-r{r}.log",
                                                ARM_MODE[arm], dev))
                    except Bad as e:
                        err = str(e)
            if err is not None:
                res["verdict"], res["reason"] = "REJECT", f"正しさ: digest 欠落または不正（不一致扱い）: {err}"
                return res
            if len(digests) != 1:
                res["verdict"], res["reason"] = "REJECT", "正しさ: 更新後パラメータの digest が不一致"
                return res
        check_env(profile, f"{d}/env_info-{label}.txt", rule_sha, psha)
        check_gate(profile, dev, f"{d}/load_gate-{label}-{dev}.log")
    except Bad as e:
        res["verdict"], res["reason"] = "INVALID", str(e)
        return res
    res["opp_before"] = opponent_ratio(d, label, dev, task, before)
    res["opp_after"] = opponent_ratio(d, label, dev, task, after)
    ratio, per = res["ratio"], res["per"]
    if ratio > REGRESS_RATIO or all(p > 1.0 for p in per):
        res["verdict"], res["reason"] = "REGRESS", "後退（ratio>1.00 または 5/5 round で ratio_r>1.00）"
    elif ratio <= IMPROVE_RATIO and all(p < 1.0 for p in per):
        res["verdict"], res["reason"] = "IMPROVE", "改善（ratio<=0.95 かつ 5/5 round で ratio_r<1.00）"
    else:
        res["verdict"], res["reason"] = "NEUTRAL", "改善・後退どちらの条件も満たさない（RULE の undetermined）"
    return res


def render(results, rule_sha):
    out = [f"# #3257 判定（rule_sha256={rule_sha[:16]}…）", "",
           "| 機体 | task | device | before 中央値 ms | after 中央値 ms | ratio_ab | ratio_r（5 round） | 判定 | 理由 | 相手最速÷before | 相手最速÷after |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        ms = lambda v: "n/a" if v is None else f"{v * 1e3:.4f}"
        per = ", ".join(f"{p:.3f}" for p in r["per"]) or "n/a"
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
        reason = r["reason"].replace("|", "/").replace("\n", " ")
        out.append(f"| {r['machine']} | {r['task']} | {r['dev']} | {ms(r['mb'])} | {ms(r['ma'])} | {ratio} | {per} "
                   f"| {r['verdict']} | {reason} | {r['opp_before']} | {r['opp_after']} |")
    return "\n".join(out)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", required=True)
    ap.add_argument("--machine", action="append", default=[], metavar="PROFILE=DIR:LABEL")
    ap.add_argument("--require-both", action="store_true")
    a = ap.parse_args(argv)
    try:
        with open(a.rule, "rb") as fh:
            rule_sha = hashlib.sha256(fh.read()).hexdigest()
        psha = patch_sha(a.rule)
    except (OSError, Bad) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    machines = {}
    for spec in a.machine:
        m = re.match(r"^(macmini|gb10)=([^:]+):([A-Za-z0-9._-]+)$", spec)
        if not m:
            print(f"error: --machine の形式不正: {spec}", file=sys.stderr)
            return 2
        machines[m.group(1)] = (m.group(2), m.group(3))
    if not machines or (a.require_both and set(machines) != set(PROFILES)):
        print("error: --machine が無い、または --require-both で両機体が揃っていない", file=sys.stderr)
        return 2
    results = []
    for prof in ("macmini", "gb10"):
        if prof in machines:
            d, label = machines[prof]
            for dev in PROFILES[prof]["devices"]:
                for task in TASKS:
                    results.append(judge_cell(prof, dev, task, d, label, rule_sha, psha))
    print(render(results, rule_sha))
    if any(r["verdict"] == "REJECT" for r in results):
        return 3
    if any(r["verdict"] == "INVALID" for r in results):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
