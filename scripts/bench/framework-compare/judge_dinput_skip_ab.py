#!/usr/bin/env python3
"""イシュー #3247: d_input 省略（#3246）A/B の判定ツール（fail-closed）。

`docs/perf/logs/dinput-skip-ab-3247/RULE.txt` の [判定] を機械的に適用する。
`run_ab_dinput_skip_train.sh` が機体ごとに出す生ログ群（before/after の JSONL・digest ログ・
負荷ゲートログ・env_info）を読み、セルごとに ADOPT / REJECT / undetermined を返す。
手書きの判定表を作らないための単一ソース。寛容パース禁止（欠損・不正・件数不足は
例外ではなく undetermined/REJECT へ明示的に落とす。握りつぶさない）。

セル: device ごとに reuse = 処置セル、fresh = ガードセル（LinearAct 経路。コード不変の対照）。
各セルの before/after は各ちょうど 5 run。ratio = median(after)/median(before)、
ratio_r = after_r/before_r（JSONL の行順 = round 順）。

処置セル:
- 正しさ（hard）: loss checksum が全 run・両腕で完全一致、PARAM_DIGEST が全 run・両腕で同値。
  不一致は REJECT（正しさの失敗。ゲート判定より先に適用して隠さない）。
- ADOPT: ratio <= 0.95 かつ 5/5 round で ratio_r < 1.00。
- REJECT: ratio > 1.00、または 5/5 round で ratio_r > 1.00。
- それ以外・ゲート不成立・env_info 不一致・run 数不足は undetermined。
ガードセル: pass = ratio <= 1.03 かつ 5/5 round 後退でない。fail は併記（終了コードには影響しない）。

終了コード: 0 = 処置セルがすべて ADOPT / 2 = undetermined あり（REJECT なし）/ 3 = REJECT あり。
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys

ROUNDS = 5
ADOPT_RATIO = 0.95
REJECT_RATIO = 1.00
GUARD_RATIO = 1.03
PROFILES = {
    "macmini": {"devices": ("cpu", "metal"), "gate": 3.0},
    "gb10": {"devices": ("cpu", "cuda"), "gate": 1.0},
}
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
DIGEST_RE = re.compile(
    r"^PARAM_DIGEST task=train mode=(fresh|reuse) device=([a-z0-9]+) "
    r"n_tensors=(\d+) n_elems=(\d+) fnv1a64=(0x[0-9a-f]{16})$"
)


class Bad(Exception):
    """データ不正。セルを undetermined にする。"""


def _read_lines(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [ln.rstrip("\n") for ln in fh]
    except OSError as e:
        raise Bad(f"{path}: 開けない ({e})")


def load_rows(path, mode):
    """JSONL から task=train・指定 mode の fandhe-ai 行を出現順に返す。不正行は Bad。"""
    rows = []
    for n, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except ValueError:
            raise Bad(f"{path}:{n}: JSON 不正")
        if not isinstance(o, dict):
            raise Bad(f"{path}:{n}: object でない")
        if o.get("task") != "train" or o.get("mode") != mode:
            continue
        if o.get("framework") != "fandhe-ai":
            raise Bad(f"{path}:{n}: framework != fandhe-ai")
        for k in ("median_s", "checksum"):
            v = o.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise Bad(f"{path}:{n}: {k} が有限数でない ({v!r})")
        if o["median_s"] <= 0:
            raise Bad(f"{path}:{n}: median_s <= 0")
        rows.append(o)
    return rows


def load_digest(path):
    """digest ログから PARAM_DIGEST 行をちょうど 1 件取り出す。欠落・重複・形式不正は Bad。"""
    hits = [ln for ln in _read_lines(path) if ln.startswith("PARAM_DIGEST")]
    if len(hits) != 1:
        raise Bad(f"{path}: PARAM_DIGEST 行が {len(hits)} 件（1 件必須）")
    m = DIGEST_RE.match(hits[0])
    if not m:
        raise Bad(f"{path}: PARAM_DIGEST 行の形式不正")
    return m.groups()


def parse_kv(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def check_env(profile, env_path, rule_sha):
    kv = {}
    for ln in _read_lines(env_path):
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k] = v
    if kv.get("profile") != profile:
        raise Bad(f"env_info の profile が {profile} でない ({kv.get('profile')!r})")
    if kv.get("rule_sha256") != rule_sha:
        raise Bad("env_info の rule_sha256 が現 RULE.txt と一致しない")
    if kv.get("precheck") != "assert":
        raise Bad("env_info の precheck が assert でない")
    if profile == "macmini" and kv.get("cpu_brand") != "Apple M4":
        raise Bad(f"Mac mini の cpu_brand が Apple M4 でない ({kv.get('cpu_brand')!r})")
    for k in ("rev_stamp_head_before", "rev_stamp_head_after", "rev_stamp_head_runner", "patch_sha256"):
        if not re.match(r"^[0-9a-f]{40,64}$", kv.get(k, "")):
            raise Bad(f"env_info の {k} が欠落または不正")
    if kv["rev_stamp_head_before"] == kv["rev_stamp_head_after"]:
        raise Bad("before と after の rev_stamp_head が同一（腕が区別できない）")


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


def cell_stats(before, after):
    mb = statistics.median(r["median_s"] for r in before)
    ma = statistics.median(r["median_s"] for r in after)
    per = [a["median_s"] / b["median_s"] for b, a in zip(before, after)]
    return mb, ma, ma / mb, per


def judge_cell(machine, profile, dev, mode, d, label, rule_sha):
    """1 セルの判定。戻り値 dict(verdict, reason, ...)。"""
    base = f"{d}/results-%s-{label}-{dev}-train.jsonl"
    res = {"machine": machine, "dev": dev, "mode": mode, "verdict": "undetermined",
           "mb": None, "ma": None, "ratio": None, "per": [], "reason": ""}
    try:
        before = load_rows(base % "before", mode)
        after = load_rows(base % "after", mode)
        if len(before) != ROUNDS or len(after) != ROUNDS:
            raise Bad(f"run 数が 5 でない (before={len(before)} after={len(after)})")
        res["mb"], res["ma"], res["ratio"], res["per"] = cell_stats(before, after)
        # hard 条件（ゲートより先。正しさの失敗を隠さない）。
        sums = {r["checksum"] for r in before + after}
        digests = set()
        digest_err = None
        for arm in ("before", "after"):
            for r in range(1, ROUNDS + 1):
                try:
                    digests.add(load_digest(f"{d}/digest-{arm}-{label}-{dev}-{mode}-r{r}.log"))
                except Bad as e:
                    digest_err = str(e)
        if len(sums) != 1:
            res["verdict"] = "REJECT"
            res["reason"] = "正しさ: loss checksum が不一致"
            return res
        if digest_err is not None:
            res["verdict"] = "REJECT"
            res["reason"] = f"正しさ: digest 欠落または不正（不一致として扱う）: {digest_err}"
            return res
        if len(digests) != 1:
            res["verdict"] = "REJECT"
            res["reason"] = "正しさ: 更新後パラメータの digest が不一致"
            return res
        check_env(profile, f"{d}/env_info-{label}.txt", rule_sha)
        check_gate(profile, dev, f"{d}/load_gate-{label}-{dev}.log")
    except Bad as e:
        res["verdict"] = "undetermined"
        res["reason"] = str(e)
        return res

    ratio, per = res["ratio"], res["per"]
    if mode == "reuse":
        if ratio > REJECT_RATIO or all(p > 1.0 for p in per):
            res["verdict"], res["reason"] = "REJECT", "後退（ratio>1.00 または 5/5 round で ratio_r>1.00）"
        elif ratio <= ADOPT_RATIO and all(p < 1.0 for p in per):
            res["verdict"], res["reason"] = "ADOPT", "効果確認（ratio<=0.95 かつ 5/5 round で ratio_r<1.00）"
        else:
            res["reason"] = "ADOPT/REJECT どちらの条件も満たさない"
    else:
        ok = ratio <= GUARD_RATIO and not all(p > 1.0 for p in per)
        res["verdict"] = "guard-pass" if ok else "guard-fail"
        res["reason"] = "ガード（ratio<=1.03 かつ 5/5 後退でない）"
    return res


def ref_ratio(ref_paths, dev, median):
    """参考: 最速の相手 median に対する比。別セッションの参照値（判定外）。"""
    best = None
    for p in ref_paths:
        for ln in _read_lines(p):
            if not ln.strip():
                continue
            o = json.loads(ln)
            if o.get("task") == "train" and o.get("device") == dev and o.get("framework") != "fandhe-ai":
                v = o.get("median_s")
                if isinstance(v, (int, float)) and v > 0 and (best is None or v < best[0]):
                    best = (v, o["framework"])
    if best is None or median is None:
        return "n/a"
    return f"{median / best[0]:.2f}x ({best[1]})"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", required=True, help="RULE.txt のパス（sha256 を env_info と突合）")
    ap.add_argument("--machine", action="append", default=[], metavar="PROFILE=DIR:LABEL",
                    help="例: macmini=docs/.../macmini:3247-macmini（DIR は生ログを平置きしたディレクトリ）")
    ap.add_argument("--ref", action="append", default=[], metavar="PROFILE=JSONL",
                    help="相手比（参考）用の median5 JSONL。複数可")
    ap.add_argument("--require-both", action="store_true", help="macmini と gb10 の両方を必須にする")
    a = ap.parse_args(argv)

    try:
        with open(a.rule, "rb") as fh:
            rule_sha = hashlib.sha256(fh.read()).hexdigest()
    except OSError as e:
        print(f"error: RULE を開けない: {e}", file=sys.stderr)
        return 2
    machines = {}
    for spec in a.machine:
        m = re.match(r"^(macmini|gb10)=([^:]+):([A-Za-z0-9._-]+)$", spec)
        if not m:
            print(f"error: --machine の形式不正: {spec}", file=sys.stderr)
            return 2
        machines[m.group(1)] = (m.group(2), m.group(3))
    refs = {}
    for spec in a.ref:
        k, _, v = spec.partition("=")
        if k not in PROFILES or not v:
            print(f"error: --ref の形式不正: {spec}", file=sys.stderr)
            return 2
        refs.setdefault(k, []).append(v)
    for k, (d, label) in machines.items():
        if not SAFE.match(label):
            print("error: label 不正", file=sys.stderr)
            return 2
    if a.require_both and set(machines) != set(PROFILES):
        print("error: --require-both: 両機体が必要", file=sys.stderr)
        return 2
    if not machines:
        print("error: --machine が無い", file=sys.stderr)
        return 2

    results = []
    for prof in ("macmini", "gb10"):
        if prof not in machines:
            continue
        d, label = machines[prof]
        for dev in PROFILES[prof]["devices"]:
            for mode in ("reuse", "fresh"):
                results.append(judge_cell(prof, prof, dev, mode, d, label, rule_sha))

    out = [f"# #3247 判定（rule_sha256={rule_sha[:16]}…）", "",
           "| 機体 | device | mode | before 中央値 ms | after 中央値 ms | ratio | ratio_r（5 round） | 判定 | 理由 | 参考: 相手比 after（別セッション参照値・判定外） | 参考: 相手比 before |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        ms = lambda v: "n/a" if v is None else f"{v * 1e3:.4f}"
        per = ", ".join(f"{p:.3f}" for p in r["per"]) or "n/a"
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
        rp = refs.get(r["machine"], [])
        ra = ref_ratio(rp, r["dev"], r["ma"]) if rp else "n/a"
        rb = ref_ratio(rp, r["dev"], r["mb"]) if rp else "n/a"
        reason = r["reason"].replace("|", "/").replace("\n", " ")
        out.append(f"| {r['machine']} | {r['dev']} | {r['mode']} | {ms(r['mb'])} | {ms(r['ma'])} | {ratio} | {per} | {r['verdict']} | {reason} | {ra} | {rb} |")
    print("\n".join(out))

    treated = [r for r in results if r["mode"] == "reuse"]
    # 正しさの失敗（REJECT）はガードセルのものも終了コードへ反映する。
    if any(r["verdict"] == "REJECT" for r in results):
        return 3
    if any(r["verdict"] != "ADOPT" for r in treated):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
