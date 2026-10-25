#!/usr/bin/env python3
"""イシュー #3261: 学習ステップ全体の CUDA Graph capture（#3260）GB10 A/B の判定ツール（fail-closed）。

`docs/perf/logs/train-step-graph-ab-3261/RULE.txt` の [前提ゲート]・[hard 条件]・[比]・[判定] を機械的に適用する。
`run_ab_train_graph_cuda.sh` が出す生ログ群を読み、ADOPT / REJECT / UNDETERMINED を返す単一ソース。
寛容パース禁止: 欠損・型不正・余分な行・件数不足は例外にせず入力不正（終了コード 2）または
UNDETERMINED へ明示的に落とす。

セル: {chain, reuse, fresh} × {off, on}（すべて GB10 の train CUDA）。判定セルは chain、対照セルは reuse・fresh。
ratio = median(on)/median(off)、ratio_r = on_r/off_r（JSONL の行順 = round 順）。相手比は参考（判定外）。

終了コード: 0 = ADOPT / 3 = REJECT / 4 = UNDETERMINED / 2 = 入力不正。
ADOPT でも既定値は本ツールで切り替えない（設計 §8.6。所有者判断）。
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
LIMIT_RATIO = 1.00
CELLS = ("chain", "reuse", "fresh")
DIGEST_CELLS = ("chain", "reuse")
OPPONENTS = {"candle": "candle", "torch": "pytorch"}
GATE_LOAD1 = 1.0
# 前提ゲートのログ名 -> 期待 passed 件数（RULE.txt [前提ゲート]）
PREGATES = {
    "r0-3252": 3,
    "r0-real-device": 2,
    "r0-optin-off": 1,
    "r1-facade": 1,
    "r1-3260": 7,
    "r2-off": 1,
    "r2-on": 1,
}
TEST_RESULT_RE = re.compile(r"^test result: ok\. (\d+) passed; 0 failed;", re.M)
DIGEST_RE = re.compile(
    r"^PARAM_DIGEST task=train mode=(reuse|chain) device=cuda "
    r"n_tensors=(\d+) n_elems=(\d+) fnv1a64=(0x[0-9a-f]{16})$"
)


class Bad(Exception):
    """入力不正（終了コード 2）。"""


class Undet(Exception):
    """規則を満たさない計測（UNDETERMINED）。"""


def read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        raise Undet(f"{path}: 開けない ({e})")


def lines_of(path):
    return [ln.rstrip("\n") for ln in read_text(path).splitlines()]


def sha256_file(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError as e:
        raise Bad(f"{path}: 開けない ({e})")


def num(o, k, where):
    v = o.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Bad(f"{where}: {k} が有限数でない ({v!r})")
    return v


def load_rows(path, fandhe, mode=None, framework=None):
    """JSONL を厳格に読む。空行以外の全行が task=train・device=cuda・size=64 の object であること。"""
    rows = []
    for n, line in enumerate(lines_of(path), 1):
        if not line.strip():
            continue
        where = f"{path}:{n}"
        try:
            o = json.loads(line)
        except ValueError:
            raise Bad(f"{where}: JSON 不正")
        if not isinstance(o, dict):
            raise Bad(f"{where}: object でない")
        if o.get("task") != "train" or o.get("device") != "cuda":
            raise Bad(f"{where}: task/device が train/cuda でない")
        if isinstance(o.get("size"), bool) or o.get("size") != 64:
            raise Bad(f"{where}: size が 64 でない ({o.get('size')!r})")
        if fandhe:
            if o.get("framework") != "fandhe-ai" or o.get("mode") != mode:
                raise Bad(f"{where}: framework/mode が fandhe-ai/{mode} でない")
            if "tf32" in o and o["tf32"] is not False:
                raise Bad(f"{where}: tf32 が有効（RULE は TF32 OFF 固定）")
            if "graph" in o and o["graph"] is not None:
                raise Bad(f"{where}: graph 指定あり（RULE は --graph なし固定）")
        elif o.get("framework") != framework:
            raise Bad(f"{where}: framework が {framework!r} でない ({o.get('framework')!r})")
        num(o, "checksum", where)
        if num(o, "median_s", where) <= 0:
            raise Bad(f"{where}: median_s <= 0")
        rows.append(o)
    return rows


def load_digest(path, cell):
    hits = [ln for ln in lines_of(path) if ln.startswith("PARAM_DIGEST")]
    if len(hits) != 1:
        raise Undet(f"{path}: PARAM_DIGEST 行が {len(hits)} 件（1 件必須）")
    m = DIGEST_RE.match(hits[0])
    if not m:
        raise Bad(f"{path}: PARAM_DIGEST 行の形式不正")
    if m.group(1) != cell:
        raise Bad(f"{path}: digest の mode が {cell} でない")
    return m.groups()[1:]


def parse_kv(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def rule_patch_sha(rule_text):
    m = re.search(r"^  patch_sha256: ([0-9a-f]{64})$", rule_text, re.M)
    if not m:
        raise Bad("RULE.txt に patch_sha256 が無い")
    return m.group(1)


def check_env(path, rule_sha, psha):
    kv = {}
    for ln in lines_of(path):
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k] = v
    if kv.get("profile") != "gb10":
        raise Undet("env_info の profile が gb10 でない")
    if "GB10" not in kv.get("gpu", ""):
        raise Undet(f"env_info の gpu に GB10 を含まない ({kv.get('gpu')!r})")
    if not kv.get("uname", "").endswith("aarch64"):
        raise Undet("env_info の uname が aarch64 でない")
    if kv.get("rule_sha256") != rule_sha:
        raise Undet("env_info の rule_sha256 が現 RULE.txt と一致しない")
    if kv.get("patch_sha256") != psha:
        raise Undet("env_info の patch_sha256 が RULE の固定値と一致しない")
    if kv.get("precheck") != "assert" or kv.get("arms") != "same_binary_env_toggle":
        raise Undet("env_info の precheck/arms が規則と不一致")
    if kv.get("rounds") != str(ROUNDS) or kv.get("tf32") != "off":
        raise Undet("env_info の rounds/tf32 が規則と不一致")
    if not re.match(r"^[0-9a-f]{40}$", kv.get("rev_stamp_head_runner", "")):
        raise Undet("env_info の rev_stamp_head_runner が欠落または不正")
    if not re.match(r"^[0-9a-f]{64}$", kv.get("bin_sha256", "")):
        raise Undet("env_info の bin_sha256 が欠落または不正")


def check_gate(path):
    lines = [ln.strip() for ln in lines_of(path) if ln.strip()]
    if not lines:
        raise Undet("専有ゲートログが空")
    head = parse_kv(lines[0])
    if head.get("series") != "official" or head.get("precheck") != "assert":
        raise Undet("専有ゲートログが official/assert でない")
    for key in ("threshold", "rule_threshold"):
        try:
            got = float(head.get(key, ""))
        except ValueError:
            got = None
        if got != GATE_LOAD1:
            raise Undet(f"{key} が規則値 {GATE_LOAD1} と不一致")
    rounds = lines[1:]
    if [ln.split()[0] for ln in rounds] != [f"round{i}" for i in range(1, ROUNDS + 1)]:
        raise Undet("round 行が round1〜5 と一致しない")
    for ln in rounds:
        r = parse_kv(ln)
        if r.get("dev") != "cuda" or r.get("gate") != "pass" or r.get("gate_ok") != "1":
            raise Undet(f"専有ゲート不通過: {ln.split()[0]}")


def check_rc0(text, path):
    rcs = re.findall(r"^rc=(\d+)$", text, re.M)
    if len(rcs) != 1 or rcs[0] != "0":
        raise Undet(f"{path}: rc=0 が 1 行だけある形でない ({rcs})")


def pregate_state(d, label):
    """前提ゲートの状態を返す: ('ok'|'fail', 理由)。ログ欠落・未実行は Undet。fail は REJECT 要因（R1・R2）。"""
    fails = []
    for name, want in PREGATES.items():
        path = f"{d}/{name}-{label}.log"
        text = read_text(path)
        try:
            check_rc0(text, path)
            counts = TEST_RESULT_RE.findall(text)
            if len(counts) != 1 or int(counts[0]) != want:
                raise Undet(f"{path}: test result ok の passed 件数が {want} 件でない ({counts})")
        except Undet as e:
            # R1・R2 の失敗は結線・bit 一致の失敗として REJECT、R0 の失敗は計測不成立（UNDETERMINED）
            if name.startswith(("r1-", "r2-")):
                fails.append(f"{name}: {e}")
            else:
                raise
    off = [ln for ln in lines_of(f"{d}/bits-off-{label}.txt") if ln.strip()]
    on = [ln for ln in lines_of(f"{d}/bits-on-{label}.txt") if ln.strip()]
    if not off or not on:
        raise Undet("R2: BITS 行が 0 行")
    if any(not ln.startswith("BITS ") for ln in off + on):
        raise Bad("R2: BITS 以外の行が混入している")
    if off != on:
        fails.append("R2: BITS が off／on で不一致（bit 不一致）")
    return fails


def opponent_ratio(d, label, fandhe_rows):
    """参考: 同一 round 内の有効な相手最速 ÷ fandhe の 5 round 中央値。読めなければ n/a（判定外）。"""
    series = []
    for opp, fw in OPPONENTS.items():
        try:
            rows = load_rows(f"{d}/results-{opp}-{label}-cuda-train.jsonl", False, framework=fw)
        except (Bad, Undet):
            continue
        if len(rows) == ROUNDS:
            series.append(rows)
    if not series:
        return "n/a"
    fmed = statistics.median(x["median_s"] for x in fandhe_rows)
    return f"{statistics.median(min(s[r]['median_s'] for s in series) / fmed for r in range(ROUNDS)):.2f}x"


def judge(d, label, rule_path):
    rule_text = read_text(rule_path)
    rule_sha = sha256_file(rule_path)
    psha = rule_patch_sha(rule_text)
    undet, reject = [], []
    cells = {}
    # 1. 取り込み（入力不正は Bad で即終了）
    for cell in CELLS:
        try:
            off = load_rows(f"{d}/results-{cell}-off-{label}-cuda-train.jsonl", True, cell)
            on = load_rows(f"{d}/results-{cell}-on-{label}-cuda-train.jsonl", True, cell)
        except Undet as e:
            undet.append(str(e))
            continue
        if len(off) != ROUNDS or len(on) != ROUNDS:
            undet.append(f"{cell}: run 数が 5 でない (off={len(off)} on={len(on)})")
            continue
        cells[cell] = (off, on)
    for fn, args in ((check_env, (f"{d}/env_info-{label}.txt", rule_sha, psha)), (check_gate, (f"{d}/load_gate-{label}-cuda.log",))):
        try:
            fn(*args)
        except Undet as e:
            undet.append(str(e))
    if lines_of_safe(f"{d}/skipped-train-graph-{label}.log"):
        undet.append("skipped ログが空でない（失敗 run・ビルド失敗の痕跡）")
    pre_fails = []
    try:
        pre_fails = pregate_state(d, label)
    except Undet as e:
        undet.append(str(e))
    # 2. hard 条件
    for cell, (off, on) in cells.items():
        if len({r["checksum"] for r in off + on}) != 1:
            reject.append(f"{cell}: checksum が off／on で不一致")
        if cell in DIGEST_CELLS:
            digests = set()
            try:
                for arm in ("off", "on"):
                    for r in range(1, ROUNDS + 1):
                        digests.add(load_digest(f"{d}/digest-{cell}-{arm}-{label}-cuda-train-r{r}.log", cell))
            except Undet as e:
                reject.append(f"{cell}: digest 欠落（不一致扱い）: {e}")
            else:
                if len(digests) != 1:
                    reject.append(f"{cell}: PARAM_DIGEST が off／on で不一致")
    reject.extend(pre_fails)
    # 3. 比
    table = []
    ratios = {}
    for cell, (off, on) in cells.items():
        moff = statistics.median(r["median_s"] for r in off)
        mon = statistics.median(r["median_s"] for r in on)
        per = [a["median_s"] / b["median_s"] for b, a in zip(off, on)]
        ratios[cell] = (mon / moff, per)
        table.append((cell, moff, mon, mon / moff, per, opponent_ratio(d, label, off), opponent_ratio(d, label, on)))
    verdict_notes = []
    if "chain" in ratios and not undet:
        rc, per = ratios["chain"]
        if rc > LIMIT_RATIO:
            reject.append(f"chain: ratio {rc:.3f} > 1.00")
        elif not (rc <= ADOPT_RATIO and all(p < 1.0 for p in per)):
            verdict_notes.append("chain: ADOPT 条件（ratio<=0.95 かつ 5/5 round で ratio_r<1.00）を満たさない")
    for cell in ("reuse", "fresh"):
        if cell in ratios and ratios[cell][0] > LIMIT_RATIO:
            reject.append(f"{cell}: ratio {ratios[cell][0]:.3f} > 1.00（stream 変更の後退）")
    # 4. 判定（入力不正 → undetermined 要因 → hard/前提ゲート失敗 → 比）
    if undet:
        verdict = "UNDETERMINED"
    elif reject:
        verdict = "REJECT"
    elif verdict_notes:
        verdict = "UNDETERMINED"
        undet.extend(verdict_notes)
    else:
        verdict = "ADOPT"
    return verdict, table, undet, reject, rule_sha


def lines_of_safe(path):
    """skipped ログ: 存在しない場合は空とみなさず Undet 相当で非空扱いにする。"""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return ["missing"]
    return [ln for ln in text.splitlines() if ln.strip()]


def render(label, verdict, table, undet, reject, rule_sha):
    out = [f"# #3261 判定 {label}（rule_sha256={rule_sha[:16]}…）", "", f"**判定: {verdict}**", ""]
    out += ["| セル | off 中央値 ms | on 中央値 ms | ratio | ratio_r（5 round） | 相手最速÷off（参考） | 相手最速÷on（参考） |",
            "|---|---|---|---|---|---|---|"]
    for cell, moff, mon, ratio, per, oo, on in table:
        role = "判定" if cell == "chain" else "対照"
        out.append(f"| {cell}（{role}） | {moff * 1e3:.4f} | {mon * 1e3:.4f} | {ratio:.3f} | "
                   f"{', '.join(f'{p:.3f}' for p in per)} | {oo} | {on} |")
    for title, items in (("REJECT 要因", reject), ("UNDETERMINED 要因", undet)):
        if items:
            out += ["", f"## {title}"] + [f"- {x}".replace("\n", " ") for x in items]
    out += ["", "相手比は参考値で判定に使わない。推定値は本表に含まない（事前表明の仮説は RULE.txt を参照）。"]
    return "\n".join(out)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9._-]+$", a.label):
        print("error: label 形式不正", file=sys.stderr)
        return 2
    try:
        verdict, table, undet, reject, rule_sha = judge(a.dir, a.label, a.rule)
    except Bad as e:
        print(f"error: 入力不正: {e}", file=sys.stderr)
        return 2
    print(render(a.label, verdict, table, undet, reject, rule_sha))
    return {"ADOPT": 0, "REJECT": 3, "UNDETERMINED": 4}[verdict]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
