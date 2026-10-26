#!/usr/bin/env python3
"""イシュー #3596: 学習チェーンと Graph capture の ordinal 単位排他（#3597）の GB10 非後退判定ツール（fail-closed）。

`docs/perf/logs/train-chain-exclusion-3596/RULE.txt` の [前提ゲート]・[hard 条件]・[比]・[判定] を機械的に適用する。
`run_ab_train_chain_exclusion_cuda.sh` が出す生ログ群を読み、NON-REGRESSION / REGRESSION / UNDETERMINED を返す単一ソース。
排他に当たる実行時トグルは無いため、比較は 2 ツリー（excl0 = #3597 の直前・excl1 = 排他あり）の別バイナリ間で行う。
寛容パース禁止: 欠損・型不正・余分な行・件数不足は例外にせず入力不正（終了コード 2）または UNDETERMINED へ明示的に落とす。

セル: chain（GB10・train CUDA・size=64・TF32 OFF）× graph モード {off, on} × 排他 {excl0, excl1}。
ratio_m = median(excl1-m の 5 run の median_s) / median(excl0-m の 5 run の median_s)、ratio_r = 同一 round 内の比。

終了コード: 0 = NON-REGRESSION / 3 = REGRESSION / 4 = UNDETERMINED / 2 = 入力不正。
本ツールの結果で既定値（TRAIN_GRAPH_DEFAULT_ENABLED）・tolerance・閾値は切り替えない（所有者判断）。
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys

ROUNDS = 5
LIMIT_RATIO = 1.00
MODES = ("off", "on")
ARMS = tuple(f"excl{e}-{m}" for m in MODES for e in (0, 1))
# 専有ゲート定数（ランナーと同値。ドリフトガードテストの対象外。RULE.txt [ドリフトガードの除外]）
GATE_LOAD1 = 1.0
GATE_MAX_WAIT_S = 1800
# 前提ゲートのログ名 -> 期待 passed 件数（RULE.txt [前提ゲート]）
PREGATES = {
    "g1-unfenced": 1,
    "g2-3260": 9,
    "g3-3252": 3,
    "g3-real-device": 2,
    "g3-optin-off": 1,
    "g4-facade": 1,
    "g4-off": 1,
    "g4-on": 1,
}
PREGATE_STOP_MARK = "pregate-stop-{label}.txt"
TEST_RESULT_RE = re.compile(r"^test result: ok\. (\d+) passed; 0 failed;", re.M)
DIGEST_RE = re.compile(
    r"^PARAM_DIGEST task=train mode=chain device=cuda "
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


def load_rows(path):
    """JSONL を厳格に読む。空行以外の全行が fandhe-ai・chain・train・cuda・size=64 の object であること。"""
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
        if o.get("framework") != "fandhe-ai" or o.get("mode") != "chain":
            raise Bad(f"{where}: framework/mode が fandhe-ai/chain でない")
        if "tf32" in o and o["tf32"] is not False:
            raise Bad(f"{where}: tf32 が有効（RULE は TF32 OFF 固定）")
        if "graph" in o and o["graph"] is not None:
            raise Bad(f"{where}: graph 指定あり（RULE は --graph なし固定）")
        num(o, "checksum", where)
        if num(o, "median_s", where) <= 0:
            raise Bad(f"{where}: median_s <= 0")
        rows.append(o)
    return rows


def load_digest(path):
    hits = [ln for ln in lines_of(path) if ln.startswith("PARAM_DIGEST")]
    if len(hits) != 1:
        raise Undet(f"{path}: PARAM_DIGEST 行が {len(hits)} 件（1 件必須）")
    m = DIGEST_RE.match(hits[0])
    if not m:
        raise Bad(f"{path}: PARAM_DIGEST 行の形式不正")
    return m.groups()


def parse_kv(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def rule_field(rule_text, pattern, what):
    m = re.search(pattern, rule_text, re.M)
    if not m:
        raise Bad(f"RULE.txt に {what} が無い")
    return m.group(1)


def check_env(path, rule_sha, rule):
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
    if kv.get("patch_sha256") != rule["patch"]:
        raise Undet("env_info の patch_sha256 が RULE の固定値と一致しない")
    if kv.get("rev_stamp_head_before") != rule["before"]:
        raise Undet("env_info の rev_stamp_head_before が RULE の before SHA と一致しない")
    if kv.get("crates_identical_to") != rule["crates_ref"]:
        raise Undet("env_info の crates_identical_to が RULE の基準 SHA と一致しない（after の crates/ 同一性が未検証）")
    if kv.get("precheck") != "assert" or kv.get("arms") != "two_trees_env_toggle":
        raise Undet("env_info の precheck/arms が規則と不一致")
    if kv.get("rounds") != str(ROUNDS) or kv.get("tf32") != "off":
        raise Undet("env_info の rounds/tf32 が規則と不一致")
    for k, n in (("rev_stamp_head_after", 40), ("rev_stamp_head_runner", 40), ("bin_sha256_excl0", 64), ("bin_sha256_excl1", 64)):
        if not re.match(r"^[0-9a-f]{%d}$" % n, kv.get(k, "")):
            raise Undet(f"env_info の {k} が欠落または不正")
    if kv["bin_sha256_excl0"] == kv["bin_sha256_excl1"]:
        raise Undet("excl0／excl1 のバイナリ sha256 が同一（腕が区別できていない）")


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
        name = ln.split()[0]
        if r.get("dev") != "cuda" or r.get("gate") != "pass" or r.get("gate_ok") != "1":
            raise Undet(f"専有ゲート不通過: {name}")
        try:
            thr = float(r.get("threshold", ""))
            l1 = float(r.get("load1", ""))
            waited = float(r.get("waited_s", ""))
        except ValueError:
            raise Undet(f"専有ゲートの threshold/load1/waited_s が欠落または数値でない: {name}")
        if not (math.isfinite(thr) and math.isfinite(l1) and math.isfinite(waited)):
            raise Undet(f"専有ゲートの実測値が有限でない: {name}")
        if thr != GATE_LOAD1:
            raise Undet(f"専有ゲートの threshold が規則値 {GATE_LOAD1} と不一致: {name}")
        if not (0.0 <= l1 < GATE_LOAD1):
            raise Undet(f"専有ゲートの load1 が {GATE_LOAD1} 未満でない ({l1}): {name}")
        if not (0.0 <= waited <= GATE_MAX_WAIT_S):
            raise Undet(f"専有ゲートの waited_s が 0〜{GATE_MAX_WAIT_S} 秒の範囲外 ({waited}): {name}")


def check_rc0(text, path):
    rcs = re.findall(r"^rc=(\d+)$", text, re.M)
    if len(rcs) != 1 or rcs[0] != "0":
        raise Undet(f"{path}: rc=0 が 1 行だけある形でない ({rcs})")


def pregate_state(d, label):
    """前提ゲートの失敗一覧を返す（空なら全合格）。ログ欠落・未実行は Undet。
    ログが存在して不合格（rc!=0・passed 件数不一致）は正しさの失敗として REGRESSION 要因にする。"""
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
            fails.append(f"{name}: {e}")
    off = [ln for ln in lines_of(f"{d}/bits-off-{label}.txt") if ln.strip()]
    on = [ln for ln in lines_of(f"{d}/bits-on-{label}.txt") if ln.strip()]
    raw_off = [ln for ln in lines_of(f"{d}/g4-off-{label}.log") if ln.startswith("BITS")]
    raw_on = [ln for ln in lines_of(f"{d}/g4-on-{label}.log") if ln.startswith("BITS")]
    if raw_off != off or raw_on != on:
        fails.append("G4: bits-off／on 抽出ファイルが生ログの BITS 行と一致しない")
    if raw_off != raw_on:
        fails.append("G4: 生ログの BITS が off／on で不一致（bit 不一致）")
    if not raw_off or not raw_on:
        fails.append("G4: BITS 行が 0 行")
    if any(not ln.startswith("BITS ") for ln in off + on):
        raise Bad("G4: BITS 以外の行が混入している")
    return fails


def lines_of_safe(path):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return ["missing"]
    return [ln for ln in text.splitlines() if ln.strip()]


def read_rule(rule_path):
    try:
        rule_text = read_text(rule_path)
        rule_sha = sha256_file(rule_path)
    except Undet as e:
        raise Bad(f"RULE 読取不能: {e}")
    rule = {
        "patch": rule_field(rule_text, r"^  patch_sha256: ([0-9a-f]{64})$", "patch_sha256"),
        "before": rule_field(rule_text, r"^- before = ([0-9a-f]{40})\b", "before SHA"),
        "crates_ref": rule_field(rule_text, r"^- after_crates_identical_to = ([0-9a-f]{40})\b", "after_crates_identical_to"),
    }
    return rule_sha, rule


def early_pregate_stop(d, label, rule_sha, rule):
    """前提ゲート不合格で計測前に停止した run の判定。印が無ければ None（通常経路）。
    印は主張にすぎないため、env_info と前提ゲートの生ログから失敗を独立に再確認し、
    再確認できなければ UNDETERMINED に倒す。"""
    mark = f"{d}/" + PREGATE_STOP_MARK.format(label=label)
    try:
        with open(mark, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise Bad(f"{mark}: 開けない ({e})")
    undet, reject = [], []
    if not [ln for ln in text.splitlines() if ln.strip()]:
        undet.append("前提ゲート停止の印が空")
    try:
        check_env(f"{d}/env_info-{label}.txt", rule_sha, rule)
        reject.extend(pregate_state(d, label))
    except Undet as e:
        undet.append(str(e))
    if not undet and not reject:
        undet.append("前提ゲート停止の印があるが、ログから G1〜G4 の失敗を再確認できない")
    if undet:
        return "UNDETERMINED", [], undet, reject, rule_sha
    return "REGRESSION", [], undet, reject, rule_sha


def judge(d, label, rule_path):
    rule_sha, rule = read_rule(rule_path)
    stop = early_pregate_stop(d, label, rule_sha, rule)
    if stop is not None:
        return stop
    undet, reject = [], []
    arms = {}
    for arm in ARMS:
        try:
            rows = load_rows(f"{d}/results-{arm}-{label}-cuda-train.jsonl")
        except Undet as e:
            undet.append(str(e))
            continue
        if len(rows) != ROUNDS:
            undet.append(f"{arm}: run 数が 5 でない ({len(rows)})")
            continue
        arms[arm] = rows
    for fn, args in ((check_env, (f"{d}/env_info-{label}.txt", rule_sha, rule)),
                     (check_gate, (f"{d}/load_gate-{label}-cuda.log",))):
        try:
            fn(*args)
        except Undet as e:
            undet.append(str(e))
    if lines_of_safe(f"{d}/skipped-train-chain-exclusion-{label}.log"):
        undet.append("skipped ログが空でない（失敗 run・ビルド失敗の痕跡）")
    pre_fails = []
    try:
        pre_fails = pregate_state(d, label)
    except Undet as e:
        undet.append(str(e))
    # hard 条件: 4 腕すべての checksum・PARAM_DIGEST が全 run で同値（欠落は不一致扱い）
    if len(arms) == len(ARMS) and len({r["checksum"] for rows in arms.values() for r in rows}) != 1:
        reject.append("checksum が 4 腕・全 run で不一致")
    digests = set()
    try:
        for arm in ARMS:
            for r in range(1, ROUNDS + 1):
                digests.add(load_digest(f"{d}/digest-{arm}-{label}-cuda-train-r{r}.log"))
    except Undet as e:
        reject.append(f"digest 欠落（不一致扱い）: {e}")
    else:
        if len(digests) != 1:
            reject.append("PARAM_DIGEST が 4 腕・全 run で不一致")
    reject.extend(pre_fails)
    # 比
    table = []
    ratios = {}
    for m in MODES:
        a0, a1 = arms.get(f"excl0-{m}"), arms.get(f"excl1-{m}")
        if a0 is None or a1 is None:
            continue
        m0 = statistics.median(r["median_s"] for r in a0)
        m1 = statistics.median(r["median_s"] for r in a1)
        per = [y["median_s"] / x["median_s"] for x, y in zip(a0, a1)]
        ratios[m] = (m1 / m0, per)
        table.append((m, m0, m1, m1 / m0, per))
    notes = []
    for m, (ratio, per) in ratios.items():
        if ratio > LIMIT_RATIO:
            if all(p > 1.0 for p in per):
                reject.append(f"{m}: ratio {ratio:.3f} > 1.00 かつ 5/5 round で ratio_r > 1.00")
            else:
                notes.append(f"{m}: ratio {ratio:.3f} > 1.00 だが 5/5 の符号一貫がない")
    # 判定の優先順: 入力不正 → undetermined 要因 → 正しさの失敗 → 比
    if undet:
        verdict = "UNDETERMINED"
    elif reject:
        verdict = "REGRESSION"
    elif notes:
        verdict = "UNDETERMINED"
        undet.extend(notes)
    else:
        verdict = "NON-REGRESSION"
    return verdict, table, undet, reject, rule_sha


def render(label, verdict, table, undet, reject, rule_sha):
    out = [f"# #3596 判定 {label}（rule_sha256={rule_sha[:16]}…）", "", f"**判定: {verdict}**", ""]
    out += ["| graph モード | 排他なし中央値 ms | 排他あり中央値 ms | ratio | ratio_r（5 round） |", "|---|---|---|---|---|"]
    for m, m0, m1, ratio, per in table:
        out.append(f"| {m} | {m0 * 1e3:.4f} | {m1 * 1e3:.4f} | {ratio:.3f} | {', '.join(f'{p:.3f}' for p in per)} |")
    for title, items in (("REGRESSION 要因", reject), ("UNDETERMINED 要因", undet)):
        if items:
            out += ["", f"## {title}"] + [f"- {x}".replace("\n", " ") for x in items]
    out += ["", "推定値は本表に含まない（事前表明の仮説は RULE.txt を参照）。"]
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
    return {"NON-REGRESSION": 0, "REGRESSION": 3, "UNDETERMINED": 4}[verdict]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
