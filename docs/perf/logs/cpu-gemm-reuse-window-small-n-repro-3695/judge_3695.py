#!/usr/bin/env python3
"""イシュー #3695: 別セッション複数系列で vafter と tunmmap の再現と分離を RULE.txt どおりに機械判定する（標準ライブラリのみ・fail-closed）。

役割: #3680 の judge_3680.py（段階 1・主判定・#3671 式・ゲートログ検査）と #3671 の aggregate.py／bench_judge.py の
ローダーを importlib で読み込んで流用し（元ファイルは編集しない）、系列ごとのゲート評価・起動順に並んだ系列から
ゲートを通った最初の 3 系列の機械的選択・N=512 の全体判定（RULE.txt「全体判定」の表を上から順に適用）を行う。
新しい閾値は作らない（ブリッジ v2 の 0.5〜2.0 と主判定の 0.5 のみ。多数決は 3 系列中 2 以上）。

データの構造的な不備（ファイル欠落・不正 JSON・rc 非 0・行の重複）は exit 1（判定を出さない）。
RULE の sha 不一致・専有／負荷ゲート不通過・系列間隔不足などは「その系列を record_only」にして次の系列へ進む。
checksum 規則（診断の bit 不一致・parity_fail_count != 0・bench の腕をまたぐ不一致）は全体を「判定不能（checksum）」とする。

使い方: judge_3695.py --series-root <dir> --series s1,s2,...   （起動順に並べる。手で並べ替えない）
        judge_3695.py --self-test
series-root の構成: <root>/<label>/{gb10/, bench/<腕>/, session-scripts/{excl_status.txt,load_gate_bench.log,env_info_3695.txt,steps.log}}
腕: base／vafter／tunmmap／vafter_tunmmap。
"""
import importlib.util
import os
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.dirname(HERE)
RULE_PATH = os.path.join(HERE, "RULE.txt")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


j80 = _load("judge_3680", os.path.join(LOGS, "cpu-gemm-reuse-window-small-n-3680", "judge_3680.py"))
agg = j80.agg
bj = j80.bj
JudgeError = bj.JudgeError

ARMS = ["base", "vafter", "tunmmap", "vafter_tunmmap"]
Y_ARMS = ARMS[1:]
SINGLE = ["vafter", "tunmmap"]
JUDGED_N = 512
USE_SERIES = 3
MAJORITY = 2
MIN_GAP_S = 1800
DISPLAY = {"vafter_tunmmap": "vafter+tunmmap"}
LABELS = ["s1", "s2", "s3", "s4", "s5"]

V_INSUFFICIENT = "判定不能（系列不足）"
V_CHECKSUM = "判定不能（checksum）"
V_NO_DIFF = "再現しない（N=512 の差が再現しない）"
V_NO_SUPPORT = "再現しない（支持が再現しない）"
V_SEPARATED = "再現した・分離できた"
V_NOT_SEPARATED = "再現した・分離できない"


class ChecksumFail(Exception):
    """RULE.txt の checksum 規則違反（全体を「判定不能（checksum）」にする。構造的不備の JudgeError とは区別する）。"""


def dn(arm):
    return DISPLAY.get(arm, arm)


def guard(fn, *a):
    """ローダーの例外のうち checksum／parity 規則違反だけを ChecksumFail に変換する（他は構造不備のまま伝播）。"""
    try:
        return fn(*a)
    except (JudgeError, agg.AggError) as e:
        msg = str(e)
        if "checksum bit 不一致" in msg or "parity_fail_count" in msg:
            raise ChecksumFail(msg)
        raise


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def load_steps(path):
    """steps.log から (hwlock acquired の UTC 時刻, 最後の hwlock released の UTC 時刻, 終端行) を返す。欠落は JudgeError。"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    acq = rel = None
    for ln in lines:
        parts = ln.split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            if parts[1] == "hwlock acquired" and acq is None:
                acq = parse_ts(parts[0])
            elif parts[1] == "hwlock released":
                rel = parse_ts(parts[0])
        except ValueError:
            raise JudgeError(f"{path}: 時刻が解析できない {ln!r}")
    done = [ln for ln in lines if ln.startswith("done.")]
    if acq is None or rel is None or not done:
        raise JudgeError(f"{path}: hwlock acquired／released／done. のいずれかが無い")
    return acq, rel, done[-1]


def load_env_3695(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise JudgeError(f"{path}: 読み込み不可 ({e})")
    env = {}
    for ln in text.splitlines():
        if ": " in ln and not ln.startswith(" "):
            k, v = ln.split(": ", 1)
            env[k.strip()] = v.strip()
    for k in ("series_label", "rev_stamp_head", "rev_stamp_dirty", "rule_sha256", "bench_binary_sha256"):
        if not env.get(k):
            raise JudgeError(f"{path}: {k} の記録が無い（ドリフトガード不能）")
    return env


def eval_series(root, label, prev, first):
    """系列のゲートを評価する。返り値: dict(label, reasons[], diag, cks_fail, acq, rel, env)。

    reasons が空でなければ record_only（判定に使わない）。prev は起動順で直前の系列の dict（無ければ None）、
    first は最初に起動した系列の dict（計測木・バイナリ同一性の基準。無ければ None）。
    """
    d = os.path.join(root, label)
    ss = os.path.join(d, "session-scripts")
    res = {"label": label, "reasons": [], "diag": None, "cks_fail": None, "acq": None, "rel": None, "env": None}
    why = res["reasons"]
    if not os.path.isdir(d):
        why.append("系列ディレクトリが無い")
        return res
    try:
        env = load_env_3695(os.path.join(ss, "env_info_3695.txt"))
        res["env"] = env
        if env["series_label"] != label:
            why.append(f"env_info の series_label={env['series_label']!r} がディレクトリと不一致")
        if env["rule_sha256"] != j80.sha256_file(RULE_PATH):
            why.append("rule_sha256 が RULE.txt の実ハッシュと不一致")
        if env["rev_stamp_dirty"] != "dirty=0":
            why.append(f"計測ツリーが dirty（{env['rev_stamp_dirty']}）")
        if first is not None and first["env"] is not None:
            if env["rev_stamp_head"] != first["env"]["rev_stamp_head"]:
                why.append("計測木の HEAD が最初の系列と不一致")
            if env["bench_binary_sha256"] != first["env"]["bench_binary_sha256"]:
                why.append("bench バイナリの sha256 が最初の系列と不一致")
    except JudgeError as e:
        why.append(f"env_info 不備: {e}")
    try:
        acq, rel, done = load_steps(os.path.join(ss, "steps.log"))
        res["acq"], res["rel"] = acq, rel
        if done != "done. rc=0":
            why.append(f"steps.log の終端行が {done!r}（done. rc=0 でない）")
        if prev is not None:
            if prev["rel"] is None or acq is None:
                why.append("直前の系列の時刻が不明で系列間隔を確認できない")
            else:
                gap = (acq - prev["rel"]).total_seconds()
                if gap < MIN_GAP_S:
                    why.append(f"系列間隔 {gap:.0f}s が {MIN_GAP_S}s 未満")
    except JudgeError as e:
        why.append(f"steps.log 不備: {e}")
    try:
        excl = bj.load_excl(os.path.join(ss, "excl_status.txt"))
        for lab in ("diag", "bench"):
            if excl.get(lab) != "pass":
                why.append(f"専有 excl={excl.get(lab)!r}（{lab}）")
    except JudgeError as e:
        why.append(f"excl_status 不備: {e}")
    try:
        bad = j80.load_gate_log_3680(os.path.join(ss, "load_gate_bench.log"), ARMS)
        if bad:
            why.append(f"bench の負荷ゲート未通過または未記録 {len(bad)} 件（例 {bad[0]}）")
    except JudgeError as e:
        why.append(f"負荷ゲートログ不備: {e}")
    try:
        diag = guard(agg.load_series, os.path.join(d, "gb10"))
        res["diag"] = diag
        if diag["kind"] != "main":
            why.append("gb10 は series_kind=main の系列でない")
        why.extend(diag["why"])
        if res["env"] is not None and diag["env"].get("rev_stamp_head") != res["env"]["rev_stamp_head"]:
            why.append("診断と bench の計測コミットが不一致")
    except ChecksumFail as e:
        res["cks_fail"] = str(e)
    except agg.AggError as e:
        why.append(f"診断系列の読み込み不可: {e}")
    return res


def select_series(root, labels):
    """起動順に評価し、ゲートを通った最初の USE_SERIES 系列で打ち切る。返り値: (評価済み全系列, 選択系列)。"""
    evaluated, selected = [], []
    first = prev = None
    for lab in labels:
        r = eval_series(root, lab, prev, first)
        evaluated.append(r)
        if first is None and r["env"] is not None:
            first = r
        prev = r
        if not r["reasons"]:
            selected.append(r)
        if len(selected) == USE_SERIES:
            break
    return evaluated, selected


def analyze_series(root, r):
    """選択した系列のデータを読み込み、N ごとの段階 1・腕ごとの G_Y と支持を返す。構造不備は JudgeError。"""
    d = os.path.join(root, r["label"])
    if r["cks_fail"]:
        raise ChecksumFail(r["cks_fail"])
    diag = r["diag"]
    bench, sums = {}, {}
    for a in ARMS:
        if not os.path.isdir(os.path.join(d, "bench", a)):
            raise JudgeError(f"{r['label']}: bench の腕 {a} が無い（RULE 必須。欠落は判定しない）")
        bench[a], sums[a] = guard(bj.load_bench_arm, os.path.join(d, "bench"), a)
    for n in agg.SIZES:
        allsum = set()
        for a in sums:
            allsum |= sums[a].get(n, set())
        if len(allsum) != 1:
            raise ChecksumFail(f"{r['label']}: bench の checksum が N={n} で一致しない: {sorted(allsum)}")
    ds = diag["summary"]
    out = {"label": r["label"], "n": {}, "sums": {n: next(iter(set().union(*[sums[a][n] for a in sums]))) for n in agg.SIZES}}
    for n in agg.SIZES:
        g_diag = ds[("resident_public", n, "iter_total")][0] - ds[("fresh", n, "iter_total")][0]
        base = bench["base"][n]
        g_bench = j80.med(base["phases"]["iter_total"]) - j80.med(base["fresh"])
        branch = j80.stage1(g_bench, g_diag)
        cell = {"g_diag": g_diag, "g_bench": g_bench, "branch": branch, "arms": {}}
        for y in Y_ARMS:
            c = bench[y][n]
            g_y = j80.med(c["phases"]["iter_total"]) - j80.med(c["fresh"])
            main_ok = j80.support_main(g_y, g_bench, g_diag)
            cell["arms"][y] = {
                "g": g_y, "main": main_ok, "rec": j80.support_record(g_y, g_diag),
                "med": j80.med(c["phases"]["iter_total"]),
                "lo": min(c["phases"]["iter_total"]), "hi": max(c["phases"]["iter_total"]),
                # 段階 1 が (a)/(b) の系列は「不支持」として数える（RULE.txt 段階 2）
                "counted": branch == "c" and main_ok,
            }
        out["n"][n] = cell
    return out


def overall(results, n_selected, checksum_failed):
    """RULE.txt「全体判定」の表を上から順に適用する。results は analyze_series の戻り値のリスト。返り値: (結果, 補足行)。"""
    notes = []
    if n_selected < USE_SERIES:
        return V_INSUFFICIENT, notes
    if checksum_failed:
        return V_CHECKSUM, notes
    c = sum(1 for r in results if r["n"][JUDGED_N]["branch"] == "c")
    notes.append(f"N={JUDGED_N}: 段階 1 が (c) の系列 = {c}/{USE_SERIES}")
    rep = {y: sum(1 for r in results if r["n"][JUDGED_N]["arms"][y]["counted"]) for y in Y_ARMS}
    for y in Y_ARMS:
        notes.append(f"rep({dn(y)}) = {rep[y]}/{USE_SERIES} 系列 → {'真' if rep[y] >= MAJORITY else '偽'}")
    if c <= 1:
        return V_NO_DIFF, notes
    rv = {y: rep[y] >= MAJORITY for y in Y_ARMS}
    if not rv["vafter"] and not rv["tunmmap"]:
        return V_NO_SUPPORT, notes
    singles = [y for y in SINGLE if rv[y]]
    both = rv["vafter_tunmmap"]
    if len(singles) == 1:
        if both:
            return f"{V_SEPARATED}（効いている腕は {singles[0]}）", notes
        return f"{V_NOT_SEPARATED}（もう一方のノブを足すと支持が消える。交互作用は断定しない）", notes
    notes.append(f"rep({dn('vafter_tunmmap')}) = {'真' if both else '偽'}（記録）")
    for r in results:
        a = r["n"][JUDGED_N]["arms"]
        le = a["vafter_tunmmap"]["g"] <= min(a["vafter"]["g"], a["tunmmap"]["g"])
        notes.append(f"{r['label']}: G_{{vafter+tunmmap}} <= min(G_vafter, G_tunmmap) = {'真' if le else '偽'}（記録）")
    return f"{V_NOT_SEPARATED}（単独の腕がどちらも超過分の半分以上を消す）", notes


STAGE_TEXT = j80.STAGE_TEXT


def report_series(r):
    for n in agg.SIZES:
        cell = r["n"][n]
        tag = "判定対象" if n == JUDGED_N else "記録のみ（判定外）"
        ratio = f"{cell['g_bench'] / cell['g_diag']:.3f}" if cell["g_diag"] != 0 else "n/a"
        print(f"  N={n} [{tag}] G_bench={cell['g_bench']:+.9f} G_diag={cell['g_diag']:+.9f} "
              f"G_bench/G_diag={ratio} 段階1={STAGE_TEXT[cell['branch']]}")
        for y in Y_ARMS:
            a = cell["arms"][y]
            print(f"    Y {dn(y)}: G_Y={a['g']:+.9f} iter_total_median={a['med']:.9f}[{a['lo']:.9f}-{a['hi']:.9f}] "
                  f"主判定支持={'yes' if a['main'] else 'no'} 記録(#3671式)支持={'yes' if a['rec'] else 'no'} "
                  f"集計に数える={'yes' if a['counted'] else 'no'}")


def run(root, labels):
    for lab in labels:
        if lab not in LABELS:
            raise JudgeError(f"系列ラベル {lab!r} は s1〜s5 のみ")
    if len(set(labels)) != len(labels):
        raise JudgeError("系列ラベルが重複している")
    evaluated, selected = select_series(root, labels)
    results, cks = [], None
    for r in selected:
        try:
            results.append(analyze_series(root, r))
        except ChecksumFail as e:
            cks = str(e)
            break
    verdict, notes = overall(results, len(selected), cks is not None)
    print("OFFICIAL:", "yes" if len(selected) == USE_SERIES else f"no（ゲートを通った系列が {len(selected)}/{USE_SERIES}）")
    print(f"評価した系列（起動順）: {', '.join(r['label'] for r in evaluated)}")
    for r in evaluated:
        used = r in selected
        state = "判定に使う" if used else "record_only（判定に使わない）"
        print(f"## 系列 {r['label']}: {state}")
        for w in r["reasons"]:
            print(f"  理由: {w}")
        if used and not cks:
            report_series(next(x for x in results if x["label"] == r["label"]))
    unevaluated = [lab for lab in labels if lab not in [r["label"] for r in evaluated]]
    if unevaluated:
        print(f"未評価（先に 3 系列が揃ったため不使用）: {', '.join(unevaluated)}")
    if cks:
        print(f"checksum 規則違反: {cks}")
    if len(results) == USE_SERIES:
        ref = {n: results[0]["sums"][n] for n in agg.SIZES}
        same = all(r["sums"][n] == ref[n] for r in results for n in agg.SIZES)
        print(f"記録: 系列をまたぐ bench checksum の一致 = {'一致' if same else '不一致（記録のみ。所見に書く）'}")
    print("## 結論（RULE.txt の規則どおり。因果の断定ではない。bench の窓の見直しは所有者判断で提案に留める）")
    for ln in notes:
        print(ln)
    print(f"N={JUDGED_N} の全体判定: {verdict}")


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        if not cond:
            print("SELF-TEST FAIL", name)
            ok = False

    rule_sha = j80.sha256_file(RULE_PATH)
    t0 = datetime(2026, 11, 1, 0, 0, 0, tzinfo=timezone.utc)

    def ts(sec):
        return datetime.fromtimestamp(t0.timestamp() + sec, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def make(root, label, idx, g, rp=0.0011, gap=3600, excl="pass", sha=None, binsha="bin1", cks=None,
             skip_rc=False, done="done. rc=0"):
        """合成系列。idx 番目は acquired=idx*(7200+gap)、released=+7200。g は腕ごとの G。"""
        d = os.path.join(root, label)
        ss = os.path.join(d, "session-scripts")
        os.makedirs(ss)

        def mut(r, rec):
            if rec["phase"] != "iter_total":
                return rec
            v = {"fresh": 0.001, "resident_public": rp, "rp_verify_touch": 0.0011, "rp_verify_gap": 0.0011,
                 "rp_hold_footprint": 0.0011, "rp_ref_compute": 0.0011}[rec["arm"]]
            return {**rec, "median_s": v}

        agg._gen_fixture(os.path.join(d, "gb10"), mutate=mut)
        for a in ARMS:
            bj._mk_bench(os.path.join(d, "bench"), a, g[a])
        if cks:
            p = os.path.join(d, "bench", "tunmmap", "gemm_fresh_n256_r1.jsonl")
            with open(p) as f:
                good = f.read()
            with open(p, "w") as f:
                f.write(good.replace('"checksum": "1.0"', '"checksum": "2.0"'))
        if skip_rc:
            os.remove(os.path.join(d, "bench", "base", "rc.log"))
        acq = idx * (7200 + gap)
        with open(os.path.join(ss, "steps.log"), "w") as f:
            f.write(f"{ts(acq)} hwlock acquired\n{ts(acq + 10)} start\n{ts(acq + 7190)} bench done\n"
                    f"{done}\n{ts(acq + 7200)} hwlock released\n")
        with open(os.path.join(ss, "excl_status.txt"), "w") as f:
            f.write(f"diag excl={excl}\nbench excl=pass\n")
        with open(os.path.join(ss, "load_gate_bench.log"), "w") as f:
            for a in ARMS:
                for n in agg.SIZES:
                    for r in range(1, bj.RUNS + 1):
                        for m in j80.MODES:
                            f.write(f"bench_{a}/n{n}/r{r}/{m} gate=pass\n")
        with open(os.path.join(ss, "env_info_3695.txt"), "w") as f:
            f.write(f"series_label: {label}\nrev_stamp_head: abc\nrev_stamp_dirty: dirty=0\n"
                    f"rule_sha256: {sha or rule_sha}\nbench_binary_sha256: {binsha}\n")

    def G(base=0.0004, vafter=0.0004, tunmmap=0.0004, vt=0.0004):
        return {"base": base, "vafter": vafter, "tunmmap": tunmmap, "vafter_tunmmap": vt}

    def verdict(specs, labels=None, **kw):
        """specs: [(label, gdict, make kwargs)]。判定結果の文字列を返す（stdout を捕捉）。"""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            for i, (lab, g, extra) in enumerate(specs):
                make(tmp, lab, i, g, **extra)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(tmp, labels or [s[0] for s in specs])
            out = buf.getvalue()
        line = [ln for ln in out.splitlines() if ln.startswith(f"N={JUDGED_N} の全体判定")][0]
        return line.split(": ", 1)[1], out

    def three(g, extras=None):
        ex = extras or [{}, {}, {}]
        return [("s1", g[0], ex[0]), ("s2", g[1], ex[1]), ("s3", g[2], ex[2])]

    only_v = G(vafter=0.0001)               # vafter が超過分を消す
    only_t = G(tunmmap=0.0001)
    both_single = G(vafter=0.0001, tunmmap=0.0001)
    # 1. vafter だけ再現・組み合わせも再現 → 分離できた
    v, _ = verdict(three([G(vafter=0.0001, vt=0.0001)] * 3))
    check("separated", v.startswith(V_SEPARATED) and "vafter" in v)
    # 2. tunmmap だけ再現・組み合わせは再現せず → 分離できない
    v, _ = verdict(three([only_t] * 3))
    check("not-separated-vt-false", v.startswith(V_NOT_SEPARATED) and "もう一方" in v)
    # 3. 両方再現 → 分離できない
    v, out = verdict(three([G(vafter=0.0001, tunmmap=0.0001, vt=0.0001)] * 3))
    check("both-singles", v.startswith(V_NOT_SEPARATED) and "どちらも" in v and "= 真（記録）" in out)
    # 4. 単独腕がどちらも再現しない（組み合わせだけ支持）→ 支持が再現しない
    v, _ = verdict(three([G(vt=0.0001)] * 3))
    check("no-support", v == V_NO_SUPPORT)
    # 5. c <= 1 → 差が再現しない（s2・s3 は逆転が消えた (a)）
    gone = G(base=-0.0001)
    v, _ = verdict(three([only_v, gone, gone]))
    check("c-le-1", v == V_NO_DIFF)
    # 6. (b) の系列は不支持として数える: s2 が (b)、s1・s3 が (c) で vafter 支持 → rep=2 で再現
    near = G(base=0.00015)  # G_diag=0.0001 → 比 1.5 → (b)。vafter が小さくても数えない
    v, out = verdict(three([only_v, G(base=0.00015, vafter=0.00005), only_v]))
    check("b-counts-unsupported", v.startswith(V_NOT_SEPARATED) or v.startswith(V_SEPARATED), )
    check("b-rep", "rep(vafter) = 2/3" in out)
    v, out = verdict(three([only_v, near, near]))
    check("b-twice-fails", v == V_NO_DIFF)
    # 7. 系列不足（2 系列のみ）
    v, _ = verdict([("s1", only_v, {}), ("s2", only_v, {})])
    check("insufficient", v == V_INSUFFICIENT)
    # 8. 専有不通過の s2 を飛ばして s4 を使う
    specs = [("s1", only_v, {}), ("s2", only_v, {"excl": "fail"}), ("s3", only_v, {}), ("s4", only_v, {})]
    v, out = verdict(specs)
    check("skip-to-s4", v.startswith(V_NOT_SEPARATED) and "## 系列 s2: record_only" in out and "## 系列 s4: 判定に使う" in out)
    # 9. 系列間隔 1800 秒未満で record_only
    v, out = verdict([("s1", only_v, {}), ("s2", only_v, {"gap": 600}), ("s3", only_v, {"gap": 3600}), ("s4", only_v, {})])
    check("gap-short", "## 系列 s2: record_only" in out and "系列間隔" in out)
    # 10. RULE sha 不一致
    v, out = verdict([("s1", only_v, {"sha": "0" * 64}), ("s2", only_v, {}), ("s3", only_v, {}), ("s4", only_v, {})])
    check("rule-sha", "## 系列 s1: record_only" in out and "rule_sha256" in out)
    # 11. バイナリ sha が最初の系列と違う系列は record_only
    v, out = verdict([("s1", only_v, {}), ("s2", only_v, {"binsha": "other"}), ("s3", only_v, {}), ("s4", only_v, {})])
    check("bin-sha", "## 系列 s2: record_only" in out and "bench バイナリ" in out)
    # 12. checksum 不一致 → 判定不能（checksum）
    v, _ = verdict(three([only_v] * 3, [{}, {"cks": True}, {}]))
    check("checksum", v == V_CHECKSUM)
    # 13. データ欠損（rc.log なし）→ JudgeError（exit 1）
    try:
        verdict(three([only_v] * 3, [{}, {"skip_rc": True}, {}]))
        check("missing-data", False)
    except JudgeError:
        pass
    # 14. 終端行が done. rc=0 でない系列は record_only
    v, out = verdict([("s1", only_v, {"done": "done. rc=gate-unpassed"}), ("s2", only_v, {}), ("s3", only_v, {}), ("s4", only_v, {})])
    check("done-rc", "## 系列 s1: record_only" in out)
    # 15. 不正ラベルは拒否
    try:
        run("/nonexistent", ["s1", "s6"])
        check("bad-label", False)
    except JudgeError:
        pass
    # 16. 純関数 overall の網羅（rep の境界）
    def fake(branch, sup):
        arms = {y: {"counted": sup.get(y, False) and branch == "c", "g": 0.0} for y in Y_ARMS}
        return {"label": "x", "n": {JUDGED_N: {"branch": branch, "arms": arms}}}
    r3 = [fake("c", {"vafter": True}), fake("c", {"vafter": True}), fake("c", {})]
    check("overall-rep2", overall(r3, 3, False)[0].startswith(V_NOT_SEPARATED))
    check("overall-checksum-first", overall(r3, 3, True)[0] == V_CHECKSUM)
    check("overall-insufficient-first", overall(r3[:2], 2, True)[0] == V_INSUFFICIENT)
    print("self-test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) != 4 or argv[0] != "--series-root" or argv[2] != "--series":
        print(__doc__, file=sys.stderr)
        return 2
    try:
        run(argv[1], [s for s in argv[3].split(",") if s])
    except (JudgeError, agg.AggError) as e:
        print(f"判定エラー: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
