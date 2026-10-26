#!/usr/bin/env python3
"""イシュー #3611: 常駐チェーン CPU train の REGRESS 切り分け診断（DIAG.txt）の集計ツール（fail-closed・stdlib のみ）。

`run_diag_resident_chain_cpu.sh` が機体ごとに出す生ログを読み、DIAG の [分類規則] をそのまま機械適用して
腕ごとの代表値・比・A/A ノイズ幅・分類 (i)〜(v) を出す。これは判定（IMPROVE／REGRESS）ではなく分類であり、
judge_resident_chain_ab.py の入力でも置き換えでもない。推定値は出さない。
入力の読み取り部（JSONL・PARAM_DIGEST・負荷ゲートログ）は judge_resident_chain_ab.py と同じ関数を再利用する
（二重定義しない）。欠損・型不正・件数不足・不一致は例外にせず終了コード 2（入力不正）または 4（判定不能）で止める
（寛容パース禁止）。

終了コード: 0 = 分類を出力（(v) 判定不能の分類も 0。規則に当たらなかっただけで入力は正しい）/
2 = 入力不正（DIAG 読取不能・label 不正）/ 4 = 判定不能（負荷ゲート不通過・hard 条件不一致・env_info 不一致・run 欠落）。
"""
import argparse
import hashlib
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_resident_chain_ab as j  # noqa: E402

ROUNDS = 5
ARMS = ("aa0", "aa1", "ch1", "rb0", "rb1", "o0", "o1")
# 腕 -> (バイナリ群, --mode)。バイナリ群: B1 = aa0/aa1/ch1、B2 = rb0/rb1、B0 = o0/o1。
ARM_MODE = {"aa0": "reuse", "aa1": "reuse", "ch1": "chain", "rb0": "reuse", "rb1": "chain", "o0": "reuse", "o1": "chain"}
BIN_GROUP = {"aa0": "B1", "aa1": "B1", "ch1": "B1", "rb0": "B2", "rb1": "B2", "o0": "B0", "o1": "B0"}
PROFILES = ("macmini", "gb10")
MIN_CHAIN_ROUNDS = 4  # (ii) の round 内比の条件（DIAG 固定）


class Bad(Exception):
    """入力不正（終了コード 2）。"""


class Undet(Exception):
    """判定不能（終了コード 4）。"""


def diag_field(text, pattern, what):
    m = re.search(pattern, text, re.M)
    if not m:
        raise Bad(f"DIAG に {what} が無い")
    return m.group(1)


def read_env(path):
    kv = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh.read().splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    kv[k] = v
    except OSError as e:
        raise Undet(f"{path}: 開けない ({e})")
    return kv


def check_env(profile, kv, diag_sha, diag_text):
    if kv.get("profile") != profile:
        raise Undet(f"env_info の profile が {profile} でない ({kv.get('profile')!r})")
    if profile == "macmini":
        if kv.get("cpu_brand") != "Apple M4":
            raise Undet(f"Mac mini の cpu_brand が Apple M4 でない ({kv.get('cpu_brand')!r})")
    elif "GB10" not in kv.get("gpu", "") or not kv.get("uname", "").endswith("aarch64"):
        raise Undet("env_info が GB10（aarch64）でない")
    if kv.get("diag_sha256") != diag_sha:
        raise Undet("env_info の diag_sha256 が現 DIAG と一致しない")
    want = {
        "patch_sha256": diag_field(diag_text, r"^  patch_sha256: ([0-9a-f]{64})$", "patch_sha256"),
        "layout_shift_sha256": diag_field(diag_text, r"^  layout_shift_sha256: ([0-9a-f]{64})$", "layout_shift_sha256"),
        "rev_stamp_head_b1": diag_field(diag_text, r"^- b1 = ([0-9a-f]{40})$", "b1 SHA"),
        "rev_stamp_head_b0": diag_field(diag_text, r"^- oldcode = ([0-9a-f]{40})$", "oldcode SHA"),
    }
    for k, v in want.items():
        if kv.get(k) != v:
            raise Undet(f"env_info の {k} が DIAG の固定値と一致しない")
    if kv.get("precheck") != "assert" or kv.get("rounds") != str(ROUNDS) or kv.get("tf32") != "off":
        raise Undet("env_info の precheck/rounds/tf32 が規則と不一致")
    for a in ARMS:
        if not re.match(r"^[0-9a-f]{64}$", kv.get(f"bin_sha256_{a}", "")):
            raise Undet(f"env_info の bin_sha256_{a} が欠落または不正")
    grp = {}
    for a in ARMS:
        g = BIN_GROUP[a]
        sha = kv[f"bin_sha256_{a}"]
        if grp.setdefault(g, sha) != sha:
            raise Undet(f"同じバイナリ群 {g} の腕で bin_sha256 が食い違う（{a}）")
    if len(set(grp.values())) != 3:
        raise Undet("B0・B1・B2 のバイナリが互いに異なっていない（配置の対照が成立しない）")
    n = kv.get("layout_symbol_diff", "")
    if not re.match(r"^[0-9]+$", n) or int(n) < 1:
        raise Undet(f"B1／B2 の主要シンボルのアドレス差が確認できていない（layout_symbol_diff={n!r}）")


def load_arm(d, label, arm):
    """腕の 5 run を (median_s のリスト, checksum 集合, digest 集合) で返す。行順 = round。"""
    mode = ARM_MODE[arm]
    res = f"{d}/results-diag-{arm}-{label}-cpu-train.jsonl"
    try:
        rows = j.load_rows(res, "train", "cpu", mode)
        if len(rows) != ROUNDS:
            raise Undet(f"{res}: run が {len(rows)} 件（{ROUNDS} 件必須）")
        digests = [j.load_digest(f"{d}/digest-diag-{arm}-{label}-cpu-train-r{r}.log", mode, "cpu")
                   for r in range(1, ROUNDS + 1)]
    except j.Bad as e:
        raise Undet(str(e))
    return [r["median_s"] for r in rows], {r["checksum"] for r in rows}, set(digests)


def classify(med):
    """med: arm -> 5 round の median_s。DIAG の [分類規則] を適用し、数値と分類を返す。"""
    rep = {a: statistics.median(med[a]) for a in ARMS}
    rho = {a: rep[a] / rep["aa0"] for a in ARMS}
    rho_r = {a: [med[a][i] / med["aa0"][i] for i in range(ROUNDS)] for a in ARMS}
    n_w = max(abs(rho["aa1"] - 1.0), max(abs(x - 1.0) for x in rho_r["aa1"]))
    d1 = rho["ch1"] - 1.0
    d2 = rho["rb1"] / rho["rb0"] - 1.0
    p = abs(rho["rb0"] - 1.0)
    c1 = sum(1 for x in rho_r["ch1"] if x > 1.0)
    c2 = sum(1 for i in range(ROUNDS) if med["rb1"][i] / med["rb0"][i] > 1.0)
    out = {"rep": rep, "rho": rho, "rho_r": rho_r, "N": n_w, "D1": d1, "D2": d2, "P": p, "c1": c1, "c2": c2,
           "o_ratio": rep["o1"] / rep["o0"] - 1.0}
    e1, e2 = abs(d1) > n_w, abs(d2) > n_w
    if not e1 and not e2:
        out["case"], out["conclusion"] = "i", "ノイズ幅内。A/A の幅より大きい差は観測されなかった（停止して所有者へ報告）"
    elif d1 < -n_w and d2 < -n_w:
        out["case"], out["conclusion"] = "iv", "chain が速い。停止して所有者へ報告"
    elif d1 > n_w and d2 > n_w and c1 >= MIN_CHAIN_ROUNDS and c2 >= MIN_CHAIN_ROUNDS:
        out["case"] = "ii"
        out["conclusion"] = "チェーン化の後退。内訳の別名規則 DIAG-breakdown.txt を計測前に登録して測る"
    elif (e1 != e2 or (e1 and e2 and (d1 > 0) != (d2 > 0))) and p > n_w:
        out["case"], out["conclusion"] = "iii", "配置の影響。コード原因とは言えない（停止して所有者へ報告）"
    else:
        out["case"], out["conclusion"] = "v", "どの規則にも当てはまらない（停止して所有者へ報告。推定で埋めない）"
    return out


def render(res, label, profile):
    L = [f"# #3611 CPU train 切り分け集計（profile={profile} label={label}）", "",
         "分類であり判定ではない。推定値は含まない。表は run 番号つき（r1..r5 = JSONL の行順）。", "",
         "| 腕 | 代表値 median_s (s) | rho | rho_r (r1..r5) |", "|---|---|---|---|"]
    for a in ARMS:
        rr = " ".join(f"{x:.4f}" for x in res["rho_r"][a])
        L.append(f"| {a} | {res['rep'][a]:.9f} | {res['rho'][a]:.4f} | {rr} |")
    L += ["",
          f"- N（A/A ノイズ幅） = {res['N']:.4f}",
          f"- D1 = rho(ch1) - 1 = {res['D1']:+.4f}（c1 = {res['c1']}/5 round で ch1 > aa0）",
          f"- D2 = rho(rb1)/rho(rb0) - 1 = {res['D2']:+.4f}（c2 = {res['c2']}/5 round で rb1 > rb0）",
          f"- P = |rho(rb0) - 1| = {res['P']:.4f}",
          f"- 分類 = ({res['case']})",
          f"- 併記（B0・#3257 時点のコード）: o1/o0 - 1 = {res['o_ratio']:+.4f}（N = {res['N']:.4f}、D1 = {res['D1']:+.4f}）",
          "", f"結論: {res['conclusion']}"]
    return "\n".join(L) + "\n"


def summarize(diag_path, d, label, profile):
    try:
        with open(diag_path, "rb") as fh:
            raw = fh.read()
        diag_text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Bad(f"DIAG 読取不能: {e}")
    diag_sha = hashlib.sha256(raw).hexdigest()
    check_env(profile, read_env(f"{d}/env_info-diag-{label}.txt"), diag_sha, diag_text)
    try:
        j.check_gate(profile, "cpu", f"{d}/load_gate-diag-{label}-cpu.log")
    except j.Bad as e:
        raise Undet(str(e))
    med, checksums, digests = {}, set(), set()
    for a in ARMS:
        m, c, g = load_arm(d, label, a)
        med[a] = m
        checksums |= c
        digests |= {x for x in g}
    if len(checksums) != 1:
        raise Undet(f"checksum が全 run・全腕で一致しない ({sorted(checksums)})")
    if len(digests) != 1:
        raise Undet(f"PARAM_DIGEST が全 run・全腕で一致しない ({len(digests)} 種)")
    return render(classify(med), label, profile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--profile", required=True, choices=PROFILES)
    a = ap.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9._-]+$", a.label):
        print("入力不正: label", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(summarize(a.diag, a.dir, a.label, a.profile))
    except Bad as e:
        print(f"入力不正: {e}", file=sys.stderr)
        return 2
    except Undet as e:
        print(f"判定不能: {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
