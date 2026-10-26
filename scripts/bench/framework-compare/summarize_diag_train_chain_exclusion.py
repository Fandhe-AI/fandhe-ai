#!/usr/bin/env python3
"""イシュー #3596 r4: 既定 OFF の chain セル差の切り分け診断（DIAG-r4.txt）の集計ツール（fail-closed・stdlib のみ）。

`run_diag_train_chain_exclusion_cuda.sh` が出す生ログを読み、DIAG の [読み方の規則] をそのまま機械適用して
腕ごとの代表値・比・A/A ノイズ幅・分類を出す。これは判定（NON-REGRESSION／REGRESSION）ではなく分類であり、
judge_train_chain_exclusion_ab.py の入力でも置き換えでもない。推定値は出さない。
欠損・型不正・件数不足・不一致は例外にせず終了コード 2（入力不正）または 4（診断不能）で止める（寛容パース禁止）。
入力の読み取り部（JSONL・PARAM_DIGEST・専有ゲートログ）は judge と同じ関数を再利用する（二重定義しない）。

終了コード: 0 = 分類を出力 / 2 = 入力不正 / 4 = 診断不能（専有ゲート不通過・hard 条件不一致・env_info 不一致）。
"""
import argparse
import hashlib
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_train_chain_exclusion_ab as j  # noqa: E402

ROUNDS = 5
ARMS = ("aa0", "aa1", "excl0-rebuild", "excl1", "rv-scopecall", "rv-kindlookup", "rv-graphlock")
RV_ARMS = ("rv-scopecall", "rv-kindlookup", "rv-graphlock")
# 既定 OFF の実行経路の外にある対照腕（DIAG の「ビルド・配置の影響あり」併記の対象）。
OFF_PATH_CONTROLS = ("excl0-rebuild", "rv-graphlock")


class Bad(Exception):
    pass


class Undet(Exception):
    pass


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


def check_env(kv, diag_sha, diag_text):
    if kv.get("profile") != "gb10" or "GB10" not in kv.get("gpu", "") or not kv.get("uname", "").endswith("aarch64"):
        raise Undet("env_info が GB10（aarch64）でない")
    if kv.get("diag_sha256") != diag_sha:
        raise Undet("env_info の diag_sha256 が現 DIAG と一致しない")
    want = {
        "patch_sha256": diag_field(diag_text, r"^  patch_sha256: ([0-9a-f]{64})$", "patch_sha256"),
        "rev_stamp_head_before": diag_field(diag_text, r"^- before = ([0-9a-f]{40})$", "before SHA"),
        "crates_identical_to": diag_field(diag_text, r"^- after_crates_identical_to = ([0-9a-f]{40})$", "after_crates_identical_to"),
    }
    for rv in RV_ARMS:
        want[f"{rv}_sha256"] = diag_field(diag_text, rf"^  {rv}_sha256: ([0-9a-f]{{64}})$", f"{rv}_sha256")
    for k, v in want.items():
        if kv.get(k) != v:
            raise Undet(f"env_info の {k} が DIAG の固定値と一致しない")
    if kv.get("precheck") != "assert" or kv.get("rounds") != str(ROUNDS) or kv.get("tf32") != "off":
        raise Undet("env_info の precheck/rounds/tf32 が規則と不一致")
    for a in ARMS:
        if not re.match(r"^[0-9a-f]{64}$", kv.get(f"bin_sha256_{a}", "")):
            raise Undet(f"env_info の bin_sha256_{a} が欠落または不正")
    if kv["bin_sha256_aa0"] != kv["bin_sha256_aa1"]:
        raise Undet("aa0／aa1 が同一バイナリでない（A/A が成立しない）")
    for a in ("excl1",) + RV_ARMS:
        if kv[f"bin_sha256_{a}"] == kv["bin_sha256_aa0"]:
            raise Undet(f"{a} のバイナリが before と同一（腕が区別できていない）")


def load_arm(d, label, arm):
    """腕の 5 run を (median_s のリスト, checksum 集合, digest 集合) で返す。run 番号は行順 = round。"""
    res = f"{d}/results-diag-{arm}-{label}-cuda-train.jsonl"
    try:
        rows = j.load_rows(res)
    except j.Undet as e:
        raise Undet(str(e))
    except j.Bad as e:
        raise Bad(str(e))
    if len(rows) != ROUNDS:
        raise Undet(f"{res}: run が {len(rows)} 件（{ROUNDS} 件必須）")
    digests = []
    for r in range(1, ROUNDS + 1):
        try:
            digests.append(j.load_digest(f"{d}/digest-diag-{arm}-{label}-cuda-train-r{r}.log"))
        except j.Undet as e:
            raise Undet(str(e))
        except j.Bad as e:
            raise Bad(str(e))
    return [r["median_s"] for r in rows], {r["checksum"] for r in rows}, set(digests)


def classify(med):
    """med: arm -> 5 round の median_s。DIAG の [読み方の規則] を適用し、数値と分類を返す。"""
    rep = {a: statistics.median(med[a]) for a in ARMS}
    rho = {a: rep[a] / rep["aa0"] for a in ARMS}
    rho_r = {a: [med[a][i] / med["aa0"][i] for i in range(ROUNDS)] for a in ARMS}
    n_width = max(abs(rho["aa1"] - 1.0), max(abs(x - 1.0) for x in rho_r["aa1"]))
    delta = rho["excl1"] - 1.0
    out = {"rep": rep, "rho": rho, "rho_r": rho_r, "N": n_width, "delta_after": delta, "arm_class": {}}
    # 対照腕の変動は場合 (i)・(iii) の早期 return でも併記する（DIAG の読み方の規則）
    moved = [a for a in OFF_PATH_CONTROLS if abs(rho[a] - 1.0) > n_width]
    out["controls_moved"] = moved
    if abs(delta) <= n_width:
        out["case"] = "i"
        out["conclusion"] = "差は A/A と同程度。コードに原因は見つからない（停止して所有者へ報告）"
        return out
    if delta < -n_width:
        out["case"] = "iii"
        out["conclusion"] = "after が速い。停止して所有者へ報告"
        return out
    out["case"] = "ii"
    for rv in RV_ARMS:
        faster = sum(1 for i in range(ROUNDS) if rho_r[rv][i] / rho_r["excl1"][i] < 1.0)
        if abs(rho[rv] - 1.0) <= n_width and faster >= 4:
            cls = "原因候補"
        elif abs(rho[rv] / rho["excl1"] - 1.0) <= n_width:
            cls = "無関係"
        else:
            cls = "部分／不明"
        out["arm_class"][rv] = cls
    cand = [rv for rv in RV_ARMS if out["arm_class"][rv] == "原因候補"]
    kl, sc = out["arm_class"]["rv-kindlookup"], out["arm_class"]["rv-scopecall"]
    if kl == "原因候補":
        out["conclusion"] = "resolved_stream_kind の mutex 等の本体コストが原因候補"
    elif sc == "原因候補" and kl == "無関係":
        out["conclusion"] = "呼び出し・スコープ型の扱い自体のコストが原因候補"
    elif not [c for c in cand if c != "rv-graphlock"] and moved:
        out["conclusion"] = "配置・ビルド起因の可能性（コード原因なし。停止して所有者へ報告）"
    else:
        out["conclusion"] = "どの規則にも当てはまらない（停止して所有者へ報告。推定で埋めない）"
    return out


def render(res, label):
    L = [f"# DIAG-r4 集計（label={label}）", "", "分類であり判定ではない。推定値は含まない。", ""]
    L += ["| 腕 | 代表値 median_s (s) | rho | rho_r (r1..r5) |", "|---|---|---|---|"]
    for a in ARMS:
        rr = " ".join(f"{x:.4f}" for x in res["rho_r"][a])
        L.append(f"| {a} | {res['rep'][a]:.9f} | {res['rho'][a]:.4f} | {rr} |")
    L += ["", f"- N（A/A ノイズ幅） = {res['N']:.4f}", f"- Delta_after = rho(excl1) - 1 = {res['delta_after']:+.4f}", f"- 場合 = ({res['case']})"]
    for rv, c in res["arm_class"].items():
        L.append(f"- {rv}: {c}")
    if res.get("controls_moved"):
        L.append("- ビルド・配置の影響あり（併記）: " + ", ".join(res["controls_moved"]))
    L += ["", f"結論: {res['conclusion']}"]
    return "\n".join(L) + "\n"


def summarize(diag_path, d, label):
    try:
        with open(diag_path, "rb") as fh:
            raw = fh.read()
        diag_text = raw.decode("utf-8")
    except OSError as e:
        raise Bad(f"DIAG 読取不能: {e}")
    diag_sha = hashlib.sha256(raw).hexdigest()
    check_env(read_env(f"{d}/env_info-diag-{label}.txt"), diag_sha, diag_text)
    try:
        j.check_gate(f"{d}/load_gate-diag-{label}-cuda.log")
    except j.Undet as e:
        raise Undet(str(e))
    med, checksums, digests = {}, set(), set()
    for a in ARMS:
        m, c, g = load_arm(d, label, a)
        med[a] = m
        checksums |= c
        digests |= g
    if len(checksums) != 1:
        raise Undet(f"checksum が全 run・全腕で一致しない ({sorted(checksums)})")
    if len(digests) != 1:
        raise Undet(f"PARAM_DIGEST が全 run・全腕で一致しない ({len(digests)} 種)")
    return render(classify(med), label)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9._-]+$", a.label):
        print("入力不正: label", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(summarize(a.diag, a.dir, a.label))
    except Bad as e:
        print(f"入力不正: {e}", file=sys.stderr)
        return 2
    except Undet as e:
        print(f"診断不能: {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
