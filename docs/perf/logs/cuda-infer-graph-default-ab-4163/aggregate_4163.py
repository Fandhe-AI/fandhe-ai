#!/usr/bin/env python3
"""イシュー #4163: 推論 Graph 既定化 A/B の集計と判定（python3 標準ライブラリのみ。厳格パース）。

規則の正本は同ディレクトリの RULE.txt（測る前に単独コミットで固定）。run_ab_infer_graph_default_cuda.sh と
run_gates_4163.sh が `results/raw/` へ出力したファイル群を読み、judge-4163.md／judge-4163.tsv を書く。

終了コード: 0 = 判定を書いた（ADOPT／REJECT／UNDETERMINED のいずれも正当な結果）、2 = 入力不正
（欠損・重複・非有限値・形式不正。集計は捏造せず例外で止める。security.md A08）。
使い方:
  python3 -I aggregate_4163.py --dir <raw dir> --rule RULE.txt --label <label> [--out-dir <dir>]
  python3 -I aggregate_4163.py --self-test
  python3 -I aggregate_4163.py --check-template RULE.template.txt
  python3 -I aggregate_4163.py --check-rule RULE.txt        # プレースホルダ 0・各 sha256 が実ファイルと一致
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

ROUNDS = 5
BATCHES = (64, 1024, 4096)
CELLS = (
    [f"infer-fresh-{b}" for b in BATCHES]
    + [f"infer-reuse-{b}" for b in BATCHES]
    + [f"infer-host-{b}" for b in BATCHES]
    + ["train-fresh", "train-reuse"]
)
# bench-fandhe の計測プロトコル所定値（bench-common の WARMUP_ITERS／MEASURE_ITERS = 20／20、train は
# TRAIN_WARMUP = 20・計測 80 step）。off／on 両腕でこの値に揃っていない計測は比較不能として止める。
EXPECTED_FRAMEWORK = "fandhe-ai"
EXPECTED_PROTOCOL = {"infer": (20, 20), "train": (20, 80)}
ATTRIBUTION = {
    "fresh": "stream 変更のみ（capture に届かない）",
    "reuse": "stream 変更のみ（常駐入力の本番経路は capture に届かない）",
    "host": "stream 変更 + capture の合算",
    "train": "stream 変更のみ（学習への波及）",
}
PINNED_KEYS = (
    "runner_sha256",
    "gates_sha256",
    "patch_sha256",
    "assumption_list_sha256",
    "aggregate_sha256",
    "bench_main_sha256",
)
# RULE 内の固定値 → リポジトリ内の実ファイル（--check-rule 用。ルート相対）。
PINNED_FILES = {
    "runner_sha256": "scripts/bench/framework-compare/run_ab_infer_graph_default_cuda.sh",
    "gates_sha256": "docs/perf/logs/cuda-infer-graph-default-ab-4163/run_gates_4163.sh",
    "patch_sha256": "docs/perf/logs/cuda-infer-graph-default-ab-4163/default-on.patch",
    "assumption_list_sha256": "docs/perf/logs/cuda-infer-graph-default-ab-4163/default-off-assumption-tests.txt",
    "aggregate_sha256": "docs/perf/logs/cuda-infer-graph-default-ab-4163/aggregate_4163.py",
    "bench_main_sha256": "scripts/bench/framework-compare/bench-fandhe/src/main.rs",
}
# R3 の 1 回あたりのログ単位（ファイル名の単位名, 集合のキーに使う crate 名）。smedump は facade の
# tests/cpu_sme_gate_bit_dump.rs::dump_cpu_sme_gate_bits を本体の実行から外して単独に流したログ（run_gates_4163.sh）。
R3_UNITS = (("fandhe-ai-backend-cuda", "fandhe-ai-backend-cuda"), ("fandhe-ai", "fandhe-ai"), ("smedump", "fandhe-ai"))
BIT_MISMATCH_RE = re.compile(r"bit.{0,12}(不一致|一致しない|mismatch)|not bit-identical|bits? differ", re.I | re.S)


class Bad(Exception):
    """入力不正（終了コード 2）。"""


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        raise Bad(f"{path}: 読めない（{e.strerror}）")


def rule_pins(text):
    pins = {}
    for ln in text.splitlines():
        m = re.match(r"^  ([a-z0-9_]+): (\S+)$", ln)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


# ---------------------------------------------------------------- 雛形・RULE の検査
def check_template(path):
    t = read_text(path)
    pins = rule_pins(t)
    miss = [k for k in ("fixed-at-utc", "source_commit", "r3_repeats") + PINNED_KEYS if k not in pins and f"{k}:" not in t]
    if "RULE fixed-at-utc:" not in t:
        miss.append("RULE fixed-at-utc")
    if miss:
        raise Bad(f"雛形に必須キーが無い: {miss}")
    if "<PLACEHOLDER" not in t:
        raise Bad("雛形にプレースホルダが無い（雛形が確定済みの RULE になっている）")
    return "template ok"


def check_rule(path, root):
    t = read_text(path)
    if "<PLACEHOLDER" in t:
        raise Bad("RULE にプレースホルダが残っている")
    pins = rule_pins(t)
    if not re.fullmatch(r"[0-9a-f]{40}", pins.get("source_commit", "")):
        raise Bad("source_commit がフル SHA でない")
    if not re.fullmatch(r"[1-9]\d*", pins.get("r3_repeats", "")):
        raise Bad("r3_repeats が正の整数でない")
    for key, rel in PINNED_FILES.items():
        want = pins.get(key, "")
        if not re.fullmatch(r"[0-9a-f]{64}", want):
            raise Bad(f"{key} が 64 桁 hex でない")
        if sha256_file(os.path.join(root, rel)) != want:
            raise Bad(f"{key} が実ファイルの sha256 と一致しない")
    return "rule ok"


# ---------------------------------------------------------------- 計測の読み取り
def load_cell(d, label, arm, cell):
    path = os.path.join(d, f"results-{arm}-{label}-{cell}.jsonl")
    lines = [ln for ln in read_text(path).splitlines()]
    if any(not ln.strip() for ln in lines):
        raise Bad(f"{path}: 空行がある")
    if len(lines) != ROUNDS:
        raise Bad(f"{path}: {len(lines)} 行（各セル・各腕ちょうど {ROUNDS} 件必須。欠損または重複）")
    exp_task = "train" if cell.startswith("train") else "infer"
    exp_mode = {"fresh": "fresh", "reuse": "reuse", "host": "reuse"}[cell.split("-")[1]] if exp_task == "infer" else cell.split("-")[1]
    exp_size = int(cell.split("-")[2]) if exp_task == "infer" else 64
    exp_diag = "host_input_reuse" if cell.startswith("infer-host") else None
    recs = []
    for i, ln in enumerate(lines, 1):
        try:
            o = json.loads(ln)
        except ValueError:
            raise Bad(f"{path}:{i}: JSON でない")
        if not isinstance(o, dict):
            raise Bad(f"{path}:{i}: object でない")
        for k in ("framework", "task", "device", "size", "median_s", "checksum", "warmup", "iters", "mode"):
            if k not in o:
                raise Bad(f"{path}:{i}: キー {k} が無い")
        if o["task"] != exp_task or o["device"] != "cuda" or o["mode"] != exp_mode or o["size"] != exp_size:
            raise Bad(f"{path}:{i}: task/device/mode/size がセルと不整合")
        if o.get("diag") != exp_diag:
            raise Bad(f"{path}:{i}: 診断タグ不整合（期待 {exp_diag!r}）")
        for k in ("median_s", "checksum"):
            v = o[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise Bad(f"{path}:{i}: {k} が有限数でない")
        if o["median_s"] <= 0:
            raise Bad(f"{path}:{i}: median_s が正でない")
        recs.append(o)
    if len({(r["warmup"], r["iters"], r["framework"]) for r in recs}) != 1:
        raise Bad(f"{path}: warmup/iters/framework が run 間で不一致")
    want = EXPECTED_PROTOCOL[exp_task]
    got = (recs[0]["warmup"], recs[0]["iters"])
    if recs[0]["framework"] != EXPECTED_FRAMEWORK or got != want:
        raise Bad(f"{path}: framework/warmup/iters が事前登録値と不一致"
                  f"（期待 {EXPECTED_FRAMEWORK}/{want[0]}/{want[1]}、実 {recs[0]['framework']}/{got[0]}/{got[1]}）")
    return recs


def expected_order():
    """ランナーと同じ規則の期待起動順（round 1..5。基準セル列を 3*(r-1) mod n だけ巡回し、奇数 round は off→on、偶数 round は on→off）。"""
    n = len(CELLS)
    out = []
    for r in range(1, ROUNDS + 1):
        start = (3 * (r - 1)) % n
        first, second = ("off", "on") if r % 2 == 1 else ("on", "off")
        for k in range(n):
            cell = CELLS[(start + k) % n]
            out.append(f"round={r} cell={cell} arm={first}")
            out.append(f"round={r} cell={cell} arm={second}")
    return out


def parse_order(d, label):
    """order-<label>.txt を期待起動順（expected_order）と行ごとに完全照合する。"""
    txt = read_text(os.path.join(d, f"order-{label}.txt")).splitlines()
    exp = expected_order()
    if len(txt) != len(exp):
        raise Bad(f"order-{label}.txt: 行数 {len(txt)} が期待値 {len(exp)} と不一致")
    for i, (got, want) in enumerate(zip(txt, exp), 1):
        if got != want:
            raise Bad(f"order-{label}.txt:{i}: 起動順が登録済みの規則と不一致（期待 {want!r}、実際 {got!r}）")


# ---------------------------------------------------------------- ゲートログ
def log_rc(path):
    """ログ末尾の rc= 行。ログ無し・rc 行無しは None（未実行）。"""
    if not os.path.exists(path):
        return None, ""
    t = read_text(path)
    ms = re.findall(r"^rc=(\d+)$", t, re.M)
    return (int(ms[-1]) if ms else None), t


def failed_detail(text):
    names = re.findall(r"^test (\S+) \.\.\. FAILED", text, re.M)
    locs = re.findall(r"panicked at (\S+?:\d+)", text)
    variant = "DeviceContextCaptureInProgress" if "DeviceContextCaptureInProgress" in text else ""
    return names, locs, variant


def _load_assumptions(rule_path):
    assumptions = set()
    alist = os.path.join(os.path.dirname(os.path.abspath(rule_path)), "default-off-assumption-tests.txt")
    if os.path.exists(alist):
        assumptions = {ln.strip() for ln in read_text(alist).splitlines() if ln.strip() and not ln.startswith("#")}
    return assumptions


def _load_reps(rule_path):
    """RULE の `  r3_repeats: <正の整数>`。無い・形式不正は None（R3 を評価せず UNDETERMINED）。"""
    v = rule_pins(read_text(rule_path)).get("r3_repeats", "")
    return int(v) if re.fullmatch(r"[1-9]\d*", v) else None


def eval_gates(d, label, assumptions, reps):
    """R0〜R3 を評価。(undet 理由, reject 理由, 表の行, R1b 詳細, R3 差分) を返す。"""
    undet, reject, rows, r1b_detail, r3_rows = [], [], [], [], []
    p = lambda n: os.path.join(d, f"{n}-{label}.log")
    if os.path.exists(os.path.join(d, f"pregate-stop-{label}.txt")):
        undet.append("前提ゲート不合格で停止した記録（pregate-stop）がある")
    # R0
    for n in ("r0-linear", "r0-graph"):
        rc, t = log_rc(p(n))
        if rc is None:
            undet.append(f"{n} 未実行"); rows.append((n, "未実行")); continue
        ok = rc == 0 and re.search(r"^test result: ok\. [1-9]\d* passed", t, re.M) is not None
        rows.append((n, "合格" if ok else "不合格"))
        if not ok:
            undet.append(f"{n} が非後退でない（rc={rc}）")
    # R1(a)
    rc, t = log_rc(p("r1a-capture"))
    if rc is None:
        undet.append("r1a 未実行"); rows.append(("r1a", "未実行"))
    else:
        # --nocapture では "test <name> ... INFER_CAPTURE_R1: PASS" のように libtest 接頭辞付きで出るため行頭に限定しない。
        passes = len(re.findall(r"INFER_CAPTURE_R1: PASS", t))
        skips = len(re.findall(r"INFER_CAPTURE_R1: SKIP", t))
        m = re.search(r"^test result: ok\. (\d+) passed", t, re.M)
        ok = rc == 0 and skips == 0 and m is not None and int(m.group(1)) == passes and passes > 0
        rows.append(("r1a", f"{'合格' if ok else '不合格'}（PASS={passes} SKIP={skips}）"))
        if not ok:
            (reject if (rc != 0 and BIT_MISMATCH_RE.search(t)) else undet).append(f"r1a 不合格（rc={rc} PASS={passes} SKIP={skips}）")
    # R1(b)
    okn = 0
    for i in range(1, ROUNDS + 1):
        rc, t = log_rc(p(f"r1b-run{i}"))
        if rc is None:
            undet.append(f"r1b-run{i} 未実行"); continue
        if rc == 0 and re.search(r"^test result: ok\. [1-9]\d* passed", t, re.M):
            okn += 1
            continue
        names, locs, variant = failed_detail(t)
        r1b_detail.append((i, names, locs, variant))
        if BIT_MISMATCH_RE.search(t):
            reject.append(f"r1b-run{i}: bit 不一致")
        else:
            undet.append(f"r1b-run{i}: エラー型の失敗（{variant or 'その他'}）")
    rows.append(("r1b", f"{okn}/{ROUNDS} 合格"))
    # R2
    for b in BATCHES:
        dumps = {}
        for mode in ("off", "on"):
            rc, t = log_rc(p(f"r2-{mode}-{b}"))
            if rc is None:
                undet.append(f"r2-{mode}-{b} 未実行"); break
            # --nocapture では最初の println! が libtest の "test <name> ... " と同一行に出るため、
            # 接頭辞を除去してから添字・値の形式を厳格に照合する（先頭要素の欠落防止）
            lines = [m.group(1) for m in re.finditer(
                r"^(?:test \S+ \.\.\. )?(out\[\d+\]\.bits=0x[0-9a-fA-F]+)[ \t]*$", t, re.M)]
            idx_ok = [re.match(r"out\[(\d+)\]", x).group(1) for x in lines] == [str(k) for k in range(len(lines))]
            if rc != 0 and BIT_MISMATCH_RE.search(t):
                # dump() は初回出力と replay 出力が不一致なら assertion で終了する（rc≠0・ダンプ 0 行）。
                # RULE は R2 の bit 不一致を REJECT と定めるため判定不能にしない
                reject.append(f"r2-{mode}-{b}: 初回出力と replay 出力の bit 不一致（rc={rc}）"); break
            if rc != 0 or len(lines) != b * 10 or not idx_ok:
                undet.append(f"r2-{mode}-{b}: rc={rc} 行数={len(lines)}（期待 {b * 10}）"); break
            dumps[mode] = lines
        if len(dumps) == 2:
            same = dumps["off"] == dumps["on"]
            rows.append((f"r2-{b}", "全行一致" if same else "不一致"))
            if not same:
                reject.append(f"r2-{b}: off／on のダンプが不一致")
        else:
            rows.append((f"r2-{b}", "未完"))
    # R3（N 回反復。#4163 所有者判断 2026-11-11）
    if reps is None:
        undet.append("r3: RULE に r3_repeats（正の整数）が無い -> R3 を評価しない")
        rows.append(("r3", "未評価（r3_repeats なし）"))
        return undet, reject, rows, r1b_detail, r3_rows, 0
    runs = {"C": [], "A": []}  # 腕 -> 回ごとの {(crate, bin, name): status}
    for arm in ("C", "A"):
        for k in range(1, reps + 1):
            res, complete = {}, True
            for unit, crate in R3_UNITS:
                rc, t = log_rc(os.path.join(d, f"r3-{arm}-{unit}-rep{k}-{label}.log"))
                tsv = os.path.join(d, f"r3-{arm}-{unit}-rep{k}-{label}.tsv")
                if rc is None or not os.path.exists(tsv):
                    undet.append(f"r3 腕{arm} {unit} rep{k} 未実行"); complete = False; continue
                if not _r3_unit(undet, arm, unit, k, rc, t, tsv, crate, res):
                    complete = False
            if complete:
                runs[arm].append(res)
    blockers, expected, unstable = [], [], []
    if len(runs["C"]) == reps and len(runs["A"]) == reps:
        keysets = {arm: [frozenset(r) for r in runs[arm]] for arm in runs}
        if any(ks != keysets[arm][0] for arm in keysets for ks in keysets[arm]):
            undet.append("r3: テスト集合が回の間で一致しない")
        else:
            kc, ka = keysets["C"][0], keysets["A"][0]
            for key in sorted(kc | ka):
                tag = f"{key[1]}::{key[2]}"
                sc = [r[key] for r in runs["C"]] if key in kc else None
                sa = [r[key] for r in runs["A"]] if key in ka else None
                if sa is None:
                    blockers.append(key)
                    r3_rows.append((key[0], tag, _r3_summary(sc), "欠落", "阻害"))
                    continue
                if sc is None:
                    undet.append(f"r3: 腕 A にだけあるテスト {tag}（テスト集合が腕間で一致しない）")
                    continue
                if len(set(sc)) > 1 or len(set(sa)) > 1:
                    if "FAILED" in sc + sa:
                        unstable.append(key)
                        r3_rows.append((key[0], tag, _r3_summary(sc), _r3_summary(sa), "不安定（判定外）"))
                    continue
                if sa[0] == "FAILED" and sc[0] == "ok":
                    (expected if tag in assumptions else blockers).append(key)
                    r3_rows.append((key[0], tag, _r3_summary(sc), _r3_summary(sa), "既定OFF前提" if tag in assumptions else "阻害"))
                elif sa[0] == "FAILED" and sc[0] != "FAILED":
                    # 腕 C で毎回成功していないため阻害に数えない（RULE: 阻害は「A で毎回失敗・C で毎回成功」だけ）
                    r3_rows.append((key[0], tag, _r3_summary(sc), _r3_summary(sa), "判定外（腕 C で成功していない）"))
    rows.append(("r3", f"N={reps}・阻害 {len(blockers)} 件・既定OFF前提 {len(expected)} 件・不安定 {len(unstable)} 件"))
    if blockers:
        bit = False
        for k in range(1, reps + 1):
            for unit, _ in R3_UNITS:
                _, t = log_rc(os.path.join(d, f"r3-A-{unit}-rep{k}-{label}.log"))
                if BIT_MISMATCH_RE.search(t):
                    bit = True
        (reject if bit else undet).append(f"r3: 阻害 {len(blockers)} 件")
    rcb, _ = log_rc(os.path.join(d, f"r3-B-fandhe-ai-backend-cuda-rep1-{label}.log"))
    rows.append(("r3-B（補助・判定外）", "未実行" if rcb is None else f"rc={rcb}"))
    return undet, reject, rows, r1b_detail, r3_rows, len(blockers)


def _r3_summary(sts):
    """回ごとの status 列を「ok×3」「ok×2/FAILED×1」の形へ（出現順）。"""
    out = []
    for st in dict.fromkeys(sts):
        out.append(f"{st}×{sts.count(st)}")
    return "/".join(out)


def _r3_unit(undet, arm, unit, k, rc, t, tsv, crate, res):
    """R3 の 1 ログ（1 回・1 単位）の TSV を res へ取り込み、完了性を検査する。完了なら True。"""
    cnt = {"ok": 0, "FAILED": 0, "ignored": 0}
    n, ok = 0, True
    where = f"r3 腕{arm} {unit} rep{k}"
    for ln in read_text(tsv).splitlines():
        f = ln.split("\t")
        if len(f) != 3:
            raise Bad(f"{tsv}: 形式不正")
        if (crate, f[0], f[1]) in res:
            raise Bad(f"{tsv}: 重複 {f[1]}")
        if f[2] not in cnt:
            undet.append(f"{where}: 未知のステータス {f[2]!r}（{f[1]}）"); ok = False
        else:
            cnt[f[2]] += 1
        res[(crate, f[0], f[1])] = f[2]
        n += 1
    if n == 0:
        undet.append(f"{where}: テスト結果が 0 件"); return False
    # 完了記録: Running／Doc-tests の各節に "test result:" 行があり、各件数の合計が TSV と一致すること。
    # Running 節の結果行はちょうど 1 本。Doc-tests 節は edition 2024 の merged doctests が merged と standalone の
    # 2 本を出すため、1 本以上を認める（所有者判断 2026-11-11 (b)）。
    # harness=false のバイナリ（例: device_param_store_resident_no_flat_grad_alloc）は libtest の結果行を出さない。
    # 結果行の無い節は、cargo が失敗対象として報告していない場合に限り完了とみなす（異常終了は失敗対象に載る）。
    pieces = re.split(r"^\s*(Running|Doc-tests) ", t, flags=re.M)[1:]
    sections = list(zip(pieces[0::2], pieces[1::2]))
    sums = re.findall(r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", t, re.M)
    failed_targets = " ".join(re.findall(r"^(?:error: test failed.*|\s+`.*`)$", t, re.M))
    bare, multi, nres = [], [], 0
    for kind, part in sections:
        c = len(re.findall(r"^test result: ", part, re.M))
        nres += c
        if c == 0:
            m = re.match(r"unittests\s+\S+|(\S+)", part)
            bare.append(re.sub(r"\.rs$", "", os.path.basename(m.group(1) or "lib")) if m else "")
        elif c > 1 and kind != "Doc-tests":
            multi.append(part.split("\n", 1)[0].strip())
    crashed = [b for b in bare if b in ("", "lib") or re.search(r"--(?:test|bin)\s+" + re.escape(b) + r"(?:[`\s]|$)", failed_targets)]
    if crashed:
        undet.append(f"{where}: 結果行の無いテストバイナリが失敗対象として報告されている／特定できない（{crashed}）")
        return False
    if multi:
        undet.append(f"{where}: 結果行が複数あるテストバイナリ（Doc-tests 以外。出力の混入）: {multi}")
        return False
    if not sections or nres != len(sums):
        undet.append(f"{where}: 完了していないテストバイナリがある（節 {len(sections)}・結果行 {len(sums)}）")
        return False
    tot = [sum(int(x[i]) for x in sums) for i in range(3)]
    if tot != [cnt["ok"], cnt["FAILED"], cnt["ignored"]]:
        undet.append(f"{where}: 抽出結果 {[cnt['ok'], cnt['FAILED'], cnt['ignored']]} が集計行 {tot} と不一致")
        return False
    # 非ゼロ終了は「テスト失敗（rc=101 かつ FAILED > 0）」でしか説明を認めない。
    if rc != 0 and not (rc == 101 and cnt["FAILED"] > 0):
        undet.append(f"{where}: 説明できない非ゼロ終了（rc={rc} FAILED={cnt['FAILED']}）"); return False
    if rc == 0 and cnt["FAILED"] > 0:
        undet.append(f"{where}: rc=0 だが FAILED がある"); return False
    return ok


# ---------------------------------------------------------------- 判定
def judge(d, rule_path, label):
    # 前提ゲート不合格で停止した計測は order／load_gate／results を作らずに終わる。未生成の計測データを要求する前に
    # ゲートログ（eval_gates）だけを分類する。R1・R2・R3 の bit 不一致は RULE が REJECT と定めるため REJECT、
    # エラー型の失敗・未実行などは UNDETERMINED。停止そのものは数値不一致の結果であり、REJECT を覆す要因にしない。
    stop = os.path.join(d, f"pregate-stop-{label}.txt")
    if os.path.exists(stop):
        detail = [ln.strip() for ln in read_text(stop).splitlines() if ln.strip()]
        g_undet, g_reject, g_rows, r1b_detail, r3_rows, _ = eval_gates(d, label, _load_assumptions(rule_path), _load_reps(rule_path))
        undet = g_undet + [f"pregate-stop: {x}" for x in detail[:20]]
        verdict = "REJECT" if g_reject else "UNDETERMINED"
        return verdict, undet, g_reject, [], [("pregate-stop", "不合格（計測前に停止）")] + g_rows, r1b_detail, r3_rows
    rule_text = read_text(rule_path)
    pins = rule_pins(rule_text)
    env = {}
    for ln in read_text(os.path.join(d, f"env_info-{label}.txt")).splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            env[k] = v
    undet, reject = [], []
    if hashlib.sha256(rule_text.encode("utf-8")).hexdigest() != env.get("rule_sha256"):
        undet.append("RULE.txt の sha256 が env_info と不一致")
    for k in PINNED_KEYS:
        if pins.get(k) is None or env.get(k) != pins.get(k):
            undet.append(f"{k} が RULE と env_info で不一致")
    # 計測対象ソースは RULE の source_commit。RULE.txt を単独コミットすると HEAD は変わるため HEAD 完全一致は要求せず、
    # ランナーが「source_commit が HEAD の祖先で、RULE.txt 以外の差分が 0」を検証した記録（source_diff_clean=1）を突合する。
    if env.get("source_commit") != pins.get("source_commit") or env.get("source_diff_clean") != "1":
        undet.append("計測対象ソースが source_commit と同一であることの記録がない（source_commit／source_diff_clean）")
    if "GB10" not in env.get("gpu", "") or not env.get("uname", "").endswith("aarch64"):
        undet.append("GB10（aarch64）の計測でない")
    # 専有ゲート
    gl = read_text(os.path.join(d, f"load_gate-{label}-cuda.log")).splitlines()
    gl = [x for x in gl if x.startswith("round")]
    if len(gl) != ROUNDS or any("gate=pass" not in x for x in gl):
        undet.append("専有ゲートが全 round で通過していない")
    if any("gate=fail" in x for x in gl):
        # ランナーは専有ゲート不通過で order／results を書き切らずに終了する。RULE は UNDETERMINED と定めるため、
        # 欠けた計測データを要求せず（Bad にせず）、前提ゲートの分類だけを添えて UNDETERMINED を返す。
        g_undet, g_reject, g_rows, r1b_detail, r3_rows, _ = eval_gates(d, label, _load_assumptions(rule_path), _load_reps(rule_path))
        return "UNDETERMINED", undet + g_undet, g_reject, [], g_rows, r1b_detail, r3_rows
    parse_order(d, label)

    data = {}
    for cell in CELLS:
        data[cell] = {arm: load_cell(d, label, arm, cell) for arm in ("off", "on")}
    cells_out = []
    ratio_fail, rounds_ok_host = [], False
    for cell in CELLS:
        off, on = data[cell]["off"], data[cell]["on"]
        if (off[0]["framework"], off[0]["warmup"], off[0]["iters"]) != (on[0]["framework"], on[0]["warmup"], on[0]["iters"]):
            raise Bad(f"{cell}: off／on で framework/warmup/iters が不一致（比較不能）")
        sums = {r["checksum"] for r in off + on}
        csum_ok = len(sums) == 1
        if not csum_ok:
            reject.append(f"{cell}: checksum が off／on の run 間で不一致")
        mo = statistics.median(r["median_s"] for r in off)
        mn = statistics.median(r["median_s"] for r in on)
        ratio = mn / mo
        ratio_r = [on[i]["median_s"] / off[i]["median_s"] for i in range(ROUNDS)]
        consistent = all(x < 1.0 for x in ratio_r)
        if ratio > 1.0:
            ratio_fail.append(cell)
        if cell.startswith("infer-host") and consistent:
            rounds_ok_host = True
        kind = "train" if cell.startswith("train") else cell.split("-")[1]
        cells_out.append((cell, mo, mn, ratio, ratio_r, consistent, csum_ok, ATTRIBUTION[kind]))
    if ratio_fail:
        reject.append("ratio > 1.00: " + ", ".join(ratio_fail))

    assumptions = _load_assumptions(rule_path)
    g_undet, g_reject, g_rows, r1b_detail, r3_rows, nblock = eval_gates(d, label, assumptions, _load_reps(rule_path))
    undet += g_undet
    reject += g_reject
    if undet:
        verdict = "UNDETERMINED"
    elif reject:
        verdict = "REJECT"
    elif nblock == 0 and rounds_ok_host:
        verdict = "ADOPT"
    else:
        verdict = "UNDETERMINED"
        undet.append("ADOPT 条件（reuse-host の少なくとも 1 セルで 5/5 round の ratio_r < 1.00）を満たさない")
    return verdict, undet, reject, cells_out, g_rows, r1b_detail, r3_rows


def render(verdict, undet, reject, cells_out, g_rows, r1b_detail, r3_rows, label):
    md = [f"# 判定 #4163（label: {label}）", "", f"**verdict: {verdict}**", ""]
    md.append("INFER_GRAPH_DEFAULT_ENABLED は本イシューでは変えない（反転は #4167・所有者承認）。")
    md += ["", "## undetermined 要因"] + ([f"- {x}" for x in undet] or ["- なし"])
    md += ["", "## reject 要因"] + ([f"- {x}" for x in reject] or ["- なし"])
    md += ["", "## セル", "", "| セル | off 中央値 [s] | on 中央値 [s] | ratio | ratio_r (round 順) | 5/5 一貫 | checksum 一致 | 帰属 |", "|---|---|---|---|---|---|---|---|"]
    tsv = ["cell\toff_median_s\ton_median_s\tratio\tconsistent\tchecksum_ok"]
    for cell, mo, mn, ratio, rr, cons, cs, attr in cells_out:
        md.append(f"| {cell} | {mo:.9f} | {mn:.9f} | {ratio:.4f} | {', '.join(f'{x:.4f}' for x in rr)} | {'はい' if cons else 'いいえ'} | {'はい' if cs else 'いいえ'} | {attr} |")
        tsv.append(f"{cell}\t{mo:.9f}\t{mn:.9f}\t{ratio:.4f}\t{int(cons)}\t{int(cs)}")
    md += ["", "## ノイズ帯の記録（判定外。NB-RANGE5 を名前で参照）", "", "| セル | ratio_r の最大-最小 |", "|---|---|"]
    for cell, _, _, _, rr, *_ in cells_out:
        md.append(f"| {cell} | {max(rr) - min(rr):.4f} |")
    md += ["", "## 前提ゲート", "", "| ゲート | 結果 |", "|---|---|"] + [f"| {n} | {r} |" for n, r in g_rows]
    md += ["", "## R1(b) の失敗詳細"]
    md += [f"- run{i}: テスト={names} 位置={locs} variant={var or '-'}" for i, names, locs, var in r1b_detail] or ["- 失敗なし"]
    md += ["", "## R3 の分類（各腕 N 回。阻害 = 腕 A で毎回失敗・腕 C で毎回成功。不安定は記録のみで判定に使わない）", "",
           "| crate | テスト | 腕C（回ごとの結果） | 腕A（回ごとの結果） | 分類 |", "|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {dd} | {e} |" for a, b, c, dd, e in r3_rows] or ["| - | - | - | - | 差分なし |"]
    return "\n".join(md) + "\n", "\n".join(tsv) + "\n"


# ---------------------------------------------------------------- self-test
def _w(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def make_fixture(d, label="t", ratios=None, **opt):
    """合成 fixture を d に作る。opt で異常を注入する。"""
    ratios = ratios or {}
    pins = {k: (str(i) * 64)[:64] for i, k in enumerate(PINNED_KEYS)}
    tree = "a" * 40
    rule = "RULE fixed-at-utc: 2026-01-01T00:00:00Z\n" + f"  source_commit: {tree}\n" + "".join(f"  {k}: {v}\n" for k, v in pins.items())
    if not opt.get("no_reps"):
        rule += f"  r3_repeats: {opt.get('reps', 3)}\n"
    rpath = os.path.join(d, "RULE.txt")
    _w(rpath, rule)
    _w(os.path.join(d, "default-off-assumption-tests.txt"), "lib::graph::tests::infer_graph_default_is_off\n")
    env = {"rule_sha256": hashlib.sha256(rule.encode()).hexdigest(), "source_commit": tree, "source_diff_clean": "1", "gpu": "NVIDIA GB10", "uname": "Linux 6 aarch64"}
    env.update(pins)
    _w(os.path.join(d, f"env_info-{label}.txt"), "".join(f"{k}={v}\n" for k, v in env.items()))
    _w(os.path.join(d, f"load_gate-{label}-cuda.log"), "series=official\n" + "".join(
        f"round{r} dev=cuda gate={'fail' if opt.get('gate_fail') and r == 3 else 'pass'}\n" for r in range(1, 6)))
    order = expected_order()
    _w(os.path.join(d, f"order-{label}.txt"), "\n".join(order) + "\n")
    for c in CELLS:
        task = "train" if c.startswith("train") else "infer"
        mode = c.split("-")[1] if task == "train" else {"fresh": "fresh", "reuse": "reuse", "host": "reuse"}[c.split("-")[1]]
        size = 64 if task == "train" else int(c.split("-")[2])
        for a in ("off", "on"):
            lines = []
            for r in range(ROUNDS):
                base = 0.01 + 0.0001 * r
                ratio = ratios.get(c, 0.95)
                med = base * (ratio if a == "on" else 1.0)
                cs = 1.5
                if opt.get("csum_bad") == c and a == "on" and r == 2:
                    cs = 1.6
                o = {"framework": "fandhe-ai", "version": "0.14.0", "task": task, "device": "cuda", "size": size,
                     "median_s": med, "checksum": cs, "warmup": 20, "iters": 80 if task == "train" else 20, "mode": mode}
                if opt.get("proto_skew") == c and a == "on":
                    o["iters"] = 1
                if opt.get("fw_skew") == c and a == "on":
                    o["framework"] = "other"
                if c.startswith("infer-host"):
                    o["diag"] = "host_input_reuse"
                lines.append(json.dumps(o))
            if opt.get("dup") == c and a == "off":
                lines.append(lines[0])
            if opt.get("missing") == c and a == "on":
                lines = lines[:4]
            _w(os.path.join(d, f"results-{a}-{label}-{c}.jsonl"), "\n".join(lines) + "\n")
    ok_t = "test result: ok. 2 passed; 0 failed; 0 ignored\nrc=0\n"
    for n in ("r0-linear", "r0-graph"):
        _w(os.path.join(d, f"{n}-{label}.log"), ok_t)
    if not opt.get("no_r1a"):
        _w(os.path.join(d, f"r1a-capture-{label}.log"), "INFER_CAPTURE_R1: PASS a\nINFER_CAPTURE_R1: PASS b\ntest result: ok. 2 passed; 0 failed\nrc=0\n")
    for i in range(1, ROUNDS + 1):
        if opt.get("r1b_err") and i == 2:
            _w(os.path.join(d, f"r1b-run{i}-{label}.log"), "test x ... FAILED\nthread panicked at crates/a.rs:10:5\nDeviceContextCaptureInProgress\nrc=101\n")
        else:
            _w(os.path.join(d, f"r1b-run{i}-{label}.log"), ok_t)
    for b in BATCHES:
        for m in ("off", "on"):
            body = "".join(f"out[{j}].bits=0x{j:08x}\n" for j in range(b * 10))
            body = "test predict_resident_bit_dump ... " + body  # libtest 接頭辞が先頭行に付く実出力を再現
            if opt.get("r2_diff") and b == 64 and m == "on":
                body = body.replace("out[3].bits=0x00000003", "out[3].bits=0x00000004")
            if opt.get("r2_bitfail") and b == 64 and m == "on":
                _w(os.path.join(d, f"r2-{m}-{b}-{label}.log"), "test x ... FAILED\nthread panicked at a.rs:1:1\n出力が bit 一致しない\nrc=101\n"); continue
            if opt.get("r2_err") and b == 64 and m == "on":
                _w(os.path.join(d, f"r2-{m}-{b}-{label}.log"), "thread panicked at a.rs:1:1\nDeviceContextCaptureInProgress\nrc=101\n"); continue
            _w(os.path.join(d, f"r2-{m}-{b}-{label}.log"), body + "rc=0\n")
    reps = opt.get("reps", 3)
    for arm in ("C", "A", "B"):
        for k in range(1, (1 if arm == "B" else reps) + 1):
            for unit, _ in R3_UNITS:
                if unit == "smedump":
                    if arm == "B" or (opt.get("smedump_missing") and arm == "A" and k == 2):
                        continue
                    rows = [("cpu_sme_gate_bit_dump", "dump_cpu_sme_gate_bits", "ok")]
                    head = "     Running tests/cpu_sme_gate_bit_dump.rs (x)\n"
                else:
                    rows = [("lib", "graph::tests::infer_graph_default_is_off", "ok"), ("t", "other", "ok")]
                    head = "     Running unittests src/lib.rs (x)\n"
                    if arm == "A" and unit == "fandhe-ai-backend-cuda":
                        rows[0] = ("lib", "graph::tests::infer_graph_default_is_off", "FAILED")
                        if opt.get("r3_block") or (opt.get("r3_flaky") and k == 2):
                            rows[1] = ("t", "other", "FAILED")
                    if opt.get("r3_c_flaky") and arm == "C" and unit == "fandhe-ai-backend-cuda" and k == 1:
                        rows[1] = ("t", "other", "FAILED")
                _w(os.path.join(d, f"r3-{arm}-{unit}-rep{k}-{label}.tsv"), "".join("\t".join(r) + "\n" for r in rows))
                nfail = sum(1 for r in rows if r[2] == "FAILED")
                body = head + "".join(f"test {r[1]} ... {r[2]}\n" for r in rows)
                body += f"test result: {'FAILED' if nfail else 'ok'}. {len(rows) - nfail} passed; {nfail} failed; 0 ignored\n"
                if unit == "fandhe-ai" and opt.get("doctest_multi"):
                    # edition 2024 の merged doctests: 1 つの Doc-tests 節に結果行が 2 本（merged と standalone）
                    body += ("   Doc-tests fandhe_ai\n\nrunning 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored\n"
                             "\nrunning 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored\n")
                if unit == "fandhe-ai-backend-cuda" and opt.get("running_multi") and arm == "C" and k == 1:
                    body += "test result: ok. 0 passed; 0 failed; 0 ignored\n"
                _w(os.path.join(d, f"r3-{arm}-{unit}-rep{k}-{label}.log"), body + f"rc={101 if nfail else 0}\n")
    # harness=false バイナリ（結果行なし）を腕 C／A の fandhe-ai ログへ追加する（全回）
    if opt.get("harnessless") or opt.get("harnessless_crash"):
        for arm in ("C", "A"):
            for k in range(1, reps + 1):
                lp = os.path.join(d, f"r3-{arm}-fandhe-ai-rep{k}-{label}.log")
                body = open(lp, encoding="utf-8").read().replace("rc=0\n", "").replace("rc=101\n", "")
                body += "     Running tests/no_flat.rs (target/x)\nok\n"
                rc = 0
                if opt.get("harnessless_crash") and arm == "C" and k == 1:
                    body += "error: test failed, to rerun pass `-p fandhe-ai --test no_flat`\n"
                    rc = 101
                _w(lp, body + f"rc={rc}\n")
    # 異常注入（R3 完了性・起動順）
    cl = os.path.join(d, f"r3-C-fandhe-ai-rep1-{label}.log")
    if opt.get("r3_trunc"):
        _w(cl, "     Running unittests src/lib.rs (x)\ntest graph::tests::infer_graph_default_is_off ... ok\nrc=101\n")
    if opt.get("r3_rc"):
        _w(cl, open(cl, encoding="utf-8").read().replace("rc=0", "rc=137"))
    if opt.get("r3_status"):
        tp = os.path.join(d, f"r3-C-fandhe-ai-rep1-{label}.tsv")
        _w(tp, open(tp, encoding="utf-8").read().replace("\tok\n", "\tweird\n", 1))
    if opt.get("order_swap"):
        op = os.path.join(d, f"order-{label}.txt")
        ls = open(op, encoding="utf-8").read().splitlines()
        ls[0], ls[1] = ls[1], ls[0]
        _w(op, "\n".join(ls) + "\n")
    return rpath


def self_test():
    def run(**kw):
        with tempfile.TemporaryDirectory() as d:
            ratios = kw.pop("ratios", None)
            r = make_fixture(d, ratios=ratios, **kw)
            # 列挙名を "lib::graph::tests::..." に合わせるため tsv の bin を lib にしてある
            return judge(d, r, "t")[0:3]

    v, u, rj = run()
    assert v == "ADOPT", (v, u, rj)
    v, u, rj = run(ratios={"train-fresh": 1.02})
    assert v == "REJECT", (v, u, rj)
    v, u, rj = run(csum_bad="infer-fresh-64")
    assert v == "REJECT", (v, u, rj)
    v, u, rj = run(r2_diff=True)
    assert v == "REJECT", (v, u, rj)
    v, u, rj = run(r2_bitfail=True)
    assert v == "REJECT" and any("r2-on-64" in x for x in rj), (v, u, rj)
    v, u, rj = run(r2_err=True)
    assert v == "UNDETERMINED" and not rj, (v, u, rj)
    v, u, rj = run(gate_fail=True)
    assert v == "UNDETERMINED", (v, u, rj)
    # 前提ゲートで停止（計測データ無し）でも Bad で落ちず UNDETERMINED を出す
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "RULE.txt")
        _w(rp, "RULE\n")
        _w(os.path.join(d, "pregate-stop-t.txt"), "r0 fail\n")
        res = judge(d, rp, "t")
        assert res[0] == "UNDETERMINED" and any("r0 fail" in x for x in res[1]), res
        render(*res, "t")
    # pregate-stop でも bit 不一致は REJECT、エラー型の失敗は UNDETERMINED（ゲートログを分類する）
    def run_stop(**kw):
        with tempfile.TemporaryDirectory() as d:
            r = make_fixture(d, **kw)
            _w(os.path.join(d, "pregate-stop-t.txt"), "gate r1b-run1 FAILED rc=101\n")
            res = judge(d, r, "t")
            render(*res, "t")
            return res
    bitlog = "test x ... FAILED\nthread panicked at crates/a.rs:10:5\n出力が bit 一致しない\nrc=101\n"
    with tempfile.TemporaryDirectory() as d:
        r = make_fixture(d)
        _w(os.path.join(d, "r1b-run1-t.log"), bitlog)
        _w(os.path.join(d, "pregate-stop-t.txt"), "gate r1b-run1 FAILED rc=101\n")
        res = judge(d, r, "t")
        assert res[0] == "REJECT" and res[5] and any("r1b-run1" in x for x in res[2]), res
    res = run_stop(r1b_err=True)
    assert res[0] == "UNDETERMINED" and res[5], res
    res = run_stop(r2_diff=True)
    assert res[0] == "REJECT", res
    # 専有ゲート不通過で order／results が未完でも INPUT ERROR にせず UNDETERMINED
    with tempfile.TemporaryDirectory() as d:
        r = make_fixture(d, gate_fail=True)
        os.remove(os.path.join(d, "order-t.txt"))
        _w(os.path.join(d, "results-off-t-infer-fresh-64.jsonl"), "")
        res = judge(d, r, "t")
        assert res[0] == "UNDETERMINED", res
        render(*res, "t")
    # harness=false バイナリ（結果行なし）は正常、失敗対象に載れば UNDETERMINED
    v, u, rj = run(harnessless=True)
    assert v == "ADOPT", (v, u, rj)
    v, u, rj = run(harnessless_crash=True)
    assert v == "UNDETERMINED", (v, u, rj)
    # 実テストの assertion 文言「bit 一致しない」を数値不一致として分類する
    assert BIT_MISMATCH_RE.search("出力が bit 一致しない") and not BIT_MISMATCH_RE.search("DeviceContextCaptureInProgress")
    v, u, rj = run(no_r1a=True)
    assert v == "UNDETERMINED", (v, u, rj)
    v, u, rj = run(r1b_err=True)
    assert v == "UNDETERMINED" and any("DeviceContextCaptureInProgress" in x for x in u), (v, u, rj)
    v, u, rj = run(r3_block=True)
    assert v == "UNDETERMINED", (v, u, rj)
    # R3 の N 回反復（所有者判断 2026-11-11 (d)）: 混ざったテストは不安定として記録し判定に使わない
    for kw in ({"r3_flaky": True}, {"r3_c_flaky": True}):
        with tempfile.TemporaryDirectory() as d:
            r = make_fixture(d, **kw)
            res = judge(d, r, "t")
            assert res[0] == "ADOPT", (kw, res[0:3])
            assert any(row[4] == "不安定（判定外）" and row[1] == "t::other" for row in res[6]), (kw, res[6])
            md, _ = render(*res, "t")
            assert "不安定（判定外）" in md
    v, u, rj = run(reps=1)
    assert v == "ADOPT", (v, u, rj)
    v, u, rj = run(reps=1, r3_block=True)
    assert v == "UNDETERMINED" and any("阻害" in x for x in u), (v, u, rj)
    # N が RULE に無い・1 回分のログが欠けている（ダンプ単独実行の欠落を含む）は UNDETERMINED
    v, u, rj = run(no_reps=True)
    assert v == "UNDETERMINED" and any("r3_repeats" in x for x in u), (v, u, rj)
    v, u, rj = run(smedump_missing=True)
    assert v == "UNDETERMINED" and any("smedump rep2 未実行" in x for x in u), (v, u, rj)
    # merged doctests（1 節に結果行 2 本）は完了扱い。Running 節の結果行が複数（出力の混入）は UNDETERMINED
    v, u, rj = run(doctest_multi=True)
    assert v == "ADOPT", (v, u, rj)
    v, u, rj = run(running_multi=True)
    assert v == "UNDETERMINED" and any("結果行が複数" in x for x in u), (v, u, rj)
    # host 腕が 5/5 で改善しない -> ADOPT 条件不足で UNDETERMINED
    v, u, rj = run(ratios={f"infer-host-{b}": 1.0 for b in BATCHES})
    assert v == "UNDETERMINED", (v, u, rj)
    # R3 の完了性: 集計行が足りない（途中で異常終了）・説明できない rc・未知ステータスは UNDETERMINED
    for kw in ({"r3_trunc": True}, {"r3_rc": True}, {"r3_status": True}):
        v, u, rj = run(**kw)
        assert v == "UNDETERMINED", (kw, v, u, rj)
    for kw in ({"dup": "train-reuse"}, {"missing": "infer-reuse-64"}, {"order_swap": True},
               {"proto_skew": "infer-fresh-64"}, {"fw_skew": "train-reuse"}):
        try:
            run(**kw)
        except Bad:
            pass
        else:
            raise AssertionError(f"Bad が出ない: {kw}")
    # レンダリングが落ちない
    with tempfile.TemporaryDirectory() as d:
        r = make_fixture(d)
        md, tsv = render(*judge(d, r, "t"), "t")
        assert "verdict: ADOPT" in md and tsv.count("\n") == len(CELLS) + 1
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--rule")
    ap.add_argument("--label")
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-template")
    ap.add_argument("--check-rule")
    a = ap.parse_args()
    try:
        if a.self_test:
            self_test()
            return 0
        if a.check_template:
            print(check_template(a.check_template))
            return 0
        if a.check_rule:
            root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
            print(check_rule(a.check_rule, root))
            return 0
        if not (a.dir and a.rule and a.label and re.fullmatch(r"[A-Za-z0-9._-]+", a.label)):
            raise Bad("--dir／--rule／--label（[A-Za-z0-9._-]+）が必要")
        res = judge(a.dir, a.rule, a.label)
        md, tsv = render(*res, a.label)
        out = a.out_dir or a.dir
        _w(os.path.join(out, "judge-4163.md"), md)
        _w(os.path.join(out, "judge-4163.tsv"), tsv)
        print(f"verdict: {res[0]}")
        return 0
    except Bad as e:
        print(f"INPUT ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
