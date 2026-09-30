#!/usr/bin/env python3
"""steel 差分候補 kernel_gpu 5 run A/B の機械判定（イシュー #2110 が規則を固定・#2111 が実測）。

役割: `orchestrate.sh` が保存した `kernel_gpu_run{1..5}.log`（診断テスト
`gemm_steel_candidate_diag_tests::steel_candidate_kernel_gpu_ab_production_sizes` の標準出力）
を読み、`RULE.txt`（実測前固定。事後に緩和しない）どおりに arm 別の判定を出力する。
python3 標準ライブラリのみ・固定の正規表現のみで解析し、eval・シェル呼び出しは使わない
（`.claude/rules/security.md` A03）。

使い方:
    python3 aggregate.py [ログディレクトリ]   # 既定はこのスクリプトのあるディレクトリ
    python3 aggregate.py --self-test          # 固定 fixture で判定ロジックを検証

出力行の書式は `crates/backend-metal/src/gemm_steel_candidate_diag_tests.rs` と 1:1 対応する
（変更時は両方を更新する）。
"""

import os
import re
import statistics
import sys

N_RUNS = 5
EXPECTED_SIZES = (512, 1024, 2048, 4096)
BASE = "base"
# RULE.txt の固定 4 候補（欠落 arm は結果から消さず INCOMPLETE にする）。
EXPECTED_ARMS = ("T0U", "LU", "T0U-LU", "T0U-LU-FB")
# RULE.txt 1. の前提ゲート 4 テスト（orchestrate.sh の GATE_TESTS と一致させる）。
GATE_TESTS = (
    "gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_all_candidates",
    "gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_dispatch_auto",
    "gemm_steel_candidate_diag_tests::unroll_load_on_off_bit_match_transposed",
    "gemm_steel_candidate_diag_tests::steel_candidate_arms_match_cpu_reference",
)

RE_BIT = re.compile(
    r"^N=(\d+) arm=(\S+) checksum=(-?[0-9.eE+-]+) bit_identical=(true|false) same_kernel=(true|false)(?: same_tile=(true|false))?$"
)
RE_RATIO = re.compile(r"^N=(\d+) arm=(\S+) head_over_base_kernel_gpu=([0-9.eE+-]+)$")


def parse_run(text):
    """1 run 分のログを {(arm, n): {...}} へ解析する。未知の行は無視する。"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        m = RE_BIT.match(line)
        if m:
            n, arm = int(m.group(1)), m.group(2)
            d = out.setdefault((arm, n), {})
            d["checksum"] = m.group(3)
            d["bit_identical"] = m.group(4) == "true"
            d["same_kernel"] = m.group(5) == "true"
            # same_tile 欠落（旧形式）は fail-closed に「同一タイル」扱い＝bit 一致を要求する。
            d["same_tile"] = m.group(6) != "false"
            continue
        m = RE_RATIO.match(line)
        if m:
            out.setdefault((m.group(2), int(m.group(1))), {})["ratio"] = float(m.group(3))
    return out


def judge(runs, reference_only=False):
    """runs: parse_run 結果のリスト。arm 別の判定 dict を返す（RULE.txt 3.〜6.）。"""
    verdicts = {}
    arms = sorted(set(EXPECTED_ARMS) | {arm for r in runs for (arm, _n) in r if arm != BASE})
    for arm in arms:
        if len(runs) != N_RUNS:
            verdicts[arm] = ("INCOMPLETE", f"run 数が {len(runs)}（必要 {N_RUNS}）")
            continue
        detail = []
        considered = []  # (n, median, signs)
        bit_bad = []
        checksum_bad = []
        for n in EXPECTED_SIZES:
            cells = [r.get((arm, n)) for r in runs]
            if any(c is None or "ratio" not in c or "bit_identical" not in c for c in cells):
                verdicts[arm] = ("INCOMPLETE", f"N={n} のデータ欠落")
                break
            # bit 一致は同一タイル cell のみ要求する（タイル形状が異なる arm 間の bit 一致は
            # metal-gemm-steel-candidates.md §5 で契約外。run 間 checksum 一致は常に要求する）。
            if any(c["same_tile"] and not c["bit_identical"] for c in cells):
                bit_bad.append(n)
            if len({c["checksum"] for c in cells}) != 1:
                checksum_bad.append(n)
            if all(c["same_kernel"] for c in cells):
                detail.append(f"N={n}: same_kernel（除外）")
                continue
            ratios = [c["ratio"] for c in cells]
            med = statistics.median(ratios)
            signs_pos = sum(1 for x in ratios if x > 1.0)
            considered.append((n, med, signs_pos))
            detail.append(f"N={n}: median={med:.4f} ratios={[round(x, 4) for x in ratios]}")
        else:
            if bit_bad:
                v = ("NOT_ADOPTABLE", f"bit_identical=false: N={bit_bad}")
            elif checksum_bad:
                v = ("NOT_ADOPTABLE", f"run 間 checksum 不一致: N={checksum_bad}")
            elif not considered:
                v = ("UNDETERMINED", "判定対象 N なし（全 N same_kernel）")
            elif any(med > 1.0 and pos == N_RUNS for _n, med, pos in considered):
                v = ("REJECT", "中央値 >1.00 かつ 5/5 run 符号一貫の N あり")
            elif all(med <= 1.0 for _n, med, _p in considered) and sum(
                1 for _n, med, _p in considered if med < 1.0
            ) >= 2:
                v = ("ADOPT_CANDIDATE", "全対象 N で中央値 <=1.00 かつ <1.00 が 2 形状以上")
            else:
                v = ("UNDETERMINED", "上記いずれにも該当せず")
            if reference_only:
                v = (v[0] + "(REFERENCE_ONLY)", v[1] + " / 負荷ゲート timeout のため参考扱い")
            verdicts[arm] = (v[0], v[1] + " | " + "; ".join(detail))
    return verdicts


def check_gate_log(text):
    """gate_run.log が前提ゲート全件 PASS を示すか検証する（RULE.txt 1.。fail-closed）。

    戻り値: (ok, 理由)。GATE_TESTS の各行が `... ok` で、`test result: ok.` かつ
    `0 failed` を含み、`FAILED`／`panicked` を含まないことを要求する。
    """
    if "FAILED" in text or "panicked" in text:
        return False, "gate_run.log に FAILED／panicked が含まれる"
    for t in GATE_TESTS:
        if not re.search(r"^test " + re.escape(t) + r" \.\.\. ok$", text, re.M):
            return False, f"ゲートテスト未成功または未実行: {t}"
    m = re.search(r"^test result: ok\. (\d+) passed; 0 failed", text, re.M)
    if not m or int(m.group(1)) != len(GATE_TESTS):
        return False, "test result: ok. 4 passed; 0 failed が確認できない"
    return True, "ok"


def load_dir(d):
    runs = []
    for i in range(1, N_RUNS + 1):
        p = os.path.join(d, f"kernel_gpu_run{i}.log")
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                runs.append(parse_run(f.read()))
    reference_only = False
    gate = os.path.join(d, "load_gate.log")
    if os.path.isfile(gate):
        with open(gate, encoding="utf-8", errors="replace") as f:
            reference_only = "TIMEOUT" in f.read()
    return runs, reference_only


def _fixture_run(ratios, bit=True, same=None, checksum="1.000000"):
    """自己テスト用の 1 run 分ログを生成する。ratios: {(arm, n): ratio}。"""
    lines = []
    same = same or set()
    for (arm, n), r in ratios.items():
        lines.append(
            f"N={n} arm={arm} checksum={checksum} bit_identical={'true' if bit else 'false'} "
            f"same_kernel={'true' if (arm, n) in same else 'false'}"
        )
        lines.append(f"N={n} arm={arm} head_over_base_kernel_gpu={r:.6f}")
    return "\n".join(lines)


def self_test():
    def build(per_n, **kw):
        return [parse_run(_fixture_run({("X", n): r for n, r in per_n.items()}, **kw))
                for _ in range(N_RUNS)]

    ok = {512: 0.99, 1024: 0.98, 2048: 1.0, 4096: 1.0}
    assert judge(build(ok))["X"][0] == "ADOPT_CANDIDATE"
    # 1 形状しか <1.00 でない → undetermined
    assert judge(build({512: 1.0, 1024: 0.99, 2048: 1.0, 4096: 1.0}))["X"][0] == "UNDETERMINED"
    # 1 形状でも中央値 >1.00 かつ 5/5 符号一貫 → REJECT
    assert judge(build({512: 0.9, 1024: 0.9, 2048: 0.9, 4096: 1.05}))["X"][0] == "REJECT"
    # 中央値 >1.00 でも符号が一貫しない → REJECT にならない
    mixed = [
        parse_run(_fixture_run({("X", 512): 0.9, ("X", 1024): 0.9, ("X", 2048): 0.9,
                                ("X", 4096): r}))
        for r in (1.05, 1.05, 1.05, 0.99, 0.99)
    ]
    assert judge(mixed)["X"][0] == "UNDETERMINED"
    # bit 不一致
    assert judge(build(ok, bit=False))["X"][0] == "NOT_ADOPTABLE"
    # same_kernel の N は除外（N=512 が 1.5 でも判定に効かない）
    same_runs = [
        parse_run(_fixture_run({("X", 512): 1.5, ("X", 1024): 0.9, ("X", 2048): 0.9,
                                ("X", 4096): 1.0}, same={("X", 512)}))
        for _ in range(N_RUNS)
    ]
    assert judge(same_runs)["X"][0] == "ADOPT_CANDIDATE"
    # 全 N same_kernel → UNDETERMINED
    all_same = [
        parse_run(_fixture_run({("X", n): 1.0 for n in EXPECTED_SIZES},
                               same={("X", n) for n in EXPECTED_SIZES}))
        for _ in range(N_RUNS)
    ]
    assert judge(all_same)["X"][0] == "UNDETERMINED"
    # run 数不足
    assert judge(build(ok)[:3])["X"][0] == "INCOMPLETE"
    # checksum が run 間で不一致
    cs = [parse_run(_fixture_run({("X", n): 0.9 for n in EXPECTED_SIZES}, checksum=f"{i}.0"))
          for i in range(N_RUNS)]
    assert judge(cs)["X"][0] == "NOT_ADOPTABLE"
    # 参考扱いの付記
    assert judge(build(ok), reference_only=True)["X"][0] == "ADOPT_CANDIDATE(REFERENCE_ONLY)"
    # データ欠落 N
    partial = [parse_run(_fixture_run({("X", 512): 0.9})) for _ in range(N_RUNS)]
    assert judge(partial)["X"][0] == "INCOMPLETE"
    # 無関係な行・base は無視
    assert parse_run("garbage\nN=1 arm=base head_over_base_kernel_gpu=1.0")[(BASE, 1)]["ratio"] == 1.0
    # 欠落 arm（固定 4 候補のうち 1 つも出力が無い）は INCOMPLETE で列挙される
    only_x = [parse_run(_fixture_run({("T0U", n): 0.9 for n in EXPECTED_SIZES}))
              for _ in range(N_RUNS)]
    v = judge(only_x)
    assert v["LU"][0] == "INCOMPLETE" and v["T0U-LU-FB"][0] == "INCOMPLETE"
    # 別タイル arm の bit 不一致は NOT_ADOPTABLE にしない／同一タイルの不一致は NOT_ADOPTABLE
    diff_tile = [
        parse_run("\n".join(
            f"N={n} arm=X checksum=1.0 bit_identical=false same_kernel=false same_tile=false\n"
            f"N={n} arm=X head_over_base_kernel_gpu=0.9" for n in EXPECTED_SIZES))
        for _ in range(N_RUNS)
    ]
    assert judge(diff_tile)["X"][0] == "ADOPT_CANDIDATE"
    same_tile = [
        parse_run("\n".join(
            f"N={n} arm=X checksum=1.0 bit_identical=false same_kernel=false same_tile=true\n"
            f"N={n} arm=X head_over_base_kernel_gpu=0.9" for n in EXPECTED_SIZES))
        for _ in range(N_RUNS)
    ]
    assert judge(same_tile)["X"][0] == "NOT_ADOPTABLE"
    # ゲートログ検証
    good = "\n".join(f"test {t} ... ok" for t in GATE_TESTS) + \
        "\ntest result: ok. 4 passed; 0 failed; 0 ignored"
    assert check_gate_log(good)[0]
    assert not check_gate_log(good.replace("... ok", "... FAILED", 1))[0]
    assert not check_gate_log("")[0]
    assert not check_gate_log(good.replace(GATE_TESTS[0], "x"))[0]
    print("self-test OK")


def main(argv):
    if len(argv) > 1 and argv[1] == "--self-test":
        self_test()
        return 0
    d = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    runs, reference_only = load_dir(d)
    gate_path = os.path.join(d, "gate_run.log")
    gate_ok, gate_reason = False, "gate_run.log が存在しない"
    if os.path.isfile(gate_path):
        with open(gate_path, encoding="utf-8", errors="replace") as f:
            gate_ok, gate_reason = check_gate_log(f.read())
    if not gate_ok:
        # 前提ゲート不成立: 採用判定を出さず REJECT を確定する（RULE.txt 1.）。
        print(f"gate=FAIL :: {gate_reason}")
        print("verdict=REJECT :: 前提ゲート不成立のため A/B 判定は行わない（RULE.txt 1.）")
        return 1
    print("gate=OK")
    verdicts = judge(runs, reference_only)
    print(f"runs={len(runs)} reference_only={reference_only}")
    for arm, (v, detail) in sorted(verdicts.items()):
        print(f"arm={arm} verdict={v} :: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
