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

# kernel_gpu A/B 本体テスト（orchestrate.sh の run_ab が実行する 1 本。RULE.txt 2.）。
AB_TEST = "gemm_steel_candidate_diag_tests::steel_candidate_kernel_gpu_ab_production_sizes"

RE_BIT = re.compile(
    r"^N=(\d+) arm=(\S+) checksum=(-?[0-9.eE+-]+) hash=([0-9a-f]{16}) bit_identical=(true|false) same_kernel=(true|false) same_tile=(true|false)$"
)
RE_RATIO = re.compile(r"^N=(\d+) arm=(\S+) head_over_base_kernel_gpu=([0-9.eE+-]+)$")
RE_MEDIAN = re.compile(
    r"^N=(\d+) arm=(\S+) resolved_tile=.* kernel_gpu_median_ms=([0-9.eE+-]+) q1=[0-9.eE+-]+ q3=[0-9.eE+-]+$"
)


def parse_run(text):
    """1 run 分のログを {(arm, n): {...}} へ解析する。未知の行は無視する。"""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        m = RE_BIT.match(line)
        if m:
            n, arm = int(m.group(1)), m.group(2)
            d = out.setdefault((arm, n), {})
            # checksum（f64 和の 6 桁丸め）は参考値。run 間出力一致は全要素 f32 ビット列の
            # FNV-1a 64bit ハッシュ（hash）で判定する（丸め・誤差相殺で見逃さないため）。
            d["checksum"] = m.group(3)
            d["hash"] = m.group(4)
            d["bit_identical"] = m.group(5) == "true"
            d["same_kernel"] = m.group(6) == "true"
            d["same_tile"] = m.group(7) == "true"
            continue
        m = RE_RATIO.match(line)
        if m:
            out.setdefault((m.group(2), int(m.group(1))), {})["ratio"] = float(m.group(3))
            continue
        m = RE_MEDIAN.match(line)
        if m:
            out.setdefault((m.group(2), int(m.group(1))), {})["median_ms"] = float(m.group(3))
    return out


def check_base_cells(runs):
    """全 run・全 N の base 行（bit 行・中央値行・比行）が揃い妥当か検証する（fail-closed）。

    base は全 arm の比の分母であり、欠落したまま候補 arm の比だけで採用判定すると
    比較基準の無い ADOPT_CANDIDATE を出しうる。戻り値: 問題の文字列リスト（空なら妥当）。
    """
    bad = []
    for i, r in enumerate(runs, start=1):
        for n in EXPECTED_SIZES:
            c = r.get((BASE, n))
            if c is None:
                bad.append(f"run{i} N={n} の base 行が無い")
                continue
            if "bit_identical" not in c or "ratio" not in c or "median_ms" not in c:
                bad.append(f"run{i} N={n} の base 行が不完全（bit／比／中央値のいずれかが無い）")
            elif c["median_ms"] <= 0.0 or c["ratio"] != 1.0 or not c["bit_identical"]:
                bad.append(f"run{i} N={n} の base 値が不正（median_ms>0・ratio=1.0・bit_identical=true が必要）")
    # base 出力の全要素ハッシュが run 間で一致すること（入力・カーネルの決定性確認）。
    for n in EXPECTED_SIZES:
        hs = {r[(BASE, n)]["hash"] for r in runs if (BASE, n) in r and "hash" in r[(BASE, n)]}
        if len(hs) > 1:
            bad.append(f"N={n} の base 出力ハッシュが run 間で不一致（{sorted(hs)}）")
    return bad


def judge(runs, reference_only=False, problems=()):
    """runs: parse_run 結果のリスト。arm 別の判定 dict を返す（RULE.txt 3.〜6.）。

    problems: 入力の欠落・不完全（run ログ欠落・失敗 run・負荷ゲート記録欠落等）の理由リスト。
    1 件でもあれば全 arm を INCOMPLETE にする（採用判定を出さない。fail-closed）。
    """
    verdicts = {}
    problems = list(problems)
    if len(runs) == N_RUNS:
        problems += check_base_cells(runs)
    arms = sorted(set(EXPECTED_ARMS) | {arm for r in runs for (arm, _n) in r if arm != BASE})
    for arm in arms:
        if problems:
            verdicts[arm] = ("INCOMPLETE", "入力不完全: " + "; ".join(problems))
            continue
        if len(runs) != N_RUNS:
            verdicts[arm] = ("INCOMPLETE", f"run 数が {len(runs)}（必要 {N_RUNS}）")
            continue
        detail = []
        considered = []  # (n, median, signs)
        bit_bad = []
        checksum_bad = []
        for n in EXPECTED_SIZES:
            cells = [r.get((arm, n)) for r in runs]
            if any(
                c is None or "ratio" not in c or "bit_identical" not in c or "median_ms" not in c
                for c in cells
            ):
                verdicts[arm] = ("INCOMPLETE", f"N={n} のデータ欠落（bit／比／中央値行のいずれかが無い）")
                break
            if any(c["median_ms"] <= 0.0 for c in cells):
                verdicts[arm] = ("INCOMPLETE", f"N={n} の median_ms が正でない（計測不正）")
                break
            if any(c["ratio"] <= 0.0 for c in cells):
                verdicts[arm] = ("INCOMPLETE", f"N={n} の ratio が正でない（計測不正）")
                break
            # bit 一致は同一タイル cell のみ要求する（タイル形状が異なる arm 間の bit 一致は
            # metal-gemm-steel-candidates.md §5 で契約外。run 間の全要素ビット列ハッシュ一致は常に要求する）。
            if any(c["same_tile"] and not c["bit_identical"] for c in cells):
                bit_bad.append(n)
            if len({c["hash"] for c in cells}) != 1:
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
                v = ("NOT_ADOPTABLE", f"run 間出力ハッシュ不一致: N={checksum_bad}")
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
                # 参考扱い系列（専有ゲート timeout／load_policy が exclusive_gate 以外）は
                # ADOPT_CANDIDATE 等の採用系判定語を一切出さず REFERENCE_ONLY のみとする（RULE.txt 7.。fail-closed）。
                v = ("REFERENCE_ONLY", f"参考扱い（専有ゲート不成立）のため判定 {v[0]} は採用根拠にしない: " + v[1])
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
    # orchestrate.sh はゲート 4 本を 1 本ずつ別プロセスで実行するため、
    # `test result: ok. 1 passed; 0 failed` が 4 件並ぶ（libtest のテスト名フィルタ複数指定に依存しない）。
    n_ok = len(re.findall(r"^test result: ok\. 1 passed; 0 failed", text, re.M))
    if n_ok != len(GATE_TESTS):
        return False, f"test result: ok. 1 passed; 0 failed が {len(GATE_TESTS)} 件必要（実際 {n_ok} 件）"
    return True, "ok"


def check_run_log(text):
    """kernel_gpu_run{i}.log が A/B 本体テストの成功を示すか検証する（RULE.txt 2.。fail-closed）。

    `orchestrate.sh` は cargo が非ゼロ終了してもログを残すため、テスト成功行・
    `test result: ok.`・`0 failed` を確認し、FAILED／panicked を含むログは失敗扱いにする。
    戻り値: (ok, 理由)。
    """
    if "FAILED" in text or "panicked" in text:
        return False, "FAILED／panicked を含む"
    # `--nocapture` 実行では libtest が `test <name> ...` を出した後にテスト本体の標準出力が続き、
    # 判定語 `ok` は後続の独立行になる。同一行（`... ok`）・別行（`...` 行 + 独立 `ok` 行）の双方を許容する。
    if not re.search(r"^test " + re.escape(AB_TEST) + r" \.\.\.(?: ok)?$", text, re.M) or not (
        re.search(r"^test " + re.escape(AB_TEST) + r" \.\.\. ok$", text, re.M)
        or re.search(r"^ok$", text, re.M)
    ):
        return False, "対象テストの成功行（... ok、または ... の後の独立 ok 行）が無い"
    if not re.search(r"^test result: ok\. 1 passed; 0 failed", text, re.M):
        return False, "test result: ok. 1 passed; 0 failed が無い"
    return True, "ok"


def check_load_gate(text, i):
    """load_gate.log から run i の負荷ゲート記録を検証する（RULE.txt 7.。fail-closed）。

    戻り値: (state, 理由)。state は "OK"／"TIMEOUT"／None（記録欠落・不正）。
    run i の記録が無い、または OK と TIMEOUT が混在・重複する場合は None。
    """
    states = re.findall(r"^run" + str(i) + r" (OK|TIMEOUT) ", text, re.M)
    if not states:
        return None, f"run{i} の負荷ゲート記録が load_gate.log に無い"
    if len(states) != 1:
        return None, f"run{i} の負荷ゲート記録が重複している（{states}）"
    return states[0], "ok"


def parse_load_policy(env_text):
    """env_info.txt の `load_policy:` 値を返す。行が無ければ None（RULE.txt 7.。`#` 以降は注釈）。"""
    m = re.search(r"^load_policy:[ \t]*([^\s#]*)", env_text, re.M)
    return m.group(1) if m else None


def load_dir(d):
    """ログディレクトリを読む。戻り値: (runs, reference_only, problems)。

    runs は成功が確認できた run の解析結果のみ。run ログ欠落・失敗・完了記録なし・
    負荷ゲート記録欠落は problems に積み、judge が全 arm を INCOMPLETE にする。
    """
    runs = []
    problems = []
    gate_text = None
    gate = os.path.join(d, "load_gate.log")
    if os.path.isfile(gate):
        with open(gate, encoding="utf-8", errors="replace") as f:
            gate_text = f.read()
    else:
        problems.append("load_gate.log が存在しない")
    env_text = ""
    env = os.path.join(d, "env_info.txt")
    if os.path.isfile(env):
        with open(env, encoding="utf-8", errors="replace") as f:
            env_text = f.read()
    reference_only = False
    # 専有ゲート（RULE.txt 7.）を宣言どおり満たす系列のみ採用根拠にできる。record_only・未記入・
    # 未知値・行欠落はすべて参考扱い（fail-closed）。load_gate.log が全 OK でも覆らない。
    policy = parse_load_policy(env_text)
    if policy != "exclusive_gate":
        reference_only = True
        print(f"note: env_info.txt の load_policy={policy!r}（exclusive_gate 以外）のため参考扱い", file=sys.stderr)
    for i in range(1, N_RUNS + 1):
        p = os.path.join(d, f"kernel_gpu_run{i}.log")
        if not os.path.isfile(p):
            problems.append(f"kernel_gpu_run{i}.log が存在しない")
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            ok, why = check_run_log(text)
            if not ok:
                problems.append(f"run{i} 失敗または未完了: {why}")
            elif not re.search(r"^run" + str(i) + r" completed at ", env_text, re.M):
                problems.append(f"run{i} の完了記録（env_info.txt）が無い")
            else:
                runs.append(parse_run(text))
        if gate_text is not None:
            state, why = check_load_gate(gate_text, i)
            if state is None:
                problems.append(why)
            elif state == "TIMEOUT":
                reference_only = True
    return runs, reference_only, problems


def _fixture_run(ratios, bit=True, same=None, checksum="1.000000", hash_="0000000000000001", with_base=True,
                 with_median=True):
    """自己テスト用の 1 run 分ログを生成する。ratios: {(arm, n): ratio}。

    with_base=True なら出現する各 N の base 行（bit・中央値・比）を補う。
    """
    lines = []
    same = same or set()
    if with_base:
        for n in sorted({n for (_a, n) in ratios}):
            lines.append(
                f"N={n} arm=base checksum={checksum} hash=0000000000000001 bit_identical=true same_kernel=true same_tile=true"
            )
            lines.append(
                f"N={n} arm=base resolved_tile=Cfg kernel_gpu_median_ms=1.0000 q1=0.9000 q3=1.1000"
            )
            lines.append(f"N={n} arm=base head_over_base_kernel_gpu=1.000000")
    for (arm, n), r in ratios.items():
        lines.append(
            f"N={n} arm={arm} checksum={checksum} hash={hash_} bit_identical={'true' if bit else 'false'} "
            f"same_kernel={'true' if (arm, n) in same else 'false'} same_tile=true"
        )
        if with_median:
            lines.append(
                f"N={n} arm={arm} resolved_tile=Cfg kernel_gpu_median_ms=0.9000 q1=0.8000 q3=1.0000"
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
    # ハッシュが run 間で不一致（checksum は全 run 同一＝和が同じでビット列が異なるケース）
    cs = [parse_run(_fixture_run({("X", n): 0.9 for n in EXPECTED_SIZES}, hash_=f"{i:016x}"))
          for i in range(N_RUNS)]
    assert judge(cs)["X"][0] == "NOT_ADOPTABLE"
    # base ハッシュが run 間で不一致 → INCOMPLETE（check_base_cells）
    bh = build(ok)
    bh[2][(BASE, 1024)]["hash"] = "00000000000000ff"
    assert judge(bh)["X"][0] == "INCOMPLETE"
    # 旧形式（hash 欄なし）の行は解析されず INCOMPLETE
    old = [parse_run(_fixture_run({("X", n): 0.9 for n in EXPECTED_SIZES}).replace(" hash=0000000000000001", ""))
           for _ in range(N_RUNS)]
    assert judge(old)["X"][0] == "INCOMPLETE"
    # 参考扱いの付記
    assert judge(build(ok), reference_only=True)["X"][0] == "REFERENCE_ONLY"
    # load_policy: exclusive_gate のみ採用根拠になる。record_only／未記入／欠落／未知値は参考扱い
    assert parse_load_policy("load_policy: exclusive_gate  # x\n") == "exclusive_gate"
    assert parse_load_policy("load_policy: record_only\n") == "record_only"
    assert parse_load_policy("load_policy:  # 未記入\n") == ""
    assert parse_load_policy("chip: x\n") is None
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
            f"N={n} arm=base checksum=1.0 hash=0000000000000001 bit_identical=true same_kernel=true same_tile=true\n"
            f"N={n} arm=base resolved_tile=Cfg kernel_gpu_median_ms=1.0 q1=0.9 q3=1.1\n"
            f"N={n} arm=base head_over_base_kernel_gpu=1.0\n"
            f"N={n} arm=X checksum=1.0 hash=0000000000000001 bit_identical=false same_kernel=false same_tile=false\n"
            f"N={n} arm=X resolved_tile=Cfg kernel_gpu_median_ms=0.9 q1=0.8 q3=1.0\n"
            f"N={n} arm=X head_over_base_kernel_gpu=0.9" for n in EXPECTED_SIZES))
        for _ in range(N_RUNS)
    ]
    assert judge(diff_tile)["X"][0] == "ADOPT_CANDIDATE"
    same_tile = [
        parse_run("\n".join(
            f"N={n} arm=base checksum=1.0 hash=0000000000000001 bit_identical=true same_kernel=true same_tile=true\n"
            f"N={n} arm=base resolved_tile=Cfg kernel_gpu_median_ms=1.0 q1=0.9 q3=1.1\n"
            f"N={n} arm=base head_over_base_kernel_gpu=1.0\n"
            f"N={n} arm=X checksum=1.0 hash=0000000000000001 bit_identical=false same_kernel=false same_tile=true\n"
            f"N={n} arm=X resolved_tile=Cfg kernel_gpu_median_ms=0.9 q1=0.8 q3=1.0\n"
            f"N={n} arm=X head_over_base_kernel_gpu=0.9" for n in EXPECTED_SIZES))
        for _ in range(N_RUNS)
    ]
    assert judge(same_tile)["X"][0] == "NOT_ADOPTABLE"
    # ゲートログ検証
    good = "\n".join(
        f"test {t} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored" for t in GATE_TESTS
    )
    assert check_gate_log(good)[0]
    assert not check_gate_log(good.replace("... ok", "... FAILED", 1))[0]
    assert not check_gate_log("")[0]
    assert not check_gate_log(good.replace(GATE_TESTS[0], "x"))[0]
    # test result 行が 4 件に満たない場合は不成立
    assert not check_gate_log(good.replace("test result: ok. 1 passed", "test result: ok. 0 passed", 1))[0]
    # 入力不完全（run 失敗・ゲート記録欠落等）は全 arm INCOMPLETE
    v = judge(build(ok), problems=["x"])
    assert v["X"][0] == "INCOMPLETE" and v["LU"][0] == "INCOMPLETE"
    # base 行欠落（全欠落・一部 run／N のみ欠落・中央値行のみ欠落・不正値）は INCOMPLETE
    no_base = [parse_run(_fixture_run({("X", n): 0.9 for n in EXPECTED_SIZES}, with_base=False))
               for _ in range(N_RUNS)]
    assert judge(no_base)["X"][0] == "INCOMPLETE"
    part = build(ok)
    del part[2][(BASE, 1024)]
    assert judge(part)["X"][0] == "INCOMPLETE"
    nomed = build(ok)
    del nomed[0][(BASE, 512)]["median_ms"]
    assert judge(nomed)["X"][0] == "INCOMPLETE"
    badbase = build(ok)
    badbase[1][(BASE, 2048)]["ratio"] = 1.2
    assert judge(badbase)["X"][0] == "INCOMPLETE"
    # 候補 arm の中央値行欠落（全欠落・1 run／1 N のみ欠落）は ADOPT_CANDIDATE に到達させず INCOMPLETE
    no_med = [parse_run(_fixture_run({("X", n): 0.9 for n in EXPECTED_SIZES}, with_median=False))
              for _ in range(N_RUNS)]
    assert judge(no_med)["X"][0] == "INCOMPLETE"
    part_med = build(ok)
    del part_med[3][("X", 2048)]["median_ms"]
    assert judge(part_med)["X"][0] == "INCOMPLETE"
    # ratio<=0 は INCOMPLETE
    assert judge(build({512: 0.0, 1024: 0.9, 2048: 0.9, 4096: 0.9}))["X"][0] == "INCOMPLETE"
    # run ログ検証
    good_run = f"test {AB_TEST} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored"
    assert check_run_log(good_run)[0]
    assert not check_run_log("")[0]
    assert not check_run_log(good_run + "\ntest x ... FAILED")[0]
    assert not check_run_log(good_run.replace("0 failed", "1 failed"))[0]
    assert not check_run_log(good_run.replace("... ok", "... ignored"))[0]
    # --nocapture 形式: `test <name> ...` の後にテスト stdout、独立 `ok` 行が続く
    nocap = f"test {AB_TEST} ...\nN=512 arm=base x\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored"
    assert check_run_log(nocap)[0]
    assert not check_run_log(nocap.replace("\nok\n", "\n"))[0]
    # 負荷ゲート記録検証
    assert check_load_gate("run1 OK load1=1 waited=0s\n", 1)[0] == "OK"
    assert check_load_gate("run1 TIMEOUT load1=9 waited=1800s\n", 1)[0] == "TIMEOUT"
    assert check_load_gate("run2 OK load1=1 waited=0s\n", 1)[0] is None
    assert check_load_gate("run1 OK a\nrun1 TIMEOUT b\n", 1)[0] is None
    assert check_load_gate("", 1)[0] is None
    print("self-test OK")


def main(argv):
    if len(argv) > 1 and argv[1] == "--self-test":
        self_test()
        return 0
    d = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    runs, reference_only, problems = load_dir(d)
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
    verdicts = judge(runs, reference_only, problems)
    print(f"runs={len(runs)} reference_only={reference_only}")
    for pr in problems:
        print(f"problem: {pr}")
    for arm, (v, detail) in sorted(verdicts.items()):
        print(f"arm={arm} verdict={v} :: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
