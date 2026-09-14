#!/usr/bin/env python3
"""イシュー #1691 マイクロベンチ A/B 集計（python3 標準ライブラリのみ）。

before_round{1..5}.log / after_round{1..5}.log を読み、
bench[<case>][<numel>].median_s= 行から各 run の値を集め、
5 run の中央値同士の比 (after/before) を計算する。
grad[<case>][<numel>].fold_bits= 行は全 run・before/after で完全一致することを検査する。

集計前に、事前登録した全セル（`train_shape`＋`general_shape` 3 形状 = 4
セル。EXPECTED_KEYS）× 全 run（既定 5 run）× before/after の bench・grad
記録がすべて過不足なく揃っていることを検証する
（`docs/perf/logs/elementwise-vjp-backend-ops-1583/aggregate.py`
〈イシュー #1583〉と同型の設計。期待セル集合をログの内容に依存せず
ソースコード内に事前登録し、実際の記録がこの集合と過不足なく一致する
ことを検査する。同一キーの bench/grad 行が同一ログ内で複数回出現した
場合も fail-closed でエラーにする）。

`crates/facade/tests/mse_backward_bench.rs::mse_backward_cases` の出力
形式（`bench[<case>][<numel>].median_s=`・
`grad[<case>][<numel>].fold_bits=`）に対応する。
"""
import re
import sys
import statistics


# 事前登録した期待セル集合（ログの内容に依存しない固定値）。
# イシュー #1691 の対象: `mse_backward_bench.rs::mse_backward_cases` の
# `train_shape`（主・640 要素）・`general_shape`（副・3 サイズ）。
EXPECTED_CASES = ("train_shape", "general_shape")
TRAIN_SHAPE_NUMEL = (640,)
GENERAL_SHAPE_NUMELS = (16384, 65536, 1048576)


def expected_keys():
    keys = [("train_shape", n) for n in TRAIN_SHAPE_NUMEL]
    keys += [("general_shape", n) for n in GENERAL_SHAPE_NUMELS]
    return sorted(keys)


class DuplicateKeyError(Exception):
    """同一ログ内で同じ (case, numel) キーの bench/grad 行が複数回出現した場合に送出する。"""


def parse(path):
    bench = {}
    grad = {}
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            m = re.match(r"bench\[([^\]]+)\]\[(\d+)\]\.median_s=([0-9.eE+-]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in bench:
                    raise DuplicateKeyError(
                        f"{path}:{lineno}: bench{list(key)} が同一ログ内で複数回出現しました"
                        f"（先行値={bench[key]!r}・後続値={m.group(3)!r}）"
                    )
                bench[key] = float(m.group(3))
                continue
            m = re.match(r"grad\[([^\]]+)\]\[(\d+)\]\.fold_bits=(0x[0-9a-f]+)", line)
            if m:
                key = (m.group(1), int(m.group(2)))
                if key in grad:
                    raise DuplicateKeyError(
                        f"{path}:{lineno}: grad{list(key)} が同一ログ内で複数回出現しました"
                        f"（先行値={grad[key]!r}・後続値={m.group(3)!r}）"
                    )
                grad[key] = m.group(3)
    return bench, grad


def run_aggregate(logdir, n_runs=5, out=sys.stdout):
    """1 件でも検証エラーがあれば (False, message) を返し、成功時は
    (True, None) を返す（`--self-test` から標準出力を汚さずに呼べるように
    print 先を引数化する）。
    """
    before_samples = {}
    after_samples = {}
    before_grad = {}
    after_grad = {}

    keys = expected_keys()
    missing = []

    for run in range(1, n_runs + 1):
        before_path = f"{logdir}/before_round{run}.log"
        after_path = f"{logdir}/after_round{run}.log"
        try:
            b, bg = parse(before_path)
            a, ag = parse(after_path)
        except DuplicateKeyError as e:
            print(f"ERROR: {e}", file=out)
            return False

        for label, path, keys_present in (
            ("before bench", before_path, set(b.keys())),
            ("after bench", after_path, set(a.keys())),
            ("before grad", before_path, set(bg.keys())),
            ("after grad", after_path, set(ag.keys())),
        ):
            lacking = set(keys) - keys_present
            extra = keys_present - set(keys)
            if lacking or extra:
                missing.append((run, label, path, sorted(lacking), sorted(extra)))

        for k in keys:
            if k in b:
                before_samples.setdefault(k, []).append(b[k])
            if k in a:
                after_samples.setdefault(k, []).append(a[k])
            if k in bg:
                before_grad.setdefault(k, []).append(bg[k])
            if k in ag:
                after_grad.setdefault(k, []).append(ag[k])

    if missing:
        print(
            f"ERROR: 集計前検証に失敗しました"
            f"（事前登録セル数={len(keys)}・期待 run 数={n_runs}）",
            file=out,
        )
        for run, label, path, lacking, extra in missing:
            if lacking:
                print(f"  run{run} {label} ({path}): 欠落セル {lacking}", file=out)
            if extra:
                print(f"  run{run} {label} ({path}): 未登録セル {extra}", file=out)
        print(file=out)
        print("事前登録した全セル・全 run の計測記録が過不足なく揃っていないため集計を中止します。", file=out)
        return False

    print(
        f"{'case':16s} {'numel':>9s} {'before_med_s':>14s} {'after_med_s':>14s} {'ratio':>8s} {'grad_ok':>8s}",
        file=out,
    )
    # `grad_all_ok`（fold_bits 完全一致）と `ratio_all_ok`（全セル 5 run
    # 中央値比 after/before <= 1.00）は独立に集計する。総合合否
    # （呼び出し元 main() の終了コードに反映される `all_ok`）は両方の
    # 論理積とする（codex-review [P1] 指摘対応: 従来は grad_ok のみで
    # `all_ok` を決めており、全セルで性能が後退していても fold_bits さえ
    # 一致すれば成功終了していた。事前登録判定規則「5 run 中央値比
    # after/before <= 1.00」を成功条件へ正しく含める）。
    grad_all_ok = True
    ratio_all_ok = True
    for k in keys:
        bench_samples_b = before_samples[k]
        bench_samples_a = after_samples[k]
        if len(bench_samples_b) != n_runs or len(bench_samples_a) != n_runs:
            print(
                f"ERROR: {k} の bench サンプル数が n_runs={n_runs} と一致しません "
                f"(before={len(bench_samples_b)}, after={len(bench_samples_a)})",
                file=out,
            )
            return False

        bmed = statistics.median(bench_samples_b)
        amed = statistics.median(bench_samples_a)
        ratio = amed / bmed if bmed > 0 else float("nan")
        # `ratio` が NaN（bmed<=0 の異常値）の場合 `ratio <= 1.00` は
        # Python の比較規則により False になるため、fail-closed で
        # ratio_ok=False として扱われる（NaN を「非後退」と誤判定しない）。
        ratio_ok = ratio <= 1.00
        if not ratio_ok:
            ratio_all_ok = False

        bvals = before_grad[k]
        avals = after_grad[k]
        grad_ok = (
            len(bvals) == n_runs
            and len(avals) == n_runs
            and len(set(bvals)) == 1
            and len(set(avals)) == 1
            and set(bvals) == set(avals)
        )
        if not grad_ok:
            grad_all_ok = False
        flag = "OK" if grad_ok else "MISMATCH"
        over = " <=1.00" if ratio_ok else " >1.00"
        print(
            f"{k[0]:16s} {k[1]:9d} {bmed:14.9f} {amed:14.9f} {ratio:8.4f} {flag:>8s}{over}",
            file=out,
        )

    all_ok = grad_all_ok and ratio_all_ok
    print(file=out)
    print("grad bit-fold all match:", grad_all_ok, file=out)
    print("all cells ratio<=1.00:", ratio_all_ok, file=out)
    print("overall judgement (grad match and ratio<=1.00):", all_ok, file=out)
    return all_ok


def main(logdir, n_runs=5):
    ok = run_aggregate(logdir, n_runs)
    if not ok:
        sys.exit(1)


# --- self-test（実機ログなしで検証ロジック自体を確認する。イシュー
# #1691 の Linux 検証手順。fixture は一時ディレクトリへ書き出して
# run_aggregate() を直接呼ぶ） ---

def _write_log(path, bench_lines, grad_lines):
    with open(path, "w") as f:
        for line in bench_lines:
            f.write(line + "\n")
        for line in grad_lines:
            f.write(line + "\n")


def _fixture_lines(median_s_by_key, fold_bits_by_key):
    bench_lines = [
        f"bench[{case}][{numel}].median_s={val:.9f}"
        for (case, numel), val in median_s_by_key.items()
    ]
    grad_lines = [
        f"grad[{case}][{numel}].fold_bits={fold}"
        for (case, numel), fold in fold_bits_by_key.items()
    ]
    return bench_lines, grad_lines


def self_test():
    import io
    import tempfile
    import os

    keys = expected_keys()
    fixed_bits = "0x00000000deadbeef"

    # --- fixture 1: 正常系（全セル・全 run 揃い・fold_bits 完全一致） ---
    with tempfile.TemporaryDirectory() as d:
        for run in range(1, 6):
            median_by_key = {k: 0.001 * (run + 1) for k in keys}
            fold_by_key = {k: fixed_bits for k in keys}
            b_lines, g_lines = _fixture_lines(median_by_key, fold_by_key)
            _write_log(os.path.join(d, f"before_round{run}.log"), b_lines, g_lines)
            median_by_key_after = {k: 0.0009 * (run + 1) for k in keys}
            b_lines_a, g_lines_a = _fixture_lines(median_by_key_after, fold_by_key)
            _write_log(os.path.join(d, f"after_round{run}.log"), b_lines_a, g_lines_a)
        buf = io.StringIO()
        ok = run_aggregate(d, 5, out=buf)
        assert ok is True, f"normal fixture should pass: {buf.getvalue()}"

    # --- fixture 2: 欠落セル（1 run で general_shape/1048576 が欠落） ---
    with tempfile.TemporaryDirectory() as d:
        for run in range(1, 6):
            median_by_key = {k: 0.001 for k in keys}
            fold_by_key = {k: fixed_bits for k in keys}
            if run == 3:
                del median_by_key[("general_shape", 1048576)]
                del fold_by_key[("general_shape", 1048576)]
            b_lines, g_lines = _fixture_lines(median_by_key, fold_by_key)
            _write_log(os.path.join(d, f"before_round{run}.log"), b_lines, g_lines)
            b_lines_a, g_lines_a = _fixture_lines(
                {k: 0.001 for k in keys}, {k: fixed_bits for k in keys}
            )
            _write_log(os.path.join(d, f"after_round{run}.log"), b_lines_a, g_lines_a)
        buf = io.StringIO()
        ok = run_aggregate(d, 5, out=buf)
        assert ok is False, "missing-cell fixture should fail"
        assert "欠落セル" in buf.getvalue()

    # --- fixture 3: 重複キー（同一ログ内で同じキーが 2 回出現） ---
    with tempfile.TemporaryDirectory() as d:
        for run in range(1, 6):
            median_by_key = {k: 0.001 for k in keys}
            fold_by_key = {k: fixed_bits for k in keys}
            b_lines, g_lines = _fixture_lines(median_by_key, fold_by_key)
            if run == 1:
                b_lines = b_lines + [b_lines[0]]  # 先頭行を複製して重複させる
            _write_log(os.path.join(d, f"before_round{run}.log"), b_lines, g_lines)
            b_lines_a, g_lines_a = _fixture_lines(
                {k: 0.001 for k in keys}, {k: fixed_bits for k in keys}
            )
            _write_log(os.path.join(d, f"after_round{run}.log"), b_lines_a, g_lines_a)
        buf = io.StringIO()
        ok = run_aggregate(d, 5, out=buf)
        assert ok is False, "duplicate-key fixture should fail"
        assert "複数回出現" in buf.getvalue()

    # --- fixture 4: fold_bits 不一致（run 間で値が変わる） ---
    with tempfile.TemporaryDirectory() as d:
        for run in range(1, 6):
            median_by_key = {k: 0.001 for k in keys}
            fold_val = fixed_bits if run != 4 else "0x0000000000000001"
            fold_by_key = {k: fold_val for k in keys}
            b_lines, g_lines = _fixture_lines(median_by_key, fold_by_key)
            _write_log(os.path.join(d, f"before_round{run}.log"), b_lines, g_lines)
            b_lines_a, g_lines_a = _fixture_lines(
                {k: 0.001 for k in keys}, {k: fixed_bits for k in keys}
            )
            _write_log(os.path.join(d, f"after_round{run}.log"), b_lines_a, g_lines_a)
        buf = io.StringIO()
        ok = run_aggregate(d, 5, out=buf)
        # fold_bits 不一致は grad_ok=False として集計自体は継続するが
        # 「grad bit-fold all match: False」で終わる（run_aggregate の
        # 戻り値も False）。
        assert ok is False, "fold_bits mismatch fixture should report all_ok=False"
        assert "grad bit-fold all match: False" in buf.getvalue()

    # --- fixture 5: ratio>1.00（after が全セルで遅い。grad は完全一致）。
    # codex-review [P1] 指摘対応の回帰確認: grad_ok が全セルで True でも
    # 性能後退があれば all_ok=False・終了コード非 0 になることを検証する
    # （旧実装は grad_ok のみで all_ok を決めていたため、この fixture は
    # 旧実装では誤って ok=True を返していた）。
    with tempfile.TemporaryDirectory() as d:
        for run in range(1, 6):
            fold_by_key = {k: fixed_bits for k in keys}
            b_lines, g_lines = _fixture_lines({k: 0.001 for k in keys}, fold_by_key)
            _write_log(os.path.join(d, f"before_round{run}.log"), b_lines, g_lines)
            # after は before の 2 倍遅い（ratio=2.0 > 1.00）。
            b_lines_a, g_lines_a = _fixture_lines({k: 0.002 for k in keys}, fold_by_key)
            _write_log(os.path.join(d, f"after_round{run}.log"), b_lines_a, g_lines_a)
        buf = io.StringIO()
        ok = run_aggregate(d, 5, out=buf)
        assert ok is False, "ratio>1.00 fixture should fail even when grad matches"
        out_text = buf.getvalue()
        assert "grad bit-fold all match: True" in out_text
        assert "all cells ratio<=1.00: False" in out_text
        assert "overall judgement (grad match and ratio<=1.00): False" in out_text

    print(
        "self-test: all fixtures passed (normal / missing-cell / duplicate-key / "
        "fold-bits-mismatch / ratio-regression)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        logdir = sys.argv[1] if len(sys.argv) > 1 else "."
        n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        main(logdir, n_runs)
