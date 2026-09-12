#!/usr/bin/env python3
"""イシュー #1583 マイクロベンチ A/B 集計（python3 標準ライブラリのみ）。

before_run{1..5}.log / after_run{1..5}.log を読み、
bench[<case>][<numel>].median_s= 行から各 run の値を集め、
5 run の中央値同士の比 (after/before) を計算する。
grad[<case>][<numel>].fold_bits= 行は全 run・before/after で完全一致することを検査する。

集計前に、事前登録した全セル（6 ケース × 3 形状 = 18 セル。EXPECTED_KEYS）×
全 run（既定 5 run）× before/after の bench・grad 記録がすべて過不足なく
揃っていることを検証する（イシュー #1583 PR #1674 codex-review・Bugbot 指摘）。

従来実装（#2 世代目）は期待セル集合を計測ログ自身（1 本目の run の
before/after bench キー和集合）から推定していたため、あるケースが
全 run・全ログ（before/after 双方）で欠落した場合、そのケースは
expected_keys にそもそも現れず「全セル一致」と誤表示され得た
（ログから期待値を導出する自己参照の構造そのものが弱点だった）。
本実装は期待セル集合をログの内容に依存せずソースコード内に事前登録し
（EXPECTED_CASES × EXPECTED_SHAPES）、実際の記録がこの集合と過不足なく
一致することを検査する。

また、同一ログ内で同じキー（case・numel の組）の bench/grad 行が複数回
出現した場合、従来は dict への代入で後勝ちに上書きされ重複を検出できな
かった。本実装は parse() 内で同一キーの再出現を検知した時点でエラーに
する（値が一致していても異なっていても、想定外の重複記録として fail-closed
に扱う）。
"""
import re
import sys
import statistics


# 事前登録した期待セル集合（ログの内容に依存しない固定値）。
# イシュー #1583 の対象: elementwise VJP（Add／Mul／Relu／Tanh／Sigmoid）の
# BackendOps 経由切り替えに対応するベンチケース 6 種 × 計測形状 3 種。
EXPECTED_CASES = (
    "fan_out_accumulate",
    "mul_broadcast",
    "mul_contig",
    "mul_transpose_upstream",
    "sigmoid",
    "tanh",
)
EXPECTED_SHAPES = (16384, 65536, 1048576)
EXPECTED_KEYS = sorted((case, shape) for case in EXPECTED_CASES for shape in EXPECTED_SHAPES)


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


def main(logdir, n_runs=5):
    before_samples = {}
    after_samples = {}
    # grad は run ごとの値をすべて保持する（欠落 run・欠落セルを件数で検知するため
    # 値集合〈set〉だけに畳み込まない。旧実装は values の union のみを見ており、
    # 一部 run が欠落していても報告された値がたまたま一致すれば見分けが付かなかった）。
    before_grad = {}
    after_grad = {}

    expected_keys = EXPECTED_KEYS
    missing = []

    for run in range(1, n_runs + 1):
        before_path = f"{logdir}/before_run{run}.log"
        after_path = f"{logdir}/after_run{run}.log"
        try:
            b, bg = parse(before_path)
            a, ag = parse(after_path)
        except DuplicateKeyError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        for label, path, keys_present in (
            ("before bench", before_path, set(b.keys())),
            ("after bench", after_path, set(a.keys())),
            ("before grad", before_path, set(bg.keys())),
            ("after grad", after_path, set(ag.keys())),
        ):
            lacking = set(expected_keys) - keys_present
            extra = keys_present - set(expected_keys)
            if lacking or extra:
                missing.append((run, label, path, sorted(lacking), sorted(extra)))

        for k in expected_keys:
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
            f"（事前登録セル数={len(expected_keys)}・期待 run 数={n_runs}）"
        )
        for run, label, path, lacking, extra in missing:
            if lacking:
                print(f"  run{run} {label} ({path}): 欠落セル {lacking}")
            if extra:
                print(f"  run{run} {label} ({path}): 未登録セル {extra}")
        print()
        print("事前登録した全セル・全 run の計測記録が過不足なく揃っていないため集計を中止します。")
        sys.exit(1)

    print(f"{'case':30s} {'numel':>9s} {'before_med_s':>14s} {'after_med_s':>14s} {'ratio':>8s} {'grad_ok':>8s}")
    all_ok = True
    for k in expected_keys:
        bench_samples_b = before_samples[k]
        bench_samples_a = after_samples[k]
        # bench 側も n_runs 件そろっていることを明示的に検査する（上の missing
        # 検査は bench 行の有無〈キー単位〉のみを見ているため、同一 run 内で
        # 同じキーが重複記録される異常なログでも件数不一致として拾えるようにする）。
        if len(bench_samples_b) != n_runs or len(bench_samples_a) != n_runs:
            print(f"ERROR: {k} の bench サンプル数が n_runs={n_runs} と一致しません "
                  f"(before={len(bench_samples_b)}, after={len(bench_samples_a)})")
            sys.exit(1)

        bmed = statistics.median(bench_samples_b)
        amed = statistics.median(bench_samples_a)
        ratio = amed / bmed if bmed > 0 else float("nan")

        bvals = before_grad[k]
        avals = after_grad[k]
        # grad_ok は「値が一意」であることに加え、報告件数が n_runs と一致する
        # ことも要求する（欠落 run があれば件数不一致で検知し MISMATCH とする）。
        grad_ok = (
            len(bvals) == n_runs
            and len(avals) == n_runs
            and len(set(bvals)) == 1
            and len(set(avals)) == 1
            and set(bvals) == set(avals)
        )
        if not grad_ok:
            all_ok = False
        flag = "OK" if grad_ok else "MISMATCH"
        over = " <=1.00" if ratio <= 1.00 else " >1.00"
        print(f"{k[0]:30s} {k[1]:9d} {bmed:14.9f} {amed:14.9f} {ratio:8.4f} {flag:>8s}{over}")

    print()
    print("grad bit-fold all match:", all_ok)


if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "."
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(logdir, n_runs)
