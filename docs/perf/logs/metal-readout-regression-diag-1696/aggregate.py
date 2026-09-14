#!/usr/bin/env python3
"""Metal readout legacy 後退の 4 腕診断（イシュー #1696）集計スクリプト。

`orchestrate.sh` が生成した `layerB-n{N}-{arm}-run{1..5}.log`
（`readout_regression_diag_tests_1695::run_size_arm` の `println!` 出力）
を正規表現で抽出し、腕×N の 5 起動中央値・起動間 spread（ノイズ床）・
腕差（`BorrowedKeepAlive - LegacyToVec` 等）・checksum 一致を
`aggregate.md` にまとめる。

python3 標準ライブラリのみ（追加依存なし。`docs/perf/logs/cpu-gemm-kc-
sweep-1315/` 等の先例と同方針）。`--self-test` は固定 fixture 文字列で
集計ロジックを検証する（実測ログは不要。CI・Linux 環境でも実行できる）。

抽出対象の出力形式（`readout_regression_diag_tests_1695.rs::run_size_arm`
より。1 run のログに 1 個だけ現れる想定）:

    N=1024 arm=LegacyToVec requested_tile=... (median over 20 trials, 20 warmup) checksum=1.234567:
        readback: median=1.2345 ms  q1=1.1000 ms  q3=1.3000 ms  (q3-q1)=0.2000 ms
        host_read: median=0.1234 ms  q1=0.1000 ms  q3=0.1500 ms  (q3-q1)=0.0500 ms
        sum of medians: 1.3579 ms
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

SIZES = (1024, 2048, 4096)
ARMS = (
    "legacy_to_vec",
    "borrowed_keep_alive",
    "borrowed_with_dummy_alloc_free",
    "pretouched_reused_dest",
)
ARM_LABELS = {
    "legacy_to_vec": "LegacyToVec",
    "borrowed_keep_alive": "BorrowedKeepAlive",
    "borrowed_with_dummy_alloc_free": "BorrowedWithDummyAllocFree",
    "pretouched_reused_dest": "PretouchedReusedDest",
}
RUNS = 5

# `N={n} arm={label} ... checksum={checksum}:` 行を捉える。
HEADER_RE = re.compile(
    r"N=(?P<n>\d+)\s+arm=(?P<label>\S+)\s+.*checksum=(?P<checksum>-?\d+\.\d+):"
)
# `    readback: median=X ms  q1=... q3=... (q3-q1)=...` 行を捉える。
READBACK_RE = re.compile(r"readback:\s+median=(?P<median>-?\d+\.\d+)\s+ms")
HOST_READ_RE = re.compile(r"host_read:\s+median=(?P<median>-?\d+\.\d+)\s+ms")


class ParseError(RuntimeError):
    pass


def parse_log_text(
    text: str,
    *,
    expected_n: int | None = None,
    expected_label: str | None = None,
) -> dict:
    """1 run のログ全文から N・腕ラベル・readback/host_read 中央値・
    checksum を抽出する。ハーネスは 1 呼び出しにつき 1 ブロックのみ
    出力するため最初の一致のみを採用する。

    `expected_n`／`expected_label` を渡した場合、ログ本文から抽出した
    N・腕ラベルがファイル名（`layerB-n{N}-{arm}-run{k}.log`）由来の
    期待値と一致することを検証する。転記・配置ミス（別サイズ／別腕の
    ログを取り違えたファイル名で保存した等）は checksum の腕内一致
    チェックだけではすり抜けるため、ここで fail-closed に検出し
    ParseError として扱う（呼び出し元の `collect()` はこれを欠損
    レコードとして扱い、`render_markdown()` が「欠損／判定不能」を
    報告する）。
    """
    header = HEADER_RE.search(text)
    if header is None:
        raise ParseError("N=.../arm=.../checksum=... 行が見つからない")
    readback = READBACK_RE.search(text)
    if readback is None:
        raise ParseError("readback: median=... 行が見つからない")
    host_read = HOST_READ_RE.search(text)
    if host_read is None:
        raise ParseError("host_read: median=... 行が見つからない")
    n = int(header.group("n"))
    label = header.group("label")
    if expected_n is not None and n != expected_n:
        raise ParseError(
            f"ファイル名由来の n={expected_n} と抽出値 n={n} が不一致"
            "（転記・配置ミスの疑い）"
        )
    if expected_label is not None and label != expected_label:
        raise ParseError(
            f"ファイル名由来の arm 由来ラベル={expected_label} と抽出値 "
            f"label={label} が不一致（転記・配置ミスの疑い）"
        )
    return {
        "n": n,
        "label": label,
        "checksum": float(header.group("checksum")),
        "readback_ms": float(readback.group("median")),
        "host_read_ms": float(host_read.group("median")),
    }


def collect(log_dir: Path) -> dict:
    """`layerB-n{N}-{arm}-run{1..5}.log` を全て読み、
    {(n, arm): [record, ...]} を返す（見つからないファイル・パース失敗
    ファイルは skip し、後段（`render_markdown()`）で欠損／判定不能と
    して報告する）。

    ファイル名から期待される n・腕ラベルを `parse_log_text()` へ渡し、
    ログ本文の抽出値と食い違う場合（転記・配置ミス）も ParseError として
    skip する。skip の理由は標準エラーへ警告として出す（無言で握り
    つぶさない）。"""
    results: dict = {}
    for n in SIZES:
        for arm in ARMS:
            expected_label = ARM_LABELS[arm]
            records = []
            for run in range(1, RUNS + 1):
                path = log_dir / f"layerB-n{n}-{arm}-run{run}.log"
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                try:
                    rec = parse_log_text(
                        text, expected_n=n, expected_label=expected_label
                    )
                except ParseError as exc:
                    print(f"warning: {path}: {exc}", file=sys.stderr)
                    continue
                records.append(rec)
            results[(n, arm)] = records
    return results


def render_markdown(results: dict) -> str:
    lines = []
    lines.append("# Metal readout legacy 後退の 4 腕診断 集計（イシュー #1696）")
    lines.append("")
    lines.append(
        "主系列（単一腕・単一サイズ・プロセス分離・各 5 run）の readback／"
        "host_read 中央値・起動間 spread（ノイズ床）・checksum 一致を集計"
        "する。判定規則は `docs/perf/metal-readout-legacy-regression-four-"
        "arm-diag.md` §5 を正とする。"
    )
    lines.append("")
    lines.append(
        "| N | 腕 | readback median (5 run) | readback ノイズ床 (max-min) "
        "| host_read median (5 run) | run 数 | checksum 一致 |"
    )
    lines.append("|---|----|--------------------------|"
                  "------------------------------|"
                  "---------------------------|--------|----------------|")

    reference_checksums: dict = {}

    for n in SIZES:
        for arm in ARMS:
            records = results.get((n, arm), [])
            label = ARM_LABELS[arm]
            # 5 起動すべてが揃っていない場合（0 件を含む）は代表値・
            # ノイズ床を「残存件数のみ」から算出せず、欠損／判定不能として
            # 報告する（codex-review 指摘: 揃わない起動から算出した中央値
            # は事前登録した 5 起動プロトコルの代表値として扱えない）。
            if len(records) < RUNS:
                status = "未実測" if not records else "欠損／判定不能"
                lines.append(
                    f"| {n} | {label} | {status} | {status} | {status} | "
                    f"{len(records)} | — |"
                )
                continue
            readbacks = [r["readback_ms"] for r in records]
            host_reads = [r["host_read_ms"] for r in records]
            checksums = [r["checksum"] for r in records]
            median_readback = statistics.median(readbacks)
            median_host_read = statistics.median(host_reads)
            noise_floor = max(readbacks) - min(readbacks)
            ref = reference_checksums.setdefault(n, checksums[0])
            checksum_ok = all(
                abs(c - ref) <= abs(ref) * 1e-9 + 1e-6 for c in checksums
            )
            lines.append(
                f"| {n} | {label} | {median_readback:.4f} ms | "
                f"{noise_floor:.4f} ms | {median_host_read:.4f} ms | "
                f"{len(records)} | {'OK' if checksum_ok else 'NG'} |"
            )

    lines.append("")
    lines.append("## 腕差（規則 4: 規模照合）")
    lines.append("")
    lines.append(
        "N=1024 の `BorrowedKeepAlive - LegacyToVec`（readback + host_read "
        "合計）を、`lowlayer-diagnosis-2026-09-12.md` §5 の Δ≈0.47 ms "
        "（matmul 区間差）と比較する。"
    )
    lines.append("")
    for n in SIZES:
        legacy = results.get((n, "legacy_to_vec"), [])
        borrowed = results.get((n, "borrowed_keep_alive"), [])
        # 腕差も同様に、両腕とも 5 起動が揃って初めて中央値差を報告する
        # （揃わない場合の残存件数のみでの算出を避ける。上の主表と同じ
        # 規則）。
        if len(legacy) < RUNS or len(borrowed) < RUNS:
            if not legacy and not borrowed:
                lines.append(f"- N={n}: 未実測")
            else:
                lines.append(
                    f"- N={n}: 欠損／判定不能（"
                    f"legacy_to_vec={len(legacy)}/{RUNS} run, "
                    f"borrowed_keep_alive={len(borrowed)}/{RUNS} run）"
                )
            continue
        legacy_total = statistics.median(
            [r["readback_ms"] + r["host_read_ms"] for r in legacy]
        )
        borrowed_total = statistics.median(
            [r["readback_ms"] + r["host_read_ms"] for r in borrowed]
        )
        diff = borrowed_total - legacy_total
        lines.append(
            f"- N={n}: LegacyToVec={legacy_total:.4f} ms, "
            f"BorrowedKeepAlive={borrowed_total:.4f} ms, "
            f"差={diff:.4f} ms"
        )

    lines.append("")
    return "\n".join(lines) + "\n"


# --- self-test（固定 fixture 文字列。実測ログ不要） -----------------------

_FIXTURE_LOG_TEMPLATE = """running 1 test
  N={n} arm={label} requested_tile=Some(TileConfig) (median over 20 trials, 20 warmup) checksum={checksum:.6f}:
    readback: median={readback:.4f} ms  q1=1.1000 ms  q3=1.3000 ms  (q3-q1)=0.2000 ms
    host_read: median={host_read:.4f} ms  q1=0.1000 ms  q3=0.1500 ms  (q3-q1)=0.0500 ms
    sum of medians: {total:.4f} ms
test readout_regression_diag_n{n}_{arm_slug} ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
"""


def _make_fixture_text(n: int, label: str, checksum: float, readback: float, host_read: float) -> str:
    # `arm_slug` はテスト関数名部分（`legacy_to_vec` 等）を表す。テスト
    # 関数名自体は集計ロジックの対象外（HEADER_RE は `println!` 出力の
    # `arm=<label>` のみを読む）なので固定のダミー値でよい。
    return _FIXTURE_LOG_TEMPLATE.format(
        n=n,
        label=label,
        arm_slug="dummy_arm_slug",
        checksum=checksum,
        readback=readback,
        host_read=host_read,
        total=readback + host_read,
    )


def self_test() -> None:
    # 1) parse_log_text が固定 fixture から期待どおりの値を抽出できること。
    text = _make_fixture_text(1024, "LegacyToVec", 123.456, 1.2345, 0.1234)
    rec = parse_log_text(text)
    assert rec["n"] == 1024, rec
    assert rec["label"] == "LegacyToVec", rec
    assert abs(rec["checksum"] - 123.456) < 1e-6, rec
    assert abs(rec["readback_ms"] - 1.2345) < 1e-9, rec
    assert abs(rec["host_read_ms"] - 0.1234) < 1e-9, rec

    # 2) ヘッダ行のない壊れたログは ParseError になること。
    try:
        parse_log_text("no header here\n")
        raise AssertionError("ParseError が送出されなかった")
    except ParseError:
        pass

    # 3) collect() が複数 run・複数腕を正しく束ねること（一時ディレクトリ
    #    を使わず、collect() 相当の処理をインラインで模して検証する）。
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        readbacks = [1.0, 1.1, 1.2, 1.05, 0.95]
        for i, rb in enumerate(readbacks, start=1):
            p = log_dir / f"layerB-n1024-legacy_to_vec-run{i}.log"
            p.write_text(
                _make_fixture_text(1024, "LegacyToVec", 42.0, rb, 0.1),
                encoding="utf-8",
            )
        results = collect(log_dir)
        recs = results[(1024, "legacy_to_vec")]
        assert len(recs) == 5, recs
        got_readbacks = sorted(r["readback_ms"] for r in recs)
        assert got_readbacks == sorted(readbacks), got_readbacks

        md = render_markdown(results)
        assert "LegacyToVec" in md
        assert "未実測" in md  # 他の (n, arm) 組は記録がないため

    # 4) collect() はファイル名由来の期待値（n・腕ラベル）とログ本文の
    #    抽出値が食い違う場合（転記・配置ミス）を ParseError として
    #    検出し、当該レコードを欠損として扱うこと（checksum が一致
    #    していても素通りしないことを確認する）。
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        # ファイル名は N=1024/legacy_to_vec だが、ログ本文の実体は
        # N=2048 のもの（転記・配置ミスを模す）。
        p = log_dir / "layerB-n1024-legacy_to_vec-run1.log"
        p.write_text(
            _make_fixture_text(2048, "LegacyToVec", 42.0, 1.0, 0.1),
            encoding="utf-8",
        )
        results = collect(log_dir)
        assert results[(1024, "legacy_to_vec")] == [], results[
            (1024, "legacy_to_vec")
        ]

    # 5) 同様に、腕ラベルが食い違う場合（ファイル名は legacy_to_vec だが
    #    本文は BorrowedKeepAlive）も欠損として検出すること。
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        p = log_dir / "layerB-n1024-legacy_to_vec-run1.log"
        p.write_text(
            _make_fixture_text(1024, "BorrowedKeepAlive", 42.0, 1.0, 0.1),
            encoding="utf-8",
        )
        results = collect(log_dir)
        assert results[(1024, "legacy_to_vec")] == [], results[
            (1024, "legacy_to_vec")
        ]

    # 6) 5 起動中 4 起動しか揃わない場合、代表値を残存件数のみから算出
    #    せず「欠損／判定不能」として報告すること。
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        for i, rb in enumerate([1.0, 1.1, 1.2, 1.05], start=1):
            p = log_dir / f"layerB-n1024-legacy_to_vec-run{i}.log"
            p.write_text(
                _make_fixture_text(1024, "LegacyToVec", 42.0, rb, 0.1),
                encoding="utf-8",
            )
        results = collect(log_dir)
        assert len(results[(1024, "legacy_to_vec")]) == 4
        md = render_markdown(results)
        assert "欠損／判定不能" in md

    print("self-test: OK", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="layerB-*.log を探すディレクトリ（既定: このスクリプトと同じ場所）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力先 Markdown（既定: <log-dir>/aggregate.md）",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="固定 fixture 文字列で集計ロジックを検証する（実測ログ不要）",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    out_path = args.out if args.out is not None else args.log_dir / "aggregate.md"
    results = collect(args.log_dir)
    markdown = render_markdown(results)
    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
