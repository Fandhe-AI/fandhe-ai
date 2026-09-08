# M4 Max Layer A（`compare_gemm_ab.py --device cpu --sizes gate`）判定表

イシュー #1301 codex-review 指摘（元の保存記録が candle 行混入により
「判定不能」のまま終わっていた件）の是正版。

## 背景・再現不能だった原因

`results/raw/results-m4max-cpu-gemm-gate-head-145639d-pzero-{off,on}.jsonl`
は `run_gemm_gate_cpu.sh`（イシュー #1117 の candle 比ゲート計測と共有する
出力形式）が発行したもので、`framework=fandhe-ai`（各 5 run × 3 size ×
fresh/reuse = 30 行）に加え `framework=candle`（各 5 run × 3 size ×
fresh のみ = 15 行）が同一ファイルに混在する。`compare_gemm_ab.py` は
`framework != "fandhe-ai"` の行を個別には warning 付きでスキップするが、
1 件でも warning があれば `main()` が fail-closed で全体を「判定不能」に
倒す設計（同スクリプト冒頭 docstring・`load_rows` 実装）である。この設計
自体は「不正行を無視して静かに ok 判定にしない」A08 対策として妥当だが、
本用途（gate 出力の再利用）では candle 行の混入が想定内であるため、
事前に `framework=fandhe-ai` 行だけを抽出したうえで比較する必要がある。

## 再現手順（決定的）

```bash
python3 - <<'PY'
import json
for tag in ["off", "on"]:
    src = f"scripts/bench/framework-compare/results/raw/results-m4max-cpu-gemm-gate-head-145639d-pzero-{tag}.jsonl"
    dst = f"docs/perf/logs/cpu-matmul-fixed-cost-1301/results-m4max-cpu-gemm-gate-head-145639d-pzero-{tag}.fandhe-only.jsonl"
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("framework") == "fandhe-ai":
                g.write(line + "\n")
PY

python3 scripts/bench/framework-compare/compare_gemm_ab.py --device cpu --sizes gate \
  docs/perf/logs/cpu-matmul-fixed-cost-1301/results-m4max-cpu-gemm-gate-head-145639d-pzero-off.fandhe-only.jsonl \
  docs/perf/logs/cpu-matmul-fixed-cost-1301/results-m4max-cpu-gemm-gate-head-145639d-pzero-on.fandhe-only.jsonl
```

抽出後の `*.fandhe-only.jsonl`（各 30 行・candle 行を含まない）は本ディレクトリに保存済み。

## 判定表（実行結果。全セル非後退・checksum 完全一致）

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 738.6 us (min 732.5 us / max 809.0 us) | 743.6 us (min 728.9 us / max 755.2 us) | 1.0068 | 完全一致 | 非後退 |
| 512/reuse | 765.3 us (min 733.9 us / max 783.9 us) | 727.7 us (min 716.2 us / max 772.8 us) | 0.9509 | 完全一致 | 非後退 |
| 1024/fresh | 3.378 ms (min 3.342 ms / max 3.395 ms) | 3.384 ms (min 3.363 ms / max 3.589 ms) | 1.0019 | 完全一致 | 非後退 |
| 1024/reuse | 3.576 ms (min 3.569 ms / max 3.738 ms) | 3.672 ms (min 3.579 ms / max 3.696 ms) | 1.0269 | 完全一致 | 非後退 |
| 2048/fresh | 23.067 ms (min 22.836 ms / max 23.488 ms) | 22.440 ms (min 21.387 ms / max 23.571 ms) | 0.9728 | 完全一致 | 非後退 |
| 2048/reuse | 23.873 ms (min 22.560 ms / max 24.901 ms) | 23.368 ms (min 22.362 ms / max 23.829 ms) | 0.9788 | 完全一致 | 非後退 |

`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §19.3 の M4 Max 列の
数値と完全一致することを確認済み（同 doc は本表を正式値として転記した
もの）。
