# Metal readback A/B（イシュー #2112）実行手順

Mac セッション（Apple M4 Max・低負荷時間帯。`uptime` で確認）向け。判定は同ディレクトリの
`RULE.txt`（実測前固定）に従う。設計は `docs/perf/metal-reuse-readback-2112.md`。

## 1. 実機テスト（bit 同一）

```sh
cargo test -p fandhe-ai-backend-metal --release --lib readback_policy -- --ignored
```

## 2. A/B 計測

```sh
cd scripts/bench/framework-compare
AB_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_ab_readback_metal.sh head-<short sha>-2112
```

- 出力: `results/raw/results-m4max-readback-ab-<label>-{fresh,parallel}.jsonl`・manifest・gate／uptime ログ
- `AB_LOAD_GATE_MODE=record_only` は非正式系列（ADOPT 不可）

## 3. 比較（判定セルごと）

```sh
# before = -fresh.jsonl、after = -parallel.jsonl（位置引数）。RULE.txt は ratio <= 1.00 なので
# 既定 threshold 1.05 ではなく --threshold 1.00 を必ず指定する
python3 compare_gemm_ab.py --device metal --task gemm --modes reuse --sizes large \
  --threshold 1.00 --require-checksum-exact <before> <after>
python3 compare_gemm_ab.py --device metal --task infer --modes reuse \
  --threshold 1.00 --require-checksum-exact <before> <after>
```

gemm は N=1024・4096 のみ計測する（N=2048 は判定セル外。表に欠損として出た場合も RULE.txt の
対象 2 セルだけで判定する）。

## 4. 収録（このディレクトリ）

`env_info`（`sysctl`・`pmset -g therm`・`uptime`）・生 JSONL・比較表を置く。内部ホスト名・
絶対パスは `<home>` 等へマスクする。判定後に `docs/perf/metal-reuse-readback-2112.md` §5 を埋め、
ADOPT の場合のみ後続 PR で `READBACK_DEST_DEFAULT` を切り替える。
