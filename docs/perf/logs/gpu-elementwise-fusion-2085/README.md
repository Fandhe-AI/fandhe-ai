# GPU elementwise allowlist 融合（区分 B-1・イシュー #2085）実測ログ

事前登録判定規則の正は `docs/perf/gpu-elementwise-fusion-b1.md`「事前登録
判定規則」（本 README では重複記述しない）。本ディレクトリは実行コマンド・
保存すべきログ一覧・記入欄のみを記載する。

**未実測**: 本エージェント実行環境に CUDA（DGX Spark GB10）・Metal
（Apple Silicon）実機への到達手段がなく、`#[ignore]` 実機テスト・A/B
ベンチは未実行のまま Mac／GB10 セッションへ申し送る。`gb10/`・
`m4max-metal/`・`m4max/`（control）はいずれも空のまま記入欄を残す。

## 実行コマンド

### 1. 正しさの検証（実機必須。ゲート ON）

```sh
# CUDA（DGX Spark GB10 等）
cargo test -p fandhe-ai-backend-cuda --release --test fused_elementwise_parity -- --ignored --nocapture

# Metal（Apple Silicon）
cargo test -p fandhe-ai-backend-metal --release --test fused_elementwise_parity -- --ignored --nocapture
```

### 2. A/B マイクロベンチ（5 run・起動順を反転）

`docs/perf/gpu-elementwise-fusion-b1.md`「事前登録判定規則」のとおり、
before = ゲート OFF（`FANDHE_BENCH_GPU_EW_FUSION=0`）、
after = ゲート ON（`FANDHE_BENCH_GPU_EW_FUSION=1`）。

```sh
# GB10（cuda）: before → after の順で 5 run、run6〜10 で after → before に反転
FANDHE_BENCH_DEVICE=cuda FANDHE_BENCH_GPU_EW_FUSION=0 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/gb10/before_run1.log
FANDHE_BENCH_DEVICE=cuda FANDHE_BENCH_GPU_EW_FUSION=1 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/gb10/after_run1.log
# … run2〜run5 も同様（#1583 と同じく run 単位で起動順を反転する）

# M4 Max（metal）: 同型（gb10 → m4max-metal・cuda → metal）
FANDHE_BENCH_DEVICE=metal FANDHE_BENCH_GPU_EW_FUSION=0 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/m4max-metal/before_run1.log
FANDHE_BENCH_DEVICE=metal FANDHE_BENCH_GPU_EW_FUSION=1 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/m4max-metal/after_run1.log

# M4 Max（cpu。control。ゲート無関係だが同じ 2 腕構成で ratio≈1.00 を確認）
FANDHE_BENCH_DEVICE=cpu FANDHE_BENCH_GPU_EW_FUSION=0 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/m4max/before_run1.log
FANDHE_BENCH_DEVICE=cpu FANDHE_BENCH_GPU_EW_FUSION=1 \
  cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_bench -- --ignored --nocapture \
  > docs/perf/logs/gpu-elementwise-fusion-2085/m4max/after_run1.log
```

### 3. 集計

```sh
python3 docs/perf/logs/gpu-elementwise-fusion-2085/aggregate.py --self-test
python3 docs/perf/logs/gpu-elementwise-fusion-2085/aggregate.py gb10
python3 docs/perf/logs/gpu-elementwise-fusion-2085/aggregate.py m4max-metal
python3 docs/perf/logs/gpu-elementwise-fusion-2085/aggregate.py m4max
```

## 保存すべきログ一覧（各ディレクトリ）

- `before_run{1..5}.log`／`after_run{1..5}.log`
- `aggregate.md`（`aggregate.py` の出力）
- `env_info.txt`（実機の GPU／OS／ドライバ版数・load average。**内部ホスト名は
  含めない**）

## 記入欄

| ディレクトリ | 実機 | 実測日 | verdict |
|---|---|---|---|
| `gb10/` | DGX Spark GB10（CUDA） | 未実測 | 保留 |
| `m4max-metal/` | Apple Silicon（Metal） | 未実測 | 保留 |
| `m4max/` | Apple Silicon（CPU control） | 未実測 | 保留 |
