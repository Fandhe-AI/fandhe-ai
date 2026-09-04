# イシュー #1162 GB10 実測証跡

親イシュー #1134 の受け入れ条件のうち、#1155（GB10 切り分け）が
PR #1180 時点で未実施のまま残っていたため、本イシューで初めて GB10
実機実行した。あわせて `specialized_mma_parity` バイナリ単独実行と
`internal-diagnostics` feature 限定の backend-cuda 全 `#[ignore]` sweep
の pass 確認証跡も収録する。

詳細な数値記録・結論は `docs/perf/cuda-parity-baseline.md` §13 を正とし、
本 README はファイル一覧・実行コマンド・要約に留める（二重管理を避ける）。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `env_info.txt` | 計測環境（GPU・driver・CUDA・rustc・転送元コミット・実行コマンド・GPU 排他性確認結果） |
| `triage_exact_integer.log` | `specialized_mma_f16_triage_exact_integer_inputs`（#1155 決定的証拠。整数厳密入力での bit 一致確認） |
| `triage_dump.log` | `specialized_mma_f16_triage_dump`（#1155 統計的証拠。fail セルの座標・ULP 距離・ヒストグラム） |
| `specialized_mma_parity_run1.log`・`run2.log` | `specialized_mma_parity` バイナリ単独 2 回実行（`--ignored --nocapture --test-threads=1`）。各回 5 passed; 0 failed |
| `full_sweep.log` | `internal-diagnostics` feature 限定の backend-cuda 全 `#[ignore]` テスト sweep（`--no-fail-fast --test-threads=1`）。157 passed / 7 failed（6 バイナリ） |

## #1155 切り分け結果（要約）

- **決定的証拠**（整数厳密入力 `{-1,0,+1}`）: `(256,512,1024)`・
  `(128,256,128)`・`(200,264,104)` の 3 形状すべてで既定カーネル・特化
  カーネルとも CPU 参照との mismatch **0 件**
- **統計的証拠**（主ダンプ）: `(256,512,1024)` seed=4003 の fail 30 件は
  |厳密値| が小さい打ち消し合いセルに集中し、ULP 距離は大半が
  `ulp<=3`（最大でも極小値セルで 44 ULP）。`row%16`/`col%8`/
  `row%BM(64)`/`col%BN(128)` のいずれのヒストグラムにもフラグメント・
  ブロックタイル境界への明確な偏りは見られなかった。追加 seed
  （7101〜7103）でも fail_count は同水準（15〜20/131072）で再現し、
  コントロール `(128,256,128)` seed=4002 は fail_count=0 で一貫
- **結論**: 機能欠陥の証跡なし。f16 出力丸め・Tensor Core 内部
  アキュムレート順序由来という既存の推定（`ParityPath::SpecializedMmaF16`
  ドキュメンテーションコメント）を実機実測で裏付けた

詳細は `cuda-parity-baseline.md` §13.3 を参照。

## `specialized_mma_parity` バイナリ結果（要約）

2 回連続実行とも `5 passed; 0 failed; 0 ignored; 0 measured; 2 filtered
out`。非 ignore smoke（同バイナリ）も `2 passed; 0 failed; 5 ignored`。
`full_sweep.log` 内の同バイナリ実行結果も同一（`5 passed; 0 failed`）。

## 全 sweep 結果（要約）

46 テストバイナリ中 40 バイナリが全 green。6 バイナリで計 7 件の FAIL:

| # | テスト | 対応 |
|---|---|---|
| 1 | `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress` | 既知 red（`cuda-parity-baseline.md` §10.7.3） |
| 2 | `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress` | 既知 red（同上） |
| 3 | `gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress` | 既知 red（同上） |
| 4 | `gemm_tf32_optin.rs::gemm_tf32_optin_on_matches_cpu_across_shapes` | 既知 red（同上） |
| 5 | `tensor_core_real_device.rs::tensor_core_parity_record`（tf32） | 既知 red（同上） |
| 6 | `tensor_core_real_device.rs::tensor_core_tflops_record` | 既知の期待挙動（`docs/perf/cuda-wmma-f16-perf-triage.md` §8.3。`MMA_PRIORITY_PRODUCTION_ENABLED=false` 下では red） |
| 7 | `gemm_tiled.rs::tiled_f32_outperforms_naive_at_4096` | **新規 FAIL**（既存記録なし。事実のみ記録・原因調査は未実施） |

詳細は `cuda-parity-baseline.md` §13.6 を参照。

## 実行コマンド

```bash
# #1155 切り分け
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test specialized_mma_f16_triage specialized_mma_f16_triage_exact_integer_inputs \
  -- --ignored --nocapture --test-threads=1

cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test specialized_mma_f16_triage specialized_mma_f16_triage_dump \
  -- --ignored --nocapture --test-threads=1

# 対象バイナリ単独 2 回
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test specialized_mma_parity -- --ignored --nocapture --test-threads=1

# 全 sweep
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --no-fail-fast -- --ignored --test-threads=1
```

## 環境

GB10（sm_121・CUDA 13.0・driver 580.173.02）。rustc 1.97.0。
転送元コミット `c48d5fcf6459e212e6f03dc1884a91934cfd2341`（origin/main
HEAD。#1194 マージ後）。GPU 排他性確認は実行前後とも異常なし（常駐
サービス〈ComfyUI・Kokoro〉のみ・utilization.gpu=0%）。詳細は
`env_info.txt` 参照。

## 秘密情報・実ホスト名の混入確認

本ディレクトリ配下を
`grep -rniE 'spark|local\.fandhe|[0-9]{1,3}(\.[0-9]{1,3}){3}'` で検査し
0 件（実ホスト名スクラブ済み）。
