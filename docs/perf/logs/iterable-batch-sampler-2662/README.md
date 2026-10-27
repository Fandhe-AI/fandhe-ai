# IterableDataset／BatchSampler の実機 round-trip 記録（イシュー #2662）

`IterableDataset`／`IterableDataLoader`／`BatchSampler` はホスト側ユーティリティで Op／BackendOps／VJP を経由しないため、バックエンド別カーネルと REQ-2 parity は該当しない（`docs/tensor-core-iterable-dataset-batch-sampler-decision.md` §4）。バッチを `Tape::var` でアップロードする round-trip のみ `#[ignore]` で分離しており、#2662 の reopen（#3018 監査 #3269 の申し送り）で GB10 と Mac mini の実機で実測した。手順の正は `docs/real-hardware-verification-env.md`（§3 転送・§7.1 Mac mini）。

## 測定コマンド

```bash
# CUDA（DGX Spark GB10 <cuda-node>。PATH に cargo と CUDA を通す）
cargo test -p fandhe-ai --test data_iterable_batch_sampler iterable_batch_upload_round_trips_on_cuda_tape -- --ignored --exact --nocapture
# Metal（Mac mini <metal-node>。`zsh -lc` 経由）
cargo test -p fandhe-ai --test data_iterable_batch_sampler iterable_batch_upload_round_trips_on_metal_tape -- --ignored --exact --nocapture
```

両機体とも `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`。CUDA 側のテストに `#[cfg]` が無く Mac でもコンパイルされるため、`--exact` で対象 1 本だけを選んだ（`running 1 test` を合格の必須条件とした）。

期待結果: アップロード後の `to_tensor()` がホスト側バッチと bit 完全一致する（純コピーで算術を伴わない）。

## 記入欄

| 実測日（UTC） | 機体 | コマンド | 結果 | 測定 sha・rustc・備考 |
|------|------|----------|------|------|
| 2026-10-27 | DGX Spark GB10 `<cuda-node>` | 上記 CUDA コマンド | pass 1 / fail 0（`iterable_batch_upload_round_trips_on_cuda_tape ... ok`。`running 1 test`・`1 passed; 0 failed`・7 filtered out・0.20s） | sha `045a1241a7b53aa2d8e384e4ef8984966b630f79`（`.rev-stamp` 突合済み）・rustc 1.97.0。実測前後の `utilization.gpu` は 0%。本テストは純コピーの bit 一致判定で性能値を採らない |
| 2026-10-27 | Mac mini `<metal-node>`（Apple M4） | 上記 Metal コマンド | pass 1 / fail 0（`iterable_batch_upload_round_trips_on_metal_tape ... ok`。`running 1 test`・`1 passed; 0 failed`・8 filtered out・0.19s） | sha `045a1241a7b53aa2d8e384e4ef8984966b630f79`（`.rev-stamp` 突合済み）・rustc 1.99.0 |

実機ロック（`~/work/.fandhe-hw-lock`）: GB10 は 2026-10-27T09:53:28Z 取得・09:53:59Z 解放、Mac mini は 09:54:00Z 取得・09:54:18Z 解放（UTC）。ロック保持中に転送・ビルド・実行を完了した。M4 Max 開発機など代替機の値は使っていない。tolerance・parity baseline・コード・テストは変更していない。
