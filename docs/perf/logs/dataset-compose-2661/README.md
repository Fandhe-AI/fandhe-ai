# Dataset 合成ユーティリティの実機 round-trip 申し送り（イシュー #2661）

`Subset`／`ConcatDataset`／`random_split` はホスト側ユーティリティで Op／BackendOps／VJP を経由しないため、バックエンド別カーネルと REQ-2 parity は該当しない（`docs/tensor-core-dataset-compose-decision.md` §4）。バッチを `Tape::var` でアップロードする round-trip のみ `#[ignore]` で分離し、本 Linux 環境には実機が無いため**未実測**として申し送る。

## 測定コマンド

```bash
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test data_dataset_compose composed_batch_upload_round_trips_on_cuda_tape -- --ignored
# Metal（Apple Silicon 実機）
cargo test -p fandhe-ai --test data_dataset_compose composed_batch_upload_round_trips_on_metal_tape -- --ignored
```

期待結果: アップロード後の `to_tensor()` がホスト側バッチと bit 完全一致する（純コピーで算術を伴わない）。

## 記入欄

| 環境 | 結果 | 実測日 |
|------|------|--------|
| CUDA（GB10） | 未実測 | - |
| Metal（M4 Max 等） | 未実測 | - |
