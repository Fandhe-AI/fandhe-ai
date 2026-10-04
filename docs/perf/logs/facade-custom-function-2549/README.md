# facade `Tape::custom`（#2549）CUDA／Metal 実機の申し送り

`docs/autodiff-custom-function-decision.md` §16.1 の確定形で `fandhe_ai::CustomFunction` の
再エクスポートと facade `Tape::custom` の委譲を公開した（イシュー #2549）。

## 実機の parity 実測は対象外

`CustomFunction::forward`／`backward` は常に host の `Tensor<f32>` 上で実行され
`BackendOps` を経由しない（§13.5・§12.5 (d)）。このため REQ-2 のバックエンド間数値一致の
判定対象外であり、CUDA（DGX Spark GB10）・Metal（M4 Max）の parity 実測・baseline は
追加しない。tolerance・baseline は変更していない。CPU Tape での検証は
`crates/facade/tests/custom_function_facade.rs`（組み込み relu との bit 一致・fail-closed の
エラー・`backward_accumulate`・`Send + Sync`）が担う。

## 任意の確認（GPU Tape 上でユーザー関数が動くか）

GPU Tape（`tape_for(Device::Cuda(0))`／`Device::Metal`）の入力は host へ実体化されてから
`forward` に渡る想定（autodiff 層 1 の `materialize_fallible`）。未確認のため、実機を使える
環境で次を任意に確かめる場合は結果を下表へ記入する。

```sh
# 実機上で、tape_for(Device::Cuda(0)) 等を使う `#[ignore]` テストを追加して実行する場合の例
cargo test -p fandhe-ai --test custom_function_facade -- --ignored --nocapture
```

| 環境 | 実施日 | 結果 | 備考 |
|------|--------|------|------|
| DGX Spark GB10（CUDA） | 未実施 | - | - |
| M4 Max（Metal） | 未実施 | - | - |
