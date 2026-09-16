# `grad_readout_contract_on_metal` 是正の実機再実測（イシュー #1898）

## 背景

M4 Max・origin/main `565300e4` で `crates/facade/tests/
device_param_store_backend_parity.rs::grad_readout_contract_on_metal`
が「param 1（bias）の resident 充填状態が期待と異なる」で FAIL していた
（`docs/perf/logs/metal-mse-backward-1691/ignored_store_parity_serial.log`）。
回帰窓 `e851e91a..565300e4` の原因コミットはイシュー #1566（PR #1659。
Metal `gemm_fp32_strict_into_with_bias_reduce_tracked` オーバーライドの
追加）: Metal は weight slot に加え bias slot も resident 経由で `Some`
を返すようになったが、テスト側の期待（「Metal も CPU・CUDA と同じく
bias slot は `None`」）が追従できていなかった。

イシュー #1898 で `device_param_store_backend_parity.rs`・
`device_param_store_metal_mixed_shape_grad.rs` の期待を現契約へ更新した
（`StrictBiasExpectation` enum を新設し `Resident`〈Metal〉／
`HostRouted`〈CPU・CUDA〉を明示。CUDA 側 `grad_readout_contract_on_cuda`
は非対象・非後退確認のみ）。実装側の本番コード（`crates/backend-metal/
src/ops.rs`・`crates/autodiff/src/optim/device_store.rs`）は無変更。

現在のバックエンド別契約（strict 版 `resident_grads_to_host` の bias
slot）:

| バックエンド | weight slot | bias slot |
|---|---|---|
| CPU | `Some` | `None` |
| CUDA（#1559） | `Some` | `None` |
| Metal（#1555 + #1566） | `Some` | `Some` |

## 対象

1. `device_param_store_backend_parity.rs::grad_readout_contract_on_metal`
   （AC 2。本イシューの是正対象。weight・bias とも `Some`・strict 版と
   統合版が bit 完全一致すること）
2. `device_param_store_backend_parity.rs::device_resident_matches_host_
   sgd_on_metal_across_100_steps`（非後退。100 step 累積比較）
3. `device_param_store_metal_mixed_shape_grad.rs::param_grads_to_host_
   succeeds_when_backward_mixes_supported_and_fallback_shapes`（AC 3。
   前段 NN フォールバック・後段 NT/TN 混在ケースで全 4 slot `Some`）
4. `device_param_store_backend_parity.rs::grad_readout_contract_on_cuda`
   （AC 3。CUDA 側は契約不変のため非後退確認のみ。GB10 実機）

## 実行方法

```sh
# Metal 実機（Apple Silicon）
docs/perf/logs/grad-readout-contract-1898/run_ignored_tests_metal.sh

# CUDA 実機（DGX Spark GB10 等）
docs/perf/logs/grad-readout-contract-1898/run_ignored_tests_cuda.sh
```

`make test-ignored-metal-facade`（イシュー #1898 で新設。`Makefile`
参照）でも項目 1・3 相当を実行できる（上記スクリプトは項目 2 も含む
点・ログの保存先が異なる点が差分）。

出力: `docs/perf/logs/grad-readout-contract-1898/{metal,cuda}/ignored/
*.log`（内部ホスト名は含めないこと）。

## 事前登録判定規則（事後緩和なし）

- 項目 1〜3（Metal）: 全 assert が pass すること。とくに項目 1 の bias
  slot は `Some`（weight と同じく resident 充填済み）であり、かつ strict
  版の bias 値が統合版 `param_grads_to_host` の同 index と bit 完全一致
  すること（`crates/facade/tests/device_param_store_backend_parity.rs::
  assert_grad_readout_contract` の追加検証。§`to_bits()` 比較）。ホスト
  参照実装との比較は従来どおり統一複合判定（相対誤差 1e-3 未満または
  絶対誤差 1e-5 未満）。tolerance 定数・判定式の変更・緩和は一切行わない
- 項目 4（CUDA）: 契約変更なしのため非後退（既存どおり pass）のみを
  確認する
- 既存 `#[ignore]` 群（同バイナリの他テスト）は非後退であること

## 結果（記入欄。未実測のまま空欄）

| 項目 | 結果 | 備考 |
|---|---|---|
| 1. `grad_readout_contract_on_metal` | 未実測 | |
| 2. `device_resident_matches_host_sgd_on_metal_across_100_steps` | 未実測 | |
| 3. `device_param_store_metal_mixed_shape_grad` | 未実測 | |
| 4. `grad_readout_contract_on_cuda`（GB10） | 未実測 | |

env_info: 未取得（内部ホスト名は書かないこと）。
