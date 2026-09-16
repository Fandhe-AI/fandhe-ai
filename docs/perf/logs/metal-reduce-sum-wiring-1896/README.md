# `MetalBackendOps::sum` 結線・判定不能 11 テストの再実測（イシュー #1896）

## 目的

2026-09-16 の M4 Max 実機キャンペーン（#1902。
`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md` §3.1）で、
Metal tape 上の `Var::sum(None)` を loss とする backward テスト 11 件が
`MetalBackendOps::sum` の `Unsupported("reduction カーネル未実装")` に
より判定不能だった。イシュー #1896（Linux 実装 PR。本ディレクトリ）で
`MetalBackendOps::sum` を `context_cache::cached_reduce` 経由で
`reduce::MetalReduce`（#1895 実装済み）へ結線したため、対象 11 テスト・
新規 `#[ignore]` テスト・既存 `#[ignore]` 群の非後退を Mac セッションで
再実測する。

本エージェント実行環境（Linux x86_64）には Apple Silicon 実機への到達
手段がないため、Linux での実装・型検査までを完了し、実測自体は本
ランブックとともに Mac セッションへ申し送る。

## 対象

1. 書き換えた macOS 限定・非 `#[ignore]` テスト（`max_remains_unsupported_
   without_device_init`〈`backend_ops_real_device.rs`〉・
   `sum_rejects_out_of_range_dim_before_touching_device`〈同〉・
   `max_is_unsupported_matching_metal_f32_backend_ops`〈`typed_ops_f16_
   parity.rs`〉・`max_remains_unsupported_without_device_init`〈`typed_ops_
   bf16_parity.rs`〉）が `cargo test -p fandhe-ai-backend-metal
   --all-features`（非 `#[ignore]`）で pass すること。
2. `reduce_parity.rs`（`#[ignore]`。#1895）3 テストが bit 完全一致で
   pass すること（結線後も起動 API 直叩きの検証として維持）。
3. 新規 `#[ignore]` テスト（本 PR。イシュー #1896）:
   - `backend_ops_real_device.rs::backend_ops_sum_matches_cpu_bit_exact`
   - `typed_ops_f16_parity.rs::sum_matches_f32_backend_ops_rounded_bit_exact`
   - `typed_ops_bf16_parity.rs::sum_matches_f32_backend_ops_rounded_bit_exact`
   - `crates/facade/tests/reduce_backend_parity.rs` の
     `metal_sum_all_forward_and_backward_match_cpu_bit_exact`／
     `metal_sum_axis_forward_and_backward_match_cpu_bit_exact`／
     `metal_mean_forward_and_backward_match_cpu_bit_exact`
4. #1902 §3.1 が判定不能とした 11 テスト（`conv2d_backend_parity.rs`・
   `conv1d_backend_parity.rs`・`nn_conv_backend_parity.rs`・
   `gather_scatter_parity.rs`・`constant_pad_parity.rs`・
   `interpolate_parity.rs`・`attention_backend_parity.rs`・
   `no_grad_detach_backend_parity.rs`・`backward_accumulate_backend_
   parity.rs` の Metal 側テスト群）。
5. `make test-ignored-metal` 相当のフル実行が #1902 実測（398 pass／5
   FAIL）比で非後退（上記 4 の FAIL が pass へ転換すること）。

## 実行方法

```sh
docs/perf/logs/metal-reduce-sum-wiring-1896/run_ignored_tests_metal.sh
```

出力: `docs/perf/logs/metal-reduce-sum-wiring-1896/metal/ignored/*.log`・
`docs/perf/logs/metal-reduce-sum-wiring-1896/env_info.txt`（内部ホスト
名は含めない）。

## 事前登録判定規則（事後緩和なし）

- 項目 1・2・3 はいずれも bit 完全一致（`assert_eq!` の `to_bits()` 比較
  または `dense_vec` の完全一致）で判定する。`sum` は CPU 参照実装
  （`fandhe_ai_backend_cpu::reduction::sum`）と bit 完全一致する契約
  （`docs/backend-metal-reduce-sum-design.md` §6）であり、tolerance 定数
  の変更・緩和は一切行わない。
- 項目 4 は各テストの既存判定規則をそのまま適用する:
  - `conv2d_backend_parity.rs`／`conv1d_backend_parity.rs`／
    `nn_conv_backend_parity.rs`: `docs/perf/logs/conv-realdevice-1771/
    README.md` 規則 2)（forward parity）／3)（backward・bit 一致契約。
    `sum` 結線により本 PR で判定可能になる想定）
  - `gather_scatter_parity.rs`／`constant_pad_parity.rs`／
    `interpolate_parity.rs`: 各テストの bit 完全一致契約
  - `attention_backend_parity.rs`／`no_grad_detach_backend_parity.rs`／
    `backward_accumulate_backend_parity.rs`: 各テストの assert（REQ-2
    統一複合判定または bit 完全一致。テスト本体の契約に従う）
- 項目 5 は #1902 実測（398 pass／5 FAIL）比で新規 FAIL がないこと。
  既知 FAIL（下記）は非後退判定に含めない。

## 既知 FAIL（本イシューのスコープ外・非後退判定に含めない）

- `rejects_huge_broadcast_view` 系 3 件（#1897）
- `grad_readout_contract_on_metal`（#1898。→ #1898 で原因 `98c3c67e`〈PR #1659〉を特定し
  テスト側の期待を bias resident 化後の契約へ追従させ是正済み。本 Mac 実測で非 FAIL 化を確認する）
- split-K `auto_entry` の `k=63` fixture（#1899）
- `command_batching` の並列実行時のみの既知 FAIL（Makefile の
  `--no-fail-fast`／`--test-threads=1` 化で吸収する運用は本スクリプト
  側で対応済み。Makefile 自体は変更しない）

## 結果（記入欄。未実測のまま空欄）

| 項目 | 結果 | 備考 |
|---|---|---|
| 1. 非 `#[ignore]` 群 | 未実測 | |
| 2. `reduce_parity` | 未実測 | |
| 3. 新規 `#[ignore]`（sum bit 一致） | 未実測 | |
| 4. 11 テスト | 未実測 | |
| 5. フル実行非後退 | 未実測 | |

env_info: 未取得（内部ホスト名は書かないこと）。
