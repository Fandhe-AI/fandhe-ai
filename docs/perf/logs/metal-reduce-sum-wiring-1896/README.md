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

## 結果（2026-09-16 M4 Max 実測。イシュー #1894）

実測: Apple M4 Max・origin/main `e64e7565`・`--release`・
`CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`・共有負荷下（load average
約 6〜9。本ディレクトリの項目はすべて parity・bit 一致の確認で負荷非依存）。
ログは `metal/ignored/*.log`・環境は `metal/env_info.txt`（内部ホスト名は
含めない。ログ内の絶対パスは `/Users/<user>/…` へ置換済み）。実装コードの
変更なし・tolerance／baseline／判定規則の事後緩和なし。

| 項目 | 結果 | 備考 |
|---|---|---|
| 1. 非 `#[ignore]` 群 | **PASS**（lib 567 pass／0 fail／93 ignored。統合テスト群も 0 fail） | `backend_metal_all_features.log`（`cargo test -p fandhe-ai-backend-metal --all-features`。書き換えた `max_remains_unsupported_*` 群・`sum_rejects_out_of_range_dim_before_touching_device` を含む） |
| 2. `reduce_parity` | **PASS**（3/3。bit 完全一致） | `reduce_parity.log`: `metal_sum_all_matches_cpu_bit_exact`・`metal_sum_axis_matches_cpu_bit_exact`・`metal_sum_axis_empty_cases_match_cpu` |
| 3. 新規 `#[ignore]`（sum bit 一致） | **PASS**（6/6。bit 完全一致） | `backend_ops_sum_bit_exact.log`（1）・`typed_ops_f16_sum.log`（1）・`typed_ops_bf16_sum.log`（1）・`reduce_backend_parity_metal.log`（3: `metal_sum_all_*`／`metal_sum_axis_*`／`metal_mean_*`） |
| 4. 11 テスト | **PASS**（11/11。全件 FAIL → pass へ転換） | 内訳は下表 |
| 5. フル実行非後退 | **非後退**（411 pass／1 FAIL。#1902 の 398 pass との by-name 差: 後退 0 件〈`command_batching` 並列限定の既知 FAIL 1 件を除く〉・新規 pass 14 件） | `full_ignored_metal.log`・`command_batching_isolation.log` |

### 項目 4 の内訳（#1902 §3.1 の 11 テスト）

| ログ | テスト | #1902 | 今回 |
|---|---|---|---|
| `conv2d_backend_parity.log` | `metal_conv2d_backward_matches_cpu` | FAIL（判定不能） | pass |
| `conv1d_backend_parity.log` | `metal_conv1d_backward_matches_cpu` | FAIL（判定不能） | pass |
| `conv1d_backend_parity.log` | `metal_conv1d_matches_manual_reshape_conv2d_bit_exact` | FAIL（判定不能） | pass |
| `nn_conv_backend_parity.log` | `metal_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact` | FAIL（判定不能） | pass |
| `gather_scatter_parity.log` | `gather_backward_matches_cpu_tape` | FAIL（判定不能） | pass |
| `constant_pad_parity.log` | `pad_backward_matches_cpu_tape` | FAIL（判定不能） | pass |
| `interpolate_parity.log` | `interpolate_backward_matches_cpu_tape` | FAIL（判定不能） | pass |
| `interpolate_parity.log` | `interpolate_bilinear_backward_matches_cpu_tape` | FAIL（判定不能） | pass |
| `attention_backend_parity.log` | `metal_sdpa_backward_dq_matches_cpu` | FAIL（判定不能） | pass |
| `no_grad_detach_backend_parity.log` | `metal_detach_weight_grad_matches_cpu` | FAIL（判定不能） | pass |
| `backward_accumulate_backend_parity.log` | `metal_backward_accumulate_weight_grad_matches_cpu` | FAIL（判定不能） | pass |

各バイナリの同居テスト（forward parity 等）も全件 pass（`conv2d` 2・
`conv1d` 4・`nn_conv` 6・`gather_scatter` 20・`constant_pad` 5・
`interpolate` 6・`attention` 2・`no_grad_detach` 1・`backward_accumulate` 1）。

### 項目 5 の内訳（フル実行 `--all-features --no-fail-fast -- --ignored`）

- 今回: 411 pass／1 FAIL（`command_batching::pool_reuse_zero_fill_does_not_
  synchronize_open_batch`）。#1902: 398 pass／5 FAIL
- by-name 比較（`test … ok` 行の名前の多重集合差）:
  - #1902 で ok・今回 ok でない: `pool_reuse_zero_fill_does_not_synchronize_
    open_batch` の 1 件のみ。#1902 §1 で記録済みの singleton `MetalContext`
    診断カウンタの**並列干渉**（既知 FAIL。非後退判定に含めない）。同
    バイナリを `--test-threads=1` で 3 回直列再実行し **3/3 pass**
    （`command_batching_isolation.log`）
  - 今回新たに ok: 14 件（#1902 の FAIL 5 件のうち `sum` 起因 4 件
    〈`gather_backward`／`pad_backward`／`interpolate_backward`／
    `interpolate_bilinear_backward`〉と split-K `auto_entry_falls_back_to_
    classic_not_eligible_for_non_split_k_shapes`〈#1899 是正〉の計 5 件が
    pass へ転換・#1899 新規 `auto_entry_rejects_precondition_violating_
    shapes_with_typed_err` 1 件・#1895 `reduce_parity` 3 件・#1896 新規
    sum テスト 5 件〈lib 2・統合 3〉）
- #1902 既知 FAIL との突合: `rejects_huge_broadcast_view` 系 3 件（#1897）は
  項目 1 の lib テストで pass（0 fail）・split-K `k=63` fixture（#1899）は
  上記のとおり pass・`grad_readout_contract_on_metal`（#1898）は facade
  クレート側のため本フル実行の対象外（#1898 ランブックで別途 2/2 pass を
  確認し #1898 へコメント）

env_info: `metal/env_info.txt`（内部ホスト名は含めない）。
