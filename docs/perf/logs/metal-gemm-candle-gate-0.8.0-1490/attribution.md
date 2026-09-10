# イシュー #1490 帰属メモ（計測前に固定）

## v0.8.0 タグ ↔ origin/main の Metal 計測経路差分

```
$ git diff v0.8.0..origin/main --stat -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src
（出力なし）
```

差分ゼロ。よって正式系列（registry `fandhe-ai =0.8.0`）と参考系列（origin/main path patch）は
同一ソースになるため、本イシューでは参考系列を計測しない（#1488 §24.2・#1489 §16.5 と同じ判断）。

## `v0.7.0..v0.8.0` の `crates/backend-metal/src`・`crates/facade/src` への変更一覧

（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §11・§14 は `fandhe-ai =0.7.0` 時点の計測。
その後 `v0.8.0` までに入った変更のうち Metal GEMM 計測経路に関わりうるものは次の 2 点）

- `9ab7a306` feat(backend-metal): split-K 2 パス GEMM カーネルと選択純関数 `should_split_k` を opt-in で実装する（#1496）
- `54cb1405` perf(backend-metal): split-K の本番結線可否を確定し `select_for_device`／`dispatch_auto` へ結線する（#1500）
- `93e11a71` feat(facade): resident `GradStaging` の重み勾配をホストへ読み出す公開 API を追加する（#1492）

その他（E7/E8/E6/E4/E3/E2/loop-unroll/tensor MPP 調査／f16 候補／3×TF32／CUDA Graph／checksum device reduction 等）は
いずれも不採用（REJECT）で `tile::select` 系・`gemm.metal` 本番選択構成を変更していない
（`docs/perf/metal-gemm-n4096-kernel-gap.md` 各節・`docs/backend-metal-mpp-tensor-decision.md` 等に記録済み）か、
CUDA／学習 step 限定でありこの計測（Metal・推論 GEMM・reuse）には到達しない。

## split-K が NN 正方 reuse 計測へ非到達である根拠

1. `crates/backend-metal/src/gemm.rs:135`
   ```rust
   pub(crate) const SPLIT_K_NUMERIC_CONTRACT_APPROVED: bool = false;
   ```
   数値契約が未承認のため、split-K プランが生成されても `dispatch` 経路は classic 経路へ
   強制フォールバックする（`gemm.rs:2303` 付近 `gated_plan` 分岐）。

2. `tile::should_split_k`（`crates/backend-metal/src/tile.rs`）は並列度（`actual_groups` 等）に基づき
   正方かつ大きい形状（N=512 以上の正方を含む）を除外する設計になっており、テスト
   `should_split_k_rejects_large_square_and_wide_shapes` がこれを回帰確認している。
   本イシューの計測対象（N=1024/2048/4096 の NN 正方）はこの除外条件に該当し、
   仮に `SPLIT_K_NUMERIC_CONTRACT_APPROVED` が `true` であっても split-K プラン自体が
   選ばれない形状である。

## readout 方式

`scripts/bench/framework-compare/bench-fandhe/src/main.rs:280` の `readout_uses_borrowed_view("metal")` は
`false`（legacy 経路。`to_tensor()` + `to_vec` 相当）のまま不変。`readout_method=legacy-metal-1452` を
manifest で検証する。#1477 の interleave 再計測は undetermined のままこの分岐を変更していない。

## 結論（計測前の期待値）

上記より、v0.8.0 時点の Metal GEMM NN 正方 reuse 計測経路は `fandhe-ai =0.7.0` 時点（§11・§14）から
**構造的に不変**。本イシューの再計測結果が §11（0.836／0.638／0.509 倍）・§14（0.700／0.969／0.710 倍）と
異なる場合、その差は共有負荷等の計測ノイズに帰属するものとし、コード変更に帰属させない。
