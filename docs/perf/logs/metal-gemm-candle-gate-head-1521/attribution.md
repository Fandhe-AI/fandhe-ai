# イシュー #1521 帰属メモ（計測前に固定）

本イシューは `docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/attribution.md`
（正式系列 `fandhe-ai =0.8.0` の非到達根拠）の後継。その後 #1527
（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切替）・#1530（split-K
本番結線。`should_split_k` 分岐を `dispatch_auto`／`select_for_device` へ
結線）が入ったため、`v0.8.0 ↔ origin/main` の Metal 計測経路差分が
初めてゼロでなくなり、参考系列（HEAD path patch）の計測が意味を持つ
状態になった。

## `v0.8.0` タグ ↔ `origin/main` の Metal 計測経路差分

```
$ git diff v0.8.0..origin/main --stat -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src
 crates/backend-metal/src/gemm.rs | 413 +++++++++++++++++++++++++++++++++------
 crates/backend-metal/src/lib.rs  |   2 +-
 crates/backend-metal/src/ops.rs  |   8 +-
 crates/backend-metal/src/tile.rs | 228 ++++++++++++++++++++-
 4 files changed, 583 insertions(+), 68 deletions(-)

$ git log --oneline v0.8.0..origin/main -- crates/backend-metal/src crates/facade/src crates/autodiff/src crates/tensor-core/src
5b2d5060 perf(backend-metal): should_split_k 分岐を dispatch_auto／select_for_device へ結線する (#1530)
ef613b9b feat(backend-metal): SPLIT_K_NUMERIC_CONTRACT_APPROVED を true へ切り替え公開入口の数値契約ゲートを解除する (#1527)
```

（生出力は同ディレクトリ `diff_v0.8.0_origin-main_metal_path.txt`）

## split-K が GEMM ゲート対象形状（NN 正方 N=1024/2048/4096 reuse）へ非到達である根拠

**独立した 2 つの理由**により、v0.7.0/v0.8.0 時点と同じ classic 経路を
なお通る:

1. **本番既定の `split_k_auto_enabled` が `false`**
   （`crates/backend-metal/src/tile.rs` の
   `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED: bool = false`。`MetalGemm::
   new` はこの定数を `split_k_auto_enabled` へそのまま渡す）。
   `dispatch_auto_with_route_impl`（`crates/backend-metal/src/gemm.rs`）は
   `if self.split_k_auto_enabled && SPLIT_K_NUMERIC_CONTRACT_APPROVED`
   の条件が偽であれば split-K 分岐へ一切入らず、`tile::select_for_device`
   を直接呼ぶ classic 経路（結線前と 1 バイトも変わらない経路）を通る。
   `SPLIT_K_NUMERIC_CONTRACT_APPROVED` が #1527 で `true` になった今も、
   `split_k_auto_enabled` 自体が `false` のためこの分岐は開かない。

2. **対象形状自体が `should_split_k` の並列度条件で `None`**
   （`crates/backend-metal/src/tile.rs::
   should_split_k_rejects_large_square_and_wide_shapes` が正方
   512〜4096 を対象に回帰確認済み）。仮に (1) の既定を将来 `true` へ
   変えたとしても、NN 正方 N=1024/2048/4096 は `split_k_tile(m,n)` の
   `actual_groups` が `max_groups`（40）以上になり MLX Case 1 の並列度
   条件で除外される。

この 2 点は本 PR に含む Linux 実行可能なテスト
（`crates/backend-metal/tests/splitk_gemm_gate_shape_attribution.rs`）で
機械的に固定した。同テストの `print_attribution_table`
（`--nocapture`）出力:

```
| shape | (m,n,k) | should_split_k | select_route_for_device |
|---|---|---|---|
| N=1024 | (1024,1024,1024) | should_split_k=None | select_route_for_device=Classic |
| N=2048 | (2048,2048,2048) | should_split_k=None | select_route_for_device=Classic |
| N=4096 | (4096,4096,4096) | should_split_k=None | select_route_for_device=Classic |
```

（`select_route_for_device` は「仮に (1) の既定が `true` だったら」の
純関数判定であり、実際の `dispatch_auto` は (1) により `select_route_for_
device` を呼ぶことすらない。ここでは (2) 単独でも `Classic` に確定する
ことを補強的に確認している。）

## `v0.7.0..v0.8.0` に入った変更との関係

`docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/attribution.md` が
記録したとおり、`v0.7.0..v0.8.0` に入った Metal GEMM 計測経路の変更
（split-K opt-in 実装 #1496・split-K 結線可否確定 #1500・GradStaging
重み勾配読み出し API #1492）はいずれも本計測（推論 GEMM・reuse）に
無関係と判断済み。本イシューが扱う `v0.8.0..origin/main` の追加差分
（#1527・#1530）も、上記 1・2 の理由により同じく無関係と判断する。

E7/E8/E6/E4/E3/E2/loop-unroll/tensor MPP 調査／f16 候補／3×TF32／
CUDA Graph／checksum device reduction 等は、いずれも不採用（REJECT）で
`tile::select` 系・`gemm.metal` 本番選択構成を変更していない
（`docs/perf/metal-gemm-n4096-kernel-gap.md` 各節等に記録済み）か、
CUDA／学習 step 限定でありこの計測には到達しない（`v0.8.0..origin/main`
時点でも変わらない）。

## readout 方式

`scripts/bench/framework-compare/bench-fandhe/src/main.rs` の
`readout_uses_borrowed_view("metal", None)` は `false`（legacy 経路。
`to_tensor()` + `to_vec` 相当）のまま不変。両系列（A: 対照
`0.8.0-ctrl-1521`・B: 参考 `head-<sha>-1521`）とも
`readout_method=legacy-metal-1452` を manifest で検証する。

## 結論（計測前の期待値）

上記より、split-K 結線後 HEAD（参考系列 B）の Metal GEMM NN 正方 reuse
計測経路は、v0.8.0 タグ時点（正式系列相当の対照 A）から**構造的に
不変**。事前登録判定規則（`docs/perf/metal-gemm-candle-gate-
remeasurement.md` §18.1）に従い、B の実測値が A の実測値と有意に
乖離する場合、その差はコード変更に帰属できず、共有負荷等の計測ノイズ
（または構造分析との矛盾・原因未確定）として扱う。
