# Metal GEMM: m=64 級形状向け細タイル候補（既定 OFF）

イシュー #3208。A/B 計測は兄弟 #3209（Mac mini 上）。本 doc は候補の定義・到達範囲・bit 一致確認・所有者への申し送りを記録する。性能は計測していない。

## 1. 背景と前提

- 対象は MLP（784→256→10・batch 64）の GEMM 6 形状。依存の #3181 は正方形状のみ、#3180 は phase 内訳のみを測っており、m=64 形状のカーネル実測は存在しない。候補は机上で決め、#3209 が測る。
- 候補は `tile::M64_THIN_CANDIDATES`（`CANDIDATES` とは別配列）、ゲートは `tile::M64_THIN_TILE_ENABLED`（**既定 false**）。シェーダは変更しない（REQ-8 の手動境界検査は既存カーネルのまま）。tolerance・baseline・FMA 契約は不変。

## 2. 実際に走る経路（コード読解。origin/main `d731babdf`）

| 形状（Metal 側の向き） | 用途 | 現行で実際に通るカーネル |
|---|---|---|
| (64, 256, 784) NN | L1 forward | reuse／infer: スカラー `gemm_tiled_bias_act`（16×16）。fresh: `dispatch_auto` → split-K（机上計算） |
| (64, 10, 256) NN | L2 forward | 同上 |
| (784, 64, 256) NT | L1 d_input | スカラー `gemm_tiled_bias_act`（strided） |
| (256, 64, 10) NT | L2 d_input | 同上 |
| (784, 256, 64) TN | L1 dW | 同上 |
| (256, 10, 64) TN | L2 dW | 同上 |

結論: train reuse と infer の Metal セルでは、m=64 の 6 形状は `tile::select`／`CANDIDATES` を通らない。ゲートが効くのは `dispatch_auto` の classic 分岐だけで、split-K が先に取る形状と常駐 strided 経路には届かない。

### イシュー前提の訂正

- 本番は `params: None`（occupancy 縮退なし）。(64,256,784) が classic に来た場合の現行選択は `CANDIDATES[2]`（32×64）で `[3]` ではない。
- 「32×32 で 16 TG」は split-K のタイル（`split_k_tile`）に当たる。
- 結合順序を変えない tile.rs 候補を足しても、既定経路の train reuse／infer セルには届かない。

## 3. 型 A 候補（結合順序を変えない。本イシューで実装）

| 名前 | bm×bn | bk | wm,wn | acc（行×列） | 狙い |
|---|---|---|---|---|---|
| T1 `[0]` | 16×32 | 16 | 2,2 | 1×2 | 横長・短辺 64。64×256 で 32 TG（現行 `[2]` は 8 TG） |
| T2 `[1]` | 16×64 | 16 | 2,2 | 1×4 | 横長。16 TG |
| T3 `[2]` | 32×16 | 16 | 2,2 | 2×1 | 縦長・短辺 64。784×64 で 100 TG（現行 `[1]` は 26 TG） |
| T4 `[3]` | 16×16 | 16 | 2,2 | 1×1 | 最大 TG 数 |

ゲート ON 時の写像（`select_m64_thin_with`）: m,n,k が全て 64 以上かつ min(m,n)==64・max(m,n)>=128 のとき、横長は T1、縦長は T3。T2・T4 は明示指定（`dispatch_variant`／`dispatch_strided_tiled_prepared`）で測る。

## 4. bit 一致の確認結果（実機）

- 機体: Apple M4 Max（開発機）、ベース `0ae3f00ff` + 本変更。
- コマンド: `cargo test -p fandhe-ai-backend-metal --release --lib gemm_m64_thin_tile_diag_tests -- --ignored --nocapture --test-threads=1`
- 比較: 非 NaN は `to_bits` 完全一致、NaN は NaN 同士。入力に ±inf・NaN・極端値を混入。
- 結果: NN 44 件（11 形状 × 4 候補。端数形状を含む）pass、NT/TN 48 件（6 形状 × 2 向き × 4 候補。戻り値の解決済み構成が要求した候補と一致することも検査）pass。全候補が型 A として bit 一致。
- 性能は未計測。計測は #3209（Mac mini）。

## 5. 型 B（結線せず所有者へ申し送り）

- B1: 常駐 strided 経路（スカラー `gemm_tiled_bias_act`。逐次 fma）を `gemm_simdgroup_tiled` へ回す。演算方式が変わり bit 一致は保証されず、simdgroup 側の strided 経路には bias／relu の epilogue もない。train／infer セルに効きうる唯一の候補。数値判定方式と結線可否は所有者判断。
- B2: split-K の適用範囲拡張（strided 経路への適用、m=64 向け partitions／タイル変更）。結合順序が変わるため `docs/backend-metal-splitk-parity-judgment-decision.md` の全形状一律 baseline の対象で、baseline 追加は実機実測値のみ・人間承認が必要。

## 6. #3209 への注記

- ゲートは `crates/backend-metal/src/tile.rs` の `pub(crate) const M64_THIN_TILE_ENABLED: bool = false;` の 1 行を書き換える（`TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` と同書式）。
- 上記のとおり既定経路の train reuse／infer には届かないため、純カーネル時間は `MetalGemm::dispatch_variant(GemmVariant::SimdgroupTiled(T))` と `dispatch_strided_tiled_prepared(.., cfg = T)` で候補を明示指定して測る。strided は m,n,k が 8 の倍数・ld が 4 の倍数の形状のみ適格。

## 7. Mac mini A/B（#3209）

**状態: 計測基盤と事前登録規則のみ出荷。Mac mini 実測は未実施（unavailable）。** 候補・ゲートの判定はすべて
undetermined とし、数値は埋めない（RULE 項 15）。実測ログと判定は、下記「再現」を Mac mini で実行して本節へ追記する。

### 7.1 条件

| 項目 | 値 |
|---|---|
| 機体 | Mac mini（Apple M4・GPU 10 コア・16 GB）。**未実測のため env_info なし** |
| 規則 | `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/RULE.txt`（sha256 `d5bfebcacf27b6b95da47297cc71ba66f2ce3e44a76d3e50c107e8cd26a26e40`。固定コミット `b5da0c2b0f4a080edf183ce7cb7fa6c7eb922d2b`。計測前に単独コミット） |
| 判定の根拠 | 純カーネル時間（`kernel_gpu`）のみ。train／infer セルは A/A の対照（record_only） |
| 計測基盤 | `gemm_m64_thin_tile_ab_diag_tests.rs`（到達追跡・CPU 参照 parity・純カーネル計測。全件 `#[ignore]`）・`MetalGemm::diag_encode_strided_tiled`（`#[cfg(test)]`。encode-only）・`orchestrate_macmini.sh`・`aggregate.py`・`run_ab_m64_thin_tile_metal.sh` |

### 7.2 開発機でのスモーク（判定に使わない）

M4 Max（`verified_tile_table=M4Max`）で診断 3 テストを 1 回ずつ実行し、行の書式・assert（1 バッチ・1 ラベル・
`kernel_gpu <= commit_wait`）・CPU 参照 parity（候補 × 8 セルすべて fallback なしで pass）・
`aggregate.py` の正規表現との一致を確認した。M4 Max は Mac mini と性能を直接比較できないため、値は収録しない。

### 7.3 実装上の確認事項

- NN の 8 非整列形状（n=10 または k=10 の 3 形状）は、本番 `dispatch_auto` と同じく 8 の倍数へゼロ詰めした実効次元
  （(64,16,256)・(256,64,16)・(256,16,64)）で計測する（`diag_encode_tiled_nn` が非整列を拒否するため）。
- 到達追跡は実効次元で行う。机上の前提（ゲート ON で変わるのは (64,256,784)→T1 と (784,64,256)→T3 の 2 件）が
  食い違う場合、ゲート判定は undetermined に落ちる（RULE 項 5・13）。

### 7.4 結果（未実測）

| 項目 | 結果 |
|---|---|
| 前提ゲート（既存 bit 一致 92 件・CPU 参照 parity） | 未実施 |
| 到達表 | 未実施 |
| 候補 × 形状の `kernel_gpu` | 未実施 |
| 候補別の判定（T1〜T4） | undetermined |
| ゲート（`M64_THIN_TILE_ENABLED` の写像）の判定 | undetermined |
| train／infer セル A/B（A/A 対照） | 未実施 |

### 7.5 所有者への申し送り

- ADOPT 候補が出た場合の結線（`M64_THIN_TILE_ENABLED` の反転や選択表への追加）は、結合順序が変わらない候補に限って
  別 PR で行う。ゲートの写像は機体非依存（M4 Max にも効く）ため、結線の前に M4 Max での確認が要る。
- 型 B（B1: 常駐 strided のスカラー経路の simdgroup 化、B2: split-K の拡張）は結合順序・演算方式が変わるため、
  引き続き所有者判断（baseline の追加は実機実測値のみ・人間承認）。
- `docs/backend-metal-command-batching-design.md` §7.3.4（Mac mini の L1 `kernel_gpu` が未実測）への反映は、
  実測後に本節を参照する 1 行で足りる（本 PR では行わない）。
