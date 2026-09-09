# 帰属表（イシュー #1309）

v0.7.0（crates.io 公開ピン）↔ origin/main `797030e`（本イシュー実装時点の HEAD）の
`crates/backend-metal/src`・`crates/facade/src`・`crates/autodiff/src`・
`crates/tensor-core/src` 差分（`diff_v0.7.0_797030e_metal_path.txt`。28 ファイル・
+11253/-363 行）を、本番 NN 正方 GEMM reuse 経路
（`Var::matmul` → `MetalBackendOps::gemm` → `dispatch_auto` →
`tile::select_for_device` → `gemm_simdgroup_tiled`）への結線有無で分類する。

## 依存イシューごとの結線状態

| イシュー | 内容 | 本番 NN 正方 GEMM 経路への結線 |
|---|---|---|
| #1280 | E5（`FINE_BARRIER_ENABLED`／`SWIZZLE_ENABLED` の A/B。#1278/#1279 実測） | 既定 `false` のまま（`tile.rs`）。結線対象なしと確定 |
| #1302 | E2（`SOURCE_SPECIALIZATION_ENABLED=false`）・E3（`tile::FRAG_LOAD_CONFIG` 不変）・E4（`tile::COOP_LOAD_CONFIG` 不変）・E6（`tile::TILE_CLASS_MODE=Legacy`）・E7/E8 候補（`select` 非到達） | いずれも REJECT・`select_for_device` 不変（`docs/perf/metal-gemm-n4096-kernel-gap.md` §18・#1304） |
| #1308 | split-K（`docs/backend-metal-splitk-decision.md`）。9/12 点で「採用検討推奨」だが**実装自体は別 issue へ切り出し提案** | `tile::select` に K 分割分岐なし。コード変更なし（PR #1466 本文で確認） |
| #1368 | E9 hfrag 候補（`gemm_simdgroup_tiled_hfrag`。opt-in カーネル追加） | `dispatch_auto` 非到達（opt-in API 未実装。`docs/perf/metal-gemm-hfrag-candidate.md` §9） |
| #1334 | 借用ビュー readout（#1335〜#1337）。Metal は runtime 判定で legacy 経路維持 | `readout_uses_borrowed_view("metal") == false`（`bench-fandhe/src/main.rs`。§13.5 で ADOPT 保留） |

## 結論

上記 5 依存イシューはいずれも本番 NN 正方 GEMM reuse 経路（`tile::select_for_device`
の選択構成）を変更していない。diff の大部分（`gemm.rs`・`gemm.metal`・
`tile.rs` の増分）は opt-in candidate カーネル・function constant 分岐・診断
テストの追加であり、既定経路（`select_for_device` が選ぶ候補・`TileClassMode::Legacy`・
`UNROLL_ACC_ENABLED=false` 等）自体は不変。

`autodiff/src/optim/device_store.rs`・`var.rs`・`facade/src/lib.rs`・
`tensor-core/src/*` の増分は #1212/#1216/#1217 系のデバイス常駐化・
`linear_forward_device` 等の API 追加によるもので、単純な NN GEMM（`matmul`）
の reuse 計測経路には非到達（`bench-fandhe gemm metal <N> reuse` は
`Var::matmul` を直接叩く単一 op 計測であり、Sequential 推論チェーンや
optimizer step の経路を通らない）。

**結論**: 本番 NN 正方 GEMM reuse 経路は v0.7.0 と HEAD で実質同一（見込み）。
正式系列（registry 解決 `fandhe-ai =0.7.0`）と参考系列（HEAD `797030e` を
`crates/facade` へ path patch）で大きな差が出た場合は、上記の理由により
コード変更へ帰属せず、まず負荷差（load average の系列間変動）を疑う
（§3 事前宣言規則の 8）。
