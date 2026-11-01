# 対応表「部分的」29 項目の追跡表

<!--
役割: 対応表（スコアボード「役割・機能の対応表」）で fandhe-ai 列が「部分的」の 9 行・29 項目を、
0.14.0 までにすべて「ある」にするための進捗の一元表。設計 → 承認 → 準備 → 実装 → 実機 parity → 再監査の
状態が issue と PR に散らばるのを防ぐ。各 PR が自分の行を同じ PR 内で更新する（§4）。
呼び出し文脈: Phase 11 の #3869（0.14.0 事前監査）と #3870（実機 parity 付録表）がこの表を入力にする（§5）。
機械集計: 自動集計スクリプトを入れるかは #3781 の判断。本書は表を 1 つだけ・ヘッダ固定・1 行 1 項目・
セル内に縦線を入れない・複数値は全角「、」で区切る形にして、判断がどちらでも壊れないようにしてある。
-->

親イシュー #3712（Phase 1）・ルート #3711。本書のイシューは #3717。

## 0. 位置づけと出典

- 29 項目の正本は 0.12.0 監査 `docs/perf/framework-compare-feature-matrix-0.12.0.md` の §8.5（所有者判断の反映）にある「0.12.0 確定件数表」と、
  棚卸し `docs/compat-unlisted-holds-inventory.md` の §4（群 A〜C の項目番号 A1〜A3-g・B1〜B5・C1〜C9）である。本書はそれを項目別の進捗表へ展開したもので、**件数の正本は書き換えない**。
- 行判定の規則は同監査 §2 の冒頭規則（行内の「未公開」0 件かつ「保留」0 件で「ある」）と、規則 3（`publish = false` の facade 外クレートは「公開」に数えない）に従う。
- 「対象外」と「公開」は追跡対象外（0.12.0 の件数表が正）。本書が追うのは、0.12.0 時点で「未公開」16 件・「保留」14 件の計 30 件（重複除去で 29 項目）である。
- 決定記録の雛形は `docs/decision-record-template.md`（#3716。§7 が本書の更新を求める）。0.14.0 の公開 API 互換方針は `docs/api-compat-0.14.0-decision.md`（#3713・#3714）が扱う。
- 基準 sha: 本書の初版は main の `49df686ca` を基準に作った。「正ガード」列の `path:line` はこの sha のもので、行番号は編集で動くため、更新する PR が現行の行に直す。

## 1. 追跡表

凡例は §3。連番 # は本表が振る通し番号 1〜29 であり、issue 番号ではない。各項目の推奨（実装／対象外）と互換の区分の集約は `docs/api-compat-0.14.0-decision.md` §12.1 を正とする（#3764）。

| # | 項目 | 対応表の行 | 0.12.0 の区分 | 設計記録 | 承認 | 実装 issue | 正ガード | 実機 parity | 0.14.0 の区分 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 低精度演算の `Var` 以外の配置 | dtype | 未公開 | #3721（`docs/autodiff-low-precision-op-extension-decision.md` §8） | 承認済み（案 X「対象外」を承認。所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674） | #3784（承認された場合のみ） | 現状は否定ガード `VarLowPrecisionOpsHoldDoctestGuard`（`crates/facade/src/lib.rs:5963`）。反転先は設計記録で確定 | 該当 issue なし（要否は #3721 で判断） | 未確定 |
| 2 | `TapeF64` の `jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` | dtype＋自動微分 | 保留 | #3722（`docs/autodiff-functional-transforms-design.md` §35）、#3723（同 §36） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674） | #3785、#3786、#3787、#3788 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 3 | `create_graph` の C 区分 24 Op | 自動微分 | 保留 | #3724〜#3728（#3724 は docs/autodiff-higher-order-grad-decision.md §34、#3725 は同 §33、#3726 は同 §35、#3727 は同 §36、#3728 は同 §37）、#3739 は同 §48 | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013） | #3789〜#3796 | 未特定（設計記録で確定） | 未実測（#3856 は GB10、#3858 は Mac mini。雛形は #3776、#3777） | 未確定 |
| 4 | `create_graph` の B 区分の未実装 Op | 自動微分 | 保留 | #3729〜#3735（#3729 は docs/autodiff-higher-order-grad-decision.md §38、#3730 は同 §39、#3731 は同 §40、#3732 は同 §41、#3733 は同 §42、#3734 は同 §43、#3735 は同 §44）、#3739 は同 §48 | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013） | #3797〜#3808、#3812 | 未特定（設計記録で確定） | 未実測（#3857 は GB10、#3859 は Mac mini） | 未確定 |
| 5 | 二階微分の拒否範囲と意味論（LayerNorm・RmsNorm・BatchNorm、Det・Inv・Solve・SlogdetLogAbsDet、Embedding の `padding_idx`） | 自動微分 | 保留 | #3736〜#3738（#3736 は docs/autodiff-higher-order-grad-decision.md §45、#3737 は同 §46、#3738 は同 §47）、#3739 は同 §48 | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013） | #3809〜#3811、#3812 | 未特定（設計記録で確定） | 未実測（#3857 は GB10、#3859 は Mac mini） | 未確定 |
| 6 | GPU 線形代数の opt-in 関数（`set_cuda_gpu_linalg_enabled`、`set_metal_gpu_linalg_enabled`） | 演算の範囲 | 未公開 | 既存 `docs/backend-gpu-linalg-decision.md` §11 | 公開形は #3140 のコメント `-6227907703` で承認済み。前提の G2（#3138・#3139 の実機 parity）は充足〈2026-10-28。`docs/backend-gpu-linalg-decision.md` §11.1〉。facade 経由の実測 #3866 は未実測 | #3405 | 否定ガード `facade_does_not_expose_gpu_linalg_optin`（`crates/facade/tests/api_surface.rs:41969`） | 未実測（#3866） | 未確定 |
| 7 | `add_upsample` の `scale_factor` 指定 | NN 層・モデル構築 | 未公開 | #3740（`docs/autodiff-spatial-layers-decision.md` §9）、#3750（`docs/compat-new-layers-io-export-decision.md` §10〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3814 | 未特定（設計記録で確定） | 未実測（#3860 は GB10、#3861 は Mac mini。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 8 | 独自 `Module` から組み込み層を呼ぶ経路 | NN 層・モデル構築 | 未公開 | #3741（docs/facade-nn-module-exposure-decision.md §23） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3815、#3816 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 9 | C1: Hardshrink・Softshrink | NN 層・モデル構築 | 未公開 | #3742（`docs/nn-shrink-bilinear-channel-shuffle-decision.md` §1）、#3750（`docs/compat-new-layers-io-export-decision.md` §10〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3817 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 10 | C2: Bilinear | NN 層・モデル構築 | 未公開 | #3743（`docs/nn-shrink-bilinear-channel-shuffle-decision.md` §2） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3818 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 11 | C3: BatchNorm3d（`Var::batch_norm` の rank 5） | NN 層・モデル構築 | 未公開 | #3744（`docs/autodiff-batch-norm-3d-decision.md` §0〜§12）、#3750（`docs/compat-new-layers-io-export-decision.md` §10〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3819、#3820 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 12 | C5: CosineSimilarity | NN 層・モデル構築 | 未公開 | #3745（`docs/autodiff-similarity-distance-decision.md` §0〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3821 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 13 | C6: PairwiseDistance | NN 層・モデル構築 | 未公開 | #3745（`docs/autodiff-similarity-distance-decision.md` §0〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3821 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 14 | C7: ChannelShuffle | NN 層・モデル構築 | 未公開 | #3746（`docs/nn-shrink-bilinear-channel-shuffle-decision.md` §3）、#3750（`docs/compat-new-layers-io-export-decision.md` §10〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3822 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 15 | C9: AdaptiveAvgPool3d・AdaptiveMaxPool3d・LPPool・FractionalMaxPool（4 種で 1 項目） | NN 層・モデル構築 | 未公開 | #3747（`docs/autodiff-pool3d-ops-decision.md` §18）、#3748（同 §19）、#3749（同 §20）、#3750（`docs/compat-new-layers-io-export-decision.md` §10〜§12） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3768#issuecomment-6554564412） | #3823〜#3827 | 未特定（設計記録で確定） | 未実測（#3860、#3861。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 16 | SWA の `FitConfig` 形（`use_swa`・`swa_start`・`swa_lr`） | 最適化・学習ループ | 未公開 | #3751（`docs/autodiff-swa-decision.md` §20） | 承認済み（案 A「対象外」を承認。所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3830（承認された場合のみ） | 現状は否定ガード `SwaHoldDoctestGuard`（`crates/facade/src/lib.rs:4608`）。反転先は設計記録で確定 | 該当 issue なし | 未確定 |
| 17 | A1: EMA と `compile_with_amp` の併用（AMP の skip step） | 最適化・学習ループ | 保留 | #3752（`docs/autodiff-ema-decision.md` §17） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3831 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 18 | A2: EMA の減衰率ウォームアップ | 最適化・学習ループ | 保留 | #3753（`docs/autodiff-ema-decision.md` §18） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3833 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 19 | A3-a: param groups と AMP の併用 | 最適化・学習ループ | 保留 | #3752（`docs/autodiff-param-groups-decision.md` §14。skip 規則は `docs/autodiff-ema-decision.md` §17） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3832 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 20 | A3-b: param groups と `LrSchedule` の併用 | 最適化・学習ループ | 保留 | #3754（`docs/autodiff-param-groups-decision.md` §15） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3834 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 21 | A3-c: param groups の `state_dict` 保存 | 最適化・学習ループ | 保留 | #3755（`docs/autodiff-param-groups-decision.md` §18） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3835、#3836 | 未特定（設計記録で確定） | 該当 issue なし | 未確定 |
| 22 | A3-d: param groups と `DeviceParamStore` 常駐更新 | 最適化・学習ループ | 保留 | #3756（`docs/autodiff-param-groups-decision.md` §19） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3837、#3838 | 未特定（設計記録で確定） | 未実測（#3862 は GB10、#3863 は Mac mini。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 23 | A3-g: `Optimizer` に Adadelta・Adamax・NAdam・RAdam | 最適化・学習ループ | 保留 | #3757（`docs/compat-optimizer-variants-0.14.0-decision.md` §0〜§13） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758） | #3839、#3840 | 否定ガードなし。反転先は設計記録 §1 で確定（正ガード `compat_optimizer_enum_variants_are_exactly_approved_set`。#3839 で追加。基準 sha `02c0e5553`） | 該当 issue なし | 未確定 |
| 24 | ROCm の facade 公開（`Device::Rocm`） | バックエンド・ハード | 保留 | #3758（先行訂正は #3720。`docs/backend-rocm-device-exposure-decision.md` §9 で R0 実装済み・Q3 解決・Q1／Q2 未回答を追記。#3758 の設計は同 §10〈推奨・9 節・論点〉） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469） | #3851〜#3854（承認された場合のみ）。公開準備は #3774 | 保留ガード `device_variant_set_is_fixed_while_3157_on_hold`（`crates/facade/tests/api_surface.rs:43039`。#3853 で `Rocm(_)` 腕を足す形に反転） | 未実測（#3867 は `<rocm-node>`。#3159 は OPEN で MI300X 待ち・2026-10-31T13:18Z 確認） | 未確定 |
| 25 | B1: ONNX モデルの学習（autograd）の facade 公開 | 相互運用 | 未公開 | #3759（`docs/onnx-autograd-decision.md` §8） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3770#issuecomment-6554565145） | #3842、#3843 | 否定ガードなし（拒否側は `scan_unapproved_onnx_pub_items` の allowlist〈`crates/facade/tests/api_surface.rs:1959`〉と `interop_module_exposes_only_approved_onnx_surface`〈`:2424`〉。反転先は設計記録 §8.1。基準 sha `81e23f9ec`） | 未実測（#3864 は GB10、#3865 は Mac mini。雛形は `crates/facade/tests/train_gpu_parity.rs`（#3778）） | 未確定 |
| 26 | B3: ONNX export の `Sigmoid` | 相互運用 | 保留 | #3760（`docs/facade-onnx-export-exposure-decision.md` §19） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3770#issuecomment-6554565145） | #3844 | 否定テスト `sigmoid_is_rejected_as_unsupported`（`crates/onnx-interop/tests/onnx_export_layers_parity.rs:234`）と `sequential_with_sigmoid_is_rejected_with_unsupported_layer`（`crates/facade/tests/interop_onnx_export_sequential.rs:296`）。正への反転先は §19.1 で確定（基準 sha `5834b0958`） | 該当 issue なし | 未確定 |
| 27 | B5: safetensors の f32 以外の型 | 相互運用 | 未公開 | #3761（docs/facade-safetensors-exposure-decision.md §13） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3770#issuecomment-6554565145） | #3845、#3846 | 現状は識別子の完全一致 allowlist interop_safetensors_reexports_exactly_expected_surface（crates/facade/tests/api_surface.rs:2764。基準 5834b0958）が否定側を担う。反転先は同テストの期待集合拡張と interop_safetensors_types_are_reachable_via_facade（:2854） | 該当 issue なし | 未確定 |
| 28 | HF Hub 連携の 2 段目（`hf-hub` の crates.io 公開と数え方） | 事前学習済みモデル | 未公開 | #3762（`docs/hf-hub-integration-design.md` §10〈推奨・9 節・論点〉） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469） | #3847。公開準備は #3773（準備済み。実公開は Phase 11） | facade 側の否定ガードなし。hf-hub 側の `lib_public_surface_is_exactly_the_expected_items`・`new_public_methods_are_present_and_download_stays_internal`（`crates/hf-hub/tests/api_surface.rs`。基準 sha `16d22e832`）を維持し、反転は論点 4 次第 | 該当 issue なし | 未確定 |
| 29 | `caches` で表せない内部状態を持つモデルの speculative decoding・連続バッチング | 推論・サービング | 未公開 | #3763（docs/facade-speculative-decoding-batching-design.md §29） | 承認済み（所有者の回答コメント 2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3770#issuecomment-6554565145） | #3848、#3849 | 否定ガードなし。反転・拡張先は設計記録 §29.1 で確定（inference_module_reexports_exactly_expected_surface〈crates/facade/tests/api_surface.rs:24369〉、speculative_scheduler_public_shape_is_pinned〈:40807〉ほか。基準 16d285c0a） | 該当 issue なし | 未確定 |

注記:

- 項目 2 は dtype 行と自動微分行の両方の「保留」に数えられている（0.12.0 の確定件数表）。そのため 0.12.0 の行別件数の合計は 30 件で、重複を除くと 29 項目になる（§2）。
- 項目 3〜5 は自動微分行の保留 4 件のうち 3 件に対応し、残る 1 件が項目 2 である。
- 「実装 issue」の issue 番号は #3711 の Phase 別表から取った。「承認された場合のみ」の項目は、承認の結果が「対象外」ならその実装 issue を起票・着手しない。
- 「承認」列が「未承認（#NNNN 待ち）」の間は、実装を進める承認も公開の承認も得ていない。「実装を進める承認」は公開の承認ではない（0.12.0 監査と同じ扱い）。
- ホスト名は書かない。計測機はノード表記 `<cuda-node>`（GB10）・`<metal-node>`（Mac mini）・`<rocm-node>`（MI300X）で表す。

## 2. 行別の残件数

0.12.0 確定件数表（`docs/perf/framework-compare-feature-matrix-0.12.0.md` §8.5）の「未公開」「保留」だけを写し、追跡表から数えた値と照合できる形にした。

| 行 | 0.12.0 未公開 | 0.12.0 保留 | 追跡表の該当 # |
|---|---|---|---|
| dtype | 1 | 1 | 未公開は 1、保留は 2 |
| 自動微分 | 0 | 4 | 保留は 2、3、4、5 |
| 演算の範囲 | 1 | 0 | 未公開は 6 |
| NN 層・モデル構築 | 9 | 0 | 未公開は 7〜15 |
| 最適化・学習ループ | 1 | 7 | 未公開は 16、保留は 17〜23 |
| バックエンド・ハード | 0 | 1 | 保留は 24 |
| 相互運用 | 2 | 1 | 未公開は 25、27、保留は 26 |
| 事前学習済みモデル | 1 | 0 | 未公開は 28 |
| 推論・サービング | 1 | 0 | 未公開は 29 |
| 合計 | 16 | 14 | 30 件（# 2 を dtype と自動微分で 2 回数えたため）。重複を除くと 29 項目 |

- 追跡表の「0.12.0 の区分」から数えると、未公開は # 1・6〜16・25・27・28・29 の 16 件、保留は # 2〜5・17〜24・26 の 13 件で、# 2 は dtype と自動微分の両行に数えるため保留は行別合計で 14 件になる。
- 公開・対象外の件数は追跡対象外（0.12.0 の確定件数表が正）。
- 0.14.0 側の同表は、追跡表の「0.14.0 の区分」が「公開」か「対象外」に移るたびに、同じ PR で行別の残件数を数え直して追記する。自動集計に寄せるかは #3781 の判断による。

## 3. 状態語彙と記入規則

| 列 | 取りうる値 | 規則 |
|---|---|---|
| 0.12.0 の区分 | 未公開、保留 | 過去値で固定し書き換えない |
| 設計記録 | issue 番号。設計記録が入ったら doc のパスも併記 | 設計記録 PR がリンクを足す |
| 承認 | 未承認（#NNNN 待ち）、承認済み（所有者の回答コメント URL）、対象外へ（同 URL） | 事実と URL に限る。得ていない承認は書かない |
| 実装 issue | issue 番号 | 実装 PR が進捗を足す。「承認された場合のみ」は承認結果に従う |
| 正ガード | 否定ガード名と `path:line`、反転後は正ガードの名前と `path:line`、特定できなければ「未特定（設計記録で確定）」 | 推測で埋めない。基準 sha を併記する |
| 実機 parity | 未実測、pass（測定日・ノード表記・測定コミット）、fail（緩めず記録）、該当 issue なし | 測っていないものを pass と書かない |
| 0.14.0 の区分 | 未確定、公開、対象外、未公開、保留 | 「対象外」へは承認欄に所有者の承認コメント URL がある項目だけ移せる |

## 4. 運用規則（各 PR の更新義務）

1. **更新責任**: 次の PR は、同じ PR の中で追跡表の該当行を更新する。更新のない PR はレビューで差し戻す。
   - 設計記録 PR（Phase 2）: 「設計記録」列にリンクを足す。記録の推奨（実装する／対象外にする）は承認待ちの文脈で記録側に書き、「0.14.0 の区分」は変えない。
   - 承認の記録（Phase 3）: 承認 issue は所有者の回答コメントで閉じる。その回答コメントの URL を「承認」列へ反映する。反映は承認を受けた後の最初の PR が行う。
   - 実装 PR（Phase 5〜9）: 「実装 issue」の進捗と「正ガード」（否定ガードから正ガードへの反転と、`crates/facade/tests/api_surface.rs` の該当行）を更新する。公開された項目の「0.14.0 の区分」は「公開」にする。
   - 実機 parity PR（Phase 10）: 「実機 parity」に結果を書く。
   - 部品だけの PR・実測前の PR は `Refs #N` を使う。`Closes #N` は完了条件を実測で満たした PR だけが使う。
2. **推測で埋めない**: 実行できなかった欄は「未実測」のままにする。Linux・M4 Max 開発機での代替値や、cfg で 0 本しか実行されなかったテストの結果を pass の根拠にしない。
3. **状態語彙を固定する**: 値は §3 の表の範囲に限る。「対象外」へ移すときは所有者の承認コメント URL を必須とする。承認欄が「未承認」のままの項目は「対象外」にしない。
4. **数値契約を動かさない**: tolerance 定数・parity baseline・ガードレール閾値は本表の更新と無関係に変えない。baseline の追加・更新は実機の実測値に限り、所有者の承認を要する（`.claude/rules/coding-rust.md`）。
5. **公開クレートの数え方**: 規則 3（`publish = false` の facade 外クレートは「公開」に数えない。0.12.0 監査 §2）を 0.14.0 でどう読むかは Phase 3 の回答（#3771）に従う。項目 24（ROCm）と 28（HF Hub）は回答までは「未確定」とする。
6. **編集競合を避ける**: 1 PR は自分の行のセルだけを触り、ほかの行を整形しない。並列 PR で競合したら rebase して双方の更新を残す。
7. **索引の更新**: 本書を新設したときだけ `docs/README.md` の索引へ載せる。以後の行更新で索引の注釈は書き換えない。

## 5. 0.14.0 監査（Phase 11）での使い方

#3869 の入力手順は次のとおり。

1. 基準 sha を固定し、追跡表の全 29 行の最終状態を読む。「0.14.0 の区分」が「未確定」のまま残る行は、監査の前に所有者へ報告する。
2. 追跡表から行別の残件（未公開・保留）を数え、新しい 0.14.0 版の件数表と突き合わせる。一致しなければ、追跡表の更新漏れか監査の誤りかを監査側で特定する。
3. 各行の「未公開」「保留」が 0 件であることを `path:line` で示す。示せない行は緩めず「部分的」と記録し、所有者へ報告して次のツリーへ引き継ぐ。
4. 「対象外」へ移した項目は、追跡表の承認 URL を根拠に引く。
5. 実機 parity の付録表（#3870）は「実機 parity」列を入力にする。「未実測」は未実測のまま付録へ写す。
6. 追跡表の最終版は監査 PR で凍結する（基準 sha と日付を追記する）。

## 6. 自動集計スクリプトの扱い

追跡表の自動集計（行別残件の数え上げ）を入れるかどうかは #3781 の判断（任意）に委ねる。本書は判断を先取りしない。#3781 が入れる場合に備え、表は §0 冒頭のコメントのとおり機械可読な形で保っている。
