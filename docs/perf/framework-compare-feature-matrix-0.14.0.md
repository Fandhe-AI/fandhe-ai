# フレームワーク横並びスコアボード「役割・機能の対応表」v0.14.0 事前監査

イシュー #3869（親 #3868・ルート #3711）。スコアボード「役割・機能の対応表」の fandhe-ai 列で 0.12.0 時点に「部分的」だった 9 行
（dtype／自動微分／演算の範囲／NN 層・モデル構築／最適化・学習ループ／バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービング）を、
0.14.0 の公開前に **main の公開面**で事前監査し、各行の根拠を「公開／未公開／対象外／保留／実機 parity」の 5 区分に分けて記録する。
0.12.0 版（`docs/perf/framework-compare-feature-matrix-0.12.0.md`・イシュー #3269）と同じ節構成で、入力は追跡表 `docs/compat-matrix-ari-tracker.md`（#3717）である。
**本書は crates.io 公開版ではなく main の事前監査である**。スコアボードへの反映・タグとの差分確認は本書の範囲外で、実機 parity の付録表は #3870 の担当である（§5）。

**結論**: イシュー題名の「9 行すべての未公開・保留が 0 件」は**達成できていない**。「ある」は 7 行（dtype・演算の範囲・NN 層・モデル構築・最適化・学習ループ・相互運用・事前学習済みモデル・推論・サービング）、
「部分的」のまま残るのは 2 行（自動微分・バックエンド・ハード）である。いずれも追跡表の「未確定」2 件（# 4・# 24）が原因で、緩めずその形で記録し、所有者への報告事項に挙げた（§6）。
「未公開」は 9 行すべてで 0 件である。

## 1. 基準面の証明

判定基準は main の公開面（`crates/facade` と、追跡表が公開に数える補助クレート）とする。本書の `path:line` は、すべて `git show 1cbb475989e32051b337efb81b38b843de80145b:<path>`（基準 sha 時点の内容）で再解決できる行番号である。
追跡表や 0.12.0 版の行番号は編集で動くため写さず、基準 sha で引き直した。

- **基準 sha**: `1cbb475989e32051b337efb81b38b843de80145b`（`git rev-parse origin/main`。2026-11-05 に固定。以後 main が進んでも動かさない）。docs/spec の pin は `3e9ea28f4b66c77c348a0cf4cf3ba08bcc4522ab`（`git ls-tree` の結果）。
- ワークスペースの版数は `Cargo.toml:149` の `version = "0.13.0"` で、タグ `v0.14*` は**未作成**（`git tag -l 'v0.14*'` が空）。版数の公開（publish）とピン更新は別の明示指示を待つ。
- 直近の公開版との差は下のとおり。0.12.0 版 §1 の 487 files（v0.11.0 比）から、公開済みの v0.13.0 比で 283 files になった。
- 保留ガード（`*HoldDoctestGuard`）の struct 名の集合は v0.12.0 と基準 sha で 43 件のまま同一である（`crates/facade/src/lib.rs` から `^struct \w+HoldDoctestGuard` を抽出して比較）。0.12.0 §2 規則 2 と同じく、struct 名の有無は未公開の根拠にしない。
- `docs/compat-unlisted-holds-inventory.md` は v0.12.0 から差分が無い（`git diff --stat v0.12.0 HEAD` が空）。
- 実機 parity の測定コミットは基準 sha と別で、「測定済み」はその測定コミットの内容が pass したことだけを意味する。実機は動かしていない。本書の調査は main 上のソース・決定記録・GitHub のコメントの読み取りだけで行った。

```
$ git diff --shortstat v0.13.0 1cbb475989e32051b337efb81b38b843de80145b -- crates
 283 files changed, 92390 insertions(+), 2064 deletions(-)
$ git diff --shortstat v0.12.0 1cbb475989e32051b337efb81b38b843de80145b -- crates
 315 files changed, 96850 insertions(+), 2139 deletions(-)
```

## 2. 判定規則

0.12.0 版 §2 の規則 1〜6（件数表記・保留ガードの扱い・facade 外クレート・承認の出所・実機未検証・保留の定義）を**そのまま踏襲し、緩めていない**。行内の「未公開」0 件かつ「保留」0 件の行だけを「ある」とする。
追加点は次の 3 つで、いずれも判定の緩和ではない。

1. **規則 3（facade 外クレート）の 0.14.0 での読み方**: 根拠は #3771 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469>）だけである。
   - `hf-hub` は案 A で、`HfHub::download` は `pub(crate)` のまま「公開に数える」。`docs/compat-api-scope.md` §0 の限定例外として扱う。
   - `backend-rocm` の crates.io 公開は 0.14.0 で承認済みである。ただし ROCm の facade 同梱は #3159・#3867 の parity pass が条件である（回答の点 1）。
   - 両クレートとも `publish = true` になった（`crates/hf-hub/Cargo.toml:36`・`crates/backend-rocm/Cargo.toml:33`）ため、規則 3 の文面の条件（`publish = false`）にはもう当たらない。**crates.io にはまだ公開していない**。
2. **「未確定」の数え方**: 追跡表の「未確定」は新しい区分を作らず、0.12.0 での区分（# 4・# 24 とも「保留」）のまま数える。「公開」には上げない。
3. **承認の出所（規則 4）への追加**: 0.12.0 版の出所に、次の所有者コメントを加える（作成者・日時は `gh api repos/Fandhe-AI/fandhe-ai/issues/comments/<id>` で確認した。いずれも `aLiz-Nancy`。本文は転記せず要約する）。
   - Phase 3 の回答 6 件（2026-11-01）: #3766 `-6554563674`（06:42:58Z。低精度演算の配置と `TapeF64` の `jvp` 系）、#3767 `-6554564013`（06:43:00Z。`create_graph` の B／C 区分と拒否範囲）、#3768 `-6554564412`（06:43:03Z。NN 層）、
     #3769 `-6554564758`（06:43:05Z。最適化）、#3770 `-6554565145`（06:43:08Z。相互運用と推論）、#3771 `-6554565469`（06:43:11Z。ROCm と HF Hub）。
   - #3711 のコメント 3 件: `-6570118646`（2026-11-02。運用の読み替え）、`-6594034494`（2026-11-03。#3977・#3978 の起票）、`-6604937225`（2026-11-03。#3978 への回答と、行 4 の「実装と実機 parity が確定するまで未確定」という条件）。
   - 「承認」はこれらのコメント ID と各イシュー番号の事実に限って書き、得ていない承認は書かない。

## 3. 対応表

セル内の `D:` は `docs/` 配下の決定記録を指す。「0.12.0 時点」の列は 0.12.0 版 §3 の判定を、その §8.5 の確定件数と §9 の訂正を踏まえて要約したものである。追跡表の項目は「# n」で示す。

| 行 | 0.12.0 時点 | 0.14.0 事前監査判定 | 根拠（公開／未公開／対象外／保留／実機 parity） |
|---|---|---|---|
| dtype | 部分的（未公開 1・保留 1） | ある | **公開（6 件）:** f64 の自動微分 `TapeF64`（`crates/facade/src/lib.rs:2781`）／`VarF64`（`crates/facade/src/lib.rs:3029`）；低精度 forward の opt-in `matmul_low_precision`（`crates/autodiff/src/var.rs:3659`）；dtype 変換と型付き演算 `CastDType`（`crates/facade/src/lib.rs:428`）・`TypedOps`（`crates/facade/src/lib.rs:440`）・`typed_ops_f64`（`crates/facade/src/lib.rs:2668`）；AMP `compile_with_amp`（`crates/facade/src/compat/training.rs:2306`）；CUDA GEMM の精度 opt-in `set_cuda_gemm_precision`（`crates/facade/src/lib.rs:3909`。既定は FP32 のまま）；# 2 の `TapeF64` の `jvp`（`crates/facade/src/lib.rs:2887`）・`jacfwd`（`crates/facade/src/lib.rs:2919`）・`jvp_multi`（`crates/facade/src/lib.rs:2959`）・`jvp_create_graph`（`crates/facade/src/lib.rs:3010`）と f64 の `backward_create_graph`（`crates/facade/src/lib.rs:2845`）。正ガードは `f64_autograd_public_surface_is_exactly_approved_delegations`（`crates/facade/tests/api_surface.rs:41871`）と `f64_jvp_family_usage_doctest_is_present_and_compiled`（`crates/facade/tests/api_surface.rs:46580`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の未公開 1 件（# 1）は対象外へ移った<br>**対象外（4 件）:** # 1 `Var` 以外への低精度演算の配置（承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674>。禁止経路専用で `VarLowPrecisionOpsHoldDoctestGuard`〈`crates/facade/src/lib.rs:7079`〉を維持し、正ガードは `var_low_precision_ops_hold_doctest_globs_all_pub_modules`〈`crates/facade/tests/api_surface.rs:29041`〉・`var_low_precision_ops_hold_doctest_probe_body_matches_fixed_contract`〈`crates/facade/tests/api_surface.rs:29065`〉・`facade_does_not_reexport_or_declare_low_precision_ops`〈`crates/facade/tests/api_surface.rs:29232`〉）；f16／bf16 を値型とする `Tape`／`Var`（#3021 の `#issuecomment-6111746880` の項目 2。0.12.0 §3 から不変）；sparse／complex（#2499 の `#issuecomment-6033824965`。0.12.0 §3 から不変）；int8 量子化（spec の除外事項「Won't・条件付き」。0.12.0 §3 から不変）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 1 件（# 2。`TapeF64` の `jvp` 系）は上のとおり公開へ移った。承認は <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674><br>**実機 parity:** f64 と低精度 opt-in は 0.12.0 §5 のとおり実測済み（pass）。# 2 の追跡表の欄は「該当 issue なし」、# 1 も「該当 issue なし」（新しい数値経路なし）。int8 は性能判定対象外 |
| 自動微分 | 部分的（保留 4） | 部分的 | **公開（15 件）:** 0.12.0 の 12 件（`backward_create_graph`〈`crates/facade/src/lib.rs:1623`〉、`custom`〈`crates/facade/src/lib.rs:2389`〉、`register_backward_hook`〈`crates/facade/src/lib.rs:2447`〉、`jacobian`〈`crates/facade/src/lib.rs:1706`〉・`hessian`〈`crates/facade/src/lib.rs:1743`〉、`gradcheck`〈`crates/facade/src/lib.rs:2299`〉・`backward_detect_anomaly`〈`crates/facade/src/lib.rs:2266`〉、`vjp`〈`crates/facade/src/lib.rs:1793`〉・`hvp`〈`crates/facade/src/lib.rs:1851`〉・`vmap`〈`crates/facade/src/lib.rs:2169`〉、`jvp`〈`crates/facade/src/lib.rs:1925`〉・`jacfwd`〈`crates/facade/src/lib.rs:1962`〉、`jvp_multi`〈`crates/facade/src/lib.rs:2030`〉、`jvp_create_graph`〈`crates/facade/src/lib.rs:2114`〉、`vmap_grad`〈`crates/facade/src/lib.rs:2233`〉、`set_deterministic`〈`crates/facade/src/lib.rs:3329`〉、`var_no_grad`〈`crates/facade/src/lib.rs:638`〉・`backward_accumulate`〈`crates/facade/src/lib.rs:657`〉・`detach`〈`crates/autodiff/src/var.rs:278`〉・`checkpoint_from`〈`crates/autodiff/src/var.rs:2502`〉。宣言行を基準 sha で引き直し、すべて存在する）；# 2 `TapeF64` の `jvp` 系（dtype 行と同じ）；# 3 `create_graph` の C 区分 24 Op（実装は #3789〜#3796。対象 Op 集合の正ガードは `create_graph_supported_op_set_matches_approved_list`〈`crates/facade/tests/api_surface.rs:47984`〉、facade 挙動テストとの対応表は `create_graph_facade_coverage_map_matches_supported_set`〈`crates/facade/tests/api_surface.rs:48174`〉、定義元は `supports_create_graph`〈`crates/autodiff/src/tape.rs:2654`〉）；# 5 二階微分の拒否範囲と意味論（`Inv`・`Solve`・`SlogdetLogAbsDet` の overflow 入力の拒否を `inv_solve_overflow_inputs_are_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2741`〉が固定）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（4 件）:** D-0 の除外名（`Var::custom` 系・`functional_ops` の自由関数・gradcheck の自由関数。#3018 の `#issuecomment-6111643437`）；ネイティブ forward-mode の JVP 規則とバッチ規則型 vmap（#2499 の `#issuecomment-6033824965`）；`jvp` の f16 対応（#3021 の `#issuecomment-6111746880` の項目 3）；`create_graph` が構造的に定義できない Op（`ResidentLeaf`・`LinearResident`・`LinearAct`・`Custom`・`OneHot` と、`CtcLoss` の恒久 fail-closed〈`ctc_loss_create_graph_is_rejected_via_facade`、`crates/facade/tests/create_graph_facade.rs:2661`〉。`Inv`・`Solve`・`SlogdetLogAbsDet` の拒否範囲の恒久化。承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>）。件数は 0.12.0 の 4 件のまま（集計の単位が項目か区分かの差で、増減はない）<br>**保留（1 件）:** # 4 `create_graph` の B 区分（追跡表では「未確定」。0.12.0 での区分は保留）。実装は済んでいる: `Prod` は #3977（`scatter_reduce_prod_create_graph_via_facade`、`crates/facade/tests/create_graph_facade.rs:897`）、`Nuc`・`Spectral` は #3991（`supports_create_graph` の `MatrixNorm` の腕、`crates/autodiff/src/tape.rs:2853`。`matrix_norm_nuc_spectral_create_graph_via_facade`、`crates/facade/tests/create_graph_facade.rs:2031`。縮退入力の拒否は同:2054）。それでも**保留のままにする理由**: #3711 の所有者コメント `-6604937225` が「実装が済み、実機 parity も確定するまで未確定」とし、`Nuc`・`Spectral` の GPU 実機 parity は未実測である（決定記録 `docs/autodiff-higher-order-grad-decision.md:5558`〈§76.5〉。`create_graph_gpu_parity.rs` の `reject_matrix_norm_nuc_degenerate`〈`crates/facade/tests/create_graph_gpu_parity.rs:2594`〉は縮退入力の拒否へ変えただけで、`Nuc`・`Spectral` の実機比較ケースは未追加）。解除条件はこの実機 parity の確定である。承認は <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>。**範囲外の注記（件数に数えない）**: `CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（`cosine_similarity_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2674`〉・`pairwise_distance_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2685`〉。決定記録 `docs/autodiff-higher-order-grad-decision.md` §73〈5388 行〉は「承認済み（#3768 Q9 (a)）・別設計待ち・恒久ではない・起票は所有者の承認事項で未起票」とする）。29 項目の外で、0.12.0 §2 規則 6 の「保留」（所有者の回答が無い論点）には当たらないが、所有者への報告事項に挙げる（§6）<br>**実機 parity:** # 2 は該当 issue なし。# 3・# 5 は pass（2026-11-04・`<cuda-node>` `6852ef6bd` と `48d0263ac`、`<metal-node>` `0c170cee2` と `bbfcaee6b`。#3856〜#3859。記録は `docs/perf/logs/create-graph-c-cuda-parity-3856/`・`docs/perf/logs/create-graph-b-cuda-parity-3857/` と各 metal 版）。# 4 は B 区分の 12 族が pass（同上）だが、`Nuc`・`Spectral` と `Prod` の追加分は**未実測**（`create-graph-b-*` の README に測定ケースが無い） |
| 演算の範囲 | 部分的（未公開 1） | ある | **公開（8 件）:** 0.12.0 の 7 件（`Var` 委譲、`rfft`〈`crates/autodiff/src/var.rs:5112`〉、`median`〈`crates/autodiff/src/var.rs:5356`〉・`kron`〈`crates/autodiff/src/var.rs:5536`〉・`pad_with_mode`〈`crates/autodiff/src/var.rs:5577`〉、`max_pool3d`〈`crates/autodiff/src/var.rs:5594`〉・`conv_transpose3d`〈`crates/autodiff/src/var.rs:5703`〉・`max_unpool2d`〈`crates/autodiff/src/var.rs:5747`〉、`fold`〈`crates/autodiff/src/var.rs:5807`〉・`local_response_norm`〈`crates/autodiff/src/var.rs:5825`〉・`weight_norm`〈`crates/autodiff/src/var.rs:5839`〉・`spectral_norm`〈`crates/autodiff/src/var.rs:5846`〉、`bce_with_logits_loss_with`〈`crates/autodiff/src/var.rs:4953`〉、`bincount`〈`crates/facade/src/lib.rs:1648`〉）；# 6 GPU 線形代数の opt-in `set_cuda_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3839`）・`set_metal_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3866`）。正ガードは `gpu_linalg_optin_is_reachable_via_facade`（`crates/facade/tests/api_surface.rs:44513`）と `workspace_declares_gpu_linalg_optin_fn_names_only_in_approved_locations`（`crates/facade/tests/api_surface.rs:44567`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 6）は公開へ移った<br>**対象外（4 件）:** 0.12.0 §3 から不変（D-0 の除外名、sparse／complex、Metal の eigh・SVD と CUDA の SVD、int／bool 演算の網羅。承認は 0.12.0 版の出所のとおり）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 6 は pass（#3866。facade 経由・ゲート ON。`<cuda-node>` は 6 本、`<metal-node>` は 5 本〈eigh は範囲外〉。2026-11-04。記録は `docs/perf/logs/gpu-linalg-optin-cuda-parity-3866/`・`docs/perf/logs/gpu-linalg-optin-metal-parity-3866/`）。0.12.0 時点で未実測だった `cuda-linalg-*`・`metal-linalg-*` の README の記入欄は、本書では再確認しておらず未実測のまま写す（§5） |
| NN 層・モデル構築 | 部分的（未公開 2。0.12.0 §8.5 の確定は未公開 9） | ある | **公開（15 件）:** 0.12.0 の 6 件（`add_max_pool3d`〈`crates/facade/src/compat/sequential.rs:2423`〉・`add_avg_pool3d`〈`crates/facade/src/compat/sequential.rs:2551`〉・`add_conv_transpose3d`〈`crates/facade/src/compat/sequential.rs:1723`〉・`add_fold`〈`crates/facade/src/compat/sequential.rs:2487`〉・`add_unfold`〈`crates/facade/src/compat/sequential.rs:2455`〉・`add_local_response_norm`〈`crates/facade/src/compat/sequential.rs:2523`〉、MaxUnpool の層経路〈`crates/facade/src/compat/functional.rs:552`〉、`parametrize_weight_norm`〈`crates/facade/src/compat/sequential.rs:838`〉・`parametrize_spectral_norm`〈`crates/facade/src/compat/sequential.rs:903`〉、`save_model`〈`crates/facade/src/compat/model_io.rs:283`〉・`load_model`〈`crates/facade/src/compat/model_io.rs:306`〉と kind 追加の `spec_kind`〈`crates/facade/src/compat/model_io.rs:336`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 7 `add_upsample_scale_factor`（`crates/facade/src/compat/sequential.rs:2978`。正ガード `compat_sequential_spatial_add_methods_have_approved_signatures`〈`crates/facade/tests/api_surface.rs:14947`〉）；# 8 `Sequential` への `nn::Module` 実装（`crates/facade/src/compat/sequential.rs:706`。正ガード `compat_sequential_nn_module_impl_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:48320`〉）；# 9 `add_hardshrink`（`crates/facade/src/compat/sequential.rs:1267`）・`add_softshrink`（`crates/facade/src/compat/sequential.rs:1291`）、`Var` 側は `hardshrink`（`crates/autodiff/src/var.rs:5937`）・`softshrink`（`crates/autodiff/src/var.rs:5953`）；# 10 `bilinear`（`crates/autodiff/src/var.rs:5981`。正ガード `var_bilinear_ops_method_is_thin_delegation`〈`crates/facade/tests/api_surface.rs:48981`〉）；# 11 `add_batch_norm3d`（`crates/facade/src/compat/sequential.rs:1934`）と rank 5 の `batch_norm`（`crates/autodiff/src/var.rs:1859`）；# 12 `cosine_similarity`（`crates/autodiff/src/var.rs:6015`）；# 13 `pairwise_distance`（`crates/autodiff/src/var.rs:6043`）；# 14 `add_channel_shuffle`（`crates/facade/src/compat/sequential.rs:3060`）、`Var` 側は `channel_shuffle`（`crates/autodiff/src/var.rs:3107`）；# 15 `add_adaptive_avg_pool3d`（`crates/facade/src/compat/sequential.rs:2780`）・`add_adaptive_max_pool3d`（`crates/facade/src/compat/sequential.rs:2807`）・`add_lp_pool1d`（`crates/facade/src/compat/sequential.rs:2588`）・`add_lp_pool2d`（`crates/facade/src/compat/sequential.rs:2623`）・`add_lp_pool3d`（`crates/facade/src/compat/sequential.rs:2658`）・`add_fractional_max_pool2d`（`crates/facade/src/compat/sequential.rs:2840`）・`add_fractional_max_pool3d`（`crates/facade/src/compat/sequential.rs:2869`）。新層の保存 kind の正ガードは `model_io_new_layer_kinds_match_approved_mapping_issue_3828`（`crates/facade/tests/api_surface.rs:51413`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 §8.5 の 9 件（# 7〜15）はすべて公開へ移った<br>**対象外（1 件）:** 層型名の公開（`fandhe_ai::nn::*` の全層型。#3022 の決定記録＝公開対象外 (i)+(iv)。`SpatialLayersHoldDoctestGuard`〈`crates/facade/src/lib.rs:4903`〉は維持）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 7・# 9・# 10・# 11 は、追跡表の欄の先頭語が「未実測」のまま残っているが、本文には `<cuda-node>`・`<metal-node>` とも pass の記録がある（#3860・#3861。2026-11-04。記録は `docs/perf/logs/nn-layers-cuda-parity-3860/`・`docs/perf/logs/nn-layers-metal-parity-3861/`。学習ステップの雛形は `docs/perf/logs/nn-train-template-cuda-parity-3860/`・`docs/perf/logs/nn-train-template-metal-parity-3861/` で pass）。# 8 は該当 issue なし。# 12〜15 の欄は追跡表のとおり写す（実機の個別記録は本書で再確認していない） |
| 最適化・学習ループ | 部分的（未公開 1。0.12.0 §8.5 の確定は未公開 1・保留 7） | ある | **公開（14 件）:** 0.12.0 の 7 件（optimizer の `Adamax`〈`crates/facade/src/optim.rs:528`〉・`Asgd`〈`crates/facade/src/optim.rs:529`〉・`Lion`〈`crates/facade/src/optim.rs:537`〉・`Rprop`〈`crates/facade/src/optim.rs:538`〉、`PolynomialLr`〈`crates/facade/src/optim.rs:530`〉・`CyclicLr`〈`crates/facade/src/optim.rs:534`〉、`ParamGroup`〈`crates/facade/src/optim.rs:554`〉、`OptimizerStateDict`〈`crates/facade/src/optim.rs:551`〉、`SwaLr`〈`crates/facade/src/optim.rs:539`〉、`fit_with_weights`〈`crates/facade/src/compat/training.rs:3026`〉・`fit_with_train_step`〈`crates/facade/src/compat/training.rs:2928`〉、`train_step_resident`〈`crates/facade/src/compat/sequential.rs:4550`〉）；# 17 EMA と AMP の併用（正ガード `ema_fit_rejection_arms_match_approved_set`〈`crates/facade/tests/api_surface.rs:20715`〉）；# 18 EMA のウォームアップ（`ema_warmup_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:20946`〉）；# 19 param groups と AMP（`compile_with_amp_and_param_groups`、`crates/facade/src/compat/training.rs:2434`）；# 20 param groups と `LrSchedule`（`compile_with_param_groups`、`crates/facade/src/compat/training.rs:2236`。正ガード `param_groups_usage_doctests_are_present_and_compiled`〈`crates/facade/tests/api_surface.rs:18621`〉）；# 21 param groups の `state_dict` 保存（`save_model`・`load_model`。# 20 と同じ正ガード）；# 22 param groups と `DeviceParamStore`（`init_device_param_store_with_param_groups`、`crates/facade/src/compat/sequential.rs:4315`）；# 23 `NAdam`（`crates/facade/src/optim.rs:546`）・`RAdam`（`crates/facade/src/optim.rs:556`）と `Optimizer` の腕（`crates/facade/src/compat/training.rs:313`・`crates/facade/src/compat/training.rs:317`）、保存 kind（`crates/facade/src/compat/model_io/compiled.rs:251`・`crates/facade/src/compat/model_io/compiled.rs:252`）、正ガード `APPROVED_COMPAT_OPTIMIZER_VARIANTS`（`crates/facade/tests/api_surface.rs:22262`。11 種）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の未公開 1 件（# 16）は対象外へ移った<br>**対象外（2 件）:** `FitConfig` への EMA（D-0 の除外名。#3018 の `#issuecomment-6111643437`。機能は `Callback::Ema`〈`crates/facade/src/compat/callbacks.rs:1750`〉で到達できる）；# 16 SWA の `FitConfig` 形（承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758>。機能は `Callback::Swa`〈`crates/facade/src/compat/callbacks.rs:1752`〉で公開済み。禁止経路専用で `SwaHoldDoctestGuard`〈`crates/facade/src/lib.rs:5721`〉と `swa_hold_doctest_globs_all_pub_modules`〈`crates/facade/tests/api_surface.rs:21058`〉を維持）<br>**保留（0 件）:** なし（0 件）。0.12.0 §8.5 の 7 件（# 17〜23）はすべて公開へ移った<br>**実機 parity:** # 22 は追跡表で未実測（#3862・#3863）。# 23 は該当 issue なし。0.12.0 §5 の `dataset-compose-2661`・`iterable-batch-sampler-2662` は host ユーティリティで記入欄が未実測のまま。学習ステップ常駐チェーンは 0.12.0 §5 のとおり実測済み |
| バックエンド・ハード | 部分的（保留 1） | 部分的 | **公開（3 件）:** `tape_for`（`crates/facade/src/lib.rs:3214`）・`available_devices`（`crates/facade/src/lib.rs:3276`）・`set_cuda_gemm_precision`（`crates/facade/src/lib.rs:3909`）；デバイス間の値転送 `transfer`（`crates/facade/src/lib.rs:895`）；分散学習の初期形 `distributed`（`crates/facade/src/lib.rs:255`）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（3 件）:** 0.12.0 §3 から不変（FSDP・ZeRO 系・tensor／pipeline parallel、cudarc の `nccl` feature、int8 量子化）<br>**保留（1 件）:** # 24 ROCm の facade 公開（追跡表では「未確定」。0.12.0 での区分は保留）。実装は済んでいる: `Device::Rocm(usize)`（`crates/tensor-core/src/device.rs:79`）、`resolve_ops` の `Device::Rocm(_)` の腕（`crates/facade/src/lib.rs:3603`）、provider（`crates/backend-rocm/src/device.rs:32`）、`backend-rocm` は `publish = true`（`crates/backend-rocm/Cargo.toml:33`。crates.io 未公開）。正ガードは `device_enum_variant_inventory_is_fixed`（`crates/facade/tests/api_surface.rs:45740`）・`facade_depends_on_backend_rocm_only_in_approved_shape`（`crates/facade/tests/api_surface.rs:45776`）・`facade_wires_rocm_into_resolve_ops_and_available_devices`（`crates/facade/tests/api_surface.rs:51959`）・`rocm_device_usage_doctests_are_present_and_compiled`（`crates/facade/tests/api_surface.rs:52039`）。それでも**保留のままにする理由**: #3771 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469>）の点 1 が「facade への同梱は #3159・#3867 の parity が pass することを条件とする」とし、両 issue は 2026-11-05 時点で OPEN（`gh issue view` で確認。#3159 は更新 07:19:09Z、#3867 は 07:19:10Z）、MI300X 実機の測定が無い。表記は「条件付き・実験的」（`crates/facade/src/lib.rs:3203`）。解除条件は #3159・#3867 の pass である<br>**実機 parity:** # 24 は `<rocm-node>` で**未実測**（追跡表の欄）。DDP は 0.12.0 §5 の `ddp-host-collective-gb10-3149` が記入欄未実測のまま（本書では再確認しておらず、未実測のまま写す）。CUDA GEMM の既定精度は FP32 のまま |
| 相互運用 | 部分的 → 0.12.0 §8.2 で「部分的」へ戻した（未公開 2・保留 1） | ある | **公開（7 件）:** 0.12.0 の 4 件（ONNX import `OnnxModel`〈`crates/facade/src/interop/onnx.rs:234`〉と export `from_sequential`〈`crates/facade/src/interop/onnx.rs:364`〉、safetensors の f32 入出力、npy／npz、`save_model`／`load_model`）；# 25 ONNX モデルの学習の入口 `bind_autograd`（`crates/facade/src/interop/onnx.rs:914`。正ガード `onnx_autograd_types_are_reachable_via_facade`〈`crates/facade/tests/api_surface.rs:2770`〉）；# 26 ONNX export の `Sigmoid`（`crates/onnx-interop/src/onnx/export_nn.rs:367`。正ガード `onnx_export_sigmoid_matches_approved_shape_issue_3844`〈`crates/facade/tests/api_surface.rs:51699`〉）；# 27 safetensors の f32 以外の型（`crates/facade/src/interop/safetensors.rs:196`・`crates/facade/src/interop/safetensors.rs:197` の再エクスポート。正ガード `interop_safetensors_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:3083`〉）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（2 件）:** 0.12.0 §3 から不変（Python バインディングと TF 系形式、`Tensor` への npy／npz の inherent メソッド）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 1 件（# 26）は公開へ移った<br>**実機 parity:** # 25 は pass（`<cuda-node>` の測り直しを含む。`docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`。#4046）。# 26・# 27 は該当 issue なし（ホスト実行のみ） |
| 事前学習済みモデル | 部分的（未公開 1） | ある | **公開（5 件）:** 0.12.0 の 4 件（`ModelRegistry`〈`crates/facade/src/model.rs:684`〉・`load`〈`crates/facade/src/model.rs:870`〉、`save`〈`crates/facade/src/model.rs:1017`〉、`download`〈`crates/facade/src/model.rs:1234`〉・`download_with`〈`crates/facade/src/model.rs:1287`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 28 HF Hub 連携の補助クレート `hf-hub`（`HfHub`〈`crates/hf-hub/src/lib.rs:342`〉・`resolve_revision`〈`crates/hf-hub/src/lib.rs:385`〉・`list_files`〈`crates/hf-hub/src/lib.rs:430`〉。`download` は `pub(crate)`〈`crates/hf-hub/src/lib.rs:452`〉のまま。`publish = true`、`crates/hf-hub/Cargo.toml:36`。crates.io には未公開）。数え方は §2 の 1（#3771 の案 A）による<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 28）は §2 の読み方で公開に数える<br>**対象外（1 件）:** 重みの同梱（#3021 の `#issuecomment-6111746880` の項目 5。0.12.0 §3 から不変）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 0.12.0 §5 の HTTPS 実通信の確認が実測済み。# 28 は該当 issue なし（ホスト上の HTTP 取得のみ） |
| 推論・サービング | 部分的（未公開 1） | ある | **公開（9 件）:** 0.12.0 の 8 件（`predict`〈`crates/facade/src/compat/sequential.rs:3152`〉・`predict_batches`〈`crates/facade/src/compat/sequential.rs:3376`〉・`predict_resident`〈`crates/facade/src/compat/sequential.rs:4431`〉・`predict_resident_input`〈`crates/facade/src/compat/sequential.rs:4478`〉、`generate`／`AutoregressiveModel`／`GenerateConfig`／`SamplingStrategy`〈`crates/facade/src/inference/mod.rs:463`〉、`new_paged_kv_cache`〈`crates/facade/src/lib.rs:2622`〉・`paged_attention_forward`〈`crates/facade/src/lib.rs:2642`〉、`BatchScheduler`〈`crates/facade/src/inference/mod.rs:472`〉、`generate_speculative`〈`crates/facade/src/inference/mod.rs:476`〉、`StringLookup`〈`crates/facade/src/text/mod.rs:151`〉、`set_cuda_graph_infer_enabled`〈`crates/facade/src/lib.rs:3763`〉・`set_cuda_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3795`〉・`set_metal_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3809`〉）；# 29 内部状態を持つモデルの speculative／連続バッチング（`StatefulAutoregressiveModel`・`generate_speculative_stateful` の再エクスポート〈`crates/facade/src/inference/mod.rs:485`〉、trait〈`crates/autodiff/src/generate/stateful.rs:50`〉、`generate_speculative_stateful`〈`crates/autodiff/src/generate/stateful.rs:150`〉、`step_stateful`〈`crates/autodiff/src/generate/scheduler.rs:687`〉。正ガードは `inference_module_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:25893`〉・`speculative_scheduler_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43252`〉・`stateful_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43326`〉）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 29）は公開へ移った<br>**対象外（5 件）:** 0.12.0 §3 から不変（汎用グラフコンパイル区分 A、HTTP／API サーバと量子化 KV、サブワードトークナイザ、beam search 等の生成機能、int8 量子化）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 29 は新しいカーネル・バックエンド経路が無く、実機 parity の対象外（追跡表の欄は「該当 issue なし」）。speculative とスケジューラ・paged attention は 0.12.0 §5 のとおり実測済み。`speculative-sampling-3102` は記入欄が未実測のまま（本書では再確認していない） |

## 4. 0.12.0 列からの差分

### 4.1 件数の集計と行判定

「ある」は未公開 0 件かつ保留 0 件の行だけで、機械的に決めた（§2）。公開・対象外の件数は「0.12.0 §3 の件数＋追跡表で移動した項目」で、未公開・保留は追跡表の集計と §4.2 のとおり照合した。

| 行 | 0.12.0 判定 | 公開 | 未公開 | 対象外 | 保留 | 0.14.0 事前監査判定 |
|---|---|---|---|---|---|---|
| dtype | 部分的 | 6 | 0 | 4 | 0 | ある |
| 自動微分 | 部分的 | 15 | 0 | 4 | 1 | 部分的 |
| 演算の範囲 | 部分的 | 8 | 0 | 4 | 0 | ある |
| NN 層・モデル構築 | 部分的 | 15 | 0 | 1 | 0 | ある |
| 最適化・学習ループ | 部分的 | 14 | 0 | 2 | 0 | ある |
| バックエンド・ハード | 部分的 | 3 | 0 | 3 | 1 | 部分的 |
| 相互運用 | ある→部分的に戻した | 7 | 0 | 2 | 0 | ある |
| 事前学習済みモデル | 部分的 | 5 | 0 | 1 | 0 | ある |
| 推論・サービング | 部分的 | 9 | 0 | 5 | 0 | ある |

- 「ある」は 7 行、「部分的」は 2 行（自動微分・バックエンド・ハード）。残る保留 2 件は、どちらも追跡表の「未確定」（# 4・# 24）である。
- 0.12.0 版の相互運用行は §8.2 で「部分的」へ戻した行で、本書では # 25〜27 がすべて公開になったため「ある」にした。

### 4.2 追跡表の集計との照合

`python3 -I scripts/tally-compat-tracker.py`（基準 sha の tree で実行。終了コード 0）の出力をそのまま載せる。

```
0.12.0 の照合: OK（§1 の集計 == §2 == §8.5 固定値）
- 行別合計 30 件（未公開 16・保留 14）、重複を除いた項目数 29

0.14.0 の区分（行別。複数行の項目は各行に数える）

| 行 | 未確定 | 未公開 | 保留 | 残件 | 公開 | 対象外 |
|---|---|---|---|---|---|---|
| dtype | 0 | 0 | 0 | 0 | 1 | 1 |
| 自動微分 | 1 | 0 | 0 | 1 | 3 | 0 |
| 演算の範囲 | 0 | 0 | 0 | 0 | 1 | 0 |
| NN 層・モデル構築 | 0 | 0 | 0 | 0 | 9 | 0 |
| 最適化・学習ループ | 0 | 0 | 0 | 0 | 7 | 1 |
| バックエンド・ハード | 1 | 0 | 0 | 1 | 0 | 0 |
| 相互運用 | 0 | 0 | 0 | 0 | 3 | 0 |
| 事前学習済みモデル | 0 | 0 | 0 | 0 | 1 | 0 |
| 推論・サービング | 0 | 0 | 0 | 0 | 1 | 0 |
| 合計 | 2 | 0 | 0 | 2 | 26 | 2 |

重複を除いた項目数: 残件 2（未確定 2・未公開 0・保留 0）、公開 25、対象外 2
```

照合表（行別。「追跡表」は上の出力の未確定＋未公開＋保留、「監査」は §3 の未公開＋保留）:

| 行 | 追跡表の残件 | 監査の未公開＋保留 | 一致 | 差の理由 |
|---|---|---|---|---|
| dtype | 0 | 0 | 一致 | — |
| 自動微分 | 1（# 4） | 1（保留 1） | 一致 | 「未確定」を 0.12.0 での区分（保留）で数えた（§2 の 2） |
| 演算の範囲 | 0 | 0 | 一致 | — |
| NN 層・モデル構築 | 0 | 0 | 一致 | — |
| 最適化・学習ループ | 0 | 0 | 一致 | — |
| バックエンド・ハード | 1（# 24） | 1（保留 1） | 一致 | 同上 |
| 相互運用 | 0 | 0 | 一致 | — |
| 事前学習済みモデル | 0 | 0 | 一致 | — |
| 推論・サービング | 0 | 0 | 一致 | — |

29 項目の外で数えた項目は 0 件である。`CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（§3 自動微分行の注記）は、承認済みで 0.12.0 §2 規則 6 の保留に当たらないため件数に数えず、追跡表の行も増やしていない（形式とスクリプトの定数を変えることになるため）。
29 項目の外の走査: `git diff --name-only v0.12.0 HEAD -- docs/` で増減した決定記録を確認し、`docs/autodiff-higher-order-grad-decision.md` §72.1 の F5（`ScalarUnary` の `Selu`・`Celu`・`Softsign`・`Hardsigmoid`・`LogSigmoid` と `Atan2` の `create_graph` fail-closed。`crates/autodiff/src/tape.rs:2609` の doc、決定記録は「記録のみ。承認の有無は事実以上に書かない」）は 0.12.0 版が数えていなかったため同じ扱い（注記）にとどめた。
`docs/compat-unlisted-holds-inventory.md` と保留ガード 43 件は 0.12.0 から変わらない。

### 4.3 0.12.0 版 §6 の所有者確認 9 件の決着

| 0.12.0 版 §6 | 決着した追跡表の項目 |
|---|---|
| 1 ROCm の facade 公開 | # 24（承認済み・parity 待ちのため「未確定」） |
| 2 SWA の `FitConfig` 形 | # 16（対象外） |
| 3 低精度演算の配置 | # 1（対象外） |
| 4 HF Hub の 2 段目 | # 28（公開に数える） |
| 5 `TapeF64` の `jvp` 系 | # 2（公開） |
| 6 `create_graph` の残り | # 3〜# 5（# 3・# 5 は公開、# 4 は「未確定」） |
| 7 GPU 線形代数 | # 6（公開） |
| 8 NN 層・推論の未公開 | # 29・# 7・# 8（公開） |
| 9 #3022 の包括承認（層型名は対象外） | 29 項目の外で 0.12.0 と同じ扱いのまま |

## 5. 実機 parity の付録表

実機 parity の付録表は #3870 の担当で、本書では作らない。#3870 の入力は追跡表の「実機 parity」列と、本書 §3 の各行の「実機 parity」欄である。
「未実測」は未実測のまま付録へ写す。本書は測定を行っておらず、未実測のものを pass と書いていない。現時点で追跡表と本書が未実測とするものは次のとおり。

- # 4 の `Nuc`・`Spectral` と `Prod`（`create-graph-b-*` に測定ケース無し）。
- # 22（param groups と `DeviceParamStore`。#3862・#3863）。
- # 24（ROCm。#3159・#3867 は OPEN）。
- 0.12.0 §5 から引き継ぐ記入欄未実測の README（`cuda-linalg-*`・`metal-linalg-*`・`dataset-compose-2661`・`iterable-batch-sampler-2662`・`ddp-host-collective-gb10-3149`・`speculative-sampling-3102`）は、本書では再確認していない。

## 6. スコープ外・申し送り・所有者への報告

### 6.1 行判定を下げた 2 行（所有者への報告）

追跡表 §5 手順 1 のとおり、「未確定」が残る行は監査の前に所有者へ報告する対象である。自動運転のため issue へのコメントは投稿していない。**報告は main セッションが行う**。

| 項目 | 状態 | 解除条件 |
|---|---|---|
| # 4 `create_graph` の B 区分（自動微分行） | 実装は済み（#3977・#3991）。`Nuc`・`Spectral` の GPU 実機 parity が未実測。所有者コメント `-6604937225` の条件を満たさない | `Nuc`・`Spectral`（と `Prod`）の実機 parity の確定。確定後、追跡表の # 4 を「公開」へ更新して再監査する |
| # 24 ROCm の facade 公開（バックエンド・ハード行） | 実装は済み（`Device::Rocm`・facade 結線）。#3159・#3867 が OPEN で MI300X の測定が無い。#3771 の回答の条件を満たさない | #3159・#3867 の parity pass |

題名の「9 行すべて 0 件」は、この 2 行が残るため達成していない。ルート #3711 の「示せない行は所有者へ報告して次のツリーへ引き継ぐ」に従い、次のツリーへ引き継ぐ。

### 6.2 29 項目の外の候補（所有者への確認事項）

- `CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（決定記録 `docs/autodiff-higher-order-grad-decision.md` §73）。承認済みで別設計待ち・恒久ではなく、起票は所有者の承認事項で未起票。
- §72.1 の F5（`Selu` など 5 variant と `Atan2` の `create_graph` fail-closed。#2649・#2634）。記録のみで承認は主張しない。
- `hf-hub` と `backend-rocm` の crates.io 公開（0.14.0 で承認済みだが、publish とピン更新は Phase 11 で所有者が明示的に指示するまで行わない）。

### 6.3 本 PR の範囲

- 本 PR は docs のみで、issue へのコメント・起票は行っていない。ruleset・リポジトリ設定も変えていない。
- 追跡表は §5 手順 6 に従い、凍結の注記だけを足した（§1 の表・列・語彙は変えていない）。

## 7. 変更していないもの

`crates/`、`docs/spec/`、tolerance 定数・parity baseline・ガードレール閾値、依存、CI、`docs/compat-api-scope.md`、0.9.0／0.11.0／0.12.0 版の本文、`docs/perf/logs/*`、
`scripts/tally-compat-tracker.py`、スコアボードの Artifact と生成スクリプト。追跡表 `docs/compat-matrix-ari-tracker.md` は凍結の注記の追記だけである。
