# フレームワーク横並びスコアボード「役割・機能の対応表」v0.14.0 事前監査

イシュー #3869（親 #3868・ルート #3711）。スコアボード「役割・機能の対応表」の fandhe-ai 列で 0.12.0 時点に「部分的」だった 9 行
（dtype／自動微分／演算の範囲／NN 層・モデル構築／最適化・学習ループ／バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービング）を、
0.14.0 の公開前に **main の公開面**で事前監査し、各行の根拠を「公開／未公開／対象外／保留／実機 parity」の 5 区分に分けて記録する。
0.12.0 版（`docs/perf/framework-compare-feature-matrix-0.12.0.md`・イシュー #3269）と同じ節構成で、入力は追跡表 `docs/compat-matrix-ari-tracker.md`（#3717）である。
**本書は crates.io 公開版ではなく main の事前監査である**。スコアボードへの反映・タグとの差分確認は本書の範囲外で、実機 parity の付録表は §5 に作成済みである（#3870）。

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
| 自動微分 | 部分的（保留 4） | 部分的 | **公開（15 件）:** 0.12.0 の 12 件（`backward_create_graph`〈`crates/facade/src/lib.rs:1623`〉、`custom`〈`crates/facade/src/lib.rs:2389`〉、`register_backward_hook`〈`crates/facade/src/lib.rs:2447`〉、`jacobian`〈`crates/facade/src/lib.rs:1706`〉・`hessian`〈`crates/facade/src/lib.rs:1743`〉、`gradcheck`〈`crates/facade/src/lib.rs:2299`〉・`backward_detect_anomaly`〈`crates/facade/src/lib.rs:2266`〉、`vjp`〈`crates/facade/src/lib.rs:1793`〉・`hvp`〈`crates/facade/src/lib.rs:1851`〉・`vmap`〈`crates/facade/src/lib.rs:2169`〉、`jvp`〈`crates/facade/src/lib.rs:1925`〉・`jacfwd`〈`crates/facade/src/lib.rs:1962`〉、`jvp_multi`〈`crates/facade/src/lib.rs:2030`〉、`jvp_create_graph`〈`crates/facade/src/lib.rs:2114`〉、`vmap_grad`〈`crates/facade/src/lib.rs:2233`〉、`set_deterministic`〈`crates/facade/src/lib.rs:3329`〉、`var_no_grad`〈`crates/facade/src/lib.rs:638`〉・`backward_accumulate`〈`crates/facade/src/lib.rs:657`〉・`detach`〈`crates/autodiff/src/var.rs:278`〉・`checkpoint_from`〈`crates/autodiff/src/var.rs:2502`〉。宣言行を基準 sha で引き直し、すべて存在する）；# 2 `TapeF64` の `jvp` 系（dtype 行と同じ）；# 3 `create_graph` の C 区分 24 Op（実装は #3789〜#3796。対象 Op 集合の正ガードは `create_graph_supported_op_set_matches_approved_list`〈`crates/facade/tests/api_surface.rs:47984`〉、facade 挙動テストとの対応表は `create_graph_facade_coverage_map_matches_supported_set`〈`crates/facade/tests/api_surface.rs:48174`〉、定義元は `supports_create_graph`〈`crates/autodiff/src/tape.rs:2654`〉）；# 5 二階微分の拒否範囲と意味論（`Inv`・`Solve`・`SlogdetLogAbsDet` の overflow 入力の拒否を `inv_solve_overflow_inputs_are_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2741`〉が固定）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（4 件）:** D-0 の除外名（`Var::custom` 系・`functional_ops` の自由関数・gradcheck の自由関数。#3018 の `#issuecomment-6111643437`）；ネイティブ forward-mode の JVP 規則とバッチ規則型 vmap（#2499 の `#issuecomment-6033824965`）；`jvp` の f16 対応（#3021 の `#issuecomment-6111746880` の項目 3）；`create_graph` が構造的に定義できない Op（`ResidentLeaf`・`LinearResident`・`LinearAct`・`Custom`・`OneHot` と、`CtcLoss` の恒久 fail-closed〈`ctc_loss_create_graph_is_rejected_via_facade`、`crates/facade/tests/create_graph_facade.rs:2661`〉。`Inv`・`Solve`・`SlogdetLogAbsDet` の拒否範囲の恒久化。承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>）。件数は 0.12.0 の 4 件のまま（集計の単位が項目か区分かの差で、増減はない）<br>**保留（1 件）:** # 4 `create_graph` の B 区分（追跡表では「未確定」。0.12.0 での区分は保留）。実装は済んでいる: `Prod` は #3977（`scatter_reduce_prod_create_graph_via_facade`、`crates/facade/tests/create_graph_facade.rs:897`）、`Nuc`・`Spectral` は #3991（`supports_create_graph` の `MatrixNorm` の腕、`crates/autodiff/src/tape.rs:2853`。`matrix_norm_nuc_spectral_create_graph_via_facade`、`crates/facade/tests/create_graph_facade.rs:2031`。縮退入力の拒否は同:2054）。それでも**保留のままにする理由**: #3711 の所有者コメント `-6604937225` が「実装が済み、実機 parity も確定するまで未確定」とし、`Nuc`・`Spectral` の GPU 実機 parity は未実測である（決定記録 `docs/autodiff-higher-order-grad-decision.md:5558`〈§76.5〉。`create_graph_gpu_parity.rs` の `reject_matrix_norm_nuc_degenerate`〈`crates/facade/tests/create_graph_gpu_parity.rs:2594`〉は縮退入力の拒否へ変えただけで、`Nuc`・`Spectral` の値のケースは #4052 で追加したが、実機は未実測。`Nuc` の縮退拒否だけが測定済み）。解除条件はこの実機 parity の確定である。承認は <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>。**範囲外の注記（件数に数えない）**: `CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（`cosine_similarity_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2674`〉・`pairwise_distance_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2685`〉。決定記録 `docs/autodiff-higher-order-grad-decision.md` §73〈5388 行〉は「承認済み（#3768 Q9 (a)）・別設計待ち・恒久ではない・起票は所有者の承認事項で未起票」とする）。29 項目の外で、0.12.0 §2 規則 6 の「保留」（所有者の回答が無い論点）には当たらないが、所有者への報告事項に挙げる（§6）<br>**実機 parity:** # 2 は該当 issue なし。# 3・# 5 は pass（2026-11-04・`<cuda-node>` `6852ef6bd` と `48d0263ac`、`<metal-node>` `0c170cee2` と `bbfcaee6b`。#3856〜#3859。記録は `docs/perf/logs/create-graph-c-cuda-parity-3856/`・`docs/perf/logs/create-graph-b-cuda-parity-3857/` と各 metal 版）。# 4 は B 区分の 12 族が pass（同上）。`Prod` は `scatter_reduce_prod` ケースが `scatter_reduce` 族で pass 済み（#3857〈`docs/perf/logs/create-graph-b-cuda-parity-3857/README.md:47`〉・#3859。両測定コミットとも #3977 の `362ab06c4` を含む）。`Nuc` は縮退拒否（Q5）だけが測定済みで、`Nuc`・`Spectral` の値の経路は**未実測**（#4052 で測る） |
| 演算の範囲 | 部分的（未公開 1） | ある | **公開（8 件）:** 0.12.0 の 7 件（`Var` 委譲、`rfft`〈`crates/autodiff/src/var.rs:5112`〉、`median`〈`crates/autodiff/src/var.rs:5356`〉・`kron`〈`crates/autodiff/src/var.rs:5536`〉・`pad_with_mode`〈`crates/autodiff/src/var.rs:5577`〉、`max_pool3d`〈`crates/autodiff/src/var.rs:5594`〉・`conv_transpose3d`〈`crates/autodiff/src/var.rs:5703`〉・`max_unpool2d`〈`crates/autodiff/src/var.rs:5747`〉、`fold`〈`crates/autodiff/src/var.rs:5807`〉・`local_response_norm`〈`crates/autodiff/src/var.rs:5825`〉・`weight_norm`〈`crates/autodiff/src/var.rs:5839`〉・`spectral_norm`〈`crates/autodiff/src/var.rs:5846`〉、`bce_with_logits_loss_with`〈`crates/autodiff/src/var.rs:4953`〉、`bincount`〈`crates/facade/src/lib.rs:1648`〉）；# 6 GPU 線形代数の opt-in `set_cuda_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3839`）・`set_metal_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3866`）。正ガードは `gpu_linalg_optin_is_reachable_via_facade`（`crates/facade/tests/api_surface.rs:44513`）と `workspace_declares_gpu_linalg_optin_fn_names_only_in_approved_locations`（`crates/facade/tests/api_surface.rs:44567`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 6）は公開へ移った<br>**対象外（4 件）:** 0.12.0 §3 から不変（D-0 の除外名、sparse／complex、Metal の eigh・SVD と CUDA の SVD、int／bool 演算の網羅。承認は 0.12.0 版の出所のとおり）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 6 は pass（#3866。facade 経由・ゲート ON。`<cuda-node>` は 6 本、`<metal-node>` は 5 本〈eigh は範囲外〉。2026-11-04。記録は `docs/perf/logs/gpu-linalg-optin-cuda-parity-3866/`・`docs/perf/logs/gpu-linalg-optin-metal-parity-3866/`）。0.12.0 時点で未実測だった `cuda-linalg-*`・`metal-linalg-*` の README の記入欄は、2026-10-27／28 に実測済み（pass。`metal-linalg-*` は 2026-10-27 の fail の後、入力を是正して 2026-10-28 に pass。§5.4） |
| NN 層・モデル構築 | 部分的（未公開 2。0.12.0 §8.5 の確定は未公開 9） | ある | **公開（15 件）:** 0.12.0 の 6 件（`add_max_pool3d`〈`crates/facade/src/compat/sequential.rs:2423`〉・`add_avg_pool3d`〈`crates/facade/src/compat/sequential.rs:2551`〉・`add_conv_transpose3d`〈`crates/facade/src/compat/sequential.rs:1723`〉・`add_fold`〈`crates/facade/src/compat/sequential.rs:2487`〉・`add_unfold`〈`crates/facade/src/compat/sequential.rs:2455`〉・`add_local_response_norm`〈`crates/facade/src/compat/sequential.rs:2523`〉、MaxUnpool の層経路〈`crates/facade/src/compat/functional.rs:552`〉、`parametrize_weight_norm`〈`crates/facade/src/compat/sequential.rs:838`〉・`parametrize_spectral_norm`〈`crates/facade/src/compat/sequential.rs:903`〉、`save_model`〈`crates/facade/src/compat/model_io.rs:283`〉・`load_model`〈`crates/facade/src/compat/model_io.rs:306`〉と kind 追加の `spec_kind`〈`crates/facade/src/compat/model_io.rs:336`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 7 `add_upsample_scale_factor`（`crates/facade/src/compat/sequential.rs:2978`。正ガード `compat_sequential_spatial_add_methods_have_approved_signatures`〈`crates/facade/tests/api_surface.rs:14947`〉）；# 8 `Sequential` への `nn::Module` 実装（`crates/facade/src/compat/sequential.rs:706`。正ガード `compat_sequential_nn_module_impl_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:48320`〉）；# 9 `add_hardshrink`（`crates/facade/src/compat/sequential.rs:1267`）・`add_softshrink`（`crates/facade/src/compat/sequential.rs:1291`）、`Var` 側は `hardshrink`（`crates/autodiff/src/var.rs:5937`）・`softshrink`（`crates/autodiff/src/var.rs:5953`）；# 10 `bilinear`（`crates/autodiff/src/var.rs:5981`。正ガード `var_bilinear_ops_method_is_thin_delegation`〈`crates/facade/tests/api_surface.rs:48981`〉）；# 11 `add_batch_norm3d`（`crates/facade/src/compat/sequential.rs:1934`）と rank 5 の `batch_norm`（`crates/autodiff/src/var.rs:1859`）；# 12 `cosine_similarity`（`crates/autodiff/src/var.rs:6015`）；# 13 `pairwise_distance`（`crates/autodiff/src/var.rs:6043`）；# 14 `add_channel_shuffle`（`crates/facade/src/compat/sequential.rs:3060`）、`Var` 側は `channel_shuffle`（`crates/autodiff/src/var.rs:3107`）；# 15 `add_adaptive_avg_pool3d`（`crates/facade/src/compat/sequential.rs:2780`）・`add_adaptive_max_pool3d`（`crates/facade/src/compat/sequential.rs:2807`）・`add_lp_pool1d`（`crates/facade/src/compat/sequential.rs:2588`）・`add_lp_pool2d`（`crates/facade/src/compat/sequential.rs:2623`）・`add_lp_pool3d`（`crates/facade/src/compat/sequential.rs:2658`）・`add_fractional_max_pool2d`（`crates/facade/src/compat/sequential.rs:2840`）・`add_fractional_max_pool3d`（`crates/facade/src/compat/sequential.rs:2869`）。新層の保存 kind の正ガードは `model_io_new_layer_kinds_match_approved_mapping_issue_3828`（`crates/facade/tests/api_surface.rs:51413`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 §8.5 の 9 件（# 7〜15）はすべて公開へ移った<br>**対象外（1 件）:** 層型名の公開（`fandhe_ai::nn::*` の全層型。#3022 の決定記録＝公開対象外 (i)+(iv)。`SpatialLayersHoldDoctestGuard`〈`crates/facade/src/lib.rs:4903`〉は維持）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 7・# 9・# 10・# 11 は、追跡表の欄の先頭語が「未実測」のまま残っているが、本文には `<cuda-node>`・`<metal-node>` とも pass の記録がある（#3860・#3861。2026-11-04。記録は `docs/perf/logs/nn-layers-cuda-parity-3860/`・`docs/perf/logs/nn-layers-metal-parity-3861/`。学習ステップの雛形は `docs/perf/logs/nn-train-template-cuda-parity-3860/`・`docs/perf/logs/nn-train-template-metal-parity-3861/` で pass）。# 8 は該当 issue なし。# 12〜15 の欄は追跡表のとおり写す（個別記録は §5.5 の `shrink-ops-3817` 等の行と、`nn-layers-{cuda,metal}-parity-3860/3861` の 19 本 pass を参照） |
| 最適化・学習ループ | 部分的（未公開 1。0.12.0 §8.5 の確定は未公開 1・保留 7） | ある | **公開（14 件）:** 0.12.0 の 7 件（optimizer の `Adamax`〈`crates/facade/src/optim.rs:528`〉・`Asgd`〈`crates/facade/src/optim.rs:529`〉・`Lion`〈`crates/facade/src/optim.rs:537`〉・`Rprop`〈`crates/facade/src/optim.rs:538`〉、`PolynomialLr`〈`crates/facade/src/optim.rs:530`〉・`CyclicLr`〈`crates/facade/src/optim.rs:534`〉、`ParamGroup`〈`crates/facade/src/optim.rs:554`〉、`OptimizerStateDict`〈`crates/facade/src/optim.rs:551`〉、`SwaLr`〈`crates/facade/src/optim.rs:539`〉、`fit_with_weights`〈`crates/facade/src/compat/training.rs:3026`〉・`fit_with_train_step`〈`crates/facade/src/compat/training.rs:2928`〉、`train_step_resident`〈`crates/facade/src/compat/sequential.rs:4550`〉）；# 17 EMA と AMP の併用（正ガード `ema_fit_rejection_arms_match_approved_set`〈`crates/facade/tests/api_surface.rs:20715`〉）；# 18 EMA のウォームアップ（`ema_warmup_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:20946`〉）；# 19 param groups と AMP（`compile_with_amp_and_param_groups`、`crates/facade/src/compat/training.rs:2434`）；# 20 param groups と `LrSchedule`（`compile_with_param_groups`、`crates/facade/src/compat/training.rs:2236`。正ガード `param_groups_usage_doctests_are_present_and_compiled`〈`crates/facade/tests/api_surface.rs:18621`〉）；# 21 param groups の `state_dict` 保存（`save_model`・`load_model`。# 20 と同じ正ガード）；# 22 param groups と `DeviceParamStore`（`init_device_param_store_with_param_groups`、`crates/facade/src/compat/sequential.rs:4315`）；# 23 `NAdam`（`crates/facade/src/optim.rs:546`）・`RAdam`（`crates/facade/src/optim.rs:556`）と `Optimizer` の腕（`crates/facade/src/compat/training.rs:313`・`crates/facade/src/compat/training.rs:317`）、保存 kind（`crates/facade/src/compat/model_io/compiled.rs:251`・`crates/facade/src/compat/model_io/compiled.rs:252`）、正ガード `APPROVED_COMPAT_OPTIMIZER_VARIANTS`（`crates/facade/tests/api_surface.rs:22262`。11 種）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の未公開 1 件（# 16）は対象外へ移った<br>**対象外（2 件）:** `FitConfig` への EMA（D-0 の除外名。#3018 の `#issuecomment-6111643437`。機能は `Callback::Ema`〈`crates/facade/src/compat/callbacks.rs:1750`〉で到達できる）；# 16 SWA の `FitConfig` 形（承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758>。機能は `Callback::Swa`〈`crates/facade/src/compat/callbacks.rs:1752`〉で公開済み。禁止経路専用で `SwaHoldDoctestGuard`〈`crates/facade/src/lib.rs:5721`〉と `swa_hold_doctest_globs_all_pub_modules`〈`crates/facade/tests/api_surface.rs:21058`〉を維持）<br>**保留（0 件）:** なし（0 件）。0.12.0 §8.5 の 7 件（# 17〜23）はすべて公開へ移った<br>**実機 parity:** # 22 は CUDA・Metal ともに実測済み（#3862: `docs/perf/logs/optim-train-template-cuda-parity-3862/README.md:16`、24 量 pass・fail 0。#3863: `docs/perf/logs/optim-train-template-metal-parity-3863/README.md:20`、24 行 pass・fail 0）。# 23 は該当 issue なし。0.12.0 §5 の `dataset-compose-2661`・`iterable-batch-sampler-2662` は host ユーティリティで、2026-10-27 に GB10・Mac mini とも pass（§5.4）。学習ステップ常駐チェーンは 0.12.0 §5 のとおり実測済み |
| バックエンド・ハード | 部分的（保留 1） | 部分的 | **公開（3 件）:** `tape_for`（`crates/facade/src/lib.rs:3214`）・`available_devices`（`crates/facade/src/lib.rs:3276`）・`set_cuda_gemm_precision`（`crates/facade/src/lib.rs:3909`）；デバイス間の値転送 `transfer`（`crates/facade/src/lib.rs:895`）；分散学習の初期形 `distributed`（`crates/facade/src/lib.rs:255`）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（3 件）:** 0.12.0 §3 から不変（FSDP・ZeRO 系・tensor／pipeline parallel、cudarc の `nccl` feature、int8 量子化）<br>**保留（1 件）:** # 24 ROCm の facade 公開（追跡表では「未確定」。0.12.0 での区分は保留）。実装は済んでいる: `Device::Rocm(usize)`（`crates/tensor-core/src/device.rs:79`）、`resolve_ops` の `Device::Rocm(_)` の腕（`crates/facade/src/lib.rs:3603`）、provider（`crates/backend-rocm/src/device.rs:32`）、`backend-rocm` は `publish = true`（`crates/backend-rocm/Cargo.toml:33`。crates.io 未公開）。正ガードは `device_enum_variant_inventory_is_fixed`（`crates/facade/tests/api_surface.rs:45740`）・`facade_depends_on_backend_rocm_only_in_approved_shape`（`crates/facade/tests/api_surface.rs:45776`）・`facade_wires_rocm_into_resolve_ops_and_available_devices`（`crates/facade/tests/api_surface.rs:51959`）・`rocm_device_usage_doctests_are_present_and_compiled`（`crates/facade/tests/api_surface.rs:52039`）。それでも**保留のままにする理由**: #3771 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469>）の点 1 が「facade への同梱は #3159・#3867 の parity が pass することを条件とする」とし、両 issue は 2026-11-05 時点で OPEN（`gh issue view` で確認。#3159 は更新 07:19:09Z、#3867 は 07:19:10Z）、MI300X 実機の測定が無い。表記は「条件付き・実験的」（`crates/facade/src/lib.rs:3203`）。解除条件は #3159・#3867 の pass である<br>**実機 parity:** # 24 は `<rocm-node>` で**未実測**（追跡表の欄）。DDP は `ddp-host-collective-gb10-3149` が 2026-10-27 に GB10 で 5 run 全 pass（§5.4）。CUDA GEMM の既定精度は FP32 のまま |
| 相互運用 | 部分的 → 0.12.0 §8.2 で「部分的」へ戻した（未公開 2・保留 1） | ある | **公開（7 件）:** 0.12.0 の 4 件（ONNX import `OnnxModel`〈`crates/facade/src/interop/onnx.rs:234`〉と export `from_sequential`〈`crates/facade/src/interop/onnx.rs:364`〉、safetensors の f32 入出力、npy／npz、`save_model`／`load_model`）；# 25 ONNX モデルの学習の入口 `bind_autograd`（`crates/facade/src/interop/onnx.rs:914`。正ガード `onnx_autograd_types_are_reachable_via_facade`〈`crates/facade/tests/api_surface.rs:2770`〉）；# 26 ONNX export の `Sigmoid`（`crates/onnx-interop/src/onnx/export_nn.rs:367`。正ガード `onnx_export_sigmoid_matches_approved_shape_issue_3844`〈`crates/facade/tests/api_surface.rs:51699`〉）；# 27 safetensors の f32 以外の型（`crates/facade/src/interop/safetensors.rs:196`・`crates/facade/src/interop/safetensors.rs:197` の再エクスポート。正ガード `interop_safetensors_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:3083`〉）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（2 件）:** 0.12.0 §3 から不変（Python バインディングと TF 系形式、`Tensor` への npy／npz の inherent メソッド）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 1 件（# 26）は公開へ移った<br>**実機 parity:** # 25 は pass（`<cuda-node>` の測り直しを含む。`docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`。#4046）。# 26・# 27 は該当 issue なし（ホスト実行のみ） |
| 事前学習済みモデル | 部分的（未公開 1） | ある | **公開（5 件）:** 0.12.0 の 4 件（`ModelRegistry`〈`crates/facade/src/model.rs:684`〉・`load`〈`crates/facade/src/model.rs:870`〉、`save`〈`crates/facade/src/model.rs:1017`〉、`download`〈`crates/facade/src/model.rs:1234`〉・`download_with`〈`crates/facade/src/model.rs:1287`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 28 HF Hub 連携の補助クレート `hf-hub`（`HfHub`〈`crates/hf-hub/src/lib.rs:342`〉・`resolve_revision`〈`crates/hf-hub/src/lib.rs:385`〉・`list_files`〈`crates/hf-hub/src/lib.rs:430`〉。`download` は `pub(crate)`〈`crates/hf-hub/src/lib.rs:452`〉のまま。`publish = true`、`crates/hf-hub/Cargo.toml:36`。crates.io には未公開）。数え方は §2 の 1（#3771 の案 A）による<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 28）は §2 の読み方で公開に数える<br>**対象外（1 件）:** 重みの同梱（#3021 の `#issuecomment-6111746880` の項目 5。0.12.0 §3 から不変）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 0.12.0 §5 の HTTPS 実通信の確認が実測済み。# 28 は該当 issue なし（ホスト上の HTTP 取得のみ） |
| 推論・サービング | 部分的（未公開 1） | ある | **公開（9 件）:** 0.12.0 の 8 件（`predict`〈`crates/facade/src/compat/sequential.rs:3152`〉・`predict_batches`〈`crates/facade/src/compat/sequential.rs:3376`〉・`predict_resident`〈`crates/facade/src/compat/sequential.rs:4431`〉・`predict_resident_input`〈`crates/facade/src/compat/sequential.rs:4478`〉、`generate`／`AutoregressiveModel`／`GenerateConfig`／`SamplingStrategy`〈`crates/facade/src/inference/mod.rs:463`〉、`new_paged_kv_cache`〈`crates/facade/src/lib.rs:2622`〉・`paged_attention_forward`〈`crates/facade/src/lib.rs:2642`〉、`BatchScheduler`〈`crates/facade/src/inference/mod.rs:472`〉、`generate_speculative`〈`crates/facade/src/inference/mod.rs:476`〉、`StringLookup`〈`crates/facade/src/text/mod.rs:151`〉、`set_cuda_graph_infer_enabled`〈`crates/facade/src/lib.rs:3763`〉・`set_cuda_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3795`〉・`set_metal_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3809`〉）；# 29 内部状態を持つモデルの speculative／連続バッチング（`StatefulAutoregressiveModel`・`generate_speculative_stateful` の再エクスポート〈`crates/facade/src/inference/mod.rs:485`〉、trait〈`crates/autodiff/src/generate/stateful.rs:50`〉、`generate_speculative_stateful`〈`crates/autodiff/src/generate/stateful.rs:150`〉、`step_stateful`〈`crates/autodiff/src/generate/scheduler.rs:687`〉。正ガードは `inference_module_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:25893`〉・`speculative_scheduler_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43252`〉・`stateful_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43326`〉）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 29）は公開へ移った<br>**対象外（5 件）:** 0.12.0 §3 から不変（汎用グラフコンパイル区分 A、HTTP／API サーバと量子化 KV、サブワードトークナイザ、beam search 等の生成機能、int8 量子化）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 29 は新しいカーネル・バックエンド経路が無く、実機 parity の対象外（追跡表の欄は「該当 issue なし」）。speculative とスケジューラ・paged attention は 0.12.0 §5 のとおり実測済み。`speculative-sampling-3102` は 2026-10-27 に GB10・Mac mini とも pass（§5.4） |

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

### 5.1 範囲と判定規則

- 数える基準 sha は §1 と同じ `1cbb475989e32051b337efb81b38b843de80145b`。`git diff --stat 1cbb4759 HEAD -- docs/perf/logs` は空で（本書の作成時点の main `5383e474f` は `docs/perf/logs` を変えていない）、基準 sha 時点の一覧と現在の一覧は同じである。
- 母集合は 172 件である。内訳は、0.12.0 §5 の 120 件（`1009825c9e9e954bf58748c5c1b42234bb12121d` 時点で番号 2500 以上）と、`1009825c` から基準 sha までに増えたディレクトリ 52 件（`comm -13` で名前を問わず数えた全件）の合計。番号で絞る正規表現は使っていない。
  - 基準 sha 時点のディレクトリは 395 件で、うち 223 件（`1009825c` 時点で番号 2500 未満だったもの）は 0.12.0 §5 と同じく走査対象外である。
  - 名前が数字で終わらないため番号で数えると落ちるもの 10 件も母集合に含めた: `onnx-train-cuda-parity-3864-rerun`・`metal-batch-norm-2e20-4031-rerun`・`metal-batch-norm-2e20-4031-rerun2`・`create-graph-gpu-parity-template`・`hw-parity-run-template`・`train-gpu-parity-template`・`framework-compare-0.12.0-remeasure`・`framework-compare-0.12.0-macmini-remeasure`・`framework-compare-0.13.0-remeasure`・`framework-compare-0.13.0-macmini-remeasure`。
  - README が無い 3 件も行として載せ、黙って除外しない: 0.12.0 から引き続き `infer-chain-graphcapture-rejudge-3205`・`train-step-graph-3260`（今も README は無い）と、新しく `create-graph-gpu-parity-template`（`RULE.txt` のみ）。#3867 はディレクトリ自体が無いので §5.6 に載せる。
- 状態は README の記入欄（「状態」の行）で決め、見出しの語では決めない（§2）。0.12.0 の 4 区分（実測済み・実機未実測・判定対象外・性能計測）を緩めずに使い、次の 4 区分を足す。
  - **無効**: 所有者の判断で判定に使わない回（`metal-batch-norm-2e20-4031` と `-rerun`。結果が pass でも根拠にしない）。
  - **雛形**: 実測ではない。`hw-parity-run-template`・`train-gpu-parity-template`・`create-graph-gpu-parity-template` の 3 件。雛形自体の検証が通っても parity の pass には数えない。
  - **開発機での再現**: `shared-target-reuse-repro-4044`（実機を使わず、Linux の開発機で再現した記録）。
  - **記入欄は未実測（別ディレクトリで実測）**: Phase 5／6 の実装時に作られた README で、記入欄が空のまま残っているが、同じテストが `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本に含まれて pass している 8 件（`shrink-ops-3817`・`bilinear-ops-3818`・`batch-norm-rank5-3819`・`similarity-ops-3821`・`channel-shuffle-3822`・`adaptive-pool3d-ops-3823`・`lp-pool-ops-3825`・`fractional-max-pool-ops-3826`）。テスト名は `nn-layers-cuda-parity-3860/README.md:45` 以降の表と、両ディレクトリの `summary.tsv` で突き合わせた。実機を使わない開発環境の代替値で pass にしていない。
- 実測値を推測で埋めていない。ホスト名は `<cuda-node>`・`<metal-node>`・`<rocm-node>` の表記だけを使う。

### 5.2 Phase 10（#3855）の子 issue 18 件

`gh api repos/Fandhe-AI/fandhe-ai/issues/3855/sub_issues` の全件。状態は 2026-11-05T08:19:21Z に取得した（CLOSED 16・OPEN 2）。判定は各 README の記入欄（`path:line`）で、実行コミットは `rev-stamp.txt` の完全 sha。#4044 の分類は `docs/real-hardware-shared-target-reuse-decision.md` §4.2（A: 全メンバー再コンパイル／B: 前の回の成果物が再利用された可能性／C: 由来を確かめられない）。

| issue | 状態 | ディレクトリ | 対応表の行／追跡表 # | 判定（記入欄） | 実測日 | 実行コミット（完全 sha） | #4044 分類 |
|---|---|---|---|---|---|---|---|
| #3856 二階微分 C 族（GB10） | CLOSED | `create-graph-c-cuda-parity-3856` | 自動微分／# 3〜# 5 | 8/8 pass（`README.md:14`） | 2026-11-04 | `6852ef6bd067a74e3090b4f390f59a5a70f8b4a5` | A |
| #3857 二階微分 B 族（GB10） | CLOSED | `create-graph-b-cuda-parity-3857` | 自動微分／# 3〜# 5 | 15/15 pass（`README.md:15`。#3801 は #3767 で対象外確定のため測定外） | 2026-11-04 | `48d0263ac3f7fb69f336d49c1203936fa8b787ae` | B |
| #3858 二階微分 C 族（Mac mini） | CLOSED | `create-graph-c-metal-parity-3858` | 自動微分／# 3〜# 5 | 8/8 pass（`README.md:16`） | 2026-11-04 | `0c170cee25c00bd5d52cb2b7848378a3691f5024` | A |
| #3859 二階微分 B 族（Mac mini） | CLOSED | `create-graph-b-metal-parity-3859` | 自動微分／# 3〜# 5 | 15/15 pass（`README.md:15`） | 2026-11-04 | `bbfcaee6bdd1d7cc0034aab14f8832e7ea8d57e1` | A |
| #3860 新しい NN 層（GB10） | CLOSED | `nn-layers-cuda-parity-3860`・`nn-train-template-cuda-parity-3860` | NN 層／# 7・# 9〜# 15・# 22 | 19/19 pass（`nn-layers-…/README.md:14`）。雛形の nn 族は 0 件で undetermined だったため（同 `:16`）、#4026 の登録後に再計測し 120 量 pass（`nn-train-template-…/README.md:16`） | 2026-11-04 | `77c2cbd58a9daca71731c2133553d898bb9f10af`／`bf118b365d8fc28b52d579ed31ac3daa001dac73` | B／A |
| #3861 新しい NN 層（Mac mini） | CLOSED | `nn-layers-metal-parity-3861`・`nn-train-template-metal-parity-3861` | NN 層／同上 | 19/19 pass（`nn-layers-…/README.md:15`）。雛形の nn 族 0 件（同 `:17`）→ 再計測で 120/120 pass（`nn-train-template-…/README.md:18`） | 2026-11-04 | `6c29359cc6d798a7f1b73ee396ff3e4e30c72f2c` | A／A |
| #3862 param groups・常駐（GB10） | CLOSED | `param-groups-resident-cuda-parity-3862`・`optim-train-template-cuda-parity-3862` | 最適化・学習ループ／# 22 | 3/3 pass（`param-groups-…/README.md:14`）。雛形の optim 族 0 件のため #4026 の登録後に再計測し 24 量 pass（`optim-train-template-…/README.md:16`） | 2026-11-04 | `6ba9db57f141c69effbc6f9da1d3517d08f1fe58`／`f81eedd9cd4478ff3f57e65f3d82c9a8e0c3f8a1` | B／A |
| #3863 param groups・常駐（Mac mini） | CLOSED | `param-groups-resident-metal-parity-3863`・`optim-train-template-metal-parity-3863` | 最適化・学習ループ／# 22 | 3/3 pass（`param-groups-…/README.md:14`）→ 再計測で 24/24 pass（`optim-train-template-…/README.md:20`） | 2026-11-04 | `39cb7d10160115ebdbbe23958ce8c43ae390c115`／`31656bccc2484d601b7b1bfa0c34b949b2e7b79e` | A／A |
| #3864 ONNX の学習（GB10） | CLOSED | `onnx-train-cuda-parity-3864`・`onnx-train-cuda-parity-3864-rerun` | 相互運用／# 25 | 1 回目 pass（`README.md:15`）。分類 C のため #4046 の rerun を正とする: pass（`onnx-train-cuda-parity-3864-rerun/README.md:14`。route は H＝ホスト実行で、経路の parity） | 2026-11-04／2026-11-05 | `a05e3caeb2d303cbceb15102dfea3df1b9fb68b1`／`284dcc41795d47a2eac9ee896cfb8d2fbc862f94` | C（rerun で解消） |
| #3865 ONNX の学習（Mac mini） | CLOSED | `onnx-train-metal-parity-3865` | 相互運用／# 25 | 1 本 pass・12 量（`README.md:15`） | 2026-11-04 | `55fa6e3e3255aba4df28491bb567911d7b3e1226` | A |
| #3866 GPU 線形代数の opt-in | CLOSED | `gpu-linalg-optin-cuda-parity-3866`・`gpu-linalg-optin-metal-parity-3866` | 演算／# 6 | CUDA 6/6 pass（`README.md:22`）・Metal 5/5 pass（`README.md:23`） | 2026-11-04 | `49a470f4b4196f736bbe9631ce4d556c825f568e`（両機体で同じ） | B／B |
| #4026 学習ステップ雛形への nn・optim 族の登録 | CLOSED | 該当ログなし（雛形の登録。再計測は #3860〜#3863 の `*-train-template-*` に収録） | — | — | - | - | - |
| #4031 Metal BatchNorm 2e20 級入力 | CLOSED | `metal-batch-norm-2e20-4031`・`-rerun`・`-rerun2` | NN 層／# 9〜# 15 のうち BatchNorm | 1 回目は無効（展開ツリーの上書き。`README.md:16`）、rerun も無効（無効回の成果物の再利用。`README.md:14`）、**rerun2 が有効で pass**（`README.md:20`）。無効 2 回の結果は pass だが根拠にしない | 2026-11-05 | `a0ae7bdbc48ebf2a1be86dbab2afe69a10c23e80`・`3493be20254b2dabe2b44533b0aa323e7ae9dd97`（無効）／`82326b2537f89b4e5c5bf1129b69c0c3044442f0`（有効） | 対象外（release プロファイル。rerun2 が正） |
| #4032 正規化統計の Metal 実装形（soft-f64） | CLOSED | 該当ログなし（docs） | — | — | - | - | - |
| #4044 共有 target の再利用対策 | CLOSED | `shared-target-reuse-repro-4044`・`hw-parity-run-template` | — | 開発機での再現と雛形。実機の parity ではない | 2026-11-05 | - | 16 計測の分類を作った側（A 10・B 5・C 1） |
| #4046 #3864 の測り直し | CLOSED | `onnx-train-cuda-parity-3864-rerun` | 相互運用／# 25 | pass（`README.md:14`） | 2026-11-05 | `284dcc41795d47a2eac9ee896cfb8d2fbc862f94` | C を解消 |
| #3159 MI300X での parity | **OPEN** | `rocm-parity-3159` | バックエンド・ハード／# 24 | **未実測**（`README.md:3`）。§5.3 | - | - | - |
| #3867 ROCm を facade 経由で測る | **OPEN** | ディレクトリ無し | バックエンド・ハード／# 24 | **未実測**。§5.3 | - | - | - |

注: 実行コミットの完全 sha は各 `rev-stamp.txt` の 1 行目から機械的に展開した（手で転記していない）。

### 5.3 #3159・#3867（実測していないものは未実測のまま）

| issue | `gh issue view --json state,updatedAt` の結果 | 確認した日時（UTC） | README の記入欄 |
|---|---|---|---|
| #3159 | state: OPEN・updatedAt: 2026-11-05T07:19:09Z | 2026-11-05T08:19:21Z | `docs/perf/logs/rocm-parity-3159/README.md:3` に「**状態: 未実測**（所有者のインスタンス起動待ち）」。記入欄は 6 区画とも未実測（`:58`・`:80`・`:109`・`:131`・`:152`・`:173`） |
| #3867 | state: OPEN・updatedAt: 2026-11-05T07:19:10Z | 2026-11-05T08:19:21Z | ディレクトリが無い（README も無い）。未実測 |

- どちらも OPEN のままで、ROCm（MI300X・gfx942）の実機 parity は未実測である。ROCm の facade 同梱は #3771 の条件を満たしていない（§6.1 と同じ）。
- 本書は実機を使っておらず、Linux や M4 Max の開発機の代替値も入れていない。

### 5.4 0.12.0 §6 の申し送り README（10 件）と README の無い 2 件

0.12.0 §5 では実機未実測だった 10 件のうち、**9 件は 2026-10-27／28 に実測済みで、残る 1 件（`rocm-parity-3159`）だけが未実測**である。

| README | 0.12.0 §5 の状態 | 現在の記入欄 | 実測日 | 測定コミット（完全 sha） | 根拠 |
|---|---|---|---|---|---|
| `cuda-linalg-lu-2674` | 実機未実測 | 実測済み（pass）。GB10・5 run 全 pass、`rc=0` | 2026-10-27 | `18479ad49a715bb00bcce9f2bcba6450c837bccd` | `cuda-linalg-lu-2674/README.md:3`・`:52` |
| `cuda-linalg-chol-qr-eigh-2675` | 実機未実測 | 実測済み（pass）。GB10・5 run 全 pass | 2026-10-27 | `f175ea28f56180c6034f85804658ed5d120a32fe` | `cuda-linalg-chol-qr-eigh-2675/README.md:3`・`:52` |
| `cuda-linalg-parity-3138` | 実機未実測（スキャフォールドのみ） | 実測済み（pass）。4 コマンド × 5 run、fail 0 | 2026-10-27 | `358b77554a4c8e483561bbe92d451458786bf74d` | `cuda-linalg-parity-3138/README.md:3`・`:46` |
| `metal-linalg-lu-chol-qr-2676` | 実機未実測 | 実測済み（pass）。**2026-10-27 は `qr_matches_cpu_on_device` が 5/5 run で fail** → 所有者判断 (a) で入力 `qr_input(3,8)` を是正し、2026-10-28 に 5/5 run 全 pass | 2026-10-27（fail）・2026-10-28（pass） | fail `1184586ee0ae1f76aaf0f80dbdffbe53067010eb`・`95a18c2d3`／pass `2e30c15b2c7016afde3e4b69686187dcd9bd259b` | `metal-linalg-lu-chol-qr-2676/README.md:3`・`:50-52` |
| `metal-linalg-parity-3139` | 実機未実測 | 実測済み（pass）。2026-10-28 に 5/5 run 全 pass。2026-10-27 の fail は履歴として残る。tolerance・判定式・baseline は不変 | 2026-10-27（fail）・2026-10-28（pass） | pass `2e30c15b2c7016afde3e4b69686187dcd9bd259b`（2026-10-27 の fail は `95a18c2d3`） | `metal-linalg-parity-3139/README.md:9`・`:19`・`:26` |
| `ddp-host-collective-gb10-3149` | 実機未実測 | 実測済み（pass）。独立 5 run × 6 比較すべて `fail=0` | 2026-10-27 | `578e820d202424ea64af31cd7790dfa2a4d9a68b` | `ddp-host-collective-gb10-3149/README.md:8`・`:72` |
| `speculative-sampling-3102` | 実機未実測（注記 *1） | 実測済み（pass）。GB10・Mac mini とも 1 本 pass | 2026-10-27 | `343d88ee7e7a1cbedb10f9ace03658963483a6be` | `speculative-sampling-3102/README.md:44`・`:45` |
| `dataset-compose-2661` | 実機未実測 | 実測済み（pass）。GB10・Mac mini とも 1 本 pass（host ユーティリティの bit 一致） | 2026-10-27 | `901a2b1f64633c4bf374a60a166dc361391cc85c` | `dataset-compose-2661/README.md:22`・`:23` |
| `iterable-batch-sampler-2662` | 実機未実測 | 実測済み（pass）。GB10・Mac mini とも 1 本 pass | 2026-10-27 | `045a1241a7b53aa2d8e384e4ef8984966b630f79` | `iterable-batch-sampler-2662/README.md:22`・`:23` |
| `rocm-parity-3159` | 実機未実測 | **未実測**（変わらず） | - | - | `rocm-parity-3159/README.md:3` |
| `infer-chain-graphcapture-rejudge-3205` | 性能計測（README なし） | 今も README なし（`RULE.txt` と `gb10/` のみ）。性能計測のまま | - | - | `ls` で確認 |
| `train-step-graph-3260` | 性能計測（README なし） | 今も README なし。性能計測のまま | - | - | `ls` で確認 |

- 0.12.0 版の注記 *1（`speculative-sampling-3102` を基準 sha 時点で据え置いた件）は、この表で現況に更新した。
- 9 件の実測日は README の記入欄の日付で、いずれも `<cuda-node>`・`<metal-node>` の実機である。M4 Max 開発機など代替機の値は使っていない（`dataset-compose-2661/README.md:25`・`iterable-batch-sampler-2662/README.md:25` に明記）。

### 5.5 README 別の全件表（172 件）

0.12.0 §5 の 120 件のうち、README が `1009825c` 以降に変わっていない 101 件は 0.12.0 の行をそのまま写した（`git diff --name-only 1009825c 1cbb4759 -- docs/perf/logs` に README が現れない）。
変わった 19 件（`compat-sequential-spatial-2522`・`cuda-linalg-chol-qr-eigh-2675`・`cuda-linalg-lu-2674`・`cuda-linalg-parity-3138`・`dataset-compose-2661`・`ddp-host-collective-gb10-3149`・`iterable-batch-sampler-2662`・`low-precision-ops-2628`・`metal-linalg-lu-chol-qr-2676`・`metal-linalg-parity-3139`・`optin-ab-macmini-3184`・`resident-chain-ab-3257`・`rocm-parity-3159`・`sme-thin-m64-ab-3236`・`speculative-sampling-3102`・`train-chain-exclusion-3596`・`train-step-chain-cuda-3252`・`train-step-graph-ab-3261`・`train-step-resident-parts-parity-3224`）は記入欄を読み直した。
このうち状態が変わったのは §5.4 の 9 件だけで、`compat-sequential-spatial-2522`（倍率指定経路の節を追加）と `low-precision-ops-2628`（HEAD 再計測を追加）は注記を足し、`train-step-chain-cuda-3252`・`train-step-resident-parts-parity-3224` はホスト名のマスク、性能計測 5 件は追記のみで区分は変わらない。
新しい 52 件は記入欄を読んで判定した。

| README | 対象機能 | 対応表の行 | 状態（記入欄） | 実測日 | 測定コミット |
|---|---|---|---|---|---|
| `activation-scalar-ops-2649/README.md` | SELU・CELU・Softsign・Hardsigmoid・LogSigmoid の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `adaptive-pool3d-ops-3823/README.md` | 3 次元適応プーリング の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `batch-norm-rank5-3819/README.md` | BatchNorm rank 5・BatchNorm3d（#3819・#3820） の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `bench-fandhe-symmetry-ab-3266/README.md` | bench-fandhe 是正前後の併記計測 | — | 性能計測（parity 判定対象外） | - | - |
| `bilinear-ops-3818/README.md` | Bilinear の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `binning-ops-2638/README.md` | histc／searchsorted／bucketize／bincount | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `channel-shuffle-3822/README.md` | ChannelShuffle の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `compat-sequential-activation-layers-2529/README.md` | `Sequential` の Mish／Hardtanh／ReLU6／GLU／PReLU | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-activation-scalar-layers-2679/README.md` | `Sequential` の活性化 9 種の `add_*` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-adaptive-max-global-pool-2527/README.md` | adaptive_max_pool・global_pool 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-conv-transpose2d-2523/README.md` | `add_conv_transpose2d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-conv-transpose3d-3033/README.md` | `add_conv_transpose3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-conv3d-2524/README.md` | `add_conv3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-dropout-embedding-bag-2528/README.md` | Dropout2d／AlphaDropout／EmbeddingBag 層 | NN 層 | 実測済み（pass）。Metal の 1 回目は 1 本 fail（`MetalBackendOps::max` が `Unsupported`）→ #3475 の結線後の 2 回目で pass | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-fold-unfold-3041/README.md` | `add_fold`／`add_unfold` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-group-instance-norm-2525/README.md` | GroupNorm／InstanceNorm 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-local-response-norm-3045/README.md` | `add_local_response_norm` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-max-unpool-3037/README.md` | `FunctionalBuilder` の MaxPool 索引・MaxUnpool | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-pixel-shuffle-2526/README.md` | PixelShuffle／PixelUnshuffle 層 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `compat-sequential-pool3d-3029/README.md` | `add_max_pool3d`／`add_avg_pool3d` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `compat-sequential-spatial-2522/README.md` | Upsample・ZeroPad2d・Identity 層 | NN 層 | 実測済み（pass）。基準 sha 以降に足された倍率指定経路（#3814）の節は README 内で未実測のまま。実測は `nn-layers-{cuda,metal}-parity-3860/3861`（19 本 pass）側 | 2026-10-19 | CUDA `b640875f`／Metal `b728b9cb`（2 回目。1 回目は `b640875f`） |
| `conv-transpose3d-max-unpool-2644/README.md` | ConvTranspose3d・MaxUnpool の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `cpu-gemm-download-owned-3672/README.md` | download_owned の before／after A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-download-owned-path-3681/README.md` | download_owned の A/B（経路を絞った再計測） | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-download-owned-regress-3679/README.md` | download_owned 後の後退 3 セルの再現確認 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-download-owned-size-branch-3694/README.md` | matmul_resident のサイズ分岐の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-gemm-device-split-3701/README.md` | gemm_device の内側の区間切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-matmul-split-3693/README.md` | matmul_resident の後退の区間切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-resident-reuse-3646/README.md` | GB10 CPU gemm 常駐入力経路の reuse>fresh 逆転の切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-fresh-3187/README.md` | CPU GEMM の reuse と fresh の差 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-release-3226/README.md` | CPU GEMM の reuse 解放 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-window-3671/README.md` | bench reuse 窓と診断の差の切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-window-small-n-3680/README.md` | N=256・512 の bench reuse 窓の差の切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-gemm-reuse-window-small-n-repro-3695/README.md` | N=512 の別セッション再現確認 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-jobgrid-3177/README.md` | m=64 GEMM の job 分割・再 pack | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-mse-backward-threshold-3195/README.md` | MSE backward 逐次しきい値の再測定 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-pack-kernel-breakdown-3614/README.md` | 6 形状の A pack・B pack・kernel 時間内訳 | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-small-m-gemm-ab-3201/README.md` | m が小さい GEMM の分割方式 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-small-m-gemm-partition-3198/README.md` | m が小さい GEMM の job 統合・事前 pack | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-thread-policy-macmini-3178/README.md` | Mac mini の CPU スレッド方針 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cpu-train-fixedcost-gb10-3188/README.md` | GB10 CPU train 固定費 | — | 性能計測（parity 判定対象外） | - | - |
| `create-graph-b-cuda-parity-3857/README.md` | 二階微分 `create_graph` 族 B の CUDA（15 本。#3801 は対象外確定で測定外） | 自動微分 | 実測済み（15/15 pass。`README.md:15`） | 2026-11-04 | `48d0263ac` |
| `create-graph-b-metal-parity-3859/README.md` | 二階微分 `create_graph` 族 B の Metal（15 本） | 自動微分 | 実測済み（15/15 pass。`README.md:15`） | 2026-11-04 | `bbfcaee6b` |
| `create-graph-c-cuda-parity-3856/README.md` | 二階微分 `create_graph` 族 C の CUDA（8 本） | 自動微分 | 実測済み（8/8 pass。`README.md:14`） | 2026-11-04 | `6852ef6bd`（対象 `b82b1e5d1`） |
| `create-graph-c-metal-parity-3858/README.md` | 二階微分 `create_graph` 族 C の Metal（8 本） | 自動微分 | 実測済み（8/8 pass。`README.md:16`） | 2026-11-04 | `0c170cee2`（対象 `b82b1e5d1`） |
| `create-graph-extended-ops-3087/README.md` | 拡張 Op の `create_graph` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `create-graph-facade-2545/README.md` | `Tape::backward_create_graph` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `create-graph-gpu-parity-template/README.md` | 二階微分 `create_graph` の実機 parity 事前登録規則の雛形（README なし。`RULE.txt` のみ） | — | 雛形（実測ではない。`RULE.txt` 冒頭に「雛形であり、未実測」） | - | - |
| `cuda-autodiff-parity-3168/README.md` | f64 自動微分・create_graph・jvp／jacfwd の CUDA 集計（7 本） | dtype・自動微分 | 実測済み（7 本 pass） | 2026-10-19 | CUDA `9e2f7ddcc` |
| `cuda-facade-parity-3165/README.md` | Transformer・Functional・可変長系列 7 群の CUDA 集計 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6` |
| `cuda-gemm-reuse-fresh-gap-3186/README.md` | CUDA GEMM の reuse と fresh の差の原因特定 | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-linalg-chol-qr-eigh-2675/README.md` | CUDA の cholesky・QR・eigh | 演算 | 実測済み（pass。GB10・5 run 全 pass、`README.md:52`） | 2026-10-27 | `f175ea28f`（`rev-stamp.txt`） |
| `cuda-linalg-lu-2674/README.md` | CUDA の LU 系（solve・inv・det） | 演算 | 実測済み（pass。GB10・5 run 全 pass、`README.md:52`） | 2026-10-27 | `18479ad49`（`rev-stamp.txt`。dirty=0） |
| `cuda-linalg-parity-3138/README.md` | CUDA GPU 線形代数の GB10 parity（Stage C） | 演算 | 実測済み（pass。GB10・4 コマンド × 5 run 全 pass、`README.md:46`） | 2026-10-27 | `358b77554`（`rev-stamp.txt`。dirty=0） |
| `cuda-readback-default-confirm-3227/README.md` | CUDA readback 既定反転の GB10 確認 | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-readback-reuse-fix-3225/README.md` | CUDA readback 是正の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-tma-stage2-ab-gb10-3215/README.md` | TMA Stage 2 の GB10 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `cuda-train-infer-gb10-3189/README.md` | CUDA train／infer の内訳と同期回数 | — | 性能計測（parity 判定対象外） | - | - |
| `cumulative-ops-2636/README.md` | cummax／cummin／logcumsumexp | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `dataset-compose-2661/README.md` | Subset／ConcatDataset／random_split | 最適化・学習ループ | 実測済み（pass。GB10・Mac mini とも 1 本 pass、`README.md:22-23`） | 2026-10-27 | `901a2b1f6`（`.rev-stamp` 突合済み） |
| `ddp-host-collective-gb10-3149/README.md` | GB10 で同一 GPU を共有する 2 プロセスの分散学習 | バックエンド・ハード | 実測済み（pass。GB10・独立 5 run × 6 比較すべて `fail=0`、`README.md:72`） | 2026-10-27 | `578e820d2`（`.rev-stamp`。dirty=0） |
| `dinput-skip-ab-3247/README.md` | d_input 省略の前後 A/B | — | 性能計測（parity 判定対象外） | - | - |
| `elementwise-loss-ops-2652/README.md` | pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `f64-autograd-facade-2599/README.md` | `TapeF64`／`VarF64`／`GradientsF64` | dtype | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |
| `f64-create-graph-3786/README.md` | `TapeF64::backward_create_graph` の GPU 実機検証 | 自動微分・dtype | 実機未実測（`README.md:6`「実機は未実測」。別ディレクトリでの実測も無い） | - | - |
| `f64-jvp-facade-3788/README.md` | `TapeF64::{jvp, jacfwd, jvp_multi, jvp_create_graph}` の GPU 実機検証 | 自動微分・dtype | 実機未実測（`README.md:6`「実機は未実測」。別ディレクトリでの実測も無い） | - | - |
| `facade-custom-function-2549/README.md` | `Tape::custom`／`CustomFunction` | 自動微分 | 判定対象外（README が REQ-2 の判定対象外と明記） | - | - |
| `fft-fft-ifft-2632/README.md` | fft／ifft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fft-rfft-irfft-2631/README.md` | rfft／irfft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fft-stft-istft-2633/README.md` | stft／istft | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `fold-unfold-2645/README.md` | Fold／Unfold の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `fractional-max-pool-ops-3826/README.md` | FractionalMaxPool2d／3d の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `framework-compare-0.12.0-macmini-remeasure/README.md` | 0.12.0 の 6 フレームワーク同席計測（Mac mini） | — | 性能計測（parity 判定対象外） | - | - |
| `framework-compare-0.12.0-remeasure/README.md` | 0.12.0 の 6 フレームワーク同席計測（GB10） | — | 性能計測（parity 判定対象外） | - | - |
| `framework-compare-0.13.0-macmini-remeasure/README.md` | 0.13.0 の 6 フレームワーク同席計測（Mac mini） | — | 性能計測（parity 判定対象外） | - | - |
| `framework-compare-0.13.0-remeasure/README.md` | 0.13.0 の 6 フレームワーク同席計測（GB10） | — | 性能計測（parity 判定対象外） | - | - |
| `functional-fit-2667/README.md` | Functional モデルの fit | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `functional-graph-2665/README.md` | Functional API のグラフ構築 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `functional-transforms-2881/README.md` | vjp／hvp／vmap の内部実装 | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `functional-transforms-facade-2931/README.md` | `Tape::vjp`／`hvp`／`vmap`（facade） | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `functional-transforms-jvp-2942/README.md` | jvp／jacfwd（double-VJP ミラーと本物の関数） | 自動微分 | 実測済み（pass）。ミラー実装は 2026-10-09、本物の `functional_ops::{jvp,jacfwd}` は 2026-10-19（#3168／#3169） | 2026-10-09・2026-10-19 | ミラー `8bbeb874c`／本物 CUDA `9e2f7ddcc`・Metal `bf894108a` |
| `functional-transforms-jvp-multi-3064/README.md` | `Tape::jvp_multi` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `functional-transforms-vmap-grad-3073/README.md` | `Tape::vmap_grad` | 自動微分 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `gpu-linalg-optin-cuda-parity-3866/README.md` | GPU 線形代数の opt-in 経路の CUDA（6 本） | 演算 | 実測済み（6/6 pass。`README.md:22`） | 2026-11-04 | `49a470f4b`（対象 `2b00e494b`） |
| `gpu-linalg-optin-metal-parity-3866/README.md` | GPU 線形代数の opt-in 経路の Metal（5 本） | 演算 | 実測済み（5/5 pass。`README.md:23`） | 2026-11-04 | `49a470f4b`（対象 `2b00e494b`） |
| `gradcheck-anomaly-2671/README.md` | gradcheck・anomaly detection（内部） | 自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `https-download-3173/README.md` | `ModelRegistry::download`／`download_with` の HTTPS 実通信とビルド | 事前学習済みモデル | 実測済み（実機確認。REQ-2 の数値 parity ではない） | 2026-10-19 | `8eb15ab57` |
| `hw-parity-run-template/README.md` | 実機 parity run スクリプトの雛形（#4044） | — | 雛形（実測ではない。README 冒頭に「実機では未実行」） | - | - |
| `indexed-update-ops-2641/README.md` | scatter_reduce／index_add 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `infer-chain-fixed-cost-3205/README.md` | 推論チェーンの固定費の削減 | — | 性能計測（parity 判定対象外） | - | - |
| `infer-chain-graphcapture-rejudge-3205/README.md` | 推論チェーンの Graph capture 再判定 | — | 性能計測（README なし。parity 判定対象外） | - | - |
| `int8-mma-probe-2608/README.md` | INT8 MMA の GB10 実行プローブ | dtype | 判定対象外（性能判定。int8 は Won't・条件付き。0.11.0 版 §8 訂正 1） | - | - |
| `int8-quant-baseline-2609/README.md` | INT8 量子化の格上げ条件 (c)・(d-2) の測定 | dtype | 判定対象外（性能判定。int8 は Won't・条件付き。0.11.0 版 §8 訂正 1） | - | - |
| `iterable-batch-sampler-2662/README.md` | IterableDataset／BatchSampler | 最適化・学習ループ | 実測済み（pass。GB10・Mac mini とも 1 本 pass、`README.md:22-23`） | 2026-10-27 | `045a1241a`（`.rev-stamp` 突合済み） |
| `jacobian-hessian-2670/README.md` | `Tape::jacobian`／`hessian` | 自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `low-precision-ops-2628/README.md` | 低精度 forward の compute_dtype opt-in | dtype | 実測済み（pass）。2026-10-11 は CUDA／Metal（#2629）、2026-10-19 は CUDA の大形状（#2994）。2026-10-27 に GB10 で HEAD 再計測（#3617。baseline 行は所有者承認済み） | 2026-10-11・2026-10-19・2026-10-27 | `56b9d6ec`／CUDA（大形状）`b640875f`／HEAD 再計測 `045a18a23` |
| `lp-pool-ops-3825/README.md` | LPPool の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `lrn-weight-reparam-2646/README.md` | LocalResponseNorm・weight_norm・spectral_norm | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `margin-focal-loss-ops-2653/README.md` | MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `merge-ops-2666/README.md` | Functional の結合層 4 種 | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `metal-autodiff-parity-3169/README.md` | f64 自動微分・create_graph・jvp／jacfwd の Metal 集計（6 本） | dtype・自動微分 | 実測済み（6 本 pass） | 2026-10-19 | Metal `bf894108a` |
| `metal-batch-norm-2e20-4031/README.md` | Metal BatchNorm 2e20 級入力（1 回目） | NN 層 | 無効（展開ツリーの上書きにより所有者判断で判定に使わない。`README.md:16`） | 2026-11-05 | `a0ae7bdbc`（実際のビルドは `cc3917d93`） |
| `metal-batch-norm-2e20-4031-rerun/README.md` | Metal BatchNorm 2e20 級入力（測り直し 1 回目） | NN 層 | 無効（無効回の成果物の再利用。所有者判断。`README.md:14`） | 2026-11-05 | `3493be202` |
| `metal-batch-norm-2e20-4031-rerun2/README.md` | Metal BatchNorm 2e20 級入力（測り直し 2 回目。正とする回） | NN 層 | 実測済み（pass。`README.md:20`） | 2026-11-05 | `82326b253` |
| `metal-compat-sequential-rerun-3164/README.md` | `Sequential` 層 9 群の Metal 再実測（10 本） | NN 層 | 実測済み（10 本 pass） | 2026-10-19 | Metal `b728b9cb` |
| `metal-facade-parity-3166/README.md` | Transformer・Functional・可変長系列 7 群の Metal 集計 | NN 層 | 実測済み（pass） | 2026-10-19 | Metal `6570171a` |
| `metal-frag-interleave-ab-macmini-3212/README.md` | simdgroup 交互配置候補の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `metal-gemm-window-macmini-3182/README.md` | Metal GEMM 計測窓の内訳 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-infer-bimodal-macmini-3183/README.md` | Metal 推論の二峰性 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-linalg-lu-chol-qr-2676/README.md` | Metal の LU 系・cholesky・QR | 演算 | 実測済み（pass）。2026-10-27 は `qr_matches_cpu_on_device` が 5/5 run で fail → 所有者判断で入力 `qr_input(3,8)` を是正し 2026-10-28 に 5/5 run 全 pass（`README.md:50-52`。tolerance・判定式・baseline は不変） | 2026-10-27（fail）・2026-10-28（pass） | fail `1184586ee`・`95a18c2d3`／pass `2e30c15b2` |
| `metal-linalg-parity-3139/README.md` | Metal 線形代数の Mac mini parity（Stage C） | 演算 | 実測済み（pass）。2026-10-27 は fail（元入力）→ 2026-10-28 に 5/5 run 全 pass（`README.md:9`・`:26`）。収録先は `metal-linalg-lu-chol-qr-2676/rerun-2026-10-28/` | 2026-10-27（fail）・2026-10-28（pass） | pass `2e30c15b2` |
| `metal-m64-thin-tile-ab-macmini-3209/README.md` | m=64 細タイル候補の A/B | — | 性能計測（parity 判定対象外） | - | - |
| `metal-max-reduce-3475/README.md` | Metal f32 `max` reduction | 演算 | 実測済み（pass） | 2026-10-19 | Metal `922c285e0` |
| `metal-phase2-4-parity-3172/README.md` | Phase 2〜4 追加 8 機能の Metal 集計（12 本） | NN 層・自動微分・推論・サービング | 実測済み（12 本 pass） | 2026-10-19 | Metal `665370196` |
| `metal-tile-macmini-3181/README.md` | Metal GEMM タイル選択の追跡 | — | 性能計測（parity 判定対象外） | - | - |
| `metal-tile-macmini-3206/README.md` | Mac mini 向けタイル表の追加前後 | — | 性能計測（parity 判定対象外） | - | - |
| `mha-config-sequential-2530/README.md` | `add_multihead_attention_with_config` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `nn-layers-cuda-parity-3860/README.md` | NN 層の CUDA（#3814・#3817〜#3827。19 本） | NN 層 | 実測済み（19/19 pass。`README.md:14`）。雛形の nn 族 0 件による undetermined は完了判定に数えない旨を記録（`README.md:16`。#4026 で登録） | 2026-11-04 | `77c2cbd58` |
| `nn-layers-metal-parity-3861/README.md` | NN 層の Metal（#3814・#3817〜#3827。19 本） | NN 層 | 実測済み（19/19 pass。`README.md:15`）。雛形の nn 族 0 件の扱いは 3860 と同じ（`README.md:17`） | 2026-11-04 | `6c29359cc` |
| `nn-train-template-cuda-parity-3860/README.md` | 学習 parity 雛形の nn 族の CUDA（18 ケース 120 量） | NN 層・最適化・学習ループ | 実測済み（1 本 pass・120 量 fail 0。`README.md:16`） | 2026-11-04 | `bf118b365`（対象 `2f59b9638`） |
| `nn-train-template-metal-parity-3861/README.md` | 学習 parity 雛形の nn 族の Metal（記録 120 行） | NN 層・最適化・学習ループ | 実測済み（120/120 pass。`README.md:18`） | 2026-11-04 | `c71ef53a2`（対象 `2f59b9638`） |
| `nonfinite-ops-2635/README.md` | isnan／isinf／isfinite／nan_to_num | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `onnx-train-cuda-parity-3864/README.md` | ONNX の学習（`bind_autograd`）の CUDA（1 回目） | 相互運用 | 実測済み（1 本 pass・12 量。`README.md:15`）。#4044 の分類 C のため #4046 の rerun を正とする | 2026-11-04 | `a05e3caeb`（対象 `b82b1e5d1`） |
| `onnx-train-cuda-parity-3864-rerun/README.md` | ONNX の学習の CUDA（#4046 の測り直し。正とする回） | 相互運用 | 実測済み（pass。`README.md:14`。route は H＝ホスト実行で、経路の parity） | 2026-11-05 | `284dcc417` |
| `onnx-train-metal-parity-3865/README.md` | ONNX の学習の Metal | 相互運用 | 実測済み（1 本 pass・12 量。`README.md:15`） | 2026-11-04 | `55fa6e3e3`（対象 `b82b1e5d1`） |
| `opp-kernel-gb10-3185/README.md` | GB10 の相手側カーネルの純時間 | — | 性能計測（parity 判定対象外） | - | - |
| `opp-kernel-macmini-3179/README.md` | Mac mini の相手側カーネルの純時間 | — | 性能計測（parity 判定対象外） | - | - |
| `optim-train-template-cuda-parity-3862/README.md` | 学習 parity 雛形の optim 族の CUDA（6 ケース 24 量） | 最適化・学習ループ | 実測済み（1 本 pass・24 量 fail 0。`README.md:16`） | 2026-11-04 | `f81eedd9c`（対象 `2f59b9638`） |
| `optim-train-template-metal-parity-3863/README.md` | 学習 parity 雛形の optim 族の Metal（記録 24 行） | 最適化・学習ループ | 実測済み（24/24 pass。`README.md:20`） | 2026-11-04 | `31656bccc`（対象 `2f59b9638`） |
| `optin-ab-gb10-3190/README.md` | GB10 の opt-in A/B | — | 性能計測（parity 判定対象外） | - | - |
| `optin-ab-macmini-3184/README.md` | Mac mini の opt-in A/B | — | 性能計測（parity 判定対象外）（README は基準 sha 以降に変更。区分は不変） | - | - |
| `packed-sequence-2647/README.md` | PackedSequence | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `pad-modes-2642/README.md` | pad の非定数モード | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `paged-attention-parity-3113/README.md` | paged attention（K-3）の CPU 参照一致と実機 parity | 推論・サービング | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `66537019` |
| `param-groups-resident-cuda-parity-3862/README.md` | param groups と常駐ストアの CUDA（3 本） | 最適化・学習ループ | 実測済み（3/3 pass。`README.md:14`）。optim 族 0 件の件は #4026 で登録済み | 2026-11-04 | `6ba9db57f` |
| `param-groups-resident-metal-parity-3863/README.md` | param groups と常駐ストアの Metal（3 本） | 最適化・学習ループ | 実測済み（3/3 pass。`README.md:14`） | 2026-11-04 | `39cb7d101` |
| `phase4-ops-autodiff-exposure-2678/README.md` | Phase 4 の演算・自動微分の facade 公開（集計） | 演算・自動微分 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `pool3d-ops-2643/README.md` | MaxPool3d／AvgPool3d の `Var` 委譲 | 演算 | 実測済み（pass） | 2026-10-19 | CUDA `b640875f`／Metal `d79f4895` |
| `resident-chain-ab-3257/README.md` | 常駐チェーンの学習・推論の計測（参考系列） | — | 性能計測（parity 判定対象外）（README は基準 sha 以降に変更。区分は不変） | - | - |
| `resident-chain-cpu-train-diag-3611/README.md` | 常駐チェーン CPU train の REGRESS 切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `resident-chain-metal-breakdown-3612/README.md` | Mac mini Metal 常駐チェーンの時間内訳 | — | 性能計測（parity 判定対象外） | - | - |
| `resident-chain-parity-3256/README.md` | 常駐チェーンの loss 列・更新後パラメータの Tape 経路との bit 一致と同期回数 | 最適化・学習ループ | 実測済み（Metal M1〜M4・CUDA C1〜C5 が各 5 run 全 pass。同期回数は Metal M5・CUDA C7）。README に実測日の記載なし | - | `d6295634e`（作業中コミット。crates の tree hash `e72669c2b`） |
| `rocm-parity-3159/README.md` | ROCm（MI300X・gfx942）対 CPU の f32 GEMM parity | バックエンド・ハード | 実機未実測（`README.md:3` の「状態: 未実測」。#3159 は OPEN。§5.3） | - | - |
| `shape-view-ops-2639/README.md` | unbind／movedim／meshgrid 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `shared-target-reuse-repro-4044/README.md` | 共有 target の成果物再利用の再現（開発機 Linux） | — | 開発機での再現（実機を使っていない。parity ではない） | 2026-11-05 | - |
| `shrink-ops-3817/README.md` | Hardshrink・Softshrink の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `similarity-ops-3821/README.md` | CosineSimilarity・PairwiseDistance の GPU 実機 parity 申し送り | NN 層 | 記入欄は未実測（`README.md:3`）。実測は `nn-layers-{cuda,metal}-parity-3860/3861` の 19 本（pass）に含まれる | 2026-11-04（nn-layers 側） | `77c2cbd58`／`6c29359cc`（nn-layers 側） |
| `sme-default-on-3231/README.md` | SME 既定 ON の bit 同一確認 | — | 性能計測（parity 判定対象外） | - | - |
| `sme-default-on-ab-3232/README.md` | SME 既定 ON の前後確認 | — | 性能計測（parity 判定対象外） | - | - |
| `sme-gb10-attribution-3613/README.md` | GB10 後退 2 セルの 3 腕帰属切り分け | — | 性能計測（parity 判定対象外） | - | - |
| `sme-macmini-3176/README.md` | Mac mini の SME ON／OFF A/B | — | 性能計測（parity 判定対象外） | - | - |
| `sme-scale-macmini-3175/README.md` | Mac mini の SME スケール | — | 性能計測（parity 判定対象外） | - | - |
| `sme-thin-m64-ab-3236/README.md` | m=64 専用 SME 薄型カーネルの A/B | — | 性能計測（parity 判定対象外）（README は基準 sha 以降に変更。区分は不変） | - | - |
| `softmin-threshold-ops-2650/README.md` | Softmin・Tanhshrink・Threshold・RReLU | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `speculative-batching-2890/README.md` | speculative decoding・スケジューラの内部実装 | 推論・サービング | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `speculative-batching-facade-2934/README.md` | `inference` の speculative・`BatchScheduler`（facade） | 推論・サービング | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `speculative-sampling-3102/README.md` | サンプリング版 speculative decoding | 推論・サービング | 実測済み（pass。GB10・Mac mini とも 1 本 pass、`README.md:44-45`） | 2026-10-27 | `343d88ee7`（`.rev-stamp` と一致） |
| `stat-reduce-ops-2637/README.md` | median／kthvalue／quantile 等 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `tape-gradcheck-facade-2847/README.md` | `Tape::gradcheck`（facade） | 自動微分 | 実測済み（pass） | 2026-10-09 | `8bbeb874c`（GB10／M4 Max） |
| `tensor-product-ops-2640/README.md` | kron／tensordot／cdist／cross | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `train-chain-exclusion-3596/README.md` | 学習チェーンと Graph capture の ordinal 単位排他の GB10 検証 | — | 性能計測（排他の検証。判定は REGRESSION／UNDETERMINED で完了条件は未達。parity 判定対象外）（README は基準 sha 以降に変更。区分は不変） | - | - |
| `train-gpu-parity-template/README.md` | NN 層・最適化・ONNX の学習の実機 parity 雛形（#3778） | — | 雛形（実測ではない。README 冒頭に「現在の状態: 未実測」） | - | - |
| `train-infer-phases-macmini-3180/README.md` | train／infer の phase 内訳と同期固定費 | — | 性能計測（parity 判定対象外） | - | - |
| `train-step-chain-cuda-3252/README.md` | CUDA の学習チェーン（Tape 経路との bit 一致・同期回数） | 最適化・学習ループ | 実測済み（B1〜B3 各 5 run pass）（README は基準 sha 以降に変更。区分は不変） | 2026-10-24 | `c9c25c785`＋未コミット差分 |
| `train-step-chain-metal-3251/README.md` | Metal の encode-only 学習チェーン（Tape 経路との bit 一致・同期回数） | 最適化・学習ループ | 実測済み（A1〜A3 各 5 run pass） | 2026-10-24 | `f15af2a98`（作業中コミット。crates の tree hash `12e9049f1`） |
| `train-step-graph-3260/README.md` | 学習ステップ全体の Graph capture | — | 性能計測（README なし。parity 判定対象外） | - | - |
| `train-step-graph-ab-3261/README.md` | 学習ステップ全体の Graph capture の GB10 A/B | — | 性能計測（parity 判定対象外）（README は基準 sha 以降に変更。区分は不変） | - | - |
| `train-step-resident-parts-parity-3224/README.md` | 学習ステップ常駐化の部品の実機 parity | 最適化・学習ループ | 実測済み。Mac mini は全 pass、GB10 の初回は C5 が 2/4 fail → 計数修正後の再計測で全 pass（README は基準 sha 以降に変更。区分は不変） | 2026-10-23 | Mac mini `e62cce21b`／GB10 初回 `45347ae0a`・再計測 `b6579c3f5` |
| `transformer-decoder-sequential-2532/README.md` | `add_transformer_decoder_layer` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `transformer-sequential-2533/README.md` | `add_transformer` | NN 層 | 実測済み（pass） | 2026-10-19 | CUDA `a115595c6`／Metal `6570171a` |
| `trig-ops-2634/README.md` | 逆三角・双曲線関数 9 種 | 演算 | 実測済み（pass） | 2026-10-11 | CUDA `784b4e79`／Metal `c8e1ceb1` |
| `var-f64-autograd-backend-parity-3168/README.md` | `Var` f64 自動微分 | dtype | 実測済み（pass） | 2026-10-19 | CUDA `9e2f7ddcc`／Metal `bf894108a` |

区分ごとの件数（合計 172）:

| 区分 | 件数 | 内訳 |
|---|---|---|
| 実測済み | 97 | 0.12.0 の 70 + §5.4 の 9 + Phase 10 の 17 ディレクトリ（issue 数 18 とは別。§5.2 の「ディレクトリ」列の数。`create-graph-*` 4・`nn-layers-*` 2・`nn-train-template-*` 2・`param-groups-resident-*` 2・`optim-train-template-*` 2・`onnx-train-*` 3・`gpu-linalg-optin-*` 2）+ `metal-batch-norm-2e20-4031-rerun2` 1 |
| 実機未実測 | 3 | `rocm-parity-3159`・`f64-create-graph-3786`・`f64-jvp-facade-3788` |
| 記入欄は未実測（別ディレクトリで実測） | 8 | §5.1 の 8 件 |
| 判定対象外 | 3 | 0.12.0 から不変（`facade-custom-function-2549`・`int8-mma-probe-2608`・`int8-quant-baseline-2609`） |
| 性能計測 | 55 | 0.12.0 の 37 + 新しい 18（`cpu-gemm-*` 10・`cpu-pack-kernel-breakdown-3614`・`resident-chain-*` 2・`sme-gb10-attribution-3613`・`framework-compare-0.1{2,3}.0-*remeasure` 4） |
| 無効 | 2 | `metal-batch-norm-2e20-4031`・`metal-batch-norm-2e20-4031-rerun` |
| 雛形 | 3 | §5.1 の 3 件 |
| 開発機での再現 | 1 | `shared-target-reuse-repro-4044` |

- 実機未実測 3 件のうち `rocm-parity-3159` は §5.3、`f64-*` の 2 件は §5.6 と §5.8 で扱う。
- 実機未検証（#2683 未完了）は 0 件のまま（§2 の 5）。
- `framework-compare-0.1{2,3}.0-*remeasure` は 6 フレームワークの性能の同席計測で、`tests-cuda`・`tests-metal` の実機テストのログを含むが、記入欄の主題は性能であり、parity の判定対象としては数えない（対応表の行は「—」）。

### 5.6 README の無い未実測

| 項目 | 現状 | 根拠 |
|---|---|---|
| # 4 の `Nuc`・`Spectral`（#3991） | 実装済み。`Nuc` の縮退拒否（Q5）だけが測定済み。値の経路の GPU 実機 parity は未実測（ケースは #4052 で追加。実測は #4052 で行う） | `docs/autodiff-higher-order-grad-decision.md` の §76 の「状態」行（§3 の自動微分行の根拠と同じ） |
| #3867（ROCm を facade 経由で測る） | 未実測。ディレクトリを作っていない | §5.3 |

### 5.7 測定コミットと基準 sha の `crates/` 差分

`git diff --shortstat <測定コミット> 1cbb4759 -- crates` の結果と、`git merge-base --is-ancestor <測定コミット> 1cbb4759` による main の履歴上かどうか。差分がある以上、基準 sha での pass は未確認である。

| 測定コミット | 基準 sha までの `crates/` 差分 | 位置づけ |
|---|---|---|
| `b82b1e5d1`（#3856・#3858・#3864・#3865 の対象） | 3 files changed, 737 insertions(+), 4 deletions(-) | main の履歴上 |
| `2f59b9638`（`*-train-template-*` の対象） | 2 files changed, 146 insertions(+), 1 deletion(-) | main の履歴上 |
| `2b00e494b`（#3866 の対象） | 3 files changed | main の履歴上 |
| `6852ef6bd`・`48d0263ac`・`0c170cee2`・`bbfcaee6b`・`77c2cbd58`・`6c29359cc`・`6ba9db57f`・`39cb7d101`・`a05e3caeb`・`55fa6e3e3`・`49a470f4b`（Phase 10 の実行コミット） | 各 3 files changed | main の履歴外（作業ブランチ上のコミット） |
| `bf118b365`・`c71ef53a2`・`f81eedd9c`・`31656bccc`（`*-train-template-*` の実行コミット） | 各 2 files changed | main の履歴外 |
| `82326b253`（#4031 rerun2。有効）・`284dcc417`（#3864 rerun） | 0 files changed | main の履歴外 |
| `a0ae7bdbc`・`3493be202`・`cc3917d93`（#4031 の無効 2 回とその実ビルド） | 0 files changed | main の履歴外 |
| `358b77554`・`343d88ee7`・`901a2b1f6`・`045a1241a` | 各 306 files changed | main の履歴上 |
| `578e820d2` | 305 files changed | main の履歴上 |

- 差分の中身を確かめたもの: `b82b1e5d1` の 3 files は `crates/backend-metal/src/batch_norm_model.rs`・`crates/backend-metal/tests/batch_norm_parity.rs`（#4031 の Metal BatchNorm 2e20）と `crates/facade/tests/train_gpu_parity.rs`（#4026 の nn・optim 族の登録）である。`2f59b9638` 以降の 2 files は #4031 の 2 本だけで、その #4031 は rerun2（`82326b253`、差分 0）で有効な pass を測っている。
- 上の差分の中身は `b82b1e5d1`・`2f59b9638`・`2b00e494b` だけで確かめた。それ以外のコミットは件数のみで、中身は確かめていない。
- 解決できない表記（作業ブランチ上で消えたコミットなど）: `18479ad49`・`f175ea28f`・`1184586ee`・`95a18c2d3`・`2e30c15b2`（§5.4 の 2026-10-27／28 の実測。この clone に無いため差分を取れない）。
- 0.12.0 から引き継いだ 101 件の測定コミットの差分は、0.12.0 §5 の表（基準 `1009825c`）のままで、本書では取り直していない。

### 5.8 食い違いと申し送り

- **追跡表の先頭語と本文の食い違い**: 追跡表 `docs/compat-matrix-ari-tracker.md` の # 7・# 9〜# 15・# 22 の「実機 parity」欄は、先頭の語が「未実測」のまま残っているが、本文には GB10・Mac mini とも pass が書いてある（本書 §3 の NN 層・最適化・学習ループ行と §5.2 を参照）。追跡表は凍結済み（#3869 の §5 手順 6）なので本 PR では直さず、ここに記録する。
- **3786・3788 の実機 parity を追跡する issue が無い**: `f64-create-graph-3786`・`f64-jvp-facade-3788`（f64 の `create_graph` と `jvp`）は CPU のみ実行され、CUDA・Metal は `#[ignore]` テストを用意しただけで未実測である。Phase 10 の子 issue にも対応する issue が見つからなかった。起票は所有者の承認が要るため、本 PR では行わず、申し送りとして PR 本文に書く。
- **# 4 の `Nuc`・`Spectral` の値の経路（`Prod` は #3857・#3859 の `scatter_reduce` 族で pass 済み）と、#3159・#3867 の MI300X 計測**は未実測のまま、所有者の判断待ちである。
- **`framework-compare-0.1{2,3}.0-*remeasure`**: 性能の同席計測で、本付録は parity の判定対象外として数えた。スコアボードへの反映は #3871 の範囲である。

## 6. スコープ外・申し送り・所有者への報告

### 6.1 行判定を下げた 2 行（所有者への報告）

追跡表 §5 手順 1 のとおり、「未確定」が残る行は監査の前に所有者へ報告する対象である。自動運転のため issue へのコメントは投稿していない。**報告は main セッションが行う**。

| 項目 | 状態 | 解除条件 |
|---|---|---|
| # 4 `create_graph` の B 区分（自動微分行） | 実装は済み（#3977・#3991）。`Nuc`・`Spectral` の値の経路の GPU 実機 parity が未実測（`Prod` は pass 済み）。所有者コメント `-6604937225` の条件を満たさない | `Nuc`・`Spectral` の値の経路の実機 parity の確定（#4052）。確定後、追跡表の # 4 を「公開」へ更新して再監査する |
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

`crates/`、`docs/spec/`、tolerance 定数・parity baseline・ガードレール閾値、依存、CI、`docs/compat-api-scope.md`、0.9.0／0.11.0／0.12.0 版の本文、`docs/perf/logs/*`（#3870 の付録表も読むだけで、回収ログと `SHA256SUMS` を変えていない）、
`scripts/tally-compat-tracker.py`、スコアボードの Artifact と生成スクリプト。追跡表 `docs/compat-matrix-ari-tracker.md` は凍結の注記の追記だけである。
