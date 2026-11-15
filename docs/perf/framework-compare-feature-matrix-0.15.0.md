# フレームワーク横並びスコアボード「役割・機能の対応表」v0.15.0 再監査

イシュー #4179（ツリー #4140 の Phase 6。#4177・#4178 の判定記録と同じ PR）。0.14.0 事前監査 `docs/perf/framework-compare-feature-matrix-0.14.0.md`（#3869）で判定した 9 行
（dtype／自動微分／演算の範囲／NN 層・モデル構築／最適化・学習ループ／バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービング）を、
公開済みタグ **v0.15.0**（2026-11-14）の公開面で再監査し、各行の根拠を「公開／未公開／対象外／保留／実機 parity」の 5 区分に分けて記録する。
0.14.0 版と同じ節構成・同じ規則（§2）・同じ入力（追跡表 `docs/compat-matrix-ari-tracker.md`〈#3717〉・集計 `scripts/tally-compat-tracker.py`）で、
基準は main の事前監査ではなく**タグ v0.15.0** である。スコアボード（`docs/perf/logs/framework-compare-0.15.0-remeasure/scoreboard/`）の「役割・機能の対応表」はこの結果をそのまま表示する。

**結論**: 9 行の判定は 0.14.0 から**変わらない**。「ある」は 8 行（dtype・自動微分・演算の範囲・NN 層・モデル構築・最適化・学習ループ・相互運用・事前学習済みモデル・推論・サービング）、
「部分的」のまま残るのは 1 行（バックエンド・ハード。追跡表の「未確定」# 24 の 1 件が原因）である。「未公開」は 9 行すべてで 0 件。理由は次の 3 つで、いずれも機械的に確かめた（§1・§4）。

1. facade の公開面は v0.14.0 から v0.15.0 で**変わっていない**（`crates/facade/src`・`hf-hub`・`backend-rocm`・`onnx-interop` の `src` に `pub` 宣言の増減が 0 件。`crates/facade/tests/api_surface.rs` は無変更。移行ノート `docs/migration/0.15.0.md` の「facade の公開 API の追加・削除・シグネチャ変更はない」と一致）。
2. 追跡表（29 項目）と `scripts/tally-compat-tracker.py` は v0.14.0 以降**無変更**で、集計は 0.14.0 の最終値（残件 1〈# 24〉・公開 26・対象外 2）と同じ（§4.2）。
3. # 24（ROCm の facade 公開）の解除条件である #3159・#3867 は OPEN のままで、MI300X の測定は**未実測**（§5.2）。緩めずその形で記録し、所有者への報告事項に挙げた（§6）。

変わったのは根拠の記述である: `backend-rocm`・`hf-hub` は 0.14.0 で crates.io へ初回公開済み（§2 の 1）、CUDA 推論 Graph が既定 ON になった（§3.1）、v0.15.0 で `create_graph` の総括判定の実装が変わった（実機 parity は 0.14.0 版の測定コミットのまま。§3.1・§5.3）。

## 1. 基準面の証明

判定基準は**公開済みタグ v0.15.0 の公開面**（`crates/facade` と、追跡表が公開に数える補助クレート）とする。0.14.0 版は main の事前監査（基準 sha `1cbb475989e32051b337efb81b38b843de80145b`）だったため、
本書の `path:line` は、その 0.14.0 版の 190 件を基準 sha から v0.15.0 へ**引き直した**ものである（`git show v0.15.0:<path>` の行で再解決できる）。

- **基準**: タグ `v0.15.0` = `22a475e2df0cfcbf5854d827a79ddc1e7df4a4b9`（2026-11-14 21:44:22 +0900）。前版タグ `v0.14.0` = `b6ae8170168c768c1fca152dae4173975b0a53d1`。docs/spec の pin は `3e9ea28f4b66c77c348a0cf4cf3ba08bcc4522ab`（`git ls-tree v0.15.0 docs/spec`。0.14.0 版と同じ）。
- ワークスペースの版数は `Cargo.toml:163` の `version = "0.15.0"`（`git show v0.15.0:Cargo.toml`）。公開済みで、タグは作成済みである。
- **行番号の引き直し**: 0.14.0 版の `path:line`（`crates/…` の 190 件）を、基準 sha から v0.15.0 へ `git diff -U0` のハンクで写し、写し先の行の文字列が元の行と同一であることを確かめた。結果は 190 件すべて一致（うち 32 件は行番号が動いた。`crates/facade/src/lib.rs` の rustdoc と `crates/facade/src/compat/sequential.rs` への挿入による）・未解決 0 件。
  `docs/` 配下の行番号（`README.md:3` など）は、`docs/perf/logs` の該当ディレクトリが v0.14.0 から無変更のため引き直していない。決定記録 `docs/autodiff-higher-order-grad-decision.md` の §73 の見出し行だけは動いたため直した（5388 行 → 5390 行）。
- 保留ガード（`*HoldDoctestGuard`）の struct 名の集合は v0.14.0 と v0.15.0 で 43 件のまま同一である（`crates/facade/src/lib.rs` から `^struct \w+HoldDoctestGuard` を抽出して比較）。0.12.0 §2 規則 2 と同じく、struct 名の有無は未公開の根拠にしない。
- `docs/compat-unlisted-holds-inventory.md`・`docs/compat-matrix-ari-tracker.md`・`docs/compat-api-scope.md`・`scripts/tally-compat-tracker.py` は v0.14.0 から差分が無い（`git diff --stat v0.14.0 v0.15.0 -- <各ファイル>` が空）。
- 実機 parity の測定コミットは基準（タグ）と別で、「測定済み」はその測定コミットの内容が pass したことだけを意味する。実機は動かしていない。本書の調査は v0.15.0 のソース・決定記録・GitHub の issue の状態の読み取りだけで行った。

```
$ git diff --shortstat v0.14.0 v0.15.0 -- crates
 78 files changed, 18695 insertions(+), 809 deletions(-)
$ git diff --shortstat 1cbb475989e32051b337efb81b38b843de80145b v0.15.0 -- crates
 78 files changed, 18743 insertions(+), 812 deletions(-)
$ git diff --shortstat v0.14.0 v0.15.0 -- crates/facade crates/hf-hub crates/backend-rocm crates/onnx-interop
 13 files changed, 233 insertions(+), 46 deletions(-)
```

v0.14.0 → v0.15.0 の `crates/` 差分 78 files の内訳（`git diff --numstat` をクレート別に集計）: autodiff 16（+14167・-518）・backend-metal 22（+1559・-85）・backend-cpu 14（+1547・-102）・backend-cuda 11（+960・-48）・facade 9（+220・-33）・tensor-core 2（+229・-10）・hf-hub 2・onnx-interop 1・backend-rocm 1（後ろの 3 つは版数の行とコメントだけ）。
facade の 9 files は `Cargo.toml`・`src/lib.rs`（rustdoc の 2 か所だけ）・`src/compat/sequential.rs`（`predict_recorded`・`predict_resident`・`predict_resident_input` の冒頭に `enter_inference_scope()` を足した 3 か所）・テスト 6 本で、`pub` 宣言の増減はない。

```
$ git diff v0.14.0 v0.15.0 -- crates/facade/src crates/hf-hub/src crates/backend-rocm/src crates/onnx-interop/src | grep -cE '^[+-]\s*pub '
0
```

新しい `pub` 宣言は内部クレート `backend-cpu` にだけある（`enter_inference_scope`・`InferenceScopeGuard`・`infer_scope` モジュール・`sme_production_enabled` ほか）。facade はこれらを再エクスポートしていない（`crates/facade/src/lib.rs` の `pub use fandhe_ai_backend_*` は `CudaGemmPrecision` の 1 件だけで、v0.14.0 と同じ）。

## 2. 判定規則

0.14.0 版 §2 の規則 1〜3（件数表記・保留ガードの扱い・facade 外クレート・「未確定」の数え方・承認の出所）を**そのまま踏襲し、緩めていない**（下は 0.14.0 版 §2 の再掲で、規則 3 の読み方の「crates.io にはまだ公開していない」だけを事実に合わせて直した）。
v0.15.0 で増えた所有者の承認はない。得ていない承認は書かない。


0.12.0 版 §2 の規則 1〜6（件数表記・保留ガードの扱い・facade 外クレート・承認の出所・実機未検証・保留の定義）を**そのまま踏襲し、緩めていない**。行内の「未公開」0 件かつ「保留」0 件の行だけを「ある」とする。
追加点は次の 3 つで、いずれも判定の緩和ではない。

1. **規則 3（facade 外クレート）の 0.14.0 での読み方**: 根拠は #3771 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469>）だけである。
   - `hf-hub` は案 A で、`HfHub::download` は `pub(crate)` のまま「公開に数える」。`docs/compat-api-scope.md` §0 の限定例外として扱う。
   - `backend-rocm` の crates.io 公開は 0.14.0 で承認済みである。ただし ROCm の facade 同梱は #3159・#3867 の parity pass が条件である（回答の点 1）。
   - 両クレートとも `publish = true` になった（`crates/hf-hub/Cargo.toml:36`・`crates/backend-rocm/Cargo.toml:33`）ため、規則 3 の文面の条件（`publish = false`）にはもう当たらない。**0.14.0（2026-11-05）で crates.io へ初回公開済み**（`docs/crates-io-publishing-order.md`。v0.15.0 でも `publish = true`）。ただし公開は ROCm の facade 同梱の条件を満たしたことを意味しない（`backend-rocm` の公開は所有者判断による例外で、ROCm 実機 parity は未実測のまま）。
2. **「未確定」の数え方**: 追跡表の「未確定」は新しい区分を作らず、0.12.0 での区分（# 24 は「保留」）のまま数える。「公開」には上げない。初版では # 4 も「未確定」（0.12.0 での区分は「保留」）として保留に数えたが、所有者の判断（#3711 の `-6656856029`）で追跡表の # 4 が「公開」になったため、公開に数え直した。
3. **承認の出所（規則 4）への追加**: 0.12.0 版の出所に、次の所有者コメントを加える（作成者・日時は `gh api repos/Fandhe-AI/fandhe-ai/issues/comments/<id>` で確認した。いずれもリポジトリ所有者のコメント。本文は転記せず要約する）。
   - Phase 3 の回答 6 件（2026-11-01）: #3766 `-6554563674`（06:42:58Z。低精度演算の配置と `TapeF64` の `jvp` 系）、#3767 `-6554564013`（06:43:00Z。`create_graph` の B／C 区分と拒否範囲）、#3768 `-6554564412`（06:43:03Z。NN 層）、
     #3769 `-6554564758`（06:43:05Z。最適化）、#3770 `-6554565145`（06:43:08Z。相互運用と推論）、#3771 `-6554565469`（06:43:11Z。ROCm と HF Hub）。
   - #3711 のコメント 3 件: `-6570118646`（2026-11-02。運用の読み替え）、`-6594034494`（2026-11-03。#3977・#3978 の起票）、`-6604937225`（2026-11-03。#3978 への回答と、行 4 の「実装と実機 parity が確定するまで未確定」という条件）。
   - 再判定で加えた所有者コメント 2 件（2026-11-05）: #4052 `-6656452767`（11:08:26Z。`Nuc`・`Spectral` の Metal 実機 parity を族単位の pass と扱う）、#3711 `-6656856029`（11:29:31Z。上の条件が満たされたとして追跡表 # 4 を「未確定」から「公開」へ動かす）。
   - 「承認」はこれらのコメント ID と各イシュー番号の事実に限って書き、得ていない承認は書かない。

## 3. 対応表

セル内の `D:` は `docs/` 配下の決定記録を指す。「0.14.0 事前監査」の列は 0.14.0 版 §3 の判定、「0.15.0 再監査」の列は本書の判定である。追跡表の項目は「# n」で示す。
行番号は §1 のとおり v0.15.0 のものに引き直した。各行の件数・保留は 0.14.0 版から**変わらない**（§4.1）。行末の「**0.14.0 からの変更**」は、v0.14.0 → v0.15.0 の `crates/` 差分のうち当該行に関わるもの（§3.1）で、判定を動かさない。

| 行 | 0.14.0 事前監査 | 0.15.0 再監査 | 根拠（公開／未公開／対象外／保留／実機 parity） |
|---|---|---|---|
| dtype | ある | ある | **公開（6 件）:** f64 の自動微分 `TapeF64`（`crates/facade/src/lib.rs:2783`）／`VarF64`（`crates/facade/src/lib.rs:3031`）；低精度 forward の opt-in `matmul_low_precision`（`crates/autodiff/src/var.rs:3659`）；dtype 変換と型付き演算 `CastDType`（`crates/facade/src/lib.rs:428`）・`TypedOps`（`crates/facade/src/lib.rs:440`）・`typed_ops_f64`（`crates/facade/src/lib.rs:2670`）；AMP `compile_with_amp`（`crates/facade/src/compat/training.rs:2306`）；CUDA GEMM の精度 opt-in `set_cuda_gemm_precision`（`crates/facade/src/lib.rs:3913`。既定は FP32 のまま）；# 2 の `TapeF64` の `jvp`（`crates/facade/src/lib.rs:2889`）・`jacfwd`（`crates/facade/src/lib.rs:2921`）・`jvp_multi`（`crates/facade/src/lib.rs:2961`）・`jvp_create_graph`（`crates/facade/src/lib.rs:3012`）と f64 の `backward_create_graph`（`crates/facade/src/lib.rs:2847`）。正ガードは `f64_autograd_public_surface_is_exactly_approved_delegations`（`crates/facade/tests/api_surface.rs:41871`）と `f64_jvp_family_usage_doctest_is_present_and_compiled`（`crates/facade/tests/api_surface.rs:46580`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の未公開 1 件（# 1）は対象外へ移った<br>**対象外（4 件）:** # 1 `Var` 以外への低精度演算の配置（承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674>。禁止経路専用で `VarLowPrecisionOpsHoldDoctestGuard`〈`crates/facade/src/lib.rs:7083`〉を維持し、正ガードは `var_low_precision_ops_hold_doctest_globs_all_pub_modules`〈`crates/facade/tests/api_surface.rs:29041`〉・`var_low_precision_ops_hold_doctest_probe_body_matches_fixed_contract`〈`crates/facade/tests/api_surface.rs:29065`〉・`facade_does_not_reexport_or_declare_low_precision_ops`〈`crates/facade/tests/api_surface.rs:29232`〉）；f16／bf16 を値型とする `Tape`／`Var`（#3021 の `#issuecomment-6111746880` の項目 2。0.12.0 §3 から不変）；sparse／complex（#2499 の `#issuecomment-6033824965`。0.12.0 §3 から不変）；int8 量子化（spec の除外事項「Won't・条件付き」。0.12.0 §3 から不変）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 1 件（# 2。`TapeF64` の `jvp` 系）は上のとおり公開へ移った。承認は <https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674><br>**実機 parity:** f64 と低精度 opt-in は 0.12.0 §5 のとおり実測済み（pass）。# 2 の追跡表の欄は「該当 issue なし」、# 1 も「該当 issue なし」（新しい数値経路なし）。int8 は性能判定対象外 |
| 自動微分 | ある | ある | **公開（16 件）:** 0.12.0 の 12 件（`backward_create_graph`〈`crates/facade/src/lib.rs:1623`〉、`custom`〈`crates/facade/src/lib.rs:2389`〉、`register_backward_hook`〈`crates/facade/src/lib.rs:2447`〉、`jacobian`〈`crates/facade/src/lib.rs:1706`〉・`hessian`〈`crates/facade/src/lib.rs:1743`〉、`gradcheck`〈`crates/facade/src/lib.rs:2299`〉・`backward_detect_anomaly`〈`crates/facade/src/lib.rs:2266`〉、`vjp`〈`crates/facade/src/lib.rs:1793`〉・`hvp`〈`crates/facade/src/lib.rs:1851`〉・`vmap`〈`crates/facade/src/lib.rs:2169`〉、`jvp`〈`crates/facade/src/lib.rs:1925`〉・`jacfwd`〈`crates/facade/src/lib.rs:1962`〉、`jvp_multi`〈`crates/facade/src/lib.rs:2030`〉、`jvp_create_graph`〈`crates/facade/src/lib.rs:2114`〉、`vmap_grad`〈`crates/facade/src/lib.rs:2233`〉、`set_deterministic`〈`crates/facade/src/lib.rs:3331`〉、`var_no_grad`〈`crates/facade/src/lib.rs:638`〉・`backward_accumulate`〈`crates/facade/src/lib.rs:657`〉・`detach`〈`crates/autodiff/src/var.rs:278`〉・`checkpoint_from`〈`crates/autodiff/src/var.rs:2502`〉。宣言行を基準 sha で引き直し、すべて存在する）；# 2 `TapeF64` の `jvp` 系（dtype 行と同じ）；# 3 `create_graph` の C 区分 24 Op（実装は #3789〜#3796。対象 Op 集合の正ガードは `create_graph_supported_op_set_matches_approved_list`〈`crates/facade/tests/api_surface.rs:47984`〉、facade 挙動テストとの対応表は `create_graph_facade_coverage_map_matches_supported_set`〈`crates/facade/tests/api_surface.rs:48174`〉、定義元は `supports_create_graph`〈`crates/autodiff/src/tape.rs:2654`〉）；# 4 `create_graph` の B 区分（実装は #3797〜#3808・#3977・#3991。`Prod` は `scatter_reduce_prod_create_graph_via_facade`〈`crates/facade/tests/create_graph_facade.rs:897`〉、`Nuc`・`Spectral` は `supports_create_graph` の `MatrixNorm` の腕〈`crates/autodiff/src/tape.rs:2853`〉と `matrix_norm_nuc_spectral_create_graph_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2028`〉。縮退入力の拒否は同:2054。初版では「未確定」として保留に数えたが、所有者の判断〈#3711 の `-6656856029`、2026-11-05〉が、コメント `-6604937225` の条件〈#3977・#3991 の実装と実機 parity の確定〉を満たしたとして追跡表の区分を「公開」へ動かした。承認は <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>）；# 5 二階微分の拒否範囲と意味論（`Inv`・`Solve`・`SlogdetLogAbsDet` の overflow 入力の拒否を `inv_solve_overflow_inputs_are_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2738`〉が固定）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（4 件）:** D-0 の除外名（`Var::custom` 系・`functional_ops` の自由関数・gradcheck の自由関数。#3018 の `#issuecomment-6111643437`）；ネイティブ forward-mode の JVP 規則とバッチ規則型 vmap（#2499 の `#issuecomment-6033824965`）；`jvp` の f16 対応（#3021 の `#issuecomment-6111746880` の項目 3）；`create_graph` が構造的に定義できない Op（`ResidentLeaf`・`LinearResident`・`LinearAct`・`Custom`・`OneHot` と、`CtcLoss` の恒久 fail-closed〈`ctc_loss_create_graph_is_rejected_via_facade`、`crates/facade/tests/create_graph_facade.rs:2658`〉。`Inv`・`Solve`・`SlogdetLogAbsDet` の拒否範囲の恒久化。承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3767#issuecomment-6554564013>）。件数は 0.12.0 の 4 件のまま（集計の単位が項目か区分かの差で、増減はない）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 4 件（# 2〜# 5）はすべて公開へ移った（# 4 は上の所有者の判断による）。**範囲外の注記（件数に数えない）**: `CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（`cosine_similarity_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2671`〉・`pairwise_distance_create_graph_is_rejected_via_facade`〈`crates/facade/tests/create_graph_facade.rs:2682`〉。決定記録 `docs/autodiff-higher-order-grad-decision.md` §73〈5390 行〉は「承認済み（#3768 Q9 (a)）・別設計待ち・恒久ではない・起票は所有者の承認事項で未起票」とする）。29 項目の外で、0.12.0 §2 規則 6 の「保留」（所有者の回答が無い論点）には当たらないが、所有者への報告事項に挙げる（§6）<br>**実機 parity:** # 2 は該当 issue なし。# 3・# 5 は pass（2026-11-04・`<cuda-node>` `6852ef6bd` と `48d0263ac`、`<metal-node>` `0c170cee2` と `bbfcaee6b`。#3856〜#3859。記録は `docs/perf/logs/create-graph-c-cuda-parity-3856/`・`docs/perf/logs/create-graph-b-cuda-parity-3857/` と各 metal 版）。# 4 は B 区分の 12 族が pass（同上）。`Prod` は `scatter_reduce_prod` ケースが `scatter_reduce` 族で pass 済み（#3857〈`docs/perf/logs/create-graph-b-cuda-parity-3857/README.md:47`〉・#3859。両測定コミットとも #3977 の `362ab06c4` を含む）。`Nuc`・`Spectral` の値の経路は、CUDA は族単位で pass（2026-11-05・`<cuda-node>`・`8966befde`。#4052。ケース別の数値は出ず、ケースの実行は代替証拠。記録は `docs/perf/logs/create-graph-nuc-spectral-4052-cuda/`）、Metal も族単位で pass（2026-11-05・`<metal-node>`・`df9e060a0`。#4052。ログにケース名の直接の跡は無く、間接の根拠〈ソースの指紋の一致・族の全ケースを打ち切らずに判定するコード構造〉にもとづき所有者の判断〈2026-11-05、#4052 の `-6656452767`〉で族単位の pass と扱う。記録は `docs/perf/logs/create-graph-nuc-spectral-4052-metal/`）<br>**0.14.0 からの変更（§3.1 の 4・5）:** `create_graph` の総括判定（拒否範囲）の実装が変わった（#4068 の CE の二階微分の修正・#4070 の WeightNorm の拒否範囲の縮小・#4101 ツリーの `Lin` の上界への置き換え。決定記録 `docs/autodiff-higher-order-grad-decision.md` §77〜§83）。対象 Op 集合・facade の公開面・正ガード（`create_graph_supported_op_set_matches_approved_list`）は不変で、件数・判定は動かない。**上の「実機 parity」の pass は 0.14.0 版の測定コミットの内容に対するもので、v0.15.0 の判定実装では再測定していない**（§5.3） |
| 演算の範囲 | ある | ある | **公開（8 件）:** 0.12.0 の 7 件（`Var` 委譲、`rfft`〈`crates/autodiff/src/var.rs:5112`〉、`median`〈`crates/autodiff/src/var.rs:5356`〉・`kron`〈`crates/autodiff/src/var.rs:5536`〉・`pad_with_mode`〈`crates/autodiff/src/var.rs:5577`〉、`max_pool3d`〈`crates/autodiff/src/var.rs:5594`〉・`conv_transpose3d`〈`crates/autodiff/src/var.rs:5703`〉・`max_unpool2d`〈`crates/autodiff/src/var.rs:5747`〉、`fold`〈`crates/autodiff/src/var.rs:5807`〉・`local_response_norm`〈`crates/autodiff/src/var.rs:5825`〉・`weight_norm`〈`crates/autodiff/src/var.rs:5839`〉・`spectral_norm`〈`crates/autodiff/src/var.rs:5846`〉、`bce_with_logits_loss_with`〈`crates/autodiff/src/var.rs:4953`〉、`bincount`〈`crates/facade/src/lib.rs:1648`〉）；# 6 GPU 線形代数の opt-in `set_cuda_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3843`）・`set_metal_gpu_linalg_enabled`（`crates/facade/src/lib.rs:3870`）。正ガードは `gpu_linalg_optin_is_reachable_via_facade`（`crates/facade/tests/api_surface.rs:44513`）と `workspace_declares_gpu_linalg_optin_fn_names_only_in_approved_locations`（`crates/facade/tests/api_surface.rs:44567`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 6）は公開へ移った<br>**対象外（4 件）:** 0.12.0 §3 から不変（D-0 の除外名、sparse／complex、Metal の eigh・SVD と CUDA の SVD、int／bool 演算の網羅。承認は 0.12.0 版の出所のとおり）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 6 は pass（#3866。facade 経由・ゲート ON。`<cuda-node>` は 6 本、`<metal-node>` は 5 本〈eigh は範囲外〉。2026-11-04。記録は `docs/perf/logs/gpu-linalg-optin-cuda-parity-3866/`・`docs/perf/logs/gpu-linalg-optin-metal-parity-3866/`）。0.12.0 時点で未実測だった `cuda-linalg-*`・`metal-linalg-*` の README の記入欄は、2026-10-27／28 に実測済み（pass。`metal-linalg-*` は 2026-10-27 の fail の後、入力を是正して 2026-10-28 に pass。§5.4）<br>**0.14.0 からの変更（§3.1 の 5）:** `logcumsumexp` の 1 階 VJP を大きな共通オフセットで安定化した（#4069。`crates/tensor-core/src/cumulative.rs`）。公開面は不変 |
| NN 層・モデル構築 | ある | ある | **公開（15 件）:** 0.12.0 の 6 件（`add_max_pool3d`〈`crates/facade/src/compat/sequential.rs:2423`〉・`add_avg_pool3d`〈`crates/facade/src/compat/sequential.rs:2551`〉・`add_conv_transpose3d`〈`crates/facade/src/compat/sequential.rs:1723`〉・`add_fold`〈`crates/facade/src/compat/sequential.rs:2487`〉・`add_unfold`〈`crates/facade/src/compat/sequential.rs:2455`〉・`add_local_response_norm`〈`crates/facade/src/compat/sequential.rs:2523`〉、MaxUnpool の層経路〈`crates/facade/src/compat/functional.rs:552`〉、`parametrize_weight_norm`〈`crates/facade/src/compat/sequential.rs:838`〉・`parametrize_spectral_norm`〈`crates/facade/src/compat/sequential.rs:903`〉、`save_model`〈`crates/facade/src/compat/model_io.rs:283`〉・`load_model`〈`crates/facade/src/compat/model_io.rs:306`〉と kind 追加の `spec_kind`〈`crates/facade/src/compat/model_io.rs:336`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 7 `add_upsample_scale_factor`（`crates/facade/src/compat/sequential.rs:2978`。正ガード `compat_sequential_spatial_add_methods_have_approved_signatures`〈`crates/facade/tests/api_surface.rs:14947`〉）；# 8 `Sequential` への `nn::Module` 実装（`crates/facade/src/compat/sequential.rs:706`。正ガード `compat_sequential_nn_module_impl_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:48320`〉）；# 9 `add_hardshrink`（`crates/facade/src/compat/sequential.rs:1267`）・`add_softshrink`（`crates/facade/src/compat/sequential.rs:1291`）、`Var` 側は `hardshrink`（`crates/autodiff/src/var.rs:5937`）・`softshrink`（`crates/autodiff/src/var.rs:5953`）；# 10 `bilinear`（`crates/autodiff/src/var.rs:5981`。正ガード `var_bilinear_ops_method_is_thin_delegation`〈`crates/facade/tests/api_surface.rs:48981`〉）；# 11 `add_batch_norm3d`（`crates/facade/src/compat/sequential.rs:1934`）と rank 5 の `batch_norm`（`crates/autodiff/src/var.rs:1859`）；# 12 `cosine_similarity`（`crates/autodiff/src/var.rs:6015`）；# 13 `pairwise_distance`（`crates/autodiff/src/var.rs:6043`）；# 14 `add_channel_shuffle`（`crates/facade/src/compat/sequential.rs:3060`）、`Var` 側は `channel_shuffle`（`crates/autodiff/src/var.rs:3107`）；# 15 `add_adaptive_avg_pool3d`（`crates/facade/src/compat/sequential.rs:2780`）・`add_adaptive_max_pool3d`（`crates/facade/src/compat/sequential.rs:2807`）・`add_lp_pool1d`（`crates/facade/src/compat/sequential.rs:2588`）・`add_lp_pool2d`（`crates/facade/src/compat/sequential.rs:2623`）・`add_lp_pool3d`（`crates/facade/src/compat/sequential.rs:2658`）・`add_fractional_max_pool2d`（`crates/facade/src/compat/sequential.rs:2840`）・`add_fractional_max_pool3d`（`crates/facade/src/compat/sequential.rs:2869`）。新層の保存 kind の正ガードは `model_io_new_layer_kinds_match_approved_mapping_issue_3828`（`crates/facade/tests/api_surface.rs:51413`）<br>**未公開（0 件）:** なし（0 件）。0.12.0 §8.5 の 9 件（# 7〜15）はすべて公開へ移った<br>**対象外（1 件）:** 層型名の公開（`fandhe_ai::nn::*` の全層型。#3022 の決定記録＝公開対象外 (i)+(iv)。`SpatialLayersHoldDoctestGuard`〈`crates/facade/src/lib.rs:4907`〉は維持）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 7・# 9・# 10・# 11 は、追跡表の欄の先頭語が「未実測」のまま残っているが、本文には `<cuda-node>`・`<metal-node>` とも pass の記録がある（#3860・#3861。2026-11-04。記録は `docs/perf/logs/nn-layers-cuda-parity-3860/`・`docs/perf/logs/nn-layers-metal-parity-3861/`。学習ステップの雛形は `docs/perf/logs/nn-train-template-cuda-parity-3860/`・`docs/perf/logs/nn-train-template-metal-parity-3861/` で pass）。# 8 は該当 issue なし。# 12〜15 の欄は追跡表のとおり写す（個別記録は §5.5 の `shrink-ops-3817` 等の行と、`nn-layers-{cuda,metal}-parity-3860/3861` の 19 本 pass を参照） |
| 最適化・学習ループ | ある | ある | **公開（14 件）:** 0.12.0 の 7 件（optimizer の `Adamax`〈`crates/facade/src/optim.rs:528`〉・`Asgd`〈`crates/facade/src/optim.rs:529`〉・`Lion`〈`crates/facade/src/optim.rs:537`〉・`Rprop`〈`crates/facade/src/optim.rs:538`〉、`PolynomialLr`〈`crates/facade/src/optim.rs:530`〉・`CyclicLr`〈`crates/facade/src/optim.rs:534`〉、`ParamGroup`〈`crates/facade/src/optim.rs:554`〉、`OptimizerStateDict`〈`crates/facade/src/optim.rs:551`〉、`SwaLr`〈`crates/facade/src/optim.rs:539`〉、`fit_with_weights`〈`crates/facade/src/compat/training.rs:3026`〉・`fit_with_train_step`〈`crates/facade/src/compat/training.rs:2928`〉、`train_step_resident`〈`crates/facade/src/compat/sequential.rs:4558`〉）；# 17 EMA と AMP の併用（正ガード `ema_fit_rejection_arms_match_approved_set`〈`crates/facade/tests/api_surface.rs:20715`〉）；# 18 EMA のウォームアップ（`ema_warmup_matches_approved_shape`〈`crates/facade/tests/api_surface.rs:20946`〉）；# 19 param groups と AMP（`compile_with_amp_and_param_groups`、`crates/facade/src/compat/training.rs:2434`）；# 20 param groups と `LrSchedule`（`compile_with_param_groups`、`crates/facade/src/compat/training.rs:2236`。正ガード `param_groups_usage_doctests_are_present_and_compiled`〈`crates/facade/tests/api_surface.rs:18621`〉）；# 21 param groups の `state_dict` 保存（`save_model`・`load_model`。# 20 と同じ正ガード）；# 22 param groups と `DeviceParamStore`（`init_device_param_store_with_param_groups`、`crates/facade/src/compat/sequential.rs:4319`）；# 23 `NAdam`（`crates/facade/src/optim.rs:546`）・`RAdam`（`crates/facade/src/optim.rs:556`）と `Optimizer` の腕（`crates/facade/src/compat/training.rs:313`・`crates/facade/src/compat/training.rs:317`）、保存 kind（`crates/facade/src/compat/model_io/compiled.rs:251`・`crates/facade/src/compat/model_io/compiled.rs:252`）、正ガード `APPROVED_COMPAT_OPTIMIZER_VARIANTS`（`crates/facade/tests/api_surface.rs:22262`。11 種）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の未公開 1 件（# 16）は対象外へ移った<br>**対象外（2 件）:** `FitConfig` への EMA（D-0 の除外名。#3018 の `#issuecomment-6111643437`。機能は `Callback::Ema`〈`crates/facade/src/compat/callbacks.rs:1750`〉で到達できる）；# 16 SWA の `FitConfig` 形（承認: <https://github.com/Fandhe-AI/fandhe-ai/issues/3769#issuecomment-6554564758>。機能は `Callback::Swa`〈`crates/facade/src/compat/callbacks.rs:1752`〉で公開済み。禁止経路専用で `SwaHoldDoctestGuard`〈`crates/facade/src/lib.rs:5725`〉と `swa_hold_doctest_globs_all_pub_modules`〈`crates/facade/tests/api_surface.rs:21058`〉を維持）<br>**保留（0 件）:** なし（0 件）。0.12.0 §8.5 の 7 件（# 17〜23）はすべて公開へ移った<br>**実機 parity:** # 22 は CUDA・Metal ともに実測済み（#3862: `docs/perf/logs/optim-train-template-cuda-parity-3862/README.md:16`、24 量 pass・fail 0。#3863: `docs/perf/logs/optim-train-template-metal-parity-3863/README.md:20`、24 行 pass・fail 0）。# 23 は該当 issue なし。0.12.0 §5 の `dataset-compose-2661`・`iterable-batch-sampler-2662` は host ユーティリティで、2026-10-27 に GB10・Mac mini とも pass（§5.4）。学習ステップ常駐チェーンは 0.12.0 §5 のとおり実測済み |
| バックエンド・ハード | 部分的 | 部分的 | **公開（3 件）:** `tape_for`（`crates/facade/src/lib.rs:3216`）・`available_devices`（`crates/facade/src/lib.rs:3278`）・`set_cuda_gemm_precision`（`crates/facade/src/lib.rs:3913`）；デバイス間の値転送 `transfer`（`crates/facade/src/lib.rs:895`）；分散学習の初期形 `distributed`（`crates/facade/src/lib.rs:255`）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（3 件）:** 0.12.0 §3 から不変（FSDP・ZeRO 系・tensor／pipeline parallel、cudarc の `nccl` feature、int8 量子化）<br>**保留（1 件）:** # 24 ROCm の facade 公開（追跡表では「未確定」。0.12.0 での区分は保留）。実装は済んでいる: `Device::Rocm(usize)`（`crates/tensor-core/src/device.rs:79`）、`resolve_ops` の `Device::Rocm(_)` の腕（`crates/facade/src/lib.rs:3605`）、provider（`crates/backend-rocm/src/device.rs:32`）、`backend-rocm` は `publish = true`（`crates/backend-rocm/Cargo.toml:33`。0.14.0 で crates.io へ初回公開済み）。正ガードは `device_enum_variant_inventory_is_fixed`（`crates/facade/tests/api_surface.rs:45740`）・`facade_depends_on_backend_rocm_only_in_approved_shape`（`crates/facade/tests/api_surface.rs:45776`）・`facade_wires_rocm_into_resolve_ops_and_available_devices`（`crates/facade/tests/api_surface.rs:51959`）・`rocm_device_usage_doctests_are_present_and_compiled`（`crates/facade/tests/api_surface.rs:52039`）。それでも**保留のままにする理由**: #3771 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3771#issuecomment-6554565469>）の点 1 が「facade への同梱は #3159・#3867 の parity が pass することを条件とする」とし、両 issue は OPEN のまま（0.14.0 監査の 2026-11-05 時点は #3159 が更新 07:19:09Z・#3867 が 07:19:10Z。本書の再確認〈2026-11-15〉は #3159 が更新 2026-11-08T11:27:48Z・#3867 が 2026-11-08T11:27:49Z。状態は変わっていない）、MI300X 実機の測定が無い。表記は「条件付き・実験的」（`crates/facade/src/lib.rs:3205`）。解除条件は #3159・#3867 の pass である<br>**実機 parity:** # 24 は `<rocm-node>` で**未実測**（追跡表の欄）。DDP は `ddp-host-collective-gb10-3149` が 2026-10-27 に GB10 で 5 run 全 pass（§5.4）。CUDA GEMM の既定精度は FP32 のまま<br>**0.14.0 からの変更（§3.1 の 1〜3・6）:** 既定の変化（CUDA 推論 Graph 既定 ON〈#4167〉・SME 既定 ON〈aarch64。#4165〉・Metal の steel〈M4Base 正方 1024／2048／4096。#4166〉）と内部クレートの破壊的変更（`backend-cpu` の `kernel_unchecked*` を `pub(crate)` に〈#4285〉）。facade の公開面は不変。CUDA GEMM の既定精度は FP32 のまま。`backend-rocm`・`hf-hub` は 0.14.0 で crates.io へ初回公開済みで、v0.15.0 の差は版数の行とコメントだけ |
| 相互運用 | ある | ある | **公開（7 件）:** 0.12.0 の 4 件（ONNX import `OnnxModel`〈`crates/facade/src/interop/onnx.rs:234`〉と export `from_sequential`〈`crates/facade/src/interop/onnx.rs:364`〉、safetensors の f32 入出力、npy／npz、`save_model`／`load_model`）；# 25 ONNX モデルの学習の入口 `bind_autograd`（`crates/facade/src/interop/onnx.rs:914`。正ガード `onnx_autograd_types_are_reachable_via_facade`〈`crates/facade/tests/api_surface.rs:2770`〉）；# 26 ONNX export の `Sigmoid`（`crates/onnx-interop/src/onnx/export_nn.rs:367`。正ガード `onnx_export_sigmoid_matches_approved_shape_issue_3844`〈`crates/facade/tests/api_surface.rs:51699`〉）；# 27 safetensors の f32 以外の型（`crates/facade/src/interop/safetensors.rs:196`・`crates/facade/src/interop/safetensors.rs:197` の再エクスポート。正ガード `interop_safetensors_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:3083`〉）<br>**未公開（0 件）:** なし（0 件）<br>**対象外（2 件）:** 0.12.0 §3 から不変（Python バインディングと TF 系形式、`Tensor` への npy／npz の inherent メソッド）<br>**保留（0 件）:** なし（0 件）。0.12.0 の保留 1 件（# 26）は公開へ移った<br>**実機 parity:** # 25 は pass（`<cuda-node>` の測り直しを含む。`docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`。#4046）。# 26・# 27 は該当 issue なし（ホスト実行のみ） |
| 事前学習済みモデル | ある | ある | **公開（5 件）:** 0.12.0 の 4 件（`ModelRegistry`〈`crates/facade/src/model.rs:684`〉・`load`〈`crates/facade/src/model.rs:870`〉、`save`〈`crates/facade/src/model.rs:1017`〉、`download`〈`crates/facade/src/model.rs:1234`〉・`download_with`〈`crates/facade/src/model.rs:1287`〉、参照モデル `models`〈`crates/facade/src/lib.rs:248`〉）；# 28 HF Hub 連携の補助クレート `hf-hub`（`HfHub`〈`crates/hf-hub/src/lib.rs:342`〉・`resolve_revision`〈`crates/hf-hub/src/lib.rs:385`〉・`list_files`〈`crates/hf-hub/src/lib.rs:430`〉。`download` は `pub(crate)`〈`crates/hf-hub/src/lib.rs:452`〉のまま。`publish = true`、`crates/hf-hub/Cargo.toml:36`。0.14.0 で crates.io へ初回公開済み）。数え方は §2 の 1（#3771 の案 A）による<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 28）は §2 の読み方で公開に数える<br>**対象外（1 件）:** 重みの同梱（#3021 の `#issuecomment-6111746880` の項目 5。0.12.0 §3 から不変）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** 0.12.0 §5 の HTTPS 実通信の確認が実測済み。# 28 は該当 issue なし（ホスト上の HTTP 取得のみ）<br>**0.14.0 からの変更:** `hf-hub` の README の 1 行（0.14.0 で初回公開済みの記述）と版数の行だけ。公開面は不変 |
| 推論・サービング | ある | ある | **公開（9 件）:** 0.12.0 の 8 件（`predict`〈`crates/facade/src/compat/sequential.rs:3152`〉・`predict_batches`〈`crates/facade/src/compat/sequential.rs:3380`〉・`predict_resident`〈`crates/facade/src/compat/sequential.rs:4435`〉・`predict_resident_input`〈`crates/facade/src/compat/sequential.rs:4484`〉、`generate`／`AutoregressiveModel`／`GenerateConfig`／`SamplingStrategy`〈`crates/facade/src/inference/mod.rs:463`〉、`new_paged_kv_cache`〈`crates/facade/src/lib.rs:2622`〉・`paged_attention_forward`〈`crates/facade/src/lib.rs:2642`〉、`BatchScheduler`〈`crates/facade/src/inference/mod.rs:472`〉、`generate_speculative`〈`crates/facade/src/inference/mod.rs:476`〉、`StringLookup`〈`crates/facade/src/text/mod.rs:151`〉、`set_cuda_graph_infer_enabled`〈`crates/facade/src/lib.rs:3767`〉・`set_cuda_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3799`〉・`set_metal_elementwise_fusion_enabled`〈`crates/facade/src/lib.rs:3813`〉）；# 29 内部状態を持つモデルの speculative／連続バッチング（`StatefulAutoregressiveModel`・`generate_speculative_stateful` の再エクスポート〈`crates/facade/src/inference/mod.rs:485`〉、trait〈`crates/autodiff/src/generate/stateful.rs:50`〉、`generate_speculative_stateful`〈`crates/autodiff/src/generate/stateful.rs:150`〉、`step_stateful`〈`crates/autodiff/src/generate/scheduler.rs:687`〉。正ガードは `inference_module_reexports_exactly_expected_surface`〈`crates/facade/tests/api_surface.rs:25893`〉・`speculative_scheduler_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43252`〉・`stateful_public_shape_is_pinned`〈`crates/facade/tests/api_surface.rs:43326`〉）<br>**未公開（0 件）:** なし（0 件）。0.12.0 の 1 件（# 29）は公開へ移った<br>**対象外（5 件）:** 0.12.0 §3 から不変（汎用グラフコンパイル区分 A、HTTP／API サーバと量子化 KV、サブワードトークナイザ、beam search 等の生成機能、int8 量子化）<br>**保留（0 件）:** なし（0 件）<br>**実機 parity:** # 29 は新しいカーネル・バックエンド経路が無く、実機 parity の対象外（追跡表の欄は「該当 issue なし」）。speculative とスケジューラ・paged attention は 0.12.0 §5 のとおり実測済み。`speculative-sampling-3102` は 2026-10-27 に GB10・Mac mini とも pass（§5.4）<br>**0.14.0 からの変更（§3.1 の 1・2）:** CUDA 推論 Graph（`set_cuda_graph_infer_enabled`・`cuda_graph_infer_enabled`）が**既定 ON** になった（#4167。`false` で無効化できる。API の形は不変）。`predict`／`predict_batches`／`predict_resident`／`predict_resident_input` が推論スコープに入る（#4161。jpw1 は aarch64 Linux だけ既定 ON）。公開 API の追加・削除はない |

### 3.1 v0.14.0 → v0.15.0 の変更と各行への影響

出典は移行ノート `docs/migration/0.15.0.md` と `git log v0.14.0..v0.15.0`。**いずれも facade の公開面を変えず、9 行の判定を動かさない**。「実機 parity」の欄は、その変更を v0.15.0 で実機測定したかどうか（本書は測定していない）。

| # | 変更（PR） | クレート | 影響する行 | 判定への影響 |
|---|---|---|---|---|
| 1 | CUDA 推論 Graph を既定 ON に（#4167・PR #4290。`INFER_GRAPH_DEFAULT_ENABLED = true`〈`crates/backend-cuda/src/graph.rs:673`〉。`set_cuda_graph_infer_enabled(false)` か環境変数で無効化。学習を含む全 CUDA 利用者の既定ストリームが created stream になる） | backend-cuda・facade の rustdoc | 推論・サービング、バックエンド・ハード | なし。API の追加・削除なし。出力は bit 同一（移行ノート §1.1 の記述〈#4290・#4291 の実機確認〉。本書は再確認していない） |
| 2 | SME を既定 ON に（#4165・PR #4284。aarch64 限定）、推論スコープの jpw1（#4161・PR #4270。aarch64 Linux だけ既定 ON）。`predict` 系 3 か所に `enter_inference_scope()` を足した | backend-cpu・facade（`compat/sequential.rs`） | 推論・サービング、バックエンド・ハード | なし。公開 API の追加・削除なし。`enter_inference_scope` は内部クレートの `pub`（facade は再エクスポートしない） |
| 3 | Metal の steel 候補 T0U-LU-FB を M4Base の正方 1024／2048／4096 へ結線（#4166・PR #4255。`STEEL_LARGE_N_ENABLED = true`〈`crates/backend-metal/src/tile.rs:1802`〉）、構築失敗時は表の経路で組み直す（#4267） | backend-metal | バックエンド・ハード | なし |
| 4 | `create_graph` の総括判定の変更: CE の二階微分の修正（#4068）、WeightNorm の拒否範囲の縮小（#4070）、`Lin` の上界への置き換え（#4101 ツリー。#4102〜#4137） | autodiff（16 files・+14167） | 自動微分 | 対象 Op 集合・facade の公開面・正ガードは不変で、件数・判定は動かない。判定の実装が変わったため、C 区分・拒否範囲の実機 parity は 0.14.0 版の測定コミットのまま（§5.3） |
| 5 | `logcumsumexp` の 1 階 VJP を大きな共通オフセットで安定化（#4069） | tensor-core | 演算の範囲、自動微分 | なし |
| 6 | `backend-cpu` の `kernel_unchecked*`（SME・AVX2・AVX-512）を `pub(crate)` に（#4285。内部クレートの破壊的変更）。thin SME の `b_stride` を `checked_mul` に（#4286） | backend-cpu | なし（内部クレート） | なし。facade は `backend-cpu` の `kernel_unchecked*` を使っていない |
| 7 | eager release の出力サイズゲート（#4195）と、各種 A/B・診断の道具立て（#4189〜#4201 ほか） | autodiff・backend-* | なし | なし。サイズゲートは REJECT で結線していない（`docs/perf/cpu-adopt-wiring-4165.md` §2。`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED = false`）。既定 OFF のまま |
| 8 | 公開 9 クレートを 0.15.0 にバンプ（#4301） | 全公開クレート | なし | なし。`hf-hub`・`backend-rocm` の差は版数の行・コメント・README の 1 行だけ |

## 4. 0.14.0 判定との差分

### 4.1 件数の集計と行判定

「ある」は未公開 0 件かつ保留 0 件の行だけで、機械的に決めた（§2）。公開・対象外の件数は 0.14.0 版 §4.1 と同じで、v0.14.0 → v0.15.0 で追跡表の項目が動かず facade の公開面も変わらないため、**どの件数も動かない**。
未公開・保留は追跡表の集計と §4.2 のとおり照合した。

| 行 | 0.14.0 判定 | 公開 | 未公開 | 対象外 | 保留 | 0.15.0 再監査判定 |
|---|---|---|---|---|---|---|
| dtype | ある | 6 | 0 | 4 | 0 | ある |
| 自動微分 | ある | 16 | 0 | 4 | 0 | ある |
| 演算の範囲 | ある | 8 | 0 | 4 | 0 | ある |
| NN 層・モデル構築 | ある | 15 | 0 | 1 | 0 | ある |
| 最適化・学習ループ | ある | 14 | 0 | 2 | 0 | ある |
| バックエンド・ハード | 部分的 | 3 | 0 | 3 | 1 | 部分的 |
| 相互運用 | ある | 7 | 0 | 2 | 0 | ある |
| 事前学習済みモデル | ある | 5 | 0 | 1 | 0 | ある |
| 推論・サービング | ある | 9 | 0 | 5 | 0 | ある |

- 「ある」は 8 行、「部分的」は 1 行（バックエンド・ハード）。残る保留 1 件は追跡表の「未確定」（# 24）である。**0.14.0 から区分が変わった行はない**。
- 0.14.0 のスコアボード（0.12.0 版の監査）では 9 行すべて「部分的」だった。0.14.0 → 0.15.0 では 9 行とも同じ判定である。

### 4.2 追跡表の集計との照合

`python3 -I scripts/tally-compat-tracker.py`（main `a2e1e9ade` の tree〈v0.14.0 から追跡表・スクリプトとも無変更〉で実行。終了コード 0。`--self-test` も PASS）の出力をそのまま載せる。0.14.0 版 §4.2 の最終版の出力と**同一**である。

```
0.12.0 の照合: OK（§1 の集計 == §2 == §8.5 固定値）
- 行別合計 30 件（未公開 16・保留 14）、重複を除いた項目数 29

0.14.0 の区分（行別。複数行の項目は各行に数える）

| 行 | 未確定 | 未公開 | 保留 | 残件 | 公開 | 対象外 |
|---|---|---|---|---|---|---|
| dtype | 0 | 0 | 0 | 0 | 1 | 1 |
| 自動微分 | 0 | 0 | 0 | 0 | 4 | 0 |
| 演算の範囲 | 0 | 0 | 0 | 0 | 1 | 0 |
| NN 層・モデル構築 | 0 | 0 | 0 | 0 | 9 | 0 |
| 最適化・学習ループ | 0 | 0 | 0 | 0 | 7 | 1 |
| バックエンド・ハード | 1 | 0 | 0 | 1 | 0 | 0 |
| 相互運用 | 0 | 0 | 0 | 0 | 3 | 0 |
| 事前学習済みモデル | 0 | 0 | 0 | 0 | 1 | 0 |
| 推論・サービング | 0 | 0 | 0 | 0 | 1 | 0 |
| 合計 | 1 | 0 | 0 | 1 | 27 | 2 |

重複を除いた項目数: 残件 1（未確定 1・未公開 0・保留 0）、公開 26、対象外 2

注: 出すのは「追跡している残件が 0 か」まで。行判定（「ある」）は公開・対象外の基準件数（§8.5）と合わせて監査側が判断する。
```

（スクリプトの区分見出しは「0.14.0 の区分」のままである。スクリプトと追跡表は v0.14.0 から変えていないため、見出しの版表記も動かしていない。）

照合表（行別。「追跡表」は上の出力の未確定＋未公開＋保留、「監査」は §3 の未公開＋保留）:

| 行 | 追跡表の残件 | 監査の未公開＋保留 | 一致 | 差の理由 |
|---|---|---|---|---|
| dtype | 0 | 0 | 一致 | — |
| 自動微分 | 0 | 0 | 一致 | — |
| 演算の範囲 | 0 | 0 | 一致 | — |
| NN 層・モデル構築 | 0 | 0 | 一致 | — |
| 最適化・学習ループ | 0 | 0 | 一致 | — |
| バックエンド・ハード | 1（# 24） | 1（保留 1） | 一致 | 「未確定」を 0.12.0 での区分（保留）で数えた（§2 の 2） |
| 相互運用 | 0 | 0 | 一致 | — |
| 事前学習済みモデル | 0 | 0 | 一致 | — |
| 推論・サービング | 0 | 0 | 一致 | — |

29 項目の外で数えた項目は 0 件である（0.14.0 版 §4.2 と同じ。`CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否は承認済みで保留に当たらず、件数に数えない）。
29 項目の外の走査: `git diff --name-only v0.14.0 v0.15.0 -- docs/`（`docs/perf/logs` 以外）で増減したファイル名を確認し、`create_graph` に関わる決定記録 2 本（`docs/autodiff-higher-order-grad-decision.md`〈§77〜§83 の追記。#4101 ツリー〉・`docs/autodiff-cumulative-ops-decision.md`）の追記行を「所有者／承認／未回答／保留／見送り／未決」で grep して読んだ。
それ以外の `docs/perf/` 配下の A/B・診断の記録と移行ノートは、ファイル名と見出しの確認にとどめ、全文は読んでいない。
追記には、所有者判断待ち・起票未了の後続候補が残っている: `dg32`（Triplet の `PowScalar` の VJP 係数の指数の食い違い）の是正（判定が変わるため所有者判断）、既存の `flag` 行の O(C²)／O(d²) の走査への上限の導入（受理 → 拒否になりうるため所有者判断）、
`Lin` の仮定（CPU の `f32::powf`／`exp` が 2 ulp 以内。設計値で、GPU・aarch64 Linux・macOS は未実測。いずれも §80〜§82）。いずれも `create_graph` の内部判定の後続候補で、facade の公開面でも追跡表の 29 項目でもないため、
0.14.0 版 §4.2 の `CosineSimilarity`・`PairwiseDistance` と同じく件数に数えず、§6.2 に確認事項として挙げた。`Lin::total` の精密化（#4132）は 2026-11-08 に契約維持で決定し閉じている（§82.15.6）。
§72.1 の F5 は 0.14.0 版と同じ扱いで、`docs/compat-unlisted-holds-inventory.md` と保留ガード 43 件も不変。

### 4.3 0.14.0 版 §4.3 の決着 9 件

0.14.0 版 §4.3 の対応表（0.12.0 版 §6 の所有者確認 9 件の決着）は v0.15.0 でも変わらない（追跡表が無変更のため）。

## 5. 実機 parity の付録表

### 5.1 範囲と判定規則

- 0.14.0 版 §5（付録表 172 件。基準 sha `1cbb4759`。同書 §5.1〜§5.8）を**そのまま引き継ぎ、再作成していない**。数える母集合も判定規則（「実測済み」はテスト単位または族単位の pass で、測定コミットの内容に限る）も変えていない。
- 引き継ぐ根拠: v0.14.0 から v0.15.0 までに `docs/perf/logs` へ新設されたディレクトリ 26 件のうち、API の実機 parity の測定は **0 件**である（`git diff --name-status v0.14.0 v0.15.0 -- docs/perf/logs` の追加ディレクトリを確認）。
  25 件は、ディレクトリ名で性能の A/B・診断・同席計測・道具と分類した（各 README の中身までは読んでいない。`cpu-gemm-reuse-release-sizegate-*`・`cpu-infer-jpw1-gb10-ab-*`・`cuda-infer-graph-default-ab-4163`・`diag-0140-*`・`diag-4299-cudatrn-stream`・`framework-compare-0.14.0-*remeasure`・`macmini-foreign-job-gate`・`metal-steel-*`・`noise-band-4142`・`prepost-0150-4168`・`sme-*`）。
  残る 1 件は `rocm-facade-3867`（#3867。**未実測**。§5.2）。既存ディレクトリの変更は `sme-default-on-3231` の 1 件だけ（性能の記録）。
- 本付録の「実機 parity」は、各行の根拠欄（§3）に 0.14.0 版から引き継いだ記述のまま載せている。実機は動かしていない。

### 5.2 #3159・#3867（実測していないものは未実測のまま）

| issue | `gh issue view --json state,updatedAt` の結果 | 確認した日 | README の記入欄 |
|---|---|---|---|
| #3159 | state: OPEN・updatedAt: 2026-11-08T11:27:48Z | 2026-11-15 | `docs/perf/logs/rocm-parity-3159/README.md:3` に「**状態: 未実測**（所有者のインスタンス起動待ち）」。0.14.0 版 §5.3 から変わっていない |
| #3867 | state: OPEN・updatedAt: 2026-11-08T11:27:49Z | 2026-11-15 | `docs/perf/logs/rocm-facade-3867/README.md:3` に「**状態: 未実測**。下の記入欄は実測後に埋める」。0.14.0 版では「ディレクトリが無い」だったが、その後に事前登録の `RULE.txt` と手順書を置いた（#4171・#4207。測定は行っていない） |

- どちらも OPEN のままで、ROCm（MI300X・gfx942）の実機 parity は未実測である。ROCm の facade 同梱は #3771 の条件を満たしていない（§6.1 と同じ）。
- ツリー #4140 の Phase 5（MI300X の使用）は、MI300X を使わない方針でツリーから外した（issue は open のまま。測定は所有者の判断待ち）。
- 本書は実機を使っておらず、Linux や M4 Max の開発機の代替値も入れていない。

### 5.3 測定コミットと v0.15.0 の `crates/` 差分

`git diff --shortstat <測定コミット> v0.15.0 -- crates` の結果と、`git merge-base --is-ancestor <測定コミット> v0.15.0` による main の履歴上かどうか。差分がある以上、v0.15.0 での pass は未確認である（0.14.0 版 §5.7 と同じ読み方）。
0.14.0 版の基準 sha との差分と合わせて、v0.14.0 → v0.15.0 の 78 files のうち parity の対象になりうるのは autodiff（`create_graph`）・tensor-core（累積）・backend-cpu（SME・jpw1）・backend-cuda（推論 Graph の既定）・backend-metal（steel）である（§3.1）。

| 測定コミット（0.14.0 版 §5.7・§5.8 から） | v0.15.0 までの `crates/` 差分 | 位置づけ |
|---|---|---|
| `b82b1e5d1`（#3856・#3858・#3864・#3865 の対象）・`2b00e494b`（#3866 の対象） | 各 81 files changed, 19480 insertions(+), 816 deletions(-) | main の履歴上 |
| `2f59b9638`（`*-train-template-*` の対象） | 80 files changed, 18889 insertions(+), 813 deletions(-) | main の履歴上 |
| `8966befde`・`df9e060a0`・`c9b5e1635`・`5c3e1472e`（#4052 の CUDA／Metal・#4053 の CUDA／Metal） | 各 78 files changed, 18699 insertions(+), 811 deletions(-) | main の履歴外（作業ブランチ上のコミット） |
| `358b77554`・`343d88ee7`・`901a2b1f6`・`045a1241a` | 各 361 files changed, 113526 insertions(+), 2381 deletions(-) | main の履歴上 |
| `578e820d2` | 360 files changed, 113399 insertions(+), 2380 deletions(-) | main の履歴上 |

- 上の差分の中身は確かめていない（件数のみ）。v0.14.0 版の `create_graph` 系の実機 parity（#3856〜#3859・#4052）は、**v0.15.0 の `create_graph` 総括判定の実装（§3.1 の 4）では再測定していない**。
  この変更には WeightNorm の拒否範囲の縮小（#4070）を含む。facade の公開面と対象 Op 集合は変えていないが、「実機 parity の pass」を v0.15.0 の判定へ延長することはしない。
- 0.14.0 版 §5.7 のその他の測定コミット（Phase 10 の実行コミットなど）の差分は、本書では取り直していない（0.14.0 版の表のまま）。
- v0.15.0 の既定の変化（CUDA 推論 Graph 既定 ON・SME 既定 ON・Metal steel）は、前後比較（`docs/perf/logs/prepost-0150-4168/`）で出力の一致（checksum・parity）を確かめた性能の記録があるが、API の実機 parity の測定ではない。本付録の件数には数えない。

### 5.4 食い違いと申し送り

- 0.14.0 版 §5.8 の食い違い（追跡表の「実機 parity」欄の先頭語が「未実測」のまま残っている項目〈# 7・# 9〜# 15・# 22〉、3786・3788 の #4053、# 4 の `Nuc`・`Spectral`）は、追跡表が無変更のため**そのまま**である。
- **#3159・#3867 の MI300X 計測**は未実測のまま、所有者の判断待ちである。

## 6. スコープ外・申し送り・所有者への報告

### 6.1 行判定を下げた 1 行（所有者への報告）

追跡表 §5 手順 1 のとおり、「未確定」が残る行は監査の前に所有者へ報告する対象である。自動運転のため issue へのコメントは投稿していない。**報告は main セッションが行う**。

| 項目 | 状態 | 解除条件 |
|---|---|---|
| # 24 ROCm の facade 公開（バックエンド・ハード行） | 実装は済み（`Device::Rocm`・facade 結線。`backend-rocm` は 0.14.0 で crates.io へ初回公開済み）。#3159・#3867 が OPEN で MI300X の測定が無い。#3771 の回答の条件を満たさない（0.14.0 の監査から変化なし） | #3159・#3867 の parity pass |

題名の「9 行すべて 0 件」は、バックエンド・ハード行の # 24 が残るため、0.14.0 に続き 0.15.0 でも未達のままである。MI300X の測定は所有者の判断待ちで、本ツリー（#4140）では行わない。

### 6.2 29 項目の外の候補（所有者への確認事項）

- `CosineSimilarity`・`PairwiseDistance` の `create_graph` 拒否（決定記録 `docs/autodiff-higher-order-grad-decision.md` §73）。承認済みで別設計待ち・恒久ではなく、起票は所有者の承認事項で未起票（0.14.0 版と同じ）。
- §72.1 の F5（`Selu` など 5 variant と `Atan2` の `create_graph` fail-closed。#2649・#2634）。記録のみで承認は主張しない（0.14.0 版と同じ）。
- `create_graph` の総括判定の実機 parity（v0.15.0 の `Lin` 化の後）。測定の起票は所有者の承認事項で、本書では起票していない（§5.3）。
- `create_graph` の内部判定の後続候補（§4.2 の `dg32` の是正・`flag` 行の走査量の上限・`Lin` の仮定の GPU／aarch64 Linux／macOS での実測）。所有者判断・起票は所有者承認後で、本書では起票していない。

### 6.3 本 PR の範囲

- 本 PR は docs のみで、issue へのコメント・起票は行っていない。ruleset・リポジトリ設定も変えていない。
- 追跡表は凍結済み（#3869）のため変更していない。0.14.0 版の本文も変更していない。

## 7. 変更していないもの

`crates/`、`docs/spec/`、tolerance 定数・parity baseline・ガードレール閾値、依存、CI、`docs/compat-api-scope.md`、`docs/compat-matrix-ari-tracker.md`（凍結済み）、`scripts/tally-compat-tracker.py`、
0.9.0／0.11.0／0.12.0／0.14.0 版の本文、`docs/perf/logs/*` の既存の収録物（読むだけで、回収ログと `SHA256SUMS` を変えていない）。
