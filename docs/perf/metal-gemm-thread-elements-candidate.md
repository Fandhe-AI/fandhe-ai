# Metal GEMM: thread_elements() 方式 BlockMMA 候補カーネル

イシュー #1693（親 #1586）。candle・MLX steel（`mlx/backend/metal/kernels/
steel/gemm/mma.h::BaseMMAFrag<T,8,8>`）が採用する「`simdgroup_float8x8::
thread_elements()`（MSL 標準 API）でレーンごとにフラグメント要素を直接
読み書きする」BlockMMA 方式を、本番 `gemm_simdgroup_tiled`
（`simdgroup_load`/`simdgroup_store` がレーン→要素対応を隠蔽する。
`docs/backend-metal-morton-mapping-decision.md`）に対する opt-in 候補
として追加する。**本番未結線・既定不変**。性能実測・本番結線可否判断は
兄弟イシュー #1694 のスコープ。

## §0 目的・範囲

- 目的: `thread_elements()` 方式が本番 `simdgroup_load`/`store` 方式と
  比べてどう振る舞うか（正しさ・bit 一致・性能）を実機で検証できる
  opt-in 経路を用意する。
- 範囲: staged（協調ロード）経路のみ。direct-load・split-K・タイル
  クラス分割・フラグメントロード方式候補・協調ロードレイアウト候補・
  条件付き loop unroll・simdgroup 細粒度同期・ソーステキスト特殊化との
  併用はいずれも対象外（`shaders/gemm.metal::gemm_simdgroup_tiled_te`
  冒頭コメント「スコープ境界」参照）。

## §1 MLX/candle 方式の要約

一次情報: `ml-explore/mlx` リポジトリ `mlx/backend/metal/kernels/steel/
gemm/mma.h::BaseMMAFrag<T,8,8>`。

- `get_coord(lane)`: `qid = lane/4; fm = (qid & 4) + ((lane/2) % 4);
  fn = (qid & 2)*2 + (lane % 2)*2;` — 各レーンは 8×8 フラグメントの
  同一行の連続 2 要素 `(fm, fn)`・`(fm, fn+1)` を担当する。
- `load`/`load_safe`/`store`/`store_safe`: レーンごとに
  `src[fm*str_x + (fn+j)*str_y]` を要素単位で読み書きする（境界チェック
  版は要素ごとに `< lim` 判定）。
- `mma`: `A_mat.thread_elements()` へ代入してから
  `simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat)` を呼ぶ
  （`thread_elements()` は MSL 標準 API として使われている）。

## §2 設計

- **カーネル**: `gemm_simdgroup_tiled_te`（`shaders/gemm.metal` 末尾）。
  本番 `gemm_simdgroup_tiled` とシグネチャを完全一致させる（buffer
  0〜6・`threadgroup float*`・`uint3 tgid`・`simd_lane`・`simd_id`）。
  プロローグ・協調ロード（float4 ベクトルロード・9 境界ヘルパによる
  要素単位 0 埋めフォールバック。REQ-8 の手動境界チェックを一切変更
  しない）は本番 staged 経路の逐語コピー。kk ループのみ `simdgroup_load`
  ではなく `thread_elements()`（インデックス代入）でフラグメントを構築
  し、エピローグも `thread_elements()` で読み出して要素単位の境界チェック
  （`row < dims.m && col < dims.n`）付きで `c` へストアする。
- **レーン座標**: `crate::tile::thread_elements_coord`（`#[cfg(test)]`
  限定の Rust 側検証モデル。MLX の式と同一）。全単射であることを単体
  テストで固定し、実機 probe（後述）でレーン→要素対応が実機仕様と
  一致するかを検証する。
- **到達経路（instance ゲート）**: `crate::tile::MmaFragLoad`
  （`SimdgroupLoad`〈既定〉/`ThreadElements`）＋
  `MetalGemm::new_with_mma_frag_load`。**`MetalGemm::pipeline_for_tile`
  自身が `self.mma_frag_load` を見てカーネル関数名を切り替える**
  （`gemm_simdgroup_tiled` ↔ `gemm_simdgroup_tiled_te`）ため、
  `ThreadElements` インスタンスでは `dispatch_auto`／
  `dispatch_tiled_prepared`／`dispatch_strided_tiled_prepared`／
  `dispatch_variant` 等の既存本番入口がそのまま候補カーネルへ到達する
  （hfrag 候補〈#1369〉のような専用 `dispatch_*_unverified` 関数は持たない
  設計）。`pipeline_for_tile` の te ガードは非 staged 候補
  （`SINGLE_SIMDGROUP_8X8` を含む）・`TileClass != Legacy` を fail-closed
  で拒否する（`tiled_cache`／`tiled_spec_cache` はそのまま共有——
  `MetalGemm` インスタンス自体が base/head で分かれるため取り違えは
  起こらない）。
- **数値契約**: 正式ゲートは REQ-2 統一複合判定（parity self-test）。
  演算オペランド列（r/c_ 昇順・kk 昇順の `simdgroup_multiply_accumulate`
  発行順）・共有メモリへ格納する値は本番 staged 経路と完全に同一のため、
  レーン→要素レイアウトが実機で `thread_elements_coord` モデルと一致
  すれば bit 同一が期待できる（`docs/backend-metal-morton-mapping-
  decision.md` が「レーン対応の *制御* は不可」とした判断と、本候補
  〈固定レイアウトを標準 `thread_elements()` API で *読む*〉は矛盾しない
  ——制御不可なのはレーン割当の変更であり、既存割当を読むこと自体は
  MSL 標準機能）。レーン→要素対応自体は MSL 仕様上
  implementation-defined のため、実機 probe（`simdgroup_thread_elements_
  layout_probe`）で確認する。

## §3 Linux 自己検証結果

- `tile::tests`: `mma_frag_load_default_is_simdgroup_load`・
  `thread_elements_coord_is_bijection_over_8x8`・
  `thread_elements_coord_matches_mlx_known_values`（3 件。全 green）。
- `tests/shader_source_evidence.rs`:
  `gemm_simdgroup_tiled_te_source_uses_thread_elements_and_matrix_unit_
  instructions`・`gemm_simdgroup_tiled_te_source_does_not_reference_out_
  of_scope_gates`・`gemm_metal_source_declares_thread_elements_layout_
  probe_kernel`・`gemm_splitk_reduce_source_still_uses_no_atomics_after_
  te_addition`（4 件追加。既存 50 件と合わせ計 54 件全 green）。
- `cargo check -p fandhe-ai-backend-metal --tests --target
  aarch64-apple-darwin`（`make check-cross-metal-tests` 相当）: エラー 0・
  dead_code 警告 0（`mma_frag_load` フィールドは `pipeline_for_tile`
  自身が参照するため macOS 非テストビルドでも dead code にならない。
  `thread_elements_coord` は純粋な検証用モデルのため `#[cfg(test)]`
  限定へ絞った）。
- `cargo clippy --workspace --all-targets --all-features -- -D
  warnings`（native・CI と同じ ubuntu-latest ターゲット）: 全 green
  （backend-metal の macOS 限定コードは Linux では cfg で除外されるため
  本チェックの対象外。macOS 側の clippy は実機実測時に別途確認する）。
- `cargo test -p fandhe-ai-backend-metal --all-features`（Linux）:
  全 green（新規 3 テストを含む 136 件の `tile::tests` 等）。
- `cargo test --workspace --all-features`（Linux）: 全 green。

**実機（Apple Silicon）での MSL コンパイル確認・parity・bit 一致・
probe の実行は未実施**（本エージェント実行環境に Apple Silicon 実機が
ないため）。

## §4 実機記入欄（未実測）

以下は Mac セッション（#1694）が記入する。

- R0（前提ゲート）: `crate::gemm::tests::te_layout_probe_matches_model`
  （probe が `thread_elements_coord` モデルと一致するか）
  結果: 未実測
- R1（parity）: `tests/gemm_te_parity.rs` の
  `te_square_shapes_all_patterns`・
  `te_tall_wide_and_k_tail_shapes_all_patterns`・`te_ragged_shapes_nn`・
  `crate::gemm::tests::all_staged_candidates_match_te_cpu_reference_
  512_nn`
  結果: 未実測
- R2（非 staged 拒否）: `te_rejects_non_staged_candidate`
  結果: 未実測
- R3（本番との bit 一致）: `te_bit_match_with_production_dispatch_auto`
  結果: 未実測

MSL 自体のコンパイル可否（構文エラーの有無）も上記実機実行で初めて
確認できる（Linux 上ではクロスコンパイル検証の対象外）。

## §5 #1694 への引き継ぎ

1. R0（probe）→ R1（parity/bit 一致）→ 性能 A/B の順に進める。
2. base 側は `MetalGemm::new(&ctx)`（`SimdgroupLoad`。既定）、head 側は
   `MetalGemm::new_with_mma_frag_load(&ctx, MmaFragLoad::ThreadElements)`
   で構築する（同一プロセス内 base/head 構成。他候補〈unroll_acc・
   frag_load 等〉と同型の A/B 運用）。
3. `diag_encode_tiled_nn`（f32 版。既存の計測境界専用入口）は
   `ThreadElements` インスタンスでもそのまま使える
   （`pipeline_for_tile` を経由するため）。
4. probe（R0）で不一致が出た場合、原因は緩和せず切り分ける（レーン
   対応の実機仕様差異そのものが本候補の viability を左右するため）。

## §6 スコープ外

- 本番結線（`tile::select`／`dispatch_auto` 既定化）。
- split-K・タイルクラス分割との併用。
- MLX 型 interleaved フラグメント配置（`w_x = WM` ストライド）。
- `simdgroup_barrier(mem_none)` 挿入。
- fused epilogue／非 `pad8` の C 直接書き込み。
- REQ-2 baseline 行の追加（人間承認必須）。
- MSL コンパイル自体の Linux 上での検証（Mac セッションへ申し送り）。
