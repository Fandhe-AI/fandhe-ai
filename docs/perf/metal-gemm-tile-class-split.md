# Metal GEMM タイルクラス分割（E6 試作。イシュー #1327）

## 0. 目的・スコープ

`gemm_simdgroup_tiled`（`crates/backend-metal/src/shaders/gemm.metal`）の
1 dispatch を、**内部タイル**（M×N に完全に収まり K が BK の倍数）用の
「threadgroup ステージングを経由しない device 直接 `simdgroup_load`
経路」と、**端タイル**（M/N/K のいずれかがブロック境界に満たない残り）
用の「現行の手動境界チェック付き staged 経路」の 2 クラスへ分割する
opt-in 機構を追加し、両クラス経路の出力が現行版と **bit 同一**である
ことを端あり形状込みで自己検証する（親イシュー #1323 の趣旨）。

**本 Issue のスコープ**: 機構の実装と bit 一致の自己検証のみ。性能実測・
`tile::select`／本番既定への結線判断は行わない（兄弟イシュー #1328 の
スコープ）。

## 1. 設計

### 1.1 MSL（`shaders/gemm.metal`）

- `TILE_CLASS`（function constant・index 15・uint。`COOP_LOAD_LAYOUT`
  〈index 14・#1298〉の直後）を追加。値: `0`＝Legacy（従来どおり
  `USE_TGP_STAGING` のみでロード方式を決める）・`1`＝Interior（内部
  タイル。direct-load ブロックを強制）・`2`＝Edge（端タイル。staged
  ブロックを強制）。`#ifdef GEMM_SPEC_ENABLED` 側にも
  `GEMM_SPEC_TILE_CLASS` を追加（`crate::spec_source` 経由。E2 のソース
  テキスト特殊化経路とも整合）。
  - **index 15 の再割当について**: `docs/perf/metal-gemm-coop-load-
    candidates.md` §1/§6 では index 15 を「XOR swizzle 軸（未実装）用に
    未割当」と記していたが、その軸は未実装のまま残されていたため、本
    Issue で `TILE_CLASS` に割り当てた。両ドキュメントに相互参照の追記
    を行った。XOR swizzle 軸を将来実装する場合は index 16 以降を使う。
- `TileClassRegion`（4 × uint32＝16 バイト。`row_off`/`col_off`/`rows`/
  `cols`）を新設し、`gemm_simdgroup_tiled` の引数へ
  `constant TileClassRegion& region [[buffer(5)]]` を追加。`TILE_CLASS
  == 0`（Legacy）でも常にバインドする（未バインドバッファ参照を作らない
  ため）。
- kernel 冒頭、スウィズル変換（`tid_y`/`tid_x`）の直後・`row0`/`col0`
  算出の前に領域ガード＋オフセット加算を挿入:
  ```
  if (TILE_CLASS != 0) {
      if (tid_y >= region.rows || tid_x >= region.cols) { return; }
      tid_y += region.row_off;
      tid_x += region.col_off;
  }
  ```
  `TILE_CLASS == 0` では function constant 畳み込みでこのブロック全体が
  消え、以降の添字計算は既存 NN 経路と完全に同一のまま（`region` が常に
  恒等領域〈`rows=tiles_m`・`cols=tiles_n`・オフセット 0〉のため、たとえ
  ブロックが残っていたとしても数値的に no-op）。
- staged/direct-load の選択条件を
  `const bool staging_active = (TILE_CLASS == 0) ? USE_TGP_STAGING :
  (TILE_CLASS == 2);` に変更し、`if (staging_active) { …staged… } else {
  …direct… }` とする。**2 ブロックの本体（境界チェックを含む）は 1 文字も
  複製・変更していない**。`tests/shader_source_evidence.rs` の既存 needle
  出現数固定テスト（`a_row < dims.m` ×3 等）は全て不変のまま pass する
  （§3 実行結果参照）。
- 境界検査は全経路で維持: `tiled_block_out_of_range` 早期 return・direct
  ブロックの `a_row < dims.m`/`b_col < dims.n`・エピローグ guard・staged
  ブロックの `group_in_bounds`／要素単位フォールバックはいずれも無変更
  （REQ-8・`.claude/rules/coding-rust.md`「カーネル実装の境界検査」）。
  `docs/backend-metal-aligned-load-decision.md`（#752→#808）で不採用と
  なった「per-load の境界チェック短絡」とは異なり、本機構は「タイル単位の
  クラス分類を dispatch 側へ出す」だけであり、per-load の分岐削減は一切
  行っていない。
- `gemm_simdgroup_tiled_f16` は `TILE_CLASS`/`region` のいずれも参照
  しない no-op 契約（`pipeline_for_tile_f16` は常に `TileClass::Legacy`
  ＝`0` を渡す。§3 証跡テスト参照）。

### 1.2 Rust（`crates/backend-metal/src/tile.rs`）

- `TileClassMode`（`Legacy`／`Split`。instance ゲート。`crate::gemm::
  MetalGemm::new_with_tile_class` が受け取る）・`TILE_CLASS_MODE`（本番
  既定 `Legacy`）。
- `TileClass`（`Legacy`／`Interior`／`Edge`。`as_u32()` で 0/1/2 を返す。
  `crate::pipeline::GemmGateConstants::tile_class`／
  `crate::spec_source::SpecializationParams::tile_class` へ渡す値）。
- `TileClassRegion`（純粋な計算用の型。MSL 側・FFI 境界用の同名型は
  `crate::gemm` 側で独立定義する契約）・`TileClassPlan`
  （`interior: Option<TileClassRegion>`・`edges: [Option<TileClassRegion>;
  2]`）。
- **純関数 `tile_class_plan(m, n, k, cfg) -> TileClassPlan`**: `k % bk !=
  0` または M/N いずれかの方向に整列済みブロックが 1 つも無い場合は
  `interior = None` かつ grid 全体を単一の Edge 領域として返す（現行の
  1 回 dispatch と同じ構成）。それ以外は `full_m = m/bm`・`full_n =
  n/bn` を境に「内部（左上 `full_m×full_n`）」「右ストリップ（列
  `>= full_n`）」「下ストリップ（行 `>= full_m`・列 `< full_n`）」の
  3 分割へ落とす。3 領域は互いに素かつ `tiles_m×tiles_n` を過不足なく
  被覆する（`tile::tests::tile_class_plan_covers_every_tile_exactly_
  once_for_all_candidates_and_shapes` が全 `CANDIDATES` × 10 形状で
  Linux 上で静的に固定）。

### 1.3 Rust（`crates/backend-metal/src/pipeline.rs`／`spec_source.rs`）

- `GemmGateConstants::tile_class: u32`（index 15 の `setConstantValue_
  type_atIndex` を既存 `unsafe` ブロック内に 1 行追加）。
- `SpecializationParams::tile_class: u32`（`#define GEMM_SPEC_TILE_CLASS`
  を出力。`new()` は既定 `0` を埋める）。

### 1.4 Rust（`crates/backend-metal/src/gemm.rs`）

- `MetalGemm::tile_class_mode: tile::TileClassMode` フィールド・
  `new_with_tile_class(ctx, mode)` コンストラクタ・`#[cfg(test)]
  tile_class_mode()` アクセサ。
- パイプラインキャッシュキーを `(TileConfig, TransposePattern,
  TileClass)` へ拡張（`tiled_cache`／`tiled_spec_cache` 双方）。
  `pipeline_for_tile` に `tile_class: TileClass` 引数を追加し、
  `gates.tile_class = tile_class.as_u32()` を渡す。`pipeline_for_tile_
  f16` は常に `tile_class: 0` を渡す（他ゲートと同じ no-op 扱い）。
- `encode_dispatch_tiled` に `region: TileClassRegion` 引数を追加し、
  grid を `region.cols`×`region.rows`（従来の `dims`/`cfg` から導出する
  全体タイル数ではなく）で張り、buffer(5) へ常に `setBytes` する。
- **`encode_tiled_by_class`**（新設の共通ヘルパ。`dispatch_tiled_
  prepared`／`dispatch_strided_tiled_prepared`／`dispatch_variant`
  〈`SimdgroupTiled` 分岐〉の 3 本番入口が共有）:
  1. `TileClassMode::Legacy` → 従来どおり `pipeline_for_tile` を
     `TileClass::Legacy` で 1 回呼び、恒等領域で 1 回だけ dispatch する
     （既存挙動と完全に同一）。
  2. `TileClassMode::Split` → Edge クラスのパイプラインを先に解決し
     （`resolved_cfg` を確定）、その `resolved_cfg` を起点に Interior
     クラスを要求する（`tile::fallback_chain` は渡した構成自身を先頭に
     持つため、両者が同じデバイス制約下にある限り同一候補で解決される
     契約）。**両者の解決構成が一致しない場合は fail-closed に
     `TileClassMode::Legacy` 単一 dispatch へフォールバックする**
     （`TILE_CLASS_SPLIT_FALLBACK_COUNT` で可観測。エラーにはしない）。
     一致した場合は `tile::tile_class_plan` が求めた領域ごとに、
     Interior（存在すれば・空でなければ）→ Edge（右ストリップ・下
     ストリップの順）で `encode_dispatch_tiled` を呼ぶ。領域は互いに素
     なので dispatch 順序は出力に影響しない。
- 空振り検出用スレッドローカルカウンタ（`#[cfg(test)]` 限定ではなく
  常時計上。`BIAS_ACT_FUSED_LAUNCH_COUNT` 等と同じ設計判断）:
  `TILE_CLASS_INTERIOR_DISPATCH_COUNT`／`TILE_CLASS_EDGE_DISPATCH_COUNT`
  ／`TILE_CLASS_SPLIT_FALLBACK_COUNT`。

## 2. bit 一致の論拠

`#536`/`#538`/`#745`/`#809`/`#1282`/`#1288`/`#1293`/`#1298` と同じ論法:

1. **演算オペランド列不変**: staged/direct-load いずれのブロックも
   テキストを 1 文字も変更していない。`acc[r][c_]` の K 方向累算
   オペランド列（値・kk 昇順）は `TILE_CLASS` の値に関わらず不変。
2. **領域の互いに素な完全被覆**: `tile_class_plan` が返す領域集合は
   `[0, tiles_m) × [0, tiles_n)` を過不足なく 1 回ずつ被覆する（§1.2・
   Linux 単体テストで固定）。よって Split モードでの複数回 dispatch は
   「同じ 1 回 dispatch を領域で分割しただけ」であり、各 C タイルは
   Interior/Edge のいずれか片方から**ちょうど 1 回**書き込まれる。
3. **領域ガードは threadgroup 全体で一様**: `tid_y >= region.rows ||
   tid_x >= region.cols` の分岐は SIMD 内分岐にならず（threadgroup 内の
   全スレッドが同一の `tgid` から同一の判定結果を得る）、REQ-8 の境界
   検査契約を壊さない。

## 3. 自己検証結果

### 3.1 Linux 実行可能な静的検証（本エージェント実行環境で実施）

- `cargo test -p fandhe-ai-backend-metal --lib`: 257 passed（tile_class
  関連の新規単体テスト 6 件を含む。被覆・互いに素・整列条件・
  `TileClassRegion` レイアウト・既定値の各テスト）。
- `cargo test -p fandhe-ai-backend-metal --test shader_source_evidence`:
  39 passed（`…all_sixteen_defines`／`…all_sixteen_function_constants`
  ／`gemm_simdgroup_tiled_source_gates_tile_class_behind_function_
  constant`／`…retains_region_guard_before_offset`／
  `gemm_simdgroup_tiled_f16_source_does_not_reference_tile_class` の
  新規証跡テストを含む）。
- `cargo clippy -p fandhe-ai-backend-metal --all-targets -- -D
  warnings`: 本クレート内の findings 0 件（`fandhe-ai-backend-cuda` の
  既存 dead_code は本 PR と無関係の pre-existing issue。`git stash` で
  再現確認済み）。
- `cargo fmt --all -- --check`: 差分なし。

### 3.2 実機（Apple M4 Max。本エージェント実行環境がそのまま実機）

`docs/perf/logs/metal-gemm-e6-tile-class-1327/` に実行ログ・env_info を
記録する（内部ホスト名は含めない）。

- **T1** `tile_class_split_bit_match_all_candidates`: 全 9 候補 ×
  N∈{512,1024,2048,4096} で base（Legacy）/head（Split）の
  `dispatch_tiled_prepared` 出力が bit 単位で一致（PASS）。staged 候補は
  Interior dispatch カウンタが実際に増加すること（空振りでないこと）も
  確認。Split→Legacy フォールバックは 0 件。
- **T2** `tile_class_split_bit_match_edge_shapes`: 端あり形状
  `{(1032,1048,1032), (1040,1056,1024), (1032,1024,1024),
  (1024,1032,1024), (8,8,8), (64,64,4096)}` × 全 9 候補で bit 一致
  （PASS）。
- **T3** `tile_class_split_bit_match_dispatch_auto`: 本番自動選択経路
  `dispatch_auto`（N=512〜4096）で bit 一致（PASS）。
- **T6** `tile_class_default_matches_production_constants`:
  `MetalGemm::new(...).tile_class_mode() == tile::TILE_CLASS_MODE ==
  Legacy` を確認（PASS）。
- **既存 `#[ignore]` 群の非後退確認**: `cargo test -p
  fandhe-ai-backend-metal --release --tests -- --ignored
  --test-threads=1`（lib 46 件 + 全 `tests/*.rs` 統合テスト）が全て
  0 failed で完走（frag_load・coop_load・unroll_acc・
  source_specialized・swizzle・fine_barrier の各 bit-match・
  `gemm_dynamic_tile_parity`／`gemm_transposed_parity`／
  `cpu_metal_parity` 等の parity 群を含む）。

**T4（転置 NT/TN 対応の bit 一致）・T5（`FragLoadConfig` との合成
bit 一致）は本試作では実装しなかった**（時間制約。§6「スコープ外」
参照）。`dispatch_strided_tiled_prepared` 自体は `encode_tiled_by_class`
を経由する結線を行っており、NT/TN/TT でも機構としては動作する見込みだが
（`pattern` を素通しするのみで `TileClass`/`region` の扱いに転置固有の
分岐はない）、実機での明示的な bit 一致検証は未実施のまま残す。

## 4. env_info

- 機種: Apple M4 Max（Mac16,6）
- OS: macOS 26.6.2（BuildVersion 25G83）
- rustc: 1.96.0
- 詳細は `docs/perf/logs/metal-gemm-e6-tile-class-1327/env_info.txt`

## 5. #1328 への引き継ぎ

- **本番結線の判断材料**: `MetalGemm::new_with_tile_class(ctx,
  TileClassMode::Split)` で A/B 計測できる（`new_with_coop_load` 等と
  同型の入口）。
- **既知の性能リスク**: `docs/perf/metal-gemm-n4096-kernel-gap.md`
  §10.4（E3 実測）では device 直接ロード系（legacy／hoisted-k1／
  hoisted-k2）が全 N で staged 比 1.6〜3.3 倍遅いという実測結果がある。
  E6 の Interior クラスはまさにこの device 経路（`USE_TGP_STAGING=
  false` の direct-load ブロック）を使うため、素朴には**内部タイルを
  staged から direct-load へ切り替えることで遅くなる可能性が高い**。
  E6 の狙いは「端タイルの境界チェックオーバーヘッドを内部タイルから
  除去する」ことだが、direct-load 自体のロード効率が staged より
  劣るという E3 の知見と綱引きになるため、#1328 では**両方の効果を
  分離して計測する**必要がある（例: Interior クラスに `FragLoadConfig
  { device_hoisted: true, ksteps: One }` を合成する候補を含める。
  `MetalGemm::new_with_gates` は `tile_class_mode` と `frag_load` を
  独立引数として受け取れるため、テスト専用コンストラクタを 1 つ追加
  すれば両軸の直交合成を計測できる）。
- **計測形状**: 候補 0/4/5/8（`CANDIDATES` の代表構成）× N=1024/2048/
  4096（`docs/perf/metal-gemm-n4096-kernel-gap.md` の既存計測形状と
  揃える）を推奨。

## 6. スコープ外（PR 本文へ記録。新規 Issue の起票はユーザー承認事項の
   ため行わない）

- 性能実測・採否・`tile::select` 組み込み判断（#1328）。
- f16 経路（`gemm_simdgroup_tiled_f16`）への 2 クラス展開。
- T4（転置 NT/TN 対応の bit 一致自己検証）・T5（`FragLoadConfig` との
  合成 bit 一致自己検証）の実機実行（時間制約。§3.2 参照）。
- 一般 stride／TT 以外の転置パターンでの性能面。
- XOR swizzle 軸（旧 index 15 予約）の index 16 以降への再割当と実装。

## 7. 実測結果と採否（#1328）

- **実測**: `docs/perf/metal-gemm-n4096-kernel-gap.md` §12 に、候補
  0/4/5/8 × N=1024/2048/4096（AC 形状）に加え、本番 `select_for_device`
  選択構成（`CANDIDATES[1]/[2]/[5]/[6]`）× N=512/1024/2048/4096 の 2 系列
  で M4 Max 実機 5 プロセス起動計測を記録した
- **§5 の懸念（E3 の direct-load 劣位知見との綱引き）は本番選択構成では
  的中**: N=2048/4096 で Split（Interior＝direct-load 縮退）が一貫して
  legacy（staged）より 2.3〜3.1 倍遅い（5/5 run 符号一貫）。N=1024 も
  おおむね後退方向（4/5 run）
- **一方、候補 0/4/8（N=1024/2048/4096 では本番非選択構成）では逆に
  Split が一貫して legacy より 0.43〜0.85 倍速い**という、§5 時点では
  予想していなかった候補依存の符号反転が観測された（候補 5 は本番選択
  構成〈N=512 での `[5]`〉と同様、いずれの N でも Split が一貫して遅い
  か符号不一致）。この差は `bk` では説明できず（本番選択構成 `[1]`/`[2]`
  も `bk=16` だが Split で一貫して後退するため）、候補 0/4/8 の legacy
  ベースライン自体が同一 N の本番選択構成の legacy より約 7〜8 倍遅い
  ことが特徴的で、Split はこの非効率を部分的に緩和したに過ぎないと
  考えられる。いずれにせよこの改善は本番が実際に選択する構成には現れず、
  外挿できないことを直接計測で確認済み（同 §12.2-B・§12.4）
- **採否: 組み込み不可（REJECT）。`tile::TILE_CLASS_MODE = Legacy` を
  維持する**（判断根拠は同 §12.4）
- **§5 の引き継ぎ事項の消化状況**: 「Interior クラスに `FragLoadConfig
  { device_hoisted: true, ksteps: One }` を合成する候補」（T5 相当）は
  REJECT が明確なため実施しなかった（時間予算をより優先度の高い本番
  選択構成の直接計測〈B 節〉へ割り当てた）。「候補 0/4/5/8 × N=1024/
  2048/4096」は §12.2-A としてそのまま実施した
- **結線しない場合の理由**: 上記のとおり本番選択構成で明確な後退が
  複数 N・複数 run にわたり一貫して確認されたため。`tile::CANDIDATES`
  自体・`select_for_device` の選択ロジックへの変更は行わない
