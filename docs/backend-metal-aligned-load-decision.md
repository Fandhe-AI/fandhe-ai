# Metal GEMM アラインメント特化ロード分岐（align_M/N/K function constant 方式）不採用・保留判断（#752）

イシュー #752「perf(backend-metal): アラインメント特化ロード分岐（align_M/N/K function constant 方式）」に対応する。
親 #737 配下のタスク。PR #764（`feat/752-metal-aligned-load`）での実装試行が
codex-review 指摘を受けて撤回された経緯を、`docs/backend-metal-async-copy-decision.md`（#546）・
`docs/backend-metal-morton-mapping-decision.md`（#544）と同型の決定記録として残す。

対象技法の定義: MLX steel classic（`mlx` の `matmul.cpp`）が採る、M/N/K のタイル整列可否（`M % BM == 0` 等）を
Metal function constant としてシェーダへ渡し、コンパイル時特殊化で「境界検査なしベクトルロード版」と
「per-element 検査版」を分岐する技法（技法参照のみ・コード転記はしない）。

## 判断サマリ

**`align_M`/`align_N`/`align_K` を Metal function constant としてホスト側の整列証明（`M % bm == 0` 等）から
導出し、その真偽でカーネル内の境界検査式を短絡させる（`ALIGN_* || guard` 型で恒真化しコンパイラに
デッドコード除去させる）構成は不採用とする。** 境界検査を維持したまま整列形状に限って有効化する
ロード最適化（ベクトル化ロード等、検査式そのものは消さない構成）は設計上あり得るが、Apple Silicon 実機での
A/B 計測なしに有効性を主張できないため、実装は保留する。イシュー #752 は完了条件（実機 A/B 計測）を
満たしていないため open のまま残す。

## 試行経緯

### c8274a7: 実装（撤回済み）

PR #764 の初期コミット `c8274a7`（`perf(backend-metal): gemm_simdgroup_tiled にアラインメント特化ロード分岐
（align_M/N/K function constant）を追加`）は以下を実装した。

- `crates/backend-metal/src/shaders/gemm.metal`: `ALIGN_M`/`ALIGN_N`/`ALIGN_K`（function constant index 8〜10）を
  宣言し、staged 協調ロード（A/B タイルの `group_in_bounds` 判定）・direct-load 経路（`a_row`/`b_col` ガード）の
  境界検査式を `ALIGN_* || <元の検査式>` の OR 合成へ書き換えた
- `crates/backend-metal/src/tile.rs`: `AlignFlags` 型・`for_dims` 導出関数（実効次元がタイルブロック幅へ厳密に
  整除することの証明。fail-closed）を追加
- `crates/backend-metal/src/pipeline.rs`: `make_pipeline_with_constants` に align 引数を追加し index 8〜10 へ設定
- `crates/backend-metal/src/gemm.rs`: `tiled_cache` のキーを `(TileConfig, AlignFlags)` へ拡張
- `docs/perf/metal-gemm-aligned-load-ab.md`: 実機セッションでの数値一致・性能 A/B 計測手順を新規整備
  （未計測。実機セッションでの消化を前提とした計測台帳）

コミットメッセージ上は「非整列形状では検査式が一字一句そのまま残るため REQ-8 と両立する
（『省略』ではなく『整列証明済みの場合の恒真化』）」という理由付けだったが、整列バリアント自体では
コンパイラが検査式をデッドコード除去しうる構成であり、後述のとおり codex-review でこの理由付け自体が
覆された。

### codex-review 指摘（PR #764）

- **P0**（`crates/backend-metal/src/shaders/gemm.metal`）: 「整列時にもカーネルの手動境界チェックを維持してください。
  `ALIGN_* || guard` により function constant が true のバリアントではコンパイラが境界検査を除去します。
  これは、証明に基づく恒真化という説明であっても、AGENTS.md および埋め込み基準の『性能・最適化を理由にした
  カーネル手動境界チェックの省略は禁止（REQ-8、P0）』に直接違反します」。該当箇所は staged A ロード
  623〜624 行付近・B ロード 649〜650 行付近・direct-load 793 行・808 行付近（PR コメント記載の行番号。
  当時の HEAD 基準）。
- **P1**（同ファイル）: 「追加された ALIGN_M/N/K は宣言されるだけでカーネルから一度も参照されていません。
  そのため整列形状でもロード処理は従来と同一で、PR が目的とするアラインメント特化は発生しません。
  一方、ホスト側は AlignFlags ごとに別の Metal pipeline をコンパイルしてキャッシュするため、同一コードの
  パイプラインを最大 8 通り構築する余分なコンパイル・メモリコストだけが加わります」。
- **P2**（`docs/perf/metal-gemm-aligned-load-ab.md`）: 「文書は境界検査を恒真化してデッドコード除去すると説明し、
  16 行目では `gemm_simdgroup_tiled_source_uses_align_flags_in_*` が検証すると記載していますが、現在の
  シェーダーは ALIGN_* を未参照と明記しており、追加されたテスト名も `always_evaluates_*` です。この手順の
  head 計測では説明された最適化を測定できず、記載されたテストも実行対象に存在しません」。

P0 と P1 は相互に矛盾する状態を指摘している点が重要である: P0 は「整列時に境界検査が消える」ことを問題視し、
P1 は「`ALIGN_*` が未参照でロードに何の効果もない」ことを問題視した。これは a92c8f2 の中間修正
（境界検査は常時評価に戻したが `ALIGN_*` 宣言のみ残した）状態が P0 を解消しても P1 を新たに固定化する
だけであり、「機能する安全なロード方式選択」が実装できていない限りこの機構自体を PR に残す理由がない
ことを意味する。

### a92c8f2 → fefbb32: 修正の試みから機構ごとの撤回へ

- `a92c8f2`（`fix(backend-metal): アラインメント整列時の境界検査を常時実行へ戻す`）は P0 対応として、
  staged ロード側 `group_in_bounds` 判定 2 箇所・direct-load 側ガード 2 箇所から `ALIGN_*` との OR 合成を除去し、
  境界検査式が整列バリアントでも常に評価される形へ戻した。`ALIGN_M/N/K` function constant 自体は宣言のみ
  残す（パイプラインバリアント識別・将来拡張点として維持する）判断だった。
- しかしこの状態は P1（`ALIGN_*` 未参照でロードに効果がなく、パイプラインバリアント数だけが増える）を
  解消しない。`fefbb32`（`fix(backend-metal): アラインメント特化ロード分岐を撤回する`）で機構全体
  （`ALIGN_M/N/K` function constant 宣言・`crate::tile::AlignFlags`・`pipeline_for_tile` の複合キャッシュキー
  拡張・`docs/perf/metal-gemm-aligned-load-ab.md`・関連テスト `gemm_dynamic_tile_parity.rs`・
  `shader_source_evidence.rs`）を撤回し、`crates/backend-metal/src/shaders/gemm.metal` の境界検査は
  撤回前の状態（`ALIGN_*` を介さない元の検査式のみ）へ戻った。

この結果、`main` から見た PR #764 の merge-base 差分はゼロになった
（`git diff origin/main...feat/752-metal-aligned-load --stat` が空。2026-08-19 確認）。

## 判断

1. **`ALIGN_* || guard` 型の境界検査短絡（コンパイル時特殊化による境界検査除去）は不採用とする。**
   `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」は「性能下限・最適化の達成を理由に、
   シェーダ・カーネル側の手動境界チェックを省略しない」と定めており、この禁止は整列であることの証明の
   正当性とは無関係に及ぶ（証明が正しく実行時に安全であっても、規約は「性能・最適化を理由にした省略」を
   形式として禁止している）。codex-review P0 の指摘とこの規約解釈は一致しており、再検討の余地はない。
2. **境界検査を維持したままの整列特化（ベクトル化ロード等、検査式自体は残しつつ整列形状でのみ
   別のロードコード経路を選ぶ構成）は、設計上は規約と両立しうる。** ただし今回の試行は「機能する
   安全なロード方式選択」の実装そのものに至らず（P1 の指摘どおり `ALIGN_*` が未参照のまま終わった）、
   かつ Metal 実機環境が本セッションで利用できないため、境界検査を維持した構成の有効性
   （TFLOPS 改善が実際に得られるか）を A/B 計測で示すことができない。効果が実証できない最適化を
   実装として残すことは `docs/perf/metal-gemm-aligned-load-ab.md` 撤去の判断（P2 指摘: 実装と乖離した
   計測文書を残さない）とも整合しないため、実装は保留とする。
3. 上記 1・2 により、PR #764 は実装の採否判断そのものが成果であり、コード差分はゼロ（merge-base 一致）
   のまま「不採用・保留の意思決定記録」として本ドキュメントを追加する形で完結させる。

## 再開条件

- Metal 実機（Apple Silicon）が利用可能なセッションで、境界検査を維持したロード方式選択
  （function constant によるコード経路分岐であって境界検査の短絡ではないもの）を実装したうえで、
  整列形状（512/1024/2048/4096 の正方等）と非整列形状の両方で `#[ignore]` 数値一致テストが全 pass すること
  （#752 受け入れ条件 1）
- 整列形状の TFLOPS が既存経路に対して非劣化かつ改善することを実機 5 回計測の中央値で確認し、
  `docs/perf/` 配下に実測記録として残すこと（#752 受け入れ条件 2）
- 実装時は `crates/backend-metal/src/shaders/gemm.metal` の境界検査式（本判断時点で `group_in_bounds` 判定
  623・649 行付近、`a_row`/`b_col` ガード 733・745 行付近）を一切短絡させないことをレビュー
  （codex-review・security-auditor 等）で機械的に確認できる形にすること
- 非公式 API（`docs/backend-metal-async-copy-decision.md` で不採用とした `simdgroup_async_copy` 系等）へは
  波及させないこと

## 出典

- イシュー #752（親 #737）
- PR #764（`feat/752-metal-aligned-load`）: コミット `c8274a7`（実装）・`a92c8f2`（境界検査復元）・
  `fefbb32`（機構撤回）・codex-review レビュースレッド（P0/P1/P2、いずれも resolved）
- `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」
- `docs/backend-metal-async-copy-decision.md`・`docs/backend-metal-morton-mapping-decision.md`
  （同型の決定記録。フォーマットの踏襲元）

## 追補（#808）: 保留の再確認と不採用への格下げ

イシュー #808「アライン特化ロード分岐の再評価（#752 保留の再確認）」（親 #790・phase:5 補完改修）への
対応として、上記「再開条件」節の保留状態を再確認した。本追補は既存の判断（1〜3 節）・再開条件の
記述を書き換えるものではなく、その上に積む形で結論を追記する。今後この論点を参照する場合は
本追補節を優先する。

### (a) #752 保留理由 2 点の再確認

再開条件は 2 系統の理由から成っていた: (1) 検査短絡型（`ALIGN_* || guard`）は REQ-8 と正面衝突するため
不採用が確定済み、(2) 検査維持型（整列形状でのみ別ロード経路を選ぶが境界検査式自体は残す構成）は
規約と両立しうるが実機 A/B 計測ができず保留、というものだった。今回再確認した結果:

- (1) は `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」が「性能下限・最適化の達成を
  理由に、シェーダ・カーネル側の手動境界チェックを省略しない」と形式で禁止しており、証明の正当性とは
  無関係に及ぶ解釈に変更はない。再検討の余地はないという既存結論を維持する。
- (2) は下記 (b) のとおり主線コードが #752 判断時点から変化しており、保留の前提（「効果が実証できて
  いないだけで、実装すれば得られる利得がある」）自体が縮小している。

### (b) #752 判断以降の主線コードの変化

`crates/backend-metal/src/shaders/gemm.metal`（本追補作成時点で f32 GEMM カーネル部分は 623・649・
733・745 行付近に位置。ファイル全体はその後の #819 f16 動的タイル選択統合で 1094 行まで増加している
が、当該行番号は f32 経路の先頭付近のため変化していない）は #752 の判断（2026-08-19 確認、
PR #764 merge-base 差分ゼロ）以降、以下の変更を経ている（コミットタイトルは `git log` 実測）:

- `f97089e`（#660）: staged 協調ロードを `float4` ベクトル読み出しへ変更。境界検査（`group_in_bounds`
  判定、A 側 623 行付近・B 側 649 行付近、HEAD 基準）はベクトルグループ単位で維持したまま、ロード自体を
  ベクトル化した。align 特化技法が本来狙う「ベクトルロード化による帯域改善」の主要部分はこの時点で
  主線に取り込まれている。
- `d25a715`（#673）: threadgroup memory のパディングを導入（bank conflict 回避）。
- `d5aa663`（#761）: レジスタ常駐アキュムレータ構造・出力タイル拡大（MLX `mma.h` の tile_matmad 型構造。
  フラグメントのレジスタ常駐化）。

これらにより、align 特化ロード分岐が仮に実装された場合の残余価値は「毎ベクトルロードで評価される
`group_in_bounds`（623・649 行付近）および `a_row`/`b_col` ガード（733・745 行付近、いずれも HEAD 基準）
という一様分岐（threadgroup 内で全スレッドが同一方向に倒れる分岐）の評価コストそのものの削減」に
絞り込まれている。この残余価値を得るための技法は、境界検査式自体をコンパイル時に消去する（整列
バリアントでは検査を評価しない）構成にしない限り実効を持たない。検査式を評価だけして分岐を保つ
構成（`if (a_row < dims.m) { ... }` を整列時も素通しで評価する形）は現行実装と実質同一であり、
新たな最適化ではない。

### (c) MLX `LoopAlignment` との対比

対比対象は ml-explore/mlx リポジトリの `mlx/backend/metal/kernels/steel/gemm/gemm.h` の
`LoopAlignment`／`M_aligned`・`N_aligned`・`K_aligned` によるテンプレート特殊化（技法参照のみ・
コード転記なし。`docs/backend-metal-mlx-classic-nax-decision.md`〈#549〉と同一方針だが、#549 が
参照する `steel_gemm_fused.metal`・`matmul.cpp` は `gemm.h` を対象としていないため、本追補用に
参照コミットを独自に解決した: `gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決した
`27fec909a3df9e572f5195607a453e273e7d80d0` 時点の `gemm.h`）。20 行目で `LoopAlignment<M_aligned,
N_aligned, K_aligned>` テンプレートを宣言し、100〜110 行目（`gemm_loop` 内）で `M_aligned`／
`N_aligned` が true の分岐では境界検査を伴わないロード関数（`load_unsafe()`）を、false の分岐では
タイル実寸を渡す境界検査付きロード関数（`load_safe(tile_dims)`）を呼び分けている（実測確認済み）。
つまり整列側の分岐では境界検査を伴うロード経路そのものが呼ばれず、コンパイル時特殊化によって
生成コードから境界検査が排除される。この技法の利得は「境界検査を実行時に評価しない」ことから
生まれており、本リポジトリの REQ-8（性能・最適化を理由に手動境界チェックを省略しない）とは前提が
異なる。REQ-8 を維持したまま MLX と同じ利得を得ることはできない。

### (d) 判断: 再評価は行わない（保留 → 不採用へ格下げ）

以上 (a)〜(c) により、**再評価（実装再試行・実機 A/B 計測）は行わない**。理由:

1. 検査短絡型は #752 で不採用確定済みであり、本追補でもその解釈に変更はない（規約緩和はユーザー
   承認事項であり、自動運転で採用側へ倒すことはできない）。
2. 検査維持型で規約と両立する構成は、#660 のベクトルロード化により差分価値が「一様分岐の評価コスト
   削減」のみへ縮退しており、有意な TFLOPS 改善を期待する根拠が乏しい。
3. 本追補作成セッションは Linux 環境であり、Metal 実機（Apple Silicon）は
   `docs/real-hardware-verification-env.md` の記載どおりローカル直接実行前提のため、本環境から
   A/B 実測は実行不可能。#752 受け入れ条件 2（実機 5 回計測中央値での確認）はもとより「再評価する
   場合」の条件であり、再評価しないという本判断により対象外となる。

よって #752 の保留（「効果が実証できていないだけ」というニュアンス）は、**不採用へ格下げ**する
（`docs/cpu-gemm-prefetch-decision.md` の #489 保留 → #751 格下げと同型の運用）。イシュー #752 自体の
open/closed の扱いは本追補の対象外とする（本文冒頭の「イシュー #752 は完了条件を満たしていないため
open のまま残す」という既存の記述は変更しない）。

### (e) 最小限の再訪条件

将来、Metal 実機セッションで「検査維持型かつ有意な TFLOPS 改善」の反証実測が得られた場合にのみ、
以下を満たす形で再訪しうる（既存「再開条件」節を上書きするのではなく、本項をその具体化として扱う）:

- 境界検査式（`group_in_bounds` 判定・`a_row`/`b_col` ガード。本追補時点で 623・649・733・745 行付近、
  HEAD 基準）を一切短絡させないこと。
- 整列形状・非整列形状の両方で数値一致テスト（`#[ignore]` 実機依存）が全 pass すること。
- 整列形状の TFLOPS が既存経路に対し非劣化かつ改善することを実機 5 回計測の中央値で確認し、
  `docs/perf/` 配下に実測記録として残すこと。

### (f) 出典

- イシュー #808（親 #790・phase:5 補完改修）
- イシュー #752（親 #737）・本ドキュメント上記節
- コミット `f97089e`（#660）・`d25a715`（#673）・`d5aa663`（#761）
- ml-explore/mlx `mlx/backend/metal/kernels/steel/gemm/gemm.h`（参照時点コミット
  `27fec909a3df9e572f5195607a453e273e7d80d0`。`gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で
  解決。20 行目 `LoopAlignment` 宣言・100〜110 行目 `gemm_loop` の `load_unsafe`／`load_safe` 分岐）
- `docs/backend-metal-mlx-classic-nax-decision.md`（#549。MLX 参照時の書式踏襲元）
- `docs/cpu-gemm-prefetch-decision.md`（#489 保留 → #751 格下げ。同型運用の踏襲元）
- `docs/real-hardware-verification-env.md`（Metal 実機環境の制約）
- `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」
