# Metal simdgroup タイル内 Morton 順スレッド→要素マッピング適用不可の判断（#544）

イシュー #544「spike(backend-metal): Morton 順スレッド→タイル位置マッピングの適用余地を調査」に対応する。
親 #530（Phase D: Metal マルチ simdgroup 化・ロード最適化）・ルート #479（GEMM 性能改善）配下の D-8 調査タスク（spike）。
参照実装 metal-flash-attention（MFA。`philipturner/metal-flash-attention`）が 8x8 simdgroup タイル内のレーン→要素
対応を Morton（Z オーダー）順に自前配置してコアレッシングを得ている技法について、本実装（`backend-metal`）へ適用できる
余地があるかを調査した結果を記録する。`docs/backend-metal-async-copy-decision.md`（#546）と同型の決定記録として、
判断とその根拠を残す。

対象技法の定義: simdgroup（Apple GPU の 32 レーン協調実行単位）が 8x8 行列タイルを保持する際、どのレーンがタイル内の
どの要素を保持するかという対応（レーン→要素マッピング）を、行優先や列優先ではなく Morton 順（Z オーダー曲線に沿った
2 次元ビットインターリーブ順）に配置する技法。MFA はこの配置を明示制御するため、Apple 公開の MSL 仕様に存在しない
自作ストレージ型 `simdgroup_matrix_storage<T>`（`#pragma METAL internals : enable` 下で定義）を用いる。出典として
特許 US11256518B2（`https://patents.google.com/patent/US11256518B2`）が MFA ソース中に明記されている。

## 判断サマリ

**標準 `simdgroup_matrix` API（`simdgroup_float8x8` 等）を使用する現行 `backend-metal` の実装
（`crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_tiled`）では、レーン→タイル内要素の Morton 順
マッピングを制御する手段は存在しない。制御するには MFA と同様に非公式 API（自作 `simdgroup_matrix_storage<T>`、
`#pragma METAL internals : enable` 下）を採る必要があるが、これは #546 で不採用と決定済みの対象技法系統（草案
§0 A-5 系統。Apple 非公開 ABI への直接依存）に属するため、レーンレベルの Morton 順マッピングは不採用とする。**

一方、`gemm_simdgroup_tiled` の `tgid`（threadgroup 位置）から C ブロック原点への割り当て
（`row0 = tgid.y * BM` / `col0 = tgid.x * BN`、`gemm.metal:353-354`）は自前の線形算術であり、標準 API の外側にある
制御可能な層である。ここに Z オーダー変換を適用する余地はあるが、狙いがレーンレベルのコアレッシングではなく
threadgroup 単位のキャッシュ局所性であり、これは既存の兄弟イシュー #540「threadgroup ID スウィズル（swizzle_log
相当）」（OPEN）のスコープと同一目的であるため、適用案の記載に留め実装は #540 に委ねる。

## 根拠

### 1. MSL 仕様側: 標準 `simdgroup_matrix` は opaque 型でレーン対応を隠蔽する

Apple 公開の Metal Shading Language 仕様における `simdgroup_matrix<T, Rows, Cols>`（`simdgroup_float8x8` はその
インスタンス化）は不透明（opaque）な型として定義され、simdgroup 内のどのスレッド（レーン）がタイル内のどの要素を
保持するかは実装定義（implementation-defined）であり、アプリケーションコードから観測・制御する API を持たない。
`simdgroup_load`/`simdgroup_store` はメモリ⇔レジスタ間の行列単位ロード・ストアを提供するのみで、引数
（`elements_per_row`・`matrix_origin`・`transpose_matrix`）はいずれもメモリ側のアドレッシングを指定するものであり、
レーン→要素対応そのものを指定する手段ではない。この opacity が、標準 API 下で Morton 順マッピングを直接制御できない
根本理由である。

### 2. MFA 側: Morton 順配置は非公式ストレージ型の thread-owned 要素オフセット算出として実装される

MFA リポジトリ（参照時点コミット `8671cddc38f19a6eadb804dee6a3ca2954b8bf32`。`gh api
repos/philipturner/metal-flash-attention/commits/main` で解決。#546 の決定記録と同一コミット）の
`Sources/FlashAttention/GEMM/GEMMHeaders.swift`:

- `:536-552` のコメントで、8x8 タイル内の 32 レーンの配置図（`0 0 1 1 8 8 9 9` … の Z オーダー）を明示し
  「This is Morton order, a method for coalescing data accesses.」「Source:
  https://patents.google.com/patent/US11256518B2」と記す
- `:553-566` の `morton_order(ushort thread_index_in_simdgroup) -> ushort2` 関数が、レーン ID から象限
  （quadrant）・象限内位置を算術分解してタイル内座標 `(N_in_simd, M_in_simd)` を明示的に計算する
- `:570` の `#pragma METAL internals : enable` 直後（`:576` 付近）に定義される自作 `simdgroup_matrix_storage<T>`
  構造体（`:579` `thread_elements()` がレーンごとに保持する `vec<T, 2>` 要素へのポインタを返す）が、この Morton
  順オフセットをレーンが自分の担当要素へアクセスする際の基盤として使われる設計になっている
- `:634` の `#pragma METAL internals : disable` で非公式領域を閉じる

`gh api "search/code?q=morton_order+repo:philipturner/metal-flash-attention"` で確認した結果、`morton_order` は
`GEMMKernel+Source.swift`・`AttentionKernel+Source.swift` 等、生成される MSL シェーダのソース文字列組み立て側でも
参照されており、MFA 全体でレーン→要素対応を明示制御する経路がこの非公式ストレージ型に一貫して依存している。

**この技法は「非公式ストレージ型が thread ごとに保持する要素へどうオフセットでアクセスするか」というレイヤに位置し、
標準 `simdgroup_matrix`（opaque・レーン対応を露出しない）では原理的に再現できない。**

### 3. 本実装側: `simdgroup_load`/`simdgroup_store` 経由でレーン対応は API 任せ

`crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_tiled`（メインの simdgroup タイル化カーネル。
`:340-609`）は、8x8 タイルへの読み書きをすべて標準 API `simdgroup_load`/`simdgroup_multiply_accumulate`/
`simdgroup_store` 経由で行う（例: `:524`・`:542`・`:543`・`:606`）。レーン→要素対応はコンパイラ・ドライバに委ねられ、
アプリケーションコードから触れる層ではない。ドキュメンテーションコメント（`:127-140`）も「`simdgroup_load`/
`simdgroup_multiply_accumulate`/`simdgroup_store` は Apple GPU の行列専用命令にディスパッチされる」ことを前提として
おり、この抽象境界は実装時点から意図された設計である。

一方、`tgid` から C ブロック原点への割り当て（`row0 = tgid.y * BM` / `col0 = tgid.x * BN`、`:353-354`）は
threadgroup 単位の線形マッピングであり、カーネル冒頭の自前算術であるため、標準 API に触れずに変換式を差し替える
余地が構造的に存在する。

### 4. 兄弟イシューとの切り分け

`tgid` 変換レベルでのブロック割り当て再マップ（キャッシュ局所性向上を狙う技法）は、D-6 #540「threadgroup ID
スウィズル（swizzle_log 相当）」（本ドキュメント作成時点で OPEN）が同一目的（MLX steel の `swizzle_log` と同族の
ブロック走査順の Z オーダー化）を扱う。両者は「Z オーダー」という共通点はあるが対象レイヤが異なる:

| | 対象レイヤ | 狙い | 制御可否 |
|---|---|---|---|
| MFA の Morton 順（本 spike の対象） | simdgroup 内レーン→タイル内要素 | レーンのメモリコアレッシング | 標準 API 下では不可（本判断） |
| `tgid` スウィズル（#540 のスコープ） | threadgroup→C ブロック割り当て | L2 キャッシュ局所性 | 標準 API のみで可能 |

`tgid` 変換レベルの適用案（概略）: `tgid.x`・`tgid.y` の各ビットをインターリーブして 1 次元（または再分割した 2 次元）
の Z オーダー index に変換し、その index から `row0`/`col0` を再計算する。カーネル冒頭 2 行（`:353-354` 相当）の
置き換えのみで実現でき、8x8 タイル内部の境界チェック（`:596-608`）には影響しない。ただし本 spike ではこの適用案の
記載に留め、実装・効果計測は #540 のスコープとする（out-of-scope-tracking.md に沿い、本 PR ではコード変更を行わない）。

### 5. 既存決定記録との整合

`docs/backend-metal-async-copy-decision.md`（#546）は、Apple 非公開 ABI（AIR intrinsic への `__asm` 直接バインド）
への依存を「コンパイラ・OS 更新で無警告に壊れうる保守負債」として不採用と決定済みである。`simdgroup_matrix_storage<T>`
（`#pragma METAL internals : enable`）は AIR intrinsic への直接バインドではないが、同じく Apple 公開 MSL 仕様の外側
にある非公式領域（`internals` プラグマ自体が Apple 非公開の内部拡張であり将来の Xcode/コンパイラ更新で動作保証がない
点は #546 の判断根拠 1〈非公式性〉と同種）であり、#546 の決定が対象とする「非公式 API 系統」（草案 §0 A-5 系統）に
含めて扱う。`docs/backend-metal-mlx-classic-nax-decision.md`（#549）でも MLX classic 経路との構成対比において
本実装が標準 API 経路を選択している前提を確認済みであり、本判断はその前提と整合する。

## REQ-8（カーネル境界検査規約）との関係

`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」は、性能下限・最適化達成を理由にシェーダ・カーネル側の
手動境界チェックを省略しないことを CPU・CUDA・Metal の全カーネルに義務付ける。本判断は「simdgroup タイル内レーン→
要素マッピングを制御する手段の採否」という性能施策の選択であり、境界検査の省略とは無関係である。本ドキュメントの
不採用判断を「境界検査を省略してよい根拠」として引用してはならない（`docs/backend-metal-async-copy-decision.md`・
`docs/backend-metal-wgpu-decision.md` と同じ扱い）。

## 再検討条件

Phase D の他手段（#530 配下の兄弟イシュー。#540 の `tgid` スウィズル・ベクトル化ロード・パディング・タイル選択強化等）
を尽くしてもなお REQ-8 の性能目標に未達の場合に限り、再検討候補としてよい。ただしその場合も、非公式 API
（`#pragma METAL internals : enable` 下の自作ストレージ型）の採用はガードレール的に人間判断が必要な事項であり
**ユーザー承認必須**とする（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・#546 共通契約と同旨）。

## 出典

- MFA リポジトリ `philipturner/metal-flash-attention`（参照時点コミット `8671cddc38f19a6eadb804dee6a3ca2954b8bf32`。
  `gh api repos/philipturner/metal-flash-attention/commits/main` で解決。#546 の記録と同一コミット）
  - `Sources/FlashAttention/GEMM/GEMMHeaders.swift:536-552`（Morton 順レーン配置図・特許出典コメント）
  - `Sources/FlashAttention/GEMM/GEMMHeaders.swift:553-566`（`morton_order()` 関数本体）
  - `Sources/FlashAttention/GEMM/GEMMHeaders.swift:570-634`（`#pragma METAL internals : enable`〜`disable` 区間・
    `simdgroup_matrix_storage<T>` 定義）
  - `gh api "search/code?q=morton_order+repo:philipturner/metal-flash-attention"`（`morton_order` 参照箇所の横断確認。
    `GEMMKernel+Source.swift`・`AttentionKernel+Source.swift` 等）
- `crates/backend-metal/src/shaders/gemm.metal`
  - `:127-140`（`simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store` が Apple GPU 行列専用命令へ
    ディスパッチされる前提のドキュメンテーションコメント）
  - `:340-609`（`gemm_simdgroup_tiled` 本体）・`:353-354`（`tgid`→ブロック原点の線形マッピング）・
    `:524`・`:542-543`・`:606`（標準 API 経由のタイル読み書き）・`:596-608`（8x8 タイル単位の手動境界チェック）
- `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」
- `.claude/rules/security.md`「自己修復ループ固有のガードレール」
- `docs/backend-metal-async-copy-decision.md`（#546。同型の決定記録。フォーマットの踏襲元、非公式 API 不採用の
  既決事項の引用元）
- `docs/backend-metal-mlx-classic-nax-decision.md`（#549。標準 API 経路の前提確認）
- イシュー #544（本 spike）・#540（`tgid` スウィズル。OPEN・切り分け先）・親 #530・ルート #479
