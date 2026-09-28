# MLX steel GEMM 最新版（v0.32.2）の読み取り解析（#549 NAX 続編）

イシュー #2096（親 #2089「他ライブラリのコード取得・詳細解析（負けセルの原因
帰属）」Phase 2）に対応する。`docs/backend-metal-mlx-classic-nax-decision.md`
（#549。2026-08-15 時点の MLX を参照コミット固定で解析）以降、MLX 本体・
本実装 `tile::CANDIDATES` の双方が更新されたため、最新版 MLX を**読み取り
のみ**で再解析し、(1) classic 経路の構成増減、(2) NAX 経路の最新条件、
(3) #549 執筆時点との差分、(4) ライセンス、の 4 点を記録する。記録内容は
Phase 3（#2098「負けセル分析・原因帰属」）の入力として使う。

## 1. 位置づけ・解析契約

- **読み取りのみ**: 上流のコード・シェーダ・派生物は本リポへ持ち込まない。
  引用は「結論＋出典（パス:行＋一行要約）」の形式のみとし、上流コード
  ブロックの逐語貼り付けは行わない
- **対象版**: `ml-explore/mlx` リポジトリ
  - タグ `v0.32.2` = `1f8e74e3f12f31365464a6867c6579f0e9b29d85`
    （`git rev-parse v0.32.2` で解決。2026-09-28 時点の最新リリースタグ）
  - `main` HEAD = `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`
    （2026-09-28T15:37:06+02:00。タグなし。§6 で「タグ以降・未リリース」
    として別枠に記す）
  - #549 参照コミット `9ab977b5649154590d598ea5d545aa1b3c97f883`
    （2026-08-15）
- **取得・再現コマンド**:
  ```
  git clone --filter=blob:none https://github.com/ml-explore/mlx.git
  git -C mlx rev-parse v0.32.2
  git -C mlx log -1 --format='%H %cI' origin/main
  git -C mlx diff --stat 9ab977b5649154590d598ea5d545aa1b3c97f883..v0.32.2 \
    -- mlx/backend/metal/kernels/steel/gemm mlx/backend/metal/matmul.cpp \
       mlx/backend/metal/matmul.h mlx/backend/metal/device.cpp
  git -C mlx diff --stat v0.32.2..origin/main \
    -- mlx/backend/metal/kernels/steel/gemm mlx/backend/metal/matmul.cpp \
       mlx/backend/metal/matmul.h mlx/backend/metal/device.cpp
  ```
  clone は scratchpad 配下で読み取り専用に行い、cmake・pip・ビルドスクリプト
  は一切実行していない（`git diff`／`git show`／`grep` のみ）。解析後に
  削除済み
- **スコープ外**: 性能実測そのもの（Phase 3）、NAX 相当機能の本実装への
  持ち込み判断、cubecl IR 相当の深掘り（本 doc の対象外の別軸）

## 2. 結論サマリ

1. **classic 経路（`steel_gemm_fused.metal`）の 6 構成は #549 時点から
   一切変わっていない**（`(64,64,16,2,2)`・`(64,64,16,1,2)`・
   `(64,32,32,2,2)`・`(32,64,16,1,2)`・`(32,32,16,2,2)`・
   `(64,32,8,4,1)`。§4）。`v0.32.2..origin/main` の diff にも
   `steel_gemm_fused.metal` は現れない
2. **本実装側 `tile::CANDIDATES` は #549 時点の 8 構成から 11 構成
   （index 0〜10）へ増えており、classic 6 構成全てに完全一致する
   index が揃った**（#549 が「完全一致なし」としていた
   `(32,64,16,1,2)` は #1143 で `CANDIDATES[8]` として追加済み。§4）。
   ただし性能実測（`docs/perf/metal-gemm-n4096-kernel-gap.md` §13）では
   本番選択構成に対し劣後と判定され `select` の選択ロジックには組み込まれ
   ていない（§8）
3. **NAX 経路のディスパッチ条件（`is_nax_available()`・
   `get_architecture_gen()`）は #549 時点から不変**。`v0.32.2` までの
   diff は `device.cpp` の Event 実装のリファクタリングのみで、NAX の
   世代ゲート条件（macOS/OS 26.2 以上・`gen>=17`〈phone は 18〉）は
   変わっていない（§5）
4. **`main` HEAD（タグ以降・未リリース）では、NAX 経路に Ultra チップ
   （`devc == 'd'`）向けの専用タイル分岐が新規追加されている**。これは
   #549・v0.32.2 のいずれにも存在しない新規差分であり、**チップ「サイズ」
   区分（base/pro・max・ultra）の分岐であり、Neural Accelerator の世代
   ゲート（`get_architecture_gen()`）とは別軸**である点に注意（§6）。
   一方 **split-K nax（`steel_gemm_splitk_nax.metal`）・gather nax
   （`steel_gemm_gather_nax.h`）は #549 時点から既に存在する既存経路
   であり、main HEAD での変更は新規追加ではなく既存実装への軽微な変更
   （split-K nax は 2 行差分）・リファクタリング（gather nax）にとどまる
   （§6・§7 の「原型のみ → リファクタリング」参照）
5. **ライセンスは MIT 単独**（`Copyright © 2023 Apple Inc.`）。GitHub API
   の SPDX も `MIT`。Issue #2096 の受入条件にある「MIT・Apache-2.0
   dual」は実測で裏付けられなかった（§9）

## 3. 取得・差分確定

`9ab977b5649154590d598ea5d545aa1b3c97f883..v0.32.2`（classic/NAX GEMM
関連パス限定）:

```
mlx/backend/metal/device.cpp | 43 +++++++++++++++++++------------------------
 1 file changed, 19 insertions(+), 24 deletions(-)
```

`steel/gemm/` 配下・`matmul.cpp`／`matmul.h` は無変更。`device.cpp` の
変更は `CommandEncoder::signal_event`／`wait_event`／`commit`／
`synchronize` の Event 型（`std::shared_ptr<EventImpl>` → 値型 `Event`
＋ `error_.check()` 方式へのリファクタリング）で、`is_nax_available()`・
`get_architecture_gen()`・`arch_gen_` 算出ロジックには触れていない
（§5 で実測比較）。

`v0.32.2..origin/main`（同パス）:

```
 mlx/backend/metal/device.cpp                                          |   9 +-
 .../steel/gemm/kernels/steel_gemm_fused_nax.metal                     |   2 +
 .../steel/gemm/kernels/steel_gemm_gather.h                            | 283 +++++++++------------
 .../steel/gemm/kernels/steel_gemm_gather_nax.h                        | 142 ++++-------
 .../steel/gemm/kernels/steel_gemm_splitk_nax.metal                    |   2 +
 mlx/backend/metal/matmul.cpp                                          | 103 ++++++--
 mlx/backend/metal/matmul.h                                            |   7 +
 7 files changed, 273 insertions(+), 275 deletions(-)
```

§6 で「タグ以降・未リリース」として個別に扱う。

## 4. classic 経路の対比表（タグ時点。v0.32.2）

出典: `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal:21-26`
（`instantiate_gemm_shapes_helper` マクロが `instantiate_gemm_transpose_helper`
を 6 回展開。#549 時点の行番号 `:21-27` から 1 行分ずれているだけで内容は
不変）。

| MLX classic `(bm,bn,bk,wm,wn)` | MLX 出典行 | 本実装 `CANDIDATES` index（現行 HEAD） | 本実装出典行 | 一致・差異 |
|---|---|---|---|---|
| `(64,64,16,2,2)` | `steel_gemm_fused.metal:21` | index 0 | `tile.rs:854-861` | 完全一致 |
| `(64,64,16,1,2)` | `steel_gemm_fused.metal:22` | index 4 | `tile.rs:892-899` | 完全一致 |
| `(64,32,32,2,2)` | `steel_gemm_fused.metal:23` | index 5 | `tile.rs:907-914` | 完全一致 |
| `(32,64,16,1,2)` | `steel_gemm_fused.metal:24` | index 8 | `tile.rs:940-947` | **完全一致（#549 時点は不一致・#1143 で追加）** |
| `(32,32,16,2,2)` | `steel_gemm_fused.metal:25` | index 3 | `tile.rs:881-888` | 完全一致 |
| `(64,32,8,4,1)` | `steel_gemm_fused.metal:26` | index 6 | `tile.rs:917-924` | 完全一致 |

本実装 `CANDIDATES`（`tile.rs:852-1004`。現行 HEAD で全 11 構成・
index 0〜10）のうち classic 6 構成に対応が付かない残り 5 構成:

| 本実装 `(bm,bn,bk,wm,wn)` | index | 出典行 | 備考 |
|---|---|---|---|
| `(64,32,16,2,2)` | 1 | `tile.rs:863-870` | classic 6 構成に同一形状なし（本実装独自の縦長候補。#549 時点から不変） |
| `(32,64,16,2,2)` | 2 | `tile.rs:872-879` | classic の横長構成とは `wm`/`wn` が異なる（#549 §1 の差異はそのまま残るが、index 8 が別途完全一致を担うため実質解消。§8 参照） |
| `SINGLE_SIMDGROUP_8X8`（`8,8,8,1,1`） | 7 | `tile.rs:240,926` | 微小形状フォールバック（#549 時点から不変） |
| `(64,64,32,2,2)` | 9 | `tile.rs:961-968` | classic 6 構成に同一形状なし（E7・#1329。`select` 系には未組み込み・明示指定限定） |
| `(128,64,16,2,2)` | 10 | `tile.rs:996-1003` | classic 6 構成に同一形状なし（E8・#1331。同上・未組み込み） |

**classic 経路の構成範囲（不変）**: 6 構成を通じて `bm`・`bn` の最大は
64、`wm*wn` の最大は 4（128 スレッド）、`bk` の最大は 32
（`(64,32,32,2,2)`）。本実装 `CANDIDATES` の index 9
（`(64,64,32,2,2)`。`bk=32`）は `wm*wn`・`bk` の軸では classic の範囲内
だが `bm=64,bn=64` は classic 6 構成のどの形状とも一致しない。index 10
（`(128,64,16,2,2)`。`bm=128`）は `bm` が classic の最大値 64 を超えて
おり、classic の構成範囲には収まらない。**`TileConfig::validate`
（`tile.rs:666` 以下）は `bm % (wm*8) == 0`／`bn % (wn*8) == 0`／
`bk % 8 == 0` の整除制約と、アキュムレータ行列サイズ・スレッド数・
共有メモリ量等のデバイス上限を検証するものであり、classic 6 構成の
範囲（`bm`/`bn` の最大値等）を検証するものではない**——index 9・
index 10 とも `validate` を通過するのは単にデバイス上限を満たすためで
あって、classic の構成範囲内であることを意味しない。両者とも classic
6 構成の形状そのものには存在しない本実装独自候補である。

## 5. NAX 経路の解析（タグ時点。v0.32.2）

出典: `mlx/backend/metal/device.cpp:947-967`（`is_nax_available()`）。
#549 が引用した内容と**完全一致**（`__builtin_available(macOS 26.2, …)`
かつ `get_architecture_gen() >= (arch=='p' ? 18 : 17)`）。行番号のみ
`:952-970` → `:947-967` へ僅かにずれている（§3 の Event リファクタリング
による前方の行数変化）。

- `steel_gemm_fused_nax.metal:22-27` の NAX 専用 6 構成
  （`(64,64,256,2,2)` `(64,128,64,2,4)` `(64,128,256,2,4)`
  `(128,128,64,4,4)` `(128,128,256,4,4)` `(128,128,512,4,4)`）も
  #549 時点から不変
- `nax.h:12,401,408,411,473,480,483`（`MetalPerformancePrimitives`
  インクルード・`mpp::tensor_ops::matmul2d` 呼び出し）も不変
- `matmul.cpp` の `use_nax = metal::is_nax_available() && …`（複数箇所。
  `:917`・`:2839`・`:2918` 付近）も v0.32.2 時点で不変（§3 の diff
  に `matmul.cpp` が現れないことから確認）

**M5 世代以降の対応状況**: `get_architecture_gen()` が返す世代番号
（Metal `MTLDevice::architecture()->name()` 末尾 2 桁）と Apple のチップ
世代（M4・M5 等）との明示的な対応表は、v0.32.2 のソース中にも見当たら
なかった（`git grep -n "gen ==" mlx/backend/metal/device.cpp` 等で該当
コメントなし）。#549 と同じく「未確認」のまま残す（推定断定はしない。
`deps-policy.md` の「推定で記述せず実測確認する」原則の準用）。

## 6. タグ以降・未リリースの変更（`main` HEAD `64ea011c…`）

**本節は v0.32.2 未収録・未リリースの変更であり、§4・§5 の結論（タグ
時点の対比）とは混ぜない。**

- `matmul.cpp` の `steel_matmul_regular_axpby_nax`・
  `steel_gemm_splitk_axpby_nax` に `devc == 'd'`（`device.cpp:616`
  `case 'd': // ultra`。チップの**サイズ区分**〈base/pro='g'・max='s'・
  ultra='d'〉であり、NAX 世代ゲート〈`get_architecture_gen()`〉とは
  独立の軸）向けの専用タイル分岐（`N > M` で `bm=128,bn=64,wm=4,wn=2`・
  それ以外で `bm=64,bn=128,wm=2,wn=4`）と `swizzle_log` 調整が追加され、
  従来は `'s'`／`'c'`（`'c'` は不明。参照コミット中に区分名コメントなし）
  と同じ扱いだった `'d'` が独立分岐へ分離された
- `steel_gemm_fused_nax.metal`・`steel_gemm_splitk_nax.metal` は
  2 行のみの差分（インクルード等の軽微変更と推定。本 doc では詳細解析
  対象外）
- `steel_gemm_gather.h`／`steel_gemm_gather_nax.h` は `align_M` function
  constant の削除・`gather_mm_offsets` 関数の新設等、gather 系 GEMM
  （MoE 等の indices 付き matmul）のリファクタリングで、classic／NAX
  の dense GEMM 構成そのものには影響しない
- `device.cpp` の main HEAD 差分は `CommandEncoder` の Event 引数を
  値渡し `Event` へ統一する追加リファクタリング（§3 の v0.32.2 時点の
  変更の延長線上）であり、`is_nax_available()`／`get_architecture_gen()`
  自体への変更はここでも確認できなかった

**注記**: `devc == 'd'`（ultra）向けタイル分岐は、Neural Accelerator の
有無（世代ゲート）とは無関係にチップサイズだけで分岐する既存の
`steel_matmul_regular_axpby_nax`（NAX 経路内部の関数）の一部である。
関数名に `_nax` を含むが、この差分自体は「NAX 経路が有効な場合の中で
ultra チップ向けにさらにタイルを変える」実装であり、NAX 経路自体の
有効化条件（§5 の `is_nax_available()`）を変えるものではない。

## 7. #549 との差分表

### MLX 側の差分

| 観点 | #549 時点（`9ab977b5…`） | v0.32.2（本 doc） | main HEAD（未リリース） |
|---|---|---|---|
| classic 6 構成 | 不変の基準 | **無変更** | 無変更（§3 diff に不在） |
| NAX 世代ゲート条件 | `gen>=17`（phone 18）・macOS 26.2 以上 | **無変更** | 無変更（§6 で確認） |
| Ultra チップ専用タイル分岐 | なし（`'s'`／`'c'`／`'d'` 共通） | なし | **新規追加**（§6） |
| gather 系 NAX 対応 | 原型のみ | 原型のみ | リファクタリング（§6） |
| `device.cpp` の Event 実装 | 旧方式（`shared_ptr<EventImpl>`） | 値型 `Event` へ移行 | 継続移行 |

### 本実装側の差分

| 観点 | #549 時点 | 現行 HEAD（本 doc） |
|---|---|---|
| `CANDIDATES` 総数 | 8（index 0〜7） | **11（index 0〜10）** |
| `(32,64,16,1,2)` の完全一致 | **なし**（最近傍 index 2 のみ・`wm`/`wn` 相違） | **あり**（index 8。#1143） |
| classic 6 構成の完全一致率 | 5/6 | **6/6** |
| `select`／`select_for_device` への組み込み | index 0〜6 が候補 | 不変（index 8〜10 は明示指定限定。§8） |

## 8. 既存記録との突合（再実験はしない）

| 既存記録 | 内容 | 本 doc との関係 |
|---|---|---|
| `docs/perf/metal-gemm-n4096-kernel-gap.md` §13（イシュー #1143） | classic 未収録構成 `(32,64,16,1,2)`（`CANDIDATES[8]`）を M4 Max 実機で追加測定した結果、劣後のため不採用・`select` 変更なし | **§4 の「完全一致」は構造上の一致であり、性能面では既に REJECT 済み**。Phase 3 で再実験する価値は低い（未試行候補には含めない） |
| `docs/perf/metal-gemm-n4096-kernel-gap.md` §7〜§19（E1〜E9） | unroll・特殊化・フラグメントロード・協調ロード・タイルクラス分割・`CANDIDATES[9]`（`(64,64,32,2,2)`。bk=32）・`CANDIDATES[10]`（`(128,64,16,2,2)`。128 幅タイル）の実機実測。いずれも本番選択構成に対し REJECT または未組み込み | classic 6 構成には `bk=32` の形状自体は存在する（`(64,32,32,2,2)`。§4）が、`CANDIDATES[9]` とは `bn`（32 対 64）が異なり完全一致しない。`bm=128` の構成（`CANDIDATES[10]`）は classic 6 構成に同型のものが存在しない（§4 の上限記述のとおり classic の `bm` 最大は 64）。NAX 側にも両候補と同型の構成はない。MLX 側との直接対応はないが、「本実装独自候補は測定済みかつ大半 REJECT」という前提は変わらない |
| `docs/backend-metal-mpp-tensor-decision.md`（#1326） | Metal 4 `tensor<>`＋MPP `matmul2d`（NAX が使うのと同じ API 系統）の可用性・純カーネル時間 A/B を M4 Max 実機で検証（採否は未確定のまま整理） | NAX 経路自体（`mpp::tensor_ops::matmul2d`）と同じ MPP API 系統を対象にした既存調査であり、本 doc の§5 NAX 経路解析と技術的に重なる。#1326 は「採否の結論は出さない」整理のみで、§3 再訪条件（M5 実機必須）は本 doc でも不変 |
| `docs/backend-metal-splitk-decision.md`・`docs/perf/metal-gemm-splitk-*.md` | 自作 split-K（f32・2 パス）の A/B・本番結線（実行時トグル opt-in・既定 `true`） | MLX 側の `steel_gemm_splitk_nax.metal`（#549 時点から存在する既存経路。main HEAD での変更は 2 行差分のみで詳細未解析。§2・§7）とは別実装。直接比較は未実施（未試行候補として§10 に列挙） |
| `docs/backend-metal-async-copy-decision.md`（#546）・`docs/backend-metal-aligned-load-decision.md`・`docs/backend-metal-morton-mapping-decision.md` | 非公開 AIR intrinsic 不採用・アラインメント特化ロード不採用・Morton マッピング不採用 | NAX 経路（MPP API）は「API 公開性」の観点では #546 と性質が異なる（§9 の #549 記述を継承）。本 doc はこの整理を覆さない |
| PR #2355（`docs/analysis/candle-metal-01.md`。マージ前） | candle Metal GEMM のタイル選択・スウィズル・同期方式を `tile::CANDIDATES`／`select_for_device` と対比。cand0 実験で差の主因はタイル選択でなくカーネル本体と結論 | 対象ライブラリが異なる（candle vs MLX）ため直接の重複はないが、「本番選択構成と特定候補を揃えた A/B」という調査方法論は共通。Phase 3 での横断比較の参考にできる |

**未試行の差分候補（Phase 3 引き継ぎ）**:

- Ultra チップ向け専用タイル分岐（§6）は DGX Spark／M4 Max のいずれの
  実機検証環境にも該当しない（Ultra チップは未保有）ため、実証手段が
  ない。§3 再訪条件と同型の「実機なし」制約
- NAX split-K（`steel_gemm_splitk_nax.metal`。#549 時点から存在する
  既存経路で main HEAD での変更は 2 行差分のみ）と自作 split-K
  （`docs/backend-metal-splitk-decision.md`）の設計対比は未実施。
  gather nax とは異なり本実装に対応する仕組み（自作 split-K）が
  あるため、Phase 3 の設計対比候補として扱う
- gather 系 GEMM（`steel_gemm_gather*.h`）は本実装の GEMM 経路に対応物
  がなく比較対象外

## 9. ライセンス記録

- `v0.32.2` の `LICENSE`（リポジトリルート）: **MIT License
  単独**（`Copyright © 2023 Apple Inc.`）
- GitHub API: `gh api repos/ml-explore/mlx --jq '.license'` →
  `{"key":"mit","name":"MIT License","spdx_id":"MIT",…}`
- `ACKNOWLEDGMENTS.md`（リポジトリルート）: 個人貢献者のクレジット一覧
  であり、第三者ライセンス通知（NOTICE 相当）ではない。`NOTICE` 相当の
  ファイルはリポジトリ中に見当たらなかった
- **イシュー #2096 の受入条件にある「MIT・Apache-2.0 dual」という記述は、
  上記実測（`LICENSE` ファイル本文・GitHub API の SPDX 値）と一致せず、
  裏付けられなかった**。イシュー記述の訂正要否はユーザー判断に委ねる
  （§10）
- 本 doc・本 PR は上流コードの引用・持ち込みを行わず、結論と出典
  （パス・行番号・上流 URL）のみを記録するため、著作権表示・ライセンス
  全文の同梱義務は発生しない。本イシューは新規の依存追加を伴わないため
  `docs/license-matrix.md` の変更は行わない

## 10. Phase 3 引き継ぎ・対象外

### Phase 3（#2098 以降）への引き継ぎ候補

- classic 6 構成は自作 `CANDIDATES` と構造上 6/6 完全一致だが、性能面
  では `(32,64,16,1,2)`（index 8）が既に REJECT 済み（§8）。MLX の
  タイル選択自体（形状ごとにどの classic 構成を選ぶか）と本実装
  `select_for_device` の選択ロジックの対比は未実施
- Ultra チップ専用分岐（main HEAD 限定）は実機なしのため検証不能。
  実機入手時の再訪候補として記録するのみ
- NAX split-K は自作 split-K との設計対比が未実施（§8・上記引き継ぎ
  候補に計上）。gather nax は本実装に対応する仕組みがなく比較対象外

### 対象外（PR 本文に記録）

- `docs/backend-metal-mlx-classic-nax-decision.md`（#549）§1 対比表の
  訂正（本 doc が新しい対比表を作ったこと自体の反映。訂正要否は
  ユーザー判断）
- `docs/README.md` の `analysis/` ブロック重複（18 行目付近・156 行目
  付近の 2 箇所。並行マージの結果。統合は別対応）
- NAX 相当機能を本実装へ持ち込むかどうかの判断（#549 §3 再訪条件が
  不変のまま引き継がれる。M5／Ultra 実機必須）
- 列挙した差分候補の実測（Phase 3・#2098 以降）
- Issue #2096 受入条件のライセンス記述（MIT・Apache-2.0 dual）との
  不一致（§9。訂正は Issue 側の事項として報告のみ）

## 11. 参照

- MLX リポジトリ `ml-explore/mlx`
  - タグ `v0.32.2`（`1f8e74e3f12f31365464a6867c6579f0e9b29d85`）
    - `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal:21-26`
      （classic 6 構成）
    - `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal:22-27`
      （NAX 6 構成）
    - `mlx/backend/metal/kernels/steel/gemm/nax.h:12,401,408,411,473,480,483`
    - `mlx/backend/metal/device.cpp:616`（`case 'd': // ultra`）・
      `:947-967`（`is_nax_available()`）
    - `mlx/backend/metal/matmul.cpp:917`（`use_nax` ディスパッチ例）
    - `LICENSE`（MIT License・Copyright © 2023 Apple Inc.）
  - `main` HEAD（`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`。
    2026-09-28T15:37:06+02:00）
    - `mlx/backend/metal/matmul.cpp`（`devc == 'd'` 専用タイル分岐）
- `gh api repos/ml-explore/mlx --jq '.license'`（SPDX 実測）
- 本実装 `crates/backend-metal/src/tile.rs:852-1004`（`CANDIDATES`。
  現行 HEAD 全 11 構成）
- `docs/backend-metal-mlx-classic-nax-decision.md`（#549。本 doc の前段）
- `docs/perf/metal-gemm-n4096-kernel-gap.md` §7〜§19（E1〜E9・
  `(32,64,16,1,2)` 追加測定）
- `docs/backend-metal-mpp-tensor-decision.md`（#1326。MPP `matmul2d` の
  可用性調査）
- `docs/backend-metal-splitk-decision.md`・`docs/perf/metal-gemm-splitk-*.md`
- `docs/backend-metal-async-copy-decision.md`（#546）・
  `docs/backend-metal-aligned-load-decision.md`・
  `docs/backend-metal-morton-mapping-decision.md`
- PR #2355（`docs/analysis/candle-metal-01.md`。マージ前）
- イシュー #2096・親 #2089・関連 #549・#1143・#1329・#1331

### 受入条件との対応

| # | 受入条件 | 対応節 |
|---|---|---|
| 1 | 最新 MLX の取得・解析時点コミット SHA の記録 | §1 |
| 2 | classic 構成の増減の新しい対比表 | §4 |
| 3 | NAX 側の最新条件・M5 世代以降の対応状況 | §5・§6 |
| 4 | #549 執筆時点との差分 | §7 |
| 5 | 既存記録との突合 | §8 |
| 6 | ライセンス記録 | §9 |
