# Metal GEMM タイル構成: MLX classic 経路との対比・NAX 経路不採用判断（#549）

イシュー #549「MLX classic 経路と本実装の構成対比・NAX 経路非適用の判断を記録」に対応する。
親 #530（Phase D: Metal マルチ simdgroup 化・ロード最適化）・ルート #479（GEMM 性能改善）の D-11。
`crates/backend-metal/src/tile.rs` の `CANDIDATES`（GEMM タイル選択候補）は MLX steel カーネル
（`mlx/backend/metal/kernels/steel/gemm/`）の BM/BN/BK/WM/WN テンプレートパラメータ化を参照実装としており、
D-1（#532・クローズ済み）で classic 経路の未収録 3 構成を追加済みである。本ドキュメントは
`docs/backend-metal-async-copy-decision.md`（#546）・`docs/backend-metal-wgpu-decision.md`（#41・TASK-1.8d）
と同型の決定記録として、(1) classic 経路 6 構成と本実装 `CANDIDATES` の対比、(2) MLX の NAX 経路を
本実装に適用しない判断とその根拠を残す。**本イシューは docs 専用（コード変更ゼロ）である。**

## 判断サマリ

**MLX の NAX 経路（`MetalPerformancePrimitives` の `mpp::tensor_ops::matmul2d` を使う経路）は
`backend-metal` に不採用とする。** classic 経路 6 構成のうち #532 で本実装 `CANDIDATES` に
完全一致で取り込み済みなのは 5 構成（index 0・3・4・5・6）であり、残り 1 構成
`(32,64,16,1,2)` は本実装に完全一致では存在しない（最近傍として index 2
`(32,64,16,2,2)` があるが `wm`/`wn` が異なるため完全一致ではない。詳細は §1）。
本ドキュメントはその対比表と NAX 不採用の根拠を記録するのみで新規のコード変更は伴わない。

## §1 classic 経路 6 構成と `CANDIDATES` の対比表

出典: MLX リポジトリ（`ml-explore/mlx`、参照時点コミット `9ab977b5649154590d598ea5d545aa1b3c97f883`。
`gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）
`mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal:21-27`
（`instantiate_gemm_shapes_helper` マクロが `instantiate_gemm_transpose_helper` を 6 回展開する形で
`(bm,bn,bk,wm,wn)` 6 構成を実体化している。マクロ引数を実測確認済み）。

| MLX classic `(bm,bn,bk,wm,wn)` | MLX 出典行 | 本実装 `CANDIDATES` index | 本実装出典行 | 一致・差異 |
|---|---|---|---|---|
| `(64,64,16,2,2)` | `steel_gemm_fused.metal:22` | index 0 | `tile.rs:270-277` | 完全一致 |
| `(64,64,16,1,2)` | `steel_gemm_fused.metal:23` | index 4 | `tile.rs:308-315` | 完全一致（#532 追加） |
| `(64,32,32,2,2)` | `steel_gemm_fused.metal:24` | index 5 | `tile.rs:322-329` | 完全一致（#532 追加） |
| `(32,64,16,1,2)` | `steel_gemm_fused.metal:25` | 最近傍: index 2 | `tile.rs:288-295` | **本実装に完全一致で存在しない**（`wm`/`wn` 相違。下記注記） |
| `(32,32,16,2,2)` | `steel_gemm_fused.metal:26` | index 3 | `tile.rs:297-304` | 完全一致 |
| `(64,32,8,4,1)` | `steel_gemm_fused.metal:27` | index 6 | `tile.rs:332-339` | 完全一致（#532 追加） |

本実装 `CANDIDATES`（`tile.rs:268-342`。全 8 構成）のうち classic 6 構成に完全一致の対応が
付かない残り 3 構成:

| 本実装 `(bm,bn,bk,wm,wn)` | index | 出典行 | 備考 |
|---|---|---|---|
| `(64,32,16,2,2)` | index 1 | `tile.rs:279-286` | classic 6 構成に同一形状なし。本実装独自の縦長候補 |
| `(32,64,16,2,2)` | index 2 | `tile.rs:288-295` | classic の `(32,64,16,1,2)`（`wm=1,wn=2`）と `bm`/`bn`/`bk` は一致するが `wm`/`wn` が異なるため完全一致ではない（詳細下記） |
| `SINGLE_SIMDGROUP_8X8`（`8,8,8,1,1`） | index 7 | `tile.rs:142-149` | classic 6 構成に同一形状なし。微小形状フォールバック（`select` の下限構成） |

**重要な記録事項（差異）**: MLX classic の横長構成は `(32,64,16,1,2)`（`wm=1,wn=2`。64 スレッド）だが、
本実装の横長候補は `(32,64,16,2,2)`（index 2・`tile.rs:288-295`。`wm=2,wn=2`。128 スレッド）であり、
`wm`/`wn` の組が異なるため完全一致ではない。この差異解消（`(32,64,16,1,2)` の構成追加、または
`wm=1,wn=2` 系の一般化）は本 docs イシューのスコープ外であり、本ドキュメントはコード変更を行わない
（対応要否は「§3 再訪条件」とは別軸の未決事項として、PR 本文でユーザー判断に委ねる）。

**classic 経路の上限**: 6 構成を通じて `wm*wn` の最大は 4（128 スレッド）、`bk` の最大は 32
（`(64,32,32,2,2)`）である。本実装 `CANDIDATES` の全構成もこの上限内に収まる
（`TileConfig::validate`（`tile.rs:190`）の `TooManyThreads`／`ExceedsSharedMemory` 検証対象）。

## §2 NAX 経路の概要と非適用の根拠

出典: 同コミット `9ab977b5649154590d598ea5d545aa1b3c97f883` の下記ファイル。

### 経路の実体

- `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal:23-29` は classic と同型の
  `instantiate_gemm_shapes_helper` マクロ展開で NAX 専用の 6 構成
  `(64,64,256,2,2)` `(64,128,64,2,4)` `(64,128,256,2,4)` `(128,128,64,4,4)` `(128,128,256,4,4)`
  `(128,128,512,4,4)` を実体化する。**`bm` 最大 128・`bn` 最大 128・`bk` 最大 512・`wm*wn` 最大 16
  （`thread_count()`＝`wm*wn*32`。`tile.rs` の同メソッド定義参照で 16*32＝512 スレッド）**であり、
  いずれも classic 6 構成（§1 表）の上限（`wm*wn<=4`・`bk<=32`）を大きく超える。
  この大構成は NAX 経路専用であり classic 経路には存在しない（§1 対比表に混入させてはならない根拠）。
- `mlx/backend/metal/kernels/steel/gemm/nax.h:12` が `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
  し、同 `:401-411`／`:473-483` で `mpp::tensor_ops::matmul2d_descriptor` と
  `mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>` を使う。これは Apple 公開の
  `MetalPerformancePrimitives` フレームワーク（Neural Accelerator 向け行列演算 API。導入 Metal バージョンは
  参照コミット中に確認できておらず本ドキュメントでは未確認とする）であり、
  `docs/backend-metal-async-copy-decision.md`（#546）が扱った非公開 AIR intrinsic 直接バインドとは
  性質が異なる（**API 自体は公開だが、ハードウェア前提が異なる**点が本経路不採用の主眼）。
- `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h:35-36` で `TM = SM/16`・`TN = SN/16`（`:45` の
  `NAXTile<AccumType, TM, TN>` 等で使用）としており、16 要素刻みのタイル（フラグメント）粒度で
  `matmul2d` を呼ぶ構造になっている。

### ディスパッチ条件（ハードウェア世代ゲート）

`mlx/backend/metal/device.cpp:598-606`（`get_architecture_gen()` の算出。`arch_gen_ = ag_tens * 10 + ag_ones;`
が `:606`）・`:952-970`（`bool is_nax_available()`）が NAX 経路の有効化条件を定義する:

```cpp
bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}
```

すなわち NAX 経路は「**macOS/iOS/tvOS/visionOS 26.2 以上**」かつ「**デバイスの GPU アーキテクチャ世代
（`get_architecture_gen()`。`device.cpp:598-606` で Metal `MTLDevice::architecture()->name()` 末尾の 2 桁数字
から算出）が非 phone で 17 以上（phone は 18 以上）**」の両方を満たす場合のみ有効化される
（`mlx/backend/metal/matmul.cpp:917-920` 等の呼び出し箇所で `use_nax = is_nax_available() && ...` の形で
GEMM ディスパッチ分岐に使われる）。MLX は世代非対応デバイスでは自動的に classic 経路（本実装が参照する
`steel_gemm_fused.metal`）へフォールバックする設計である。

### 非適用の根拠

1. **実機検証環境の世代ゲート不一致**: 本実装の実機検証環境は Apple M4 Max・macOS 26.6
   （`docs/perf/metal-gemm-dynamic-tile.md:10,53` 実測）であり、macOS バージョン条件（26.2 以上）自体は
   満たすが、GPU ハードウェアに Neural Accelerator（`MetalPerformancePrimitives` の `matmul2d` が利用する
   専用ユニット）を搭載しない。Apple は Neural Accelerator を M5 世代 GPU コアの新機能として公表しており
   （Apple 公式発表。参照 URL は「§4 参照」節）、`is_nax_available()` の世代しきい値（`gen>=17`）は
   この世代差をハードウェア側で機械的にゲートしていると解釈できる。ただし `get_architecture_gen()` が
   実際に返す世代番号と Apple のチップ世代（M4・M5 等）とのマッピングを MLX ソース中に明示するコメントは
   参照コミット中に見当たらず、**厳密な数値対応（M4 が具体的に何番の gen 値になるか）は本ドキュメントでは
   未確認**とし、推定断定はしない（`deps-policy.md` の「推定で記述せず実測確認する」原則の準用）。
   確実に言えるのは「NAX 経路には macOS バージョンとは別に GPU 世代のゲートが存在し、M4 Max 実機の
   結果がそのまま NAX 経路の有効化を保証しない」という構造的事実である。
2. **MLX 自身のハードウェア世代ディスパッチ**: 上記のとおり MLX 自体が `is_nax_available()` により
   世代非対応デバイスでは NAX 経路を選択せず classic 経路へフォールバックする設計であり、
   NAX 専用構成（`bm=128,bn=128,bk=512,wm=4,wn=4` 等）は Neural Accelerator 非搭載デバイス上では
   そもそも実行されない。
3. **公開 API 依存方針との整合**: NAX 経路が使う `MetalPerformancePrimitives`（`mpp::tensor_ops::matmul2d`）
   自体は Apple 公開 API であり、`docs/backend-metal-async-copy-decision.md`（#546）が不採用とした
   非公開 AIR intrinsic（`__asm("air.simdgroup_async_copy_2d...")`）とは性質が異なる。本不採用判断は
   API の公開性を理由とするものではなく、**現行実機検証環境（M4 Max）でハードウェア的に検証不能**な
   経路をコードベースへ持ち込まない、という実証可能性の観点による（`.claude/rules/coding-rust.md`
   「テスト・ベンチ」節: 実機依存は `#[ignore]` 分離が前提であり、そもそも実行できない経路を
   `#[ignore]` テストとして追加しても実証手段がない）。
4. **classic 経路で REQ-8 目標を追求する現行方針との整合**: `docs/kernel-fusion.md`
   （複合ワークロードでカーネル融合を性能目標の前提にしない）と同様、本実装は現行の実機検証環境で
   実証可能な手段（Phase D の他施策: マルチ simdgroup 化・ロード最適化・タイル選択強化）を優先する
   方針であり、実証不能な NAX 経路を性能目標達成の前提に組み込まない。

## §3 再訪条件（`out-of-scope-tracking.md` 対応）

以下をすべて満たす場合に限り、NAX 経路採否を再検討してよい。

1. **M5 世代（Neural Accelerator 搭載）実機の入手**: `docs/real-hardware-verification-env.md` の
   実機検証環境に M5 世代 Mac が追加されること。
2. **対応 macOS・`MetalPerformancePrimitives` の利用可否確認**: 実機上で macOS 26.2 以上が稼働し、
   `is_nax_available()` 相当の判定が真になること（`gen>=17` を満たす世代であることの実機実測）。
3. **その時点の REQ-8 目標達成状況**: classic 経路（Phase D の他施策込み）で REQ-8 性能下限に
   未達の場合に限り、NAX 経路を再検討候補とする（`docs/backend-metal-async-copy-decision.md`
   「再検討条件」節と同型の判断軸）。

再訪時は新規イシュー起票（`.claude/rules/out-of-scope-tracking.md` のフロー）から着手し、
**ユーザー承認必須**とする（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・
本イシュー共通契約と同旨）。本ドキュメント自体は再訪条件の記録に留め、イシュー化は行わない。

`(32,64,16,1,2)` 差異（§1）についても、対応要否の判断・イシュー化はユーザー判断に委ねる
（本ドキュメントは記録のみ）。

## §4 参照

- MLX リポジトリ `ml-explore/mlx`（参照時点コミット `9ab977b5649154590d598ea5d545aa1b3c97f883`。
  `gh api repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）
  - `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal:21-27`（classic 6 構成）
  - `mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal:23-29`（NAX 6 構成）
  - `mlx/backend/metal/kernels/steel/gemm/nax.h:12,401-411,473-483`
    （`MetalPerformancePrimitives` インクルード・`mpp::tensor_ops::matmul2d` 呼び出し）
  - `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h:35-36`（`TM=SM/16`・`TN=SN/16` の 16 要素粒度タイル）
  - `mlx/backend/metal/device.cpp:598-606,952-970`（`get_architecture_gen()`・`is_nax_available()`）
  - `mlx/backend/metal/matmul.cpp:917-920`（`use_nax` ディスパッチ分岐の呼び出し例）
- Apple 公式発表（M5 GPU コアへの Neural Accelerator 搭載）: <https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/>
- 本実装 `crates/backend-metal/src/tile.rs:268-342`（`CANDIDATES`）・`tile.rs:142-149`（`SINGLE_SIMDGROUP_8X8`）・
  `tile.rs:190`（`TileConfig::validate`）
- `docs/perf/metal-gemm-dynamic-tile.md:10,53`（実機検証環境: Apple M4 Max・macOS 26.6）
- `docs/backend-metal-async-copy-decision.md`（#546。同型の決定記録・公開/非公開 API の対比軸）
- `docs/backend-metal-wgpu-decision.md`（#41・TASK-1.8d。同型の決定記録のフォーマット踏襲元）
- `docs/kernel-fusion.md`（TASK-12.2b。実証不能な性能施策を目標の前提にしない方針との整合）
- `.claude/rules/coding-rust.md`「テスト・ベンチ」節（実機依存テストの `#[ignore]` 分離）
- `.claude/rules/out-of-scope-tracking.md`・`.claude/rules/security.md`「自己修復ループ固有のガードレール」
- イシュー #549・親 #530（D-11）・ルート #479・関連 #532（classic 経路取り込み）・#546（同型決定記録）
