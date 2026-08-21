# Metal GEMM バッファプール要否調査・判断記録（#809）

イシュー #809「perf(backend-metal): simdgroup_barrier 細粒度同期とバッファプール要否の調査」のうち、バッファ確保
パターンの調査と MLX バッファプール相当機構の導入要否判断を記録する。**本イシューのスコープはこの要否判断・
記録までであり、プール実装そのもの・接続作業は含まない**（`.claude/rules/out-of-scope-tracking.md` に従い切り出し
提案を末尾に記録する）。

## 1. 確保パターンの実態

`crates/backend-metal/src/gemm.rs`・`ops.rs` を Linux worktree で静的調査した結果:

- ベンチ・スライス入口 `MetalGemm::dispatch_variant`／`run_tiled_bias_act_f32` 等（`gemm.rs`）は**呼び出しごとに毎回**
  `MetalBuffer::new_with_data`（`newBufferWithBytes_length_options`）／`new_zeroed`（`newBufferWithLength_options`）で
  バッファを確保しており、再利用機構は無い。
- 本番経路 `MetalBackendOps::gemm`（`ops.rs:151-181`）は **1 呼び出しごとに** `MetalContext::new`（デバイス・
  コマンドキュー初期化）+ `MetalGemm::new`（`shaders/gemm.metal` のコンパイル + `gemm_naive`/`gemm_tiled`/
  `gemm_simdgroup`/`gemm_simdgroup_f16`/`gemm_tiled_bias_act` の 5 パイプライン構築。動的タイル選択構成は
  `pipeline_for_tile`/`pipeline_for_tile_f16` の遅延キャッシュ経由でさらに追加）+ 入出力バッファ確保を行う
  （`ops.rs::MetalBackendOps` のドキュメンテーションコメント: 「`MetalContext`／`MetalGemm` は各メソッド呼び出し時に
  都度構築する（`backend-cuda::ops::CudaBackendOps` と同じ設計判断。TASK-1.9b のデバイスハンドル常駐が未着地の
  ため）」）。すなわち固定費は**バッファ確保だけでなくパイプライン構築（MSL コンパイル込み）にも存在する**。
- **prepared 入口**（`dispatch_tiled_prepared`／`dispatch_f16_prepared_unverified`。イシュー #572）は、呼び出し元が
  あらかじめアップロード済みの `MetalBuffer` を渡す設計のため、確保・転送コストを計測境界から除外できる
  （`gemm_swizzle_ab_bench.rs`・本イシューの `gemm_fine_barrier_ab_bench.rs` の A/B 計測がこの入口を使う理由）。
  ただしこれは呼び出し側が確保を明示的に肩代わりする設計であり、`MetalBackendOps::gemm`（本番経路）自体には
  接続されていない。

## 2. 既存プール機構との関係

`crates/tensor-core/src/pool.rs::PooledMemory<M>`（TASK-#201・REQ-14 14-3。`docs/memory-pool-design.md`）は、
サイズ別バケット・総量上限・グローバル LRU 破棄を備えた opt-in デコレータ型として既に存在する。`M: MemoryOps +
PoolZeroFill` を満たす任意のバックエンドメモリ型を包めるため、`PooledMemory<crate::memory::MetalMemory>` は型として
成立する。しかし GEMM のスライスベース経路（`dispatch_variant`／`dispatch_auto`／`dispatch_tiled_prepared` 等）は
`MetalBuffer`（`buffer.rs`）を直接扱い `tensor_core::MemoryOps` トレイト経由の抽象を通らないため、**この機構は
GEMM 経路に未接続**である。

## 3. 導入要否の判断

**現時点では「導入すべき」と断定できるだけの定量根拠が実機側で得られていない**（本判断は Linux worktree での
コード事実に基づく定性判断であり、実機マイクロ計測は未実施。§4 参照）。判断の理由:

- 転送込み境界（`dispatch_auto`。ベンチマークで採用している計測境界。`docs/perf/gemm-optimization-baseline.md`
  §計測境界参照）では、既にホスト→デバイスのアップロード・デバイス→ホストの読み戻しという転送コスト自体が
  支配的になりうる形状帯（特に大形状）が存在し、バッファ確保コスト単体の寄与度が転送コストに対して相対的に
  小さい可能性がある。この相対寄与を実機計測なしに見積もるのは推定にとどまるため、断定を避ける。
- 一方、`MetalBackendOps::gemm`（本番経路）は `MetalContext::new` + `MetalGemm::new`（パイプライン構築）を
  **呼び出しごとに**行っており、これは §1 で確認した通りバッファ確保より重い固定費になりうる（MSL コンパイル込み
  の 5 パイプライン構築 + 動的タイル選択構成の遅延コンパイル）。この固定費はバッファプール導入では解消されず、
  コンテキスト／パイプラインのキャッシュ化という**別軸の対処**が必要になる。バッファプールのみを導入しても
  本番経路の主要な固定費（パイプライン構築）を解消できない可能性が高く、導入効果が限定的になるリスクがある。

よって本イシューでは、**バッファプール単体の導入を「未確定・実機計測待ち」として保留**し、プールを検討する際は
パイプライン構築コストの解消（下記スコープ外節）と合わせて評価する方針を記録する。

## 4. 期待効果の定量根拠（実機計測待ち）

Mac 実機（M4 Max）でのマイクロ計測（`MetalBuffer::new_with_data`/`new_zeroed` のサイズ別所要時間・5 回計測中央値、
`dispatch_variant`〈転送込み〉対 `dispatch_tiled_prepared`〈確保・転送除外〉の既存計測境界差分との突合）は本
Linux worktree セッションでは実施できていない（Metal 実機が同一セッションで使用できないため）。

Mac 実機セッションで以下を実施し、本節を実測値で更新すること:

1. `MetalBuffer::new_with_data`/`new_zeroed` の所要時間を size ∈ {256, 512, 1024, 2048, 4096} で 5 回計測し中央値を
   記録する（新規の独立 example または `gemm_fine_barrier_ab_bench.rs` への補助フェーズとして実施）。
2. `docs/perf/gemm-optimization-baseline.md`（転送込み境界）と `docs/perf/metal-gemm-tgid-swizzle-ab.md`（prepared
   境界）の既存計測記録から、同一サイズでの転送込み/prepared 境界の差分（＝確保 + 転送コストの合計）を突き合わせ、
   確保コスト単体の寄与度を見積もる。
3. 上記 2 点の実測値を基に、§3 の判断（導入要否）を再評価し、必要であれば本節・§3 を更新する。

## 5. 導入する場合の推奨接続点（実測後に判断を確定する前提の設計メモ）

もし §4 の実測で確保コストの寄与が有意（本番経路の総コストに対して無視できない割合）と判明した場合の推奨接続点:

- **`PooledMemory<MetalMemory>` の再利用**（推奨）: REQ-14 14-3 の係数上限・LRU 破棄・解放 API 契約に既に従って
  いるため、新規プール実装を追加せず既存機構を GEMM スライス経路へ接続する設計が望ましい。ただし現在の GEMM
  経路（`MetalBuffer` 直接操作）を `MemoryOps` トレイト経由へリファクタする必要があり、影響範囲が
  `gemm.rs`/`ops.rs`/`buffer.rs` 全体に及ぶため、独立した実装イシューとして切り出す。
- **GEMM 経路独自プール新設**（非推奨）: `PooledMemory` と重複する総量上限・LRU 機構を再実装することになり、
  2 つの独立したプール実装がバックエンド内に併存するリスクがある（保守コスト増）。既存機構の再利用を優先する。

## 6. スコープ外（記録のみ・本イシューへ混入禁止）

`.claude/rules/out-of-scope-tracking.md` に従い、実装対象外として記録する（ユーザー承認を得てから Issue へ切り
出す）:

- バッファプールの実装・`PooledMemory<MetalMemory>` の GEMM 経路接続（§5）
- `MetalBackendOps::gemm` の per-call `MetalContext::new` + パイプラインコンパイルのキャッシュ化（§1・§3 で判明した
  別軸の固定費。バッファプール単体より本番経路への影響が大きい可能性がある）
- §4 の実機マイクロ計測（Mac 実機セッションでの後続対応）

## 参照

- `crates/backend-metal/src/gemm.rs`（`MetalGemm::dispatch_variant`・`pipeline_for_tile`/`pipeline_for_tile_f16`）
- `crates/backend-metal/src/ops.rs`（`MetalBackendOps::gemm`）
- `crates/backend-metal/src/buffer.rs`（`MetalBuffer::new_with_data`/`new_zeroed`）
- `crates/tensor-core/src/pool.rs`（`PooledMemory<M>`。REQ-14 14-3・`docs/memory-pool-design.md`）
- `docs/perf/gemm-optimization-baseline.md`（計測境界の定義）
- `docs/perf/metal-gemm-fine-barrier-ab.md`（同イシュー #809 の simdgroup 細粒度同期 A/B 計測。本ドキュメントとは
  独立の判断対象）
