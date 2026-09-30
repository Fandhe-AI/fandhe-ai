# tape・中間バッファのホスト arena 再利用（opt-in・既定 OFF）

イシュー #2104（Phase 3 親 #2099。前提設計イシュー #2103 の設計 doc が未着のため、実装に要る最小限の設計判断を本 doc に記録する。#2103 の設計 doc が main に入った場合は本 doc をそちらへ寄せる）。

状態: **実装済み・既定 OFF・両機体（M4 Max・GB10）未実測**。既定切替は両機体 A/B が ADOPT となった後の別 PR。

## 1. 背景

`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 のとおり、CPU の train／infer（small batch・多層 model）は演算本体の外側の固定費が負けの一因になっている。その候補が、`Tensor<f32>`（`Arc<Storage{ data: Vec<f32> }>`）の毎回の確保・解放。`bench-fandhe` は `train --mode reuse` でも step ごとに `Tape` を新しく作り、`infer`（`Sequential::predict`／`predict_resident`）も呼び出しごとに内部 Tape を作って捨てる。したがってプールの寿命が Tape 1 個分では step をまたぐ再利用が起きず、**プールは Tape より長く生きる必要がある**。

## 2. 設計判断

| 論点 | 決定 | 理由 |
|------|------|------|
| thread-local か global か | thread-local（`crates/tensor-core/src/alloc.rs`） | CPU カーネルの出力確保は呼び出しスレッドで並列区間より前に行われ、lock・contention が不要。rayon worker で確保が起きても空プールなので新規確保に落ちるだけ |
| 回収点 | 明示ハーベスト（`Drop for Tape`・`Tape::reset` の truncate 前・`Drop for Gradients`）。`Storage` 全般の Drop フックは採らない | 対象を tape 生存期間内のバッファに限定でき、ジェネリック `Storage<T>` の `f32` 特殊化（`Any` ダウンキャスト等）や新規 `unsafe` を避けられる |
| 回収条件 | `Arc` 一意所有・offset 0・contiguous・`data.len() == numel`（`Tensor::try_into_unique_vec`。`Arc::try_unwrap`。`pub(crate)`） | 参照中のメモリは絶対に再利用しない。共有 view・外へ返した tensor は素通し |
| 取り出し条件 | 要素数の完全一致のみ | `pool.rs` のバイトサイズ完全一致方針と揃え、`data.len()==numel` 契約を壊さない |
| 総量上限 | スレッドごとアイドル保持 64 MiB（`HOST_ARENA_MAX_BYTES`）。超過投入・上限超の単一バッファは即解放（追い出しなし） | REQ-14 14-3 の安全側。`pool.rs` 既定 128 MiB の半分。ガードレール閾値ではない |
| 再利用時の初期化 | `take_zeroed_f32` は必ず全要素ゼロ埋め。`take_cleared_f32` は len 0 で返し呼び出し側が全要素を書く | bit 同一と前回データ残留（情報漏えい）防止 |
| opt-in の切替 | `HOST_ARENA_DEFAULT_ENABLED: bool = false`（`pub(crate)`）＋スコープ限定上書き `override_enabled_for_scope`（`#[doc(hidden)] pub`・RAII） | facade の公開面を広げない。A/B は after 側 worktree で定数のみ `true` に反転 |
| 既存プールとの棲み分け | `pool.rs`（`MemoryOps`／`BufferHandle` 層のデバイスバッファ）・`DeviceParamStore` とは別層。本 arena はホスト `Vec<f32>` のみ | `docs/memory-pool-design.md`・`docs/device-memory-pool-design.md` と重複・干渉しない |

## 3. 既存 REJECT との非重複

`docs/cpu-matmul-fixed-cost-{design,impl}.md` の `zeroed_output`（rayon 並列ゼロ書き込み。M4 Max N=2048 で後退し本番無効）は「確保後のゼロ埋めの並列化」であり、本件の「確保そのものの再利用」とは別機構。再実行にはあたらない。`zeroed_output_with_threshold` の並列分岐の意味は変えていない（小サイズ分岐の確保のみ arena 経由）。

## 4. 差し替えた確保箇所（MLP 経路）

- `backend-cpu`: `zeroed_output`（gemm）・`gemm_bias_act`・`gemm_resident_lhs`・`mse_loss_backward`（2 箇所）・`binary_elementwise`／`unary_elementwise`／`where_cond`／`masked_fill`・`fused_elementwise` の出力
- `autodiff`: `elementwise_mul_mask`（ReLU VJP）の出力 3 箇所（`take_cleared_f32`）
- 対象外（決定）: `Tensor::contiguous` の確保（ジェネリックで `f32` 専用分岐が要る。MLP 経路での確保回数の寄与が未確認のため今回は入れない）、`batch_norm`・`softmax` 等その他カーネル、サイズクラスの切り上げ、LRU 追い出し、CUDA・Metal 側への同等機構

## 5. bit 同一の論証と検証

- 出力は `take_zeroed_f32` で全要素ゼロ埋め済み、または `take_cleared_f32` の len 0 から呼び出し側が全要素を書くため、旧確保と同じ内容になる。演算順序・並列度は変えない
- `crates/facade/tests/tape_arena_bit_identity.rs`（CI 実行・非 ignore）: 決定的シードの多層 MLP（Linear+ReLU×2・MSE・SGD）を arena OFF／ON で学習し、loss・推論出力・最終パラメータを `to_bits()` で完全一致比較。使い捨て Tape・`reset` 反復・checkpoint 併用の 3 経路。ON 側は `recycled > 0`・`hit > 0` も確認（素通し経路だけを見る偽陽性の排除）
- `crates/tensor-core/src/alloc.rs` の単体テスト: 素通し・完全一致再利用・ゼロ埋め・上限・共有／view の非回収・ガードのネスト

## 6. A/B 手順（両機体）

`docs/perf/logs/tape-arena-reuse-2104/` の `RULE.txt`（事前登録）・`README.md` を参照。`scripts/bench/framework-compare/run_ab_tape_arena_cpu.sh` を M4 Max・GB10 それぞれで実行する。

## 7. 状態

- x86_64 ホストでは実装・CI テストまで。M4 Max・GB10 の 5 round 実測は申し送り
- ADOPT 後の作業: `HOST_ARENA_DEFAULT_ENABLED` を `true` に切り替える後続 PR
