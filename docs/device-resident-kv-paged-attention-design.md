# デバイス常駐 KV（K-3）と paged attention の設計記録

イシュー #3109（親 #3108・Phase 4 #3088・ルート #3018）。基準コミットは `origin/main` の `ab14678b`。
本書は**設計の記録だけ**で、コード・依存・`docs/spec/` の変更はない。

## 0. 位置づけ・承認状況

- **本記録は推奨形の記録であり、K-3 の承認を意味しない**。`docs/kv-cache-design.md` §6 項 3 は「未取得」のままで、本書はその文言を書き換えない
- ルート #3018 の所有者コメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6111643437>、2026-10-11）の D-A 行は、対象を「3D 層化、SWA の fit 結線、save_model の kind 追加、vmap(grad)、jvp の拡張、二階微分の対象 Op 拡張、Unicode、出力モード、generate 系、ModelRegistry 書き込み」と列挙し、「推奨形で一括承認。記録に形が無い点だけ承認依頼に戻す」と回答している。**K-3 も paged attention も名指しされていない**。「generate 系」に含まれるかは解釈を要するので、本書は判断せず §9 の確認事項 1 に戻す。D-C（unsafe 系 4 件）にも K-3 は含まれない
- spec の位置づけ: `docs/spec/04-requirements.md` の REQ-9「2026-10-08 追記」（#2624）は paged attention を「条件付き」とし、移行条件を K-3 と CPU 参照の成立とする。同追記は公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張を承認していない
- **停止条項**: #3110〜#3113 は、§9 の確認事項 1 に所有者の回答（コメント URL）が付くまで着手しない。回答が「含まれる」なら本書の推奨形で進める。「含まれない」なら本書へ追記し、承認依頼に切り替える
- 他の承認は代行しない。§9 の 2〜4 も別に戻す
- 所有者の回答（コメント）: <https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6207013704>（#3110 着手時に参照。承認内容の詳細は当該コメントを正とし本書へは書き写さない）

## 1. 背景とコード事実（基準 `ab14678b` で確認）

| 事実 | 出典 | 設計への影響 |
|---|---|---|
| `TapeNode::value` はホストの `OnceCell<Tensor<f32>>` | `crates/autodiff/src/tape.rs:2730` | デバイスバッファを `TapeNode` に持たせない |
| `Tape: Send` の静的アサーションがある | `crates/autodiff/tests/fusion_backend_integration.rs:388-391` | 同上。`inference-chain-single-sync-design.md` 決定 2 と同じ制約 |
| `KvCache` はホストの `Tensor<f32>`、レイアウト `[B, S, E]`、k／v は両方 Some か両方 None | `crates/autodiff/src/nn/attention.rs:1555` | K-1 経路は変えない |
| mask 規則 (c) は `offset_allowed_mask` | `crates/autodiff/src/nn/attention.rs:1612` | paged 版も同じ意味を有効長とオフセットで表す |
| `MemoryOps` には `alloc_zeroed`・`upload`・`download`・`upload_into`・`with_host_view` しかない。デバイス内のオフセット指定コピーはない | `crates/tensor-core/src/buffer.rs:378-479` | 追記と attention には新メソッドが要る（§4） |
| `BackendOps` に `softmax` はあるが sdpa／attention はない | `crates/tensor-core/src/backend_ops.rs:2355` | 同上 |
| `Tape` の外でデバイスバッファを持つ先例が `DeviceParamStore`。`ops` は持たず `&Tape` 経由で `tape.ops()`（`pub(crate)`）を使う | `crates/autodiff/src/optim/device_store.rs:394`・`tape.rs:3475` | 所有者の形をこれに揃える |
| Metal の encode-only 経路は `*_tracked` で failure_token を登録し、`download` 後に再検査する。CUDA は同一ストリームの FIFO に頼り `_tracked` を上書きしない | `docs/inference-chain-single-sync-design.md` 決定 4・5 | §4 で同型にする |
| speculative の巻き戻しは案 (i)（`KvCache::clone()` を退避）で、`KvCache` の公開メソッドは増やしていない | `crates/autodiff/src/generate/kv_rewind.rs`、`docs/facade-speculative-decoding-batching-design.md` §17.7 | 巻き戻しは #3095／#3096 に合わせる |
| facade は `nn::kv_cache` を純再エクスポート 1 文で出し、`Tape::stateful_attention_forward` は 1 行委譲 | `crates/facade/src/nn/kv_cache.rs`、`crates/facade/src/lib.rs:1701` | §6 の公開形の先例 |

## 2. K-3 の設計（推奨 1 案）

- **所有者**: `Tape` の外に置く値型 `PagedKvCache`（仮名。autodiff の `nn` 内）。`DeviceParamStore` と同じ形で、`device`・設定・層ごとの K／V ページプール（`DeviceBuffer<f32>`）・ブロックテーブル（ホストの `Vec<u32>`）・行ごとの有効長・`poisoned: AtomicBool`・`failure_token: DispatchFailureCell` を持つ。`ops` は持たない
  - 不採用 (a) `TapeNode::value` をデバイス対応にする: `Tape: Send` と全 Op の前提を崩す
  - 不採用 (b) バックエンドの不透明ハンドルだけを持つ: 所有権と poison の検査が各バックエンドに散る
- **`TapeNode::value` との関係**: 変更しない。K-3 は推論専用で `Var` にしない。勾配が要る経路は K-1 の `forward_with_cache` に残す。入力 `Var` が `requires_grad` なら `InvalidArgument` で拒否する（黙って勾配を切らない）
- **確保・拡張・解放**
  - 構築時の設定で `max_blocks` を必須にする（0 と無制限は拒否。`SchedulerLimits` と同方針）。バイト数は `checked_mul` と `isize::MAX` で検査する
  - 拡張は空きページをブロックテーブルに足すだけで、既存ページはコピーしない（K-1 の `Var::cat` による O(T) の再確保・コピーをなくすのが目的）
  - 解放は `clear`・`Drop` でプールへ返す。巻き戻しは有効長を縮めて末尾ページを空きリストへ戻すホスト操作にする。#3095／#3096 が決まったらそれに合わせる
  - 失敗した操作は状態を変えない（原子性）
- **ホストとの同期点**: 1 ステップは「新しい k／v と q の `upload`（同期なし。Metal の `upload_into` は synchronize するので使わない）→ 追記と attention の launch／encode → 出力の `download`（ここで 1 回だけ同期）」。不変条件は「1 ステップあたり `synchronize()` 1 回」
- **CPU 参照の位置づけ**: ページを `Vec<f32>` で持つ安全な Rust の参照実装（autodiff）と、`CpuBackendOps` 側の同メソッド実装の 2 つ。新規 `unsafe` なし。内積は `f32::mul_add`（FMA 契約）。spec の移行条件「CPU 参照の成立」を満たす実体で、GPU 実装の期待値になる

## 3. paged attention の設計

- **ページのレイアウト**: 層ごとに K／V それぞれ `[num_blocks, block_size, E]`（E = H·Dh）。K-1 の `[B, S, E]` と行の並びを揃え、追記は射影出力の行をそのまま書く。head はカーネル内の添字計算で分ける。不採用: `[num_blocks, H, block_size, Dh]`（追記のたびに転置が要る。第 1 段階では採らない）
- **ブロックサイズ**: 設定値、既定 16 トークン。2 のべき乗で 1〜256。範囲外は `InvalidArgument`
- **ブロックテーブル**: `[B, max_blocks_per_seq]` の `u32` と行ごとの有効長 `seq_lens: [B]`。ホストで持ちステップごとにデバイスへ渡す。カーネルは添字が `num_blocks` 未満・位置が有効長未満であることを手動で検査する（REQ-8）
- **カーネルの入力と出力**: q `[B, L_new, E]`、K／V ページ、ブロックテーブル、`seq_lens`、`scale`、head 数、因果オフセット（K-1 の mask 規則 (a)〜(c) と同じ意味）。出力は `[B, L_new, E]`
- **縮約の順序**: 第 1 段階はトークンを論理順に走査し、max → exp の和 → 加重和の 2 パスで、ブロックをまたぐ並列分割（flash-decoding 型 split）はしない。split 版を入れる場合は「結合順序が単一の連続 K ループと異なるカーネル」にあたり、§5 の baseline 方針の対象になる
- **padding・可変長（#3104）**: 第 1 段階は現行 `KvCache` の不変条件（バッチ内の全行が同じ `S_cached`）を継ぐ。カーネルの入力は行ごとの `seq_lens` を受ける形にしておく。#3104 は左詰め padding と `KvCache` 内の行ごとの有効長を決定した（`docs/facade-speculative-decoding-batching-design.md` §21）。有効長はカーネル入力の `seq_lens` と同じ意味なので、カーネルを変えずに広げられ、mask の入力は足さない。実装は #3105
- **既存経路**: K-1 の `KvCache`・`forward_with_cache`・`generate`・`BatchScheduler` は変えない

## 4. バックエンドごとの方針と `unsafe`

| バックエンド | 確保 | 実行・同期 | 新規 `unsafe` |
|---|---|---|---|
| CPU | `CpuBackendOps` の `MemoryOps`（ホストメモリ） | 同期実行 | 不要 |
| CUDA | `CudaMemory`（既存プール経路）。`cudarc` 動的ロード・cfg ベースのまま | NVRTC カーネル。同一ストリームの FIFO に頼り `_tracked` は上書きしない（決定 5 と同型）。遅延エラーは最終 `download` で出る | 要る見込み（launch は FFI 境界）。`crates/backend-cuda` の既存 launch ラッパー内に限る |
| Metal | `MetalMemory`（`cfg(target_os = "macos")`） | MSL カーネル。encode-only なので `*_tracked` 版で failure_token を登録し、`download` 後に再検査して `poisoned` へ移す（決定 4 と同型） | 要る見込み（objc2 の dispatch とバッファ書き込み）。`crates/backend-metal` の既存 encode 経路内に限る |

- `BackendOps`／`MemoryOps` に増やす候補: ページへの追記、paged attention の forward、その Metal `_tracked` 版。既定は `Unsupported` にし 3 バックエンドで揃える
- 既定が `Unsupported` のとき K-1 経路へ暗黙にフォールバックしない。opt-in の型を選んだ利用者には `Unsupported` を返す（fail-closed）
- security-auditor の監査範囲: #3111 で増える `unsafe` ブロックすべて（launch・バッファのポインタ・境界検査の維持）。#3110・#3112 は `unsafe` を増やさないことを受け入れ条件にする。監査記録は `docs/` に残す

## 5. 判定方式

- CPU 参照と K-1 経路の比較: GEMM の形状依存ブロッキングがあるので bit 一致を前提にせず、統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比べる（`kv-cache-design.md` §3.5 と同じ）
- CUDA／Metal と CPU 参照の比較: 統一複合判定。実機テストは `#[ignore]` に分け、実測は Phase 8（#3171／#3172）に回す
- baseline 方式が要る形状は**実測前に決めない**。実機実測で fail が出た場合に限り、`ParityBaseline`／`SplitKParityBaseline` の先例に従い、実測値のみ・人間承認を経て追加する。tolerance 定数は変えない
- ブロック境界のテスト: block_size ±1・複数ブロックにまたがる長さ・空キャッシュを #3110／#3113 の対象にする

## 6. 公開形（推奨。opt-in・追加のみ）

- autodiff: `PagedKvCache`・`PagedKvCacheConfig`（block_size・max_blocks）と、`StatefulAttention` の入口（例: `forward_paged(&mut self, tape, x_new, cache: &mut PagedKvCache)`。名前は #3112 で 1 つに決める）
- facade: `fandhe_ai::nn::kv_cache` の承認形 `pub use` 1 文に `PagedKvCache`・`PagedKvCacheConfig` を足す（純再エクスポート。newtype は作らない）。facade の `Tape` に `stateful_attention_forward` と同型の 1 行委譲 `paged_attention_forward` を足す（`Tape::ops()` が `pub(crate)` で REQ-12 のため）
- 既定は変えない opt-in。既存 `KvCache` 経路は既定のままでシグネチャも変えない（`fandhe-ai =0.11.0` に対して追加のみ）
- `crates/facade/tests/api_surface.rs` の更新対象（#3112 へ引き継ぐ）: `nn_kv_cache_module_reexports_exactly_expected_surface`・`facade_reexports_kv_cache_items_only_in_approved_shape` の承認形定数、`nn_mod_public_items_match_expected_set`、委譲の正ガード（`tape_stateful_attention_forward_is_thin_delegation` と同型）、宣言場所のガード（`workspace_declares_stateful_attention_forward_only_in_facade_lib` と同型）、doctest の実在検査、残る hold doctest の glob 一覧。あわせて `docs/compat-api-scope.md` §5 の適用記録

## 7. 上限・セキュリティ（OWASP）

- A04: `max_blocks`・block_size・`B·S·E` のバイト数を `checked_mul` と `isize::MAX` で検査し、無制限の確保をしない
- A03: ブロックテーブルの添字と有効長はカーネルへ渡す前にホストで検査し、カーネル側でも手動で境界を検査する（REQ-8）
- A08: 承認を代行しない（§0）
- A06: 依存は追加しない

## 8. 後続 issue への引き継ぎ

| issue | 受け入れの要点 |
|---|---|
| #3110（CPU 参照） | 安全な Rust のみ・`unsafe` なし。K-1 との統一複合判定。ブロック境界・空キャッシュ・上限超過・原子性のテスト。期待値ユーティリティの置き場所を決める |
| #3111（CUDA／Metal） | §4 のメソッドを 3 バックエンドで揃える。Metal は `_tracked` と `download` 後の再検査。`unsafe` は既存ラッパー内に限り security-auditor の監査記録を残す。実機テストは `#[ignore]` |
| #3112（facade 公開） | §6 の追加のみ。`api_surface.rs` の更新対象を漏れなく反転・追加。`unsafe` を増やさない |
| #3113（テスト） | 判定の切り替え構成（統一複合判定が既定・baseline は実測後）。実機実測は Phase 8 へ申し送り |

## 9. 所有者へ戻す点

| # | 論点 | 記録上の状態 | 戻す理由 |
|---|---|---|---|
| 1 | K-3 が D-A の一括承認に含まれるか | `kv-cache-design.md` §6 項 3 は「未取得」 | D-A の列挙に K-3・paged attention が名指しされていない |
| 2 | `BackendOps`／`MemoryOps` へのメソッド追加（追記・paged attention・`_tracked`） | 未承認 | spec REQ-9 2026-10-08 追記は `Op`／`BackendOps` の拡張を承認しない。speculative 記録 §10 も別承認としている |
| 3 | backend-cuda／backend-metal への新規 `unsafe` | 未承認 | D-C の 4 件に K-3 が含まれない。security-auditor の個別監査は承認とは別に必須 |
| 4 | paged attention の facade 公開面の追加（`compat-api-scope.md` §5 経路 2） | 未承認 | D-A の列挙に明記がない |
| 5 | 可変長への拡張 | 決定済み | #3104 で決定（同 `facade-speculative-decoding-batching-design.md` §21）。所有者確認ではなく依存で、実装は #3105 |

## 10. 変えないもの・出典

- 変えないもの: 依存・tolerance 定数・baseline・ガードレール閾値・`docs/spec/`・既存の公開シグネチャ・REQ-8 の境界検査・FMA 契約
- 出典: `docs/kv-cache-design.md`、`docs/facade-speculative-decoding-batching-design.md`（§10 論点 7・§17.7）、`docs/inference-chain-single-sync-design.md`、`docs/device-resident-update-design.md`、`docs/device-memory-pool-design.md`、`docs/backend-metal-command-batching-design.md`、`docs/compat-api-scope.md`、§1 の各コード

## 11. #3110 実装記録（K-3 の CPU 参照）

- 配置: `crates/autodiff/src/nn/paged_kv.rs`（`#![forbid(unsafe_code)]`）・統合テスト `crates/autodiff/tests/nn_paged_kv.rs`。`nn` から `PagedKvCacheConfig`・`PagedKvCacheReference`・`paged_attention_reference`・`DEFAULT_BLOCK_SIZE` を `pub use`
- 型名: ホスト参照は `PagedKvCacheReference`。デバイス版 `PagedKvCache` の名前は #3111／#3112 に残す。ブロック管理（`BlockTable`）はバックエンド非依存で、デバイス版が再利用できる形（非公開。#3111 で可視性を調整する）
- 検査: `block_size` は 2 のべき乗で 1〜256、`max_blocks` は 1〜`u32::MAX`、K／V 合計バイト数は `checked_mul` と `isize::MAX` で検査。プールは `try_reserve_exact`、追記は先に全検証してから書き込み（失敗時は状態不変）
- 空きブロックは「返却済みスタック＋未使用連番（0 から）」で決定的。空キャッシュの `gather` は `Err`、`L_new == 0` の attention は `[B, 0, E]`
- `rewind_to` は `pub`（dead_code 回避）。facade への公開形は #3112、`KvCache` の巻き戻しは非公開 `kv_rewind` のまま
- 縮約順序: 論理順の単一連続ループ（query へ先に scale、score・加重和は `mul_add`）。統一複合判定をそのまま適用（flash-decoding 型 split は baseline 方式の対象）
- 期待値ユーティリティ: `paged_attention_reference` は autodiff の `nn` から到達できる。backend-cuda には autodiff の dev-dependency が未設定で、足すかは #3111／#3113 の判断（workspace の path 依存）
- 未実施（後続）: デバイス版・`BackendOps`／`MemoryOps` 追加・新規 `unsafe`（#3111）、Tape 入口・facade 公開（#3112）、実機 parity（#3113）、可変長（#3104）

## 12. #3111 実装記録（K-3 のデバイス版を CUDA・Metal・CPU に実装）

- 承認: 所有者が #3018 のコメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6207013704>・2026-10-16）で §9 の 1〜4 を承認済み。§0・§9 は承認前の記録として残す。**依存追加・tolerance・ガードレール閾値の変更は承認に含まれず、行っていない**。新規 `unsafe` の security-auditor 個別監査は未実施（`docs/device-resident-kv-unsafe-audit.md`）
- `BackendOps` への追加（既定は `Unsupported`。K-1 へ暗黙フォールバックしない）: `paged_kv_append`／`paged_kv_append_tracked`／`paged_attention_forward`／`paged_attention_forward_tracked`。`MemoryOps` には追加しない（確保は `alloc_zeroed`、解放は `DeviceBuffer` の RAII）。追記はスロット写像方式（`block_id · block_size + pos % block_size` の列）、attention は平坦化ブロックテーブル `[B, max_blocks_per_seq]`（未使用枠は番兵 `u32::MAX`）と `seq_lens`
- 型・検証: `tensor-core::paged_kv`（`PagedKvLayout`・`PagedAttentionArgs`・`PagedAttentionPlan`・`validate_paged_kv_append`・`validate_paged_attention`）。3 バックエンドが FFI より前に同じ関数を通す。作業バッファの上限は `1 << 30` 要素
- 配置: CPU `crates/backend-cpu/src/paged_kv.rs`、CUDA `kernels_paged_kv.rs`／`paged_kv.rs`（`context_cache::cached_paged_kv`）、Metal `shaders/paged_kv.metal`／`paged_kv.rs`／`paged_kv_model.rs`（cfg なし）、autodiff `nn/paged_kv_device.rs`（`PagedKvCache`。`BlockTable` を `pub(crate)` 化して再利用）
- 縮約順序: CUDA・Metal とも `j` 軸（系列方向）は単一の連続ループ（max・和は thread 0 が論理順に逐次）、`Dh` 軸は `fma` の逐次累積。`q · scale` を先に作る。したがって REQ-2 の baseline 方式の対象ではなく、統一複合判定をそのまま適用する。flash-decoding 型の split・warp／SIMD-group 集約・アトミックは入れていない（ソース不変条件テストで固定）
- `PagedKvCache`（1 層）: 追記は `BlockTable` を複製して確保し、書き込み成功後にだけ確定する。バックエンドの実行時エラーは poison（`Unsupported` は poison しない）。`attention` は `upload`（同期しない）→ `paged_attention_forward_tracked` → `download`（唯一の同期点）→ `failure_token` の再検査。`rewind_to`／`clear` はホストのテーブルだけを操作する
- 設計判断: backend-cuda に autodiff の dev-dependency は**足さない**（§11 が保留していた点）。CUDA・Metal の実機テストは `CpuBackendOps` を直接の参照にし、autodiff の参照との一致は `crates/backend-cpu/tests/paged_kv_backend_parity.rs`（CI）が `PagedKvCache` 経由で確認する
- 可変長: カーネル入力は行ごとの `seq_lens` を受けるが、owner は全行同長の不変条件を引き継ぐ（#3104 の結論待ち）
- テスト: tensor-core（検証関数・既定 `Unsupported`・`_tracked` 委譲）、autodiff（Mock の `BackendOps` による owner の状態遷移）、backend-cpu（参照との一致・検証エラー）、backend-cuda（ソース不変条件・launch 引数・poison／世代検査・検証拒否）、backend-metal（`paged_kv_model`。Linux でも実行）。実機の `#[ignore]` テストは CUDA・Metal とも用意した
- **未検証**: CUDA カーネルの NVRTC コンパイルと GB10 実機の一致（#3171）、MSL のコンパイルと Mac 実機の一致（#3172）。Metal の macOS 限定コードは `make check-cross-metal-tests`（aarch64-apple-darwin の型検査と clippy）までで、実行していない。`#[ignore]` テストの実機判定が fail した場合も tolerance は緩めず、baseline は実測値に基づき人間の承認を得てから #3113 で扱う
- 後続への引き継ぎ: #3112（`StatefulAttention::forward_paged` 等の Tape 入口・facade 公開・`compat-api-scope.md` §5 の適用記録）、#3113（形状マトリクスの網羅・判定方式の構成・`RULE.txt`・README の実行手順）。Metal の per-call 作業バッファ確保と attention の性能最適化は結合順序を変えうるので別判断
