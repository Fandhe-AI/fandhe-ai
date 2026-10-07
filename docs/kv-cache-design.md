# KV キャッシュの設計確定（既存 Var 演算の合成のみ）

対応イシュー: #2083（親 #2059。実装先は兄弟イシュー #2084）。
位置づけ: **設計判断の記録のみ・コード変更なし**（`crates/**`・
`Cargo.toml`・`docs/spec/`・tolerance／baseline・ガードレール閾値は
一切変更しない）。基準コミット `edac45b2`（本 doc のコード事実・
`path:line` 引用はすべてこのコミットで確認済み）。

## 0. 結論

最小版 KV キャッシュは、**既存の `Var` 演算（`Var::cat`／`narrow`／
`detach`／`to_tape`・`nn/attention.rs` の `project`／`split_heads`／
`sdpa_compose`・`nn::Linear`）の合成のみ**で実装できる。新規 `Op`／
`BackendOps` メソッド／カーネル／依存は不要（REQ-1・REQ-2・REQ-12 不変）。

キャッシュはホスト `Tensor<f32>` として `Tape` の外に保持する
（`TapeNode::value` がホスト側 `OnceCell<Tensor<f32>>` である現行構造
——`tape.rs:1874`——では、`Var` 合成のままデバイス常駐バッファへ
到達できないため）。兄弟イシュー #2084 の「デバイス常駐 buffer」という
文言は本 doc の設計（ホスト保持）へ読み替え、デバイス常駐化自体は
K-3（段階 0・将来候補）へ切り分ける。

`docs/compat-api-scope.md` §5 経路 2（ユーザー承認＋issue 起票）の
承認のうち、K-1（本 autodiff 内部実装。§6 承認事項 1）は 2026-09-24 に
ユーザー承認済み。facade 公開（K-2。§6 承認事項 2）は 2026-10-07 にリポジトリ所有者本人が
承認済み（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965。§14）。K-3・`sdpa_compose` 置換
（§6 承認事項 3・4）は未取得のまま（§6 承認事項）。

## 1. 背景

- `docs/facade-inference-serving-scope-decision.md`（#1962）§6 は
  KV キャッシュを「実装する」（5.1 案 B: 既存 `Var` 演算の合成のみ・
  新規 `Op`／`BackendOps`／カーネル／依存なし）と判定し、§9 に起票案
  K-1（autodiff 最小版）・K-2（facade 到達経路）・K-3（デバイス常駐・
  段階 0）を残した。
- `docs/compat-api-scope.md` §2（329〜334 行目）は KV キャッシュを
  「未定義」残余のうち §5 経路 2 の起票案がある項目として記録し、
  「ユーザー承認前のため Tier 1／Tier 2 表への行追加は行わない」と
  明記している。本 doc の追記後もこの記録は維持する（Tier 表への
  行追加はしない）。
- 自己回帰デコードは、キャッシュなしでは各ステップで全系列を
  再計算するため計算量が O(T²) になる（§3 の計算効率指標で定量化）。
  ライブラリ側で吸収すべき落とし穴として、`causal_blocked_mask`
  （`crates/autodiff/src/attention.rs:61-77`）は top-left aligned
  （`blocked[i][j] = j > i`）であり、decode（`L=1`）で単純に
  `is_causal=true` を渡すと先頭 key のみに attend する無言の誤答に
  なる（#1962 §2.1）。この落とし穴は §2 の API 設計で構造的に
  吸収する。

## 2. API 案（未承認のまま列挙。名称は #2084 が最終決定）

- 値型 `KvCache`（内部クレート `fandhe_ai_autodiff::nn` を想定）:
  `k: Option<Tensor<f32>>`・`v: Option<Tensor<f32>>`（射影済み K/V、
  レイアウト `[B, S_cached, E]`・contiguous）、`seq_len()`・
  `batch()`・`clear()`・`is_empty()`。`Tensor<f32>` は内部
  `storage: Arc<Storage<T>>` を共有する値型（`Var::detach` doc・
  `var.rs:265-269`）のため、`clone()` は `Arc` のポインタ複製のみで
  実データコピーを伴わない。
- `MultiheadAttentionVars::forward_with_cache(&self, query_new:
  &Var<'t>, key_new: &Var<'t>, value_new: &Var<'t>, cache: &mut
  KvCache) -> Result<Var<'t>, AutodiffError>`:
  - `query_new: [B, L_new, E]`・`key_new`/`value_new: [B, L_new, E]`
    （新規トークン分のみを渡す）。
  - 手順: ①shape 検査（rank 3・`B`・`E`・cache との `B`／`E` 一致）
    → ②`project`（`crates/autodiff/src/nn/attention.rs:675`。
    `LinearVars::forward` に委譲）で q/k/v 射影 → ③cache が非空
    なら `tape.var_no_grad(&cache.k)`（`tape.rs:2174`）で葉として
    登録し `Var::cat(&[k_cached, k_new_proj], 1)`（`var.rs:2909`。
    `v` も同様）→ ④`split_heads`（`nn/attention.rs:714`）→
    ⑤attention 本体（後述の mask 規則）→ ⑥head 結合 → ⑦`out`
    射影 → ⑧`cache.k = k_cat.to_tensor()`（`var.rs:143`。`Arc`
    共有のため算術・実データコピーなしで bit 完全一致）。
  - **`detach`（`var.rs:277`）は推奨経路では呼ばない**: `var_no_grad`
    による新規葉登録で十分であり、`detach` は同一 `Tape` 上に
    冗長な葉を積むだけになる。`detach` は §5 の案 B'（同一 `Tape`
    上で `Var` を保持し続ける案）の部品としてのみ登場する。
  - **mask 規則（`is_causal` 落とし穴の構造的な吸収）**: 本 API は
    `is_causal` 引数を**持たない**（利用者が `true` を渡す経路を
    構造的に塞ぐ fail-closed）。内部規則:
    (a) cache 空かつ `L_new == S_total`（prefill）→ 内部で
        `is_causal=true` として `sdpa_compose` を呼ぶ、
    (b) `L_new == 1`（decode の通常ケース）→ mask なし・
        `is_causal=false`（全 key に attend してよいため
        `masked_fill` を 1 回省ける）、
    (c) `L_new > 1` かつ cache 非空（複数トークンの追記）→
        明示 `attn_mask: Tensor<bool> [L_new, S_total]`
        （`allowed[i][j] = j <= S_prev + i`。`true` = attend
        規約——`nn/attention.rs` モジュール doc・`attention.rs:109`
        の規約に整合）を `is_causal=false` で渡す（`attn_mask` と
        `is_causal` は `attention.rs:141-143` で相互排他が検査
        される）。
  - 任意の追加 `attn_mask`（padding 等）は最小版では受理しない
    （§7 対象外）。受理する場合は (c) の offset mask との AND 合成が
    必要になるため #2084 以降の課題とする。
  - `KvCache::from_full_sequence`（全系列 prefill 結果からの初期化）は
    上記 (a) と同じ経路のため独立 API を設けない。
- facade 到達経路（K-2。#2084 が挙げる `add_stateful_attention`・
  `StatefulAttention` 相当の 2 `pub fn`）は**本 doc では設計しない**
  （§6 承認事項に列挙するに留める）。`compat::Sequential`
  （`crates/facade/src/compat/sequential.rs`）は MultiheadAttention 層を
  self-attention 固定・`is_causal=false` で呼ぶ（1845 行目
  `vars.forward(&current, &current, &current, None, false)`）ため、
  K-2 では `Sequential` 経由ではなく独立モジュール（`nn::rnn` の
  純再エクスポート方式・#1955 と同型）を候補として記録する。

## 3. 実装戦略（既存 Var 合成のみ）

### 3.1 contract 確認表（新規 `Op`／`BackendOps`／VJP／依存の不要性）

| 必要演算 | 場所（`path:line`） | 備考 |
|---|---|---|
| `Var::cat` | `crates/autodiff/src/var.rs:2909` | `Op::Concat` ノードを記録（`push_eager`。2938〜2944 行目）。VJP は各入力へ `upstream.narrow` を zero-copy に分配（既存） |
| `Var::narrow` | `crates/autodiff/src/var.rs:3017` | view（zero-copy）。既存 VJP |
| `Var::detach` | `crates/autodiff/src/var.rs:277` | 案 B' の部品としてのみ使用（推奨経路〈案 B〉では不使用） |
| `Var::to_tape` | `crates/autodiff/src/var.rs:353` | 別 `Tape` へ渡す場合の経路（本 doc の推奨経路では `to_tensor` + `var_no_grad` を使うため必須ではない） |
| `Var::to_tensor` | `crates/autodiff/src/var.rs:143` | cache 書き戻し（`Arc` 共有のため算術なし） |
| `Tape::var_no_grad` | `crates/autodiff/src/tape.rs:2174` | cache を新規ステップの葉として再登録 |
| `nn/attention.rs::project` | `crates/autodiff/src/nn/attention.rs:675` | q/k/v 射影（既存 `nn::Linear` 合成） |
| `nn/attention.rs::split_heads` | `crates/autodiff/src/nn/attention.rs:714` | zero-copy view（`reshape` → `permute`） |
| `nn/attention.rs::sdpa_compose` | `crates/autodiff/src/nn/attention.rs:817` | attention 本体。`attn_mask`／`is_causal` は `attention.rs:141-143` で相互排他検査 |

いずれも既存 `Op` の合成に留まり、新規 `Op`／`BackendOps` メソッド／
VJP／カーネル／依存は不要である。

### 3.2 コード事実（設計を制約する現行構造）

- `TapeNode::value` はホスト `Tensor<f32>`（`OnceCell<Tensor<f32>>`。
  `crates/autodiff/src/tape.rs:1874`。`docs/inference-chain-single-sync-design.md`
  §2 表と同型の制約）であるため、キャッシュもホスト保持とする。
  GPU 経路では各演算が既存どおり H2D／D2H を伴う。
- `BackendOps::concat` の既定実装は `Unsupported`
  （`crates/tensor-core/src/backend_ops.rs:2053`）であり CPU／CUDA／
  Metal のいずれも override していない（`grep -rn "fn concat"
  crates/backend-{cpu,cuda,metal}/src` は 0 件）。そのため
  `Var::cat` は常に `eval::concat`（`crates/autodiff/src/eval.rs:1887`。
  strided view 入力可）というホスト実行にフォールバックする。
  decode の各ステップで `[B, S_total, E]` の新規バッファ確保＋
  コピー（O(B·T·E)）が層ごとに発生する。`Var`（immutable）には
  in-place 追記が存在しないため、この確保＋コピーは最小版の
  受容コストとし、事前確保リングバッファ／デバイス常駐化は K-3 へ
  切り分ける。
- `Var::cat` は `push_eager` により `Op::Concat` ノードを記録する
  （`var.rs:2938-2944`）。decode ループで `Tape` を使い回すと
  `Op::Concat` ノードが際限なく蓄積するため、ステップごとに
  新規 `Tape` を作成するか `Tape::reset`（`tape.rs:2389`。演算前に
  登録した葉のみ保持）で切り詰める運用を推奨する。cache 自体は
  ホスト `Tensor<f32>` として `Tape` の外に持つため、`reset`／
  再作成の影響を受けない。
- キャッシュレイアウトは `[B, S, E]`（`project` 出力は常に
  contiguous）を採用し、`split_heads`（zero-copy view）は cat の
  「後」に 1 回だけ適用する。`[B, H, S, Dh]` 保持案は `cat` が
  view 入力を受け付けるため技術的には可能だが、`permute` view を
  毎ステップ cat する利点がないため採らない（§5 比較表）。

### 3.3 `sdpa_compose` の扱い

`nn/attention.rs` の private 複製 `sdpa_compose`
（モジュール doc 1〜24 行目・817 行目）を
`crate::attention::scaled_dot_product_attention` へ置き換える是正は
#1962 §11 が別件として整理済みである。本 doc は「#2084 着手時の
前提整理（K-1 草案に含まれる）」として記録するに留め、**本イシューでは
触れない**。

### 3.4 `MultiheadAttention` への state 統合 sketch

`nn::Module` trait は変更しない。擬似コード:

```rust
// prefill（全系列）
let mut cache = KvCache::default();
let y0 = mha.forward_with_cache(&x_prompt, &x_prompt, &x_prompt, &mut cache)?; // 内部規則 (a): is_causal=true 相当
// decode（seq_len=1 を N 回）
for _ in 0..n_new {
    let tape = tape_for(device);                    // または tape.reset()
    let x_t = tape.var_no_grad(&embedded_token);     // [B, 1, E]
    let y_t = mha.bind(&tape).forward_with_cache(&x_t, &x_t, &x_t, &mut cache)?; // 内部規則 (b): mask なし
    // ... argmax / topk（既存 Var::argmax／topk）→ 次トークン
}
```

`TransformerEncoderLayer`（#2211 で実装済み。
`crates/autodiff/src/nn/transformer_encoder_layer.rs`）への波及は、
「内部の `self_attn` 呼び出しを `forward_with_cache` へ差し替える
decode 版 forward」を将来候補として記録するに留め、本 doc では
設計しない。

### 3.5 数値契約

「全系列再計算」と「cache あり decode の各ステップ末尾行」は、
演算の種類（射影 → cat〈ホスト・算術なし〉→ sdpa）自体は同じだが、
GEMM の形状が異なる（q 射影・`QKᵀ` の M が `T+1` 対 `1`）。CPU GEMM は
入力形状に依存するブロッキングパラメータ（`KC` の決定。
`crates/backend-cpu/src/gemm_blis/cache_params.rs:251`
`kc_theoretical.min(super::KC).clamp(KC_MIN, KC_MAX)`。
`docs/perf/logs/cpu-gemm-*` の shape 依存実測）を持つため、
**CPU の bit 完全一致は「事前登録の仮説」として記載するに留め、
shape 依存ブロッキングで不成立の場合は REQ-2 統一複合判定
（相対誤差 1e-3 未満または絶対誤差 1e-5 未満。tolerance／baseline は
不変）を #2084 の受入条件とする**（#1962 §7 も「想定」に留めている）。
(b) 経路は `masked_fill` を省くが、`-inf` fill は非 masked 要素の
値を変えないため差分要因にはならない。GPU 経路は既存 sdpa／matmul／
softmax の parity 契約へそのまま帰着し、tolerance／baseline の新設は
ない。

### 3.6 計算効率指標（解析値のみ・forward 限定。backward は対象外）

1 ステップ（cache 長 `T`・新規 1 トークン・batch `B`・埋め込み `E`・
head 数 `H`・`Dh = E/H`）の MAC 数を式で示す（実測値は #2084 の
`docs/perf/logs/kv-cache-2084/` へ申し送り、本 doc には載せない）。

- cache あり: 射影 `3·B·E²` + scores `B·H·T·Dh = B·E·T` + weights·V
  `B·E·T` + out 射影 `B·E²` ≒ **`4·B·E² + 2·B·E·T`**。加えて cat の
  コピーが層ごとに `B·T·E` 要素発生する。
- 全系列再計算（長さ `T+1`）: ≒ **`4·B·(T+1)·E² + 2·B·E·(T+1)²`**。
- `N` トークン生成の総和: cache あり `O(N·B·E² + N²·B·E)`、
  再計算 `O(N²·B·E² + N³·B·E)`。射影は `T` 倍、attention は `T` 倍
  それぞれ削減される一方、cat のコピーは attention 項と同オーダー
  （帯域律速）で残る。
- **メモリ指標**: cache 常駐量は層ごとに `2·B·T·E·4` byte（K/V・
  f32）。`Var::cat` が新規バッファを確保する構造上、各ステップで
  旧 cache と新 cache が同時に生存する瞬間があり一時的に約 2 倍
  （`Arc` の drop で解放）になる。`Tape` 側には `Op::Concat`
  ノード（値はホスト `Tensor<f32>` の `Arc` 共有）が積まれるため、
  §3.2 の `Tape::reset`／再作成の運用が前提になる。事前確保
  リングバッファによる定数メモリ化は K-3 の対象。

## 4. 契約整理（不変）

- REQ-1（許容依存 9 区分・完全自作コア）・REQ-2（統一複合判定・
  FMA 契約）・REQ-8（新規カーネルを追加しないため本 doc の範囲では
  適用対象外）・REQ-12（`BackendOps` 注入禁止）・
  `docs/compat-api-scope.md` §0 サポート境界。いずれも本 doc の
  設計はこれらを変更しない。

## 5. 設計案の比較

| 案 | 概要 | 新規 Op | REQ-12 | 難度 | `Tape` 寿命との相性 |
|---|---|---|---|---|---|
| A | 利用者が既存 `Var` 演算を自前で合成（ライブラリ側 API なし・段階 0 維持） | なし | 問題なし | 利用者負担大（`is_causal` 落とし穴を利用者が踏みうる） | 制約なし |
| **B（本 doc 推奨）** | `KvCache`（ホスト `Tensor<f32>` 保持）+ `forward_with_cache` | なし | 問題なし | 中（§3 の合成） | `Tape` 外で保持するため `reset`／再作成の影響を受けない |
| B' | `detach` した `Var` を同一 `Tape` 上で保持し続ける | なし | 問題なし | 中 | `Tape::reset`／再作成で葉が消えるため decode ループの運用制約が強い |
| C（K-3・段階 0） | デバイス常駐バッファとして KV を保持 | 要検討 | 要検討 | 高（`TapeNode::value` がホスト前提の現行構造を変更する規模） | 本 doc の対象外。K-3 として別途評価 |

案 B' は `Tape` のステップごとの再作成・切り詰め運用と相性が悪い
ため、案 B（本 doc の推奨）を採る。

## 6. 承認事項

1. **K-1 実装着手（#2084。`docs/compat-api-scope.md` §5 経路 2）**:
   2026-09-24 にユーザー承認済み。承認範囲は autodiff 内部実装
   （`crates/autodiff/src/nn/attention.rs` の `KvCache`・
   `MultiheadAttentionVars::forward_with_cache`・
   `StatefulAttention`）に限られ、facade 公開（下記 2）は含まない。
2. facade 公開面拡張（#2084 の `add_stateful_attention`／
   `StatefulAttention` 相当の 2 `pub fn`・`api_surface.rs`）: ~~未取得~~ →
   **2026-10-07 にリポジトリ所有者本人が承認済み**（§11.6 の P1〜P4 を推奨どおり。§14 参照）
3. K-3（デバイス常駐 KV。段階 0）: **未取得**
4. `sdpa_compose` 置換の別 issue 起票: **未取得**

これまでの類似イシュー（#2065／#2068）が「ツリーを `autoMerge=true`
で起動した実行指示」を根拠に進められた前例は、**設計 doc の作成まで**
にのみ及ぶものであり、上記 2〜4 には及ばない（1 は上記のとおり別途
ユーザー承認済み）。

## 7. スコープ外

- CUDA Graph 等カーネル段階の最適化・prefetch／memory layout 最適化
  （イシュー本文のスコープ外）
- paged attention／連続バッチング／speculative decoding／量子化
  KV（#1962 §6 非目標）
- padding 用の追加 `attn_mask` との合成（§2 の (c) offset mask との
  AND 合成が必要になるため #2084 以降）
- `Module::forward_host`（tape 不要経路）対応
- `compat::Sequential::predict_resident` の MultiheadAttention 対応

## 8. 出典

- `docs/facade-inference-serving-scope-decision.md`（#1962。§6・§9
  K-1／K-2／K-3・§10）
- `docs/compat-api-scope.md`（§2 329〜334 行目・§5）
- `docs/inference-chain-single-sync-design.md`（§2。`TapeNode::value`
  がホスト `Tensor<f32>` である契約）
- `docs/compat-feature-gap.md`（§2.16 推論・その他）
- `crates/autodiff/src/tape.rs`（`TapeNode::value`・`var_no_grad`・
  `reset`）
- `crates/autodiff/src/var.rs`（`cat`／`narrow`／`detach`／
  `to_tape`／`to_tensor`）
- `crates/autodiff/src/attention.rs`（`causal_blocked_mask`・
  `scaled_dot_product_attention` の `attn_mask`／`is_causal` 相互
  排他検査）
- `crates/autodiff/src/nn/attention.rs`（`project`／`split_heads`／
  `sdpa_compose`・モジュール doc）
- `crates/tensor-core/src/backend_ops.rs`（`concat` 既定
  `Unsupported`）
- `crates/autodiff/src/eval.rs`（`concat` ホスト実行）
- `crates/backend-cpu/src/gemm_blis/cache_params.rs`（`KC` の
  shape 依存ブロッキング）
- `crates/facade/src/compat/sequential.rs`（MultiheadAttention 層の
  self-attention 固定・`is_causal=false` 呼び出し）
- `crates/autodiff/src/nn/transformer_encoder_layer.rs`（#2211）

## 9. 実装記録（#2084。K-1 最小版）

§2 の API 案がほぼそのまま確定名称になった。差分・実装時の確認事項を
記録する。

- **確定名称**: `KvCache`（`crates/autodiff/src/nn/attention.rs`）・
  `MultiheadAttentionVars::forward_with_cache`・`StatefulAttention`。
  §2 の案どおり。
- **`L_new != L_new_kv` の明示拒否を追加**: §2 は `key_new`/`value_new:
  [B, L_new, E]` とだけ記していたが、実装では `query_new` の系列長
  （`l_new`）と `key_new`/`value_new` の系列長（`l_new_kv`）が異なる
  場合を `InvalidArgument` で明示的に拒否する規則を追加した。(a)/(c)
  の offset mask 規則が self-attention（`L_new == L_new_kv`）を前提と
  するため（cross-attention 用の非対称追記は最小版の対象外。§7 に
  同じ理由の記載あり）。
- **原子的な更新**: §2 の手順⑧（cache 書き戻し）は、全段（shape 検査
  → 射影 → cat → attention → head 結合 → out 射影）が成功した後に
  のみ実行する。途中でエラーになった場合、`cache` は呼び出し前の
  状態のまま変化しない（`crates/autodiff/tests/nn_kv_cache.rs` の
  エラー経路テスト群で確認）。
- **`StatefulAttention` は `Module` trait を実装しない**: `Module::
  forward` は `&self` を取るため `cache` を更新できない。`RefCell` で
  内部可変にすると「状態を持たない forward」という `Module` の前提を
  壊すため、あえて実装しない。`compat::Sequential` への結線は行わない
  （§2「facade 到達経路」の判断を踏襲）。
- **parity 結果（CPU・観測値）**: `crates/autodiff/tests/nn_kv_cache.rs`
  の「全系列再計算」対「prefill → decode」突合は `NaiveOps`
  （`common::req2_close`。REQ-2 統一複合判定）で検証し、規則 (a)
  （空 cache からの prefill）は `forward(..., None, true)` と bit
  完全一致することを確認した（同一の `sdpa_compose` 呼び出しへ帰着
  するため。§3.5 の「事前登録の仮説」のうち bit 一致が成立する経路）。
  規則 (b)/(c) を含む「全系列再計算」対「prefill+decode」の突合は
  `crates/facade/tests/kv_cache_backend_parity.rs` で CPU 本番 ops
  （`CpuBackendOps`）上でも実施し、REQ-2 統一複合判定で一致することを
  確認した（bit 完全一致は本番 ops 上では未確認——§3.5 が予告した
  「CPU GEMM の shape 依存ブロッキングパラメータにより不成立の可能性」
  はこの環境の実測では顕在化しなかったが、恒久的な bit 一致契約とは
  していない）。
- **facade 公開（K-2）**: （記録時点では）未承認のため保留。2026-10-07 の承認（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）により解除。§14 参照。`add_stateful_attention`・
  `StatefulAttention` 相当の facade `pub fn`／再エクスポートは追加して
  いない。`crates/facade/tests/api_surface.rs::
  facade_does_not_expose_kv_cache_stateful_attention` が「未公開」を
  fail-closed に固定する（§6 承認事項リストは本節追記後も「未取得」の
  まま変更しない）。
- **CUDA／Metal 実機**: 未実測。申し送りは
  `docs/perf/logs/kv-cache-2084/README.md`。
- **触れなかったもの**: K-3（デバイス常駐・リングバッファ）・
  `sdpa_compose` の `crate::attention::scaled_dot_product_attention`
  への置換・`TransformerEncoderLayer` の decode 版・padding 用
  `attn_mask` との AND 合成。いずれも §6／§7 の記載どおり対象外の
  ままとした。

## 10. facade 公開（K-2）の保留固定と承認依頼用の事前設計（#2084）

（2026-10-07 の承認〈https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965〉により、本節の保留固定は解除済み。§14 参照。以下は承認前の記録である。）
§6 承認事項 2（facade 公開面拡張）は本節追記時点で**未取得のまま**である。
本節は (a) 保留固定の多層防御構成、(b) 承認後に外すもの、(c) 承認依頼に
向けた K-2 の事前設計と承認者が判断すべき論点、の 3 つを記録する。
コード変更は否定ガードの強化のみで、K-2 の実装本体には着手していない。

### 10.1 保留固定の多層構成

既存の 1 行単位ソース走査（`facade_does_not_expose_kv_cache_stateful_
attention`）だけでは、複数行・ネストした group での `pub use`、facade
独自の `struct`／`type` 宣言、`compat::Sequential` への inherent
メソッド追加、facade 外（autodiff 等）での同名宣言の増加、という穴が
残る。前例（#2064／#2133／#2141・イシュー #2141 の bool_ops 保留）と
同型の多層防御に揃えた。

| 迂回パターン | 塞ぐ層 |
|---|---|
| 単一行／複数行／ネストした group の `pub use`・別名再エクスポート | `crates/facade/src/lib.rs::KvCacheHoldDoctestGuard`（正のプローブ doctest）＋ `facade_does_not_reexport_or_declare_kv_cache_items`（トークン方式） |
| `pub use fandhe_ai_autodiff::nn::*` 等の glob 再エクスポート | 既存の `facade_source_uses_only_modelable_structures`／`facade_pub_use_leaves_are_not_modules`（facade 全体で拒否済み）＋正のプローブ |
| facade 内の `struct`／`enum`／`type`／`trait` `KvCache`・`StatefulAttention` の独自宣言 | 正のプローブ＋`facade_does_not_reexport_or_declare_kv_cache_items` |
| `compat::Sequential` の inherent `add_stateful_attention` | 正のプローブ（inherent メソッドがトレイトメソッドより優先解決されるため型・引数不一致で失敗）＋`facade_does_not_reexport_or_declare_kv_cache_items` |
| doctest の無効化（`ignore`／`no_run`／`compile_fail` への書き換え・`# ` 隠し行・プローブの削除） | `extract_single_bare_fenced_doctest_block`（装飾なしのフェンスを 1 つだけ許す）＋`kv_cache_hold_doctest_probe_body_matches_fixed_contract`（本文の固定文言検査） |
| `pub mod` を追加したのに doctest の glob を更新し忘れる | `kv_cache_hold_doctest_globs_all_pub_modules`（glob 集合の一致検査） |

facade の外（`autodiff` 等）で同名宣言が増える経路は、workspace 全体の
名前インベントリ（`workspace_declares_kv_cache_items_only_in_autodiff_
attention`）としていったん導入したが、facade 到達可能性の保証と無関係
な private 宣言（別バックエンド・K-3 実装の内部型・関数等）まで固定して
しまい、正当な変更を不当に fail させる指摘（codex-review・PR #2252）を
受けて撤去した。正のプローブ（`__fandhe_kv_hold_probe` モジュールが
`KvCache`／`StatefulAttention`／`add_stateful_attention` を type/value
として使用）は、facade が `pub use fandhe_ai_autodiff::nn::*` のような
glob 再エクスポートで同名の別定義を巻き込んだ場合、型・引数の不一致で
コンパイル失敗するため、この迂回パターンは正のプローブ＋
`facade_does_not_reexport_or_declare_kv_cache_items` の既存 2 層で
引き続き塞がれている。

### 10.2 承認後に外すもの・置き換えるもの

K-2 の承認を得た日が来たら、次を同時に行う（他の保留系〈#2133 等〉と
同じ手順）:

- `crates/facade/src/lib.rs::KvCacheHoldDoctestGuard`（doctest 足場）を
  削除する。
- `facade_does_not_expose_kv_cache_stateful_attention`・
  `facade_does_not_reexport_or_declare_kv_cache_items`（自己テスト含む）
  を削除するか、正ガード（実装した公開面が到達可能であることを検査する
  テスト）へ置き換える。

### 10.3 承認依頼に向けた K-2 事前設計と、承認者が判断すべき論点

イシュー #2084 の文面は「`add_stateful_attention`・`StatefulAttention`
相当の 2 `pub fn`」と素朴に書かれているが、実装（§9）を踏まえると
そのままでは完結せず、承認範囲を広げる判断が必要になる。

**(a) `MultiheadAttention` 自体が facade から未到達**: `StatefulAttention::
new(mha: MultiheadAttention)`（§2 の API 案どおり）だが、`MultiheadAttention`
は facade ルートから再エクスポートされていない。到達経路は
`compat::Sequential::add_multihead_attention`（`crates/facade/src/
compat/sequential.rs:469`）経由の内部保持のみで、呼び出し側が
`MultiheadAttention` 値を直接取り出す手段がない
（`crates/facade/src/compat/sequential.rs:113` の `use`
〈非公開 import〉により `MultiheadAttention` 型自体はクレート内から
参照できるが、facade 外の呼び出し側からは到達できない）。したがって
K-2 は「2 `pub fn` を足すだけ」では
完結せず、`MultiheadAttention` 自体の公開（コンストラクタ・型）か、
次元・ヘッド数を直接取る `StatefulAttention` 用の別コンストラクタが
必要になる。いずれを選ぶにせよ承認範囲は #2084 の文面より広がる。

**(b) `Sequential::add_*` は §9 の判断と矛盾する**: §9「実装記録」は
「`StatefulAttention` は `Module` trait を実装しない（`&self` の
forward ではキャッシュを更新できないため）」「`compat::Sequential` へ
の結線は行わない」と明記した。`add_stateful_attention` を
`compat::Sequential` へのメソッドとして追加する形は、この判断と
正面から矛盾する。承認候補は `compat::Sequential` へのメソッドでは
なく、`crates/facade/src/nn/` 配下の純再エクスポートのサブモジュール
（`nn::rnn`〈#1955〉と同じ形。`KvCache`・`StatefulAttention` 型と、
`StatefulAttention::new`／`forward_with_cache` 相当の自由関数または
inherent メソッドを再エクスポートするだけの薄い層）。イシュー文面の
「2 `pub fn`」表現は設計 doc（本 doc）より前に書かれたものであり、
形の確定は承認時にあわせて行う必要がある。

**(c) K-3（デバイス常駐）は別承認のまま**: `TapeNode::value` がホスト
`Tensor<f32>` である現行構造（§0・§3.2）を変える規模の変更であり、
K-2 とは独立に別承認が必要（§6 承認事項 3 のまま変更なし）。

本節は記録のみであり、Issue 起票・spec 提案の投稿は行わない。

## 11. #2578（facade 公開形の確定）の着手時判定と承認依頼

### 11.1 経緯

イシュー #2578 は、ルート #2499 の一括承認（2026-10-04）の下で、本 doc §2・§6・§10 の
**推奨形**により K-2（facade 公開）の形を確定することを求めた。一括承認が及ぶのは
本 doc に書かれた推奨形だけであり、推奨形が無い論点・複数案併記のままの論点が
残る場合は、実装せずに記録追記と承認依頼へ切り替える（イシューの停止条項）。
基準コミット `a5cba8a7`（origin/main）で突合した結果、後述 11.3 の論点が残っていたため
**facade・autodiff のコードは変更していない**。**本節は承認の取得を意味しない**
（§6 承認事項 2 は本節追記時点で未取得。2026-10-07 の承認〈https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965〉により解除。§14 参照）。

### 11.2 確定済みの形（内部 API。§2・§9 と実装の突合結果）

| 項目 | 形 | 出典 |
|---|---|---|
| `KvCache` | `new`・`is_empty`・`seq_len`・`batch() -> Option<usize>`・`embed_dim() -> Option<usize>`・`clear`・`k()/v() -> Option<&Tensor<f32>>`。外から Tensor を注入するセッターは無い | `crates/autodiff/src/nn/attention.rs:1554` |
| `forward_with_cache` | `MultiheadAttentionVars::forward_with_cache(&self, query_new, key_new, value_new, cache: &mut KvCache) -> Result<Var<'t>, AutodiffError>`。`is_causal` 引数なし。更新は原子的 | 同 `:1681` |
| `StatefulAttention` | `new(mha: MultiheadAttention)`・`forward(&mut self, tape: &'t autodiff::Tape, x_new: &Var<'t>)`・`reset_cache`・`cache`・`seq_len`・`mha`。`Module` 非実装 | 同 `:1859`・`:1866` |
| エラー型 | 既存の `#[non_exhaustive] AutodiffError`（variant 追加なし）。facade は既に再エクスポート済み | `crates/facade/src/lib.rs:207` |
| `Sequential` | 結線しない（§9）。`add_stateful_attention` 形は不採用 | §9・§10.3 (b) |

### 11.3 停止根拠（未決の論点）

1. §2 は facade 到達経路を設計しないと明記し、§10.3 は自らを「承認依頼用の事前設計」とし
   形の確定を承認時に行うと書く。確定形の宣言が本 doc に存在しない。
2. §10.3 (a) の `MultiheadAttention` 到達経路は 2 案併記のまま（型とコンストラクタの公開、
   または次元・ヘッド数を直接受ける別コンストラクタ）で、どちらを推すか未記載。
   `crates/facade/src/nn/mod.rs` は MHA に `Sequential::add_*` 経由でのみ到達すると契約しており、
   型を公開するなら例外の明文化が要る。
3. §10.3 (b) の配置は候補止まり（サブモジュール名・再エクスポート集合・
   `MultiheadAttentionConfig` を出すか）。
4. **本 doc に無かった新論点**: facade の `Tape` は `pub struct Tape(pub(crate) fandhe_ai_autodiff::Tape)`
   （`crates/facade/src/lib.rs:329`）で、利用者は生 `Tape` を取り出せない（REQ-12）。
   `StatefulAttention::forward`・`MultiheadAttention::bind` は生 `Tape` を取るため、
   純再エクスポートだけでは facade から forward を呼べず、`Tape::rnn_forward_seq` と同型の
   委譲メソッドが要る。`MultiheadAttentionVars` も facade では未再エクスポート
   （`lib.rs:207` は `nn::LinearVars` のみ）。
5. 派生論点: `StatefulAttention::new(mha)`／`mha()` は、MHA 型を公開しない案では
   facade から名指しできない型を扱う公開メソッドとして残る。

### 11.4 論点ごとの推奨案（2026-10-07 に P1〜P4 すべて推奨どおりで承認済み。§14 参照）

| 論点 | 推奨案 | 比較した他案 |
|---|---|---|
| P1 配置 | 純再エクスポートの `pub mod fandhe_ai::nn::kv_cache`（新ファイル `crates/facade/src/nn/kv_cache.rs`。`nn::rnn` と同型）。`compat::Sequential` へのメソッドは追加しない | `compat::Sequential::add_stateful_attention`（§9 と矛盾するため不採用） |
| P2 MHA 到達 | `MultiheadAttention` は公開しない。autodiff に `StatefulAttention::from_config(&MultiheadAttentionConfig, seed)` 相当の追加コンストラクタを置く。このコンストラクタは `forward_with_cache` が `InvalidArgument` で拒否する 3 条件（`batch_first=false`・`kdim != embed_dim`・`vdim != embed_dim`）を構築時に検証し、同じく `InvalidArgument` で拒否する契約とする（`MultiheadAttentionConfig` はこれらの設定を許すため、構築後の forward で初めて失敗する経路を残さない）。facade は `KvCache`・`StatefulAttention`・`MultiheadAttentionConfig` を再エクスポート。理由: `nn/mod.rs` の契約を例外化せず、`bind(&Tape)`・`q_proj()`・`from_parameters`・`Module` 実装等の広い面を 0.10.x の非破壊契約で凍結せずに済む | MHA 型とコンストラクタを公開し Transformer と同じ例外として明文化する（承認範囲が広い）。残課題: `new(mha)`／`mha()` が facade から呼べない公開メソッドとして残る点は承認者が判断する |
| P3 forward 入口 | facade `Tape` に `stateful_attention_forward<'t>(&'t self, sa: &mut nn::kv_cache::StatefulAttention, x_new: &Var<'t>) -> Result<Var<'t>, AutodiffError>`（`&self.0` を渡すだけの薄い委譲）。`forward_with_cache` は出さない | `MultiheadAttentionVars` を再エクスポートして直接公開（Tape 問題が残る） |
| P4 ガード | 承認後（#2579）に §10.2 の否定ガード群を、公開面の到達可能性を検査する正ガードへ置換し、prefill → decode の doctest を 1 つ置く。今回は実施しない | なし |

### 11.5 `fandhe-ai =0.10.0` 非破壊の確認

| 論点 | 確認結果 |
|---|---|
| P1 | `pub mod` の新設は追加のみ。既存の `pub mod`／`pub use` は不変。同名項目の glob 衝突は無い見込み（#2579 で再確認） |
| P2 | `StatefulAttention` への inherent コンストラクタ追加と `MultiheadAttentionConfig` の再エクスポートは追加のみ |
| P3 | facade `Tape` への inherent メソッド追加は追加のみ。利用者トレイトの同名メソッドとの解決順の変化は Rust の minor 変更の通常範囲として受容する |
| P4 | ガードはテスト資産で `add_stateful_attention` もプローブ内の名前にすぎず、公開 API ではない。外しても破壊的変更にならない |
| エラー型 | 既存 `AutodiffError` を使い variant は追加しない。既存型には触れない |

### 11.6 ユーザーに決めてほしい事項

- A: P1〜P4 の推奨案で承認し、#2579 で実装する
- B: P2 を MHA 公開案に替える（`nn/mod.rs` の例外を明文化する）
- C: K-2 を保留のままにする

承認コメントが形を名指しするまで #2579 は着手しない。

> 2026-10-07 の承認（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）により解除。リポジトリ所有者本人が A（P1〜P4 すべて推奨どおり）を承認した。§14 参照。

### 11.7 本 PR で行わないこと

facade／autodiff のコード変更、ガードの削除・反転、`compat-api-scope.md` への適用記録、
Issue 起票、spec 提案の投稿。K-3 と `sdpa_compose` の置換は §6 承認事項 3・4 のまま。

## 12. #2579（facade 公開の実装）の着手時判定

### 12.1 経緯と判定

#2579 は §2・§6・§10 の推奨形による facade 公開（K-2）を求めたが、基準コミット `ca65c4b0` で突合した結果、
§11.4 の P1〜P4 は**未承認の推奨案**であり、§11.6 の A／B／C に対する承認コメントが存在しなかった。
P2（`MultiheadAttention` の到達経路）は B 案との択一も残る。このため停止条項に従い実装せず、
facade・autodiff のコードは変更していない。§6 承認事項 2 は当時**未取得**だった（2026-10-07 の承認〈https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965〉により解除。§14 参照）。

### 12.2 承認コメントの確認範囲

2026-10-05 に #2579・#2578・#2577・#2542・#2499 のコメントを確認し、A／B／C を名指しするコメントは 0 件だった。
ルート #2499 の一括承認は記録が公開形を決めている項目に限られ、§11 の推奨案は対象外である。

### 12.3 承認後の責務分担

§11.4 P4 は否定ガード群の正ガード置換を #2579 としているが、兄弟 #2580 の受入条件にもガード反転・
`compat-api-scope.md` §5 適用記録・実装記録が含まれる。承認後は #2579 を公開面の実装、#2580 を記録更新とする。
ただし公開面を追加した時点で既存の否定ガード（`crates/facade/src/lib.rs:2057` の `KvCacheHoldDoctestGuard`・
`crates/facade/tests/api_surface.rs` の保留系テスト）が失敗するため、同一 PR での最小限のガード差し替えが
必要になりうる。統合可否は承認後の実行時に判断する。

### 12.4 後続への影響

K-2 の実装は §11.6 の承認取得後（2026-10-07 に取得済み。§14 参照）に再着手が必要で、#2580 も同じ理由で停止対象となる。
generate() 公開（`facade-generate-decision.md`）も KvCache の facade 到達を前提としている。

### 12.5 本記録で行わないこと

コード変更、ガードの削除・反転、`compat-api-scope.md` への適用記録、Issue 起票・コメント投稿、spec 提案。
K-3 と `sdpa_compose` の置換は §6 承認事項 3・4 のまま。

## 13. #2580（保留ガード反転・記録更新）の着手時判定

本節は docs のみの停止記録であり、**承認を得たことを意味しない**。§6 承認事項 2 は記録時点で未取得だった（2026-10-07 の承認〈https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965〉により解除。§14 参照）。

### 13.1 判定

- 基準コミット `55646fac`（origin/main）・確認日 2026-10-05
- 依存 #2579 は PR #2759 でクローズ済みだが、中身は §12 の停止記録のみで、facade に公開物はない。`crates/facade/src` で
  `KvCache`／`StatefulAttention` がヒットするのは `lib.rs` の `KvCacheHoldDoctestGuard` の doc ブロック
  （`crates/facade/src/lib.rs:1979-2057`）と他ガードからの参照コメントだけである
- §11.6 の A／B／C を名指しする承認コメントは 0 件だった（確認対象: #2580・#2578 を直接確認。#2579・#2577・#2542・#2499 は §12.2 の確認結果を引用）
- 正ガードは「承認した公開形だけを許す」検査であり、公開物がない状態では反転先が存在しない

### 13.2 現状維持するもの

`KvCacheHoldDoctestGuard`（`crates/facade/src/lib.rs:2057`）と `crates/facade/tests/api_surface.rs` の保留系 6 項目
（`facade_does_not_expose_kv_cache_stateful_attention`・`kv_cache_hold_doctest_globs_all_pub_modules`・
`kv_cache_hold_doctest_probe_body_matches_fixed_contract`・`scan_kv_cache_reexports_and_declarations`・
`facade_does_not_reexport_or_declare_kv_cache_items`・`facade_does_not_reexport_or_declare_kv_cache_items_detects_each_category`）は
撤去・縮小・反転しない（2026-10-07 の承認により解除。§14 参照）。

### 13.3 受入条件ごとの扱い

| 受入条件 | 扱い |
|---|---|
| ガードの正ガード反転 | 公開物がないため不可 |
| `compat-api-scope.md` §5 適用記録、本 doc §2・§6・§10 の実装記録 | 公開が実施されていないため書かない（書くと事実と異なる） |
| facade 経由の利用例（doctest／tests） | 対象 API が未公開のため追加不可 |

### 13.4 解除の順序

1. §11.6 で A／B／C のどれを選ぶかのユーザー承認（2026-10-07 に A で取得済み。§14 参照）
2. #2579 の再着手（または再起票）で facade 公開を実装する。§12.3 のとおり、公開面追加と同時に既存否定ガードが落ちるため、同一 PR での最小限の差し替えが必要になりうる
3. 本イシュー相当の作業で §10.2 の手順に従い否定ガードを正ガード（公開面の到達可能性検査。prefill → decode の doctest を 1 つ含む。§11.4 P4）へ置換し、`compat-api-scope.md` §5 と本 doc §2・§6・§10 へ実装記録を書く
4. generate() 公開（`facade-generate-decision.md` §15）がこれに続く

### 13.5 本記録で行わないこと

コード変更、ガードの削除・反転、`compat-api-scope.md` への適用記録、Issue 起票・コメント投稿、spec 提案、依存追加、`unsafe`、tolerance 変更。
K-3 と `sdpa_compose` の置換は §6 承認事項 3・4 のまま。

## 14. #2579 実装記録（facade 公開）と承認記録

### 14.0 承認記録

- 承認者・日時: リポジトリ所有者本人（GitHub アカウント aLiz-Nancy）、2026-10-07（コメント作成 08:12 UTC）
- 根拠コメント: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965（ルート issue #2499 への承認コメント。該当行「#2577 | `docs/kv-cache-design.md` §11 の推奨案〈§11.6 の P1〜P4 はすべて推奨どおり〉」）
- 同コメントは「各記録にある『ユーザー承認まで着手しない』『公開と保留ガードの反転を停止する』という条件は、本コメントをもって満たされたものとします」と明記しており、§11.6・§12・§13 の停止条件はこれにより解除された。
- 前提条件: `fandhe-ai =0.10.0` の公開 API を壊さない追加のみ。依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更しない。承認は記録に書かれた推奨形に限る（記録に形が書かれていない点は実装せず承認依頼へ戻す）。
- §11.6 の選択: **A（P1〜P4 の推奨案で承認し、#2579 で実装する）**。B（MHA 公開案）・C（保留）は選ばれていない。承認された各項目（§11.4 の推奨案）:
  - **P1 配置**: 純再エクスポートの `pub mod fandhe_ai::nn::kv_cache`（新ファイル `crates/facade/src/nn/kv_cache.rs`。`nn::rnn` と同型）。`compat::Sequential` へのメソッドは追加しない。
  - **P2 MHA 到達**: `MultiheadAttention` は公開しない。autodiff に `StatefulAttention::from_config(&MultiheadAttentionConfig, seed)` を追加し、`forward_with_cache` が拒否する 3 条件（`batch_first=false`・`kdim != embed_dim`・`vdim != embed_dim`）を構築時に `InvalidArgument` で拒否する。facade は `KvCache`・`StatefulAttention`・`MultiheadAttentionConfig` を再エクスポートする。
  - **P3 forward 入口**: facade `Tape` に `stateful_attention_forward<'t>(&'t self, sa: &mut nn::kv_cache::StatefulAttention, x_new: &Var<'t>) -> Result<Var<'t>, AutodiffError>`（`&self.0` を渡すだけの薄い委譲）。`forward_with_cache` は出さない。
  - **P4 ガード**: §10.2 の否定ガード群を、公開面の到達可能性を検査する正ガードへ置換し、prefill → decode の doctest を 1 つ置く。
- これ以前の §11〜§13 の記録（Claude による記録コメントを含む）は承認の根拠ではない。承認の根拠は上記のユーザー本人のコメントのみである。

### 14.0.1 実装の概要

§11.4 の P1〜P4 に沿って実装した。

### 14.1 公開した名前

| 項目 | 内容 |
|---|---|
| `fandhe_ai::nn::kv_cache`（`crates/facade/src/nn/kv_cache.rs`） | `KvCache`・`StatefulAttention`・`MultiheadAttentionConfig` の純再エクスポート 1 文。`compat::Sequential` にメソッドは追加していない |
| `StatefulAttention::from_config(&MultiheadAttentionConfig, seed)`（autodiff） | 非対応設定（`batch_first=false`・`kdim`／`vdim != embed_dim`）を構築時に `InvalidArgument` で拒否し、`MultiheadAttention::from_config` へ委譲する |
| `Tape::stateful_attention_forward`（`crates/facade/src/lib.rs`） | `sa.forward(&self.0, x_new)` の 1 行委譲 |

新規 `Op`・`BackendOps`・カーネル・依存・`unsafe` は無く、数値経路は K-1 のままである。公開しないもの: `MultiheadAttention`・`MultiheadAttentionVars`・`forward_with_cache`。

### 14.2 ガードの差し替え

- 削除: `KvCacheHoldDoctestGuard` と、`api_surface.rs` の保留系 6 項目（§13.2）および `KV_CACHE_HOLD_PROBE_BODY`
- 追加（正ガード）: `facade_reexports_kv_cache_items_only_in_approved_shape`（承認形 1 文への完全一致。自己テスト付き）・`tape_stateful_attention_forward_is_thin_delegation`・`workspace_declares_stateful_attention_forward_only_in_facade_lib`・`kv_cache_is_reachable_via_facade_only`
- 期待集合の拡張: `nn_mod_declares_only_approved_submodules`（旧名 `..._init_and_rnn_...`）・`nn_mod_public_items_match_expected_set`・`GRAD_SCALER_PROBE_MODULES`。残る全 hold doctest の glob 一覧へ `nn::kv_cache::*` を追加した
- 判断点: `facade_reexports_multihead_attention_config_only_from_compat`（#2530 の正ガード）は `MultiheadAttentionConfig` の公開を `compat/mod.rs` の 1 件に固定しており、P2 の再エクスポートと衝突した。期待集合を承認済みの 2 件（`compat/mod.rs` の単独形と `nn/kv_cache.rs` の 3 名グループ形）へ拡張し、完全一致検査は維持した。公開面 inventory の更新であり、tolerance・閾値の緩和ではない
- 変更していないもの: `generate()` の保留ガード（`GenerateHoldDoctestGuard` 等。#2575 系の別スコープ）

### 14.3 テスト

- autodiff: `from_config` の単体テスト 3 件（`attention.rs`）
- facade: `tests/kv_cache_facade.rs`（autodiff 直経路との一致・全系列 prefill との一致・状態推移と reset・エラー経路の原子性と `TapeMismatch`・拒否条件・backward）。判定は既存 `assert_parity` で定数は不変
- doctest: `nn::kv_cache` モジュール doc に prefill → decode を 1 つ置いた

### 14.4 #2580 へ残すもの

`compat-api-scope.md` §5 の適用記録、本 doc §2・§6・§10 の実装記録、`docs/README.md`・`docs/compat-feature-gap.md` の更新。CUDA／Metal の実機 parity は未実測で、申し送り先は `docs/perf/logs/kv-cache-2084/README.md`。
