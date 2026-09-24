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
承認は、本 doc 単体では**取得済みと主張しない**（§6 承認事項）。

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

## 6. 承認事項（本 doc は承認記録ではない・すべて未取得として列挙）

1. K-1 実装着手（#2084。`docs/compat-api-scope.md` §5 経路 2）
2. facade 公開面拡張（#2084 の `add_stateful_attention`／
   `StatefulAttention` 相当の 2 `pub fn`・`api_surface.rs`）
3. K-3（デバイス常駐 KV。段階 0）
4. `sdpa_compose` 置換の別 issue 起票

これまでの類似イシュー（#2065／#2068）が「ツリーを `autoMerge=true`
で起動した実行指示」を根拠に進められた前例は、**設計 doc の作成まで**
にのみ及ぶものであり、上記 1〜4 には及ばない。

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
- **facade 公開（K-2）**: 未承認のため保留。`add_stateful_attention`・
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
