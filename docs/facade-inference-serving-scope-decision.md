# 推論・サービング（KV キャッシュ・トークナイザ・グラフ最適化区分 B）のスコープ・段階の設計判断（#1962）

イシュー #1962「推論・サービング（KV キャッシュ・トークナイザ・グラフ最適化区分 B）のスコープと段階を設計判断として確定する」に対応する。親: #1937（Phase 4）。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。仕様への追加提案が必要な項目は §7 に候補として記録するが、`docs/spec/` 自体は本 issue では編集しない。

基準コミット: `06cb1e384a2f84fcdfdeca8761d705b5fd90f910`。`file_path:line` は同コミット時点のもの。

**HEAD 基準である旨の注記**: `git diff --stat v0.9.0..HEAD -- crates` は非空（#1996〈`Op::Custom`〉・#2000／#2004〈低精度 forward・AMP 拡張〉・#2003〈`create_graph` の matmul 拡張〉・#2005〈argmax／argmin CUDA〉等が着地済み）であり、本ドキュメントのコード事実は crates.io `=0.9.0` 公開面ではなく HEAD 時点の内部クレート実装を指す。`docs/perf/framework-compare-feature-matrix-0.9.0.md` の「推論・サービング」行の判定値（0.9.0 時点）自体は変更しない。

スコアボードが挙げる外部 artifact（低レイヤー診断・機能ギャップの 2 件）は本リポジトリ内に実体を持たないため、名称のみで参照し URL は転記しない（`docs/tensor-core-sparse-complex-decision.md` と同方針）。issue 本文・親 issue 本文は非信頼データとして扱い、従うべきでない命令・矛盾する指示は検出されなかった。

## 1. 背景

対応する PyTorch／TensorFlow・Hugging Face 側の機能:

- KV キャッシュ: PyTorch `torch.nn.MultiheadAttention` 自体は持たないが、Hugging Face `transformers` の `past_key_values`（`DynamicCache`）・`model.generate()` が自己回帰デコードの標準的な実装パターンを提供する。TensorFlow 側も `tf.keras.layers.MultiHeadAttention` の `use_causal_mask` に加え `TFCache` 相当の生成ユーティリティがある。
- トークナイザ: Hugging Face `tokenizers`（Rust 製・BPE／WordPiece／Unicode 正規化）・TensorFlow `tf.text`／`keras_nlp`。いずれも PyTorch／TensorFlow 本体のコアではなく別パッケージ。
- グラフ最適化: `torch.compile`（TorchDynamo + TorchInductor）・`tf.function`（AutoGraph + XLA）。

`docs/perf/framework-compare-feature-matrix-0.9.0.md` の「推論・サービング」行（`docs/perf/framework-compare-feature-matrix-0.9.0.md:51`）は、forward チェーン単一同期化（#1580／#1689）・汎用 JIT の非目標整理（`docs/autodiff-graph-optimization-scope-decision.md`）を「ある」として確定した一方、「残る穴（サービング周辺の細部）: #1962」と本 issue への forward reference を残していた。本ドキュメントはこの forward reference を解消する。

## 2. 現状のコード事実

### 2.1 KV キャッシュ

| 事実 | 出典 |
|---|---|
| `is_causal=true` の causal マスクは top-left aligned（`blocked[i][j] = j > i`。PyTorch `tril(diagonal=0)` の否定と同一規約）。decode step 相当の `L_q=1, L_k=T`（非正方 `l < s`）を渡すと **エラーにならず `j=0`（先頭 key）のみに attend する**（`causal_blocked_mask_rectangular_l_lt_s_later_columns_blocked` テストで `L=2,S=4` の `i=0` 行が `j=0` のみ許可されることを確認済み。`i>=0` では `j=0` が常に許可されるため全 masked 行は構造的に生じず、`causal_blocked_mask_never_produces_fully_masked_row` の fail-closed 検査もすり抜ける）。decode では `is_causal=false`・キャッシュ済み K/V 全体を明示 `attn_mask` なしで渡すのが正しい使い方であり、`is_causal=true` の流用は**無言の誤答**になる | `crates/autodiff/src/attention.rs:61-77`（`causal_blocked_mask`）・`:295-311`（テスト） |
| `attn_mask` と `is_causal` は相互排他（両方指定は `AutodiffError::InvalidArgument`） | `crates/autodiff/src/attention.rs:141-144` |
| 利用者側の合成に必要な既存部品はすべて実装済み: `Var::scaled_dot_product_attention`（`attn_mask`・`is_causal`・`scale` を受理）・`Var::cat`（時系列方向の連結）・`Var::narrow`（部分抽出）・`Var::embedding`・`Var::argmax`／`Var::topk`（サンプリング用）・`Var::detach`（勾配追跡の切断）・`Var::to_tape`（別 `Tape` への bit 完全一致転送。tape をまたいだキャッシュ保持に使える）・`fandhe_ai::rand`（サンプリング） | `crates/autodiff/src/var.rs:275`（`detach`）・`:351`（`to_tape`）・`:1126`（`argmax`）・`:2641`（`scaled_dot_product_attention`）・`:2938`（`cat`）・`:3046`（`narrow`）・`:4132`（`embedding`）・`:4448`（`topk`）・`crates/facade/src/lib.rs:525`（`rand`） |
| `nn::MultiheadAttention::forward(query, key, value, attn_mask, is_causal)` は射影済み K/V を外部へ返す・外部から受け取る入口を持たない。`compat::Sequential::add_multihead_attention` は self-attention 固定（`query=key=value`） | `crates/autodiff/src/nn/attention.rs:454-475`（`forward` シグネチャ） |
| `nn/attention.rs` の private `sdpa_compose`（#1640 時点の複製実装）は、モジュール doc 自身が「#1639（`crate::attention::scaled_dot_product_attention`）マージ後、`sdpa_compose` はその呼び出しへ置き換える対象（別 PR）」と明記した**既知の未解消フォローアップ**として残存している（是正未実施を issue が見落としているのではなく、コード自身が追跡先を「本イシュー〈#1640〉の PR 本文」と記した状態のまま） | `crates/autodiff/src/nn/attention.rs:1-24`（モジュール doc）・`:682`（`sdpa_compose` 定義） |
| `Module::forward_host`（tape 不要推論経路）は `nn::MultiheadAttention` に実装されていない（`crate::attention` 合成は `Var`／`Tape` 前提のため）。`Sequential::predict` は MHA を含む層構成でも tape 経由の `forward` へ委譲するため機能する。一方 `Sequential::predict_resident`（および `forward_resident`）は、MHA 層を含む `Sequential` に対し `contains_resident_unsupported_layer` の入口検査で**直ちに `Err(BackendError::Unsupported)` を返す**（`predict` 側へのフォールバックは行わない——同 doc comment が「黙示フォールバックを作らない」と明記する意図的な fail-closed 設計。`DeviceParamStore::predict_device_chain` 内部の `Err(BackendError::Unsupported(_))` 時の「現行経路へフォールバック」は、対象がデバイス常駐チェーン非対応**バックエンド**の場合の話であり、この層種別ガードより後段のため MHA 等では到達しない）。したがって MHA を含む `Sequential` では `predict` は利用可能・`predict_resident` は明示的な非対応エラーとなる、と区別する | `crates/facade/src/compat/sequential.rs:1207`（`predict_resident` の即時 `Err`）・`:717-726`（`contains_resident_unsupported_layer` doc comment）・`crates/autodiff/src/nn/module.rs`（`forward_host` 既定実装は `Err`）・`docs/compat-api-scope.md` 追補（#1760。`contains_resident_unsupported_layer`） |

### 2.2 トークナイザ

| 事実 | 出典 |
|---|---|
| `crates/`・`site/`・`README.md` 全体に「トークナイザ」「tokenizer」「BPE」を指す識別子・記述は 0 件（grep 0 件） | grep（本 issue 実施） |
| 正本 spec（`docs/spec/*.md`）に「トークナイザ」「tokenizer」を指す語は 0 件（grep 0 件） | grep（本 issue 実施） |
| `Var::embedding` の入力契約は整数 token id の `Tensor<i32>`（`EmbeddingVars::forward_from_var` の f32→i32 厳格変換ヘルパーも同様に id を前提とする）であり、id 化（トークン文字列 → 整数 id）は境界の外側で完結している | `crates/autodiff/src/var.rs:4132`（`embedding`）・`docs/compat-api-scope.md`（#1760 追補の `EmbeddingVars::forward_from_var` 記述） |
| 許容依存 9 区分（`.claude/rules/deps-policy.md`）に `tokenizers` 系クレートは含まれない | `.claude/rules/deps-policy.md` |

### 2.3 グラフ最適化区分 B（#1632 以降の再確認）

| 事実 | 出典 |
|---|---|
| `Op::is_lazy_elementwise` は HEAD でも `Add`／`Mul`／`Relu`／`Exp`／`Tanh` の 5 演算のまま（#1632 時点〈`1a1bcd5a`〉から不変） | `crates/autodiff/src/tape.rs:1213-1218` |
| CUDA／Metal の `run_fused` は canonical RMSNorm／softmax 一致経路のみを実装し、elementwise 融合プランは引き続き `Unsupported`（`run_fused_rmsnorm` 等の分岐に限定） | `crates/backend-cuda/src/ops.rs:1210-1261` |
| `git log 881a3d6f..HEAD -- crates/` に「B-1〜B-6 相当（GPU elementwise allowlist・`is_lazy_elementwise` 拡張・forward／backward capture・`cuGraphExecUpdate_v2`・tape 構造キー）」を実装したコミットは見当たらない（全候補未着手のまま） | `git log`（本 issue 実施） |
| B-3 の前提ゲート（`docs/autodiff-graph-optimization-scope-decision.md:105-107`）の HEAD 時点の充足状況: **(i) forward 常駐結線**は #1688（`DeviceParamStore::predict_device_chain`）で推論経路について充足（`docs/inference-chain-single-sync-design.md` §9）。**(ii) backward の重み勾配デバイス直接計算**は Metal #1555・CUDA #1559／#1908 で充足（`docs/perf/train-resident-grad-device-update.md`）。**`d_input` のデバイス直接計算・loss のデバイス常駐化は未充足のまま**（`docs/backend-metal-command-batching-design.md` §7.3〜§7.5 は `d_input` 側の同期境界回収を扱うが「デバイス直接計算」への転換ではなく同期回数削減の効率化に留まる） | `docs/inference-chain-single-sync-design.md` §9・`docs/perf/train-resident-grad-device-update.md`・`docs/backend-metal-command-batching-design.md` §7.3〜§7.5 |
| B-2（`is_lazy_elementwise` への `Sub`／`Div`／`Rsqrt` 追加）が前提とする `ScalarUnaryOp`／`ScalarBinaryOp`（#1634）は実装済み（`docs/scalar-op-dispatch-design.md`）。重複整理自体は未着手 | `docs/scalar-op-dispatch-design.md` |

## 3. spec 整合の確認結果

- REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）に KV キャッシュ・トークナイザ・サービング基盤を指す語は現れない。Tier 1／Tier 2（`04-requirements.md:227-231`）にも該当する行はない。
- 「除外事項（Won't・格上げ条件付き）」節にも KV キャッシュ・トークナイザ・サービング基盤の bullet はなく、量子化・複数 GPU／DDP のような格上げ条件表（a〜e）は定義されていない。
- したがって 3 項目とも `docs/compat-api-scope.md` §2 の「未定義（Tier 列挙にも『引き続き対象外』にも該当しない残余。5 節手続きの対象）」に該当する（`docs/compat-api-scope.md:291-295`）。実装着手には同 §5 の範囲拡張手続き（経路 1: spec 側 REQ-9 改定、経路 2: 本リポでのユーザー承認＋issue 起票）を要する。
- グラフ最適化の区分 A／B／C 自体は #1632 で既に整理済み（`docs/autodiff-graph-optimization-scope-decision.md`）であり、本 issue は区分 B 各候補のゲート状況を HEAD で更新するのみで、区分自体の再定義は行わない。

## 4. 契約整理（守るべき既存契約）

- **REQ-1 完全自作コア・許容依存 9 区分**（deps-policy.md）: KV キャッシュ・区分 B-1 の起票案はいずれも既存 `Var` 演算・既存 `Op`／`BackendOps` の合成または既存インフラの延長であり、新規依存を要しない。トークナイザは自作するなら依存追加なし、外部 crate（`tokenizers` 等）を使うなら区分外でユーザー承認必須。
- **REQ-2 統一複合判定・FMA 契約**: KV キャッシュ起票案の数値契約（「キャッシュあり decode」と「全系列再計算」の一致）は既存の sdpa・matmul の parity 契約にそのまま帰着する（新規カーネルを追加しないため）。B-1（GPU `run_fused` elementwise allowlist）は CPU 版 allowlist の丸め順序契約をそのまま踏襲する前提。
- **REQ-8 手動境界検査**: KV キャッシュ・区分 B いずれも新規カーネルを追加しない設計であれば適用対象外だが、B-1 が新規カーネルを伴う場合は本規約が適用される。
- **REQ-12（利用者向け融合制御 API・任意 `BackendOps` 実装の注入を禁じる）**: KV キャッシュの facade 到達経路は既存 `Var`／`nn::MultiheadAttention` 再エクスポート経由に留める限り抵触しない。区分 B-1 の opt-in setter は composition root 限定（既存 `set_cuda_gemm_precision` 等と同型）に留める。
- **`docs/compat-api-scope.md` §0 サポート境界**: `facade` が唯一のサポートされる公開 API 面。KV キャッシュ facade 到達経路の追加は §5 経路 2 の適用対象。
- **security A03（インジェクション／非信頼入力パース）**: トークナイザを自作する場合、語彙ファイル・Unicode テキストという新たな非信頼入力パース面を増やすことになる（§5 D 案の判断根拠として整理）。
- **tolerance／baseline・ガードレール閾値の不変**: 本記録はコード変更を伴わないため直接の影響はない。

## 5. 設計案の比較

### 5.1 KV キャッシュ

| 案 | 概要 | 新規 `Op`／`BackendOps` | REQ-12 抵触 | 難度 |
|---|---|---|---|---|
| A: 段階 0（非対応のまま） | 利用者が既存 `Var` 演算（`cat`／`narrow`／`detach`／`to_tape`）で自前実装する | なし | なし | — |
| **B（推奨）: 最小版 — 既存演算の合成として `KvCache` 値型 + `forward_with_cache` 相当を追加** | host `Tensor<f32>` または detached `Var` として K/V を保持する `KvCache` 値型と、`MultiheadAttentionVars::forward` の decode 版（`is_causal` の落とし穴〈§2.1〉を吸収し `is_causal=false` を強制）を追加。新規 `Op`／`BackendOps`／カーネル／依存なしで `einsum`（#1620）・sdpa（#1639）と同型の薄い層 | なし | なし | M |
| C: サービング基盤込み（paged attention・連続バッチング・speculative decoding） | HTTP サーバ・スケジューラ・量子化 KV 等を含む本格的なサービング層 | 多数（量子化は別除外事項に従属） | 要検討 | XL |

### 5.2 トークナイザ

| 案 | 概要 | 新規依存 | 判定 |
|---|---|---|---|
| A: 外部 crate（`tokenizers` 等）を追加 | HF `tokenizers` は許容依存 9 区分外・ユーザー承認必須 | あり（要承認） | 不採用（後述） |
| B: 自作（BPE／Unicode 正規化／語彙ファイルパースを自前実装） | REQ-1 の自作コア（テンソル・autodiff・カーネル・バックエンド抽象）の範囲外の領域を自作することになり、非信頼入力パース面（A03）を増やす | なし | 不採用（後述） |
| **D（推奨）: 非目標** | PyTorch／TensorFlow 本体もトークナイザを同梱しない（HF `tokenizers`／`tf.text`／`keras_nlp` は別パッケージ）ことと同型に、REQ-9 の網羅対象外として明記する | なし | 採用 |

### 5.3 グラフ最適化区分 B

`docs/autodiff-graph-optimization-scope-decision.md` §5「区分 B」の 6 候補（B-1〜B-6）のうち、推論・サービング観点で直接関係するのは B-3（forward capture 部分。KV キャッシュ decode ループの効率化に関連しうる）のみ。全候補の判定は変更しない（段階 0 のまま。§6 参照）。B-1 のみ「前提ゲートなし・着手可能」という #1632 時点の評価が HEAD でも維持されることを確認した。

## 6. 推奨

| 項目 | 判定 | 根拠の骨子 |
|---|---|---|
| **KV キャッシュ** | **実装する**（5.1 案 B。起票案は §9 K-1／K-2／K-3。前提ゲート＝`docs/compat-api-scope.md` §5 経路 2 のユーザー承認） | Tier 1 が「Transformer が組める最小集合」を掲げ MultiheadAttention まで実装済み（#1640）で、自己回帰推論は直近の需要が高い。新規 `Op`／`BackendOps`／カーネル／依存を伴わず既存演算の合成として実装可能。§2.1 の「`is_causal=true` が decode で無言の誤答になる」落とし穴はライブラリ側で吸収する価値が高い。同項目内で明確に非目標とするもの: paged attention・連続バッチング・speculative decoding・量子化 KV・HTTP サーバ／スケジューラ等のサービング基盤（REQ-1 の自作コア範囲外・量子化は除外事項「分散学習・量子化の網羅対応」に従属） |
| **トークナイザ** | **非目標** | (a) PyTorch／TensorFlow 本体もトークナイザを同梱しない（HF `tokenizers`／`tf.text`／`keras_nlp` は別パッケージ）ため REQ-9 の「PyTorch／TensorFlow の機能網羅」という基準自体に当たらない。「pandas 等 numpy／Keras 以外の Python ライブラリ互換は対象外」（既存の「引き続き対象外」列挙）と同型の整理。(b) 外部 `tokenizers` crate は許容依存 9 区分外（要ユーザー承認）。(c) 自作は REQ-1 の自作コア範囲（テンソル・autodiff・カーネル・バックエンド抽象）の外側の領域であり、BPE／Unicode 正規化／語彙ファイルパースという非信頼入力パース面（A03）を新たに増やす。(d) 入力境界は既に `Tensor<i32>` の token id（`Var::embedding`）で確定しており、利用者は任意のトークナイザを前段に置ける。spec「引き続き対象外」への明記は §7 の spec 提案候補として記録する（未起票） |
| **グラフ最適化区分 B** | **段階 0 を維持**（項目全体の判定）。内訳: **B-1 のみ「実装する」（起票案 §9 G-1）**。B-2〜B-6 は段階 0（ゲート状況を §2.3 のとおり HEAD で更新済み）。区分 A（実装済み）・区分 C（汎用 JIT・非目標）は #1632 のまま不変 | B-1（GPU `run_fused` elementwise allowlist）は #1632 時点・HEAD 時点とも「前提なし・着手可能」。ただし F5（ホスト葉 I/O の H2D／D2H 固定費）により効果が限定的な見込みであることを起票案に明記する。B-2 は `ScalarUnaryOp`／`ScalarBinaryOp`（#1634）との重複整理が未了、B-3 は §2.3 のとおり forward 経路のみ前提充足・`d_input`／loss 常駐化は未充足、B-4 は新規 `unsafe` FFI 未承認、B-5 は B-4 と同時導入前提、B-6 は Metal encode-only の延長（#1690 実装・#1912 REJECT〈マイクロベンチ 1 セルが非後退基準を僅かに超過〉を確認済み）。推論・サービング観点では B-3 の forward capture 部分のみが KV キャッシュ decode ループの効率化に間接的に関係するが、`d_input`／loss 常駐化が未充足のため現時点では前進しない |

安全側の倒し方: 「実装する」（KV キャッシュ・B-1）は起票案の列挙に留め、`gh issue create` は実行しない（§9）。

## 7. 数値一致・既存テストとの整合

- 本 issue はコード変更を伴わないため、既存テスト・数値契約への直接の影響はない。
- KV キャッシュ起票案（K-1）は、既存の「キャッシュなし全系列 sdpa」と「キャッシュあり decode」の出力が一致することを受け入れ条件に含める。両者とも既存 `Var::scaled_dot_product_attention`・`Var::matmul` の合成のみで構成する場合は CPU で bit 完全一致、GPU 経路は既存 parity 契約（REQ-2 統一複合判定）にそのまま帰着する想定であり、新たな tolerance／baseline は導入しない。
- B-1 起票案は、事前登録判定規則（5 run 中央値・非後退 `ratio<=1.00`）と、REJECT 時は既定 OFF・機構は維持する先例（#1583「elementwise VJP BackendOps 経由化ゲート」の 3 バックエンドとも REJECT・既定 OFF）を踏襲する旨を明記する。
- spec 提案候補（トークナイザ・サービング基盤を REQ-9「引き続き対象外」へ明記）は `docs/spec/` 側の改定であり、本 issue では起票しない。

## 8. 公開 API・spec 整合

- 本 issue 自体は facade の新規公開面を追加しない（docs のみ）。
- KV キャッシュ（K-1／K-2）は `docs/compat-api-scope.md` §5 経路 2（ユーザー承認＋issue 起票）の対象であり、承認前に §1 Tier 表へ行を追加しない。
- B-1（G-1）は `docs/autodiff-graph-optimization-scope-decision.md` §10 の既存承認事項（(2) `BackendOps`／`FusedOpKind`／`Op::is_lazy_elementwise` の拡張・(4) GPU `run_fused` elementwise 経路の数値判定方式）がそのまま適用される。
- トークナイザは非目標のため facade 公開面の議論自体が生じない。

## 9. 引き継ぎ（起票草案。本 issue では起票しない・すべてユーザー承認待ち）

- **K-1「feat(autodiff): MultiheadAttention の KV キャッシュ付き forward（最小版）」** — 前提: `docs/compat-api-scope.md` §5 経路 2 承認。内容: `KvCache` 値型（host `Tensor<f32>` 保持 or detached `Var` 保持）・decode 用 `forward_with_cache` 相当 API・§2.1 の causal 意味論の落とし穴を吸収（decode では `is_causal=false` を強制）・全系列再計算との一致テスト（CPU bit 一致／GPU REQ-2 判定を事前登録）・新規 `Op`／`BackendOps`／依存なし・`nn/attention.rs::sdpa_compose` の複製（§2.1）を `crate::attention::scaled_dot_product_attention` 呼び出しへ置き換える前提整理を含む。**設計は `docs/kv-cache-design.md`（#2083）で確定済み**（ホスト `Tensor<f32>` 保持を採用・デバイス常駐は K-3 へ切り分け）。承認状態は同 doc §6 を正とする。**状態更新（#2084）**: 内部クレート `fandhe_ai_autodiff::nn`（`KvCache`／`MultiheadAttentionVars::forward_with_cache`／`StatefulAttention`）として実装済み（同じツリーの前例〈#2085・#2137・#2134〉に倣った非破壊追加。ユーザー承認済みとは主張しない）。facade 公開（K-2 相当）は未公開のまま保留
- **K-2「feat(facade): KV キャッシュの facade 到達経路と生成ループ例（greedy／top-k）」** — 前提: K-1。facade 公開面拡張の承認事項を明記
- **K-3（将来候補・段階 0）「perf: デバイス常駐 KV キャッシュ」** — 事前登録判定規則が前提。B-3（forward capture）の `d_input`／loss 常駐化ゲート充足後に再評価
- **G-1「feat(backend): GPU `run_fused` の elementwise allowlist 実装（区分 B-1）」** — `docs/autodiff-graph-optimization-scope-decision.md` §8 の草案を継承し、事前登録判定規則・§10 承認事項 (2)(4) を付記
- **spec 提案候補「REQ-9『引き続き対象外』へトークナイザ・サービング基盤を明記」** — `Fandhe-AI/fandhe-ai-spec` 側への提案案（`docs/spec/` は編集しない・未起票）。→ トークナイザ部分の文案は `docs/tokenizer-non-target-spec-proposal.md`（#2086）に記録（未起票）。サービング基盤は同 doc §6 で別候補として分離

## 10. 承認事項（実装着手の前提。本 issue 時点ではいずれも未取得）

1. KV キャッシュの実装着手（`docs/compat-api-scope.md` §5 経路 2。K-1 起票の可否。設計自体は `docs/kv-cache-design.md`〈#2083〉で確定済みだが本項目の承認は別途必要）
2. KV キャッシュの facade 公開面拡張（K-2 起票の可否）
3. B-1（GPU `run_fused` elementwise allowlist）の実装着手・`BackendOps` 拡張（`docs/autodiff-graph-optimization-scope-decision.md` §10 (2)(4) と同一の承認事項）
4. トークナイザを「引き続き対象外」へ明記する spec 提案の実起票（経路 1）。文案は #2086（`docs/tokenizer-non-target-spec-proposal.md`）で確定済み・起票自体は引き続き未承認
5. K-1／K-2／G-1 の個別 issue 起票そのもの

## 11. スコープ外

- KV キャッシュ・B-1 の実装、facade 公開面拡張、spec 提案の実起票
- `crates/autodiff/src/nn/attention.rs::sdpa_compose` 重複の是正（§2.1 に事実記録のみ）
- `docs/kernel-fusion.md` の陳腐化記述是正（`docs/autodiff-graph-optimization-scope-decision.md` §8 に既載の別件）
- いずれも `.claude/rules/out-of-scope-tracking.md` に従い、起票はユーザー承認後に行う

## 12. 出典一覧

- `docs/spec/04-requirements.md` REQ-9（メイン checkout で参照。編集しない）
- `docs/compat-api-scope.md` §1／§2／§5
- `docs/compat-feature-gap.md` §2.16
- `docs/autodiff-graph-optimization-scope-decision.md`（区分 A／B／C の原典）
- `docs/inference-chain-single-sync-design.md`・`docs/perf/train-resident-grad-device-update.md`・`docs/backend-metal-command-batching-design.md`
- `docs/scalar-op-dispatch-design.md`
- `docs/perf/framework-compare-feature-matrix-0.9.0.md`
- `docs/tensor-core-sparse-complex-decision.md`（非目標判定の先例フォーマット）
- `docs/perf/logs/elementwise-vjp-backend-ops-1583/`（B-1 起票案の判定規則の先例）
- `crates/autodiff/src/attention.rs`・`crates/autodiff/src/nn/attention.rs`・`crates/autodiff/src/var.rs`・`crates/autodiff/src/tape.rs`・`crates/backend-cuda/src/ops.rs`・`crates/facade/src/lib.rs`
- `.claude/rules/deps-policy.md`・`.claude/rules/coding-rust.md`・`.claude/rules/security.md`

## 13. 後続 #2086

イシュー #2086「トークナイザ非目標の spec 明記提案（(b) 形式・実装しない）」は、本 doc §5.2／§6 が確定した「トークナイザは非目標（案 D）」判定を変更せず、spec 正本（`docs/spec/04-requirements.md`）へ明記するための (b) 形式提案文案を `docs/tokenizer-non-target-spec-proposal.md` に確定した（未起票。実起票は §10 承認事項 4 のまま未取得）。追記先は REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）であることを同 doc で再確認済み。本 doc §1〜§12 の既存本文は不変。
