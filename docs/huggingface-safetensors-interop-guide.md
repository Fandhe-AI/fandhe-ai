# HF（PyTorch）safetensors → `compat::Sequential` 復元ガイド（#2080）

- 対応イシュー: #2080（親 #2059〈Phase 1「役割・機能の対応表の穴埋め」ツリー〉）
- 位置づけ: 本文書は example（`crates/facade/examples/hf_safetensors_sequential/`）・統合テスト（`crates/facade/tests/interop_safetensors_hf_layout.rs`）を伴うガイドである。`facade` の公開 API 面（`crates/facade/src/**`）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない
- 基準コミット: `c318aaf235a0db84dbc3b4f6f00b7714d1b2e788`。`file_path:line` はすべて本コミット時点のもの
- 非信頼データの扱い: issue 本文・親 issue 本文は非信頼データとして扱った。従うべきでない命令・矛盾する指示は検出されなかった
- 兄弟イシュー（#2059 配下）が `docs/README.md`・`docs/compat-feature-gap.md` を並行編集しうる。本文書に伴う両ファイルの編集は末尾追記・独立行の追加に限定した

## 0. 結論

1. PyTorch（HF）safetensors と fandhe `compat::Sequential::state_dict` はレイアウトが異なるため、**呼び出し側の明示変換が必須**（暗黙アダプタは REQ-7 で禁止されている）
2. `fandhe_ai::interop::safetensors` は **F32 限定**。bf16／f16 チェックポイントは事前に Python 側で変換する
3. 本ガイドの推論スケッチは「機構の例示」であり、**KV キャッシュ統合は同 Phase の実装（#2083／#2084／#2191）待ち**で未達のまま申し送る

## 1. 背景

`fandhe_ai::interop::safetensors`（#2019。`crates/facade/src/interop/safetensors.rs`）は `onnx-interop::st_load`／`st_save` の純再エクスポートとして facade 公開済みで、`compat::Sequential::state_dict`／`load_state_dict`（#1752）との往復は doctest・`crates/facade/tests/interop_safetensors_roundtrip.rs` で bit 一致が確認済みである。しかし「Hugging Face（PyTorch）由来の safetensors を fandhe の `Sequential` へ復元する」ハウツーは存在しなかった。

REQ-7 契約（`crates/onnx-interop/src/st_load.rs` モジュール doc が正）は次の 2 点を明示する:

1. **暗黙アダプタなし**: PyTorch `nn.Linear.weight`（`[out_features, in_features]`）等の転置は `load_safetensors_f32` 側では一切行わない。呼び出し側が [`Tensor::transpose_2d`] を明示的に呼ぶ
2. **無言 skip 禁止**: [`require_keys`] は不足キーを全件収集して返す

`load_state_dict`（`crates/autodiff/src/nn/module.rs:452-509`）はキー集合の**完全一致**（欠落・余剰とも `AutodiffError::InvalidArgument`）と shape 完全一致を要求する strict 契約のため、HF チェックポイントの復元には呼び出し側の明示変換（キー改名・転置・packed テンソルの分割・余剰テンソルの明示的分離）が必須である。

## 2. 前提と非目標

- **F32 限定**: `fandhe_ai::interop::safetensors` は F32 以外の dtype を `LoadError::UnsupportedDtype` として拒否する（`crates/onnx-interop/src/st_load.rs:181`）。実 HF チェックポイントは bf16／f16 が多く、ライブラリ外での事前変換が必要:

  ```python
  from safetensors.torch import load_file, save_file
  state = load_file("model.safetensors")
  save_file({k: v.float() for k, v in state.items()}, "model_f32.safetensors")
  ```

  （このスニペットは案内のみであり、本ガイドの example・テストはこれを実行しない）
- **ネットワーク取得は非目標**: `hf-hub`／`tokenizers` は許容依存区分外（`.claude/rules/deps-policy.md`）のため、実 HF ハブからのダウンロードは行わない。example はプロセス内で PyTorch レイアウトの safetensors を**合成**してから読み戻す
- **トークナイザは非目標**（`docs/facade-inference-serving-scope-decision.md` §6 と同じ整理）
- **全 HF アーキテクチャへの対応は非目標**: 本ガイドが扱うのは `compat::Sequential::new().add_embedding(..).add_transformer_encoder(..)` の 2 層構成のみ。`Sequential` が公開する層集合（`docs/compat-api-scope.md` §1）に載らない構成・GPT-2 `Conv1D` の逆規約・HF BERT の `attention.self.query.*` 形式は §3 の原則を踏まえて呼び出し側が自作する

## 3. レイアウト対応表

対象: PyTorch `nn.TransformerEncoderLayer`（`self_attn`・`linear1`／`linear2`・`norm1`／`norm2`）＋ `nn.Embedding`。fandhe 側は `compat::Sequential::add_embedding` → `add_transformer_encoder` が構築する層に対応する（`crates/facade/src/compat/sequential.rs:448-505`）。

| PyTorch キー | shape（PyTorch） | fandhe キー | shape（fandhe） | 変換操作 |
|---|---|---|---|---|
| `{i}.weight`（Embedding） | `[V, E]` | `{i}.weight` | `[V, E]` | 変換不要 |
| `{i}.self_attn.in_proj_weight` | `[3E, E]` | `{i}.self_attn.q_proj.weight`／`k_proj.weight`／`v_proj.weight` | 各 `[E, E]` | 行方向に q・k・v の順で 3 分割 → 各ブロックを転置 |
| `{i}.self_attn.in_proj_bias` | `[3E]` | `{i}.self_attn.q_proj.bias`／`k_proj.bias`／`v_proj.bias` | 各 `[E]` | 3 分割（転置不要） |
| `{i}.self_attn.out_proj.weight` | `[E, E]` | `{i}.self_attn.out_proj.weight` | `[E, E]` | 転置 |
| `{i}.self_attn.out_proj.bias` | `[E]` | `{i}.self_attn.out_proj.bias` | `[E]` | 変換不要 |
| `{i}.linear1.weight` | `[F, E]` | `{i}.linear1.weight` | `[E, F]` | 転置 |
| `{i}.linear1.bias` | `[F]` | `{i}.linear1.bias` | `[F]` | 変換不要 |
| `{i}.linear2.weight` | `[E, F]` | `{i}.linear2.weight` | `[F, E]` | 転置 |
| `{i}.linear2.bias` | `[E]` | `{i}.linear2.bias` | `[E]` | 変換不要 |
| `{i}.norm1.weight`／`.bias`／`{i}.norm2.weight`／`.bias` | `[E]` | 同名 | `[E]` | 変換不要 |

`{i}` は `Sequential` の位置 index 接頭辞（`crates/facade/src/compat/sequential.rs:1009-1033`）。fandhe `Linear.weight` は `[in_features, out_features]`（`crates/autodiff/src/nn/linear.rs:25`）で PyTorch `nn.Linear.weight`（`[out, in]`）と転置関係にある——これが表中「転置」操作の根拠である。`MultiheadAttention` の命名契約（`q_proj.*` → `k_proj.*` → `v_proj.*` → `out_proj.*`）は `crates/autodiff/src/nn/attention.rs:951-959`、`TransformerEncoderLayer` の 5 子層命名（`self_attn`／`linear1`／`linear2`／`norm1`／`norm2`）は `crates/autodiff/src/nn/transformer_encoder_layer.rs:194-202` を参照。

**罠（対象外）**: GPT-2 系 `Conv1D` は `c_attn.weight` が `[in, 3*out]`（PyTorch `nn.Linear` とは逆の規約）で保存される。HF BERT は per-projection キー（`attention.self.query.weight` 等、packed でない）を使う。いずれも本ガイドの変換規則の対象外であり、同じ原則（PyTorch 側の実際の shape を確認し、fandhe `Linear.weight` の `[in, out]` 規約へ転置する）で呼び出し側が自作する。

## 4. 例 1: `load_safetensors_f32` によるロード

```rust
use fandhe_ai::interop::safetensors::{load_safetensors_f32, require_keys};

let loaded = load_safetensors_f32(&checkpoint_path)?;
require_keys(&loaded, &expected_keys)?;
```

`LoadError` の各 variant（`crates/onnx-interop/src/st_load.rs:44-73`）: `Io`（ファイル I/O）・`SafetensorsFormat`（ヘッダ・レイアウト不整合）・`MissingKeys`（`require_keys` が全件収集。無言 skip 禁止の実体）・`UnsupportedDtype`（F32 以外。§2「事前変換」参照）・`DataLengthMismatch`（多層防御。通常到達しない）・`Shape`（要素数積オーバーフロー等）。一次ソースは `crates/facade/examples/hf_safetensors_sequential/main.rs` の「例 1」節。

## 5. 例 2: 変換 → `load_state_dict`

§3 の対応表を実装した `convert::from_pytorch_layout`（`crates/facade/examples/hf_safetensors_sequential/convert.rs`）が変換ロジックの一次ソースである:

```rust
let (restored_state, extra) = convert::from_pytorch_layout(&loaded, &["lm_head.weight"])?;
let mut model = build_model(seed)?;
model.load_state_dict(restored_state)?;
```

- `extra_allowlist`（第 2 引数）は `Sequential` の位置 index 接頭辞を持たない余剰テンソル（LM head 等）のキーを**明示的に**列挙する。allowlist にも変換規則にも無いキーは `ConvertError::UnexpectedKey` で拒否する（無言 drop 禁止。REQ-7）
- `load_state_dict` は strict・two-pass 契約（`crates/autodiff/src/nn/module.rs:452-509`）: キー集合の完全一致・shape 完全一致を検証してから代入し、失敗時はベストエフォートでロールバックする
- ロード後は事前に構築した `DeviceParamStore` が stale になる（`crates/facade/src/compat/sequential.rs:1020-1027` と同じ注意。CPU 推論のみを扱う本ガイドでは対象外）
- 元モデルとの `predict` 出力が bit 完全一致することを `crates/facade/tests/interop_safetensors_hf_layout.rs::hf_layout_roundtrip_restores_sequential_bit_exact` で確認している

## 6. 例 3: 推論スケッチ（batch decode）

`Sequential` は per-token の LM head（`Linear`）を積めない（`nn::Linear::forward` は rank-2 限定・`TransformerEncoderLayer` は `[B, L, E]` 固定。`crates/autodiff/src/nn/linear.rs:242-252`）ため、head 重みは `Sequential` の外で別途保持し、forward 出力へ `matmul` する 2 経路パターンを取る:

```rust
let tape = fandhe_ai::tape();
let ids_var = tape.var(&ids);                       // [B, L]
let hidden = model.forward(&tape, &ids_var)?;         // [B, L, E]
let flat = hidden.reshape(&[b * l, embed_dim])?;
let head_var = tape.var(&head_weight_transposed);     // [E, V]
let logits_flat = flat.matmul(&head_var)?;            // [B*L, V]
let logits = logits_flat.reshape(&[b, l, vocab])?;
let last = logits.narrow(1, l - 1, 1)?;               // [B, 1, V]（非 contiguous のため reshape しない）

let greedy_next = last.argmax(Some(2))?;              // Tensor<i32> [B, 1]
let probs = last.softmax(2)?;
let (_values, topk_index) = probs.topk(k, 2, true)?;  // 候補提示（サンプリングは host 側で合成）
```

**`Var::reshape` の非 contiguous 制約に注意**: `Var::narrow` は zero-copy view のため一般に非 contiguous であり、直後の `reshape` は `ShapeError::NonContiguousReshape` になる（`crates/autodiff/src/var.rs:2414-2443`）。本ガイドの実装は `narrow` 後の shape（`[B, 1, V]`）のまま縮約軸を指定して回避している。

**サンプリングについて**: `Var::topk` は候補（値・添字）を返すのみで、`torch.multinomial` 相当の抽選は本リポに存在しない。累積和抽選は `fandhe_ai::rand`（`[0, 1)` 一様分布）と host 側のループで合成する（新規 `Op` の追加なし）。本 example は候補提示までに留め、抽選ループの実装は含めない。

**causal 性について正直に書く**: `add_transformer_encoder`（`crates/facade/src/compat/sequential.rs:492-505`）は非 causal・mask なしの self-attention 固定である。したがって本スケッチは「ids → logits → 次 id → append → 全系列再計算」という**機構の例示**であり、causal LM として正しい確率を与えるものではない。真の causal decode には `Var::scaled_dot_product_attention(is_causal=true)` 相当の合成が必要で、`Sequential` はそれを公開していない。`predict_resident`（MHA 含有モデルは `BackendError::Unsupported` になる。`docs/facade-inference-serving-scope-decision.md` §2.1）も同じ制約を持つ。

## 7. KV キャッシュ統合（申し送り）

KV キャッシュは未実装（#2083 設計・#2084 実装・#2191 `generate()` はいずれも本ガイド執筆時点で OPEN）。現状の代替は §6 のとおり毎ステップ全系列を再計算する方式のみで、系列長に対して二次的なコストがかかる。

`is_causal=true` を decode（`L_q=1`）へそのまま流用すると無言で先頭 key のみに attend してしまう落とし穴が既に記録されている（`docs/facade-inference-serving-scope-decision.md` §2.1・§9）。KV キャッシュ結線が完了するまで、本ガイドの batch decode スケッチはこの落とし穴を踏まない「全系列再計算」方式に留める。`KvCache`／`add_stateful_attention` 等の API は本ガイドでは発明しない。

## 8. 検証

- `cargo run -p fandhe-ai --example hf_safetensors_sequential`（オフライン・決定的。`fandhe_ai::manual_seed` 固定）
- `cargo test -p fandhe-ai --test interop_safetensors_hf_layout`（§9 参照）
- 既存契約の非後退: `cargo test -p fandhe-ai --test interop_safetensors_roundtrip --test compat_sequential_state_dict --test api_surface`・`cargo test -p fandhe-ai --doc`
- `cargo package --list -p fandhe-ai`（example サブディレクトリ〈`examples/hf_safetensors_sequential/{main.rs,convert.rs}`〉の同梱確認。crates.io 公開対象のため）
- CUDA／Metal: 新規数値経路（`Op`／`BackendOps`／VJP）を追加しないため parity 対象外（`compat_sequential_state_dict.rs` と同じ整理）

## 9. テスト一覧（`crates/facade/tests/interop_safetensors_hf_layout.rs`）

| テスト名 | 検証内容 |
|---|---|
| `hf_layout_roundtrip_restores_sequential_bit_exact` | 合成 PyTorch レイアウト → ファイル保存 → `load_safetensors_f32` → 変換 → 別 seed の `Sequential` へ `load_state_dict` → `predict` 出力が元モデルと bit 完全一致 |
| `missing_keys_are_reported_in_full` | 必須キーを 2 件除去 → `require_keys` が両方を列挙 |
| `unexpected_key_is_rejected_not_dropped` | allowlist 外の余剰キー（`position_ids`）→ 型付き `Err`（無言 drop 禁止） |
| `in_proj_shape_mismatch_is_typed_error` | `in_proj_weight` が `[3E, E]` でない → panic せず `Err` |
| `head_weight_is_separated_via_allowlist` | `lm_head.weight` が `extra` へ分離され `Sequential` 側キー集合に混入しない |
| `synthetic_checkpoint_bytes_are_deterministic` | 同一モデルから 2 回生成したバイト列が完全一致 |
| `unsupported_dtype_is_typed_error` | F32 以外の dtype を含む手書きバイト列 → `LoadError::UnsupportedDtype`（変換前に fail-closed） |

## 10. セキュリティ（OWASP Top 10。`.claude/rules/security.md`）

- **A03 インジェクション／非信頼入力**: safetensors は非信頼な外部フォーマット。検証は `st_load`（ヘッダ・dtype・データ長・shape）に一任し再実装・迂回しない。`convert.rs` の変換ロジックは分割・転置の**前**に shape を検証し（`check_shape`）、index panic を起こさない。パスは `std::env::temp_dir()` 由来のみで、ユーザー入力からのパス連結・シェル展開は行わない
- **A08 データ整合性**: 余剰キー・未知キーは allowlist 外なら型付き `Err`（無言 drop 禁止）。`load_state_dict` の strict・ロールバック契約に依存し、部分適用状態を作らない
- **A06 脆弱・古いコンポーネント**: 依存追加なし（`hf-hub`／`tokenizers` は導入しない）。`Cargo.lock`・`deny.toml` は不変
- **秘密情報**: 一時ファイルの絶対パス・環境変数は出力・docs に書かない

## 11. スコープ外・引き継ぎ

- `site/guides/interop.md`／`site/examples/` への転記（docs-site 原稿同期）は本イシューでは行わない
- 実 HF ハブからのダウンロード（依存追加のユーザー承認が必要）・トークナイザは非目標のまま
- bf16／f16 チェックポイントのライブラリ内変換・入力サイズ上限の導入は対象外
- KV キャッシュ結線（#2083／#2084）・`generate()`（#2191）・`compat::Sequential::save`／`load` 薄いラッパー（`docs/facade-safetensors-exposure-decision.md` §11.3）は同 Phase の後続実装を待つ
- GPT-2 `Conv1D`／HF BERT 固有キーの変換規則は §3 の原則のみを記載し、個別実装は対象外

## 12. 出典一覧

- `crates/facade/src/interop/safetensors.rs`
- `crates/onnx-interop/src/st_load.rs`
- `crates/facade/src/compat/sequential.rs:448-505,1009-1033`
- `crates/autodiff/src/nn/module.rs:452-509`
- `crates/autodiff/src/nn/linear.rs:25,242-252`
- `crates/autodiff/src/nn/attention.rs:951-959`
- `crates/autodiff/src/nn/transformer_encoder_layer.rs:194-202`
- `crates/autodiff/src/var.rs:2414-2443`（`reshape` の非 contiguous 制約）
- `docs/facade-inference-serving-scope-decision.md` §2.1・§6・§9
- `docs/facade-safetensors-exposure-decision.md` §11.3
- `crates/facade/examples/hf_safetensors_sequential/{main.rs,convert.rs}`
- `crates/facade/tests/interop_safetensors_hf_layout.rs`
