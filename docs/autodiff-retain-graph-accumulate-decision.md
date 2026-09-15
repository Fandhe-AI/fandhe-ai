# `retain_graph` 契約と複数回 backward の勾配蓄積（イシュー #1749）

親: #1612（compat API §5 経路 2 承認）。前提: #1748（`Tape::var_no_grad`／
`Var::detach`。#1859 でマージ済み）。

## 1. 背景

現行 `Tape::backward(&self, loss)`（`crates/autodiff/src/backward.rs`）は
`&self` で呼べ、ノードを追加せず、呼び出しごとに独立した `Gradients` を
新規生成する。グラフノード（`TapeNode::value`）は `Tape::reset`（#1048）
または `Tape` の drop まで保持される。つまり **PyTorch の
`retain_graph=True` が常時成立**しており、`docs/compat-feature-gap.md`
§2.11 行も「フラグ自体が不要な設計」と整理済みだった。

一方で PyTorch の `.grad` 蓄積（複数回 `backward()` で勾配が加算される
契約。勾配蓄積によるマイクロバッチ学習・複数損失の合成に使う）に相当
する機能は無く、利用者が同一テープ上で複数の loss を backward して
勾配を足し合わせる手段が `Gradients` の外側での手作業（`get()` して
`eval` 系で加算）しかなかった。`Gradients` のフィールドは private で
あり、利用者が安全に合成できない。

本イシューは (a) `retain_graph` の契約を明文化・テストで固定し、
(b) 複数回 backward の勾配蓄積を明示 API として追加する。

## 2. 設計判断

### 2.1 `retain_graph`: 解放モードは実装せず「常時保持」契約を確定する

**結論**: `retain_graph=false` 相当の「backward 後にノード値を解放する
モード」は追加しない。`Tape` はグラフを `reset`／drop まで保持する＝
`retain_graph=True` 相当を無条件契約とし、明示的な解放手段は既存
`Tape::reset`（#1048）とする。

**理由（コード事実）**: `Var::value()`／`Var::to_tensor()`
（`crates/autodiff/src/var.rs`）は層 2 `materialize_non_fallible`
（`tape.rs`）を経由し、未実体化かつ `recompute == false` のノードは
契約違反として `debug_assert! + safe_zeros` で吸収される。backward
後に内部ノードの `value` を `take()` すると、生存中の内部 `Var`
（`Copy`・`&'t Tape` 借用）からの `value()` が release ビルドで
**黙ってゼロを返す**穴になる。PyTorch が解放するのは「backward 用に
保存した中間テンソル」であって出力値ではないが、本テープでは両者が
同一（`TapeNode::value`）のため分離できない。checkpoint（#1624）は
`recompute=true` を立てて再計算契約を与える別機構であり、解放モードの
代替にはならない。

**契約（テストで固定）**: 同一 epoch 内で `backward` を何度呼んでも
(1) 成功する、(2) ノードを追加しない（`Tape::len()` 不変）、(3) 呼び
出し間で勾配が bit 同一、(4) 各 `Gradients` は独立（暗黙の蓄積なし）。
checkpoint 区間を含む場合の再呼び出しは既存
`crates/autodiff/tests/checkpoint.rs::backward_called_twice_after_checkpoint_yields_same_gradient`
が既に固定しているため重複追加せず参照する。

### 2.2 蓄積 API: `Tape::backward_accumulate`

```rust
pub fn backward_accumulate(
    &self,
    loss: &Var<'_>,
    into: &mut Gradients,
) -> Result<(), AutodiffError>
```

配置: `crates/autodiff/src/backward.rs` の `impl Tape`（`Gradients` の
フィールドが private のため同モジュール内で実装する）。PyTorch の
「`loss.backward()` を繰り返すと `.grad` に加算される」契約の **opt-in
版**。素の `backward` の意味論（独立 `Gradients`）は不変。

処理順:

1. `loss.tape_id() != self.id` → `Err(TapeMismatch)`（既存
   `backward_impl` と同じ）。
2. `into.tape_id != self.id` または `into.epoch != self.epoch()` →
   `Err(TapeMismatch)`（`Gradients::get` と同じ世代検査。`reset` を
   またいだ蓄積を fail-closed に拒否する）。
3. `into.resident_fingerprint.is_some()` →
   `Err(AutodiffError::Backward(..))`（resident 経路
   〈`DeviceParamStore::backward`〉由来の `Gradients` は蓄積対象外。
   下記参照）。
4. `self.backward_impl(loss, None)` で新規 `Gradients`（`fresh`）を
   完全に計算する（ここで `Err` なら `into` は無変更）。
5. **原子的マージ**: `into.grads` と `fresh.grads` の長い方の長さ
   `n` で新規 `Vec<Option<Tensor<f32>>>` を作り、各 index について
   `(Some(a), Some(b))` → `grad::vjp_elementwise_add(ops, &a, &b)`
   （backward 内 fan-out 蓄積 `accumulate()` と同一ヘルパ）、片側
   のみ `Some` → clone、両方 `None` → `None`。**全要素の加算が成功
   してから** `into.grads` へ代入する（途中 `Err` で `into` を部分
   更新しない）。
6. 長さ不一致: `into` 生成後に同一 epoch で追加されたノード分だけ
   `fresh` が長くなりうる（`reset` を挟まない限りノード数は単調増加
   のため `into` が `fresh` より長くなることはないが、`get()` ベース
   の走査で添字越えでも panic しない防御を入れている）。

`requires_grad == false` ノードは `backward_impl` が `None` のまま
残すため、マージ後も `None`（`Gradients::get` は引き続き
`GradientTrackingDisabled`）。

**resident フィンガープリント拒否の理由**: `DeviceParamStore::step`
は `Gradients::resident_fingerprint()`（`(store_id, backward_serial,
pending 世代)`）で「渡された `Gradients` が今回のストア・今回の
backward 呼び出し・今まさに消費しようとしている pending の結果で
あること」を検証する。蓄積結果（2 回分以上の合算）を渡すとこの
「1 回の backward 呼び出しの結果そのもの」という契約が壊れるため、
resident 経由の `Gradients` は fail-closed に拒否する
（`ResidentResolver::resident_backward_fingerprint` は resolver を
使った場合に必ず `Some` を返すため、`backward`（`resolver: None`）
経由の `Gradients` は常に `None` のまま——`backward_accumulate` の
唯一の入力経路である素の `backward` が生む `Gradients` は本チェックに
引っかからない）。

変更しないもの: `Gradients` に `Clone`／pub フィールドを追加しない。
新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant は追加しない
（既存 `TapeMismatch`／`Backward(String)` で表現できる）。3 バックエンド
共通のホスト実装であり、バックエンド固有カーネルは不要。

## 3. 受け入れ条件（テンプレート項目）の本 issue への対応表

| issue 側の項目 | 本 issue での対応 |
|---|---|
| 対応する Op／`BackendOps` メソッドの追加 | **非該当**。演算グラフの新ノードやバックエンドカーネルを要しない |
| Var／Tape メソッドの追加 | `fandhe_ai_autodiff::Tape::backward_accumulate` と facade `fandhe_ai::Tape::backward_accumulate`（薄い委譲） |
| VJP の追加・数値微分／解析的検証 | 新規 VJP なし。蓄積は既存 `grad::vjp_elementwise_add` を再利用。検証は (a) 同一 loss 2 回蓄積＝`2·g` bit 一致（解析的）・(b) `l1`→`l2` 蓄積 ≈ `backward(l1.add(l2))`＋中央差分数値微分 |
| parity テスト | CPU 本番 ops（`fandhe_ai::tape()`）と `NaiveOps` の勾配 **bit 完全一致**（`mul`／`sum` のみのフィクスチャ）。Metal／CUDA は `#[ignore]` で分離・未実測を申し送り |
| `docs/compat-api-scope.md` 更新 | §1.2「autograd 制御（no_grad／detach／retain_graph）」行を実装済みへ更新 |
| fmt／clippy | 実装時に通す |

## 4. スコープ外

- backward 後にノード値を解放する `retain_graph=false` 相当モード
  （§2.1 の理由で不採用）。
- `DeviceParamStore`（resident 経路・`backward_device_param_store`）に
  対する勾配蓄積（fingerprint 契約と衝突するため fail-closed に拒否
  のみ）。
- optimizer 側の「蓄積回数で割る」等の正規化（利用者が `scale_grads`
  等で行う）。
- CUDA／Metal 実機での `#[ignore]` parity 実行（Mac／GB10 セッション
  へ申し送り）。

## 5. 実装記録

- `crates/autodiff/src/backward.rs`: `Tape::backward_accumulate`・
  private ヘルパ `merge_gradient_vecs` を実装。
- `crates/autodiff/src/tape.rs`: `Tape` 構造体 doc「学習ループでの
  運用」節へ retain_graph 契約を追記。
- `crates/autodiff/tests/retain_graph_accumulate.rs`: §2 のテスト群
  （12 件。全 green）。
- `crates/facade/src/lib.rs`: `Tape::backward_accumulate` を追加。
- `crates/facade/tests/backward_accumulate_backend_parity.rs`: CPU
  本番 ops と naive 参照実装の bit 完全一致確認（属性なし）・
  Metal／CUDA `#[ignore]`（未実測）。
