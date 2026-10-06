# pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL PyTorch 参照値フィクスチャの出自

イシュー #2652（親 #2651）の `tests/elementwise_loss_parity.rs` が参照する固定フィクスチャ。
`softmin-threshold-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4
  （JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で代替していない。
- 固定シード `2652`。損失は各 `F.*` を上流勾配 1 の `loss.backward()` で微分する。
  勾配は BCE が `input`・`target`、Hinge／SoftMargin が `input`、GaussianNLL が
  `input`・`target`・`var`。入力・損失値・勾配・shape・パラメータをすべて JSON に保存する
  （Rust 側で入力を再生成しない）。dtype は float32。
- `cases`（計 74 件。NaN／inf を含まず、hinge は `x == margin` ちょうどを含まない）:
  - `bce_pos_weight`（30 件）: 形状 `[6]`・`[4,3]`・`[2,3,4]` × `pos_weight` 形（`[C]`・`[1]`・
    完全一致）× Mean／Sum（hard／soft target 交互。18 件）、`pos_weight = ones`（6 件）、
    `pos_weight` なし〈委譲経路〉（6 件）
  - `hinge_embedding`（18 件）: 形状 3 種 × margin（1.0・0.5・2.0）× Mean／Sum
  - `soft_margin`（6 件）: 形状 3 種 × Mean／Sum（`|x| <= 8`）
  - `gaussian_nll`（20 件）: 形状 3 種 × `full` × Mean／Sum（12 件）、`var < eps`
    （`var == 0`・`var < eps` を含む。eps = 1e-6／0.3。8 件）
- `edge_cases`（11 件）: NaN／inf は JSON で表せないため
  `{"class": nan|pos_inf|neg_inf|finite, "value"}` で保存する。実測で確定した PyTorch の挙動:
  - **hinge**: `x == margin`（`y = -1`）の勾配は **0**（`clamp_min` の「境界で通す」契約ではない。
    本実装は `margin - x > 0` のときのみ勾配を通す）。`y = -1` で `x = NaN` の勾配は 0、
    `x = -inf` の勾配は `-1`、`x = +inf` は 0
  - **gaussian_nll**: `var == 0`（`eps` 未満）でも `dvar` は clamp 後の値で評価され遮られない
    （`-1.25e11` 等。`dx = d / eps`）。`var = +inf` の損失は `+inf`・勾配は 0
  - **bce_pos_weight**: `x = ±inf`・`y = 1, x = -inf` の `(1 - y)·x` は NaN（`0 · inf`）
  - **soft_margin**: `x = 0` の勾配は `∓0.5`。大振幅（`-y·x = 200`）は PyTorch が f32 の
    `log1p(exp(.))` で `inf`・勾配 NaN になる（`diverges` 付き。本実装は安定形で有限値を返す差分。
    `docs/autodiff-elementwise-loss-ops-decision.md` §5）

## 再生成手順

一時 venv（リポジトリ外）で CPU 版 PyTorch 2.14.0 を導入して実行する。venv は worktree 内に置かない。

```sh
python3 -m venv "$VENV"
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
"$VENV/bin/python" gen_reference.py > elementwise_loss_reference.json
```

## sha256

```
a7da1374e5096e3c614651ba63d598d61f1c0a50eeac2de2838bc97a180d0360  gen_reference.py
9a141ae2b47272598b2fc592757f37b68bbbc21152098acb42580fe6f406e354  elementwise_loss_reference.json
```
