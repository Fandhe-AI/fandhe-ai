# MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss PyTorch 参照値フィクスチャの出自

イシュー #2653（親 #2651）の `tests/margin_focal_loss_parity.rs` が参照する固定フィクスチャ。
`elementwise-loss-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4
  （JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で代替していない。
- マージン系 3 種は `torch.nn.functional` の `multi_margin_loss`／`multilabel_margin_loss`／
  `multilabel_soft_margin_loss` をそのまま呼ぶ。**`torch.nn.functional` に focal loss は存在しない**ため、
  `torchvision.ops.sigmoid_focal_loss` と同じ式（`p = σ(x)`・`ce = BCEWithLogits(none)`・
  `p_t = p·t + (1−p)(1−t)`・`loss = ce·(1−p_t)^γ`・`α_t = α·t + (1−α)(1−t)`）を `gen_reference.py` 内で
  **素の torch 2.14.0 の autograd で評価**している（torchvision は導入していない）。
- 固定シード `2653`。損失は上流勾配 1 の `loss.backward()` で `input` のみ微分する
  （`target`・`weight` は非追跡）。入力・損失値・勾配・shape・パラメータをすべて JSON に保存する
  （Rust 側で入力を再生成しない）。dtype は float32。
- `cases`（計 164 件。NaN／inf を含まず、マージン系は `z == 0` ちょうどを含まない）:
  - `multi_margin`（48 件）: 形状 `[5]`〈target は 0 次元〉・`[4,3]`・`[6,5]` × `p ∈ {1,2}` ×
    `margin ∈ {1.0, 0.5}` × weight 有無 × Mean／Sum
  - `multilabel_margin`（8 件）: 形状 `[5]`・`[4,5]`・`[3,6]` × Mean／Sum（target 個数を 0〜C で散らし、
    `-1` 以降にゴミ値を含む行を混ぜる）＋ 決め打ち 4×5（空集合行・全クラス行・終端以降に値を持つ行）× Mean／Sum
  - `multilabel_soft_margin`（24 件）: 形状 `[5]`・`[4,3]`・`[6,5]` × weight 有無 × hard／soft label × Mean／Sum
  - `sigmoid_focal`（84 件）: 形状 `[6]`・`[4,3]`・`[2,3,4]` × `(alpha, gamma)` ∈
    `{(0.25,2)・(なし,2)・(0.5,0)・(なし,0)・(0.75,1)・(0.25,0.5)・(0.25,3.5)}` × hard／soft × Mean／Sum
- `edge_cases`（23 件）: NaN／inf は JSON で表せないため `{"class": nan|pos_inf|neg_inf|finite, "value"}`
  で保存する。実測で確定した PyTorch の挙動:
  - **`z == 0` ちょうどの勾配は 0**（multi_margin の p=1・p=2、multilabel_margin）。判定は `z > 0` の厳密不等号
  - **NaN は損失・勾配に寄与しない**（`z > 0` が偽）。`x_y = NaN` の行は丸ごと 0、`x_j = NaN` の `j` のみ除外。
    `z = +inf` は損失 `+inf`（multilabel_margin）
  - **multilabel_margin**: 重複 target 添字は重複分だけ加算される・先頭が `-1` の行は損失 0・全クラスが target の行
    （終端なし）は損失 0・`-1` 以降の値は集合に入らず無視される。**終端以降も含めた全要素が `-1 <= t < C` でなければ
    PyTorch は `target is out of range` で拒否する**（実測。本実装も同じ範囲検査）
  - **multilabel_soft_margin**: `x = ±inf` と `t ∈ {0,1}` の組で `0·inf = NaN`（損失 NaN）、`x = +inf, t = 0` は損失 `+inf`。
    勾配は有限（`(σ(x) − t)/C`）
  - **sigmoid_focal**: 飽和域（`|x| = 40, 20`）の勾配は `γ ∈ {0, 1, 2}` で PyTorch も有限（0）だが、**`γ = 0.5` では
    `1 − p_t` が f32 で 0 に潰れ pow の逆伝播が NaN になる**（`focal_saturated_g0.5`。`diverges` 付き。数学的理由は
    `docs/autodiff-margin-focal-loss-ops-decision.md` §5）。`x = ±inf` の 1 要素ずつの値クラスは `focal_inf_*` で固定

## 再生成手順

一時 venv（リポジトリ外）で CPU 版 PyTorch 2.14.0 を導入して実行する。venv は worktree 内に置かない。

```sh
python3 -m venv "$VENV"
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
"$VENV/bin/python" gen_reference.py > margin_focal_loss_reference.json
```

出力は決定的（同一環境での再生成で JSON の sha256 が一致することを確認済み）。

## sha256

```
519aea360e2a9ce7805ecdb88dca8d36d120b1d09a7ca33f6c1cff00173b5a42  gen_reference.py
6169d7b93c3a534eac09910c6babc95bd45f8978ca6f9f0cb1f9cbc91b4561eb  margin_focal_loss_reference.json
```
