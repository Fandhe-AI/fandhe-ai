# cosine_similarity・pairwise_distance PyTorch 参照値フィクスチャの出自

イシュー #3821（親 #3813）の `tests/similarity_distance_pytorch_parity.rs` が参照する固定フィクスチャ。
`softmin-threshold-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録）。numpy や手計算値で代替していない。
- 損失は `(out * g).sum()`（`g` は固定シード `3821` 由来の一様乱数 `[-1, 1)`）。入力・上流勾配・出力・入力勾配・
  shape・パラメータをすべて JSON に保存する（Rust 側で入力を再生成しない）。dtype は float32。
- `cases`（計 25 件。NaN／inf を含まない）:
  - `cosine_similarity`: 形状 `[5]`・`[3,4]`・`[2,3,4]` の各軸（6 件。`eps = 1e-8`）
  - `pairwise_distance`: 形状 `[4]`・`[3,5]`・`[2,3,4]` × `p ∈ {1, 2, 3}` × `keepdim ∈ {false, true}`
    （18 件。`eps = 1e-6`）と `eps = 0` の 1 件
- `clamp_cases`（2 件）: 行 0 が `0 < ‖x1‖ < eps`、行 1 が `x1` のゼロベクトル、行 2 が通常の行。`eps = 1.0` と
  `eps = 1e-8`。PyTorch の straight-through 勾配（クランプを勾配に記録しない）をそのまま保存し、設計記録
  `docs/autodiff-similarity-distance-decision.md` §5 の勾配式がこれと一致することを確かめる。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python -I gen_reference.py > similarity_distance_reference.json
rm -rf /path/to/venv
```

Python 依存は `Cargo.toml` の依存ではなく、生成ツールとしてだけ使う（先例と同じ。リポの lock には入れない）。

## sha256

```
39261ae8152e8f6c103d455fdc18b29a5c10013dd288326bcc88a9281f81ba3c  similarity_distance_reference.json
0c883ba728e839700d7b974e8868fd3616cdb5f5a80f1e1b157f1e8bc1e211d4  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
