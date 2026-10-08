# vjp・hvp・vmap PyTorch 参照値フィクスチャの出自

イシュー #2877（親 #2841）の `tests/functional_ops_pytorch_parity.rs` が参照する固定フィクスチャ。
`jacobian-hessian-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。設計の正は `docs/autodiff-functional-transforms-design.md` §6・§18。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  float32 で実行。numpy や手計算値で代替していない（numpy は未導入で、スクリプトも使わない）。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力・定数・出力（`out`）・余接 `u`／方向 `v`・
  期待値（`expected`）をすべて保存し、Rust 側で入力を再生成しない。乱数は固定シード 2877。
- 使った API（JSON の `api` に記録。`torch_func_has_hvp` の実測値は **false**。
  2.14.0 の `torch.func` に `hvp` は無い）:
  - vjp: `torch.func.vjp(f, x)` の `vjp_fn(u)[0]`。
  - hvp: 主参照は `torch.func.vjp(torch.func.grad(g), x)` の `vjp_fn(v)[0]`（fandhe 側と同じ
    reverse-over-reverse）。スクリプト内で `torch.autograd.functional.hvp(g, x, v)[1]` と
    `torch.allclose`（既定許容）で一致することを assert して相互検証している（JSON には主参照のみ保存）。
  - vmap: `torch.func.vmap(f, in_dims=k, out_dims=0)(x)`。forward 値のみ（vmap 出力の
    backward・vjp・hvp は #2878 の範囲）。
- `torch.func.grad` はスカラー出力を要求するため、shape `[1, 1]` の loss（`loss_1x1`）は grad に渡す
  関数だけを `.reshape(())` で包む（`out` に保存する forward 値は包む前の `[1, 1]`）。
- `expected_f64`: 同じ f32 入力を float64 へ上げて同じ API で計算した値。**診断用でありゲートには使わない**
  （判定不能の分類根拠として PyTorch の f32 値と f64 真値の差を見る場合にだけ使う）。

## ケース

プログラム名は Rust 側（`functional_ops_pytorch_parity.rs`）が同名で `Var` 演算により組む。

- `vjp_cases`（9 件。`build_vjp_program`）: `vec_elementwise`・`scalar_sum`（rank 0 の余接）・
  `matmul_tanh`・`mlp`（rank 2 の matmul 合成）・`transpose_out`（非 contiguous 出力）・
  `broadcast_out`・`independent_rows`（入力に依存しない行）・`mean_dim`・`scalar_in`（rank 0 入力）。
- `hvp_cases`（9 件。`build_hvp_program`）: `quadratic`・`linear`（全ゼロ）・`tanh_sum`・`exp_mean`・
  `mlp`・`scalar_in`・`loss_1x1`・`cat_transpose`・`relu_cubic`。`Op::supports_create_graph` の対象 Op のみ。
- `vmap_cases`（7 件。`build_vmap_program`）: `elementwise`・`in_dim1`（`in_dim = 1`）・
  `matmul_closure`（rank 3 入力のスライス `[2, 3]` に定数 `w` を閉包して matmul）・`transpose`
  （スライス出力が非 contiguous）・`to_scalar`（スライス→rank 0）・`expand`・`cat`（スライス→高 rank／長い軸）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python -I gen_reference.py functional_transforms_reference.json
rm -rf /path/to/venv
```

## sha256

```
c74a1d773eb65fdd3b54a6b46b2bf44eca4749012cdfd2c88dd9606e6d27c3c0  functional_transforms_reference.json
8764574b2860aaf8948d2f48648eda8e3350ec7c902562b3c8b74889beb23f7f  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
