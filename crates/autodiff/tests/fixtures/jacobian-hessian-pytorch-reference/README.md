# jacobian・hessian PyTorch 参照値フィクスチャの出自

イシュー #2670（親 #2668）の `tests/jacobian_hessian_parity.rs` が参照する固定フィクスチャ。
`packed-sequence-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  `torch.autograd.functional.jacobian`／`hessian` を `vectorize=False`・`create_graph=False`・float32 で実行。
  numpy や手計算値で代替していない（使い捨て venv に numpy は入っているがスクリプトは使わない）。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力・定数・出力（`out`）・期待値（`expected`）を
  すべて保存し、Rust 側で入力を再生成しない。乱数は固定シード 2670。
- `jacobian_cases`（9 件）: 要素ごとの合成（`vec_elementwise`）・スカラー出力（`scalar_sum`）・matmul 合成で
  出力・入力とも rank 2（`matmul_tanh`・`mlp`）・非 contiguous な出力（`transpose_out`・
  `broadcast_out`）・入力に依存しない行を含む出力（`independent_rows`）・縮約（`mean_dim`）・
  rank 0 の入力と出力（`scalar_in`）。
- `hessian_cases`（9 件）: 二次形式（`quadratic`。閉形式 `A + Aᵀ`）・入力に線形な loss（`linear`。全ゼロ）・
  `tanh_sum`・`exp_mean`・小さな MLP（`mlp`）・rank 0 の入力と loss（`scalar_in`）・shape `[1, 1]` の loss
  （`loss_1x1`）・`cat`＋`transpose`（`cat_transpose`）・`relu_cubic`。プログラムはすべて
  `Op::supports_create_graph` の対象 Op に 1 対 1 で写せる演算のみで組む。
- プログラム名は Rust 側 `build_jacobian_program`／`build_hessian_program` が同名で組む。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py jacobian_hessian_reference.json
rm -rf /path/to/venv
```

## sha256

```
9b07c742c754ca44016f98198f0a1c0b5120723a85c1301b3c22315589935ede  jacobian_hessian_reference.json
37e6ff6e50212d40b5048d57f7853488862617fe2e96ed70336f5466a238dd15  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
