# LPPool（lp_pool1d・lp_pool2d・lp_pool3d）PyTorch 参照値フィクスチャの出自

イシュー #3825 の `tests/lp_pool.rs` の `parity` モジュール が参照する固定フィクスチャ。
`adaptive-pool3d-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。`F.lp_pool3d`／`nn.LPPool3d` が
  存在することもスクリプト冒頭で assert している。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits` を保存する。損失は
  `(out * g).sum()`、dtype は float32、乱数は固定シード 3825、`ceil_mode=False`。
- `cases`（22 件）: 1d 9・2d 7・3d 6。`norm_type` は 0.5・1・1.5・2・3、stride の有無、全零窓、負の入力
  （整数 `p` と非整数 `p`〈forward が NaN〉）、`0 < p < 1` かつ窓内に 0、を含む。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/python -I gen_reference.py > lp_pool_reference.json
rm -rf /path/to/venv
```

## sha256

```
4cc4d9ab8bc47f6face79ae3e8b86327f59e0fbd76df3c20b423fa34d1ee0576  lp_pool_reference.json
9b3d0242691820b91c88f9a9c0a7c15f5f919985371be2c5a0cae98f4a4c7313  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
