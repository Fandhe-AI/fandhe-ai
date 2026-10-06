# 索引付き更新（scatter_reduce・index_add・index_copy・masked_scatter）PyTorch 参照値フィクスチャの出自

イシュー #2641（親 #2625）の `tests/indexed_update_parity.rs` が参照する固定フィクスチャ。
`cumulative-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・src（`src_bits`。`index_add`／`index_copy`／`masked_scatter` では `source`）・上流勾配
  `g_bits`・出力 `out_bits`・入力勾配 `grad_x_bits`・src 勾配 `grad_src_bits` を保存する。損失は
  `(out * g).sum()`、dtype は float32、乱数は固定シード 2641。
- `cases`（250 件）:
  - `scatter_reduce` 234 件 = 5 mode（`sum`／`prod`／`mean`／`amax`／`amin`）× `include_self` 真偽 × 各形状。
    - `group = "finite"`（156 件）: 重複添字あり／なし・未書き込み位置あり・rank 1〜3 で `dim` を変える・
      `index` が非 `dim` 軸で入力より小さい形・空 index・タイ・`prod` の 0 が 0／1／2 個以上・`±0`。
    - `group = "selfeq"`（8 件）: `include_self == false` の `amax`／`amin` で、触れられた位置の入力値が
      結果と偶然一致する行（生成スクリプトが自動判定して群を分ける）。PyTorch は入力も分配数に数える
      ため勾配分配が本実装と異なる（決定記録 §5）。
    - `group = "nonfinite"`（70 件）: NaN・`±inf` を src／入力に含む 7 ケース × 5 mode × 2。
  - `index_add` 4 件・`index_copy` 4 件（重複添字は `index_add` のみ。`index_copy` の重複は PyTorch 未定義の
    ため fixture 突合に使わない）・`masked_scatter` 8 件（余剰 source・全偽・全真・小さい／列／大きい
    broadcast mask・source の shape が異なる）。
- `error_cases`（17 件）: 範囲外／負の添字・範囲外 `dim`・shape 不一致・source 不足・mask の broadcast 不能・
  0 次元入力などで torch が例外を出すか否かを実測して記録（`torch_raises`）。本実装との突き合わせは
  `docs/autodiff-indexed-update-ops-decision.md` §5。
- 勾配が例外になった組合せは無かった（全ケースで `grad_raises` は空）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > indexed_update_reference.json
rm -rf /path/to/venv
```

## sha256

```
0c702214d1fd3e907e7ba6e8a1b571e2f9fdf7c8bcaa0fa3f09514136132540d  indexed_update_reference.json
91fd00a6478feb47ccb6e9cc6de4d14344131b167b9c7abb9fde02724f591939  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
