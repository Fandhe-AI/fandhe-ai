# Functional API 結合ノード（Concatenate・Add・Multiply・Average）PyTorch 参照値フィクスチャの出自

イシュー #2666（親 #2663）の `crates/facade/src/compat/functional/merge_tests.rs` が参照する固定フィクスチャ。
`functional-graph-pytorch-reference/README.md`（#2665）と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。
#2665 のフィクスチャ（sha256 固定物）は再生成していない。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy や手計算値で代替していない
  （numpy 未導入の使い捨て venv で実行）。
- PyTorch に Functional API は無いため、`F.linear`・活性化・`torch.cat`／`+`／`*`／`(a + b + …) / n` を手で結線した
  参照を使う。目的は**結線・結合ノード経由の勾配合流・結合の出力を別の結合へ入れる連鎖**。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力・重み・上流勾配・各出力・各入力勾配を保存し、Rust 側で
  再生成しない。dtype は float32、乱数は固定シード 2666。
- **重みレイアウトの写像は `gen_reference.py` 側で行う**（本リポの `Linear.weight` は `[in, out]`）。キーは全ブロックを
  通した層の通し番号 `"{i}.weight"`／`"{i}.bias"`。結合ノードは層を持たず通し番号に影響しない。
- ケース（6 件）: `residual_add`（入力とブロック出力の加算）・`concat_two_branches_then_linear`（2 分岐の連結後に
  Linear）・`multiply_gate`（sigmoid ゲート × tanh）・`average_three_branches`・`chained_merges_two_inputs`（結合の
  出力をさらに結合へ入れる連鎖・2 入力・2 出力）・`merge_only_no_blocks`（ブロック 0 個の結合のみグラフ）。
- 損失は `Σ_k Σ(out_k ⊙ g_k)`。保存する勾配は各入力に対するもの。パラメータ勾配は `bind` 経路（#2667）が必要なため
  本フィクスチャでは照合しない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > functional_merge_reference.json
rm -rf /path/to/venv
```

## sha256

```
2ea4fe33690f79fd1ea835e5b4b085526bd9ebb116da0137096057975016a85c  functional_merge_reference.json
dc6f675a08a62fb66f5263f9e4bbf81e1a4d35a2321e05650d93a61f4d35a21b  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
