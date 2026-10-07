# Functional API（多入力・多出力グラフ）PyTorch 参照値フィクスチャの出自

イシュー #2665（親 #2663）の `crates/facade/src/compat/functional/tests.rs` が参照する固定フィクスチャ。
`crates/autodiff/tests/fixtures/packed-sequence-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を
記録した状態で JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy や手計算値で代替していない
  （numpy 未導入の使い捨て venv で実行）。
- PyTorch に Functional API は無いため、`F.linear` と活性化（ReLU／Tanh／Sigmoid）を手で結線した参照を使う。
  層ごとの数値は既存テストが担保済みで、本フィクスチャの目的は**結線・出力順・入力順・fan-out の勾配合流**。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力・重み・上流勾配・各出力・各入力勾配を保存し、Rust 側で
  再生成しない。dtype は float32、乱数は固定シード 2665。
- **重みレイアウトの写像は `gen_reference.py` 側で行う**: 本リポの `Linear.weight` は `[in, out]`（`x·W` 規約）、
  PyTorch は `[out, in]`。JSON には本リポのレイアウトで保存し、キーは全ブロックを通した層の通し番号
  `"{i}.weight"`／`"{i}.bias"`（Block ノードの挿入順に層数を累積。活性化層も 1 層と数える）。
- ケース（5 件）: `two_in_two_out_disjoint`（独立 2 経路）・`fan_out`（共有ブロックから 2 分岐・出力順を挿入順と逆に
  指定・入力勾配の合流）・`chain_with_inner_output`（3 ブロック連鎖で中間ノードも出力・1 ブロック内に Linear 2 層）・
  `inputs_permuted`（入力ノードの生成順と `inputs` の指定順が逆・1 入力を 2 ブロックが消費）・
  `passthrough_output`（入力ノードをそのまま出力に含める）。
- 損失は `Σ_k Σ(out_k ⊙ g_k)`（`g_k` は保存した上流勾配）。保存する勾配は各入力に対するもの。パラメータ勾配は
  `bind` 経路（#2667）が必要なため本フィクスチャでは照合しない。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > functional_graph_reference.json
rm -rf /path/to/venv
```

## sha256

```
4f595e498f4165fa3d5252f208ad6179e274d23ee88e71236b688be013131a22  functional_graph_reference.json
71f1c3407a1046bfb6d96c727cf19e61a389b018f8c2604eed5f4691f58af1d5  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
