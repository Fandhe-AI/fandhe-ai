# Functional API 学習（パラメータ勾配・fit の軌跡）PyTorch 参照値フィクスチャの出自

イシュー #2667（親 #2663）の `crates/facade/src/compat/functional/fit_parity_tests.rs` が参照する固定フィクスチャ。
`functional-merge-pytorch-reference/README.md`（#2666）と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。
#2665・#2666 のフィクスチャ（sha256 固定物）は再生成していない。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の `torch_version`／
  `python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy や手計算値で代替していない
  （numpy 未導入の使い捨て venv で実行）。
- PyTorch に Functional API は無いため、`F.linear`・活性化・`torch.cat`／`+`／`*`／`(a + b + …) / n` を手で結線した
  参照を使う。`Linear.weight` は本リポの `[in, out]` レイアウトのリーフを直接持ち、`F.linear(x, W.t(), b)` で使う
  （保存する勾配・最終値も本リポのレイアウト）。キーは全ブロックを通した層の通し番号 `"{i}.weight"`／`"{i}.bias"`。
- **f32 はすべて u32 のビットパターン配列で保存**する。入力・初期重み・目標・勾配・epoch 損失・最終パラメータを保存し、
  Rust 側で再生成しない。dtype は float32（分類目標のみ int64 → JSON の整数列）、乱数は固定シード 2667。
- ケース（5 件）:
  - `grad_two_in_two_out_fan_out`（kind `grad`）: 2 入力・fan-out（分岐が add と concatenate の両方へ入る）・結合 2 種・
    2 出力。損失は出力ごとの `F.mse_loss`（mean）の和。**初期パラメータでの全パラメータ勾配**と損失値を保存する
    （#2666 の fixture は入力勾配のみ。パラメータ勾配は `bind` 経路が要るため本 fixture が初出）。
  - `grad_multiply_average_fan_out`（kind `grad`）: multiply・3 入力 average・fan-out を含む単一出力グラフ。
  - `fit_sgd_momentum_two_in_two_out`・`fit_adam_two_in_two_out`（kind `fit`）: 同じ 2 入力 2 出力グラフを
    `torch.optim.SGD(lr=0.05, momentum=0.9)`／`torch.optim.Adam(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`
    で 3 epoch・バッチ 4・10 サンプル（端数バッチ 2 件を含む）・`shuffle=false` で学習。epoch 損失は
    Rust 側の集計式と同じ「バッチ損失 × バッチ件数の f64 合計 / 全件数」を 1 回 f32 化した値。最終パラメータを保存する。
  - `fit_cross_entropy_chain`（kind `fit`）: 単一入力・単一出力のチェーン（Linear→tanh→Linear）を `F.cross_entropy`（mean）・
    int64 クラス目標・`SGD(lr=0.1)` で学習。
- 意図した差分（PyTorch との相違）: なし（本 fixture の範囲では optimizer の定義・損失の reduction とも PyTorch と同一）。
  本リポ側で追加の拒否（`Lbfgs`・`accumulate_steps != 1` 等）がある点は fixture の対象外で、ユニットテスト
  （`fit_tests.rs`）が固定する。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > functional_fit_reference.json
rm -rf /path/to/venv
```

## sha256

```
dbe5da153e942512dd56fec826fb74b205c5a951a324083f538d63afc54bab6e  functional_fit_reference.json
59e30f3f06997617a07866d758cdb69a9e55820343f73701531280d1a29b8403  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
