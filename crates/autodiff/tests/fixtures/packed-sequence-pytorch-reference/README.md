# 可変長系列（pack／unpack・RNN 系 packed 実行）PyTorch 参照値フィクスチャの出自

イシュー #2647（親 #2625）の `tests/packed_sequence_parity.rs` が参照する固定フィクスチャ。
`shape-view-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため。NaN の payload も保存）。
  入力・上流勾配・各出力・各勾配を保存し、Rust 側で入力を再生成しない。dtype は float32、乱数は固定シード 2647。
- **重みレイアウトの写像は `gen_reference.py` 側で行う**: 本リポは `x·W` 規約（`weight_ih: [D, G*H]`・
  `weight_hh: [H, G*H]`）、PyTorch は `[G*H, D]`・`[G*H, H]`（転置）。JSON には本リポのレイアウトで
  保存し、重み勾配も同じ向きで保存する。ゲート順は LSTM `i,f,g,o`・GRU `r,z,n` で PyTorch と同じ
  （`docs/autodiff-rnn-cell-tape-design.md` 決定 5・12）。
- `pack_cases`（16 件）: `batch_first` 両方・`enforce_sorted` 両方・長さ未整列・同長（タイ）・全長同一・長さ 1 の系列・
  `max(lengths) < T`・trailing 次元なし／1 個／2 個・バッチ 1・非 contiguous 入力（`pre_transpose`）・
  非有限値（NaN payload `0x7FC00001`／`0xFFC00002`・`±inf`・`-0.0`。名前に `nonfinite`）。保存するもの:
  入力・`data`・`batch_sizes`・`sorted_indices`・`unsorted_indices`・上流勾配・入力勾配。
- `unpack_cases`（12 件）: `padding_value`（0・非 0・NaN・inf）・`total_length`（`None`・`T_max`・`T_max` 超）・
  `batch_first` 両方・trailing 次元なし・全長同一。葉の `data` から直接組んだ `PackedSequence` の勾配を保存する。
- `rnn_cases`（19 件）: `nn.RNN`（tanh）／`nn.LSTM`／`nn.GRU` について、単層単方向・2 層・双方向・2 層双方向
  （`dropout = 0`）、`h0`（`c0`）の省略と指定、`enforce_sorted = false` の未整列長、バイアスなし。保存するもの:
  重み・`output.data`・`h_n`（`c_n`）・入力・全重み・`h0`（`c0`）への勾配。損失は
  `Σ(output.data ⊙ g1) + Σ(h_n ⊙ g2)`（LSTM は `+ Σ(c_n ⊙ g3)`）。名前に `control` を含む 3 件は全系列長 = `T` の
  対照ケースで、PyTorch の **非 packed 実行**の出力・`h_n` も保存する（packed の不一致が pack 起因か既存セル起因かの
  切り分け用）。
- `error_cases`（8 件）: 長さ 0・`T` 超過・`lengths` 数不一致・`enforce_sorted = true` で未整列・rank 1 入力・
  `batch_first` で `T` 超過・`total_length < T_max`・`total_length == T_max` で torch が例外を出すか（`torch_raises`）。
  実測の要点は `docs/autodiff-packed-sequence-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > packed_sequence_reference.json
rm -rf /path/to/venv
```

## sha256

```
f28d19005954f1c6abf9a58296738a2956894bec0c75367a1e0c879c348161a0  packed_sequence_reference.json
b5bcf1e43e01fca80d13c1041e5f716df561c66f9515bc2c9ba29e563db4ffde  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
