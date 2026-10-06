# ConvTranspose3d・MaxUnpool PyTorch 参照値フィクスチャの出自

イシュー #2644（親 #2625）の `tests/conv_transpose3d_parity.rs`・`tests/max_unpool_parity.rs` が参照する固定
フィクスチャ。`pool3d-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。入力・重み・bias・
  上流勾配 `g`・出力・各入力勾配、MaxUnpool は索引（torch の int64。`(n, c)` 平面内 flat 添字）も保存する。
  損失は `(out * g).sum()`、dtype は float32、乱数は固定シード 2644。
- **`torch.set_num_threads(1)` で生成する**。重複索引の `max_unpool` は PyTorch の並列 CPU カーネルが書き込みを
  競合させ、複数スレッドでは同一入力でも勝者が実行ごとに変わる（実測: 1d の `[1,1,3,3]` 索引で出力の勝者が
  3.0 と 4.0 に揺れ、JSON の sha256 が実行ごとに変わった）。単一スレッドでは再生成で同一の JSON になる
  （3 回実行して同一 sha256 を確認）。単一スレッドでは最後の書き手が残る。
- `conv_transpose3d_cases`（10 件）: 基本（k=2・s=1）・bias の `Cout == Wout` 軸回帰・stride 2＋padding 1＋
  output_padding 1・非等方（kernel／stride／padding／output_padding／dilation が軸ごとに異なる・N=2）・
  dilation 2・groups=2・depthwise・N=2 かつ Cin≠Cout・1×1×1 kernel・stride 3＋output_padding 2。
  各ケースに入力・重み・bias・`g`・出力・`d_input`／`d_weight`／`d_bias` を保存。
- `conv_transpose3d_error_cases`（11 件）: torch が例外を出すか否かを実測して記録（`torch_raises`）。実測で
  確定した点:
  - `output_padding >= stride` は拒否されるが、`output_padding < dilation` なら受理される
    （`stride=1・dilation=2・output_padding=1` は受理）。本実装は `output_padding < stride` のみ受理（差分）。
  - rank 4（バッチなし）は torch が受理する。本実装は rank 5 のみ（差分）。
  - `N = 0` は torch も受理する。
- `max_unpool_cases`（12 件）: `F.max_pool{1,2,3}d(return_indices=True)` 由来の（値・索引）を入力にした
  1d／2d／3d 各 4 件（k=2・s=2／k=3・s=2・p=1／奇数長＋`output_size`／stride≠kernel）。stride < kernel の
  重なり窓を含むケースは索引が重複する（同じ最大値要素由来なので値は同じ）。
- `max_unpool_dup_cases`（5 件）: 重なり窓（k=3・s=1）由来の重複索引（1d／2d／3d）と、値が異なる手作りの重複索引
  （1d／2d。forward の「最後の書き手」判定用）。
- `max_unpool_nonfinite_cases`（3 件）: NaN／±inf／`-0.0` を含む入力（コピーのみなのでビット保存を確認）。
- `max_unpool_error_cases`（14 件）: 索引範囲外・負値・索引 shape 不一致・`output_size` の境界値・バッチなし入力・
  kernel／stride の 0・`N = 0`・`C = 0`・rank 不一致を実測。実測で確定した点:
  - `output_size`（空間軸のみ）は `default − stride < size < default + stride` の開区間（default=6・stride=2 で
    5・7 は受理、3・4・8 は拒否）。
  - `kernel = 0` は受理（空出力）・`C = 0` は拒否・`N = 0` は受理。
  - 重複索引の勾配は `gather(grad_output, index)`（全書き手へ配る）。
  - 本実装との突き合わせは `docs/autodiff-conv-transpose3d-max-unpool-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > conv_transpose3d_max_unpool_reference.json
rm -rf /path/to/venv
```

## sha256

```
e83b2f0b611d87376a10a05783089f64bc6e248b22e43de462f43e3bafae1197  conv_transpose3d_max_unpool_reference.json
1484e309048a751ff1b99010d3f911bd98388a6b7a134105787f7d4dccaccea3  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。エラーケースの期待表
（`tests/conv_transpose3d_parity.rs::err_spec`・`tests/max_unpool_parity.rs::err_spec`）の `torch_raises` は
fixture と照合されるため、再生成後に食い違えば表を更新する。
