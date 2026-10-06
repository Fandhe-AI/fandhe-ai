# Fold／Unfold PyTorch 参照値フィクスチャの出自

イシュー #2645（親 #2625）の `tests/fold_unfold_parity.rs` が参照する固定フィクスチャ。
`conv-transpose3d-max-unpool-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON を
コミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。入力・上流勾配 `g`・出力・
  入力勾配を保存する。損失は `(out * g).sum()`、dtype は float32、乱数は固定シード 2645、
  `torch.set_num_threads(1)`。再生成で同一の JSON（同一 sha256）になることを確認済み。
- `unfold_cases`（10 件）: k=2・s=1／k=3・p=1／stride 2／stride > kernel（窓の隙間あり）／dilation 2／
  非等方（kernel・stride・padding・dilation が軸ごとに異なる・N=2・C=3）／kernel=1／kernel=入力全体（L=1）／
  大きめ padding／N=3・stride と padding が軸ごとに異なる。
- `fold_cases`（11 件）: 上と対になる設定。各出力位置への寄与の最大値を PyTorch 側で実測し `overlapping` に記録
  （窓が重なる 6 件・重ならない 5 件）。テストは重なる設定を REQ-2 統一複合判定、重ならない設定を bit 一致で突合する。
- `nonfinite_cases`（4 件）: NaN／±inf／`-0.0` を含む入力（unfold 2 件・窓が重ならない fold 1 件・重なる fold で
  `inf + (-inf)` が NaN になる 1 件）。
- `error_cases`（19 件）: PyTorch が例外を出すか否かを実測して記録（`torch_raises`）。実測で確定した点:
  - バッチなし入力（`unfold` は `[C,H,W]`・`fold` は `[K,L]`）は受理される（本実装はバッチ入力のみ。差分）。
  - `N = 0` は受理される（`unfold`・`fold` とも）。
  - `C = 0`（`fold` は `K = 0`）は拒否される（本実装は空出力として受理。差分）。
  - kernel／stride／dilation の 0・カーネルが入力より大きい・`K` 非整除・`L` 不一致・`output_size` が小さすぎる／0・
    rank 不一致は拒否される。
  - 本実装との突き合わせは `docs/autodiff-fold-unfold-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > fold_unfold_reference.json
rm -rf /path/to/venv
```

## sha256

```
de76a901a822f811d414f2a208296d03cecaf35e6e6e13a2737911236d3f0bc3  fold_unfold_reference.json
2bd6619d34d7bcafd013a119204399b1349cb7949219eb9b367f205c20b4327c  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。エラーケースの期待表
（`tests/fold_unfold_parity.rs::err_spec`）の `torch_raises` は fixture と照合されるため、再生成後に食い違えば表を
更新する。
