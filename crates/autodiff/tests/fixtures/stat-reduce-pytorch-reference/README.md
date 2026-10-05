# 順序統計・NaN 無視縮約（median・kthvalue・quantile・nansum・nanmean）PyTorch 参照値フィクスチャの出自

イシュー #2637（親 #2625）の `tests/stat_reduce_parity.rs` が参照する固定フィクスチャ。
`cumulative-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で
JSON をコミットする。CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む
（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4（JSON の
  `torch_version`／`python_version` に記録。スクリプトは 2.14.0 系でなければ assert で止まる）。
  numpy や手計算値で代替していない（numpy 未導入の使い捨て venv で実行）。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。
  入力 `x_bits`・上流勾配 `g_bits`・出力 `out_bits`・入力勾配 `grad_bits`・`quantile` の `q_bits`、
  `median`（軸指定）／`kthvalue` は索引 `index`（torch の int64）も保存する。損失は `(values * g).sum()`、
  dtype は float32、乱数は固定シード 2637。
- `finite_cases`（436 件）: タイのない入力（軸長 1・奇数長・偶数長・バッチ・rank 3 で `dim=0/1/2`・
  `dim=None`）に対する全 6 演算。`kthvalue` は `k=1`・中間・`k=n`、`quantile` は 5 補間 × `q ∈ {0, 0.1, 0.25,
  0.5, 0.75, 1.0}`。加えて `quantile` の **rank 精度の判別ケース**（`n=11`・`n=21` で `f32` の積なら整数ちょうど、
  `f64` なら僅かに外れる `q`、`n=5` の `.5` ケース）。
- `tie_cases`（129 件）: 0.5 刻みのタイ・全要素同値・`±0` 混在。タイ時の索引は PyTorch が規定しないため
  記録用（受入判定は `tests/stat_reduce_parity.rs` 冒頭と `docs/autodiff-stat-reduce-ops-decision.md` §5）。
- `nan_cases`（141 件）: NaN の先頭・中間・末尾・複数・全 NaN・バッチ内の一部 lane のみ NaN。
- `inf_cases`（186 件）: `+inf`／`-inf`／混在／NaN と混在。
- `nonfinite_upstream_cases`（12 件）: `nansum`／`nanmean` に `inf`／NaN／0 の上流勾配を流したときの
  NaN 位置・全 NaN lane の勾配。
- `error_cases`（28 件）: 軸長 0・要素数 0・`k=0`・`k>n`・`q` 範囲外／NaN・`dim` 範囲外・0 次元入力・負の `dim`
  で torch が例外を出すか否か（`torch_raises`）と出力 shape。実測の要点は
  `docs/autodiff-stat-reduce-ops-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > stat_reduce_reference.json
rm -rf /path/to/venv
```

## sha256

```
89f617e5e5594afc64502b60abf26acf6ed47ffcb6bcfe5d3cfbc71477cba183  stat_reduce_reference.json
449f11d9d991654542a56311ad32480440f67c288c0e21c1a52055c9ee6ae544  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
