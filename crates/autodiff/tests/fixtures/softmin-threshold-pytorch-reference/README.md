# Softmin・Tanhshrink・Threshold・RReLU PyTorch 参照値フィクスチャの出自

イシュー #2650（親 #2648・Phase 親 #2625）の `tests/softmin_threshold_parity.rs` が
参照する固定フィクスチャ。`trig-ops-pytorch-reference/README.md` と同じ方針で、生成条件・
sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に依存せず、
コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14.4
  （JSON の `torch_version`／`python_version` に記録）。numpy や手計算値で代替していない。
- 損失は `(out * g).sum()`（`g` は固定シード `2650` の一様乱数 `[-1, 1)`）。
- 入力・上流勾配・出力・入力勾配・shape・パラメータをすべて JSON に保存する
  （Rust 側で入力を再生成しない）。dtype は float32。`cases` は NaN／inf を含まない
  こと、`x == threshold`・`x == 0` ちょうどの要素を含まないことを生成時に前提とする
  （境界は `edge_cases` で扱う）。
- `cases`（計 33 件）:
  - `softmin`: 形状 `[5]`・`[2,3]`・`[2,3,4]` の各軸（6 件）
  - `tanhshrink`: `[6]`・`[2,3]`・`[2,2,2]`（3 件）
  - `threshold`: 形状 3 種 × パラメータ 3 組（正負の `threshold`・`value`。9 件）
  - `rrelu_eval`（推論）: 形状 3 種 × `(lower, upper)` 3 組（既定値・任意範囲・
    `lower == upper`。9 件）
  - `rrelu_train`（学習）: 形状 3 種 × 2 組（6 件）。乱数列が PyTorch と一致しないため、
    `F.rrelu(training=True)` を 1 回だけ計算し、`autograd.grad(y.sum(), x)` で同じ標本の
    noise を取得して保存する（`y == x * noise` を生成時に assert 済み）。Rust 側は
    この noise を `rrelu_with_noise` へ渡して突合する。
- `edge_cases`（8 件）: NaN／inf は JSON で表せないため、入力・出力・勾配とも要素ごとに
  `{"class": nan|pos_inf|neg_inf|finite, "value"}` で保存する（勾配は上流勾配 1 で測定）。
  主な実測クラス:
  - `threshold`: NaN は素通し（出力 NaN・勾配 1）、`+inf` も素通し（勾配 1）、
    `x == threshold` は `value` へ置換（勾配 0）、`-inf` は置換（勾配 0）
  - `rrelu` 推論: `x > 0` は恒等、`x <= 0`（`±0`・NaN・`-inf` を含む）は傾き `(lower+upper)/2`
    を勾配に持つ（`x == 0` と NaN の勾配が傾き）
  - `rrelu` 学習: `x == 0` の勾配は noise（傾き）、NaN の勾配は 1（noise = 1）
  - `softmin`: `+inf` を含む行は当該要素が 0、`-inf` を含む行は全要素 NaN、
    `±1e30` で桁あふれせず、全要素同値は一様
  - `tanhshrink`: `±inf` の出力は `±inf`（勾配 1）、NaN は NaN、`|x| = 1e-4` の勾配は 0

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > softmin_threshold_reference.json
rm -rf /path/to/venv
```

## sha256

```
2200434de67dda41bfac1f3224677be1410c1b6301cff5ce4ee14dde8be0f25c  softmin_threshold_reference.json
42de09cad87e58e800bb7a3d1ab8f03a856832f9d04411a657c80f84ebdee22f  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
