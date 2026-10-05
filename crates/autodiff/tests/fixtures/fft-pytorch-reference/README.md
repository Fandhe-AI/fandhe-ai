# rfft／irfft PyTorch 参照値フィクスチャの出自

イシュー #2631（親 #2630）の `tests/fft_parity.rs` が参照する固定
フィクスチャ。`radam-pytorch-reference/README.md` と同じ方針で、生成条件・
sha256 を記録した状態で JSON をコミットする。CI は Python／PyTorch に
依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python
  3.14.4（JSON の `torch_version`／`python_version` に記録）。numpy や手計算
  値で代替していない。
- 複素 autograd の規約差を避けるため、複素数は `torch.view_as_real`／
  `torch.view_as_complex` で末尾次元 2 の実テンソル対として扱い、損失は
  実数 `(out * g).sum()`（`g` は固定シード `20261005` の `torch.randn`）。
  - rfft: 実の葉 `x` → `view_as_real(rfft(x, n, dim, norm))`・`x.grad`
  - irfft: 実の葉 `xr`（`[..., m, 2]`）→ `irfft(view_as_complex(xr), n, dim,
    norm)`・`xr.grad`
- 入力・上流勾配・出力・入力勾配・shape・引数をすべて JSON に保存する
  （Rust 側で入力を再生成しない）。dtype は float32。生成時に NaN を含まない
  ことを assert 済み。
- ケース（`gen_reference.py::RFFT_CASES`／`IRFFT_CASES`、計 33 件）:
  - rfft: `L ∈ {1,2,3,4,5,8}`・norm 3 種・ゼロ詰め（6→8・3→7）・切り詰め
    （8→5・5→2）・バッチ付き／非末尾 `dim`（`[2,5,3]` で `dim=1` ほか）
  - irfft: 既定 `n`（偶数）・norm 3 種・明示の奇数 `n`（`m=4,n=5`）・
    `m > n/2+1`（切り詰め）・`m < n/2+1`（ゼロ詰め）・`m=1` で `n=1,2,3`・
    バッチ付き／非末尾 `dim`。入力の虚部は一様乱数のため DC・Nyquist の
    虚部は非ゼロで、無視されることが突合される
- `error_cases`: torch が例外を出すか否かを実測して記録する境界ケース
  （Rust 側の拒否方針との突き合わせ用）。実測結果:
  `rfft n=0`・`rfft dim` 範囲外・`rfft` 空軸で `n` 省略・`irfft` で `m=1` かつ
  `n` 省略・`irfft n=0`・`irfft dim` 範囲外は torch が例外。**`rfft` 空軸
  （長さ 0）で `n=4` を明示した場合は torch は受理する**（ゼロ詰め）。本実装も
  同じく受理する。なお `irfft` の入力 bin 数 `m=0` は本実装が拒否する
  （torch の挙動は未実測。`docs/autodiff-fft-ops-decision.md` §5）。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > fft_reference.json
rm -rf /path/to/venv
```

## sha256

```
30abdc78f514b993608b8a69370eb909e8d32b5be3e1c587572acd15a04fbd2b  fft_reference.json
c9cfd520111cd5840f6d0f6a77a22c53f56e19aa78ea6f4fd2ee8c1bad6a40a1  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する。
