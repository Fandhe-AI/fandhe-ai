# LocalResponseNorm・weight_norm・spectral_norm PyTorch 参照値フィクスチャの出自

イシュー #2646（親 #2625）の `tests/lrn_parity.rs`・`tests/weight_reparam_parity.rs` が参照する固定フィクスチャ。
`pool3d-pytorch-reference/README.md` と同じ方針で、生成条件・sha256 を記録した状態で JSON をコミットする。CI は
Python／PyTorch に依存せず、コミット済み JSON のみを読む（`.claude/rules/ci.md`）。

## 生成条件

- **実 PyTorch 実行値**。`torch.__version__ == "2.14.0+cpu"`・Python 3.14（JSON の `torch_version`／`python_version` に
  記録。スクリプトは 2.14.0 系でなければ assert で止まる）。numpy や手計算値で代替していない（numpy 未導入の使い捨て
  venv で実行）。`torch.set_num_threads(1)`・固定シード 2646。
- **f32 はすべて u32 のビットパターン配列で保存**する（JSON は NaN／inf を運べないため）。入力・上流勾配・出力・入力勾配を
  すべて保存し、Rust 側で入力を再生成しない。損失は `(out * up).sum()`、dtype は float32。
- `lrn_cases`（21 件）: `F.local_response_norm` の rank 3／4／5・`size` 奇数・偶数（2・4・6・8）・1・`size > C`・
  `size == C`・`C = 1`・既定値と非既定の `alpha`／`beta`（0・0.5・1・2）／`k`（0 を含む）・`alpha = 0`・負の `alpha`。
  `lrn_nonfinite_cases`（5 件）: NaN 単数・複数チャネル、`+inf`、`-inf`、NaN と inf の混在。
  `lrn_error_cases`（10 件）: rank 1・2、`size = 0`、`N = 0`・`C = 0`・空間軸 0、非有限の `alpha`／`beta`／`k` について、
  torch が例外を出すか否かを実測して記録（`torch_raises`）。実測で確定した点:
  - torch は rank < 3（`ValueError`）と `size = 0`（`RuntimeError`）を拒否し、`N = 0`・`C = 0`・空間軸 0 は受理する
    （空出力）。非有限の `alpha`／`beta`／`k` は無検査で受理する（本実装は拒否。差分）。
- `weight_norm_cases`（17 件）: `torch._weight_norm(v, g, dim)`・`torch.norm_except_dim(v, 2, dim)` を直接呼ぶ。rank 1／2／3／4・
  `dim = 0`／`1`／`2`／`3`／全体（torch の `dim = -1`）・`g` が `norm_except_dim(v)` に等しい／スケール／微小／負・
  軸長 1。`v`・`g` 両方の勾配と `norm_except_dim` の値・shape も保存（`norm_shape`／`norm_bits`）。
  `weight_norm_nonfinite_cases`（4 件）: ゼロノルムのグループ・全体ゼロ・NaN・inf。
  `weight_norm_error_cases`（6 件）: `dim` 範囲外・rank 0・`g` の shape 違い 4 種。実測で確定した点:
  - torch は `dim` 範囲外と rank 0 を拒否する（`IndexError`）。**`g` の shape は検査しない**（`[3]`・長さ違いの `[2, 1]`・
    別軸の `[1, 4]`・`dim = -1` に `[1, 1]` を渡しても受理）。本実装は `norm_except_dim` の出力 shape と完全一致のみ
    受理する（差分）。
  - `norm_except_dim(v, 2, d)` は `d` 以外が 1 の keepdim 形（rank 1 は `[n]`）、`d = -1` は 0-dim を返す。
- `spectral_cases`（18 件。training 15・eval 3）: `_SpectralNorm(weight, n_power_iterations, dim, eps)` を直接生成し、
  **初期化直後の `u0`／`v0`（乱数初期化＋予備反復 15 回後）と、初期化に使った重みとは別の重み `w` で forward を 1 回だけ
  呼んだ後の `u1`／`v1`** の両方を保存する（training では forward ごとに反復が走るため 1 記録につき forward は 1 回）。
  rank 2／3／4・`dim = 0`／`1`／`2`／`3`・`n_power_iterations = 1`／`2`／`3`・重みのスケール違い・`eps` が効くケース・
  `1×N`／`N×1`。出力・`weight` 勾配も保存。初期化乱数は Rust 側で再現しない（`u0`／`v0` から状態を作る）。
  特異値の分離を要求しない（同一の `u0`／`v0`・`w`・反復回数に対する決定的な写像を突合するため）。
  `spectral_error_cases`（6 件）: rank 1・`n_power_iterations = 0`・`dim` 範囲外・要素数 0・負／NaN の `eps`。実測で確定した点:
  - torch は `n_power_iterations = 0`（`ValueError`）と `dim` 範囲外（`IndexError`）を拒否する。rank 1（`F.normalize` へ縮退）・
    要素数 0・負／NaN の `eps` は受理する（本実装は拒否。差分）。
  - 本実装との突き合わせは `docs/autodiff-lrn-weight-reparam-decision.md` §5。

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch==2.14.0
/path/to/venv/bin/python gen_reference.py > lrn_weight_reparam_reference.json
rm -rf /path/to/venv
```

## sha256

```
a6afb75dd145751072abe52d432562c8d207a8b6accd8423a1ee15c5a7d1b456  lrn_weight_reparam_reference.json
3832abf97496c4917b8b8cb07a3c7e3b2c08c41f3e56edf90b52969c8e45662d  gen_reference.py
```

`gen_reference.py` を変更した場合は JSON を再生成し、上記を更新する（2 回生成して同一 sha256 になることを確認済み）。
