# extremum_ops::amax／amin（#2154）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-amax-grad-distribution-decision.md` §9「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::extremum_ops`（`amax`／`amin`）の
`crates/facade/tests/extremum_ops_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする 4 テスト（forward／backward × 2 種類 ×
CUDA／Metal）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test extremum_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test extremum_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_amax_amin_forward_bit_matches_naive_reference`・
`cpu_amax_amin_gradient_bit_matches_naive_reference`・
`cpu_amax_rejects_huge_broadcast_before_materializing`）で既に
検証済み（green）。

## 期待結果

- **forward** は `Op::Max`／`Op::Min` の forward 経路をそのまま再利用
  する（`amax`／`amin` は専用 `BackendOps` メソッドを追加していない）
  ため、CUDA／Metal でも CPU と **bit 完全一致**するはず——`Var::max`／
  `min` の既存 CUDA／Metal カーネル（`min` は `BackendOps::min` が
  `Unsupported` を返す場合ホスト参照実装 `eval::min` へフォールバック）
  の数値契約をそのまま継承する。
- **backward**（`grad::extremum_even_split_vjp`）は `input`／
  `out_value` から `g / k`（`k`: タイ数の整数カウント）を計算する
  **純粋なホスト側計算**であり、CUDA／Metal 用の専用カーネルを持たない
  （`grad.rs::vjp()` は `materialize_fallible` でホスト側 `Tensor<f32>`
  を取得してから本関数を呼ぶ。バックエンドに依らず同一のホスト計算
  経路を通る）。したがって CUDA／Metal での backward も CPU と
  **bit 完全一致**するはずである（`f32` 除算 1 回のみで縮約を伴わない
  ため丸め誤差の入る余地がない。`docs/autodiff-amax-grad-distribution-
  decision.md` §5 の数値契約参照）。

この前提が崩れる場合（想定外の丸めが混入した、`min` のフォールバック
経路が機能していない等）は本 README の「期待結果」を更新し、bit 不一致
の事実を型付き findings として PR へ記録すること（tolerance の単独緩和
は行わない。`.claude/rules/coding-rust.md`）。
