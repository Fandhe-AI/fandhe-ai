# reduce_ops（#2147）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-reduce-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::reduce_ops`（`prod`／`logsumexp`／`any`／`all`／
`norm_p`）の `crates/facade/tests/reduce_ops_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする 8 テスト（forward／backward × 4 種類 ×
CUDA／Metal）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test reduce_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test reduce_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_any_all_forward_bit_matches_naive_reference`・
`cpu_any_all_gradient_is_bit_exact_zero`・
`cpu_prod_logsumexp_norm_p_forward_matches_naive_reference_within_tolerance`・
`cpu_prod_logsumexp_norm_p_backward_matches_naive_reference_within_tolerance`）
で既に検証済み（green）。

## 期待結果

- `any`／`all` の forward・backward（勾配ゼロ）は出力が厳密に
  `0.0`／`1.0` のみで縮約順序に依存しないため、CUDA／Metal でも CPU
  と **bit 完全一致**するはず（`x.ne(&zero)` の比較カーネルは既存
  `scalar_binary_with_fallback` を経由・`max`／`min` は縮約だが
  0/1 のみの入力に対しては丸め誤差の入る余地がない）。
- `prod`（`cumprod` の合成）・`logsumexp`・`norm_p` の forward・
  backward は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
  1e-5 未満）で比較する。`logsumexp`／`norm_p` はいずれも
  `BackendOps::logsumexp`／`vector_norm_p` が既定 `Unsupported` を
  返すため、実機テストは常にホスト参照実装（`eval::logsumexp_along`／
  `vector_norm_p_along`）へフォールバックする経路になる想定——CUDA／
  Metal 側に専用カーネルが無いため、この 2 演算の CPU との差は
  「同じホスト参照実装をどのデバイスの `Tape` 経由で呼ぶか」の違いに
  留まり、数値的な乖離は生じにくいと予想される。

この前提が崩れる場合（想定外の丸めが混入した、フォールバック経路が
機能していない等）は本 README の「期待結果」を更新し、REQ-2 判定を
外れた事実を型付き findings として PR へ記録すること（tolerance の
単独緩和は行わない。`.claude/rules/coding-rust.md`）。
