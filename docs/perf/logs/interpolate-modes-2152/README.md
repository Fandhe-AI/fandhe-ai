# interpolate の残り 5 モード（#2152）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-interpolate-modes-decision.md` 参照。本実装エージェント
実行環境は CUDA／Metal 実機に到達できないため、`crates/facade/tests/
interpolate_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする
**計 2 テスト**（`metal_interpolate_new_modes_forward_matches_cpu`／
`cuda_interpolate_new_modes_forward_matches_cpu`。各テストが 5 モード
〈`NearestExact`／`Area`／`Linear`／`Trilinear`／`Bicubic`〉を内部で
ループして検証する構成）は `#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test interpolate_backend_parity -- --ignored --nocapture cuda_interpolate_new_modes

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test interpolate_backend_parity -- --ignored --nocapture metal_interpolate_new_modes
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_interpolate_nearest_exact_matches_naive_reference`・
`cpu_interpolate_area_matches_naive_reference`・
`cpu_interpolate_linear_matches_naive_reference`・
`cpu_interpolate_trilinear_matches_naive_reference`・
`cpu_interpolate_bicubic_matches_naive_reference`）で既に検証済み
（green。CPU ネイティブとホスト参照〈NaiveOps〉が forward・backward
とも bit 完全一致することを確認済み）。

## 期待結果

CUDA／Metal は本イシューでカーネル実装をしておらず、`BackendOps::
interpolate` が新規 5 モードに対して `Unsupported` を返すため
`grad::interpolate_with_fallback` が必ずホスト参照実装
（`autodiff::eval::interpolate_*`）へフォールバックする。したがって
CUDA tape・Metal tape 経由の forward 値も CPU tape の値（同じホスト
参照実装を経由）と **bit 完全一致するはず**である（テストの判定自体
は既存の `assert_parity`〈REQ-2 統一複合判定〉で行い、tolerance は
変更しない——モードによっては将来専用カーネルを実装した際に丸めが
変わりうるため、判定契約自体は複合判定のまま保つ設計）。

この前提が崩れる場合（`BackendOps::interpolate` の未対応 variant
フォールバック分岐が変わった、デバイス間で想定外の丸めが混入した等）
は本 README の「期待結果」を更新し、想定した契約を維持できない事実を
型付き findings として PR へ記録すること（tolerance の単独緩和は
行わない。`.claude/rules/coding-rust.md`）。
