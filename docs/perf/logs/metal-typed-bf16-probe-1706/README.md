# Metal bf16 コンパイルプローブ（イシュー #1706）実測ログ

`crate::typed_bf16_probe_diag_tests`（`crates/backend-metal/src/
typed_bf16_probe_diag_tests.rs`）の `#[ignore]` テスト（P0〜P4。MSL
`bfloat` 型・`simdgroup_bfloat8x8` の実機コンパイル可否を調査する非 gating
プローブ）を Apple Silicon 実機で実行するログ置き場。

**実測済み（2026-09-16・Apple M4 Max・origin/main `3e43bbd0`）**: `typed_bf16_probe.log`・`typed_ops_bf16_parity.log`・`env_info.txt` を収めた。結果の転記は `docs/backend-dtype-dispatch-design.md` §15.6。以下は記入欄作成時（本 PR 時点）の記述をそのまま残す。

**現状（本 PR 時点）**: このディレクトリはまだ空。本 PR を書いた実行環境
（Linux）には Apple Silicon 実機への到達手段がないため、下記の実測は
未実施のまま記入欄として残す（`docs/backend-dtype-dispatch-design.md`
§14.6・`typed_bf16.rs` モジュール doc「(a) と (b) の分離」参照）。Mac
セッションで下記コマンドを実行し、生成物をこのディレクトリへ収める。

## 位置づけ

本プローブの結果は `crate::typed_bf16`（`impl TypedOps<half::bf16> for
MetalBackendOps`。ホスト側変換＋既存 f32 カーネル委譲）の実装可否には
**影響しない**（そちらは本イシューで実装済み・§14.2 参照）。本プローブが
決めるのは、将来のデバイス常駐ネイティブ bf16 経路（bf16 のまま H2D・
`simdgroup_bfloat8x8` GEMM 等）の実現可能性という別の問いのみである。

## 実行手順

```sh
cargo test -p fandhe-ai-backend-metal --release \
  --lib typed_bf16_probe_diag_tests -- --ignored --nocapture \
  2>&1 | tee typed_bf16_probe.log
```

（`--lib` 経由。プローブは `src/typed_bf16_probe_diag_tests.rs` に
`cfg(all(test, target_os = "macos"))` で配置されており、integration test
ではなくクレート内部テストモジュールとして走る）

## 生成物

- `typed_bf16_probe.log`: 上記コマンドの全出力。次の 5 テストの結果と
  `println!` 記録（`bf16_probe label=... ...` 形式）を含む
  - `p0_device_attributes`（デバイスアーキテクチャ名・
    Apple7/8/9・Metal3 family 対応）
  - `p1_bfloat_scalar_compile_probe`（`bfloat` スカラー・`bfloat4`・
    変換のコンパイル可否。既定言語版・`Version3_1` の 2 条件）
  - `p2_simdgroup_bfloat_mma_compile_probe`（`simdgroup_bfloat8x8` の
    `simdgroup_load`／`simdgroup_multiply_accumulate` コンパイル可否）
  - `p3_simdgroup_store_bfloat_compile_probe`（`simdgroup_store(float8x8,
    device bfloat*)` の型不一致でのコンパイル可否。f16 版〈#380〉と
    同様に不可の見込み）
  - `p4_bfloat_roundtrip_numeric_smoke`（P1 が可の場合のみ意味を持つ
    数値スモーク。ホスト `half::bf16::from_f32` との bit 一致件数・
    不一致件数）
- `env_info.txt`: `uname -srm`・`sw_vers`・`rustc -V`・
  `sysctl machdep.cpu.brand_string`・実行コミットの `git rev-parse HEAD`
  （内部ホスト名は含めない）

## 判定基準

本プローブは非 gating（成否を記録するだけで pass/fail 判定を行わない）。
実測完了後、結果を `docs/backend-dtype-dispatch-design.md` §15.6（bf16 の記入欄。§14.6 は f16〈#1705〉向け）へ転記し、
(b) の可否（デバイス常駐ネイティブ bf16 経路の実現可能性）を記録する。
