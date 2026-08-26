# バックエンド構成

## feature フラグなしの cfg ベース切替

`fandhe-ai` はバックエンド（CPU・CUDA・Metal）の切替に Cargo
feature を使いません。**`target_os`／条件付きコンパイル（`cfg`）だけで
切り替わる構成**です。

| バックエンド | 依存の分離方法 | 備考 |
|---|---|---|
| CPU（`rayon`） | 無条件依存 | 常に利用可能な既定バックエンド |
| CUDA（`cudarc`） | 無条件依存＋動的ロード（`dynamic-loading` feature） | CUDA toolkit 非搭載環境でも `cargo build` が成立し、toolkit の要求は実行時のみ |
| Metal（`objc2`・`objc2-foundation`・`objc2-metal`） | `[target.'cfg(target_os = "macos")'.dependencies]` 分離 | 非 macOS ではコード・依存ごとビルド対象から外れる |

feature の組合せが増えるほど CI の検証マトリクスが組合せ的に増加し、
feature 指定漏れによる経路欠落が起きえます。cfg ベースはターゲットから
自動的に決定されるため、この失敗モード自体が存在しません。設計の詳細・
PoC 実測（PoC-v2-1／PoC-v2-3／PoC-v2-5）根拠は
[`docs/backend-switching-design.md`](https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/backend-switching-design.md)
を参照してください。

## `tape()` と `tape_for(Device)` の使い分け

利用者が composition root（`Device` → 具体バックエンドの結線）に触れる
入口は 2 関数だけです。

- `fandhe_ai::tape()`: 常に利用可能な既定バックエンド（CPU）で `Tape` を
  構築します。デバイスの存在検証が不要なため非 fallible（`Result` を
  返しません）。
- `fandhe_ai::tape_for(Device)`: `Device::Cpu`／`Device::Cuda(ordinal)`／
  `Device::Metal`（macOS 限定）を明示的に指定します。CUDA・Metal は
  構築時にドライバ・デバイスの存在検証を行い、`Result<Tape, BackendError>`
  を返します。

`crates/facade/examples/backend_switching.rs`（`cargo run -p fandhe-ai
--example backend_switching` で実行確認済み。転記コードは
[Getting Started の「バックエンド切替」節](/getting-started/)参照）が
示すとおり、`Device::Cuda(0)` が失敗した場合の CPU フォールバックは
呼び出し側が `Result` を見て自分で書く処理です。

## fail-fast 設計（自動フォールバックはしない）

`Device::Cuda(ordinal)`／`Device::Metal` はいずれも構築時にデバイスの
存在を検証し、ドライバ不在・範囲外 ordinal の場合は
`BackendError` を返します。**`fandhe-ai` はデバイスが利用できないときに
自動的に別のバックエンドへフォールバックすることはしません。**
フォールバックが必要なら、呼び出し側で `Result` を見て分岐してください。

`Device::available()` 相当の自動デバイス検出・列挙 API は現時点では
提供していません（サポート境界は
[API Reference](/api/)参照）。

## バックエンド間の丸め方針（FMA 契約）

CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal
`simdgroup_multiply_accumulate`）の既定 FMA 契約と揃えています。詳しい
数値一致の判定方法は [数値一致契約](/guides/numerical-parity/)を
参照してください。
