# facade の `Device` への ROCm 追加 決定記録（所有者判断待ち）

イシュー #3157（親 #3150）。本書は実装の前に出す決定記録で、`Device` の variant 追加と facade の結線（Phase B）は **所有者が §6 の Q1〜Q3 に答えるまで実装しない**。コード変更はなく、公開面を固定する保留ガード（`crates/facade/tests/api_surface.rs`）と doc テストのみを足す。本書は所有者の承認を得ていない（承認を得たとは記載しない）。

## 1. 結論

- `Device` は `#[non_exhaustive]` ではない公開 enum である。variant を足すと下流の網羅的 `match` が壊れ、本リポの「破壊的」の定義（`docs/compat-api-scope.md` §145〜158）に当たる。非破壊に足せる形は見つからなかった（§3）。
- `backend-rocm` は `publish = false` で、公開クレートの `facade` が通常依存に持つと `cargo publish` が成立しない。
- `RocmBackendOps`（`docs/backend-rocm-ops-coverage-plan.md` の R0）は未実装・未起票。現時点では `tape_for(Device::Rocm(_))` は型付きエラーしか返せない。
- よって次の 3 点を所有者へ戻す: Q1 公開 API の契約、Q2 クレートの公開、Q3 R0 との境界（§6）。

## 2. 確認した事実（origin/main aaf8865c）

| 事実 | 出典 |
|---|---|
| `Device` は `Cpu`・`Cuda(usize)`・`#[cfg(target_os = "macos")] Metal`。`#[non_exhaustive]` なし。derive は `Debug, Clone, Copy, PartialEq, Eq, Hash` | `crates/tensor-core/src/device.rs` の `pub enum Device` |
| `DeviceInfo`・`BackendError` は `#[non_exhaustive]`。`BackendError` には `CudaUnavailable`・`DeviceUnavailable` が既にある | 同ファイル |
| facade は `Device` を再エクスポートしている（サポート対象の公開面） | `crates/facade/src/lib.rs` の `pub use fandhe_ai_tensor_core::{BackendError, Device, ...}` |
| `tape_for` は `resolve_ops` を呼び、`resolve_ops` は `Device` を網羅的に `match` する。`release_cached_memory`・`memory_pool_stats` も経由する | `crates/facade/src/lib.rs` |
| `available_devices` は Cpu → Cuda → (macOS) Metal の provider を束ねる。一貫性契約は「列挙された各 Device で `tape_for` が Ok」 | 同 `available_devices` の doc |
| 本番コードで `Device` を網羅的に `match` するのは `tensor-core::select_from` と `facade::resolve_ops` の 2 か所。backend-cpu・backend-cuda の `select` はワイルドカード腕を持つ | `git grep` の実測 |
| `Tape::device()`・`BackendOps::device()` は `Device` を返す | `crates/facade/src/lib.rs`・`tensor-core::backend_ops` |
| `backend-rocm` は tensor-core に依存せず、`BackendOps`・`DeviceProvider` を実装しない | `crates/backend-rocm/` |
| HIP のデバイス属性（gfx target・warp 幅）の取得は意図的に未実装 | `docs/backend-rocm-hiprtc-gemm-decision.md` |
| 公開クレートが非公開クレートへ依存する場合は、先に公開クレートへ昇格する先例 | `docs/facade-onnx-import-exposure-decision.md` §2 |

## 3. 候補の比較

| 案 | 内容 | 非破壊性 | 判定 |
|---|---|---|---|
| A | `Rocm(usize)` を無条件の variant として追加（`Cuda` と同じく動的ロードなので cfg なし） | 破壊的（下流の網羅的 `match` が壊れる。cargo-semver-checks の `enum_variant_added`） | 推奨形。契約変更のため承認が要る（Q1） |
| B | `#[non_exhaustive]` を付けてから足す | 付けること自体が破壊的（`compat-api-scope.md`） | A の付随案（将来の追加に備える）。Q1 に含める |
| C | `Device` を変えない別入口（`tape_for_rocm` 等） | 非破壊 | 成立しない。`Tape::device()`・`available_devices() -> Vec<Device>` が `Device` 値を要する |
| D | 独自 rustc cfg で variant を囲う | 既定ビルドでは非破壊 | 不採用候補。docs.rs から見えず `check-cfg` 設定も要り、facade → backend-rocm の公開問題も残る |
| E | 0.x の minor bump（0.12.0）で破壊を許容 | 版上は合法だが spec の「非破壊（追加のみ）」と食い違う | A の承認と同時に版数を所有者が決める |

### 推奨形（所有者が承認しやすい形）

- `Device::Rocm(usize)`。`usize` は HIP のデバイス ID（`hipSetDevice` の引数）。cfg は付けない。
- `available_devices` の順序は Cpu → Cuda → Metal → Rocm（既存の相対順を保つため末尾）。
- エラーは既存の `BackendError::DeviceUnavailable` を使い公開名を増やさない（`RocmUnavailable` の新設は `#[non_exhaustive]` なので非破壊で、Q1 の付随選択肢）。

## 4. 影響

- 下流の網羅的 `match`（非 macOS では `Cpu`・`Cuda` の 2 腕で網羅できる）がコンパイルエラーになる。
- `tensor-core::select_from`・`facade::resolve_ops` の 2 か所に腕を足す。`docs/public-api-design.md` §4.1・`tensor-core` のモジュール doc も更新する。
- `backend-rocm` の公開には crates.io 名の確認（`docs/crates-io-naming-decision.md`）、`docs/crates-io-publishing-order.md`、`release-all.yml` の `RELEASE_CRATES`、内部依存の `version` 併記が要る。公開物の推移的依存に `libloading`（第 12 区分。承認済み）が入る。
- `crates/facade/tests/api_surface.rs` の保留ガード（G1・G2）を反転する。

## 5. R0 より前の挙動の方針

`RocmBackendOps` と公開問題が片付くまで `available_devices` には載せない。これで一貫性契約（列挙された Device で `tape_for` が Ok）を崩さない。HIP 不在・範囲外 ID・R0 未結線のときは `tape_for(Device::Rocm(_))` が `DeviceUnavailable` を返し、panic しない。

## 6. 所有者への確認事項

- **Q1（公開 API の契約）**: 案 A（任意で B・E を併用）を「破壊的だが承認する」とするか、ROCm の facade 公開を見送るか。
- **Q2（クレートの公開）**: `backend-rocm` を 8 クレート目として公開するか。代替は facade の結線自体を見送ること。
- **Q3（R0 との境界）**: `RocmBackendOps` の骨格（R0）を先に起票・実装し、#3157 の Phase B をその後に置くか（被覆計画 §8 の提案どおり）。

## 7. 承認後の実装手順（Phase B。今回は実装しない）

1. 前提: R0 がマージ済み、`backend-rocm` の公開が承認済み。
2. `tensor-core/src/device.rs` に `Rocm(usize)` と `select_from` の `"rocm"` 腕を足す。
3. `backend-rocm` に tensor-core 依存と `RocmDeviceProvider` を足す（既存の `hip::load()` + `device_count()` を使い、新しい FFI・`unsafe` は足さない。HIP 不在なら `enumerate` は `Ok(vec![])`、`select` は `DeviceUnavailable`）。
4. facade の `resolve_ops`・`available_devices`・`Cargo.toml` を結線する。
5. `api_surface.rs` の G1 を `Rocm(_)` 込みに反転し、G2 を onnx-interop と同型の「承認済みの形」ガードへ反転する。
6. doc テストで `Device::Rocm(0)` が列挙されないとき `tape_for` が `Err` になることを確認する。
7. 新しい `unsafe` が出たら `docs/backend-rocm-unsafe-audit.md` に追記し、ルート `Cargo.toml`・`backend-rocm` の「facade からは参照しない」コメントを更新する。

## 8. 未確認事項

- HIP のデバイス ID と gfx target の取得方法（実機での確認は #3159）。
- 下流で `Device` を網羅的に `match` している利用の実態。
