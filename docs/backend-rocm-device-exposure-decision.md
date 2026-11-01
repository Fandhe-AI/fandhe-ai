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

## 9. 訂正（#3720・2026-10-31）

§1〜§8 は決定記録として書き換えず、本節で訂正を追記する（先例: `docs/perf/framework-compare-feature-matrix-0.12.0.md` §8.5）。どれも誤記ではなく、記録後に起きた状態の変化である。本節も所有者の承認を得たものとは記載しない。

- 確認時点: base `7e04294e5`・2026-10-31T13:18Z（UTC）

### 9.1 訂正表

| 元の記述 | 訂正後の事実 | 根拠 |
|---|---|---|
| `:9`「`RocmBackendOps`（R0）は未実装・未起票」 | R0 は #3431 として起票され、2026-10-18 に実装・close 済み。R0〜R13（#3431〜#3444）も全件が CLOSED | `crates/backend-rocm/src/ops.rs:158`（`pub struct RocmBackendOps`）・`:727`（`impl BackendOps for RocmBackendOps`）、各 issue の closedAt |
| `:9`「`tape_for(Device::Rocm(_))` は型付きエラーしか返せない」 | 変わっていない。`Device::Rocm` は未追加で、facade は backend-rocm に結線されていない。`DeviceProvider` も backend-rocm には未実装 | `crates/backend-rocm/`・`crates/tensor-core/src/device.rs`・`crates/facade/Cargo.toml` に該当なし |
| `:23`「backend-rocm は tensor-core に依存せず、`BackendOps`・`DeviceProvider` を実装しない」 | いまは tensor-core に依存し（`crates/backend-rocm/Cargo.toml:28`）、`BackendOps` を実装している。`DeviceProvider` は未実装のまま | 同上 |
| `:52`「R0 未結線のとき」 | R0 は backend-rocm 内に実装済み。残るのは facade への結線（`resolve_ops`・`available_devices`）と `RocmDeviceProvider`。§5 の方針は「facade 未結線のとき」と読み替えれば変わらない | 同上 |

### 9.2 Q1〜Q3 の状態

- Q3（R0 との境界）: #3150 の所有者コメント（`#issuecomment-6248286174`・2026-10-18）が「§8 の境界は提案どおり」と判断し、R0 も実装済みのため実質解決した。§6 の本文は書き換えない。
- Q1（公開 API の契約）・Q2（クレートの公開）: まだ回答がない（`docs/compat-matrix-ari-tracker.md` 項目 24 の承認列は未承認・#3771 待ち。後続の設計は #3758）。
- §7 手順 1 の前提のうち「R0 がマージ済み」は満たされた。「`backend-rocm` の公開が承認済み」は満たされていない。

### 9.3 実機（#3159）

#3159 は 2026-10-31T13:18Z（UTC）の `gh issue view` で OPEN。実測は未了で、`docs/perf/logs/rocm-parity-3159/README.md:3` の「状態: 未実測」と矛盾しない。

### 9.4 本節で変えていないもの

§1〜§8 の本文・`crates/`・`api_surface.rs` の保留ガード（`device_variant_set_is_fixed_while_3157_on_hold`）。

## 10. ROCm の facade 公開の設計（イシュー #3758・0.14.0 互換方針）

本節は #3758（Phase 2・親 #3711）の設計記録で、書式は `docs/decision-record-template.md` の「保守・運用チェックリスト」9 節に従う。§1〜§9 は書き換えない。後続の #3851〜#3854・#3774・#3867 が引用する設計の正本とする。

- 承認: **未承認（#3771 待ち）**。Q1（公開 API の契約）・Q2（クレートの公開）はまだ回答がない。
- 引用してよい承認は、#3718 の所有者回答（<https://github.com/Fandhe-AI/fandhe-ai/issues/3718#issuecomment-6543681116>・2026-10-31）が決めた「破壊的変更のまとめ方」（`Device` だけを 0.14.0 で変える・版数は 0.14.0・spec 改訂提案は提案 1 だけ）に限る。ROCm を facade に公開する承認ではない。
- spec 側: 改訂提案 1 は Fandhe-AI/fandhe-ai-spec#87 として起票済みで未受理。#3714 §8 手順 3 により、spec の受理と #3771 の承認がそろうまで実装しない。
- 項目番号: #3758 本文は「項目 25」と書くが、追跡表（`docs/compat-matrix-ari-tracker.md`）と `docs/api-compat-0.14.0-decision.md` では ROCm は**項目 24**（25 は ONNX 学習）。本節は 24 を使う。
- 境界: 本 PR は設計のみ。コード・`api_surface.rs` のガード・依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変えない。spec REQ の ROCm は「Won't・条件付き」のままで、Could へ格上げされるまで REQ-2 の確定バックエンドに含めず、実機 CI も常設しない。facade への公開はこれを変えない（spec の行番号は submodule が動くため書かない）。

### 10.0 推奨

**実装する（案 A＋B＋E）**。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| A（推奨） | `Device::Rocm(usize)` を cfg なしで追加する（`Cuda` と同じ動的ロード構成） | R0〜R13 実装済みで、残りは結線に限られる | 公開 enum の variant 追加で破壊的（`enum_variant_added`） |
| B（A に付随・推奨） | 同じリリースで `Device` に `#[non_exhaustive]` を付ける | 以後の variant 追加が非破壊になる。#3714 §3.2「同一リリースで当該型の破壊が確定している型にだけ付ける」に当たる | 外部の網羅 `match` に `_` 腕が必須になる（`enum_marked_non_exhaustive`） |
| E（推奨） | 版数は 0.14.0 | workspace の lockstep。#3718 論点 2 の回答と一致 | なし |

C・D は §3 の不成立のまま。

理由: (1) #3718 の方針で破壊は 0.14.0 の 1 回にまとめると決まっている。(2) 残る作業は結線に限られる。(3) 見送ると、バックエンド行を「ある」にする道は (i) しか残らない。

見送る場合の比較:

| 見送りの形 | 内容 | 影響 |
|---|---|---|
| (i) 対象外へ移す | 所有者の承認 URL を添えて `docs/compat-api-scope.md` の対象外へ移す | バックエンド行を「ある」にできる。spec が ROCm を Won't とする点と整合 |
| (ii) 保留のまま | 現状維持 | 行は「部分的」に残り、0.14.0 の目標（9 行すべて「ある」）に届かない |

見送るなら (i) を推奨する。どちらでも保留ガード G1・G2 は承認まで維持する。

### 10.1 公開形

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device { Cpu, Cuda(usize), #[cfg(target_os = "macos")] Metal, Rocm(usize) }
```

- derive は不変（`Copy`・`Eq`・`Hash` を保つ）。`usize` は HIP のデバイス ID。
- 初期は **`Rocm(0)` だけを有効**とする。`RocmBackendOps` はデバイス 0 に固定されている（`crates/backend-rocm/src/ops.rs` の型 doc）。`Rocm(n≥1)` は `DeviceUnavailable`（fail-closed）。複数デバイスは論点 4。
- 新しい公開名は増やさない（`fandhe_ai::Device::Rocm` のみ）。`DeviceProvider`・`DeviceInfo`・`GfxTarget`・`RocmBackendOps` は facade から再エクスポートしない。

facade 内の配置:

| 箇所 | 変更 |
|---|---|
| `tensor-core::select_from` | `Device::Rocm(_) => "rocm"` の腕を足す（同一クレートなので `_` 腕なしで網羅を保つ） |
| `backend-rocm` に `RocmDeviceProvider` を新設 | 既存の `hip::load()`・`device_count()` と gfx target 判定を使う。HIP なしのとき `enumerate` は `Ok(vec![])`、`select` は `DeviceUnavailable`。`is_available` は `is_available()` と同じ判定 |
| `facade::resolve_ops` | `Device::Rocm(_)` の腕で `select_from` により存在を確かめてから `RocmBackendOps::new(target)` を構築。`#[non_exhaustive]` により別クレートでは `_` 腕が必須で、型付きの `DeviceUnavailable` を返す |
| `facade::available_devices` | macOS・非 macOS 双方の `providers` 配列の末尾に ROCm を足す。順序は Cpu → Cuda → Metal → Rocm |
| `tape_for`・`release_cached_memory`・`memory_pool_stats` | `resolve_ops` を通るため追加変更なし。ROCm はプールを持たず `Ok(())`・`Ok(None)` |
| `RocmBackendOps::device()` | 仮置きの `Device::Cpu`（`ops.rs`）を `Device::Rocm(0)` に差し替え、同ファイル内のテストも直す |

正ガードへの反転計画（#3714 §3.2 がテスト名の確定をこの記録に委ねている。#3853 が実施）:

| 現行（`crates/facade/tests/api_surface.rs`） | 反転後 |
|---|---|
| G1 `device_variant_set_is_fixed_while_3157_on_hold` | 削除し 2 本に置換。`device_variant_classify_with_wildcard_arm_includes_rocm`（`Rocm(_)` 腕と `_` 腕を持つ classify の正のプローブ）、`device_enum_variant_inventory_is_fixed`（`crates/tensor-core/src/device.rs` の `pub enum Device` の variant 名集合が {Cpu, Cuda, Metal, Rocm} と一致し、直前に `#[non_exhaustive]` があることをソース走査で確かめる。`pb_has_non_exhaustive_before` と同型） |
| G2 `facade_does_not_depend_on_backend_rocm_outside_dev_dependencies_while_3157_on_hold` | `facade_depends_on_backend_rocm_only_in_approved_shape` に反転。`facade_depends_on_onnx_interop_only_in_approved_shape` と同型で、素の `[dependencies]` に丁度 1 回（cfg なしの無条件依存）・`path = "../backend-rocm"`・`version` が tensor-core と同値・`git`／`registry`／`branch`／`rev`／`tag`／`optional`／`features`／別名／`package` を持たないことを検査 |
| `backend_rocm_guard_detects_header_style_dependency` | 承認形の合成入力に対する正と負のプローブ `backend_rocm_approved_shape_synthetic_inputs` に組み替える |

否定プローブは「backend-rocm の型が facade の公開面に出ない」「上記の再エクスポート禁止」を残す。

### 10.2 rustdoc と doctest

- `Device::Rocm` の doc に、役割（HIP デバイス ID・動的ロード・ROCm なしでもビルド可）、「条件付き・実験的。REQ-2 の確定バックエンド外」、初期は `Rocm(0)` のみ有効、を書く。
- `available_devices` の doctest は、網羅 `match` に `Rocm(_)` 腕と `_` 腕を足す形へ直し、「ROCm は未公開」の文言を更新する。
- 新しい doctest: `available_devices()` に `Rocm(0)` が含まれないときだけ `tape_for(Device::Rocm(0))` が `Err` になることを確かめる（ROCm 搭載ホストでは `Ok` なので無条件に `Err` を assert しない）。実機での演算例は `no_run`。
- 失敗系は `compile_fail` に頼らず、10.1 の在庫テスト（正のプローブ）と範囲外 ID の実行時テストで組む。

### 10.3 エラー型

- 既存の `BackendError::DeviceUnavailable` を使い、公開名は増やさない。`RocmUnavailable` の新設は `#[non_exhaustive]` なので非破壊だが既定では足さない（論点 6）。
- 検証順（fail-closed・`unwrap`／`expect` なし・HIP の確保より前）: (1) ordinal の範囲（`usize` → `i32` は `i32::try_from`、切り捨て防止）、(2) `hip::load()` の成否、(3) `device_count()` と ordinal の比較、(4) gfx target の allowlist 照合（`parse_gcn_arch_name` または論点 3 の判定）、(5) `RocmBackendOps` の構築。
- `RocmError` は facade に出さず `DeviceUnavailable(String)` へ写す。メッセージにパスやホスト名を入れない。

### 10.4 テスト計画

CI（`ubuntu-latest`・ROCm なし。`scripts/check-rocm-absent.sh` が非搭載を fail-closed で保証）:

| 受け入れ基準 | テスト | 置き場 |
|---|---|---|
| HIP なしで列挙が空 | `rocm_device_provider_enumerate_is_empty_without_hip` | `crates/backend-rocm/tests/device_provider_absent.rs` |
| HIP なしで `select` が型付きエラー | `rocm_device_provider_select_is_device_unavailable_without_hip` | 同上 |
| `available_devices` に `Rocm` が載らない（一貫性契約） | `available_devices_excludes_rocm_without_hip` | `crates/facade/tests/device_enumeration.rs` |
| `tape_for(Rocm(0))` が `DeviceUnavailable` | `tape_for_rocm_is_device_unavailable_without_hip` | 同上 |
| 範囲外 ID で panic しない | `tape_for_rocm_out_of_range_ordinal_is_err`（`usize::MAX`） | 同上 |
| G1・G2 の反転 | 10.1 のテスト群 | `api_surface.rs` |
| `RocmBackendOps::device()` が `Rocm(0)` | 単体テスト | `crates/backend-rocm/src/ops.rs` |

`#[ignore]`（実機依存。Phase 10 の #3867 が `<rocm-node>` で実行）:

| 受け入れ基準 | テスト | 置き場 |
|---|---|---|
| 列挙された `Rocm(0)` で `tape_for` が `Ok` | `rocm_available_devices_lists_rocm0_and_tape_for_ok` | `crates/facade/tests/rocm_facade_real_device.rs` |
| facade 経由の matmul・MLP の REQ-2 parity（CPU 比） | `rocm_facade_matmul_parity_vs_cpu`・`rocm_facade_mlp_train_step_parity_vs_cpu` | 同上 |
| 実機で範囲外 ID が型付きエラー | `rocm_tape_for_out_of_range_ordinal_is_err` | 同上 |

| 実測列 | 状態 |
|---|---|
| `<cuda-node>` | 未実測 |
| `<metal-node>` | 未実測 |
| `<rocm-node>` | 未実測（ログは #3867 が `docs/perf/logs/rocm-facade-3867/` に作る） |

#3159（backend 単体の parity）は facade に依存しない。推奨（論点 5）: 実装（#3851〜#3853）は Q1・Q2 の承認と spec#87 の受理がそろえば進めてよい。ただし追跡表の「0.14.0 の区分＝公開」と #3774 による crates.io 公開を 0.14.0 に含めるには、#3159 と #3867 の pass を条件にする。Phase 11 の時点で未実測なら所有者へ戻す。未実測を pass と書かず、cfg で 0 本しか実行されなかった結果も pass の根拠にしない。

### 10.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を適用する。`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えない。
- 結線のみのため FMA 契約・f64 長軸縮約の契約に影響しない（backend-rocm の既存カーネルの契約をそのまま使う）。
- baseline の追加・更新はない。必要になれば実機の実測値に限り所有者承認を得る。

### 10.6 互換への影響（#3714 §8 の判定）

| 型 | 判定 |
|---|---|
| `Device`（項目 24） | **破壊的で承認要**（`enum_variant_added`・`enum_marked_non_exhaustive`）。承認は #3771、spec の前提は spec#87 の受理 |
| `BackendError` | 追加のみ（既に `#[non_exhaustive]`・variant は増やさない） |
| `available_devices`・`tape_for` | シグネチャ不変。戻り値に新しい値が増える（移行ノートに挙動の変化として書く） |
| `release_cached_memory`・`memory_pool_stats` | 同上 |

`_` 腕が要る箇所の在庫（#3853 の変更集合）。#3714 §3.2 は「`_` 腕が要るのは `resolve_ops` だけ」と書くが、テスト側を含めると次になる。

- `crates/facade/src/lib.rs` の `resolve_ops`
- `available_devices` の doctest
- `api_surface.rs` の G1
- `crates/facade/tests/rnn_stacked_backend_parity.rs`・`spatial_layers_backend_parity.rs`・`transformer_decoder_backend_parity.rs`・`gpu_elementwise_fusion_bench.rs` の `Device` の網羅 `match`（#3853 が `grep` で再確認する）

`device_enumeration.rs` は既に `_` 腕がある。framework-compare は `fandhe-ai =0.13.0` 固定で影響を受けない。移行ノート（`docs/migration/0.14.0.md`）の作例は #3714 §5 に従い、実装 PR は `type(scope)!:` と `BREAKING CHANGE:` を付ける。

### 10.7 適用記録・索引・追跡表

適用記録の文案（#3854 が `docs/compat-api-scope.md` へ転記。承認 URL は承認後に入れる）:

> **適用記録（イシュー #3854・親 #3711。承認: <#3771 の回答 URL>）**: facade の Device に Rocm(usize) を追加し #[non_exhaustive] を付与（0.14.0・破壊的）。backend-rocm を fandhe-ai-backend-rocm として crates.io 公開。初期は Rocm(0) のみ・条件付き実験的。公開状態: 未公開

見送り (i) の場合は、同じ書式で「ROCm の facade 公開を対象外へ移す（承認: <URL>）」とする。

| 更新対象 | 担当 |
|---|---|
| `docs/compat-api-scope.md` の適用記録 | #3854 |
| `docs/README.md` の索引 | 本 PR と #3854 |
| 追跡表の設計記録セル | 本 PR |
| 追跡表の承認セル | 承認後の最初の PR |
| 追跡表の正ガードのセル | #3853 |
| 追跡表の実機 parity のセル | #3867 |
| `docs/public-api-design.md` §4.1・tensor-core のモジュール doc・facade の `description`（「CPU/CUDA/Metal」）・`crates/backend-rocm/README.md` の「非公開」「仮置き」記述 | #3854 |

### 10.8 ベンチと性能への影響

- 既定経路（`tape()`・`Device::Cpu`、CUDA と Metal の `tape_for`）は match の腕が増えるだけで変わらない。
- `available_devices` に `RocmDeviceProvider::enumerate` が増え、ROCm なし環境では dlopen の試行が 1 回増える。ROCm 搭載環境では論点 3 の判定コストも増える（(c) は code object を 1 回ロードし結果をキャッシュ）。
- 前後比較: `available_devices` の呼び出し時間を ROCm なしの ubuntu で 5 回計測し中央値を比べる。規則は計測前に固定する（事前登録）。担当は #3853。ROCm 搭載環境の値は #3867 で「未実測」から始める。

### 10.9 依存の追加の有無と Q2 の公開手順

- 外部依存の追加は**なし**。`libloading =0.9.0` は第 12 区分で承認済みで、cudarc の推移的依存として lock に既にある。新しく生じるのは「公開クレート `fandhe-ai-backend-rocm` の直接依存として crates.io に出る」点のみ（deps-policy の「利用クレートは backend-rocm だけ」は不変）。facade の公開物の依存木への影響は #3774 が `cargo tree -p fandhe-ai -e normal` で実測する。禁止リストとの照合は該当なし。
- 新規 `unsafe` は論点 3 の選択で変わる。(c) は既存の FFI のみでなし。(b) は HIP のシンボル追加・`docs/backend-rocm-unsafe-audit.md` への追記・security-auditor の監査が要る。§7 手順 3「新しい FFI・`unsafe` は足さない」は、gfx target の照会と両立しない場合があるため、ここで訂正として書く。
- `docs/license-matrix.md` は外部依存の追加がないため更新不要の見込み（#3774 が実測して確かめる）。

Q2 の手順（#3774）:

1. `docs/crates-io-naming-decision.md` の「確認方法」で `fandhe-ai-backend-rocm` の空きを実測する（本 PR の時点では未実測）。
2. `Cargo.toml` に `publish = true`・英語の `description`・`repository.workspace`・`readme`・`keywords`・`categories` を入れ、`README.md` を利用者向けに書き直し、`LICENSE-APACHE`・`LICENSE-MIT` を同梱する。
3. `[dependencies]` の `fandhe-ai-tensor-core`・`fandhe-ai-backend-cpu` に `version = "=x.y.z"` を併記する。
4. 公開順: backend-rocm は tensor-core と backend-cpu に依存するため、`backend-cpu` の後・`fandhe-ai` の前に publish する。
5. `release-all.yml` の `RELEASE_CRATES` と `release.yml` の `crate` の選択肢に追加する。
6. `docs/crates-io-publishing-order.md` に新しい節を足し、非公開一覧と内部依存の `version` 併記の箇所数を更新する。
7. 多パッケージの `cargo publish --dry-run --locked` と `cargo deny` で確かめる。
8. ルート `Cargo.toml` と `crates/backend-rocm/Cargo.toml` の「facade からは参照しない」コメントを更新する。

クレートの番号: hf-hub（#3762・#3773）は facade に依存するため、両方が承認されれば backend-rocm が先に出て 8 クレート目、hf-hub が 9 クレート目になる。片方だけなら、それが 8 クレート目。

セキュリティ（OWASP）: A03 gfx target は完全一致 allowlist の enum 値でのみ使い、文字列連結でオプションを作らない。ordinal は `i32::try_from` で検査してから HIP に渡す。ライブラリ名・シンボル名は固定の `&'static str` で、target の選択に環境変数や外部入力を使わない。A06 `libloading` の固定は不変で、公開前に `cargo deny` で依存木を監査する。A08 破壊的変更は人がレビューした PR で入れ、自己修復ループの自動適用の対象にしない。公開は `environment: crates-io-release` の承認ゲートを通す。

### 10.10 所有者へ戻す論点（#3771 で問う）

回答がないときは実装に進まず、保留ガード G1・G2 を維持する。

| 点 | 選択肢 | 推奨 |
|---|---|---|
| 1 Q1 公開 API | A＋B＋E ／ 見送り (i) 対象外 ／ 見送り (ii) 保留 | A＋B＋E（見送るなら (i)） |
| 2 Q2 クレートの公開 | 8〜9 クレート目として公開 ／ 見送り（Q1 も見送りになる） | 公開 |
| 3 gfx target の判定 | (a) `Gfx942` を固定し一貫性契約に例外を設ける ／ (b) HIP のデバイス属性を照会する FFI を足す（新規 `unsafe`。`hipGetDeviceProperties` は ROCm の版で構造体の配置が変わる危険あり） ／ (c) 既存の FFI で gfx942 の code object をそのデバイスにロードし成否で判定（新規 `unsafe` なし。確かさは #3159・#3867 で実測） | (c)。実機で不確かと分かれば (b) を別承認で検討 |
| 4 ordinal | `Rocm(0)` のみ（範囲外は `DeviceUnavailable`）／ `hipSetDevice` で全デバイス対応 | `Rocm(0)` のみ。複数デバイスは別 issue |
| 5 実機 parity との順序 | 実装は承認後に進め 0.14.0 への同梱は #3159・#3867 の pass を条件にする ／ #3159 の pass を実装着手の前提にする | 前者（未実測なら Phase 11 で戻す） |
| 6 エラー variant | 既存の `DeviceUnavailable` ／ `RocmUnavailable` を新設 | 既存 |
| 7 実験的の表示 | rustdoc と `description` に「条件付き・実験的」と書く ／ 書かない | 書く |
| 8 elementwise 融合の opt-in ゲート（`docs/backend-rocm-ops-coverage-plan.md` が #3157 に委ねた点） | ゲートなし（CPU と同じ）／ CUDA と同じ `AtomicBool` ゲート | ゲートなし |

- 確認時点: base `81e23f9ec`・2026-11-01（UTC）。#3159・#3771・spec#87 の状態は書き込み時点の認識で、実装 PR 着手時に再確認する。
