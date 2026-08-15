# backend-metal 実機（Apple Silicon）テスト実行手順

イシュー #42「test(backend-metal): TASK-1.8e 実機テスト（`#[ignore]` 分離）の整備」に対応する。
TASK-1.8（親 #37）配下の TASK-1.8a〜d（#38〜#41）が暫定的に置いた `#[ignore]` 実機テスト群（`crates/backend-metal/tests/` 5 ファイル・約 20 テスト）の「本格的な実機 CI 整備」を仕上げるドキュメントであり、TASK-1.8e の受け入れ条件「`cargo test -- --ignored` で実機テストが実行でき、通常 CI では除外される」の実機側の担保を、実測が行えない本環境の代わりに手順として固定する。

## 前提

- Apple Silicon Mac（macOS）
- rustup（`rustup show` でツールチェーンが解決できること）
- `git submodule update --init` 済み（`docs/spec` submodule）

CUDA 側の同種ドキュメント（`.claude/rules/`・各テストファイルの doc コメントに分散）と異なり、`backend-metal` は本ドキュメントを実行手順の正本とする。

## 実行コマンド

```bash
make test-ignored-metal
# 相当コマンド（--release 推奨。理由は後述）:
cargo test -p backend-metal --release -- --ignored --nocapture
```

`--release` を既定にしている理由: `tests/cpu_metal_parity.rs` の `k4096_stress_poc_v2_5`（M=N=512, K=4096 の CPU 参照実装によるストレスケース）は debug ビルドでは著しく遅い。各テストファイル冒頭の doc コメントも同じコマンドを推奨している。

`backend-cuda` 側の `make test-ignored-cuda`（TASK-1.7e・#36）と対になる per-backend ターゲットであり、`make test-ignored`（`cargo test --workspace -- --ignored --skip wmma_tf32_basic_kernel_parity_does_not_regress --nocapture`。除外理由は `Makefile` の当該ターゲットコメント・イシュー #491 参照）のように他バックエンド（CUDA 実機必須のテストを含む）を巻き込まない。

## なぜ通常 CI では除外されるか（cfg + `#[ignore]` の二重分離）

1. **`#![cfg(target_os = "macos")]`**: 各テストファイル冒頭に付与。`objc2` 系依存自体が macOS 限定（`.claude/rules/deps-policy.md`）のため、CI（GitHub ホステッド・Linux）ではそもそもコンパイル対象外になる。
2. **`#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]`**: macOS 実機上でも GPU 実体が必要なテストには個別に付与。通常の `cargo test`（`--ignored` 指定なし）からは除外される。

この二重分離により、受け入れ条件「通常 CI では除外される」が Linux CI（GitHub ホステッド）上でも機械的に成立する（`cargo test --workspace --all-features` のログで `#[ignore]` テストが実行されないことを確認可能。TASK-1.8e 実装時に確認済み）。

macOS 側の型検査は Linux CI でも成立させている（後述の「Linux CI での型検査」節）。

## テスト一覧と対応 REQ/TASK

`crates/backend-metal/tests/` は 12 ファイル。うち 11 ファイルが `#[ignore]` 実機テストを持ち、合計 52 件
（イシュー #380 実機実測で確定。件数根拠は per-file 合計を正とする。`cargo test` 出力の
`N filtered out` 行は lib unittest ターゲットが非 `#[ignore]` テストを除外した数で偶然の一致にすぎず
根拠に使わない）。残る 1 ファイル（`shader_source_evidence.rs`）は `#[ignore]` を持たず Linux CI でも実行
される。

| ファイル | `#[ignore]` 件数 | 対応 TASK/Issue | 検証内容 |
|---------|:---:|-----------------|---------|
| `tests/device.rs` | 1 | TASK-1.9a（#44） | `MetalDeviceProvider` の実機デバイス選択（`select_metal_device_on_real_hardware`） |
| `tests/device_smoke.rs` | 3 | TASK-1.8a（#38） | デバイス初期化・コマンドキュー・バッファ確保・readback・不正長入力の拒否（OWASP A03） |
| `tests/gemm_naive_parity.rs` | 6 | TASK-1.8b（#39） | naive GEMM の CPU 参照実装との複合判定（REQ-2）。境界形状・中規模形状・K ストレス・縮退形状・`dispatch`/`dispatch_variant(Naive)` 同値性・不正形状の拒否 |
| `tests/gemm_simdgroup_parity.rs` | 9 | TASK-1.8c（#40） | tiled/simdgroup GEMM の CPU 参照実装との複合判定（REQ-2）。境界形状・中規模形状・K ストレス・縮退形状・不正形状の拒否 |
| `tests/cpu_metal_parity.rs` | 5 | TASK-2.2c（#55） | CPU-Metal ペアの数値一致回帰（REQ-2 統一複合判定の固定。K=4096 ストレス・境界形状・決定性・falsification） |
| `tests/backend_ops_real_device.rs` | 5 | TASK-1.9d（#47） | `tensor_core::BackendOps`（`MetalBackendOps::gemm`。`dispatch_auto` 動的タイル選択経由）と CPU `BackendOps::gemm` の数値一致（REQ-2）。`tile.rs::select` の動的タイル選択境界（`SMALL=64`）近傍・縦長横長分岐・`ops_for` 経由ディスパッチ・elementwise/reduction の `Unsupported` 契約を含む |
| `tests/gemm_auto_parity.rs` | 3 | TASK-1.8f 前段 | `dispatch_backend_auto` の閾値前後（境界 511/512）での CPU 参照一致 |
| `tests/gemm_dynamic_tile_parity.rs` | 6 | TASK-1.8f（#188） | 動的タイル選択（`dispatch_auto`）の全タイル候補・非倍数形状・K ストレスでの CPU 参照一致 |
| `tests/cpu_metal_f16_parity.rs` | 6 | TASK-8.3b（#156） | `gemm_simdgroup_f16` の CPU 参照一致（8×8×8 基準・512 基準・非倍数境界・K=4096 ストレス・決定性・不正入力の拒否）。累算精度契約はイシュー #380 で f32 累算へ変更済み（`docs/perf/metal-f16-vs-mps-f16.md`「精度契約」節） |
| `tests/dispatch_boundary.rs` | 2 | #382 | `dispatch_auto` の境界形状 TFLOPS 記録・`dispatch_backend_auto` の出力と CPU 参照実装との数値一致検証（実機が実際にどの経路を選んだかの検証ではない。`route_verified=false`）。TFLOPS 数値の転記・`METAL_SIMDGROUP_MIN_DIM` の妥当性判定は #382 で実施済み（`docs/perf/dispatch-boundary-measurement.md`） |
| `tests/memory_roundtrip.rs` | 6 | TASK-2.1 系 | メモリ確保・ゼロ初期化・プール再利用・アップロード/ダウンロード roundtrip・リーク検査 |
| `tests/shader_source_evidence.rs` | 0（`#[ignore]` なし。Linux CI でも実行） | TASK-11.3（#70） | `gemm.metal` の行列演算ユニット命令（`simdgroup_matrix` API）実在検査・REQ-8 境界チェック維持検査 |

判定は数値一致系ファイル共通で `backend_cpu::parity::{compare, assert_parity}`（REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体）を使う。**閾値の独自定義・緩和はしない**（`.claude/rules/security.md`・`.claude/rules/coding-rust.md`）。入力生成は `bench_harness::rng::Xorshift64Star`（決定的シード）で固定する。

## Linux CI での型検査（macOS runner 未登録の代替）

macOS 実機 runner は未登録のため、`backend-metal` の `#[ignore]` テストは Linux CI 上では**実行できない**。代わりに、`aarch64-apple-darwin` ターゲットへの `cargo check` でコンパイル可能性（型検査）のみを Linux CI 上で担保する。

```bash
make check-cross-metal-tests
# 相当コマンド:
cargo check -p backend-metal --tests --target aarch64-apple-darwin
```

`.github/workflows/ci.yml` の `build` ジョブに同一コマンドのステップとして組み込み済み（`cargo build --workspace --locked --target aarch64-apple-darwin` の後段）。

**`--workspace --all-targets` ではなく `-p backend-metal --tests` に限定する理由**: `--workspace --all-targets` は `bench-harness` の `dev-dependencies`（`criterion`）を経由して `alloca`（macOS ターゲットでネイティブ C ビルドを要する）を引き込み、macOS クロスコンパイラ非搭載の Linux CI runner では

```
cc: error: unrecognized command-line option '-arch'
cc: error: unrecognized command-line option '-mmacosx-version-min=11.0'
```

で失敗することを実測した（本イシュー実装時に確認）。`-p backend-metal --tests` に限定すると、`backend-metal` が `bench-harness` を参照する経路は `[dependencies]`（`criterion` を含まない）のみが解決されるため、この失敗を回避しつつ `tests/` 配下の型検査が成立する。`cargo check` はリンクを行わないため、macOS SDK が無い Linux 環境でも成立する（`cargo build` は実機フレームワークのリンクが必要になるため使わない）。

## 実機実行の実測状況

TASK-1.8e（#42）実装時点では、本ドキュメントの手順自体は Linux 環境からは実行不能（Apple Silicon 実機が必要）のため、上記コマンドが実際に green になることの実測はユーザー側での確認を依頼していた。**イシュー #380 で実機実測が完了し、以下の内容で確定した。**

### 実行環境（#380）

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max（メモリ 64GB） |
| OS | macOS 26.6（build 25G72） |
| toolchain | `stable-aarch64-apple-darwin`（`rust-toolchain.toml` の override 適用） |

### 実行結果（`make test-ignored-metal`。`--no-fail-fast --test-threads=1` で全 12 バイナリを中断なく実行）

52 件全件 PASS（0 FAIL）。内訳は上記「テスト一覧と対応 REQ/TASK」表の `#[ignore]` 件数列のとおり。

実装計画時点のインベントリでは `cpu_metal_f16_parity.rs` の 4 件（`f16_parity_baseline_8x8x8`・
`f16_parity_boundary_shapes_non_multiple_of_eight`・`f16_parity_baseline_shape_512`・`f16_k4096_stress`）が
`gemm_simdgroup_f16` の累算精度契約（half 統一アキュムレータ）に起因して REQ-2 複合判定を外れていたが、
根本原因を修正（アキュムレータを `simdgroup_float8x8`〈f32 累算〉へ変更。詳細は
`docs/perf/metal-f16-vs-mps-f16.md`「精度契約」節）した結果、6 件全件が green になった
（`backend_cpu::parity` の判定式・閾値は無変更。「許容誤差の緩和」ではなく「累算精度の向上」）。

MSL 構文検証（`MetalGemm::new` による `gemm.metal` 全体のランタイムコンパイル。`gemm_naive`/`gemm_tiled`/
`gemm_simdgroup`/`gemm_simdgroup_f16`/`gemm_simdgroup_tiled` の全タイル候補 function constant 組合せを含む）
も実機コンパイル成功を確認済み。

### 運用上の注記

- インベントリ確認・記録用の再実行は `--no-fail-fast`（cargo は既定で最初に失敗したテストバイナリで
  中断するため、全 12 バイナリの結果を 1 回の実行で取得するには必須）・`--test-threads=1`（GPU 競合による
  計測揺れを排除）を付けて行う。**正本コマンド `make test-ignored-metal` はこれらのフラグを付けない**
  （デバッグ・インベントリ用途と受入チェック用途を分離する。`Makefile` は変更していない）
- `dispatch_boundary.rs::boundary_shapes_tflops_record` は TFLOPS を出力するが本ドキュメントでは
  pass/fail のみを記録対象とする。数値の `docs/perf/dispatch-boundary-measurement.md` への転記・
  `METAL_SIMDGROUP_MIN_DIM` の妥当性判定は #382 で完了済み（判定: 変更提案あり・提案値 384。
  コード未変更・実施は別レビュー・別 PR・ユーザー承認）

過去に機械検証済みだった以下の項目は、上記の実機実行によって補完・上書きされた:

- `cargo test --workspace --all-features`（`#[ignore]` テストが実行されないログの確認 = 「通常 CI では除外される」の検証）
- `cargo check -p backend-metal --tests --target aarch64-apple-darwin`（型検査成立の確認）
- `cargo fmt --all --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`（Linux ホスト分。#380 では実機側でも再確認済み）

## 将来課題（スコープ外）

- **macOS（Metal 実機）runner の登録と実機 CI ジョブの追加**: runner 未登録のため TASK-1.8e では実施しない。追加する場合は runner ラベルで対象 runner を明示する（`.claude/rules/ci.md`「実機依存」節）
- **CUDA 側の同種整備**: `backend-cuda` は既に `make test-ignored-cuda`（TASK-1.7e・#36）で整備済み。本ドキュメントはその Metal 版に相当する

以下は Metal 実機検証・ベンチ計測トラッキングツリー（親 #379）完了時点（イシュー #387・総括反映）で
残存が確認された未実施項目の集約（受け入れ条件「残存する未実施項目を明示的に記録する」への対応）:

- **`METAL_SIMDGROUP_MIN_DIM` 変更提案（384 への引き下げ）の実施**: #382 で記録済みの変更提案はコード
  未変更のまま。実施は別レビュー・別 PR・ユーザー承認を要する（[`docs/perf/dispatch-boundary-measurement.md`](./perf/dispatch-boundary-measurement.md)
  「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節）。連動して `crate::tile::select`/`CANDIDATES` の
  「暫定値」コメント更新も同一の別 PR まで据え置く（[`docs/perf/metal-gemm-dynamic-tile.md`](./perf/metal-gemm-dynamic-tile.md)
  「未実施・後続作業」節）
- **bench-harness の Metal 版起動コスト `#[ignore]` E2E テスト未追加**: 起動コスト実測（#384）自体は
  完了済みだが、回帰検出用の `#[ignore]` E2E テストの追加は本ツリーのスコープ外のまま
  （[`docs/perf/startup-cost-measurement.md`](./perf/startup-cost-measurement.md)「後続」節）
- **Transformer 複合ワークロードの Metal 実機実測は記入待ちのまま**: 本ツリー（#379）の対象外（#155 系）。
  実機（Apple M4 Max・DGX Spark GB10）値は未実測（[`docs/perf/transformer-workload-measurement.md`](./perf/transformer-workload-measurement.md)
  「実機実測（記入待ち）」節）
- **f16 の自動ディスパッチ規則への統合はスコープ外のまま**: `docs/dispatch-rules-design.md` への統合は
  実装計画時点から対象外（[`docs/perf/metal-f16-vs-mps-f16.md`](./perf/metal-f16-vs-mps-f16.md)「未実施・
  後続作業」節）。REQ-11 系の後続課題として別途追跡する
- **短命プロセス対応方針の再判定は TASK-13.2（#172・人間判断）待ち**: CUDA（#391）・Metal（#384）の起動
  コスト転記完了によりトリガー自体は消化済みだが、再判定・方針決定は人間判断のスコープであり本ツリーでは
  行わない（[`docs/short-lived-process-decision.md`](./short-lived-process-decision.md)「再判定トリガー」節）
