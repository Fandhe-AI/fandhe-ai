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

`backend-cuda` 側の `make test-ignored-cuda`（TASK-1.7e・#36）と対になる per-backend ターゲットであり、`make test-ignored`（`cargo test --workspace -- --ignored --nocapture`）のように他バックエンド（CUDA 実機必須のテストを含む）を巻き込まない。

## なぜ通常 CI では除外されるか（cfg + `#[ignore]` の二重分離）

1. **`#![cfg(target_os = "macos")]`**: 各テストファイル冒頭に付与。`objc2` 系依存自体が macOS 限定（`.claude/rules/deps-policy.md`）のため、CI（self-hosted・Linux）ではそもそもコンパイル対象外になる。
2. **`#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]`**: macOS 実機上でも GPU 実体が必要なテストには個別に付与。通常の `cargo test`（`--ignored` 指定なし）からは除外される。

この二重分離により、受け入れ条件「通常 CI では除外される」が Linux self-hosted CI 上でも機械的に成立する（`cargo test --workspace --all-features` のログで `#[ignore]` テストが実行されないことを確認可能。TASK-1.8e 実装時に確認済み）。

macOS 側の型検査は Linux CI でも成立させている（後述の「Linux CI での型検査」節）。

## テスト一覧と対応 REQ/TASK

| ファイル | 対応 TASK/Issue | 検証内容 |
|---------|-----------------|---------|
| `tests/device.rs` | TASK-1.9a（#44） | `MetalDeviceProvider` の実機デバイス選択（`select_metal_device_on_real_hardware`） |
| `tests/device_smoke.rs` | TASK-1.8a（#38） | デバイス初期化・コマンドキュー・バッファ確保・readback・不正長入力の拒否（OWASP A03） |
| `tests/gemm_naive_parity.rs` | TASK-1.8b（#39） | naive GEMM の CPU 参照実装との複合判定（REQ-2）。境界形状・中規模形状・K ストレス・縮退形状（TASK-1.8e で追加）・`dispatch`/`dispatch_variant(Naive)` 同値性（TASK-1.8e で追加）・不正形状の拒否 |
| `tests/gemm_simdgroup_parity.rs` | TASK-1.8c（#40） | tiled/simdgroup GEMM の CPU 参照実装との複合判定（REQ-2）。境界形状・中規模形状・K ストレス・縮退形状（TASK-1.8e で追加）・不正形状の拒否 |
| `tests/cpu_metal_parity.rs` | TASK-2.2c（#55） | CPU-Metal ペアの数値一致回帰（REQ-2 統一複合判定の固定。K=4096 ストレス・境界形状・決定性・falsification） |
| `tests/backend_ops_real_device.rs` | TASK-1.9d（#47） | `tensor_core::BackendOps`（`MetalBackendOps::gemm`。`dispatch_auto` 動的タイル選択経由）と CPU `BackendOps::gemm` の数値一致（REQ-2）。`tile.rs::select` の動的タイル選択境界（`SMALL=64`）近傍・縦長横長分岐・`ops_for` 経由ディスパッチ・elementwise/reduction の `Unsupported` 契約を含む |

判定は全ファイル共通で `backend_cpu::parity::{compare, assert_parity}`（REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体）を使う。**閾値の独自定義・緩和はしない**（`.claude/rules/security.md`・`.claude/rules/coding-rust.md`）。入力生成は `bench_harness::rng::Xorshift64Star`（決定的シード）で固定する。

## Linux CI での型検査（macOS runner 未登録の代替）

macOS self-hosted runner は未登録のため、`backend-metal` の `#[ignore]` テストは Linux CI 上では**実行できない**。代わりに、`aarch64-apple-darwin` ターゲットへの `cargo check` でコンパイル可能性（型検査）のみを Linux CI 上で担保する。

```bash
make check-cross-metal-tests
# 相当コマンド:
cargo check -p backend-metal --tests --target aarch64-apple-darwin
```

`.github/workflows/ci.yml` の `build` ジョブに同一コマンドのステップとして組み込み済み（`cargo build --workspace --locked --target aarch64-apple-darwin` の後段）。

**`--workspace --all-targets` ではなく `-p backend-metal --tests` に限定する理由**: `--workspace --all-targets` は `bench-harness` の `dev-dependencies`（`criterion`）を経由して `alloca`（macOS ターゲットでネイティブ C ビルドを要する）を引き込み、macOS クロスコンパイラ非搭載の self-hosted runner（Linux）では

```
cc: error: unrecognized command-line option '-arch'
cc: error: unrecognized command-line option '-mmacosx-version-min=11.0'
```

で失敗することを実測した（本イシュー実装時に確認）。`-p backend-metal --tests` に限定すると、`backend-metal` が `bench-harness` を参照する経路は `[dependencies]`（`criterion` を含まない）のみが解決されるため、この失敗を回避しつつ `tests/` 配下の型検査が成立する。`cargo check` はリンクを行わないため、macOS SDK が無い Linux 環境でも成立する（`cargo build` は実機フレームワークのリンクが必要になるため使わない）。

## 実機実行の実測状況

TASK-1.8e（#42）実装時点では、本ドキュメントの手順自体は Linux 環境からは実行不能（Apple Silicon 実機が必要）のため、上記コマンドが実際に green になることの実測はユーザー側での確認を依頼する。実装環境で機械検証済みなのは以下の項目である。

- `cargo test --workspace --all-features`（`#[ignore]` テストが実行されないログの確認 = 「通常 CI では除外される」の検証）
- `cargo check -p backend-metal --tests --target aarch64-apple-darwin`（型検査成立の確認）
- `cargo fmt --all --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`（Linux ホスト分）

## 将来課題（スコープ外）

- **macOS（Metal 実機）self-hosted runner の登録と実機 CI ジョブの追加**: runner 未登録のため TASK-1.8e では実施しない。追加する場合は runner ラベルで対象 runner を明示する（`.claude/rules/ci.md`）
- **CUDA 側の同種整備**: `backend-cuda` は既に `make test-ignored-cuda`（TASK-1.7e・#36）で整備済み。本ドキュメントはその Metal 版に相当する
