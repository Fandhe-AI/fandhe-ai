# generate()（#2191）CUDA／Metal 実機未実測の申し送り

`docs/facade-generate-decision.md`「5. テスト・実測」参照。本実装
エージェント実行環境は CUDA／Metal 実機に到達できないため、自己回帰
生成ループ（`fandhe_ai_autodiff::generate`）の
`crates/facade/tests/generate_backend_parity.rs` のうち CUDA
（`cuda_backend_ops_matches_cpu_backend_ops_for_greedy_generation`）を
対象とする 1 テストは `#[ignore]` のまま未実測である（Metal 用テストは
本 PR 時点で未追加——facade 公開が保留のため facade 越しの Metal
`AutoregressiveModel` 実装を書く手段がなく、CPU／CUDA の 2 バックエンド
のみで parity を確認している。facade 公開〈`docs/facade-generate-
decision.md` §8〉が承認された後、Metal 版 parity テストを追加する際に
本 README も更新すること）。

## 本エージェント環境で実測できない理由

`docs/perf/logs/kv-cache-2084/README.md` と同じ理由: 本エージェント
実行環境（x86_64 Linux）は NVIDIA GPU の driver は確認できても NVRTC・
CUDA toolkit が導入されておらず、CUDA backend の実行時 JIT が成立
しない。また代替実機（本リポジトリの実機ではない GPU）は「baseline は
実機実測値のみ・人間承認必須」（`.claude/rules/coding-rust.md` テスト・
ベンチ節）の規約上使えない。Metal は本環境が Linux のため対象外。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test generate_backend_parity -- --ignored --nocapture cuda
```

CPU（`CpuBackendOps`）版は上記テストファイルの属性なしテスト
（`cpu_backend_ops_matches_naive_ops_for_greedy_generation`・
`cpu_backend_ops_matches_naive_ops_for_top_k_generation_with_same_seed`）
で既に検証済み（green。`docs/facade-generate-decision.md` §5 参照）。
`fandhe_ai_autodiff` 側の KV キャッシュ結線・3 戦略・seed 決定性の単体
検証は `crates/autodiff/tests/nn_generate.rs`（18 件・全 pass。CPU の
みで完結する `NaiveOps` 相当の直接テストのため実機非依存）で完結して
いる。
