# CUDA `TypedOps<f64>` ネイティブカーネル実装（イシュー #2060）実測ランブック

`crates/backend-cuda/src/typed_f64.rs`（ネイティブ CUDA C カーネル
実装。`docs/backend-dtype-dispatch-design.md` §17）の GB10 実機実測
手順・保存すべきログ一覧・事前登録判定規則を記載する。

本実装エージェントの実行環境には CUDA 実機（DGX Spark GB10）への
到達手段がないため、本イシュー時点では実測は未実施のまま記入欄を
残す（`docs/perf/` 配下の他の多数の申し送りと同じ形式）。

## 実行コマンド

```bash
cargo test -p fandhe-ai-backend-cuda --test typed_ops_f64_parity -- --ignored --nocapture
```

対象テスト（`crates/backend-cuda/tests/typed_ops_f64_parity.rs`）:

- `typed_f64_gemm_is_bit_identical_to_reference_fma`
- `typed_f64_elementwise_and_reduction_match_cpu_reference`
- `typed_f64_empty_and_boundary_shapes_match_cpu_semantics`

既存回帰の非後退確認（`typed_f64`／`typed_f16`／`typed_bf16` を含む
`backend-cuda` の実機 `#[ignore]` テスト全体）:

```bash
cargo test -p fandhe-ai-backend-cuda --all-features -- --ignored --nocapture
```

## 事前登録判定規則

- **正しさ（数値契約）**:
  - `gemm`／`add`／`mul`／`relu`・軸指定 `sum`／`max`: CPU 参照実装
    （`crates/backend-cpu/src/typed_f64.rs`・
    `fandhe_ai_backend_cpu::matmul_reference_fma_f64`）と **bit 完全
    一致**（`assert_eq!` による `host_slice()` 完全一致）
  - `exp`／`tanh`・全軸縮約 `sum`／`max`: CPU 参照実装と
    `fandhe_ai_backend_cpu::assert_parity_f64`（REQ-2 統一複合判定。
    相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致
  - 判定式・tolerance 定数（`RELATIVE_TOLERANCE`／
    `ABSOLUTE_RESCUE_THRESHOLD`）は変更しない
- **checksum・run-to-run 決定性**: 実機 `#[ignore]` テストを 2 回以上
  実行し、同一入力に対して同一の bit パターンが得られることを確認する
  （既存の `typed_ops_f16_parity.rs` 等と同型の確認手順）
- **既存回帰の非後退**: `crates/backend-cuda` の実機 `#[ignore]`
  テスト全体を実行し、本イシュー実装前から既知の FAIL 件数を超えない
  ことを確認する（新規追加テスト以外の既存 FAIL 件数は変化しない見込み。
  f32 実装本体〈`elementwise.rs`／`reduce.rs`／`gemm.rs`／`kernels.rs`／
  `kernels_elementwise.rs`／`kernels_reduce.rs`〉は無変更のため）

## 保存すべきログ

- `typed_ops_f64_parity_run1.log`／`_run2.log`（上記対象テストの実行
  ログ 2 回分・`--nocapture`）
- `backend_cuda_ignored_all.log`（既存回帰の非後退確認ログ）
- `env_info.txt`（GPU 型番・CUDA driver／NVRTC バージョン・
  `nvidia-smi` 出力。内部ホスト名は含めない）

## 実測欄（未実施）

| 項目 | 結果 |
|---|---|
| `gemm` bit 完全一致 | 未実施 |
| `add`／`mul`／`relu` bit 完全一致 | 未実施 |
| 軸指定 `sum`／`max` bit 完全一致 | 未実施 |
| `exp`／`tanh` REQ-2 複合判定 | 未実施 |
| 全軸縮約 `sum`／`max` REQ-2 複合判定 | 未実施 |
| 空縮約・境界形状の意味論一致 | 未実施 |
| 既存回帰の非後退 | 未実施 |
