# onnx-autograd（#2078）CUDA／Metal 実機未実測の申し送り

`docs/onnx-autograd-decision.md` §5・§7 参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`onnx::autograd::BoundGraph` の GPU tape
（`fandhe_ai_backend_cuda::CudaBackendOps`／`fandhe_ai_backend_metal::MetalBackendOps`）
上での forward／backward parity は未実測のまま Mac／GB10 セッションへ申し送る。

## 測定コマンド案

```sh
cargo test -p fandhe-ai-onnx-interop --test onnx_autograd -- --nocapture
```

CPU tape（`CpuBackendOps`）版は上記コマンドで既に検証済み（20 テスト全 green）。
GPU tape 版は `Tape::new_with_ops(Box::new(CudaBackendOps::new(..)))` 等へ差し
替えた同等テストを別ファイル（`onnx_autograd_device_parity.rs`。`#[ignore]`）
として追加し、`crates/backend-cpu::assert_parity` で CPU 結果と突合する想定
（`Erf`／`Softmax`／`LayerNormalization` は `Tape::custom` の契約上 GPU tape
上でもホスト実行になるため、device 差分は原理的に生じない）。

## 期待結果

- forward: `Gemm`／`MatMul`／`Add`／`Mul`／`Div`／`Sqrt`／`Relu`／`Sigmoid` は
  CPU バックエンドの forward 契約（FMA 契約。`.claude/rules/coding-rust.md`）に
  従う各バックエンドの `ops::*` 実装と REQ-2 統一複合判定で一致するはず。
- backward: `Tape::custom` の backward はホスト実行のみのため device 差分なし。

- 追記（#3778）: GPU tape 上の ONNX の学習の parity テストは onnx-interop の `onnx_autograd_device_parity.rs` 案ではなく、facade の `crates/facade/tests/train_gpu_parity.rs` の `onnx` 族に置く。測定は #3864（GB10）・#3865（Mac mini）。
- 追記（#3864）: GB10（`<cuda-node>`）で `train_parity_onnx_cuda_matches_cpu` を実測し pass（2026-11-04）。`onnx_from_sequential_forward_backward_s1`・`_s2` の forward・loss・4 つの勾配の計 12 量がすべて REQ-2 統一複合判定で一致（最大絶対誤差は loss の 5.960464e-8、ほかは 0）。route は H（経路の parity）。記録は `docs/perf/logs/onnx-train-cuda-parity-3864/`。Metal（Mac mini）は #3865 で未実測のまま。
- 追記（#3865）: Mac mini（`<metal-node>`）で `train_parity_onnx_metal_matches_cpu` を実測し pass（2026-11-04）。`onnx_from_sequential_forward_backward_s1`・`_s2` の forward・loss・4 つの勾配の計 12 量がすべて REQ-2 統一複合判定で一致（最大絶対誤差は s2 の loss の 2.980232e-8、ほかは 0）。route は H（経路の parity）。記録は `docs/perf/logs/onnx-train-metal-parity-3865/`。
- 追記（#4046）: #3864 の回はテストバイナリの由来を記録から確かめられなかったため（#4044 の棚卸しで C）、共有 target の成果物を消してから GB10 で測り直し、pass（2026-11-05）。12 量の値は #3864 の回と同一。記録は `docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`。
