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
