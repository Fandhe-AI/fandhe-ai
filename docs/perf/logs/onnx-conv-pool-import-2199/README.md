# ONNX Conv（1D 拡張）・MaxPool・AveragePool import（イシュー #2199・親 #2185）実測ログ

## 位置づけ

`ops::conv`（1D 対応拡張）・`ops::max_pool`／`ops::average_pool`（新規）の
CPU 参照実装は本 PR で実装・CI（GitHub ホステッド `ubuntu-latest`）でテスト
済み。CUDA／Metal 実機（DGX Spark GB10・Metal 実機）での parity 実測は
本 PR のスコープ外であり、本ファイルに申し送りを記録する。

## GPU 実機 parity が構造的に N/A である根拠

ONNX interp（`crate::onnx::interp::run_impl`）の `Conv`／`MaxPool`／
`AveragePool` ディスパッチは、device 実行 dispatcher（`interp_device`。
イシュー #2077／#2222）の対象外に固定されている。`run_with_ops`（`dev_ops`
が `Some`）を呼んでも、これら 3 op は常にホスト実装（`ops::conv`／
`ops::max_pool`／`ops::average_pool`）を通る。この到達性は
`crates/onnx-interop/tests/onnx_interp_backend_dispatch.rs::
run_with_ops_conv_1d_and_pool_stay_on_host`（`RecordingOps`〈device
経路を持つ `BackendOps`〉を渡しても `DispatchReport::host_nodes` に
記録され `device_nodes` が空であることを確認）で固定している。

したがって、CUDA／Metal の `BackendOps` 実装（`backend-cuda`・
`backend-metal`）がどのような `max_pool2d`／`avg_pool2d`／`conv2d`
カーネルを持つかに関わらず、ONNX import 経路からはこれらの GPU カーネルへ
到達しない。GPU 実機 parity 計測は「計測対象の経路が存在しない」という
意味で構造的に N/A である。

## 将来 device 結線を行う場合に必要な実機計測

将来 `interp_device` へ `Conv`／`MaxPool`／`AveragePool` の device 結線を
追加する場合（別イシュー）、`backend-cuda`／`backend-metal` の
`max_pool2d`／`avg_pool2d`／`conv2d`（or `gemm_fp32_strict` 経由の
im2col+GEMM）カーネルとホスト参照実装（`ops::pool`／`ops::conv`）との
REQ-2 統一複合判定（相対誤差 1e-3 未満 または絶対誤差 1e-5 未満）実測が
必要になる。`ops::pool`（`max_pool`／`average_pool`）は純粋な選択・
縮約演算であり、`fandhe_ai_backend_cpu::pooling` と同じ走査順・タイ
規則・NaN 規則・`f64` 縮約契約を踏襲しているため（`ops/pool.rs` モジュール
doc 参照）、既存の pooling parity 実測（`docs/perf/logs/`
配下の pooling 関連ログ）を出発点にできる見込みである。

## torch 実生成 fixture による突合の未実施

本 PR の計画立案・実装時点で作業環境に `torch`（PyTorch）が導入されて
いないため（`import torch` が `ModuleNotFoundError`）、
`torch.onnx.export` が実際に出力する ONNX グラフ由来の fixture は
作成できなかった。代わりに、`crates/onnx-interop/tests/
onnx_interp_conv_pool.rs`・`crates/facade/tests/
interop_onnx_conv_pool_import.rs` は `ops::*` を直接呼ぶ経路と
`fandhe_ai_autodiff::nn::{Conv1d, Conv2d, MaxPool1d, MaxPool2d, AvgPool1d,
AvgPool2d}::forward_host` を突合する方式で代替した（`onnx_interp_conv_
pool.rs` モジュール冒頭コメント §3.3 参照）。torch 実生成 fixture による
突合は今後の課題として申し送る（ユーザー承認なしに Issue は起票しない。
`.claude/rules/out-of-scope-tracking.md`）。
