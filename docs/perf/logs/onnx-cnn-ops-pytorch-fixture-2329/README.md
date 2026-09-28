# ONNX Conv・Pool・BN・Flatten import の PyTorch 実生成 fixture 突合（イシュー #2329・親 #2185）実測ログ

## 位置づけ

親 #2185（Conv・Pool・BN・Flatten の ONNX import 拡大）は「PyTorch の
ONNX export を import して bit 同一を確認する」という受け入れ条件チェック
ボックスを持っていたが、子 #2199（PR #2314）・#2200（PR #2312）の実装環境には
`torch` が無かったため、`fandhe_ai_autodiff::nn::*::forward_host` との
**内部突合**（自作 export → 自作 import の自己整合）で代替されていた
（`docs/perf/logs/onnx-conv-pool-import-2199/README.md`「torch 実生成
fixture による突合の未実施」節）。本イシューでは `torch==2.14.0+cpu` を
使い捨て venv（`pip install --index-url
https://download.pytorch.org/whl/cpu torch`）へ導入し、
`torch.onnx.export` が実生成した ONNX を import → 実行 → PyTorch 参照値
との突合を実施した。

**2026-09-28 ユーザー承認で判定方式を正式化**: 縮約系（`Conv`・
`AveragePool`・`GlobalAveragePool`・`BatchNormalization`）は PyTorch CPU
実行系との結合順序差により bit 完全一致を原理的に目標にできないため、
bit 同一を縮約系の受け入れ条件から外し、REQ-7 事前固定式
（`abs_err / (|ref| + 1e-6) <= 1e-3` の `fail_count == 0`。必須条件として
維持）とケースごとの実測上限 baseline への fail-closed 非後退判定を併用
する方式へ正式移行した（`docs/onnx-pytorch-fixture-reduction-parity-
judgment-decision.md` が決定記録の正）。`#2185` の受け入れ条件チェック
ボックスの表現は実装リポ issue 側の記述であり `docs/spec/04-requirements.md`
REQ-7 自体は bit 同一を要求していないため、本改定は spec の緩和ではなく
実装リポ issue の受け入れ条件を spec の定めに整合させる変更である
（詳細は決定記録 §1）。

fixture・生成環境・exporter ごとの op 列・sha256 の詳細は
`crates/onnx-interop/tests/fixtures/pytorch-onnx-cnn-ops/README.md`
（正）を参照する。本ファイルは実測結果と判断根拠のみを記録する
（二重管理しない）。

## 生成環境

- `torch==2.14.0+cpu`・`onnx==1.23.0`・`onnxscript==0.7.2`
- Python 3.14.4・x86_64・Linux
- 決定的シード（`gen_reference.py::deterministic_seed`。ケース名の
  MD5 先頭 4 バイトを `torch.manual_seed` へ渡す）。同じ入力から
  再生成しても `reference.json` は sha256 完全一致することを 2 回連続
  生成で確認済み

## 判定側（Rust テスト）の実行環境・決定性

`cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture`
は x86_64 Linux（このリポジトリの CI・ローカル worktree ともに GitHub
ホステッド `ubuntu-latest` 相当・x86_64）で実行した。計算経路
（`ops::conv`／`ops::pool`／`ops::batch_norm`／`ops::global_average_pool`・
dynamo 分解経路の `ReduceMean`〈`interp_ext::compute_reduce_mean` →
`fandhe_ai_autodiff::Var::mean` → `default_ops::NaiveOps::sum` →
`autodiff::eval::sum` の `Iterator::sum`〉）はいずれも `rayon`・SIMD
intrinsics・`is_x86_feature_detected!` 等のランタイム分岐を含まない
単純な逐次ループ（`f32::mul_add`／`f64` 蓄積、走査順は入力の row-major
順に固定）であることをソースで確認済み（`crates/onnx-interop/src/ops/
conv.rs`・`pool.rs`・`batch_norm.rs`・`global_average_pool.rs`・
`crates/autodiff/src/eval.rs::sum`）。`fandhe-ai-onnx-interop` クレート
自体も `rayon` に依存しない（`Cargo.toml` に `rayon` の記載なし）。
したがって本ページの baseline は CPU feature 差・スレッド数差による
揺れが生じず、CI・ローカルを問わず単一の値で成立する（経路ごとに
baseline を分ける必要はない）。

## テスト実行コマンド

```bash
cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture -- --nocapture
```

40 テスト（19 ケース × 2 exporter + 期待値表整合性検査 1 件 + baseline
整合性検査 1 件）すべて pass。`#[ignore]` テストは 0 件（縮約系の
「bit 同一」受け入れ条件は本改定で正式に廃止されたため、それを直接
検査する `#[ignore]` テストは削除した——`make test-ignored`
〈`cargo test --workspace -- --ignored`〉が常時 fail する状態を残さない）。

## ケース × exporter の実測結果

`--nocapture` の `op_types=`／`bit_mismatch=`／`max_abs_diff=`／
`max_rel_err=`／`req7_fail_count=`／`mean_abs_diff=` 出力を転記する
（`req7_fail_count` は REQ-7 事前固定式
`abs_err / (|ref| + 1e-6) <= 1e-3` での fail 要素数）。**baseline 列は
`REDUCTION_BASELINES`（`crates/onnx-interop/tests/
onnx_interp_pytorch_cnn_fixture.rs`）へ記録した ceiling で、実測値
そのもの（余裕係数なし）**。BitExact 判定のケース（MaxPool・Flatten）は
baseline を持たない（bit 完全一致のみを要求する別方式のため）。

| ケース | exporter | 実際の op 列 | total | fail_count | max_abs_diff | max_rel_err | mean_abs_diff | 判定方式 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `conv2d_basic` | ts | `Conv` | 256 | 0 | 0 | 0 | 0 | Req7BaselineNonRegression |
| `conv2d_basic` | dynamo | `Conv` | 256 | 0 | 0 | 0 | 0 | Req7BaselineNonRegression |
| `conv2d_stride_dil_group` | ts | `Conv` | 100 | 0 | 1.1920929e-7 | 2.7999095e-6 | 3.3006072044372556e-8 | Req7BaselineNonRegression |
| `conv2d_stride_dil_group` | dynamo | `Conv` | 100 | 0 | 1.1920929e-7 | 2.7999095e-6 | 3.3006072044372556e-8 | Req7BaselineNonRegression |
| `conv2d_nobias` | ts | `Conv` | 256 | 0 | 0 | 0 | 0 | Req7BaselineNonRegression |
| `conv2d_nobias` | dynamo | `Conv` | 256 | 0 | 0 | 0 | 0 | Req7BaselineNonRegression |
| `conv1d_basic` | ts | `Conv` | 20 | 0 | 1.1920929e-7 | 4.256559e-6 | 1.862645149230957e-8 | Req7BaselineNonRegression |
| `conv1d_basic` | dynamo | `Conv` | 20 | 0 | 1.1920929e-7 | 4.256559e-6 | 1.862645149230957e-8 | Req7BaselineNonRegression |
| `maxpool2d_basic` | ts/dynamo | `MaxPool` | 48 | — | 0 | 0 | — | BitExact |
| `maxpool2d_pad_dil_ceil` | ts/dynamo | `MaxPool` | 48 | — | 0 | 0 | — | BitExact |
| `maxpool1d_basic` | ts/dynamo | `MaxPool` | 15 | — | 0 | 0 | — | BitExact |
| `avgpool2d_include_pad` | ts/dynamo | `AveragePool` | 48 | 0 | 5.9604645e-8 | 4.206918e-6 | 1.3812496035825461e-8 | Req7BaselineNonRegression |
| `avgpool2d_exclude_pad` | ts/dynamo | `AveragePool` | 48 | 0 | 5.9604645e-8 | 1.2547697e-6 | 1.415416287879149e-8 | Req7BaselineNonRegression |
| `avgpool2d_ceil_overhang_incl` | ts/dynamo | `AveragePool` | 48 | 0 | 5.9604645e-8 | 6.191493e-7 | 9.216212977965673e-9 | Req7BaselineNonRegression |
| `avgpool2d_ceil_overhang_excl` | ts/dynamo | `AveragePool` | 48 | 0 | 1.1920929e-7 | 6.8043673e-6 | 1.5056381622950237e-8 | Req7BaselineNonRegression |
| `avgpool1d_basic` | ts/dynamo | `AveragePool` | 15 | 0 | 1.1920929e-7 | 1.0836598e-7 | 1.5397866566975913e-8 | Req7BaselineNonRegression |
| `gap2d` | ts | `GlobalAveragePool` | 3 | 0 | 3.7252903e-9 | 6.6650045e-8 | 1.241763432820638e-9 | Req7BaselineNonRegression |
| `gap2d` | dynamo | `ReduceMean` | 3 | 0 | 1.4901161e-8 | 1.9995015e-7 | 1.1175870895385742e-8 | Req7BaselineNonRegression |
| `gap1d` | ts | `GlobalAveragePool` | 3 | 0 | 0 | 0 | 0 | Req7BaselineNonRegression |
| `gap1d` | dynamo | `Unsqueeze, ReduceMean, Squeeze` | 3 | 0 | 5.9604645e-8 | 9.2281624e-8 | 2.9802322387695313e-8 | Req7BaselineNonRegression |
| `bn2d_eval` | ts/dynamo | `BatchNormalization` | 150 | 0 | 4.7683716e-7 | 9.008323e-7 | 2.966572841008504e-8 | Req7BaselineNonRegression |
| `bn2d_eval_eps` | ts/dynamo | `BatchNormalization` | 150 | 0 | 2.3841858e-7 | 6.0607965e-7 | 3.9380975067615506e-8 | Req7BaselineNonRegression |
| `bn1d_eval` | ts/dynamo | `BatchNormalization` | 42 | 0 | 1.1920929e-7 | 1.2867288e-7 | 2.9979717163812546e-8 | Req7BaselineNonRegression |
| `flatten_default` | ts | `Flatten` | 120 | — | 0 | 0 | — | BitExact |
| `flatten_default` | dynamo | `Reshape` | 120 | — | 0 | 0 | — | BitExact |
| `flatten_start2` | ts | `Shape, Constant×4, Slice, Concat, Reshape` | 120 | — | 0 | 0 | — | BitExact |
| `flatten_start2` | dynamo | `Reshape` | 120 | — | 0 | 0 | — | BitExact |

**全 38 ケース×exporter の組で `req7_fail_count=0`**（REQ-7 事前固定式で
fail するケースは実測で無かった）。`max_rel_err` の最大値は
`avgpool2d_ceil_overhang_excl` の `6.80e-6`（イシュー #2329 PR #2343
codex-review 指摘対応で fixture の入力形状を `7x7` から `6x6` へ変更し
再実測。旧実測では `avgpool2d_include_pad` の `4.2e-6` が最大だった）で、
閾値 `1e-3` に対して十分な余裕がある。

`REDUCTION_BASELINES` は縮約系 14 ケース × 2 exporter = 28 行を記録して
おり、上表の `total`／`fail_count`／`max_abs_diff`／`max_rel_err`／
`mean_abs_diff` 列と完全一致する（ceiling = 実測値そのもの。
`reduction_baselines_are_well_formed` テストが `EXPECTATIONS` の
`Req7BaselineNonRegression` エントリとの集合完全一致・ceiling の有限性・
非負性を機械検査する）。

## R1〜R6 との対応（受け入れ基準チェック）

| # | 要件 | 結果 |
|---|---|---|
| R1 | 対象 6 op ごとに `torch.onnx.export` 実生成 `.onnx` をコミット | 達成。19 ケース × 2 exporter = 38 ファイル（272 KB） |
| R2 | initializer と `state_dict` の bit 一致 | 達成。全ケース `assert_r2_initializers_match_state_dict` で bit 完全一致検証済み |
| R3 | 純粋な選択・形状操作（MaxPool・Flatten）の bit 完全一致 | 達成。全ケースで `bit_mismatch=0` |
| R4 | 縮約系の実測記録・判定 | **達成**。縮約系（Conv・AveragePool・GlobalAveragePool・BatchNormalization・`ReduceMean` 経路）は REQ-7 事前固定式（必須条件）＋ケースごとの実測上限 baseline への fail-closed 非後退判定の併用方式で判定する（2026-09-28 ユーザー承認）。全 28 行が baseline と完全一致で pass |
| R5 | import 失敗・非対応ケースの列挙 | 下記「R5: import 非対応ケース」参照 |
| R6 | #2185 の受け入れ条件との対応 | **達成**（受け入れ条件を正式改定）。純粋な選択・形状操作系（MaxPool・Flatten。R3）は bit 同一を維持する。縮約系（R4）は bit 同一を受け入れ条件から外し、REQ-7 式＋baseline 非後退判定へ正式移行した（2026-09-28 ユーザー承認・`docs/onnx-pytorch-fixture-reduction-parity-judgment-decision.md`）。#2185 のチェックボックス文言はこの改定に合わせて別途更新する |

## R5: import 非対応ケース

**dynamo exporter の external data 既定挙動**が唯一の非対応ケースである。
dynamo exporter はテンソルサイズに応じ initializer を external data
（`data_location=EXTERNAL`・companion `.data` file）へ既定で逃がす
（`conv.weight`〈432 バイト〉が external、`conv.bias`〈16 バイト〉は
inline のままだったことから、閾値はごく小さい模様）。本クレートの
`onnx::graph::decode_tensor` は external data を意図的に非対応
（`raw_data`/`float_data` の完全一致検証のみを行い、
`external_data`/`data_location` フィールドは未宣言のまま無視する設計）
のため、この生の dynamo 出力を素直に import しようとすると
`GraphError::RawDataByteLenMismatch`（期待バイト長 > 0・実バイト長 0）で
**fail-closed に拒否される**（無言で 0 埋めしたりスキップしたりしない。
正しい動作ではあるが import は失敗する）。

本 fixture は CNN op の意味論突合を主目的とするため、生成スクリプト
（`gen_reference.py::export_one`）が export 直後に external data を
inline へ変換してからコミットしている（詳細はフィクスチャ README
「exporter」節）。**external data サポート自体の要否（実装するか、
非対応のまま維持するか）は別イシューで対応予定**（ユーザー決定済み。
main セッションが起票する）。

これ以外に import が失敗した exporter 出力・op_type の組み合わせは
無かった。dynamo exporter が `GlobalAveragePool` を `ReduceMean`
（+`Unsqueeze`／`Squeeze`）へ、`Flatten` を `Reshape` へ分解するケース
（`gap1d`・`gap2d`・`flatten_default`・`flatten_start2` の dynamo 列）も
含め、分解後の op がすべて本クレートに実装済みのため `run` は成功する。

## AveragePool の `count_include_pad`／`ceil_mode` divisor クリップ規則の実証

`avgpool2d_ceil_overhang_incl`／`avgpool2d_ceil_overhang_excl`（入力
`6x6`・kernel `3x3`・stride `2`・padding `1`・`ceil_mode=1`。出力窓が
入力+padding 領域からはみ出す形状）は、`ops::pool` モジュール doc が
記録する「PyTorch／ONNX Runtime 準拠の padded 座標でのクリップ規則」を
実証する目的で構成した。入力形状は当初 `7x7` だったが、この形状では
`ceil_mode` が生む最終窓の右端・下端が padded 領域（7+2*1=9）の終端に
一致するのみで実際にははみ出さず、divisor クリップ規則を実証できて
いなかった（イシュー #2329 PR #2343 codex-review 指摘・2026-09-28
`6x6` へ修正。`6x6` は floor_mode の 3x3 出力に対し `ceil_mode` が
4x4 出力へ 1 行・1 列増やし、その最終窓が padded 領域〈6+2*1=8〉を
実際に越える）。実測ではいずれも REQ-7 式・baseline 双方の判定を
通過しており（`max_rel_err` はそれぞれ `6.19e-7`・`6.80e-6`）、この
divisor 規則が PyTorch 実行値と整合することを確認した。

## GPU 実機 parity が構造的に N/A である根拠

`onnx::interp::run_impl` のディスパッチ表（`interp.rs:1552-1557` 付近）
は `Conv`／`MaxPool`／`AveragePool`／`BatchNormalization`／
`GlobalAveragePool`／`Flatten` の全 6 op を `(compute_*(..)?, false)` に
固定しており、`run_with_ops`（device 実行 dispatcher）を呼んでも常に
ホスト実装を通る。この構造は `docs/perf/logs/onnx-conv-pool-import-2199/
README.md`「GPU 実機 parity が構造的に N/A である根拠」節が Conv・
MaxPool・AveragePool について既に固定した論拠と同じであり、本イシューで
BatchNormalization・GlobalAveragePool・Flatten へ横展開しても構造は
変わらない（`onnx_interp_backend_dispatch.rs` の既存テストが
`DispatchReport::host_nodes` への計上を確認済み）。したがって GPU 実機
（DGX Spark GB10・Metal 実機）での本 fixture 突合は「計測対象の経路が
存在しない」という意味で構造的に N/A である。

## 将来 device 結線を行う場合の申し送り

将来 `interp_device` へこれら 6 op の device 結線を追加する場合（別
イシュー）、`backend-cuda`／`backend-metal` の対応カーネルとホスト参照
実装（`ops::conv`／`ops::pool`／`ops::batch_norm`／
`ops::global_average_pool`／`ops::flatten`）との REQ-2 統一複合判定
実測が必要になる。既存の pooling／conv／BatchNorm 関連 parity 実測
（`docs/perf/logs/cuda-pooling-1729/`・`docs/perf/logs/metal-pooling-1730/`・
`docs/perf/logs/cuda-batch-norm-1735/`・`docs/perf/logs/metal-batch-norm-
1736/`・`docs/perf/logs/conv-realdevice-1771/` 等）を出発点にできる
見込みである。

## 対応済み事項（まとめ）

1. **REQ-7 式＋baseline 非後退判定の正式採用**: 縮約系の受け入れ条件を
   bit 同一から REQ-7 式＋baseline 非後退判定の併用方式へ正式改定した
   （2026-09-28 ユーザー承認。決定記録は
   `docs/onnx-pytorch-fixture-reduction-parity-judgment-decision.md`）。
2. **external data 非対応**: dynamo exporter の既定挙動（external data）
   への対応要否は別イシューで対応予定（ユーザー決定済み）。本 PR では
   fixture 側で inline 化する運用を継続する。
