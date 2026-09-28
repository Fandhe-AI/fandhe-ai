# ONNX Conv・Pool・BN・Flatten import の PyTorch 実生成 fixture 突合（イシュー #2329・親 #2185）実測ログ

## 位置づけ

親 #2185（Conv・Pool・BN・Flatten の ONNX import 拡大）は「PyTorch の
ONNX export を import して bit 同一を確認する」という受け入れ条件を
持っていたが、子 #2199（PR #2314）・#2200（PR #2312）の実装環境には
`torch` が無かったため、`fandhe_ai_autodiff::nn::*::forward_host` との
**内部突合**（自作 export → 自作 import の自己整合）で代替されていた
（`docs/perf/logs/onnx-conv-pool-import-2199/README.md`「torch 実生成
fixture による突合の未実施」節）。本イシューでは `torch==2.14.0+cpu` を
使い捨て venv（`pip install --index-url
https://download.pytorch.org/whl/cpu torch`）へ導入し、
`torch.onnx.export` が実生成した ONNX を import → 実行 → PyTorch 参照値
との突合を実施した。

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

## テスト実行コマンド

```bash
cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture -- --nocapture
```

39 テスト（19 ケース × 2 exporter + 期待値表整合性検査 1 件）すべて
pass。

**縮約系の bit 同一（親 #2185 の受け入れ条件）を直接検査する
`#[ignore]` テストが別途ある**（codex-review 指摘対応。イシュー #2329
PR #2343。レビュースレッド `PRRT_kwDOTuUCJc6mhmHL`・
`PRRT_kwDOTuUCJc6mhpBB`）:

```bash
cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture -- --ignored
```

上記通常 39 テストの `Req7Provisional`（`fail_count == 0`）による pass
は、REQ-7 事前固定式に対する fail-closed な回帰ガードに過ぎず、
親 #2185 の「bit 同一」受け入れ条件そのものの合格を意味しない。
`reduction_ops_bit_exact_acceptance_pending_approval`
（`crates/onnx-interop/tests/onnx_interp_pytorch_cnn_fixture.rs`）は
`Req7Provisional` の全 23 ケース × exporter について
`bit_mismatch_count == 0` を直接検査し、**実測時点で 23/23 件 fail する
（意図的に red）**。判定方式（下記「承認待ち事項」1. の採否）が確定し
`Expectation` 表・本ファイルが更新されるまで、このテストを削除・
green 化してはならない。

## ケース × exporter の実測結果

`--nocapture` の `op_types=`／`bit_mismatch=`／`max_abs_diff=`／
`max_rel_err=`／`req7_fail_count=` 出力を転記する（`req7_fail_count` は
REQ-7 事前固定式 `abs_err / (|ref| + 1e-6) <= 1e-3` での fail 要素数）。

| ケース | exporter | 実際の op 列 | bit_mismatch | max_abs_diff | max_rel_err | 判定 |
|---|---|---|---|---|---|---|
| `conv2d_basic` | ts | `Conv` | 0/256 | 0 | 0 | BitExact |
| `conv2d_basic` | dynamo | `Conv` | 0/256 | 0 | 0 | BitExact |
| `conv2d_stride_dil_group` | ts | `Conv` | 63/100 | 1.19e-7 | 2.80e-6 | Req7Provisional（暫定 pass） |
| `conv2d_stride_dil_group` | dynamo | `Conv` | 63/100 | 1.19e-7 | 2.80e-6 | Req7Provisional（暫定 pass） |
| `conv2d_nobias` | ts | `Conv` | 0/256 | 0 | 0 | BitExact |
| `conv2d_nobias` | dynamo | `Conv` | 0/256 | 0 | 0 | BitExact |
| `conv1d_basic` | ts | `Conv` | 7/20 | 1.19e-7 | 4.26e-6 | Req7Provisional（暫定 pass） |
| `conv1d_basic` | dynamo | `Conv` | 7/20 | 1.19e-7 | 4.26e-6 | Req7Provisional（暫定 pass） |
| `maxpool2d_basic` | ts/dynamo | `MaxPool` | 0/48 | 0 | 0 | BitExact |
| `maxpool2d_pad_dil_ceil` | ts/dynamo | `MaxPool` | 0/48 | 0 | 0 | BitExact |
| `maxpool1d_basic` | ts/dynamo | `MaxPool` | 0/15 | 0 | 0 | BitExact |
| `avgpool2d_include_pad` | ts/dynamo | `AveragePool` | 31/48 | 5.96e-8 | 4.21e-6 | Req7Provisional（暫定 pass） |
| `avgpool2d_exclude_pad` | ts/dynamo | `AveragePool` | 30/48 | 5.96e-8 | 1.25e-6 | Req7Provisional（暫定 pass） |
| `avgpool2d_ceil_overhang_incl` | ts | `AveragePool` | 19/48 | 5.96e-8 | 6.19e-7 | Req7Provisional（暫定 pass） |
| `avgpool2d_ceil_overhang_incl` | dynamo | `AveragePool` | 19/48 | 5.96e-8 | 6.19e-7 | Req7Provisional（暫定 pass） |
| `avgpool2d_ceil_overhang_excl` | ts/dynamo | `AveragePool` | 22/48 | 1.19e-7 | 6.80e-6 | Req7Provisional（暫定 pass） |
| `avgpool1d_basic` | ts/dynamo | `AveragePool` | 6/15 | 1.19e-7 | 1.08e-7 | Req7Provisional（暫定 pass） |
| `gap2d` | ts | `GlobalAveragePool` | 1/3 | 3.73e-9 | 6.67e-8 | Req7Provisional（暫定 pass） |
| `gap2d` | dynamo | `ReduceMean` | 3/3 | 1.49e-8 | 2.00e-7 | Req7Provisional（暫定 pass） |
| `gap1d` | ts | `GlobalAveragePool` | 0/3 | 0 | 0 | BitExact |
| `gap1d` | dynamo | `Unsqueeze, ReduceMean, Squeeze` | 2/3 | 5.96e-8 | 9.23e-8 | Req7Provisional（暫定 pass） |
| `bn2d_eval` | ts/dynamo | `BatchNormalization` | 54/150 | 4.77e-7 | 9.01e-7 | Req7Provisional（暫定 pass） |
| `bn2d_eval_eps` | ts/dynamo | `BatchNormalization` | 83/150 | 2.38e-7 | 6.06e-7 | Req7Provisional（暫定 pass） |
| `bn1d_eval` | ts/dynamo | `BatchNormalization` | 22/42 | 1.19e-7 | 1.29e-7 | Req7Provisional（暫定 pass） |
| `flatten_default` | ts | `Flatten` | 0/120 | 0 | 0 | BitExact |
| `flatten_default` | dynamo | `Reshape` | 0/120 | 0 | 0 | BitExact |
| `flatten_start2` | ts | `Shape, Constant×4, Slice, Concat, Reshape` | 0/120 | 0 | 0 | BitExact |
| `flatten_start2` | dynamo | `Reshape` | 0/120 | 0 | 0 | BitExact |

**全 38 ケース×exporter の組で `req7_fail_count=0`**（暫定 REQ-7 判定でも
fail するケースは実測で無かった）。`max_rel_err` の最大値は
`avgpool2d_ceil_overhang_excl` の `6.80e-6`（イシュー #2329 PR #2343
codex-review 指摘対応で fixture の入力形状を `7x7` から `6x6` へ変更し
再実測。旧実測では `avgpool2d_include_pad` の `4.2e-6` が最大だった）で、
閾値 `1e-3` に対して十分な余裕がある。

## R1〜R6 との対応（受け入れ基準チェック）

| # | 要件 | 結果 |
|---|---|---|
| R1 | 対象 6 op ごとに `torch.onnx.export` 実生成 `.onnx` をコミット | 達成。19 ケース × 2 exporter = 38 ファイル（272 KB） |
| R2 | initializer と `state_dict` の bit 一致 | 達成。全ケース `assert_r2_initializers_match_state_dict` で bit 完全一致検証済み |
| R3 | 純粋な選択・形状操作（MaxPool・Flatten）の bit 完全一致 | 達成。全ケースで `bit_mismatch=0` |
| R4 | 縮約系の実測記録・暫定判定 | 実測記録は完了（本ファイル上表）。ただし縮約系（Conv・AveragePool・BatchNormalization・`gap*`／`ReduceMean` 経路）は bit 一致しないケースが多く、暫定 REQ-7 判定（`Req7Provisional`）で全ケース pass を確認したのみ。**この暫定判定を最終判定方式として採用するかはユーザー承認待ちであり、R4 は未確定**（達成とは言わない） |
| R5 | import 失敗・非対応ケースの列挙 | 下記「R5: import 非対応ケース」参照 |
| R6 | #2185 の受け入れ条件（「PyTorch の ONNX export を import して bit 同一を確認する」）との対応 | **一部未達**。純粋な選択・形状操作系（MaxPool・Flatten。R3）は bit 同一を達成したが、縮約系（R4）は bit 同一ではなく `Req7Provisional` という暫定基準で pass 扱いにしている。すなわち #2185 の受け入れ条件を縮約系についてはそのままの形では満たせておらず、暫定基準への切り替え可否はユーザー承認待ち（下記「承認待ち事項」1.）。承認が得られるまで R6 は縮約系について未確定のまま据え置く |

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
非対応のまま維持するか）はユーザー判断に委ねる**（修正するか別 issue に
するかは本 PR のスコープ外）。

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
実際に越える）。実測ではいずれも暫定 REQ-7 判定を通過しており
（`max_rel_err` はそれぞれ `6.19e-7`・`6.80e-6`）、この divisor
規則が PyTorch 実行値と整合することを確認した。

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

## 承認待ち事項（まとめ）

1. **暫定 REQ-7 判定の適用**: 縮約系の一部ケースで bit 一致しなかったため
   既存の REQ-7 事前固定式を暫定適用した（tolerance の新設・緩和ではなく
   既存式の再利用）。最終判定方式としての採否はユーザー承認事項。
2. **external data 非対応**: dynamo exporter の既定挙動（external data）
   への対応要否（実装するか、非対応のまま維持し fixture 側で inline 化
   する運用を継続するか）はユーザー判断事項。
