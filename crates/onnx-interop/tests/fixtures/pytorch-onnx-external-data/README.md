# PyTorch dynamo exporter の external data 生出力 fixture（イシュー #2347）

## 位置づけ

`../pytorch-onnx-cnn-ops/`（イシュー #2329）は PyTorch dynamo exporter が
initializer を external data（`.onnx.data` companion ファイル）として
出力した場合、生成スクリプト側で本体へ inline し直してから fixture 化
していた（`../pytorch-onnx-cnn-ops/README.md` R5 節・
`gen_reference.py::export_one` 参照）。本ディレクトリは #2347（ONNX
external data の initializer 読み込み対応）が、**再 inline 化しない生の
dynamo 出力**（`.onnx` + companion `.onnx.data`）を import できることを
確認する fixture である。

## 生成方法

`gen_external.py` は `../pytorch-onnx-cnn-ops/gen_reference.py` の
`CASES`・`deterministic_seed`・`tensor_record` を import で再利用し、
同じ決定的シードでモデル・入力を再構成する。dynamo exporter で export した
のち、companion `.data` を削除・再 inline せずそのまま残す点のみが
`gen_reference.py::export_one` と異なる。

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
/path/to/venv/bin/pip install onnx==1.23.0 onnxscript==0.7.2
/path/to/venv/bin/python gen_external.py   # このディレクトリで実行
```

生成環境: `torch==2.14.0+cpu`・`onnx==1.23.0`・`onnxscript==0.7.2`・
Python 3.14.4・x86_64・Linux（`../pytorch-onnx-cnn-ops/README.md` と同一
構成）。

## 実測結果（対象ケースの絞り込み）

`../pytorch-onnx-cnn-ops/` の 19 ケースすべてに対して dynamo export を
試みたところ、実際に initializer が external data
（`data_location=EXTERNAL`・非空の companion `.data`）になったのは
**Conv 系の重み（432／288 バイト）のみ**だった。他ケース（Pool・BN・
Flatten・GAP）は companion `.data` ファイル自体は作られるが 0 バイト
（external な initializer が 1 件も無い）であり、external data 経路を
検査する fixture として意味を持たない。

INT64 の shape 定数（`flatten_start2` の `Reshape` shape・`gap1d` の
`Unsqueeze`／`Squeeze` shape 等）も 16〜24 バイトと小さく、external には
ならなかった（実装計画 §4.5-4 の「INT64 が external にならなかった場合は
実測値に合わせて要件を調整」に基づく調整。A1 要件は F32 initializer が
1 件以上 external であることで満たす）。

このため本 fixture は 3 ケースへ絞り込んだ（時間制約による絞り込みで
あることも記録する）:

| ケース | initializer | dims | バイト数 |
|---|---|---:|---:|
| `conv2d_basic` | `conv.weight` | `[4,3,3,3]` | 432 |
| `conv2d_nobias` | `conv.weight` | `[4,3,3,3]` | 432 |
| `conv2d_stride_dil_group` | `conv.weight` | `[4,2,3,3]` | 288 |

いずれも `conv.bias`（存在する場合）は 16 バイトで external にならず
inline のまま（`raw_data` 経由）。

## 参照値が自己完結である理由（重要な実測上の注意）

**`../pytorch-onnx-cnn-ops/reference.json` は再利用しない。** 同じ
`deterministic_seed`／`CASES`（同じ MD5 由来シード・同じモデル定義）を
import して `torch.manual_seed(seed)` 直後に `nn.Conv2d` を構築しても、
本実装時（2026-09-28・イシュー #2347）に生成した重みの bit パターンは
`../pytorch-onnx-cnn-ops/reference.json`（イシュー #2329・2026-09-28 生成
記録）の `state_dict` と一致しなかった（`conv2d_basic` の `conv.weight`
先頭 4 要素で実測差異を確認: 新規生成 `[0.1838, -0.0940, -0.0023,
0.1748]` に対し既存記録 `[-0.0425, 0.0992, -0.1599, -0.0462]`）。torch・
onnx・Python バージョン・プラットフォームはいずれも一致しているため
（`meta` 節参照）、原因は特定していない（PyTorch 内部の初期化アルゴリズム
の非公開な実装詳細・生成環境の微差等が考えられるが未調査。決定記録
`docs/onnx-external-data-decision.md` に「起票候補」として記録する）。

このため `manifest.json` の各ケースエントリは **このスクリプトが今回
生成した重みに対する、このスクリプトが今回計算した PyTorch 参照出力**
（`input`／`output`。`tensor_record` 形式）を自己完結で保持する。判定
（`tests/onnx_interp_pytorch_cnn_fixture.rs::
external_data_fixture_matches_self_contained_reference`）はこの自己完結
ペアに対してのみ行い、`../pytorch-onnx-cnn-ops/` 側の baseline・
reference.json とは独立である。

## テスト実行コマンド

```bash
cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture -- --nocapture
```

`external_data_fixture_matches_self_contained_reference`・
`external_data_fixture_actually_uses_external_path`・
`external_data_fixture_bytes_entry_point_still_rejects` の 3 テストが
本 fixture を対象とする（他 40 テストは `../pytorch-onnx-cnn-ops/` 側）。
