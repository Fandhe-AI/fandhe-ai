# ONNX external data（外部 `.onnx.data` ファイル）import 対応の実測ログ（イシュー #2347）

## 位置づけ

`onnx::external_data::build_graph_with_external_data`（`crates/
onnx-interop/src/onnx/external_data.rs`）が、PyTorch dynamo exporter の
生の external data 出力（再 inline 化しない `.onnx` + companion
`.onnx.data`）を実際に import できることを、`torch==2.14.0+cpu` を
使い捨て venv へ導入して実測した記録。決定記録は
`docs/onnx-external-data-decision.md` を正とする（本ファイルは実測結果
のみを記録し二重管理しない）。fixture・生成環境の詳細は `crates/
onnx-interop/tests/fixtures/pytorch-onnx-external-data/README.md`（正）を
参照する。

## 生成環境

- `torch==2.14.0+cpu`・`onnx==1.23.0`・`onnxscript==0.7.2`
- Python 3.14.4・x86_64・Linux
- `../pytorch-onnx-cnn-ops/gen_reference.py` と同じ `CASES`・
  `deterministic_seed`（ケース名の MD5 先頭 4 バイトを `torch.manual_seed`
  へ渡す）を import で再利用

## 実測結果（対象ケースの絞り込み）

19 ケース全件に対して dynamo export を試みたところ、実際に initializer
が external data になったのは **Conv 系の重み（432／288 バイト）3 ケース
のみ**だった（`conv2d_basic`・`conv2d_nobias`・`conv2d_stride_dil_group`。
`conv.weight` が external・`conv.bias`〈存在する場合〉は 16 バイトで
inline のまま）。他 16 ケース（Pool・BN・Flatten・GAP）は companion
`.data` ファイル自体は作られるが 0 バイト（external な initializer が
1 件も無い）だった。INT64 の shape 定数（`flatten_start2`・`gap1d` 等）も
16〜24 バイトと小さく external にならなかった。この実測に基づき、
時間制約もあり fixture を上記 3 ケースへ絞り込んだ（実装計画 §4.5-4
「INT64 が external にならなかった場合は実測値に合わせて要件を調整」に
対応する調整。A1 要件〈F32 initializer が 1 件以上 external〉はこの
3 ケースで満たす）。

## 重要な実測上の注意: 重みの再現性

同じ `deterministic_seed`／`CASES`（同じ MD5 由来シード・同じモデル定義）
を import し、同じ torch/onnx バージョン・同じ Python バージョン・同じ
プラットフォーム（`../pytorch-onnx-cnn-ops/reference.json` の `meta` 節と
完全一致）で `torch.manual_seed(seed)` 直後に `nn.Conv2d` を構築しても、
本実装時（2026-09-28）に生成した `conv2d_basic` の `conv.weight` 先頭
4 要素は `[0.1838, -0.0940, -0.0023, 0.1748]` となり、`../pytorch-onnx-
cnn-ops/reference.json`（#2329・2026-09-28 生成記録）の `state_dict` に
記録された値 `[-0.0425, 0.0992, -0.1599, -0.0462]` と一致しなかった。
原因は特定していない（PyTorch 内部初期化アルゴリズムの非公開な実装
詳細・生成環境の微差等が考えられるが未調査）。

このため本 fixture は `../pytorch-onnx-cnn-ops/reference.json` を
再利用せず、`manifest.json` に自己完結の参照値（このスクリプトが今回
生成した重みに対する、このスクリプトが今回計算した PyTorch 参照出力）
を保持する方式に変更した（`gen_external.py`・fixture `README.md`
「参照値が自己完結である理由」節参照）。この非再現性自体を
`docs/onnx-external-data-decision.md` §7 のスコープ外事項として起票候補
に記録する（自動運転中のため未起票）。

## テスト実行コマンド

```bash
cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture -- --nocapture
```

`external_data_fixture_matches_self_contained_reference`・
`external_data_fixture_actually_uses_external_path`・
`external_data_fixture_bytes_entry_point_still_rejects` の 3 テストが
pass（他 40 テストは `../pytorch-onnx-cnn-ops/` 側で従来どおり pass）。
`cargo test -p fandhe-ai-onnx-interop --test onnx_external_data`
（合成入力の網羅テスト 39 件）もすべて pass。

## CI への影響

CI（GitHub ホステッド）はコミット済み fixture（`.onnx`／`.onnx.data`／
`manifest.json`）のみを読み、torch には依存しない（`../pytorch-onnx-
cnn-ops/README.md` と同じ方針。`.claude/rules/ci.md`「グローバル状態を
汚す処理を workflow に書かない」）。
