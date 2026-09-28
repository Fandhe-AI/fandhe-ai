# PyTorch 実生成 ONNX Conv・Pool・BN・Flatten fixture

イシュー #2329（親 #2185）。`tests/onnx_interp_pytorch_cnn_fixture.rs` が
参照する固定 fixture。`torch.onnx.export` が実生成した `.onnx`（TorchScript
exporter・dynamo exporter の両方）と、PyTorch の実行結果・`state_dict` を
`reference.json` へ記録している。

`crates/onnx-interop/tests/onnx_interp_cnn_ops.rs`（#2200）・
`tests/onnx_interp_conv_pool.rs`（#2199）は
`fandhe_ai_autodiff::nn::*::forward_host` との**内部突合**（自作 export →
自作 import の自己整合）に留まっていた
（`docs/perf/logs/onnx-conv-pool-import-2199/README.md`
「torch 実生成 fixture による突合の未実施」節）。本 fixture はその残作業を
実施し、PyTorch が実際に出す ONNX 表現を import できるかを検証する。

## 生成環境

- `torch==2.14.0+cpu`（`pip install --index-url
  https://download.pytorch.org/whl/cpu torch`）
- `onnx==1.23.0`・`onnxscript==0.7.2`（dynamo exporter が要求する）
- Python 3.14.4・x86_64・Linux
- `torch.set_num_threads(1)`・`model.eval()`・`torch.no_grad()`・
  `torch.manual_seed(<ケース名の MD5 先頭 4 バイト>)`（決定的な乱数列。
  Python 組込み `hash()` は文字列に対して既定でプロセスごとに salt が
  かかり非決定的なため使わない。`gen_reference.py::deterministic_seed`）

## exporter

各ケースを 2 通りの exporter で export している:

- `ts`: `torch.onnx.export(..., dynamo=False, opset_version=13)`
  （TorchScript ベースの旧 exporter。PyTorch 2.9 以降 `DeprecationWarning`
  が出るが動作する）
- `dynamo`: `torch.onnx.export(..., dynamo=True)`（既定 exporter。opset 20）

**dynamo exporter の external data 既定挙動**: dynamo exporter はテンソル
サイズに応じて initializer を external data
（`<name>.onnx.data` companion file・`TensorProto.data_location=EXTERNAL`）
へ既定で逃がす（本 fixture 生成中に実測で判明した挙動。`conv.weight`
〈432 バイト〉は external、`conv.bias`〈16 バイト〉は inline のままだった
ことから、閾値はごく小さい）。本クレートの `onnx::graph::decode_tensor`
は external data を意図的に非対応（`raw_data`/`float_data` の完全一致検証
のみを行い、`external_data`/`data_location` フィールドは未宣言のまま無視
する設計）で、external data 付きの `.onnx` を素直に読むと
`GraphError::RawDataByteLenMismatch`（期待バイト長 > 0・実バイト長 0）で
拒否される。この動作自体は fail-closed で正しい（無言で 0 埋めしたり
黙ってスキップしたりしない）が、本 fixture の主目的（CNN op の意味論突合）
には無関係な失敗要因のため、`gen_reference.py::export_one` は export 直後に
`onnx.load`（既定で external data をメモリへ読み込む）→
`onnx.save_model(mdl, path, save_as_external_data=False)` で companion file
を消し、自己完結 `.onnx` へ inline し直してからコミットしている
（各ケースの `was_external_data`（`reference.json`）が `true` の場合、
生の dynamo 出力は external data 付きだったことを示す）。

**R5 の記録（import 非対応ケース）**: 上記の「dynamo exporter の生
external data 出力」自体が、本クレートが意図的に非対応とする ONNX
表現の実例である。inline 化前の生ファイルを import しようとすると
`GraphError::RawDataByteLenMismatch` で拒否される。external data
サポートの要否は別途ユーザー判断（修正するか別 issue にするか）に
委ねる。これ以外に import が失敗した exporter 出力・op_type はない
（inline 化後の 19 ケース × 2 exporter = 38 通りすべてで `run` が成功
した。下表参照）。

**external data 非対応はイシュー #2347 で対応済み**: 新規モジュール
`onnx::external_data`（`onnx-interop` 内部限定。基点ディレクトリを受け
取る新入口 `build_graph_with_external_data`）が、上記の生 dynamo 出力
（再 inline 化しない external data 付き `.onnx` + `.onnx.data`）を
fail-closed に検証・読み込めるようにした。バイト列入口（本 fixture が
使う `decode_model` → `build_graph`）自体は変更しておらず、
`RawDataByteLenMismatch` で拒否する挙動は不変（回帰テストで固定済み）。
生の external data 出力に対する専用 fixture は
`crates/onnx-interop/tests/fixtures/pytorch-onnx-external-data/` を参照
（決定記録は `docs/onnx-external-data-decision.md`）。

## ケース × exporter × 実際の op 列（実測）

`torch.onnx.export` は同じ PyTorch モジュールでも exporter によって
異なる ONNX op 列へ変換することがある（例: `AdaptiveAvgPool2d(1)` を
`ts` は `GlobalAveragePool` 1 op へ、`dynamo` は `ReduceMean` へ変換する）。
実測（`cargo test -p fandhe-ai-onnx-interop --test
onnx_interp_pytorch_cnn_fixture -- --nocapture` の `op_types=` 出力を転記）:

| ケース | ts の op 列 | dynamo の op 列 |
|---|---|---|
| `conv2d_basic` | `Conv` | `Conv` |
| `conv2d_stride_dil_group` | `Conv` | `Conv` |
| `conv2d_nobias` | `Conv` | `Conv` |
| `conv1d_basic` | `Conv` | `Conv` |
| `maxpool2d_basic` | `MaxPool` | `MaxPool` |
| `maxpool2d_pad_dil_ceil` | `MaxPool` | `MaxPool` |
| `maxpool1d_basic` | `MaxPool` | `MaxPool` |
| `avgpool2d_include_pad` | `AveragePool` | `AveragePool` |
| `avgpool2d_exclude_pad` | `AveragePool` | `AveragePool` |
| `avgpool2d_ceil_overhang_incl` | `AveragePool` | `AveragePool` |
| `avgpool2d_ceil_overhang_excl` | `AveragePool` | `AveragePool` |
| `avgpool1d_basic` | `AveragePool` | `AveragePool` |
| `gap2d` | `GlobalAveragePool` | `ReduceMean` |
| `gap1d` | `GlobalAveragePool` | `Unsqueeze, ReduceMean, Squeeze` |
| `bn2d_eval` | `BatchNormalization` | `BatchNormalization` |
| `bn2d_eval_eps` | `BatchNormalization` | `BatchNormalization` |
| `bn1d_eval` | `BatchNormalization` | `BatchNormalization` |
| `flatten_default` | `Flatten` | `Reshape` |
| `flatten_start2` | `Shape, Constant×4, Slice, Concat, Reshape` | `Reshape` |

dynamo が分解した `ReduceMean`／`Unsqueeze`／`Squeeze`／`Reshape` はいずれも
本クレートが実装済みのオペのため、分解経路でも `run` は成功する
（**別の演算列を経由して同じ計算を行う**。GPU device 実行の到達性は
これら分解後オペを含め本 fixture の対象外——`onnx::interp::run` の
ディスパッチはいずれもホスト実行のみで、`interp_device`
〈device 実行対象 op 一覧〉には Conv 系 6 op を含め含まれない）。

## 判定方式・実測結果

`tests/onnx_interp_pytorch_cnn_fixture.rs` モジュール doc の「判定方式に
ついての注記」・`docs/onnx-pytorch-fixture-reduction-parity-judgment-
decision.md` を正とする（2026-09-28 ユーザー承認で正式方式へ移行）。要約:

- **選択・形状操作**（`MaxPool`・`Flatten`）: フォールバックなしの bit
  完全一致のみを要求する（`Expectation::BitExact`）。全ケースで成立（実測）。
- **縮約系**（`Conv`・`AveragePool`・`GlobalAveragePool`・
  `BatchNormalization`。dynamo 分解経路の `ReduceMean` を含む）: bit 完全
  一致は結合順序差により原理的に目標にできないため受け入れ条件から外し、
  REQ-7 事前固定式（`abs_err / (|ref| + 1e-6) <= 1e-3` の `fail_count == 0`）
  と、ケースごとの実測上限 baseline（`REDUCTION_BASELINES`）への fail-closed
  非後退判定を併用する（`Expectation::Req7BaselineNonRegression`。実測で
  bit 一致したケース〈`conv2d_basic`・`conv2d_nobias`（ts/dynamo 双方）・
  `gap1d`〈`ts` のみ〉〉も含め全 14 ケース × 2 exporter = 28 行を baseline
  へ記録済み——実測で bit 一致したケースは ceiling を `0` として記録し、
  成立した厳しさをそのまま非後退契約に固定する）。全ケースが REQ-7 式・
  baseline 双方を通過した（実測の `max_rel_err` は最大でも
  `avgpool2d_ceil_overhang_excl` の `6.80e-6` 程度で、1e-3 の閾値に対して
  十分な余裕がある。詳細な実測値は
  `docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/README.md` を参照）。

## `count_include_pad`／`ceil_mode` の divisor クリップ規則の実証

`avgpool2d_ceil_overhang_incl`／`avgpool2d_ceil_overhang_excl`
（入力 `6x6`・kernel `3x3`・stride `2`・padding `1`・`ceil_mode=1`）は、
出力窓が入力+padding 領域からはみ出す（右端・下端）ケースを狙って
構成した。入力形状は当初 `7x7` だったが、この形状では `ceil_mode`
が生む最終窓の右端・下端が padded 領域（7+2*1=9）の終端に一致するのみで
実際にははみ出さず、divisor クリップ規則を実証できていなかった
（イシュー #2329 PR #2343 codex-review 指摘・2026-09-28 修正）。`6x6`
は floor_mode（3x3 出力）に対し `ceil_mode` が窓を 4x4 出力へ 1 行・
1 列増やし、その最終窓が padded 領域（6+2*1=8）を越えるため実際に
クリップが発生する（`divisor_override` で強制クリップ無効化した出力
との差分が非ゼロであることを実測確認済み。`7x7` では同じ比較の差分が
ゼロだった）。実測では両ケースとも REQ-7 式・baseline 双方の判定を
通過しており、`ops::pool` モジュール doc が記録する「PyTorch／ONNX
Runtime 準拠の padded 座標でのクリップ規則」が PyTorch 実行値と整合する
ことを実証した。

## R2: initializer と `state_dict` の対応

`reference.json` の各ケース・各 exporter は `name_map`
（ONNX initializer 名 -> PyTorch `state_dict` キー）を持つ。対応付けは
生成時（`gen_reference.py::export_one`）に**値ベース**（shape + 全要素の
bit パターンが一致する `state_dict` エントリを探索）で行っており、
テスト側の `assert_r2_initializers_match_state_dict` はこの対応表に
従って改めて bit 完全一致を検査する。対応が見つからない initializer
（`unmapped_initializers`。例: `Reshape`／`ReduceMean` の shape・axes
定数）はモデルパラメータではないため R2 の対象外。

## `.onnx` ファイル一覧・sha256

再生成時は以下の値と一致することを確認する（`sha256sum *` の出力）。

```
005953d44a6e15fb4df0e60782056b0660897706ac4c72379bf5b641e8d78c2c  avgpool2d_exclude_pad_dynamo.onnx
060eb0aea53a199a76370b3010dbfa1ee77ef17ee00128fb1df431538ba40fe9  maxpool2d_basic_ts.onnx
0ba69cc7c791726b167dcb29427d47b7e05214d3b22bd0c67b1a847dff53aca8  avgpool2d_include_pad_ts.onnx
1128eccb16d09e34e2c5ee89dc6cf5f8a3de8248f6f1d83bc16bd28832692eab  avgpool2d_ceil_overhang_excl_dynamo.onnx
1da92f41d55cec5e31183d07e25b897dfafb01deb0e4b937e304a16b354171fc  conv1d_basic_dynamo.onnx
2229d01aae1916a69bcc2f4126ae5baefabd77bca4cb27fa7bb09aa10e4f7cdd  flatten_default_dynamo.onnx
27d129e22c6c594f21016487b1bb02be925b0538f62414a5db364974373738fc  gap2d_dynamo.onnx
28a974354afc7377c595885e33a56b6baad52f0425361f2abcb90344abcbbe4f  gap1d_dynamo.onnx
2bb359aba5326fc932f29e8cb0e26661dc74a632ad8a95390d5ec51d2ceb6550  gen_reference.py
367085732674305654584ba45845fd7a8c535b8d21d00f38c5c5d22006381285  conv1d_basic_ts.onnx
397ce80626f7eb00f766fd35ff8da8bc6c7acff5ac96545ffd0cce2cf7dffa3f  conv2d_stride_dil_group_ts.onnx
43746486a1f834c9a85ad99e8e781ff0058a1008b68c41890bef0d2afbdb66c3  bn2d_eval_dynamo.onnx
438eaf0469d4467b7d8e3e99a919be562b8f0968f3ddfdad756e7daafe0bc0a0  maxpool2d_basic_dynamo.onnx
468d8267afab96dfdc533ec99d4ade1de0223df9770f0a92727c8ce4df3b5cf3  flatten_default_ts.onnx
499964f2598a6d666c20dfe4c5d621cc270a3294fb4a03f6d48bf1d416ce044b  maxpool2d_pad_dil_ceil_dynamo.onnx
672aa6c77938b7185a02c71ad90793c95f4ee7a314f99912230735976e7da6fa  reference.json
67bfb98449322f19b0925025ae3c724ad69ee758c1eddb0a01537b3872335444  gap2d_ts.onnx
68ea0d4a8f710a1771649de626fdfd8bb56a378f37fcad28b57c8f1a58a978cb  avgpool2d_ceil_overhang_excl_ts.onnx
6e7c6176316915a3e47272d3d17091d8cb237fa9c37f3aa0b002baf741095f35  gap1d_ts.onnx
70ccbe0110c7cb6165af565cbeff7906467376248c567636f7a8ea859194137d  bn1d_eval_ts.onnx
774ecefd246a476ff778ed8be2c9da5176c246f0948e4d29c7f0850b1607065f  flatten_start2_ts.onnx
7e361f9559b692b987d178c1919bceec3ef47989977da4b6bcfdc5852f4180f1  conv2d_basic_ts.onnx
8236f2c6553ba90992288cde01ebd74e1929382eed4fcc5214c0bf746fb462d4  maxpool2d_pad_dil_ceil_ts.onnx
8c45258c677bbc826763e91db6f814f84822c95c6a527e690698c944aa7e6cc5  bn2d_eval_eps_dynamo.onnx
90b6e0ea7b8a644305c17cb9a85ff6ee107a6afc507e49e0eebe0e6df933580f  avgpool1d_basic_dynamo.onnx
97936bb034e104bad927452850e69ca7ec10079f5c9736d87e146c1f236d8222  bn2d_eval_eps_ts.onnx
97c958872308dfa3ee26a71ceedf9d66c5801f898add907f270017e8f1dcbc23  maxpool1d_basic_dynamo.onnx
9c2766126ce399dc0b3af45e6ed253ba5ea3c5d304811648c50e58f0edd34cd4  conv2d_basic_dynamo.onnx
a65a00b7f8b9150e0b12d7e6204701628b1942b98db7b0937ff93920be2877c7  flatten_start2_dynamo.onnx
ac9bf7808c386b3833fe9687075767083d5dd955e7092d3244265ba6ac02027b  avgpool2d_ceil_overhang_incl_ts.onnx
bc5511f7710d3c4dc5c0051f20e2aa9a568dc33a83ccd277915b930e12bca639  avgpool2d_include_pad_dynamo.onnx
c83e383ee90a1806f8f5a7a04b93177acd6e2d422c02e07cc67a1725bfb3a301  bn1d_eval_dynamo.onnx
cd3429efbaee2bcfe7ff6ef530c0d5231d59018cc40dc7769af41fee355427e1  maxpool1d_basic_ts.onnx
cfc806d1b4495f5557c83fb44f0ccf3922fa4bc333e85083c0ff6d0dc35ae841  avgpool2d_exclude_pad_ts.onnx
da1fda04a3bf30a57f5b27df4bba48a0cc0f88c06a2eeea97c5835861c77b0b7  conv2d_nobias_ts.onnx
de09f29219bf124760113ca13a35b1b4a80025b384063293dfd2285b4f10b28f  avgpool1d_basic_ts.onnx
e38c75fe9b8c3abd43425325836049791046740c1cdeb31783885f410884e5fd  conv2d_stride_dil_group_dynamo.onnx
e7d62c9f3dc08e2f1f2bf9e633e631112e07c67166c9f524bbc720cbb082ff56  avgpool2d_ceil_overhang_incl_dynamo.onnx
fb687fcf22bf1ce068bb70386a93ce03afb3eaeed9e727c9446ce749018eae39  bn2d_eval_ts.onnx
fb68937cc64b693c7388f73ab488167d2046f3ef6d43f88b516de6966d69c1bc  conv2d_nobias_dynamo.onnx
```

## 再生成手順

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
/path/to/venv/bin/pip install onnx onnxscript
/path/to/venv/bin/python gen_reference.py   # このディレクトリで実行. *.onnx・reference.json を上書きする
sha256sum *.onnx reference.json gen_reference.py   # 本 README の値と照合する
```

CI はコミット済み `.onnx`・`reference.json` のみを読み、torch には依存
しない（`crates/autodiff/tests/fixtures/lbfgs-pytorch-reference/README.md`
と同じ方針）。
