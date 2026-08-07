# `transformer.onnx` end-to-end 実測・対応可否確定 記録（#88・TASK-7.4b）

イシュー #88「docs(interop): TASK-7.4b 対応可否確定・実測結果の記録」の記録文書。
親タスク TASK-7.4（`docs/spec/05-tasks.md:255`、REQ-7 受け入れ基準
`docs/spec/04-requirements.md:161`）の成果物「`transformer.onnx` end-to-end 実測結果記録」に対応する。
受け入れ条件「実測結果記録と対応可否の結論が文書化されている」を満たすことを目的とする。

## 対応可否の結論: 条件付き非対応（現時点。インタープリタの I64 算術未対応で推論が停止）

**イシュー #87（TASK-7.4a）で e2e 推論実測テスト・フィクスチャを実装し、実際に実行した結果、
`transformer.onnx` の推論は `decode`・`build_graph` までは成功するが、`interp::run` 実行中に
`/layers.0/self_attn/Div` ノードで停止し、対応可否を数値一致まで確定できないことを実測確認した。**

判定根拠（下記「前提確認の実測」節に詳細）:

1. `decode → build_graph` は成功する。`node=165`・`initializer=12`・`op_type` 種別数 20 を実測確認済み
   （`tests/onnx_decode.rs` の `transformer_onnx_decodes_expected_graph_structure` と同一実測値）
2. `interp::run` は `/layers.0/self_attn/Div`（入力: `Shape -> Gather` 経由の I64 値、`head_dim` 算出用の
   整数除算）で `InterpError::TypeMismatch { node: "/layers.0/self_attn/Div", expected: "f32" }` を返し停止する。
   `compute_div`（`crates/onnx-interop/src/onnx/interp.rs:550`）が F32 のみを受け付け、`Shape` 由来の I64
   算術（`Div`・`Mul`）に非対応なため（静的走査で `Div` 1 件・`Mul` 3 件、計 4 箇所の同型経路を確認。詳細は
   「未対応要因」節）
3. Attention 系オペ（`MatMul`／`Softmax`／`Erf`。TASK-7.3c・#84）・`LayerNormalization`（TASK-7.3d・#85）は
   本セッション時点で実装済み（origin/main マージ済み）であることを確認済みだが、上記 I64 算術の欠落により
   その手前で停止するため、これらのオペ自体の e2e 経路での動作は未検証のまま

この状態は `docs/perf/cuda-tensor-core-measurement.md`（#64）の先例と同型（実測を試みた結果、判明した未対応
要因により最終結論を確定できない）であるため、同ファイルの構成に倣い実測事実をそのまま記録する。

**「条件付き非対応」は恒久的な判定ではなく「現時点で I64 算術対応待ちのため実測不能」を意味する。** REQ-7 が
定義する恒久的なスコープ外制約とは異なる（`docs/spec/04-requirements.md:160`）。

## 前提確認の実測（#87 実装セッションで再実施）

対象コミット SHA: `f90272a496e3fd26776d64a9b25a7656cafcce49`（origin/main、確認日時 2026-08-07）
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`

| 確認項目 | 結果 |
|---------|------|
| `gh pr list --search "Closes #87"`・`"87 in:title"` | 0 件（#87 の実装は本セッションが最初の着手） |
| `#78`・`#84`・`#85` の実装状況 | origin/main に既にマージ済み（`interp::run` が存在し `MatMul`／`Softmax`／`Erf`／`LayerNormalization` もディスパッチ対象。`crates/onnx-interop/src/onnx/interp.rs` 実測） |
| `transformer.onnx`（v1 リポ commit `a14568897521f7bea6eac93218fe917cf2a25f04`）取得・sha256 | 一致（`6f6430e6b99408c949635da16ed7d6e7cdc2a500db050ae80c660b3b8b057b0f`） |
| `reference.json`（同 commit）取得・sha256 | 一致（`84c0d0055ccd6a4cf32c4b5f9a0b6f6b1028e3344ded2f1d763ac426b41915c8`）。`crates/onnx-interop/tests/fixtures/pytorch-transformer/reference.json` としてコミット |
| `decode → build_graph`（`tests/onnx_transformer_e2e.rs` 実測） | 成功。`node=165`・`initializer=12` |
| `interp::run`（`tests/onnx_transformer_e2e.rs` 実測、`ONNX_INTEROP_TRANSFORMER_ONNX` 設定・`--ignored`） | 失敗。`/layers.0/self_attn/Div` で `InterpError::TypeMismatch { node: "/layers.0/self_attn/Div", expected: "f32" }` |
| `cargo fmt --all -- --check`／`cargo clippy -p onnx-interop --all-targets --all-features -- -D warnings` | pass |
| `cargo test -p onnx-interop`（`--ignored` なし。既存テスト回帰） | pass（新規 e2e テストは `#[ignore]` のためスキップ対象） |

## 対応オペ一覧（実装状況の突合表）

PoC-v2-6（`docs/spec/03-poc/poc-v2-6-interop/README.md:45`）実測の `transformer.onnx` 実オペ集合（20 種別）に対する、
本リポ `crates/onnx-interop/src/ops/mod.rs` 現在の実装状況（`pub use` 一覧・`grep` 実測、上表参照コミット時点）。

| オペ | 実装状況 | 実装元（TASK・イシュー） | 備考 |
|------|---------|---------------------|------|
| `Gemm` | 実装済み | TASK-7.2c（PoC-v2-6 由来） | `crates/onnx-interop/src/ops/gemm.rs` |
| `Relu` | 実装済み | TASK-7.2c | `ops/activation.rs` |
| `Sigmoid` | 実装済み | TASK-7.2c | `ops/activation.rs` |
| `Shape` | 実装済み | TASK-7.2c | `ops/shape_ops.rs` |
| `Gather` | 実装済み | TASK-7.2c | `ops/gather.rs` |
| `Unsqueeze` | 実装済み | TASK-7.2c | `ops/shape_ops.rs` |
| `Concat` | 実装済み | TASK-7.2c | `ops/concat.rs` |
| `Slice` | 実装済み | TASK-7.2c | `ops/slice.rs`（動的境界パターン対応済み。PoC-v2-6 `slice_repro.onnx` 実測 相対誤差 0.000000） |
| `Add` | 実装済み | TASK-7.3a・#82 | `ops/arith.rs` |
| `Mul` | 実装済み | TASK-7.3a | `ops/arith.rs` |
| `Div` | 実装済み | TASK-7.3a | `ops/arith.rs` |
| `Mod` | 実装済み | TASK-7.3a | `ops/arith.rs` |
| `Sqrt` | 実装済み | TASK-7.3a | `ops/arith.rs` |
| `Constant` | 実装済み | TASK-7.3a | `ops/constant.rs` |
| `Cast` | 実装済み | TASK-7.3b・#83 | `ops/cast.rs` |
| `Reshape` | 実装済み | TASK-7.3b | `ops/shape_transform.rs` |
| `Squeeze` | 実装済み | TASK-7.3b | `ops/shape_transform.rs` |
| `Transpose` | 実装済み | TASK-7.3b | `ops/shape_transform.rs` |
| `MatMul` | 実装済み | TASK-7.3c・#84 | `ops/matmul.rs`（Attention 系・バッチ行列積） |
| `Softmax` | 実装済み | TASK-7.3c・#84 | Attention 系 |
| `Erf` | 実装済み | TASK-7.3c・#84 | Attention 系（GELU 等で使用） |
| `LayerNormalization` | 実装済み | TASK-7.3d・#85 | 正規化層系 |

被覆率: 20/20 種別（100%）。オペ関数自体・インタープリタディスパッチ（TASK-7.2b・#78）はいずれも
origin/main にマージ済みで、`interp::run` からすべて到達可能であることを実測確認した。

**新たに判明した未対応要因（オペ被覆とは別軸）**: オペ関数の実装状況・ディスパッチの有無だけでは
対応可否を過大評価しうる。実際に `transformer.onnx` を `interp::run` へ通すと、`Shape -> Gather -> Div`
（`head_dim` 算出等のグラフ内整数演算）で `Div` が I64 入力を受け付けず停止することが分かった
（次節「未対応要因の有無」参照）。

## 未対応要因の有無: あり（インタープリタの I64 算術未対応）

- **F32 限定オペが `Shape` 由来の I64 算術に非対応**: `compute_div`／`compute_mul`／`compute_add`／
  `compute_mod`／`compute_sqrt`（`crates/onnx-interop/src/onnx/interp.rs:550` 付近）はいずれも
  `get_f32` で入力を取得し、`Value::I64` を渡すと `InterpError::TypeMismatch` で拒否する。
  `transformer.onnx` は `Shape -> Gather` で得た I64 のテンソル次元値を `Div`／`Mul` で算術演算する
  （PyTorch `nn.TransformerEncoderLayer` のトレース由来。`head_dim = d_model // n_heads` 相当）ため、
  この経路で必ず停止する
  - 実測で確認した最初の停止点: `/layers.0/self_attn/Div`（`TypeMismatch { expected: "f32" }`）
  - 静的走査（`Shape` からの経路を `Gather`／`Squeeze`／`Unsqueeze` を透過して追跡）で同型の疑い
    箇所を計 4 件確認（`Div` 1 件・`Mul` 3 件、いずれも `/layers.0/self_attn/` 配下）。実測で確認した
    のは最初の 1 件のみで、これを解消すると次の同型経路で再度停止する可能性が高い
  - 対応範囲は TASK-7.3a（`Add`/`Mul`/`Div`/`Mod`/`Sqrt`。#82・クローズ済み）の残課題に相当し、本ファイル
    の担当（#88・記録専任）・#87（テスト専任）のいずれのスコープでもないため、本セッションでは実装しない
    （`out-of-scope-tracking.md`。新規 Issue 起票はユーザー承認が必要なため本セッションでは行わず、
    #87 の実装報告に out-of-scope 項目として記録する）

動的境界 Slice パターン（v1 の `burn-onnx` 失敗パターン、`docs/spec/04-requirements.md:161`）は
**対応済み**であることを実測根拠とともに再確認した（PoC-v2-6 `slice_repro.onnx` 実測、相対誤差 0.000000、
`docs/spec/03-poc/poc-v2-6-interop/README.md:35`）。本リポの `crates/onnx-interop/src/ops/slice.rs` も
同パターンに対応するオペ関数として実装済みである。したがって「未対応要因」は動的境界パターンという
構造的な問題ではなく、`Shape` 由来の I64 算術という別のグラフパターンに起因する。

## 数値一致の実測結果: 未実施

`transformer.onnx` end-to-end 推論が `interp::run` の I64 算術非対応で完走しないため、REQ-7 判定式
`abs_err / (|ref| + 1e-6) ≤ 1e-3`（`docs/spec/04-requirements.md:159`）による実測値は取得していない。

参考値（個別オペ・別モデルでの既存実測。e2e とは別スコープの数値であり、上記判定式の対象は
あくまで `transformer.onnx` 全体である点に注意）:

- safetensors 経路（MLP、PoC-v2-6）: 最大相対誤差 0.000000
- ONNX インタープリタ経路（MLP `model.onnx`）: 最大相対誤差 0.000000
- ONNX インタープリタ経路（動的境界 Slice 最小再現グラフ `slice_repro.onnx`）: 最大相対誤差 0.000000

**注記（REQ-2 との混同回避）**: 上記判定式は REQ-7 固有の指標であり、REQ-2 のバックエンド間数値一致
統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」（`docs/spec/04-requirements.md:159` 明記の
とおり）とは別指標である。両者を混同しないこと。

## 性能の実測結果: 未実施

e2e 推論が完走しないため実行時間の実測値はない。対 PyTorch 性能下限の正式確定は TASK-8.3（#154、
bench-harness）のスコープであり、本記録は I64 算術対応後も参考値の位置づけにとどまる
（`docs/spec/04-requirements.md:167` REQ-8・`docs/spec/06-roadmap.md` 依存関係
`TASK-7.4 → TASK-8.3` 参照）。

## 再現手順（インタープリタの I64 算術対応後に本節のコマンドで再実測し、下記テンプレートへ転記する）

```sh
git fetch origin
git checkout main   # I64 Div/Mul 等の対応マージ後
ONNX_INTEROP_TRANSFORMER_ONNX=<取得した transformer.onnx のパス> \
  cargo test -p onnx-interop --test onnx_transformer_e2e -- --ignored --nocapture
```

現時点で確認済みの回帰チェック（#87 実装セッションで実施）:

```sh
cargo fmt --all -- --check                                              # pass
cargo clippy -p onnx-interop --all-targets --all-features -- -D warnings # pass
cargo test -p onnx-interop                                              # pass（新規 e2e テストは #[ignore] でスキップ）
```

### 実測結果（#87 実装セッションで判明した範囲まで記入。数値一致・性能は I64 算術対応後に追記）

| 項目 | 値 |
|------|-----|
| e2e テスト名（#87 で確定） | `transformer_onnx_end_to_end_matches_pytorch_reference_within_req7_tolerance`（`crates/onnx-interop/tests/onnx_transformer_e2e.rs`） |
| `transformer.onnx` の parse・グラフ構築 | pass（`node=165`・`initializer=12`） |
| 未対応オペ検出（実行時） | なし（オペ被覆 20/20。ただし `interp::run` は F32 限定オペの I64 入力非対応で `/layers.0/self_attn/Div` にて `TypeMismatch` で停止） |
| 最大相対誤差（REQ-7 判定式） | 未取得（推論未完走） |
| 実行時間（CPU インタープリタ経路） | 未取得（推論未完走） |
| 対応可否の最終結論 | 条件付き非対応（I64 算術対応後に再実測が必要） |
| commit SHA | `f90272a496e3fd26776d64a9b25a7656cafcce49`（origin/main） |
| 実施日 | 2026-08-07 |

複合判定・数値一致が実測で外れた場合は許容誤差を緩和せず、本節に実測値・エラー内容を記録したうえで
制約事項として扱う（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず人間の承認を
経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。

## #89（TASK-7.5 移行チェックリスト）への引き継ぎ事項

- 本ファイルの「対応可否の結論」は、インタープリタの I64 算術対応（`compute_div`／`compute_mul` 等）が
  完了し e2e が完走した後に実測値で更新される。#89 の移行チェックリスト作成時は本ファイルの最新版
  （実測後）を参照し、実測未完了のままチェックリストへ「対応済み」と記載しないこと
- 未対応要因が実測で新たに見つかった場合（I64 `Div`／`Mul` 以外にも、静的走査で疑いのある箇所が
  残っている）は、`docs/spec/06-roadmap.md:211` のリスク対応方針（「構造的な矛盾でなければ Phase 4 への
  差し戻しは不要」）に従い、制約事項として本ファイルに記録したうえで #89 へ引き継ぐ
- I64 算術対応の実装は、`crates/onnx-interop/src/onnx/interp.rs` の `compute_div`／`compute_mul`／
  `compute_add`／`compute_mod`／`compute_sqrt` が `Value::I64` も受け付けるようディスパッチを拡張する
  形が想定される（`get_f32` 限定の現行実装からの変更）。本ファイルの「未対応要因の有無」節が
  その時点での対応範囲を示す参照点になる

## 関連イシューとの役割分担（二重管理を避ける）

- **#87**（TASK-7.4a・e2e 推論実測テストの実装）: 実測コード・フィクスチャ・数値比較の実装そのものを担う。
  本ファイル（#88・TASK-7.4b）はその実測結果の記録・対応可否の結論確定に専念し、テストコード自体は
  実装しない。#87 で判明した I64 算術未対応の詳細は本ファイルへ転記済み
- **#78**（TASK-7.2b・インタープリタ本体）・**#84**（TASK-7.3c・Attention 系）・**#85**（TASK-7.3d・
  `LayerNormalization`）: いずれも実装完了（origin/main マージ済み）。オペ被覆・ディスパッチの
  観点では #87 のブロッカーではなくなった
- **#89**（TASK-7.5・移行チェックリスト）: 本ファイルの実測結果を踏まえた PyTorch 移行チェックリストの作成。
  本ファイルは実測記録に留め、チェックリスト形式の成果物は作成しない
- **#154**（TASK-8.3・bench-harness 対 PyTorch 性能下限確定）: 本ファイルの性能実測は参考値に留め、
  正式な下限確定は #154 のスコープ

## 未実施・後続作業

- インタープリタの I64 算術（`Shape` 由来の `Div`／`Mul` 等）対応、および対応後の再実測による本ファイルの
  「実測結果」節の更新（新規 Issue 起票はユーザー承認が必要なため、本セッションでは行わない。#87 の
  実装報告に out-of-scope 項目として記録する）
- REQ-7 側のステータス更新（受け入れ基準チェックボックスの反映等）が必要な場合、`docs/spec/`
  は本リポで編集しないため spec リポ（`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーに
  提案するに留める
