# `transformer.onnx` end-to-end 実測・対応可否確定 記録（#88・TASK-7.4b）

イシュー #88「docs(interop): TASK-7.4b 対応可否確定・実測結果の記録」の記録文書。
親タスク TASK-7.4（`docs/spec/05-tasks.md:255`、REQ-7 受け入れ基準
`docs/spec/04-requirements.md:161`）の成果物「`transformer.onnx` end-to-end 実測結果記録」に対応する。
受け入れ条件「実測結果記録と対応可否の結論が文書化されている」を満たすことを目的とする。

## 対応可否の結論: 条件付き非対応（I64 算術対応後・e2e 完走後も REQ-7 数値一致基準を超過）

**イシュー #87（TASK-7.4a）残作業として、インタープリタの `i64` 算術対応（`Add`／`Mul`／`Div`／`Mod`。
`ops/arith.rs::{add_i64,mul_i64,div_i64,mod_i64}`・`onnx/interp.rs::{compute_add,compute_mul,compute_div,
compute_mod}`）を実装し、`transformer.onnx` の end-to-end 推論を完走させた。しかし出力を PyTorch 参照値
（`reference.json`）と REQ-7 判定式 `abs_err / (|ref| + 1e-6) ≤ 1e-3` で突合した結果、16,384 要素中 7 要素
（0.043%）が基準を超過し、受け入れ条件「PyTorch 出力と数値一致」を完全には満たせないことを実測確認した。**

判定根拠（下記「前提確認の実測」節に詳細）:

1. `decode → build_graph → interp::run` は完走する（1 回目の実測でブロッカーだった
   `/layers.0/self_attn/Div` の `TypeMismatch` は `i64` 算術対応で解消）。出力 shape は `reference.json`
   の `output_shape` と一致
2. REQ-7 判定式で全要素を突合した結果、`max_rel_err=0.007529871`（閾値 `1e-3` の約 7.5 倍）・
   `exceed_count=7/16384` 要素
3. **超過した 7 要素はいずれも参照値の絶対値が小さい（`|expected| < 1e-3`）。実測した最大絶対誤差は
   `2.7418137e-6` と極めて小さい。** REQ-7 判定式の分母 `|ref| + 1e-6` は参照値そのものが 0 に近い箇所で
   非常に小さくなるため、絶対誤差が極めて小さくても相対誤差が閾値を超えやすい（判定式の構造上の性質）。
   超過要素の絶対誤差自体は REQ-2 の複合判定が定める絶対誤差許容 `1e-5` 未満に収まっており、モデル全体
   としての推論結果は PyTorch 参照値に極めて近いが、REQ-7 の相対誤差単独判定はこれを「不一致」として扱う
4. Attention 系オペ（`MatMul`／`Softmax`／`Erf`）・`LayerNormalization`・`i64` 算術のいずれも
   `TypeMismatch` 等の型不一致・未対応オペによる停止はなく、実行時エラー由来の問題ではない

この状態は `docs/perf/cuda-tensor-core-measurement.md`（#64）の先例と同型（実測を試みた結果、判明した
未達要因により最終結論を確定できない）であるため、同ファイルの構成に倣い実測事実をそのまま記録する。

**許容誤差・REQ-7 判定式自体は緩和しない**（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の
変更は必ず人間の承認を経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で
緩和しない」）。この超過が「REQ-7 判定式の近ゼロ値での構造的特性」によるものか「累積丸め誤差等の実装上の
精度改善余地」によるものかの切り分け・対応方針の決定はユーザー承認を要するため、本セッションでは判定式・
許容誤差を一切変更せず、実測事実のみを記録して次工程（#89）へ引き継ぐ。

**「条件付き非対応」は恒久的な判定ではない。** REQ-7 が定義する恒久的なスコープ外制約とは異なる
（`docs/spec/04-requirements.md:160`）。

## 前提確認の実測（1 回目・#87 実装セッション初回）

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

## 前提確認の実測（2 回目・I64 算術対応後の再実測）

対象コミット: origin/main `021e8c2`（`f90272a` の直後にマージされた `test(interop): TASK-7.4a
transformer.onnx end-to-end 推論実測 (#300)` を含む）を基点に、本イシュー #87 の残作業ブランチで
`i64` 算術対応を実装したうえで再実測。rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`（1 回目と同一環境）

失敗ノード `/layers.0/self_attn/Div`（入力 `Gather_2_output_0`・`Constant_6_output_0`）の実測トレース
（`onnx::graph::build_graph` の decode 結果を直接読んで確認）:

- `Gather_2_output_0`: `Shape -> Gather` 経由（`Shape` は ONNX 仕様上常に `INT64` を出力するため `i64`）
- `Constant_6_output_0`: `Constant` の `value` 属性、`TensorProto.data_type=7`（`INT64`）を実測確認
- → `Div` は `(I64, I64)` の組（`F32`/`I64` 混在ではない）で発生していた。プラン前提「静的走査で
  I64/I64 と推定」を実測で裏付け
- `Div` の出力は直後の `Cast(to=7)`（`INT64 -> INT64` の恒等 Cast。PyTorch トレースの副生成物）を経由し、
  さらに 2 段目の `Cast(to=7)` を経て `Reshape` の shape 引数として使われる（`Sqrt` には到達しない）ため、
  `compute_sqrt` の `i64` 対応は不要と実測確認（計画の想定どおり）

| 確認項目 | 結果 |
|---------|------|
| `compute_add`／`compute_mul`／`compute_div`／`compute_mod` の `(I64, I64)` 対応実装 | 完了（`ops/arith.rs::{add_i64,mul_i64,div_i64,mod_i64}`。`checked_add`/`checked_mul`/`checked_div`/`checked_rem` でオーバーフロー・0 除算を型付きエラー拒否） |
| `decode → build_graph → interp::run`（`tests/onnx_transformer_e2e.rs`、`--ignored --nocapture`） | 完走（`TypeMismatch` 解消）。出力 shape が `reference.json` の `output_shape` と一致 |
| REQ-7 判定式による全要素突合 | 16,384 要素中 7 要素が超過（詳細は次表） |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy -p onnx-interop --all-targets --all-features -- -D warnings` | pass |
| `cargo test -p onnx-interop`（`--ignored` なし。既存テスト回帰） | pass（177 件、新規 `i64` 単体・結合テスト 15 件を含む） |

### 数値一致の実測結果（詳細）

| 項目 | 実測値 |
|------|--------|
| 総要素数 | 16,384（`output_shape` = batch × seq_len × d_model） |
| REQ-7 判定式超過要素数 | 7（0.043%） |
| 最大相対誤差 `max_rel_err` | `0.007529871`（閾値 `1e-3` の約 7.5 倍） |
| 最大絶対誤差 `max_abs_err`（全要素中の最大値） | `2.7418137e-6` |
| 超過 7 要素の参照値の絶対値 | いずれも `1e-3` 未満（近ゼロ値。分母 `|ref| + 1e-6` が小さいため相対誤差が拡大） |
| REQ-2 複合判定（絶対誤差 `1e-5` 未満）での評価 | 超過 7 要素を含め全要素が絶対誤差 `1e-5` 未満に収まる（参考。REQ-7 とは別指標であり判定には用いない） |

代表例（`transformer_onnx_end_to_end_matches_pytorch_reference_within_req7_tolerance` 実行時に最初に
`assert!` で検出された要素）: `(b=0, s=5, d=507)` `expected=0.00019158138` `actual=0.00019303149`
`abs_err=0.0000014501129` `rel_err=0.007529871`

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
| `Add` | 実装済み（`f32`／`i64` 両対応） | TASK-7.3a・#82（`i64` はイシュー #87 残作業） | `ops/arith.rs` |
| `Mul` | 実装済み（`f32`／`i64` 両対応） | TASK-7.3a（`i64` は #87 残作業） | `ops/arith.rs` |
| `Div` | 実装済み（`f32`／`i64` 両対応） | TASK-7.3a（`i64` は #87 残作業） | `ops/arith.rs` |
| `Mod` | 実装済み（`f32`／`i64` 両対応） | TASK-7.3a（`i64` は #87 残作業） | `ops/arith.rs` |
| `Sqrt` | 実装済み（`f32` 限定。ONNX 仕様上 float 型のみ対応のため `i64` 対応は行わない） | TASK-7.3a | `ops/arith.rs` |
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
`interp::run` からすべて到達可能であり、`i64` 算術対応後は e2e 実行時にも型不一致による停止なく
全オペが実行されることを実測確認した。

## 未対応要因の有無: なし（オペ・型対応は完了。REQ-7 数値一致のみ未達）

- 1 回目の実測で判明した「F32 限定オペが `Shape` 由来の `i64` 算術に非対応」は本セッションで解消した
  （`ops/arith.rs::{add_i64,mul_i64,div_i64,mod_i64}`・`onnx/interp.rs::{compute_add,compute_mul,
  compute_div,compute_mod}` が `(I64, I64)` の組を受け付ける。`f32`/`i64` 混在ペアは引き続き
  `InterpError::TypeMismatch` で拒否する。暗黙変換なし・`Cast` 明示経由の既存方針を維持）
- 静的走査で疑いのあった残り 3 件の `Mul`（`/layers.0/self_attn/` 配下）も、同じ `compute_mul` の
  `(I64, I64)` 分岐で処理され、e2e 完走時に型不一致エラーは発生しなかった
- 残る未達要因は「オペ・型対応の欠落」ではなく「REQ-7 数値一致判定式が近ゼロ参照値で相対誤差を
  拡大する」という判定式の構造的性質（「対応可否の結論」節 3. 参照）であり、本節が扱う実装上の
  未対応オペ・未対応型とは性質が異なる

動的境界 Slice パターン（v1 の `burn-onnx` 失敗パターン、`docs/spec/04-requirements.md:161`）は
**対応済み**であることを実測根拠とともに再確認した（PoC-v2-6 `slice_repro.onnx` 実測、相対誤差 0.000000、
`docs/spec/03-poc/poc-v2-6-interop/README.md:35`）。本リポの `crates/onnx-interop/src/ops/slice.rs` も
同パターンに対応するオペ関数として実装済みである。

## 数値一致の実測結果

`transformer.onnx` end-to-end 推論は完走し、REQ-7 判定式 `abs_err / (|ref| + 1e-6) ≤ 1e-3`
（`docs/spec/04-requirements.md:159`）による実測値を取得した。結果は「対応可否の結論」節・「数値一致の
実測結果（詳細）」表のとおり、16,384 要素中 7 要素（0.043%）が基準を超過する（`max_rel_err=0.007529871`）。

参考値（個別オペ・別モデルでの既存実測。e2e とは別スコープの数値であり、上記判定式の対象は
あくまで `transformer.onnx` 全体である点に注意）:

- safetensors 経路（MLP、PoC-v2-6）: 最大相対誤差 0.000000
- ONNX インタープリタ経路（MLP `model.onnx`）: 最大相対誤差 0.000000
- ONNX インタープリタ経路（動的境界 Slice 最小再現グラフ `slice_repro.onnx`）: 最大相対誤差 0.000000

**注記（REQ-2 との混同回避）**: 上記判定式は REQ-7 固有の指標であり、REQ-2 のバックエンド間数値一致
統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」（`docs/spec/04-requirements.md:159` 明記の
とおり）とは別指標である。両者を混同しないこと。

## 性能の実測結果（参考値）

e2e 推論（`decode → build_graph → interp::run`。CPU インタープリタ経路、release ビルド、5 回計測の中央値）:

| 項目 | 実測値 |
|------|--------|
| 実行時間（`interp::run` のみ。decode／build_graph・出力検証ループは含まない） | 244.121415ms（中央値。5 回: 242.310428ms / 242.525747ms / 244.121415ms / 244.398814ms / 245.319512ms） |
| ビルドプロファイル | `cargo test --release` |

対 PyTorch 性能下限の正式確定は TASK-8.3（#154、bench-harness）のスコープであり、本記録は参考値の
位置づけにとどまる（`docs/spec/04-requirements.md:167` REQ-8・`docs/spec/06-roadmap.md` 依存関係
`TASK-7.4 → TASK-8.3` 参照）。

## 再現手順

```sh
git fetch origin
git checkout <本イシュー #87 の実装ブランチ、または i64 算術対応マージ後の main>
ONNX_INTEROP_TRANSFORMER_ONNX=<取得した transformer.onnx のパス> \
  cargo test -p onnx-interop --test onnx_transformer_e2e -- --ignored --nocapture
```

確認済みの回帰チェック（#87 実装セッションで実施）:

```sh
cargo fmt --all -- --check                                              # pass
cargo clippy -p onnx-interop --all-targets --all-features -- -D warnings # pass
cargo test -p onnx-interop                                              # pass（177 件。e2e は #[ignore] でスキップ）
```

### 実測結果（最終）

| 項目 | 値 |
|------|-----|
| e2e テスト名 | `transformer_onnx_end_to_end_matches_pytorch_reference_within_req7_tolerance`（`crates/onnx-interop/tests/onnx_transformer_e2e.rs`） |
| `transformer.onnx` の parse・グラフ構築 | pass（`node=165`・`initializer=12`） |
| 未対応オペ・未対応型検出（実行時） | なし（オペ被覆 20/20。`i64` 算術対応後は型不一致による停止なし） |
| 最大相対誤差（REQ-7 判定式） | `0.007529871`（閾値 `1e-3` を超過。超過要素 7/16384） |
| 実行時間（CPU インタープリタ経路、release、5 回中央値） | `244.121415ms` |
| 対応可否の最終結論 | 条件付き非対応（REQ-7 数値一致基準を超過。判定式・許容誤差は変更せず実測事実を記録） |
| commit SHA（本イシュー実装の基点） | `021e8c265dcdc6388f30f2666fe64058d9b933ec`（origin/main） |
| 実施日 | 2026-08-07 |

複合判定・数値一致が実測で外れた場合は許容誤差を緩和せず、本節に実測値・エラー内容を記録したうえで
制約事項として扱う（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず人間の承認を
経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。

## #89（TASK-7.5 移行チェックリスト）への引き継ぎ事項

- `transformer.onnx` の e2e 推論は完走するが、REQ-7 判定式では 16,384 要素中 7 要素が基準を超過する
  （すべて参照値が近ゼロの要素。絶対誤差は最大 `2.7418137e-6` と極小）。#89 の移行チェックリスト作成時は
  この事実を踏まえ、REQ-7 側の受け入れ基準チェックボックスへ「対応済み」と単純に記載しないこと
- 超過要因の切り分け（判定式の近ゼロ値での構造的性質か、実装上の精度改善余地か）は未実施。切り分けの
  実施・対応方針（判定式の見直しは spec リポ側の検討事項、実装側の精度改善は別イシューでの対応
  等）はユーザー承認のうえで決定する必要がある
- `i64` 算術対応の実装は完了済み（`crates/onnx-interop/src/ops/arith.rs::{add_i64,mul_i64,div_i64,
  mod_i64}`・`crates/onnx-interop/src/onnx/interp.rs::{compute_add,compute_mul,compute_div,
  compute_mod}` が `(I64, I64)` を受け付ける。`f32`/`i64` 混在は引き続き拒否）。オペ被覆・型対応の
  観点では #89 のブロッカーは解消済み

## 関連イシューとの役割分担（二重管理を避ける）

- **#87**（TASK-7.4a・e2e 推論実測テストの実装）: 実測コード・フィクスチャ・数値比較の実装、および
  インタープリタの `i64` 算術対応（残作業）を担う。本ファイル（#88・TASK-7.4b）はその実測結果の記録・
  対応可否の結論確定に専念し、テストコード自体は実装しない。#87 で判明した実測事実は本ファイルへ転記済み
- **#78**（TASK-7.2b・インタープリタ本体）・**#84**（TASK-7.3c・Attention 系）・**#85**（TASK-7.3d・
  `LayerNormalization`）: いずれも実装完了（origin/main マージ済み）。オペ被覆・ディスパッチの
  観点では #87 のブロッカーではなくなった
- **#89**（TASK-7.5・移行チェックリスト）: 本ファイルの実測結果を踏まえた PyTorch 移行チェックリストの作成。
  本ファイルは実測記録に留め、チェックリスト形式の成果物は作成しない
- **#154**（TASK-8.3・bench-harness 対 PyTorch 性能下限確定）: 本ファイルの性能実測は参考値に留め、
  正式な下限確定は #154 のスコープ

## 未実施・後続作業

- REQ-7 数値一致基準の超過（近ゼロ参照値での相対誤差拡大、7/16384 要素）の切り分け・対応方針決定
  （ユーザー承認が必要なため、本セッションでは判定式・許容誤差を変更しない。#87 の実装報告に
  out-of-scope 項目として記録する）
- REQ-7 側のステータス更新（受け入れ基準チェックボックスの反映等）が必要な場合、`docs/spec/`
  は本リポで編集しないため spec リポ（`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーに
  提案するに留める
