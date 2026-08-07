# `transformer.onnx` end-to-end 実測・対応可否確定 記録（#88・TASK-7.4b）

イシュー #88「docs(interop): TASK-7.4b 対応可否確定・実測結果の記録」の記録文書。
親タスク TASK-7.4（`docs/spec/05-tasks.md:255`、REQ-7 受け入れ基準
`docs/spec/04-requirements.md:161`）の成果物「`transformer.onnx` end-to-end 実測結果記録」に対応する。
受け入れ条件「実測結果記録と対応可否の結論が文書化されている」を満たすことを目的とする。

## 対応可否の結論: 非対応（現時点。前提タスク未完了のため実測不能）

**現時点では `transformer.onnx` の end-to-end 推論は実行できず、対応可否を実測で確定できない。**

判定根拠（下記「前提確認の実測」節に詳細）:

1. 兄弟イシュー #87（TASK-7.4a・e2e 推論実測テスト）が未実装（open・PR なし・実装ブランチなし）であり、
   `transformer.onnx` フィクスチャ・e2e テストコードが本リポに存在しない
2. Attention 系オペ（`MatMul`／`Softmax`／`Erf`。TASK-7.3c・#84）・`LayerNormalization`（TASK-7.3d・#85）が
   いずれも未実装（open）であり、REQ-7 受け入れ基準が要求する残 14 種別のうち 4 種別が欠けている
3. `crates/onnx-interop/src/ops/mod.rs` 冒頭コメント（`crates/onnx-interop/src/ops/mod.rs:9-10`）が明記する
   とおり、ONNX グラフ実行のインタープリタディスパッチ（op 名 → オペ関数の解決）自体が TASK-7.2b・#78 の担当
   として未実装であり、現状は個別オペ関数がグラフ実行から到達不能な状態（「後続の結線待ち」）にある

この状態は `docs/perf/cuda-tensor-core-measurement.md`（#64）の先例と同型（実機・前提未整備により実測不能）
であるため、同ファイルの構成に倣い「実測手順＋記録テンプレート＋現時点で判明している事実」を固定する形式を
採用し、前提タスク（#78・#84・#85・#87）完了後に実測値を本ファイルへ転記する運用とする。

**「非対応」は恒久的な判定ではなく「現時点で実測不能」を意味する。** 一時的な機能欠如であり、REQ-7 が定義する
恒久的なスコープ外制約とは異なる（REQ-7 の対応オペ範囲・段階的優先度は `docs/spec/04-requirements.md:160` に
既定義済みで、残 14 種別の実装自体は計画済み・進行中）。

## 前提確認の実測（本セッションで実施）

対象コミット SHA: `5bdae3ee2924494e95f191f34552dc1f93500cce`（origin/main、確認日時 2026-08-07）
rustc: `rustc 1.96.0 (ac68faa20 2026-05-25)`

| 確認項目 | 結果 |
|---------|------|
| `gh issue view 87 --json state` | `OPEN`（未マージ）。関連 PR 検索（`gh pr list --search "Closes #87"`・`"87 in:title"`）も 0 件 |
| `gh issue view 84 --json state` | `OPEN`（Attention 系 `MatMul`／`Softmax`／`Erf` 未実装） |
| `gh issue view 85 --json state` | `OPEN`（`LayerNormalization` 未実装） |
| `find crates/onnx-interop -iname "*transformer*"` | 該当なし（`transformer.onnx` フィクスチャ未取り込み） |
| `find crates/onnx-interop/tests -type f` | `st_load.rs`（safetensors 経路の既存テスト）のみ。e2e 推論テストなし |
| `crates/onnx-interop/src/ops/mod.rs` インタープリタディスパッチ記述 | 「インタープリタのディスパッチ（op 名 → 本モジュール関数の解決）は #78（未実装・ブランチなし）の担当であり、本モジュールの公開関数は現時点でグラフ実行から到達不能（後続の結線待ち）」（`crates/onnx-interop/src/ops/mod.rs:9-10`） |
| `cargo fmt --all -- --check` | pass（回帰なし） |

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
| `MatMul` | **未実装** | TASK-7.3c・#84（open） | Attention 系（バッチ行列積） |
| `Softmax` | **未実装** | TASK-7.3c・#84（open） | Attention 系 |
| `Erf` | **未実装** | TASK-7.3c・#84（open） | Attention 系（GELU 等で使用） |
| `LayerNormalization` | **未実装** | TASK-7.3d・#85（open） | 正規化層系 |

被覆率: 16/20 種別（80.0%）。未実装 4 種別はいずれも Attention・正規化層系で、TASK-7.3c／TASK-7.3d の
スコープとして計画済み（未着手）。

**インタープリタ結線の欠落（オペ被覆とは別軸の未対応要因）**: 上記 16 種別のオペ関数自体は実装済みだが、
ONNX グラフを読み込みノード列を順にディスパッチする「インタープリタ本体」（TASK-7.2b・#78）が未実装のため、
現状は個別オペ関数を単体テストから呼び出すことはできても、`transformer.onnx` のようなグラフ全体を
実行する経路が存在しない。オペ被覆率のみでは対応可否を過大評価しうる点に注意（`ops/mod.rs` の
到達不能性コメントが根拠）。

## 未対応要因の有無: あり（前提タスク完了待ち）

- **インタープリタディスパッチ未実装**（#78）: グラフ実行のエントリポイントがない
- **Attention 系オペ未実装**（`MatMul`／`Softmax`／`Erf`。#84）
- **`LayerNormalization` 未実装**（#85）
- **e2e テスト・`transformer.onnx` フィクスチャ未取り込み**（#87。取得・sha256・再取得手順は
  PoC-v2-6 の先例〈`docs/spec/03-poc/poc-v2-6-interop/README.md:18`〉に倣い #87 側で整備される想定）

動的境界 Slice パターン（v1 の `burn-onnx` 失敗パターン、`docs/spec/04-requirements.md:161`）は
**対応済み**であることを実測根拠とともに再確認した（PoC-v2-6 `slice_repro.onnx` 実測、相対誤差 0.000000、
`docs/spec/03-poc/poc-v2-6-interop/README.md:35`）。本リポの `crates/onnx-interop/src/ops/slice.rs` も
同パターンに対応するオペ関数として実装済みである。したがって「未対応要因」は動的境界パターンという
構造的な問題ではなく、単純に残タスクの実装進捗（前提タスク未完了）に起因する一時的な欠落である。

## 数値一致の実測結果: 未実施

`transformer.onnx` end-to-end 推論が実行不能なため、REQ-7 判定式
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

e2e 推論が実行不能なため実行時間の実測値はない。対 PyTorch 性能下限の正式確定は TASK-8.3（#154、
bench-harness）のスコープであり、本記録は前提タスク完了後も参考値の位置づけにとどまる
（`docs/spec/04-requirements.md:167` REQ-8・`docs/spec/06-roadmap.md` 依存関係
`TASK-7.4 → TASK-8.3` 参照）。

## 再現手順（前提タスク完了後に本節のコマンドで実測し、下記テンプレートへ転記する）

```sh
git fetch origin
git checkout main   # #78・#84・#85・#87 マージ後
cargo test -p onnx-interop --release -- --nocapture   # #87 の e2e テスト名は #87 マージ後に確定
```

現時点で確認可能な回帰チェック（本セッションで実施済み）:

```sh
cargo fmt --all -- --check                                              # pass
cargo clippy --workspace --all-targets --all-features -- -D warnings    # 変更なし（docs のみ）のため実行省略。既存 CI に委ねる
cargo test --workspace --locked                                         # 変更なし（docs のみ）のため実行省略。既存 CI に委ねる
```

### 実測結果（記入待ち。#78・#84・#85・#87 完了後）

| 項目 | 値 |
|------|-----|
| e2e テスト名（#87 で確定） | （記入） |
| `transformer.onnx` の parse・グラフ構築 | （記入: pass/fail） |
| 未対応オペ検出（実行時） | （記入: なし／オペ名一覧） |
| 最大相対誤差（REQ-7 判定式） | （記入） |
| 実行時間（CPU インタープリタ経路。計測条件・回数・環境を明記） | （記入） |
| 対応可否の最終結論 | （記入: 対応／条件付き対応／非対応） |
| commit SHA | （記入） |
| 実施日 | （記入） |

複合判定・数値一致が実測で外れた場合は許容誤差を緩和せず、本節に実測値・エラー内容を記録したうえで
制約事項として扱う（`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず人間の承認を
経る」・`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。

## #89（TASK-7.5 移行チェックリスト）への引き継ぎ事項

- 本ファイルの「対応可否の結論」は前提タスク（#78・#84・#85・#87）完了後に実測値で更新される。
  #89 の移行チェックリスト作成時は本ファイルの最新版（実測後）を参照し、実測未完了のまま
  チェックリストへ「対応済み」と記載しないこと
- 未対応要因が実測で新たに見つかった場合（現時点の 4 種別以外）は、`docs/spec/06-roadmap.md:211`
  のリスク対応方針（「構造的な矛盾でなければ Phase 4 への差し戻しは不要」）に従い、制約事項として
  本ファイルに記録したうえで #89 へ引き継ぐ
- インタープリタディスパッチ（#78）の実装設計は、オペ関数の呼び出しインターフェース
  （`crates/onnx-interop/src/ops/mod.rs` の `pub use` 一覧）との整合を要する。本ファイルの
  「対応オペ一覧」節がその時点でのインターフェース確定状況を示す参照点になる

## 関連イシューとの役割分担（二重管理を避ける）

- **#87**（TASK-7.4a・e2e 推論実測テストの実装）: 実測コード・フィクスチャ・数値比較の実装そのものを担う。
  本ファイル（#88・TASK-7.4b）はその実測結果の記録・対応可否の結論確定に専念し、テストコード自体は
  実装しない
- **#78**（TASK-7.2b・インタープリタ本体）・**#84**（TASK-7.3c・Attention 系）・**#85**（TASK-7.3d・
  `LayerNormalization`）: 本ファイルが「未対応要因」として指す残実装。それぞれの担当スコープで実装される
- **#89**（TASK-7.5・移行チェックリスト）: 本ファイルの実測結果を踏まえた PyTorch 移行チェックリストの作成。
  本ファイルは実測記録に留め、チェックリスト形式の成果物は作成しない
- **#154**（TASK-8.3・bench-harness 対 PyTorch 性能下限確定）: 本ファイルの性能実測は参考値に留め、
  正式な下限確定は #154 のスコープ

## 未実施・後続作業

- 本ファイルの「実測結果」節（再現手順の下）は #78・#84・#85・#87 完了後に実測して埋める
  （新規 Issue 起票はユーザー承認が必要なため、本セッションでは行わない。#78・#84・#85・#87 は
  既存 Issue として存在するため新規起票は不要）
- REQ-7 側のステータス更新（受け入れ基準チェックボックスの反映等）が必要な場合、`docs/spec/`
  は本リポで編集しないため spec リポ（`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーに
  提案するに留める
