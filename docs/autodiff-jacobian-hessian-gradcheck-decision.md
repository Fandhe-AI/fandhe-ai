# autodiff の jacobian・hessian・gradcheck（および anomaly detection）の既存有無と設計の推奨案（イシュー #2669・親 #2668）

本記録は **推奨案の記録であり、承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、事実はソースで再確認した。基準は `origin/main` `f020e443`（2026-10-07）。以下の行番号は同 sha のもの。

## 1. 位置づけ

- ツリー: ルート #2499 → Phase 4 #2625 → #2668 → {**#2669（本記録）**, #2670（jacobian／hessian の内部実装）, #2671（gradcheck／anomaly detection の内部実装）}。公開形の承認依頼は #2677、公開は #2678。
- 本記録が確定させる推奨案: 既存有無の判定、機能ごとの契約案、parity の層構成、facade 公開形（1 案）、保留ガード名。
- **anomaly detection も本記録に含める**: 題は 3 機能だが、親 #2668 が anomaly detection を含み、#2671 が参照できる記録が他にないため。粒度は設計スケッチと公開形に留め、詳細は #2671 で確定する。
- 承認は取得していない。#2617（forward-mode／vmap の spec 改定提案）も #2677（一括承認依頼）も承認済みとは扱わない。

## 2. 着手時判定（基準 sha で確認した事実）

### 2.1 既存有無

再現コマンド（読み取り専用）:

```
grep -rnoiE "jacobian|hessian|gradcheck|gradgradcheck|grad_check|anomaly" --include='*.rs' crates
grep -rnE "fn (numeric_grad|finite_diff[a-z_]*|central_diff[a-z_]*|analytic_hessian|assert_hessian_close|assert_grad_matches_finite_difference)\b" --include='*.rs' crates
```

| 分類 | 内容 | 判定 |
|---|---|---|
| ライブラリ API（`crates/*/src` の非テスト項目） | jacobian／hessian／gradcheck／anomaly 検出の関数・型は 0 件 | **既存実装なし** |
| コメント言及のみ | `crates/autodiff/src/lib.rs:34`・`grad.rs:1`・`eval/linalg.rs:1129`（「vector-Jacobian product」の語）、`create_graph.rs:1075`・`:1402-1429`（Hessian の説明）、`crates/facade/src/lib.rs:624`（`backward_create_graph` の doc が Hessian・HVP に言及） | 実装ではない |
| テスト専用ヘルパー | `crates/autodiff/tests/create_graph.rs` の `finite_diff_hessian`(:65)・`analytic_hessian`(:103)・`assert_hessian_close`(:158)・`hvp_small_mlp_matches_finite_difference`(:1022)、`create_graph.rs` 内 `#[cfg(test)]` の `hessian_maximum／minimum_*`(:1485, :1512)、`crates/facade/tests/create_graph_facade.rs:40` の `hessian_diag_cubic`、`crates/onnx-interop/tests/onnx_autograd.rs:706` の `assert_grad_matches_finite_difference`、ほか `numeric_grad`／`central_diff_*`／`finite_diff_*` 系が上記 2 本目のコマンドで計 33 関数 | 再利用可能な API ではない（テストごとの私的関数） |
| 親が未精査とした「gradcheck」ヒット | `crates/autodiff/src/grad.rs:9631`（`#[cfg(test)]` の grad-check モジュール doc が `torch.autograd.gradcheck` を引用）、`crates/autodiff/tests/poc_v2_2_parity.rs:27`（同じく doc。`GradCheckCase`(:102)・`gen_grad_check_case`(:119) はテスト） | いずれもテストコード・コメントで、ライブラリの gradcheck ではない |
| anomaly | `crates/backend-cuda/examples/cuda_floor_bench.rs:1156` のメッセージ文字列 1 件のみ | 無関係。検出モードは存在しない |

**結論**: ライブラリとしての既存実装は無い。ただし jacobian／hessian は既存の `Tape::backward`／`Tape::backward_create_graph` の合成で作れ、新しい `Op`・`BackendOps` メソッド・VJP は不要。

### 2.2 土台として使える既存 API

- `Tape::backward(&self, loss) -> Result<Gradients, AutodiffError>`（`backward.rs:166`）: 呼び出しごとに独立した `Gradients` を返しテープを消費しない（`docs/autodiff-retain-graph-accumulate-decision.md`）。非スカラー loss は全要素 1 のシード（暗黙の総和）。`Gradients` の `grads` フィールドは非公開（`backward.rs:49`）で、ノード列挙用 accessor は無い。
- `Tape::backward_create_graph(&self, loss, child)`（`create_graph.rs:276`）と `CreateGraphResult::{first_order(:196), grad(:213), child_var(:224)}`。対象 Op は `Op::supports_create_graph()`（`tape.rs:2432`）で、対象外は入口で `Err(AutodiffError::Backward)`（fail-closed）。facade では #2545 で公開済み（`crates/facade/src/lib.rs:656`）。
- `Var::flatten`(`var.rs:3001`)・`Var::narrow`(`var.rs:3237`)、`Var::tape()` は `pub(crate)`(`var.rs:223`)、`Tape::reset`(`tape.rs:3201`)・`Tape::is_empty`(`:3114`)・`Tape::var_no_grad`(`:2939`)・`Tape::custom`(`:3003`)。
- `f64_autograd`（別テープ・facade 非公開）、`determinism`（プロセスワイド opt-in の先例）。

### 2.3 spec・既存記録からの制約

- spec REQ-9（`docs/spec/04-requirements.md:232`）は Tier 2 に「高階微分」を列挙。一方 `:235`（#2194 追記）は利用者向け汎用 forward-mode AD API・vmap・`torch.func` 相当の合成関数変換を引き続き対象外とする。
- `docs/autodiff-forward-mode-vmap-spec-proposal.md`（#2617）は関数型ラッパー（`jacrev`／`hessian` 等）を対象内へ移す spec 改定の**提案**に留まり、#2617 は承認記録なしで close されている（承認コメントは存在しない）。**本記録はこの提案に依存しない**。
- `docs/autodiff-higher-order-grad-decision.md` §17.3・§19.4 は HVP 専用 API・子テープ構築ヘルパー・`TapeRef` 版・残り Op の拡張を保留継続としている。
- grad-check のテスト閾値（`H=1e-3`・`TAU=1e-4`・`REL_TOL=1e-2`・`ABS_TOL=1e-3`。`grad.rs:9656-9659`、判定は `grad.rs:9685-9689`）は #223 で承認済みの**テスト専用**閾値で、REQ-2 統一複合判定（相対 1e-3 未満または絶対 1e-5 未満）とは別系統。f32 の中央差分は丸め誤差床が約 1e-4 のため REQ-2 判定は満たせない（`grad.rs:9631-9640`）。

### 2.4 facade 側の制約

- facade `Tape` は newtype（`crates/facade/src/lib.rs:329`）。内部の関数が `&fandhe_ai_autodiff::Tape` を取る場合、利用者は facade の委譲メソッド経由でしか渡せない。
- コールバックへテープ文脈を渡す橋渡しは `TapeRef<'t>`（`lib.rs:888`。`var`／`var_from`／`var_no_grad` のみ・`unsafe` なし）が確立済み。
- `Var` は直接再エクスポートのため、inherent メソッドを足すと即座に公開面が広がる。
- モジュール再エクスポートは `facade_pub_use_leaves_are_not_modules`（`crates/facade/tests/api_surface.rs:7448`）が拒否する。
- `v0.10.0` タグの `crates/facade/src` では `jacobian`／`gradcheck`／`anomaly` の出現が 0 件、`hessian` は L-BFGS の説明コメント 2 件のみ（`git grep -nEi "jacobian|hessian|gradcheck|anomaly" v0.10.0 -- crates/facade/src`）。よって推奨形は追加のみで、`fandhe-ai =0.10.0` の公開 API は非破壊。

## 3. 推奨案

推奨は計画時点の設計判断。公開面を狭く・fail-closed に寄せる方針で書く。

### 3.1 スコープ境界（spec との関係）

- jacobian／hessian は、reverse-mode テープの**繰り返し実行で値を返すユーティリティ**（`torch.autograd.functional.jacobian`／`hessian` の reverse-mode・`vectorize=False` 相当）と位置づけ、Tier 2「高階微分」の範囲で扱う。
- 対象外のまま: forward-mode（`jacfwd`）、`vectorize=True`（vmap）、関数を受け取り関数を返す合成関数変換（`torch.func`）、結果自体を微分可能にする `create_graph=True` 版。
- この位置づけの採否自体をユーザー決定事項とする（§4 (a)）。

### 3.2 jacobian（推奨: 記録済みグラフを受け取る形）

- 契約案: `jacobian(tape: &Tape, output: &Var<'_>, input: &Var<'_>) -> Result<Tensor<f32>, AutodiffError>`。戻り値の shape は `output.shape ++ input.shape`、非微分のホスト値。
- 方式: `output` を 1 次元へ平坦化し、要素 i を `narrow` で取り出して `Tape::backward` を要素数ぶん繰り返す。`backward.rs`／`grad.rs` は無変更。`output` がスカラーなら平坦化を省く。
- 不採用案: クロージャを受け取る形（テープ・ライフタイムの整合が複雑で合成関数変換との境界が曖昧）／シード付き backward の新設（`backward_impl` の変更を伴う）。
- 契約: テープへ補助ノード（平坦化 1＋取り出し m）が増える／計算量は backward m 回／`input` が非追跡なら `GradientTrackingDisabled`／別テープは `TapeMismatch`／`output` が `input` に依存しない行（`Gradients::get` が `Ok(None)`）は全ゼロ／結果要素数 `m×n` は検査付き乗算で算出しオーバーフロー時は型付きエラー／単一入力のみ／resident・fused 経路は既存 `backward` に従う。
- CUDA／Metal: 新しい `BackendOps` メソッドを足さないため `Unsupported` フォールバックの追加対象がない（既存 Op の合成で到達）。先例は `docs/autodiff-low-precision-op-extension-decision.md` §6.1。

### 3.3 hessian（推奨: 子テープを受け取る形）

- 契約案: `hessian(tape: &Tape, loss: &Var<'_>, input: &Var<'_>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`。shape は `input.shape ++ input.shape`。
- 方式: `backward_create_graph` → `CreateGraphResult::grad(input)` で子テープ上の 1 階勾配 → 平坦化・要素取り出し → `child.backward` を要素数ぶん → `child_var(input)` の勾配を集める。`ops` の供給は higher-order doc §7 の既存決定（呼び出し側が空の子テープを渡す）を踏襲し、子テープ構築ヘルパーは作らない。
- 契約: 対象は `supports_create_graph()` の範囲のみで、それ以外は既存の `Err(AutodiffError::Backward)` を伝播（新しいエラー variant は足さない）／`child` は空・同一デバイス（既存検査）／**`loss` は要素数 1 に限り非スカラーは型付きエラーで拒否**（後から緩めるのは非破壊だが逆は破壊的なため厳しい側を初期値にする）／`grad(input)` が `None`、または子テープの backward で勾配が届かない行は全ゼロ／呼び出し後の `child` には記録が残る（再利用前に呼び出し側が `reset`）。
- **HVP 専用 API は含めない**（higher-order doc §17.3 の保留項目。公開済みの `backward_create_graph` で合成できる。本記録はその保留を消費しない）。

### 3.4 gradcheck（推奨: クロージャ＋専用テープ、閾値は必須引数）

- 摂動点での再評価が要るためクロージャ形が必須。内部の契約案: `gradcheck(make_tape: M, f: F, inputs: &[Tensor<f32>], options: &GradcheckOptions) -> Result<GradcheckReport, AutodiffError>`、`M: Fn() -> Tape`、`F: for<'a> Fn(&'a Tape, &[Var<'a>]) -> Result<Var<'a>, AutodiffError>`。クロージャへテープを渡す（渡さないと `var_no_grad` で定数を作れない）。facade 側は `TapeRef<'a>` に包んで渡す薄いアダプタとする（`unsafe` なし。1 行委譲ではない点に注意）。
- **テープは評価ごとに新規作成する**。既存の `Tape::reset` は葉プレフィックス（`retained_leaf_len`。`tape.rs:3201-3218`）を保持して非葉ノードのみ切り詰めるため、評価ごとに `reset` しても入力葉は残り、摂動点ごとに入力を登録し直すと葉が蓄積して空テープには戻らない。このため `reset` 再利用方式は採らず、`make_tape` を呼んで解析用 1 回・摂動用（要素ごとの `+eps`／`-eps`）の評価ごとに新しい `Tape` を作る（ファクトリ引数にするのは、`Tape::new()`／`Tape::new_with_ops` のどのバックエンドで作るかを呼び出し側が決めるため）。入力ごとに摂動値は新テープへ `var` で登録する。評価後のテープは drop され、利用者のテープを変更しない。計算コストは評価回数に比例して増えるが、gradcheck はテスト・検証用途であり許容する。保持葉を再利用する案（`reset` 後に葉へ値を差し替える API の新設）は `Tape` の変更を伴うため不採用。
- 数値方式: f32 の forward 値から中央差分を f64 で集計（既存テストヘルパーと同じ）。解析側は 3.2 と同じ行取り出し＋`backward`。
- `GradcheckOptions`（`eps`・`atol`・`rtol`・`tau`）は**既定値を持たない必須引数**（`Default` を実装しない。有限・正値の検証で fail-closed）。既定値は新しい閾値定数になるため置かない。
- **判定式**: 要素ごとに `abs = |a − n|`、`rel = abs / max(|a|, |n|, tau)` とし、`rel <= rtol` **または** `abs <= atol` で合格。どちらかが非有限なら不合格。#223 承認済みの grad-check テスト判定（`grad.rs:9685-9689`）と同じ式で、#2671 のテストは承認済みの組（`eps=1e-3`・`tau=1e-4`・`rtol=1e-2`・`atol=1e-3`）をそのまま渡せる。PyTorch の `allclose` 形（`atol + rtol·|n|`）は緩くなるため採らず、意図的な差として記録する。
- `GradcheckReport` は合否に加え最大絶対誤差・最大相対誤差・最悪要素の位置を返す。不一致は `Err` ではなく `Ok(report)` の不合格で返す。
- **3 種の閾値を混同しない**: (a) REQ-2 統一複合判定（不変・バックエンド間／PyTorch 値の突合用）、(b) #223 承認済みの grad-check テスト閾値（不変・テスト専用）、(c) gradcheck の利用者引数（アルゴリズム引数であり REQ-2 tolerance ではない。先例: `docs/autodiff-linalg-ops-decision.md` の `rcond`）。#2671 のテストが渡す値は (b) の組だけとし、他の値が要るならユーザー承認を得る。
- 適用外: 低精度 forward（丸めが区分定数）、キンク・タイ近傍（利用者責任として doc に注意書き）。`VarF64` 版は対象外。

### 3.5 anomaly detection（設計スケッチ。推奨: 明示呼び出しの検出付き backward）

- 推奨: `backward_detect_anomaly(tape: &Tape, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`。通常の `Tape::backward` の結果と記録済み forward 値を事後走査し、最初に非有限値（NaN／±inf）を生んだノードを特定して `Err(AutodiffError::Backward(..))` にノード番号と Op 種別を載せて返す。非有限が無ければ通常の `Gradients` をそのまま返す。
- **必要な内部変更**: `Gradients.grads` は非公開のため、#2671 は `Gradients` に読み取り専用の `pub(crate)` accessor（既存の `synthetic`／`resident_fingerprint` と同種）を 1 つ足す必要がある。`backward_impl` の処理を変えない追加であり、1 階経路 bit 不変の契約には抵触しない（`backward.rs` を 1 行も触らない、という意味ではない）。
- 検出範囲の限界: backward 後に forward 値が実体化済みのノードと、勾配が到達したノードに限る。未実体化・未到達のノードは範囲外で、検出のための新たな実体化はしない（融合の実行経路を変えない）。
- 不採用案: プロセスワイドのモード（`determinism` 同型。PyTorch `set_detect_anomaly` に最も近いが `backward_impl` に分岐が入り、並列テストが互いの backward を失敗させうる）／テープ単位のモード（`Tape` 構造体と `backward_impl` を変更する）。
- 契約: 検出なしの経路（既存 `backward`）は 1 ビットも変えない／検出は読み取りのみ／新しいエラー variant を足さない／GPU ではホスト読み出しが発生する（デバッグ用途の opt-in）／エラーメッセージにテンソルの中身を載せない（ノード番号・Op 種別・形状のみ）／`compat::Sequential::fit` への結線は対象外。

### 3.6 parity の事前登録

| 層 | 対象 | 判定 | 実行 |
|---|---|---|---|
| J1 | jacobian（CPU）: PyTorch 2.14.0 `torch.autograd.functional.jacobian`（f32・reverse-mode）の fixture | REQ-2 統一複合判定（定数不変） | CI |
| J2 | hessian（CPU）: 同 `hessian` の fixture。create_graph 対象 Op のみ | 同上 | CI |
| J3 | 閉形式・既存テストヘルパー（`finite_diff_hessian` 相当）との突合 | 既存テストと同じ基準（新しい閾値を作らない） | CI |
| J4 | fail-closed: 非対象 Op・非スカラー loss・非空の子テープ・別テープ・非追跡入力・要素数オーバーフロー | 型付きエラー、テープ／子テープ無変更 | CI |
| J5 | CUDA／Metal vs CPU | 統一複合判定。`#[ignore]` | 実機（未実測なら `docs/perf/logs/jacobian-hessian-2670/README.md` へ申し送り） |
| G1 | gradcheck の解析ヤコビアン: PyTorch f32 autograd の fixture | 統一複合判定 | CI |
| G2 | gradcheck の合否: 正しい VJP は合格、`CustomFunction` で意図的に誤らせた VJP は不合格（差が閾値より十分大きい設計ケースのみ） | 合否の一致 | CI |
| G3 | gradcheck の数値ヤコビアン（f32 中央差分）と PyTorch f64 値の差 | **非ゲートの観測値**（有限差分は REQ-2 の対象外） | CI（記録のみ） |
| A1 | anomaly: 非有限を生む Op の特定・非有限なしで `backward` と bit 一致・検出なし経路の不変 | 挙動テスト。PyTorch の値 fixture は存在しない | CI |

- 「判定不能」は第三者比較（PyTorch fixture）に限り、自前バックエンド間（J5）の不一致は通常の失敗として扱う。fixture は既存 `crates/autodiff/tests/fixtures/*-pytorch-reference/` と同じ運用（生成スクリプト・README に生成条件と sha256・CI は Python 非依存）。実 `CpuBackendOps` での検証が要る場合は `crates/facade/tests/` に置く（`autodiff` は `backend-cpu` に依存しない）。

### 3.7 facade 公開形（推奨を 1 つに決める）

| 案 | 形 | 評価 |
|---|---|---|
| **A（推奨）** | facade `Tape` の薄い委譲メソッド（`Tape::jacobian`／`hessian`／`gradcheck`／`backward_detect_anomaly`）＋必要な型（`GradcheckOptions`／`GradcheckReport`）の crate ルート再エクスポート | `Tape::backward`／`backward_create_graph`（#2545）と同系統。`hessian`・`gradcheck` は `Tape` を引数に取るため newtype 経由の委譲が必須。全 4 機能を同形に揃えられる。`gradcheck` のみクロージャ引数を `TapeRef` に包むアダプタ |
| B | `Var` の inherent 委譲メソッド | 子テープ・専用テープを facade の `Tape` で渡せない。演算ではなく backward 系の操作 |
| C | モジュール再エクスポート | `facade_pub_use_leaves_are_not_modules` が拒否 |
| D | crate ルートの自由関数 | `Tape` を newtype で受ける変換が要り、既存の backward 系と配置が割れる |
| E | `Sequential::add_*` | 層ではない |

- 追加のみで、既存項目のシグネチャ・意味論・`FitConfig`・`AutodiffError` の variant は不変（§2.4 の `v0.10.0` 突合が非破壊の根拠）。
- **承認までは公開しない**。#2677 用の一覧行:

| 機能 | 公開形 | 非破壊性 | 保留ガード名 |
|---|---|---|---|
| jacobian | 案 A: `Tape::jacobian` | 追加のみ | `JacobianHessianHoldDoctestGuard` |
| hessian | 案 A: `Tape::hessian` | 追加のみ | `JacobianHessianHoldDoctestGuard` |
| gradcheck | 案 A: `Tape::gradcheck`＋`GradcheckOptions`／`GradcheckReport` | 追加のみ | `GradcheckAnomalyHoldDoctestGuard` |
| anomaly detection | 案 A: `Tape::backward_detect_anomaly` | 追加のみ | `GradcheckAnomalyHoldDoctestGuard` |

### 3.8 保留ガード（#2670／#2671 への指定）

- #2670 と #2671 は依存宣言のない兄弟で並列に走りうるため、ガードは issue ごとに分ける: `JacobianHessianHoldDoctestGuard`（#2670）・`GradcheckAnomalyHoldDoctestGuard`（#2671）。いずれも「正のプローブ 1 ブロック」方式＋`api_surface.rs` の 5 テスト構成（低精度 doc §6.3 と同型: glob 固定・プローブ本体固定・再エクスポート／宣言の否定・その検出力の自己テスト・fn 名インベントリの許可場所）。
- 内部の置き場所の推奨: `crates/autodiff/src/jacobian_ops.rs`（#2670）、`crates/autodiff/src/gradcheck.rs`・`crates/autodiff/src/anomaly.rs`（#2671）。新規 `pub mod` の自由関数とし、`Var`／`Tape` に `pub` の inherent メソッドを足さない。
- gradcheck の解析側は jacobian と同じ行取り出しを使うため、**#2670 → #2671 の順で直列が望ましい**（並列になった場合は後着が rebase）。

### 3.9 後続への実装スケッチ（本記録では実施しない）

- #2670: `jacobian_ops` に jacobian／hessian、J1〜J4 と fixture、保留ガード。
- #2671: `gradcheck`／`anomaly`、`Gradients` への `pub(crate)` accessor、G1〜G3・A1、保留ガード。
- #2677: §3.7 の表を一括承認依頼へ載せる。#2678: 承認後に facade へ公開。

## 4. ユーザーに決めてほしい事項

(a) jacobian／hessian を Tier 2「高階微分」範囲のユーティリティと位置づけること（3.1）／(b) jacobian のグラフ受け取り形・単一入力／(c) hessian の子テープ受け取り形と非スカラー loss の拒否／(d) gradcheck の閾値を必須引数にし既定値を置かないこと／(e) gradcheck の判定式（#223 と同じ式。PyTorch `allclose` 形を採らない）／(f) #2671 のテストが #223 承認済み閾値の組のみを使うこと／(g) anomaly detection を明示呼び出し形にしグローバルモードを置かないこと、`Gradients` への `pub(crate)` accessor 追加／(h) 公開形 A とメソッド名・再エクスポートする型名／(i) parity の層構成（3.6）／(j) 実機未実測のまま内部実装を進めてよいか／(k) HVP 専用 API・複数入力・微分可能な jacobian・`fit` への anomaly 結線の後続起票の要否。

## 5. スコープ外・申し送り

- 実装とガード（#2670／#2671）、承認依頼（#2677）、公開（#2678）、実機 parity。
- forward-mode・vmap・`torch.func`（#2617 の提案は未承認）、HVP 専用 API、`VarF64` 版、create_graph 対象 Op の拡張、テストに複製された `numeric_grad` 系ヘルパーの統合（事実の記録のみ）。
- higher-order doc §8 の variant 表が古い件（別件・既知）。

## 6. セキュリティ観点

- 設計は fail-closed に固定する（hessian の対象外 Op・非スカラー loss・非空の子テープは型付きエラー／要素数は検査付き乗算／gradcheck に暗黙の既定閾値を置かない／anomaly 検出は値を書き換えない）。
- 新規依存・新規 `unsafe`・tolerance 変更を前提にしない。承認を代行しない。anomaly のエラーメッセージにテンソルの中身を出さない。

## 7. 出典

- `docs/autodiff-higher-order-grad-decision.md` §7・§17.3・§19.4、`docs/autodiff-low-precision-op-extension-decision.md`（章立ての先例）、`docs/autodiff-forward-mode-vmap-spec-proposal.md`、`docs/autodiff-retain-graph-accumulate-decision.md`、`docs/autodiff-linalg-ops-decision.md`
- `docs/spec/04-requirements.md` REQ-9（:232・:235）・REQ-2
- 本文中の `file:line` が参照するソース（基準 sha `f020e443`）
