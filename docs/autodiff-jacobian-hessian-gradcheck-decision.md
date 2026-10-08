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

## 8. 実装記録（#2670）

本節は §3.2・§3.3・§3.6・§3.8 の推奨案のうち **jacobian／hessian の内部実装と保留ガード**を実装した記録である（イシュー #2670・親 #2668）。**facade 公開形は未承認のままで、承認依頼は #2677・公開は承認後の #2678**（本記録は承認記録ではない）。gradcheck・anomaly detection は #2671。

### 8.1 実装

| 項目 | 内容 |
|---|---|
| 置き場所 | `crates/autodiff/src/jacobian_ops.rs`（`pub mod jacobian_ops`）。自由関数のみで、`Var`・`Tape` へ inherent メソッドを足していない（`Var` は facade から再エクスポートされ公開面が広がるため） |
| シグネチャ | `jacobian(tape: &Tape, output: &Var<'_>, input: &Var<'_>) -> Result<Tensor<f32>, AutodiffError>`／`hessian(tape: &Tape, loss: &Var<'_>, input: &Var<'_>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`（§3.2・§3.3 どおり） |
| 変更しないもの | `backward.rs`・`grad.rs`・`create_graph.rs`・`AutodiffError`（variant 追加なし）・`Cargo.toml`／`Cargo.lock`・tolerance・baseline・`docs/spec/`。新規 `Op`・`BackendOps` メソッド・VJP・`unsafe` もなし |
| jacobian の入口検査（テープへノードを足す前。順序固定） | ①`output`／`input` が `tape` の現世代に属さない → `TapeMismatch`（`Tape::reset` をまたいだ世代違いを含む）②`input` 非追跡 → `GradientTrackingDisabled`③`m`・`n`・`m×n` を検査付き乗算で算出（`ShapeError::ElementCountOverflow`）④`output` が追跡なし、または要素数 0 → backward を呼ばず全ゼロ（要素数 0 は空テンソル） |
| hessian の入口検査（`backward_create_graph` の前。失敗時 `child` 無変更） | ①テープ不一致 → `TapeMismatch`②`input` 非追跡 → `GradientTrackingDisabled`③`loss` の要素数が 1 でない → `InvalidArgument`（`[]`・`[1]`・`[1,1]` は可）④`n×n` の検査付き乗算（`input` 要素数 0 は空テンソルを返し `child` は検査しない）。以降は `backward_create_graph` の既存検査（対象外 Op・非空の子テープ・デバイス不一致・checkpoint 済み親・追跡なし loss）を伝播 |
| ゼロ行の扱い | jacobian: `Gradients::get` が `Ok(None)`（`input` へ届かない行）。hessian: `grad(input)`／`child_var(input)` が `None`、1 階勾配が定数（`requires_grad == false`。例: 入力に線形な loss）、子テープの backward で届かない行 |
| 行の取り出し | jacobian・hessian 共用の非公開ヘルパー（#2671 の gradcheck も共用予定）。rank 0 は自身を 1 要素として使い、それ以外は `Var::contiguous`（非 contiguous のときのみノードを積む）→ `reshape([numel])` → 要素ごとに `narrow(0, i, 1)`。**補助ノード数は「`contiguous` 0〜1＋`reshape` 1＋`narrow` m」**で、§3.2 の「平坦化 1＋取り出し m」とは `contiguous` の分だけ異なる。1 要素の行は shape `[1]` のまま `backward` へ渡す（シードは全要素 1 の暗黙の総和なので `sum` ノードは足さない） |
| 計算量 | backward を jacobian は出力要素数 `m` 回、hessian は入力要素数 `n` 回。結果は `m×n`（`n×n`）個の `f32` を確保する。大きな形状は呼び出し側の責任で避ける（モジュール doc に明記） |
| hessian 後の `child` | 記録が残る（再利用前に呼び出し側が作り直す）。子テープ上の数値方式は 1 階 VJP と bit 同一を主張しない（`create_graph` の既存契約） |

### 8.2 検証結果

| 層 | 内容 | 結果 |
|---|---|---|
| J1 | PyTorch 2.14.0 実行値 fixture（`tests/fixtures/jacobian-hessian-pytorch-reference/`・生成条件と sha256 は同 `README.md`）の jacobian 9 件と forward 値を REQ-2 統一複合判定（`common::req2_close`）で全要素突合 | 全件 pass |
| J2 | 同 hessian 9 件（二次形式・入力に線形・`tanh`・`exp`・小さな MLP・rank 0・`[1,1]` loss・`cat`＋`transpose`・`relu`）を同判定で突合 | 全件 pass |
| J3 | 二次形式の閉形式 `A + Aᵀ`・`exp(x)·x` の平均の閉形式・スカラー出力の jacobian と `Tape::backward` の勾配の bit 一致・全 hessian ケースの対称性 | pass |
| J4 | 非追跡入力・別テープ・非スカラー loss・非空の子テープ・非対象 Op（`Max`・rank 3 `matmul`）・追跡なし出力・入力へ届かない行・要素数 0・オーバーフロー（純関数の単体テスト）・呼び出し前後で既存ノード値と `backward` 結果が bit 不変 | pass。拒否時に `child` が空（非空の子テープ入力は 1 ノードのまま） |
| J5 | `crates/facade/tests/jacobian_hessian_backend_parity.rs`: 実 `CpuBackendOps` tape と `NaiveOps` tape を `assert_parity` で突合＋手計算 1 件 | pass。CUDA／Metal 版は `#[ignore]`・**実機未実測**（`docs/perf/logs/jacobian-hessian-2670/README.md`） |

保留ガードの検出力は、facade の `Tape` へ仮に `pub fn jacobian` を足して doctest・`facade_does_not_reexport_or_declare_jacobian_hessian`・インベントリが落ちることを一時変更で確認し、戻した（コミットしていない）。

### 8.3 保留ガード（§3.8）

- `crates/facade/src/lib.rs::JacobianHessianHoldDoctestGuard`（`#[cfg(doctest)]`。全 `pub mod` glob import のスコープへ、ローカルモジュール `jacobian_ops`・裸の自由関数 `jacobian`／`hessian`・`Var`／`Tape`／`Tensor<f32>` 向けの同名メソッドのプローブを置く正のプローブ 1 ブロック方式）。検出範囲は列挙した名前に限る。
- `crates/facade/tests/api_surface.rs` の 5 テスト: `jacobian_hessian_hold_doctest_globs_all_pub_modules`・`jacobian_hessian_hold_doctest_probe_body_matches_fixed_contract`（固定文言 `JACOBIAN_HESSIAN_HOLD_PROBE_BODY`）・`facade_does_not_reexport_or_declare_jacobian_hessian`・同 `_detects_each_category`・`workspace_declares_jacobian_hessian_fn_names_only_in_allowed_locations`（期待値は `autodiff/src/jacobian_ops.rs::jacobian`／`::hessian` 各 1 件）。
- 承認後（#2678）は doctest ガードを削除し、否定ガードを正ガードへ反転する。

### 8.4 申し送り

- facade 公開（`Tape::jacobian`／`Tape::hessian`）: #2677 の承認後に #2678。
- CUDA（GB10）・Metal（M4 Max）実機 parity: `docs/perf/logs/jacobian-hessian-2670/README.md`（未実測）。
- HVP 専用 API・複数入力・微分可能な jacobian・forward-mode／vmap・`VarF64` 版・create_graph 対象 Op の拡張は本イシューの対象外。


## 9. 実装記録（#2671）

本節は §3.4・§3.5・§3.6・§3.8 の推奨案のうち **gradcheck と anomaly detection の内部実装と保留ガード**を実装した記録である（イシュー #2671・親 #2668）。**facade 公開形は未承認のままで、承認依頼は #2677・公開は承認後の #2678**（本記録は承認記録ではない）。

### 9.1 実装

| 項目 | 内容 |
|---|---|
| 置き場所 | `crates/autodiff/src/gradcheck.rs`（`pub mod gradcheck`）・`crates/autodiff/src/anomaly.rs`（`pub mod anomaly`）。自由関数と内部型のみで、`Var`・`Tape` へ inherent メソッドを足していない。クレートルートへの型の再エクスポートもない |
| gradcheck | `gradcheck(make_tape: Fn() -> Tape, f: for<'a> Fn(&'a Tape, &[Var<'a>]) -> Result<Var<'a>, AutodiffError>, inputs: &[Tensor<f32>], options: &GradcheckOptions) -> Result<GradcheckReport, AutodiffError>`。解析側は既存 `jacobian_ops::jacobian` を入力ごとに呼び、数値側は評価ごとに `make_tape()` で新しいテープを作って中心差分（forward 値を `f64` へ昇格）。評価回数は `1 + 2·Σn_k` |
| `GradcheckOptions` | `new(eps, atol, rtol, tau)` が有限かつ正を検査（違反は `InvalidArgument`）。`Default` なし・フィールド非公開。判定式は #223 承認済みの grad-check テスト判定と同式（`rel = abs / max(|a|, |n|, tau)`、`rel <= rtol` または `abs <= atol`、非有限は不合格） |
| `GradcheckReport` | 合否・最大絶対誤差・最大相対誤差・最悪要素の位置（入力番号・出力の平坦添字・入力の平坦添字。絶対誤差最大の要素）・検査要素数。不一致は `Err` ではなく `Ok(不合格)` |
| gradcheck の入口検査 | 空の `inputs`・出力要素数 0・摂動点での出力 shape 変化・`eps` が入力値に対して小さすぎ f32 で摂動が潰れる場合は `InvalidArgument`。要素数は検査付き乗算。出力が渡したテープに属さなければ `TapeMismatch`。クロージャ・`make_tape` の `Err` は伝播 |
| anomaly | `backward_detect_anomaly(tape: &Tape, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`。`Tape::backward` を 1 回呼んだうえで、実体化済みのノード値を node id 昇順（forward 段階）、勾配を node id 降順（gradient 段階）に読み取り専用で走査し、最初の NaN／±inf を `AutodiffError::Backward` で報告。非有限が無ければ `Tape::backward` の `Gradients` をそのまま返す（bit 一致）。グローバル／テープ単位のモードは置かない |
| メッセージ | 段階・node id・Op 種別名・shape のみ（gradient 段階は消費側ノード候補を最大 4 件併記し、断定表現にしない）。Op 種別名は `Op` の `Debug` 出力のうち識別子文字の先頭部分だけを取る非公開関数で得る。`CrossEntropyLoss` の targets・スカラー演算の定数・`Custom` の利用者定義名・テンソル値は載らない |
| 変更したもの | `backward.rs` へ `Gradients` の読み取り専用 `pub(crate)` accessor（`grad_slots`）1 つ、`lib.rs` へ `pub mod` 2 件。`backward_impl`・`Op`・VJP・`AutodiffError`・`Cargo.toml`／`Cargo.lock`・tolerance・baseline・`docs/spec/` は不変。新規 `unsafe` なし |

§3.4・§3.5 の推奨案との差分:

- **行取り出しの共用は行わず `jacobian` の呼び出しで代替**: 非公開の行取り出しヘルパーは共有せず、公開済みの `jacobian_ops::jacobian` を入力ごとに呼ぶ。`jacobian_ops.rs` は無変更。入力数ぶん backward が増えるが、検証用途の使い捨てテープなので許容。
- **数値微分の分母は `2·eps` ではなく実際の摂動幅**: `(x + eps) as f32` と `(x − eps) as f32` の差（f32 丸め後）で割る。丸めで摂動が潰れる場合（差が 0 以下）は `InvalidArgument`。
- **anomaly の「最初」の規則**: forward 段階を先に見て、見つかれば gradient 段階は走査しない（原因に近い側を先に報告）。forward は実体化済みノードのみで、未実体化の遅延ノードや `ResidentLeaf` は新たに実体化しない。

### 9.2 検証結果

| 層 | 内容 | 結果 |
|---|---|---|
| G1 | PyTorch 2.14.0 実行値 fixture（`tests/fixtures/gradcheck-anomaly-pytorch-reference/`・生成条件と sha256 は同 `README.md`）の 7 プログラム（要素ごとの合成・matmul＋tanh・縮約・小さな MLP・rank 0 入力・複数入力・キンクを避けた relu）について、forward 値と解析ヤコビアンを REQ-2 統一複合判定（`common::req2_close`）で全要素突合。あわせて同じプログラムが `gradcheck`（`eps=1e-3`・`atol=1e-3`・`rtol=1e-2`・`tau=1e-4`。#223 承認済みの組のみ）に合格し、検査要素数が出力要素数 × 入力要素数の総和と一致 | 全件 pass |
| G2 | 正しい VJP は合格。`Tape::custom` で backward を `2·upstream` に誤らせた恒等関数は不合格で、最大絶対誤差 1・最悪位置は対角 | pass |
| G3 | 数値ヤコビアンと PyTorch f64 の差を**非ゲートで記録**（assert なし）。gradcheck の最大絶対誤差は 6.5e-6〜1.7e-4、最大相対誤差は 6.5e-6〜8.0e-4 | 記録のみ（閾値内） |
| A1 | forward の NaN（`log` に負値）・+inf（`exp` のオーバーフロー）を node id・Op 名・shape で検出／forward は有限で勾配が非有限（`sqrt` の 0）を gradient 段階で検出／異常なしでは `Tape::backward` と全入力の勾配が bit 一致／呼び出し前後でノード数と値が不変／追跡なし loss・別テープは既存エラーがそのまま返る／メッセージに `Custom` の利用者定義名が載らない（`CrossEntropyLoss` の targets は単体テストで `Debug` 全体には含まれるが種別名には出ないことを確認） | 全件 pass |
| fail-closed | 非有限・非正のオプション・空の `inputs`・出力要素数 0・別テープの出力・クロージャの `Err` 伝播・摂動点での shape 変化・f32 で潰れる `eps`・`make_tape` の呼び出し回数が `1 + 2·Σn`・利用者の既存テープが無変更 | 全件 pass |
| backend | `crates/facade/tests/gradcheck_anomaly_backend_parity.rs`: 実 `CpuBackendOps` tape と `NaiveOps` tape で合否・検査要素数・解析ヤコビアン（`assert_parity`）・anomaly のメッセージが一致＋手計算 1 件 | pass。CUDA／Metal 版は `#[ignore]`・**実機未実測**（`docs/perf/logs/gradcheck-anomaly-2671/README.md`） |

保留ガードの検出力は、facade の `Tape` へ仮に `pub fn gradcheck` を足して doctest・`facade_does_not_reexport_or_declare_gradcheck_anomaly`・インベントリが落ちることを一時変更で確認し、戻した（コミットしていない）。

### 9.3 保留ガード（§3.8）

- `crates/facade/src/lib.rs::GradcheckAnomalyHoldDoctestGuard`（`#[cfg(doctest)]`。全 `pub mod` glob import のスコープへ、ローカルモジュール `gradcheck`・`anomaly`・裸の自由関数 `gradcheck`／`backward_detect_anomaly`・単位構造体 `GradcheckOptions`／`GradcheckReport`・`Var`／`Tape`／`Tensor<f32>` 向けの同名メソッドのプローブを置く正のプローブ 1 ブロック方式。モジュール `gradcheck` と関数 `gradcheck` は名前空間が別のため同一プローブ内で共存でき、両方を検査している）。検出範囲は列挙した名前に限る。
- `crates/facade/tests/api_surface.rs` の 5 テスト: `gradcheck_anomaly_hold_doctest_globs_all_pub_modules`・`gradcheck_anomaly_hold_doctest_probe_body_matches_fixed_contract`（固定文言 `GRADCHECK_ANOMALY_HOLD_PROBE_BODY`）・`facade_does_not_reexport_or_declare_gradcheck_anomaly`・同 `_detects_each_category`・`workspace_declares_gradcheck_anomaly_fn_names_only_in_allowed_locations`（期待値は `autodiff/src/gradcheck.rs::gradcheck`・`autodiff/src/anomaly.rs::backward_detect_anomaly` 各 1 件）。
- 承認後（#2678）は doctest ガードを削除し、否定ガードを正ガードへ反転する。

### 9.4 申し送り

- facade 公開（`Tape::gradcheck`〈`TapeRef` アダプタ〉／`Tape::backward_detect_anomaly`＋`GradcheckOptions`／`GradcheckReport` の再エクスポート）: #2677 の承認後に #2678。
- CUDA（GB10）・Metal（M4 Max）実機 parity: `docs/perf/logs/gradcheck-anomaly-2671/README.md`（未実測）。
- `VarF64` 版 gradcheck・gradgradcheck・複数出力・`compat::Sequential::fit` への anomaly 結線・プロセスワイド／テープ単位の検出モード・テスト内に散在する私的な数値勾配ヘルパーの統合は本イシューの対象外。

## 10. #2678 実装記録（Phase 4 の facade 公開）

- 状態: **§7 の公開形を #2678 で承認形どおり公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 29・30を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2678 時点で当該コメントの承認に更新された（承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき）。
- 公開した識別子: facade `Tape::jacobian(&self, output, input) -> Result<Tensor<f32>, _>`・`Tape::hessian(&self, loss, input, child: &Tape)`・`Tape::backward_detect_anomaly(&self, loss) -> Result<Gradients, _>`（いずれも `&self.0`／`&child.0` を渡すだけの 1 行委譲）。本体は `crate::<module>::<fn>` への 1 行委譲（`Var`）／`&self.0` を渡すだけの 1 行委譲（`Tape`）に固定し、新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` は追加していない。
- ガードの反転・縮小: `JacobianHessianHoldDoctestGuard` は `Tape` の 2 本を外すと残すプローブが `jacobian_ops` モジュール名・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッドになり、`GradcheckAnomalyHoldDoctestGuard` は `Tape::backward_detect_anomaly` を外して `gradcheck`・`GradcheckOptions`・`GradcheckReport`・`anomaly` モジュール名等のプローブを残した（先例 #2516）。`api_surface.rs` の否定ガードは、承認済みの型名を識別子表から外し、`Tape` の承認済みメソッドを `fn` 宣言走査から除外したうえで、承認形だけを許す正ガードへ反転した。宣言場所インベントリには `autodiff/src/var.rs`（`Tape` 分は `facade/src/lib.rs`）の各 1 件を追加した。
- **`Tape::gradcheck` と `GradcheckOptions`／`GradcheckReport` は保留した（→ facade シグネチャは §11 で確定。公開は #2847）。** §3.4・§3.7・§9.4 は内部契約（`gradcheck(make_tape: Fn() -> Tape, f, inputs, options)`）と「facade 側はクロージャ引数を `TapeRef` に包むアダプタにする」ことしか定めておらず、facade メソッドのレシーバ（`&self` か関連関数か）と、newtype の `Tape` でテープ生成をどう受けるか（`tape_for` が fallible な点を含む）が書かれていない（複数の形が成り立つ。「記録に形が書かれていない点」）。型 2 つは `gradcheck` なしでは使えないため一緒に保留した。#2677 へ事実のみをコメント済み。
- `Tape::hessian` は `backward_create_graph` と同じく `child: &Tape`（facade の `Tape`）を取り、本体は `&child.0` を渡す。
- テスト: `crates/facade/tests/phase4_ops_facade.rs`（tape_jacobian_hessian_and_anomaly_signatures。`fandhe_ai::` だけを import し、fn ポインタ型でシグネチャを固定して厳密に決まる値を確認）と、`crates/facade/tests/api_surface.rs` の正ガード（`var_phase4_ops_methods_are_thin_delegations`・`facade_reexports_phase4_ops_types_only_in_approved_shape`・`facade_tape_phase4_methods_are_thin_delegations`・各 `workspace_declares_*`）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。CUDA／Metal 実機 parity は未実測で、`docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md` へ申し送る（新しい数値経路はなく、1 行委譲のため既存の各 `*_backend_parity.rs` の結果がそのまま適用される）。

## 11. #2846 決定記録（`Tape::gradcheck` の facade シグネチャ）

- 状態: **記録のみ（コード変更なし）。** 本節は §3.4・§3.7・§9.4・§10 が書いていなかった facade メソッドのレシーバとテープ生成の受け方を 1 案に定める。公開（`Tape::gradcheck`・`GradcheckOptions`・`GradcheckReport` の facade 公開と保留ガードの反転）は #2847 が行う。**公開までは `GradcheckAnomalyHoldDoctestGuard` と `api_surface.rs` の否定ガードを維持する。**
- 基準: `origin/main` `36741049`（2026-10-08）。
- 承認の根拠: ルート #2499 の 2026-10-08 ユーザーコメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）が示した次の 4 点の方向に限る。(1) `Tape::gradcheck` は `&self` を取らない関連関数、(2) テープは facade 内部で facade の通常の生成経路から作り、利用者にテープ生成関数を渡させない（生成失敗は `Err`）、(3) クロージャは `TapeRef<'a>` と `&[Var<'a>]` を受け `Result<Var<'a>, _>` を返す、(4) `GradcheckOptions`・`GradcheckReport` はクレートルートへ再エクスポート。下の 11.1〜11.2 のうち `device` 引数の有無と内部境界の変更は、(2) の「通常の生成経路」「生成失敗は `Err`」の 2 条件から導いた解釈であり、コメントが明示した文言ではない。

### 11.1 facade シグネチャ

```rust
impl Tape {
    pub fn gradcheck<F>(
        device: Device,
        f: F,
        inputs: &[Tensor<f32>],
        options: &GradcheckOptions,
    ) -> Result<GradcheckReport, AutodiffError>
    where
        F: for<'a> Fn(TapeRef<'a>, &[Var<'a>]) -> Result<Var<'a>, AutodiffError>;
}
```

- レシーバなし（関連関数）。呼び出しは `Tape::gradcheck(device, f, &inputs, &options)`。検査のたびに新しいテープを作り既存テープに触れないため `&self` を取る意味がない。
- 引数順は内部関数 `(make_tape, f, inputs, options)` の `make_tape` を `device` に置き換えた並び。
- ジェネリック境界は `F` の 1 つだけ。`'static`・`Send`・`Sync` は要求しない（内部関数も要求しない）。
- 戻り値とエラー型は内部関数と同じ。不一致は `Err` ではなく `Ok(report)` の `passed() == false`（§3.4 のまま）。判定式・`GradcheckOptions` に既定値を置かないこと・適用範囲・入口検査の順序は §3.4 と内部実装から変えない。

### 11.2 テープの生成元と失敗時の扱い

- 生成元は `tape_for(device)`（`crates/facade/src/lib.rs:1363`。`resolve_ops` を唯一の `Device` から `BackendOps` への変換点のまま使う）。評価ごとに 1 回呼ぶ（計 `1 + 2·Σn_k` 回）。
- 失敗は `BackendError` を `AutodiffError::Backend(_)` へ包んで `Err` で返す（既存の `From<BackendError>`。`crates/autodiff/src/error.rs`。新しい variant なし）。途中の評価で生成が失敗した場合も同じ経路で `Err` とし、部分的な `GradcheckReport` は返さない。
- 検査順序: 内部の入口検査（`inputs` が空・要素数オーバーフロー）が先、テープ生成はその後。「入力が空かつデバイス不正」は `InvalidArgument` が先に返る。facade 側でデバイスの事前検証は追加しない。
- `device` は必須引数とし既定デバイスを暗黙に選ばない（CPU で検査するなら `Device::Cpu`。`tape()` と `tape_for(Device::Cpu)` はどちらも `CpuBackendOps::new()` を結線するため挙動は同じ）。閾値に既定値を置かない §3.4 の方針と揃える。
- 既知のコスト: Metal では `tape_for` のたびにデバイス存在確認が走る（#2114）。gradcheck は検証用途のため許容する。
- **内部の変更（#2847 への指定）**: 内部 `gradcheck`／`evaluate` の `M` 境界を `Fn() -> Tape` から `Fn() -> Result<Tape, AutodiffError>` へ変える（`make_tape()` に `?` を付ける）。`crates/autodiff/src/gradcheck.rs` はタグ `v0.10.0` に含まれない（2026-10-07 追加・#2810）ため出荷済み API は壊れない。内部の呼び出し元 `crates/autodiff/tests/gradcheck_anomaly_parity.rs`・`crates/facade/tests/gradcheck_anomaly_backend_parity.rs` は `Ok(..)` で包む修正が要る。§3.4 の「内部の契約案」はこの形へ読み替える。
- facade 本体（アダプタ。1 行委譲ではない）:

```rust
fandhe_ai_autodiff::gradcheck::gradcheck(
    || tape_for(device).map(|t| t.0).map_err(AutodiffError::Backend),
    |t, xs| f(TapeRef::from_autodiff(t), xs),
    inputs,
    options,
)
```

**不採用案**

| 案 | 不採用の理由 |
|---|---|
| デバイス引数なしで `tape()`（CPU 固定） | 失敗経路が存在せず「生成失敗は `Err`」が空文になる。GPU 上の検査（`gradcheck_anomaly_backend_parity.rs` が想定）が facade から行えない |
| 内部境界 `Fn() -> Tape` を保ち facade で 1 回だけ事前検証する | `resolve_ops` の変換を二重化する。評価途中の生成失敗を `Err` にできない（`panic!`／フォールバックが要る） |
| 内部境界を保ち、失敗を `Cell` 等へ退避してダミーのテープを返す | 失敗後も評価が進み誤った結果を計算しうる（fail-closed でない） |
| `&self` を取り受け手と同じバックエンドで新テープを作る | `Tape` から `BackendOps` を複製する経路が無く `Tape` 本体の変更が要る。承認された方向（関連関数）とも異なる |
| 利用者にテープ生成クロージャを渡させる | 承認された方向に反する |

### 11.3 再エクスポート位置

- クレートルート `fandhe_ai::GradcheckOptions`・`fandhe_ai::GradcheckReport`。
- 形: `crates/facade/src/lib.rs` に `pub use fandhe_ai_autodiff::gradcheck::{GradcheckOptions, GradcheckReport};` に相当する、別名・newtype なしの 1 文 1 行。`gradcheck` モジュール自体と裸の自由関数 `gradcheck` は再エクスポートしない。型の中身（非公開フィールド・`GradcheckOptions::new` の fail-closed 検査・`Default` 非実装・アクセサ）は変えない。

### 11.4 `TapeRef<'a>` アダプタ

- **要る。** 内部がクロージャへ渡すのは `&'a fandhe_ai_autodiff::Tape` で、facade の利用者はこの型を名指しできない。所有型の facade `Tape` は借用から作れないため、承認方向の「`Tape` または `TapeRef<'a>`」は `TapeRef<'a>` に確定する。
- 形は 11.2 の本体の第 2 引数 `|t, xs| f(TapeRef::from_autodiff(t), xs)`。`TapeRef` は `Copy` で値渡し、`&[Var<'a>]` と戻り値はそのまま通す。`unsafe`・新しい型・トレイトなし。
- クロージャ内の定数は `TapeRef::var_no_grad`、追加の葉は `TapeRef::var`／`var_from`。外側のテープで作った `Var` を返すと内部の検査で `Err(TapeMismatch)`（既存契約のまま）。

### 11.5 anomaly 検出系の扱い

- `Tape::backward_detect_anomaly` は #2678 で公開済みで本件では変更しない。
- 保留ガードに残る anomaly 系（モジュール名 `anomaly`、裸の自由関数 `backward_detect_anomaly`、`Var`／`Tensor<f32>` 上の同名メソッド）は、決定記録に公開形が無いため**保留を続ける**。`compat::Sequential::fit` への結線、プロセスワイド／テープ単位の検出モードも §9.4 のとおり対象外。

### 11.6 #2847 への申し送り

- 内部 `gradcheck`／`evaluate` の `M` 境界の変更と、既存テスト 2 ファイルの追随。
- `facade_tape_phase4_methods_are_thin_delegations`（`api_surface.rs`）の 1 行委譲表には載せられないため、`gradcheck` 用の正ガード（承認形の本体トークン列の固定）を別に置く。
- `GradcheckAnomalyHoldDoctestGuard` から `Tape::gradcheck` と型 2 つのプローブを外し、残りの保留（`gradcheck`／`anomaly` モジュール名、裸の自由関数 2 つ、`Var`／`Tensor<f32>` 上のメソッド）を維持する。`GRADCHECK_ANOMALY_HOLD_PROBE_BODY`・否定ガード・宣言場所インベントリの更新を伴う。
- `crates/autodiff/src/gradcheck.rs` のモジュール doc と `crates/autodiff/src/lib.rs` の「保留」記述の更新。
- CUDA／Metal 実機 parity は未実測のまま `#[ignore]` で分離し、`docs/perf/logs/<slug>-2847/README.md` へ申し送る。
- HRTB クロージャのアダプタがコンパイルできない等、本節の形で実装できない点が出たら、実装せず停止して承認依頼へ戻す。
- 本節はコード・依存・tolerance・baseline・ガードレール閾値・`docs/spec` を変更しない。
