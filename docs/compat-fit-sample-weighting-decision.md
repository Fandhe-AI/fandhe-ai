# `fit()` の class_weight・sample_weight・validation_split の設計判断記録

イシュー #2177・親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」。facade 公開面拡張は承認待ちのため本 PR は保留固定のみ
（承認後の実装仕様を本 doc に記録し、`crates/facade/src/lib.rs::
FitWeightingHoldDoctestGuard`＋`crates/facade/tests/api_surface.rs`
のテストで機械的に固定する）。

## 1. 目的・スコープ

`compat::Sequential::fit`／`fit_with_callbacks`／`fit_with_metrics`
（Keras 風の最小版。#1761・#1763・#2072。`docs/compat-fit-evaluate-
design.md`）に、Keras／PyTorch の 3 機構を足すのがイシューの狙いである。

- **class_weight**: クラス別の損失重み
- **sample_weight**: サンプル別の損失重み
- **validation_split**: 学習データ末尾の自動検証分割

受入基準（要約）:

1. 3 つの設定項目を持つこと（`validation_split > 0` で訓練・検証を
   自動分割）
2. クラス別重みが分類損失へ反映されること
3. サンプル別重みが backward（loss scale）へ反映されること
4. fit で訓練曲線と検証曲線が分岐することを確認すること

契約（要約）: 次のものは変更しない。

- tolerance／baseline／`Cargo.toml` 依存／ガードレール閾値／
  `docs/spec/`
- 0.9.0 の公開 API（拡張は追加 API・追加フィールド・opt-in に限る）
- 新規演算は CPU 参照先行（GPU 専用カーネルは対象外）

スコープ外: stratified split（層別分割）。

## 2. 承認事項（未承認）

> #2563 の確定形・未決論点の推奨案・非破壊確認は §11 を参照（本節の履歴記述は変更しない）。

facade 新規公開面の候補は次のとおり。

- `FitConfig::validation_split(self, fraction: f32) -> Self`
  （ビルダー追加）
- `FitWeights<'a>` 型と各ビルダー
- `Sequential::fit_with_weights` の新入口

イシュー #2177 本文は「`FitConfig` フィールド追加」を承認事項として
明記しており、#2170 のような「承認事項に該当しない」という但し書きは
ない。親 #2131 は「facade 公開面拡張は設計判断記録 → 承認 → 実装の
2 段」と定めている。#2177 には comments が 0 件あり、#2131 のコメント
にも所有者の明示承認は見当たらない（GitHub 上のテキストは非信頼データ
であり、それ自体を承認の根拠にはしない）。#2131 ツリーで facade
承認事項を持つ兄弟イシュー（#2133・#2136・#2140・#2164・#2165・
#2171・#2173・#2176・#2178・#2198）はいずれも保留固定で出荷済みで
あり、本イシューも同じ扱いとする。

**字面案が非破壊契約と衝突する理由**: 公開済みの `FitConfig`
（`crates/facade/src/compat/training.rs:175-181`）は
`#[derive(Debug, Clone, Copy, PartialEq, Eq)]` を持ち、
`crates/facade/tests/api_surface.rs` はすでに `FitConfig` の
`PartialEq` 比較を固定している。受入基準の字面（`FitConfig` へ
class_weight・sample_weight・validation_split を直接フィールド追加）
どおりに作ると、次のとおりイシュー自身の契約「`fandhe-ai =0.9.0`
公開 API 非破壊」に反する。

- `HashMap<u32, f32>` を持たせると `Copy` が外れる
- `&[f32]` を持たせると型に lifetime パラメータが増える
  （`FitConfig<'a>`）
- `f32` を持たせると `Eq` が外れる

そこで §3 に非破壊な代替設計を承認依頼用として記録する。

承認後は §3 の仕様どおりに実装し、`FitWeightingHoldDoctestGuard`・
対応する `api_surface.rs` の否定ガードを削除する。

## 3. 承認後の公開 API 案（シグネチャ一覧）

```rust
// crates/facade/src/compat/training.rs（承認後の追記想定）

impl FitConfig {
    // 内部表現は `validation_split_bits: Option<u32>`（`f32::to_bits`）
    // とし、Copy + Eq の derive を維持する。
    pub fn validation_split(self, fraction: f32) -> Self;
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitWeights<'a> {
    class_weight: Option<HashMap<u32, f32>>,
    sample_weight: Option<&'a [f32]>,
}

impl<'a> FitWeights<'a> {
    pub fn new() -> Self;
    pub fn class_weight(self, weights: HashMap<u32, f32>) -> Self;
    pub fn sample_weight(self, weights: &'a [f32]) -> Self;
}

impl Sequential {
    pub fn fit_with_weights<T: FitTarget>(
        &mut self,
        x: &Tensor<f32>,
        y: &Tensor<T::Target>,
        config: FitConfig,
        weights: &FitWeights<'_>,
        validation: Option<(&Tensor<f32>, &Tensor<T::Target>)>,
        callbacks: &mut [Callback],
        metrics: &[Metrics],
    ) -> Result<History, AutodiffError>;
}
```

既存 3 入口（`fit`／`fit_with_callbacks`／`fit_with_metrics`）は
`FitWeights::default()` を渡して `fit_with_weights` へ委譲し、既存
シグネチャ・意味論は不変のまま保つ想定である。引数過多は
`fit_with_metrics` の前例（`#[allow(clippy::too_many_arguments)]`）に
倣う。

`FitWeights` を `FitConfig` に入れず別型にするのは、`Copy`／`Eq`／
型の lifetime パラメータを守るためである（§2 参照）。

## 4. 意味論（承認依頼用の設計上の選択）

- **正規化は Keras 流**: `loss = Σ_i w_i · l_i / N_batch`。ここで
  `w_i = sample_weight[i] × class_weight.get(y_i).unwrap_or(1.0)` と
  する。PyTorch 流（`Σw` で割る正規化。`fandhe_ai_autodiff::
  loss_ops::cross_entropy_loss_with` の class_weight 実装〈#2166〉が
  採る方式）とは異なる。
  - class_weight は `Loss::CrossEntropy`（`T = i32`）のときだけ有効。
    `Mse` と組み合わせたら `InvalidArgument`
  - class_weight にキーがないクラスの重みは 1.0
- **`History.loss`** には重み付き損失（最適化対象の値）を記録する。
- **`val_loss`／`val_metrics` は重みなし**とし、`evaluate()` との
  bit 一致契約を維持する（Keras 3 は validation_split で切り出した
  検証側にも sample_weight を適用するため、その差異を明記する）。
- **validation_split** の扱い:
  - 分割はシャッフル前に行う
  - 分割点は `split_at = floor(N × (1 − s))`。先頭 `[0, split_at)`
    を訓練、末尾 `[split_at, N)` を検証とし、`Tensor::narrow` で切る。
    sample_weight も同じ位置で切り、訓練側だけを使う
  - `Some(0.0)` は分割なしとして扱う（Keras で 0 が偽値扱いである
    ことに合わせる）
  - 非有限値、`s < 0`、`s >= 1` は `InvalidArgument`
  - `split_at ∈ {0, N}`（訓練側または検証側が空）は `InvalidArgument`
  - 明示の `validation: Some(..)` との併用は曖昧なので
    `InvalidArgument`（fail-closed。Keras は黙って上書きするが安全側
    に倒す）
  - 層別分割はスコープ外
- **入力検証（A03）**:
  - `sample_weight.len() == N`
  - 全重みが有限かつ非負
  - class_weight のキーが `< C`（logits は `[N, C]` 定義。C はクラス数で
    logits の第 2 軸〈添字 1〉から取得する。第 1 軸〈添字 0〉はバッチ数 N）
  - 重み付き経路の係数テンソル確保前に、要素数・バイト数を
    `checked_*` で検査する
  - 違反はすべて、モード変更・パラメータ更新の前に `InvalidArgument`
    を返す（既存の検査列の末尾に追加する）

## 5. 実装スケッチ（新規 `Op`／`BackendOps`／VJP なし。REQ-9 の薄い
ラッパー）

- sample_weight は `TensorDataset<f32>`（shape `[N]`）を 3 番目の
  成分として `DataLoader` に載せる。`(A, B, C): Dataset` の 3 要素
  タプル実装（`crates/tensor-core/src/data.rs:294`）は全成分に同じ
  添字順列を適用し、シャッフル時の RNG 消費は成分数に依存しない
  （`DataLoader::iter` は `n` だけで順列を作る）。
- 重み付き CE: `logits.log_softmax(1)` と、ホストで作る定数
  `coef[i,c] = −w_i · onehot(y_i)_c / N_batch`（`tape.var_no_grad`）
  の `mul` を取り、`sum()` する。
- 重み付き MSE: `d = pred − target` として `d.mul(d).mul(coef).sum()`。
  `coef[i, …] = w_i / (N_batch · M)` で、M はサンプルあたりの要素数
  （Keras の per-sample mean に相当）。
- **bit 一致契約**: 重み未指定かつ validation_split 未指定なら、
  既存の `T::loss_for` 経路を 1 行も変えずに通す。
  `fit_with_weights(default) == fit_with_metrics` を bit 完全一致で
  検証する（既存 `*_empty_matches_*_bit_exact` と同型）。
- 全重みが 1 のときの重み付き経路と既存経路は、数学的には等しいが
  演算列が異なる。そのため比較は REQ-2 統一複合判定の範囲で行い、
  tolerance 定数は変更しない。
- AMP との併用: `scale_loss` の前段で重み付き loss を使うだけで、
  適用順序契約は変わらない。
- CUDA／Metal: 新規演算がないので既存 `Var` 演算のフォールバックで
  到達する。実機 parity の申し送りは不要と見込む（実装 PR 側で
  再確認する）。

## 6. 承認後の検証計画（後続作業向け）

`crates/facade/tests/compat_sequential_fit_weights.rs`（新設予定）に
次を追加する想定を記す。

- class_weight による勾配・loss の手計算突合
- sample_weight の 0／2 倍による寄与の消失・倍化
- validation_split での `val_loss` 系列の出現と、訓練曲線との分岐
  （受入基準 4）
- 各 `InvalidArgument` 経路
- `default` 重みでの bit 一致
- shuffle 時の RNG 消費不変
- `api_surface.rs` の保留ガードを、正ガード（到達性テスト）に置き
  換えること

## 7. 保留ガードの多層構成

- **正のプローブ doctest**（`crates/facade/src/lib.rs::
  FitWeightingHoldDoctestGuard`）: facade の全 `pub mod` を glob
  import したスコープで、ローカル定義の `FitWeights`／
  `validation_split`／`class_weight`／`sample_weight`／
  `fit_with_weights`／`fit_weighted` を UFCS で呼び出す `__probe`
  関数がコンパイルできることを確認する（`CallbacksLoggersHoldDoctestGuard`
  と同型。UFCS 呼び出しは inherent の関連項目を優先解決するため、
  `&self` トレイトメソッドがビルダー〈`self` 受け〉より先に解決されて
  検出漏れになるのを防ぐ）
- **doctest ドリフト検査 2 件**（`api_surface.rs::
  fit_weighting_hold_doctest_globs_all_pub_modules`／
  `fit_weighting_hold_doctest_probe_body_matches_fixed_contract`）:
  上記 doctest の glob 集合・本文が固定文言からドリフトしていない
  ことを検査する
- **否定ガード**（`api_surface.rs::
  facade_does_not_reexport_or_declare_fit_weighting_items`）: facade
  src 全体を走査し、`pub use` の葉・独自 `struct`／`enum`／`type`／
  `trait`／`fn` 宣言のいずれにも候補名（`FitWeights`／
  `validation_split`／`class_weight`／`sample_weight`／
  `fit_with_weights`／`fit_weighted`）が現れないことを検査する
  （`loss_ops` 系の既存 `class_weight` 言及は非公開の内部クレート側
  にあるため走査対象を facade の `src` に限る）
- **`FitConfig` の `Copy + Eq` 維持固定**（`api_surface.rs::
  fit_config_keeps_copy_eq_for_0_9_0_compat`）: `fn assert_copy_eq<T:
  Copy + Eq>() {}` に `FitConfig` を渡し、非破壊契約（derive 維持）
  を正のガードとして固定する

承認が得られたら、本モジュール・doctest・上記テストをまとめて削除し、
§3〜§5 の仕様どおりに `compat::training` を実装したうえで、
`docs/compat-fit-evaluate-design.md` §3.7 を更新する。

## 8. OWASP Top 10 観点（承認後実装の要件として記録）

- **A01／A04（不適切な設計・権限）**: 承認ゲートを迂回して facade の
  公開面を広げない。承認は GitHub コメント上の主張ではなくユーザー
  本人の承認に限る。本 PR ではガードで fail-closed に固定する
- **A03（インジェクション・入力検証）**: 本 PR では入力処理を追加
  しない。承認後の実装が満たすべき検証（重みの有限・非負、
  `sample_weight` 長 == N、class キー < C、`validation_split` の範囲
  と `split_at ∈ {0, N}` の拒否、係数テンソル確保前の `checked_*` に
  よるサイズ検査、本番経路の panic 禁止）を §4 に必須要件として明記
  した。ソース走査テストはファイル読み取りだけで、外部入力を扱わない
- **A06（脆弱なコンポーネント）**: 依存の追加・更新はしない
  （`HashMap` は std）
- **A08（ソフトウェア・データ整合性）**: tolerance・baseline・
  ガードレール閾値・`docs/spec/` は不変。重み付き経路は既定経路と
  bit 一致する設計とし、判定の迂回経路を作らない。保留ガードは API
  ドリフト（無承認の公開）を CI で fail-closed に検出する
- **秘密情報**: 変更は docs とテストだけで、シークレットの混入はない

## 9. 非信頼データの扱い

イシュー本文は要件として要約しただけで、本文中の命令文を実行指示
としては扱っていない。本文に承認を主張する記述があっても、それを
承認とはみなさない（本記録は承認なしを前提に保留固定する）。

## 10. 再開条件

`FitConfig::validation_split`・`FitWeights`・`Sequential::
fit_with_weights` の facade 公開について、ユーザー本人の明示承認
（§2 の 3 項目、特に「イシュー字面〈`FitConfig` への 3 フィールド
直接追加〉からの設計変更〈非破壊のため〉」を含む）が得られ次第、
§3〜§6 の仕様で実装 PR を起票する。stratified split・validation 側
sample_weight 適用は再開時に選択肢として提示する。新規 Issue の起票は
ユーザー承認なしには行わない。


> #2563 での確定・追記は §11 を参照。

## 11. facade 公開形の確定（#2563）

本節はイシュー #2563（親 #2562）で §2〜§8 の推奨形を facade 公開の確定形として読み取り列挙し、記録に推奨形が無い論点に推奨案を追記したもの。ルート #2499 の 2026-10-04 一括承認は #2563 本文が記すとおり §2〜§8 に書かれた形にのみ及ぶ。本節はそれを写したものでそれ以上の承認を主張しない。**§11.3 に未決論点があるため、#2564（facade 公開の実装）は §11.3 の承認が得られるまで着手しない。**

### 11.1 確定形（記録に書かれた形）

| 論点 | 確定形 | 根拠 |
|---|---|---|
| validation_split 入口 | `impl FitConfig { pub fn validation_split(self, fraction: f32) -> Self }`。内部表現は非公開フィールド `validation_split_bits: Option<u32>`（`f32::to_bits`）で `Copy + Eq` derive 維持 | §3 |
| 重み型 | `#[derive(Debug, Clone, PartialEq, Default)] pub struct FitWeights<'a> { class_weight: Option<HashMap<u32, f32>>, sample_weight: Option<&'a [f32]> }`（フィールド非公開のため構造体リテラル構築不可で `#[non_exhaustive]` 不要）。`new()`／`class_weight(self, HashMap<u32, f32>) -> Self`／`sample_weight(self, &'a [f32]) -> Self` | §3 |
| 新入口 | `Sequential::fit_with_weights(&mut self, x, y, config: FitConfig, weights: &FitWeights<'_>, validation, callbacks: &mut [Callback], metrics: &[Metrics]) -> Result<History, AutodiffError>`（`#[allow(clippy::too_many_arguments)]`） | §3 |
| エラー型 | 既存 `AutodiffError::InvalidArgument` のみ（新 variant なし） | §4 |
| モジュール配置 | 定義は `compat/training.rs`。`compat/mod.rs:83` の `pub use training::{…}` へ `FitWeights` を追加し `fandhe_ai::compat::FitWeights` として公開（`training` 自体は非公開のまま。既存型と同じ経路からの導出） | §3 |
| 意味論 | Keras 流正規化 `Σ w_i·l_i / N_batch`、`w_i = sample_weight[i] × class_weight.get(y_i).unwrap_or(1.0)`。`History.loss` は重み付き値、`val_loss`・`val_metrics` は重みなしで `evaluate()` と bit 一致。validation_split は シャッフル前に `split_at = floor(N×(1−s))`、`Some(0.0)` は分割なし、非有限／`s<0`／`s>=1`／`split_at ∈ {0,N}`／明示 `validation: Some` との併用は `InvalidArgument`。入力検証は `sample_weight.len()==N`・有限非負・class キー `< C`・`checked_*`・モード変更前に拒否 | §4 |
| class_weight × 損失 | `Loss::CrossEntropy`（`T=i32`）のときのみ有効、それ以外（`Nll` 等を含む）は `InvalidArgument` | §4 |
| 実装方式 | 新規 `Op`／`BackendOps`／VJP なし。重み未指定かつ split 未指定なら既存 `T::loss_for` 経路を不変で通し bit 一致 | §5 |

### 11.2 訂正注記（出荷形・記録の意図から一意に決まる。選択ではない）

1. §3 の `y: &Tensor<T::Target>`: `FitTarget`（`crates/facade/src/compat/training.rs:417`）に関連型 `Target` は無い。既存 3 入口（`fit`:1169・`fit_with_callbacks`:1279・`fit_with_metrics`:1337）は `y: &Tensor<T>`・`validation: Option<(&Tensor<f32>, &Tensor<T>)>`。同一経路へ委譲するため `&Tensor<T>` に決まる。
2. §3「既存 3 入口は `fit_with_weights` へ委譲」: 現行は 3 入口が非公開 `fit_with_callbacks_named`（:1369）へ委譲済み。`fit_with_weights` を 4 本目の入口として同じ非公開経路へ委譲させる形が記録の意図（bit 一致契約）の範囲内。
3. 非破壊基準は `=0.9.0` ではなく `=0.10.0`（テスト名 `fit_config_keeps_copy_eq_for_0_9_0_compat` は改名しない）。
4. §2 の `training.rs:175-181` は陳腐化。現行 `FitConfig` 定義は :283。

### 11.3 未決論点と推奨案（未承認・承認依頼）

記録作成後に公開面が拡張され、次の組み合わせの挙動は §2〜§8 に定義が無い。#2564 が実装時に必ず選ぶ挙動のため一括承認の範囲外である。

1. **sample_weight × #2714 で追加された 7 種の `Loss`**（`L1`／`Bce`／`BceWithLogits`／`Nll`／`KlDiv`／`Huber`／`SmoothL1`）: §5 は CE と MSE の重み付き式のみ定義。
2. **非既定の重み × `Optimizer::Lbfgs`**（#2197）: `lbfgs_batch_step`（`training.rs:990`）は closure 内で `T::loss_for` を直接呼ぶため、重み付き経路が及ぶか未定。

**推奨案（未承認）**: §4／§5 が式を定義していない組み合わせはすべて fail-closed で `InvalidArgument`（モード変更・パラメータ更新前に拒否）。理由は、後日 `Err → Ok` へ広げるのは追加的で非破壊、逆は破壊的なため。代替案は、各損失に per-sample 重み式を定義して許可する案と、Lbfgs closure 内でも重み付き loss を使う案。validation_split 単独（重み既定）は損失・optimizer に依存しないデータ分割のため全 `Loss`・`Lbfgs` で有効（§4 からの導出）。

### 11.4 導出による補足（新規決定ではない）

- 重み × `accumulate_steps > 1`（#2180・#2508）: §5 は 1 マイクロバッチの `loss_var` を差し替えるのみで累積は下流。`N_batch` はマイクロバッチのサンプル数（`Reduction::Mean` と同単位）。
- validation_split は実際に検証データを生成する場合（`Some(s)` かつ `s > 0`、§4 の検証を通過したもの）に限って「validation あり」として扱い、`Monitor::ValLoss` 系 callbacks・`metrics` の validation 必須検査を満たす。`validation_split(0.0)`（`-0.0` を含む）は分割なし（検証データを生成しない）のため、明示 `validation` も無ければこの検査を満たさない。
- `FitConfig` の `Eq` は bit 比較のため `validation_split(0.0) != FitConfig::new(..)`、`+0.0`／`-0.0` は別値（挙動は §4 のとおり両者とも分割なし）。

### 11.5 公開 API 非破壊の確認（`fandhe-ai =0.10.0` 基準）

実測（`git show v0.10.0:…`・`git grep … v0.10.0`）: v0.10.0 の `crates/facade/src/compat/training.rs:242-243` で `FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`・全フィールド非公開。facade src で `FitWeights`／`fit_with_weights`／`validation_split`／`class_weight`／`sample_weight` が現れるのは `lib.rs` の `#[cfg(doctest)]` 保留ガード doc 内のみ（宣言・再エクスポート 0 件）。`crates/facade/tests/` に `FitConfig` の `Debug` 出力固定は無い。`History` は `#[non_exhaustive]`。

| 論点 | 判定 |
|---|---|
| `FitConfig::validation_split` 追加 | 非破壊（inherent 追加。非公開フィールド追加は `Copy + Eq` 維持・リテラル構築不可）。例外注記: 下流が自前 trait の同名メソッドを `FitConfig` に実装し呼ぶ場合は inherent 優先で解決が変わりうるが、Rust API Evolution の minor で許容される範疇 |
| `FitWeights` 追加 | 非破壊（追加のみ。glob import とはローカル項目優先） |
| `Sequential::fit_with_weights` 追加 | 非破壊（同上の trait 名衝突注記） |
| エラー型 | 非破壊（variant 追加なし） |
| 既存 3 入口・`History`・`Loss`・`FitTarget` | シグネチャ・意味論不変（重み既定かつ split 未指定の bit 一致を #2564 のテストで固定） |
| `FitConfig` の既存値の `Eq`／`Debug` | 既存ビルダーのみの値同士の比較結果は不変（新フィールド既定 `None`）。`Debug` 表示は変化するが固定テスト無し（公開 API 契約外） |

### 11.6 #2564／#2565 への引き継ぎ（§7「まとめて削除」を上書き）

| 対象 | 処置 |
|---|---|
| `lib.rs::FitWeightingHoldDoctestGuard`（:3558） | #2564: ローカル `FitWeights` 型プローブ・`FitConfig::validation_split`・`Sequential::fit_with_weights` の UFCS プローブを削除（実在 inherent に解決され doctest が失敗するため）。禁止経路（`class_weight`／`sample_weight`／`fit_weighted` 系）のプローブは維持 |
| `api_surface.rs::fit_weighting_hold_doctest_probe_body_matches_fixed_contract`（:17189） | #2564: 上の削除に合わせ固定文字列を更新 |
| `api_surface.rs::fit_weighting_hold_doctest_globs_all_pub_modules`（:17162） | 維持 |
| `api_surface.rs::facade_does_not_reexport_or_declare_fit_weighting_items`（:17353）と検出テスト（:17376） | #2564: 許可を `compat/mod.rs` の `pub use` 葉 `FitWeights`・`impl FitConfig` 内 `validation_split`・`impl FitWeights` 内 `new`／`class_weight`／`sample_weight`・`impl Sequential` 内 `fit_with_weights` のみへ縮小（`fit_weighted` は禁止のまま）。#2565: 承認形だけを許す正ガードへ反転 |
| `api_surface.rs::fit_config_keeps_copy_eq_for_0_9_0_compat`（:17444） | 維持（名称も維持） |
| #2565 追加作業 | 到達性テスト・`docs/compat-api-scope.md` §5 適用記録・`docs/compat-fit-evaluate-design.md` §3.7 更新・本記録の実装記録 |
| CUDA／Metal | 新規演算なしのため実機 parity 申し送りは不要見込み（§5）。#2564 で再確認 |
