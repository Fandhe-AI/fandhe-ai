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
  - class_weight のキーが `< C`（C は logits の第 1 軸）
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
