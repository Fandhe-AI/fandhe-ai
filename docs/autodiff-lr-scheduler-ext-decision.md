# LR scheduler 5 種（CosineAnnealingWarmRestarts・CyclicLr・LambdaLr・SequentialLr・MultiStepLr）の設計判断記録

イシュー #2176（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の行内深掘り）」）。

## 1. 背景

`docs/compat-feature-gap.md` の scheduler 表の対応漏れのうち、PyTorch
`torch.optim.lr_scheduler.{MultiStepLR, CosineAnnealingWarmRestarts,
CyclicLR, LambdaLR, SequentialLR}` に相当する 5 種が
`fandhe_ai_autodiff::nn::optim::lr_scheduler` に未実装だった
（`ConstantLr`・`StepLr`・`CosineAnnealingLr`・`ExponentialLr`・
`LinearWarmupLr`・`OneCycleLr` の 6 種のみ）。本イシューはこの 5 種を
追加する。

## 2. 公開 API とシグネチャ（イシュー記述との差異）

イシュー本文（受入基準）は「epoch → 学習率 `f32`」という抽象的な
記述だったが、既存 6 種はすべて共通 trait
[`LrScheduler::lr_at(&self, step: usize) -> f32`] を実装する形に
統一されている。本イシューもこの契約に揃え、新しいメソッドは追加
しない（公開面を最小にし、既存 trait と一貫させるため）。

型名も、イシュー受入基準に現れる別表記（`CosineWarmRestarts`・
`CyclicLR` 等）ではなく、承認事項節の表記（`CosineAnnealingWarmRestarts`・
`CyclicLr`・`LambdaLr`・`SequentialLr`・`MultiStepLr`）を採用した
（既存の `*Lr` 命名規約〈`StepLr`・`ExponentialLr` 等〉に揃えるため）。

各型の `new` シグネチャ:

```rust
pub fn MultiStepLr::new(base_lr: f32, milestones: &[usize], gamma: f32) -> Result<Self, AutodiffError>;
pub fn CosineAnnealingWarmRestarts::new(base_lr: f32, t_0: usize, t_mult: usize, eta_min: f32) -> Result<Self, AutodiffError>;
pub fn CyclicLr::new(base_lr: f32, max_lr: f32, step_size_up: usize, step_size_down: Option<usize>) -> Result<Self, AutodiffError>;
pub fn LambdaLr::<F: Fn(usize) -> f64>::new(base_lr: f32, lr_lambda: F) -> Result<Self, AutodiffError>;
pub fn SequentialLr::new(schedulers: Vec<Box<dyn LrScheduler>>, milestones: Vec<usize>) -> Result<Self, AutodiffError>;
```

`OneCycleLrConfig` のような専用 Config 構造体は導入しない（各型の
引数が 2〜4 個と少なく、位置引数のままでも可読性が保てるため。
`ConstantLr`・`StepLr`・`CosineAnnealingLr`・`ExponentialLr`・
`LinearWarmupLr` と同じ規則）。

## 3. 各型の数値仕様と PyTorch 突合

### 3.1 MultiStepLr

`lr(step) = base_lr * gamma^n`（`n` は `milestones` のうち `step` 以下
の個数）。`milestones` は構築時に昇順ソートして保持し、重複は保持する
（PyTorch `MultiStepLR.__init__` が `Counter(milestones)` で重複を
多重にカウントする挙動を再現するため）。`n` はソート済み列に対する
`partition_point(|&m| m <= step)` で求まる（`bisect_right(step)` と
同値の閉形式）。`gamma` の累乗は `f64::powf`（`count as f64`）で計算
する（`i32` へのキャストによる巨大 `milestones` 列でのラップを避ける）。

### 3.2 CosineAnnealingWarmRestarts

`lr(step) = eta_min + (base_lr - eta_min) * (1 + cos(π * t_cur / t_i)) / 2`
（`t_cur`／`t_i` は現在の周期内の位置と周期長）。

**意図的な逸脱**: PyTorch で `epoch` 引数を渡す経路（`step(epoch)`）は
周期番号を `int(math.log(...))` の**浮動小数**で求めるため、
`T_0=1, T_mult=10` 付近の境界（例: `epoch=111` で
`log(1000, 10) = 2.9999...`）で桁落ちにより 1 つ前の周期に誤って
丸まることがある。本実装は `t_cur`／`t_i` を**整数演算**
（`t_mult == 1` なら `usize` の剰余、それ以外は `u128` の
`checked_add`／`checked_mul` で周期境界を順に進める）で求めるため、
この誤判定を再現しない。PyTorch の引数なし `step()` による逐次更新
経路（本フィクスチャが使う経路）とは一致することを実測で確認した
（§4）。回帰テストは `nn_optim_lr_scheduler_ext.rs::
cosine_annealing_warm_restarts_fixes_pytorch_float_log_boundary`
（`t_0=1, t_mult=10, step=111` が `base_lr` を返すことを固定）。

`t_mult` による指数的な周期拡大が `u128` の範囲を超える病的な設定
（`step` が `usize::MAX` 付近かつ `t_mult` が非常に大きい場合）は
「現周期が `step` を含む」とみなし `t_i` を実質無限大
（`u128::MAX`）として扱う（`t_cur/t_i` が 0 に近づき `lr ≈ base_lr`
を返す安全側の近似。事実上到達しない防御的経路）。

### 3.3 CyclicLr

`mode='triangular'` 固定の三角波: 周期内位置 `pos = step % (up +
down)` として、`pos <= up` なら `scale = pos / up`（上昇フェーズ）、
それ以外は `scale = (up + down - pos) / down`（下降フェーズ）を用い
`lr = base_lr + (max_lr - base_lr) * scale`。この式は PyTorch の
`x = 1 + step/T - floor(1 + step/T)`（半周期換算）から導かれる式と
代数的に同値であることを閉形式突合で確認した。`mode='triangular2'`／
`'exp_range'`・`scale_fn`・`cycle_momentum` は対象外（§8「スコープ外」
参照）。

### 3.4 LambdaLr

`lr(step) = base_lr * lr_lambda(step)`（PyTorch `LambdaLR` と同じ）。
`lr_lambda: F where F: Fn(usize) -> f64` の generic として実装し
（`Box` 割り当てを要さない。呼び出し側の `Send`／`Sync` 性を引き継げ
る）、`Fn`（`FnMut` ではない）に限定する。param group ごとの lambda
リストは対象外。`lr_lambda` が非有限値を返した場合は `lr_at` の
戻り値へそのまま伝播する（`Result` を返せない trait 契約のため。
`ExponentialLr` 等と同じ扱い）。

### 3.5 SequentialLr

`milestones` で区切られた区間ごとに異なる `LrScheduler` を、区間内
では局所 epoch（`step - milestones[idx-1]`。区間先頭で 0 から再開）
で駆動する。`idx = milestones.partition_point(|&m| m <= step)`。これは
PyTorch `SequentialLR` が milestone で次段を `last_epoch=0` へ戻して
駆動する挙動の閉形式である（§4 の 2 段・3 段フィクスチャで実測突合
済み）。

PyTorch は全段が同じ optimizer の `initial_lr` を共有する前提だが、
本実装は各段が自分の `base_lr` を独立に持つ（各段は独立に構築した
`LrScheduler` のため）。PyTorch と同じ学習率系列を再現したい場合は、
呼び出し側が各段へ同じ `base_lr` を渡す（本実装のフィクスチャ・
facade テストはいずれもこの前提で構築している）。

状態保持型の `ReduceLrOnPlateau` は `Box<dyn LrScheduler>` へ所有権が
移り `step(metric)` を呼び出せなくなるため、`SequentialLr` の段としては
実質的に非対応（`lr_at` は現在値を返すだけの `ConstantLr` 型の段として
振る舞う）。

## 4. PyTorch 実行値 fixture

`crates/autodiff/tests/fixtures/lr-scheduler-ext-pytorch-reference/`
（README 参照）。torch 2.14.0+cpu の実行値と本実装の `lr_at` を
相対誤差 1e-6（期待値 0 の場合は完全一致）で突合する。10 ケース
（MultiStep 2・CAWR 2・Cyclic 2・Lambda 2・Sequential 2 段／3 段）を
30 epoch ぶん記録し、全ケース全 epoch で判定を通すことを確認済み
（`nn_optim_lr_scheduler_ext.rs::*_matches_pytorch_reference`）。

## 5. `docs/compat-api-scope.md` への反映

§1.2 の scheduler 行（`ConstantLr／StepLr／CosineAnnealingLr／
ExponentialLr／LinearWarmupLr／ReduceLROnPlateau／OneCycleLr` を
列挙している行）へ、本イシューで追加した 5 種を追記する（実装済み・
facade 公開は保留の旨を明記）。

## 6. OWASP Top 10 観点

- **A03 インジェクション／入力検証**: 全コンストラクタで入力を
  fail-closed で検証する（非有限値・非正・退化設定・長さ不一致・
  非単調 milestones）。整数周期演算は `checked_add`／`checked_mul`
  または `u128` で行い、巨大 `step`・巨大 `t_mult`／`step_size` に
  よる overflow・panic・無限ループを防ぐ（DoS 耐性）。シェル呼び出し・
  外部入力のパースはない。
- **A04 安全でない設計**: `lr_at` は `&self` だけを受けるので、副作用
  がないことは型で保証される。非有限な学習率は下流の `LrSchedule`／
  `set_lr` が既存の fail-closed 契約で拒否する。`LambdaLr` は利用者の
  コードを実行するが、信頼境界の内側（ライブラリ利用者自身のコード）
  であり新しい攻撃面ではない。
- **A06 脆弱なコンポーネント**: 依存は追加しない（`Cargo.toml`／
  `Cargo.lock` は不変）。torch は fixture 生成時に repo 外の一時 venv
  でのみ使い、CI にも依存グラフにも入れない。
- **A08 データ整合性**: fixture の生成条件と sha256 を README に記録
  する。facade の公開保留は正のプローブ doctest とソース走査テストの
  多層構成で固定し、承認なしで公開面が広がる経路を塞ぐ。

## 7. GPU（該当なし）

本イシューはホスト上の `f32`／`f64` 純関数のみを扱い、`Op`／
`BackendOps`／`Var`／VJP・CUDA／Metal のカーネルには一切触れない。
数値一致の複合判定（REQ-2）・tolerance 定数は対象外。実機依存テスト・
`docs/perf/logs` への申し送りもない。

## 8. 承認事項（本 PR では実施しない）

- **facade 純再エクスポート**（`docs/compat-api-scope.md` §5 経路 2）:
  `fandhe_ai::optim::{MultiStepLr, CosineAnnealingWarmRestarts,
  CyclicLr, LambdaLr, SequentialLr}` の `crates/facade/src/optim.rs`
  への追加。`ConstantLr`／`StepLr`／`CosineAnnealingLr`／
  `ExponentialLr`／`LinearWarmupLr`／`OneCycleLr` はすでにこの経路で
  公開済みだが、本 5 種は未承認のため保留する。親 #2131 が定める
  「facade 公開面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則
  （先例 #2171・#2173）に従う。保留固定は `crates/facade/src/
  lib.rs::LrSchedulerExtHoldDoctestGuard`（正のプローブ 1 ブロック
  方式。型名のみが対象で inherent メソッド追加を伴わないため trait
  プローブは不要）と `crates/facade/tests/api_surface.rs` の 4 テスト
  （doctest ドリフト検査 2 件・ソース走査 1 件＋自己テスト）で多層
  固定する。

承認後の作業: `optim.rs` へ `pub use fandhe_ai_autodiff::nn::optim::
{MultiStepLr, CosineAnnealingWarmRestarts, CyclicLr, LambdaLr,
SequentialLr};` を追加、`api_surface.rs` の
`optim_module_reexports_exactly_expected_surface` 期待集合・
`optim_types_are_reachable_via_facade_only` への追加、
`LrSchedulerExtHoldDoctestGuard`・対応する 4 テストの削除。

## 9. スコープ外

- `CyclicLr` の `mode='triangular2'`／`'exp_range'`・`scale_fn`・
  `cycle_momentum`（momentum cycling）
- `LambdaLr` の param group ごとの lambda リスト
- SGD／AdamW 等 optimizer との結線（`compile()` 統合。既存 6 種と
  同じく `LrSchedule::per_epoch` を経由する facade 側の責務）
