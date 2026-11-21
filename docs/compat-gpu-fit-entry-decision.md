# GPU 学習入口（Keras 風 `fit`）の公開形 決定記録

イシュー #4359（親 #4358「入口の設計記録」・ルート #4357「Sequential の GPU 学習」）。
`docs/compat-train-step-hook-decision.md` §0 が定める「公開面の追加は設計判断記録 →
実装の 2 段」の 1 段目にあたる。本記録は**入口の形の比較と推奨形の提示のみ**を扱い、
コード（`.rs`）・`Cargo.toml`・`docs/spec`・tolerance・baseline・`compat-api-scope.md` は変更しない。
推奨形は**未承認**であり、承認は #4361（owner:human）で行う。

## §0 結論

| 項目 | 内容 |
|---|---|
| 推奨形 | **案 B**（専用入口 `Sequential::fit_on`／`fit_on_with_callbacks` の追加。§4） |
| 変更の種類 | **追加のみ**。既存 `fit` 系 6 系統・`FitConfig`・`train_step_resident` のシグネチャと挙動は不変 |
| 公開状態 | **未公開・未承認**（承認は #4361。実装は #4374 配下の #4375〜#4378） |
| 未対応構成 | store を作る前に型付きエラー（`Unsupported`）で拒否し、CPU へ暗黙にフォールバックしない（§6） |
| `unsafe`・依存 | 追加なし |
| REQ-12 | 入口が受けるのは `Device` 識別子だけ（3 案とも同じ）。詳細は #4360 が §11 に記入 |

## §1 位置づけ・出典

- 課題: `compat::Sequential` の `fit` 系は内部で `crate::tape()`（CPU 固定）を使い、デバイスを
  指定できない（`crates/facade/src/compat/training.rs`）。GPU 学習の既存入口は低レベルの
  `init_device_param_store` → 専用の学習ステップ `train_step_resident`（MSE mean・SGD 固定、
  Linear＋ReLU のみ。`crates/facade/src/compat/sequential.rs`）と、Tape の組み合わせ経路
  （`forward_resident` → 損失 → `Tape::backward_device_param_store` →
  `Tape::step_device_param_store_adam` 等）である。MSE mean・SGD・Linear＋ReLU の制限は
  専用の学習ステップ入口 `train_step_resident` に限る。どちらも Keras 風の書き方ではなく、
  `fit` 系からは GPU を使えない。
- ハンズオン（2026-11-21）で、低レベル経路なら CrossEntropy＋Adam が Metal 上で動いたと
  イシュー本文に記載されている。**これはイシュー本文の記述の引用であり、本記録では検証していない**。
- 関連 issue: 兄弟 #4360（対応範囲の段階分け・REQ-2・REQ-12・転送回数）、承認 #4361、
  実装 #4374（#4375〜#4378）、層拡張 #4379／#4383／#4386、parity #4392、site #4396、
  チェーン拡張 #4362（CrossEntropy）・#4367（Adam／AdamW）・#4370（活性化）。
- 関連 doc: `compat-resident-train-step-decision.md`（§3 案 A/B・§4 P1〜P7・§5 範囲外 2 段・
  §6 REQ-12・§7 不変条件）、`compat-fit-evaluate-design.md`、`compat-callbacks-design.md`
  （§4.2 callbacks は `FitConfig` に含めない・§4.5 LR 同期）、`compat-train-step-hook-decision.md`、
  `facade-device-handle-design.md`、`train-step-resident-parts-design.md`、`compat-api-scope.md` §0・§5。
- コードで確認した現状（シンボル名で示す。行番号は動くため書かない）:
  - `fandhe_ai::tape_for(Device)` は存在する（`crates/facade/src/lib.rs`）
  - 常駐入口の拒否は `Sequential::contains_resident_unsupported_layer`・
    `reject_untracked_parametric_layer_resident`
  - `FitConfig` は `Copy + Eq`（`api_surface.rs::fit_types_are_reachable_via_facade_only` が固定）
  - `Device` は `Copy + Eq + Hash`・`#[non_exhaustive]`

## §2 比較する候補

| 案 | 内容 |
|---|---|
| A | `FitConfig` にデバイスを持たせる（`FitConfig::new(..).device(Device::Metal)` を既存 `fit` 系へ渡す） |
| B | 専用入口を足す（`Sequential::fit_on(device, x, y, config)` と callbacks・validation・metrics 版） |
| C | `compile` 時に指定する（`compile_on(device, optimizer, loss)`。`Compiled` がデバイスと `DeviceParamStore` を持ち続ける） |
| D（参考） | 既存 `fit` を `tape_for(device)` の Tape 経路で回す（入口追加なし）。`TapeNode` の値はホスト専用で演算ごとに転送が起きる。`compat-resident-train-step-decision.md` §3 の案 B（Tape のデバイス値化）不採用と同じ理由で退ける |

## §3 比較表

| 軸 | A: FitConfig | B: 専用入口（推奨） | C: compile 時 |
|---|---|---|---|
| store・デバイス側 optimizer 状態の置き場と寿命 | fit ごとに作るか `Sequential` に持つか決める必要があり、`fit` を 2 回呼ぶ不変条件（`fit_twice_equals_fit_once_with_double_epochs`）に波及 | 呼び出しの中に閉じる。終了時にホストへ書き戻す。ホストが重みの正のまま | `Compiled` が store を持ち続ける。`save_model`／`state_dict`／`predict`／`evaluate` との間で重みの正が二重になる。CPU と GPU の fit を交互に呼ぶと不整合を管理する必要がある |
| `FitConfig: Copy + Eq` | `Device` は `Copy + Eq` なので型としては載るが、資源の指定を値の設定へ混ぜる。#4415 の `Default`・builder 提案とは両立するが `Default` は CPU になる | 不変 | 不変 |
| 入口の増え方 | `fit`・`fit_with_callbacks`・`fit_with_metrics`・`fit_with_prefetch`・`fit_with_train_step`・`fit_with_weights` の 6 系統すべてに未対応組合せの検査が要る | 新入口 2 本だけ。フック・prefetch・重み付けは引数がなく構造的に受けない | 既存 6 系統すべてに波及 |
| auto trait | 影響は設計次第 | 影響なし | `DeviceParamStore` は `!Send`。`Sequential` が持つと `Send` 性が変わりうる。現状の `Sequential` は `nn::Module` に `Send` 境界がなく `Vec<Box<dyn Module>>` を保持するため `!Send` と読めるが、コンパイルでの確認は**未確認**。もし `Send` なら案 C は追加のみにならない |
| 既存挙動の変化 | 設定値で既存シグネチャの挙動が分岐する | なし | `compile` 後の fit の挙動がデバイスで分岐する |
| 未対応構成の拒否 | 実行時検査のみ | シグネチャで受けない組合せを作れる。残りは実行時に拒否 | 実行時検査のみ |
| REQ-12 | `Device` 識別子だけ | 同左 | 同左 |

案 D は演算ごとのホスト転送が残るため、転送回数の面で GPU 学習の目的（#4357）を満たせず不採用。

## §4 推奨形（案 B）

本記録の規範（「〜とする」）は次の 2 つに限る。(A) 公開形（入口の名前・署名・エラー型の区分・推奨案と比較）、
(B) ユーザーから観測できる挙動の既存 CPU 入口との等価（本節の等価規定・(i)〜(iii)・例外 E1〜E4）。
内部実装の契約（経路の切替・store の扱い・転送・D2H の頻度等）は規範にせず §17 へ申し送る。

理由: 既存 `fit` 系 6 系統と `FitConfig` を変えない・未対応の組合せをシグネチャで受けない・
store を呼び出しの中へ閉じて重みの正を二重にしない・`compat-resident-train-step-decision.md`
P5 の「将来の損失・最適化は別入口の追加で扱う」と同じ方向、の 4 点。

下流 issue（#4375〜#4378）向けのたたき台（最終形は #4361 の承認時に確定する。変える場合は同じ PR で本記録を更新する）。

```rust
impl Sequential {
    pub fn fit_on<T: FitTarget>(
        &mut self, device: Device, x: &Tensor<f32>, y: &Tensor<T>, config: FitConfig,
    ) -> Result<History, AutodiffError>;

    pub fn fit_on_with_callbacks<T: FitTarget>(
        &mut self, device: Device, x: &Tensor<f32>, y: &Tensor<T>, config: FitConfig,
        validation: Option<(&Tensor<f32>, &Tensor<T>)>,
        callbacks: &mut [Callback], metrics: &[Metrics],
    ) -> Result<History, AutodiffError>;
}
```

`compile` は既存のまま使い、optimizer・loss はデバイスに依らない設定として `Compiled` に残す。
`Device::Cpu` も受け付ける（CPU の `fit` へは委譲しない）。

### 既存 CPU 入口との等価（(B)。本記録の他の節はこの規定を参照し、独自の細部を書かない）

`fit_on`／`fit_on_with_callbacks` は、既存の `fit_with_callbacks_named` → `run_fit` →
`run_evaluate_with_metrics`（`crates/facade/src/compat/training.rs`）と、次を**同じにする**。
各項目は既存の同名関数の挙動を正とし、本記録では細部を言い換えない（細部が要るときは該当関数を読む）。

- 処理順と検査の位置: 引数・組合せの検証群 → `EarlyStopping` のリセット → ロガー・EMA・SWA の `begin_fit` →
  `run_fit`（データセット／ローダー構築 → epoch ループ）。検査をどの位置で行うかも既存と同じ（E1 を除く）
- バッチ分割と端数バッチ: 学習は `FitConfig`（`drop_last` を含む）の扱いを含めて `run_fit` と同じ、
  検証は `run_evaluate_with_metrics` と同じ。異なる分割・集計は許さない
  （`History.val_loss`・`EarlyStopping`・`ModelCheckpoint` の判断を変えるため）。
  分割はホスト側の既存ローダーが担うため、`drop_last` は初期段階の `Unsupported` にしない
- callback のリセット規則: 既存の `fit_with_callbacks` と同じ（`callbacks.rs` 冒頭「epoch 番号の数え方」）。
  エラー直後に観測できるのは失敗した呼び出しが残した状態のままで、再呼び出し後は既存の規則に従う
  （`EarlyStopping` は `reset_for_fit` 到達でリセット、`ModelCheckpoint`・`LrSchedule`・`Ema`・`Swa` は継続）。
  引き継がせたくない callback は呼び出し側が新しく用意する（`fit_on` は callback を受けない）
- 学習率: optimizer に残るのは各 epoch 開始時に反映した値（最終 epoch で実際に使った値）である。
  epoch 末に callback が更新した次回用の値は callback が保持し、optimizer へは反映しない（既存を正とする。
  `compat-callbacks-design.md` §4.5）
- optimizer 状態: fit 呼び出しの間の持ち越しと終了時のホストへの保存（初回呼び出しで生成される状態を含む）は
  既存と同じ。ホストへ戻せない optimizer は初期段階で `Unsupported`（E3）
- 終了時の重み: `restore_best_weights` を含め既存と同じ（正常終了時のみベスト重みを適用し、重みだけを戻す）
- エラー: 既存の処理が返すエラーは既存と同じ型・variant で返す（エラー表）
- **ユーザーから観測できる出力**（`History`、callback・`save_model` に渡す重み、終了後のホストモデル）は、
  次の 3 段で既存と等価にする:
  - (i) **形式と手順**: 出力の型・キー・形状、`History` の各列の集計手順（どのバッチをどう重み付けするか）、
    callback を呼ぶ順序と回数は既存と同じ。重みは既存の `Sequential::state_dict()` と同じキー（`0.weight` 等）・
    形状とし、同じモデルの `load_state_dict` で復元できる形（メモリ内保存・ファイル保存の両方）にする。
    callback が保存する重みは学習中の現在の重みである（ホストモデルの学習前の重みではない）。デバイス上の
    中間表現（スロット順のテンソル列・世代つきの loss ハンドル等）は出力へそのまま出さない
  - (ii) **浮動小数点値の合否**: (a) 常駐チェーンと同一デバイスの Tape 経路が両方成立する構成の経路間比較は
    bit 一致。CPU 版との比較は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を適用する。
    **baseline 方式を使えるのは、`.claude/rules/coding-rust.md`／AGENTS.md の条件（単一の連続縮約ループとは
    異なる結合順序で部分和を求める構造を持つカーネル・実機実測値のみ・人間承認）を満たす場合に限り、
    それ以外は統一複合判定を適用する**。(b) Tape 経路だけが成立する構成の比較先と判定方式は #4360（§10）で
    確定する（上記の条件を引く）。REQ-2 の判定式・tolerance は変えない
  - (iii) **離散的な結果**（`argmax` に基づく分類 metrics、監視値による `EarlyStopping`／`ModelCheckpoint`／
    `LrSchedule` の判断）: 浮動小数点値が (ii) で合格でも CPU 版と一致するとは限らない（許容誤差内の
    logits の差で `argmax` や監視値の大小が変わりうる）。CPU 版との一致を要求するか、受け入れ条件は #4360 で
    決める。本記録では一致を保証しない

これと違ってよいのは次の例外**だけ**である（例外以外の検証・位置・エラーは既存どおり。既存が転送後・実行時に
失敗しうる検査も、そのまま失敗しうる）:

- E1（GPU 固有の事前検査の位置）: デバイスの解決・対応可否、モデル・損失・optimizer の静的な未対応判定、
  検証指定時の eval 経路の可否は GPU 固有で既存に対応物がないため、引数・組合せの検証群の直後・
  `EarlyStopping` のリセットの前に置く。これだけが転送前の静的検査である。**既存と同じ検査（データセット／
  ローダーの検査・形状の適合検査・検証データの検査等）は前倒しせず、既存と同じ位置で同じエラーを返す**
- E2（失敗時の重み）: `Err` を返すときは、モデル重み・optimizer 状態とも呼び出し前の状態を残し、ベスト重みも
  適用しない（既存の `run_fit` は成功済みバッチの更新と取得済みのベスト重みを残す）。callback の副作用
  （チェックポイント・ログ・最良値・進捗）は巻き戻さない。残存したチェックポイントの扱いは呼び出し側が担い、
  `Err` 時にチェックポイントが書かれていた場合の通知方法は #4360 で定める
- E3（初期段階の `Unsupported`）: §6 の拒否一覧・`Ema`／`Swa`・`Lambda` の扱い（#4360）
- E4（書き戻し時点）: ホストモデルへの書き戻しは終了時のみ

### エラー表（(A)。新しいエラー型・variant は足さない。3 区分で variant は完全名で書く）

| 区分 | 事象 | エラー |
|---|---|---|
| 既存の処理が返すもの | 入力検証・未 compile・容量確保の失敗など、既存の `fit`／`fit_with_callbacks` が返すもの | **既存と同じ型・variant**（`AutodiffError::InvalidArgument`、`AutodiffError::Shape(..)` 等。既存を正とし、本記録は列挙しない。`BackendError::InvalidArgument` ではない） |
| GPU 固有の未対応 | 未対応の層・損失・optimizer・組合せ（§6・E1・E3） | `AutodiffError::Backend(BackendError::Unsupported(String))`（`String` に理由を入れる） |
| デバイス・バックエンド由来 | デバイス不在・範囲外 ordinal | `tape_for` の戻り値は `Result<Tape, BackendError>`。`Err(e)` を `AutodiffError::Backend(e)`（既存の `From<BackendError> for AutodiffError`）で包んで返す。`e` の variant と内容は変えない |
| 〃 | 状態の poison 等 | `AutodiffError::Backend(BackendError::StorePoisoned)`（P6 と同じ） |

## §5 公開面と既存 `train_step_resident`（P5）との関係

- P5 のシグネチャ（SGD・MSE mean 固定）は変えない。`fit_on` は P5 より上の入口である
- CrossEntropy（#4362）・Adam／AdamW（#4367）・活性化（#4370）のチェーン拡張は内部として足し、
  **公開面は `fit_on` 系だけ**を推奨する。facade から再エクスポートしない（`DeviceParamStore` は型ごと
  再エクスポートされるため、その `pub fn` を足すと公開面が増える。呼び出し口の形は §17）。P5 の別損失版など
  低レベルの公開入口を足すかは本記録では推奨せず、必要になったら別の承認事項にする。
  #4361 の承認対象は「`fit_on` 系 2 本」に絞られる

## §6 未対応の扱い

- 暗黙の CPU フォールバック（`tape_for(device)` の代わりに `crate::tape()` で回す）と、演算単位のホスト
  フォールバック（`BackendOps` 内でホストへ落とす。`train-step-resident-parts-design.md` と同じ方針）は**禁止**する
- 未対応構成は store を作る前に `Unsupported`（エラー表）で拒否する（位置は E1）。同一デバイス内で別の計算経路
  （Tape 経路）へ切り替えるかとその条件は、本記録では規範にせず §17 と #4360 に委ねる
- 初期段階で拒否する組合せ（E3）: 常駐非対応の層（#4379／#4383／#4386 で段階的に解除）・追跡対象外の
  独自層・未対応の損失と optimizer（L-BFGS を含む）・AMP・param groups・`accumulate_steps > 1`・
  `validation_split`（解除は #4360）・未対応の callback（§7）。チェーン非対応の構成（空モデル・未対応の
  活性化）の扱いも #4360 の段階分けに従う
- `FitWeights`・学習 step フック・prefetch は入口に引数がなく、構造的に受けない

## §7 callbacks・metrics・validation

- 初期段階の callback の対応（E3）: `LrSchedule`・`EarlyStopping`・`ModelCheckpoint`・`CsvLogger`／`JsonLogger`
  は対応する。`Ema`／`Swa` は `Unsupported`。`Lambda` は既存シグネチャでは重みを受けないため重み同期が
  不要で、対応するかは #4360 で決める
- validation・metrics の分割・集計・引数検証は §4 の等価規定に従う。CPU 版との比較の合否は §4 の (ii)(iii)
- `evaluate`・`predict` の GPU 版は #4377 の範囲

## §8 後続のための予約節

#4360 は本ファイルへ追記し、別ファイルを作らない。

## §9 対応範囲の段階分け（#4360 で記入）

未記入（担当 #4360）。

## §10 数値一致の判定方式（REQ-2。#4360 で記入）

未記入（担当 #4360）。

## §11 REQ-12 との整合の詳細（#4360 で記入）

未記入（担当 #4360）。

## §12 ホスト⇔デバイス転送回数の上限（#4360 で記入）

未記入（担当 #4360）。loss の読み出し方式の前提は §17 の項目 9。

## §13 承認結果（#4361 で記入）

未記入（担当 #4361）。本記録は承認を得ていない。

## §14 不変条件

- 既存の公開シグネチャ・挙動は不変（追加のみ）。`pub use` は 1 文 1 行
- tolerance・baseline・FMA 契約・f64 アキュムレータ契約・REQ-8（カーネル境界検査）は不変
- 新規 `unsafe`・依存追加なし。必要になったら停止して security-auditor の監査と所有者確認へ戻す

## §15 後続への引き継ぎ

| issue | 内容 |
|---|---|
| #4362／#4367／#4370 | 内部チェーン拡張（公開しない） |
| #4375 | 入口と型付きエラー |
| #4376 | epoch ループ・shuffle・バッチ転送・History・状態の書き戻し |
| #4377 | callbacks・evaluate・predict |
| #4378 | CPU の fit との比較テスト |
| #4392 | 実機 parity |
| #4396 | site |

公開時は `api_surface.rs` の正ガード（名前・inherent `pub fn` 集合・関数ポインタ型の署名プローブ・
利用例 doctest）と `compat-api-scope.md` §5 の適用記録を実装 issue で書く。

## §16 OWASP Top 10 観点

- A01／A04: 未対応構成は store を作る前に型付きエラーで止め、黙って CPU や別の計算へ切り替えない。
  別デバイス・別ストア・古い世代のハンドルの拒否は P1〜P6 の検査（`store_id`・`device`・`generation`）を流用する
- A03: 転送前に行う検査は GPU 固有の静的な事前検査（§4 の E1）だけである。それ以外の検査
  （`checked_*` による算術オーバーフロー防止を含む）は既存の `fit` と同じ位置で行い、既存と同じく
  転送後・実行時に失敗しうる（前倒しはしない）。カーネル側の手動境界チェックは省略しない（REQ-8）
- A05／A06: 依存追加なし・新規 `unsafe` なし
- A08: 学習結果の合否は §4 の (ii)(iii) に従い、tolerance・baseline は変えない
- A09: 同期・転送回数は診断カウンタで実機確認する受け入れ条件を #4360・#4392 に残す

## §17 実装 issue への申し送り（規範ではない）

内部実装の契約。各項目は「事実（既存の関数と契約）＋ 実装で守る条件 ＋ 担当」。本記録の規範（§4 の (A)(B)）と
食い違う場合は §4 を正とする。

1. **公開面を増やさない呼び出し口**: 事実: `DeviceParamStore::train_step_device_chain` は `autodiff` の `pub`
   メソッドで、別クレートの `facade` から呼ぶには `pub(crate)` では足りない。条件: 既存シグネチャを維持して
   既存口から拡張へ委譲できるか、内部口が要るかを確かめる。口は `DeviceParamStore` の `pub fn`
   （`#[doc(hidden)]` を含む）にせず、`optim::resident_param_groups` と同様に facade が再エクスポートしない
   モジュールの自由関数にする。担当: #4362・#4361
2. **事前の経路選択と実行時の再試行（Tape 経路）**: 事実: P5 の契約（`compat-resident-train-step-decision.md`
   §5 第 2 段・「失敗の境界」）は、store に何も書いていない段階で部品が `Unsupported` を返したときに限り
   同じ step を Tape 経路へ落とせる。それ以外の `Err` は poison もありうる。条件: 事前の静的な経路選択と、
   実行時の再試行を区別する。再試行は事前検証を通過し、チェーンが書き込み前に store を変えず `Unsupported` を
   返した場合に限る。入力エラー・実行時エラーは store が未変更でもそのまま返す。Tape 経路の許可と範囲は
   #4360 が決める。担当: #4360・#4376
3. **検証用 Tape 経路は読み取り専用**: 事実: `forward_resident` は `register_resident_params` で pending
   forward を登録する（学習用）。`predict_resident` は `snapshot_resident_params`（`pending` を変えない読み取り専用）を使う。
   条件: 検証用の Tape 経路は後者相当で現在のパラメータを読み、pending や世代を進めない。担当: #4377
4. **eval 経路の事前判定**: 事実: `predict_resident_input` は `build_device_chain_steps` でチェーンを構築できないと
   `Unsupported` を返す。条件: E1 の位置で、デバイス能力（`MemoryOps` 等）・モデル構成（空モデル・Sigmoid 等の
   `build_device_chain_steps` と同じ条件。`contains_resident_unsupported_layer` だけでは不足）を静的に判定する。
   担当: #4375
5. **store の生成と書き戻し**: 事実: `tape_for(device)` → `init_device_param_store`、終了時は
   `Tape::sync_device_param_store_to_host`（`Vec<Tensor<f32>>`）。条件: 取得後に `apply_parameters` 相当で層へ
   反映する。スロット列から `state_dict()` のキー（`0.weight` 等）への対応付けは実装側の責務。担当: #4376
6. **失敗時の巻き戻し（E2）**: 条件: 呼び出し前の重みと optimizer 状態を保持し、`Err` 時に戻す。ベスト重みは
   正常終了時のみ、最終書き戻しの後に適用する（逆順にしない）。担当: #4376・#4377
7. **optimizer 状態の持ち越し**: 事実: `fit_twice_equals_fit_once_with_double_epochs` は `shuffle=false`・
   callback なしの条件で成立する。条件: 同値性の主張はその範囲に限る。Adam 系の初回生成分も書き戻せることを
   確かめ、戻せなければ `Unsupported`。担当: #4367・#4376
8. **callback の D2H**: 事実（`training.rs` の呼び出し位置）: `ema_update_after_step` は step 後、
   `swa_update_after_epoch` は epoch 末、`ModelCheckpoint::observe`・`EarlyStopping::observe` は epoch 末の
   callback ループ。条件: `EarlyStopping`（`restore_best_weights` 時は改善時）・`ModelCheckpoint`（保存の直前）の
   スナップショットは現在の store から取得した重みを使い、ホストモデルの `state_dict()` を使わない。
   取得（`sync_device_param_store_to_host`）は失敗しうるため、改善判定 → 取得 → 成功後に最良値とスナップショットを
   確定の順にし、失敗は `Err` として伝播する（既存 `observe` のクロージャは失敗を返せない。拡張か薄い写しかは
   実装で確かめる）。担当: #4377
9. **loss の読み出し**: 事実: `read_train_step_loss` は古い世代のハンドルを `InvalidArgument` で拒否する
   （`train_chain.rs` の世代検査）。条件: `History.loss` は既存の集計手順（各バッチの損失をサンプル数で重み付け）と
   同じ（§4 の (i)）。epoch 単位で転送するなら、各 step の損失を上書き前に保持する等の仕組みが要る
   （`ResidentLoss::read` を遅らせるだけでは成立しない）。方式は #4360（§12）。担当: #4360・#4376
10. **Tape 経路だけが成立する構成の比較先**: 条件: §4 の (ii)(b)。確定するまで当該構成は `Unsupported`。担当: #4360（§9・§10）
11. **学習率の反映**: 条件: 常駐チェーン・Tape 経路のどちらでも、epoch 開始時の学習率をその epoch の更新
    （`step_device_param_store*` を含む）へ渡し、終了時に optimizer へ残るのは §4 の規定どおり。担当: #4367・#4376
