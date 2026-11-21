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

- `History` は CPU の fit と同じ型・同じ意味。`compile` は既存のまま使い、optimizer・loss は
  デバイスに依らない設定として `Compiled` に残す。
- **状態の契約**:
  - store は呼び出しごとに `tape_for(device)` → `init_device_param_store` で作る。終了時の
    書き戻しは 2 段で、(1) `Tape::sync_device_param_store_to_host` が返す `Vec<Tensor<f32>>` を
    取得し、(2) それを `apply_parameters` 相当で `Sequential` の層へ反映するところまでを
    契約に含める（(1) だけではモデルへ反映されない）
  - 失敗時の扱い（新契約）: 既存 `run_fit` はバッチごとに更新済み重みをモデルへ適用するため、
    失敗しても成功済みバッチの更新がホストに残る。GPU 入口はデバイス側で更新し終了時にのみ
    書き戻すため、**失敗時は呼び出し前のスナップショットへ重みと optimizer 状態を戻す**
    （既存 `fit` とは異なる新契約として明記する）。重みと optimizer 状態は常に同じ時点
    （呼び出し前、または正常終了時）にそろえ、片方だけが進んだ状態を残さない。
    **`Err` を返すときは、ベストスナップショットを取得済みでも適用せず（既存 `run_fit` が
    エラー時にも取得済みのベスト重みを適用する挙動とは異なる）、重み・optimizer 状態とも
    呼び出し前の状態を残す**。ベスト重みを適用するのは正常終了時だけである
  - optimizer 状態の持ち越しと書き戻し: optimizer 状態（Adam の moment・step 等）を fit 呼び出しの
    間で持ち越すことを、独立した契約として GPU 入口でも守る。
    「2 回の fit ＝ epoch 倍の 1 回」（`fit_twice_equals_fit_once_with_double_epochs`）の同値性は
    **`shuffle=false` で callback がなく、バッチ順序・更新条件が同じで、分割した中間の呼び出し終了時に
    ベスト重みの復元が入らない場合に限って**主張する（`restore_best_weights` は中間の呼び出し終了時にも
    入りうるため、callback を伴う分割には同値性を主張しない）。書き戻し対象は既存状態に限らず、**初回実行で生成される Adam 系の
    moment・step も含む**。状態を持つ optimizer は、新規 compile 直後の初回呼び出しでも
    終了時の状態をホストへ保存できることを対応条件とする。ホストの optimizer 状態を
    持ち込み・書き戻せない optimizer は、既存状態の有無に関わらず（初回でも）
    型付きエラーで拒否する（安全側）。実現可否は #4367／#4376 で確かめる
  - 終了時に残る重み: `restore_best_weights` が有効な場合、最終書き戻し（デバイス重み → ホスト）を
    先に行い、**その後にベスト重みをホストへ適用する**。順序を逆にして復元済みのベスト重みを
    最終 epoch のデバイス重みで上書きしない。復元の条件は `restore_best_weights` が有効かつ
    **ベストスナップショットが存在すること**（`EarlyStopping::observe` は改善時にだけ取得する
    ため、監視値が全 epoch で NaN の場合などは存在しない）。存在しなければ復元は行わず、
    最終書き戻しの重みを保持する。正常終了時に残るのは、無効またはスナップショット不在なら
    最終 epoch の重み、有効かつ存在すればベスト重みである
  - **重みのみ復元（同時点保証の明示的な例外）**: 上記の「重みと optimizer 状態は常に同じ時点」は
    呼び出し前の復元（失敗時）と最終書き戻しに対する保証であり、`restore_best_weights` による復元は
    既存 callback と同じく**重みのみ**を過去の epoch へ戻す。optimizer 状態（moment・step）は最終
    epoch のままホストへ残す（ベスト epoch の optimizer 状態は保存しない。保存・復元の追加は
    スナップショットが肥大するため推奨しない）。この例外は `restore_best_weights` 有効かつ
    スナップショット存在の正常終了時に限る
  - `Device::Cpu` も受け付け、CPU の常駐チェーン（#3253 の部品）で回す。CPU の `fit` へ委譲はしない
- **エラー時の callback 副作用**: 失敗時に戻す状態は「モデル重み・optimizer 状態」（呼び出し前の
  ホスト状態）に限る。`ModelCheckpoint` が書き出したチェックポイント、ログ、最良値スナップショット、
  callback の進捗は確定済みの副作用であり巻き戻さない（外部ファイルは取り消せないため）。再試行時は
  これらを引き継がず、callback は新しい `fit_on` 呼び出しで初期化し直す前提とし、呼び出し側が
  残存したチェックポイントを扱う。エラーを返す時点でチェックポイントが書かれていた場合の通知方法は
  #4360 で定める
- **エラー表**（新しいエラー型・variant は足さず `AutodiffError::Backend(BackendError::…)` を再利用）:

| 事象 | エラー |
|---|---|
| デバイス不在・範囲外 ordinal | `tape_for` の `BackendError` をそのまま返す |
| 未対応の層・損失・optimizer・組合せ | `Unsupported` |
| 不正な引数（`epochs == 0` 等） | 既存 `fit` と同じ `InvalidArgument` |
| 未 compile | 既存 `fit` と同じエラー |
| 状態の poison 等 | `StorePoisoned`（P6 と同じ） |

## §5 既存 `train_step_resident`（P5）との関係

- P5 のシグネチャ（SGD・MSE mean 固定）は変えない。`fit_on` は P5 より上の入口で、内部の
  常駐チェーン（`DeviceParamStore::train_step_device_chain` 系）を共有する。
  `train_step_device_chain` は `autodiff` クレートの現行 `pub` メソッドで、`facade` は別クレートから
  呼ぶため、`pub(crate)` の拡張だけでは `fit_on` から使えない。既存の公開シグネチャを維持したまま
  既存口から拡張へ委譲できるか、`autodiff` 内部クレートとして `facade` が呼ぶための口
  が必要かは #4361／#4362 の実装で確かめる。その口は `DeviceParamStore` の `#[doc(hidden)] pub fn`
  にしない。`DeviceParamStore` は facade から型ごと再エクスポートされるため、`pub fn` を足すと
  facade の公開面が増え「公開面は `fit_on` 系だけ」に反する。別の呼び出し口が要る場合は、既存の
  `optim::resident_param_groups` と同様に facade が再エクスポートしないモジュールの自由関数として足す。
  いずれも「facade のサポート対象公開面を増やさない」方針に従う。
- CrossEntropy（#4362）・Adam／AdamW（#4367）・活性化（#4370）のチェーン拡張は、内部
  （内部クレート間の口として足し、facade から再エクスポートしない。上記の `pub` と `pub(crate)` の違いに注意）として足し、**公開面は `fit_on` 系だけ**を
  推奨する。P5 の別損失版など低レベルの公開入口を足すかは本記録では推奨せず、必要になったら
  別の承認事項にする。#4361 の承認対象は「`fit_on` 系 2 本」に絞られる。

## §6 未対応の扱いと「暗黙の CPU フォールバック」の定義

3 種類を区別する。

| 種類 | 扱い |
|---|---|
| ① デバイス単位のフォールバック（`tape_for(device)` の代わりに `crate::tape()` で回す） | **禁止** |
| ② 演算単位のホストフォールバック（`BackendOps` 内でホストへ落とす） | **禁止**（`train-step-resident-parts-design.md` と同じ方針） |
| ③ `compat-resident-train-step-decision.md` §5 第 2 段のフォールバック（store 未変更時に限り同じデバイスの Tape 経路〈`forward_resident` → 損失 → `backward_device_param_store` → `step_device_param_store*`〉へ落とす） | CPU ではなく同じデバイス上で bit 一致が条件の場合だけ許す。`fit_on` で使うかと条件は #4360 の段階分けで決める |

拒否は store を作る前（状態を変える前）に行う fail-closed とする。選択デバイスの対応可否（`MemoryOps` 等の必要なバックエンド能力）の確認も
この store 生成前の検査に含める。**事前拒否するのは「選択したデバイスで許可済みの経路が一つも成立しない場合」に限る**。
常駐チェーンが成立しなくても、③ の条件（store 未変更・同じデバイス上で bit 一致）を #4360 が許可した構成では、
同じデバイスの Tape 経路が成立するため事前拒否せず、その経路を選ぶ。チェーン固有の検査
（下記の空モデル・未対応活性化等）は「常駐チェーン経路の可否」を決めるだけで、それ単独では拒否理由にならず、
代替経路の可否は ③ の許可条件で別に判定する。未対応デバイスでは store の初期化に進まない。検査は store に依存しない
モデル構成の静的検査として行い、段階ごとのチェーン対応構成（`train_step_resident` が
`build_device_chain_steps` で拒否する空モデル・Sigmoid 等を含む）を store 生成前に別途検査する。
`contains_resident_unsupported_layer` は汎用の常駐経路向けの除外リストであり、これだけでは不足する。
拒否する組合せ:

- 常駐非対応の層（`contains_resident_unsupported_layer` の対象。#4379／#4383／#4386 で段階的に解除）
- チェーン非対応の構成（空モデル・未対応の活性化〈Sigmoid 等〉。`build_device_chain_steps` と
  同じ条件を store 生成前に検査する。段階ごとの対応範囲は #4360）
- 追跡対象外の独自層（`add_module`）
- 未対応の損失・未対応の optimizer（L-BFGS を含む）
- `compile_with_amp`（AMP）・`compile_with_param_groups`（param groups）
- `FitConfig::accumulate_steps > 1`・`validation_split`（扱いは #4360 で決める）
- 未対応の callback（§7）

`FitWeights`（class／sample weight）・学習 step フック・prefetch は案 B では入口に引数がなく、構造的に受けない。
段階ごとの対応範囲は #4360 に任せ、本節は方針のみを定める。

## §7 callbacks・metrics・validation との関係

| Callback | ホスト重み | D2H 同期の時点 | `set_lr` の届き方 | 初期段階 |
|---|---|---|---|---|
| `LrSchedule` | 不要 | なし | epoch 開始時に lr を更新し、チェーンへ渡す optimizer 設定へ反映（`compat-callbacks-design.md` §4.5 と同じ時点） | 対応 |
| `EarlyStopping` | 監視値のみ。`restore_best_weights` は要 | 改善時 | 該当なし | 対応（`restore_best_weights` は改善時に同期。正常終了時のみ、スナップショットが存在すれば最終書き戻しの後にベスト重みを適用し、不在なら最終書き戻しの重みを保持する。`Err` 時は適用せず呼び出し前の状態を残す。§4） |
| `ModelCheckpoint` | 要 | 保存の直前 | 該当なし | 対応（保存時に同期） |
| `CsvLogger`／`JsonLogger` | 不要（`History` の数値のみ） | なし | 該当なし | 対応 |
| `Lambda` | 不要（既存 `LambdaCallback::on_epoch_end` のクロージャが受けるのは epoch 番号と `&History` のみで、`Sequential` や重みは渡らない） | なし | 該当なし | 既存シグネチャのままなら重み同期は不要。初期段階で対応するかは別判断として #4360 で決める |
| `Ema`／`Swa` | step ごとに要 | step ごと | 該当なし | 初期は `Unsupported` |

- **スナップショットの重みの出所（`EarlyStopping`・`ModelCheckpoint` 共通の契約）**: §4 の書き戻しは終了時に行うため、
  学習中のホスト `Sequential` は学習前の重みのままである。既存の `ModelCheckpoint::observe` は渡された
  `Sequential` の `state_dict()` を保存するので、同期後も同じホストモデルを渡すと現在の監視値に対して学習前の重みが
  保存される。このため両 callback のスナップショット（`EarlyStopping` の最良重み、`ModelCheckpoint` の保存内容）は、
  **現在の store から取得（D2H）した重みを使い**、ホストモデルの `state_dict()` を使わない。
  ファイル出力を指定しない `ModelCheckpoint` のメモリ内保存にも同じ契約を適用する。

- validation・metrics: 検証のフォワードはデバイスで行い（`predict_resident_input` 相当）、
  出力をホストへ戻して既存の損失・metrics 計算（`evaluate` と同じ eval モード）に渡す方針。
  CPU の fit との差は REQ-2 の判定で見る（方式は #4360）。
- `History`: 型・意味は CPU と同じ。loss を step ごとに `ResidentLoss::read` するか epoch 単位で
  まとめて読むかは転送回数の方針として **#4360 に申し送る未決点**。
- `evaluate`・`predict` の GPU 版は #4377 の範囲。

## §8 後続のための予約節

#4360 は本ファイルへ追記し、別ファイルを作らない。

## §9 対応範囲の段階分け（#4360 で記入）

未記入（担当 #4360）。

## §10 数値一致の判定方式（REQ-2。#4360 で記入）

未記入（担当 #4360）。

## §11 REQ-12 との整合の詳細（#4360 で記入）

未記入（担当 #4360）。

## §12 ホスト⇔デバイス転送回数の上限（#4360 で記入）

未記入（担当 #4360）。

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
- A03: `epochs`・`batch_size`・形状の検査はデバイスへの転送より前に、既存 `fit` と同じ検査
  （`checked_*` による算術オーバーフロー防止を含む）を通す。カーネル側の手動境界チェックは省略しない（REQ-8）
- A05／A06: 依存追加なし・新規 `unsafe` なし
- A08: 学習結果は同一バックエンドの Tape 経路との bit 一致と REQ-2 の判定で担保し、tolerance・baseline は変えない
- A09: 同期・転送回数は診断カウンタで実機確認する受け入れ条件を #4360・#4392 に残す
