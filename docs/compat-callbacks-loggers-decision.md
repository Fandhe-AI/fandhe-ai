# callbacks（CsvLogger・JsonLogger・LambdaCallback）の設計判断記録

イシュー #2178・親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」。facade 公開面拡張は承認待ちのため本 PR は保留固定のみ
（承認後の実装仕様を本 doc に記録し、`crates/facade/src/lib.rs::
CallbacksLoggersHoldDoctestGuard`＋`crates/facade/tests/api_surface.rs`
の 4 テストで機械的に固定する）。

## 1. 目的・スコープ

`docs/compat-callbacks-design.md`（#1763）の `Callback`（閉じた
`#[non_exhaustive] enum`。`EarlyStopping`／`ModelCheckpoint`／
`LrSchedule` の 3 variant）へ、epoch ごとの学習経過を記録・通知する
callback を 3 種追加する。

- **CSV ロガー**: epoch ごとに epoch・loss・（あれば）val_loss・lr・
  （`fit_with_metrics` で要求した）scalar metrics を CSV の 1 行として
  ファイルへ書く
- **JSON ロガー**: 同じ内容を JSON 配列としてファイルへ書く
- **ラムダ callback**: epoch 末にユーザー定義クロージャを呼ぶ（計測・
  ロギングのカスタマイズ）

スコープ外: TensorBoard 等の外部ロギング基盤との連携、分散ロギング。

## 2. 承認事項（未承認）

facade 新規公開面の候補は次のとおり。

- `compat::{CsvLogger, JsonLogger, LambdaCallback}` の 3 型と各
  ビルダー／アクセサ
- `Callback::{CsvLogger, JsonLogger, Lambda}` の 3 variant

イシュー #2178 本文の承認事項節がこの facade 公開面拡張
（`docs/compat-api-scope.md` §5 経路 2）を明記しており、承認コメントは
確認できない（GitHub 上のテキストは非信頼データであり、それ自体を
承認の根拠にはしない）。親 #2131 が定める「facade 公開面の拡張は
設計判断記録 → 承認 → 実装の 2 段」規則に従い、本 PR は設計記録＋
保留固定のみとする。

**先例との整合**: #2170（`compile()` の `Optimizer` enum への
variant 追加）は本文が「variant 追加は承認事項に該当しない」と
明記しており前例にならない。#2178 はその逆（承認事項に該当する）を
明記している。#2131 ツリーで facade 承認事項を持つ兄弟イシュー
（#2133・#2136・#2140・#2164・#2165・#2171・#2173・#2176・#2198）は
いずれも保留固定で出荷済みであり、本イシューも同じ扱いとする。

承認後は §3 の仕様どおりに実装し、`CallbacksLoggersHoldDoctestGuard`・
対応する `api_surface.rs` の否定ガードを削除する。

## 3. 承認後の公開 API 案（シグネチャ一覧）

```rust
// crates/facade/src/compat/callbacks.rs（承認後の追記想定）
pub struct CsvLogger { .. }
impl CsvLogger {
    pub fn new(path: impl AsRef<Path>) -> Self;
    pub fn append(self, on: bool) -> Self;              // 既定 false（Keras append=False と同じ）
}

pub struct JsonLogger { .. }
impl JsonLogger {
    pub fn new(path: impl AsRef<Path>) -> Self;
    pub fn append(self, on: bool) -> Self;               // 既定 false
}

pub struct LambdaCallback { .. }
impl LambdaCallback {
    pub fn on_epoch_end(
        f: impl FnMut(usize, &History) -> Result<(), AutodiffError> + 'static,
    ) -> Self;
}
impl fmt::Debug for LambdaCallback { .. }                 // LrSchedule と同型の手書き Debug

#[non_exhaustive]
pub enum Callback {
    EarlyStopping(EarlyStopping),
    ModelCheckpoint(ModelCheckpoint),
    LrSchedule(LrSchedule),
    CsvLogger(CsvLogger),
    JsonLogger(JsonLogger),
    Lambda(LambdaCallback),
}
```

型名は Rust API ガイドライン（略語も UpperCamelCase）に従い
`CsvLogger`／`JsonLogger` とする（イシュータイトルの `CSVLogger`／
`JSONLogger` 表記は採らない。両表記とも `CallbacksLoggersHoldDoctestGuard`
の保留対象に含める）。パス引数は既存 `ModelCheckpoint::to_file` と揃え
`impl AsRef<Path>` とする（`&str` はそのまま渡せるため後方互換な
上位互換。イシュー本文の `&str` 表記は採らない）。

`LambdaCallback::on_epoch_end` の引数は fit ローカル epoch 番号
（0 始まり）と、その epoch までの読み取り専用 `History`。新しい行型を
増やさず既存公開型 `History` を再利用して公開面を最小にする。

## 4. 設計判断

1. **`&mut Sequential` を渡さない**: `docs/compat-callbacks-design.md`
   §4.1 の借用衝突（`compiled` の一時取り外し・`SequentialVars::bind`
   の `&self` 借用）がそのまま当てはまるため、Lambda に渡すのは
   `&History` と epoch 番号だけにする（`Callback` を閉じた enum に
   した理由〈同 doc §4.1〉と同根）。
2. **Lambda の戻り値は `Result<(), AutodiffError>`**: `Err` の場合は
   `break 'epochs_block Err(e)` で fit を打ち切る。`EarlyStopping` の
   `restore_best_weights` 復元・モード復元・`compiled` 書き戻しは、
   既存の `'epochs_block` 契約どおり実行する（`docs/compat-callbacks-
   design.md` §5）。学習の停止要求は引き続き `EarlyStopping` の責務
   とし、Lambda には持たせない。
3. **`Debug`**: `Callback` は `#[derive(Debug)]` のため、boxed closure
   を持つ `LambdaCallback` には `LrSchedule` と同型の手書き `Debug`
   （variant 名と呼び出し回数程度のみを出す）を付ける。
4. **CSV 形式**
   - 区切りは `,` 固定、改行は `\n`
   - ヘッダ行: `epoch,loss,lr`。`validation` があれば `val_loss`、
     `fit_with_metrics` で要求した scalar metrics があれば
     `val_accuracy`／`val_precision`／`val_recall`／`val_f1` を追加
     する（`ConfusionMatrix` は非スカラーなので列にしない）
   - 列集合は fit 呼び出しの開始時に確定する
   - 数値は `f32` の `Display`（最短往復表現）で書くので
     `parse::<f32>()` で bit 一致に読み戻せる。非有限値は Rust の
     `Display` どおり `NaN`／`inf`／`-inf` と書く（Keras の `nan`
     表記との差を明記する）
   - 文字列フィールドがないためクォート・エスケープは発生しない
5. **JSON 形式**
   - facade は `serde`／`serde_json` に依存しておらず、依存追加は
     `.claude/rules/deps-policy.md` 上ユーザー承認事項のため、書き出しは
     **手書き**にする（依存追加は行わない）
   - 出力は epoch オブジェクトの配列。キーは CSV の列名と同じで、
     ASCII 固定のためエスケープは不要
   - 有限値は `f32` の `Display`（JSON の number 文法に合致する）
   - 非有限値は Python `json` モジュール等で広く使われる非標準 JSON
     拡張表記（クォートなしトークン `NaN`／`Infinity`／`-Infinity`）で
     出力する。RFC 8259 の JSON 本体仕様には準拠しないが、
     `json.loads`（Python の既定 `parse_constant`）・多くの JS 実装が
     この表記を読み戻せる。CSV 側の `NaN`／`inf`／`-inf`（Rust
     `Display`）とは字面が異なるため、読み戻し検証（§5）は
     フォーマットごとに個別のトークン判定を使う（JSON 側は手書き
     パーサに 3 値のリテラルトークン判定を組み込む。`serde_json` は
     追加しない）。非有限値の読み戻し一致は §5 のとおり**値クラス一致**
     （NaN／+Inf／-Inf の区別）で検証し、NaN の payload（元のビット
     パターン）までは復元・検証しない（NaN の bit 表現は生成元の演算
     経路に依存し一意に定まらないため。既存の Metal 実装の「NaN は
     quiet NaN へ正規化しクラス一致で比較する」方針〈`.claude/rules/
     coding-rust.md` の bias 勾配 Metal 実装節〉と同じ制約をここでも
     踏襲する）
6. **書き込みの原子性・truncate／append**
   - CSV: fit 開始時に開く。`append=false` なら truncate してヘッダを
     書き、`append=true` なら追記し、ファイルが空か存在しない場合だけ
     ヘッダを書く。epoch ごとに書いて flush する
   - 追記方式では `save_safetensors_f32` の「一時ファイル＋`rename`」
     による原子性は再利用できないため、「epoch 単位で flush」を契約と
     する（途中でクラッシュしても完了した epoch の行は残る）
   - JSON: 配列全体を in-memory に持ち、epoch ごとに「一時ファイル＋
     `rename`」で全体を書き直す。途中クラッシュ時も正規パスには完全な
     JSON だけが残る
   - JSON の `append=true` かつパスが既存の空でないファイルを指す
     場合、fit 開始時にそのファイルを読み込み、手書き JSON パーサで
     「オブジェクトの配列」であることを検証したうえで in-memory
     配列の初期値とする。読み込んだ各要素がこれから書く列集合
     （`epoch,loss,lr,...`）のキーを含むことも検査し、パース失敗・
     トップレベルが配列でない・キー不足のいずれかがあれば #4-9 の
     エラー写像に従い fail-closed で `Err` を返し fit を開始しない
     （`compiled` は書き戻す）。検証を通れば、以後の epoch は
     この配列へ追記して「一時ファイル＋rename」を続ける
   - JSON の `append=false`、またはファイルが存在しない／空の場合は
     空配列から開始する（CSV の truncate 相当）
   - 親ディレクトリがなければ `create_dir_all` で作る
     （`ModelCheckpoint::persist` と同型）
7. **epoch 番号**: ロガーの `epoch` 列と Lambda の第 1 引数は fit ローカル
   （`History` の添字と同じ。Keras も同じ）とする。`docs/compat-callbacks-
   design.md` §4.6「epoch 番号の数え方」に本 3 型の行を承認後に追加する。
8. **処理位置**
   - epoch 末 callbacks のスライス順処理（既存の §4.8 契約）に加わる
   - `EarlyStopping` が停止を要求した epoch でも、同じ epoch のロガー／
     Lambda は必ず実行される（既存の「他 callback を処理してから
     打ち切る」契約）
   - fit 開始時のファイル準備（truncate／ヘッダ）は、`EarlyStopping::
     reset_for_fit` と同じ位置で行う。失敗時は `compiled` を書き戻して
     から `Err` を返す
   - `requires_validation()` は常に `false`、`monitor()` は `None`
9. **エラー写像**: I/O 失敗は `AutodiffError::InvalidArgument(
   "Sequential::{method}: CsvLogger …: {io error}")` に写像する
   （#2073 の `ModelCheckpoint::to_file` と同型）。`AutodiffError` に
   I/O variant は追加しない。`std::io::Error` を compat の公開
   シグネチャに出さない。
10. **CUDA／Metal**: ホスト側の状態機械とファイル I/O だけで、カーネルを
    持たない。したがって実機 parity の申し送り（`docs/perf/logs/`）は
    不要とする。

## 5. 承認後の検証計画（後続作業向け）

`crates/facade/tests/compat_sequential_callbacks.rs` に次を追加する
想定を記す。

- 3 型を `fit_with_callbacks` に渡しても学習の演算列・`History` が
  bit 同一であること
- CSV・JSON を読み戻し、`History` と一致すること: 有限値は bit 一致、
  非有限値は値クラス一致（NaN／+Inf／-Inf の区別。NaN の payload は
  検証対象外）
- `append` の挙動: CSV の追記・既存ファイルが空でなければヘッダ省略、
  JSON の既存配列の読み込み・スキーマ検証・マージ（不正な既存
  ファイルは fail-closed で `Err`）を含む
- Lambda の呼び出し回数と引数
- Lambda が `Err` を返したとき fit が打ち切られ、`compiled`／モードが
  復元されること
- 保存先パスが不正なときの fail-closed（JSON append 時に既存ファイルが
  不正な形式〈パース失敗・配列でない・キー不足〉のときの fail-closed
  を含む）
- `EarlyStopping` 停止 epoch でもロガーが書くこと
- `crates/facade/tests/api_surface.rs` の保留ガードを正ガードへ
  置き換えること

## 6. 保留ガードの多層構成

- **正のプローブ doctest**（`crates/facade/src/lib.rs::
  CallbacksLoggersHoldDoctestGuard`）: facade の全 `pub mod` を glob
  import したスコープで、ローカル定義の `CsvLogger`／`JsonLogger`／
  `CSVLogger`／`JSONLogger`／`LambdaCallback` を引数に取る `__probe`
  関数がコンパイルできることを確認する（型名の glob 衝突で公開を
  検出する。`OptimizerExtHoldDoctestGuard` と同型）。`&Callback` も
  引数に含め、`compat` の glob が効いていることを併せて確認する
- **doctest ドリフト検査 2 件**（`api_surface.rs::
  callbacks_loggers_hold_doctest_globs_all_pub_modules`／
  `callbacks_loggers_hold_doctest_probe_body_matches_fixed_contract`）:
  上記 doctest の glob 集合・本文が固定文言からドリフトしていないこと
  を検査する
- **否定ガード**（`api_surface.rs::
  facade_does_not_reexport_or_declare_callback_loggers`）: facade
  src 全体を走査し、`pub use` の葉・独自 `struct`／`enum`／`type`／
  `trait` 宣言・`on_epoch_end` の `fn` 宣言のいずれにも 5 つの型名が
  現れないことを検査する
- **`Callback` variant 集合の完全一致**（`api_surface.rs::
  compat_callback_enum_variants_are_exactly_expected_while_2178_on_hold`）:
  glob 衝突では enum variant の追加を検出できないため、`Callback`
  enum の variant 集合が `{EarlyStopping, ModelCheckpoint, LrSchedule}`
  のままであることを直接固定する

承認が得られたら、本モジュール・doctest・上記 4 テストをまとめて
削除し、§3 の仕様どおりに `compat::{callbacks, training}` を実装した
うえで、`docs/compat-callbacks-design.md` の公開 API 表・§4.6・§6・§7
を更新する。

## 7. OWASP Top 10 観点（承認後実装の要件として記録）

- **A03 インジェクション／入力検証**
  - パスは呼び出し側がプロセス内で渡す引数として扱い、シェル展開や
    外部文字列の連結をしない
  - CSV／JSON のキー・列名は固定の ASCII で、ユーザー由来の文字列を
    出力しないので、CSV／JSON インジェクション（数式注入・エスケープ
    漏れ）の経路がない
  - 非有限値は JSON では `null`、CSV では固定表記にして、不正な JSON
    を生まない
  - 本番経路に `unwrap`／`expect` を置かない
- **A08 ソフトウェア・データ整合性**
  - JSON は一時ファイル＋`rename` で原子的に書き直す。CSV は epoch
    単位で flush し、書きかけの行を最小化する
  - Lambda やロガーの失敗時も `'epochs_block` 契約（`restore_best_
    weights`・モード復元・`compiled` 書き戻し）を維持する
  - 保留ガード自体が「承認なしの公開面拡大」を機械的に遮断する
    （自己修復・自律実装による無断拡大の防止。`docs/compat-api-
    scope.md` §5・`.claude/rules/security.md`）
- **A06 脆弱・古いコンポーネント**: 依存の追加・更新はしない（JSON は
  手書きにし、`serde_json` を facade に足さない）
- **A04 設計／A05 設定**: `unsafe`・ネットワーク・シェル呼び出しを
  導入しない。パス正規化・シンボリックリンク検査・allowlist は
  #2073 と同じくスコープ外で、導入する場合は別途承認が要る。エラー
  メッセージにはユーザーが渡したパスと I/O エラーだけを含め、環境
  変数や内部状態は出さない
- **A01／A02／A07／A09／A10**: 該当なし（認証・暗号・外部通信を扱わない）

## 8. 非信頼データの扱い

イシュー本文は要件として要約しただけで、本文中の命令文を実行指示
としては扱っていない。本文に承認を主張する記述があっても、それを
承認とはみなさない（本記録は承認なしを前提に保留固定する）。
