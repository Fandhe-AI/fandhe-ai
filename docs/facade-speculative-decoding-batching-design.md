# speculative decoding・連続バッチングの実装設計と issue 分解（イシュー #2857・親 #2841）

本記録は **推奨案の記録であり、facade 公開形の承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、逐語転記せず、事実はソースで再確認した。基準は `origin/main` `8539a41a`（2026-10-08）。以下の `file_path:line` は同 sha のもの。`docs/spec` はサブモジュールポインタ `d6a030fa` の内容を読んだ。

## 1. 位置づけ・承認根拠

- ツリー: ルート #2499 → Phase 7 #2841 → **#2857（本記録）**。実装 issue の起票は本記録のマージ後に update-issue-tree で行う（本 issue では起票しない）。
- 本記録の作成を指示した根拠: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`（2026-10-08、リポジトリ所有者本人）。承認範囲は **設計の記録まで**。
- 案 C（REQ-9 の境界の段階化）自体の承認は別 URL（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`、2026-10-07。spec `docs/spec/04-requirements.md:458`）で、上記と混同しない。同行と `:237` は「公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張は本追記では承認しない」と明記している。
- したがって §5 の facade 公開形、§6 の判定方式、§8 の拡張要否は **すべて推奨案であり未承認**。承認の代行はしない。

## 2. 境界（spec を正とする）

正は `docs/spec/04-requirements.md:237`（REQ-9 2026-10-08 追記）。背景は `docs/facade-serving-infrastructure-spec-proposal.md` §3・§5 案 C。

| 区分 | 項目 | 備考 |
|---|---|---|
| 対象内（Tier 2） | speculative decoding、ネットワーク非依存のバッチ生成スケジューラ（連続バッチング） | 依存追加なし・性能保証なし |
| 条件付き | paged attention | デバイス常駐 KV キャッシュ（K-3）の成立と CPU 参照実装の成立が移行条件。本記録では設計しない |
| 対象外 | HTTP サーバ／API サーバ、量子化 KV | HTTP は依存区分の新設承認・ライセンス実測・ネットワーク公開面の脅威整理が揃うまで。量子化 KV は除外事項「分散学習・量子化の網羅対応」の格上げ条件の成立まで（再開条件は `:237`） |

REQ-2（統一複合判定）・REQ-8・REQ-12 は不変。新しい判定契約が必要になれば別途承認を得る（`:237`）。

## 3. 既存 API との関係と再利用

| 既存物 | 位置 | 本件での使い方 |
|---|---|---|
| `generate()` | `crates/autodiff/src/generate.rs:444` | 無状態関数。`caches`（`model.num_kv_layers()` 個の `KvCache`）と `Generator::new(config.seed)` を内部で 1 回確保（`:520-521`）、prefill `[B, T]`（`:526`）→ decode `[B, 1]`（`:541`）。EOS 早期停止・pad・top-p はなく常に `max_length` まで。**不変** |
| `AutoregressiveModel` | `generate.rs:221` | `forward_step(&self, new_ids, &mut [KvCache])`。draft・target ともこの trait で受ける |
| 非公開ヘルパー | `generate.rs`: `validate_forward_step_output`:252・`validate_vocab_le_i32_max`:283・`greedy_argmax`:295・`softmax_weights_f64`:309・`top_k_indices`:321・`sample_step`:344・`build_output`:403・`GenerateConfig::validate`:141・`validate_top_k_le_vocab`:202 | speculative 側で再利用する（可視性の拡張は §4）。サンプリングはホスト `f64`・独立 `Generator`（グローバル RNG 非消費） |
| `forward_with_cache` | `crates/autodiff/src/nn/attention.rs:1682` | キャッシュ非空かつ `L_new > 1` を mask 規則 (c)（`offset_allowed_mask`:1612、`allowed[i][j] = j <= s_prev + i`）で扱える。**target が draft の K トークンを 1 回の forward で検証できる根拠**。書き戻しはエラー時に呼び出し前の状態を保つ（原子的） |
| `KvCache` | `attention.rs:1555` | `k`／`v` は `[B, S_cached, E]` のホスト `Tensor<f32>`。公開メソッドは `new`／`is_empty`／`seq_len`／`batch`／`embed_dim`／`clear`／`k`／`v` のみ。**切り詰め・セッターはない**。書き手は `forward_with_cache` のみという不変条件（doc `:1538-1548`）。`clone()` は `Arc` 共有でデータを複製しない |
| `Tensor::narrow` | `crates/tensor-core/src/tensor.rs:515` | 算術なしの切り出し。巻き戻しに使える |
| `Generator` | `crates/tensor-core/src/rng.rs:676` | `Clone` あり（`:775`）。公開メソッドは `bernoulli`（`:705`）／`normal`／`multinomial`／`manual_seed`／`initial_seed`。一様乱数を直接引く公開メソッドはなく、受理判定は `bernoulli`（確率 `min(1, p_target / p_draft)`）で組める見込み（実装 issue で要確認） |
| facade `inference` | `crates/facade/src/inference/mod.rs:116` | `generate` 系 4 名を 1 文の純再エクスポート。`crates/facade/tests/api_surface.rs:22499` の `GENERATE_APPROVED_REEXPORT` がその 1 文の完全一致を検査する |

制約（既存記録から）:

- `docs/kv-cache-design.md` §7: バッチ内の全系列が同じ `S_cached` を共有し、padding 用の追加 mask は非対応（`crates/facade/src/nn/kv_cache.rs:26-32` の既知の制限）。**よって初期スコープの speculative decoding は B = 1 に限る**（B > 1 では行ごとに受理長が異なりキャッシュ長が揃わない）。
- `docs/facade-generate-decision.md` §17.4: facade のみの利用者は `forward_step` に渡る `caches` へ書き込めない（`forward_with_cache` は非公開）。KV を使う facade 側のモデルは `RefCell<StatefulAttention>` を内部に持ち `num_kv_layers()` は 0。この形のモデルは外から巻き戻せない（§10 論点 3）。

## 4. 内部実装の配置（推奨）

- 担当は core-builder（`crates/autodiff`・`crates/facade`）。
- **speculative decoding**: `crates/autodiff/src/generate/speculative.rs`（`generate.rs` のサブモジュール化）。理由: 非公開ヘルパーを `pub(super)` で共有でき、`pub(crate)` の面積を増やさない。別ファイル `generate_speculative.rs` 案はヘルパーを `pub(crate)` に広げる必要があり、推さない。既存 `generate` のシグネチャ・挙動は不変。
- **連続バッチング**: ホスト側ロジックのみ。同じく `crates/autodiff/src/generate/scheduler.rs`。
- facade は純再エクスポート（`nn::rnn`・`nn::kv_cache`・`inference` の先例と同型）。facade 独自 newtype・`Sequential::*` への結線は作らない。

### 4.1 `generate()` と `KvCache` の拡張箇所

- `generate()` 本体は変えず**別関数を追加**する。`GenerateConfig` にはフィールドを足さない（承認形のため）。
- KV の巻き戻し（draft の棄却位置以降を捨てる）の方式比較:

| 案 | 内容 | 公開面 | 余分な計算 | 評価 |
|---|---|---|---|---|
| (i) | 検証 forward 前に `KvCache::clone()`（`Arc` 共有で O(1)）を保存し、棄却時に復元して受理分だけを再 forward | 増えない | 受理分の再 forward が 1 回増える | 不変条件を緩めない。**推奨** |
| (ii) | `attention.rs` 内に `pub(crate)` の切り詰め（ホスト `narrow`・算術なし）を足す | 増えない（crate 内部） | なし | 効率的だが「書き手は `forward_with_cache` のみ」の不変条件を緩める。次点 |
| (iii) | `KvCache::truncate` を `pub` で追加 | **増える** | なし | 公開面の追加＝承認事項。§10 論点 4 |

- (i) は draft 側の `caches` にも同様に適用する。draft と target は別々の `AutoregressiveModel`・別々の `caches` 配列で、語彙サイズ一致を fail-closed で検査する（§7）。
- (i)・(ii) が働くのは `num_kv_layers() > 0` のモデル（渡された `caches` を使う形）だけである。`num_kv_layers() == 0` の内部状態保持型は対象外（§10 論点 3）。

## 5. facade 公開形（推奨 1 案・未承認）

配置は `fandhe_ai::inference`。`GENERATE_APPROVED_REEXPORT` の 1 文は変えず、**別の `pub use` 文**で追加し、対応する承認形定数・正ガードを `api_surface.rs` に足す（実装 issue 側の受け入れ条件。本 docs PR では触らない）。

- speculative: `generate_speculative<T: AutoregressiveModel + ?Sized, D: AutoregressiveModel + ?Sized>(target: &T, draft: &D, input_ids: &Tensor<i32>, config: &GenerateConfig, spec: &SpeculativeConfig) -> Result<Tensor<i32>, AutodiffError>` 相当。`SpeculativeConfig`（draft 先読み長 `k` のみ）は `#[non_exhaustive]`。**受理統計は返さない**（戻り値の型を `generate` と揃えて薄く保つため。統計が要る場合は別承認）。
- 連続バッチング: 素の `struct`（名前案 `BatchScheduler`）。同期メソッドは `new(limits)`（上限は必須。§7）・`submit(input_ids, &GenerateConfig) -> Result<RequestId, AutodiffError>`・`step(&mut self, model) -> Result<usize, AutodiffError>`（進行中の各要求を 1 トークン進め、完了した要求数を返す）・`take_finished() -> Vec<(RequestId, Tensor<i32>)>`。要求ごとに `GenerateConfig` と独立シードを持つ。
- エラー型は既存 `AutodiffError` を流用（新しい型・variant は足さない）。
- `fandhe-ai =0.10.0` に対して追加のみ。既存シグネチャ・意味論・`FitConfig` は不変。
- 公開の手続きは `docs/compat-api-scope.md` §5（ユーザー承認）。承認までは保留ガードで固定し、承認後に正ガードへ反転する（`Tape::gradcheck` の流れと同じ）。

## 6. 数値一致の判定方式

新しい tolerance・baseline を作らない。REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）と既存の `assert_parity` をそのまま使う。

### 6.1 greedy 版

- 判定対象は token 列の同一性（target 単独の `generate`〈Greedy〉と同じ列）。
- **bit 一致を確定事実として書かない**。`docs/kv-cache-design.md` §3.5 のとおり、CPU GEMM のブロッキング（`KC`）は入力形状に依存するため、検証 forward（`L_new ≈ K`）と逐次 forward（`L_new = 1`）の logits が bit 一致する保証はなく、僅差タイで argmax が反転しうる。
- 状態を持たない、または形状に依らず bit 同一の logits を返すテスト用モデル（表引き等）では、token 列の**完全一致を要求**する。
- KV キャッシュ付きの実モデルでは、token 列一致は「事前登録の仮説」とし、契約上の判定は **位置ごとの logits に対する既存の統一複合判定**へ帰着させる。テスト入力は上位 2 候補の margin を十分に取って反転を避ける。margin 付き入力でも仮説が破れる場合は既存判定の範囲外になりうるため、承認依頼に戻す（§10 論点 2）。
- 境界ケース: draft == target、`k = 1`、残り長 < `k`、draft が常に外す、draft が常に当たる。

### 6.2 サンプリング版

- speculative sampling は target の出力分布を保つが、RNG の消費順が `generate` と変わるため、**同一シードで `generate` と token 列が一致することは構成上ない**。保証できるのは分布の一致で、検定には統計的な判定契約（標本数・有意水準）が要る。これは統一複合判定の範囲外であり新しい判定契約になる。
- **tolerance・標本数は提案しない**。承認依頼に戻す論点として列挙し（§10 論点 1）、サンプリング版の実装 issue は承認待ちでブロックとする。
- 既存判定の範囲で検査できるもの: 同一シード・同一入力での決定性、グローバル RNG の非消費、受理確率・残差分布の `f64` 計算を独立実装の参照値と突合、決定的に定まる退化ケース（draft == target なら全受理）。

### 6.3 連続バッチング

- 判定対象は「各要求の出力が、その要求を単独で `generate` した結果と一致すること」。
- 第 1 段階（§8.1）は要求ごとに B = 1 で forward するため forward の形状が単独実行と同じで、**token 列の完全一致を要求できる**（要求間でキャッシュ・`Generator` を共有しないことが前提）。
- テンソル単位で束ねる段階は形状が変わるため、§6.1 と同じ「統一複合判定へ帰着」になる。

### 6.4 実機 parity

CUDA／Metal 対 CPU の parity は実装 issue で `#[ignore]` 分離する。未実測なら `docs/perf/logs/speculative-batching-<実装 issue 番号>/README.md` に測定コマンドと空の記入欄を置く申し送りとし、実測値は推測で書かない（先例: `docs/perf/logs/generate-2191/README.md`）。本 issue では `docs/perf/logs/` を作らない。

## 7. fail-closed 条件

| 対象 | 条件 | 結果 |
|---|---|---|
| speculative | `config.validate()` 失敗、prompt 空、`max_length < prompt_len`、B ≠ 1 | `Err(InvalidArgument／Shape)`（`generate` と同順で検査） |
| speculative | `k == 0`、`k` が残り長を超える設定は残り長で丸める（丸めは doc に明記） | `k == 0` は `Err(InvalidArgument)` |
| speculative | draft と target の語彙サイズ不一致 | `Err(Shape(ShapeMismatch))`。両モデルの最初の戻り shape から確定 |
| 両者 | `forward_step` の戻り shape が `[B, L_new, V]` でない、`V == 0`、`V > i32::MAX` | 既存 `validate_forward_step_output`／`validate_vocab_le_i32_max` を流用 |
| 両者 | サンプリングに使う位置の logits が非有限 | 既存どおり `Err` |
| 両者 | `top_k > vocab` | 既存 `validate_top_k_le_vocab` |
| 出力 | `B × max_length` の `checked_mul` と `isize::MAX` バイト検査 | 既存方式（`generate.rs:480-503`）を踏襲 |
| スケジューラ | 同時要求数・キュー長・要求ごとの `max_length` の上限は**構築時に必須** | 超過は `Err(InvalidArgument)`。暗黙の無制限を許さない |
| スケジューラ | 要求の途中で `forward_step` が `Err` | 当該要求を失敗として切り離し、他要求の状態は変えない（実装 issue で型を決める） |

本番経路で `unwrap`／`expect` を使わない。KV 巻き戻しに失敗した場合は、復元後の状態で処理を続けず `Err` を返す。

## 8. 連続バッチングのスケジューラ

### 8.1 推奨の第 1 段階

要求ごとに `Vec<KvCache>` と `Generator` を所有し、`step()` 1 回で進行中の各要求を 1 トークン進める同期スケジューラ。イテレーション単位で要求の追加・完了分の退出ができる。**テンソル単位のバッチ化による利得はない**（性能保証なしは spec どおり）。完了条件は当面 `max_length` 到達のみ（EOS 停止は生成ループ拡張の別系統で、本記録では先取りしない）。

対象とするモデルの契約（§6.3 の単独実行一致と §7 の要求間の状態分離の前提）:

| モデルの形 | `num_kv_layers()` | 第 1 段階での扱い |
|---|---|---|
| 生成状態を渡された `caches` だけに持つ形 | `> 0` | **対象**。要求ごとの `Vec<KvCache>` で状態が分かれる |
| 状態を持たない形（毎回 prompt 全体から計算する等） | `0` | **対象**。要求間で共有する状態がない |
| 内部状態保持型（§3 の `RefCell<StatefulAttention>` 等。`docs/facade-generate-decision.md` §17.4） | `0` | **対象外**。渡された `caches` を使わないため、同じ `model` で要求 A・B を交互に進めると内部 KV が混在する |

- `num_kv_layers() == 0` だけでは状態なしと内部状態保持型を型でも実行時でも区別できない。第 1 段階はこれを**doc の契約**として `BatchScheduler`（と `step`）に明記し、内部状態保持型を渡した場合の出力は保証しない。
- 内部状態保持型を対象に含めるには、要求ごとのモデル状態の分離（要求ごとに別インスタンスを渡す、またはモデル側の状態の退避・復元フック）が要る。これは trait・公開面の変更を伴うため承認事項とする（§10 論点 8）。

### 8.2 テンソル単位で束ねる段階（条件付き・承認事項）

同じ `S_cached` の要求のグルーピング、または padding mask・行単位 gather が要る。`KvCache` のバッチ軸連結・分割や padding mask は、公開面と「単一の書き手」不変条件に触れるため分離して承認を取る（§10 論点 5）。

### 8.3 外部依存・スレッド・ネットワークを要さない形

- `std` のみ。`std::thread`・チャネル・async ランタイム・スケジューラ層での `rayon` の直接使用・ファイル／ソケット I/O を持たず、呼び出しスレッド上で同期実行する。
- 実装 issue の受け入れ条件に「該当モジュールへの grep で 0 件」を入れる（`scheduler.rs` に対する `std::thread|std::sync::mpsc|tokio|async |std::net|std::fs|rayon` の検査）。

## 9. 依存追加・新規 `unsafe`・`Op`／`BackendOps` 拡張の要否

| 項目 | 要否 | 根拠 |
|---|---|---|
| 依存の追加 | **不要** | ホスト側ロジックと既存 forward の合成のみ。`Cargo.toml`／`Cargo.lock`／`deny.toml` 不変 |
| 新規 `unsafe` | **不要** | 同上 |
| 新規 `Op`／`BackendOps`／VJP | **不要** | 推論専用で autodiff に載せない |
| 要になりうるもの（別承認） | `KvCache` の公開メソッド追加、padding mask、テンソル単位バッチ化、paged attention の専用カーネル | §10 |

## 10. 承認依頼に戻す論点（実装せず止める）

1. サンプリング版 speculative decoding の分布一致の判定契約（統計検定の標本数・有意水準）。
2. greedy の token 列一致の仮説が実モデルで破れた場合の扱い（既存判定に収まらない場合）。
3. §17.4 の到達経路: `caches` を使う形の公開、またはモデル側の巻き戻しフック。フックを `AutoregressiveModel` の既定メソッドとして足す案は既存実装を壊さないが trait の拡張で公開面の変更になる。別 trait にする案は既存実装に影響しない代わりに型パラメータが増える。いずれも未決。
4. `KvCache` の公開メソッド追加（`truncate` 等）と「単一の書き手」不変条件の緩和。
5. テンソル単位バッチ化に要る padding mask・可変長キャッシュ。
6. facade 公開面の追加そのもの（`docs/compat-api-scope.md` §5）。
7. paged attention（条件付きのまま。K-3 が未着手）。
8. 連続バッチングで内部状態保持型のモデル（§8.1 の表）を扱うための、要求ごとのモデル状態の分離方式。論点 3 と同じく trait の拡張か別 trait かが未決。決まるまで第 1 段階は内部状態保持型を対象外とする。

## 11. 2 時間粒度の実装 issue 分解案

共通条件（変えないもの）: 依存・tolerance・baseline・ガードレール閾値・`docs/spec/`・既存公開シグネチャ。各 issue は 1 関心事・単独 PR。仮番号は本表内のみ。

| # | タイトル案 | 依存 | 受け入れ条件の要点 | 担当 | 承認待ち |
|---|---|---|---|---|---|
| 1 | `docs(facade): speculative decoding・連続バッチングの公開形と判定方式の承認依頼` | なし | §10 論点 1〜8 の確定依頼、`compat-api-scope.md` §5.1 への行追加。承認は実装 Agent が代行しない | core-builder | 承認依頼そのもの |
| 2 | `refactor(autodiff): generate のヘルパーをサブモジュールから使えるようにする` | なし | `generate/` ディレクトリ化と `pub(super)` 化のみ。挙動・既存テスト・公開シグネチャ不変 | core-builder | なし |
| 3 | `feat(autodiff): KV キャッシュ巻き戻しの内部実装` | 2 | 推奨 (i)（clone 保存・復元）。復元後の状態が呼び出し前と同一であることの単体テスト | core-builder | 論点 4 の結論次第で (ii) |
| 4 | `feat(autodiff): speculative decoding（greedy）の内部実装` | 2, 3 | §7 の fail-closed 全件。B = 1 限定。境界ケース（§6.1） | core-builder | なし |
| 5 | `test(autodiff): speculative greedy の一致テスト` | 4 | 状態なしモデルで token 列完全一致、KV モデルで位置ごとの logits を統一複合判定 | test-runner | 論点 2 の結論次第 |
| 6 | `feat(autodiff): speculative decoding（サンプリング）の内部実装` | 4 | 決定性・グローバル RNG 非消費・受理確率の `f64` 突合 | core-builder | **論点 1 が未承認の間はブロック** |
| 7 | `feat(autodiff): 連続バッチング スケジューラの第 1 段階` | 2 | §7 の上限必須化・失敗の切り離し。`step`／`take_finished`。§8.1 の対象モデルの契約（内部状態保持型は対象外）を doc に明記 | core-builder | なし（内部状態保持型の対応は論点 8） |
| 8 | `test(autodiff): スケジューラの単独実行一致と非依存の検査` | 7 | `num_kv_layers() > 0` のモデルと状態なしモデルの両方で、要求 A・B を交互に進めても各要求が単独 `generate` と token 列完全一致、§8.3 の grep が 0 件 | test-runner | なし |
| 9 | `feat(facade): speculative decoding・連続バッチングを記録の形で公開` | 1 の承認, 4〜8 | 別 `pub use` 文、承認形定数・正ガード、doctest。保留ガードの反転 | core-builder | 承認後 |
| 10 | `test(backend): CUDA／Metal 実機 parity の #[ignore] テストと perf/logs 申し送り` | 5, 8 | 実測値は推測で書かない | backend-builder | なし |
| 11 | `docs(facade): 周辺 docs と本記録への実装記録の追記` | 9 | `compat-api-scope.md` §5 適用記録、`compat-feature-gap.md`、本記録への実装記録 | docs-writer | なし |

規模見積り（推測）: 提案文書の概算（speculative 6〜10、連続バッチング 8〜14）に対し本案は 11 件。paged attention と生成ループ拡張（EOS 等）を含まないため。

## 12. スコープ外・申し送り

- スコープ外: 実装そのもの、実装 issue の起票、HTTP／API サーバ、量子化 KV、paged attention、EOS 停止・top-p などの生成ループ拡張、B > 1 の speculative decoding。
- 要対応事項（ユーザー）: §10 の論点 1〜8 の承認。ruleset・branch protection・リポジトリ設定は変更しない。

## 13. セキュリティ観点（OWASP）

- A08: 承認を代行しない。2 つの承認 URL を範囲付きで別々に引用し、公開形・判定方式・拡張要否は未承認と明記した。
- A03: 非信頼の外部入力は token id と logits のみ。形状・語彙・有限性・上限を §7 のとおり先に検証する。
- A04: スケジューラの同時要求数・キュー長・`max_length` の上限を必須にし、出力バッファは `checked_mul` と `isize::MAX` バイト検査で確保する。
- A02: 乱数は xorshift64*（非暗号）。生成 token をセキュリティ用途に使わない既存注記を引き継ぐ。
- A06: 依存追加なし。
- A01／A05／A07: ネットワーク公開面を持ち込まない（HTTP サーバは対象外のまま）。
- 新規 `unsafe` なし。秘密情報は含まない。

## 14. 出典

- spec: `docs/spec/04-requirements.md:237`・`:458`（読むのみ）
- `docs/facade-serving-infrastructure-spec-proposal.md`（§3・§5）、`docs/facade-generate-decision.md`（§13・§17.4）、`docs/kv-cache-design.md`（§3.5・§7・§15.5）、`docs/facade-predict-batches-phase-metrics-decision.md` §8.4、`docs/compat-api-scope.md` §5、`docs/autodiff-functional-transforms-design.md`（書式の先例）
- ソース: `crates/autodiff/src/generate.rs`、`crates/autodiff/src/nn/attention.rs`、`crates/tensor-core/src/{rng.rs,tensor.rs}`、`crates/facade/src/inference/mod.rs`、`crates/facade/src/nn/kv_cache.rs`、`crates/facade/tests/api_surface.rs`

## 15. 承認依頼の所在（イシュー #2883・親 #2882）

- §5 の公開形・§6 の判定方式・§9 の拡張要否・§10 の論点 1〜8 は、`docs/compat-api-scope.md` §5.1 末尾の「Phase 8 公開形（承認依頼 #2883）」ブロックへ転記した（行ラベル `S1`〜`S3`）。本記録 §1〜§14 の内容は変えていない。
- §11 の仮番号と実 issue の対応（親 #2882 の sub-issues で確認）: 1→#2883・2→#2884・3→#2885・4→#2886・5→#2887・7→#2888・8→#2889・10→#2890。6（サンプリング版。論点 1 でブロック）・9（facade 公開。承認後）・11（周辺 docs）は未起票。
- 承認の状況: §5・§6・§9 と論点 1〜8 はすべて未承認のまま。承認は実装 Agent が代行しない。保留ガードの名前と設置担当、`RequestId`・`limits`・`step` の `model` 引数の型、失敗の表現型は未定で、承認時の決定事項として同ブロックに列挙した。
