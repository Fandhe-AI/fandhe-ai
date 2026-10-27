# speculative decoding・連続バッチングの実装設計と issue 分解（イシュー #2857・親 #2841）

本記録は **推奨案の記録であり、facade 公開形の承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、逐語転記せず、事実はソースで再確認した。基準は `origin/main` `8539a41a`（2026-10-08）。以下の `file_path:line` は同 sha のもの。`docs/spec` はサブモジュールポインタ `d6a030fa` の内容を読んだ。

## 1. 位置づけ・承認根拠

- ツリー: ルート #2499 → Phase 7 #2841 → **#2857（本記録）**。実装 issue の起票は本記録のマージ後に update-issue-tree で行う（本 issue では起票しない）。
- 本記録の作成を指示した根拠: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`（2026-10-08、リポジトリ所有者本人）。承認範囲は **設計の記録まで**。
- 案 C（REQ-9 の境界の段階化）自体の承認は別 URL（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`、2026-10-07。spec `docs/spec/04-requirements.md:458`）で、上記と混同しない。同行と `:237` は「公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張は本追記では承認しない」と明記している。
- したがって §5 の facade 公開形、§6 の判定方式、§8 の拡張要否は **すべて推奨案であり未承認**。承認の代行はしない。
- 追記（#2933）: その後、`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650` 項 1 で第 1 段階の公開形（3 名・greedy・B = 1）が承認された。範囲と確定シグネチャは §17。判定方式・サンプリング版・論点 1・2・3・5・7・8 は未承認のまま。

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
| `generate()` | `crates/autodiff/src/generate/mod.rs`（`pub fn generate`） | 無状態関数。`caches`（`model.num_kv_layers()` 個の `KvCache`）と `Generator::new(config.seed)` を内部で 1 回確保、prefill `[B, T]` → decode `[B, 1]`。EOS 早期停止・pad・top-p はなく常に `max_length` まで。**不変** |
| `AutoregressiveModel` | `generate/mod.rs`（`pub trait AutoregressiveModel`） | `forward_step(&self, new_ids, &mut [KvCache])`。draft・target ともこの trait で受ける |
| 非公開ヘルパー | `generate/mod.rs`: `validate_forward_step_output`・`validate_vocab_le_i32_max`・`greedy_argmax`・`softmax_weights_f64`・`top_k_indices`・`sample_step`・`build_output`・`GenerateConfig::validate`・`validate_top_k_le_vocab` | speculative 側で再利用する（可視性の拡張は §4）。サンプリングはホスト `f64`・独立 `Generator`（グローバル RNG 非消費） |
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
- **speculative decoding**: `crates/autodiff/src/generate/speculative.rs`（`generate/mod.rs` のサブモジュール）。理由: 子モジュールは親の非公開項目へ `super::` で到達できるため、ヘルパーは非公開のまま `super::` で呼べ、`pub(crate)`／`pub(super)` の面積を増やさない。別ファイル `generate_speculative.rs` 案はヘルパーを `pub(crate)` に広げる必要があり、推さない。既存 `generate` のシグネチャ・挙動は不変。
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

> 追記（#2933）: 本節の推奨形は 2026-10-08 のルート #2499 コメント（`issuecomment-6067263650` 項 1）で第 1 段階の 3 名に限って承認され、確定した公開形は §17 に記録した。以下の本文は承認前の推奨の記録として残す。

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

追記（#3100）: 論点 1 は選択肢 B で確定（§20）。

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
| 出力 | `B × max_length` の `checked_mul` と `isize::MAX` バイト検査 | 既存方式（`generate/mod.rs` の `generate()` 内 `checked_mul`／`isize::MAX` 検査）を踏襲 |
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

→ #3104 追記: 公開形・不変条件・判定方式・上限は §21 で決定（案 A。第 1 段階の `step` は残し、束ねる経路は別メソッド `step_batched` の opt-in。実装は #3105〜#3107）。

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

1. サンプリング版 speculative decoding の分布一致の判定契約（統計検定の標本数・有意水準）。（追記（#3100）: 論点 1 は選択肢 B で確定。§20 参照）
2. greedy の token 列一致の仮説が実モデルで破れた場合の扱い（既存判定に収まらない場合）。
3. §17.4 の到達経路: `caches` を使う形の公開、またはモデル側の巻き戻しフック。フックを `AutoregressiveModel` の既定メソッドとして足す案は既存実装を壊さないが trait の拡張で公開面の変更になる。別 trait にする案は既存実装に影響しない代わりに型パラメータが増える。いずれも未決。 → #3095 追記: §19 で決定（案 C）。
4. `KvCache` の公開メソッド追加（`truncate` 等）と「単一の書き手」不変条件の緩和。
5. テンソル単位バッチ化に要る padding mask・可変長キャッシュ。 → #3104 追記: §21 で決定（案 A）。
6. facade 公開面の追加そのもの（`docs/compat-api-scope.md` §5）。（#2933 追記: 第 1 段階の 3 名に限り承認済み。§17 参照）
7. paged attention（条件付きのまま。K-3 が未着手）。
8. 連続バッチングで内部状態保持型のモデル（§8.1 の表）を扱うための、要求ごとのモデル状態の分離方式。論点 3 と同じく trait の拡張か別 trait かが未決。決まるまで第 1 段階は内部状態保持型を対象外とする。 → #3095 追記: §19 で決定（案 C）。 → #3097 追記: rustdoc・フィクスチャ・一致テストへ反映済み（§23）。

## 11. 2 時間粒度の実装 issue 分解案

共通条件（変えないもの）: 依存・tolerance・baseline・ガードレール閾値・`docs/spec/`・既存公開シグネチャ。各 issue は 1 関心事・単独 PR。仮番号は本表内のみ。

| # | タイトル案 | 依存 | 受け入れ条件の要点 | 担当 | 承認待ち |
|---|---|---|---|---|---|
| 1 | `docs(facade): speculative decoding・連続バッチングの公開形と判定方式の承認依頼` | なし | §10 論点 1〜8 の確定依頼、`compat-api-scope.md` §5.1 への行追加。承認は実装 Agent が代行しない | core-builder | 承認依頼そのもの |
| 2 | `refactor(autodiff): generate のヘルパーをサブモジュールから使えるようにする` | なし | `generate/` ディレクトリ化のみ（ヘルパーは非公開のまま `super::` で参照）。挙動・既存テスト・公開シグネチャ不変 | core-builder | なし |
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
- 要対応事項（ユーザー）: §10 の論点 1〜8 の承認。（#2933 追記: 論点 4・6 は §17 で決着。1・2・3・5・7・8 は保留継続）ruleset・branch protection・リポジトリ設定は変更しない。

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
- ソース: `crates/autodiff/src/generate/mod.rs`、`crates/autodiff/src/nn/attention.rs`、`crates/tensor-core/src/{rng.rs,tensor.rs}`、`crates/facade/src/inference/mod.rs`、`crates/facade/src/nn/kv_cache.rs`、`crates/facade/tests/api_surface.rs`

## 15. 承認依頼の所在（イシュー #2883・親 #2882）

- §5 の公開形・§6 の判定方式・§9 の拡張要否・§10 の論点 1〜8 は、`docs/compat-api-scope.md` §5.1 末尾の「Phase 8 公開形（承認依頼 #2883）」ブロックへ転記した（行ラベル `S1`〜`S3`）。本記録 §1〜§14 の内容は変えていない。
- §11 の仮番号と実 issue の対応（親 #2882 の sub-issues で確認）: 1→#2883・2→#2884・3→#2885・4→#2886・5→#2887・7→#2888・8→#2889・10→#2890。6（サンプリング版。論点 1 でブロック）・9（facade 公開。承認後）・11（周辺 docs）は未起票。→ 現状（2026-10-08）: 9（facade 公開）は #2934 で公開済み（§18）。11（周辺 docs）は `docs/compat-api-scope.md` の #2934 記録で扱った。
- 承認の状況: §5・§6・§9 と論点 1〜8 はすべて未承認のまま。承認は実装 Agent が代行しない。保留ガードの名前と設置担当、`RequestId`・`limits`・`step` の `model` 引数の型、失敗の表現型は未定で、承認時の決定事項として同ブロックに列挙した。→ 現状（2026-10-08）: 第 1 段階（3 名・greedy・B = 1）は所有者コメントで承認され #2934 で公開済み（§17・§18）。論点 6 は第 1 段階の 3 名に限って承認。論点 1・2・3・5・7・8 は保留継続。

## 16. 実装記録（#2890・親 #2882。§11 の分解案 10）

- 追加: `crates/facade/tests/speculative_batching_backend_parity.rs`。属性なし 3 件（`cpu_backend_ops_matches_naive_ops_for_speculative_greedy`・`cpu_backend_ops_matches_naive_ops_for_scheduler_greedy`・`comparison_detects_logits_divergence`）と、`#[ignore]` の CUDA 2 件・Metal 2 件（`cfg(target_os = "macos")` 限定）。
- 比較範囲と判定: Greedy のみ。呼び出しごとの logits を REQ-2 統一複合判定（`assert_parity`）で突合し、呼び出し列の構造（回数・`seq_before`・ids・shape）の一致を要求する。token 列一致は事前登録の仮説 H1（全行 margin 前提。`SKIP_C` は #2887 と同値の固定値）で、破れた場合は §10 論点 2 へ戻す。スケジューラは同一バックエンドの単独 `generate` との一致（§6.3）も確認する。
- 申し送り先: `docs/perf/logs/speculative-batching-2890/README.md`（CUDA／Metal 実機は未実測。記入欄は空）。
- CI で走らない理由: CUDA は `#[ignore]`、Metal は cfg により Linux ではコンパイルもされない（`aarch64-apple-darwin` 向けの `cargo check` でコンパイルのみ確認）。
- 新規物なし: `src`・依存・tolerance・baseline・`unsafe` は変更していない。
- 承認状況: facade 公開（S1・S3）は未承認のまま。本記録は承認を代行しない。→ 現状（2026-10-08）: facade 公開（S1・S3）は #2934 で公開済み（§18。`docs/compat-api-scope.md` の S1 行）。
- スコープ外: バックエンド横断の TopK／Temperature 比較、サンプリング版 speculative、B > 1、`num_kv_layers() == 0` の speculative、facade 公開、周辺 docs と `compat-api-scope.md` の適用記録（行 11）。

## 17. 承認された公開形（#2933・親 #2932。§11 の分解案 9 の前段）

**公開は #2934 で実施済み（実装記録は §18）。** 本節は §5 の推奨形に、main にある内部実装（`f4dc7573`）のシグネチャをそのまま当てた**公開形の記録**である。コード（`crates/**`）・`Cargo.toml`／`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変更しない。`pub use` の追加・保留ガードの反転・doctest は後続の #2934 が本節どおりに行う。§1〜§16 の本文は書き換えず、事実を併記する（§1・§5・§10・§12 から本節を指す）。

### 17.1 承認の根拠と範囲

- 根拠: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`（2026-10-08、リポジトリ所有者本人）の項 1。§1 の 2 URL（`6052732061`＝設計の記録まで、`6033824965`＝案 C の境界）とは承認範囲が異なる。
- 承認された範囲（これ以上に広げない）:
  - 第 1 段階の 3 名だけを公開する: `generate_speculative`（greedy・B = 1）・`SpeculativeConfig`・`BatchScheduler`。公開形は §5 の推奨形に main の内部実装のシグネチャをそのまま当てたもの
  - KV の巻き戻しは §4.1 の案 (i)（clone 保存・復元）のまま。`KvCache` の公開メソッドは増やさない
  - 内部状態保持型のモデルが対象外であることを doc の契約として明記する
  - 論点 1・2・3・5・7・8 は保留。保留ガードは公開した名前の分だけ正ガードへ反転し、未承認経路のプローブは残す
  - `fandhe-ai =0.10.0` は追加のみ。依存・tolerance・baseline・ガードレール閾値は変えない
- 記録と実装の差をどちらに合わせるか決められない場合は止めて承認依頼に戻す、という条件が付いている（17.4 で全件に処置を与えた）。

### 17.2 公開パスと `pub use` 文

- 公開パスは `fandhe_ai::inference::{generate_speculative, SpeculativeConfig, BatchScheduler, RequestId, SchedulerLimits}`。`BatchScheduler` の署名に現れる `RequestId`・`SchedulerLimits` は、利用者が名前を書けるよう同時に公開する。facade 独自の newtype・`Tape`／`Sequential` メソッド・別名は作らない（純再エクスポート）。
- `GENERATE_APPROVED_REEXPORT` の 1 文は**変えない**。追加は**別の `pub use` 文を 2 文**とする（サブモジュールごとに 1 文。既存の PREDICT_BATCHES／GENERATE の 2 文と同型で、承認形の定数と 1 対 1 に対応する）。
  - `pub use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, RequestId, SchedulerLimits};`
  - `pub use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};`
- 入れ子 1 文（`generate::{scheduler::{…}, speculative::{…}}`）は採らない（定数との 1 対 1 対応が崩れ、ガードの正規化も複雑になるため）。承認形の定数名の案は `SCHEDULER_APPROVED_REEXPORT`・`SPECULATIVE_APPROVED_REEXPORT`（最終名は #2934 が決める）。

### 17.3 確定シグネチャ（main `f4dc7573` の実装）

| 公開名 | シグネチャ・属性 | 出典 |
|---|---|---|
| `SpeculativeConfig` | `#[derive(Debug, Clone, PartialEq)]`・`#[non_exhaustive]`・`pub k: usize`・`pub fn new(k: usize) -> SpeculativeConfig`。外部からは `new` でしか構築できない。derive は公開後に外せない（公開 API の約束になる） | `crates/autodiff/src/generate/speculative.rs:48-60` |
| `generate_speculative` | `pub fn generate_speculative<T, D>(target: &T, draft: &D, input_ids: &Tensor<i32>, config: &GenerateConfig, spec: &SpeculativeConfig) -> Result<Tensor<i32>, AutodiffError> where T: AutoregressiveModel + ?Sized, D: AutoregressiveModel + ?Sized`。入力は `[T]` または `[1, T]` で出力 rank は入力に合わせる。受理統計は返さない。Greedy 以外は `Err(InvalidArgument)`。検査順 1〜9 は同関数の rustdoc を正とする | 同 `:155-183` |
| `RequestId` | `#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct RequestId(u64)`。不透明で、`submit` ごとに単調増加して採番される | `scheduler.rs:62-64` |
| `SchedulerLimits` | `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`・`#[non_exhaustive]`・非公開フィールド。`pub fn new(max_active: usize, max_queued: usize, max_length: usize) -> Result<SchedulerLimits, AutodiffError>`（いずれか 0、または `max_length` 分の `i32` バッファが `isize::MAX` バイト超なら `Err(InvalidArgument)`）。アクセサ `max_active()`／`max_queued()`／`max_length()` | `scheduler.rs:66-112` |
| `BatchScheduler` | derive なし（`Debug` は件数だけを出す手書き impl）。`new(limits: SchedulerLimits) -> BatchScheduler`・`queued_len(&self) -> usize`・`active_len(&self) -> usize`・`submit(&mut self, input_ids: &Tensor<i32>, config: &GenerateConfig) -> Result<RequestId, AutodiffError>`・`step<M: AutoregressiveModel + ?Sized>(&mut self, model: &M) -> Result<usize, AutodiffError>`・`take_finished(&mut self) -> Vec<(RequestId, Tensor<i32>)>`・`take_failed(&mut self) -> Vec<(RequestId, AutodiffError)>` | `scheduler.rs:205-385` |

- エラー型は既存の `AutodiffError`（`fandhe_ai::AutodiffError`）をそのまま使い、新しい型・variant は足さない。
- `step` の `Err` はスケジューラ全体の不変条件破れに予約され、第 1 段階では返す経路がない。要求単位の失敗は他の要求から切り離して `take_failed` に積む。

### 17.4 記録と実装の差分と処置

| 項目 | 設計記録 §5 | main 実装 | 処置 |
|---|---|---|---|
| `generate_speculative` の型境界 | 「相当」・未確定 | `<T, D>` とも `AutoregressiveModel + ?Sized` | 実装を採用（承認で委任済み） |
| `SpeculativeConfig` の derive・構築 | `#[non_exhaustive]` のみ | `Debug, Clone, PartialEq` ＋ `new(k)`（`GenerateConfig` に合わせる） | 実装を採用。derive は公開の約束になることを明記 |
| `k` の丸め | 「残り長に丸める」 | 残り長 `R ≥ 2` で `k_eff = min(k, R - 1)`、`R == 1` は target の 1 step decode（全受理時の bonus token を含めて `max_length` に収まるため） | 記録の文言が粗かっただけで矛盾ではない。実装の規則を正とする（`speculative.rs:27-31`・`:51-53`） |
| `num_kv_layers() == 0` の speculative | 対象外（論点 3） | target／draft とも `Err(InvalidArgument)` で拒否 | 一致。黙って受けず fail-closed で拒否することを契約にする |
| `limits` の型 | 承認時に決める事項 | `SchedulerLimits` | 実装を採用 |
| `RequestId` | 承認時に決める事項 | 不透明 `u64` newtype | 実装を採用 |
| 失敗の表現型 | 実装 issue で決める（§7） | `take_failed() -> Vec<(RequestId, AutodiffError)>` | 実装を採用 |
| `queued_len`／`active_len`／手書き `Debug` | 記録にない | `pub` の inherent | Rust の可視性上、`BatchScheduler` を `pub use` すると pub の inherent メソッドはすべて公開される。裁量による拡張ではなく「実装のシグネチャをそのまま当てる」承認の帰結として列挙する |

どの差も「どちらに合わせるか決められない」には当たらない（承認が選択を実装に委ねている）。後続で main のシグネチャが本表と食い違った場合は、公開せず承認依頼に戻す。

### 17.5 対象モデルの契約（doc 契約として固定する）

- `generate_speculative` が対象とするのは `num_kv_layers() > 0` で、生成状態を渡された `caches` だけに持つモデル。`num_kv_layers() == 0` は拒否する（fail-closed）。
- `BatchScheduler` が対象とするのは「`caches` だけに状態を持つ型」と「状態なしの型」（§8.1 の表）。**内部状態保持型**は対象外で、出力を保証しない。
- 内部状態保持型は「生成状態を渡された `caches` の外に持つ型」と定義し、`num_kv_layers()` の値では決めない。`num_kv_layers() > 0` を返しつつ内部に状態を持つ型も作れ、両関数とも型でも実行時でも検出できない。この点を利用者の保証事項として rustdoc に書く。
- 既知の制限（論点 3 が保留中の帰結）: facade だけの利用者は `forward_with_cache` を使えず、`KvCache` にもセッターがない（`docs/facade-generate-decision.md` §17.4）。`caches` を正しく進めるモデルを facade の公開面だけで組む経路は現状では限られる。到達経路は本 issue でも #2934 でも追加しない。 → #3095 追記: §19.6 で解消方針を決定。 → #3097 追記: rustdoc の記述を §19.6 の契約へ更新済み（§23）。

### 17.6 保留ガードの反転範囲（#2934 への申し送り。本 issue では触らない）

S1〜S3 専用の `*HoldDoctestGuard` は存在しない（`docs/compat-api-scope.md` §5.1 S ブロックのとおり）。「反転」の実体は次の 3 つ。

1. `crates/facade/tests/api_surface.rs` の `inference_module_reexports_exactly_expected_surface` の期待集合を 2 文から 4 文にする。
2. `LOWERCASE_PUB_USE_LEAF_ALLOWLIST` に `generate_speculative` を足す。
3. 承認形の定数 2 個、各文 1 件ちょうどを要求する正ガード、`fandhe_ai::inference::…` 経由の到達プローブ、利用例の doctest を新設する。

注意: 追加する 2 文はパスに `generate` トークンを含むため、現行の `generate_exposures`（`facade_exposes_generate_items_only_in_approved_shape`。期待 1 件）に拾われて落ちる。#2934 は (a) 当該ガードの期待集合へ新しい 2 文を加える、(b) 検出を葉が公開名のいずれかに絞る、のどちらかを選ぶ。どちらでも `GENERATE_APPROVED_REEXPORT` の文字列は変えない。

残す否定プローブ（未承認経路）:

- サンプリング版の入口がないこと（`SamplingStrategy::Greedy` 以外は `Err`）
- `KvCache` の公開メソッド集合が `new/is_empty/seq_len/batch/embed_dim/clear/k/v` から増えていないこと（`truncate` 等なし。論点 4）
- `kv_rewind`／`KvSnapshot` に facade から到達できないこと
- `scheduler`／`speculative` のモジュール自体を facade で再エクスポートしないこと（フラットな名前だけ）
- B > 1 と padding mask の経路がないこと

否定ガードは stable rustdoc の `compile_fail` コード照合を当てにせず、正のプローブとインベントリで組む。

### 17.7 保留を続ける論点

- （#3109 追記）論点 7 の設計は `docs/device-resident-kv-paged-attention-design.md` に推奨形として記録済み。承認の確認は同 doc §9 の所有者確認表による。以下の保留状態は変えない。
- （#3112 追記）論点 7 は #3018 の承認（https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6207013704）を受け、#3112 で paged attention の入口（`Tape::new_paged_kv_cache`・`Tape::paged_attention_forward`・`PagedKvCache`・`PagedKvCacheConfig`）を facade へ公開した。`generate`／`BatchScheduler` への結線は未実施（`docs/device-resident-kv-paged-attention-design.md` §13）。
- 追記（#3100）: 論点 1 は選択肢 B で確定（§20）。以下の列挙のうち 1 は解消済み。
- → #3104 追記: 論点 5 は案 A で確定（§21）。以下の列挙のうち 5 も解消済み。
- 論点 1（サンプリング版の判定契約）・2（greedy の token 列一致の仮説が破れた場合）・3（内部状態保持型への到達経路）・5（padding mask・可変長キャッシュ・テンソル単位のバッチ化）・7（paged attention）・8（スケジューラでの内部状態保持型の扱い）は保留のまま。
- 論点 4 は案 (i) で確定し `KvCache` は不変。論点 6 は第 1 段階の 3 名（と署名に現れる `RequestId`・`SchedulerLimits`）に限って解消した。
- 対象外のまま: B > 1、EOS 停止・top-p、受理統計、HTTP サーバ、量子化 KV、facade の `caches` 到達経路。

### 17.8 変えないもの・セキュリティ観点

- 依存・新規 `unsafe`・`Op`／`BackendOps`・tolerance・baseline・閾値・`docs/spec/`・`fandhe-ai =0.10.0` の既存シグネチャ（追加のみ）・`FitConfig`。
- A04: `SchedulerLimits` の 3 上限は必須で 0 は拒否する（暗黙の無制限を作らない）。出力バッファは `checked_mul` と `isize::MAX` バイトで検査し、要求単位の失敗は他の要求から切り離す。
- A03: 外部入力は token id と logits だけで、形状・語彙・有限性の検査順は 17.3 の rustdoc の検査順を正とする。
- A02: `Generator` は非暗号の xorshift64* であり、生成 token をセキュリティ用途に使わない。
- スケジューラは同期実行でスレッド・I/O を持たず、ネットワーク公開面はない。

## 18. 実装記録（#2934・親 #2932。§11 の分解案 9）

§17 の公開形どおりに `fandhe_ai::inference` へ公開した。承認根拠は §17.1（`issuecomment-6067263650` の項 1）。`Cargo.toml`／`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/`・`KvCache` の公開メソッド・`forward_with_cache` の可視性・`GENERATE_APPROVED_REEXPORT` の文字列は変えていない。`fandhe-ai =0.10.0` に対しては追加のみ。

### 18.1 追加した `pub use`（`crates/facade/src/inference/mod.rs`）

既存の generate の 1 文の後ろに、別の 2 文を足した（入れ子・glob・別名・モジュール再エクスポートなし）。

- `pub use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, RequestId, SchedulerLimits};`
- `pub use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};`

### 18.2 ガード反転の対応（§17.6 の 1〜3 と注意）

| §17.6 | 実施内容（`crates/facade/tests/api_surface.rs`） |
|---|---|
| 1 | `inference_module_reexports_exactly_expected_surface` の期待集合を 4 文に変更。自己テストに 4 文版の正例と、追加文・欠落・別名・glob・モジュール再エクスポート・`pub(crate)` の負例を追加 |
| 2 | `LOWERCASE_PUB_USE_LEAF_ALLOWLIST` に `generate_speculative` を追加 |
| 3 | 承認形の定数 `SCHEDULER_APPROVED_REEXPORT`／`SPECULATIVE_APPROVED_REEXPORT`、正ガード `facade_exposes_speculative_scheduler_items_only_in_approved_shape`（各 1 件ちょうど。自己テスト付き）、到達プローブ `speculative_scheduler_items_are_reachable_via_facade_inference_path`（署名・`?Sized` 境界・derive・内部クレートの型との同一性）、doctest 実在検査 `speculative_scheduler_usage_doctests_are_present_and_compiled` を新設 |
| 注意 | 案 (a) を採用。`facade_exposes_generate_items_only_in_approved_shape` の期待を 3 文（generate・scheduler・speculative）に広げ、検出器 `generate_exposures` 本体と自己テストは変えていない（検出範囲を狭める案 (b) は採らず fail-closed の強さを保つ） |

### 18.3 新設した否定インベントリ・プローブ

- `speculative_scheduler_public_shape_is_pinned`: `BatchScheduler`／`SchedulerLimits`／`SpeculativeConfig` の pub fn 集合・フィールド・`#[non_exhaustive]`・`RequestId` の非公開フィールドと、`speculative.rs` の自由な pub fn が `generate_speculative` 1 件のみであること（サンプリング版の入口が無いことを兼ねる）。将来 pub メソッドを足すと本テストが落ちるため、承認とセットで更新する
- `kv_cache_public_methods_stay_unchanged_for_speculative_scheduler`: `KvCache` の公開メソッドが `new/is_empty/seq_len/batch/embed_dim/clear/k/v` の 8 件のままであること（`truncate` 等なし。論点 4）
- `kv_rewind_stays_private_and_absent_from_facade`: `kv_rewind` が非公開で、facade に `kv_rewind`／`KvSnapshot` が現れないこと
- モジュール自体・glob の再エクスポート禁止は 18.2 の 4 文完全一致と小文字葉 allowlist が担う
- 検出範囲の限界: トークン走査のため、マクロ生成・`use … as` 別名経由の到達は範囲外。否定ガードは `compile_fail` のコード照合を使わず、正のプローブとインベントリで組む

### 18.4 利用例 doctest の決定

`BatchScheduler` は状態なしモデルで成功する例（単独 `generate` との一致まで確認）を付けた。`generate_speculative` は **facade の公開面だけでは成功する実行を示せない**（`num_kv_layers() == 0` は `Err` で拒否され、`caches` を進めるモデルを組む経路が facade に無い。論点 3 は保留）ため、fail-closed 契約（`num_kv_layers() == 0` と TopK の拒否）を示す例とした。成功経路は `crates/facade/tests/speculative_batching_facade.rs` の結合テストが担う。doctest のために `KvCache` のセッターや `truncate`、`forward_with_cache` の公開は足していない。

→ #3098 追記: #3096 の入口（`Tape::stateful_attention_forward_with_cache`）で `caches[0]` を進める、facade の公開面だけの表引き KV モデルによる成功例 doctest を `inference/mod.rs` に追加した（出力は target 単独 `generate` と token 列一致。fail-closed 例は維持）。`api_surface.rs` の doctest 実在検査へ成功例の必須語と、成功例が内部クレート（`fandhe_ai_autodiff`）を使わないことの否定検査（自己テスト付き）を足した。結合テストには、k・draft（全受理／全棄却／部分受理）・prompt 長・`max_length`・入力 rank（1／2）・target 層数の掃引（`speculative_matches_greedy_generate_on_public_surface_table_model_sweep`）を足した。変えないもの: `pub use`・承認形の定数・tolerance・baseline・依存・`unsafe`・`docs/spec`。承認根拠は D-A の範囲に限り、追加の承認は得ていない。

### 18.5 対象モデルの契約の doc 化

「内部状態保持型」を「生成状態を、渡された `caches` の外に持つ型」と定義し、`num_kv_layers()` の値では決まらない・型でも実行時でも検出できない・対象外で出力を保証しない・利用者が保証する、と facade の `inference` モジュール doc と autodiff の `BatchScheduler`／`generate_speculative` の rustdoc に書いた（従来の「内部状態保持型は `num_kv_layers() == 0` を返す」は §17.5 と食い違うため改めた）。

### 18.6 テストと実機

- 結合テスト `crates/facade/tests/speculative_batching_facade.rs`: スケジューラ×状態なし／KV モデルで単独 `generate` と完全一致、上限・不正要求の fail-closed、要求単位の失敗の切り離し、speculative の Greedy 一致（k・draft・prompt 長・`max_length`・rank 1／2 の掃引）、サンプリング・k = 0・B = 2・`num_kv_layers() == 0`・空 prompt・`max_length < T` の拒否
- 表引き KV モデルは形状に依らない logits を返すため完全一致を要求する。実 MHA の logits による token 列一致の仮説（論点 2）と CPU 対 CUDA／Metal の parity は `speculative_batching_backend_parity.rs`（#2890）が担う。新しい `#[ignore]` テストは足していない
- 実機（GB10／M4 Max）は未測定。申し送りは `docs/perf/logs/speculative-batching-facade-2934/README.md`

### 18.7 保留・スコープ外

追記（#3100）: 論点 1 は選択肢 B で確定（§20）。

論点 1・2・3・5・7・8 は保留のまま。B > 1、EOS 停止・top-p、受理統計、HTTP サーバ、量子化 KV、facade の `caches` 到達経路、実機測定は本 issue の対象外。

→ #3095 追記: 論点 3・8 の公開形は §19 で決定（`caches` の到達経路を開ける案 C。実装は #3096）。 → #3097 追記: 論点 8 の「状態分離」は新機構を作らず、要求ごとの `Vec<KvCache>` と非公開 `kv_rewind` が担う（§23）。

## 19. caches の到達経路と内部状態保持型の公開形（#3095・親 #3094）

§10 の論点 3（`caches` の到達経路・巻き戻しフック）と論点 8（要求ごとのモデル状態の分離）は「trait の拡張か別 trait か」で保留だった。本節はその公開形を 1 つに決める。**docs のみの変更**で、`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変えない。実装は #3096 以降。

### 19.1 承認根拠と範囲

- 根拠: D-A（`https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6111643437`、2026-10-11、リポジトリ所有者 aLiz-Nancy）。generate 系を含む「保留のまま実装できる項目」を推奨形で一括承認するもので、記録に形が書けた点はそのまま実装へ進めてよく、形が書けない点だけを所有者へ戻す、という範囲である。本節はこの範囲を超える承認を主張しない。
- §17.1 の決定（`KvCache` の公開メソッドは増やさない。論点 4）は維持する。

### 19.2 調査事実（基準 `origin/main` `ab14678b`）

| 事実 | 出典 |
|---|---|
| `AutoregressiveModel` は `num_kv_layers`・`forward_step(&self, &Tensor<i32>, &mut [KvCache])` の 2 メソッド。「モデルは不変で、KV キャッシュだけを外から借りる」設計 | `crates/autodiff/src/generate/mod.rs` |
| `KvCache` の書き手は `forward_with_cache` だけ。公開メソッドは 8 件。セッターは置かない | `crates/autodiff/src/nn/attention.rs`（`forward_with_cache` は 1682 行） |
| `forward_with_cache` は失敗時にキャッシュを変えない（原子的）。rank・B・E・`L_new != L_new_kv` を先に拒否する | 同上 |
| `StatefulAttention::forward(&mut self, tape, x_new)` は `mha.bind(tape).forward_with_cache(x_new, x_new, x_new, &mut self.cache)` への 1 行委譲 | `attention.rs:1911` |
| facade は `KvCache`・`StatefulAttention`・`MultiheadAttentionConfig` だけを再エクスポートし、`MultiheadAttention`・`forward_with_cache` は出さない（P2／P3） | `docs/kv-cache-design.md` §11.4 |
| facade の入口 `Tape::stateful_attention_forward` は `sa.forward(&self.0, x_new)` への薄い委譲 | `crates/facade/src/lib.rs:1701` |
| `generate_speculative` は内部の非公開 `kv_rewind`（退避 → 復元 → 受理分の再 forward）で渡された `caches` を自分で巻き戻す。検査順 9 は `num_kv_layers() == 0` を拒否する | `crates/autodiff/src/generate/speculative.rs`・`generate/kv_rewind.rs` |

帰結: facade の利用者が `forward_step` の中で渡された `&mut KvCache` を `forward_with_cache` 相当で進められさえすれば、`generate`・`generate_speculative`（巻き戻し込み）・`BatchScheduler`（要求ごとの `Vec<KvCache>`）はシグネチャを変えずに動く。巻き戻しは `generate_speculative` が内部で済ませるため、利用者向けの巻き戻しメソッドも trait のフックも要らない。

### 19.3 選択肢の比較（論点 3・8 共通）

| 案 | 内容 | 既存 trait 実装への影響 | 既存 2 関数・`BatchScheduler::step` のシグネチャ | 型パラメータ | fail-closed | 利用者の書きやすさ | `BatchScheduler` への影響 | 評価 |
|---|---|---|---|---|---|---|---|---|
| A | `AutoregressiveModel` に既定メソッド（状態の退避・復元。既定は「非対応」）を足す | 壊れない。ただし利用者の別 trait と名前が衝突しうる | 変えない | 増えない | **できない**。既定が「非対応」だと、状態なしモデル（`num_kv_layers() == 0`）と override 忘れの内部状態保持型を区別できない。拒否すれば既存の状態なしモデルを壊し、受け入れれば黙って誤出力する | override 忘れに気づけない | 退避・復元を `step` に入れ、状態型を `Box<dyn Any>` 等に型消去する必要がある | 不採用 |
| B | 別 trait（関連型 `State` と `save`／`restore`）＋専用の入口 | 影響しない | 既存 2 関数は変えられないため `generate_speculative_stateful`・`step_stateful` 等を新設する | 増える | できる | 入口が 2 系統になり、状態型も利用者が設計する | `BatchScheduler` は非ジェネリックで、要求ごとの状態を型消去して持つしかない | 次点（公開面が最大） |
| **C** | **`caches` の到達経路を開ける**（`StatefulAttention` に外部 `&mut KvCache` を受ける `&self` の forward を足し、facade `Tape` に薄い委譲を 1 件足す。trait は変えない） | 影響しない | 変えない | 増えない | できる（`forward_with_cache` の検査と原子性を受け継ぐ。`KvCache` の書き手は `forward_with_cache` のみのまま） | `RefCell` 不要。trait doc の「モデルは不変・キャッシュは外から借りる」に沿う | 波及しない（要求ごとの `Vec<KvCache>` で状態が分かれ、`&self` のモデルを共有できる） | **推奨** |

簡約点: 案 C で到達経路を開くと、facade の KV 付きモデル（これまで `RefCell<StatefulAttention>` を内部に持ち `num_kv_layers()` が 0 だった形）は「`caches` だけに状態を持つ形」に書き直せる。そうなれば論点 3 の巻き戻しは既存の `kv_rewind` が、論点 8 の状態分離は既存の要求ごとの `Vec<KvCache>` が受け持つ。新しいフックは 0 個で足りる。

### 19.4 推奨形の公開名・シグネチャ

| 追加物 | シグネチャ | 置き場所 | 公開の経路 |
|---|---|---|---|
| autodiff の inherent メソッド | `pub fn forward_with_external_cache<'t>(&self, tape: &'t Tape, x_new: &Var<'t>, cache: &mut KvCache) -> Result<Var<'t>, AutodiffError>`（本体は `self.mha.bind(tape).forward_with_cache(x_new, x_new, x_new, cache)` の 1 行委譲） | `impl StatefulAttention`（`crates/autodiff/src/nn/attention.rs`） | `StatefulAttention` は公開済みのためメソッドは自動的に facade へ出る。`pub use` 文は増やさない |
| facade `Tape` の inherent メソッド | `pub fn stateful_attention_forward_with_cache<'t>(&'t self, sa: &nn::kv_cache::StatefulAttention, x_new: &Var<'t>, cache: &mut nn::kv_cache::KvCache) -> Result<Var<'t>, AutodiffError>`（本体は `sa.forward_with_external_cache(&self.0, x_new, cache)` の 1 行委譲） | `crates/facade/src/lib.rs` の `impl Tape`（既存 `stateful_attention_forward` の隣） | `Tape` に置く理由は P3 と同じ（REQ-12 により生の autodiff `Tape` を取り出せない） |

名前は本節で確定し、後続 issue で変えない。

- **`KvCache` に足す公開メソッド: なし。** `truncate` は、巻き戻しを `generate_speculative` が内部の `kv_rewind` で行うため不要。セッター（任意の `Tensor` を入れる経路）は、書き込みを `forward_with_cache` が担うため不要で、注入経路を作らないので「両方 `Some` か両方 `None`」「`[B, S, E]` contiguous」「B・E の一致」を外部入力から守る検査も新たに要らない。よって **論点 4 は見直さない**（Issue が想定した「公開メソッドを追加して論点 4 を見直す」流れは採らない）。
- **巻き戻しフックの置き場所: 公開しない。** 非公開の `generate::kv_rewind` のまま。trait の既定メソッドも別 trait も作らない。
- `AutoregressiveModel` は 2 メソッドのまま。関連型・関連定数も足さない。

### 19.5 不変条件と fail-closed 条件（新しい入口）

- `forward_with_cache` の検査をそのまま受け継ぐ。rank が 3 でない・B／E が `cache` と合わない・`L_new != L_new_kv`・別 `Tape` の `Var`（`TapeMismatch`）・`S_prev + L_new` の overflow はいずれも `Err` を返し、`cache` は呼び出し前のまま変わらない（原子的）。非既定 config は `from_config` が構築時に拒否済み。
- `sa` の内部キャッシュは読みも書きもしない（`&self` なので型でも保証される。rustdoc と単体テストで固定する。#3096 へ申し送り）。
- `KvCache` の書き手は引き続き `forward_with_cache` だけで、不変条件は緩めない。
- 1 つの `sa` を複数の要求・複数の `caches` で共有しても状態は混ざらない。
- 本番経路で `unwrap`／`expect` を使わない。`AutodiffError` の新 variant は足さない。

### 19.6 各関数に渡す条件と §17.5 の置き換え

| 関数 | 受け付けるモデル | 変更 |
|---|---|---|
| `generate` | `caches` だけに状態を持つ形（新しい入口で facade だけで組める）／状態なしの形 | シグネチャ・挙動とも変えない |
| `generate_speculative` | `num_kv_layers() > 0` で `caches` だけに状態を持つ形 | 検査順 1〜9 は変えない（9 の `num_kv_layers() == 0` 拒否も維持）。facade だけで組める形が `> 0` を返すようになるため拒否を緩める必要がない |
| `BatchScheduler::step` | `caches` だけに状態を持つ形／状態なしの形 | 変えない。内部状態保持型は引き続き対象外 |

§17.5 の置き換え（#3097 が rustdoc に反映する）: 「内部状態保持型」の定義（生成状態を渡された `caches` の外に持つ型。`num_kv_layers()` の値では決まらない）と「対象外で出力を保証しない・型でも実行時でも検出できず利用者が保証する」はそのまま残す。そのうえで「facade だけで KV 付きモデルを組むときは `Tape::stateful_attention_forward_with_cache` で `caches[i]` を進める形にする（`RefCell<StatefulAttention>` 形は推奨しない）」という移行経路を加える。§17.5 の「既知の制限（論点 3 が保留中の帰結）」と `docs/facade-generate-decision.md` §17.4 の「facade だけの KV 付きモデルは `RefCell<StatefulAttention>` 形で `num_kv_layers()` は 0」は、この経路ができたことで解消する（実装は #3096）。

### 19.7 既存公開シグネチャの非破壊確認（追加のみ）

| 項目 | 確認結果 |
|---|---|
| `AutoregressiveModel`・`generate`・`generate_speculative`・`BatchScheduler`／`SchedulerLimits`／`RequestId`／`SpeculativeConfig` | 変更なし |
| `KvCache` | 変更なし（8 メソッドのまま） |
| `StatefulAttention` | inherent メソッドを 1 つ追加するだけ |
| facade `Tape` | inherent メソッドを 1 つ追加するだけ。利用者 trait に同名メソッドがある場合の解決順の変化は Rust の minor 変更の通常範囲として受け入れる（`docs/kv-cache-design.md` §11.5 の P3 と同じ扱い） |
| `pub use` 文 | 増やさない・変えない（`KV_CACHE_APPROVED_REEXPORT`・`inference` の 4 文・`GENERATE_APPROVED_REEXPORT` は不変） |
| `forward_with_cache`・`MultiheadAttention`・`MultiheadAttentionVars` | 非公開のまま。新しい入口は公開済みの `StatefulAttention` に機能を足す形で、P3 に対する追加になる |
| `fandhe-ai =0.11.0` の既存シグネチャ・依存・tolerance・baseline・`unsafe` | 変更なし |

### 19.8 反転・更新する api_surface ガード（`crates/facade/tests/api_surface.rs`）

| ガード | 推奨形での扱い | 担当 |
|---|---|---|
| `generate_public_shape_violations`（`AutoregressiveModel` を 2 メソッドに固定し関連型・定数を禁止）と呼び出しテスト | 不変（案 A を採らないため反転しない） | — |
| `kv_cache_public_methods_stay_unchanged_for_speculative_scheduler` | 不変（8 件）。doc コメントの「論点 3・4 は保留」を「論点 4 は維持、論点 3 は §19 で解消」へ書き換える | #3096 |
| `kv_rewind_stays_private_and_absent_from_facade`・`speculative_scheduler_public_shape_is_pinned` | 不変 | — |
| `facade_exposes_speculative_scheduler_items_only_in_approved_shape`・`inference_module_reexports_exactly_expected_surface`・`facade_exposes_generate_items_only_in_approved_shape` | 不変 | — |
| `nn_kv_cache_module_reexports_exactly_expected_surface`・`facade_reexports_kv_cache_items_only_in_approved_shape`・`KV_CACHE_APPROVED_REEXPORT` | 不変（再エクスポートは増やさない） | — |
| `tape_stateful_attention_forward_is_thin_delegation`・`workspace_declares_stateful_attention_forward_only_in_facade_lib` | 既存は不変。新メソッド用に同型の正ガードを新設（本体が `sa.forward_with_external_cache(&self.0, x_new, cache)` の 1 行委譲であること、宣言が `facade/src/lib.rs` の 1 件だけであること） | #3096 |
| `kv_cache_is_reachable_via_facade_only` | 新 `Tape` メソッドと `StatefulAttention::forward_with_external_cache` のシグネチャを関数ポインタ型注釈で固定するプローブを追加 | #3096 |
| `StatefulAttention` の pub fn インベントリ（現状なし。`docs/kv-cache-design.md` §15.5 の残課題） | 新設。`new`・`from_config`・`forward`・`forward_with_external_cache`・`reset_cache`・`cache`・`seq_len`・`mha` のちょうど 8 件 | #3096 |
| `kv_cache_usage_doctests_are_present_and_compiled` | 外部キャッシュを使う利用例の doctest 実在検査を追加（既存例は残す） | #3096 |
| `speculative_scheduler_usage_doctests_are_present_and_compiled`・`speculative_scheduler_items_are_reachable_via_facade_inference_path` | 成功例を必須トークンとして追加（fail-closed 例は残す） | #3098 |
| 結合テストのフィクスチャ（`crates/facade/tests/speculative_batching_facade.rs` の内部クレート利用） | 新入口へ移せる部分を移す | #3097／#3098 |

否定ガードは stable rustdoc の `compile_fail` コード照合に頼らず、正のプローブとインベントリで組む（§18.3 の方針を引き継ぐ）。

### 19.9 後続 issue への読み替え

- #3096: 「`KvCache` の公開メソッド追加」は**追加なし**、「巻き戻しフック」は**非公開の `kv_rewind` のまま**と読み替える。実装は 19.4 の 2 メソッドと 19.8 のガードに絞る。
- #3097: 受入基準「内部状態保持型で要求 A・B を交互に進めても単独 `generate` と一致する」は、従来 `RefCell<StatefulAttention>` で書いていたモデルを新入口で `caches` 形に書き直したものに対する検査として満たす。`caches` の外に状態を持つ型は引き続き対象外で、型でも実行時でも検出できない点は利用者が保証する。
- #3098: doctest は新入口を使った表引き KV モデルで成功例を書ける。

### 19.10 所有者へ戻す点

| 点 | 扱い |
|---|---|
| 所有者へ戻す点 | **なし**（形がすべて記録に書けた。D-A に従う） |
| `caches` で表せない内部状態（KV 以外。RNN の隠れ状態等）を持つモデル | 保留・スコープ外。需要が現時点で記録になく、必要になったら案 B を出発点に別 issue で扱う。所有者への承認依頼ではない |

スコープ外: コード実装（#3096〜#3098）、B > 1、padding mask、paged attention、サンプリング版 speculative（論点 1）、論点 2・5・7。

→ #3104 追記: 本行の padding mask・論点 5 は §21 で決定。

### 19.11 セキュリティ観点

- A03: `KvCache` に任意の `Tensor` を入れる経路を作らない。新入口は `forward_with_cache` の形状・B・E・Tape 検査と原子的更新を受け継ぐ。
- A04: 案 A は既定実装で黙って誤動作しうる（fail-open）ため採らない。`SchedulerLimits` の 3 上限必須・要求単位の失敗の切り離し・`generate_speculative` の検査順 1〜9 は変えない。
- A06: 依存・`unsafe` の追加なし。ネットワーク公開面なし（スケジューラは同期実行）。
- A08: 承認を代行しない（19.1）。ruleset・branch protection・リポジトリ設定は変更しない。
- A02: `Generator` は非暗号（xorshift64*）で、生成 token をセキュリティ用途に使わない既存の注記を引き継ぐ。

## 20. サンプリング版の判定契約（#3100・親 #3099。§10 論点 1 の決着）

### 20.1 確定の根拠と範囲

- 根拠: イシュー #3100 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/3100#issuecomment-6111753082`、2026-10-11。内容は「全て推奨で」）。推奨は選択肢 B。補助の回帰テストを足すかどうかは本記録の判断に任された。tolerance 定数・baseline・判定規則は変えず、spec の改訂も要らないとされた。
- 本記録はこれ以上の承認を主張しない。実装（#3101）とテスト（#3102）は本節を正として参照する。
- spec REQ-2 は「新しい判定契約が必要になれば別途承認」とする。選択肢 B は既存の判定（統一複合判定・決定性）の範囲で完結するため、新しい数値契約は導入しない。

### 20.2 選択肢の比較と結論

| 案 | 内容 | 評価 |
|---|---|---|
| A | 分布検定（標本数・有意水準）を契約にする | 新しい数値契約が要り、統一複合判定の範囲外。採らない |
| B（採用） | 決定性・グローバル RNG 非消費・受理確率と残差分布の `f64` 突合・退化ケースを契約にする | 既存の判定の範囲で完結し flaky にならない |
| C | 保留を続ける | 後続が止まるため採らない |

結論は B。本契約は target 分布の保存を統計的には保証しない。理論上の保存はアルゴリズムの構成（受理棄却と残差再サンプリング）に依拠する。

### 20.3 用語と計算の定義（#3101 が推測で埋めないための固定事項）

- `p`（target）と `q`（draft）は、同じ `GenerateConfig`（strategy・temperature・top_k）をそれぞれの logits に同じ手順で適用した後の最終的なサンプリング分布と定義する。計算は `generate` の `sample_step` と同じ `softmax_weights_f64`・`top_k_indices` を使う。分布の保存はこの組み合わせについてだけ成り立つ。
- 計算は `f64` で行い、`Generator` へ渡す時点で `f32` に落とす。受理確率 `min(1, p(x)/q(x))` は `[0,1]` に収めてから `f32` として `bernoulli` へ渡す。残差は `max(0, p - q)` を正規化し、`f32` の重みとして `multinomial` へ渡す（`sample_step` の「`f64` の softmax から `f32` の重み」と同形）。
- `q(x)` は draft のサンプリングで `multinomial` に渡したのと同じ `f32` の重みベクトルから読み、計算し直さない。ただし `softmax_weights_f64` が返す `f32` の重みは丸めにより総和が厳密に 1 とは限らず、`Generator::multinomial` は重みを `f64` で累積し総和を掛けた乱数で抽選する。実際の draft 確率は `q(x) = q_weight[x] / sum(q_weight)` であるため、保存した `f32` 重みを `f64` の総和で正規化した値を `q(x)` とする（生の重みを `q(x)` としない）。target 側の `p(x)` も同じ定義（`f32` 重みを `f64` の総和で正規化）で構成し、`p` と `q` の正規化規則を揃える。`q(x) <= 0` または非有限は内部不整合として `Err` を返す。
- 残差の和が 0 になるのは `p == q` のときだけで、そのとき受理確率は常に 1 となり残差には到達しない。それでも残差の和が 0 または NaN の場合は fail-closed で `Err` にする。

### 20.4 RNG の消費順（決定性の前提）

- `Generator::new(config.seed)` を呼び出しごとに 1 個だけ作り、グローバル RNG は消費しない。
- 1 ラウンドの消費順の案: draft が k 回 `multinomial` → 検証位置ごとに `bernoulli` 1 回 → 棄却時は残差を `multinomial` 1 回、全受理時は bonus を `multinomial` 1 回。最終順序は #3101 で確定した（§25.2。prefill 末尾の multinomial 1 回で最初のトークン → 各ラウンド → 最終 1 step の multinomial 1 回）。消費回数の契約は `docs/rng-distributions-generator-decision.md` §1.1（`multinomial` は `m * num_samples` 回固定）に従う。
- 同一シード・同一入力・同一モデルなら、**同一実行環境（同一バイナリ・同一プラットフォーム・同一 toolchain）での再現性**として出力は bit 単位で一意に決まる。プラットフォーム間の bit 一致は保証しない。`bernoulli` は `next_unit_f64` と浮動小数点確率を比較し、`multinomial` は `f64` の累積和・乗算を使い、`softmax_weights_f64` は `f64::exp` を使うため、数学関数の実装差や浮動小数点演算の差で確率境界付近の token 選択が分岐し token 列が変わりうる。プラットフォーム間一致が必要になった場合は、数学関数・logits・浮動小数点演算の条件を別途定義して決定記録へ追記する。`Generator`（`crates/tensor-core/src/rng.rs:676`）は非暗号の xorshift64* であり、生成 token をセキュリティ用途に使わない。

### 20.5 検査項目（#3101・#3102 が参照する契約）

| # | 項目 | 判定 | 置き場所の推奨 |
|---|---|---|---|
| C1 | 決定性: 同一シード・同一入力で 2 回実行し出力が一致する | 完全一致 | 結合テスト |
| C2 | グローバル RNG を消費しない（`crates/autodiff/tests/nn_generate.rs:580` と同型: `manual_seed(s)`→読む→`manual_seed(s)`→実行→読む） | 完全一致 | 結合テスト |
| C3 | 受理確率と残差分布の `f64` 突合: テスト内の独立した `f64` 参照と、`Generator` へ渡す時点の `f32` 値を既存の統一複合判定の定数で比べる。残差の非支持位置（`p <= q`）は構造上厳密に `0.0` | 既存の複合判定＋構造上の厳密値 | 計算を非公開の純関数へ切り出し `speculative.rs` 内ユニットテストで比べる |
| C4 | 退化ケース（a）: draft == target（形状に依存しない logits を返す表引きモデル）なら全受理（受理確率がちょうど 1.0） | 完全一致 | 結合テスト |
| C5 | 退化ケース（b）: `TopK(1)` または重みが厳密に one-hot になる logits なら、出力は Greedy 版の token 列と一致する（RNG 値に依存しない） | 完全一致 | 結合テスト |
| C6 | fail-closed: 語彙サイズ不一致・`top_k > vocab`・非有限 logits・`k == 0`・残差の和が 0 または NaN を `Err` で返す | `Err` の種別 | 結合テストとユニットテスト |

C4 の注意: 実 MHA モデルでは検証 forward と逐次 forward の logits が bit 一致する保証がない（§6.1）。全受理の契約は形状に依存しない logits を返すモデルに限る。

### 20.6 分布検定（契約外）と補助テストの判断

- 分布検定は契約にしない。標本数・有意水準は設定しない。
- 補助の回帰テストとしても閾値つきの統計量（χ² 等）は足さない（記録に任された範囲での判断）。固定シードでも閾値が新しい数値判定になるため。理論分布を構成する要素は C3・C5 で決定的に検査できる。
- #3102 で分布検定の作業は「契約にしないので該当しない」。何かを足すなら固定シードでの完全一致（スナップショット）に限る。
- 参考（契約外）: 将来 A を再開する場合の統計量は、有限語彙上の経験分布と理論分布の Pearson χ²（固定シード・標本数 N・有意水準 α。値は空欄）。再開には所有者の承認と spec の別途承認が要る。

### 20.7 実機（CUDA／Metal）の扱い

- サンプリングは `generate` と同じくホスト側 `f64` で行うため、実機 parity の対象は logits だけになる。
- 統一複合判定に収まる logits 差でも `bernoulli`／`multinomial` の結果が反転して列が分岐しうる。実機で比べる場合は、CPU 側で記録した入力 id 列を実機モデルへ再生して logits を `assert_parity` で比べる（推奨）。または呼び出し列の最初の分岐点までに限って比べる。token 列の一致は契約でも事前登録の仮説でもない。
- `#[ignore]` で分離し、実測は #3102 が挙げる Phase 8 の #3171・#3172 に委ねる。本 issue では `docs/perf/logs/` を作らない。

### 20.8 変えないもの・申し送り

- 変えないもの: `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・baseline・判定規則・`docs/spec/`・依存・`unsafe`・`Op`／`BackendOps`・`generate_speculative` の既存シグネチャ。本 issue は docs のみで `crates/**` に触れない。
- 申し送り: `speculative.rs` のモジュール doc の「未承認でブロック中」の記述と検査 2（`Greedy` 以外の拒否）の解除は #3101 で行う。

## 21. テンソル単位バッチ化の公開形と `KvCache` 不変条件の変更（#3104・親 #3103。§10 論点 5 の決着）

§8.2・§10 論点 5 は「padding mask・可変長キャッシュ・テンソル単位のバッチ化」を保留にしていた。本節は公開形・不変条件・`BatchScheduler` への波及・判定方式・上限を 1 つの推奨形に決める。**docs のみの変更**で、`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変えない。実装は #3105（`KvCache` の可変長対応）→ #3106（`BatchScheduler` の束ねる経路）→ #3107（一致テスト）。

### 21.1 承認根拠と範囲

- 根拠: D-A（`https://github.com/Fandhe-AI/fandhe-ai/issues/3018#issuecomment-6111643437`、§19.1 と同じ）。記録に形が書けた点はそのまま実装へ進めてよく、形が書けない点だけを所有者へ戻す、という範囲である。本節はこの範囲を超える承認を主張しない。
- §17.1・§19.1・§19.4 の「`KvCache` の公開メソッドは増やさない」は #2933／#3095 の範囲での決定だった。不変条件を変える場所としてツリー上で #3104 が指定されているため、本節で「公開メソッドを 1 件（`valid_len`）足し、非公開の内部状態を足す」と決める。矛盾ではなく範囲の違いである。論点 4 のうち `truncate` と任意 `Tensor` を入れるセッターは引き続き採らない。

### 21.2 調査事実（基準 `origin/main` `a2983333`。行番号は実装時に grep で確かめ直す）

| 事実 | 出典 |
|---|---|
| `KvCache` は `k`／`v: Option<Tensor<f32>>` で形は `[B, S_cached, E]`。公開メソッドは 8 件（`new/is_empty/seq_len/batch/embed_dim/clear/k/v`）、セッターはない | `crates/autodiff/src/nn/attention.rs:1555` |
| `forward_with_cache` は `Var::cat(axis 1)` で末尾へ追記する。mask 規則は (a) prefill は causal、(b) `L_new == 1` は mask なし、(c) それ以外は `offset_allowed_mask`（`allowed[i][j] = j <= s_prev + i`）。書き戻しは原子的で B・E の一致を検査する | 同 `:1682`・`:1612` |
| `sdpa_compose` の `attn_mask` は `negate_broadcast_mask` で検査され `[B, 1, L, S]` を broadcast できる。行がすべて mask されると `reject_fully_masked_rows` が `Err` にする。非公開の `combine_masks_with_key_padding` と `forward_with_key_padding_mask` がすでにある | 同 `:1435`・`:1314`・`:1283`・`:1357`・`:901` |
| `KvSnapshot::capture` は各層を `(seq_len, batch, embed_dim)` で同一視する。複数層にわたる `forward_step` は層をまたいで原子的でない | `crates/autodiff/src/generate/kv_rewind.rs:66` |
| `Active` は要求ごとに `Vec<KvCache>`・`Generator`・語彙サイズを持ち、`advance` は B = 1 で prefill と decode を行う。要求単位の失敗は `take_failed` に積む | `crates/autodiff/src/generate/scheduler.rs:138`・`:153`・`:384` |
| facade のテスト用フィクスチャと `speculative.rs` は `caches[0].seq_len()` を位置として使う | `crates/facade/tests/speculative_batching_facade.rs:91`・`crates/autodiff/src/generate/speculative.rs:437` |
| K-3 のカーネル入力は行ごとの `seq_lens: [B]` を既に持つ | `docs/device-resident-kv-paged-attention-design.md` §3 |

### 21.3 選択肢の比較（論点 5）

可変長の 3 案（(1) padding + mask、(2) 行ごとの長さ配列、(3) 同じ長さのグルーピング）のうち (1) と (2) は対立しない。`[B, S_max, E]` の物理テンソルに行ごとの有効長を持たせれば、それが padding であり mask は長さから導ける。実際の比較軸は次の 4 案とする。

| 案 | 内容 | 公開面・不変条件 | 新 `Op` | 既存モデルとの互換（位置の計算） | K-3 との整合 | fail-closed | `BatchScheduler` への影響 | 評価 |
|---|---|---|---|---|---|---|---|---|
| **A** | 長さを `KvCache` の内部に持ち、mask は `forward_with_cache` が長さから導く（左詰め padding） | 公開は `valid_len` 1 件だけ。内部に有効長を足す（不変条件は 21.4） | 不要 | padding なしでは `seq_len()` が従来と同じ値。既存モデルは無変更 | `seq_lens` と同じ意味で合う | できる（長さと mask は内部で一致を保つ） | 一時的に束ねる形が作れる | **推奨** |
| B | 呼び出し側が自由形式の mask テンソルを渡す | mask の注入経路ができる | 不要 | 呼び出し側の責任 | 合わない | できない（長さと mask の食い違いを検出できない） | 利用者が mask を組む | §19.4 の「任意の `Tensor` を入れる経路を作らない」に反するため不採用 |
| C | padding を持たない ragged 表現（行ごとの別テンソル・packed） | 公開形が大きく変わる | varlen カーネル（新 `Op`）が要る | 1 つのテンソルとして forward できない | 別カーネル | 別途 | 作り直し | 不採用 |
| D | 同じ `S_cached` の要求だけを束ねる | 不変 | 不要 | 不変 | 不変 | できる | prompt 長が違う行は二度と揃わず、表現として不十分 | 単独では不採用。A の上では「全行が同じ長さなら padding なしの既存経路」として自然に含まれる |

### 21.4 推奨形の不変条件と mask 規則

- **padding は左詰め**とする（全行が物理位置 `S` で終わり、有効区間 `[S - len_r, S)` が連続したまま `cat(axis 1)` で追記できる。右詰めは行ごとの書き込み位置が要り `cat` が使えないので採らない）。
- `KvCache` に非公開フィールド `lens: Option<Vec<usize>>`（名前は #3105 で確定。`None` は padding なしで全行の長さが `S`）を足す。
- 新しい不変条件（維持するものを含む）:
  - `k`／`v` は両方 `Some` か両方 `None`（維持）。`[B, S, E]` は contiguous で `S` は物理長（維持）
  - `lens` が `Some` なら `k` も `Some`、`lens.len() == B`、全行で `1 <= lens[r] <= S`、`max(lens) == S`（全行が padding の先頭列を持たない）
  - padding 位置の K／V は `0.0`（有限値。重み 0 × 有限値 = 0 を保証する）
  - 空のキャッシュと `clear()` 後は `lens == None`
- 書き手: `forward_with_cache`（追記で全行の `lens[r] += L_new`）と、crate 内部の行の連結・分割（`pub(crate)`。算術なしのデータ並べ替えで、入力は既存の `KvCache` に限る）。公開のセッター・`truncate` は作らない。
- `seq_len()` は**物理長 `S`** の意味のまま変えない。padding がなければ従来と同じ値なので、既存テストとフィクスチャは無変更で通る。
- mask 規則（`pad_r = S_prev - lens[r]`。padding なしのときは従来の経路をそのまま使い bit 同一）:
  - (a) prefill: 空のキャッシュは `lens` を持たないので、padding 付きの prefill は構造上起こらない。従来どおり
  - (b) `L_new == 1`: `allowed[r][0][j] = j >= pad_r`。自分の token は常に有効なので、すべて mask される行は生じない
  - (c) `L_new > 1`: `allowed[r][i][j] = pad_r <= j <= S_prev + i`
  - いずれも `[B, 1, L, S_total]` を `sdpa_compose` の `attn_mask` へ渡す。`combine_masks_with_key_padding` を再利用するかは #3105 の裁量とし、意味だけをここで固定する
- 数値: -inf fill は非 mask 要素の値を変えない（`docs/kv-cache-design.md` §3.5）。ただし GEMM のブロッキングは形状に依存するため、padding あり対なしの bit 一致は保証せず、21.8 の判定へ帰着させる。

### 21.5 公開名・シグネチャ（追加のみ。名前は本節で確定する）

| 追加物 | シグネチャ | 公開の経路 |
|---|---|---|
| `KvCache` の inherent メソッド | `pub fn valid_len(&self, row: usize) -> Option<usize>`。空、または `row >= B` なら `None`。padding がなければ `seq_len()` と同じ | `nn::kv_cache` の既存 `pub use` で facade へ出る。`pub use` 文は増やさない |
| `BatchScheduler` の inherent メソッド | `pub fn step_batched<M: AutoregressiveModel + ?Sized>(&mut self, model: &M) -> Result<usize, AutodiffError>`。戻り値の意味は `step` と同じ（完了した要求数） | `inference` の既存 `pub use` で facade へ出る |
| crate 内部（非公開） | `pub(crate)` の行連結（B = 1 のキャッシュ列を左詰めで `[B, S_max, E]` にする）と行分割（各行を有効長へ切り詰め padding なしの B = 1 に戻す）。名前案は `stack_left_padded`／`split_rows` | 公開しない |

`AutoregressiveModel`（2 メソッド）・`SchedulerLimits::new`（3 引数）・`step`・`generate`・`generate_speculative` は変えない。新しい trait・型・`AutodiffError` の variant は足さない。

### 21.6 `BatchScheduler` への波及

- 第 1 段階の `step` はそのまま残す（token 列の完全一致の契約を維持する）。束ねる経路は別メソッド `step_batched` の opt-in とし、`step` の内部では束ねない。
- **正のキャッシュは要求ごとの B = 1**（`Active::caches` は padding を持たない）。`step_batched` 1 回の中で次の順に処理する。
  1. 待ち行列からの受け入れは `step` と同じ
  2. prefill 前の要求は B = 1 の `advance` で進める（第 1 段階と同一）
  3. decode 中の要求は、層ごとに一時的な `[B, S_max, E]` キャッシュを連結し、`[B, 1]` で `forward_step` を 1 回呼ぶ。戻り値 `[B, 1, V]` を検査し、行ごとに各要求の `GenerateConfig`・`Generator` で `sample_step` する
  4. 一時キャッシュを分割して各要求の B = 1 キャッシュへ戻す
- 持続的なバッチキャッシュ（参加・退出のたびに再編する形）は採らない。複数層にわたる `forward_step` が層をまたいで原子的でないため、失敗時に全行の状態が壊れ原因の行を切り分けられない。正を要求ごとに持てば、**完了した要求が抜けるときのキャッシュ再編は不要**（従来どおり外すだけ）で、参加も次回の連結に自然に含まれる。
- 失敗の契約（#3106 の作業へ先に渡す）: 束ねた forward・連結・戻り shape の検査が `Err` になったら一時キャッシュを捨てる。各要求の正のキャッシュは変わっていない。その step に限り対象の要求を 1 件ずつ B = 1 の `advance` でやり直し、失敗した要求だけを `take_failed` に積んで残りは継続する（forward が 1 回余分にかかる）。`step_batched` の `Err` は `step` と同じく、スケジューラ全体の不変条件が破れた場合のために取っておく。
- 対象モデルの契約（型でも実行時でも検出できないため利用者が保証する。§17.5 と同じ扱い）。`step_batched` を呼ぶのは次を満たすモデルに限る。
  - attention はすべて `forward_with_cache` 系の入口（#3096 の `Tape::stateful_attention_forward_with_cache` 等）を通す
  - 行ごとの位置は `caches[i].valid_len(r)` から決め、`seq_len()` は使わない
  - attention 以外は行ごとに独立した演算に限る（バッチ方向の統計を取る層を持たない）
  - 内部状態保持型は従来どおり対象外
- §8.3 の制約（同期実行・スレッド／I/O／`rayon` なし。grep で 0 件）は維持する。

### 21.7 `kv_rewind`・speculative への波及

- speculative は B = 1 限定のまま（B > 1 の speculative はスコープ外）。正のキャッシュは常に padding なしなので、`kv_rewind` の clone による退避・復元と `KvSnapshot::capture` の `(seq_len, batch, embed_dim)` による同一視は変えない。
- fail-closed を補うため、`capture` は padding 付きのキャッシュ（`valid_len(r) != seq_len()` の行がある）を `InvalidArgument` で拒否する。#3105 の「巻き戻しで行ごとの有効長を扱う」作業はこの拒否の追加と読み替える。層をまたいで原子的でないという既存の注記はそのまま当てはまる。

### 21.8 判定方式（§6.3 を具体化する。新しい数値は作らない）

- `step_batched` の各要求の出力を、同じ要求を単独で `generate`（B = 1）した結果と比べる。位置ごとの logits は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。既存の `assert_parity`）で判定し、token 列の一致は `Greedy` の要求に限り、上位 2 候補の margin を十分に取った入力でだけ確かめる（margin を取っても一致しなければ論点 2 へ戻す。所有者への承認依頼）。`TopK`／`Temperature` の要求では margin は token 列の一致を保証しない（padding による許容誤差内の微小な logits 差でも、同じ RNG 値が累積確率の境界をまたいで別 token を選びうる）。このため token 列の一致を契約にも事前登録の仮説にもせず、論点 2 へも戻さない。§20.7 と同様に、単独 `generate` が選んだ入力 id 列を `step_batched` へ再生して位置ごとの logits を `assert_parity` で比べる。分岐後の異なる文脈同士は比べない。
- 必須の検査: 全行が同じ長さのとき padding なしの経路と bit 一致する（#3105）／padding の量だけを変えた同じ要求の出力が一致する（#3107）／途中参加・途中退出・長さの混在で他の要求の出力が変わらない（#3107）／第 1 段階の既存テストが無変更で通る／実機は `#[ignore]` で分離する。
- tolerance・baseline は変えない。

### 21.9 上限（OWASP A04）

すべて確保の前に検査し、失敗したら原子的に何も変えない。

- 行数 `B <= max_active`（新しい上限フィールドは足さない。`SchedulerLimits::new` は変えない）。物理長 `S_max <= max_length`。
- `B × S_max × E × size_of::<f32>()` を K と V それぞれについて `checked_mul` で計算し、`isize::MAX` バイト以下であることを層ごとに確かめる。mask の `B × L × S_total` は既存の `checked_ls` 相当で検査する。logits `[B, 1, V]` は既存の `validate_forward_step_output`／`validate_vocab_le_i32_max` で検査する。
- 連結は、`B != 1` の入力・E の不一致・層数の不一致・`lens` の不変条件違反を `Err` にする。超過は 21.6 の B = 1 フォールバックへ回す。

### 21.10 既存公開シグネチャの非破壊確認（追加のみ）

| 項目 | 確認結果 |
|---|---|
| `KvCache` の 8 メソッド | 意味も不変。`seq_len` は物理長のままで padding なしでは従来と同じ値。追加は `valid_len` の 1 件 |
| `forward_with_cache`・`StatefulAttention`・`AutoregressiveModel`・`generate`・`generate_speculative` | 変更なし（`forward_with_cache` は非公開のまま） |
| `BatchScheduler` の既存メソッド・`SchedulerLimits::new`・`RequestId`・`SpeculativeConfig` | 変更なし。追加は `step_batched` の 1 件 |
| `pub use` 文 | 増やさない・変えない（`KV_CACHE_APPROVED_REEXPORT`・`inference` の 4 文・`GENERATE_APPROVED_REEXPORT` は不変） |
| `fandhe-ai =0.11.0` の既存シグネチャ・依存・tolerance・baseline・`unsafe` | 変更なし |

`KvCache` に非公開フィールドを足すと `Debug` の出力が変わる。安定 API ではないため受け入れる。

### 21.11 反転・更新する api_surface ガード（`crates/facade/tests/api_surface.rs`。名前と行は実装時に grep で確かめる）

| ガード | 扱い | 担当 |
|---|---|---|
| `kv_cache_public_methods_stay_unchanged_for_speculative_scheduler`（`:39657`） | 8 件 → 9 件（`valid_len`）。doc コメントに #3104 の決定を書く | #3105 |
| `kv_cache_is_reachable_via_facade_only`（`:8476`） | `valid_len` のシグネチャ・プローブを足す | #3105 |
| `speculative_scheduler_public_shape_violations`（`:39496`）／`speculative_scheduler_public_shape_is_pinned`（`:39606`） | `step_batched` を固定形へ足す | #3106 |
| `speculative_scheduler_items_are_reachable_via_facade_inference_path`（`:39363`）・`speculative_scheduler_usage_doctests_are_present_and_compiled`（`:39454`） | `step_batched` の到達プローブと利用例を足す | #3106／#3107 |
| §17.6 の否定プローブ「B > 1 と padding mask の経路がないこと」 | 実装箇所を grep で特定して記録する。scheduler 側は反転し、speculative の B > 1 の否定は残す | #3106／#3107 |
| `kv_rewind_stays_private_and_absent_from_facade`・`nn_kv_cache_module_reexports_exactly_expected_surface`・`facade_reexports_kv_cache_items_only_in_approved_shape`・`KV_CACHE_APPROVED_REEXPORT`・`inference_module_reexports_exactly_expected_surface` | 変えない（新しい型・`pub use` を足さないため） | — |

否定ガードは stable rustdoc の `compile_fail` コード照合に頼らず、正のプローブとインベントリで組む（§18.3 の方針を引き継ぐ）。

### 21.12 後続 issue への読み替え

- #3105: 表現は 21.4（左詰め＋非公開の `lens`）、公開追加は `valid_len` だけ。巻き戻しは「padding 付きキャッシュの拒否」と読み替える。
- #3106: `step_batched` と 21.6 の失敗の契約・対象モデルの契約。上限は 21.9。
- #3107: 21.8 の必須検査。フィクスチャのモデルは `valid_len` を位置に使う。

### 21.13 所有者へ戻す点

| 点 | 扱い |
|---|---|
| 所有者へ戻す点 | **なし**（6 つの論点すべてに名前付きで形を書けた。D-A に従う） |
| 残るリスク | `step_batched` の対象モデルの契約は型でも実行時でも検出できず、利用者が保証する（§17.5 と同じ扱い）。承認の依頼ではない |

実装中に 21.4 の mask 規則または 21.6 の失敗の契約をコードの事実から決められないと分かった場合は、「なし」とせずその点を本表へ載せて所有者へ戻す。

スコープ外: コード実装（#3105〜#3107）、B > 1 の speculative、padding 付きの prefill、paged attention／K-3（`docs/device-resident-kv-paged-attention-design.md`）、EOS 停止、論点 2・7。

### 21.14 セキュリティ観点

- A03: `KvCache` に任意の `Tensor` や mask を入れる経路を作らない。mask は内部の `lens` から導き、行の連結・分割は `pub(crate)` で入力は既存の `KvCache` に限る。形状・B・E・`lens` の不変条件を確保の前に検査する。
- A04: 21.9 の上限を確保の前に原子的に検査する。失敗した step は一時キャッシュを捨てて B = 1 でやり直し、黙って誤った出力を出さない。opt-in の別メソッドにすることで既存の `step` の契約を変えない。対象モデルの契約を守れない場合の誤出力は残るリスクとして明記した。
- A06: 依存・`unsafe` の追加なし。ネットワーク公開面なし（スケジューラは同期実行）。
- A08: 承認を代行しない（21.1）。ruleset・branch protection・リポジトリ設定は変更しない。
- A02: `Generator` は非暗号（xorshift64*）で、生成 token をセキュリティ用途に使わない既存の注記を引き継ぐ。
- 論点 2・3・5・7・8 は保留のまま（§17.7）。

## 22. 実装記録（#3096・親 #3094）

- 実装: §19.4 の 2 メソッド（`StatefulAttention::forward_with_external_cache`・`Tape::stateful_attention_forward_with_cache`。いずれも 1 行委譲）。承認根拠は §19.1 の D-A の範囲に限る
- §19.9 による読み替え: `KvCache` の公開メソッドは追加なし（8 件のまま）、巻き戻しフックは非公開の `kv_rewind` のまま
- §19.8 のうち担当が #3096 の行: `kv_cache_public_methods_stay_unchanged_for_speculative_scheduler` はアサーション無変更（doc のみ更新）。薄い委譲ガード・宣言インベントリ・`StatefulAttention` の `pub fn` 集合固定・`kv_cache_is_reachable_via_facade_only` への署名と実行プローブ・`kv_cache_usage_doctests_are_present_and_compiled` への必須トークンを追加した。autodiff 側メソッドの署名は第 1 引数が生の `Tape` で facade から名指しできないため、委譲ガード・インベントリ・実行プローブで補う
- 申し送り: §17.5 の rustdoc 置き換えとフィクスチャ移行は #3097、facade だけで書く成功例 doctest は #3098
- 依存・tolerance・baseline・`unsafe`・`docs/spec` は変更なし

## 23. 実装記録（#3097・親 #3094）

§19.6・§19.9 の読み替えに従い、内部状態保持型を generate 系へつなぐ経路を「rustdoc の契約更新・facade だけで組むフィクスチャ・一致テスト」として実装した。承認根拠は §19.1 の D-A の範囲に限り、それ以上の承認は主張しない。

- 状態分離: 新しい仕組みは作らない。従来 `RefCell<StatefulAttention>` で書いていた KV 付きモデルを、#3096 の入口 `Tape::stateful_attention_forward_with_cache` で `caches[i]` を進める形（`num_kv_layers() > 0`）に書き直せば、要求ごとの `Vec<KvCache>`（`BatchScheduler`）と非公開 `kv_rewind`（`generate_speculative`。案 (i)）がそのまま状態分離・巻き戻しを担う。
- 作業 2「`num_kv_layers() == 0` 拒否の見直し」は §19.6 により行わなかった（検査順 1〜9 は不変。facade だけで組める形が `> 0` を返すため緩める必要がない）。
- rustdoc 更新: `crates/autodiff/src/generate/scheduler.rs`（モジュール doc・`BatchScheduler` doc）、`crates/autodiff/src/generate/speculative.rs`（`generate_speculative` の契約節）、`crates/facade/src/inference/mod.rs`（generate 節の「既知の制限」を「KV 付きモデルの組み方」へ、speculative 節の契約・「既知の制限（保留論点 3）」を新契約へ）。内部状態保持型の定義と「対象外・利用者が保証」は維持。doctest のコードブロックは変更していない。
- フィクスチャ移行: `crates/facade/tests/speculative_batching_facade.rs` の `TableLm` を facade 公開面だけ（`StatefulAttention` を層ごとに持ち `stateful_attention_forward_with_cache` で `caches` を進める形）で組み直し、`fandhe_ai_autodiff` の import を 0 にした。既存 6 テストのアサーションは不変。`generate_facade.rs` の `CachedLm`（RefCell 形）は後方互換の検査として残し、コメントだけを更新した。
- 追加テスト: (1) `scheduler_interleaves_caches_models_like_standalone_generate`: `caches` 形の表引きモデル（1・2 層）と attention 依存 logits モデルで、1 つのモデルを共有し `max_active` 未満の枠で 3 要求を交互に進め、単独 `generate` と token 列が完全一致（スケジューラ経路は単独と同じ forward 形状のため）。完了後に `StatefulAttention` の内部キャッシュ長が 0 のままであることも確認。(2) `speculative_matches_generate_on_attention_caches_model`: attention 依存 KV モデルで、トークン選択位置の logits を REQ-2 統一複合判定（`assert_parity`。tolerance・baseline 不変）で単独 `generate` と突き合わせ、事前登録の仮説 H1（margin 前提下の token 列一致）も確認。検証 forward（`L_new > 1`）と rewind 後の再 forward が 1 回以上起きたことを fail-closed で確認。`SKIP_C = 3.0` は事前固定。
- H1 の結果: 掃引（layers 1・2、k 1・3、T 1・3、draft 3 種）で token 列は全件一致し、破れは観測されなかった。
- `scheduler.rs` の `std::thread|std::sync::mpsc|tokio|async |std::net|std::fs|rayon` の grep は 0 件。
- 変えないもの: tolerance・baseline・依存・`unsafe`・`docs/spec/`・公開シグネチャ（新しい `pub` 項目・`pub use`・trait メソッド・`AutodiffError` variant なし）・`SchedulerLimits` の 3 上限必須・要求単位の失敗の切り離し。
- 申し送り（#3098）: 成功例 doctest（`inference/mod.rs`）、`api_surface.rs` の doctest 実在検査の更新、`k`・draft・prompt 長・`max_length` の掃引拡張、§18.4 追記。`caches` で表せない内部状態（RNN 隠れ状態等）を持つモデルは §19.10 のとおり案 B を起点に別 issue（起票はユーザー承認後）。

## 24. 実装記録（#3105・親 #3103）

- 実装: §21.4 の表現。`KvCache` に非公開フィールド `lens: Option<Vec<usize>>` を追加（`None` = padding なし。`Some` は左詰め padding で、`max(lens) == S`・少なくとも 1 行が `lens[r] < S`。全行が `S` なら必ず `None` へ正規化し、全行同長なら従来経路と bit 一致を構造的に保つ）。padding 位置の K／V は 0.0
- 公開追加は `KvCache::valid_len(row) -> Option<usize>` の 1 件のみ（§21.5）。空・範囲外は `None`。`seq_len()` の意味は物理長 `S` のまま
- mask: 非公開関数 `padded_offset_allowed_mask(lens, s_prev, l_new, s_total)`（`[B,1,L_new,S_total]`・`true` = attend・`pad_r <= j && j <= s_prev + i`）。`forward_with_cache` は `lens` が `Some` のとき (b)／(c) とも本関数を使い、`None` のときは従来コードをそのまま使う。`lens + l_new_kv` は射影の前に `checked_add` で計算し、書き戻しは全段成功後
- 束ね・分割: `KvCache::stack_left_padded`／`split_rows` は `pub(crate)` で、`#[cfg(test)]` の子モジュール `nn/attention/kv_padding.rs` に置く（本イシュー時点で非テスト呼び出し元がなく、`#[allow]` を避けるため。api_surface の `without_test_mod` が最初の `#[cfg(test)]` 以降を切り落とすので、本体 impl の途中には cfg 属性を置かない）。**#3106 で `cfg(test)` を外した（§26）**
- kv_rewind（§21.7）: `common_state`（`capture`・`rewind_and_replay` が経由）が `valid_len(r) != Some(seq_len())` の行を持つキャッシュを `InvalidArgument` で拒否する。新しい内部 API は足していない
- ガード: `kv_cache_public_methods_stay_unchanged_for_speculative_scheduler` の期待集合を 9 件（`valid_len` 追加）へ更新。`kv_cache_is_reachable_via_facade_only` に署名固定と実行プローブを追加
- テスト: 可変長の B=1 一致（L_new=1・2。統一複合判定）・padding 内容の非影響（bit 一致）・全行同長の bit 一致往復・拒否時の状態不変・`valid_len`・mask 単体
- 依存・tolerance・baseline・`unsafe`・`docs/spec` は変更なし

## 25. 実装記録（#3101・親 #3099）

サンプリング版 `generate_speculative`（§20 の選択肢 B の契約に従う実装）。公開シグネチャ・`SpeculativeConfig`・`pub use` は不変で、`config.strategy` が `TopK`／`Temperature` のとき speculative sampling で進む。Greedy は従前のコードパスのまま（RNG を作らない）。

### 25.1 実装形

- `generate/mod.rs`: 非公開 `sampling_weights_f32`（TopK は上位 k 外を 0.0、Temperature は全語彙の `softmax_weights_f64`）を新設し、`sample_step` と共有（演算順は不変で `generate` は bit 同一）
- `generate/speculative.rs`: 検査 2（Greedy 以外の拒否）を削除し Errors を 1〜8 へ繰り上げ（順序は不変）。`row_at`（行の取り出しと有限性検査）・`normalize_f64`・`acceptance_prob_f32`・`residual_weights_f32`・`sample_from`・`accept_draw`・`verify_sampling` を非公開で追加
- `q(x)` は draft が引いたときの f32 重みを保存し、`p`・`q` とも `normalize_f64`（index 順 `f64` 総和）で正規化して読み直す。受理確率は `[0,1]` に clamp して `f32` 化（`p == q` で厳密 1.0）。`q(x) <= 0`・残差和 0／非有限・正規化総和 0 は RNG を引く前に `Err`

### 25.2 RNG 消費順（§20.4 の確定）

prefill 末尾 multinomial × 1 → ラウンドごとに draft multinomial × `k_eff` → 検証位置ごとに bernoulli × 1（最初の棄却まで）→ 棄却時は残差 multinomial × 1、全受理時は bonus multinomial × 1 → 最終 1 step で multinomial × 1。`Generator::new(config.seed)` は呼び出しごとに 1 個で、`max_length == prompt_len` の早期 return では作らない。

### 25.3 検査項目 C1〜C6 の置き場所

C1・C3・C4・C5・C6 は `speculative.rs` のユニットテスト、C2 は `crates/autodiff/tests/nn_generate_speculative.rs`。分布検定は契約外（§20.6）で追加していない。

### 25.4 既存テストの扱い（Issue 文言との差分）

Greedy に関するアサーションは無変更。外した検査の否定表明だけを反転した: `speculative_batching_facade.rs` の `TopK(3)`／`Temperature(0.8)` は `Ok`（shape `[6]`）に、代わりに `TopK(V + 1)` が `InvalidArgument` になることを追加。`speculative.rs` 内では `rejects_invalid_inputs_before_any_forward` から topk／temp の 2 行を削除し、`validation_order_is_fixed` の期待 needle を `SpeculativeConfig.k` へ変更。`api_surface.rs` は別入口を作らない旨のコメント更新のみ。

### 25.5 変えないもの

tolerance・baseline・判定規則・依存・`unsafe`・`docs/spec`・`generate()` の挙動。facade レベルの拡充と実機 `#[ignore]` テストは #3102。

## 26. 実装記録（#3106・親 #3103）

承認の根拠は D-A（§21.1）の範囲に限る。§21.5・§21.6 の形で `BatchScheduler::step_batched` を追加した（追加のみ）。

- 公開形: `pub fn step_batched<M: AutoregressiveModel + ?Sized>(&mut self, model: &M) -> Result<usize, AutodiffError>`。新しい trait・型・`AutodiffError` の variant・`pub use` は足していない（`inference` の既存 `pub use` で facade へ出る）
- 実装形: 受け入れ（`admit`）と token 追加後の完了判定（`Active::push_and_maybe_finish`）を非公開ヘルパーへ切り出し、`step`／`advance` の挙動は変えずに共有した。束ねは副作用のない試行 `try_batched_decode` と、全段成功後の commit に分けた。decode 中の要求が 2 件未満なら B = 1 の `advance` と同一
- 上限検査の置き場所（§21.9）: `B <= max_active`・層ごとの `seq_len() <= max_length` は `scheduler.rs` 側で `stack_left_padded` の呼び出し前に検査する。`stack_left_padded` のシグネチャは不変で、`B x S_max x E x 4` の `checked_mul`／`isize::MAX` バイト検査は同関数内にある。`kv_padding` の `cfg(test)` を外し、`lens_for_test` はファイル末尾の `#[cfg(test)] impl` へ移した
- 失敗契約の採否: §21.6 のとおり、束ねた forward・連結・戻り shape／語彙・サンプリング・分割のいずれかが `Err` なら試行全体を捨て、その step だけ B = 1 でやり直して失敗要求だけを `take_failed` に積む（「全要求を失敗させる」案は採らない）。サンプリングは行ごとに複製した RNG で行い、commit 前なので副作用はない
- ガード更新: `speculative_scheduler_public_shape_violations` の `impl BatchScheduler` を 8 件へ、署名固定・`?Sized` プローブ・束ねる経路の doctest プローブを追加し、変異テストを足した。`kv_cache_public_methods_stay_unchanged_for_speculative_scheduler`（9 件）等は不変
- §17.6 の否定プローブ「B > 1 と padding mask の経路がないこと」: 実体は (a) `speculative.rs` の自由な `pub fn` が `generate_speculative` 1 件のみというインベントリと、(b) `submit` の B > 1 拒否である。どちらも残した（speculative の B > 1 否定と `submit` の B > 1 拒否は不変）。scheduler 側の反転は上の正プローブで表し、削除したガードはない
- テスト（`scheduler.rs`）: 束ねが実際に起きること（decode が `[B, 1]` の 1 回だけ）・状態なしモデルの完全一致・KV 全行同長の完全一致・KV 長さ混在の logits を単独 `generate` と統一複合判定で比較（Greedy は margin を fail-closed で確認し、padding を伴う呼び出しが 1 回以上あることも確認）・束ね失敗時のフォールバック・要求単位の切り離し・層数変更・空／prompt のみ・`step` との交互呼び出し。網羅テスト（margin 掃引・TopK／Temperature の再生比較）と実機 parity は #3107 の担当
- 対象モデルの契約は型でも実行時でも検出できず、利用者が保証する（§17.5 と同じ扱い）
- `scheduler.rs` に対する `std::thread|std::sync::mpsc|tokio|async |std::net|std::fs|rayon` の grep は 0 件。依存・tolerance・baseline・`unsafe`・`docs/spec` は変更なし

## 27. 実装記録（#3102・親 #3099）

テストと docs のみの変更。`crates/**/src`・依存・tolerance・baseline・判定規則・`docs/spec` は変更しない。

- 追加したテスト（`crates/facade/tests/speculative_sampling_facade.rs`。facade の公開面だけを使い、内部クレートは import しない。別バイナリに分け、全テストが `GLOBAL_RNG_LOCK` を取る）:

  | テスト | 契約 |
  |---|---|
  | `sampling_matches_exact_f64_reference_replay` | C3（受理確率・残差分布・消費順の統合） |
  | `sampling_is_deterministic_for_same_seed`・`different_seeds_can_change_output` | C1 とシード差 |
  | `sampling_does_not_consume_global_rng` | C2 |
  | `identical_draft_accepts_everything` | C4 |
  | `top_k_one_and_one_hot_logits_match_greedy` | C5 |
  | `sampling_fails_closed` | C6 |

- 分布検定は §20.6 により契約外のため追加しない（閾値つき統計量は足していない）
- 厳密入力の選定理由: logits を集合上 `0.0`・外 `-1e4`、集合の大きさを 2 のべき乗に限ると、`Generator` へ渡る `f32` 値（サンプリング重み・受理確率 0／0.5／1・残差重み）が libm・計算順序・プラットフォームによらず bit 単位で一意に決まる。再生オラクル `reference_speculative` は同じ `Generator` を §25.2 の消費順で引く独立実装で、`generate_speculative` と token 列の完全一致を固定掃引（5120 通り）で要求する。draft 規則 5 種で受理確率 0／0.5／1・残差・bonus の全経路を通ることを統計で確認する
- 実機: `speculative_batching_backend_parity.rs` に §20.7 の再生方式（参照側で記録した id 列を比較側へ再生し logits だけを `assert_parity` で比べる。token 列は比べない）を追加した。属性なしの CPU 対 Naive（`cpu_backend_ops_matches_naive_ops_for_speculative_sampling`）と再生ハーネスの検出力テストは CI 実行、`cuda_speculative_sampling_matches_cpu`・`metal_speculative_sampling_matches_cpu` は `#[ignore]` で、実機（GB10／Mac mini）で pass 実測済み（2026-10-27。`docs/perf/logs/speculative-sampling-3102/README.md`）
- 実機テストを #3171（CUDA）・#3172（Metal）の実測対象に加えるかは所有者の判断事項
- 2026-10-27 の reopen（所有者判断）で、この実機テストを測定対象に加えると確定した。実測結果は `docs/perf/logs/speculative-sampling-3102/README.md`

## 28. 実装記録（#3107・親 #3103）

承認の根拠は D-A（§21.1）の範囲に限り、新しい承認は主張しない。テストと docs の追加だけで、`crates/*/src`・依存・`unsafe`・`docs/spec` は変更していない。

### 28.1 追加したテストと §21.8 必須検査の対応

| 検査（§21.8） | テスト |
|---|---|
| prompt 長・`max_length` が違う要求の出力が単独 `generate` と一致（logits は統一複合判定、token 列は margin 付き Greedy） | `step_batched_lockstep_mixed_lengths_matches_standalone_generate`（`speculative_batching_step_batched_facade.rs`） |
| padding の量だけを変えても出力が変わらない | `step_batched_padding_amount_does_not_change_output` |
| 途中参加・途中退出・長さ混在で他の要求の出力が変わらない | `step_batched_mid_join_and_leave_do_not_change_other_outputs` |
| TopK／Temperature は分岐前の logits だけ比べる | `step_batched_sampling_requests_match_before_divergence` |
| 形状に依らない表引きモデルは全戦略で token 列完全一致 | `step_batched_table_model_matches_exactly_for_all_strategies` |
| テストが決定的（同形状どうし bit 一致） | `step_batched_is_deterministic` |
| 実機は `#[ignore]` で分離 | `cuda_scheduler_batched_greedy_matches_cpu`・`metal_scheduler_batched_greedy_matches_cpu`（`speculative_batching_backend_parity.rs`。属性なしの CPU 対 Naive 版は CI で実行） |

### 28.2 設計上の判断

- フィクスチャの契約: 位置は `valid_len(r)`（`seq_len()` は使わない）、戻りは `[b, l, V]`。B 非対応モデルは `step_batched` が黙って B = 1 へ戻るため、束ねを検証するテストはすべて `b > 1 && l == 1` の呼び出しが実在することを検査する（空振り防止）。lockstep では decode の k 回目の行を `valid_len == T + k` で要求へ対応づけ、対応づけを fail-closed にした
- TopK／Temperature は token 列を主張せず、単独 `generate` と出力が一致している間の位置の logits だけ統一複合判定で比べる。最初の decode 位置は prefill が B = 1 で同形状のため必ず比較される
- 途中参加・退出の計画では、スケジュール順に依らず decode 行を（位置 = `valid_len`・入力 id）で要求へ対応づけ、位置ごとの logits を単独 `generate` の行と統一複合判定で比べる（全 decode 位置の被覆と、全行がいずれかの要求へ対応することを fail-closed に検査）。加えて出力水準の比較（単独との一致と、他要求の構成を変えても同じ R の出力）も残す
- bit 一致は主張しない（§21.4。GEMM のブロッキングは形状に依存する）。決定性テストは同形状・同入力の再実行だけを bit 比較する
- margin 前提（`assert_margin`）は REQ-2 定数から導いたフィクスチャの妥当性検査で、新しい tolerance ではない。`SKIP_C` は変更していない
- H1（バックエンドをまたぐ token 列一致）が破れた場合は判定を緩めず §10 論点 2 として承認依頼に戻す

### 28.3 変えないもの・申し送り

tolerance・baseline・判定規則・依存・`unsafe`・`docs/spec` は不変。実機（CUDA／Metal）のバッチ経路は未実測で、`docs/perf/logs/speculative-batching-facade-2934/README.md` の「#3107 追記」に測定コマンドと空の記入欄を置き、Phase 8（#3171・#3172）へ申し送る。対象外: B > 1 の speculative、padding 付き prefill、持続的なバッチキャッシュ、paged attention との結線、EOS 停止。
