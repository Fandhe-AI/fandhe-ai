# RNN／LSTM／GRU の多層・双方向・dropout（`RnnConfig`）設計判断記録

イシュー #2164・親 #2131。

## §0 結論

`fandhe_ai_autodiff::nn`（内部クレート）に `RnnConfig`（`num_layers`・
`bidirectional`・`dropout` の 3 オプション）と、これを受け取る新しい型
`StackedRnn`／`StackedLstm`／`StackedGru`（Sequence レベル。単層・単方向の
既存 `Rnn`／`Lstm`／`Gru` とは別型）を実装した。`RnnConfig::default()`
（`num_layers=1`・`bidirectional=false`・`dropout=0.0`）で構築した
`Stacked*` は既存の `Rnn`／`Lstm`／`Gru` と bit 完全一致する。facade
（`fandhe_ai`）への公開は #2164 時点では未承認のまま対象外とし、
`RnnConfigHoldDoctestGuard` ＋`api_surface.rs` の 2 テストで保留を多層固定
した。**#2535 で facade 公開済み**（§9 実装記録。ルート #2499 の一括承認）。

## §1 背景

#1955 で RNN 系（`Rnn`／`Lstm`／`Gru`）を facade に公開したが、提供して
いるのは単層・単方向のみで、PyTorch `nn.RNN`／`nn.LSTM`／`nn.GRU` の
`num_layers`・`bidirectional`・`dropout`（層間）に相当する表現力がない。
既存の `Rnn` を `forward_seq` で単純に重ねると、層 2 以降の入力が
`Tensor` へ detach され前段層への逆伝播が切れる
（`docs/autodiff-rnn-cell-tape-design.md` 決定 4a (ii)）。

## §2 設計

### 2.1 既存型に手を入れない理由

`crates/facade/src/nn/rnn.rs` は既存 8 型（`Rnn`／`Lstm`／`Gru` を含む）
を純再エクスポートしている（#1955）。既存型へ inherent メソッド（例
`Rnn::with_config`）を追加すると、facade の公開面が引数型を facade から
名指しできなくても `Default::default()` の型推論経由で自動的に広がる。
本 issue は facade 公開を未承認事項として保留する（§8）ため、多層化は
既存型に触れず新しい型（`StackedRnn`／`StackedLstm`／`StackedGru`）で
提供する。`crates/autodiff/src/nn/rnn.rs` の公開シグネチャ・
`RnnCell`／`LstmCell`／`GruCell`・`Rnn`／`Lstm`／`Gru` 自体は一切変更
していない（`pub(super)` 化した 5 個の private ヘルパーのみ可視性を
上げ、ロジックは不変）。

### 2.2 Sequence レベルのスタックと決定 4a の関係

`docs/autodiff-rnn-cell-tape-design.md` 決定 4a (i) は「セル単位で
per-step に交互適用すれば勾配が連続する」ことを示している。
`StackedRnn`／`StackedLstm`／`StackedGru::forward_seq` はこの方針に従い、
`RnnCell`／`LstmCell`／`GruCell::bind` を各 `(layer, direction)` ごとに
1 回だけ呼び、層 1 以降の入力には前層の出力 `Var` を `Tensor` へ detach
せずそのまま渡す（`forward_seq` 内部で完結し、利用者が手組みする必要が
ない）。双方向（決定 4a の対象外として列挙されていた項目）も本 issue で
実装した。

### 2.3 セル配置・シード導出

`cells: Vec<XCell>`（`X` は `Rnn`／`Lstm`／`Gru`）を
`index = layer * num_directions + direction`（PyTorch の `h_n` と同順）
で保持する。シード導出は `crates/autodiff/src/nn/init.rs::
RNN_STACK_SEED_SALT`（新規ソルト値 `11`。既存の `0..=10` と衝突しない）
を使い、index `k` のセル構築シードは
`derive_seed(derive_seed(seed, RNN_STACK_SEED_SALT), k)` という 2 段の
`derive_seed` 合成で導出する（`ENC_ATTN_SEED_SALT` 等と同じ構造）。
ただし index 0（layer 0・forward 方向）だけは `seed` をそのまま渡し
2 段合成を適用しない——これにより `RnnConfig::default()` で構築した
`StackedRnn::new(.., seed, ..)` が `Rnn::new(.., seed)` と bit 一致する
（`nn::rnn_stacked::tests::stacked_rnn_default_config_bit_matches_rnn`
等が検証）。

### 2.4 層間 dropout

`layer < num_layers - 1` の出力にのみ `Var::dropout(p, training)`
（`Var::dropout` 自体の早期リターン: `!training || p == 0.0` で新規ノード
を積まずグローバル RNG も消費しない）を適用する。RNG 消費順は
「layer 昇順 → t 昇順」（forward_seq のループ順そのまま）。
`num_layers == 1` かつ `dropout > 0` は PyTorch と同じく受理し、
最終層（かつ唯一の層）には適用されない no-op として扱う（エラーには
しない）。

### 2.5 双方向の結合

`Var::cat(&[fwd[t], rev[t]], 1)`（tape 経路）／
`crate::grad::concat_with_fallback`（`forward_host`。tape 不要経路）で
時刻ごとに `[B, 2H]` へ結合する。単方向では cat を挟まず forward 方向の
出力をそのまま使う（既定 config での bit 一致契約のため）。

### 2.6 命名契約

`Module::named_parameters` は `l{layer}.`（forward 方向）／
`l{layer}_reverse.`（reverse 方向）を接頭辞として使う（PyTorch の
`weight_ih_l0`／`weight_ih_l0_reverse` の `_l{layer}[_reverse]` 部分に
対応するが、本クレートの命名契約〈struct フィールド名ベース〉に従う
ためキー文字列としては PyTorch と一致しない）。`set_parameter` は
この接頭辞をパースして該当セルへ委譲する。

## §3 契約（不変条件）

- tolerance・baseline・`Cargo.toml` 依存・ガードレール閾値・
  `docs/spec/` は変更していない。
- 新規 `unsafe` は追加していない。
- 公開済み `fandhe-ai =0.9.0` の API を破壊していない（追加 API のみ）。
- 新規演算は既存 Op（`rnn_cell`／`lstm_cell`／`gru_cell`／`Concat`／
  `Dropout`）の合成のみで到達する。GPU 専用カーネルは追加していない。

## §4 テスト

- `crates/autodiff/src/nn/rnn_stacked.rs` 内の単体テスト（18 件）:
  `RnnConfig` の検証・既定 config の bit 一致（3 型）・多層の勾配連続性・
  双方向の shape／順序・`forward_host` と `forward_seq` の eval モード
  一致・eval モードでの dropout no-op（RNG 非消費）・入力検証（h0 長さ
  不一致）・`named_parameters`／`set_parameter` の layer 接頭辞契約・
  `set_requires_grad` の全セル伝播・LSTM の `c_n` 長さ。
- 既存の `nn_rnn.rs`（26 件）・`nn_dropout.rs`（15 件）・
  `nn_module_mode.rs`（16 件）はすべて非後退で pass。
- facade 側は `RnnConfigHoldDoctestGuard`（doctest）＋
  `crates/facade/tests/api_surface.rs::rnn_config_hold_doctest_globs_
  all_pub_modules`／`rnn_config_hold_doctest_probe_body_matches_fixed_
  contract` の 2 テストで facade 未公開を固定する。既存の
  `nn_rnn_module_reexports_exactly_expected_surface`・
  `compat_sequential_does_not_expose_rnn_add_methods` は不変のまま
  pass（`RnnConfig`／`Stacked*` を facade の再エクスポート集合へ
  追加していないため）。
- `crates/facade/tests/rnn_stacked_backend_parity.rs`（新規）:
  `StackedRnn`（forward・backward）／`StackedLstm`／`StackedGru`
  （forward）の L=2・双方向を `CpuBackendOps` と `NaiveOps` で
  REQ-2 複合判定により突き合わせる。`StackedRnn` の CUDA／Metal
  `#[ignore]` テストを対称に実装済み（実機実測は §7 参照）。

## §5 スコープ外

- facade への `RnnConfig`／`Stacked*` の公開（§8「承認事項」）。
- `compat::Sequential::add_rnn`／`add_lstm`／`add_gru`: #1955 の承認済み
  決定「選択肢 C」（`Sequential::add_*` は追加しない）により設けない。
  イシュー #2164 の本文「対象範囲」節が `compat/sequential.rs` の
  `add_rnn` 等 config 対応を挙げているが、これは既存の承認済み決定
  および `compat_sequential_does_not_expose_rnn_add_methods` 否定ガード
  と矛盾するため実施しない（本文との食い違いとして記録する）。
- GPU 専用カーネル（別 issue）。
- `from_cells`（外部重みの持ち込み。層間 shape の整合検証が別途必要）・
  `batch_first`・`proj_size`（LSTM）・`nonlinearity='relu'`・可変長系列・
  truncated BPTT・ONNX export 対応。
- CUDA（DGX Spark GB10）・Metal（M4 Max）実機での parity 実測:
  `docs/perf/logs/rnn-stacked-2164/README.md` へ申し送る。

## §6 セキュリティ（OWASP Top 10）

- **A03 インジェクション／入力検証**: `RnnConfig::validate` で
  `num_layers >= 1` と `dropout` の有限性・範囲を構築時に検査する。
  `num_layers*num_directions`・`num_directions*hidden_size` は
  `checked_mul`、`cells` の確保は `try_reserve_exact`、`h0`／`c0` の
  長さ検証を確保より前に行う。本番経路で `unwrap`／`expect`／panic は
  使わない。
- **A04 安全でない設計**: 既存の facade 再エクスポート型に inherent
  メソッドを追加しない（§2.1）ことで未承認の公開面拡張を構造的に防ぎ、
  doctest の正のプローブ＋ソース走査の二重ガードで固定する。
- **A06 脆弱なコンポーネント**: 依存の追加・更新はない
  （`Cargo.toml`／`Cargo.lock` は不変）。
- **A08 データ整合性**: tolerance・baseline は不変。dropout の RNG
  消費順を doc で固定し再現性を保証する。
- **unsafe**: 新規追加なし。

## §7 実機申し送り

CUDA（DGX Spark GB10）・Metal（M4 Max）実機での `Stacked*` の
バックエンド parity は本エージェント実行環境に実機がないため未実測。
`docs/perf/logs/rnn-stacked-2164/README.md` に実行コマンドを記録して
申し送る。

## §8 承認事項（未承認・実施しない）

1. facade への `RnnConfig`・`StackedRnn`／`StackedLstm`／`StackedGru`・
   `StackedRnnSeqOutput`／`StackedLstmSeqOutput` の公開、および
   `Tape::stacked_rnn_forward_seq`／`stacked_lstm_forward_seq`／
   `stacked_gru_forward_seq` の委譲メソッド新設
   （`docs/compat-api-scope.md` §5 経路 2）。
2. 上記の承認が得られた場合に限り、
   `nn_rnn_module_reexports_exactly_expected_surface` の期待集合を
   更新し、`RnnConfigHoldDoctestGuard`・対応する `api_surface.rs` の
   否定ガードを正ガードへ置き換える。

## §9 実装記録（イシュー #2535・2026-10-04・親 #2534・ルート #2499 の一括承認）

§8 の推奨形（1 案に確定）をそのまま実施した。§8 本文は履歴として残す。

### 公開した名前

- `fandhe_ai::nn::rnn`（`crates/facade/src/nn/rnn.rs`）へ 6 型を純再エクスポート
  （既存 8 型と合わせ 14 型）: `RnnConfig`・`StackedRnn`・`StackedLstm`・
  `StackedGru`・`StackedRnnSeqOutput`・`StackedLstmSeqOutput`。
- `Tape` の委譲メソッド 3 件（`&self.0` を渡すだけ。`crates/facade/src/lib.rs`）:
  - `stacked_rnn_forward_seq(&StackedRnn, &Tensor<f32>, Option<&[Var]>) ->
    Result<StackedRnnSeqOutput<RnnCellVars>, AutodiffError>`
  - `stacked_lstm_forward_seq(&StackedLstm, &Tensor<f32>, Option<&[Var]>,
    Option<&[Var]>) -> Result<StackedLstmSeqOutput, AutodiffError>`
  - `stacked_gru_forward_seq(&StackedGru, &Tensor<f32>, Option<&[Var]>) ->
    Result<StackedRnnSeqOutput<GruCellVars>, AutodiffError>`
  - `h0`／`c0` は長さ `num_layers * num_directions`（index = `layer *
    num_directions + direction`）。

### ガード反転

- `RnnConfigHoldDoctestGuard` を縮小: 公開済みの `RnnConfig`／`Stacked*`・
  `Tape` メソッドのプローブを撤去し、未承認の `Rnn`／`Lstm`／`Gru::with_config`
  禁止のみを残した（`PixelShuffleHoldDoctestGuard` の #2526 縮小形と同型）。
  `RNN_CONFIG_HOLD_PROBE_BODY` も同期更新。
- `api_surface.rs` の正ガード: `nn_rnn_module_reexports_exactly_expected_surface`
  （期待集合 8 → 14）・`nn_rnn_stacked_types_are_reachable_via_facade_only`・
  `tape_stacked_rnn_methods_are_thin_delegations`・
  `workspace_declares_stacked_rnn_tape_fn_names_only_in_facade_lib`・
  `workspace_declares_no_rnn_with_config_fn`。
  `MIN_KNOWN_PROBE_BLOCKS` は 20 → 19（プローブモジュール 1 件減）。

### 検証

- `crates/facade/tests/nn_rnn_stacked_facade_bit_identity.rs`: facade 経路と内部
  クレート直接経路（`CpuBackendOps`）の forward・backward bit 一致（L=2 双方向・
  L=3 単方向・h0/c0 明示・dropout=0.5 同シード・既定 config と facade `Rnn` の
  bit 一致）。
- CUDA／Metal 実機 parity は §7 の申し送りがそのまま有効（新設せず）。

### 既知の制限と対象外

- facade からは `set_training(false)`（eval モード）へ到達できない
  （`Module` trait 非公開）。`dropout > 0` の facade 経由 forward は常に学習
  モード。推論用途は `dropout = 0.0` で構築する（`Tape::stacked_*_forward_seq`
  doc に明記）。
- 引き続き対象外: `Rnn`／`Lstm`／`Gru::with_config`（§2.1）・
  `compat::Sequential::add_rnn` 等（#1955「選択肢 C」）・`Stacked*` の facade
  `nn::Module` 実装・`from_cells`／`batch_first`／`proj_size` 等。
- 不変: `Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/`・
  新規 `unsafe` なし・新規 Op／カーネルなし。

## §10 イシュー #2536 の扱い（承認依頼・未承認・実施しない）

イシュー #2536 の受け入れ条件のうち、(1)「`nn_rnn_module_reexports_exactly_expected_surface`
の期待集合更新」は #2535（PR #2727）で実施済み（§9「ガード反転」。期待集合 8 → 14）。
残る (2)〜(4) は §8 の承認範囲（ルート #2499 の一括承認が及ぶのは記録に形が書かれた
推奨形のみ）の外であり、§5・§9「既知の制限と対象外」および #1955 の承認済み決定
「選択肢 C」と矛盾するため、本記録の時点では**実施しない**。承認は未取得である。

### 実施しない項目と根拠

| 項目 | 根拠 |
|------|------|
| `compat::Sequential` への Stacked 系 `add_*` と学習経路（`bind`・`trainable_parameters` 等）への結線 | §5・§9 が対象外と明記。`crates/facade/src/nn/rnn.rs` のモジュール doc が構造的不整合（入力が `&Tensor<f32>` `[T,B,D]`・隠れ状態の引き回し・複数 `Var` を返す戻り値）を説明。`impl Module for Stacked*`（`crates/autodiff/src/nn/rnn_stacked.rs`）の `forward` は設計上 `Err` で、Sequential の層として積んでも forward できない。`add_stacked_*` という別名での追加は否定ガード `compat_sequential_does_not_expose_rnn_add_methods` の字面上は検出されないが、趣旨上の抜け道であり採らない |
| `nn::Module` への `as_*` フック追加・`save_model`／`load_model` 対応 | `as_*` は autodiff 内部 `Module`（`crates/autodiff/src/nn/module.rs`）のフックで、facade の `nn::Module`（`crates/facade/src/nn/module.rs`）は REQ-12 により内部フックを載せない。§9 は `Stacked*` の facade `nn::Module` 実装を対象外とする。保存経路は `crates/facade/src/compat/model_io.rs`（`&Sequential` のみ）で、`add_module` の利用者定義層は既に `ModelIoError::UnsupportedModel` で fail-closed |
| resident 経路の未対応層拒否テスト | 上記 2 項目が前提（Sequential に載せない限り対象層が存在しない） |

### イシュー本文との食い違い

- 対象として挙がる `crates/facade/src/model.rs` は事前学習済みモデルのハブ読み込みで、
  保存経路ではない（正は `compat/model_io.rs`）。
- 同種の食い違い処理の先例: §5（#2164）・`docs/compat-fit-sample-weighting-decision.md`。

### ユーザーが選ぶ選択肢

- **案 A（推奨）**: 選択肢 C を維持し、#2536 を「対象外として完了」とする。保存需要には
  既存の保存経路は存在しない旨を案内する（`save_model`／`load_model` は `&Sequential` 専用で
  `Stacked*` は扱えない。`named_parameters` は命名契約に従うパラメータの列挙 API であり
  保存・復元 API ではない）。`named_parameters`／`set_parameter` は autodiff の `Module` trait のメソッドで、
  facade（`crates/facade/src/nn/rnn.rs`）は同 trait を再エクスポートしないため、**facade のみに
  依存する利用者は手動保存も行えない**（facade 単独では保存経路が一切無い）。手動で保存する場合は
  内部クレート `autodiff` への直接依存が必要で、`named_parameters` で列挙したテンソルを利用者側で
  書き出し、復元時は `set_parameter` で同じキーへ書き戻す運用となる（`facade` が唯一のサポート対象
  公開面である方針の外。形式・検証・エラー型は本ライブラリの保証外。公式の保存・復元 API は案 C の記録作成・承認後）。あわせて否定ガードの禁止集合へ
  `add_stacked_rnn`／`add_stacked_lstm`／`add_stacked_gru` を加える別 PR の可否も判断する。
- **案 B**: 選択肢 C を覆し、系列入力・隠れ状態を扱える Sequential 側 API を新設する。
  `Var -> Var` 平坦鎖前提の再設計・否定ガード反転・manifest kind 追加・resident 拒否の設計が
  必要なため、別の設計記録（記録作成 → 承認 → 実装）が要る。
- **案 C**: Sequential を経由せず `Stacked*` 単体の保存・復元 API のみ公開する。公開形
  （関数名・エラー型・manifest 形式）が未決のため、これも記録作成 → 承認の 2 段が要る。

### 不変事項

コード変更なし・新規 `unsafe` なし・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` 不変。#2536 は本記録では close しない（親 #2534 の完了条件に影響するため
ユーザー判断を待つ）。
