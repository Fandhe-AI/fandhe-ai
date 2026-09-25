# RNN／LSTM／GRU の多層・双方向・dropout（`RnnConfig`）設計判断記録

イシュー #2164・親 #2131。

## §0 結論

`fandhe_ai_autodiff::nn`（内部クレート）に `RnnConfig`（`num_layers`・
`bidirectional`・`dropout` の 3 オプション）と、これを受け取る新しい型
`StackedRnn`／`StackedLstm`／`StackedGru`（Sequence レベル。単層・単方向の
既存 `Rnn`／`Lstm`／`Gru` とは別型）を実装した。`RnnConfig::default()`
（`num_layers=1`・`bidirectional=false`・`dropout=0.0`）で構築した
`Stacked*` は既存の `Rnn`／`Lstm`／`Gru` と bit 完全一致する。facade
（`fandhe_ai`）への公開は未承認のまま対象外とし、`RnnConfigHoldDoctestGuard`
（正のプローブ doctest）＋`api_surface.rs` の 2 テストで保留を多層固定する。

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
