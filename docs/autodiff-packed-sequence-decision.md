# 可変長系列（`pack_padded_sequence`・`pad_packed_sequence`・`PackedSequence`）と RNN 系 packed 実行の CPU 実装記録（イシュー #2647）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-shape-view-ops-decision.md`（#2639）・`docs/autodiff-rnn-stacked-config-decision.md`（#2164）の
「既存 `Op` の合成のみで構成する」方式を再適用した実装記録であり、**承認記録ではない**
（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678・#2679）。

## 0. 結論

- PyTorch `torch.nn.utils.rnn` の `pack_padded_sequence`／`pad_packed_sequence`／`PackedSequence` 相当を、内部クレート
  `fandhe_ai_autodiff::nn::packed_sequence`（`crates/autodiff/src/nn/packed_sequence.rs`）へ追加し、既存 6 型の RNN
  （`Rnn`／`Lstm`／`Gru`・`StackedRnn`／`StackedLstm`／`StackedGru`）から packed 入力を処理できるようにした。
  - 型: `PackedSequence`（検証付きコンストラクタ `PackedSequence::new`）・出力型 4 種（`PackedRnnSeqOutput`・`PackedLstmSeqOutput`・
    `StackedPackedRnnSeqOutput`・`StackedPackedLstmSeqOutput`）。
  - 自由関数 8 本: `pack_padded_sequence`・`pad_packed_sequence`・`rnn_forward_packed`・`gru_forward_packed`・`lstm_forward_packed`・
    `stacked_rnn_forward_packed`・`stacked_gru_forward_packed`・`stacked_lstm_forward_packed`。
  - **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**。`crates/tensor-core`・`crates/backend-*`・`tape.rs`・`grad.rs`・`error.rs`・`var.rs`・
    `rnn.rs`・`rnn_stacked.rs` は変更していない。
- facade 公開は行わない。`PackedSequenceHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。実機テストは `#[ignore]` のまま未実測（§10）。
- 追加物は **自由関数と本モジュール専用の型のみ**。facade が再エクスポート済みの型（`Var`・`Tape`・`Tensor`・`nn::rnn` の 14 型と
  `rnn.cell()` 経由の 3 セル型）へ inherent メソッドを足していない（足すと facade 公開面が広がるため）。

## 1. 着手時の判定（事実のみ）

- `docs/autodiff-rnn-cell-tape-design.md`（決定 2・§5）は「`pack_padded_sequence` 相当の可変長系列」を v1 スコープ外としていた。
  既存の `forward_seq` は全系列が同じ長さの `[T,B,D]` しか受けられない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2647 の受入条件
  （内部実装・PyTorch 突合・フォールバック経路のテスト・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について
  承認済みとは記録しない（承認依頼 #2677 は別途）。
- pack／unpack は値のコピーだけで算術を含まず、公開済みクレート `fandhe-ai-tensor-core` の trait 面を広げる理由がないため、
  共有カーネルと `BackendOps::*` の追加は採らなかった。

## 2. 実装方式・合成表・バックエンド到達性

| 処理 | 合成 | forward のバックエンド呼び出し | backward |
|---|---|---|---|
| `pack_padded_sequence` | `Var::contiguous`→`reshape([T*B, *])`→`index_select(0, idx)`（`Op::Gather` 1 個。`batch_first` は転置せず添字側で吸収） | `gather` | `Op::Gather` の VJP（scatter 系の `*_with_fallback`） |
| `pad_packed_sequence` | `Var::cat([data, pad_row], 0)`→`index_select(0, idx)`→`reshape`（範囲外位置は末尾のパディング行を指す。パディング行は `var_no_grad`） | `concat`・`gather` | `Op::Concat`／`Op::Gather` の VJP |
| RNN 系 1 方向 | 各 step で `data.narrow(0, offset, b)` を入力にセル `Op`（`Var::rnn_cell`／`lstm_cell`／`gru_cell`）を適用。終了した行は `narrow` で切り出して最終状態へ保存し、最後に `cat` | セルの `BackendOps`（gemm・pointwise 等）・`concat` | 既存セル `Op` の VJP＋`Op::Narrow`／`Op::Concat` の VJP |
| 逆方向 | 各系列が自分の最終 step から始まるよう、系列が合流する step で初期状態の該当行を `cat` で連結 | 同上 | 同上 |
| 多層・双方向 | 層ごとに方向ぶん回し、packed データ同士を `Var::cat(&[fwd, rev], 1)`（1 回）で結合して次層の入力にする | 同上 | 同上 |

- `Var::cat`＋`index_select` で書けない事情は生じなかったため、代替の `Var::scatter`／`indexed_update_ops::index_copy` は使っていない。
- **`gather`／`scatter` は CPU・CUDA・Metal が実カーネルを持つ**。よって実機では pack／unpack がフォールバックにはならず GPU の
  gather が走る。受入基準 2 の「`Unsupported` フォールバックでホスト計算へ到達する」ことは、`gather`／`scatter`／`concat` と
  LSTM／GRU の pointwise を `Unsupported` にする呼び出し回数カウント付きモック `BackendOps` のテストで固定した（§6）。
- `cell.bind(tape)` は `(layer, direction)` ごとに 1 回だけ呼ぶ（BPTT の重み共有。既存 `forward_seq` と同じ）。
  `params` は `index = layer * num_directions + direction` 順。
- 命名規律: 素の `fn pack_padded_sequence` など 8 名は `autodiff/src/nn/packed_sequence.rs` の各 1 件のみ（workspace インベントリが固定。
  着手前の実測で workspace に同名の `fn` はなかった）。`forward_packed` という名の `fn` は 0 件。
- `h0`／`c0`／`h_n`／`c_n` は **元のバッチ順**の `[B,H]`（`sorted_indices` で内部的に並べ替える。PyTorch と同じ）。
- `PackedSequence` のフィールドは非公開で、不変条件（`batch_sizes` が非空・各要素 ≥ 1・非増加・総和 = `data.shape()[0]`・
  `sorted_indices` が置換）を満たさない値は構築できない。`unsorted_indices` は逆置換として導出する。
- 非 checkpoint・高階微分（`create_graph`）・f64／f16／bf16 自動微分経路での保証は対象外。挙動は合成先の `Op` に従う。

## 3. 数値契約

- pack／unpack の forward はコピーのみで算術を含まず、PyTorch と bit 完全一致する（NaN payload `0x7FC00001`／`0xFFC00002`・`±inf`・`-0.0` も保存。
  fixture の `nonfinite_*` ケースで確認）。
- 勾配と RNN の出力は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する。bit 一致は要求しない。
  tolerance 定数は新設・変更していない（autodiff 側は `tests/common` の `req2_close`、facade 側は
  `fandhe_ai_backend_cpu::parity::assert_parity`）。
- **対照ケース（全系列長 = `T`）**: packed 実行の出力と `h_n` は、同じセルを同じ値へ適用するため既存 `forward_seq`（step ごとの出力を連結した
  もの）と **bit 一致**することを実測で確認し、テストで bit 一致を要求している（`full_length_packed_equals_forward_seq_and_pytorch_padded`）。
  PyTorch の非 packed 実行とは REQ-2 判定で一致する。これにより「pack 起因の誤り」と「既存セルの乖離」を切り分けられる。
  既存セル（`Rnn`／`Lstm`／`Gru` と `Stacked*`）は PyTorch 2.14.0 の `nn.RNN`／`nn.LSTM`／`nn.GRU` と REQ-2 判定の範囲で一致しており、
  既存セル側の乖離は見つからなかった。
- FMA 契約・`f64` アキュムレータ契約（`.claude/rules/coding-rust.md`）の新たな対象はない（新しい算術・縮約を足していない）。

## 4. 境界検査

- 外部入力（`lengths`・`batch_first`・`total_length`・`padding_value`・`h0`／`c0`・`PackedSequence::new` の `batch_sizes`／`sorted_indices`）は、
  tape へノードを積む前にすべて検査する。引数起因のエラーで孤児ノードを残さないことを `tape.len()` の不変で固定した。
- 検査内容: rank ≥ 2・`lengths.len() == B ≥ 1`・各 `lengths[i]` が `1..=T`・`enforce_sorted` なら非増加・`T*B`／`N = Σ lengths`／`N+1`／
  `T_out*B` を `checked_mul`／`checked_add`・`checked_axis_len_as_i32`・`checked_index_alloc_len`（既存の 1 GiB 上限。新しい上限定数は作らない）・
  `Vec` は `try_reserve_exact`・`usize → i32` は `i32::try_from`。巨大な `total_length` による確保 panic／abort を確保前に型付きエラーへ変える。
- RNN 側は `cell.bind` と状態のゼロ確保より前に、入力幅（`[N, D]` の `D == input_size`）・`h0`／`c0` の shape（`[B, H]`）・同一 tape・
  stacked の状態スライス長（`num_layers * num_directions`）を検査する。
- エラーメッセージには形状・長さのみを含め、テンソルの中身を出さない。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

実測は `crates/autodiff/tests/fixtures/packed-sequence-pytorch-reference/`（出自は同 `README.md`）。

- **タイの並び**: `enforce_sorted = false` で同じ長さの系列がある場合、PyTorch の `sorted_indices` は元の添字昇順（安定ソート）だった
  （`ties_tm`／`ties_bf`: 長さ `[2,4,2,4,2]` → `sorted_indices = [1,3,0,2,4]`）。本実装も安定ソートで bit 一致する。
- **torch が例外を出さない入力**（本実装は型付きエラー。`INTENDED_DIFFS` で管理）:
  - `lengths` が `T` を超える（`pack_len_gt_T`・`pack_batch_first_len_gt_T`）。
  - `lengths` の個数がバッチ数と一致しない（`pack_len_count_mismatch`）。
  - 本実装は確保・添字生成の前にエラーにする（範囲外参照を作らないため）。
- torch と本実装がともにエラーにするもの: 長さ 0・`enforce_sorted = true` での未整列・rank 1 入力・`total_length < T_max`。
  `total_length == T_max` はともに成功する。
- 層間 dropout は packed の `data` へ層ごとに 1 回適用する（PyTorch も `data` へ適用する）。padded 版 `forward_seq`（step ごとに適用）とは
  **RNG 消費順が異なる**。eval 時と `p = 0` は no-op でテストで固定した。
- `lengths`・添字は `usize`。長さ 0 の系列・負の添字・テンソル引数形の `lengths` は非対応。
- `h_n`（`c_n`）を stacked では `Vec<Var>`（`index = layer * num_directions + direction`）で返す（PyTorch は `[L*dirs, B, H]` の 1 テンソル）。
  既存 `StackedRnnSeqOutput` と同じ形に揃えた。
- 双方向の逆方向は、各系列が自分の最終 step から始まる（padded 実行とは結果が変わる点）。可変長バッチの packed 実行が、各系列を `B = 1`・
  自分の長さで個別に `forward_seq` した結果と REQ-2 判定で一致することで、独立に検証した。

## 6. テスト構成

| ファイル | 内容 |
|---|---|
| `crates/autodiff/src/nn/packed_sequence.rs`（`#[cfg(test)]`） | 既知値の pack／unpack・`batch_first` と time-major の一致・タイの安定性・往復・エラー時の tape 不変・`PackedSequence::new` の不変条件違反の拒否・勾配が有効位置だけに流れること |
| `crates/autodiff/tests/packed_sequence_parity.rs` | fixture 突合（pack 16・unpack 12・RNN 19 件）・対照ケース（`forward_seq` と bit 一致）・per-sequence 独立検証・エラーケースの torch 照合・入力検証・決定性・dropout・モック `BackendOps` による `Unsupported` フォールバックと他エラーの伝播 |
| `crates/facade/tests/packed_sequence_backend_parity.rs` | CPU tape と NaiveOps tape の突合（pack／unpack は bit 一致・RNN 系は `assert_parity`）・手計算の期待値・CUDA／Metal 実機（`#[ignore]`） |
| `crates/facade/tests/api_surface.rs` | 保留ガード 5 本（§9） |

## 7. facade 公開形の推奨案（ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§12 参照）

推奨は 1 つ。**モジュール再エクスポート**: `fandhe_ai::nn::rnn`（#1955・#2535 で確立した純再エクスポートモジュール）へ、`PackedSequence`・
出力型 4 種・自由関数 8 本を `pub use` で追加する。

- 理由:
  - 自由関数が生の `Tape` を取らない（tape は `input.data().tape()` から得る）ため、`Tape` newtype の委譲メソッドが要らない。
  - `Var` メソッドにすると RNN 専用の概念が `Var` の全利用者に見える。
  - `compat::Sequential` は `Var → Var` の平坦鎖で、複数出力・状態引き回しの RNN とは構造が合わない（`crates/facade/src/nn/rnn.rs` モジュール doc の既存判断）。
- 不採用案と理由:
  - `Var` への委譲メソッド（`Var::pack_padded_sequence` 等）: 上記のとおり RNN 専用概念が `Var` に広がる。
  - `Tape::*_forward_packed` の委譲: `Tape` newtype のメソッドが 6 本増え、生の `Tape` を引数に取る形は `Rnn::forward_seq` と二重になる。
  - `Sequential::add_*`: 状態の引き回しと複数出力を表せない。
  - `Rnn` 等への inherent メソッド（`forward_packed`）: 再エクスポート済みの型に足すと 1 つで facade 公開面が広がり、`with_config` の保留と同じ問題になる。
- 承認事項（**すべて未承認**）: 再エクスポート対象の一覧、関数名、`PackedSequence::new` を公開に含めるか、`lengths`・添字が `usize` であること、
  `sorted_indices` の `Option` 表現、`h_n` を `Vec<Var>` で返す stacked の形。承認依頼は #2677、公開は承認後の #2678・#2679。
- 承認後は `PackedSequenceHoldDoctestGuard` と否定ガードを承認形の正ガードへ反転する。

本節は推奨案の記録であり、承認を得たことの記録ではない。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。保留ガードの正ガードへの反転を含む。
- GPU 専用カーネル、CUDA（DGX Spark GB10）／Metal（M4 Max）実機での `#[ignore]` テスト実測（§10）。
- `pack_sequence`・`pad_sequence`・`unpad_sequence`・`unpack_sequence`（PyTorch の周辺ユーティリティ）。
- tape を使わない推論経路（`Module::forward_host` の packed 版）。
- 既存 `forward_seq`（padded 版）の変更、`Rnn` 等への inherent メソッド追加、`Module::forward` の packed 対応。
- 負の添字・`lengths` のテンソル引数形・長さ 0 の系列。
- `create_graph`（高階微分）・activation checkpoint・f64／f16／bf16 自動微分経路での保証。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定・`MIN_KNOWN_PROBE_BLOCKS` の更新。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `PackedSequenceHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の自由関数 8 名・型 5 名・モジュール `packed_sequence` が facade に公開されるか、`Var`／`Tape`／`Tensor<f32>` と `nn::rnn` の 6 型へ同名の inherent メソッドが公開されるとコンパイルが失敗する正のプローブ |
| `packed_sequence_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `packed_sequence_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_packed_sequence`（＋自己テスト） | facade src の再エクスポート・型の独自宣言・`pub mod packed_sequence`・9 名の `fn` 宣言の否定検査 |
| `workspace_declares_packed_sequence_fn_names_only_in_allowed_locations` | workspace 全体で `fn` 宣言が `autodiff/src/nn/packed_sequence.rs` の自由関数 8 名各 1 件だけ（`forward_packed` は 0 件）であること |

検出範囲は列挙した名前・型に限り、マクロ生成や別名経由までは保証しない。stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは
正のプローブ＋インベントリで組んでいる。**有効性の確認**: 一時的に facade の `nn/rnn.rs` へ
`pub use fandhe_ai_autodiff::nn::packed_sequence::pack_padded_sequence;` を足すと doctest（E0659・E0061）と
`facade_does_not_reexport_or_declare_packed_sequence` が落ち、`Rnn` へ `pub fn forward_packed(&self) {}` を足すと doctest（E0308）と
`workspace_declares_packed_sequence_fn_names_only_in_allowed_locations` が落ちることを確認したうえで、いずれも元に戻した。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_packed_sequence_matches_cpu_reference`・
`metal_packed_sequence_matches_cpu_reference`）は `#[ignore]` のまま未実測。手順は `docs/perf/logs/packed-sequence-2647/README.md`。
実機で走るのは既存カーネルの新しい呼び出し形の確認であり、新規カーネルの parity ではない。

## 11. 出典

- `docs/autodiff-shape-view-ops-decision.md`（#2639。保留ガード一式の雛形）・`docs/autodiff-rnn-stacked-config-decision.md`（#2164）・
  `docs/autodiff-rnn-cell-tape-design.md`（決定 2・§5）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/packed-sequence-pytorch-reference/README.md`

## 12. #2679 実装記録（facade 公開）


- 状態: **§7 の推奨形を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 16 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子: `fandhe_ai::nn::rnn`（`crates/facade/src/nn/rnn.rs`）へ `PackedSequence`・`PackedRnnSeqOutput`・`PackedLstmSeqOutput`・
  `StackedPackedRnnSeqOutput`・`StackedPackedLstmSeqOutput` と自由関数 8 本（`pack_padded_sequence`・`pad_packed_sequence`・
  `rnn_forward_packed`・`gru_forward_packed`・`lstm_forward_packed`・`stacked_rnn_forward_packed`・`stacked_gru_forward_packed`・
  `stacked_lstm_forward_packed`）を `pub use`（`fandhe_ai_autodiff::nn::packed_sequence` からの純再エクスポート）。
- 記録に明記のない点の扱い: `PackedSequence::new`（検証付きコンストラクタ）は型の純再エクスポートに伴い到達可能になる。§0 が型の一部として
  挙げているため newtype で隠さずそのまま公開した。`Var`／`Tape` への委譲・`Rnn::forward_packed`・`Sequential::add_*` は不採用のまま追加していない。
- ガード（§9）の反転: `PackedSequenceHoldDoctestGuard` は型 5・自由関数 8 の衝突プローブを削除し、未承認経路（モジュール `packed_sequence` の公開・
  `Var`／`Tape`／`Tensor<f32>` の同名メソッド・`Tape::*_forward_packed`・`Rnn` 等の `forward_packed`）のプローブだけに縮小した。
  `facade_does_not_reexport_or_declare_packed_sequence` は承認形を違反としない走査（モジュール再エクスポート・`forward_packed`・型の独自宣言・同名 `fn` を禁止）へ
  変更し、承認形の過不足は `facade_exposes_packed_sequence_only_in_approved_shape`（`nn/rnn.rs` の承認 5 文のトークン列一致）が固定する。
  到達性は `packed_sequence_types_are_reachable_via_facade_only`、利用例は `nn::rnn` のモジュール doc の doctest
  （`packed_sequence_usage_doctests_are_present_and_compiled` が存在を検査）。`fn` 宣言の場所インベントリは不変。
- `packed_sequence_backend_parity.rs` は型・自由関数を `fandhe_ai::nn::rnn` 経由へ切り替えた（生の `Tape` との突き合わせだけ内部クレートを使う）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
