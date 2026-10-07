# pad の非定数モード（reflect・replicate・circular）の CPU 実装記録（イシュー #2642）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-cumulative-ops-decision.md`（#2636）の「共有カーネルを
`tensor-core` に置く」方式を再適用した実装記録であり、**承認記録ではない**（facade 公開形の承認は
#2677 で依頼中。公開自体は承認後の #2678・#2679）。

## 0. 結論

- `F.pad` の非定数 3 モード（reflect／replicate／circular）を CPU 参照実装として内部クレートへ追加した。
  既存の `Var::pad`（定数埋め）・`BackendOps::pad`・`pad_out_shape` のシグネチャと意味論は変更していない。
  - 共有カーネル: `fandhe_ai_tensor_core::pad_modes`（`PadMode`・`PadModeError`・`pad_modes_layout`・
    `pad_modes_host`・`pad_modes_vjp_host`）。添字写像・蓄積契約の単一情報源。
  - `BackendOps::pad_modes_forward` を既定 `Unsupported` で追加（非破壊拡張）。CPU は共有カーネルを呼ぶだけの
    override。CUDA／Metal は変更なし。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::pad_ops`（`pad_with_mode`）。専用 `Op::PadMode` と VJP は
    `grad.rs`。
- facade 公開は行わない。`PadModesHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。既定 `Unsupported` → 共有ホストカーネルへのフォールバックで動作する。
  実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 非定数 3 モードは REQ-9 Tier 2 の列挙に名前がない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、
  #2642 の受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは
  記録しない（承認依頼 #2677 は別途）。
- 既存 `Var::pad` は定数埋めのみ（`crates/autodiff/src/var.rs`）。`tensor-core/src/fft/stft.rs` に
  STFT 用の反射パディング（`StftPadMode`・`fn pad_mode`）が既存で、本実装とは共通化していない（§8）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル | `tensor-core/src/pad_modes.rs` | レイアウト検査（`PadModesLayout`）・forward・VJP |
| バックエンド抽象 | `tensor-core/src/backend_ops.rs` | `pad_modes_forward`（既定 `Unsupported`） |
| CPU | `backend-cpu/src/ops.rs` | 共有カーネルを呼ぶだけの override（`contiguous()` 済み入力を渡す） |
| autodiff | `autodiff/src/pad_ops.rs`・`tape.rs`・`grad.rs` | 自由関数・`Op::PadMode`・VJP |

- 3 モードはすべて「軸ごとの添字写像」で、forward は `out[o] = x[map(o)]`、backward は
  `d_x[map(o)] += g[o]`。多軸は軸ごとの写像の直積（出力位置 → 入力平坦添字は `Σ map[axis][o_axis] * in_stride[axis]`）。
  カーネルは写像表を確保せず、行ごとに基準オフセットを求めて最終軸で写像を適用する。
- 命名規律: `fn pad_mode` は `tensor-core/src/fft/stft.rs` に既存のため使わず、`fn pad_with_mode`（
  `autodiff/src/pad_ops.rs` の 1 件のみ。workspace インベントリが固定）・`pad_modes_forward`（trait メソッド）・
  `*_host`（共有カーネル）とした。
- `pads` は既存 `Var::pad` と同じ「先頭軸から順の `(before, after)` を rank 個」（PyTorch の末尾軸からの平坦リストとは
  異なる既存設計を踏襲）。
- `Var`／`Tape` に inherent メソッドは足していない（足すと facade 公開面が広がる）。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応。f64 自動微分経路は対象外。

## 3. 数値契約

- forward は算術を含まない純粋なコピーで **bit 完全一致**（NaN payload・`±inf`・`-0.0` も保存）。
- backward は添字が重複する scatter-add。入力要素ごとに `f64` アキュムレータへ出力の行優先順で蓄積し、最後に 1 回だけ
  `f32` へ downcast する（`.claude/rules/coding-rust.md` の勾配長軸縮約の規定）。蓄積順が固定のため run-to-run で
  bit 一致する。`f64` 蓄積は同一入力要素へ `1e8, 1, -1e8` が流れるケースで `1.0` になることで固定した。
- matmul 系 FMA 契約・tolerance・baseline には触れない。

軸 1 本の写像（`len` = 入力軸長、`p = o - before`）:

| モード | `o < before` | 内部 | `o >= before + len` | 事前条件（パディングがある軸のみ） |
|---|---|---|---|---|
| Reflect | `before - o` | `p` | `2*(len-1) - p` | `before < len` かつ `after < len` |
| Replicate | `0` | `p` | `len - 1` | `len >= 1` |
| Circular | `o + len - before` | `p` | `p - len` | `before <= len` かつ `after <= len` |

`(0, 0)` の軸には制約を課さない（軸長 0 を許し、出力は空になる）。写像は `usize` の分岐だけで書き、符号付きキャストや
wrapping 演算に依存しない。

## 4. 境界検査

- `pad_modes_layout` が確保より前に検査: rank 一致・加算オーバーフロー・出力要素数／バイト数（`pad_out_shape` を再利用）、
  モード別の pad 上限（違反は `PadModeError::InvalidArgument`。軸番号・軸長・pad 幅を文言に含む）。
- autodiff 入口は実体化より前に出力要素数へ既存の 1 GiB 上限（`checked_index_alloc_len`）を適用し、replicate の巨大 pad を拒否
  する（新しい閾値は作らない）。エラー時は tape を一切操作しない。
- カーネルは入力／上流スライス長を再検査し、入力アクセスは `get` による検査付き（`unsafe`／`get_unchecked` なし。REQ-8）。確保は
  `try_reserve_exact` で失敗を型付きエラーにする。
- バックエンドの戻り shape を再検証し、`Unsupported` のときだけホストへフォールバックする（他のエラーは伝播）。

## 5. PyTorch 2.14.0 との差分

実測の正は `crates/autodiff/tests/fixtures/pad-modes-pytorch-reference/`（`error_cases` に `torch_raises` と例外文面）。

- 本実装は PyTorch の上位集合: **任意軸・任意 rank** にパディングできる。PyTorch は `torch_pad` の個数と入力 rank の組
  を限定し、実測（fixture の `error_cases`）では rank 1 入力（pad 2 個）・rank 2 の 2 軸（pad 4 個）・rank 3 の 3 軸（pad 6 個）・
  pad 0 個を `NotImplementedError` で拒否する。PyTorch が拒否し本実装が受理する形（意図的な差分。テストは独立オラクルで検証）:
  rank 1 入力・先頭軸へのパディング・rank 2 の 2 軸・全 0 pad（`torch_pad` が空）。
- 両方が拒否する形: reflect の `pad >= 軸長`、circular の `pad > 軸長`、パディング軸の長さ 0。
- 両方が受理する形: circular の `pad == 軸長`（出力 shape を fixture と照合）、バッチ軸長 0（空出力）。
- 負のパディング（クロップ）は `usize` のため表現不可で対象外。`mode='constant'` は既存 `Var::pad` のまま。
- `pads` の並びは PyTorch（末尾軸から）と異なり先頭軸から（既存 `Var::pad` と同じ）。

## 6. テスト構成

- `crates/tensor-core/src/pad_modes.rs`（単体）: 3 モードの手計算値・境界ちょうど・事前条件違反・rank 不一致・オーバーフロー・
  軸長 0・rank 0・スライス長不一致・VJP 手計算値・`f64` 蓄積・決定性。`backend_ops.rs` に既定 `Unsupported` のテスト。
- `crates/autodiff/tests/pad_modes_parity.rs`: PyTorch fixture 突合（forward bit 一致・勾配 REQ-2）・意図的差分の分類表・
  独立オラクル（符号付き整数の別実装。先頭軸・全軸・rank 1・rank 5）・内積恒等式・中心差分・フォールバック／エラー伝播・
  事前検査（バックエンド呼び出し前に拒否）・tape 記録数・連鎖／共有入力・`create_graph` 型付きエラー・決定性。
- `crates/backend-cpu/tests/pad_modes_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::pad_modes_forward` の直接呼び出しと、
  CUDA（macOS では Metal も）が `Unsupported` を返し panic しないこと。
- `crates/facade/tests/pad_modes_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `pad_ops` への 1 行委譲で公開する。

- `Var::pad_with_mode(&self, pads: &[(usize, usize)], mode: PadMode) -> Result<Var<'t>, AutodiffError>`
  （`Var::pad` と同じ `pads` の並び・戻り値形）
- `fandhe_ai::PadMode` を再エクスポートする。`PadModeError`・`pad_ops`・`tensor_core::pad_modes` は再エクスポートしない
  （エラーは既存の `AutodiffError` に写像済み）。
- 非破壊（追加のみ。既存 `Var::pad` は不変）。`Sequential::add_*`（`ReflectionPad2d` 等の層）は別件として推奨に含めない。
- 承認依頼は #2677、公開は承認後の #2678・#2679。承認後は `PadModesHoldDoctestGuard` と否定ガードを承認形の正ガード
  （委譲本体の固定を含む）へ反転する。

本節は推奨案の記録であり承認記録ではない。承認事項（**すべて未承認**）: 上記メソッドの公開、`PadMode` の再エクスポート、
メソッド名・引数形。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。
- CUDA／Metal の GPU 専用カーネルと実機計測（§10）。
- 負のパディング、`mode='constant'` の統合、`ReflectionPad*`／`ReplicationPad*`／`CircularPad*` 層、`Sequential::add_*`。
- `create_graph`（高階微分）・activation checkpoint 対象化・f64／f16／bf16 自動微分経路。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- STFT の反射パディング（`tensor-core/src/fft/stft.rs`）との共通化、VJP の上流 shape 検査ヘルパー
  （`grad.rs::check_fft_upstream_shape`。名称が FFT 固有）の改名。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `PadModesHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・型・メソッドが facade 公開面に現れるとコンパイルが失敗する正のプローブ |
| `pad_modes_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `pad_modes_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_pad_modes`（＋自己テスト） | facade src の再エクスポート・`pub mod pad_ops`／`pub mod pad_modes`・`PadMode`／`PadModeError` の独自宣言・`fn pad_with_mode` 宣言の否定検査（`StftPadMode` は別トークンで対象外） |
| `workspace_declares_pad_modes_fn_names_only_in_allowed_locations` | workspace 全体で `fn pad_with_mode` が `autodiff/src/pad_ops.rs` の 1 件のみ |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_pad_modes_ops_match_cpu_reference`・
`metal_pad_modes_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は
`docs/perf/logs/pad-modes-2642/README.md`。

## 11. 出典

- `docs/autodiff-cumulative-ops-decision.md`（#2636）・`docs/autodiff-stat-reduce-ops-decision.md`（#2637。型付きエラーの置き方）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/pad-modes-pytorch-reference/README.md`

## 12. #2678 実装記録（Phase 4 の facade 公開）

- 状態: **§7 の公開形を #2678 で承認形どおり公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 11を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2678 時点で当該コメントの承認に更新された（承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき）。
- 公開した識別子: `Var::pad_with_mode(&self, pads: &[(usize, usize)], mode: PadMode)`、クレートルートへ `PadMode` の再エクスポート（既存 `Var::pad` は不変）。本体は `crate::<module>::<fn>` への 1 行委譲（`Var`）／`&self.0` を渡すだけの 1 行委譲（`Tape`）に固定し、新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` は追加していない。
- ガードの反転・縮小: `PadModesHoldDoctestGuard` から `Var` の impl・UFCS 行と `PadMode` のローカル定義を外し、`pad_ops`／`pad_modes` モジュール名・`PadModeError`・`Tape` 上の同名メソッドのプローブだけを残した（先例 #2516）。`api_surface.rs` の否定ガードは、承認済みの型名を識別子表から外し、`Tape` の承認済みメソッドを `fn` 宣言走査から除外したうえで、承認形だけを許す正ガードへ反転した。宣言場所インベントリには `autodiff/src/var.rs`（`Tape` 分は `facade/src/lib.rs`）の各 1 件を追加した。
- テスト: `crates/facade/tests/phase4_ops_facade.rs`（pad_with_mode_method。`fandhe_ai::` だけを import し、fn ポインタ型でシグネチャを固定して厳密に決まる値を確認）と、`crates/facade/tests/api_surface.rs` の正ガード（`var_phase4_ops_methods_are_thin_delegations`・`facade_reexports_phase4_ops_types_only_in_approved_shape`・`facade_tape_phase4_methods_are_thin_delegations`・各 `workspace_declares_*`）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。CUDA／Metal 実機 parity は未実測で、`docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md` へ申し送る（新しい数値経路はなく、1 行委譲のため既存の各 `*_backend_parity.rs` の結果がそのまま適用される）。
