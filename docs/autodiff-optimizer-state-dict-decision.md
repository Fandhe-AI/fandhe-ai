# optimizer state_dict（save/load・safetensors 経由）設計判断記録

イシュー #2174（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-param-groups-decision.md`（#2173）と同型の記録。

## §0 結論

PyTorch `torch.optim.Optimizer.state_dict()`／`load_state_dict()` 相当の
機能を、**`fandhe_ai_autodiff::nn::optim::OptimizerStateDict`**
（`crates/autodiff/src/nn/optim/state_dict.rs`）として内部クレート限定
で実装した。9 optimizer（`AdamW`・`Adam`・`RmsProp`・`Adagrad`・
`Lamb`・`Adadelta`・`Adamax`・`NAdam`・`RAdam`）が本 trait を実装し、
`state_dict()` は `HashMap<String, Tensor<f32>>` を返す。既存の
`step()`／`step_with_slot_hparams` の演算列には一切触れていない
（bit ドリフトなし）。

facade（`fandhe_ai::optim`）への再エクスポート・inherent メソッド追加
は、親 #2131 の「facade 公開面の拡張は設計判断記録 → 承認 → 実装の
2 段」規則により**未承認のため保留**する（§5「承認事項」）。

## §1 背景

イシュー #2174・親 #2131 のいずれにも所有者の承認コメントはない
（着手前に `gh issue view --json comments` で確認済み）。

`crates/facade/src/optim.rs` は `AdamW`／`Adam`／`RmsProp`／`Adagrad`／
`Lamb` を**名前指定で再エクスポート**している（glob ではない）。この
ため、これらの型へ inherent の `pub fn state_dict`／`load_state_dict`
を足すと、その時点で facade 公開面が広がる。イシュー本文は「承認事項
なし」「各 optimizer にメソッドを追加」と書いているが、上記の事実
（facade 名前指定再エクスポート）と #2173 の先例
（`docs/autodiff-param-groups-decision.md` §1）に鑑み、本実装は
`OptimizerStateDict` を**別 trait**として実装することで、facade が
これら型を再エクスポートしていても本 trait（内部クレート限定）が
facade 経由でスコープに入らない構造にした（trait メソッドは trait が
import されて初めて `.method()` 呼び出しが可能になる Rust の名前解決
規則を利用。#2173 と同方式）。

イシュー本文がスコープ外の追跡先として挙げている #2175 は、実際には
「RmsProp・Adagrad・LAMB の `DeviceParamStore` 常駐化」であり
（complete checkpoint ではない）、次の 2 つには追跡 Issue がない:

- model weight と optimizer state を同時に保存する complete checkpoint
- `compat::Sequential` 単位の再開 API（`fit` の再開）

`out-of-scope-tracking.md` の規約（ユーザー承認なしに Issue を起票し
ない）に従い、本イシューでは起票せず本 doc に記録するに留める。

## §2 設計

### 2.1 キー配置（形式バージョン 1）

- **種別マーカー** `__optimizer__.<kind>`（shape `[1]`・値 `1.0`）。
  `Adam`／`AdamW` はバッファ名（`m`／`v`）・スカラー構造
  （`beta1_pow_t`／`beta2_pow_t` あり）が完全に同一のため、マーカーが
  ないと黙って取り違えを受理してしまう。マーカー検証により
  fail-closed で拒否する（PyTorch は種別の異なる state の読み込みを
  許すが、本実装は安全側に逸脱する。`nn_optim_state_dict.rs::
  adam_state_dict_is_rejected_by_adamw_load_state_dict` で固定）
- **スカラー状態**（ロスレス符号化。§2.2）:
  `step_count.u64_u16x4`（全 9 種）・`num_slots.u64_u16x4`（全 9 種・
  必須。§2.3「単一バッファ optimizer の既知の限界」への是正として
  PR #2304 で追加。実在するバッファキーの最大添字からスロット数を
  推測するのではなく、独立したメタデータとして保存・照合する）・
  `beta1_pow_t.f64_u16x4`（`AdamW`・`Adam`・`Lamb`・`RAdam`・
  `Adamax`）・`beta2_pow_t.f64_u16x4`（`AdamW`・`Adam`・`Lamb`・
  `RAdam`・`NAdam`）・`mu_product`（shape `[1]` の生 f32。`NAdam`
  のみ）
- **スロットバッファ** `state.<i>.<buffer>`。バッファ名の対応表
  （PyTorch 側の対応する状態名も併記）:

| optimizer | 本実装のバッファ名 | PyTorch 対応 |
|---|---|---|
| AdamW・Adam・Lamb | `m`・`v` | `exp_avg`・`exp_avg_sq` |
| RmsProp | `square_avg`・`grad_avg`・`momentum_buffer` | 同名 |
| Adagrad | `state_sum` | 同名 |
| Adadelta | `square_avg`・`acc_delta` | 同名 |
| Adamax | `exp_avg`・`exp_inf` | 同名 |
| NAdam・RAdam | `exp_avg`・`exp_avg_sq` | 同名 |

ハイパーパラメータ（config）は保存しない。呼び出し側が同じ config で
`new` してから load する契約とする（PyTorch の `param_groups` が
ハイパーパラメータも保存するのとの差）。`states` が空（初回 `step()`
前）のときはスロットキーを一切出さない。

### 2.2 スカラーの符号化（ロスレス・NaN パターンを出さない）

`u64`（`step_count`）・`f64`（`beta*_pow_t`。`to_bits()` 経由）を下位
から 16bit ずつ 4 語に切り出し、各語を整数値の `f32`
（`0.0..=65535.0`）として shape `[4]` に格納する。`f32::from_bits` に
よるビットキャストと異なり NaN の bit パターンを一切生まないため、
NaN を正規化しうる外部ツールを経由しても壊れない。復号時は各要素が
有限・整数・`0.0..=65535.0` であること、shape が厳密に `[4]` である
ことを検証する。`beta*_pow_t`・`mu_product` はいずれもさらに「有限かつ
`[0.0, 1.0]`」を検証する（`mu_product` は初期値 `1.0` から
`mu ∈ [0, beta1)` を逐次乗じる積のため、有限性のみの検証では正常な
逐次積では生じない負値・`f32::MAX` 等を受理してしまう。P0 レビュー
指摘・イシュー #2174 PR #2304 是正で `validate_mu_product_range` として
`beta*_pow_t` と同じ検証関数に統一済み）。バッファ値自体（`m`・`v` 等）
は値域を検証しない。

**`step_count` は load 時に値域を検査しない**（P1 レビュー指摘・
イシュー #2174 PR #2304: 当初は「load 直後の 1 回の `step()` が確実に
成功する」ことを保証する fail-early 検証〈`validate_step_count_
headroom`〉として `step_count > u64::MAX - 1`〈`NAdam` は `u64::MAX -
2`〉を拒否していたが、各 optimizer の `step()` は `step_count ==
u64::MAX - 1` からの呼び出しに成功し、その結果生じた `step_count ==
u64::MAX` の状態を `state_dict()` で保存できるため、load 側がそれを
拒否すると optimizer 自身が生成した state_dict を復元できず save/
load の往復契約が破れる。`load_state_dict` の受理集合は `step()` の
到達可能な値域 `0..=u64::MAX` と一致させ、overflow 判定は各
optimizer の `step()` 側の `self.step_count.checked_add(1)` に一元化
した。`checked_add` の失敗時に状態が一切変わらないアトミック性を
保証するため、9 optimizer すべての `step()` で `checked_add` の結果
（新しい `step_count`）を、`self.states` の遅延初期化より前に計算し、
失敗時は早期に `Err` を返す順序へ変更した。既存の「エラー時は
`step_count`・状態が一切進まない」契約・テストは維持する）。

### 2.3 `load_state_dict` の検証順（fail-closed・状態変更前に全件検証）

`crates/autodiff/src/nn/optim/state_dict.rs::decode_state_dict` に集約
した:

1. `num_slots.u64_u16x4` を復号する（欠落は即 `Err`）。実在する
   バッファキーの最大添字から推測するのではなく、`state_dict` が
   独立に書き出したメタデータそのものを使う。続けて
   `num_slots * バッファ本数` を checked 乗算し、オーバー
   フローまたは実際のキー総数（`state.len()`）を上回る場合は即
   `Err` とする（1 スロットにつき最低 1 本のバッファキーが実在
   しなければならないため、正当な `state_dict` ではこの不等式は
   成立しない。巨大な `num_slots` 1 件で `0..num_slots` の全走査・
   大量の文字列生成を誘発する DoS を、期待キー集合を構築する前に
   遮断する）
2. キー集合の完全一致検査（`Module::load_state_dict` と同じ「欠落 →
   余剰」の順・昇順列挙の書式）。期待キー集合は 1. で確定した
   `num_slots` から `0..num_slots` の `state.<i>.<buf>` を機械的に
   列挙して作るため、非正規表記（`01`・`+1` 等）の添字キーは
   「余剰キー」として、末尾スロットの全バッファ欠落は「欠落キー」
   として、いずれも検出される
3. マーカーの shape・値検証
4. スカラーの復号・検証
5. スロットバッファの shape 一致検査・[`crate::eval::dense_vec`] に
   よる論理 row-major 順の読み出し（非 contiguous view にも対応）

ここまでをすべてローカル変数（`DecodedState`）に閉じ込めてから、各
optimizer の `load_state_dict` shim が最後に自身のフィールドを一括
置換する。途中で `Err` の場合、状態は一切変わらない。

**単一バッファ optimizer の既知の限界への是正（P0 レビュー指摘・
イシュー #2174 PR #2304）**: 当初実装は `Adagrad`（`state_sum` のみ
など、1 スロットにつきバッファが 1 本しかない optimizer）で「末尾
スロットの全バッファが欠落すると、そのスロットが最初から存在
しなかったのと区別できず、`step_count` だけ進んだ状態で load が
成功してしまう」限界を持っていた（実在するバッファキーの最大添字
から `num_slots` を推測していたため）。`num_slots` を独立した必須
メタデータへ切り出したことで、この限界は解消済みである
（`nn_optim_state_dict.rs::load_state_dict_rejects_trailing_slot_
fully_missing_without_mutation` で固定）。あわせて、少数キーでも
巨大な `num_slots` を宣言する攻撃入力を弾く checked arithmetic ＋
入力規模の上限検査も導入した
（`load_state_dict_rejects_oversized_num_slots_claim_without_
mutation`）。

## §3 対象外ファイル

`crates/facade/src/interop/safetensors.rs` は変更していない。符号化は
autodiff 側で完結し、すべて `Tensor<f32>` として既存の
`save_safetensors_f32*`／`load_safetensors_f32*` をそのまま通るため、
dtype マッピングの追加は不要。facade 非公開の trait への intra-doc
link は `cargo doc -D warnings` を落とすため、コード中のリンクは
バッククォート表記に留めている。

## §4 facade 公開保留の多層防御（#2173 と同型）

`crates/facade/src/lib.rs::OptimizerStateDictHoldDoctestGuard`（正の
プローブ 1 ブロック方式。`ParamGroupsHoldDoctestGuard` と同型）と、
`crates/facade/tests/api_surface.rs` の 4 テスト
（`optimizer_state_dict_hold_doctest_globs_all_pub_modules`・
`optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_optimizer_state_dict`・
`workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations`）
が機械的に固定する。既存の `optim_module_reexports_exactly_expected_
surface` 等の期待集合は変更していない。

## §5 承認事項（未承認のため保留）

facade（`fandhe_ai::optim`）公開面の拡張は次のいずれも未承認:

1. `fandhe_ai::optim::OptimizerStateDict` の再エクスポート、または
   `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb` への inherent
   `state_dict`／`load_state_dict` メソッド追加
2. `compat::Sequential` の optimizer state 取得／復元 API（`fit` の
   再開）
3. model と optimizer の complete checkpoint（未起票。起票にはユーザー
   承認が必要。§1 参照）

承認を得た日が来たら、`OptimizerStateDictHoldDoctestGuard`・対応する
4 テストを削除し、正のガード（実際の再エクスポート・facade テスト）
へ置き換える。

## §6 スコープ外

- `crate::optim::Sgd`（momentum バッファ）・`Lbfgs`（受入基準の 9 種に
  含まれない）
- `crate::optim::device_store::DeviceParamStore` 常駐更新経路の
  state_dict 対応
- param groups（#2173）の group 別ハイパーパラメータの保存
- F32 以外の dtype
- 入力サイズ上限の導入（既存 `interop::safetensors` と同じ扱い）
- CUDA／Metal parity: ホスト `Tensor<f32>` 値型のみを扱うため、GPU
  カーネル・数値一致 parity の申し送りは不要

## §7 検証コマンドと結果

- `cargo build -p fandhe-ai-autodiff` → warning 0 件
- `cargo clippy -p fandhe-ai-autodiff --all-targets --all-features -- -D
  warnings` → 新規・変更ファイルに findings なし（`crates/backend-cuda`
  の dead-code lint は本 PR と無関係の既存事象。`origin/main` でも
  同一コマンドで再現することを別 worktree で確認済み）
- `cargo test -p fandhe-ai-autodiff --lib nn::optim` → 184 passed
  （既存 optimizer 単体テスト、無修正で green）
- `cargo test -p fandhe-ai-autodiff --test nn_optim_state_dict` →
  137 passed（新規。9 optimizer × キー集合・往復・検証失敗時の非変更・
  非 contiguous 入力・種別マーカー拒否）
- `cargo test -p fandhe-ai-autodiff` → 全 test バイナリで 0 failed
  （既存統合テスト、無修正で green）
- `cargo test -p fandhe-ai --doc` → 33 passed
  （`OptimizerStateDictHoldDoctestGuard` の正のプローブを含む）
- `cargo test -p fandhe-ai --test api_surface` → 209 passed（新規 4
  テストを含む）
- `cargo test -p fandhe-ai --test interop_safetensors_optimizer_state` →
  13 passed（新規。9 optimizer の safetensors バイト列往復・決定性・
  ファイル往復・checkpoint 再開軌跡一致〈`AdamW` 代表・途中／step 0
  の 2 ケース〉）
- `cargo test -p fandhe-ai` → 全 test バイナリで 0 failed
- `git diff --stat` で `Cargo.toml`／`Cargo.lock`・`docs/spec` に差分が
  ないことを確認済み（新規依存・仕様変更なし）
