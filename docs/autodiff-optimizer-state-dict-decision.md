# optimizer state_dict（save/load・safetensors 経由）設計判断記録

イシュー #2174（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-param-groups-decision.md`（#2173）と同型の記録。

## §0 結論

> **更新（#2556）**: 以下の「内部クレート限定」「facade 公開は保留」は #2174 時点の記述で、
> facade への公開は #2556 で実施済み（§11）。

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

> **更新（#2556）**: 本節の `OptimizerStateDictHoldDoctestGuard` と 2 テストは #2556 で撤去し、
> 否定ガードは正ガードへ反転した（§11）。以下は当時の記録。

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

> **更新（#2556）**: 項目 1 は §9 の推奨案 A で承認・公開済み（§11）。項目 2・3 は引き続き対象外。

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

- `Lbfgs`（受入基準の 9 種に含まれない。専用 inherent API は #2366、
  `Sgd` は #2367 で実装済み。§8 参照）
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

## §8 `Sgd` への実装（イシュー #2367・親 #2131）

`crate::optim::Sgd`（`crates/autodiff/src/optim/sgd.rs`）へ
`OptimizerStateDict` を実装した。`compat::save_model`／`load_model` で
momentum 付き `Sgd` を bit 一致で再開するための内部 API（manifest への
結線は #2372）。facade への再エクスポートは行わない。

- **キー**: `__optimizer__.sgd`（マーカー）・`num_slots.u64_u16x4`・
  `state.<i>.momentum_buffer`（バッファ名は PyTorch と同じ）。
  `step_count`／`beta*_pow_t`／`mu_product` は持たず、`step_count` キーの
  混入は余剰キーとして拒否する。形式バージョンは 1 のまま
- **`velocity == None`**: `num_slots = 0` でスロットキーなし
- **デコーダ**: 既存 `decode_state_dict` のシグネチャ・挙動は不変。本体を
  private な共通関数へ移し、`pub(crate) decode_slot_only_state_dict` が
  `step_count` を要求しない形で同じ検証順（DoS 上限・キー集合完全一致・
  マーカー・shape）を再利用する。`state_dict` モジュールは `pub(crate)`
  化した（クレート外・facade の公開面は不変）
- **`load_state_dict`**: 全件検証後に一括代入（失敗時 `self` 不変）。
  `momentum == 0.0` の `Sgd` への `num_slots > 0` は `InvalidArgument`
  （`step()` の「velocity が `Some` ⇔ momentum ≠ 0」前提の保護）。
  velocity と params の件数・shape 整合は次の `step()` が検査する
- **縮退ケース（受容）**: momentum 有効で params 0 件の step が作る
  `Some(vec![])` は `num_slots = 0` に潰れ、load で `None` に戻る。差は
  次の n>0 件 step が件数変化エラーでなく初回 step になる点のみ
- **facade 保留ガード**: `OptimizerStateDictHoldDoctestGuard` のプローブ
  対象（5 型）は本イシューでは変更しない（固定文言と同時変更が必要なため）。
  `Sgd` をプローブへ追加する多層防御の強化はフォローアップ。
  `api_surface.rs` の定義元インベントリに `sgd.rs` の 2 エントリを追加
- 新規 `Op`／`BackendOps`／カーネル／`unsafe`／依存の追加なし。
  `step()` の演算列は不変（bit ドリフトなし）

## §9 #2556（facade 公開形）の着手時判定と承認依頼

> **更新**: 本節の推奨案はルート #2499 のコメントで承認され、§11 で実装した。

### 9.1 経緯

親 #2555（その親 #2542）は、学習の再開のため `OptimizerStateDict` を facade
へ公開することを求める。子は #2556（公開形の確定と公開。本件）と #2557
（保留ガードの正ガード化と docs 更新）。#2556 の受入条件は「§4・§5 の推奨形を
確定形として公開する。推奨形がない、または複数案のままなら、記録へ推奨案を
追記し承認依頼を残して停止する」である。

### 9.2 着手時判定（停止条項を適用）

調査基準は origin/main `6787457d`。

1. **公開経路が 2 案併記のまま**: §5 項目 1 は「再エクスポート、または
   inherent メソッド追加」と併記し、どちらを採るか決めていない。§4・§5 に
   推奨形の記述はない。#2499 の一括承認は記録に書かれた推奨形にのみ及ぶため、
   本件には及ばない。
2. **公開されるメソッドの範囲が未決**: `OptimizerStateDict` の impl は 10 型
   （`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・
   `NAdam`・`RAdam`・`Sgd`。`crates/autodiff/src/nn/optim/*.rs`・
   `crates/autodiff/src/optim/sgd.rs:432`）で、いずれも facade 公開済み。§5 項目 1
   が挙げる 5 型より多く、保留ガードのプローブも 5 型のみ（§8）。
3. **trait の sealing が未決**: facade から `pub` trait を再エクスポートすると、
   下流クレートが impl でき、後からのメソッド追加が破壊的変更になる。sealing は
   `autodiff` 側の変更を要し「facade へ公開」の範囲を超える。
4. **`Lbfgs` との関係**: `Lbfgs` は trait を実装せず、シグネチャの異なる
   inherent の `state_dict`／`load_state_dict`
   （`crates/autodiff/src/nn/optim/lbfgs.rs:403`・`:450`。#2366）を持ち、#2502 以降
   facade から到達できる。trait の再エクスポートと名前は衝突しない。
5. **§5 項目 2・3 は本件の対象外**: `compat::Sequential` の optimizer 状態 API
   （`fit` 再開）と complete checkpoint は #2556 に含めない。

### 9.3 推奨案 A（承認待ち。確定形ではない）

`crates/facade/src/optim.rs` に
`pub use fandhe_ai_autodiff::nn::optim::OptimizerStateDict;` を 1 文 1 行で
追加する。根拠:

- `crates/autodiff/src/nn/optim/mod.rs:122` で crate 内の公開再エクスポートが
  既にある。
- シグネチャが使う `HashMap`・`Tensor<f32>`・`AutodiffError` は facade から名前で
  指せる（`AutodiffError` は `crates/facade/src/lib.rs` で再エクスポート済み）。
- §1 は facade の再エクスポートが 1 行で済むよう別 trait として設計した経緯を
  記す。既存の optim 公開も素の再エクスポートである
  （`docs/facade-optimizer-promotion-decision.md` §4 案 A）。
- 案 B（inherent メソッド）は、facade 側から外部型へ inherent impl を足せない
  ため `autodiff` 側に 10 型分の重複メソッドが要り、trait メソッドと名前が重なる。

### 9.4 ユーザーに決めてほしい事項

- (a) 公開経路: 案 A（trait 再エクスポート）か案 B（inherent）か。推奨は A。
- (b) 到達できる impl 集合: 10 型（`Sgd` 含む）すべてを受け入れるか。推奨は
  受け入れる。保留ガードのプローブとの差（5 型 / 10 型）は反転後の正ガードで
  10 型を固定して吸収する。
- (c) sealing: 下流 impl を許すままにし、メソッドを追加しない契約を記録するか。
  sealing する場合は `autodiff` 側の別 issue と別承認が要る。推奨は
  sealing しない。
- (d) `Lbfgs` を trait の対象外のままにする（#2366 の inherent API を維持）
  ことの確認。
- (e) #2556 の閉じ方: 承認後に新規 issue で本実装するか、#2556 を reopen するか。
  #2557 は本承認待ちでブロックされている。

### 9.5 承認後の実装スケッチ（本 PR では実施しない）

- `optim.rs` に `pub use` を 1 行追加し、モジュール doc に対象 10 型・`Lbfgs`
  対象外・config を保存しないため同じ config で `new` してから load する契約・
  種別マーカーによる fail-closed・`save_safetensors_f32`／
  `load_safetensors_f32` での往復を示す doctest を追加。「facade 非公開」の
  記述を更新する。
- `lib.rs` の `OptimizerStateDictHoldDoctestGuard` を撤去し、`api_surface.rs` の
  保留 4 テストのうち否定ガードを、承認した形だけを許す正ガードへ反転する。
  `optim_module_reexports_exactly_expected_surface` の期待集合へ追加する。
- facade のみを import した単体テスト（10 型の往復・load 失敗時の状態不変・
  種別違いの拒否）を追加する。
- 新規 `Op`／カーネル／`unsafe`／依存はなく、CUDA／Metal parity の申し送りは不要。

### 9.6 本節の位置づけ

本節は承認の取得を意味しない。§5 の保留と `OptimizerStateDictHoldDoctestGuard`・
4 テストは維持している。本 PR では `crates/`・`Cargo.*`・tolerance・`docs/spec`
を変更していない。

## §10 #2557 着手時判定（#2556 未実装・§9 未承認のため停止）

> **更新**: 停止の理由（§9 未承認・#2556 未実装）は §11 で解消した。

調査基準は `origin/main` `795894c6`（2026-10-04 確認）。本節は停止の記録であり、
承認を取得したことを意味しない。

### 10.1 判定

- 依存の #2556 は PR #2741 で §9（推奨案 A・承認依頼）を記録してクローズされた。
  `fandhe_ai::optim` への `OptimizerStateDict` 再エクスポートも各型への inherent
  メソッド追加も `crates/facade/` には存在しない（`optim.rs` の出現はモジュール doc
  のみ）。
- 正ガードへの反転は「承認・公開済みの形だけを許す」検査であり、公開物が無い状態では
  反転先が存在しない。
- §9.4 (a)〜(e) は未承認。#2499・#2555・#2556・#2557 に承認コメントは無い。§4・§5 には
  推奨形が無く（§5 項目 1 は再エクスポートと inherent メソッド追加の 2 案併記）、
  ルート #2499 の一括承認は §9 の推奨案に及ばない（`docs/autodiff-param-groups-decision.md`
  §10.1 と同じ判断）。

### 10.2 結論

- 停止条項に従い、`OptimizerStateDictHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の次の 4 テストは撤去も反転もせず現状維持する。
  - `optimizer_state_dict_hold_doctest_globs_all_pub_modules`
  - `optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract`
  - `facade_does_not_reexport_or_declare_optimizer_state_dict`
  - `workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations`
- 解除の順序は、§9.4 (a)〜(e) のユーザー承認 → #2556 の reopen または新規実装イシューでの
  facade 公開 → 保留ガードの反転（#2557 の受入条件）。承認だけでは解除されない。
- #2557 の受入条件 3 点（ガード反転・`docs/compat-api-scope.md` §5 の適用記録・facade
  経由の利用例テスト）は未達。受入条件 2 のうち §4・§5 への実装記録（公開した名前・
  ガード反転内容）は、公開・反転を行っていないため書けない。
- §8 のフォローアップ（保留ガードのプローブへの `Sgd` 追加）は、承認後の正ガードで
  10 型を固定して吸収する方針（§9.4 (b)）のため本イシューでは行わない。

### 10.3 本イシューで行わないこと

- `crates/facade/**` の変更、保留ガードの撤去・反転
- `docs/compat-api-scope.md` §5 への適用記録（公開を適用していないため）
- facade 経由の利用例 doctest・テストの追加（対象 API が存在しないため）
- 追跡 Issue の起票・#2556 の reopen（ユーザー承認が必要）

## §11 #2556 実装記録（facade 公開）

### 11.1 承認の根拠

§9.4 の (a)〜(e) は、ルート #2499 のコメント
（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）の
「#2555: §9 の推奨案（sealing はしない）」に記録された承認を根拠とする。本節はそれ以上の
承認があったことを意味しない。記録に形が書かれていない論点は見つからなかった。

### 11.2 確定形

| 論点 | 確定形 |
|---|---|
| (a) 公開経路 | 案 A。`crates/facade/src/optim.rs` に `pub use fandhe_ai_autodiff::nn::optim::OptimizerStateDict;` を 1 文 1 行で追加（素の再エクスポート。inherent メソッドは追加しない） |
| (b) 到達する impl | 10 型（`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・`NAdam`・`RAdam`・`Sgd`） |
| (c) sealing | しない。代わりに「trait へメソッドを追加しない」契約を trait doc に記録した（下流 impl を壊さないため） |
| (d) `Lbfgs` | trait の対象外のまま（既存の inherent `state_dict`／`load_state_dict`／`history_len` を維持） |
| (e) 進め方 | #2556 を reopen して本実装 |

公開した名前は `fandhe_ai::optim::OptimizerStateDict` の 1 つだけ。`fandhe-ai =0.10.0` の
既存の公開 API の削除・シグネチャ変更はなく、追加のみ。

### 11.3 撤去・反転したガード

- `crates/facade/src/lib.rs::OptimizerStateDictHoldDoctestGuard`（doctest 足場）を撤去。
- `api_surface.rs`: `optimizer_state_dict_hold_doctest_globs_all_pub_modules`・
  `optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract`・固定文言定数
  `OPTIMIZER_STATE_DICT_HOLD_PROBE_BODY` を撤去。
- `facade_does_not_reexport_or_declare_optimizer_state_dict`（否定ガード）を
  `facade_reexports_optimizer_state_dict_only_in_approved_form`（正ガード）へ反転。
  `OptimizerStateDict`／`state_dict` を識別子に含む `pub use` 文が、`src/optim.rs` の承認形 1 文
  （別名なし・完全一致）だけであることを文単位で fail-closed に検査する（複数行の波括弧形・
  他モジュールへの重複公開・欠落も検出）。検出器の自己検証は
  `optimizer_state_dict_reexport_guard_detects_each_category`。
- `optim_module_reexports_exactly_expected_surface` の期待集合へ `OptimizerStateDict` を追加。
- `hold_doctest_probe_blocks_reference_every_glob_imported_item` の既知プローブ名の期待を、
  削除済みの `__fandhe_optim_state_dict_hold_probe` から現存プローブ候補のいずれかへ変更。
- `workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations`（定義元
  インベントリ）は不変（素の再エクスポートは `fn` 宣言を増やさない）。

### 11.4 追加したテスト

- `crates/facade/tests/optim_state_dict_facade.rs`（`fandhe_ai` だけを import）: 10 型
  （`Sgd` は momentum あり・なし）の保存→復元後に続きの `step()` が bit 一致、初回 step 前の
  復元、`Sgd`（momentum なし）がスロットキーを持たないこと、キー欠落・余剰キー・shape 不一致で
  load が失敗し状態が変わらないこと、`Adam` の状態を `AdamW` へ読み込む試みの拒否、
  safetensors バイト列経由の復元。
- `api_surface.rs::optimizer_state_dict_is_reachable_via_facade_only`: 10 型が trait 境界を
  満たすことのコンパイル時固定。
- `fandhe_ai::optim` のモジュール doc に、`step` → `state_dict` → safetensors バイト列 →
  新規 optimizer へ `load_state_dict` → 次の `step` が一致する doctest を追加。

### 11.5 #2557 へ残した事項

`docs/compat-api-scope.md` §5 の適用記録、正ガードの追加強化、利用例の拡充は #2557 の範囲。

### 11.6 検証

実機（CUDA／Metal）parity はホスト値型のみのため不要（§6）。新規 `Op`／`BackendOps`／カーネル／
`unsafe`／依存の追加はない。`Cargo.toml`／`Cargo.lock`／tolerance／baseline／`guardrail.toml`／
`docs/spec` は変更していない。
