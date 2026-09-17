# 高階微分（grad of grad）の設計判断（イシュー #1622）

- 対応イシュー: #1622（親 #1573〈Tier 2〉→ ルート #1570）。本追記: イシュー #1941
- 位置づけ: 本文書は**設計判断のみ**を記録する。`crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。tolerance／baseline（数値一致の許容誤差・非後退ベースライン）の変更も対象外
- 基準コミット: `origin/main` `92d75265`（イシュー #1941。前段の基準コミットは `581d5208`〈#1675 時点〉）。行番号・事実はすべて本コミットで再確認した

## 0. #1941 追記の要点

前提 issue（#1593・#1597・#1599・#1601・#1612）は本追記時点ですべて **CLOSED** であることを `gh issue view` で確認した（§2「前提 issue の完了確認」）。これを受け、旧版（#1622 執筆時点）が「前提未完了のため段階 0（非対応の明文化）に留める」としていた記述を HEAD 基準へ更新し、主案 A-2（子テープ方式の `create_graph`）の API 契約を「契約案」として一段具体化した（§7 の 2 案・§8 の対象 Op 分類表・§9 の API 契約節）。**採否の確定・実装着手はいずれもユーザー承認が前提**（§10）であり、本追記でも実装は行わない。段階 1（設計確定・実装未着手）へ位置づけを更新する。

## 1. 背景

- 機能ギャップ表 `docs/compat-feature-gap.md` §2.11「高階微分（`grad of grad`）」行（`docs/compat-feature-gap.md:312`）は「なし（テープは 1 階のみを前提とした構造と推定）・必要物: Op 自体を微分可能にする再設計（VJP の VJP）・難度 XL」のまま（本追記でも実装状態は「なし」で不変）
- spec 側は 2026-09-12 の REQ-9 改定（`docs/spec/04-requirements.md:232`。`docs/spec` submodule ポインタは `581d5208`〜`92d75265` 間で変更なし〈`git log 581d5208..92d75265 -- docs/spec` が空〉ため行番号は不変）で高階微分を **Tier 2**（長尾。PyTorch／TensorFlow 機能網羅の第 2 段階）に明記済み。実装リポ側の `docs/compat-api-scope.md` §1.3 Tier 2 表は当該行を「高階微分 | #1622（設計記録。`docs/autodiff-higher-order-grad-decision.md`）」（`docs/compat-api-scope.md:248`）としており、本 doc がその設計記録に該当する
- 対比対象: PyTorch `torch.autograd.grad(create_graph=True)`（VJP をテープ上の演算として記録し、その計算グラフをさらに `backward()` できる）、TensorFlow `GradientTape` のネスト（外側テープが内側テープの `gradient()` 呼び出しを記録する）。いずれも Hessian-vector product・二階の損失正則化（gradient penalty）・メタ学習（MAML 等）の実装に使われる
- 目的（受入基準の構造化。#1941 版）:
  1. 前提 issue 完了後の HEAD で §2 の事実を再検証し、陳腐化した行番号・記述を更新する
  2. 子テープ方式で二階微分を取れる**対象 Op の一覧**（初期スコープ）と**非対象**（resident／fused 経路を含む）を Op enum の全 variant について分類し記録する
  3. 公開 API（`Tape::backward_create_graph` 相当）の契約を承認事項として列挙する（確定は §10 のユーザー承認に委ねる）
  4. tolerance・baseline は変更しない。コード変更は行わない

## 2. 前提 issue の完了確認・現状のコード事実（`origin/main` `92d75265`）

**前提 issue の状態**（`gh issue view <n> --json state`）:

| issue | 内容 | 状態 |
|---|---|---|
| #1593 | `Var::sub`／`neg`／`div`／`pow`／`sqrt`／`log` 系・比較演算 | CLOSED |
| #1597 | `permute`／`squeeze`／`unsqueeze`／`expand`（`broadcast_to`）／`flatten` | CLOSED |
| #1599 | `narrow`／`where`／`masked_fill`／`gather`／`scatter`／`scatter_add`／`index_select` | CLOSED |
| #1601 | `mean`／`min`／`argmax`／`argmin`／`var`／`std`／`norm`・複数軸／keepdim | CLOSED |
| #1612 | `no_grad`／`detach`／`retain_graph`・勾配蓄積 | CLOSED |

前提 5 件がすべて完了したことで、旧版が挙げていた「VJP 内部で使う演算のうち `Var` に公開 API として存在しないもの」という制約は解消済みである（下記「VJP ヘルパー → Var 演算」対応表参照）。

**コード事実**（HEAD `92d75265`）:

| 事実 | 出典 |
|---|---|
| `grad::vjp(op, out_value, upstream, nodes, ops, resident, tape_id, tape_epoch) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError>` は入力・出力とも**生 `Tensor<f32>`**（`Var` ではない）で計算する。`ops: &dyn BackendOps`（forward と同じカーネル。#1674 で elementwise VJP を `BackendOps` 経由化）と `eval::*`（ホスト参照実装。三角ソルブ等）を直接呼ぶのみで `Var`／`Tape::push_*` を一切使わない。**二階の計算グラフはテープ上に一切生成されない**（この事実自体は #1941 時点でも不変） | `crates/autodiff/src/grad.rs:188`（`vjp`）・`:127`（`vjp_elementwise_add`。ファイル総行数 10547 行へ拡大） |
| `Tape::backward_impl` は #1624（activation checkpointing）により、逆走査ループ全体を単一の不変借用で完結させる旧設計から**反復ごとに `Ref` を取得しその反復内で drop する**方式へ再構成済み。ただし「ある反復の処理中（`Ref` が生存している間）に同一テープへ `push_*`〈`borrow_mut()`〉で VJP 演算を記録できない」という結論（同一テープ方式 A-1 の困難の核心）は不変（`Ref` の生存期間が反復単位に狭まっただけ） | `crates/autodiff/src/backward.rs:119`（`backward`）・`:153`（`backward_accumulate`。#1749）・`:194`（`backward_impl`）。`docs/autodiff-checkpoint-design.md` §3.1 点 4 |
| `retain_graph`（グラフを `reset`／drop まで常時保持し複数回 `backward` を無条件成功させる契約）は #1749 で確定済み。追加 API は不要（`Tape::backward` を素朴に複数回呼べば成立）。`Tape::backward_accumulate`（同一世代の `Gradients` へ複数回分の backward 結果を蓄積する opt-in API）は `grad::vjp_elementwise_add` への委譲のみで新規 `Op`／`BackendOps` を伴わない。resident 勾配経路（`DeviceParamStore::backward`）由来の `Gradients` は fingerprint 契約と衝突するため `backward_accumulate` から fail-closed に拒否される | `crates/autodiff/src/backward.rs:153`。`docs/autodiff-retain-graph-accumulate-decision.md` |
| `Gradients { tape_id, epoch, grads, resident_fingerprint }` は `Vec<Option<Tensor<f32>>>` を保持する値型で `Tape` を借用しない。`Tape::reset()` は葉プレフィックスのみ `truncate` で残し `epoch` を 1 進める。`Gradients::get` は `tape_id`／`epoch` 不一致で `TapeMismatch` を返す fail-closed 設計 | `crates/autodiff/src/backward.rs`（`Gradients` 定義・`reset`／`get` 実装） |
| `Tape` は `Send` が静的アサーションで固定されている（`assert_send::<Tape>()`） | `crates/autodiff/tests/fusion_backend_integration.rs:391-392` |
| `Tape::var_no_grad`（#1748）・`Var::detach`（同 issue。専用 `Op::Detach` は追加せず既存葉ノード機構を再利用）・`TapeNode::requires_grad`（前方伝播）・`AutodiffError::GradientTrackingDisabled` が実装済み。`backward_impl` は `loss.requires_grad() == false` を実体化前に検査し `Err` を返す | `crates/autodiff/src/tape.rs:1897`（`var_no_grad`）・`:1702`（`requires_grad` フィールド）・`crates/autodiff/src/backward.rs:194-215`（`backward_impl` の requires_grad 検査）・`crates/autodiff/src/error.rs:72`（`GradientTrackingDisabled`） |
| `Op`（`pub(crate)` のクローズド enum）は 69 variant（旧版「少数」時点から線形代数・RNN・pooling・loss 系等の追加を経て拡大。§8 で全 variant を分類）。うち `QrQ{r}`／`QrR{q}`／`SvdU{s,vh}`／`SvdS`／`SvdVh`／`CrossEntropyLoss{targets: Tensor<i32>}` は**非追跡ペイロード**（`Tensor<f32>`／`Tensor<i32>` を直接持つ）を持つ。`LinearAct`／`LinearResident`／`ResidentLeaf`（reuse 経路）の d_weight は `ResidentResolver::fill_resident_weight_grad` がデバイス常駐 staging へ直接書き込み `Gradients` に一切現れない | `crates/autodiff/src/tape.rs:91-993`（`Op` enum 本体）・`:288`（`CrossEntropyLoss`）・`:348`（`ResidentLeaf`）・`:370`（`LinearResident`）・`:402`（`LinearAct`）・`:596,599`（`QrQ`／`QrR`）。`crates/autodiff/src/grad.rs:657`（`Op::LinearResident` match arm） |
| **VJP ヘルパー → Var 演算 対応表**（旧版「`Var` に公開 API として存在しないもの」を置き換え。前提 issue 完了によりいずれも既存 `Var` 演算で再表現可能）: `tanh_grad_factor`（1−y²）→ `1 - y.pow(2.0)`（`Var::pow`）または `y.mul(&y)` と `sub`／リテラル、`sigmoid_grad_factor`（y(1−y)）→ `y.mul(&(1 - y))` 相当、`mse_loss_scale` → スカラー倍（`Var` にスカラー乗算の合成は既存演算で可能）、`elementwise_mul_mask` → `Var::mul` と `Var::where_cond`／`masked_fill`（#1599）の組み合わせ、`reduce_to_shape`／`unreduce_broadcast`（broadcast の逆・keepdim 復元）→ `Var::sum_dims`／`Var::broadcast_to`（#1597／#1601）、`extremum_first_match_vjp`（`Max`／`Min` の先勝ち決定的散布）→ `Var::eq` と `Var::where_cond`（先勝ちタイ規則は `eq` マスクの単純合成では再現不可。§8「非対象」参照）、`softmax_vjp_along`／`log_softmax_vjp_along`（行ごとの内積と broadcast）→ `Var::mul`／`Var::sum_dims`／`Var::broadcast_to`、`cross_entropy_loss_vjp`（one-hot／gather 相当）→ `Var::one_hot`（#1755）／`Var::gather`（#1599）、線形代数 VJP（`inv`／`det`／`cholesky` の三角ソルブ等）→ 既存 `Var::solve`／`Var::matmul` 等はあるが三角ソルブ自体を `Var` 演算として直接合成する経路は未整備（§8「保留」） | `crates/autodiff/src/grad.rs:4335`（`tanh_grad_factor`）・`:4345`（`sigmoid_grad_factor`）・`:5514`（`mse_loss_scale`）・`:4101`（`elementwise_mul_mask`）・`:3956`（`reduce_to_shape`）・`:4055`（`reduce_bias_grad`）・`:5065`（`unreduce_broadcast`）・`:5176`（`extremum_first_match_vjp`）・`:4362`（`softmax_vjp_along`）・`:4408`（`log_softmax_vjp_along`）・`:5628`（`cross_entropy_loss_vjp`） |
| 遅延評価（`push_lazy`・`MAX_FUSED_CHAIN_LEN`）・`push_view`（reshape／transpose の再計算方式）は forward の elementwise 連鎖のみを対象とする。`docs/fusion-graph-design.md`「backward（VJP）は融合対象外（初期スコープ外）」が明示契約（HEAD でも同一の見出しが存在し不変） | `docs/fusion-graph-design.md:789`（見出し）・`:774,782,1873` |
| 公開面: `facade::Tape(pub(crate) fandhe_ai_autodiff::Tape)` の newtype・`Gradients`／`Var` を素で再エクスポート。`crates/facade/tests/api_surface.rs` が `pub use` での `Tape`／`BackendOps`／`new_with_ops` 再エクスポート禁止・`pub fn` が `BackendOps` を直接引数に取ることの禁止を機械検査する（両検査とも HEAD で健在） | `crates/facade/src/lib.rs:35-90`（doc comment）・`crates/facade/tests/api_surface.rs:66-113` |
| `AutodiffError`（`#[non_exhaustive]`）の variant は `Shape`／`Backward(String)`／`TapeMismatch`／`InvalidArgument(String)`／`Backend`／**`GradientTrackingDisabled`（#1748 追加）**／**`DeviceMismatch{requested, actual}`（#1614 追加）**。「（現時点で）未対応の演算」を表す専用 variant はなお存在せず、既存実装は `InvalidArgument` を「shape 検査より前に弾く構築時エラー」用途で使っている | `crates/autodiff/src/error.rs:19-83` |
| `Var::device()`／`Var::to(device)`（同一デバイス恒等・不一致は `DeviceMismatch`）・`Var::to_tape`（別 `Tape` への値転送。bit 完全一致・非微分境界）が #1614 で実装済み。子テープ方式（案 A-2）の「元テープの葉 ↔ 子テープの葉」対応の実装土台として利用できる | `crates/autodiff/src/var.rs:287`（`device`）・`:309`（`to`）。`docs/facade-device-transfer-enumeration-design.md` |
| REQ-12「利用者向け融合制御 API を提供しない」の受け入れ基準は `facade` を唯一の公開面としバックエンド結線を composition root に集約することで充足する設計（`create_graph` 相当のフラグを追加する場合、これが融合制御 API に該当しないことの整理が必要。§9 で整理） | `docs/spec/04-requirements.md:277-280`（submodule ポインタ不変につき行番号も不変） |
| PoC-v2-2（`docs/spec/03-poc/poc-v2-2-autodiff/README.md`）・`docs/public-api-design.md` §3 は 1 階の動的テープのみを確定しており高階微分への言及がない（本追記でも変更なしを grep で再確認） | `docs/spec/03-poc/poc-v2-2-autodiff/README.md`・`docs/public-api-design.md` |

## 3. 契約整理（設計が守るべき既存契約。不変）

1. 1 階の `Tape::backward` の結果は**bit 同一で不変**（既存 backward／parity／bit 一致テスト群を後退させない）。REQ-2 統一複合判定・FMA 契約・tolerance／baseline はいずれも不変
2. `docs/fusion-graph-design.md`「backward（VJP）は融合対象外」契約（現状は「VJP がテープに一切乗らない」というさらに強い状態であり、これを緩めて VJP をテープへ乗せる案は本契約の解釈拡張を伴う）
3. `Tape: Send`（`crates/autodiff/tests/fusion_backend_integration.rs:392`）は維持する
4. `TapeId`／`epoch` の世代契約（`Gradients::get` の fail-closed 検査）・`Tape::reset` の葉プレフィックス契約・`retain_graph`（#1749）契約は維持する
5. REQ-12「利用者向け融合制御 API を提供しない」・facade 公開面の機械検査（`api_surface.rs`）は維持する
6. reuse（デバイス常駐）経路・`Op::LinearResident`／`ResidentLeaf`・CUDA Graph capture（#1349）は高階微分の対象外とする（staging 書き込みは `Gradients` に現れず、二階の入力として使えない）
7. `TapeNode::requires_grad`（#1748）・`AutodiffError::GradientTrackingDisabled` の契約は維持する（子テープ方式でも requires_grad=false の葉から二階勾配を要求されたら fail-closed に拒否する）

## 4. 設計案の比較（旧版から不変。§5〜§9 で A-2 を一段具体化）

| 案 | 概要 | テープ再設計 | `RefCell` 借用の回避 | 前提 issue | §3 契約への影響 | 数値契約 | facade 公開面 | 実装難度 |
|---|---|---|---|---|---|---|---|---|
| **A-1: 同一テープへの `create_graph`** | `backward` 中に VJP 演算を同一 `Tape` へ `push_*` で記録する | 要（backward を「不変借用のまま逆走査」から「都度 borrow を解放する二相化」へ再設計） | 借用衝突を構造的に解消する必要（走査と記録を分離するか、`RefCell` を別のインデックス方式に置換） | 完了済み（#1593／#1597／#1599／#1601） | 契約 2（VJP を融合対象に含めるか要再定義）・契約 4（backward 中に生成したノードの世代扱い）に抵触。承認事項 | 変更なし（backward の実行順序自体は不変） | `Var` を返す新 API が必要 | 高（backward の内部設計を破壊的に変更） |
| **A-2: 子テープ（`create_graph` を新規 `Tape` へ記録）** | `backward` が既存 `Tape` とは別の新規 `Tape` を構築し、VJP を `Var` 演算としてそこへ記録して返す | 不要（既存 `backward_impl` のロック方式は温存し、外側からは別インスタンスとして扱う） | 解消（別インスタンスの `RefCell` のため衝突しない） | 完了済み（同上） | 契約 2・3（子 `Tape` の `ops`〈`Box<dyn BackendOps + Send>`〉の所有権をどこから得るか）は要検討だが契約自体の書き換えは不要 | 変更なし | `Tape::backward_create_graph(&self, loss) -> (Gradients, Tape, HashMap<NodeId, NodeId>)` 相当の新 API（§9 で具体化） | 高（VJP の `Var` 再表現＋子テープの `ops` 供給経路の設計） |
| **B: ネストテープ（TF `GradientTape` 方式）** | 外側テープが内側 `backward` の各 VJP 呼び出しを `Op::Backward{...}` 相当の 1 ノードとして記録し、その VJP（＝二階 VJP）を Op ごとに手書きする | 要（新 Op variant・二階 VJP の手書き実装が Op の数だけ必要） | 解消（内側 backward は独立して完結） | 手書き二階 VJP の網羅性検討が別途必要（前提 issue は A と同一集合＋線形代数・非追跡ペイロード Op の二階 VJP 追加） | 契約 6（非追跡ペイロード Op の二階微分は未定義のまま据え置きやすい）との整合は取りやすいが、Op 追加のたびに二階 VJP を保守する負担が恒久的に生じる | 変更なし | 新 API が必要 | 非常に高（Op ごとに二階 VJP を手書き・保守コスト大） |
| **C: forward-over-reverse（JVP／双対数）による HVP 限定提供** | Hessian-vector product のみを目的とし、`vjp(x)` の JVP（前方モード）を計算する。テープを介さない | 不要 | 該当なし（テープを使わない） | `Op` ごとの JVP 追加（`BackendOps` trait 拡張に相当。#1623〈custom Op〉の設計と関係） | 契約 2〜6 への抵触なし（VJP／テープの外側で完結） | 変更なし | HVP 専用の新 API（`facade` へ載せる場合は §9 の手続きが必要） | 中（HVP 用途に限定すれば Op 種別は絞れるが、`BackendOps` trait 拡張は承認事項） |
| **D: 有限差分による HVP の暫定提供** | `(grad(x + εv) − grad(x)) / ε` を数値的に計算する | 不要 | 該当なし | なし | 契約への抵触なし | 変更なし（tolerance は新規に定めない） | 載せない（診断用途限定） | 低（既存 `grad` の呼び出しを組み合わせるだけ） |
| **E: 段階 0 のまま維持** | 高階微分は未対応のまま留保する | 不要 | 該当なし | 前提 issue の完了が再開条件（達成済み） | 契約への抵触なし | 変更なし | 変更なし | 極小 |

## 5. 推奨（前提充足後・段階 1 設計確定）

自動運転での作業のため、以下は「推奨」であり**採用の確定はユーザー承認に委ねる**（§10）。

- 前提 issue（#1593・#1597・#1599・#1601・#1612）は**すべて CLOSED 済み**（§2）。段階 0（案 E・非対応の明文化）から**段階 1（設計確定・実装未着手）**へ位置づけを更新する
- **段階 1（本追記で設計確定・実装は別イシューで承認後着手）**: 主案として **案 A-2（子テープ方式の `create_graph`）**、代替として **案 C（HVP 限定 JVP）** を比較実装 issue の起票候補とする。対象 Op は elementwise・matmul・sum／max・softmax・MSE／CE に限定し、線形代数 Op（非追跡ペイロード）・reuse 経路・`LinearAct`・タイ規則を伴う extremum 系は初期スコープ外とする（§8 の対象 Op 分類表で全 Op を機械的に整理）
- **判断根拠**（旧版から不変）:
  - 同一テープ方式（案 A-1）は `backward_impl` の借用規律（#1624 で反復単位へ narrow 化されたが依然として「反復処理中の同一テープへの push は不可」という制約が残る。§2）を壊すため、backward の実行モデル自体の再設計を要する。子テープ方式（案 A-2）は既存 `backward_impl` を温存したまま「二階の勾配グラフをどこに置くか」という論点だけを切り出せるため、既存契約への影響が最小
  - ネストテープ方式（案 B）は Op 追加のたびに二階 VJP を手書きで保守する恒久コストを生む。本リポは `Op` を頻繁に拡張しており（69 variant まで拡大。§2）、保守負担の増大が大きい
  - 案 C（JVP／HVP 限定）はテープ再設計が不要で契約への影響が最小だが、提供できる機能が HVP に限られ「grad of grad」の一般形（任意階数の `create_graph`）には届かない。段階 1 の代替案として位置づける
  - 案 D（有限差分）は実装難度が最小だが数値精度が ε 依存であり REQ-2 の統一複合判定の対象にできない。診断・デバッグ用途の暫定手段としてのみ有効

## 6. 数値一致・既存テストとの整合

- 1 階 backward の不変性テスト（既存 `crates/autodiff/tests/backward.rs`・`tape_recording.rs` 等）の非後退を、段階 1 実装 issue の受入条件として引き継ぐ
- 二階検証テスト案（段階 1 で具体化）: 小形状（2×2〜4×4 程度）での解析解（手計算または `f64` ホスト参照実装）との突合、案 D（有限差分）との相互検証を組み合わせる
- **新規 tolerance は本追記でも定めない**。段階 1 で二階微分専用の許容誤差・baseline が必要になった場合は、既存の tolerance／baseline 変更と同様にユーザー承認を要する（§10）

## 7. `ops` 供給方式の比較（#1941 追記・§9 API 契約の前提）

子テープ（案 A-2）を構築する際、子テープの `ops: Box<dyn BackendOps + Send>`（`crates/autodiff/src/tape.rs:1751`）をどこから得るかが実装上の論点になる。

| 選択肢 | 概要 | trait／bound 変更 | 評価 |
|---|---|---|---|
| **(i) 呼び出し側が子 `Tape` を渡す（推奨）** | `backward_create_graph(&self, loss: &Var<'_>, child: &Tape) -> Result<(Gradients, HashMap<NodeId, NodeId>), AutodiffError>` 相当のシグネチャとし、呼び出し側があらかじめ `Tape::new(ops)`（または `Tape::default()`）で構築した子テープを渡す。`child.device() == self.device()` を既存 `Var::device`／`Var::to` と同型の検査（`AutodiffError::DeviceMismatch`）で検査する | なし（`BackendOps` に `Clone`／`Sync` 等を要求しない） | trait 拡張ゼロで既存の `Tape::device`（#1614）をそのまま転用できる。呼び出し側が「同じバックエンドの新規テープを用意する」責務を負うが、facade 側で `tape.child()` 相当のヘルパーを 1 つ用意すれば利用者体験は損なわれない |
| (ii) `Arc<dyn BackendOps + Send + Sync>` 共有 | `Tape` 内部の `ops` フィールドを `Box` から `Arc` へ変更し、子テープと親テープで同一インスタンスを共有する | `BackendOps: Sync` の追加要求（既存 3 バックエンド実装が `Sync` を満たすかの確認が必要）・`Tape` 内部表現の破壊的変更 | 内部実装の破壊的変更を伴い影響範囲が広い。承認事項が増える |
| (iii) `BackendOps::box_clone` 追加 | trait に `fn box_clone(&self) -> Box<dyn BackendOps + Send>` を追加し、子テープ構築時に親の `ops` を複製する | trait 拡張（3 バックエンド実装への追加実装が必要） | 3 クレート（backend-cpu／cuda／metal）すべてに実装が要る。`BackendOps` は非公開契約であり trait 拡張自体は REQ-12 に抵触しないが、実装コストが (i) より高い |

(i) を主案として §9 の API 契約を設計する（trait・bound の変更を要しないため §3 契約 5〈facade 公開面の機械検査〉への影響が最小）。

## 8. 対象 Op 分類表（#1941 追記・受入条件 1）

`crates/autodiff/src/tape.rs:91-993` の `Op` enum 69 variant を、次の基準で「対象（初期スコープ）」「非対象」「保留」に分類する。基準:

- (a) VJP が既存 `Var` 演算の合成で表現できる（§2「VJP ヘルパー → Var 演算」対応表）
- (b) resident／fused 経路でない
- (c) 非追跡ペイロードを持たない
- (d) 合成では再現できない数値契約（`f64` アキュムレータ縮約契約・先勝ちタイ規則等）を持たない

| 分類 | Op | 根拠 |
|---|---|---|
| **対象（初期スコープ）** | `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`ScalarUnary`・`ScalarBinary`・`MatMul`・`Sum`（`dim=None` または単一軸）・`Reshape`・`Transpose`・`Permute`・`BroadcastTo`・`Narrow`・`Concat`・`Contiguous`・`Softmax`・`LogSoftmax`・`Where`・`MaskedFill`・`Gather`・`Scatter`・`MseLoss`・`CrossEntropyLoss`（target は非微分入力として扱う。(c) の非追跡ペイロードは入力側であり出力勾配の合成自体は可能） | (a)(b)(d) を満たす。VJP は §2 の対応表がすでに `Var` 演算への分解を示す。`CrossEntropyLoss` は target を非微分入力として扱えば合成可能（PyTorch の `create_graph=True` でも target 自体には勾配は流れない） |
| **非対象（resident／fused／CUDA Graph）** | `ResidentLeaf`・`LinearResident`・`LinearAct` | 契約 6（§3）。d_weight がデバイス常駐 staging（`ResidentResolver::fill_resident_weight_grad`）へ直接書き込まれ `Gradients` に現れないため、二階の入力として使えない。CUDA Graph capture（#1349）区間も同様に非対象 |
| **非対象（非追跡ペイロード）** | `QrQ{r}`・`QrR{q}`・`SvdU{s,vh}`・`SvdS`・`SvdVh` | (c) 抵触。多出力ノードが `Tensor<f32>` を直接ペイロードとして持ち、それ自体をテープ上の追跡演算として再構成する経路が未整備 |
| **非対象（合成では再現できない数値契約）** | `Max`（先勝ちタイ規則。`extremum_first_match_vjp`）・`Min`（同上）・`Var`（分散。`f64` アキュムレータ縮約契約）・`Std`・`VectorNorm`・`MatrixNorm`・`RmsNorm`・`LayerNorm`・`BatchNorm`・`Conv2d`（d_weight／d_bias の `f64` 縮約契約）・`AvgPool2d`・`AdaptiveAvgPool2d`（`f64` 相当縮約契約） | (d) 抵触。`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレータで統一する」契約・`extremum_first_match_vjp` の先勝ち決定的方式（#1718）は、既存 `Var` 演算（`eq` マスク等）の単純合成では bit 再現できない。子テープで再構成すると縮約順序が変わり 1 階と異なる丸め誤差を生む可能性がある |
| **保留（合成経路が未整備）** | `Inv`・`Solve`・`Det`・`Cholesky`・`Dropout`（マスク再利用契約）・`Sort`・`Topk`・`Cumprod`・`Embedding`・`Interpolate`・`MaxPool2d`（索引ベース VJP）・`RnnCell`・`LstmCell`・`LstmHidden`・`GruCell`（時系列ループの再構成）・`HuberLoss`・`BceLoss`・`NllLoss`・`KlDivLoss`（融合カーネル VJP。§2 対応表に含まれない） | (a) 未確認または (d) 未評価。三角ソルブ等の VJP を `Var` 演算として直接合成する経路が未整備（§2）。線形代数系は checkpoint（#1624）と同様に別途の合成方式を要する可能性がある |
| **非対象（非微分演算）** | `OneHot` | VJP が明示ゼロの非微分演算のため、二階微分の対象そのものが存在しない |

**checkpoint 区間（#1624）との相互作用**: `Tape::checkpoint`／`Var::checkpoint_from` で構成した区間は forward 値を破棄し `recompute_value` で再計算する。子テープ方式で二階勾配を取る際、checkpoint 済みノードの VJP（1 階）を子テープへ記録するには forward 値の再計算（1 階と同じ `recompute_value` 経路）が必要になり、対象 Op（`MatMul`／`Sigmoid`／`Sum`／`Max`）が checkpoint 対象と重なる場合は追加の設計整理を要する（本追記では踏み込まず段階 1 実装 issue へ引き継ぐ）。

**`var_no_grad`／`detach` 葉との相互作用**: `TapeNode::requires_grad == false` の葉（#1748）は 1 階の `backward` 自体が到達を拒否する契約であり、子テープ方式でも同じ検査を子テープ構築前に行う（契約 7）。

**機構案**: `Op::is_checkpoint_eligible()`（`crates/autodiff/src/tape.rs:1217`）・`Op::for_each_input`（`:1395`）と同型の網羅 match 述語 `Op::supports_create_graph() -> bool` を新設し、非対象 Op に到達した場合は `backward_create_graph` が fail-closed で型付きエラーを返す設計とする（承認事項。§10）。

## 9. API 契約節（#1941 追記・受入条件 2）

以下は「契約案」であり、決裁が要る点はすべて §10 承認事項へ列挙する。

- **シグネチャ案**: `Tape::backward_create_graph(&self, loss: &Var<'_>, child: &Tape) -> Result<CreateGraphResult, AutodiffError>`（`ops` 供給方式は §7 (i) を採用。`child` は呼び出し側があらかじめ構築した空の `Tape`）
- **戻り値**: 1 階 `Gradients`（既存 `backward_impl` の無変更な結果）＋子テープ上に構築した勾配 `Var<'child>`（対象ノードごと `Option`）＋「元テープの葉ノード ↔ 子テープの葉ノード」対応表（`HashMap<NodeId, NodeId>` 等）。元テープの葉を子テープへ写す方法は `Var::to_tape`（#1614。非微分境界・bit 完全一致）を用い、子テープ側で改めて葉として登録する
- **1 階勾配の bit 同一性**（既存契約 1）: 選択肢 (i) 1 階 `Gradients` は無変更の `backward_impl` で得て、子グラフは別パス（`Op::supports_create_graph()` を満たすノードのみを対象に、子テープへ `Var` 演算として再記録する専用の走査）で構築する。子グラフ上で計算した値は 1 階の `Gradients` と REQ-2 統一複合判定で突合する（**推奨**。bit 同一が自明）／選択肢 (ii) 単一パス化し CPU のみ bit 同一を要求する（Metal NT/TN matmul VJP〈#1215〉は REQ-2 契約のため bit 不成立になり、バックエンド間で判定方式が割れる）。採否は決定しない（承認事項）
- **世代契約**: 子テープは独立の `TapeId`／`epoch` を持つ（親と衝突しない）。親の `reset()` 後も子テープは独立して有効（`retain_graph` 契約〈#1749〉と同様に子テープ自身の `reset`／drop まで保持）。`backward_accumulate`（#1749）との併用は対象外とする（子テープ上の勾配蓄積は別途検討）。`Tape: Send` は子テープ・親テープとも維持する
- **エラー契約**: `Op::supports_create_graph() == false` の Op へ到達した場合は fail-closed。既存 `AutodiffError::Backward(String)` を流用する案と、専用 variant（例: `UnsupportedForCreateGraph`）を新設する案を比較する（`AutodiffError` は `#[non_exhaustive]` のため新設自体は破壊的変更にならないが、facade 再エクスポート型の変更は承認事項）。`requires_grad == false` の葉に到達した場合は既存 `GradientTrackingDisabled` をそのまま再利用する
- **facade**: newtype `Tape`（`crates/facade/src/lib.rs`）が子テープもラップする設計とし、`new_with_ops` は引き続き非公開のまま（`api_surface.rs` の機械検査を維持）。`docs/compat-api-scope.md` §5「範囲拡張の手続き」の対象となる（正本 REQ-9 改定または本リポのユーザー承認を経る必要がある）
- **REQ-12 整理**: `create_graph`（二階の勾配グラフを構築するか否かの選択）は「カーネル融合の制御」（`docs/fusion-graph-design.md` §3.3 の融合境界）とは別軸である。`docs/spec/04-requirements.md:280`「autodiff の ops 受け取り構築子はサポート外の内部 API」と同型の整理により、`backward_create_graph` は融合制御 API に該当しないと整理する（承認事項として明記）
- **`docs/fusion-graph-design.md` §3.3 との整合**: 子テープに記録された VJP 演算（`Var` 演算として再表現されたもの）は、子テープ自身の forward 経路として通常の融合対象になりうる（§3.3「backward（VJP）は融合対象外」は**親テープの 1 階 backward**を指す契約であり、子テープ上で `Var` 演算として明示的に記録された二階の「forward」相当の計算列には及ばない、という解釈を採る）。この解釈自体を承認事項とする
- **検証方針案**: 小形状の解析解・`f64` ホスト参照実装・有限差分（案 D）との相互検証。**新規 tolerance／baseline は定めない**

## 10. 承認事項（実装着手の前提）

1. `docs/fusion-graph-design.md` §3.3「VJP は融合対象外」契約の解釈拡張（§9「`docs/fusion-graph-design.md` §3.3 との整合」で示した「子テープ上の再表現には及ばない」という解釈の採否）
2. `ops` 供給方式（§7 の (i)〜(iii) のいずれを採るか。推奨は (i)）
3. 1 階勾配の bit 同一性の判定方式（§9「1 階勾配の bit 同一性」の選択肢 (i)／(ii) のいずれを採るか。推奨は (i)）
4. エラー契約（既存 `AutodiffError::Backward` 流用か専用 variant 新設か）
5. facade 公開面への高階 API 追加（`docs/compat-api-scope.md` §5 手続き）
6. 対象 Op の初期スコープ（§8 の分類表。「対象」区分を実装対象として確定するか）
7. 二階微分の数値判定方式（新規 tolerance／baseline を伴う場合）
8. `Op::supports_create_graph()` 機構の新設可否（§8「機構案」）
9. 段階 1 実装 issue の起票

## 11. スコープ外（不変）

- reuse／resident 経路（`LinearResident`／`ResidentLeaf`）・CUDA Graph capture（#1349）・線形代数 Op の高階微分
- 混合精度（#1625。実装済み〈#1721/#1722〉だが高階微分との相互作用は未検討）・activation checkpointing（#1624。再計算との併用は §8 で言及に留める）
- custom autograd Function（#1623。`Op` enum 拡張の論点は重なるが、本 doc では参照に留め決定しない）

## 12. 出典

- Issue #1622・#1573・#1570・#1941
- `docs/compat-feature-gap.md:312`（§2.11）
- `docs/compat-api-scope.md:248`（§1.3）
- `docs/spec/04-requirements.md:232`（REQ-9 2026-09-12 追記）・`docs/spec/04-requirements.md:277-280`（REQ-12）
- `docs/fusion-graph-design.md:789`（backward は融合対象外の見出し）・`:774,782,1873`
- `docs/autodiff-view-recompute-decision.md`
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`
- `docs/autodiff-retain-graph-accumulate-decision.md`
- `docs/autodiff-checkpoint-design.md` §3.1 点 4
- `docs/autodiff-amax-grad-distribution-decision.md`（`extremum_first_match_vjp` の先勝ち決定的方式）
- `docs/facade-device-transfer-enumeration-design.md`（`Var::device`／`to`／`to_tape`）
- `docs/device-resident-update-design.md` §3.3b
- `docs/public-api-design.md` §3
- `docs/spec/03-poc/poc-v2-2-autodiff/README.md`
- `crates/autodiff/src/grad.rs:127,188,3956,4055,4101,4335,4345,4362,4408,5065,5176,5514,5628`
- `crates/autodiff/src/backward.rs:119,153,194-215`
- `crates/autodiff/src/tape.rs:91-993,1217,1395,1702,1751,1897,2013,2063,2143`
- `crates/autodiff/src/error.rs:19-83`
- `crates/autodiff/src/var.rs:287,309`
- `crates/autodiff/tests/fusion_backend_integration.rs:391-392`
- `crates/facade/src/lib.rs:35-90`
- `crates/facade/tests/api_surface.rs:66-113`

内部ホスト名・秘密情報は含めない。
