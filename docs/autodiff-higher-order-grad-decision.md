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
| **VJP ヘルパー → Var 演算 対応表**（旧版「`Var` に公開 API として存在しないもの」を置き換え。前提 issue 完了によりいずれも既存 `Var` 演算で再表現可能）: `tanh_grad_factor`（1−y²）→ `1 - y.pow(2.0)`（`Var::pow`）または `y.mul(&y)` と `sub`／リテラル、`sigmoid_grad_factor`（y(1−y)）→ `y.mul(&(1 - y))` 相当、`mse_loss_scale` → スカラー倍（`Var` にスカラー乗算の合成は既存演算で可能）、`elementwise_mul_mask` → `Var::mul` と `Var::where_cond`／`masked_fill`（#1599）の組み合わせ、`reduce_to_shape`／`unreduce_broadcast`（broadcast の逆・keepdim 復元）→ `Var::sum_dims`／`Var::broadcast_to`（#1597／#1601）、`extremum_first_match_vjp`（`Max`／`Min` の先勝ち決定的散布）→ `Var::eq` と `Var::where_cond`（先勝ちタイ規則は `eq` マスクの単純合成では再現不可。§8「非対象」参照）、`softmax_vjp_along`／`log_softmax_vjp_along`（行ごとの内積と broadcast）→ 形の上では `Var::mul`／`Var::sum_dims`／`Var::broadcast_to` へ分解できるが、両ヘルパーは縮約後の減算・乗算まで `f64` で保持し最後に 1 回だけ `f32` へ downcast する契約（`.claude/rules/coding-rust.md` の縮約契約と同型。有限入力〈例 `y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]`〉でも overflow を防ぐための設計）を持ち、`Var` 演算の合成（各演算が `f32` で中間値を持つ）では同じ有限性を再現できないため、既存 `Var` 合成では再現できない数値契約を持つ（§8「非対象」参照）、`cross_entropy_loss_vjp`（one-hot／gather 相当）→ `Var::one_hot`（#1755）／`Var::gather`（#1599）、線形代数 VJP（`inv`／`det`／`cholesky` の三角ソルブ等）→ 既存 `Var::solve`／`Var::matmul` 等はあるが三角ソルブ自体を `Var` 演算として直接合成する経路は未整備（§8「保留」） | `crates/autodiff/src/grad.rs:4335`（`tanh_grad_factor`）・`:4345`（`sigmoid_grad_factor`）・`:5514`（`mse_loss_scale`）・`:4101`（`elementwise_mul_mask`）・`:3956`（`reduce_to_shape`）・`:4055`（`reduce_bias_grad`）・`:5065`（`unreduce_broadcast`）・`:5176`（`extremum_first_match_vjp`）・`:4362`（`softmax_vjp_along`）・`:4408`（`log_softmax_vjp_along`）・`:5628`（`cross_entropy_loss_vjp`） |
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
- **段階 1（本追記で設計確定・実装は別イシューで承認後着手）**: 主案として **案 A-2（子テープ方式の `create_graph`）**、代替として **案 C（HVP 限定 JVP）** を比較実装 issue の起票候補とする。対象 Op の一覧は本節では列挙せず §8 の対象 Op 分類表を唯一の定義として参照する（elementwise・matmul・sum 等の合成可能な Op が対象である一方、softmax／max 等の「合成では再現できない数値契約」を持つ Op は §8 の是正〈#1941〉により非対象へ分類済み。線形代数 Op〈非追跡ペイロード〉・reuse 経路・`LinearAct` も同表の非対象区分に含まれる）。
- **判断根拠**（旧版から不変）:
  - 同一テープ方式（案 A-1）は `backward_impl` の借用規律（#1624 で反復単位へ narrow 化されたが依然として「反復処理中の同一テープへの push は不可」という制約が残る。§2）を壊すため、backward の実行モデル自体の再設計を要する。子テープ方式（案 A-2）は既存 `backward_impl` を温存したまま「二階の勾配グラフをどこに置くか」という論点だけを切り出せるため、既存契約への影響が最小
  - ネストテープ方式（案 B）は Op 追加のたびに二階 VJP を手書きで保守する恒久コストを生む。本リポは `Op` を頻繁に拡張しており（69 variant まで拡大。§2）、保守負担の増大が大きい
  - 案 C（JVP／HVP 限定）はテープ再設計が不要で契約への影響が最小だが、提供できる機能が HVP に限られ「grad of grad」の一般形（任意階数の `create_graph`）には届かない。段階 1 の代替案として位置づける
  - 案 D（有限差分）は実装難度が最小だが数値精度が ε 依存であり REQ-2 の統一複合判定の対象にできない。診断・デバッグ用途の暫定手段としてのみ有効

（#2545／#2546 追記）主案 A-2 は facade へ公開済み。実装記録は §18・§19。

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
| **対象（初期スコープ）** | `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`ScalarUnary`・`ScalarBinary`・`MatMul`・`Sum`（`dim=None` または単一軸）・`Mean`（`dim=None` または単一軸）・`Reshape`・`Transpose`・`Permute`・`BroadcastTo`・`Narrow`・`Concat`・`Contiguous`・`Where`・`MaskedFill`・`Gather`・`Scatter`・`Pad`・`MseLoss`・`CrossEntropyLoss`（`CrossEntropyLoss` の整数ラベル target は非微分入力として扱う。`MseLoss` の浮動小数 target は追跡可能な入力で、`dTarget=-dPred` の寄与を維持する。(c) の非追跡ペイロードは入力側であり出力勾配の合成自体は可能） | (a)(b)(d) を満たす。VJP は §2 の対応表がすでに `Var` 演算への分解を示す。`Mean` は `mean_vjp`（`crates/autodiff/src/grad.rs:5118`）が `Sum` の VJP（`unreduce_broadcast`）を `1/n` でスケールしたものと一致し追加の数値契約を持たないため `Sum` と同じ扱いとする。`Pad` は `pad_vjp`（`:1568`）が `Op::Narrow` の forward と同じ `narrow` 呼び出し連鎖のみで構成され算術を含まないため `Narrow` と同じ扱いとする。`CrossEntropyLoss` は整数ラベル target が非微分入力のため合成可能（PyTorch の `create_graph=True` でも整数ラベルには勾配は流れない）。浮動小数の target を持つ損失（`MseLoss`・`PoissonNllLoss`・`GaussianNllLoss` 等）は target を追跡可能な入力として扱い、非微分と一律に見なさない（§21.3 #3082）。`Softmax`／`LogSoftmax` は #1941 是正により本行から除外し「非対象（合成では再現できない数値契約）」へ移した（下表参照） |
| **非対象（resident／fused／CUDA Graph）** | `ResidentLeaf`・`LinearResident`・`LinearAct` | 契約 6（§3）。d_weight がデバイス常駐 staging（`ResidentResolver::fill_resident_weight_grad`）へ直接書き込まれ `Gradients` に現れないため、二階の入力として使えない。CUDA Graph capture（#1349）区間も同様に非対象 |
| **非対象（非追跡ペイロード）** | `QrQ{r}`・`QrR{q}`・`SvdU{s,vh}`・`SvdS`・`SvdVh` | (c) 抵触。多出力ノードが `Tensor<f32>` を直接ペイロードとして持ち、それ自体をテープ上の追跡演算として再構成する経路が未整備 |
| **非対象（合成では再現できない数値契約）** | `Softmax`・`LogSoftmax`（`softmax_vjp_along`／`log_softmax_vjp_along` が縮約後の減算・乗算まで `f64` で保持し最後に 1 回だけ `f32` へ downcast する契約。§2）・`Max`（先勝ちタイ規則。`extremum_first_match_vjp`）・`Min`（同上）・`Var`（分散。`f64` アキュムレータ縮約契約）・`Std`・`VectorNorm`・`MatrixNorm`・`RmsNorm`・`LayerNorm`・`BatchNorm`・`Conv2d`（d_weight／d_bias の `f64` 縮約契約）・`ConvTranspose2d`（同上。イシュー #2067）・`AvgPool2d`・`AdaptiveAvgPool2d`（`f64` 相当縮約契約）・`Cumsum`（`cumsum_vjp_along`〈`:4654`〉が lane ごと `f64` アキュムレータで逆順累積和を構築する契約） | (d) 抵触。`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレータで統一する」契約・`extremum_first_match_vjp` の先勝ち決定的方式（#1718）は、既存 `Var` 演算（`eq` マスク等）の単純合成では bit 再現できない。子テープで再構成すると縮約順序が変わり 1 階と異なる丸め誤差を生む可能性がある。`Softmax`／`LogSoftmax` も同型: `y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` のような有限入力で `softmax_vjp_along` は有限値を返すが、`Var::mul`／`Var::sum_dims`／`Var::broadcast_to` への素朴な合成（各演算が `f32` で中間値を持つ）では途中の乗算・減算が overflow し `inf`／`NaN` を生む（#1941 是正。codex-review 指摘） |
| **保留（合成経路が未整備）** | `Inv`・`Solve`・`Det`・`Cholesky`・`Dropout`（マスク再利用契約）・`Sort`・`Topk`・`Cumprod`・`Embedding`・`Interpolate`・`MaxPool2d`（索引ベース VJP）・`RnnCell`・`LstmCell`・`LstmHidden`・`GruCell`（時系列ループの再構成）・`HuberLoss`・`BceLoss`・`NllLoss`・`KlDivLoss`（融合カーネル VJP。§2 対応表に含まれない） | (a) 未確認または (d) 未評価。三角ソルブ等の VJP を `Var` 演算として直接合成する経路が未整備（§2）。線形代数系は checkpoint（#1624）と同様に別途の合成方式を要する可能性がある |
| **非対象（非微分演算）** | `OneHot` | VJP が明示ゼロの非微分演算のため、二階微分の対象そのものが存在しない |

上記で 69 variant すべて（対象 26・非対象〈resident/fused〉3・非対象〈非追跡ペイロード〉5・非対象〈数値契約〉15・保留 19・非対象〈非微分〉1）を分類済み。

**70 番目の variant（`Op::Custom`。イシュー #1946。上記分類表は本 variant 新設前の
69 variant 時点のもの）**: `custom autograd Function` プラグイン機構（案 B。
`docs/autodiff-custom-function-decision.md`）の実装で追加された `Op::Custom { inputs,
func: CustomFn }` も「非対象」に分類する。`CustomFunction::backward` はユーザー提供の
数値関数（上流勾配のテンソルを受け取り入力勾配のテンソルを返すのみ）であり、
(a) の基準「VJP が既存 `Var` 演算の合成で表現できる」を構造的に満たさない
（ブラックボックスの host 計算であり `Var` 演算列として再生する手段がない）。
詳細・実装記録は `docs/autodiff-custom-function-decision.md` §14 を参照。

**checkpoint 区間（#1624）との相互作用**: `Tape::checkpoint`／`Var::checkpoint_from` で構成した区間は forward 値を破棄し `recompute_value` で再計算する。子テープ方式で二階勾配を取る際、checkpoint 済みノードの VJP（1 階）を子テープへ記録するには forward 値の再計算（1 階と同じ `recompute_value` 経路）が必要になり、対象 Op（`MatMul`／`Sigmoid`／`Sum`／`Max`）が checkpoint 対象と重なる場合は追加の設計整理を要する（本追記では踏み込まず段階 1 実装 issue へ引き継ぐ）。

**`var_no_grad`／`detach` 葉との相互作用**（#1941 是正: 到達拒否と勾配要求拒否の区別）: 1 階 `backward_impl` は非追跡葉（`requires_grad == false`）への**到達自体は拒否しない**——`backward.rs:316-333` は当該ノードへの寄与を `accumulate` へ渡さず静かに捨てるのみで（`grads[target]` は `None` のまま）、走査自体は継続する。これは `loss = x.mul(&x).mul(&c)`（`x` は学習対象・`c` は `var_no_grad` の定数）のような通常の二階微分パターンで `c` を定数として扱うために必要な挙動であり、拒否してしまうと成立しなくなる。拒否されるのは別の 2 点のみ（実装値を実測して訂正。#1941 追加是正）: (i) **`loss` 自身**が非追跡の場合（`Tape::backward` 冒頭の `requires_grad` 検査。`backward.rs:209-215` の実装は `Err(AutodiffError::Backward(String))` を返す——「勾配追跡対象の祖先を持たない」旨のメッセージ付きであり `GradientTrackingDisabled` ではない）、(ii) 非追跡の葉**自体の勾配値を明示的に要求**した場合（`Gradients::get`。`backward.rs:87-89` の実装は `Err(AutodiffError::GradientTrackingDisabled)` を返す——こちらは「未到達」（`Ok(None)`）とは型で区別される別の分岐であり、`None` を返す契約ではない）。子テープ方式でも同じ区別を踏襲し、子テープ構築前に検査するのは (i)（`loss` の requires_grad）のみとする。非追跡葉への到達自体（寄与の破棄）は子テープでも 1 階と同じ挙動（当該葉への VJP 記録をスキップし定数として扱う）とし、拒否しない（契約 7 を上記のとおり精密化）。

**機構案**: `Op::is_checkpoint_eligible()`（`crates/autodiff/src/tape.rs:1217`）・`Op::for_each_input`（`:1395`）と同型の網羅 match 述語 `Op::supports_create_graph() -> bool` を新設し、非対象 Op に到達した場合は `backward_create_graph` が fail-closed で型付きエラーを返す設計とする（承認事項。§10）。

## 9. API 契約節（#1941 追記・受入条件 2）

以下は「契約案」であり、決裁が要る点はすべて §10 承認事項へ列挙する。

- **シグネチャ案**: `Tape::backward_create_graph(&self, loss: &Var<'_>, child: &Tape) -> Result<CreateGraphResult, AutodiffError>`（`ops` 供給方式は §7 (i) を採用。`child` は呼び出し側があらかじめ構築した空の `Tape`）
- **戻り値**: 1 階 `Gradients`（既存 `backward_impl` の無変更な結果）＋子テープ上に構築した勾配 `Var<'child>`（対象ノードごと `Option`）＋「元テープの葉ノード ↔ 子テープの葉ノード」対応表（`HashMap<NodeId, NodeId>` 等）。元テープの葉を子テープへ写す方法は `Var::to_tape`（#1614。非微分境界・bit 完全一致）を用い、子テープ側で改めて葉として登録する
- **1 階勾配の bit 同一性**（既存契約 1）: 選択肢 (i) 1 階 `Gradients` は無変更の `backward_impl` で得て、子グラフは別パス（`Op::supports_create_graph()` を満たすノードのみを対象に、子テープへ `Var` 演算として再記録する専用の走査）で構築する。子グラフ上で計算した値は 1 階の `Gradients` と REQ-2 統一複合判定で突合する（**推奨**。bit 同一が自明）／選択肢 (ii) 単一パス化し CPU のみ bit 同一を要求する（Metal NT/TN matmul VJP〈#1215〉は REQ-2 契約のため bit 不成立になり、バックエンド間で判定方式が割れる）。採否は決定しない（承認事項）
- **世代契約**: 子テープは独立の `TapeId`／`epoch` を持つ（親と衝突しない）。親の `reset()` 後も子テープは独立して有効（`retain_graph` 契約〈#1749〉と同様に子テープ自身の `reset`／drop まで保持）。`backward_accumulate`（#1749）との併用は対象外とする（子テープ上の勾配蓄積は別途検討）。`Tape: Send` は子テープ・親テープとも維持する
- **エラー契約**: `Op::supports_create_graph() == false` の Op へ到達した場合は fail-closed。既存 `AutodiffError::Backward(String)` を流用する案と、専用 variant（例: `UnsupportedForCreateGraph`）を新設する案を比較する（`AutodiffError` は `#[non_exhaustive]` のため新設自体は破壊的変更にならないが、facade 再エクスポート型の変更は承認事項）。**`requires_grad == false` の葉への到達自体は拒否しない**（§8「`var_no_grad`／`detach` 葉との相互作用」の区別と整合させる。#1941 追加是正）: 子テープ構築前に検査するのは `loss` 自身の requires_grad のみ（§8 (i)。実装は `Err(AutodiffError::Backward(String))` を返す）であり、既存 `GradientTrackingDisabled`（§8 (ii)。`Gradients::get` が非追跡ノードに対して返す型）を子テープ構築の到達時エラーとして再利用するわけではない。子テープ側の `Gradients::get` 相当 API を設ける場合、そこでの明示要求時のみ `GradientTrackingDisabled` を再利用する対象になりうる（承認事項）
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

（#2545／#2546 追記）承認事項 5 について、ルート #2499 本文の一括承認（2026-10-04）の範囲として #2545 で §17.2 の確定形を公開した。記録は §17〜§19。本追記は他の承認事項の状態には言及しない。

## 11. スコープ外（不変）

- reuse／resident 経路（`LinearResident`／`ResidentLeaf`）・CUDA Graph capture（#1349）・線形代数 Op の高階微分
  （#3086 で `Inv`・`Solve`・`Det`・`SlogdetLogAbsDet` は対象へ。残る線形代数 Op の扱いは §30）
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

## 13. 実装記録（#1942・段階 2「部分実装」）

自動運転（ユーザー承認待ちを介さない実装 issue）のため、§10 承認事項の
うち内部クレート限定の範囲（trait／API の新規追加は `fandhe_ai_autodiff`
非公開のまま）に限定して実装した。**facade 公開（承認事項 5）は未承認の
まま不実施**。

- **実装物**: `crates/autodiff/src/create_graph.rs`（新規モジュール）に
  `Tape::backward_create_graph(&self, loss: &Var<'_>, child: &'c Tape) ->
  Result<CreateGraphResult<'c>, AutodiffError>` と戻り値型
  `CreateGraphResult<'c>`（`first_order()`・`grad()`・`child_var()`）を
  実装した。`tape.rs` に `Op::supports_create_graph()`（新設。
  `pub(crate)`・網羅 match）・`Tape::has_registered_checkpoints()`（新設）
  を追加し、`Op::for_each_input` を `pub(crate)` へ可視性緩和した
  （本体・網羅性は無変更）。
- **§10 承認事項の採否（本実装時点の確定）**:
  1. `docs/fusion-graph-design.md` §3.3 解釈拡張 → 採用（子テープ上の
     再生は対象外という解釈のまま実装）。
  2. `ops` 供給方式 → §7 の (i)（呼び出し側が子 `Tape` を渡す）を採用。
  3. 1 階勾配の bit 同一性 → §9 の選択肢 (i)（2 パス方式）を採用。
     `backward_create_graph` は既存 `Tape::backward`（無変更）をそのまま
     呼ぶため、1 階勾配は単体で `backward` を呼んだ場合と bit 同一
     （`tests/create_graph.rs::create_graph_first_order_matches_plain_
     backward_and_leaves_parent_intact` で確認）。
  4. エラー契約 → 既存 `AutodiffError::Backward(String)` を流用（専用
     variant は新設せず）。
  5. facade 公開 → **未承認のまま不実施**（内部クレート限定）。
  6. 対象 Op の初期スコープ → `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・
     `Tanh`・`Sigmoid`・`Sum`・`Mean`・`Reshape`・`BroadcastTo` の 11
     variant のみ実装（§8「対象」区分のうち `MatMul`・
     `ScalarUnary`／`ScalarBinary`・`Transpose`／`Permute`／`Narrow`／
     `Concat`／`Contiguous`／`Where`／`MaskedFill`／`Gather`／
     `Scatter`／`Pad`／`MseLoss`／`CrossEntropyLoss` は未実装のまま
     `Op::supports_create_graph() == false` に残し、#1943 等の後続
     イシューへ引き継ぐ）。
  7. 二階微分の数値判定方式 → 新規 tolerance／baseline は定めない
     （§6 の方針どおり）。二階側の子テープ上の演算列は既存 VJP
     ヘルパー（`grad.rs`）とは独立の実装（`Var` 演算の合成）であり
     bit 同一は主張しない。正しさは有限差分突合（1 階解析勾配
     `Tape::backward` の中央差分。`create_graph` を経由しない独立経路）
     ＋代表的合成の閉形式突合で検証した
     （`crates/autodiff/tests/create_graph.rs`）。
  8. `Op::supports_create_graph()` 機構 → 新設（上記）。
  9. 段階 1 実装 issue の起票 → 本 issue（#1942）自体がそれに該当。
- **機構（設計 doc §8「機構案」の実装）**: 祖先集合は `loss` から
  `Op::for_each_input` を辿る走査で求め、`requires_grad == false` の
  ノードでは descend しない（当該部分木はまるごと `child.var_no_grad`
  の定数葉へ変換し、Op 種別を問わず replay しない。`ResidentLeaf`／
  `LinearResident`／`LinearAct` は `requires_grad` の値に関わらず値の
  実体化を試みる前に無条件で `Err` とする）。子テープの葉プレフィックス
  契約（`Tape::reset` doc）を保つため、葉相当（`Op::Leaf`・
  `requires_grad == false` の任意ノード）をすべて先に登録してから、
  残り（`supports_create_graph() == true` の非葉ノード）を昇順で再生
  する 2 段構成とした。
- **checkpoint 併用**: `Tape::has_registered_checkpoints()` が親テープに
  登録済みの checkpoint 区間を検出すると fail-closed に拒否する
  （checkpoint 済みノードの forward 値解放・再計算経路との統合は §8
  「checkpoint 区間との相互作用」のとおり本 issue のスコープ外）。
- **`retain_graph`／`backward_accumulate` との併用**: 子テープは独立の
  `TapeId`／`epoch` を持ち、`Tape::reset`／drop まで保持され続ける
  （§9「世代契約」）ため `retain_graph`（#1749）と同型に複数回
  `child.backward(..)` を呼べる（`tests/create_graph.rs` の Hessian
  各テストが `x` の成分ごとに `child.backward` を反復呼び出しすること
  で検証済み）。`backward_accumulate` との組み合わせは対象外のまま
  （§9 の方針を維持）。
- **数値実測**: CPU（`common::naive_ops()`。ホスト参照実装）でのみ
  実装・検証した。CUDA／Metal は機構上同一経路（`Var` 演算の合成の
  みで新規カーネルを追加していない）だが実機実測は未実施のまま Mac／
  GB10 セッションへ申し送る。
- **facade／compat-api-scope への反映**: `docs/compat-api-scope.md`
  §1.3「高階微分」行・`docs/compat-feature-gap.md` §2.11 を本追記と
  同時に更新した（内部クレート限定の部分実装であることを明記）。

## 14. 実装記録（#1943・§8「対象」区分の `MatMul` 拡張・HVP 例）

親 #1940・前段 #1942 を受け、`Op::supports_create_graph()` の対象を
`MatMul`（**rank 2 × rank 2 限定**）へ拡張し、小型 MLP（`Linear`→
`tanh`→`Linear`）の HVP（ヘッセ・ベクトル積）を有限差分と突合する
統合テストを追加した。§13 の 11 variant からの累積で **12 variant**
が対象になる。

- **`MatMul` の VJP（`crates/autodiff/src/create_graph.rs::build_cgrads`
  の `Op::MatMul` 腕）**: `da = g.matmul_fp32_strict(&bᵀ)`・
  `db = aᵀ.matmul_fp32_strict(&g)`（1 階 `grad.rs::matmul_vjp` の
  rank 2 経路と同一のオペランド順序）を `Var::matmul_fp32_strict`
  （`ops().gemm_fp32_strict` 経由）／`transpose` の合成として子テープ
  へ記録する（当初案の `Var::matmul`〈`ops().gemm`〉から PR #2003
  codex-review 指摘を受けて切り替えた。§14a 参照）。
- **rank≥3 を対象外とした理由**: 1 階 `matmul_vjp` の rank≥3 経路は
  `reduce_batch_axes_f64`（`f64` アキュムレータの broadcast 縮約。
  `.claude/rules/coding-rust.md` の勾配長軸縮約契約）を経由するが、
  子テープ側の [`reduce_to`]（`Var::narrow`＋`Var::add` の `f32` 逐次和）
  はこれを逐語再現しない。PR #1998 の codex-review 指摘（§13 実装記録
  参照）と同型の数値乖離を生みうるため、rank 2 × rank 2 のみを対象
  とした（rank≥3 は `validate_ancestors` が型付き `Err` で事前拒否）。
- **入口検査の新設（`validate_ancestors`）**: `loss` から到達する祖先
  ノードを `Tape::backward_create_graph` の入口で `build_mirror`／
  `build_cgrads` より前に走査し、(a) resident／fused 経路
  （`ResidentLeaf`／`LinearResident`／`LinearAct`）・(b) 未対応 Op・
  (c) rank≥3 の `MatMul` をまとめて `Err(AutodiffError::Backward)` で
  拒否する。**素の `Tape::backward` より前に実行する**——resident
  グラフでは素の `backward` 自体が `AutodiffError::InvalidArgument`
  （`DeviceParamStore::backward` を使えという誤誘導的なメッセージ）を
  返してしまうため、順序を入れ替えて正確な型付きエラーを先に返す
  ようにした（§13 時点の実装では `self.backward(loss)?` が
  `collect_ancestors`／検査より先だったが、本 issue でこの順序自体を
  入れ替えた）。拒否時は `child` へ一切書き込まれない
  （`crates/autodiff/src/optim/device_store.rs::tests::
  create_graph_rejects_resident_path_with_typed_error` で resident
  経路の型付き拒否・`child.is_empty()`・拒否後も通常の
  `store.backward(&tape, &loss)` が成功することを検証済み）。
- **数値契約**: 子テープの `MatMul` VJP は `Var::matmul_fp32_strict`
  （= `ops().gemm_fp32_strict`）経由であり、1 階 `matmul_vjp`（同じく
  `ops.gemm_fp32_strict` 経由）と入口が揃っている。CPU バックエンドは
  両者が同一カーネルへ帰着するため bit 同一で、CUDA TF32 opt-in
  （`docs/cuda-tf32-optin-api-decision.md`）が有効な場合も 1 階
  `matmul_vjp` と同じく常に FP32 厳密のまま計算されるため
  `first_order()` との bit 一致が崩れない（§14a）。本モジュールは
  元々「子テープの数値方式は一般に bit 同一を主張せず、正しさは
  有限差分突合で検証する」立場（`create_graph.rs` モジュール doc）の
  ためこの整理はその範囲内に収まる。tolerance／baseline は無変更。
- **テスト**: `crates/autodiff/tests/create_graph.rs` に以下を追加（31
  件が全 green。CPU `common::naive_ops()` でのみ検証）。
  - `hessian_matmul_quadratic_w_matches_finite_difference_and_closed_form`／
    `hessian_matmul_quadratic_x_matches_finite_difference`: matmul の
    二次形式で `da`／`db` 両腕を網羅し、閉形式（`2・XᵀX ⊗ I`）とも突合。
  - `hessian_linear_with_bias_w_matches_finite_difference`／
    `_b_matches_finite_difference`: `MatMul`→`Add`（bias パターン）の
    合成（`nn::Linear` 既定 forward 経路と同型）。
  - `hessian_via_nn_linear_forward_matches_manual_composition`:
    `nn::Linear::from_parameters(..).bind(&tape).forward(..)` 経由の
    2 階勾配が手動 `matmul`＋`add` 合成と数値的に一致することを確認。
  - `hvp_small_mlp_matches_finite_difference`: 2 層 MLP（`Linear`→
    `tanh`→`Linear`。二乗誤差 loss は未対応 `Var::sub` を避け
    `add`＋自乗＋`mean` で構成）の HVP（`Σ_p grad_p・v_p` を子テープ上で
    合成し再度 `backward`）を、`create_graph` を経由しない独立な方向
    微分の中央差分と突合。判定は既存 `common::req2_close`（REQ-2
    統一複合判定）のみを使用し、tolerance は変更していない。
  - `create_graph_rejects_unsupported_op_sub`（旧
    `create_graph_rejects_unsupported_op_matmul` を置き換え）・
    `create_graph_rejects_fused_linear_act`・
    `create_graph_rejects_rank3_matmul`: いずれも `child.is_empty()`
    まで確認する fail-closed 契約テスト。
- **残る対象外事項**: `ScalarUnary`／`ScalarBinary`（`Var::sub` 等）・
  `Transpose`／`Permute`／`Narrow`／`Concat`／`Contiguous`／`Where`／
  `MaskedFill`／`Gather`／`Scatter`／`Pad`／`MseLoss`／
  `CrossEntropyLoss`（§8「対象」区分の残り）は引き続き未実装のまま
  `Op::supports_create_graph() == false`。facade 公開（§10 承認事項 5）
  も引き続き未承認のまま不実施。checkpoint 併用・rank≥3 matmul の
  対応は後続イシューへ引き継ぐ。
- **CUDA／Metal 実機実測**: CPU（`naive_ops()`）でのみ検証済み。新規
  カーネルは追加していない（既存 `Var::matmul_fp32_strict`／
  `transpose` の合成のみ）が、実機実測は未実施のまま Mac／GB10
  セッションへ申し送る。
- **facade／compat-api-scope への反映**: `docs/compat-api-scope.md`
  §1.3「高階微分」行・`docs/compat-feature-gap.md` §2.11 を本追記と
  同時に更新した。

## 14a. 是正記録（PR #2003 codex-review 指摘・イシュー #1943）

§14 時点の実装は 2 点の精度契約上の問題を含んでいた。いずれも
codex-review（PR #2003）の指摘を受けて是正した。

- **問題 1（P1・`build_cgrads` の `Op::MatMul` 腕）**: 当初 `da =
  g.matmul(&bᵀ)`・`db = aᵀ.matmul(&g)`（`Var::matmul` = `ops().gemm`）
  として子テープへ記録していたため、CUDA TF32 opt-in
  （`set_cuda_gemm_precision`）が有効な間、1 階 `grad.rs::matmul_vjp`
  （`ops.gemm_fp32_strict` 経由で常に FP32 厳密）が守る
  「バックプロパゲーションは常に FP32 厳密」という契約が、二階微分の
  記録経路でだけ TF32 相当まで精度低下していた。`Var::matmul_fp32_strict`
  （`crates/autodiff/src/var.rs`。`ops().gemm_fp32_strict` を forward
  値の計算に使う以外は `Var::matmul` と同一の `pub(crate)` メソッド）
  を新設し、`build_cgrads` の `Op::MatMul` 腕をこちらへ切り替えた。
- **問題 2（P1・checkpoint との相互作用）**: `Var::matmul_fp32_strict`
  も記録するノードは通常版と同じ `Op::MatMul(a, b)` であり、`Op::
  MatMul` は `is_checkpoint_eligible() == true`（`docs/
  autodiff-checkpoint-design.md`）。区別する情報が `Op` 側にないため、
  子テープ上で `g`（`matmul_fp32_strict` の結果）から `h = g.mul(&g)?.
  sum(None)?` を作り `h.checkpoint_from(&[])` を呼ぶと `g` が解放対象
  になり、後続 `child.backward(&h)` の再計算（`tape.rs::
  recompute_value` の `Op::MatMul` 分岐）が非厳密な `matmul_forward`
  （`ops.gemm`）を使ってしまい、問題 1 と同じ精度低下が checkpoint
  経由で再発する経路があった。`TapeNode`（`tape.rs`）へ `fp32_strict:
  bool` フィールド（既定 `false`）を追加し、`Var::matmul_fp32_strict`
  が `push_eager` 直後に戻り値ノードへ限定して `true` を立てる。
  `release_checkpoint_region`（`Tape::register_checkpoint`／
  `Tape::release_checkpoints_ending_at` 共通の解放ロジック。唯一の
  `is_checkpoint_eligible()` 呼び出し箇所）の判定を `node.op.
  is_checkpoint_eligible() && !node.fp32_strict` へ変更し、フラグが
  立ったノードは checkpoint 区間に含まれても解放されない（`value`
  は常に厳密精度のまま保持され、`recompute_fallible`／`recompute_
  infallible` がこのノードを再計算する経路へは一切到達しない）よう
  にした。`Op::MatMul` の通常版（`Var::matmul`）・他の全 Op variant の
  checkpoint 適格性は無変更。
- **影響範囲**: `TapeNode` を直接構築する全箇所（`tape.rs` の
  `push_eager`／`push_leaf`／`push_resident_leaf`／`push_view`／
  `push_lazy`、`grad.rs` のテストフィクスチャ 3 箇所）へ
  `fp32_strict: false` の初期化を追加。フィールド追加自体が構造体
  リテラルの網羅性によりコンパイルエラーで検出されるため、更新漏れ
  はビルドで機械的に防がれる。
- **CUDA／Metal 実機実測**: 問題 1・2 とも数値契約の是正であり新規
  カーネルは追加していない。実機実測は未実施のまま Mac／GB10
  セッションへ申し送る（§14 の既存申し送りと同一）。

## 15. facade 公開の保留記録（イシュー #2063）

イシュー #2063「facade: 高階微分 API の公開面追加」の実装着手時
（2026-09-19・main HEAD `adcf468a`）に、`gh issue view 2063 --comments`・
`gh issue view 2059 --comments`（親 issue）を確認したところ、§10
承認事項 5「facade 公開面への高階 API 追加」を対象とするリポジトリ
所有者の明示的な承認コメントは存在しなかった（#2063 のコメントは
2026-09-19T13:20Z の「実装保留（ユーザー承認待ち）」通知 1 件のみ、
親 #2059 はコメント 0 件）。issue が起票されていること自体は
承認事項 5 の承認にはならない（前例: #1955／#1758／#2017 はいずれも
明示的な承認コメントが先行してから facade 実装に着手している）。

自動運転（承認待ち不可）かつ判断は安全側に倒す方針、`docs/
compat-api-scope.md` §5（範囲拡張は経路 1／2 の承認必須）、`.claude/
rules/security.md`（自己修復による無断拡大禁止）に基づき、本 issue
では `crates/facade/src/**`・`crates/autodiff/**` を一切変更せず、
`crates/facade/tests/api_surface.rs` に否定ガード 2 件
（`facade_does_not_reexport_create_graph_result`・`facade_tape_does_
not_expose_backward_create_graph_method`）を追加して「facade 未公開」
状態を機械固定した（`facade_does_not_reexport_custom_function`・
`facade_tape_does_not_expose_custom_forwarding_method`〈§10 承認事項
5 と同種の未承認公開面〉と同型の走査）。

承認取得後に実施する変更範囲（事前提示。#2063 コメント本文と同一）:

- `crates/facade/src/lib.rs`: `pub use fandhe_ai_autodiff::
  CreateGraphResult;`（1 行の再エクスポート）・`impl Tape` への
  `backward_create_graph` 委譲メソッド 1 件（`&self.0.
  backward_create_graph(loss, &child.0)` を返すだけの薄い委譲。
  `transfer`／`rnn_forward_seq` と同型）
- `crates/autodiff/src/create_graph.rs`: モジュール doc の
  「facade 非公開」記述を「facade 公開済み」へ更新（ロジック変更なし）
- `crates/facade/tests/api_surface.rs`: 上記否定ガード 2 件を削除し、
  ソース走査による正ガード・facade 経由の到達性テストへ差し替え
- `docs/compat-api-scope.md`・`docs/compat-feature-gap.md`: 実装記録・
  適用記録の追記

イシューは close せず、承認取得後に別 PR で経路 B（公開実施）を行う。

（#2545／#2546 追記）本節の保留は #2545 で解除済み。事前提示した変更範囲との対応は §19。

## 16. 実装記録（イシュー #2062・§8「対象」区分の残り Op 拡張）

`Op::supports_create_graph()`（`tape.rs`）が判定する対象を、#1942・
#1943 で実装済みの 12 variant から以下へ拡張した。数値方式・doc の
出典は `crates/autodiff/src/create_graph.rs`（`scalar_unary_
replayable`／`scalar_binary_replayable`・`replay_op`／`build_cgrads`
の各腕コメント）を正とし、本節では要点のみ記す。

- **`Transpose`・`Permute`・`Narrow`・`Concat`・`Contiguous`・`Where`**:
  無条件で対象。`replay_op` は既存 `Var` の公開メソッド
  （`transpose`／`permute`／`narrow`／`Var::cat`／`contiguous`／
  `Var::where_cond`）へ薄く委譲する。`build_cgrads` は 1 階 VJP
  （`grad.rs`）と同型の構成（対合性・Split⟷Concat 双対性・恒等・
  `grad.rs::inverse_permutation` の再利用〈`pub(crate)` 化〉）。
- **`ScalarBinary`（既知 13 variant）**: `scalar_binary_replayable` が
  すべて `true` を返す。`replay_op` は `Var::scalar_binary`
  （`pub(crate)`。同一 crate のため呼べる）への委譲で variant 分岐が
  不要。`build_cgrads` は 1 階 VJP（`eval::scalar::binary_partials`）の
  式をそのまま `Var` 演算へ写す（`Sub`／`Mul`／`Div`／`Pow`〈`a==0`／
  `b==0` ガードは host マスク〉／`Maximum`／`Minimum`〈勝ち／タイ／
  `NaN` の 3 分類 host マスク〉／比較 6 種〈直接ゼロ定数。`0 * upstream`
  を経由せず `inf`／`NaN` 汚染を避ける〉）。broadcast 縮約は
  `reduce_bias_grad_var`（1 階の `reduce_bias_grad` と同一の bias
  パターン判定）を使う。
- **`ScalarUnary`（`Gelu`・`GeluTanh` を除く既知 variant）**:
  `scalar_unary_replayable` が `Neg`・`Abs`・`Sqrt`・`Log`／`Log2`／
  `Log10`・`Sin`／`Cos`／`Tan`・`Relu`／`Exp`／`Tanh`／`Sigmoid`
  （既存）・`Silu`・`Hardswish`・`LeakyRelu`・`Elu`・`Softplus`・
  `Clamp`・`PowScalar` を `true` にする。滑らかな variant は写し
  （`x_m`／`y_m`）から `Var` 演算で係数を合成し（`eval::scalar::
  unary_grad_factor` の式の順序を踏襲）、区分定数な variant（`Abs`・
  `LeakyRelu`・`Elu`・`Softplus`・`Clamp`・`Hardswish`）は host マスク
  （`mask_from_pred`）＋`Var::where_cond` で選択する（`vjp_elementwise_
  mul` のゲート付き乗算と異なり、素の `g.mul(&0/1 定数)` は `inf * 0 =
  NaN` を生みうるため使わない）。`Gelu`（誤差関数版）は導関数が `erf`
  を要し `Var` 演算の合成だけでは再現できないため対象外のまま
  （`.claude/rules/deps-policy.md` により `erf` crate 依存は追加不可）。
  `GeluTanh` は式の複雑さから本イシューでは見送り、`false` のまま残す
  （後続イシューへ引き継ぐ）。`PowScalar` は `Var` 側に公開 API がなく
  （スカラー指数版は CUDA／Metal カーネル未実装のため `Var::pow` は
  Var×Var 限定）現状は到達不能だが、`Op::ScalarUnary { op:
  PowScalar { .. }, .. }` が将来公開されたときに備え実装だけ済ませて
  ある。
- **`Op::Pad`／`Op::MaskedFill`／`Op::Gather`／`Op::Scatter`／
  `Op::MseLoss`／`Op::CrossEntropyLoss`**: 本イシューのスコープ外の
  まま残す（`supports_create_graph` は `false`）。`Pad`／`MaskedFill`
  は置換定数 `value` を `Op` payload が保持しないため、子テープでの
  forward 再生に `value: f32` フィールド追加が必要（設計は Plan 段階
  で確定済みだが本イシューの実装時間の都合で見送った）。`Gather`／
  `Scatter` は索引付き scatter_add／scatter の子テープ再生、
  `MseLoss`／`CrossEntropyLoss` は損失関数の閉形式 VJP 合成が必要で、
  いずれも後続イシューへ引き継ぐ（`.claude/rules/
  out-of-scope-tracking.md`）。
- **既知の不整合（スコープ外・記録のみ）**: 子テープの `Op::
  BroadcastTo` 腕は `reduce_to`（f32 逐次和）のままだが、1 階
  `grad.rs::Op::BroadcastTo` は codex-review P1 是正で `reduce_bias_
  grad` へ変更済み（2026-09-12）。bias パターン（`[m,n]` → `[1,n]`
  の明示 broadcast）で子テープと 1 階が数値的に乖離しうる。本イシュー
  では触れず、別イシュー化を PR 側で提案する。
- **CUDA／Metal 実機実測**: 新規カーネルは追加していない（既存
  `Var` 演算の合成のみ）ため機構上 REQ-2 非後退。実機実測は Mac／GB10
  セッションで `cargo test -p fandhe-ai-autodiff --test create_graph`
  を実バックエンド `ops` で再実行する手順として
  `docs/perf/logs/create-graph-remaining-ops-2062/README.md` へ申し
  送る。
- **有限差分突合テストの件数（中断復旧・#2062 継続実装）**:
  `crates/autodiff/tests/create_graph.rs` は 51 件（`Hardswish` の
  interior／飽和境界の 2 件を追加）。`Hardswish` の 2 階マスク境界
  （`mask_from_pred` の `<=`／`>=`）を `tensor-core::scalar_op::
  hardswish_grad` の飽和域境界規約（`x <= -3.0`／`x >= 3.0` で定数
  勾配・曲率 0）へ一致させる回帰テストを追加した（境界ちょうどは
  1 階導関数自体が不連続な kink 点のため有限差分突合の対象にできず、
  解析 Hessian が飽和域〈曲率 0〉を使うことを直接検査する形とした）。
  `Maximum`／`Minimum`（勝ち／タイ／`NaN` の 3 分類）は `Var::
  scalar_binary` が `pub(crate)` のため統合テスト（別クレート扱い）
  から呼べず、`Op::Contiguous` と同じ理由で `crates/autodiff/src/
  create_graph.rs` 内部の `#[cfg(test)] mod maximum_minimum_tests` へ
  3 件追加した（勝ち／負け／タイの対角曲率突合 2 件・`NaN` 要素の
  1 階勾配ゼロと子テープ再構成が `Tape::backward` 経路と bit 一致
  することの突合 1 件）。`Var` に `maximum`／`minimum` の公開ラッパー
  は存在しない（`ScalarBinaryOp::Maximum`／`Minimum` を直接公開する
  メソッドが `var.rs` に未実装のまま）ため、facade はもちろん通常の
  `Var` API からも到達不能——このギャップの解消は本イシューのスコープ
  外として別イシューへ引き継ぐ。

内部ホスト名・秘密情報は含めない。


## 17. facade 公開形の確定記録（イシュー #2544）

基準コミット: `origin/main` `89bb480a`。ルート #2499 → Phase 3 #2542 → #2543
（高階微分の公開）の第 1 段として、§5・§10・§15 に**書かれた形**を確定形として
読み出し、`fandhe-ai =0.10.0` に対する非破壊性を論点別に確認した記録。
本節は既存節（§5／§10／§15）を書き換えず、承認範囲を超える新しい形は導入しない
（ルート #2499 の一括承認〈2026-10-04〉は決定記録に書かれた形にのみ及ぶものとして読む）。

### 17.1 論点の判定（推奨形が一意に記載済みか）

| 論点 | 記録上の根拠 | 判定 |
|------|-------------|------|
| 方式 | §5 主案 A-2（子テープ方式） | 確定 |
| `ops` 供給 | §7 (i)（呼び出し側が空の子 `Tape` を渡す）。§10 承認事項 2 の推奨と同一 | 確定 |
| facade の型 | §9「facade」: newtype `Tape` が子テープもラップ・`new_with_ops` は非公開維持 | 確定 |
| 公開シグネチャ・配置 | §15「承認取得後に実施する変更範囲」: crate ルートの再エクスポート 1 行＋`impl Tape` への薄い委譲 1 件 | 確定 |
| エラー型 | §9 の比較のうち既存 `AutodiffError::Backward(String)` 流用（専用 variant 新設なし）。§15 の変更範囲にも variant 追加がない | 確定 |
| 1 階勾配 bit 同一性 | §9 選択肢 (i)（2 パス方式。1 階は無変更の `backward_impl`） | 確定 |
| `tape.child()` 等の補助ヘルパー | §7 (i) の評価欄に言及があるのみで §15 の変更範囲に含まれない | 含めない（17.3） |

未決・複数案のまま残る論点はないため、承認依頼のための停止は行わない。

### 17.2 確定形

- **再エクスポート**: `crates/facade/src/lib.rs` の crate ルートに `pub use fandhe_ai_autodiff::CreateGraphResult;`
- **委譲メソッド**: facade `impl Tape` に
  `pub fn backward_create_graph<'c>(&self, loss: &Var<'_>, child: &'c Tape) -> Result<CreateGraphResult<'c>, AutodiffError>`
  を追加し、本体は `self.0.backward_create_graph(loss, &child.0)` のみとする
  （`transfer`／`rnn_forward_seq` と同型の薄い委譲。§15 本文の先頭 `&` は表記揺れであり、戻り値は所有値の `Result`）。
  `child` は facade の newtype `Tape`。子テープは既存の `fandhe_ai::tape()`／`tape_for(parent.device())` で構築する
- **エラー型**: 既存 `AutodiffError`（`#[non_exhaustive]`・0.10.0 で再エクスポート済み）の既存 variant のみ。
  `Backward(String)`（未対応 Op・同一テープ指定・非空の子テープ・checkpoint 併用・resident／fused 経路・rank≥3 matmul 等）、
  `TapeMismatch`、`DeviceMismatch`、`GradientTrackingDisabled` ほか内部から伝播しうる既存 variant。新規 variant は追加しない
- **`CreateGraphResult` の公開メソッド**: `first_order() -> &Gradients`、
  `grad(&Var) -> Result<Option<Var<'c>>, AutodiffError>`、`child_var(&Var) -> Result<Option<Var<'c>>, AutodiffError>` を無変更で公開
  （`crates/autodiff/src/create_graph.rs`）。フィールドは非公開のため外部構築不可。
  シグネチャに現れる `Gradients`・`Var`・`AutodiffError` はいずれも 0.10.0 で facade 再エクスポート済みで、
  `NodeId`／`TapeId`／`BackendOps` は露出しない
- **モジュール配置**: 新規モジュールなし（crate ルート＋既存 `Tape` 型上のメソッド）

### 17.3 本公開形に含めないもの

- `tape.child()` 等の補助ヘルパー（§15 の変更範囲に無く、一括承認の範囲を広げない）。#2545 で追加しない
- `TapeRef::backward_create_graph`、`new_with_ops` への到達経路（REQ-12 の非公開維持）
- HVP 専用 API（案 C）、`backward_accumulate` 併用、checkpoint 併用、残る非対象 Op の拡張
- いずれも必要になれば別途「決定記録の作成 → ユーザー承認」の 2 段を経る

### 17.4 `fandhe-ai =0.10.0` 非破壊確認（`v0.10.0` タグ対 `89bb480a`）

| 論点 | 確認結果 |
|------|---------|
| crate ルート `CreateGraphResult` の再エクスポート | 0.10.0 の facade `lib.rs` に `CreateGraph`／`backward_create_graph` の出現が 0 件。追加のみ（minor）。利用者側 glob import との同名衝突は Cargo SemVer ガイド上 minor 扱いで、既存の再エクスポート追加と同じ受容 |
| `Tape::backward_create_graph` | 0.10.0 `Tape` に同名メソッドなし（`api_surface.rs` の否定ガード 2 件〈`facade_does_not_reexport_create_graph_result`・`facade_tape_does_not_expose_backward_create_graph_method`〉が同タグに存在＝未公開の証跡）。inherent メソッド追加のみで既存メソッドの意味論は不変 |
| エラー型 | `AutodiffError` は 0.10.0 で `#[non_exhaustive]` かつ再エクスポート済み。使用 variant（`Backward`・`TapeMismatch`・`DeviceMismatch`・`GradientTrackingDisabled`）はすべて 0.10.0 に存在し、variant 追加なし |
| 戻り値型に現れる型 | `Gradients`／`Var`／`AutodiffError` は 0.10.0 で公開済み。新たな内部型の露出なし |
| 子テープ構築 | 既存 `tape()`／`tape_for(Device)`／`Tape::device()` が 0.10.0 に存在し、新規構築子は不要。`new_with_ops` 非公開は不変 |
| 内部シグネチャ | `backward_create_graph` と `CreateGraphResult` の公開メソッド（`first_order`／`grad`／`child_var`）は `v0.10.0` と `89bb480a` で差分なし |
| 既存 1 階 backward | `backward_create_graph` は既存 `Tape::backward` を無変更で呼ぶのみ（§13）で、`Tape::backward` の意味論は不変 |

結論: 追加のみ（minor 相当）で 0.10.0 に対する破壊的変更はない。tolerance／baseline／ガードレール閾値／依存は変更しない。

### 17.5 後続イシューへの申し送り

- #2545: 17.2 の確定形のみを実装する（`create_graph.rs` のモジュール doc 更新を含む）。17.3 の項目は追加しない
- #2546: 否定ガード 2 件の正ガード化、`docs/compat-api-scope.md` §5 適用記録、`docs/compat-feature-gap.md` の更新
- #2063（先行 issue）のクローズ判断は本節の範囲外

## 18. 実装記録（#2545）

- §17.2 の確定形のみを実装した: crate ルート `pub use fandhe_ai_autodiff::CreateGraphResult;`（1 行）と facade `Tape::backward_create_graph`（`self.0.backward_create_graph(loss, &child.0)` の薄い委譲 1 件）。autodiff 側のロジック変更なし（`create_graph.rs` のモジュール doc のみ更新）。§17.3 の項目は追加していない
- **ガード反転の前倒し**: §17.5 は否定ガード 2 件の正ガード化を #2546 に割り当てていたが、公開と同時にそれらが失敗し `cargo test --workspace` が green にならないため、#2545 で `tests/api_surface.rs` を正ガード 4 件（再エクスポート形状・薄い委譲本体・宣言インベントリ・シグネチャ到達性）へ置き換えた（#2727・#2729〜#2731 と同じ運用）
- **#2546 の残りスコープ**: `docs/compat-api-scope.md` §5 適用記録・`docs/compat-feature-gap.md` §2.11 の更新（必要なら追加のガード強化）。**#2546 で実施済み（§19）**
- 公開経路テストは `crates/facade/tests/create_graph_facade.rs`（CPU 9 件・GPU `#[ignore]`）と `Tape::backward_create_graph` の doctest。GPU 実機手順は `docs/perf/logs/create-graph-facade-2545/README.md`

## 19. facade 公開の実装記録（#2545／#2546）

§17.2 の確定形は PR #2735（#2545。`0e5737f1`）で公開済みである。承認の範囲はルート #2499 本文の一括承認（2026-10-04）に基づく §17 のとおりで、本節はそれを超える承認を主張しない。本節は #2546 による記録であり、公開面は変更しない。

### 19.1 公開した名前

- crate ルート `pub use fandhe_ai_autodiff::CreateGraphResult;`（1 件）
- facade `Tape::backward_create_graph<'c>(&self, loss: &Var<'_>, child: &'c Tape) -> Result<CreateGraphResult<'c>, AutodiffError>`（本体は `self.0.backward_create_graph(loss, &child.0)` の 1 行委譲）
- `CreateGraphResult` の公開メソッド `first_order`／`grad`／`child_var`（無変更）

### 19.2 ガード反転の対応表

| #2063 の否定ガード | 現行の正ガード（`crates/facade/tests/api_surface.rs`） |
|---|---|
| `facade_does_not_reexport_create_graph_result` | `facade_reexports_create_graph_result_in_approved_shape`（`lib.rs` の 1 行再エクスポートのみ許可） |
| `facade_tape_does_not_expose_backward_create_graph_method` | `facade_tape_backward_create_graph_is_thin_delegation`（宣言の存在＋本体 `self.0.backward_create_graph(loss, &child.0)` の固定） |
| （新設） | `workspace_declares_backward_create_graph_only_in_approved_locations`（workspace 内の `fn backward_create_graph` は autodiff 本体 1＋facade 委譲 1 のみ） |
| （新設） | `tape_backward_create_graph_is_reachable_via_facade`（公開シグネチャ固定＋CPU 最小呼び出し） |

反転の実施は #2735（#2545 での前倒し。§18）で、#2546 では判定ロジック・メッセージとも変更していない。追加のガード強化は不要と判断した。理由は次のとおり。

- 宣言インベントリが facade 側の `backward_create_graph` を 1 件に固定しており、同ファイル内で `TapeRef` 版などを追加すると件数 2 で fail-closed に検出される
- create_graph 用の `*HoldDoctestGuard`（部分反転が必要な doctest プローブ）は存在しない
- 4 正ガードの失敗メッセージは既に §17.2 を根拠としている

### 19.3 テスト・利用例

- `Tape::backward_create_graph` の rustdoc doctest
- `crates/facade/tests/create_graph_facade.rs`（#2545 の CPU 9 件・GPU `#[ignore]` 群）
- #2546 で追加した 2 件（CPU のみ・tolerance は新設せず既存 `req2_close` を使用）:
  - `child_built_with_tape_for_parent_device_works`（§17.2 の子テープ構築 `tape_for(parent.device())`・ミラー取得・2 階微分の閉形式一致）
  - `rejects_parent_with_registered_checkpoint_via_facade`（facade 到達可能な `Var::checkpoint_from` で登録した親を `Err(AutodiffError::Backward(_))` で拒否し、子を変更しない）

### 19.4 保留継続（承認範囲外）

§17.3 の列挙をそのまま維持する。

- `tape.child()` 等の子テープ構築ヘルパー
- `TapeRef::backward_create_graph`、`new_with_ops` への到達経路
- HVP 専用 API（案 C）、`backward_accumulate` 併用、checkpoint 併用、残る非対象 Op の拡張

### 19.5 数値契約・実機・互換性

- tolerance／baseline／ガードレール閾値は不変。数値判定は REQ-2 統一複合判定のまま
- GPU 実機手順は `docs/perf/logs/create-graph-facade-2545/README.md` の申し送りが有効（#2546 の追加テストは CPU のみ）
- `Cargo.toml`／`Cargo.lock`・`docs/spec/` は不変。`fandhe-ai =0.10.0` に対する非破壊の結論は §17.4 を参照
- 適用記録は `docs/compat-api-scope.md` §5、追記は `docs/compat-feature-gap.md`

### 19.6 §15 事前提示の変更範囲との対応

| §15 の事前提示 | 実施 |
|---|---|
| `crates/facade/src/lib.rs` の再エクスポート＋委譲メソッド | #2735（#2545） |
| `crates/autodiff/src/create_graph.rs` のモジュール doc 更新 | #2735（#2545） |
| `api_surface.rs` の否定ガード 2 件の差し替え | #2735（#2545。§19.2） |
| `docs/compat-api-scope.md`・`docs/compat-feature-gap.md` の記録 | #2546（本節） |

## 20. 2026-10-11 所有者判断の反映（#3020）

- §16 が「スコープ外のまま残す」とした Op 群（`Pad`・`MaskedFill`・`Gather`・`Scatter`・`MseLoss`・`CrossEntropyLoss` ほか）と §19.4 の保留継続は、D-A により対象 Op 拡張として実装へ進む（#3075）。
- Gelu（erf）は依存追加不可の制約が従来どおり残る。
- 基づく判断: 2026-10-11 の所有者判断 D-A（推奨形で一括承認。#3018 ツリー起票の根拠。`compat-api-scope.md` §5 末尾の適用記録を参照）。本節は記録の反映であり新規承認ではない。
- 各 issue で本記録に公開形を書き、その形のまま実装まで進む。記録に形が無い点だけ承認依頼へ戻す。
- 依存・`unsafe`・tolerance・baseline・`docs/spec` は変更していない。既存本文は書き換えていない。

## 21. create_graph の対象 Op 拡張の範囲と族ごとの二階 VJP 方針（#3076・親 #3075）

- 基準コミットは `origin/main` `27e0472d`。本節の `file:line` はこのコミットで確かめた（`tensor-core/src/scalar_op.rs` 以外は `crates/autodiff/src/` 配下）。
- 本節は設計判断だけを書く。コード・依存・`unsafe`・tolerance・baseline・`docs/spec`・既存節の本文は変えていない。根拠は §20（2026-10-11 の D-A）で、新しい承認ではない。

### 21.1 共通方式

**(1) 不変条件**
- 1 階の `Tape::backward` と `CreateGraphResult::first_order()` は親テープで加工せずに計算する。bit 同一は構造上保たれる（§3 契約 1・§9 選択肢 (i)）。
- 子テープ上の `cg.grad(..)` は 1 階勾配と REQ-2 統一複合判定で突き合わせる（§9 (i)）。bit 同一は要求しない。V3 を採る族だけは値が bit 同一になる。
- 新しい tolerance・baseline は作らない。

**(2) 再生 R**
- `replay_op` は公開 `Var` メソッドへ薄く委譲する（#2062 と同形）。forward と同じカーネルを通るので写しの値は親と一致する。
- 再生に必要で payload に無い値は `Op` へ足す。`Op` は `pub(crate)` なので公開 API には影響しない。

**(3) 二階 VJP の構成方式**

| 方式 | 内容 | 前例 |
|---|---|---|
| V1 | 1 階 VJP の式を replayable な `Var` 演算の合成で子テープへ書く（f32） | #2062 の `Silu` など |
| V2 | 選択・区分定数の部分は親の記録値からホストでマスク・索引を作り、`where_cond`／`gather`／`scatter` で選ぶ。マスク規則は 1 階と同じなので先勝ちタイが保たれる。`g.mul(0/1 定数)` は `inf*0=NaN` を生むので使わない | `mask_from_pred`（`create_graph.rs:1353`）、`Maximum`／`Minimum` の 3 分類 |
| V3 | 値は 1 階 VJP と同じホスト関数（f64 縮約契約を持つもの）で計算し、構造は V1 の合成グラフを入力に、VJP が前方値に依存しない Op タグ（`Contiguous`〈恒等〉・`Sum`）で `Tape::push_eager` する。値は 1 階と bit 同一、二階の微分は合成グラフを流れる。ホスト関数は `pub(crate)` にする（#2062 の `inverse_permutation` と同様）。孫テープで再生すると合成側の f32 が再計算されるが、この差は REQ-2 の範囲で受け入れる | `reduce_bias_grad_var`（`create_graph.rs:1310`） |
| V4 | 1 階 VJP そのものを専用 Op にし、その VJP を手で書く。§4／§5 が案 B を一般方針として退けているので既定では採らない。採る族は個別に理由を書く | なし |

**(4) 閉包性**
- `jvp_create_graph`（`functional_ops.rs:742`、`:799` の `child.backward_create_graph(&prod, grandchild)`）は子テープをさらに孫テープへ再生する。#3087 は jacfwd が拡張 Op を通ることを確かめる。
- したがって `build_cgrads` が子テープへ積むノードは、すべて `supports_create_graph == true` でなければならない。族の節に「積む Op」を書き、他の族・fail-closed に依存する場合は 21.4 の依存順に載せる。

**(5) 数値契約**
- f64 縮約は原則 V3。overflow 回避は 1 階の回帰ベクトル（§8 の `y=[0.25,0.75]`・`g=[3e38,-3e38]` など）を族のテストで必須にする。先勝ちタイは V2。
- **中間値の overflow（V3 の値補正では防げない）**: V3 はノードの最終値だけをホスト関数に置き換えるので、構造側の合成グラフが f32 で持つ中間値（例: `x=1e20` の二乗が `inf`、そこから `rstd=0`）は二階微分・孫テープ再生にそのまま使われる。したがって f64 縮約契約を持つ族は、V3 の値補正に加えて次のいずれかで中間統計量を安全に作ることを族の節に明記する。(S1) スケール付き合成: 行ごとの最大絶対値 `s`（ホスト定数。全ゼロ行・全要素が 0 の行では `s=0` となり `x/s` が `NaN` になるため、`s=0` の行は正のスケール `s=1` を使う規則とする。`x` が全ゼロなら二乗和は 0 で、正の `eps` を持つ `LayerNorm`・`RmsNorm` の統計量・入力勾配は 1 階と同じく有限になる）で `x/s` を作ってから二乗和・rstd を合成し、`rstd = rstd_scaled / s` で戻す。孫テープ再生でも同じスケール付き構造が再計算される（ただし upstream・weight・`rstd` の組み合わせで構造の中間値が f32 を超えうるので、構造の各段が f32 に収まることは族ごとの総括的な適用判定で確かめ、overflow しないことの保証はその判定を通った入力に限る。#3078 は §27.3）。**eps の値域保護（S1 の適用条件）**: スケール後の統計量に載せる `eps` は `eps/s²` に変換する。これを f64 で計算して f32 へ丸めた値が 0 になる（`s` が大きく `eps/s²` が f32 正規最小値 `1.17549435e-38` 未満）場合でも、`s` が「中心化後の偏差の最大絶対値」（`LayerNorm`。`RmsNorm` は `x` の最大絶対値）である限り、スケール後の分散（平均二乗）は少なくとも `1/n`（最大要素の寄与が `1`）なので `eps/s²` は丸めで消えるだけで `rstd_scaled` は変わらず有限である。`LayerNorm` で `s` を中心化前の `x` の最大絶対値から選んではならない（定数行 `x=[1e30,1e30]`・`eps=1e-5` は分散 0 のまま `eps/s²=1e-65` が f32 で 0 になり `rstd_scaled=inf` になる）。`s` は構造側と同一手順の f32 演算で得た中心化後偏差から算出し、全偏差が 0 の行は `s=1`（`eps` を変換せず `rstd=1/√eps` で有限）とする。**eps 側の overflow 保護（小スケール入力）**: `s` が極小だと `eps/s²` が f32 で overflow する（`RmsNorm` の `x=[1e-30,1e-30]`・`eps=1e-5` は `s=1e-30` で `eps/s²≈1e55` が `inf` になり、`rstd_scaled=0`・復元した `rstd=0` と誤る。既存 `row_rms_stats` は約 316.23。`0` は有限なので S2 への切替条件にも当たらない）。このため `s` は「最大絶対値（`LayerNorm` は中心化後偏差の最大絶対値）」と `√eps` の大きい方を 2 のべき乗へ切り上げた値とする（`eps>0` のとき `s≥√eps` なので `eps/s²≤1` で常に有限。`s` が最大絶対値で決まる場合は上記の議論どおり。`√eps` で決まる場合は `eps/s²∈(1/4,1]` かつ `x/s` の絶対値が 1 以下で、`rstd_scaled≤2` となり有限）。`eps=0` の行は従来どおり最大絶対値のみで `s` を決め、`s=0` なら `s=1` とする。この規則は全ゼロ偏差行も包含する（`s=√eps` で `rstd=1/√eps`）。`rstd_scaled` が非有限になる（スケール後分散と `eps/s²` の和が 0）入力は S1 の適用外として S2 へ切り替える（既存 `row_ln_stats` は `eps` を f64 で保持するので S2 の値側と整合する）。(S2) 中間統計量（`rstd` 等）を 1 階と同じ f64 先昇格のホスト関数で計算し、`push_eager` した定数ノードとして保持した上で、二階の微分に必要な項は解析式で別途合成する。(S3) V4（専用 VJP）。既定は S1。いずれの場合も、1 階・`cg.grad`・孫テープ再生（`jvp_create_graph` 経由）の 3 段すべてで有限入力の overflow 回帰ベクトルを必須テストにする（1 階だけの確認では不足）。

### 21.2 `supports_create_graph` の Op 割り当て

`tape.rs:2432` の網羅 match は `true` が 18 variant、payload 判定の `ScalarUnary`・`ScalarBinary` が 2、`false` が 103 で、enum 全体 123 と一致する（18 + 2 + 103）。103 件は次で機械的に抽出した（`ScalarBinary` を除く）。

```bash
sed -n 2453,2570p crates/autodiff/src/tape.rs | grep -oE "Op::[A-Za-z0-9]+" | sed 's/Op:://' | sort -u
```

**A. 恒久対象外（5）**

| Op | 根拠 |
|---|---|
| `ResidentLeaf`・`LinearResident`・`LinearAct` | §3 契約 6（resident／fused はテープ外の値を持つ） |
| `Custom` | §8・`autodiff-custom-function-decision.md` §14（利用者関数の二階は定義できない） |
| `OneHot` | 非微分 |

**B. 族へ割り当て（74）**

| 族 | Op | 数 | 方式 |
|---|---|---|---|
| #3077 | `Softmax`・`LogSoftmax` | 2 | V3 |
| #3078 | `LayerNorm`・`RmsNorm` | 2 | V3 |
| #3079 | `BatchNorm` | 1 | V3（推論時は V1） |
| #3080 | `Conv2d`・`ConvTranspose2d`・`Conv3d`・`ConvTranspose3d`・`Unfold`・`Fold` | 6 | d_input は転置畳み込み合成、d_weight／d_bias は V3。`Unfold`・`Fold` は閉包性のため同梱 |
| #3082 | `MseLoss`・`L1Loss`・`HuberLoss`・`PoissonNllLoss`・`GaussianNllLoss`・`KlDivLoss`・`BceLoss`・`BceWithLogitsPosWeightLoss`・`SoftMarginLoss`・`SigmoidFocalLoss`・`MarginRankingLoss`・`HingeEmbeddingLoss`・`CrossEntropyLoss`・`CrossEntropyLossWithOptions`・`NllLoss`・`MultiMarginLoss`・`MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`CosineEmbeddingLoss`・`TripletMarginLoss` | 20 | V1。f64 の scale・縮約を持つものは V3 |
| #3082（fail-closed） | `CtcLoss` | 1 | 族に置いたまま fail-closed。ホストの動的計画法（alpha／beta）を `Var` 演算で再生できない |
| #3083 | `Gather`・`Scatter`・`IndexedScatterReduce`・`Embedding`・`Cumsum`・`Cumprod`・`Logcumsumexp`・`Cummax`・`Cummin`・`Sort`・`Topk` | 11 | 索引 V1、累積 V3、選択 V2 |
| #3084 | `Pad`・`PadMode`・`MaskedFill` | 3 | V1／V2 |
| #3085 | `MaxPool2d`・`MaxPool3d`・`AvgPool2d`・`AvgPool3d`・`AdaptiveAvgPool2d` | 5 | Max 系 V2、Avg 系 V3 |
| #3086 対象 | `Inv`・`Solve`・`Det`・`SlogdetLogAbsDet` | 4 | V1（`Det` は V3。下記 `Det` の数値契約） |
| #3086 条件付き | `Fft`・`Ifft`・`Rfft`・`Irfft` | 4 | 随伴写像を g に作用させる形。随伴が replayable な `Var` 演算（`Var::fft`／`ifft`・スケール・Hermitian 重みマスク）で書けることを #3086 の冒頭で確かめ、書けなければ fail-closed |
| #3086 fail-closed | `QrQ`・`QrR`・`SvdU`・`SvdS`・`SvdVh`・`EighValues`・`EighVectors`（多出力 payload。§8）、`Cholesky`・`Pinv`・`Lstsq`・`MatrixNorm`（二階式が複雑）、`SlogdetSign`・`MatrixRank`（1 階 VJP が恒等ゼロ。`grad.rs:2279`／`:2303`）、`Stft`・`Istft`（窓と overlap-add） | 15 | fail-closed |

- #3086 の判定基準は「1 階 VJP を replayable な公開 `Var` メソッドだけで書き直せるか」。`Inv`／`Solve`／`Det` は `Var::inv`／`solve`／`det`（`var.rs:6740`／`:6761`／`:6806`）で再生でき、cgrad は写しの `Inv` 出力・`matmul`・`transpose` の合成（例 `-Yᵀ g Yᵀ`、`g·det·A⁻ᵀ`）で書ける。`Inv` を最初に replayable にする。
- **`Det` の数値契約（V1 単独では不足。V3 に変更）**: `g·det·A⁻ᵀ` を再生された `Var::det` の f32 出力で組むと、`A=diag(1e20,1e20)`・`g=1` で `det=1e40` が f32 の `inf` に丸まり、本来有限な対角勾配 `1e20` が `inf`、非対角が `NaN` になる。既存の `eval::linalg::det_vjp`（`crates/autodiff/src/eval/linalg.rs:1229`）は forward の丸め済み出力を使わず det と逆行列の積を f64 で保持して 1 回だけ downcast するので、この入力でも有限である。したがって `Det` は V3 とする。値は `det_vjp` と同じホスト関数（`pub(crate)`）で 1 階と bit 同一に計算する。構造は大スケール入力を再生時まで保つ**行別**スケール付き合成: 行 `i` ごとに `r_i`（その行の最大絶対値以上の 2 のべき乗。厳密に割れる。全ゼロ行は `r_i=1`）を取り、`B=D⁻¹A`（`D=diag(r)`）を作る。`det(A)=(∏r_k)·det(B)`、`A⁻ᵀ=D⁻¹B⁻ᵀ` なので `∂det/∂A_ij=(∏_{k≠i}r_k)·det(B)·(B⁻ᵀ)_ij` となる。係数 `∏_{k≠i}r_k` は指数（2 のべき乗の指数の和 `E_i`）をホストで整数として足すが、**係数そのものを 1 個の f32 定数として `push_eager` してはならない**（`A=diag(1e20,1e20,1e20)`・`g=1e-30` では既存 `det_vjp` の対角勾配は約 `1e10` で有限だが、行別係数は約 `1e40` で f32 定数化により `inf` になる。`det(B)`・`B⁻¹` は有限なので S2 への切替条件にも当たらない）。**指数分割と g との適用順**: `g` を行列全体で共通の 2 のべき乗 `s_g` で割った `g̃=g/s_g` を作り（`s_g` は **f32 で表現可能な範囲 `[2^-126, 2^127]` に収める**: `s_g = clamp(max(abs(g)) 以上の最小の 2 のべき乗, 2^-126, 2^127)`。`g=0` は `s_g=1`。2 のべき乗による除算は厳密。`max(abs(g)) > 2^127` の入力〈例 `A=I(2×2)`・`g=3e38` では素朴な選び方だと `s_g=2^128` が f32 で `inf` になり `g/s_g` が `0` になる〉は `s_g=2^127` に固定し、`g̃` は絶対値 2 以下の有限値になる〈`3e38/2^127≈1.76`〉。`g̃` の絶対値が 1 を超えうる点は、後続チャンクの単調性〈実行中の値が最終値を超えて overshoot しない〉の議論に影響しない上限 2 倍の余裕として扱う。`log2(s_g)` は f32 定数ではなくホスト整数として `c_i` に足すので、clamp で `g̃` に残らなかった指数分も `c_i` が保持し最終値は変わらない）、`g̃` を作り、各行の合計指数 `c_i=E_i+log2(s_g)` を ±2^100 以下の 2 のべき乗チャンクの積へ分割して、`g̃·det(B)·(B⁻ᵀ)_ij` へチャンクごとに `Mul` で順に掛ける（各チャンクは f32 で有限な定数。符号は `c_i` の符号に揃え、実行中の値の大きさが最終値へ向かって単調に動き最終値を超えて overshoot しないようにする。最終値が真に f32 を超える場合のみ 1 階と同じく `inf`）。上記の例では `c_i` が最終的に相殺されて途中の値が `1e40` を経由せず `1e10` で有限になる。この構造（`Mul` のチャンク列。`Contiguous`・`Sum` 等の VJP が前方値に依存しない Op タグ）を `push_eager` する。**最終値だけの V3 補正では修復できない理由**: V3 はノードの最終値だけをホスト関数に置き換えるため、`inf` の中間定数ノードが二階微分・孫テープ再生にそのまま使われる。構造側でチャンク分割を保たなければ 1 階が bit 一致でも二階以降が `inf`／`NaN` になる。下記の適用条件を満たす入力では、孫テープ再生でも同じ行別スケール構造が再計算される（孫テープ再生での有限性は #3086 の必須テストで確かめる）。真に f32 を超える勾配は 1 階と同じく `inf`。**一律スケーリング（最大要素 1 つで全体を割る方式）は採らない**: `A=diag(1e30,1e-30)`・`g=1` では既存 `det_vjp` は有限な `[1e-30,1e30]` を返すが、`s≈1e30` で `B` を作ると第 2 対角が約 `1e-60` で f32 の `0` に丸まり `B` が特異になって `B⁻ᵀ` を構築できず、V3 の値補正でも構造側の失敗は直らない。行別スケールなら `B=I`・`r=[1e30,1e-30]` で有限に保たれる。**適用条件と代替経路**: 行別スケール後も `B` が f32 で特異・非有限になる入力（1 行の内部に f32 の表現幅を超える桁差があり行別スケールでも情報が失われる場合。`det_vjp` の f64 では有限）は本方式の適用外とし、構造は S2 に切り替える（`det_vjp` と同じ f64 ホスト関数で勾配値を `push_eager` した定数ノードとして保持し、二階の微分は解析式〈`∂²det` は余因子行列と `A⁻¹` の積〉で別途合成する。それも合成できない場合は V4〈専用 VJP〉）。適用外の判定はホストで `B` の f32 `det`／逆行列が有限かを確認して行い、**行列式のアンダーフローも適用外に含める**: `B` の f32 `det(B)` が `0` または非正規化数になる一方で `B⁻¹` は有限・f64 の `det(B)` は非零という入力（例 `A=1.001·I`〈160×160〉・`g=1` では行別スケール後の `B=0.5005·I` の `det(B)≈8e-49` が f32 で `0` になるが、`B⁻¹` は有限なので有限性判定だけでは S2 へ切り替わらない）を、ホストで `det(B)` を f64 で計算して f32 の最小正規数 `2^-126` 未満の非零値になるか、または f32 `det(B)==0` かつ f64 `det(B)≠0` となることで検出し、S2／V4 へ切り替える（`det(B)` を仮数・指数に分離して保持する構成は、`det(B)` が `Var` の f32 ノードである限り再生側で表現できないため採らない）。真に特異な行列（f64 でも `det(B)==0`）は適用外にせず上記の既存扱いに従う。**総括的な適用判定（個別条件は既知の失敗例であり網羅を主張しない）**: 上記の個別条件に加え、構造を組む時点でホストが、構築する f32 構造の各段（チャンク適用前の積 `g̃·det(B)·(B⁻ᵀ)_ij` と各チャンク適用後の値）を f32 で再計算し、`det_vjp` と同じ f64 計算による参照値と比べて、有限性の食い違い（f64 で有限なのに f32 で `inf`／`NaN`）またはゼロ化の食い違い（f64 で非零なのに f32 で `0`）が 1 要素でもあれば S2／V4 に切り替える。指数復元前の積のアンダーフロー（例: 256 次上三角・対角 1.5・`A[0,1]=2^-100`・`g=2^-100` で積が約 `2^-207` となり f32 で `0` になる）もこの判定で検出する。本項の数値契約が保証するのは、この判定を通った入力で 1 階・`cg.grad` が有限かつ 1 階と REQ-2 判定で一致することまでであり、判定外の入力は S2／V4 の契約に従う。`Inv`／`Solve` も同様に中間積が f32 範囲を超えうるため、族の必須テストに大スケール入力（`diag(1e20,1e20)`・`g=1`。`g` 側の回帰として `A=I(2×2)`・`g=3e38`、行列式アンダーフローの回帰として `A=1.001·I`〈160×160〉・`g=1`〈S2／V4 へ切り替わり有限〉も必須。指数復元前の積アンダーフローの回帰として 256 次上三角〈対角 1.5・`A[0,1]=2^-100`〉・`g=2^-100`〈総括的な適用判定で S2／V4 へ切り替わる〉も必須。係数オーバーフローの回帰として `diag(1e20,1e20,1e20)`・`g=1e-30` も必須。1 階の対角勾配約 `1e10`）と混在スケール入力（`diag(1e30,1e-30)`・`g=1`。既存 `det_vjp` は `[1e-30,1e30]`）を含め、1 階・`cg.grad`・孫テープ再生で有限かつ 1 階と REQ-2 判定で一致することを確かめる。特異行列（`det_vjp` の既存扱い）も同じ契約に従う。上表 #3086 の `Det` の方式欄は V1 ではなく本項（V3）を正とする。
- 表の数: 族割当 2+2+1+6+20+11+3+5+4+4 = 58、`CtcLoss` 1、#3086 fail-closed 15 で、合計 74。

**C. 本ラウンドで fail-closed に固定（族なし。24）**

| 候補グループ | Op | 補足 |
|---|---|---|
| 選択縮約 | `Max`・`Min`・`Amax`・`Amin`・`OrderSelect`・`MedianAll`・`Quantile` | V2 の前例があり技術的には可能 |
| 統計縮約 | `Var`・`Std`・`VectorNorm`・`PNorm`・`LogSumExp`・`Nansum`・`Nanmean` | V3 |
| 構造変換 | `Dropout`・`Interpolate`・`MaxUnpool` | `Dropout` は記録済みマスクの `where_cond` で最小 |
| 正規化・再パラメータ | `LocalResponseNorm`・`WeightNorm`・`SpectralNorm` | |
| RNN セル | `RnnCell`・`LstmCell`・`LstmHidden`・`GruCell` | 恒久寄り。利用者が replayable な primitive で手組みすれば create_graph は通る |

- 割り当てには新規イシュー起票（ユーザー承認）が要るため、自動運転では fail-closed に固定する。
- 照合: A 5 + B 74 + C 24 = 103。`ScalarUnaryOp` は `Op` の行を持たないので #3081（21.5）は表の外。

### 21.3 族ごとの方針

- **#3077 Softmax・LogSoftmax**: R は `Var::softmax`／`log_softmax`（`var.rs:2042`／`:2081`）。V3 で、値は `softmax_vjp_along`／`log_softmax_vjp_along`（`grad.rs:6997`／`:7043`。f64 保持・最後に 1 回 downcast）。構造は Softmax が `y⊙g − y⊙Σ(g⊙y)`（§8 のベクトルで中間値が overflow しない並べ替え）、LogSoftmax が `g − exp(y_log)⊙Σg` を縮約前スケーリングで組んだ `s⊙(g/s − exp(y_log)⊙Σ(g/s))`（`s` は行ごとの `max(abs(g))` のホスト定数で、`s=0` の行は `s=1`。`g/s` の各要素は絶対値 1 以下なので `Σ(g/s)` は要素数以下に収まる。素朴な `Σg` は `y_log=[ln0.5,ln0.5]`・`g=[2e38,2e38]` で `inf` になり、最終値だけを V3 で補正しても二階微分・孫テープ再生にこの `inf` が残るため。§21.1 (5) の S1 に準拠し、`log_softmax_vjp_along` の f64 縮約が有限値を返すことと整合させる）。積む Op は `Mul`・`Sub`・`Sum`・`Exp`・`BroadcastTo`・`Contiguous`。必須テスト: §8 のベクトルに加え LogSoftmax は `y_log=[ln0.5,ln0.5]`・`g=[2e38,2e38]` で、1 階・`cg.grad`・孫テープ再生（`jvp_create_graph` 経由）のすべてが有限かつ 1 階と bit 一致。V4 は採らない。
- **#3078 LayerNorm・RmsNorm**: V3。値は `layer_norm_vjp_rows`／`rmsnorm_vjp_rows`（`grad.rs:6043`／`:5979`）。構造は mean／rstd を `Mean`・`Sub`・`Mul`・`Sqrt`・`Div` で組み直す。**`LayerNorm` の中心化は `Mean` の f32 出力を早期に引かない**（`x=[16777216,16777218]` の正しい平均は 16777217 だが f32 の `Mean` 出力は 16777216 に丸まり、偏差が `[-1,1]` ではなく `[0,2]` になる。既存 `row_ln_stats`／`layer_norm_vjp_rows` は平均と偏差を f64 で保持するため構造側と食い違い、二階微分・孫テープ再生に誤差が残る）。安定な中心化として、行ごとの基準値 `x_ref`（行の先頭要素を `Narrow` で取り `BroadcastTo`。いずれも replayable）を先に引いて `d=x−x_ref` を作り、`m=Mean(d)`、`dev=d−m` とする（上記の入力では `d=[0,2]`・`m=1`・`dev=[-1,1]` で厳密）。さらに `dev2=dev−Mean(dev)` の補正パスを 1 回加えて残差の丸めを除く（§21.1 (5) の S1 の `s` は `dev2` の最大絶対値から取る）。`d=x−x_ref` 自体が f32 で非有限になる入力（`x=[3e38,-3e38]` 等）は S2 へ切り替える。`Rsqrt`／`Reciprocal`・`Op::Var` は replayable でないので使わない（閉包性）。二乗和の「先に f64 へ昇格してから二乗」契約（`coding-rust.md`）は V3 の値側で保つ。中間値の overflow は V3 の値補正では防げない（`x=1e20` の f32 二乗が `inf`、`rstd=0` となりこの誤値が二階微分・再生に使われる）ため、21.1 (5) の S1（スケール付き合成。行ごとの最大絶対値 `s` で `x/s` を作ってから二乗和・rstd を合成し `rstd/s` で戻す）を既定とし、`LayerNorm` の分散項にも同様に適用する。`eps` は `eps/s²` に変換し、`s` は 21.1 (5) の規則（`eps/s²` の overflow 保護のため `max(最大絶対値, √eps)` を 2 のべき乗へ切り上げる。小スケール入力 `x=[1e-30,1e-30]`・`eps=1e-5` で `rstd≈316.23` を保つ）に従い、最大絶対値は `LayerNorm` では中心化後偏差の最大絶対値、`RmsNorm` では `x` の最大絶対値とする（`eps/s²` が f32 で 0 に丸まっても、スケール後分散が `1/n` 以上なので `rstd` は不変。中心化前の `x` から `s` を取ると定数行で `rstd_scaled=inf` になるため禁止）。非有限になる入力は S2 へ切り替える。必須テスト: 中心化の丸め回帰 `x=[16777216,16777218]`（偏差が厳密に `[-1,1]`・`mean` が 16777217 相当、既存 `row_ln_stats` と一致）、有限入力 `x=1e20`、小スケール行 `x=[1e-30,1e-30]`・`eps=1e-5`（`eps/s²` overflow の回帰。`RmsNorm` の `rstd≈316.23` が 1 階と一致）、定数行 `x=[1e30,1e30]`・`eps=1e-5`（`LayerNorm` の `eps/s²` 丸め消失の回帰。`rstd=1/√eps` で有限）と全ゼロ行（`s=0`→`s=1` 規則の回帰。統計量・入力勾配が有限で `NaN` を含まない）で、1 階・`cg.grad`・孫テープ再生のすべてが有限かつ 1 階と REQ-2 判定で一致する。**総括的な適用判定（§27.3。PR #3346 のレビューで追加）**: upstream `g`・`w`・`rstd` の組み合わせによる構造の中間値（`cg.grad`・二階の逆伝播・孫テープ再生）の overflow は、構造を組む前にホストが各段の上界を f64 で求めて判定し、f32 の安全域を外れる入力は型付きエラーで拒否する。本項の数値契約が保証するのはこの判定（と §27.2 の入口検査）を通った入力に限る。
- **#3079 BatchNorm**: V3（`batch_norm_vjp_channels`、`grad.rs:6139`）。学習時と推論時で分岐し、推論時は統計量が定数なので V1 で足りる。
- **#3080 Conv 系**: d_input は転置畳み込みの合成（族内で閉じる）。d_weight／d_bias は V3（f64 縮約契約。`grad.rs:1380` 以降）で、構造は `Var::unfold`（`var.rs:5575`）＋matmul を推奨し、閉包性のため `Unfold`・`Fold` を同梱する。`compute_dtype` が F32 でない `Conv2d` は `TapeNode::low_precision` を立てない（立てるのは `low_precision_ops.rs:93`・`var.rs:3652` の matmul 系だけ）ので、`validate_ancestors` に明示的な拒否を足す。
- **#3081 Gelu・GeluTanh**: 21.5。
- **#3082 損失**: 要素ごとの合成は V1、1 階が f64 の scale・縮約を持つものは V3。`CrossEntropyLoss` 系は cgrad に Softmax を積むので #3077 に依存する。target を一律に非微分入力として扱わない。非微分なのは整数ラベル（`CrossEntropyLoss` 系の `Tensor<i32>` ペイロード）だけであり、浮動小数の target は追跡可能な入力として区別する。既存の 1 階 VJP が返す全入力への寄与を子テープでも維持する: `MseLoss` は `dTarget=-dPred`、`PoissonNllLoss` は `dtarget`、`GaussianNllLoss` は `dtarget` と `dvar` を返すため、target（と var）を定数化すると target 自身の勾配に加え pred と target の混合二階微分も失われる。`build_cgrads` は 1 階の `grad.rs` が返す入力数と同数の寄与を子テープに積み、各寄与の依存関係（pred・target・var への経路）を保つ。必須テスト: target に `requires_grad` を持たせ、`d²L/dpred·dtarget` の混合項をループ参照（有限差分または解析式）と一致確認する。
- **#3083 索引・累積**: 索引系は「Gather の VJP は scatter_add」の双対性を使い、ホストの index を固定して V1 で書く。`Cumsum` は V3（`cumsum_vjp_along`、`grad.rs:7289`。lane ごとの f64）、`Cumprod`・`Logcumsumexp` も V3。`Sort`・`Topk`・`Cummax`・`Cummin` は V2（ホスト index と scatter）。
- **#3084 Pad・PadMode・MaskedFill**: 21.5。`PadMode`（reflect／replicate／circular）は `value` を持たない別 variant で、cgrad は index ベースの scatter-add なので #3083 の索引系に依存する。
- **#3085 プーリング**: Max 系は V2（先勝ち argmax をホストで作り scatter で流す）で #3083 の索引系に依存。Avg 系・Adaptive は V3（`avg_pool2d_vjp`／`adaptive_avg_pool2d_vjp`、`grad.rs:7116`／`:7219`）で、構造は `Fold`（col2im）経由なので #3080 に依存する。
- **#3086 線形代数・FFT**: 21.2 の行ごとの判定に従う。

### 21.4 再分解方針と族間の依存順

2 時間を超える族は次のサブイシューへ分ける。起票は `out-of-scope-tracking.md` に従いユーザー承認後で、本 PR では行わない。

| 族 | 分割 |
|---|---|
| #3080 | (a) 2D（`Conv2d`・`ConvTranspose2d`・`Unfold`・`Fold`）／(b) 3D（`Conv3d`・`ConvTranspose3d`） |
| #3082 | (a) 回帰系 6（`Mse`・`L1`・`Huber`・`PoissonNll`・`GaussianNll`・`KlDiv`）／(b) 二値・マージン系 6（`Bce`・`BceWithLogitsPosWeight`・`SoftMargin`・`SigmoidFocal`・`MarginRanking`・`HingeEmbedding`）／(c) クラス軸・距離系 8（`CE`・`CEWithOptions`・`Nll`・`MultiMargin`・`MultiLabelMargin`・`MultiLabelSoftMargin`・`CosineEmbedding`・`TripletMargin`） |
| #3083 | (a) 索引 4／(b) 累積 3／(c) 選択 4 |
| #3085 | (a) Max 系 2／(b) Avg 系 3 |
| #3086 | (a) 線形代数 4／(b) FFT 4（条件付き） |

依存順（積む Op が他族にある場合）:
- #3083(a) → #3083(c)・#3084 の `PadMode`・#3085(a)
- #3080(a) → #3085(b)
- #3077 → #3082(c)
- #3081 は `Erf` を同時に replayable にする（族内で閉じる）

### 21.5 個別論点の結論

**`Gelu`: 実装する**
- R は `Var::scalar_unary(Gelu)`。V1 で `g⊙(Φ(x)+x·φ(x))`、Φ は `0.5·(1+erf(x/√2))` を `ScalarUnaryOp::Erf`（`tensor-core/src/scalar_op.rs:228`）に載せ、φ は `Exp` で作る。
- 閉包性のため `Erf` も `scalar_unary_replayable` に加え、`build_cgrads` に `g⊙(2/√π)·exp(−x²)` の腕を足す（導関数は `erf_grad`〈`scalar_op.rs:756`〉と同じ式）。
- forward の `erf` は自作近似（Abramowitz–Stegun 7.1.26、最大絶対誤差 1.5e-7）、導関数は解析式という規約を 1 階の `erf_grad`・`gelu_erf_grad`（`scalar_op.rs:719`）が既に採っているので踏襲する。有限差分テストは近似誤差 / h が REQ-2 を超えない刻みを選ぶ。
- 1 階は f64 で計算し 1 回だけ downcast する。子テープの f32 合成との差は §9 (i) の範囲。`erf` クレートは deps-policy により追加しない。
- `new_2649_kinds_are_not_replayable`（`create_graph.rs:1610`）は、`Erf` を `true` にしても `Selu` などが `false` のままであることを保つ回帰テストとして残す。

**`GeluTanh`: 実装する**
- V1 で `0.5(1+tanh u) + 0.5·x·sech²(u)·u'`（`Tanh`・`Mul`・`Add`）。`gelu_tanh_grad`（`scalar_op.rs:731`）の `sech2 == 0.0` ガードはホストの `mask_from_pred` と `where_cond` へ写し（`Hardswish` の前例）、飽和域の第 2 項を定数 0 にする。

**`Pad`・`MaskedFill`: payload に `value: f32` を足す**
- `value` を使うのは `replay_op` だけで、どちらの VJP も使わない（Pad は narrow の連鎖、MaskedFill は `g⊙(mask==0)`）。
- 生成側は `Var::pad`（`var.rs:3319`）と `Var::masked_fill`（`var.rs:4404`）が設定する。分解側は `grad.rs:2514`・`:2536` が `..` なしで束縛しているので `..` を足す（`tape.rs:2204`／`:2219` は済み）。
- cgrad は Pad が narrow の連鎖、MaskedFill が `where_cond(mask, 0, g)`（積は `inf*0` を避けるため使わない）。`Op` は `pub(crate)` で `api_surface.rs` に影響しない。

### 21.6 fail-closed 契約

- `supports_create_graph == false` の Op に `backward_create_graph` が到達したら `Err(AutodiffError::Backward(_))` を返し、子テープは変更しない（`validate_ancestors` が子テープへ書く前に拒否する）。
- 判定対象は「`requires_grad == true` かつ葉でない祖先」だけで、追跡なしノードは Op の種類によらず `build_mirror` の段 1 で定数葉になる。
- 入口の判定順は、resident／fused、`low_precision`、`supports_create_graph`、`MatMul` の rank、（追加予定）`Conv2d` の `compute_dtype`。
- 網羅 match（ワイルドカードなし）と、`ScalarUnaryOp`／`ScalarBinaryOp` の未知 variant を `false` へ倒す規約は弱めない。
- 基準テストは `create_graph_rejects_unsupported_op_custom`（`crates/autodiff/tests/create_graph.rs:734`。`child.is_empty()` の検査を含む）。族イシューでは、fail-closed のまま残る隣接 Op を 1 つ選び拒否を確かめるテストを推奨する。

### 21.7 所有者へ戻す論点

記録に形が書けない点だけを挙げる。
1. C 区分 24 variant を将来の族として起票するか（21.2。起票にはユーザー承認が要る）。
2. 21.4 のサブイシューを起票するか。
3. V3（値と Op タグを分ける方式）を `reduce_bias_grad_var` 以外へ広げる点は前例があるので推奨形として確定しているが、レビューで争点になった場合に限り戻す。

### 21.8 変更していないもの

コード・依存・`unsafe`・tolerance・baseline・`docs/spec`・既存節の本文。

## 23. 実装記録（#3080・Conv 系 2D の create_graph 対応）

- 基準コミットは `origin/main` `6d08419a`。方針の正は §21（21.2 の #3080 族・21.3 の conv 方針・21.4 の 2D／3D 分割・21.6 の fail-closed）で、本節は実装結果だけを書く。
- 実装範囲は §21.4 (a)＝2D の 4 variant（`Conv2d`・`ConvTranspose2d`・`Unfold`・`Fold`）。`Conv3d`・`ConvTranspose3d` は `supports_create_graph() == false` のまま残し、入口検査が拒否する（(b) への申し送り）。

### 23.1 実装した内容

| 項目 | 実装 |
|---|---|
| 対象判定 | `Op::supports_create_graph`（`tape.rs`）の `true` 側へ 4 variant を移動した。網羅 match を維持し、ワイルドカードは足していない |
| 入口検査 | `validate_ancestors`（`create_graph.rs`）に `Conv2d` の `compute_dtype != F32` 拒否を追加した（判定順は `MatMul` の rank の直前）。子テープへ書く前に `Err(Backward)` で拒否し、`child.is_empty()` が保たれる。`replay_op` の `Conv2d` 腕にも同条件を置いた（多層防御） |
| forward 再生 | `create_graph/conv.rs` の `replay_*`。公開 `Var` メソッドへの委譲 |
| 二階 VJP | `create_graph/conv.rs` の `conv2d_vjp`・`conv_transpose2d_vjp`・`unfold_vjp`・`fold_vjp` |
| 共有ホスト関数 | `grad.rs` に `conv2d_dw_from_col`・`conv_transpose2d_dw_from_col` を抽出した（1 階の既存腕は式・評価順・呼び出し回数を変えず、これらを呼ぶだけにした） |

- `output_padding`（`ConvTranspose2d`）と `output_size`（`Fold`）は `Op` に足さず、再生対象ノード自身の shape から導出する。`output_padding = out − ((in−1)·s − 2p + d·(k−1) + 1)` を `checked_*` で計算し、負・桁あふれは型付きエラー。`output_padding < stride` の検査は委譲先の `Var::conv_transpose2d` が行う。
- 子テープへ積む Op は `Narrow`・`Reshape`・`Permute`・`Contiguous`・`Concat`・`MatMul`（rank 2）・`Unfold`・`Fold`・`Sum` だけで、すべて replayable（閉包性。§21.1(4)）。N 軸は `Permute`＋`Reshape` で列軸へ畳み込み、groups は `Narrow` のループと `Var::cat` で扱う。
- matmul は `matmul_fp32_strict` を使う（PR #2003 の前例）。`Var::conv2d`／`conv_transpose2d` を VJP 内で呼ばないのは、forward が非 strict の `gemm_batched` を通り、CUDA の TF32 opt-in 下でも backprop を FP32 厳密に保つ契約を破るため。
- K 軸の並び `(c, kh, kw)` は、groups 付き im2col（`[N, G, K_g, P]`）の平坦化と groups = 1 の `unfold`／`fold` で一致する。このため groups > 1 でも `Var::unfold`／`fold` を使える。
- 要素数 0 の入力は matmul を呼ばず、子テープ上のゼロ定数を返す。

### 23.2 方式（V1／V3）

- d_input は V1（unfold／fold と groups ごとの rank 2 strict matmul の合成）。
- d_weight は V3。構造は V1 と同じ合成、値は 1 階と同じホスト関数（`gemm_batched_fp32_strict`＋N 軸 `f64` 縮約）の結果を専用 Op `Op::ConvWeightGradHost { structure, g, x, params, transposed }` で `push_eager` する。VJP は恒等で `structure` のみへ渡し、孫テープでの再生（`replay_op`）は同じホスト関数で値を再計算する。
- d_bias は V3。`reduce_bias_grad_var`（`eval::reduce_bias_grad_rows`）を再利用する。
- `Unfold`／`Fold` は V1（1 階と同じ `col2im`／`im2col` を `fold`／`unfold` として書く。値も一致する）。

### 23.3 数値契約と残余（§21.1(5) との関係）

- 1 階と `cg.grad`（d_weight・d_bias）は bit 一致する。N·P 方向の長軸縮約が f32 逐次和では overflow し、`f64` 縮約では有限になる入力（N=3・1×1 カーネル・`x=[1,1,-1]`・`g=[3e38;3]`）の回帰テストで固定した。
- conv は x と w に対する双線形写像で、構造側の合成（`MatMul`・`Permute`・`Reshape`・`Narrow`・`Concat`・`Unfold`・`Fold`）の VJP は各ノード自身の前方値を読まない（`MatMul` が読むのはオペランドの値だけ）。そのため二階の微分値は V3 の前方値だけで有限かつ正しい。
- 孫テープ（3 段目以降）で再生しても d_weight の前方値は同じホスト関数で再計算され、上の overflow 入力でも有限のまま 1 階と一致する（`Op::ConvWeightGradHost`。`Op::Contiguous` タグでは再生時に f32 の構造側合成へ落ちて `inf` になっていた〈レビュー指摘 P1〉）。孫テープ上で `h = (gw·1e-38)²` を作って更に微分した x 勾配が通常の `backward` と bit 一致する回帰テストで固定した。
- `ConvTranspose2d` の `output_padding` は `out + 2p − ((in−1)·s + d·(k−1) + 1)` を符号付き（`i128`）で導出する（出力長から `output_padding` を除いた base は `2p` が大きいと負になりうる）。
- d_bias は入力・重みがゼロ次元でも上流 `g` が非空なら通常どおり縮約する（`Cin = 0` で 1 階と一致）。
- 代替の S1（`g/s_g`・`col/s_x` で matmul を取り 2 のべき乗で戻す）は採らない。順方向は有限に保てるが、逆伝播で上流勾配へ `2^(e_g+e_x)` のチャンク積が先に掛かり、真値が有限でも二階の微分値が `inf` になる（overflow を二階側へ移すだけ）。
- tolerance・baseline・FMA 契約は変更しない。

### 23.4 テスト

`crates/autodiff/tests/create_graph.rs`（追加分）:

- Hessian と有限差分の突合: `Conv2d`・`ConvTranspose2d` の `{bias あり・なし} × {groups 1・2}`、`stride=2・padding=1`、`dilation=2`、`ConvTranspose2d` の `stride=2・output_padding=1` と `dilation=2`、`Unfold`・`Fold` 単体（窓が重なる設定）。x・w・b を 1 本の葉から切り出し、純ブロックと混合ブロックを同時に検証する。loss は多項式 `sum(y⊙y⊙c)`。f32 勾配の丸めノイズが小さい Hessian 成分で絶対許容を超えるため、h を大きく取り Richardson 外挿（多項式 loss では打ち切り誤差が厳密に消える）で補った。tolerance は変更していない。
- 子テープ上の 1 階勾配（`cg.grad`）と `Tape::backward` の REQ-2 一致、`first_order()` の bit 同一。
- d_weight・d_bias の overflow 回帰（`Conv2d`・`ConvTranspose2d` それぞれ）と、二階・3 段目の微分値が有限であること。
- `jvp_create_graph`（親→子→孫）と独立な中央差分の方向微分の一致（`Conv2d`・`ConvTranspose2d`）。
- fail-closed: `Conv3d` が `Err(Backward)` かつ `child.is_empty()`。低精度 `Conv2d`（F16）は `Var::conv2d_low_precision` と F16 の typed ops が crate 内限定のため、`create_graph.rs` 内の単体テスト `conv_low_precision_tests` で固定した。

既存テストの更新: `tape.rs` の `unfold_and_fold_op_meta_properties`（`supports_create_graph` を `true` へ反転）、`conv2d_family_supports_create_graph`（新規）、`fold_unfold_parity.rs` の `records_exactly_one_node_and_supports_create_graph`（旧: 拒否を確認）。`conv_transpose3d_parity.rs`・`pool3d_parity.rs` の拒否テストは 3D のため変更していない。

### 23.5 所有者へ戻す論点

1. §21.4 (b)（3D: `Conv3d`・`ConvTranspose3d`）のサブイシュー起票（起票にはユーザー承認が要る）。
2. CUDA・Metal 実機で create_graph と conv を組み合わせた実測は未実施（申し送り）。

### 23.6 変更していないもの

1 階 `Tape::backward` の数値（bit 同一）・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文。

## 22. 実装記録（#3081: Gelu・GeluTanh・Erf の create_graph 対応）

§21.5 の決定に従い実装した。コードは `crates/autodiff/src/create_graph.rs`（`scalar_unary_replayable`・`build_cgrads` の `Op::ScalarUnary` 腕）。

### 22.1 実装形

- `scalar_unary_replayable` に `Gelu`・`GeluTanh`・`Erf` を追加した。`Erf` は `Gelu` の `Φ(x)` を子テープへ `ScalarUnary(Erf)` として記録するため、孫テープでの再生（閉包性、§21.1 (4)）に必要で同時に対象化した。`replay_op` は `Var::scalar_unary` への委譲で全 variant に対応済みのため変更していない。
- `Erf`: `g ⊙ (2/√π)·exp(-x²)`（`erf_grad` と同じ順序）。
- `Gelu`: `g ⊙ (Φ(x) + x·φ(x))`。`Φ = 0.5(1 + erf(x/√2))`、`φ = exp(-x²/2)/√(2π)`（`gelu_erf_grad` と同順）。子テープに積む Op は `Mul`・`Add`・`Exp`・`ScalarUnary(Erf)` のみ。
- `GeluTanh`: `0.5(1+tanh u) + 0.5·x·sech²(u)·u'`。飽和判定（`1 - tanh² == 0`）は 1 階 `gelu_tanh_grad` と同じ f32 演算でホスト側マスク化する。さらに `where_cond` の死に枝へも 0 の余接が流れ、後続 `Mul` の VJP が `0·相方` を作る（相方が `inf` だと NaN が 2 階へ混入する）ため、飽和要素を 0 に置換した `x_safe` で滑らかな式を組み（double-where）、飽和要素では `limit`（厳密に 0 または 1）を選ぶ。
- `Gelu` の裾（`x·x = inf`）は `exp(-inf) = 0` となり、2 階 backward でも `0·inf` が追跡ノードへ流れないことを回帰テストで確認した。追加の安全化は不要だった。

### 22.2 検証

- 有限差分（h=1e-3、点 `[-1.3, -0.4, 0.0, 0.7, 1.2]`）と REQ-2 統一複合判定で一致（tolerance は不変）。f32 の 1 階勾配を差分するためノイズが約 1e-4 あり、|x| が大きいと 2 階導関数が絶対 1e-5 を満たせないので、|x| ≥ 3 は f64 閉形式（Gelu: `φ(x)(2−x²)`）と突合した。
- 近似 erf（A-S 7.1.26）を含む 1 階と解析式の φ との整合は、上記突合が REQ-2 内で通ることをもって確認した（不整合は出なかったため方式の見直しは不要。最大差の数値は未採取）。
- 巨大入力（±1e20・±1e30・±3e38）で `cg.grad`・Hessian が有限（Hessian は `abs < 1e-6`）。
- `first_order` は素の `backward` と bit 同一。孫テープでの再生は `functional_ops_jvp_tests` の `j2`・`double_vjp_feasibility` の `c2` に gelu・gelu_tanh・erf を追加して通した。

### 22.3 fail-closed テストの差し替え

`create_graph_rejects_unsupported_op_gelu` を廃止し、`create_graph_rejects_custom_downstream_of_gelu`（`Custom` 到達で `child.is_empty()`）と `create_graph_rejects_unsupported_scalar_unary_selu`（隣接 Op の拒否）へ置換した。`functional_ops_jvp_tests`・`double_vjp_feasibility` の gelu 拒否行は削除した（`selu` は残存）。

### 22.4 変更していないもの

1 階 `Tape::backward` の経路（`grad.rs`・スカラー評価・`tensor-core`）、`Op` payload、公開 API、依存、`unsafe`、tolerance・baseline、`docs/spec`、既存節の本文。

## 24. 実装記録（#3077・Softmax／LogSoftmax の create_graph 対応）

§21.1・§21.3 の V3 方式で `Op::Softmax`・`Op::LogSoftmax` を `supports_create_graph` の対象に加えた（網羅 match のまま）。

### 24.1 構造グラフと値

- 値は 1 階 VJP と同じホスト関数。構造は子テープに積む `Var` 演算で、値付きの `Op::Contiguous` ノードとして登録する。登録前に値と構造の shape が一致することを `Err(Backward)` で検査する。
- Softmax: `y⊙g − y⊙Σ_dim(g⊙y)`。積む Op は `Mul`・`Sum`・`Contiguous`・`Reshape`・`BroadcastTo`・`ScalarBinary(Sub)`。
- LogSoftmax: `s ⊙ (g/s − exp(y) ⊙ Σ_dim(g/s))`。`s` はレーンごとの `max|g|` をホストで求めた定数葉。`s == 0` と非有限のレーンは `s = 1` とし、IEEE の伝播は構造側に任せる。
- §21.3 の「積む Op」一覧に `ScalarBinary(Div)` を補足する。逆数を掛ける形は `s` が f32 最大値付近で非正規数になるため採らない。
- 空テンソル（shape に 0 を含む）は 1 階の早期 return と揃え、全ゼロの定数葉を寄与として返す。

### 24.1b 二階の逆伝播の overflow 保護

- Softmax: `y` を使用箇所（`y⊙g`・`y⊙dot`・`g⊙y` の縮約）ごとに入力から別の `softmax` ミラーとして立てる。各寄与を別々に Softmax VJP へ通してから入力側で蓄積するため、同一ノードでの寄与の和が f32 範囲を超えない（定数シフト不変性で相殺される量）。
- LogSoftmax: `exp(y)·Σgs` を `exp(y + ln n)·(Σgs / n)` に分け、`exp` ノードへの寄与を `-s·mean_gs`（`≤ s`）に抑える。さらに `q` の VJP 積 `q⊙寄与`（最大 `n·s`）が LogSoftmax VJP の手前で溢れる（確率偏り＋大スケール。例 `x=[ln9,0]`・`g=[3e38,3e38]`）ため、`q·mean_gs` を入力から別々に立てた `log_softmax` ミラーへ等分し、各寄与を別々に VJP へ通して入力側で蓄積する。分割数はホストで `n·p·s·|mean_gs|` の上界から決める（2e38 超過分のみ増やす・最大 64・通常 1）。`y` への寄与の総和は従来と同値。限界: 分割上限を超える規模は保護できない。分割数 2 以上の状態でさらに jvp_create_graph（三階）へ通す経路は NaN を生む場合を確認しており未保護（二階 Hessian 経路のみ保証）。
- 上流勾配が入力依存の場合の保護（codex-review 指摘）: `s⊙(gs − q·mean_gs)` を `g − q·mean_g`（`mean_g = Σ g/(n·pieces)`）へ恒等変形し、`s` を掛ける積と `g/s` 除算を作らない。`mean_g` 側へ `-s·q`（最大 `n·s`）が逆伝播しない。回帰: `create_graph_log_softmax_input_dependent_large_upstream_hessian_is_finite`（`x=[ln9,0]`・`loss=Σ y²·5e37`）。
- 回帰テスト: `create_graph_softmax_large_upstream_second_order_is_finite`・`create_graph_log_softmax_large_upstream_second_order_is_finite`（`cg.grad` の各要素を入力で微分した Hessian を f64 閉形式と比較）。
- 限界: 真の二階寄与そのものが f32 範囲を超える入力は保護できない。

### 24.2 LogSoftmax の値の dispatch（決定記録の解釈）

§21.3 は値を「ホスト関数」と書くが、1 階は `log_softmax_backward`（backend カーネル）を先に使う。§21.1(1) の bit 同一を CUDA でも満たすため、3 分岐 dispatch を `grad::log_softmax_vjp_dispatch` へ切り出し、1 階と子テープの両方が同じ関数を `ops` に対して呼ぶ。1 階の挙動は変えていない。Softmax には backend backward がないので dispatch は不要。`softmax_vjp_along`・`log_softmax_vjp_along` は `pub(crate)` にした。

### 24.3 テスト

- `crates/autodiff/tests/create_graph.rs`: `hessian_softmax_matches_finite_difference`・`hessian_log_softmax_matches_finite_difference`（軸 0／最終軸／rank 3 中間軸・自乗・`⊙x`・`tanh` 連鎖）、`create_graph_softmax_first_order_is_bit_identical`、`create_graph_softmax_first_order_chain_matches_req2`、`create_graph_softmax_large_upstream_does_not_overflow`・`create_graph_log_softmax_large_upstream_does_not_overflow`（§8・§21.3 のベクトル。1 階・`cg.grad`・`jvp_create_graph` の孫テープ再生がすべて有限）、`create_graph_softmax_empty_tensor_returns_zero_shaped_grad`、`create_graph_rejects_unsupported_op_logsumexp`（拒否契約は `LogSumExp` で維持）、`create_graph_rejects_softmax_mixed_with_unsupported_op_and_keeps_child_empty`。
- 孫テープ再生の値は REQ-2 判定で比較した（§21.1(3) のとおり合成側の f32 再計算差は範囲内として受け入れる）。bit 一致は主張しない。
- `crates/facade/tests/create_graph_facade.rs`: 実 CPU バックエンドでの 1 階 bit 一致と Hessian の列和の中央差分突合。

### 24.4 変更していないもの

1 階 `Tape::backward` の値・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文。

## 26. 実装記録（#3082・損失 Op 第 1 弾の create_graph 対応）

§21.4 の損失族のうち、1 階 VJP が f32 の要素式だけで f64 縮約契約を持たない 5 variant を `supports_create_graph` の対象に加えた（網羅 match のまま）。残りは fail-closed のまま下表に記録する。サブイシューの起票は所有者承認が要るため行っていない。

### 26.1 対象にした 5 variant と方式

| variant | 方式 | 二階の構造 |
|---|---|---|
| `MseLoss` | V1 | `dpred = (pred − target) ⊙ s`、`dtarget = −dpred`。`s` は Mean で `(g·2)/n`、Sum で `g·2` |
| `L1Loss` | V2 | `sign(d)` は区分定数。`pos`／`neg`／`nan` のマスクを親の値から作り `where_cond` で選ぶ。`d == 0` は 0、NaN は `d_m` を通して伝播する |
| `HuberLoss`（`Huber`／`SmoothL1`） | V1＋V2 | 二次領域（`!(|d| >= delta)`）は `g_b ⊙ d`（`SmoothL1` は `d/delta`）、線形領域は `g_b ⊙ copysign(delta または 1, d)`（非零定数）。`where_cond` で選ぶ |
| `CrossEntropyLoss` | V1 | `dlogits = (softmax(logits) − onehot) ⊙ s`。子テープに `Softmax` を積む。`onehot` は親の `targets` から作る定数 |
| `NllLoss` | V2 | `dinput = where(onehot_mask, −s, 0)`。入力について定数なので、二階の寄与は `g` を経由する分だけ |

- 上流勾配 `g` は子テープ上の `Var` のまま扱う（損失が根でない場合や孫テープ再生で、二階の勾配が `g` を通って流れるため）。scale は `g` に定数を掛ける・割る形で作り、演算順は 1 階（`mse_loss_scale`・`nll_loss_scale` 等）に合わせた。
- float の target は追跡可能な入力として扱い、`dtarget` は子テープの `Var`（`dpred.neg()`）にした。`requires_grad == false` の target は `accumulate` が捨てる。
- 区分定数の部分は §21.1 V2 に従い `where_cond` で作り、`g.mul(0/1 定数)` は使わない（`inf·0 = NaN` を避ける）。
- 範囲外の `targets` 添字は 1 階と同じくスキップする（forward が検査済み。panic しない）。空テンソルは 1 階と同じく全ゼロを返す。
- `replay_op` は公開 `Var` メソッド（`mse_loss_with`・`l1_loss`・`huber_loss_impl`・`cross_entropy_loss`・`nll_loss`）へ委譲し、forward の検証を迂回しない。`huber_loss_impl` は `pub(crate)` にした（公開 API は不変）。未知の `HuberKind` は `Err(Backward)` に倒す。

### 26.2 残した 16 variant（fail-closed のまま）

| 族（§21.4） | variant | 理由・後続 |
|---|---|---|
| (a) 回帰 | `PoissonNllLoss`・`GaussianNllLoss`・`KlDivLoss` | 未着手。次弾の候補 |
| (b) 二値・マージン | `BceLoss`・`BceWithLogitsPosWeightLoss`・`SoftMarginLoss`・`SigmoidFocalLoss`・`MarginRankingLoss`・`HingeEmbeddingLoss` | 未着手 |
| (c) クラス軸 | `CrossEntropyLossWithOptions`・`MultiMarginLoss`・`MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`CosineEmbeddingLoss`・`TripletMarginLoss` | 未着手 |
| 恒久 | `CtcLoss` | §21.2 のとおり恒久的に fail-closed |

### 26.3 `reduction` と既知の差

- `crate::var::Reduction` は `Mean`／`Sum` の 2 variant だけで、`None` は存在しない。`None` の追加は公開 API の変更なので本 issue では行わない（所有者承認が必要）。両 variant をすべての対象損失で検証し、「none 相当」は要素ごとの二乗誤差（非スカラー損失）の Hessian テストで代替した。
- `L1Loss` は `d == 0` かつ `g` が `±inf` のとき、1 階は `inf·0 = NaN` だが子テープは 0 になる。§21.1 V2 の規則を優先した結果。

### 26.4 テスト

- `crates/autodiff/tests/create_graph.rs`: 5 損失 × `Mean`／`Sum` の Hessian と有限差分の REQ-2 一致（恒等入力・`tanh` 連鎖・MSE の閉形式）、CE・NLL のクラス軸バリエーション（軸 0・最終軸・rank 3 中間軸）、`CE ≡ NLL∘LogSoftmax` の Hessian 突合、pred・target 両方を追跡する混合二階微分、`first_order()` と `Tape::backward` の bit 一致、`jvp_create_graph` の孫テープ再生と方向微分の突合、none 相当、大きな logits（`±1e30`）の有限性、空テンソル、残る損失（`KlDivLoss`）の拒否。
- `crates/autodiff/src/tape.rs`: 5 variant が `true`・隣接する `KlDivLoss` が `false` を固定する単体テスト。
- `crates/facade/tests/create_graph_facade.rs`: 実 CPU バックエンド（融合 `mse_loss_backward`・`nll_loss_backward` 経路）で、1 階との REQ-2 一致と Hessian 列和の中央差分突合。
- 5 損失は V3 でないため、`cg.grad` と 1 階の bit 一致は主張せず REQ-2 で比較する。1 階 `first_order()` が `Tape::backward` と bit 同一である点（§21.1(1)）は構造上保たれる。

### 26.5 変更していないもの

1 階 `Tape::backward`・`grad.rs`・`backward.rs` の値、tolerance・baseline、FMA 契約、依存（`Cargo.toml`・`Cargo.lock`）、`unsafe`、公開 API、`docs/spec`、既存節の本文。CUDA／Metal 実機実測は未実施（#3080 と同じく申し送り）。


## 25. 実装記録（#3083・索引・累積 Op の create_graph 対応）

§21.2 の「索引 V1・累積 V3・選択 V2」のうち、`Gather`・`Scatter`・`Embedding`・`Sort`・`Topk`・`Cummax`・`Cummin`・`Cumsum` の 8 variant を `supports_create_graph` の対象に加えた（網羅 match のまま）。実装は `crates/autodiff/src/create_graph/index_cum.rs`。承認範囲は §20（D-A）の一括承認で、新しい承認は得ていない。

### 25.1 11 variant の扱い

| variant | 本 PR | 方式 | replay（写しの forward） | cgrad（子テープへ積む式） |
|---|---|---|---|---|
| `Gather` | 対象 | V1 | `gather(dim, index)` | `zeros.scatter_add(dim, index, g)` |
| `Scatter`（Add／Overwrite） | 対象 | V1＋V2 | reduce に応じて `scatter_add`／`scatter` | d_src は `gather`（Overwrite は last-writer マスクの `where_cond`）。d_input は Add が恒等、Overwrite が `g.scatter(dim, index, 0)` |
| `Embedding` | 対象 | V1＋V2 | `weight.gather(0, index)` | `scatter_add`（dim 0）の後に `padding_idx` 行を `where_cond` で 0 |
| `Sort`・`Topk` | 対象 | V2 | `gather(dim, index)` | Gather と同じ |
| `Cummax`・`Cummin` | 対象 | V2 | `gather(dim, index)` | Gather と同じ（索引は必ず重複する） |
| `Cumsum` | 対象 | V3 | `cumsum(dim)` | 構造 `flip → cumsum → flip`、値 `cumsum_vjp_along` |
| `IndexedScatterReduce` | fail-closed のまま | — | — | 1 階 VJP がモードごとに入力依存のホストカーネル（prod／amax／amin のタイ規則）で、replayable な `Var` 演算だけの構造グラフを組めない |
| `Cumprod` | fail-closed のまま | — | — | 1 階は `WideFloat` の Horner 型再帰。合成グラフでは §21.1(5) の overflow 契約を保てない |
| `Logcumsumexp` | fail-closed のまま | — | — | 1 階は lane ごとの f64 prefix 再計算。合成すると `exp` の中間 overflow が出るため S1 スケーリングの設計が要り、2h 枠を超える |

- 残る 3 variant は §21.4 の (a)(b) の残りに当たる。起票には所有者承認が要るため本 PR では起票していない。
- 閉包性（§21.1(4)）: `Gather` の cgrad は `Scatter{Add}`、`Scatter` の d_src は `Gather` を積むため、両者は同時に対象にした。`Var::flip` は `index_select` 経由で `Op::Gather` になる。ほかに積む Op は `Where`・`Narrow`・`Contiguous`・`Cumsum` で、いずれも対象。

### 25.2 方式の要点

- 索引系は 1 階と同じ `scatter_with_fallback(Add)` を同じ引数で通るため、`cg.grad` は 1 階と bit 一致する（重複索引は `ScatterReduce::Add` の f64 決定的集約）。
- `Sort`・`Topk`・`Cummax`・`Cummin` の `Op` payload は `descending`／`largest` を持たない。写しは値を選ぶだけの `Var::gather` で再生する（値は親と bit 一致）。§21.1(2) からの逸脱は子テープ上の Op タグが `Gather` になる点だけで、VJP は 1 階と同じ scatter_add(Add) である。
- `Scatter` の d_src は `src_shape[axis] < input_shape[axis]` の非 dim 軸ごとに `g.narrow(axis, 0, len)` してから `gather` する（1 階と同じ）。Overwrite の d_src は `mul` ではなく `where_cond` で選ぶ（`inf*0` を作らない）。
- `ScatterReduce` は `#[non_exhaustive]`。未知 variant は `validate_ancestors` が子テープへ書く前に拒否する（`child.is_empty()` を保つ）。`replay_op`・`build_cgrads` の `_ =>` は多層防御として残す。`Embedding` の `padding_idx` の範囲も入口で再確認する。
- `Cumsum` の値は 1 階と同じホスト関数 `cumsum_vjp_along`（lane ごとの f64 アキュムレータ）。構造 `flip → cumsum → flip` の前方値がホスト値と bit 一致することを単体テストで確認した（`cumsum_structure_value_matches_host_bitwise`。桁落ち列・`[3e38, -3e38, 3e38]` を含む）。そのため孫テープで `Op::Contiguous` が構造を再計算しても §23.3 の P1（f32 へ落ちて inf になる問題）は起きず、`ConvWeightGradHost` 型の専用ホスト値 Op は不要と判断した。空テンソルは 1 階に揃えゼロ定数の葉を返す。

### 25.3 Embedding の padding_idx の二階の意味論（所有者へ戻す論点）

写しを `Gather` として再生するため、二階の逆伝播は forward 経路を通って行 `padding_idx` へ流れる寄与を 0 にしない。これは「1 階勾配関数を有限差分で微分した値」と一致する（forward は行 `p` の値を実際に使い、1 階勾配の行 `p` は常に 0 のため `H[p, *] = 0`、`H[*, p]` は forward 経由で非零）。一方 PyTorch の double backward とは行 `p` の混合二階項で異なりうる。受入条件が有限差分突合のため本方式を採った。PyTorch 準拠へ寄せるかは所有者判断として残す。

### 25.4 テスト

- `crates/autodiff/tests/create_graph.rs`（「索引・累積族」節）:
  - Hessian の有限差分突合（REQ-2 判定・tolerance 不変）: Gather（重複索引・dim 0／1）・Scatter Add（input と src の混合ブロック・非 dim 軸 narrow・重複索引）・Scatter Overwrite（重複索引・最後の書き手）・Embedding（重複 id・`padding_idx` あり／なし・行 `p` と他行が結合する `(Σy)²` 項）・Sort（昇順／降順）・Topk・Cummax・Cummin（重複索引）・Cumsum（軸 0／最終軸／rank 3 中間軸）。loss は y³⊙c で勾配が 2 次多項式のため中央差分が厳密になる。
  - `cg.grad` と `Tape::backward` の突合（線形 loss は bit 一致、連鎖 loss は REQ-2）と `first_order()` の bit 同一。
  - Cumsum の数値回帰: 桁落ち列 `[1, 1e8, -1e8, 1]`（f64 縮約 2 と f32 逐次和 1 が分かれる）と `[3e38, -3e38, 3e38]` で 1 階と `cg.grad` が bit 一致、孫テープ（`jvp_create_graph`）も有限。
  - 孫テープ: `scatter_add → cumsum → gather` を含む関数の `jvp_create_graph` を独立な方向微分と突合。
  - fail-closed: `create_graph_rejects_custom_downstream_of_gather`・`create_graph_rejects_unsupported_op_cumprod`・`create_graph_rejects_gather_mixed_with_unsupported_op_and_keeps_child_empty`（いずれも拒否時 `child.is_empty()`）。旧 `create_graph_rejects_unsupported_op_gather` は廃止（Gather が対象になったため）。`Logcumsumexp`・`IndexedScatterReduce` の既存拒否テストは変更なしで green。
- 内部単体テスト: `tape.rs::index_cumulative_family_supports_create_graph`（8 件が true・3 件が false）、`index_cum.rs` の構造値 bit 一致と `replay_scatter`。

### 25.5 変更していないもの

1 階 `Tape::backward` の値（`grad.rs` の差分は `cumsum_vjp_along`・`scatter_overwrite_last_writer_mask` の可視性を `pub(crate)` にしただけ）・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文。CUDA／Metal 実機での create_graph×索引系の実測は未実施（#3168・#3169 へ申し送り）。

## 27. 実装記録（#3078・§21.3「#3078 LayerNorm・RmsNorm」の実装）

- 親は #3075、方針の正は §21.1（V3・S1・閉包性）と §21.3。本節は実装した形と、§21.3 から外れた点だけを書く。
- 変更ファイル: `crates/autodiff/src/create_graph_norm.rs`（新規）・`create_graph.rs`・`tape.rs`・`grad.rs`（`rmsnorm_vjp_rows`／`layer_norm_vjp_rows` を `pub(crate)` にしただけ。本体・数値は不変で 1 階は bit 同一）・`lib.rs`。テストは `crates/autodiff/tests/create_graph_norm.rs`。

### 27.1 実装形

- `supports_create_graph` は `Op::RmsNorm`・`Op::LayerNorm` を `true` にした（網羅 match のまま。ワイルドカードは足していない）。`replay_op` は `Var::rms_norm`／`layer_norm` へ委譲し、`build_cgrads` は `create_graph_norm::{rms_norm_cgrads, layer_norm_cgrads}` の寄与列を `accumulate` する。
- **値（V3）**: `dx`・`dw`・`db` の値は 1 階と同じ `*_vjp_rows`。`dy` は子テープ上の `g` の実値（`reduce_bias_grad_var` と同じ借用パターン）。CPU・`naive_ops` では 1 階 VJP も同じ関数へフォールバックするので、`g` が親の upstream と bit 一致する範囲では `cg.grad` は 1 階と bit 同一。CUDA は 1 階が専用カーネルを通るため、REQ-2 統一複合判定の範囲の一致のみを主張する。`db` も `reduce_bias_grad_var` ではなく V3 で書く（`rows == 1` の符号付きゼロの扱いを 1 階と揃えるため）。
- **構造（S1）**: 行ごとの 2 のべき乗 `s`（`max(最大絶対値, √eps)` を切り上げ。`eps=0` は最大絶対値のみ・0 なら `s=1`。指数は `[-126, 127]` にクランプ）で `x/s` を作り、`ms`・`rstd_s=1/√(ms+eps/s²)`・`xhat=xs·rstd_s`・`dx=(rstd_s/s)·(dxhat−…)` を合成する。`1/s` の乗算は `s=2^127` で非正規化数になるため `Div` を使う。`eps/s²` は f64 で計算して f32 へ丸める。出力ごとに `Op::Contiguous`（VJP は恒等）のノードを `push_eager` し、値だけをホスト関数の結果にする。
- **`LayerNorm` の中心化**: `x_ref=Narrow(x,1,0,1)`・`d=x−x_ref`・`m=Mean(d)`・`dev=d−m`・補正 `dev2=dev−Mean(dev)`。`s` は中心化前の `x` ではなく `dev2` の子テープ上の実値から決める。
- **積む Op**: `Reshape`・`Contiguous`・`Narrow`・`Mean`・`Sum`・`Mul`・`Add`・`ScalarBinary{Sub, Div}`・`ScalarUnary{Sqrt}`。すべて `supports_create_graph == true` なので孫テープ再生（`backward_create_graph` の再帰・`jvp_create_graph`）が通る。`Rsqrt`／`Reciprocal`／`Op::Var`・`BroadcastTo` は使っていない。

### 27.2 §21.3 より狭い拒否（所有者へ戻す論点）

§21.3 は「`d=x−x_ref` が非有限、または `rstd_scaled` が非有限になる入力は S2 へ切り替える」とした。S2（ホストの統計定数と二階の項の手書き合成）は 1 イシューの粒度を超え、誤ると Hessian が黙って誤値になるため、本実装は S2 を作らず、**子テープへ書く前**の `validate_ancestors`（`create_graph_norm::validate_norm_s1`）で次の入力を `Err(AutodiffError::Backward)` として拒否する。拒否時に `child` は空のまま保たれる。**これは新しい承認ではなく、S2 または代替の採否を所有者へ戻す。**

| Op | 拒否条件（親テープの `x` の値だけで決まる保守的な判定） |
|---|---|
| `LayerNorm` | (a) 非有限要素 (b) f32 の `x_i−x_0` が非有限 (c) f64 で `hidden·max｜x_i−x_0｜ ≥ 2^125`（`Mean` の f32 部分和が overflow しうる。例 `[-2e38,1e38,1e38]`）(d) `eps==0` かつ `max｜x_i−x_0｜ < 2^-100`（定数行・極小偏差行で `rstd=inf`） |
| `RmsNorm` | (a) 非有限要素 (d') `eps==0` かつ全ゼロ行 |

§21.3 の必須ベクトル（`x=1e20`・`[16777216,16777218]`・`[1e-30,1e-30]`/`eps=1e-5`・`[1e30,1e30]`/`eps=1e-5`・全ゼロ行）はいずれもこの判定を通り、1 階・`cg.grad`・孫テープ再生が有限かつ REQ-2 判定で一致する（テストで固定）。将来の代替案として「S0 前スケーリング（中心化の前に `x` を行ごとの 2 のべき乗で割る）」も候補に残すが実装していない。

### 27.3 総括的な適用判定と数値契約の保証範囲

- **upstream のスケーリング（構造の形）**: `dxhat = g·w` を行ごとの 2 のべき乗 `gs`（`g·w` の f64 行最大絶対値から決める）で割ってから `Sum` へ渡し、縮約の**直後**に `gs/hidden`（定数葉）を掛けて `mean_dxhat·gs`・`mean_dot·gs` を作る。`dx` の構造は `rstd/s · (g·w − mean_dxhat·gs − xhat·mean_dot·gs)` で、`gs` を縮約の手前（逆伝播では縮約の後ろ）で復元しない。
- **個別条件では収束しなかった経緯**: PR #3346 のレビューでは、有限な入力・upstream で構造（`cg.grad`・二階の逆伝播・孫テープ再生）の中間値が f32 を超える指摘が同じ類型で続いた。`g=[3e38,3e38]`（RmsNorm `x=[1,1]`）の縮約、LayerNorm `x=[1,-1,1,-1]`・`g=±1e38` の `gs` 復元と `hidden·g` 規模の cotangent 和（HVP が NaN。真値は約 ±3e33）、低分散行の `rstd` 増幅（Bugbot）、RmsNorm `x=[0.75×4]`・追跡ありの `w=0`・`g=8e37` で `g·w=0` の検査を素通りし、dw 構造の `rstd_s` への cotangent（約 2.4e38）に `Div` の分母側係数（約 −1.78）が掛かって `−inf` になる経路（真の混合二階微分は各約 1.9e33）。経路ごとに条件を足す方式では網羅できないため、次の総括的な判定へ置き換えた。
- **総括的な適用判定（`create_graph_norm::check_structure_stages`）**: `rms_norm_cgrads`／`layer_norm_cgrads` が構造を積む前に、ホストが構造の各段を f64 で再計算し、各段の絶対値の上界を求める。対象は (i) dx／dw／db 構造の forward の各段（統計量・`dxhat`・行方向和・`inner`・構造の出力・dw／db の行をまたぐ和）、(ii) 二階の逆伝播で各段に現れる「cotangent × 微分係数」と行方向の部分和の上界、(iii) 境界ノード（`x`・`w`・`g` の写しのうち勾配を受け取るもの）へ出る cotangent。(ii) は `cg.grad` の出力へ入る seed の各要素の絶対値が 1 以下という前提で、構造の演算順（`binary_partials` の `Div` 係数 `-(a/b)/b`・`Sqrt` の `0.5/y`・`Mean` の `1/hidden`）を逐語でなぞり、三角不等式で押さえる。LayerNorm の `s`・`xs` は子テープ上の `dev2` の実値から取り、構造と同じ値で判定する。どれか 1 段でも上界が `2^125` 以上（または非有限）、または `g` に非有限要素があれば `Err(AutodiffError::Backward)` で拒否し、`Tape::backward_create_graph` のロールバックが `child` を空へ戻す（同じ `child` で再試行できる）。閾値 `2^125` は入口検査（`MEAN_OVERFLOW_GUARD`）と同値で、f32 最大値との 8 倍の差は f32 丸めの累積への余裕である。上に挙げた経路は既知の失敗例であり、網羅を主張するのはこの判定の方である。
- **保証範囲**: 本節の数値契約が保証するのは、判定を通った入力で 1 階・`cg.grad`・二階（seed の各要素の絶対値 ≤ 1）・孫テープ再生が有限であり、`cg.grad` が 1 階と REQ-2 判定で一致し、孫テープ再生の二階が子テープの二階と REQ-2 判定で一致することまでである。孫テープ側で子テープ上の正規化の写しが再び微分される場合（例: `g = 2c·norm` のように upstream が入力に依存する損失）は、そこでも同じ判定が走り、通らなければ同じく型付きエラーで孫テープは空へ戻る。上界は符号を見ない最悪値なので拒否は保守的で、拒否は実際の overflow を意味しない。構造の forward 出力も判定するため、1 階の値が有限でも `2^125` 以上の入力は拒否する（孫テープ再生では `Op::Contiguous` が構造の f32 値を再計算するため）。対象外: `g` を作る上流の Op の逆伝播（例: `g = 2c·norm` の `Mul` が cotangent に `2c` を掛ける段。各 Op の契約に従う）、絶対値 1 を超える seed、三階以上の微分、真の値そのものが f32 を超える場合（1 階と同じく `inf`）。
- 1 階側も要素積 `dy·w`・`dy·xhat` は f32 で計算するので、要素単位の overflow は 1 階と同じ挙動になる。
- 拒否範囲を狭める（S2 化）かは所有者判断（§27.2 と同じく所有者へ戻す論点）。
- CUDA／Metal 実機での `cg.grad` の parity は未確認（申し送り）。

### 27.4 テスト

`crates/autodiff/tests/create_graph_norm.rs`: Hessian の有限差分突合（RmsNorm の weight なし／あり・LayerNorm の weight・bias 4 通り・rank 1／2／3・`x`／`w`／`b` 各微分・`x`×`w` の混合項・tanh 合成）、`first_order` の bit 同一と `cg.grad` の REQ-2 一致、§21.3 必須ベクトルの 3 段（1 階・`cg.grad`・孫テープ再生）、`eps=0` の非退化行の受理、§27.2 の拒否と `child.is_empty()`、fail-closed のまま残る隣接 Op（`LocalResponseNorm`）。有限差分は測定側の丸め雑音を抑えるため 5 点中央差分（h=1e-2）を使う（判定式・tolerance は不変）。§27.3 の判定は正のプローブで確かめる: レビューで指摘された入力そのもの（と LayerNorm の dw 同型・`[3,4]` の db 行をまたぐ和）が拒否され `child` が空になること、同じ入力を upstream 1e30 倍へ下げると受理され全段が有限かつ一致すること、kind × weight（なし・0・非ゼロ・定数）× bias × 微分対象 `x`／`w`／`b` × 幅 4／32 × 行数 1／3 × upstream の大きさ（1〜3e38）× 線形／入力依存（`g = 2c·norm`）の損失 × seed の符号パターンの掛け合わせで「受理なら全段が有限で一致・拒否なら `child` が空」が成り立ち、受理と拒否の両方が現れること。

### 27.5 変更していないもの

1 階 `Tape::backward` の数値、tolerance・baseline、FMA 契約、依存、`unsafe`、`docs/spec`、既存節の本文（§21.1 (5) の S1 と §21.3 の #3078 項に、保証を §27.3 の判定を通った入力に限る旨を足した以外）。

## 28. 実装記録（#3079・`BatchNorm` の create_graph 対応。親 #3075）

- 親は #3075、方針の正は §21.1（V3・S1・閉包性）と §21.3（#3079 行）。数値契約の考え方は §27.3（総括的な適用判定）をそのまま使い、BatchNorm 用に判定を拡張した。本節は実装した形と、§21.3・§27 から外れた点だけを書く。
- 変更ファイル: `crates/autodiff/src/create_graph_norm.rs`（#3078 のモジュールへ BatchNorm を追加）・`create_graph.rs`（`validate_ancestors` の空入力拒否・`replay_op`・`build_cgrads` の薄い腕）・`tape.rs`（`supports_create_graph` で `Op::BatchNorm` を `true`・`Op::BatchNormGradHost` の追加）・`grad.rs`（`batch_norm_vjp_channels` を `pub(crate)` にしただけ。本体・数値は不変で 1 階は bit 同一）。テストは `crates/autodiff/tests/create_graph.rs`（BatchNorm 節）と `crates/autodiff/tests/create_graph_batch_norm.rs`（判定の正のプローブ・掃引）。

### 28.1 方式の割り当て（出力ごと）

| 出力 | train | eval | 理由 |
|---|---|---|---|
| dx | V3 | V3 | 値は 1 階と同じ `batch_norm_vjp_channels`。eval の構造は定数統計との要素積だけ（§21.3「推論時は V1」の構造に相当）だが、値は f64 の `rstd` を使う 1 階と bit で揃えるためホスト関数で与える |
| dw・db | V3 | V3 | どちらのモードも 1 階は長軸縮約を f64 で蓄積する（§21.1 (5)「f64 縮約は原則 V3」） |

- 値は `Op::Contiguous` ではなく **`Op::BatchNormGradHost`**（`structure`・`g`・`x`・`weight` と `fixed_stats` を持ち、VJP は `structure` への恒等）に載せる。`replay_op` は同じホスト関数で値を再計算するため、構造側の f32 縮約とホストの f64 縮約で丸めが異なる入力（例 `g=[2^24,1,1]` の db は f32 逐次和が `2^24`、f64 縮約が `2^24+2`）でも、孫テープ再生の値が 1 階と bit 同一のまま保たれる（PR #3348 の Codex P1「孫テープへの再生でもホスト補正値を保持する」への対応。§27 の `Op::Contiguous` 方式より強い）。
- `replay_op` は `fixed_stats` payload をそのまま使い、`Var::batch_norm`／`batch_norm_infer` へ委譲する。

### 28.2 構造（S1。§27.1 と同じ構造をチャネル行へ）

チャネル軸を前に出した行 `[c, m = n·spatial]`（`Reshape`・`Permute`・`Contiguous`）を行として組む。BatchNorm 専用の別方式は持たない。

- **train**: `LayerNorm` と同じ構造（§27.1）。安定な中心化（`x_ref`・`d`・`dev`・`dev2`）、`dev2` の実値から決める行スケール `s`、`rstd_s = 1/√(mean((dev2/s)²)+eps/s²)`・`xhat`、dx は `gs` で割ってから縮約し縮約の直後に `gs/m` を掛ける `(rstd_s/s)·(g·w − mean_dxhat·gs − xhat·mean_dot·gs)`。違いは `w`・`b` が行ごと（`[c]`。`g·w` は `[c,1]` の broadcast）で、dw＝`Σ_m g·xhat`・db＝`Σ_m g` が行方向の `Sum(1)` であることだけ。構造の組み立ては `LayerNorm`／`RmsNorm` と共通の関数（`stable_centering`・`scaled_stats`・`dx_rows`）を使う。
- **eval**: 統計量が定数なので `xhat = (x − mean)·rstd`（`mean`・`rstd` は定数葉。`rstd` は f64 で求めて f32 へ丸めた値）、dx の構造は `(g·w)·rstd`、dw・db は train と同じ行方向和。
- 積む Op は `Reshape`・`Permute`・`Contiguous`・`Narrow`・`Mean`・`Sum`・`Mul`・`Add`・`Sub`・`ScalarBinary{Div}`・`ScalarUnary{Sqrt}`・`BatchNormGradHost` のみで、すべて `supports_create_graph == true`（閉包性。孫テープ再生で閉じる）。

### 28.3 総括的な適用判定の拡張（§27.3 の `check_structure_stages` を共用）

- PR #3348 のレビューでは、前置スケール方式（`x` を `2^p` で割ってから中心化・`s_g` で割ってから縮約・`±100` 段の 2 のべき乗で復元）に対し、有限な入力・upstream で構造（`cg.grad`・二階の逆伝播・孫テープ再生）の中間値が f32 を超える指摘が同じ類型で続いた（Codex P1 × 8・Cursor Bugbot 1 件。§28.5 の表）。`p`・`s_g` の上限・下限を経路ごとに足す方式では、逆伝播の倍率・中心化の差・平均の縮約のどこかで再発したため、前置スケールを撤去し、#3078 の総括的な適用判定（§27.3）へ載せ替えた。
- `create_graph_norm::check_structure_stages` に 2 つの区分を足した（判定の段・閾値 `2^125`〈`STAGE_BOUND_GUARD`。値は不変〉・「seed の各要素の絶対値 ≤ 1」の前提は §27.3 と同じ）:
  - **`ParamAxis`**（`Column`＝`LayerNorm`／`RmsNorm`、`Row`＝`BatchNorm`）: `w` の添字と、dw／db 構造の縮約・`w` への cotangent の蓄積を列ごと（行をまたぐ和）か行ごと（行方向の和）かで切り替える。三角不等式の積み方は同じ。BatchNorm train は `Row` として、`LayerNorm` と同じ forward の各段・二階の逆伝播の各段（中心化の 3 段の `Mean` を含む）を判定する。
  - **`FixedStats`**（BatchNorm eval）: `rstd`・`x − mean`・`xhat`・`g·w`・dx 構造の出力（`g·w·rstd`）・`g·xhat`・dw／db の縮約と、二階の逆伝播で `g`（`|w|·rstd`・`|xhat|`・1）・`w`（`Σ|g|·rstd`）・`x`（`|g|·rstd`）へ出る cotangent の上界を判定する。
- 入口検査（`validate_norm_s1`）に BatchNorm の腕を足した: train はチャネル行ごとに `LayerNorm` と同じ (a) 非有限要素 (b) f32 の `x_i − x_0` が非有限 (c) f64 で `m·max|x_i − x_0| ≥ 2^125`（`Mean` の f32 部分和）(d) `eps == 0` かつ `max|x_i − x_0| < 2^-100`。eval は `x`・`mean`・`var` の非有限要素。形状の積は取る前にゼロ軸を判定し（空入力は `validate_ancestors` が拒否）、`n·spatial` は `checked_mul` で型付きエラーにする。
- **保証範囲**（§27.3 と同じ書き方）: 判定を通った入力で 1 階・`cg.grad`・二階（seed の各要素の絶対値 ≤ 1）・孫テープ再生が有限で、`cg.grad` は 1 階と REQ-2 判定で一致する。拒否は `Err(AutodiffError::Backward)` で、`backward_create_graph` のロールバックにより `child` は空へ戻る。上界は符号を見ない最悪値なので拒否は保守的。対象外は §27.3 と同じ（`g` を作る上流の Op の逆伝播、絶対値 1 を超える seed、三階以上、真の値そのものが f32 を超える場合。eval の `xhat` は固定統計からの距離で有界でないため、`x` が固定平均から遠い入力の入力依存損失では真の二階微分自体が f32 を超えうる）。
- **f32 精度の限界（overflow とは別）**: train の dw は正規化と同じ行で縮約するため、入力依存の損失（`g = 2c·bn`）では `∂dw/∂x` が `xhat` のスケール不変性（`Σxhat = 0`・`Σxhat² ≈ m`）で打ち消し合う `c` 規模の項の残差になり、真値が `c·eps/var` 程度まで小さくなりうる。この桁落ちでは f32 の丸め誤差が真値を上回りうるので、子テープ（`grad.rs` の VJP）と孫テープ（子テープ上の `Var` 合成）の二階の値は REQ-2 判定で一致しないことがある（両方とも有限）。テストではこの組み合わせの x への二階のみ有限性を確かめ、一致は問わない。
- 拒否範囲を狭める（S2 化・前置スケールの再導入）かは所有者判断（§27.2 と同じく所有者へ戻す論点）。CUDA／Metal 実機での `cg.grad` の parity は未確認（申し送り。#3168・#3169）。

### 28.4 running stats・fail-closed

- `replay_op`・`build_cgrads` は `Var::` プリミティブとホスト関数だけを呼び、`nn::BatchNorm*` の buffer・`num_batches_tracked` へ触れる経路がない。子テープ構築・二階 backward・孫テープ再生（`jvp_create_graph`）で二重更新されないことを統合テストで固定した。
- 空テンソル（`n`／`c`／`spatial` のいずれかが 0）は `validate_ancestors` で子テープへ書く前に拒否する（形状の積ではなく `shape.contains(&0)` で判定。`[usize::MAX, 2, 0]` で積が overflow して panic するため）。
- 対象外のまま残る Op（例: `local_response_norm`）は従来どおり入口で拒否され、子テープは無変更。

### 28.5 レビュー指摘（PR #3348）と判定後の扱い

| 指摘 | 入力（要旨） | 判定後 |
|---|---|---|
| Codex 1 | eval `x=mean=3e38`・`var=0`・追跡 `w=1`・`loss=sum(y)` の `∂dw/∂x`（train 側の同型も） | 受理。`x − mean = 0` で全段が安全域。`∂dw/∂x ≈ 316.23` を突合 |
| Codex 2 | 空入力 `[usize::MAX, 2, 0]` の形状積 | `contains(&0)` で拒否（panic しない）。`n·spatial` は `checked_mul` |
| Codex 3 | train `x=[0,0]`・`g=1e38`・`cg.grad(x)` の二階 | 拒否（dx 構造の `Σdxhat` への cotangent `k·gs` が `2^125` 超）。`g=1e30` は受理で Hessian 0 と一致 |
| Codex 4 | eval `x=mean=3e38`・`g=1e20` の `∂dw/∂x` | 受理。`∂dw/∂x ≈ 3.16e22` を突合 |
| Codex 5 | train `x=[3e38,-3e38]`・`g=1e38` | 拒否（入口 (b)：`x_i − x_0 = -inf`） |
| Codex 6 | train `x=[0,0]`・`g=1e38`・`cg.grad(w)` の二階 | 拒否（`xhat` への cotangent `|g|` が `2^125` 超）。`g=1e30` は受理で 0 と一致 |
| Codex 7 | eval `[3,1]`・追跡 `b`・`loss=sum(y²·[1e38,1e38,-1e38])`（孫テープでホスト補正値が失われる） | 拒否（db 構造の縮約の上界）。判定を通る入力で孫テープの `db` がホスト値と bit 同一であることは別テストで固定（§28.1） |
| Codex 8 | train `[3,1]`・`x=[3e38,-3e38,-3e38]`・`g=1e38`（中心化の平均の縮約） | 拒否（入口 (b)） |
| Cursor Bugbot | 同符号の巨大入力で前置スケールの下限が余計な倍率を足す | 前置スケールを撤去。`x=[3e38,2.9e38]` の `∂dw/∂x` は受理で有限 |

### 28.6 テスト

- `crates/autodiff/tests/create_graph.rs`（BatchNorm 節）: Hessian の有限差分突合（rank 2・train／eval・affine の有無・`x`／`w`／`b`・`x`×`w` の混合項）と rank 2／4 の独立合成との突合、1 階一致（`first_order()` は bit 同一・`cg.grad` は REQ-2）、running stats 不変、閉包性（孫テープへの再生）、`local_response_norm` の fail-closed、空入力の拒否、回帰ベクトルの 3 段（受理側: `x=1e20` 系・中心化の丸め・小スケール・定数・全ゼロ・`3e38` の定数行・上流 1e30 版。拒否側: `[3e38,-3e38]`・巨大な上流勾配・eval の逆符号大値。拒否側は `child` が空）。
- `crates/autodiff/tests/create_graph_batch_norm.rs`: §28.5 の全例を正のプローブ（受理なら 1 階・`cg.grad`・二階・孫テープ再生が有限かつ一致、拒否なら `Err(Backward)` で `child` が空）で固定し、拒否例を 1e30 規模へ下げると受理されること、eval の `∂dw/∂x` の値、判定を通る入力で孫テープの `db` がホスト値と bit 同一であること、拒否後の同じ `child` での再試行を確かめる。掃引は train／eval（通常分散・分散 0）× rank 2／3／4 × weight なし／あり（0 と非ゼロ）／定数・bias × チャネル値（定数・交互符号・低分散・小スケール・1e20）× upstream の大きさ（1〜3e38）× 線形／入力依存の損失 × upstream と seed の符号パターンで、受理と拒否の両方が現れることも確かめる。判定（`stage`）を一時的に常に通すと、レビュー例・掃引・再試行の 3 テストが落ちることを確認した（デバッグビルドで約 16 秒）。
- §28.3 の f32 精度の限界（train・入力依存損失・dw の x への二階）と、eval の入力依存損失で真の二階微分が f32 を超える入力は、掃引で有限性のみ確かめる／除外する（いずれも判定の対象外で、tolerance は変更していない）。

### 28.7 変えていないもの

1 階 `Tape::backward` の結果（bit 同一）・tolerance・baseline・`STAGE_BOUND_GUARD` 等の閾値・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文（`LayerNorm`／`RmsNorm` の構造は共通関数へ切り出しただけで、組む Op と値は不変。子テープへ積むノードの順序が一部入れ替わるのみ）。facade の `hvp`／`jacfwd` の総合確認は #3087、実機（CUDA／Metal）の parity は #3168・#3169 の範囲。

## 29. 実装記録（#3084・`Pad`・`PadMode`・`MaskedFill` の create_graph 対応。親 #3076）

### 29.1 対象と方式

§21.2 の B 表どおり 3 variant を `supports_create_graph() == true` にした。実装は `crates/autodiff/src/create_graph/pad_mask.rs`（cgrad）と `create_graph.rs`（replay・分岐）。

| Op | replay（写しの forward） | cgrad（子テープへ積む式） | 方式 | 積む Op |
|---|---|---|---|---|
| `Pad` | `mirror(input).pad(&pads, value)` | 各軸へ `narrow(axis, before, input_len)` を連鎖（1 階と同じ順序） | V1 | `Narrow` |
| `MaskedFill` | `mirror(input).masked_fill(mask_bool, value)` | `where_cond(mask, 0, g)`（`mul` は `inf*0=NaN` のため使わない） | V1／V2 | `Where` |
| `PadMode` | `pad_with_mode(&mirror(input), &pads, mode)` | `g` の平坦化 → `zeros.scatter_add(0, idx, ·)` → 入力 shape へ reshape。値は 1 階と同じ `pad_modes_vjp_host`（`f64` 縮約・1 回 downcast）で `push_v3` により結合 | V3 | `Reshape`・`Contiguous`・`Scatter` |

- `PadMode` を V3 にした理由: §21.2 は族の方式を V1／V2 としているが、`PadMode` の 1 階 VJP は添字重複を `f64` で縮約する契約（`docs/autodiff-pad-modes-decision.md`）を持つ。§21.1(5) の原則（`f64` 縮約は V3）に従い、バックエンドの `scatter` 実装に依らず値が 1 階と bit 一致することを構造上保証する。
- 添字表 `idx`（出力の行優先順の入力平坦添字）は tensor-core に追加した `pad_modes::pad_modes_source_indices` から得る。forward・VJP と同じ内部走査（`walk_sources`）を共有し、写像を autodiff 側へ複製しない。内部クレートへの非破壊の追加で、facade の pad_modes 保留ガード（`api_surface.rs`）の対象外。`usize → i32` は `try_from` で行い、失敗は型付き `Err`（panic・wrap しない）。空テンソル（`in_numel == 0`／`out_numel == 0`）は 1 階に揃えゼロ定数を返す。

### 29.2 payload の変更

`Op::Pad` と `Op::MaskedFill` に `value: f32` を追加した（§21.5）。子テープの forward 再生（replay）専用で、VJP は使わない。`Op` は `pub(crate)` のため facade の公開面は変わらない。`grad.rs` の差分は束縛の `..` とコメントのみで、1 階の値は不変（テストで bit 同一を固定）。

### 29.3 閉包性

積む Op（`Narrow`・`Where`・`Reshape`・`Contiguous`・`Scatter`）はすべて `supports_create_graph == true`。孫テープで再生すると `PadMode` の V3 ノード（`Contiguous`）は構造（`scatter_add`）を再計算する。これは §21.1(3) の受け入れ範囲内で、`jvp_create_graph` のテストで確認した。

### 29.4 テスト

- `crates/autodiff/tests/create_graph.rs`「Pad・PadMode・MaskedFill 族」節: `value ≠ 0` の Hessian と `value = 0` との対照、pad の連続適用、broadcast する mask、attention 型（`-1e4` → softmax）、`value = -inf` で勾配が有限かつマスク位置が 0、3 モードの Hessian（reflect／replicate〈pad が軸長超〉／circular）、1 階 bit 一致（線形 loss）と REQ-2 判定（連鎖 loss）、桁落ち入力（`[1,1e8,-1e8,1]`）での 1 階 bit 一致、`jvp_create_graph`（孫テープ）、空テンソル・恒等形、`cumprod` との混在拒否（`Err(Backward)`・`child` 空）。判定は既存の `finite_diff_hessian`・`assert_hessian_close`・`req2_close` を使い、tolerance は追加・変更していない。
- fail-closed の差し替え: 旧 `create_graph_rejects_unsupported_op_masked_fill` は廃止し、fail-closed のまま残る隣接 Op（`cumprod`）との混在拒否テストへ置換した（単独の `cumprod` 拒否は既存 `create_graph_rejects_unsupported_op_cumprod`）。
- `crates/autodiff/tests/pad_modes_parity.rs`: 旧 `create_graph_is_a_typed_error_not_a_panic` を `create_graph_succeeds_and_matches_first_order`（3 モードで `cg.grad` が 1 階と bit 一致）へ置換した。
- 単体: `tape.rs::pad_mask_family_supports_create_graph`、`pad_mask.rs` の構造値テスト、tensor-core の `pad_modes_source_indices` テスト（`pad_modes_host` に iota を与えた結果との一致）。

### 29.5 変えていないもの・申し送り

1 階 `Tape::backward` の値（bit 同一）・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・checkpoint の適格性（`Pad`・`PadMode`・`MaskedFill` は非適格のまま）・facade の公開面。承認範囲は §20（D-A）の一括承認であり、新しい承認は得ていない。CUDA／Metal 実機での create_graph×pad 系の実測は未実施（申し送り先の Issue 起票は所有者承認事項のため本変更では起票していない）。`Dropout`・`Interpolate`・`MaxUnpool` など §21.2 C 群、`PadMode` の activation checkpoint 対応、`f64` 自動微分経路は対象外のまま。

## 30. 実装記録（#3085・プーリング Op 群の create_graph 対応。親 #3076）

§21.2・§21.3 の「#3085 プーリング」の実装記録。コードは `crates/autodiff/src/create_graph/pool.rs`、`create_graph.rs` の `replay_op`／`build_cgrads`、`tape.rs` の `Op::supports_create_graph`。

### 30.1 対象にしたもの・残したもの

| variant | 方式 | forward 再生 | 子テープへ積む Op | 値の扱い |
|---|---|---|---|---|
| `MaxPool2d`（`adaptive_max_pool2d` が記録するものを含む） | V1（記録済み `index` の固定） | `contiguous → reshape [N·C, H·W] → gather(1, index) → reshape` | `Contiguous`・`Reshape`・`Gather`・`Scatter` | 1 階と同じ `scatter_with_fallback(Add)`（`f64` 決定的集約）で bit 一致 |
| `MaxPool3d` | 同上（`[N·C, D·H·W]`） | 同上 | 同上 | 同上 |
| `AvgPool2d` | V3 | `avg_pool2d`（`ceil_mode=false`） | `Contiguous`・`Reshape`・`Mul`・`Gather`・`Scatter` | 1 階と同じ `avg_pool2d_vjp` を `push_v3` で登録 |
| `AvgPool3d` | V3 | `avg_pool3d`（`ceil_mode=false`） | 同上 | `grad::avg_pool3d_vjp`（1 階の腕から抜き出し。式・評価順は不変） |
| `AdaptiveAvgPool2d` | V3 | `adaptive_avg_pool2d`（出力サイズはノード shape から） | 同上 | 1 階と同じ `adaptive_avg_pool2d_vjp` |

残したもの（fail-closed のまま）: `MaxUnpool`（§21.2 C 区分）。`Conv3d`／`ConvTranspose3d` は #3080(b)。

### 30.2 §21.2・§21.3 からの差分

- **Max 系は V2 ではなく V1 になった**: argmax をホストで再計算せず、forward が確定した `index`（先勝ちタイ・NaN 規則込み）をそのまま使う。タイ規則は構造上 1 階と一致する。`index` が入力空間の範囲外なら子テープへ書く前に型付きエラーにする。
- **Avg 系の構造は `Fold` 経由ではなく索引の合成に統一した**: Adaptive は窓の大きさが可変（`H % Ho != 0` では固定の kernel/stride にならない）、3D には fold が無いため、§21.3 の「`Fold`（col2im）経由」は `AvgPool2d` にしか当てはまらない。3 variant を同一の構造（`g` を `[N·C, 出力空間]` へ平坦化 → 窓の分類ごとに `gather(1, 出力位置)` → `1/s` → `zeros.scatter_add(1, 入力位置)` → `s/d` → 入力 shape へ）に揃えた。窓の列挙は 1 階と同じ情報源（`pool_window_input_pos`・`adaptive_window`・`pool3d_layout` の窓規約）から作る。
- 索引テンソルは `[N·C, E]`（`E` は 1 平面の窓要素の総数）。`i32` の範囲と `N·C·E` の桁あふれは確保前に型付きエラーで拒否する。

### 30.3 数値契約

- 値は 1 階のホスト関数（`f64` アキュムレータ・1 回 downcast）を `push_v3` で登録するため `cg.grad` は 1 階と bit 一致する。構造側は窓を divisor `d` ごとに `gather` → `1/s` → `scatter_add`（`f64` 集約）→ `s/d` で組む（`s` は `d` 以下の最大の 2 のべき乗。重み `s/d ∈ (0.5, 1]`。codex-review 指摘・PR #3352）。(a) 除算の前に窓内を `f64` で集約するため相殺する上流でも丸め誤差が残らない。(b) 集約前に厳密な `1/s` を掛けて窓内和を `|g|` 程度に抑える。(c) 重みを 1 以下にするため、二階の逆伝播で seed を増幅する乗算が無く（`d=3` で `seed=3e38` が `inf` になる旧構造を解消）、`gather` の VJP が集約する値は `|seed|·m/d ≤ |seed|` に収まる（§21.1(5)）。(d) divisor が異なる分類（`count_include_pad=false` の境界窓・Adaptive の不均一窓）は部分和を f32 で丸めてから足すため、桁違いの打ち消しで孫テープ再生値が 1 階値から外れうる。構造を組む前にホストで構造を模擬し（`check_structure_contract`）、REQ-2 複合判定を外れる入力・集約前中間和が f32 範囲を超える入力は型付きエラーで拒否する（§27.3 と同型の fail-closed。保証は判定を通った入力に限る。同一 `g` の再生を前提とし、tolerance は不変）。
- tolerance・baseline・FMA 契約・依存・`unsafe` は変更していない。

### 30.4 テスト

- `crates/autodiff/tests/create_graph_pool.rs`: Hessian と有限差分の突合（平均系は重なり・padding・`count_include_pad`、Adaptive は 4→2・5→3、最大系はタイなしの入力・重なり・padding・dilation）、先勝ちタイの解析曲率と重複索引の 1 階 bit 一致（`first_order()` も）、平均系の値が打ち消し入力で 1 階と bit 一致すること、3e38 の上流での 1 階・`cg.grad`・二階の有限性、`jvp_create_graph`（孫テープ）が独立な参照と一致すること、`MaxUnpool` の fail-closed と `child` が空のままであること。
- `create_graph/pool.rs` の単体テスト: 構造値が 1 階のホスト関数と REQ-2 で一致、Max の gather 再生が forward と bit 一致（NaN・タイ・負値）、範囲外 `index` が型付きエラー。
- `tests/pool3d_parity.rs`: 旧 `create_graph_is_a_typed_error_not_a_panic` を、成功して 1 階勾配と bit 一致する確認へ反転した。`tape.rs` に `pooling_family_supports_create_graph` を追加。
- f32 overflow を引き起こす平均系の打ち消し入力は divisor があるため構成できなかった。構成できない回帰ベクトルは主張しない（代わりに大振幅の上流での有限性を確認）。

### 30.5 所有者へ戻す論点・申し送り

- CUDA／Metal 実機での create_graph とプーリングを組み合わせた parity 実測は未実施（#3168・#3169 系への申し送り）。
- §21.4 のサブイシュー（(a) Max 系／(b) Avg 系）は起票されていなかったため、5 variant を本イシューで扱った。

### 30.6 変えていないもの

1 階 `Tape::backward` の結果（bit 同一）・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・公開 API（`api_surface`）。

## 31. 実装記録（#3086・線形代数 Op と FFT の create_graph 対応の可否と対応分。親 #3076）

### 31.1 23 件の判断

§21.2 を正とした（#3086 の issue は `Stft`・`Istft` を実装側に置く案だったが、§21.2 の理由が成立するため fail-closed にした）。

| Op | 判断 | 方式 | 理由・子テープへ積む Op |
|---|---|---|---|
| `Fft`・`Ifft` | 実装 | V1 | 随伴は逆向きの FFT（`norm` は `Backward <-> Forward` を入れ替え、`Ortho` は不変）に、実軸の `narrow`／ゼロ定数との `concat` を足すだけで書ける。積む Op: `Ifft`／`Fft`・`Narrow`・`Concat` |
| `Rfft` | 実装 | V1 | `irfft(g * h, n, dim, swap(norm))`。`h` は DC と（`n` 偶数の）Nyquist で 1、それ以外で 0.5 のホスト定数（0 を含まないので `inf * 0` は起きない）。積む Op: `Irfft`・`Mul`・`Narrow`・`Concat` |
| `Irfft` | 実装 | V1 | `rfft(g, n, dim, swap(norm)) * c`。`c` は DC と Nyquist で 1、それ以外で 2。bin 軸を入力の bin 数 `m` へ合わせる。積む Op: `Rfft`・`Mul`・`Narrow`・`Concat` |
| `Stft`・`Istft` | fail-closed | — | 窓とフレームの overlap-add を再生可能な `Var` 演算で組むには O(T) 個のノードが要る（§21.2）。`center=true` の反射パディングは #3084 で `PadMode` が再生可能になったため保留理由ではない |
| `Inv` | 実装 | V1 | `dA = -(X^T g X^T)`（`X` は `Inv` ノードの写し）。積む Op: `Transpose`・`MatMul`（`matmul_fp32_strict`）・`ScalarUnary{Neg}` |
| `Solve` | 実装 | V1 | `dB = solve(A^T, g)`、`dA = -dB X^T`。積む Op: `Solve`・`Transpose`・`Contiguous`・`MatMul`・`Neg` |
| `SlogdetLogAbsDet` | 実装 | V1 | `dA = broadcast(g) * inv(A)^T`。積む Op: `BroadcastTo`・`Inv`・`Transpose`・`Mul` |
| `Det` | 実装（判定付き V1） | V1 | `dA = broadcast(g * det) * inv(A)^T`。§21.2 の行別スケール＋2 のべき乗チャンク構造（V3）は実装していない。V3 が必要になる入力は 29.3 の判定で拒否する |
| `Cholesky` | fail-closed | — | 二階の式が複雑（§21.2） |
| `Pinv`・`Lstsq` | fail-closed | — | 二階の式が複雑で、`rcond` による打ち切りが区分定数になる（§21.2） |
| `MatrixNorm` | fail-closed | — | 二階の式が複雑で、`Nuc`／`Spectral` は SVD の縮退を含む（§21.2） |
| `QrQ`・`QrR`・`SvdU`・`SvdS`・`SvdVh`・`EighValues`・`EighVectors` | fail-closed | — | 多出力ノードが兄弟の値を非追跡ペイロードとして持つ（§8 (c)）。固有値・特異値が縮退すると二階微分が不安定になる |
| `SlogdetSign`・`MatrixRank` | fail-closed | — | 1 階 VJP が恒等的にゼロ（`grad.rs` の明示ゼロ）で二階の対象がない |

件数: 実装 8（FFT 4・線形代数 4）、fail-closed 15、合計 23。

### 31.2 FFT の数値契約

- 値は 1 階 VJP（`fandhe_ai_tensor_core::fft::*_vjp_host`）と REQ-2 統一複合判定で一致する。bit 同一は主張しない（子テープは `Var` の FFT 呼び出し、1 階はホスト関数で、どちらも f64 内部だが定数 2 のべき乗の掛け順が異なる）。
- 掛ける定数（0.5・2・ゼロ詰め）は 2 のべき乗で丸めに影響しない。FFT は有限和で段をまたぐ増幅がないため overflow の上界判定は付けない。真の値が f32 を超える場合だけ 1 階と同じく `inf` になる。
- `FftNorm` は `#[non_exhaustive]`。未知 variant は `validate_ancestors` が子テープへ書く前に `Err(Backward)` で拒否する（`check_fft_norm`）。

### 31.3 線形代数の数値契約と総括的な適用判定

1 階 VJP は f64 の LU で計算し f32 へ丸めるのは最後の 1 回だけだが、子テープの構造は f32 の `Var` 演算の連鎖である。そのため構造を積む前に f64 参照（`eval::linalg::inv_and_det_f64`。読み取り専用の補助関数で既存関数の数値は不変）で各段の値と上界を求め、次のいずれかなら `Err(Backward)` で拒否する（呼び出し側のロールバックで `child` は空へ戻る。§27.3・§28.3 と同じ形）。

- 上流 `g`（`Solve` は `b` も）に非有限要素がある。行列が特異で逆行列を作れない。
- 構造の段（`X`、`X^T g`、`X^T g X^T`／`solve(A^T, g)`・`dB X^T`／`g * X`、`Det` は `det` と `g * det`）のどれかが、非有限・`2^125` 以上（`STAGE_BOUND_GUARD`。値は変えていない）・「非零なのに f32 の正規化最小値未満」（ゼロ化・subnormal 化しうる）。
- 二階の逆伝播の上界（seed の各要素の絶対値 1 以下を仮定し、`Inv`・`Solve` は `(n * max|X|)^4 * max(1,|g|)`、`Det` 系は `^3` に `max(1,|det|)` を掛けた保守的な見積もり）が `2^125` 以上。

保証範囲: 判定を通った入力では、`cg.grad` が 1 階と REQ-2 で一致し、seed が 1 以下の二階・孫テープへの再生が有限になる。対象外は §27.3 と同じ（上流 Op・|seed|>1・三階以上・真の値が f32 を超える場合）。判定は保守的で、拒否されても実際に overflow するとは限らない。`Det` は V3 を作らなかったため、`A = diag(1e20, 1e20)`（`det = 1e40`）のような「V3 なら受理できる」入力は拒否する。

### 31.4 テスト

- `crates/autodiff/tests/create_graph_linalg_fft.rs`: 8 件の Hessian を 5 点中央差分と REQ-2 判定で突合（FFT は norm 3 種 × 長さ縮小・同長・拡大 × rank 1〜3・dim 先頭／中央、`Solve` は `A`・`B` の混合項を含む）。1 階の bit 同一・`cg.grad` の REQ-2 一致・孫テープ再生の一致。拒否例（`A=[[1e-39]]` の `Inv`、`A=[[1e-10]]`・`B=[[1e30]]` の `Solve`、`diag(1e20)` の `Det`）が `Err(Backward)` で `child` が空・同じ `child` で再試行できること。ほどほどの入力が受理されること。fail-closed の 15 件が型付きエラーで拒否されること、`requires_grad == false` の部分木の fail-closed Op は定数として扱われること。
- `tape.rs` の `linalg_fft_family_supports_create_graph`: 8 件が `true`、隣接 5 件が `false`。
- 二次形式の FFT 損失は勾配が入力に対し厳密に線形なので、f32 の勾配丸め雑音を抑えるため刻み 0.5 の中央差分を使った（判定式・tolerance は変えていない）。

### 31.5 所有者へ戻す論点

1. `Det`・`Inv`・`Solve`・`SlogdetLogAbsDet` で判定に引っかかる入力を、拒否ではなく §21.2 の S2／V4（行別スケール＋チャンク）で受理する方向へ広げるか。
2. `SlogdetSign` を損失に使う経路は fail-closed のため create_graph 全体が拒否される（使い勝手上の制約）。
3. `Stft`・`Istft` は overlap-add の構造（O(T) 個のノードが要る）を解く設計が決まったら再検討する（`PadMode` は #3084 で再生可能になり、保留理由ではなくなった）。
4. 分解系（`Svd*`・`Eigh*`・`Qr*`）の縮退の扱いを決めたうえでの起票（起票にはユーザーの承認が要るため本 issue では行っていない）。
5. CUDA／Metal 実機での `cg.grad` の parity は未確認（線形代数の forward は GPU からホスト f64 へフォールバックする。#3168・#3169 と同じ申し送り）。

### 31.6 変えていないもの

1 階 `Tape::backward` の結果（bit 同一）・tolerance・baseline・`STAGE_BOUND_GUARD` の値・FMA 契約・依存・`unsafe`・`docs/spec`・facade の公開面。

## 32. 実装記録（#3087・facade 経由の確認と拒否基準の差し替え。親 #3075）

- facade の `backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi` で、§22〜§31 で拡張した Op 族の代表を確認するテストを足した（対応表は `autodiff-functional-transforms-design.md` §34.1）。§21.6 の基準テストに対応する facade 側のテストは `crates/facade/tests/create_graph_facade.rs` の `*_create_graph_via_facade`（17 件）。
- 拒否の基準 Op を `LogSumExp`・`Max`（C 区分。将来の拡張で対象になりうる）から恒久対象外の `Op::Custom`（§21.2 A 区分）に替えた。`.is_err()` は `Err(AutodiffError::Backward(_))` の一致に強めた。
- 1 階 `first_order()` は素の `backward` と bit 一致。`cg.grad` は §9 (i) のとおり REQ-2 が契約で、gelu・gelu_tanh・inv・det・huber・cross_entropy は REQ-2、それ以外は実測で bit 一致のため bit 一致で固定した。
- GPU 実機（LayerNorm・Conv2d・MaxPool2d）は `#[ignore]` で追加。実測は #3168・#3169 へ申し送る（`docs/perf/logs/create-graph-extended-ops-3087/README.md`）。
- 変えていないもの: 公開 API・1 階 `backward` の値・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・`crates/autodiff/src`。

## 33. 決定記録（イシュー #3725・親 #3719。create_graph の C 区分・統計縮約 7 Op）

### 33.0 位置づけ・承認根拠・境界・推奨

- **状態**: 設計記録のみ。承認は #3767 の回答待ちで、本節は何も承認しない。回答があるまでコード・rustdoc・公開 API・`docs/compat-api-scope.md` は変えない
- **基準**: `origin/main` の `69881d9da`。以下の `file:line` はこの時点で確かめた値で、実装 PR の前に引き直す
- **対象**: §21.2 の C 区分（族を割り当てず fail-closed に固定した 24 Op）のうち、統計縮約の 7 Op（`Var`・`Std`・`VectorNorm`・`PNorm`・`LogSumExp`・`Nansum`・`Nanmean`）。§21.2 の C 表は方式欄を「V3」とだけ書くが、Op ごとに中身が違う（33.10）
- **根拠**: §21.1（共通方式 V1〜V4・S1〜S3・閉包性）、§21.6（fail-closed 契約）、前例は §24（Softmax／LogSoftmax の V3）、§27（LayerNorm／RmsNorm の S1 と `check_structure_stages`）、§28、§31（総括的な判定）。ルート #3711 の「実装を既定に設計する」趣旨は設計方針の根拠であり、承認ではない
- **後続**: 実装は #3791（Var・Std・VectorNorm・PNorm）と #3792（LogSumExp・Nansum・Nanmean）。承認は #3767、実機 parity は #3856（`<cuda-node>`）・#3858（`<metal-node>`）、雛形は #3776・#3777、測り方は #3739
- **推奨**: **実装する（7 Op すべて）**。理由は次の 4 つ
  1. 1 階 VJP がすべてホストの f64 関数で、V3 の値にそのまま流用できる（`crates/autodiff/src/grad.rs` の `var_vjp`:8110・`std_vjp`:8181・`vector_norm_vjp`:8268・`logsumexp_vjp`:8362・`pnorm_vjp`:8472、`crates/tensor-core/src/stat_reduce.rs` の `nan_reduce_vjp_host`:583）
  2. 構造に要る Op がすべて既に `supports_create_graph == true`（33.11）
  3. S1 と総括判定の前例（§27.3）がある
  4. 分散・ノルム・logsumexp を含む損失が `jvp`／`jacfwd`／`hvp` で拒否される実害がある

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | 7 Op を実装する | 正則化項・logsumexp 損失の HVP が使える。追跡表の項目 3 が「ある」に近づく | 実装と保守の費用。総括判定の追加 |
| B | 対象外にして fail-closed を恒久契約にする | 費用ゼロ | 上記の損失で高階変換が使えないまま残る |

### 33.1 公開形

該当なし。新しい公開名は増えない。`Tape::backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` は公開済みで、受理する Op の範囲が広がるだけである。`crates/facade/tests/api_surface.rs` には、この 7 Op に関わる否定ガードも保留 doctest ガードも無く、正ガードへ反転するものも無い。実装 PR で変える箇所:

- facade の rustdoc（`crates/facade/src/lib.rs`）にある「対象 Op の正本は §21.2・§22〜§31」の範囲に §33 を加える
- `Op` の doc コメント（`crates/autodiff/src/tape.rs`）
- `supports_create_graph`（`tape.rs:2526`）の網羅 match で 7 Op を false 側から true 側へ移す。ワイルドカードは足さない。`VectorNorm` は payload で判定し（`ord` が `L1`／`L2` なら true）、`#[non_exhaustive]` の未知 `ord` は false にする（`ScalarUnary` の前例）

### 33.2 rustdoc と doctest

- 新しい公開 API が無いので必須の doctest は無い。任意で facade の `backward_create_graph` に `std` か `logsumexp` を含む CPU だけで通る HVP 例を 1 つ足す。GPU の例は置かない（置くなら `no_run`／`#[ignore]`）
- 失敗系は `compile_fail` に頼らず、ランタイムの `Err(AutodiffError::Backward(_))` 一致と `child.is_empty()` で確かめる

### 33.3 エラー型

- 新しい variant は足さない。総括判定で外れた入力、非対応の `ord`、非有限レーンの拒否（33.12）は `AutodiffError::Backward(String)` にする
- 検証順: 判定は子テープへ書く前に行う（`validate_ancestors` か cgrad 構築の冒頭）。拒否したら `Tape::backward_create_graph` のロールバックで `child` を空に戻し、再試行できるようにする（§27.3 と同じ）。shape・`dim`・`correction` は forward 側の確保前検証を通ったものだけが届く。本番経路で `unwrap`／`expect` を使わない

### 33.4 テスト計画

候補名は案であり、実装 PR で確定する。CI の行は新規 `crates/autodiff/tests/create_graph_stat_reduce.rs` に置く。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|--------------|--------|------|---------------|----------------|----------------|
| 1 階が素の `backward` と bit 同一、`cg.grad` が REQ-2 で一致（V3 は CPU で bit 一致も見る） | `stat_reduce_first_order_bit_identical_and_cgrad_req2` | CI | 未実測 | 未実測 | 未実測 |
| ヘッシアンが有限差分と一致（7 Op × `dim` None／Some × rank 1〜3 × `correction` 0／1 × `p` ∈ {0.5, 1, 1.5, 2, 3}。5 点中央差分 h=1e-2 は §27.4 の前例どおり。判定式は変えない） | `stat_reduce_hessian_matches_finite_difference` | CI | 未実測 | 未実測 | 未実測 |
| 必須ベクトルが 3 段（1 階・`cg.grad`・`jvp_create_graph` 経由の孫テープ再生）で有限 | `stat_reduce_mandatory_vectors_finite_in_three_stages` | CI | 未実測 | 未実測 | 未実測 |
| 分母ゼロのレーン（std=0・‖x‖=0・p<1 で x_i=0・全 −inf・nanmean の count=0） | `stat_reduce_zero_denominator_lanes` | CI | 未実測 | 未実測 | 未実測 |
| 安全でない入力の拒否・`child` が空・再試行できる | `stat_reduce_unsafe_inputs_rejected_child_empty_and_retry` | CI | 未実測 | 未実測 | 未実測 |
| 構造の全経路が有限か拒否（§27.4 と同型の掛け合わせ。受理と拒否の両方が出ることも確かめる） | `stat_reduce_sweep_every_structure_path_is_finite_or_rejected` | CI | 未実測 | 未実測 | 未実測 |
| 隣接 Op（`MatrixNorm`）が fail-closed のまま | `stat_reduce_neighbouring_unsupported_op_stays_fail_closed` | CI | 未実測 | 未実測 | 未実測 |
| facade 経由（代表は `std`・`logsumexp`） | `crates/facade/tests/create_graph_facade.rs` に `*_create_graph_via_facade` を追加 | CI | 未実測 | 未実測 | 未実測 |
| CUDA／Metal の parity（forward がホストへフォールバックするかを含む。33.12） | `create_graph_stat_reduce_cuda_parity`／`create_graph_stat_reduce_metal_parity`（後者は `cfg(target_os = "macos")`）。置き場・名前は #3776・#3777 の雛形が確定するまで仮。測り方は #3739 に従う | `#[ignore]` | 未実測 | 未実測 | 未実測 |

実機の実行は #3856・#3858 が担う。Linux や開発機の代替値で pass にしない。

### 33.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えない。新しい baseline は作らない（作るなら実機の実測値に限り所有者承認）
- 1 階は構造上 bit 同一（§21.1(1)）。V3 の Op は `cg.grad` の値も CPU で 1 階と bit 同一。V1／V2 は REQ-2
- f64 長軸縮約の契約（`coding-rust.md`。二乗和は先に f64 へ上げてから二乗する）は V3 の値側で守る。構造側の f32 合成は値を決めない。FMA 契約には影響しない
- 必須ベクトル（3 段すべてで有限かつ一致。外れる場合は総括判定で拒否され `child` が空になることを固定する）:
  - 大スケール `x=1e20`・`[-1e20, 1e20]`
  - 中心化の丸め `[16777216, 16777218]`（Var・Std）
  - 小スケール `[1e-30, 1e-30]` と非正規化数の入力
  - 大きな upstream `g=3e38`・`g=[3e38, -3e38]`
  - 定数レーン（std=0）・全ゼロレーン
  - LogSumExp の `[-inf, -inf]`・`[+inf, 1]`・`[+inf, +inf]`
  - PNorm の `p=0.5` でゼロ要素を含むレーン、極小 `p`（例 0.01）× 長いレーン
  - NaN を含むレーン（Nansum／Nanmean）
- 保証範囲は §27.3 と同じ書き方にする。総括判定を通った入力で、seed の各要素の絶対値が 1 以下のときに限る。三階以上の微分と、真値が f32 を超える場合（1 階と同じく `inf`）は対象外

### 33.6 互換への影響

**追加のみ**。`api-compat-0.14.0-decision.md` §8 の手順で判定する。触れる公開型が無く（`Op` は `pub(crate)`）、`AutodiffError` の variant も公開シグネチャも変わらない。以前 `Err` だった入力が `Ok` になる受理範囲の拡大であり、§1.1 の破壊的条件には当たらない。`Op` の payload（`dim`・`correction`・`ord`・`p`）は再生に必要な値を持っており、追加は要らない。

### 33.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）
- `docs/compat-api-scope.md` への適用記録は #3767 の承認後の実装 PR で書く。文案: `**適用記録（イシュー #3791・#3792・親 #3725。承認: <#3767 の回答 URL>）**: 統計縮約 7 Op を create_graph の対象にした。公開名の追加なし。`
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（承認後の PR で行う。承認が無いうちは書かない）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md` を更新した（項目 3 の設計記録の欄に §33 を追記）

### 33.8 ベンチと性能への影響

既定経路（`Tape::backward`・forward）は変わらない。`supports_create_graph` は create_graph の入口でしか呼ばれない。増えるのは create_graph 経路の構造ノード（1 Op あたり数〜十数ノード）と総括判定のホスト走査 O(numel) だけである。既定経路に触れないので前後比較は不要と判断する。実装で既定経路に触れた場合は、5 回計測の中央値で前後を比べ、判定規則を計測前に固定する。実機の値は未実測。

### 33.9 依存の追加の有無

追加なし。許容 12 区分・禁止リストとの照合で変化なし。新しい `unsafe` なし。`erf` 等の外部クレートも使わない。

### 33.10 族の方式と式（Op ごと）

記号: `x_m` は子テープ上の入力の写し、`g` は子テープ上の upstream、`g_b` は `g` を縮約軸へ broadcast したもの。縮約軸は `dim` を permute／reshape で `[lanes, axis_len]` に寄せて扱い、終わったら戻す（`Permute`・`Reshape`・`Contiguous`。`dim=None` はレーン 1 本）。再生（R）は公開メソッドへ委譲する（`Var::var`:7026・`std`:7087・`norm_l1`:7109・`norm_l2`:7116・`logsumexp`:5999・`norm_p`:6017・`nansum`:5245・`nanmean`:5239。いずれも `crates/autodiff/src/var.rs`）。V3 の値に使う 1 階のホスト関数は `pub(crate)` にする（本体・数値は変えない）。

**#3791 側**

- **Var（V3。中心化を安定化する）**: cgrad = `g_b · 2·dev / (n−c)`。dev は §21.3／§27.1 の安定な中心化（`x_ref=Narrow`・`d=x−x_ref`・`dev=d−Mean(d)`・補正 `dev2=dev−Mean(dev)`）で作る。`create_graph_norm::stable_centering`（`create_graph_norm.rs:640`）を軸一般化して共用する。dev について線形なので S1 のスケールは要らない。値は `var_vjp`。`Mean` の f32 部分和の overflow（§27.2(c) と同じ型）は総括判定で拒否する
- **Std（V3 + 2 段スケール）**: cgrad = `g_b ⊙ dev / ((n−c)·std)`。dev／std はスケール不変なので、まず `x/s0`（s0 は `max|x|` 以上の 2 のべき乗。指数は [-126, 127] にクランプし、全ゼロは 1）で前スケールし、安定な中心化のあと、さらに `dev/s1`（s1 は子テープ上の実値 `max|dev2|` 以上の 2 のべき乗）で正規化する。そのうえで `std_s = sqrt(Σ dev_s² / (n−c))` を Sum・Mul・Div・Sqrt で合成する。`|dev_s / std_s| ≤ √(n−c)` に収まる。値は `std_vjp`。`(std as f32)==0` のレーンは 1 階が 0 勾配にマスクしているので V2 にする。出力側の `where_cond` だけでは足りない。捨てた側の `Div` の VJP が分母 0 で NaN を作るため、分母を `where_cond(mask, 1, std_s)` で先に守る。さらに分母だけの保護でも足りない。出力を 0 に置き換えると逆伝播で `Sqrt` にゼロの cotangent が届き、`Sqrt` の微分係数 `0.5/y`（`crates/autodiff/src/eval/scalar.rs:90`）が `y=0` で `0×inf=NaN` になる。そこで定数レーンでは `Sqrt` に入る二乗和 `Σ dev_s²/(n−c)` を先に `where_cond(mask, 1, 二乗和)` で正の定数へ置換し、そのあとで出力を `where_cond(mask, 0, …)` でゼロにする（`Sqrt` の入力側を守る）。前スケールは §27.2 が将来案に残した「S0」を、この族で採る形になる（§27.2 の本文は書き換えない）。代替は `x_m.std` の自己再生（`Op::Std` 自身を積む）で、33.14 に載せる
- **VectorNorm L1（V3 の値 + V2 の構造）**: 1 階は `g·sign(x)` を f32 の乗算で計算している（`grad.rs:8268` 以降。`g=inf` かつ `x=0` で NaN）。構造は `where_cond(x>0, g_b, where_cond(x<0, −g_b, 0))` とし、0/1 定数を掛ける方式は使わない（§21.1(3) の `inf*0` 規則）。値は `vector_norm_vjp` を V3 で入れて 1 階と bit 一致させる。x による二階は 0
- **VectorNorm L2（V3 + S1）**: cgrad = `g_b ⊙ x_s / ‖x_s‖₂`。`x_s = x/s` で、x/‖x‖ はスケール不変なので戻しは要らない。‖x_s‖ は Sum・Mul・Sqrt で合成し、値は 1 以上 √n 以下になる。‖x‖=0 のレーンは分母を `where_cond` で 1 にし、出力を 0 にする。ただし `Sqrt` の出力だけを置換しても、逆伝播で `Sqrt` にゼロの cotangent が届き、微分係数 `0.5/y`（`crates/autodiff/src/eval/scalar.rs:90`）が `y=0` で `0×inf=NaN` になる。そのため全ゼロレーンでは `Sqrt` に入る二乗和 `Σ x_s²` を先に `where_cond(mask, 1, 二乗和)` で正の定数へ置換し、そのあとで出力をゼロにする（`Sqrt` の入力側を守る）。NaN は IEEE のまま伝播させる（1 階の NaN 分岐と同じ）。値は `vector_norm_vjp`
- **PNorm（V3 + S1 + V2）**: cgrad = `g_b ⊙ sign(x) ⊙ (|x_s| / ‖x_s‖_p)^(p−1)`。sign は `where_cond` の 3 分岐にする。`x_i=0` の要素は `|x_s|` を `where_cond(x==0, 1, |x_s|)` で守ってから `PowScalar` に入れ、出力を 0 にする（p<1 で `0^負` が inf になるのを防ぐ。1 階の劣勾配の選び方と同じ）。`‖x_s‖_p = (Σ |x_s|^p)^(1/p)` は Abs・PowScalar・Sum で合成する。全ゼロレーンでは `Sum` の結果（`PowScalar(1/p)` の入力）も同様に `where_cond` で正の定数へ先に置換する（`0^(1/p)` の微分係数が `p>1` で `0×inf=NaN` になるのを防ぐ）。極小の p では f32 で overflow するので総括判定で拒否する。`±inf` を含むレーン（1 階は均等分配の特別分岐）は 33.12 と 33.14 で扱う。値は `pnorm_vjp`

**#3792 側**

- **LogSumExp（V3）**: 有限なレーンの cgrad = `g_b ⊙ softmax(x_m, dim)`（`dim=None` は reshape して軸 0）。Softmax は #3077（§24）で対象になっており依存は満たされている。値の範囲は [0, 1] で `|cgrad| ≤ |g|`。代替の `g·exp(x_m − lse_m)` は LogSumExp 自身を積むので採らない。値は `logsumexp_vjp`。非有限レーンの 3 分岐（全 −inf は 0、+inf は均等分配、NaN は伝播）はこの構造では表せない。推奨は V2: 入力を `where_cond(非有限レーン, 0, x_m)` で守り、全 −inf レーンは 0、+inf レーンはホスト定数 `1/inf_count` を +inf の位置に置いて `g_b` と選び合わせ、NaN レーンは伝播に任せる。代替は非有限な x を入口で拒否する案（§27.2(a) 型）で、33.14 に載せる
- **Nansum（V3 の値 + V2 の構造）**: 構造は `where_cond(isnan_mask, 0, g_b)` で `x_m` に依存しない（x による二階は 0）。値は `nan_reduce_vjp_host`（NaN の位置は `0×t` なので、`g` が非有限なら NaN）。構造との違いは `g` が非有限な場合の構造側だけに限られる
- **Nanmean（V3 の値 + V2 の構造）**: 構造は `where_cond(isnan_mask, 0, g_b / count)`（count はホストの整数定数）。値は `nan_reduce_vjp_host`（f64 で割ってから 1 回 downcast）。count=0（全 NaN）のレーンは分母を 1 に守って構造を有限にし、値はホスト関数の NaN をそのまま使う（1 階と同じ）

### 33.11 閉包性と族間の依存

- 子テープへ積む Op: `Reshape`・`Permute`・`Contiguous`・`Narrow`・`BroadcastTo`・`Sum`・`Mean`・`Mul`・`Add`・`Where`・`ScalarBinary`（Sub・Div）・`ScalarUnary`（Sqrt・Abs・PowScalar・Neg）・`Softmax`（LogSumExp だけ）。基準時点で `supports_create_graph`（`tape.rs:2526`）と `scalar_unary_replayable`（`create_graph.rs:145`）で true になっていることを確かめた。`ScalarBinary` は実装前に origin/main で再確認する
- 他族への依存は #3077（Softmax）だけで、マージ済み。7 Op はどれも自分自身を構造に積まない（Std の自己再生は代替案）
- 孫テープの再生では、子テープ上にある 7 Op の forward の写しが再生されるので、7 Op 自身が true であることが要る（族の中で閉じる）
- 依存順: #3791 と #3792 は互いに独立。`stable_centering` の共用化は #3791 が持つ

### 33.12 fail-closed・総括的な適用判定・拒否テスト

- **総括判定**: `check_structure_stages`（`create_graph_norm.rs:338`）と同型の判定を族全体に置く。構造の forward の各段、二階の逆伝播の「cotangent × 微分係数」、境界の cotangent の上界をホストが f64 で求め、`2^125` 以上か非有限になる段があれば拒否する（閾値と seed の絶対値 1 以下という前提は §27.3 と同じ）。個別の条件を足していく方式は、§27.3 でレビューが収束しなかった経緯から採らない。特に危ない段は PNorm の `(Σ)^(1/p)`、Std／L2 の低分散レーンで `1/std²` 型の係数が cotangent を増幅する段、Var の `Mean` の部分和
- **入口での拒否の候補**: 非有限要素を含む x（Std／L2／PNorm／Var。推奨は「NaN は伝播、±inf は拒否」）、非有限の g（§27.3 と同じ）。33.14 で問う
- **既存テストの差し替え（#3792 の作業）**: `crates/autodiff/tests/create_graph.rs` の `create_graph_rejects_unsupported_op_logsumexp`（:697）と `create_graph_rejects_softmax_mixed_with_unsupported_op_and_keeps_child_empty`（:2955）は LogSumExp を拒否の見本にしており、LogSumExp が true になると壊れる。隣接 Op は `MatrixNorm`（§21.2 の #3086 で fail-closed。ノルムに近く、兄弟の C 区分 issue の対象でもない）に替える。実装時点で `MatrixNorm` が true になっていたら恒久対象外の `Op::Custom` に替える。`Max`／`Amax` は兄弟 issue と同時に true になりうるので使わない
- **CUDA／Metal の forward**: forward は 1 階の既存経路のままで、本設計では変えない。create_graph の構造ノードはホストの再生経路（既存のホスト実装）で評価し、GPU カーネルを新設しない。GPU で cgrad の値が一致するかは未実測で、#3856・#3858 が `#[ignore]` テストで確かめる（33.4）
- §21.6 の判定順・網羅 match・未知 variant を false にする規約は弱めない

### 33.13 セキュリティ観点・スコープ外・Closes／Refs

- A03: shape・`dim`・`correction` は forward 側の確保前検証を通ったものだけが届く。総括判定はホスト走査で、新しい確保は O(numel) の f64 一時領域に限る
- A04: 拒否時に子テープを空に保ち、fail-closed を維持する
- A06: 依存を追加しない。A08: 該当なし。秘密情報・ホスト名・個人のパスは書かない
- **スコープ外**: GPU 実機の parity（#3856・#3858）、測り方（#3739）、f64 テープの create_graph（#3722）、S2 の実装（拒否範囲を狭めるかは 33.14）、`VectorNormOrd` の新 variant。新しい issue は承認なしに起票しない
- `Closes #3725`。`Refs #3791 #3792 #3767 #3719`

### 33.14 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 族の採否 | 7 Op を実装する／対象外にする | 実装する | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 2. LogSumExp・PNorm の ±inf レーン | V2 で 1 階の極限の意味を写す／入口で拒否 | V2 で写す | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 3. Std／L2／PNorm／Var の非有限な x | NaN は伝播し ±inf は拒否／全部拒否／全部受理 | NaN は伝播し ±inf は拒否 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 4. 総括判定で外れる入力 | 型付きエラーで拒否（§27.2 と同じ）／S2 を作って受理 | 型付きエラーで拒否 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 5. Std の構造 | 2 段スケールの合成／`Op::Std` の自己再生 | 2 段スケールの合成 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 6. 前スケール S0 をこの族で採ること（§27.2 の将来案を先に使う） | 採る／採らない | 採る | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 7. GPU parity の判定方式 | #3739 の推奨に従う（tolerance は変えない）／別方式 | #3739 に従う | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |

### 33.15 変えていないもの

コード・公開 API・rustdoc・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・`docs/compat-api-scope.md`。

## 34. 設計記録（イシュー #3724・親 #3719 系。create_graph の C 区分・選択縮約 7 Op）

### 34.0 位置づけ・承認根拠・境界・推奨

- 状態: **設計記録のみ**。承認は #3767 の回答待ちで、本節は承認を代行しない。コード・公開面・tolerance・baseline は変えない。基準 sha は `69881d9da`。
- 対象 7 Op: `Max`・`Min`・`Amax`・`Amin`・`OrderSelect`・`MedianAll`・`Quantile`。いずれも `crates/autodiff/src/tape.rs` の `supports_create_graph`（:2526）で `false` の腕にあり、`backward_create_graph` は型付きエラーで拒否する（fail-closed）。
- 根拠: §20（D-A・2026-10-11）、§21.2 の C 区分表、ルート #3711 の起票経緯（実装を既定とする方針）。これらは設計方針の根拠であって承認ではない。
- 推奨: **実装する**。理由は次の 3 点。
  - 7 Op とも 1 階 VJP が「選択（マスク・索引）× 定数重み」の形で、V1／V2／V3 の既存前例（§21.1(3)・§25・§30）で閉じる。
  - 子テープへ積む Op は既に対象の Op とこの族自身だけで、他の C 区分の族（#3725〜#3728）に依存しない。
  - 既定経路（1 階 `backward`）は変わらない。
- 境界: 統計縮約（`Nansum`・`Nanmean` 等。#3725）は扱わない。公開面の追加はしない。実装は #3789（`Max`・`Min`・`Amax`・`Amin`）と #3790（`OrderSelect`・`MedianAll`・`Quantile`）に分ける。
- 訂正の注記: §21.2 の表は選択縮約を「V2」と一括りにしている。本節では Op ごとに V1／V2／V3 を割り当てる。§21.2 の本文は書き換えない。

### 34.1 公開形

- 該当なし。新しい公開名は無い。facade は `Var` をそのまま再エクスポートし（`crates/facade/src/lib.rs:281`）、`max`／`min`／`amax`／`amin`／`median`／`kthvalue`／`quantile` と `backward_create_graph`（:935）は公開済みで、変わるのは受理する Op の範囲だけである。
- `api_surface.rs` に本族を拒否する否定ガードは無い（fail-closed は挙動で固定されている）。正ガードに当たるものは 34.4 の facade 挙動テスト `*_create_graph_via_facade` とし、追跡表のセルは実装 PR で更新する。

### 34.2 rustdoc と doctest

- facade `backward_create_graph` の rustdoc（`lib.rs:910` 付近）の「対象 Op の正本」へ選択縮約を追記する。実装 PR で行う。
- doctest は CPU で通る `x.max(None)` を含む二階の例を 1 つ置く。失敗系は `compile_fail` に頼らず挙動テストで検証する。

### 34.3 エラー型

- 新しい variant は足さず、既存の `AutodiffError::Backward` を使う。拒否するのは次の場合。
  - 未知の `QuantileInterpolation`（`#[non_exhaustive]`。`stat_reduce.rs:53`。`validate_ancestors` に `check_fft_norm`〈`create_graph/linalg_fft.rs:68`〉と同形の検査を足す）
  - `OrderSelect` の `index` の範囲外と i32 の桁あふれ（§30.1 の前例）
  - V3 の shape 不一致（`push_v3`）
- 検査はすべて確保より前、かつ子テープへ書く前に行い、拒否時は `child` が空のまま残る。本番経路で `unwrap`／`expect` は使わない。

### 34.4 テスト計画

| 区分 | 内容 |
|---|---|
| CI（ubuntu-latest） | 新設 `crates/autodiff/tests/create_graph_select_reduce.rs`。7 Op の Hessian を 5 点中央差分と REQ-2 判定で突き合わせる。選択だけでは二階が 0 なので合成形を使う（`max(x⊙x)`・`max(x)²`・`amax(x)²`・`kthvalue(x³)`・`median(x)²`・`quantile(x², q)`。`dim` は None／中央軸、補間は 5 種すべて） |
| CI | 1 階 `first_order()` の bit 一致、`cg.grad` の bit 一致、`jvp_create_graph`（孫テープ）と独立参照の一致 |
| CI | タイと NaN の回帰（34.12 の回帰ベクトル） |
| CI | 隣接する fail-closed の拒否テスト: `Op::Custom` との組み合わせで拒否され `child` が空のままであること |
| CI | `tape.rs` 単体テスト `selection_reduce_family_supports_create_graph`: 7 Op が `true`、隣接の `Nansum`・`Nanmean` が `false`（兄弟の族が反転した PR がこの一覧を直す） |
| CI | `create_graph/` 配下の単体テスト: マスク・重みが 1 階ホスト関数と bit 一致、範囲外 `index` が型付きエラー |
| CI | facade `crates/facade/tests/create_graph_facade.rs` に `max_`／`amax_`／`kthvalue_`／`median_`／`quantile_create_graph_via_facade` を `check_create_graph_family` で追加 |
| CI | 34.13 の既存テスト差し替え |
| `#[ignore]`（実機） | `gpu_extended_ops_match_cpu` のケース表へ `max`・`amax`・`quantile` を追加（`create_graph_extended_ops_{cuda,metal}_matches_cpu` で走る）。#3776／#3777 の雛形ができたら登録表へ移す |

実機の結果はすべて **未実測**。実測は #3856（`<cuda-node>`）・#3858（`<metal-node>`）に申し送る。

### 34.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を使う。tolerance 定数・baseline・ガードレール閾値・FMA 契約は変えない。f64 長軸縮約の契約はホスト関数側でそのまま保たれる。baseline の追加は無い。詳細は 34.12。

### 34.6 互換への影響

- 該当なし。facade の公開型とシグネチャは変わらない。`api-compat-0.14.0-decision.md` §1.2 の項目 3（挙動のみ。`Err` が `Ok` になる受理拡大）にあたり、§8 の手順に当てはめても破壊的変更は生じない。
- tensor-core に `pub fn`（34.10(e)）を足す場合も内部クレートへの追加のみで、facade は `stat_reduce` を再エクスポートしない。

### 34.7 適用記録・索引・追跡表

- `compat-api-scope.md` の適用記録は承認後の実装 PR で書く。本 PR では行わない。
- 本 PR で行うのは `docs/README.md` の索引と、`docs/compat-matrix-ari-tracker.md` 行 3 の「設計記録」セルの更新だけ。

### 34.8 ベンチ

- 既定経路（1 階 `backward`）に影響は無い。変わるのは create_graph 経路と `supports_create_graph` の腕だけで、共有ヘルパーの切り出しも値を変えない。前後比較は不要。

### 34.9 依存

- 追加しない。許容 12 区分と禁止リストに照らして新規なし。新しい `unsafe` も無い。

### 34.10 方式と式

| Op | 方式 | 再生 | cgrad の構造 | 子テープへ積む Op |
|---|---|---|---|---|
| `Max`／`Min` | V2 | `Var::max`／`min` | 1 階 `extremum_first_match_vjp`（`grad.rs:7912`）と同じ走査で bool マスクをホスト生成し `where_cond(M, g を keepdim→broadcast, 0)` | `Reshape`・`BroadcastTo`・`Where`・定数 `Leaf` |
| `Amax`／`Amin` | V3（推奨） | `Var::amax`／`amin` | 一致全要素のマスクと一致数 `k`。`k_safe = max(k,1)` の f32 定数で割り `where_cond`。値は `extremum_even_split_vjp`（:8009）を `push_v3` で載せる | `Reshape`・`BroadcastTo`・`ScalarBinary{Div}`・`Where`・`Contiguous` |
| `OrderSelect` | V1 | `gather(dim, index_keep).reshape` | `gather_like_vjp` ＝ `zeros.scatter_add`。1 階と同じ `scatter_with_fallback(Add)` を通る | `Gather`・`Reshape`・`Scatter{Add}`・定数 `Leaf` |
| `MedianAll` | V3 | `Var::median(None)` | 1 階と同じ規則で中央値 `m` を求め、等価マスク（`m` が NaN なら NaN 同士）と `count` から `where_cond`。値は `median_all_vjp_host` を `push_v3` で載せる | `Reshape`・`BroadcastTo`・`ScalarBinary{Div}`・`Where` |
| `Quantile` | V3 | `Var::quantile(q, dim, interp)` | 1 階と同じ安定ソートと rank 規則で lane ごとの下側・上側の添字と重みを得て、互いに素なマスクと f32 重み定数の `Mul`・`Add` で組む。値は `quantile_vjp_host`（`stat_reduce.rs:507`）を `push_v3` で載せる | `Where`・`Mul`・`Add`・`Gather`・`Scatter{Add}` |

(a) `Max`／`Min`
- 走査は外側 × 軸 × 内側の順で、記録値 `out_value` と `==` で一致した lane 内の最初の要素だけが選ばれる。`out_value` が NaN なら全要素が不一致で勾配は全ゼロになる。規則を 1 階と共用するため、走査を `pub(crate)` ヘルパーに切り出す（1 階の値は変えず、既存テストが無変更で通ることをガードにする）。
- マスクは区分定数なので x による二階微分は 0 で、`g` を通る混合項だけが流れる。

(b) `Amax`／`Amin`
- 1 階は一致全要素へ f32 の `g / (k as f32)` を 1 回だけ配る。`k == 0` の lane は 0 のまま。
- `k_safe` を使う理由は、`k == 0` の lane で `div` の VJP（`-g·y/k²`）が `inf·0` を生まないようにするため（前例: `create_graph.rs:1930` 付近の `safe_a_for_pow`）。
- V3 とする理由は、デバイスの `Div`（Metal の fast-math 等）が正しい丸めである保証が無く、V3 ならどのバックエンドでも `cg.grad` が 1 階と bit 一致するため。`k > 2^24` の丸めは 1 階に既にある限界と同じで、新しい拒否は足さない。

(c) `OrderSelect`
- payload に `k` が無く `kthvalue` を再実行できないため、子テープのタグは `Gather`＋`Reshape` になる（§25 の `Sort` と同じ前例）。

(d) `MedianAll`
- 1 階は f64 で `g/count` を計算し f32 へ丸める。構造側の f32 除算との bit 一致をデバイス演算に委ねないため、値はホスト関数で固定する。要素数 0 の入力では 1 階が空ベクトルを返すことと整合させる。

(e) `Quantile`
- lane に NaN があれば安定順で最後の NaN へ `g` を流す。`lo == hi` なら `g` を直接代入し、それ以外は `g·(1-w)`・`g·w` を配る。f32 に丸めると 0 になる重みの項は、乗算ではなくマスクで外す（`inf*0` を避ける）。
- 規則を 1 か所で持つため、tensor-core の `stat_reduce` に lane ごとの選択位置を返す `pub fn`（仮名 `quantile_select_host`）を足す案とする。1 階側の `sort_order`・`quantile_positions` は private。足すかどうかは #3790 で確定し、代案は 34.15 の論点 3。
- 重みは x について区分定数で、x による二階微分は 0。

### 34.11 閉包性

- 子テープへ積む Op は、`Reshape`・`BroadcastTo`・`Where`・`Contiguous`・`ScalarBinary{Div, Mul}`・`Add`・`Gather`・`Scatter{Add}`・定数 `Leaf` と、写しの forward として積む `Max`・`Min`・`Amax`・`Amin`・`MedianAll`・`Quantile`。前者は既に対象で、後者はこの族を `true` にしたときに初めて孫テープでも閉じる。このため #3789 と #3790 は、担当する Op を同じ PR の中で `true` にする。
- 依存先は §25（#3083 の索引系）と §16（#2062 の `Where`／`ScalarBinary`）だけで、いずれもマージ済み。他の C 区分の族には依存しない。§21.4 の依存順には「#3789・#3790 は §16・§25 の後、他の C 区分と独立」の 1 行を足す扱いとする。

### 34.12 タイ・NaN・1 階との bit 一致

- 1 階 `first_order()` は親テープを加工しないので構造上 bit 同一（§21.1(1)）。
- `cg.grad` が bit 一致する理由は Op で異なる。`Max`・`Min`・`OrderSelect` は選択だけだから。`Amax`・`Amin`・`MedianAll`・`Quantile` は V3 で 1 階のホスト値を使うから。
- 契約として約束するのは REQ-2（§9 (i)）で、テストでは bit 一致で固定する（§32 の運用と同じ）。孫テープ（`jvp_create_graph`）で再生した値は REQ-2 で一致すればよい。
- 回帰ベクトル: タイ（先勝ち・均等分配）、`[0.0, -0.0]`、NaN を含む lane（`Max` は勾配ゼロ、`Quantile` は最後の NaN）、`Amax` で `k == 0` の lane の二階が NaN を生まないこと、上流 `3e38` で 1 階・`cg.grad`・二階が有限であること。

### 34.13 既存テストの差し替え（#3789 で必須）

次の 7 か所は `x.max(None)` を「拒否される Op」の代表に使っており、#3789 で `Max` を `true` にした時点で壊れる。

- `crates/autodiff/tests/double_vjp_feasibility.rs:434`（`c4_unsupported_ops_are_rejected_fail_closed`）
- `crates/autodiff/tests/functional_ops_parity.rs:541`（`h3_rejections_leave_both_tapes_unchanged`）
- `crates/autodiff/src/functional_ops_jvp_tests.rs:419`（`j5_`）・:793（`m5_`。親テープの増分 `len + 2` を検査）・:1177（`c7_zero_paths_and_node_counts`）
- `crates/autodiff/tests/functional_ops_composition.rs:469`（`c7_fail_closed_on_composition`）
- `crates/autodiff/tests/jacobian_hessian_parity.rs:406`（`j4_hessian_propagates_create_graph_rejections_without_touching_child`）

代わりの拒否 Op は恒久的に対象外の `Op::Custom`（A 区分）に統一する（§32 と同じ判断）。`Var`・`LogSumExp`・`Nansum` 等は #3725 で `true` になりうるため使わない。`custom` のケースを既に持つテストでは `max` のケースを「受理される」確認へ反転するか削除し、ノード数を検査するテストは期待値を直す。`stat_reduce_parity.rs` の `create_graph_is_a_typed_error_not_a_panic` は `nansum` を使うので触らない（#3725 の扱い）。

### 34.14 GPU の forward 経路と Phase 10

- `Max`・`Min`・`Amax`・`Amin` の forward はデバイスカーネルで走る（CUDA: `crates/backend-cuda/src/ops.rs:6371`・:6380。Metal: `crates/backend-metal/src/ops.rs:4122`・:4220。Metal の `min` は ±0 の符号が実装依存で bit 一致は主張しない）。マスクは親の記録値から `==` で作るので、各バックエンドの NaN／±0 規則と 1 階の結果が構造上一致する。
- `OrderSelect`・`MedianAll`・`Quantile` の `BackendOps::stat_*` を実装するのは `backend-cpu`（`ops.rs:3665`）だけで、CUDA／Metal では `Unsupported` になりホストのカーネルへフォールバックする（`stat_reduce_ops.rs` 冒頭の経路 ③）。
- 子テープの構造 Op（`where_cond`・`div`・`mul`・`scatter_add`・`gather`）はデバイスで実行され、V3 の値はホストで計算する。実機の結果は全部**未実測**で、Phase 10 の #3856・#3858 へ申し送る。

### 34.15 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. `Amax`／`Amin` の値の方式 | V3 ／ V2 のみ（GPU では REQ-2 で一致） | V3 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 2. `cg.grad` の bit 一致を 7 Op のテストで固定するか | 固定する ／ V3 の Op だけ REQ-2 にする | 固定する（契約は REQ-2 のまま） | 同上 |
| 3. `Quantile` の位置情報の取得元 | tensor-core に `pub fn` を足す ／ autodiff 側で同じ規則を写す | `pub fn` を足す（規則を 1 か所で持つ） | 同上 |
| 4. 7 Op を実装するか | 全部実装する ／ 一部を対象外にする | 全部実装する | 同上 |

### 34.16 その他

- セキュリティ観点: 外部入力は扱わない。`index`・形状・`q`・enum variant・確保サイズは確保前に検証する。秘密情報は書かない。
- スコープ外: 統計縮約（#3725）など他の C 区分の族、公開面の追加、f64 テープ（#3722）。
- Issue の紐付け: 本 PR は `Closes #3724`、`Refs #3767 #3789 #3790`。
- 変えていないもの: コード・依存・`unsafe`・tolerance・baseline・`docs/spec`・既存節の本文・公開 API。

## 35. 設計記録（イシュー #3726・親 #3719。create_graph の C 区分・構造変換 3 Op）

### 35.0 位置づけ・承認根拠・境界・推奨

- **状態**: 設計記録のみ。承認は #3767 の回答待ちで、本節は何も承認しない。回答があるまでコード・rustdoc・公開 API・`docs/compat-api-scope.md` は変えない
- **基準**: `origin/main` の `ad92b6aa7`。以下の `file:line` はこの時点で確かめた値で、実装 PR の前に引き直す
- **対象**: §21.2 の C 区分（族を割り当てず fail-closed に固定した 24 Op）のうち、構造変換の 3 Op（`Dropout`・`Interpolate`・`MaxUnpool`）
- **根拠**: §21.1（共通方式 V1〜V3・閉包性）、§21.6（fail-closed 契約）。前例は §25（索引系・`scatter_vjp`）、§29（`MaskedFill` の `where_cond`）、§30（`MaxPool` の記録済み index を固定する V1、Avg 系の V3 と `check_structure_contract`）、§32（拒否見本を `Op::Custom` に替える方針）。ルート #3711 の「実装を既定に設計する」趣旨は設計方針の根拠であり、承認ではない
- **後続**: 実装は #3793。承認は #3767、実機 parity は #3856（`<cuda-node>`）・#3858（`<metal-node>`）、雛形は #3776・#3777、測り方は #3739
- **最初に押さえる事実**: 3 Op はどれも入力 x について**線形**（y = A·x。A は Dropout が記録済みマスク、Interpolate が座標式から決まる固定の重み行列、MaxUnpool が記録済み index による散布）。このため cgrad は `Aᵀ g` で g だけに依存し、x による二階の項は恒等的に 0 になる。二階微分は g を通る経路（下流の非線形）からだけ流れる
- **推奨**: **実装する（3 Op すべて）**。理由は次の 3 つ
  1. 1 階 VJP が supported な `Gather`・`Scatter`・`Mul`・`Where`・`BroadcastTo` と、§30 で前例のある V3 だけで子テープに書ける
  2. `Op` の payload の追加が要らない（Dropout は `mask`、Interpolate は `size`・`mode`、MaxUnpool は `index` とノード shape で足りる）
  3. 学習時の Dropout や U-Net 系の Interpolate・MaxUnpool を含むモデルが `hvp`・`jvp`・`jacfwd` で拒否される実害がある

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | 3 Op を実装する | Dropout 入りの学習モデルや decoder 系の HVP が使える。追跡表の項目 3 が「ある」に近づく | 実装と保守の費用。Area の構造判定の一般化 |
| B | 対象外にして fail-closed を恒久契約にする | 費用ゼロ | 上記のモデルで高階変換が使えないまま残る |

### 35.1 公開形

該当なし。新しい公開名は増えない。`Tape::backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` は公開済みで、受理する Op の範囲が広がるだけである。`crates/facade/tests/api_surface.rs` には、この 3 Op の create_graph に関わる否定ガードも保留 doctest ガードも無い（`DropoutEmbeddingBagHoldDoctestGuard`・`ConvTranspose3dMaxUnpoolHoldDoctestGuard` は層の公開の保留で、create_graph とは無関係なので触らない）。実装 PR（#3793）で変える箇所:

- `supports_create_graph`（`crates/autodiff/src/tape.rs:2526`）の網羅 match で `Dropout`・`MaxUnpool` を true 側へ移す。`Interpolate` は payload で判定する（`InterpolateMode` は `#[non_exhaustive]` なので内側の match に `_ => false` を置く。`ScalarUnary` の前例。§21.6 の「未知 variant は false」を守る）。外側の match にワイルドカードは足さない
- `Op` の doc コメント（`tape.rs:1109` Dropout・`:1365` MaxUnpool の「高階微分非対応」・`:1431` Interpolate・`:2506` の「隣接する `MaxUnpool` は `false` のまま」）、`crates/autodiff/src/max_unpool_ops.rs` のモジュール doc、`crates/autodiff/src/create_graph.rs` の「`MaxUnpool` は対象外」の記述
- facade の rustdoc（`crates/facade/src/lib.rs`）にある「対象 Op の正本は §21.2・§22〜§31」の範囲に §35 を加える
- `is_checkpoint_eligible`（`tape.rs:2014`・`:2073`・`:2098`）は変えない

### 35.2 rustdoc と doctest

- 新しい公開 API が無いので必須の doctest は無い。任意で facade の `backward_create_graph` に `interpolate`（Nearest）を含む CPU だけで通る HVP 例を 1 つ足す。GPU の例は置かない（置くなら `no_run`／`#[ignore]`）
- 失敗系は `compile_fail` に頼らず、ランタイムの `Err(AutodiffError::Backward(_))` 一致と `child.is_empty()` で確かめる

### 35.3 エラー型

- 新しい variant は足さない。拒否はすべて `AutodiffError::Backward(String)`: Interpolate の未知 `mode`、`sp_in_numel > i32::MAX`、索引テンソル長の桁あふれ、Area の構造判定外れ、MaxUnpool の index 範囲外・shape 不整合
- 検証順: 判定は子テープへ書く前に行う（`validate_ancestors` か cgrad 構築の冒頭）。拒否したら `child` を空に戻し、再試行できるようにする（§27.3・§30 と同じ）。本番経路で `unwrap`／`expect` を使わない

### 35.4 テスト計画

候補名は案であり、実装 PR で確定する。CI の行は新規 `crates/autodiff/tests/create_graph_structural.rs` に置く。線形 Op は `hessian(sum(op(x)))` が恒等的に 0 で有限差分が何も確かめないので、**非線形を合成した損失**で突き合わせる。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|--------------|--------|------|---------------|----------------|----------------|
| 1 階が素の `backward` と bit 同一、`cg.grad` が REQ-2 で一致（CPU では bit 一致も見る。Dropout の非有限 g の drop 位置は 35.10 の既知差として別に固定） | `structural_first_order_bit_identical_and_cgrad_req2` | CI | 未実測 | 未実測 | 未実測 |
| ヘッシアンが有限差分と一致: `sum(interpolate(x)²)`（全 7 mode × `align_corners` × アップ・ダウン・恒等 size × rank 3〜5）、`sum(max_unpool(max_pool(x))²)`（重なりなし・重複 index あり）、`sum(dropout_with_mask(x)²)`。5 点中央差分 h=1e-2 は §27.4 の前例どおり。判定式は変えない | `structural_hessian_matches_finite_difference` | CI | 未実測 | 未実測 | 未実測 |
| 素の和のヘッシアンが厳密に 0（線形性） | `structural_plain_sum_hessian_is_exact_zero` | CI | 未実測 | 未実測 | 未実測 |
| 孫テープ（`jvp_create_graph`）の再生が独立な参照と一致し、3 段（1 階・`cg.grad`・孫）すべてで有限 | `structural_grandchild_replay_matches_reference` | CI | 未実測 | 未実測 | 未実測 |
| Dropout の replay がグローバル RNG を消費しない（`manual_seed` で固定し、`backward_create_graph`／`jvp_create_graph` の前後で次の `rand` が同じ）。`Dropout2d`・`AlphaDropout`（同じ `Op::Dropout` 経路）と `p=1.0` も含む | `dropout_create_graph_does_not_consume_global_rng` | CI | 未実測 | 未実測 | 未実測 |
| 必須ベクトル（35.5）の有限性・bit 一致 | `structural_mandatory_vectors` | CI | 未実測 | 未実測 | 未実測 |
| 拒否（未知 mode は単体テスト、`sp_in` の i32 超過、Area の構造判定外れ、MaxUnpool の範囲外 index）で `child` が空・再試行できる | `structural_unsafe_inputs_rejected_child_empty_and_retry` | CI | 未実測 | 未実測 | 未実測 |
| 隣接 Op（`Op::Custom`）が fail-closed のまま | `structural_neighbouring_unsupported_op_stays_fail_closed` | CI | 未実測 | 未実測 | 未実測 |
| `tape.rs` のメタテスト: Interpolate の既知 7 mode が true、未知は false | `structural_family_supports_create_graph`（単体） | CI | — | — | — |
| facade 経由（代表は `dropout`・`interpolate`・`max_unpool2d`） | `crates/facade/tests/create_graph_facade.rs` に `*_create_graph_via_facade` を追加 | CI | 未実測 | 未実測 | 未実測 |
| CUDA／Metal の parity。GPU ネイティブの mode（Nearest・Bilinear）とホストフォールバックの mode（Bicubic・Area など）を最低 1 つずつ。Dropout と MaxUnpool も含める | `create_graph_structural_cuda_parity`／`create_graph_structural_metal_parity`（後者は `cfg(target_os = "macos")`）。置き場・名前は #3776・#3777 の雛形が確定するまで仮。測り方は #3739 に従う | `#[ignore]` | 未実測 | 未実測 | 未実測 |

実機の実行は #3856・#3858 が担う。Linux や開発機の代替値で pass にしない。

**既存テストで反転・差し替えが要るもの（#3793 の必須作業）**:

- `crates/autodiff/src/tape.rs` の `pooling_family_supports_create_graph`: `!MaxUnpool` を assert しているので true へ反転
- 同 `conv_transpose3d_max_unpool_op_tests::max_unpool_op_meta_properties`: `supports_create_graph()` だけ true へ（`is_checkpoint_eligible` は false のまま）
- `crates/autodiff/tests/create_graph_pool.rs` の `create_graph_still_rejects_max_unpool_after_pooling`: 成功して 1 階と bit 一致する確認へ反転（隣接 Op の拒否は上の新テストへ移す）
- `crates/autodiff/tests/max_unpool_parity.rs` の `records_exactly_one_node_and_rejects_create_graph`: 1 ノード記録の確認は残し、create_graph は成功と 1 階 bit 一致へ

隣接 Op には恒久対象外の `Op::Custom`（§21.2 A 区分。§32 の前例）を使う。`ConvTranspose3d`（#3797）・`LocalResponseNorm`（#3794）など兄弟 issue が並行して true にしうる Op は使わない。

### 35.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えない。新しい baseline は作らない（作るなら実機の実測値に限り所有者承認）
- 1 階は構造上 bit 同一（§21.1(1)。1 階の VJP には触れない）
- `cg.grad` の CPU での bit 一致の見込み: Nearest・NearestExact・MaxUnpool はコピーと f64 決定的集約なので bit 一致。固定 tap の mode は 1 階と同じ f32 積 `u·w` と同じ `scatter_with_fallback(Add)`（f64 集約）を通るので bit 一致。Area は V3 なので値が bit 一致。Dropout は keep 位置で bit 一致し、drop 位置で g が非有限のときだけ 35.10 の既知差がある
- GPU（CUDA・Metal）では `Mul`・scatter を backend カーネルで評価するので、契約は REQ-2 とし、bit 一致は主張しない
- f64 長軸縮約の契約（`coding-rust.md`）: Area の値は 1 階の `interpolate_area_vjp`（f64 アキュムレータ・最後に 1 回 downcast）を V3 でそのまま使う。固定 tap の mode は f64 集約の `ScatterReduce::Add` に乗る。FMA 契約には影響しない
- 必須ベクトル（3 段すべてで有限かつ一致。外れる場合は型付き拒否で `child` が空になることを固定する）:
  - 大きな upstream `g=3e38`・`g=[3e38, -3e38]`（Bicubic の負の重みを含む）
  - 非有限の g（drop 位置・keep 位置の両方）
  - MaxUnpool の重複 index（last-writer）と NaN・inf の入力
  - Interpolate の `outer==0`（空 batch）
  - Area の不均一な窓（例 5→3）と打ち消し合う upstream
  - Dropout の `p=1.0`（mask がすべて 0）
- 保証範囲は §27.3・§30.3 と同じ書き方にする。Area は構造判定を通った入力で、seed の各要素の絶対値が 1 以下のときに限る。三階以上の微分と、真値が f32 を超える場合（1 階と同じく `inf`）は対象外

### 35.6 互換への影響

**追加のみ**。`api-compat-0.14.0-decision.md` §8 の手順で判定する。触れる公開型が無く（`Op` は `pub(crate)`）、`AutodiffError` の variant も公開シグネチャも変わらない。以前 `Err` だった入力が `Ok` になる受理範囲の拡大であり、§1.1 の破壊的条件には当たらない。`Op` の payload の追加は要らない。

### 35.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）
- `docs/compat-api-scope.md` への適用記録は #3767 の承認後の実装 PR で書く。文案: `**適用記録（イシュー #3793・親 #3726。承認: <#3767 の回答 URL>）**: 構造変換 3 Op を create_graph の対象にした。公開名の追加なし。`
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（承認後の PR で行う。承認が無いうちは書かない）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md` を更新した（項目 3 の設計記録の欄に §35 を追記）

### 35.8 ベンチと性能への影響

既定経路（`Tape::backward`・forward）は変わらない。`supports_create_graph` は create_graph の入口でしか呼ばれない。増えるのは create_graph 経路の構造ノードと、索引・重み定数（`[outer, sp_out·K]`）のホスト確保だけなので、前後比較は不要と判断する。実装で既定経路に触れた場合は、5 回計測の中央値で前後を比べ、判定規則を計測前に固定する。実機の値は未実測。

### 35.9 依存の追加の有無

追加なし。許容 12 区分・禁止リストとの照合で変化なし。新しい `unsafe` なし。

### 35.10 族の方式と式（Op ごと）

記号: `x_m` は子テープ上の入力の写し、`g` は子テープ上の upstream。

- **Dropout（V1 + V2）**
  - R: `x_m.dropout_with_mask(mask.clone())`（`crates/autodiff/src/var.rs:4499`。`pub(crate)`）。forward と同じ `dropout_with_fallback`（`grad.rs:3935`。`ops.mul` と drop 位置のゼロ上書き）を通るので写しの値は bit 一致。**`Var::dropout`（`var.rs:4473`）は呼ばない**: マスクを引き直してグローバル RNG を消費してしまうため（`docs/rng-global-contract-design.md` §3.2）。replay は RNG に触れない
  - cgrad（推奨）: `where_cond(mask != 0, g.mul(mask_const), 0)`。keep 位置は 1 階の `vjp_elementwise_mul`（`grad.rs:84`）と同じ IEEE 乗算 1 回なので bit 一致。drop 位置は常に 0 で、g が非有限のとき 1 階（`inf·0 = NaN`）と違う。§21.2 の「記録済みマスクの `where_cond`」・§21.1(3) の `inf*0` 規則・forward のゼロ出力契約に揃う。代替は素の `g.mul(mask_const)`（1 階と完全に bit 一致し NaN も 1 階と同じ）で、35.14 の論点 2 に載せる
  - 1 階の非有限 upstream の扱いは変えない
  - `Dropout2d`・`AlphaDropout` も同じ `Op::Dropout` を記録するので追加作業なしで対象になる（AlphaDropout の定数バイアス加算は `Add`）
- **Interpolate**
  - R: `x_m.interpolate(&size, mode)`（`var.rs:4610`）。payload で足りる
  - Nearest・NearestExact（V1）: `g → contiguous → reshape [outer, sp_out] → zeros[outer, sp_in].scatter_add(1, index, ·) → reshape input_shape`。index は 1 階と同じ `nearest_src_index_map`（`grad.rs:5502`）・`nearest_exact_src_index_map`（`grad.rs:5620`）から作る
  - Linear・Bilinear・Trilinear・Bicubic（V1。K = 2／4／8／16）: `g → reshape [outer, sp_out, 1] → broadcast_to [outer, sp_out, K] → contiguous → reshape [outer, sp_out·K] → mul(weight_const) → zeros.scatter_add(1, index, ·) → reshape`。index と weight は 1 階と同じ `*_src_index_and_weight_map`（forward と共有する座標式）から作り、必要なものは `pub(crate)` にする（本体と数値は変えない）。**重み定数との `Mul` を使ってよい理由**: 1 階 VJP（`interpolate_fixed_tap_scatter_vjp`、`grad.rs:5812`）自身が同じ f32 積 `u·w` を作ってから同じ f64 集約の scatter_add を通るので、§21.1(3) が禁じる「選択の 0/1 マスクを掛ける」形には当たらない（Bicubic の負の重みや 0 の重みでも 1 階と同じ積になる）
  - Area（V3）: 値は 1 階の `interpolate_area_vjp`（`grad.rs:5876`。`pub(crate)` にする）を `push_v3` で登録する。構造は §30 の Adaptive と同じ窓の分類ごとの `gather → 1/s → scatter_add → s/d` で、`pool.rs` の `plan_adaptive2d`（:273）・`window_scatter_structure`（:385）・`check_structure_contract`（:443）を空間 N 次元へ一般化して共用する。§30.3(d) と同じく、divisor の違う分類が混ざる入力で REQ-2 を外れるものや、集約前の中間和が f32 を超えるものは型付きエラーで拒否する。代替（Area だけ fail-closed のまま残す）は 35.14 の論点 3
  - 共通の拒否: `outer == 0` は 1 階と同じくゼロの定数を返す（大きな確保をしない）。`sp_in_numel > i32::MAX` と `outer·sp_out·K` の桁あふれは確保の前に拒否する（1 階 `grad.rs:2887` 付近と同じ順序）
- **MaxUnpool（V1 + V2）**
  - R: `max_unpool_ops` に `pub(crate)` の入口を新設する。引数は `(x_m, index, out_shape)` で、`replay_op` が受け取るノード shape を使う（`pool::replay_max_pool(&mirror, &index, shape)`〈`create_graph/pool.rs:58`〉と同じ形）。index の範囲検査と `scatter_with_fallback(Overwrite)`（`grad.rs:4732`）を通して `Op::MaxUnpool` を積む。payload の変更は要らない。公開入口（`max_unpool1d/2d/3d`）は kernel・stride・padding を要求するので replay には使わない
  - cgrad: `where_cond(last_writer_mask, g_flat.gather(1, index_flat), 0)` を `[N·C, in_plane]` から入力 shape へ戻す。これは `index_cum::scatter_vjp`（`create_graph/index_cum.rs:119`）の Overwrite の `d_src` の枝そのものなので、`ScatterSpec { input_shape: flat_out, src_shape: flat_in, dim: 1, index, reduce: Overwrite }`・`need_input=false` で再利用する。gather はコピーで、選択は 1 階の `elementwise_mul_mask`（`grad.rs:6773`。選択であって乗算ではない）と同じ規則なので bit 一致
  - 重複 index の意味論は 1 階どおり last-writer（PyTorch とは違うが `docs/autodiff-conv-transpose3d-max-unpool-decision.md` で決定済み）。新しい論点にはしない
  - 空（`nc==0`・`in_plane==0`・`out_plane==0`）はゼロの定数

### 35.11 閉包性と族間の依存

- 子テープへ積む Op: `Reshape`・`Contiguous`・`BroadcastTo`・`Mul`・`Where`・`Gather`・`Scatter`（Add）と、forward の写しの `Dropout`・`Interpolate`・`MaxUnpool`。基準時点の `supports_create_graph`（`tape.rs:2526`）で前の 7 つが true であることを確かめた
- 他族への依存は #3083（索引系 `Gather`／`Scatter`。§25）と #3085（`pool.rs` の窓構造。§30）で、どちらもマージ済み
- 孫テープの再生では、子テープ上の写し（`Dropout`・`Interpolate`・`MaxUnpool`）が再生されるので、3 Op 自身が true であることが要る（族の中で閉じる）

### 35.12 fail-closed・拒否テスト・GPU の forward

- §21.6 の判定順・網羅 match・未知 variant を false にする規約は弱めない
- 隣接 Op は `Op::Custom`（35.4）
- CUDA／Metal の forward: `Interpolate` は Nearest と Bilinear だけ GPU カーネルを持ち（`crates/backend-cuda/src/ops.rs`・`crates/backend-metal/src/ops.rs`）、ほかの mode は `Unsupported` を受けてホストの参照実装へフォールバックする。`Dropout` は `ops.mul`（GPU）のあとホストでゼロ上書きする。`MaxUnpool` は `scatter_with_fallback(Overwrite)` を通る。本設計は forward も GPU カーネルも変えない。子テープの構造ノードは子テープの backend で評価される。GPU での一致は未実測で、#3856・#3858 が `#[ignore]` テストで確かめる（35.4）

### 35.13 セキュリティ観点・スコープ外・Closes／Refs

- A03: shape・`size`・`mode`・index は forward 側の確保前検証を通ったものだけが届く。そのうえで子テープへ書く前に i32 範囲・確保長の桁あふれ・index 範囲を検査する。新しい確保は `[outer, sp_out·K]` の索引・重み定数と Area の構造判定の f64 一時領域に限る
- A04: 拒否時に子テープを空に保つ fail-closed、網羅 match、未知 `InterpolateMode` を false にする規約を維持する。Dropout の replay がグローバル RNG を消費しないことを契約とテストで固定する（学習の再現性の破れを防ぐ）
- A06: 依存を追加しない。A08: 判定・閾値・tolerance の迂回経路を作らない。秘密情報・ホスト名・個人のパスは書かない
- **スコープ外**: GPU 実機の parity（#3856・#3858）、測り方（#3739）、1 階の Dropout の非有限 upstream の意味論、MaxUnpool の重複 index を PyTorch に寄せること、`InterpolateMode` の新 variant、f64 テープの create_graph（#3722）。新しい issue は承認なしに起票しない
- `Closes #3726`。`Refs #3793 #3767 #3719`

### 35.14 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 族の採否 | 3 Op を実装する／対象外にする | 実装する | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 2. Dropout の drop 位置で g が非有限のときの cgrad | `where_cond` で 0（§21.2 の記載どおり）／素の `mul` で 1 階と bit 一致（NaN） | `where_cond` | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 3. Interpolate の Area | V3（§30 の窓構造を N 次元へ一般化し、判定外れは拒否）／Area だけ fail-closed のまま残す | V3 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 4. MaxUnpool の replay | `max_unpool_ops` に out_shape を受ける `pub(crate)` 入口を新設して `Op::MaxUnpool` を積む／`Scatter(Overwrite)` の合成で再生する | 新設の入口 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 5. GPU parity の判定方式 | #3739 の推奨に従う（tolerance は変えない）／別方式 | #3739 に従う | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |

### 35.15 変えていないもの

コード・公開 API・rustdoc・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・`docs/compat-api-scope.md`。

## 36. 決定記録（イシュー #3727・親 #3719。create_graph の C 区分・正規化・再パラメータ 3 Op）

### 36.0 位置づけ・承認根拠・境界・推奨

- **状態**: 設計記録のみ。承認は #3767 の回答待ちで、本節は何も承認しない。回答があるまでコード・rustdoc・公開 API・`docs/compat-api-scope.md` は変えず、`supports_create_graph` もまだ `true` にしない
- **基準**: `origin/main` の `6402d6087`。以下の `file:line` はこの時点で確かめた値で、実装 PR の前に引き直す
- **対象**: §21.2 の C 区分（族を割り当てず fail-closed に固定した 24 Op）のうち、正規化・再パラメータの 3 Op（`Op::LocalResponseNorm`・`Op::WeightNorm`・`Op::SpectralNorm`）。§21.2 の C 表ではこの 3 Op の方式欄が空で、本節が埋める。3 Op は `supports_create_graph`（`crates/autodiff/src/tape.rs:2526`）の網羅 match の false 側（`tape.rs:2667`〜`:2669`）にある
- **根拠**: §21.1（共通方式 V1〜V4・S1〜S3・閉包性）、§21.6（fail-closed 契約）、前例は §27（S1 と `check_structure_stages`）、§33（Sqrt 入力側の保護・2 のべき乗スケール）、§21.2 の `Det`（大きな指数を 1 個の f32 定数にしないチャンク分割）。ルート #3711 の「実装を既定に設計する」趣旨は設計方針の根拠であり、承認ではない
- **後続**: 実装は #3794、承認は #3767、実機 parity は #3856（`<cuda-node>`）・#3858（`<metal-node>`）、雛形は #3776・#3777、測り方は #3739
- **推奨**: **実装する（3 Op すべて）**。理由は次の 4 つ
  1. 1 階 VJP がすべてホストの f64 関数で、V3 の値にそのまま使える（`crates/autodiff/src/grad.rs:3536`〈LRN → `local_response_norm_vjp_host`。`crates/tensor-core/src/lrn.rs:200`〉・`:3560`〈WeightNorm → `weight_norm_vjp_host`。`weight_reparam.rs:219`〉・`:3590`〈SpectralNorm → `spectral_norm_vjp_host`。`weight_reparam.rs:422`〉）
  2. 構造に要る Op がすべて既に replayable（36.11）
  3. S1 と総括判定の前例がある（§27.3・§33.12）
  4. `weight_norm`・`spectral_norm` は GAN の判別器や勾配ペナルティ（二階微分が要る典型例）で使われ、LRN は CNN で使われる。拒否されたままだと `hvp`／`jvp`／`jacfwd` が使えない実害がある

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | 3 Op を実装する | 勾配ペナルティ・正規化層を含む損失の HVP が使える。追跡表の項目 3 が「ある」に近づく | 実装と保守の費用。総括判定の追加 |
| B | 対象外にして fail-closed を恒久契約にする | 費用ゼロ | 上記の損失で高階変換が使えないまま残る |

### 36.1 公開形

該当なし。新しい公開名は増えない。`Tape::backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` は公開済みで、受理する Op の範囲が広がるだけである。`Var::{local_response_norm, weight_norm, spectral_norm}` と `SpectralNormState` は #2851 で公開済みで、形は変えない。

`crates/facade/tests/api_surface.rs` には、この 3 Op の create_graph を対象にした否定ガードも保留 doctest ガードも無く、反転するものは無い。`LrnWeightReparamHoldDoctestGuard`（`crates/facade/src/lib.rs:7044`）が固定しているのは `norm_except_dim` や層 struct などの保留名で、create_graph とは無関係なので反転せず残す。実装 PR（#3794）で変える箇所:

- facade の rustdoc にある「対象 Op の正本は §21.2・§22〜§31」（`crates/facade/src/lib.rs` の 6 か所）の範囲に §36 を加える
- `Op` の doc コメント（`tape.rs:1378` 以降の「高階微分非対応」の記述）と、`lrn_ops.rs`・`weight_reparam_ops.rs` 冒頭の「高階微分（create_graph）は対象外」
- `supports_create_graph` の網羅 match で 3 Op を false 側から true 側へ移す。ワイルドカードは足さない
- `docs/autodiff-lrn-weight-reparam-decision.md` の「create_graph 対象外」の記述への追記（実装 PR の作業。本 PR では書き換えない）

### 36.2 rustdoc と doctest

- 新しい公開 API が無いので必須の doctest は無い。任意で facade の `backward_create_graph` に CPU だけで通る `weight_norm` の HVP 例を 1 つ足す。GPU の例は置かない（置くなら `no_run`／`#[ignore]`）
- 失敗系は `compile_fail` に頼らず、ランタイムの `Err(AutodiffError::Backward(_))` 一致と `child.is_empty()` で確かめる

### 36.3 エラー型

- 新しい variant は足さない。総括判定で外れた入力、`d_s <= 0` のレーン、ゼロノルムのグループ、`σ` が 0 または非有限、非有限の upstream の拒否は `AutodiffError::Backward(String)` にする
- 検証順: 判定は子テープへ書く前に行う（`validate_ancestors` か cgrad 構築の冒頭）。拒否したら `Tape::backward_create_graph` のロールバックで `child` を空に戻し、再試行できるようにする（§27.3 と同じ）。shape・`dim`・`LrnParams` は forward 側の確保前検証を通ったものだけが届く。本番経路で `unwrap`／`expect` を使わない

### 36.4 テスト計画

候補名は案であり、実装 PR で確定する。CI の行は新規 `crates/autodiff/tests/create_graph_norm_reparam.rs`（仮名）に置く。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|--------------|--------|------|---------------|----------------|----------------|
| 1 階が素の `backward` と bit 同一、`cg.grad` が REQ-2 で一致（V3 は CPU で bit 一致も見る） | `norm_reparam_first_order_bit_identical_and_cgrad_req2` | CI | 未実測 | 未実測 | 未実測 |
| ヘッシアンが有限差分と一致（LRN: `size` 奇数・偶数 × β ∈ {0.5, 0.75, 1, 2} × k ∈ {1, 2}。WeightNorm: `dim` None／0／末尾 × v と g の混合二階項。SpectralNorm: `dim` 0／1 × rank 2〜4。5 点中央差分 h=1e-2 は §27.4 の前例どおり。判定式は変えない） | `norm_reparam_hessian_matches_finite_difference` | CI | 未実測 | 未実測 | 未実測 |
| 必須ベクトルが 3 段（1 階・`cg.grad`・`jvp_create_graph` 経由の孫テープ再生）で有限か、拒否されて `child` が空 | `norm_reparam_mandatory_vectors_finite_in_three_stages` | CI | 未実測 | 未実測 | 未実測 |
| SpectralNorm の replay と `hvp` が power iteration を回さず、`SpectralNormState` の `u`・`v` を変えない（training／eval の両方） | `spectral_norm_create_graph_does_not_advance_power_iteration_state` | CI | 未実測 | 未実測 | 未実測 |
| 安全でない入力の拒否・`child` が空・再試行できる | `norm_reparam_unsafe_inputs_rejected_child_empty_and_retry` | CI | 未実測 | 未実測 | 未実測 |
| 構造の全経路が有限か拒否（§27.4 と同型の掛け合わせ。受理と拒否の両方が出ることも確かめる） | `norm_reparam_sweep_every_structure_path_is_finite_or_rejected` | CI | 未実測 | 未実測 | 未実測 |
| 隣接 Op（`Op::Custom`）が fail-closed のまま | `norm_reparam_neighbouring_unsupported_op_stays_fail_closed` | CI | 未実測 | 未実測 | 未実測 |
| facade 経由（代表は `weight_norm`・`spectral_norm`） | `crates/facade/tests/create_graph_facade.rs` に `*_create_graph_via_facade` を追加 | CI | 未実測 | 未実測 | 未実測 |
| CUDA／Metal の parity（forward がホストへフォールバックすることを含む。36.12） | `create_graph_norm_reparam_cuda_parity`／`create_graph_norm_reparam_metal_parity`（後者は `cfg(target_os = "macos")`）。置き場・名前は #3776・#3777 の雛形が確定するまで仮。測り方は #3739 に従う | `#[ignore]` | 未実測 | 未実測 | 未実測 |

**差し替える既存テスト（#3794 の作業）**: 3 Op が true になると次の 3 件が壊れる。

1. `crates/autodiff/src/tape.rs:5817` の `lrn_weight_norm_spectral_norm_op_meta_properties`: `!supports_create_graph()` の検査を `supports_create_graph()` に直す（checkpoint 対象外と `for_each_input` の検査は残す）
2. `crates/autodiff/tests/create_graph.rs:4674` の `batch_norm_neighbor_local_response_norm_stays_rejected`: 上流の拒否見本を LRN から恒等の `Op::Custom` へ替え、shape `[1,4,2,2]` を保つ。テスト名も直す
3. `crates/autodiff/tests/create_graph_norm.rs:582` の `neighbouring_unsupported_norm_op_stays_fail_closed`: 同じく `Op::Custom` に替える

隣接 Op に `MatrixNorm` を使わない理由: §33 が既に使っており、#3734（B 区分の線形代数）が設計中で true に変わりうる。`Op::Custom` は §21.6 が基準テストに使う恒久的な対象外である。実機の実行は #3856・#3858 が担い、Linux や開発機の代替値で pass にしない。

### 36.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えない。新しい baseline は作らない（作るなら実機の実測値に限り所有者承認）
- 1 階は構造上 bit 同一（§21.1(1)）。3 Op とも V3 なので `cg.grad` の値も CPU で 1 階と bit 同一。CUDA／Metal も 1 階 VJP がホストで同じ関数を通るが、主張は REQ-2 の範囲までにする（実機は未実測）
- f64 長軸縮約の契約（`coding-rust.md`）は V3 の値側で守る。LRN の二乗和は先に f64 へ上げてから二乗し、WeightNorm・SpectralNorm の内積は要素積を f32 で確定してから f64 へ上げる。構造側の f32 合成は値を決めない。FMA 契約には触れない
- 必須ベクトル（3 段すべてで有限かつ一致。外れる場合は総括判定で拒否され `child` が空になることを固定する）:
  - LRN: `x=1e20`（f32 の二乗が overflow する）、`k=0` かつ窓内が全ゼロ（`d=0`）、`α<0` で `d<0`（β が非整数なら 1 階は NaN）、小さな `x=[1e-30, …]`、`g=3e38`・`g=[3e38, -3e38]`、偶数 `size` の非対称窓
  - WeightNorm: ゼロノルムのグループ（1 階は `0·inf=NaN`）、`dim=None`、`v=1e20`、小スケールの `v`、大きな `g`・upstream
  - SpectralNorm: `σ=0`（1 階は NaN／inf を伝播）、極小の `σ`、`W=1e20`、`eps` で clamp された `u`・`v`、training と eval の両方
- 保証範囲は §27.3 と同じ書き方にする。総括判定を通った入力で、seed の各要素の絶対値が 1 以下のときに限る。三階以上の微分と、真値が f32 を超える場合（1 階と同じく `inf`）は対象外

### 36.6 互換への影響

**追加のみ**。`api-compat-0.14.0-decision.md` §8 の手順で判定する。触れる公開型が無く（`Op` は `pub(crate)`）、`AutodiffError` の variant も公開シグネチャも変わらない。以前 `Err` だった入力が `Ok` になる受理範囲の拡大であり、§1.1 の破壊的条件には当たらない。payload（`params`・`dim`・`u`/`v` スナップショット）は再生に必要な値を既に持っており、追加は要らない。`SpectralNormState` の公開形も変えない。

### 36.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）
- `docs/compat-api-scope.md` への適用記録は #3767 の承認後の実装 PR で書く。文案: `**適用記録（イシュー #3794・親 #3727。承認: <#3767 の回答 URL>）**: 正規化・再パラメータ 3 Op を create_graph の対象にした。公開名の追加なし。`
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（承認後の PR で行う。承認が無いうちは書かない）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md` を更新した（項目 3 の設計記録の欄に §36 を追記）

### 36.8 ベンチと性能への影響

既定経路（`Tape::backward`・forward）は変わらない。`supports_create_graph` は create_graph の入口でしか呼ばれない。増えるのは create_graph 経路の構造ノード（LRN は窓幅に比例して O(size) 個の Pad／Narrow／Add、ほかは十数ノード）と総括判定のホスト走査 O(numel) だけである。既定経路に触れないので前後比較は不要と判断する。実装で既定経路に触れた場合は、5 回計測の中央値で前後を比べ、判定規則を計測前に固定する。実機の値は未実測。

### 36.9 依存の追加の有無

追加なし。許容 12 区分・禁止リストとの照合で変化なし。新しい `unsafe` なし。

### 36.10 族の方式と式（Op ごと）

記号は §33.10 に揃える。`x_m` は子テープ上の入力の写し、`g` は子テープ上の upstream、`g_b` は `g` を縮約軸へ broadcast したもの。V3 の値は 1 階のホスト関数を `pub(crate)` にして使う（本体・数値は変えない）。結合は `push_v3`（`crates/autodiff/src/create_graph.rs:986`）。

- **LocalResponseNorm（V3 + S1 + 総括判定）**
  - 1 階の式（`lrn.rs:14-31`）: `dx_m = g_m·d_m^{−β} − (2αβ/size)·x_m·Σ_{c∈W'(m)} g_c·x_c·d_c^{−β−1}`、`d_c = k + (α/size)·Σ_{W(c)} x²`。窓は前方 `W(c) = [c−⌊size/2⌋, c+⌊(size−1)/2⌋]`、第 2 項は反転窓 `W'(m) = [m−⌊(size−1)/2⌋, m+⌊size/2⌋]` で、偶数 `size` では非対称になるため別々に作る
  - **R**: `Var::local_response_norm(x_m, size, alpha, beta, k)`。`LrnParams` のアクセサから値を取る。forward は子テープ上でもホストの共有カーネルへ落ちるので値は同じ
  - **構造**: `[N, C, *S]` を `[N, C, inner]` へ Reshape し、レーンを `(n, inner)` 列（チャネル方向）とする。レーンごとの 2 のべき乗スケール `s` をホスト定数で決め、`x_s = x_m/s`、`d_s = k/s² + (α/size)·Σ x_s²`（`d = s²·d_s`）とする。窓和は、チャネル軸を `Pad`（値 0）してから `Narrow` でずらして `Add` で作る。ずらしのオフセット範囲は、窓（前方窓 `W` と反転窓 `W'` それぞれ）のオフセット範囲を `[−(C−1), C−1]` と交差させたもので、空になる寄与だけを除く（`size` を `C` で打ち切らない。例: `C=3`・`size=5` では `−2..+2` の 5 オフセットすべてが非ゼロ寄与を持ち、`lrn.rs` の境界切り詰めと一致する）。`size<=C` の通常ケースに加え `size>C` も 36.12 の確認対象に含める。この窓和を使うと `dx = s^{−2β} · [ g⊙d_s^{−β} − (2αβ/size)·x_s ⊙ Σ_{W'} g·x_s·d_s^{−β−1} ]` となり、スケール `s^{−2β}` は最後に 1 回掛ければよい。冪は `PowScalar(−β)`・`PowScalar(−β−1)`。0/1 定数の行列を掛ける方式は §21.1(3) の `inf*0` 規則に触れるので採らず、累積和の差分も非負の和の桁落ちになるので採らない（`lrn.rs` が直接加算を選んだ理由と同じ）
  - **`s` の規則**: §21.1(5) の eps 規則を写し、`s = pow2ceil(max(max|x|, √(|k|·size/|α|)))`。指数は [−126, 127] にクランプし、全ゼロのレーンは `s=1`。`α=0` は `y = x·k^{−β}` に縮退し構造は `x` の線形（V1）で足りるので別分岐にする
  - **定数の指数**: `s^{−2β}` は β が非整数だと 1 個の f32 定数では範囲を外れうる。§21.2 の `Det` と同じく 1 個の f32 定数にせず、±2^100 以下の 2 のべき乗チャンクに分けて順に `Mul` する。2 のべき乗にならない端数は 1 個の定数にする
  - **拒否**: `d_s <= 0` のレーン（`k<=0` で窓が全ゼロ、または `α<0`）は拒否する。`LrnParams::new` は `size==0` と非有限の α/β/k だけを拒否し（`lrn.rs:55-75`）、`k<=0` と負の α は受理して結果へ伝播する。1 階は NaN／inf を伝播するが、構造の `PowScalar` の微分係数が `0×inf` を生むため。代替（V2 でマスクして 1 階の値を写す）は 36.14
  - 値は `local_response_norm_vjp_host`
- **WeightNorm（V3 + S1。2 入力）**
  - 1 階の式（`weight_reparam.rs:15-34`）: `dot = Σ up·v`、`dg = dot/n`、`dv = (g/n)·up − (g·dot/n³)·v`（`n` はグループのノルム）
  - **R**: `Var::weight_norm(v_m, g_m, dim)`
  - **構造**: `dim` を先頭へ `Permute` し `[groups, rest]` へ `Reshape` する（`dim=None` は `[1, numel]`）。グループごとの 2 のべき乗スケール `s` で `v_s = v_m/s`、`n_s = Sqrt(Sum(v_s²))`、単位ベクトル `v̂ = v_s/n_s` を作る。`dg = Σ(up⊙v̂)`、`dv = (g_b/(s·n_s))⊙(up − dg_b⊙v̂)`。`v̂` はスケール不変で各要素の絶対値は 1 以下、`1/s` は 2 のべき乗なので厳密に表せる
  - `build_cgrads` は `v` と `g` の**両方**へ寄与を返す（§21.3 #3082 の「入力を定数化しない」と同じ）。`g` から `v` への混合二階項を保つため、`g_m` は子テープ上の Var のまま使う
  - **ゼロノルム**: §33.10 の VectorNorm L2 と同じく、`Sqrt` に入る二乗和を先に `where_cond(mask, 1, …)` で正の定数へ置き換え（`Sqrt` の微分係数 `0.5/y` が `y=0` で `0×inf=NaN` になるのを防ぐ）、そのあとで出力側を処理する。推奨はゼロノルムのグループを拒否すること（1 階の forward 自体が NaN）。代替（V2 で 1 階の NaN／inf を写す）は 36.14
  - 値は `weight_norm_vjp_host` の `(dv, dg)` を、それぞれ `push_v3` で入れる
- **SpectralNorm（V3 + S1。power iteration は定数）**
  - 1 階の式: `σ = uᵀ W_mat v`、`dW = up/σ − (s/σ²)·u vᵀ`、`s = Σ up·W`。`u`・`v` は forward 時点の**非追跡スナップショット**（`tape.rs:1378` 以降の payload `{weight, u, v, dim}`、`grad.rs:3590`）で、1 階 VJP は両者を定数として扱う
  - **冪乗法の扱い（イシューの論点）**: **定数として扱う**。1 階 VJP が既に `u`・`v` を detach したスナップショットとして扱っており、PyTorch も `_power_method` を no_grad で回し buffer を clone する。反復を微分すると 1 階と別の関数を微分することになり、「1 階 bit 同一」の契約と PyTorch の二階微分の意味論の両方を壊す。真の最大特異値の微分（SVD の微分）にする案も同じ理由で採らない
  - **R**: 公開の `spectral_norm(&mut state, training)`（`crates/autodiff/src/weight_reparam_ops.rs:296`）は使わない。training のときは power iteration を回して状態を書き戻すため、孫テープの再生で状態が進んでしまう。`pub(crate)` の再生用入口（例 `spectral_norm_from_snapshot(weight, u: &Tensor, v: &Tensor, dim)`）を新設する。この入口は layout の検査、`spectral_norm_forward`（またはホストの forward）、同じスナップショットで `Op::SpectralNorm` を積むことだけを行う。**replay は power iteration を回さず、どの `SpectralNormState` にも触れない**ことを必須要件とする。代替は `W_m/σ_m` を `Mul`（ホスト定数 `uvᵀ`）・`Sum`・`Div` で合成する案（Op 自身を積まない）で、36.14 に載せる
  - **構造**: `uvᵀ` を W の shape へ並べたホスト定数 `P` を用意する（`dim` を先頭にした `W_mat` の並びを逆 permute する）。2 のべき乗スケール `s ≥ max|W|` で `W_s = W_m/s`、`σ_s = Sum(W_s⊙P)`、`Ŵ = W_s/σ_s` とする。cgrad は `(1/(s·σ_s))⊙(up − Sum(up⊙Ŵ)·P)`。`u`・`v` が正規化済みなら `P` の各要素の絶対値は 1 以下
  - **拒否**: `σ_s` が 0 または非有限（1 階は伝播する）
  - 値は `spectral_norm_vjp_host`

### 36.11 閉包性と族間の依存

- 子テープへ積む Op: `Reshape`・`Permute`・`Contiguous`・`Pad`・`Narrow`・`BroadcastTo`・`Sum`・`Mul`・`Add`・`Where`・`ScalarBinary`（Sub・Div）・`ScalarUnary`（Sqrt・PowScalar・Neg・Abs）。基準時点で `supports_create_graph`（`tape.rs:2526`）と `scalar_unary_replayable`（`create_graph.rs:145`）で true になっていることを確かめた。`ScalarBinary` は実装前に origin/main で再確認する
- 他族への依存は #3084 の `Pad`（§29。マージ済み）だけ。3 Op はどれも構造に自分自身を積まない（SpectralNorm の R 用入口は Op 自身を積むが、族の中で閉じる）
- 孫テープの再生では子テープ上にある 3 Op の forward の写しが再生されるので、3 Op 自身が true であることが要る。3 Op は互いに依存しないので、#3794 の中でどの順に実装してもよい

### 36.12 fail-closed・総括的な適用判定・拒否テスト

- **総括判定**: `check_structure_stages`（`crates/autodiff/src/create_graph_norm.rs:338`）と同型の判定を族全体に置く。構造の forward の各段、二階の逆伝播の「cotangent × 微分係数」（`PowScalar` の `−β·d_s^{−β−1}`、`Div` の `−(a/b)/b`、`Sqrt` の `0.5/y`）、境界の cotangent の上界をホストが f64 で求め、`2^125` 以上か非有限になる段があれば拒否する。閾値と seed の絶対値 1 以下という前提は §27.3 と同じ。個別の条件を足していく方式は、§27.3 でレビューが収束しなかった経緯から採らない
- **入口での拒否**: 非有限の upstream は §27.3 と同じく拒否する。非有限の `x`／`v`／`g`／`W` は §33.14 の論点 3 と同じ規則（NaN は伝播し ±inf は拒否）に合わせる。LRN の `d_s<=0`、ゼロノルム、`σ=0` は 36.10 のとおり拒否する。いずれも 36.14 で問う
- **CUDA／Metal の forward**: 3 Op とも `BackendOps::{lrn_forward, weight_norm_forward, spectral_norm_forward}`（`crates/tensor-core/src/backend_ops.rs:2669`〜`:2708`）の既定は `Unsupported` で、override しているのは `backend-cpu` だけである。CUDA／Metal／ROCm にデバイスカーネルは無く、forward も 1 階 VJP もホストの共有カーネルで計算する。create_graph の構造ノードも既存のホスト再生経路で評価し、GPU カーネルを新設しない。#3856・#3858 が測るのはデバイスからホストへの実体化を含む経路の parity であり、カーネル間の parity ではない。値は未実測
- **LRN の `size>C` ケース**: `C=3`・`size=5`（偶数 `size` では非対称窓も）で、前方窓・反転窓の窓和と二階微分が `lrn.rs` の境界切り詰めと一致することを、実装時の数値テストに含める（打ち切り方式との差が出る形状）
- §21.6 の判定順・網羅 match・未知 variant を false にする規約は弱めない

### 36.13 セキュリティ観点・スコープ外・Closes／Refs

- A03: shape・`dim`・`LrnParams` は forward 側の確保前検証を通ったものだけが届く。総括判定はホスト走査で、新しい確保は O(numel) の f64 一時領域と、`P`（W と同じ shape）に限る。LRN のずらし回数は、前方窓・反転窓それぞれのオフセット範囲を `[−(C−1), C−1]` と交差させた個数（最大 `2C−1` 回）に抑える。`size` が `C` より大きくても、窓は境界で切り詰められるだけで寄与は消えないため、`min(size, C)` のような打ち切りはしない
- A04: 拒否時に子テープを空に保ち、fail-closed を維持する
- A06: 依存を追加しない。A08: SpectralNorm の状態が create_graph で変わらないことをテストで固定する（36.4）。秘密情報・ホスト名・個人のパスは書かない
- **スコープ外**: GPU 実機の parity（#3856・#3858）、測り方（#3739）、f64 テープの create_graph、S2 の実装（拒否範囲を狭めるかは 36.14）、`Sequential::add_weight_norm`／`add_spectral_norm`・`norm_except_dim` の公開。新しい issue は承認なしに起票しない
- `Closes #3727`。`Refs #3794 #3767 #3719`

### 36.14 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 族の採否 | 3 Op を実装する／対象外にする | 実装する | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 2. SpectralNorm の power iteration | 定数として扱う（1 階・PyTorch と同じ）／反復を微分する／SVD の微分 | 定数 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 3. SpectralNorm の R | `pub(crate)` のスナップショット再生入口／`Mul`・`Sum`・`Div` の合成で Op を積まない | スナップショット再生入口 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 4. LRN の `d_s<=0`・WeightNorm のゼロノルム・`σ=0` | 型付きエラーで拒否／V2 で 1 階の NaN／inf を写す | 拒否 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 5. 非有限な入力 | §33.14 の論点 3 と同じ規則（NaN は伝播し ±inf は拒否）／全部拒否 | §33 と同じ規則 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 6. 総括判定で外れる入力 | 型付きエラーで拒否（§27.2 と同じ）／S2 を作って受理 | 型付きエラーで拒否 | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 7. LRN の窓和の構造 | `Pad` + ずらした `Narrow` + `Add`／0/1 の帯行列との MatMul／Cumsum の差分 | `Pad` + `Narrow` | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 8. GPU parity の判定方式 | #3739 の推奨に従う（tolerance は変えない）／別方式 | #3739 に従う | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |

### 36.15 変えていないもの

コード・公開 API・rustdoc・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・`docs/compat-api-scope.md`・`docs/autodiff-lrn-weight-reparam-decision.md`。

## 37. 設計記録（イシュー #3728・親 #3719。create_graph の C 区分・RNN セル 4 Op）

### 37.0 位置づけ・承認根拠・境界・推奨

- 状態: **設計記録のみ**。承認は #3767 の回答待ちで、本節は承認を代行しない。コード・公開面・tolerance・baseline は変えない。基準 sha は `b0a48b25d`。
- 対象 4 Op: `RnnCell`・`LstmCell`・`LstmHidden`・`GruCell`。いずれも `crates/autodiff/src/tape.rs` の `supports_create_graph`（:2526。RNN の行は :2634〜2637）で `false` の腕にあり、`backward_create_graph` は型付きエラーで拒否する（fail-closed）。
- 根拠: §20（D-A）、§21.1、§21.2 の C 区分表、§21.6、ルート #3711 の方針。これらは設計方針の根拠であって承認ではない。
- 推奨: **A. 実装する（4 Op すべて）**。

| 案 | 内容 | 評価 |
|---|---|---|
| A（推奨） | 4 Op を `Var::rnn_cell`／`lstm_cell`／`gru_cell` への自己再生で子テープに写し、cgrad を primitive の合成で書く | payload の追加が不要。既定経路は不変 |
| B | 対象外にし、fail-closed を恒久契約にする | RNN を含むモデルの `hvp`・`jvp`・`jacfwd` が拒否され続ける |
| C | モジュールに非 fused モードを足し、利用者に primitive 経路を渡す | 公開面の追加になるため別承認。推奨しない |

- §21.2 は RNN セルを「恒久寄り。利用者が replayable な primitive で手組みすれば create_graph は通る」とする。fused Op は性能のためにあり、その言い分には理がある。ただし次の事実が反対側にある。
  - facade は `GateParams` を再エクスポートしていない（`fandhe_ai_autodiff` の `lib.rs:430` にだけある）ので、facade 利用者は `Var::rnn_cell` 等を直接組めない。
  - facade の RNN 系公開 API（`Tape::rnn_forward_seq`／`lstm_forward_seq`／`gru_forward_seq`、stacked 系、packed 系。`crates/facade/src/lib.rs:1779〜1864`）はどれも fused Op を積む。
  - よって facade 利用者には手組みの経路が事実上無く、RNN を含むモデルの二階微分が拒否される実害が残る。
- A を推す理由:
  - 1 階 VJP が「pointwise 因子 × アフィン（GEMM）」で、どちらも対象の Op で書ける。
  - payload の追加が要らない。
  - 他の C 区分の族に依存しない。
  - 既定経路（1 階 `backward`）は変わらない。
- 境界: 統計縮約（#3725）・選択縮約（#3724）・構造変換（#3726）・正規化（#3727）は扱わない。実装は #3795（`RnnCell`・`GruCell`）と #3796（`LstmCell`・`LstmHidden`）に分ける。
- 前提となるコードの事実（基準 sha の値。実装 PR の前に引き直す）:
  - payload は、`RnnCell` が NodeId だけ、`LstmCell` が `gates_ifg [B,3H]`、`LstmHidden { cell, gate_o [B,H] }`、`GruCell` が `gates_rzn [B,3H]` と `q [B,H]`。いずれも追跡しない Tensor（`tape.rs:771〜830`）。
  - push 順序の契約: `Var::lstm_cell` は `LstmCell` → `LstmHidden` をこの順で続けて push し `(h_t, c_t)` を返す（`var.rs:6528`・:6625・:6638）。1 階 VJP の `LstmHidden` 腕は `nodes[cell].op` が `LstmCell` でなければ型付きエラーにする（`grad.rs:2085〜2118`）。
  - `for_each_input` では `LstmHidden` の入力は `cell` だけ（`tape.rs:2452`）。VJP は o ゲート経由で x・h_prev・w_ih・w_hh・b にも寄与する。
  - `is_checkpoint_eligible` にも 4 Op が並ぶ（`tape.rs:1854`・:1954〜1957）。本件では変えない。

### 37.1 公開形

- 該当なし。新しい公開名は無い。変わるのは受理する Op の範囲だけ。
- `api_surface.rs` に RNN の create_graph を拒否する否定ガードや保留 doctest ガードは無い（nn::rnn の公開面ガードは層の公開範囲を見るもので無関係）。正ガードの代わりは 37.4 の facade 挙動テスト `*_create_graph_via_facade` とする。
- 実装 PR で変える箇所: `supports_create_graph` で 4 Op を `true` 側へ移す（ワイルドカードは足さない）、`Op` の doc コメント、facade の rustdoc（`lib.rs:910` 付近ほか）が指す「§21.2・§22〜§31」の範囲に本節を加える。

### 37.2 rustdoc と doctest

- 必須の doctest は無い。任意で、facade の `backward_create_graph` に `Tape::rnn_forward_seq` を含む CPU だけで通る HVP 例を 1 つ足す。GPU の例は `no_run`／`#[ignore]`。
- 失敗系は `compile_fail` に頼らず、`Err(AutodiffError::Backward(_))` と `child.is_empty()` で確かめる。

### 37.3 エラー型

- 新しい variant は足さず `AutodiffError::Backward(String)` を使う。拒否するのは次の場合。
  - `validate_ancestors` で、親の `LstmHidden.cell` が `LstmCell` を指していない（子テープに書く前に判定する）。
  - 再生直後に、子テープ側の対応ノード（37.10）が `LstmHidden { cell: 写しの c の id }` になっていない（push 順序契約の違反。`rollback_to` で `child` を空に戻す。`create_graph.rs:366〜373`）。
  - shape 不整合。
- 本番経路で `unwrap`／`expect` は使わない。

### 37.4 テスト計画

CI の行は新設 `crates/autodiff/tests/create_graph_rnn_cell.rs` に置く。テスト名は案で、実装 PR で確定する。

| 受け入れ基準 | テスト（案） | 区分 |
|---|---|---|
| 1 階 `first_order()` が素の `backward` と bit 同一。`cg.grad` が REQ-2 で一致（4 Op × bias None／Some × B=1・H=1 と B=3・D=4・H=5） | `rnn_cells_first_order_bit_identical_and_cgrad_req2` | CI |
| ヘッシアンと有限差分の一致。x・h_prev・c_prev・w_ih・w_hh・b のそれぞれと混合二階（weight × input）。損失は `sum(h²)`・`sum(h⊙c)` など非線形を合成。5 点中央差分 h=1e-2（§27.4 の前例）。判定式は変えない | `rnn_cells_hessian_matches_finite_difference` | CI |
| BPTT T=3 で重みを共有し fan-in がある場合（`nn::Rnn/Lstm/Gru` の forward_seq 経由）。LSTM は h だけ使う／c だけ使う（`LstmHidden` が祖先に無い）／両方使う、の 3 通り | `rnn_cells_bptt_fan_in_and_lstm_output_subsets` | CI |
| 3 段（1 階・`cg.grad`・`jvp_create_graph` 経由の孫テープ再生）で、有限か 1 階と同じ非有限か。飽和ゲート（pre=±1e4・±1e30）、`g=3e38`、大きな `c_prev`・`q` | `rnn_cells_mandatory_vectors_three_stages` | CI |
| 掃引: 1 階が有限なら cgrad・二階も有限 | `rnn_cells_sweep_first_order_finite_implies_cgrad_finite` | CI |
| push 順序契約の違反が型付きエラーになり `child` が空のまま残る | `create_graph.rs` 内の単体テスト（`Op` が `pub(crate)` のため） | CI |
| 4 Op が `true`、隣接の `CtcLoss` が `false` | `rnn_cell_family_supports_create_graph`（`tape.rs` 単体） | CI |
| 隣接する fail-closed: `Op::Custom` と組み合わせると拒否され `child` が空（§32・§34 の方針） | `rnn_cells_neighbouring_unsupported_op_stays_fail_closed` | CI |
| facade 経由 | `create_graph_facade.rs` に `rnn_`／`lstm_`／`gru_create_graph_via_facade` を `check_create_graph_family`（:552）で追加（`Tape::*_forward_seq` 経由） | CI |
| 実機 parity | `gpu_extended_ops_match_cpu`（:826）のケース表へ rnn／lstm／gru を追加（`create_graph_extended_ops_{cuda,metal}_matches_cpu`）。#3776・#3777 の雛形ができたら登録表へ移す。ROCm は雛形が無く `<rocm-node>` は未実測 | `#[ignore]` |

- 既存テストの差し替え: **無し**。RNN セルを「拒否される Op」の見本にしている既存テストは無い（`create_graph*.rs`・`double_vjp_feasibility.rs`・`functional_ops*.rs`・`jacobian_hessian_parity.rs`・`functional_ops_jvp_tests.rs`・`nn_rnn.rs`・facade の `create_graph_facade.rs` を grep して 0 件）。§34.13 のような差し替えは要らない。
- 実機の結果はすべて **未実測**。実行は #3856（`<cuda-node>`）と #3858（`<metal-node>`）が担う。開発機の代替値で pass にしない。

### 37.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値・FMA 契約は変えない。新しい baseline は作らない。
- 1 階は構造上 bit 同一（§21.1(1)）。写しの値も、forward と同じ公開 `Var` メソッド・同じカーネルを通るので 1 階と bit 同一になる。
- `cg.grad` は REQ-2 で一致させる。pointwise 因子を子テープの primitive 合成で作り直すので、fused カーネル・FMA（GRU の forward は `mul_add`。`eval.rs:5405〜5590`）・Metal fast-math とは演算順が違う。
- bias の項は V3（`reduce_bias_grad_var`。`create_graph.rs:2475`）。保証は「同じ入力に対する bias 縮約は 1 階と同じ f64 契約を保つ」ことに限る。bias 勾配全体の値は上流の `g` を含め他の `cg.grad` と同じく REQ-2 統一複合判定の範囲の一致を主張し、1 階との bit 一致は主張しない。
- 保証範囲: seed の絶対値が 1 以下で、1 階が有限な入力に限る。真値が f32 を超える場合（1 階と同じく `inf`）と三階以上は対象外。

### 37.6 互換への影響

- **追加のみ**。`api-compat-0.14.0-decision.md` §1.2 項目 3（`Err` が `Ok` になる受理範囲の拡大）にあたり、§8 の手順に当てはめても破壊的変更は生じない。`Op` は `pub(crate)` で、payload は追加しない。

### 37.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）。
- [ ] `compat-api-scope.md` の適用記録（承認後の実装 PR で行う）
- [x] `docs/README.md` の索引
- [x] `docs/compat-matrix-ari-tracker.md` 行 3 の「設計記録」セル

### 37.8 ベンチと性能への影響

- 既定経路（forward・`Tape::backward`）は変わらない。増えるのは create_graph 経路だけ。
  - LSTM／GRU は cgrad の中でゲートを作り直すため、セル 1 つあたり GEMM が 2 本と pointwise 数ノード増える。
  - `RnnCell` は写しの出力 `h_m` で済むので GEMM は増えない。
- 既定経路に触れないので前後比較は不要。触れた場合は 5 回計測の中央値で比べ、判定規則を計測前に固定する。

### 37.9 依存

- 追加しない。許容 12 区分と禁止リストに照らして新規なし。新しい `unsafe` も無い。

### 37.10 方式と式

共通事項:

- 再生 R は公開メソッド `Var::rnn_cell`／`lstm_cell`／`gru_cell` への自己再生にする（§21.1(2)）。cgrad は V1（primitive の合成）で書き、bias だけ V3 にする。
- 記号: `x_m`・`h_m`・`c_prev_m`・`W_m` は写し、`g` は upstream。
- 子テープの GEMM は親ノードの `fp32_strict` を引き継ぎ、`transpose` ＋ `matmul_fp32_strict` を使う（§14a、`matmul_vjp` と同じオペランド順）。
- `dW` の部分幅ブロックは `Pad`（§29）かゼロ定数との `Concat` で列位置へ戻す（1 階の `embed_columns_*` の写し）。`db` は `reduce_bias_grad_var`（V3）を通したあと同じ方法で戻す。
- **payload のゲートは cgrad で定数として使えない**。`gates_ifg`・`gate_o`・`gates_rzn`・`q` は x・h_prev・W の関数で、定数扱いにすると二階が落ちる。そこで子テープ上で `pre = x_m·W_ih_m (+b_ih) + h_m·W_hh_m (+b_hh)` → `narrow` → `sigmoid`／`tanh` と作り直す。
- **GRU は共通手順の例外（reset-after）**。`pre_i = x·W_ih (+b_ih)` と `pre_h = h_prev·W_hh (+b_hh)` を分離したまま作り、`r = sigmoid(pre_i_r + pre_h_r)`、`z = sigmoid(pre_i_z + pre_h_z)`、`q = pre_h_n`、`n = tanh(r⊙q + pre_i_n)` と再構築する（`eval.rs::gru_pointwise` と同じ。`n = tanh(pre_i_n + pre_h_n)` ではなく `r` の乗算が入る）。

| Op | 方式 | cgrad の式 |
|---|---|---|
| `RnnCell` | V1＋bias V3 | `d_pre = g ⊙ (1 − h_m²)`（`h_m` は写しの出力。`Op::Tanh` の腕と同じ形）。`dx = d_pre·W_ihᵀ`、`dW_ih = x_mᵀ·d_pre`。h 側も同形 |
| `LstmCell` | V1＋bias V3 | `g` は fan-in 合算後の `dc`（同じ step の `LstmHidden` 腕の寄与と、次 step の `dc_prev`）。`eval.rs:5429` 付近と同じ式: `d_pre_i = dc⊙g⊙i⊙(1−i)`、`d_pre_f = dc⊙c_prev_m⊙f⊙(1−f)`、`d_pre_g = dc⊙i⊙(1−g²)`、`dc_prev = dc⊙f`。`i,f,g` は作り直したゲート。積は 1 階と同じく左から順に掛け、非有限になるかを 1 階と揃える。列ブロック i,f,g を `Concat` して `[B,3H]` を作り、`W[:, :3H]` を `narrow` |
| `LstmHidden` | V1＋bias V3 | `d_pre_o = dh⊙tanh(c_m)⊙o⊙(1−o)`、`dc = dh⊙o⊙(1−tanh(c_m)²)`。`o` は作り直したゲート、`c_m` は写しの c。x・h_prev・W・b への寄与は親の `nodes[cell].op` から NodeId を読み列ブロック `[3H, 4H)` に置く。`(cell, dc)` も `accumulate` する |
| `GruCell` | V1＋bias V3 | 式は `eval.rs:5525` 付近と同じ。`q` は `h_m·W_hh_m (+b_hh)` の n ブロックから作り直す。`d_pre_i` と `d_pre_h`（n ブロックだけ `⊙r`）を別々の affine に流す。`h_prev` への 2 系統（直接の `dh⊙z` と affine 経由）は `accumulate` で合算する |

**`LstmCell`／`LstmHidden` の 2 ノード再生（#3796 の核）**

- §21.2 が QR／SVD を多出力 payload で fail-closed にしたのとは事情が違う。`h_t` は `cell` を NodeId の入力として持ち、`c_t` は祖先として先に写される。
- 案 (a)（推奨）: `LstmCell` の再生で `Var::lstm_cell` を呼び `mirror[cell] = c_m` とする。`LstmHidden { cell }` の再生では push 順序契約を使い、子テープの `c_m.id + 1` のノードを `Var::from_raw` で取り、`Op::LstmHidden { cell: c_m.id }` であることを検査して違えば fail-closed にする。`build_mirror`（`create_graph.rs:588`）のシグネチャを変えず、状態も持たない。
- 案 (b): `build_mirror` に側表（親の cell id → `h_m`）を持たせる。
- h を使わない場合、子テープに孤立した `LstmHidden` が残るがどこからも到達しないので無害。

### 37.11 閉包性と族間の依存

- 子テープへ積む Op は、構造側が `MatMul`（rank 2・strict）・`Transpose`・`Add`・`Mul`・`ScalarBinary`（Sub）・`Sigmoid`・`Tanh`・`Narrow`・`Concat`・`Pad`・`Reshape`・`BroadcastTo`・`Sum`（V3 のタグ）・定数 `Leaf`、写しの forward として積む 4 Op 自身。構造側はいずれも `supports_create_graph == true` であることを確かめた（`Pad`・`ScalarBinary` は実装前に origin/main でもう一度確かめる）。
- 他の C 区分の族には依存しない。孫テープでは 4 Op 自身が再生されるので、#3795 は `RnnCell` と `GruCell` を、#3796 は `LstmCell` と `LstmHidden` を同じ PR の中で `true` にする。LSTM の 2 Op は必ず一緒に反転する。
- §21.4 の依存順には「#3795・#3796 は互いに独立、他の C 区分とも独立」の 1 行を足す扱いとする（§21.4 の本文は書き換えない）。

### 37.12 fail-closed・総括的な判定

- 構造は 1 階と同じ f32 の積を同じ順で並べるだけで、スケール変換も f64 縮約も入らない。このため既定では §27.3 型の総括判定を置かない。
- 掃引テストで「1 階が有限なら cgrad・二階も有限」を固定する。違反する入力が見つかったら、実装 PR で型付きの拒否を足し、本節に逸脱として追記する（37.14 の論点 5）。
- `is_checkpoint_eligible` は触らない。§21.6 の判定順・網羅 match・未知 variant を `false` に倒す規約は弱めない。

### 37.13 GPU の forward 経路と Phase 10

- forward と 1 階の pointwise は CPU・CUDA・Metal・ROCm のすべてでデバイスカーネルが走り、ホストへのフォールバックは無い（`backend-cpu/src/ops.rs:3032〜3148`、`backend-cuda/src/ops.rs:7614〜7799`、`backend-metal/src/ops.rs:5927〜6082`、`backend-rocm/src/ops.rs:833〜872`）。`gemm_bias_act` は常に FP32 厳密。
- 子テープの構造 Op（GEMM・pointwise）もデバイスで走る。V3 の bias は f64 のホスト縮約。
- Metal の fast-math では、pointwise の作り直しとカーネルの値が ulp 単位でずれる。これは REQ-2 で受け、bit 一致は主張しない。
- 実機の結果はすべて **未実測**。#3856（`<cuda-node>`）・#3858（`<metal-node>`）へ申し送る。ROCm は雛形が無く `<rocm-node>` は未実測のまま記録する。

### 37.14 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 族の採否 | 4 Op を実装する／対象外にする（B）／非 fused モードの公開（C。別承認） | 実装する | 実装へ進めず fail-closed を維持 |
| 2. 写し R の方式 | `Var::*_cell` の自己再生／primitive 合成＋値を V3 で親の記録値に置き換え（子テープに 4 Op が載らないが §21.1(2) から外れ、置き換えなければずれが T step で累積する） | 自己再生 | 同上 |
| 3. `LstmHidden` の対応づけ | push 順序契約（`c_m.id+1` を検査）／側表 | push 順序契約 | 同上 |
| 4. pointwise 因子の値 | V1（REQ-2）／V3（`push_v3` で 1 階カーネル値に bit 一致） | V1（GEMM の経路でも bit 一致は保証されないため） | 同上 |
| 5. 総括判定 | 置かない（掃引で固定し違反時に拒否を足す）／§27.3 型を置く | 置かない | 同上 |
| 6. GPU parity の判定方式 | #3739 に従う（tolerance は変えない）／別方式 | #3739 | 同上 |

### 37.15 その他

- セキュリティ観点: 外部入力は forward 側の確保前検証（rank・`D,H>0`・`checked_gate_width`）を通ったものに限る。新しい検査は子テープへ書く前に行い、拒否時は rollback で `child` を空に保つ。秘密情報・実機のホスト名・個人のパスは書かない。
- スコープ外: 実機の parity（#3856・#3858）、測り方（#3739）、f64 テープ（#3722）、他の C 区分の族、非 fused モードの公開、双方向／projection など新しい variant。
- Issue の紐付け: 本 PR は `Closes #3728`、`Refs #3767 #3795 #3796 #3719`。
- 変えていないもの: コード・依存・`unsafe`・tolerance・baseline・FMA 契約・`docs/spec`・既存節の本文・公開 API・`compat-api-scope.md`。
## 39. 設計記録（イシュー #3730・親 #3719。create_graph の B 区分・損失の回帰・二値・マージン系 9 variant）

### 39.0 位置づけ・承認根拠・境界・推奨

- **状態**: 設計記録のみ。承認は #3767 の回答待ちで、本節は何も承認しない。回答があるまでコード・rustdoc・公開 API・`docs/compat-api-scope.md` は変えず、`supports_create_graph` もまだ `true` にしない
- **基準**: `origin/main` の `c6c73f9c2`。以下の `file:line` はこの時点で確かめた値で、実装 PR の前に引き直す
- **対象**: §26.2 の表で第 1 弾（§26。Mse・L1・Huber・CE・Nll の 5 つ）に含まれず `supports_create_graph()`（`crates/autodiff/src/tape.rs:2526`）が `false` のまま残った損失 16 variant のうち、§21.4 の (a) 回帰系の残り 3 つ（`PoissonNllLoss`・`GaussianNllLoss`・`KlDivLoss`）と (b) 二値・マージン系 6 つ（`BceLoss`・`BceWithLogitsPosWeightLoss`・`SoftMarginLoss`・`SigmoidFocalLoss`・`MarginRankingLoss`・`HingeEmbeddingLoss`）の計 9 variant。網羅 match の false 側（`tape.rs:2615`〜`:2630` 付近）にある。残るクラス軸系は #3731、`CtcLoss` は #3801
- **根拠**: §21.1（V1〜V3・S1・閉包性）、§21.3 の #3082（target を定数化しない・入力の数と同じ数の寄与を返す）、§21.6（fail-closed 契約）、§26（第 1 弾の方式）、§27.3・§36.12（総括判定）。ルート #3711 の「実装を既定に設計する」趣旨は設計方針の根拠であり、承認ではない
- **後続**: 実装は #3798（回帰 3 variant）と #3799（二値・マージン 6 variant）、承認は #3767、実機 parity は #3857（`<cuda-node>`）・#3859（`<metal-node>`）、雛形は #3776・#3777、測り方は #3739
- **推奨**: **実装する（9 variant すべて）**。理由は次の 4 つ
  1. 9 つとも 1 階 VJP が要素ごとの閉じた式で、replayable な Op の合成で書ける
  2. f64 で評価する 7 つは、1 階のホスト関数をそのまま V3 の値に使える
  3. §26 の部品（`loss_scale_bcast`・`pairwise_mask`・`where_cond`・`push_v3`）を流用できる
  4. BCE・KL・Gaussian NLL・focal loss は二値分類・VAE・不確実性推定・検出で一般的で、勾配ペナルティやメタ学習のように HVP・二階を使う手法が拒否されると実害がある

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | 9 variant を実装する | 損失を含むグラフで `hvp`・`jvp`・`jacfwd`・`jvp_create_graph` が使える。追跡表の項目 4 が「ある」に近づく | 実装と保守の費用。総括判定の追加 |
| B | 対象外にして fail-closed を恒久契約にする | 費用ゼロ | 上記の損失で高階変換が使えないまま残る |

### 39.1 公開形

該当なし。公開名は増えない。`Tape::backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` は公開済みで、受理する Op の範囲が広がるだけである。9 損失の `Var` メソッド（`bce_loss`・`bce_with_logits_loss`・`kl_div_loss`・`kl_div_loss_with_log_target`・`margin_ranking_loss`・`hinge_embedding_loss`・`soft_margin_loss`・`bce_with_logits_loss_with`・`gaussian_nll_loss`・`sigmoid_focal_loss`・`poisson_nll_loss`。`crates/autodiff/src/var.rs`）は公開済みで、形は変えない。

`crates/facade/tests/api_surface.rs` には、この 9 損失の create_graph を対象にした否定ガードも保留 doctest ガードも無く、反転するものは無い（損失まわりのシグネチャのガードは create_graph と無関係なので残す）。実装 PR（#3798・#3799）で変える箇所:

- `supports_create_graph` の網羅 match で 9 variant を false 側から true 側へ移す。ワイルドカードは足さない
- `create_graph.rs` のモジュール doc にある「対象外」の列挙（77〜85 行付近）
- facade の rustdoc にある「対象 Op の正本は §21.2・§22〜§31」（`crates/facade/src/lib.rs` の 6 か所）の範囲に §39 を加える
- §26.2 の表の更新（本 PR では既存節の本文を書き換えない。実装 PR の作業）

### 39.2 rustdoc と doctest

- 新しい公開 API が無いので必須の doctest は無い。任意で facade の `backward_create_graph` か `hvp` に、CPU だけで通る `bce_with_logits_loss` の HVP 例を 1 つ足す。GPU の例は置かない（置くなら `no_run`／`#[ignore]`）
- 失敗系は `compile_fail` に頼らず、ランタイムの `Err(AutodiffError::Backward(_))` 一致と `child.is_empty()` で確かめる

### 39.3 エラー型

- 新しい variant は足さず、`AutodiffError::Backward(String)` に統一する。拒否するもの: 総括判定で外れた入力、非有限の upstream、±inf の入力、未知の `BceKind`／`KlDivTarget`（§26 の未知 `HuberKind` と同じく fail-closed）
- 検証順: 判定は子テープへ書く前に行う（`validate_ancestors` か cgrad 構築の冒頭）。拒否したら `Tape::backward_create_graph` のロールバックで `child` を空に戻し、再試行できるようにする（§27.3 と同じ）。shape・`eps`・`margin`・`alpha`・`gamma`・`pos_weight`・`y`（±1）・Bce Probabilities の `[0,1]` は forward 側の検証を通ったものだけが届く。本番経路で `unwrap`／`expect` を使わない

### 39.4 テスト計画

候補名は案であり、実装 PR で確定する。CI の行は新規 `crates/autodiff/tests/create_graph_loss_b.rs`（仮名）に置く。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|--------------|--------|------|---------------|----------------|----------------|
| 1 階が素の `backward` と bit 同一、`cg.grad` が REQ-2 で一致（V3 の 7 つは CPU で bit 一致も見る） | `loss_b_first_order_bit_identical_and_cgrad_req2` | CI | 未実測 | 未実測 | 未実測 |
| ヘッシアンが有限差分と一致（9 variant × Mean／Sum × 各 kind・option。Bce は Probabilities／Logits、KlDiv は Probabilities／LogProbabilities、Poisson は log_input × full、Gaussian は full と `var>=eps` の通常領域を含む、focal は γ ∈ {0, 0.5, 1, 2} × alpha 有無。中央差分の刻みは §26.4・§27.4 の前例どおり。判定式は変えない。`var<eps` の clamp 区間は有限差分で検証しない: 既存の `gaussian_nll_loss_vjp` は呼ぶたびに `c = max(var, eps)` を計算するため、clamp 区間で `var` を摂動した `dx` の有限差分は 0 になる一方、§39.10 の素通し構造の `∂dx/∂var` は `−s_b·d/c²` で 0 にならず（丸め誤差ではなく構造の差）、両者は一致しない。clamp 区間は次の行の独立した解析式で検証する） | `loss_b_hessian_matches_finite_difference` | CI | 未実測 | 未実測 | 未実測 |
| 追跡する target・var との混合二階項（`d²L/dx·dt`。Gaussian は `dx·dvar`・`dt·dvar`）が、通常領域（`var>=eps`）では有限差分と一致し、Gaussian の clamp 区間（`var<eps`）では §39.10 の素通し契約から導いた独立の解析式（`∂dx/∂var = −s_b·d/c²` など。有限差分は使わない）と一致 | `loss_b_mixed_second_derivatives_with_tracked_target` | CI | 未実測 | 未実測 | 未実測 |
| `jvp_create_graph` で孫テープに再生し、方向微分と一致 | `loss_b_grandchild_replay_matches_directional_derivative` | CI | 未実測 | 未実測 | 未実測 |
| 必須ベクトル（39.5）が 3 段（1 階・`cg.grad`・孫テープ）で有限か、拒否されて `child` が空 | `loss_b_mandatory_vectors_finite_or_rejected_in_three_stages` | CI | 未実測 | 未実測 | 未実測 |
| 区分の境界（Hinge の `x == margin`、MarginRanking の `raw == 0`、Bce の両クランプ境界、KlDiv の `t == 0`、Poisson の `t == 1`、Gaussian の `var == eps`）で 1 階の分岐規則と一致 | `loss_b_piecewise_boundaries_follow_first_order_rule` | CI | 未実測 | 未実測 | 未実測 |
| 安全でない入力の拒否・`child` が空・再試行できる | `loss_b_unsafe_inputs_rejected_child_empty_and_retry` | CI | 未実測 | 未実測 | 未実測 |
| 空テンソルで全ゼロ | `loss_b_empty_tensor_returns_zeros` | CI | 未実測 | 未実測 | 未実測 |
| 隣接 Op（`Op::Custom`）が fail-closed のまま | `loss_b_neighbouring_unsupported_op_stays_fail_closed` | CI | 未実測 | 未実測 | 未実測 |
| facade 経由（代表は `bce_loss`〈CPU 融合経路〉・`kl_div_loss`・`gaussian_nll_loss`） | `crates/facade/tests/create_graph_facade.rs` に `*_create_graph_via_facade` を追加 | CI | 未実測 | 未実測 | 未実測 |
| CUDA／Metal の parity（Bce・KlDiv は融合カーネルを含む経路、残る 7 つはホスト実体化を含む経路。39.12） | `create_graph_loss_b_cuda_parity`／`create_graph_loss_b_metal_parity`（後者は `cfg(target_os = "macos")`）。置き場・名前は #3776・#3777 の雛形が確定するまで仮。測り方は #3739 に従う | `#[ignore]` | 未実測（#3857 が測る） | 未実測（#3859 が測る） | 未実測 |

**差し替える既存テスト（#3798・#3799 の作業）**: 9 variant が true になると次の 2 件が壊れる。

1. `crates/autodiff/tests/create_graph.rs:3498` の `create_graph_still_rejects_remaining_loss_ops`: 拒否見本が `kl_div_loss` なので、恒等の `Op::Custom` へ替え、テスト名も直す
2. `crates/autodiff/src/tape.rs:5862` の `first_wave_losses_are_supported_and_neighbors_stay_closed`: 9 variant を `supported` 側へ移し、`closed`（現状 `Op::KlDivLoss`）を `Op::Custom` 系に替える

隣接見本に `CtcLoss` を使わない理由: §21.2 では恒久 fail-closed だが #3801 が実装 issue として起票済みで true に変わりうる。`CrossEntropyLossWithOptions`・`CosineEmbeddingLoss` などクラス軸系も #3731・#3800 で変わりうる。`Op::Custom` は §21.6・§32 が基準テストに使う恒久的な対象外である（§36.4 と同じ理由づけ）。二つの実装 PR のどちらが先に入っても壊れないよう、最初の PR で両テストを `Op::Custom` 基準へ替える。実機の実行は #3857・#3859 が担い、Linux や開発機の代替値で pass にしない。

### 39.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えない。新しい baseline は作らない（作るなら実機の実測値に限り所有者承認）
- 1 階は構造上 bit 同一（§21.1(1)）。V3 の 7 つは `cg.grad` の値も CPU で 1 階と bit 同一。Bce と KlDiv は V1 なので REQ-2 で比べる（1 階の dinput は融合カーネルを通り、子テープは f32 合成。§26.4 の MSE・NLL と同じ扱い）
- f64 の契約は V3 の値側で守る。構造側の f32 合成は値を決めない。FMA 契約には触れない
- 必須ベクトル（3 段すべてで有限かつ一致。外れる入力は総括判定で拒否され `child` が空になることを固定する）:
  - Poisson: log_input の `x=100`（f32 の `exp` が overflow）、非 log の `x+eps=0`、`t=1` の境界、full で `t` が大きい場合
  - Gaussian: `var=0 < eps`、`d=1e20`、`eps=1e-6` で `d²/c²` が overflow する組
  - KlDiv: `t=0`、LogProbabilities の `t=90`（`exp` が overflow）
  - Bce Probabilities: `p=0`・`p=1`・`p=1e-30`。Logits: `x=±1e30`
  - BcePosWeight: `x=±1e30`・`pos_weight=0`
  - SoftMargin: `x=±1e30`
  - focal: `q` が f32 で 0 になる飽和域（`|x|` が大きい）を γ < 1・γ = 0 で
  - 全般: `g=3e38`・`g=[3e38, -3e38]`
- 保証範囲は §27.3 と同じ書き方にする。総括判定を通った入力で、seed の各要素の絶対値が 1 以下のときに限る。三階以上は対象外
- 既知の差を明記する: (1) focal の `0<γ<1` で `q=0` のとき、1 階は第 2 項を極限値 0 にする。(2) `Softplus` の threshold（20）で恒等に切り替わる区間では構造の二階が 0 になる。真値は約 2e-9 で、REQ-2 の絶対誤差の救済範囲に収まる

### 39.6 互換への影響

**追加のみ**。`api-compat-0.14.0-decision.md` §8 の手順で判定する。触れる型は `Op`（`pub(crate)`）だけで、`AutodiffError` の variant・公開シグネチャ・オプション型（`PoissonNllOptions`・`GaussianNllOptions`・`BceWithLogitsOptions`・`SigmoidFocalLossOptions`）・`Reduction` は変えない。以前 `Err` だった入力が `Ok` になる受理範囲の拡大であり、破壊的条件には当たらない。payload の追加も要らない（再生に要る値は `tape.rs:303`〜`:522` の各 payload に揃っている）。

### 39.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）
- `docs/compat-api-scope.md` への適用記録は #3767 の承認後の実装 PR で書く。文案: `**適用記録（イシュー #3798／#3799・親 #3730。承認: <#3767 の回答 URL>）**: 損失の回帰・二値・マージン系 9 variant を create_graph の対象にした。公開名の追加なし。`
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（承認後の PR で行う。承認が無いうちは書かない）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md` を更新した（項目 4 の設計記録の欄に §39 を追記）

### 39.8 ベンチと性能への影響

既定経路（forward・`Tape::backward`・融合カーネル）は変わらない。`supports_create_graph` は create_graph の入口でしか呼ばれない。増えるのは create_graph 経路の要素ごとのノード（variant ごとに十数ノード）と、マスク・総括判定のホスト走査 O(numel) だけである。既定経路に触れないので前後比較は不要と判断する。実装で既定経路に触れた場合は、5 回計測の中央値で前後を比べ、判定規則を計測前に固定する。実機の値は未実測。

### 39.9 依存の追加の有無

追加なし。許容 12 区分・禁止リストとの照合で変化なし。新しい `unsafe` なし。

### 39.10 方式と式（variant ごと）

記号: `x_m`・`t_m`・`v_m` は子テープ上の入力の写し、`s_b = loss_scale_bcast(child, &g, reduction, false, n = numel, &shape)`（`create_graph.rs:2559`。`g` は子テープの `Var` のまま）。V3 の値は 1 階のホスト関数を `pub(crate)` にして `push_v3`（`create_graph.rs:986`）で入れる（本体・数値は変えない）。区分定数は V2（親の記録値からホストでマスクを作り `where_cond`。`g.mul(0/1 定数)` は使わない）。マスクで選ばれない分岐の入力は、`Log`・`Div`・`PowScalar` に入れる前に安全な値へ置き換える（§26 の Huber の `d_safe` と同じ。ゼロ余接 × 無限大の微分係数で NaN が出るのを防ぐ）。1 階の scale は、Bce・KlDiv が f32（`grad.rs:8742`・`:9672`）、残る 7 つが f64（`grad.rs:8906` の `elementwise_loss_scale` か、その場の `g as f64 / n as f64`）。9 つとも Mean の分母は `numel` で、`n == 0` なら zeros を返す。

- **KlDivLoss（V1 + V2。f32。融合カーネルあり）**
  - R: `kind` で `kl_div_loss`／`kl_div_loss_with_log_target` へ委譲する。未知の kind は `Err`
  - Probabilities: `dinput = −s_b⊙t_m`、`dtarget = where(t≠0, s_b⊙(Log(t_safe) + 1 − x_m), 0)`。`t_safe` は `t==0` のレーンを 1 に置き換えたもの
  - LogProbabilities: `dinput = −s_b⊙Exp(t_m)`、`dtarget = s_b⊙Exp(t_m)⊙(t_m − x_m + 1)`
  - 負の `t` で出る NaN は 1 階と同じく伝播させる
- **BceLoss（V1 + V2。f32。融合カーネルあり）**
  - R: `kind` で `bce_loss`／`bce_with_logits_loss` へ委譲する
  - Logits: `dinput = s_b⊙(Sigmoid(x_m) − t_m)`、`dtarget = −s_b⊙x_m`
  - Probabilities: `dinput = s_b⊙(x_m − t_m)/D`。`D` は `p(1−p) < 1e-12` のレーン（マスクは 1 階と同じ f32 式でホストで作る）では定数 1e-12、それ以外では `x_m⊙(1 − x_m)`。`dtarget = −s_b⊙(L(x_m) − L(1 − x_m))`。`L(u)` は `ln u < −100` のレーンでは定数 −100、それ以外では `Log(u_safe)`
  - 境界の近くでヘッシアンは `1/(p(1−p))²` に比例して大きくなる。正しい値であり tolerance は変えない。上界は総括判定で見る
- **PoissonNllLoss（V3 + V2。2 入力）**
  - R: `poisson_nll_loss(t_m, &options, reduction)`
  - `log_input` は payload の定数で、マスクではなく構造の分岐にする。log の場合: `dx = s_b⊙(Exp(x_m) − t_m)`、`dt = −s_b⊙x_m`。log でない場合: `dx = s_b⊙(1 − t_m/(x_m + eps))`、`dt = −s_b⊙Log(x_m + eps)`
  - `full` かつ `t>1` のレーン（V2 マスク）だけ、`dt` に `s_b⊙(Log(t_m) + 0.5/t_m)` を足す
  - 値は `poisson_nll_loss_vjp`（`grad.rs:9025`）
- **GaussianNllLoss（V3。3 入力）**
  - R: `gaussian_nll_loss(t_m, v_m, &options, reduction)`
  - `c = max(var, eps)` の勾配は、1 階が clamp で遮らない（`var<eps` でも `dvar` を流す。PyTorch の no_grad 下での `clamp_` と同じ）。これに合わせ、構造は**素通しの形** `c_m = (v_m − const(var)) + const(c)` にする（値はちょうど `c`、`var` についての微分は 1）。`where_cond(var≥eps, v_m, eps)` は clamp 区間で微分が 0 になり 1 階と食い違うので採らない（39.16 の論点 4）
  - `d = x_m − t_m`、`dx = s_b⊙d/c_m`、`dt = −dx`、`dvar = 0.5·s_b⊙(1/c_m − (d/c_m)²)`
  - 値は `gaussian_nll_loss_vjp`（`eval/elementwise_loss.rs:271`）の 3 出力を、それぞれ `push_v3` で入れる
- **BceWithLogitsPosWeightLoss（V3。2 入力。`pos_weight` は定数 `P`）**
  - R: `bce_with_logits_loss_with` に、`pos_weight` を input の shape へ展開済みの `P` として渡す（展開が冪等であることを実装時に確かめる。不可なら `pub(crate)` の内部入口を使う）
  - `lw = 1 + (P − 1)⊙t_m`、`dx = s_b⊙((1 − t_m) − lw⊙Sigmoid(−x_m))`、`dt = s_b⊙(−x_m + (P − 1)⊙Softplus(−x_m; β=1, threshold=20))`
  - 値は `bce_with_logits_pos_weight_loss_vjp`（`eval/elementwise_loss.rs:98`）
- **SoftMarginLoss（V3。`y` は定数）**
  - R: `soft_margin_loss(&y, reduction)`
  - `dx = s_b⊙(−y)⊙Sigmoid(−y⊙x_m)`。値は `soft_margin_loss_vjp`（`eval/elementwise_loss.rs:208`）
- **HingeEmbeddingLoss（V3 + V2。x について区分定数）**
  - `dx = where(y==1, s_b, where(margin − x > 0, −s_b, 0))`。マスクは 1 階と同じ f64 比較でホストで作る。境界 `x == margin` は 0
  - 曲率は `g` を経由する分しか流れない（§26 の Nll と同じ形）。値は `hinge_embedding_loss_vjp`（`eval/elementwise_loss.rs:156`）
- **MarginRankingLoss（V3 + V2。2 入力）**
  - `active = (−y·(x1 − x2) + margin ≥ 0)`。f64 でホストで判定し、境界は通す（`clamp_min` の契約）。`dx1 = where(active, −s_b⊙y, 0)`、`dx2 = −dx1`。値は `margin_ranking_loss_vjp`（`grad.rs:8918`）
- **SigmoidFocalLoss（V3 + V2。最も難しい。`target`・`alpha_t` は定数）**
  - `p = Sigmoid(x_m)`、`q = t⊙Sigmoid(−x_m) + (1 − t)⊙p`、`ce = (1 − t)⊙x_m + Softplus(−x_m)`
  - `dx = s_b⊙α_t⊙[(p − t)⊙PowScalar(q, γ) + γ⊙ce⊙PowScalar(q, γ−1)⊙(1 − 2t)⊙p⊙(1 − p)]`
  - `γ == 0` は payload の定数で分岐し、第 2 項を構造に積まない
  - 飽和域（f32 で `q==0`）で `γ<1` のレーン、および二階で `q^(γ−2)` が発散する `1<γ<2` の飽和レーンは、推奨では総括判定で拒否する。代替は V2 で 1 階の極限値 0 を写す案で、39.16 の論点 5 に載せる
  - 値は `sigmoid_focal_loss_vjp`（`eval/margin_focal_loss.rs:311`）

### 39.11 閉包性と族間の依存

- 子テープへ積む Op: `Sub`・`Mul`・`Div`・`Add`（ScalarBinary）、`Neg`・`Exp`・`Log`・`Sigmoid`・`Softplus`・`PowScalar`（ScalarUnary）、`Where`・`BroadcastTo`・`Contiguous`（`push_v3`）。基準時点で `scalar_unary_replayable`（`create_graph.rs:145`）と `scalar_binary_replayable`（`:180`）で true になっていることを確かめた。実装前に `origin/main` で再確認する
- 他の族に依存しない（Softmax や索引系を使わない）。9 variant は自分自身を構造に積まない。孫テープの再生では写しの forward が再生されるので、9 variant 自身が true であることが要る。#3798 と #3799 は互いに独立

### 39.12 fail-closed・総括判定・GPU の forward

- **総括判定**: `check_structure_stages`（`crates/autodiff/src/create_graph_norm.rs:338`）と同型の判定を、`Exp`・`Log`・`Div`・`PowScalar` を含む構造（Poisson・Gaussian・Bce Probabilities・KlDiv LogProbabilities・focal）に置く。構造の forward の各段、二階の逆伝播の「cotangent × 微分係数」、境界の cotangent の上界をホストが f64 で求め、`2^125` 以上か非有限の段があれば拒否する。閾値と seed の前提は §27.3 と同じ。値域が有界な構造（Sigmoid と線形だけ）にも upstream の検査は適用する
- **入口での拒否**: 非有限の upstream は拒否する。非有限の入力は §33.14 の論点 3 と同じ規則（NaN は伝播し ±inf は拒否）に合わせる。§21.6 の判定順・網羅 match・未知 variant を false にする規約は弱めない
- **GPU の forward**:
  - `BceLoss`・`KlDivLoss` は CPU・CUDA・Metal・ROCm に融合 forward と backward があり（`crates/backend-*/src/ops.rs` の `bce_loss_backward`／`kl_div_loss_backward`）、子テープ上の replay もその backend の forward を通る。parity はカーネルを含む経路で測ることになる
  - 残る 7 つはどの backend でもホストの `push_eager` で計算するので、#3857・#3859 が測るのはデバイスからホストへの実体化を含む経路であって、カーネル間の parity ではない
  - GPU カーネルは新設しない。値は未実測。Linux や開発機の代替値で pass にしない

### 39.13 実装の段分け（Phase 5・2h 単位）

- #3798: ① KlDiv（V1 + V2・融合カーネル経路の REQ-2 確認。39.4 の既存テスト差し替えもここで行う）→ ② PoissonNll（V3・full のマスク・総括判定）→ ③ GaussianNll（V3・3 入力・素通し c・総括判定）
- #3799: ① Bce + BceWithLogitsPosWeight → ② SoftMargin + Hinge + MarginRanking（V3 + V2）→ ③ SigmoidFocal（V3・飽和域の拒否）
- 各段で 39.4 の該当行と facade テストを足す。実機の実行は #3857・#3859 に任せる

### 39.14 セキュリティ観点

- A03: forward の検証を通った値だけが届く。新しい確保はマスクと O(numel) の f64 一時領域に限る
- A04: 拒否時に子テープを空に保ち、fail-closed を維持する
- A06: 依存を追加しない。A08: 総括判定と網羅 match を迂回する経路を作らない。秘密情報・ホスト名・個人のパスは書かない

### 39.15 スコープ外・Closes／Refs

- **スコープ外**: クラス軸系 6 variant（#3731）、`CtcLoss`（#3801）、`Reduction::None` の追加（公開 API の変更。§26.3）、f64 テープの create_graph、GPU 実機の parity（#3857・#3859）、測り方（#3739）。新しい issue は承認なしに起票しない
- `Closes #3730`。`Refs #3798 #3799 #3767 #3719`

### 39.16 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 9 variant の採否 | 実装する／対象外にする | 実装する | 確認があるまで実装へ進めない（fail-closed を維持） |
| 2. f64 で評価する 7 つの方式 | V3（値を 1 階と bit 同一にする）／V1（f32 合成で REQ-2） | V3 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 3. Bce・KlDiv の方式 | V1 + V2（融合カーネルの 1 階と REQ-2 で比べる）／ホスト関数による V3 | V1 + V2 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 4. GaussianNll の clamp の二階 | 素通し（1 階・PyTorch に一致）／`where` で clamp 区間を 0 にする | 素通し | 確認があるまで実装へ進めない（fail-closed を維持） |
| 5. focal の飽和域・Poisson の `x+eps<=0`・overflow する入力 | 総括判定で型付き拒否（§27.2 と同じ）／V2 で 1 階の値（極限 0・inf）を写す | 拒否 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 6. 非有限の入力 | §33.14 の論点 3 と同じ規則（NaN は伝播し ±inf は拒否）／全部拒否 | §33 と同じ規則 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 7. Softplus の threshold | 構造に `Softplus(β=1, threshold=20)` を使い差は REQ-2 の範囲として受け入れる／`Sigmoid`・`Log` で組む | threshold=20 | 確認があるまで実装へ進めない（fail-closed を維持） |
| 8. GPU parity の判定方式 | #3739 の推奨に従う（tolerance は変えない）／別方式 | #3739 に従う | 確認があるまで実装へ進めない（fail-closed を維持） |
| 9. 隣接見本 | `Op::Custom`／`CtcLoss` | `Op::Custom` | 確認があるまで実装へ進めない（fail-closed を維持） |

### 39.17 変えていないもの

コード・公開 API・rustdoc・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・`docs/compat-api-scope.md`。

## 40. 設計記録（イシュー #3731・親 #3719。create_graph の B 区分・損失のクラス軸系 6 variant と CtcLoss）

### 40.0 位置づけ・承認根拠・境界・推奨

- 状態: **設計記録のみ**。承認は #3767 の回答待ちで、本節は承認を代行しない。コード・公開面・tolerance・baseline は変えない。基準 sha は `49d627a5b`。
- 対象: §26.2 表の (c) クラス軸系のうち未実装の 6 variant（`CrossEntropyLossWithOptions`・`MultiMarginLoss`・`MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`CosineEmbeddingLoss`・`TripletMarginLoss`）と、§21.2 で恒久 fail-closed とした `CtcLoss`。いずれも `crates/autodiff/src/tape.rs` の `supports_create_graph`（:2606〜2630 付近）の網羅 match で `false` 側にあり、`backward_create_graph` は型付きエラーで拒否する（fail-closed）。
- 根拠: §20（D-A）、§21.1（V1〜V4・S1・閉包性・数値契約）、§21.2、§21.3 の #3082、§21.6、§26、§27.3（総括判定）。これらは設計方針の根拠であって承認ではない。
- 推奨: **6 variant は A「実装する」**、**CtcLoss は B「対象外にする（恒久 fail-closed を維持）」**。ただし採否は #3767 で個別に問う。

| 案（6 variant） | 内容 | 評価 |
|---|---|---|
| A（推奨） | 各 variant を自己再生＋cgrad の primitive 合成で書く。hvp・jvp・jacfwd・jvp_create_graph が通る | 1 階 VJP が閉じた式で、`Softmax`・`Sigmoid`・`Sqrt`・`Abs`・`PowScalar`・`Sum`・`Gather`・`Where` 等の replayable な Op で書ける |
| B | 対象外にし fail-closed を恒久契約にする | label smoothing 付き CE や metric learning 損失を含むモデルの二階微分が拒否され続ける |

| 案（CtcLoss） | 内容 | 評価 |
|---|---|---|
| A | α／β を `Var` の合成で子テープに再生する | 40.10 の理由 1〜5 で不採用 |
| B（推奨） | 対象外（恒久 fail-closed を維持） | 現状のまま。上限なしのノード生成を避けられる |
| C | V4（専用の二階 VJP） | §4／§5 が一般方針として退けているため不採用 |

- 6 variant を推す理由: (1) 1 階 VJP がすべて閉じた式、(2) 6 つとも 1 階が f64 のホスト関数なので V3 の値に流用できる、(3) §26 の部品（`loss_scale_bcast`・`onehot_*`・`pairwise_mask`・`push_v3`）を使い回せる、(4) CE の label smoothing・class weight と metric learning（Cosine・Triplet）は実用頻度が高い。
- 境界: #3730（§39 の 9 variant）、他の B 区分、C 区分（`VectorNorm`・`PNorm`・`LogSumExp`・`Max` は #3725・#3724 で設計中の fail-closed Op）は扱わない。**構造に C 区分の Op を積まない**。

### 40.1 公開形

- 該当なし。公開名は増えない。対象の `Var` メソッド（`cross_entropy_loss_with`・`cosine_embedding_loss`・`multilabel_margin_loss`・`multi_margin_loss`・`multilabel_soft_margin_loss`・`triplet_margin_loss`・`ctc_loss`）は公開済みで形は変えない。変わるのは受理する Op の範囲だけ。
- `crates/facade/tests/api_surface.rs` に、これらの create_graph を拒否する否定ガードや保留 doctest ガードは無い（`__Fandhe*HoldMarker` 系はメソッド公開のガードで無関係）。**反転するガードは無い**。正ガードの代わりは 40.4 の facade 挙動テストとする。
- 実装 PR（#3800）で変える箇所: `supports_create_graph` で 6 variant を `true` 側へ移す（ワイルドカードは足さない）、`create_graph.rs` のモジュール doc の「対象外」列挙、facade `lib.rs` の rustdoc が指す「§21.2・§22〜§31」の範囲に本節を加える、§26.2 表の更新（本 PR では行わない）。

### 40.2 rustdoc と doctest

- 必須の doctest は無い。任意で、facade の `hvp` か `backward_create_graph` に CPU だけで通る `cross_entropy_loss_with`（label_smoothing 付き）の HVP 例を 1 つ足す。GPU の例は `no_run`／`#[ignore]`。
- 失敗系は `compile_fail` に頼らず、`Err(AutodiffError::Backward(_))` と `child.is_empty()` で確かめる。

### 40.3 エラー型

- 新しい variant は足さず `AutodiffError::Backward(String)` に統一する。拒否するのは、総括判定（40.12）を外れた入力、非有限の upstream、±inf の入力（§33.14 論点 3 と同じ規則。NaN は伝播し ±inf は拒否）、未知の enum 値、内部の shape 不整合。
- 判定は子テープへ書く前に行い（`validate_ancestors` か cgrad 構築の冒頭）、拒否時は `Tape::backward_create_graph` のロールバックで `child` を空に戻す。
- 外部入力（targets 範囲・`y` が ±1・`p>=1`・`eps>=0`・`margin` 有限・class_weight 非負）は forward の検証を通ったものだけが届く。本番経路で `unwrap`／`expect` は使わない。
- CtcLoss が対象外のままなら、拒否経路は現行の `supports_create_graph == false` のまま変えない。

### 40.4 テスト計画

CI の行は新設 `crates/autodiff/tests/create_graph_loss_class.rs`（仮名）に置く。名前は案で、実装 PR で確定する。

| 受け入れ基準 | テスト（案） | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 1 階が素の `backward` と bit 同一。`cg.grad` は REQ-2 で一致（V3 の値は CPU で bit 一致も見る） | `loss_class_first_order_bit_identical_and_cgrad_req2` | CI | 未実測 | 未実測 | 未実測 |
| ヘッシアンが有限差分と一致（6 variant × Mean／Sum）。CEWO は label_smoothing・ignore_index・class_weight の各組と class_dim 0／最終軸／rank3 中間軸、MultiMargin は p=1／2・weight 有無、Triplet は p ∈ {1, 2, 3}・swap 有無・同値タイ。刻みは §26.4・§27.4 の前例どおり | `loss_class_hessian_matches_finite_difference` | CI | 未実測 | 未実測 | 未実測 |
| 多入力の混合二階（Cosine の `d²L/dx1·dx2`、Triplet の a·p・a·n・p·n） | `loss_class_mixed_second_derivatives` | CI | 未実測 | 未実測 | 未実測 |
| `jvp_create_graph` で孫テープへ再生し方向微分と一致 | `loss_class_grandchild_replay_matches_directional_derivative` | CI | 未実測 | 未実測 | 未実測 |
| 必須ベクトル（40.5）が 3 段（1 階・`cg.grad`・孫テープ）で有限か、拒否されて `child` が空 | `loss_class_mandatory_vectors_three_stages` | CI | 未実測 | 未実測 | 未実測 |
| 区分の境界が 1 階の規則と一致（Multi* は `z>0` 厳密、Cosine・Triplet は `>=0`、Triplet swap の同値タイは 0.5 等分、CEWO の早期 return 2 種） | `loss_class_piecewise_boundaries_follow_first_order_rule` | CI | 未実測 | 未実測 | 未実測 |
| ignore 行・全クラス重み 0・`Mean && W==0` で logits に NaN があっても 0 を返す | `cewo_zero_weight_paths_return_zero_not_nan` | CI | 未実測 | 未実測 | 未実測 |
| 安全でない入力の拒否・`child` が空・再試行できる | `loss_class_unsafe_inputs_rejected_child_empty_and_retry` | CI | 未実測 | 未実測 | 未実測 |
| 空テンソルで全ゼロ | `loss_class_empty_tensor_returns_zeros` | CI | 未実測 | 未実測 | 未実測 |
| 6 variant が `true` | `tape.rs` 単体テスト `class_axis_losses_support_create_graph` | CI | 未実測 | 未実測 | 未実測 |
| 隣接する fail-closed: `Op::Custom` との組み合わせは拒否され `child` が空 | `loss_class_neighbouring_unsupported_op_stays_fail_closed` | CI | 未実測 | 未実測 | 未実測 |
| facade 経由（6 損失。`check_create_graph_family` で） | `*_create_graph_via_facade` | CI | 未実測 | 未実測 | 未実測 |
| 実機 parity: `gpu_extended_ops_match_cpu`（`create_graph_facade.rs`）のケース表へ 6 損失を足す。#3776・#3777 の雛形ができたら登録表へ移す。測り方は #3739 | 同左（置き場と名前は仮） | `#[ignore]` | 未実測（#3857） | 未実測（#3859） | 未実測（雛形なし） |

- 既存テストの差し替えは無い。6 variant と CtcLoss を拒否の見本にしている既存テストは無い。`tape.rs` の `first_wave_losses_are_supported_and_neighbors_stay_closed` と `tests/create_graph.rs` の `create_graph_still_rejects_remaining_loss_ops` は `KlDivLoss` を見本にしており、#3730 系の PR が `Op::Custom` へ替える予定（§39.4）。#3800 の実装 PR では、その時点の main で引き直し、6 variant が見本にされていないことを再確認する。
- 隣接見本は `Op::Custom` とする。CtcLoss は #3767 の回答で変わりうるので見本にしない。
- 実機の実行は #3857・#3859 が担う。Linux や開発機の代替値で pass にしない。

### 40.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値・FMA 契約は変えない。新しい baseline は作らない（作る場合は実機の実測値に限り所有者承認が要る）。
- 1 階は構造上 bit 同一（§21.1(1)）。6 つとも V3 なので `cg.grad` の値も CPU で 1 階と bit 同一になる。構造側の f32 合成は二階と孫テープにだけ効く。
- f64 の契約（`coding-rust.md`。二乗和は先に昇格）は V3 の値側で守る。構造側の overflow は S1 と総括判定で防ぐ。
- 必須ベクトル（3 段すべて）:
  - CEWO: logits `±1e30`、class_weight に大きな値（Σw が f32 を超える組は拒否）、`g=3e38`。
  - Multi*: `x=±1e30`・p=2 で `z²` 相当が大きい組、`g=3e38`。
  - MultiMargin p=2: 1 行 2 クラス・`x=[-1e38, 1e38]`・target=0・`w_y=1e-38`・margin=0・`g=1`（f64 VJP は有限だが f32 構造の `H` が overflow するため、総括判定で拒否されること）。
  - Cosine: `x1=x2=1e20` の行（f32 の二乗が inf）、零ベクトル行（`m=ε`）、`x=1e-30` の行（`ε/s²` の保護）、`y=-1` で `cos==margin`。
  - Triplet: 差分 `1e20`、`p=1024` 相当、差分がちょうど `-eps` になるレーン、swap の同値、`d_an=inf` の側。
  - 全般: `g=[3e38]`・`g=-3e38`。
- 保証範囲は §27.3 の書き方に揃える。総括判定を通った入力で、seed の各要素の絶対値が 1 以下のときに限る。三階以上は対象外。

### 40.6 互換への影響

- **追加のみ**。`docs/api-compat-0.14.0-decision.md` §1.2 項目 3（`Err` が `Ok` になる受理範囲の拡大）にあたり、§8 の手順に当てはめても破壊的条件には当たらない。
- 型ごと: `Op`（`pub(crate)`）は payload を追加しない（再生に要る値は `tape.rs` の既存 payload に揃っている）。`AutodiffError` の variant は不変。オプション型（`CrossEntropyOptions`・`TripletMarginOptions`・`MultiMarginOptions`・`MultiLabelSoftMarginOptions`・`CtcLossOptions`）と `Reduction` は不変。公開シグネチャは不変。
- CtcLoss を対象外にした場合は現状のまま（`Err`）なので影響は無い。

### 40.7 適用記録・索引・追跡表

- 公開状態: 未公開（設計のみ）。
- `compat-api-scope.md` 適用記録の文案（承認後の実装 PR で書く）: 「**適用記録（イシュー #3800・親 #3731。承認: <#3767 の回答 URL>）**: 損失のクラス軸系 6 variant を create_graph の対象にした。公開名の追加なし。」CtcLoss は「対象外に確定（承認 URL）」と「実装した場合」の 2 通りを書く。
- チェック欄:
  - [ ] `compat-api-scope.md`（承認後の実装 PR で行う）
  - [x] `docs/README.md` の索引
  - [x] 追跡表（`docs/compat-matrix-ari-tracker.md`）行 4 の設計記録の欄に本節を記入

### 40.8 ベンチと性能への影響

- 既定経路（forward・`Tape::backward`）は変わらない。`supports_create_graph` を呼ぶのは create_graph の入口だけ。
- 増えるのは create_graph 経路だけ。variant ごとに数〜二十数ノード、CEWO は子テープに `Softmax` が 1 つ増え、Cosine・Triplet は行ごとのスケール・ノルム合成が増える。マスクと総括判定のホスト走査は O(numel)。
- 既定経路に触れないので前後比較は不要と判断する。触れた場合は 5 回計測の中央値で比べ、判定規則を計測前に固定する。実機の値は未実測。

### 40.9 依存の追加の有無

- 追加なし。許容 12 区分・禁止リストとの照合で変化なし。新しい `unsafe` も無い（`docs/license-matrix.md` の更新は不要）。

### 40.10 方式と式（variant ごと）

記号: `x_m` 等は子テープ上の写し、`g` は子テープ上の upstream `Var`。R は公開メソッドへの自己再生（§21.1(2)。forward の検証を迂回しない）。V3 の値は 1 階のホスト関数（`grad.rs` の各 VJP、`eval/margin_focal_loss.rs`）を `pub(crate)` にして `push_v3`（`create_graph.rs`）で入れる。本体と数値は変えない。区分定数は V2 で作り（親の記録値からホストで 1 階と同じ f64 判定でマスクを作り `where_cond`）、`g.mul(0/1 定数)` は使わない（`inf·0=NaN` を避けるため）。

Mean の分母は variant ごとに違う。`loss_scale_bcast`（`create_graph.rs`）は `n: usize` しか取らない。

| variant | Mean の分母 |
|---|---|
| CEWO | 非 ignore サンプルの重み和 `W`（f64）。`loss_scale_bcast` は使えないので `g / const(W as f32)` の別ヘルパーを足す |
| Multi* 3 種 | 行数 N（`rows_cols`。rank 1 は 1） |
| Cosine・Triplet | 行数 n（`shape[0]`。rank 1 は 1） |

- **CrossEntropyLossWithOptions（V3 + V1 構造。`Softmax` に依存）**
  - R: `cross_entropy_loss_with(&targets, class_dim, reduction, &options)`。非既定オプションなので再生でも同じ variant が積まれる。
  - 構造: `dx = where(valid, s_b ⊙ (A_b ⊙ softmax(x_m, class_dim) − B), 0)`。`A = (1−ε)·w_t + (ε/C)·Σw`（サンプルごと）、`B_c = (1−ε)·w_t·δ_{c,t} + (ε/C)·w_c` はホスト定数（f64 で作り f32 へ）。`valid` は ignore でない行のマスク。
  - 早期 return（`Σw==0`、`Mean && W==0`）は定数ゼロの cgrad にする。1 階と同じく NaN を出さない。
  - `class_weight` を定数化しても二階は失われない（非追跡データのため）。V1 単独で REQ-2 比較にする案もある（論点 3）。
- **MultiMarginLoss（V3 + V2）**
  - `z_j = margin − x_y + x_j`。`x_y` は `gather(x_m, y)` を broadcast して作る。活性マスク `M = (j≠y ∧ z_j>0)` はホストで作る。
  - p=1: `dx = where(K≠0, s_b ⊙ const(K), 0)`（`K` は `w_y/C` の符号付き個数）。
  - p=2: `H = where(M, 2·Z, 0)`、`dx = s_b ⊙ c_w ⊙ where(onehot_y, −Sum(H, 1)_b, H)`（`c_w = w_y/C` は行定数）。二階は p=2 でだけ曲率を持ち、p=1 は `g` を経由する分だけ。
- **MultiLabelMarginLoss（V3 + V2）**: x について区分定数なので `dx = where(K≠0, s_b ⊙ const(K/C), 0)`。`K` は活性な (t, d) 組の符号付き個数で、重複 target は重複分だけ数える（1 階と同じ）。
- **MultiLabelSoftMarginLoss（V3 + V1）**: `dx = s_b ⊙ (const(w_c/C) ⊙ (Sigmoid(x_m) − t))`。掛ける順はこの括弧のとおりで、中間値が最終値を超えないようにする。
- **CosineEmbeddingLoss（V3 + V2 + S1 構造。2 入力）**
  - `coeff ∈ {−1, +1, 0}`（`y>0` → −1、`y<0 ∧ cos−margin>=0` → +1、それ以外 0）は V2。ホストで 1 階と同じ f64 の `cos` から判定する。
  - 行ごとのノルムを `Mul`・`Sum(dim)`・`Sqrt`・`Div` で組む。**`Op::VectorNorm` は使わない**（C 区分の fail-closed Op）。
  - `x²` は `1e20` で f32 の inf になるので §21.1(5) の S1 を使う。行ごとに `s_i = max(max|x_i|, √ε)` を 2 のべき乗へ切り上げ、`x/s` を作り、`ε` を `ε/s²` に変換する（ε=1e-12）。
  - 式: `dx1 = s_b·coeff/s1·(b'/denom' − cos·a'/m1')`。`dx2` も同形。1/s の掛け順は総括判定で確かめる。rank 1 は `[1, d]` へ reshape する。
- **TripletMarginLoss（V3 + V2 + S1 構造。3 入力）**
  - `v = a_m − p_m + eps`（an・pn も同形）。p ノルムは 1 階の `p_norm_f64` と同じスケール形 `mx·(Σ(|v|/mx)^p)^(1/p)` で組む。`mx` は行ごとの 2 のべき乗のホスト定数で、`Abs`・`PowScalar`・`Sum`・`Mul` を使う。勾配は `sign(u)·(|u|/norm')^(p−1)`。
  - V2 にするもの: `hinge_active`（`>=0`）、`an_coeff`（1／0／0.5。乗算でなく `where` で選び、0.5 の同値だけ非零定数を掛ける）、`norm==0`（PyTorch と同じく 0）、p=1 の符号。
  - 寄与は `da = g_ap − c_an·g_an`、`dp = −g_ap − c_pn·g_pn`、`dn = c_an·g_an + c_pn·g_pn`（1 階 VJP と同じ）。
  - `1<p<2` で `u==0` のレーンは二階が発散する（真値も特異）。推奨は総括判定で拒否（論点 5）。
- **CtcLoss（推奨: 対象外）**
  - 実装案 A: α／β を `Var` の合成で子テープに再生する。1 セルは 3 項以下の logaddexp と emission で、`Gather`・`Exp`・`Log`・`Add`・`ScalarBinary::Maximum`・`Where` は replayable。logaddexp は `m + Log(Exp(a−m)+Exp(b−m))`（m はホスト定数）で書ける。値は `ctc_loss_vjp` の V3、`γ = Exp(α+β+nll)` を構造にする。
  - 不採用の理由:
    1. `ctc_add_emission` の「−inf は加算相手を問わず支配する」規約と β の規約（`docs/autodiff-ctc-design.md` §2.4）が、合成では `−inf−(−inf)=NaN` を生む。到達不能セルごとに V2 マスクが要る。
    2. サンプルごとの `nll==+inf`・`zero_infinity`・NaN の分岐はホストの制御フローで、構造に写すと分岐ごとの定数化が要る。
    3. ノード数がセル単位で O(N·T·L') × 約 8（例: N=32・T=100・L'=50 で約 128 万ノード）。時刻ごとにベクトル化しても O(N·T) × 十数ノードで β 側で倍になる。上限なしの消費は A04／DoS の観点からも拒否理由になる。
    4. 1 階は f64 の α／β だが構造は f32 の対数空間になる。長い T で誤差が積み上がり、二階が REQ-2 に収まる保証が弱い。
    5. V4（専用の二階 VJP）は §4／§5 が退けている。
  - 補足: PyTorch が CTC の double backward を提供するかは**未確認**。実装判断の前に一次資料（derivatives.yaml）で確かめる。推測で事実化しない。
  - 対象外に決まった場合、#3801 は #3767 の回答 URL を設計記録・追跡表に書く docs のみの PR で閉じる。実装に決まった場合の段分けは 40.13。

### 40.11 閉包性と族間の依存

- 子テープへ積む Op: `Add`・`Mul`・`Sum`・`BroadcastTo`・`Reshape`・`Narrow`・`Concat`・`Contiguous`（`push_v3`）・`Where`・`Gather`・`Softmax`（#3077・V3）・`ScalarBinary`（Sub・Div・Maximum）・`ScalarUnary`（Neg・Abs・Sqrt・Sigmoid・PowScalar・Exp・Log）。基準時点で `true` であることを `supports_create_graph` で確かめた。実装前に origin/main で再確認する。
- C 区分の `VectorNorm`・`PNorm`・`LogSumExp`・`Max` は積まない。
- 依存は #3077（Softmax）→ CEWO の 1 本だけ（§21.4 の「#3077 → #3082(c)」と一致）。#3083(a) の `Gather` は実装済み。
- 孫テープでは 6 variant 自身が再生されるので、各 variant の `true` への反転は、その variant の cgrad と同じ PR で行う。

### 40.12 fail-closed・総括判定・GPU の forward

- 総括判定: §27.3 の `check_structure_stages`（`create_graph_norm.rs`）と同型の判定を、Cosine・Triplet（スケール・ノルム・`PowScalar`）と CEWO（大きな class_weight を含む定数）に置く。構造の各段・二階の cotangent × 微分係数の上界をホストが f64 で求め、`2^125` 以上か非有限なら拒否する。閾値と seed の前提は §27.3 と同じ。MultiLabelMargin・MultiLabelSoftMargin と MultiMarginLoss の p=1 は x について線形・有界なので upstream の検査だけを適用する。**MultiMarginLoss の p=2 は対象に含める**（線形ではない）。`H = 2·Z`、`Sum(H, 1)`、`c_w ⊙ …`、`s_b ⊙ …` の各段の上界をホストが f64 で求め、同じ `2^125` 閾値で拒否する。重み `w_y` を後から掛ける構造のため、f64 では有限（例: 1 行 2 クラス・`x=[-1e38, 1e38]`・target=0・`w_y=1e-38`・margin=0・`g=1` は VJP が約 `[-2, 2]`）でも f32 の `H` が約 4e38 で overflow する入力を、この段別の上界検査で拒否する（重みを先に掛ける演算順への組み替えは採らない。`Z` 自体が f32 を超えるため解決しない）。必須ベクトルにこの反例を加える。
- §21.6 の判定順・網羅 match・未知 variant を `false` に倒す規約は弱めない。CtcLoss は `false` のまま。
- GPU の forward: 7 Op はどれも全 backend でホストの `push_eager` で計算し、融合カーネルは無い。子テープの構造 Op（Softmax・pointwise・Sum）はデバイスで走る。よって #3857・#3859 が測るのは「デバイス→ホストの実体化を含む経路」で、カーネル間の parity ではない。GPU カーネルは新設しない。値は未実測。

### 40.13 実装の段分け（Phase 5・2h 単位）

- #3800:
  1. CEWO（W 分母のスケールヘルパー・早期 return・Softmax 依存）
  2. MultiMargin + MultiLabelMargin + MultiLabelSoftMargin（行数の分母・V2 マスク）
  3. CosineEmbedding（S1 の行スケール・ε 保護・総括判定・混合二階）
  4. TripletMargin（スケール形 p ノルム・swap・`1<p<2` の扱い・3 入力の混合二階）
  - 各段で 40.4 の該当行と facade テストを足す。
- #3801（実装に決まった場合だけ）: ① α の合成 → ② β と γ の合成 → ③ 分岐の V2 化・ノード数上限の拒否・テスト。
- 実機は #3857・#3859 に任せる。

### 40.14 セキュリティ観点

- A03: forward の検証を通った値だけが届く。新しい確保はマスクと O(numel) の f64 一時領域に限る。
- A04: 拒否時は子テープを空に保つ。CtcLoss を対象外にする理由の 1 つは上限なしのノード生成（DoS）を避けること。
- A06: 依存を追加しない。A08: 総括判定と網羅 match を迂回する経路を作らない。
- 秘密情報・実機のホスト名・個人のパスは書かない。

### 40.15 スコープ外・Closes／Refs

- スコープ外: §39 の 9 variant（#3730）、他の B 区分（#3729・#3732〜#3735）、C 区分の `VectorNorm` 等（#3725）、`Reduction::None` の追加（公開 API の変更。§26.3）、f64 テープ（#3722）、GPU 実機 parity（#3857・#3859）、測り方（#3739）。
- 本 PR は `Closes #3731`、`Refs #3800 #3801 #3767 #3719`。

### 40.16 所有者へ戻す論点（#3767 で問う）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 6 variant の採否 | 実装する／対象外 | 実装する | 実装へ進めず fail-closed を維持 |
| 2. CtcLoss の採否 | 対象外（恒久 fail-closed）／α・β の `Var` 合成で実装／V4 | 対象外 | 同上 |
| 3. CEWO の方式 | V3（1 階 bit 一致）／V1 単独（REQ-2 比較） | V3 | 同上 |
| 4. Cosine・Triplet のノルム構造 | S1（行ごとの 2 のべき乗スケール＋総括判定）／S2（統計量をホスト定数化し解析式） | S1 | 同上 |
| 5. Triplet の `1<p<2`・`u==0` | 総括判定で拒否／V2 で 1 階の値を写す | 拒否 | 同上 |
| 6. hinge 境界 | 1 階の規則を継承（Multi* は `z>0` 厳密、Cosine・Triplet は `>=0`。不揃いは既存規則の引き継ぎで本設計が作るものではない）／揃える（1 階の変更になるため別承認） | 継承 | 同上 |
| 7. 非有限の入力 | §33.14 論点 3 と同じ（NaN は伝播し ±inf は拒否）／全部拒否 | §33 と同じ | 同上 |
| 8. GPU parity の判定方式 | #3739 に従う（tolerance は変えない）／別方式 | #3739 | 同上 |
| 9. 隣接見本 | `Op::Custom`／`CtcLoss` | `Op::Custom` | 同上 |

### 40.17 変えていないもの

- コード・公開 API・rustdoc・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・既存節の本文・`docs/compat-api-scope.md`。
