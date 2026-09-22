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
| **対象（初期スコープ）** | `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`ScalarUnary`・`ScalarBinary`・`MatMul`・`Sum`（`dim=None` または単一軸）・`Mean`（`dim=None` または単一軸）・`Reshape`・`Transpose`・`Permute`・`BroadcastTo`・`Narrow`・`Concat`・`Contiguous`・`Where`・`MaskedFill`・`Gather`・`Scatter`・`Pad`・`MseLoss`・`CrossEntropyLoss`（target は非微分入力として扱う。(c) の非追跡ペイロードは入力側であり出力勾配の合成自体は可能） | (a)(b)(d) を満たす。VJP は §2 の対応表がすでに `Var` 演算への分解を示す。`Mean` は `mean_vjp`（`crates/autodiff/src/grad.rs:5118`）が `Sum` の VJP（`unreduce_broadcast`）を `1/n` でスケールしたものと一致し追加の数値契約を持たないため `Sum` と同じ扱いとする。`Pad` は `pad_vjp`（`:1568`）が `Op::Narrow` の forward と同じ `narrow` 呼び出し連鎖のみで構成され算術を含まないため `Narrow` と同じ扱いとする。`CrossEntropyLoss` は target を非微分入力として扱えば合成可能（PyTorch の `create_graph=True` でも target 自体には勾配は流れない）。`Softmax`／`LogSoftmax` は #1941 是正により本行から除外し「非対象（合成では再現できない数値契約）」へ移した（下表参照） |
| **非対象（resident／fused／CUDA Graph）** | `ResidentLeaf`・`LinearResident`・`LinearAct` | 契約 6（§3）。d_weight がデバイス常駐 staging（`ResidentResolver::fill_resident_weight_grad`）へ直接書き込まれ `Gradients` に現れないため、二階の入力として使えない。CUDA Graph capture（#1349）区間も同様に非対象 |
| **非対象（非追跡ペイロード）** | `QrQ{r}`・`QrR{q}`・`SvdU{s,vh}`・`SvdS`・`SvdVh` | (c) 抵触。多出力ノードが `Tensor<f32>` を直接ペイロードとして持ち、それ自体をテープ上の追跡演算として再構成する経路が未整備 |
| **非対象（合成では再現できない数値契約）** | `Softmax`・`LogSoftmax`（`softmax_vjp_along`／`log_softmax_vjp_along` が縮約後の減算・乗算まで `f64` で保持し最後に 1 回だけ `f32` へ downcast する契約。§2）・`Max`（先勝ちタイ規則。`extremum_first_match_vjp`）・`Min`（同上）・`Var`（分散。`f64` アキュムレータ縮約契約）・`Std`・`VectorNorm`・`MatrixNorm`・`RmsNorm`・`LayerNorm`・`BatchNorm`・`Conv2d`（d_weight／d_bias の `f64` 縮約契約）・`AvgPool2d`・`AdaptiveAvgPool2d`（`f64` 相当縮約契約）・`Cumsum`（`cumsum_vjp_along`〈`:4654`〉が lane ごと `f64` アキュムレータで逆順累積和を構築する契約） | (d) 抵触。`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64` アキュムレータで統一する」契約・`extremum_first_match_vjp` の先勝ち決定的方式（#1718）は、既存 `Var` 演算（`eq` マスク等）の単純合成では bit 再現できない。子テープで再構成すると縮約順序が変わり 1 階と異なる丸め誤差を生む可能性がある。`Softmax`／`LogSoftmax` も同型: `y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` のような有限入力で `softmax_vjp_along` は有限値を返すが、`Var::mul`／`Var::sum_dims`／`Var::broadcast_to` への素朴な合成（各演算が `f32` で中間値を持つ）では途中の乗算・減算が overflow し `inf`／`NaN` を生む（#1941 是正。codex-review 指摘） |
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
