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
