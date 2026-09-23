# custom autograd Function（ユーザー定義 Op）プラグイン機構の設計判断

対応イシュー #1623（親 #1573〈Tier 2〉→ ルート #1570）。位置づけは**設計判断の記録のみ**であり、
本 PR では `crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。数値一致の
複合判定（REQ-2）・tolerance／baseline も変更しない。基準コミット: `origin/main` `2d434c9a`
（#1676「高階微分（grad of grad）の設計判断」マージ後）。

**注記（#1945）**: §2 の「現状のコード事実」は基準コミット `2d434c9a` 時点のスナップショットで
あり不変のまま残す。段階 0 の再開条件 3 件（#1612／#1593／#1634）が完了した HEAD 基準の再確認・
`CustomFunction` trait 境界の確定は §12 を参照。実装は兄弟イシュー #1946 が担当する。

## 1. 背景

機能ギャップ表 `docs/compat-feature-gap.md` §2.11「autograd」の custom `autograd.Function` 行は、
現状「なし（`Op` enum は crate 非公開の固定 variant 集合。ユーザー定義 Op を挿す口がない）・
実装に必要なもの: 拡張可能な Op プラグイン機構の設計（現行のクローズドな `Op` enum 設計を変更）・
難度 XL」として設計から要検討のまま残っている（`docs/compat-feature-gap.md:311`）。

spec 側は REQ-9 の 2026-09-12 追記で custom autograd Function を **Tier 2（長尾）** に明記済み
（`docs/spec/04-requirements.md:232`）。実装リポ側 `docs/compat-api-scope.md` §1.3 Tier 2 表は
当該行を「custom autograd Function \| #1623（設計記録）」としており（`docs/compat-api-scope.md:243`）、
本イシューがその設計記録を担当する。

対比対象:
- PyTorch `torch.autograd.Function`: `forward`／`backward` の静的メソッド対＋`ctx.save_for_backward`
  で任意の順伝播・逆伝播ペアをグラフへ挿入できる
- TensorFlow `tf.custom_gradient`: 順伝播の戻り値と勾配関数の組（クロージャ）を返すデコレータ方式

想定用途: straight-through estimator（STE）・gradient reversal・数値安定な独自式の forward/backward・
外部カーネル呼び出しの勾配化など、既存の固定 `Op` variant 集合では表現できない演算をユーザーが
グラフへ挿す経路。

受入基準（イシュー本文を構造化したもの。逐語引用はしない）:
1. 現行のクローズドな `Op` enum を拡張してユーザー定義 Op を挿せるようにする場合の可否・設計内容・
   判断根拠を doc として記録する（実装は含めない）
2. REQ-12「利用者向け融合制御 API を提供しない」との関係を整理する
3. 承認事項があれば doc 本文に明記し、実装着手の前提として残す
4. tolerance・baseline は変更しない

## 2. 現状のコード事実（基準コミット `2d434c9a` で確認）

| 事実 | 出典 |
|---|---|
| `Op` は `pub(crate)`・`#[derive(Debug, Clone)]`（`Copy` は持たない）のクローズド enum。29 variant（`Leaf`・`MatMul`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`Sum`・`Max`・`MseLoss`・`CrossEntropyLoss`・`ResidentLeaf`・`LinearResident`・`LinearAct`・`Reshape`・`Transpose`・`Inv`・`Solve`・`Det`・`Cholesky`・`QrQ`・`QrR`・`SvdU`・`SvdS`・`SvdVh`・`MatrixNorm`・`Softmax`・`LogSoftmax`）。`Copy` を持たない理由は `CrossEntropyLoss` 等クラス添字（`Tensor<i32>`）を直接保持する非追跡ペイロードを持つ variant があるため（RNN／LSTM／GRU 系 Op は #1647 で作業中・本基準コミット時点では未マージ） | `crates/autodiff/src/tape.rs:87-332` |
| `grad::vjp(op: &Op, ...) -> Result<Vec<(NodeId, Tensor<f32>)>>` が `op.clone()` して `match` し全 variant を網羅的にディスパッチする。`Op::` の参照は `grad.rs`・`tape.rs`（`is_lazy_elementwise`／`is_view`／`resolve_view`／`build_lazy_plan`）・`var.rs`（shape 計算）・`optim/device_store.rs`・`eval/linalg.rs`・`nn/linear.rs`・`backward.rs`・`eval.rs`・`lib.rs`・`layout.rs`・`nn/activation.rs`・`nn/loss.rs` に及ぶ。variant 追加はこれら全てに波及しうる | `crates/autodiff/src/grad.rs` |
| ノード登録 API（`Tape::push_eager`／`push_view`／`push_lazy`／`push_resident_leaf`）・`Var::from_raw`・`Tape::ops()` はすべて `pub(crate)`。利用者が葉を作る唯一の入口は `Tape::var(&Tensor<f32>) -> Var` | `crates/autodiff/src/tape.rs:696,808,819,847,876,951`・`crates/autodiff/src/var.rs:80` |
| `TapeNode { op, shape, value: OnceCell<Tensor<f32>>, lazy_chain_size }`。遅延評価対象は elementwise 5 演算（`is_lazy_elementwise`）のみ。`BackendOps` に対応メソッドのない `Sigmoid`／`MseLoss`／`CrossEntropyLoss` は常に `push_eager`（実体化済み＝融合境界） | `crates/autodiff/src/tape.rs:577,517,530` |
| `Tensor<f32>` は host 常駐（`Storage<T> { data: Vec<T> }`）。デバイス常駐は `DeviceBuffer`／`ResidentLeaf` の別経路 | `crates/tensor-core/src/tensor.rs:33-34` |
| `Tape::backward_impl` は逆走査ループ全体を単一の不変借用 `self.nodes.borrow()` で完結させ、各ノードで `materialize_fallible` → `grad::vjp` → `accumulate` を呼ぶ。`Tape: Send` は静的アサーション（`fn assert_send<T: Send>() {} assert_send::<Tape>();`）で固定 | `crates/autodiff/src/backward.rs:132`・`crates/autodiff/tests/fusion_backend_integration.rs:391-392` |
| `ResidentResolver`（`pub(crate)` trait）は backward に注入されるフック（`Option<&dyn ResidentResolver>`）の既存先例。`Op::LinearResident` の VJP が d_weight をデバイス staging へ直接書く | `crates/autodiff/src/tape.rs:387` |
| `nn::Module` trait（`forward(&self, tape, input)`／`forward_host`）は `autodiff` で `pub` だが、facade は `nn::LinearVars` のみ再エクスポート。`compat::Sequential` は `Vec<Box<dyn Module>>` を持つが公開ビルダーは `add_linear`／`add_relu`／`add_sigmoid`／`add_tanh` に固定（`add_module` はない）→ `Sequential` へ独自層として登録する経路が現行の公開面から到達不能（ただし公開 `Var` 演算を直接合成し `Sequential` を介さずに使うこと自体は妨げられない） | `crates/autodiff/src/nn/module.rs:33`・`crates/facade/src/compat/sequential.rs:116,129,135,141` |
| `Var` の公開演算は `matmul`・`add`・`mul`・`sum`・`max`・`relu`・`exp`・`tanh`・`sigmoid`・`softmax`・`log_softmax`・`mse_loss`・`cross_entropy_loss`・`reshape`・`transpose`・線形代数（`inv`…`matrix_norm`）。`sub`／`neg`／スカラー倍（#1593）・`detach`／`no_grad`（#1612）は本基準コミット時点で未実装（OPEN） | `crates/autodiff/src/var.rs`・#1593／#1612 |
| facade 公開面の機械検査 `api_surface.rs`: `pub use` での `Tape`／`BackendOps`／`new_with_ops` 再エクスポート禁止・`pub fn` が `BackendOps` を引数に取ることを禁止・`compat` の `pub fn` が生 `fandhe_ai_autodiff::Tape` を取ることを禁止 | `crates/facade/tests/api_surface.rs:53-164` |
| `BackendOps` は crates.io 公開済み trait。非破壊拡張は「既定 `Unsupported` のデフォルトメソッド追加」パターン（`sgd_step_device`・`memory_ops` 等）が慣例 | `crates/tensor-core/src/backend_ops.rs` |
| `AutodiffError`（`#[non_exhaustive]`）に「未対応演算」専用 variant はない | `crates/autodiff/src/error.rs:19-21` |
| REQ-12 の v2 充足方法: `facade` を唯一の公開面・composition root に結線集約・利用者向け公開面を `Device` 識別子のみに限定。「autodiff の ops 受け取り構築子はサポート外の内部 API で融合制御 API に該当しない」整理 | `docs/spec/04-requirements.md` REQ-12「2026-08-08 注記」・REQ-9「2026-08-08 追記」 |
| REQ-9 2026-09-12 追記: 「引き続き対象外」に「任意 `BackendOps` 実装を注入できる推論入口（REQ-12）」を明記 | `docs/spec/04-requirements.md:233` |
| `docs/fusion-graph-design.md` §3.3「backward（VJP）は融合対象外」契約・`Op` ノード粒度は融合の適用有無に関わらず変更しない | `docs/fusion-graph-design.md` |
| #1622（高階微分）doc §4 案 C が「`Op` ごとの JVP 追加は #1623〈custom Op〉の設計と関係」と参照し、末尾で「custom autograd Function（#1623。`Op` enum 拡張の論点は重なるが、本 doc では参照に留め決定しない）」と本イシューへ委ねている | `docs/autodiff-higher-order-grad-decision.md` |

## 3. 契約整理（設計が守るべき既存契約）

1. 既存 `Tape::backward` の結果は bit 同一で不変（既存 backward／parity／bit 一致テスト群を後退させない）。REQ-2 統一複合判定・FMA 契約・tolerance／baseline は変更しない
2. `docs/fusion-graph-design.md` §3.3: `Op` ノード粒度は融合の適用有無に関わらず変更しない・VJP は融合対象外。ユーザー定義 Op は必ず `push_eager`（融合境界）とし、`is_lazy_elementwise`／`is_view` 判定を変えない
3. `Tape: Send` 維持（trait object を持たせるなら `Send + Sync` 境界が必要）・`Op: Clone + Debug` 維持（`Arc<dyn …>` と手書き `Debug`／`name()` 実装が必要になる）
4. `TapeId`／`epoch` の世代契約・`Tape::reset` の葉プレフィックス契約を維持する
5. REQ-12「利用者向け融合制御 API を提供しない」「任意 `BackendOps` 実装を注入できる公開 API を設けない」・`api_surface.rs` 機械検査を維持する（ユーザー定義 Op に `&dyn BackendOps` を渡さない）
6. reuse（デバイス常駐）経路・`LinearResident`／`ResidentLeaf`・CUDA Graph capture（#1349）は対象外（ユーザー定義 Op が resident 経路に混在した場合は fail-closed なエラーとする）
7. `backward_impl` は不変借用 `self.nodes.borrow()` の内側でノードごとに `vjp` を呼ぶ。ユーザーコードがこの内側で走る場合、`Tape` への再入（`push_*` の呼び出し）は `RefCell` の借用契約により panic となる。これは構造的な禁止事項であり、ユーザー定義 `backward` の中から新たなグラフノードを追加することはできない

## 4. 設計案の比較

| 案 | 概要 | `Op` enum 変更 | テープ再設計 | REQ-12 整合 | `api_surface.rs` 影響 | 数値契約 | facade 公開面 | 前提 issue | 難度 |
|---|---|---|---|---|---|---|---|---|---|
| **A: 合成のみ（custom backward なし）** | 既存 `Var` 演算の合成で独自 forward を書く | 不要 | 不要 | 抵触なし | 影響なし | 既存演算の VJP のみ（合成の連鎖規則で自動導出） | 追加不要（既存 `Var` 演算のみ） | なし | S（`Var` 直接合成としての到達自体は可能。`Sequential` への統合は `add_module` 欠如で別途不可だが、案 A を退ける決定的理由ではない） |
| **B（主案）: `Op::Custom { inputs: Vec<NodeId>, func: Arc<dyn CustomFunction + Send + Sync> }`** | `trait CustomFunction { fn name(&self) -> &str; fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>>; fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>>; fn backward(&self, inputs: &[&Tensor<f32>], out_value: &Tensor<f32>, upstream: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> }`。host `Tensor<f32>` のみを受け渡し、`BackendOps` は非露出。常に `push_eager`（融合境界） | `Op` に 1 variant 追加（`Arc` 化・手書き `Debug`／`name()` が必要） | 不要（既存の逆走査アルゴリズムのまま） | 抵触なし（`BackendOps` を渡さない） | 影響なし（新規 trait は `BackendOps` を引数に取らない） | 同一ビット入力・同一 `Tape` 状態・決定的な `CustomFunction` 実装である限り、host 実行のためバックエンド間で構造的に bit 同一（REQ-2 判定対象外・ユーザー責任。乱数使用や並列縮約順序依存等の非決定的な実装ではこの限りではない） | 新規 `pub trait CustomFunction`＋`Var::custom(...)` 相当の入口（要承認） | #1612（`detach`）・#1593（`sub`／スカラー演算）・#1634（ScalarOp dispatch） | M |
| **B′: 案 B に `&dyn BackendOps` を渡す** | `forward`／`backward` にバックエンド実装への参照を渡しユーザーコードからカーネル選択を可能にする | 同上 | 不要 | **抵触**（任意 `BackendOps` 実装を注入できる公開 API を設けない、REQ-9 2026-09-12 追記の「引き続き対象外」に正面から該当） | `api_surface.rs` の「`pub fn` が `BackendOps` を引数に取ることを禁止」検査に抵触 | 同上 | 抵触するため不可 | — | — |
| **C: 勾配のみ差し替える固定集合の組み込み Op** | `Op::StraightThrough`／`Op::GradReverse(scale)`／`Op::GradClamp` 等を通常の Op として追加（`custom_gradient`-lite） | `Op` に固定個数の variant 追加 | 不要 | 抵触なし | 影響なし | 既存 Op と同じ扱い（parity 体系に自然に乗る） | 追加不要または最小限（`Var::straight_through()` 等の個別メソッド） | #1612（`detach`。役割が重なる） | S〜M（用途ごとに個別 issue が必要） |
| **D: `BackendOps` レベルのレジストリ／テーブル拡張** | 承認済み enum の範囲で backend 実装を選ぶ方式（#1634 ScalarOp dispatch の延長） | 不要（`ScalarUnary`／`ScalarBinary` 等の enum を拡張） | 不要 | 抵触なし | 影響なし | 既存契約のまま | 追加不要 | #1634 | 任意関数は挿せず「ユーザー定義」には届かない。将来 GPU カーネル化が必要になった場合の別軸の案として位置づけ |
| **E: 現時点では非対応と明文化し再開条件を定義する（段階 0）** | 何も実装しない。ギャップ表・spec の「Tier 2・XL」評価をそのまま維持し、再開条件を明記する | なし | なし | 抵触なし | 影響なし | 変更なし | 変更なし | — | 既存契約への影響ゼロ |

案 A（既存 `Var` 演算の合成）自体は公開 `Var` メソッドのみで表現でき到達可能である。`Module` が
facade 非公開・`Sequential` に `add_module` がないことは、独自層を `Sequential` へ登録する経路を塞ぐ
別の制限（§9 参照）であり、案 A 自体への到達を妨げるものではない。案 A を退ける決定的な理由は、
独自 backward を一切表現できないことにある。ただし `detach`（#1612）＋`sub`（#1593）が揃えば、
`x + (f(x) - x).detach()` 型の STE や、スカラー倍 `detach` による gradient reversal を**合成のみ**で
表現できる可能性がある（これは検証済みの事実ではなく、案 A の再開条件として記録するに留める）。

## 5. 推奨（段階的・前提ゲート付き）

- **段階 0（本イシューで完了）**: **案 E**。現時点では非対応と明文化する。理由は §2・§3 の契約整理から、
  案 B（主案）の前提となる基礎演算（`detach`／`sub`／スカラー演算／ScalarOp dispatch）が本基準コミット
  時点でまだ揃っていないため。再開条件は次の 3 件の完了:
  - #1612（`detach`／`no_grad`。合成で代替できる範囲の確定）
  - #1593（`sub`／`neg`／スカラー演算）
  - #1634（ScalarOp dispatch。汎用 elementwise の受け皿）
- **段階 1（前提充足後・別イシュー）**: 主案 **案 B**（host-`Tensor<f32>` に閉じた `CustomFunction`）、
  代替 **案 C**（固定集合の勾配差し替え Op）を比較実装 issue の起票候補とする。初期スコープは
  fresh 経路・単一出力・host `Tensor<f32>`・CPU/CUDA/Metal 共通の host 実行に限定する。

判断根拠:
- 案 B は `Op` の閉鎖性への影響を「1 variant 追加」に抑え、`backward_impl` の実行モデル
  （不変借用・逆走査）を変えない
- 案 B′ は REQ-9 2026-09-12 追記「引き続き対象外」・`api_surface.rs` の機械検査に正面から抵触するため不採用
- 案 A は独自 backward を一切表現できないため退ける。`Module` 非公開・`Sequential::add_module`
  欠如は `Sequential` への登録経路のみを塞ぐ別の制限であり、`Var` 直接合成としての案 A 自体への
  到達を妨げるものではない
- 案 D は「ユーザー定義」の演算（任意の forward/backward ペア）には届かず、承認済み演算の
  バックエンド選択に留まる
- 案 E は既存契約への影響がゼロで、前提が揃っていない現時点で選べる唯一の安全な選択肢

## 6. 数値一致・既存テストとの整合

1 階 backward・parity・bit 一致テスト群は不変。段階 1 で `Op::Custom`（案 B）を実装する場合、
ユーザー定義 Op 自体は REQ-2 の判定対象外（ユーザー責任）だが、同一ビット入力・同一 `Tape` 状態・
決定的な `CustomFunction` 実装の場合に限り、host 実行によりバックエンド間で構造的に bit 同一になる
（乱数使用や並列縮約順序依存等の非決定的な実装ではこの限りではなく、GPU 上では H2D／D2H を伴い
性能は保証しない）。この bit 同一性はユーザー定義 Op 自体の出力についての契約であり、グラフ全体の
数値一致を保証するものではない。tolerance／baseline は
本イシュー・段階 1 とも変更しない。

段階 1 で追加するテスト候補（記録のみ・本 PR では実装しない）:
- `Op::Custom` の forward/backward 解析解一致（小形状）
- shape 検証の fail-closed（`output_shape` の宣言と `forward` の実出力形状の不一致を検出）
- resident 経路混在時の型付きエラー（`AutodiffError` への新規 variant 追加が必要になる見込み）
- `Tape: Send` 静的アサーション維持・融合境界であること（`lazy_chain_size == 0` 相当の検査）
- `api_surface.rs` の非後退（新規 trait が `BackendOps` を引数に取らないことの機械検査）

## 7. 公開 API・spec 整合（REQ-12 の両読み）

- **読み (i)**: ユーザー定義 Op（案 B）は `Sigmoid`／`MseLoss` と同じく「`push_eager` により
  融合境界になる副作用」を持つだけで、融合の可否・カーネル選択を利用者が制御する API ではない。
  `BackendOps` を露出せず、`Device` 識別子以外の結線を利用者に渡さない限り REQ-12 と矛盾しない
- **読み (ii)**: REQ-9「引き続き対象外: 任意 `BackendOps` 実装を注入できる推論入口（REQ-12）」に
  隣接し、「利用者コードを演算グラフに挿す」こと自体が融合機構への介入と解釈されうる
- 本 doc の推奨は (i) だが、確定は §8 の承認事項とする。案 B′ は (ii) に該当するため不採用とする
- facade 公開面: Tier 2 列挙済みのため `docs/compat-api-scope.md` §5 の再適用は不要だが、
  新規 `pub trait`（`CustomFunction`）・`Var::custom(...)`／`Tape::custom(...)` 相当の入口の
  API 形状は段階 1 実装時の承認事項とする。`api_surface.rs` の検査（`BackendOps` 非露出）を
  そのまま通る形に限定する

## 8. 段階 1 実装イシューへの引き継ぎ（起票草案。本 PR では起票しない）

- 対象: 案 B（主案）／案 C（代替）の比較実装
- 前提: #1612・#1593・#1634 の完了
- スコープ外: resident／reuse 経路・CUDA Graph capture・複数出力・高階微分（#1622）との併用・
  activation checkpointing（#1624）との併用
- 受入基準候補: 既存テスト非後退・解析解一致・fail-closed 検証・facade 公開面変更時の承認完了

## 9. スコープ外

- 高階微分（#1622。別 doc `docs/autodiff-higher-order-grad-decision.md` が扱う）
- activation checkpointing（#1624）
- `torch.compile` 相当の整理（#1632）
- `nn::Module` の facade 再エクスポート／`Sequential::add_module`（隣接論点。必要なら別 issue で扱う）
- GPU カーネル化されたユーザー定義 Op（案 D 系。将来検討課題）

## 10. 承認事項（実装着手の前提）

1. `Op` enum への trait object variant（`Arc<dyn CustomFunction + Send + Sync>`）追加の可否
2. facade 公開面への新規 `pub trait`／入口メソッド追加（API 形状）
3. REQ-12 の読み（(i) を採るか）。必要なら spec 側への注記提案（`docs/spec/` は本 PR では編集しない）
4. ユーザー定義 Op の数値契約（REQ-2 判定対象外・同一ビット入力／同一 `Tape` 状態／決定的実装に
   限り host 実行で bit 同一・tolerance／baseline 不変）
5. `nn::Module` 再エクスポート／`Sequential::add_module` を本件と切り離すか
6. 段階 1 実装 issue の起票

## 11. 出典

- `crates/autodiff/src/tape.rs`・`grad.rs`・`var.rs`・`backward.rs`・`nn/module.rs`
- `crates/facade/src/compat/sequential.rs`・`crates/facade/tests/api_surface.rs`
- `crates/tensor-core/src/tensor.rs`・`backend_ops.rs`
- `crates/autodiff/src/error.rs`
- `docs/spec/04-requirements.md` REQ-9（2026-09-12 追記）・REQ-12（2026-08-08 注記）
- `docs/compat-api-scope.md` §1.3・`docs/compat-feature-gap.md` §2.11
- `docs/fusion-graph-design.md` §3.3
- `docs/autodiff-higher-order-grad-decision.md`（#1622）
- 関連イシュー: #1570（ルート）・#1573（親）・#1622・#1612・#1593・#1634・#1624・#1632

## 12. HEAD 基準での確定（#1945）

対応イシュー #1945（親 #1944〈段階 0 解除の実装ツリー〉。実装は兄弟イシュー #1946）。
段階 0 の再開条件 3 件（#1612〈detach／no_grad〉・#1593〈sub 等〉・#1634〈ScalarOp dispatch〉）が
いずれも CLOSED になったことを受け、`Op` enum への trait object variant（案 B）を HEAD
（`origin/main` `92d75265`）のコードで再確認し、`CustomFunction` trait の境界・承認事項を確定する。
本節はドキュメントのみの追記であり、`crates/` 配下のコード変更は行わない。

### 12.1 前提充足の確認

- #1612（`detach`／`no_grad`）: CLOSED。`Var::detach`（§12.2 参照）・`Tape::var_no_grad` 実装済み
  （`docs/autodiff-nograd-leaf-dinput-skip-decision.md`）
- #1593（`sub`／`neg`／スカラー演算）: CLOSED。`Var::sub`・`Var::neg` 実装済み（§12.2 参照）
- #1634（ScalarOp dispatch）: CLOSED。`ScalarUnaryOp`／`ScalarBinaryOp` 実装済み
  （`docs/scalar-op-dispatch-design.md`）

3 件とも実装済みのため、§5 で確定した段階 0（現時点非対応）の再開条件は充足済みと判断する。

### 12.2 HEAD で変わった事実（§2 の記述の更新）

§2 の表は基準コミット `2d434c9a` 時点のスナップショットとして不変のまま残す。HEAD
（`92d75265`）で確認し直した事実は次のとおり（file:line は本追記時点の HEAD 実測）。

| §2 時点の記述 | HEAD の事実 | 出典 |
|---|---|---|
| `Op` は 29 variant のクローズド enum | `pub(crate) enum Op`（`#[derive(Debug, Clone)]`）は約 69 variant へ増加。`ScalarUnary`／`ScalarBinary`（#1634）・RNN／LSTM／GRU 系・Pooling・Conv2d 等が追加済み。`Copy` を持たない理由（`CrossEntropyLoss` 等の非追跡ペイロード）は不変 | `crates/autodiff/src/tape.rs:90-91` |
| `grad::vjp(op: &Op, ...)` が `op.clone()` して網羅 match | シグネチャは `vjp(op: &Op, out_value: &Tensor<f32>, upstream: &Tensor<f32>, nodes: &[TapeNode], ops: &dyn BackendOps, resident: Option<&dyn ResidentResolver>, tape_id: TapeId, tape_epoch: u64) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError>` へ拡張済み（`Op::LinearResident` の resident 勾配経路のため `tape_id`／`tape_epoch` が追加）。`op.clone()` して網羅 match する構造自体は不変 | `crates/autodiff/src/grad.rs:188-208` |
| `Op::` を参照する網羅 match は `grad.rs`・`is_lazy_elementwise`／`is_view` 等 | 網羅 match（ワイルドカードなし）の腕は `Op::Custom` を新設する場合の必須改修点として次を確認: `Op::is_lazy_elementwise`（elementwise 融合対象か）・`Op::is_view`（view 系か）・`Op::is_checkpoint_eligible`（checkpoint 再計算対象か）・`Op::for_each_input`（`requires_grad`／poison 前方伝播の入力列挙）・`grad::vjp` 本体の match。いずれも `crate::tape::Op` 定義直後の同一ファイル内メソッド | `crates/autodiff/src/tape.rs:1178`（`is_lazy_elementwise`）・`1192`（`is_view`）・`1217`（`is_checkpoint_eligible`）・`1395`（`for_each_input`） |
| `backward_impl` は単一の不変借用で完結 | #1624（activation checkpointing）により「反復ごとに `self.nodes.borrow()` を取得し drop する」構造へ再構成済み。`grad::vjp` は依然その借用が生きている間に呼ばれるため、ユーザー `backward` からの `Tape` 再入（`push_*` 呼び出し）が `RefCell` panic になる契約（§3 項 7）自体は不変 | `crates/autodiff/src/backward.rs:194`（`backward_impl`）。§3.4／§3.5 の経緯は `docs/autodiff-checkpoint-design.md` |
| `Tape: Send` は静的アサーションで固定 | 不変。`fn assert_send<T: Send>() {} assert_send::<Tape>();` | `crates/autodiff/tests/fusion_backend_integration.rs:388-393` |
| `AutodiffError` に「未対応演算」専用 variant はない | 不変（`#[non_exhaustive]` の enum は不変）。variant は `Shape`／`Backward`／`TapeMismatch`／`InvalidArgument`／`Backend`／`GradientTrackingDisabled`／`DeviceMismatch` の 7 種 | `crates/autodiff/src/error.rs:15-21,72,83` |
| facade は `AutodiffError` を再エクスポート済み | 不変 | `crates/facade/src/lib.rs:134` |
| `Var` の公開演算に `sub`／`neg`／`detach` は未実装（OPEN） | 実装済み。`Var::detach`（専用 `Op::Detach` は追加せず既存葉ノード機構を再利用）・`Var::neg`・`Var::sub`・`Var::to_tape`・`Var::checkpoint_from` | `crates/autodiff/src/var.rs:275`（`detach`）・`351`（`to_tape`）・`682`（`neg`）・`746`（`sub`）・`2257`（`checkpoint_from`） |
| `api_surface.rs` の機械検査（3 種） | 不変（テスト関数名・検査内容は同一。行番号のみ更新）: `facade_does_not_reexport_tape_or_backend_ops`・`facade_public_functions_do_not_accept_backend_ops_argument`・`compat_public_functions_do_not_accept_raw_autodiff_tape_argument` | `crates/facade/tests/api_surface.rs:70,100,164` |
| `nn::Module` 再エクスポート／`Sequential::add_module` の欠如 | 不変（§9 のスコープ外事項のまま。`compat::Sequential::add_*` は個別メソッド追加方式が定着済み〈#1603／#1714 等〉であり `add_module` 汎用入口は依然なし） | `crates/autodiff/src/nn/module.rs`・`crates/facade/src/compat/sequential.rs` |

### 12.3 案 A（合成のみ）の再評価

前提の `detach`（#1612）・`sub`（#1593）が実装済みになったため、§4 で「検証済みの事実ではなく
再開条件として記録するに留める」としていた「合成のみで STE／gradient reversal を表現できる
可能性」を HEAD のシグネチャで確認する。

- `Var::detach(&self) -> Result<Var<'t>, AutodiffError>`（既存葉ノード機構の再利用。専用 `Op` を
  追加しないため、勾配追跡を切った上で元の値をそのまま新しい葉として持ち込む）
- `Var::sub(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError>`

この 2 つのシグネチャから、`x + (f(x) - x).detach()`（`f(x)` の値を forward に使いつつ backward
では `x` への勾配のみを流す straight-through estimator）を**型の上では**構成できることを確認した
（`add`・`sub`・`detach` はいずれも既存 `Var` 演算のみで完結し、`Op::Custom` を必要としない）。
ただし本イシューはコード変更を含まないためこの合成の**実行結果**（数値・勾配の検証）までは行って
いない。実行確認は「検証済み」ではなく「型シグネチャ上の到達可能性の確認」に留まる。

この確認は§4・§5 の結論（案 A は独自 backward を一切表現できないため退ける）を変更しない。
固定パターン（STE・gradient reversal 等）は合成で到達できる可能性があるが、任意の
forward/backward ペアをユーザーが自由に定義する用途には引き続き届かないため、**案 B が主案の
まま不変**とする。

### 12.4 `CustomFunction` trait の確定（案 B・AC-1）

前提充足（§12.1）・HEAD 事実の再確認（§12.2）を踏まえ、案 B の trait 境界を次のとおり確定する。
これは実装可能な設計としての確定であり、実装自体は #1946 へ引き継ぐ。

```rust
// crates/autodiff 内部（pub(crate) または autodiff クレート内 pub。facade 公開は §12.5(b) 未承認）
pub trait CustomFunction: Send + Sync + 'static {
    /// ログ・エラーメッセージ表示用の識別名。`Op::Debug` の手書き実装が参照する。
    fn name(&self) -> &str;

    /// 入力 shape 列から出力 shape を宣言する。`forward` の実出力 shape との
    /// 不一致は fail-closed（型付き `Err`）で検出する（§6「shape 検証」）。
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>, AutodiffError>;

    /// host `Tensor<f32>` のみを受け渡す（`BackendOps` 非露出。REQ-12 §7 読み (i)）。
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>, AutodiffError>;

    /// 戻り値は `inputs` と同じ長さ。`requires_grad` は `inputs` と同じ長さで、
    /// 各要素は呼び出し元（`grad::vjp`）が該当入力について判断済みの要否
    /// （`TapeNode::requires_grad` 前方伝播の結果。
    /// `docs/autodiff-nograd-leaf-dinput-skip-decision.md`）を渡す。
    /// `requires_grad[i] == false` の入力は計算を省略し `None` を返してよい
    /// （`Some` を返すこと自体は禁止しない。無駄な計算を避けるための情報提供であり、
    /// `backward_impl` 側は `requires_grad[i] == false` の要素を受け取った場合
    /// `Some`／`None` のいずれでも安全に破棄する）。`requires_grad[i] == true` の
    /// 入力に対して `None` を返した場合は fail-closed で `Err` とする
    /// （勾配欠落をユーザー実装の不備として検出するため）。
    /// `Some` の要素は対応する入力と同じ shape でなければならず、不一致は
    /// fail-closed で拒否する。
    fn backward(
        &self,
        inputs: &[&Tensor<f32>],
        out_value: &Tensor<f32>,
        upstream: &Tensor<f32>,
        requires_grad: &[bool],
    ) -> Result<Vec<Option<Tensor<f32>>>, AutodiffError>;
}
```

決定点ごとの根拠は次のとおり。

| 決定点 | 確定方針 | 根拠 |
|---|---|---|
| 保持形 | `Op::Custom { inputs: Vec<NodeId>, func: Arc<dyn CustomFunction> }` | `Op: Clone` を保つには `Arc` のクローンで十分。`Tape: Send`（§12.2）を保つには `Arc<dyn CustomFunction>: Send` の条件である `CustomFunction: Send + Sync` が必須 |
| `'static` 境界 | trait 自体に `'static` を課す | `&Tape` や `Var<'t>` を捕捉した実装を型で弾き、backward からの Tape 再入（§3 項 7・§12.2 の `RefCell` panic 契約）を構造的に不可能にする |
| `Op: Debug` との整合 | `Op` 全体の `#[derive(Debug)]` は維持せず、`Op::Custom` 腕のみ `.name()` を出力する手書き `Debug` 実装（または `Op` 全体を手書き `Debug` へ切替え）に置き換える | ユーザー実装に `Debug` を強制しない。`Arc<dyn CustomFunction>` 自体は `Debug` を実装できないため `derive` のままでは追加不可 |
| メソッド集合 | `name`／`output_shape`／`forward`／`backward` の 4 メソッド | 既存案 B（§4 表）を踏襲。`ctx.save_for_backward` 相当は持たず、backward は `inputs`＋`output`＋`upstream` のみから計算する（tape 側の追加状態を持たせない） |
| backward の戻り値 | `Vec<Option<Tensor<f32>>>`（長さ・shape 不一致は `Err`）。`backward` 自身にも `requires_grad: &[bool]`（`inputs` と同じ長さ）を渡す | 当初案（`requires_grad` を渡さない）では `backward_impl` の寄与破棄が VJP 実行後に起きるため、ユーザー実装はどの入力で計算を省略してよいか判断できず `None` 枠が無駄な確保回避の役に立たなかった（codex-review 指摘。§12.2 で棚卸し済みの `TapeNode::requires_grad`／`for_each_input` を `grad::vjp` 側で入力ごとに解決し `backward` へ渡すことで解消）。`requires_grad[i] == false` の入力への寄与は `backward_impl` 側で改めて捨てる契約（`docs/autodiff-nograd-leaf-dinput-skip-decision.md`）と整合させる。`requires_grad[i] == true` に対する `None` は fail-closed で拒否し、勾配欠落の実装不備を早期検出する |
| 純関数・冪等契約 | `forward`／`backward` は決定的・副作用なしとする。同一ノードの `backward` は複数回呼ばれ得る | `Tape::backward_accumulate`（#1749）・`retain_graph` 常時保持契約により VJP が複数回走るため |
| エラー型 | 4 メソッドとも `AutodiffError` を返す（新規エラー型は起こさない） | facade が既に再エクスポート済み（§12.2）・`#[non_exhaustive]` のため将来 variant 追加も非破壊 |
| 入口 | `Tape::custom(&self, func: Arc<dyn CustomFunction>, inputs: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError>`（`fandhe_ai_autodiff::Tape` の inherent method。autodiff 内部 `pub fn`） | `Var` ではなく `Tape` に置く。facade は `pub use fandhe_ai_autodiff::Var` で `Var` 型そのものを再エクスポートしている（`crates/facade/src/lib.rs:134`）ため、`Var` に `pub fn custom` を生やすと facade 非公開の主張（§12.5(a)/(b) の分離）に反し `fandhe_ai::Var::custom(...)` として自動的に到達可能になってしまう（codex-review 指摘）。一方 `fandhe_ai_autodiff::Tape` 自体は facade の `api_surface.rs` 機械検査（`facade_does_not_reexport_tape_or_backend_ops`）で再エクスポート禁止が固定されており、facade 独自の `pub struct Tape(pub(crate) fandhe_ai_autodiff::Tape)`（`crates/facade/src/lib.rs:206`）が転送実装するメソッドのみが facade から呼び出し可能（`Tape::var_no_grad` の前例と同型。`crates/facade/src/lib.rs:236`）。よって内部クレート限定の `Tape::custom` を追加しても、facade 側 `Tape` に対応する転送メソッドを追加しない限り facade からは到達不能。全入力の `tape_id` 一致を検査し不一致は `TapeMismatch` |
| forward 時の評価 | 入力を実体化してから `forward` へ渡し、常に `push_eager`（融合境界） | §3 項 2（`docs/fusion-graph-design.md` §3.3）を維持。`output_shape` の宣言と実出力 shape の不一致は fail-closed |
| 各網羅 match の腕（§12.2 で棚卸し） | `is_lazy_elementwise` = false／`is_view` = false／`is_checkpoint_eligible` = **false**（ユーザーコードの再実行結果が bit 同一である保証がないため安全側に倒す。checkpoint 区間内で解放されない）／`for_each_input` = `inputs` 全件を yield（`requires_grad`／poison 伝播が自動的に乗る）／`grad::vjp` = `inputs`・`out_value` を実体化し、各入力の `TapeNode::requires_grad`（`for_each_input` と同じ情報源）から `requires_grad: &[bool]` を組み立てて `func.backward` へ渡し、`requires_grad[i] == true` かつ `None` の要素があれば `Err` として拒否したうえで `(NodeId, Tensor)` へ変換 | 既存機構（`docs/autodiff-checkpoint-design.md`・`docs/autodiff-nograd-leaf-dinput-skip-decision.md`）にそのまま乗せる |
| resident／reuse 経路との相互作用 | `Op::ResidentLeaf` 由来（ホスト値なし）の入力が混ざる場合は型付き `Err` で拒否する（`materialize_fallible` の既存契約に落とし込む） | §3 項 6・§9 のスコープ外整理を踏襲。GPU tape 上でも host 実行とし、性能は保証しない |
| 数値契約 | §6 を踏襲（REQ-2 判定対象外・同一ビット入力／同一 `Tape` 状態／決定的実装に限り host 実行で bit 同一） | 変更なし |

### 12.5 承認事項（AC-2。#1647〈RNN〉の前例を踏襲し内部クレート実装と facade 公開を分離）

- (a) **内部クレート `fandhe_ai_autodiff` で閉じる事項**（#1946 はこの範囲のみで完結可能）:
  - `Op::Custom` variant の追加・`Op` の `Debug` 実装変更
  - `pub trait CustomFunction`（autodiff クレート内 pub。crates.io 公開クレートの一部になる点は
    facade 非公開でも変わらないため、この意味で「内部だが公開物」であることを明記する）
  - `Tape::custom` 入口（`fandhe_ai_autodiff::Tape` の inherent method。autodiff 内部 pub。
    §12.4「入口」行の理由により `Var::custom` ではなく `Tape::custom` とし、facade が
    `Var` 型を再エクスポート済み（§12.4 出典）でも facade 非公開のまま保てることを
    設計上担保する）
  - `AutodiffError` への新規 variant 追加（`#[non_exhaustive]` のため非破壊。要否は実装時に判断）
- (b) **facade 公開面**（**未承認**。#1946 の対象外とし別途承認を得る）:
  - `fandhe_ai::CustomFunction` の再エクスポート
  - facade `Tape` への `custom(...)` 転送メソッド追加（facade 独自 `struct Tape` が
    (a) の `fandhe_ai_autodiff::Tape::custom` を呼び出す薄いラッパー。追加するまでは
    facade から到達不能）
  - `api_surface.rs` の検査拡張（新規 trait が `BackendOps` を引数に取らないことの機械検査）
- (c) REQ-12 の読み（(i) を採る）の確認。§7 の推奨方針は不変。spec 側への注記提案の要否は
  未承認のまま（`docs/spec/` は本 PR・#1946 とも編集しない）
- (d) 数値契約（REQ-2 判定対象外・同一ビット入力／同一 `Tape` 状態／決定的実装に限り host 実行で
  bit 同一・tolerance／baseline 不変）の確認
- (e) `nn::Module` の facade 再エクスポート／`Sequential::add_module` は本件と切り離したまま
  据え置く（§9 のスコープ外整理を維持）

本 PR は上記いずれの承認も取得しない。実装（#1946）は (a) の範囲のみで着手可能であり、(b) は
別途のユーザー承認を得てから着手する。

### 12.6 #1946 への引き継ぎ（テスト候補の追補）

§6 の既存テスト候補一覧に加え、次を追加する:

- `Tape::backward_accumulate`（#1749）で同一 `Op::Custom` ノードの `backward` を 2 回呼んだ場合の
  勾配蓄積の整合
- `backward` に渡す `requires_grad: &[bool]` が `TapeNode::requires_grad` の実値と一致すること、
  および `requires_grad[i] == false` の入力に対して `backward` が `None` を返すケースの正しい取り扱い
  （§6 の「resident 経路混在時の型付きエラー」に加える）
- `requires_grad[i] == true` の入力に対して `backward` が `None` を返した場合に fail-closed で
  `Err` となること（勾配欠落の実装不備検出）
- checkpoint 区間内に `Op::Custom` が混在した場合、`is_checkpoint_eligible = false` により
  当該ノードが再計算対象から除外され続けることの確認
- `for_each_input` 経由の poison 伝播（`docs/autodiff-checkpoint-design.md` §3.5）が
  `Op::Custom` の子孫へも正しく伝播すること
- `Tape: Send` 静的アサーション（`fn assert_send::<Tape>()`）が `Op::Custom` 追加後も
  コンパイルを通ること

### 12.7 出典（本節追加分）

- HEAD（`origin/main` `92d75265`）: `crates/autodiff/src/tape.rs:90-91,1178,1192,1217,1395`・
  `crates/autodiff/src/grad.rs:188-208`・`crates/autodiff/src/backward.rs:194`・
  `crates/autodiff/src/var.rs:275,351,682,746,2257`・`crates/autodiff/src/error.rs:15-21,72,83`・
  `crates/facade/src/lib.rs:134`・`crates/facade/tests/api_surface.rs:70,100,164`・
  `crates/autodiff/tests/fusion_backend_integration.rs:388-393`
- `docs/autodiff-checkpoint-design.md`（§3.4／§3.5 poison 伝播・`recompute_value` 反復化）
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（`requires_grad` 前方伝播・`Tape::var_no_grad`）
- `docs/autodiff-retain-graph-accumulate-decision.md`（`Tape::backward_accumulate`）
- `docs/scalar-op-dispatch-design.md`（#1634 ScalarOp dispatch）
- `docs/autodiff-rnn-cell-tape-design.md`（内部クレート実装・facade 公開分離の前例。#1647）
- 関連イシュー: #1945（本節）・#1946（実装引き継ぎ先）・#1944（親）・#1612・#1593・#1634（前提。
  いずれも CLOSED）

## 13. 実装記録（#1946）

§12.5 (a)（内部クレート限定）の範囲で実装済み。§12.4 の確定設計をそのまま実装しており、
以下は実装時に判明した差異・実測のみを記録する。

### 13.1 イシュー文面との差異

イシュー #1946 の文面は `Box<dyn CustomFunction>`・入口名 `apply_custom` を挙げていたが、
§12.4（本 issue #1945 の確定設計）は `Arc<dyn CustomFunction>`・入口名 `Tape::custom` を
確定させており、実装は §12.4 に従った。`grad::vjp` が `op.clone()` して網羅 match する契約
（`Op: Clone` が必須）のため、`Box<dyn Trait>`（`Clone` 不可）では成立せず `Arc` が必須である。

### 13.2 実装ファイル・構成

- `crates/autodiff/src/custom.rs`（新規）: `pub trait CustomFunction`（`name`／`output_shape`／
  `forward`／`backward` の 4 メソッド）・`pub(crate) struct CustomFn(Arc<dyn CustomFunction>)`
  （`derive(Clone)` + 手書き `Debug`。`.name()` のみ出力し `Op` 全体の `derive(Debug)` を保つ）。
- `crates/autodiff/src/tape.rs`: `Op::Custom { inputs: Vec<NodeId>, func: CustomFn }` を追加。
  `is_checkpoint_eligible` へ `Op::Custom => false` の腕・`for_each_input` へ `Op::Concat` と
  合流する腕（`inputs` 全件 yield）を追加。`Tape::custom` を新設（`Var::cat` と同じ「層 1で
  実体化 → `RefCell` 借用を閉じてからユーザー `forward` を呼ぶ」規律）。
- `crates/autodiff/src/grad.rs`: `vjp` の match に `Op::Custom` の腕を 1 つ追加（既存の腕は
  1 行も変更していない——受入基準 1「既存 backward は bit 同一のまま」の構造的担保）。
- `crates/autodiff/src/lib.rs`: `mod custom;`・`pub use custom::CustomFunction;` とクレート doc
  1 段落を追加。`AutodiffError` に新規 variant は追加していない（既存 `InvalidArgument`／
  `Shape`／`Backward`／`TapeMismatch` で全ケースを表現できたため）。

### 13.3 コンパイルエラー箇所の実測（受入基準 3）

`Op::Custom` variant を追加した直後（`for_each_input`／`is_checkpoint_eligible` へ腕を
追加する前）に `cargo check -p fandhe-ai-autodiff` を実行し、非網羅 match によるコンパイル
エラーが計画どおり `is_checkpoint_eligible`・`for_each_input`・`grad::vjp` の 3 箇所のみで
あることを実測確認した（他の `match &node.op`／`match self`（`is_lazy_elementwise`・
`is_view`・`recompute_value` 内の各 match 等）は `matches!` マクロまたは `_` ワイルドカードを
持つため非到達で影響なし）。

### 13.4 テスト

- `crates/autodiff/src/tape.rs::custom_op_tests`（`pub(crate)` API を直接使う必要がある
  契約のみ。クレート内単体テスト・4 件）: `is_checkpoint_eligible() == false` 固定・
  `for_each_input` が `inputs` を発生順に yield・`Tape::custom` が `Op::ResidentLeaf` 入力を
  `InvalidArgument` で拒否・空 `inputs` を `InvalidArgument` で拒否。
- `crates/autodiff/src/custom.rs::tests`（1 件）: `CustomFn` の `Debug` 出力が `.name()` の
  みを表示すること。
- `crates/autodiff/tests/custom_function.rs`（公開 API のみを経由する統合テスト・11 件）:
  自作 `CustomRelu` と組み込み `Var::relu` の grad 経路が forward／`dx`／`dw` すべて bit
  完全一致（受入基準 2）・2 入力の解析的 `CustomMul`・`output_shape` 宣言と実出力の不一致
  検出・`backward` 戻り値の長さ不一致／shape 不一致／`requires_grad[i]==true` への `None`
  のいずれも fail-closed（`AutodiffError::Backward`／`Shape`）・`requires_grad` 前方伝播
  （`Tape::var_no_grad` 入力に対し `false` が正しく渡り `None` を受理し、`Some` を返しても
  破棄されること）・`Tape::backward_accumulate` で `backward` が 2 回呼ばれ勾配が単純 2 倍
  になること（bit 一致）・クロステープ検査・`Arc<dyn CustomFunction>: Send + Sync` の静的
  アサーション。
- `crates/facade/tests/api_surface.rs`（否定ガード 2 件追加）:
  `facade_does_not_reexport_custom_function`（`pub use` に `CustomFunction` を含まない）・
  `facade_tape_does_not_expose_custom_forwarding_method`（facade 独自 `struct Tape` に
  `pub fn custom(` が存在しない）。§12.5 (b) 承認取得時にこれらのガードを更新・撤去する。

checkpoint 区間との相互作用（`Op::Custom` が常に checkpoint 解放対象から除外される）・
`for_each_input` 経由の poison 伝播は、`is_checkpoint_eligible`／`for_each_input` の実装
自体が `Op::Concat`・既存の非適格演算群（`Op::Where`／`Op::Gather` 等）と完全に同型の
網羅 match の腕として構成されているため、既存の checkpoint／poison 回帰テスト
（`tests/checkpoint.rs`・`tests/checkpoint_review_1624.rs`）が固定する不変条件がそのまま
`Op::Custom` にも適用される（個別の統合テストとしては追加していない）。

### 13.5 数値契約・実機実測

`CustomFunction::forward`／`backward` は常に host 実行（`BackendOps` 非経由）のため、
REQ-2（バックエンド間数値一致）の対象外・CUDA／Metal 実機実測の対象外である
（§6・§12.5 (d) のとおり）。

### 13.6 スコープ外（§12.5 (b)・実装せず）

facade 公開面（`fandhe_ai::CustomFunction` 再エクスポート・facade `Tape::custom` 転送
メソッド・`api_surface.rs` の公開面拡張検査）は本 issue の対象外のまま、別途ユーザー
承認を得てから着手する。

## 14. `create_graph`（高階微分。#1942／#1943）との関係

本 issue（#1946）と `create_graph`（`docs/autodiff-higher-order-grad-decision.md`。
イシュー #1942／#1943）は並行して実装され、`origin/main` へ取り込む際に `Op` の網羅
match（`Op::supports_create_graph()`。同 doc §8 の 69 variant 分類は `Op::Custom`
新設前のもので同 variant を含まない）が `Op::Custom` を欠いたままコンパイル不能に
なることが判明した（PR #1996 マージ時。イシュー #1946）。

`Op::Custom` は `create_graph` の**対象外**（`Op::supports_create_graph()` は
`Op::Custom { .. } => false` を返す）と確定する。理由: `CustomFunction::backward` は
上流勾配（`upstream: &Tensor<f32>`）を受け取り数値テンソルの VJP のみを返す契約
（§12.4／§13.2）であり、子テープ（`create_graph.rs::backward_create_graph`）が要求する
「入力ノードを起点に `Var` 演算として再生可能な演算列」を一切持たない。ユーザー定義
`forward`／`backward` はブラックボックスの数値関数であり、子テープ上でその微分演算
自体を記録する手段が構造的に存在しないため、対応するには `CustomFunction` に
二階微分専用の別メソッド（例: `backward_of_backward`）を追加する API 拡張が必要になる
（本 issue のスコープ外）。

拒否時の挙動は他の非対象 Op（`Op::ScalarUnary`／`Op::Softmax` 等）と同型で、
`Tape::backward_create_graph` の入口検査（`validate_ancestors`）が子テープへ一切
書き込む前に型付き `Err(AutodiffError::Backward(_))` を返す（fail-closed）。
`crates/autodiff/tests/create_graph.rs::create_graph_rejects_unsupported_op_custom`
で固定した。

## 15. facade 公開の保留記録（イシュー #2064）

イシュー #2064「facade: custom autograd Function の公開面追加」の実装着手時
（2026-09-22・main HEAD `5d78c638`）に、`gh issue view 2064 --comments`・
`gh issue view 2059 --comments`（親 issue）を確認したところ、§12.5 (b)
「facade 公開面（`CustomFunction` 再エクスポート・facade `Tape::custom`
委譲）」を対象とするリポジトリ所有者の明示的な承認コメントは存在しなかった
（#2064 のコメントは 2026-09-19T15:45:46Z の「実装保留（ユーザー承認待ち）」
通知 1 件のみ、親 #2059 はコメント 0 件）。issue が起票されていること自体は
承認事項の承認にはならない（前例: #2063「facade: 高階微分 API の公開面追加」
も同様に未承認のまま保留し、PR #2208（`ef415e6d`）で facade／autodiff src を
一切変更せず否定ガード＋保留記録 doc のみをマージした。同 doc `docs/autodiff-
higher-order-grad-decision.md` §15 と本節は同型）。

自動運転（承認待ち不可）かつ判断は安全側に倒す方針、`docs/compat-api-scope.md`
§5（範囲拡張は経路 1／2 の承認必須）、`.claude/rules/security.md`（自己修復に
よる無断拡大禁止）に基づき、本イシューでは `crates/facade/src/**`・
`crates/autodiff/src/**` を一切変更せず、次の否定ガード 4 件（新規 4 件。
うち AC-4〈高度な合成用 Op の非露出〉分は `crates/facade/tests/api_surface.rs`
の新規 2 件）を追加して「facade 未公開」状態を機械固定した。#2064 の
2026-09-19T15:45:46Z コメント本文は AC-4 分を「3 件」と記載しているが、
実装した新規ガードは下記のとおり AC-4 分 2 件・§12.5 (b) 第 3 項分 2 件の
計 4 件であり、本節（実装記録）の件数を正とする（コメント本文側の件数表記の
誤りはロジックに影響しないため本 PR では修正しない。GitHub コメントは
投稿後に本文編集していない）:

- `crates/facade/tests/api_surface.rs::facade_does_not_reexport_custom_function`
  （既存。§12.5 (b) 未承認のまま維持する旨を doc comment へ追記）
- `crates/facade/tests/api_surface.rs::
  facade_tape_does_not_expose_custom_forwarding_method`（既存。同上）
- `crates/facade/tests/api_surface.rs::
  facade_public_functions_do_not_take_custom_function`（新規。`src/` 全体に
  `CustomFunction` 識別子が一切現れないことを固定し、`pub use` 行のみを
  見る既存ガードの死角〈`pub use` を経由しない合成入口〉を塞ぐ）
- `crates/facade/tests/api_surface.rs::
  compat_sequential_does_not_expose_custom_add_method`（新規。`Sequential::
  add_custom` 相当の合成メソッドが生えていないことを固定する）
- `crates/autodiff/tests/architecture_boundaries.rs::
  custom_function_trait_signatures_are_host_tensor_only`（新規。§12.5 (b)
  第 3 項「新規 trait が `BackendOps` 等を引数に取らないことの機械検査」。
  `CustomFunction` trait 定義本体のみを抽出し `BackendOps`／`Tape`／`Var`／
  `Device`／`NodeId` を含まないことを固定する）
- `crates/autodiff/tests/architecture_boundaries.rs::
  autodiff_src_does_not_declare_pub_fn_custom_on_var`（新規。同項。`crates/autodiff/src` 配下の
  全ファイルを走査し `Var` の impl ブロックに `pub fn custom`／
  `add_custom` 宣言が無いことを固定し、`Var::custom` という facade
  再エクスポート経由の別到達口が生えないことを構造的に保証する）

**ソース文字列走査ガードの多層防御化（PR #2212。codex-review／Bugbot 指摘の
複数ラウンドを経て確定）**: `autodiff_src_does_not_declare_pub_fn_custom_on_var`
系のソース走査（heuristics）は、`Var` への trait impl（`impl Trait for Var { fn
custom(&self) {} }`。トレイトのメソッドは可視性修飾子を書かずに宣言でき
トレイト自体の可視性がそのまま公開 API として機能する）・`Var` の import
alias／type alias（`use ... Var as V`・`type X = crate::Var;` のようなパス
修飾形・`use path::Var::{self as V}` の `self` 再エクスポート形）・alias
宣言が `mod` 以外のブロックスコープ（`fn`／`impl`／`trait` 本体等）に
隠れているケースなど、複数ラウンドの指摘に応じて検出範囲を段階的に拡張して
きた（`crates/autodiff/tests/architecture_boundaries.rs::
var_impl_block_bodies_with_aliases_and_kind`／`type_alias_target_is_var`／
`find_var_aliases_in_use_body`／`find_var_alias_declarations` の各 doc
コメント参照）。この種の字句レベルの走査は性質上「新しい迂回手口が
指摘される→検出ロジックを拡張する」というレビュー往復から完全には
逃れられないため、**`crates/facade/src/lib.rs` に `compile_fail,E0599`
doctest（`VarCustomHoldDoctestGuard`。facade の全 `pub mod` を glob
import したスコープで `Var::custom`／`.custom(...)`／`.add_custom(...)`
を呼ぼうとするとコンパイルできないことを rustc の名前解決そのもので
固定する）を追加し、これを本命ガードと位置づけた**。ソース走査は
「変更差分の早期発見・迂回経路の類型化」を担う多層防御の 1 層という
位置づけへ変わり、doctest 側は trait impl 経由・alias 経由・
再エクスポート経由のいずれであっても実際に facade の公開面を
glob import した状態で失敗することを機械的に保証するため、走査ロジック
の見落としに対する耐性が高い。doctest が glob import する `pub mod`
集合と `src/lib.rs` の実宣言集合のドリフトは
`crates/facade/tests/api_surface.rs::
custom_function_hold_doctest_globs_all_pub_modules` が固定する。
禁止呼び出し 3 種（`.custom(...)`・`Var::custom(...)`・`.add_custom(...)`）
は**それぞれ独立した `compile_fail` ブロック**で検証する。rustdoc は
ブロック全体が失敗すれば合格と判定するため、1 ブロックへまとめると
1 種だけの部分公開を検出できない（codex-review 指摘・PR #2212）。
上記ドリフト検査はブロック単位で glob 集合を突き合わせ、ブロック数が
3 であることも固定する（1 ブロックへの再統合を拒否）。

**glob 対象・ドリフト検査・否定ガードの再帰化（PR #2212 P1 是正・追加ラウンド）**:
上記ドリフト検査（`custom_function_hold_doctest_globs_all_pub_modules`）は
当初 `src/lib.rs` 直下の `pub mod` 宣言しか見ておらず、`nn::rnn`・
`interop::onnx`・`interop::safetensors` のようなネストした公開面（かつ
将来 `compat::extensions` のような新設モジュールが追加された場合）に
生える trait 経由の合成入口を見逃しうる指摘を受けた。`crates/facade/
tests/api_surface.rs::collect_public_module_paths`（トークン列を
`scan_top_level_pub_mods` で再帰走査し、`pub mod name;` は解決先ファイル
〈`<dir>/<name>.rs` または `<dir>/<name>/mod.rs`〉へ辿り、インライン
`pub mod name { ... }` は本体トークン列を直接再帰する。非 `pub` な
`mod`・`fn`/`impl` 等のブレース内部は到達不能として丸ごと読み飛ばす）
へ置き換え、3 つの `compile_fail` doctest ブロックと足場成功版すべてに
`use fandhe_ai::nn::rnn::*;`・`use fandhe_ai::interop::onnx::*;`・
`use fandhe_ai::interop::safetensors::*;` を追加した。あわせて、
`pub fn custom`／`pub fn add_custom` の宣言のみを検査していた既存の
2 否定ガード（`facade_tape_does_not_expose_custom_forwarding_method`・
`compat_sequential_does_not_expose_custom_add_method`）を補完する形で、
`facade_source_declares_no_custom_fn_in_any_context`（`declares_fn_named`。
可視性キーワード・宣言文脈〈inherent impl・trait impl・trait 定義・
自由関数のいずれか〉を問わず facade 全ソースの `fn custom`／
`fn add_custom` 宣言を検出する）を追加した。

承認取得後に実施する変更範囲（事前提示。#2064 の保留コメント本文と同旨。
ただし AC-4 の否定ガード件数は上記の実装記録どおり 2 件が正で、保留コメント
本文中の「3 件」という表記はその後の実装で確定した数と一致していない）:

- `crates/facade/src/lib.rs`: `pub use fandhe_ai_autodiff::CustomFunction;`
  （1 行の再エクスポート）・`impl Tape` への `custom` 委譲メソッド 1 件
  （`self.0.custom(func, inputs)` を返すだけの薄い委譲。`transfer`／
  `rnn_forward_seq` と同型）
- `crates/facade/tests/api_surface.rs`: 上記否定ガード 2 件（既存）を削除し
  正ガードへ差し替え、AC-4 の否定ガード（新規 2 件）は「`CustomFunction` を
  含む `pub fn` 行は `Tape::custom` 1 件のみ」という形へ更新する
- `crates/facade/tests/custom_function_facade.rs`: facade 経由の統合テスト
  新規追加
- `crates/autodiff/src/custom.rs`・`lib.rs`・`tape.rs`: モジュール doc の
  「facade 非公開」記述を「facade 公開済み」へ更新（ロジック変更なし）
- `docs/autodiff-custom-function-decision.md`・`docs/compat-api-scope.md`・
  `docs/compat-feature-gap.md`・`docs/perf/framework-compare-feature-
  matrix-0.9.0.md`・`docs/README.md`: 実装記録・適用記録の追記
- `docs/perf/logs/facade-custom-function-2064/README.md`: CUDA／Metal 実機
  実測なしの申し送り（常に host 実行のため REQ-2 対象外）

イシューは close せず、承認取得後に別 PR で経路 B（公開実施）を行う。
