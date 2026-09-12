# custom autograd Function（ユーザー定義 Op）プラグイン機構の設計判断

対応イシュー #1623（親 #1573〈Tier 2〉→ ルート #1570）。位置づけは**設計判断の記録のみ**であり、
本 PR では `crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。数値一致の
複合判定（REQ-2）・tolerance／baseline も変更しない。基準コミット: `origin/main` `2d434c9a`
（#1676「高階微分（grad of grad）の設計判断」マージ後）。

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
| `nn::Module` trait（`forward(&self, tape, input)`／`forward_host`）は `autodiff` で `pub` だが、facade は `nn::LinearVars` のみ再エクスポート。`compat::Sequential` は `Vec<Box<dyn Module>>` を持つが公開ビルダーは `add_linear`／`add_relu`／`add_sigmoid`／`add_tanh` に固定（`add_module` はない）→ 合成ベースの独自層は現行の公開面から到達不能 | `crates/autodiff/src/nn/module.rs:33`・`crates/facade/src/compat/sequential.rs:116,129,135,141` |
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
| **A: 合成のみ（custom backward なし）** | 既存 `Var` 演算の合成で独自 forward を書く | 不要 | 不要 | 抵触なし | 影響なし | 既存演算の VJP のみ（合成の連鎖規則で自動導出） | 追加不要（既存 `Var` 演算のみ） | なし | S（ただし現行公開面からは `Sequential::add_module` 欠如で到達しづらい） |
| **B（主案）: `Op::Custom { inputs: Vec<NodeId>, func: Arc<dyn CustomFunction + Send + Sync> }`** | `trait CustomFunction { fn name(&self) -> &str; fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>>; fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>>; fn backward(&self, inputs: &[&Tensor<f32>], out_value: &Tensor<f32>, upstream: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> }`。host `Tensor<f32>` のみを受け渡し、`BackendOps` は非露出。常に `push_eager`（融合境界） | `Op` に 1 variant 追加（`Arc` 化・手書き `Debug`／`name()` が必要） | 不要（既存の逆走査アルゴリズムのまま） | 抵触なし（`BackendOps` を渡さない） | 影響なし（新規 trait は `BackendOps` を引数に取らない） | host 実行のためバックエンド間で構造的に bit 同一（REQ-2 判定対象外・ユーザー責任） | 新規 `pub trait CustomFunction`＋`Var::custom(...)` 相当の入口（要承認） | #1612（`detach`）・#1593（`sub`／スカラー演算）・#1634（ScalarOp dispatch） | M |
| **B′: 案 B に `&dyn BackendOps` を渡す** | `forward`／`backward` にバックエンド実装への参照を渡しユーザーコードからカーネル選択を可能にする | 同上 | 不要 | **抵触**（任意 `BackendOps` 実装を注入できる公開 API を設けない、REQ-9 2026-09-12 追記の「引き続き対象外」に正面から該当） | `api_surface.rs` の「`pub fn` が `BackendOps` を引数に取ることを禁止」検査に抵触 | 同上 | 抵触するため不可 | — | — |
| **C: 勾配のみ差し替える固定集合の組み込み Op** | `Op::StraightThrough`／`Op::GradReverse(scale)`／`Op::GradClamp` 等を通常の Op として追加（`custom_gradient`-lite） | `Op` に固定個数の variant 追加 | 不要 | 抵触なし | 影響なし | 既存 Op と同じ扱い（parity 体系に自然に乗る） | 追加不要または最小限（`Var::straight_through()` 等の個別メソッド） | #1612（`detach`。役割が重なる） | S〜M（用途ごとに個別 issue が必要） |
| **D: `BackendOps` レベルのレジストリ／テーブル拡張** | 承認済み enum の範囲で backend 実装を選ぶ方式（#1634 ScalarOp dispatch の延長） | 不要（`ScalarUnary`／`ScalarBinary` 等の enum を拡張） | 不要 | 抵触なし | 影響なし | 既存契約のまま | 追加不要 | #1634 | 任意関数は挿せず「ユーザー定義」には届かない。将来 GPU カーネル化が必要になった場合の別軸の案として位置づけ |
| **E: 現時点では非対応と明文化し再開条件を定義する（段階 0）** | 何も実装しない。ギャップ表・spec の「Tier 2・XL」評価をそのまま維持し、再開条件を明記する | なし | なし | 抵触なし | 影響なし | 変更なし | 変更なし | — | 既存契約への影響ゼロ |

案 A は現行の公開面（`Module` が facade 非公開・`Sequential` に `add_module` がない）から独自層を挿す経路が
無く、かつ独自 backward を一切表現できない。ただし `detach`（#1612）＋`sub`（#1593）が揃えば、
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
- 案 A は現行公開面から到達不能（`Module` 非公開・`add_module` なし）で、かつ独自 backward が
  表現できない
- 案 D は「ユーザー定義」の演算（任意の forward/backward ペア）には届かず、承認済み演算の
  バックエンド選択に留まる
- 案 E は既存契約への影響がゼロで、前提が揃っていない現時点で選べる唯一の安全な選択肢

## 6. 数値一致・既存テストとの整合

1 階 backward・parity・bit 一致テスト群は不変。段階 1 で `Op::Custom`（案 B）を実装する場合、
ユーザー定義 Op 自体は REQ-2 の判定対象外（ユーザー責任）だが、host 実行によりバックエンド間で
構造的に bit 同一になる（GPU 上では H2D／D2H を伴い性能は保証しない）。tolerance／baseline は
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
4. ユーザー定義 Op の数値契約（REQ-2 判定対象外・host 実行で bit 同一・tolerance／baseline 不変）
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
