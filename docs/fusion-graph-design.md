# 演算グラフ表現設計（TASK-12.1a）

> 役割・参照元: 本文書は REQ-12（`docs/spec/04-requirements.md:243`）の
> v2 読み替え後タスク分解 TASK-12.1（`docs/spec/05-tasks.md:356`。「自作
> elementwise 融合機構の設計・初期実装」）の第 1 段（TASK-12.1a・本イシュー
> #161）の成果物である。**演算グラフ表現（遅延評価境界を含む）の設計のみ**
> を扱い、コード変更（型の実装）は含まない（`docs/fusion-graph-design.md`
> の新規追加のみ）。後続の連鎖検出（#162）・融合カーネル生成（#163）・
> ディスパッチ統合（#164）・テスト（#165）・GEMM epilogue 融合（#203）は
> 本文書を正本として実装する。体裁は先行の設計イシュー TASK-11.2a
> （#67 → `docs/dispatch-rules-design.md`）に倣う。

## 1. 判断サマリ

- 融合対象は **elementwise 演算連鎖（4〜6 段程度）** を初期スコープとする
  （TASK-12.1 の内容規定、`docs/spec/05-tasks.md:356`）。reduction エピロー
  グ・matmul／softmax を挟む複合ワークロードは初期スコープ外とする。
  - 根拠: v1 PoC-9 実測（`docs/spec/03-poc/poc-9-kernel-fusion/README.md`）
    は単純な elementwise 連鎖（`ew4`／`ew6`／`ew_fanout`）が 2.25〜3.19 倍
    短縮する一方、`.sum()` 等の reduction エピローグは自動融合の対象外、
    matmul をまたぐ連鎖（`ew_matmul_ew`）は融合セグメントが分断されると
    実測している（同 README「生成カーネル数によるパターン別融合適用範囲」
    表）。REQ-12 受け入れ基準も「matmul・softmax を含む複合ワークロードで
    は融合の効果を前提とした性能目標を設定しないこと」と明記する
    （`docs/spec/04-requirements.md:250`）。GEMM epilogue（bias・activation
    の融合）は別イシュー #203 で拡張する。
- 利用者向けの融合制御 API は**提供しない**。融合は内部機構としてのみ働き、
  既存の `Var` 演算経路（`autodiff::eval`）に内部的に組み込む形で実現し、
  新規の公開エントリ関数は追加しない（経路の具体化は §3.5、`BackendOps`
  契約との接続は §3.4 で規定する）。
  - 根拠: REQ-12 受け入れ基準「ライブラリ利用者が明示的に融合を制御する
    API は提供しないこと」（`docs/spec/04-requirements.md:249`）。REQ-11
    読み替え設計（`docs/dispatch-rules-design.md` §1「利用者向けの明示切替
    API は提供しない」）と同方針であり、本ライブラリ全体の一貫した設計
    判断として踏襲する。**この受け入れ基準が要求するのは「利用者が融合を
    制御する API を提供しないこと」であり、「`&dyn BackendOps` を直接呼ぶ
    既存経路にも融合を透過的に適用すること」までは要求しない**（誤読の
    余地があったため本改訂で明記。§3.4 参照）。`&dyn BackendOps` を直接
    呼ぶ既存経路（`ops_for` 経由の呼び出し含む）は本設計の変更対象外とし、
    従来どおり eager・非融合のまま維持する。
- transpose を挟む連鎖は**融合しない（非融合フォールバックへ倒す）**。
  - 根拠: PoC-9 実測（`ew_reshape`）は、fusion **有効時**は transpose が
    メタデータ変換のみで融合セグメントへ取り込まれ、fusion **無効時**は
    実データコピーとして具体化され最大 13.89 倍の性能劣化を招くと確認
    している（同 README、REQ-12 受け入れ基準
    `docs/spec/04-requirements.md:252`）。この 13.89 倍差は「非融合状態の
    ペナルティの大きさ」を示す数値であり、本来は transpose を融合対象に
    含める動機になりうる。しかし v1 のメタデータのみでの取り込みは
    Burn/CubeCL の融合エンジン内部実装（ストライド付きビューを融合
    セグメント内で扱う機構）に依存した挙動であり、本ライブラリの自作
    融合 IR（§2）は現時点でストライド付きビューを表現・伝播する仕組みを
    持たない。これを初期スコープで再現する設計コストは TASK-12.1 の
    「elementwise 演算連鎖（4〜6 段程度）」という規定範囲を超える。
    正当性（誤った実行結果を出さないこと）を優先する安全側の設計として
    初期スコープでは transpose 検出時に融合セグメントを打ち切ることとし、
    **v1 融合有効時の性能水準（PoC-9 実測で最大 13.89 倍差）を初期
    スコープでは達成しないという受け入れコストを明示的に記録する**
    （§6.2 に未決事項として追跡）。

## 2. グラフ表現（IR）の設計

### 2.1 ノード種別

融合対象を閉じた enum で表現する。初期集合は `BackendOps` trait
（`crates/tensor-core/src/backend_ops.rs:63`）が定義する既存 op 集合と
1:1 対応させ、融合機構が扱う演算の全体像を `BackendOps` の実装済み契約
からはみ出させない。

```rust
/// 融合グラフのノード種別（スケッチ。実装は #162 以降）。
/// `BackendOps`（backend_ops.rs:63）の各メソッドと 1:1 対応させる。
pub(crate) enum FusionOp {
    /// リーフノード（グラフへの入力テンソル）。
    Input,
    // elementwise binary（backend_ops.rs の `add`/`mul` に対応）
    Add(NodeId, NodeId),
    Mul(NodeId, NodeId),
    // elementwise unary（`relu`/`exp`/`tanh` に対応）
    Relu(NodeId),
    Exp(NodeId),
    Tanh(NodeId),
    // 融合境界ノード（融合しない。到達時に実体化する。§3 参照）
    Gemm(NodeId, NodeId),
    Sum { input: NodeId, dim: Option<usize> },
    Max { input: NodeId, dim: Option<usize> },
}
```

- elementwise binary／unary の 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）
  が融合の直接対象。`gemm`・`sum`・`max` は **融合境界ノード**として同じ
  enum に含めるが、融合セグメントには組み込まない（§3.2 の実体化条件 (a)
  (b) に対応する印として扱う）。
- `BackendOps` は f32 固定スコープ（`backend_ops.rs:56`「v1 は PoC-v2-5
  実測 API（`MetalOps`）のスコープに合わせて `f32` 固定とする」）であり、
  本融合グラフも同じ f32 固定スコープに揃える。f16 対応は §6 の未決事項
  とする。

### 2.2 グラフ構造

ノード ID＋隣接（入力エッジ）リストによる DAG とする。`autodiff` クレート
の `Tape`／`NodeId` と同型の設計（ノード列 `Vec<FusionNode>` への添字を
表す newtype、`crates/autodiff/src/tape.rs:35`）を踏襲する。

```rust
/// テープ内ノードの識別子（tape.rs:35 の `NodeId` と同型パターン）。
pub(crate) struct FusionNodeId(pub(crate) usize);

pub(crate) struct FusionNode {
    op: FusionOp,
    /// 融合可否判定に使う静的メタデータ（§2.3）。
    meta: NodeMeta,
    /// このノードの出力を入力として参照するノード数（fan-out。§2.4）。
    use_count: usize,
}
```

`autodiff::Tape` と同様、ノードは発生順に `Vec` へ追記され、`FusionOp` は
入力を `FusionNodeId` で保持することで融合可能な部分グラフ（elementwise
のみで閉じた連結成分）を後方から辿って検出できる（#162 が実装する連鎖
検出アルゴリズムの入力形式）。

### 2.3 ノードメタデータ

```rust
/// 融合可否判定に使う静的メタデータ（shape・stride・dtype）。
pub(crate) struct NodeMeta {
    shape: Vec<usize>,
    /// contiguous かどうか。false の場合は transpose／broadcast view を
    /// 示唆し、§1 の非融合フォールバック判定に使う。
    contiguous: bool,
    dtype: DType,
}
```

- `dtype` は `crates/tensor-core/src/dispatch.rs` の既存 `DType`
  （`dispatch.rs:31`）を再利用する。初期は `DType::F32` 固定
  （`BackendOps` の現行スコープ、§2.1 と整合）。
- `contiguous` フラグにより transpose／broadcast view をメタデータで検出
  可能にする。**transpose 混在連鎖（`contiguous == false` のノードを含む
  連鎖）は融合しない**という §1 の境界条件を、この 1 フィールドの真偽値
  判定として型レベルで表現できる設計とする（#165 のテスト対象）。

### 2.4 fan-out の扱い

fan-out（1 つのノード出力が複数ノードから参照される）は `use_count`
フィールド（出力の被参照数）で表現し、**fan-out であること自体を融合
不能条件にしない**。

- 根拠: PoC-9 実測（`ew_fanout` パターン、`a = x + y; b = a * a; c = b + x`）
  で、中間テンソル `a` を 2 回消費する fan-out 連鎖も `ElemwiseFuse` 1 個
  へ完全融合されると確認済みである（`docs/spec/03-poc/poc-9-kernel-fusion/README.md`
  「`ew_fanout` … 融合される（fan-out も対象）」）。fan-out を融合不能条件
  に含めると、この実測知見に反し不要に融合範囲を狭める。
- 融合カーネル生成（#163）はレジスタ内で fan-out を解決する（PoC-9 の
  `ElemwiseFuse` 実装が同じ方式を採ると観測されている）方針を前提として
  よいが、その実装判断自体は #163 のスコープである。

### 2.5 配置

新設モジュール `crates/tensor-core/src/fusion/` を提案する（実装は #162
以降）。TASK-12.1 成果物規定「`tensor-core` または独立モジュール」
（`docs/spec/05-tasks.md:358`）のうち、`device.rs`（TASK-1.9a）・
`backend_ops.rs`（TASK-1.9c）が確立済みの依存逆転構成（trait を
`tensor-core` に置き `backend-*` が実装する。`tensor-core` →
`backend-*` の逆依存を作らない）をそのまま踏襲できる `tensor-core` 内
配置を採る。融合グラフ自体はバックエンド非依存の中間表現であり、
`backend-*` 側は融合カーネルの実装（#163）でのみ関与する。

## 3. 遅延評価境界

### 3.1 方式

既定の eager 実行は変えず、**融合対象区間のみ内部 API でグラフを遅延
構築する「明示的遅延バッファ」方式**を第一案とする。

- 全面 lazy 化（すべてのテンソル演算をグラフ構築のみに留め、明示的な
  実体化まで一切計算しない方式）は不採用とする。理由:
  1. REQ-13（起動コスト対策、`docs/spec/04-requirements.md`）の方針は
     JIT コンパイル・autotune 探索由来の起動コストを避けることにあり、
     全面 lazy 化はグラフ構築・スケジューリングの実行時オーバーヘッドを
     全演算パスへ持ち込む。融合対象区間（elementwise 連鎖）のみへ限定
     すれば、このオーバーヘッドを融合が効く範囲だけに閉じ込められる。
  2. 既存の `BackendOps` 呼び出し規約（各メソッドが `Tensor<f32>` を
     受け取り即座に `Tensor<f32>` を返す、`backend_ops.rs:63` 付近の
     契約）・`autodiff` の値計算契約（`Var` の演算メソッドが `Tape::push`
     と同時に forward 値を計算する、`tape.rs` 冒頭コメント）を全面 lazy
     化は破壊する。既存テスト資産（TASK-1.9d 等）への影響が大きく、
     安全側の選択として不採用とする。

### 3.2 実体化（materialization）ポイントの列挙

融合対象区間の遅延構築は、以下いずれかの条件到達時に実体化（実際の
カーネル呼び出しによる計算実行）へ切り替える。

| # | 条件 | 根拠 |
|---|------|------|
| (a) | reduction ノード（`sum`／`max`）へ到達 | PoC-9 実測で reduction エピローグは自動融合対象外（§1）。融合境界ノードとして扱う |
| (b) | `gemm` ノードへ到達 | PoC-9 実測で matmul をまたぐ融合は分断される（§1）。#203（GEMM epilogue 融合）までは境界として扱う |
| (c) | 遅延ハンドル（§3.4・§3.5）から `Tensor<f32>` への変換（`into_tensor` 等。呼び出し元がホストへ読み出す `as_slice`／`contiguous`／`get` 等を呼ぶには、まず具体的なデータを得る必要がある）| 遅延構築されたグラフの結果を呼び出し元へ返す時点で計算が確定していなければならない。`Tensor`（`tensor.rs:53`）自体は `Arc<Storage<T>>` を必須で保持する既存表現のまま変更しないが（§3.4）、`Storage<T>`（非公開）は §3.5 のとおり `Pending` バリアントを持ちうるため、`as_slice`／`contiguous`／`get` はデータ読み出し前に `Storage::Pending` を検知し実体化する契約とする（§3.5） |
| (d) | 連鎖長上限（4〜6 段）到達 | TASK-12.1 の内容規定（4〜6 段程度）。PoC-9 の代表ワークロード規模（`ew4`／`ew6`）とも整合する上限であり、無制限連鎖によるカーネル生成コスト・レジスタ圧の増大を避ける |
| (e) | 非融合対象パターン検出（transpose 混在等、`NodeMeta.contiguous == false`）| §1・§2.3 の非融合フォールバック方針 |
| (f) | 異なる `FusionSession` に属する `Pending` 同士が二項演算で合流 | §3.5「既存の演算経路からの遅延連鎖構築」ケース 4（PR #357 review 再指摘 P1-1／P1-2 への回答）。越境する側を即時実体化してから葉ノードとして埋め込む契約とし、セッション間の循環参照・backend 越境転送を構造的に排除する |

### 3.3 autodiff との関係

動的テープ式 autodiff（PoC-v2-2、`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`。
実装は `crates/autodiff/src/tape.rs`・`eval.rs`）は forward 値計算の
**下層**で融合が透過に働く構成とする。すなわち:

- `Var` の演算メソッドが呼ぶ forward 値計算（`eval.rs`）が内部で融合グラフ
  を構築・実体化する経路を持つ場合でも、`Tape::push` が記録する
  `Op`（`tape.rs` の `Op` enum、MatMul／Add／Mul／Relu／Exp／Tanh 等）の
  ノード単位の粒度は変更しない。
- 勾配計算（VJP、`grad.rs::vjp`）は `Op` 単位のまま変更しない契約とする。
  融合はあくまで forward 値計算の実行方式（どうカーネルを呼ぶか）の最適化
  であり、テープが記録する計算グラフの構造（VJP が辿るノード単位）には
  影響を与えない。

### 3.4 遅延グラフと `BackendOps`・`Tensor` 契約の接続

（PR #357 review 指摘への対応で追加。§1・§3.1〜3.3 は「透過的」「遅延
構築」という表現のみで、遅延グラフの所有場所・具体的な型・`BackendOps`
（`crates/tensor-core/src/backend_ops.rs:63`）との接続経路を規定して
いなかった。`BackendOps` の各メソッドは具体化済みの `Tensor<f32>` を
受け取り直ちに具体的な `Tensor<f32>` を返す契約であり、`Tensor`
（`crates/tensor-core/src/tensor.rs:53`）は `Arc<Storage<T>>` を必須で
保持する公開型としては変わらない。本節はこの契約と遅延グラフをどう接続
するかを明示する（`Tensor` 自体を経由した実際の伝播経路の具体化は
§3.5 で行う）。）

- **`Tensor` は変更しない**。公開型 `Tensor`（構造体・フィールド型・
  メソッドシグネチャ）は破壊的変更を避ける（公開 API 非破壊はガード
  レール条件、`.claude/rules/security.md`「A08」・
  `docs/spec/04-requirements.md` の REQ-12 受け入れ基準とも整合させる
  安全側の選択）。ただし `Tensor` が保持する非公開の `Storage<T>`
  （`tensor.rs:33`。公開 API 面には現れない crate-private 型）自体は
  §3.5 で「未実体化」を表す内部バリアントを持ちうる拡張を行う。この
  `Storage` 拡張は `Tensor` の公開契約（構造体の形・メソッド
  シグネチャ）を変えないため、本項の「`Tensor` は変更しない」という
  判断とは矛盾しない。
- **`BackendOps` trait 自体の契約も変更しない**。既存の各メソッド
  シグネチャ（具体的な `&Tensor<f32>` を受け取り具体的な
  `Result<Tensor<f32>, BackendError>` を返す）は現状のまま維持する。
  `&dyn BackendOps` を直接呼ぶ既存経路（`ops_for` 経由を含む）は §1 の
  とおり本設計の対象外であり、遅延構築を経由しない。
- **遅延構築は `BackendOps` より上位の新規内部型で行う**。`BackendOps`
  を実装しない、`tensor-core` 内の新規 crate-private 型
  （`crates/tensor-core/src/fusion/` 配下、§2.5）として次を追加する
  （実装は #164 のスコープ。以下は #164 が満たすべき接続契約）:

  ```rust
  /// 遅延構築中の値。§3.5 が規定する既存呼び出し経路（`autodiff::eval`
  /// の `add`/`mul`/`relu`/`exp`/`tanh`）が、`Tensor` の `Storage` 経由で
  /// 暗黙に受け渡す（§3.5。利用者向け公開 API・`Var`/`Tensor` のシグネチャ
  /// はいずれも変更しない）。`Tensor` を置き換えるものではなく、`Tensor`
  /// とは別の内部表現。
  pub(crate) enum FusionValue {
      /// 既に実体化済み（葉ノード・過去の実体化結果）。
      Materialized(Tensor<f32>),
      /// 未実体化。`session` が保持する `FusionGraph`（§2.2）内の
      /// `FusionNodeId` を指す。
      Pending {
          session: FusionSession,
          node: FusionNodeId,
      },
  }

  /// 1 回の融合対象区間（連鎖 1 本）に対応するグラフの所有者。
  /// `Arc<Mutex<FusionGraph>>` に加え、実体化に使う `BackendOps` 実装
  /// 自体（`ops`）をセッション生成時点で所有値として捕獲する（下記
  /// 「ops 解決の所有モデル」参照。単一スレッド内で完結するディスパッチ
  /// 層のローカル値として構築・破棄される点は変更しない）。
  #[derive(Clone)]
  pub(crate) struct FusionSession {
      graph: std::sync::Arc<std::sync::Mutex<FusionGraph>>,
      /// このセッションが実体化に使う `BackendOps` 実装（後述「ops 解決の
      /// 所有モデル」）。`Storage::Pending`（§3.5）に埋め込まれた
      /// `FusionSession` のクローンだけで、追加引数なしに `&self` の
      /// `Tensor::get`／`Tensor::as_slice`（§3.2 (c)）から実体化を発火
      /// できるようにするための必須フィールドである。
      ops: std::sync::Arc<dyn BackendOps + Send + Sync>,
  }

  impl FusionValue {
      /// §3.2 の実体化条件 (a)〜(e) いずれかに到達した時点でディスパッチ層
      /// が呼ぶ。`Pending` を `Tensor<f32>` へ変換し、以降の消費者
      /// （呼び出し元・後続の非融合 `BackendOps` 呼び出し）へ渡す。
      /// `ops` は外部から渡さず `session` が既に保持する値を使う
      /// （§3.2 (c) が要求する「`&self` のみの `get`／`as_slice` からも
      /// 呼べる」制約に合わせたシグネチャ）。
      pub(crate) fn into_tensor(self) -> Result<Tensor<f32>, BackendError> {
          match self {
              FusionValue::Materialized(t) => Ok(t),
              FusionValue::Pending { session, node } => session.materialize(node),
          }
      }
  }
  ```

  - `FusionSession` は**スレッドローカルなグローバル状態にしない**。
    `dispatch.rs` の既存方針（`select_gemm_kernel` は環境変数・グローバル
    設定による経路上書きを持たない副作用なしの純関数設計、
    `crates/tensor-core/src/dispatch.rs:9-17`）と整合させるため、融合
    グラフの所有もディスパッチ層のローカル値（1 連鎖のエントリで生成し
    連鎖終了で破棄する明示的な値）に限定し、暗黙のグローバル・
    スレッドローカルレジストリを設けない。
  - `Arc<Mutex<_>>`（当初案の `Rc<RefCell<_>>` から本改訂で変更。理由は
    下記）を採る。融合対象区間自体は 1 回のディスパッチ呼び出し内で閉じ
    スレッドを跨がない（§3.1 の前提は変わらない）が、`FusionSession` は
    §3.5 のとおり `Storage<T>::Pending`（`Tensor` が `Arc` 経由で複数
    箇所から共有されうる非公開フィールド）に埋め込まれる。`Element`
    trait は既に `Copy + Send + Sync + ...`（`crates/tensor-core/src/element.rs:24`）
    を要求しており、`Tensor<T>`（`Arc<Storage<T>>`）は現状この境界により
    自動的に `Send + Sync` になる。`Storage::Pending` の内部に
    `Rc<RefCell<_>>` を置くと `Tensor<T>` から暗黙に `Send`/`Sync` が
    失われ、`backend-cpu` の `rayon`（`crates/backend-cpu/src/elementwise.rs:62`
    以降・`gemm.rs:30`）が前提とする「複数スレッドから安全に扱える」
    という既存の暗黙契約を破壊する非破壊のはずの `Storage` 拡張が実は
    破壊的変更になってしまう（Cursor Bugbot 想定指摘に先回りする本改訂
    の変更点）。`Arc<Mutex<_>>` を採ることで `Tensor<T>: Send + Sync` を
    崩さない。`Mutex` のロック競合は 1 融合セグメント＝1 ディスパッチ
    呼び出しに閉じるため実務上は無視できる想定（実測は #164 の受け入れ
    条件に含める）。`Storage::Pending` が保持するもう一方の値
    `Arc<dyn BackendOps + Send + Sync>`（下記「ops 解決の所有モデル」）が
    `Send + Sync` であることも同じ理由で必要であり、`Tensor<T>: Send + Sync`
    は `Arc<Mutex<FusionGraph>>` と `Arc<dyn BackendOps + Send + Sync>` の
    **両方**が揃って初めて成立する（Codex P1 指摘への回答は次段落）。

  **ops 解決の所有モデル（P1 指摘「実体化時に使用する `BackendOps` を
  取得できない」への回答）**:

  - `FusionSession::materialize`（内部で `run_fused` を呼ぶ）は、外部から
    毎回渡される `&dyn BackendOps` ではなく、**セッション生成時に捕獲し
    所有した `Arc<dyn BackendOps + Send + Sync>`** を使う。理由: `Tensor::get`／
    `Tensor::as_slice`（`tensor.rs:201`・`tensor.rs:231`、公開 API・
    `&self` のみで追加引数を取れない）から実体化を発火する契約（§3.2 (c)）
    である以上、`FusionSession` 自身が実体化に必要な情報をすべて所有値
    として持っていなければならない。借用 `&dyn BackendOps` はライフタイム
    を `Tensor`／`Storage` へ伝播でき（`Tensor` は `'static` 相当の寿命を
    要求される公開型）ないため不採用とする。
  - **`BackendOps` trait 定義（`backend_ops.rs:82`）自体は変更しない
    （Codex P1 指摘「`Send + Sync` スーパートレイト追加は既存の外部実装を
    破壊する」への回答。本改訂で撤回・訂正する）**: 旧稿は `BackendOps`
    に `Send + Sync` をスーパートレイトとして追加し「既存 3 実装は無条件に
    満たすため非破壊」としていたが、これは誤りだった。`BackendOps` は
    `pub trait` であり、本リポ外の crate が独自に実装する可能性を
    排除できない（trait 定義側の変更が非破壊かどうかは自クレート内の
    実装数ではなく、trait を実装しうる全ての利用者に対して判定する
    必要がある。`.claude/rules/security.md` の A08・本リポ全体の
    「公開 API 非破壊はガードレール条件」方針）。`Send + Sync` を
    スーパートレイトとして追加すると、これを満たさない既存の外部
    `BackendOps` 実装（内部可変状態に `Rc`／`RefCell` 等を使う実装）は
    コンパイル不能になり、破壊的変更（`!` 接頭辞・`BREAKING CHANGE:` 告知
    が必要な変更。`.claude/rules/conventional-commits.md`）に該当する。
  - 本改訂は Codex 指摘が挙げた代案「`FusionSession` が保持する専用
    ラッパー側だけに `Send + Sync` を要求する」を採用する。`BackendOps`
    trait 定義は変更せず、`FusionSession`／`Tape::with_backend`／
    `ops_for_arc` 等、融合機構が実際に `Arc` として所有・スレッド境界を
    越えて保持する箇所でのみ**トレイトオブジェクト側**に
    `Arc<dyn BackendOps + Send + Sync>` という束縛を課す（trait
    定義への `Send + Sync` 追加ではなく、trait オブジェクト型の指定に
    `+ Send + Sync` を付けるだけであり、これは呼び出し側の型注釈に
    留まる非破壊な変更）。既存の `&dyn BackendOps`（`ops_for` 等、融合を
    経由しない既存経路。§3.4 冒頭）は今までどおり `Send`／`Sync` を要求
    しないため、この経路を使う既存の外部実装は本改訂の影響を一切受けない。
    影響を受けるのは「`Tape::with_backend`／`ops_for_arc` を呼んで融合を
    有効化したい」利用者のみであり、その場合に限り自身の `BackendOps`
    実装が `Send + Sync` を満たす必要がある（満たさない実装は融合機構
    への接続 API を呼べずコンパイルエラーになるが、既存の非融合経路は
    影響を受けず既存コードは変更なしにコンパイルが通り続ける）。
    `backend-cpu`／`backend-cuda`／`backend-metal` の 3 実装はいずれも
    内部可変状態を持たないディスパッチ用の空構造体・関数集合であり、
    この境界を無条件に満たす（CI の
    `cargo clippy --workspace --all-targets --all-features` で回帰検知
    できる）。
  - **`Arc<dyn BackendOps + Send + Sync>` をどの時点で捕獲するか（Codex 再指摘
    「バックエンドの供給方法が未定義のまま設計を確定している」への回答。
    本改訂で確定する）**: `FusionSession` を最初に開くのは §3.5 の
    `eval::add`／`mul`／`relu`／`exp`／`tanh`（`crates/autodiff/src/eval.rs`）
    だが、これらの関数は**現状シグネチャに `Device`／`BackendOps` を
    一切持たない**（同ファイル冒頭 doc コメント「`backend-cpu`
    （TASK-1.6・#20 以降）がまだ未完のため、TASK-1.9（バックエンド抽象層
    への接続）で backend 経由の実行に置き換えるまでの暫定実装」）。
    当初案（本節旧稿）は供給元を「#164 が確定する実装詳細」とし、暫定
    既定値として `Device::Cpu` 固定を排除しない、という**未確定のまま**
    にしていた。これは 2 点で成立しない: (i) `Var`／`Tensor` は
    device/backend 情報を一切保持しない設計（§3.4 冒頭「`Tensor` は
    変更しない」）のため、`Device::Cpu` 固定を選ぶと利用者が選択した
    CUDA／Metal backend を伝達する経路が存在しないまま既定化され、
    `docs/public-api-design.md` §4.1「既定デバイス選択ロジック（本節の
    未決事項）は…実装しない（列挙と明示選択のみを提供する。ユーザー
    承認が必要な事項のため自動運転では安全側に倒した）」という本リポ
    全体の確立済み方針（TASK-1.9・#46）に反する。(ii) 一方で供給元を
    「#164 が確定する」とだけ書くと、公開 API を含む供給契約が本文書
    （TASK-12.1a の成果物）に存在しないまま後続イシューへ丸投げされる
    ことになり、これ自体が指摘の対象である。本改訂は Codex 指摘文が
    挙げた代案「`Tape` が backend を所有し生成時に明示選択する」を採用
    し、供給元を `Tape` の公開コンストラクタとして今ここで確定する
    （コード変更自体は #164 のスコープ。本書はコード変更を含まないという
    TASK-12.1a の制約を維持しつつ、公開 API 契約は以下で確定する）:

    ```rust
    impl Tape {
        /// 既存の既定コンストラクタ（`tape.rs:154`）。**backend を持たない
        /// まま**、変更しない。このコンストラクタで作られた `Tape` 上の
        /// `Var` 演算は `eval::add` 等の暫定 CPU 参照実装（`eval.rs`
        /// 冒頭コメント）をそのまま呼び、融合（本文書全体）は一切適用
        /// されない（`FusionSession` を開始する `ops` を持たないため）。
        /// 既存テスト資産（TASK-1.5〜1.8 等）はこのパスを使い続けており、
        /// 挙動もシグネチャも本改訂で変更しない（非破壊）。
        pub fn new() -> Tape { /* 既存実装のまま */ }

        /// 融合を有効化する明示的コンストラクタ（実装は #164。本節が
        /// 確定する供給契約）。`ops` は呼び出し元が明示的に選択した
        /// backend を **所有値** として渡す（`ops_for_arc(&providers, device)`。
        /// 下記「`ops_for` と `Arc` の不整合（Codex 再指摘）への回答」
        /// 参照）。**既定デバイス選択ロジックはここでも導入しない**
        /// （`docs/public-api-design.md` §4.1 の確立済み方針を踏襲。
        /// `Device::Cpu` への暗黙フォールバックは行わない）。`ops` の型は
        /// `BackendOps` trait 定義自体は変更せず、本コンストラクタの
        /// 引数型としてのみ `+ Send + Sync` を課す（上記「ops 解決の
        /// 所有モデル」で訂正済み。trait への破壊的スーパートレイト追加は
        /// 行わない）。この束縛により `Arc<dyn BackendOps + Send + Sync>`
        /// はスレッド境界を越えて `Storage::Pending`（`Tensor<f32>` 経由）へ
        /// 格納しても `Tensor<T>: Send + Sync`（§3.4 冒頭「`Arc<Mutex<_>>`
        /// を採ることで…崩さない」）を壊さない。
        pub fn with_backend(ops: std::sync::Arc<dyn BackendOps + Send + Sync>) -> Tape {
            /* 既存フィールドに `ops: Arc<dyn BackendOps + Send + Sync>` を追加保持
               する以外は `Tape::new()` と同じ初期化（実装は #164）。 */
        }
    }
    ```

    **`ops_for` と `Arc` の不整合（Codex 再指摘。本改訂で解消）への回答**:
    当初案は「既存 `ops_for(&providers, device)`（`backend_ops.rs:88-115`）
    の戻り値をそのまま `Tape::with_backend` へ渡す」としていたが、
    `ops_for` の戻り値は `Result<&'a dyn BackendOps, BackendError>`（借用）
    であり、元の所有者（`providers: &[&'a dyn BackendOps]` の各要素が
    指す実体）の情報なしに `Arc<dyn BackendOps + Send + Sync>` へ変換することはできない
    （`Arc::new(*borrowed)` は `dyn BackendOps` が `Sized` でないため
    コンパイルできず、`unsafe` な再構築も行わない）。既存 `ops_for` は
    §1・§3.4・§3.5 のとおり「`&dyn BackendOps` を直接呼ぶ既存経路（融合
    非対応のまま据え置く経路）」が使う関数であり、その借用ベースの
    シグネチャ自体は変更しない（既存呼び出し元・テストへの非破壊）。

    融合を有効化する供給経路（`Tape::with_backend`）は、これとは別に
    `tensor-core` へ次の関数を **非破壊な追加** として新設する（実装は
    #164。`ops_for` と同じ選択ロジックを `Arc` ベースの入力へ適用する
    だけの姉妹関数であり、既存 `ops_for` の置き換えではない）:

    ```rust
    /// `ops_for`（`backend_ops.rs:88`）の `Arc` 版。呼び出し元が
    /// `Arc<dyn BackendOps + Send + Sync>` として所有する backend 集合（例:
    /// `vec![Arc::new(CpuOps::new()) as Arc<dyn BackendOps + Send + Sync>, ...]`。
    /// 各バックエンド実装は元々値として構築されるため、呼び出し元が
    /// `Arc::new(..)` で包むだけで用意できる）から `device` に一致する
    /// 実装を選び、参照カウントを 1 増やして clone を返す（実体データの
    /// コピーは発生しない。`Arc::clone` は O(1)）。`Tape::with_backend`
    /// が要求する所有値の `Arc<dyn BackendOps + Send + Sync>` はこの関数の戻り値を
    /// 渡す（`ops_for` の借用戻り値は渡せないため使わない）。
    ///
    /// 対応する実装が見つからない場合の挙動（`BackendError::DeviceUnavailable`）
    /// は `ops_for` と同一の意味論を保つ。
    pub fn ops_for_arc(
        ops: &[std::sync::Arc<dyn BackendOps + Send + Sync>],
        device: Device,
    ) -> Result<std::sync::Arc<dyn BackendOps + Send + Sync>, BackendError> {
        ops.iter()
            .find(|candidate| candidate.device() == device)
            .cloned()
            .ok_or_else(|| {
                BackendError::DeviceUnavailable(format!(
                    "no BackendOps registered for device {device:?}"
                ))
            })
    }
    ```

    - `ops_for`（既存）と `ops_for_arc`（新設）はどちらも
      `tensor-core::backend_ops` に共存し、呼び出し元が持つコレクションの
      所有形態（借用スライス／`Arc` 所有）に応じて使い分ける。融合を
      使わない既存経路・テストは引き続き `ops_for` を使い、
      `Tape::with_backend` を呼ぶ経路（融合を有効化したい呼び出し元）は
      `ops_for_arc` を使う。両者の選択ロジック（`device` 一致検索・
      `DeviceUnavailable` の意味論）は同一であり重複実装ではあるが、
      戻り値の所有形態が異なる（借用 vs `Arc` clone）ため 1 実装に
      統合できない（`&'a dyn BackendOps` から `Arc<dyn BackendOps + Send + Sync>` を
      安全に生成する手段がない、上記のとおり）。

    - `Tape` は非公開フィールドとして `ops: Option<Arc<dyn BackendOps + Send + Sync>>`
      を追加保持する（`None` は `Tape::new()` 経由、`Some` は
      `Tape::with_backend()` 経由。フィールド追加は `Tape` の構造体を
      非公開のまま拡張するだけであり、`pub` フィールドを持たない現行の
      `Tape`（`tape.rs:140`）の公開契約を破らない）。
    - `Var::add`／`mul`／`relu`／`exp`／`tanh`（`var.rs:122` 付近）は
      `self.tape.ops`（`pub(crate)` アクセサ）を読み、`eval::add` 等へ
      そのまま渡す。`Var` の**公開**シグネチャは変更しない（§3.4 冒頭の
      制約を維持）。`eval::add`／`mul`／`relu`／`exp`／`tanh`（いずれも
      `pub(crate)`）は #164 で `ops: Option<Arc<dyn BackendOps + Send + Sync>>` 引数を
      追加する。これが TASK-1.9 が本来担う「backend 経由の実行への置き
      換え」の実体でもあり、フュージョン統合はこの置き換えと**同時に**
      行う（独立した前提ではなく #164 の作業内容そのものとして扱う）。
    - `ops` が `None`（`Tape::new()` 経由）の場合、`eval::add` 等は
      `FusionSession` を開始せず、`Pending` を生成しない（常に
      `Storage::Materialized`。§3.5「スコープの明示」と同様の非融合
      経路）。これにより「backend 未指定なら融合なしの現行動作のまま」
      という後方互換性が保たれ、`Device::Cpu` への暗黙フォールバックを
      一切行わない。
    - これにより「セッション生成時に `Arc<dyn BackendOps + Send + Sync>` を捕獲する」
      という本節の前提が満たされる: `Tape::with_backend(ops)` で生成した
      `Tape` 上で `eval::add` が `Pending` を新規に開始する際、
      `self.tape.ops`（`Some` であることが呼び出し前提）をそのまま
      `FusionSession::new(ops)` へ渡し、以降の `Pending` 複製（延伸）は
      すべて同じ `Arc` をクローンして引き継ぐ（§3.5 で具体化）。
- **実際のカーネル呼び出し経路（`FusionValue::into_tensor` の内部実装が
  呼ぶ `FusionSession::materialize`）は `BackendOps` の非破壊拡張
  （デフォルトメソッド）で提供する**。`backend_ops.rs` 冒頭コメントが
  既に採用している拡張パターン（「`BackendOps` の非破壊拡張（デフォルト
  メソッド追加等）」`backend_ops.rs:27` 付近）をそのまま踏襲する。

  Cursor Bugbot 指摘（本 PR review）への修正: 当初案は `run_fused` の
  引数型 `FusionPlan` を未定義のまま `pub trait BackendOps`（外部クレート
  `backend-cpu`／`backend-cuda`／`backend-metal` が実装）のメソッド
  シグネチャに置いていた。`FusionOp`／`FusionNode`／`FusionGraph`（§2）は
  `pub(crate)`（`tensor-core` 内限定）のままであり、`pub` trait のメソッド
  シグネチャに `pub(crate)` 型を直接使うと privacy 違反（外部クレートが
  型を命名できない）になる。よって `FusionPlan` は `tensor-core` 内で
  `pub`（フィールドは非公開）の不透明ハンドルとして新設し、内部の
  `pub(crate)` グラフ表現をラップする:

  Codex 再指摘（本 PR review）への追加修正: 当初案はハンドル自体の
  privacy 解消（`pub struct FusionPlan` 新設）のみを行い、外部 backend
  が `FusionPlan` の中身（演算列・入力・メタデータ）を読み取るアクセサを
  「#163 で追加する」と先送りしていた。しかし `FusionOp`／`FusionNode`
  （§2）は `pub(crate)` のまま変更しない（§2.5 の設計判断）ため、
  「アクセサはいずれ追加する」とだけ書いても `impl FusionPlan` の
  `pub` メソッドの戻り値・引数に `pub(crate)` 型を直接使うことはできず
  （同じ privacy 制約の再発）、結局この節だけでは #163 が実装可能な
  契約になっていない。本改訂はアクセサの型を DTO（data transfer object）
  として今ここで確定する:

  ```rust
  /// `FusionPlan` 内のノード位置を指す公開インデックス。内部の
  /// `FusionNodeId`（`pub(crate)`、§2.2）はそのまま公開できないため、
  /// `FusionPlan` 内でのみ意味を持つ 0 起点の連番（発生順）として
  /// 別の型を用意する。
  pub type FusedNodeIndex = usize;

  /// `FusionPlan::ops`（下記）が列挙する 1 ノード分の演算内容。内部
  /// `pub(crate)` の `FusionOp`（§2.1）と 1:1 対応するが、融合境界
  /// ノード（`Gemm`／`Sum`／`Max`）は §3.2 (a)(b) のとおり `FusionPlan`
  /// 内に現れない（実体化境界のため、融合対象区間そのものには含まれ
  /// ない）ので列挙しない。フィールドは `FusedNodeIndex`（plain
  /// `usize`）のみで構成し、`pub(crate)` 型を一切参照しない。
  #[derive(Debug, Clone, Copy)]
  pub enum FusedOpKind {
      /// 葉ノード（このプランへの外部入力）。`leaf_index` は
      /// `run_fused` の `leaves: &[&Tensor<f32>]`（下記）の添字と対応する。
      Input { leaf_index: FusedNodeIndex },
      Add { lhs: FusedNodeIndex, rhs: FusedNodeIndex },
      Mul { lhs: FusedNodeIndex, rhs: FusedNodeIndex },
      Relu { input: FusedNodeIndex },
      Exp { input: FusedNodeIndex },
      Tanh { input: FusedNodeIndex },
  }

  /// `run_fused`（`BackendOps` の非破壊拡張、下記）へ渡す公開の不透明
  /// ハンドル。`BackendOps` は `pub trait`（`backend-cpu`／
  /// `backend-cuda`／`backend-metal` が実装）のため、その既定メソッドの
  /// 引数型は `pub` でなければならない（privacy 制約）。内部の融合 IR
  /// （`FusionGraph`／`FusionNode`／`FusionOp`。§2、`pub(crate)` のまま
  /// 変更しない）はフィールドとして非公開のまま包み、`tensor-core` 外
  /// からは構築・分解できない。読み取りは下記 `impl FusionPlan` の
  /// `pub` メソッドを通じてのみ行う（フィールドを直接公開しない理由:
  /// 内部 IR の表現変更が `FusionPlan` の公開契約に波及しないようにする
  /// ため）。
  pub struct FusionPlan {
      // `Rc` ではなく `Arc` とする（`FusionSession` の `Arc<Mutex<_>>` 化
      // と同じ Send/Sync 保持の理由。上記「ops 解決の所有モデル」直前の
      // 段落参照）。
      graph: std::sync::Arc<FusionGraph>,
  }

  // 以下はシグネチャのみを確定するスケッチであり、本体の実装は #163
  // が担う（§2.1／§2.2／§3.4 冒頭の `FusionOp`／`FusionSession` の
  // シグネチャスケッチと同じ体裁。「アクセサをいつか追加する」という
  // 先送りではなく、外部 backend が呼べる関数シグネチャそのものを
  // 本文書で確定する）。
  impl FusionPlan {
      /// 発生順（トポロジカル順。§2.2「ノードは発生順に `Vec` へ追記」）
      /// で `FusedOpKind` を列挙する。#163 はこの順で辿ることで、各
      /// ノードの入力（`lhs`／`rhs`／`input` が指す `FusedNodeIndex`）が
      /// 走査済みであることを保証できる（トポロジカル順の定義そのもの）。
      /// 実装は `self.graph`（`pub(crate)` の `FusionNode` 列）を発生順
      /// に走査し `FusionOp`（§2.1）を対応する `FusedOpKind` へ変換する
      /// （境界ノード Gemm／Sum／Max は §3.2 (a)(b) によりプラン内に
      /// 現れないため列挙対象外）。
      pub fn ops(&self) -> impl Iterator<Item = FusedOpKind> + '_;

      /// このプランが表す出力テンソルの shape（`NodeMeta.shape`。§2.3）。
      pub fn output_shape(&self) -> &[usize];

      /// このプランの dtype（`NodeMeta.dtype`。§2.3、`DType` は
      /// `dispatch.rs:31` で `pub` 定義済み）。§2.1 のとおり現状は
      /// 常に `DType::F32`。
      pub fn dtype(&self) -> DType;

      /// このプランが要求する葉ノード（外部入力）の個数。`run_fused` の
      /// `leaves: &[&Tensor<f32>]`（下記）の長さはこの値と一致する契約
      /// とし、不一致は #163 が shape 検証と同様の扱いで拒否する
      /// （§4「グラフ構築 API はテンソル shape／stride の検証を先行
      /// させる」と同型の契約）。
      pub fn leaf_count(&self) -> usize;

      /// 指定ノードの被参照数（§2.4 の `NodeMeta.use_count` を公開する。
      /// #163 のレジスタ内 fan-out 解決が読む）。**この値はプラン内
      /// （融合セグメント内）からの被参照数のみを数える**。境界ノード
      /// （Gemm／Sum／Max。プラン外）から参照される場合、その参照は
      /// ここに含まれない。#163 はこの値とプラン全体の出力有無を突き
      /// 合わせ、プラン内で使い切られない中間値（境界ノードへ流出する
      /// 値）はレジスタ内に留めず実体化して渡す必要があると判定する。
      pub fn use_count(&self, node: FusedNodeIndex) -> usize;
  }

  pub trait BackendOps {
      // 既存メソッド（gemm／add／mul／relu／exp／tanh／sum／max）は不変。

      /// 融合グラフ（#162 が検出した連鎖・#163 が生成するカーネル）を
      /// 1 回のカーネル呼び出しで実行する。デフォルト実装は
      /// `BackendError::Unsupported` を返す fail-safe（backend_ops.rs の
      /// 既存 elementwise・reduction 未実装カーネルと同型の設計）。
      /// 各バックエンドが融合カーネル生成（#163）を実装した時点で
      /// override する。
      fn run_fused(
          &self,
          plan: &FusionPlan,
          leaves: &[&Tensor<f32>],
      ) -> Result<Tensor<f32>, BackendError> {
          Err(BackendError::Unsupported("run_fused: default fail-safe".into()))
      }
  }
  ```

  - `FusionSession::materialize` は自身が所有する `self.ops`
    （`Arc<dyn BackendOps + Send + Sync>`。上記「ops 解決の所有モデル」）から
    `FusionGraph` を `FusionPlan` へ構築したうえで `self.ops.run_fused(&plan, leaves)`
    を試し、`BackendError::Unsupported` が返った場合は §4 の fail-safe
    方針に従い、グラフのノードを発生順に辿って既存の
    `add`／`mul`／`relu`／`exp`／`tanh` 呼び出しへ 1 段ずつ逐次
    フォールバックする（融合の有無に関わらず最終結果は同一の数値一致
    複合判定を満たす。§4）。追加の `ops` 引数を外部から要求しないため、
    §3.2 (c) の「`&self` のみの `get`／`as_slice` から発火できる」制約を
    満たす。
  - `run_fused` はデフォルト実装を持つため、既存の 3 バックエンド
    実装（CPU／CUDA／Metal）は本節追加時点で override 不要のまま
    コンパイルが通る（trait の破壊的変更にならない）。`BackendOps`
    trait 定義自体には手を加えないため（上記「ops 解決の所有モデル」で
    訂正済み。`Send + Sync` は `Tape::with_backend`／`ops_for_arc` の
    引数・戻り値型としてのみ課す）、`run_fused` 追加とあわせて既存の
    外部 `BackendOps` 実装への影響はない。
- **まとめ（2 件の P1 指摘への回答。本改訂で確定）**: 「遅延値を保持
  できる `Tensor` 表現への変更」は採らない（`Tensor` 不変）。「別の内部
  lazy handle」（`FusionValue`／`FusionSession`）を採用する。「連鎖全体を
  受け取る明示的な内部 API」として `BackendOps::run_fused`（非破壊拡張の
  デフォルトメソッド）を追加する。グラフの所有は `FusionSession` が
  `Arc<Mutex<FusionGraph>>` として明示的に保持し（`Tensor<T>: Send + Sync`
  を壊さないよう `Rc<RefCell<_>>` から本改訂で変更）、実体化に使う
  `BackendOps` 実装自体も `FusionSession` が `Arc<dyn BackendOps + Send + Sync>` として
  所有値で保持する（外部から都度渡す借用ではない）。この `Arc<dyn BackendOps + Send + Sync>`
  はセッションを開始する `eval::add`／`mul`／`relu`／`exp`／`tanh`
  （`crates/autodiff/src/eval.rs`。現状は device/ops を持たない暫定 CPU
  参照実装）から供給されるが、その**供給元自体**は本改訂で
  `Tape::with_backend(ops: Arc<dyn BackendOps + Send + Sync>)`（新規の公開コンストラクタ。
  上記「`Arc<dyn BackendOps + Send + Sync>` をどの時点で捕獲するか」）として確定した。
  `Tape::new()`（backend なし）は融合を一切適用しない後方互換パスの
  ままとし、`Device::Cpu` への暗黙フォールバックは行わない
  （`docs/public-api-design.md` §4.1 の既定デバイス選択ロジック不採用
  方針を踏襲）。#164 は `eval::add` 等への `ops: Option<Arc<dyn BackendOps + Send + Sync>>`
  引数追加を TASK-1.9 が求める「backend 経由の実行への置き換え」と同時に
  行う。伝播経路の具体化は §3.5 で規定する。外部 backend
  （`backend-cpu`／`backend-cuda`／`backend-metal`）が `run_fused` 内で
  融合グラフの演算内容を読み取る手段も本改訂で確定した: `FusionPlan` は
  `pub`（フィールド非公開）の不透明ハンドルとし、`impl FusionPlan` の
  `pub fn ops() -> impl Iterator<Item = FusedOpKind>`／`output_shape`／
  `dtype`／`leaf_count`／`use_count`（上記コード例）という公開 DTO
  アクセサ経由でのみ読み取らせる。内部の `pub(crate)` `FusionOp`／
  `FusionNode`／`FusionGraph`（§2）は非公開のまま変更しない。既存
  `BackendOps` 呼び出し規約・`Tensor` 表現とは非破壊に接続される。

### 3.5 既存の演算経路からの遅延連鎖構築（PR #357 review 追加指摘への対応）

（`docs/fusion-graph-design.md:245` 付近への指摘で追加。§3.4 が定義した
`FusionValue`／`FusionSession` を「誰が・どの既存呼び出し経路で・複数回
の `add`／`mul` をまたいで」実際に伝播させるかが未規定だった。利用者が
書く通常の `Tensor` 演算チェーンは各段が具体化済み `Tensor` を要求する
契約のため、**この経路では連鎖を構築できない**（§1・§3.4 のとおり
`&dyn BackendOps` を直接呼ぶ経路は本設計の対象外のまま）。本節は対象
経路を具体的なコード呼び出し関係まで特定し、伝播の実装点を規定する。)

- **対象経路の特定**: 融合対応にする経路は「`Var` の演算メソッド
  （`add`／`mul`／`relu`／`exp`／`tanh`。`crates/autodiff/src/var.rs`）
  が forward 値計算のために呼ぶ `eval::add`／`eval::mul`／`eval::relu`／
  `eval::exp`／`eval::tanh`（`crates/autodiff/src/eval.rs`）」に限定する。
  この経路が「複数回の `add`／`mul` 呼び出しをまたいで」実際に連鎖する
  唯一の既存経路である理由:
  - `Var::add`（`var.rs:122` 付近）・`Var::mul`（`var.rs:133` 付近）は、
    それぞれ独立した呼び出しのたびに `eval::add`／`eval::mul` を呼び、
    戻り値の `Tensor<f32>` を `Tape::push` でテープノードへ記録する
    （`var.rs` の①〜④の処理順、ファイル冒頭コメント）。呼び出し元
    コード（例: `a.add(&b)?.mul(&c)?`）は Rust の通常のメソッドチェーン
    であり、各段が返す `Var` は次段の入力としてそのまま渡される。
  - 一方 `ops_for(device).add()`（`BackendOps` を直接呼ぶ経路。
    `backend_ops.rs:98`）は §1 のとおり対象外のまま据え置く。呼び出し元
    が `Tensor` を直接操作するこの経路には融合を適用しない。
- **伝播の実装点（`Storage` 経由。P1 指摘「既存の `&self` 読み出し API
  では `Pending` を指定どおり実体化できない」への回答。本改訂で具体化）**:
  `Var::add`/`mul`/… が呼ぶたびに `eval::add` 等へ渡す入力・戻り値は
  依然として型としては `Tensor<f32>` のままである（`Var`／`Tape`／
  `Tensor` いずれのシグネチャも変更しない）。複数回の独立した呼び出しを
  またいで `FusionValue::Pending` を実際に持ち越すには、その
  `Tensor<f32>` が内部的に「まだ実体化していない」ことを表現できる必要
  がある。これを `Tensor`（`tensor.rs:52`）自体ではなく、`Tensor` が保持
  する非公開の `Storage<T>`（`tensor.rs:33`。既に crate-private であり
  公開 API 面には現れない）側に持たせる。ただし当初案の
  `enum Storage<T> { Materialized(Vec<T>), Pending(FusionValue) }` は
  P1 指摘のとおり 2 つの理由で成立しない: (i) `Tensor` は `Arc<Storage<T>>`
  を共有し、`get`／`as_slice` は `&self` の共有参照しか持たないため、
  バリアントそのものを `Pending → Materialized` へ置換する操作ができない。
  (ii) `as_slice` は `Storage` 内部の `Vec<T>` を借用する `&[T]` を返す
  契約であり、置換を許す `RefCell` 等の内部可変化と併用しても
  「借用元を書き換えながら借用を返す」ことはできない。本改訂は以下の
  設計へ変更する（実装は #164 のスコープ。以下は #164 が満たすべき契約）。

  ```rust
  /// `Storage<T>` の `Pending` が必要とする実体化操作の型消去境界
  /// （Codex レビュー再指摘への対応。下記「型消去境界」参照）。
  /// `T` は `Storage<T>` の `T` とそのまま一致させ、`Vec<f32>` を
  /// `Vec<T>` へ横流しする代入は書かない（型システムに証明させる）。
  pub(crate) trait Materializer<T: Element>: Send + Sync {
      /// §3.2 の実体化条件到達時に呼ばれる。成功時は稠密 `Vec<T>` を、
      /// 失敗時は型付きの `BackendError` を返す（Codex 再指摘「遅延
      /// 実体化時の `BackendError` が消失する」への回答。下記「失敗
      /// 経路の扱い」）。`run_fused` は `Unsupported` 以外にも GPU
      /// 実行・コンパイル・デバイス障害等で失敗しうるため、`Option`
      /// による握り潰しは行わない。
      fn materialize(&self) -> Result<Vec<T>, BackendError>;
  }

  /// `f32` 専用の `Materializer` 実装。`FusionSession`／`FusionNodeId`
  /// （§3.4・§2.2）を保持し、`materialize` 内で
  /// `session.materialize(node)`（§3.4。`Result<Tensor<f32>, BackendError>`
  /// を返す）を呼んで稠密化する。`Materializer<f32>` のみを実装し、
  /// 他の `T`（`i32` 等）向けの実装は存在しない（§2.1 の f32 固定
  /// スコープをこの 1 impl に閉じ込める）。
  pub(crate) struct FusionMaterializer {
      session: FusionSession,
      node: FusionNodeId,
  }

  impl Materializer<f32> for FusionMaterializer {
      fn materialize(&self) -> Result<Vec<f32>, BackendError> {
          // `.ok()` で握り潰さない： `BackendError` をそのまま呼び出し元
          // （`StorageData::Pending::cache`。下記）へ伝播させる。
          self.session
              .materialize(self.node)
              .map(|t| /* contiguous().as_slice() 相当の稠密化。下記参照 */ Vec::new())
      }
  }

  /// バリアントの「置換」ではなく `OnceLock` による一度限りの初期化で
  /// 実体化を表現する。`Pending` 自体は消えず、内部の `OnceLock` が
  /// 空 → 実体化済みへ 1 度だけ遷移する（P1 (ii) への回答: `&self` の
  /// ままキャッシュを書き込める。`RefCell` と異なり、`OnceLock::get`
  /// は初期化後にそのまま `&Vec<T>` を返せるため「借用元を書き換え
  /// ながら借用を返す」問題が生じない）。
  enum StorageData<T: Element> {
      Materialized(Vec<T>),
      Pending {
          /// 実体化結果のキャッシュ。**`Option<Vec<T>>` ではなく
          /// `Result<Vec<T>, BackendError>` を保持する**（Codex 再指摘
          /// への回答。旧稿は失敗時に `None` へ丸め、`BackendError` を
          /// 恒久的に失っていた。`OnceLock::get_or_init`（安定版 API。
          /// `get_or_try_init` は未安定化）はクロージャの戻り値を型
          /// そのまま格納できるため、戻り値の型を `Result<..>` にする
          /// だけで「一度だけ初期化」の性質（`OnceLock` を選んだ理由。
          /// 上記コメント）を保ったまま失敗理由を保持できる
          /// （`get_or_try_init` の安定化を待つ必要はない）。
          /// `BackendError` は再読み出しのたびに `clone` して返せるよう
          /// `#[derive(Clone)]` を追加する（`device.rs:183` 付近。現状
          /// `#[derive(Debug)]` のみ）。`BackendError::ShapeMismatch` は
          /// `ShapeError`（`error.rs:18` 付近）を保持するため、
          /// `ShapeError` 側にも同様に `#[derive(Clone)]` を追加する
          /// 必要がある（現状 `#[derive(Debug)]` のみ。`error.rs:19`
          /// 以降の各 variant は `usize`／`String` フィールドのみで
          /// 構成され `Clone` 可能）。`#[non_exhaustive]` はいずれも
          /// derive の妨げにならない（未知 variant を追加する権利を
          /// 保持するだけで、既知 variant の trait 実装は制限されない）。
          /// 両者とも既存の `Debug` 実装を維持したまま `Clone` を追加
          /// するのみであり、公開 API 非破壊のまま拡張できる。
          /// 読み出し側は 2 系統に分岐する（下記「実体化の発火点」）:
          /// 公開 `get`／`as_slice`（`tensor.rs:201`・`tensor.rs:231`。
          /// 既存の `Option` 返却契約を変更しない）は `Result` を
          /// `.ok()` で `Option` へ変換して返す。境界ノード
          /// （`gemm`／`sum`／`max`。§3.2 (a)(b)）を実装する
          /// `eval::matmul`／`eval::sum`／`eval::max` は、`Option` を
          /// 経由しない新設の `pub(crate)` フォールブルアクセサ経由で
          /// `BackendError` をそのまま受け取り、自身の戻り値
          /// （`Result<Tensor<f32>, BackendError>`。下記）として
          /// 呼び出し元へ伝播する。本番経路で panic しない方針
          /// （`.claude/rules/coding-rust.md`）と整合。
          cache: std::sync::OnceLock<Result<Vec<T>, BackendError>>,
          /// 型消去された実体化操作（下記「型消去境界」参照）。
          /// `Box<dyn Materializer<T>>` は `T` ごとに別の trait
          /// （`Materializer<f32>`・`Materializer<i32>`・…）であり、
          /// `materialize(&self) -> Result<Vec<T>, BackendError>` は
          /// その `T` をそのまま返す。`Vec<f32>` を `Vec<T>` として代入する箇所は
          /// どこにも存在しない。
          source: Box<dyn Materializer<T>>,
          /// このノードが属する `FusionSession`（§3.4）と、その中での
          /// `FusionNodeId`（§2.2）。**PR #357 review 再指摘「異なる
          /// `FusionSession` に属する `Pending` 同士の二項演算合流が
          /// 未定義」への回答として本改訂で追加する**フィールドである
          /// （下記「異なるセッション間の合流」参照）。`source` の
          /// `Box<dyn Materializer<T>>` への型消去（上記「型消去境界」）
          /// は実体化**結果**の型（`Vec<T>` は `T` ごとに異なる）を
          /// 隠すためのものであり、`session`／`node` は `T` に依存しない
          /// 型（`FusionSession` は非総称型、`FusionNodeId` は `usize`
          /// 相当。§2.2）であるため、これらを `Storage<T>` に直接
          /// 持たせても型消去境界が要求する性質（`Vec<f32>` を `Vec<T>`
          /// として代入する箇所を作らない）を壊さない。したがって
          /// 51d0194（`session`／`node` を `FusionMaterializer` の内部へ
          /// 押し込めた回）からの後退ではなく、型消去とは独立の理由
          /// （合流判定に `eval::add` 自身がセッション識別子へ到達する
          /// 必要がある）による再追加である。
          session: FusionSession,
          node: FusionNodeId,
      },
  }

  struct Storage<T: Element> {
      data: StorageData<T>,
  }
  ```

  - **型消去境界（Codex レビュー再指摘「f32 の実体化結果を総称型
    キャッシュへ格納できない」への回答）**: 当初案（本節旧稿）は
    `Pending { session: FusionSession, node: FusionNodeId }` を
    `Storage<T>` に直接埋め込み、「`Pending` は実行時には `T = f32` の
    場合にしか構築されない」という**実行時契約のみ**で `Vec<f32>` を
    `Vec<T>` として扱おうとしていた。これは Rust の型システムでは証明
    できない（`impl<T: Element> Tensor<T>` は総称のままコンパイルされ、
    `T` が具体的に何であるかをコンパイル時に知らない）ため、指摘のとおり
    記載どおりには実装できない。本改訂は `Pending` の実体化操作を
    `Box<dyn Materializer<T>>`（トレイトオブジェクトによる型消去）へ
    切り出す。`Materializer<T>::materialize(&self) -> Result<Vec<T>, BackendError>`
    は呼び出し側の `T` をそのまま返す型シグネチャであり、`f32` 専用の
    `FusionMaterializer` は `Materializer<f32>` だけを実装する。
    `Storage<i32>::Pending` は型としては構築可能（`Box<dyn Materializer<i32>>`
    という型自体は存在する）だが、`Materializer<i32>` を実装する型が
    どこにも定義されないため、**実際に構築するコードを書けるのは
    `eval::add`／`mul`／`relu`／`exp`／`tanh`（いずれも `Tensor<f32>`
    のみを引数に取る。`eval.rs` に `i32` 版の対応する演算は存在しない）
    の `FusionMaterializer` 経由に限られる**。「T=f32 でしか使わない」
    という契約が実行時の申し合わせではなく、実装可能な型が 1 つしか
    存在しないという型システム上の事実として保証される。
    `Tensor<i32>` を扱う経路（`cross_entropy_loss` の `targets` 等、
    `dense_vec_i32` 経由の読み出し専用パス）は `Pending` を構築する
    呼び出し元を持たないため、実行時には常に `Materialized` のままで
    ある。
  - `eval::add(&a, &b, ops: Option<Arc<dyn BackendOps + Send + Sync>>)` は、`ops` が
    `None`（`Tape::new()` 経由。§3.4「`Arc<dyn BackendOps + Send + Sync>` をどの時点で
    捕獲するか」）であれば `Pending` を一切生成せず既存の非融合計算へ
    そのままフォールバックする。`ops` が `Some`（`Tape::with_backend()`
    経由）かつ §3.2 の実体化条件 (a)〜(e) のいずれにも未到達であれば、
    `a`／`b` それぞれの `Storage` が `Materialized` か `Pending` かで
    次の 4 通りに分岐する（**両者が異なる `FusionSession` に属する
    `Pending` である場合が本改訂まで未定義だった。PR #357 review
    再指摘「`(a + b) * (c + d)` のように両入力が別々の `FusionSession`
    に属する場合が仕様に存在しない」への回答**）:
    1. **両者とも `Materialized`**: §3.4 のとおり自身が受け取った `ops`
       から新規に `FusionSession` を開始し、`a`／`b` を葉ノードとして
       登録したうえで `FusionOp::Add` ノードを追加する（既存記載どおり。
       変更なし）。
    2. **一方のみ `Pending`**: `Pending` 側の `FusionSession` へ延伸する。
       `Materialized` 側は同じセッション内に葉ノード（既存の
       `Tensor<f32>` 値をそのまま保持するノード。§2.2）として登録して
       から `FusionOp::Add` ノードを追加する（既存記載「延伸」の意味を
       明確化。以前の記載は葉ノード登録に触れていなかった）。
    3. **両者とも `Pending` で同一セッション**（`Arc::ptr_eq` を
       `FusionSession::graph`〈`Arc<Mutex<FusionGraph>>`。§3.4〉同士に
       適用して判定。`ops` フィールドでの比較は使わない: 2 つの独立した
       セッションが同じ `Arc<dyn BackendOps + Send + Sync>` を共有しうるため、`ops` の
       一致は同一セッションであることを含意しない）: そのセッションへ
       延伸し、両者の既存 `FusionNodeId` を入力とする `FusionOp::Add`
       ノードを追加する（既存記載どおり）。
    4. **両者とも `Pending` で異なるセッション**: **グラフの合流
       （マージ）は行わない**という結論は維持するが、「`b` の `Pending`
       をそのまま `a` 側セッションの葉ノードとして埋め込む」という
       当初案（本節旧稿）は撤回する（Codex レビュー再指摘 P1-1「セッション
       間参照によって循環グラフとデッドロックを構築できる」・P1-2
       「異なる backend のテンソルを転送なしで同一カーネルへ渡している」
       への回答。本項）。代わりに**境界を跨ぐ側（`b`）をこの場で即時
       実体化してから埋め込む**方式へ変更する。「即時実体化」は新しい
       発火経路を追加するのではなく、`b` 自身の `Storage::Pending` が
       既に持つ発火点をこの場で前倒しに呼ぶだけである:
       - `b` の `Storage::Pending { cache, source, .. }`（§3.5「実行時
         `T` 制約の解消」参照。`source` は `Box<dyn Materializer<f32>>`
         ＝ `FusionMaterializer { session: FusionSession（session B）,
         node }`）の `cache.get_or_init(|| source.materialize())` を
         **同期的に**呼ぶ。これは下記「実体化の発火点」が §3.2 条件
         (a)(b) や VJP から呼ぶのと**同一の発火点**であり、新規の API を
         追加しない。`source.materialize()`（`FusionMaterializer` の
         `Materializer<f32>` 実装）は内部で `session.materialize(node)`
         （`b` 自身の `Arc<Mutex<FusionGraph>>` を取得・解放し、`b` 自身
         が捕獲した `Arc<dyn BackendOps + Send + Sync>` で `run_fused`
         を実行する。§3.4「ops 解決の所有モデル」の契約はそのまま
         維持する）を呼んで `Result<Tensor<f32>, BackendError>` を得た
         のち、稠密化した `Result<Vec<f32>, BackendError>` を返す
         （`FusionMaterializer::materialize` の定義そのもの。上記
         「型消去境界」参照）。失敗時は `BackendError` をそのまま
         呼び出し元（`eval::add` 等）へ伝播する（下記「実体化の発火点」
         と同じ型付きエラー伝播規約）。
       - 成功して得た `Vec<f32>` から、`b` とは別の**新規** `Tensor<f32>`
         （`Storage::Materialized` のみを持つ、`Pending` を経由しない
         通常の構築経路。`b` の shape をそのまま引き継ぐ）を組み立て、
         これを `a` 側セッションへ葉ノード登録してから `FusionOp::Add`
         ノードを追加する（この時点でケース 2「一方のみ `Pending`」の
         処理へ収束する。`a` 側は `Pending` のまま延伸を継続できる）。
       - **`b` 自身（呼び出し元が保持し続けるかもしれない元の
         `Var`／`Tensor<f32>` 値）の `Storage` 変種は変更しない**: `b`
         はケース分岐上は引き続き `Pending`（§3.5 の `Storage::Pending`
         は「実体化が発火したか」ではなく「`Pending` として構築された
         か」を表すタグであり、`cache` が埋まっていても変種は変わらない）
         のままである。一方 `cache.get_or_init` を `b` 自身の `Storage`
         に対して直接呼んだため、`b` の `cache` は今回の結果でそのまま
         埋まっている（`OnceLock::get_or_init` は二重発火しない）。
         したがって `b` を後で読み出す・再度別の演算へ渡しても再計算は
         起きない。この「`Storage` 変種は `Pending` のまま、`cache` だけ
         が先に埋まる」という状態は、下記「循環参照が発生しないことの
         根拠」の `d = b + c` の議論で `b` が再びケース 4（またはケース
         3）の分岐対象になる前提としてそのまま使う。
       - **循環参照が発生しないことの根拠（P1-1 への回答。旧稿の根拠は
         成立しないことを認める）**: 旧稿は「ノードは自身より前に構築
         済みのノードしか参照できない DAG 構築規則により、セッション間
         参照は常に『後から構築されたセッション → 先に構築された
         セッション』の一方向のみになる」と主張していたが、これは
         セッションの**構築順**にのみ着目しており、後続の演算がどちら
         を延伸元に選ぶかには着目していなかったため誤りだった。指摘の
         具体例で確認する: `c = a + b`（`a` が session A・`b` が
         session B）はケース 4 の規則（先に評価される側 = `a`）に従い
         session A を延伸し、`b`（session B）を葉として埋め込む
         （A → B の参照）。続けて `d = b + c` を構築すると、先に評価
         される側は `b`（session B）であり、規則に従えば session B を
         延伸し `c`（session A に属する）を葉として埋め込むことになる
         （B → A の参照）。これは A・B 間に**双方向**の参照を作り、
         旧稿が前提としていた一方向性そのものを破る。本改訂後は
         `Pending` が他セッションの `Pending` を参照する経路自体を
         作らない（越境時は必ずその場で `Materialized` へ確定させてから
         埋め込む）ため、セッション間の参照は常に「`Mutex` を解放済みの
         確定値を指すだけ」になる。`c = a + b` の時点で `b` は即座に
         実体化されて session A の葉として埋め込まれ、以後 session A
         からは session B の `Mutex<FusionGraph>` を参照する経路が
         一切残らない。したがって `d = b + c` を構築しても session B
         が session A を参照する余地はなく（`c` 自体は session A の
         `Pending` のままなので、`d = b + c` は `c` 側〈session A〉を
         即時実体化してから `b`〈session B〉の葉として埋め込む、ケース 4
         の対称パスを通る）、`Mutex` 同士が互いを待ち合う構造は構築時
         点で作りようがない（ロック取得順序の議論に頼らない、より強い
         保証）。
       - **backend 越境転送が安全であることの根拠（P1-2 への回答）**:
         `b` の実体化（`source.materialize()` が内部で呼ぶ
         `session.materialize(node)`）は `b` 自身の `ops`（backend）で
         行われるが、`session.materialize(node)` の戻り値は常に host
         常駐の `Tensor<f32>` であり（§3.4「`session.materialize(node)`
         〈`Result<Tensor<f32>, BackendError>` を返す〉」参照）、それを
         稠密化した `Vec<f32>`（`FusionMaterializer::materialize` の
         戻り値。上記）から新規に組み立てる `Tensor<f32>` も同型である。
         これは実装の規律（「各バックエンドが host へ確定させる契約を
         守る」という申し合わせ）ではなく、**型として保証される**:
         `tensor-core::Tensor<T>`
         （`crates/tensor-core/src/tensor.rs:52`）は `storage:
         Arc<Storage<T>>` を保持し、`Storage<T>`（同ファイル
         `tensor.rs:33`）は `data: Vec<T>` フィールドのみを持つ ——
         デバイスメモリのハンドル・`device: Device` タグを一切持たない
         構造体である。デバイス常駐データを表す型は別に存在する
         （`DeviceBuffer<T>`。`crates/tensor-core/src/buffer.rs:130`
         付近。`device: Device` フィールドを持つ）が、`BackendOps::gemm`
         （§2.1）を含む本文書の融合対象演算はいずれも `Tensor<f32>` を
         引数・戻り値とするシグネチャであり `DeviceBuffer` を経由しない。
         したがって `b` 側バックエンドの `gemm` 実装〈`backend-cpu`／
         `backend-cuda`／`backend-metal`〉がどのデバイスで計算しようとも、
         `materialize` が返せる型はコンパイル時点で「デバイス残留情報を
         持ちえない」`Tensor<f32>` に固定される（§3.4）。`a` 側の
         `self.ops.run_fused(&plan, leaves)`（`leaves: &[&Tensor<f32>]`）
         はどの葉も等しく host 常駐の参照として受け取り、必要な
         デバイス転送（アップロード）は各 `run_fused` 実装内部の責務で
         ある（§3.4 の default 実装コメント参照）。したがって `a` 側と
         `b` 側の backend（`ops.device()`）が異なっていても、`b` の
         実体化結果は `a` 側 `run_fused` へ渡る前に必ず host
         `Tensor<f32>` を経由するため、GPU/CPU をまたぐデバイスメモリの
         直接受け渡し（転送を経ない生ポインタ・ハンドルの混在）は構造的
         に発生しない。これは本改訂で新設する検証ではなく、「実体化は
         常に host `Tensor<f32>` を返す」という §3.4 の既存契約をケース 4
         にも一貫して適用した結果であり、`ops.device()` の一致検証や
         新規の明示的転送 API を追加する必要はない（`DeviceBuffer` を
         葉に直接持たせる設計へ将来移行する場合はこの前提が崩れるため、
         その時点で `ops.device()` 一致検証の追加が必要になる。§6.2
         未決事項へ記録する）。
       - `(a + b) * (c + d)` のような越境合流では、越境した側
         （`b` または `d`）の連鎖がその場で実体化されるため、`a`（または
         `c`）側の 1 回の `run_fused` 呼び出しへ単純化される。単一
         カーネルへの融合機会が失われる点は旧稿の想定と変わらないため、
         性能特性の記録は引き続き §6.2 未決事項に残す（下記）。
    いずれのケースでも、返す `Pending` な `Tensor<f32>`（`OnceLock`
    未初期化の `Storage::Pending` を積んだだけのプレースホルダ）自体の
    型・構築規約は変わらない。これが `Tape::push` によりそのまま
    テープノードの `value` として記録される（`tape.rs` の
    `nodes[id].value: Tensor<f32>` 自体の型は変わらない）。
  - 次に呼ばれる `Var::mul`（別の独立した呼び出し）は、直前の
    `Var::add` が返した `Var` の `.value()`（`var.rs:70` 付近）を経由
    してこの `Pending` な `Tensor<f32>` を読み出し、`eval::mul` へ渡す。
    こうして**別々の Rust メソッド呼び出しをまたいで** `FusionValue`
    が連鎖する（§1 冒頭の懸念「複数回の `add`／`mul` 呼び出しをまたいで
    伝播する呼び出し元が存在しない」への回答: 伝播を担うのは呼び出し元
    コードではなく `Storage` 内部に隠れた `Pending` 状態であり、呼び出し
    元は通常どおり `Var` のメソッドを連続して呼ぶだけでよい）。
  - **実体化の発火点（2 系統。Codex 再指摘「`BackendError` が消失する」
    への回答で 2 系統に分離した）**: `Storage::Pending` の実体化は
    `cache.get_or_init(|| source.materialize())`（`source.materialize()`
    は上記のとおり `Result<Vec<T>, BackendError>` を返す）で発火する
    という点は変わらないが、これを呼ぶ経路を用途別に分ける。
    - **系統 1（公開 `get`／`as_slice`。`tensor.rs:201`・`tensor.rs:231`
      付近。シグネチャは変更しない）**: 既存どおり `Option` を返す。
      内部で `cache.get_or_init(...)` を呼んだ後 `.as_ref().ok()` で
      `Result` を `Option` へ変換する（今日の非 contiguous ケースで
      `None` を返す既存契約と同じ形）。この経路はライブラリ利用者が
      直接呼びうる汎用アクセサであり、シグネチャ変更は破壊的変更に
      なるため `Option` 契約を維持する（`BackendError` の詳細はここでは
      表面化しない。これは意図的な設計判断であり、次点の系統 2 で
      補う）。
    - **系統 2（`dense_vec` を単一の発火点に統一する）**: 境界ノード
      （`gemm`／`sum`／`max`。§3.2 (a)(b)）に限らず、**VJP（`grad.rs::vjp`）
      も forward 記録値を読み出す実質的な実体化境界である**ことが本改訂
      で判明した（下記「VJP・`Tape` 構造への影響の訂正」参照）。両方の
      呼び出し元を通じて呼ばれるのは既存の `eval::dense_vec`
      （`eval.rs:41`。`autodiff` クレート `pub(crate)`、公開 API では
      ないためシグネチャ変更が可能）1 箇所であるため、実体化の型付き
      エラー伝播はここ 1 箇所に集約する。
      - `tensor-core` 側に新設するフォールブルアクセサ
        `Tensor::try_dense(&self) -> Result<Vec<T>, BackendError>`
        は **`pub`（`#[doc(hidden)]` 付与）とし `pub(crate)` にはしない**
        （codex-review 指摘 P1・PR #357: 呼び出し元 `eval::dense_vec` は
        別クレート `autodiff` に属し、`pub(crate)` は定義元クレート
        `tensor-core` 内からしか参照できないためコンパイルが通らない。
        クレート間可視性を Rust は `pub` と `pub(crate)` の間の粒度で
        提供しないため、ワークスペース内クレートから呼べる最小の可視性
        は `pub` である）。`#[doc(hidden)]` により `cargo doc` の公開 API
        一覧・compat API 層（REQ-9）の想定表面には現れず、ライブラリ
        利用者向けの契約を広げない（`.claude/rules/coding-rust.md`
        「互換 API 層は自作コアの上の薄いラッパーに徹する」を侵さない）。
        （`Materialized` は保持する `Vec<T>` を、`Pending` は
        `cache.get_or_init(...)` で実体化したキャッシュ済み融合出力を、
        それぞれ**呼び出し元 `Tensor` の基底ストレージ**として扱い、
        いずれの場合も同一のビュー適用ロジックで呼び出し元 `Tensor` が
        保持する `offset`／`shape`／`strides` を適用した稠密コピーを
        返す。`Pending` を「キャッシュ済み融合出力全体をそのまま
        `.clone()` して返す」旧稿の記述は誤りだった: transpose・
        narrow・reshape は `Storage::Pending` を共有したまま `Tensor`
        側の view メタデータのみを変更できる（下記「`offset`／
        `shape`／`strides` のみを扱う view 系操作」参照）ため、
        `Pending` を実体化する際に呼び出し元 `Tensor` のビューを
        適用しなければ、view 演算後に元テンソル全体の値・形状が
        返ってしまい正当性契約に反する。**戻り値は借用 `&[T]` ではなく
        所有 `Vec<T>` とする**: `Materialized` であっても strided
        （transpose・narrow 由来の非連続）view の場合は稠密化に新規
        `Vec` の確保が必要であり、既存の `as_slice()` が非連続入力に
        `None` を返す（`tensor.rs:769` `as_slice_contiguous_only`）のと
        同じ理由で `&[T]` を安全に返せない。`eval::dense_vec` が既に
        `-> Vec<f32>`（所有値）を返す契約（`eval.rs:41`）と合わせる
        ことで、既存の稠密化利用パターンをそのまま踏襲する。
      - `eval::dense_vec` は `tensor.try_dense()` を呼び、`BackendError`
        を握り潰さず `Result<Vec<f32>, BackendError>` としてそのまま
        返すようシグネチャを変更する（`-> Vec<f32>` から変更。#164 で
        `eval::add` が `ops: Option<Arc<dyn BackendOps + Send + Sync>>` を追加するのと
        同時に行う、§3.4 の変更と一体の作業）。
      - `dense_vec` の全呼び出し元（`eval::matmul`／`eval::sum`／
        `eval::max`〈§3.2 (a)(b) の境界ノード。連鎖長上限到達
        （§3.2 (d)）・非融合パターン検出（§3.2 (e)）を含む〉、および
        `grad.rs::vjp` とその内部で呼ばれる各演算の VJP 関数
        （`matmul_vjp` 等、`dense_vec` を呼ぶ全関数）は、`?` 演算子で
        `BackendError` をそのまま伝播できるよう戻り値を
        `Result<_, BackendError>` へ変更する。これらはいずれも
        `autodiff` クレートの `pub(crate)` 内部関数（`eval.rs`・
        `grad.rs` 双方の非公開シグネチャ）であり、公開 API ではないため
        この変更自体は非破壊（下記「`Tape::backward` までの型契約
        （Codex 再指摘）への回答」で公開境界の扱いを別途規定する）。
        #164 のスコープに `grad.rs` のシグネチャ変更（`vjp` 自体を含む）
        を明示的に含める（下記「VJP・`Tape` 構造への影響の訂正」参照）。
  - **失敗経路の扱い（Codex 再指摘への回答。旧稿の「理論上到達しない」
    という前提を撤回する）**: 旧稿は `materialize` の失敗を「`run_fused`
    が `Unsupported` を返す場合のみ」と仮定し、`Unsupported` は §4 の
    fail-safe で必ず逐次フォールバックに吸収されるため実質到達しない
    契約違反として扱っていた。これは誤りである。`run_fused`
    （`BackendOps` の非破壊拡張メソッド。上記コード例）は `Unsupported`
    以外にも、GPU 側カーネル実行失敗（`KernelLaunchFailed`）・NVRTC／MSL
    コンパイル失敗・デバイス側障害（`DeviceAllocationFailed`・
    `TransferFailed`）等、実行時に実際に起こりうる理由で
    `Err(BackendError)` を返しうる（`device.rs:184` 以降の
    `BackendError` variant 一覧のとおり、これらは実機依存の実行時障害を
    表す既存 variant であり「グラフ構築時の shape・dtype 検証で排除
    済みの契約違反」だけではない）。したがって系統 2（`dense_vec` 経由の
    すべての呼び出し元）の `Result` はこれら実行時障害を型付きエラーと
    して呼び出し元へ表面化させる必要があり、`debug_assert!` で握り潰す
    設計は採らない。系統 1（公開 `get`／`as_slice`）は既存の `Option`
    契約を維持するため引き続き詳細を失うが、これは「非 contiguous 等の
    理由で稠密データを返せない」という既存契約の延長として許容する
    （呼び出し元がエラー詳細を必要とする場合は系統 2 のアクセサ、または
    `FusionValue::into_tensor`／`FusionSession::materialize`（§3.4）を
    経由する必要がある）。
  - `offset`／`shape`／`strides` のみを扱う view 系操作（transpose・
    narrow・reshape の `Arc::clone(&self.storage)` 経路）はデータを
    読まないため、`Pending` のまま複製してよい（実体化を強制しない）。
    ただし §1 のとおり transpose 混在連鎖は非融合フォールバックへ倒す
    契約（`NodeMeta.contiguous == false`）のため、`FusionOp` へは組み
    込まず実体化境界として扱う（§3.2 (e) と整合）。**正当性契約
    （codex-review 再指摘 P1 への回答。本改訂で明記）**: この
    `Arc::clone` 経路は `Storage::Pending` 自体を複製するだけであり、
    view 演算後の `Tensor` は複製元と同じ `Storage::Pending` を共有し
    つつ `offset`／`shape`／`strides` だけが異なる状態になる。この
    `Tensor` に対する `try_dense`（上記「`Materialized` は保持する
    `Vec<T>` を…」の改訂契約）・VJP（`dense_vec` 経由）・後続の境界
    演算はいずれも、
    複製元テンソル全体ではなく複製後の `Tensor` が保持する
    `offset`／`shape`／`strides` を適用した結果を受け取らなければ
    ならない。`try_dense` を「基底ストレージ実体化＋呼び出し元ビュー
    適用」という単一契約に統一した（上記改訂）のはこれを満たすためで
    あり、`Pending` 側にだけ別のビュー無視経路が残らないようにする。
    transpose・narrow・reshape を `Storage::Pending` 上に適用した後の
    読み出し・VJP の正当性は、#164（ディスパッチ統合。`Storage::Pending`
    実装）の実装に付随するテストとして #165（テスト）のスコープに
    含める（§6.1 対応表に反映）。
- **VJP・`Tape` 構造への影響の訂正（本改訂）**: `Tape` が記録する `Op`
  単位のノード粒度・`grad.rs::vjp` の走査対象（`Op` 列）自体には影響
  しない（§3.3 の契約を変更しない。ノードグラフの形は不変）。ただし
  旧稿は「VJP は `Storage::Pending` を通じて**自動的に**実体化される」
  と記述しており、これは実体化が失敗しうる事実（上記「失敗経路の
  扱い」）を踏まえると不正確だった。`grad.rs::vjp`（および内部で呼ぶ
  `matmul_vjp` 等の各演算 VJP 関数）は forward 記録値
  （`nodes[id].value: Tensor<f32>`）を読み出す際に `dense_vec`
  （上記「系統 2」）を経由するため、`Op::Leaf` を除く VJP 呼び出しは
  実質的に実体化境界であり、`run_fused` の実行時障害を受けて失敗
  しうる。本改訂は `vjp`（`grad.rs:31`）とその内部の全 VJP 関数の
  戻り値を `Result<_, BackendError>` 化する（系統 2 と同じ理由）。これは
  `Tape`／`Op` の**構造**（ノード粒度・走査順）には影響しないが、`vjp`
  系関数群の**シグネチャ**には影響する（#164 のスコープに明示的に含める。
  本節冒頭「VJP・`Tape` 構造への非影響」という誤った見出しは本改訂で
  撤回する）。

  **`Tape::backward` までの型契約（Codex P1 再指摘「`Result<Gradients,
  AutodiffError>` に `BackendError` variant も `From<BackendError>` も
  存在せず、記載どおりの `?` はコンパイルできない」への回答。本改訂で
  確定する）**: 旧稿は「`BackendError` を型付きで `Tape::backward` まで
  伝播させる」とだけ記述しており、`backend.rs:73` の既存公開シグネチャ
  `pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`
  との整合を示さないまま `?` で素通しできるかのように書いていたが、これは
  誤りだった（`AutodiffError`（`crates/autodiff/src/error.rs:21`）は現状
  `BackendError` を包む variant も `From<BackendError>` 実装も持たない）。
  戻り値を `Result<Gradients, BackendError>` 等へ変更することは公開 API の
  破壊的変更になるため不採用とし、以下の 2 層契約で解決する（#164 の
  スコープに明示的に含める）:
  - 内部関数群（`tensor-core` の `Tensor::try_dense`〈`pub` +
    `#[doc(hidden)]`。上記のとおりクレート境界を跨ぐため `pub(crate)`
    にはできない〉、および `autodiff` クレート内で完結する
    `eval::dense_vec`／`grad.rs::vjp`／`matmul_vjp` 等の各演算 VJP 関数
    〈こちらは呼び出し元・呼び出し先が同一クレート内のため引き続き
    `pub(crate)` でよい〉）はいずれも `Result<_, BackendError>` のまま
    `?` で連鎖させる（上記のとおり）。
  - `Tape::backward`（`backward.rs:73`。公開シグネチャは
    `Result<Gradients, AutodiffError>` のまま変更しない）は、内部で
    `vjp(...)` を呼ぶ箇所においてのみ `BackendError` を
    `AutodiffError` へ変換する。変換は `AutodiffError`
    （`#[non_exhaustive]`。`error.rs:19`）への非破壊 variant 追加で行う:
    ```rust
    pub enum AutodiffError {
        // 既存 variant（Shape／Backward／TapeMismatch／InvalidArgument）は変更しない。
        /// 逆伝播中の実体化・カーネル実行で発生した型付きバックエンド
        /// エラー（TASK-12.1a／#164。`tensor_core::BackendError` をラップ）。
        Backend(tensor_core::BackendError),
    }

    impl From<tensor_core::BackendError> for AutodiffError {
        fn from(err: tensor_core::BackendError) -> Self {
            AutodiffError::Backend(err)
        }
    }
    ```
    `#[non_exhaustive]` enum への variant 追加・新規 `From` 実装はいずれも
    公開 API 非破壊（既存の呼び出し元の網羅的 `match` を壊さない。
    `error.rs:15-18` の既存方針と同じ理由）。`autodiff` は既に
    `tensor-core` に依存している（`crates/autodiff/Cargo.toml:14`）ため
    新規依存の追加は不要。`Tape::backward` 内部で `vjp(...)?` と書けば
    `?` 演算子が `From<BackendError> for AutodiffError` を経由して
    自動変換する（`vjp` 自体は `Result<_, BackendError>` を返す関数の
    まま、`backward` 側だけが `AutodiffError` へ変換する境界になる）。
  - `error.rs:66` 以降の `impl fmt::Display for AutodiffError` は
    `match` で全 variant を網羅しているため、`Backend` variant 追加時は
    対応する `Display` アームの追加も同時に行う（追加を怠るとコンパイル
    エラーになる。実装時の見落とし防止のため本節に明記する）。
- **スコープの明示**: 本節が対象とするのは `Var` 経由（autodiff テープ
  構築）の演算チェーンに限る。`&dyn BackendOps` を直接呼ぶ、autodiff を
  経由しない生の `Tensor` 操作（§1・§3.4 で既に対象外と明記済み）は
  引き続き対象外であり、この経路の `Storage` は常に `Materialized` の
  ままでよい（`ops_for(...).add()` 等の実装は `Pending` を生成しない）。

## 4. バックエンド・規約との契約

- 融合カーネル（#163 で生成）も **FMA 契約統一**（CPU `f32::mul_add`・
  GPU 既定 FMA 契約）と**数値一致複合判定「相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満」**に従うこと（`.claude/rules/coding-rust.md`
  「バックエンド構成（REQ-2）」）。融合の有無で許容誤差を変えない。
  許容誤差はユーザー承認必須事項であり、本文書では緩和しない。
- **REQ-8 境界検査規約**: 融合カーネル生成時もシェーダ・カーネル側の手動
  境界チェックを省略しないことを設計制約として明記する
  （`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」）。
  融合による最適化（ベクトル化ロード・タイル端の分岐削減等）を適用する
  場合も、手動境界チェックを維持したうえで行う。CPU（intrinsics）・
  CUDA（NVRTC/mma）・Metal（simdgroup）いずれの融合カーネルにも適用する。
- 未実装カーネル・非対応バックエンドは `BackendError::Unsupported`
  （`crates/tensor-core/src/device.rs:218`）による fail-safe（非融合経路
  へフォールバック）とする既存方針を踏襲する（`backend_ops.rs` の
  elementwise・reduction 未実装カーネルに対する既存の fail-safe 設計と
  同型）。
- グラフ構築 API はテンソル shape／stride の検証を先行させる。既存の
  `ShapeError`（`crates/tensor-core/src/error.rs:19`）経路をそのまま
  再利用し、融合グラフ構築時に独自の検証経路を新設しない（§5 参照）。

## 5. セキュリティ設計制約（OWASP A03・A08 観点）

- **A03（インジェクション）**: 融合カーネル生成（#163）は本文書 §2.1 の
  閉じた `FusionOp` enum の組み合わせからのみ NVRTC（CUDA）／MSL（Metal）
  ソースを組み立てる。外部入力文字列（ユーザーが渡すテンソル値・shape
  以外の任意文字列）をカーネルソースへ直接展開しない。グラフ構築 API は
  §4 のとおり `ShapeError` 検証を先行させる。
- **A08（ソフトウェア・データ整合性）**: 融合経路は数値一致複合判定・
  ガードレール 3 分岐判定の**迂回経路にならない**（§4）。融合の有無で
  テスト許容誤差を変える実装は認めない。

## 6. 後続イシューへの引き継ぎ・未決事項

### 6.1 対応表

| イシュー | 実装する節 |
|---|---|
| #162（連鎖検出） | §2（グラフ表現・ノード種別・メタデータ・fan-out）を用いた融合可能連鎖（elementwise のみで閉じた 4〜6 段の連結成分）の検出アルゴリズム |
| #163（融合カーネル生成） | §2.4 の fan-out レジスタ内解決方針、§3.4 で確定した `FusionPlan::ops`（`FusedOpKind` 列挙）／`output_shape`／`dtype`／`leaf_count`／`use_count` の公開 DTO アクセサを読んだカーネルソース生成、§4・§5 の境界検査・数値一致・インジェクション対策 |
| #164（ディスパッチ統合） | §1 の「利用者向け制御 API を提供しない」方針に基づく融合対応経路の実装、§3.2 の実体化条件・§3.4 の `FusionValue`／`FusionSession`／`BackendOps::run_fused` 接続契約（`Arc<Mutex<FusionGraph>>`・`Arc<dyn BackendOps + Send + Sync>` 所有モデル。`BackendOps` trait 定義自体は変更せず `Send + Sync` は融合機構の型注釈側でのみ課す）・§3.4 で確定した `Tape::with_backend(ops)` 新規公開コンストラクタと `eval::add`／`mul`／`relu`／`exp`／`tanh` への `ops: Option<Arc<dyn BackendOps + Send + Sync>>` 引数追加（＝ TASK-1.9 の backend 経由実行への置き換えと同時実施）・§3.5 の `Storage` への `OnceLock` ベース `Pending` バリアント追加と `autodiff::eval` 側の伝播ロジックの実装（§3.5「`Materialized` は保持する `Vec<T>` を…」で確定した `try_dense` の「基底ストレージ実体化＋呼び出し元ビュー適用」契約を含む）・§3.5「`Tape::backward` までの型契約」で確定した `AutodiffError::Backend(BackendError)` variant と `From<BackendError>` 実装の追加（`Display` アーム追加を含む） |
| #165（テスト） | §1・§2.3 の transpose 非融合フォールバック、§2.4 の fan-out 融合、§3.3 の autodiff 契約（VJP がノード単位のまま変わらないこと）の検証、§3.5「正当性契約（codex-review 再指摘 P1 への回答）」で明記した transpose・narrow・reshape を `Storage::Pending` 上に適用した後の `try_dense`／VJP／後続の境界演算が元テンソル全体ではなく view 適用後の値・形状を返すことの検証 |
| #203（GEMM epilogue 融合） | §3.2 (b) の `gemm` 境界を bias／activation epilogue まで拡張する設計変更 |

### 6.2 未決事項（スコープ外）

- **transpose 混在連鎖のメタデータ融合**: §1・§2.3 のとおり初期スコープ
  では transpose 検出時に非融合フォールバックへ倒すため、v1 fusion 有効
  時の性能水準（PoC-9 `ew_reshape` 実測で最大 13.89 倍差）は初期スコープ
  では達成しない。ストライド付きビューを融合 IR（§2）内で表現・伝播する
  設計（`NodeMeta` へのストライド情報追加等）ができれば transpose を
  融合対象へ含められる可能性があり、#162 以降の拡張候補として記録する。
- **f16 対応**: `BackendOps`（§2.1）・`NodeMeta.dtype`（§2.3）とも現状
  f32 固定であり、f16 融合カーネルの型設計は未着手。`BackendOps` 自体の
  f16 ジェネリック化（`backend_ops.rs` コメント「f16 経路のジェネリック化
  は保留」）に追従する形で将来検討する。
- **reduction エピローグの手動融合**: REQ-12 受け入れ基準は「性能クリティ
  カルな箇所では、CubeCL カスタムカーネルによる手動融合（reduction を
  含めた完全融合）を組込み演算として提供する選択肢を将来検討課題とする
  こと」と記載する（`docs/spec/04-requirements.md:251`。v1 CubeCL 前提の
  文言だが、自作カーネルでの reduction epilogue 融合という論点自体は
  引き継ぐ）。本文書は §3.2 (a) で reduction を実体化境界として扱う
  （初期スコープ外）に留め、将来の手動融合対応は本節への記録に留める
  （`.claude/rules/out-of-scope-tracking.md` の規約上、自動運転中のため
  新規 Issue 起票は行わずここに記録する）。
- **REQ-12 自体の v2 書き直し未実施**: `docs/spec/05-tasks.md:552` が
  指摘するとおり、REQ-12 の受け入れ基準文言は `burn-wgpu` `fusion`
  feature・`CUBECL_DEBUG_LOG` を前提としたまま（v2 全面改定を受けていない）
  である。本文書は TASK-12.1〜12.2 の読み替え（自作 elementwise 融合機構）
  に基づいて設計しているが、REQ-12 自体の文言更新は `docs/spec/`
  正本リポジトリ側の課題であり、本イシューのスコープ外である
  （`docs/spec/` は編集しない。`.claude/rules/delegation-impl.md`）。
- **異なる `FusionSession` を跨ぐ融合境界の性能特性**: §3.5「異なる
  セッション間の合流」（PR #357 review 再指摘 P1-1／P1-2 への回答で
  「合流せず、越境する側をその場で実体化してから葉ノードとして埋め込む」
  へ確定）の契約は正しさ（循環参照の非発生・backend 越境転送の安全性）
  を保つが、`(a + b) * (c + d)` のように独立に構築された 2 連鎖が
  1 つの二項演算で合流する場合、越境した側（`c + d`）はその場で
  実体化され、`a+b` 側の `run_fused` 呼び出し 1 回へ単純化される
  （単一カーネルへは融合されない。両連鎖が偶然同一 backend でも
  同様）。この性能上限（未融合になる境界の頻度・実測影響）は #164 の
  受け入れ条件には含まれておらず、実装後のベンチ（bench-runner。5 回
  計測中央値）で計測し、必要なら合流検出・並べ替え等の追加最適化を
  #162 以降の拡張候補として別途検討する。
- **葉ノードを `DeviceBuffer` 直接参照へ最適化する場合の device 一致
  検証**: §3.5 ケース 4（PR #357 review 再指摘 P1-2 への回答）は葉が
  常に host 常駐の `Tensor<f32>` を経由する現行契約に依拠して backend
  越境転送の安全性を導いている。将来 `run_fused` の葉を `DeviceBuffer`
  （§4.2。デバイスメモリハンドルを直接保持）へ最適化し host 往復を
  省く設計に変更する場合、この前提が崩れるため、その時点で葉ごとに
  `ops.device()` と `DeviceBuffer::device()` の一致を fail-closed で
  検証する（不一致は型付きエラーで拒否する）契約を新設する必要がある。
  現行スコープ（TASK-12.1a）では `Tensor<f32>` 経由の host 往復のみを
  対象とするためこの検証は不要であり、#162 以降の拡張候補として記録
  する。
