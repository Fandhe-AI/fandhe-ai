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
  `autodiff::Tape` が既に保持する内部状態（`ops`。§3.4）を使う内部
  実装の一部として実現し、新規の公開エントリ関数は追加しない（適用
  箇所の具体化は §3.5、`BackendOps` 契約との接続は §3.4 で規定する）。
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
  - **公開コンストラクタの選択は融合スイッチにならない（codex-review P1
    指摘への回答。本改訂で確定する契約）**: `autodiff::Tape` の公開
    コンストラクタは `Tape::new()`（既存）・`Tape::with_backend(ops)`
    （§3.4 で新設）の 2 つを持つが、**いずれも同一の非公開フィールド
    `ops: Option<Box<dyn BackendOps>>`（§3.4）へ到達し、`Tape::backward`
    の VJP 連鎖内部（§3.5.2）・将来の複合エントリポイント（§3.5.3）が
    これを共通して参照する**。融合するか否かはこれらの内部実装が判定
    する次の 2 条件のみで決まり、呼び出し元がどちらの公開コンストラクタ
    を呼んだかには依存しない:
    1. **バックエンド解決可否**: 実行に使う `BackendOps` 実装（`ops`）を
       内部方針で解決できたか。`Tape::with_backend(ops)` は解決手段の
       1 つ（呼び出し元が明示供給する）にすぎず、`Tape::new()` も内部で
       既定バックエンド解決を試みたうえで同じ判定に合流する（§3.4
       「`Tape::new()`」の項）。
    2. **演算列の融合可否判定**: §3.2 の実体化条件・§2.3 の
       `NodeMeta.contiguous` 等、演算列自体の構造から決まる判定。
    - `with_backend` は「バックエンドを明示供給する」手段であり、融合の
      有効化手段ではない（命名・doc コメントも §3.4 でこの理解に統一
      する）。「`with_backend` を呼べば融合し、`new()` は融合しない」と
      いう対応関係を本文書のどこにも記載しない（§3.4・§3.5 の記載を
      本改訂で統一する）。「融合対象区間・単一の fallible 呼び出し」の
      範囲自体は §1「遅延の生存窓は単一の fallible 呼び出しの内部に
      限定する」（下記）で確定する、より狭い契約に置き換わる。
  - **遅延の生存窓は単一の fallible 呼び出しの内部に限定する
    （codex-review 第 5 波 P1 指摘への回答。本改訂で確定する契約。
    詳細は §3.2・§3.4・§3.5）**: 第 4 波までの設計は `Var::value`／
    `Var::to_tensor` を非 fallible な公開読み出し境界としたまま、その
    背後で `Storage::Pending`（未実体化状態）を複数回の独立した公開
    `Var` 演算呼び出し（`a.add(&b)?.mul(&c)?` のように、各 `?` が別々の
    呼び出し境界を構成する連鎖）をまたいで持ち越し、実体化失敗を
    `cache` へ「キャッシュ」して後続の呼び出しで表面化させる方式を
    採っていた。しかし `value`／`to_tensor` は呼び出し自体が「結果を
    受け取る」という参照行為であるにもかかわらず、その時点でまだ
    実体化を試みていない・実体化に失敗していても呼び出し元へ一切
    通知されない、という**契約破壊**を生む（第 5 波 P1 指摘）。本改訂は
    実体化を失敗しない境界まで前倒しする案を採る: **`Var` の fallible
    演算（`add`／`mul`／`matmul`／`sum`／`max`。既に `Result<Var<'_>,
    AutodiffError>` を返す契約）は、返る前に自分の出力を実体化済みに
    する**。融合実行の失敗はその演算自身の `Err` として型付きで返り、
    呼び出し元は結果を受け取った時点で常に実体化済みの値を得る。
    **`relu`／`exp`／`tanh`（shape 不変の単項演算。「構造的に失敗し
    えない」という既存設計判断〈`docs/public-api-design.md` §3.2〉に
    より非 fallible な `fn relu(&self) -> Var<'t>` のまま〈`var.rs:257`
    以降〉）はこの窓の対象外のままとし、本文書は非 fallible 契約を
    変更しない**（§3.5.1 で確定する詳細な取り扱いを参照）。したがって
    利用者が保持する
    `Var`／公開 `Tensor` は常に実体化済みであり、`value`／`to_tensor`／
    `get`／`as_slice` の既存非 fallible 契約は完全に不変のまま保たれる
    （第 4 波で導入した「系統 3」「`Storage::Pending` の `cache` への
    `Err` キャッシュ」「`value_raw` による内部連鎖読み出しの分離」は
    本改訂で撤回・削除する。§3.5 で置き換え後の設計を規定する）。
    融合の効果はこの縮小された窓の内側でのみ得られる（§3.2 (c)・
    §3.4・§3.5 で具体化）: (a) `Tape::backward` の VJP 連鎖内部（`backward`
    自体が単一の fallible 呼び出しであり、その内部で複数のノードの
    VJP 計算を融合しうる）、(b) 単一の fallible 呼び出し内部で複数の
    elementwise 演算を行う複合エントリポイント（現状の `Var` 演算 API は
    1 呼び出し 1 演算の粒度であるため、この窓の恩恵は将来
    `compat::Sequential::forward` 相当〈設計段階。
    `docs/public-api-design.md`〉や、複数演算を 1 回の `Result` で返す
    将来の複合演算 API が追加された時点で顕在化する）、(c)
    `FusionSession::materialize`（§3.4・§3.5.4）の直接呼び出し。**個々の公開
    `Var` 演算の呼び出しをまたぐ遅延は行わない**。REQ-12 の「透過的
    融合」（利用者制御 API を提供しないまま融合が働くこと）は、この
    (a)〜(c) の窓の内側で成立する設計として読み替える。これにより
    第 1〜4 波で確定した契約（view 適用・融合スイッチ非提供・`Option`
    へのエラー非流入・公開 `Tensor` 常時実体化）はいずれも「公開
    `Var` 演算は常に実体化済みの結果を返す」という単純な前提のもとで
    自動的に成立する（矛盾する記述の整理は §3.2・§3.4・§3.5・§6.1 で
    横断的に行う）。この縮小の対価として、PoC-9 実測（`ew4`／`ew6`）が
    示す 2.25〜3.19 倍の高速化は、独立した公開 `Var` 呼び出しをまたぐ
    elementwise 連鎖（現状の API 形状そのもの）には及ばなくなる。
    これを初期スコープの受け入れコストとして §6.2 に明示的に記録する
    （§1 冒頭の PoC-9 実測引用は連鎖検出（#162）が対象とする IR 一般の
    妥当性根拠として維持しつつ、公開 API を横断した自動適用の主張は
    ここで撤回する）。
    - **比較検討: 「`Result` を返す読み出し API を追加し、遅延値は
      必ずそこから取得させる」案（不採用）**: 第 5 波指摘が挙げるもう
      一方の選択肢として、`Var::value`／`Var::to_tensor` の非 fallible
      契約はそのまま残し、代わりに `Var::try_value`／`Tensor::try_get`
      相当の `Result` 返却アクセサを新設し、遅延値の実体化失敗はそちら
      からのみ観測させる、という設計も検討した。この案は `Storage`
      に「まだ誰も実体化を試みていない」という状態が存在し続けること
      自体は許容するため、利用者が非 fallible な `value`／`to_tensor`
      を呼んだ場合には引き続き失敗が通知されない契約破壊が残る
      （新設した `Result` 版を呼ばない限り安全側にならない、いわば
      オプトインの回避策にすぎない）。加えて、互換 API 層（REQ-9）が
      前提とする「自作コアの上の薄いラッパーに徹する」方針に対し、
      遅延値専用の新しい公開アクセサ系列を追加することは公開 API 面を
      不必要に広げる（第 4 波で系統 1 について行った比較検討と同じ
      理由）。採用した前倒し案は、公開シグネチャを一切追加せず
      （`Var` の演算メソッドは既に `Result` を返す契約であり、そこへ
      実体化失敗を合流させるだけで済む）、かつ「呼び出し自体が結果
      参照である」という `value`／`to_tensor` の意味論を字義どおり
      満たせるため、本改訂ではこちらを採用する。
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
| (c) | 遅延ハンドル（§3.4・§3.5）から `Tensor<f32>` への変換（`FusionSession::materialize` 呼び出し）を含む、**単一の fallible 呼び出しが自身の結果を返す直前**（§1「遅延の生存窓は単一の fallible 呼び出しの内部に限定する」） | 遅延構築されたグラフの結果を呼び出し元へ返す時点で計算が確定していなければならない。`Tensor`（`tensor.rs:53`）自体は `Arc<Storage<T>>` を必須で保持する既存表現のまま変更せず、`Storage<T>`（非公開）も `Pending` バリアントを持たない（§3.5 で確定。旧稿は `Pending` バリアントを追加していたが本改訂で撤回する）。したがって `Tensor::get`／`as_slice`／`contiguous`（`tensor-core` の汎用アクセサ）にも「未実体化」を表す分岐は存在しない。実体化を発火させるのは、その fallible 呼び出し自身の内部実装（§3.2 (a)(b) の境界ノード到達、または呼び出しが自身の出力を組み立てて `Ok` を返す直前）だけであり、`autodiff` の外へ渡る値・`Var::value`／`Var::to_tensor`／`Gradients::get` が観測する値は常にこの窓の内側で実体化試行済みである（§3.5 参照） |
| (d) | 連鎖長上限（4〜6 段）到達 | TASK-12.1 の内容規定（4〜6 段程度）。PoC-9 の代表ワークロード規模（`ew4`／`ew6`）とも整合する上限であり、無制限連鎖によるカーネル生成コスト・レジスタ圧の増大を避ける |
| (e) | 非融合対象パターン検出（transpose 混在等、`NodeMeta.contiguous == false`）| §1・§2.3 の非融合フォールバック方針 |

### 3.3 autodiff との関係

動的テープ式 autodiff（PoC-v2-2、`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`。
実装は `crates/autodiff/src/tape.rs`・`eval.rs`）は、**forward・
backward いずれの実行方式にも透過的に融合が働きうる**構成とする
（codex-review 第 5 波 P1 指摘を受けた本改訂での訂正: 旧稿「forward
値計算の下層」という限定は不正確だった。§1 で確定した「単一の
fallible 呼び出しの内部」という窓は forward・backward のどちらの
呼び出しにも同じ形で適用される）。すなわち:

- `Tape` が記録する `Op`（`tape.rs` の `Op` enum、MatMul／Add／Mul／
  Relu／Exp／Tanh 等）のノード単位の粒度は、融合の適用有無に関わらず
  変更しない。
- 勾配計算（VJP、`grad.rs::vjp`）は `Op` 単位のまま変更しない契約と
  する。融合はあくまで**ある単一の fallible 呼び出しの内部でどう
  カーネルを呼ぶか**という実行方式の最適化であり、テープが記録する
  計算グラフの構造（VJP が辿るノード単位）には影響を与えない。
- **実質的な適用箇所（§1 で確定した窓に一致する）**: 個々の `Var`
  fallible 演算メソッド（`add`／`mul`／`matmul`／`sum`／`max`）は
  1 呼び出し 1 演算の粒度であり、その内部だけで完結する融合機会は
  実質存在しない（融合対象は複数演算の連鎖であるため。`relu`／
  `exp`／`tanh` は非 fallible なままこの窓の対象外。§3.5.1）。透過的
  融合の実質的な適用箇所は `Tape::backward` の VJP 連鎖内部（窓 (a)。
  §3.5.2）、および将来追加されうる複合エントリポイント（窓 (b)。
  §1・§3.5.3）に限られる。

### 3.4 遅延グラフと `BackendOps`・`Tensor` 契約の接続

（PR #357 review 指摘への対応で追加。codex-review 第 5 波 P1 指摘を
受け、本節は §1「遅延の生存窓は単一の fallible 呼び出しの内部に限定
する」という縮小後の契約に合わせて全面改訂する。§1・§3.1〜3.3 は
「透過的」「遅延構築」という表現のみで、遅延グラフの所有場所・具体的な
型・`BackendOps`（`crates/tensor-core/src/backend_ops.rs:63`）との接続
経路を規定していなかった。`BackendOps` の各メソッドは具体化済みの
`Tensor<f32>` を受け取り直ちに具体的な `Tensor<f32>` を返す契約であり、
`Tensor`（`crates/tensor-core/src/tensor.rs:53`）は `Arc<Storage<T>>` を
必須で保持する公開型としては変わらない。本節はこの契約と遅延グラフを
どう接続するかを明示する（窓の内側での実際の使用点の具体化は §3.5 で
行う）。旧稿（第 4 波まで）は遅延グラフを `Tensor` の `Storage` へ
埋め込み、複数回の独立した公開呼び出しをまたいで持ち越す設計を
採っていたため、`Storage` から複数箇所で共有されうる `Tensor` の
`Send + Sync` を保つための `Arc<Mutex<_>>`・`Arc<dyn BackendOps + Send
+ Sync>` 所有モデルを要求していた。本改訂はその前提（遅延グラフが
`Tensor` を経由して外部へ漏れ出すこと）自体を撤回するため、以下は
その帰結として全面的に単純化される。）

- **`Tensor` は変更しない**。公開型 `Tensor`（構造体・フィールド型・
  メソッドシグネチャ）は破壊的変更を避ける（公開 API 非破壊はガード
  レール条件、`.claude/rules/security.md`「A08」・
  `docs/spec/04-requirements.md` の REQ-12 受け入れ基準とも整合させる
  安全側の選択）。**非公開の `Storage<T>`（`tensor.rs:33`）にも「未実体化」
  を表すバリアントは追加しない**（第 4 波までの旧稿は `Storage::Pending`
  を新設していたが、本改訂で撤回する。§3.5 参照）。`Tensor` は構造体・
  非公開実装のいずれも本節の変更対象外である。
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
  /// 単一の fallible 呼び出し（`Tape::backward` 内の VJP 連鎖、または
  /// 将来の複合エントリポイント。§1）の実行スタック内だけで構築・破棄
  /// される、融合対象区間 1 本分のグラフビルダー。呼び出し元の関数
  /// フレームを越えて共有・保持されることはなく、`Tensor`／`Storage`
  /// のどのフィールドにも格納されない。**`Arc`／`Mutex`／`Send + Sync`
  /// 境界は一切不要である**（旧稿はこれらを `Storage::Pending` として
  /// `Tensor` へ埋め込むために要求していたが、その前提自体を §1 で
  /// 撤回したため本改訂で単純化する）。
  pub(crate) struct FusionSession<'ops> {
      graph: FusionGraph,
      /// このセッションの生存期間だけ借用する `BackendOps` 実装。
      /// 呼び出し元の関数フレーム内で完結するため所有権を持つ必要が
      /// ない（下記「ops の受け渡しは借用で足りる」参照）。
      ops: &'ops dyn BackendOps,
  }

  /// グラフ構築中に扱う 1 つの中間値。既に確定済みの `Tensor<f32>`
  /// （葉ノード・外部から渡された既存値）か、`session` 内にまだ実行
  /// していないノードとして積まれているか（`Pending`）のいずれか。
  /// `Pending` はこの呼び出しが所有する `FusionSession` のグラフ内に
  /// のみ存在し、呼び出しの外へ持ち出されることはない。
  pub(crate) enum FusionValue {
      Materialized(Tensor<f32>),
      Pending(FusionNodeId),
  }

  impl<'ops> FusionSession<'ops> {
      /// §3.2 の実体化条件 (a)〜(e) のいずれかに到達した時点、または
      /// 呼び出し元の fallible 関数が自身の結果を返す直前に呼ぶ。
      /// `FusionValue::Materialized` はそのまま返し、`Pending` は
      /// `self.graph`／`self.ops` を使って実際に計算する。
      pub(crate) fn materialize(&self, value: FusionValue) -> Result<Tensor<f32>, BackendError> {
          match value {
              FusionValue::Materialized(t) => Ok(t),
              FusionValue::Pending(node) => {
                  // `FusionPlan::from_graph`／`FusionGraph::leaves` の
                  // シグネチャは下記「`FusionPlan` の構築・葉の収集」で
                  // 確定する（実装は #163／#164）。
                  let plan = FusionPlan::from_graph(&self.graph, node);
                  let leaves: Vec<&Tensor<f32>> = self.graph.leaves().iter().collect();
                  self.ops.run_fused(&plan, &leaves)
              }
          }
      }
  }
  ```

  - `FusionSession` は**スレッドローカルなグローバル状態にしない**。
    `dispatch.rs` の既存方針（`select_gemm_kernel` は環境変数・グローバル
    設定による経路上書きを持たない副作用なしの純関数設計、
    `crates/tensor-core/src/dispatch.rs:9-17`）と整合させるため、融合
    グラフの所有もディスパッチ層のローカル値（1 回の fallible 呼び出しの
    実行中にのみ生成し、その呼び出しが返る前に破棄する明示的な値）に
    限定し、暗黙のグローバル・スレッドローカルレジストリを設けない。
  - **`ops` の受け渡しは借用で足りる（codex-review 第 5 波 P1 指摘を
    受けた本改訂での単純化）**: 旧稿（第 4 波まで）は `FusionSession` を
    `Arc<Mutex<FusionGraph>>`・`Arc<dyn BackendOps + Send + Sync>` として
    所有値で保持していた。理由は「`Storage::Pending`（`Tensor` が `Arc`
    経由で複数箇所から共有されうる非公開フィールド）へ埋め込まれ、
    `&self` のみの `Tensor::get`／`as_slice` から追加引数なしに実体化を
    発火できる必要がある」ことだった。本改訂は §1 のとおり `Storage` に
    `Pending` バリアントを一切追加しない契約へ縮小したため、この前提
    そのものが消滅する。`FusionSession` は呼び出し元の関数フレーム内で
    構築され、その関数が返る前に消費し尽くされる（`materialize` を呼び
    終えたら破棄される）ローカル値であるため、`graph: FusionGraph`
    （所有値。`Arc`／`Mutex` 不要）・`ops: &'ops dyn BackendOps`
    （借用。`Arc`／`Send + Sync` 不要）で足りる。`Mutex` によるロック・
    スレッド越境共有への配慮（旧稿が検討していた懸念）はそもそも
    生じない。
  - **`BackendOps` trait 定義（`backend_ops.rs:82`）自体は変更しない**:
    `Send + Sync` をスーパートレイトとして追加しない。`BackendOps` は
    `pub trait` であり、本リポ外の crate が独自に実装する可能性を
    排除できない（trait 定義側の変更が非破壊かどうかは自クレート内の
    実装数ではなく、trait を実装しうる全ての利用者に対して判定する
    必要がある。`.claude/rules/security.md` の A08・本リポ全体の
    「公開 API 非破壊はガードレール条件」方針）。`Send + Sync` を
    スーパートレイトとして追加すると、これを満たさない既存の外部
    `BackendOps` 実装（内部可変状態に `Rc`／`RefCell` 等を使う実装）は
    コンパイル不能になり、破壊的変更（`!` 接頭辞・`BREAKING CHANGE:` 告知
    が必要な変更。`.claude/rules/conventional-commits.md`）に該当する。
    この理由は §1 の窓の縮小とは独立に成り立つ（trait 定義への
    スーパートレイト追加は、それを要求する側の設計がどう変わっても
    常に破壊的変更である）ため、本改訂でも維持する。上記の単純化に
    より、`FusionSession`／`Tape::with_backend`（下記）のいずれも
    `Send + Sync` を要求しない（旧稿はトレイトオブジェクト型の指定にの
    み `+ Send + Sync` を課していたが、その必要性自体が消滅したため
    本改訂で撤回する）。
  - **`ops` をどの時点で・どの形で受け渡すか（codex-review 第 5 波 P1
    指摘を受けた本改訂での単純化）**: `FusionSession` を開くのは §3.5 の
    とおり `Tape::backward` の VJP 連鎖内部、または将来の複合エントリ
    ポイント（§1 (b)）であり、いずれも `Tape` が既に保持する
    `BackendOps` 実装を使う。`Tape` は非公開フィールドとして
    `ops: Option<Box<dyn BackendOps>>` を保持する（`None` はバックエンド
    解決が不能だった場合、`Some` は解決に成功した場合を表す。フィールド
    追加は `Tape` の構造体を非公開のまま拡張するだけであり、`pub`
    フィールドを持たない現行の `Tape`（`tape.rs:140`）の公開契約を
    破らない）。旧稿（第 4 波まで）は `Storage::Pending` へ埋め込むために
    `Arc<dyn BackendOps + Send + Sync>` の所有値・`ops_for_arc`（`ops_for`
    の `Arc` 版姉妹関数）を新設していたが、§1 の窓の縮小によりこの前提
    が消滅したため、本改訂で `Box<dyn BackendOps>`（`Send + Sync` 不要・
    `ops_for_arc` 新設も不要）へ単純化する。`Tape::backward`（下記）は
    `self.ops.as_deref()`（`Option<&dyn BackendOps>`）を、VJP 連鎖の
    融合を行う内部ヘルパーへ**借用として**渡す（§3.4 冒頭「`FusionSession`
    は借用 `ops: &'ops dyn BackendOps` を保持する」）。`FusionSession` は
    その呼び出しの実行中だけ生存するローカル値であるため、所有権の移動・
    `Arc` によるクローンはいずれも不要である。
    ```rust
    impl Tape {
        /// 既存の既定コンストラクタ（`tape.rs:154`）。シグネチャは
        /// 変更しない（非破壊）。**内部では `with_backend` と同一の
        /// フィールド（`ops`）へ到達する**: 既定バックエンド解決を内部
        /// 方針として試み、解決できれば `Some` を、できなければ `None` を
        /// `self.ops` に保持する（§1「公開コンストラクタの選択は融合
        /// スイッチにならない」）。**既定バックエンド解決の具体的な規則
        /// （どの `Device` をどう選ぶか）は本文書では確定しない**
        /// （`docs/public-api-design.md` §4.1「既定デバイス選択ロジックは
        /// …実装しない」の方針のとおり、CUDA 既定有効化を含む構成決定は
        /// ユーザー承認必須のまま未確定。§6.2 に未決事項として記録する）。
        /// **この規則が確定するまでの間、`Tape::new()` 経由の既定
        /// バックエンド解決は常に失敗し、`self.ops` は `None` のままと
        /// なる**。これは「`new()` だから融合しない」という公開 API
        /// 起因の分岐ではなく、「バックエンド解決可否」という内部方針が
        /// 現時点で不能という判定結果にすぎない（規則が確定すれば
        /// `Tape::new()` 経由でも解決に成功しうる）。既存テスト資産
        /// （TASK-1.5〜1.8 等）はこのパスを使い続けており、現時点の
        /// 挙動もシグネチャも本改訂で変更しない（非破壊）。
        pub fn new() -> Tape { /* 既存実装のまま */ }

        /// バックエンドを明示供給するコンストラクタ（実装は #164。本節が
        /// 確定する供給契約。TASK-1.9「backend 経由の実行への置き換え」
        /// の一環として新設する）。**融合を有効化する手段ではない**
        /// （§1）: `Tape::backward` が VJP 連鎖内部で融合できるかは
        /// §1 の 2 条件（バックエンド解決可否・演算列の融合可否判定）
        /// のみで決まり、本コンストラクタは「バックエンド解決」を
        /// 呼び出し元の明示供給で満たす 1 手段にすぎない。**既定デバイス
        /// 選択ロジックはここでも導入しない**（`docs/public-api-design.md`
        /// §4.1 の確立済み方針を踏襲。`Device::Cpu` への暗黙フォール
        /// バックは行わない）。
        pub fn with_backend(ops: Box<dyn BackendOps>) -> Tape {
            /* 既存フィールドに `ops: Option<Box<dyn BackendOps>> = Some(ops)`
               を追加保持する以外は `Tape::new()` と同じ初期化（実装は #164）。 */
        }
    }
    ```
- **実際のカーネル呼び出し経路（`FusionSession::materialize` が内部で
  呼ぶ `run_fused`）は `BackendOps` の非破壊拡張（デフォルトメソッド）で
  提供する**。`backend_ops.rs` 冒頭コメントが既に採用している拡張
  パターン（「`BackendOps` の非破壊拡張（デフォルトメソッド追加等）」
  `backend_ops.rs:27` 付近）をそのまま踏襲する。

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
      // 所有値として構築する（`Arc`／`Rc` は不要）。`FusionSession::materialize`
      // が `self.graph`〈§3.4 冒頭。ローカル所有の `FusionGraph`〉から
      // その場で構築し、`run_fused` の呼び出しが終わるまでの間だけ
      // 生存すれば足りる（旧稿は `Storage::Pending` へ埋め込む前提で
      // `Arc` 所有・`Send + Sync` 保持を要求していたが、§1 の窓の縮小に
      // よりこの前提は消滅した。本改訂で単純化する）。
      graph: FusionGraph,
  }

  /// `FusionPlan` の構築・葉の収集（codex-review 第 5 波指摘への回答。
  /// `FusionSession::materialize`〈上記〉が呼ぶ `FusionPlan::from_graph`／
  /// `FusionGraph::leaves` のシグネチャを本改訂で確定する。「アクセサは
  /// いずれ追加する」という先送りを避けるため、`impl FusionPlan` の
  /// 公開 DTO アクセサ〈下記〉と同じ体裁でここに固定する）。
  impl FusionGraph {
      /// このグラフに登録済みの葉ノード（`FusionOp::Input`。§2.1）に
      /// 対応する実体 `Tensor<f32>` を発生順に返す。グラフ構築側
      /// （`FusionSession` へのノード追加処理。実装は #164）が
      /// `FusionOp::Input` の追加と同時に記録する（`pub(crate)`。
      /// `tensor-core` 内から `FusionSession::materialize` のみが呼ぶ）。
      pub(crate) fn leaves(&self) -> &[Tensor<f32>];
  }

  impl FusionPlan {
      /// `graph` のうち `root` を出力とする部分グラフから融合対象区間
      /// （境界ノード Gemm／Sum／Max を含まない、§3.2 (a)(b) で実体化
      /// 済みの部分より内側）を切り出し、`FusionPlan` を構築する
      /// （`pub(crate)`。実装は #163／#164。§2.4 の fan-out 情報
      /// 〈`NodeMeta.use_count`〉もこの構築時に算出し、下記
      /// `use_count` アクセサへ引き継ぐ）。
      pub(crate) fn from_graph(graph: &FusionGraph, root: FusionNodeId) -> FusionPlan;

      // 以下はシグネチャのみを確定するスケッチであり、本体の実装は #163
      // が担う（§2.1／§2.2／§3.4 冒頭の `FusionOp`／`FusionSession` の
      // シグネチャスケッチと同じ体裁。「アクセサをいつか追加する」という
      // 先送りではなく、外部 backend が呼べる関数シグネチャそのものを
      // 本文書で確定する）。
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

  - `FusionSession::materialize` は自身が借用する `self.ops`
    （`&'ops dyn BackendOps`。§3.4 冒頭）を使い、`self.graph` から
    `FusionPlan` を構築したうえで `self.ops.run_fused(&plan, leaves)`
    を試し、`BackendError::Unsupported` が返った場合は §4 の fail-safe
    方針に従い、グラフのノードを発生順に辿って既存の
    `add`／`mul`／`relu`／`exp`／`tanh` 呼び出しへ 1 段ずつ逐次
    フォールバックする（融合の有無に関わらず最終結果は同一の数値一致
    複合判定を満たす。§4）。この呼び出しは §3.2 (c) が指す「単一の
    fallible 呼び出しが自身の結果を返す直前」に、その呼び出しの関数
    フレーム内で完結する。
  - **`run_fused` の追加と「trait 定義自体には手を加えない」の関係
    （codex-review 第 5 波 P2 指摘への回答。本改訂で文言統一する）**:
    `run_fused` はデフォルト実装付きのメソッドとして `BackendOps` の
    trait 定義（`backend_ops.rs:82`）へ追加する。§3.4 冒頭「`BackendOps`
    trait 自体の契約も変更しない」・「ops 解決の所有モデル」節の
    「`BackendOps` trait 定義自体は変更しない」は、いずれも既存メソッド
    （`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）の
    シグネチャ・契約を変更しないこと、および `Send + Sync` を trait の
    スーパートレイトとして追加しないことを指す限定表現であり、
    「trait 定義へ一切変更を加えない」という意味ではない（本改訂で
    誤解の余地を解消する）。統一後の契約は次のとおり: **既存メソッドの
    契約（シグネチャ・意味論）は一切変更せず、`run_fused` をデフォルト
    実装付きで trait 定義へ追加する**。デフォルト実装により、既存の
    3 バックエンド実装（CPU／CUDA／Metal）は本節追加時点で override
    不要のままコンパイルが通り（trait の破壊的変更にならない）、
    `BackendOps` を実装する既存クレート（本リポ外の実装を含む）は
    変更不要である。`Send + Sync` は `run_fused` のシグネチャにも、
    `Tape::with_backend` の引数型にも課さない（§3.4 冒頭「`ops` の
    受け渡しは借用で足りる」で確定したとおり、`Storage::Pending` への
    埋め込みという前提自体が消滅したため、この束縛はもはや不要である）。
- **まとめ（codex-review 第 5 波 P1 指摘への回答。本改訂で確定）**:
  「遅延値を保持できる `Tensor` 表現への変更」は採らない（`Tensor`
  不変。`Storage` にも `Pending` バリアントを追加しない）。融合対象
  区間の構築・実体化はいずれも「単一の fallible 呼び出しの内部だけで
  生存するローカル値」（`FusionValue`／`FusionSession`）として行う。
  「連鎖全体を受け取る明示的な内部 API」として `BackendOps::run_fused`
  （非破壊拡張のデフォルトメソッド）を追加する。グラフの所有は
  `FusionSession` が `graph: FusionGraph`（所有値）として保持し、
  実体化に使う `BackendOps` 実装は `ops: &'ops dyn BackendOps`
  （借用）として保持する。`Arc`／`Mutex`／`Rc`／`RefCell`／
  `Send + Sync` はいずれも不要である（旧稿〈第 4 波まで〉はこれらを
  `Storage::Pending` として `Tensor` へ埋め込むために要求していたが、
  §1 でその前提自体を撤回したため、本改訂で全面的に単純化した）。
  `FusionSession` を開くのは `Tape::backward` の VJP 連鎖内部（§3.5）、
  または将来の複合エントリポイント（§1 (b)）であり、いずれも呼び出し
  元の関数フレーム内で `Tape` が保持する `ops: Option<Box<dyn
  BackendOps>>`（新規の公開コンストラクタ `Tape::with_backend(ops:
  Box<dyn BackendOps>)` による明示供給、または `Tape::new()` の内部
  既定解決のいずれか）を借用して使う。`Tape::new()` も同一の内部構造
  （`self.ops`）へ到達し、内部で既定バックエンド解決を試みる（§1
  「公開コンストラクタの選択は融合スイッチにならない」）。既定
  バックエンド解決の具体的な規則（`Device::Cpu` 固定を含む、いずれの
  規則も）は本文書では確定しない（`docs/public-api-design.md` §4.1 の
  既定デバイス選択ロジック不採用方針を踏襲。§6.2 未決事項）。この規則
  が確定するまでの間、`Tape::new()` 経由の既定バックエンド解決は常に
  失敗し、`self.ops` は `None` のまま融合を伴わない既存の非融合経路の
  みを通る（`Device::Cpu` への暗黙フォールバックは行わない）。これは
  「バックエンド解決可否」という内部方針の現時点の解決結果であり、
  `Tape::new()` という公開 API 選択自体が融合を禁じているのではない。
  外部 backend（`backend-cpu`／`backend-cuda`／`backend-metal`）が
  `run_fused` 内で融合グラフの演算内容を読み取る手段も本改訂で確定
  した: `FusionPlan` は `pub`（フィールド非公開）の不透明ハンドルとし、
  `impl FusionPlan` の `pub fn ops() -> impl Iterator<Item =
  FusedOpKind>`／`output_shape`／`dtype`／`leaf_count`／`use_count`
  （上記コード例）という公開 DTO アクセサ経由でのみ読み取らせる。
  内部の `pub(crate)` `FusionOp`／`FusionNode`／`FusionGraph`（§2）は
  非公開のまま変更しない。既存 `BackendOps` 呼び出し規約・`Tensor`
  表現とは非破壊に接続される。使用点の具体化は §3.5 で規定する。

### 3.5 単一 fallible 呼び出し内での融合適用と公開 `Var` 演算の常時実体化契約

（本節は codex-review 第 5 波 P1 指摘を受けて全面改訂する。第 4 波までの
旧稿は「複数回の独立した公開 `Var` 呼び出し（`a.add(&b)?.mul(&c)?` の
ような連鎖）をまたいで `Storage::Pending` を持ち越し、実体化失敗は
`cache` へキャッシュして後続の呼び出しで表面化させる」という設計を
採っていた。しかし `Var::value`／`Var::to_tensor` は呼び出し自体が
「結果を受け取る」という参照行為であるにもかかわらず、その時点でまだ
実体化を試みていない・失敗していても通知されないという契約破壊を生む
（§1「遅延の生存窓は単一の fallible 呼び出しの内部に限定する」で確定
した第 5 波 P1 指摘への回答）。本節は縮小後の契約を具体化する。)

### 3.5.1 公開 `Var` 演算の常時実体化契約

- **`Var` の fallible 演算（`add`／`mul`／`matmul`／`sum`／`max`。既に
  `Result<Var<'_>, AutodiffError>` を返す契約。`var.rs:111`〜`:159`）は、
  返る前に自分の出力を実体化済みにする**。`Tape::push` が記録する
  `nodes[id].value: Tensor<f32>`（`tape.rs`）は常に完成したデータを
  持ち、`Tensor` が保持する非公開の `Storage<T>`（`tensor.rs:33`）には
  「未実体化」を表すバリアントを一切追加しない（§3.4 で確定済み）。
  融合実行の失敗はその演算自身の `Err` として型付きで返る。
- **`relu`／`exp`／`tanh`（`var.rs:257`〜`:275`。shape 不変の単項演算の
  ため「構造的に失敗しえない」という既存設計判断〈`docs/public-api-design.md`
  §3.2〉により非 fallible な `fn ..(&self) -> Var<'t>` のまま。
  codex-review 第 5 波指摘への回答で本改訂時に事実確認した）は本節の
  対象外である**: `Err` を返す経路自体を持たないため、この 3 演算は
  内部で融合グラフの構築・実体化を一切試みず、既存の非 fallible な
  `eval::relu`／`exp`／`tanh` をそのまま直接呼ぶ（1 呼び出し 1 演算の
  粒度であり単独では融合機会がないという §3.3 の帰結とも整合する）。
  これらの unary 演算が VJP の内部計算式（3.5.2 の `tanh` 例のような
  `mul`／`sub` の連鎖）に現れる場合でも、失敗を運ぶのは `Var::relu`
  等自身ではなく、それらを呼び出す fallible な `grad.rs::vjp`（§3.5.2）
  である。
- この結果、`Var::value`／`Var::to_tensor`（`var.rs:74`・`var.rs:81`
  付近）・`Gradients::get`・`Tensor::get`／`as_slice`／`contiguous`
  （`tensor-core` の汎用アクセサ）はいずれも**シグネチャ・意味論を
  一切変更しない**。「未実体化」による分岐・`OnceLock` によるキャッシュ・
  実体化を発火させるための特別な内部アクセサ（第 4 波で導入した
  `Var::value_raw`）はいずれも不要であり、本改訂で新設しない（旧稿の
  該当記述は撤回する）。`get`／`as_slice` の既存契約「範囲外・非
  contiguous のみ `None`」もそのまま維持される（実体化に起因する
  `None` 分岐は存在しない）。
- `&dyn BackendOps` を直接呼ぶ既存経路（`ops_for` 経由を含む。§1・§3.4）
  は引き続き本設計の対象外であり、この経路の `Storage` は常に
  `Materialized` のまま（`ops_for(...).add()` 等の実装は融合グラフを
  一切構築しない）。

### 3.5.2 窓 (a): `Tape::backward` の VJP 連鎖内部

- `Tape::backward`（`backward.rs:73`。公開シグネチャ
  `pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`
  は変更しない）は、それ自体が単一の fallible 呼び出しである。内部で
  テープを逆順に辿り各ノードの VJP（`grad.rs::vjp`。`Op` 単位のまま。
  §3.3）を計算する過程で、1 つの VJP 計算式が複数の elementwise 演算
  から成る場合（例: `tanh` の VJP `grad * (1 - y * y)` は `mul`・`sub`
  の連鎖）、`self.ops`（`Tape` が保持する `Option<Box<dyn BackendOps>>`。
  §3.4）を借用して `FusionSession` を開き、その VJP 計算の内部だけで
  完結するグラフを構築・実体化してよい。これが REQ-12「透過的融合」の
  実質的な適用箇所である（§3.3 で確定した「forward 値計算の下層」の
  読み替え）。
- **`Gradients::get` は非 fallible のまま**: `backward` は自身が返す
  `Gradients` に含まれるすべての勾配 `Tensor` を、`Ok(Gradients { .. })`
  を返す直前までに実体化し終える（3.5.1 の契約を `backward` 自身の
  戻り値にも適用した帰結）。したがって `Gradients::get` は追加の
  実体化発火点を必要としない。
- **`BackendError` の発生源は `FusionSession::materialize` の直接呼び出し
  1 箇所のみである（codex-review 第 5 波指摘への回答。本改訂で確定
  する）**: `eval::dense_vec`（`eval.rs:41`。`grad.rs` が
  `use crate::eval::{.., dense_vec}` で再利用する。forward 記録値
  `nodes[id].value: Tensor<f32>` を稠密な `Vec<f32>` として読み出す
  既存ヘルパー）は 3.5.1 のとおり `Storage` が常に `Materialized` で
  ある以上、非 fallible なまま変更しない（`-> Vec<f32>`。既存の
  `ShapeError` を返さない契約〈`eval.rs` 冒頭コメント〉ともそのまま
  整合する。第 4 波までの旧稿はこれを `Tensor::try_dense`〈`pub` +
  `#[doc(hidden)]`〉として `Result` 化する設計を採っていたが、`Tensor`
  自体が実体化していない状態を持ちえない以上その `Result` に到達
  可能な `Err` 経路が存在しないことが第 5 波で判明したため、本改訂で
  撤回する）。`BackendError` が発生しうるのは、`grad.rs::vjp` および
  各演算 VJP 関数（`matmul_vjp` 等）が**自身の VJP 計算式の内部**
  （例: `tanh` の VJP `grad * (1 - y * y)`）で `FusionSession` を開き
  `FusionSession::materialize`（§3.4。1 回の呼び出しで `Result<Tensor<f32>,
  BackendError>` を返す）を呼ぶ場合のみである。この呼び出しを行う
  `vjp`／各演算 VJP 関数は `Result<_, BackendError>` を返すシグネチャ
  （いずれも `autodiff` クレート内で完結する `pub(crate)` 関数であり
  公開 API ではないため非破壊）へ変更し、`?` でそのまま伝播させる。
  失敗はキャッシュに留まらず直ちに呼び出し元（`backward`）へ伝播する
  （第 4 波までの旧稿にあった「`cache` へキャッシュされ後続の系統 2
  経由で表面化する」という間接的な伝播は本改訂で撤回する。伝播は
  常に同一呼び出し内の `?` による直接伝播である）。`Tape::backward`
  （公開シグネチャは
  `Result<Gradients, AutodiffError>` のまま変更しない）は、内部で
  `vjp(...)` を呼ぶ箇所においてのみ `BackendError` を `AutodiffError`
  へ変換する。変換は `AutodiffError`（`#[non_exhaustive]`。`error.rs:19`）
  への非破壊 variant 追加で行う:
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
  自動変換する。`error.rs:66` 以降の `impl fmt::Display for
  AutodiffError` は `match` で全 variant を網羅しているため、`Backend`
  variant 追加時は対応する `Display` アームの追加も同時に行う（追加を
  怠るとコンパイルエラーになる。実装時の見落とし防止のため本節に
  明記する）。
- `Tape` が記録する `Op` 単位のノード粒度・`grad.rs::vjp` の走査対象
  （`Op` 列）自体には影響しない（§3.3 の契約を変更しない）。本節が
  変更するのは `vjp`（`grad.rs:31`）とその内部の全 VJP 関数の
  **シグネチャ**（`Result<_, BackendError>` 化）のみであり、`Tape`／`Op`
  の**構造**（ノード粒度・走査順）には影響しない（#164 のスコープに
  明示的に含める）。

### 3.5.3 窓 (b): 将来の複合エントリポイント

- 現状の `Var` 演算 API は 1 呼び出し 1 演算の粒度であり（`add`／`mul`
  等それぞれが独立した `Result` を返す）、演算単体の内部に複数
  elementwise 演算を持たないため、この窓の恩恵は現時点では顕在化
  しない。将来、複数の演算を 1 回の `Result` で返す複合エントリ
  ポイント（`compat::Sequential::forward` 相当。`docs/public-api-design.md`
  に設計段階として記載。または将来の「グラフ一括実行」API）が追加
  された場合、その内部実装は 3.5.2 と同じ要領で `FusionSession` を
  開き、自身が `Ok` を返す直前に実体化を完了させる契約に従う。この
  窓は #164 の必須スコープではなく、当該複合 API が実装される時点で
  適用される（本節はその際に従うべき契約を先に確定しておくもの）。

### 3.5.4 窓 (c): `FusionSession::materialize` — 融合実行のフォールブルな単一発火点

（本節は codex-review 第 5 波指摘を受けて全面改訂する。第 4 波までの
旧稿は `tensor-core` 側に `Tensor::try_dense(&self) -> Result<Vec<T>,
BackendError>` という新設アクセサを置き、そこへ `BackendError` を
集約する設計を採っていた。しかし 3.5.1 の契約により `Tensor` の
`Storage` は常に `Materialized` であり、`Pending` は `FusionSession`
の外へ一切出ない（§3.4）。したがって `Tensor` の `&self` メソッドが
`FusionSession` へ到達する経路自体が存在せず、`try_dense` の `Result`
に到達可能な `Err` 経路がない（第 5 波で判明）。本節は撤回し、
「窓 (c)」の実体を §3.4 で既に定義した `FusionSession::materialize`
そのものへ置き換える。）

- 3.5.2・3.5.3 の内部実装（VJP 計算・将来の複合エントリポイント）が
  「これから構築する融合対象区間の結果を確定させる」ために呼ぶ唯一の
  フォールブルな発火点は `FusionSession::materialize`（§3.4。
  `pub(crate) fn materialize(&self, value: FusionValue) -> Result<Tensor<f32>,
  BackendError>`）である。この関数は `FusionValue::Materialized` を
  そのまま返し、`FusionValue::Pending` は `run_fused` を試みたうえで
  §4 の fail-safe（逐次フォールバック）に従う（§3.4 で確定済み）。
- 一方、`Tensor`（既に実体化済み。3.5.1）の値を稠密な `Vec<f32>` として
  読み出す処理（VJP 計算式が入力に使う既存の forward 記録値の稠密化）
  は、既存の `eval::dense_vec`（`eval.rs:41`）がそのまま担う。3.5.2で
  確定したとおり `dense_vec` 自身は非 fallible のまま変更しない
  （`Tensor::contiguous()`／`as_slice()` の既存の非 fallible 契約の
  延長であり、この処理自体は失敗しない）。
- **失敗経路**: `run_fused`（`BackendOps` の非破壊拡張メソッド）は
  `Unsupported` 以外にも、GPU 側カーネル実行失敗
  （`KernelLaunchFailed`）・NVRTC／MSL コンパイル失敗・デバイス側
  障害（`DeviceAllocationFailed`・`TransferFailed`）等、実行時に
  実際に起こりうる理由で `Err(BackendError)` を返しうる
  （`device.rs:184` 以降の `BackendError` variant 一覧のとおり）。
  `FusionSession::materialize` を呼ぶ `vjp`／各演算 VJP 関数（3.5.2）は
  これらすべてを型付きエラーとして `?` でそのまま伝播させ、
  `debug_assert!` 等で握り潰さない。

### 3.5.5 view 系操作（transpose・narrow・reshape）

- `offset`／`shape`／`strides` のみを扱う view 系操作
  （`Arc::clone(&self.storage)` 経路）は、3.5.1 の契約により
  `Storage` が常に `Materialized` であるため、他の `tensor-core` の
  既存 view 演算と同様に振る舞う。「未実体化のまま view を複製する」
  という旧稿（第 4 波まで）の複雑さは、`Storage::Pending` 自体を廃止
  したことで解消される（本改訂の帰結）。
- 3.5.2・3.5.3 の内部実装が構築するグラフ（`FusionSession` が保持する
  ローカルな `FusionGraph`）の内部では、transpose を挟む部分列は
  §1・§2.3 のとおり非融合フォールバックへ倒す（`NodeMeta.contiguous
  == false` が §3.2 (e) の実体化条件に対応する）。これは内部の
  グラフ構築ロジックの挙動であり、公開 `Tensor` の view 契約には
  一切影響しない。

### 3.5.6 §1 の PoC-9 実測に対する受け入れコスト（再掲）

- §1「遅延の生存窓は単一の fallible 呼び出しの内部に限定する」で
  確定したとおり、PoC-9 実測（`ew4`／`ew6`）が示す 2.25〜3.19 倍の
  高速化は、独立した公開 `Var` 呼び出し（`a.add(&b)?.mul(&c)?`）を
  またぐ elementwise 連鎖には及ばない。この対価は §6.2 に明示的な
  未決事項として記録する（本節の重複記載は避け、§1・§6.2 を参照する）。

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
| #164（ディスパッチ統合） | §1 の「利用者向け制御 API を提供しない」方針・「公開コンストラクタの選択は融合スイッチにならない」契約・「遅延の生存窓は単一の fallible 呼び出しの内部に限定する」契約（codex-review 第 5 波 P1 指摘への回答）に基づく融合対応経路の実装。§3.4 で確定した `FusionValue`／`FusionSession`（借用ベース・`Arc`／`Mutex`／`Send + Sync` 不要）／`BackendOps::run_fused`（デフォルト実装付きで trait 定義へ追加。既存メソッドの契約は変更しない。第 5 波 P2 指摘への回答）接続契約、`Tape` の非公開フィールド `ops: Option<Box<dyn BackendOps>>` と新規公開コンストラクタ `Tape::with_backend(ops: Box<dyn BackendOps>)` の追加（＝ TASK-1.9 の backend 経由実行への置き換えと同時実施）。§3.5.1 で確定した「`Var` の fallible 演算（`add`／`mul`／`matmul`／`sum`／`max`）は返る前に自身の出力を実体化済みにする」契約の実装（`relu`／`exp`／`tanh` は非 fallible なまま対象外。`Storage` に `Pending` バリアントは追加しない。`Var::value_raw`・実体化の「系統 1〜3」分岐はいずれも新設しない）。§3.5.2 で確定した `Tape::backward` の VJP 連鎖内部での `FusionSession` 利用・`grad.rs::vjp` とその内部の各演算 VJP 関数の `Result<_, BackendError>` 化（`FusionSession::materialize` の直接呼び出しのみが `BackendError` を発生させる。§3.5.4）・`AutodiffError::Backend(BackendError)` variant と `From<BackendError>` 実装の追加（`Display` アーム追加を含む）。既存の `eval::dense_vec`（forward 記録値の稠密化）は非 fallible のまま変更しない（§3.5.2・§3.5.4） |
| #165（テスト） | §1・§2.3 の transpose 非融合フォールバック、§2.4 の fan-out 融合、§3.3 の autodiff 契約（VJP がノード単位のまま変わらないこと）の検証、**§1「公開コンストラクタの選択は融合スイッチにならない」契約の検証**（同一演算列を `Tape::new()` と `Tape::with_backend(ops)` の双方で実行し、数値結果が数値一致複合判定〈§4〉を満たすこと、および融合の発生有無がどちらのコンストラクタを呼んだかではなく §1 の 2 条件〈バックエンド解決可否・演算列の融合可否判定〉のみで決まることの検証）、**§3.5.1「公開 `Var` 演算の常時実体化契約」の検証**（codex-review 第 5 波 P1 指摘への回答）: (i) 独立した公開 `Var` 呼び出し（`a.add(&b)?.mul(&c)?`）をまたいで `Pending` な状態が一切観測されないこと（各呼び出しの戻り値の `Storage` が常に `Materialized` であることをテスト用アクセサで確認する、または `run_fused` の呼び出しタイミングが各 `Var` 呼び出しの `?` 到達前に必ず完了していることをカウンタ付き `BackendOps` テスト実装で確認する）、(ii) 融合実行の失敗（`run_fused` がテスト用に `Unsupported` 以外の `Err` を返す実装）が、それを引き起こした公開 `Var` 演算自身の `Err` として直接返ること（キャッシュ経由の遅延表面化が発生しないこと）、(iii) `Tape::backward` の VJP 連鎖内部で融合が発生する場合（§3.5.2）に、その融合が失敗すると `Tape::backward` が `AutodiffError::Backend` を返すこと、かつ成功時は `Gradients::get` がそのまま非 fallible に値を返せること、(iv) §2.4 の fan-out が単一呼び出し内の融合グラフ構築で正しく解決されることの検証 |
| #203（GEMM epilogue 融合） | §3.2 (b) の `gemm` 境界を bias／activation epilogue まで拡張する設計変更 |

### 6.2 未決事項（スコープ外）

- **`Tape::new()` が使う既定バックエンド解決の具体的な規則**: §1「公開
  コンストラクタの選択は融合スイッチにならない」・§3.4「`Tape::new()`」
  は `Tape::new()` も内部で既定バックエンド解決を試みたうえで
  `with_backend` と同一の内部ディスパッチへ到達する契約を確定したが、
  **どの `Device` をどう既定選択するか（`Device::Cpu` 固定を含むいかなる
  具体規則も）は本文書では確定しない**。理由は 2 点: (i)
  `docs/public-api-design.md` §4.1「既定デバイス選択ロジックは…実装
  しない（列挙と明示選択のみを提供する。ユーザー承認が必要な事項のため
  自動運転では安全側に倒した）」という本リポ全体の確立済み方針、(ii)
  REQ-2 が「CUDA 既定有効化の構成決定」を未検証のまま残している
  （`docs/spec/04-requirements.md` REQ-2 受け入れ基準）ため、既定解決
  規則の確定にはユーザー承認が必要。この規則が確定するまでの間、
  `Tape::new()` 経由の既定バックエンド解決は常に失敗し（`ops` は
  `None` のまま）、実質的に現行の非融合パスと同じ挙動になる（§3.4・
  §3.5 に明記）。既定解決規則の確定は #164 以降、ユーザー承認を得た
  うえで別途検討する。
- **独立した公開 `Var` 呼び出しをまたぐ elementwise 連鎖融合を行わない
  受け入れコスト（codex-review 第 5 波 P1 指摘への回答。本改訂で
  §1「遅延の生存窓は単一の fallible 呼び出しの内部に限定する」を確定
  した帰結として記録する）**: PoC-9 実測（`ew4`／`ew6`）が示す
  2.25〜3.19 倍の高速化は、`a.add(&b)?.mul(&c)?` のように独立した公開
  `Var` 呼び出しをまたぐ elementwise 連鎖で測定されたものである。第 4
  波までの設計はこの連鎖をまたいで `Storage::Pending` を持ち越すことで
  この高速化を再現しようとしていたが、`value`／`to_tensor` の非
  fallible 契約を壊さずには実現できないことが第 5 波 P1 指摘で判明した
  （§1）。本改訂後の契約（§3.5.1「公開 `Var` 演算は返る前に自身の出力を
  実体化済みにする」）のもとでは、現状の `Var` 演算 API（1 呼び出し
  1 演算の粒度）を素朴に使う限り、この高速化は得られない。融合の効果
  は §1 (a)〜(c) の窓（`Tape::backward` の VJP 連鎖内部・将来の複合
  エントリポイント・`FusionSession::materialize` の直接呼び出し）に
  限定される。加えて、実現可能な窓のうち (a) の VJP 計算式は `tanh`
  の `grad * (1 - y * y)` のように 2〜3 段程度の短い連鎖にとどまり、
  §1・§3.2 (d) が定める 4〜6 段の連鎖長上限に到達する例は現時点では
  想定しにくい（複合エントリポイント（b）が追加されれば 4〜6 段規模の
  連鎖が実現しうる）。これは
  transpose 非融合（13.89 倍差、上記エントリ）と同種の「正当性・契約
  健全性を優先した安全側の設計判断による受け入れコスト」であり、
  #162 以降で複合エントリポイント（§3.5.3「窓 (b)」）を追加すること
  により部分的に回復しうる拡張候補として記録する。
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
- **異なる `FusionSession` を跨ぐ融合境界の性能特性（本改訂で解消。
  旧稿の記録を撤回する）**: PR #357 review 再指摘 P1-1／P1-2 が対象と
  していた「異なる `FusionSession` に属する `Pending` 同士が二項演算で
  合流する」というシナリオ（`(a + b) * (c + d)` が独立した 2 つの
  セッションをまたぐケース）は、§1「遅延の生存窓は単一の fallible
  呼び出しの内部に限定する」の確定により構造的に発生しなくなった:
  `Pending`（`FusionValue::Pending`）は特定の 1 回の fallible 呼び出し
  （§3.5.2・§3.5.3）の実行中にのみ存在し、その呼び出しが返る前に必ず
  実体化されるため、呼び出しをまたいで生き残る `FusionSession` 自体が
  存在しない。したがって「複数の `FusionSession` が合流する」という
  状況そのものが起こりえず、本エントリが記録していた性能上限の懸念は
  対象を失った（旧稿の議論は削除する）。
- **葉ノードを `DeviceBuffer` 直接参照へ最適化する場合の device 一致
  検証**: §3.4・§3.5.2〜3.5.4 の設計は、融合対象区間の葉が常に host
  常駐の `Tensor<f32>` を経由する契約に依拠して backend 越境の安全性を
  保っている。将来 `run_fused` の葉を `DeviceBuffer`（§4.2。デバイス
  メモリハンドルを直接保持）へ最適化し host 往復を省く設計に変更する
  場合、この前提が崩れるため、その時点で葉ごとに `ops.device()` と
  `DeviceBuffer::device()` の一致を fail-closed で検証する（不一致は
  型付きエラーで拒否する）契約を新設する必要がある。現行スコープ
  （TASK-12.1a）では `Tensor<f32>` 経由の host 往復のみを対象とする
  ためこの検証は不要であり、#162 以降の拡張候補として記録する。
