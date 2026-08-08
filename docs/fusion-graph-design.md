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
    `ops: Option<Box<dyn BackendOps>>`（§3.4）へ到達し、§3.5 が規定する
    実体化の発火点（層 1〈fallible 境界。§3.5.2〉・層 2〈非 fallible
    境界。§3.5.3〉・将来の複合エントリポイント〈§3.5.5〉）が
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
  - **演算跨ぎの遅延を復活し、実体化を 3 層の境界で規定する
    （codex-review 第 6 波 P1 指摘への回答。第 5 波で確定した「遅延の
    生存窓は単一の fallible 呼び出しの内部に限定する」契約を撤回し
    本改訂で置き換える。詳細は §3.2・§3.4・§3.5）**: 第 5 波は
    `value`／`to_tensor` の非 fallible 契約を守るため、遅延
    （`Pending`）の生存窓を 1 回の fallible 呼び出しの内部だけへ縮小
    した。しかしこの縮小は同時に、TASK-12.1 の中核要件である現行公開
    API（`a.add(&b)?.relu().exp().tanh()` のように独立した複数回の
    公開 `Var` 呼び出しをまたぐ連鎖）上での 4〜6 段 elementwise 連鎖の
    融合を不可能にした（本改訂が解く codex-review 第 6 波 P1 指摘、
    §3.3 の当該記述の訂正は §3.3 で行う）。本改訂は「複数回の公開
    `Var` 呼び出しをまたいで遅延を持ち越す」設計を復活させたうえで、
    `value`／`to_tensor` の非 fallible 契約を壊さないよう実体化の
    発火点を次の 3 層で規定する（型・実装の詳細は §3.5）:
    1. **fallible 境界**: `Var::add`／`mul`／`matmul`／`sum`／`max`
       （既に `Result<Var<'_>, AutodiffError>` を返す契約）が自身の
       計算のために入力側の未実体化値を必要とする場合、`Tape::backward`
       の VJP 連鎖内部、および §3.2 (d) の連鎖長上限に fallible な
       演算の呼び出し中に到達した場合。融合実行を試み、**失敗は型付き
       エラー（`AutodiffError::Backend(BackendError)`。§3.5.2 で確定
       済みの variant をそのまま再利用する）として `?` で呼び出し元へ
       伝播する**。これが失敗の主経路であり、CPU フォールバックは
       行わない（バックエンド実行失敗は利用者が結果を受け取るより前に
       必ず型付きで観測されるべきという第 3〜5 波の契約をそのまま
       踏襲する）。
    2. **非 fallible 境界**: `Var::value`／`Var::to_tensor`
       （`-> Ref<'_, Tensor<f32>>`／`-> Tensor<f32>` の既存シグネチャを
       一切変更しない）・`Gradients::get`、および §3.2 (d) の連鎖長
       上限に非 fallible な演算（`relu`／`exp`／`tanh`）の呼び出し中に
       到達した場合。融合実行を試み、**失敗した場合は記録済みの演算列
       を CPU 参照実装（`eval::relu`／`exp`／`tanh` 等の既存非
       fallible 経路。§3.5.4 で確定するとおり構造的に失敗しない純
       `Vec<f32>` 演算）で逐次 eager 再実行するフォールバックにより
       必ず正しい値を返す**。誤った値・欠落値・`panic!` のいずれも
       発生しないため、第 4〜5 波で確定した契約 4（`get`／`as_slice`／
       `value`／`to_tensor` の非 fallible 契約が観測可能な意味論も
       含め完全不変）・契約 5（実体化失敗は必ず型付きで通知されるか、
       利用者に誤った値・欠落値を渡さない）を同時に満たす。フォール
       バックの発生は内部で観測可能にする（テスト用カウンタ。§6.1
       #165 に記録）。
    3. **`relu`／`exp`／`tanh`（`var.rs:257` 以降。shape 不変の単項
       演算のため「構造的に失敗しえない」という既存設計判断
       〈`docs/public-api-design.md` §3.2〉により非 fallible な
       `fn ..(&self) -> Var<'t>` のまま）は自身の出力を実体化しない**
       まま返す（＝連鎖を延長する）ことを許容する。これが本改訂で
       4〜6 段連鎖を実現する主要因である（`relu`／`exp`／`tanh` は
       任意回・任意順に連結できるため、この 3 演算だけで §3.2 (d) の
       上限まで連鎖を伸ばせる。§3.5.1・§3.5.3 で確定する）。`add`／
       `mul`／`matmul`／`sum`／`max` は引き続き**返る前に自分の出力を
       実体化済みにする**（第 5 波の契約を維持。ただし入力側が
       `relu`／`exp`／`tanh` の遅延連鎖であった場合、その入力の実体化は
       上記層 1 の fallible 境界として扱う）。融合の効果はこの単項
       演算の遅延連鎖（層 1・層 2 のいずれかで実体化される）の内側で
       得られる。
    - この結果、利用者が保持する `Var`／公開 `Tensor` の「実体化済み
      かどうか」という状態自体は `autodiff::TapeNode`（`tape.rs`。
      `pub(crate)` 非公開実装）にのみ存在し、`Tensor`／`Storage` へは
      一切漏れ出さない（`tensor.rs:33` の `Storage<T>` は本改訂でも
      変更しない。§3.4・§3.5.1 参照）。第 1〜4 波で確定した契約
      （view 適用・融合スイッチ非提供・`Option` へのエラー非流入・
      公開 `Tensor` 常時実体化）は「`Tensor` はそもそも遅延状態を
      持てない」という構造そのものにより自動的に成立する（矛盾する
      記述の整理は §3.2・§3.4・§3.5・§6.1 で横断的に行う）。
    - **比較検討 1: 「`Result` を返す読み出し API を追加し、遅延値は
      必ずそこから取得させる」案（不採用。第 5 波比較検討を踏襲）**:
      `Var::value`／`Var::to_tensor` の非 fallible 契約はそのまま残し、
      代わりに `Var::try_value`／`Tensor::try_get` 相当の `Result`
      返却アクセサを新設し、遅延値の実体化失敗はそちらからのみ観測
      させる案。この案は「まだ誰も実体化を試みていない」という状態が
      存在し続けること自体は許容するため、利用者が非 fallible な
      `value`／`to_tensor` を呼んだ場合には引き続き失敗が通知されない
      契約破壊が残る（新設した `Result` 版を呼ばない限り安全側に
      ならない、オプトインの回避策にすぎない）。加えて、互換 API 層
      （REQ-9）が前提とする「自作コアの上の薄いラッパーに徹する」
      方針に対し、遅延値専用の新しい公開アクセサ系列を追加することは
      公開 API 面を不必要に広げる。本改訂が採用する CPU フォール
      バック案は公開シグネチャを一切追加せず、かつ「呼び出し自体が
      結果参照である」という `value`／`to_tensor` の意味論を字義
      どおり満たせるため、引き続きこちらを不採用とする。
    - **比較検討 2: 非 fallible 境界で実体化に失敗した場合は
      `panic!` する案（不採用）**: `run_fused` の失敗（`Unsupported`
      以外の `KernelLaunchFailed` 等、実行時に実際に起こりうる理由）を
      `value`／`to_tensor` 内で検知した際に、`Result` を新設せず
      `panic!`／`unwrap()` で停止させる案も比較対象とした。この案は
      公開シグネチャを変更しない点は CPU フォールバック案と同じだが、
      (i) 契約 5「実体化失敗は必ず型付きで通知されるか、利用者に誤った
      値・欠落値を渡さない」の後半しか満たさず、利用者が正常応答を
      期待する非 fallible API の内部で任意のバックエンド障害
      （GPU メモリ確保失敗等、利用者の入力とは無関係な一時的環境要因）
      がプロセス停止に直結するため可用性上望ましくない、(ii) 本番経路
      で `panic!`／`unwrap()`／`expect()` を使わない方針
      （`.claude/rules/coding-rust.md`「コード品質」）に反する。CPU
      フォールバック案は同じ失敗を検知したうえで既存の eager 経路
      （融合前から動作している `eval::*`）へ落とすだけであり、
      追加の失敗モードを生まないため本改訂ではこちらを採用しない
      （不採用の理由の記録として残す）。
    - **比較検討 3: 「`Storage::Pending` を含む第 4 波までの設計を
      そのまま復活させる」案（不採用）**: 第 4 波は遅延状態を
      `Tensor` の非公開 `Storage<T>`（`tensor.rs:33`）へ埋め込んで
      いたため、`Tensor` が `Arc` 経由で複数箇所から共有されうる以上
      `Arc<Mutex<_>>`・`Send + Sync` 境界を要求し（§3.4 第 5 波での
      撤回理由）、かつ実体化失敗を `cache` へキャッシュして後続の
      呼び出しで間接的に表面化させる設計（第 5 波 P1 指摘が契約破壊と
      認定した設計）だった。本改訂はこの 2 点を復活させない: 遅延
      状態は `Tensor`／`Storage` ではなく `autodiff::TapeNode`
      （`tape.rs`。`autodiff` クレート内 `pub(crate)`）にのみ持たせる
      （§3.5.1）ため `Tensor` は変更されず `Arc<Mutex<_>>` も不要で
      あり、実体化失敗は間接キャッシュではなく発火点（層 1 の `?`
      直接伝播、層 2 の CPU フォールバック）でその場に処理する。
      「`Storage::Pending` を復活させる」のではなく「`TapeNode` の
      値スロットを遅延可能にする」点が第 4 波との相違であり、第 4 波
      の問題点（`Tensor` への漏出・間接キャッシュ）は再導入しない。
    - この対価として、PoC-9 実測（`ew4`／`ew6`）が示す 2.25〜3.19 倍の
      高速化は、`add`／`mul` 等の fallible 演算どうしが直接連続する
      区間（両者とも即座に実体化されるため）には及ばない。§6.2 の
      該当エントリはこの範囲に限定して受け入れコストを記録する
      （撤回・再記録の詳細は §6.2 参照）。
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
| (c) | `Var::value`／`Var::to_tensor`／`Gradients::get`（非 fallible 境界）、または fallible な `Var` 演算・`Tape::backward` の VJP 連鎖内部が入力側の未実体化値を必要とした時点（fallible 境界）。いずれも `autodiff` 側の materialize ヘルパー（§3.5.1〜3.5.3。`FusionPlan::from_ops` を経由して `BackendOps::run_fused` を直接呼ぶ。`tensor-core` 側の `FusionSession::materialize`〈§3.4〉と同じ「融合を試み、`Unsupported` は fail-safe で処理する」方針を、`autodiff` クレート内で完結する形で実装したもの）に帰着する（codex-review 第 6 波 P1 指摘への回答。§1「演算跨ぎの遅延を復活し、実体化を 3 層の境界で規定する」） | `Tensor`（`tensor.rs:53`）自体は `Arc<Storage<T>>` を必須で保持する既存表現のまま変更せず、`Storage<T>`（非公開）も `Pending` バリアントを持たない（§3.5.1 で確定）。したがって `Tensor::get`／`as_slice`／`contiguous`（`tensor-core` の汎用アクセサ）にも「未実体化」を表す分岐は存在しない。遅延状態は `autodiff::TapeNode`（`tape.rs`）だけが持つ（§3.5.1）。fallible 境界での実体化失敗は型付きエラーとして `?` で伝播し（層 1）、非 fallible 境界での実体化失敗は CPU 参照実装による eager 再実行で必ず正しい値を返す（層 2。§3.5.4） |
| (d) | 連鎖長上限（4〜6 段）到達 | TASK-12.1 の内容規定（4〜6 段程度）。PoC-9 の代表ワークロード規模（`ew4`／`ew6`）とも整合する上限であり、無制限連鎖によるカーネル生成コスト・レジスタ圧の増大を避ける。上限に到達させた演算が fallible か非 fallible かにより (c) の層 1／層 2 いずれかへ合流する（§3.5.3） |
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
- **実質的な適用箇所（codex-review 第 6 波 P1 指摘を受け本節を訂正する。
  §1 で確定した「演算跨ぎの遅延を復活し、実体化を 3 層の境界で規定
  する」契約に一致させる）**: 個々の `Var` fallible 演算メソッド
  （`add`／`mul`／`matmul`／`sum`／`max`）自体は 1 呼び出し 1 演算の
  粒度のままだが、**`relu`／`exp`／`tanh`（非 fallible な単項演算）は
  複数回の独立した公開 `Var` 呼び出しをまたいで遅延（`Pending`）を
  持ち越せる**（§3.5.1）ため、`a.add(&b)?.relu().exp().tanh()` の
  ような現行公開 API の記述形そのものが 4〜6 段の elementwise 連鎖を
  形成しうる。透過的融合の実質的な適用箇所は次の 3 箇所である:
  (i) `relu`／`exp`／`tanh` が積み上げた遅延連鎖を、後続の fallible な
  `Var` 演算が入力として読み出す時点（窓 (a)・層 1。§3.5.2）、
  (ii) `Tape::backward` の VJP 連鎖内部（窓 (a)・層 1。§3.5.2）、
  (iii) `Var::value`／`Var::to_tensor`／`Gradients::get` が同じ遅延
  連鎖を直接読み出す時点（窓 (a)・層 2、非 fallible 境界。§3.5.3）。
  将来追加されうる複合エントリポイント（窓 (b)。§3.5.5）は現時点では
  必須スコープ外のまま据え置く。

### 3.4 遅延グラフと `BackendOps`・`Tensor` 契約の接続

（PR #357 review 指摘への対応で追加。codex-review 第 5 波 P1 指摘を
受け、本節は §1「遅延の生存窓は単一の fallible 呼び出しの内部に限定
する」という縮小後の契約に合わせて全面改訂し、さらに第 6 波 P1 指摘を
受け、本節は「複数の公開 `Var` 呼び出しをまたいで遅延を持ち越す」設計
（§1・§3.5）が使う実体化契約として再改訂する。§1・§3.1〜3.3 は
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
+ Sync>` 所有モデルを要求していた。本改訂（第 6 波）は「複数回の公開
呼び出しをまたぐ遅延の持ち越し」自体は復活させるが、持ち越す場所を
`Tensor`／`Storage` ではなく `autodiff::TapeNode`（§3.5.1）に限定する
ため、`Arc<Mutex<_>>`・`Send + Sync` を要求した第 4 波の前提（遅延が
`Tensor` を経由して外部へ漏れ出すこと）は再導入しない。以下は
その帰結として単純化されたままの契約である。

- **`FusionPlan` は `tensor-core` と `autodiff` の双方から構築される
  一方、`FusionSession` 自体は `tensor-core` 内限定のまま変わらない
  （第 6 波での訂正）**: §3.5.2・§3.5.3 が示すとおり、実体化の発火点
  （`Var` の fallible 演算・`Var::value`／`to_tensor`・`Tape::backward`）
  はいずれも `autodiff` クレート側のコードである。`FusionOp`／
  `FusionNode`／`FusionGraph`（§2）・`FusionSession`（下記）が
  `pub(crate)`（`tensor-core` 内限定）のままでは、別クレートである
  `autodiff` から構築・呼び出しができない（Rust の可視性は依存関係の
  向きではなくクレート境界そのもので決まる）。`autodiff` は
  `FusionSession` を経由せず、`FusionPlan::from_ops`（新設）で直接
  `FusionPlan` を組み立てたうえで `BackendOps::run_fused`（既に
  `pub` トレイトメソッド）を直接呼ぶ（§3.5.1〜3.5.3 の materialize
  ヘルパーが行う手順）。`FusionSession` は `tensor-core` 内で
  `FusionGraph` が既に存在する場合（#162 の連鎖検出アルゴリズムが
  `tensor-core` 内で完結して使う将来のユースケースに備える）の
  ための内部機構として残す。この接続のために、`FusionPlan` の構築
  経路を 2 系統に分ける:
  1. `FusionPlan::from_graph`（下記。`pub(crate)`、`tensor-core` 内
     限定）: `tensor-core` 内で `FusionGraph` から構築する経路（#162
     の連鎖検出アルゴリズムが `tensor-core` 内で完結して使う場合に
     備えて残す）。
  2. `FusionPlan::from_ops`（新設。`pub`、`#[doc(hidden)]`）: 既に
     `pub` な DTO（`FusedOpKind`／`DType`／`FusedNodeIndex`。下記
     `impl FusionPlan`）だけを引数に取り、`tensor-core` 内部の
     `pub(crate)` 型（`FusionGraph`／`FusionNode`／`FusionOp`）を
     一切経由せずに `FusionPlan` を組み立てる。`autodiff` は自身が
     保持する `TapeNode`／`Op`（§3.5.1）の遅延連鎖を `FusedOpKind` の
     列へ直接変換し（`Op::Relu`/`Add`/... と `FusedOpKind::Relu`/
     `Add`/... は既に 1:1 対応、§2.1・§3.4 下記）、`from_ops` で
     `FusionPlan` を構築したうえで `BackendOps::run_fused`（既に
     `pub` トレイトメソッド、下記）を直接呼ぶ。`#[doc(hidden)]` を
     付す理由: この経路は `autodiff` という単一の内部利用者のための
     クレート間契約であり、利用者向けの融合制御 API ではない
     （REQ-12「利用者が明示的に融合を制御する API は提供しないこと」
     への抵触を避けるため、`pub` API ドキュメントには現れない内部
     専用シグネチャとして扱う。第 4 波で `Tensor::try_dense` に
     適用した `pub` + `#[doc(hidden)]` パターン〈第 5 波で撤回済み〉
     と同型の解決策を、可視性制約が実在するこの箇所にのみ限定して
     再適用する）。`FusionGraph`／`FusionNode`／`FusionOp` 自体は
     `pub(crate)` のまま変更しない（§2.5 の設計判断を維持）。

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
    指摘を受けた単純化を、第 6 波 P1 指摘を受け §3.5 の 3 層すべてへ
    一般化する）**: `FusionSession` を開くのは §3.5.2 の層 1（後続の
    fallible `Var` 演算・`Tape::backward` の VJP 連鎖内部）・§3.5.3 の
    層 2（`Var::value`／`Var::to_tensor`／`Gradients::get`）・§3.5.4
    （連鎖長上限到達時）、または将来の複合エントリポイント（§3.5.5）
    であり、いずれも `Tape` が既に保持する `BackendOps` 実装を使う。
    `Tape` は非公開フィールドとして
    `ops: Option<Box<dyn BackendOps>>` を保持する（`None` はバックエンド
    解決が不能だった場合、`Some` は解決に成功した場合を表す。フィールド
    追加は `Tape` の構造体を非公開のまま拡張するだけであり、`pub`
    フィールドを持たない現行の `Tape`（`tape.rs:140`）の公開契約を
    破らない）。旧稿（第 4 波まで）は `Storage::Pending` へ埋め込むために
    `Arc<dyn BackendOps + Send + Sync>` の所有値・`ops_for_arc`（`ops_for`
    の `Arc` 版姉妹関数）を新設していたが、この前提が消滅したため、
    本改訂で `Box<dyn BackendOps>`（`Send + Sync` 不要・
    `ops_for_arc` 新設も不要）へ単純化したまま維持する。`Tape::backward`
    （下記）・§3.5.1 の materialize ヘルパーはいずれも
    `self.ops.as_deref()`（`Option<&dyn BackendOps>`）を、融合を行う
    内部ヘルパーへ**借用として**渡す（§3.4 冒頭「`FusionSession`
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

      /// `autodiff` クレート専用の構築経路（新設。codex-review 第 6 波
      /// P1 指摘への回答。§3.4 冒頭「`FusionSession`／`FusionPlan` は
      /// `tensor-core` と `autodiff` の双方から使われる」で確定した
      /// 可視性上の必要から追加する）。`tensor-core` 内部の
      /// `pub(crate)` 型（`FusionGraph`／`FusionNode`／`FusionOp`）を
      /// 一切経由せず、既に `pub` な DTO のみから直接構築する。
      /// `autodiff` 側は自身の `TapeNode`／`Op` の遅延連鎖（§3.5.1）を
      /// この `ops` へ変換して渡す（`Op::Relu`/`Add`/`Mul`/`Exp`/
      /// `Tanh` と `FusedOpKind` の対応は §2.1 のとおり 1:1）。
      /// `#[doc(hidden)]` を付し、利用者向け公開 API のドキュメントには
      /// 現れないクレート間内部契約として扱う（REQ-12「利用者が明示的
      /// に融合を制御する API は提供しない」への抵触を避ける）。
      /// 実装は #164 のスコープ。引数の整合性（`ops` が参照する
      /// `FusedNodeIndex` が範囲内であること・葉ノード数と
      /// `leaf_count` の整合）は §4「グラフ構築 API はテンソル
      /// shape／stride の検証を先行させる」と同型の検証を行い、
      /// 不整合は `ShapeError` 相当として扱う（呼び出し元の `autodiff`
      /// はこの検証済みの `ops` しか渡さないため、実運用では到達
      /// しない防御的検証と位置付ける）。
      #[doc(hidden)]
      pub fn from_ops(
          ops: Vec<FusedOpKind>,
          output_shape: Vec<usize>,
          dtype: DType,
          leaf_count: usize,
      ) -> FusionPlan;

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
  `autodiff` 側の materialize ヘルパー（`FusionSession` は経由せず
  `FusionPlan::from_ops` + `BackendOps::run_fused` を直接呼ぶ。上記
  「`FusionPlan` は `tensor-core` と `autodiff` の双方から構築される」
  参照）を呼ぶのは（codex-review 第 6 波 P1 指摘を受け本節を訂正する）
  §3.5.2 の層 1（後続の fallible `Var` 演算・`Tape::backward` の VJP
  連鎖内部）・§3.5.3 の層 2（`Var::value`／`Var::to_tensor`／
  `Gradients::get`）・§3.5.4（連鎖長上限到達時。fallible／非 fallible
  いずれの経路にも合流する）、または将来の複合エントリポイント
  （§3.5.5）であり、いずれも呼び出し元の関数フレーム内で `Tape` が
  保持する `ops: Option<Box<dyn BackendOps>>`（新規の公開コンストラクタ
  `Tape::with_backend(ops: Box<dyn BackendOps>)` による明示供給、または
  `Tape::new()` の内部既定解決のいずれか）を借用して使う。この呼び出しは
  「呼び出し元の関数フレーム内だけで完結する」という `FusionSession`
  と同じ性質（上記コード例のドキュメンテーションコメント）を保つ
  （変わるのは「フレーム」の粒度が単一の `Var` 演算呼び出しに限られ
  なくなったことではなく、実体化の発火点が §3.5.1〜3.5.4 の複数箇所に
  増えたことである）。`Tape::new()` も同一の内部構造
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

### 3.5 演算跨ぎの遅延と 3 層の実体化境界

（本節は codex-review 第 6 波 P1 指摘を受けて全面改訂する。第 5 波の
旧稿は `Var` の fallible 演算が返る前に自分の出力を必ず実体化する
契約（本節でも層 1 の一部として維持する）を根拠に、遅延の生存窓を
1 回の呼び出し内部だけに縮小していた。しかしこれは TASK-12.1 の中核
要件である「現行公開 API 上での 4〜6 段 elementwise 連鎖の融合」を
不可能にする（第 6 波 P1 指摘、§1・§3.3）。本節は「複数回の独立した
公開 `Var` 呼び出しをまたいで遅延を持ち越す」設計を、`value`／
`to_tensor` の非 fallible 契約を壊さない形で復活させる。)

### 3.5.1 `TapeNode` の構造と遅延を許容する演算の切り分け

- 遅延状態は `autodiff::TapeNode`（`tape.rs:118`。`pub(crate)`）だけに
  持たせ、`tensor-core` の `Tensor`／`Storage<T>`（`tensor.rs:33`）は
  一切変更しない（§3.4 で確定済み）。`TapeNode` を次のとおり拡張する
  （実装は #164）:
  ```rust
  pub(crate) struct TapeNode {
      pub(crate) op: Op,
      /// 構造的に確定する出力 shape。実体化なしに算出できる（`add`／
      /// `mul` は broadcast、`matmul` は行列積、`sum`／`max` は縮約、
      /// `relu`／`exp`／`tanh` は恒等の shape 計算式であり、いずれも
      /// 入力の `shape` フィールドだけを読めば求まる。`var.rs` の
      /// 既存の shape 検証ロジック（`broadcast_shape`／
      /// `matmul_out_shape`／`reduce_out_shape`）は今日すでに
      /// `.value().shape()` ではなく形状情報のみを消費しているため、
      /// 本節はこの検証ロジック自体を変更しない）。
      pub(crate) shape: Vec<usize>,
      /// 実体化済みの値。空（`OnceCell::get() == None`）は「未実体化」
      /// を表す。`OnceCell::get_or_init`／`set` はいずれも `&self`
      /// （共有参照）で呼べるため、`RefCell<Vec<TapeNode>>` の
      /// `borrow()`（共有借用）だけで埋められる。`Tape::push`
      /// （`tape.rs:186`。`borrow_mut()` を要する唯一の追記経路）を
      /// 実体化処理が再入することはない（新規ノードを追加しないため。
      /// 下記「materialize ヘルパー」参照）。
      pub(crate) value: std::cell::OnceCell<Tensor<f32>>,
  }
  ```
- **遅延を許容する演算とその場で実体化する演算の切り分け**: `relu`／
  `exp`／`tanh`（`var.rs:257` 以降。shape 不変の単項演算のため
  「構造的に失敗しえない」という既存設計判断〈`docs/public-api-design.md`
  §3.2〉により非 fallible な `fn ..(&self) -> Var<'t>` のまま）は
  `Tape::push` 時に `value` を空の `OnceCell` のまま記録し、**自身の
  出力を実体化せずに返す**（連鎖を延長する）。これが本改訂で 4〜6 段
  連鎖を実現する主要因である: `relu`／`exp`／`tanh` は任意回・任意順
  に連結できるため、`a.relu().exp().tanh().relu()`（4 段）のように
  §3.2 (d) の連鎖長上限まで単独で連鎖を伸ばせる。一方 `add`／`mul`／
  `matmul`／`sum`／`max`（`Result<Var<'_>, AutodiffError>` を返す
  fallible 演算。`var.rs:111`〜`:159`）は、第 5 波の契約を維持し
  **返る前に自分の出力を実体化済みにする**（`OnceCell` を埋めてから
  返す）。これらの演算は shape 検証を実体化前に完了できる（上記
  `shape` フィールドのみを読む）が、実際の計算（`eval::add` 等）には
  入力の具体値が要るため、入力が `relu`／`exp`／`tanh` の遅延連鎖で
  あった場合はその実体化を自身の実行の一部として行う（§3.5.2 の
  「層 1」）。
- **葉ノード（`Op::Leaf`）は常に実体化済み**: `Tape::var(&tensor)`
  （`tape.rs:164`）は呼び出し時点で既に具体的な `Tensor<f32>` を受け
  取るため、`value` を即座に `OnceCell::from(tensor.clone())` で埋めて
  push する。これにより「実体化されていないノードの入力を遡ると、
  有限回で必ず実体化済みノードまたは `Op::Leaf` に到達する」という
  帰納法の基底が成り立つ。
- **走査順が既に発生順トポロジカル順であること**: `Tape::push`
  （`tape.rs:186`〜`:190`）は `NodeId(nodes.len())` を採番してから
  追記するため、あるノードの入力 `NodeId` は常に自分自身の `NodeId`
  より小さい。したがって遅延連鎖を先頭（未実体化の最も古いノード）
  から辿る際、単純な逆方向の連結リスト走査（各ノードの入力 `NodeId`
  を辿る）だけでよく、循環検出・グラフ探索アルゴリズムは不要である
  （§2 の `FusionGraph`／連結成分検出とは異なり、`autodiff` 側の遅延
  連鎖は常に単純な線形チェーンである。なぜなら `relu`／`exp`／`tanh`
  はいずれも入力 1 個の単項演算だからである）。

### 3.5.2 層 1（fallible 境界）: 後続の fallible `Var` 演算・`Tape::backward` の VJP 連鎖内部

- **後続の fallible `Var` 演算**（`add`／`mul`／`matmul`／`sum`／`max`）
  が入力側の値（`self.value()`／`other.value()`）を読む際、その入力の
  `TapeNode.value` が未実体化であれば、この読み出しの内部で実体化を
  行う。実体化は次の手順で行う（`autodiff` クレート内の
  `pub(crate)` ヘルパー。実装は #164）:
  1. 未実体化ノードから入力方向へ、`OnceCell` が埋まっているノードまで
     辿り、間に挟まる `relu`／`exp`／`tanh` の連続列を `FusedOpKind`
     （§3.4）へ変換する（`Op::Relu`/`Exp`/`Tanh` と `FusedOpKind::Relu`/
     `Exp`/`Tanh` は 1:1 対応。§2.1・§3.4）。
  2. `self.ops`（`Tape` が保持する `Option<Box<dyn BackendOps>>`。
     §3.4）が `Some(ops)` であれば、`FusionPlan::from_ops`（§3.4。
     `pub` + `#[doc(hidden)]`）で `FusionPlan` を構築し
     `ops.run_fused(&plan, &[&base])` を試す。
  3. `run_fused` が `Ok` を返せば、または `self.ops` が `None`／
     `run_fused` が `Err` を返せば、連鎖の各ノードを発生順
     （§3.5.1「走査順」）に既存の `eval::relu`／`exp`／`tanh` へ
     1 段ずつ逐次フォールバックする（§4 の fail-safe 方針。CPU
     参照実装は構造的に失敗しない。§3.5.3 参照）。
  4. `run_fused` が `Ok` 以外を返し、かつこの呼び出しが「後続の
     fallible `Var` 演算」の内部（本節）である場合は、**フォール
     バックせず** `Err(BackendError)` をそのまま
     `AutodiffError::Backend(BackendError)` へ変換して呼び出し元へ
     `?` で伝播する。すなわち **層 1 では `run_fused` の失敗は
     フォールバックしない**（層 2〈§3.5.3〉との違い）。手順 3 の
     フォールバックは「`run_fused` を試みず `self.ops` が `None`」
     の場合、または「上限到達時にその場で実体化する必要がある
     （§3.5.4）が呼び出し元が非 fallible」の場合にのみ適用される。
  実体化が完了した各ノードの `OnceCell` はその場で `set()` する
  （既に空であることを直前に確認済みのため、`OnceCell::set` の
  `Err` 分岐（二重設定）には構造的に到達しない。到達しないことが
  自明な `Result` は `.unwrap_or_else(|_| unreachable!("..."))`
  で扱う既存パターン〈`eval.rs:88`〉に倣う。`unwrap()`／`expect()`
  は使わない〈`.claude/rules/coding-rust.md`〉）。これにより同じ
  ノードへの 2 回目以降の読み出しは再計算しない。
- `Tape::backward`（`backward.rs:73`。公開シグネチャ
  `pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`
  は変更しない）は、それ自体が単一の fallible 呼び出しである。内部で
  テープを逆順に辿り各ノードの VJP（`grad.rs::vjp`。`Op` 単位のまま。
  §3.3）を計算する過程で、1 つの VJP 計算式が複数の elementwise 演算
  から成る場合（例: `tanh` の VJP `grad * (1 - y * y)` は `mul`・`sub`
  の連鎖）も、上記と同じ手順（`self.ops` を借用して融合を試み、失敗は
  フォールバックせず `?` で伝播する）を用いる。
- **`Gradients::get` は非 fallible のまま**: `backward` は自身が返す
  `Gradients` に含まれるすべての勾配 `Tensor` を、`Ok(Gradients { .. })`
  を返す直前までに実体化し終える。したがって `Gradients::get` は
  追加の実体化発火点を必要としない。
- **`AutodiffError::Backend` variant（変更なし。第 5 波で確定済みの
  設計をそのまま踏襲する）**:
  ```rust
  pub enum AutodiffError {
      // 既存 variant（Shape／Backward／TapeMismatch／InvalidArgument）は変更しない。
      /// 融合実行・実体化で発生した型付きバックエンドエラー
      /// （TASK-12.1a／#164。`tensor_core::BackendError` をラップ）。
      Backend(tensor_core::BackendError),
  }

  impl From<tensor_core::BackendError> for AutodiffError {
      fn from(err: tensor_core::BackendError) -> Self {
          AutodiffError::Backend(err)
      }
  }
  ```
  `#[non_exhaustive]` enum（`error.rs:19`）への非破壊 variant 追加・
  新規 `From` 実装（既存の呼び出し元の網羅的 `match` を壊さない。
  `error.rs:15-18` の既存方針と同じ理由）。`error.rs:66` 以降の
  `impl fmt::Display for AutodiffError` は `match` で全 variant を
  網羅しているため、`Backend` variant 追加時は対応する `Display`
  アームの追加も同時に行う（実装時の見落とし防止のため本節に明記
  する）。
- `Tape` が記録する `Op` 単位のノード粒度・`grad.rs::vjp` の走査対象
  （`Op` 列）自体には影響しない（§3.3 の契約を変更しない）。本節が
  変更するのは `Var` の各演算メソッドおよび `vjp`（`grad.rs:31`）と
  その内部の全 VJP 関数の**内部実装**（入力読み出し時の実体化・
  `Result<_, BackendError>` 伝播）のみであり、`Tape`／`Op` の**構造**
  （ノード粒度・走査順）には影響しない（#164 のスコープに明示的に
  含める）。

### 3.5.3 層 2（非 fallible 境界）: `Var::value`／`Var::to_tensor` と CPU フォールバック

- `Var::value`（`var.rs:74`。`-> Ref<'_, Tensor<f32>>`）・
  `Var::to_tensor`（`var.rs:81`。`-> Tensor<f32>`）は**シグネチャを
  一切変更しない**。対象ノードが未実体化であれば、§3.5.2 と同じ
  手順 1・2 で融合実行を試みるが、**`run_fused` が `Ok` 以外を返した
  場合、または `self.ops` が `None` の場合は `Err`／`None` を呼び出し
  元へ伝播せず、記録済みの `relu`／`exp`／`tanh` の連鎖を発生順に
  `eval::relu`／`exp`／`tanh`（非 fallible。§3.5.1 の走査順により
  再帰なしで辿れる）で逐次 eager 再実行し、必ず `Tensor<f32>` を
  返す**。`OnceCell::get_or_init`（`&self` で呼べる、`FnOnce() -> T`
  の非 fallible なクロージャを取る）にこの「融合を試み、失敗したら
  CPU 再実行する」処理全体を渡せばよく、`get_or_try_init`（unstable）
  は使わない。この経路は構造的に失敗しない（§3.5.1 の shape 検証は
  各演算の呼び出し時点で既に完了しており、`eval::relu`／`exp`／`tanh`
  自身も `-> Tensor<f32>`（非 fallible）である。§3.5.4 も参照）ため、
  `Var::value`／`Var::to_tensor` は誤った値・欠落値を返すことも
  `panic!` することもない。
  ```rust
  // materialize ヘルパー（`tape.rs` 内 `pub(crate)`。イメージ）。
  // `nodes` は `self.tape.nodes.borrow()` で得た共有借用であり、
  // `get_or_init` のクロージャ内で他ノードの `value` を読む際も
  // 同じ共有借用を再利用する（`borrow_mut()` を一切要求しない）。
  fn materialize_non_fallible<'a>(
      nodes: &'a Vec<TapeNode>,
      ops: Option<&dyn BackendOps>,
      id: NodeId,
  ) -> &'a Tensor<f32> {
      nodes[id.0].value.get_or_init(|| {
          // 手順 1・2（§3.5.2）を試み、失敗したら CPU 参照実装で
          // 逐次 eager 再実行する（§3.5.4）。
      })
  }
  ```
- `Gradients::get` も同じ非 fallible 境界として扱う（`backward` が
  返す直前に全勾配を実体化し終える契約〈§3.5.2〉のもとでは追加の
  発火点として使われる場面は稀だが、契約としては層 2 と同一に扱う）。
- **`value()` が `Ref` を保持している最中に他の `Var` の実体化が
  発生しても panic しない**: `Var::value()` は `Ref::map(self.tape
  .nodes.borrow(), |nodes| nodes[self.id.0].value.get_or_init(..))`
  のように、`nodes.borrow()`（共有借用）から得た `Ref` を返す。
  複数の `Var::value()` 呼び出し（例: `let a = x.value(); let b =
  y.value();`）はいずれも `borrow()`（共有借用は多重に取得できる）
  であり、`y` が未実体化でも `get_or_init` は `&self`（共有参照）の
  みで完結するため、`x.value()` の `Ref` を保持したまま `y.value()`
  を呼んでも `RefCell` の二重可変借用 panic は起きない（`Tape::push`
  の `borrow_mut()`〈`tape.rs:186`〉が要求する契約「呼び出し元は
  借用を閉じてから呼ぶ」は、実体化処理がノードを追加しない
  〈§3.5.1〉ため元々抵触しない）。
- `Tensor::get`／`as_slice`／`contiguous`（`tensor-core` の汎用
  アクセサ）は本節の対象外のまま**シグネチャ・意味論を一切変更
  しない**（`get`／`as_slice` の既存契約「範囲外・非 contiguous
  のみ `None`」もそのまま維持される。実体化に起因する `None` 分岐は
  存在しない）。`Var::value_raw`・第 4 波の「系統 1〜3」はいずれも
  新設しない。
- `&dyn BackendOps` を直接呼ぶ既存経路（`ops_for` 経由を含む。§1・§3.4）
  は引き続き本設計の対象外であり、この経路の `Tensor`／`Storage` は
  常に実体化済みのまま（`ops_for(...).add()` 等の実装は `Tape` を
  経由せず遅延連鎖を一切構築しない）。

### 3.5.4 連鎖長上限（§3.2 (d)）との相互作用

- `relu`／`exp`／`tanh` が `TapeNode` を push する直前に、自身が延長
  しようとしている遅延連鎖の長さ（未実体化の連続する入力ノード数 + 1）
  を数える。§3.2 (d) の上限（4〜6 段。具体的な段数は #164 実装時に
  確定する）に達する場合、**その場で自分自身のノードを実体化してから
  返す**（連鎖はここでリセットされ、次の演算から新しい連鎖として
  数え直す）。`relu`／`exp`／`tanh` 自身は非 fallible なままである
  ため、この実体化は §3.5.3 の非 fallible 境界（層 2）の手順（融合を
  試み、失敗したら CPU フォールバック）を使う。
- 一方、fallible な `Var` 演算・`Tape::backward` の VJP 連鎖内部
  （層 1）が読み出しの過程で上限超過の連鎖に遭遇した場合は、§3.5.2 の
  手順（融合を試み、失敗は `?` で伝播）にそのまま従う。
- 上限が「到達させた演算が fallible か非 fallible かにより層 1／層 2
  いずれかへ合流する」（§1・§3.2 (d)）とはこの意味である。

### 3.5.5 窓 (b): 将来の複合エントリポイント

- 現状の `Var` 演算 API は 1 呼び出し 1 演算の粒度であるが、§3.5.1〜
  3.5.4 の設計により `relu`／`exp`／`tanh` の連鎖に限れば複数回の
  呼び出しをまたいで融合が働く。将来、複数の演算を 1 回の `Result`
  で返す複合エントリポイント（`compat::Sequential::forward` 相当。
  `docs/public-api-design.md` に設計段階として記載。または将来の
  「グラフ一括実行」API）が追加された場合、その内部実装は §3.5.2 と
  同じ要領（融合を試み、失敗は `?` で伝播）に従う。この窓は #164 の
  必須スコープではなく、当該複合 API が実装される時点で適用される
  （本節はその際に従うべき契約を先に確定しておくもの）。

### 3.5.6 view 系操作（transpose・narrow・reshape）

- `offset`／`shape`／`strides` のみを扱う view 系操作
  （`Arc::clone(&self.storage)` 経路）は、`Tensor`／`Storage` が本節
  でも一切変更されないため、他の `tensor-core` の既存 view 演算と
  同様に振る舞う。「未実体化のまま view を複製する」という複雑さは
  遅延状態が `Tensor` に一切到達しないこと自体により生じない。
- §3.5.1〜3.5.4 の内部実装が構築する融合グラフ（`FusionPlan`）の
  内部では、transpose を挟む部分列は §1・§2.3 のとおり非融合
  フォールバックへ倒す（`NodeMeta.contiguous == false` が §3.2 (e)
  の実体化条件に対応する）。`relu`／`exp`／`tanh` の遅延連鎖は
  shape 不変の単項演算のみで構成されるため、この境界条件が実際に
  作用するのは #163／#164 が transpose を伴う演算を融合対象へ拡張
  する場合に限られる（現状スコープでは transpose 系操作は遅延連鎖に
  含まれない）。

### 3.5.7 CPU フォールバックの数値面の注意

- §3.5.3 の CPU フォールバックが選ばれた場合、その区間の結果は
  `run_fused` が成功していた場合と数値的に完全一致しない可能性がある
  （丸め順序・FMA 契約の違いにより発生しうる差。CPU 参照実装は
  `f32::mul_add` を用いる統一契約〈`.claude/rules/coding-rust.md`
  「バックエンド構成（REQ-2）」〉に従う）。この差は**バックエンド間
  数値一致で既に許容されている複合判定「相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満」（§4）と同じ性質の差であり、フォールバックは
  この判定の対象外側へ逸脱しない**。融合の有無・フォールバックの
  発生有無によってテスト許容誤差を緩和する実装は認めない（§4・§5
  A08 の既存方針をそのまま適用する）。
- **run-to-run 非決定性としての扱い**: `run_fused` の成否（デバイス
  障害・一時的なリソース枯渇等）は決定的シード設定（学習系回帰
  テストの基本方針。`.claude/rules/coding-rust.md`「テスト・ベンチ」）
  で制御できない新しい非決定性の発生源になりうる。学習系回帰テストが
  融合発生の有無に依存しない結果を要求する場合は、テスト側で使用
  する `BackendOps` 実装を固定する（`Tape::with_backend` に決定的な
  テスト用実装を渡す、または `Tape::new()` を使い `self.ops` を
  常に `None`〈§6.2 の既定バックエンド解決未確定の間の既定挙動〉に
  保つ）ことで、フォールバックの発生有無自体を固定する必要がある
  （§6.2 に記録する）。

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
| #164（ディスパッチ統合） | §1 の「利用者向け制御 API を提供しない」方針・「公開コンストラクタの選択は融合スイッチにならない」契約・「演算跨ぎの遅延を復活し、実体化を 3 層の境界で規定する」契約（codex-review 第 6 波 P1 指摘への回答）に基づく融合対応経路の実装。§3.4 で確定した `FusionValue`／`FusionSession`（借用ベース・`Arc`／`Mutex`／`Send + Sync` 不要）・`FusionPlan::from_ops`（`autodiff` 専用のクレート間構築経路。`pub` + `#[doc(hidden)]`）／`BackendOps::run_fused`（デフォルト実装付きで trait 定義へ追加。既存メソッドの契約は変更しない）接続契約、`Tape` の非公開フィールド `ops: Option<Box<dyn BackendOps>>` と新規公開コンストラクタ `Tape::with_backend(ops: Box<dyn BackendOps>)` の追加（＝ TASK-1.9 の backend 経由実行への置き換えと同時実施）。§3.5.1 で確定した `TapeNode`（`shape: Vec<usize>` ＋ `value: OnceCell<Tensor<f32>>`）への拡張と、`relu`／`exp`／`tanh` が遅延連鎖を延長し `add`／`mul`／`matmul`／`sum`／`max` が返る前に自身の出力を実体化する切り分けの実装。§3.5.2（層 1・fallible 境界。融合失敗は `?` でそのまま伝播）・§3.5.3（層 2・非 fallible 境界。融合失敗は `eval::relu`／`exp`／`tanh` による CPU 参照実装への逐次フォールバックで必ず成功させる。`OnceCell::get_or_init` を使い `get_or_try_init`（unstable）は使わない）・§3.5.4（連鎖長上限との相互作用）の実装。`AutodiffError::Backend(BackendError)` variant と `From<BackendError>` 実装の追加（`Display` アーム追加を含む）。既存の `eval::dense_vec`・`eval::relu`／`exp`／`tanh`（CPU 参照実装）は非 fallible のまま変更しない |
| #165（テスト） | §1・§2.3 の transpose 非融合フォールバック、§2.4 の fan-out 融合、§3.3 の autodiff 契約（VJP がノード単位のまま変わらないこと）の検証、**§1「公開コンストラクタの選択は融合スイッチにならない」契約の検証**（同一演算列を `Tape::new()` と `Tape::with_backend(ops)` の双方で実行し、数値結果が数値一致複合判定〈§4〉を満たすこと、および融合の発生有無がどちらのコンストラクタを呼んだかではなく §1 の 2 条件〈バックエンド解決可否・演算列の融合可否判定〉のみで決まることの検証）、**§3.5「演算跨ぎの遅延と 3 層の実体化境界」の検証**（codex-review 第 6 波 P1 指摘への回答）: (i) 独立した公開 `Var` 呼び出しをまたぐ `relu`／`exp`／`tanh` の連鎖（例: `x.add(&y)?.relu().exp().tanh().relu()`。4 段）が単一の `run_fused` 呼び出しへ融合されること（カウンタ付き `BackendOps` テスト実装で `run_fused` が 1 回だけ呼ばれ、`add` 単体では呼ばれないことを確認する）、(ii) 層 1（fallible 境界。§3.5.2）での融合失敗が、それを引き起こした後続の fallible `Var` 演算自身の `Err(AutodiffError::Backend)` として直接返ること（キャッシュ経由の遅延表面化が発生しないこと）、(iii) 層 2（非 fallible 境界。§3.5.3）での融合失敗時、`Var::value`／`Var::to_tensor` が `panic!` せず、CPU フォールバックで計算した値と融合が成功していた場合の値が数値一致複合判定〈§4〉を満たすこと（フォールバックは値の正しさを保証するのみで #163 の融合カーネル自体のバグを隠さないことの検証。フォールバック発生をテスト用カウンタで観測できることも確認する）、(iv) `x.value()` で得た `Ref` を保持したまま別の未実体化 `Var` の `value()`／`to_tensor()` を呼んでも panic しないこと（§3.5.3「`value()` が `Ref` を保持している最中…」の検証）、(v) `Tape::backward` の VJP 連鎖内部で融合が発生する場合（§3.5.2）に、その融合が失敗すると `Tape::backward` が `AutodiffError::Backend` を返すこと、かつ成功時は `Gradients::get` がそのまま非 fallible に値を返せること、(vi) §3.5.4 の連鎖長上限に到達した場合に fallible／非 fallible いずれの経路でも連鎖がその場でリセットされ、後続の演算が正しい実体化済み値を入力として使えること、(vii) §2.4 の fan-out が単一の融合グラフ構築で正しく解決されることの検証 |
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
- **（第 6 波で撤回）独立した公開 `Var` 呼び出しをまたぐ elementwise
  連鎖融合を行わない受け入れコスト**: 第 5 波はこのエントリで
  「`a.add(&b)?.mul(&c)?` のように独立した公開 `Var` 呼び出しをまたぐ
  elementwise 連鎖は融合されない」ことを受け入れコストとして記録して
  いたが、これは TASK-12.1 の中核要件そのものを満たさないという
  codex-review 第 6 波 P1 指摘により撤回された。本改訂（§1・§3.5）で
  `relu`／`exp`／`tanh` の遅延連鎖が複数回の公開 `Var` 呼び出しを
  またいで融合対象になったため、このエントリは受け入れコストでは
  なくなった。**残る限定（撤回ではなく縮小として記録する）**:
  `add`／`mul`／`matmul`／`sum`／`max` どうしが直接連続する区間
  （両者とも層 1 で即座に実体化される。§3.5.1）には融合が及ばない。
  PoC-9 実測（`ew4`／`ew6`）の正確な演算構成が unary 演算主体で
  あることを前提に本改訂の設計判断（§3.5.1「切り分け」）を行っており、
  binary 演算どうしが連続する区間への拡張（構造的な shape 追跡を
  `add`／`mul` にも適用する設計）は #162 以降の拡張候補として別途
  検討する。
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
- **異なる `Tape` を跨ぐ融合境界（第 5 波で「解消」と記録した内容を
  第 6 波で再オープンし、解決方針を訂正する）**: PR #357 review 再
  指摘 P1-1／P1-2 が対象としていた「独立した `Pending` 同士が二項
  演算で合流する」というシナリオ（`(a + b) * (c + d)` が独立した
  2 つの遅延連鎖をまたぐケース）は、第 5 波では「`FusionSession` は
  1 回の fallible 呼び出し内でしか生存しないため構造的に発生しない」
  として解消済みと記録していた。本改訂（§1・§3.5）は演算跨ぎの遅延を
  復活させたため、この懸念は形を変えて**再び生じうる**: `(a + b) *
  (c + d)` のように、それぞれ独立した遅延連鎖を持つ 2 つの `Var`
  （`a+b` の連鎖と `c+d` の連鎖）が `mul` で合流するケースである。
  ただし本改訂の設計（§3.5.1）では、遅延連鎖を延長できるのは `relu`／
  `exp`／`tanh`（すべて入力 1 個の単項演算）のみであり、`add`／
  `mul` 自身は遅延連鎖を延長せず返る前に実体化する（§3.5.2）ため、
  「2 つの遅延連鎖が 1 つの二項演算へ合流する」状況自体は生じない
  （合流点となる `mul` 呼び出しは、両オペランドをそれぞれ独立に
  §3.5.2 の手順で実体化してから `eval::mul` を呼ぶ）。**別テープ間の
  合流**（`check_same_tape`／`var.rs:87`〜`:92` が既に拒否する
  `AutodiffError::TapeMismatch`）はこれと別の懸念であり、遅延連鎖は
  常に単一の `Tape` の `nodes: RefCell<Vec<TapeNode>>` の内部だけで
  構築されるため（§3.5.1 の走査順の前提「あるノードの入力 `NodeId`
  は常に自分より小さい」は同一 `Tape` 内でのみ意味を持つ）、別テープの
  ノードが遅延連鎖に紛れ込むことはない（`check_same_tape` は演算入口
  で shape 検証より前に呼ばれる既存の検査であり、本改訂でも変更しない）。
  したがって本エントリの懸念は本改訂の設計（`relu`／`exp`／`tanh`
  限定の単項連鎖）の範囲では発生しないと結論する一方、§6.2 冒頭で
  記録した「binary 演算どうしが連続する区間への拡張」を将来行う場合は、
  この合流シナリオへの対処（合流点での実体化の強制、または合流を
  許容する場合の `FusionGraph` 側の連結成分検出〈§2〉の再設計）を
  同時に検討する必要がある拡張候補として記録する。
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
- **CPU フォールバック（§3.5.3・§3.5.7）が持ち込む run-to-run
  非決定性（本改訂で新規に記録する）**: `run_fused` の成否はデバイス
  障害・一時的なリソース枯渇等の環境要因に左右されうるため、決定的
  シード設定（`.claude/rules/coding-rust.md`「テスト・ベンチ」）だけ
  では「融合実行されたか CPU フォールバックしたか」を再現できない
  場合がある。両経路の結果は §4 の数値一致複合判定を満たすことを
  #165 で検証するが（§6.1 #165 (iii)）、学習系回帰テストが bit-exact
  に近い再現性を要求する場合は、テスト側で使用する `BackendOps`
  実装を固定する（決定的なテスト用実装を `Tape::with_backend` に渡す、
  または `Tape::new()`〈`self.ops` が既定で `None`。上記エントリ〉を
  使う）ことでフォールバックの発生有無自体を固定する必要がある。
  #164 の実装ガイドとして記録する。
- **CPU フォールバックは融合カーネル（#163）自体の正しさを保証しない
  （本改訂で新規に記録する）**: §3.5.3 の CPU フォールバックは
  `run_fused` が失敗した場合に利用者へ正しい値を返すための安全網
  であり、`run_fused` が誤った値を「成功」として返す不具合（#163 の
  融合カーネル生成バグ）を検出・防止するものではない。#165 は
  フォールバック経路の値と融合成功時の値を突き合わせるテスト
  （§6.1 #165 (iii)）を融合カーネルの正しさの検証としても位置付け、
  フォールバックの存在を理由にこの突き合わせテストを省略しない。
