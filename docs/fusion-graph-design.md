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
  新規の公開エントリ関数は追加しない（適用箇所の具体化は §3.5、
  `BackendOps` 契約との接続は §3.4 で規定する）。
  - 根拠: REQ-12 受け入れ基準「ライブラリ利用者が明示的に融合を制御する
    API は提供しないこと」（`docs/spec/04-requirements.md:249`）。REQ-11
    読み替え設計（`docs/dispatch-rules-design.md` §1「利用者向けの明示切替
    API は提供しない」）と同方針であり、本ライブラリ全体の一貫した設計
    判断として踏襲する。**この受け入れ基準が要求するのは「利用者が融合を
    制御する API を提供しないこと」であり、「`&dyn BackendOps` を直接呼ぶ
    既存経路にも融合を透過的に適用すること」までは要求しない**（誤読の
    余地があったため明記済み。§3.4 参照）。`&dyn BackendOps` を直接
    呼ぶ既存経路（`ops_for` 経由の呼び出し含む）は本設計の変更対象外とし、
    従来どおり eager・非融合のまま維持する。
  - **単一の公開構築経路への統合（codex-review 第 14 波 P1 指摘への回答。
    第 13 波までの「`Tape::new()`（非融合）と `Tape::with_backend(ops)`
    （融合）の選択」構成・「REQ-12 が保護する『利用者』は compat 層」と
    いう防御的な範囲限定は、いずれも本改訂で撤回する）**: 第 13 波は
    「`autodiff` クレート自身の 2 つのコンストラクタが融合の有無を
    決めること自体は否定しないが、REQ-12 が保護する『利用者』は compat
    層（REQ-9）の公開面であり、compat 層は常に `Tape::with_backend` を
    使うため抵触しない」という論法を採っていた。第 14 波 P1 指摘は、
    この論法が「`Tape::new()` と `Tape::with_backend(ops)` の選択が
    融合の有無を決める」という事実そのものを消していない点を問題視した
    （利用者の範囲をどう限定しても、コンストラクタ選択＝融合スイッチと
    いう構造は `autodiff` の公開 API に残り続ける）。本改訂はこの構造
    自体を解消する: **`autodiff::Tape` の非公開フィールドを `Option`
    ではなく必須の所有値 `ops: Box<dyn BackendOps>` とし、`Tape` を
    構築できる 2 つの公開コンストラクタ（`Tape::new()`／
    `Tape::with_backend(ops)`）のどちらを経由しても必ず `ops` が
    埋まった状態でしか `Tape` が存在できないようにする**（`Option` を
    経由した「未解決」状態が型として存在しない。第 10 波の懸念
    「`None` = 未解決 = 公開コンストラクタが事実上の融合スイッチに
    なる」は、`None` という値自体をコンパイル時に排除することで解消
    する——「利用者の範囲をどう定義するか」という防御ではなく、
    「融合を無効化できる構築経路がそもそも存在しない」という構造的
    事実によって REQ-12 を満たす）:
    ```rust
    // crates/autodiff/src/tape.rs（非公開フィールド。実装は #164）
    pub struct Tape {
        id: TapeId,
        nodes: RefCell<Vec<TapeNode>>,
        /// 常に何らかの `BackendOps` 実装を保持する（`Option` ではない
        /// 必須所有値）。§3.5 が規定する 3 層の実体化境界に従って
        /// 記録を遅延し、実体化時に `ops.run_fused`（§3.4）による融合を
        /// 試みる。`Tape::new()`・`Tape::with_backend(ops)` のどちらで
        /// 構築しても、この後段の融合方針自体は一切変わらない
        /// （変わるのは注入される具体的なバックエンド実装のみ）。
        ops: Box<dyn BackendOps>,
    }
    ```
    - **`Tape::new()`**（既存の既定コンストラクタ。シグネチャは変更
      しない）: 内部で `backend_cpu::CpuBackendOps::new()`
      （`crates/backend-cpu/src/ops.rs:27`。引数なし・`Result` を
      返さない無条件成功のコンストラクタ）を構築し `ops` へ格納する
      （2026-08-08 ユーザー承認「既定バックエンドを `Device::Cpu`
      とすること」に基づく確定。承認記録は §6.2「`Tape::new()` が
      使う既定バックエンドの供給規則」参照）。**これに伴い `autodiff`
      は `backend-cpu` への workspace path 依存を新規に持つ**
      （依存方向: `backend-cpu` → `tensor-core`、`autodiff` →
      `{tensor-core, backend-cpu}`。`backend-cpu` は `autodiff` へ
      依存しないため循環は生じない。第 9 波が確定した「`autodiff →
      backend-cpu` の workspace path 依存は追加しない」という契約は
      本改訂で撤回する——撤回の理由・整理は §3.4「`autodiff →
      backend-cpu` の workspace path 依存を追加する」参照）。
      `Tape::new()` を呼ぶすべての経路（compat 層を含む・含まない
      いずれも）で融合が既定で・透過的に効く。
    - **`Tape::with_backend(ops: Box<dyn BackendOps>)`**（新設。実装は
      #164）: 呼び出し元が指定した `BackendOps` 実装（CUDA／Metal
      加速や決定的なテスト用実装等）を `ops` へ格納する。**バックエンド
      の明示選択**手段として提供する（`docs/public-api-design.md` §4.1
      が定める `Device` の「列挙と明示選択」方針と同じ趣旨。`Device`
      から具体的な `BackendOps` 実装〈CUDA／Metal 加速〉への結線規則
      自体は §6.2「未決事項として残る部分」が定めるとおり本文書では
      未確定であり、`Tape::with_backend` はその結線が決まった際の
      注入口として使う）、
      §3.5 が規定する 3 層の実体化境界に従って記録を遅延し、実体化時に
      `ops.run_fused`（§3.4）を試み、`BackendError::Unsupported` は
      同じ `ops` の per-op メソッドへの逐次呼び出しへフォールバック
      する（層 2 に限りそれも失敗した場合は `eval.rs` を最終手段として
      使う。§3.5.2・§3.5.3）。
    - **契約: 両コンストラクタは同一の融合方針を共有する**。`Tape::new()`
      と `Tape::with_backend(ops)` はいずれも「§3.5 の 3 層の実体化境界
      に従って記録を遅延し、`run_fused` による融合を試みる」という
      後段の挙動を完全に共有する（`FusionSession`／`FusionPlan`／
      `run_fused` は §3.4 のとおりどちらの経路にも同一に関与する）。
      **選択が変えるのは実行バックエンドの種類（既定の `backend-cpu` か
      呼び出し元が指定した任意の `BackendOps` 実装か）のみであり、
      融合の有無ではない**。「`with_backend` だけが融合する」という
      記述は本改訂の対象文書全体（本文書・`docs/public-api-design.md`）
      から全廃する（該当箇所は §3.4・§3.5・§6.1・§6.2 で横断的に
      更新する）。
  - **演算跨ぎの遅延・二項 elementwise 演算の遅延化（codex-review 第 6
    波・第 13 波 P1-a 指摘への回答。詳細は §3.2・§3.4・§3.5）**: 遅延の
    生存窓は「複数回の独立した公開 `Var` 呼び出しをまたぐ」形で持ち
    越す（`a.add(&b)?.relu().exp().tanh()` のように独立した複数回の
    公開 `Var` 呼び出しをまたぐ連鎖が融合対象になる）。**単項演算
    （`relu`／`exp`／`tanh`）だけでなく `add`／`mul`（二項 elementwise
    演算）も出力を遅延グラフへ保持する**（第 11〜12 波は `add`／`mul`
    が返る前に必ず自身の出力を実体化する設計だったため PoC-9 実測の
    `ew4`／`ew6`／`ew_fanout`〈`add`・`mul` の連鎖・fan-out を含む〉が
    融合対象から外れていた。本改訂でこの限定を解く）。`matmul`／
    `sum`／`max`（非 elementwise）は引き続き実体化境界のままとする
    （§3.2 (a)(b)）。**shape 検証と実行（forward 値計算）を分離する**:
    `add`／`mul` を含むすべての演算は、shape 検証を `Tape::push` に
    よるノード追加より前に完了し、不正な shape は当該演算の呼び出しが
    その場で `Err` を返す（記録すらされない）。一方、検証を通過した
    演算の実際の計算（forward 値の算出）は実体化境界に到達するまで
    遅延しうる。この分離により、**`Var::add`／`mul` が `Ok` を返す
    ことは「shape が妥当でノードが記録された」ことのみを意味し、
    「加算・乗算が計算済みである」ことを意味しなくなる**（`ops` は
    §1 のとおり `Tape::new()`／`Tape::with_backend` のいずれでも常に
    埋まっているため、この非対称性は例外なく常に成り立つ）。バック
    エンド実行の失敗は次の実体化境界（後続の `matmul`／
    `sum`／`max`・`Tape::backward`・`Var::value`／`to_tensor`）で
    初めて表面化しうる（§3.5.2 参照）。`value`／`to_tensor` の非
    fallible 契約を壊さないよう、実体化の発火点を次の 3 層で規定する
    （型・実装の詳細は §3.5）:
    1. **fallible 境界**: `Var::matmul`／`sum`／`max`（既に
       `Result<Var<'_>, AutodiffError>` を返す契約。非 elementwise の
       実体化境界、§3.2 (a)(b)）が自身の計算のために入力側の未実体化
       値を必要とする場合、`Tape::backward` の VJP 連鎖内部、および
       §3.2 (d) の連鎖長上限に fallible な演算の呼び出し中に到達した
       場合。`ops`（§1。常に埋まっている必須所有値）の `ops.run_fused`
       が `Unsupported` 以外の失敗を返した場合に限り、型付きエラー
       （`AutodiffError::Backend(BackendError)`。§3.5.2 で確定済みの
       variant をそのまま再利用する）として `?` で呼び出し元へ伝播する
       （バックエンド実行失敗は利用者が結果を受け取るより前に必ず
       型付きで観測されるべきという契約をそのまま踏襲する）。
    2. **非 fallible 境界**: `Var::value`／`Var::to_tensor`
       （`-> Ref<'_, Tensor<f32>>`／`-> Tensor<f32>` の既存シグネチャを
       一切変更しない）・`Gradients::get`、および §3.2 (d) の連鎖長
       上限に非 fallible な演算（`relu`／`exp`／`tanh`）の呼び出し中に
       到達した場合。`ops.run_fused` が失敗した場合（種別を問わない）
       は、まず `ops` 自身の既存 per-op メソッドへ逐次フォールバック
       し、それも失敗した場合に限り記録済みの演算列を `eval.rs` の
       逐次呼び出し（§3.5.3）で再計算するフォールバックにより必ず
       正しい値を返す。誤った値・欠落値・`panic!` のいずれも発生
       しないため、契約 4（`get`／
       `as_slice`／`value`／`to_tensor` の非 fallible 契約が観測可能な
       意味論も含め完全不変）・契約 5（実体化失敗は必ず型付きで通知
       されるか、利用者に誤った値・欠落値を渡さない）を同時に満たす。
       フォールバックの発生は内部で観測可能にする（テスト用カウンタ。
       §6.1 #165 に記録）。
    3. **`add`／`mul`／`relu`／`exp`／`tanh`（elementwise 5 演算。
       `var.rs:122`〜`:141`・`:257` 以降）は常に自身の出力を実体化
       しないまま返す（＝遅延グラフを延長する）**（`ops` は §1 のとおり
       `Tape::new()`／`Tape::with_backend` のいずれでも必須所有値と
       して常に埋まっているため、この遅延は例外なく発生する）。
       これが本改訂で 4〜6 段連鎖を実現する主要因である。**shape 検証
       は `Tape::push` によるノード追加より前に完了し、不正な shape は
       その場で `Err` を返す**（`Var::add`／`mul` の既存の検査順序
       「①クロステープ検査 → ②shape 検査 → ③forward 値計算 →
       ④ノード記録」〈`var.rs` 冒頭コメント〉のうち、③を実体化境界へ
       遅延させるだけであり、①②の順序・即時性は変更しない）。
       **`Var::add`／`mul` が `Ok` を返すことは「shape が妥当でノードが
       記録された」ことのみを意味し、加算・乗算が計算済みであることを
       意味しない**（§1 参照）。`matmul`／`sum`／`max` は引き続き
       **返る前に自分の出力を実体化済みにする**（非 elementwise の
       実体化境界。ただし入力側が elementwise 演算の遅延グラフで
       あった場合、その入力の実体化は上記層 1 の fallible 境界として
       扱う）。
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
      フォールバック案は同じ失敗を検知したうえで `eval.rs` の逐次実行
      （決定的に成功する非 fallible な参照実装。§3.4）へ落とすだけで
      あり、追加の失敗モードを生まないため本改訂でもこちらを採用しない
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
    - **受け入れコストの解消（codex-review 第 13 波 P1-a 指摘への
      回答）**: 第 11〜12 波は「`add`／`mul` 等の fallible 演算どうしが
      直接連続する区間には融合が及ばない」ことを受け入れコストとして
      記録していたが、本改訂（`add`／`mul` の遅延化）によりこの限定は
      解消した。PoC-9 実測（`ew4`／`ew6`／`ew_fanout`。いずれも `add`／
      `mul` の連鎖・fan-out を含む構成）が示す 2.25〜3.19 倍の高速化は、
      `Tape::new()`／`Tape::with_backend` いずれの経路にも及ぶ（§6.2 の
      該当エントリは本改訂で撤回する）。
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

**配置は `tensor-core` の 1 か所に閉じる（codex-review 第 13 波 P1-b
指摘への回答。第 11〜12 波が導入した「`autodiff` 内のコア融合実行器」を
撤回し、本改訂で置き換える）**: 本節が定める融合 IR（`FusionOp`／
`FusionNode`／`FusionGraph`／`FusionPlan`）は `tensor-core` に置き、
クレート境界を越えて `BackendOps::run_fused`（§3.4）へ渡すバックエンド
非依存の中間表現として使う。**実行主体（融合カーネルの実装）は
`autodiff` に置かない**——`crates/autodiff` は演算グラフの記録・
遅延・実体化制御のみを担い、実際の計算（融合の有無を問わず）は常に
注入された `BackendOps` 実装（§4.2、`crates/backend-cpu`／
`backend-cuda`／`backend-metal` が実装する）経由に限る。**CPU 向けの
融合実行（未対応時の fail-safe な参照実装を含む）は `backend-cpu` 側
の `BackendOps` 実装（`run_fused` オーバーライド）として提供する**
——REQ-1（`docs/spec/04-requirements.md:43`・`:49`）が定めるとおり
CPU 演算カーネルの実装は `backend-cpu` の責務であり、`autodiff` に
「実体的な CPU カーネル」を新設しない（`.claude/rules/delegation-impl.md`
「`crates/backend-cpu`…REQ-2・REQ-11〜13 系」とも整合する）。

**`autodiff` 側の役割（責務分界線。第 12 波の記述をこの前提に合わせて
訂正し、第 14 波 P1 指摘を受けた本改訂で `eval.rs` の位置づけを再確定
する）**: `autodiff` が担うのは、記録済みの演算列の走査・実体化発火点
の判定・`FusionPlan::from_ops`（§3.4）によるクレート間 DTO 変換という
**制御のみ**であり、要素ごとのスカラー計算（`add`／`mul`／`relu`／
`exp`／`tanh` の数式そのもの）を独自に実装しない。`ops`（§1。
`Tape::new()`／`Tape::with_backend` のいずれでも常に埋まっている
必須所有値）の `ops.run_fused` が `BackendError::Unsupported` を
返したときの主たるフォールバックは、**同じ `ops`（注入された
`BackendOps` の実装）の既存 per-op メソッド（`ops.add`／`mul`／
`relu`／`exp`／`tanh`。§4.2）を記録順に逐次呼び出す**ことであり、
`eval.rs` を経由しない（§3.5.2 手順 3・§3.5.3）——これにより「実際の
計算は常に注入された `BackendOps` 経由」という原則を、`run_fused`
失敗時のフォールバックにおいても保つ。**例外は 1 か所のみ**: 層 2
（非fallible 境界。`Var::value`／`Var::to_tensor`）は失敗の種別を
問わず必ず正しい値を返す契約のため、`ops` の per-op メソッドさえ
失敗した場合（対応バックエンドが elementwise 未実装の場合。現状の
CUDA／Metal 等）に限り、`autodiff` 自身の `eval.rs` 参照実装への
再計算を**最終手段**として用いる（§3.5.3。層 1〈fallible 境界〉には
この最終手段はなく、`ops` の per-op メソッドも失敗すれば
`AutodiffError::Backend` として `?` で伝播する。§3.5.2 手順 4 と
同じ扱い）。この最終手段を除けば、`eval.rs` 自体への変更（可視性
変更・新規モジュール新設のいずれも）は不要である。**`eval.rs` の
役割の変化（第 13 波までとの相違点）**: 第 13 波までは `Tape::new()`
（`ops: None`）がすべての演算を `eval.rs` 経由で即時計算する「既定の
eager 経路」を持っていたため、`eval.rs` は (a) `Tape::new()` の
全演算パスと (b) 層 2 の最終手段フォールバックという 2 つの役割を
兼ねていた。本改訂は `Tape::new()` 自身が既定バックエンド（`backend-
cpu` の `CpuBackendOps`）を注入するため（§1）、(a) の役割は消滅し、
`eval.rs` は (b) 層 2 の最終手段フォールバックにのみ用いられる（層 1
では使わない。§3.5.2）。既存の TASK-1.5〜1.8 テスト資産が `Tape::
new()` を eager 実行として前提していた場合、実行経路が「常に
`ops.run_fused`／`ops` の per-op メソッド経由」へ変わるため、#164
実装時に §4 の数値一致複合判定を満たすことを再検証する必要がある
（許容誤差の緩和はユーザー承認必須。`.claude/rules/coding-rust.md`
「テスト・ベンチ」・§6.1 #164 参照）。

## 3. 遅延評価境界

### 3.1 方式

**非 elementwise 演算（`matmul`／`sum`／`max`）は返る前に自身の出力を
実体化する既定方針を変えず、融合対象区間（elementwise 連鎖）のみ内部
API でグラフを遅延構築する「明示的遅延バッファ」方式**を第一案とする。

- 全面 lazy 化（すべてのテンソル演算をグラフ構築のみに留め、明示的な
  実体化まで一切計算しない方式）は不採用とする。理由:
  1. REQ-13（起動コスト対策、`docs/spec/04-requirements.md`）の方針は
     JIT コンパイル・autotune 探索由来の起動コストを避けることにあり、
     全面 lazy 化はグラフ構築・スケジューリングの実行時オーバーヘッドを
     全演算パスへ持ち込む。融合対象区間（elementwise 連鎖）のみへ限定
     すれば、このオーバーヘッドを融合が効く範囲だけに閉じ込められる。
  2. 既存の `BackendOps` 呼び出し規約（各メソッドが `Tensor<f32>` を
     受け取り即座に `Tensor<f32>` を返す、`backend_ops.rs:63` 付近の
     契約）を全面 lazy 化は破壊する。`autodiff` 側の値計算契約も、
     本改訂（§1。`Tape::new()` 自身が既定バックエンドを注入し常に
     融合を試みる）により elementwise 5 演算は `Tape::push` と同時に
     forward 値を計算しなくなるが、`matmul`／`sum`／`max` は引き続き
     返る前に実体化済みにする（本節冒頭の既定方針）ため、全面 lazy
     化のような無制限のグラフ構築・スケジューリングオーバーヘッドの
     持ち込みには当たらない。既存テスト資産（TASK-1.5〜1.9d 等）は
     `Tape::new()` の実行経路が変わることに伴い #164 実装時に §4 の
     数値一致複合判定を満たすことの再検証を要する（§2.5「`eval.rs`
     の役割の変化」参照）が、これは全面 lazy 化とは異なる、局所的な
     再検証コストである。

### 3.2 実体化（materialization）ポイントの列挙

融合対象区間の遅延構築は、以下いずれかの条件到達時に実体化（実際の
カーネル呼び出しによる計算実行）へ切り替える。

| # | 条件 | 根拠 |
|---|------|------|
| (a) | reduction ノード（`sum`／`max`）へ到達 | PoC-9 実測で reduction エピローグは自動融合対象外（§1）。融合境界ノードとして扱う |
| (b) | `gemm` ノードへ到達 | PoC-9 実測で matmul をまたぐ融合は分断される（§1）。#203（GEMM epilogue 融合）までは境界として扱う |
| (c) | `Var::value`／`Var::to_tensor`／`Gradients::get`（非 fallible 境界）、または `matmul`／`sum`／`max`・`Tape::backward` の VJP 連鎖内部が入力側の未実体化値を必要とした時点（fallible 境界）。いずれも `autodiff` 側の materialize ヘルパー（§3.5.1〜3.5.3）に帰着する。`ops`（§1。`Tape::new()`／`Tape::with_backend` のいずれでも常に埋まっている必須所有値）を使い `FusionPlan::from_ops` を経由して `BackendOps::run_fused` を試み、`BackendError::Unsupported` は同じ `ops` の per-op メソッドへフォールバックする（§3.5.2・§3.5.3。codex-review 第 13 波 P1-b 指摘への回答により、`autodiff` 内の実行主体〈第 11〜12 波の「コア融合実行器」〉は撤回し、フォールバック先を `ops` 自身の既存 per-op メソッドへ一本化する。層 2 に限りそれも失敗した場合の最終手段として `eval.rs` を使う。§2.5「`autodiff` 側の役割」） | `Tensor`（`tensor.rs:53`）自体は `Arc<Storage<T>>` を必須で保持する既存表現のまま変更せず、`Storage<T>`（非公開）も `Pending` バリアントを持たない（§3.5.1 で確定）。したがって `Tensor::get`／`as_slice`／`contiguous`（`tensor-core` の汎用アクセサ）にも「未実体化」を表す分岐は存在しない。遅延状態は `autodiff::TapeNode`（`tape.rs`）だけが持つ（§3.5.1）。fallible 境界での実体化失敗は型付きエラーとして `?` で伝播し（層 1。`Unsupported` 以外の失敗時のみ）、非 fallible 境界での実体化失敗は per-op メソッド（必要なら最終手段の `eval.rs`）による再計算で必ず正しい値を返す（層 2。§3.5.4） |
| (d) | 連鎖長上限（4〜6 段）到達 | TASK-12.1 の内容規定（4〜6 段程度）。PoC-9 の代表ワークロード規模（`ew4`／`ew6`）とも整合する上限であり、無制限連鎖によるカーネル生成コスト・レジスタ圧の増大を避ける。上限に到達させた演算が `matmul`／`sum`／`max`（fallible）か `add`／`mul`／`relu`／`exp`／`tanh`（非 fallible）かにより (c) の層 1／層 2 いずれかへ合流する（§3.5.3・§3.5.4）。**`add`／`mul` 自身も上限到達時は自分自身のノードを実体化してから返る**ため、この場合の `Var::add`／`mul` は「shape 妥当性 + バックエンド実行結果」の両方を表すことになる（§3.5.4） |
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
- **実質的な適用箇所（codex-review 第 6 波・第 13 波 P1-a 指摘を受け
  本節を確定する。§1「演算跨ぎの遅延・二項 elementwise 演算の遅延化」
  契約に一致させる）**: 個々の `Var` 演算メソッドの呼び出し粒度自体
  （1 呼び出し 1 演算）は変更しないが、**`add`／`mul`／`relu`／`exp`／
  `tanh`（elementwise 5 演算）は常に複数回の独立した公開 `Var` 呼び
  出しをまたいで遅延（`Pending`）を持ち越せる**（`ops` は §1 のとおり
  `Tape::new()`／`Tape::with_backend` のいずれでも必須所有値として
  常に埋まっているため、この遅延は例外なく発生する。§3.5.1）ため、
  `a.add(&b)?.mul(&c)?.relu().exp().tanh()` のような現行公開 API の
  記述形そのものが 4〜6 段の elementwise 連鎖（二項・単項の混在を
  含む）を形成しうる。透過的融合の実質的な適用箇所は次の 3 箇所で
  ある: (i) elementwise の遅延グラフを、後続の `matmul`／`sum`／
  `max`（非 elementwise の fallible 演算）が入力として読み出す時点
  （窓 (a)・層 1。§3.5.2）、(ii) `Tape::backward` の VJP 連鎖内部
  （窓 (a)・層 1。§3.5.2）、(iii) `Var::value`／`Var::to_tensor`／
  `Gradients::get` が同じ遅延グラフを直接読み出す時点（窓 (a)・層 2、
  非 fallible 境界。§3.5.3）。将来追加されうる複合エントリポイント
  （窓 (b)。§3.5.5）は現時点では必須スコープ外のまま据え置く。

### 3.4 遅延グラフと `BackendOps`・`Tensor` 契約の接続

**本節の適用範囲（codex-review 第 13・14 波 P1 指摘への回答。第 11〜12
波の「`Executor::Core`／`Executor::Backend` の二経路」、第 13 波までの
「`ops: Some(ops)`（`Tape::with_backend` 経路）にのみ関与し `ops: None`
〈`Tape::new()`〉は経由しない」という限定は、いずれも本改訂で撤回し
単一経路へ統合する）**: 本節が定義する `FusionSession`／`FusionPlan`／
`FusedOpKind`／`BackendOps::run_fused` は、`Tape::new()`／`Tape::
with_backend` のどちらで構築した `Tape` の実行経路にも同一に関与する
（§1。`ops` は必須所有値であり、どちらの経路でも常に埋まっている）。
**`autodiff` クレート内で完結する独立実行器（第 11〜12 波の「コア融合
実行器」）は撤回する**——CPU 向けの融合実行（`run_fused` の CPU 実装。
未対応時の fail-safe な参照実装を含む）は `backend-cpu` 側の
`BackendOps` 実装として提供し、`autodiff` は演算グラフの記録・遅延・
実体化制御のみを担う（§2.5「`autodiff` 側の役割」）。

（PR #357 review 指摘への対応で追加した節を、codex-review 各波の指摘を
反映しつつ育ててきた。§1・§3.1〜3.3 は「透過的」「遅延構築」という
表現のみで、遅延グラフの所有場所・具体的な型・`BackendOps`
（`crates/tensor-core/src/backend_ops.rs:63`）との接続経路を規定して
いなかった。`BackendOps` の各メソッドは具体化済みの `Tensor<f32>` を
受け取り直ちに具体的な `Tensor<f32>` を返す契約であり、`Tensor`
（`crates/tensor-core/src/tensor.rs:53`）は `Arc<Storage<T>>` を必須で
保持する公開型としては変わらない。本節はこの契約と遅延グラフをどう
接続するかを明示する（窓の内側での実際の使用点の具体化は §3.5 で
行う）。旧稿（第 4 波まで）は遅延グラフを `Tensor` の `Storage` へ
埋め込み、複数回の独立した公開呼び出しをまたいで持ち越す設計を
採っていたため、`Storage` から複数箇所で共有されうる `Tensor` の
`Send + Sync` を保つための `Arc<Mutex<_>>`・`Arc<dyn BackendOps + Send
+ Sync>` 所有モデルを要求していた。持ち越す場所を `Tensor`／
`Storage` ではなく `autodiff::TapeNode`（§3.5.1）に限定するため、
`Arc<Mutex<_>>`・`Send + Sync` を要求した第 4 波の前提（遅延が
`Tensor` を経由して外部へ漏れ出すこと）は再導入しない。以下は
その帰結として単純化された契約である。

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
  - **`ops` をどの時点で・どの形で受け渡すか（codex-review 第 13・14 波
    P1 指摘への回答。第 11〜12 波の `Executor` enum、第 13 波の
    `Option<Box<dyn BackendOps>>` フィールドは、いずれも本改訂で単一の
    必須所有値 `ops: Box<dyn BackendOps>` フィールドへ置き換える）**:
    materialize ヘルパーが呼ばれるのは §3.5.2 の層 1（`matmul`／`sum`／
    `max`・`Tape::backward` の VJP 連鎖内部）・§3.5.3 の層 2
    （`Var::value`／`Var::to_tensor`／`Gradients::get`）・§3.5.4（連鎖長
    上限到達時）、または将来の複合エントリポイント（§3.5.5）であり、
    いずれも `Tape` が既に保持する非公開フィールド `ops: Box<dyn
    BackendOps>`（§1）を借用して使う。フィールド追加は `Tape` の構造体
    を非公開のまま拡張するだけであり、`pub` フィールドを持たない現行の
    `Tape`（`tape.rs:140`）の公開契約を破らない。**`Option` を採らない
    理由（第 10・14 波の懸念への回答）**: 第 10 波は `Option<Box<dyn
    BackendOps>>` を「`None` = 未解決 = 公開コンストラクタが事実上の
    融合スイッチになる」ことを理由に撤回し、第 13 波は「REQ-12 が保護
    する『利用者』は compat 層の公開面である」という範囲限定でこの懸念
    に応じた。第 14 波 P1 指摘は、この範囲限定が `autodiff` の公開 API
    自体に残る「コンストラクタ選択＝融合スイッチ」という構造そのものを
    消していないことを問題視した。本改訂は `ops` を `Option` ではなく
    必須所有値にすることで、この構造自体を型システムで排除する——
    `None` に相当する値がそもそも存在しないため、「未解決」も「融合
    スイッチとして機能するコンストラクタ選択」も構築不能である
    （§1 参照）。**`BackendOps` は `Debug` をスーパートレイトに持たない
    ため（外部実装を破壊しないための既存方針。下記「`BackendOps` trait
    定義自体は変更しない」参照）、現行の `#[derive(Debug)]`
    （`tape.rs:139`）はそのままでは `ops` フィールド追加後にコンパイル
    できない。#164 では `#[derive(Debug)]` を撤去し、手書き
    `impl fmt::Debug for Tape` へ置き換える**（`ops` の中身を表示せず
    `"Tape { .. }"` 等の固定文字列を表示する。`ops` は常に埋まっている
    ため `None`／`Some` を区別する分岐は不要であり、バックエンド実装の
    中身〈`Device` を含む〉も表示しない）。これにより `Tape: Debug`
    という公開契約自体は変更後も維持され（実装手段が `derive` から
    手書きへ変わるのみ）、公開 API 非破壊の方針
    （`.claude/rules/security.md`）を満たす。**`Tape: Send` の自動
    導出への影響（`Debug` と対になる論点として明記する）**: 現行
    `Tape`（`id: TapeId`・`nodes: RefCell<Vec<TapeNode>>`）は
    `TapeNode` の値（`Arc<Storage<T>>` を経由する `Tensor<f32>`／
    `Tensor<i32>`）が `Send` であるため `Tape: Send` が自動導出される。
    `ops: Box<dyn BackendOps>`（`Send` 境界なし）の追加により、`ops`
    に実際に格納される `BackendOps` 実装が `Send` でない場合、この
    自動導出は後退しうる。これは本改訂で新規に持ち込まれる帰結では
    なく、第 13 波の `ops: Option<Box<dyn BackendOps>>` フィールド
    追加時点で既に生じていた帰結を本改訂で明示的に記録するものである。
    `Tape` は元々 `RefCell`／`OnceCell`（§3.5.1）により `!Sync` であり
    単一スレッド内での利用を前提とする型である（§3.5.1・§3.5.3 の
    借用規律もこの前提に立つ）ため設計意図とは矛盾しないが、`pub` 型の
    auto trait 後退は公開 API 非破壊の観点で見落としやすい変化である。
    #164 実装時に `Send` 自動導出の後退が実際に生じるか（`Tape::new()`
    が注入する `backend_cpu::CpuBackendOps` が `Send` かどうか、
    `Tape::with_backend` に渡されうる実装が `Send` かどうかに依存する）
    を確認し、生じる場合は Conventional Commits の `!`／`BREAKING
    CHANGE:` 告知の要否を判断すること
    （`.claude/rules/conventional-commits.md`）。`Tape::backward`
    （`backward.rs`）・§3.5.1 の materialize ヘルパーはいずれも
    `self.ops`（`&dyn BackendOps`。`.as_ref()` で得られる借用）を読み、
    融合を試み `run_fused` を呼ぶ内部ヘルパーへ**借用として**渡す
    （§3.4 冒頭「`FusionSession` は借用 `ops: &'ops dyn BackendOps` を
    保持する」）。`FusionSession` はその呼び出しの実行中だけ生存する
    ローカル値であるため、所有権の移動・`Arc` によるクローンはいずれも
    不要である。
  - **CPU 融合実行の提供元は `backend-cpu`（codex-review 第 13 波 P1-b
    指摘への回答。第 11〜12 波が `crates/autodiff` 内に置いていた
    「コア融合実行器」を撤回する）**: `run_fused` の CPU 実装（未対応
    時の fail-safe な参照実装を含む）は `backend-cpu` 側の `BackendOps`
    実装（`run_fused` オーバーライド）として提供する。`crates/
    autodiff` は演算グラフの記録・遅延・実体化制御のみを担い、実行は
    常に注入された `BackendOps`（`ops`。§1 のとおり必須所有値）経由に
    限る。`autodiff` 内に新たなカーネル実装（第 11〜12 波の
    `fusion_exec.rs`／`run_core_fused`／`UnaryFusedOp`）は新設しない。
    `backend-cpu` 側の CPU 融合実装（SIMD・rayon 等による最適化を含む）
    の詳細設計は `backend-cpu` の担当範囲であり本文書のスコープ外と
    する（結線のみを本文書で確定する）。
    - **`autodiff → backend-cpu` の workspace path 依存を追加する
      （codex-review 第 14 波 P1 指摘への回答。第 9 波 P1 指摘が確定し
      第 13 波まで維持していた「`autodiff → backend-cpu` の workspace
      path 依存は追加しない」という契約を、本改訂で撤回する）**: 第 9
      波は「`autodiff` は具体バックエンド実装への依存を持たない」こと
      を理由にこの依存追加を禁じ、第 13 波はこれに合わせて `Tape::
      new()` を eager・非融合のまま据え置き、既定バックエンドの注入を
      compat 層（REQ-9）の責務としていた。第 14 波 P1 指摘は、この
      「compat 層が注入する」という設計自体が「`Tape::new()` と
      `Tape::with_backend(ops)` のどちらを呼ぶかが融合の有無を決める」
      という構造を温存しており REQ-12 に抵触することを指摘した。本改訂
      はこれを受け、**既定バックエンドの注入責任を compat 層から
      `autodiff::Tape::new()` 自身へ一段引き下げる**（§1・§6.2「`Tape::
      new()` が使う既定バックエンドの供給規則」）。この結果、`crates/
      autodiff/Cargo.toml` へ `backend-cpu = { path = "../backend-cpu" }`
      を追加する（deps-policy.md の許容依存 8 区分〈外部クレート〉の
      対象外であり、workspace member 間の path 依存のためユーザー承認
      フローの対象にならない。`backend-cpu/Cargo.toml` の既存コメントに
      あるとおり `tensor-core = { path = "../tensor-core" }` と同種の
      整理）。**循環依存は生じない**: 依存方向は `backend-cpu` →
      `tensor-core`、`autodiff` → `{tensor-core, backend-cpu}` であり、
      `backend-cpu/Cargo.toml`（`[dependencies]` は `tensor-core`／
      `rayon`／`half` のみ）は `autodiff` へ依存しない。**第 9 波の
      懸念への回答（撤回の実質的な理由）**: 第 9 波が守ろうとしていた
      不変条件は「`autodiff` が実際の計算をバックエンド固有カーネルへ
      直接実装しないこと」であり、「`autodiff` が `backend-cpu` という
      具体クレートの存在を一切知らないこと」自体が目的ではなかった。
      本改訂後も `autodiff` は演算の実行をすべて `tensor-core` が定義
      する `BackendOps` トレイト経由でのみ行い（§2.5「`autodiff` 側の
      役割」）、`backend-cpu` への依存は「`Tape::new()` が既定インスタ
      ンス `backend_cpu::CpuBackendOps::new()`（引数なし・
      `Result` を返さない無条件成功のコンストラクタ、
      `crates/backend-cpu/src/ops.rs:27`）を構築するため」だけに使う。
      `autodiff` にカーネルコード（数式・SIMD・並列化実装）が移動する
      ことはなく、§2.5 が定める責務分界線（`autodiff` は制御のみ、
      計算は常に `BackendOps` 経由）は変更しない。`with_backend` が
      受け取る `Box<dyn BackendOps>` は引き続きトレイトオブジェクトで
      あり、`Tape::new()` 以外の経路（CUDA／Metal 加速・テスト用実装）
      では呼び出し元が具体型を構築して渡す。
    - **`backend_ops.rs` は変更しない**: `ops_for`（`backend_ops.rs:171`。
      借用ベース・候補注入契約）は本改訂の対象外であり変更しない。
    - **compat 層の結線は不要（簡素化。第 13 波の「compat 層への既定
      注入」設計を撤回する）**: 既定バックエンドの注入が
      `autodiff::Tape::new()` 自身の責務になったため、REQ-9 の compat
      層（`compat::array`／`compat::Sequential` 等）は `Tape` 構築時に
      **`autodiff::Tape::new()` をそのまま使えばよく、独自の結線
      （`Tape::with_backend` の呼び出し・`backend-cpu` への直接依存）を
      必要としない**。compat 層が `tensor-core`／`autodiff` 以外に
      `backend-cpu` へ依存する必要もなくなる（自作コアの上の薄い
      ラッパーに徹する REQ-9 の方針とも整合する）。GPU バックエンド
      （CUDA／Metal）を明示的に使いたい場合、compat 層は引き続き
      `Tape::with_backend` を選べる（§1「バックエンドの明示選択」）が、
      これは compat 層固有の実装判断であり本文書は結線を強制しない。
    ```rust
    impl Tape {
        /// 既存の既定コンストラクタ（`tape.rs:154`）。シグネチャは
        /// 変更しない（非破壊）。既定バックエンド `backend_cpu::
        /// ops::CpuBackendOps::new()`（無条件成功のコンストラクタ）を
        /// `self.ops` へ格納する（2026-08-08 ユーザー承認に基づく。
        /// §6.2「`Tape::new()` が使う既定バックエンドの供給規則」）。
        /// 以後の実行は §3.5 の 3 層の実体化境界に従い、常に融合を
        /// 試みたうえで `ops` の per-op メソッド・`eval.rs`（層 2 限定）
        /// へ段階的にフォールバックする——`Tape::with_backend` と完全に
        /// 同一の実行方針であり、選択が変えるのは注入されるバック
        /// エンドの種類のみである（§1）。
        pub fn new() -> Tape {
            Tape {
                id: TapeId(NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed)),
                nodes: RefCell::new(Vec::new()),
                ops: Box::new(backend_cpu::CpuBackendOps::new()),
            }
        }

        /// バックエンドを明示供給するコンストラクタ（実装は #164。本節が
        /// 確定する供給契約。TASK-1.9「backend 経由の実行への置き換え」
        /// の一環として新設する。CUDA／Metal 加速や決定的なテスト用
        /// 実装の注入に使う「バックエンドの明示選択」手段であり、
        /// `Tape::new()` と同一の融合方針を共有する（§1）。
        pub fn with_backend(ops: Box<dyn BackendOps>) -> Tape {
            Tape {
                id: TapeId(NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed)),
                nodes: RefCell::new(Vec::new()),
                ops,
            }
        }
    }
    // `impl Default for Tape { fn default() -> Self { Tape::new() } }`
    // （既存実装、変更しない）は `Tape::new()` へ委譲するため、既定
    // バックエンドの注入を自動的に継承する——`Default::default()` は
    // `Tape::new()` を経由しない別の構築経路ではなく、単なる委譲先の
    // 別名である（#164 実装時の確認事項として明記する）。
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
    フレーム内で完結する。**この記述は `tensor-core` 内で `FusionSession`
    自体が使われる場合（§3.4 冒頭「`FusionSession` は `tensor-core` 内で
    `FusionGraph` が既に存在する場合のための内部機構として残す」。#162
    の連鎖検出が `tensor-core` 内で完結する将来のユースケース）に限定
    される**。`autodiff::Tape` の実体化（§3.5）はこの `FusionSession`
    を経由しない（`tensor-core` → `autodiff` の逆依存を作れないため）。
    `autodiff` 側は `FusionPlan::from_ops` + `BackendOps::run_fused` を
    直接呼び、`Unsupported` のときは同じ `ops` の per-op メソッドへ
    フォールバックする（層 2 に限りそれも失敗した場合の最終手段として
    `eval.rs` を使う。§3.5.2・§3.5.3。本節とは別の、`autodiff` クレート
    内で完結する契約である）。
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
- **まとめ（codex-review 第 13 波 P1-b 指摘への回答。本改訂で最終形へ
  確定する）**: 「遅延値を保持できる `Tensor` 表現への変更」は採らない
  （`Tensor` 不変。`Storage` にも `Pending` バリアントを追加しない）。
  融合対象区間の構築・実体化はいずれも「単一の fallible 呼び出しの
  内部だけで生存するローカル値」（`FusionValue`／`FusionSession`）と
  して行う。「連鎖全体を受け取る明示的な内部 API」として
  `BackendOps::run_fused`（非破壊拡張のデフォルトメソッド）を追加する。
  グラフの所有は `FusionSession` が `graph: FusionGraph`（所有値）と
  して保持し、実体化に使う `BackendOps` 実装は `ops: &'ops dyn
  BackendOps`（借用）として保持する。`Arc`／`Mutex`／`Rc`／`RefCell`／
  `Send + Sync` はいずれも不要である。
  `autodiff` 側の materialize ヘルパーが呼ぶ実行経路は単一である
  （第 11〜12 波の `Executor::Core`／`Executor::Backend` の二経路分岐は
  撤回する）: **`self.ops`（`&dyn BackendOps`。§1 のとおり常に埋まって
  いる必須所有値）を使い** `FusionPlan::from_ops` + `BackendOps::
  run_fused` を直接呼び（`FusionSession` は経由しない。上記
  「`FusionPlan` は `tensor-core` と `autodiff` の双方から構築される」
  参照）、失敗した場合は §3.5.2・§3.5.3 が確定する種別ごとの契約に
  従って、まず `ops` 自身の既存 per-op メソッド（`add`／`mul`／
  `relu`／`exp`／`tanh`）へ逐次フォールバックし、層 2（非 fallible
  境界）に限りそれも失敗した場合の最終手段として `eval.rs` の既存
  関数へフォールバックする（§2.5「`autodiff` 側の役割」）。`Tape::
  new()`／`Tape::with_backend` のどちらで構築した `Tape` でもこの経路
  は同一であり、経路自体に到達しない場合（遅延グラフが構築されない
  場合）は存在しない——`add`／`mul`／`relu`／`exp`／`tanh` は常に
  自身の出力を実体化せずに返す（§3.5.1）。
  いずれの発火点で呼ぶかは §3.5.2 の層 1（`matmul`／`sum`／`max`・
  `Tape::backward` の VJP 連鎖内部）・§3.5.3 の層 2（`Var::value`／
  `Var::to_tensor`／`Gradients::get`）・§3.5.4（連鎖長上限到達時。
  fallible／非 fallible いずれの経路にも合流する）、または将来の複合
  エントリポイント（§3.5.5）が決める。呼び出しは「呼び出し元の関数
  フレーム内だけで完結する」という性質（`FusionSession` について
  上記コード例のドキュメンテーションコメントが述べる性質と同じ）を保つ
  （実体化の発火点が §3.5.1〜3.5.4 の複数箇所に増えたことが変化点で
  あり、「フレーム」の粒度自体は変わらない）。`Tape::new()` は
  **承認済みの単一の既定バックエンド（`backend-cpu` の
  `CpuBackendOps`）を構造的に必ずハードコードする**（§1・§6.2）。
  これは `docs/public-api-design.md` §4.1 が定める `Device::
  available()` に基づく選択ロジックの実装ではない——`Tape::new()` は
  デバイスを列挙・比較して選ぶのではなく、承認済みの単一の既定を
  無条件に注入するだけであり、§4.1 が現在も未決事項として残す
  「CUDA を既定で有効化するかどうかの選択ロジック」とは別の論点で
  ある（§6.2 参照）。CPU 融合実行の提供元は `backend-cpu`（§2.5
  「`autodiff` 側の役割」・§3.4「CPU 融合実行の提供元は
  `backend-cpu`」）であり、`autodiff::Tape::new()` 自身がこれを注入
  するため、`Tape::new()` を呼ぶすべての経路（REQ-9 の compat 層を
  含む・含まないいずれも）に融合が常時・透過的に効く（§1）。GPU
  バックエンド（CUDA／Metal）を既定として使う規則は本文書では確定
  しない（§6.2 に未決事項として残す）。
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

（本節は codex-review 第 6 波・第 13 波 P1-a 指摘を受けて確定する。
「複数回の独立した公開 `Var` 呼び出しをまたいで遅延を持ち越す」設計を、
`value`／`to_tensor` の非 fallible 契約を壊さない形で成立させる。
第 13 波は、単項演算（`relu`／`exp`／`tanh`）に限定していた遅延対象を
`add`／`mul`（二項 elementwise 演算）にも拡張した——TASK-12.1 の中核
要件は「現行公開 API 上での 4〜6 段 elementwise 連鎖の融合」であり、
PoC-9 実測（`ew4`／`ew6`／`ew_fanout`）が示す構成は `add`／`mul` の
連鎖・fan-out を含むため、単項演算限定では対象範囲が実測条件と一致
しなかった。）

### 3.5.1 `TapeNode` の構造と遅延を許容する演算の切り分け

- 遅延状態は `autodiff::TapeNode`（`tape.rs:118`。`pub(crate)`）だけに
  持たせ、`tensor-core` の `Tensor`／`Storage<T>`（`tensor.rs:33`）は
  一切変更しない（§3.4 で確定済み）。`ops` は §1 のとおり
  `Tape::new()`／`Tape::with_backend` のいずれでも必須所有値として
  常に埋まっているため、`add`／`mul`／`relu`／`exp`／`tanh`（elementwise
  5 演算）は例外なく `value` を空の `OnceCell` のまま記録する（下記
  「遅延を許容する演算とその場で実体化する演算の切り分け」）。一方
  `matmul`／`sum`／`max`（非 elementwise）は §3.2 (a)(b) のとおり
  常に返る前に `OnceCell::from(...)` で `value` を即座に埋める。
  `Op::Leaf`（葉ノード）も同様に即座に埋める（下記「葉ノード」参照）。
  `TapeNode` を次のとおり拡張する（実装は #164）:
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
- **遅延を許容する演算とその場で実体化する演算の切り分け（codex-review
  第 13 波 P1-a 指摘への回答。単項演算限定を撤回する）**: `ops`（§1。
  常に埋まっている必須所有値）を使い、elementwise 5 演算——`add`／
  `mul`（`var.rs:122`〜
  `:141`。`Result<Var<'_>, AutodiffError>` を返す）と `relu`／`exp`／
  `tanh`（`var.rs:257` 以降。非 fallible な `fn ..(&self) -> Var<'t>`）
  ——はいずれも `Tape::push` 時に `value` を空の `OnceCell` のまま記録
  し、**自身の出力を実体化せずに返す**（遅延グラフを延長する）。これが
  4〜6 段連鎖を実現する主要因である。**shape 検証と実行を分離する**:
  `add`／`mul` は既存の検査順序「①クロステープ検査 → ②shape 検査 →
  ③forward 値計算 → ④ノード記録」（`var.rs` 冒頭コメント）のうち、
  ①②は従来どおり `Tape::push` より前に即時実行し、不正な shape は
  ノードを記録せずその場で `Err` を返す。③（`eval::add`／`mul` の
  呼び出し）だけを実体化境界まで遅延させる。**したがって `Var::add`／
  `mul` が `Ok` を返すことは「shape が妥当でノードが記録された」ことの
  みを意味し、「加算・乗算が計算済みである」ことを意味しない**（§1
  参照。バックエンド実行の失敗は次の実体化境界で初めて表面化しうる）。
  一方 `matmul`／`sum`／`max`（非 elementwise。`var.rs:111`〜`:119`・
  `:144`〜`:171`）は、**返る前に自分の出力を実体化済みにする**
  （`OnceCell` を埋めてから返す）。これらの演算は shape 検証を実体化前
  に完了できる（`shape` フィールドのみを読む）が、実際の計算には入力
  の具体値が要るため、入力が elementwise の遅延グラフであった場合は
  その実体化を自身の実行の一部として行う（§3.5.2 の「層 1」）。この
  `OnceCell` の埋め込みは §3.5.2 が定める通常分岐規定（`set` の `Err`
  は「他経路が先にこのノードを実体化済み」という fan-out に伴う二重
  到達として扱い、`panic!`／`unreachable!()` を使わず `get()` で読み
  直した既存値を採用する）にそのまま従う。
- **葉ノード（`Op::Leaf`）は常に実体化済み**: `Tape::var(&tensor)`
  （`tape.rs:164`）は呼び出し時点で既に具体的な `Tensor<f32>` を受け
  取るため、`value` を即座に `OnceCell::from(tensor.clone())` で埋めて
  push する。これにより「実体化されていないノードの入力を遡ると、
  有限回で必ず実体化済みノードまたは `Op::Leaf` に到達する」という
  帰納法の基底が成り立つ。
- **走査順が既に発生順トポロジカル順であること（DAG 一般化。codex-review
  第 13 波 P1-a 指摘への回答）**: `Tape::push`（`tape.rs:186`〜`:190`）
  は `NodeId(nodes.len())` を採番してから追記するため、あるノードの
  入力 `NodeId` は常に自分自身の `NodeId` より小さい。この事実は
  `add`／`mul` が入力 2 個を持つようになった本改訂後も変わらず、
  遅延グラフは常に非巡回（DAG）である——**ただし `add`／`mul` の遅延
  化により、遅延グラフは「単純な線形チェーン」ではなく一般の DAG に
  なる**（第 12 波までの「`relu`／`exp`／`tanh` はいずれも入力 1 個の
  単項演算だから常に線形チェーン」という記述は撤回する。例:
  `(a.add(&b)?).mul(&c.add(&d)?)?` のように、2 つの独立した `add` の
  遅延ノードが 1 つの `mul` ノードへ合流する fan-in や、同じノードを
  複数の後続演算が参照する fan-out〈§2.4 と同種の懸念。§6.2「異なる
  `Tape` を跨ぐ融合境界」参照〉が実際に生じうる）。それでも「あるノード
  の入力 `NodeId` は常に自分より小さい」という不変条件だけで、実体化
  対象ノードから入力方向へ辿る走査は常に停止し（有限回で葉ノードまたは
  実体化済みノードに到達する）、循環検出アルゴリズムは不要という結論
  自体は変わらない（`NodeId` の単調増加が非巡回性を構造的に保証する
  ため）。

### 3.5.2 層 1（fallible 境界）: 後続の fallible `Var` 演算・`Tape::backward` の VJP 連鎖内部

- **後続の非 elementwise fallible `Var` 演算**（`matmul`／`sum`／`max`）
  および `Tape::backward` の VJP 連鎖内部は、入力側の値を読む際に
  `Var::value()`（層 2・非 fallible・実体化失敗を eager フォールバックで
  必ず吸収する API。§3.5.3）を**呼ばない**。かわりに専用の内部経路
  `materialize_fallible`（`autodiff` クレート内の `pub(crate)` フリー
  関数。`tape.rs` に実装。§3.5.3 の `materialize_non_fallible` と同じ
  借用規律に合わせ、`nodes.borrow()` で得た共有借用の中身をそのまま
  受け取るシグネチャとする）を**必ず**使用する契約とする:
  ```rust
  fn materialize_fallible<'a>(
      nodes: &'a [TapeNode],
      ops: &dyn BackendOps,
      id: NodeId,
  ) -> Result<&'a Tensor<f32>, AutodiffError> { .. }
  ```
  `value()` とは別関数であり、`ops.run_fused` の失敗のうち
  `BackendError::Unsupported`（実行開始前に
  判明する能力不足）**以外**は eager フォールバックで吸収せず型付き
  エラーのまま呼び出し元へ返す（`Unsupported` の扱いは下記手順 3 を
  参照。§4 の fail-safe 方針を層 1 でも一貫させる）。実装は #164。
  `value()` 経由では層 1 が要求する `AutodiffError::Backend` の直接
  伝播に構造的に到達できないため、読み出し経路を層ごとに分離する。
  この経路は対象の `TapeNode.value` が未実体化であれば、その場で実体化
  を行う。手順は次のとおり:
  1. 未実体化ノードから入力方向へ、`OnceCell` が埋まっているノードまで
     辿り、間に挟まる `add`／`mul`／`relu`／`exp`／`tanh` の遅延部分
     グラフ（§3.5.1「走査順」。一般に DAG）を得る。
  2. 遅延部分グラフを `FusedOpKind`（§3.4）の列へ変換し（`Op::Add`/
     `Mul`/`Relu`/`Exp`/`Tanh` と `FusedOpKind::Add`/`Mul`/`Relu`/
     `Exp`/`Tanh` は 1:1 対応。§2.1・§3.4）、`FusionPlan::from_ops`
     （§3.4。`pub` + `#[doc(hidden)]`）で `FusionPlan` を構築したうえで
     `ops.run_fused(&plan, &leaves)` を試す（`ops` は §1 のとおり常に
     埋まっている必須所有値であり、この経路に到達しない場合〈`ops` が
     未解決の状態〉は存在しない）。
  3. `run_fused` が `Ok` を返せば、その融合結果をそのまま実体化値として
     用いる。**`run_fused` が `Err(BackendError::Unsupported(_))` を
     返した場合**は、遅延部分グラフの各ノードをトポロジカル順に
     **同じ `ops`（注入された `BackendOps` の実装）の既存 per-op
     メソッド**（`ops.add`／`mul`／`relu`／`exp`／`tanh`。§4.2）で
     逐次再計算する（§4 の fail-safe 方針。§2.5「`autodiff` 側の役割」
     が定めるとおり、実際の計算は常に注入された `BackendOps` 経由に
     限る）。この逐次呼び出しがさらに失敗した場合（対応バックエンドが
     elementwise 未実装〈`BackendError::Unsupported`〉の場合を含む）、
     層 1 にはこれ以上のフォールバックがないため、その `Err` をそのまま
     `AutodiffError::Backend(BackendError)` として `?` で伝播する
     （手順 4 と同じ扱い。`eval.rs` への最終手段フォールバックは層 2
     〈§3.5.3〉限定であり、層 1 では使わない）。`run_fused` の
     `Unsupported` フォールバック自体は記録済みの演算列から正しい値を
     再計算するのみであり、`Err` を呼び出し元へ流入させない（「エラー
     を `Option` へ流入させない」契約と矛盾しない。§4）。
  4. `run_fused` が `Err(BackendError::Unsupported(_))` **以外**の
     `Err`（起動失敗・メモリ割り当て失敗・転送失敗等、実行開始後に
     判明する障害。`run_fused` をオーバーライドした `BackendOps`
     〈`backend-cpu`／`backend-cuda`／`backend-metal` の融合実装〉が
     返しうる）を返した場合、**per-op メソッドへのフォールバックも
     試みず** `Err(BackendError)` をそのまま `AutodiffError::Backend
     (BackendError)` へ変換して `materialize_fallible` の戻り値として
     呼び出し元へ `?` で伝播する。すなわち **層 1（`materialize_
     fallible` の内部）では、`Unsupported` 以外の `run_fused` の失敗は
     フォールバックしない**（層 2〈§3.5.3〉との違い。層 2 はエラー
     種別を問わず常にフォールバックする。§3.5.3 参照）。手順 3・手順 4
     は互いに排他な `run_fused` の結果（`Ok`・`Unsupported`・
     `Unsupported` 以外の `Err`）に対応しており矛盾しない。既定の
     `backend-cpu`（`CpuBackendOps`）は per-op メソッドが構造的に
     成功する参照実装であるため、`Tape::new()`（既定コンストラクタ）
     を使う演算（TASK-1.5〜1.8 の既存テスト資産が使う経路を含む）は
     手順 3 のフォールバックまでで必ず成功する（#164 実装時に §4 の
     数値一致複合判定を満たすことを再検証する。§2.5「`eval.rs` の
     役割の変化」）。
  実体化が完了したら、**実体化を要求した対象ノード（手順 1 の起点。
  `matmul`／`sum`／`max` 自身の入力、または `Tape::backward` の VJP
  計算が読み出そうとした 1 ノード）自身の `OnceCell` にのみ**その場で
  `set()` する（連鎖の**途中**にある `add`／`mul`／`relu`／`exp`／
  `tanh` ノードの `OnceCell` は空のまま残る。ここで中間ノードまで
  `set()` してしまうと、まさに §2.5「`autodiff` 側の役割」で明記した
  「中間 `Tensor` 実体化の除去」という融合の利得そのものを打ち消して
  しまう。したがって同じ中間ノードを別の `Var` から後で独立に読み出す
  場合は、その都度手順 1 の走査からやり直し、遅延部分グラフを再評価
  する——結果はキャッシュされないが数値は §3.5.7 のとおり非融合 eager
  経路と一致するため、この非キャッシュは正しさに影響しない）。
  **`OnceCell::set` の `Err`（二重設定）は通常分岐として扱う
  （codex-review 第 12 波 P1-b 指摘への回答。旧稿は「既に空であることを
  直前に確認済みのため構造的に到達しない」として `Err` を
  `.unwrap_or_else(|_| unreachable!("..."))` で扱っていたが、
  `unreachable!()` はマクロ名にかかわらず実行されれば `panic!` であり
  本番経路 panic 禁止〈`.claude/rules/coding-rust.md`〉に違反する。加えて
  「構造的に到達しない」という前提自体が本設計と整合しない: 層 1 は
  `get_or_try_init`（unstable。§3.5.3 で不採用が確定済み）を使えないため
  「空チェック → 計算 → `set`」という 2 段階の手順にならざるを得ず、
  §2.4「fan-out の扱い」が既に認めているとおり同一ノードが複数の
  VJP 連鎖・複数の入力経路から共有されうる本設計では、この 2 段階の間に
  別経路が同じノードを先に実体化し `set` してしまうことは通常の走査
  結果として起こりうる。`OnceCell` は `!Sync` であり本設計はスレッド間
  競合を前提としない〈単一スレッド内での再入のみ〉ため、「競合」ではなく
  「fan-out に伴う二重到達」と言い換える）**: `set` が `Err(rejected)`
  を返した場合、それは「別経路が先にこのノードを実体化済み」を意味する
  ため、今回計算した値（`rejected`。`OnceCell::set` は失敗時に渡した
  値をそのまま返す）を破棄し、`get()` で読み直した既存値を正として使う。
  `set` 成功直後・失敗直後のいずれも `get()` は理論上必ず `Some` を返す
  契約だが、それでも `unwrap()`／`expect()` は使わず、`None` だった場合は
  他の関数（`eval.rs:88` の `build_tensor` 等）と同じ「`debug_assert!` で
  契約違反を検知しつつ安全側へフォールする」パターンに倣う——ただし
  `materialize_fallible` は参照 `&'a Tensor<f32>` を返す関数であり、
  `None` 分岐では捏造した値への参照を作れないため、安全側のフォール先は
  値ではなく型付きエラーとする。既存 variant のうち `AutodiffError::
  Backward(String)`（`error.rs:34`。「将来のグラフ不整合検出に備え先行
  定義した予約 variant」で現状どこからも構築されていない）を、この
  `OnceCell` 不変条件違反（実体化グラフの走査ロジック自体のバグ）の
  検出用途にあてる。variant 名は `Tape::backward` を想起させるが、
  `error.rs` が定める用途は「テープのグラフ不整合」であり、実体化
  グラフ走査の不変条件違反もこれに該当するため転用に矛盾はない
  （`materialize_fallible` は `Var::add`／`mul`／`matmul` 等の forward
  演算からも呼ばれるため、この `None` 分岐は理論上 forward 側からも
  `Backward` variant を返しうる。#164 実装時に `error.rs:26`〜`:33` の
  「現時点で構築箇所はまだない」コメントをこの用途に合わせて更新する）。
  `panic!`／`unwrap()`／`expect()` はいずれも使わない。これにより同じ
  ノードへの 2 回目以降の読み出しは再計算しない。
- `Tape::backward`（`backward.rs:73`。公開シグネチャ
  `pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>`
  は変更しない）は、それ自体が単一の fallible 呼び出しである。内部で
  テープを逆順に辿り各ノードの VJP（`grad.rs::vjp`。`Op` 単位のまま。
  §3.3）を計算する過程で、1 つの VJP 計算式が複数の elementwise 演算
  から成る場合（例: `tanh` の VJP `grad * (1 - y * y)` は `mul`・`sub`
  の連鎖）も、上記と同じ `materialize_fallible`（`self.ops` を借用し、
  `Some(ops)` なら融合を試みて `Unsupported` のみ `ops` の per-op
  メソッドへフォールバックし（それも失敗すれば `?` で伝播する）、
  それ以外の失敗はフォールバックせず `?` で伝播する。`None` なら
  実体化そのものが発生しない。上記手順 2〜4）を用いる。
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

### 3.5.3 層 2（非 fallible 境界）: `Var::value`／`Var::to_tensor` とフォールバック

- **`value()` と `materialize_fallible`（§3.5.2）の役割分担**:
  `Var::value`／`Var::to_tensor` は本節が定める `materialize_non_fallible`
  （`ops.run_fused` の失敗を `ops` の per-op メソッドへのフォールバック、
  さらにそれも失敗した場合は `eval.rs` への最終手段フォールバックで
  必ず吸収し `panic!` も `Err` も返さない）だけを呼び、§3.5.2 の
  `materialize_fallible`（`run_fused` の失敗のうち `Unsupported` 以外は
  型付きエラーのまま伝播する）を呼ばない。逆に `matmul`／`sum`／
  `max`・`Tape::backward` は `materialize_fallible` のみを呼び
  `value()` を呼ばない（§3.5.2）。両者は対象ノードの実体化という
  同じ責務を、失敗時の扱い（型付き伝播／段階的フォールバック）で
  排他に分担する。
- `Var::value`（`var.rs:74`。`-> Ref<'_, Tensor<f32>>`）・
  `Var::to_tensor`（`var.rs:81`。`-> Tensor<f32>`）は**シグネチャを
  一切変更しない**。対象ノードが未実体化であれば、§3.5.2 と同じ
  手順 1・2 で `run_fused` による実体化を試みるが、**`ops.run_fused`
  が `Ok` 以外を返した場合は `Err` を呼び出し元へ伝播せず**（層 1
  〈§3.5.2〉と異なり、`BackendError::Unsupported` か否かでエラー種別
  を区別しない。非 fallible な `value`／`to_tensor` は失敗の種別に
  関わらず必ず正しい値を返す契約〈契約 4・5〉のため層 2 の挙動自体
  はこの区別を持たない）、まず §3.5.2 手順 3 と同じ **`ops` 自身の
  既存 per-op メソッド**（`ops.add`／`mul`／`relu`／`exp`／`tanh`）を
  トポロジカル順に逐次呼び出して再計算を試みる（§2.5「`autodiff` 側の
  役割」）。**この per-op フォールバックさえ失敗した場合（対応バック
  エンドが elementwise 未実装の場合等、エラー種別を問わない）に限り、
  `autodiff` 自身の `eval.rs` の既存関数（トポロジカル順に逐次呼び
  出し。構造的に失敗しない参照実装）を最終手段として用いて再計算し、
  必ず `Tensor<f32>` を返す**（§2.5 で確定した層 2 限定の例外）。
  `OnceCell::get_or_init`（`&self` で呼べる、`FnOnce() -> T` の非
  fallible なクロージャを取る）にこの「融合加速を試み、失敗したら
  `ops` の per-op メソッドで再計算し、それも失敗したら `eval.rs` を
  最終手段として再計算する」処理全体を渡せばよく、`get_or_try_init`
  （unstable）は使わない。この経路は構造的に失敗しない（§3.5.1 の
  shape 検証は各演算の呼び出し時点で既に完了しており、`eval.rs` の
  各関数自身も `-> Tensor<f32>`（非 fallible）である。§3.5.4 も参照）
  ため、`Var::value`／`Var::to_tensor` は誤った値・欠落値を返すことも
  `panic!` することもない。
  ```rust
  // materialize ヘルパー（`tape.rs` 内 `pub(crate)`。イメージ）。
  // `nodes` は `self.tape.nodes.borrow()` で得た共有借用であり、
  // `get_or_init` のクロージャ内で他ノードの `value` を読む際も
  // 同じ共有借用を再利用する（`borrow_mut()` を一切要求しない）。
  fn materialize_non_fallible<'a>(
      nodes: &'a Vec<TapeNode>,
      ops: &dyn BackendOps,
      id: NodeId,
  ) -> &'a Tensor<f32> {
      nodes[id.0].value.get_or_init(|| {
          // 手順 1・2（§3.5.2）で融合加速を試み、失敗したら `ops` の
          // per-op メソッドで逐次再計算し、それも失敗したら `eval.rs`
          // を最終手段として再計算する（§3.5.4）。`matmul`／`sum`／
          // `max`／`Op::Leaf` はこのクロージャ自体が呼ばれない
          // （`value` は `Tape::push` の時点で常に埋まっている）。
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

- `add`／`mul`／`relu`／`exp`／`tanh` が `TapeNode` を push する直前に、
  自身が延長しようとしている遅延部分
  グラフの長さ（未実体化の連続する入力ノード数 + 1）を数える。§3.2 (d)
  の上限（4〜6 段。具体的な段数は #164 実装時に確定する）に達する
  場合、**その場で自分自身のノードを実体化してから返す**（遅延部分
  グラフはここでリセットされ、次の演算から新しく数え直す）。
  - `relu`／`exp`／`tanh`（非 fallible）の実体化は §3.5.3 の非
    fallible 境界（層 2）の手順（融合を試み、失敗したら段階的に
    フォールバック）を使う。
  - `add`／`mul` の実体化は §3.5.2 の fallible 境界（層 1）の手順
    （融合を試み、`Unsupported` 以外の失敗は `?` で伝播）を使う。
    **この場合の `Var::add`／`mul` は「shape 妥当性」に加えて
    「バックエンド実行結果」も表すことになる**（§1「shape 検証と
    実行を分離する」の例外——連鎖長上限到達という構造的な理由により
    実行が即時化されるため。§3.5.1 参照）。
- 一方、`matmul`／`sum`／`max`・`Tape::backward` の VJP 連鎖内部
  （層 1）が読み出しの過程で上限超過の遅延部分グラフに遭遇した場合は、
  §3.5.2 の `materialize_fallible`（融合を試み、失敗は `?` で伝播）に
  そのまま従う。
- 上限が「到達させた演算が `matmul`／`sum`／`max`（fallible）か
  `add`／`mul`／`relu`／`exp`／`tanh`（非 fallible。ただし `add`／
  `mul` は自身の実体化時のみ層 1 の失敗伝播規約も併せ持つ）かにより
  層 1／層 2 いずれかへ合流する」（§1・§3.2 (d)）とはこの意味である。

### 3.5.5 窓 (b): 将来の複合エントリポイント

- 現状の `Var` 演算 API は 1 呼び出し 1 演算の粒度であるが、§3.5.1〜
  3.5.4 の設計により elementwise 5 演算の遅延グラフに限れば複数回の
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
  の実体化条件に対応する）。elementwise 5 演算はいずれも shape 不変
  または広義の broadcast shape のみで完結し、transpose を含まないため、
  この境界条件が実際に作用するのは #163／#164 が transpose を伴う演算
  を融合対象へ拡張する場合に限られる（現状スコープでは transpose 系
  操作は遅延部分グラフに含まれない）。

### 3.5.7 フォールバック（融合失敗時）の数値面の注意

- §3.5.2・§3.5.3 のフォールバックには 2 段階ある: 主たるフォールバック
  は**同じ `ops` の per-op メソッド**（`ops.add`／`mul`／`relu`／
  `exp`／`tanh`）への逐次呼び出しであり、`eval.rs` は層 2（非
  fallible 境界）に限りそれも失敗した場合の**最終手段**として使う
  （§2.5「`autodiff` 側の役割」）。それぞれ数値面の性質が異なる。
  - **per-op メソッドへのフォールバック**（層 1・層 2 共通の主経路）:
    `run_fused` が成功していた場合と数値的に完全一致しない可能性が
    ある。**差の発生源**: `run_fused`（融合カーネル。#163）は乗算・
    加算の連鎖を 1 カーネル内で FMA 契約（`f32::mul_add`。
    `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」）や
    SIMD／rayon 並列化を用いて評価しうるのに対し、per-op メソッドは
    同じ `ops` 実装の非融合カーネルを 1 段ずつ逐次呼ぶため、中間結果の
    丸め・命令選択が異なりうる。超越関数（`exp`／`tanh`）についても
    同様に、融合カーネルとバックエンド既存の非融合カーネルとで具体
    実装が異なりうる。この差は**バックエンド間数値一致で既に許容
    されている複合判定「相対誤差 1e-3 未満 または絶対誤差 1e-5 未満」
    （§4）と同じ性質の差であり、フォールバックはこの判定の対象外側へ
    逸脱しない**。融合の有無・フォールバックの発生有無によってテスト
    許容誤差を緩和する実装は認めない（§4・§5 A08 の既存方針をそのまま
    適用する）。
  - **`eval.rs` への最終手段フォールバック**（層 2 限定）: `eval.rs`
    の逐次実行は毎回同一の決定的な参照実装（`eval::add`／`mul`／
    `relu`／`exp`／`tanh`）を同じ順序で呼ぶため、同一の演算列に対して
    常にビット単位で同じ結果を返す。この一致は `eval.rs` 最終手段の
    経路にのみ成り立つ保証であり、`run_fused` 成功時や per-op メソッド
    へのフォールバック（融合カーネル・バックエンド固有の非融合カーネル
    の実装。SIMD／rayon 並列化や FMA 契約を含みうる）との比較には
    及ばない（上記のとおり複合判定〈§4〉の対象として扱う）。**非融合の
    数値基準が必要なテスト**（例: 融合カーネルの正しさを非融合結果と
    突き合わせる #165 のテスト）は、`ops: None` という「基準となる
    非融合コンストラクタ」を使うのではなく（§1 のとおりそのような
    コンストラクタは存在しない）、`run_fused` が常に
    `BackendError::Unsupported` を返すテスト用 `BackendOps` 実装を
    `Tape::with_backend` へ注入し、per-op メソッドへのフォールバック
    結果を基準として用いる（それも失敗するよう設定すれば `eval.rs`
    最終手段の結果を基準にできる。§6.1 #165 参照）。
- **run-to-run 非決定性としての扱い**: `Tape::new()`／`Tape::
  with_backend` はいずれも `run_fused` を試みるため（§1。両者は同一の
  融合方針を共有する）、`run_fused` の成否（デバイス障害・一時的な
  リソース枯渇等）は決定的シード設定（学習系回帰テストの基本方針。
  `.claude/rules/coding-rust.md`「テスト・ベンチ」）で制御できない
  新しい非決定性の発生源に**どちらの経路でも**なりうる。学習系回帰
  テストが「バックエンド融合成功／`eval.rs` へのフォールバック」
  いずれで実行されたかに依存しない結果を要求する場合は、テスト側で
  使用する `BackendOps` 実装を **`Tape::with_backend` に決定的な
  テスト用実装を渡すことで固定する**必要がある。§6.1 #165 (i) のよう
  に**実際に `run_fused` が呼ばれバックエンド加速が発生する経路自体を
  観測・検証したいテスト**は、`run_fused` を融合実装（またはカウンタ
  付きテスト用実装）でオーバーライドした `BackendOps` を `Tape::
  with_backend` で明示供給する必要がある（既定の `Tape::new()` が
  注入する `backend-cpu` の `CpuBackendOps` 実装がどう `run_fused` を
  実装するかは `backend-cpu` 側の担当範囲であり、テストが要求する
  観測点〈カウンタ等〉を持つとは限らない）ことで、加速の発生有無自体
  を固定する必要がある（§6.2 に記録する）。

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
- **`BackendError::Unsupported` の fail-safe 契約（codex-review 第 8・
  13・14 波 P1 指摘への回答。§3.5.2・§3.5.3 の全節で一貫させる）**:
  `ops`（§1。`Tape::new()`／`Tape::with_backend` のいずれでも常に
  埋まっている必須所有値）の未実装カーネル・非対応バックエンド
  （実行開始前に判明する能力不足）が `BackendError::Unsupported`
  （`crates/tensor-core/src/device.rs:218`）を返した場合は、**同じ
  `ops` の既存 per-op メソッド**（`add`／`mul`／`relu`／`exp`／
  `tanh`）への逐次フォールバックによる fail-safe とする既存方針を
  踏襲する（`backend_ops.rs` の elementwise・reduction 未実装カーネル
  に対する既存の fail-safe 設計と同型）。**このフォールバックは融合を
  取り下げるが、実際の計算は依然として注入された `BackendOps`（`ops`）
  経由のままである**（§2.5「`autodiff` 側の役割」。第 11〜12 波は
  `crates/autodiff` 内の「コア融合実行器」がフォールバック先だったが
  その実行器は撤回済みであり、本改訂のフォールバック先は `ops` 自身の
  非融合カーネルである。融合カーネル自体のバグ検出という §6.1 #165 の
  検証意図は変わらない）。**このフォールバックは層 1（fallible 境界。
  §3.5.2 手順 3）・層 2（非 fallible 境界。§3.5.3）の両方に一貫して
  適用する**: `run_fused` のデフォルト実装（§3.4）が返す `Unsupported`
  は、呼び出し元が `matmul`／`sum`／`max` であっても非 fallible な
  `value`／`to_tensor` であっても同じ per-op メソッドへのフォール
  バックへ倒れる。一方、実行開始後に判明する障害（起動失敗・メモリ
  割り当て失敗・転送失敗等、`Unsupported` 以外の `Err`）は層 1 では
  `AutodiffError::Backend` として型付き伝播し（§3.5.2 手順 4）、層 2
  では引き続き per-op メソッドへのフォールバックで吸収する（§3.5.3。
  非 fallible API は失敗の種別を問わず必ず正しい値を返す契約のため）。
  **層 2 に限り、per-op メソッドへのフォールバックも失敗した場合
  （対応バックエンドが elementwise 未実装の場合等）、`autodiff` 自身の
  `eval.rs` を最終手段として用いる**（§2.5・§3.5.3。層 1 にはこの
  最終手段がなく、`AutodiffError::Backend` として `?` で伝播する）。
  これらのフォールバックは記録済みの演算列を正しく再計算するだけで
  あり、`Err`／`None` を `Option` へ流入させることはない（「エラーを
  `Option` へ流入させない」契約と矛盾しない）。
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
| #164（ディスパッチ統合） | §1 の「利用者向け制御 API を提供しない」方針（第 14 波改訂: 単一の公開構築経路への統合。「compat 層が REQ-12 の『利用者』面を担う」という第 13 波までの範囲限定は撤回する）・「二項 elementwise 演算の遅延化」契約（codex-review 第 6・13・14 波 P1 指摘への回答）に基づく融合対応経路の実装。§3.4 で確定した `FusionValue`／`FusionSession`（借用ベース・`Arc`／`Mutex`／`Send + Sync` 不要）・`FusionPlan::from_ops`（`autodiff` 専用のクレート間構築経路。`pub` + `#[doc(hidden)]`）／`BackendOps::run_fused`（デフォルト実装付きで trait 定義へ追加。既存メソッドの契約は変更しない）接続契約、`Tape` の非公開フィールド `ops: Box<dyn BackendOps>`（第 10・13・14 波 P1 指摘への回答。`Executor` enum・`Option<Box<dyn BackendOps>>` はいずれも撤回済み。必須所有値へ統合）と、既定バックエンド（`backend_cpu::CpuBackendOps::new()`）を注入する `Tape::new()`（既存シグネチャのまま非破壊）・バックエンドを明示供給する新規公開コンストラクタ `Tape::with_backend(ops: Box<dyn BackendOps>)` の実装（＝ TASK-1.9 の backend 経由実行への置き換えと同時実施）。**CPU 融合実行（fail-safe な参照実装を含む）は `backend-cpu` 側の `BackendOps` 実装（`run_fused` オーバーライド）として提供し、`crates/autodiff` 内に新規カーネル実装を持たない**（codex-review 第 13 波 P1-b 指摘への回答。第 11〜12 波の「コア融合実行器」〈`fusion_exec.rs`／`run_core_fused`／`UnaryFusedOp`〉・`eval.rs` の `unary`／`nan_propagating_max` の可視性変更はいずれも撤回する。`eval.rs` は非公開のまま変更せず、層 2 の最終手段フォールバックにのみ用いる）。`tensor-core` の `backend_ops.rs`（`ops_for` を含む）は変更しない。**`crates/autodiff/Cargo.toml` へ `backend-cpu = { path = "../backend-cpu" }` を追加する**（第 9 波 P1 指摘が確定し第 13 波まで維持していた「`autodiff → backend-cpu` の workspace path 依存は追加しない」という契約を、第 14 波 P1 指摘を受けた本改訂で撤回する。§3.4「`autodiff → backend-cpu` の workspace path 依存を追加する」参照。deps-policy.md の許容依存 8 区分の対象外・ユーザー承認フロー対象外の workspace member 間 path 依存であり、`backend-cpu` は `autodiff` へ依存しないため循環は生じない）。**compat 層（REQ-9）の結線は不要**（第 13 波の「compat 層が `Tape` 構築時に `backend-cpu` の `BackendOps` を注入し `Tape::with_backend` を使う」設計は撤回する。compat 層は `autodiff::Tape::new()` をそのまま使えばよい。`docs/public-api-design.md` §4.1 の改訂と対を成す）。§3.5.1 で確定した `TapeNode`（`shape: Vec<usize>` ＋ `value: OnceCell<Tensor<f32>>`）への拡張と、`add`／`mul`／`relu`／`exp`／`tanh` が常に遅延グラフを延長し `matmul`／`sum`／`max` が返る前に自身の出力を実体化する切り分けの実装。§3.5.2（層 1・fallible 境界。入力読み出しは `Var::value()` を呼ばず専用の `materialize_fallible`〈`pub(crate)`。`run_fused` の失敗のうち `BackendError::Unsupported` のみ `ops` の per-op メソッドへフォールバックし〈それも失敗すれば型付き `AutodiffError::Backend` のまま `?` で伝播する〉、`Unsupported` 以外は最初から per-op メソッドへのフォールバックを試みず型付き `AutodiffError::Backend` のまま `?` で伝播する〉のみを経由する）・§3.5.3（層 2・非 fallible 境界。`materialize_non_fallible` を経由し、融合失敗はエラー種別を問わず `ops` の per-op メソッドへの逐次フォールバックで、それも失敗した場合は `eval.rs` を最終手段として、必ず成功させる。`OnceCell::get_or_init` を使い `get_or_try_init`〈unstable〉は使わない）・§3.5.4（連鎖長上限との相互作用。`add`／`mul` が上限到達時に自身を実体化する場合は層 1 の失敗伝播規約も併せ持つ）の実装。`AutodiffError::Backend(BackendError)` variant と `From<BackendError>` 実装の追加（`Display` アーム追加を含む）。§3.5.2 の `materialize_fallible` における `OnceCell::set` の `Err`（二重設定。fan-out に伴う二重到達）は `panic!`／`unreachable!()` を使わず通常分岐として扱い、`get()` で読み直した既存値を採用する（`None` 到達時は `AutodiffError::Backward` へ安全側フォールする）。**既存テスト資産（TASK-1.5〜1.8）の再検証**: `Tape::new()` が eager から常時融合実行（既定 `backend-cpu` 経由）へ変わることに伴い、既存テストが §4 の数値一致複合判定を満たすことを再検証する（許容誤差の緩和はユーザー承認必須。`.claude/rules/coding-rust.md`「テスト・ベンチ」）。 |
| #165（テスト） | §1・§2.3 の transpose 非融合フォールバック、§2.4 の fan-out 融合、§3.3 の autodiff 契約（VJP がノード単位のまま変わらないこと）の検証、**§1「単一の公開構築経路への統合」の検証**（`Tape::new()`〈既定バックエンド `backend-cpu`〉・`Tape::with_backend(ops)`〈明示供給バックエンド〉のいずれで構築した `Tape` でも、演算列が §3.2 の判定を満たす限り融合が試みられること〈`run_fused` を融合実装でオーバーライドしたカウンタ付きテスト用 `BackendOps` を `Tape::with_backend` に渡して検証する〉、両者の数値結果が数値一致複合判定〈§4〉を満たすこと、`run_fused` が実際に呼ばれたことをカウンタで確認すること、`Tape::new()` を直接呼ぶ経路・compat 層経由で構築した `Tape` のいずれも利用者の明示指定なしに融合されること〈両者とも同一の融合方針を持つことの確認〉、非融合の数値基準が必要なテストは `run_fused` が常に `BackendError::Unsupported` を返す〈またはさらに per-op メソッドも失敗する〉テスト用 `BackendOps` を `Tape::with_backend` へ注入して基準を固定すること〈`ops: None` という非融合コンストラクタは存在しないため、この基準固定手段を正規の代替として文書化する〉を検証する）、**§3.5「演算跨ぎの遅延と 3 層の実体化境界」の検証**（codex-review 第 6・8・13 波 P1 指摘への回答）: (i) 独立した公開 `Var` 呼び出しをまたぐ `add`／`mul`／`relu`／`exp`／`tanh` の混在連鎖（例: `x.add(&y)?.relu().mul(&z)?.exp().tanh()`。二項・単項混在の 4〜6 段）が単一の評価呼び出しへ融合されること（`run_fused` の呼び出し回数が 1 回だけであることを確認する。`run_fused` が `Unsupported` を返す設定では `ops` の per-op メソッドへのフォールバック結果が、`run_fused` が常に `Unsupported` を返すよう固定した非融合基準〈上記〉と同一入力に対し数値一致複合判定〈§4〉を満たす値を返すことも検証する）、(i') **shape エラーが記録時に当該演算の `Err` として即時返ること（実行は遅延済みのまま）**: 不正な shape の `add`／`mul` 呼び出しがその場で `Err(AutodiffError::Shape(_))` を返し、テープにノードが記録されないこと（§1「shape 検証と実行を分離する」の検証）、(ii) 層 1（fallible 境界。§3.5.2）での融合失敗の種別ごとの分岐: (ii-a) `run_fused` が `BackendError::Unsupported` を返した場合は `ops` の per-op メソッドへフォールバックし、後続の `matmul`／`sum`／`max` が `Ok` を返すこと（値は数値一致複合判定〈§4〉を満たす）、(ii-b) `run_fused` が `Unsupported` 以外の `Err` を返した場合は、それを引き起こした後続の演算自身の `Err(AutodiffError::Backend)` として直接返ること（キャッシュ経由の遅延表面化が発生しないこと）、(iii) 層 2（非 fallible 境界。§3.5.3）での融合失敗時（エラー種別を問わない）、`Var::value`／`Var::to_tensor` が `panic!` せず、`ops` の per-op メソッド（それも失敗する設定では `eval.rs` の最終手段）へのフォールバックで計算した値と融合が成功していた場合の値が数値一致複合判定〈§4〉を満たすこと（フォールバックは値の正しさを保証するのみで #163 の融合カーネル自体のバグを隠さないことの検証。フォールバック発生をテスト用カウンタで観測できることも確認する）、(iv) `x.value()` で得た `Ref` を保持したまま別の未実体化 `Var` の `value()`／`to_tensor()` を呼んでも panic しないこと（§3.5.3「`value()` が `Ref` を保持している最中…」の検証）、(v) `Tape::backward` の VJP 連鎖内部で融合が発生する場合（§3.5.2）に、`Unsupported` 以外の失敗では `Tape::backward` が `AutodiffError::Backend` を返すこと、`Unsupported` の場合は `ops` の per-op メソッドへのフォールバックにより成功すること、per-op メソッドが構造的に成功する既定の `backend-cpu` を使う場合は常に成功すること、成功時は `Gradients::get` がそのまま非 fallible に値を返せること、(vi) §3.5.4 の連鎖長上限に到達した場合に fallible／非 fallible いずれの経路でも遅延グラフがその場でリセットされ、後続の演算が正しい実体化済み値を入力として使えること、(vii) §2.4 の fan-out・fan-in（`(a+b)*(c+d)` 形の合流。§3.5.1「走査順」の DAG 一般化）が単一の融合グラフ構築で正しく解決されることの検証、(viii) **TASK-1.5〜1.8 の既存テスト資産が `Tape::new()` の実行経路変更後も §4 の数値一致複合判定を満たすことの回帰確認**（#164 の再検証をテストとして固定する） |
| #203（GEMM epilogue 融合） | §3.2 (b) の `gemm` 境界を bias／activation epilogue まで拡張する設計変更 |

### 6.2 未決事項（スコープ外）

- **`Tape::new()` が使う既定バックエンドの供給規則
  （承認記録を含む。codex-review 第 8・10・13・14 波 P1 指摘への回答）**:
  - **承認記録（承認内容と本改訂の結論を分けて記録する）**: 2026-08-08、
    AskUserQuestion によりユーザーが**承認した内容**は「既定バックエンド
    を `Device::Cpu`（`backend-cpu` 実装）とすること」である（この
    一文は過去の波から変更しない）。**本改訂（第 14 波 P1 指摘への
    回答）の結論**（この承認に基づき本文書が導出した具体的な結線）は、
    「`autodiff::Tape::new()` 自身が既定で `backend-cpu` の
    `CpuBackendOps` を注入する（`ops` フィールドへ格納する）」という
    ものである——承認は「既定バックエンドを CPU にする」という結論を
    与え、本改訂はその結論を「`autodiff` クレート自身の既定コンスト
    ラクタが担う」という結線の形に落とし込んだ（第 11〜14 波の設計
    変遷を経て到達した形。経緯は下記）。記録場所は本文書（本エントリ）
    および `docs/public-api-design.md` §4.1（後述の改訂。同文書側も
    承認内容と本改訂の結論を分けて記載する）。**GPU バックエンド
    （CUDA／Metal）を既定として使う規則はこの承認の対象外**であり、
    別途 REQ-2 の 27 組再検証後にユーザー承認を得て決定する（未決事項
    として本エントリの末尾に残す）。
  - **経緯**: 第 10 波は `Tape::new()`（autodiff 単体）の既定解決先を
    `Device::Cpu` 固定の `tensor_core::reference_cpu_ops()` とする案を
    確定していたが、CPU 演算実装の配置がクレート責務境界と衝突する点
    を理由に第 11 波で撤回し、`crates/autodiff` 内で完結する「コア
    融合実行器」（`Executor::Core`）へ置き換えた。codex-review 第 13
    波 P1-b 指摘は、CPU 融合実行の主体を `autodiff` に置くこと自体
    （コア融合実行器という設計）を撤回するよう求めた。第 13 波はこれ
    を受け、CPU 融合実行の提供元を `backend-cpu` の `BackendOps` 実装
    へ戻したうえで、その注入を **compat 層**（REQ-9 の公開面。
    `autodiff` 単体ではない）の責務として再配置した——これにより
    `autodiff::Tape::new()` は `ops: None`（eager・非融合）のまま
    据え置かれ、既定バックエンドの注入は compat 層が `Tape::
    with_backend` を呼ぶことで行う設計だった。**しかし codex-review
    第 14 波 P1 指摘は、この第 13 波の設計自体が「`Tape::new()`
    （非融合）と `Tape::with_backend(ops)`（融合）の選択」という
    構造を `autodiff` の公開 API に残し続けており、REQ-12「利用者向け
    融合制御 API を提供しない」に抵触することを指摘した**。本改訂は
    これを受け、既定バックエンドの注入責務を compat 層から
    `autodiff::Tape::new()` 自身へ一段引き下げる。**`ops` フィールドも
    `Option<Box<dyn BackendOps>>` から必須所有値 `Box<dyn BackendOps>`
    へ変更し**（§1・§3.4）、`Tape::new()`／`Tape::with_backend` の
    どちらで構築しても同一の融合方針（常に融合を試みる）を持つよう
    統合した。**この結果、2026-08-08 の承認が対象としていた
    「`backend-cpu` の `BackendOps` を既定で注入する」という結線は、
    第 13 波が想定した compat 層ではなく `autodiff::Tape::new()`
    自身が行使する**（承認は無効化されたのではなく、適用層を再度
    訂正したうえで行使する）。
  - **本改訂後の帰結**: `autodiff::Tape::new()` は承認済みの単一の
    既定バックエンド（`backend-cpu` の `CpuBackendOps`）を**構造的に
    必ず**注入する。これは `docs/public-api-design.md` §4.1 が定める
    `Device::available()` に基づく既定デバイス選択ロジック（複数の
    候補から動的に選ぶロジック）の実装ではない——`Tape::new()` は
    デバイスを列挙・比較して選ぶのではなく、承認済みの単一の既定を
    無条件にハードコードするだけであり、§4.1 の既定デバイス選択
    ロジック不採用方針とは交差しない（§1・§3.4 参照）。`Tape::new()`
    を呼ぶすべての経路（REQ-9 の compat 層を含む・含まないいずれも）
    に CPU 上での融合が既定で・透過的に効く。
  - **未決事項として残る部分**: **CUDA／Metal をバックエンド加速として
    既定にする規則は本文書では確定しない**（承認対象外。理由は引き
    続き 2 点: (i) `docs/public-api-design.md` §4.1「既定デバイス選択
    ロジックは…実装しない」という本リポ全体の確立済み方針、(ii)
    REQ-2 が「CUDA 既定有効化の構成決定」を未検証のまま残している
    〈`docs/spec/04-requirements.md` REQ-2 受け入れ基準〉ため、GPU
    加速の既定供給規則の確定には REQ-2 の 27 組再検証後の別途ユーザー
    承認が必要）。GPU 加速バックエンドの既定供給規則の確定は #164
    以降、ユーザー承認を得たうえで別途検討する。
- **`backend-cpu` の最適化実装の結線層（codex-review 第 9・13・14 波
  P1 指摘への回答）**: 第 9 波は「CPU 候補を `ops_for` へ注入する結線
  機構・担当層」を未決事項として記録していた。第 13 波はこの結線層を
  compat 層として確定していたが、**本改訂（第 14 波 P1 指摘への回答）
  はこの結線層を `autodiff::Tape::new()` 自身へ変更する**——第 11〜12
  波が検討していた「`autodiff` 内で完結する既定実行器」案（結線層が
  既定パスには不要になるという想定）とは異なり、本改訂は「`autodiff`
  が計算そのものを実装する」のではなく「`autodiff` の既定コンスト
  ラクタが `backend-cpu` という具体クレートへの依存を持ち、既定
  インスタンスを構築するだけ」という限定的な結線である（§2.5
  「`autodiff` 側の役割」は変更しない。計算は常に `BackendOps` 経由）。
  **`crates/autodiff` は `backend-cpu` への workspace path 依存を持つ**
  （codex-review 第 9 波 P1 指摘が確定し第 13 波まで維持していた
  「`autodiff → backend-cpu` の workspace path 依存は追加しない」と
  いう契約を、本改訂で撤回する。撤回の理由は §3.4「`autodiff →
  backend-cpu` の workspace path 依存を追加する」参照）。compat
  クレート（または compat を実装するクレート）は独自の結線を必要と
  せず、`autodiff::Tape::new()` をそのまま使えばよい（§3.4「compat
  層の結線は不要」）。
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
- **同一テープ内での遅延グラフの合流（fan-in）は正規サポート対象、
  異なる `Tape` を跨ぐ合流は既存検査により構造的に排除される
  （codex-review 第 6・13 波 P1 指摘を受け、本改訂で最終確定する）**:
  PR #357 review 再指摘 P1-1／P1-2 が対象としていた「独立した
  `Pending` 同士が二項演算で合流する」というシナリオ（`(a + b) *
  (c + d)` が独立した 2 つの遅延グラフをまたぐケース）は、第 5 波
  以前は `add`／`mul` 自身が遅延化されない設計だったため発生せず、
  第 11〜12 波（`relu`／`exp`／`tanh` の単項連鎖限定）でも `add`／
  `mul` が遅延グラフを延長しないため発生しなかった。**本改訂（§1・
  §3.5.1「切り分け」）は `add`／`mul` 自身も遅延化したため、この
  合流シナリオは同一 `Tape` 内で実際に生じる**: `(a.add(&b)?).mul(
  &(c.add(&d)?))?` のように、それぞれ独立した遅延ノードを持つ 2 つの
  `Var`（`a+b` のノードと `c+d` のノード）が `mul` の 2 入力として
  合流する fan-in である。これは欠陥ではなく §3.5.1「走査順」で
  DAG へ一般化した設計がそのまま対応する対象であり、`FusionPlan::
  from_ops`（§3.4）を構築する際に両オペランドの遅延ノードをそれぞれ
  独立にグラフへ取り込めばよい（§2.4 の fan-out と対称の関係にある。
  #162／#164 の連鎖検出・グラフ構築アルゴリズムが扱う通常のケース
  として実装する）。**別テープ間の合流**（`check_same_tape`／
  `var.rs:87`〜`:92` が既に拒否する `AutodiffError::TapeMismatch`）は
  これと別の懸念であり、遅延グラフは常に単一の `Tape` の `nodes:
  RefCell<Vec<TapeNode>>` の内部だけで構築されるため（§3.5.1 の走査順
  の前提「あるノードの入力 `NodeId` は常に自分より小さい」は同一
  `Tape` 内でのみ意味を持つ）、別テープのノードが遅延グラフに紛れ込む
  ことはない（`check_same_tape` は演算入口で shape 検証より前に呼ばれる
  既存の検査であり、本改訂でも変更しない）。
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
- **フォールバック（§3.5.3・§3.5.7）が持ち込む run-to-run 非決定性**:
  `Tape::new()`／`Tape::with_backend` のいずれも `run_fused` を試みる
  ため（§1）、`run_fused` の成否はデバイス障害・一時的なリソース枯渇
  等の環境要因に左右されうる（既定の `backend-cpu` を含む）。決定的
  シード設定（`.claude/rules/coding-rust.md`「テスト・ベンチ」）だけ
  では「バックエンド融合が成功したか `ops` の per-op メソッドへ
  フォールバックしたか（さらに `eval.rs` の最終手段まで倒れたか）」を
  再現できない場合がある。各経路の結果は §4 の数値一致複合判定を
  満たすことを #165 で検証するが（§6.1 #165 (iii)）、学習系回帰テスト
  が bit-exact に近い再現性を要求する場合は、テスト側で使用する
  `BackendOps` 実装を**決定的なテスト用実装を `Tape::with_backend`
  に渡すことで固定する**（詳細・理由は §3.5.7「run-to-run 非決定性
  としての扱い」を参照）。#164 の実装ガイドとして記録する。
- **フォールバックは融合カーネル（#163）自体の正しさを保証しない**:
  §3.5.2・§3.5.3 のフォールバック（`ops` の per-op メソッド、層 2 限定
  の `eval.rs` 最終手段のいずれも）は `run_fused` が失敗した場合に
  利用者へ正しい値を返すための安全網であり、`run_fused` が誤った値を
  「成功」として返す不具合（#163 の融合カーネル生成バグ）を検出・
  防止するものではない。#165 はフォールバック経路の値と融合成功時の
  値を突き合わせるテスト（§6.1 #165 (iii)）を融合カーネルの
  正しさの検証としても位置付け、フォールバックの存在を理由にこの
  突き合わせテストを省略しない。
