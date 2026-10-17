# DDP 初期形のホスト側 collective の設計記録（#3143）

> 本 doc は、DDP 初期形（std のみの TCP によるホスト側 ring all-reduce。spec `:386-393`）の設計判断を、実装前に具体値まで確定する記録である。後続 #3144〜#3149 は本記録に従う。所有者判断 D-A（推奨形を記録に書き、そのまま実装へ進む）と D-B（DDP は実装へ進める）に基づく（§11）。

- 基準コミット: origin/main `0d2b22c9`。`docs/spec` ポインタ `7534c9c93bd4443bc13b1a6ab91c7a2db4c2e033`
- 参照する spec の行（`docs/spec/04-requirements.md`）: `:363`・`:377-385`・`:386-393`。行番号は submodule 更新でずれうるため、参照時に再確認すること
- コード変更なし・依存追加なし。`docs/spec` と既存の決定記録は編集しない

## 0. 結論

| 論点 | 推奨形 |
|---|---|
| 1 配置 | クレートは増やさない。`crates/autodiff/src/distributed/` に実装し、facade に `fandhe_ai::distributed` を新設する（§2） |
| 2 ランク発見 | ランク 0 を集合点（rendezvous）にする。起動情報は明示的な設定値のみ（環境変数の読み取りは初期形では公開しない）。world_size の上限は 64（§3） |
| 3 リング・手順 | 固定メッセージ集合・長さ前置きのフレーム。reduce-scatter／all-gather 各 W−1 ステップ。加算順序はチャンクごとに回転した固定順の左結合（§4） |
| 4 障害時 | 全ブロッキング操作にタイムアウトを設け、型付きエラーで終了する。一度失敗した communicator は poisoned。復旧は範囲外（§5） |
| 5 数値契約 | 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。tolerance・baseline は変更しない。ランク方向は固定順の f32 加算（§6） |
| 6 待受の安全性 | 既定は loopback。受信長は確保の前に検証し、ランク番号は `0..world_size` を検証する。認証・暗号化は無い（§7） |

- 所有者へ戻す論点は無い
- **Could への格上げは宣言しない**（spec `:393`）
- 検証は**単一ノード複数プロセス**に限る（§1）

## 1. 位置づけ・前提

| 時期 | 出来事 | 出典 |
|---|---|---|
| 2026-09 | 段階 0 確定・案 A〜E 比較 | #1628（`docs/facade-multi-gpu-ddp-decision.md`） |
| 2026-10-08 | DDP 格上げ条件表（NCCL 前提） | #2074・#2612（`docs/ddp-grade-up-conditions.md`。spec `:377-385`） |
| 2026-10-12 | 初期形（案 C）を spec へ反映 | #3142・fandhe-ai-spec#85（`docs/ddp-host-collective-spec-proposal.md`） |
| 本記録 | 設計の確定 | #3143 |
| 後続 | トランスポート・ランク発見／ring all-reduce／optimizer 結線／facade 公開面／CPU 複数プロセス検証／GB10 実測 | #3144／#3145／#3146／#3147／#3148／#3149 |

初期形の条件 6 項目（`docs/ddp-host-collective-spec-proposal.md` §6）と本記録の対応は次のとおり。

| 条件 | 本記録 |
|---|---|
| loopback 既定 | §7 |
| 長さ上限を割り当て前に検査 | §4・§7 |
| ランク検証 | §3・§7 |
| タイムアウト | §5 |
| 重複／想定外接続の拒否 | §7 |
| 汎用 RPC なし | §4 |

**検証範囲**: 手元の機材は GB10（GPU 1 基）と Mac mini（M4）で、複数 GPU 機は無い。検証は単一ノード複数プロセス（CI では CPU の loopback、GB10 では同一 GPU を共有する 2 プロセス）に限る。非 loopback（複数ノード）の経路は実機で検証しない。

## 2. 論点 1: 配置

- モジュールを `crates/autodiff/src/distributed/` に新設する（例: `mod.rs`・`transport.rs`〈フレーム・接続〉・`rendezvous.rs`〈ランク発見〉・`ring.rs`〈all-reduce〉）。`autodiff` の `lib.rs` に `pub mod distributed` を置く。facade 側には公開モジュール `fandhe_ai::distributed` を新設する（#3147）
- 想定クレート 10 個は増やさない
- 理由
  - #3146 が結線する `DeviceParamStore` は autodiff にある（`crates/autodiff/src/optim/device_store.rs:1976` の `backward`、`:3942` の `param_grads_to_host`）
  - facade が唯一の公開面であり、内部クレートを素通しで出さない方針と合う
- 依存は `std::net` のみ（`TcpListener`・`TcpStream`・`connect_timeout`・`set_read_timeout`・`set_write_timeout`）。cfg 分岐は要らず、`build-no-cuda-toolkit` の不変条件にも影響しない
- 検討した代替: autodiff に trait（例 `GradReducer`）だけを置き、TCP 実装を facade に置く案。ソケットを autodiff に入れずに済むが、#3146 の単体テスト（CPU の `DeviceParamStore` を 2 スレッドの loopback で動かす）が autodiff 内で完結しなくなるため推奨しない。#3146 の実装中に不都合が見つかった場合はこの代替へ切り替えてよい

## 3. 論点 2: ランク発見と起動情報

- **ランク 0 を集合点にする**。不採用: ランクごとの固定ポート（並列 CI テストでポートが衝突する。想定外接続の検査点が分散する）
- 起動情報は明示的な設定値のみ: world_size・rank・集合点アドレス・非 loopback を許可する opt-in・各種タイムアウト・セッション識別子（`u64`）。環境変数の読み取り（`from_env` 相当）は初期形では公開しない（後から追加しても非破壊）。プロセスの起動と環境変数の受け渡しは呼び出し側の責務で、#3148 は独自の環境変数名で子プロセスへ渡す
- 手順
  1. ランク 0 が集合点アドレスで待ち受ける。ポート 0 を許可し、実際に bind したアドレスをピア待ちの前に取得できる API にする（空きポートを先に調べる TOCTOU 競合を避ける）
  2. 各ランクはリング用の待受を自分の bind アドレスのポート 0 で開き、ランク 0 へ接続して Hello を送る。Hello は固定長で、magic・プロトコル版・world_size・rank・セッション識別子・リング用ポートを含む
  3. ランク 0 は全ランクの Hello を検証して集め、各ランクのリング用アドレスの表を全ランクへ送る。表の IP は自己申告ではなく、ランク 0 から観測した接続元 IP を使う
  4. 各ランクは次のランク `(r+1) % W` へ接続し、前のランク `(r-1) % W` からの接続を受ける。リング接続でも Hello を交わし、期待したランクであることを検証する
  5. 全ランクが揃ったら集合点の待受を閉じる
- world_size の上限は **64**。`world_size == 1` は通信せず、ソケットも開かない

## 4. 論点 3: リング構成・all-reduce・チャンク分割・加算順序

- メッセージは固定集合（Hello／ランク表／reduce-scatter チャンク／all-gather チャンク）に限る。汎用 RPC は持たない
- フレーム: 長さ前置き。ヘッダは magic・プロトコル版・メッセージ種別・ステップ連番（`u64`）・総要素数（`u64`）・チャンク番号・セグメントのオフセット・セグメント長。続けてペイロード（`f32` little-endian）
- チャンク分割: 総要素数 n を W 個に分ける。`base = n / W`・`rem = n % W` とし、先頭の `rem` 個を 1 要素多くする
  - n = 0: 空のまま正常に返す
  - n < W: 空チャンクが出るが、ヘッダは送る
  - 1 チャンクは最大 **4 MiB** のセグメントに分けて送る。受信側は確保済みの出力バッファへ直接読み込み、確保量がピアの申告値に左右されない
- reduce-scatter: W−1 ステップ。ステップ s でランク r はチャンク `(r − s) mod W` を次のランクへ送り、受信側は「受け取った部分和 + 自分の値」を次のステップへ回す
- **加算順序**: チャンク c は、ランク `c, c+1, …, c+W−1 (mod W)` の順に左結合で加算する（`((x_c + x_{c+1}) + x_{c+2}) + …`）。完了するのはランク `(c−1) mod W`。ランク番号順ではなくチャンクごとに回転した順序だが、トポロジーが固定なので run 間で決定論的になる。#3145 の参照実装は「同じ回転順序の逐次加算」と定義し、bit 一致で検証する
  - 不採用: 全ランクで all-gather してからランク順に加算する案（帯域とメモリが W 倍。reduce-scatter／all-gather の構成からも外れる）
- 平均: 完了したチャンクの合計を、所有ランクが `sum / (W as f32)` で 1 回だけ割る（逆数の乗算は使わない。W ≤ 64 は f32 で正確）
- all-gather: W−1 ステップで、所有ランクが確定した値をビット単位で複写して回す。全ランクが bit 単位で同じ結果を持つ

## 5. 論点 4: 障害時の挙動

- 一部のランクが落ちる、または応答しない場合でも、すべてのブロッキング操作はタイムアウトで必ず戻り、型付きエラーを返す
- 推奨の既定値（#3144 で設定可能にする）: 接続 10 秒、集合の確立全体 60 秒、1 回の読み書き 30 秒、all-reduce 1 回の全体期限。エラーを検出した側は即座に全ソケットを shutdown し、隣のランクが EOF／RST を受けてエラーがリングを伝わり全ランクが早く終了するようにする
- エラー型: autodiff 内部に `#[non_exhaustive]` の型付きエラー。variant の例は、設定不正・範囲外ランク・重複ランク・world_size 不一致・プロトコル違反（magic・版・長さ・連番・総要素数の不一致）・タイムアウト・ピア切断・I/O・poisoned。facade には内部 enum をペイロードに持たない自己完結型（`OnnxError` と同方針。例 `DistributedError`）を出し、具体名は #3147 で確定する
- 一度エラーを返した communicator は poisoned に遷移し、以降の呼び出しは即座に拒否する（`DeviceParamStore` の poisoned と同型）
- 範囲外: 復旧・再接続・ランクの再参加・エラスティック学習
- 本番経路で `unwrap()`／`expect()` を使わない

## 6. 論点 5: 数値契約・決定論性

- 判定: N ランクの分散学習で k ステップ進めたパラメータを、単一プロセスで同じ初期値・同じ全バッチにより学習した結果と比べる。判定式は統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。**tolerance・baseline・FMA 契約は変更しない**。新しい baseline も作らない
- 比較が意味を持つ前提（必須）
  - loss は平均で縮約する
  - 各ランクの局所バッチの大きさが等しい（W が全バッチ数を割り切る）。標本数による重み付けは範囲外
  - 全ランクのパラメータ初期値が同一（同じシード。決定的シードユーティリティを使う）。ランク 0 からの配信メッセージは初期形では持たない
  - BatchNorm の running stats の同期は範囲外
  - Adam 等の optimizer 状態は、パラメータと勾配が全ランクで同一なので同期しなくても一致する
- f64 アキュムレータ: ランク方向は W ≤ 64 の短い軸で、`.claude/rules/coding-rust.md` が対象にする「正規化統計・勾配の長軸縮約」に当たらない。よって**固定順の f32 加算を設計とし、f64 アキュムレータは適用しない**。これは実測ではなく設計判断であり、#3145 の bit 一致テストと #3148 の統一複合判定で検証する。将来 f64 の部分和へ切り替える場合はワイヤ形式が変わるためプロトコル版を上げる
- 決定論性の範囲
  - 同じ入力・同じ W なら collective の結果は run 間で bit 一致する。W が異なれば bit 一致しない
  - W = 1 は無通信の素通しで、既存のステップと bit 一致する（#3146 の受入条件）
  - エンドツーエンドの bit 一致は各ランクの backward の決定性に依存する。CPU は `crates/autodiff/src/determinism.rs` の棚卸しで決定的と確認されているが、GPU 経路は未検証（`docs/autodiff-determinism-mode-design.md`）
- NaN／inf は IEEE 754 のまま伝播させ、collective 側では特別扱いしない
- #3146 への申し送り: backward → `param_grads_to_host`（登録順）→ 1 本の連続バッファへ平坦化 → all-reduce（平均）→ 分割 → optimizer のステップ。all-reduce が失敗したらどのパラメータも更新しない（fail-closed）。既存ステップ関数のシグネチャは変えない

## 7. 論点 6: 待受の安全性

- 既定は loopback のみ（`127.0.0.1`／`::1`）。それ以外で bind するには明示的な opt-in（設定値の bool）が必要
- opt-in が無い場合、次も loopback に限る（opt-in の迂回を防ぐ）: **接続先**として使うランク表の全アドレス、accept した接続の接続元
- 受信長の検証（A03）
  - 各ランクは `ParamLayout` から各セグメントの期待長を受信前に知っている。受信長は期待長との**完全一致**と絶対上限（4 MiB）以下の両方で検証する
  - 検証は**確保の前**に行い、受信は確保済みの出力バッファへ読み込む
  - ヘッダの連番・総要素数が一致しなければプロトコル違反として拒否する
- ランク番号の検証（A03／A04）
  - `0..world_size` の範囲・world_size の一致・重複なしを、処理の前に検証する
  - 形式は正しいが範囲外・重複・world_size 不一致の Hello は設定誤りとして、集合の確立を型付きエラーで失敗させる（fail-closed）
  - magic・版・セッション識別子が合わない、または長さが不正な接続は、その場で閉じて受付を続ける。ただし期限は延ばさない
  - 全ランクが揃ったあとの集合点への接続や、リング接続の 2 本目は拒否する
- 認証・暗号化は無い。初期形は信頼できるネットワーク（既定は同一ホストの loopback）の中でだけ使う。セッション識別子は別ジョブとの混線防止用で、認証ではない。非 loopback の opt-in では、なりすましと盗聴のリスクを利用者が受け入れる（`docs/facade-multi-gpu-ddp-decision.md` §6 を引き継ぐ）

## 8. 公開形の推奨（#3147 への申し送り）

- `fandhe_ai::distributed` 配下に最小限で置く: 設定型（builder 形式。構築時に world_size・rank・アドレス・opt-in を検証し、範囲外は型付きエラー）・communicator（または初期化関数）・分散ステップの入口
- 追加のみで既存シグネチャは変えない。enum は `#[non_exhaustive]`。1 文 1 行の再エクスポート規約に従う
- `crates/facade/tests/api_surface.rs` に `distributed` の保留ガードは無いため、#3147 で正のガードを追加する
- doc テストは外部通信をしない形（world_size 1 か `no_run`）にする
- `docs/compat-api-scope.md` §5 への適用記録は #3147 が書く（本記録では変更しない）

## 9. セキュリティ観点（OWASP）

| 観点 | 対応 | 節 |
|---|---|---|
| A03 入力検証 | 受信長は確保前に完全一致＋上限で検証。ランク・world_size・連番・総要素数を処理前に検証 | §4・§7 |
| A04 設計（DoS 耐性） | 全ブロッキング操作にタイムアウト。重複・想定外接続を拒否。固定メッセージ。world_size ≤ 64 | §3・§5・§7 |
| A05 設定ミス | 既定は loopback。非 loopback は bind と接続先の両方で opt-in | §7 |
| A07／A08 認証・整合性 | 認証なし。信頼できるネットワーク限定。セッション識別子は認証ではない | §7 |

## 10. 不変事項・スコープ外

- 変更しない: tolerance・baseline・FMA 契約・依存ピン・`docs/spec`・既存の公開 API
- 対象外: NCCL 経路（#2613。据え置き）・GPU 間の直接転送・FSDP／ZeRO・tensor／pipeline 並列・汎用の分散 RPC・スケーリング効率の測定（条件 (f)）・復旧と再接続・パラメータの配信・DistributedSampler 相当・BatchNorm の同期・`from_env`・複数ノードでの実機検証

## 11. 承認・前提

- D-A（推奨形を記録に書き、そのまま実装へ進む）・D-B（DDP は実装へ進める）: #3018 issuecomment-6111643437、`docs/compat-api-scope.md` §5 の #3020 適用記録
- 上記以外の承認は得ていない。所有者へ戻す論点は無い

## 12. 出典

- spec `docs/spec/04-requirements.md` `:363`・`:377-385`・`:386-393`（fandhe-ai-spec#85）
- `docs/ddp-host-collective-spec-proposal.md`・`docs/ddp-grade-up-conditions.md`・`docs/facade-multi-gpu-ddp-decision.md`・`docs/autodiff-determinism-mode-design.md`
- `crates/autodiff/src/optim/device_store.rs:1976`・`:3942`、`crates/autodiff/src/determinism.rs`
- イシュー #3141〜#3149
