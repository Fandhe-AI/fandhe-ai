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
  2. 各ランクはリング用の待受を自分の bind アドレスのポート 0 で開き、ランク 0 へ接続して Hello を送る（#3144 実装時に順序を「接続 → その接続の `local_addr` IP で待受 → Hello」へ改めた。§13）。Hello は固定長で、magic・プロトコル版・world_size・rank・セッション識別子・リング用ポートを含む
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

## 13. 追補: リング待受アドレスの導出と期限の扱い（#3144 実装時・2026-10-17）

本節は #3144（PR #3409）の実装レビューで判明した類型と、採った設計を記録する。§3 手順 2 の順序（「リング用待受を開いてから集合点へ接続」）は本節で改める。§0〜§12 のそれ以外の判断は変えない。

### 13.1 類型: 配布アドレスと待受アドレスのずれ

ランク表の IP は §3 手順 3 のとおり「ランク 0 が集合点で観測した接続元 IP」である。当初の実装はこれとは別の規則（設定値 `ring_bind_ip`。既定は集合点と同族の loopback）でリング待受の bind IP を決めていたため、両者が一致しない組み合わせが次々に見つかった。1 件ずつ塞ぐと次の組み合わせが指摘される構造だった。

| 指摘 | ずれ方 |
|---|---|
| 3・4 | 集合点 `[::1]` で、リング待受は `127.0.0.1`、配布は `[::1]:ring_port` |
| 5・6 | 集合点 `127.0.0.2`（Linux）への接続元は通常 `127.0.0.1`。待受の指定と観測接続元が OS の送信元選択次第でずれる |
| 7 | 集合点 `[::ffff:127.0.0.1]` を loopback と判定する関数と判定しない `IpAddr::is_loopback` が混在 |
| 8 | 非 loopback 集合点で opt-in だけ付けると、配布は LAN IP、待受は loopback のまま |

### 13.2 採った設計: 待受 IP を観測される IP そのものから導く

- **ランク 0 以外**: 先に集合点へ接続し、その接続ソケットの `local_addr().ip()` にリング待受を bind してから、そのポートを Hello で申告する。`local_addr` は集合点側が `peer_addr` として観測する IP と同一なので、配布されるアドレスには必ず待受がある
- **ランク 0**: 集合点の待受 IP（ピアが実際に到達した IP）にリング待受を bind し、自分のエントリはリング待受の `local_addr` そのものとする
- **IPv4-mapped**: 設定時（`RendezvousConfig::new`）と観測時（ランク 0 の記録・ランク 0 以外の bind）の双方で `to_canonical` により IPv4 へ正規化する
- **設定値 `ring_bind_ip` は廃止した**（公開は #3147 のため内部 API の変更で足りる）。ずれた指定という状態自体が作れない
- **受信側の検証（fail-closed）**: ランク 0 以外はランク表を受けたら、接続前に次を検証する。自ランクのエントリが `(自分のリング待受 IP, ポート)` と一致すること・ランク 0 のエントリの IP が設定した集合点の IP と一致すること（不一致は `ProtocolViolation::RankTableAddressMismatch`）、unspecified と重複エントリが無いこと（`MalformedRankTable`）、opt-in なしなら全エントリが loopback であること（`NonLoopbackNotAllowed`）。ずれがあっても後の connect をタイムアウトまで待たせず、確立前に型付きエラーで止める
- **配布できない集合点の拒否**: マルチキャスト・ブロードキャストと IPv6 リンクローカル（ランク表がスコープ ID を運べない）は、opt-in の有無に関わらず `validate` が `InvalidConfig` で拒否する。unspecified の集合点は従来どおり拒否する
- **非 loopback** は従来どおり `allow_non_loopback(true)` の明示 opt-in でだけ許す。opt-in があればリング待受の追加設定は要らない
- **前提**: ランク間に NAT が無いこと（loopback か信頼できる同一ネットワーク。§1・§7 の範囲）。NAT があると観測 IP と `local_addr` がずれるが、上記の自エントリ検証が確立前に検出する
- **前提（到達性）**: 各ランクが集合点へ到達するのに使った IP に、ほかのランク（とくにリング上の前のランク）も到達できること（単一のネットワークセグメント）。複数のインターフェースを持つホストで集合点への経路と隣接ランクへの経路が異なる構成は対象外で、その場合は黙って成功せず、前のランクの connect がタイムアウトして型付きエラー（`Timeout`）で終わる（fail-closed）
- ワイヤ形式（Hello はポートのみ申告）とプロトコル版は変えない

| ケース | 結果 |
|---|---|
| `127.0.0.1` | 全ランクの待受が観測 IP 上にある（構造上成立） |
| `127.0.0.2`（Linux） | ランク 0 は `127.0.0.2`、ほかは OS が選んだ送信元 IP（通常 `127.0.0.1`）で待受し、その IP が配布される |
| `[::1]` | 全ランク IPv6 で待受・配布 |
| `[::ffff:127.0.0.1]` | IPv4 へ正規化して `127.0.0.1` と同じ |
| 非 loopback（opt-in あり） | 各ランクの LAN IP で待受・配布 |
| 非 loopback（opt-in なし） | ソケット作成前に `NonLoopbackNotAllowed` |

### 13.3 期限の扱い（指摘 1・2 と総点検）

- 期限の算出は `Instant::checked_add` のみとし、`Duration::MAX` 等の巨大値は `validate` が `InvalidConfig` で拒否する（`establish` 側でも checked で再算出し panic しない）。1 通の Hello の読み取り期限は `min(io_timeout, 全体期限)` で、加算できなければ全体期限で打ち切る
- accept・connect・read・write の各ループは、反復の先頭で絶対期限を検査する。accept が成功し続けても、不正な接続が届き続けても期限は延びない
- 1 本の不正な接続で集合全体を失敗させない: accept 前に RST した接続（BSD／macOS の `ConnectionAborted`・`ConnectionReset`）と、受理直後のソケット設定に失敗した接続は捨てて受付を続ける（総点検で追加）
- 受信長は本体の確保・読み取りの前に上限と期待長の完全一致で検証する（§4・§7 のまま。フレーム層に変更なし）

### 13.4 テスト（`crates/autodiff/src/distributed/`）

- 正のプローブ（リングが確立し、`to_next` の接続先 ＝ 次ランクの待受、隣接とフレームを bit 一致で往復）: `127.0.0.1`（W=2〜4）・`[::ffff:127.0.0.1]`（W=2・3）・`[::1]`（W=2・3。IPv6 が無効な環境では bind が型付き I/O エラーになることを検査する分岐）・`127.0.0.2`（Linux 限定。macOS の lo0 は既定で `127.0.0.1` のみ）・非 loopback（opt-in あり。UDP の connect で送信せずに得たローカルの非 loopback IP がある環境に限る。取れない CI では到達性の宛先が無く、外部通信なしの前提に反するため設定段の判定だけを検査する）
- 拒否: opt-in なしの非 loopback、自エントリ・ランク 0 エントリの不一致、重複・unspecified エントリ、マルチキャスト・ブロードキャスト・リンクローカルの集合点
- 期限: `Duration::MAX` の各タイムアウトが型付きエラー、集合点と accept への接続の洪水が期限で終わること

## 14. #3145 実装時の追補（ring all-reduce）

実装は `crates/autodiff/src/distributed/ring.rs`（`RingCommunicator`・`ReduceOp`・`CollectiveOptions`・`chunk_ranges`・`reference_all_reduce`）。§0〜§13 の判断は変えない。ワイヤ形式・プロトコル版・tolerance・依存は不変。

- **加算順序の食い違いの解決**: イシュー #3145 の作業 3 は「ランク番号順に固定」と書くが、本記録 §4 は「チャンク c はランク `c, c+1, …, c+W-1 (mod W)` の順（チャンクごとに回転した固定順）」と定める。**§4 に従う**。reduce-scatter の構成から自然に出る順序で、ランク番号順にするには §4 が不採用とした「all-gather してからランク順に加算」案が要る。どちらも run 間で決定論的なので数値契約（§6）の論点は生じない。参照実装 `reference_all_reduce` は同じ回転順の逐次加算（`部分和 + 新しい値` の左結合）で、テストは bit 一致で検証する
- **送受信の並行化**: 各ステップは送信を spawn したスレッド、受信を現在のスレッドで同時に行う（`thread::scope`・`unsafe` なし。送受信チャンクは常に別なので `split_at_mut` で借用を分ける）。順番に行うと 4 MiB 級のセグメントがソケットの送信バッファを超えたとき全ランクが書き込みで止まる。片側が失敗したら相手側のソケットを shutdown して早く戻し、最悪でも全体期限で終わる。根本原因のエラー（先に失敗した側）を返す
- **全体期限**: `CollectiveOptions { io_timeout, op_timeout }`（既定 30 秒・120 秒。0 と巨大値は `InvalidConfig`）。`RendezvousConfig` は変えない。#3147 が facade の設定型から両方を組み立てる
- **n = 0（W ≥ 2）**: 「空のまま正常に返す」を、長さ 0 のセグメントでリングを 1 周させてから `Ok` を返す、と解釈する。連番が全ランクでそろい続け、別ランクが n ≠ 0 で呼んだ場合も型付きのプロトコル違反になる（タイムアウト待ちにならない）。n < W の空チャンクも同様にヘッダだけ送る
- **W = 1**: 通信せず入力を変えない。`Mean` でも割らない（`x / 1.0` は signaling NaN を quiet 化して bit が変わりうるため。#3146 の「W = 1 で既存ステップと bit 一致」を守る）。平均は `W ≥ 2` で所有チャンクを `W as f32` で 1 回だけ割る
- **Err 時の `buf`**: 不定で、communicator は `Poisoned`（両ソケットを shutdown）。#3146 は `Ok` のときだけ勾配を書き戻して「失敗したらどのパラメータも更新しない」（§6）を満たす
- **`op` はワイヤに載らない**: 全ランクが同じ `op` で呼ぶのは呼び出し側の契約（ワイヤ形式を変えないため検出しない）。要素数の不一致は検出する

## 15. #3146 実装時の追補（勾配平均の optimizer ステップへの結線）

- **構造**: AMP 結線（`device_store/amp.rs`）と同じ形にした。`param_grads_to_host` → 平坦化（登録順＝`layout[i].offset` 順）→ `all_reduce(Mean)` → 分割 → `Gradients::synthetic` → 既存 `step*`。更新フェーズ（CUDA Graph capture・`pending_backup` 復元・`poisoned` 遷移・optimizer 状態の確定）は変更も複製もしない。合成 `Gradients` は `resident_fingerprint = None` で `any_resident == false` 経路になる
- **公開入口はフリー関数**: `distributed::step::{step_sgd, step_adam, step_adamw}`。facade は `DeviceParamStore` を再エクスポートしているため、inherent method に `&mut RingCommunicator` を取らせると内部型が facade の公開面へ漏れる（§5 の「facade には自己完結型を出す」に反する）。facade への公開は #3147。内部ヘルパは `DeviceParamStore::mean_reduce_grads`（`pub(crate)`。`optim/device_store/distributed.rs`）
- **エラー型**: `DistributedStepError { Collective(DistributedError), Backend(BackendError), GradientLengthMismatch }`。`AutodiffError` は facade から再エクスポートされるため variant を足さない
- **W = 1**: ヘルパは `Ok(None)` を返し、元の `grads` で既存 `step*` を呼ぶ（読み出しも通信もしない。resident 経路を含め bit 一致）
- **失敗時の状態**: all-reduce が `Err` なら `step*` を呼ばないので、パラメータも optimizer 状態も不変。store は poisoned にしない（デバイス側は未変更）。`pending` は消費しない（呼び出し側が `abandon_pending_forward` でクリアする）。communicator は §5 の契約どおり `Poisoned`。通信前に平坦化長を `total_numel` と照合し、不一致は `GradientLengthMismatch`（ピア検出に頼らないローカルの fail-closed）
- **ローカルの事前失敗の非対称性**: 勾配欠落などで通信に入る前に失敗したランクがあると、他ランクは all-reduce の中で `op_timeout` により `Timeout`／`PeerDisconnected` を受け、更新せずに戻る。ハングはしない既知の挙動で、早期通知（abort API）は範囲外。all-reduce 成功後に既存 `step*` が失敗するケース（一方のランクだけのバックエンド障害による発散）も範囲外
- **対象 optimizer**: SGD／Adam／AdamW（AMP と同じ）。RmsProp／Adagrad／LAMB は範囲外。SGD は通信前に `SgdConfig::validate()` を呼ぶ。Adam／AdamW の検証は `step_adam_impl` から抽出しない（全ランク対称に失敗するため）
- **f64 アキュムレータは使わない**（§6 の再確認）: ランク方向は W ≤ 64 の短い軸で「長軸縮約」に当たらない。tolerance・baseline・FMA 契約は不変
- **テスト**: `MockDeviceOps`（autodiff は backend-cpu に依存できない）で、W = 1 の bit 一致（resident あり・なし）、2 ランク半バッチと全バッチの統一複合判定での一致とランク間 bit 一致、ピア切断時の非更新・poisoned・ハングなし、平坦化順序を検証する。モックは `adam_step_device` を持たないため Adam／AdamW は `Unsupported` の素通しのみ確認し、数値検証は #3148 に委ねる

## 16. #3147 実装時の追補（facade 公開面 `fandhe_ai::distributed`）

実装は `crates/facade/src/distributed/`（`mod.rs`・`config.rs`・`error.rs`・`communicator.rs`・`step.rs`）。§0〜§15 の判断は変えない。**承認は §11 の D-A・D-B のみ**で、それ以外の承認は得ていない。依存・tolerance・baseline・`docs/spec/`・ワイヤ形式・既存の公開シグネチャは不変（追加のみ）。

### 16.1 確定した公開名（5 型 + 3 関数）

公開パスは `fandhe_ai::distributed` のみ（クレートルートへは再エクスポートしない。サブモジュールは非公開）。`mod.rs` の `pub use` は承認形 4 文（1 文 1 行）。

| 公開名 | 役割・要点 | 内部型との対応 |
|---|---|---|
| `DistributedConfig`（`builder`・`world_size`・`rank`・`rendezvous_addr`・`is_non_loopback_allowed`） | 検証済みの設定 | `RendezvousConfig` + `CollectiveOptions`（`pub(crate)` で保持） |
| `DistributedConfigBuilder`（`allow_non_loopback`・`session_id`・`connect_timeout`・`establish_timeout`・`io_timeout`・`op_timeout`・`build`） | `build()` が world_size・rank・アドレス種別・ポート・タイムアウトを**ソケットを開く前に**検証する。`io_timeout` は確立時と all-reduce の両方に適用。未指定のタイムアウトは内部既定（接続 10 秒・確立 60 秒・io 30 秒・op 120 秒）をそのまま使う | `RendezvousConfig::new`／`validate`・`CollectiveOptions::validate` |
| `Rendezvous`（`bind`・`local_addr`・`establish`） | §3 手順 1 の 2 段構成（ポート 0 の実アドレスを先に取得して TOCTOU を避ける） | `Rendezvous` |
| `Communicator`（`init`・`rank`・`world_size`・`is_poisoned`） | 確立済みの ring ハンドル。`Send` | `RingCommunicator` |
| `DistributedError` | 自己完結の型付きエラー（`#[non_exhaustive]`） | 下表 |
| `step_sgd`／`step_adam`／`step_adamw`（`(&Tape, &mut DeviceParamStore, &Gradients, &Config, &mut Communicator)`） | §15 と同型のフリー関数。委譲とエラー写像だけ | `distributed::step::step_*` |

公開シグネチャに `fandhe_ai_autodiff::Tape`・`RingCommunicator`・`RingLinks`・内部の `Rendezvous`／`RendezvousConfig`／`CollectiveOptions`／`ReduceOp`／`ProtocolViolation`／`DistributedError`／`DistributedStepError` を出さない。`From` impl は公開せず、写像は `pub(crate)` の `from_inner`／`from_step` に置く。

### 16.2 エラー写像

| facade の variant | 内部 |
|---|---|
| `InvalidConfig`・`WorldSizeOutOfRange`・`RankOutOfRange`・`NonLoopbackNotAllowed`・`DuplicateRank`・`WorldSizeMismatch`・`UnexpectedRank`・`Timeout`・`PeerDisconnected`・`Io`・`Poisoned` | 同名の `DistributedError` variant |
| `Protocol { message }` | `Protocol(ProtocolViolation)` の `Display`（固定文と数値だけ。ピアのバイト列は含まない） |
| `Backend(BackendError)` | `DistributedStepError::Backend`（`BackendError` は facade の既存公開型） |
| `GradientLengthMismatch` | `DistributedStepError::GradientLengthMismatch` |
| `Other { message }` | 内部 enum が `#[non_exhaustive]` のため必須になる wildcard 腕の受け皿。現行の全 variant は上記へ写り、単体テストで `Other` に落ちる variant が 0 件であることを保証する |

`DistributedStepError::Collective(e)` は `e` に上の写像を再帰適用する。`Display` は固定文・数値・内側エラーの `Display` だけで、`Io` は `ErrorKind` のみを出す（OWASP A09）。

### 16.3 `Communicator::init` のポート 0 拒否

`world_size >= 2` でランク 0 の集合点ポートが 0 の構成は、ピアが実ポートを知る手段が無く必ずタイムアウトするため、`init` は `InvalidConfig` で拒否する（構築段で止める。OWASP A04）。ポート 0 を使うときは `Rendezvous::bind` → `local_addr` → ピアへ伝達 → `establish` を使う。`world_size == 1` は無通信なので拒否しない。

### 16.4 不採用（所有者へ戻す論点。後から追加しても非破壊）

決定記録に形が無いため本イシューでは公開しない。汎用の `all_reduce` と `ReduceOp`（損失・メトリクスの平均等）・`MAX_WORLD_SIZE` の公開・`from_env`・RmsProp／Adagrad／LAMB／AMP の分散ステップ・abort API・`Tape` の inherent method としての入口。

### 16.5 セキュリティ前提（公開 doc に明記）

認証も暗号化も無い（信頼できるネットワーク内に限る）。`session_id` は誤接続の検出用で認証ではない。既定は loopback で、非 loopback は `allow_non_loopback(true)` の明示 opt-in だけ。アドレスは `SocketAddr` のみ（DNS 解決を持ち込まない）。all-reduce が失敗したらパラメータを更新しない（fail-closed。公開契約として doc とテストで固定）。

### 16.6 ガードとテスト

- `crates/facade/tests/api_surface.rs`: `distributed_is_public_and_submodules_stay_private`・`facade_exposes_distributed_items_only_in_approved_shape`（`pub use` 4 文の完全一致・`distributed/` の完全公開項目インベントリ・型ごとの `pub fn` 集合）とその自己テスト `..._detects_each_category`・`distributed_unapproved_paths_are_absent`（ルート再エクスポート・内部 `distributed` の再エクスポート・シグネチャ内の内部型・`unsafe` の不在）・`distributed_items_are_reachable_via_facade_distributed_path`（関数ポインタ型での署名固定・`Communicator: Send`・wildcard 付き網羅 `match`）・`distributed_usage_doctest_is_present_and_compiled`（world_size 1 の実行例と world_size 2 以上の `no_run` 例）。`GRAD_SCALER_PROBE_MODULES` を 17 件にし、`facade_pub_use_leaves_are_not_modules` の小文字葉 allowlist に `step_*` を追加。`distributed` の保留ガードは存在しなかったため正ガードの新設である。43 個の `*HoldDoctestGuard` doctest に `use fandhe_ai::distributed::*;` を追加した（`models` 追加時と同じ手当て）
- `crates/facade/tests/distributed_facade.rs`: 設定拒否の各ケース（world_size 0／65・rank 範囲外・unspecified・マルチキャスト・タイムアウト 0／`Duration::MAX`・ランク 1 のポート 0・opt-in なしの非 loopback）・`init` のポート 0 拒否・world_size 1 の既存ステップとの bit 一致（SGD／Adam／AdamW）・2 ランク（同一プロセス 2 スレッド・`127.0.0.1:0` 限定）の全バッチ単一更新との統一複合判定での一致とランク間 bit 一致・ピア切断時の非更新・`is_poisoned`・再利用拒否
- `crates/facade/src/distributed/error.rs` の単体テスト: 内部の全 variant が専用 variant へ写ること
- doc テストは外部通信をしない（world_size 1 は実行、world_size 2 以上は `no_run`）

実機（CUDA／Metal）依存は無く `#[ignore]` は不要。CPU 複数プロセスでの検証は #3148、GB10 実測は #3149。
