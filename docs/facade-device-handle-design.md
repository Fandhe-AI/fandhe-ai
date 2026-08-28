# デバイスハンドル再利用の公開 API 設計（イシュー #931）

イシュー #931「デバイスハンドル再利用の公開 API 設計」の設計判断文書。受け入れ条件
（1. 設計文書の作成、2. 数値一致契約・fail-fast・facade 1 クレート境界の維持、
3. spec 側更新の要否判断）に対応する。**本イシューはコード変更を伴わない設計判断
イシューであり、本文書が唯一の成果物である**（実装計画 §2「やらないこと」）。

## 1. 背景・実測根拠

フレームワーク横並びベンチ（PR #915、`scripts/bench/framework-compare/results/summary.md`、
計測 2026-08-28）の DGX Spark GB10 実測で、fandhe-ai の CUDA GEMM 中央値が行列サイズに
ほぼ非依存の 440〜460 ms 帯に張り付いた（同ファイル「環境 2」節）。candle / Burn は
デバイスハンドルを呼び出し間で使い回す API 設計のため、同一プロトコルでも初期化を
繰り返さない。この差が本イシューの起点である。

その後の調査・対応の経緯（本イシューの計画立案後、実装セッション中に merge 済みで
あることを確認した。§2 で詳述）:

- イシュー #925/#944: `bench-harness` に「デバイス/tape 再利用モード」（`--mode reuse`）
  を追加し、fresh（毎回 `tape_for` を呼ぶ）と reuse（1 回だけ `tape_for` した `Tape` を
  使い回す）の差分を計測した
- イシュー #926/#945: `docs/perf/cuda-tape-init-cost-diagnosis.md` として、`tape_for`
  自体ではなく `matmul` 呼び出し時の遅延初期化（`CudaBackendOps::gemm` 内部の
  `CudaDevice::new` + `CudaGemm::new`）が計測区間に含まれる機構を診断した
- イシュー #929/#946（本文書執筆時点で最新の main、コミット `008d381`）: 上記診断を
  踏まえ、`crates/backend-cuda/src/context_cache.rs` としてプロセス内・`ordinal` キーの
  常駐キャッシュを実装し、`CudaBackendOps` の全メソッドをこの経由に結線した

したがって本文書が判断すべき問いは、計画立案時点（#929 未着手時点）とは前提が変わって
いる。**「facade に新規公開型（デバイスハンドル）を追加すべきか」を、#929 がすでに
（公開 API を変えずに）同種の効果を狙って実装済みという事実を踏まえて再評価する**のが
本文書の実質的な内容になる。

## 2. 現状診断

### 2.1 fresh/reuse 比較が明らかにした構造的事実

`scripts/bench/framework-compare/results/summary.md`「環境 3」節（イシュー #925、
RTX 3060、#929 着手前の計測）は次を実測した:

- 単に `tape_for` の呼び出し回数を 1 回に減らす（reuse モード: 同一 `Tape`
  ＝同一 `CudaBackendOps` インスタンスを使い回す）だけでは、2 回目以降の `matmul`
  呼び出しも 260〜506 ms の固定コストを毎回支払い続けた
- これは「`tape_for` の呼び出し回数を減らせば解決する」という素朴な仮説
  （facade 層に薄いハンドルを追加し、その保持個数を減らすだけの案）を実測が
  否定したことを意味する。原因は `CudaBackendOps::gemm`（当時の実装）が
  インスタンスの同一性に関わらず**メソッド呼び出しごとに** `device_handle()` →
  `CudaDevice::new` と `CudaGemm::new`（NVRTC コンパイル 8 本 + `load_module`/
  `load_function` 8 回）を都度実行する構造だったためである
  （`crates/backend-cuda/src/ops.rs` 現行コメント、`docs/perf/
  cuda-tape-init-cost-diagnosis.md` §2 参照）

この事実は本文書の設計判断に直接効く。**「facade 公開面に `DeviceHandle` 型を追加し、
利用者にその生存期間中インスタンスを握らせる」という素朴な案 A だけでは、
`CudaBackendOps` 自身が呼び出しごとに資源を再構築する現行構造を変えない限り
何の改善にもならない**。つまり案 A が効くためには、結局 `CudaBackendOps`（または
その内部）が `Arc` 等で状態を跨いで共有する実装が必要であり、これは案 B（バックエンド
内部常駐化）と同型の実装作業を要求する。案 A は単独では成立せず、常に案 B を前提とする。

### 2.2 #929/#946 の実装内容（すでに merge 済み）

`crates/backend-cuda/src/context_cache.rs`（コミット `008d381`）は `ordinal` を
キーとするプロセスワイドな `HashMap<usize, Arc<Mutex<Option<Arc<T>>>>>` キャッシュを
`OnceLock` で保持し、次を共有する:

- `cached_device`（`CudaDevice`。`CudaContext::new` + `default_stream` + name/CC 取得）
- `cached_gemm`／`cached_elementwise`／`cached_rmsnorm`／`cached_softmax`
  （各カーネルスイート。NVRTC コンパイル + `load_module` を内包）

`crates/backend-cuda/src/ops.rs` は現在この経由へ全面的に結線済みである
（`device_handle`: `ops.rs:67-69`、`gemm`: `ops.rs:328`・`ops.rs:427`、
elementwise: `ops.rs:100`・`ops.rs:122`、rmsnorm: `ops.rs:184`、softmax: `ops.rs:232`。
いずれも `context_cache::cached_*(self.ordinal, ...)` 呼び出し）。

`facade::resolve_ops`（`crates/facade/src/lib.rs:64` 付近のコメント）はこの結線を
前提に「同一プロセス内で 2 回目以降に `tape_for(Device::Cuda(_))` を呼んでも
`CudaContext` 生成・NVRTC コンパイルは再実行されない」ことをすでにドキュメント化
している。**すなわち、facade の公開 API（`tape`／`tape_for`）を一切変更せずに、
2.1 の「呼び出しごとの都度構築」という構造的問題は解消済みである**。

### 2.3 未確認事項（正直な記録）

- `docs/perf/cuda-tape-init-cost-diagnosis.md` の実測（§6）は本文書執筆時点で
  **未実測**（実装セッションから DGX Spark GB10 へ到達不能）のまま。したがって
  「440〜460 ms のオーバーヘッドが #929 によって実際にどこまで削減されたか」の
  実機での定量値は本文書では確認できない。本文書が言えるのは「#929 は 2.1 で
  特定した構造的原因（呼び出しごとの都度構築）を機構として解消した」という
  設計上の主張であり、削減幅の実機再計測は §7 のスコープ外・後続項目とする
- `scripts/bench/framework-compare/` は承認済みピン構成の独立 workspace であり
  本イシュー・#929 のいずれも変更していない（`docs/framework-compare-harness-decision.md`
  の変更禁止方針どおり）。したがって #929 適用後の fresh/reuse 再計測は現時点で
  未実施

### 2.4 Metal 側も同型の対応が完了済み（イシュー #930/#948）

本文書の執筆後、イシュー #930（診断 #927 が特定した Metal 側の固定オーバーヘッド
〈約 5 ms・N 非依存〉の常駐化）が PR #948 で実装され、docs/931 ブランチへも
base 取り込み済みである。`crates/backend-metal/src/context_cache.rs` が
`crates/backend-cuda/src/context_cache.rs`（#929/#946）と同型のプロセス内
常駐キャッシュを提供し、`crates/backend-metal/src/ops.rs`（`MetalBackendOps`）は
各メソッド呼び出しで `context_cache::cached_context`／`cached_gemm`／
`cached_elementwise`／`cached_rmsnorm`／`cached_softmax` を経由するよう
書き換わっている（`ops.rs` 冒頭コメント「イシュー #930 で常駐化完了」参照）。

`Device::Metal` は ordinal を持たない単一 variant のため、CUDA 側の
`HashMap<usize, Arc<T>>`（ordinal キー）とは異なり、`OnceLock<Mutex<Option<Arc<T>>>>`
による型ごと単一エントリのプロセスワイドシングルトンで足りる設計となっている
（`context_cache.rs` 冒頭コメント）。fail-fast 契約（ミス時の構築失敗はキャッシュ
しない）・生存期間（プロセス生存期間中 evict されない）は CUDA 側と同じ方針を
踏襲しており、したがって CUDA・Metal の両バックエンドで 2.2 の問題は解消済みと
なった。

`CpuBackendOps`（`crates/backend-cpu/src/ops.rs:25-32`）は unit struct で毎回の
構築コストを持たず（GPU コンテキスト・NVRTC/シェーダコンパイルに相当する重い
初期化がそもそも存在しない）、本設計判断の対象外である。

## 3. 要件・制約の整理（維持すべき不変条件）

以下は #929 適用前後で変えていない・変えるべきでない不変条件であり、本文書の
「受け入れ条件 2」に対応する検証項目でもある。

- **facade 1 クレート境界**: `Device` → 具体 `BackendOps` の結線ロジックを持つのは
  `facade` クレートのみ（`crates/facade/src/lib.rs` 冒頭コメント）。`resolve_ops` の
  シグネチャ・`tape`／`tape_for`（`lib.rs:135`・`lib.rs:147`）は #929 前後で無変更
- **REQ-12（任意 `BackendOps` 注入経路を公開しない）**: `facade` の公開面は
  `tape`／`tape_for`／newtype `Tape` のままであり、`crates/facade/tests/api_surface.rs`
  （ソース走査による機械固定）も無変更。#929 の常駐化は `backend-cuda`
  クレート内部（`context_cache` は非公開モジュール）に閉じており、facade の公開面へ
  一切影響しない
- **fail-fast 契約**: `context_cache.rs` は「ミス時の構築が失敗した場合、その `Err` は
  キャッシュへ格納しない」（同ファイル冒頭コメント「fail-fast 契約」節。
  `get_or_build` 実装 `crates/backend-cuda/src/context_cache.rs:136-157`。
  `crates/backend-metal/src/context_cache.rs` も同じ契約を踏襲する）。
  この検証（driver 不在・範囲外 ordinal・NVRTC コンパイル失敗）が再実行されるのは
  **キャッシュミス時のみ**であり、driver が後から利用可能になった環境でも次回の
  ミス（＝未構築のスロットへの呼び出し）で `tape_for(Device::Cuda(_))` は正しく
  回復する。**キャッシュヒット時**は `get_or_build` が既存の `Arc` をそのまま
  `clone` して返す（同ファイル 151-153 行目）ため、構築時の検証を再実行しない
  ——これは受け入れ条件 1（2 回目以降が初期化コストを支払わない）そのものであり、
  ヒット時に検証を省略しても「driver が後から利用可能になった環境の回復」を
  妨げない（回復の主体はミス時の再試行であり、ヒット経路の検証省略とは独立）。
  これは資源の**寿命**を延ばす最適化であり、ミス時の検証を遅延・省略する変更では
  ない
- **数値一致契約・FMA 契約**: #929 は同一の `CudaGemm`／`CudaElementwise` 等の
  インスタンスを使い回すだけで、カーネルのソース・コンパイルオプション・実行経路は
  一切変更していない。計算結果に影響する変更ではないため、バックエンド間数値一致の
  統一複合判定（`.claude/rules/coding-rust.md`）は非対象のまま維持される

## 4. 先行事例対比（candle／Burn の再利用構造の要点）

`scripts/bench/framework-compare/bench-candle`・`bench-burn` の実装は、いずれも
呼び出し側（ベンチコード）が `Device` ハンドルをループ外で 1 回構築し、以降の
呼び出しへ**明示的に**引き回す API 形状を取る（`candle_core::Device::new_cuda(0)`／
Burn のデバイス型を関数引数として渡す構成）。これは「公開 API レベルで資源の
生存期間をユーザーコードが管理する」設計であり、本文書の案 A（facade 公開型としての
`DeviceHandle`）に相当する構造である。

対して #929 が採用したのは、**利用者からは見えない内部キャッシュ**（プロセス内
`ordinal` キー）による透過的な再利用である。candle/Burn の方式と異なり、利用者が
`tape_for` を毎回呼んでも初期化コストの再計上が起きない。§5 で両者を比較する。

## 5. 設計案の比較と採否

| 案 | 内容 | 状態 |
|----|------|------|
| 案 A | facade に新規公開型（`DeviceHandle` 等）を追加し、利用者が明示的に生存期間を管理する | 不採用（本文書の判断） |
| 案 B | バックエンド内部（`backend-cuda`／`backend-metal`）にプロセス内常駐キャッシュを持たせ、`tape_for` を透過的に高速化する | **CUDA（#929/#946）・Metal（#930/#948）ともに実装済み** |
| 案 C | A + B 併用 | 不採用（B のみで目的を達成できるため A を追加する必然性がない） |

### 採否判断

**案 B のみを採用する（すでに CUDA について実装済みの構成を追認し、公開 API の
追加変更は行わない）。**

根拠:

1. **2.1 の実測事実**: 案 A（facade レベルのハンドル保持）は、バックエンド内部が
   呼び出しごとに資源を再構築する構造を変えない限り単独では効果がない。効果を
   出すには結局バックエンド内部の常駐化（案 B 相当の実装）が必要であり、
   案 A は追加の公開面というコストを払うだけで、案 B に対して独立した価値を
   提供しない
2. **REQ-12 との緊張回避**: `docs/spec/04-requirements.md:249`（および
   `docs/spec/05-tasks.md:316`）は「利用者向け公開面を `Device` 識別子のみに
   限定する」ことを明記する。案 A（新規公開型の追加）はこの記述と字面上の
   緊張関係を持つが、**案 B のみを採用する場合はそもそも公開面を追加しない
   ため、この緊張は生じない**。spec 側の記述と無変更で整合する（§6 で詳述）
3. **facade 1 クレート境界の維持**: 案 B は `backend-cuda` クレート内部の
   非公開モジュール（`context_cache`）に閉じた変更であり、facade の結線ロジック
   一本化という構造的境界を一切変えない。案 A は新規公開型の設計・
   `api_surface.rs` の拡張・ドキュメント整備という追加コストを要求するが、
   2.1 の実測によりその追加コストに見合う効果がない
4. **fail-fast・数値一致の不変条件**: §3 のとおり案 B はいずれの不変条件も
   壊さない。案 A を追加した場合も同じ不変条件を維持する設計は可能だが、
   1.〜3. の理由により追加自体が不要と判断する

案 A を将来的に再検討する条件があるとすれば、「利用者が明示的に資源解放
タイミングを制御したい」（プロセスワイド常駐ではなく、ハンドルの `Drop` で
即座に GPU 資源を解放したい）というユースケースが要件として明確化された
場合である。現時点でそのような要件は本イシューの受け入れ条件・spec のいずれにも
記載がなく、仮説的なユースケースのために公開面を増やすことは REQ-12 の
「必要最小限の公開面」という設計原則（`crates/facade/src/lib.rs` 冒頭コメント
「公開面の設計」節）に反する。

## 6. spec 側更新の要否判断（受け入れ条件 3）

`docs/spec/04-requirements.md:249` および `docs/spec/05-tasks.md:316` は
「利用者向け公開面を `Device` 識別子のみに限定する」ことを明記している。

**判断: spec 側の更新は不要。**

理由: 本文書が採用する案 B は facade の公開面を一切変更しない（`tape`／
`tape_for`／`Device` のみのまま）。したがって「公開面を `Device` 識別子のみに
限定する」という spec の記述と本文書の判断は無矛盾であり、記述を緩和・変更する
必要がある事態（＝新規公開型を追加する案を採用する事態）は生じなかった。
spec 側 Issue の起票提案は不要と判断する。

## 7. スコープ外・後続イシュー提案（承認待ち事項）

以下は本文書が識別したが本イシューの範囲外の事項である。ユーザー承認を得た
うえで別イシューとして追跡することを提案する（`out-of-scope-tracking.md`
方針。本エージェントはここでの Issue 起票は行わない）。

**Metal 側の常駐キャッシュ実装**（旧 §2.4 の後続課題）は本文書の執筆後、
イシュー #930・PR #948 で実装が完了したため、以下からは除外した（§2.4 参照）。

1. **CUDA 側の削減幅の実機再計測**: `docs/perf/cuda-tape-init-cost-diagnosis.md`
   §6（フェーズ別内訳）、および `scripts/bench/framework-compare/` の
   fresh/reuse 比較（環境 3 相当）を #929/#946 適用後の状態で DGX Spark GB10
   にて再実測し、440〜460 ms 帯からの実際の削減幅を定量記録する（実機ツリー
   #408 への引き継ぎ事項）
2. **Metal 側の削減幅の実機再計測**: イシュー #927（診断）・#930/#948（常駐化
   実装）を踏まえ、Metal 実機で常駐化前後の固定オーバーヘッド（約 5 ms 帯）の
   実際の削減幅を定量記録する（同じく実機ツリー #408 への引き継ぎ事項）

## 8. 出典・関連文書

- `scripts/bench/framework-compare/results/summary.md`（PR #915、計測 2026-08-28。
  環境 2: 440〜460 ms 実測、環境 3: fresh/reuse 比較・#925）
- `docs/perf/cuda-tape-init-cost-diagnosis.md`（#926/#945。呼び出しごとの
  遅延初期化の機構診断）
- `crates/backend-cuda/src/context_cache.rs`（#929/#946。実装本体）
- `crates/backend-cuda/src/ops.rs`（`context_cache` への結線箇所）
- `crates/backend-metal/src/context_cache.rs`（#930/#948。CUDA 側と同型の実装本体）
- `crates/backend-metal/src/ops.rs`（`context_cache` への結線箇所）
- `crates/facade/src/lib.rs`（公開面の設計・`Device::Cuda(_)` の構築規則コメント）
- `docs/spec/04-requirements.md:249`・`docs/spec/05-tasks.md:316`（REQ-12「公開面を
  `Device` 識別子のみに限定する」記述）
- `docs/public-api-design.md` §6（本文書への相互参照を追記）
