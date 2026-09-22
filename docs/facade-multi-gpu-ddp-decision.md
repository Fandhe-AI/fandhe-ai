# 複数 GPU（DataParallel／DDP・勾配 all-reduce）の設計判断記録（#1628）

イシュー #1628「複数 GPU（DataParallel／DDP・勾配 all-reduce）の設計判断を記録する」に対応する。親: #1573（Tier 2）・ルート: #1570。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とし、本リポでは `docs/spec/` を編集しない。

基準コミット: `f811efb27f44b797d4c32cb3088f8c185484324e`（2026-09-13）。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 1. 背景

対応する PyTorch／TensorFlow 機能: `torch.nn.DataParallel`／`torch.nn.parallel.DistributedDataParallel`（NCCL ベースの勾配 all-reduce）、`tf.distribute.MirroredStrategy`（複数デバイス間の変数複製・勾配集約）。

`docs/compat-feature-gap.md` §2.13「device」の「複数 GPU・`DataParallel`/`DDP`」行は「なし・実装に必要なもの: 勾配 all-reduce・パラメータ複製の設計（ネットワーク層から必要）・難度 XL」のまま未着手（`docs/compat-feature-gap.md:331`）。spec REQ-9 の 2026-09-12 追記は本機能を Tier 2（長尾）に分類しつつ、除外事項「分散学習・量子化の網羅対応」（Won't・条件付き）へ従属させている（`docs/spec/04-requirements.md:232`）。

**除外事項の非対称性（本判断の中心）**: 同じ Tier 2 の量子化（FP8/INT8 GEMM）には格上げ条件表（a〜e、`docs/spec/04-requirements.md:404`）が定義されている一方、**分散学習（複数 GPU／DDP）には格上げ条件表が一切存在しない**。`docs/spec/04-requirements.md:356` は「分散学習の網羅対応は本項目のまま Won't（条件整理なし）を維持する」と明記し、2026-09-12 注記（イシュー #66）もこの Won't 判断を変更していない。したがって複数 GPU／DDP は量子化以上に再開のハードルが高い——再開するには spec リポ側で格上げ条件表そのものを新規提案する必要があり、それ自体が別途ユーザー承認・spec 側審査を要する。

`docs/compat-api-scope.md` §5 は既に本 issue の結論を先取り確定している: 「**#1628（DDP）は設計判断の記録（docs のみ）に留め、実装・通信層の依存追加は行わない**」（`docs/compat-api-scope.md:456` 付近）。本ドキュメントの役割はこの結論の技術的根拠を掘り下げて記録することである。

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `Device` enum は `Cpu`／`Cuda(usize)`（ordinal 1 個）／`Metal`（macOS 限定）の 3 variant。1 `Tape` は 1 デバイス・1 `BackendOps` インスタンスに固定され、複数デバイスを跨いで協調する概念が type レベルで存在しない | `crates/tensor-core/src/device.rs:39-52` |
| `Device::available()`（デバイス列挙 API）は `docs/public-api-design.md` §4.1 で設計のみ記載され未実装。`tensor-core` は `backend-*` を直接参照できないため `enumerate_all`／`select_from` を代替提供するに留まり、集約入口の結線は TASK-1.9c／1.9d（#46／#47）へ引き継がれたまま未着手（複数 GPU 制御の前提となる「何台あるか」を返す統一 API が現状ない） | `crates/tensor-core/src/device.rs:6-19` |
| CUDA 側は ordinal をキーとするプロセス内キャッシュ（`context_cache`）で `CudaDevice`／`CudaGemm` 等を ordinal ごとに保持できる構造があるが、これは「同一プロセスが複数 ordinal を扱いうる」という下地に過ぎず、複数 ordinal 間の勾配同期・パラメータ複製を協調させる仕組みは一切ない | `crates/backend-cuda/src/context_cache.rs` |
| CUDA バックエンドの同期契約は「ordinal ごとに単一ストリーム」を不変条件として明文化済み（I1〜I5）。複数 ordinal 間通信（P2P／NCCL）は本契約のスコープ外で未検討 | `docs/backend-cuda-async-execution-design.md` §2.1 |
| Metal は macOS cfg 限定の単一 GPU 前提の設計（`objc2`／`objc2-metal`）。Apple 環境にはベンダー標準の NCCL 相当（マルチ GPU collective communication ライブラリ）が存在せず、CUDA と対称な設計ができない | `crates/tensor-core/src/device.rs:49-52`・`.claude/rules/deps-policy.md` |
| CPU バックエンドは `rayon`（共有メモリ内スレッド並列）のみで、プロセス間／ノード間通信の概念がない | `.claude/rules/deps-policy.md`（許容依存 9 区分表） |
| 許容依存 9 区分（`deps-policy.md`）に「ネットワーク／分散通信」区分は存在しない | `.claude/rules/deps-policy.md` |

### 2.1 計画段階の仮説の検証（cudarc `nccl` feature の link 契約）

本 issue の計画段階では「cudarc（`=0.19.8`）の `nccl` feature を有効化すると、ビルド時に `libnccl` の静的リンク解決（`cargo:rustc-link-lib=dylib=nccl`）が要求され、CI の `build-no-cuda-toolkit` ジョブ（CUDA toolkit 非搭載環境でのビルド成立を fail-closed 検証する。`.github/workflows/ci.yml:116`）と構造的に衝突する」という仮説を立てていた。**この仮説はソース読解による検証の結果、誤りと判明した**（ビルドでの実証は未実施。以下はソースコード読解に基づく）:

- cudarc 0.19.8 の `build.rs` は `cargo:rustc-link-lib=dylib=nccl` を含む静的リンク処理（`dynamic_linking()` 関数）を `#[cfg(feature = "dynamic-linking")]` の場合にのみ呼び出す（`build.rs:149-150`）。fandhe-ai の workspace 依存は `dynamic-linking` ではなく `dynamic-loading` feature を使用する（`Cargo.toml:112-118`）ため、`nccl` feature を追加してもこの静的リンク処理自体が呼び出されず、ビルド時に `libnccl` の存在は要求されない。
- cudarc の NCCL バインディング（`src/nccl/sys/mod.rs`）は CUDA driver 本体（`cudarc::driver::sys`）と同型の実行時 `dlopen` パターンを備える: `#[cfg(feature = "dynamic-loading")] fn load<F>(...)` はシンボルを呼び出し時に `libloading` 経由で解決し（`src/nccl/sys/mod.rs:9-12`）、`is_culib_present()`／`culib()` は `libnccl` を実行時に遅延ロードする（`src/nccl/sys/mod.rs:1362-1385`）。
- `libloading` は既に `dynamic-loading` feature の推移的依存としてロックファイルに存在する（`Cargo.lock:467-469`。`libloading =0.9.0`）ため、`nccl` feature を追加しても新規クレートは増えない（ライセンス実測への影響なし。区分自体は「許容依存 9 区分に含まれない cudarc feature の拡張」として引き続きユーザー承認事項）。

この訂正の結果、案 A（後述）を退ける理由から「CI 不変条件との具体的な link-time 衝突」は除去される。ただし、以下（§4「案 A」）のとおり案 A を退ける独立の理由は複数残る。

## 3. 契約整理（守るべき既存契約）

- **正本 spec の除外事項（Won't・格上げ条件表なし）という非対称性**（§1 参照。量子化との対比）。
- **`docs/compat-api-scope.md` §5 の既存記述**（「#1628 は設計記録のみ・依存追加なし」）が既に本 issue の結論を先取り確定していること。
- **`.claude/rules/deps-policy.md` 許容依存 9 区分**にネットワーク層区分がないこと。cudarc は既に許容依存（CUDA 区分）だが、`deps-policy.md` は cudarc の**有効化する feature を明示列挙**しており（`driver／nvrtc／dynamic-loading／cuda-13000／f16`）、`nccl` feature の追加はこの列挙の拡張に当たるためユーザー承認が必要（§2.1 の訂正により link-time 衝突という即座の技術的ブロッカーは解消したが、承認要件そのものは変わらない）。
- **`.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」**が明記する「バックエンド切替は feature フラグなしの cfg ベース」という方針。§2.1 の訂正により「nccl の常時有効化自体は CI を壊さない」ため cargo feature で回避する必要性は薄いが、それでも新規機能を既存の cfg ベース設計へどう位置づけるかは未解決の論点として残る。
- **CI `build-no-cuda-toolkit` ジョブ**（`.github/workflows/ci.yml:116`）が検証する「CUDA toolkit 非搭載でもビルド成立」不変条件。§2.1 のとおり `nccl` feature 単体は本契約を破壊しないが、この検証は本 doc の主張の裏付けであり継続して尊重する。
- **本番経路で `unwrap()`／`expect()` を使わない・型付きエラーとする方針**（`.claude/rules/coding-rust.md`「コード品質」節）。cudarc の NCCL バインディングは `is_culib_present()` が失敗すると `panic_no_lib_found` で panic し、`load()` はシンボル欠落時に `panic!("Missing symbol {name}")` で panic する（§2.1 で確認したのと同じ `src/nccl/sys/mod.rs` の実装）。これは CUDA driver 自体の扱い（`crates/backend-cuda/src/device.rs:265-278` が `is_culib_present()` を probe として使い `BackendError::CudaUnavailable` へマップする既存パターン）と同型の対処が必要になることを意味し、実装時には同じ「probe → 型付きエラー」設計が要る。
- **REQ-12「利用者向け融合制御 API を提供しない」・`facade` 唯一の公開面**という既存宣言。複数デバイス制御 API を追加する場合、どの層（`facade` か `tensor-core`／`backend-cuda`）に公開面を置くかは未検討で、再開時に個別の整合確認が要る。

## 4. 設計案の比較

| 案 | 概要 | 新規依存 | CI 不変条件への影響 | スコープ | 難度 |
|---|---|---|---|---|---|
| **A: NCCL 経由（cudarc `nccl` feature）** | PyTorch DDP に最も近い設計。`cudarc::nccl`（`Comm`／`Id`／`ReduceOp` 等）で勾配 all-reduce を行う | 新規クレートなし（`libloading` は既存推移的依存。ただし cudarc の許容 feature 列挙の拡張はユーザー承認事項） | §2.1 の訂正により静的リンク衝突は生じない。ただし実行時 `libnccl` 不在（`build-no-cuda-toolkit` 環境や NCCL 未インストール環境）では `is_culib_present()` probe が false を返す設計にしない限り panic 経路に落ちるため、CUDA driver と同型の「probe → `BackendError` variant」実装が新たに必要 | CUDA 限定（単一ノード内複数 GPU、または NVLink／ネットワーク越しの複数ノード）。Metal・CPU に対称手段がなく `BackendOps` 抽象を跨いだ統一設計にならない | XL（`docs/compat-feature-gap.md:331` の既存評価どおり） |
| **B: 汎用分散通信ライブラリ（gloo／MPI 等の新規クレート）** | ベンダー中立な collective communication | 許容依存 9 区分にない新規区分の追加が必要（ユーザー承認必須） | 新規区分のため deps-policy.md の全面改訂が要る | GPU 間直接転送を持たないため GPU-GPU 転送は別途 D2H/H2D を要し性能上不利。全バックエンド横断は可能だが低速 | XL |
| **C: 自作 TCP ベース ring-allreduce（標準ライブラリのみ）** | 依存追加ゼロで rank 間の勾配集約を自作 | なし | 影響なし | rank／world discovery・フォールト耐性・NCCL 相当の GPU 直接転送最適化を欠く。`docs/compat-feature-gap.md:331` の「難度 XL」評価どおり実装量が大きく、単一ノード内複数 GPU にすら性能上の見込みが不明瞭 | XL |
| **D: 単一ノード内複数 GPU 限定（NCCL 不使用。CUDA P2P や host 経由コピーでの勾配平均化）** | ネットワーク層は不要 | なし（追加の CUDA API 呼び出しのみ） | 影響なし | 真の DDP（複数ノード）には届かない。CPU／Metal に対称な複数デバイス概念がなく `BackendOps` を跨いだ統一設計が困難（§2「現状のコード事実」） | L〜XL |
| **E（推奨）: 段階 0 — 現時点では非対応と明文化し、再開条件を記録する** | 実装着手せず、本 doc に契約整理・再開条件を残す | なし | 既存契約への影響ゼロ | — | — |

## 5. 推奨（段階的）

**案 E（段階 0）を採用する。** 理由:

1. `docs/compat-api-scope.md` §5 が既に「#1628 は設計記録のみ・依存追加なし」と確定しており、本 issue の受け入れ条件（実装を含めない）とも一致する。
2. 正本 spec の除外事項に格上げ条件表が存在しない（量子化と異なり再開の道筋自体が未定義。§1）ため、実装着手の前提条件（spec 側の格上げ判断）がそもそも成立していない。
3. 案 A は §2.1 の訂正により当初想定した CI 不変条件との即時衝突こそ解消したが、cudarc の許容 feature 列挙拡張というユーザー承認事項・panic 経路を型付きエラーへ変換する実装・Metal/CPU との非対称という独立の課題が残っており、単独では採用を正当化しない。
4. 案 B／C は新規依存区分の追加または大規模自作実装を要し、いずれも本 issue のスコープ（設計記録のみ）を超える。
5. 案 D は複数ノード分散に届かず、`Device` の設計（`Cuda(usize)` 単一 ordinal・`Tape` は 1 デバイス固定）自体が複数デバイス協調の前提を持たないため、いずれの案を選んでも `tensor-core`／`facade` 層の型設計から着手する必要がある。

## 6. 再開条件（将来の別 issue への引き継ぎ・本 PR では起票しない）

- **前提 1**: 正本 spec（`fandhe-ai-spec`）側で除外事項「分散学習・量子化の網羅対応」に格上げ条件表（量子化の a〜e 相当）を新規提案・承認すること（spec 提案自体が別途ユーザー承認事項）。
- **前提 2**: `Device::available()`（デバイス列挙 API。§2）の実装（前提となる基礎 API が現状皆無）。
- **前提 3**: 採用するネットワーク層方式（案 A〜D）の確定とそれに伴う依存追加（cudarc `nccl` feature の許容依存列挙への追加を含む）・CI 不変条件・feature flag 方針の扱いについての個別ユーザー承認。
- **前提 4**: 案 A を選ぶ場合、`is_culib_present()` probe を経由した `BackendError` variant 設計（CUDA driver の既存パターン。`crates/backend-cuda/src/device.rs:265-278`）と、NCCL バージョンピン（`nccl = ["nccl-02030"]` = NCCL 2.30 系）に対する実機（DGX Spark GB10）インストール済み NCCL バージョンの検証が必要（本セッションは実機到達手段がなく未検証のまま記録する）。
- **スコープ想定（記録のみ・起票はしない）**: 初期スコープを単一ノード内複数 GPU（同一プロセス・複数 CUDA ordinal）に限定し、真の複数ノード分散は別段階とする案を「起票候補」として記載する。

### セキュリティ観点の留意点（将来の実装着手時。本 issue ではコード変更を伴わないため対策の実装は不要）

- **A08（ソフトウェア・データ整合性）**: 複数ノード分散を将来実装する場合、rank／world discovery やパラメータ複製の経路は改竄・なりすまし（悪意あるノードの参加）に対する検証が必要になる（NCCL 自体は認証機構を持たないため、ネットワーク層を自作する場合は別途検討が要る）。
- **A03 に類する論点**: 通信層を自作する場合、受信したメッセージ長・shape 情報の検証（外部入力の境界検査。`.claude/rules/security.md`「A03」節と同じ観点）が必要になる。

## 7. スコープ外

- 量子化（#1627。別 issue・同じ除外事項に従属）。
- 混合精度・f64/bf16 演算等の他 Tier 2 項目。

## 8. 承認事項（実装着手の前提。列挙のみ）

1. spec 側での格上げ条件表の新規提案・承認の要否判断。
2. ネットワーク層方式（NCCL feature 有効化／新規クレート／自作実装）の選択。
3. NCCL 採用時の cudarc 許容 feature 列挙拡張（`deps-policy.md` 更新）の可否。
4. `Device::available()` 等の前提 API 実装の着手可否。
5. 再開後の実装 issue 起票。

## 9. 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/compat-feature-gap.md:331` | 「複数 GPU・`DataParallel`/`DDP`」行の現状評価（未実装・難度 XL） |
| `docs/compat-api-scope.md:456` 付近 | §5「#1628（DDP）は設計判断の記録（docs のみ）に留め、実装・通信層の依存追加は行わない」 |
| `docs/spec/04-requirements.md:232` | REQ-9 2026-09-12 追記の Tier 2 列挙（複数 GPU／DDP を含む）と除外事項への従属関係 |
| `docs/spec/04-requirements.md:356` | 除外事項「分散学習・量子化の網羅対応」（Won't・条件整理なしの明記） |
| `docs/spec/04-requirements.md:404` | 量子化の格上げ条件表（a〜e）。複数 GPU／DDP には同等の表が存在しないことの対比根拠 |
| `crates/tensor-core/src/device.rs:6-52` | `Device` enum・`Device::available()` 未実装の経緯コメント |
| `crates/backend-cuda/src/context_cache.rs` | ordinal ごとのプロセス内キャッシュ構造 |
| `crates/backend-cuda/src/device.rs:265-278` | `is_culib_present()` probe → `BackendError::CudaUnavailable` の既存パターン |
| `docs/backend-cuda-async-execution-design.md` §2.1 | CUDA 非同期実行の同期契約（ordinal ごと単一ストリーム） |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分・cudarc の許容 feature 列挙 |
| `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」節 | feature フラグなし・cfg ベースのバックエンド切替方針 |
| `.github/workflows/ci.yml:116` | `build-no-cuda-toolkit` ジョブ |
| `Cargo.toml:112-118` | workspace の cudarc feature 指定（`dynamic-loading` を含む） |
| cudarc `=0.19.8` `build.rs:80-150` | `dynamic_linking()` が `dynamic-linking` feature 限定で呼び出される経路 |
| cudarc `=0.19.8` `build.rs:186-204` | `dynamic_linking()` 内 `cargo:rustc-link-lib=dylib=nccl` 発行条件 |
| cudarc `=0.19.8` `src/nccl/sys/mod.rs:9-13, 1362-1385` | `dynamic-loading` feature 下の NCCL 遅延シンボル解決・`libnccl` dlopen |
| `Cargo.lock:467-469` | `libloading =0.9.0`（`dynamic-loading` の既存推移的依存） |
| `docs/backend-abstraction-amd-readiness-decision.md`・`docs/autodiff-custom-function-decision.md`・`docs/autodiff-higher-order-grad-decision.md` | 同型の「実装しない・設計記録のみ」文書構成の precedent |

## 10. 後続 #2074

§2.1「ビルドでの実証は未実施」は #2074（`docs/ddp-grade-up-conditions.md`）で実証済みに更新された（`cargo build`／`cargo test --features cudarc/nccl --locked` が exit 0・実行時プローブで `libnccl present = false`・panic なしを確認）。§6 前提 2「`Device::available()` 未実装」は #1614（`fandhe_ai::available_devices()`）により達成済みであり陳腐化している。分散学習の格上げ条件表（量子化 a〜e 相当）の文案は `docs/ddp-grade-up-conditions.md` §3・§4 を正とし、本 doc の既存本文（基準コミット固定の記録）は書き換えない。
