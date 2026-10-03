# DDP（複数 GPU・勾配 all-reduce）の格上げ条件表（(b) 形式 spec 提案文案）と nccl リンク契約の実測（#2074）

基準コミット: `d81d7801caa4621d450572ea8226ef5e0a6a0a2d`（2026-09-22）。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

#2612 改訂の基準: HEAD `4be494be`、`docs/spec` submodule `2e998dd7`（2026-09-29）。§0・§3・§3a・§4・§5・§8・§9 の最新行番号はこの時点のもの（§2 は #2074 時点のまま不変。ただし §2.2 末尾に #2613 の小節を追記した）。

#2613 実測の基準: HEAD `1c6f9735`（2026-10-03）。§2.2 の「#2613 追記」小節、§0 の該当 bullet、§5 項 2、§9.2（#2613 節）がこの時点のもの。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値は一切変更していない）。
- `docs/facade-multi-gpu-ddp-decision.md`（#1628）が確定した**段階 0（現時点非対応）は不変**。本 doc はその再開条件を「格上げ条件表」という具体的な形に構造化し、条件 (a)（nccl リンク契約）を実測で裏付けたものである。
- §4 の spec (b) 形式提案文案は**起票していない**（未実施。§5 の承認事項 1 を参照）。
- 量子化（除外事項の同じ Won't 項目）には既に格上げ条件表 a〜e（`docs/spec/04-requirements.md:360`）が存在するが、分散学習（複数 GPU／DDP）には存在しない（`docs/spec/04-requirements.md:356`「分散学習の網羅対応は本項目のまま Won't（条件整理なし）を維持する」）。本 doc の §3 はこの非対称を埋める文案である。
- **#2612 で追記した点**: 上記の `04-requirements.md:356`・`:360` は #2074 時点の行番号で、現行（submodule `2e998dd7`）では除外事項が `:358`、量子化の格上げ条件表が `:360-363`、承認時固定契約・取り決めが `:364-365` である。§4 文案を現時点の事実（REQ-9 追記 #2193／#2194・0.10.0 出荷・deps-policy 第 10 区分新設）へ最新化した（差分は §9.1）。
- **FSDP は `docs/spec/` に記述が 0 件**で、本提案の格上げ条件表の対象外とする（§3a）。FSDP 用の条件は創作していない。
- #1964（2026-09-17）で「現状維持（選択肢 C）」がユーザー承認済み。§4 の spec 起票は引き続き**未起票・未承認**である（§5）。
- **#2613 で追記した点**: `[workspace.dependencies].cudarc.features` に `nccl` を足した一時複製での実測（§2.2 末尾）では、`Cargo.lock` は byte 一致のまま（`--locked` 成功）、ワークスペース全体の crate 集合差分はゼロ、`cargo deny` の bans／licenses／sources は ok、テスト実行ファイルの `NEEDED` に `libnccl`／`libcuda`／`libnvrtc` は現れなかった。deps-policy への `nccl` 追加は**承認依頼済み・承認未取得**（#2613 コメント）。本リポの `Cargo.toml`／`Cargo.lock`／deps-policy／license-matrix は変更していない。

## §1 位置づけ

イシュー #2074「DDP の格上げ条件表の新規提案（(b) 形式）と nccl リンク契約再検証」。入力:

- `docs/facade-multi-gpu-ddp-decision.md`（#1628。§6「再開条件」・§8「承認事項」）
- `crates/facade/src/lib.rs:602`（`fandhe_ai::available_devices()`。#1614 で実装済み。#1628 doc §6 前提 2 の陳腐化を訂正する根拠）
- 正本 spec 除外事項「分散学習・量子化の網羅対応」（`docs/spec/04-requirements.md:356`）・REQ-9 Tier 2（`docs/spec/04-requirements.md:222-235`）

「(b) 形式」とは、実装リポ側の doc を出典として、spec 側には短い規定（本件では格上げ条件表）だけを追記する提案形式を指す（先例: `docs/candle-parity-tolerance-contract-decision.md` §7・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2 の「タイトル案 + ````markdown フェンスの起票用本文」形式。実起票は Fandhe-AI/fandhe-ai-spec#64）。案 (a)（spec 本体の要件・判定式そのものを改定する形式）とは対になる。

## §2 条件 (a) nccl リンク契約の再検証

### §2.1 ソース読解（#1628 §2.1 の再確認）

cudarc `=0.19.8`（`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.8/`。本リポの workspace 依存と同一版）:

- `Cargo.toml:127`: `nccl = ["nccl-02030"]`・`Cargo.toml:139`: `nccl-02030 = ["driver"]`。optional 依存を一切追加しない（推移的クレート集合は不変）。
- `build.rs:149-150`: `dynamic_linking(major, minor)` は `#[cfg(feature = "dynamic-linking")]` の場合にのみ呼ばれる。本リポの workspace 依存は `dynamic-linking` ではなく `dynamic-loading` feature を使用する（本リポ `Cargo.toml:122-128`）ため、この関数自体が呼び出されない。
- `build.rs:186-205`: `dynamic_linking()` 内で `cargo:rustc-link-lib=dylib=nccl` を発行する条件は `nccl`／`nccl-02022`〜`nccl-02030`／`nccl-version-from-build-system` feature の有効化だが、この行に到達するのは `dynamic-linking` 経由でこの関数が呼ばれた場合に限る。
- `src/nccl/sys/mod.rs:10-12`: `#[cfg(feature = "dynamic-loading")] fn load<F>(...)` はシンボルを呼び出し時に `libloading` 経由で解決する。`src/nccl/sys/mod.rs:1363-1372`（`is_culib_present`）・`1373-1385`（`culib`）は `libnccl` を実行時に遅延ロードする non-panicking／panicking プローブをそれぞれ提供する。
- `src/nccl/mod.rs:13-18`: `pub mod sys;`・`pub mod safe;` を公開しており、`cudarc::nccl::sys::is_culib_present()` へ到達可能。

以上から、**`dynamic-loading` 下では `nccl` feature を有効化しても link-time に `libnccl` を要求しない**という #1628 §2.1 の「訂正」（ソース読解のみ・ビルド未実証）が成立する構造であることを確認した。

### §2.2 実測（本イシューで新規実施）

環境: 本エージェント実行環境（Linux・`bash scripts/check-cuda-toolkit-absent.sh assert` PASS〈CUDA toolkit 非搭載〉・`ldconfig -p | grep -i nccl` 該当なし〈libnccl 非搭載〉・`nvidia-smi -L` は NVIDIA GeForce RTX 3060 を検出〈driver あり〉）。CI `build-no-cuda-toolkit` ジョブと同じ「toolkit 非搭載」前提だが、driver ありという点で別状態である。

| # | コマンド | 期待 | 実測結果 |
|---|---|---|---|
| 1 | `cargo tree --locked -p fandhe-ai-backend-cuda -e normal --prefix none`（base）と `--features cudarc/nccl` 付き（nccl 有効化）の `sort -u` 差分 | クレート集合の差分ゼロ | **差分ゼロ**（`diff` 出力なし。`NO CRATE DIFF`） |
| 2 | `cargo tree --locked -p fandhe-ai-backend-cuda --features cudarc/nccl -e features -i cudarc \| grep nccl` | `nccl`／`nccl-02030` feature が有効化 | 両 feature とも `(command-line)`／`(*)` として出力（有効化を確認） |
| 3 | `cargo build -p fandhe-ai-backend-cuda --features cudarc/nccl --locked` | exit 0 | **exit 0**（3.99s） |
| 4 | `cargo build -p fandhe-ai-backend-cuda --features cudarc/nccl --locked --tests` | exit 0 | **exit 0**（11.06s） |
| 5 | `cargo test -p fandhe-ai-backend-cuda --features cudarc/nccl --locked` | 既存テスト全 pass・panic なし | **49 passed; 0 failed**（0.00s。ドキュメントテスト 0 件） |
| 6 | 警告数の比較（`grep -c "^warning:"`）: feature 無し vs `--features cudarc/nccl` | 同数（既存 dead_code 由来で nccl 有効化により新規警告が増えないこと） | 両条件とも **77**（同数。nccl 有効化前後で新規警告なし） |
| 7 | `git status --porcelain -- Cargo.toml Cargo.lock` | 空 | **空**（変更なし） |

コマンド 1〜7 はいずれも `Cargo.toml` を編集せず `--features cudarc/nccl`（CLI 一時指定）のみで実施した。

#### #2613 追記: ワークスペースマニフェスト変更時の実測（一時複製）

#2074 の上記測定は `-p fandhe-ai-backend-cuda -e normal` に CLI の `--features cudarc/nccl` を一時指定したものだった。ここではワークスペースの `[workspace.dependencies].cudarc.features`（`Cargo.toml`）に `"nccl"` を 1 行足した場合の影響を、`mktemp -d` で作った**一時複製**（`.git`・`target`・`.claude`・`.codex` を除いてコピーし、複製側の `Cargo.toml` だけを編集。測定後に削除済み。本リポの `Cargo.toml`／`Cargo.lock` は未変更）で測った。base は本リポ（HEAD `1c6f9735`）。`CARGO_NET_OFFLINE=true`。

環境: Linux x86_64。`bash scripts/check-cuda-toolkit-absent.sh assert` は OK（`libcuda.so` ドライバは検出されるが toolkit ではないため判定対象外）。`ldconfig -p | grep -i nccl` は該当なし（libnccl 非搭載）。`nvidia-smi -L` は NVIDIA GeForce RTX 3060 を 1 基検出。

| # | コマンド（複製側で実行。#2 の base は本リポ側） | 期待 | 実測結果 |
|---|---|---|---|
| 1 | `cargo tree --locked --workspace`（`--locked` のまま解決できるか）と `cmp <複製>/Cargo.lock <本リポ>/Cargo.lock` | `--locked` 成功・byte 一致 | **exit 0・byte 一致**（`Cargo.toml` 1 行の変更で `Cargo.lock` は不変） |
| 2 | `cargo tree --workspace --all-features --locked -e normal,build,dev --target all --prefix none`（`check-forbidden-deps.sh tree` と同形）の base／複製を、ワークスペースメンバーのパス表記を正規化したうえで `sort -u` して `diff` | 差分ゼロ | **146 行ずつ・差分ゼロ**（新規 crate なし） |
| 3 | `cargo tree --locked --workspace -e features -i cudarc`（`-p` なし） | `nccl`／`nccl-02030` がマニフェスト由来で有効化 | `cudarc` の feature として `nccl`／`nccl-02030` が現れ、有効化元は `fandhe-ai-backend-cuda`・`bench-harness`（`cudarc.workspace = true` で継承）。facade（`fandhe-ai`）は `fandhe-ai-backend-cuda` 経由。CLI 指定（`(command-line)`）ではない |
| 4 | `cargo build -p fandhe-ai-backend-cuda --locked --tests` | exit 0 | **成功**（21.2s。初回ビルド） |
| 5 | `target/.../build/cudarc-*/output` の `rustc-link-lib` 行 | `nccl` 行なし | **`rustc-link-lib` 行は 0 件**（`dynamic-loading` 下では build script がリンク指示を一切出力しない、という観測結果。複製側に出力自体がないため base との出力比較は省略） |
| 6 | `cargo test -p fandhe-ai-backend-cuda --locked --no-run --message-format=json` で得た実行ファイル 83 本に `readelf -d <exe>` の `NEEDED` 検査 | `libnccl`／`libcuda`／`libnvrtc` を含まない | **83 本すべてで該当なし**（`NEEDED` は `libc.so.6`・`libgcc_s.so.1`・`ld-linux-x86-64.so.2`、一部に `libm.so.6` のみ） |
| 7 | `cargo test -p fandhe-ai-backend-cuda --locked` | 全 pass・panic なし | **1317 passed; 0 failed; 292 ignored**（`test result` 82 行の合計。`panicked`／`FAILED` は 0 件） |
| 8 | `cargo deny --manifest-path <複製>/Cargo.toml --locked check --config <複製>/deny.toml bans licenses sources` | ok | **`bans ok, licenses ok, sources ok`**（`advisories` はネットワーク取得が要るため未実施） |
| 9 | 複製ルートで `bash scripts/check-forbidden-deps.sh tree` | PASS | **OK**（依存禁止リストの混入なし） |
| 10 | 本リポで `git status --porcelain -- Cargo.toml Cargo.lock` | 空 | **空** |

#2074 との違い: (i) CLI の一時指定ではなくマニフェスト変更、(ii) `-p` 単体ではなく `--workspace --all-features --target all`・dev／build エッジまで含む、(iii) `Cargo.lock` の byte 比較、(iv) 生成した実行ファイルの `NEEDED` 検査、(v) `cargo deny` と `check-forbidden-deps.sh` の実行。§2.3 の実行時プローブは再実施していない（#2074 の結果が不変のため）。

結論（上記の実測の範囲）: `nccl` を許容 feature に加える場合の変更は `Cargo.toml` の 1 行、`.claude/rules/deps-policy.md` の CUDA 行と `docs/license-matrix.md` の cudarc 行の記載更新だけで済む。`Cargo.lock`・crate 集合・リンク要件・既存テストは変わらない。ただしネットワーク層方式（§5 項 3）が未決のうちは `nccl` feature を有効にしても使う側のコードがない。GB10 実機の NCCL 版数・ordinal 数は未確認（§5 項 5）。

### §2.3 実行時プローブ（scratchpad の使い捨て Cargo プロジェクト。リポにはコミットしない）

`is_culib_present()` の「`libnccl` 不在で `false` を返し panic しない」契約はワークスペース内にコードを足さずには検証できないため、scratchpad に独立の `[workspace]` を持つ使い捨てプロジェクトを作成し検証した（`unsafe` はこのプローブにのみ存在し、リポには一切追加しない。`CudaDevice::is_available`〈`crates/backend-cuda/src/device.rs:273-278`〉と同じ根拠: `is_culib_present()` は dlopen 試行のみで事前条件を要求しない non-panicking なプローブである）。

```rust
fn main() {
    // SAFETY: cudarc の is_culib_present は dlopen 試行のみで事前条件を要求しない
    // （backend-cuda の CudaDevice::is_available と同じ根拠。#2074 nccl リンク契約プローブ）
    let cuda = unsafe { cudarc::driver::sys::is_culib_present() };
    let nccl = unsafe { cudarc::nccl::sys::is_culib_present() };
    println!("libcuda present = {cuda}, libnccl present = {nccl}");
}
```

依存: `cudarc = { version = "=0.19.8", default-features = false, features = ["driver", "nvrtc", "dynamic-loading", "cuda-13000", "f16", "nccl"] }`。

- `cargo build --offline`: 成功（cudarc 0.19.8 は同一版。`Locking 25 packages to latest compatible versions` と表示され、**推移的依存はワークスペースの `Cargo.lock` とは非同一**〈`zerocopy v0.8.57` を解決。ワークスペース側は `0.8.56`〉。`--offline` は `--locked` と異なり lockfile 固定を意味しないため、この scratchpad プローブは cudarc 本体の版一致のみを裏付け、ワークスペース全体の再現性検証ではない）。
- `cargo run --offline`: **`libcuda present = true, libnccl present = false`・exit 0**（panic なし）。CI（GitHub ホステッド `ubuntu-latest`。driver も NCCL も非搭載）では `libcuda present = false` になる想定であり、本実測とは別状態であることに注意する。
- `cargo tree --offline | grep -c libloading`: **1**（`libloading` が既存推移的依存であることを確認。新規クレートの追加なし）。

### §2.4 結論

- `dynamic-loading` 下では `nccl` feature の有効化は link-time に `libnccl` を要求しない（**実証済み**。§2.2 コマンド 1・3・4・5）。
- 実行時は `is_culib_present()` probe が `libnccl` 不在で `false` を返し panic しない non-panicking な設計であることを確認した（§2.3）。ただし `culib()`（本体呼び出し用）は `panic_no_lib_found` で panic する設計のため、実装着手時には CUDA driver と同型の「probe → 型付きエラー」設計（`crates/backend-cuda/src/device.rs:287-293` の `CudaError::DriverUnavailable` パターン）が必要になる。この設計自体は本イシューのスコープ外（実装しない）であり、格上げ条件 (c) に記載する。

## §3 格上げ条件 (a)〜(e) 一覧と充足状況

| 条件 | 内容 | 現時点の記載 |
|---|---|---|
| (a) | cudarc `nccl` feature が `dynamic-loading` 下で link-time の `libnccl` を要求しない（CI `build-no-cuda-toolkit` 不変条件と両立）こと | **達成済み**（本イシュー実測。§2.2〜§2.3） |
| (b) | デバイス列挙 API（`Device::available()` 相当）が facade 公開面に存在すること | **達成済み**（#1614 `fandhe_ai::available_devices()`〈#2074 時点 `crates/facade/src/lib.rs:602`、#2612 時点の現行は `crates/facade/src/lib.rs:772`〉。#1628 doc §6 前提 2「`Device::available()` 未実装」は陳腐化しており本 doc で訂正する） |
| (c) | ネットワーク層方式（#1628 §4 案 A〜D）の確定と、それに伴う cudarc 許容 feature 列挙拡張（`.claude/rules/deps-policy.md`）・新規区分の要否について個別ユーザー承認があること。NCCL 採用時は `is_culib_present()` probe → 型付き `CudaError` variant（`crates/backend-cuda/src/device.rs:287-293` と同型）で panic 経路（`culib()` の `panic_no_lib_found`）を閉じる設計が確認されていること | 未達（承認事項） |
| (d) | 実機側の分母: 単一ノード内に複数 CUDA ordinal を持つ実機が確保され（複数ノード構成は §3a の初期スコープ外のため本条件では満たさない。複数ノードは将来の別段階として別途条件を定める）、インストール済み NCCL 版数が cudarc ピン（`nccl-02030` = NCCL 2.30 系）と整合することがプローブ記録されていること。GB10 は現状 `nvidia-smi -L` 1 行のみの記録（`docs/perf/gemm-peak-memory-measurement.md:66`）で複数 ordinal の記録がないため、単一ノード内複数 GPU という初期スコープ自体の実現可能性を実機で確認する必要がある | 未達（実機未検証。GB10 の NCCL 版数・ordinal 数のプローブは未実施で、数値の記録はない。申し送りは `docs/perf/logs/ddp-nccl-link-contract-2074/README.md` 参照） |
| (e) | 依存追加なし（新規クレートなし・許容依存区分内）で成立し、REQ-2 統一複合判定・tolerance・baseline を一切変更せずに all-reduce 後の勾配（平均化の縮約順序）の数値一致検証が成立する設計が確認されていること | 未達 |

**Could→Should 案（量子化表の f・g と同型の 2 段目。文案に含める）**:

| 条件 | 内容 | 現時点の記載 |
|---|---|---|
| (f) | 実測でのスケーリング効率（例: 2 GPU で 1 GPU 比の学習スループット改善）が確認されていること | 未達 |
| (g) | facade 公開面の非破壊拡張（REQ-12「融合制御 API を提供しない」との整合）が確認されていること | 未達 |

## §3a FSDP・その他の並列化方式の扱い（#2612 で再確認）

スコアボード（`docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html:100`）の PyTorch 列には「DDP/FSDP」が載っている。spec 側の扱いを再確認した結果を記録する。

- **検索**: `grep -rniE 'fsdp|fully ?sharded|zero-?[123]|tensor parallel|pipeline parallel|model parallel' docs/spec`（submodule `2e998dd7`）の結果は **0 件**。FSDP・ZeRO・sharded・tensor／pipeline／model parallel のいずれも spec に記述がない。実装リポ側では `docs/compat-feature-gap.md:52` が「多 GPU 学習（DDP・model/tensor parallel）はない」と記すのみである。
- **spec の関連記述は 3 つだけ**: (1) 除外事項「分散学習・量子化の網羅対応」（`04-requirements.md:358`。分散学習は Won't・条件整理なし）、(2) REQ-9 Tier 2 の「複数 GPU／DDP」（`:232`。(1) に従属）、(3) 「引き続き対象外」の「分散 RPC」（`:233`）。FSDP は明示されていないが、(1) の包括的な「分散学習の網羅対応」に含まれ Won't と読む。
- **本提案の対象**: 格上げ条件表の対象は **DDP（パラメータ複製＋勾配 all-reduce によるデータ並列。初期スコープは単一ノード内の複数 GPU）に限定**する。
- **対象外（Won't・条件整理なしのまま）**: FSDP（PyTorch `FullyShardedDataParallel`）・ZeRO 系の optimizer state／勾配／パラメータ分割・tensor／pipeline／model parallel・TF の `MultiWorkerMirroredStrategy`／`ParameterServerStrategy` 相当。これらの格上げ条件は本 doc で創作しない。扱う場合は DDP の格上げ後に別の (b) 形式提案として検討する（§5 の項 6）。
- **分散 RPC**（`torch.distributed.rpc` 相当）は REQ-9 の「引き続き対象外」であり、DDP の格上げとは別項目。本提案はこの列挙を変えない。

## §4 spec (b) 形式提案文案（起票用 draft。未起票）

以下は `docs/spec-proposal-req2-candle-parity-tolerance.md` §2 と同型の「タイトル案 + ````markdown フェンスの本文案」形式で用意した draft である。**ユーザー承認（§5 の項 1）を得るまで実起票はしない。**

（#2612 で最新化。#2074 版との差分は §9.1。起票時は spec 側の現行行番号を再確認すること。）

**タイトル案**:

```
docs(requirements): 除外事項「分散学習・量子化の網羅対応」に DDP（複数 GPU）の格上げ条件表を新規追加する（実装リポ Fandhe-AI/fandhe-ai#2074・#2612 提案）
```

**本文案**:

````markdown
## 背景

除外事項「分散学習・量子化の網羅対応」（`04-requirements.md` 該当箇所。2026-09-29 の REQ-9
追記〈実装リポ #2193／#2194〉後も、分散学習は「Won't（条件整理なし）」のまま）は、量子化
（FP8/INT8 GEMM）には格上げ条件表（Won't→Could の a〜e、Could→Should の f・g）を
定義済みだが、分散学習（複数 GPU／DDP）には「Won't（条件整理なし）」とのみ記載され、
再開の道筋自体が未定義である。実装リポ側の設計記録（`docs/facade-multi-gpu-ddp-decision.md`
〈#1628〉・`docs/ddp-grade-up-conditions.md`〈#2074〉）でこの非対称を確認し、量子化と
同形式の格上げ条件表を提案する。実装リポは `fandhe-ai =0.10.0` を crates.io へ出荷済みで
公開 API の非破壊が契約になっている。比較スコアボード上では PyTorch 列に「DDP/FSDP」、TensorFlow 列に
「分散戦略」が載っており、fandhe-ai 側との差分が残っている（FSDP は spec に記述がなく、本提案の
対象外とする）。

## 提案: DDP 格上げ条件表の追加

除外事項「分散学習・量子化の網羅対応」の分散学習部分に、量子化と同型の 2 段構成
（Won't→Could の a〜e、Could→Should の f・g）の格上げ条件表を追加する。

| 現状 | 格上げ先 | 格上げ条件（すべて満たすこと） |
|------|---------|------------------------------|
| 対象 | DDP（パラメータ複製＋勾配 all-reduce のデータ並列）に限る | FSDP／ZeRO・tensor／pipeline／model parallel は対象外で、本項目の Won't（条件整理なし）に残す |
| Won't | Could | (a) cudarc `nccl` feature が `dynamic-loading` 下で link-time の `libnccl` を要求しないこと（実装リポで実証済み・実装リポ Fandhe-AI/fandhe-ai#2074）。(b) デバイス列挙 API（`Device::available()` 相当）が facade 公開面に存在すること（実装リポで実装済み・Fandhe-AI/fandhe-ai#1614）。(c) ネットワーク層方式の確定と、それに伴う依存追加（cudarc `nccl` feature の許容依存列挙への追加を含む）についての個別ユーザー承認があること（未達）。なお許容依存は現行 10 区分（本体直接依存は第 1〜8・第 10 区分）で、ネットワーク／分散通信の区分はない。本提案は既存の CUDA 区分の feature 列挙拡張に限り、新規区分は追加しない。(d) 実機側の分母（単一ノード内に複数 CUDA ordinal を持つ実機。複数ノード構成は初期スコープ外）が確保され、NCCL 版数の整合がプローブ記録されていること（未達・実機未検証）。(e) 依存追加なし（新規クレートなし）で REQ-2 統一複合判定・tolerance・baseline を変更せずに all-reduce 後の勾配の数値一致検証が成立する設計が確認されていること（未達）。 |
| Could | Should | Could の条件に加え、(f) 実測でのスケーリング効率の確認、(g) REQ-12 との整合を保った facade 公開面の非破壊拡張の確認（いずれも未達）。 |

**承認時に固定する契約**（格上げ後に新 REQ を起票する際も引き継ぐ前提。先取りの設計確定
ではない）: REQ-2・REQ-7 の既存 tolerance と REQ-8 の下限値は変更しない／カーネル側の
手動境界チェックを省略しない（REQ-8 受け入れ基準と同一）／許容依存区分（deps-policy.md）
の外側への新規区分追加はしない（cudarc の feature 列挙拡張のみ）／公開 API の非破壊（拡張は追加
API・opt-in・`#[non_exhaustive]` enum への variant 追加に限り、出荷済み 0.10.0 の公開面を壊さない）／
CI `build-no-cuda-toolkit` の不変条件（CUDA toolkit・NCCL 非搭載でもビルドが成立すること）の維持。

**実装リポ側との取り決め**: spec 側で本提案が承認されるまで、実装リポは DDP の通信層・
NCCL 呼び出しコードを起票・実装しない。

本提案は除外事項の Won't 判断自体・REQ-9 Tier 2 の従属関係・「引き続き対象外」の分散 RPC・
Phase 4 判定追補のスコープ件数（Should 8・Could 1・Won't 11）・REQ-2／REQ-7／REQ-8 を変更しない。
````

## §5 ユーザー承認事項（未実施）

1. spec リポ（Fandhe-AI/fandhe-ai-spec）への §4 (b) 形式提案の起票可否。#1964（2026-09-17）で承認されたのは「現状維持（選択肢 C）」であり、起票（同 issue の選択肢 D）は未承認。判断の選択肢: (i) 現状維持を継続する／(ii) §4 文案を spec リポへ起票する。
2. cudarc 許容 feature 列挙への `nccl` 追加（`.claude/rules/deps-policy.md` 更新）の可否（条件 (c)）。**実測済み**（§2.2 の #2613 小節）・**承認依頼済み**（#2613 コメント）・**承認未取得**。判断の選択肢: (i) 保留（#1964 の現状維持 C と整合）／(ii) 承認し、別 issue で deps-policy・license-matrix・`Cargo.toml` の 1 行追加を実施する。
3. ネットワーク層方式（#1628 §4 案 A〜D）の選択。
4. 実装着手時の新規 `unsafe`（NCCL FFI 境界）・facade 公開面拡張・`CudaError` variant 追加の可否。
5. GB10 実機での NCCL 版数・ordinal 数プローブの実施（本イシューでは申し送り。`docs/perf/logs/ddp-nccl-link-contract-2074/README.md` 参照）。
6. FSDP 等（§3a の対象外方式）を別の (b) 形式提案で扱うかどうか。扱う場合も DDP の格上げ後に検討し、本 doc では条件を創作しない。

## §6 スコープ外・申し送り

- GB10 実機での NCCL 初期化プローブ・2-GPU PoC 設計（`docs/perf/logs/ddp-nccl-link-contract-2074/README.md` へ申し送り）。
- all-reduce カーネル・NCCL API 呼び出しの実装。
- cudarc 許容 feature 列挙の拡張（`deps-policy.md` 更新）そのもの。

## §7 セキュリティ観点

#1628 doc §6 のセキュリティ観点留意点を継承する（コード変更を伴わないため対策の実装は不要）。

- **A08（ソフトウェア・データ整合性）**: 複数ノード分散を将来実装する場合、rank／world discovery やパラメータ複製の経路は改竄・なりすましに対する検証が必要になる。
- **A03 に類する論点**: 通信層を自作する場合、受信メッセージ長・shape 情報の検証が必要になる。
- **A06（脆弱・古いコンポーネント）**: 本イシューでは依存の版数・feature 集合を一切変更していない。`--features cudarc/nccl` は CLI 一時指定であり `Cargo.toml` に残していない。`nccl` feature は新規クレートを引き込まない（§2.2 コマンド 1 で確認）。
- **`unsafe`**: scratchpad プローブ内の `is_culib_present()` 呼び出しのみ（dlopen 試行・事前条件なし）。リポには一切追加していない。

## §8 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/facade-multi-gpu-ddp-decision.md` | 段階 0（現時点非対応）確定の元となった設計判断記録（#1628） |
| `docs/compat-api-scope.md:512` | 「#1628 の設計記録は…完了した」の既存確定記述 |
| `docs/compat-feature-gap.md:334` | 「複数 GPU・`DataParallel`/`DDP`」行の現状評価（なし・難度 XL） |
| `docs/spec/04-requirements.md:232-235, 358-365`（#2612 時点。submodule `2e998dd7`） | Tier 2「複数 GPU／DDP」・「引き続き対象外」（分散 RPC）・#2193／#2194 追記・除外事項と量子化の格上げ条件表（現行行番号） |
| `crates/facade/src/lib.rs:772`・`Cargo.toml:137-143`（#2612 時点） | `available_devices()` と cudarc の workspace 依存（版 `=0.19.8`・feature 不変）の現行位置 |
| #1964 のコメント（2026-09-17） | 量子化・DDP の格上げ条件の再棚卸しで「現状維持（C）」を承認。spec 起票（D）は未承認 |
| `docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html:100`・`docs/compat-feature-gap.md:52` | スコアボードの PyTorch 列「DDP/FSDP」・実装リポ側の多 GPU 学習の現状評価 |
| `docs/spec/04-requirements.md:222-235`（#2074 時点） | REQ-9 2026-09-12 追記の Tier 2 列挙・除外事項への従属関係 |
| `docs/spec/04-requirements.md:356`（#2074 時点。現行は `:358`） | 除外事項「分散学習・量子化の網羅対応」（分散学習は Won't・条件整理なし） |
| `docs/spec/04-requirements.md:357-363`（#2074 時点。現行は `:359-365`） | 量子化の格上げ条件表（a〜g）・承認時固定契約・実装リポ側との取り決めの記載形式（本 doc §3・§4 が踏襲した precedent） |
| `crates/facade/src/lib.rs:602`（#2074 時点） | `fandhe_ai::available_devices()`（#1614）。#1628 doc §6 前提 2 の陳腐化を訂正する根拠 |
| `crates/backend-cuda/src/device.rs:273-293` | `is_culib_present()` probe → `CudaError::DriverUnavailable` の既存パターン |
| `Cargo.toml:122-128`（#2074 時点） | workspace の cudarc feature 指定（`dynamic-loading` を含む。nccl 非有効化） |
| cudarc `=0.19.8` `Cargo.toml:127, 139` | `nccl = ["nccl-02030"]`・`nccl-02030 = ["driver"]` |
| cudarc `=0.19.8` `build.rs:149-150, 186-205` | `dynamic_linking()` が `dynamic-linking` feature 限定で呼び出される経路・`cargo:rustc-link-lib=dylib=nccl` 発行条件 |
| cudarc `=0.19.8` `src/nccl/sys/mod.rs:10-12, 1363-1385` | `dynamic-loading` feature 下の NCCL 遅延シンボル解決・`libnccl` dlopen |
| cudarc `=0.19.8` `src/nccl/mod.rs:13-18` | `pub mod sys` 公開・到達性 |
| `docs/perf/gemm-peak-memory-measurement.md:66` | GB10 実機の `nvidia-smi -L` 実測（複数 ordinal の記録なし） |
| `docs/spec-proposal-req2-candle-parity-tolerance.md` §2 | (b) 形式起票用本文の precedent 形式 |
| `docs/perf/logs/adam-device-step-cuda-2069/README.md` | 実機未到達時の申し送り README の構成 precedent |
| `docs/perf/logs/ddp-nccl-link-contract-2074/README.md` | 本イシューの Linux 側実測ログ・GB10 実機申し送り |

## §9.1 改訂履歴（#2612）

§4 文案の #2074 版からの差分（コード変更・spec 起票・承認の代行はなし）:

- タイトル案に #2612 を併記した。
- 背景: 2026-09-29 の REQ-9 追記（#2193／#2194）後も分散学習は Won't（条件整理なし）のままであること、`fandhe-ai =0.10.0` 出荷済み、スコアボード上の DDP／FSDP の差分を追記した。
- 条件表: 対象を DDP に限る行を追加した（FSDP／ZeRO・tensor／pipeline／model parallel は Won't に残す）。(c) に現行の許容依存区分（第 1〜8・第 10 区分、ネットワーク区分なし）との整合を、(d) に「実機未検証」を明記した。
- 承認時固定契約: 公開 API の非破壊と CI `build-no-cuda-toolkit` 不変条件の維持を追加した。
- 不変事項: 「引き続き対象外」の分散 RPC・スコープ件数（Should 8・Could 1・Won't 11）・REQ-2／REQ-7／REQ-8 を追加した。
- §3a（FSDP の扱い）・§5 項 6 を新設し、§0・§8 に最新の行番号を追記した。§2（実測）は変更していない。

## §9.2 改訂履歴（#2613）

- §2.2 に「#2613 追記」小節を新設した（マニフェスト変更時の `Cargo.lock` 不変・crate 集合差分ゼロ・feature 伝播・`NEEDED` 検査・`cargo deny`）。#2074 の §2.2 表と §2.3・§2.4 は書き換えていない。
- §0 に #2613 の要約を、§5 項 2 に実測済み・承認依頼済み・承認未取得の状態を追記した。
- コード・依存・`Cargo.toml`／`Cargo.lock`・deps-policy・license-matrix・`docs/spec/` は変更していない。承認の代行も spec 起票もしていない。
