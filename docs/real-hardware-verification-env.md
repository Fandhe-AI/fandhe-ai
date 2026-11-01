# 実機検証環境

実機依存の測定・テスト実行時に、どのマシンで・どうやってビルド・テスト・計測を回すかを判断するための環境構成記録。issue #379（Mac Metal）・#388（DGX Spark CUDA）配下の実装エージェントが本ドキュメント内の情報だけで実行判断できることを前提とする。

> **公開版に関する注記（イシュー #461）**: 本ドキュメントは公開リポジトリ向けに、内部ホスト名・内部 venv パス・常駐サービスの実名をプレースホルダへ置換済み。実値（SSH ホスト名等）は Git 管理外の `docs/real-hardware-verification-env.local.md`（`docs/real-hardware-verification-env.local.md.example` をコピーして作成）を参照する。実行判断に必要な手順の構造・注意点はこの公開版のみで完結する。
>
> **実行前の準備（codex-review 指摘対応）**: 本ドキュメントのコード例は `<cuda-node>` の実ホスト名をシェル変数 `CUDA_NODE` として参照する（`ssh "$CUDA_NODE" '...'` の形。山括弧のプレースホルダをそのまま貼り付けて実行すると、POSIX shell がリダイレクト（`<`）と解釈しホストが存在せず入力元ファイルが見つからないエラーで失敗する）。実行前に `export CUDA_NODE="<実ホスト名>"`（山括弧を含む値は未クォートだと `export` 自体も同じくリダイレクトと誤解釈されるためクォート必須。実ホスト名は `docs/real-hardware-verification-env.local.md` 参照）を設定してから各コード例を実行する。
>
> **ROCm ノード（`<rocm-node>`）**: MI300X のクラウドインスタンスの総称。実ホスト名はシェル変数 `ROCM_NODE`（クォート必須。`CUDA_NODE` と同じ扱い）で参照し、実値は `.local.md` にだけ書く（§9）。
>
> **プレースホルダの約束（2026-10-27 所有者決定）**: `<cuda-node>` は DGX Spark（GB10）の総称で、特定の機体を指さない。測定に使った機体は Git 管理外の `docs/real-hardware-verification-env.local.md` とノード上の `.rev-stamp` の記録で追う。同じ文の中で複数の GB10 を区別する必要がある箇所だけ `<cuda-node-2>` のように番号を付け、番号はその文書の中だけの区別とする（機体に固定しない）。

## 1. 対象と役割分担

| 環境 | 用途 | 実行方法 |
|------|------|---------|
| Mac mini（`<metal-node>`。Apple M4 10 コア・16GB・macOS 27） | Metal バックエンド実機テスト・ベンチ・計測（2026-10-11 から） | SSH リモート実行（§7） |
| 開発機 Mac（Apple M4 Max・64GB） | 実装・ビルド・CPU テスト・CUDA ノードへの転送元。**Metal の計測・実機 parity の記録には使わない** | ローカル直接実行 |
| DGX Spark GB10（`<cuda-node>`） | CUDA バックエンド実機テスト・ベンチ（#388） | SSH リモート実行 |
| MI300X クラウドインスタンス（`<rocm-node>`。所有者が起動・停止。実機 CI なし） | ROCm バックエンド parity 実測（#3159・#3867。手順は §9） | SSH リモート実行 |

Mac 上では CUDA テストは実行不可（hardware 非対応）。

**Metal の実行先の変更（2026-10-11 所有者指示）**: 開発機の M4 Max は他アプリが常駐して負荷が安定せず、計測値・実機テストの再現性を担保できないため、Metal の計測と実機 parity の記録は専用の Mac mini（`<metal-node>`）で行う。2026-10-11 より前の Metal 実測記録は M4 Max の値で、Mac mini（M4）の値とは性能を直接比較できない。記録には必ず `env_info.txt`（チップ名・macOS・rustc）を添え、どちらの機体の値かを明示する。

## 2. CUDA 実機ノード（`<cuda-node>`）

### 2.1 ハードウェア・OS・ツールチェーン仕様（2026-08-09 実測）

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GB10（sm_121）・driver 580.159.03 |
| 統合メモリ | 121GiB（`nvidia-smi` の `memory.total` は `[N/A]` で表示不可） |
| OS | Ubuntu 24.04 aarch64 |
| CPU | 20 コア・空き RAM 約 110GB |
| CUDA | `/usr/local/cuda`・nvcc release 13.0 V13.0.88 |
| Rust | `~/.cargo/bin` に rustup の `stable-aarch64-unknown-linux-gnu` インストール済み |
| C++ | g++ 13.3.0 |

### 2.2 環境変数・PATH の制約

**非ログイン shell（`ssh host 'command'`）の PATH に cargo・nvcc が含まれていない**。以下の形式で実行を指定する（`CUDA_NODE` の設定はファイル冒頭の注記を参照）：

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
  cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture'
```

直接呼び出しの `ssh host cargo ...` は「コマンドが見つかりません」で失敗する。

### 2.3 ネットワーク接続性

| 接続先 | 可否 | 用途 |
|--------|------|------|
| `github.com`（HTTPS） | ✗ 不可 | git clone / fetch は失敗 |
| `crates.io`・`index.crates.io` | ✓ 可能 | `cargo fetch`・キャッシュ参照 |
| `static.rust-lang.org` | ✓ 可能 | rustup component 取得 |

**git を使わずコード転送は rsync で行う**（詳細は 3 節）。

### 2.4 ノード選定理由

他ノード（実名はローカル版 `docs/real-hardware-verification-env.local.md` 参照）は常駐サービスが GPU メモリを占有しており計測に不向き。`<cuda-node>` は常駐サービスの GPU 使用量が小さく GPU utilization 0% のため計測ノードとして最適（常駐サービスの内訳・実測値はローカル版参照）。

## 3. コード転送（Mac → `<cuda-node>`）

git clone / fetch は使わない（ノード側に GitHub 認証鍵が無く HTTPS も不通。過去の `git push ssh://...` が 2 回目以降ハングした実績）。作業 worktree のルートで以下を実行：

```bash
# 転送元コミットを記録してから rsync する（ノード側に .git を置かないため、
# ノード上で検証対象リビジョンを確認する唯一の手段になる）
git rev-parse HEAD > .rev-stamp

# --filter=':- .gitignore' で .gitignore を rsync のフィルタとして適用する
# （rsync は .gitignore を自動参照しないため、指定しないと .env*・
#  .claude/settings.local.json・.venv*/ 等の Git 管理外ファイルまで共有実機へ
#  転送される。codex-review P0 指摘対応）。
# --delete-excluded は受け側に残った除外対象ファイルも削除する（過去の転送で
# 残った管理外ファイルを回収する）。ビルドキャッシュは同期ツリー外の
# CARGO_TARGET_DIR に置くため、この削除では失われない。
# .env* / .claude/settings.local.json / .venv*/ / real-hardware-verification-env.local.md
# は .gitignore でも除外されるが、秘密情報・内部実値の転送は fail-closed で防ぐため
# 明示的にも除外する（多層防御。.local.md は内部ホスト名・パスの実値を持つため、
# gitignore フィルタが省略・誤設定された場合でも共有ノードへ渡さない）。
rsync -a --delete --delete-excluded \
  --filter=':- .gitignore' \
  --exclude '.git/' --exclude '.codex/' \
  --exclude '.env*' --exclude '.claude/settings.local.json' --exclude '.venv*/' \
  --exclude 'real-hardware-verification-env.local.md' \
  ./ "$CUDA_NODE":~/work/rust-ai-library-run/

rm .rev-stamp

# 転送後にノード側で必ずリビジョンを確認する（古いカーネルの数値を記録する事故を防ぐ）
ssh "$CUDA_NODE" 'cat ~/work/rust-ai-library-run/.rev-stamp'

# 秘密情報・内部実値が渡っていないことを確認する（初回・フィルタ変更時）
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  find . -name ".env*" -o -name "settings.local.json" \
    -o -name "real-hardware-verification-env.local.md" | head'
```

### 注意

- ノード側の作業ディレクトリは `~/work/rust-ai-library-run`（ソースのみ。ビルドキャッシュは置かない）
- **ビルドキャッシュは同期ツリー外の `$HOME/work/target-fandhe-ai` に置く**（`CARGO_TARGET_DIR` で指定）。同期ツリー内に `target/` を置くと `--delete-excluded` で消えるため、この分離が前提。この構成なら再 rsync 後もキャッシュが残り warm ビルドになる（実測: 再 rsync 後も 176MB のキャッシュが残存し `--ignored` テストが 0.29s で pass）
- `~/work/rust-ai-library`（末尾 `-run` なし）は 2026-08-04 時点の古い checkout。使わない
- `docs/spec`（submodule）の実体も worktree のコピーとして転送される（PyTorch 参照スクリプトが submodule 配下にあるため必要）

## 4. ビルド・テスト実行

### 4.1 基本形式

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture'
```

### 4.2 パフォーマンス

- cold build（`cargo build -p fandhe-ai-backend-cuda --release --all-targets`・外部 `CARGO_TARGET_DIR` 新規作成時）: 6.28s 実測
- warm build（キャッシュ残存後）: 0.1s 未満
- `tests/device.rs` の `select_device_zero_on_real_hardware`（`--ignored`）: 実機で pass 済み

### 4.3 既知の warning（解消済み）

`cargo build --workspace --all-targets` で `backend-cpu`・`backend-metal`・`facade` の example `gemm_bench` が出力名衝突 warning を出していた（エラーではなかった）。#1919 で example ターゲット名を `cpu_gemm_bench`／`metal_gemm_bench`／`facade_gemm_bench` へ分離し解消済み（ファイル名 `gemm_bench.rs` 自体は各クレートとも不変）。

### 4.4 長時間実行の切り離し

`ssh` は cargo が fd を保持する限り返らないため、ログへリダイレクトして親プロセスから切り離す：

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  setsid nohup env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture \
  > $HOME/work/cuda-test.log 2>&1 < /dev/null & echo started'
ssh "$CUDA_NODE" 'tail -5 $HOME/work/cuda-test.log'
```

### 4.5 setmaxnreg プローブ（#484）の外部タイムアウト運用契約

`crates/backend-cuda/tests/setmaxnreg_probe_{dec,incdec}_{base,accel}_real_device.rs`（4 ファイル）は、命令拒否とハングを区別するための静的な `num_regs` 実行ゲートを持たない設計（`docs/cuda-tensor-core-design.md` §13.1「実行契約」節）。ロードに成功したカーネルは常に起動・同期するため、ハング対策は各テストファイルを個別に**外部タイムアウト付きで実行する運用契約**へ一本化している。4 ファイルとも `--test <ファイル名>` を単体指定し、`timeout` でラップして実行すること（`cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture` で一括実行しない）：

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  timeout 120 cargo test -p fandhe-ai-backend-cuda --release \
      --test setmaxnreg_probe_dec_base_real_device -- --ignored --nocapture'
# 他 3 ファイル（setmaxnreg_probe_dec_accel_real_device / incdec_base / incdec_accel）も同様に個別実行する
```

結果の転記手順は §8「結果の記録先」に従う。

### 4.6 既知の解消済み問題

- **JIT キャッシュ系テストの `os error 22`（イシュー #1107・解消済み）**: GB10（aarch64 Linux）でのみ
  `open_dir_nofollow`（`crates/backend-cuda/src/nvrtc.rs`）がキャッシュルート pin 時に
  `failed to pin cache root …: Invalid argument (os error 22)` で失敗していた事象。原因は
  Linux 用 `O_NOFOLLOW`／`O_DIRECTORY` の自前定数が x86_64（asm-generic）の値のみを定義しており、
  aarch64 の UAPI（`arch/arm64/include/uapi/asm/fcntl.h`）では別値だったこと（aarch64 上で
  asm-generic の `O_DIRECTORY` 値はカーネル的には `O_DIRECT` を意味し、ディレクトリ open が
  `EINVAL` になる）。`target_arch` 分岐で修正済み（対象コメントは同ファイルの `O_NOFOLLOW`／
  `O_DIRECTORY` 定数を参照）。回避手順は不要になったため本節は記録のみ。

### 4.7 module_cache 系実機テストは `--test-threads=1` 推奨（イシュー #1107）

`crates/backend-cuda/src/module_cache_wiring_tests.rs`・`init_cost_diag_tests.rs` はプロセス
ワイド `static` の共有 LRU（`module_cache::KernelModuleCache::global`。既定容量 32）の hit/miss
増分を厳密検査する。他の `#[ignore]` 実機テスト（`kernels_mma.rs`・`kernels_wmma_opt.rs` 等の
shape 特化カーネル群）と同一プロセスで既定の並列テストハーネスにより実行すると、それらが同じ
共有 LRU へ並行 insert し容量超過で evict が起こり、`module_cache_wiring_tests.rs` の hit
アサーションのみ flaky に fail しうる（GB10 実機実測 2026-09-03。実装〈キャッシュキー・容量〉
の欠陥ではなくテストの隔離不足）。lib の `--ignored` テストをまとめて実行する場合は次のように
直列化することを推奨する：

```bash
cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored --test-threads=1
```

### 4.8 sm_121 ISA プローブ（#2122）の外部 timeout 運用契約

`docs/cuda-sm121-isa-probe.md`（#2122）のプローブは、意図的に不正命令（`ILLEGAL_INSTRUCTION` 等）や `barrier.cluster` のデッドロックを起こしうるため、**1 (プローブ, target) = 1 プロセス**で外部 `timeout` を付けて実行する（共有 GB10 ではノードが空いているときに実行する）。個別に手で回さず、`docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh` を使う（起動列は同ディレクトリの `RULE.txt` の `PROCESS:`／`PROBE:`／`TARGET:` 行から導出し、各プロセスの終了コードを proc 記録としてログへ残す。timeout 秒数は同 `RULE.txt` の `PROCESS:` 行の `timeout=` が正で、`orchestrate.sh` はそこから読む。終了コード 124 は timeout として集計される）。出力先に既存ログがあると開始前に停止する。ノード側には `.git` が無いため、転送元で rsync の直前に `.rev-stamp` を 2 行形式（1 行目 HEAD・2 行目 `dirty=<件数>`。§3 の 1 行形式ではなく `docs/cuda-sm121-isa-probe.md` §6.1 の形式）で作る。`orchestrate.sh` は `.rev-stamp` から provenance を読む（無ければ開始前に停止）。

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  ./docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh'
```

実測は 2026-10-01 に本手順で実施済み（正式実行 1 系列・219 プロセスすべて exit 0・`aggregate.py` exit 0。結果は `docs/cuda-sm121-isa-probe.md` §5、生ログと集計は `docs/perf/logs/sm121-isa-probe-2122/`）。legacy は `setmaxnreg_probe_*` 4 件に加え、`tma_probe_real_device` の 3 テストを `tma_probe_real_device@<テスト関数>` として 1 テストずつ別プロセスで再実行する。結果の転記手順は §8「結果の記録先」と `docs/cuda-sm121-isa-probe.md` §7 に従う。

### 4.9 二階微分（`create_graph`）の CUDA parity 雛形（#3776）

雛形は `crates/facade/tests/create_graph_gpu_parity.rs`（登録表 `CASES` に 1 行足して族を増やす。測り方は `docs/autodiff-higher-order-grad-decision.md` §48）。GPU テストは `#[ignore]` で、実装 issue ごとに `create_graph_<族>_cuda_matches_cpu` を 1 本ずつ、§4.1 の形式で `cargo test -p fandhe-ai --test create_graph_gpu_parity -- --ignored --exact --nocapture --test-threads=1 <テスト名>` として別プロセスで実行する（`--release` なし。事前に `--no-run` でビルドを済ませ、各コマンドの前に `--list` で選ばれるテストがちょうど 1 件か確かめる）。事前登録の雛形は `docs/perf/logs/create-graph-gpu-parity-template/RULE.txt`。baseline の置き場 `crates/facade/tests/common/create_graph_gpu_parity_baseline.rs` は空で、行の追加は実機の実測値と所有者の承認 URL が揃った場合に限る。この雛形の時点では実機で実行していない（未実測）。

## 5. PyTorch 参照値の再計測（同一実機）

### 5.1 venv の利用

ノードには system torch が無い。実ホスト上の既存 venv（実パスは `docs/real-hardware-verification-env.local.md` 参照）を**読み取り利用のみ**とする（追加・更新はしない）。venv は torch 2.13.0+cu130（`torch.cuda.is_available() == True`）。

### 5.2 実行手順

実測値・env override（`CUDA_FLOOR_BENCH_PYTORCH_SOURCE` 等）の正本は `docs/perf/cuda-floor-remeasurement.md`。本ドキュメントは実行環境の補足のみを提供する。

## 6. 計測時の GPU 排他性・禁止事項

### 6.1 計測前後の占有状況確認

```bash
ssh "$CUDA_NODE" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; \
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

他プロセスが計測中に現れたランは破棄して取り直す。

### 6.2 常駐サービス

**常駐サービス（実名はローカル版参照）を停止してはいけない**。ノードは 2026-07-12 以降 sudoers の NOPASSWD が削除済みで、非対話 sudo は使用不可。停止・再開は人間の対話的 sudo が必要な運用（先例: 他ノードの常駐サービスをオペレーターが対話的に停止 → ベンチ → 再開・応答確認を実施）。停止が必要と判断した場合は**エージェントが実行せずユーザーへ確認する**。

**2026-08-19 時点の追記（出典: イシュー #739）**: ncu の GPU カウンタアクセスが `RmProfilingAdminOnly`
制限下にあるため、プロファイリング用途向けに sudo NOPASSWD が（オペレーターにより）再設定済みで、
非対話 `sudo ncu` 実行が可能になっている。上記 2026-07-12 時点の「sudoers の NOPASSWD が削除済みで
非対話 sudo は使用不可」という記述は、この ncu プロファイリング用途に限り現状と一致しない。**ただし
常駐サービス停止禁止・停止の判断はユーザー確認必須という規律自体は変更しない**（NOPASSWD の再設定は
プロファイリング実行のみを対象とし、常駐サービスの停止操作を許可するものではない）。sudoers 設定の
対象ユーザー名・コマンドパス等の具体的な設定内容は本ドキュメント（公開版）には書かず、
`docs/real-hardware-verification-env.local.md`（Git 管理外）側で管理する。

### 6.3 メモリ判定の方法

`nvidia-smi` の `memory.total` は統合メモリのため `[N/A]` 表示で空き判定に使えない。代わり以下を用いる：
- `utilization.gpu`（GPU utilization）
- `nvidia-smi --query-compute-apps`（実行中プロセス一覧）

## 7. Mac 側（Metal）

### 7.1 実機テスト・ベンチ（Mac mini `<metal-node>` で実行）

`<metal-node>` の実ホスト名はシェル変数 `METAL_NODE` として扱う（`export METAL_NODE="<実ホスト名>"`。実値は `docs/real-hardware-verification-env.local.md`）。転送は CUDA ノードと同じ rsync 方式（§3 のコマンドの転送先を `"$METAL_NODE":~/work/rust-ai-library-run/` に置き換える。`.git/`・`.claude/worktrees/` は転送しない）で、ビルドキャッシュも同じく同期ツリー外の `$HOME/work/target-fandhe-ai` に置く。非対話 SSH では PATH が通らないため `zsh -lc` で包む:

```bash
ssh "$METAL_NODE" 'cat ~/work/rust-ai-library-run/.rev-stamp'   # 転送リビジョンの確認
ssh "$METAL_NODE" 'zsh -lc "cd ~/work/rust-ai-library-run && \
  CARGO_TARGET_DIR=\$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture"'
ssh "$METAL_NODE" 'zsh -lc "cd ~/work/rust-ai-library-run && \
  CARGO_TARGET_DIR=\$HOME/work/target-fandhe-ai \
  cargo bench -p fandhe-ai-backend-metal --bench gemm_metal_f32 --release -- --ignored"'
```

- Xcode 本体は不要（Command Line Tools のみで可）。Metal シェーダは実行時に `newLibraryWithSource` でコンパイルされる（2026-10-11 実測: `tests/device.rs` の `select_metal_device_on_real_hardware` が pass）
- 内蔵 SSD は 228GiB と小さい。target ディレクトリは `$HOME/work/target-fandhe-ai` の 1 つに保ち、worktree ごとに作らない
- 長時間の計測は SSH 切断で止まらないよう `nohup` で切り離し、ログをファイルへ出して回収する（macOS には `setsid` がない）

### 7.2 PyTorch MPS 参照計測

torch は system には未導入。venv 作成手順（`python3 -m venv .venv-mps-bench` → `pip install torch`）の正本は `docs/perf/metal-f16-vs-mps-f16.md`。venv はリポジトリ管理外に置く（`.venv*/` は `.gitignore` 済み）。

実測スクリプト: `scripts/bench/gemm_bench_torch_mps_f16.py`

イシュー #383 実測時点の実測事実: `python3 -m venv .venv-mps-bench && ./.venv-mps-bench/bin/pip install torch` で torch 2.13.0（Python 3.12.12・`cp312-cp312-macosx_14_0_arm64` ホイール）が導入され、`torch.backends.mps.is_available() == True` を確認済み（Apple M4 Max・macOS 26.6）。venv はワークツリー直下に作成し、計測後もコミットしていない。

## 8. 結果の記録先

実測結果は各 issue 本文が指定する `docs/perf/` 配下の該当ファイルへ記録する。例：
- `docs/perf/cuda-floor-remeasurement.md`（CUDA 参照値）
- `docs/perf/metal-f16-vs-mps-f16.md`（Metal MPS 比較）
- `docs/perf/startup-cost-measurement.md`（起動時間計測）
- `docs/backend-cuda-real-device-testing.md`（CUDA 実機 `#[ignore]` テスト 51 件の実行結果・失敗と対処。#389）

### 重要な制約

**REQ-8 の下限値変更・数値一致許容誤差の緩和はユーザー承認必須**（`.claude/rules/coding-rust.md` 参照）。計測 issue では変更しない。

## 9. ROCm 実機ノード（`<rocm-node>`・MI300X）

MI300X のクラウドインスタンス（所有者が起動・停止する。実機 CI は置かない）で、(A) backend 単体の parity（#3159。いま実行できる）と (B) facade 経由の parity（#3867。条件付きで現時点では実行できない）を測る手順。課金時間を短くするため、GPU を使わない作業はすべて起動前に済ませる（#3779）。本節は手順だけを書き、実測値は書かない（実行できなかった項目は「未実測」のまま残す）。

- 判定規則の正は `docs/perf/logs/rocm-parity-3159/RULE.txt`（事前登録・編集しない）。本節に規則を写さない。記入先は同ディレクトリの `README.md`
- 手配・費用の正は `docs/rocm-cloud-amd-procurement-decision.md`（§2・§4・§7）。契約・起動・停止・課金は所有者が行い、エージェントは行わない
- Linux 開発機や M4 Max の代替値で pass にしない。0 本しか実行されなかった結果も pass の根拠にしない。tolerance・baseline は変えない
- ホスト名は `ROCM_NODE`（`docs/real-hardware-verification-env.local.md` に実値。冒頭注記と同じくクォート必須）で渡す

### 9.1 起動前（GPU を使わない作業）

1. 単価と可用性を再確認する（procurement §4 の判定規則）。約 1 万円を超えそうなら着手せず所有者へ戻す。
2. `export ROCM_NODE="<実ホスト名>"` を設定する（クォート必須）。
3. 転送するコミットを決める。push 済みで、`RULE.txt` を入れたコミット（2026-10-18）より後のものに限る。`git status --porcelain` が空であること。
4. 開発機（ROCm なし）で次を実行する。
   - `cargo test -p fandhe-ai-backend-rocm --release --no-run`（ROCm なしでビルドできること）
   - `cargo test -p fandhe-ai-backend-rocm --release -- --ignored --list 2>/dev/null | grep -c ': test$'` が 16 になること。この値を README の「ignored 本数」の予定値として控える
5. `git rev-parse HEAD > .rev-stamp`（40 桁の小文字 16 進数 1 行だけ。スクリプトはそれ以外を拒否する）。
6. (B) を同じセッションで行えるかを判定する。条件は §9.5 の前提がすべてそろっていること。そろっていなければ (A) だけを行い、(B) のために起動を遅らせない。

### 9.2 起動直後の確認（ノード上・数分）

```bash
uname -m; grep -E '^(NAME|VERSION_ID)=' /etc/os-release
cat /opt/rocm/.info/version; hipconfig --version
rocminfo | grep -E '^[[:space:]]*(Name:[[:space:]]+gfx|Wavefront Size:)'
ls /opt/rocm/lib/libamdhip64.so /opt/rocm/lib/libhiprtc.so
ldconfig -p | grep -E 'libamdhip64\.so |libhiprtc\.so '
```

- `rocminfo`・`amd-smi`・`rocm-smi` の出力全文は貼らない（UUID・BDF を含むため）。上の allowlist の grep に限る
- ローダーは版なしの soname（`libamdhip64.so`・`libhiprtc.so`。`crates/backend-rocm/src/hip.rs`・`hiprtc.rs`）しか試さない。無い場合は probe が false になり 2 回の実行が無駄になる。dev パッケージの導入で解消するか、解消できなければ「未実測（理由: ライブラリ不在）」として止める。対処は記録に残す
- gfx942 でない、または wave64 でない場合は `RULE.txt` の `machine` に当たらない。結果は参考扱いとし所有者へ戻す

### 9.3 転送・ビルド・実行（(A) #3159）

- 転送: §3 の rsync を流用する（転送先は `"$ROCM_NODE":~/work/rust-ai-library-run/`。`--filter=':- .gitignore'` と `.env*`・`*.local.md`・`.claude/settings.local.json` の明示除外を残す）。転送後に `.rev-stamp` を照合し、§3 の find で秘密ファイルが渡っていないことを確かめる
- toolchain: rustup の stable（`rust-toolchain.toml` が stable と rustfmt・clippy を指定）。`CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`（同期ツリーの外）。非対話 SSH の PATH は §2.2 と同じ形で通す
- 実行: ノード上で `bash docs/perf/logs/rocm-parity-3159/run_rocm_parity.sh <tree-abs-path> <out-abs-path>`（ビルド後に run1・run2 を順に実行）。計測中は他の cargo を走らせない。SSH 切断で止まらないよう §4.4 と同じく `setsid nohup` で切り離してログを tail する
- 外部 `timeout` で打ち切った場合は raw を回収せず「中断・未実測」と記録する。合格のための追加実行・差し替えはしない（`RULE.txt` の `runs`）

### 9.4 回収・停止・記入（費用順）

1. 回収するのはマスク済みの出力だけ（`.rev-stamp`・`env_info.txt`・`run1/`・`run2/` の `ignored.log` と `exit_code`）。rsync で `docs/perf/logs/rocm-parity-3159/` へ戻す。`*.raw` は回収しない。
2. 補助診断（任意）: オフラインの `clang -x hip --offload-arch=gfx942 --offload-device-only -O3 -S` で `v_fma_*` を数える。時間に余裕があるときだけ行い、行わなければ記入欄に「未実施」と書く。
3. **停止**: ノード上の作業ツリーと target を消し、所有者がインスタンスを停止・削除する（ストレージ課金も止める）。以降の記入は停止後にオフラインで行う。
4. マスク漏れの手動検査（スクリプトが見るのは hostname・`$HOME`・`$USER` のみ）。回収ファイルに次を当て、該当 0 件を確かめる: IPv4 パターン、`[Uu]uid`、BDF（`[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]`）、事業者名・事業者ドメイン、`.local.md` に書いた実ホスト名（照合はローカルだけで行い実値を出力に残さない）。該当があればそのファイルをコミットしない。
5. §9.6 の対応表に沿って README の記入欄を埋める。

### 9.5 facade 経由（(B) #3867。条件付き・現時点では実行できない）

- 前提（すべて必要）: #3771 で Q1・Q2 が承認済み、spec#87 が受理済み、#3851〜#3853 のマージで `Device::Rocm` が main にある、#3867 が `RULE.txt` とログ置き場 `docs/perf/logs/rocm-facade-3867/` を用意している（本節の作業ではこのディレクトリを作らない）。現時点でこれらは未成立（`docs/backend-rocm-device-exposure-decision.md` §10）
- 予定のテスト（同 §10.4）: `crates/facade/tests/rocm_facade_real_device.rs` の `rocm_available_devices_lists_rocm0_and_tape_for_ok`・`rocm_facade_matmul_parity_vs_cpu`・`rocm_facade_mlp_train_step_parity_vs_cpu`・`rocm_tape_for_out_of_range_ordinal_is_err`
- 最初に `cargo test -p fandhe-ai --test rocm_facade_real_device -- --ignored --list` で 4 本が出ることを確かめる。出なければ「未実測」と記録して止める
- facade は `ureq` と vendored OpenSSL に依存する（deps-policy 第 11 区分）。ノードに `perl`・`make`（build-essential）が要る（backend-rocm 単体では不要）
- (A) の後に同じセッションで行い、2 回目の起動を避ける。判定は REQ-2 統一複合判定のみ

### 9.6 README 記入欄との対応表

`docs/perf/logs/rocm-parity-3159/README.md` の記入欄と出典の対応。

| README の記入欄 | 出典 |
|---|---|
| 実施日 | run 開始時刻の `date -u +%F`（開発機で控える） |
| .rev-stamp | `<out>/.rev-stamp` |
| ROCm 版 | `env_info.txt` の `rocm_version:`（と `hipconfig:`） |
| gfx ターゲット | `env_info.txt` の `Name: gfx…` |
| wave 幅 | `env_info.txt` の `Wavefront Size:` |
| rustc | `env_info.txt` の `rustc:` |
| probe（`is_available()`） | `run*/ignored.log` の `probe_true_on_rocm_host ... ok/FAILED` |
| ignored 本数 | §9.1 手順 4 の `--list` の件数 |
| passed 合計（run1／run2） | `run*/ignored.log` の `test result:` 行の passed 合計。`0 passed` で終わったバイナリが無いこと |
| exit_code（run1／run2） | `run*/exit_code` |
| 判定（pass／fail） | 両方の `exit_code` が 0、passed 合計が本数と一致、全形状で fail 0。どれかが欠ければ fail か未実測 |
| 形状ごと | `grep -P '^ROCM_PARITY\t' run*/ignored.log` |
| 決定性 | `grep -P '^ROCM_DETERMINISM\t' run*/ignored.log`（記録のみ） |
| unsafe 監査 §6 | 完走で間接的に確かめられた項目だけを書き、ほかは「未確認」 |
| linalg・softmax 系・reduce・loss・正規化・paged KV の各「実施日」「判定」と演算（ケース）ごとの表 | `ROCM_PARITY_LINALG`／`_SOFTMAX`／`_REDUCE`／`_LOSS`／`_NORM`／`_PAGED_KV` 行と該当テストの ok/FAILED |
| 各「HIPRTC コンパイル」 | 該当テストがコンパイルエラーで落ちずに走ったか（`ignored.log` のエラー文） |
| 「`v_fma_f64` の有無」・全軸 `sum` の `bit_mismatch` 合計 | §9.4 手順 2 の補助診断と `ROCM_PARITY_REDUCE` の集計。行わなければ「未実施」 |

### 9.7 書かないもの

インスタンス識別子・IP・認証情報・内部ホスト名・ユーザー名・ホームのパス・GPU の UUID／BDF・事業者のアカウント情報は、`<rocm-node>`・`<user>`・`<home>` に置き換えるか収録しない。実値は Git 管理外の `.local.md` にだけ書き、PR・Issue・コミットへ転記しない。
