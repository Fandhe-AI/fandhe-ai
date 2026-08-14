# 実機検証環境

実機依存の測定・テスト実行時に、どのマシンで・どうやってビルド・テスト・計測を回すかを判断するための環境構成記録。issue #379（Mac Metal）・#388（DGX Spark CUDA）配下の実装エージェントが本ドキュメント内の情報だけで実行判断できることを前提とする。

> **公開版に関する注記（イシュー #461）**: 本ドキュメントは公開リポジトリ向けに、内部ホスト名・内部 venv パス・常駐サービスの実名をプレースホルダへ置換済み。実値（SSH ホスト名等）は Git 管理外の `docs/real-hardware-verification-env.local.md`（`docs/real-hardware-verification-env.local.md.example` をコピーして作成）を参照する。実行判断に必要な手順の構造・注意点はこの公開版のみで完結する。
>
> **実行前の準備（codex-review 指摘対応）**: 本ドキュメントのコード例は `<cuda-node>` の実ホスト名をシェル変数 `CUDA_NODE` として参照する（`ssh "$CUDA_NODE" '...'` の形。山括弧のプレースホルダをそのまま貼り付けて実行すると、POSIX shell がリダイレクト（`<`）と解釈しホストが存在せず入力元ファイルが見つからないエラーで失敗する）。実行前に `export CUDA_NODE="<実ホスト名>"`（山括弧を含む値は未クォートだと `export` 自体も同じくリダイレクトと誤解釈されるためクォート必須。実ホスト名は `docs/real-hardware-verification-env.local.md` 参照）を設定してから各コード例を実行する。

## 1. 対象と役割分担

| 環境 | 用途 | 実行方法 |
|------|------|---------|
| Mac（Apple M4 Max・64GB・macOS 26.6） | Metal バックエンド実機テスト・ベンチ（#379） | ローカル直接実行 |
| DGX Spark GB10（`<cuda-node>`） | CUDA バックエンド実機テスト・ベンチ（#388） | SSH リモート実行 |

Mac 上では CUDA テストは実行不可（hardware 非対応）。

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
  cargo test -p backend-cuda --release -- --ignored --nocapture'
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
- **ビルドキャッシュは同期ツリー外の `$HOME/work/target-rust-ai-library` に置く**（`CARGO_TARGET_DIR` で指定）。同期ツリー内に `target/` を置くと `--delete-excluded` で消えるため、この分離が前提。この構成なら再 rsync 後もキャッシュが残り warm ビルドになる（実測: 再 rsync 後も 176MB のキャッシュが残存し `--ignored` テストが 0.29s で pass）
- `~/work/rust-ai-library`（末尾 `-run` なし）は 2026-08-04 時点の古い checkout。使わない
- `docs/spec`（submodule）の実体も worktree のコピーとして転送される（PyTorch 参照スクリプトが submodule 配下にあるため必要）

## 4. ビルド・テスト実行

### 4.1 基本形式

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-rust-ai-library \
  cargo test -p backend-cuda --release -- --ignored --nocapture'
```

### 4.2 パフォーマンス

- cold build（`cargo build -p backend-cuda --release --all-targets`・外部 `CARGO_TARGET_DIR` 新規作成時）: 6.28s 実測
- warm build（キャッシュ残存後）: 0.1s 未満
- `tests/device.rs` の `select_device_zero_on_real_hardware`（`--ignored`）: 実機で pass 済み

### 4.3 既知の warning

`cargo build --workspace --all-targets` で `backend-cpu` と `backend-metal` の example `gemm_bench` が出力名衝突 warning を出す。エラーではない。

### 4.4 長時間実行の切り離し

`ssh` は cargo が fd を保持する限り返らないため、ログへリダイレクトして親プロセスから切り離す：

```bash
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  setsid nohup env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-rust-ai-library \
  cargo test -p backend-cuda --release -- --ignored --nocapture \
  > $HOME/work/cuda-test.log 2>&1 < /dev/null & echo started'
ssh "$CUDA_NODE" 'tail -5 $HOME/work/cuda-test.log'
```

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

### 6.3 メモリ判定の方法

`nvidia-smi` の `memory.total` は統合メモリのため `[N/A]` 表示で空き判定に使えない。代わり以下を用いる：
- `utilization.gpu`（GPU utilization）
- `nvidia-smi --query-compute-apps`（実行中プロセス一覧）

## 7. Mac 側（Metal）

### 7.1 実機テスト・ベンチ

Mac 上でそのまま実行：

```bash
cargo test -p backend-metal --release -- --ignored --nocapture
cargo bench -p backend-metal --bench gemm_metal_f32 --release -- --ignored
```

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
