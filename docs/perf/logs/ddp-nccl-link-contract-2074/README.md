# DDP nccl リンク契約の Linux 側実測ログと GB10 実機申し送り（#2074）

`docs/ddp-grade-up-conditions.md` §2 の格上げ条件 (a) の実測記録。本ディレクトリは
実行コマンド・保存すべきログ一覧・実測サマリのみを記載する README である（本イシュー
時点ではコード変更を伴わず、CUDA 実機〈DGX Spark GB10〉に本セッションは到達できない
ため、実機側は未実施のまま申し送る）。

## 現状（本イシュー #2074 時点）

- **Linux（本エージェント実行環境。driver あり・toolkit 非搭載・NCCL 非搭載）での実測は完了**（下記コマンド 1〜7。全項目期待どおり）。
- GB10 実機（NCCL インストール済み版数の確認・複数 CUDA ordinal の有無）は本イシューでは未実施。

## 実行コマンド（実施済み・Linux）

```sh
bash scripts/check-cuda-toolkit-absent.sh assert
ldconfig -p | grep -i nccl || echo "libnccl absent"
nvidia-smi -L 2>&1 | head -3

cargo tree --locked -p fandhe-ai-backend-cuda -e normal --prefix none | sort -u > tree-base.txt
cargo tree --locked -p fandhe-ai-backend-cuda --features cudarc/nccl -e normal --prefix none | sort -u > tree-nccl.txt
diff tree-base.txt tree-nccl.txt && echo "NO CRATE DIFF"
cargo tree --locked -p fandhe-ai-backend-cuda --features cudarc/nccl -e features -i cudarc | grep nccl

cargo build -p fandhe-ai-backend-cuda --features cudarc/nccl --locked
cargo build -p fandhe-ai-backend-cuda --features cudarc/nccl --locked --tests
cargo test -p fandhe-ai-backend-cuda --features cudarc/nccl --locked

cargo build -p fandhe-ai-backend-cuda --locked 2>&1 | grep -c "^warning:"
cargo build -p fandhe-ai-backend-cuda --features cudarc/nccl --locked 2>&1 | grep -c "^warning:"

git status --porcelain -- Cargo.toml Cargo.lock
```

## 実行コマンド（GB10 実機。後続セッションでの実施を想定）

```sh
# NCCL インストール済み版数の確認（cudarc ピン nccl-02030 = NCCL 2.30 系との整合確認）
ldconfig -p | grep -i nccl
# 見つかった libnccl のバージョン文字列を確認できるビルドがあれば記録する

# 複数 CUDA ordinal の有無（単一ノード内複数 GPU スコープの実現可能性の前提）
nvidia-smi -L

# Linux 実測と同じ実行時プローブ（scratchpad の使い捨て [workspace] プロジェクト）
# cudarc = { version = "=0.19.8", default-features = false,
#            features = ["driver", "nvrtc", "dynamic-loading", "cuda-13000", "f16", "nccl"] }
cargo run --offline
```

## 実測サマリ（2026-09-22・Linux・本エージェント実行環境）

| # | コマンド | 期待 | 実測 |
|---|---|---|---|
| 1 | `check-cuda-toolkit-absent.sh assert` | PASS | PASS（`OK: CUDA toolkit の構成物は検出されませんでした（非搭載を確認）`） |
| 2 | `ldconfig -p \| grep -i nccl` | 該当なし | 該当なし（`libnccl absent`） |
| 3 | `nvidia-smi -L` | driver 検出 | `GPU 0: NVIDIA GeForce RTX 3060` を検出（driver あり） |
| 4 | `cargo tree`（base vs `--features cudarc/nccl`）差分 | ゼロ | ゼロ（`NO CRATE DIFF`） |
| 5 | `cargo tree -e features -i cudarc \| grep nccl` | `nccl`／`nccl-02030` 有効化 | 確認済み |
| 6 | `cargo build --features cudarc/nccl --locked` | exit 0 | exit 0（3.99s） |
| 7 | `cargo build --features cudarc/nccl --locked --tests` | exit 0 | exit 0（11.06s） |
| 8 | `cargo test --features cudarc/nccl --locked` | 全 pass | 49 passed; 0 failed |
| 9 | 警告数（feature 無し vs あり） | 同数 | 77 / 77（同数） |
| 10 | `git status --porcelain -- Cargo.toml Cargo.lock` | 空 | 空 |
| 11 | scratchpad プローブ（`is_culib_present`） | `libcuda=true, libnccl=false`・exit 0 | `libcuda present = true, libnccl present = false`・exit 0（panic なし） |
| 12 | scratchpad `cargo tree \| grep -c libloading` | 1（新規クレートなし） | 1 |

**注**: scratchpad プローブの `cargo build --offline` は `Locking 25 packages to latest compatible versions` と表示し `zerocopy v0.8.57` を解決した（ワークスペース `Cargo.lock` は `zerocopy 0.8.56`）。cudarc 本体は `=0.19.8` で workspace と同一版だが、`--offline` は `--locked` と異なり lockfile 固定を意味しないため、この scratchpad プローブは推移的依存も含めた完全な再現性検証ではなく cudarc 本体の版一致・実行時挙動の確認に限られる。

## GB10 実機への申し送り（未実施事項）

- インストール済み NCCL 版数と cudarc ピン（`nccl-02030` = NCCL 2.30 系）の整合確認。
- `nvidia-smi -L` での複数 CUDA ordinal の有無確認（単一ノード内複数 GPU スコープの実現可能性の分母）。
- 上記いずれも `docs/ddp-grade-up-conditions.md` §3 の格上げ条件 (d) の充足判定に用いる。

内部ホスト名・パスは記載しない。GB10 実機実測を行う場合はログ内パスを `<home>` へ置換すること。
