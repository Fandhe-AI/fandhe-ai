# fandhe-ai の開発コンテナ。
#
# 環境非依存の開発・検証用（make docker-shell / docker-ci から利用。compose.yaml 参照）。
# コンテナ内で使えるのは CPU（rayon）バックエンドのみ:
# - Metal はホスト macOS 直接実行のみ（コンテナから GPU 不可視。objc2 系は
#   cfg(target_os = "macos") 分離のため Linux コンテナではコンパイル対象外）
# - CUDA は実機（DGX Spark GB10 等）でのホスト実行が前提（実機 CI ジョブを追加する
#   場合は `.claude/rules/ci.md`「実機依存」節）
#   （cudarc の動的ロード方式のため「ビルド」は CUDA toolkit 無しの本コンテナでも成立する）
# Fandhe-AI/rust-ai-library-v1 の Dockerfile と同一方針。
FROM rust:1.88-slim-bookworm

# ビルド・検証に必要な最小ツールのみ導入する（レイヤ削減のため 1 RUN に集約）。
# python3（イシュー #472・PR #626 codex-review 指摘）: make ci の runner-policy
# ターゲット（scripts/check-workflow-runner-policy.sh）が呼ぶ検査本体
# scripts/check-workflow-runner-policy.py は python3 標準ライブラリのみで実装されて
# いる（外部 PyPI パッケージへの依存は deps-policy.md 上ユーザー承認必須のため意図的
# に避けている）が、コンテナ側に python3 自体が無いと fail-closed 設計
# （check-workflow-runner-policy.sh の ensure_python）により make docker-ci が必ず
# 失敗していた。Debian bookworm 標準の python3 パッケージ（追加 PyPI パッケージなし）
# を導入し、README が案内する環境非依存の検証経路（make docker-shell / docker-ci）を
# 復旧する。
# aarch64-apple-darwin ターゲットは `make ci` が `build-cross`（TASK-2.1b・イシュー #50）を
# 経由して必ず要求するため、root（RUSTUP_HOME 既定は /usr/local/rustup で dev ユーザーは
# 書き込み不可）でイメージビルド時に事前導入しておく。未導入のまま dev ユーザーで
# `rustup target add` を実行すると権限エラーになるため（PR #238 Bugbot 指摘）。
#
# stable ツールチェーンの事前導入（イシュー #325）: リポジトリルートの
# rust-toolchain.toml（channel = "stable"。CI ベースラインの rust-base-ci reusable
# workflow が前提とする単一真実源）により、/work でマウントされたワークスペースでの
# cargo / rustup 実行はイメージ既定の 1.88 ではなく stable を解決する。未導入のまま
# dev ユーザーで cargo を実行すると stable の自動インストールが root 所有の
# RUSTUP_HOME への書き込みで権限エラーになるため、イメージビルド時に components /
# target ごと導入しておく（PR #344 Bugbot 指摘）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        git \
        make \
        curl \
        ca-certificates \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy \
    && rustup target add aarch64-apple-darwin \
    && rustup toolchain install stable \
        --component rustfmt \
        --component clippy \
        --target aarch64-apple-darwin

# ホストと UID を合わせやすい非 root ユーザーで作業する（成果物の所有権事故防止）
ARG UID=1000
RUN useradd -m -u "${UID}" dev

# named volume（compose.yaml の cargo-registry / target-cache）は初回マウント時に
# イメージ内の該当パスの内容・所有者を引き継ぐため、dev 所有で事前作成しておく。
# これを行わないとマウントポイントが root 所有になり、非 root の dev ユーザーが
# crate キャッシュ・target へ書き込めず cargo が失敗する。
# /usr/local/cargo は rust イメージの CARGO_HOME（registry 以外の git/ 等にも書き込みが
# 発生するため CARGO_HOME 全体を chown する）。
RUN mkdir -p /usr/local/cargo/registry /work/target \
    && chown -R dev:dev /usr/local/cargo /work

USER dev
WORKDIR /work

# make ci / make deny（deny.toml。TASK-1.3）がクリーンなコンテナでも即実行できるよう、
# cargo-deny をイメージビルド時に導入しておく（Makefile の deny ターゲット自体も
# 未導入なら自動導入する自己修復を持つが、初回 docker-ci でのネットワーク依存を避けるため
# ここで先行導入する。dev ユーザーの CARGO_HOME 配下にインストールされる）。
RUN cargo install cargo-deny --locked

CMD ["bash"]
