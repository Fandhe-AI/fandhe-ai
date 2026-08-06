# rust-ai-library の開発コンテナ。
#
# 環境非依存の開発・検証用（make docker-shell / docker-ci から利用。compose.yaml 参照）。
# コンテナ内で使えるのは CPU（rayon）バックエンドのみ:
# - Metal はホスト macOS 直接実行のみ（コンテナから GPU 不可視。objc2 系は
#   cfg(target_os = "macos") 分離のため Linux コンテナではコンパイル対象外）
# - CUDA は実機（DGX Spark GB10 等）の self-hosted runner / ホスト実行が前提
#   （cudarc の動的ロード方式のため「ビルド」は CUDA toolkit 無しの本コンテナでも成立する）
# Fandhe-AI/rust-ai-library-v1 の Dockerfile と同一方針。
FROM rust:1.88-slim-bookworm

# ビルド・検証に必要な最小ツールのみ導入する（レイヤ削減のため 1 RUN に集約）。
# aarch64-apple-darwin ターゲットは `make ci` が `build-cross`（TASK-2.1b・イシュー #50）を
# 経由して必ず要求するため、root（RUSTUP_HOME 既定は /usr/local/rustup で dev ユーザーは
# 書き込み不可）でイメージビルド時に事前導入しておく。未導入のまま dev ユーザーで
# `rustup target add` を実行すると権限エラーになるため（PR #238 Bugbot 指摘）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        git \
        make \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy \
    && rustup target add aarch64-apple-darwin

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
