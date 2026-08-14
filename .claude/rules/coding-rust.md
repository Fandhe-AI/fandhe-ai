# Rust コーディング規約

## 基盤方針（REQ-1 v2、変更禁止）

- **完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）とする。Burn 等の既存 ML フレームワークへの統合は行わない
- 依存は許容依存 8 区分のみ（詳細は [deps-policy.md](./deps-policy.md)）。禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査する（TASK-1.2）
- 互換 API 層（`compat::array`／`compat::Sequential` 相当）は自作コアの上の薄いラッパーに徹する（REQ-9）

## バックエンド構成（REQ-2）

- バックエンド切替は **feature フラグなしの cfg ベース**を基本とする（PoC-v2-5 実証構成）。`cudarc` は無条件依存＋動的ロード（CUDA toolkit 非搭載環境でもビルド成立）、`objc2`・`objc2-foundation`・`objc2-metal` は `cfg(target_os = "macos")` 分離
- バックエンド間数値一致は統一複合判定「**相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満**」（全ペア共通。REQ-2 は TF32 前提の複合指標に改定済み）
- 丸め方針（FMA 契約）をバックエンド間で統一する: CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み）

## コード品質

- `cargo fmt --all`・`cargo clippy --workspace --all-targets --all-features -- -D warnings` を通す（`#[allow]` の安易な追加で黙らせない）
- `unsafe` は FFI 境界（cudarc・objc2 系）等の必要最小限に留め、理由をコメントで明記しレビュー必須とする
- エラーは型付きエラーとし、本番経路で `unwrap()` / `expect()` を使わない
- 依存クレートの追加・更新はライセンス確認（`docs/license-matrix.md` 更新）とセットで行い、ユーザー承認必須（deps-policy.md）

## カーネル実装の境界検査（REQ-8）

- **性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界チェックを省略しない**
- 境界検査を無効化する最適化（ベクトル化ロード・タイル端の分岐削減等）を適用する場合は、シェーダ側で手動境界チェックを維持したうえで行う
- 本規約は CPU（intrinsics）・CUDA（NVRTC/mma）・Metal（simdgroup）の全カーネルに適用する

## テスト・ベンチ

- 受け入れ基準（`docs/spec/04-requirements.md`）に対応するテストを同一 PR に含める
- 実機（DGX Spark GB10・Metal 実機）依存テストは `#[ignore]` で分離し、CI（GitHub ホステッド。[`ci.md`](./ci.md)）で実行可能なテストと区別する
- バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和しない（ポリシー除外リストのブラインドスポット対象）
- ベンチは 5 回計測の中央値を採用し、学習系回帰テストには決定的シード設定ユーティリティを使う

## 関連ルール

- 依存管理は [deps-policy.md](./deps-policy.md)
- コメントは [code-comment-style.md](./code-comment-style.md)
- セキュリティは [security.md](./security.md)
- CI は [ci.md](./ci.md)
