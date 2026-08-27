# 依存管理規約（REQ-1 v2）

## 許容依存 9 区分（これ以外の追加はユーザー承認必須）

第 1〜8 区分は本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）の直接依存。
第 9 区分（ベンチ比較対象）は `scripts/bench/oss-gemm-compare/` および
`scripts/bench/framework-compare/`（適用範囲拡張。下表参照）限定であり本体
workspace には入らない。CLAUDE.md 等で本体依存の区分数を指して「8 区分」と記述
している箇所は、指している対象（本体 workspace の直接依存）が第 9 区分と異なる
ため矛盾しない。本 PR ではそれらの記述を変更しない。

| 区分 | クレート | 条件 |
|------|---------|------|
| CUDA | `cudarc`（`driver`／`nvrtc`／`dynamic-loading`／`cuda-13000`／`f16` feature） | 無条件依存。動的ロード方式（CUDA toolkit 非搭載環境でもビルド成立）。`cuda-13000` は cudarc の CUDA API バージョン feature（指定必須。未指定ではビルド不能）で、DGX Spark GB10 実機の CUDA 13.0 系・PoC-v2-3／PoC-v2-5 実測構成を踏襲した採用。イシュー #412 でユーザー承認済み。ライセンス実測は `docs/license-matrix.md` 4 節を参照 |
| Metal | `objc2`・`objc2-foundation`・`objc2-metal` | `cfg(target_os = "macos")` 限定 |
| 相互運用 | `safetensors` | ワイヤフォーマット処理のみ（テンソルへのマッピングは自作） |
| 相互運用 | `prost` | ONNX の protobuf デコードのみ。`prost-build`（`protoc` ビルド時依存）は使わない（手書き derive。PoC-v2-6） |
| シリアライズ | `serde`・`serde_json` | 構造化データのシリアライズ |
| CPU 並列 | `rayon` | PoC-v2-1 で採用（naive/blocked 比 約 6〜8.5 倍改善） |
| 数値型 | `half` | f16 型 |
| ベンチ | `criterion` | `dev-dependencies` 限定 |
| ベンチ比較対象（フレームワーク横並び） | `candle-core`・`burn`（およびその推移的依存ツリー。禁止リスト掲載クレートの推移的混入を含む） | **`scripts/bench/framework-compare/`（独自の `[workspace]` を持つ独立 Cargo workspace。本体 workspace 外）限定**の第 9 区分の適用範囲拡張。`=x.y.z` 完全固定（`candle-core =0.11.0`・`burn =0.21.0`・`fandhe-ai =0.3.0`〈crates.io 公開版の自社クレート〉）で、同 workspace の `Cargo.lock` をコミットして再現性を確保する。目的はフレームワーク横並びベンチ（GEMM / MLP 学習 / 推論）の比較対象であり、比較対象という性質上、同 workspace の `Cargo.lock` には依存禁止リストのクレート（`candle-*`・`burn-*`・`cubecl`・`ndarray`・`tch` 等）が**意図的に含まれる**。このため `scripts/check-forbidden-deps.sh` の `lock-all` 走査対象には**含めない**（意図的除外。同スクリプトのコメント参照）。本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）への混入は引き続き禁止で、ルート `Cargo.lock`・`cargo tree` 検査が fail-closed に検出する。専用 `deny.toml` によるライセンス監査は本拡張時点では未導入（依存ツリーが大きく、allow リスト拡張にユーザー承認が必要なため。導入要否は maintainer 判断の残件として PR で明示する）。**本拡張はユーザー（maintainer）承認が確定するまで暫定であり、承認記録は本拡張を導入する PR を出典とする** |
| ベンチ比較対象（OSS GEMM） | `matrixmultiply`・`gemm` | `scripts/bench/oss-gemm-compare/`（`[workspace]` を空テーブルで持つ独立 Cargo プロジェクト）限定。`=x.y.z` 完全固定（`matrixmultiply =0.3.11`・`gemm =0.19.0`）。本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）への追加は禁止（依存禁止リスト検査とは別に、この限定はレビューで担保する）。同ハーネスの `Cargo.lock` も依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）の対象とし、`scripts/check-forbidden-deps.sh` の走査対象へ含める。本パッケージ専用の `deny.toml`（`scripts/bench/oss-gemm-compare/deny.toml`。allow リストは本区分と同一方針）による `cargo deny --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml --locked check --config scripts/bench/oss-gemm-compare/deny.toml licenses sources` を CI（`.github/workflows/ci.yml` の `deps-forbidden` ジョブ）へ必須ステップとして組み込む。**本区分を有効化する PR（イシュー #755・PR #770）は、上記条件（`scripts/check-forbidden-deps.sh` の走査対象化・専用 `deny.toml` による CI 監査ステップの追加・本体 workspace への非混入）を満たして初めてマージ可能とする**。条件を満たさない追加・変更は通常どおりユーザー承認が必要。2026-08-20 ユーザー承認（イシュー #755）。設計判断・実測記録（ライセンス実測値を含む）は PR #770 で記録される `docs/oss-comparison-harness-decision.md`（イシュー #755）を出典として参照する |

## 依存禁止リスト（CI で機械検査。TASK-1.2）

- `burn` 系一式（`burn`・`burn-core`・`burn-store`・`burn-onnx`・`burn-import` 等）
- `cubecl`・`candle`・`tch`・`ndarray`

## バージョン固定

- 許容依存は `Cargo.toml` で **`=x.y.z` 完全固定**とし、`Cargo.lock` をコミットする
- 依存の追加・更新は `docs/license-matrix.md` の更新とセットで行い、**AI 自律メンテナンスの自動適用対象外（人間承認必須）**とする（REQ-5）

## ライセンス要件

- 適合基準は MIT OR Apache-2.0 系（`objc2-metal` は Zlib OR Apache-2.0 OR MIT の三重ライセンス）
- MPL-2.0 等コピーレフトの推移的混入は、feature 除外による回避を**推定で記述せず**、有効化しうる feature 組合せごとの `cargo tree` 実測で個別に適合確認する（旧 issue #2 の教訓）
- 新規追加時は境界の判断軸 a〜e（`docs/spec/01-brainstorm.md`）に基づく判断理由を記録する
