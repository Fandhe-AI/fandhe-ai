# 依存管理規約（REQ-1 v2）

## 適用範囲（本体 workspace 限定）

本規約（許容依存 8 区分・バージョン固定・ライセンス要件）が統制するのは
**本体 workspace**（ルート `Cargo.toml`／`Cargo.lock`）の依存グラフである。

`scripts/bench/oss-gemm-compare/`（本体 workspace 外の独立 Cargo プロジェクト。
`[workspace]` を空テーブルで持ち本体 workspace の member ではないため、本体
workspace の依存グラフ・CI の依存禁止検査の走査対象には現れない構成）に限り、
`matrixmultiply =0.3.11`・`gemm =0.19.0`（いずれも許容依存 8 区分の対象外）を、
以下の条件を満たす場合に限り本規約の統制対象外として扱う（2026-08-20 ユーザー
承認。イシュー #755）:

1. 本体 workspace（ルート `Cargo.toml`／`Cargo.lock`）への混入を禁止する
2. 本パッケージ専用の `deny.toml` によるライセンス監査（`cargo deny --manifest-path
   scripts/bench/oss-gemm-compare/Cargo.toml --locked check licenses sources`）を
   CI（`.github/workflows/ci.yml` の `deps-forbidden` ジョブ）へ必須ステップとして
   組み込む

上記 2 条件を満たさない追加・変更（本体 workspace への混入・CI 監査ステップの
欠落）は通常どおり本規約の許容依存 8 区分の対象となりユーザー承認が必要。本例外は
`scripts/bench/oss-gemm-compare/` パッケージに限定したものであり、他パッケージ・
本体クレートへは適用されない。ハーネス自体の設計判断・実測記録は
`docs/oss-comparison-harness-decision.md`（イシュー #755）を参照（同ドキュメントは
本規約が定める例外条件に従って書かれる）。

## 許容依存 8 区分（これ以外の追加はユーザー承認必須）

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
