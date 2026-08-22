# oss-gemm-compare

CPU GEMM の OSS 直接比較ハーネス（イシュー #755）。本体の現行最適 CPU 経路
（`fandhe_ai_backend_cpu::gemm_blis_parallel`。BLIS 5-loop + rayon 並列。
crates.io 公開向け rename はイシュー #879・`docs/crates-io-naming-decision.md`）を、
`matrixmultiply`・`gemm` crate（いずれも許容依存第 9 区分〈ベンチ比較対象。
`.claude/rules/deps-policy.md`〉として条件付きユーザー承認済み〈2026-08-20〉。
本体 workspace の直接依存〈第 1〜8 区分〉には含まれない）と同一プロトコルで計測する。

設計判断（本パッケージがなぜ本体 workspace 外の独立プロジェクトなのか）は
`docs/oss-comparison-harness-decision.md` を、計測境界・再現手順・実測記録は
`docs/perf/oss-gemm-comparison-baseline.md` を参照。

## 使い方

```sh
cd scripts/bench/oss-gemm-compare
cargo build --release
./target/release/oss-gemm-compare                    # 既定サイズ（512/1024/2048/4096）
./target/release/oss-gemm-compare --sizes 64,128,256  # サイズを明示指定（正整数カンマ区切り）
```

標準出力に JSON Lines（1 行 1 レコード = 1 実装 × 1 サイズ）を出力する。各レコードは
`output_match`（基準実装 `gemm_blis_parallel` との統一複合判定: 相対誤差 1e-3 未満
または 絶対誤差 1e-5 未満での突合結果）と `mismatch_detail`（不一致時のみ詳細文字列、
一致時は `null`）を含む。標準エラー出力には実行時メタデータ（git commit・HW 表記・
rayon スレッド数）と、突合 NG の警告詳細を出力する。

**既定で fail-closed**（レビュー指摘対応。イシュー #755）: 出力突合 NG を 1 件でも
検出したら、全サイズの JSON Lines 出力を終えたうえで非 0 終了する（性能値の正しさを
検証しない既定挙動は許容しない）。既知の限界として、大きい K（1024〜4096）では
OSS 実装間の縮約順序差に由来する丸め誤差の蓄積により、実装バグなしに複合判定を
わずかに超える不一致が生じうることが分かっている
（`docs/oss-comparison-harness-decision.md`「出力突合とその限界」節）。この既知の
限界を理由に既定挙動を非 fatal に戻すことはしない。統一複合判定の許容誤差の値自体
（相対誤差 1e-3・絶対誤差 1e-5）は変更していない（`.claude/rules/coding-rust.md`
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」の保護対象。本ハーネスは
OSS 比較でありバックエンド間比較ではないため直接の適用対象ではないが、数値自体は
予防的に据え置いている）。

## ライセンス注記

- `matrixmultiply` (=0.3.11): MIT/Apache-2.0（デュアルライセンス。crates.io API 実測）
- `gemm` (=0.19.0): MIT（crates.io API 実測）

本パッケージは本体 workspace の member ではなく、独立の `[workspace]` を持つ
Cargo プロジェクトである（ルート `Cargo.toml` / `Cargo.lock` に一切影響しない）。
そのため `docs/license-matrix.md`（本体 workspace 直接依存の第 1〜8 区分の可否表）
には掲載せず、第 9 区分（ベンチ比較対象）の実測記録として上記ライセンス注記を
本 README に個別記載する。

## 依存関係

`Cargo.toml` を参照。`backend-cpu`・`bench-harness` は本体 workspace member への
path 依存（外部依存の新規追加ではない）。`matrixmultiply`・`gemm` は本パッケージ
専用の外部依存として `=x.y.z` 完全固定し、`Cargo.lock` をコミットして
再現性を確保する。
