# oss-gemm-compare

CPU GEMM の OSS 直接比較ハーネス（イシュー #755）。本体の現行最適 CPU 経路
（`backend_cpu::gemm_blis_parallel`。BLIS 5-loop + rayon 並列）を、
`matrixmultiply`・`gemm` crate（いずれも本体 workspace の許容依存 8 区分
〈`.claude/rules/deps-policy.md`〉の対象外）と同一プロトコルで計測する。

設計判断（本パッケージがなぜ本体 workspace 外の独立プロジェクトなのか）は
`docs/oss-comparison-harness-decision.md` を、計測境界・再現手順・実測記録は
`docs/perf/oss-gemm-comparison-baseline.md` を参照。

## 使い方

```sh
cd scripts/bench/oss-gemm-compare
cargo build --release
./target/release/oss-gemm-compare                    # 既定サイズ（512/1024/2048/4096）
./target/release/oss-gemm-compare --sizes 64,128,256  # サイズを明示指定（正整数カンマ区切り）
./target/release/oss-gemm-compare --strict-compare    # 出力突合 NG を非 0 終了として扱う（下記参照）
```

標準出力に JSON Lines（1 行 1 レコード = 1 実装 × 1 サイズ）を出力する。各レコードは
`output_match`（基準実装 `gemm_blis_parallel` との統一複合判定: 相対誤差 1e-3 未満
または 絶対誤差 1e-5 未満での突合結果）と `mismatch_detail`（不一致時のみ詳細文字列、
一致時は `null`）を含む。標準エラー出力には実行時メタデータ（git commit・HW 表記・
rayon スレッド数・`strict_compare` の指定有無）と、突合 NG の警告詳細を出力する。

**既定では突合 NG は非 fatal**（プロセスは 0 終了し性能計測を継続する）。本ハーネスの
主目的（#735 各 Phase 完了時に既定引数のまま素朴に再実行して再計測する）を成立させる
ため、既知の限界（大きい K での OSS 実装間の突合が複合判定をわずかに超えうる実測結果。
`docs/oss-comparison-harness-decision.md`「出力突合とその限界」節）により既定引数の
まま非 0 終了して主目的を阻害しないようにしている。突合 NG を検出したら即座に非 0 終了
する従来の fail-closed 挙動が必要な場合（CI での回帰検知等）は `--strict-compare` を
指定する。統一複合判定の許容誤差の値自体（相対誤差 1e-3・絶対誤差 1e-5）は変更していない
（`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和
しない」の保護対象。本ハーネスは OSS 比較でありバックエンド間比較ではないため直接の
適用対象ではないが、数値自体は予防的に据え置いている）。

## ライセンス注記

- `matrixmultiply` (=0.3.11): MIT/Apache-2.0（デュアルライセンス。crates.io API 実測）
- `gemm` (=0.19.0): MIT（crates.io API 実測）

本パッケージは本体 workspace の member ではなく、独立の `[workspace]` を持つ
Cargo プロジェクトである（ルート `Cargo.toml` / `Cargo.lock` に一切影響しない）。
そのため `docs/license-matrix.md`（許容依存 8 区分の対象）の掲載対象外であり、
上記ライセンス注記を本 README に個別記載する。

## 依存関係

`Cargo.toml` を参照。`backend-cpu`・`bench-harness` は本体 workspace member への
path 依存（外部依存の新規追加ではない）。`matrixmultiply`・`gemm` は本パッケージ
専用の外部依存として `=x.y.z` 完全固定し、`Cargo.lock` をコミットして
再現性を確保する。
