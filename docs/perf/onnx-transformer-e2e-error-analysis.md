# `transformer.onnx` e2e REQ-7 誤差超過の原因分析（#413）

イシュー #413「test(interop): REQ-7 transformer e2e のごく小さな誤差超過の解消検討
（条件付き非対応の解消）」の分析記録。#88（`docs/perf/onnx-transformer-e2e-measurement.md`）が
記録した「条件付き非対応」（16,384 要素中 7 要素が REQ-7 判定式を超過）について、実装改善で
解消可能か（帰結 (a)）、判定式の構造的性質が支配的か（帰結 (b)）を実験で切り分ける。

## 結論: 帰結 (b)。実装改善のみでは解消不能（判定式の構造的性質が支配的）

`erf` 近似精度の改善（E1）・全主要オペの累積を f64 化する全面改善（E_all）を順に試したが、
**E_all（最も強い改善）を適用してもなお 1 要素が REQ-7 判定式を超過する**（`max_rel_err=0.0014883588`。
閾値 `1e-3` の約 1.49 倍）。REQ-7 判定式・許容誤差は変更していない
（`.claude/rules/coding-rust.md`・`.claude/rules/security.md` 準拠）。

## 実験設計

`.claude/rules/delegation-impl.md`・`out-of-scope-tracking.md` に従い、判定式・許容誤差・
`tests/onnx_transformer_e2e.rs` の `assert!` には一切手を触れない。原因候補への一時的な
パッチ（f64 化等）は本ドキュメントの実測値記録のみに用い、**いずれもコミットしない**
（`git diff` で `crates/onnx-interop/src/ops/` に差分が残っていないことを各実験後に確認済み）。

決定規則（実装計画 Phase 2 に基づく）:

- E1（`erf` 精度改善）単独で `exceed_count=0` → 帰結 (a)。局所的改善のため FMA 契約
  （CPU 参照実装は `f32::mul_add`。`.claude/rules/coding-rust.md`）を崩さず採用可能
- E1 で解消せず、f64 累積（MatMul/Gemm/LayerNorm/Softmax）が必要 → 累積契約の変更は
  FMA 契約・バックエンド間一貫性設計に波及するため、実装側で先走らず帰結 (b)（提案文書化）
- 最強実験（E_all: E1 + 全累積 f64 化）でも超過が残る → 判定式の構造的性質が支配的と確定
  できるため帰結 (b)

## ベースライン再現

環境: `rustc 1.96.0 (ac68faa20 2026-05-25)`。`transformer.onnx`
（v1 リポ commit `a14568897521f7bea6eac93218fe917cf2a25f04`、sha256
`6f6430e6b99408c949635da16ed7d6e7cdc2a500db050ae80c660b3b8b057b0f`。取得手順は
`crates/onnx-interop/tests/fixtures/README.md` 参照）を用いて

```sh
ONNX_INTEROP_TRANSFORMER_ONNX=<path> \
  cargo test -p onnx-interop --test onnx_transformer_e2e -- --ignored --nocapture
```

を実行し、#88 記録と一致する結果を確認した:

```
EXCEED (b=0,s=5,d=507)  expected=0.00019158138  actual=0.00019303149  abs_err=0.0000014501129 rel_err=0.007529871
EXCEED (b=0,s=12,d=499) expected=-0.00025896312 actual=-0.0002594656  abs_err=0.00000050247763 rel_err=0.0019328804
EXCEED (b=0,s=15,d=447) expected=0.00018594891  actual=0.00018553587  abs_err=0.00000041304156 rel_err=0.0022093821
EXCEED (b=1,s=1,d=369)  expected=-0.000100252546 actual=-0.00009987713 abs_err=0.00000037541759 rel_err=0.0037077349
EXCEED (b=1,s=6,d=326)  expected=-0.000105179904 actual=-0.00010454383 abs_err=0.0000006360715  rel_err=0.005990507
EXCEED (b=1,s=9,d=491)  expected=0.0004965767   actual=0.0004970804   abs_err=0.0000005036709  rel_err=0.0010122476
EXCEED (b=1,s=13,d=16)  expected=-0.0006739199  actual=-0.00067315536 abs_err=0.0000007645576  rel_err=0.0011328124

max_rel_err=0.007529871 (threshold=1e-3) exceed_count=7/16384
```

7 要素全て `|expected| < 1e-3` の近ゼロ値で、abs_err は最大 `1.45e-6` と極小
（#88 の実測 `max_abs_err=2.7418137e-6`（全要素中最大）と整合）。符号は正負混在
（(0,5,507) は actual が expected を上回る、(0,15,447) は下回る）であり、系統的なバイアス
（軸取り違え・符号反転等のオペ実装バグ）ではなく丸め誤差の蓄積であることを示唆する。

## 原因候補と実験結果

| # | 実験 | 変更内容（一時パッチ・非コミット） | `max_rel_err` | `exceed_count` |
|---|------|-----------------------------------|---------------|-----------------|
| baseline | — | なし | `0.007529871` | 7/16384 |
| E1 | `erf` 精度改善 | `crates/onnx-interop/src/ops/activation.rs::erf_approx` を Abramowitz & Stegun 7.1.26（f32・誤差上界 1.5e-7）から、f64 での Simpson 則数値積分（`erf(x) = 2/√π ∫₀ˣ e^{-t²} dt`、2,000 分割）へ置換。既知の erf 参照値との整合を確認済みの高精度近似（誤差 1.5e-7 → 実質的に f64 丸め誤差レベルまで縮小） | `0.006673144` | 7/16384（超過要素の一部が入れ替わる。改善はわずか） |
| E_all | E1 + 累積 f64 化 | 上記 E1 に加え、`ops/matmul.rs`・`ops/gemm.rs` の内積累積（`f32::mul_add` → f64 `mul_add`）、`ops/layer_norm.rs` の平均・分散累積（f64 化）、`ops/softmax.rs` の総和（f64 化）を全て f64 で実施 | `0.0014883588` | **1/16384**（`(b=1,s=6,d=326)` のみ残存。`expected=-0.000105179904` `actual=-0.00010533794` `abs_err=1.580338e-7`） |

E1 単独では 7 要素中の超過は解消せず、むしろ超過要素の内訳が変化した（`(1,9,491)` が解消する一方
`(1,7,68)` が新規に超過）。これは `erf` 単独が支配的要因ではなく、複数オペの丸め誤差が組み合わさって
近ゼロ出力の相対誤差を生んでいることを示す。

E_all（最も強い改善: erf 高精度化 + 全内積・総和累積の f64 化）を適用すると exceed_count は
7 → 1 まで大幅に減少し、max_rel_err も `0.00753` → `0.00149` へ改善するが、**閾値 `1e-3` を
なお約 1.49 倍超過する要素が 1 件残る**。決定規則により、この時点で「実装改善では解消不能」と
確定できる。

## 結論の妥当性についての考察

- E_all 後も残る 1 要素は `expected=-0.000105`・`abs_err=1.58e-7` という、依然として近ゼロ値かつ
  絶対誤差が極小（REQ-2 の絶対誤差許容 `1e-5` の約 1/63）のケースである。この規模の絶対誤差は、
  比較対象である PyTorch 参照値自体が f32 で計算・保存されている以上、PyTorch 側の丸め誤差
  （典型的に f32 の 1 ULP 〜 数 ULP、`1e-4` 近傍の値では概ね `1e-11`〜`1e-7` オーダー）と
  同じ桁数に達する。すなわち「本実装を無限精度（真の実数演算）に近づけても、PyTorch 参照値
  自体の f32 丸め誤差が下限として残る」ため、これ以上の実装側改善では原理的に埋められない
  残差が存在する
- REQ-7 判定式 `abs_err / (|ref| + 1e-6) ≤ 1e-3` は、`|ref|` が `1e-4` オーダーまで小さくなると
  分母がほぼ `|ref|` そのものになり、絶対誤差にして `|ref| × 1e-3 ≈ 1e-7` 程度の誤差でも
  相対誤差が閾値を超える。この分母の挙動自体は判定式の設計（実装の不具合ではない）であり、
  「近ゼロ参照値での相対誤差判定は原理的に脆弱になる」という構造的性質が、今回の残存 1 要素の
  直接原因である
- 以上により、本イシューの帰結は **(b) 判定式の構造的性質が支配的** と結論する

## 実装側の改善余地（参考記録・未採用）

E_all の結果は「累積精度を上げるほど exceed_count が減る」こと自体は実証しており、全くの
無駄ではない。ただし:

- MatMul/Gemm/LayerNorm/Softmax の累積を f64 化する変更は、CPU 参照実装の丸め方針（FMA 契約。
  `f32::mul_add`。`.claude/rules/coding-rust.md`）を変更する設計判断であり、GPU バックエンド
  （CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）との丸め方針統一にも波及するため、
  本イシューの自動運転スコープでは実装しない（`delegation-impl.md`「実装 Agent にガードレール
  閾値・テスト許容誤差を緩和させない」と同種の慎重姿勢を、丸め契約の変更にも適用した）
- 仮に f64 累積を正式採用しても、上記の考察のとおり PyTorch 参照値自体の f32 丸め誤差により
  残存 1 要素の解消は保証されない

## spec への提案（ユーザー承認・spec リポ側対応が必要。本リポでは決定しない）

`docs/spec/`（正本 submodule）は本リポで編集しない。以下は選択肢の提示に留め、採否・
`Fandhe-AI/fandhe-ai-spec` への反映はユーザー判断事項とする
（`.claude/rules/out-of-scope-tracking.md`）。

1. **REQ-7 判定式を REQ-2 と同型の複合判定へ改定する案**: `相対誤差 1e-3 未満 または 絶対誤差
   1e-5 未満`。今回の超過要素はいずれも絶対誤差が `1e-5` を大きく下回る
   （E_all 後の残存要素も `1.58e-7`）ため、この複合判定であれば全要素が基準内に収まる
2. **判定式は維持しつつ、近ゼロ参照値（例: `|ref| < 1e-3`）を判定対象外とする除外規定を
   追加する案**: 判定式自体は変えず、近ゼロ値域の構造的脆弱性を明示的にスコープ外とする
3. **現状維持（判定式・許容誤差を変更しない）案**: 「条件付き非対応」の記録を維持し、
   `transformer.onnx` e2e は REQ-7 の参考実測（正式な受け入れ基準達成ではなく）として
   位置づける

いずれの案も REQ-7 判定式・許容誤差の変更を伴うため、`.claude/rules/security.md`
「ガードレール閾値・テスト許容誤差の変更は必ず人間の承認を経る」・`.claude/rules/coding-rust.md`
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」に従い、本セッションでは
判定式・許容誤差・`tests/onnx_transformer_e2e.rs` の `assert!` を一切変更していない。

## 検証（本ドキュメント作成時点）

| 項目 | コマンド | 結果 |
|------|---------|------|
| 実験用一時パッチの非残存確認 | `git diff --stat -- crates/onnx-interop/src/ops/` | 差分なし（全実験パッチは実測後に原状復帰） |
| e2e テスト（変更なし） | `ONNX_INTEROP_TRANSFORMER_ONNX=<path> cargo test -p onnx-interop --test onnx_transformer_e2e -- --ignored --nocapture` | ベースラインと同一結果で失敗（`exceed_count=7/16384`。#88 と同じ「条件付き非対応」状態を維持） |
| 既存テスト回帰 | `cargo test -p onnx-interop`（`--ignored` なし） | pass |
| fmt | `cargo fmt --all -- --check` | pass |
| clippy | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | pass |

## 関連ドキュメント

- `docs/perf/onnx-transformer-e2e-measurement.md`（#88）: 「条件付き非対応」の当初記録。
  本ドキュメントはその後続の原因分析
- `crates/onnx-interop/tests/onnx_transformer_e2e.rs`: REQ-7 判定式の実装（変更なし）
- `crates/onnx-interop/tests/fixtures/README.md`: `transformer.onnx` の出自・取得手順
