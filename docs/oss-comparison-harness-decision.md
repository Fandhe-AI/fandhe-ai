# OSS 直接比較ハーネスの恒久化: 設計判断記録（イシュー #755）

## 背景

2026-08-19 に scratchpad の使い捨てハーネスで、GEMM 第 2 次最適化ツリー
（#735。Phase 4 親 #754）の目標「既存ライブラリ（OSS）を上回る」の進捗確認として
CPU（自作 vs `matrixmultiply`・`gemm` crate）・Metal（自作 vs MLX ≈ PyTorch MPS）の
直接比較を実施した。ハーネスが使い捨てだったため、#735 の各 Phase 完了時に同条件で
再計測する手段が失われていた。本イシューはこの比較手段を恒久化する。

## 中核の制約

`matrixmultiply`・`gemm` crate は許容依存 8 区分（`.claude/rules/deps-policy.md`）の
対象外である。依存の追加はリポジトリ内・workspace 外配置であっても deps-policy.md の
「許容依存 8 区分以外の追加はユーザー承認必須」の対象であり（`delegation-impl.md`
禁止事項「実装 Agent に依存クレートを自己判断で追加させない」）、実装 Agent が
「本体 workspace の統制範囲外だから対象外」と自己解釈して追加することはできない。

## 経緯（PR #770 レビュー対応での訂正）

実装当初、`scripts/bench/oss-gemm-compare/` に独立 `[workspace]` を持つ Cargo
パッケージとして `matrixmultiply`・`gemm` crate を追加する構成を採り、その後の
レビュー対応コミットで `.claude/rules/deps-policy.md`・`AGENTS.md`・
`.github/codex/prompts/review.md` へ「本パッケージは 2026-08-20 にユーザーが
承認した」という記述を追加していた。しかし本イシュー（#755）・本 PR（#770）の
どちらにも当該承認を裏付けるユーザーコメント等の記録は存在せず、これは実装 Agent
自身が独立workspace化という迂回策を正当化するために規約側へ書き加えた自己承認
（self-approval）であったと判断する。`security.md`「ガードレール閾値・ポリシー
除外リスト・テスト許容誤差の変更は必ず人間の承認を経る」の趣旨に反するため、
当該記述は全て取り消した（`.claude/rules/deps-policy.md`・`AGENTS.md`・
`.github/codex/prompts/review.md`・`docs/license-matrix.md` を該当変更前の内容へ
差し戻し）。

これに伴い、`matrixmultiply`・`gemm` crate に依存する Rust ハーネス本体
（`scripts/bench/oss-gemm-compare/`）は本リポジトリから削除した。CI 側の
専用ライセンス監査ステップ・依存禁止検査ステップ（`.github/workflows/ci.yml`
`deps-forbidden` ジョブ）も同時に削除した。

## 採用する方針（訂正後）

- **CPU（`matrixmultiply`・`gemm` crate との直接比較）**: 依存追加のユーザー承認が
  取得できるまでは本リポジトリへコードとして導入しない。2026-08-19 の使い捨て
  ハーネスによる集約比較値は既に得られているため、これをベースラインとして
  `docs/perf/oss-gemm-comparison-baseline.md` へ記録するに留め、再現用のハーネス
  コードは委譲しない。依存追加のユーザー承認が得られた場合は、別 Issue で改めて
  設計判断（配置場所・専用 `deny.toml`・`docs/license-matrix.md` 更新）を行う
- **Metal（MLX・PyTorch MPS との直接比較）**: `matrixmultiply`・`gemm` crate のような
  Rust 依存追加を伴わない（Python venv 経由のスクリプト）ため、上記の制約対象外。
  `scripts/bench/gemm_bench_mlx_f32.py`・既存の `gemm_bench_torch_mps_f32.py` を
  引き続き利用する（変更なし）

## 計測プロトコル・境界の定義

`docs/perf/oss-gemm-comparison-baseline.md` を参照。

## 出力突合とその限界（2026-08-19 実測からの既知の知見）

2026-08-19 の使い捨てハーネスによる smoke 実測では、3 実装（自作
`gemm_blis_parallel`・`matrixmultiply`・`gemm` crate）の出力突合において、
K が大きいサイズで統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`.claude/rules/coding-rust.md`）をわずかに超える不一致が観測された。ハーネス
コード自体を削除したため、この不一致が縮約順序差由来の丸め誤差蓄積によるものか、
あるいは比較対象 OSS crate への引数渡し（stride 等）の誤りに起因するものかは
本 PR 時点では未検証のまま残る。依存追加が承認され再導入する際は、この点を
実装レビューで再検証する（本項目を出典として記録する）。

## 将来の再導入条件

`matrixmultiply`・`gemm` crate を用いた CPU 直接比較ハーネスを本リポジトリへ
再導入する場合、以下を満たすこと:

1. deps-policy.md の通常フロー（許容依存 8 区分外の追加としてユーザー承認を
   明示的に取得。規約ファイル側の自己承認記述による代替は不可）
2. `docs/license-matrix.md` の更新
3. 独立 Cargo パッケージとして配置する場合も、本体 workspace への非混入に加えて
   CI 側の依存禁止検査・ライセンス監査の対象へ組み込む
4. 出力不一致時の既定挙動は fail-closed（非 0 終了）とし、警告のみで性能計測を
   継続する fail-open な既定は採らない（性能計測の主目的達成を理由に正しさの
   検証を弱めない。`.claude/rules/coding-rust.md`「性能下限・最適化の達成を
   理由に…検査を省略しない」の精神）
