# Metal f32 split-K 経路の受け入れ判定方式の決定記録 draft（イシュー #1511）

イシュー #1474（`docs/perf/metal-gemm-splitk-two-pass.md` §5.1〜§5.7）で、Metal f32 split-K
GEMM 経路（`MetalGemm::dispatch_split_k_strided_prepared*`）の受け入れ判定に「実測ベースライン
非後退方式」（`crates/backend-cuda/tests/common/parity_baseline.rs` と同型。以下「baseline 非後退
方式」）を適用する実装を一度導入したが、PR #1496 の codex-review P1 指摘（`docs/cuda-tensor-
core-parity-judgment-decision.md` が根拠とする spec REQ-2「2026-09-02 追記・Tensor Core 経路の
受け入れ判定方式」は CUDA 側 TF32/f16 Tensor Core 経路限定であり、Metal f32 split-K への適用
拡張は別途ユーザー承認が必要）を受けて厳密ゼロ fail 判定（`fandhe_ai_backend_cpu::assert_parity`）
へ差し戻された経緯がある。本ドキュメントは、この適用拡張の是非を判断するための**決定記録
draft** であり、`docs/cuda-tensor-core-parity-judgment-decision.md`（先例）と同じ構成
（背景・候補比較・決定・回帰検出契約・スコープ限定・spec 起票要否・承認記録）で記す。

**本ドキュメントは自動実装フロー（Implement エージェント）が作成した draft であり、末尾
「7. 承認記録」節のとおり未承認である。** `.claude/rules/security.md`「自己修復ループ固有の
ガードレール」・`.claude/rules/coding-rust.md`「テスト・ベンチ」節が要求する「バックエンド間
数値一致テストの許容誤差（判定方式を含む）の変更は必ず人間の承認を経る」を踏まえ、本ドキュメント
自体は判定方式・数値契約ゲート・spec 側への正式起票のいずれも変更しない（詳細は「5. スコープ
限定」を参照）。

## 1. 背景

### 1.1 症状（イシュー #1474 の M4 Max 実測）

split-K 経路は、パス 1 で K 方向を `partitions` 個の区間へ分割してそれぞれ独立に部分和
（f32・`simdgroup_multiply_accumulate` による FMA 連鎖）を求め、パス 2 でパーティション昇順の
固定順序 Neumaier 補償和により結合する構造上、単一の連続 K ループで求める古典（classic）経路の
FMA 累積とは加算の結合順序（associativity）が異なり、丸め誤差の生じ方も異なる。

対象 11 形状（`docs/backend-metal-splitk-decision.md` §3 の 9 形状 + K 端数境界 2 形状）× 4
転置パターン（NN/NT/TN/TT）の全数実測（`docs/perf/logs/metal-gemm-splitk-two-pass-1474/
parity_survey_all_shapes.log`。4 転置パターンは論理的に同一の行列積を異なる物理レイアウトで
計算するだけであり実測でも常に同一集計値になるため `(m, n, k)` 単位で 1 行記録）は次のとおり
（`docs/perf/metal-gemm-splitk-two-pass.md` §5.3 からの転記）:

| (m, n, k) | total | fail_count | mean_abs_diff | max_abs_diff | max_rel_err |
|---|---|---|---|---|---|
| (32, 32, 2048) | 1024 | 0 | 9.04e-6 | 8.39e-5 | 2.02e-4 |
| (32, 32, 4096) | 1024 | 0 | 1.68e-5 | 2.02e-4 | 1.70e-4 |
| (32, 32, 8192) | 1024 | 1 | 3.53e-5 | 3.36e-4 | 1.07e-3 |
| (64, 64, 2048) | 4096 | 2 | 8.76e-6 | 9.54e-5 | 3.88e-3 |
| (64, 64, 4096) | 4096 | 0 | 1.75e-5 | 2.06e-4 | 7.52e-4 |
| (64, 64, 8192) | 4096 | 2 | 3.53e-5 | 3.05e-4 | 2.23e-3 |
| (128, 128, 2048) | 16384 | 2 | 8.94e-6 | 1.18e-4 | 3.35e-3 |
| (128, 128, 4096) | 16384 | 8 | 1.78e-5 | 1.64e-4 | 8.53e-3 |
| (128, 128, 8192) | 16384 | 2 | 3.53e-5 | 3.81e-4 | 1.61e-3 |
| (64, 64, 2056)（K 端数） | 4096 | 1 | 9.06e-6 | 1.26e-4 | 1.27e-3 |
| (128, 128, 2064)（K 端数） | 16384 | 2 | 9.13e-6 | 1.22e-4 | 1.31e-1 |

11 形状中 3 形状（(32,32,2048)・(32,32,4096)・(64,64,4096)）は厳密ゼロ fail を満たすが、残り 8
形状は満たさない。`(128,128,2064)` の `max_rel_err` が突出しているのは、真値が 0 近傍の要素で
桁落ちが起きた（`abs_diff` 自体は他行と同水準）ためであり、K 端数固有の異常ではなく split-K
全般が持つ「近ゼロ要素での相対誤差外れ値」という性質の一例と判断されている（同 §5.3）。

### 1.2 「カーネルのバグではない」ことの確認

以下の実測根拠（`docs/perf/metal-gemm-splitk-two-pass.md` §5.2）により、上記の非ゼロ fail は
split-K カーネル実装の数値バグではなく、パーティション分割という構造そのものに起因する恒常
特性であると判断できる:

- **縮約アルゴリズムの改善では解消しない**: `gemm_splitk_reduce` を単純逐次加算から Neumaier
  改良版 Kahan 補償和へ変更しても fail は解消しなかった（`(32,32,8192)` NN: 是正前後とも
  `fail_count=1/1024`・`max_rel_err≈1.07e-3`）。これは誤差の発生源が縮約側の結合方法ではなく、
  パーティション分割そのもの（各パーティションが独立した FMA 連鎖で部分和を求め、単一の連続
  K ループとは異なる結合順序で最終加算される）にあることを示す
- **最小分割の時点で既に発生**: `(64,64,2048)` を `partitions ∈ {2,4,8,16,32}` で計測したところ、
  最小分割の `partitions=2` の時点で既に一部要素が閾値を超過した。分割数を増やすほど
  `fail_count` は非減少に推移する（2/4/8/16/32 partitions でそれぞれ 0/1/2/2/2 件、4096 要素中）
- **classic 経路は同一入力で bit 完全一致**: 古典経路（`dispatch_strided_tiled_prepared` +
  `tile::select_for_device`）は同一乱数シードの `(32,32,8192)` で CPU 参照実装
  （`matmul_reference_fma`）と bit 完全一致（`fail_count=0/1024`・`max_abs_diff=0.0`）すること
  を別途確認しており、この誤差が split-K 固有の現象であることを裏付ける
- **run-to-run で決定的に再現**: AC-1（`split_k_run_to_run_bit_match_for_target_shapes_and_
  partition_counts`）は 2 回実行で bit 完全一致することを確認済み（`docs/perf/logs/
  metal-gemm-splitk-two-pass-1474/bit_match.log`）。決定的な入力（固定シード）を用いるため
  実機負荷や non-determinism には依存しない

## 2. 候補比較

### 候補 1: 厳密ゼロ fail 判定を維持（現状）

`gemm_splitk_parity.rs` は現在この方式（PR #1496 codex-review 指摘による差し戻し後の状態）。

- **内容**: `fandhe_ai_backend_cpu::assert_parity` による REQ-2 統一複合判定を全要素へ適用し、
  1 要素でも不合格なら panic する
- **不採用理由**: split-K の誤差は §1.2 のとおり構造的特性であり改善不可能と診断済みのため、
  対象 11 形状中 8 形状で `#[ignore]` 実機テストが恒常的に FAIL する既知の状態が残る
  （`docs/perf/metal-gemm-splitk-two-pass.md` §5.5）。加えて公開入口
  `dispatch_split_k_strided_prepared` の自動判定分岐（`should_split_k`）は `SPLIT_K_NUMERIC_
  CONTRACT_APPROVED=false` により classic 経路へ常時フォールバックする設計（同 §5.6）のため、
  厳密ゼロ fail 判定を維持し続ける限り本番結線（`dispatch_auto` への組み込み）も事実上凍結
  される

### 候補 2: 実測ベースライン非後退方式（CUDA 型の適用拡張）

PR #1496 で一度実装され、codex-review 指摘により未使用のまま維持されている方式
（`crates/backend-metal/tests/common/splitk_parity_baseline.rs::BASELINES`。11 行）。

- **内容**: `fail_count`／`mean_abs_diff`／`max_abs_diff`／`max_rel_err`／`total` の 5 点を
  §1.1 の実測表からそのまま記録した baseline と比較し、いずれかが悪化していれば fail する
  fail-closed 非後退検査（`assert_no_split_k_parity_regression`）
- **推奨**: 本決定記録における推奨候補。理由は「4. 回帰検出契約」を参照

### 候補 3: 候補 A′ 救済項（スケール付き絶対誤差救済）を `assert_parity` 本体へ追加適用

`docs/candle-parity-tolerance-contract-decision.md`（イシュー #1241 で確定承認済み）の A-1
形式（`c=0.5` のスケール付き絶対誤差救済項を OR 条件として追加する案）を、Metal split-K の
判定にも使えないか検討した候補。

- **不採用（評価不能につき見送り）**:
  (a) イシュー #1241 のユーザー承認は「framework-compare ハーネス限定」のスコープ
      （`docs/candle-parity-tolerance-contract-decision.md` §8-3「本体 `compare`／
      `assert_parity`／`ParityBaseline` 不変」）であり、本体 `fandhe_ai_backend_cpu::compare`／
      `assert_parity` への適用は #1241 の承認範囲外の別個の拡大提案になる
  (b) 現在保有する実測データ（§1.1 の `parity_survey_all_shapes.log`）は形状単位の集計値
      （`fail_count`／`mean_abs_diff`／`max_abs_diff`／`max_rel_err`）のみであり、要素単位の
      `abs_diff`／真値ダンプを持たないため、A′ 救済項が実際に何要素救済するかを机上計算できない
      （candle 側の #1237／#1238 相当の要素ダンプ計装が本イシューの範囲外として必要）
  (c) 適用対象が本体判定式全体（全バックエンド共通）になるため、Metal split-K 単体より影響
      範囲が大きく、影響範囲を Metal split-K テスト・baseline に限定できる候補 2 より変更の
      ブラスト半径が大きい

## 3. 決定（未承認のまま記録）

**推奨は候補 2（実測ベースライン非後退方式）である。ただし「承認」自体は本ドキュメントでは
確定しない**（下記「7. 承認記録」を参照。自動実装フローでは対話的なユーザー承認を得られない
ため、本ドキュメントは draft のまま完成させ、実際の承認は本 issue または対応する PR への人間の
コメントを一次記録とする）。

対象 11 形状すべてに一律で baseline 非後退方式を適用する設計（`docs/perf/metal-gemm-splitk-
two-pass.md` §5.3・`splitk_parity_baseline.rs` の既存実装のとおり）とし、CUDA 先例
（`docs/cuda-tensor-core-parity-judgment-decision.md` §2）のような「ゼロ fail 成立形状は
厳密判定・不成立形状のみ baseline」という**形状二分方式は採用しない**。理由:

- §1.2 の実測が示すとおり split-K の誤差は形状だけでなく入力データ（近ゼロ要素の有無）にも
  依存する。本番の自動判定入口（`dispatch_split_k_strided_prepared`）にとって「この形状は
  過去のテストで厳密ゼロ fail だった」という事実は、任意の入力に対する安全性を保証しない
- 一方、本テストハーネス自体は形状ごとに決定的なシード（`Xorshift64Star::new(m as u64 * 7 +
  k as u64 + 1)` 等。`crates/backend-metal/tests/gemm_splitk_parity.rs`）を使うため、テストの
  非後退契約としては baseline 方式を全形状へ一律適用しても実装は単純で一貫性がある（CUDA 側の
  ように「一部は厳密判定・一部は baseline」という条件分岐を持ち込む必要がない）

## 4. 回帰検出契約

`docs/cuda-tensor-core-parity-judgment-decision.md` §3 と同型の契約を Metal split-K へ適用する
（`assert_no_split_k_parity_regression` 実装。`crates/backend-metal/tests/common/
splitk_parity_baseline.rs`）:

- **fail_count の増加で必ず fail する**: 記録済みベースラインより 1 要素でも不合格要素数が
  増えれば panic する
- **mean_abs_diff ceiling の超過でも必ず fail する**: fail_count が変わらなくても誤差の平均値
  が悪化する回帰を検出する
- **max_abs_diff・max_rel_err ceiling の超過でも必ず fail する**: fail_count 同数のまま個別
  要素の誤差が悪化する回帰（max-regression ブラインドスポット）を検出する
- **total（形状・要素数）の不一致でも必ず fail する**: 記録時と異なる形状で計測してしまう
  取り違えを防ぐ
- **baseline の更新自体に承認が要る**: fail_count・各 ceiling の上限を緩める「上方更新」は
  ユーザー承認必須（`.claude/rules/security.md` A08 と同列のガードレール）。推定値の記入は
  禁止で、更新は実機実測値とセットでのみ行う

**`(128,128,2064)` の `max_rel_err` ceiling（0.14。§1.1 実測値 1.31e-1 に表記丸め天井を加えた
値。実際の `splitk_parity_baseline.rs` 記録値は §1.1 の実測どおり）が実効的な保護になっていない
点を明示する**: 真値が 0 近傍の要素で相対誤差が桁落ちにより外れ値化する現象（§1.1 の説明）で
あり、この行の回帰検出は実質的に `fail_count`（2 件固定）と `max_abs_diff`（1.22e-4。他行と
同水準）が担う。`max_rel_err` ceiling 自体は緩いが、他 10 行では引き続き有効な検査として機能
するため、契約全体としては維持される。

以上により、非後退方式は「既知の不合格分布を許容する」という点で厳密ゼロ fail 判定より緩いが、
「その既知の分布から悪化していないか」を fail-closed に検査する点で回帰検出契約としては維持
される。

## 5. スコープ限定

- **tolerance 定数の変更は本決定の対象外**: `RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`
  （`crates/backend-cpu/src/parity.rs`）自体の変更は、本決定が扱う「判定方式（テスト個別の
  合否基準）の使い分け」とは別軸であり、引き続きユーザー承認必須
  （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）
- **`dispatch_split_k_strided_prepared`（自動判定入口）の `SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  ゲート解除は別イシュー（#1513）**: 本ドキュメントは判定方式の是非を検討する draft に閉じ、
  フラグの値は変更しない
- **`crates/backend-metal/tests/gemm_splitk_parity.rs` の判定方式切替は別イシュー（#1512）**:
  本ドキュメントの承認（未承認のまま）を踏まえてテストを baseline 方式へ切り替える作業は含まない
- **本ドキュメント自体は `crates/`・`scripts/`・`.github/`・`docs/spec/` を変更しない**: 決定
  記録の draft 作成に閉じる

## 6. spec 起票要否

CUDA 先例（`docs/cuda-tensor-core-parity-judgment-decision.md`）は spec REQ-2 へ「2026-09-02
追記・Tensor Core 経路の受け入れ判定方式」として正式追記済み（Fandhe-AI/fandhe-ai-spec PR #63）
だが、この追記は文言上 TF32/f16 **Tensor Core 経路**限定である。Metal f32 split-K は Tensor Core
経路ではない（f32 SIMT・`simdgroup_multiply_accumulate` による classic カーネルの派生）ため、
CUDA 先例の追記文言をそのまま適用対象化するのではなく、**「実測ベースライン非後退方式」という
判定パターン自体を「単一の連続 K/縮約ループとは異なる結合順序で部分和を求めるカーネル全般」
（Tensor Core 経路に限らない）へ一般化する spec 改定が必要か**を論点として明示する。

`docs/spec-proposal-req2-candle-parity-tolerance.md` の書式（起票用タイトル案・本文案テンプレー
ト・「そのまま貼り付け可能な形式」）に倣い、**承認された場合に使う起票用本文の draft** を
以下に用意する（未起票のまま。実起票は承認後の別イシューで行う）:

---

**起票用タイトル案**:

```
docs(requirements): REQ-2 の実測ベースライン非後退方式を Tensor Core 経路限定から
「結合順序が単一連続ループと異なるカーネル全般」へ一般化する（実装リポ Fandhe-AI/fandhe-ai#1509
提案）
```

**起票用本文案**（draft。承認後に実測データ・承認記録へのリンクを差し替えて起票する）:

````markdown
## 背景

REQ-2「2026-09-02 追記・Tensor Core 経路の受け入れ判定方式」は、CUDA TF32/f16 Tensor Core
経路（`wmma_tf32`・`wmma_tf32_opt`・`wmma_tf32_staged` 等）が「厳密ゼロ fail 判定は実機実測で
成立が確認された形状に限り、成立しない形状は実測ベースライン非後退方式を正式な受け入れ判定と
する」という判定方式で運用されることを定めている。

実装リポ側で、Metal f32 split-K GEMM 経路（K 方向を複数パーティションへ分割し独立に部分和を
求めたうえで固定順序で縮約する構造）についても、同種の「単一の連続ループとは異なる結合順序に
由来する丸め誤差の恒常特性」が実機実測（Apple M4 Max。対象 11 形状中 8 形状が厳密ゼロ fail
不成立）で確認された。この経路は Tensor Core を用いない f32 SIMT カーネルであるため、現行
REQ-2 追記の文言（Tensor Core 経路限定）はそのままでは適用対象に含まれない。

## 提案

REQ-2 の当該追記を、対象を「TF32/f16 Tensor Core 経路」から「単一の連続 K ループ（または
縮約ループ）とは異なる結合順序で部分和を求める構造を持つカーネル全般（Tensor Core 経路を
含むがそれに限らない）」へ一般化することを提案する。判定方式自体（形状ごとの厳密ゼロ fail
成立確認 → 不成立形状は実測ベースライン非後退方式）は変更しない。

## 影響範囲

- 本提案は判定**方式**の適用対象を広げるものであり、`RELATIVE_TOLERANCE`／
  `ABSOLUTE_RESCUE_THRESHOLD` 等の tolerance 定数自体は変更しない
- 実装リポ側の適用（Metal f32 split-K への具体的な baseline 登録）は、実装リポ側の別途
  ユーザー承認（`docs/backend-metal-splitk-parity-judgment-decision.md`）を経て行う
````

---

## 7. 承認記録

**未承認**（2026-09-10 時点。draft 作成完了・自動実装フローのためユーザー承認待ち。
本 issue または本 PR への人間のコメントが承認の一次記録となる）。

## 8. 関連

- イシュー #1474（split-K 2 パス実装・§5 判定方式の実測・差し戻し）・#1511（本決定記録・本
  イシュー）・親 #1510・ルート #1509
- `docs/perf/metal-gemm-splitk-two-pass.md`（§5. AC-2 判定方式の実測・差し戻しの経緯本体）
- `docs/backend-metal-splitk-decision.md`（split-K 採否判断・§4 本番結線可否〈結線しない〉・
  承認依頼の要点 (a)(b)(c)）
- `docs/cuda-tensor-core-parity-judgment-decision.md`（先例。判定方式決定記録の構成・spec 反映
  フローの雛形）
- `docs/candle-parity-tolerance-contract-decision.md`（A-1 救済項の出典・「ハーネス限定」スコープ
  限定の先例。候補 3 不採用理由 (a) の根拠）
- `docs/spec-proposal-req2-candle-parity-tolerance.md`（spec 起票用本文の書式テンプレート）
- `crates/backend-metal/tests/common/splitk_parity_baseline.rs`（`BASELINES`・
  `assert_no_split_k_parity_regression` 実装本体。現状未使用のまま維持）
- `crates/backend-metal/tests/gemm_splitk_parity.rs`（現状: 厳密ゼロ fail 判定。切替は #1512）
- `.claude/rules/security.md`「自己修復ループ固有のガードレール」・`.claude/rules/coding-rust.md`
  「テスト・ベンチ」節（本決定記録が draft のまま未承認とすべき根拠）
