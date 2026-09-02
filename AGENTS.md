# AGENTS.md

## 文書の位置づけ

本リポジトリで作業するすべての AI エージェント・開発者、および Codex による PR 自動
レビュー（`.github/workflows/codex-review.yml` wrapper、イシュー #326）が共通で用いる
レビュー観点集。Codex のカスタム prompt（`.github/codex/prompts/review.md`、イシュー
#376）は PR の base コミットの本ファイルを読み、prompt に埋め込まれた P0/P1 基準に
**加えて**適用する（優先度定義が矛盾する場合は本ファイルを優先する契約。矛盾を作らない
ため、本ファイルの優先度定義は prompt と同一とする）。

二重管理を避けるための役割分担:

- **`.github/codex/prompts/review.md`**: レビュー手順・完了判定（`review_completed`）・
  プロンプトインジェクション耐性・P0/P1 の禁止事項列挙（機械的 enforcement の正）
- **本ファイル**: セキュリティ / アーキテクチャ整合 / 再利用・アセット化の 3 観点と
  リポジトリ固有観点の**観点整理の正**。個別基準の詳細は一次情報源
  （`CLAUDE.md`・`.claude/rules/`・`docs/`・`docs/spec/`）を正とし、本書では要点と
  参照のみを記載する。内容が食い違う場合は一次情報源を正とする

## 優先度の定義

`.github/codex/prompts/review.md` と同一。

| 優先度 | 意味 | CI ゲート |
|--------|------|-----------|
| P0 | マージ不可。脆弱性・データ破壊・ガードレール迂回・契約破壊に直結 | ジョブ失敗 |
| P1 | 修正必須。基盤方針・依存規約・CI 規約・運用規約への違反 | ジョブ失敗 |
| P2 | 修正推奨。可読性・保守性・テスト網羅の改善 | 通過（コメントのみ） |
| P3 | 任意。好みの範囲の提案 | 通過（コメントのみ） |

## 1. セキュリティ観点

`.claude/rules/security.md` / `.claude/rules/deps-policy.md` を正とする。

- **シークレットの混入（P0）**: API キー・トークン・パスワード・秘密鍵・`.env` を
  コード・ログ・hooks・CI 設定・コミットメッセージへ含めない
- **依存監査（P0/P1）**: 依存禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・
  `ndarray`。直接・推移を問わない）の混入は P0。許容依存 9 区分以外の追加、
  `=x.y.z` 完全固定でないバージョン指定、`docs/license-matrix.md` 更新・ユーザー承認
  記録を伴わない依存追加・更新は P1（A06。`deny.toml` の licenses / sources /
  advisories / bans 検査と `scripts/check-forbidden-deps.sh` が機械検査する）。
  第 9 区分（ベンチ比較対象。`matrixmultiply`・`gemm`、および適用範囲拡張の
  `candle-core`・`burn`〈推移的依存ツリー込み〉）は
  `scripts/bench/oss-gemm-compare/`・`scripts/bench/framework-compare/`
  （いずれも独立 Cargo プロジェクト／workspace。本体 workspace 外）
  限定であり、同区分の必須条件（`=x.y.z` 完全固定・本体 workspace〈ルート
  `Cargo.toml`／`Cargo.lock`〉への非混入・各ディレクトリ専用 `deny.toml` による
  CI 監査〈advisories / bans / licenses / sources〉。
  `.claude/rules/deps-policy.md`「許容依存 9 区分」表を参照）の違反は
  従来どおり P1（oss-gemm-compare: 2026-08-20 ユーザー承認・イシュー #755。
  framework-compare: 2026-08-28 ユーザー承認・PR #915。承認記録は
  `docs/framework-compare-harness-decision.md`）。framework-compare 配下の
  禁止リストクレートは、承認済み比較対象（burn 0.21.0・candle-core 0.11.0）とその
  推移的依存ツリーとしての意図的保持に限り P0 の混入には当たらない。同 workspace は
  禁止リスト grep の代わりに `scripts/check-forbidden-deps.sh lock-all` の専用
  fail-closed 契約検査（`[workspace]` 隔離・承認済みピンのドリフト検出・直接依存 allowlist）で統制され、
  この検査・専用 deny.toml・適用範囲を緩める変更は P0
- **外部フォーマットのパース検証（P0）**: safetensors / ONNX（prost）・TOML 設定・
  guardrail CLI 入力のパース時は長さ・形状の事前検証を行う。検証の欠落・後退、
  シェル呼び出しへの外部入力の非クォート展開等のインジェクション経路は P0（A03）
- **ガードレールの完全性（P0/P1）**: 自己修復ループが AI 生成変更を取り込む際の
  ガードレール 3 分岐判定を迂回する経路の追加は P0（A08）。ガードレール閾値
  （`guardrail.toml`）・ポリシー除外リスト（`policy-exclusion.toml`）・バックエンド間
  数値一致テストの許容誤差（tolerance）を人間承認の記録なしに緩和・変更する差分は P1
- **`unsafe` の統制（P0/P1）**: `// SAFETY:` コメントのない `unsafe`、不変条件の根拠が
  不十分な `unsafe` は P0。`unsafe` の使用域は FFI 境界（cudarc・objc2 系）・CPU SIMD
  intrinsics 等の必要最小限に限り、正当化のない拡大は P1
- **fail-closed の維持（P0）**: fail-closed で設計された既存分岐（ガードレール判定・
  CI ゲート・検査スクリプトの self-test）の fail-open 化は P0

## 2. アーキテクチャ・設計整合の観点

`.claude/rules/coding-rust.md` と `docs/` 配下の設計文書を正とする。

- **完全自作コア方針の維持（P0）**: テンソル・autodiff・演算グラフ／カーネル融合・
  計算カーネル・バックエンド抽象層は完全自作とする（REQ-1 v2、変更禁止）。既存 ML
  フレームワークへの統合・方針放棄は P0
- **クレート境界の維持（P1）**: 9 クレート構成（`tensor-core`・`autodiff`・
  `backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・
  `self-repair`・`bench-harness`）の責務境界を維持する。互換 API 層（compat）は
  自作コアの上の薄いラッパーに徹する（REQ-9、`docs/public-api-design.md`）
- **cfg ベースバックエンド切替（P1）**: バックエンド切替は feature フラグなしの
  cfg ベース（PoC-v2-5 実証構成、`docs/backend-switching-design.md`）。`cudarc` は
  無条件依存＋動的ロード、objc2 系は `cfg(target_os = "macos")` 分離。feature
  フラグによる切替の持ち込みは P1
- **数値契約の統一（P1）**: バックエンド間数値一致は統一複合判定（相対誤差 1e-3 未満
  または絶対誤差 1e-5 未満、全ペア共通）。丸め方針（FMA 契約）はバックエンド間で
  統一する（CPU 参照実装は `f32::mul_add`）。matmul 系の FMA 契約はこのまま不変。
  加えて、**正規化統計・勾配の長軸縮約**（rmsnorm の `rstd` 二乗和・dw の行方向蓄積等）
  は `f64` アキュムレータで統一する。この一般原則は 2 系統で扱いが異なる（精密化。
  イシュー #1102・codex-review 指摘・PR #1120）: **正規化統計の二乗和**は要素を
  先に `f64` へ昇格してから二乗する（`f32` のまま二乗すると有限入力でも overflow
  しうるため）。**勾配の長軸縮約の要素積**は overflow リスクが小さいため要素積を
  `f32` で確定してから `f64` へ昇格して蓄積する。Metal は `double` 非対応のため
  Neumaier 改良版 Kahan 補償和 + scale/ssq 方式〈LAPACK SLASSQ 系の overflow-safe
  な二乗和アルゴリズム。単純な Kahan 補償和のみでは有限入力でも二乗が `f32`
  overflow して `NaN` を生むため必須。`NaN`／`inf` 入力の伝播も明示的に扱う〉を
  正規化統計の「`f64` 相当」実装形として適用する（イシュー #1102。ユーザー承認
  2026-09-01。GB10 実機実測・Metal 実機実測: `docs/perf/cuda-parity-baseline.md`
  §9.8〜§9.10）。契約の片側変更（一部バックエンドのみ精度を上げる等）は P1
- **TF32/f16 Tensor Core 経路の parity テスト判定方式（P1。テストの弱体化禁止の
  例外を明記する規約。正本仕様 `docs/spec/04-requirements.md` REQ-2「2026-09-02
  追記・Tensor Core 経路の受け入れ判定方式」〈fandhe-ai-spec PR #63〉が正式な
  合格条件として規定する形状別判定方式の転記であり、受け入れ基準の正は spec 側）**: `wmma_tf32`／`wmma_tf32_opt`／`wmma_tf32_staged` 等の
  Tensor Core 経路は、DGX Spark GB10（sm_121）実機実測で厳密ゼロ fail（複合判定
  `fandhe_ai_backend_cpu::assert_parity`。全要素が相対誤差 1e-3 未満または絶対誤差
  1e-5 未満）が成立しない形状が存在することが判明している（TF32 丸め〈f32 仮数部
  23bit→10bit〉に由来する K 方向蓄積誤差の既知の恒常特性。opt/basic/staged
  カーネル間で bit-identical、sm_86/GB10 世代間でも差分なし。
  `docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`・
  `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`〈#995〉）。この既知
  特性を持つ形状に厳密ゼロ fail 判定を適用すること自体がテスト設計の不整合で
  あり、**カーネル側の数値バグとは区別する**。よって:
  - **厳密ゼロ fail 判定**（`assert_parity`）は、GB10 実機実測でゼロ fail の成立が
    確認された形状にのみ適用する
  - **実測でゼロ fail が成立しないと判明した形状**は、実測 baseline 非後退方式
    （`fail_count` が記録値以下・総要素数が記録値と完全一致・`mean_abs_diff`／
    `max_abs_diff`／`max_rel_err` が記録 ceiling 以下であることを fail-closed に
    機械検査する。max 系 ceiling は fail_count 同数のまま個別要素の誤差が大幅
    悪化し平均で相殺される回帰を検出するための必須項目〈spec 同追記の規定〉。
    `crates/backend-cuda/tests/common/parity_baseline.rs::
    ParityBaseline`／`assert_no_parity_regression`）を正式な受け入れ判定とする。
    baseline の新規追加・更新は必ず GB10 実機実測値を伴う（推定値の記入は禁止。
    `docs/perf/cuda-parity-baseline.md` 「ベースライン更新規約」）
  - **tolerance 定数**（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`。
    `crates/backend-cpu/src/parity.rs`）**の変更は本規約の対象外であり、引き続き
    ユーザー承認必須**（本規約は判定方式〈テスト個別の合否基準〉の使い分けを
    定めるものであり、統一複合判定の閾値自体を変更するものではない）
  - PR #1115（イシュー #1106）で確定した「本体 `assert_parity` は green 必須の
    まま維持し、既知の不合格分布に対する非後退監視を別テストとして併設する」
    設計（`mma_f16_k4096_stress`／`tensor_core_parity_record`〈tf32 部分〉／
    `gemm_tf32_optin_on_matches_cpu_across_shapes`・および同型で追加した
    `wmma_f16_k4096_stress`／`wmma_f16_opt_k4096_stress` の計 5 テスト）は
    **本規約が覆すものではなく、現状維持**とする。本規約が定める「対象形状を
    縮小し baseline 非後退方式へ移行する」方式（案 A。イシュー #1106・PR #1124）
    は、上記 5 テストとは別の対応方式として並立する（どちらを適用するかは
    テストの記録元・ヒストリーに応じて個別に判断してよく、両方式の間で優劣を
    定めない）
  - 承認記録: 2026-09-02 ユーザー承認（イシュー #1106 のスコープ分割コメント。
    #995 の GB10 世代差記録が判断根拠）。spec 側の正式化は fandhe-ai-spec PR #63
    （2026-09-02 マージ済み）・実装リポ PR #1128（`docs/spec` submodule 前進）で
    完了しており、本節はその転記（経緯の一次記録は
    `docs/cuda-tensor-core-parity-judgment-decision.md`）
- **設計文書との整合（P1/P2）**: 方式決定済みの領域（`docs/backend-metal-wgpu-decision.md`・
  `docs/dispatch-rules-design.md`・`docs/typed-shape-design.md`・
  `docs/memory-pool-design.md` 等）と矛盾する実装は、設計文書の改訂とセットでない限り
  P1。新規の設計判断は docs へ記録する（欠落は P2）
- **spec 正本の不可侵（P1）**: `docs/spec/`（fandhe-ai-spec submodule）実体の
  書き換えは禁止。仕様変更は spec リポジトリ側で行う（submodule ポインタの前進自体は
  通常の更新）

## 3. 再利用・アセット化の観点

本ライブラリは汎用 AI/ML ライブラリとして外部利用・他リポジトリからの参照を想定する
資産であり、次の観点で評価する。

- **公開 API 設計（P1/P2）**: 公開 API は `docs/public-api-design.md`（REQ-9）に従い、
  PyTorch からの移行容易性（`docs/pytorch-migration-checklist.md`）を保つ。破壊的
  変更・内部表現の公開 API への漏出は P1。公開 API の doc comment 欠落は P2
- **ハードコード回避（P1）**: 実機固有値（デバイス名・パス・ノード名）・閾値・
  許容誤差のロジックへの直書きを避け、設定（`guardrail.toml` 等）・定数モジュール
  （`bench-harness::threshold` 等）へ集約する。閾値の分散定義（単一真実源の破壊）は P1
- **汎用化可能な実装の分離（P2）**: バックエンド非依存のロジックは `tensor-core` /
  抽象層側に置き、特定バックエンド固有の前提を共通層へ持ち込まない。guardrail /
  self-repair 等、他リポジトリでも転用し得る機構は gateway 固有型への依存を作らず
  CLI・ライブラリ両面で使える構造を保つ
- **ドキュメント整備（P2）**: 新機能・新設定・依存変更は README・CLAUDE.md・該当
  設計文書・`docs/license-matrix.md` への追随とセットで行う（ライセンス表の追随漏れは
  P1。deps-policy.md）

## 4. リポジトリ固有の観点

- **性能予算（P0/P1）**: 対 PyTorch 性能下限は `docs/performance-targets.md`（REQ-8、
  正は `crates/bench-harness/src/threshold.rs::floor_spec`）に従う。下限値・許容誤差の
  人間承認なき変更は P1。**性能・最適化を理由にしたカーネル手動境界チェックの省略は
  P0**（CPU intrinsics・CUDA NVRTC/mma・Metal simdgroup の全カーネルに適用）。ベンチは
  5 回計測の中央値を採用する
- **テストの弱体化禁止（P1）**: 受け入れ基準対応テストの削除、`#[ignore]` 追加による
  ごまかし、実機非依存テストの実機依存化は P1。実機（DGX Spark GB10・Metal 実機）
  依存テストは `#[ignore]` で分離し CI 実行可能テストと区別する。学習系回帰テストは
  決定的シード設定ユーティリティを使う（flaky 化の防止）
- **本番経路の panic 禁止（P1）**: テスト・examples を除くライブラリ・CLI コードでの
  `.unwrap()` / `.expect()`、panic を境界外へ漏らす経路は P1。エラーは型付きエラーと
  する
- **CI 規約（P1）**: GitHub ホステッド（`ubuntu-latest`）既定（public 区分。例外は
  codex-review の codex 実行ジョブのみ）・self-hosted への逆戻りは P1（#457 Phase 1〜3
  完了・移行済み。`runs-on: self-hosted` の出現は `runner-policy` 契約検査〈#472〉でも
  fail する）・larger runner（有料の大型ホステッドランナー）の使用・
  `timeout-minutes` 必須（reusable workflow 呼び出しジョブを除く）・action /
  reusable workflow の SHA 固定・`permissions` 最小化・`ci-complete` の fail-closed
  集約判定の維持（`.claude/rules/ci.md`）。`pull_request_target` 等の fork PR へ
  secrets を露出するトリガー追加は P0（public 化により fork PR が現実化するため
  重要度が上がる）
- **コミット・承認フロー（P2）**: 日本語 Conventional Commits・`--no-verify` 禁止
  （回避は P1）。依存の追加・更新、ガードレール閾値・テスト許容誤差の変更は
  ユーザー承認必須

## Fandhe-AI/actions の参照方式（@latest）

2026-08-18・オーナー判断。`Fandhe-AI/actions`（組織内の上流リポジトリ）への `uses:` 参照は
可変タグ `@latest` を使う。本書の「SHA 固定」に関する観点は**第三者 action にのみ**適用し、
`Fandhe-AI/actions` への `@latest` 参照・SHA pin の除去を指摘しない。`latest` は上流の
`.github/workflows/move-latest-tag.yml` が main への push ごとに付け替える。
