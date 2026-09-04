# CLAUDE.md

## Overview

Rust 製 AI/ML ライブラリの実装リポジトリ（v2）。Burn 依存を排した**完全自作コア**（テンソル・autodiff・演算グラフ／カーネル融合機構・計算カーネル・バックエンド抽象層）で実装する。仕様の正本は [Fandhe-AI/fandhe-ai-spec](https://github.com/Fandhe-AI/fandhe-ai-spec)（`docs/spec` submodule）にあり、本リポでは編集しない。本リポジトリ自体は **public**（#457 Phase 1〜3 完了）で、CI は GitHub ホステッド `ubuntu-latest` 既定へ移行済み（self-hosted への逆戻りは `runner-policy` ジョブ〈#472〉が fail-closed で検知。詳細 → `.claude/rules/ci.md`）。仕様 submodule（`docs/spec`）と旧実装（v1）は private を維持する（README「位置づけ」節）。

- 想定クレート 10 個: `tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`・`facade`（TASK-9.3・イシュー #410 で新設した composition root に、TASK-9.4・イシュー #411 で `autodiff::compat` から compat 公開面〈`compat::array`・`compat::Sequential`〉を移設済み。`facade` が唯一のサポートされる公開 API 面であり `tensor-core`・`autodiff`・`backend-*` は内部クレート。`docs/compat-api-scope.md` §0）に加え、GitHub Pages 公開ツリー（#865 Phase 1）向けの開発者・CI 専用 SSG クレート `docs-site`（11 個目・`publish = false`。イシュー #868/#869。本体ライブラリの公開 API とは無関係で外部依存ゼロ）。上記の名称はディレクトリ名（`crates/<name>`）であり変更しない。crates.io 公開対象 6 クレート（`facade`・`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`）は `[package] name` を `fandhe-ai` prefix 付き公開名（`fandhe-ai`・`fandhe-ai-tensor-core`・`fandhe-ai-autodiff`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal`）へ rename 済み（イシュー #877/#879・#878 でユーザー承認済み。`docs/crates-io-naming-decision.md`）。`onnx-interop`・`guardrail`・`self-repair`・`bench-harness`・`docs-site` は非公開のため対象外
- crates.io への公開は一括リリース `.github/workflows/release-all.yml`（workflow_dispatch 1 回・environment `crates-io-release` 承認 1 回で公開 6 クレートを依存順に publish）を基本とし、単一クレートの再実行・障害復旧には `.github/workflows/release.yml` を使う（いずれも `CARGO_REGISTRY_TOKEN`〈org secret〉・fail-closed ガード群。手順・版数運用の正は `docs/crates-io-publishing-order.md`）。初回公開（v0.3.0・6 クレート）は 2026-08-23 に完了済み（crates.io・docs.rs 反映確認済み。同 doc §10 追補）。v0.4.0 は 2026-08-29 に `release-all.yml` で公開済み（同 doc §10 追補）。v0.5.0 は 2026-08-31 に同ワークフローで公開済み（同 doc §10 追補。framework-compare の承認ピンをイシュー #1011 で `=0.5.0` へ更新済み）。v0.6.0 は 2026-09-02 に同ワークフロー（run 33503500987）で公開済み（同 doc §10 追補。framework-compare の承認ピンを `=0.6.0` へ更新済み）
- 依存は許容 8 区分のみ・`=x.y.z` 完全固定（`.claude/rules/deps-policy.md`）。禁止リスト（`burn` 系・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査
- バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成）
- 現状 M0 完了（TASK-1 通し完了: workspace Cargo.toml・9 クレート雛形・本体 workspace 直接依存の許容 8 区分〈ベンチ比較対象の第 9 区分は `scripts/bench/oss-gemm-compare/` 限定。計 9 区分の正は `.claude/rules/deps-policy.md`〉の =x.y.z 完全固定・依存禁止検査・deny.toml・license-matrix.md）。crates.io 初回公開 v0.3.0（2026-08-23）・GitHub Pages 公開済み（`crates/docs-site` + `site/` + `.github/workflows/docs-site.yml`。ヘッダーのホバーメニュー・セクション連動サイドバーは #909）。CI・Makefile の cargo 系チェック（fmt / clippy / test / deny / deps-forbidden）は全て有効化済み

## Repository Structure

```
fandhe-ai/
├── CLAUDE.md                # 本ファイル
├── README.md                # 開発環境構築・実装方針の要点
├── LICENSE-APACHE           # Apache License 2.0 全文（MIT/Apache-2.0 デュアルライセンス。#462）
├── LICENSE-MIT              # MIT ライセンス本文（同上）
├── Makefile                 # make setup / ci / docker-* タスクランナー
├── lefthook.yml             # git hooks（rustfmt-check・secrets-guard・commit-msg・pre-push）
├── .editorconfig            # インデント・改行規約
├── Dockerfile / compose.yaml # 環境非依存の開発コンテナ（CPU バックエンドのみ）
├── skills-lock.json         # 導入スキルのハッシュ管理（npx skills）
├── Cargo.toml                # workspace 定義（本体 10 クレート + docs-site〈開発ツール〉・許容依存 8 区分を =x.y.z 固定）
├── Cargo.lock                # 依存解決の完全固定（deps-policy.md）
├── rust-toolchain.toml       # toolchain 単一真実源（stable + rustfmt/clippy。rust-base-ci 前提。#325）
├── deny.toml                 # cargo-deny 設定（licenses 許可リスト・sources = crates.io 限定〈TASK-1.3〉+ advisories / bans〈#353〉）
├── guardrail.toml             # guardrail 判定閾値の確定設定（TASK-4.3c・#117。default プリセット）
├── crates/                  # tensor-core・autodiff・backend-cpu・backend-cuda・backend-metal・
│                             # onnx-interop・guardrail・self-repair・bench-harness・facade（composition root・compat 公開面）・
│                             # docs-site（GitHub Pages 公開ツリー向け SSG。開発者・CI 専用。#868/#869）
├── scripts/
│   ├── check-forbidden-deps.sh # 依存禁止リストの検査ロジック（ci.yml・Makefile 共用。TASK-1.2）
│   ├── check-workflow-runner-policy.sh # self-hosted runner 逆戻り防止の fail-closed 契約検査の呼び出し面（ci.yml・Makefile 共用。#472）
│   ├── check-workflow-runner-policy.py # 同検査の本体（python3 標準ライブラリのみの自前 YAML サブセットパーサー方式。追加依存なしで表記トリック迂回を遮断。#472・PR #626）
│   ├── run-verification-gates.sh # AI 自律メンテナンス検証 4 ゲート（build/test/clippy/bench）の実行ロジック（ci.yml・Makefile 共用。TASK-6.1c）
│   ├── run-guardrail-regression.sh # guardrail 2 層検証ロジック（ci.yml・schedule 共用。TASK-6.1a）
│   ├── report-guardrail-schedule-result.sh # schedule 定期実行失敗時の Issue 起票・復旧クローズ（TASK-6.1b）
│   ├── report-clippy-nocache-schedule-result.sh # キャッシュなしフルビルド clippy 定期検証の失敗時 Issue 起票・復旧クローズ（イシュー #918）
│   ├── testdata/             # 上記の self-test 用固定 fixture
│   └── bench/
│       ├── oss-gemm-compare/ # CPU GEMM OSS 直接比較ハーネス（本体 workspace 外の独立 Cargo パッケージ。matrixmultiply・gemm crate。イシュー #755）
│       ├── gemm_bench_torch_mps_f16.py／gemm_bench_torch_mps_f32.py # PyTorch MPS 参照計測
│       ├── gemm_bench_torch_cpu_f32.py # PyTorch CPU f32 GEMM 参照計測（イシュー #1141）
│       └── gemm_bench_mlx_f32.py # MLX f32 GEMM 計測（イシュー #755）
├── .github/workflows/
│   ├── ci.yml               # rust-ci（Fandhe-AI/actions rust-base-ci 呼び出し: fmt / clippy / test / deny。#325）+ 固有ジョブ（build / build-no-cuda-toolkit / deps-forbidden / runner-policy / guardrail-regression / verification-gates）+ ci-complete
│   ├── codex-review.yml     # Codex PR 自動レビュー wrapper（Fandhe-AI/actions codex-review を SHA 固定呼び出し。#326。public 構成〈post-feedback-runner-label: ubuntu-latest〉へ切替済み。#469）
│   ├── verification-gate-bench.yml # bench ゲート（schedule／workflow_dispatch。TASK-6.1c）
│   ├── guardrail-regression-schedule.yml # guardrail 2 層検証の schedule 定期実行・失敗時 Issue 可視化（TASK-6.1b）
│   ├── clippy-nocache-schedule.yml # キャッシュなしフルビルド clippy の定期検証・失敗時 Issue 可視化（イシュー #918）
│   ├── release.yml          # crates.io publish（workflow_dispatch + `CARGO_REGISTRY_TOKEN`・environment `crates-io-release` 承認ゲート。#884。手順は `docs/crates-io-publishing-order.md` §9〜11・`.claude/rules/ci.md` release.yml 節）
│   ├── update-external.yml  # docs/spec・.claude/skills の自動追従
│   └── docs-site.yml        # GitHub Pages ビルド・デプロイ（Fandhe-AI/actions `pages-deploy.yml` 呼び出し）
├── site/                    # GitHub Pages 公開原稿（`nav.toml` + Markdown。#873/#874/#875。`docs/spec` の内容は含めない）
├── .claude/
│   ├── agents/              # research / implement / testing / quality / docs
│   ├── rules/               # 委譲・コーディング・依存・CI・セキュリティ等の規約
│   ├── skills/              # npx skills add で導入（skills-lock.json 管理）
│   ├── workflows/           # implement-issue-tree.js（skills への相対 symlink）
│   └── settings.json        # SessionStart / PostToolUse hooks
└── docs/
    ├── autodiff-view-recompute-decision.md # view 系ノード（reshape / transpose）の再計算方式化の設計（push_view／resolve_view による中間バッファ非確保・融合境界化・実測記録。#1043 ツリー・#1047）
    ├── backend-cuda-async-execution-design.md # CUDA 非同期実行モデルの同期契約（ストリーム順序・エラー伝播・D2H 境界・poison/invalidate 状態機械。#1011 ツリー・#1012）
    ├── backend-cuda-pool-allocator-decision.md # CUDA サイズクラス別プールアロケータ（自作 SizeClassPool<H> 案 B・driver プール〈cuMemPoolTrimTo〉併用）の採用判断・alloc_uninit 適用確認範囲・実測記入欄（#1018 ツリー・#1020）
    ├── backend-cuda-ptx-embedding-decision.md # CUDA GEMM ビルド時 PTX 事前埋め込み（candle 方式）不採用判断・JIT キャッシュ結線範囲の更新（#1024）
    ├── backend-metal-aligned-load-decision.md # Metal GEMM アラインメント特化ロード分岐（align_M/N/K function constant 方式）不採用判断（#752 保留 → #808 格下げ）
    ├── backend-metal-async-copy-decision.md # Metal 非公式 simdgroup_async_copy 系 AIR intrinsic 不採用の決定記録（#546）
    ├── backend-metal-command-batching-design.md # コマンドバッファ・エンコーダ共有と同期境界（waitUntilCompleted）の設計・CUDA 側契約との対応（#1015 ツリー・#1016。§7 に #1017 実装記録追記）
    ├── backend-metal-mlx-classic-nax-decision.md # MLX classic 経路と CANDIDATES の構成対比・NAX 経路不採用判断（#549）
    ├── backend-metal-morton-mapping-decision.md # 標準 simdgroup_matrix API 下での Morton 順レーン→要素マッピング適用不可の判断（#544）
    ├── backend-metal-splitk-decision.md # split-K ディスパッチ分岐の MLX 選択条件対比・採否判断（#810）
    ├── backend-metal-transpose-collapse-design.md # 転置パターン別 strided GEMM 入口（GemmStrides）・先頭次元 collapse の設計・実機実測記入欄（#1029 ツリー・#1040）
    ├── backend-metal-wgpu-decision.md  # Metal バックエンド実装方式（wgpu 非採用）の決定記録
    ├── backend-switching-design.md     # cfg ベースバックエンド切替の設計
    ├── cpu-gemm-b-packing-sharing-decision.md # B パネル packing のスレッド間共有化の設計検討・適用可否判断（#565）
    ├── cpu-gemm-prefetch-decision.md   # aarch64 プリフェッチ intrinsics 到達可能性調査・E-7 保留判断→原則不要へ格下げ（#489・#751）
    ├── crates-io-naming-decision.md # crates.io 公開クレート名（fandhe-ai prefix）の空き確認・最終名ユーザー承認記録（#878/#879）
    ├── crates-io-publishing-order.md # crates.io 公開 6 クレート間 path 依存の version 併記方針（[dependencies] は付与・[dev-dependencies] は strip）・公開順序（トポロジカル順）・workspace.version 一括バンプ運用（#881）
    ├── cuda-streamk-decision.md        # CUDA GEMM StreamK スケジューリングの機構要約・wave 定量化・採否判断（保留。#812）
    ├── cuda-tensor-core-design.md      # TASK-11.1a WMMA/mma カーネル設計メモ（#60）
    ├── cuda-tf32-optin-api-decision.md # CUDA GEMM の TF32 Tensor Core 経路を opt-in で選択する公開 API（`fandhe_ai::set_cuda_tf32_gemm_enabled`）の設計判断・既定 OFF／fail-closed 方針・framework-compare `--tf32` の C-1/C-2 分割（#1042）
    ├── device-memory-pool-design.md # デバイスメモリのサイズクラス・プールアロケータ設計（ハンドル非依存 SizeClassPool<H>〈take／put の RAII 返却・record_allocation／record_loan_end／record_release（Mutex 系）と record_pending_return／record_pending_merge（AtomicU64 の lock-free 系。push／take と同一クリティカルセクションで対にして順序逆転を排除）による PoolStats 更新契約・Metal は synchronize 完了まで put を遅延し同期の成否に関わらず pending_return_bytes で可視化した返却待ちを合流・take_one_for_release とバックエンド別フェーズを持つ release_cached のトランザクション型解放〉・PoolConfig／PoolStats・サイズクラス表・寿命・断片化・スレッド安全・解放戦略・REQ-14 解放 API の facade 到達経路・§8 設計確定事項／実装確定事項の区分。#1018 ツリー・#1019）
    ├── device-resident-update-design.md # 学習ループのパラメータ更新デバイス常駐化の設計（更新経路・所有権・数値一致契約。#933 ツリー・#934）
    ├── facade-device-handle-design.md # デバイスハンドル再利用の公開 API 設計判断（案 B のみ採用・#929/#946 実装済みの追認・#931）
    ├── facade-optimizer-promotion-decision.md # facade optimizer 公開 API 昇格の設計判断（#932）
    ├── git-history-exposure-decision.md # git 履歴残存内部情報・個人メールアドレスの扱い判断・暫定方針（#477）
    ├── guardrail-change-policy.md    # TASK-6.2 判定器変更時フローの明文化（#149）
    ├── guardrail-self-repair-cli.md  # guardrail／self-repair CLI コマンド仕様（#183）
    ├── inference-forward-fixed-cost-design.md # 推論 forward の固定費削減（tape 不要経路・活性化デバイス常駐チェーン）の設計・bit-exactness 契約・実測記録（#1028）
    ├── kernel-fusion.md     # TASK-12.2b カーネル融合の適用範囲・限界（複合WLで融合を性能目標の前提にしない。#168）
    ├── license-matrix.md    # 許容依存 8 区分のライセンス可否表（TASK-1.3）
    ├── matmul-vjp-zero-copy-decision.md # matmul VJP の転置ゼロコピー化（`eval::matmul` の stride 対応）・CPU BLIS／CUDA／Metal gemm 結線を別イシューへ引き継ぐスコープ判断・実測記入欄（#1043 ツリー・#1046）
    ├── oss-comparison-harness-decision.md # OSS 直接比較ハーネス（matrixmultiply・gemm crate・MLX・PyTorch）の恒久化・本体 workspace 外配置の設計判断・matrixmultiply/gemm の許容依存第 9 区分〈ベンチ比較対象〉としての条件付きユーザー承認記録（#755）
    ├── perf/                # 性能実測・下限確定の記録群（`performance-floor-decision.md` ほか。GEMM 最適化ツリー #479 の実測記録を含む）
    │   ├── performance-floor-decision.md # REQ-8 段階的下限の確定判断・追補記録（#158・#386・#393・#577）
    │   ├── gemm-optimization-baseline.md # REQ-8 GEMM 5 行の分母・分子（対象カーネル・実機・PyTorch 版・出典）の突合基準（#481）
    │   ├── oss-gemm-comparison-baseline.md # OSS 直接比較の再現手順・計測境界・ベースライン・再計測キャンペーン表（#755）
    │   ├── cuda-parity-baseline.md # CUDA Tensor Core 経路 parity 非後退契約のベースライン記録（#491。#1158 で f16 MatrixUnit 経路 mma 優先化の GB10 非後退確認 §12 を追記）
    │   ├── sm121-device-attributes.md # sm_121（DGX Spark GB10）デバイス属性・L1/L2 実効帯域の実測記録（#482）
    │   ├── cuda-gemm-bottleneck-diagnosis.md # CUDA GEMM M=N=K=4096 データ再利用崩壊の定量診断（#486）
    │   ├── metal-gemm-bottleneck-diagnosis.md # Metal GEMM 1024 以降スループット頭打ちの定量診断（#487。#744 是正前・context_cache 導入前の前提。再診断は metal-gemm-bottleneck-rediagnosis.md を参照）
    │   ├── metal-gemm-bottleneck-rediagnosis.md # Metal GEMM 1024 以降頭打ちの context_cache 後の再診断（M4 Max 実機実測完了。fandhe-ai 自系列内の転送〈アップロード＋readback〉寄与は確認したが、candle 側転送分離測定〈#1103 追補〉の結果 candle比ギャップの主因とは確定できず・GPU counters は引き続き未計測〈GPU Service が対象デバイス非対応と報告〉・タイル形状は現行 CANDIDATES[3] が 4096 で最良・1024/2048 は [5]/[6] と同等〈相対 5% 未満〉。#1036・#1103）
    │   ├── metal-gemm-splitk-shapes.md # split-K 対象形状（K 支配的非正方）の劣化定量化実測記録（#810）
    │   ├── cuda-fresh-gemm-n2048-overhead-diagnosis.md # fresh モード CUDA GEMM N=2048 固有の約 166〜184 ms 残存オーバーヘッド診断（DGX Spark GB10 実機実測完了・HEAD 時点で非再現を確認・コード修正なし。#956・#1025。#1157 で §11 追記: #1130 ツリー〈#1146/#1149/#1153〉の結論と #956/#1025 非再現の関係。確定機構〈32 MiB 固定上限〉は #956 の 16 MiB を説明せず別原因・同族の可能性は推測として区別・N=4096 異常値は #1130 と整合・対策なし〈環境要因〉）
    │   ├── burn-wgpu-metal-gemm-zero-result.md # framework-compare の Burn(wgpu) Metal GEMM N>=512 全ゼロの原因切り分け（upstream 既知バグ。#965）
    │   ├── cuda-tensor-core-tolerance-opt-remeasurement.md # opt 版 WMMA TF32 カーネルの数値一致誤差分布再実測（GB10 実機計測完了・sm_86 との差分なし。#994・#995）
    │   ├── cuda-tensor-core-tolerance-gb10-scale-sweep.md # GB10（sm_121）実機での入力スケールスイープ再実測・sm_86 との世代差記録（#995）
    │   ├── cpu-gemm-candle-cpu-retune.md # CPU GEMM マイクロカーネル・packing 再チューニング（pc 外側ループ・A 1 回 pack 候補〈SharedBPcOuter〉。対 gemm crate 逆転狙い。GB10・M4 Max とも実機実測完了・いずれも非採用と結論〈#1140・#1141〉。#1041。#1144 で本番結線不要（`RowPanel` 維持）と確定・次候補は承認待ち）
    │   ├── train-linear-epilogue-fusion.md # 学習 forward の Linear+ReLU epilogue 融合（gemm_bias_act／gemm_resident_rhs_act 結線）の起動数 before/after・CPU 実測・Metal/DGX Spark 未実測の明記（#1044）
    │   ├── train-step-phase-breakdown.md # CPU / CUDA / Metal 学習 1 step のフェーズ分解実機実測（M4 Max・DGX Spark GB10・5 回計測）・支配項トップ 3（backward が 83.6〜97.3% で全バックエンド共通の支配項）・#1008 配下 Issue 優先順位の更新案（#1010）
    │   ├── cuda-async-sync-removal-framework-compare-ab.md # CUDA 都度同期廃止（#1011）の framework-compare 実践規模 A/B 計測記録（DGX Spark GB10 実機計測完了。受入根拠は同一プロトコルの fresh 0.928 倍のみ。reuse 0.440 倍〈約 2.3 倍短縮〉は #1059 の resident forward/backward 経路変更との複合効果につき参考値〈codex-review P1 対応〉・checksum 複合判定 ok。#1083）
    │   ├── cuda-tf32-optin-parity.md # CUDA TF32 opt-in GEMM（`crate::precision`）の複合判定・実機実行手順・実測記入欄（本エージェント実行環境に CUDA 実機なしのため未実測明記。#1042）
    │   ├── cuda-wmma-f16-perf-triage.md # WMMA(f16) 性能外れ値（`wmma_f16_opt`≈`wmma_f16_basic` が `mma_sync_f16` を約3〜13倍〈形状依存。dim2048で約7倍・dim4096で約11〜13倍〉恒常的に下回る・tiled f32 カーネル単体も下回る〈GB10実測。#64 f16 assert red〉）の診断・GB10実機実測（2026-09-03）・大形状計測をカーネル単体プロトコルへ切替えたテスト是正記録・到達性整理・切り出し先（#1130・#1131）。§8 で `tensor_core_tflops_record` の f16 計測を本番 f16 経路（`mma_sync_f16`）へ追従させ assert pass 転換・`wmma_f16_opt` はフォールバック限定維持と確定（GB10実機実測2026-09-04・#1160）。`MMA_PRIORITY_PRODUCTION_ENABLED` 本番結線は PR #1179 codex-review 指摘〈K=4096 非後退ゲートの `MmaF16` baseline ceiling 未承認〉により `false` へ差し戻し・承認待ち（`docs/perf/cuda-gemm-auto-f16-mma-switch.md` §0）。#1123
    │   ├── cuda-large-buffer-percall-alloc-transfer-threshold.md # 大容量バッファ per-call アロケーション＋転送（`clone_htod`／`alloc_zeros`／`clone_dtoh`）のP0〜P6フェーズ分解・サイズスイープ（8〜64 MiB・12段階）実機実測（GB10・2026-09-03。PR #1169 codex-review・Bugbot 指摘対応のsynchronize是正〈P0含む全フェーズ〉後にGB10実機で再実測完了）。閾値32 MiB自体（glibc mmapしきい値・H-B/H-C棄却）は事実として確定・再現済みだが、これが#1123の元のper-call D2H症状（dim4096合算約261〜263ms）の主因であることは未確定（dim4096相当規模での直接再現・降順走査限定の確率的スパイクの発生条件特定が必要）・32 MiB通過直後の降順走査でのみ確率的に発生する追加スパイク（二峰性）も記録。32→33 MiBの別段差（デバイス確保・H2D側）も新規に観測・未解明（#1146）
    │   ├── cuda-percall-alloc-pool-threshold-ab.md # cuMemPool release threshold 引き上げ・cuMemAlloc 同期割当への切替の A/B 計測（なし／A／B／A+B の4条件・P0〜P4＋CudaMmaGemm本番経路レプリカP7〈dim4096〉）。#1146 §4.3の32→33 MiB段差・#1130元症状の対策候補を決着させる位置づけ。GB10実機実測完了（イシュー#1153でPhase 0として実行）: 孤立マイクロベンチマーク（P1〜P3）ではrelease threshold引き上げ（案A）が段差比1.00〜1.03倍へ縮小したが、`CudaMmaGemm::run_f16`の現実的レプリカ（P7）では同じ案Aがmedianを約10倍悪化させる正反対の結果となり、案Aの本番結線は不採用と確定（GB10 unified memory環境での物理メモリ競合と推定）。#1149
    │   ├── cuda-gemm-mma-f16-pool-wiring.md # `CudaMmaGemm::run_f16`のper-call確保（`clone_htod`／`alloc_zeros::<f16>`直呼び）のSizeClassPool経由化を試作（#1153）。`pool.rs::CudaSliceHandle`のdtype一般化（`PoolDtype`トレイト・第2プール非新設）・`gemm_mma.rs`のview ベース起動実装集約とpooled API（`pub`・`internal-diagnostics`feature限定）追加。GB10実機の別バイナリ比較・同一バイナリ交互実行の両方でdim4096がプール経由化により明確に悪化（slow-path到達率raw 50%対pooled 17%）することを確認し、`run_f16`への本番結線は見送り（512〜2048は改善）。数値一致回帰は既知failのfail_countまで含めbefore/after完全一致
    │   ├── cuda-gemm-tiled-f32-swizzle-ab.md # tiled f32（classic）経路へのブロック実行順スウィズル（#1034）横展開の判定基準・#1164 後の到達性整理（整列N=1024/2048/4096はpipeline経路のためclassic非到達）・GB10実機実測（ゲート0 PASS・ゲート1がN=512で判定基準未達のため不採用〈REJECT〉。`CudaGemm::new`への結線は行わない。#1139）
    │   ├── cuda-gemm-tiled-pipeline.md # cp.async 多段パイプライン（#1033）の GB10 実測記録・本番結線判断（イシュー #1137。bit 一致・parity 0 fail・N=1024/2048/4096 で 1.51〜1.74 倍改善を確認し `CudaGemm::run_tiled_f32` 系 3 入口へ形状条件付きで結線〈ADOPT〉。GB10実機実測2026-09-03）
    │   ├── cuda-gemm-candle-gate-remeasurement.md # FP32 SIMT GEMM N=1024/2048/4096 reuse の candle 比 5 回計測中央値再計測・#1031 ゲート達成判定の確定記録（正式系列〈fandhe-ai =0.6.0〉・参考系列〈#1164 結線後 HEAD〉の 2 系列併記・N=2048 candle 無効データの原因・再現条件記録。GB10 実機実測。イシュー #1142。#1184 で fail 2 要素の実値・厳密真値突合を §5.3 追記）
    │   ├── logs/cuda-gemm-candle-gate-1142/ # 上記の実行ログ・env_info（内部ホスト名は含めない。イシュー #1142）
    │   ├── logs/cuda-gemm-candle-parity-1184/ # N=2048 candle 側 parity fail 2 要素の実値ダンプ・厳密真値突合結果・env_info（内部ホスト名は含めない。イシュー #1184）
    │   ├── cuda-gemm-reuse-phase-breakdown.md # GEMM reuse 計測境界を H2D／カーネル／D2H／同期でフェーズ分解し、#1142 §4.3 の「H2D/D2H 固定費が希釈要因」推定を精緻化（実際の主因は host_copy／checksum というハーネス診断コスト）。matmul 単体は candle fresh を上回る（N=1024: 1.59倍・N=4096: 1.47倍）ことを確定。N=4096 D2H の二峰性（#1169 関連）は未確定のまま記録。GB10実機実測。イシュー #1182
    │   ├── logs/cuda-gemm-reuse-phase-1182/ # 上記の実行ログ・env_info（内部ホスト名は含めない。イシュー #1182）
    │   ├── cuda-gemm-auto-f16-mma-switch.md # CudaGemmAuto::run_f16 の MatrixUnit 分岐 mma 優先・wmma フォールバック切替（#1156）の前後比較記録。GB10実機実測完了・512/1024/2048 は非後退（1.75〜4.67倍）・4096 は#1130 病態下で base 5run範囲内。本番結線（`MMA_PRIORITY_PRODUCTION_ENABLED = true`）は PR #1179 codex-review 指摘〈K=4096 非後退ゲートの `MmaF16` baseline ceiling 未承認〉により差し戻し・`false` で承認待ち（§0）。#1160
    │   ├── metal-gemm-transpose-tiled.md # gemm_simdgroup_tiled の転置ロード（TRANS_A/TRANS_B）拡張・NT/TN/TT へのタイル variant 選択適用の実装・実機正確性実測記録（M4 Max。NN 非後退ビット同一・NT/TN/TT parity 確認済み。イシュー #1138）。性能 A/B は #1186 で `gemm_transpose_route_ab_bench.rs` を追加し計測を試みたが、同一マシンで並走する他セッションの GPU 負荷によりフェーズ 1 安定性ゲートが 4 回の調整試行でも不成立のため `verdict=undetermined`（判定不可）。`dispatch_strided_bias_act_prepared` への自動ルーティング結線は本 PR でも未実施のまま
    │   ├── logs/metal-gemm-transpose-route-ab-1186/ # 上記 A/B 計測試行の実行ログ・env_info（内部ホスト名は含めない。イシュー #1186）
    │   ├── metal-gemm-n4096-kernel-gap.md # N=4096 カーネル純境界の candle 比ギャップ（約 9.9 対 13.17 TFLOPS）縮小調査。`MTLComputePipelineState` 反射値によるレジスタ圧仮説（H1）の検証（占有率上限に不足なし・反射値レベルでは非支持）・MLX steel classic 未収録構成 `(32,64,16,1,2)`（`CANDIDATES[8]`）の追加測定（劣後のため不採用・選択ロジック変更なし）。M4 Max 実機実測 2 run（イシュー #1143）
    │   ├── metal-gemm-candle-gate-remeasurement.md # Metal GEMM N=1024/2048/4096 reuse の candle 比 5 回計測中央値再計測・#1037 ゲート達成判定の確定記録（正式系列〈fandhe-ai =0.6.0〉・参考系列〈#1167/#1168 反映後 HEAD〉の 2 系列併記。いずれも未達成。M4 Max 実機実測。イシュー #1147）
    │   ├── logs/metal-gemm-candle-gate-1147/ # 上記の実行ログ・env_info（内部ホスト名は含めない。イシュー #1147）
    │   ├── cpu-gemm-candle-gate-remeasurement.md # CPU GEMM N=512/1024/2048 reuse の candle 比 5 回計測中央値再計測・#1117 ゲート判定の確定記録（単一系列〈fandhe-ai =0.6.0〉。CPU GEMM 本番経路が v0.6.0 と HEAD で同一のため参考系列は計測せず。DGX Spark GB10〈Grace CPU〉・Apple M4 Max とも実機実測完了。両実機 N=512/1024 未達・DGX N=2048 は candle 側要素誤差超過により判定不能〈5 run で完全に決定的に再現〉・M4 Max N=2048 は未達。未達原因分析〈計測境界固定費・並列化の非単調性・マイクロカーネル効率・packing〉を含む。イシュー #1148）
    │   └── logs/cpu-gemm-candle-gate-1148/ # 上記の実行ログ・env_info・RAYON_NUM_THREADS スイープログ（内部ホスト名は含めない。イシュー #1148）
    ├── performance-targets.md # REQ-8 段階的下限の全バックエンド横断一覧（TASK-8.4・#159）
    ├── public-api-design.md            # compat API 層の公開 API 設計（REQ-9）
    ├── real-hardware-verification-env.md # 実機検証環境（Mac Metal / DGX Spark CUDA。実ホスト名はローカル管理外ファイル参照）の接続・転送・計測手順（#408・#461）
    ├── real-hardware-verification-env.local.md.example # 上記の実値（内部ホスト名等）を書くローカル用テンプレート（#461。実体は .gitignore 対象）
    ├── self-repair-candidate-isolation.md # 候補実行の OS レベル縦深防御の調査結果・採否判断（#414）
    ├── self-repair-revalidation-plan.md # TASK-3.3a 自己修復ループ再実証の実証計画・題材選定（#140）
    └── spec/                # 正本 submodule（fandhe-ai-spec。編集禁止）
        ├── 04-requirements.md  # REQ-1〜14
        ├── 05-tasks.md         # TASK 一覧（4h 粒度）
        ├── 06-roadmap.md       # M0〜M5・全 51 タスク
        └── 03-poc/             # PoC 実測（v2 系は poc-v2-*）
```

## 委譲方針（必読）

main はコンテキスト消費を抑えるため判断と統合に専念し、調査・実装・テスト・レビューは subagent へ委譲する。詳細は `.claude/rules/delegation.md`（調査・設計）・`delegation-impl.md`（作成・編集）を参照。

### model 配分

| 用途 | model |
|------|-------|
| 複雑な横断判断・アーキテクチャ設計 | opus または fable（fable は特に大規模設計・横断判断の最上位 tier） |
| 調査・生成・実装・レビュー | sonnet |
| 機械的集計・lint・ドキュメント更新 | haiku |

## Sub-agents

| カテゴリ | subagent_type | 担当 | model |
|---------|---------------|------|-------|
| research | explorer | コードベース・docs/spec 横断調査（読み取り専用） | sonnet |
| research | reference-researcher | cudarc/CUDA・objc2/Metal・safetensors/ONNX 等の外部仕様調査 | sonnet |
| implement | core-builder | `tensor-core`・`autodiff`・workspace 骨格・`facade`（composition root・compat API 層） | sonnet |
| implement | backend-builder | `backend-cpu`・`backend-cuda`・`backend-metal`・数値一致回帰テスト | sonnet |
| implement | interop-builder | `onnx-interop`（safetensors / prost 自前取り込み） | sonnet |
| implement | runtime-builder | `guardrail`・`self-repair`・`bench-harness` | sonnet |
| testing | test-runner | テスト実行・追加・失敗解析（実機依存は `#[ignore]` 分離） | sonnet |
| testing | bench-runner | ベンチ計測・性能回帰検出（5 回計測中央値・読み取り専用） | sonnet |
| quality | reviewer | コードレビュー（spec 突合・読み取り専用） | sonnet |
| quality | security-auditor | OWASP Top 10・unsafe・ライセンス監査（読み取り専用） | sonnet |
| quality | linter | fmt / clippy / frontmatter lint の機械的実行 | haiku |
| docs | docs-writer | CLAUDE.md・README・license-matrix 等の更新 | haiku |

## Rules

| ファイル | 内容 |
|---------|------|
| `.claude/rules/delegation.md` | 調査・設計フェーズの委譲原則・パスベース切り替え |
| `.claude/rules/delegation-impl.md` | 作成・編集フェーズの委譲マッピング・実装フロー標準 |
| `.claude/rules/coding-rust.md` | 完全自作コア方針・cfg ベースバックエンド・FMA 契約統一・品質基準 |
| `.claude/rules/deps-policy.md` | 許容依存 8 区分・`=x.y.z` 完全固定・禁止リスト・ライセンス要件 |
| `.claude/rules/ci.md` | **CI は GitHub ホステッド（`ubuntu-latest`）既定**（例外は codex-review の codex 実行ジョブのみ）・fork PR 対策・timeout 必須・SHA 固定・fail-closed 集約 |
| `.claude/rules/security.md` | OWASP Top 10・秘密情報混入防止・自己修復ループのガードレール |
| `.claude/rules/japanese-style.md` | 日本語出力スタイル |
| `.claude/rules/conventional-commits.md` | Conventional Commits 詳細規約（`--no-verify` 禁止） |
| `.claude/rules/code-comment-style.md` | コメント規約（役割・責務・呼び出し文脈・spec 根拠を埋め込む） |
| `.claude/rules/out-of-scope-tracking.md` | 実装対象外の追跡規約（スコープ外事項を放置しない） |

## Current Skills

`npx skills add` で導入済み（`skills-lock.json` 管理。更新は update-external.yml が自動追従）。

- **Git/GitHub 運用**: create-commit・create-pr・create-issue・create-issue-tree・update-issue-tree
- **実装フロー**: create-plan・implement-issue・implement-issue-tree・implement-review・implement-review-pr
- **ドキュメント**: update-docs・comment-code
- **スキル管理**: init-claude・update-claude・contribute-skill・sync-skills-lock
- **技術リファレンス**: rust・nvidia-cuda・apple-silicon・amd-rocm・lefthook・editorconfig・commitlint・github-docs

## Conventions

- 日本語でやりとり・報告・コミット・PR を書く（`japanese-style.md`）
- Conventional Commits 厳守・`--no-verify` 禁止（`conventional-commits.md`）
- 依存の追加・更新、ガードレール閾値・テスト許容誤差の変更はユーザー承認必須
- `docs/spec/`（正本 submodule）は編集しない。仕様変更は spec リポ側で対応する
- implement-issue は計画のユーザー承認後に実装する
- スコープ外事項は `out-of-scope-tracking.md` の規約に沿って Issue で追跡する

## hooks（settings.json）

- **SessionStart**: 日本語・委譲・完全自作コア・CI は GitHub ホステッド・Conventional Commits のリマインダーを表示する
- **PostToolUse**（Edit|Write）: `.rs` 編集時に rustfmt を自動適用する（Cargo.toml の edition を検出。未追加時は 2021 フォールバック）
