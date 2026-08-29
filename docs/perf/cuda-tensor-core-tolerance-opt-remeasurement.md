# opt 版 WMMA TF32 カーネルでの数値一致誤差分布の再実測（イシュー #994）

## 1. 位置づけ

- イシュー #994「docs(perf): opt 版 WMMA TF32 カーネルでの数値一致誤差分布の再実測」の実測記録。親トラッキングは #992（REQ-2 閾値確定の前提検証）。依存イシュー #993（PR #997）は origin/main へマージ済み。
- `docs/perf/cuda-tensor-core-tolerance-evaluation.md` §2.1 の TF32 実測は TASK-11.1c 時点の基本版カーネル（`kernels::WMMA_TF32_F32`）を RTX 3060（sm_86）で計測したものであり、TASK-11.1d（#63）で追加された共有メモリ・タイル最適化版 `kernels_wmma_opt::WMMA_TF32_F32_OPT`（opt 版）は同ドキュメント作成時点では未計測だった（同 doc 冒頭「測定条件の失効に関する注記」参照）。本ドキュメントはその opt 版を GB10 実機で再実測し、基本版との差の有無を記録する。
- 対象はスケール `s = 1` 固定・`SHAPES`（15 形状。16 定義中 1 件は意図的重複でマージ）× `SEEDS`（5 シード）の全項目。スケールスイープ（`s ∈ {0.1, 1, 10, 100}`）は後続イシュー #995 の対象であり、本イシューでは実施しない。
- **閾値定数・判定式・テスト許容誤差（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・`fandhe_ai_backend_cpu::compare`）は一切変更しない**（`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。

## 2. 実施状況（重要）

**本ドキュメントの初版時点で GB10 実機による計測は未実施である。** 実装セッションは GPU（NVIDIA driver・CUDA toolkit）非搭載環境で作業しており、`docs/real-hardware-verification-env.md` が定める DGX Spark GB10 実機への接続手段を持たない。

計画（`実装計画`「自動運転の制約」節）に従い、実機不達の場合は数値を推定・捏造せず、以下のみを先行実装として完了させた:

- `crates/backend-cuda/src/gemm.rs`: `internal-diagnostics` feature 限定の診断コンストラクタ `CudaGemm::new_tf32_opt_only`／`CudaGemm::new_tf32_basic_only`（`run_wmma_tf32` の 3 段選択〈staged→opt→basic〉から opt／basic のいずれかへ強制する。fail-closed: 強制先カーネルが使用不能な場合は基本版へ黙ってフォールバックせず `Err` を返す）と、可用性フラグの検証のみを行う `#[ignore]` 実機テスト `wmma_tf32_diagnostic_constructors_force_expected_kernel`。
- `crates/backend-cuda/examples/wmma_tolerance_probe.rs`: `--tf32-kernel <auto|opt|basic>`（環境変数 `WMMA_TOLERANCE_PROBE_TF32_KERNEL` フォールバック）を追加。`internal-diagnostics` feature 無効ビルドでは `opt`／`basic` を fail-closed で拒否する（黙って `auto` に縮退させない）。

**以降の §3〜§6 は、実機計測が完了した時点で追記する骨子（手順・機構の説明）である。§4・§5 の表は見出しのみを示し、実測値は含まない。**

## 3. 計測手順（GB10 実機。実測時にこのまま実行する）

`docs/real-hardware-verification-env.md` §3・§4・§6 準拠。ノード実名はローカル管理外ファイル（`docs/real-hardware-verification-env.local.md`）から取得し、値を本ドキュメント・PR・コミットに書かない。

1. rsync 転送（`.git`・`.codex`・`.env*`・`.local.md` 除外。`--filter=':- .gitignore'`）。転送前後で `.rev-stamp` を記録し、転送先に秘密情報が残っていないことを確認する。
2. 計測前に `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` と `nvidia-smi --query-gpu=utilization.gpu --format=csv` で GPU が空いていることを確認する（`utilization.gpu` が 0% でなければ待機・再確認。常駐サービスは停止しない）。
3. ビルド:

   ```sh
   env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
       CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
       cargo build --release -p fandhe-ai-backend-cuda \
       --features internal-diagnostics --example wmma_tolerance_probe
   ```

4. 実行（3 構成。いずれも `--scales 1` を付け、カーネル可用性ヘッダ・`kernel` 列付き 16 列表のスイープモード出力を得る）。手順 3 の `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai` はビルド成果物の出力先であって `PATH` には追加されないため、生成された example バイナリを裸のコマンド名で呼ぶと `command not found` になる。ビルド成果物の絶対パス（`$HOME/work/target-fandhe-ai/release/examples/wmma_tolerance_probe`）を明示するか、`cargo run` 形式で呼び出す:

   ```sh
   env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
       "$HOME/work/target-fandhe-ai/release/examples/wmma_tolerance_probe" \
       --scales 1 --tf32-kernel opt   > gb10-tf32-opt-s1.md
   env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
       "$HOME/work/target-fandhe-ai/release/examples/wmma_tolerance_probe" \
       --scales 1 --tf32-kernel basic > gb10-tf32-basic-s1.md
   env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
       "$HOME/work/target-fandhe-ai/release/examples/wmma_tolerance_probe" \
       --scales 1 --tf32-kernel auto  > gb10-tf32-auto-s1.md
   ```

   もしくは（`cargo run` 形式。この場合もビルド時と同じ `CARGO_TARGET_DIR`・`--features internal-diagnostics` を揃える）:

   ```sh
   env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
       CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
       cargo run --release -p fandhe-ai-backend-cuda \
       --features internal-diagnostics --example wmma_tolerance_probe -- \
       --scales 1 --tf32-kernel opt   > gb10-tf32-opt-s1.md
   ```

   各構成について exit code 0 を確認し、2 回ずつ実行して stdout が `diff` で完全一致する（決定性）ことを確認する。
5. 計測後に再度 GPU 占有状況を確認し、他プロセスが混入したランは破棄して取り直す。
6. 環境記録: `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`・`/usr/local/cuda/version.json`（または `nvcc --version`）・`rustc -V`・`.rev-stamp` の sha。
7. 生ログ 3 本を `docs/perf/tensor-core-tolerance/`（`gb10-tf32-opt-s1.md`／`gb10-tf32-basic-s1.md`／`gb10-tf32-auto-s1.md`）へ保存し、そこから §4・§5 の表を機械的に集計する（形状ごとに `fail_count` 合計・`max_abs_diff`／`max_rel_err`／`max_fail_abs_diff` の最大値。256×256×256 の重複 2 定義は 1 行へマージ）。

## 4. 実行カーネル種別の確認機構

`--tf32-kernel opt`／`basic` は `CudaGemm::new_tf32_opt_only`／`new_tf32_basic_only`（`crates/backend-cuda/src/gemm.rs`）を経由して `wmma_tf32_staged`（および opt 強制時はさらに `wmma_tf32_opt`）スロットを無効化する。`wmma_tolerance_probe` の `tf32_kernel_availability_header`（可用性ヘッダ）・`tf32_kernel_kind`（`kernel` 列）はいずれも `CudaGemm` の公開アクセサ（`wmma_tf32_staged_available`・`wmma_tf32_opt_available`・`wmma_tf32_routed_path_is_staged` 等）をそのまま読むだけの実装であり、独自の推定ロジックを持たない。そのため:

- `--tf32-kernel opt` の可用性ヘッダは `staged=no (disabled by CudaGemm::new_tf32_opt_only (diagnostic))`・`opt=yes` を表示し、`kernel` 列は全形状で `opt` になる。
- `--tf32-kernel basic` の可用性ヘッダは `staged=no (…)`・`opt=no (disabled by CudaGemm::new_tf32_basic_only (diagnostic))` を表示し、`kernel` 列は全形状で `basic` になる。
- `--tf32-kernel auto`（現行の本番既定コンストラクタ `CudaGemm::new` 相当）は形状ごとに staged／staged-swizzle／opt のいずれかが選ばれる（`run_wmma_tf32` の 3 段選択どおり）。

実測完了後、この節に 3 構成それぞれの可用性ヘッダの実測値（引用）と、`auto` 構成で各形状がどのカーネルへ実際にルーティングされたかの一覧を追記する。

## 5. opt 版 TF32 誤差分布表（15 形状 × 5 シード集計）

計測未了（GB10 実機不達。#994 の後続で追記）。

| shape | fail/total (5 seeds 合計) | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|
| （実測完了後に `docs/perf/tensor-core-tolerance/gb10-tf32-opt-s1.md` から転記） | - | - | - | - |

## 6. 基本版との差分表

計測未了（GB10 実機不達。#994 の後続で追記）。

| shape | opt fail/total | basic fail/total | opt max_fail_abs_diff | basic max_fail_abs_diff | 差の有無 |
|---|---|---|---|---|---|
| （実測完了後に `docs/perf/tensor-core-tolerance/gb10-tf32-{opt,basic}-s1.md` の行単位 `diff` から転記） | - | - | - | - | - |

## 7. sm_86 基本版実測（既存 §2.1）との対比

計測未了。実測完了後、`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §2.1（RTX 3060・sm_86・基本版）との対比を記す。GPU 世代（sm_86 vs GB10 の Blackwell 系譜）・カーネル実装（基本版 vs opt 版）の両方が異なる比較であるため、差異が生じても片方の要因に単純帰属せず、外挿もしない旨をここに明記する。

## 8. 結論と制約

- 閾値定数・判定式（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・`compare`）は本イシューでは変更しない。
- スケール依存性の検証（`s ∈ {0.1, 1, 10, 100}`）は #995、閾値改定候補の選定は #996 のスコープ。
- `docs/perf/cuda-parity-baseline.md` の `wmma_tf32`（基本版）行 2 件の provenance 未確定（`baseline_provenance_unconfirmed: true`。別シード 2000／8888）は、本イシューの計測では確定しない。今回追加した basic 強制入口（`CudaGemm::new_tf32_basic_only`）を使えば別イシューで再測定可能。
- 本イシューは GB10 実機不達のため §5・§6・§7 の数値部分が未了のまま完了させる（`実装計画`「自動運転の制約」節の承認済み方針）。実機アクセスが確保でき次第、§3 の手順をそのまま再現して数値を追記する。
