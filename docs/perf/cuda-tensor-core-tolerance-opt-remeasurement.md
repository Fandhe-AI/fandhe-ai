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

**追記（イシュー #995・2026-08-29）**: 上記の実機不達は本ドキュメント初版時点のものであり、後続イシュー #995 が GB10 実機での計測を実施した。#995 のスケールスイープ計測（`--scales 0.1,1,10,100 --tf32-kernel {auto,opt,basic}`）は `s = 1` を含むため、その `scale = 1` 行が本イシュー §3 手順が想定していた `--scales 1` 単独実行と同一の入力・シード方針・カーネル強制構成に相当する（`ScaleConfig::Sweep` は指定したスケール集合ごとに独立して乱数列を再シードするため、`--scales 1` 単独実行と `--scales 0.1,1,10,100` 実行の `scale=1` 行は数値的に同一になる）。そのため個別に `--scales 1` の 3 ラン・2 回ずつを追加実行せず、`docs/perf/tensor-core-tolerance/gb10-sweep-tf32-{auto,opt,basic}.md` の `scale = 1` 行を本イシューの §5〜§7 の実測データとして転用する。計測環境・再現コマンドは `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §2〜§3 を参照（本ドキュメントでは重複記載しない）。§5〜§7 の数値部分を以下のとおり実測値で更新する。

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

実測（#995・2026-08-29。GB10・compute_121）: 3 構成の可用性ヘッダは以下のとおり。

```
# --tf32-kernel auto
kernel availability: staged=yes, staged-swizzle group width=8, opt=yes, basic=yes
routing rule: staged if n%4==0 && k%4==0 (swizzle variant if size condition) -> opt -> basic

# --tf32-kernel opt
tf32 kernel select: opt
kernel availability: staged=no (disabled by CudaGemm::new_tf32_opt_only (diagnostic)), staged-swizzle group width=none, opt=yes, basic=yes

# --tf32-kernel basic
tf32 kernel select: basic
kernel availability: staged=no (disabled by CudaGemm::new_tf32_basic_only (diagnostic)), staged-swizzle group width=none, opt=no (disabled by CudaGemm::new_tf32_basic_only (diagnostic)), basic=yes
```

`auto` 構成でのルーティング先（15 形状）: `staged`（32x32x32・64x64x64・128x128x128・256x256x256〈両定義〉・512x512x512・100x100x100・64x96x128・256x256x512・256x256x1024・256x256x4096）、`opt`（1x1x1・17x23x19・17x19x23・33x31x65・130x70x90）。`basic` への降格は auto 構成では発生しなかった。

## 5. opt 版 TF32 誤差分布表（15 形状 × 5 シード集計。s=1）

実測（#995 のスケールスイープ生ログ `gb10-sweep-tf32-opt.md` の `scale=1` 行を集計。手法は `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §3 の集計ワンライナーと同一）:

| shape | fail/total (5 seeds 合計) | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|
| 32x32x32（block tile） | 807/5120 | 1.857e-3 | 2.556e-1 | 1.857e-3 |
| 64x64x64（block tile ×2） | 3373/20480 | 3.377e-3 | 1.046e0 | 3.377e-3 |
| 128x128x128（block tile ×4） | 13140/81920 | 4.337e-3 | 1.911e0 | 4.224e-3 |
| 256x256x256（block tile ×8。重複マージ済み） | 53435/327680 | 6.827e-3 | 1.927e0 | 6.827e-3 |
| 512x512x512（block tile ×16） | 212611/1310720 | 9.986e-3 | 1.980e0 | 9.986e-3 |
| 1x1x1（sub-K-tile） | 0/5 | 1.228e-4 | 5.941e-4 | 0.000e0 |
| 17x23x19（非倍数エッジ） | 312/1955 | 1.265e-3 | 8.375e-1 | 1.199e-3 |
| 17x19x23（非倍数エッジ） | 240/1615 | 1.562e-3 | 1.495e-1 | 1.562e-3 |
| 33x31x65（非倍数エッジ） | 826/5115 | 2.724e-3 | 5.746e-1 | 2.724e-3 |
| 100x100x100（非倍数エッジ） | 7957/50000 | 3.624e-3 | 1.884e0 | 3.624e-3 |
| 130x70x90（非倍数エッジ） | 7280/45500 | 4.167e-3 | 1.539e0 | 4.167e-3 |
| 64x96x128（非正方） | 5066/30720 | 4.180e-3 | 1.743e0 | 4.177e-3 |
| 256x256x512（K スイープ） | 52797/327680 | 9.242e-3 | 1.929e0 | 9.242e-3 |
| 256x256x1024（K スイープ） | 53494/327680 | 1.297e-2 | 1.891e0 | 1.297e-2 |
| 256x256x4096（K スイープ） | 53286/327680 | 2.544e-2 | 1.998e0 | 2.544e-2 |

## 6. 基本版との差分表

**全形状で差分なし**: `opt` 強制構成と `basic` 強制構成の生ログを `kernel` 列を除いて突合したところ、全形状・全シード（`s=1` に限らず #995 が計測した全スケールでも同様）で数値（`fail/total`・`max_abs_diff`・`max_rel_err`・percentile 列・`max_fail_abs_diff`）が完全一致した（`diff` 差分 0 行）。GB10 では TF32 WMMA カーネルのタイル戦略（共有メモリ・swizzle の有無）が誤差分布に一切影響しない。詳細・全スケールでの確認結果は `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §5 を参照。

| shape | opt fail/total | basic fail/total | opt max_fail_abs_diff | basic max_fail_abs_diff | 差の有無 |
|---|---|---|---|---|---|
| 全 15 形状（§5 の表と同一値） | （§5 と同一） | （§5 と同一） | （§5 と同一） | （§5 と同一） | なし |

## 7. sm_86 基本版実測（既存 §2.1）との対比

実測（#995）: `basic` 強制構成（sm_86 実測時点のカーネルソースと同一）で対比した結果、**fail 率・`max_fail_abs_diff` はいずれの形状も表示桁（3 桁）でほぼ完全一致し、系統的な世代差は見られなかった**（最大差は 256x256x4096 の `max_fail_abs_diff` で 2.544e-2〈GB10〉vs 2.535e-2〈sm_86〉の 0.4% 差）。GPU 世代（sm_86 vs GB10）とカーネル実装（基本版 vs opt 版）を分離するため、本対比は `basic` 強制構成（sm_86 と同一カーネルソース）のみを用いており、opt 版固有の差異は §6 のとおり `basic` と数値的に同一なので追加で考慮する必要はない。詳細な形状別比較表は `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §10 を参照（本ドキュメントでは重複記載しない）。

## 8. 結論と制約

- 閾値定数・判定式（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`・`compare`）は本イシューでは変更しない。
- スケール依存性の検証（`s ∈ {0.1, 1, 10, 100}`）は #995、閾値改定候補の選定は #996 のスコープ。
- `docs/perf/cuda-parity-baseline.md` の `wmma_tf32`（基本版）行 2 件の provenance 未確定（`baseline_provenance_unconfirmed: true`。別シード 2000／8888）は、本イシューの計測では確定しない。今回追加した basic 強制入口（`CudaGemm::new_tf32_basic_only`）を使えば別イシューで再測定可能。
- §5・§6・§7 の数値部分はイシュー #995 の GB10 実機計測（`scale=1` 行）により実測完了した（2026-08-29 追記）。
