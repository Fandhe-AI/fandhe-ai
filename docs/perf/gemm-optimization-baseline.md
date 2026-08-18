# REQ-8 GEMM 性能下限表 — 分母・分子の突合基準（#481）

GEMM 最適化ツリー（ルート #479・Phase A 親 #480 の A-1）の成果物。`docs/performance-targets.md`
§2 の REQ-8 全 5 行（CPU / CUDA f32 / CUDA f16 / Metal f32 / Metal f16）について、後続 Phase
（B〜G）が共通の前提で参照できるよう「対象カーネル関数名（分子）・実機・PyTorch バージョン
（分母）・計測形状・出典 file:line」を突合し 1 つの基準ドキュメントへ確定する。

## §0 位置づけ

先例 `docs/performance-targets.md` §1・`docs/perf/performance-floor-decision.md` §1 と同じく、
本ドキュメントは**判断案**であり、記載内容は本イシュー #481 の PR レビュー・マージ（人間承認）
をもって成立する。

**本ドキュメントは REQ-8 の下限値・実測比率の数値を一切変更しない**（転記のみ）。下限値の変更は
Phase F の人間承認タスク（#577）のスコープであり、本ドキュメントでは扱わない。

## §1 REQ-8 全 5 行の突合表

| 行 | 対象カーネル（分子） | 実機 | PyTorch（分母） | 計測形状 | 出典 |
|---|---|---|---|---|---|
| CPU | PoC-v2-1 旧経路（SIMD 未適用の初期実装。現行の本番演算経路は `gemm_blis_parallel`〈`crates/backend-cpu/src/gemm_blis/mod.rs`・`crates/backend-cpu/src/ops.rs:67` から呼ばれる BLIS 5-loop・NEON/AVX2/AVX-512 マイクロカーネル dispatch〉であり、5.3% の分子（PoC-v2-1 計測対象）とは別物。現行経路での実値確定は Phase A の A-8〈#488〉のスコープ。A-8 の計測ハーネス・実測記録は
`docs/perf/cpu-gemm-baseline-remeasurement.md` を参照〈実装セッション時点では M4 Max 実機未到達の
ため実測値は未記入〉。Phase E 完了時点の再計測は `docs/perf/cpu-gemm-phase-e-remeasurement.md`
（#567）、Phase F の最適化後確定計測は `docs/perf/cpu-gemm-optimized-remeasurement.md`（#574。同じく
M4 Max 実機未到達のため実測値は未記入）を参照） | Apple M4 Max | PyTorch 2.13.0 macOS arm64 | 2048/4096（512 は参考値。起動オーバーヘッド支配のため判定対象外） | `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「計測結果」節・`docs/performance-targets.md:25` |
| CUDA f32 | `wmma_tf32`（Rust 側入口 `CudaGemm::run_wmma_tf32`／`CudaGemm::launch_wmma_tf32`〈`crates/backend-cuda/src/gemm.rs:657,1026`〉。NVRTC カーネル本体は opt 経路 `gemm_wmma_tf32_opt`〈`crates/backend-cuda/src/kernels_wmma_opt.rs:196`〉、基本版 `gemm_wmma_tf32`〈`crates/backend-cuda/src/kernels.rs:303`〉） | DGX Spark GB10 | PyTorch 2.13.0+cu130（同一 GB10 個体で #390 実機再計測） | 4096 が最小（25.64〜25.69%） | `docs/perf/cuda-floor-remeasurement.md`「実測結果（#390 実機実測・DGX Spark GB10・実施日 2026-08-10）」節（該当表は同ファイル 200〜233 行付近） |
| CUDA f16 | `mma_f16`（`mma.sync` パイプライン。launch-only 計測へ境界統一済み。NVRTC カーネル本体 `gemm_mma_f16`〈`crates/backend-cuda/src/kernels_mma.rs:234`〉） | 同上 | 同上 | 2048 が最小（12.97%） | 同上 |
| Metal f32 | **2 系列併記**（§2 参照）: (a) PoC-v2-4 旧 simdgroup カーネル・バッファ常駐前提の計測、(b) 現行 `gemm_simdgroup_tiled`〈`crates/backend-metal/src/shaders/gemm.metal:333`〉＋動的タイル選択入口 `MetalGemm::dispatch_auto`〈`crates/backend-metal/src/gemm.rs:245`〉・アップロード/readback 込みの計測 | Apple M4 Max | PyTorch 2.13.0 | 4096（系列 (a) の 23.2% が現行下限の分子。系列 (b) は対 PyTorch 比 未算出） | `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「PyTorch MPS 比」表・`docs/perf/metal-gemm-dynamic-tile.md`「実測結果（イシュー #381・2026-08-10 実測）」「PoC-v2-4・REQ-8 との関係」節 |
| Metal f16 | `gemm_simdgroup_f16`（simdgroup 経路。MSL カーネル本体〈`crates/backend-metal/src/shaders/gemm.metal:247`〉。Rust 側計測入口 `MetalGemm::dispatch_f16_prepared_unverified`〈`crates/backend-metal/src/gemm.rs:421`〉、ベンチ入口 `crates/backend-metal/examples/gemm_f16_bench.rs`） | Apple M4 Max | PyTorch 2.13.0（MPS f16） | 4096 が最小（18.6%。2048 は 21.6%） | `docs/perf/metal-f16-vs-mps-f16.md`「実測結果（イシュー #383・実機実測済み）」節（該当行は同ファイル 157〜158 行付近） |

補足:

- CUDA `wmma_tf32`・`mma_f16` は `docs/perf/cuda-floor-remeasurement.md`「数値一致（parity）状態の限定条件」節に記載の通り、#389 §5.3 の parity 恒常 fail 対象と一致する（#186 由来。REQ-2 改定は spec リポジトリ側対応待ち。詳細は `docs/performance-targets.md` §6）。
- 上表の CUDA f32/f16 行は Phase B（親 #490）着手前・TASK-8.3c（#157/#390）時点の実測。Phase B・Phase C（親 #503）適用後の確定計測は `docs/perf/cuda-optimized-remeasurement.md`（#571・Phase F-1）が別ファイルとして記録する（実測値記入は実機セッションへ申し送り中）。
- Transformer 複合ワークロード行（非実機参考値 約 6.1%。QEMU 仮想 CPU）は本表のスコープ外（対象は GEMM 5 行のみ）。#479 の整理（分母に使わない・実機実測は Phase G で確定予定）を参照。

## §2 Metal 2 系列の対応関係と基準系列の決定

### 系列の対応関係

| | 系列 (a) PoC-v2-4（バッファ常駐前提） | 系列 (b) #381 `dispatch_auto`（転送込み） |
|---|---|---|
| naive @4096 | 1.271 TFLOPS | 0.9198 TFLOPS |
| tiled @4096 | 2.123 TFLOPS | 1.2207 TFLOPS |
| simdgroup @4096 | 3.134 TFLOPS | 1.7432 TFLOPS |
| dynamic-tile-auto @4096 | （未計測。系列 (b) のみが持つ経路） | 3.0283 TFLOPS |
| 出典 | `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md` | `docs/perf/metal-gemm-dynamic-tile.md`「実測結果」節・正方形状表 |

同一カーネル（naive/tiled/simdgroup）で系列 (a) が系列 (b) を上回るのは、計測範囲の違い（(a) は
バッファ常駐前提、(b) は 1 ディスパッチごとに A・B アップロードと C readback を含む）による。
`docs/perf/metal-gemm-dynamic-tile.md`「PoC-v2-4・REQ-8 との関係」節が明記する通り、**両系列は計測
境界が異なり絶対値を直接比較できない**。

### 「1024 以降 2.2〜2.4 TFLOPS プラトー」の帰属

この値域は「候補構成別（size=2048 固定・協調ロード有無比較）」表の `staged` 系列にのみ対応する
（`docs/perf/metal-gemm-dynamic-tile.md`「候補構成別」節）: `bm64_bn64_bk16_staged` 2.3572 TFLOPS・
`bm32_bn32_bk16_staged` 2.4030 TFLOPS。

Appendix の再現性確認ラン（run2/run3）における `dynamic-tile-auto` の値は、この帰属の裏付けには
**ならない**。run2 は size 1024/2048/4096 で 1.3981/2.4505/2.1868 TFLOPS、run3 は 1.3392/2.1593/2.4974
TFLOPS であり、size=1024 はいずれも 1.3 TFLOPS 台でプラトー値域（2.2〜2.4 TFLOPS）の外にあるうえ、
size=2048→4096 の傾向も run2（下降）と run3（上昇）で逆向きであり、両者は run 間で一貫した傾向を
示さない。よって run2/run3 は「1024 以降 2.2〜2.4 TFLOPS プラトー」の根拠には含めない（正方形状の
run 間ばらつきの一例としてのみ扱う）。

一方、正方形状の canonical run1 系列（size 1024→2048→4096 で 1.2909→2.5001→3.0283 TFLOPS と
**単調増加**）もこのプラトーを示さない。よって「1024 以降 2.2〜2.4 TFLOPS プラトー」は候補構成別
の staged 系列（size=2048 固定でのパラメータ探索）に限定された観測であり、canonical run1 の正方
形状スケーリングとも run2/run3 の再現ばらつきとも別系列として扱う。A-7（#487）の定量診断はこの
限定された帰属（staged 系列のみ）を前提に行う。

### 基準系列の決定（判断案）

**基準系列は本ドキュメントでは確定しない**。ただし系列 (a)・(b) は同列の理由で除外されるわけではない
（「プロトコル適合性」と「再現性」は別問題として分けて扱う）:

- 系列 (a)（PoC-v2-4・バッファ常駐前提）は `docs/performance-targets.md` §4 の計測プロトコルに
  **適合する**歴史的基準である（同 §4「warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3 を記録する
  （PoC-v2-1/3/4 はすべて本条件で実施済み）」・「同期方式は『ホスト転送を伴わない完了待ち』で統一する」
  の双方を PoC-v2-4 計測時点で満たしている。本ドキュメント §2「同一カーネル（naive/tiled/simdgroup）で
  系列 (a) が系列 (b) を上回るのは…(a) はバッファ常駐前提」の記述の通り、系列 (a) はホスト転送を計測
  区間外に置く構成である）。ただし現行 f32 API（`dispatch`／`dispatch_variant`／`dispatch_auto`／
  `dispatch_backend_auto`）には同型の prepared（バッファ常駐）入口が存在しないため、**現時点では
  再計測により再現できない**（後述 2）。
- 系列 (b)（現行 `dispatch_auto`・転送込み）は 1 ディスパッチごとに A・B アップロードと C readback を
  計測区間に含む構成のため、単独では `docs/performance-targets.md` §4 の同期方式契約（「ホスト転送を
  伴わない完了待ち」。REQ-8 5 行・全バックエンド共通で比較 2 系列の境界を揃えれば良いという相対規定
  ではない）を**満たさない**。

理由:

1. 系列 (b)（現行 `dispatch_auto`）は 1 ディスパッチごとに A・B アップロードと C readback を計測区間に
   含む構成のため、単独で `docs/performance-targets.md` §4 の同期方式契約を満たさない。よって系列 (b)
   への一本化は同 §4 と整合しない。
2. `docs/performance-targets.md` §4 準拠のバッファ常駐（ホスト転送を計測区間外に置く）計測入口は、
   Metal f16 側にはすでに存在する（`MetalGemm::dispatch_f16_prepared_unverified`〈`crates/backend-metal/
   src/gemm.rs:421`〉。`crates/backend-metal/examples/gemm_f16_bench.rs` のドキュメンテーションコメント
   「パディング・バッファ確保／アップロードは計測ループの外で 1 回だけ行い、計測対象はディスパッチ
   （エンコード＋コマンドバッファ完了待ち）のみ」を参照）。一方 f32 側には同型の prepared 入口が存在せず、
   現行ベンチ入口（`crates/backend-metal/examples/gemm_bench.rs`）を素朴に再実行しても系列 (a) と同一の
   計測境界を再現できない。これは系列 (a) 自体が `docs/performance-targets.md` §4 に不適合であることを
   示すものではなく、現行 API での**再現性の問題**である。
3. 既確定の下限（初期リリース 20%・最適化後 30%。分子 23.2% は系列 (a) 由来）は本ドキュメントでは
   **変更しない**。系列 (a) は `docs/performance-targets.md` §4 に適合する歴史的基準であり分子 23.2% の
   適格性は本整理により損なわれないが、再現性の問題（理由 2）があるため以降の Metal f32 目標値の基準
   系列を確定するには、(i) f16 と同型の同 §4 準拠 prepared ディスパッチ入口を f32 側にも用意したうえで
   PyTorch MPS 側も同一の転送除外境界で再計測するか、(ii) `docs/performance-targets.md` §4 の計測
   プロトコル自体を正式な承認付き変更として改定するか、いずれかが前提となる。両者ともコード変更・
   プロトコル改定を伴い本イシュー（ドキュメントの分母・分子突合のみ）のスコープ外であり、別タスクとして
   切り出す。(i) の f32 側再計測は Phase F の Metal 確定計測タスク（#572。f32 prepared 入口
   `MetalGemm::dispatch_tiled_prepared`〈`crates/backend-metal/src/gemm.rs`〉を追加し、
   計測手順・記録テンプレートを `docs/perf/metal-floor-remeasurement.md` に整備済み。実測値の記入は
   Mac 実機セッションへ申し送り）、下限値への反映は Phase F の人間承認タスク（#577）が既存の追跡先
   である。(ii) のプロトコル改定は上記いずれの既存子イシューにも
   明示のスコープとして含まれておらず、新規の子イシュー起票が必要だが、起票自体は
   `.claude/rules/out-of-scope-tracking.md` の定める人間承認事項のため本ドキュメントでは起票しない
   （承認後に Phase D〈#530〉または Phase F 配下へ追加する）。

## §3 CPU 対象実機の決定と E-8 スイープ対象の確定

### 決定（判断案）

REQ-8 CPU 行の基準実機は **Apple M4 Max（PyTorch 2.13.0 macOS arm64）を維持する**。

理由:

1. 確定済み下限（初期リリース 5%・最適化後 20%）の分母は M4 Max 実測であり、実機の変更は分母の
   付け替え＝実質的な下限再設定に相当するため、本イシューの「値を変更しない」制約の範囲を超え
   Phase F（#577・人間承認）の判断事項になる。
2. PoC-v2-1・Metal 系列（§2）が同一個体で計測衛生を確立済み（`docs/perf/metal-gemm-dynamic-tile.md`
   計測環境表の先例）であり、実機を揃えることで REQ-8 5 行間の計測衛生の一貫性を保てる。

これに伴い、**A-8（#488）の再計測・E-8（#564）の MC/KC/NC 再チューニングスイープは M4 Max 実機で
実施する**と確定する。

DGX Spark GB10 の CPU（Grace・Cortex-X925/A725）は参考系列と位置づけ、REQ-8 の分母には使わない
（本表に別行を新設しない）。BLIS 系参照 config（thunderx2/altra/firestorm 等。
`crates/backend-cpu/src/gemm_blis/mod.rs` の MC/KC/NC 選定コメント参照）は E-8 実装時の探索起点
としてのみ参考にする。同ファイルの MC/KC/NC 定数コメント（`mod.rs:75〜87` 付近）は「再チューニング
は #24 のスコープ」と記すのみで、対象実機の明記はない。対象実機（M4 Max）の明記自体は本イシューが
新たに確定する事項であり、対応するコメント更新は E-8 実装のスコープとする（本イシューでは触らない）。

QEMU 仮想 CPU 環境（`docs/perf/cpu-gemm-rayon-tuning.md`）は改善「比」専用の参考値であり、絶対値
比較には使えない（既存整理の再掲）。

**E-8（#564）の実施状況**: `crates/backend-cpu/src/gemm_blis/mod.rs` の MC/KC/NC 定数コメントへ
対象実機（M4 Max）とスイープ記録ドキュメントへの参照を追記し、パラメータ化・パリティテスト・
実機スイープハーネスまで整備した（`docs/perf/cpu-gemm-blocking-sweep.md`）。M4 Max 実機での実測・
選定は環境ゲート未達のため未実施（fail-closed。現行値 128/256/512 を維持）。

## §4 共通契約の遵守

GEMM 最適化ツリー（#479）の共通契約に対する本ドキュメントの遵守状況:

- **境界チェック不省略**: 対象外（`.claude/rules/coding-rust.md` のカーネル実装規約はコード変更を
  伴わない本ドキュメントには適用対象がない）。
- **tolerance 不緩和**: `crates/bench-harness/src/threshold.rs` の下限定数・数値一致テストの許容誤差
  は変更していない。
- **依存追加なし**: `Cargo.toml`・`Cargo.lock` は変更していない。
- **`docs/spec/`（正本 submodule）不編集**: 変更していない。
- **REQ-8 下限値不変更**: `docs/performance-targets.md` §2 の下限値・状態列・実測比率は本ドキュメント
  からリンクを追加した以外の変更をしていない（下限値の変更判断は Phase F・#577 に委ねる）。

