# Metal 最適化後 f32/f16 対 PyTorch MPS 比 確定計測 記録（#572・Phase F-2）

イシュー #572「bench(backend-metal): 最適化後の Metal f32/f16 対 PyTorch MPS 比を確定計測」の実測記録。
GEMM 最適化ツリー（ルート #479）の Phase F（親 #569「再計測・parity 非後退確認・REQ-8 下限再確定」）の
F-2 に対応する。

## 目的・受け入れ条件対応

Phase D（Metal マルチ simdgroup 化・ロード最適化。親 #530、D-1〜D-5 が本番経路〈`MetalGemm::dispatch_auto`〉
に適用済み）完了後の Metal f32/f16 GEMM について、`docs/performance-targets.md` §4 の計測プロトコル
（warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3、ホスト転送を伴わない完了待ち、決定的シード、判定
対象形状 2048/4096）で対 PyTorch MPS 比を**確定計測**し、既存 `docs/perf/` と同形式で記録する。

本イシューの核心は f32 側の計測境界問題の解消である。`docs/perf/gemm-optimization-baseline.md` §2 が
確定したとおり:

- Metal f32 の現行ベンチ入口 `MetalGemm::dispatch_auto`（`crates/backend-metal/src/gemm.rs:297`）は
  1 ディスパッチごとに A/B アップロード＋C readback を含む「転送込み」境界であり、単独では §4 の
  同期方式契約（ホスト転送を伴わない完了待ち）を満たさない
- f16 側には §4 準拠の prepared 入口 `dispatch_f16_prepared_unverified`（同 `gemm.rs:475`。エンコード＋
  コマンドバッファ完了待ちのみ計測）が既に存在するが、**f32 側には同型の prepared 入口が存在しなかった**
- 同ドキュメント §2 は「(i) f16 と同型の §4 準拠 prepared ディスパッチ入口を f32 側にも用意したうえでの
  f32 再計測は Phase F の #572 のスコープ」と明記していた

本イシューでこの (i) を解消する [`MetalGemm::dispatch_tiled_prepared`]（`crates/backend-metal/src/gemm.rs`）
を追加した（下記「実測バイナリ」参照）。

依存 #547（D-10。`docs/perf/metal-gemm-dynamic-tile.md`「Phase D 完了時点再計測」節）は close 済み
（PR #696。計測手順・記録テンプレート整備済み）。

## 実行環境の制約（本ドキュメント作成セッション）

**2026-08-18 追記**: 本節は当初のドキュメント整備セッション時点の記録であり、Metal 実機（Apple M4
Max）に到達できないという制約は当時のものである。2026-08-18 に実機（Mac ローカル）セッションで計測を
完了した（「f32 結果」「f16 結果」節以降の実測値・「状態」節を参照）。本節は経緯の記録として書き換え
ず残す。

本ドキュメントは Linux worktree で作成された。`docs/real-hardware-verification-env.md` §1 のとおり
Metal 実機（Apple M4 Max）は「ローカル直接実行」であり、本 Linux 環境からは到達できない（SSH リモートは
CUDA ノードのみ）。よって本イシューは #547（D-10）の先例と同方式を採る:

1. f32 prepared 入口のコード整備（Linux でビルド・clippy・非実機テストまで検証可能）
2. ベンチ入口・parity テストの整備
3. 計測手順＋記録テンプレートの完全整備

**実測値の記入は Mac 実機セッションへ申し送る**（下記「状態」節参照）。

## 実測バイナリ

### f32: `crates/backend-metal/examples/gemm_f32_prepared_bench.rs`（本イシューで新規追加）

- `MetalGemm::dispatch_tiled_prepared`（本イシューで新規追加。`crates/backend-metal/src/gemm.rs`）を
  使う。`tile::select(m, n, k)` で選んだ [`TileConfig`] 候補と、事前確保・アップロード済みの
  [`MetalBuffer`]（A/B/C。実効次元 8 の倍数へ [`pad8`] 済み）を渡し、エンコード＋コマンドバッファ
  完了待ちのみを計測する（f16 側 `dispatch_f16_prepared_unverified` と同型の計測境界）
- `pipeline_for_tile` がデバイス上限超過等でサイレントに `TileConfig::SINGLE_SIMDGROUP_8X8` へ
  フォールバックしうるため、`dispatch_tiled_prepared` は実際に採用された構成（resolved）を返す。
  ベンチ出力の `resolved_tile_config=` にこれを含め、フォールバック透明性を確保する
- 既存 `gemm_bench.rs`（`dispatch_auto` 経由・転送込み境界。#381 比較系列）は改変しない。両者は
  独立した計測系列として維持する
- 形状: M=N=K = 512／1024／2048／4096（`gemm_f16_bench.rs` と同一形状帯。512 は起動オーバーヘッド
  支配のため参考値）
- 計測プロトコル: `bench_harness::protocol::run`（warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3。
  TASK-8.1）・決定的シード `0xC0FFEE`

### f16: `crates/backend-metal/examples/gemm_f16_bench.rs`（既存。イシュー #156・#380 で確立済み）

`MetalGemm::dispatch_f16_prepared_unverified` を使う既存バイナリをそのまま再利用する（変更なし）。

## 入力検証（OWASP A03。`.claude/rules/security.md`）

`dispatch_tiled_prepared` の呼び出し元は任意の実効次元・バッファ長を渡せるため、エンコード（FFI）前に
[`validate_prepared_inputs_f32`]（`crates/backend-metal/src/gemm.rs`）が以下を fail-closed で検証する
（f16 版 `validate_prepared_inputs`・PR #346 codex-review P1-1 指摘と同水準）:

1. `m_eff`/`n_eff`/`k_eff` がいずれも 8 の倍数であること
2. `a_buf.len() == m_eff*k_eff`・`b_buf.len() == k_eff*n_eff`・`c_buf.len() == m_eff*n_eff`

回帰確認は `tests/gemm_dynamic_tile_parity.rs::dispatch_tiled_prepared_rejects_undersized_and_misaligned_inputs`
（`#[ignore]`・Metal 実機依存。`MetalBuffer` の確保に Metal デバイスが必要なため Linux 上の pure 単体
テストは書けない。`crates/backend-metal/src/gemm.rs` 内コメント参照）で行う。

### `scripts/bench/gemm_bench_torch_mps_f32.py` docstring 注記の適用範囲整理

同スクリプトの docstring は「f32 側は `dispatch_auto` が転送込み境界のため …
本スクリプトによる対 MPS f32 比は計測境界差の注記付き参考値とし、REQ-8 の分母・分子には使わない」と
記す。この注記は **`crates/backend-metal/examples/gemm_bench.rs`（`dispatch_auto`。転送込み境界）と
対で比較する場合**（`docs/perf/gemm-optimization-baseline.md` §2 系列 (b)）に限定した注意書きであり、
本ドキュメントが用いる `dispatch_tiled_prepared`（§4 準拠 prepared 入口。エンコード＋コマンドバッファ
完了待ちのみ計測）との比較には適用されない。同スクリプト自体は PyTorch 側テンソルを事前に MPS デバイス
へ配置した状態で `torch.mm()` 呼び出しのみを計測しており（アップロード・readback を計測区間に含まない）、
Metal 側の計測境界を prepared 入口へ揃えた本ドキュメントの比較では両者の同期方式契約（ホスト転送を
伴わない完了待ち）が一致する。したがって本ドキュメントの f32 対 PyTorch 比は REQ-8 の分母・分子として
そのまま使用できる（本イシュー #572 の核心である「f32 側計測境界問題の解消」の帰結。「目的・受け入れ
条件対応」節参照）。

## 数値一致（parity）確認

`tests/gemm_dynamic_tile_parity.rs::dispatch_tiled_prepared_matches_dispatch_variant`（`#[ignore]`・
Metal 実機依存）が、`dispatch_tiled_prepared`（prepared 入口）と `dispatch_variant`（一括入口）の
出力が完全一致することを確認する（計測境界のみが異なる同一カーネル呼び出しのため）。既存 tolerance
定数・REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は変更しない。

## 計測手順（Apple Silicon 実機）

```sh
git fetch origin
git checkout bench/572-metal-floor-remeasurement   # 本イシューの実装ブランチ

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture

# 2. Rust 側ベンチを各 5 回独立実行し、size ごとに中央値を採用する
#    （MeasurementConfig::default() 自体が warmup 20・計測 20・中央値を内包するため、
#    5 プロセス独立実行との組み合わせで「5 回計測の中央値」下限
#    〈.claude/rules/coding-rust.md〉を二重に満たす。#547 先例と同方式）
cargo run -p fandhe-ai-backend-metal --example gemm_f32_prepared_bench --release
cargo run -p fandhe-ai-backend-metal --example gemm_f16_bench --release
```

PyTorch 側は一時 venv（リポジトリ管理外。`.venv-mps-bench` 先例）で実行する:

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f32.py
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

Rust 側と同様に各 5 回独立実行し、size ごとの中央値を採用する。

計測衛生（#381・#383・#547 先例と同方式）: AC 電源接続、外部ディスプレイのコンポジタ負荷を許容するが
他 GPU 負荷アプリ（ブラウザ動画・Xcode ビルド・ローカル LLM 等）は終了する。Rust 側・PyTorch 側の
同時実行を避け、各ラン前後に
`pgrep -fl "gemm_f32_prepared_bench|gemm_f16_bench|gemm_bench_torch_mps"` で他プロセスとの競合が
ないことを確認する（競合検出時は破棄・取り直す）。

## 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max |
| OS | macOS 26.6.1 (25G76) |
| rustc | 1.96.0 |
| torch | 2.13.0（`torch.backends.mps.is_available()` true） |
| 計測コミット SHA | `abaa94e`（下記「計測対象コミットの補足」参照） |
| 実施日 | 2026-08-18 |
| 計測衛生 | AC 電源接続・他 GPU 負荷アプリなし。各ラン前後に `pgrep -fl "gemm_f32_prepared_bench\|gemm_f16_bench\|gemm_bench_torch_mps"` で競合プロセス非介在を確認 |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20・計測 20・中央値/Q1/Q3。TASK-8.1）を 5 回独立実行し size ごとに中央値採用（Rust・PyTorch 双方） |
| 決定的シード | `0xC0FFEE` |
| 同期境界 | Rust: コマンドバッファ完了待ち（f32: `dispatch_tiled_prepared`／f16: `dispatch_f16_prepared_unverified`）／PyTorch: `torch.mps.synchronize()` |

### 計測対象コミットの補足

計測手順「状態」節が挙げるブランチ `bench/572-metal-floor-remeasurement` は origin に存在しない
（イシュー消化後にブランチ削除済みと判断）。当該ドキュメントを新設した実装コミット `35514db`
（PR #700）は main へマージ済みで、実測時点の main tip `abaa94e` はその子孫であるため、`abaa94e` を
計測対象とした。`35514db..abaa94e` の `crates/backend-metal` 差分（#717 の elementwise カーネル追加・
`gemm_bias_act` の実融合化〈`encode_dispatch_bias_act`・`validate_bias_act_dims` 等はいずれも
`run_tiled_bias_act_f32` 専用の新規追加関数〉・#724 の clippy 修正）を `git diff` で確認し、計測対象
入口（`dispatch_tiled_prepared`・`dispatch_f16_prepared_unverified`・`encode_dispatch`・
`validate_prepared_inputs_f32`・`tile::select`）本体に差分がないことを直接確認した。

## f32 結果（`dispatch_tiled_prepared`。§4 準拠 prepared 入口）

`docs/performance-targets.md` §4 は各計測の中央値に加え Q1/Q3 の記録を必須とする。Q1/Q3 は
`gemm_f32_prepared_bench` の出力（`q1_tflops=`/`q3_tflops=`。`bench_harness::protocol::Measurement`
の `q1_secs`/`q3_secs` を TFLOPS へ変換したもの）をそのまま転記する。5 回独立実行する運用（「計測手順」
節）のため、Metal 列は中央値を採用した run の Q1/Q3 を記入する。

| size | Metal f32 TFLOPS（5 回中央値） | Metal f32 Q1/Q3 TFLOPS | 採用 TileConfig（resolved） | PyTorch MPS f32 TFLOPS（5 回中央値） | Metal/PyTorch 比 |
|------|------|------|------|------|------|
| 512  | 0.5769 | 0.5424 / 0.7004 | `{bm:64,bn:64,bk:16,wm:2,wn:2,staged:true}` | 0.8221 | 70.17% |
| 1024 | 1.4650 | 1.3483 / 1.4947 | `{bm:64,bn:64,bk:16,wm:2,wn:2,staged:true}` | 5.3128 | 27.57% |
| 2048 | 1.4543 | 1.4472 / 1.4685 | `{bm:64,bn:64,bk:16,wm:2,wn:2,staged:true}` | 10.0123 | 14.53% |
| 4096 | 1.5666 | 1.5601 / 1.5731 | `{bm:64,bn:64,bk:16,wm:2,wn:2,staged:true}` | 12.0447 | **13.01%** |

採用 TileConfig は全 5 run・全 size で一貫して `{bm:64,bn:64,bk:16,wm:2,wn:2,staged:true}` が resolved
された（`pipeline_for_tile` によるフォールバック発生なし）。

判定対象形状（REQ-8「判定対象形状」節）は **M=N=K=2048・4096 の実測比率の最小値**。512/1024 は参考値。
判定対象形状の最小比率 = 13.01%（4096）。

候補下限値（参考算出。`bench_harness::floor_lower_bound` を用いる）: `floor_lower_bound(13.01%)` = **10%**

## f16 結果（`dispatch_f16_prepared_unverified`。既存入口）

f32 側と同様、`gemm_f16_bench` の出力（`q1_tflops=`/`q3_tflops=`）を Q1/Q3 列へ転記する
（`docs/performance-targets.md` §4 必須。本イシューで `gemm_f16_bench.rs` の出力へ Q1/Q3 を追加した）。

| size | Metal f16 TFLOPS（5 回中央値） | Metal f16 Q1/Q3 TFLOPS | PyTorch MPS f16 TFLOPS（5 回中央値） | Metal/PyTorch 比 | 対 #383 分母（PyTorch）変化率 |
|------|------|------|------|------|------|
| 512  | 0.8749 | 0.8520 / 0.8918 | 0.9755 | 89.69% | -19.08%（#383: 1.2055） |
| 1024 | 2.0661 | 2.0360 / 2.0865 | 5.4229 | 38.10% | -2.60%（#383: 5.5679） |
| 2048 | 2.4791 | 2.4696 / 2.4902 | 12.5509 | 19.75% | +11.26%（#383: 11.2803） |
| 4096 | 2.5230 | 2.4852 / 2.5504 | 13.4365 | **18.78%** | +11.41%（#383: 12.0605） |

`f16_run3` の 512 形状（0.3730 TFLOPS）は中央値からの外れ値だが、判定対象形状（2048/4096）ではなく
512 は参考値のため候補下限値の算出には影響しない（生ログ `bench572/rust/f16_run3.log` に保持）。

判定対象形状は f32 と同じく M=N=K=2048・4096 の最小値。判定対象形状の最小比率 = 18.78%（4096）。

対 #383（`metal-f16-vs-mps-f16.md`。イシュー #387）比較（判定対象形状）: 分母（PyTorch）側は 2048 で
+11.26%・4096 で +11.41% 増加（M4 Max ローカル環境の PyTorch MPS 側性能向上）。分子（Metal）側は 2048
で 2.4426→2.4791（+1.5%）・4096 で 2.2411→2.5230（+12.6%）。結果として比自体（Metal/PyTorch）は
2048 が 21.6%→19.75%（-1.85pt）・4096 が 18.6%→18.78%（+0.18pt）と、4096 側はほぼ横ばい・2048 側は
わずかに悪化した。

候補下限値（参考算出。`bench_harness::floor_lower_bound` を用いる）: `floor_lower_bound(18.78%)` = **15%**

### 温度ドリフト注記

run 系列で単調減少傾向（例: `torch/f32_run{1,2,3}.log` の size=4096 が 12.6100→12.5838→12.0447）を
観測した。worst-case ペアリング（最遅 Metal ÷ 最速 PyTorch）でも候補下限値が変わらないことを確認する
ため以下を算出した:

| 形状 | worst-case 比（min Metal / max PyTorch） | 丸め後 floor |
|------|--------------------------------------------|--------------|
| f32 4096 | min(1.5645) / max(12.6100) = 12.41% | 10（不変） |
| f32 2048 | min(1.4484) / max(11.5636) = 12.53% | 10（不変。判定対象形状の最小ではないため参考） |
| f16 4096 | min(2.3919) / max(13.9954) = 17.09% | 15（不変） |

候補下限値（f32=10%・f16=15%）は観測された温度ドリフト（各ラン独立実行に伴う自然な性能ばらつき）に
対して頑健であることを確認した。

## REQ-8 下限値の扱い

**REQ-8 下限値（初期リリース 20%／最適化後 30%、f16 15%／未設定）は本ドキュメントでは変更しない。**
変更は F-5（#577・人間承認タスク）のみが行う。本ドキュメントは候補下限値の参考算出（上記
`bench_harness::floor_lower_bound` 欄）を提供するに留め、下限の最終確定・
`docs/spec/04-requirements.md` への反映判断は行わない（`docs/spec/` は本リポでは編集しない）。

現行値との比較（参考。最終判断は F-5）:

| 精度 | 現行 REQ-8 値（最適化後） | 本ドキュメントの候補下限値 |
|------|------------------------------|------------------------------|
| f32  | 30% | 10% |
| f16  | 未設定 | 15% |

f32 は現行 30% を候補下限値 10% が下回る（Metal f32 の計測境界を §4 準拠 prepared 入口へ揃えたことで
判明した実態値。「目的・受け入れ条件対応」節参照）。f16 は現行未設定のため、候補下限値 15% が
「自作カーネルでの f16 実測後に丸め規則で設定する」方針（`docs/perf/performance-floor-decision.md`
の Metal f16 行の申し送り）に沿った初の具体値となる。

## 状態: 実測完了（2026-08-18・Apple M4 Max）

本ドキュメントは当初 Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用でき
ないため計測手順・記録テンプレートのみを整備していた（#547 節・`metal-gemm-float4-staged-load.md` 先例
と同方式）。2026-08-18 に Mac 実機セッションで「計測手順」節の手順に沿って計測し、上記「f32 結果」
「f16 結果」「計測環境」「REQ-8 下限値の扱い」の各表・節を実測値で埋めた。

**#547 節（`docs/perf/metal-gemm-dynamic-tile.md`「Phase D 完了時点再計測」）の未計測テンプレートの
記入は本イシューのスコープ外**（close 済みイシューの記録）のため実施していない。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録済み）。

## 動作確認（実機セッションで実施済み）

- `cargo build --workspace --all-targets`
- `cargo build -p fandhe-ai-backend-metal --examples --release`（`gemm_f32_prepared_bench`・`gemm_f16_bench` の
  ビルド成立を確認。生ログ `bench572/build.log`）
- `cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture` — 80 テスト全 pass（0 failed。
  `dispatch_tiled_prepared_matches_dispatch_variant`・`cpu_metal_f16_parity` 系 6 件を含む。生ログ
  `bench572/parity_test.log`。「数値一致（parity）確認」節参照）
- `git diff 35514db..abaa94e -- crates/bench-harness/src/threshold.rs` および数値一致 tolerance 定数
  （`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・`docs/spec/`・`guardrail.toml` に差分がない
  ことを確認

## Appendix: 生ログ抜粋

実測実行時の生ログはセッション限定の scratchpad 配下（`bench572/`）にのみ存在しコミット対象外のため、
本節へ最小限を転記する。代表値として採用した run は size ごとに異なる（5 回独立実行した中央値を採用
する運用〈「計測手順」節〉のため。詳細は `bench572/aggregate.md`）。

### f32 代表 run（`bench572/rust/f32_run{2,3,4,2}.log`。512=run2・1024=run3・2048=run4・4096=run2）

```
size=512  metal_f32_simdgroup_tiled_tflops=0.5769 q1_tflops=0.5424 q3_tflops=0.7004 resolved_tile_config=TileConfig { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, staged: true }
size=1024 metal_f32_simdgroup_tiled_tflops=1.4650 q1_tflops=1.3483 q3_tflops=1.4947 resolved_tile_config=TileConfig { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, staged: true }
size=2048 metal_f32_simdgroup_tiled_tflops=1.4543 q1_tflops=1.4472 q3_tflops=1.4685 resolved_tile_config=TileConfig { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, staged: true }
size=4096 metal_f32_simdgroup_tiled_tflops=1.5666 q1_tflops=1.5601 q3_tflops=1.5731 resolved_tile_config=TileConfig { bm: 64, bn: 64, bk: 16, wm: 2, wn: 2, staged: true }
```

### f16 代表 run（`bench572/rust/f16_run{2,3,1,3}.log`。512=run2・1024=run3・2048=run1・4096=run3）

```
size=512  metal_f16_simdgroup_tflops=0.8749 q1_tflops=0.8520 q3_tflops=0.8918
size=1024 metal_f16_simdgroup_tflops=2.0661 q1_tflops=2.0360 q3_tflops=2.0865
size=2048 metal_f16_simdgroup_tflops=2.4791 q1_tflops=2.4696 q3_tflops=2.4902
size=4096 metal_f16_simdgroup_tflops=2.5230 q1_tflops=2.4852 q3_tflops=2.5504
```

### PyTorch MPS 参照値（各 5 回独立実行の中央値。`bench572/torch/{f32,f16}_run{1..5}.log`）

```
torch=2.13.0 device=mps
f32: size=512 median=0.8221  size=1024 median=5.3128  size=2048 median=10.0123 size=4096 median=12.0447
f16: size=512 median=0.9755  size=1024 median=5.4229  size=2048 median=12.5509 size=4096 median=13.4365
```

### parity 実行結果（`bench572/parity_test.log`）

- `cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture`: 80 テスト全 pass（0 failed）
- `dispatch_tiled_prepared_matches_dispatch_variant`（`tests/gemm_dynamic_tile_parity.rs`）: pass
- `cpu_metal_f16_parity.rs`: 6 テスト全 pass

## 未実施・後続作業

- **実機実測**: 「状態」節のとおり 2026-08-18 実測完了。本節は完了扱い
- **候補下限値の最終確定・REQ-8 反映判断**: F-5（#577・人間承認）が本ドキュメントの実測結果
  （f32 候補 10%・f16 候補 15%）を受けて対応する
- **#547 節の未計測テンプレート記入**: 本イシューのスコープ外のため引き続き未実施（「状態」節参照）
