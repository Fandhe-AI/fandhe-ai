# Metal GEMM: candle／MLX steel 解析差分からの未試行候補（イシュー #2110）

`gemm_simdgroup_tiled`（f32・NN 主対象）に対する、candle 0.11.0 と MLX v0.32.2 の steel GEMM 読み取り解析
（`docs/analysis/candle-metal-01.md`〈#2090〉・`docs/analysis/mlx-v0.32.2-steel-gemm.md`〈#2096〉）が
示した差分のうち、**過去に kernel_gpu 5 run A/B が行われていない組合せ**を opt-in で実装した記録。
実機実測・結線判断は #2111 のスコープで、**本 doc 時点では実機未実測**（Linux 実装環境。型検査は
`make check-cross-metal-tests`、shader は証跡テストのみ）。

- 本番既定（`MetalGemm::new`・`tile::select*`・`dispatch_auto`・`UNROLL_LOAD_ENABLED=false`）は不変
- tolerance・parity baseline・依存・`docs/spec` は不変。新規 `unsafe`・公開 API 追加なし
  （構築入口 `MetalGemm::new_with_steel_candidate` は `#[cfg(test)] pub(crate)`）

## 1. 背景

スコアボード（2026-09-19 版）で M4 Max の Metal GEMM は正方 N=256〜4096 の 5 セルが candle に負けている
（0.59〜0.93×。`docs/perf/loss-attribution-matrix.md` M-MTL-G*）。解析の結論は「差の主因はタイル選択ではなく
カーネル本体（unroll・ロード方式・同期）」で、candle-metal-01 §6 が未試行の差分候補を 1〜4 として挙げた。

## 2. 候補 arm（事前登録）

| arm | 構成 | 未試行である根拠 |
|---|---|---|
| `base` | 本番選択 `tile::select_for_device` と全ゲート既定（classic 経路） | 基準 |
| `T0U` | candle 相当タイル選択（下記）+ `UNROLL_ACC` instance フラグ ON（acc 積 16 以上のタイルにのみ実効） | candle は N>=1024 の正方 f32 NN で常に `TILE_64_64_16_2_2`（= `CANDIDATES[0]`）+ full unroll（candle-metal-01 §3.2・§5）。既存計測はこの組合せを本番選択と直接比べていない: E1 §7.3/7.4 は別プロトコル、#1284 §7.10.1 の正方 4 形状は構造上 cand0 に到達せず、E7/E8 は unroll ゲートが無効のまま計測 |
| `LU` | 本番選択タイル + 新軸 `UNROLL_LOAD`（協調ロード `vi` ループの固定反復数化 + full unroll） | E1／`UNROLL_ACC_ENABLED` の対象はアキュムレータ系 10 ループのみで協調ロードの 4 ループは未着手。candle は BlockLoader の読み出しに `STEEL_PRAGMA_UNROLL`（candle-metal-01 §5・§6 候補 1・2） |
| `T0U-LU` | `T0U` + `LU` | 組合せ未試行 |
| `T0U-LU-FB` | `T0U-LU` + `FINE_BARRIER` | candle の fragment ロードと MMA の間の `simdgroup_barrier(mem_none)` の**近似**（粒度は厳密一致しない可能性。§6）。FB 単体は #1278 で undetermined のため単体 arm にしない |

### candle 相当タイル選択（`tile::select_candle_equivalent`、`#[cfg(test)]`）

candle 0.11.0 `select_tile_config` の f32・NN・batch=1 部分を純関数化（コードは持ち込まない）:
`m < 16` → `CANDIDATES[3]`、`m*n >= 2^20` → `CANDIDATES[0]`、それ以外 → `CANDIDATES[5]`。
**M4 Max が candle の Max／Medium／Ultra 区分に解決される仮定**（未確認。candle-metal-01 §3.2・§8）で、
影響は N=512 のみ（N>=1024 はデバイス区分に依らず `CANDIDATES[0]`）。

## 3. 実装（`UNROLL_LOAD_ENABLED`、function constant index 18）

- `shaders/gemm.metal`: `gemm_simdgroup_tiled` staged 経路の協調ロード 4 ブロック（A-NN／A-T／B-NN／B-T）を
  `if (UNROLL_LOAD_ENABLED) { <固定反復数 + unroll(full)> } else { <現行コードと本体同一（字下げのみ差。証跡テストが needle で固定）> }` に複製
  （#1282 E1 と同方式）。unroll 版は `it < ceil(vecs / threads_total)` の固定反復で `vi = local_tid + it*threads_total`、
  部分反復は `if (vi < *_vecs)` ガード（cand6 の B 等）。
- `pipeline.rs`／`spec_source.rs`: index 18 の設定と `GEMM_SPEC_UNROLL_LOAD_ENABLED`。
  f16／hfrag／te／split-K パス 1 は参照しない no-op 契約（ホスト側は常に `false`）。
- **bit 一致の論拠**: 各スレッドが担当する float4 グループの集合（`vi` の像）と共有メモリへ書く値・位置は
  ループ形状に依らず不変。`threadgroup_barrier` 以降のフラグメントロードと MMA オペランド列も不変
  （#536/#538/#1282/#1298 と同型）。
- **REQ-8**: ループ本体（`*_group_in_bounds` 判定・スカラー 0 埋めフォールバック）は両 variant で同一。
  整列可否による分岐ロードは行わない。`tests/shader_source_evidence.rs` が両 variant の維持を固定。

## 4. 除外した候補（#2111 が逸脱しないための記録）

| 候補 | 除外理由 |
|---|---|
| acc ループの無条件 unroll（acc 積 8 以下にも適用） | §7.6a で本番 N=512〜2048 が 15〜35% 後退し撤回済み（`T0U` は acc 積 16 以上のみ実効） |
| align_M/N/K 分岐ロード | `docs/backend-metal-aligned-load-decision.md`（#808）で不採用確定・REQ-8。`UNROLL_LOAD` は整列可否で分岐しない |
| async copy（非公開 AIR intrinsic） | #546 で不採用 |
| tgid swizzle | candle も `swizzle_log=0` で無効（candle-metal-01 §4.1）、#1279 で判定不可 |
| smem XOR swizzle + E3 フラグメントロード | 本番のフラグメントロードが `tgp-k1` で #1970 の実測 arm と同じ |
| `tgp-k2` を含む組合せ | E3 §10.4 で N=4096 が 2.19 倍後退 |
| MLX Ultra 分岐・NAX | 実機なし・#549 の再訪条件 |
| kk ループの unroll | acc 系 unroll ブロックが二重化し既存 pragma 証跡テストの前提が崩れるためスコープ外 |

## 5. 自己検証

Linux で実行済み: `tile`／`spec_source` の単体テスト、`shader_source_evidence`（index 18・複製数・REQ-8・no-op 契約）、
`cargo check --tests --target aarch64-apple-darwin`。
実機 `#[ignore]`（`gemm_steel_candidate_diag_tests`）: `unroll_load_on_off_bit_match_all_candidates`／`_dispatch_auto`／
`_transposed`（assert）、`steel_candidate_arms_match_cpu_reference`（REQ-2 `assert_parity`。tolerance 不変）、
`unroll_load_effective_gate_is_forwarded`、`steel_candidate_kernel_gpu_ab_production_sizes`（記録のみ）。
タイル形状が異なる arm 同士の bit 一致は E7 §13.4 と同様に契約外で、assert せず `aggregate.py` が記録・判定する。

## 6. 実測記入欄（#2111 が埋める）

判定規則は実測前に固定済み（`docs/perf/logs/metal-gemm-candidate-ab-2111/RULE.txt`）。

| arm | 判定 | N=512 | N=1024 | N=2048 | N=4096 | 備考 |
|---|---|---|---|---|---|---|
| T0U | 未実測 | | | | | |
| LU | 未実測 | | | | | |
| T0U-LU | 未実測 | | | | | |
| T0U-LU-FB | 未実測 | | | | | |

## 7. スコープ外

E7（`CANDIDATES[9]`）・E8（`[10]`）は acc 積 16 以上なのに unroll ゲート無効のまま計測されており、§14.3 の
4.5〜7.6 倍後退は交絡の可能性がある（candle／MLX 由来ではないため本 issue 外）。ほか kk ループ unroll・
ロード先アドレスの K ループ外事前計算・f16／hfrag／te／split-K への `UNROLL_LOAD` 展開・NT/TN/TT の性能 A/B・
M4 Max の `MTLDevice.architecture.name` 確認・MLX Ultra 分岐／NAX split-K。
