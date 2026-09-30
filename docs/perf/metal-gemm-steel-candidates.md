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
タイル形状が異なる arm 同士の bit 一致は E7 §13.4 と同様に契約外で、assert せず記録のみとする。`aggregate.py` は `same_tile=true`（base と同一タイル）の cell のみ bit 一致を採用可否へ反映し、run 間の出力一致は全要素 f32 ビット列の FNV-1a 64bit ハッシュ（`hash=`。checksum＝f64 和は参考値）の一致として全 cell で要求する。前提ゲート（`gate_run.log`）の全件成功は `orchestrate.sh` の run 起動前と `aggregate.py` の双方が機械検証し、不成立なら計測・判定を拒否する。

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

## 8. 結線手順書（イシュー #2111・判定結果別の条件付き手順）

本節は `aggregate.py` の arm 判定を入口とする手順書であり、**本番コード・既定値は #2111 の Linux 側実装では変更していない**。
判定規則の正は `docs/perf/logs/metal-gemm-candidate-ab-2111/RULE.txt`（実測前固定。事後に緩和しない）。

### 8.1 全 arm が REJECT／UNDETERMINED／NOT_ADOPTABLE／INCOMPLETE の場合

結線しない。§6 に判定と中央値を記入し、`aggregate.md` と生ログを収録する PR で完了とする（本番コードは不変）。
`REFERENCE_ONLY` の系列は採用根拠にしない（RULE.txt 7.）。

### 8.2 `LU` が ADOPT_CANDIDATE の場合（定数 1 個の切替）

1. **結線前にユーザー承認を得る**（RULE.txt 10.）。
2. `tile::UNROLL_LOAD_ENABLED`（`crates/backend-metal/src/tile.rs`）を `true` にする。`MetalGemm::new` 系はこの定数を渡すため `dispatch_auto` にも伝播する。f16／hfrag／te／split-K パス 1 はホスト側で常に `false` とする no-op 契約のため影響しない。
3. 既定値ドリフトテスト（`tile.rs` の既定 `false` 固定テスト、`STEEL_ARMS[0].unroll_load == UNROLL_LOAD_ENABLED` の前提、`spec_source.rs` の既定 `false` 期待値）を「承認済み既定 `true`」へ整合させる。
4. 注意: 本番 `dispatch_auto` は計測した正方 4 形状以外（非正方・小形状・境界形状）にも同じ定数で効く。bit 一致は `unroll_load_on_off_bit_match_all_candidates`／`_dispatch_auto`／`_transposed` が担保するが、**性能は正方 4 形状しか計測していない**。承認時に「非正方代表形状で非後退を確認する」か「正方に限定するスコープ付きゲートにする」かをユーザーへ提示する。
5. 結線後は Mac で gate の 4 テストと Metal parity テスト（split-K baseline 含む）を `--ignored` で再実行し、framework-compare gemm metal の前後 A/B を収録する（`metal-gemm-n4096-kernel-gap.md` §19 と同型）。

### 8.3 `T0U`／`T0U-LU`／`T0U-LU-FB` が ADOPT_CANDIDATE の場合（別設計 PR が必要）

- `select_candle_equivalent` と `STEEL_ARMS` は `#[cfg(test)]` 限定で、`UNROLL_ACC_ENABLED` はグローバル定数（acc 積 16 以上の候補すべてに効く）。無条件 unroll は §7.6a で N=512〜2048 が 15〜35% 後退し撤回済みのため、定数の切替では結線できない。
- 形状スコープ付きの選択機構（計測済み正方 N のみ対象・CANDIDATES[0] 選択時だけ unroll_acc を実効化）が必要で、設計判断となる。ユーザー承認と別イシュー／別 PR で扱う（`select_for_device` の本番化・`unroll_acc_loops_for` との整合・既存タイル選択テストの改修を含む）。
- **N=512 は結線対象から外す（保留）**。RULE.txt 9. の candle デバイス区分仮定に依存するため、`env_info.txt` の `mtl_architecture_name` で M4 Max の区分が確認されるまで結線しない。N>=1024 は区分に依らない。
- `T0U-LU-FB` の FINE_BARRIER は candle の近似で、FB 単体は #1278 で undetermined。ADOPT でも FB 軸だけの寄与は分離していないことを記録する。

### 8.4 複数 arm が ADOPT の場合

事前登録にない新規則で採用 arm を選ばない。候補と中央値をユーザーへ提示し、選択は承認事項とする。
いずれの場合も tolerance・parity baseline・`Cargo.toml` は不変。

## 9. 実測状況・申し送り（イシュー #2111）

**状況: 未実測。** #2111 の実装担当ホストは Linux で Apple M4 Max へ到達できず、計測もユーザー承認も取れないため、
§6 の記入欄・`env_info.txt` の値・`aggregate.md` には一切値を入れていない。

Linux で実施済みの検証: `aggregate.py --self-test`（OK）、`orchestrate.sh` の `gate`／`1`／`5` の `--dry-run`（成功・ログ非生成）、
不正引数（`6`・`gate --foo`）の拒否、`cargo test -p fandhe-ai-backend-metal --lib`（550 件 pass）・`--test shader_source_evidence`（59 件 pass）。

Mac セッションで実施する手順:

1. `main` を最新にし `env_info.txt` を記入する（chip・gpu_cores・macOS・rustc・git_commit・date_utc・`mtl_architecture_name`）。専有できない場合は **run 1 の前に** `load_policy: record_only` と理由を宣言する（RULE.txt 7.）。
2. `./orchestrate.sh gate` — 1 件でも FAIL なら REJECT を確定し A/B は実施しない。
3. `./orchestrate.sh 1` 〜 `5`（差し替え・追加・選別はしない）。
4. `python3 aggregate.py` の出力を `aggregate.md` に保存する。
5. ログ中のホスト名・ユーザー名・絶対パスを `<home>` 等へマスクする。
6. §6 を記入し実測 PR を作る。
7. ADOPT_CANDIDATE があれば §8 に従い、**ユーザー承認後**に結線 PR（別 PR）を作る。
8. 両 PR のマージ後に #2111 をクローズする。
