# Metal 線形代数（LU 系・cholesky・QR。#2676・#3137）実機 parity の記録

**状態（2026-10-28 更新）**: 所有者判断 (a) に従い `qr_input(3,8)` を是正して Mac mini で再実測し、**5/5 run で全件 pass**（末尾「2026-10-28 入力是正と再実測」節。2026-10-27 の fail は元入力のもので、記録として残す）。以下は 2026-10-27 時点の状態: Mac mini（Apple M4）で事前登録 `RULE.txt` どおり実測し、**5 回中 5 回とも `qr_matches_cpu_on_device` が fail（要対応・所有者判断待ち）**。LU 系・cholesky は pass。詳細は末尾「2026-10-27 Mac mini 実測」節。

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。Stage A（#2676）は `unsafe` なしまでで止め、Stage B（#3137）で
案 A（既存ヘルパーの共有モジュールへの移設。正味の新規 `unsafe` ブロックなし。`docs/gpu-linalg-kernel-launch-unsafe-audit.md` §3）により起動を結線した。
Metal の eigh は決定記録 §2 のとおり対象外（trait 既定のまま）。

## 実装済み（Stage A）

- `crates/backend-metal/src/shaders/linalg.metal`: `linalg_lu_f32`・`linalg_lu_solve_f32`・`linalg_cholesky_f32`・
  `linalg_qr_f32`（1 行列 = 1 threadgroup・`#pragma clang fp contract(off)`・`precise::sqrt`／`precise::divide`・
  `fma(` 不使用・QR は scale/ssq ノルム・手動境界チェック維持・早期 return でバリアを不一致にしない）。
- `crates/backend-metal/src/linalg.rs`: ゲート（既定 OFF・facade 非公開）、サイズ検証（`u32` 添字・確保上限）、ホスト後処理
  （`lu_diag_product`／`permutation_sign`／`qr_finalize`。`backend-cpu` の複製）、`scale_ssq_norm_f32`。
- `crates/backend-metal/src/linalg_model.rs`: 上記 4 カーネルと演算順序が同一の f32 ホスト逐語モデル。
- `crates/backend-metal/src/linalg_kernel.rs`: MSL ソース保持と `MetalLinalg::new`（パイプライン構築。safe API のみ）。
- `ops.rs`: `linalg_inv`／`solve`／`det`／`cholesky`／`qr` を置換。ゲート OFF は従来どおり `Unsupported`。ゲート ON は
  rank・正方性・サイズ検証（デバイス非接触）→ 空入力はホストで CPU と同じ形 → それ以外は fail-closed の `Unsupported`。
- テスト: `tests/linalg_model_parity.rs`（f32 モデル対 CPU 参照を `assert_parity` で突合。Linux 実行可）、
  `tests/linalg_source_evidence.rs`（MSL 契約トークンの固定）、`tests/linalg_parity.rs`（ゲート・形状・空入力。実機は `#[ignore]`）。

## 実装済み（Stage B。#3137）

- `crates/backend-metal/src/encoder_args.rs`: `setBuffer`／`setBytes` の共有ヘルパー（案 A。`rnn_cell.rs`・`sort.rs` から移設）。
- `linalg_kernel.rs`: `run_lu_solve_f32`・`run_lu_det_f32`・`run_cholesky_f32`・`run_qr_f32`（`dispatch_sync` 固定・
  `status` は 0 初期化・因数分解は単一 threadgroup・幅は `maxTotalThreadsPerThreadgroup` で clamp・`k == 0` は solve を encode しない）。
- `linalg.rs`: `decode_lu_status`・`decode_cholesky_status`・`det_from_lu`・`clamp_tg_width`・`validate_launch_stride_headroom`（Linux で単体テスト）。
- `context_cache::cached_linalg`、`ops.rs` の 5 演算を実起動へ置換。エラー写像: solve／inv のピボット 0 は `InvalidArgument`、
  det のピボット 0 は `0.0`、非正定値は `InvalidArgument`、デバイス失敗は `map_metal_error`。
- 実機の数値一致（`--ignored` 全件）は Stage B 時点では未実測だった（2026-10-27 に Mac mini で実測。末尾節を参照）。計測の事前登録・入力集合・記録は `docs/perf/logs/metal-linalg-parity-3139/`（#3139）。
  #3139 で入力集合を是正した（QR 17×17 を κ₂ ≤ 1e3 の入力へ差し替え・det は n ≤ 17・QR に 1e±20 倍と `1e20 × I₂` を追加。tolerance は不変）。

## 測定コマンド（実機）

```sh
# MSL コンパイルとパイプライン構築（Stage A でも検証可能）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture pipeline_compiles_on_device
# parity（Stage B 実装済み。実測は #3139）
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | Apple M4 Max | `pipeline_compiles_on_device` | pass | MSL 4 カーネルのコンパイルとパイプライン構築のみ（起動なし） |
| 未実測 | Apple M4 Max | parity（`--ignored` 全件） | 未実測 | #3139 で Mac mini に置き換え |
| 未実測（2026-10-16 時点） | Mac mini（Apple M4） | parity（`--ignored` 全件） | 未実測 | 実機に到達できず未実行。事前登録 `metal-linalg-parity-3139/RULE.txt`。Mac 側での実行を申し送り |
| 2026-10-27 | Mac mini `<metal-node>`（Apple M4） | `RULE.txt` のコマンド × 5 run | **fail（5/5 run で同一）** | 各 run `running 6 tests`・5 passed／1 failed・終了コード 101。fail は `qr_matches_cpu_on_device` のみ。sha・環境は下記 |
| 2026-10-27 | Mac mini `<metal-node>`（Apple M4） | #3139 `RULE.txt` 改訂のコマンド × 5 run（`running 7 tests`） | **fail（5/5 run で同一）** | `qr_matches_cpu_on_device` のみ fail（#2676 と同一）。コミット `95a18c2d3`。GPU は f32 ホストモデルとビット一致・未到達ケースの診断値は `../metal-linalg-parity-3139/README.md` |
| 2026-10-28 | Mac mini `<metal-node>`（Apple M4） | RULE.txt 改訂のコマンド × 5 run（`running 7 tests`） | **pass（5/5 run）** | `qr_input(3,8)` 是正後。各 run 7 passed／0 failed・終了コード 0。コミット `2e30c15b2c7016afde3e4b69686187dcd9bd259b`。生ログ `rerun-2026-10-28/`。#3139（Stage C）の閉じる根拠を兼ねる |

## 2026-10-27 Mac mini 実測（イシュー #2676）

- 事前登録: `RULE.txt`（実測前にコミット `1184586ee0ae1f76aaf0f80dbdffbe53067010eb` で固定。判定方式・入力集合は `../metal-linalg-parity-3139/RULE.txt` を採用し、対象コミット条件のみ差し替え。#3139 の README・RULE は登録が異なるため更新していない）
- 実測した展開ツリー: `rev-stamp.txt`（1 行目がフル SHA `1184586ee0ae1f76aaf0f80dbdffbe53067010eb`・2 行目 `dirty=0`）
- 環境: `env.txt`（Apple M4・macOS 27.0.1・rustc 1.99.0・cargo 1.99.0）。ホスト名・ユーザー名・パスは `<metal-node>`・`<home>` にマスク
- 実機ロック: 取得 2026-10-27T10:18:11Z／解放 2026-10-27T10:18:59Z（ノード時計の UTC。RULE.txt の宣言時刻はコミット前の見込みで、先後関係の正はコミット `1184586ee` が転送・実測より前に作られている事実）
- 生ログ: `run1.log`〜`run5.log`（末尾に `exit_code=`）

### 結果

| テスト | 判定対象 | run1〜5 |
|---|---|---|
| `pipeline_compiles_on_device` | ○ | ok |
| `cholesky_matches_cpu_on_device` | ○ | ok |
| `lu_family_matches_cpu_on_device` | ○（inv・solve・det） | ok |
| `qr_matches_cpu_on_device` | ○ | **FAILED（5 run とも同一メッセージ）** |
| `status_mapping_end_to_end_on_device` | 記録のみ | ok |
| `gate_on_non_empty_does_not_return_unsupported` | 記録のみ | ok |

fail の内容（5 run 全て同一。決定的）:

```
qr R 3x8 scale=1e20: 複合判定 FAIL（fail_count=1/24, max_abs_diff=3.738e13, max_rel_err=1.000e0,
mean_abs_diff=8.384e12, mean_rel_err=4.167e-2, p50_abs_diff=4.398e12, p99_abs_diff=3.738e13, p999_abs_diff=3.738e13）
```

- テストは最初の `assert_parity` 失敗で打ち切られる。形状 (8,3) の全倍率と (3,8) の等倍までは通過し、(3,8) の 1e20 倍で fail。**以降は未到達のため未実測**: (3,8) の 1e-20 倍、(17,17) の全倍率、`1e20 × I₂`、「R・Q が有限」「R の対角が 0 でない」「倍率で正規化した R の parity」（後者の NaN・0 検査は fail した倍率のケースでは実行されていない）。
- 判定は `RULE.txt` に従い **fail**。tolerance・入力集合・カーネル・判定方式・baseline は変えていない。再実行で結果を選んでいない。M4 Max の値は使っていない。
- 本 issue の閉じる条件（実機 pass）は**未達**。`Closes #2676` は書かない。

### 要対応（所有者判断待ち）

- 仮説（未検証）: R は上三角で 3x8 の下三角側は理論上 0 の要素を含み、CPU 参照と GPU のどちらかが丸め残差（1e20 倍で約 3.7e13、スケール比 約 3.7e-7）を持ち、もう一方が厳密に 0 となる。`max_rel_err=1.0` は片側が 0 の形と整合する。1e20 倍では絶対救済（1e-5）が効かず相対誤差だけが判定になるため、本来は 0 のはずの要素が fail として現れる可能性がある。原因が (a) カーネルの Householder 適用順序・scale/ssq ノルムの丸め、(b) 入力集合・判定（0 要素の扱い）のどちらにあるかは、fail 要素の値を出力して切り分ける必要がある。
- 是正方針の候補: fail 要素の got／cpu 値を診断出力して原因を確定 → カーネル側の是正、または判定契約の見直し（後者は tolerance 変更に当たるため所有者承認が必要）。f32 ホスト逐語モデル（`linalg_model_parity.rs`）が同入力で CPU と parity するかも突き合わせる。
- Issue の起票・コメント、baseline 化は承認なしでは行っていない。

## 2026-10-28 入力是正と再実測（所有者判断 (a)）

- 所有者判断: 2026-10-28「入力の是正」（#2676 コメント 6445945182・#3139 コメント 6445947161）。#2676 と #3139 は同じ作業で、実装は本件の PR で行い #3139 は Refs で参照する。
- 変更: `crates/backend-metal/tests/linalg_parity_inputs.rs` の `qr_input(3,8)` だけ主対角へ `+2.0`（`QR_WIDE_DIAG_SHIFT`）。(8,3)・(17,17)・`1e20 × I₂`・LU・cholesky の入力、tolerance 定数、判定式、カーネル、CPU 参照は不変。baseline 方式へは移っていない。経緯と退けた候補は `docs/backend-gpu-linalg-decision.md` §5「QR 入力 (3,8) の是正」。
- 原因（2026-10-27 の fail。既存 `run1.log`〜`run5.log` に残す）: 旧 (3,8) は第 0 列と第 2 列が偶然直交し `R[0,2]` の真値が厳密 0。1e20 倍で f32（GPU・ホストモデル。ビット一致）と f64（CPU 参照）の丸め残差の差が相対誤差 1.0 として露出した。カーネルのずれではない。
- 事前登録: `RULE.txt` の「改訂（2026-10-28）」を実測前にコミット `2e30c15b2c7016afde3e4b69686187dcd9bd259b` で固定。実測ツリーはそのコミットのアーカイブ展開で、`rerun-2026-10-28/rev-stamp.txt` はフル SHA と `dirty=0`。
- 正誤（2026-10-29 追記。`RULE.txt` 本体は事前登録の記録なので書き換えない）: `RULE.txt` の改訂の「宣言日時（UTC）: 2026-10-28T06:00Z」は記入時の誤り。改訂を固定したコミット `2e30c15b2` の実際の時刻は **2026-10-27T18:56:25Z**（PR #3665 のブランチ上のコミット。main の squash コミット `963a5c0be` の時刻ではない）。実測より前に固定した事実は変わらない。根拠は、実測ツリーが `2e30c15b2` のアーカイブ展開であること（`rerun-2026-10-28/rev-stamp.txt`）。生ログには時刻が残っていない。
- 環境: `rerun-2026-10-28/env.txt`（Apple M4・macOS 27.0.1・rustc 1.99.0・cargo 1.99.0）。ホスト名・ユーザー名・パスは `<metal-node>`・`<home>`・`<user>` にマスク。実機ロック `~/work/.fandhe-hw-lock` は実測前に取得し実測後に解放した。
- 生ログ: `rerun-2026-10-28/run1.log`〜`run5.log`（末尾 `rc=`）。

### 結果

| テスト | 判定対象 | run1〜5 |
|---|---|---|
| `pipeline_compiles_on_device` | ○ | ok |
| `cholesky_matches_cpu_on_device` | ○ | ok |
| `lu_family_matches_cpu_on_device` | ○ | ok |
| `qr_matches_cpu_on_device` | ○ | **ok（5 run とも。旧未到達ケースを含む全ケース）** |
| `status_mapping_end_to_end_on_device`・`gate_on_non_empty_does_not_return_unsupported`・`qr_diagnose_all_cases_on_device` | 記録のみ | ok |

- 各 run: `running 7 tests`・`7 passed; 0 failed`・`rc=0`。判定は `RULE.txt` 改訂どおり 5 run すべてで満たし pass。再実行で結果を選んでおらず、ケースも除いていない。
- 旧未到達ケース（(3,8)×1e-20、(17,17) の全倍率、`1e20 × I₂`、R・Q の有限性、R 対角が 0 でないこと、倍率で正規化した R の parity）は `qr_matches_cpu_on_device` の assert が通ったことで判定済み。1e±20 倍の QR が NaN／0 を返さないことも同 assert の通過が根拠。
- 診断出力（記録のみ）: 全 `gpu-vs-cpu`・`model-vs-cpu` が `fail_count=0`、`gpu-vs-model` は `max_abs_diff=0`（ビット一致）。(3,8)×1e20 の R は `max_rel_err=5.97e-7`、(17,17)×1e20 の R は `max_rel_err=5.79e-6`。
- 事前検証（実機の根拠にはしない）: 構造的 0 ガード `qr_inputs_have_no_structural_zero_in_r` の `|R_ij|/‖a_j‖` 最小値は (8,3) が 3.75e-1、(3,8) が 6.41e-2（旧入力は約 1e-17 で検出）、(17,17) が 4.50e-4。f32 モデル対 CPU の常時テスト `qr_parity_inputs_model_matches_cpu` も pass。
- 本 issue の閉じる条件（実機 pass）を満たした。
