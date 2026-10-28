# Metal 線形代数（LU 系・cholesky・QR）実機 parity 計測ログ（イシュー #3139・Stage C）

事前登録規則は同ディレクトリの `RULE.txt`（元の本文 + 実機実行前の「改訂（2026-10-27）」節）。判定方式は `docs/backend-gpu-linalg-decision.md` §5
（Metal f32 対 CPU f64 は精度クラスの差なので baseline では吸収せず、入力集合を固定して `assert_parity` を fail 0 件で
厳密に適用する）。親は #3137（Stage B・起動の結線）。

## 結果

**状態（2026-10-27）**: Mac mini（`<metal-node>`・Apple M4）で改訂 `RULE.txt` どおり実測し、**5 回中 5 回とも `qr_matches_cpu_on_device` が fail（判定は fail。要対応・所有者判断待ち）**。
LU 系・cholesky・`pipeline_compiles_on_device` は pass。本 issue の閉じる条件（実機 pass）は**未達**で、PR は `Refs #3139`（`Closes` は書かない）。

- 事前登録: 実測前にコミット `95a18c2d324a0d5271e1b9e4c84829bc51d08de9`（診断テスト追加 + `RULE.txt` 改訂）で固定
- 実測した展開ツリー: `rev-stamp.txt`（1 行目がフル SHA `95a18c2d324a0d5271e1b9e4c84829bc51d08de9`・2 行目 `dirty=0`。転送後にノード上へ直接書いた）
- 環境: `env.txt`（Apple M4・macOS 27.0.1・rustc 1.99.0・cargo 1.99.0）。ホスト名・ユーザー名・パスは `<metal-node>`・`<home>` にマスク
- 実機ロック: 取得 2026-10-27T11:46:29Z／解放 2026-10-27T11:47:19Z（ノード時計の UTC）
- 生ログ: `run1.log`〜`run5.log`（各 `running 7 tests`・末尾に `exit_code=101`）。診断出力の抜粋は `diag.log`（run1 の `DIAG` 行。5 run の `DIAG` 行は md5 が一致＝完全に同一）。f32 ホストモデル対 CPU のデバイス不要診断は `model-diag.log`
- 手順上の訂正: run2・run3 を最初に誤って同時に起動したため、その 2 本は破棄して順次で取り直した（両方とも同じ結果）。以降 run4・run5 も順次。再実行で結果を選んではいない

### 判定結果（run1〜5 共通）

| テスト | 判定対象 | run1〜5 |
|---|---|---|
| `pipeline_compiles_on_device` | ○ | ok |
| `cholesky_matches_cpu_on_device` | ○ | ok |
| `lu_family_matches_cpu_on_device` | ○（inv・solve・det） | ok |
| `qr_matches_cpu_on_device` | ○ | **FAILED（5 run とも同一メッセージ）** |
| `status_mapping_end_to_end_on_device` | 記録のみ | ok |
| `gate_on_non_empty_does_not_return_unsupported` | 記録のみ | ok |
| `qr_diagnose_all_cases_on_device`（診断。assert しない） | 記録のみ | ok |

fail の内容（`run1.log` 79 行。5 run 同一・決定的）:

```
qr R 3x8 scale=1e20: 複合判定 FAIL（fail_count=1/24, max_abs_diff=3.738e13, max_rel_err=1.000e0,
mean_abs_diff=8.384e12, mean_rel_err=4.167e-2, p50_abs_diff=4.398e12, p99_abs_diff=3.738e13, p999_abs_diff=3.738e13）
```

これは #2676 の 2026-10-27 実測（コミット `1184586ee`）の fail と同一の内容である。tolerance・入力集合・カーネル・判定方式・baseline は変えていない。M4 Max・Linux の値は判定に使っていない。

### 判定テストが途中で止まり未到達だったケースの記録（診断テストによる記録値）

`qr_matches_cpu_on_device` は (3,8) の 1e20 倍で止まり、以降の assert には**判定としては到達していない**（未到達）。
下表は記録専用の `qr_diagnose_all_cases_on_device`（assert しない）が同じ入力を実行して出した値で、**判定の pass の根拠にはしない**。

| ケース | R の CPU 対 GPU | R と Q が有限 | R 対角が 0 でない | 倍率で正規化した R の CPU 対 GPU |
|---|---|---|---|---|
| 3x8 scale=1e-20（判定として未到達） | fail 0/24（max_abs 3.2e-27） | true | true | fail 0/24 |
| 17x17 scale=1e0・1e20・1e-20（同上） | fail 0/289（3 倍率とも） | true | true | fail 0/289（1e±20） |
| `1e20 × I₂`（同上） | fail 0/4・max_abs 0 | true | true | fail 0/4 |
| 8x3 全倍率（判定として到達・pass） | fail 0/9 | true | true | fail 0/9 |
| 3x8 scale=1e0（判定として到達・pass） | fail 0/24 | true | true | （等倍は対象外） |
| 3x8 scale=1e20（判定で fail） | **fail 1/24**（`[0,2]` のみ） | true | true | fail 0/24 |

1e±20 倍と `1e20 × I₂` で NaN・0 への潰れは見られなかった（`diag.log` の `finite=true diag_nonzero=true`）。

### fail 要素と原因の判別

fail した要素は `R[0,2]` ただ 1 つで、値は GPU＝-4.3980465e12、CPU 参照（f64）＝-1.6384e4（`diag.log` の `gpu-vs-cpu FAIL [0,2]`）。
倍率で割り戻すと GPU 側は約 -4.4e-8、CPU 側は約 -1.6e-16 であり、等倍の `qr_input(3,8)` は第 0 列と第 2 列が直交するため `R[0,2]` の真値は厳密に 0 である。
つまり**どちらの値も 0 に対する丸め残差**で、f32 は約 1e-7 相対、f64 は約 1e-16 相対の残差になる。等倍なら残差が絶対誤差の救済（1e-5）に収まって pass し、
1e20 倍では救済が効かず相対誤差だけが判定になり `max_rel_err=1.0` で fail する。

判別（RULE.txt の事前宣言どおり）:

- **GPU 対 f32 ホストモデル**は全ケース・R の全要素で `fail_count=0`・`max_abs_diff=0`（`diag.log` の `gpu-vs-model`）。GPU はホストモデルとビット一致しており、**カーネルがモデルからずれている証拠はない**
- **f32 ホストモデル対 CPU**（デバイス不要。`model-diag.log`）が同じ要素 `[0,2]` で同値（model=-4.3980465e12・cpu=-1.6384e4）の不一致を出す
- 以上から、原因は f32 と f64 の精度クラスの差が、構造的に 0 になる要素と 1e20 倍の組合せで相対判定に露出したものと判断する（コード修正で消える種類の不一致ではない可能性が高い）。この判断は上記の 5 run と model-diag の突合に基づくが、カーネル側の別の是正余地がないことの証明ではない
- 既存の `linalg_model_parity.rs` の QR テストは Xorshift 乱数の入力を使い `qr_input` を使わないため、この事象を検出していなかった

### 要対応（所有者判断待ち。いずれも承認必須で、本 PR では実施していない）

- (a) 入力集合の是正: `qr_input(3,8)` の偶発的な直交構造（`R[0,2]` が厳密 0）を除く。結果を見た後の入力変更なので、別の決定記録と承認が要る
- (b) 判定契約の見直し: 構造的に 0 になる要素の扱い、またはスケールを考慮した救済（`bench-common::parity` のスケール付き絶対誤差項に類する）。tolerance 変更に当たるため承認と spec 側（fandhe-ai-spec）の対応が要る
- (c) カーネルとモデルのずれが見つかった場合の #3137 追加作業としての是正（本実測ではずれは見つかっていないので、現時点で該当しない）
- (d) 現状のまま fail として据え置く（G2 は未充足のまま。#3405 の着手条件は満たされない）

Issue の起票・コメント、baseline 化は承認なしでは行っていない。

## Linux で完了した準備

- 入力集合の正本 `crates/backend-metal/tests/linalg_parity_inputs.rs` と、κ₂ ≤ 1e3 の fail-closed ガード
  `tests/linalg_parity_inputs_guard.rs`（Linux で実行）。
- 入力の是正（`RULE.txt` 参照）: QR 17×17 は旧式だと κ₂ ≈ 3.5e16 のため対角シフトで差し替え、det は n ≤ 17 に限定、
  QR に 1e±20 倍と `1e20 × I₂` を追加。tolerance は不変。
- `make check-cross-metal-tests`（aarch64-apple-darwin 向けの型検査と clippy）で macOS 限定の `linalg_parity.rs` を検査済み。
- 診断テスト 2 件の追加（今回。判定ロジック不変）: `qr_diagnose_all_cases_on_device`（実機）、`qr_input_model_vs_cpu_diagnostic`（デバイス不要）。

## 再実行コマンド（Mac 側）

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
```

## 2026-10-28 追記: 入力是正と再実測（所有者判断 (a)）

`qr_input(3,8)` の是正と Mac mini 再実測は #2676 の PR で実施した（同じ作業。判定の正は `../metal-linalg-lu-chol-qr-2676/RULE.txt` の「改訂（2026-10-28）」）。
結果は 5/5 run で全件 pass（`running 7 tests`・`7 passed; 0 failed`）。記録は `../metal-linalg-lu-chol-qr-2676/README.md` の「2026-10-28 入力是正と再実測」節と `rerun-2026-10-28/`。
本ディレクトリの 2026-10-27 の fail ログは元入力のものとして残す。tolerance・判定式は不変。G2（`docs/backend-gpu-linalg-decision.md` §11.1）のうち Metal 側は充足、CUDA #3138 は別記録。
