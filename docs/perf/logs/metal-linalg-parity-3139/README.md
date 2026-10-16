# Metal 線形代数（LU 系・cholesky・QR）実機 parity 計測ログ（イシュー #3139・Stage C）

事前登録規則は同ディレクトリの `RULE.txt`（実機実行前に固定）。判定方式は `docs/backend-gpu-linalg-decision.md` §5
（Metal f32 対 CPU f64 は精度クラスの差なので baseline では吸収せず、入力集合を固定して `assert_parity` を fail 0 件で
厳密に適用する）。親は #3137（Stage B・起動の結線）。

## 結果

**未実測**（2026-10-16 時点）。理由: 本作業を行った Linux 環境から Mac mini（`<metal-node>`）に到達できなかった
（`docs/real-hardware-verification-env.local.md` が無く、ssh の接続先も未設定）。
実機の pass／fail・`.rev-stamp`・`run.log` は無いため記録しない。受け入れ基準の実機項目は未達で、Mac 側での実行を申し送る。

### 1e±20 倍 QR・`1e20 × I₂` の NaN／0 確認

未確認（実機未実行）。`qr_matches_cpu_on_device` は倍率ケースで R・Q の有限性と R 対角の非 0、倍率で正規化した
R の parity を assert する（1e-20 倍は絶対誤差の救済で 0 を返しても自明 pass しうるため）。

## Linux で完了した準備

- 入力集合の正本 `crates/backend-metal/tests/linalg_parity_inputs.rs` と、κ₂ ≤ 1e3 の fail-closed ガード
  `tests/linalg_parity_inputs_guard.rs`（Linux で実行）。
- 入力の是正（`RULE.txt` 参照）: QR 17×17 は旧式だと κ₂ ≈ 3.5e16 のため対角シフトで差し替え、det は n ≤ 17 に限定、
  QR に 1e±20 倍と `1e20 × I₂` を追加。tolerance は不変。
- `make check-cross-metal-tests`（aarch64-apple-darwin 向けの型検査と clippy）で macOS 限定の `linalg_parity.rs` を検査済み。

## 実機での手順（Mac 側）

1. `.rev-stamp` に実行コミットのフル SHA を書いて転送し、`RULE.txt` の対象コミットと整合することを確認する。
2. 他のビルドを並走させず、次を実行して `run.log`・`env.txt`（`sw_vers`・`sysctl -n machdep.cpu.brand_string`・`rustc -V`）を
   このディレクトリへ置く（ホスト名・ユーザー名・内部パスは書かない）。

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
cargo test -p fandhe-ai-backend-metal --release --test linalg_parity -- --ignored --nocapture
```

3. 4 件の parity 判定対象と、記録のみの 2 件の結果を本 README・`metal-linalg-lu-chol-qr-2676/README.md` の実測記入欄へ書く。

## fail 時の記録欄

（実機未実行のため該当なし。fail した場合は tolerance・入力・判定を変えず、fail_count・max_abs_diff と
カーネル側の是正方針を「要対応（#3137 の追加作業。ユーザー判断待ち）」として記録する。）
