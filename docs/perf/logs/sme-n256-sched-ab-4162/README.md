# N=256 で SME のスレッド数と分配を変えて A/B する（イシュー #4162）

## 状態

**PR-1（道具立て）時点は実測なし**。機構（既定 OFF）・腕の patch・RULE 雛形・実行キット・集計を入れた段階で、実機の値は一つもない。判定は PR-3 で記録する。

| PR | 内容 | 状態 |
|---|---|---|
| PR-1 | 機構・patch・`RULE.template.txt`・`orchestrate_macmini.sh`・`aggregate.py`・doc | 本 PR（`Refs #4162`。イシューは閉じない） |
| PR-2 | Mac mini のセッションが `RULE.template.txt` を埋めて `RULE.txt` として単独コミット | 未着手 |
| PR-3 | Mac mini での実測と判定の記録（`Closes #4162`） | 未着手 |

## 目的と根拠

Mac mini（Apple M4・4P+6E）の CPU GEMM N=256 は、SME ON でも PyTorch 比 0.888 でまだ約 12% 足りない
（`../sme-default-on-ab-3232/aggregate-macmini.md`）。N=256 の SME 経路はグローバル rayon プール（10 スレッド）と `jobs_per_worker=2` で動く。
`../../cpu-sme-thread-scaling-macmini.md` §2・§3（#3175）では (256,256,128) の SME は T=4 が最大で、T=10 は 0.90 倍に落ちる。
小形状ではワーカー起床・fork-join の固定費と job の粒度が効いている可能性があるため、スレッド数（専用プール）と分配（jpw）を事前登録で測る。

イシュー本文は #4151 の区間別内訳を対象に腕を選ぶとしているが、#4151 の PR-3（内訳の実測）はまだ結果がない。
そのため腕は既存の #3175 実測だけから事前登録した。#4151 の結果が先に main に入った場合は RULE の参考として追記するだけで、腕の組は変えない。

**既定は切り替えない**（Phase 4 の #4165 と所有者承認の範囲）。SME 自体が既定 OFF なので、ADOPT でも本番に効くのは #4156 で SME 既定 ON が ADOPT された場合だけ。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形。PR-2 で `<未記入>` を埋めて `RULE.txt` にする |
| `on-arm.patch` | `SME_PRODUCTION_ENABLED` を true にする 1 行（`../sme-default-on-ab-4156/on-arm.patch` と同一。sha256 一致） |
| `arm-{t4,t6,jpw1,jpw4,t4-jpw1}.patch` | `sme_small_sched.rs` の定数行だけを変える（ENABLED=true と THREADS／JPW） |
| `patch_sha256.txt` | 全 patch の sha256 |
| `orchestrate_macmini.sh` | 7 腕の実行キット（`--dry-run` あり。bash 3.2 互換） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`--self-test`・`--check-template`・`--fingerprint`） |

機構の実装は `crates/backend-cpu/src/gemm_blis/sme_small_sched.rs`。aarch64 版 `dispatch_two_d_dynamic` の SME 分岐の中だけから呼ばれ、NEON・x86・scalar は通らない。
main では `ENABLED=false`・`THREADS=0`・`JPW=0` で、SME 自体も既定 OFF のため本番の挙動は変わらない。

## 腕・セル・判定

| 腕 | THREADS | JPW | 狙い |
|---|---|---|---|
| base | 0（10） | 0（2） | 0.888 の基準（SME ON） |
| aa | 0 | 0 | base と同一バイナリ。ノイズ幅の記録だけ |
| t4 | 4 | 0 | P コア数・#3175 の T=4 最大 |
| t6 | 6 | 0 | 中間 |
| jpw1 | 0 | 1 | 分配の粗化 |
| jpw4 | 0 | 4 | 分配の細分化 |
| t4-jpw1 | 4 | 1 | 組み合わせ |

判定セルは gemm 256 の fresh／reuse。ガードは gemm 512 fresh／reuse と train size=64 fresh／reuse。
候補は判定 2 セルとも 5/5 round で比 ≤ 1.00 かつガードに 5/5 後退なし（checksum・P-BIT 一致）で ADOPT 適格。適格のうち幾何平均が最小のものを選ぶ。
詳細は `RULE.template.txt`。

## 手順（実機セッション。Linux の自動運転では実行しない）

1. PR-2: `RULE.template.txt` の PARAMS（`fixed-at-utc`・`tree_sha`・`tree_fingerprint`・`aggregate.py`／`orchestrate_macmini.sh` の sha256・P-DIAG）を埋め、`RULE.txt` として単独コミットして push する。
2. BASE（クリーンな clone）で `RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。
3. `BASE=<clone> WORK=<作業> TORCH_PY=<venv python> bash docs/perf/logs/sme-n256-sched-ab-4162/orchestrate_macmini.sh`。
4. `python3 -I aggregate.py macmini > aggregate-macmini.md`。perf doc §5.18 に結果を転記する。

## マスク規則・実機ロック

- ホスト名は `<metal-node>`、`$HOME` は `<home>`。ユーザー名・絶対パス・IP は除去し、収録前の grep が 0 件でなければ収録しない。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。

## 検証状況

- Linux で検証済み: lib テスト（`select`・帯の境界・専用プール／jpw の bit 完全一致）、`aggregate.py --self-test`／`--check-template`、`bash -n`、`--dry-run`、patch の `git apply --check`、aarch64-apple-darwin のコンパイル確認。
- **未検証**: Mac mini 上での `#[ignore]` テスト（P-FIRE・P-BIT）、A/B の実行、P-RT。

## 限界

- 候補が 5 つあるため多重比較になる（補正はしない）。
- 専用プールのワーカーとグローバルプールの spin が混在する場面での干渉は、ガードセルが帯の外でプールが生成されないため検出できない。
- macOS には pinning の手段がなく、T=4 が P コアだけで動く保証はない（#3175 §5）。
- NB-RANGE5・NB-SEP は #4142 が所有者承認待ちのため記録だけにする。承認済みとは書かない。

## スコープ外

PR-2・PR-3、#4151 の実測、既定の切り替え（#4165）、SME の unsafe 独立監査（#4157）、#4156 の SME 既定 ON の判定。
