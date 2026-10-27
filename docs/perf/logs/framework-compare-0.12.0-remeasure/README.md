# 0.12.0 の 6 フレームワーク同席計測（イシュー #3273）

**状態: 未実測（計測部品と事前登録の草案のみ）。** 実機（Mac mini・GB10）での 5 round 計測は未実施で、本ディレクトリに実測値はない。受け入れ条件 A1〜A6 は未達成（下表）。

## 目的

0.12.0 で bench-fandhe の非対称を是正した効果（#3265・#3272。`docs/framework-compare-harness-decision.md` §7）を、6 フレームワーク（fandhe-ai・candle・burn・PyTorch・TensorFlow・SciPy）同席の正式計測で確かめる。判定規則は 0.11.0 から変えない（比 = 同一 run 内の最速の有効な相手 / fandhe-ai reuse の 5 run 中央値。1 超で 1 位・0.90 以上で僅差・parity fail は判定不能）。対戦成績・対応表の更新は #3274 の範囲で、本イシューでは行わない。

## 収録物（本 PR 時点）

| パス | 内容 |
|---|---|
| `RULE.txt` | GB10 の事前登録規則（未固定の草案。固定時刻・SHA は計測直前に記入） |
| `scripts/dgx-run-0120.sh` | 1 round。新旧 2 ツリー・事前検査・stage4（旧条件腕 28 行）・run ごとの `cargo tree` 検査（v0.12.0 でなければ fail） |
| `scripts/dgx-0120-loop.sh` | 5 round を同じループで実行。各 round 前に load1 < 1.0 かつ GPU 0% が 3 連続のゲート。`ALL-DONE completed=5/5` |
| `aggregate/median_rounds.py` | 0.11.0 版のコピー（変更なし） |
| `aggregate/swap_fandhe_arm.py` | 旧腕判定用: 新腕 JSONL の fandhe-ai 行を同 run の旧腕行で置換（`--self-test` あり） |
| `scoreboard/gen_0120.py` | スコアボード生成器（#3274）。0.11.0 Mac mini 版 `gen_0110_macmini.py` の派生。判定関数群は不変で、前版比の入力・`--run-ver`・旧条件の注記（`--m4-old`／`--gb-old`）・届かなかったセルの表（`--unreached-tsv`）を足した |
| `scoreboard/body_0120.html`・`style.css` | テンプレートと色定義（style.css は 0.11.0 版の逐語コピー）。注記 2 点（burn の TF32・Python 側 checksum）・届かなかったセルの節・対応表の 0.12.0 版（タグ基準）を含む |
| `scoreboard/gen_args.txt` | 負荷・日付・skipped の注記引数の記入欄（未実測。推測の値は入れない） |
| `aggregate/check_identity.py` | A1 の照合（tree・rev-stamp・行数。欠損は非零。`--self-test` あり。skipped ログの内容〈Mac mini は burn Metal の 4 件のみ許容〉は照合対象外で run スクリプト側が検査する） |

Mac mini 側は `../framework-compare-0.12.0-macmini-remeasure/`。Mac mini と GB10 の値は同じ表に混ぜない。

## 条件（計測時に確定する項目）

- 旧条件腕 `0ae5670359c0f9d7a82f8bbd71791fb516713627`（#3272 直前・`fandhe-ai =0.12.0`）、新条件腕は計測直前の origin/main（`ff0ac8e4e` 以降）。腕は `.rev-stamp`・`cargo tree --locked`・行頭アンカー付き grep（新ツリーのみ 1 件ヒットを確認済み）で同定する。
- 既定設定（SME OFF・Graph capture OFF。`crates/backend-cpu/src/gemm_blis/mod.rs` の `SME_PRODUCTION_ENABLED = false` は v0.12.0 タグと main で同じ）。
- 旧腕は round 内で測る位置が新腕と違う（GB10 は stage2 の後・stage3 の前）。新旧差には位置による偏りがありうる。

## 結果

未実測。実測後に、機体別に新条件・旧条件・0.11.0 の 3 条件を並べ、比が 1.0 以下のセルへ 3 分類（ハードの上限／計測窓の定義／自分たちの制約）の理由を数字つきで書く（`docs/perf/loss-attribution-matrix.md` §10 の (3)(4) は目標値の列で、合否は実測比のみ。制約の緩和は提案しない）。

### #3256 の `#[ignore]` 実機テスト（C1〜C9・GB10）

未実測（記入欄）。

## 受け入れ条件

| # | 要件 | 状態 | 根拠 |
|---|---|---|---|
| A1 | 全 run の tree が v0.12.0・rev-stamp が同一 | 未実測 | `aggregate/check_identity.py`（自己検査のみ通過） |
| A2 | 5 run 中央値を新・旧・0.11.0 で並べる | 未実測 | — |
| A3 | 全セルの比と 1.0 以下の理由 | 未実測 | — |
| A4 | 事前登録規則を計測前に固定 | 草案のみ | `RULE.txt`（固定時刻は未記入） |
| A5 | 機体を混ぜない | 部品のみ | Mac mini 側 precheck が `Apple M4` を検査 |
| A6 | #3256 の実機テスト結果 | 未実測 | — |

## 再現手順（実機で実施する人向け）

1. 展開ツリー 3 つ（新腕の `scripts/bench/framework-compare`・旧腕の同パス・新腕の全体）を `git archive` で作り `.rev-stamp`（フル SHA）を置く。
2. 行数スモークを 1 回行い、RULE.txt を固定（時刻・SHA・sha256 を記入）。
3. 実機ロックを取り、`ROOT`・`OLD_ROOT`・`EXPECT_NEW_SHA`・`EXPECT_OLD_SHA` を渡して `dgx-0120-loop.sh` を `setsid nohup` で起動。
4. 集計: `median_rounds.py`、旧腕は `swap_fandhe_arm.py`（GB10 は主 JSONL と extra JSONL の両方を NEW に渡す）→ `median_rounds.py`、`check_identity.py`。スコアボード生成: `scoreboard/gen_0120.py`（0.11.0 の読み方は `framework-compare-0.11.0-macmini-remeasure/README.md` の「スコアボード」の引数に準じ、`--m4-prev`・`--gb-prev` の既定は 0.11.0 の中央値。旧条件の判定は `swap_fandhe_arm.py` で置換した入力に同じ生成器をもう 1 回かけて `gen_0120_oldarm.out` に出す）。セル表生成器は未作成（残作業）。

## スコアボード（#3274）

- **状態: 生成器とテンプレートのみ（0.12.0 実測待ち）**。`gen_0120.out`（判定一覧）・HTML・Artifact は未生成で、0.12.0 の勝敗・届かなかったセルの値はまだ無い。
- 生成器は 0.11.0 の正式データで回帰確認済み（`--run-ver 0.11.0 --prev-ver 0.10.0 --m4-prev ''` で標準出力が `gen_0110_macmini.out` と完全一致。tally 6 1 20 invalid 1 noise 1 total 27）。旧条件の注記は表示のみで、有無で判定一覧が変わらない。`--unreached-tsv` は分類が 3 種以外・比が数値でない行で非零終了する。
- 対応表の fandhe-ai 列はタグ `v0.12.0` 基準（`docs/perf/framework-compare-feature-matrix-0.12.0.md` §8）。相互運用行は #3268 の棚卸しとの照合で「部分的」へ戻した（所有者確認待ち）。
- **公開先（既存 Artifact の更新か新規か）と公開範囲: 所有者未確認（記入欄）**。確認前は公開しない。0.11.0 版のページは上書きしない。
- 注意: 「約 5 倍」「0.33〜0.40」は出典がなく採らない。Python 側 checksum の補正は `docs/perf/loss-attribution-matrix.md` §10.6 の実測（Python÷Rust 3.66〜4.17）に合わせた。
