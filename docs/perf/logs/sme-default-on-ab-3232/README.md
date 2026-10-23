# SME 既定 ON の前後を Mac mini と GB10 で確認する（イシュー #3232）

## 状態

**実測済み（総合判定 REJECT。2026-10-23 UTC・計測コミット `9cfadc452fbf6ee3633b8a5c38c016976f02f4a3`）**。
Mac mini は到達 10 セルすべてで ON が 5/5 速い。GB10 は非到達のガードだが `gemm 256 fresh`・`gemm 2048 reuse` が 5/5 一貫の後退で、
RULE の合成規則により総合は REJECT。GB10 の PyTorch 相手比は未計測（torch 2.14.0+cu130 の venv なし）。
判定は `aggregate-macmini.md`・`aggregate-gb10.md`（`aggregate.py` の機械出力）と
`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.13 に記録した。`SME_PRODUCTION_ENABLED` を戻すかは所有者判断待ち（Rust は未変更）。

## 目的

#3231 で SME `fmopa` マイクロカーネルを既定 ON にした。Mac mini（Apple M4）での効果は #3176 の A/B からの推定で、
既定 ON の main そのものでは測っていない。GB10（SME 非搭載）の非後退（再開条件 2）も未確認である。

1. Mac mini で ON（main）と OFF（同じコミットで定数 1 行を戻したもの）を 5 round 測り、事前登録規則で ADOPT／REJECT を判定する。
2. GB10 で同じ 2 腕を 5 round 測り、既存経路に 5/5 一貫の後退が無いことを確かめる。
3. 相手（PyTorch）との比をセルごとに記録する（記録のみ）。

判定が REJECT・FAIL、または GB10 で後退ありなら `SME_PRODUCTION_ENABLED` を `false` に戻す（別コミット）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前にコミットして固定。訂正は末尾への追記のみ） |
| `orchestrate_macmini.sh` | Mac mini: ロック → P-TREE → P-R0 → A/B → マスク収録（`--dry-run` あり） |
| `orchestrate_gb10.sh` | GB10: ロック → 外側ゲート → P-TREE → P-R0 → RT → A/B → bitdump → マスク収録（`--dry-run` あり） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`） |
| `aggregate-macmini.md`・`aggregate-gb10.md` | `python3 -I aggregate.py macmini macmini`／`gb10 gb10` の出力（手で直さない。gb10 を先に実行する） |
| `macmini/`・`gb10/` | 実測ログ（マスク済み） |
| `../../../../scripts/bench/framework-compare/run_ab_sme_default_on_cpu.sh` | 2 機共通の A/B 本体（`AB_HOST=macmini\|gb10` 必須。#3176 の `run_ab_sme_macmini_cpu.sh` の派生で元は変更しない） |

OFF 腕のパッチは `docs/perf/logs/sme-default-on-3231/off-arm.patch`（`true` を `false` に戻す 1 行。sha256 は `RULE.txt` に固定）を使う。
腕の向きは before = OFF・after = ON（`compare_gemm_ab.py` の「after/before > 1.00 = 後退」をそのまま使うため）。

## 手順（概要。詳細は各スクリプト冒頭）

1. 開発機で本ディレクトリを含むコミットの `git bundle` を作り、実機へ rsync して clone する（ON ツリー。クリーンに保つ）。
2. 実機の `${WORK}`（同期ツリーの外）を決め、`RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。
3. Mac mini: `ON=<ツリー> TORCH_PY=<venv python> WORK=<作業> bash docs/perf/logs/sme-default-on-ab-3232/orchestrate_macmini.sh`
   （macOS に `setsid` は無い。`nohup` のサブシェルから `zsh -lc` で起動し `ps` で生存確認。torch 2.14.0・numpy 2.5.4 の venv はリポジトリの外）。
4. GB10: `PATH` に `${HOME}/.cargo/bin` を明示し `setsid nohup` で `orchestrate_gb10.sh` を起動する。転送後は先に再ビルドする
   （openrsync の `--delete-excluded` で `framework-compare/target` が消える落とし穴）。
5. 回収後、`aggregate.py` で 2 本の `aggregate-*.md` を作り、perf doc §5.13 を書く。

## マスク規則・実機ロック

- ホスト名は `masked`、`$HOME` は `<home>`、ユーザー名と絶対パスは除去する。実機のホスト名・接続情報はリポジトリに書かない。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。ロックを持たずに測った値は記録に使わない。
- Mac mini の値と M4 Max（開発機）の値を混ぜない。

## スコープ外

- m=64（train・infer の forward）への SME 適用は #3233。
- `docs/perf/logs/sme-threshold-remeasurement-2118/` の記述（既定 OFF 前提）は本イシューでは触らない。
- candle・burn・TF・SciPy を含む 6 フレームワークのスコアボード更新は次の framework-compare 再計測の範囲。
