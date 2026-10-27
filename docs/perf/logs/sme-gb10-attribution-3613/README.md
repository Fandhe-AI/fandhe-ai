# #3613 GB10 後退 2 セルの 3 腕帰属切り分け

## 状態

- GB10 実測済み（2026-10-27 UTC）。計測ツリーは `e2660eba0`（理由は `RULE.txt` §8）。判定 2 セルはともに非再現（ノイズ帯・判定不能）。数値は `aggregate-gb10.md`、ログは `gb10/`、解釈は perf doc §5.15。
- 既定 ON へ戻すかの判断は所有者判断待ち（承認未取得）。
- 規則は計測前に push して固定する。以後の訂正は `RULE.txt` 末尾への追記だけ。ただし §8 追記のみ push 前に `gb10/prereg_append.txt` のタイムスタンプ（hwlock 取得より前）で固定した。

## 目的

#3232 の GB10 A/B で、SME 非搭載の GB10（両腕とも NEON 経路・bitdump bit 同一）でも `gemm 256 fresh`・`gemm 2048 reuse` が 5/5 round で後退した。
原因（選択ロジック／コード配置／ノイズ）を、HEAD 基準の 3 腕（before・prime・after）で切り分ける。前例は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.7。

## 収録物

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（腕・組・巡回順・前提ゲート・分類表・マスク規則・パッチ sha256） |
| `on-arm.patch`・`on-arm-prime.patch` | 計測専用の差分（main には入れない。HEAD `3e5d41016` 基準） |
| `orchestrate_gb10.sh` | GB10 上の一括実行（ロック → ゲート → P-TREE → P-R0 → P-RT → A/B → bitdump → 収録）。`--dry-run` あり |
| `aggregate.py` | RULE §4 の分類を機械適用する厳密集計。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py gb10` |
| `../../../../scripts/bench/framework-compare/run_ab_sme_attribution_gb10_cpu.sh` | 3 腕 A/B 本体（#3232 の `run_ab_sme_default_on_cpu.sh` の派生。元は凍結） |

## 手順（概要）

1. 登録コミットの bundle を GB10 へ転送して clone（転送後は先に再ビルド。openrsync の protect フィルタに注意）
2. `BASE=<clone> WORK=<作業> setsid nohup bash docs/perf/logs/sme-gb10-attribution-3613/orchestrate_gb10.sh`（PATH は `${HOME}/.cargo/bin` を明示）
3. 回収・マスク検査（実ホスト名・`/home/`・ユーザー名・IP の grep が 0 件）後、`aggregate.py gb10` で `aggregate-gb10.md` を作る
4. 結果を perf doc §5.15 と audit doc §12.13 へ追記する

## スコープ外

- `SME_PRODUCTION_ENABLED` の切替・既定 ON の再 A/B・是正（所有者判断）。
- GB10 の PyTorch 相手比（venv なし。「未計測」と記録）。
