# GB10 CPU train 固定費の実測ログ（イシュー #3188）

`cpu-reuse-device-update-2106`（device_update 内訳）と `cpu-predict-resident-fixedcost-2105`
（predict_resident 固定費）の診断を、GB10 で pre／head の 2 系列について 5 run ずつ実行した記録。
結論は `docs/perf/cpu-reuse-device-update.md` §11 と `docs/perf/cpu-predict-resident-fixedcost.md` §6。

| パス | 内容 |
|---|---|
| `RULE.txt` | 追加規則（計測前に固定。継承 RULE の sha256 を記録） |
| `measure_series.sh` | GB10 ノード上で 1 系列を直列実行するラッパー（ラベルは `pre`／`head` の allowlist） |
| `{pre,head}/.rev-stamp` | 計測ツリーの SHA（RULE の SHA と一致） |
| `{pre,head}/tree_sha256_{before,after}.txt`・`tree_verify.txt` | 計測前後のツリー指紋と Mac 側 `git archive` 展開との突き合わせ（差分 0 件） |
| `{pre,head}/aarch64_ci_test_*.txt` | aarch64 での bit 一致テストの pass 記録（パスはマスク済み） |
| `{pre,head}/device-update-2106/` | `run{1..5}.jsonl`・`env_info.txt`・`load_gate*`・`aggregate.md` |
| `{pre,head}/predict-resident-2105/` | 同上 |

- pre = `17c7767b588a8c4c52eccccc39ba880519962d53`、head = `4f6716f638a6997559847480ddf93cf6924a96e3`
- ノード側には `.git` がないため `env_info.txt` の `git_head` が空なのは想定どおり（同定は `.rev-stamp`）
- 再現手順: 系列ごとに追跡ファイルだけを専用ディレクトリへ展開し `.rev-stamp` を置いた後、`bash measure_series.sh <pre|head>`
