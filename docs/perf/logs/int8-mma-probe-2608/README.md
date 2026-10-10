# INT8 MMA（`mma.sync m16n8k32 s32.s8.s8.s32`）GB10 実行プローブ（イシュー #2608）

格上げ条件 (b) の INT8 側（`docs/int8-quant-grade-up-verification-plan.md` §3）の実行段・bit 一致・TOPS の実測記録。
判定規則は同ディレクトリ `RULE.txt`（実測前に固定）。**充足の宣言は行わない**（spec 側の判断）。

- 受理段（S1〜S3）は #2122 で記録済み（`../sm121-isa-probe-2122/`、`policy=accept_only`）。本記録は S4 起動・S5 同期・S6 bit 一致と TOPS を追加する
- 実装: `crates/backend-cuda/tests/int8_mma_probe_real_device.rs`（#2122 共有ハーネスを再利用し新規 `unsafe` なし。本番経路 `crates/*/src` は不変）
- 環境: `env_info.txt`（GB10・CC 12.1・CUDA/NVRTC 13.0・ノードは `<cuda-node>`）

## 測定コマンド

```bash
# 1 (probe, target) = 1 プロセス・外部 timeout 120 秒。ノードが空いているときに実行
INT8_PROBE_ID=<int8.t1.ones|int8.t2.pattern|int8.k128.pos|int8.k128.neg|int8.tops> \
INT8_PROBE_TARGET=<compute_121|compute_121a|compute_121f> \
timeout -k 10 120 cargo test -p fandhe-ai-backend-cuda --all-features --release \
  --test int8_mma_probe_real_device -- --ignored --nocapture
```

TOPS のみ各 target 5 プロセス（`exec/int8.tops@<target>-run<N>.log`）。他は各 1 プロセス（`exec/<probe>@<target>.log`）。

## 結果（GB10 実測）

### 実行可否・bit 一致（全 12 プロセス）

対照 `ctl.copy`・S1〜S5 は全プロセスで ok、終了コード 0。S6 は全て `bit_exact`（`mismatch=0`）。

| probe | compute_121 | compute_121a | compute_121f |
|---|---|---|---|
| int8.t1.ones（D 全語 == 32） | ok | ok | ok |
| int8.t2.pattern（ホスト i32 参照と一致） | ok | ok | ok |
| int8.k128.pos（+2,064,512） | ok | ok | ok |
| int8.k128.neg（-2,064,512） | ok | ok | ok |

- RULE.txt J3 に従い、全 3 正式 target で「S1〜S5 ok かつ t1・t2・k128 の S6 が全て ok」→ 条件 (b) の INT8 側の充足**候補**（宣言は spec 側）
- t2 は参照モデルの fragment 配置が事前登録時 `unverified` だったが、実機出力と bit 一致したため配置は GB10 実機で裏付けられた（J2 の「判定不能」には該当しない）
- エラーコード: なし（`launch`・`sync` とも `code="-"`）

### TOPS（単一命令ループの発行レート。合否ゲートではない）

`tops = 8192 × 4 chains × 65536 iters × 384 blocks / elapsed_ns / 1e3`（ops = 824,633,720,832）。各 run とも 4 連鎖の d0 が `ITERS×32` と bit 一致、`%globaltimer` の NVRTC 受理も確認。

| target | run1 | run2 | run3 | run4 | run5 | 中央値 |
|---|---|---|---|---|---|---|
| compute_121 | 244.470 | 244.509 | 244.449 | 244.423 | 244.463 | 244.463 |
| compute_121a | 244.481 | 244.604 | 244.465 | 244.407 | 244.365 | 244.465 |
| compute_121f | 244.428 | 244.467 | 244.465 | 244.403 | 244.407 | 244.428 |

単位は TOPS（`mma.sync` 発行のみ。メモリ・エピローグを含まない理論上限寄りの値）。`docs/perf/gemm-optimization-baseline.md` の `mma_f16` 行とは参考比較のみで、spec (f) の判定式には使わない。

## 申し送り（対象外）

- FP8（E4M3／E5M2／`kind::f8f6f4`）の実行プローブは本イシューの対象外（受理段のみ #2122 で記録済み）
- 量子化 GEMM カーネル本体・`QuantOps`・`ScalarDType::I8` は正本 spec の除外事項ゲート（承認が前提）。本プローブはテスト専用
- `docs/cuda-tensor-core-knowledge.md` §2.4「インライン PTX の NVRTC 受理は未検証」の陳腐化是正（既存申し送り）
