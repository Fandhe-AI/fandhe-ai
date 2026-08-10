# ピークメモリ計測手段と環境差

<!--
本ファイルの役割: REQ-14（`docs/spec/04-requirements.md`）「計測手段の環境差」
受け入れ基準・TASK-14.3（`docs/spec/05-tasks.md`）・#180 の成果物。
「内部計測 API を計測手段の主軸とし、外部計測は補助的な裏取りに留める」方針と、
その根拠となる環境差（統合メモリアーキテクチャでの外部計測手段の制約）を記す。
docs のみの変更であり `docs/spec/`（正本 submodule）は編集しない。
-->

## 背景・目的

v1（Burn/CubeCL 実装）では Rust 側に PyTorch `torch.cuda.max_memory_allocated()`
相当の内部計測手段がなく、ピークメモリの計測を外部計測（`nvidia-smi`）に頼らざ
るを得なかった。この制約は v1 PoC-5（`docs/spec/03-poc/poc-5-performance/README.md`）
で顕在化しており、DGX Spark（GB10）は統合メモリアーキテクチャのため
`nvidia-smi --query-gpu=memory.used` が `Not Supported`（`[N/A]`）を返すこと
が実測で確認されている（同 README 37 行目）。代替手段として採用したプロセス
単位ポーリング（`nvidia-smi --query-compute-apps=pid,used_memory`、0.3〜0.4 秒
間隔）には、計測手段・計測単位の非対称性（プロセス全体の累積値であり区間を
切り出せない）という限界があった（同 README 83〜89 行目）。

REQ-14（`docs/spec/04-requirements.md:268` 以降）はこの教訓を踏まえ、v2 では
「内部計測 API の必須提供」を独立した受け入れ基準として明記し、その上で
「計測手段の環境差」（同ファイル、REQ-14 受け入れ基準の該当項目）として
「内部計測 API を計測手段の主軸とし、外部計測は補助的な裏取りに留めること」
のドキュメント化を求めている。本ファイルはその成果物である。

## 内部計測 API（主軸）

TASK-14.1（#173〜#176、完了済み）により、`tensor-core::memory_stats::MemoryStats`
トレイトが CPU/CUDA/Metal の 3 バックエンドで同一シグネチャで実装されている
（`crates/tensor-core/src/memory_stats.rs`）。

| メソッド | 意味 | PyTorch 対応 |
|---------|------|--------------|
| `allocated_bytes()` | 現在の確保済みバイト数（生存中のアロケーション合計） | `torch.cuda.memory_allocated()` 相当 |
| `peak_allocated_bytes()` | 直近の `reset_peak()` 以降のピーク値 | `torch.cuda.max_memory_allocated()` 相当 |
| `reset_peak()` | ピーク値を現在値へリセットし計測区間を区切る | `torch.cuda.reset_peak_memory_stats()` 相当 |

実装状況:

- `CpuMemory`（`crates/backend-cpu/src/memory.rs`）
- `CudaMemory`（`crates/backend-cuda/src/memory.rs`）
- `MetalMemory`（`crates/backend-metal/src/memory.rs`）
- プール導入時のデコレータ構成は `docs/memory-pool-design.md` を参照（明示解放 API を含む）

### 運用上の前提

- トラッカー（`AllocationTracker`）はプロセスグローバルな `static` ではなく
  `Arc` で保持される。共有されるのは**同一インスタンス、またはそのインスタンス
  の参照（`&CpuMemory` 等）／`clone()` を経由した場合のみ**であり、
  `CpuMemory::new()`／`CudaMemory::new()`／`MetalMemory::new()` はそれぞれ
  独立した新規 `AllocationTracker` を生成する（プロセスグローバルにも他
  インスタンスにも自動では共有されない）。これは (a) 並列実行される単体
  テスト間の計数混線（フレーキーテスト化）を避ける、(b) グローバル可変
  状態を避ける安全側判断、の 2 点による
  （`crates/tensor-core/src/memory_stats.rs` モジュールコメント）。
  `CpuMemory`／`CudaMemory` は `Arc` 保持の `tracker` フィールドを安価に
  複製できるため `derive(Clone)` を持つが、`MetalMemory` は `MetalContext`
  が `Clone` を導出していないため `Clone` を持たない（`crates/backend-metal/
  src/memory.rs` の `MetalMemory` doc コメント参照）。`MetalMemory` を
  複数箇所で共有したい場合は `clone()` ではなく、単一インスタンスへの
  参照（`&MetalMemory` や呼び出し側での `Arc<MetalMemory>` 包装）を渡す。
- spec が言う「プロセス内のピーク値」は、計測対象プロセスがバックエンド入口
  （`CpuMemory` 等）を単一インスタンスとして生成し、計測に関わる全経路で
  それを共有する運用（ベンチハーネスが想定する形。共有の具体手段は上記の
  とおりバックエンドにより異なる）でのみ満たされる。複数箇所で `new()`
  すると `AllocationTracker` が別々に生成され、`peak_allocated_bytes()` は
  プロセス内の確保を集約せず、正本とする測定値が過少になる。ベンチハーネス
  は単一入口インスタンスの共有を必須条件とする。計測区間を区切りたい場合は
  `reset_peak()` を境界で呼び出す。

### 計測粒度の制約

本 API が計測するのは `MemoryOps`（`alloc_zeroed`／`upload`）経由のデバイス
バッファ確保のみであり、`BackendOps` 演算内部が一時的に確保する `Vec<f32>`
（例: CPU バックエンドの GEMM 出力バッファ）は対象外である。計測要否は
TASK-14.2（GEMM 4096³ 係数上限の実測）で判断し、必要であれば別イシューで
追跡する（`.claude/rules/out-of-scope-tracking.md`）。

## 計測手段の環境差

### ディスクリート GPU 環境

`nvidia-smi --query-gpu=memory.used` によって GPU デバイス単位のメモリ使用量
を取得できる（従来型の GPU メモリ管理を持つ環境）。

### 統合メモリアーキテクチャ（Apple Silicon・DGX Spark GB10 等）

統合メモリアーキテクチャでは GPU 専用メモリという概念自体が存在しないため、
外部計測手段に以下の制約が生じる。

- **NVIDIA 系（DGX Spark GB10）**: `nvidia-smi --query-gpu=memory.used` が
  `Not Supported`（`[N/A]`）を返す（v1 PoC-5 実測、
  `docs/spec/03-poc/poc-5-performance/README.md:37`）。代替として
  `nvidia-smi --query-compute-apps=pid,used_memory` によるプロセス単位ポーリ
  ング（0.3〜0.4 秒間隔）を用いる方法があるが、以下の限界を伴う（同
  README 83〜89 行目）。
  - ランタイムオーバーヘッド込みの値になる
  - サンプリング間隔（0.3〜0.4 秒）による瞬間ピークの取りこぼしがありうる
  - プロセス全体の累積値であり、演算区間（例: GEMM のみ）を切り出せない
- **Apple Silicon（Metal）**: `nvidia-smi` に相当する GPU 別メモリ計測ツール
  が存在せず、プロセスレベルの外部観測（OS のプロセスメモリ計測等）に頼る
  ほかない。

## 方針: 内部計測 API を主軸、外部計測は補助的な裏取り

上記の環境差により、外部計測手段は環境（ディスクリート GPU か統合メモリか）
によって可用性・精度・粒度が一様でない。このため v2 では次の方針を採る。

- **数値の正本は内部計測 API（`MemoryStats::peak_allocated_bytes()`）の値と
  する**。バックエンド共通シグネチャであり、環境（CPU/CUDA/Metal・ディスク
  リート GPU か統合メモリか）によらず同一の取得方法で計測できる。
- **外部計測（`nvidia-smi` 等）は補助的な裏取りに留める**。用途はオーダー
  確認（内部計測値が桁として妥当か）とリーク兆候の相互検証（内部計測が示さ
  ない箇所でメモリが増え続けていないかの確認）に限る。統合メモリアーキテク
  チャでは外部計測自体が利用できない、または区間を切り出せない場合があるた
  め、外部計測の欠如・粗さを理由に内部計測 API の値を疑う判断はしない。

## 関連タスクへの導線

- **TASK-14.2**（GEMM 4096³ 係数上限の実測記録、#177〜#179）: 内部計測 API
  導入後の実測に基づき、REQ-14 の係数上限（初期リリースは理論最小ワーキング
  セット ≈ 192MiB の 2 倍以内、`docs/spec/04-requirements.md` REQ-14）を確定
  する。実測値は本ファイルではなく TASK-14.2 側の成果物に記録する。**確定済み
  （#179。#385・#392 で Metal・CUDA 実機実測完了により再判定済み）**: CPU・Metal・CUDA
  いずれも実測（対理論比 1.000）に基づき係数 2.0 を維持・確定した（超過なし。詳細は
  `docs/peak-memory-coefficient-decision.md`）。
- `docs/memory-pool-design.md`: プール導入時の係数上限維持・明示解放 API の
  設計。

## 参考（v1 実測値の位置づけ）

v1 PoC-5 では GEMM 4096³ の外部計測値（`nvidia-smi --query-compute-apps`）が
3235MiB（理論最小ワーキングセット ≈ 192MiB の約 17 倍）に達した
（`docs/spec/03-poc/poc-5-performance/README.md` 結論節）。これは Burn/CubeCL
のバッファプール蓄積挙動に起因すると推定される事象であり、**プールを持たな
い v2 の自作アロケータ設計には適用されない参考値**である（REQ-14
「v1 実測値の位置づけ」、`docs/spec/04-requirements.md`）。v2 の実測値は
TASK-14.2 で確定するまで本ファイルには記載しない（確定記録は
`docs/peak-memory-coefficient-decision.md` を参照。本ファイルは計測手段の
方針文書に留め、確定値の重複記載は行わない）。
