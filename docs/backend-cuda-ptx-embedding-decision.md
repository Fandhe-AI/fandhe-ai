# CUDA GEMM ビルド時 PTX 事前埋め込み（candle 方式）不採用判断（#1024）

イシュー #1024「f32 GEMM 本番経路を module_cache／NVRTC ディスクキャッシュへ結線する」の
検討過程で、実行時 NVRTC コンパイルそのものを回避する代替案として、candle の `build.rs`
事前生成方式（ビルド時に PTX を生成し `include_str!`／`OUT_DIR` 経由でバイナリへ埋め込む
手法）の採否を検討した。本ドキュメントは決定記録として判断とその根拠を残す。

## 判断サマリ

**本クレート（`backend-cuda`）ではビルド時 PTX 事前埋め込みを不採用とする。実行時 NVRTC
コンパイル方式を維持し、コンパイル結果の再利用は module_cache（プロセス内 LRU）で行う。**

## 根拠

1. **CUDA toolkit 非搭載環境でのビルド成立契約との衝突（REQ-2・PoC-v2-5）**: `build.rs` で
   PTX を生成するには `nvcc` または NVRTC をビルド時に実行できる環境が必要になる。本クレートは
   `cudarc` の動的ロード方式（`dynamic-loading` feature）により CUDA toolkit 非搭載環境でも
   ビルドが成立する契約（`.claude/rules/deps-policy.md` CUDA 区分・`.github/workflows/ci.yml`
   の `build-no-cuda-toolkit` ジョブが機械検証）を持つ。ビルド時コンパイルはこの契約と直接衝突する。
2. **対象アーキテクチャの実行時可変性**: PTX の対象 arch（`--gpu-architecture=sm_XX`）は
   `device.arch()` により実行時に決まる（sm_86・sm_121 等、実機の compute capability に依存）。
   ビルド時に単一 arch へ固定する、または全 arch 分の fat バイナリを埋め込む方式のいずれも、
   NVRTC バージョン差・`<mma.h>` include path 解決（`nvrtc.rs` ドキュメンテーションコメント
   「NVRTC ヘッダ問題」参照）といった実行時性の高い要素を静的に固定できない。
3. **ソースとの二重管理（ドリフトリスク）**: 生成済み PTX をリポジトリへコミットする方式は、
   カーネルソース（`kernels.rs`／`kernels_wmma_opt.rs` の NVRTC 文字列）と PTX 成果物の二重管理
   になり、再生成を忘れるとソースと PTX が乖離する。再生成には toolkit 搭載のビルド環境
   （self-hosted runner の追加）が要り、`.claude/rules/ci.md`「runner」節（self-hosted 追加は
   `docs/runner-policy.md` の例外整理に従いユーザー承認が要る）に抵触しうる。
4. **既存の依存追加方針との整合**: `prost-build`（`protoc` ビルド時依存）を使わずビルド時外部
   ツール依存を増やさない方針（`.claude/rules/deps-policy.md` 相互運用区分）と同じ考え方を、
   ビルド時コンパイラ依存にも適用する。

## 再検討条件

以下のいずれかが満たされた場合に再検討しうる:

- toolkit 搭載のビルド環境（CI・crates.io ビルド双方）を常設できる合意が得られた場合。
- ディスク上の PTX 成果物の真正性を保証する認証済み検証手段（署名検証。
  `docs/cuda-jit-cache-design.md`「ディスク PTX を実行入力にしない判断」節と同じ論点）が
  許容依存として承認され、ビルド成果物の署名検証を実行時に行える場合。

## 採用した代替策

実行時 NVRTC コンパイルは維持しつつ、同一プロセス・同一 context 上での再コンパイルを避ける
プロセス内 LRU（`crates/backend-cuda/src/module_cache.rs::load_function_cached`）を
`CudaGemm::new`（f32 GEMM 本番経路）へ結線した。効果の範囲・実測は
`docs/perf/cuda-tape-init-cost-diagnosis.md` §3.1・§6.5・§8(b) を参照。
