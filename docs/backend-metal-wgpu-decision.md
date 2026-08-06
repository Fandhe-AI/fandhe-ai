# Metal バインディング経路: `wgpu` 不採用判断（#41・TASK-1.8d）

イシュー #41「docs(backend-metal): TASK-1.8d wgpu 不採用判断の明記」に対応する。
TASK-1.8（親 #37）の受け入れ条件「`wgpu` 不採用の判断がコードコメント・ドキュメントに明記されている」を満たすためのドキュメント。判断そのものは PoC-v2-4 で実測済み（`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「経路選定の比較判断」節が正本）であり、本ドキュメントは実装リポ側にその要点と出典を明記する。

## 判断サマリ

**`backend-metal` の Metal バインディング経路は `objc2-metal` 直接呼び出しを採用し、`wgpu` は採用しない。**

根拠は次の 2 点に集約される。

1. 同一アルゴリズムでの実測比較で `objc2-metal` 直接実装が `wgpu` 経由より明確に高速（約 2.3 倍、後述）。
2. `simdgroup_matrix`（Metal のハードウェア行列演算命令）は WGSL に相当命令が存在せず、`wgpu` 経由では原理的に到達できない。`backend-metal` の性能改善余地（多 simdgroup 化等）はこの命令を前提とするため、`wgpu` を選ぶと将来の改善余地自体を失う。

実測環境: Apple M4 Max（GPU 40 コア・macOS 26.6）。出典: `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`。

## 実測根拠

### f32 GEMM スループット（TFLOPS 中央値。warmup 20 回・計測 20 回、size=4096 の naive のみ warmup 2 回・計測 5 回）

| カーネル | size=512 | size=2048 | size=4096 |
|---------|---------|-----------|-----------|
| Metal naive | 0.561 | 1.334 | 1.271 |
| Metal tiled | 0.813 | 2.127 | 2.123 |
| Metal simdgroup | 0.984 | 3.159 | 3.134 |
| wgpu tiled（境界検査無効化後） | 0.403 | 0.909 | 0.920 |

出典: 同 README「計測結果」節の表（`evidence/rust_wgpu_throughput.log` / `evidence/rust_metal_throughput.log`）。

### 「約 2.3 倍」の前提条件（数値の取り扱い上の注意）

- 比較対象は **同一アルゴリズム**（tiled、threadgroup／workgroup 共有メモリによる古典的タイル化 GEMM）どうし。`simdgroup_matrix` を使う `Metal simdgroup`（3.134 TFLOPS）はさらに別枠であり、tiled 比で約 1.5 倍上乗せされる。
- **「約 2.3 倍」は wgpu 側の naga ランタイム境界検査を無効化した後の値との比較である**（`create_shader_module_trusted` + `ShaderRuntimeChecks::unchecked()`。size=4096: Metal tiled 2.123 TFLOPS vs wgpu tiled 0.920 TFLOPS）。
- 境界検査を有効のまま（wgpu のデフォルト `create_shader_module`）で計測すると wgpu 側は 0.531 TFLOPS まで落ち、差は約 4 倍に拡大する。上表・約 2.3 倍はいずれも**無効化後**の、wgpu にとって有利な側の数値であることに注意する。
- WGSL 側自体は `if (row < m && col < n)` 等の境界チェックを手動で維持したまま計測している（naga の自動挿入検査のみを無効化）。

出典: 同 README「計測結果」節「wgpu の計測値についての注記」・「経路選定の比較判断」節「性能」小節。

## 保守性比較

| 観点 | wgpu | Metal 直接（`objc2-metal`） |
|------|------|------------------------------|
| 実装行数（PoC 実測） | 327 行 + WGSL 65 行 | 336 行 + MSL 120 行（3 カーネル分） |
| `unsafe` 面積 | なし（すべて safe API） | バッファ生成・`setBytes`/`setBuffer` 呼び出しに `unsafe` あり。ただし 1 ファイル内に局所化 |
| 依存 crate | `wgpu`（124 パッケージロックの大規模依存）+ `pollster` | `objc2` 系 3 crate（`objc2`／`objc2-foundation`／`objc2-metal`） |
| デバッグ手段 | wgpu のバリデーションレイヤー | Xcode Metal Debugger（GPU フレームキャプチャ・シェーダデバッガ）が直接使える |
| クロスプラットフォーム性 | あり（Vulkan/DX12/WebGPU 等に展開可能） | なし（Metal/Apple プラットフォーム専用） |
| ハードウェア機能への到達 | 低い（WGSL の共通集合に制限） | 高い（`simdgroup_matrix` 等 Metal 固有命令に到達可能） |

出典: 同 README「経路選定の比較判断」節「保守性」小節の表。

## REQ-8（カーネル境界検査規約）との関係

`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」は、性能下限・最適化達成を理由にシェーダ・カーネル側の手動境界チェックを省略しないことを CPU・CUDA・Metal の全カーネルに義務付ける。

PoC-v2-4 の wgpu 計測で無効化したのは **naga がストレージバッファアクセスに自動挿入するランタイム境界検査**のみであり、WGSL コード自身に書かれた手動境界チェック（`if (row < m && col < n)` 等）は維持したまま計測している。つまり「境界検査を切れば速い」という短絡的な結論の根拠にこの PoC を使うことはできない。

`backend-metal` の MSL カーネル実装（本 TASK-1.8 以降）でも、ベクトル化ロード・タイル端の分岐削減等の最適化を適用する場合は手動境界チェックを維持したうえで行う。本判断ドキュメントの数値（約 2.3 倍・約 4 倍）は「境界検査の有無」ではなく「`wgpu`（WGSL/naga 経由）と `objc2-metal`（MSL 直接）というバインディング経路の違い」に起因するものであり、境界検査省略を正当化する根拠として引用してはならない。

## 再検討条件

`wgpu_gemm.rs`（PoC-v2-4 内）は削除せず、将来クロスプラットフォーム対応（WebGPU/Vulkan 等への展開）が要件化した場合の代替経路の参考実装として温存する。`01-brainstorm.md`「v2 自作範囲の境界定義」は Apple Silicon（Metal）と CUDA を別実装として許容しており、現時点では `wgpu` の「1 実装で複数バックエンドに展開できる」利点を活かす前提になっていない。この前提（要件が CUDA デフォルト対応 + Apple Silicon/AMD 拡張である点）が変わり、単一実装での複数 GPU バックエンド対応が要件化した場合に再評価する。

## 出典

- 正本: `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「計測結果」節・「経路選定の比較判断」節・「要件への示唆」節
- `docs/spec/01-brainstorm.md`「v2 自作範囲の境界定義」
- `docs/spec/05-tasks.md` TASK-1.8（親 #37 受け入れ条件）
- `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」
- `.claude/rules/deps-policy.md`（Metal 許容依存区分: `objc2`／`objc2-foundation`／`objc2-metal`。`wgpu` は許容依存 8 区分に含まれない）
