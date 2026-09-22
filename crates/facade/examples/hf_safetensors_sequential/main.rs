//! HF（PyTorch）レイアウトの safetensors から `compat::Sequential` を
//! 復元する example（イシュー #2080・親 #2059）。
//!
//! `docs/huggingface-safetensors-interop-guide.md` の一次ソース
//! （`inference.rs` ↔ `site/examples/inference.md` と同じ慣習。
//! `.claude/rules/code-comment-style.md`）。ガイドのコード片は本ファイル
//! からの抜粋であり、doc 自身のコードブロックはコンパイル検査の対象
//! ではない——本 example の実行成功
//! （`cargo run -p fandhe-ai --example hf_safetensors_sequential`）が
//! ガイドの受け入れ条件を担保する。
//!
//! オフライン・決定的（ネットワーク接続なし・`manual_seed` 固定）。
//! 実 HF ハブからのダウンロードは行わない——PyTorch が保存する
//! `nn.TransformerEncoderLayer` 形式の safetensors を、fandhe
//! `Sequential`（`Embedding → TransformerEncoderLayer`）から**合成**
//! してから読み戻す（`convert.rs` の役割）。
//!
//! 実行する 3 経路:
//! 1. `load_safetensors_f32(path)` によるロード＋`require_keys` 全件検査
//! 2. `convert::from_pytorch_layout` による復元＋`load_state_dict`
//!    （元モデルとの bit 完全一致を検証）
//! 3. batch decode（greedy／top-k）のスケッチ（causal ではない。
//!    ガイド §6・§7「KV キャッシュは申し送り」参照）

#[path = "convert.rs"]
mod convert;

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{load_safetensors_f32, require_keys, save_safetensors_f32};

// モデル構成（意図的に小さい。example の実行時間・可読性のため）。
const VOCAB: usize = 6;
const EMBED_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const FEED_FORWARD: usize = 16;
const BATCH: usize = 2;
const SEQ_LEN: usize = 3;

fn build_model(seed: u64) -> Result<Sequential, Box<dyn std::error::Error>> {
    let model = Sequential::new()
        .add_embedding(VOCAB, EMBED_DIM, /* padding_idx = */ None, seed)?
        .add_transformer_encoder(EMBED_DIM, NUM_HEADS, FEED_FORWARD, seed + 1)?;
    Ok(model)
}

fn ids_tensor(ids: &[[i32; SEQ_LEN]; BATCH]) -> Tensor<f32> {
    let mut data = Vec::with_capacity(BATCH * SEQ_LEN);
    for row in ids {
        for &v in row {
            data.push(v as f32);
        }
    }
    Tensor::new(data, &[BATCH, SEQ_LEN]).expect("固定 shape の構築は失敗しない")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fandhe_ai::manual_seed(2080);

    // --- 合成チェックポイントの用意（HF チェックポイントの代替） ---
    //
    // 実際の HF チェックポイントは Python 側で用意する。F32 以外
    // （bf16／f16 が多い）の場合は事前変換が必要（ガイド §2「非目標」
    // 参照。以下はライブラリ内では実行しない・案内のみのスニペット）:
    //
    // ```python
    // from safetensors.torch import load_file, save_file
    // state = load_file("model.safetensors")
    // save_file({k: v.float() for k, v in state.items()}, "model_f32.safetensors")
    // ```
    let source_model = build_model(/* seed = */ 42)?;
    let fandhe_state = source_model.state_dict();
    let mut pt_layout = convert::to_pytorch_layout(&fandhe_state)?;

    // HF チェックポイントには `Sequential` の層集合に属さない余剰
    // テンソル（LM head 等）が含まれることが多い。合成データにも
    // 同様の余剰テンソルを 1 つ加え、`from_pytorch_layout` の
    // allowlist 分離を実演する。
    let lm_head = fandhe_ai::rand(&[VOCAB, EMBED_DIM])?; // PyTorch 慣習 [out, in]
    pt_layout.insert("lm_head.weight".to_string(), lm_head.clone());

    let temp_dir_guard = convert::TempDirGuard::create("example")?;
    let checkpoint_path = temp_dir_guard.checkpoint_path();
    save_safetensors_f32(&checkpoint_path, &pt_layout)?;

    // === 例 1: load_safetensors_f32（パス版）＋ require_keys ===
    let loaded = load_safetensors_f32(&checkpoint_path)?;
    let mut expected_keys: Vec<&str> = pt_layout.keys().map(String::as_str).collect();
    expected_keys.sort_unstable();
    require_keys(&loaded, &expected_keys)?;
    println!(
        "例 1: {} 件のテンソルをロードし、必須キー {} 件の充足を確認した",
        loaded.len(),
        expected_keys.len()
    );

    // === 例 2: from_pytorch_layout による復元 → load_state_dict ===
    let (restored_state, extra) = convert::from_pytorch_layout(&loaded, &["lm_head.weight"])?;
    let mut restored_model = build_model(/* seed = */ 999)?; // 元と異なる seed で初期化
    restored_model.load_state_dict(restored_state)?;
    println!(
        "例 2: Sequential へ復元完了。余剰テンソル（allowlist 分離）: {:?}",
        extra.keys().collect::<Vec<_>>()
    );

    // 元モデルと復元モデルの推論結果が bit 完全一致することを確認する
    // （`.claude/rules/coding-rust.md`「許容誤差を単独で緩和しない」の
    // 趣旨に沿い、ここでは完全一致で示す）。
    let probe_ids = ids_tensor(&[[0, 1, 2], [3, 4, 5]]);
    let out_source = source_model.predict(&probe_ids)?;
    let out_restored = restored_model.predict(&probe_ids)?;
    let bit_exact = tensors_bit_exact(&out_source, &out_restored);
    println!("例 2: 元モデルと復元モデルの推論出力はビット一致: {bit_exact}");
    if !bit_exact {
        return Err("復元モデルの出力が元モデルと一致しない".into());
    }

    // === 例 3: batch decode（greedy／top-k）のスケッチ ===
    //
    // 注意（ガイド §6「causal 性について正直に書く」節と同一の警告）:
    // `add_transformer_encoder` は非 causal・mask なしの self-attention
    // であるため、本経路は「ids → logits → 次 id → append → 全系列
    // 再計算」という**機構の例示**であり、causal LM として正しい
    // 確率を与えるものではない。真の causal decode には
    // `Var::scaled_dot_product_attention(is_causal=true)` 相当の合成
    // が必要で、`Sequential` は公開していない（KV キャッシュ統合は
    // #2083／#2084／#2191 の完了を待つ。ガイド §7 参照）。
    let head_t = extra
        .get("lm_head.weight")
        .expect("lm_head.weight は allowlist 経由で extra に入る")
        .transpose_2d()?
        .contiguous(); // PyTorch [V, E] → matmul 右辺 [E, V]

    let mut ids = probe_ids.clone();
    for step in 0..2 {
        let (b, l) = (ids.shape()[0], ids.shape()[1]);
        let tape = fandhe_ai::tape();
        let ids_var = tape.var(&ids);
        let hidden = restored_model.forward(&tape, &ids_var)?; // [B, L, E]
        let flat = hidden.reshape(&[b * l, EMBED_DIM])?;
        let head_var = tape.var(&head_t);
        let logits_flat = flat.matmul(&head_var)?; // [B*L, V]
        let logits = logits_flat.reshape(&[b, l, VOCAB])?;
        // `narrow` は zero-copy view のため一般に非 contiguous（`Var::reshape`
        // の契約「非 contiguous は `ShapeError::NonContiguousReshape`」に
        // 触れる）。ここでは reshape せず、`last` を `[B, 1, V]` のまま
        // 縮約軸 2（`V`）へ `argmax`／`softmax`／`topk` を適用する。
        let last = logits.narrow(1, l - 1, 1)?; // [B, 1, V]

        // greedy: 最終位置の argmax（縮約軸 2 = V）。
        let greedy_next = last.argmax(Some(2))?; // Tensor<i32> [B, 1]

        // top-k: 最終位置の softmax 確率上位 k 件（サンプリング自体は
        // host 側の合成が必要なため、ここでは候補の提示に留める。
        // `multinomial` 相当の host 側抽選はガイド §6 参照）。
        let probs = last.softmax(2)?;
        let (_topk_values, topk_index) = probs.topk(/* k = */ 2, /* dim = */ 2, true)?;

        println!(
            "step {step}: greedy 次 id（batch 0）= {:?}, top-2 候補（batch 0）= {:?}",
            greedy_next.get(&[0, 0]),
            [topk_index.get(&[0, 0, 0]), topk_index.get(&[0, 0, 1])]
        );

        ids = append_ids(&ids, &greedy_next);
    }
    println!(
        "例 3: 2 ステップ decode 後の生成系列（batch 0）: {:?}",
        (0..ids.shape()[1])
            .map(|j| ids.get(&[0, j]))
            .collect::<Vec<_>>()
    );

    // `temp_dir_guard` の `Drop` がディレクトリごと削除するため、
    // ここでの明示的なファイル削除は不要（RAII で異常終了時も
    // リークしない）。`drop` は経路を明示するためのドキュメント的
    // 呼び出し（`main` 終端まで生存させても実害はないが、以降で
    // 誤って `checkpoint_path` 経由の I/O を追加されても安全に
    // 倒すため早期に手放す）。
    drop(temp_dir_guard);
    Ok(())
}

/// `ids [B, L]` の末尾へ `next_ids [B, 1]`（`Var::argmax(Some(2))` が
/// 返す greedy 選択済みクラス id。縮約軸 2 のみ潰れるため shape は
/// `[B, 1]`）を 1 列追加し `[B, L+1]` を返す（`Tensor` に `cat` 相当が
/// 無いため、行優先データを直接組み立てる。`convert::pack_in_proj` と
/// 同じ方式）。
fn append_ids(ids: &Tensor<f32>, next_ids: &Tensor<i32>) -> Tensor<f32> {
    let (b, l) = (ids.shape()[0], ids.shape()[1]);
    let ids_c = ids.contiguous();
    let ids_slice = ids_c.as_slice().expect("contiguous() 直後は Some");
    let mut data = Vec::with_capacity(b * (l + 1));
    for row in 0..b {
        data.extend_from_slice(&ids_slice[row * l..(row + 1) * l]);
        let next = next_ids
            .get(&[row, 0])
            .expect("greedy_next の shape は [B, 1] のはず");
        data.push(next as f32);
    }
    Tensor::new(data, &[b, l + 1]).expect("固定 shape の構築は失敗しない")
}

fn tensors_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let a_c = a.contiguous();
    let b_c = b.contiguous();
    let (Some(a_s), Some(b_s)) = (a_c.as_slice(), b_c.as_slice()) else {
        return false;
    };
    a_s.iter().zip(b_s).all(|(x, y)| x.to_bits() == y.to_bits())
}
