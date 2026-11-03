# model-io-0.13.0 fixture（イシュー #3828）

0.13.0（タグ `v0.13.0` = `db2b385fb`）の `compat::save_model` が書いた実ファイルです。現行の `load_model` が
0.13.0 のファイルを読めること（後方互換。横断方針 #3750 の R4）と、新しい kind の追加が既存 kind のスキーマを
変えていないこと（R3）を、`crates/facade/tests/compat_sequential_model_io_compat_0_13_0.rs` で確かめます。
手で作ったり推測した値は置いていません。すべて下の生成プログラムの出力です。

## ケース

| ディレクトリ | 層（0.13.0 に存在する kind だけ） | 意図 |
|---|---|---|
| `mlp_bn/` | linear / batch_norm1d / relu / dropout / layer_norm / linear | buffer（running stats）を経路に通す。BN は train モードの predict を 3 回回して初期値から動かし、eval で保存 |
| `cnn/` | conv2d / batch_norm2d / relu / max_pool2d / flatten / linear | 4 次元入力と BN2d の buffer |

各ディレクトリの中身は `manifest.json`・`model.<世代名>.safetensors`（0.13.0 の出力そのまま）と、
`expected.json`（保存時点の `state_dict` の shape と f32 の bit 列、`training`、eval の `predict` の bit 列）です。
`predict_bits` は記録のみで、テストでは比較しません（CPU カーネルのアーキテクチャ差で bit が変わりうるため。
tolerance を持ち込まない）。

## 再生成の手順

1. `git archive v0.13.0 | tar -x -C <空の一時ディレクトリ>` で 0.13.0 のソースを展開する（リポジトリには何も足さない）。
2. 展開先の `crates/facade/examples/gen_model_io_fixture.rs` に下のプログラムを置く。
3. `CARGO_TARGET_DIR=<一時> cargo run -p fandhe-ai --example gen_model_io_fixture -- <出力先>` を実行する。

生成物のファイル名の `<世代名>` は保存ごとに決まる名前で、内容ハッシュではありません。再生成すると
名前が変わるため、README の sha256 とテストの前提を合わせて更新してください。

```rust
use fandhe_ai::Tensor;
use fandhe_ai::compat::{Sequential, save_model};
use std::fmt::Write as _;

fn input(shape: &[usize], base: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let d: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + base).sin()).collect();
    Tensor::new(d, shape).unwrap()
}

fn emit(name: &str, mut m: Sequential, x: Tensor<f32>, out_root: &str) {
    // train モードの predict は BatchNorm の running stats を更新する。
    for _ in 0..3 {
        m.predict(&x).unwrap();
    }
    m.eval();
    let dir = format!("{out_root}/{name}");
    std::fs::create_dir_all(&dir).unwrap();
    save_model(&m, &dir).unwrap();
    let sd = m.state_dict();
    let mut keys: Vec<_> = sd.keys().cloned().collect();
    keys.sort();
    let mut s = String::new();
    writeln!(s, "{{\n  \"training\": {},\n  \"params\": {{", m.training()).unwrap();
    for (i, k) in keys.iter().enumerate() {
        let t = sd[k].contiguous();
        let bits: Vec<String> = t
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits().to_string())
            .collect();
        let sh: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        writeln!(
            s,
            "    \"{k}\": {{\"shape\": [{}], \"bits\": [{}]}}{}",
            sh.join(","),
            bits.join(","),
            if i + 1 < keys.len() { "," } else { "" }
        )
        .unwrap();
    }
    let y = m.predict(&x).unwrap().contiguous();
    let yb: Vec<String> = y
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect();
    let xs: Vec<String> = x.shape().iter().map(|d| d.to_string()).collect();
    writeln!(
        s,
        "  }},\n  \"input_shape\": [{}],\n  \"predict_bits\": [{}]\n}}",
        xs.join(","),
        yb.join(",")
    )
    .unwrap();
    std::fs::write(format!("{dir}/expected.json"), s).unwrap();
}

fn main() {
    fandhe_ai::manual_seed(0);
    let out = std::env::args().nth(1).unwrap();
    let mlp = Sequential::new()
        .add_linear(4, 6, 1)
        .unwrap()
        .add_batch_norm1d(6, 1e-5, 0.1)
        .unwrap()
        .add_relu()
        .add_dropout(0.25)
        .unwrap()
        .add_linear(6, 6, 2)
        .unwrap()
        .add_layer_norm(6, 1e-5)
        .unwrap()
        .add_linear(6, 2, 3)
        .unwrap();
    emit("mlp_bn", mlp, input(&[5, 4], 0.1), &out);
    let cnn = Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, 4)
        .unwrap()
        .add_batch_norm2d(2, 1e-5, 0.1)
        .unwrap()
        .add_relu()
        .add_max_pool2d([2, 2], None, [0, 0], [1, 1])
        .unwrap()
        .add_flatten(1, 3)
        .add_linear(8, 3, 5)
        .unwrap();
    emit("cnn", cnn, input(&[3, 1, 4, 4], 0.2), &out);
}
```

## sha256

```text
1fe5519ff724029def1f9ba22cb6cdeb74d61a21dc664981ce40af0cda7608c1  cnn/expected.json
ccc67fa7e94fbb94f2a6eea3d399345b84c92ac155b2c797404062fac7e7c999  cnn/manifest.json
cc540f4da4986166e7dda990882249ff61ed8e58d6452bbbca542dfe0d9d7240  cnn/model.99c023f9e717f348d78abcef88a7c834.safetensors
ec743784ccdbfe6ba0fb779550c1771d528b799312753fae3db347aa91c4f804  mlp_bn/expected.json
bca6c7c5e727abaf9b405af90ec5accd3d7ae1a2e2b78a736b6811209133be51  mlp_bn/manifest.json
1de0ad6ac468545773bd9d9ca619a9e73d769c513c4a1e31c4b507f8f61e8630  mlp_bn/model.a82545ed12bf7bfa98c3069cd760619a.safetensors
```

## 注意

- 0.13.0 以前のリーダーは新しい kind を `UnsupportedModel` で拒否します（前方互換なし。`docs/migration/0.14.0.md` §3.4）。
- fixture は非信頼入力として `load_model` の検証経路（固定スキーマ・正準表記・上限・no-follow）をそのまま通ります。
- fixture の更新は人間承認が必要です（baseline 追加と同様に、実測値のみを置く）。
