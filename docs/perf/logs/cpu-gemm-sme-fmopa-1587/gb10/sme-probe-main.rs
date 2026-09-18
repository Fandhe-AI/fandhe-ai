// イシュー #1978 GB10 R0: 実行 CPU の SME 検出結果（fail-closed）を記録する使い捨てプローブ。
fn main() {
    println!("sme_report={:?}", fandhe_ai_backend_cpu::sme_report());
}
