mod network {
    pub mod activation;
}

use network::activation;

fn main() {
    let input = vec![1.0, 2.0, 3.0];
    let output = activation::softmax(&input);
    println!("{:?} ==== softmaxed ====> {:?}", input, output);
}