pub fn softmax(input: &Vec<f64>) -> Vec<f64> {
    let sum: f64 = input.iter().map(|x| x.exp()).sum();
    input.iter().map(|x| x.exp() / sum).collect()
}

pub fn softmax_derivative(input: &Vec<f64>) -> Vec<f64> {
    let softmax = softmax(input);
    softmax.iter().zip(softmax.iter()).map(|(x, y)| x * (1.0 - y)).collect()
}

pub fn relu(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|x| x.max(0.0)).collect()
}

pub fn relu_derivative(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect()
}

pub fn sigmoid(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
}

pub fn sigmoid_derivative(input: &Vec<f64>) -> Vec<f64> {
    let sigmoid = sigmoid(input);
    sigmoid.iter().zip(sigmoid.iter()).map(|(x, y)| x * (1.0 - y)).collect()
}

pub fn tanh(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|x| x.tanh()).collect()
}

pub fn tanh_derivative(input: &Vec<f64>) -> Vec<f64> {
    let tanh = tanh(input);
    tanh.iter().zip(tanh.iter()).map(|(x, y)| 1.0 - y.powi(2)).collect()
}