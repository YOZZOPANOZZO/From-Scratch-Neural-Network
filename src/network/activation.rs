pub trait Activation {
    fn forward(&self, input: f64) -> f64;
    fn backward(&self, input: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn backward(&self, input: f64) -> f64 {
        let sigmoid = self.forward(input);
        sigmoid * (1.0 - sigmoid)
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, input: f64) -> f64 {
        input.max(0.0)
    }

    fn backward(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct Tanh;

impl Activation for Tanh {
    fn forward(&self, input: f64) -> f64 {
        input.tanh()
    }

    fn backward(&self, input: f64) -> f64 {
        let tanh = self.forward(input);
        1.0 - tanh.powi(2)
    }
}

pub struct Softmax;

impl Activation for Softmax {
    fn forward(&self, input: f64) -> f64 {
        input.exp()
    }

    fn backward(&self, input: f64) -> f64 {
        let softmax = self.forward(input);
        softmax * (1.0 - softmax)
    }
}

pub struct Linear;

impl Activation for Linear {
    fn forward(&self, input: f64) -> f64 {
        input
    }

    fn backward(&self, _input: f64) -> f64 {
        1.0
    }
}
