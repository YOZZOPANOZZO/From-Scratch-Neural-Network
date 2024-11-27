mod data;
mod utils;

use data::loader;

fn main() {
    let file_path = "assets/ML-CUP24-TR.csv";
    let features: Vec<String> = (1..=11).map(|x| format!("FEATURE_{:?}", x)).collect();
    let targets: Vec<String> = vec!["x", "y", "z"]
        .into_iter()
        .map(|x| format!("TARGET_{}", x))
        .collect();

    println!("Loading data from file: {}...", file_path);
    println!("Features: {:?}", features);
    println!("Targets: {:?}", targets);

    let (X_train, X_test, Y_train, Y_test) =
        loader::load_from_csv(file_path, features, targets, true, Some(42), 0.8).unwrap();

    println!("X_train: {:?}", X_train.len());
    println!("Y_train: {:?}", Y_train.len());
    println!("X_test: {:?}", X_test.len());
    println!("Y_test: {:?}", Y_test.len());

    println!("✅ Data loaded successfully!");
}
