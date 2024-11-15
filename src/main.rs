mod data {
    pub mod loader;
}

mod utils {
    pub mod fs;
}

use data::loader;

fn main() {
    let dataset = loader::load_from_csv(
        "assets/Pokemon.csv", // download from https://www.kaggle.com/datasets/abcsds/pokemon?resource=download
        (0.5, 0.25, 0.25), 
        vec!["Attack", "Defense", "Speed", "Sp. Atk", "Sp. Def"],
        "HP",
        false
    ).unwrap();

    let training_size = dataset.training.len();
    let validation_size = dataset.validation.len();
    let test_size = dataset.test.len();
    

    println!("Training size: {}", training_size);
    println!("Validation size: {}", validation_size);
    println!("Test size: {}", test_size);

    println!("{:?}", dataset);
}
