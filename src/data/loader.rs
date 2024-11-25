use crate::utils::fs::{file_exists, read_lines};
use rand_core::SeedableRng;
use rand::Rng;
use std::io;


pub fn load_from_csv(
    file_path: &str,
    features: Vec<String>,
    targets: Vec<String>,
    shuffle: bool,
    random_seed: Option<u64>,
) -> io::Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let mut X = vec![];
    let mut Y = vec![];

    if !file_exists(file_path) {
        return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
    }

    let mut lines = read_lines(file_path)?;

    let headers: Vec<String> = lines.next().unwrap().unwrap().split(',').map(|s| s.to_string()).collect();

    let feature_indices: Vec<usize> = features.iter().map(|feature| {
        headers.iter().position(|column| column == feature.as_str()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Feature '{}' not found in headers", feature))
        })
    }).collect::<Result<Vec<_>, _>>()?;


    let target_indices: Vec<usize> = targets.iter().map(|target| {
        headers.iter().position(|column| column == target.as_str()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Target '{}' not found in headers", target))
        })
    }).collect::<Result<Vec<_>, _>>()?;

    for line in lines {
        if let Ok(line) = line {

            let row: Vec<f64> = line.split(",").map(|x| {
                if let Ok(x) = x.parse::<f64>() {
                    x
                } else {
                    0.0
                }
            }).collect();

            if row.len() != headers.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Row length does not match header length"));
            }
            let labels:Vec<f64>  = target_indices.iter().map(|&index| row[index]).collect();
            let features: Vec<f64> = feature_indices.iter().map(|&index| row[index]).collect();

            X.push(features);
            Y.push(labels);
        }
    }

    if shuffle {
        let mut rng = rand::rngs::StdRng::seed_from_u64(random_seed.unwrap_or(0));
        
        for i in 0..X.len() {
            let random_index = rng.gen_range(0..X.len());
            // do a swap
            X.swap(i, random_index);
            Y.swap(i, random_index);
        }
    }

    Ok((X, Y))
}
