use crate::utils::fs::{file_exists, read_lines};
use rand::seq::SliceRandom;
use std::io;

#[derive(Debug)]
pub struct Dataset {
    pub training: Vec<(Vec<f64>, Vec<f64>)>,
    pub validation: Vec<(Vec<f64>, Vec<f64>)>,
    pub test: Vec<(Vec<f64>, Vec<f64>)>,
}

pub fn load_from_csv(
    file_path: &str,
    split: (f32, f32, f32),
    features: Vec<&str>,
    target: &str,
    shuffle: bool,
) -> io::Result<Dataset> {
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();

    if split.0 + split.1 + split.2 > 1.0 || split.0 < 0.0 || split.1 < 0.0 || split.2 < 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid split values",
        ));
    }

    if !file_exists(file_path) {
        return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
    }

    let mut lines = read_lines(file_path)?;
    let headers: Vec<String> = lines.next().unwrap().unwrap().split(',').map(|s| s.to_string()).collect();
    let feature_indices: Vec<usize> = features.iter().map(|&f| {
        headers.iter().position(|h| h == f).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Feature '{}' not found in headers", f))
        })
    }).collect::<Result<Vec<_>, _>>()?;
    let target_index = headers.iter().position(|h| h == target).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("Target '{}' not found in headers", target))
    })?;

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
            let label = row[target_index];
            let features: Vec<f64> = feature_indices.iter().map(|&i| row[i]).collect();

            let rand = rand::random::<f32>();
            if rand < split.0 {
                training.push((features, vec![label]));
            } else if rand < split.0 + split.1 {
                validation.push((features, vec![label]));
            } else {
                test.push((features, vec![label]));
            }
        }
    }

    if shuffle {
        training.shuffle(&mut rand::thread_rng());
        validation.shuffle(&mut rand::thread_rng());
        test.shuffle(&mut rand::thread_rng());
    }

    Ok(Dataset {
        training,
        validation,
        test,
    })
}
