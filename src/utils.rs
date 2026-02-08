use std::fs::File;
use std::io::{Write, BufWriter};

//--------------------------------------------------------------------------------------------------

pub fn calc_index(source: usize, target: usize, num_neurons: usize) -> usize {
    target * num_neurons + source
}

//--------------------------------------------------------------------------------------------------

pub fn pulse_at_pos(strength: f32, pos: usize, num_neurons:usize) -> Vec<f32> {
    let mut external_current: Vec<f32> = vec![0.0; num_neurons];
    external_current[pos] = strength;
    external_current
}

//--------------------------------------------------------------------------------------------------

pub fn pulse_at_center(strength: f32, num_neurons: usize) -> Vec<f32> {
    let pos: usize = num_neurons / 2;
    let external_current = pulse_at_pos(strength, pos, num_neurons);
    external_current
}

//--------------------------------------------------------------------------------------------------

pub fn save_to_csv<T: std::fmt::Display>(data: &Vec<Vec<T>>, filename: &str) {
    let file = File::create(filename).expect("Unable to create file.");
    let mut writer = BufWriter::new(file);

    for row in data {
        let rowstring: String = row.iter().map(|element| element.to_string())
                                          .collect::<Vec<String>>()
                                          .join(",");
        writeln!(writer, "{}", rowstring).expect("Unable to write data row.");
    }
}