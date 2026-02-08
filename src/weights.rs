use crate::utils::calc_index;

#[allow(dead_code)]
pub fn get_gaussian_weights(num_neurons: usize, strength: f32, sigma: f32) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; num_neurons * num_neurons];
    for source in 0..num_neurons {
        for target in 0..num_neurons {
            let distance: f32 = (source as f32 - target as f32).abs();
            let weight: f32 = strength * (-distance.powi(2) / (2.0*sigma.powi(2))).exp();
            weights[calc_index(source, target, num_neurons)] = weight;
        }
    }
    weights 
}

#[allow(dead_code)]
pub fn get_mexican_hat_weights(num_neurons: usize, strength_e: f32, sigma_e: f32, 
                                                   strength_i: f32, sigma_i: f32) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; num_neurons * num_neurons];
    for source in 0..num_neurons {
        for target in 0..num_neurons {
            let distance: f32 = (source as f32 - target as f32).abs();
            let weight_e: f32 = strength_e * (-distance.powi(2) / (2.0*sigma_e.powi(2))).exp();
            let weight_i: f32 = strength_i * (-distance.powi(2) / (2.0*sigma_i.powi(2))).exp();
            weights[calc_index(source, target, num_neurons)] = weight_e - weight_i;
        }
    }
    weights
}

#[allow(dead_code)]
pub fn get_mexican_hat_weights_circ(num_neurons: usize, strength_e: f32, sigma_e: f32, 
                                                   strength_i: f32, sigma_i: f32) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; num_neurons * num_neurons];
    for source in 0..num_neurons {
        for target in 0..num_neurons {
            let mut distance: f32 = (source as f32 - target as f32).abs();
            distance = distance.min(num_neurons as f32 - distance);

            let weight_e: f32 = strength_e * (-distance.powi(2) / (2.0*sigma_e.powi(2))).exp();
            let weight_i: f32 = strength_i * (-distance.powi(2) / (2.0*sigma_i.powi(2))).exp();
            weights[calc_index(source, target, num_neurons)] = weight_e - weight_i;
        }
    }
    weights
}
