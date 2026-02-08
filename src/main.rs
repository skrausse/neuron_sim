use crate::{utils::pulse_at_pos, weights::get_mexican_hat_weights_circ};

mod lif;
mod weights;
mod field;
mod utils;

use field::NeuralField;
use utils::{pulse_at_center, save_to_csv};

//--------------------------------------------------------------------------------------------------
//-----------------------------         MAIN            --------------------------------------------
//--------------------------------------------------------------------------------------------------

fn main() {
    // neural population parameters
    let num_neurons:usize = 100;
    let v_rest:f32 = 0.0;
    let v_thresh:f32 = 40.0;
    let tau:f32 = 10.0;
    let strength_e:f32 = 250.0;
    let strength_i:f32 = 50.0;
    let sigma_e:f32 = 10.0;
    let sigma_i:f32 = 40.0;
    let stimulation_length: usize = 100;

    // Simulation parameters
    let dt: f32 = 1.0;
    let num_timesteps: usize = 1000;
    
    // Define neural population and connectivity
    let weights: Vec<f32> = get_mexican_hat_weights_circ(num_neurons, strength_e, sigma_e, 
                                                                      strength_i, sigma_i);
    let mut field: NeuralField = NeuralField::new(num_neurons, v_rest, v_thresh, tau, weights);
    
    // define current pulses
    let background_current: Vec<f32> = vec![0.0; num_neurons];
    let pulse_current = pulse_at_center(50.0, num_neurons);
    let second_pulse_current = pulse_at_pos(50.0, 80, num_neurons);

    // Buffer for result logging
    let mut voltage_buffer: Vec<Vec<f32>> = Vec::with_capacity(num_timesteps);
    let mut spiking_buffer: Vec<Vec<i32>> = Vec::with_capacity(num_timesteps);

    // Simulation loop
    for index in 0..num_timesteps {
        if index<stimulation_length { 
            field.step(dt, &pulse_current);
        } else if  (index > 500) && (index < 500 + stimulation_length) {
            field.step(dt, &second_pulse_current)
        } else { 
            field.step(dt, &background_current); 
        }
        
        // Store voltage in buffer
        let voltage_at_t: Vec<f32> = field.population.iter().map(|neuron| neuron.v).collect();
        voltage_buffer.push(voltage_at_t);

        // Store spiking activity to buffer
        let spikes_at_t: Vec<i32> = if field.buffer_a_is_prev {
            field.spike_buffer_b.iter().map(|&s| s as i32).collect()
        } else {
            field.spike_buffer_a.iter().map(|&s| s as i32).collect()
        };

        spiking_buffer.push(spikes_at_t);
    };

    save_to_csv(&voltage_buffer, &"output/voltage_data.csv");
    save_to_csv(&spiking_buffer, &"output/spiking_data.csv");

}
