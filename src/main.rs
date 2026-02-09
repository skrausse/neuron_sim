mod lif;
mod weights;
mod field;
mod utils;
mod config;

use field::NeuralField;
use std::fs;
use crate::{utils::pulse_at_pos, weights::get_mexican_hat_weights_circ};
use utils::{pulse_at_center, save_to_csv};
use config::Config;

//--------------------------------------------------------------------------------------------------
//-----------------------------         MAIN            --------------------------------------------
//--------------------------------------------------------------------------------------------------

fn main() {
    // Read config file
    // Read config file
    let config_content = fs::read_to_string("config.toml")
        .expect("Failed to read config.toml");
    let config: Config = toml::from_str(&config_content)
        .expect("Failed to parse TOML");
    
    // Define neural population and connectivity
    let weights: Vec<f32> = get_mexican_hat_weights_circ(config.neuron.num_neurons, 
                                                         config.weights.strength_e, 
                                                         config.weights.sigma_e,
                                                         config.weights.strength_i, 
                                                         config.weights.sigma_i);

    let mut field: NeuralField = NeuralField::new(config.neuron.num_neurons, 
                                                  config.neuron.v_rest, 
                                                  config.neuron.v_thresh, 
                                                  config.neuron.tau, 
                                                  weights);
    
    // define current pulses
    let background_current: Vec<f32> = vec![config.simulation.background_strength; 
                                            config.neuron.num_neurons];
    let pulse_current = pulse_at_center(config.simulation.first_pulse_strength, 
                                                  config.neuron.num_neurons);
    let second_pulse_current = pulse_at_pos(config.simulation.second_pulse_strength, 
                                                      80, 
                                                      config.neuron.num_neurons);

    // Buffer for result logging
    let mut voltage_buffer: Vec<Vec<f32>> = Vec::with_capacity(config.simulation.num_timesteps);
    let mut spiking_buffer: Vec<Vec<i32>> = Vec::with_capacity(config.simulation.num_timesteps);

    // Simulation loop
    for index in 0..config.simulation.num_timesteps {
        if index<config.simulation.stimulation_length { 
            field.step(config.simulation.dt, &pulse_current);
        } else if  (index > 500) && (index < 500 + config.simulation.stimulation_length) {
            field.step(config.simulation.dt, &second_pulse_current)
        } else { 
            field.step(config.simulation.dt, &background_current); 
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
