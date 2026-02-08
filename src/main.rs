use std::fs::File;
use std::io::{Write, BufWriter};

//--------------------------------------------------------------------------------------------------
//-------------------       Simple LIF neuron implementation        --------------------------------
//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct LIF {
    v: f32,
    v_rest: f32,
    v_thresh: f32,
    tau: f32,
}

//--------------------------------------------------------------------------------------------------

impl LIF {
    fn new(v_rest: f32, v_thresh: f32, tau: f32) -> Self {
        Self {
            v: v_rest,
            v_rest: v_rest,
            v_thresh: v_thresh,
            tau: tau,
        }
    }

    fn step(&mut self, dt: f32, input_current: f32) -> bool {
        let mut spike: bool = false;
        self.v = self.v + (dt / self.tau) * (self.v_rest - self.v) + input_current;
        if self.v > self.v_thresh {
            self.v = self.v_rest;
            spike = true;
        }
        spike
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------      Neural Population implementation        --------------------------------
//--------------------------------------------------------------------------------------------------

struct NeuralField {
    num_neurons: i32,
    population: Vec<LIF>,       // size (num_neurons)
    spike_buffer_a: Vec<bool>,  // size (num_neurons)
    spike_buffer_b: Vec<bool>,  // size (num_neurons)
    buffer_a_is_prev: bool,     // To avoid allocating memory, we do A-B buffering
    weights: Vec<f32>,          // size (num_neurons x num_neurons)
}

//--------------------------------------------------------------------------------------------------

#[allow(dead_code)]
impl NeuralField {
    fn new(num_neurons: i32, v_rest: f32, v_thresh: f32, tau: f32) -> Self {
        Self {
            num_neurons: num_neurons, 
            population: vec![LIF::new(v_rest, v_thresh, tau); num_neurons as usize], 
            spike_buffer_a: vec![false; num_neurons as usize], 
            spike_buffer_b: vec![false; num_neurons as usize], 
            buffer_a_is_prev: true,
            weights: vec![0.0; (num_neurons * num_neurons) as usize]} // (target, source)
    }

    fn set_gaussian_weights(&mut self, strength: f32, sigma: f32) {
        for source in 0..self.num_neurons {
            for target in 0..self.num_neurons {
                let distance: f32 = ((source - target) as f32).abs();
                let weight: f32 = strength * (-distance.powi(2) / (2.0*sigma.powi(2))).exp();
                self.weights[calc_index(source, target, self.num_neurons)] = weight;
            }
        }
    }

    fn set_mexican_hat_weights(&mut self, strength_e: f32, sigma_e: f32, 
                                          strength_i: f32, sigma_i: f32) {
        for source in 0..self.num_neurons {
            for target in 0..self.num_neurons {
                let distance: f32 = ((source - target) as f32).abs();
                let weight_e: f32 = strength_e * (-distance.powi(2) / (2.0*sigma_e.powi(2))).exp();
                let weight_i: f32 = strength_i * (-distance.powi(2) / (2.0*sigma_i.powi(2))).exp();
                self.weights[calc_index(source, target, self.num_neurons)] = weight_e - weight_i;
            }
        }
    }

    fn set_mexican_hat_weights_circ(&mut self, strength_e: f32, sigma_e: f32, 
                                               strength_i: f32, sigma_i: f32) {
        for source in 0..self.num_neurons {
            for target in 0..self.num_neurons {
                // compute circular distance here
                let mut distance: f32 = ((source - target) as f32).abs();
                distance = distance.min(self.num_neurons as f32 - distance);

                let weight_e: f32 = strength_e * (-distance.powi(2) / (2.0*sigma_e.powi(2))).exp();
                let weight_i: f32 = strength_i * (-distance.powi(2) / (2.0*sigma_i.powi(2))).exp();
                self.weights[calc_index(source, target, self.num_neurons)] = weight_e - weight_i;
            }
        }
    }


    fn step(&mut self, dt: f32, external_current: &Vec<f32>) {
        // Unpack fields from self to allow independent borrowing
        let NeuralField {
            num_neurons,
            population,
            spike_buffer_a,
            spike_buffer_b,
            buffer_a_is_prev,
            weights,
        } = self;

        // declare immutable data fields
        let weights: &Vec<f32> = &*weights;
        let num_neurons: &i32 = &*num_neurons;

        // Now decide which is read and which is write using the unpacked variables
        let (read_buffer, write_buffer) = if *buffer_a_is_prev {
            (&*spike_buffer_a, &mut *spike_buffer_b)
        } else {
            (&*spike_buffer_b, &mut *spike_buffer_a)
        };

        // Use *num_neurons because num_neurons is now a reference to the value
        for target in 0..*num_neurons {
            let mut internal_current: f32 = 0.0;
            for source in 0..*num_neurons {
                if read_buffer[source as usize] {
                    internal_current = internal_current 
                                       + weights[calc_index(source, target, *num_neurons)];
                }
            }

            let input_current: f32 = internal_current + external_current[target as usize];
            write_buffer[target as usize] = population[target as usize].step(dt, input_current);
        }
        
        *buffer_a_is_prev = !*buffer_a_is_prev; // flip flag
    }
}


//--------------------------------------------------------------------------------------------------
//-----------------------       Helper Functions       ---------------------------------------------
//--------------------------------------------------------------------------------------------------

fn calc_index(source: i32, target: i32, num_neurons: i32) -> usize {
    (target * num_neurons + source) as usize
}

fn pulse_at_pos(strength: f32, pos: usize, num_neurons:i32) -> Vec<f32> {
    let mut external_current: Vec<f32> = vec![0.0; num_neurons as usize];
    external_current[pos] = strength;
    external_current
}

fn pulse_at_center(strength: f32, num_neurons: i32) -> Vec<f32> {
    let pos: usize = (num_neurons / 2) as usize;
    let external_current = pulse_at_pos(strength, pos, num_neurons);
    external_current
}

fn save_to_csv<T: std::fmt::Display>(data: &Vec<Vec<T>>, filename: &str) {
    let file = File::create(filename).expect("Unable to create file.");
    let mut writer = BufWriter::new(file);

    for row in data {
        let rowstring: String = row.iter().map(|element| element.to_string())
                                          .collect::<Vec<String>>()
                                          .join(",");
        writeln!(writer, "{}", rowstring).expect("Unable to write data row.");
    }
}

//--------------------------------------------------------------------------------------------------
//-----------------------------         MAIN            --------------------------------------------
//--------------------------------------------------------------------------------------------------

fn main() {
    // neural population parameters
    let num_neurons:i32 = 100;
    let v_rest:f32 = 0.0;
    let v_thresh:f32 = 40.0;
    let tau:f32 = 5.0;

    // Simulation parameters
    let dt: f32 = 1.0;
    let num_timesteps: i32 = 1000;
    
    // Define neural population and connectivity
    let mut field: NeuralField = NeuralField::new(num_neurons, v_rest, v_thresh, tau);
    field.set_mexican_hat_weights_circ(200.0, 3.0, 50.0, 30.0);

    // define current pulses
    let background_current: Vec<f32> = vec![0.0; num_neurons as usize];
    let pulse_current = pulse_at_center(50.0, num_neurons);

    // Buffer for result logging
    let mut voltage_buffer: Vec<Vec<f32>> = Vec::with_capacity(num_timesteps as usize);
    let mut spiking_buffer: Vec<Vec<i32>> = Vec::with_capacity(num_timesteps as usize);

    // Simulation loop
    for index in 0..num_timesteps {
        if index == 0 { 
            field.step(dt, &pulse_current);
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
