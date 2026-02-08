use crate::lif::LIF;
use crate::utils::calc_index;

pub struct NeuralField {
    pub num_neurons: usize,
    pub population: Vec<LIF>,       // size (num_neurons)
    pub spike_buffer_a: Vec<bool>,  // size (num_neurons)
    pub spike_buffer_b: Vec<bool>,  // size (num_neurons)
    pub buffer_a_is_prev: bool,     // To avoid allocating memory, we do A-B buffering
    pub weights: Vec<f32>,          // size (num_neurons x num_neurons)
}

//--------------------------------------------------------------------------------------------------


impl NeuralField {
    pub fn new(num_neurons: usize, v_rest: f32, v_thresh: f32, tau: f32, weights: Vec<f32>) -> Self {
        Self {
            num_neurons: num_neurons, 
            population: vec![LIF::new(v_rest, v_thresh, tau); num_neurons], 
            spike_buffer_a: vec![false; num_neurons], 
            spike_buffer_b: vec![false; num_neurons], 
            buffer_a_is_prev: true,
            weights: weights,
        }
    }

    pub fn step(&mut self, dt: f32, external_current: &[f32]) {
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
        let num_neurons: &usize = &*num_neurons;

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
                if read_buffer[source] {
                    internal_current = internal_current 
                                       + weights[calc_index(source, target, *num_neurons)];
                }
            }

            let input_current: f32 = internal_current + external_current[target];
            write_buffer[target] = population[target].step(dt, input_current);
        }
        
        *buffer_a_is_prev = !*buffer_a_is_prev; // flip flag
    }
}