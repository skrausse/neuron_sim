use crate::neuron::Neuron;
use crate::utils::calc_index;



pub struct Population<N: Neuron> {
    pub num_neurons: usize,
    pub neurons: Vec<N>,       // size (num_neurons)
    pub spike_buffer_a: Vec<bool>,  // size (num_neurons)
    pub spike_buffer_b: Vec<bool>,  // size (num_neurons)
    pub buffer_a_is_prev: bool,     // To avoid allocating memory, we do A-B buffering
    pub weights: Vec<f32>,          // size (num_neurons x num_neurons)
}

//--------------------------------------------------------------------------------------------------


impl <N: Neuron> Population<N> {
    pub fn new<F>(num_neurons: usize, weights: Vec<f32>, mut constructor: F) -> Self 
    where F: FnMut() -> N 
    {
        let mut neurons = Vec::with_capacity(num_neurons);
        for _ in 0..num_neurons {
            neurons.push(constructor());
        }

    Self {
            num_neurons: num_neurons, 
            neurons: neurons,
            spike_buffer_a: vec![false; num_neurons], 
            spike_buffer_b: vec![false; num_neurons], 
            buffer_a_is_prev: true,
            weights: weights,
        }
    }

    pub fn step(&mut self, dt: f32, external_current: &[f32]) {
        // Unpack fields from self to allow independent borrowing
        let Population {
            num_neurons,
            neurons,
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
            write_buffer[target] = neurons[target].step(dt, input_current);
        }
        
        *buffer_a_is_prev = !*buffer_a_is_prev; // flip flag
    }
}