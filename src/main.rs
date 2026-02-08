#[derive(Clone, Copy)]
struct LIF {
    v: f32,
    v_rest: f32,
    v_thresh: f32,
    tau: f32,
}

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

struct NeuralField {
    num_neurons: i32,
    population: Vec<LIF>,       // size (num_neurons)
    spike_buffer_a: Vec<bool>,  // size (num_neurons)
    spike_buffer_b: Vec<bool>,  // size (num_neurons)
    buffer_a_is_prev: bool,     // To avoid allocating memory, we do A-B buffering
    weights: Vec<f32>,          // size (num_neurons x num_neurons)
}

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

fn calc_index(source: i32, target: i32, num_neurons: i32) -> usize {
    (target * num_neurons + source) as usize
}

fn main() {
    let num_neurons:i32 = 100;
    let v_rest:f32 = 0.0;
    let v_thresh:f32 = 40.0;
    let tau:f32 = 10.0;

    let mut field: NeuralField = NeuralField::new(num_neurons, v_rest, v_thresh, tau);
    field.set_gaussian_weights(1.0, 1.0);

    let dt: f32 = 0.1;
    let external_current: Vec<f32> = vec![5.0; num_neurons as usize];
    
    for _ in 0..1000 {
        field.step(dt, &external_current);
        println!("Voltage of Neuron 1: {}", field.population[0].v)
    };
}
