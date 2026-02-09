use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub neuron: NeuronParams,
    pub weights: WeightParams,
    pub simulation: SimParams,
}

#[derive(Deserialize)]
pub struct NeuronParams {
    pub num_neurons: usize,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub tau: f32,
}

#[derive(Deserialize)]
pub struct WeightParams {
    pub strength_e: f32,
    pub strength_i: f32,
    pub sigma_e: f32,
    pub sigma_i: f32,
}

#[derive(Deserialize)]
pub struct SimParams {
    pub dt: f32,
    pub num_timesteps: usize,
    pub stimulation_length: usize,
    pub background_strength: f32,
    pub first_pulse_strength: f32,
    pub second_pulse_strength: f32,
}