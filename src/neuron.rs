pub trait Neuron {
    fn step(&mut self, dt: f32, input_current: f32) -> bool;
    fn get_voltage(&self) -> f32;
}

//--------------------------------------------------------------------------------------------------
//-------------------   Leaky integrate and Fire Neuron    -----------------------------------------
//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct LIF {
    pub v: f32,
    v_rest: f32,
    v_thresh: f32,
    tau: f32,
}

//--------------------------------------------------------------------------------------------------

impl LIF {
    pub fn new(v_rest: f32, v_thresh: f32, tau: f32) -> Self {
        Self {
            v: v_rest,
            v_rest: v_rest,
            v_thresh: v_thresh,
            tau: tau,
        }
    }

    pub fn step(&mut self, dt: f32, input_current: f32) -> bool {
        let mut spike: bool = false;
        self.v = self.v + (dt / self.tau) * (self.v_rest - self.v + input_current);
        if self.v > self.v_thresh {
            self.v = self.v_rest;
            spike = true;
        }
        spike
    }
}

//--------------------------------------------------------------------------------------------------

impl Neuron for LIF {
    fn step(&mut self, dt: f32, input_current: f32) -> bool {
        self.step(dt, input_current)
    }

    fn get_voltage(&self) -> f32 {
        self.v
    }
}

//--------------------------------------------------------------------------------------------------
//-------------------   Refractory Leaky integrate and Fire Neuron    ------------------------------
//--------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct RefLIF {
    pub v: f32,
    pub tau_ref: f32,
    v_rest: f32,
    v_thresh: f32,
    tau: f32,
    t_ref: f32, 
}

//--------------------------------------------------------------------------------------------------

impl RefLIF {
    pub fn new(v_rest: f32, v_thresh: f32, tau: f32, tau_ref:f32) -> Self {
        Self {
            v: v_rest,
            tau_ref: tau_ref,
            v_rest: v_rest,
            v_thresh: v_thresh,
            tau: tau,
            t_ref: -1.0,
        }
    }

    pub fn step(&mut self, dt: f32, input_current: f32) -> bool {
        let mut spike: bool = false;

        // Update refractory time:
        self.t_ref -= dt;

        // Update voltage when not in refractory time.
        if self.t_ref < 0.0 {
            self.v = self.v + (dt / self.tau) * (self.v_rest - self.v + input_current);
        }

        // If spike, reset refractory time and voltage and emit a spike
        if self.v > self.v_thresh {
            self.t_ref = self.tau_ref;
            self.v = self.v_rest;
            spike = true;
        }
        spike
    }
}

//--------------------------------------------------------------------------------------------------

impl Neuron for RefLIF {
    fn step(&mut self, dt: f32, input_current: f32) -> bool {
        self.step(dt, input_current)
    }

    fn get_voltage(&self) -> f32 {
        self.v
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

// Define a general NeuronType enum object that handles the selection between different types
pub enum NeuronType {
    Lif(LIF),
    Ref(RefLIF),
}

impl Neuron for NeuronType {
    fn step(&mut self, dt: f32, input_current: f32) -> bool {
        match self {
            Self::Lif(n) => n.step(dt, input_current),
            Self::Ref(n) => n.step(dt, input_current),
        }
    }
    fn get_voltage(&self) -> f32 {        
        match self {
            Self::Lif(n) => n.get_voltage(),
            Self::Ref(n) => n.get_voltage(),
        }
    }
}