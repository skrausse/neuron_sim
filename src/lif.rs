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