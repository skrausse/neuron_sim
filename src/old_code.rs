fn single_neuron(){

    let v_rest: f32 = 0.0;
    let v_thresh: f32 = 40.0;
    let tau: f32 = 10.0;
    
    let mut neuron: LIF = LIF::new(v_rest, v_thresh, tau);

    let input: f32 = 5.0;
    let dt: f32 = 0.1;
    let mut spike: bool;

    for _ in 0..100 {
        spike = neuron.step(dt, input);
        println!("Voltage: {}", neuron.v);
        if spike {
            println!("Spike emitted!!\n");
        }
    }
}