# Neural Simulator

This small project of mine is my excuse to get rusty. I want to learn some Rust basics, so I thought that implementing a simple neural simulator is a great excuse for this.

## Repository structure

- analysis/             # Python analysis scripts for visualization and data analysis of the results of the simulation
- output/               # Output files of specific runs (csv-files, plots, etc.)
- src/                  # Rust neural simulator code
- default_params.toml   # Parameter settings to be changed for specific runs
- main.sh               # Wrapper shell script to execute the simulation and python analysis in one complete run.

## Version 0.2.2 - Robust multi-neuron-type implementation

In the file `neurons.rs`, there are now all neuron-definitions together with all the setup needed to integrate with the main simulation. The `config.rs` code sets up the corresponding namespace for easy selection of the correct neurons through the config file. All of this is handled using enum operations to select between the different versions, making it easily extendable. 

## Version 0.2.1. - Getting towards multiple neuron types

A next exciting step is to make the simulator more general. In principle I want to move a way from one single neuron type with one single population connectivity.
Rather, I would like to spawn an arbitrary amount of populations that each have one of many neuron-types that could be implemented.
To get towards that goal, I: 

- [x] Implemented a refractory LIF (RefLIF) neuron as a second neuron type and a test case
- [x] Created more general coding structure to have a trait 'neuron' that is handled by the neural population (prev. neural field) class.
- [x] Tested the simulation for both LIF and RefLIF neuron types, it is now as simple as switching a single line of code to construct a different neuron type.

### Next step in the 0.2.x release

- [x] In the future, I would like to declare this with just a single flag in the config file, so that would be the immediate next open to-do. 

### Towards 0.3.x

After having this implemented, I think it would be great to try and figure out, whether we can have multiple neuron populations talking to each other. But actually, a first step in that direction would most likely be defining inputs and weights externally. After that milestone, we have a truly general neural population simulator and we can think about hooking up multiple populations.

---

## Version 0.1 - A first simulator

For a very first simulator version, this seems to be quite promising already. Here is a list of things that are implemented in the very first version:

- [x] Implementation of a first neuron model type (LIF-neuron)
- [x] Implementation of a simple neuron-population using pre-defined weights
- [x] Implementation of typical ring-attractor connectivity, like gaussian and mexican hat.
- [x] Implementation of the main simulation loop
- [x] CSV-file link between rust simulation and simple python visualization. 
- [x] Parameter definition through TOML-file. 

Essentially, you can specify a simulation in a TOML file, then run the main.sh script that will execute the rust simulation, which stores the results of the run to CSV, where a python script then fetches the results and creates a very simple visualization of the activity. I think that

## Next features / wishlist

There is a bunch of ideas I would have for the future of this project. 


- [ ] Easily configurable simulation (e.g., for parameter sweeps or just simply setting up a simulation in a config).
- [ ] Parallelization of the simulation loop
- [ ] Interactive visualization tool (live interaction with inputs)
- [ ] Introducing multiple neuron model types, like a resonate and fire neuron or an adaptive lif
- [ ] Allowing for multiple populations and rings to interact with arbitrary weights
- [ ] Allow simple learning rules for dynamic weight matrices