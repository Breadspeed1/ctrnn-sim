use burn::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use mnist_sim::{MnistEnvironment, tui_logger::Metrics};
use sim_core::Runner;
use sim_core::SimLogger;
use sim_core::brain::ctrnn::{CtrnnBrain, MutationConfig};
use std::sync::{Arc, Mutex};

type Backend = NdArray;
const N_TOTAL: usize = 10;
const BATCH_SIZE: usize = 100; // Smaller for investigation
const CONCURRENT_SIMS: usize = 10;

struct SimpleLogger {
    generation: usize,
}

impl<B: burn::prelude::Backend<FloatElem = f32>, ES, BS, A> SimLogger<B, ES, BS, A>
    for SimpleLogger
{
    fn record_env(&mut self, _state: ES, _generation: usize) {}
    fn record_ephemeral_brain(&mut self, _state: BS, _generation: usize) {}
    fn record_brain(&mut self, _brain: A, _generation: usize) {}
    fn record_fitness(&mut self, fitness: Tensor<B, 2>, generation: usize) {
        let avg: f32 = fitness.clone().mean().into_scalar();
        let max: f32 = fitness.max().into_scalar();
        println!(
            "Gen {}: Avg Fitness = {:.4}, Max Fitness = {:.4}",
            generation, avg, max
        );
    }
}

#[test]
fn test_fitness_progress() {
    let device = NdArrayDevice::default();
    let env = MnistEnvironment::<Backend, N_TOTAL>::new("data/", 100, 1.0, &device);

    let brain = CtrnnBrain::<Backend, N_TOTAL, 12>::new(
        BATCH_SIZE,
        30, // num_neurons
        MutationConfig::default(),
        &device,
    );

    let mut runner = Runner::new(
        env,
        brain,
        SimpleLogger { generation: 0 },
        0.1, // motor_dt
        10,  // brain_ticks_per_motor
        CONCURRENT_SIMS,
    );

    println!("Starting simulation investigation...");
    runner.evolve(100);
}
