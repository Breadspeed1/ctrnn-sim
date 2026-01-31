use burn::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Int;
use sim_core::brain::ctrnn::{CtrnnBrain, MutationConfig};
use sim_core::{AgentBrain, Environment, Runner, SimLogger};

type Backend = NdArray;
const INS: usize = 2;
const OUTS: usize = 2;
const BATCH_SIZE: usize = 100;
const PAR_SIMS: usize = 5;

// A trivially easy environment: output should match input
struct MatchInputEnv {
    inputs: Tensor<Backend, 3>,
    targets: Tensor<Backend, 2>,
    fitness: Tensor<Backend, 2>,
    step: usize,
    done: bool,
}

#[derive(Clone)]
struct MatchInputState {}

impl MatchInputEnv {
    fn new(device: &<Backend as burn::prelude::Backend>::Device) -> Self {
        Self {
            inputs: Tensor::zeros([1, INS, 1], device),
            targets: Tensor::zeros([1, 1], device),
            fitness: Tensor::zeros([1, 1], device),
            step: 0,
            done: true,
        }
    }
}

impl Environment<Backend, MatchInputState, INS, OUTS> for MatchInputEnv {
    fn input_size(&self) -> usize {
        INS
    }
    fn output_size(&self) -> usize {
        OUTS
    }

    fn reset(&mut self, batch_size: usize, concurrent_sims: usize) {
        let device = &self.inputs.device();
        // Input: constant signal
        self.inputs = Tensor::ones([batch_size, INS, concurrent_sims], device);
        self.targets = Tensor::ones([batch_size, concurrent_sims], device);
        self.fitness = Tensor::zeros([batch_size, concurrent_sims], device);
        self.step = 0;
        self.done = false;
    }

    fn get_sensors(&self) -> Tensor<Backend, 3> {
        self.inputs.clone()
    }

    fn apply_motors(&mut self, motors: Tensor<Backend, 3>) {
        // Fitness = how close outputs are to target [1, -1]
        let [batch_size, _, concurrent_sims] = motors.dims();

        // Target: first output > 0, second output < 0
        let out0 = motors
            .clone()
            .slice([0..batch_size, 0..1, 0..concurrent_sims])
            .squeeze_dim(1);
        let out1 = motors
            .clone()
            .slice([0..batch_size, 1..2, 0..concurrent_sims])
            .squeeze_dim(1);

        // Reward: out0 - out1 (want out0 high, out1 low)
        let reward = out0 - out1;
        self.fitness = self.fitness.clone() + reward;

        self.step += 1;
        if self.step >= 10 {
            self.done = true;
        }
    }

    fn calculate_fitness(&self) -> Tensor<Backend, 2> {
        self.fitness.clone()
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn get_state(&self) -> MatchInputState {
        MatchInputState {}
    }
}

struct PrintLogger;

impl<B: burn::prelude::Backend<FloatElem = f32>, ES, BS, A> SimLogger<B, ES, BS, A>
    for PrintLogger
{
    fn record_env(&mut self, _state: ES, _generation: usize) {}
    fn record_ephemeral_brain(&mut self, _state: BS, _generation: usize) {}
    fn record_brain(&mut self, _brain: A, _generation: usize) {}
    fn record_fitness(&mut self, fitness: Tensor<B, 2>, generation: usize) {
        let avg: f32 = fitness.clone().mean().into_scalar();
        let max: f32 = fitness.max().into_scalar();
        println!("Gen {}: Avg={:.4}, Max={:.4}", generation, avg, max);
    }
}

#[test]
fn test_evolution_on_simple_task() {
    let device = NdArrayDevice::default();
    let env = MatchInputEnv::new(&device);

    let brain = CtrnnBrain::<Backend, INS, OUTS>::new(
        BATCH_SIZE,
        10, // small network
        MutationConfig {
            survival_rate: 0.2,
            mutation_rate: 0.1,
            mutation_power: 0.3,
            node_add_prob: 0.0, // keep it simple
        },
        &device,
    );

    let mut runner = Runner::new(env, brain, PrintLogger, 0.1, 10, PAR_SIMS);

    println!("Testing evolution on simple task (want out0 > out1):");
    runner.evolve(50);

    // The max fitness should increase over generations
    // With 10 steps and target reward = out0 - out1,
    // max possible is 10 * 2 = 20 (if out0=1, out1=-1)
}
