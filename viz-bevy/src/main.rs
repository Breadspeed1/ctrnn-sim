use burn::{backend::Wgpu, prelude::Backend};
use mnist_sim::{MnistEnvironment, MnistSimState};
use sim_core::{
    Runner, SimLogger,
    brain::ctrnn::{CtrnnBrain, CtrnnState, MutationConfig},
};

type B = Wgpu;

struct BasicLogger;

impl<B: Backend, const INS: usize, const OUTS: usize>
    SimLogger<B, MnistSimState<B>, CtrnnState<B>, CtrnnBrain<B, INS, OUTS>> for BasicLogger
{
    fn record_env(&mut self, _: MnistSimState<B>, _: usize) {
        println!("env rec");
    }

    fn record_ephemeral_brain(&mut self, _: CtrnnState<B>, _: usize) {
        println!("eph brain rec");
    }

    fn record_brain(&mut self, _: CtrnnBrain<B, INS, OUTS>, _: usize) {
        println!("brain rec");
    }
}

fn main() {
    let device = <B as Backend>::Device::default();

    let brain = CtrnnBrain::new(100, 50, MutationConfig::default(), &device);

    let env = MnistEnvironment::<B, 10>::new("../mnist-sim/data", 200, 1., &device);

    let mut runner = Runner::new(env, brain, BasicLogger, 0.1, 10, 5);

    println!("running.");

    runner.evolve(100);
}
