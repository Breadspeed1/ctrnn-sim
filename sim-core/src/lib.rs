use burn::{Tensor, prelude::Backend};

pub mod brain;

// A -> num agents
// N -> num neurons
// B -> num parallel simulations
// M -> actions
// I -> inputs

pub trait AgentBrain<B: Backend, S, const INS: usize, const OUTS: usize> {
    fn batch_size(&self) -> usize;

    /// inputs: [A, I, B]
    /// return: [A, M, B]
    fn forward(&mut self, inputs: Tensor<B, 3>, state: S, dt: f32) -> (Tensor<B, 3>, S);

    /// fitness: [A, B]
    fn mutate(&mut self, fitness: Tensor<B, 2>);

    fn init_state(&self, par_sims: usize) -> S;
}

pub trait Environment<B: Backend, S, const INS: usize, const OUTS: usize> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn reset(&mut self, batch_size: usize, concurrent_sims: usize);

    fn get_sensors(&self) -> Tensor<B, 3>;

    fn apply_motors(&mut self, motors: Tensor<B, 3>);

    fn calculate_fitness(&self) -> Tensor<B, 2>;

    fn is_done(&self) -> bool;

    fn get_state(&self) -> S;
}

pub trait SimLogger<B: Backend, ES, BS, A> {
    fn record_env(&mut self, state: ES, generation: usize);
    fn record_ephemeral_brain(&mut self, state: BS, generation: usize);
    fn record_brain(&mut self, brain: A, generation: usize);
}

pub struct Runner<B, ES, BS, E, A, L, const INS: usize, const OUTS: usize> {
    env: E,
    brain: A,
    logger: L,
    _phantom: std::marker::PhantomData<(B, ES, BS)>,

    // params
    concurrent_sims: usize,
    motor_dt: f32,
    brain_ticks_per_motor: u32,
}

impl<B, ES, BS, E, A, L, const INS: usize, const OUTS: usize> Runner<B, ES, BS, E, A, L, INS, OUTS>
where
    BS: Clone,
    B: Backend,
    E: Environment<B, ES, INS, OUTS>,
    A: AgentBrain<B, BS, INS, OUTS> + Clone,
    L: SimLogger<B, ES, BS, A>,
{
    pub fn new(
        env: E,
        brain: A,
        logger: L,
        motor_dt: f32,
        brain_ticks_per_motor: u32,
        concurrent_sims: usize,
    ) -> Self {
        Self {
            concurrent_sims,
            motor_dt,
            brain_ticks_per_motor,
            env,
            brain,
            logger,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn evolve(&mut self, generations: usize) {
        let brain_dt = self.motor_dt / self.brain_ticks_per_motor as f32;

        for generation in 0..generations {
            self.logger.record_brain(self.brain.clone(), generation);

            self.env
                .reset(self.brain.batch_size(), self.concurrent_sims);

            let mut state = self.brain.init_state(self.concurrent_sims);

            while !self.env.is_done() {
                let inputs = self.env.get_sensors();

                // do n - 1 brain loops
                for _ in 0..(self.brain_ticks_per_motor - 1) {
                    let (_, new_state) = self.brain.forward(inputs.clone(), state, brain_dt);
                    state = new_state;
                    self.logger
                        .record_ephemeral_brain(state.clone(), generation);
                }

                // one more brain loop and take motor functions
                let (motors, new_state) = self.brain.forward(inputs, state, brain_dt);
                state = new_state;
                self.logger
                    .record_ephemeral_brain(state.clone(), generation);

                self.env.apply_motors(motors);

                self.logger.record_env(self.env.get_state(), generation);
            }

            let fitness = self.env.calculate_fitness();

            self.brain.mutate(fitness);
        }
    }
}
