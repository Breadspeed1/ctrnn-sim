use burn::{Tensor, prelude::Backend};

pub trait AgentBrain<B: Backend, S> {
    fn random(batch_size: usize, input_size: usize, output_size: usize, device: &B::Device)
    -> Self;

    fn batch_size(&self) -> usize;

    fn forward(&mut self, inputs: Tensor<B, 2>, dt: f32) -> Tensor<B, 2>;

    fn mutate(&mut self, fitness: Tensor<B, 1>);

    fn get_state(&self) -> S;
}

pub trait Environment<B: Backend> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

    fn reset(&mut self, batch_size: usize);

    fn get_sensors(&self) -> Tensor<B, 2>;

    fn apply_motors(&mut self, motors: Tensor<B, 2>);

    fn calculate_fitness(&self) -> Tensor<B, 1>;

    fn is_done(&self) -> bool;
}

pub trait SimLogger<B: Backend> {
    //TODO decide
}

pub struct Runner<B, S, E, A, L> {
    env: E,
    brain: A,
    logger: L,
    _phantom: std::marker::PhantomData<(B, S)>,
}

impl<B, S, E, A, L> Runner<B, S, E, A, L>
where
    B: Backend,
    E: Environment<B>,
    A: AgentBrain<B, S>,
    L: SimLogger<B>,
{
    pub fn new(env: E, brain: A, logger: L) -> Self {
        Self {
            env,
            brain,
            logger,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn evolve(&mut self, generations: usize) {
        for generation in 0..generations {
            self.env.reset(self.brain.batch_size());

            while !self.env.is_done() {
                let inputs = self.env.get_sensors();

                let motors = self.brain.forward(inputs, 0.1);

                self.env.apply_motors(motors);
            }

            let fitness = self.env.calculate_fitness();

            self.brain.mutate(fitness);

            todo!("logging")
        }
    }
}
