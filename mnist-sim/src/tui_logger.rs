use burn::prelude::Backend;
use burn::tensor::Tensor;
use sim_core::SimLogger;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct Metrics {
    pub generation: usize,
    pub best_fitness: Vec<f32>,
    pub avg_fitness: Vec<f32>,
}

pub struct TuiLogger {
    metrics: Arc<Mutex<Metrics>>,
}

impl TuiLogger {
    pub fn new(metrics: Arc<Mutex<Metrics>>) -> Self {
        Self { metrics }
    }
}

impl<B: Backend, ES, BS, A> SimLogger<B, ES, BS, A> for TuiLogger {
    fn record_env(&mut self, _state: ES, _generation: usize) {
        // Optimized: skip high-frequency environment state
    }

    fn record_ephemeral_brain(&mut self, _state: BS, _generation: usize) {
        // Optimized: skip high-frequency brain state
    }

    fn record_brain(&mut self, _brain: A, _generation: usize) {
        // Optimized: skip recording brains for now
    }

    fn record_fitness(&mut self, fitness: Tensor<B, 2>, generation: usize) {
        let mut m = self.metrics.lock().unwrap();
        m.generation = generation;

        // Calculate best and average fitness
        let best = fitness.clone().max().into_scalar();
        let avg = fitness.mean().into_scalar();

        let best_f32: f32 = format!("{:?}", best).parse().unwrap_or(0.0);
        let avg_f32: f32 = format!("{:?}", avg).parse().unwrap_or(0.0);

        m.best_fitness.push(best_f32);
        m.avg_fitness.push(avg_f32);
    }
}
