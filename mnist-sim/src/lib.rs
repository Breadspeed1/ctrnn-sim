use burn::{
    prelude::Backend,
    tensor::{Distribution, Int, Tensor},
};
use mnist::MnistBuilder;
use sim_core::Environment;

const MNIST_OUTPUTS: usize = 12;

#[derive(Clone)]
pub struct MnistSimState<B: Backend> {
    pub positions: Tensor<B, 3>,          // [A, S, 2]
    pub active_digits: Tensor<B, 1, Int>, // [S]
    pub fitness: Tensor<B, 2>,            // [A, S]
    pub step_count: usize,
    pub last_logits: Tensor<B, 3>,   // [A, 10, S]
    pub labels: Tensor<B, 1, Int>,   // [S]
    pub sensors: Tensor<B, 3>,       // [A, INS, S]
    pub active_images: Tensor<B, 3>, // [S, 28, 28]
}

pub struct MnistEnvironment<B: Backend, const N: usize> {
    // Data
    images: Tensor<B, 3>,      // [60000, 28, 28]
    labels: Tensor<B, 1, Int>, // [60000]

    // Simulation Parameters
    max_steps: usize,
    velocity_scale: f32,

    // Current State
    positions: Tensor<B, 3>,          // [A, S, 2]
    active_digits: Tensor<B, 1, Int>, // [S]
    fitness: Tensor<B, 2>,            // [A, S]
    step_count: usize,
    done: bool,
    last_logits: Tensor<B, 3>,   // [A, 10, S]
    active_images: Tensor<B, 3>, // [S, 28, 28]
}

impl<B: Backend, const N: usize> MnistEnvironment<B, N> {
    pub fn new(data_path: &str, max_steps: usize, velocity_scale: f32, device: &B::Device) -> Self {
        // Compile-time check for INS
        const {
            assert!(
                (N - 1).isqrt().pow(2) == N - 1,
                "N - 1 MUST BE A PERFECT SQUARE"
            );
        }

        let mnist = MnistBuilder::new()
            .base_path(data_path)
            .training_set_length(60000)
            .test_set_length(10000)
            .finalize();

        let train_data = mnist.trn_img;
        let train_labels = mnist.trn_lbl;

        let images = Tensor::<B, 1>::from_floats(
            train_data
                .into_iter()
                .map(|b| b as f32 / 255.0)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .reshape([60000, 28, 28]);

        let labels = Tensor::<B, 1, Int>::from_ints(
            train_labels
                .into_iter()
                .map(|b| b as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        Self {
            images,
            labels,
            max_steps,
            velocity_scale,
            positions: Tensor::zeros([1, 1, 2], device),
            active_digits: Tensor::zeros([1], device),
            fitness: Tensor::zeros([1, 1], device),
            step_count: 0,
            done: true,
            last_logits: Tensor::zeros([1, 10, 1], device),
            active_images: Tensor::zeros([1, 28, 28], device),
        }
    }

    pub fn get_images_for_tui(&self) -> Tensor<B, 3> {
        self.images.clone()
    }

    pub fn get_state(&self) -> MnistSimState<B> {
        self.get_state_inner()
    }

    fn get_state_inner(&self) -> MnistSimState<B> {
        MnistSimState {
            positions: self.positions.clone(),
            active_digits: self.active_digits.clone(),
            fitness: self.fitness.clone(),
            step_count: self.step_count,
            last_logits: self.last_logits.clone(),
            labels: self.labels.clone().select(0, self.active_digits.clone()),
            sensors: self.get_sensors(),
            active_images: self.active_images.clone(),
        }
    }
}

impl<B: Backend, const N: usize> Environment<B, MnistSimState<B>, N, { MNIST_OUTPUTS }>
    for MnistEnvironment<B, N>
{
    fn input_size(&self) -> usize {
        N
    }

    fn output_size(&self) -> usize {
        MNIST_OUTPUTS
    }

    fn reset(&mut self, batch_size: usize, concurrent_sims: usize) {
        let device = self.images.device();

        self.active_digits = Tensor::<B, 1, Int>::random(
            [concurrent_sims],
            Distribution::Uniform(0.0, 60000.0),
            &device,
        );

        // Cache active images [S, 28, 28]
        self.active_images = self.images.clone().select(0, self.active_digits.clone());

        // Start at center (14, 14)
        self.positions = Tensor::full([batch_size, concurrent_sims, 2], 14.0, &device);

        // Reset fitness and step count
        self.fitness = Tensor::zeros([batch_size, concurrent_sims], &device);
        self.step_count = 0;
        self.done = false;
        self.last_logits = Tensor::zeros([batch_size, 10, concurrent_sims], &device);
    }

    fn get_sensors(&self) -> Tensor<B, 3> {
        let [batch_size, concurrent_sims, _] = self.positions.dims();
        let device = self.images.device();

        let k = ((N - 1) as f32).sqrt() as i32;
        let half_k = k / 2;

        let mut sensors = Vec::with_capacity(N);
        let end = if k % 2 == 1 { half_k } else { half_k - 1 };

        for dy in -half_k..=end {
            for dx in -half_k..=end {
                let x = self
                    .positions
                    .clone()
                    .slice([0..batch_size, 0..concurrent_sims, 0..1])
                    .squeeze_dim(2)
                    + dx as f32;
                let y = self
                    .positions
                    .clone()
                    .slice([0..batch_size, 0..concurrent_sims, 1..2])
                    .squeeze_dim(2)
                    + dy as f32;

                let ix = x.clone().round().int().clamp(0, 27);
                let iy = y.clone().round().int().clamp(0, 27);

                let in_bounds = x
                    .clone()
                    .greater_equal_elem(0.0)
                    .bool_and(x.clone().lower_elem(27.5))
                    .bool_and(y.clone().greater_equal_elem(0.0))
                    .bool_and(y.clone().lower_elem(27.5));

                // Index into active_images: [S, 28, 28]
                // We need pixels for each simulation index S
                // iy and ix are [A, S], in_bounds is [A, S]
                let s_idx = Tensor::<B, 1, Int>::arange(0..concurrent_sims as i64, &device)
                    .unsqueeze_dim::<2>(0)
                    .repeat_dim(0, batch_size);

                let flat_img_idx = s_idx * (28 * 28) + iy * 28 + ix;

                let pixels = self
                    .active_images
                    .clone()
                    .flatten::<1>(0, 2)
                    .select(0, flat_img_idx.flatten::<1>(0, 1))
                    .reshape([batch_size, concurrent_sims]);

                let masked_pixels = pixels.mask_where(
                    in_bounds.bool_not(),
                    Tensor::full([batch_size, concurrent_sims], -1.0, &device),
                );

                sensors.push(masked_pixels.unsqueeze_dim(1));
            }
        }

        sensors.push(Tensor::full([batch_size, 1, concurrent_sims], 1.0, &device));

        Tensor::cat(sensors, 1)
    }

    fn apply_motors(&mut self, motors: Tensor<B, 3>) {
        if self.done {
            return;
        }

        let [batch_size, concurrent_sims, _] = self.positions.dims();

        let logits = motors
            .clone()
            .slice([0..batch_size, 0..10, 0..concurrent_sims]);
        let velocity = motors
            .clone()
            .slice([0..batch_size, 10..12, 0..concurrent_sims]);

        let velocity = velocity.swap_dims(1, 2);

        self.positions = self.positions.clone() + velocity * self.velocity_scale;
        self.positions = self.positions.clone().clamp(0.0, 27.0);

        let predictions = logits.clone().argmax(1).squeeze_dim(1);
        let active_digits_broad = self
            .active_digits
            .clone()
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch_size);
        let targets = self
            .labels
            .clone()
            .select(0, active_digits_broad.flatten::<1>(0, 1))
            .reshape([batch_size, concurrent_sims]);

        let correct = predictions.equal(targets).float();

        self.step_count += 1;
        let weight = self.step_count as f32 / self.max_steps as f32;

        self.fitness = self.fitness.clone() + correct * weight;
        self.last_logits = logits;

        if self.step_count >= self.max_steps {
            self.done = true;
        }
    }

    fn calculate_fitness(&self) -> Tensor<B, 2> {
        self.fitness.clone()
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn get_state(&self) -> MnistSimState<B> {
        self.get_state_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray;

    #[test]
    fn test_data_loading() {
        let device = NdArrayDevice::default();
        let env = MnistEnvironment::<TestBackend, 10>::new("data/", 10, 1.0, &device);

        assert_eq!(env.images.dims(), [60000, 28, 28]);
        assert_eq!(env.labels.dims(), [60000]);
    }

    #[test]
    fn test_reset_and_sensors() {
        let device = NdArrayDevice::default();
        let mut env = MnistEnvironment::<TestBackend, 10>::new("data/", 10, 1.0, &device);

        env.reset(2, 3);

        let sensors = env.get_sensors();
        assert_eq!(sensors.dims(), [2, 10, 3]);

        let bias = sensors.clone().slice([0..2, 9..10, 0..3]);
        let bias_val: f32 = bias.mean().into_scalar();
        assert_eq!(bias_val, 1.0);

        // Verify digits are synchronized across agents
        let state = env.get_state();
        assert_eq!(state.active_digits.dims(), [3]);
    }

    #[test]
    fn test_movement_and_fitness() {
        let device = NdArrayDevice::default();
        let mut env = MnistEnvironment::<TestBackend, 10>::new("data/", 5, 2.0, &device);

        env.reset(1, 1);
        let start_pos = env.positions.clone();

        let mut motors_data = vec![0.0; 12];
        motors_data[10] = 1.0;
        let motors = Tensor::<TestBackend, 1>::from_floats(motors_data.as_slice(), &device)
            .reshape([1, 12, 1]);

        env.apply_motors(motors);

        let end_pos = env.positions.clone();
        let diff = end_pos - start_pos;
        let dx: f32 = diff.clone().slice([0..1, 0..1, 0..1]).into_scalar();
        let dy: f32 = diff.slice([0..1, 0..1, 1..2]).into_scalar();

        assert_eq!(dx, 2.0);
        assert_eq!(dy, 0.0);

        assert!(!env.is_done());

        for _ in 0..4 {
            let motors = Tensor::<TestBackend, 3>::zeros([1, 12, 1], &device);
            env.apply_motors(motors);
        }

        assert!(env.is_done());
    }
}
