use burn::{
    prelude::Backend,
    tensor::{Distribution, Int, Tensor},
};
use mnist::MnistBuilder;
use sim_core::Environment;

pub struct MnistEnvironment<B: Backend, const INS: usize, const OUTS: usize> {
    // Data
    images: Tensor<B, 3>,      // [60000, 28, 28]
    labels: Tensor<B, 1, Int>, // [60000]

    // Simulation Parameters
    kernel_size: usize,
    max_steps: usize,
    velocity_scale: f32,

    // Current State (Batch, Sims)
    // A -> Num Agents (Batch Size)
    // S -> Concurrent Sims (Num parallel digits per agent)
    positions: Tensor<B, 3>,          // [A, S, 2] (x, y)
    active_digits: Tensor<B, 2, Int>, // [A, S] (indices into images/labels)
    fitness: Tensor<B, 2>,            // [A, S]
    step_count: usize,
    done: bool,

    // Last outputs for classification
    last_logits: Option<Tensor<B, 3>>, // [A, S, 10]
}

impl<B: Backend, const INS: usize, const OUTS: usize> MnistEnvironment<B, INS, OUTS> {
    pub fn new(
        data_path: &str,
        kernel_size: usize,
        max_steps: usize,
        velocity_scale: f32,
        device: &B::Device,
    ) -> Self {
        let mnist = MnistBuilder::new()
            .base_path(data_path)
            .training_set_length(60000)
            .test_set_length(10000)
            .finalize();

        // Convert raw bytes to tensors
        // MnistBuilder gives Vec<u8>
        let train_data = mnist.trn_img; // 60000 * 28 * 28 = 47,040,000
        let train_labels = mnist.trn_lbl; // 60000

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
            kernel_size,
            max_steps,
            velocity_scale,
            positions: Tensor::zeros([1, 1, 2], device),
            active_digits: Tensor::zeros([1, 1], device),
            fitness: Tensor::zeros([1, 1], device),
            step_count: 0,
            done: true,
            last_logits: None,
        }
    }
}

impl<B: Backend, const INS: usize, const OUTS: usize> Environment<B, (), INS, OUTS>
    for MnistEnvironment<B, INS, OUTS>
{
    fn input_size(&self) -> usize {
        INS
    }

    fn output_size(&self) -> usize {
        OUTS
    }

    fn reset(&mut self, batch_size: usize, concurrent_sims: usize) {
        let device = self.images.device();

        // Randomly select digits for each agent and each parallel simulation
        self.active_digits = Tensor::<B, 2, Int>::random(
            [batch_size, concurrent_sims],
            Distribution::Uniform(0.0, 60000.0),
            &device,
        );

        // Start at center (14, 14)
        self.positions = Tensor::full([batch_size, concurrent_sims, 2], 14.0, &device);

        // Reset fitness and step count
        self.fitness = Tensor::zeros([batch_size, concurrent_sims], &device);
        self.step_count = 0;
        self.done = false;
        self.last_logits = None;
    }

    fn get_sensors(&self) -> Tensor<B, 3> {
        let [batch_size, concurrent_sims, _] = self.positions.dims();
        let device = self.images.device();
        let n = self.kernel_size;
        let half_n = n as i32 / 2;

        // We want to extract [Batch, Concurrent_Sims, N*N + 1]
        // But the brain expects [Batch, INS, Concurrent_Sims]
        // Wait, CtrnnBrain expects [Batch, INS, Concurrent_Sims] based on sim-core/lib.rs:14
        // /// inputs: [A, I, B]

        // Let's implement kernel extraction.
        // For each agent/sim, we look at positions[a, s, :] and images[active_digits[a, s]].

        // This is tricky to do purely in Burn without a lot of indexing if we want efficiency.
        // However, N is small (e.g. 3x3).

        let mut sensors = Vec::with_capacity(n * n + 1);

        for dy in -half_n..=half_n {
            for dx in -half_n..=half_n {
                // For each offset, get the pixel value at position + offset
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

                // round positions
                let ix = x.clone().round().int();
                let iy = y.clone().round().int();

                // Mask for in-bounds
                let in_bounds = ix
                    .clone()
                    .greater_equal_elem(0)
                    .bool_and(ix.clone().lower_elem(28))
                    .bool_and(iy.clone().greater_equal_elem(0))
                    .bool_and(iy.clone().lower_elem(28));

                // Index into images
                // This is still a bit complex to do "efficiently" for all agents/sims at once.
                // But we can flatten images to [60000*28*28] or just use gathered indexing.

                let flat_img_idx = self.active_digits.clone() * (28 * 28) + iy * 28 + ix;

                // Gather pixels
                let pixels = self
                    .images
                    .clone()
                    .flatten::<1>(0, 2)
                    .select(0, flat_img_idx.flatten::<1>(0, 1))
                    .reshape([batch_size, concurrent_sims]);

                // Apply out-of-bounds -1
                let masked_pixels = pixels.mask_where(
                    in_bounds.bool_not(),
                    Tensor::full([batch_size, concurrent_sims], -1.0, &device),
                );

                sensors.push(masked_pixels.unsqueeze_dim(1));
            }
        }

        // Add bias input
        sensors.push(Tensor::full([batch_size, 1, concurrent_sims], 1.0, &device));

        // Concat into [Batch, N*N + 1, Concurrent_Sims]
        // We have [Batch, 1, Concurrent_Sims] * (N*N + 1)

        Tensor::cat(sensors, 1)
    }

    fn apply_motors(&mut self, motors: Tensor<B, 3>) {
        if self.done {
            return;
        }

        let [batch_size, concurrent_sims, _] = self.positions.dims();

        // motors is [A, OUTS, B] -> [Batch, 12, Concurrent_Sims]
        let logits = motors
            .clone()
            .slice([0..batch_size, 0..10, 0..concurrent_sims]);
        let velocity = motors
            .clone()
            .slice([0..batch_size, 10..12, 0..concurrent_sims]);

        // Transpose velocity to [Batch, Concurrent_Sims, 2]
        let velocity = velocity.swap_dims(1, 2);

        // Update positions
        self.positions = self.positions.clone() + velocity * self.velocity_scale;
        self.positions = self.positions.clone().clamp(0.0, 27.0);

        // Calculate current step fitness
        // argmax(logits) along dim 1
        let predictions = logits.clone().argmax(1).squeeze_dim(1); // [Batch, Concurrent_Sims]
        let targets = self
            .labels
            .clone()
            .select(0, self.active_digits.clone().flatten::<1>(0, 1))
            .reshape([batch_size, concurrent_sims]);

        let correct = predictions.equal(targets).float();

        // Weight: (step + 1) / max_steps
        self.step_count += 1;
        let weight = self.step_count as f32 / self.max_steps as f32;

        self.fitness = self.fitness.clone() + correct * weight;

        self.last_logits = Some(logits);

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

    fn get_state(&self) -> () {
        ()
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
        // Assuming data is in ./data relative to this test
        let env = MnistEnvironment::<TestBackend, 10, 12>::new("data/", 3, 10, 1.0, &device);

        assert_eq!(env.images.dims(), [60000, 28, 28]);
        assert_eq!(env.labels.dims(), [60000]);
    }

    #[test]
    fn test_reset_and_sensors() {
        let device = NdArrayDevice::default();
        let mut env = MnistEnvironment::<TestBackend, 10, 12>::new("data/", 3, 10, 1.0, &device);

        env.reset(2, 3); // 2 agents, 3 parallel dims each

        let sensors = env.get_sensors();
        // Kernel 3x3 -> 9 pixels + 1 bias = 10
        // Expected shape: [Batch, INS, Concurrent_Sims] -> [2, 10, 3]
        assert_eq!(sensors.dims(), [2, 10, 3]);

        // Initial position is center (14, 14)
        // Check if bias is 1.0
        let bias = sensors.clone().slice([0..2, 9..10, 0..3]);
        let bias_val: f32 = bias.mean().into_scalar();
        assert_eq!(bias_val, 1.0);
    }

    #[test]
    fn test_movement_and_fitness() {
        let device = NdArrayDevice::default();
        let mut env = MnistEnvironment::<TestBackend, 10, 12>::new("data/", 3, 5, 2.0, &device);

        env.reset(1, 1);
        let start_pos = env.positions.clone();

        // Apply motors: [1, 12, 1]
        // Velocity (indices 10, 11) = 1.0, 0.0
        let mut motors_data = vec![0.0; 12];
        motors_data[10] = 1.0;
        let motors = Tensor::<TestBackend, 1>::from_floats(motors_data.as_slice(), &device)
            .reshape([1, 12, 1]);

        env.apply_motors(motors);

        let end_pos = env.positions.clone();
        let diff = end_pos - start_pos;
        let dx: f32 = diff.clone().slice([0..1, 0..1, 0..1]).into_scalar();
        let dy: f32 = diff.slice([0..1, 0..1, 1..2]).into_scalar();

        assert_eq!(dx, 2.0); // velocity_scale * 1.0
        assert_eq!(dy, 0.0);

        // Fitness should be updated (likely 0.0 initially if logits are 0)
        assert!(!env.is_done());

        for _ in 0..4 {
            let motors = Tensor::<TestBackend, 3>::zeros([1, 12, 1], &device);
            env.apply_motors(motors);
        }

        assert!(env.is_done());
    }
}
