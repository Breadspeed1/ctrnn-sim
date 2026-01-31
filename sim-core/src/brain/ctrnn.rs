use burn::{Tensor, prelude::Backend, tensor::Float};

use crate::AgentBrain;

// A -> num agents
// N -> num neurons
// B -> num parallel simulations
// I -> num inputs

struct CtrnnBrain<B: Backend> {
    /// [A, N, N]
    weights: Tensor<B, 3>,

    /// [A, N]
    taus: Tensor<B, 2>,

    /// [A, N]
    biases: Tensor<B, 2>,

    /// [A, N]
    active_mask: Tensor<B, 2>,

    /// [A, N]
    non_input_mask: Tensor<B, 2>,

    mutation_config: MutationConfig,
}

struct MutationConfig {
    pub survival_rate: f32,
    pub mutation_rate: f32,
    pub mutation_power: f32,
    pub node_add_prob: f32,
}

struct CtrnnState<B: Backend> {
    // [A, N, B]
    states: Tensor<B, 3>,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            survival_rate: 0.10,  // Top 10% survive
            mutation_rate: 0.05,  // 5% of weights change
            mutation_power: 0.1,  // StdDev of noise
            node_add_prob: 0.001, // Chance to awaken a padding neuron
        }
    }
}

impl<B: Backend, const INS: usize, const OUTS: usize> AgentBrain<B, CtrnnState<B>, INS, OUTS>
    for CtrnnBrain<B>
{
    fn batch_size(&self) -> usize {
        self.weights.shape()[0]
    }

    fn forward(
        &mut self,
        inputs: Tensor<B, 3>,     // [Batch, Input_Count, Par_Sims]
        mut state: CtrnnState<B>, // Wraps [Batch, Neurons, Par_Sims]
        dt: f32,
    ) -> (Tensor<B, 3>, CtrnnState<B>) {
        let [batch_size, _num_neurons, par_sims] = state.states.dims();

        // 1. INJECT INPUTS
        // We slice inputs into the state.
        // Inputs shape: [Batch, Input, Sims] fits perfectly into [Batch, N, Sims]
        state.states = state
            .states
            .slice_assign([0..batch_size, 0..INS, 0..par_sims], inputs);

        // 2. ACTIVATION
        let firing_rates = state.states.clone().tanh(); // [Batch, N, Sims]

        // 3. MATMUL (The Fix)
        // Weights: [Batch, N, N]
        // Rates:   [Batch, N, Sims]
        // Result:  [Batch, N, Sims]
        // Rule: (B, N, N) x (B, N, S) -> (B, N, S).
        // We do NOT need to unsqueeze 'firing_rates' because it already has the 3rd dimension!
        let incoming_signals = self.weights.clone().matmul(firing_rates);

        // 4. BROADCASTING CONSTANTS
        // Biases are [Batch, N]. We need [Batch, N, 1] to add to [Batch, N, Sims].
        let biases_broad = self.biases.clone().unsqueeze_dim(2);
        let taus_broad = self.taus.clone().unsqueeze_dim(2);

        // Masks are [Batch, N]. Need [Batch, N, 1].
        let active_broad = self.active_mask.clone().unsqueeze_dim(2);
        let non_input_broad = self.non_input_mask.clone().unsqueeze_dim(2);

        // 5. DYNAMICS (The Physics)
        // Force = -y + (W * y) + b
        let force = -state.states.clone() + incoming_signals + biases_broad;

        // Delta = Force / Tau
        let delta = force / taus_broad;

        // 6. UPDATE
        // Apply update only to active, non-input neurons
        let update_mask = active_broad.mul(non_input_broad);
        let change = delta * update_mask * dt;

        state.states = state.states + change;

        // 7. OUTPUTS
        // Slice out the motor neurons: [Batch, Output_Count, Sims]
        let motor_potentials =
            state
                .states
                .clone()
                .slice([0..batch_size, INS..(INS + OUTS), 0..par_sims]);

        // Return the motor outputs AND the updated state
        (motor_potentials.tanh(), state)
    }

    fn mutate(&mut self, fitness: Tensor<B, 2>) {
        let device = self.weights.device();
        let batch_size =
            <CtrnnBrain<B> as AgentBrain<B, CtrnnState<B>, INS, OUTS>>::batch_size(self);

        // --- CONFIGURATION ---
        let survival_rate = self.mutation_config.survival_rate;
        let mutation_rate = self.mutation_config.mutation_rate;
        let mutation_power = self.mutation_config.mutation_power;
        let node_add_prob = self.mutation_config.node_add_prob;

        // --- PHASE 1: SELECTION (Hybrid CPU/GPU) ---

        // 1. Collapse fitness [Batch, Sims] -> [Batch]
        let scores: Tensor<B, 1> = fitness.mean_dim(1).squeeze_dim(1);

        // 2. Move to CPU for sorting (Burn doesn't always have easy Argsort on all backends)
        let scores_data = scores.to_data();
        let scores_vec: Vec<f32> = scores_data.as_slice::<f32>().unwrap().to_vec();

        // 3. ArgSort: Create (Index, Score) pairs and sort descending
        let mut indices: Vec<usize> = (0..batch_size).collect();
        indices.sort_by(|&a, &b| {
            scores_vec[b]
                .partial_cmp(&scores_vec[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Create Parent Mapping
        // The top 'num_elites' keep their spot.
        // The rest are replaced by looping through the elites.
        let num_elites = (batch_size as f32 * survival_rate).ceil() as usize;
        let mut parent_indices_vec = vec![0; batch_size];

        for i in 0..batch_size {
            if i < num_elites {
                parent_indices_vec[indices[i]] = indices[i]; // Elite survives in place (optional, or pack them)
                // Actually, simpler strategy: Pack Elites at the top, overwrite the rest.
                // Let's re-order the whole population to be sorted.
                // Index 'i' in the new population comes from 'indices[i]' (Elite)
                // Index '100' (Loser) comes from 'indices[100 % num_elites]' (Clone of Elite)
            }
        }

        // Revised Mapping: Sorted Population
        // Row 0 = Best Agent
        // Row N = Clone of (N % Elites)
        let mut sorted_parent_indices = vec![0; batch_size];
        for i in 0..batch_size {
            if i < num_elites {
                sorted_parent_indices[i] = indices[i];
            } else {
                sorted_parent_indices[i] = indices[i % num_elites];
            }
        }

        let parent_indices = Tensor::from_ints(sorted_parent_indices.as_slice(), &device);

        // --- PHASE 2: CLONING ---

        // Overwrite self tensors with the sorted/cloned versions
        self.weights = self.weights.clone().select(0, parent_indices.clone());
        self.biases = self.biases.clone().select(0, parent_indices.clone());
        self.taus = self.taus.clone().select(0, parent_indices.clone());
        self.active_mask = self.active_mask.clone().select(0, parent_indices.clone());
        // non_input_mask follows the agent structure
        self.non_input_mask = self
            .non_input_mask
            .clone()
            .select(0, parent_indices.clone());

        // --- PHASE 3: MUTATION ---

        let dims_3d = self.weights.shape();
        let dims_2d = self.biases.shape();

        // Helper: Create a mask of 1.0s with probability 'p'
        let make_rate_mask = |shape: burn::tensor::Shape, p: f32| -> Tensor<B, 3> {
            Tensor::<B, 3, Float>::random(
                shape.clone(),
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(p)
            .float()
        };

        // Helper 2d
        let make_rate_mask_2d = |shape: burn::tensor::Shape, p: f32| -> Tensor<B, 2> {
            Tensor::<B, 2, Float>::random(
                shape.clone(),
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(p)
            .float()
        };

        // A. Mutate Weights
        let noise_w = Tensor::random(
            dims_3d.clone(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask_w = make_rate_mask(dims_3d, mutation_rate);
        // Only mutate connections between active nodes? Or allow growth?
        // Let's allow weight changes everywhere, but masking usually restricts to active topology.
        // For simplicity: Mutate everything, but apply active_mask later in forward pass.
        self.weights = self.weights.clone() + (noise_w * mask_w * mutation_power);

        // B. Mutate Biases & Taus
        let noise_b = Tensor::random(
            dims_2d.clone(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask_b = make_rate_mask_2d(dims_2d.clone(), mutation_rate);
        self.biases = self.biases.clone() + (noise_b * mask_b * mutation_power);

        let noise_t = Tensor::random(
            dims_2d.clone(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mask_t = make_rate_mask_2d(dims_2d.clone(), mutation_rate);
        self.taus = self.taus.clone() + (noise_t * mask_t * mutation_power);

        // C. Topology Growth (Add Nodes)
        // Find inactive nodes: (1 - active)
        let inactive = self.active_mask.clone().neg().add_scalar(1.0);
        let growth_roll: Tensor<B, 2, Float> = Tensor::random(
            dims_2d.clone(),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        // If (inactive AND roll < prob), activate it.
        let new_nodes = growth_roll.lower_equal_elem(node_add_prob).float() * inactive;

        self.active_mask = self.active_mask.clone() + new_nodes;
        // Clamp to 1.0 just in case
        self.active_mask = self.active_mask.clone().clamp(0.0, 1.0);
    }

    fn init_state(&self, par_sims: usize) -> CtrnnState<B> {
        let [batch_size, num_neurons, _] = self.weights.dims();

        let states = Tensor::zeros([batch_size, num_neurons, par_sims], &self.weights.device());

        CtrnnState { states }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use burn::backend::Wgpu;
    use burn::backend::cuda::CudaDevice;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Distribution;

    type TestBackend = Wgpu;

    const BATCH: usize = 250;
    const NEURONS: usize = 250;
    const PAR_SIMS: usize = 5;
    const INS: usize = 2;
    const OUTS: usize = 2;

    const BENCH_STEPS: usize = 2000;

    // Helper to construct a random brain since we are inside the module
    // and can access private fields.
    fn create_test_brain(device: &<TestBackend as Backend>::Device) -> CtrnnBrain<TestBackend> {
        let shape_3d = [BATCH, NEURONS, NEURONS];
        let shape_2d = [BATCH, NEURONS];

        let weights = Tensor::random(shape_3d, Distribution::Normal(0.0, 1.0), device);
        let taus = Tensor::ones(shape_2d, device); // tau=1.0 for simple math
        let biases = Tensor::zeros(shape_2d, device);
        let active_mask = Tensor::ones(shape_2d, device);

        let mutation_config = MutationConfig::default();

        // Setup input mask: first INS neurons are inputs
        let mut non_input_mask_data = vec![1.0; BATCH * NEURONS];
        for b in 0..BATCH {
            for n in 0..INS {
                non_input_mask_data[b * NEURONS + n] = 0.0;
            }
        }
        let non_input_mask =
            Tensor::<TestBackend, 1, Float>::from_floats(non_input_mask_data.as_slice(), device)
                .reshape(shape_2d);

        CtrnnBrain {
            weights,
            taus,
            biases,
            active_mask,
            non_input_mask,
            mutation_config,
        }
    }

    #[test]
    fn bench() {
        let device = WgpuDevice::default();

        let mut brain = create_test_brain(&device);
        let mut state = <CtrnnBrain<TestBackend> as AgentBrain<
            TestBackend,
            CtrnnState<TestBackend>,
            INS,
            OUTS,
        >>::init_state(&brain, PAR_SIMS);

        // Create fake inputs: [Batch, INS, Sims]
        // All inputs = 1.0
        let inputs = Tensor::ones([BATCH, INS, PAR_SIMS], &device);
        let dt = 0.1;

        let time = Instant::now();

        // Run one step
        for _ in 0..BENCH_STEPS {
            let (_, new_state) = <CtrnnBrain<TestBackend> as AgentBrain<
                TestBackend,
                CtrnnState<TestBackend>,
                INS,
                OUTS,
            >>::forward(&mut brain, inputs.clone(), state, dt);

            state = new_state
        }

        let now = Instant::now();

        let total = now - time;

        println!(
            "Total time to run {} iterations: {}ms",
            BENCH_STEPS,
            total.as_millis()
        );
    }

    #[test]
    fn test_init_state_shapes() {
        let device = Default::default();
        let brain = create_test_brain(&device);

        // Check if init_state returns correct shape [Batch, Neurons, Sims]
        let state = <CtrnnBrain<TestBackend> as AgentBrain<
            TestBackend,
            CtrnnState<TestBackend>,
            INS,
            OUTS,
        >>::init_state(&brain, PAR_SIMS);
        let [b, n, s] = state.states.dims();

        assert_eq!(b, BATCH);
        assert_eq!(n, NEURONS);
        assert_eq!(s, PAR_SIMS);

        // Should initialize to zeros
        let sum: f32 = state.states.sum().into_scalar();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_forward_step_integration() {
        let device = Default::default();
        let mut brain = create_test_brain(&device);
        let state = <CtrnnBrain<TestBackend> as AgentBrain<
            TestBackend,
            CtrnnState<TestBackend>,
            INS,
            OUTS,
        >>::init_state(&brain, PAR_SIMS);

        // Create fake inputs: [Batch, INS, Sims]
        // All inputs = 1.0
        let inputs = Tensor::ones([BATCH, INS, PAR_SIMS], &device);
        let dt = 0.1;

        // Run one step
        let (motors, new_state) = <CtrnnBrain<TestBackend> as AgentBrain<
            TestBackend,
            CtrnnState<TestBackend>,
            INS,
            OUTS,
        >>::forward(&mut brain, inputs.clone(), state, dt);

        // 1. Check Input Clamping
        // The state of the first INS neurons should be EXACTLY the input (1.0)
        // because we use slice_assign at the start of forward.
        let state_data = new_state.states.to_data();
        let slice = new_state.states.clone().slice([0..1, 0..INS, 0..1]);
        let input_val: f32 = slice.mean().into_scalar();
        assert_eq!(
            input_val, 1.0,
            "Input neurons must be clamped to sensor values"
        );

        // 2. Check Motor Output Shape
        let [b, o, s] = motors.dims();
        assert_eq!(b, BATCH);
        assert_eq!(o, OUTS);
        assert_eq!(s, PAR_SIMS);

        // 3. Check Integration (Dynamics)
        // Since we initialized state to 0 and inputs to 1,
        // the non-input neurons should have moved from 0.0 towards non-zero.
        // We check a hidden neuron (index INS).
        let hidden_val: f32 = new_state
            .states
            .slice([0..1, INS..INS + 1, 0..1])
            .sum()
            .into_scalar();

        assert!(
            hidden_val != 0.0,
            "Hidden neurons should integrate dynamics"
        );
    }

    #[test]
    fn test_mutation_elitism() {
        let device = Default::default();
        let mut brain = create_test_brain(&device);

        // Snapshot the weights of Agent 0 (The expected Elite)
        let original_weights = brain.weights.clone();

        // Construct Fitness: Agent 0 is best (100.0), Agent 1 is worst (0.0)
        // Fitness shape: [Batch, Sims]
        let mut fitness_data = vec![0.0; BATCH * PAR_SIMS];
        fitness_data[0] = 100.0; // Agent 0 is the king
        let fitness =
            Tensor::<TestBackend, 1, Float>::from_floats(fitness_data.as_slice(), &device)
                .reshape([BATCH, PAR_SIMS]);

        // Run Mutation
        <CtrnnBrain<TestBackend> as AgentBrain<TestBackend, CtrnnState<TestBackend>, INS, OUTS>>::mutate(&mut brain, fitness);

        // 1. Verify Elitism (Agent 0 should be unchanged)
        // Note: This assumes your mutate logic puts the Elite back in slot 0
        // or keeps it somewhere. Based on our previous logic "sorted_parent_indices[0] = indices[0]",
        // Agent 0 (who was best) should remain in slot 0.
        let new_weights = brain.weights;

        let w0_old = original_weights.clone().slice([0..1]);
        let w0_new = new_weights.clone().slice([0..1]);

        // Calculate difference
        let diff: f32 = (w0_old - w0_new).abs().sum().into_scalar();
        assert_eq!(diff, 0.0, "The elite agent (Index 0) should not be mutated");

        // 2. Verify Mutation (Agent 1 should be different)
        // Agent 1 was low fitness, so it should be a mutated clone of Agent 0.
        let w1_old = original_weights.slice([1..2]);
        let w1_new = new_weights.slice([1..2]);

        let diff_1: f32 = (w1_old - w1_new).abs().sum().into_scalar();
        assert!(diff_1 > 0.0, "The loser agent (Index 1) should be mutated");
    }
}
