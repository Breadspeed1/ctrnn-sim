use burn::{Tensor, prelude::Backend};

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

    input_count: usize,
    output_count: usize,
}

struct CtrnnState<B: Backend> {
    // [A, N, B]
    states: Tensor<B, 3>,
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
            .slice_assign([0..batch_size, 0..self.input_count, 0..par_sims], inputs);

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
        let motor_potentials = state.states.clone().slice([
            0..batch_size,
            self.input_count..(self.input_count + self.output_count),
            0..par_sims,
        ]);

        // Return the motor outputs AND the updated state
        (motor_potentials.tanh(), state)
    }

    fn mutate(&mut self, fitness: Tensor<B, 2>) {}

    fn init_state(&self) -> CtrnnState<B> {
        todo!()
    }
}
