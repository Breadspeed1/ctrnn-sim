use burn::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use sim_core::AgentBrain;
use sim_core::brain::ctrnn::{CtrnnBrain, MutationConfig};

type Backend = NdArray;
const INS: usize = 10;
const OUTS: usize = 12;
const BATCH_SIZE: usize = 5;
const PAR_SIMS: usize = 3;

#[test]
fn test_ctrnn_produces_diverse_outputs() {
    let device = NdArrayDevice::default();

    let mut brain =
        CtrnnBrain::<Backend, INS, OUTS>::new(BATCH_SIZE, 30, MutationConfig::default(), &device);

    let mut state = brain.init_state(PAR_SIMS);

    // Create simple inputs: all ones
    let inputs = Tensor::ones([BATCH_SIZE, INS, PAR_SIMS], &device);

    println!(
        "Initial state mean: {:.4}",
        state.states.clone().mean().into_scalar()
    );

    // Run for many steps to let the network settle
    let dt = 0.1;
    for step in 0..100 {
        let (outputs, new_state) = brain.forward(inputs.clone(), state, dt);
        state = new_state;

        if step % 20 == 0 {
            let out_data = outputs.to_data();
            let out_slice = out_data.as_slice::<f32>().unwrap();

            // Check output diversity
            let mean: f32 = outputs.clone().mean().into_scalar();
            let max: f32 = outputs.clone().max().into_scalar();
            let min: f32 = outputs.clone().min().into_scalar();

            println!(
                "Step {}: Output mean={:.4}, min={:.4}, max={:.4}, range={:.4}",
                step,
                mean,
                min,
                max,
                max - min
            );

            // Print first few outputs for agent 0
            println!("  Agent 0 outputs: {:?}", &out_slice[0..OUTS.min(5)]);
        }
    }

    // Final outputs
    let (final_outputs, _) = brain.forward(inputs, state, dt);

    // Check that different agents have different outputs
    let agent0_mean: f32 = final_outputs.clone().slice([0..1]).mean().into_scalar();
    let agent1_mean: f32 = final_outputs.clone().slice([1..2]).mean().into_scalar();
    let agent4_mean: f32 = final_outputs.clone().slice([4..5]).mean().into_scalar();

    println!("\nFinal agent output means:");
    println!("  Agent 0: {:.4}", agent0_mean);
    println!("  Agent 1: {:.4}", agent1_mean);
    println!("  Agent 4: {:.4}", agent4_mean);

    // Check output range
    let range: f32 =
        final_outputs.clone().max().into_scalar() - final_outputs.clone().min().into_scalar();
    println!("Final output range: {:.4}", range);

    // Assert outputs are diverse (different random initializations should give different outputs)
    assert!(
        range > 0.01,
        "Outputs should have some diversity, but range was {}",
        range
    );
}

#[test]
fn test_different_brains_produce_different_outputs() {
    let device = NdArrayDevice::default();

    // Create two different brains
    let mut brain1 =
        CtrnnBrain::<Backend, INS, OUTS>::new(1, 30, MutationConfig::default(), &device);
    let mut brain2 =
        CtrnnBrain::<Backend, INS, OUTS>::new(1, 30, MutationConfig::default(), &device);

    let mut state1 = brain1.init_state(1);
    let mut state2 = brain2.init_state(1);

    let inputs = Tensor::ones([1, INS, 1], &device);

    // Run both brains
    for _ in 0..50 {
        let (_, new_state1) = brain1.forward(inputs.clone(), state1, 0.1);
        let (_, new_state2) = brain2.forward(inputs.clone(), state2, 0.1);
        state1 = new_state1;
        state2 = new_state2;
    }

    let (out1, _) = brain1.forward(inputs.clone(), state1, 0.1);
    let (out2, _) = brain2.forward(inputs, state2, 0.1);

    let diff: f32 = (out1 - out2).abs().mean().into_scalar();
    println!(
        "Mean absolute difference between two random brains: {:.4}",
        diff
    );

    assert!(
        diff > 0.01,
        "Two random brains should produce different outputs"
    );
}
