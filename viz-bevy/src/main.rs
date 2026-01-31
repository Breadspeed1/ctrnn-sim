use bevy::prelude::{App, Assets, DefaultPlugins, Handle, Image, Res, ResMut, Resource, Update};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use burn::Tensor;
use burn::backend::Wgpu;
use burn::prelude::Backend;
use crossbeam_channel::{Receiver, Sender};
use mnist_sim::{MnistEnvironment, MnistSimState};
use serde::{Deserialize, Serialize};
use sim_core::{
    Runner, SimLogger,
    brain::ctrnn::{CtrnnBrain, CtrnnState, MutationConfig},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

type B = Wgpu;

// --- Communication Types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessStats {
    pub generation: usize,
    pub max_fitness: f32,
    pub avg_fitness: f32,
    pub min_fitness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentData {
    pub position: [f32; 2],
    pub sensors: Vec<f32>,
    pub logits: Vec<f32>,
    pub neurons: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimTick {
    pub generation: usize,
    pub step_count: usize,
    pub agents: Vec<AgentData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliteBrainData {
    pub generation: usize,
    pub index: usize,
    pub weights: Vec<f32>, // [N, N]
    pub biases: Vec<f32>,  // [N]
    pub taus: Vec<f32>,    // [N]
    pub num_neurons: usize,
    pub firing_rates: Vec<f32>, // [N]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimMessage {
    GenerationStart {
        generation: usize,
        images: Vec<Vec<f32>>, // [S, 784]
    },
    Tick(SimTick),
    GenerationEnd(FitnessStats),
    EliteBrain(EliteBrainData),
    EliteFiringRates {
        generation: usize,
        firing_rates: Vec<f32>,
    },
}

// --- Simulation Logger ---

pub struct BevyLogger {
    pub sender: Sender<SimMessage>,
    pub last_generation: Option<usize>,
    pub paused: Arc<AtomicBool>,
    pub speed_ms: Arc<AtomicU32>,
    pub top_index: Option<usize>,
    pub last_firing_rates: Vec<f32>,
    pub stop_signal: Arc<AtomicBool>,
}

impl<B: Backend, const INS: usize, const OUTS: usize>
    SimLogger<B, MnistSimState<B>, CtrnnState<B>, CtrnnBrain<B, INS, OUTS>> for BevyLogger
{
    fn record_env(&mut self, state: MnistSimState<B>, generation: usize) {
        if self.stop_signal.load(Ordering::Relaxed) {
            // We can't easily return but we can stop sending
            return;
        }

        while self.paused.load(Ordering::Relaxed) {
            if self.stop_signal.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // ... previous logic to send Tick ...
        // I'll re-implement the Tick sending here to include the FiringRates update

        if self.last_generation != Some(generation) {
            self.last_generation = Some(generation);
            let images_data = state.active_images.into_data(); // Corrected from state.images
            let images_vec = images_data.as_slice::<f32>().unwrap();
            let s = images_data.shape[0];
            let mut all_images = Vec::with_capacity(s);
            for i in 0..s {
                let start = i * 28 * 28;
                all_images.push(images_vec[start..start + 28 * 28].to_vec());
            }
            let _ = self.sender.send(SimMessage::GenerationStart {
                generation,
                images: all_images,
            });
        }

        let positions_data = state.positions.into_data();
        let positions = positions_data.as_slice::<f32>().unwrap();

        let sensors_flat = state
            .sensors
            .clone()
            .swap_dims(1, 2)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let logits_flat = state
            .last_logits
            .clone()
            .swap_dims(1, 2)
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let [a, ins, s] = state.sensors.dims();
        let mut agents = Vec::with_capacity(a * s);

        for i in 0..a {
            for j in 0..s {
                let pos_idx = (i * s + j) * 2;
                let sensor_start = (i * s + j) * ins;
                let logit_start = (i * s + j) * 10;

                agents.push(AgentData {
                    position: [positions[pos_idx], positions[pos_idx + 1]],
                    sensors: sensors_flat[sensor_start..sensor_start + ins].to_vec(),
                    logits: logits_flat[logit_start..logit_start + 10].to_vec(),
                    neurons: Vec::new(),
                });
            }
        }

        let _ = self.sender.send(SimMessage::Tick(SimTick {
            generation,
            step_count: state.step_count,
            agents,
        }));

        // Send Elite Firing Rates ONCE PER STEP alongside Tick
        if !self.last_firing_rates.is_empty() {
            let _ = self.sender.send(SimMessage::EliteFiringRates {
                generation,
                firing_rates: self.last_firing_rates.clone(),
            });
        }
    }

    fn record_ephemeral_brain(&mut self, brain_state: CtrnnState<B>, _generation: usize) {
        // After mutation (Gen > 0), the elite is at index 0.
        // For Gen 0, we can also just show index 0 as a representative.
        self.last_firing_rates = brain_state
            .states
            .clone()
            .tanh()
            .select(0, Tensor::from_ints([0], &brain_state.states.device())) // [1, N, B]
            .select(2, Tensor::from_ints([0], &brain_state.states.device())) // [1, N, 1]
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
    }

    fn record_brain(&mut self, brain: CtrnnBrain<B, INS, OUTS>, generation: usize) {
        let device = &brain.weights.device();
        let idx_tensor = Tensor::from_ints([0], device); // Always watch index 0 (The Champion)

        let weights = brain
            .weights
            .clone()
            .select(0, idx_tensor.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let biases = brain
            .biases
            .clone()
            .select(0, idx_tensor.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let taus = brain
            .taus
            .clone()
            .select(0, idx_tensor.clone())
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let num_neurons = brain.num_neurons();

        let _ = self.sender.send(SimMessage::EliteBrain(EliteBrainData {
            generation,
            index: 0,
            weights,
            biases,
            taus,
            num_neurons,
            firing_rates: vec![0.0; num_neurons],
        }));
    }

    fn record_fitness(&mut self, fitness: burn::tensor::Tensor<B, 2>, generation: usize) {
        let data = fitness.into_data();
        let fitness_data = data.as_slice::<f32>().unwrap();
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut sum = 0.0;
        let mut max_idx = 0;
        for (i, &f) in fitness_data.iter().enumerate() {
            if f > max {
                max = f;
                max_idx = i;
            }
            if f < min {
                min = f;
            }
            sum += f;
        }
        let avg = sum / fitness_data.len() as f32;

        self.top_index = Some(max_idx);

        let _ = self.sender.send(SimMessage::GenerationEnd(FitnessStats {
            generation,
            max_fitness: max,
            avg_fitness: avg,
            min_fitness: min,
        }));
    }
}

// Special logger that also can send images at gen start
// This struct is no longer needed as BevyLogger now handles image sending.
// struct EnhancedLogger<B: Backend> {
//     sender: Sender<SimMessage>,
//     env: Option<MnistEnvironment<B, 10>>, // This is a bit hacky but we need access to env images
//     last_gen: Option<usize>,
// }

// --- Bevy Resources & Systems ---

#[derive(Resource)]
struct SimChannel {
    rx: Receiver<SimMessage>,
}

#[derive(Resource)]
struct SimControl {
    paused: Arc<AtomicBool>,
    speed_ms: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct SimulationState {
    fitness_history: Vec<FitnessStats>,
    current_tick: Option<SimTick>,
    generation_images: Vec<Handle<Image>>,
    generation: usize,
    selected_agent: Option<usize>,
    elite_brain: Option<EliteBrainData>,
}

fn main() {
    let (tx, rx) = crossbeam_channel::unbounded();

    let sim_control = SimControl {
        paused: Arc::new(AtomicBool::new(false)),
        speed_ms: Arc::new(AtomicU32::new(0)),
    };
    let paused_clone = sim_control.paused.clone();
    let speed_ms_clone = sim_control.speed_ms.clone();
    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();

    // Spawn simulation thread
    std::thread::spawn(move || {
        run_simulation::<B, 100, 10>(tx, paused_clone, speed_ms_clone, stop_signal_clone);
    });

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .insert_resource(SimChannel { rx })
        .insert_resource(sim_control)
        .init_resource::<SimulationState>()
        .add_systems(Update, (data_ingestion_system, ui_system))
        .run();
}

fn run_simulation<B: Backend, const INS: usize, const OUTS: usize>(
    tx: Sender<SimMessage>,
    paused: Arc<AtomicBool>,
    speed_ms: Arc<AtomicU32>,
    stop_signal: Arc<AtomicBool>,
) {
    // Increase WGPU limits if possible or configure backend
    // Burn doesn't expose a simple "remove limits" API without a custom runtime
    // But we can try to increase the simulation count and see if it hits the limit.

    let device = <B as Backend>::Device::default();
    let brain = CtrnnBrain::new(5000, 50, MutationConfig::default(), &device);
    let env = MnistEnvironment::<B, 10>::new("../mnist-sim/data", 200, 1.0, &device);

    let logger = BevyLogger {
        sender: tx.clone(),
        last_generation: None,
        paused,
        speed_ms,
        top_index: None,
        last_firing_rates: Vec::new(),
        stop_signal: stop_signal.clone(),
    };
    let mut runner = Runner::new(env, brain, logger, 0.1, 10, 50); // Increased from 5 to 12

    loop {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }
        runner.evolve(1);
    }
}

fn data_ingestion_system(
    mut sim_state: ResMut<SimulationState>,
    channel: Res<SimChannel>,
    mut images: ResMut<Assets<Image>>,
) {
    while let Ok(msg) = channel.rx.try_recv() {
        match msg {
            SimMessage::GenerationStart {
                generation,
                images: raw_images,
            } => {
                sim_state.generation = generation;
                sim_state.generation_images.clear();
                for img_data in raw_images {
                    let mut pixels = Vec::with_capacity(img_data.len() * 4);
                    for val in img_data {
                        let b = (val * 255.0).clamp(0.0, 255.0) as u8;
                        pixels.push(b); // R
                        pixels.push(b); // G
                        pixels.push(b); // B
                        pixels.push(255); // A
                    }
                    let image = Image::new_fill(
                        bevy::render::render_resource::Extent3d {
                            width: 28,
                            height: 28,
                            depth_or_array_layers: 1,
                        },
                        bevy::render::render_resource::TextureDimension::D2,
                        &pixels,
                        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                        bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD
                            | bevy::render::render_asset::RenderAssetUsages::MAIN_WORLD,
                    );
                    sim_state.generation_images.push(images.add(image));
                }
            }
            SimMessage::Tick(tick) => {
                sim_state.generation = tick.generation;
                sim_state.current_tick = Some(tick);
            }
            SimMessage::GenerationEnd(stats) => {
                sim_state.fitness_history.push(stats);
            }
            SimMessage::EliteBrain(elite) => {
                sim_state.elite_brain = Some(elite);
            }
            SimMessage::EliteFiringRates { firing_rates, .. } => {
                if let Some(elite) = &mut sim_state.elite_brain {
                    elite.firing_rates = firing_rates;
                }
            }
        }
    }
}

fn ui_system(
    mut contexts: EguiContexts,
    mut sim_state: ResMut<SimulationState>,
    sim_control: Res<SimControl>,
) {
    let texture_ids: Vec<_> = sim_state
        .generation_images
        .iter()
        .map(|img| contexts.add_image(img.clone_weak()))
        .collect();

    egui::Window::new("Simulation Dashboard")
        .id(egui::Id::new("sim_dashboard"))
        .default_pos(egui::pos2(10.0, 10.0))
        .show(contexts.ctx_mut(), |ui| {
            ui.heading(format!("Generation: {}", sim_state.generation));

            ui.horizontal(|ui| {
                let mut paused = sim_control.paused.load(Ordering::Relaxed);
                if ui.checkbox(&mut paused, "Paused").changed() {
                    sim_control.paused.store(paused, Ordering::Relaxed);
                }
            });

            ui.horizontal(|ui| {
                ui.label("Tick Speed (ms):");
                let mut speed = sim_control.speed_ms.load(Ordering::Relaxed);
                if ui.add(egui::Slider::new(&mut speed, 0..=200)).changed() {
                    sim_control.speed_ms.store(speed, Ordering::Relaxed);
                }
            });

            if let Some(tick) = &sim_state.current_tick {
                ui.label(format!("Step: {}", tick.step_count));
                ui.label(format!("Agents: {}", tick.agents.len()));

                if let Some(selected) = sim_state.selected_agent {
                    ui.label(format!("Selected Agent: {}", selected));
                    if ui.button("Deselect").clicked() {
                        sim_state.selected_agent = None;
                    }
                }
            }

            ui.separator();
            ui.heading("Fitness History");

            // Simple plot of fitness
            let max_points: egui_plot::PlotPoints = sim_state
                .fitness_history
                .iter()
                .enumerate()
                .map(|(i, s)| [i as f64, s.max_fitness as f64])
                .collect();

            let avg_points: egui_plot::PlotPoints = sim_state
                .fitness_history
                .iter()
                .enumerate()
                .map(|(i, s)| [i as f64, s.avg_fitness as f64])
                .collect();

            if !sim_state.fitness_history.is_empty() {
                let plot = egui_plot::Plot::new("fitness_plot")
                    .view_aspect(2.0)
                    .height(200.0)
                    .legend(egui_plot::Legend::default());

                plot.show(ui, |plot_ui| {
                    plot_ui.line(egui_plot::Line::new(max_points).name("Max Fitness"));
                    plot_ui.line(egui_plot::Line::new(avg_points).name("Avg Fitness"));
                });
            } else {
                ui.label("Waiting for data...");
            }
        });

    // Viewport window
    let mut new_selection = None;
    egui::Window::new("Agent Viewport")
        .id(egui::Id::new("agent_viewport"))
        .default_pos(egui::pos2(10.0, 350.0))
        .default_size(egui::vec2(600.0, 400.0))
        .show(contexts.ctx_mut(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if let Some(tick) = &sim_state.current_tick {
                    let cols = 4;
                    egui::Grid::new("viewport_grid")
                        .spacing([10.0, 10.0])
                        .show(ui, |ui| {
                            for (i, &texture_id) in texture_ids.iter().enumerate() {
                                ui.vertical(|ui| {
                                    ui.label(format!("Sim {}", i));

                                    let (rect, response) = ui.allocate_exact_size(
                                        egui::vec2(112.0, 112.0),
                                        egui::Sense::click(),
                                    );
                                    ui.painter().image(
                                        texture_id,
                                        rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                    // Overlay agents for this simulation
                                    let s = texture_ids.len();
                                    let a = if s > 0 { tick.agents.len() / s } else { 0 };

                                    // Handle clicking to select nearest agent in this sim
                                    if response.clicked() {
                                        if let Some(pointer_pos) = response.interact_pointer_pos() {
                                            let local_pos = pointer_pos - rect.min;
                                            let norm_x = local_pos.x / rect.width() * 28.0;
                                            let norm_y = local_pos.y / rect.height() * 28.0;

                                            let mut best_dist = 5.0; // Selection radius

                                            for j in 0..a {
                                                let idx = j * s + i;
                                                if let Some(agent) = tick.agents.get(idx) {
                                                    let dx = agent.position[0] - norm_x;
                                                    let dy = agent.position[1] - norm_y;
                                                    let dist = (dx * dx + dy * dy).sqrt();
                                                    if dist < best_dist {
                                                        best_dist = dist;
                                                        new_selection = Some(idx);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    for j in 0..a {
                                        let idx = j * s + i;
                                        if let Some(agent) = tick.agents.get(idx) {
                                            let pos = egui::pos2(
                                                rect.min.x
                                                    + (agent.position[0] / 28.0) * rect.width(),
                                                rect.min.y
                                                    + (agent.position[1] / 28.0) * rect.height(),
                                            );
                                            let is_elite =
                                                idx % (tick.agents.len() / texture_ids.len()) == 0; // Index 0 of each sim is usually the elite or its clone
                                            let color = if sim_state.selected_agent == Some(idx) {
                                                egui::Color32::from_rgb(0, 255, 0)
                                            } else if is_elite {
                                                egui::Color32::from_rgb(255, 255, 0) // Yellow for elites
                                            } else {
                                                egui::Color32::from_rgba_unmultiplied(
                                                    255, 0, 0, 100,
                                                )
                                            };
                                            ui.painter().circle_filled(pos, 1.5, color);
                                        }
                                    }
                                });
                                if (i + 1) % cols == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                }
            });
        });

    if new_selection.is_some() {
        sim_state.selected_agent = new_selection;
    }

    // Elite Brain Monitor window
    if let Some(elite) = &sim_state.elite_brain {
        egui::Window::new("Elite Brain Monitor")
            .id(egui::Id::new("elite_brain_monitor"))
            .default_pos(egui::pos2(600.0, 50.0))
            .default_size(egui::vec2(500.0, 400.0))
            .show(contexts.ctx_mut(), |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Generation: {}", elite.generation));
                    ui.separator();
                    ui.label(format!("Neurons: {}", elite.num_neurons));
                });

                ui.separator();

                egui::ScrollArea::both().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.heading("Weights Heatmap");
                            if !elite.weights.is_empty() {
                                let n = elite.num_neurons;
                                egui::Grid::new("weight_heatmap").spacing([1.0, 1.0]).show(
                                    ui,
                                    |ui| {
                                        for row in 0..n {
                                            for col in 0..n {
                                                let w = elite.weights[row * n + col];
                                                let (rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(3.0, 3.0),
                                                    egui::Sense::hover(),
                                                );
                                                let color = if w > 0.0 {
                                                    egui::Color32::from_rgb(
                                                        0,
                                                        (w * 255.0).min(255.0) as u8,
                                                        0,
                                                    )
                                                } else {
                                                    egui::Color32::from_rgb(
                                                        (w.abs() * 255.0).min(255.0) as u8,
                                                        0,
                                                        0,
                                                    )
                                                };
                                                ui.painter().rect_filled(rect, 0.0, color);
                                            }
                                            ui.end_row();
                                        }
                                    },
                                );
                            } else {
                                ui.label("Waiting for weight data...");
                            }
                        });

                        ui.separator();

                        ui.vertical(|ui| {
                            ui.heading("Parameters & Activity");
                            egui::Grid::new("elite_params").show(ui, |ui| {
                                ui.label("ID");
                                ui.label("Firing");
                                ui.label("Bias");
                                ui.label("Tau");
                                ui.end_row();

                                for i in 0..elite.num_neurons {
                                    ui.label(format!("{}", i));

                                    let firing = elite.firing_rates.get(i).cloned().unwrap_or(0.0);
                                    let (rect_f, _) = ui.allocate_exact_size(
                                        egui::vec2(40.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        rect_f,
                                        0.0,
                                        egui::Color32::from_gray((firing.abs() * 255.0) as u8),
                                    );

                                    if !elite.biases.is_empty() {
                                        let bias = elite.biases[i];
                                        let (rect_b, _) = ui.allocate_exact_size(
                                            egui::vec2(40.0, 10.0),
                                            egui::Sense::hover(),
                                        );
                                        let b_color = if bias > 0.0 {
                                            egui::Color32::GREEN
                                        } else {
                                            egui::Color32::RED
                                        };
                                        ui.painter().rect_filled(
                                            rect_b,
                                            0.0,
                                            b_color.gamma_multiply(bias.abs()),
                                        );

                                        let tau = elite.taus[i];
                                        let (rect_t, _) = ui.allocate_exact_size(
                                            egui::vec2(40.0, 10.0),
                                            egui::Sense::hover(),
                                        );
                                        ui.painter().rect_filled(
                                            rect_t,
                                            0.0,
                                            egui::Color32::BLUE.gamma_multiply(tau.min(2.0) / 2.0),
                                        );
                                    } else {
                                        ui.label("-");
                                        ui.label("-");
                                    }
                                    ui.end_row();
                                }
                            });
                        });
                    });
                });
            });
    }

    // Inspection window
    if let Some(selected_idx) = sim_state.selected_agent {
        if let Some(tick) = &sim_state.current_tick {
            if let Some(agent) = tick.agents.get(selected_idx) {
                egui::Window::new(format!("Agent {} Inspection", selected_idx))
                    .id(egui::Id::new("agent_inspection"))
                    .default_pos(egui::pos2(10.0, 750.0))
                    .show(contexts.ctx_mut(), |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.heading("Sensors");
                                let grid_size = (agent.sensors.len() as f32 - 1.0).sqrt() as usize;
                                // N-1 because of bias
                                if grid_size > 0 {
                                    egui::Grid::new("sensor_grid").show(ui, |ui| {
                                        for y in 0..grid_size {
                                            for x in 0..grid_size {
                                                let val = agent.sensors[y * grid_size + x];
                                                let color = if val < 0.0 {
                                                    egui::Color32::from_gray(0)
                                                } else {
                                                    egui::Color32::from_gray((val * 255.0) as u8)
                                                };
                                                let (rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().rect_filled(rect, 0.0, color);
                                            }
                                            ui.end_row();
                                        }
                                    });
                                }
                            });

                            ui.separator();

                            ui.vertical(|ui| {
                                ui.heading("Logits");
                                for (i, &val) in agent.logits.iter().enumerate() {
                                    ui.add(
                                        egui::ProgressBar::new(val.tanh().max(0.0))
                                            .text(format!("{}: {:.2}", i, val)),
                                    );
                                }
                            });
                        });

                        if !agent.neurons.is_empty() {
                            ui.separator();
                            ui.heading("Neuron Activations");
                            // Grid of all neurons
                        }
                    });
            }
        }
    }
}
