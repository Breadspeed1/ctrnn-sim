use burn::backend::{
    Wgpu,
    ndarray::{NdArray, NdArrayDevice},
    wgpu::WgpuDevice,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use mnist_sim::{
    MnistEnvironment,
    tui_logger::{Metrics, TuiLogger},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
};
use sim_core::Runner;
use sim_core::brain::ctrnn::{CtrnnBrain, MutationConfig};
use std::{
    io,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

type Backend = Wgpu;
const N_TOTAL: usize = 10;
const BATCH_SIZE: usize = 1000;
const CONCURRENT_SIMS: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let metrics = Arc::new(Mutex::new(Metrics::default()));
    let metrics_for_runner = Arc::clone(&metrics);

    // Run simulation in a background thread
    let _sim_thread = thread::spawn(move || {
        let device = WgpuDevice::default();
        let env = MnistEnvironment::<Backend, N_TOTAL>::new("data/", 200, 1.0, &device);

        let brain = CtrnnBrain::<Backend, N_TOTAL, 12>::new(
            BATCH_SIZE,
            50, // num_neurons
            MutationConfig::default(),
            &device,
        );

        let logger = TuiLogger::new(metrics_for_runner);
        let mut runner = Runner::new(
            env,
            brain,
            logger,
            0.1, // motor_dt
            10,  // brain_ticks_per_motor
            CONCURRENT_SIMS,
        );

        runner.evolve(1000);
    });

    // Main TUI loop
    let res = run_tui(&mut terminal, metrics);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_tui<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    metrics: Arc<Mutex<Metrics>>,
) -> io::Result<()> {
    let tick_rate = Duration::from_millis(10);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &metrics))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Char('q') = key.code {
                    return Ok(());
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

fn ui(f: &mut ratatui::Frame, metrics: &Arc<Mutex<Metrics>>) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(f.area());

    let m = metrics.lock().unwrap();
    let generation = m.generation;
    let latest_best = m.best_fitness.last().cloned().unwrap_or(0.0);
    let latest_avg = m.avg_fitness.last().cloned().unwrap_or(0.0);

    let info_text = format!(
        "Generation: {} | Best Fitness: {:.4} | Avg Fitness: {:.4} | Press 'q' to quit",
        generation, latest_best, latest_avg
    );
    let info =
        Paragraph::new(info_text).block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(info, chunks[0]);

    let best_data: Vec<(f64, f64)> = m
        .best_fitness
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();
    let avg_data: Vec<(f64, f64)> = m
        .avg_fitness
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v as f64))
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Best")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&best_data),
        Dataset::default()
            .name("Avg")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(&avg_data),
    ];

    let max_fitness = m
        .best_fitness
        .iter()
        .fold(1.0, |max, &v| if v > max { v } else { max }) as f64
        * 1.1;
    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Fitness Over Time"),
        )
        .x_axis(
            Axis::default()
                .title("Generation")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, generation.max(1) as f64]),
        )
        .y_axis(
            Axis::default()
                .title("Fitness")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_fitness]),
        );

    f.render_widget(chart, chunks[1]);
}
