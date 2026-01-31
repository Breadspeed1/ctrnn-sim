use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use mnist_sim::MnistEnvironment;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::Color,
    widgets::{Block, Borders, Paragraph, canvas::Canvas},
};
use sim_core::Environment;
use std::{
    io,
    time::{Duration, Instant},
};

type Backend = NdArray;
const N_TOTAL: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Setup environment
    let device = NdArrayDevice::default();
    let mut env = MnistEnvironment::<Backend, N_TOTAL>::new("data/", 100, 2.0, &device);
    env.reset(1, 1);

    // Run loop
    let res = run_app(&mut terminal, env);

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

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut env: MnistEnvironment<Backend, N_TOTAL>,
) -> io::Result<()> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &env))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('r') => env.reset(1, 1),
                    KeyCode::Up => move_agent(&mut env, 0.0, -1.0),
                    KeyCode::Down => move_agent(&mut env, 0.0, 1.0),
                    KeyCode::Left => move_agent(&mut env, -1.0, 0.0),
                    KeyCode::Right => move_agent(&mut env, 1.0, 0.0),
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

fn move_agent(env: &mut MnistEnvironment<Backend, N_TOTAL>, dx: f32, dy: f32) {
    let device = NdArrayDevice::default();
    let mut motors_data = vec![0.0; 12];
    motors_data[10] = dx;
    motors_data[11] = dy;
    let motors =
        Tensor::<Backend, 1>::from_floats(motors_data.as_slice(), &device).reshape([1, 12, 1]);
    env.apply_motors(motors);
}

fn ui(f: &mut ratatui::Frame, env: &MnistEnvironment<Backend, N_TOTAL>) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Min(0)])
        .split(f.area());

    let state = env.get_state();
    let pos_data = state.positions.to_data();
    let pos_slice = pos_data.as_slice::<f32>().unwrap();
    let agent_x = pos_slice[0];
    let agent_y = pos_slice[1];

    // Draw the digit pixels
    let image_data = state.active_images.slice([0..1, 0..28, 0..28]).to_data();
    let pixels = image_data.as_slice::<f32>().unwrap();
    let digit_data = state.active_digits.to_data();
    let digit_indices = digit_data.as_slice::<i64>().unwrap();
    let digit_idx = digit_indices[0];

    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title("MNIST Digit"))
        .x_bounds([0.0, 28.0])
        .y_bounds([0.0, 28.0])
        .paint(|ctx| {
            for y in 0..28 {
                for x in 0..28 {
                    let val = pixels[y * 28 + x];
                    if val > 0.1 {
                        ctx.print(x as f64, (27 - y) as f64, "█");
                    }
                }
            }

            // Highlighting the kernel center
            ctx.print(agent_x as f64, (27.0 - agent_y) as f64, "X");

            // Draw kernel boundary
            let k = ((N_TOTAL - 1) as f32).sqrt() as f64;
            ctx.draw(&ratatui::widgets::canvas::Rectangle {
                x: agent_x as f64 - k / 2.0,
                y: (27.0 - agent_y) as f64 - k / 2.0,
                width: k,
                height: k,
                color: Color::Yellow,
            });
        });

    f.render_widget(canvas, chunks[0]);

    let sensor_data = state.sensors.to_data();
    let sensors = sensor_data.as_slice::<f32>().unwrap();
    let sensor_str = sensors
        .iter()
        .enumerate()
        .map(|(i, &v)| format!("I{}: {:.2}", i, v))
        .collect::<Vec<_>>()
        .join(", ");

    let info_text = format!(
        "Step: {}\nPos: ({:.1}, {:.1})\nDigit Index: {}\n\nInputs:\n{}\n\nControls:\nArrows: Move\nR: Reset\nQ: Quit",
        state.step_count, agent_x, agent_y, digit_idx, sensor_str
    );
    let info = Paragraph::new(info_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Simulation Info"),
    );
    f.render_widget(info, chunks[1]);
}
