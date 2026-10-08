mod app;
mod camera;
mod geometry;
mod particle;
mod state;
mod ui;

use clap::Parser;

use crate::app::AppConfig;

#[derive(Parser)]
#[command(about = "N-body gravity simulator")]
struct Args {
    #[arg(long, default_value_t = 50_000)]
    particles: usize,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
}

fn main() {
    let args = Args::parse();
    println!("Particles: {}", args.particles);

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let app_config = AppConfig {
        particles: args.particles,
        width: args.width,
        height: args.height,
    };
    let mut app = app::App::new(app_config);
    event_loop.run_app(&mut app).unwrap();
}
