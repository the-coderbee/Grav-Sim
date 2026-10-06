mod app;
mod camera;
mod geometry;
mod particle;
mod state;
mod ui;

use clap::Parser;

#[derive(Parser)]
#[command(about = "N-body gravity simulator")]
struct Args {
    #[arg(long, default_value_t = 50_000)]
    particles: usize,
}

fn main() {
    let args = Args::parse();
    println!("Particles: {}", args.particles);

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut app = app::App::new(args.particles);
    event_loop.run_app(&mut app).unwrap();
}
