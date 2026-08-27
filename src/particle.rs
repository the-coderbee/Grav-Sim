use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rand::Rng;
use std::f32::consts::TAU;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub mass: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub mass: f32,
    pub _padding: f32,
}

impl GpuParticle {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuParticle>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

pub struct GalaxyConfig {
    pub particle_count: usize,
    pub central_mass: f32,
    pub disk_partial_mass: f32,
    pub g_const: f32,
    pub min_radius: f32,
    pub max_radius: f32,
    pub perturbation_factor: f32,
}

pub fn generate_galaxy(config: &GalaxyConfig) -> Vec<Particle> {
    let mut rng = rand::thread_rng();
    let mut particles = Vec::with_capacity(config.particle_count);

    particles.push(Particle {
        position: Vec2::ZERO,
        velocity: Vec2::ZERO,
        mass: config.central_mass,
    });

    let mut total_disk_momentum = Vec2::ZERO;
    for _ in 1..config.particle_count {
        let r2 = config.min_radius * config.min_radius
            + rng.r#gen::<f32>()
                * (config.max_radius * config.max_radius - config.min_radius * config.min_radius);
        let r = r2.sqrt();
        let theta = rng.r#gen::<f32>() * TAU;

        let pos = Vec2::new(r * theta.cos(), r * theta.sin());

        let orbital_speed = (config.g_const * config.central_mass / r).sqrt();

        let tangent = Vec2::new(-theta.sin(), theta.cos());

        let noise_magnitude = orbital_speed * config.perturbation_factor;
        let perturbation = Vec2::new(
            (rng.r#gen::<f32>() - 0.5) * 2.0 * noise_magnitude,
            (rng.r#gen::<f32>() - 0.5) * 2.0 * noise_magnitude,
        );

        let vel = tangent * orbital_speed + perturbation;

        total_disk_momentum += vel * config.disk_partial_mass;

        particles.push(Particle {
            position: pos,
            velocity: vel,
            mass: config.disk_partial_mass,
        });
    }
    particles[0].velocity = -total_disk_momentum / config.central_mass;
    particles
}
