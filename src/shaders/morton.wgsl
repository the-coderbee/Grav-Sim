struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    _padding: f32,
}

struct MortonEntry {
    code: u32,
    particle_idx: u32,
}

struct SimBounds {
    min_bound: vec2<f32>,
    max_bound:
}

@group(0) @binding(0) var<storage, read_write> morton_a: array<MortonEntry>;
@group(0) @binding(1) var<storage, read_write> morton_b: array<MortonEntry>;
