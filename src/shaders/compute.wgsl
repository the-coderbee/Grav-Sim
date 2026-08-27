struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    padding: f32,
};

// Since only need 2 fields are needed, use this smaller struct to reduce memory footprint.
struct CachedParticle {
    pos: vec2<f32>,
    mass: f32,
};

struct SimParams {
    g_const: f32,
    softening: f32,
    dt: f32,
    particle_count: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;

// 1. The Shared Desk: A fast workgroup cache that holds 256 particles
const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> tile_cache: array<CachedParticle, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cp_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let p_idx = global_id.x;

    // Store the particle this specific thread is responsible for
    var my_particle: Particle;
    if (p_idx < params.particle_count) {
        my_particle = particles_src[p_idx];
    }

    var total_acceleration = vec2<f32>(0.0, 0.0);

    // Calculate how many "tiles" (library trips) we need to make
    let num_tiles = (params.particle_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    // Loop through every tile
    for (var tile = 0u; tile < num_tiles; tile++) {

        let load_idx = tile * WORKGROUP_SIZE + local_id.x;

        // 2. THE FETCH: Every thread grabs exactly ONE particle from slow VRAM
        if (load_idx < params.particle_count) {
            tile_cache[local_id.x] = CachedParticle(
                particles_src[load_idx].pos,
                particles_src[load_idx].mass
            );
        } else {
            tile_cache[local_id.x] = CachedParticle(vec2<f32>(0.0, 0.0), 0.0);
        }

        // 3. THE BARRIER: Wait for everyone to finish putting their book on the desk
        workgroupBarrier();

        // 4. FAST MATH: Loop through the 256 cached particles locally
        for (var j = 0u; j < WORKGROUP_SIZE; j++) {
            let other = tile_cache[j];

            // Don't calculate gravity against ourselves or empty padding!
            if (other.mass > 0.0 && (tile * WORKGROUP_SIZE + j) != p_idx) {
                let diff = other.pos - my_particle.pos;

                // + 10.0 is a "softening factor" so stars don't explode if they overlap
                let dist_sq = dot(diff, diff) + params.softening * params.softening;
                let inv_dist = inverseSqrt(dist_sq);
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                total_acceleration += diff * (params.g_const * other.mass * inv_dist3);
            }
        }

        // 5. THE BARRIER: Wait for everyone to finish reading before loading the next tile!
        // If we didn't do this, some fast threads might overwrite the cache while others are still reading.
        workgroupBarrier();
    }

    // 6. Final Update: Apply physics and write back to the destination buffer
    if (p_idx < params.particle_count) {
        my_particle.vel += total_acceleration * params.dt;
        my_particle.pos += my_particle.vel * params.dt;

        particles_dst[p_idx] = my_particle;
    }
}
