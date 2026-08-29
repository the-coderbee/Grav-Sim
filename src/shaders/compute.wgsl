struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    padding: f32,
};

struct GpuNode {
    center_of_mass: vec2<f32>,
    mass: f32,
    size: f32,
    children: vec4<i32>,
    is_leaf: i32,
    padding0: i32,
    padding1: i32,
    padding2: i32,
};

struct SimParams {
    g_const: f32,
    softening: f32,
    dt: f32,
    particle_count: u32,
    theta: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;
@group(0) @binding(3) var<storage, read> tree_nodes: array<GpuNode>;

const WORKGROUP_SIZE: u32 = 256u;
// const THETA: f32 = 0.5;

@compute @workgroup_size(WORKGROUP_SIZE)
fn cp_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let p_idx = global_id.x;

    // Ensure threads dont go out of bounds
    if (p_idx >= params.particle_count) {
        return;
    }

    var my_particle = particles_src[p_idx];
    var total_acceleration = vec2<f32>(0.0, 0.0);

    // Initializw the Barnse-Hut traversal stack
    var stack: array<i32, 64>;
    var stack_ptr: i32 = 0;

    // Push root node onto stack (always index 0)
    stack[0] = 0;
    stack_ptr = 1;

    // Tree traversal loop
    while (stack_ptr > 0) {
        // Pop the top node off the stack
        stack_ptr -= 1;
        let node_idx = stack[stack_ptr];
        let node = tree_nodes[node_idx];

        let diff = node.center_of_mass - my_particle.pos;

        // Include softening directly in the squared distance to avoid division by zero
        let dist_sq = dot(diff, diff) + (params.softening * params.softening);
        let dist = sqrt(dist_sq);

        // Barnes-Hut Threshold Test
        if (node.is_leaf == 1 || (node.size / dist) < params.theta) {
            // Fast path: calculate gravity against this node.
            // sip self-gravity (if a leaf node's distance is extremely small)
            if (dist > 0.0001) {
                let inv_dist = inverseSqrt(dist_sq);
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                total_acceleration += diff * (params.g_const * node.mass * inv_dist3);
            }
        } else {
            // Slow path: Node is too close and is a branch.
            // Push its active children onto the stack to evaluate them deeper.
            if (node.children.x != -1) { stack[stack_ptr] = node.children.x; stack_ptr += 1; }
            if (node.children.y != -1) { stack[stack_ptr] = node.children.y; stack_ptr += 1; }
            if (node.children.z != -1) { stack[stack_ptr] = node.children.z; stack_ptr += 1; }
            if (node.children.w != -1) { stack[stack_ptr] = node.children.w; stack_ptr += 1; }
        }
    }

    // Final Update: Apply physics and write back
    my_particle.vel += total_acceleration * params.dt;
    my_particle.pos += my_particle.vel * params.dt;

    particles_dst[p_idx] = my_particle;
}
