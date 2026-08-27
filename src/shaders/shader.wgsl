struct CameraUniform {
    pan: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
};

struct InstanceInput {
    @location(1) pos: vec2<f32>,
    @location(2) vel: vec2<f32>,
    @location(3) mass: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = vertex.quad_pos;

    // 1. Handle the Supermassive Central Core (The Black Hole)
    if (instance.mass >= 1000.0) {
        // Push the central black hole off-screen so it renders as completely empty space
        out.clip_position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
        return out;
    }

    // 2. Orbiting Stars Setup
    let radius = 0.25;

    // Calculate distance from the exact center of the galaxy (0,0)
    let dist_from_center = length(instance.pos);

    // Normalize the distance.
    // IMPORTANT: If your galaxy is still completely cyan, lower this number (e.g., to 1000.0 or 500.0).
    // If it's completely orange, raise this number (e.g., to 5000.0 or 10000.0).
    let norm_dist = clamp(dist_from_center / 100.0, 0.0, 1.0);

    // 3. Dynamic Color Mixing based on Position
    let core_color = vec3<f32>(0.2, 0.8, 1.0);  // Bright Cyan for the inner core
    let edge_color = vec3<f32>(1.0, 0.4, 0.1);  // Warm Orange for the outer arms

    // Smoothly blend between Cyan and Orange based on the distance from the center
    let mixed_color = mix(core_color, edge_color, norm_dist);
    out.color = vec4<f32>(mixed_color, 1.0);

    // 4. Position Math
    let world_pos = instance.pos + (vertex.quad_pos * radius);
    let camera_pos = (world_pos - camera.pan) * camera.zoom;
    let ndc_x = camera_pos.x / camera.aspect_ratio;

    out.clip_position = vec4<f32>(ndc_x, camera_pos.y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist_sq = dot(in.uv, in.uv);

    // Discard pixels outside the circle radius to make smooth round stars
    if (dist_sq > 1.0) {
        discard;
    }

    // Smooth anti-aliased edge falloff for a glowing halo effect
    let alpha_falloff = 1.0 - smoothstep(0.7, 1.0, dist_sq);

    // Use the dynamic distance-based color passed directly from the vertex shader
    var color = in.color;
    color.a *= alpha_falloff;

    return color;
}
