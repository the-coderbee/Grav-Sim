use bytemuck::{Pod, Zeroable};

pub struct UiState {
    pub dt: f32,
    pub g_const: f32,
    pub theta: f32,
    pub paused: bool,
    pub fps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    pub g_const: f32,
    pub softening: f32,
    pub dt: f32,
    pub particle_count: u32,
    pub theta: f32,
}
