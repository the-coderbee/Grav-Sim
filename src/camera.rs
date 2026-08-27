use bytemuck::{Pod, Zeroable};
use glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub pan: [f32; 2],
    pub zoom: f32,
    pub aspect_ration: f32,
}

pub struct CameraController {
    pub pan: Vec2,
    pub zoom: f32,
    pub is_dragging: bool,
    pub last_mouse_pos: Vec2,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 0.05,
            is_dragging: false,
            last_mouse_pos: Vec2::ZERO,
        }
    }
}
