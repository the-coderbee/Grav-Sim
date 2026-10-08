use glam::Vec2;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::state::State;

#[derive(Debug, Clone, Copy)]
pub struct AppConfig {
    pub particles: usize,
    pub width: u32,
    pub height: u32,
}

pub struct App {
    state: Option<State>,
    config: AppConfig,
}

impl App {
    pub fn new(app_config: AppConfig) -> Self {
        Self {
            state: None,
            config: app_config,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("GravSim - Particle Simulator [Winit + WGPU]")
                .with_inner_size(winit::dpi::PhysicalSize::new(
                    self.config.width,
                    self.config.height,
                ))
                .with_visible(true);

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let state = pollster::block_on(State::new(window, self.config));

            state.window().request_redraw();

            self.state = Some(state);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() {
            Some(state) => state,
            None => return,
        };

        if state.window().id() != _id {
            return;
        }

        // let egui intercept events first
        let response = state.egui_state.on_window_event(&state.window, &event);
        if response.consumed {
            return; // if egui handled the click/scroll, don't pass it to our camera
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_factor = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => 1.0 + y * 0.1,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => 1.0 + (pos.y as f32) * 0.005,
                };
                state.camera_controller.zoom *= zoom_factor;
            }

            WindowEvent::MouseInput {
                state: button_state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                state.camera_controller.is_dragging = button_state.is_pressed();
            }

            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = Vec2::new(position.x as f32, position.y as f32);
                if state.camera_controller.is_dragging {
                    let delta = current_pos - state.camera_controller.last_mouse_pos;
                    state.camera_controller.pan.x -=
                        delta.x / (state.camera_controller.zoom * state.size.width as f32);
                    state.camera_controller.pan.y -=
                        delta.y / (state.camera_controller.zoom * state.size.height as f32);
                }
                state.camera_controller.last_mouse_pos = current_pos;
            }

            WindowEvent::RedrawRequested => {
                state.update();
                if let Err(e) = state.render() {
                    eprintln!("Render error: {e}");
                }
                state.window().request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window().request_redraw();
        }
    }
}
