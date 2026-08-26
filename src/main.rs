use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rand::Rng;
use std::f32::consts::TAU;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use egui_wgpu::Renderer as EguiRenderer;
use egui_wgpu::ScreenDescriptor;
use egui_winit::State as EguiWinitState;
use std::time::Instant;

pub struct UiState {
    pub dt: f32,
    pub g_const: f32,
    pub paused: bool,
    pub fps: f32,
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

pub struct State {
    // the order is needed to prevent segmentation fault.
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    quad_buffer: wgpu::Buffer,
    compute_bind_groups: [wgpu::BindGroup; 2],
    particle_buffers: [wgpu::Buffer; 2],
    sim_params_buffer: wgpu::Buffer,
    egui_renderer: EguiRenderer,

    device: wgpu::Device,
    queue: wgpu::Queue,

    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    egui_state: EguiWinitState,
    window: Arc<Window>,

    size: winit::dpi::PhysicalSize<u32>,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    frame_count: usize,
    particle_count: u32,
    egui_ctx: egui::Context,
    ui_state: UiState,
    last_frame_time: Instant,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            size.width = 800;
            size.height = 600;
        }

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .unwrap();

        let info = adapter.get_info();
        println!(
            "Using GPU: {} ({:?}, backend: {:?})",
            info.name, info.device_type, info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
                memory_hints: Default::default(),
                ..Default::default()
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Srgb,
        };
        surface.configure(&device, &surface_config);

        let galaxy_config = GalaxyConfig {
            particle_count: 50_000,
            central_mass: 150_000.0,
            disk_partial_mass: 1.0,
            g_const: 1000.0,
            min_radius: 5.0,
            max_radius: 150.0,
            perturbation_factor: 0.03,
        };
        let initial_particles = generate_galaxy(&galaxy_config);
        let particle_count = initial_particles.len() as u32;

        let sim_params = SimParams {
            g_const: 1000.0,
            softening: 0.5,
            dt: 0.016,
            particle_count: particle_count,
        };

        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params Buffer"),
            contents: bytemuck::bytes_of(&sim_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let gpu_particles: Vec<GpuParticle> = initial_particles
            .iter()
            .map(|p| GpuParticle {
                pos: p.position.to_array(),
                vel: p.velocity.to_array(),
                mass: p.mass,
                _padding: 0.0,
            })
            .collect();

        let particle_bytes = bytemuck::cast_slice(&gpu_particles);

        let buffer_usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;

        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer A"),
            contents: particle_bytes,
            usage: buffer_usage,
        });
        let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer B"),
            contents: particle_bytes,
            usage: buffer_usage,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    // Binding 0: SimParams (Uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: particle_src (Storage, Read-Only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: particle_dst (Storage, Read-Write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A (A -> B)"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_b.as_entire_binding(),
                },
            ],
        });

        let bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B (B -> A)"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_a.as_entire_binding(),
                },
            ],
        });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("../compute.wgsl"));
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                ..Default::default()
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("N-Body Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("cp_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
        });

        let camera_controller = CameraController::new();
        let aspect_ration = size.width as f32 / size.height as f32;

        let camera_uniform = CameraUniform {
            pan: camera_controller.pan.to_array(),
            zoom: camera_controller.zoom,
            aspect_ration,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quadd Vertex Buffer"),
            contents: bytemuck::cast_slice(QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                ..Default::default()
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Instanced Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(QuadVertex::desc()), Some(GpuParticle::desc())],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // egui initialisation
        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();

        // connect egui to the winit window
        let egui_state = EguiWinitState::new(
            egui_ctx.clone(),
            viewport_id,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = EguiRenderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let ui_state = UiState {
            dt: 0.0016,
            g_const: 1000.0,
            paused: false,
            fps: 0.0,
        };
        let last_frame_time = Instant::now();

        Self {
            window,
            surface,
            device,
            queue,
            surface_config,
            size,
            render_pipeline,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            quad_buffer,

            compute_pipeline,
            compute_bind_groups: [bind_group_a, bind_group_b],
            particle_buffers: [buffer_a, buffer_b],
            sim_params_buffer,
            frame_count: 0,
            particle_count,
            // egui
            egui_ctx,
            egui_state,
            egui_renderer,
            ui_state,
            last_frame_time,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn update(&mut self) {
        self.camera_uniform.pan = self.camera_controller.pan.to_array();
        self.camera_uniform.zoom = self.camera_controller.zoom;
        self.camera_uniform.aspect_ration = self.size.width as f32 / self.size.height as f32;

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
        // Example of how you will use it later in update()
        // self.queue.write_buffer(
        //     &self.sim_params_buffer,
        //     0,
        //     bytemuck::bytes_of(&new_sim_params),
        // );
    }

    pub fn render(&mut self) -> Result<(), String> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            _ => {
                return Ok(());
            }
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        self.ui_state.fps = 1.0 / frame_time;

        // build the ui
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
            egui::Window::new("Simulation Controls").show(ctx, |ui| {
                ui.label(format!("FPS: {:.1}", self.ui_state.fps));
                ui.label(format!("Particles: {}", self.particle_count));
                ui.separator();

                ui.checkbox(&mut self.ui_state.paused, "Pause Simulation");
                ui.add(
                    egui::Slider::new(&mut self.ui_state.dt, 0.0001..=0.05).text("Time Step (dt)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.ui_state.g_const, -2000.0..=5000.0)
                        .text("Gravity (G)"),
                );
                if ui.button("Reset Camera").clicked() {
                    self.camera_controller.pan = glam::Vec2::ZERO;
                    self.camera_controller.zoom = 0.05;
                }
            });
        });

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
        let clipped_primtives = self
            .egui_ctx
            .tessellate(full_output.shapes, self.window.scale_factor() as f32);

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        let active_dt = if self.ui_state.paused {
            0.0
        } else {
            self.ui_state.dt
        };
        let new_params = SimParams {
            g_const: self.ui_state.g_const,
            softening: 0.5,
            dt: active_dt,
            particle_count: self.particle_count,
        };

        self.queue
            .write_buffer(&self.sim_params_buffer, 0, bytemuck::bytes_of(&new_params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        for (id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, image_delta);
            }
        }
        // clean up UI textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        full_output.textures_delta.clear();
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primtives,
            &screen_descriptor,
        );

        // compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("N-Body Compute Pass"),
                timestamp_writes: None,
            });

            let bind_group_index = self.frame_count % 2;

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_groups[bind_group_index], &[]);

            let workgroup_count = (self.particle_count + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });

            // if compute wrote from buffer B we must render from buffer B
            let render_buffer_index = (self.frame_count + 1) % 2;

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.particle_buffers[render_buffer_index].slice(..));
            render_pass.draw(0..4, 0..self.particle_count);
            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &clipped_primtives,
                &screen_descriptor,
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.window.pre_present_notify();
        self.queue.present(surface_texture);

        self.frame_count += 1;

        Ok(())
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("GravSim - Particle Simulator [Winit + WGPU]")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
                .with_visible(true);

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let state = pollster::block_on(State::new(window));

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

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec2,
    pub velocity: Vec2,
    pub mass: f32,
}

impl Particle {
    // Constructor for convinience
    pub fn new(x: f32, y: f32, vx: f32, vy: f32, mass: f32) -> Self {
        Self {
            position: Vec2::new(x, y),
            velocity: Vec2::new(vx, vy),
            mass,
        }
    }
}

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct QuadVertex {
    pub position: [f32; 2],
}

impl QuadVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

const QUAD_VERTICES: &[QuadVertex] = &[
    QuadVertex {
        position: [-1.0, -1.0],
    },
    QuadVertex {
        position: [1.0, -1.0],
    },
    QuadVertex {
        position: [-1.0, 1.0],
    },
    QuadVertex {
        position: [1.0, 1.0],
    },
];

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    pub g_const: f32,
    pub softening: f32,
    pub dt: f32,
    pub particle_count: u32,
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
