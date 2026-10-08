# Grav-Sim

This is a GPU-accelerated 2D gravitational N-body simulation written in Rust. It uses the Barnes-Hut algorithm on wgpu compute shaders and simulates a 100,000-body galaxy interactively.

![Grav-Sim simulating a 50,000-body galaxy](docs/grav-sim.gif)

![Simulation with the live profiler panel](docs/grav-sim-ss.png)

## What it does

Grav-Sim generates a disk galaxy with a massive central body and orbiting disk particles, then evolves it under mutual gravity in real time. Every particle attracts every other particle, so structure forms on its own: clumps, streams and clusters emerge from a nearly uniform starting disk.

A live control panel lets you change the gravitational constant, the time step and the Barnes-Hut accuracy threshold (θ) while the simulation runs. A built-in profiler shows where each frame's time goes.

## How it works

### Barnes-Hut instead of brute force

A direct N-body simulation computes every pair of interactions, which is O(N²): 10 billion force calculations per step at 100,000 bodies. Barnes-Hut groups distant particles into a quadtree and treats each distant group as a single mass at its centre of mass. That reduces the cost to O(N log N).

The θ parameter controls the trade-off. A tree node is approximated as one mass when `node_size / distance < θ`. Lower θ is more accurate; higher θ is faster.

### Per-frame pipeline

1. **Read back** the current particle positions from the GPU into a staging buffer.
2. **Build the quadtree on the CPU**, computing the total mass and centre of mass of every node, and flatten it into a GPU-friendly array (`GpuNode`, 48 bytes, explicitly padded for WGSL layout rules).
3. **Upload the tree** to a GPU storage buffer.
4. **Compute pass:** one thread per particle (workgroups of 256) traverses the tree with an explicit stack, since WGSL has no recursion, and accumulates acceleration from nodes that pass the θ test.
5. **Integrate** with semi-implicit (symplectic) Euler: velocity first, then position using the new velocity. A softening term in the distance prevents the force from blowing up when two particles get very close.
6. **Render** all particles in a single instanced draw call, colour-graded by distance from the galactic centre, followed by the egui panel. Stars use additive blending, so overlapping light accumulates and dense regions such as the core and clusters glow brighter than sparse ones. A minimum on-screen size keeps stars visible at any zoom level.

Particle state lives in two buffers used in a ping-pong arrangement: each frame reads from one and writes to the other, so no thread ever reads a position that another thread has already updated in the same step.

### Initial conditions

Disk particles are placed uniformly by area between a minimum and maximum radius and given circular orbital velocities around the central mass, plus a small random perturbation so the disk doesn't stay perfectly symmetric. The central body's velocity is set to cancel the disk's total momentum, so the galaxy as a whole doesn't drift off-screen.

## Performance

Measured on an **NVIDIA GeForce GTX 1650 Ti** (laptop, Vulkan backend), release build, 800×600 window, θ = 0.5, after letting each run settle for 60 seconds. Values are averages over the profiler's 60-frame window. GPU times come from wgpu timestamp queries written at the start and end of each pass.

| Particles | FPS | Frame total | Tree build (CPU) | Tree upload | GPU compute | GPU render | Tree nodes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 143.2 | 6.94 ms | 1.27 ms | 0.25 ms | 2.17 ms | 0.09 ms | ~29,300 |
| 50,000 | 35.8 | 27.84 ms | 7.89 ms | 1.27 ms | 16.83 ms | 0.13 ms | ~150,500 |
| 100,000 | 15.7 | 64.75 ms | 17.98 ms | 2.57 ms | 41.55 ms | 0.26 ms | ~303,900 |

*GPU render includes the egui panel.*

### Where the time goes

At 100,000 particles, a frame breaks down roughly like this:

- **GPU Barnes-Hut traversal: 41.6 ms (64%).** This is the largest cost by far.
- **CPU quadtree build: 18.0 ms (28%).**
- **Tree upload: 2.6 ms (4%).**
- **Drawing the particles: 0.26 ms.** Rendering is effectively free.

Two findings came out of measuring this rather than guessing:

1. **The compute shader, not the CPU, is the bottleneck.** Before adding timestamp queries, the CPU tree build looked like the main cost because it was the only thing the CPU timers could see. The GPU timestamps show the traversal takes more than twice as long.
2. **The frame is serial.** The GPU compute time and the CPU time add up to almost the whole frame (63.4 of 64.8 ms at 100k), because the GPU has to finish before the CPU can read positions back, and the CPU has to finish the tree before the GPU can start the next step. Neither side works while the other does.

Traversal cost also grows faster than the O(N log N) the algorithm suggests: doubling from 50k to 100k particles multiplies GPU compute time by about 2.5×. I haven't profiled the cause yet; the likely candidates are scattered memory access during tree traversal and threads in the same workgroup taking very different paths through the tree.

## Running it

Requires a Rust toolchain and a GPU supported by wgpu (Vulkan, Metal or DirectX 12).

```bash
git clone https://github.com/the-coderbee/Grav-Sim.git
cd Grav-Sim
cargo run --release -- --particles 50000
```

Always use `--release`; debug builds are dramatically slower.

### Controls

| Input | Action |
|---|---|
| Scroll | Zoom |
| Left-drag | Pan |
| **Reset Camera** button | Return to the default view |
| **Pause Simulation** checkbox | Freeze time (rendering continues) |
| Sliders | Time step (dt), gravity (G), θ |

## Limitations and next steps

Ordered by expected impact, based on the measurements above:

- **Make tree traversal cheaper on the GPU.** It is 64% of the frame at 100k. Sorting particles along a space-filling curve (Morton order) would make neighbouring threads walk similar paths through the tree and read nearby memory, which should help with both suspected causes above.
- **Overlap CPU and GPU work.** Building the tree from the previous frame's positions would let the CPU build and GPU traversal run in parallel, at the cost of one frame of latency in the tree. At 100k, the frame would then be bounded by the slower of the two (about 42 ms) instead of their sum (about 60 ms).
- **Build the quadtree on the GPU.** This removes the per-frame readback, the CPU tree build and the upload entirely.
- **Fixed traversal stack.** The compute shader uses a 64-entry stack with no overflow guard, which limits the maximum tree depth it can safely traverse.
- **2D only, single precision.** Positions use `f32`, which limits accuracy for very large or very long-running simulations.

## Built with

[wgpu](https://wgpu.rs/) · [winit](https://github.com/rust-windowing/winit) · [egui](https://github.com/emilk/egui) · [glam](https://github.com/bitshifter/glam-rs) · [bytemuck](https://github.com/Lokathor/bytemuck)
