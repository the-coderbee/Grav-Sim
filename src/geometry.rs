use bytemuck::{Pod, Zeroable};

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

pub const QUAD_VERTICES: &[QuadVertex] = &[
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
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuNode {
    pub center_of_mass: [f32; 2],
    pub mass: f32,
    pub size: f32,
    pub children: [i32; 4],
    pub is_leaf: i32,
    pub _padding: [i32; 3],
}

impl GpuNode {
    pub fn new_empty(size: f32) -> Self {
        Self {
            center_of_mass: [0.0, 0.0],
            mass: 0.0,
            size,
            children: [-1, -1, -1, -1],
            is_leaf: 1,
            _padding: [0; 3],
        }
    }
}

#[derive(Clone, Copy)]
pub struct BoundingBox {
    pub center_x: f32,
    pub center_y: f32,
    pub half_width: f32,
}

pub struct QuadTree {
    pub nodes: Vec<GpuNode>,
}

impl QuadTree {
    pub fn new(size: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(size),
        }
    }

    pub fn build(&mut self, particles: &[[f32; 2]], masses: &[f32]) {
        self.nodes.clear();

        let bounds = self.calculate_root_bounds(particles);

        self.nodes.push(GpuNode::new_empty(bounds.half_width * 2.0));

        for (i, &pos) in particles.iter().enumerate() {
            self.insert(0, pos, masses[i], bounds);
        }

        self.compute_mass_distribution(0);
    }

    fn calculate_root_bounds(&self, particles: &[[f32; 2]]) -> BoundingBox {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for p in particles {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }

        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        let width = max_x - min_x;
        let height = max_y - min_y;
        let half_width = (width.max(height) / 2.0).max(0.0001);

        BoundingBox {
            center_x,
            center_y,
            half_width,
        }
    }

    fn insert(
        &mut self,
        mut current_idx: usize,
        pos: [f32; 2],
        mass: f32,
        mut bounds: BoundingBox,
    ) {
        // we loop to traverse down the tree dynamically
        loop {
            let node = self.nodes[current_idx];

            if node.is_leaf == 1 {
                if node.mass == 0.0 {
                    // empty leaf
                    self.nodes[current_idx].center_of_mass = pos;
                    self.nodes[current_idx].mass = mass;
                    return;
                }

                if (node.center_of_mass[0] - pos[0]).abs() < 1e-5
                    && (node.center_of_mass[1] - pos[1]).abs() < 1e-5
                {
                    self.nodes[current_idx].mass += mass;
                    return;
                }

                // mark as branch
                self.nodes[current_idx].is_leaf = 0;

                // extract the old star's data
                let old_pos = node.center_of_mass;
                let old_mass = node.mass;

                // create 4 empty child nodes
                let child_size = bounds.half_width; // new size is half the parent's width
                let base_child_idx = self.nodes.len();
                for _ in 0..4 {
                    self.nodes.push(GpuNode::new_empty(child_size));
                }

                // link children to parent
                self.nodes[current_idx].children = [
                    base_child_idx as i32,
                    (base_child_idx + 1) as i32,
                    (base_child_idx + 2) as i32,
                    (base_child_idx + 3) as i32,
                ];

                // re-insert the old star into correct new child
                let old_quadrant = Self::get_quadrant(&bounds, old_pos);
                let old_child_bounds = Self::get_child_bounds(&bounds, old_quadrant);

                // note: use recursion for the displaced old star but continue the loop
                // for the new star to avoid double-recursion complexity.
                self.insert(
                    base_child_idx + old_quadrant,
                    old_pos,
                    old_mass,
                    old_child_bounds,
                );
            }
            let quadrant = Self::get_quadrant(&bounds, pos);
            current_idx = self.nodes[current_idx].children[quadrant] as usize;
            bounds = Self::get_child_bounds(&bounds, quadrant)
        }
    }

    fn get_quadrant(bounds: &BoundingBox, pos: [f32; 2]) -> usize {
        let mut index = 0;
        if pos[0] >= bounds.center_x {
            index += 1
        } // right half
        if pos[1] >= bounds.center_y {
            index += 2
        } // bottom half
        index
    }

    fn get_child_bounds(parent: &BoundingBox, quadrant: usize) -> BoundingBox {
        let quarter = parent.half_width / 2.0;
        let dir_x = if quadrant % 2 == 0 { -1.0 } else { 1.0 };
        let dir_y = if quadrant < 2 { -1.0 } else { 1.0 };

        BoundingBox {
            center_x: parent.center_x + (dir_x * quarter),
            center_y: parent.center_y + (dir_y * quarter),
            half_width: quarter,
        }
    }

    fn compute_mass_distribution(&mut self, node_idx: usize) {
        let node = self.nodes[node_idx];

        if node.is_leaf == 1 {
            return;
        }

        let mut total_mass = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;

        for &child_idx in &node.children {
            if child_idx != -1 {
                let c_idx = child_idx as usize;
                // recursive bottom up call
                self.compute_mass_distribution(c_idx);

                let child = self.nodes[c_idx];
                total_mass += child.mass;
                weighted_x += child.center_of_mass[0] * child.mass;
                weighted_y += child.center_of_mass[1] * child.mass;
            }
        }

        self.nodes[node_idx].mass = total_mass;
        if total_mass > 0.0 {
            self.nodes[node_idx].center_of_mass[0] = weighted_x / total_mass;
            self.nodes[node_idx].center_of_mass[1] = weighted_y / total_mass;
        }
    }
}
