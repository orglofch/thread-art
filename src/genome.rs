extern crate gl;
extern crate rand;

use config::{Config, Peg};
use render::{Vertex, buffer_to_gpu};
use self::rand::Rng;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::ptr;

/// A single action in creating thread art.
///
/// # Example
///
/// {1, 3} => Wrap thread 1 around peg 3.
/// {2, 4} => Wrap thread 2 around peg 4.
/// {1, 2} => Wrap thread 1 around peg 2.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Action {
    /// The id of the thread to use.
    thread_id: usize,

    /// The id of the peg to wrap around.
    peg_id: usize,
}

impl Action {
    fn new(thread_id: usize, peg_id: usize) -> Action {
        Action {
            thread_id: thread_id,
            peg_id: peg_id,
        }
    }
}

/// The structural representation of thread art.
///
/// The full genome is represented as a set of `Genes` which are actions
/// which should be applied in order in order to create thread art.
/// TODO(orglofch): Move public exports to the top.
#[derive(Clone, Debug)]
pub struct Genome {
    /// The individual actions in the representation of thread art.
    ///
    /// These are intentionally ordered to preserve render / construction order.
    actions: Vec<Action>,

    /// The set of mutable pegs which can be wrapped around.
    ///
    /// These are intentionally ordered to use indices as ids. We maintain a
    /// separate mechanism for ensuring pegs are reasonably far from eachother instead
    /// of relying on deduping.
    mutable_pegs: Vec<Peg>,

    // TODO(orglofch): Share between genomes.
    thread_vao: u32,
    thread_vbo: u32,
    thread_ebo: u32,

    peg_vao: u32,
    peg_vbo: u32,
    peg_ebo: u32,
}

impl Genome {
    pub(crate) fn new(initial_pegs: Vec<Peg>) -> Genome {
        let mut thread_vao = 0;
        let mut thread_vbo = 0;
        let mut thread_ebo = 0;

        let mut peg_vao = 0;
        let mut peg_vbo = 0;
        let mut peg_ebo = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut thread_vao);
            gl::GenBuffers(1, &mut thread_vbo);
            gl::GenBuffers(1, &mut thread_ebo);

            gl::GenVertexArrays(1, &mut peg_vao);
            gl::GenBuffers(1, &mut peg_vbo);
            gl::GenBuffers(1, &mut peg_ebo);
        }

        Genome {
            actions: Vec::new(),
            mutable_pegs: initial_pegs,
            thread_vao: thread_vao,
            thread_vbo: thread_vbo,
            thread_ebo: thread_ebo,
            peg_vao: peg_vao,
            peg_vbo: peg_vbo,
            peg_ebo: peg_ebo,
        }
    }

    /// Mutate a `Genome` to inserting or removing structures.
    ///
    /// There are 2 operations:
    /// 1) Injecting a new action into the genome actions.
    /// 2) Removing an existing action from the genome actions.
    pub(crate) fn mutate(&mut self, config: &Config) {
        let mutation_config = &config.mutation_config;

        let mut prob_sum = mutation_config.insert_action_prob + mutation_config.remove_action_prob;
        if mutation_config.mutate_pegs {
            prob_sum += mutation_config.insert_peg_prob + mutation_config.remove_peg_prob +
                mutation_config.move_peg_prob;
        }

        let mut rng = rand::thread_rng();

        // If the sum of the probabilities is greater than 1, normalize the probability range
        // so the sum consistutes a 100% probability. Otherwise, allow the total probability
        // to be less than 1 to allow for a probability of no mutations.
        let mut prob = rng.gen_range::<f32>(0.0, prob_sum.max(1.0));

        if prob < mutation_config.insert_action_prob {
            self.mutate_insert_action(config);
            return;
        }
        prob -= mutation_config.insert_action_prob;

        if prob < mutation_config.remove_action_prob {
            self.mutate_remove_action();
            return;
        }
        prob -= mutation_config.remove_action_prob;

        if mutation_config.mutate_pegs {
            if prob < mutation_config.insert_peg_prob {
                self.mutate_insert_peg(config);
                return;
            }
            prob -= mutation_config.insert_peg_prob;

            if prob < mutation_config.remove_peg_prob {
                self.mutate_remove_peg(config);
                return;
            }
            prob -= mutation_config.remove_peg_prob;

            if prob < mutation_config.move_peg_prob {
                self.mutate_move_peg(config);
                return;
            }
        }
    }

    /// Mutate a `Genome` by inserting a new `ActionGene`.
    fn mutate_insert_action(&mut self, config: &Config) {
        // TODO(orglofch): Prevent actions wrapping a thread around the same peg multiple times.
        if self.mutable_pegs.is_empty() && config.fixed_pegs.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();

        let thread_id = rng.gen_range::<usize>(0, config.threads.len());
        let peg_id = rng.gen_range::<usize>(0, config.fixed_pegs.len() + self.mutable_pegs.len());

        let action = Action::new(thread_id, peg_id);

        // Select a position in the action sequence to insert into.
        // Note, t's possible to insert at the beginning and end.
        let insertion_pos = rng.gen_range::<usize>(0, self.actions.len() + 1);

        self.actions.insert(insertion_pos, action);
    }

    /// Mutate a `Genome` by removing an existing `ActionGene`.
    fn mutate_remove_action(&mut self) {
        if self.actions.len() == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        let removal_pos = rng.gen_range::<usize>(0, self.actions.len());

        // TODO(orglofch): Swap remove.
        self.actions.remove(removal_pos);
    }

    /// Mutate a `Genome` by inserting a new `PegGene`.
    fn mutate_insert_peg(&mut self, config: &Config) {
        let mut rng = rand::thread_rng();

        let x = rng.gen_range::<u32>(0, config.source_img.width() / config.cell_size.0);
        let y = rng.gen_range::<u32>(0, config.source_img.height() / config.cell_size.1);

        let peg = Peg::new((x, y));

        // Generated pegs are never fixed
        self.mutable_pegs.push(peg);
    }

    /// Mutate a `Genome` by removing an existing `PegGene`.
    fn mutate_remove_peg(&mut self, config: &Config) {
        if self.mutable_pegs.len() == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        let removal_pos = rng.gen_range::<usize>(0, self.mutable_pegs.len());

        self.mutable_pegs.remove(removal_pos);

        // The real position including fixed pegs.
        let real_pos = removal_pos + config.fixed_pegs.len();

        // Removal all actions connection to this peg.
        // TODO(orglofch): This is all quite inefficient when threads is large.
        self.actions.retain(|action| {
            let peg_id = action.peg_id;

            if peg_id == real_pos {
                return false;
            }
            return true;
        });

        // Modify actions referencing larger peg ids to account for the removal
        // of the lower peg.
        self.actions
            .iter_mut()
            .filter(|action| action.peg_id > real_pos)
            .for_each(|action| action.peg_id -= 1);
    }

    /// Mutate a `Genome` by moving an existing `PegGene`.
    fn mutate_move_peg(&mut self, config: &Config) {
        if self.mutable_pegs.len() == 0 {
            return;
        }

        let mut rng = rand::thread_rng();

        let move_pos = rng.gen_range::<usize>(0, self.mutable_pegs.len());

        let peg = &mut self.mutable_pegs[move_pos];

        let x = rng.gen_range::<u32>(0, config.source_img.width() / config.cell_size.0);
        let y = rng.gen_range::<u32>(0, config.source_img.height() / config.cell_size.1);

        peg.pos = (x, y);
    }

    /// Reindex the `Genome` threads into a format which can be buffered to the GPU for rendering.
    fn reindex_threads_for_rendering(&self, config: &Config) -> (Vec<Vertex>, Vec<u32>) {
        // TODO(orglofch): Presize.
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut final_index_by_vertex: HashMap<Action, usize> = HashMap::new();
        let mut last_index_by_thread_id: HashMap<usize, usize> = HashMap::new();
        for gene in &self.actions {
            let thread = config.threads[gene.thread_id];
            let peg = if gene.peg_id < config.fixed_pegs.len() {
                &config.fixed_pegs[gene.peg_id]
            } else {
                &self.mutable_pegs[gene.peg_id - config.fixed_pegs.len()]
            };

            // Convert the cell position to view coordinates.
            let view_x = (peg.pos.0 * config.cell_size.0) as f32 /
                config.source_img.width() as f32 * 2.0 - 1.0;
            let view_y = (peg.pos.1 * config.cell_size.1) as f32 /
                config.source_img.height() as f32 * 2.0 - 1.0;

            let vertex = Vertex::new((view_x, view_y), thread);

            if !final_index_by_vertex.contains_key(gene) {
                vertices.push(vertex);
            }

            let current_index = *final_index_by_vertex.entry((*gene).clone()).or_insert(
                vertices.len() -
                    1,
            );

            // Try to obtain the index of the start of the line segment if it exists.
            if !last_index_by_thread_id.contains_key(&gene.thread_id) {
                last_index_by_thread_id.insert(gene.thread_id, current_index);
                continue;
            }

            let prev_index = *last_index_by_thread_id.get(&gene.thread_id).unwrap();

            // Push the previous index as the start of the line segement since we're rendering
            // lines instead of line strips.
            indices.push(prev_index as u32);

            // Push the new index to complete the line segment.
            indices.push(current_index as u32);

            // Record the current line segments tail as the head of the next segment.
            last_index_by_thread_id.insert(gene.thread_id, current_index);
        }

        (vertices, indices)
    }

    /// Render a `Genome` to the onscreen buffer or back-buffer.
    pub(crate) fn render_threads(&self, config: &Config) {
        let (thread_vertices, thread_indices) = self.reindex_threads_for_rendering(config);

        if thread_vertices.len() != 0 && thread_indices.len() != 0 {
            unsafe {
                buffer_to_gpu(
                    &thread_vertices,
                    &thread_indices,
                    self.thread_vao,
                    self.thread_vbo,
                    self.thread_ebo,
                );

                gl::BindVertexArray(self.thread_vao);
                gl::DrawElements(
                    gl::LINES,
                    thread_indices.len() as i32,
                    gl::UNSIGNED_INT,
                    ptr::null(),
                );

                gl::BindVertexArray(0);
            }
        }
    }

    /// Reindex the `Genome` pegs into a format which can be buffered to the GPU for rendering.
    fn reindex_pegs_for_rendering(&self, config: &Config) -> (Vec<Vertex>, Vec<u32>) {
        // TODO(orglofch): Presize.
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for peg in self.mutable_pegs.iter() {
            // Convert the cell position to view coordinates.
            let view_x = (peg.pos.0 * config.cell_size.0) as f32 /
                config.source_img.width() as f32 * 2.0 - 1.0;
            let view_y = (peg.pos.1 * config.cell_size.1) as f32 /
                config.source_img.height() as f32 * 2.0 - 1.0;

            let vertex = Vertex::new((view_x, view_y), (0.5, 0.5, 0.5));

            vertices.push(vertex);
            indices.push(vertices.len() as u32);
        }

        // TODO(orglofch): Doing this each frame is unnecessary.
        for peg in config.fixed_pegs.iter() {
            // Convert the position to view coordinates.
            let view_x = (peg.pos.0 * config.cell_size.0) as f32 /
                config.source_img.width() as f32 * 2.0 - 1.0;
            let view_y = (peg.pos.1 * config.cell_size.1) as f32 /
                config.source_img.height() as f32 * 2.0 - 1.0;

            let vertex = Vertex::new((view_x, view_y), (0.5, 0.5, 0.5));

            vertices.push(vertex);
            indices.push(vertices.len() as u32);
        }

        (vertices, indices)
    }

    /// Render a `Genome` to the onscreen buffer or back-buffer.
    pub(crate) fn render_pegs(&self, config: &Config) {
        let (peg_vertices, peg_indices) = self.reindex_pegs_for_rendering(config);
        if peg_vertices.len() != 0 && peg_indices.len() != 0 {
            unsafe {
                buffer_to_gpu(
                    &peg_vertices,
                    &peg_indices,
                    self.peg_vao,
                    self.peg_vbo,
                    self.peg_ebo,
                );

                gl::BindVertexArray(self.peg_vao);
                gl::DrawElements(
                    gl::POINTS,
                    peg_indices.len() as i32,
                    gl::UNSIGNED_INT,
                    ptr::null(),
                );
            }
        }
    }

    /// Calculates the fitness `Genome` relative to it's source image.
    pub(crate) fn fitness(&self, config: &Config) -> f32 {
        let pixel_values = config.source_img.width() * config.source_img.height() * 3;

        // TODO(orglofch): Retina displays render at double the pixel size.
        // instead of reading everythinzg, render into a FBO half the window size.
        let mut buffer: Vec<u8> = vec![0_u8; (pixel_values * 4) as usize];

        // Read back the rendered state.
        unsafe {
            gl::ReadPixels(
                0,
                0,
                config.source_img.width() as i32 * 2,
                config.source_img.height() as i32 * 2,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                &mut buffer[0] as *mut u8 as *mut c_void,
            );
        }

        // Construct a new buffer to hold the difference.
        let mut diff_buffer: Vec<u8> = vec![0_u8; (pixel_values * 4) as usize];

        let mut fitness = 0.0;
        for r in 0..config.source_img.height() {
            for c in 0..config.source_img.width() {
                // Note, y-axis is inverted between the two representations.
                let pixel = config.source_img.get_pixel(
                    c as u32,
                    config.source_img.height() - r as u32 - 1,
                );

                let i = (c * 2 + r * 2 * config.source_img.width() * 2) * 3;

                let (gene_r, gene_g, gene_b) = (
                    buffer[i as usize] as i32,
                    buffer[(i + 1) as usize] as i32,
                    buffer[(i + 2) as usize] as i32,
                );
                let (source_r, source_g, source_b) = (
                    pixel.data[0] as i32,
                    pixel.data[1] as i32,
                    pixel.data[2] as i32,
                );

                diff_buffer[i as usize] = (gene_r - source_r).abs() as u8;
                diff_buffer[i as usize] = (gene_g - source_g).abs() as u8;
                diff_buffer[i as usize] = (gene_b - source_b).abs() as u8;

                let mut diff = 0.0;
                diff += (((gene_r - source_r) as f32 / 255.0).abs() + 1.0).log(2.0);
                diff += (((gene_g - source_g) as f32 / 255.0).abs() + 1.0).log(2.0);
                diff += (((gene_b - source_b) as f32 / 255.0).abs() + 1.0).log(2.0);

                //diff += ((gene_r - source_r) as f32 / 255.0).powi(2);
                //diff += ((gene_g - source_g) as f32 / 255.0).powi(2);
                //diff += ((gene_b - source_b) as f32 / 255.0).powi(2);

                fitness += diff;
            }
        }

        // TODO(orglofch): Blur shader.

        /*TODO(orglofch): Buffer iterator for (i, &byte) in source_img.iter().enumerate() {
            println!("Buffer {}", (byte as f32 / 255.0));
            let value = byte as f32 / 255.0;

            let diff = (buffer[i] - value).powi(2);

            fitness += diff as f32;
        }*/

        // Return the normalized fitness.
        return 1.0 - fitness / pixel_values as f32 - 0.000000025 * self.actions.len() as f32;
    }
}

pub type Population = Vec<Genome>;
