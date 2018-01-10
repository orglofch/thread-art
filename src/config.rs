extern crate gl;
extern crate glutin;
extern crate image;
extern crate rand;

use self::glutin::GlContext;
use self::image::{ImageBuffer, Rgb, RgbImage};
use self::rand::Rng;
use shader::Shader;
use std::collections::HashMap;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;

pub struct CheckpointConfig {
    /// The function to invoke on every checkpoint.
    ///
    /// Checkpointing functions are free to change the configuration based on the
    /// state of the configuration or however else they can think of.
    checkpoint_fn: fn(&mut Config, &Genome),

    /// The number of iterations / generations between checkpointing function invocations.
    checkpoint_rate: u32,
}

impl CheckpointConfig {
    pub fn new(checkpoint_fn: fn(&mut Config, &Genome), checkpoint_rate: u32) -> CheckpointConfig {
        CheckpointConfig {
            checkpoint_fn: checkpoint_fn,
            checkpoint_rate: checkpoint_rate,
        }
    }
}

pub struct MutationConfig {
    /// The probability of inserting an action during `Genome` mutation.
    insert_action_prob: f32,

    /// The probability of removing an action during `Genome` mutation.
    ///
    /// Must be smaller than `new_action_prob` otherwise the structure
    /// won't advance.
    remove_action_prob: f32,

    // TODO(orglofch): Consider adding actions for moving a thread or moving a peg instead of requiring
    // a deletion than an insertion to accomplish the same thing. This would be especially useful for
    // pegs since they could maintain their same general structure assuming we "moved" to a nearby
    // neighbour peg.
    /// The probability of inserting a peg during `Genome` mutation.
    insert_peg_prob: f32,

    /// The probability of removing a peg during `Genome` mutation.
    ///
    /// Mut be smaller than `insert_peg_prob` otherwise the structure
    /// won't advance.
    remove_peg_prob: f32,

    /// The probability of moving a peg during `Genome` mutation.
    move_peg_prob: f32,

    /// Whether the positions of pegs should be mutated in addition to
    /// the thread wrapping.
    ///
    /// TODO(orglofch): Potentially turn of thread mutation as well allowing for mutating a fixed
    /// topology of threads through only moving the pegs around.
    mutate_pegs: bool,
}

impl MutationConfig {
    pub fn new() -> MutationConfig {
        MutationConfig::default()
    }

    pub fn set_insert_action_prob(&mut self, prob: f32) -> &MutationConfig {
        self.insert_action_prob = prob;
        self
    }

    pub fn set_remove_action_prob(&mut self, prob: f32) -> &MutationConfig {
        self.remove_action_prob = prob;
        self
    }

    pub fn set_insert_peg_prob(&mut self, prob: f32) -> &MutationConfig {
        self.insert_peg_prob = prob;
        self
    }

    pub fn set_remove_peg_prob(&mut self, prob: f32) -> &MutationConfig {
        self.remove_peg_prob = prob;
        self
    }

    pub fn set_move_peg_prob(&mut self, prob: f32) -> &MutationConfig {
        self.move_peg_prob = prob;
        self
    }

    pub fn set_mutate_pegs(&mut self, enabled: bool) -> &MutationConfig {
        self.mutate_pegs = enabled;
        self
    }

    fn validate(&self) {
        assert!(
            self.insert_action_prob > self.remove_action_prob,
            "Probability of additive action mutation should be larger than the probability of a destructive action mutation"
        );
        assert!(
            self.insert_peg_prob > self.remove_peg_prob,
            "Probability of additive peg mutation should be larger than the probability of a destructive peg mutation"
        );
    }
}

impl Default for MutationConfig {
    fn default() -> MutationConfig {
        MutationConfig {
            insert_action_prob: 0.2,
            remove_action_prob: 0.2,
            insert_peg_prob: 0.2,
            remove_peg_prob: 0.2,
            move_peg_prob: 0.2,
            mutate_pegs: false,
        }
    }
}

pub struct FitnessConfig {
    /// The coefficient of the factor used in punishing large Genomes.
    ///
    /// A values of 0 indicates that there is no complexity punishment.
    complexity_punishment_coefficient: f32,

    /// The fitness threshold at which the generator will terminate.
    fitness_termination_threshold: f32,
}

impl FitnessConfig {
    pub fn new() -> FitnessConfig {
        FitnessConfig::default()
    }

    pub fn set_fitnss_termination_threshold(&mut self, threshold: f32) -> &FitnessConfig {
        self.fitness_termination_threshold = threshold;
        self
    }

    pub fn set_complexity_punishment_coefficient(&mut self, coefficient: f32) -> &FitnessConfig {
        self.complexity_punishment_coefficient = coefficient;
        self
    }
}

impl Default for FitnessConfig {
    fn default() -> FitnessConfig {
        FitnessConfig {
            fitness_termination_threshold: 1.0,
            complexity_punishment_coefficient: 0.0,
        }
    }
}

pub struct Config {
    /// The filepath of the source image.
    source_img: RgbImage,

    /// The `(r, g, b)` colours of threads that can be used in creating the thread art.
    threads: Vec<(f32, f32, f32)>,

    /// The fixed pegs that a thread can be wrapped around.
    ///
    /// These pegs can not be mutated.
    fixed_pegs: Vec<Peg>,

    /// The initial mutable pegs that a thread can be wrapped around, used in seeding initial `Genomes`.
    ///
    /// Note, these pegs are *not* necessarily fixed. If `mutate_pegs == true` then these pegs may
    /// be removed as part of the regular mutation process.
    mutable_pegs: Vec<Peg>,

    /// The background colour.
    background_colour: (f32, f32, f32),

    /// The `(width, height)` of the peg grid cells.
    ///
    /// A width and height of 1 means a peg can be placed on any pixel in the image, which may not
    /// be ideal for practical application since pegs are generally have larger diameters in reality.
    ///
    /// Ideally this is a factor of the source image width and height, otherwise the
    /// right side and bottom will accumulate a margin.
    cell_size: (u32, u32),

    mutation_config: MutationConfig,

    fitness_config: FitnessConfig,

    checkpoint_config: Option<CheckpointConfig>,
}

impl Config {
    pub fn new(source_img_path: &str) -> Config {
        let source_img = image::open(source_img_path).unwrap().to_rgb();

        Config {
            source_img: source_img,
            threads: Vec::new(),
            fixed_pegs: Vec::new(),
            mutable_pegs: Vec::new(),
            background_colour: (0.0, 0.0, 0.0),
            cell_size: (1, 1),
            mutation_config: MutationConfig::default(),
            fitness_config: FitnessConfig::default(),
            checkpoint_config: None,
        }
    }

    /// Add a new thread to the configuration.
    ///
    /// # Arguments
    ///
    /// * `thread` - The `(r, g, b)` colour of the thread.
    pub fn add_thread(&mut self, thread: (f32, f32, f32)) -> &Config {
        self.threads.push(thread);
        self
    }

    /// Add a new fixed peg to the configuration.
    pub fn add_fixed_peg(&mut self, peg: Peg) -> &Config {
        self.fixed_pegs.push(peg);
        self
    }

    /// Add a new mutable peg to the configuration.
    pub fn add_mutable_peg(&mut self, peg: Peg) -> &Config {
        self.mutable_pegs.push(peg);
        self
    }

    pub fn set_background_colour(&mut self, background_colour: (f32, f32, f32)) -> &Config {
        self.background_colour = background_colour;
        self
    }

    pub fn set_cell_size(&mut self, size: (u32, u32)) -> &Config {
        self.cell_size = size;
        self
    }

    pub fn set_mutation_config(&mut self, mutation_config: MutationConfig) -> &Config {
        self.mutation_config = mutation_config;
        self
    }

    pub fn set_fitness_config(&mut self, fitness_config: FitnessConfig) -> &Config {
        self.fitness_config = fitness_config;
        self
    }

    pub fn set_checkpoint_config(
        &mut self,
        checkpoint_config: Option<CheckpointConfig>,
    ) -> &Config {
        self.checkpoint_config = checkpoint_config;
        self
    }

    /// Create a new `Genome` through simulated evolution.
    pub fn build(&mut self) -> Genome {
        // Validate the configuration a runtime. We don't validate at set time since some parameters
        // are dependent on one another and we don't want to require set ordering.
        self.validate();

        let mut events_loop = glutin::EventsLoop::new();
        let window = glutin::WindowBuilder::new()
            .with_title("Thread Art Live Evolution")
            .with_dimensions(self.source_img.width(), self.source_img.height());
        let context = glutin::ContextBuilder::new().with_vsync(true);

        let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

        unsafe {
            gl_window.make_current().unwrap();
            gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);
        }

        // Set up GL Context.
        unsafe {
            // TODO(orglofch): Support using an arbitrary texture for the background.
            gl::ClearColor(
                self.background_colour.0,
                self.background_colour.1,
                self.background_colour.2,
                1.0,
            );

            gl::Enable(gl::LINE_SMOOTH);
            gl::Disable(gl::DEPTH_TEST);
            gl::Disable(gl::CULL_FACE);
        }

        unsafe {
            let shader = Shader::create("data/shader/vs.vert", "data/shader/fs.frag");
            gl::UseProgram(shader.id);
        }

        let mut genome = Genome::new(self.mutable_pegs.clone());

        //genome.mutable_pegs.push(Peg::new((512, 512)));
        //genome.mutable_pegs.push(Peg::new((1020, 1020)));
        //genome.actions.push(Action::new(0, 0));
        //genome.actions.push(Action::new(0, 1));

        let mut previous_fitness = 0.0;
        let mut i = 0;
        loop {
            events_loop.poll_events(|_| ());

            unsafe {
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }

            // TODO(orglofch): Support using an arbitrary texture for the background.

            let mut new_child = genome.clone();
            new_child.mutate(&self);

            new_child.render_threads(&self);

            // TODO(orglofch): This isn't really tied to the genome, more the buffer.
            let new_fitness = new_child.fitness(&self);

            // Render pegs after the treads so they don't affect fitness.
            new_child.render_pegs(&self);

            gl_window.swap_buffers().unwrap();

            println!("{}", previous_fitness);

            let mut rng = rand::thread_rng();

            // TODO(orglofch): Add boltzmann probability based on fitness for lower
            // fitness genomes.
            if new_fitness >= previous_fitness || rng.gen_range::<f32>(0.0, 1.0) > 1.1 {
                genome = new_child;
                previous_fitness = new_fitness;
            }

            // Checkpoint if necessary.
            // TODO(orglofch): Checkpoint

            i += 1;
        }

        genome
    }

    /// Validates the configuration prior to running.
    fn validate(&self) {
        assert!(
            !self.threads.is_empty(),
            "There needs to be at least 1 thread"
        );
        // TODO(orglofch): Validate the pegs are reasonably far from eachother.

        // TODO(orglofch): Validate the canvas size isn't infinite if mutate_pegs = true.
    }
}

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

/// A single peg.
#[derive(Clone, Debug)]
pub struct Peg {
    /// The `(x, y)` grid position of the peg.
    pos: (u32, u32),
}

impl Peg {
    pub fn new(pos: (u32, u32)) -> Peg {
        Peg { pos: pos }
    }
}

/// Vertex information used in rendering the thread art.
#[derive(Clone, Debug)]
struct Vertex {
    /// The `(x, y)` position of the peg in view space (E.g. ((-1, 1), (-1, 1))).
    pos: (f32, f32),

    colour: (f32, f32, f32),
}

impl Vertex {
    fn new(peg: (f32, f32), thread: (f32, f32, f32)) -> Vertex {
        Vertex {
            pos: peg,
            colour: thread,
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
    fn new(initial_pegs: Vec<Peg>) -> Genome {
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
    fn mutate(&mut self, config: &Config) {
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
    fn render_threads(&self, config: &Config) {
        let (thread_vertices, thread_indices) = self.reindex_threads_for_rendering(config);

        if thread_vertices.len() != 0 && thread_indices.len() != 0 {
            unsafe {
                self.buffer_to_gpu(
                    &thread_vertices,
                    &thread_indices,
                    self.thread_vao,
                    self.thread_vbo,
                    self.thread_ebo);

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
    fn render_pegs(&self, config: &Config) {
        let (peg_vertices, peg_indices) = self.reindex_pegs_for_rendering(config);
        if peg_vertices.len() != 0 && peg_indices.len() != 0 {
            unsafe {
                self.buffer_to_gpu(
                    &peg_vertices,
                    &peg_indices,
                    self.peg_vao,
                    self.peg_vbo,
                    self.peg_ebo);

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

    /// Buffer `Vertex` information to the GPU.
    unsafe fn buffer_to_gpu(&self, vertices: &Vec<Vertex>, indices: &Vec<u32>, vao: u32, vbo: u32, ebo: u32) {
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        let size = (vertices.len() * size_of::<Vertex>()) as isize;
        let data = &vertices[0] as *const Vertex as *const c_void;
        gl::BufferData(gl::ARRAY_BUFFER, size, data, gl::STATIC_DRAW);

        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        let size = (indices.len() * size_of::<u32>()) as isize;
        let data = &indices[0] as *const u32 as *const c_void;
        gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, size, data, gl::STATIC_DRAW);

        let size = size_of::<Vertex>() as i32;
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            size,
            offset_of!(Vertex, pos) as *const c_void,
        );
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(
            1,
            3,
            gl::FLOAT,
            gl::FALSE,
            size,
            offset_of!(Vertex, colour) as *const c_void,
        );

        gl::BindVertexArray(0);
    }

    /// Calculates the fitness `Genome` relative to it's source image.
    fn fitness(&self, config: &Config) -> f32 {
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
                let pixel = config.source_img.get_pixel(c as u32, r as u32);

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
                diff += ((gene_r - source_r) as f32 / 255.0).powi(2);
                diff += ((gene_g - source_g) as f32 / 255.0).powi(2);
                diff += ((gene_b - source_b) as f32 / 255.0).powi(2);

                fitness += diff;
            }
        }

        /*TODO(orglofch): Buffer iterator for (i, &byte) in source_img.iter().enumerate() {
            println!("Buffer {}", (byte as f32 / 255.0));
            let value = byte as f32 / 255.0;

            let diff = (buffer[i] - value).powi(2);

            fitness += diff as f32;
        }*/

        // Return the normalized fitness.
        return 1.0 - fitness / pixel_values as f32 - 0.000000000001 * self.actions.len() as f32;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_reindex_for_rendering() {
        let conf = Config::new("/yolo");
    }
}
