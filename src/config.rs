extern crate gl;
extern crate glutin;
extern crate image;
extern crate rand;

use genome::Genome;
use checkpoint::CheckpointConfig;
use fitness::FitnessConfig;
use mutation::MutationConfig;
use self::glutin::GlContext;
use self::image::RgbImage;
use self::rand::Rng;
use shader::Shader;
use std::time::Instant;

/// A single peg.
#[derive(Clone, Debug)]
pub struct Peg {
    /// The `(x, y)` grid position of the peg.
    pub(crate) pos: (u32, u32),
}

impl Peg {
    pub fn new(pos: (u32, u32)) -> Peg {
        Peg { pos: pos }
    }
}

pub struct Config {
    /// The filepath of the source image.
    pub(crate) source_img: RgbImage,

    /// The `(r, g, b)` colours of threads that can be used in creating the thread art.
    pub(crate) threads: Vec<(f32, f32, f32)>,

    /// The fixed pegs that a thread can be wrapped around.
    ///
    /// These pegs can not be mutated.
    pub(crate) fixed_pegs: Vec<Peg>,

    /// The initial mutable pegs that a thread can be wrapped around, used in seeding initial `Genomes`.
    ///
    /// Note, these pegs are *not* necessarily fixed. If `mutate_pegs == true` then these pegs may
    /// be removed as part of the regular mutation process.
    pub(crate) mutable_pegs: Vec<Peg>,

    /// The `(r, g, b)` background colour.
    pub(crate) background_colour: (f32, f32, f32),

    /// The `(width, height)` of the peg grid cells.
    ///
    /// A width and height of 1 means a peg can be placed on any pixel in the image, which may not
    /// be ideal for practical application since pegs are generally have larger diameters in reality.
    ///
    /// Ideally this is a factor of the source image width and height, otherwise the
    /// right side and bottom will accumulate a margin.
    pub(crate) cell_size: (u32, u32),

    pub(crate) mutation_config: MutationConfig,

    pub(crate) fitness_config: FitnessConfig,

    pub(crate) checkpoint_config: Option<CheckpointConfig>,
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
    pub fn add_thread(&mut self, thread: (u8, u8, u8)) -> &Config {
        // TODO(orglofch): Change storage to bytes.
        self.threads.push((
            thread.0 as f32 / 255.0,
            thread.1 as f32 / 255.0,
            thread.2 as f32 / 255.0,
        ));
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

        // Setup window.
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

        // Load shaders.
        unsafe {
            let shader = Shader::create("data/shader/vs.vert", "data/shader/fs.frag");
            gl::UseProgram(shader.id);
        }

        let mut genome = Genome::new(self.mutable_pegs.clone());

        let mut previous_fitness = 0.0;
        let mut i = 0;
        while previous_fitness < self.fitness_config.fitness_termination_threshold {
            let now = Instant::now();

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
            if new_fitness >= previous_fitness || rng.gen_range::<f32>(0.0, 1.0) > 0.99 {
                genome = new_child;
                previous_fitness = new_fitness;
            }

            // Checkpoint if necessary.
            let should_checkpoint = match self.checkpoint_config {
                Some(ref config) => i % config.checkpoint_rate == 0,
                None => false,
            };

            if should_checkpoint {
                // TODO(orglofch): Temporary since we want the checkpoint_config to be mutable as well.
                let checkpoint_config = self.checkpoint_config.take().unwrap();
                (checkpoint_config.checkpoint_fn)(self, &Vec::new()); // TODO(orglofch): populate.
                self.checkpoint_config = Some(checkpoint_config);
            }

            println!("FPS: {}", now.elapsed().subsec_nanos() as f64 / 1e-9);
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

        self.mutation_config.validate();
        // TODO(orglofch): Validate the canvas size isn't infinite if mutate_pegs = true.
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
