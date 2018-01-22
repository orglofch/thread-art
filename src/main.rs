#[macro_use]
pub mod macros;

pub mod checkpoint;
pub mod config;
pub mod genome;
pub mod fitness;
pub mod mutation;
pub mod render;
pub mod shader;

use checkpoint::CheckpointConfig;
use config::{Config, Peg};
use fitness::FitnessConfig;
use genome::Population;
use mutation::MutationConfig;

fn checkpoint(_: &mut Config, _: &Population) {
    println!("Checkpoint");
}

fn main() {
    let mut mutation_config = MutationConfig::new();
    mutation_config.set_mutate_pegs(true);

    let fitness_config = FitnessConfig::new();

    let checkpoint_config = CheckpointConfig::new(checkpoint, 100);

    let mut config = Config::new("data/images/clooney.jpg");
    config.add_thread((0, 0, 0));
    config.set_background_colour((1.0, 1.0, 1.0));
    config.set_mutation_config(mutation_config);
    config.set_fitness_config(fitness_config);
    config.set_checkpoint_config(Some(checkpoint_config));
    config.set_cell_size((4, 4));

    for r in 0..512 / 16 {
        for c in 0..910 / 16 {
            config.add_fixed_peg(Peg::new((c * 4, r * 4)));
        }
    }

    let _ = config.build();
}
