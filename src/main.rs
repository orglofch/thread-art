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
use genome::Genome;
use mutation::MutationConfig;

fn checkpoint(config: &mut Config, genome: &Genome) {
    println!("Checkpoint");
}

fn main() {
    let mut mutation_config = MutationConfig::new();
    //mutation_config.set_mutate_pegs(true);

    let mut fitness_config = FitnessConfig::new();

    let mut checkpoint_config = CheckpointConfig::new(checkpoint, 100);

    let mut config = Config::new("data/images/landscape.jpg");
    config.add_thread((225, 212, 150));
    config.add_thread((192, 194, 162));
    config.add_thread((138, 120, 59));
    config.add_thread((204, 176, 93));
    config.add_thread((165, 151, 74));
    config.add_thread((96, 75, 36));
    config.add_thread((29, 20, 14));
    config.add_thread((63, 46, 26));
    config.add_thread((125, 125, 79));
    config.set_background_colour((1.0, 1.0, 1.0));
    config.set_mutation_config(mutation_config);
    config.set_fitness_config(fitness_config);
    config.set_checkpoint_config(Some(checkpoint_config));
    config.set_cell_size((8, 8));

    for r in 0..256 {
        for c in 0..256 {
            config.add_fixed_peg(Peg::new((r / 8, c / 8)));
        }
    }

    config.build();
}
