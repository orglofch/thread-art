#[macro_use]
pub mod macros;

pub mod config;
pub mod shader;

use config::{Config, FitnessConfig, MutationConfig, Peg};

fn main() {
    let mut mutation_config = MutationConfig::new();
    mutation_config.set_mutate_pegs(true);

    let mut fitness_config = FitnessConfig::new();

    let mut config = Config::new("data/images/girl_with_the_pearl_earring.jpg");
    config.add_thread((0.78, 0.6, 0.41));
    config.add_thread((0.15, 0.26, 0.56));
    config.add_thread((0.47, 0.29, 0.14));
    config.set_background_colour((0.0, 0.0, 0.0));
    config.set_mutation_config(mutation_config);
    config.set_fitness_config(fitness_config);
    config.set_cell_size((16, 16));

    config.build();
}
