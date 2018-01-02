#[macro_use]
pub mod macros;

pub mod config;
pub mod shader;

use config::{Config, FitnessConfig, MutationConfig, Peg};

fn main() {
    let mut mutation_config = MutationConfig::new();
    mutation_config.set_mutate_pegs(true);

    let mut fitness_config = FitnessConfig::new();

    let mut config = Config::new("data/images/cross_hatch.jpg");
    config.add_thread((0.0, 0.0, 0.0));
    config.set_background_colour((1.0, 1.0, 1.0));
    config.set_mutation_config(mutation_config);
    config.set_fitness_config(fitness_config);

    for c in -100..100 {
        for r in -100..100 {
            config.add_mutable_peg(Peg::new((c as f32 / 100.0, r as f32 / 100.0)));
        }
    }

    config.build();
}
