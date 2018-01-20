pub enum LossFn {
    Log,
    Square,
}

pub struct FitnessConfig {
    /// The coefficient of the factor used in punishing large Genomes.
    ///
    /// A values of 0 indicates that there is no complexity punishment.
    pub(crate) complexity_punishment_coefficient: f32,

    /// The fitness threshold at which the generator will terminate.
    pub(crate) fitness_termination_threshold: f32,

    /// The loss function used in calculating the fitness.
    pub(crate) loss_fn: LossFn,
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

    pub fn set_loss_fn(&mut self, loss_fn: LossFn) -> &FitnessConfig {
        self.loss_fn = loss_fn;
        self
    }
}

impl Default for FitnessConfig {
    fn default() -> FitnessConfig {
        FitnessConfig {
            fitness_termination_threshold: 1.0,
            complexity_punishment_coefficient: 0.0,
            loss_fn: LossFn::Square,
        }
    }
}
