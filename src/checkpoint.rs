use config::Config;
use genome::Genome;

pub struct CheckpointConfig {
    /// The function to invoke on every checkpoint.
    ///
    /// Checkpointing functions are free to change the configuration based on the
    /// state of the configuration or however else they can think of.
    pub(crate) checkpoint_fn: fn(&mut Config, &Genome),

    /// The number of iterations / generations between checkpointing function invocations.
    pub(crate) checkpoint_rate: u32,
}

impl CheckpointConfig {
    pub fn new(checkpoint_fn: fn(&mut Config, &Genome), checkpoint_rate: u32) -> CheckpointConfig {
        CheckpointConfig {
            checkpoint_fn: checkpoint_fn,
            checkpoint_rate: checkpoint_rate,
        }
    }
}
