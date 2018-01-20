pub struct MutationConfig {
    /// The probability of inserting an action during `Genome` mutation.
    pub(crate) insert_action_prob: f32,

    /// The probability of removing an action during `Genome` mutation.
    ///
    /// Must be smaller than `new_action_prob` otherwise the structure
    /// won't advance.
    pub(crate) remove_action_prob: f32,

    // TODO(orglofch): Consider adding actions for moving a thread or moving a peg instead of requiring
    // a deletion than an insertion to accomplish the same thing. This would be especially useful for
    // pegs since they could maintain their same general structure assuming we "moved" to a nearby
    // neighbour peg.
    /// The probability of inserting a peg during `Genome` mutation.
    pub(crate) insert_peg_prob: f32,

    /// The probability of removing a peg during `Genome` mutation.
    ///
    /// Mut be smaller than `insert_peg_prob` otherwise the structure
    /// won't advance.
    pub(crate) remove_peg_prob: f32,

    /// The probability of moving a peg during `Genome` mutation.
    pub(crate) move_peg_prob: f32,

    /// Whether the positions of pegs should be mutated in addition to
    /// the thread wrapping.
    ///
    /// TODO(orglofch): Potentially turn of thread mutation as well allowing for mutating a fixed
    /// topology of threads through only moving the pegs around.
    pub(crate) mutate_pegs: bool,
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
