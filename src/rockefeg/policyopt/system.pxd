cdef class BaseSystem:
    cpdef BaseSystem copy(self, copy_obj = ?)

    cpdef bint is_done_training(self) except *

    cpdef void prep_for_epoch(self) except *

    cpdef bint is_ready_for_evaluation(self) except *

    cpdef action(self, observation)

    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *

    cpdef void update_policy(self) except *

    cpdef void receive_score(self, double score) except *

    cpdef void output_final_log(self, log_dirname, datetime_str) except *
    # string log_dirname
    # string datetime_str