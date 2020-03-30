cdef class BaseSystem:
    cpdef bint is_done_training(self) except *

    cpdef void prep_for_generation(self) except *

    cpdef bint is_ready_for_evaluation(self) except *

    cpdef object actions(self, object observations)

    #step_wise feedback
    cpdef void receive_feedback(self, object feedback) except *

    cpdef void update_policy(self) except *

    cpdef void receive_score(self, score) except *

    cpdef void output_final_log(
        self,
        object log_dirname,
        object datetime_str
        ) except *