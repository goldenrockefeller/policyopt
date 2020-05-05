cimport cython

@cython.warn.undeclared(True)
cdef class BaseSystem:
    cpdef copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef bint is_done_training(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void prep_for_epoch(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef bint is_ready_for_evaluation(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef action(self, observation):
        raise NotImplementedError("Abstract method.")

    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void update_policy(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void receive_score(self, score) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        raise NotImplementedError("Abstract method.")