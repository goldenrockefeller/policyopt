cimport cython

@cython.warn.undeclared(True)
cdef class BaseSystem:
    cpdef bint is_done_training(self) except *:
        raise NotImplementedError

    cpdef void prep_for_generation(self) except *:
        raise NotImplementedError

    cpdef bint is_ready_for_evaluation(self) except *:
        raise NotImplementedError

    cpdef object actions(self, object observations):
        raise NotImplementedError

    #step_wise feedback
    cpdef void receive_feedback(self, object feedback) except *:
        raise NotImplementedError

    cpdef void update_policy(self) except *:
        raise NotImplementedError

    cpdef void receive_score(self, score) except *:
        raise NotImplementedError

    cpdef void output_final_log(
            self,
            object log_dirname,
            object datetime_str
            ) except *:
        raise NotImplementedError