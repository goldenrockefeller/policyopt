cdef class BaseDomain:
    cpdef copy(self, copy_obj = ?)

    cpdef void prep_for_epoch(self) except *

    cpdef void reset_for_training(self) except *

    cpdef observation(self)

    cpdef void step(self, action) except *

    cpdef feedback(self)

    cpdef void reset_for_evaluation(self) except *

    cpdef bint episode_is_done(self) except *

    cpdef double score(self) except *

    cpdef void output_final_log(self, log_dirname, datetime_str) except *