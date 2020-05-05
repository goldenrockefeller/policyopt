cdef class BaseDomain:
    cpdef void prep_for_generation(self) except *

    cpdef void reset_for_training(self) except *

    cpdef object observations(self)

    cpdef void step(self, object actions) except *

    cpdef object feedback(self)

    cpdef void reset_for_evaluation(self) except *

    cpdef double score(self)

    cpdef void output_final_log(
        self,
        object log_dirname,
        object datetime_str
        ) except *