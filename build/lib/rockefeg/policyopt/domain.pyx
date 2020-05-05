cimport cython

@cython.warn.undeclared(True)
cdef class BaseDomain:
    cpdef void prep_for_generation(self) except *:
        raise NotImplementedError

    cpdef void reset_for_training(self) except *:
        raise NotImplementedError

    cpdef object observations(self):
        raise NotImplementedError

    cpdef void step(self, object actions) except *:
        raise NotImplementedError

    cpdef object feedback(self):
        raise NotImplementedError

    cpdef void reset_for_evaluation(self) except *:
        raise NotImplementedError

    cpdef double score(self):
        raise NotImplementedError

    cpdef void output_final_log(
            self,
            object log_dirname,
            object datetime_str
            ) except *:
        raise NotImplementedError