cimport cython

@cython.warn.undeclared(True)
cdef class BaseDomain:
    cpdef BaseDomain copy(self, copy_obj = None):
        pass

    cpdef void prep_for_epoch(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void reset_for_training(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef observation(self):
        raise NotImplementedError("Abstract method.")

    cpdef void step(self, action) except *:
        raise NotImplementedError("Abstract method.")

    cpdef feedback(self):
        raise NotImplementedError("Abstract method.")

    cpdef void reset_for_evaluation(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef double score(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef bint episode_is_done(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void output_final_log(
            self,
            log_dirname,
            datetime_str
            ) except *:
        raise NotImplementedError("Abstract method.")