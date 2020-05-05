cimport cython

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BasePolicy:

    cpdef copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef void set_parameters(self, parameters) except *:
        raise NotImplementedError("Abstract method.")

    cpdef action(self, observation):
        raise NotImplementedError("Abstract method.")

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseDifferentiablePolicy(BasePolicy):

    cpdef grad_wrt_parameters(self, observation):
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_observation(self, observation):
        raise NotImplementedError("Abstract method.")
