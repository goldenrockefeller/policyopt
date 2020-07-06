cimport cython

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseMap:

    cpdef copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef void set_parameters(self, parameters) except *:
        raise NotImplementedError("Abstract method.")

    cpdef eval(self, input):
        raise NotImplementedError("Abstract method.")

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseDifferentiableMap(BaseMap):

    cpdef grad_wrt_parameters(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_input(self, input):
        raise NotImplementedError("Abstract method.")
