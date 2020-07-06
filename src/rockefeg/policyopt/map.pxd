cdef class BaseMap:
    cpdef copy(self, copy_obj = ?)

    cpdef parameters(self)
    cpdef void set_parameters(self, parameters) except *

    cpdef eval(self, input)

cdef class BaseDifferentiableMap(BaseMap):
    cpdef grad_wrt_parameters(self, input)
    cpdef grad_wrt_input(self, input)