from rockefeg.cyutil.array cimport DoubleArray

cdef class BaseMap:
    cpdef copy(self, copy_obj = ?)

    cpdef parameters(self)
    cpdef void set_parameters(self, parameters) except *

    cpdef eval(self, input)

cdef class BaseDifferentiableMap(BaseMap):
    cpdef jacobian_wrt_parameters(self, input)
    cpdef jacobian_wrt_input(self, input)
    cpdef grad_wrt_parameters(self, input, output_grad = ?)
    cpdef grad_wrt_input(self, input, output_grad = ?)

cpdef list default_jacobian_wrt_parameters(
    BaseDifferentiableMap map,
    DoubleArray input)

cpdef list default_jacobian_wrt_input(
    BaseDifferentiableMap map,
    DoubleArray input)