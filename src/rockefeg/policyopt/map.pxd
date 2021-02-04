from rockefeg.cyutil.array cimport DoubleArray

cdef class BaseMap:
    cpdef BaseMap copy(self, copy_obj = ?)

    cpdef parameters(self)
    cpdef void set_parameters(self, parameters) except *
    cpdef Py_ssize_t n_parameters(self) except *

    cpdef eval(self, input)

cdef class BaseDifferentiableMap(BaseMap):
    cpdef BaseDifferentiableMap copy(self, copy_obj = ?)

    cpdef DoubleArray eval(self, input)
    cpdef list jacobian_wrt_parameters(self, input)
    cpdef list jacobian_wrt_input(self, input)
    cpdef DoubleArray grad_wrt_parameters(self, input, output_grad = ?)
    cpdef DoubleArray grad_wrt_input(self, input, output_grad = ?)

cpdef list default_jacobian_wrt_parameters(
    BaseDifferentiableMap map,
    DoubleArray input)

cpdef list default_jacobian_wrt_input(
    BaseDifferentiableMap map,
    DoubleArray input)

cdef class ContinuousCriticMap(BaseMap):
    cdef BaseMap __super_map

    cpdef ContinuousCriticMap copy(self, copy_obj = ?)

    cpdef BaseMap super_map(self)
    cpdef void set_super_map(self, BaseMap super_map) except *

cdef ContinuousCriticMap new_ContinuousCriticMap(BaseMap super_map)
cdef void init_ContinuousCriticMap(
    ContinuousCriticMap map,
    BaseMap super_map
    ) except *

cdef class DifferentiableCriticMap(BaseDifferentiableMap):
    cdef BaseDifferentiableMap __super_map

    cpdef DifferentiableCriticMap copy(self, copy_obj = ?)

    cpdef BaseMap super_map(self)
    cpdef void set_super_map(self, BaseDifferentiableMap super_map) except *

cdef DifferentiableCriticMap new_DifferentiableCriticMap(
    BaseDifferentiableMap super_map)

cdef void init_DifferentiableCriticMap(
    DifferentiableCriticMap map,
    BaseDifferentiableMap super_map
    ) except *