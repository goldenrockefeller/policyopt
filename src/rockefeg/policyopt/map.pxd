from rockefeg.cyutil.array cimport DoubleArray

cdef class BaseMap:
    cpdef copy(self, copy_obj = ?)

    cpdef parameters(self)
    cpdef void set_parameters(self, parameters) except *
    cpdef Py_ssize_t n_parameters(self) except *

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

cdef class ContinuousCriticMap(BaseMap):
    cdef __super_map

    cpdef super_map(self)
    cpdef void set_super_map(self, super_map) except *

cdef ContinuousCriticMap new_ContinuousCriticMap(BaseMap super_map)
cdef void init_ContinuousCriticMap(
    ContinuousCriticMap map,
    BaseMap super_map
    ) except *

cdef class DifferentiableCriticMap(BaseDifferentiableMap):
    cdef __super_map

    cpdef super_map(self)
    cpdef void set_super_map(self, super_map) except *

cdef DifferentiableCriticMap new_DifferentiableCriticMap(
    BaseDifferentiableMap super_map)

cdef void init_DifferentiableCriticMap(
    DifferentiableCriticMap map,
    BaseDifferentiableMap super_map
    ) except *