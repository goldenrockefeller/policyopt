from .map cimport BaseMap, BaseDifferentiableMap

cdef class TargetEntry:
    cdef public object input
    cdef public object target

    cpdef copy(self, copy_obj = ?)

cdef TargetEntry new_TargetEntry()

cdef class BaseFunctionApproximator(BaseMap):
    cpdef batch_update(self, entries)


cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
    cdef object __super_map # BaseDifferentiableMap
    cdef double __learning_rate

    cpdef super_map(self)
    cpdef set_super_map(self, map)

    cpdef double learning_rate(self) except *
    cpdef void set_learning_rate(self, double learning_rate) except *

cdef DifferentiableFunctionApproximator new_DifferentiableFunctionApproximator(
    BaseDifferentiableMap super_map)

cdef void init_DifferentiableFunctionApproximator(
    DifferentiableFunctionApproximator approximator,
    BaseDifferentiableMap super_map
    ) except *