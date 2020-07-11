from .map cimport BaseMap, BaseDifferentiableMap

cdef class TargetEntry:
    cdef public object input
    cdef public object target

cdef TargetEntry new_TargetEntry(input, target)
cdef void init_TargetEntry(TargetEntry entry, input, target) except *

cdef class BaseFunctionApproximator(BaseMap):
    cpdef batch_update(self, list entries)


cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
    cdef object __super_map # BaseDifferentiableMap
    cdef double __learning_rate

    cpdef super_map(self)
    cpdef _set_super_map(self, map)

    cpdef double learning_rate(self) except *
    cpdef void set_learning_rate(self, double learning_rate) except *

cdef DifferentiableFunctionApproximator new_DifferentiableFunctionApproximator(
    BaseDifferentiableMap super_map)

cdef void init_DifferentiableFunctionApproximator(
    DifferentiableFunctionApproximator xxx
    BaseDifferentiableMap super_map
    ) except *