from .map cimport BaseMap, BaseDifferentiableMap
from goldenrockefeller.cyutil.array cimport DoubleArray

import cython
from typing import Sequence

cdef class TargetEntry:
    cdef public object input
    cdef public object target

    cpdef TargetEntry copy(self, copy_obj = ?)

cdef TargetEntry new_TargetEntry()

cdef class BaseFunctionApproximator(BaseMap):
    cpdef BaseFunctionApproximator copy(self, copy_obj = ?)

    @cython.locals(entries = list)
    cpdef void batch_update(self, entries: Sequence[TargetEntry]) except *
#

cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
    cdef BaseDifferentiableMap __super_map
    cdef double __learning_rate

    cpdef DifferentiableFunctionApproximator copy(self, copy_obj = ?)

    cpdef BaseDifferentiableMap super_map(self)
    cpdef void set_super_map(self, BaseDifferentiableMap map) except *

    cpdef double learning_rate(self) except *
    cpdef void set_learning_rate(self, double learning_rate) except *

cdef DifferentiableFunctionApproximator new_DifferentiableFunctionApproximator(
    BaseDifferentiableMap super_map)

cdef void init_DifferentiableFunctionApproximator(
    DifferentiableFunctionApproximator approximator,
    BaseDifferentiableMap super_map
    ) except *
#
# cdef class RobustFunctionApproximator(DifferentiableFunctionApproximator):
#     cdef double __line_search_shrink_rate
#     cdef double __line_search_growth_rate
#     cdef double __line_search_step_size
#     # "nes" means Natural Evolutionary Strategy
#     # "nes" factors should be positive
#     cdef double __nes_size_growth_rate
#     cdef double __nes_direction_growth_rate
#     cdef double __nes_size_shrink_rate
#     cdef double __nes_direction_shrink_rate
#     cdef DoubleArray __nes_factors
#
#     cpdef double line_search_shrink_rate(self) except *
#     cpdef void set_line_search_shrink_rate(
#         self,
#         double line_search_shrink_rate
#         ) except *
#
#     cpdef double line_search_growth_rate(self) except *
#     cpdef void set_line_search_growth_rate(
#         self,
#         double line_search_growth_rate
#         ) except *
#
#     cpdef double line_search_step_size(self) except *
#     cpdef void set_line_search_step_size(
#         self,
#         double line_search_step_size
#         ) except *
#
#     cpdef double nes_size_growth_rate(self) except *
#     cpdef void set_nes_size_growth_rate(
#         self,
#         double nes_size_growth_rate
#         ) except *
#
#     cpdef double nes_direction_growth_rate(self) except *
#     cpdef void set_nes_direction_growth_rate(
#         self,
#         double nes_direction_growth_rate
#         ) except *
#
#     cpdef double nes_size_shrink_rate(self) except *
#     cpdef void set_nes_size_shrink_rate(
#         self,
#         double nes_size_shrink_rate
#         ) except *
#
#     cpdef double nes_direction_shrink_rate(self) except *
#     cpdef void set_nes_direction_shrink_rate(
#         self,
#         double nes_direction_shrink_rate
#         ) except *
#
#     cpdef double nes_factor(self, Py_ssize_t id) except *
#     cpdef DoubleArray nes_factors(self)
#
#     cpdef void set_nes_factor(self, Py_ssize_t id, double val) except *
#     cpdef void set_nes_factors(self, DoubleArray factors) except *
#
# cdef RobustFunctionApproximator new_RobustFunctionApproximator(
#     BaseDifferentiableMap super_map)
#
# cdef void init_RobustFunctionApproximator(
#     RobustFunctionApproximator approximator,
#     BaseDifferentiableMap super_map
#     ) except *