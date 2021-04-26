cimport cython
from rockefeg.cyutil.array cimport new_DoubleArray
from rockefeg.cyutil.typed_list cimport is_sub_full_type
# import numpy as np
# from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
# from numpy.random cimport bitgen_t
# from numpy.random import PCG64
# from numpy.random.c_distributions cimport random_standard_normal, random_standard_cauchy
# from libc.math cimport sqrt, log, exp
#
# # Random numpy c-api from https://numpy.org/doc/stable/reference/random/extending.html
# cdef const char *capsule_name = "BitGenerator"
# cdef bitgen_t *rng
# rnd_bitgen = PCG64()
# capsule = rnd_bitgen.capsule
# # Optional check that the capsule if from a BitGenerator
# if not PyCapsule_IsValid(capsule, capsule_name):
#     raise ValueError("Invalid pointer to anon_func_state")
# # Cast the pointer
# rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)


from typing import Sequence

import cython


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TargetEntry:
    cpdef TargetEntry copy(self, copy_obj = None):
        cdef TargetEntry new_entry

        if copy_obj is None:
            new_entry = TargetEntry.__new__(TargetEntry)
        else:
            new_entry = copy_obj

        new_entry.input = self.input
        new_entry.target = self.target

        return new_entry

@cython.warn.undeclared(True)
cdef TargetEntry new_TargetEntry():
    cdef TargetEntry entry

    entry = TargetEntry.__new__(TargetEntry)

    return entry


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseFunctionApproximator(BaseMap):

    cpdef BaseFunctionApproximator copy(self, copy_obj = None):
        pass

    @cython.locals(entries = list)
    cpdef void batch_update(self, entries: Sequence[TargetEntry]) except *:
        raise NotImplementedError("Abstract method.")

#
# @cython.warn.undeclared(True)
# @cython.auto_pickle(True)
# cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
#     def __init__(self, BaseDifferentiableMap super_map):
#         init_DifferentiableFunctionApproximator(self, super_map)
#
#     cpdef DifferentiableFunctionApproximator copy(self, copy_obj = None):
#         cdef DifferentiableFunctionApproximator new_approximator
#         if copy_obj is None:
#             new_approximator = (
#                 DifferentiableFunctionApproximator.__new__(
#                     DifferentiableFunctionApproximator))
#         else:
#             new_approximator = copy_obj
#
#         new_approximator.__super_map = self.__super_map.copy()
#         new_approximator.__learning_rate = self.__learning_rate
#
#         return new_approximator
#
#     cpdef parameters(self):
#         return self.super_map().parameters()
#
#     cpdef void set_parameters(self, parameters) except *:
#         self.super_map().set_parameters(parameters)
#
#     cpdef Py_ssize_t n_parameters(self) except *:
#         return self.super_map().n_parameters()
#
#     cpdef eval(self, input):
#         return self.super_map().eval(input)
#
#     @cython.locals(entries = list)
#     cpdef void batch_update(self, entries: Sequence[TargetEntry]) except *:
#         cdef object input
#         cdef DoubleArray parameters
#         cdef DoubleArray error
#         cdef DoubleArray error_grad_wrt_parameters
#         cdef DoubleArray sum_error_grad_wrt_parameters
#         cdef BaseDifferentiableMap map
#         cdef TargetEntry entry
#         cdef Py_ssize_t n_entries
#         cdef Py_ssize_t n_entry_parameters
#         cdef Py_ssize_t n_parameters
#         cdef Py_ssize_t output_id
#         cdef Py_ssize_t parameter_id
#         cdef object entries_item_type
#         cdef double learning_rate
#
#         # entries_item_type = entries.item_type()
#         #
#         # if not is_sub_full_type(entries_item_type, TargetEntry):
#         #     raise (
#         #         TypeError(
#         #             "The entries list's item type "
#         #             "(entries.item_type() = {entries_item_type}) "
#         #             "must be a subtype of TargetEntry."
#         #             .format(**locals())))
#
#         map = self.super_map()
#         # print(type(map.parameters()))
#         parameters = map.parameters()
#         parameters = parameters.copy()
#         n_parameters = map.n_parameters()
#         sum_error_grad_wrt_parameters = new_DoubleArray(n_parameters)
#         sum_error_grad_wrt_parameters.set_all_to(0.)
#         n_entries = len(entries)
#         learning_rate = self.learning_rate()
#
#
#         for entry in entries:
#
#             error = error_eval(map, entry)
#
#             # Add gradient for this entry to the cumulative gradient.
#             input = entry.input
#             error_grad_wrt_parameters = map.grad_wrt_parameters(input, error)
#             n_entry_parameters = len(error_grad_wrt_parameters)
#             #
#             if n_parameters != n_entry_parameters:
#                 raise (
#                     ValueError(
#                         "The number of the map's parameters can not change "
#                         "during a batch update. The starting number of "
#                         "parameters (len(self.super_map().parameters() = "
#                         "{n_parameters}) does not match the current number "
#                         "of parameters (len(map.super_map."
#                         "grad_wrt_parameters(input, error) = "
#                         "{n_entry_parameters})."
#                         .format(**locals()) ))
#             #
#             for parameter_id in range(n_parameters):
#                 sum_error_grad_wrt_parameters.view[parameter_id] += (
#                     error_grad_wrt_parameters.view[parameter_id]
#                     * learning_rate
#                     / n_entries)
#
#         # Add gradient to parameters
#         for parameter_id in range(n_parameters):
#             parameters.view[parameter_id] += (
#                 sum_error_grad_wrt_parameters.view[parameter_id])
#
#         # Update the maps parameters.
#         map.set_parameters(parameters)
#
#     cpdef BaseDifferentiableMap super_map(self):
#         return self.__super_map
#
#     cpdef void set_super_map(self, BaseDifferentiableMap map) except *:
#         self.__super_map = map
#
#     cpdef double learning_rate(self) except *:
#         return self.__learning_rate
#
#     cpdef void set_learning_rate(self, double learning_rate) except *:
#         self.__learning_rate = learning_rate
#
#
# @cython.warn.undeclared(True)
# cdef DifferentiableFunctionApproximator new_DifferentiableFunctionApproximator(
#         BaseDifferentiableMap super_map):
#
#     cdef DifferentiableFunctionApproximator approximator
#
#     approximator = (
#         DifferentiableFunctionApproximator.__new__(
#             DifferentiableFunctionApproximator))
#
#     init_DifferentiableFunctionApproximator(approximator, super_map)
#
#     return approximator
#
#
# @cython.warn.undeclared(True)
# cdef void init_DifferentiableFunctionApproximator(
#     DifferentiableFunctionApproximator approximator,
#     BaseDifferentiableMap super_map
#     ) except *:
#
#     if approximator is None:
#         raise TypeError("The approximator (approximator) cannot be None.")
#
#     if super_map is None:
#         raise TypeError("The map (super_map) cannot be None.")
#
#     approximator.__super_map = super_map
#     approximator.__learning_rate = 1e-4
#
# cdef DoubleArray error_eval(
#         BaseDifferentiableMap map,
#         TargetEntry entry):
#     cdef object input
#     cdef DoubleArray output
#     cdef DoubleArray error
#     cdef DoubleArray target
#     cdef Py_ssize_t output_size
#     cdef Py_ssize_t target_size
#
#     input = entry.input
#     output = map.eval(input)
#
#     # Calculate the error for this entry.
#     target = entry.target
#     output_size = len(output)
#     target_size = len(target)
#     #
#     if output_size != target_size:
#         raise (
#             ValueError(
#                 "The output size (len(self.super_map()"
#                 ".eval(entries.input[?])) "
#                 "= {output_size}) "
#                 "must be equal to the target size "
#                 "(len(entries.target[?]) = {target_size})."
#                 .format(**locals()) ))
#     #
#     error = new_DoubleArray(output_size)
#     #
#     for output_id in range(output_size):
#         error.view[output_id] = (
#             target.view[output_id]
#             - output.view[output_id])
#
#     return error
#




