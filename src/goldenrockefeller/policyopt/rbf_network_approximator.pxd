import cython
from typing import Sequence
from libcpp.memory cimport shared_ptr
from rockefeg.cyutil.array cimport DoubleArray
from .cpp_rbf_network_approximator cimport RbfNetworkApproximator as CppRbfNetworkApproximator
from .function_approximation cimport BaseFunctionApproximator, TargetEntry
from .rbf_network cimport RbfNetwork


cdef class RbfNetworkApproximator(BaseFunctionApproximator):
    cdef shared_ptr[CppRbfNetworkApproximator] core
    cdef public RbfNetwork network

    cpdef double eval_offset(self) except *
    cpdef void set_eval_offset(self, double eval_offset) except *

    cpdef double info_retention_factor(self) except *
    cpdef void set_info_retention_factor(self, double info_retention_factor) except *

    cpdef DoubleArray eval(self, input)

    @cython.locals(trajectory = list)
    cpdef void update(self, trajectory: Sequence[ExperienceDatum]) except *

    @cython.locals(entries = list)
    cpdef void batch_update(self, entries: Sequence[TargetEntry]) except *
