from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from .libcpp_valarray cimport valarray
from .cpp_experience cimport ExperienceDatum
from .cpp_rbf_network cimport RbfNetwork

cdef extern from "cpp_core/rbf_network_approximator.hpp" namespace "rockefeg::policyopt" nogil:
    cdef cppclass RbfNetworkApproximator:
        double eval_offset
        double info_retention_factor

        RbfNetworkApproximator() except +
        RbfNetworkApproximator(shared_ptr[RbfNetwork]) except +

        valarray[double] eval(const valarray[double]&) except +
        void update(
            const vector[
                ExperienceDatum[
                    valarray[double],
                    valarray[double],
                    double ]]&
        ) except +