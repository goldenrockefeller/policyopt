from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from .libcpp_valarray cimport valarray

cdef extern from "cpp_core/rbf_network.hpp" namespace "rockefeg::policyopt" nogil:
    cdef cppclass RbfNetwork:
        cppclass Center:
            valarray[double] location
            valarray[double] shape
        vector[Center] centers
        vector[valarray[double]] linear

        RbfNetwork(size_t, size_t, size_t) except +
        unique_ptr[RbfNetwork] copy() except +
        size_t n_in_dims() except +
        size_t n_centers() except +
        size_t n_out_dims() except +
        size_t n_parameters() except +
        valarray[double] parameters() except +
        void set_parameters(valarray[double]) except +

        valarray[double] activations(const valarray[double]&) except +

        valarray[double] eval(const valarray[double]&) except +

        valarray[double] grad_wrt_center_locations(
            const valarray[double]&,
            const valarray[double]&,
            ) except +

        valarray[double] flattened_center_locations() except +
        void set_center_locations_from_valarray(const valarray[double]&) except +