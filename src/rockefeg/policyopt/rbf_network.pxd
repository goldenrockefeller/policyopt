from .map cimport BaseMap
from rockefeg.cyutil.array cimport DoubleArray
from libcpp.memory cimport shared_ptr
from .cpp_rbf_network cimport RbfNetwork as CppRbfNetwork
from .libcpp_valarray cimport valarray


cdef class RbfNetwork(BaseMap):
    cdef shared_ptr[CppRbfNetwork] core

    cpdef DoubleArray center_location(self, Py_ssize_t center_id)
    cpdef void set_center_location(
        self,
        Py_ssize_t center_id,
        DoubleArray location
        ) except *
    cpdef double center_location_element(
        self,
        Py_ssize_t center_id,
        Py_ssize_t element_id
        ) except *
    cpdef void set_center_location_element (
        self,
        Py_ssize_t center_id,
        Py_ssize_t element_id,
        double val
        ) except *


    cpdef DoubleArray center_shape(self, Py_ssize_t center_id)
    cpdef void set_center_shape(
        self,
        Py_ssize_t center_id,
        DoubleArray shape
        ) except *
    cpdef double center_shape_element(
        self,
        Py_ssize_t center_id,
        Py_ssize_t element_id
        ) except *
    cpdef void set_center_shape_element (
        self,
        Py_ssize_t center_id,
        Py_ssize_t element_id,
        double val
        ) except *

    cpdef DoubleArray weights(self, Py_ssize_t out_dim_id)
    cpdef void set_weights(
        self,
        Py_ssize_t out_dim_id,
        DoubleArray weights
        ) except *
    cpdef double weights_element(
        self,
        Py_ssize_t out_dim_id,
        Py_ssize_t element_id
        ) except *
    cpdef void set_weights_element (
        self,
        Py_ssize_t out_dim_id,
        Py_ssize_t element_id,
        double val
        ) except *

    cpdef Py_ssize_t n_in_dims(self) except *
    cpdef Py_ssize_t n_centers(self) except *
    cpdef Py_ssize_t n_out_dims(self) except *

    cpdef DoubleArray activations(self, DoubleArray input)

    cpdef DoubleArray grad_wrt_center_locations(
        self,
        DoubleArray input,
        DoubleArray out_grad = ?)

    cpdef DoubleArray flattened_center_locations(self)
    cpdef void set_center_locations_from_valarray(
        self,
        DoubleArray flattened_center_locations
        ) except *

    cpdef DoubleArray parameters(self)

    cpdef DoubleArray eval(self, input)

# TODO cdef new and init RbfNetwork functions for speed




