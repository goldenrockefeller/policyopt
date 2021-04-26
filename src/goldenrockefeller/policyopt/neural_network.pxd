from .map cimport BaseDifferentiableMap
from rockefeg.cyutil.array cimport DoubleArray
from libcpp.vector cimport vector

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator-() const
        valarray operator= (const valarray&)
        T& operator[] (size_t)
        T sum() const
        T min() const

        valarray operator* (const valarray&, const valarray&)
        valarray operator* (const T&, const valarray&)
        valarray operator* (const valarray&, const T&)
        valarray operator/ (const valarray&, const valarray&)
        valarray operator/ (const T&, const valarray&)
        valarray operator/ (const valarray&, const T&)
        valarray operator+ (const valarray&, const valarray&)
        valarray operator+ (const T&, const valarray&)
        valarray operator+ (const valarray&, const T&)
        valarray operator- (const valarray&, const valarray&)
        valarray operator- (const T&, const valarray&)
        valarray operator- (const valarray&, const T&)

cdef class TanhLayer(BaseDifferentiableMap):
    cpdef TanhLayer copy(self, copy_obj = ?)
    cdef public BaseDifferentiableMap super_map

    cpdef Py_ssize_t n_parameters(self) except *


cdef TanhLayer new_TanhLayer(BaseDifferentiableMap super_map)

cdef void init_TanhLayer(
    TanhLayer neural_network,
    BaseDifferentiableMap super_map
    ) except *



cdef class ReluLinear(BaseDifferentiableMap):
    cdef vector[valarray[double]] linear0
    cdef valarray[double] bias0
    cdef vector[valarray[double]] linear1
    cdef valarray[double] bias1
    cdef public double leaky_scale # a hyperparameter, not a regular parameter
    cdef public bint linear1_is_fixed

    cpdef ReluLinear copy(self, copy_obj = ?)

    cpdef tuple shape(self)

    cpdef Py_ssize_t n_parameters(self) except *



cdef ReluLinear new_ReluLinear(
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims,
    bint linear1_is_fixed = ?)

cdef void init_ReluLinear(
    ReluLinear neural_network,
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims,
    bint linear1_is_fixed = ?
    ) except *


cdef class Rbfn(BaseDifferentiableMap): # Radial Basis Function Network
    cdef vector[valarray[double]] centers #[n_centers][n_in_dim]
    cdef vector[valarray[double]] scalings #[n_centers][n_in_dim]
    cdef vector[valarray[double]] transform #[n_out_dim][n_center]
    cdef public bint scalings_are_fixed
    cdef public bint normalizes_activations

    cpdef Rbfn copy(self, copy_obj = ?)

    cpdef tuple shape(self)

    cpdef Py_ssize_t n_parameters(self) except *

    cpdef DoubleArray activations_eval(self, input)

cdef Rbfn new_Rbfn(
    Py_ssize_t n_in_dims,
    Py_ssize_t n_centers,
    Py_ssize_t n_out_dims)

cdef void init_Rbfn(
    Rbfn neural_network,
    Py_ssize_t n_in_dims,
    Py_ssize_t n_centers,
    Py_ssize_t n_out_dims
    ) except *

cpdef DoubleArray rbfn_pre_norm_activations_eval(Rbfn self, DoubleArray input)

cpdef DoubleArray normalization_for_DoubleArray(DoubleArray arr)

