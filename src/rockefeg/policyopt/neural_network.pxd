# distutils: language = c++

from .map cimport BaseDifferentiableMap
from libcpp.vector cimport vector

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator= (const valarray&)
        valarray operator= (const T&)
        T& operator[] (size_t)
        valarray operator* (const valarray&, const valarray&)
        valarray operator* (const T&, const valarray&)
        valarray operator* (const valarray&, const T&)
        valarray operator+ (const valarray&, const valarray&)
        T sum() const

    valarray[double] tanh (const valarray[double]&)

cdef class ReluTanh(BaseDifferentiableMap):
    cdef vector[valarray[double]] linear0
    cdef valarray[double] bias0
    cdef vector[valarray[double]] linear1
    cdef valarray[double] bias1

    cpdef tuple shape(self)

cdef ReluTanh new_ReluTanh(
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims)

cdef void init_ReluTanh(
    ReluTanh neural_network,
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims
    ) except *

cpdef Py_ssize_t n_parameters_for_ReluTanh(ReluTanh neural_network)

cdef class ReluLinear(BaseDifferentiableMap):
    cdef vector[valarray[double]] linear0
    cdef valarray[double] bias0
    cdef vector[valarray[double]] linear1
    cdef valarray[double] bias1
    cdef public double leaky_scale # a hyperparameter, not a regular parameter

    cpdef tuple shape(self)

cdef ReluLinear new_ReluLinear(
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims)

cdef void init_ReluLinear(
    ReluLinear neural_network,
    Py_ssize_t n_in_dims,
    Py_ssize_t n_hidden_neurons,
    Py_ssize_t n_out_dims
    ) except *

cpdef Py_ssize_t n_parameters_for_ReluLinear(ReluLinear neural_network)

