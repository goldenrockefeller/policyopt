# distutils: language = c++

from rockefeg.policyopt.policy cimport BaseDifferentiablePolicy
from libcpp.vector cimport vector

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator= (const valarray&)
        T& operator[] (size_t)
        valarray operator* (const valarray&, const valarray&)
        valarray operator+ (const valarray&, const valarray&)
        T sum() const

    valarray[double] tanh (const valarray[double]&)

cdef class BaseNeuralNetwork:
    cpdef copy(self, copy_obj = ?)

    cpdef eval(self, x)

    cpdef parameters(self)

    cpdef void set_parameters(self, parameters) except *

    cpdef grad_wrt_parameters(self, observation)

    cpdef grad_wrt_observation(self, observation)

cdef class NeuroPolicy(BaseDifferentiablePolicy):
    cdef object __neural_network

    cpdef neural_network(self)
    cpdef void set_neural_network(self, neural_network) except *

cdef NeuroPolicy new_NeuroPolicy(BaseNeuralNetwork neural_network)
cdef void init_NeuroPolicy(
    NeuroPolicy policy,
    BaseNeuralNetwork neural_network
    ) except *

cdef class ReluTanh(BaseNeuralNetwork):
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

