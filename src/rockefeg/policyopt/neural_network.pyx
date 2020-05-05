cimport cython
from libc cimport math as cmath
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray


import numpy as np

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseNeuralNetwork:
    cpdef copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef eval(self, x):
        raise NotImplementedError("Abstract method.")

    cpdef parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef void set_parameters(self, parameters) except *:
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_parameters(self, observation):
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_observation(self, observation):
        raise NotImplementedError("Abstract method.")

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class NeuroPolicy(BaseDifferentiablePolicy):
    def __init__(self, neural_network):
        init_NeuroPolicy(self, neural_network)

    cpdef copy(self, copy_obj = None):
        cdef NeuroPolicy new_policy

        if copy_obj is None:
            new_policy = NeuroPolicy.__new__(NeuroPolicy)
        else:
            new_policy = copy_obj

        new_policy.__neural_network = (
            (<BaseNeuralNetwork?>self.__neural_network).copy())

        return new_policy

    cpdef parameters(self):
        return (<BaseNeuralNetwork?>self.neural_network()).parameters()

    cpdef void set_parameters(self, parameters) except *:
        (<BaseNeuralNetwork?>self.neural_network()).set_parameters(parameters)

    cpdef action(self, observation):
        return (
            (<BaseNeuralNetwork?>self.neural_network()).eval(observation))

    cpdef grad_wrt_parameters(self, observation):
        return (
            (<BaseNeuralNetwork?>self.neural_network())
            .grad_wrt_parameters(observation))

    cpdef grad_wrt_observation(self, observation):
        return (
            (<BaseNeuralNetwork?>self.neural_network())
            .grad_wrt_observation(observation))

    cpdef neural_network(self):
        return self.__neural_network

    cpdef void set_neural_network(self, neural_network) except *:
        self.__neural_network = <BaseNeuralNetwork?>neural_network

@cython.warn.undeclared(True)
cdef NeuroPolicy new_NeuroPolicy(BaseNeuralNetwork neural_network):
    cdef NeuroPolicy policy

    policy = NeuroPolicy.__new__(NeuroPolicy)
    init_NeuroPolicy(policy, neural_network)

    return policy

@cython.warn.undeclared(True)
cdef void init_NeuroPolicy(
        NeuroPolicy policy,
        BaseNeuralNetwork neural_network
        ) except *:
    if policy is None:
        raise TypeError("The policy (policy) cannot be None.")

    if neural_network is None:
        raise TypeError("The neural network (neural_network) cannot be None.")

    policy.__neural_network = neural_network

cimport cython


@cython.warn.undeclared(True)
cdef class ReluTanh:
    def __init__(self, n_in_dims, n_hidden_neurons, n_out_dims):
        init_ReluTanh(self, n_in_dims, n_hidden_neurons, n_out_dims)

    def __reduce__(self):
        return unpickle_ReluTanh, (self.shape(), self.parameters())

    cpdef copy(self, copy_obj = None):
        cdef ReluTanh new_neural_network

        if copy_obj is None:
            new_neural_network = ReluTanh.__new__(ReluTanh)
        else:
            new_neural_network = copy_obj

        new_neural_network.linear0 = self.linear0
        new_neural_network.linear1 = self.linear1
        new_neural_network.bias0 = self.bias0
        new_neural_network.bias1 = self.bias1

        return new_neural_network

    cpdef tuple shape(self):
        return (self.linear0[0].size(), self.bias0.size(), self.bias1.size())


    cpdef parameters(self):
        cdef DoubleArray parameters
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        parameters = new_DoubleArray(n_parameters_for_ReluTanh(self))
        parameter_id = 0

        for i in range(self.linear0.size()):
            for j in range(self.linear0[0].size()):
                parameters.view[parameter_id] = self.linear0[i][j]
                parameter_id += 1
        #
        for i in range(self.bias0.size()):
            parameters.view[parameter_id]  = self.bias0[i]
            parameter_id += 1

        for i in range(self.linear1.size()):
            for j in range(self.linear1[0].size()):
                parameters.view[parameter_id] = self.linear1[i][j]
                parameter_id += 1
        #
        for i in range(self.bias1.size()):
            parameters.view[parameter_id]  = self.bias1[i]
            parameter_id += 1

        return parameters

    cpdef void set_parameters(self, parameters) except *:
        cdef DoubleArray cy_parameters = <DoubleArray?> parameters
        cdef Py_ssize_t n_setting_parameters
        cdef Py_ssize_t n_self_parameters
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        n_setting_parameters = len(cy_parameters)
        n_self_parameters = n_parameters_for_ReluTanh(self)

        if n_setting_parameters != n_self_parameters:
            raise (
                ValueError(
                    "The number of setting parameters "
                    "(len(parameters)={n_setting_parameters}) "
                    "must be equal to the number of neural network parameters "
                    "(len(self.parameters())={n_self_parameters})."
                    .format(**locals())))

        parameter_id = 0

        for i in range(self.linear0.size()):
            for j in range(self.linear0[0].size()):
                self.linear0[i][j] = cy_parameters.view[parameter_id]
                parameter_id += 1
        #
        for i in range(self.bias0.size()):
            self.bias0[i] = cy_parameters.view[parameter_id]
            parameter_id += 1

        for i in range(self.linear1.size()):
            for j in range(self.linear1[0].size()):
                self.linear1[i][j] = cy_parameters.view[parameter_id]
                parameter_id += 1
        #
        for i in range(self.bias1.size()):
            self.bias1[i] = cy_parameters.view[parameter_id]
            parameter_id += 1

    cpdef eval(self, x):
        cdef DoubleArray cy_x = <DoubleArray?> x
        cdef DoubleArray eval
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t x_size
        cdef valarray[double] x_valarray
        cdef valarray[double] linear0_res
        cdef valarray[double] bias0_res
        cdef valarray[double] relu_res
        cdef valarray[double] linear1_res
        cdef valarray[double] bias1_res
        cdef valarray[double] tanh_res
        cdef size_t i
        cdef Py_ssize_t x_id
        cdef Py_ssize_t eval_id

        x_size = len(cy_x)
        self_n_in_dims = self.linear0[0].size()

        if x_size != self_n_in_dims:
            raise (
                ValueError(
                    "The input size (len(x)={x_size}) "
                    "must be equal to the neural network's number of "
                    "input dimensions (= {self_n_in_dims})."
                    .format(**locals())))

        x_valarray.resize(x_size)
        for x_id in range(x_size):
            x_valarray[x_id] = cy_x.view[<Py_ssize_t>x_id]

        linear0_res.resize(self.linear0.size())
        for i in range(linear0_res.size()):
            linear0_res[i] = (self.linear0[i]*x_valarray).sum()

        bias0_res = linear0_res + self.bias0

        relu_res = bias0_res
        for i in range(relu_res.size()):
            relu_res[i] = relu_res[i] * (relu_res[i] > 0.)

        linear1_res.resize(self.linear1.size())
        for i in range(linear1_res.size()):
            linear1_res[i] = (self.linear1[i]*relu_res).sum()

        bias1_res = linear1_res + self.bias1
        tanh_res = tanh(bias1_res)
        eval = new_DoubleArray(bias1_res.size())

        for eval_id in range(len(eval)):
            eval.view[eval_id] = tanh_res[<Py_ssize_t>eval_id]

        return eval


    cpdef grad_wrt_parameters(self, observation):
        # TODO
        raise NotImplementedError("Not implmented yet")

    cpdef grad_wrt_observation(self, observation):
        # TODO
        raise NotImplementedError("Not implmented yet")



@cython.warn.undeclared(True)
cdef ReluTanh new_ReluTanh(
        Py_ssize_t n_in_dims,
        Py_ssize_t n_hidden_neurons,
        Py_ssize_t n_out_dims):
    cdef ReluTanh neural_network

    neural_network = ReluTanh.__new__(ReluTanh)
    init_ReluTanh(neural_network, n_in_dims, n_hidden_neurons, n_out_dims)

    return neural_network

@cython.warn.undeclared(True)
cdef void init_ReluTanh(
        ReluTanh neural_network,
        Py_ssize_t n_in_dims,
        Py_ssize_t n_hidden_neurons,
        Py_ssize_t n_out_dims
        ) except *:
    cdef double in_stdev
    cdef double hidden_stdev
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double[:, ::1] np_linear0
    cdef double[:, ::1] np_linear1
    cdef double[::1] np_bias0
    cdef double[::1] np_bias1

    if neural_network is None:
        raise TypeError("The neural network (neural_network) cannot be None.")

    if n_in_dims <= 0:
        raise (
            ValueError(
                "The number of input dimensions (n_in_dims = {n_in_dims}) "
                "must be positive."
                .format(**locals())))

    if n_hidden_neurons <= 0:
        raise (
            ValueError(
                "The number of hidden neurons "
                "(n_hidden_neurons = {n_hidden_neurons}) "
                "must be positive."
                .format(**locals())))

    if n_out_dims <= 0:
        raise (
            ValueError(
                "The number of output dimensions (n_out_dims = {n_out_dims}) "
                "must be positive."
                .format(**locals())))

    # Get initialization matrices using numpy.
    in_stdev =  cmath.sqrt(3./ n_in_dims)
    np_linear0 = (
        np.random.uniform(
            -in_stdev,
            in_stdev,
            (n_hidden_neurons, n_in_dims) ))
    np_bias0 = (
        np.random.uniform(
            -in_stdev,
            in_stdev,
            n_hidden_neurons))
    #
    hidden_stdev = cmath.sqrt(3./ n_hidden_neurons)
    np_linear1 =  (
        np.random.uniform(
            -hidden_stdev,
            hidden_stdev,
            (n_out_dims, n_hidden_neurons) ))
    np_bias1 = (
        np.random.uniform(
            -hidden_stdev,
            hidden_stdev,
            n_out_dims))

    # Copy initialization matrices data into ReluTanh fields.
    neural_network.linear0.resize(np_linear0.shape[0])
    for i in range(np_linear0.shape[0]):
        neural_network.linear0[i].resize(np_linear0.shape[1])
        for j in range(np_linear0.shape[1]):
            neural_network.linear0[i][j] = np_linear0[i, j]
    #
    neural_network.linear1.resize(np_linear1.shape[0])
    for i in range(np_linear1.shape[0]):
        neural_network.linear1[i].resize(np_linear1.shape[1])
        for j in range(np_linear1.shape[1]):
            neural_network.linear1[i][j] = np_linear1[i, j]
    #
    neural_network.bias0.resize(np_bias0.shape[0])
    for i in range(np_bias0.shape[0]):
        neural_network.bias0[i] = np_bias0[i]
    #
    neural_network.bias1.resize(np_bias1.shape[0])
    for i in range(np_bias1.shape[0]):
        neural_network.bias1[i] = np_bias1[i]

@cython.warn.undeclared(True)
cpdef Py_ssize_t n_parameters_for_ReluTanh(ReluTanh neural_network):
    cdef Py_ssize_t n_parameters

    n_parameters = 0
    n_parameters += (
        neural_network.linear0.size()
        * neural_network.linear0[0].size())
    n_parameters += (
        neural_network.linear1.size()
        * neural_network.linear1[0].size())
    n_parameters += neural_network.bias0.size()
    n_parameters += neural_network.bias1.size()

    return n_parameters

def unpickle_ReluTanh(shape, parameters):
    neural_network = ReluTanh(shape[0], shape[1], shape[2])
    neural_network.set_parameters(parameters)

    return neural_network
