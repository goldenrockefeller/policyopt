cimport cython
from libc cimport math as cmath
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from .map cimport default_jacobian_wrt_parameters, default_jacobian_wrt_input

import numpy as np

@cython.warn.undeclared(True)
cdef class ReluTanh(BaseDifferentiableMap):
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
            for j in range(self.linear0[i].size()):
                parameters.view[parameter_id] = self.linear0[i][j]
                parameter_id += 1
        #
        for i in range(self.bias0.size()):
            parameters.view[parameter_id]  = self.bias0[i]
            parameter_id += 1

        for i in range(self.linear1.size()):
            for j in range(self.linear1[i].size()):
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

    cpdef eval(self, input):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray eval
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t input_size
        cdef valarray[double] input_valarray
        cdef valarray[double] linear0_res
        cdef valarray[double] bias0_res
        cdef valarray[double] relu_res
        cdef valarray[double] linear1_res
        cdef valarray[double] bias1_res
        cdef valarray[double] tanh_res
        cdef size_t i
        cdef Py_ssize_t input_id
        cdef Py_ssize_t eval_id

        input_size = len(cy_input)
        self_n_in_dims = self.linear0[0].size()

        if input_size != self_n_in_dims:
            raise (
                ValueError(
                    "The input size (len(input)={input_size}) "
                    "must be equal to the neural network's number of "
                    "input dimensions (= {self_n_in_dims})."
                    .format(**locals())))

        input_valarray.resize(input_size)
        for input_id in range(input_size):
            input_valarray[<size_t>input_id] = cy_input.view[input_id]

        linear0_res.resize(self.linear0.size())
        for i in range(linear0_res.size()):
            linear0_res[i] = (self.linear0[i]*input_valarray).sum()

        bias0_res = linear0_res + self.bias0

        relu_res.resize(bias0_res.size())
        for i in range(relu_res.size()):
            relu_res[i] = bias0_res[i] * (bias0_res[i] > 0.)

        linear1_res.resize(self.linear1.size())
        for i in range(linear1_res.size()):
            linear1_res[i] = (self.linear1[i]*relu_res).sum()

        bias1_res = linear1_res + self.bias1
        tanh_res = tanh(bias1_res)
        eval = new_DoubleArray(bias1_res.size())

        for eval_id in range(len(eval)):
            eval.view[eval_id] = tanh_res[<size_t>eval_id]

        return eval

    cpdef jacobian_wrt_parameters(self, input):
        return default_jacobian_wrt_parameters(self, input)

    cpdef jacobian_wrt_input(self, input):
        return default_jacobian_wrt_input(self, input)

    cpdef grad_wrt_parameters(self, input, output_grad = None):
        # TODO
        raise NotImplementedError("Not implmented yet")

    cpdef grad_wrt_input(self, input, output_grad = None):
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


@cython.warn.undeclared(True)
cdef class ReluLinear(BaseDifferentiableMap):
    def __init__(self, n_in_dims, n_hidden_neurons, n_out_dims):
        init_ReluLinear(
            self,
            n_in_dims,
            n_hidden_neurons,
            n_out_dims)

    def __reduce__(self):
        return (
            unpickle_ReluLinear,
            (self.shape(), self.parameters(), self.leaky_scale))

    cpdef copy(self, copy_obj = None):
        cdef ReluLinear new_neural_network

        if copy_obj is None:
            new_neural_network = ReluLinear.__new__(ReluLinear)
        else:
            new_neural_network = copy_obj

        new_neural_network.linear0 = self.linear0
        new_neural_network.linear1 = self.linear1
        new_neural_network.bias0 = self.bias0
        new_neural_network.bias1 = self.bias1
        new_neural_network.leaky_scale = self.leaky_scale

        return new_neural_network

    cpdef tuple shape(self):
        return (self.linear0[0].size(), self.bias0.size(), self.bias1.size())


    cpdef parameters(self):
        cdef DoubleArray parameters
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        parameters = new_DoubleArray(n_parameters_for_ReluLinear(self))
        parameter_id = 0

        for i in range(self.linear0.size()):
            for j in range(self.linear0[i].size()):
                parameters.view[parameter_id] = self.linear0[i][j]
                parameter_id += 1
        #
        for i in range(self.bias0.size()):
            parameters.view[parameter_id]  = self.bias0[i]
            parameter_id += 1

        for i in range(self.linear1.size()):
            for j in range(self.linear1[i].size()):
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
        n_self_parameters = n_parameters_for_ReluLinear(self)

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

    cpdef eval(self, input):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray eval
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t input_size
        cdef valarray[double] input_valarray
        cdef valarray[double] linear0_res
        cdef valarray[double] bias0_res
        cdef valarray[double] relu_res
        cdef valarray[double] linear1_res
        cdef valarray[double] bias1_res
        cdef size_t i
        cdef Py_ssize_t input_id
        cdef Py_ssize_t eval_id

        input_size = len(cy_input)
        self_n_in_dims = self.linear0[0].size()

        if input_size != self_n_in_dims:
            raise (
                ValueError(
                    "The input size (len(input)={input_size}) "
                    "must be equal to the neural network's number of "
                    "input dimensions (self.shape()[0] = {self_n_in_dims})."
                    .format(**locals())))

        input_valarray.resize(input_size)
        for input_id in range(input_size):
            input_valarray[<size_t>input_id] = cy_input.view[input_id]

        linear0_res.resize(self.linear0.size())
        for i in range(linear0_res.size()):
            linear0_res[i] = (self.linear0[i]*input_valarray).sum()

        bias0_res = linear0_res + self.bias0

        relu_res.resize(bias0_res.size())
        for i in range(relu_res.size()):
            relu_res[i] = (
                bias0_res[i] * (bias0_res[i] > 0.)
                + bias0_res[i] * (bias0_res[i] <= 0.) * self.leaky_scale )

        linear1_res.resize(self.linear1.size())
        for i in range(linear1_res.size()):
            linear1_res[i] = (self.linear1[i]*relu_res).sum()

        bias1_res = linear1_res + self.bias1
        eval = new_DoubleArray(bias1_res.size())

        for eval_id in range(len(eval)):
            eval.view[eval_id] = bias1_res[<size_t>eval_id]

        return eval

    cpdef jacobian_wrt_parameters(self, input):
        return default_jacobian_wrt_parameters(self, input)

    cpdef jacobian_wrt_input(self, input):
        return default_jacobian_wrt_input(self, input)

    cpdef grad_wrt_parameters(self, input, output_grad = None):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray cy_output_grad
        cdef DoubleArray grad_wrt_parameters
        cdef vector[valarray[double]] grad_wrt_linear0
        cdef valarray[double] grad_wrt_bias0
        cdef vector[valarray[double]] grad_wrt_linear1
        cdef valarray[double] grad_wrt_bias1
        cdef valarray[double] linear0_res
        cdef valarray[double] bias0_res
        cdef valarray[double] relu_res
        cdef valarray[double] output_grad_valarray
        cdef valarray[double] input_valarray
        cdef valarray[double] grad_wrt_relu_res
        cdef valarray[double] grad_wrt_bias0_res
        cdef Py_ssize_t input_size
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t output_grad_size
        cdef Py_ssize_t self_n_out_dims
        cdef Py_ssize_t input_id
        cdef Py_ssize_t output_grad_id
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        self_n_in_dims = self.linear0[0].size()
        self_n_out_dims = self.bias1.size()


        input_size = len(cy_input)
        if input_size != self_n_in_dims:
            raise (
                ValueError(
                    "The input size (len(input)={input_size}) "
                    "must be equal to the neural network's number of "
                    "input dimensions (self.shape()[0] = {self_n_in_dims})."
                    .format(**locals())))

        # Convert the input argument to valarray for convenience.
        input_valarray.resize(input_size)
        for input_id in range(input_size):
            input_valarray[<size_t>input_id] = cy_input.view[input_id]

        if output_grad is None:
            # Use a default output gradient set to a vector of 1.'s;
            output_grad_size = self_n_out_dims
            output_grad_valarray.resize(output_grad_size)
            output_grad_valarray = 1.
        else:
            # Check if the output gradient argument is valid and then
            # convert the output gradient argument to valarray  for convenience.
            cy_output_grad = <DoubleArray?> output_grad
            output_grad_size = len(cy_output_grad)
            #
            if output_grad_size != self_n_out_dims:
                    raise (
                        ValueError(
                            "The output gradient size "
                            "(len(output_grad)={output_grad_size}) "
                            "must be equal to the neural network's number of "
                            "output dimensions "
                            "(self.shape()[2] = {self_n_out_dims})."
                            .format(**locals())))
            #
            output_grad_valarray.resize(output_grad_size)
            for output_grad_id in range(output_grad_size):
                output_grad_valarray[<size_t>output_grad_id] = (
                    cy_output_grad.view[output_grad_id])

        # Get intermediate evaluation results that are needed to calculate the
        # gradient.
        linear0_res.resize(self.linear0.size())
        for i in range(linear0_res.size()):
            linear0_res[i] = (self.linear0[i]*input_valarray).sum()
        #
        bias0_res = linear0_res + self.bias0
        #
        relu_res.resize(bias0_res.size())
        for i in range(relu_res.size()):
            relu_res[i] = (
                bias0_res[i] * (bias0_res[i] > 0.)
                + bias0_res[i] * (bias0_res[i] <= 0.) * self.leaky_scale )

        # Initialize sub-gradients w.r.t. parameters to the correct sizes.
        grad_wrt_linear0.resize(self.linear0.size())
        for i in range(grad_wrt_linear0.size()):
            grad_wrt_linear0[i].resize(self.linear0[i].size())
        #
        grad_wrt_linear1.resize(self.linear1.size())
        for i in range(grad_wrt_linear1.size()):
            grad_wrt_linear1[i].resize(self.linear1[i].size())
        #
        grad_wrt_bias0.resize(self.bias0.size())
        #
        grad_wrt_bias1.resize(self.bias1.size())


        # Calculate intermediate gradients.
        grad_wrt_relu_res.resize(self.linear1[0].size())
        grad_wrt_relu_res = 0.
        for i in range(self.linear1.size()):
            grad_wrt_relu_res = (
                grad_wrt_relu_res
                + self.linear1[i]
                * output_grad_valarray[i])
        #

        grad_wrt_bias0_res.resize(bias0_res.size())
        for i in range(bias0_res.size()):
            grad_wrt_bias0_res[i] = (
                grad_wrt_relu_res[i]
                * (bias0_res[i] > 0.)
                + grad_wrt_relu_res[i]
                * (bias0_res[i] <= 0.)
                * self.leaky_scale )

        # Calculate sub-gradient w.r.t. parameters.
        grad_wrt_bias1 = output_grad_valarray
        #
        grad_wrt_bias0 = grad_wrt_bias0_res
        #
        for i in range(self.linear1.size()):
            for j in range(self.linear1[i].size()):
                grad_wrt_linear1[i][j] = (
                    output_grad_valarray[i]
                    * relu_res[j])
        #
        for i in range(self.linear0.size()):
            for j in range(self.linear0[i].size()):
                grad_wrt_linear0[i][j] = (
                    grad_wrt_bias0_res[i]
                    * input_valarray[j])

        # Concatenate sub-gradient w.r.t. parameters into a single DoubleArray.
        grad_wrt_parameters = new_DoubleArray(n_parameters_for_ReluLinear(self))
        parameter_id = 0
        #
        for i in range(grad_wrt_linear0.size()):
            for j in range(grad_wrt_linear0[i].size()):
                grad_wrt_parameters.view[parameter_id] = grad_wrt_linear0[i][j]
                parameter_id += 1
        #
        for i in range(grad_wrt_bias0.size()):
            grad_wrt_parameters.view[parameter_id]  = grad_wrt_bias0[i]
            parameter_id += 1

        for i in range(grad_wrt_linear1.size()):
            for j in range(grad_wrt_linear1[i].size()):
                grad_wrt_parameters.view[parameter_id] = grad_wrt_linear1[i][j]
                parameter_id += 1
        #
        for i in range(grad_wrt_bias1.size()):
            grad_wrt_parameters.view[parameter_id]  = grad_wrt_bias1[i]
            parameter_id += 1

        return grad_wrt_parameters


    cpdef grad_wrt_input(self, input, output_grad = None):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray cy_output_grad
        cdef DoubleArray grad_wrt_input
        cdef valarray[double] linear0_res
        cdef valarray[double] bias0_res
        cdef valarray[double] output_grad_valarray
        cdef valarray[double] input_valarray
        cdef valarray[double] grad_wrt_relu_res
        cdef valarray[double] grad_wrt_bias0_res
        cdef valarray[double] grad_wrt_input_valarray
        cdef Py_ssize_t input_size
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t output_grad_size
        cdef Py_ssize_t self_n_out_dims
        cdef Py_ssize_t input_id
        cdef Py_ssize_t output_grad_id
        cdef size_t i

        self_n_in_dims = self.linear0[0].size()
        self_n_out_dims = self.bias1.size()

        input_size = len(cy_input)
        if input_size != self_n_in_dims:
            raise (
                ValueError(
                    "The input size (len(input)={input_size}) "
                    "must be equal to the neural network's number of "
                    "input dimensions (self.shape()[0] = {self_n_in_dims})."
                    .format(**locals())))

        input_valarray.resize(input_size)
        for input_id in range(input_size):
            input_valarray[<size_t>input_id] = cy_input.view[input_id]

        if output_grad is None:
            output_grad_size = self_n_out_dims
            output_grad_valarray.resize(output_grad_size)
            output_grad_valarray = 1.
        else:
            cy_output_grad = <DoubleArray?> output_grad
            output_grad_size = len(cy_output_grad)

            if output_grad_size != self_n_out_dims:
                    raise (
                        ValueError(
                            "The output gradient size "
                            "(len(output_grad)={output_grad_size}) "
                            "must be equal to the neural network's number of "
                            "output dimensions "
                            "(self.shape()[2] = {self_n_out_dims})."
                            .format(**locals())))


            output_grad_valarray.resize(output_grad_size)
            for output_grad_id in range(output_grad_size):
                output_grad_valarray[<size_t>output_grad_id] = (
                    cy_output_grad.view[output_grad_id])

        linear0_res.resize(self.linear0.size())
        for i in range(linear0_res.size()):
            linear0_res[i] = (self.linear0[i]*input_valarray).sum()

        bias0_res = linear0_res + self.bias0

        grad_wrt_relu_res.resize(self.linear1[0].size())
        grad_wrt_relu_res = 0.
        for i in range(self.linear1.size()):
            grad_wrt_relu_res = (
                grad_wrt_relu_res
                + self.linear1[i]
                * output_grad_valarray[i])


        grad_wrt_bias0_res.resize(bias0_res.size())
        for i in range(bias0_res.size()):
            grad_wrt_bias0_res[i] = (
                grad_wrt_relu_res[i]
                * (bias0_res[i] > 0.)
                + grad_wrt_relu_res[i]
                * (bias0_res[i] <= 0.)
                * self.leaky_scale )

        grad_wrt_input_valarray.resize(self.linear0[0].size())
        grad_wrt_input_valarray = 0.
        for i in range(self.linear0.size()):
            grad_wrt_input_valarray = (
                grad_wrt_input_valarray
                + self.linear0[i]
                * grad_wrt_bias0_res[i])

        grad_wrt_input = new_DoubleArray(input_size)
        for input_id in range(input_size):
            grad_wrt_input.view[input_id] = (
                grad_wrt_input_valarray[<size_t>input_id])

        return grad_wrt_input



@cython.warn.undeclared(True)
cdef ReluLinear new_ReluLinear(
        Py_ssize_t n_in_dims,
        Py_ssize_t n_hidden_neurons,
        Py_ssize_t n_out_dims):
    cdef ReluLinear neural_network

    neural_network = ReluLinear.__new__(ReluLinear)
    init_ReluLinear(
        neural_network,
        n_in_dims,
        n_hidden_neurons,
        n_out_dims)

    return neural_network

@cython.warn.undeclared(True)
cdef void init_ReluLinear(
        ReluLinear neural_network,
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

    # Copy initialization matrices data into ReluLinear fields.
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


    # Set the leaky scale for the ReluLinear.
    neural_network.leaky_scale = 0.

@cython.warn.undeclared(True)
cpdef Py_ssize_t n_parameters_for_ReluLinear(ReluLinear neural_network):
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

def unpickle_ReluLinear(shape, parameters, leaky_scale):
    neural_network = ReluLinear(shape[0], shape[1], shape[2])
    neural_network.set_parameters(parameters)
    neural_network.leaky_scale = leaky_scale


    return neural_network