cimport cython
from libc cimport math as cmath
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from .map cimport default_jacobian_wrt_parameters, default_jacobian_wrt_input
from libc.math cimport tanh

import numpy as np

@cython.warn.undeclared(True)
cdef class TanhLayer(BaseDifferentiableMap):
    def __init__(self, BaseDifferentiableMap super_map):
        init_TanhLayer(self, super_map)

    def __reduce__(self):
        return unpickle_TanhLayer, (self.super_map)

    cpdef copy(self, copy_obj = None):
        cdef TanhLayer new_neural_network

        if copy_obj is None:
            new_neural_network = TanhLayer.__new__(TanhLayer)
        else:
            new_neural_network = copy_obj

        new_neural_network.super_map = self.super_map.copy()

        return new_neural_network


    cpdef Py_ssize_t n_parameters(self) except *:
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map

        return super_map.n_parameters()


    cpdef parameters(self):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map

        return super_map.parameters()

    cpdef void set_parameters(self, parameters) except *:
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map

        super_map.set_parameters(parameters)

    cpdef eval(self, input):
        cdef BaseDifferentiableMap super_map
        cdef DoubleArray eval
        cdef Py_ssize_t eval_id

        super_map = self.super_map

        eval = super_map.eval(input)
        for eval_id in range(len(eval)):
            eval.view[eval_id] = tanh(eval.view[eval_id])

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
cdef TanhLayer new_TanhLayer(BaseDifferentiableMap super_map):
    cdef TanhLayer neural_network

    neural_network = TanhLayer.__new__(TanhLayer)
    init_TanhLayer(neural_network, super_map)

    return neural_network

@cython.warn.undeclared(True)
cdef void init_TanhLayer(
        TanhLayer neural_network,
        BaseDifferentiableMap super_map
        ) except *:
    neural_network.super_map = super_map



def unpickle_TanhLayer(super_map):
    neural_network = TanhLayer(super_map)

    return neural_network


@cython.warn.undeclared(True)
cdef class ReluLinear(BaseDifferentiableMap):
    def __init__(
            self,
            n_in_dims,
            n_hidden_neurons,
            n_out_dims,
            linear1_is_fixed = False):
        init_ReluLinear(
            self,
            n_in_dims,
            n_hidden_neurons,
            n_out_dims,
            linear1_is_fixed)

    def __reduce__(self):
        cdef DoubleArray all_parameters
        cdef bint linear1_is_fixed

        linear1_is_fixed = self.linear1_is_fixed
        self.linear1_is_fixed = False
        all_parameters = self.parameters()
        self.linear1_is_fixed = linear1_is_fixed

        return (
            unpickle_ReluLinear,
            (
            self.shape(),
            all_parameters,
            self.leaky_scale,
            self.linear1_is_fixed
            ) )

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
        new_neural_network.linear1_is_fixed = self.linear1_is_fixed

        return new_neural_network

    cpdef tuple shape(self):
        return (self.linear0[0].size(), self.bias0.size(), self.bias1.size())

    cpdef Py_ssize_t n_parameters(self) except *:
        cdef Py_ssize_t n_parameters

        n_parameters = 0
        n_parameters += (
            self.linear0.size()
            * self.linear0[0].size())
        if not self.linear1_is_fixed:
            n_parameters += (
                self.linear1.size()
                * self.linear1[0].size())
        n_parameters += self.bias0.size()
        n_parameters += self.bias1.size()

        return n_parameters


    cpdef parameters(self):
        cdef DoubleArray parameters
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        parameters = new_DoubleArray(self.n_parameters())
        parameter_id = 0

        for i in range(self.linear0.size()):
            for j in range(self.linear0[i].size()):
                parameters.view[parameter_id] = self.linear0[i][j]
                parameter_id += 1
        #
        for i in range(self.bias0.size()):
            parameters.view[parameter_id]  = self.bias0[i]
            parameter_id += 1

        if not self.linear1_is_fixed:
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
        n_self_parameters = self.n_parameters()

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

        if not self.linear1_is_fixed:
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
            assign_valarray_to_double(output_grad_valarray, 1.)
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
        # for i in range(grad_wrt_linear0.size()):
        #     grad_wrt_linear0[i].resize(self.linear0[i].size())
        #
        grad_wrt_linear1.resize(self.linear1.size())
        # for i in range(grad_wrt_linear1.size()):
        #     grad_wrt_linear1[i].resize(self.linear1[i].size())
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
            grad_wrt_linear1[i] = output_grad_valarray[i] * relu_res
            # for j in range(self.linear1[i].size()):
            #     grad_wrt_linear1[i][j] = (
            #         output_grad_valarray[i]
            #         * relu_res[j])
        #
        for i in range(self.linear0.size()):
            grad_wrt_linear0[i] = grad_wrt_bias0_res[i] * input_valarray
            # for j in range(self.linear0[i].size()):
            #     grad_wrt_linear0[i][j] = (
            #         grad_wrt_bias0_res[i]
            #         * input_valarray[j])



        # Concatenate sub-gradient w.r.t. parameters into a single DoubleArray.
        grad_wrt_parameters = new_DoubleArray(self.n_parameters())
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

        if not self.linear1_is_fixed:
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
            assign_valarray_to_double(output_grad_valarray, 1.)
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
        assign_valarray_to_double(grad_wrt_input_valarray, 0.)
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
        Py_ssize_t n_out_dims,
        bint linear1_is_fixed = False):
    cdef ReluLinear neural_network

    neural_network = ReluLinear.__new__(ReluLinear)
    init_ReluLinear(
        neural_network,
        n_in_dims,
        n_hidden_neurons,
        n_out_dims,
        linear1_is_fixed)

    return neural_network

@cython.warn.undeclared(True)
cdef void init_ReluLinear(
        ReluLinear neural_network,
        Py_ssize_t n_in_dims,
        Py_ssize_t n_hidden_neurons,
        Py_ssize_t n_out_dims,
        bint linear1_is_fixed = False
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

    neural_network.linear1_is_fixed = linear1_is_fixed

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
            if linear1_is_fixed:
                 neural_network.linear1[i][j] = 1.
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



def unpickle_ReluLinear(shape, parameters, leaky_scale, linear1_is_fixed):
    neural_network = ReluLinear(shape[0], shape[1], shape[2])
    neural_network.linear1_is_fixed = False
    neural_network.set_parameters(parameters)
    neural_network.leaky_scale = leaky_scale
    neural_network.linear1_is_fixed = linear1_is_fixed


    return neural_network


@cython.warn.undeclared(True)
cdef class Rbfn(BaseDifferentiableMap): # Radial Basis Function Network
    def __init__(
            self,
            n_in_dims,
            n_centers,
            n_out_dims):
        init_Rbfn(
            self,
            n_in_dims,
            n_centers,
            n_out_dims)

    def __reduce__(self):
        cdef DoubleArray all_parameters
        cdef bint scalings_are_fixed

        scalings_are_fixed = self.scalings_are_fixed
        self.scalings_are_fixed = False
        all_parameters = self.parameters()
        self.scalings_are_fixed = scalings_are_fixed

        return (
            unpickle_Rbfn,
            (
            self.shape(),
            all_parameters,
            self.scalings_are_fixed,
            self.normalizes_activations
            ) )


    cpdef copy(self, copy_obj = None):
        cdef Rbfn new_neural_network

        if copy_obj is None:
            new_neural_network = Rbfn.__new__(Rbfn)
        else:
            new_neural_network = copy_obj

        new_neural_network.centers = self.centers
        new_neural_network.scalings = self.scalings
        new_neural_network.transform = self.transform
        new_neural_network.scalings_are_fixed = self.scalings_are_fixed
        new_neural_network.normalizes_activations = self.normalizes_activations

        return new_neural_network

    cpdef tuple shape(self):
        return (
            self.centers[0].size(),
            self.centers.size(),
            self.transform.size())

    cpdef Py_ssize_t n_parameters(self) except *:
        cdef Py_ssize_t n_parameters

        n_parameters = 0


        n_parameters += (
            self.centers.size()
            * self.centers[0].size())
        if not self.scalings_are_fixed:
            n_parameters += (
                self.scalings.size()
                * self.scalings[0].size())
        n_parameters += (
            self.transform.size()
            * self.transform[0].size())
        return n_parameters

    cpdef parameters(self):
        cdef DoubleArray parameters
        cdef Py_ssize_t parameter_id
        cdef size_t i
        cdef size_t j

        parameters = new_DoubleArray(self.n_parameters())
        parameter_id = 0

        for i in range(self.centers.size()):
            for j in range(self.centers[i].size()):
                parameters.view[parameter_id] = self.centers[i][j]
                parameter_id += 1
        #
        if not self.scalings_are_fixed:
            for i in range(self.scalings.size()):
                for j in range(self.scalings[i].size()):
                    parameters.view[parameter_id] = self.scalings[i][j]
                    parameter_id += 1
        #
        for i in range(self.transform.size()):
            for j in range(self.transform[i].size()):
                parameters.view[parameter_id] = self.transform[i][j]
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
        n_self_parameters = self.n_parameters()

        if n_setting_parameters != n_self_parameters:
            raise (
                ValueError(
                    "The number of setting parameters "
                    "(len(parameters)={n_setting_parameters}) "
                    "must be equal to the number of neural network parameters "
                    "(len(self.parameters())={n_self_parameters})."
                    .format(**locals())))

        parameter_id = 0


        for i in range(self.centers.size()):
            for j in range(self.centers[0].size()):
                self.centers[i][j] = cy_parameters.view[parameter_id]
                parameter_id += 1
        #
        if not self.scalings_are_fixed:
            for i in range(self.scalings.size()):
                for j in range(self.scalings[0].size()):
                    self.scalings[i][j] = cy_parameters.view[parameter_id]
                    parameter_id += 1
        #
        for i in range(self.transform.size()):
            for j in range(self.transform[0].size()):
                self.transform[i][j] = cy_parameters.view[parameter_id]
                parameter_id += 1

    cpdef activations_eval(self, input):
        cdef DoubleArray eval
        cdef double eval_sum

        eval = rbfn_pre_norm_activations_eval(self, input)

        if self.normalizes_activations:
            eval = normalization_for_DoubleArray(eval)

        return eval

    cpdef eval(self, input):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray eval
        cdef DoubleArray activations
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t input_size
        cdef valarray[double] input_valarray
        cdef valarray[double] activations_valarray
        cdef valarray[double] transform_res
        cdef size_t i
        cdef Py_ssize_t input_id
        cdef Py_ssize_t eval_id

        input_size = len(cy_input)
        self_n_in_dims = self.centers[0].size()

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

        eval = new_DoubleArray(self.transform.size())

        activations = self.activations_eval(input)
        activations_valarray.resize(len(activations))
        for eval_id in range(len(activations)):
            activations_valarray[<size_t>eval_id] = activations.view[eval_id]

        transform_res.resize(self.transform.size())
        for i in range(transform_res.size()):
            transform_res[i] = (self.transform[i]*activations_valarray).sum()

        for eval_id in range(len(eval)):
            eval.view[eval_id] = transform_res[<size_t>eval_id]

        return eval

    cpdef jacobian_wrt_parameters(self, input):
        return default_jacobian_wrt_parameters(self, input)

    cpdef jacobian_wrt_input(self, input):
        return default_jacobian_wrt_input(self, input)

    cpdef grad_wrt_parameters(self, input, output_grad = None):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray cy_output_grad
        cdef DoubleArray grad_wrt_parameters
        cdef DoubleArray pre_norm_activations
        cdef DoubleArray activations
        cdef vector[valarray[double]] grad_wrt_centers
        cdef vector[valarray[double]] grad_wrt_scalings
        cdef vector[valarray[double]] grad_wrt_transform
        cdef valarray[double] pre_norm_activations_valarray
        cdef valarray[double] activations_valarray
        cdef valarray[double] output_grad_valarray
        cdef valarray[double] input_valarray
        cdef valarray[double] grad_wrt_pre_norm_activations_res
        cdef valarray[double] grad_wrt_activations_eval
        cdef valarray[double] separation_res
        cdef valarray[double] scaling_res
        cdef Py_ssize_t input_size
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t output_grad_size
        cdef Py_ssize_t self_n_out_dims
        cdef Py_ssize_t input_id
        cdef Py_ssize_t output_grad_id
        cdef Py_ssize_t parameter_id
        cdef Py_ssize_t eval_id
        cdef size_t i
        cdef size_t j

        self_n_in_dims = self.centers[0].size()
        self_n_out_dims = self.transform.size()


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
            assign_valarray_to_double(output_grad_valarray, 1.)
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
        pre_norm_activations = rbfn_pre_norm_activations_eval(self, input)
        pre_norm_activations_valarray.resize(len(pre_norm_activations))
        for eval_id in range(len(pre_norm_activations)):
            pre_norm_activations_valarray[<size_t>eval_id] = (
                pre_norm_activations.view[eval_id])
        #
        activations = pre_norm_activations
        if self.normalizes_activations:
            activations = normalization_for_DoubleArray(pre_norm_activations)
        activations_valarray.resize(len(activations))
        for eval_id in range(len(activations)):
            activations_valarray[<size_t>eval_id] = (
                activations.view[eval_id])



        # Initialize sub-gradients w.r.t. parameters to the correct sizes.
        grad_wrt_centers.resize(self.centers.size())
        # for i in range(grad_wrt_centers.size()):
        #     grad_wrt_centers[i].resize(self.centers[i].size())
        #
        grad_wrt_scalings.resize(self.scalings.size())
        # for i in range(grad_wrt_scalings.size()):
        #     grad_wrt_scalings[i].resize(self.scalings[i].size())
        #
        grad_wrt_transform.resize(self.transform.size())
        # for i in range(grad_wrt_transform.size()):
        #     grad_wrt_transform[i].resize(self.transform[i].size())
        #


        # Calculate intermediate gradients.
        grad_wrt_activations_eval.resize(self.transform[0].size())
        assign_valarray_to_double(grad_wrt_activations_eval, 0.)
        for i in range(self.transform.size()):
            grad_wrt_activations_eval = (
                grad_wrt_activations_eval
                + output_grad_valarray[i]
                * self.transform[i] )
        #
        grad_wrt_pre_norm_activations_res = grad_wrt_activations_eval
        if self.normalizes_activations:
            grad_wrt_pre_norm_activations_res = (
                (
                grad_wrt_pre_norm_activations_res
                - (grad_wrt_activations_eval * activations_valarray).sum()
                )
                / pre_norm_activations_valarray.sum())

        # Calculate sub-gradient w.r.t. parameters.
        for i in range(self.transform.size()):
            grad_wrt_transform[i] = output_grad_valarray[i] * activations_valarray
        #
        for i in range(self.centers.size()):
            separation_res = self.centers[i] - input_valarray
            scaling_res =  self.scalings[i] * separation_res
            grad_wrt_centers[i] = (
                -2.
                * grad_wrt_pre_norm_activations_res[i]
                * pre_norm_activations_valarray[i]
                * pre_norm_activations_valarray[i]
                * scaling_res
                * self.scalings[i] )
            grad_wrt_scalings[i] = (
                -2.
                * grad_wrt_pre_norm_activations_res[i]
                * pre_norm_activations_valarray[i]
                * pre_norm_activations_valarray[i]
                * scaling_res
                * separation_res )


        # Concatenate sub-gradient w.r.t. parameters into a single DoubleArray.
        grad_wrt_parameters = new_DoubleArray(self.n_parameters())
        parameter_id = 0
        #
        for i in range(self.centers.size()):
            for j in range(self.centers[i].size()):
                grad_wrt_parameters.view[parameter_id] = grad_wrt_centers[i][j]
                parameter_id += 1
        #
        if not self.scalings_are_fixed:
            for i in range(self.scalings.size()):
                for j in range(self.scalings[i].size()):
                    grad_wrt_parameters.view[parameter_id] = grad_wrt_scalings[i][j]
                    parameter_id += 1
        #
        for i in range(self.transform.size()):
            for j in range(self.transform[i].size()):
                grad_wrt_parameters.view[parameter_id] = grad_wrt_transform[i][j]
                parameter_id += 1

        return grad_wrt_parameters

    cpdef grad_wrt_input(self, input, output_grad = None):
        cdef DoubleArray cy_input = <DoubleArray?> input
        cdef DoubleArray cy_output_grad
        cdef DoubleArray grad_wrt_input
        cdef DoubleArray pre_norm_activations
        cdef DoubleArray activations
        cdef valarray[double] pre_norm_activations_valarray
        cdef valarray[double] activations_valarray
        cdef valarray[double] output_grad_valarray
        cdef valarray[double] input_valarray
        cdef valarray[double] grad_wrt_pre_norm_activations_res
        cdef valarray[double] grad_wrt_activations_eval
        cdef valarray[double] separation_res
        cdef valarray[double] scaling_res
        cdef valarray[double] grad_wrt_input_valarray
        cdef Py_ssize_t input_size
        cdef Py_ssize_t self_n_in_dims
        cdef Py_ssize_t output_grad_size
        cdef Py_ssize_t self_n_out_dims
        cdef Py_ssize_t input_id
        cdef Py_ssize_t output_grad_id
        cdef Py_ssize_t parameter_id
        cdef Py_ssize_t eval_id
        cdef size_t i
        cdef size_t j

        self_n_in_dims = self.centers[0].size()
        self_n_out_dims = self.transform.size()


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
            assign_valarray_to_double(output_grad_valarray, 1.)
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
        pre_norm_activations = rbfn_pre_norm_activations_eval(self, input)
        pre_norm_activations_valarray.resize(len(pre_norm_activations))
        for eval_id in range(len(pre_norm_activations)):
            pre_norm_activations_valarray[<size_t>eval_id] = (
                pre_norm_activations.view[eval_id])
        #
        activations = pre_norm_activations
        if self.normalizes_activations:
            activations = normalization_for_DoubleArray(pre_norm_activations)
        activations_valarray.resize(len(activations))
        for eval_id in range(len(activations)):
            activations_valarray[<size_t>eval_id] = (
                activations.view[eval_id])

        # Calculate intermediate gradients.
        grad_wrt_activations_eval.resize(self.transform[0].size())
        assign_valarray_to_double(grad_wrt_activations_eval, 0.)
        for i in range(self.transform.size()):
            grad_wrt_activations_eval = (
                grad_wrt_activations_eval
                + output_grad_valarray[i]
                * self.transform[i] )
        #
        grad_wrt_pre_norm_activations_res = grad_wrt_activations_eval
        if self.normalizes_activations:
            grad_wrt_pre_norm_activations_res = (
                (
                grad_wrt_pre_norm_activations_res
                - (grad_wrt_activations_eval * activations_valarray).sum()
                )
                / pre_norm_activations_valarray.sum())

        # Concatenate sub-gradient w.r.t. input into a single DoubleArray.
        grad_wrt_input_valarray.resize(self.centers[0].size())
        assign_valarray_to_double(grad_wrt_input_valarray, 0.)
        for i in range(self.centers.size()):
            separation_res = self.centers[i] - input_valarray
            scaling_res =  self.scalings[i] * separation_res
            grad_wrt_input_valarray = (
                2.
                * grad_wrt_pre_norm_activations_res[i]
                * pre_norm_activations_valarray[i]
                * pre_norm_activations_valarray[i]
                * self.scalings[i]
                * scaling_res
                + grad_wrt_input_valarray
                )
        #
        grad_wrt_input = new_DoubleArray(input_size)
        for input_id in range(input_size):
            grad_wrt_input.view[input_id] = (
                grad_wrt_input_valarray[<size_t>input_id])

        return grad_wrt_input



@cython.warn.undeclared(True)
cdef Rbfn new_Rbfn(
        Py_ssize_t n_in_dims,
        Py_ssize_t n_centers,
        Py_ssize_t n_out_dims):
    cdef Rbfn neural_network

    neural_network = Rbfn.__new__(Rbfn)
    init_Rbfn(
        neural_network,
        n_in_dims,
        n_centers,
        n_out_dims)

    return neural_network

@cython.warn.undeclared(True)
cdef void init_Rbfn(
        Rbfn neural_network,
        Py_ssize_t n_in_dims,
        Py_ssize_t n_centers,
        Py_ssize_t n_out_dims,
        ) except *:
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double[:, ::1] np_centers
    cdef double[:, ::1] np_transform

    if neural_network is None:
        raise TypeError("The neural network (neural_network) cannot be None.")

    neural_network.scalings_are_fixed = False
    neural_network.normalizes_activations = True

    if n_in_dims <= 0:
        raise (
            ValueError(
                "The number of input dimensions (n_in_dims = {n_in_dims}) "
                "must be positive."
                .format(**locals())))

    if n_centers <= 0:
        raise (
            ValueError(
                "The number of centers (n_centers = {n_centers}) "
                "must be positive."
                .format(**locals())))

    if n_out_dims <= 0:
        raise (
            ValueError(
                "The number of output dimensions (n_out_dims = {n_out_dims}) "
                "must be positive."
                .format(**locals())))

    # Get initialization matrices for the transform layer using numpy.
    np_centers= (
        np.random.uniform(
            -1.,
            1.,
            (n_centers, n_in_dims) ))

    np_transform= (
        np.random.uniform(
            -1.,
            1.,
            (n_out_dims, n_centers) ))

    # Copy initialization matrices data into neural network's fields.
    neural_network.centers.resize(np_centers.shape[0])
    for i in range(np_centers.shape[0]):
        neural_network.centers[i].resize(np_centers.shape[1])
        for j in range(np_centers.shape[1]):
            neural_network.centers[i][j] = np_centers[i, j]
    #
    neural_network.scalings.resize(np_centers.shape[0])
    for i in range(np_centers.shape[0]):
        neural_network.scalings[i].resize(np_centers.shape[1])
        for j in range(np_centers.shape[1]):
            neural_network.scalings[i][j] = 1.
    #
    neural_network.transform.resize(np_transform.shape[0])
    for i in range(np_transform.shape[0]):
        neural_network.transform[i].resize(np_transform.shape[1])
        for j in range(np_transform.shape[1]):
            neural_network.transform[i][j] = np_transform[i, j]

@cython.warn.undeclared(True)
cpdef DoubleArray rbfn_pre_norm_activations_eval(Rbfn self, DoubleArray input):
    cdef DoubleArray eval
    cdef Py_ssize_t self_n_in_dims
    cdef Py_ssize_t input_size
    cdef valarray[double] input_valarray
    cdef valarray[double] scaling_res
    cdef valarray[double] separation_res
    cdef valarray[double] scaling_res_sqr
    cdef double dist_sqr
    cdef size_t i
    cdef Py_ssize_t input_id
    cdef Py_ssize_t eval_id

    input_size = len(input)
    self_n_in_dims = self.centers[0].size()

    if input_size != self_n_in_dims:
        raise (
            ValueError(
                "The input size (len(input)={input_size}) "
                "must be equal to the neural network's number of "
                "input dimensions (self.shape()[0] = {self_n_in_dims})."
                .format(**locals())))

    input_valarray.resize(input_size)
    for input_id in range(input_size):
        input_valarray[<size_t>input_id] = input.view[input_id]

    eval = new_DoubleArray(self.centers.size())

    for i in range(self.centers.size()):
        separation_res = self.centers[i] - input_valarray
        scaling_res =  self.scalings[i] * separation_res
        scaling_res_sqr = scaling_res * scaling_res
        dist_sqr = scaling_res_sqr.sum()
        # Inverse quadratic radial basis function:
        eval.view[<Py_ssize_t>i] = 1. / (1. + dist_sqr)

    return eval

@cython.warn.undeclared(True)
cpdef DoubleArray normalization_for_DoubleArray(DoubleArray arr):
    cdef double arr_sum
    cdef Py_ssize_t i
    cdef DoubleArray normalized_arr

    normalized_arr = new_DoubleArray(len(arr))

    arr_sum = 0.
    for i in range(len(arr)):
        arr_sum += arr.view[i]

    if arr_sum == 0.:
        normalized_arr.view[i] = 1. / <double>len(arr)

    else:
        for i in range(len(arr)):
            normalized_arr.view[i] = arr.view[i] / arr_sum

    return normalized_arr

cdef inline void assign_valarray_to_double(
        valarray[double] arr,
        double val
        ) except *:
    cdef size_t id

    for id in range(arr.size()):
        arr[id] = val

def unpickle_Rbfn(shape, parameters, scalings_are_fixed, normalizes_activations):
    neural_network = Rbfn(shape[0], shape[1], shape[2])
    neural_network.scalings_are_fixed = False
    neural_network.set_parameters(parameters)
    neural_network.scalings_are_fixed = scalings_are_fixed
    neural_network.normalizes_activations = normalizes_activations

    return neural_network