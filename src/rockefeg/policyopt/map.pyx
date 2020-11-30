cimport cython

from rockefeg.cyutil.array cimport new_DoubleArray
from .experience cimport ExperienceDatum

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseMap:

    cpdef BaseMap copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef void set_parameters(self, parameters) except *:
        raise NotImplementedError("Abstract method.")

    cpdef Py_ssize_t n_parameters(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef eval(self, input):
        raise NotImplementedError("Abstract method.")


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseDifferentiableMap(BaseMap):

    cpdef BaseDifferentiableMap copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

    cpdef DoubleArray parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef DoubleArray eval(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef list jacobian_wrt_parameters(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef list jacobian_wrt_input(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef DoubleArray grad_wrt_parameters(self, input, output_grad = None):
        raise NotImplementedError("Abstract method.")

    cpdef DoubleArray grad_wrt_input(self, input, output_grad = None):
        raise NotImplementedError("Abstract method.")

cpdef list default_jacobian_wrt_parameters(
        BaseDifferentiableMap map,
        DoubleArray input):
    cdef list jacobian
    cdef Py_ssize_t output_size
    cdef Py_ssize_t id
    cdef DoubleArray output_grad
    cdef object output

    output = map.eval(input)
    output_size = len(output)
    jacobian = [None] * output_size

    output_grad = new_DoubleArray(output_size)
    output_grad.set_all_to(0.)

    for id in range(output_size):
        output_grad.view[id] = 1.
        jacobian[id] = map.grad_wrt_parameters(input, output_grad)
        output_grad.view[id] = 0.

    return jacobian

cpdef list default_jacobian_wrt_input(
        BaseDifferentiableMap map,
        DoubleArray input):
    cdef list jacobian
    cdef Py_ssize_t output_size
    cdef Py_ssize_t id
    cdef DoubleArray output_grad
    cdef object output

    output = map.eval(input)
    output_size = len(output)
    jacobian = [None] * output_size

    output_grad = new_DoubleArray(output_size)
    output_grad.set_all_to(0.)

    for id in range(output_size):
        output_grad.view[id] = 1.
        jacobian[id] = map.grad_wrt_input(input, output_grad)
        output_grad.view[id] = 0.


    return jacobian


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ContinuousCriticMap(BaseMap):
    def __init__(self, BaseMap super_map):
        init_ContinuousCriticMap(self, super_map)

    cpdef ContinuousCriticMap copy(self, copy_obj = None):
        cdef ContinuousCriticMap new_map

        if copy_obj is None:
            new_map = ContinuousCriticMap.__new__(ContinuousCriticMap)
        else:
            new_map = copy_obj

        BaseMap.copy(self, new_map)

        new_map.__super_map = self.__super_map.copy()

        return new_map

    cpdef parameters(self):
        return self.super_map().parameters()

    cpdef void set_parameters(self, parameters) except *:
        self.super_map().set_parameters(parameters)

    cpdef Py_ssize_t n_parameters(self) except *:
        return self.super_map().n_parameters()


    cpdef eval(self, input):
        return self.super_map().eval(concatenate_observation_action(input) )

    cpdef BaseMap super_map(self):
        return self.__super_map

    cpdef void set_super_map(self, BaseMap super_map) except *:
        self.__super_map = super_map

@cython.warn.undeclared(True)
cdef ContinuousCriticMap new_ContinuousCriticMap(BaseMap super_map):
    cdef ContinuousCriticMap map

    map = ContinuousCriticMap.__new__(ContinuousCriticMap)
    init_ContinuousCriticMap(map, super_map)

    return map

@cython.warn.undeclared(True)
cdef void init_ContinuousCriticMap(
        ContinuousCriticMap map,
        BaseMap super_map
        ) except *:
    if map is None:
        raise TypeError("The critic map (map) cannot be None.")

    if super_map is None:
        raise TypeError("The map (super_map) cannot be None.")

    map.__super_map = super_map

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DifferentiableCriticMap(BaseDifferentiableMap):
    def __init__(self, BaseDifferentiableMap super_map):
        init_DifferentiableCriticMap(self, super_map)

    cpdef DifferentiableCriticMap copy(self, copy_obj = None):
        cdef DifferentiableCriticMap new_map

        if copy_obj is None:
            new_map = DifferentiableCriticMap.__new__(DifferentiableCriticMap)
        else:
            new_map = copy_obj

        BaseDifferentiableMap.copy(self, new_map)

        new_map.__super_map = self.__super_map.copy()

        return new_map

    cpdef parameters(self):
        return self.super_map().parameters()

    cpdef void set_parameters(self, parameters) except *:
        self.super_map().set_parameters(parameters)

    cpdef Py_ssize_t n_parameters(self) except *:
        return self.super_map().n_parameters()

    cpdef DoubleArray eval(self, input):
        return self.super_map().eval(concatenate_observation_action(input) )

    cpdef list jacobian_wrt_parameters(self, input):
        return (
            self.super_map().jacobian_wrt_parameters(
                concatenate_observation_action(input) ))

    cpdef list jacobian_wrt_input(self, input):
        return (
            self.super_map().jacobian_wrt_input(
                concatenate_observation_action(input) ))

    cpdef DoubleArray grad_wrt_parameters(self, input, output_grad = None):
        return (
            self.super_map().grad_wrt_parameters(
                concatenate_observation_action(input), output_grad))

    cpdef DoubleArray grad_wrt_input(self, input, output_grad = None):
        return (
            self.super_map().grad_wrt_input(
                concatenate_observation_action(input), output_grad))

    cpdef BaseDifferentiableMap super_map(self):
        return self.__super_map

    cpdef void set_super_map(self, BaseDifferentiableMap super_map) except *:
        self.__super_map = super_map

@cython.warn.undeclared(True)
cdef DifferentiableCriticMap new_DifferentiableCriticMap(
        BaseDifferentiableMap super_map):
    cdef DifferentiableCriticMap map

    map = DifferentiableCriticMap.__new__(DifferentiableCriticMap)
    init_DifferentiableCriticMap(map, super_map)

    return map

@cython.warn.undeclared(True)
cdef void init_DifferentiableCriticMap(
        DifferentiableCriticMap map,
        BaseDifferentiableMap super_map
        ) except *:
    if map is None:
        raise TypeError("The critic map (map) cannot be None.")

    if super_map is None:
        raise TypeError("The map (super_map) cannot be None.")

    map.__super_map = super_map


cpdef DoubleArray concatenate_observation_action(input):
    cdef ExperienceDatum cy_input
    cdef DoubleArray observation
    cdef DoubleArray action
    cdef DoubleArray observation_action
    cdef Py_ssize_t n_observation_dims
    cdef Py_ssize_t n_action_dims
    cdef Py_ssize_t id

    if isinstance(input, DoubleArray):
        return input

    cy_input = input


    observation = cy_input.observation
    action = cy_input.action

    n_observation_dims = len(observation)
    n_action_dims = len(action)
    observation_action = new_DoubleArray(n_observation_dims + n_action_dims)

    for id in range(n_observation_dims):
        observation_action.view[id] = observation.view[id]

    for id in range(n_action_dims):
        observation_action.view[id + n_observation_dims] = action.view[id]

    return observation_action

