cimport cython

from rockefeg.cyutil.array cimport new_DoubleArray
from .experience cimport ExperienceDatum

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseMap:

    cpdef copy(self, copy_obj = None):
        pass

    cpdef parameters(self):
        raise NotImplementedError("Abstract method.")

    cpdef void set_parameters(self, parameters) except *:
        raise NotImplementedError("Abstract method.")

    cpdef eval(self, input):
        raise NotImplementedError("Abstract method.")

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseDifferentiableMap(BaseMap):

    cpdef jacobian_wrt_parameters(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef jacobian_wrt_input(self, input):
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_parameters(self, input, output_grad = None):
        raise NotImplementedError("Abstract method.")

    cpdef grad_wrt_input(self, input, output_grad = None):
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

    cpdef copy(self, copy_obj = None):
        cdef ContinuousCriticMap new_map
        cdef BaseMap super_map

        if copy_obj is None:
            new_map = ContinuousCriticMap.__new__(ContinuousCriticMap)
        else:
            new_map = copy_obj

        BaseMap.copy(self, new_map)

        super_map = self.__super_map
        new_map.__super_map = super_map.copy()

        return new_map

    cpdef parameters(self):
        cdef BaseMap super_map

        super_map = self.super_map()

        return super_map.parameters()

    cpdef void set_parameters(self, parameters) except *:
        cdef BaseMap super_map

        super_map = self.super_map()

        super_map.set_parameters(parameters)


    cpdef eval(self, input):
        cdef BaseMap super_map

        super_map = self.super_map()

        return super_map.eval(concatenate_state_action(input) )

    cpdef super_map(self):
        return self.__super_map

    cpdef void set_super_map(self, super_map) except *:
        self.__super_map = <BaseMap?>super_map

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

    cpdef copy(self, copy_obj = None):
        cdef DifferentiableCriticMap new_map
        cdef BaseDifferentiableMap super_map

        if copy_obj is None:
            new_map = DifferentiableCriticMap.__new__(DifferentiableCriticMap)
        else:
            new_map = copy_obj

        BaseDifferentiableMap.copy(self, new_map)

        super_map = self.__super_map
        new_map.__super_map = super_map.copy()

        return new_map

    cpdef parameters(self):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return super_map.parameters()

    cpdef void set_parameters(self, parameters) except *:
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        (<BaseDifferentiableMap?>self.__super_map).set_parameters(parameters)

    cpdef eval(self, input):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return super_map.eval(concatenate_state_action(input) )

    cpdef jacobian_wrt_parameters(self, input):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return (
            super_map.jacobian_wrt_parameters(
                concatenate_state_action(input) ))

    cpdef jacobian_wrt_input(self, input):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return (
            super_map.jacobian_wrt_input(
                concatenate_state_action(input) ))

    cpdef grad_wrt_parameters(self, input, output_grad = None):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return (
            super_map.grad_wrt_parameters(
                concatenate_state_action(input), output_grad))

    cpdef grad_wrt_input(self, input, output_grad = None):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return (
            super_map.grad_wrt_input(
                concatenate_state_action(input), output_grad))

    cpdef super_map(self):
        return self.__super_map

    cpdef void set_super_map(self, super_map) except *:
        self.__super_map = <BaseDifferentiableMap?>super_map

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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DoubleArray concatenate_state_action(input):
    cdef ExperienceDatum cy_input
    cdef DoubleArray state
    cdef DoubleArray action
    cdef DoubleArray state_action
    cdef Py_ssize_t n_state_dims
    cdef Py_ssize_t n_action_dims
    cdef Py_ssize_t id

    if isinstance(input, DoubleArray):
        return input

    cy_input = <ExperienceDatum?> input


    state = <DoubleArray?>cy_input.state
    action = <DoubleArray?>cy_input.action

    n_state_dims = len(state)
    n_action_dims = len(action)
    state_action = new_DoubleArray(n_state_dims + n_action_dims)

    for id in range(n_state_dims):
        state_action.view[id] = state.view[id]

    for id in range(n_action_dims):
        state_action.view[id + n_state_dims] = action.view[id]

    return state_action

