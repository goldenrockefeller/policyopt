cimport cython

from rockefeg.cyutil.array cimport new_DoubleArray

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseMap:

    cpdef copy(self, copy_obj = None):
        raise NotImplementedError("Abstract method.")

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
