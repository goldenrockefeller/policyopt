cimport cython
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ValueTargetEntry:
    def __init__(self, input, target):
        init_ValueTargetEntry(self, input, target)


@cython.warn.undeclared(True)
cdef ValueTargetEntry new_ValueTargetEntry(input, target):
    cdef ValueTargetEntry entry

    entry = ValueTargetEntry.__new__(ValueTargetEntry)
    init_ValueTargetEntry(entry, input, target)

    return entry

@cython.warn.undeclared(True)
cdef void init_ValueTargetEntry(ValueTargetEntry entry, input, target) except *:
    if entry is None:
        raise TypeError("The INSERT_entry (entry) cannot be None.")

    entry.input = input
    entry.target = target


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseFunctionApproximator(BaseMap):

    cpdef batch_update(self, list entries):
        raise NotImplementedError("Abstract method.")


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
    def __init__(self, super_map):
        init_DifferentiableFunctionApproximator(self, super_map)

    cpdef copy(self, copy_obj = None):
        cdef DifferentiableFunctionApproximator new_approximator

        if copy_obj is None:
            new_approximator = (
                DifferentiableFunctionApproximator.__new__(
                    DifferentiableFunctionApproximator))
        else:
            new_approximator = copy_obj

        new_approximator.__super_map = (<BaseMap?>self.__super_map).copy()
        new_approximator.__learning_rate = self.__learning_rate

        return new_approximator

    cpdef parameters(self):
        return (<BaseMap?>self.__super_map).parameters()

    cpdef void set_parameters(self, parameters) except *:
        (<BaseMap?>self.__super_map).set_parameters(parameters)

    cpdef eval(self, input):
        return (<BaseMap?>self.__super_map).eval(input)

    cpdef batch_update(self, list entries):
        cdef DoubleArray parameters
        cdef DoubleArray input
        cdef DoubleArray output
        cdef DoubleArray target
        cdef DoubleArray error
        cdef DoubleArray entry_grad_wrt_parameters
        cdef DoubleArray sum_grad_wrt_parameters
        cdef BaseDifferentiableMap map
        cdef ValueTargetEntry entry
        cdef Py_ssize_t n_entries
        cdef Py_ssize_t n_entry_parameters
        cdef Py_ssize_t n_parameters
        cdef Py_ssize_t output_size
        cdef Py_ssize_t target_size
        cdef Py_ssize_t output_id
        cdef Py_ssize_t parameter_id

        if entries is None:
            raise TypeError("The entries (entries) can not be None.")

        map = self.__super_map
        parameters = (<DoubleArray?>map.parameters()).copy()
        n_parameters = len(parameters)
        sum_grad_wrt_parameters = new_DoubleArray(n_parameters)
        sum_grad_wrt_parameters.set_all_to(0.)
        n_entries = len(entries)

        for entry in entries:


            input = entry.input
            output = map.eval(input)

            # Calculate the error for this entry.
            target = entry.target
            output_size = len(output)
            target_size = len(target)
            #
            if output_size != target_size:
                raise (
                    ValueError(
                        "The output size (len(self.super_map()"
                        ".eval(entries.input[?])) "
                        "= {output_size}) "
                        "must be equal to the target size "
                        "(len(entries.target[?]) = {target_size})."
                        .format(**locals()) ))
            #
            error = new_DoubleArray(output_size)
            #
            for output_id in range(output_size):
                error.view[output_id] = (
                    target.view[output_id]
                    - output.view[output_id])

            # Add gradient for this entry to the cumulative gradient.
            entry_grad_wrt_parameters = map.grad_wrt_parameters(input, error)
            n_entry_parameters = len(entry_grad_wrt_parameters)
            #
            if n_parameters != n_entry_parameters:
                raise (
                    ValueError(
                        "The number of the map's parameters can not change "
                        "during a batch update. The starting number of "
                        "parameters (len(self.super_map().parameters() = "
                        "{n_parameters}) does not match the current number "
                        "of parameters (len(map.super_map."
                        "grad_wrt_parameters(input, error) = "
                        "{n_entry_parameters})."
                        .format(**locals()) ))
            #
            for parameter_id in range(n_parameters):
                sum_grad_wrt_parameters.view[parameter_id] -= (
                    entry_grad_wrt_parameters.view[parameter_id]
                    * self.__learning_rate
                    / n_entries)

        # Add gradient to parameters
        for parameter_id in range(n_parameters):
            parameters.view[parameter_id] += (
                sum_grad_wrt_parameters.view[parameter_id])

        # Update the maps parameters.
        map.set_paramters(parameters)


    cpdef super_map(self):
        return self.__super_map

    cpdef _set_super_map(self, map):
        if map is None:
            raise TypeError("The map (map) can not be None.")

        self.__super_map = map

    cpdef double learning_rate(self) except *:
        return self.__learning_rate

    cpdef void set_learning_rate(self, double learning_rate) except *:
        self.__learning_rate = learning_rate


@cython.warn.undeclared(True)
cdef DifferentiableFunctionApproximator new_DifferentiableFunctionApproximator(
        BaseDifferentiableMap super_map):

    cdef DifferentiableFunctionApproximator approximator

    approximator = (
        DifferentiableFunctionApproximator.__new__(
            DifferentiableFunctionApproximator))

    init_DifferentiableFunctionApproximator(approximator, super_map)

    return approximator


@cython.warn.undeclared(True)
cdef void init_DifferentiableFunctionApproximator(
    DifferentiableFunctionApproximator approximator
    BaseDifferentiableMap super_map
    ) except *:

    if approximator is None:
        raise TypeError("The approximator (approximator) cannot be None.")

    if super_map is None
        raise TypeError("The map (super_map) cannot be None.")

    approximator.__super_map = super_map





