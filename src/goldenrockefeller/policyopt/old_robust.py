# distutils: libraries = NP_RANDOM_LIB
# distutils: include_dirs = NP_INCLUDE
# distutils: library_dirs = NP_RANDOM_PATH

cimport cython
from rockefeg.cyutil.array cimport new_DoubleArray
from rockefeg.cyutil.typed_list cimport BaseReadableTypedList, is_sub_full_type
import numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport random_standard_normal, random_standard_cauchy
from libc.math cimport sqrt, log, exp

# Random numpy c-api from https://numpy.org/doc/stable/reference/random/extending.html
cdef const char *capsule_name = "BitGenerator"
cdef bitgen_t *rng
rnd_bitgen = PCG64()
capsule = rnd_bitgen.capsule
# Optional check that the capsule if from a BitGenerator
if not PyCapsule_IsValid(capsule, capsule_name):
    raise ValueError("Invalid pointer to anon_func_state")
# Cast the pointer
rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TargetEntry:
    cpdef copy(self, copy_obj = None):
        cdef TargetEntry new_entry

        if copy_obj is None:
            new_entry = TargetEntry.__new__(TargetEntry)
        else:
            new_entry = copy_obj

        new_entry.input = self.input
        new_entry.target = self.target

        return new_entry

@cython.warn.undeclared(True)
cdef TargetEntry new_TargetEntry():
    cdef TargetEntry entry

    entry = TargetEntry.__new__(TargetEntry)

    return entry


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseFunctionApproximator(BaseMap):

    cpdef void batch_update(self, entries) except *:
        raise NotImplementedError("Abstract method.")


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DifferentiableFunctionApproximator(BaseFunctionApproximator):
    def __init__(self, super_map):
        init_DifferentiableFunctionApproximator(self, super_map)

    cpdef copy(self, copy_obj = None):
        cdef DifferentiableFunctionApproximator new_approximator
        cdef BaseDifferentiableMap super_map

        if copy_obj is None:
            new_approximator = (
                DifferentiableFunctionApproximator.__new__(
                    DifferentiableFunctionApproximator))
        else:
            new_approximator = copy_obj

        super_map = self.__super_map
        new_approximator.__super_map = super_map.copy()
        new_approximator.__learning_rate = self.__learning_rate

        return new_approximator

    cpdef parameters(self):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return super_map.parameters()

    cpdef void set_parameters(self, parameters) except *:
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        super_map.set_parameters(parameters)

    cpdef Py_ssize_t n_parameters(self) except *:
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return super_map.n_parameters()

    cpdef eval(self, input):
        cdef BaseDifferentiableMap super_map

        super_map = self.super_map()

        return super_map.eval(input)

    cpdef void batch_update(self, entries) except *:
        cdef BaseReadableTypedList entries_cy = <BaseReadableTypedList?> entries
        cdef object input
        cdef DoubleArray parameters
        cdef DoubleArray error
        cdef DoubleArray error_grad_wrt_parameters
        cdef DoubleArray sum_error_grad_wrt_parameters
        cdef BaseDifferentiableMap map
        cdef TargetEntry entry
        cdef Py_ssize_t n_entries
        cdef Py_ssize_t n_entry_parameters
        cdef Py_ssize_t n_parameters
        cdef Py_ssize_t output_id
        cdef Py_ssize_t parameter_id
        cdef object entries_item_type
        cdef double learning_rate

        entries_item_type = entries_cy.item_type()

        if not is_sub_full_type(entries_item_type, TargetEntry):
            raise (
                TypeError(
                    "The entries list's item type "
                    "(entries.item_type() = {entries_item_type}) "
                    "must be a subtype of TargetEntry."
                    .format(**locals())))

        map = self.super_map()
        parameters = map.parameters()
        parameters = parameters.copy()
        n_parameters = map.n_parameters()
        sum_error_grad_wrt_parameters = new_DoubleArray(n_parameters)
        sum_error_grad_wrt_parameters.set_all_to(0.)
        n_entries = len(entries_cy)
        learning_rate = self.learning_rate()


        for entry in entries_cy:

            error = error_eval(map, entry)

            # Add gradient for this entry to the cumulative gradient.
            input = entry.input
            error_grad_wrt_parameters = map.grad_wrt_parameters(input, error)
            n_entry_parameters = len(error_grad_wrt_parameters)
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
                sum_error_grad_wrt_parameters.view[parameter_id] += (
                    error_grad_wrt_parameters.view[parameter_id]
                    * learning_rate
                    / n_entries)

        # Add gradient to parameters
        for parameter_id in range(n_parameters):
            parameters.view[parameter_id] += (
                sum_error_grad_wrt_parameters.view[parameter_id])

        # Update the maps parameters.
        map.set_parameters(parameters)

    cpdef super_map(self):
        return self.__super_map

    cpdef set_super_map(self, map):
        self.__super_map = <BaseDifferentiableMap?> map

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
    DifferentiableFunctionApproximator approximator,
    BaseDifferentiableMap super_map
    ) except *:

    if approximator is None:
        raise TypeError("The approximator (approximator) cannot be None.")

    if super_map is None:
        raise TypeError("The map (super_map) cannot be None.")

    approximator.__super_map = super_map
    approximator.__learning_rate = 1e-4

cdef class RobustFunctionApproximator(DifferentiableFunctionApproximator):
    def __init__(self, super_map):
        init_RobustFunctionApproximator(self, super_map)

    cpdef copy(self, copy_obj = None):
        cdef RobustFunctionApproximator new_approximator

        if copy_obj is None:
            new_approximator = (
                RobustFunctionApproximator.__new__(
                    RobustFunctionApproximator))
        else:
            new_approximator = copy_obj

        DifferentiableFunctionApproximator.copy(self, new_approximator)

        new_approximator.__line_search_shrink_rate = (
            self.__line_search_shrink_rate)

        new_approximator.__line_search_growth_rate = (
            self.__line_search_growth_rate)

        new_approximator.__line_search_step_size = (
            self.__line_search_step_size)

        new_approximator.__nes_size_growth_rate = (
            self.__nes_size_growth_rate)

        new_approximator.__nes_direction_growth_rate = (
            self.__nes_direction_growth_rate)

        new_approximator.__nes_size_shrink_rate = (
            self.__nes_size_shrink_rate)

        new_approximator.__nes_direction_shrink_rate = (
            self.__nes_direction_shrink_rate)

        new_approximator.__nes_factors = (
            self.__nes_factors.copy())

        return new_approximator

    cpdef void batch_update(self, entries) except *:
        cdef BaseReadableTypedList entries_cy = <BaseReadableTypedList?> entries
        cdef object input
        cdef DoubleArray nes_factors
        cdef DoubleArray parameters
        cdef DoubleArray new_parameters
        cdef DoubleArray line_search_parameters
        cdef DoubleArray nes_parameters
        cdef DoubleArray error
        cdef DoubleArray line_search_error
        cdef DoubleArray error_grad_wrt_parameters
        cdef DoubleArray line_search_grad
        cdef DoubleArray nes_mutation
        cdef DoubleArray nes_pre_mutation
        cdef DoubleArray nes_sqr_pre_mutation_normalized
        cdef BaseDifferentiableMap map
        cdef TargetEntry entry
        cdef Py_ssize_t n_entries
        cdef Py_ssize_t n_entry_parameters
        cdef Py_ssize_t n_parameters
        cdef Py_ssize_t output_id
        cdef Py_ssize_t parameter_id
        cdef object entries_item_type
        cdef double learning_rate
        cdef double line_search_shrink_rate
        cdef double line_search_growth_rate
        cdef double line_search_step_size
        cdef double nes_adjustment
        cdef double nes_size_growth_rate
        cdef double nes_direction_growth_rate
        cdef double nes_size_shrink_rate
        cdef double nes_direction_shrink_rate
        cdef double line_search_loss
        cdef double line_search_alignment
        cdef double nes_sqr_sum
        cdef double nes_exp_size_grow
        cdef double nes_exp_direction_grow
        cdef double nes_exp_size_shrink
        cdef double nes_exp_direction_shrink

        # "nes" means Natural Evolutionary Strategy

        entries_item_type = entries_cy.item_type()

        if not is_sub_full_type(entries_item_type, TargetEntry):
            raise (
                TypeError(
                    "The entries list's item type "
                    "(entries.item_type() = {entries_item_type}) "
                    "must be a subtype of TargetEntry."
                    .format(**locals())))

        map = self.super_map()
        parameters = map.parameters()
        parameters = parameters.copy()
        n_parameters = map.n_parameters()
        n_entries = len(entries_cy)
        learning_rate = self.learning_rate()
        nes_factors = self.nes_factors()
        line_search_shrink_rate = self.line_search_shrink_rate()
        line_search_growth_rate = self.line_search_growth_rate()
        line_search_step_size = self.line_search_step_size()

        nes_size_growth_rate = self.nes_size_growth_rate()
        nes_direction_growth_rate = self.nes_direction_growth_rate()
        nes_size_shrink_rate = self.nes_size_shrink_rate()
        nes_direction_shrink_rate = self.nes_direction_shrink_rate()


        for entry in entries_cy:

            input = entry.input
            output = map.eval(input)

            # Calculate the error for this entry.
            error = error_eval(map, entry)
            loss = dot_product(error, error)

            # # Perform natural evolution strategy (NES).
            # #
            # nes_parameters = parameters.copy()
            # nes_pre_mutation = nes_pre_mutation_eval(n_parameters)
            # #
            # nes_sqr_sum = dot_product(nes_pre_mutation, nes_pre_mutation)
            # #
            # nes_sqr_pre_mutation_normalized = nes_pre_mutation.copy()
            # for parameter_id in range(n_parameters):
            #     nes_sqr_pre_mutation_normalized.view[parameter_id] *= (
            #         nes_pre_mutation.view[parameter_id]
            #         / nes_sqr_sum)
            # #
            # nes_mutation = nes_pre_mutation.copy()
            # for parameter_id in range(n_parameters):
            #     nes_mutation.view[parameter_id] *= (
            #         nes_factors.view[parameter_id])
            # #
            # for parameter_id in range(n_parameters):
            #     nes_parameters.view[parameter_id] += (
            #         nes_mutation.view[parameter_id])
            # map.set_parameters(nes_parameters)
            # #
            # nes_error = error_eval(map, entry)
            # nes_loss = dot_product(nes_error, nes_error)
            # nes_adjustment = log(<double> n_parameters) # Approximate adjustment
            #
            # nes_exp_size_grow = exp(nes_size_growth_rate)
            # nes_exp_direction_grow = (
            #     exp(
            #         nes_direction_growth_rate
            #         * nes_adjustment ))
            #
            # nes_exp_size_shrink = exp(nes_size_shrink_rate)
            # nes_exp_direction_shrink = (
            #     exp(
            #         nes_direction_shrink_rate
            #         * nes_adjustment ))
            # #
            # if nes_loss < loss:
            #     for parameter_id in range(n_parameters):
            #         nes_factors.view[parameter_id] *= (
            #             nes_exp_size_grow
            #             *
            #             (
            #             (nes_exp_direction_grow - 1.)
            #             * nes_sqr_pre_mutation_normalized.view[parameter_id]
            #             + 1.
            #             )
            #         )
            # elif nes_loss == loss:
            #     pass
            # else:
            #     for parameter_id in range(n_parameters):
            #         nes_factors.view[parameter_id] /= (
            #             nes_exp_size_shrink
            #             *
            #             (
            #             (nes_exp_direction_shrink - 1.)
            #             * nes_sqr_pre_mutation_normalized.view[parameter_id]
            #             + 1.
            #             )
            #         )
            # # Restore parameters
            # map.set_parameters(parameters)

            # Get gradient parameters
            error_grad_wrt_parameters = map.grad_wrt_parameters(input, error)

            # Transform gradient with NES factors
            for parameter_id in range(n_parameters):
                error_grad_wrt_parameters.view[parameter_id] *= (
                    nes_factors.view[parameter_id]
                    *nes_factors.view[parameter_id])

            # Perform line search.
            #
            line_search_parameters = parameters.copy()
            #
            for parameter_id in range(n_parameters):
                line_search_parameters.view[parameter_id] += (
                    error_grad_wrt_parameters.view[parameter_id]
                    * line_search_step_size)
            #

            map.set_parameters(line_search_parameters)
            line_search_error = error_eval(map, entry)
            line_search_loss = dot_product(line_search_error, line_search_error)
            #
            line_search_grad = map.grad_wrt_parameters(input, error)
            for parameter_id in range(n_parameters):
                line_search_grad.view[parameter_id] *= (
                    nes_factors.view[parameter_id]
                    *nes_factors.view[parameter_id])
            #
            line_search_alignment = (
                dot_product(
                    line_search_grad,
                    error_grad_wrt_parameters))
            #
            if line_search_loss < loss and line_search_alignment >= 0.:
                line_search_step_size *= line_search_growth_rate
            elif line_search_loss == loss:
                pass
            else:
                line_search_step_size /= line_search_shrink_rate
            map.set_parameters(parameters)

            # Perform Gradient Descent on map
            for parameter_id in range(n_parameters):
                parameters.view[parameter_id] += (
                    line_search_step_size
                    * learning_rate
                    * error_grad_wrt_parameters.view[parameter_id])
            map.set_parameters(parameters)

        self.set_line_search_step_size(line_search_step_size)
        self.set_nes_factors(nes_factors)
        #print(np.asarray(nes_factors.view[...]))
        print(loss, line_search_loss)

    cpdef double line_search_shrink_rate(self) except *:
        return self.__line_search_shrink_rate

    cpdef void set_line_search_shrink_rate(
            self,
            double line_search_shrink_rate
            ) except *:
        self.__line_search_shrink_rate = line_search_shrink_rate

    cpdef double line_search_growth_rate(self) except *:
        return self.__line_search_growth_rate

    cpdef void set_line_search_growth_rate(
            self,
            double line_search_growth_rate
            ) except *:
        self.__line_search_growth_rate = line_search_growth_rate

    cpdef double line_search_step_size(self) except *:
        return self.__line_search_step_size

    cpdef void set_line_search_step_size(
            self,
            double line_search_step_size
            ) except *:
        self.__line_search_step_size = line_search_step_size

    cpdef double nes_size_growth_rate(self) except *:
        return self.__nes_size_growth_rate

    cpdef void set_nes_size_growth_rate(
            self,
            double nes_size_growth_rate
            ) except *:
        self.__nes_size_growth_rate = nes_size_growth_rate

    cpdef double nes_direction_growth_rate(self) except *:
        return self.__nes_direction_growth_rate

    cpdef void set_nes_direction_growth_rate(
            self,
            double nes_direction_growth_rate
            ) except *:
        self.__nes_direction_growth_rate

    cpdef double nes_size_shrink_rate(self) except *:
        return self.__nes_size_shrink_rate

    cpdef void set_nes_size_shrink_rate(
            self,
            double nes_size_shrink_rate
            ) except *:
        self.__nes_size_shrink_rate = nes_size_shrink_rate

    cpdef double nes_direction_shrink_rate(self) except *:
        return self.__nes_direction_shrink_rate

    cpdef void set_nes_direction_shrink_rate(
            self,
            double nes_direction_shrink_rate
            ) except *:
        self.__nes_direction_shrink_rate = nes_direction_shrink_rate

    cpdef double nes_factor(self, Py_ssize_t id) except *:
        return self.__nes_factors.view[id]

    cpdef DoubleArray nes_factors(self):
        return self.__nes_factors.copy()

    cpdef void set_nes_factor(self, Py_ssize_t id, double val) except *:
        self.__nes_factors.view[id] = val

    cpdef void set_nes_factors(self, DoubleArray factors) except *:
        cdef Py_ssize_t factors_len
        cdef Py_ssize_t self_factors_len

        factors_len = len(factors)
        self_factors_len = len(self.__nes_factors)

        if factors_len != self_factors_len:
            raise (
                ValueError(
                    "The number of new conditioning factors "
                    "(len(factors) = {len_factors}) must be equal to the "
                    "number of current conditioning factors "
                    "(len(self.nes_factors() = {self_factors_len})."
                    .format(**locals()) ))

        self.__nes_factors = factors.copy()

cdef RobustFunctionApproximator new_RobustFunctionApproximator(
        BaseDifferentiableMap super_map):
    cdef RobustFunctionApproximator approximator

    approximator = (
        RobustFunctionApproximator.__new__(
            RobustFunctionApproximator))

    init_RobustFunctionApproximator(approximator, super_map)

    return approximator


cdef void init_RobustFunctionApproximator(
        RobustFunctionApproximator approximator,
        BaseDifferentiableMap super_map
        ) except *:
    cdef Py_ssize_t n_parameters

    init_DifferentiableFunctionApproximator(approximator, super_map)

    approximator.__line_search_growth_rate = 1.0001
    approximator.__line_search_shrink_rate = 2. # Golden Ratio
    approximator.__line_search_step_size = 1.

    approximator.__nes_size_growth_rate = .0025 #0.25
    approximator.__nes_direction_growth_rate = 0.25#.25 #0.25
    approximator.__nes_size_shrink_rate = 0.00125
    approximator.__nes_direction_shrink_rate = 0.0#0.25

    n_parameters = super_map.n_parameters()
    approximator.__nes_factors = new_DoubleArray(n_parameters)
    approximator.__nes_factors.set_all_to(1.)


cdef DoubleArray error_eval(
        BaseDifferentiableMap map,
        TargetEntry entry):
    cdef object input
    cdef DoubleArray output
    cdef DoubleArray error
    cdef DoubleArray target
    cdef Py_ssize_t output_size
    cdef Py_ssize_t target_size

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

    return error

cdef double dot_product(DoubleArray arr_a, DoubleArray arr_b) except *:
    cdef double dot_product_res
    cdef Py_ssize_t len_a
    cdef Py_ssize_t len_b
    cdef Py_ssize_t id

    len_a = len(arr_a)
    len_b = len(arr_b)

    if len_a != len_b:
        raise (
            ValueError(
                "The size of the two arrays "
                "(len(arr_a) = {len_a}, len(arr_b) = {len_b}) must match."
                .format(**locals()) ))

    dot_product_res = 0.
    for id in range(len_a):
        dot_product_res += arr_a.view[id] * arr_b.view[id]

    return dot_product_res

cpdef DoubleArray nes_pre_mutation_eval(Py_ssize_t n_parameters):
    global rng

    cdef DoubleArray pre_mutation
    cdef Py_ssize_t id
    cdef double rnd_num
    cdef double sqr_norm
    cdef double inv_norm

    pre_mutation = new_DoubleArray(n_parameters)

    sqr_norm = 0.
    for id in range(n_parameters):
        rnd_num = random_standard_normal(rng)
        sqr_norm += rnd_num * rnd_num
        pre_mutation.view[id] = rnd_num

    if sqr_norm == 0.:
        pre_mutation.set_all_to(0.)
        pre_mutation.view[0] = 1.
        return pre_mutation

    inv_norm = 1./ sqrt(sqr_norm)
    rnd_num = random_standard_cauchy(rng)

    for id in range(n_parameters):
        pre_mutation.view[id] *= inv_norm * rnd_num

    return pre_mutation









