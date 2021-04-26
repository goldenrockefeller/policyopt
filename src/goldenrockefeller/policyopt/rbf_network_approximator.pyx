# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/rbf_network.cpp cpp_core/rbf_network_approximator.cpp
cimport cython
from libcpp.memory cimport shared_ptr, make_shared
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from .libcpp_valarray cimport valarray
from libcpp.vector cimport vector
from .cpp_experience cimport ExperienceDatum as CppExperienceDatum
from .experience cimport ExperienceDatum

ctypedef CppExperienceDatum[valarray[double], valarray[double], double] RbfExperienceDatum

cdef valarray[double] valarray_from_DoubleArray(DoubleArray arr) except *:
    cdef valarray[double] new_arr
    cdef Py_ssize_t id

    new_arr.resize(len(arr))

    for id in range(len(arr)):
        new_arr[<size_t>id] = arr.view[id]

    return new_arr

cdef DoubleArray DoubleArray_from_valarray(valarray[double] arr):
    cdef DoubleArray new_arr
    cdef Py_ssize_t id

    new_arr = new_DoubleArray(<Py_ssize_t>arr.size())

    for id in range(len(new_arr)):
        new_arr.view[id] = arr[<size_t>id]

    return new_arr

cdef class RbfNetworkApproximator(BaseFunctionApproximator):
    def __init__(self, RbfNetwork network):

        cdef Py_ssize_t n_out_dims

        n_out_dims = network.n_out_dims()

        if network.n_out_dims() != 1:
            raise (
                ValueError(
                    "The number of output dimensions (n_out_dims = {n_out_dims}) "
                    "must be equal to 1."
                    .format(**locals())))

        self.network = network

        self.core = make_shared[CppRbfNetworkApproximator](network.core)

    cpdef double eval_offset(self) except *:
        return self.core.get().eval_offset

    cpdef void set_eval_offset(self, double eval_offset) except *:
        self.core.get().eval_offset = eval_offset


    cpdef double info_retention_factor(self) except *:
        return self.core.get().info_retention_factor

    cpdef void set_info_retention_factor(self, double info_retention_factor) except *:
        self.core.get().info_retention_factor = info_retention_factor


    cpdef DoubleArray eval(self, input):

        return (
            DoubleArray_from_valarray(
                self.core.get().eval(
                    valarray_from_DoubleArray(
                        concatenate_observation_action(
                            input )))))

    @cython.locals(trajectory = list)
    cpdef void update(self, trajectory: Sequence[ExperienceDatum]) except *:
        cdef vector[RbfExperienceDatum] rbf_trajectory
        cdef Py_ssize_t experience_id
        cdef ExperienceDatum experience

        rbf_trajectory.resize(len(trajectory))

        for experience_id in range(0, len(trajectory)):
            experience = trajectory[experience_id]

            rbf_trajectory[experience_id].observation = (
                valarray_from_DoubleArray(
                    experience.observation ))

            rbf_trajectory[experience_id].action = (
                valarray_from_DoubleArray(
                    experience.action ))

            rbf_trajectory[experience_id].feedback = experience.reward

        self.core.get().update(rbf_trajectory)


    @cython.locals(entries = list)
    cpdef void batch_update(self, entries: Sequence[TargetEntry]) except *:
        cdef TargetEntry target_entry

        for target_entry in entries:
            self.update(target_entry.input)


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
