from .system cimport BaseSystem
from .function_approximation cimport BaseFunctionApproximator
from .buffer cimport ShuffleBuffer
from .value_target cimport BaseValueTargetSetter

import cython
from typing import List, Sequence

cdef class FitnessCriticSystem(BaseSystem):
    cdef BaseSystem _super_system
    cdef BaseFunctionApproximator _intermediate_critic
    cdef ShuffleBuffer _trajectory_buffer
    cdef ShuffleBuffer _critic_target_buffer
    cdef BaseValueTargetSetter _value_target_setter
    cdef _current_observation
    cdef _current_action
    cdef list _current_trajectory
    cdef Py_ssize_t _n_trajectories_per_critic_update_batch
    cdef Py_ssize_t _critic_update_batch_size
    cdef Py_ssize_t _n_critic_update_batches_per_epoch
    cdef bint _redistributes_critic_target_updates

    cpdef FitnessCriticSystem copy(self, copy_obj = ?)

    cpdef BaseSystem super_system(self)
    cpdef void set_super_system(self, BaseSystem super_system) except *

    cpdef BaseFunctionApproximator intermediate_critic(self)
    cpdef void set_intermediate_critic(
        self,
        BaseFunctionApproximator intermediate_critic
        ) except *

    cpdef ShuffleBuffer trajectory_buffer(self)
    cpdef void _set_trajectory_buffer(self, ShuffleBuffer buffer) except *

    cpdef ShuffleBuffer critic_target_buffer(self)
    cpdef void _set_critic_target_buffer(self, ShuffleBuffer buffer) except *

    cpdef BaseValueTargetSetter value_target_setter(self)
    cpdef void set_value_target_setter(self, BaseValueTargetSetter value_target_setter) except *

    cpdef current_observation(self)
    cpdef void _set_current_observation(self, observation) except *

    cpdef current_action(self)
    cpdef void _set_current_action(self, action) except *

    cpdef list current_trajectory(self)
    # type: (...) -> Sequence[ExperienceDatum]

    @cython.locals(trajectory = list)
    cpdef void _set_current_trajectory(
        self,
        trajectory: List[ExperienceDatum]
        ) except *

    cpdef Py_ssize_t n_trajectories_per_critic_update_batch(self) except *
    cpdef void set_n_trajectories_per_critic_update_batch(
        self,
        Py_ssize_t n_trajectories
        ) except *

    cpdef Py_ssize_t critic_update_batch_size(self) except *
    cpdef void set_critic_update_batch_size(self, Py_ssize_t size) except *

    cpdef Py_ssize_t n_critic_update_batches_per_epoch(self) except *
    cpdef void set_n_critic_update_batches_per_epoch(
        self,
        Py_ssize_t n_batches
        ) except *

    cpdef bint redistributes_critic_target_updates(self) except *
    cpdef void set_redistributes_critic_target_updates(
        self,
        bint redistributes_updates
        ) except *


cdef FitnessCriticSystem new_FitnessCriticSystem(
    BaseSystem super_system,
    BaseFunctionApproximator intermediate_critic)

cdef void init_FitnessCriticSystem(
    FitnessCriticSystem system,
    BaseSystem super_system,
    BaseFunctionApproximator intermediate_critic
    ) except *

