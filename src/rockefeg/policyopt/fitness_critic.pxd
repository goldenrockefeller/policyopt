from .system cimport BaseSystem
from .function_approximation cimport BaseFunctionApproximator

cdef class FitnessCriticSystem(BaseSystem):
    cdef __super_system
    cdef __intermediate_critic
    cdef __trajectory_buffer
    cdef __critic_target_buffer
    cdef __value_target_setter
    cdef __current_state
    cdef __current_action
    cdef __current_trajectory
    cdef Py_ssize_t __n_trajectories_per_critic_update_batch
    cdef Py_ssize_t __critic_update_batch_size
    cdef Py_ssize_t __n_critic_update_batches_per_epoch
    cdef bint __redistributes_critic_target_updates

    cpdef copy(self, copy_obj = ?)

    cpdef super_system(self)
    cpdef void set_super_system(self, super_system) except *

    cpdef intermediate_critic(self)
    cpdef void set_intermediate_critic(self, intermediate_critic) except *

    cpdef trajectory_buffer(self)
    cpdef void _set_trajectory_buffer(self, buffer) except *

    cpdef critic_target_buffer(self)
    cpdef void _set_critic_target_buffer(self, buffer) except *

    cpdef value_target_setter(self)
    cpdef void set_value_target_setter(self, value_target_setter) except *

    cpdef current_state(self)
    cpdef void _set_current_state(self, state) except *

    cpdef current_action(self)
    cpdef void _set_current_action(self, action) except *

    cpdef current_trajectory(self)
    cpdef void set_current_trajectory(self, trajectory) except *

    cpdef Py_ssize_t n_trajectories_per_critic_update_batch(self) except *
    cpdef void set_n_trajectories_per_critic_update_batch(
        self,
        n_trajectories
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

