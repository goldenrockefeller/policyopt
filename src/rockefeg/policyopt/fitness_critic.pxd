from .system cimport BaseSystem
from .function_approximation cimport BaseFunctionApproximator
from .buffer cimport ShuffleBuffer
from .value_target cimport BaseValueTargetSetter
from rockefeg.cyutil.typed_list cimport TypedList, BaseReadableTypedList

cdef class FitnessCriticSystem(BaseSystem):
    cdef BaseSystem __super_system
    cdef BaseFunctionApproximator __intermediate_critic
    cdef ShuffleBuffer __trajectory_buffer
    cdef ShuffleBuffer __critic_target_buffer
    cdef BaseValueTargetSetter __value_target_setter
    cdef __current_observation
    cdef __current_action
    cdef TypedList __current_trajectory
    cdef Py_ssize_t __n_trajectories_per_critic_update_batch
    cdef Py_ssize_t __critic_update_batch_size
    cdef Py_ssize_t __n_critic_update_batches_per_epoch
    cdef bint __redistributes_critic_target_updates

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

    cpdef BaseReadableTypedList current_trajectory(self)
    cpdef void _set_current_trajectory(
        self,
        TypedList trajectory
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

