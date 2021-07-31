cimport cython
import cython

# from .value_target cimport new_TotalRewardTargetSetter
from .buffer cimport new_ShuffleBuffer
from .experience cimport ExperienceDatum, new_ExperienceDatum
from .function_approximation cimport TargetEntry
from goldenrockefeller.cyutil.array cimport DoubleArray

from libc.math cimport isfinite

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class FitnessCriticSystem(BaseSystem):
    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)

    cpdef FitnessCriticSystem copy(self, copy_obj = None):
        cdef FitnessCriticSystem new_system

        if copy_obj is None:
            new_system = FitnessCriticSystem.__new__(FitnessCriticSystem)
        else:
            new_system = copy_obj

        new_system._super_system = self._super_system.copy()
        new_system._intermediate_critic = self._intermediate_critic.copy()
        new_system._trajectory_buffer = self._trajectory_buffer.copy()
        new_system._critic_target_buffer = self._critic_target_buffer.copy()

        # Shallow copy observation and action.
        new_system._current_observation = self._current_observation
        new_system._current_action = self._current_action


        new_system._current_trajectory = (
            self._current_trajectory.shallow_copy() )


        new_system._n_trajectories_per_critic_update_batch = (
            self._n_trajectories_per_critic_update_batch)

        new_system._critic_update_batch_size = (
            self._critic_update_batch_size)

        new_system._n_critic_update_batches_per_epoch = (
            self._n_critic_update_batches_per_epoch)

        new_system._redistributes_critic_target_updates = (
            self._redistributes_critic_target_updates)

        # new_system._value_target_setter = self._value_target_setter.copy()

        return new_system

    cpdef bint is_done_training(self) except *:
        return self.super_system().is_done_training()

    @cython.locals(trajectory = list, target_entries = list)
    cpdef void prep_for_epoch(self) except *:
        # cdef Py_ssize_t batch_id
        # cdef Py_ssize_t trajectory_id
        # cdef Py_ssize_t target_id
        # cdef Py_ssize_t n_trajectories_per_batch
        # cdef Py_ssize_t n_batches
        # cdef Py_ssize_t batch_size
        # cdef ShuffleBuffer trajectory_buffer
        # cdef ShuffleBuffer critic_target_buffer
        # # cdef BaseValueTargetSetter value_target_setter
        # trajectory: List[ExperienceDatum]
        # target_entries: List[TargetEntry]
        # cdef TargetEntry target_entry
        # cdef BaseFunctionApproximator intermediate_critic
        # cdef ExperienceDatum experience
        #
        # n_batches = (
        #     self.n_critic_update_batches_per_epoch())
        #
        # n_trajectories_per_batch = (
        #     self.n_trajectories_per_critic_update_batch())
        #
        # trajectory_buffer = self.trajectory_buffer()
        # critic_target_buffer = self.critic_target_buffer()
        # batch_size = self.critic_update_batch_size()
        # intermediate_critic = self.intermediate_critic()

        # value_target_setter = self.value_target_setter()
        raise NotImplementedError("This function needs a redo")

        # if not trajectory_buffer.is_empty():
        #     for batch_id in range(n_batches):
        #         for trajectory_id in range(n_trajectories_per_batch):
        #             trajectory = trajectory_buffer.next_shuffled_datum()
        #             target_entries = (
        #                 value_target_setter.value_target_entries(
        #                     trajectory))
        #             for target_entry in target_entries:
        #                 critic_target_buffer.add_staged_datum(target_entry)
        #
        #
        #         target_entries = [None] * batch_size
        #
        #         for target_id in range(batch_size):
        #             target_entry = critic_target_buffer.next_shuffled_datum()
        #             target_entries[target_id] = target_entry
        #
        #         intermediate_critic.batch_update(target_entries)
        #     raise NotImplementedError("This function needs a redo")
        #     print(intermediate_critic.eval(trajectory[0]).view[0])
        #
        # self.super_system().prep_for_epoch()

    cpdef bint is_ready_for_evaluation(self) except *:
        cdef BaseSystem system

        system = self.super_system()

        return system.is_ready_for_evaluation()

    cpdef action(self, observation):
        cdef object action

        action = self.super_system().action(observation)

        self._set_current_observation(observation)
        self._set_current_action(action)

        return action

    #step_wise feedback
    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        cdef DoubleArray intermediate_eval

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        self.current_trajectory().append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]


        if not isfinite(new_feedback):
            raise RuntimeError("Something went wrong: feedback is not finite.")

        self.super_system().receive_feedback(new_feedback)


    cpdef void update_policy(self) except *:
        self.super_system().update_policy()
        self.trajectory_buffer().add_staged_datum(self.current_trajectory())
        self._set_current_trajectory([])

    cpdef void receive_score(self, double score) except *:
        self.super_system().receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        self.super_system().output_final_log(log_dirname, datetime_str)

    cpdef BaseSystem super_system(self):
        return self._super_system

    cpdef void set_super_system(self, BaseSystem super_system) except *:
        self._super_system = super_system

    cpdef BaseFunctionApproximator intermediate_critic(self):
        return self._intermediate_critic

    cpdef void set_intermediate_critic(
            self,
            BaseFunctionApproximator intermediate_critic
            ) except *:
        self._intermediate_critic = intermediate_critic

    cpdef ShuffleBuffer trajectory_buffer(self):
        return self._trajectory_buffer

    cpdef void _set_trajectory_buffer(self, ShuffleBuffer buffer) except *:

        cdef object buffer_item_type

        buffer_item_type = buffer.item_type()

        self._trajectory_buffer = buffer

    # cpdef ShuffleBuffer critic_target_buffer(self):
    #     return self._critic_target_buffer
    #
    # cpdef void _set_critic_target_buffer(self, ShuffleBuffer buffer) except *:
    #     self._critic_target_buffer = buffer

    # cpdef BaseValueTargetSetter value_target_setter(self):
    #     return self._value_target_setter
    #
    # cpdef void set_value_target_setter(
    #         self,
    #         BaseValueTargetSetter value_target_setter
    #         ) except *:
    #     self._value_target_setter = value_target_setter

    cpdef current_observation(self):
        return self._current_observation

    cpdef void _set_current_observation(self, observation) except *:
        self._current_observation = observation

    cpdef current_action(self):
         return self._current_action

    cpdef void _set_current_action(self, action) except *:
            self._current_action = action

    cpdef list current_trajectory(self):
        # type: (...) -> Sequence[ExperienceDatum]
        return self._current_trajectory


    @cython.locals(trajectory = list)
    cpdef void _set_current_trajectory(
            self,
            trajectory: List[ExperienceDatum]
            ) except *:
        self._current_trajectory = trajectory

    # cpdef Py_ssize_t n_trajectories_per_critic_update_batch(self) except *:
    #     return self._n_trajectories_per_critic_update_batch

    # cpdef void set_n_trajectories_per_critic_update_batch(
    #         self,
    #         Py_ssize_t n_trajectories
    #         ) except *:
    #     self._n_trajectories_per_critic_update_batch =  n_trajectories
    #
    # cpdef Py_ssize_t n_trajectories_per_critic_update_batch(self) except *:
    #     return self._n_trajectories_per_critic_update_batch
    #
    # cpdef void set_n_trajectories_per_critic_update_batch(
    #         self,
    #         Py_ssize_t n_trajectories
    #         ) except *:
    #     self._n_trajectories_per_critic_update_batch =  n_trajectories
    #
    #
    # cpdef Py_ssize_t critic_update_batch_size(self) except *:
    #     return self._critic_update_batch_size
    #
    # cpdef void set_critic_update_batch_size(self, Py_ssize_t size) except *:
    #     self._critic_update_batch_size = size
    #
    # cpdef Py_ssize_t n_critic_update_batches_per_epoch(self) except *:
    #     return self._n_critic_update_batches_per_epoch
    #
    # cpdef void set_n_critic_update_batches_per_epoch(
    #         self,
    #         Py_ssize_t n_batches
    #         ) except *:
    #     self._n_critic_update_batches_per_epoch = n_batches
    #
    # cpdef bint redistributes_critic_target_updates(self) except *:
    #     # Todo: implement this
    #     return self._redistributes_critic_target_updates
    #
    # cpdef void set_redistributes_critic_target_updates(
    #         self,
    #         bint redistributes_updates
    #         ) except *:
    #     self._redistributes_critic_target_updates = redistributes_updates
    #



@cython.warn.undeclared(True)
cdef FitnessCriticSystem new_FitnessCriticSystem(
        BaseSystem super_system,
        BaseFunctionApproximator intermediate_critic):
    cdef FitnessCriticSystem system

    system = FitnessCriticSystem.__new__(FitnessCriticSystem)
    init_FitnessCriticSystem(system, super_system, intermediate_critic)

    return system

@cython.warn.undeclared(True)
cdef void init_FitnessCriticSystem(
        FitnessCriticSystem system,
        BaseSystem super_system,
        BaseFunctionApproximator intermediate_critic
        ) except *:
    if system is None:
        raise TypeError("The fitness critic system (system) cannot be None.")

    if super_system is None:
        raise TypeError("The super system (system) cannot be None.")

    if intermediate_critic is None:
        raise (
            TypeError(
                "The intermediate critic (intermediate_critic) cannot be "
                "None." ))

    system._super_system = super_system
    system._intermediate_critic = intermediate_critic
    system._trajectory_buffer = new_ShuffleBuffer()
    # system._critic_target_buffer = new_ShuffleBuffer()
    system._current_observation = None
    system._current_action = None
    system._current_trajectory = []
    # system._critic_update_batch_size = 1
    # system._n_trajectories_per_critic_update_batch = 1
    # system._n_critic_update_batches_per_epoch = 1
    # system._redistributes_critic_target_updates = False
    # system._value_target_setter = new_TotalRewardTargetSetter()

