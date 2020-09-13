cimport cython

from .value_target cimport new_TotalRewardTargetSetter
from .value_target cimport BaseValueTargetSetter
from .buffer cimport ShuffleBuffer, new_ShuffleBuffer
from .experience cimport ExperienceDatum, new_ExperienceDatum
from .function_approximation cimport TargetEntry
from rockefeg.cyutil.typed_list cimport TypedList, new_TypedList
from rockefeg.cyutil.typed_list cimport is_sub_full_type

from libc.math cimport isfinite

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class FitnessCriticSystem(BaseSystem):
    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)

    cpdef copy(self, copy_obj = None):
        cdef FitnessCriticSystem new_system
        cdef BaseSystem system
        cdef BaseFunctionApproximator intermediate_critic
        cdef ShuffleBuffer buffer
        cdef TypedList current_trajectory
        cdef BaseValueTargetSetter value_target_setter

        if copy_obj is None:
            new_system = FitnessCriticSystem.__new__(FitnessCriticSystem)
        else:
            new_system = copy_obj

        system = self.__super_system
        new_system.__super_system = system.copy()

        intermediate_critic = self.__intermediate_critic
        new_system.__intermediate_critic = intermediate_critic.copy()

        buffer = self.__trajectory_buffer
        new_system.__trajectory_buffer = buffer.copy()

        buffer = self.__critic_target_buffer
        new_system.__critic_target_buffer = buffer.copy()

        new_system.__current_state = self.__current_state

        new_system.__current_action = self.__current_action

        current_trajectory = self.__current_trajectory
        new_system.__current_trajectory = current_trajectory.shallow_copy()


        new_system.__n_trajectories_per_critic_update_batch = (
            self.__n_trajectories_per_critic_update_batch)

        new_system.__critic_update_batch_size = (
            self.__critic_update_batch_size)

        new_system.__n_critic_update_batches_per_epoch = (
            self.__n_critic_update_batches_per_epoch)

        new_system.__redistributes_critic_target_updates = (
            self.__redistributes_critic_target_updates)

        value_target_setter = self.__value_target_setter
        new_system.__value_target_setter = value_target_setter.copy()


        return new_system

    cpdef bint is_done_training(self) except *:
        cdef BaseSystem system

        system = self.super_system()

        return system.is_done_training()

    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t batch_id
        cdef Py_ssize_t trajectory_id
        cdef Py_ssize_t target_id
        cdef Py_ssize_t n_trajectories_per_batch
        cdef Py_ssize_t n_batches
        cdef Py_ssize_t batch_size
        cdef ShuffleBuffer trajectory_buffer
        cdef ShuffleBuffer critic_target_buffer
        cdef BaseValueTargetSetter value_target_setter
        cdef TypedList trajectory
        cdef TypedList target_entries
        cdef TargetEntry target_entry
        cdef list target_entry_list
        cdef BaseFunctionApproximator intermediate_critic
        cdef ExperienceDatum experience
        cdef BaseSystem system

        n_batches = (
            self.n_critic_update_batches_per_epoch())

        n_trajectories_per_batch = (
            self.n_trajectories_per_critic_update_batch())

        trajectory_buffer = self.trajectory_buffer()
        critic_target_buffer = self.critic_target_buffer()
        batch_size = self.critic_update_batch_size()
        intermediate_critic = self.intermediate_critic()

        value_target_setter = self.value_target_setter()

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()
                    target_entries = (
                        value_target_setter.value_target_entries(
                            trajectory))
                    for target_entry in target_entries:
                        critic_target_buffer.add_staged_datum(target_entry)


                target_entry_list = [None] * batch_size

                for target_id in range(batch_size):
                    target_entry = critic_target_buffer.next_shuffled_datum()
                    target_entry_list[target_id] = target_entry

                target_entries = new_TypedList(TargetEntry)
                target_entries.set_items(target_entry_list)
                intermediate_critic.batch_update(target_entries)

            print(intermediate_critic.eval(trajectory.item(0)).view[0])

        system = self.super_system()
        system.prep_for_epoch()

    cpdef bint is_ready_for_evaluation(self) except *:
        cdef BaseSystem system

        system = self.super_system()

        return system.is_ready_for_evaluation()

    cpdef action(self, observation):
        cdef object action
        cdef BaseSystem system

        system = self.super_system()

        action = system.action(observation)

        self._set_current_state(observation)
        self._set_current_action(action)

        return action

    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        cdef TypedList current_trajectory
        cdef BaseSystem system

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.state = self.current_state()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        new_feedback = intermediate_critic.eval(experience).view[0]

        if not isfinite(new_feedback):
            raise RuntimeError("Something went wrong: feedback is not finite.")

        system.receive_feedback(new_feedback + experience.reward)


    cpdef void update_policy(self) except *:
        cdef ShuffleBuffer trajectory_buffer
        cdef BaseSystem system
        cdef TypedList current_trajectory

        system = self.super_system()

        trajectory_buffer = self.trajectory_buffer()

        system.update_policy()
        current_trajectory = self.current_trajectory()
        trajectory_buffer.add_staged_datum(current_trajectory)
        self.set_current_trajectory(new_TypedList(ExperienceDatum))



    cpdef void receive_score(self, score) except *:
        cdef BaseSystem system

        system = self.super_system()

        system.receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        cdef BaseSystem system

        system = self.super_system()

        system.output_final_log(log_dirname, datetime_str)

    cpdef super_system(self):
        return self.__super_system

    cpdef void set_super_system(self, super_system) except *:
        self.__super_system = <BaseSystem?>super_system

    cpdef intermediate_critic(self):
        return self.__intermediate_critic

    cpdef void set_intermediate_critic(self, intermediate_critic) except *:
        self.__intermediate_critic = (
            <BaseFunctionApproximator?>intermediate_critic)

    cpdef trajectory_buffer(self):
        return self.__trajectory_buffer

    cpdef void _set_trajectory_buffer(self, buffer) except *:
        cdef ShuffleBuffer setting_buffer = <ShuffleBuffer?> buffer
        cdef object buffer_item_type

        buffer_item_type = setting_buffer.item_type()

        if not is_sub_full_type(buffer_item_type, (TypedList, ExperienceDatum)):
            raise (
                TypeError(
                    "The trajectory buffer's item type "
                    "(buffer.item_type() = {buffer_item_type}) "
                    "must be a subtype of a subtype of (TypedList, ExperienceDatum)."
                    .format(**locals())))

        self.__trajectory_buffer = setting_buffer

    cpdef critic_target_buffer(self):
        return self.__critic_target_buffer

    cpdef void _set_critic_target_buffer(self, buffer) except *:
        cdef ShuffleBuffer setting_buffer = <ShuffleBuffer?> buffer
        cdef object buffer_item_type

        buffer_item_type = setting_buffer.item_type()

        if not is_sub_full_type(buffer_item_type, TargetEntry):
            raise (
                TypeError(
                    "The critic target entry buffer's item type "
                    "(buffer.item_type() = {buffer_item_type}) "
                    "must be a subtype of a subtype of TargetEntry."
                    .format(**locals())))

        self.__critic_target_buffer = setting_buffer

    cpdef value_target_setter(self):
        return self.__value_target_setter

    cpdef void set_value_target_setter(self, value_target_setter) except *:
        self.__value_target_setter = <BaseValueTargetSetter?>value_target_setter

    cpdef current_state(self):
        return self.__current_state

    cpdef void _set_current_state(self, state) except *:
        self.__current_state = state

    cpdef current_action(self):
         return self.__current_action

    cpdef void _set_current_action(self, action) except *:
            self.__current_action = action

    cpdef current_trajectory(self):
        return self.__current_trajectory

    cpdef void set_current_trajectory(self, trajectory) except *:
        cdef TypedList setting_trajectory = <TypedList?> trajectory
        cdef object trajectory_item_type

        trajectory_item_type = setting_trajectory.item_type()

        if trajectory_item_type is not ExperienceDatum:
            raise (
                TypeError(
                    "The trjectory's item type "
                    "(trajectory.item_type() = {trajectory_item_type}) "
                    "must be a subtype of ExperienceDatum."
                    .format(**locals())))

        self.__current_trajectory = setting_trajectory

    cpdef Py_ssize_t n_trajectories_per_critic_update_batch(self) except *:
        return self.__n_trajectories_per_critic_update_batch

    cpdef void set_n_trajectories_per_critic_update_batch(
            self,
            n_trajectories
            ) except *:
        self.__n_trajectories_per_critic_update_batch =  n_trajectories


    cpdef Py_ssize_t critic_update_batch_size(self) except *:
        return self.__critic_update_batch_size

    cpdef void set_critic_update_batch_size(self, Py_ssize_t size) except *:
        self.__critic_update_batch_size = size

    cpdef Py_ssize_t n_critic_update_batches_per_epoch(self) except *:
        return self.__n_critic_update_batches_per_epoch

    cpdef void set_n_critic_update_batches_per_epoch(
            self,
            Py_ssize_t n_batches
            ) except *:
        self.__n_critic_update_batches_per_epoch = n_batches

    cpdef bint redistributes_critic_target_updates(self) except *:
        # Todo: implement this
        return self.__redistributes_critic_target_updates

    cpdef void set_redistributes_critic_target_updates(
            self,
            bint redistributes_updates
            ) except *:
        self.__redistributes_critic_target_updates = redistributes_updates




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

    system.__super_system = super_system
    system.__intermediate_critic = intermediate_critic
    system.__trajectory_buffer = new_ShuffleBuffer((TypedList, ExperienceDatum))
    system.__critic_target_buffer = new_ShuffleBuffer(TargetEntry)
    system.__current_state = None
    system.__current_action = None
    system.__current_trajectory = new_TypedList(ExperienceDatum)
    system.__critic_update_batch_size = 1
    system.__n_trajectories_per_critic_update_batch = 1
    system.__n_critic_update_batches_per_epoch = 1
    system.__redistributes_critic_target_updates = False
    system.__value_target_setter = new_TotalRewardTargetSetter()

