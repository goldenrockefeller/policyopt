cimport cython
from .system cimport BaseSystem



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class MultiagentSystem(BaseSystem):
    def __init__(self):
        init_MultiagentSystem(self)

    cpdef copy(self, copy_obj = None):
        cdef MultiagentSystem new_system
        cdef BaseSystem system
        cdef Py_ssize_t agent_id

        if copy_obj is None:
            new_system = MultiagentSystem.__new__(MultiagentSystem)
        else:
            new_system = copy_obj

        new_system.__agent_systems = (
            self.agent_systems_deep_copy())

        return new_system

    cpdef bint is_done_training(self) except *:
        cdef bint is_done_training
        cdef BaseSystem system

        is_done_training = True

        for system in self.__agent_systems:
            is_done_training = is_done_training and system.is_done_training()

        return  is_done_training

    cpdef void prep_for_epoch(self) except *:
        cdef BaseSystem system

        for system in self.__agent_systems:
            system.prep_for_epoch()

    cpdef bint is_ready_for_evaluation(self) except *:
        cdef BaseSystem system
        cdef bint is_ready_for_evaluation

        is_ready_for_evaluation = True

        for system in self.__agent_systems:
            is_ready_for_evaluation = (
                is_ready_for_evaluation
                and system.is_ready_for_evaluation())

        return  is_ready_for_evaluation


    cpdef action(self, observation):
        cdef list joint_action
        cdef list joint_observation
        cdef BaseSystem system
        cdef Py_ssize_t agent_id

        joint_observation = observation

        joint_action = [None] * self.n_agents()

        for agent_id in range(self.n_agents()):
            system = self.__agent_systems[agent_id]
            joint_action[agent_id] = system.action(joint_observation[agent_id])

        return joint_action


    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef list cy_feedback
        cdef BaseSystem system
        cdef Py_ssize_t agent_id

        cy_feedback = feedback

        for agent_id in range(self.n_agents()):
            system = self.__agent_systems[agent_id]
            system.receive_feedback(cy_feedback[agent_id])

    cpdef void update_policy(self) except *:
        cdef BaseSystem system

        for system in self.__agent_systems:
            system.update_policy()

    cpdef void receive_score(self, score) except *:
        cdef BaseSystem system

        for system in self.__agent_systems:
            system.receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        cdef BaseSystem system

        for system in self.__agent_systems:
            system.output_final_log(log_dirname, datetime_str)

    cpdef Py_ssize_t n_agents(self) except *:
          return len(self.__agent_systems)

    cpdef void append_agent_system(self, agent_system) except *:
        self.__agent_systems.append(<BaseSystem?>agent_system)

    cpdef pop_agent_system(self, Py_ssize_t index):
        return self.__agent_systems.pop(index)

    cpdef void insert_agent_system(
            self,
            Py_ssize_t index,
            agent_system
            ) except *:
        self.__agent_systems.insert(index, <BaseSystem?>agent_system)

    cpdef agent_system(self, Py_ssize_t index):
        return self.__agent_systems[index]

    cpdef void set_agent_system(self, Py_ssize_t index, agent_system) except *:
        self.__agent_systems[index] = <BaseSystem?>agent_system

    cpdef list _agent_systems(self):
        return self.__agent_systems

    cpdef list agent_systems_shallow_copy(self):
        cdef list agent_systems_copy
        cdef Py_ssize_t agent_system_id

        agent_systems_copy = [None] * len(self.__agent_systems)

        for agent_system_id in range(len(self.__agent_systems)):
            agent_systems_copy[agent_system_id] = (
                self.__agent_systems[agent_system_id])

        return agent_systems_copy

    cpdef list agent_systems_deep_copy(self):
        cdef list agent_systems_copy
        cdef Py_ssize_t agent_system_id
        cdef BaseSystem agent_system

        agent_systems_copy = [None] * len(self.__agent_systems)

        for agent_system_id in range(len(self.__agent_systems)):
            agent_system = self.__agent_systems[agent_system_id]
            agent_systems_copy[agent_system_id] = agent_system.copy()

        return agent_systems_copy

    cpdef void set_agent_systems(self, list agent_systems) except *:
        cdef Py_ssize_t agent_system_id
        cdef BaseSystem agent_system

        for agent_system_id in range(len(agent_systems)):
            agent_system = agent_systems[agent_system_id]
            if not isinstance(agent_system, BaseSystem):
                raise (
                    TypeError(
                        "All objects in (agent_systems) must be instances of "
                        "BaseSystem. (type(agent_systems[{agent_system_id}]) = "
                        "{agent_system.__class__})."
                        .format(**locals()) ))

        self.__agent_systems = agent_systems

@cython.warn.undeclared(True)
cdef MultiagentSystem new_MultiagentSystem():
    cdef MultiagentSystem system

    system = MultiagentSystem.__new__(MultiagentSystem)
    init_MultiagentSystem(system)

    return system

@cython.warn.undeclared(True)
cdef void init_MultiagentSystem(MultiagentSystem system) except *:
    if system is None:
        raise TypeError("The system (system) cannot be None.")

    system.__agent_systems = []