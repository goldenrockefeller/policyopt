cimport cython
from .system cimport BaseSystem
from rockefeg.cyutil.typed_list cimport BaseReadableTypedList,  new_TypedList
from rockefeg.cyutil.typed_list cimport is_sub_full_type


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class MultiagentSystem(BaseSystem):
    def __init__(self):
        init_MultiagentSystem(self)

    cpdef MultiagentSystem copy(self, copy_obj = None):
        cdef MultiagentSystem new_system
        cdef BaseSystem agent_system
        cdef list new_agent_systems_list
        cdef Py_ssize_t agent_id

        if copy_obj is None:
            new_system = MultiagentSystem.__new__(MultiagentSystem)
        else:
            new_system = copy_obj


        new_agent_systems_list  = [None] * len(self.__agent_systems)
        for agent_id in range(len(self.__agent_systems)):
            agent_system = self.__agent_systems.item(agent_id)
            new_agent_systems_list[new_agent_systems_list] = agent_system.copy()
        new_system.__agent_systems = new_TypedList(BaseSystem)
        new_system.__agent_systems.set_items(new_agent_systems_list)

        return new_system

    cpdef bint is_done_training(self) except *:
        cdef bint is_done_training
        cdef BaseSystem system

        is_done_training = True

        for system in self.agent_systems():
            is_done_training = is_done_training and system.is_done_training()

        return  is_done_training

    cpdef void prep_for_epoch(self) except *:
        cdef BaseSystem system

        for system in self.agent_systems():
            system.prep_for_epoch()

    cpdef bint is_ready_for_evaluation(self) except *:
        cdef BaseSystem system
        cdef bint is_ready_for_evaluation

        is_ready_for_evaluation = True

        for system in self.agent_systems():
            is_ready_for_evaluation = (
                is_ready_for_evaluation
                and system.is_ready_for_evaluation())

        return  is_ready_for_evaluation


    cpdef action(self, observation):
        cdef list joint_action
        cdef list joint_observation
        cdef BaseSystem system
        cdef Py_ssize_t agent_id
        cdef BaseReadableTypedList agent_systems

        agent_systems = self.agent_systems()

        joint_observation = observation

        joint_action = [None] * len(self.agent_systems())

        for agent_id in range(len(self.agent_systems())):
            system = agent_systems.item(agent_id)
            joint_action[agent_id] = system.action(joint_observation[agent_id])

        return joint_action


    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef list cy_feedback
        cdef BaseSystem system
        cdef Py_ssize_t agent_id
        cdef BaseReadableTypedList agent_systems

        agent_systems = self.agent_systems()

        cy_feedback = feedback

        for agent_id in range(len(self.agent_systems())):
            system = agent_systems.item(agent_id)
            system.receive_feedback(cy_feedback[agent_id])

    cpdef void update_policy(self) except *:
        cdef BaseSystem system

        for system in self.agent_systems():
            system.update_policy()

    cpdef void receive_score(self, double score) except *:
        cdef BaseSystem system

        for system in self.agent_systems():
            system.receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        cdef BaseSystem system

        for system in self.agent_systems():
            system.output_final_log(log_dirname, datetime_str)

    cpdef TypedList agent_systems(self):
        return self.__agent_systems

    cpdef void set_agent_systems(self, TypedList agent_systems) except *:
        cdef object systems_item_type

        systems_item_type = agent_systems.item_type()

        if not is_sub_full_type(systems_item_type, BaseSystem):
            raise (
                TypeError(
                    "The agent system's item type "
                    "(agent_systems.item_type() = {systems_item_type}) "
                    "must be a subtype of BaseSystem."
                    .format(**locals())))

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

    system.__agent_systems = new_TypedList(BaseSystem)