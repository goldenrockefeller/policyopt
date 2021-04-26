cimport cython
from .system cimport BaseSystem
from .system import BaseSystem

import cython
from typing import Generic, TypeVar, List, Sequence

T = TypeVar('T', bound = BaseSystem)
ActionT = TypeVar('ActionT')
ObservationT = TypeVar('ObservationT')
FeedbackT = TypeVar('FeedbackT')

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class MultiagentSystem(BaseSystem):
    def __init__(self):
        init_MultiagentSystem(self)

    cpdef MultiagentSystem copy(self, copy_obj = None):
        cdef MultiagentSystem new_system
        cdef BaseSystem agent_system
        cdef Py_ssize_t agent_id

        if copy_obj is None:
            new_system = MultiagentSystem.__new__(MultiagentSystem)
        else:
            new_system = copy_obj

        # Deep copy.
        new_system._agent_systems  = [None] * len(self._agent_systems)
        for agent_id in range(len(self._agent_systems)):
            agent_system = self._agent_systems[agent_id]
            new_system._agent_systems[agent_id] = agent_system.copy()

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

    @cython.locals(
        agent_systems = list,
        joint_action = list,
        joint_observation = list)
    cpdef action(self, observation):
        joint_action: List[ActionT]
        joint_observation: List[ObservationT]
        cdef BaseSystem system
        cdef Py_ssize_t agent_id
        agent_systems: Sequence[T]

        agent_systems = self.agent_systems()

        joint_observation = observation

        joint_action = [None] * len(self.agent_systems())

        for agent_id in range(len(self.agent_systems())):
            system = agent_systems[agent_id]
            joint_action[agent_id] = system.action(joint_observation[agent_id])

        return joint_action


    #step_wise feedback
    @cython.locals(agent_systems = list, cy_feedback = list)
    cpdef void receive_feedback(self, feedback: Sequence[FeedbackT]) except *:
        cy_feedback: Sequence[FeedbackT]
        cdef BaseSystem system
        cdef Py_ssize_t agent_id
        agent_systems: List[T]

        agent_systems = self.agent_systems()

        cy_feedback = feedback

        for agent_id in range(len(self.agent_systems())):
            system = agent_systems[agent_id]
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

    cpdef list agent_systems(self):
        # type: (...) -> List[T]
        return self._agent_systems

    @cython.locals(agent_systems = list)
    cpdef void set_agent_systems(self, agent_systems: List[T]) except *:
        self._agent_systems = agent_systems

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

    system._agent_systems = []