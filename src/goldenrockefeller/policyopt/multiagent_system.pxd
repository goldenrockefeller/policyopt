from .system cimport BaseSystem

import cython

from typing import Sequence


cdef class MultiagentSystem(BaseSystem):
    cdef list _agent_systems
    # list<BaseSystem>[n_agents] _agent_systems

    cpdef MultiagentSystem copy(self, copy_obj = ?)

    cpdef list agent_systems(self)
    # type: (...) -> List[T]

    @cython.locals(agent_systems = list)
    cpdef void set_agent_systems(
        self,
        agent_systems: Sequence[T]
        ) except *

cdef MultiagentSystem new_MultiagentSystem()
cdef void init_MultiagentSystem(MultiagentSystem system) except *
