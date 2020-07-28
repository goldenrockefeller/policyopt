from .system cimport BaseSystem



cdef class MultiagentSystem(BaseSystem):
    cdef __agent_systems
    # list<BaseSystem>[n_agents] __agent_systems

    cpdef agent_systems(self)
    cpdef void set_agent_systems(self, agent_systems) except *

cdef MultiagentSystem new_MultiagentSystem()
cdef void init_MultiagentSystem(MultiagentSystem system) except *
