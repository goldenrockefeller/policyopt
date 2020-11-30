from .system cimport BaseSystem
from rockefeg.cyutil.typed_list cimport TypedList



cdef class MultiagentSystem(BaseSystem):
    cdef TypedList __agent_systems
    # list<BaseSystem>[n_agents] __agent_systems

    cpdef MultiagentSystem copy(self, copy_obj = ?)

    cpdef TypedList agent_systems(self)
    cpdef void set_agent_systems(self, TypedList agent_systems) except *

cdef MultiagentSystem new_MultiagentSystem()
cdef void init_MultiagentSystem(MultiagentSystem system) except *
