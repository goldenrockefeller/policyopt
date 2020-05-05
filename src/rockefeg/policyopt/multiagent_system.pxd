from .system cimport BaseSystem



cdef class MultiagentSystem(BaseSystem):
    cdef list __agent_systems
    # list<BaseSystem>[n_agents] __agent_systems

    cpdef Py_ssize_t n_agents(self) except *
    cpdef void append_agent_system(self, agent_system) except *
    cpdef pop_agent_system(self, Py_ssize_t index)
    cpdef void insert_agent_system(
        self,
        Py_ssize_t index,
        agent_system
        ) except *
    cpdef agent_system(self, Py_ssize_t index)
    cpdef void set_agent_system(self, Py_ssize_t index, agent_system) except *
    cpdef list _agent_systems(self)
    cpdef list agent_systems_shallow_copy(self)
    cpdef list agent_systems_deep_copy(self)
    cpdef void set_agent_systems(self, list agent_systems) except *

cdef MultiagentSystem new_MultiagentSystem()
cdef void init_MultiagentSystem(MultiagentSystem system) except *
