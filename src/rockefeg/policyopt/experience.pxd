cdef class ExperienceDatum:
    cdef public object state
    cdef public object action
    cdef public double reward

    cpdef copy(self, copy_obj = ?)

cdef ExperienceDatum new_ExperienceDatum()
