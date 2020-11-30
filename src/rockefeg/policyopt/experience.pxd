cdef class ExperienceDatum:
    cdef public object observation
    cdef public object action
    cdef public double reward

    cpdef ExperienceDatum copy(self, copy_obj = ?)

cdef ExperienceDatum new_ExperienceDatum()
