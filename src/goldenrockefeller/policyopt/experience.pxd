cdef class ExperienceDatum:
    cdef public object observation
    cdef public object action
    cdef public double reward
    # TODO add a "cdef object feedback" member for extra flexbility.
    # TODO cont'd: replace "double reward" with "object feedback" if computation cheap

    cpdef ExperienceDatum copy(self, copy_obj = ?)

cdef ExperienceDatum new_ExperienceDatum()
