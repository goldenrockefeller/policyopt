cimport cython


@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ExperienceDatum:
    cpdef copy(self, copy_obj = None):
        cdef ExperienceDatum new_datum

        if copy_obj is None:
            new_datum = ExperienceDatum.__new__(ExperienceDatum)
        else:
            new_datum = copy_obj

        new_datum.state = self.state
        new_datum.action = self.action
        new_datum.reward = self.reward

        return new_datum

@cython.warn.undeclared(True)
cdef ExperienceDatum new_ExperienceDatum():
    cdef ExperienceDatum datum

    datum = ExperienceDatum.__new__(ExperienceDatum)

    return datum

