from rockefeg.cyutil.typed_list cimport TypedList, BaseReadableTypedList

cdef class ShuffleBuffer:
    cdef TypedList __shuffled_data
    cdef TypedList __staged_data
    cdef __item_type
    cdef Py_ssize_t __capacity
    cdef Py_ssize_t __buffer_pos

    cpdef ShuffleBuffer copy(self, copy_obj = ?)

    cpdef void add_staged_datum(self, datum) except *

    cpdef next_shuffled_datum(self)

    cpdef void clear(self) except *

    cpdef bint is_empty(self) except *
    # pos too

    cpdef Py_ssize_t capacity(self) except *
    cpdef void set_capacity(self, Py_ssize_t capacity) except *

    cpdef Py_ssize_t _buffer_pos(self) except *
    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *

    cpdef BaseReadableTypedList staged_data(self)
    cpdef TypedList _staged_data(self)

    cpdef BaseReadableTypedList shuffled_data(self)
    cpdef TypedList _shuffled_data(self)

    cpdef item_type(self)


cdef ShuffleBuffer new_ShuffleBuffer(item_type)
cdef void init_ShuffleBuffer(ShuffleBuffer buffer, item_type) except *