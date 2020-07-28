cdef class ShuffleBuffer:
    cdef __shuffled_data
    cdef __staged_data
    cdef __item_type
    cdef Py_ssize_t __capacity
    cdef Py_ssize_t __buffer_pos

    cpdef copy(self, copy_obj = ?)

    cpdef void add_staged_datum(self, datum) except *

    cpdef next_shuffled_datum(self)

    cpdef void clear(self) except *

    cpdef bint is_empty(self) except *
    # pos too

    cpdef Py_ssize_t capacity(self) except *
    cpdef void set_capacity(self, Py_ssize_t capacity) except *

    cpdef Py_ssize_t _buffer_pos(self) except *
    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *

    cpdef fixed_len_staged_data(self)
    cpdef _staged_data(self)

    cpdef fixed_len_shuffled_data(self)
    cpdef _shuffled_data(self)

    cpdef item_type(self)







cdef ShuffleBuffer new_ShuffleBuffer(item_type)
cdef void init_ShuffleBuffer(ShuffleBuffer buffer, item_type) except *