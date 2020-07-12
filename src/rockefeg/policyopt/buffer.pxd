cdef class ShuffleBuffer:
    cdef list __shuffled_data
    cdef list __staged_data
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

    cpdef Py_ssize_t n_staged_data_points(self) except *
    cpdef void _append_staged_datum(self, staged_datum) except *
    cpdef _pop_staged_datum(self, Py_ssize_t index = ?)
    cpdef void _insert_staged_datum(
        self,
        Py_ssize_t index,
        staged_datum
        ) except *
    cpdef void _set_staged_datum(self, Py_ssize_t index, staged_datum) except *
    cpdef list _staged_data(self)
    cpdef void _set_staged_data(self, list staged_data) except *

    cpdef Py_ssize_t n_shuffled_data_points(self) except *
    cpdef void _append_shuffled_datum(self, shuffled_datum) except *
    cpdef _pop_shuffled_datum(self, Py_ssize_t index = ?)
    cpdef void _insert_shuffled_datum(
        self,
        Py_ssize_t index,
        shuffled_datum) except *
    cpdef void s_et_shuffled_datum(
        self,
        Py_ssize_t index,
        shuffled_datum) except *
    cpdef list _shuffled_data(self)
    cpdef void _set_shuffled_data(self, list shuffled_data) except *




cdef ShuffleBuffer new_ShuffleBuffer(Py_ssize_t capacity)
cdef void init_ShuffleBuffer(ShuffleBuffer buffer, Py_ssize_t capacity) except *