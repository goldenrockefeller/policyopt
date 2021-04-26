import cython
from typing import List


cdef class ShuffleBuffer() :
    cdef list _shuffled_data
    cdef list _staged_data
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _buffer_pos

    cpdef ShuffleBuffer copy(self, copy_obj = ?)

    cpdef void add_staged_datum(self, datum) except *

    cpdef next_shuffled_datum(self)

    cpdef void clear(self) except *

    cpdef bint is_empty(self) except *

    cpdef Py_ssize_t capacity(self) except *
    cpdef void set_capacity(self, Py_ssize_t capacity) except *

    cpdef Py_ssize_t _buffer_pos(self) except *
    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *


    cpdef list staged_data(self)
    # type: (...) -> Sequence[T]

    @cython.locals(staged_data = list)
    cpdef void _set_staged_data(self, staged_data: List[T]) except *


    cpdef list shuffled_data(self)
    # type: (...) -> Sequence[T]

    @cython.locals(shuffled_data = list)
    cpdef void _set_shuffled_data(self, shuffled_data: List[T])  except *



cdef ShuffleBuffer new_ShuffleBuffer()
cdef void init_ShuffleBuffer(ShuffleBuffer buffer) except *