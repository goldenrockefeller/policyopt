cimport cython
import cython
import numpy as np

from typing import Generic, List, Sequence, TypeVar

# todo make an ImmutableShuffleBuffer class

T = TypeVar('T')

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ShuffleBuffer():
    def __init__(self):
        init_ShuffleBuffer(self)

    @cython.locals(shuffled_data = list)
    cpdef ShuffleBuffer copy(self, copy_obj = None):
        cdef ShuffleBuffer new_buffer
        shuffled_data: List[T]

        if copy_obj is None:
            new_buffer = ShuffleBuffer.__new__(ShuffleBuffer)
        else:
            new_buffer = copy_obj

        new_buffer._staged_data = self._staged_data.copy()

        new_buffer._shuffled_data = self._shuffled_data.copy()

        new_buffer._capacity = self._capacity
        new_buffer._buffer_pos = self._buffer_pos

        # Reshuffle the new buffer to decorrelate with self buffer.
        # TODO: .shuffle  can be optimized
        shuffled_data =  self._shuffled_data.copy()
        np.random.shuffle(shuffled_data)
        self._shuffled_data = shuffled_data

        return new_buffer

    cpdef void add_staged_datum(self, datum) except *:
        cdef staged_data
        cdef Py_ssize_t buffer_pos

        staged_data = self._staged_data

        if len(staged_data) < self.capacity():
            staged_data.append(datum)
        else:
            # Replace at buffer position and move the buffer position.
            buffer_pos = self._buffer_pos
            staged_data[buffer_pos] =  datum
            buffer_pos += 1
            if buffer_pos == self.capacity():
                buffer_pos = 0
            self._set_buffer_pos(buffer_pos)

    @cython.locals(new_shuffled_data = list)
    cpdef next_shuffled_datum(self):
        new_shuffled_data: List[T]

        if self.is_empty():
            raise (
                IndexError(
                    "Can not get next shuffled datum when the buffer is empty. "
                    "(self.is_empty() = True)" ))

        # TODO .shuffle() can be optimized
        new_shuffled_data =  self.staged_data().copy()
        np.random.shuffle(new_shuffled_data)
        self._set_shuffled_data(new_shuffled_data)

        return self._shuffled_data.pop()


    cpdef void clear(self) except *:
        self._set_staged_data([])
        self._set_shuffled_data([])
        self._set_buffer_pos(0)

    cpdef bint is_empty(self) except *:
        return (
            (len(self.staged_data()) == 0)
            and (len(self.shuffled_data()) == 0) )

    cpdef Py_ssize_t capacity(self) except *:
        return self.__capacity

    @cython.locals(staged_data = list, new_staged_data = list)
    cpdef void set_capacity(self, Py_ssize_t capacity) except *:

        cdef Py_ssize_t buffer_pos
        cdef Py_ssize_t datum_id
        staged_data: List[T]
        new_staged_data: List[T]

        staged_data = self._staged_data

        if capacity <= 0:
            raise (
                ValueError(
                    "The capacity (capacity = {capacity}) must be positive."
                    .format(**locals()) ))

        if capacity < self._capacity:
            new_staged_data = [None] * capacity

            buffer_pos = self._buffer_pos
            buffer_pos += self._capacity - capacity
            if buffer_pos > self._capacity:
                buffer_pos -= self._capacity

            for datum_id in range(len(new_staged_data)):
                new_staged_data[datum_id] = staged_data[buffer_pos]
                buffer_pos += 1
                if buffer_pos > self._capacity:
                    buffer_pos -= self._capacity

            self._set_staged_data(new_staged_data)
            self._set_buffer_pos(0)

        self._capacity = capacity



    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *:
        cdef Py_ssize_t n_staged_data_points

        n_staged_data_points = len(self._staged_data)

        if buffer_pos < 0:
            raise (
                ValueError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be non-negative."
                    .format(**locals()) ))

        if buffer_pos >= len(self._staged_data):
            raise (
                IndexError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be less than the number of staged data points "
                    "(self.n_staged_data_points() = {n_staged_data_points})."
                    .format(**locals()) ))

        self._buffer_pos = buffer_pos

    cpdef list staged_data(self):
        # type: (...) -> Sequence[T]
        return self._staged_data

    @cython.locals(staged_data = list)
    cpdef void _set_staged_data(self, staged_data: List[T]) except *:
        self._staged_data = staged_data

    cpdef list shuffled_data(self):
        # type: (...) -> list[T]
        return self._shuffled_data

    @cython.locals(shuffled_data = list)
    cpdef void _set_shuffled_data(self, shuffled_data: List[T])  except *:
        self._shuffled_data = shuffled_data

@cython.warn.undeclared(True)
cdef ShuffleBuffer new_ShuffleBuffer():
    cdef ShuffleBuffer buffer

    buffer = ShuffleBuffer.__new__(ShuffleBuffer)
    init_ShuffleBuffer(buffer)

    return buffer

@cython.warn.undeclared(True)
cdef void init_ShuffleBuffer(ShuffleBuffer buffer) except *:
    if buffer is None:
        raise TypeError("The buffer (buffer) cannot be None.")

    buffer._staged_data = []
    buffer._shuffled_data = []
    buffer._capacity = 1
    buffer._buffer_pos = 0
