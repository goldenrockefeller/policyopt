cimport cython

from rockefeg.cyutil.typed_list cimport TypedList, new_TypedList
from rockefeg.cyutil.typed_list cimport new_FixedLenTypedList

import numpy as np

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ShuffleBuffer:
    def __init__(self, item_type):
        init_ShuffleBuffer(self, item_type)

    cpdef ShuffleBuffer copy(self, copy_obj = None):
        cdef ShuffleBuffer new_buffer
        cdef list shuffled_data_list

        if copy_obj is None:
            new_buffer = ShuffleBuffer.__new__(ShuffleBuffer)
        else:
            new_buffer = copy_obj

        new_buffer.__staged_data = self.__staged_data.shallow_copy()

        new_buffer.__shuffled_data = self.__shuffled_data.shallow_copy()

        new_buffer.__capacity = self.__capacity
        new_buffer.__buffer_pos = self.__buffer_pos
        new_buffer.__item_type = self.__item_type

        # Reshuffle the new buffer to decorrelate with self buffer.
        # TODO: .shuffle  can be optimized
        shuffled_data_list =  self.__shuffled_data.items_shallow_copy()
        np.random.shuffle(shuffled_data_list)
        self.__shuffled_data.set_items(shuffled_data_list)

        return new_buffer

    cpdef void add_staged_datum(self, datum) except *:
        cdef TypedList staged_data
        cdef Py_ssize_t buffer_pos

        staged_data = self._staged_data()

        if len(staged_data) < self.capacity():
            staged_data.append(datum)
        else:
            # Replace at buffer position and move the buffer position.
            buffer_pos = self._buffer_pos()
            staged_data.set_item(buffer_pos, datum)
            buffer_pos += 1
            if buffer_pos == self.capacity():
                buffer_pos = 0
            self._set_buffer_pos(buffer_pos)

    cpdef next_shuffled_datum(self):
        cdef list shuffled_data_list
        cdef TypedList shuffled_data
        cdef TypedList staged_data

        staged_data = self._staged_data()
        shuffled_data = self._shuffled_data()

        if self.is_empty():
            raise (
                IndexError(
                    "Can not get next shuffled datum when the buffer is empty. "
                    "(self.is_empty() = True)" ))

        if len(shuffled_data) == 0:
            if len(staged_data) == 0:
                raise (
                    RuntimeError(
                        "Something went wrong: Buffer may or may not be empty"))
            else:
                # TODO .shuffle() can be optimized
                shuffled_data_list =  staged_data.items_shallow_copy()
                np.random.shuffle(shuffled_data_list)
                shuffled_data.set_items(shuffled_data_list)

        return shuffled_data.pop()


    cpdef void clear(self) except *:
        self._staged_data().set_items([])
        self._shuffled_data().set_items([])
        self._set_buffer_pos(0)

    cpdef bint is_empty(self) except *:
        return (
            (len(self.staged_data()) == 0)
            and (len(self.shuffled_data()) == 0) )

    cpdef Py_ssize_t capacity(self) except *:
        return self.__capacity

    cpdef void set_capacity(self, Py_ssize_t capacity) except *:
        cdef list new_staged_data
        cdef Py_ssize_t buffer_pos
        cdef Py_ssize_t datum_id
        cdef TypedList staged_data

        staged_data = self._staged_data()

        if capacity <= 0:
            raise (
                ValueError(
                    "The capacity (capacity = {capacity}) must be positive."
                    .format(**locals()) ))

        if capacity < self.__capacity:
            new_staged_data = [None] * capacity

            buffer_pos = self._buffer_pos()
            buffer_pos += self.__capacity - capacity
            if buffer_pos > self.__capacity:
                buffer_pos -= self.__capacity

            for datum_id in range(len(new_staged_data)):
                new_staged_data[datum_id] = staged_data.item(buffer_pos)
                buffer_pos += 1
                if buffer_pos > self.__capacity:
                    buffer_pos -= self.__capacity

            staged_data.set_items(new_staged_data)
            self._set_buffer_pos(0)

        self.__capacity = capacity

    cpdef Py_ssize_t _buffer_pos(self) except *:
        return self.__buffer_pos

    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *:
        cdef Py_ssize_t n_staged_data_points

        n_staged_data_points = len(self._staged_data())

        if buffer_pos < 0:
            raise (
                ValueError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be non-negative."
                    .format(**locals()) ))

        if buffer_pos >= len(self._staged_data()):
            raise (
                IndexError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be less than the number of staged data points "
                    "(self.n_staged_data_points() = {n_staged_data_points})."
                    .format(**locals()) ))

        self.__buffer_pos = buffer_pos

    cpdef BaseReadableTypedList staged_data(self):
        return self.__staged_data

    cpdef TypedList _staged_data(self):
        return self.__staged_data


    cpdef BaseReadableTypedList shuffled_data(self):
        return self.__shuffled_data

    cpdef TypedList _shuffled_data(self):
        return self.__shuffled_data

    cpdef item_type(self):
        return self.__item_type


@cython.warn.undeclared(True)
cdef ShuffleBuffer new_ShuffleBuffer(item_type):
    cdef ShuffleBuffer buffer

    buffer = ShuffleBuffer.__new__(ShuffleBuffer)
    init_ShuffleBuffer(buffer, item_type)

    return buffer

@cython.warn.undeclared(True)
cdef void init_ShuffleBuffer(ShuffleBuffer buffer, item_type) except *:
    if buffer is None:
        raise TypeError("The buffer (buffer) cannot be None.")

    buffer.__staged_data = new_TypedList(item_type)
    buffer.__shuffled_data = new_TypedList(item_type)
    buffer.__capacity = 1
    buffer.__buffer_pos = 0
    buffer.__item_type = item_type
