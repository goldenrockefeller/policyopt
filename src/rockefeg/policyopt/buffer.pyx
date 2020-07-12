cimport cython

import numpy as np

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class ShuffleBuffer:
    def __init__(self, Py_ssize_t capacity):
        init_ShuffleBuffer(self, capacity)

    cpdef copy(self, copy_obj = None):
        cdef ShuffleBuffer new_buffer

        if copy_obj is None:
            new_buffer = ShuffleBuffer.__new__(ShuffleBuffer)
        else:
            new_buffer = copy_obj

        new_buffer.__shuffled_data = self.__shuffled_data.copy()
        new_buffer.__staged_data = self.__staged_data.copy()
        new_buffer.__capacity = self.__capacity
        new_buffer.__buffer_pos = self.__buffer_pos

        return new_buffer

    cpdef void add_staged_datum(self, datum) except *:
        if len(self.__staged_data) >= capacity:
            self._append_staged_datum(datum)
        else:
            # Replace at buffer position and move the buffer position.
            self._set_staged_datum(self.__buffer_pos, datum)
            self._set_buffer_pos(self.__buffer_pos + 1)
            if self.__buffer_pos == capacity:
                self._set_buffer_pos(0)



    cpdef next_shuffled_datum(self, Py_ssize_t index = ?):
        cdef list new_shuffled_data

        if self.is_empty():
            raise (
                IndexError(
                    "Can not get next shuffled datum when the buffer is empty. "
                    "(self.is_empty() = True)" ))

        if len(self.__shuffled_datum) == 0:
            if len(self.__staged_datum) == 0:
                raise (
                    RuntimeError(
                        "Something went wrong: Buffer may or may not be empty"))
            else:
                new_shuffled_data = self.__staged_data.copy()
                np.random.shuffle(new_shuffled_data)
                self._set_shuffled_data(new_shuffled_data)

        return self._pop_shuffled_datum()


    cpdef void clear(self) except *:
        self._set_shuffled_data([])
        self._set_staged_data([])
        self._set_buffer_pos(0)

    cpdef bint is_empty(self) except *:
        return (
            (self.n_staged_data_points() == 0)
            and (self.n_shuffled_data_points() == 0) )


    cpdef Py_ssize_t capacity(self) except *:
        return self.__capacity

    cpdef void set_capacity(self, Py_ssize_t capacity) except *:
        if capacity <= 0:
            raise (
                ValueError(
                    "The capacity (capacity = {capacity}) must be positive."
                    .format(**locals()) ))
        self.__capacity = capacity


    cpdef Py_ssize_t _buffer_pos(self) except *:
        return self.__buffer_pos


    cpdef void _set_buffer_pos(self, Py_ssize_t buffer_pos) except *:
        cdef Py_ssize_t n_staged_data_points

        n_staged_data_points = self.n_staged_data_points()

        if buffer_pos <= 0:
            raise (
                ValueError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be positive."
                    .format(**locals()) ))
        if buffer_pos >= len(self.__staged_data):
            raise (
                IndexError(
                    "The buffer position (buffer_pos = {buffer_pos}) "
                    "must be less than the number of staged data points "
                    "(self.n_staged_data_points() = {n_staged_data_points})."
                    .format(**locals()) ))

        self.__buffer_pos = buffer_pos

    cpdef Py_ssize_t n_staged_data_points(self) except *:
          return len(self.__staged_data)

    cpdef void _append_staged_datum(self, staged_datum) except *:
        self.__staged_data.append(staged_datum)

    cpdef _pop_staged_datum(self, Py_ssize_t index = -1):
        return self.__staged_data.pop(index)

    cpdef void _insert_staged_datum(
            self,
            Py_ssize_t index,
            staged_datum
            ) except *:
        self.__staged_data.insert(index, staged_datum)

    cpdef void _set_staged_datum(self, Py_ssize_t index, staged_datum) except *:
        self.__staged_data[index] = staged_datum

    cpdef list _staged_data(self):
        return self.__staged_data

    cpdef void _set_staged_data(self, list staged_data) except *:
        cdef Py_ssize_t staged_datum_id
        cdef object staged_datum

        for staged_datum_id in range(len(staged_data)):
            staged_datum = staged_data[staged_datum_id]

        self.__staged_data = staged_data

    cpdef Py_ssize_t n_shuffled_data_points(self) except *:
          return len(self.__shuffled_data)

    cpdef void _append_shuffled_datum(self, shuffled_datum) except *:
        self.__shuffled_data.append(shuffled_datum)

    cpdef _pop_shuffled_datum(self, Py_ssize_t index = -1):
        return self.__shuffled_data.pop(index)

    cpdef void _insert_shuffled_datum(
            self,
            Py_ssize_t index,
            shuffled_datum
            ) except *:
        self.__shuffled_data.insert(index, shuffled_datum)

    cpdef void _set_shuffled_datum(self, Py_ssize_t index, shuffled_datum) except *:
        self.__shuffled_data[index] = shuffled_datum

    cpdef list _shuffled_data(self):
        return self.__shuffled_data

    cpdef void _set_shuffled_data(self, list shuffled_data) except *:
        cdef Py_ssize_t shuffled_datum_id
        cdef object shuffled_datum

        for shuffled_datum_id in range(len(shuffled_data)):
            shuffled_datum = shuffled_data[shuffled_datum_id]

        self.__shuffled_data = shuffled_data




@cython.warn.undeclared(True)
cdef ShuffleBuffer new_ShuffleBuffer(Py_ssize_t capacity):
    cdef ShuffleBuffer buffer

    buffer = ShuffleBuffer.__new__(ShuffleBuffer)
    init_ShuffleBuffer(buffer, capacity)

    return buffer

@cython.warn.undeclared(True)
cdef void init_ShuffleBuffer(
        ShuffleBuffer buffer,
        Py_ssize_t capacity
        ) except *:
    if buffer is None:
        raise TypeError("The buffer (buffer) cannot be None.")

    if capacity <= 0:
        raise (
            ValueError(
                "The capacity (capacity = {capacity}) must be positive."
                .format(**locals()) ))

    __staged_data = []
    __data = []
    __capacity = capacity
    __buffer_pos = 0