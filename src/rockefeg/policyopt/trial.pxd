from .system cimport BaseSystem
from .domain cimport BaseDomain

cdef class Trial:
    cdef public BaseSystem system
    cdef public BaseDomain domain
    cdef public object experiment_name
    cdef public object mod_name
    cdef public bint prints_score
    cdef public bint deletes_final_save_file
    cdef public object save_period
    cdef public Py_ssize_t n_training_episodes_elapsed
    cdef public Py_ssize_t n_generations_elapsed
    cdef public Py_ssize_t n_training_steps_elapsed
    cdef public object datetime_str
    cdef public object log_dirname
    cdef public list score_history

    cpdef void save(self) except *

    cpdef void delete_final_save_file(self) except *

    cpdef void log_score_history(self) except *

    cpdef void run(self) except *
