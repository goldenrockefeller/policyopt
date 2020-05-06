from .system cimport BaseSystem
from .policy cimport BasePolicy

cdef class BasePhenotype:
    cdef object __policy

    cpdef copy(self, copy_obj = ?)

    cpdef child(self, args = ?)

    cpdef void mutate(self, args = ?) except *

    cpdef void receive_feedback(self, feedback) except *

    cpdef action(self, observation)

    cpdef void prep_for_epoch(self) except *

    cpdef policy(self)

    cpdef void set_policy(self, policy) except *

cdef void init_BasePhenotype(
    BasePhenotype phenotype,
    BasePolicy policy
    ) except *

cdef class DefaultPhenotype(BasePhenotype):
    cdef double __mutation_rate
    cdef double __mutation_factor
    cdef double __fitness



    cpdef double mutation_rate(self) except *
    cpdef void set_mutation_rate(self, double mutation_rate) except *
    cpdef double mutation_factor(self) except *
    cpdef void set_mutation_factor(self, double mutation_factor) except *

    cpdef double fitness(self) except *
    cpdef void set_fitness(self, double fitness) except *


cdef DefaultPhenotype new_DefaultPhenotype(BasePolicy policy)

cdef void init_DefaultPhenotype(
    DefaultPhenotype phenotype,
    BasePolicy policy
    ) except *

cdef class BaseEvolvingSystem(BaseSystem):
    cdef list __phenotypes
    cdef list __unevaluated_phenotypes
    cdef object __best_phenotype
    cdef object __acting_phenotype
    cdef Py_ssize_t __max_n_epochs
    cdef Py_ssize_t __n_epochs_elapsed

    cpdef void operate(self) except *

    cpdef Py_ssize_t n_phenotypes(self) except *
    cpdef void append_phenotype(self, phenotype) except *
    cpdef pop_phenotype(self, Py_ssize_t index)
    cpdef void insert_phenotype(self, Py_ssize_t index, phenotype) except *
    cpdef phenotype(self, Py_ssize_t index)
    cpdef void set_phenotype(self, Py_ssize_t index, phenotype) except *
    cpdef list _phenotypes(self)
    cpdef list phenotypes_shallow_copy(self)
    cpdef list phenotypes_deep_copy(self)
    cpdef void set_phenotypes(self, list phenotypes) except *

    cpdef Py_ssize_t n_unevaluated_phenotypes(self) except *
    cpdef void _append_unevaluated_phenotype(
        self,
        unevaluated_phenotype
        ) except *
    cpdef _pop_unevaluated_phenotype(self, Py_ssize_t index)
    cpdef void _insert_unevaluated_phenotype(
        self,
        Py_ssize_t index,
        unevaluated_phenotype
        ) except *
    cpdef _unevaluated_phenotype(self, Py_ssize_t index)
    cpdef unevaluated_phenotype_copy(self, Py_ssize_t index)
    cpdef void _set_unevaluated_phenotype(
        self,
        Py_ssize_t index,
        unevaluated_phenotype
        ) except *
    cpdef list _unevaluated_phenotypes(self)
    cpdef list _unevaluated_phenotypes_shallow_copy(self)
    cpdef list unevaluated_phenotypes_deep_copy(self)
    cpdef void _set_unevaluated_phenotypes(
        self,
        list unevaluated_phenotypes
        ) except *

    cpdef Py_ssize_t max_n_epochs(self) except *
    cpdef void set_max_n_epochs(self, Py_ssize_t max_n_epochs) except*

    cpdef Py_ssize_t n_epochs_elapsed(self) except *
    cpdef void _set_n_epochs_elapsed(self, Py_ssize_t n_epochs_elapsed) except *

    cpdef best_phenotype(self)
    cpdef void _set_best_phenotype(self, phenotype) except *

    cpdef acting_phenotype(self)
    cpdef void _set_acting_phenotype(self, phenotype) except *


cdef BaseEvolvingSystem new_BaseEvolvingSystem()
cdef void init_BaseEvolvingSystem(BaseEvolvingSystem system) except *

cdef class DefaultEvolvingSystem(BaseEvolvingSystem):
    pass


cdef DefaultEvolvingSystem new_DefaultEvolvingSystem()
cdef void init_DefaultEvolvingSystem(DefaultEvolvingSystem system) except *
