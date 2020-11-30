from .system cimport BaseSystem
from .map cimport BaseMap
from rockefeg.cyutil.typed_list cimport TypedList, BaseReadableTypedList

cdef class BasePhenotype:
    cdef BaseMap __policy

    cpdef BasePhenotype copy(self, copy_obj = ?)

    cpdef BasePhenotype child(self, args = ?)

    cpdef void mutate(self, args = ?) except *

    cpdef void receive_feedback(self, feedback) except *

    cpdef action(self, observation)

    cpdef void prep_for_epoch(self) except *

    cpdef BaseMap policy(self)

    cpdef void set_policy(self, BaseMap policy) except *

    cpdef double fitness(self) except *
    cpdef void set_fitness(self, double fitness) except *

cdef void init_BasePhenotype(
    BasePhenotype phenotype,
    BaseMap policy
    ) except *

cdef class DefaultPhenotype(BasePhenotype):
    cdef double __mutation_rate
    cdef double __mutation_factor
    cdef double __fitness

    cpdef DefaultPhenotype copy(self, copy_obj = ?)

    cpdef DefaultPhenotype child(self, args = ?)

    cpdef double mutation_rate(self) except *
    cpdef void set_mutation_rate(self, double mutation_rate) except *
    cpdef double mutation_factor(self) except *
    cpdef void set_mutation_factor(self, double mutation_factor) except *

cdef DefaultPhenotype new_DefaultPhenotype(BaseMap policy)

cdef void init_DefaultPhenotype(
    DefaultPhenotype phenotype,
    BaseMap policy
    ) except *

cdef class BaseEvolvingSystem(BaseSystem):
    cdef TypedList __phenotypes
    cdef TypedList __unevaluated_phenotypes
    cdef BasePhenotype __best_phenotype
    cdef BasePhenotype __acting_phenotype
    cdef Py_ssize_t __max_n_epochs
    cdef Py_ssize_t __n_epochs_elapsed

    cpdef BaseEvolvingSystem copy(self, copy_obj = ?)

    cpdef void operate(self) except *

    cpdef TypedList phenotypes(self)
    cpdef void set_phenotypes(self, TypedList phenotypes) except *

    cpdef BaseReadableTypedList unevaluated_phenotypes(self)
    cpdef TypedList _unevaluated_phenotypes(self)
    cpdef void _set_unevaluated_phenotypes(self, TypedList phenotypes) except *

    cpdef Py_ssize_t max_n_epochs(self) except *
    cpdef void set_max_n_epochs(self, Py_ssize_t max_n_epochs) except *

    cpdef Py_ssize_t n_epochs_elapsed(self) except *
    cpdef void _set_n_epochs_elapsed(self, Py_ssize_t n_epochs_elapsed) except *

    cpdef BasePhenotype best_phenotype(self)
    cpdef void _set_best_phenotype(self, BasePhenotype phenotype) except *

    cpdef BasePhenotype acting_phenotype(self)
    cpdef void _set_acting_phenotype(self, BasePhenotype phenotype) except *


cdef BaseEvolvingSystem new_BaseEvolvingSystem()
cdef void init_BaseEvolvingSystem(BaseEvolvingSystem system) except *

cdef class DefaultEvolvingSystem(BaseEvolvingSystem):
    cpdef DefaultEvolvingSystem copy(self, copy_obj = ?)


cdef DefaultEvolvingSystem new_DefaultEvolvingSystem()
cdef void init_DefaultEvolvingSystem(DefaultEvolvingSystem system) except *
