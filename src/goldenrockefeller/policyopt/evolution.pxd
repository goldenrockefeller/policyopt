import cython

from .system cimport BaseSystem
from .map cimport BaseMap

from typing import List, Seqeunce, Generic


cdef class BasePhenotype:
    cdef BaseMap _policy

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
    cdef double _mutation_rate
    cdef double _mutation_factor
    cdef double _fitness

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

# T must be a BasePhenotype
cdef class BaseEvolvingSystem(BaseSystem):
    cdef list _phenotypes
    cdef list _unevaluated_phenotypes
    cdef BasePhenotype _best_phenotype
    cdef BasePhenotype _acting_phenotype
    cdef Py_ssize_t _max_n_epochs
    cdef Py_ssize_t _n_epochs_elapsed

    cpdef BaseEvolvingSystem copy(self, copy_obj = ?)

    cpdef void operate(self) except *

    cpdef list phenotypes(self)
    # type: (...) -> List[T]

    @cython.locals(phenotypes = list)
    cpdef void set_phenotypes(self, phenotypes: List[T]) except *

    cpdef list unevaluated_phenotypes(self)
    # type: (...) -> Sequence[T]

    @cython.locals(phenotypes = list)
    cpdef void _set_unevaluated_phenotypes(
        self,
        phenotypes: List[T]
        ) except *

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
