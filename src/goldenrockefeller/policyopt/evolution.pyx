cimport cython

from goldenrockefeller.cyutil.array cimport DoubleArray

from typing import List, Sequence, Generic, TypeVar


import random
import numpy as np
@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BasePhenotype:
    def __init__(self, BaseMap policy):
        init_BasePhenotype(self, policy)

    cpdef BasePhenotype copy(self, copy_obj = None):
        cdef BasePhenotype new_phenotype

        if copy_obj is None:
            new_phenotype = BasePhenotype.__new__(BasePhenotype)
        else:
            new_phenotype = copy_obj

        new_phenotype._policy = self._policy.copy()

        return new_phenotype

    cpdef BasePhenotype child(self, args = None):
        raise NotImplementedError("Abstract method.")

    cpdef void mutate(self, args = None) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void receive_feedback(self, feedback) except *:
        raise NotImplementedError("Abstract method.")

    cpdef action(self, observation):
        return self.policy().eval(observation)

    cpdef double fitness(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void set_fitness(self, double fitness) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void prep_for_epoch(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef BaseMap policy(self):
        return self._policy

    cpdef void set_policy(self, BaseMap policy) except *:
        self._policy = policy

@cython.warn.undeclared(True)
cdef void init_BasePhenotype(
        BasePhenotype phenotype,
        BaseMap policy
        ) except *:
    if phenotype is None:
        raise TypeError("The phenotype (phenotype) cannot be None.")

    if policy is None:
        raise TypeError("The policy (policy) cannot be None.")

    phenotype._policy = policy

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DefaultPhenotype(BasePhenotype):
    def __init__(self, BaseMap policy):
        init_DefaultPhenotype(self,  policy)

    cpdef DefaultPhenotype copy(self, copy_obj = None):
        cdef DefaultPhenotype new_phenotype

        if copy_obj is None:
            new_phenotype = DefaultPhenotype.__new__(DefaultPhenotype)
        else:
            new_phenotype = copy_obj

        new_phenotype = BasePhenotype.copy(self, new_phenotype)
        new_phenotype._mutation_rate = self._mutation_rate
        new_phenotype._mutation_factor = self._mutation_factor
        new_phenotype._fitness = 0.

        return new_phenotype

    cpdef DefaultPhenotype child(self, args = None):
        cdef DefaultPhenotype child

        child = self.copy()
        child.mutate()

        return child

    cpdef void mutate(self, args = None) except *:
        cdef DoubleArray parameters
        cdef object mutation
        cdef Py_ssize_t param_id
        cdef double[:] mutation_view

        # TODO Optimize, getting mutation vector is done through python (numpy).
        parameters =  self.policy().parameters()

        mutation = (
            self.mutation_factor()
            * np.random.standard_cauchy(len(parameters)))
        mutation *= (
            np.random.uniform(0, 1, len(parameters))
            < self.mutation_rate())

        # TODO avoid double view setting if possible
        mutation_view = mutation

        for param_id in range(len(parameters)):
            parameters.view[param_id] += mutation_view[param_id]

        self.policy().set_parameters(parameters)

    cpdef void receive_feedback(self, feedback) except *:
        cdef double feedback_as_double

        feedback_as_double = feedback
        self.set_fitness(self.fitness() + feedback_as_double)

    cpdef double fitness(self) except *:
        return self._fitness

    cpdef void prep_for_epoch(self) except *:
        self.set_fitness(0.)

    cpdef void set_fitness(self, double fitness) except *:
        self._fitness = fitness

    cpdef double mutation_rate(self) except *:
        return self._mutation_rate

    cpdef void set_mutation_rate(self, double mutation_rate) except *:
        if mutation_rate < 0. or mutation_rate > 1.:
            raise (
                ValueError(
                    "The mutation rate (mutation_rate = {mutation_rate}) must "
                    "be in the range [0., 1.]"
                    .format(**locals())))

        self._mutation_rate = mutation_rate

    cpdef double mutation_factor(self) except *:
        return self._mutation_factor

    cpdef void set_mutation_factor(self, double mutation_factor) except *:
        if mutation_factor < 0.:
            raise (
                ValueError(
                    "The mutation rate (mutation_factor = {mutation_factor}) "
                    "must be non-negative."
                    .format(**locals())))
        self._mutation_factor = mutation_factor

@cython.warn.undeclared(True)
cdef DefaultPhenotype new_DefaultPhenotype(BaseMap policy):
    cdef DefaultPhenotype phenotype

    phenotype = DefaultPhenotype.__new__(DefaultPhenotype)
    init_DefaultPhenotype(phenotype, policy)

    return phenotype

@cython.warn.undeclared(True)
cdef void init_DefaultPhenotype(
        DefaultPhenotype phenotype,
        BaseMap policy
        ) except *:
    if phenotype is None:
        raise TypeError("The INSERT_phenotype (phenotype) cannot be None.")

    init_BasePhenotype(phenotype, policy)
    phenotype._mutation_rate = 0.
    phenotype._mutation_factor = 0.
    phenotype._fitness = 0.

T = TypeVar('T', bound = BasePhenotype)

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseEvolvingSystem(BaseSystem):
    def __init__(self):
        init_BaseEvolvingSystem(self)

    cpdef BaseEvolvingSystem copy(self, copy_obj = None):
        cdef BaseEvolvingSystem new_system
        cdef BasePhenotype phenotype
        cdef Py_ssize_t phenotype_id

        if copy_obj is None:
            new_system = BaseEvolvingSystem.__new__(BaseEvolvingSystem)
        else:
            new_system = copy_obj

        # Deep Copy
        new_system._phenotypes = [None] * len(self._phenotypes)
        for phenotype_id in range(len(self._phenotypes)):
            phenotype = self._phenotypes[phenotype_id]
            new_system._phenotypes[phenotype_id] = phenotype.copy()

        # Deep Copy
        new_system._unevaluated_phenotypes = (
            [None] * len(self._unevaluated_phenotypes) )

        for phenotype_id in range(len(self._unevaluated_phenotypes)):
            phenotype = self._unevaluated_phenotypes.item(phenotype_id)
            new_system._unevaluated_phenotypes[phenotype_id] = phenotype.copy()

        new_system._best_phenotypes = None
        new_system._acting_phenotype = None
        new_system._max_n_epochs = self._max_n_epochs
        new_system._n_epochs_elapsed = self._n_epochs_elapsed

        return new_system

    cpdef bint is_done_training(self) except *:
        return self.n_epochs_elapsed() >= self.max_n_epochs()

    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t n_phenotypes
        cdef BasePhenotype phenotype
        cdef Py_ssize_t last_phenotype_id

        n_phenotypes = len(self.phenotypes())

        if n_phenotypes == 0:
            raise (
                RuntimeError(
                    "The number of phenotypes (len(self.phenotypes()) = "
                    "{n_phenotypes}) must be positive."
                    .format(**locals())))

        if self.is_done_training():
            raise (
                RuntimeError(
                "Cannot prepare for the next epoch. "
                "The system has stopped training."))

        for phenotype in self.phenotypes():
            phenotype.prep_for_epoch()

        self._set_unevaluated_phenotypes(self.phenotypes().copy())

        last_phenotype_id = len(self.unevaluated_phenotypes()) - 1
        self._set_acting_phenotype(
            self.unevaluated_phenotypes()[
                last_phenotype_id ] )

        self._set_best_phenotype(self.acting_phenotype())

    cpdef bint is_ready_for_evaluation(self) except *:
        return len(self.unevaluated_phenotypes()) == 0

    cpdef action(self, observation):
        return self.acting_phenotype().action(observation)

    cpdef void receive_feedback(self, feedback) except *:
        self.acting_phenotype().receive_feedback(feedback)

    @cython.locals(unevaluated_phenotypes = list)
    cpdef void update_policy(self) except *:
        cdef BasePhenotype acting_phenotype
        cdef BasePhenotype best_phenotype
        cdef double acting_fitness
        cdef double best_epoch_fitness
        cdef Py_ssize_t last_phenotype_id
        unevaluated_phenotypes: List[T]

        acting_phenotype = self.acting_phenotype()
        acting_fitness = acting_phenotype.fitness()
        best_phenotype = self.best_phenotype()
        best_epoch_fitness = best_phenotype.fitness()
        unevaluated_phenotypes = self._unevaluated_phenotypes

        # Update the best phenotype.
        if acting_fitness > best_epoch_fitness:
            self._set_best_phenotype(acting_phenotype)

        # Remove current evaluated phenotype.
        unevaluated_phenotypes.pop()

        if len(unevaluated_phenotypes) > 0:
            # Get new phenotype to evaluate.
            last_phenotype_id = len(unevaluated_phenotypes) - 1
            self._set_acting_phenotype(
                unevaluated_phenotypes[
                    last_phenotype_id ] )

        elif self.n_epochs_elapsed() < self.max_n_epochs():
            # The epoch has ended, operate of phenotype population.
            # Set the policy to the best phenotype for evaluation.
            self._set_acting_phenotype(self.best_phenotype())
            self.operate()

            # An epoch/generation has passed
            self._set_n_epochs_elapsed(self.n_epochs_elapsed() + 1)

    cpdef void operate(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void receive_score(self, double score) except *:
        pass

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        pass

    cpdef list phenotypes(self):
        # type: (...) -> List[T]
        return self._phenotypes

    @cython.locals(phenotypes = list)
    cpdef void set_phenotypes(self, phenotypes: List[T]) except *:
        self._phenotypes = phenotypes

    cpdef list unevaluated_phenotypes(self):
        # type: (...) -> Sequence[T]
        return self._unevaluated_phenotypes


    @cython.locals(phenotypes = list)
    cpdef void _set_unevaluated_phenotypes(
            self,
            phenotypes: List[T]
            ) except *:
        self._unevaluated_phenotypes = phenotypes

    cpdef Py_ssize_t max_n_epochs(self) except *:
        return self._max_n_epochs

    cpdef void set_max_n_epochs(self, Py_ssize_t max_n_epochs) except*:
        cdef Py_ssize_t n_epochs_elapsed

        n_epochs_elapsed = self.n_epochs_elapsed()

        if max_n_epochs < n_epochs_elapsed:
            raise (
                ValueError(
                    "The new maximum number of epochs (max_n_epochs = "
                    "{max_n_epochs}) must not be less than the current "
                    "number of epochs elapsed (self.n_epochs_elapsed() = "
                    "{n_epochs_elapsed})."
                    .format(**locals())))

        if max_n_epochs <= 0:
            raise (
                ValueError(
                    "The new maximum number of epochs (max_n_epochs = "
                    "{max_n_epochs}) must be positive."
                    .format(**locals())))

        self._max_n_epochs = max_n_epochs


    cpdef Py_ssize_t n_epochs_elapsed(self) except *:
        return self._n_epochs_elapsed

    cpdef void _set_n_epochs_elapsed(self, Py_ssize_t n_epochs_elapsed) except*:
        if n_epochs_elapsed < 0:
            raise (
                ValueError(
                    "The number epochs elapsed (n_epochs_elapsed = "
                    "{n_epochs_elapsed}) must be non-negative."
                    .format(**locals())))

        self._n_epochs_elapsed = n_epochs_elapsed

    cpdef BasePhenotype best_phenotype(self):
        return self._best_phenotype

    cpdef void _set_best_phenotype(self, BasePhenotype phenotype) except *:
        self._best_phenotype = phenotype

    cpdef BasePhenotype acting_phenotype(self):
        return  self._acting_phenotype

    cpdef void _set_acting_phenotype(self, BasePhenotype phenotype) except *:
        self._acting_phenotype = phenotype

@cython.warn.undeclared(True)
cdef BaseEvolvingSystem new_BaseEvolvingSystem():
    cdef BaseEvolvingSystem system

    system = BaseEvolvingSystem.__new__(BaseEvolvingSystem)
    init_BaseEvolvingSystem(system)

    return system

@cython.warn.undeclared(True)
cdef void init_BaseEvolvingSystem(BaseEvolvingSystem system) except *:
    if system is None:
        raise TypeError("The system (system) cannot be None.")

    system._phenotypes = []
    system._unevaluated_phenotypes = []
    system._best_phenotype = None
    system._acting_phenotype = None
    system._max_n_epochs = 1
    system._n_epochs_elapsed = 0

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DefaultEvolvingSystem(BaseEvolvingSystem):
    def __init__(self):
        init_DefaultEvolvingSystem(self)

    cpdef DefaultEvolvingSystem copy(self, copy_obj = None):
        cdef DefaultEvolvingSystem new_system

        if copy_obj is None:
            new_system = DefaultEvolvingSystem.__new__(DefaultEvolvingSystem)
        else:
            new_system = copy_obj

        new_system = BaseEvolvingSystem.copy(self, new_system)

        return new_system

    @cython.locals(phenotypes = list)
    cpdef void operate(self) except *:
        cdef Py_ssize_t match_id
        cdef DefaultPhenotype contender_a
        cdef DefaultPhenotype contender_b
        cdef double fitness_a
        cdef double fitness_b
        phenotypes: List[DefaultPhenotype]

        phenotypes = self.phenotypes().copy()

        # TODO: Optimize random shuffle with non-python random shuffle.
        random.shuffle(phenotypes)

        for match_id in range(len(phenotypes)// 2):
            # Find the match winner amongst contenders.
            contender_a = phenotypes[2 * match_id]
            contender_b = phenotypes[2 * match_id + 1]
            fitness_a = contender_a.fitness()
            fitness_b = contender_b.fitness()
            if fitness_a > fitness_b:
                phenotypes[2 * match_id + 1] = contender_a.child()
            else:
                phenotypes[2 * match_id] = contender_b.child()

        self.set_phenotypes(phenotypes)

    @cython.locals(phenotypes = list)
    cpdef void set_phenotypes(
            self,
            phenotypes: List[DefaultPhenotype]
            ) except *:
        BaseEvolvingSystem.set_phenotypes(self, phenotypes)

@cython.warn.undeclared(True)
cdef DefaultEvolvingSystem new_DefaultEvolvingSystem():
    cdef DefaultEvolvingSystem system

    system = DefaultEvolvingSystem.__new__(DefaultEvolvingSystem)
    init_DefaultEvolvingSystem(system)

    return system

@cython.warn.undeclared(True)
cdef void init_DefaultEvolvingSystem(DefaultEvolvingSystem system) except *:
    if system is None:
        raise TypeError("The system (system) cannot be None.")

    init_BaseEvolvingSystem(system)
    system._phenotypes = []
    system._unevaluated_phenotypes = []





