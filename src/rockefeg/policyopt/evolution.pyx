cimport cython

from rockefeg.cyutil.array cimport DoubleArray

import random
import numpy as np
@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BasePhenotype:
    def __init__(self, policy):
        init_BasePhenotype(self, policy)

    cpdef copy(self, copy_obj = None):
        cdef BasePhenotype new_phenotype

        if copy_obj is None:
            new_phenotype = BasePhenotype.__new__(BasePhenotype)
        else:
            new_phenotype = copy_obj

        new_phenotype.__policy = (<BasePolicy?>self.__policy).copy()

        return new_phenotype

    cpdef child(self, args = None):
        raise NotImplementedError("Abstract method.")

    cpdef void mutate(self, args = None) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void receive_feedback(self, feedback) except *:
        raise NotImplementedError("Abstract method.")

    cpdef action(self, observation):
        return (<BasePolicy?>self.__policy).action(observation)

    cpdef void prep_for_epoch(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef policy(self):
        return self.__policy

    cpdef void set_policy(self, policy) except *:
        self.__policy = <BasePolicy?>policy

@cython.warn.undeclared(True)
cdef void init_BasePhenotype(
        BasePhenotype phenotype,
        BasePolicy policy
        ) except *:
    if phenotype is None:
        raise TypeError("The phenotype (phenotype) cannot be None.")

    if policy is None:
        raise TypeError("The policy (policy) cannot be None.")

    phenotype.__policy = policy

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DefaultPhenotype(BasePhenotype):
    def __init__(self, policy):
        init_DefaultPhenotype(self, policy)

    cpdef copy(self, copy_obj = None):
        cdef DefaultPhenotype new_phenotype

        if copy_obj is None:
            new_phenotype = DefaultPhenotype.__new__(DefaultPhenotype)
        else:
            new_phenotype = copy_obj

        new_phenotype = BasePhenotype.copy(self, new_phenotype)
        new_phenotype.__mutation_rate = self.__mutation_rate
        new_phenotype.__mutation_factor = self.__mutation_factor
        new_phenotype.__fitness = 0.

        return new_phenotype

    cpdef child(self, args = None):
        cdef DefaultPhenotype child

        child = self.copy()
        child.mutate()

        return child

    cpdef void mutate(self, args = None) except *:
        cdef BasePolicy policy
        cdef DoubleArray parameters
        cdef object mutation
        cdef Py_ssize_t param_id
        cdef double[:] mutation_view

        # TODO Optimize, getting mutation vector is done through python (numpy).
        policy = <BasePolicy?>self.policy()

        parameters = <DoubleArray?> policy.parameters()

        mutation = (
            self.mutation_factor()
            * np.random.standard_cauchy(len(parameters)))
        mutation *= (
            np.random.uniform(0, 1, len(parameters))
            < self.mutation_rate())

        mutation_view = mutation

        for param_id in range(len(parameters)):
            parameters.view[param_id] += mutation_view[param_id]

        policy.set_parameters(parameters)

    cpdef void receive_feedback(self, feedback) except *:
        self.set_fitness(self.fitness() + <double?>feedback)

    cpdef double fitness(self) except *:
        return self.__fitness

    cpdef void prep_for_epoch(self) except *:
        self.set_fitness(0.)

    cpdef void set_fitness(self, double fitness) except *:
        self.__fitness = fitness

    cpdef double mutation_rate(self) except *:
        return self.__mutation_rate

    cpdef void set_mutation_rate(self, double mutation_rate) except *:
        if mutation_rate < 0. or mutation_rate > 1.:
            raise (
                ValueError(
                    "The mutation rate (mutation_rate = {mutation_rate}) must "
                    "be in the range [0., 1.]"
                    .format(**locals())))

        self.__mutation_rate = mutation_rate

    cpdef double mutation_factor(self) except *:
        return self.__mutation_factor

    cpdef void set_mutation_factor(self, double mutation_factor) except *:
        if mutation_factor < 0.:
            raise (
                ValueError(
                    "The mutation rate (mutation_factor = {mutation_factor}) "
                    "must be non-negative."
                    .format(**locals())))
        self.__mutation_factor = mutation_factor

@cython.warn.undeclared(True)
cdef DefaultPhenotype new_DefaultPhenotype(BasePolicy policy):
    cdef DefaultPhenotype phenotype

    phenotype = DefaultPhenotype.__new__(DefaultPhenotype)
    init_DefaultPhenotype(phenotype, policy)

    return phenotype

@cython.warn.undeclared(True)
cdef void init_DefaultPhenotype(
        DefaultPhenotype phenotype,
        BasePolicy policy
        ) except *:
    if phenotype is None:
        raise TypeError("The INSERT_phenotype (phenotype) cannot be None.")

    init_BasePhenotype(phenotype, policy)
    phenotype.__mutation_rate = 0.
    phenotype.__mutation_factor = 0.
    phenotype.__fitness = 0.



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseEvolvingSystem(BaseSystem):
    def __init__(self):
        init_BaseEvolvingSystem(self)

    cpdef copy(self, copy_obj = None):
        cdef BaseEvolvingSystem new_system
        cdef BasePhenotype phenotype
        cdef Py_ssize_t phenotype_id

        if copy_obj is None:
            new_system = BaseEvolvingSystem.__new__(BaseEvolvingSystem)
        else:
            new_system = copy_obj

        new_system.__phenotypes = self.phenotypes_deep_copy()
        new_system.__unevaluated_phenotypes = (
            new_system.phenotypes_shallow_copy())
        new_system.__best_phenotypes = None
        new_system.__acting_phenotype = None
        new_system.__max_n_epochs = self.__max_n_epochs
        new_system.__n_epochs_elapsed = self.__n_epochs_elapsed

        return new_system

    cpdef bint is_done_training(self) except *:
        return self.__n_epochs_elapsed >= self.__max_n_epochs

    cpdef void prep_for_epoch(self) except *:
        cdef BasePhenotype phenotype
        cdef Py_ssize_t n_phenotypes

        n_phenotypes = self.n_phenotypes()

        if n_phenotypes == 0:
            raise (
                RuntimeError(
                    "The number of phenotypes (self.n_phenotypes() = "
                    "{n_phenotypes}) must be positive."
                    .format(**locals())))

        if self.is_done_training():
            raise (
                RuntimeError(
                "Cannot prepare for the next epoch. "
                "The system has stopped training."))

        for phenotype in self.__phenotypes:
            phenotype.prep_for_epoch()

        self.__unevaluated_phenotypes = (
            self.phenotypes_shallow_copy())
        self._set_acting_phenotype(self.__unevaluated_phenotypes[-1])
        self._set_best_phenotype(self.__acting_phenotype)

    cpdef bint is_ready_for_evaluation(self) except *:
        return len(self.__unevaluated_phenotypes) == 0

    cpdef action(self, observation):
        return (
            (<BasePhenotype?>self.__acting_phenotype).action(observation))

    cpdef void receive_feedback(self, feedback) except *:
        (<BasePhenotype?>self.__acting_phenotype).receive_feedback(feedback)

    cpdef void update_policy(self) except *:
        cdef BasePhenotype acting_phenotype
        cdef double acting_fitness
        cdef double best_epoch_fitness

        acting_phenotype = <BasePhenotype?>self.__acting_phenotype
        acting_fitness = acting_phenotype.fitness()
        best_epoch_fitness = self.best_phenotype().fitness()

        # Update the best phenotype.
        if acting_fitness > best_epoch_fitness:
            self._set_best_phenotype(acting_phenotype)

        # Remove current evaluated phenotype.
        self.__unevaluated_phenotypes.pop()

        if len(self.__unevaluated_phenotypes) > 0:
            # Get new phenotype to evaluate.
            self._set_acting_phenotype(self.__unevaluated_phenotypes[-1])
        elif self.__n_epochs_elapsed < self.__max_n_epochs:
            # The epoch has ended, operate of phenotype population.
            # Set the policy to the best phenotype for evaluation.
            self._set_acting_phenotype(self.__best_phenotype)
            self.operate()
            self._set_n_epochs_elapsed(self.__n_epochs_elapsed + 1)

    cpdef void operate(self) except *:
        raise NotImplementedError("Abstract method.")

    cpdef void receive_score(self, score) except *:
        pass

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        pass

    cpdef Py_ssize_t n_phenotypes(self) except *:
          return len(self.__phenotypes)

    cpdef void append_phenotype(self, phenotype) except *:
        self.__phenotypes.append(<BasePhenotype?>phenotype)

    cpdef pop_phenotype(self, Py_ssize_t index):
        return self.__phenotypes.pop(index)

    cpdef void insert_phenotype(self, Py_ssize_t index, phenotype) except *:
        self.__phenotypes.insert(index, <BasePhenotype?>phenotype)

    cpdef phenotype(self, Py_ssize_t index):
        return self.__phenotypes[index]

    cpdef void set_phenotype(self, Py_ssize_t index, phenotype) except *:
        self.__phenotypes[index] = <BasePhenotype?>phenotype

    cpdef list _phenotypes(self):
        return self.__phenotypes

    cpdef list phenotypes_shallow_copy(self):
        cdef list phenotypes_copy
        cdef Py_ssize_t phenotype_id

        phenotypes_copy = [None] * len(self.__phenotypes)

        for phenotype_id in range(len(self.__phenotypes)):
            phenotypes_copy[phenotype_id] = self.__phenotypes[phenotype_id]

        return phenotypes_copy

    cpdef list phenotypes_deep_copy(self):
        cdef list phenotypes_copy
        cdef Py_ssize_t phenotype_id
        cdef BasePhenotype phenotype

        phenotypes_copy = [None] * len(self.__phenotypes)

        for phenotype_id in range(len(self.__phenotypes)):
            phenotype = self.__phenotypes[phenotype_id]
            phenotypes_copy[phenotype_id] = phenotype.copy()

        return phenotypes_copy

    cpdef void set_phenotypes(self, list phenotypes) except *:
        cdef Py_ssize_t phenotype_id
        cdef BasePhenotype phenotype

        for phenotype_id in range(len(phenotypes)):
            phenotype = phenotypes[phenotype_id]
            if not isinstance(phenotype, BasePhenotype):
                raise (
                    TypeError(
                        "All objects in (phenotypes) must be instances of "
                        "BasePhenotype. (type(phenotypes[{phenotype_id}]) = "
                        "{phenotype.__class__})."
                        .format(**locals()) ))

        self.__phenotypes = phenotypes

    cpdef list _unevaluated_phenotypes(self):
        return self.__unevaluated_phenotypes

    cpdef list unevaluated_phenotypes_shallow_copy(self):
        cdef list unevaluated_phenotypes_copy
        cdef Py_ssize_t unevaluated_phenotype_id

        unevaluated_phenotypes_copy = (
            [None] * len(self.__unevaluated_phenotypes))

        for unevaluated_phenotype_id in range(len(self.__unevaluated_phenotypes)):
            unevaluated_phenotypes_copy[unevaluated_phenotype_id] = self.__unevaluated_phenotypes[unevaluated_phenotype_id]

        return unevaluated_phenotypes_copy

    cpdef list unevaluated_phenotypes_deep_copy(self):
        cdef list unevaluated_phenotypes_copy
        cdef Py_ssize_t unevaluated_phenotype_id
        cdef BasePhenotype unevaluated_phenotype

        unevaluated_phenotypes_copy = (
            [None] * len(self.__unevaluated_phenotypes))

        for unevaluated_phenotype_id in range(len(self.__unevaluated_phenotypes)):
            unevaluated_phenotype = (
                self.__unevaluated_phenotypes[unevaluated_phenotype_id])
            unevaluated_phenotypes_copy[unevaluated_phenotype_id] = (
                unevaluated_phenotype.copy())

        return unevaluated_phenotypes_copy

    cpdef void set_unevaluated_phenotypes(
            self,
            list unevaluated_phenotypes
            ) except *:
        cdef Py_ssize_t unevaluated_phenotype_id
        cdef BasePhenotype unevaluated_phenotype

        for unevaluated_phenotype_id in range(len(unevaluated_phenotypes)):
            unevaluated_phenotype = unevaluated_phenotypes[unevaluated_phenotype_id]
            if not isinstance(unevaluated_phenotype, BasePhenotype):
                raise (
                    TypeError(
                        "All objects in (unevaluated_phenotypes) must be instances of "
                        "BasePhenotype. (type(unevaluated_phenotypes[{unevaluated_phenotype_id}]) = "
                        "{unevaluated_phenotype.__class__})."
                        .format(**locals()) ))

        self.__unevaluated_phenotypes = unevaluated_phenotypes

    cpdef Py_ssize_t max_n_epochs(self) except *:
        return self.__max_n_epochs

    cpdef void set_max_n_epochs(self, Py_ssize_t max_n_epochs) except*:
        if max_n_epochs < self.__n_epochs_elapsed:
            raise (
                ValueError(
                    "The new maximum number of epochs (max_n_epochs = "
                    "{max_n_epochs}) must not be less than the current "
                    "number of epochs elapsed (self.n_epochs_elapsed() = "
                    "{self.__n_epochs_elapsed})."
                    .format(**locals())))

        if max_n_epochs <= 0:
            raise (
                ValueError(
                    "The new maximum number of epochs (max_n_epochs = "
                    "{max_n_epochs}) must be positive."
                    .format(**locals())))

        self.__max_n_epochs = max_n_epochs


    cpdef Py_ssize_t n_epochs_elapsed(self) except *:
        return self.__n_epochs_elapsed

    cpdef void _set_n_epochs_elapsed(self, Py_ssize_t n_epochs_elapsed) except*:
        if n_epochs_elapsed < 0:
            raise (
                ValueError(
                    "The number epochs elapsed (n_epochs_elapsed = "
                    "{n_epochs_elapsed}) must be non-negative."
                    .format(**locals())))

        self.__n_epochs_elapsed = n_epochs_elapsed

    cpdef best_phenotype(self):
        return self.__best_phenotype

    cpdef void _set_best_phenotype(self, phenotype) except *:
        self.__best_phenotype = <BasePhenotype?>phenotype

    cpdef acting_phenotype(self):
        return  self.__acting_phenotype

    cpdef void _set_acting_phenotype(self, phenotype) except *:
        self.__acting_phenotype = <BasePhenotype?>phenotype

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

    system.__phenotypes = []
    system.__unevaluated_phenotypes = []
    system.__best_phenotype = None
    system.__acting_phenotype = None
    system.__max_n_epochs = 1
    system.__n_epochs_elapsed = 0

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class DefaultEvolvingSystem:
    def __init__(self):
        init_DefaultEvolvingSystem(self)

    cpdef copy(self, copy_obj = None):
        cdef DefaultEvolvingSystem new_system

        if copy_obj is None:
            new_system = DefaultEvolvingSystem.__new__(DefaultEvolvingSystem)
        else:
            new_system = copy_obj

        new_system = BaseEvolvingSystem.copy(self, new_system)

        return new_system

    cpdef void operate(self) except *:
        cdef Py_ssize_t match_id
        cdef BasePhenotype contender_a
        cdef BasePhenotype contender_b
        cdef double fitness_a
        cdef double fitness_b
        cdef list phenotypes

        phenotypes = self._phenotypes()

        # TODO Optimize random shuffle with non-python random shuffle.
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





