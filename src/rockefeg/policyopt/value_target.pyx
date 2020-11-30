cimport cython

from .function_approximation cimport TargetEntry, new_TargetEntry
from .experience cimport ExperienceDatum
from rockefeg.cyutil.array cimport new_DoubleArray
from rockefeg.cyutil.typed_list cimport BaseReadableTypedList, new_TypedList
from rockefeg.cyutil.typed_list cimport is_sub_full_type, TypedList

@cython.warn.undeclared(True)
cpdef DoubleArray concatenation_of_DoubleArray(
        DoubleArray arr1,
        DoubleArray arr2):
    cdef DoubleArray concatenation
    cdef Py_ssize_t arr1_size
    cdef Py_ssize_t arr2_size
    cdef Py_ssize_t id

    arr1_size = len(arr1)
    arr2_size = len(arr2)
    concatenation = new_DoubleArray(arr1_size + arr2_size)

    for id in range(arr1_size):
        concatenation.view[id] = arr1.view[id]

    for id in range(arr2_size):
        concatenation.view[id + arr1_size] = arr2.view[id]

    return concatenation


@cython.warn.undeclared(True)
cpdef DoubleArray rewards_from_path(BaseReadableTypedList path):
    cdef ExperienceDatum experience
    cdef Py_ssize_t experience_id
    cdef DoubleArray rewards
    cdef object path_item_type

    path_item_type = path.item_type()

    if not is_sub_full_type(path_item_type, ExperienceDatum):
        raise (
            TypeError(
                "The path list's item type "
                "(path.item_type() = {path_item_type}) "
                "must be a subtype of ExperienceDatum."
                .format(**locals())))

    rewards = new_DoubleArray(len(path))

    for experience_id in range(len(path)):
        experience = path.item(experience_id)
        rewards.view[experience_id] = experience.reward

    return rewards

@cython.warn.undeclared(True)
cpdef DoubleArray q_value_evals(
        BaseReadableTypedList path,
        BaseMap critic):
    cdef ExperienceDatum experience
    cdef Py_ssize_t experience_id
    cdef DoubleArray q_values
    cdef DoubleArray critic_eval_array
    cdef object path_item_type

    if path is None:
        raise TypeError("The path (path) cannot be None.")

    if critic is None:
        raise TypeError("The critic (critic) cannot be None.")

    path_item_type =  path.item_type()

    if not is_sub_full_type(path.item_type(), ExperienceDatum):
        raise (
            TypeError(
                "The path list's item type "
                "(path.item_type() = {path_item_type}) "
                "must be a subtype of ExperienceDatum."
                .format(**locals())))

    q_values = new_DoubleArray(len(path))

    for experience_id in range(len(path)):
        experience = path.item(experience_id)
        critic_eval_array = critic.eval(experience)
        q_values.view[experience_id] = critic_eval_array.view[0]

    return q_values

@cython.warn.undeclared(True)
cpdef DoubleArray sarsa_q_update_evals(
        DoubleArray rewards,
        DoubleArray q_values,
        double discount_factor,
        bint using_terminal_step ):
    cdef Py_ssize_t n_rewards
    cdef Py_ssize_t n_q_values
    cdef Py_ssize_t n_q_updates
    cdef Py_ssize_t update_id
    cdef double next_q_value
    cdef DoubleArray q_updates

    if rewards is None:
        raise TypeError("The rewards vector (rewards) cannot be None.")

    if q_values is None:
        raise TypeError("The Q values vector (q_values) cannot be None.")

    n_rewards  = len(rewards)
    n_q_values = len(q_values)

    if n_rewards != n_q_values:
        raise (
            ValueError(
                "The number of rewards (len(rewards) = {n_rewards}) "
                "must be equal to the number of Q values "
                "(len(q_values) = {n_q_values})."
                .format(**locals()) ))

    n_q_updates = n_q_values
    if not using_terminal_step:
        n_q_updates -= 1

    if n_q_updates <= 0:
        raise (
            ValueError(
                "The number of rewards (len(rewards) = {n_rewards}) "
                "and the number Q values (len(q_values) = {n_q_values}) "
                "must be positive if using terminal step "
                "(using_terminal_step = {using_terminal_step}) and greater "
                "than 1 if not using terminal step. "
                .format(**locals())))

    q_updates = new_DoubleArray(n_q_updates)

    for update_id in range(n_q_updates):
        # Avoid out of bounds error when using terminal step.
        if update_id + 1 == len(q_values):
            next_q_value = 0.
        else:
            next_q_value = q_values.view[update_id + 1]

        q_updates.view[update_id] = (
            q_values.view[update_id]
            - rewards.view[update_id]
            - discount_factor
            * next_q_value )

    return q_updates

@cython.warn.undeclared(True)
cpdef DoubleArray trace_convolution_while_redistributing_tail(
        DoubleArray q_updates,
        DoubleArray trace,
        Py_ssize_t min_lookahead,
        double discount_factor):
    cdef DoubleArray convolution
    cdef Py_ssize_t trace_id
    cdef Py_ssize_t convolution_id
    cdef Py_ssize_t trace_len

    if q_updates is None:
        raise TypeError("The Q updates vector (q_updates) cannot be None.")

    if trace is None:
        raise TypeError("The trace vector (trace) cannot be None.")

    if len(trace) > len(q_updates):
        trace = tail_subtracted_trace(trace, len(q_updates), discount_factor)

    convolution = new_DoubleArray(len(q_updates) - min_lookahead)
    convolution.set_all_to(0.)

    for convolution_id in range(len(convolution)):

        if len(convolution) - convolution_id < len(trace):
            trace = (
                tail_subtracted_trace(
                    trace,
                    len(convolution) - convolution_id,
                    discount_factor))

        trace_len = len(trace)

        for trace_id in range(len(trace)):
            if convolution_id + trace_id < len(q_updates):
                convolution.view[convolution_id] += (
                    trace.view[trace_id]
                    * q_updates.view[convolution_id + trace_id]
                    / trace_len )
            else:
                raise (
                    RuntimeError(
                        "Something went wrong, the trace is too long."))

    return convolution

@cython.warn.undeclared(True)
cpdef DoubleArray trace_convolution(
        DoubleArray q_updates,
        DoubleArray trace,
        Py_ssize_t min_lookahead):
    cdef DoubleArray convolution
    cdef Py_ssize_t trace_id
    cdef Py_ssize_t convolution_id

    if q_updates is None:
        raise TypeError("The Q updates vector (q_updates) cannot be None.")

    if trace is None:
        raise TypeError("The trace vector (trace) cannot be None.")

    convolution = new_DoubleArray(len(q_updates) - min_lookahead)
    convolution.set_all_to(0.)

    for convolution_id in range(len(convolution)):
        for trace_id in range(len(trace)):
            if convolution_id + trace_id < len(q_updates):
                convolution.view[convolution_id] += (
                    trace.view[trace_id]
                    * q_updates.view[convolution_id + trace_id] )
            else:
                break

    return convolution


@cython.warn.undeclared(True)
cpdef DoubleArray trace_convolution_while_normalizing_variance(
        DoubleArray q_updates,
        DoubleArray trace,
        Py_ssize_t min_lookahead,
        double discount_factor):
    cdef DoubleArray convolution
    cdef Py_ssize_t trace_id
    cdef Py_ssize_t convolution_id

    if q_updates is None:
        raise TypeError("The Q updates vector (q_updates) cannot be None.")

    if trace is None:
        raise TypeError("The trace vector (trace) cannot be None.")

    if len(trace) > len(q_updates):
        trace = tail_subtracted_trace(trace, len(q_updates), discount_factor)

    convolution = new_DoubleArray(len(q_updates) - min_lookahead)
    convolution.set_all_to(0.)

    for convolution_id in range(len(convolution)):

        if len(convolution) - convolution_id < len(trace):
            trace = (
                tail_subtracted_trace(
                    trace,
                    len(convolution) - convolution_id,
                    discount_factor))


        for trace_id in range(len(trace)):
            if convolution_id + trace_id < len(q_updates):
                convolution.view[convolution_id] += (
                    trace.view[trace_id]
                    * q_updates.view[convolution_id + trace_id] )
            else:
                raise (
                    RuntimeError(
                        "Something went wrong, the trace is too long."))

    return convolution

@cython.warn.undeclared(True)
cpdef DoubleArray tail_redistributed_trace(
        DoubleArray trace,
        Py_ssize_t new_trace_len,
        double discount_factor):
    cdef DoubleArray new_trace
    cdef Py_ssize_t new_trace_id
    cdef Py_ssize_t trace_len
    cdef double T_0
    cdef double normalization_factor

    if trace is None:
        raise TypeError("The trace vector (trace) cannot be None.")

    trace_len = len(trace)

    if new_trace_len > trace_len:
        raise (
            TypeError(
                "The new trace length (new_trace_len = {new_trace_len}) "
                "must not be greater than the trace length "
                "(len(trace) = {trace_len})."
                .format(**locals())))

    if new_trace_len == trace_len:
        return trace.copy()

    T_0 = trace.view[0]
    new_trace = tail_subtracted_trace(trace, new_trace_len, discount_factor)
    normalization_factor = T_0 / new_trace.view[0]

    # Renormalize trace after subtracting tail.
    for new_trace_id in range(len(new_trace)):
        new_trace.view[new_trace_id] *= normalization_factor

    return new_trace

@cython.warn.undeclared(True)
cpdef DoubleArray tail_subtracted_trace(
        DoubleArray trace,
        Py_ssize_t new_trace_len,
        double discount_factor):
    cdef DoubleArray new_trace
    cdef Py_ssize_t new_trace_id
    cdef Py_ssize_t trace_len
    cdef double R_n_weight
    cdef double T_0
    cdef double T_n
    cdef double discount_weight

    if trace is None:
        raise TypeError("The trace vector (trace) cannot be None.")

    trace_len = len(trace)

    if new_trace_len > trace_len:
        raise (
            TypeError(
                "The new trace length (new_trace_len = {new_trace_len}) "
                "must not be greater than the trace length "
                "(len(trace) = {trace_len})."
                .format(**locals())))

    if new_trace_len == trace_len:
        return trace.copy()

    new_trace = new_DoubleArray(new_trace_len)

    T_0 = trace.view[0]
    T_n = trace.view[new_trace_len]
    R_n_weight = T_n * discount_factor ** -(new_trace_len)

    for new_trace_id in range(len(new_trace)):
        discount_weight = discount_factor ** new_trace_id

        new_trace.view[new_trace_id] = (
            (
            trace.view[new_trace_id]
            / discount_weight
            - R_n_weight
            )
            * discount_weight )

    return new_trace


@cython.warn.undeclared(True)
cpdef value_target_entries(
        BaseReadableTypedList path,
        DoubleArray target_values):
    cdef TypedList entries_typed_list
    cdef list entries
    cdef Py_ssize_t entry_id
    cdef DoubleArray target
    cdef TargetEntry target_entry
    cdef object path_item_type

    if path is None:
        raise TypeError("The path (path) cannot be None.")

    if target_values is None:
        raise (
            TypeError(
                "The target values vector (target_values) cannot be None."))

    path_item_type =  path.item_type()

    if not is_sub_full_type(path.item_type(), ExperienceDatum):
        raise (
            TypeError(
                "The path list's item type "
                "(path.item_type() = {path_item_type}) "
                "must be a subtype of ExperienceDatum."
                .format(**locals())))

    entries = [None] * len(path)

    for entry_id in range(len(entries)):
        target = new_DoubleArray(1)
        target.view[0] = target_values.view[entry_id]
        target_entry = new_TargetEntry()
        target_entry.input = path.item(entry_id)
        target_entry.target = target
        entries[entry_id] = target_entry

    entries_typed_list = new_TypedList(TargetEntry)
    entries_typed_list.set_items(entries)

    return entries_typed_list

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseValueTargetSetter:
    cpdef BaseValueTargetSetter copy(self, copy_obj = None):
        pass

    cpdef TypedList value_target_entries(self, BaseReadableTypedList path):
        raise NotImplementedError("Abstract method.")

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TotalRewardTargetSetter(BaseValueTargetSetter):
    def __init__(self):
        init_TotalRewardTargetSetter(self)

    cpdef TotalRewardTargetSetter copy(self, copy_obj = None):
        cdef TotalRewardTargetSetter new_target_setter

        if copy_obj is None:
            new_target_setter = (
                TotalRewardTargetSetter.__new__(
                    TotalRewardTargetSetter))
        else:
            new_target_setter = copy_obj

        return new_target_setter

    cpdef TypedList value_target_entries(self, BaseReadableTypedList path):
        cdef Py_ssize_t experience_id
        cdef list value_target_entries_list
        cdef BaseReadableTypedList value_target_entries_ret
        cdef DoubleArray target_values
        cdef double total_reward
        cdef ExperienceDatum experience #
        cdef object path_item_type

        path_item_type = path.item_type()

        if not is_sub_full_type(path_item_type, ExperienceDatum):
            raise (
                TypeError(
                    "The path list's item type "
                    "(path.item_type() = {path_item_type}) "
                    "must be a subtype of ExperienceDatum."
                    .format(**locals())))

        # Add up all rewards from both paths
        total_reward = 0
        for experience in path:
            total_reward += experience.reward

        target_values = new_DoubleArray(len(path))
        target_values.set_all_to(total_reward)

        value_target_entries_ret = (
            value_target_entries(
                path,
                target_values ))

        return value_target_entries_ret

@cython.warn.undeclared(True)
cdef TotalRewardTargetSetter new_TotalRewardTargetSetter():
    cdef TotalRewardTargetSetter target_setter

    target_setter = TotalRewardTargetSetter.__new__(TotalRewardTargetSetter)
    init_TotalRewardTargetSetter(target_setter)

    return target_setter

@cython.warn.undeclared(True)
cdef void init_TotalRewardTargetSetter(
        TotalRewardTargetSetter
        target_setter
        ) except *:
    if target_setter is None:
        raise (
            TypeError(
                "The target setter (target_setter) cannot be None." ))



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class BaseTdTargetSetter(BaseValueTargetSetter):

    def __init__(self):
        init_BaseTdTargetSetter(self)

    cpdef BaseTdTargetSetter copy(self, copy_obj = None):
        cdef BaseTdTargetSetter new_target_setter

        if copy_obj is None:
            new_target_setter = BaseTdTargetSetter.__new__(BaseTdTargetSetter)
        else:
            new_target_setter = copy_obj

        new_target_setter.__discount_factor = self.__discount_factor
        new_target_setter.__critic = self.__critic
        new_target_setter.__min_lookahead = self.__min_lookahead
        new_target_setter.__forces_only_min_lookahead = (
            self.__forces_only_min_lookahead)
        new_target_setter.__uses_terminal_step = self.__uses_terminal_step

        return new_target_setter

    cpdef DoubleArray trace(self, Py_ssize_t trace_len):
        raise NotImplementedError("Abstract method.")


    cpdef TypedList value_target_entries(self, BaseReadableTypedList path):
        cdef DoubleArray trace
        cdef DoubleArray q_values
        cdef DoubleArray rewards
        cdef DoubleArray q_updates
        cdef DoubleArray target_updates
        cdef DoubleArray target_values
        cdef BaseReadableTypedList value_target_entries_ret
        cdef Py_ssize_t trace_len
        cdef Py_ssize_t target_id
        cdef Py_ssize_t path_len
        cdef Py_ssize_t n_q_updates
        cdef Py_ssize_t min_lookahead
        cdef object path_item_type

        path_item_type = path.item_type()

        if not is_sub_full_type(path_item_type, ExperienceDatum):
            raise (
                TypeError(
                    "The path list's item type "
                    "(path.item_type() = {path_item_type}) "
                    "must be a subtype of ExperienceDatum."
                    .format(**locals())))

        path_len = len(path)
        if self.uses_terminal_step():
            n_q_updates = path_len
        else:
            n_q_updates = path_len - 1

        min_lookahead = self.min_lookahead()
        if n_q_updates < min_lookahead + 1:
            raise (
                IndexError(
                    "The number of Q updates to calculate (len(path) - 1 + "
                    "self.uses_terminal_step() = {n_q_updates}) "
                    "must not be less "
                    "than 1 plus the minimum lookahead for target updates  "
                    "(self.min_lookahead() = {min_lookahead})."
                    .format(**locals())))

        # Calculate the trace length.
        if self.forces_only_min_lookahead():
            trace_len = min_lookahead + 1
        else:
            trace_len = n_q_updates
        #
        trace = self.trace(trace_len)

        rewards = rewards_from_path(path)

        q_values = q_value_evals(path, self.critic())

        q_updates = (
            sarsa_q_update_evals(
                rewards,
                q_values,
                self.discount_factor(),
                self.uses_terminal_step() ))

        target_updates = (
            trace_convolution(
                q_updates,
                trace,
                min_lookahead))

        target_values = new_DoubleArray(n_q_updates - min_lookahead)

        for target_id in range(len(target_values)):
            target_values.view[target_id] = (
                q_values.view[target_id]
                + target_updates.view[target_id])

        value_target_entries_ret = (
            value_target_entries(
                path,
                target_values ))

        return value_target_entries_ret

    cpdef double discount_factor(self) except *:
        return self.__discount_factor

    cpdef void set_discount_factor(self, double discount_factor) except *:
        self.__discount_factor = discount_factor

    cpdef BaseMap critic(self):
        return self.__critic

    cpdef void set_critic(self, BaseMap critic) except *:
        self.__critic = critic

    cpdef Py_ssize_t min_lookahead(self) except *:
        return self.__min_lookahead

    cpdef void set_min_lookahead(self, Py_ssize_t min_lookahead) except *:
        if min_lookahead < 0:
            raise (
                ValueError(
                    "The minimum lookahead for target updates "
                    "(min_lookahead = {min_lookahead}) must be non-negative."
                    .format(**locals()) ))
        self.__min_lookahead = min_lookahead

    cpdef bint forces_only_min_lookahead(self) except *:
        return self.__forces_only_min_lookahead

    cpdef void set_forces_only_min_lookahead(
            self,
            bint forces_only_min_lookahead
            ) except *:
        self.__forces_only_min_lookahead = forces_only_min_lookahead


    cpdef bint uses_terminal_step(self) except *:
        return self.__uses_terminal_step


    cpdef void set_uses_terminal_step(self, bint uses_terminal_step) except *:
        self.__uses_terminal_step = uses_terminal_step




@cython.warn.undeclared(True)
cdef void init_BaseTdTargetSetter(BaseTdTargetSetter target_setter) except *:
    if target_setter is None:
        raise TypeError("The target_setter (target_setter) cannot be None.")

    target_setter.__discount_factor = 1.
    target_setter.__min_lookahead = 0
    target_setter.__critic = None # Must be initialized later.
    target_setter.__forces_only_min_lookahead = False
    target_setter.__uses_terminal_step = True

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TdLambdaTargetSetter(BaseTdTargetSetter):
    def __init__(self, BaseMap critic):
        init_TdLambdaTargetSetter(self, critic)

    cpdef TdLambdaTargetSetter copy(self, copy_obj = None):
        cdef TdLambdaTargetSetter new_target_setter

        if copy_obj is None:
            new_target_setter = (
                TdLambdaTargetSetter.__new__(
                    TdLambdaTargetSetter))
        else:
            new_target_setter = copy_obj

        BaseTdTargetSetter.copy(self, new_target_setter)
        new_target_setter.__trace_decay = self.__trace_decay
        new_target_setter.__redistributes_trace_tail = (
            self.__redistributes_trace_tail
        )

        return new_target_setter

    cpdef DoubleArray trace(self, Py_ssize_t trace_len):
        cdef DoubleArray trace
        cdef double discount_factor
        cdef double decay_discount_rate
        cdef Py_ssize_t trace_id

        if trace_len <= 0:
            raise (
                ValueError(
                    "The trace length "
                    "(trace_len = {trace_len}) must be positive."
                    .format(**locals()) ))

        discount_factor = self.discount_factor()
        decay_discount_rate = self.trace_decay() * discount_factor


        if self.redistributes_trace_tail():
            trace = new_DoubleArray(trace_len + 1)
        else:
            trace = new_DoubleArray(trace_len )

        for trace_id in range(len(trace)):
            if trace_id == 0:
                trace.view[trace_id] = 1.
            else:
                trace.view[trace_id] = decay_discount_rate ** trace_id

        if self.redistributes_trace_tail():
            trace = tail_redistributed_trace(trace, trace_len, discount_factor)

        return trace

    # TODO: backward view for TD eligibity traces (more efficient)

    cpdef TypedList value_target_entries(self, BaseReadableTypedList path):
        cdef DoubleArray trace
        cdef DoubleArray q_values
        cdef DoubleArray rewards
        cdef DoubleArray q_updates
        cdef DoubleArray target_updates
        cdef DoubleArray target_values
        cdef BaseReadableTypedList value_target_entries_ret
        cdef Py_ssize_t trace_len
        cdef Py_ssize_t target_id
        cdef Py_ssize_t path_len
        cdef Py_ssize_t n_q_updates
        cdef Py_ssize_t min_lookahead
        cdef object path_item_type

        path_item_type = path.item_type()

        if not is_sub_full_type(path_item_type, ExperienceDatum):
            raise (
                TypeError(
                    "The path list's item type "
                    "(path.item_type() = {path_item_type}) "
                    "must be a subtype of ExperienceDatum."
                    .format(**locals())))

        if not self.redistributes_trace_tail():
            return BaseTdTargetSetter.value_target_entries(self, path)

        else:
            min_lookahead = self.min_lookahead()
            path_len = len(path)
            if self.uses_terminal_step():
                n_q_updates = path_len
            else:
                n_q_updates = path_len - 1

            if n_q_updates < min_lookahead + 1:
                raise (
                    IndexError(
                        "The number of Q updates to calculate (len(path) - 1 + "
                        "self.uses_terminal_step() = {n_q_updates}) "
                        "must not be less "
                        "than 1 plus the minimum lookahead for target updates  "
                        "(self.min_lookahead() = {min_lookahead})."
                        .format(**locals())))

            # Calculate the trace length.
            if self.forces_only_min_lookahead():
                trace_len = min_lookahead + 1
            else:
                trace_len = n_q_updates
            #
            trace = self.trace(trace_len)

            rewards = rewards_from_path(path)

            q_values = q_value_evals(path, self.critic())

            q_updates = (
                sarsa_q_update_evals(
                    rewards,
                    q_values,
                    self.discount_factor(),
                    self.uses_terminal_step() ))

            target_updates = (
                trace_convolution_while_redistributing_tail(
                    q_updates,
                    trace,
                    min_lookahead,
                    self.discount_factor() ))

            target_values = new_DoubleArray(n_q_updates - min_lookahead)

            for target_id in range(len(target_values)):
                target_values.view[target_id] = (
                    q_values.view[target_id]
                    + target_updates.view[target_id])

            value_target_entries_ret = (
                value_target_entries(
                    path,
                    target_values ))

            return value_target_entries_ret

    cpdef double trace_decay(self) except *:
        return self.__trace_decay


    cpdef void set_trace_decay(self, double trace_decay) except *:
        self.__trace_decay = trace_decay

    cpdef bint redistributes_trace_tail(self)  except *:
        return self.__redistributes_trace_tail

    cpdef void set_redistributes_trace_tail(
            self,
            bint redistributes_trace_tail
            ) except *:
        self.__redistributes_trace_tail = redistributes_trace_tail



@cython.warn.undeclared(True)
cdef TdLambdaTargetSetter new_TdLambdaTargetSetter(BaseMap critic):
    cdef TdLambdaTargetSetter target_setter

    target_setter = TdLambdaTargetSetter.__new__(TdLambdaTargetSetter)
    init_TdLambdaTargetSetter(target_setter, critic)

    return target_setter

@cython.warn.undeclared(True)
cdef void init_TdLambdaTargetSetter(
        TdLambdaTargetSetter target_setter,
        BaseMap critic
        ) except *:
    if target_setter is None:
        raise TypeError("The target setter (target_setter) cannot be None.")

    if critic is None:
        raise TypeError("The critic (critic) cannot be None.")

    init_BaseTdTargetSetter(target_setter)
    target_setter.__critic = critic
    target_setter.__trace_decay = 0.
    target_setter.__redistributes_trace_tail = False



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TdHeavyTargetSetter(BaseTdTargetSetter):
    def __init__(self, BaseMap critic):
        init_TdHeavyTargetSetter(self, critic)

    cpdef TdHeavyTargetSetter copy(self, copy_obj = None):
        cdef TdHeavyTargetSetter new_target_setter

        if copy_obj is None:
            new_target_setter = TdHeavyTargetSetter.__new__(TdHeavyTargetSetter)
        else:
            new_target_setter = copy_obj

        BaseTdTargetSetter.copy(self, new_target_setter)
        new_target_setter.__covariance_factor = self.__covariance_factor
        new_target_setter.__normalizes_trace_variance = (
            self.__normalizes_trace_variance)

        return new_target_setter

    cpdef DoubleArray trace(self, Py_ssize_t trace_len):
        cdef DoubleArray trace
        cdef DoubleArray stdev_factors
        cdef double stdev_factor
        cdef double sum_inv_stdev_factors
        cdef double discount_factor
        cdef double normalization_factor
        cdef double trace_value
        cdef double covariance_factor
        cdef Py_ssize_t trace_id
        cdef Py_ssize_t stdev_id

        if trace_len <= 0:
            raise (
                ValueError(
                    "The trace length "
                    "(trace_len = {trace_len}) must be positive."
                    .format(**locals()) ))

        stdev_factors = new_DoubleArray(trace_len)

        discount_factor = self.discount_factor()

        covariance_factor = self.covariance_factor()

        stdev_factor = 0.
        for stdev_id in range(len(stdev_factors)):
            stdev_factor = (
                (
                1.
                + stdev_factor * stdev_factor
                + 2. * stdev_factor * covariance_factor
                )
                ** 0.5 )
            stdev_factors.view[stdev_id] = stdev_factor

        sum_inv_stdev_factors = 0.
        for stdev_id in range(len(stdev_factors)):
            sum_inv_stdev_factors += 1. / stdev_factors.view[stdev_id]

        trace = new_DoubleArray(trace_len)
        trace_value = sum_inv_stdev_factors
        for trace_id in range(trace_len):
            stdev_id = trace_id
            trace.view[trace_id] = trace_value * (discount_factor ** trace_id)
            trace_value -= (
                1. / stdev_factors.view[stdev_id])

        if not self.normalizes_trace_variance():
            normalization_factor = 1. / trace.view[0]
            for trace_id in range(trace_len):
                trace.view[trace_id] *= normalization_factor

        return trace

    cpdef TypedList value_target_entries(self, BaseReadableTypedList path):
        cdef DoubleArray trace
        cdef DoubleArray q_values
        cdef DoubleArray rewards
        cdef DoubleArray q_updates
        cdef DoubleArray target_updates
        cdef DoubleArray target_values
        cdef BaseReadableTypedList value_target_entries_ret
        cdef Py_ssize_t trace_len
        cdef Py_ssize_t target_id
        cdef Py_ssize_t path_len
        cdef Py_ssize_t n_q_updates
        cdef Py_ssize_t min_lookahead
        cdef object path_item_type

        path_item_type = path.item_type()

        if not is_sub_full_type(path_item_type, ExperienceDatum):
            raise (
                TypeError(
                    "The path list's item type "
                    "(path.item_type() = {path_item_type}) "
                    "must be a subtype of ExperienceDatum."
                    .format(**locals())))

        if not self.normalizes_trace_variance():
            return BaseTdTargetSetter.value_target_entries(self, path)

        else:
            min_lookahead = self.min_lookahead()
            path_len = len(path)
            if self.uses_terminal_step():
                n_q_updates = path_len
            else:
                n_q_updates = path_len - 1

            if n_q_updates < min_lookahead + 1:
                raise (
                    IndexError(
                        "The number of Q updates to calculate (len(path) - 1 + "
                        "self.uses_terminal_step() = {n_q_updates}) "
                        "must not be less "
                        "than 1 plus the minimum lookahead for target updates  "
                        "(self.min_lookahead() = {min_lookahead})."
                        .format(**locals())))

            # Calculate the trace length.
            if self.forces_only_min_lookahead():
                trace_len = min_lookahead + 1
            else:
                trace_len = n_q_updates
            #
            trace = self.trace(trace_len)

            rewards = rewards_from_path(path)

            q_values = q_value_evals(path, self.critic())

            q_updates = (
                sarsa_q_update_evals(
                    rewards,
                    q_values,
                    self.discount_factor(),
                    self.uses_terminal_step() ))

            target_updates = (
                trace_convolution_while_normalizing_variance(
                    q_updates,
                    trace,
                    min_lookahead,
                    self.discount_factor()))

            target_values = new_DoubleArray(n_q_updates - min_lookahead)

            for target_id in range(len(target_values)):
                target_values.view[target_id] = (
                    q_values.view[target_id]
                    + target_updates.view[target_id])

            value_target_entries_ret = (
                value_target_entries(
                    path,
                    target_values ))

            return value_target_entries_ret


    cpdef double covariance_factor(self) except *:
        return self.__covariance_factor

    cpdef void set_covariance_factor(self, covariance_factor) except *:
        self.__covariance_factor = covariance_factor

    cpdef bint normalizes_trace_variance(self)  except *:
        return self.__normalizes_trace_variance

    cpdef void set_normalizes_trace_variance(
            self,
            bint normalizes_trace_variance
            ) except *:
        self.__normalizes_trace_variance = normalizes_trace_variance


@cython.warn.undeclared(True)
cdef TdHeavyTargetSetter new_TdHeavyTargetSetter(BaseMap critic):
    cdef TdHeavyTargetSetter target_setter

    target_setter = TdHeavyTargetSetter.__new__(TdHeavyTargetSetter)
    init_TdHeavyTargetSetter(target_setter, critic)

    return target_setter

@cython.warn.undeclared(True)
cdef void init_TdHeavyTargetSetter(
        TdHeavyTargetSetter target_setter,
        BaseMap critic
        ) except *:
    if target_setter is None:
        raise TypeError("The target setter (target_setter) cannot be None.")

    if critic is None:
        raise TypeError("The critic (critic) cannot be None.")

    init_BaseTdTargetSetter(target_setter)
    target_setter.__critic = critic
    target_setter.__covariance_factor = 1.
    target_setter.__normalizes_trace_variance = True





