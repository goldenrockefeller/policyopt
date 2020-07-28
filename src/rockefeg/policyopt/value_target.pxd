
from .map cimport BaseMap
from rockefeg.cyutil.array cimport DoubleArray

cpdef DoubleArray concatenation_of_DoubleArray(
    DoubleArray arr1,
    DoubleArray arr2)

cdef class BaseValueTargetSetter:
    cpdef copy(self, copy_obj = ?)

    cpdef value_target_entries(self, path)


cdef class TotalRewardTargetSetter(BaseValueTargetSetter):
    pass

cdef TotalRewardTargetSetter new_TotalRewardTargetSetter()
cdef void init_TotalRewardTargetSetter(
    TotalRewardTargetSetter target_setter
    ) except *


cdef class BaseTdTargetSetter(BaseValueTargetSetter):
    cdef double __discount_factor
    cdef object __critic
    cdef Py_ssize_t __min_lookahead
    cdef bint __forces_only_min_lookahead
    cdef bint __uses_terminal_step

    cpdef trace(self, Py_ssize_t trace_len)

    cpdef double discount_factor(self) except *
    cpdef void set_discount_factor(self, double discount_factor) except *

    cpdef critic(self)
    cpdef void set_critic(self, critic) except *

    cpdef Py_ssize_t min_lookahead(self)  except *
    cpdef void set_min_lookahead(self, Py_ssize_t n_steps) except *

    cpdef bint forces_only_min_lookahead(self)  except *
    cpdef void set_forces_only_min_lookahead(
        self,
        bint forces_only_min_lookahead
        ) except *

    cpdef bint uses_terminal_step(self)  except *
    cpdef void set_uses_terminal_step(self, bint uses_terminal_step) except *


cdef void init_BaseTdTargetSetter(BaseTdTargetSetter target_setter) except *

cdef class TdLambdaTargetSetter(BaseTdTargetSetter):
    cdef double __trace_decay
    # lambda (not lambda*gamma, the effective trace decay)

    cdef bint __redistributes_trace_tail

    cpdef double trace_decay(self) except *
    cpdef void set_trace_decay(self, double trace_decay) except *

    cpdef bint redistributes_trace_tail(self)  except *
    cpdef void set_redistributes_trace_tail(
        self,
        bint redistributes_trace_tail
        ) except *


cdef TdLambdaTargetSetter new_TdLambdaTargetSetter(BaseMap critic)
cdef void init_TdLambdaTargetSetter(
    TdLambdaTargetSetter target_setter,
    BaseMap critic
    ) except *

cdef class TdHeavyTargetSetter(BaseTdTargetSetter):
    cdef double __covariance_factor
    cdef bint __normalizes_trace_variance

    cpdef double covariance_factor(self)  except *
    cpdef void set_covariance_factor(self, covariance_factor) except *

    cpdef bint normalizes_trace_variance(self)  except *
    cpdef void set_normalizes_trace_variance(
        self,
        bint normalizes_trace_variance
        ) except *

cdef TdHeavyTargetSetter new_TdHeavyTargetSetter(BaseMap critic)
cdef void init_TdHeavyTargetSetter(
    TdHeavyTargetSetter target_setter,
    BaseMap critic
    ) except *
