cdef class BasePolicy:
    cpdef copy(self, copy_obj = ?)

    cpdef parameters(self)
    cpdef void set_parameters(self, parameters) except *

    cpdef action(self, observation)

cdef class BaseDifferentiablePolicy(BasePolicy):
    cpdef grad_wrt_parameters(self, observation)
    cpdef grad_wrt_observation(self, observation)