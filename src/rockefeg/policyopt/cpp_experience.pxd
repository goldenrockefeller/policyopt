from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from .libcpp_valarray cimport valarray

cdef extern from "cpp_core/experience.hpp" namespace "rockefeg::policyopt" nogil:
    cdef cppclass ExperienceDatum[ObservationT, ActionT, FeedbackT]:
        ObservationT observation
        ActionT action
        FeedbackT feedback