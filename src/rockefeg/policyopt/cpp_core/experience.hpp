#ifndef ROCKEFEG_POLICYOPT_EXPERIENCE_HPP 
#define ROCKEFEG_POLICYOPT_EXPERIENCE_HPP

#include <valarray>

namespace rockefeg { 
namespace policyopt {

template<
	typename ObservationT = std::valarray<double>,
	typename ActionT = std::valarray<double>,
	typename FeedbackT = double
>
struct ExperienceDatum {
	ObservationT observation;
	ActionT action;
	FeedbackT feedback;
};


}
}

#endif