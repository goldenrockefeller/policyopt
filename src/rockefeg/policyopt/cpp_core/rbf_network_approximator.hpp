#ifndef ROCKEFEG_POLICYOPT_RBF_NETWORK_APPROXIMATOR_HPP 
#define ROCKEFEG_POLICYOPT_RBF_NETWORK_APPROXIMATOR_HPP

#include "rbf_network.hpp"
#include "experience.hpp" 
// ExperienceDatum
#include <valarray>
#include <vector>
#include <memory>


namespace rockefeg {
namespace policyopt {
	
struct RbfNetworkApproximator {
	std::shared_ptr<RbfNetwork> rbf_network;
	
	double eval_offset;
	double epsilon;
	double base_dt;
	double shape_scaling_factor;
	double anchor_radius;
	double anchor_update_rate;
	double info_retention_factor;
	double damp_ratio;

	std::valarray<double> weights_pressures;
	std::valarray<double> weights_momenta;
	std::valarray<double> center_location_pressures;
	std::valarray<double> counters;

	std::vector<std::valarray<double>> center_location_momenta;
	std::vector<std::valarray<double>> anchors;
	std::vector<std::valarray<double>> variances;

	RbfNetworkApproximator();
	RbfNetworkApproximator(std::shared_ptr<RbfNetwork> rbf_network);

	std::unique_ptr<RbfNetworkApproximator> copy() const;
	
	std::valarray<double> parameters() const;
	void set_parameters(const std::valarray<double>& parameters);
	std::size_t n_parameters() const;

	std::valarray<double> eval(const std::valarray<double>& input) const;
	void update(const std::vector<ExperienceDatum<>>& trajectory);
}; 

}} //namespace rockefeg::policyopt


#endif