#include "rbf_network_approximator.hpp"
#include <valarray>
#include <vector>
#include <sstream>
#include <exception>
#include <random>
#include <cmath>



using std::valarray;
using std::vector;
using std::size_t;
using std::unique_ptr;
using std::shared_ptr;
using std::slice;
using std::move;
using std::exp;
using std::abs;
using std::sqrt;
using std::min;
using std::ostringstream;
using std::invalid_argument;
using std::runtime_error;
using std::make_shared;

valarray<double> concatenate_valarray(const valarray<double>& a, const valarray<double>& b) {
	valarray<double> ab;

	ab.resize(a.size() + b.size());

	ab[slice(0, a.size(), 1)] = a;
	ab[slice(a.size(), b.size(), 1)] = b;

	return ab;
}

namespace rockefeg {
namespace policyopt {

RbfNetworkApproximator::RbfNetworkApproximator() : 
	RbfNetworkApproximator::RbfNetworkApproximator(
		make_shared<RbfNetwork>()
	) 
{}

RbfNetworkApproximator::RbfNetworkApproximator(
	shared_ptr<RbfNetwork> rbf_network
) :

	rbf_network(rbf_network),
	eval_offset(0.),
	epsilon(1.e-9),
	base_dt(0.25),
	shape_scaling_factor(1000.),
	anchor_radius(0.5),
	anchor_update_rate(0.5),
	info_retention_factor(0.999),
	damp_ratio(1.),
	weights_pressures(0., rbf_network->n_centers()),
	weights_momenta(0., rbf_network->n_centers()),
	center_location_pressures(0., rbf_network->n_centers()),
	counters(0., rbf_network->n_centers()),
	center_location_momenta(rbf_network->n_centers(), valarray<double>(0., rbf_network->n_in_dims())),
	anchors(rbf_network->n_centers(), valarray<double>(0., rbf_network->n_in_dims())),
	variances(rbf_network->n_centers(), valarray<double>(0., rbf_network->n_in_dims()))
{
	size_t n_centers = rbf_network->n_centers();
	

	// TODO make sure that the RBF network only has 1 output

	for (size_t center_id = 0; center_id < n_centers; center_id++) {
		this->anchors[center_id] = rbf_network->centers[center_id].location;
		this->variances[center_id] = 1. / rbf_network->centers[center_id].shape / this->shape_scaling_factor;
	}
}

unique_ptr<RbfNetworkApproximator> RbfNetworkApproximator::copy() const {
	return unique_ptr<RbfNetworkApproximator> {
		new RbfNetworkApproximator(move(this->rbf_network->copy()))
	};
}

valarray<double> RbfNetworkApproximator::parameters() const {
	return this->rbf_network->parameters();
}


void RbfNetworkApproximator::set_parameters(const valarray<double>& parameters) {
	this->rbf_network->set_parameters(parameters);
}

size_t RbfNetworkApproximator::n_parameters() const {
	return this->rbf_network->n_parameters();
}

valarray<double> RbfNetworkApproximator::eval(const valarray<double>& input) const {
	return this->rbf_network->eval(input);
}

void RbfNetworkApproximator::update(const vector<ExperienceDatum<>>& trajectory) {
	size_t n_centers = this->rbf_network->n_centers();
	size_t n_in_dims = this->rbf_network->n_in_dims();

	// TODO check the quality of trajectory (same number of input dimensions for all)
	

	// Get fitness.
	double fitness{ this->eval_offset };
	for (const ExperienceDatum<>& experience : trajectory) {
		fitness += experience.feedback;
	}

	// Scan trajectory and get relevant values.
	double fitness_estimate{ 0. };
	vector<valarray<double>> activations_trajectory;
	vector<valarray<double>> inputs;
	valarray<double> sum_activations;
	size_t trajectory_size = trajectory.size();
	//
	inputs.resize(trajectory_size);
	activations_trajectory.resize(trajectory_size);
	sum_activations.resize(this->rbf_network->n_centers());
	//
	for (size_t experience_id{ 0 }; experience_id < trajectory_size; experience_id++) {
		const ExperienceDatum<>& experience = trajectory[experience_id];

		valarray<double> input = concatenate_valarray(
			experience.observation,
			experience.action
		);
		inputs[experience_id] = input;

		activations_trajectory[experience_id] = this->rbf_network->activations(input);
		sum_activations += activations_trajectory[experience_id];

		fitness_estimate += this->rbf_network->eval(input)[0];
	}

	double total_sum_activation = sum_activations.sum();

	// Get error.
	double error = fitness - fitness_estimate;
	double scaled_error = error / (total_sum_activation + this->epsilon);

	// Decay weights information.
	this->weights_pressures *= this->info_retention_factor;
	//
	valarray<double> weights_time_ratio 
		= sum_activations 
		/ (sum_activations + this->weights_pressures + this->epsilon);
	//
	this->weights_momenta *= 1. - this->info_retention_factor * weights_time_ratio;
	
	// Update weights dynamics.
	this->weights_momenta 
		+= (scaled_error - 2 * this->damp_ratio * this->weights_momenta) 
		* this->base_dt * weights_time_ratio;
	this->weights_pressures += sum_activations;
	this->rbf_network->linear[0] += this->weights_momenta * this->base_dt * weights_time_ratio;
	
	
	// Decay center location information.
	this->center_location_pressures *= this->info_retention_factor;
	valarray<double> local_pressure = abs(error) * sum_activations;
	//
	valarray<double> center_location_time_ratio
		= local_pressure
		/ (
			local_pressure
			+ this->center_location_pressures 
			+ this->epsilon
		);

	// Update center location.
	for (size_t center_id = 0; center_id < n_centers; center_id++) {
		valarray<double>& center_location_momentum = this->center_location_momenta[center_id];
		valarray<double>& center_location = this->rbf_network->centers[center_id].location;

		// Decay center location information.
		center_location_momentum *= 1. - this->info_retention_factor * center_location_time_ratio[center_id];

		// Get center location impulse
		double dir 
			= error * this->rbf_network->linear[0][center_id]
			/ (abs(error * this->rbf_network->linear[0][center_id]) + this->epsilon);
		//
		valarray<double> center_location_impulse(0., n_in_dims);
		//
		for (size_t experience_id{ 0 }; experience_id < trajectory_size; experience_id ++ ) {
			center_location_impulse 
				+= (inputs[experience_id] - center_location) 
				* activations_trajectory[experience_id][center_id];
		}
		center_location_impulse 
			*= dir
			* min(
				1., 
				abs(
					error * sum_activations[center_id] 
					/ (this->rbf_network->linear[0][center_id] * total_sum_activation + this->epsilon)
				)
			)
			/ (sum_activations[center_id] + this->epsilon);

		// Update center location dynamics
		center_location_momentum
			+= center_location_impulse - 2 * this->damp_ratio * center_location_momentum
			* this->base_dt * center_location_time_ratio[center_id];
		center_location += center_location_momentum * this->base_dt * center_location_time_ratio[center_id];

	}
	this->center_location_pressures += local_pressure;

	
	/*
	vector<valarray<double>> local_activations_trajectory;
	valarray<double> counters_update(0., n_centers);

	local_activations_trajectory.resize(trajectory_size);

	for (valarray<double>& local_activations : local_activations_trajectory) {
		local_activations.resize(n_centers);
		local_activations = 0.;
	}

	for (size_t experience_id{ 0 }; experience_id < trajectory_size; experience_id ++ ) {
		const valarray<double>& input = inputs[experience_id];


		// Get local variance.
		valarray<double> local_shape(0., n_in_dims);
		double local_sum_activation = activations_trajectory[experience_id].sum();
		//
		for (size_t center_id = 0; center_id < n_centers; center_id++) {
			const valarray<double>& shape = this->rbf_network->centers[center_id].shape;
			local_shape
				+= shape * activations_trajectory[experience_id][center_id]
				/ (local_sum_activation + this->epsilon);
		}


		// Get local activations
		valarray<double>& local_activations = local_activations_trajectory[experience_id];
		//
		for (size_t center_id = 0; center_id < n_centers; center_id++) {
			
			valarray<double>& anchor = this->anchors[center_id];
			valarray<double> separation{ input - anchor };
			double radius_sqr{ (separation * separation * local_shape).sum() };
			local_activations[center_id] = exp(-radius_sqr);
		}

		local_activations /= local_activations.sum() + this->epsilon;
		counters_update += local_activations;
	}
	
	// Decay counters
	this->counters *= this->info_retention_factor;
	
	// Update anchors and variances.
	for (size_t center_id = 0; center_id < n_centers; center_id++) {
		
		valarray<double>& anchor = this->anchors[center_id];
		valarray<double>& variance = this->variances[center_id];
		valarray<double> prev_anchor = anchor;
		valarray<double> variance_moment = variance * this->counters[center_id];
		
		for (size_t experience_id{ 0 }; experience_id < trajectory_size; experience_id ++ ) {
			const valarray<double>& input = inputs[experience_id];
			valarray<double> separation{ input - prev_anchor };
			double local_activation = local_activations_trajectory[experience_id][center_id];

			//Update Anchors
			anchor += 
				separation 
				* local_activation 
				/ (this->counters[center_id] + counters_update[center_id] + this->epsilon);
		}

		for (size_t experience_id{ 0 }; experience_id < trajectory_size; experience_id ++ ) {
			const valarray<double>& input = inputs[experience_id];
			valarray<double> prev_separation{ input - prev_anchor };
			valarray<double> separation{ input - anchor };
			double local_activation = local_activations_trajectory[experience_id][center_id];

			variance_moment += separation * prev_separation * local_activation;
		}
		anchor = anchor * this->anchor_update_rate + prev_anchor * (1 - this->anchor_update_rate);
		this->counters[center_id] += counters_update[center_id];
		variance = variance_moment / (this->counters[center_id] + this->epsilon);

		// Update center shapes
		this->rbf_network->centers[center_id].shape = 1. / (variance + this->epsilon) / this->shape_scaling_factor;
		
		
		// Constrain center locations
		valarray<double> separation{ this->rbf_network->centers[center_id].location - anchor };
		valarray<double> shaped_separation = separation * this->rbf_network->centers[center_id].shape;
		double shaped_separation_sqr_norm = (shaped_separation * shaped_separation).sum();
		if (shaped_separation_sqr_norm > this->anchor_radius* this->anchor_radius) {
			this->rbf_network->centers[center_id].location
				= anchor
				+ this->anchor_radius
				* separation
				/ sqrt(shaped_separation_sqr_norm);
		}
		
	} 
	*/
}

	
	

}} // namespace rockefeg::policyopt
