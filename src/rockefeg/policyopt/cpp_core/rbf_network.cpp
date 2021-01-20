#include "rbf_network.hpp"
#include <valarray>
#include <vector>
#include <sstream>
#include <exception>
#include <random>



using std::valarray;
using std::vector;
using std::size_t;
using std::unique_ptr;
using std::slice;
using std::move;
using std::exp;
using std::ostringstream;
using std::invalid_argument;
using std::runtime_error;

namespace rockefeg {
namespace policyopt {
	RbfNetwork::RbfNetwork () : RbfNetwork::RbfNetwork(1, 1, 1) {}

	RbfNetwork::RbfNetwork(size_t n_in_dims, size_t n_centers, size_t n_out_dims) {
		if (n_in_dims <= 0) {
			ostringstream msg;
			msg << "The number of input dimensions (n_in_dims = "
				<< n_in_dims
				<< ") must be positive.";
			throw invalid_argument(msg.str());
		}

		if (n_centers <= 0) {
			ostringstream msg;
			msg << "The number of input dimensions (n_centers = "
				<< n_centers
				<< ") must be positive.";
			throw invalid_argument(msg.str());
		}

		if (n_out_dims <= 0) {
			ostringstream msg;
			msg << "The number of input dimensions (n_out_dims = "
				<< n_out_dims
				<< ") must be positive.";
			throw invalid_argument(msg.str());
		}

		this->centers.resize(n_centers);

		for (Center& center : this->centers) {
			center.location.resize(n_in_dims);
			center.shape.resize(n_in_dims);
		}

		this->linear.resize(n_out_dims);

		for (valarray<double>& weights : this->linear) {
			weights.resize(n_centers);
		}

        // Initialize linear weights.
		for (valarray<double>& weights : this->linear) {
			weights = 1.;
		}

		// Initialize center shapes.
		for (Center& center : this->centers) {
			center.shape = 1.;
		}

		std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937_64 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> distrib(-1., 1.);

		// Initialize center locations.
		for (Center& center : this->centers) {
			for (size_t i = 0; i < n_in_dims; i++) {
				center.location[i] = distrib(gen);
			}
		}
		
	}

	unique_ptr<RbfNetwork> RbfNetwork::copy() const {
		unique_ptr<RbfNetwork> new_rbf_network_core{
			new RbfNetwork(this->n_in_dims(), this->n_centers(), this->n_out_dims()) };

		new_rbf_network_core->set_parameters(this->parameters());

		return move(new_rbf_network_core);
	}

	size_t RbfNetwork::n_in_dims() const {
		return this->centers[0].location.size();
	}

	size_t RbfNetwork::n_centers() const {
		return this->centers.size();
	}

	size_t RbfNetwork::n_out_dims() const {
		return this->linear.size();
	}

	size_t RbfNetwork::n_parameters() const {
		size_t n_parameters{ 0 };

		// Parameters for center locations.
		n_parameters += this->n_in_dims() * this->n_centers();

		// Parameters for center scale.
		n_parameters += this->n_in_dims() * this->n_centers();

		// Parameters for linear layer.
		n_parameters += this->n_out_dims() * this->n_centers();

		return n_parameters;
	}


	valarray<double> RbfNetwork::parameters() const {
		valarray<double> parameters;
		size_t slice_start{ 0 };
		size_t slice_length{ 0 };

		parameters.resize(this->n_parameters());

		for (const Center& center : this->centers) {
			slice_length = center.location.size();
			parameters[slice(slice_start, slice_length, 1)] = center.location;
			slice_start += slice_length;
		}

		for (const Center& center : this->centers) {
			slice_length = center.shape.size();
			parameters[slice(slice_start, slice_length, 1)] = center.shape;
			slice_start += slice_length;
		}

		for (const valarray<double>& weights : this->linear) {
			slice_length = weights.size();
			parameters[slice(slice_start, slice_length, 1)] = weights;
			slice_start += slice_length;
		}

		if (slice_start != this->n_parameters()) {
			ostringstream msg;
			msg << "Something went wrong. slice_start (slice_start = "
				<< slice_start
				<< ") should be equal to the number of parameters for the RBF network "
				<< "(this->n_parameters() =  "
				<< this->n_parameters()
				<< ").";
			throw runtime_error(msg.str());
		}
		
		return parameters;
	}


	void RbfNetwork::set_parameters(const valarray<double>& parameters) {
		size_t slice_start{ 0 };
		size_t slice_length{ 0 };

		if (parameters.size() != this->n_parameters()) {
			ostringstream msg;
			msg << "The number of setting parameters (parameters.size() = "
				<< parameters.size()
				<< ") must be equal to the number of parameters for the RBF network "
				<< "(this->n_parameters() =  "
				<< this->n_parameters()
				<< ").";
			throw invalid_argument(msg.str());
		}

		for (Center& center : this->centers) {
			slice_length = center.location.size();
			center.location = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;
		}

		for (Center& center : this->centers) {
			slice_length = center.shape.size();
			center.shape = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;
		}

		for (valarray<double>& weights : this->linear) {
			slice_length = weights.size();
			weights = parameters[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;
		}

		if (slice_start != this->n_parameters()) {
			ostringstream msg;
			msg << "Something went wrong. slice_start (slice_start = "
				<< slice_start
				<< ") should be equal to the number of parameters for the RBF network "
				<< "(this->n_parameters() =  "
				<< this->n_parameters()
				<< ").";
			throw runtime_error(msg.str());
		}

	}

	valarray<double> RbfNetwork::activations(const valarray<double>& input) const {
		valarray<double> activations; 

		if (input.size() != this->n_in_dims()) {
			ostringstream msg;
			msg << "The size of the input (input.size() = "
				<< input.size()
				<< ") must be equal to the number of input dimensions for the RBF network "
				<< "(this->n_in_dims() =  "
				<< this->n_in_dims()
				<< ").";
			throw invalid_argument(msg.str());
		}

		activations.resize(this->n_centers());

		size_t n_centers = this->n_centers();
		for (size_t center_id{ 0 }; center_id < n_centers; center_id++) {
			const Center& center = this->centers[center_id];
			valarray<double> separation{ center.location - input };
			double radius_sqr{ (separation * separation * center.shape).sum() };
			activations[center_id] = exp(-radius_sqr);
		}

		return activations;
	}

	valarray<double> RbfNetwork::eval(const valarray<double>& input) const {
		valarray<double> activations;
		valarray<double> output;

		if (input.size() != this->n_in_dims()) {
			ostringstream msg;
			msg << "The size of the input (input.size() = "
				<< input.size()
				<< ") must be equal to the number of input dimensions for the RBF network "
				<< "(this->n_in_dims() =  "
				<< this->n_in_dims()
				<< ").";
			throw invalid_argument(msg.str());
		}

		activations = this->activations(input);
		output.resize(this->n_out_dims());

		size_t n_out_dims = this->n_out_dims();
		for (size_t out_id{ 0 }; out_id < n_out_dims; out_id++) {
			output[out_id] = (activations * this->linear[out_id]).sum();
		}

		return output;
	}

	valarray<double> RbfNetwork::grad_wrt_center_locations(
		const valarray<double>& input,
		const valarray<double>& output_grad
		) const
	{
		valarray<double> grad;
		valarray<double> grad_wrt_activations;
		valarray<double> activations;
		size_t n_center_locations_parameters;
		size_t slice_start{ 0 };
		size_t slice_length{ 0 };

		if (input.size() != this->n_in_dims()) {
			ostringstream msg;
			msg << "The size of the input (input.size() = "
				<< input.size()
				<< ") must be equal to the number of input dimensions for the RBF network "
				<< "(this->n_in_dims() =  "
				<< this->n_in_dims()
				<< ").";
			throw invalid_argument(msg.str());
		}

		if (output_grad.size() != this->n_out_dims()) {
			ostringstream msg;
			msg << "The size of the output gradient (output_grad.size() = "
				<< output_grad.size()
				<< ") must be equal to the number of output dimensions for the RBF network "
				<< "(this->n_out_dims() =  "
				<< this->n_out_dims()
				<< ").";
			throw invalid_argument(msg.str());
		}

		// Get the gradient with respect to the activations.
		grad_wrt_activations.resize(this->n_centers());
		grad_wrt_activations = 0.;
		
		size_t n_out_dims = this->n_out_dims();
		for (size_t out_id{ 0 }; out_id < n_out_dims; out_id++) {
			const valarray<double>& weights{ this->linear[out_id] };
			grad_wrt_activations += weights * output_grad[out_id];
		}

		activations = this->activations(input);

		// Get the gradient with respect to center location.
		size_t n_centers = this->n_centers();
		n_center_locations_parameters = this->n_in_dims() * n_centers;
		grad.resize(n_center_locations_parameters);
		for (size_t center_id{ 0 }; center_id < n_centers; center_id++) {
			const Center& center = this->centers[center_id];
			slice_length = center.location.size();

			grad[slice(slice_start, slice_length, 1)] 
				= -2. 
				* activations[center_id]
				* grad_wrt_activations[center_id]
				* center.shape 
				* (center.location - input);

			slice_start += slice_length;	
		}		

		if (slice_start != grad.size()) {
			ostringstream msg;
			msg << "Something went wrong. slice_start (slice_start = "
				<< slice_start
				<< ") should be equal to the number of gradient parameters for "
				<< "the RBF network with respect to the center locations "
				<< "(n_center_locations_parameters =  "
				<< n_center_locations_parameters
				<< ").";
			throw runtime_error(msg.str());
		}

		return grad;
	}

	valarray<double> RbfNetwork::flattened_center_locations() const {
		valarray<double> center_locations_parameters;
		size_t n_center_locations_parameters;
		size_t slice_start{ 0 };
		size_t slice_length{ 0 };

		n_center_locations_parameters = this->n_in_dims() * this->n_centers();

		center_locations_parameters.resize(n_center_locations_parameters);

		for (const Center& center : this->centers) {
			slice_length = center.location.size();
			center_locations_parameters[slice(slice_start, slice_length, 1)] = center.location;
			slice_start += slice_length;
		}

		if (slice_start != n_center_locations_parameters) {
			ostringstream msg;
			msg << "Something went wrong. slice_start (slice_start = "
				<< slice_start
				<< ") should be equal to the number of gradient parameters for "
				<< "the RBF network with respect to the center locations "
				<< "(this->n_in_dims() * this->n_centers() =  "
				<< n_center_locations_parameters
				<< ").";
			throw runtime_error(msg.str());
		}

		return center_locations_parameters;
	}


	void RbfNetwork::set_center_locations_from_valarray(const valarray<double>& flattened_center_locations) {
		size_t slice_start{ 0 };
		size_t slice_length{ 0 };
		size_t n_center_locations_parameters;

		n_center_locations_parameters = this->n_in_dims() * this->n_centers();

		if (flattened_center_locations.size() != n_center_locations_parameters) {
			ostringstream msg;
			msg << "The number of flattened center location parameters "
				<< "(flattened_center_locations.size() = "
				<< flattened_center_locations.size()
				<< ") must be equal to the number of gradient parameters for "
				<< "the RBF network with respect to the center locations "
				<< "(n_center_locations_parameters =  "
				<< n_center_locations_parameters
				<< ").";
			throw invalid_argument(msg.str());
		}

		for (Center& center : this->centers) {
			slice_length = center.location.size();
			center.location = flattened_center_locations[slice(slice_start, slice_length, 1)];
			slice_start += slice_length;
		}

		if (slice_start != n_center_locations_parameters) {
			ostringstream msg;
			msg << "Something went wrong. slice_start (slice_start = "
				<< slice_start
				<< ") should be equal to the number of gradient parameters for "
				<< "the RBF network with respect to the center locations "
				<< "(n_center_locations_parameters =  "
				<< n_center_locations_parameters
				<< ").";
			throw runtime_error(msg.str());
		}
	}

}
}