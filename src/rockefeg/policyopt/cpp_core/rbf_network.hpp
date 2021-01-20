#ifndef ROCKEFEG_POLICYOPT_RBF_NETWORK_HPP 
#define ROCKEFEG_POLICYOPT_RBF_NETWORK_HPP

#include <valarray>
#include <vector>
#include <memory>


namespace rockefeg { 
namespace policyopt {
	struct RbfNetwork {

		struct Center {
			std::valarray<double> location;
			std::valarray<double> shape;
		};


		std::vector<Center> centers;
		std::vector<std::valarray<double>> linear;

		RbfNetwork();
		RbfNetwork(std::size_t n_in_dims, std::size_t n_centers, std::size_t n_out_dims);

		std::unique_ptr<RbfNetwork> copy() const;

		std::size_t n_in_dims() const;
		std::size_t n_centers() const;
		std::size_t n_out_dims() const;
		std::size_t n_parameters() const;
		std::valarray<double> parameters() const;
		void set_parameters(const std::valarray<double>& parameters);

		std::valarray<double> activations(const std::valarray<double>& input) const;

		std::valarray<double> eval(const std::valarray<double>& input) const;

		std::valarray<double> grad_wrt_center_locations(
			const std::valarray<double>& input,
			const std::valarray<double>& output_grad
		) const;

		std::valarray<double> flattened_center_locations() const;
		void set_center_locations_from_valarray(const std::valarray<double>& flattened_center_locations);
	};
}
}

#endif