# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/rbf_network.cpp

from libcpp.memory cimport make_shared
from rockefeg.cyutil.array cimport new_DoubleArray


cdef valarray[double] valarray_from_DoubleArray(DoubleArray arr) except *:
    cdef valarray[double] new_arr
    cdef Py_ssize_t id

    new_arr.resize(len(arr))

    for id in range(len(arr)):
        new_arr[<size_t>id] = arr.view[id]

    return new_arr

cdef DoubleArray DoubleArray_from_valarray(valarray[double] arr):
    cdef DoubleArray new_arr
    cdef Py_ssize_t id

    new_arr = new_DoubleArray(<Py_ssize_t>arr.size())

    for id in range(len(new_arr)):
        new_arr.view[id] = arr[<size_t>id]

    return new_arr

cdef class RbfNetwork(BaseMap):
    def __init__(
            self,
            Py_ssize_t n_in_dims,
            Py_ssize_t n_centers,
            Py_ssize_t n_out_dims):

        if n_in_dims <= 0:
            raise (
                ValueError(
                    "The number of input dimensions (n_in_dims = {n_in_dims}) "
                    "must be positive."
                    .format(**locals())))

        if n_centers <= 0:
            raise (
                ValueError(
                    "The number of centers (n_centers = {n_centers}) "
                    "must be positive."
                    .format(**locals())))

        if n_out_dims <= 0:
            raise (
                ValueError(
                    "The number of output dimensions "
                    "(n_out_dims = {n_out_dims}) "
                    "must be positive."
                    .format(**locals())))

        self.core = (
            make_shared[CppRbfNetwork](n_in_dims, n_centers, n_out_dims) )

    cpdef RbfNetwork copy(self, copy_obj = None):
        cdef RbfNetwork new_network

        if copy_obj is None:
            new_network = RbfNetwork.__new__(RbfNetwork)
        else:
            new_network = copy_obj

        # TODO Use cdef init_Rbf.. and new_Rbf...
        new_network = RbfNetwork(1, 1, 1)

        # Move the copy of the core RBF network.
        new_network.core.reset(self.core.get().copy().release())

        return new_network

    cpdef DoubleArray center_location(self, Py_ssize_t center_id):
        cdef Py_ssize_t n_centers

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().centers[center_id].location ))

    cpdef void set_center_location(
            self,
            Py_ssize_t center_id,
            DoubleArray location
            ) except *:
        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims
        cdef Py_ssize_t location_size

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))


        n_in_dims = self.n_in_dims()
        location_size = len(location)

        if location_size != n_in_dims:
            raise (
                ValueError(
                    "The location size (len(location) = {location_size}) "
                    "must be equal to the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        self.core.get().centers[center_id].location = (
            valarray_from_DoubleArray(location) )

    cpdef double center_location_element(
            self,
            Py_ssize_t center_id,
            Py_ssize_t element_id
            ) except *:
        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        n_in_dims = self.n_in_dims()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_in_dims:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        return self.core.get().centers[center_id].location[element_id]

    cpdef void set_center_location_element (
            self,
            Py_ssize_t center_id,
            Py_ssize_t element_id,
            double val
            ) except *:
        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        n_in_dims = self.n_in_dims()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_in_dims:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        self.core.get().centers[center_id].location[element_id] = val


    cpdef DoubleArray center_shape(self, Py_ssize_t center_id):
        cdef Py_ssize_t n_centers

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().centers[center_id].shape ))


    cpdef void set_center_shape(
            self,
            Py_ssize_t center_id,
            DoubleArray shape
            ) except *:
        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims
        cdef Py_ssize_t shape_size

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))


        n_in_dims = self.n_in_dims()
        shape_size = len(shape)

        if shape_size != n_in_dims:
            raise (
                ValueError(
                    "The shape size (len(shape) = {shape_size}) "
                    "must be equal to the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        self.core.get().centers[center_id].shape = (
            valarray_from_DoubleArray(shape) )


    cpdef double center_shape_element(
            self,
            Py_ssize_t center_id,
            Py_ssize_t element_id
            ) except *:

        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        n_in_dims = self.n_in_dims()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_in_dims:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        return self.core.get().centers[center_id].shape[element_id]


    cpdef void set_center_shape_element (
            self,
            Py_ssize_t center_id,
            Py_ssize_t element_id,
            double val
            ) except *:

        cdef Py_ssize_t n_centers
        cdef Py_ssize_t n_in_dims

        n_centers = self.n_centers()

        if center_id < 0:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if center_id >= n_centers:
            raise (
                ValueError(
                    "The center index (center_id = {center_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        n_in_dims = self.n_in_dims()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_in_dims:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of input dimensions "
                    "(self.n_in_dims() = {n_in_dims})."
                    .format(**locals())))

        self.core.get().centers[center_id].shape[element_id] = val


    cpdef DoubleArray weights(self, Py_ssize_t out_dim_id):
        cdef Py_ssize_t n_out_dims

        n_out_dims = self.n_out_dims()

        if out_dim_id < 0:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if out_dim_id >= n_out_dims:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be less than the number of output dimensions "
                    "(self.n_out_dims() = {n_out_dims})."
                    .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().linear[out_dim_id] ))

    cpdef void set_weights(
            self,
            Py_ssize_t out_dim_id,
            DoubleArray weights
            ) except *:
        cdef Py_ssize_t n_out_dims
        cdef Py_ssize_t n_centers
        cdef Py_ssize_t weights_size


        n_out_dims = self.n_out_dims()

        if out_dim_id < 0:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if out_dim_id >= n_out_dims:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be less than the number of output dimensions "
                    "(self.n_out_dims() = {n_out_dims})."
                    .format(**locals())))

        n_centers = self.n_centers()
        weights_size = len(weights)

        if weights_size != n_centers:
            raise (
                ValueError(
                    "The weights size (len(weights) = {weights_size}) "
                    "must be equal to the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))

        self.core.get().linear[out_dim_id] = (
            valarray_from_DoubleArray(weights) )

    cpdef double weights_element(
            self,
            Py_ssize_t out_dim_id,
            Py_ssize_t element_id
            ) except *:

        cdef Py_ssize_t n_out_dims
        cdef Py_ssize_t n_centers

        n_out_dims = self.n_out_dims()

        if out_dim_id < 0:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if out_dim_id >= n_out_dims:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be less than the number of output dimensions "
                    "(self.n_out_dims() = {n_out_dims})."
                    .format(**locals())))

        n_centers = self.n_centers()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_centers:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))


        return self.core.get().linear[out_dim_id][element_id]

    cpdef void set_weights_element (
            self,
            Py_ssize_t out_dim_id,
            Py_ssize_t element_id,
            double val
            ) except *:

        cdef Py_ssize_t n_out_dims
        cdef Py_ssize_t n_centers

        n_out_dims = self.n_out_dims()

        if out_dim_id < 0:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if out_dim_id >= n_out_dims:
            raise (
                ValueError(
                    "The output dimension index (out_dim_id = {out_dim_id}) "
                    "must be less than the number of output dimensions "
                    "(self.n_out_dims() = {n_out_dims})."
                    .format(**locals())))

        n_centers = self.n_centers()

        if element_id < 0:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be non-negative."
                    .format(**locals())))

        if element_id >= n_centers:
            raise (
                ValueError(
                    "The element index (element_id = {element_id}) "
                    "must be less than the number of centers "
                    "(self.n_centers() = {n_centers})."
                    .format(**locals())))


        self.core.get().linear[out_dim_id][element_id] = val

    cpdef Py_ssize_t n_in_dims(self) except *:
        return self.core.get().n_in_dims()

    cpdef Py_ssize_t n_centers(self) except *:
        return self.core.get().n_centers()

    cpdef Py_ssize_t n_out_dims(self) except *:
        return self.core.get().n_out_dims()

    cpdef DoubleArray activations(self, DoubleArray input):
        # cdef Py_ssize_t n_in_dims
        # cdef Py_ssize_t input_size
        #
        # n_in_dims = self.n_in_dims()
        # input_size = len(input)
        #
        # if input_size != n_in_dims:
        #     raise (
        #         ValueError(
        #             "The input size (len(input) = {input_size}) "
        #             "must be equal to the number of input dimensions "
        #             "(self.n_in_dims() = {n_in_dims})."
        #             .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().activations(
                    valarray_from_DoubleArray(
                        input ))))

    cpdef DoubleArray grad_wrt_center_locations(
            self,
            DoubleArray input,
            DoubleArray out_grad = None ):

        # cdef Py_ssize_t input_size
        # cdef Py_ssize_t n_in_dims
        # cdef Py_ssize_t out_grad_size
        # cdef Py_ssize_t n_out_dims
        #
        # n_in_dims = self.n_in_dims()
        # input_size = len(input)
        #
        # if input_size != n_in_dims:
        #     raise (
        #         ValueError(
        #             "The input size (len(input) = {input_size}) "
        #             "must be equal to the number of input dimensions "
        #             "(self.n_in_dims() = {n_in_dims})."
        #             .format(**locals())))
        #
        # if out_grad is None:
        #     out_grad = new_DoubleArray(self.n_out_dims())
        #     out_grad.set_all_to(1.)
        #
        # n_out_dims = self.n_out_dims()
        # out_grad_size = len(out_grad)
        #
        # if out_grad_size != n_out_dims:
        #     raise (
        #         ValueError(
        #             "The out_grad size (len(out_grad) = {out_grad_size}) "
        #             "must be equal to the number of out_grad dimensions "
        #             "(self.n_out_dims() = {n_out_dims})."
        #             .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().grad_wrt_center_locations(
                    valarray_from_DoubleArray(
                        input ),
                    valarray_from_DoubleArray(
                        out_grad ) )))

    cpdef DoubleArray flattened_center_locations(self):
        return (
            DoubleArray_from_valarray(
                self.core.get().flattened_center_locations() ))

    cpdef void set_center_locations_from_valarray(
            self,
            DoubleArray flattened_center_locations
            ) except *:

        # cdef Py_ssize_t flattened_center_locations_size
        # cdef Py_ssize_t n_in_dims
        # cdef Py_ssize_t n_centers
        # cdef Py_ssize_t n_center_location_parameters
        #
        # n_in_dims = self.n_in_dims()
        # n_centers = self.n_centers()
        #
        # flattened_center_locations_size = len(flattened_center_locations)
        # n_center_location_parameters = n_in_dims * n_centers
        #
        # if flattened_center_locations_size != n_center_location_parameters:
        #     raise (
        #         ValueError(
        #             "The flattened center locations size "
        #             "(len(flattened_center_locations) = "
        #             "{flattened_center_locations_size}) "
        #             "must be equal to the number of center location parameters "
        #             "(self.n_in_dims() * self.n_centers() = "
        #             "{n_center_location_parameters})."
        #             .format(**locals())))


        self.core.get().set_center_locations_from_valarray(
            valarray_from_DoubleArray(
                flattened_center_locations ))

    cpdef DoubleArray parameters(self):
        return DoubleArray_from_valarray(self.core.get().parameters())

    cpdef void set_parameters(self, parameters) except *:
        cdef DoubleArray cy_parameters = parameters

        # cdef Py_ssize_t n_parameters
        # cdef Py_ssize_t parameters_size
        #
        # n_parameters = self.n_parameters()
        # parameters_size = len(parameters)
        #
        # if parameters_size != n_parameters:
        #     raise (
        #         ValueError(
        #             "The parameters size (len(parameters) = {parameters_size}) "
        #             "must be equal to the number of input dimensions "
        #             "(self.n_parameters() = {n_parameters})."
        #             .format(**locals())))


        self.core.get().set_parameters(
            valarray_from_DoubleArray(
                cy_parameters ))

    cpdef Py_ssize_t n_parameters(self) except *:
        return self.core.get().n_parameters()


    cpdef DoubleArray eval(self, input):
        cdef DoubleArray cy_input = input
        # cdef Py_ssize_t input_size
        # cdef Py_ssize_t n_in_dims
        #
        # n_in_dims = self.n_in_dims()
        # input_size = len(cy_input)
        #
        # if input_size != n_in_dims:
        #     raise (
        #         ValueError(
        #             "The input size (len(input) = {input_size}) "
        #             "must be equal to the number of input dimensions "
        #             "(self.n_in_dims() = {n_in_dims})."
        #             .format(**locals())))

        return (
            DoubleArray_from_valarray(
                self.core.get().eval(
                    valarray_from_DoubleArray(
                        cy_input ))))

