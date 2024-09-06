/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>

// -- fxns --

void init_spix_prop_dev(py::module &);
void init_get_params(py::module &m);
void init_split_disconnected(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_spix_prop_dev(m);
  init_get_params(m);
  init_split_disconnected(m);
}
