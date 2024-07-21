/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>

// -- fxns --
void init_spix_prop_dev(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_spix_prop_dev(m);
}
