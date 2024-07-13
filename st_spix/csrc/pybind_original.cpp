/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>

// -- fxns --
void init_bass(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_bass(m);
}
