/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
namespace py = pybind11;

// -- fxns --
void init_sim_sum(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_sim_sum(m);
}
