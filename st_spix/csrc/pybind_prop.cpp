/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>

// -- fxns --
// void init_bass(py::module &);
// void init_pwd(py::module &);
// void init_scatter_img(py::module &);
// void init_scatter_spix(py::module &);
// void init_sp_pooling(py::module &);
// // void init_spix_prop_dev(py::module &);

// -- fxns --
void init_fill_missing(py::module &);
void init_split_disconnected(py::module &m);
void init_refine_missing(py::module &m);
// void init_bass_iters(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_fill_missing(m);
  init_split_disconnected(m);
  init_refine_missing(m);
  // init_bass_iters(m);
}
