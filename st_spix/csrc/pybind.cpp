/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
#include "bass/share/my_sp_struct.h"
// #include "prop/seg_utils.h"
namespace py = pybind11;


// -- fxns --
void init_bass(py::module &);
void init_pwd(py::module &);
void init_scatter_img(py::module &);
void init_scatter_spix(py::module &);
void init_sp_pooling(py::module &);
// void init_spix_prop_dev(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  init_pwd(m);
  init_bass(m);
  init_scatter_img(m);
  init_scatter_spix(m);
  init_sp_pooling(m);

  // // -- nicer superpixel parameter IO --
  // pybind registers structs globally, so we only keep it in pybind_prop
  // py::class_<PySuperpixelParams>(m, "SuperpixelParams")
  //   .def(py::init<>())
  //   .def(py::init<torch::Tensor,torch::Tensor,torch::Tensor,
  //        torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>())
  //   .def_readwrite("mu_i", &PySuperpixelParams::mu_i)
  //   .def_readwrite("mu_s", &PySuperpixelParams::mu_s)
  //   .def_readwrite("sigma_s", &PySuperpixelParams::sigma_s)
  //   .def_readwrite("logdet_Sigma_s", &PySuperpixelParams::logdet_Sigma_s)
  //   .def_readwrite("counts", &PySuperpixelParams::counts)
  //   .def_readwrite("prior_counts", &PySuperpixelParams::prior_counts);

  // init_spix_prop_dev(m);
}
