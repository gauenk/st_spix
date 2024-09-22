/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
#include "bass/share/my_sp_struct.h"
namespace py = pybind11;

// -- fxns --
void init_rgb2lab(py::module &m);
void init_split_disconnected(py::module &m);
void init_fill_missing(py::module &);
void init_refine_missing(py::module &m);
void init_seg_utils(py::module &m);
void init_prop_bass(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  init_rgb2lab(m);
  init_split_disconnected(m);
  init_fill_missing(m);
  init_refine_missing(m);
  init_seg_utils(m);
  init_prop_bass(m);

  // -- nicer superpixel parameter IO --
  py::class_<PySuperpixelParams>(m, "SuperpixelParams")
    .def(py::init<>())
    .def(py::init<
         torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor>())
    .def("__repr__", [](const PySuperpixelParams& d){ return "Superpixel Parameters"; })
    .def_readwrite("mu_app", &PySuperpixelParams::mu_app)
    .def_readwrite("sigma_app", &PySuperpixelParams::sigma_app)
    .def_readwrite("logdet_sigma_app", &PySuperpixelParams::logdet_sigma_app)
    .def_readwrite("prior_mu_app", &PySuperpixelParams::prior_mu_app)
    .def_readwrite("prior_sigma_app", &PySuperpixelParams::prior_sigma_app)
    .def_readwrite("prior_mu_app_count", &PySuperpixelParams::prior_mu_app_count)
    .def_readwrite("prior_sigma_app_count", &PySuperpixelParams::prior_sigma_app_count)
    // -- shape --
    .def_readwrite("mu_shape", &PySuperpixelParams::mu_shape)
    .def_readwrite("sigma_shape", &PySuperpixelParams::sigma_shape)
    .def_readwrite("logdet_sigma_shape", &PySuperpixelParams::logdet_sigma_shape)
    .def_readwrite("prior_mu_shape", &PySuperpixelParams::prior_mu_shape)
    .def_readwrite("prior_sigma_shape", &PySuperpixelParams::prior_sigma_shape)
    .def_readwrite("prior_mu_shape_count", &PySuperpixelParams::prior_mu_shape_count)
    .def_readwrite("prior_sigma_shape_count", &PySuperpixelParams::prior_sigma_shape_count)
    // -- helpers --
    .def_readwrite("counts", &PySuperpixelParams::counts)
    .def_readwrite("prior_counts", &PySuperpixelParams::prior_counts)
    .def_readwrite("ids", &PySuperpixelParams::ids);

}
