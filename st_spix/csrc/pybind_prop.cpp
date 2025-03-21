/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
#include "bass/share/my_sp_struct.h"
namespace py = pybind11;

// -- fxns --
void init_rgb2lab(py::module &m);
void init_split_disconnected(py::module &m);
void init_sp_video_pooling(py::module &m);
void init_fill_missing(py::module &);
void init_refine_missing(py::module &m);
void init_simple_refine_missing(py::module &m);
void init_seg_utils(py::module &m);
void init_prop_bass(py::module &m);
void init_bass(py::module &m);
void init_shift_labels(py::module &m);
void init_shift_tensor(py::module &m);
void init_shift_tensor_ordered(py::module &m);
void init_shift_order(py::module &m);
void init_sparams_io(py::module &m);
void init_refine(py::module &m);

// void init_relabel(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  init_rgb2lab(m);
  init_split_disconnected(m);
  init_fill_missing(m);
  init_refine_missing(m);
  init_simple_refine_missing(m);
  init_seg_utils(m);
  init_prop_bass(m);
  init_bass(m);
  init_sp_video_pooling(m);
  init_shift_labels(m);
  init_shift_tensor(m);
  init_shift_tensor_ordered(m);
  init_shift_order(m);
  init_sparams_io(m);
  init_refine(m);
  // init_relabel(m);

  // -- nicer superpixel parameter IO --
  py::class_<PySuperpixelParams>(m, "SuperpixelParams")
    .def(py::init<>())
    .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
         torch::Tensor, torch::Tensor, torch::Tensor>(),
         py::keep_alive<1, 2>(),py::keep_alive<1, 3>(),py::keep_alive<1, 4>(),
         py::keep_alive<1, 5>(),py::keep_alive<1, 6>(),py::keep_alive<1, 7>(),
         py::keep_alive<1, 8>(),py::keep_alive<1, 9>(),py::keep_alive<1, 10>(),
         py::keep_alive<1, 11>(),py::keep_alive<1, 12>(),py::keep_alive<1, 13>(),
         py::keep_alive<1, 14>(),py::keep_alive<1, 15>(),py::keep_alive<1, 16>(),
         py::keep_alive<1, 17>())
    .def("__repr__", [](const PySuperpixelParams& d){ return "Superpixel Parameters"; })
    // -- appearance --
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
