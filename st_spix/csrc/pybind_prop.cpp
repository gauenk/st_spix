/*************************************

       Pybind for Pytorch & C++

*************************************/

#include <torch/extension.h>
#include "bass/share/my_sp_struct.h"
namespace py = pybind11;

// -- fxns --
void init_fill_missing(py::module &);
void init_split_disconnected(py::module &m);
void init_refine_missing(py::module &m);
void init_seg_utils(py::module &m);
// void init_prop_bass(py::module &m);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  init_fill_missing(m);
  init_split_disconnected(m);
  init_refine_missing(m);
  init_seg_utils(m);
  // init_prop_bass(m);

  // -- nicer superpixel parameter IO --
  py::class_<PySuperpixelParams>(m, "SuperpixelParams")
    .def(py::init<>())
    .def(py::init<torch::Tensor,torch::Tensor,torch::Tensor,
         torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>())
    .def("__repr__", [](const PySuperpixelParams& d){ return "Superpixel Parameters [.mu_i, .mu_s, .sigma_s, .logdet_Sigma_s, .counts, .prior_counts, .ids]"; }) // probably a better way but oh well...
    .def_readwrite("mu_i", &PySuperpixelParams::mu_i)
    .def_readwrite("mu_s", &PySuperpixelParams::mu_s)
    .def_readwrite("sigma_s", &PySuperpixelParams::sigma_s)
    .def_readwrite("logdet_Sigma_s", &PySuperpixelParams::logdet_Sigma_s)
    .def_readwrite("counts", &PySuperpixelParams::counts)
    .def_readwrite("prior_counts", &PySuperpixelParams::prior_counts)
    .def_readwrite("ids", &PySuperpixelParams::prior_counts);

  // py::class_<superpixel_params>(m, "superpixel_params")
  //   .def(py::init<>())
  //   .def(py::init<float,float,float,double,double,double,
  //        double,double,double,int,double,int,float,int>())
  //   .def_readwrite("mu_i_x", &superpixel_params::mu_i.x)
  //   .def_readwrite("mu_i_y", &superpixel_params::mu_i.y)
  //   .def_readwrite("mu_i_z", &superpixel_params::mu_i.z)
  //   .def_readwrite("mu_s_x", &superpixel_params::mu_s.x)
  //   .def_readwrite("mu_s_y", &superpixel_params::mu_s.y)
  //   .def_readwrite("sigma_s_x", &superpixel_params::sigma_s.x)
  //   .def_readwrite("sigma_s_y", &superpixel_params::sigma_s.y)
  //   .def_readwrite("sigma_s_z", &superpixel_params::sigma_s.z)
  //   .def_readwrite("logdet_sigma_s", &superpixel_params::logdet_Sigma_s)
  //   .def_readwrite("count", &superpixel_params::count)
  //   .def_readwrite("log_count", &superpixel_params::log_count)
  //   .def_readwrite("valid", &superpixel_params::valid)
  //   .def_readwrite("prior_count", &superpixel_params::prior_count)
  //   .def_readwrite("parent_spix", &superpixel_params::parent_spix);

}
