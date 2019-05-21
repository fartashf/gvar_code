#include <string>

#include <torch/extension.h>

#include "pybind/extern.hpp"
#include "src/common.hpp"
#include "src/ops_gpu.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  std::string name = std::string("Foo");
  py::class_<Foo>(m, name.c_str())
      .def(py::init<>())
      .def("setKey", &Foo::setKey)
      .def("getKey", &Foo::getKey)
      .def("__repr__", [](const Foo &a) { return a.toString(); });

  m.def("AddGPU", &AddGPU<float>);
  py::class_<QDQ<float>>(m, "QDQ")
      .def(py::init<unsigned int, at::Tensor>())
      .def(py::init<unsigned int, at::Tensor, unsigned int>())
      .def("qdqGPU", &QDQ<float>::qdqGPU);
}
