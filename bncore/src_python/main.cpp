#include "bncore/discretization/manager.hpp"
#include "bncore/graph/graph.hpp"
#include "bncore/inference/compiler.hpp"
#include "bncore/inference/engine.hpp"
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "Core C++ extension for bncore";

  nb::class_<bncore::VariableMetadata>(m, "VariableMetadata")
      .def_ro("id", &bncore::VariableMetadata::id)
      .def_ro("name", &bncore::VariableMetadata::name)
      .def_ro("states", &bncore::VariableMetadata::states)
      .def("num_states", &bncore::VariableMetadata::num_states);

  nb::class_<bncore::Graph>(m, "Graph")
      .def(nb::init<>())
      .def("add_variable", &bncore::Graph::add_variable, nb::arg("name"),
           nb::arg("states"))
      .def("split_state", &bncore::Graph::split_state, nb::arg("id"),
           nb::arg("state_idx"), nb::arg("new_state1"), nb::arg("new_state2"))
      .def("add_edge",
           nb::overload_cast<bncore::NodeId, bncore::NodeId>(
               &bncore::Graph::add_edge),
           nb::arg("parent"), nb::arg("child"))
      .def("add_edge",
           nb::overload_cast<const std::string &, const std::string &>(
               &bncore::Graph::add_edge),
           nb::arg("parent_name"), nb::arg("child_name"))
      .def("get_variable",
           nb::overload_cast<const std::string &>(&bncore::Graph::get_variable,
                                                  nb::const_),
           nb::arg("name"))
      .def("get_parents", &bncore::Graph::get_parents, nb::arg("id"))
      .def("get_children", &bncore::Graph::get_children, nb::arg("id"))
      .def("num_variables", &bncore::Graph::num_variables)
      .def(
          "set_cpt",
          [](bncore::Graph &g, const std::string &name,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> cpt) {
            std::vector<double> probs(cpt.data(), cpt.data() + cpt.size());
            g.set_cpt(name, probs);
          },
          nb::arg("name"), nb::arg("cpt"));

  nb::class_<bncore::JunctionTree>(m, "JunctionTree");

  nb::class_<bncore::JunctionTreeCompiler>(m, "JunctionTreeCompiler")
      .def_static(
          "compile",
          [](const bncore::Graph &graph, const std::string &heuristic) {
            return bncore::JunctionTreeCompiler::compile(graph, heuristic);
          },
          nb::arg("graph"), nb::arg("heuristic") = "min_fill");

  nb::class_<bncore::BatchExecutionEngine>(m, "BatchExecutionEngine")
      .def(nb::init<const bncore::JunctionTree &, std::size_t, std::size_t>(),
           nb::arg("jt"), nb::arg("num_threads") = 0,
           nb::arg("chunk_size") = 1024)
      .def(
          "evaluate",
          [](bncore::BatchExecutionEngine &engine,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> evidence,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> output,
             bncore::NodeId query_var) {
            std::size_t batch_size = evidence.shape(0);
            std::size_t num_vars = evidence.shape(1);

            if (output.shape(0) != batch_size) {
              throw std::invalid_argument(
                  "Output buffer shape must match batch size.");
            }

            // release GIL
            nb::gil_scoped_release release;
            engine.evaluate(evidence.data(), batch_size, num_vars,
                            output.data(), query_var);
          },
          nb::arg("evidence"), nb::arg("output"), nb::arg("query_var"));

  nb::class_<bncore::DiscretizationManager>(m, "DiscretizationManager")
      .def(nb::init<std::size_t>(), nb::arg("max_bins_per_var") = 50)
      .def("should_split", &bncore::DiscretizationManager::should_split,
           nb::arg("graph"), nb::arg("var"))
      .def("split_bin", &bncore::DiscretizationManager::split_bin,
           nb::arg("graph"), nb::arg("var"), nb::arg("state_idx"));
}
