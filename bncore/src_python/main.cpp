#include "bncore/discretization/cpd_integrator.hpp"
#include "bncore/discretization/manager.hpp"
#include "bncore/graph/graph.hpp"
#include "bncore/inference/compiler.hpp"
#include "bncore/inference/engine.hpp"
#include "bncore/inference/hybrid_engine.hpp"
#include "bncore/inference/junction_tree.hpp"
#include "bncore/inference/workspace.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "Core C++ extension for bncore";

  nb::class_<bncore::VariableMetadata>(m, "VariableMetadata")
      .def_ro("id", &bncore::VariableMetadata::id)
      .def_ro("name", &bncore::VariableMetadata::name)
      .def_ro("states", &bncore::VariableMetadata::states)
      .def_ro("cpt", &bncore::VariableMetadata::cpt)
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
      .def("get_variable",
           nb::overload_cast<bncore::NodeId>(&bncore::Graph::get_variable,
                                             nb::const_),
           nb::arg("id"))
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
          nb::arg("name"), nb::arg("cpt"))
      .def("validate_cpts", &bncore::Graph::validate_cpts,
           nb::arg("tolerance") = 1e-6);

  nb::class_<bncore::JunctionTree::Stats>(m, "JunctionTreeStats")
      .def_ro("num_cliques", &bncore::JunctionTree::Stats::num_cliques)
      .def_ro("max_clique_size", &bncore::JunctionTree::Stats::max_clique_size)
      .def_ro("treewidth", &bncore::JunctionTree::Stats::treewidth)
      .def_ro("total_table_entries", &bncore::JunctionTree::Stats::total_table_entries);

  nb::class_<bncore::JunctionTree>(m, "JunctionTree")
      .def("stats", &bncore::JunctionTree::stats);

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
      .def("invalidate_workspace_cache",
           &bncore::BatchExecutionEngine::invalidate_workspace_cache)
      .def("set_dsep_enabled",
           &bncore::BatchExecutionEngine::set_dsep_enabled,
           nb::arg("enabled"))
      .def("dsep_enabled",
           &bncore::BatchExecutionEngine::dsep_enabled)
      .def(
          "evaluate",
          [](bncore::BatchExecutionEngine &engine,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> evidence,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> output,
             bncore::NodeId query_var) {
            std::size_t batch_size = evidence.shape(0);
            std::size_t num_vars = evidence.shape(1);

            std::size_t n_states = engine.junction_tree()
                                       .graph()
                                       ->get_variable(query_var)
                                       .states.size();
            if (output.ndim() != 2 || output.shape(0) != batch_size ||
                output.shape(1) != n_states) {
              throw std::invalid_argument("Output buffer MUST natively match "
                                          "(batch_size, num_states).");
            }

            // release GIL
            nb::gil_scoped_release release;
            engine.evaluate(evidence.data(), batch_size, num_vars,
                            output.data(), query_var);
          },
          nb::arg("evidence"), nb::arg("output"), nb::arg("query_var"))
      .def(
          "evaluate_multi",
          [](bncore::BatchExecutionEngine &engine,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> evidence,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> output,
             nb::ndarray<std::int64_t, nb::c_contig, nb::device::cpu>
                 query_vars,
             nb::ndarray<std::int64_t, nb::c_contig, nb::device::cpu>
                 output_offsets) {
            if (query_vars.ndim() != 1) {
              throw std::invalid_argument(
                  "query_vars must be a 1D int64 ndarray.");
            }
            if (output_offsets.ndim() != 1 ||
                output_offsets.shape(0) != query_vars.shape(0) + 1) {
              throw std::invalid_argument(
                  "output_offsets must be 1D with len(query_vars)+1 entries.");
            }

            std::size_t batch_size = evidence.shape(0);
            std::size_t num_vars = evidence.shape(1);
            if (output.ndim() != 2 || output.shape(0) != batch_size) {
              throw std::invalid_argument(
                  "output MUST have shape (batch_size, total_states).");
            }

            const std::size_t num_queries = query_vars.shape(0);
            std::vector<std::size_t> qvars(num_queries, 0);
            std::vector<std::size_t> offsets(num_queries + 1, 0);

            for (std::size_t i = 0; i < num_queries; ++i) {
              const std::int64_t qv = query_vars.data()[i];
              if (qv < 0) {
                throw std::invalid_argument(
                    "query_vars contains negative node id.");
              }
              qvars[i] = static_cast<std::size_t>(qv);
            }
            for (std::size_t i = 0; i < num_queries + 1; ++i) {
              const std::int64_t off = output_offsets.data()[i];
              if (off < 0) {
                throw std::invalid_argument(
                    "output_offsets contains negative value.");
              }
              offsets[i] = static_cast<std::size_t>(off);
              if (i > 0 && offsets[i] < offsets[i - 1]) {
                throw std::invalid_argument(
                    "output_offsets must be monotonically non-decreasing.");
              }
            }

            const std::size_t total_states = offsets.back();
            if (output.shape(1) != total_states) {
              throw std::invalid_argument(
                  "output second dimension does not match output_offsets[-1].");
            }

            for (std::size_t i = 0; i < num_queries; ++i) {
              const std::size_t expected_states = engine.junction_tree()
                                                      .graph()
                                                      ->get_variable(qvars[i])
                                                      .states.size();
              const std::size_t width = offsets[i + 1] - offsets[i];
              if (width != expected_states) {
                throw std::invalid_argument(
                    "output_offsets widths must match each query variable "
                    "state count.");
              }
            }

            nb::gil_scoped_release release;
            engine.evaluate_multi(evidence.data(), batch_size, num_vars,
                                  qvars.data(), num_queries, offsets.data(),
                                  output.data());
          },
          nb::arg("evidence"), nb::arg("output"), nb::arg("query_vars"),
          nb::arg("output_offsets"))
      .def(
          "evaluate_map",
          [](bncore::BatchExecutionEngine &engine,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> evidence,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> output) {
            if (evidence.ndim() != 2) {
              throw std::invalid_argument(
                  "evidence must be a 2D int ndarray.");
            }

            const std::size_t batch_size = evidence.shape(0);
            const std::size_t num_vars = evidence.shape(1);
            const std::size_t model_vars =
                engine.junction_tree().graph()->num_variables();
            if (output.ndim() != 2 || output.shape(0) != batch_size ||
                output.shape(1) != model_vars) {
              throw std::invalid_argument(
                  "output MUST have shape (batch_size, model_num_vars).");
            }

            nb::gil_scoped_release release;
            engine.evaluate_map(evidence.data(), batch_size, num_vars,
                                output.data());
          },
          nb::arg("evidence"), nb::arg("output"))
      .def(
          "set_soft_evidence",
          [](bncore::BatchExecutionEngine &engine, bncore::NodeId var,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> likelihoods) {
            if (likelihoods.ndim() != 1)
              throw std::invalid_argument("likelihoods must be 1D.");
            engine.set_soft_evidence(var, likelihoods.data(),
                                     likelihoods.shape(0));
          },
          nb::arg("var"), nb::arg("likelihoods"))
      .def(
          "set_soft_evidence_matrix",
          [](bncore::BatchExecutionEngine &engine, bncore::NodeId var,
             nb::ndarray<double, nb::c_contig, nb::device::cpu> likelihoods_matrix) {
            if (likelihoods_matrix.ndim() != 2)
              throw std::invalid_argument("likelihoods_matrix must be 2D.");
            engine.set_soft_evidence_matrix(
                var, likelihoods_matrix.data(),
                likelihoods_matrix.shape(0) * likelihoods_matrix.shape(1));
          },
          nb::arg("var"), nb::arg("likelihoods_matrix"))
      .def("clear_soft_evidence",
           &bncore::BatchExecutionEngine::clear_soft_evidence);

  nb::class_<bncore::DiscretizationManager>(m, "DiscretizationManager")
      .def(nb::init<std::size_t>(), nb::arg("max_bins_per_var") = 40)
      .def("should_split", &bncore::DiscretizationManager::should_split,
           nb::arg("graph"), nb::arg("var"))
      .def("split_bin", &bncore::DiscretizationManager::split_bin,
           nb::arg("graph"), nb::arg("var"), nb::arg("state_idx"))
      .def("initialize_graph",
           &bncore::DiscretizationManager::initialize_graph, nb::arg("graph"))
      .def("rebuild_cpts", &bncore::DiscretizationManager::rebuild_cpts,
           nb::arg("graph"))
      .def("refine", &bncore::DiscretizationManager::refine, nb::arg("graph"))
      .def("last_max_error",
           &bncore::DiscretizationManager::last_max_error)
      .def("converged", &bncore::DiscretizationManager::converged,
           nb::arg("eps_entropy"), nb::arg("eps_kl"))
      .def("add_threshold", &bncore::DiscretizationManager::add_threshold,
           nb::arg("var_id"), nb::arg("threshold"))
      .def(
          "register_normal",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name, const std::vector<bncore::NodeId> &parents,
             std::function<double(const bncore::ParentBins &)> mu_fn,
             std::function<double(const bncore::ParentBins &)> sigma_fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced, bool rare_event_mode) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::NormalCpd>(std::move(mu_fn),
                                                    std::move(sigma_fn)),
                domain_lo, domain_hi, initial_bins, log_spaced,
                rare_event_mode);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("mu_fn"), nb::arg("sigma_fn"), nb::arg("domain_lo"),
          nb::arg("domain_hi"), nb::arg("initial_bins") = 8,
          nb::arg("log_spaced") = false, nb::arg("rare_event_mode") = false)
      .def(
          "register_uniform",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name, const std::vector<bncore::NodeId> &parents,
             std::function<double(const bncore::ParentBins &)> a_fn,
             std::function<double(const bncore::ParentBins &)> b_fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced, bool rare_event_mode) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::UniformCpd>(std::move(a_fn),
                                                     std::move(b_fn)),
                domain_lo, domain_hi, initial_bins, log_spaced,
                rare_event_mode);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("a_fn"), nb::arg("b_fn"), nb::arg("domain_lo"),
          nb::arg("domain_hi"), nb::arg("initial_bins") = 8,
          nb::arg("log_spaced") = false, nb::arg("rare_event_mode") = false)
      .def(
          "register_exponential",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name, const std::vector<bncore::NodeId> &parents,
             std::function<double(const bncore::ParentBins &)> rate_fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced, bool rare_event_mode) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::ExponentialCpd>(std::move(rate_fn)),
                domain_lo, domain_hi, initial_bins, log_spaced,
                rare_event_mode);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("rate_fn"), nb::arg("domain_lo"), nb::arg("domain_hi"),
          nb::arg("initial_bins") = 8, nb::arg("log_spaced") = true,
          nb::arg("rare_event_mode") = false)
      .def(
          "register_lognormal",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name, const std::vector<bncore::NodeId> &parents,
             std::function<double(const bncore::ParentBins &)> log_mu_fn,
             std::function<double(const bncore::ParentBins &)> log_sigma_fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced, bool rare_event_mode) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::LogNormalCpd>(
                    std::move(log_mu_fn), std::move(log_sigma_fn)),
                domain_lo, domain_hi, initial_bins, log_spaced,
                rare_event_mode);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("log_mu_fn"), nb::arg("log_sigma_fn"), nb::arg("domain_lo"),
          nb::arg("domain_hi"), nb::arg("initial_bins") = 8,
          nb::arg("log_spaced") = true, nb::arg("rare_event_mode") = false)
      .def(
          "register_deterministic",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name,
             const std::vector<bncore::NodeId> &parents,
             std::function<double(const bncore::ParentBins &)> fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced, bool monotone, std::size_t n_samples) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::DeterministicCpd>(
                    std::move(fn), monotone, n_samples),
                domain_lo, domain_hi, initial_bins, log_spaced, false);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("fn"), nb::arg("domain_lo"), nb::arg("domain_hi"),
          nb::arg("initial_bins") = 8, nb::arg("log_spaced") = false,
          nb::arg("monotone") = false, nb::arg("n_samples") = 32)
      .def(
          "register_user_function",
          [](bncore::DiscretizationManager &dm, bncore::NodeId var_id,
             const std::string &name, const std::vector<bncore::NodeId> &parents,
             std::function<double(double, double, const bncore::ParentBins &)> fn,
             double domain_lo, double domain_hi, std::size_t initial_bins,
             bool log_spaced) {
            dm.register_variable(
                var_id, name, parents,
                std::make_unique<bncore::UserFunctionCpd>(std::move(fn)),
                domain_lo, domain_hi, initial_bins, log_spaced);
          },
          nb::arg("var_id"), nb::arg("name"), nb::arg("parents"),
          nb::arg("fn"), nb::arg("domain_lo"), nb::arg("domain_hi"),
          nb::arg("initial_bins") = 8, nb::arg("log_spaced") = false)
      .def(
          "variable_edges",
          [](const bncore::DiscretizationManager &dm, bncore::NodeId var_id) {
            for (const auto &v : dm.variables())
              if (v.id == var_id) return v.edges;
            throw std::invalid_argument("variable not registered in manager");
          },
          nb::arg("var_id"))
      .def(
          "variable_posterior",
          [](const bncore::DiscretizationManager &dm, bncore::NodeId var_id) {
            for (const auto &v : dm.variables())
              if (v.id == var_id) return v.posterior;
            throw std::invalid_argument("variable not registered in manager");
          },
          nb::arg("var_id"));

  nb::class_<bncore::ParentBins>(m, "ParentBins")
      .def_ro("continuous_values", &bncore::ParentBins::continuous_values)
      .def_ro("discrete_states", &bncore::ParentBins::discrete_states);

  nb::class_<bncore::HybridEngine::RunConfig>(m, "HybridRunConfig")
      .def(nb::init<>())
      .def_rw("max_iters", &bncore::HybridEngine::RunConfig::max_iters)
      .def_rw("eps_entropy", &bncore::HybridEngine::RunConfig::eps_entropy)
      .def_rw("eps_kl", &bncore::HybridEngine::RunConfig::eps_kl);

  nb::class_<bncore::HybridEngine::RunResult>(m, "HybridRunResult")
      .def_ro("iterations_used",
              &bncore::HybridEngine::RunResult::iterations_used)
      .def_ro("final_max_error",
              &bncore::HybridEngine::RunResult::final_max_error)
      .def_ro("posteriors", &bncore::HybridEngine::RunResult::posteriors)
      .def_ro("edges", &bncore::HybridEngine::RunResult::edges);

  nb::class_<bncore::HybridEngine>(m, "HybridEngine")
      .def(nb::init<bncore::Graph &, bncore::DiscretizationManager &,
                    std::size_t>(),
           nb::arg("graph"), nb::arg("manager"), nb::arg("num_threads") = 1)
      .def(
          "run",
          [](bncore::HybridEngine &eng,
             nb::ndarray<int, nb::c_contig, nb::device::cpu> evidence,
             nb::ndarray<std::int64_t, nb::c_contig, nb::device::cpu> queries,
             const bncore::HybridEngine::RunConfig &cfg) {
            if (evidence.ndim() != 1)
              throw std::invalid_argument("evidence must be 1D int array");
            if (queries.ndim() != 1)
              throw std::invalid_argument("queries must be 1D int64 array");
            std::vector<bncore::NodeId> qv(queries.shape(0));
            for (std::size_t i = 0; i < qv.size(); ++i) {
              qv[i] = static_cast<bncore::NodeId>(queries.data()[i]);
            }
            return eng.run(evidence.data(), evidence.shape(0), qv.data(),
                           qv.size(), cfg);
          },
          nb::arg("evidence"), nb::arg("queries"), nb::arg("config"))
      .def("set_evidence_continuous",
           &bncore::HybridEngine::set_evidence_continuous,
           nb::arg("var"), nb::arg("value"))
      .def("set_soft_evidence_continuous",
           &bncore::HybridEngine::set_soft_evidence_continuous,
           nb::arg("var"), nb::arg("likelihood"))
      .def("clear_evidence_continuous",
           &bncore::HybridEngine::clear_evidence_continuous,
           nb::arg("var"));
}
