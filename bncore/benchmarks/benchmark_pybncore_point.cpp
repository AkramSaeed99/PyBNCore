#include "../../smile_cpp/smile.h"
#include "../../smile_license (1)/smile_license.h"

#include "bncore/graph/graph.hpp"
#include "bncore/inference/compiler.hpp"
#include "bncore/inference/engine.hpp"

#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static std::string trim_ascii(std::string s) {
  const char *ws = " \t\r\n";
  const auto begin = s.find_first_not_of(ws);
  if (begin == std::string::npos) return std::string();
  const auto end = s.find_last_not_of(ws);
  return s.substr(begin, end - begin + 1);
}

static bncore::Graph load_graph_from_xdsl_via_smile(const std::string &xdsl_path) {
  DSL_network net;
  if (net.ReadFile(xdsl_path.c_str()) != DSL_OKAY) {
    throw std::runtime_error("SMILE failed to read XDSL: " + xdsl_path);
  }

  bncore::Graph graph;
  std::vector<int> node_handles;
  std::unordered_map<int, std::string> handle_to_name;
  node_handles.reserve(256);

  int h = net.GetFirstNode();
  while (h >= 0) {
    node_handles.push_back(h);
    DSL_node *node = net.GetNode(h);
    const std::string node_name = node->GetId();
    handle_to_name[h] = node_name;

    std::vector<std::string> states;
    const int num_states = node->Definition()->GetNumberOfOutcomes();
    const DSL_idArray *state_ids = node->Definition()->GetOutcomeIds();
    states.reserve(static_cast<std::size_t>(num_states));
    for (int s = 0; s < num_states; ++s) {
      if (state_ids && s < state_ids->GetSize() && (*state_ids)[s]) {
        states.emplace_back((*state_ids)[s]);
      } else {
        states.emplace_back("S" + std::to_string(s));
      }
    }
    graph.add_variable(node_name, states);
    h = net.GetNextNode(h);
  }

  for (int child_handle : node_handles) {
    const auto &parents = net.GetParents(child_handle);
    const std::string &child_name = handle_to_name.at(child_handle);
    for (int i = 0; i < parents.GetSize(); ++i) {
      const int parent_handle = parents[i];
      graph.add_edge(handle_to_name.at(parent_handle), child_name);
    }
  }

  for (int handle : node_handles) {
    DSL_node *node = net.GetNode(handle);
    const std::string &node_name = handle_to_name.at(handle);
    const DSL_Dmatrix *mat = node->Definition()->GetMatrix();
    if (!mat) {
      throw std::runtime_error("Missing CPT matrix for node '" + node_name + "'.");
    }
    std::vector<double> cpt(static_cast<std::size_t>(mat->GetSize()));
    for (int i = 0; i < mat->GetSize(); ++i) {
      cpt[static_cast<std::size_t>(i)] = (*mat)[i];
    }
    graph.set_cpt(node_name, cpt);
  }

  return graph;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./benchmark_pybncore_point <xdsl_file> <evidence_csv> [target_node]\n";
    return 1;
  }

  const std::string xdsl_file = argv[1];
  const std::string evidence_csv = argv[2];
  const std::string target_node = (argc >= 4) ? argv[3] : "L9_N9";

  bncore::Graph graph;
  try {
    graph = load_graph_from_xdsl_via_smile(xdsl_file);
  } catch (const std::exception &e) {
    std::cerr << "Failed to load graph: " << e.what() << "\n";
    return 1;
  }

  auto jt = bncore::JunctionTreeCompiler::compile(graph, "min_fill");
  bncore::BatchExecutionEngine engine(*jt, /*num_threads=*/1, /*chunk_size=*/1);

  const auto &target_meta = graph.get_variable(target_node);
  const bncore::NodeId target_id = target_meta.id;
  const std::size_t target_states = target_meta.states.size();
  const std::size_t num_vars = graph.num_variables();

  std::ifstream file(evidence_csv);
  if (!file) {
    std::cerr << "Cannot open evidence CSV: " << evidence_csv << "\n";
    return 1;
  }

  std::string line;
  if (!std::getline(file, line)) {
    std::cerr << "Evidence CSV is empty.\n";
    return 1;
  }

  std::vector<bncore::NodeId> evidence_nodes;
  {
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
      token = trim_ascii(token);
      if (token.empty()) continue;
      try {
        evidence_nodes.push_back(graph.get_variable(token).id);
      } catch (...) {
        std::cerr << "Evidence node '" << token << "' not found in graph.\n";
        return 1;
      }
    }
  }

  std::vector<int> evidence_flat;
  std::size_t batch_size = 0;
  while (std::getline(file, line)) {
    if (trim_ascii(line).empty()) continue;
    std::vector<int> row(static_cast<std::size_t>(num_vars), -1);
    std::stringstream ss(line);
    std::string val;
    std::size_t col = 0;
    while (std::getline(ss, val, ',')) {
      if (col >= evidence_nodes.size()) {
        std::cerr << "Evidence row has more columns than header.\n";
        return 1;
      }
      const int obs = std::stoi(trim_ascii(val));
      const bncore::NodeId nid = evidence_nodes[col];
      if (obs >= 0) {
        const std::size_t n_states = graph.get_variable(nid).states.size();
        if (obs >= static_cast<int>(n_states)) {
          std::cerr << "Evidence state out of range for node '" <<
              graph.get_variable(nid).name << "'.\n";
          return 1;
        }
        row[static_cast<std::size_t>(nid)] = obs;
      }
      ++col;
    }
    if (col != evidence_nodes.size()) {
      std::cerr << "Evidence row column count mismatch.\n";
      return 1;
    }
    evidence_flat.insert(evidence_flat.end(), row.begin(), row.end());
    ++batch_size;
  }

  if (batch_size == 0) {
    std::cerr << "No evidence rows found.\n";
    return 1;
  }

  std::vector<double> output(target_states, 0.0);

  auto run_pass = [&](double &seconds_out) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t r = 0; r < batch_size; ++r) {
      const int *ev = evidence_flat.data() + r * num_vars;
      engine.evaluate(ev, 1, num_vars, output.data(), target_id);
      volatile double sink = output[0];
      (void)sink;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    seconds_out = std::chrono::duration<double>(t1 - t0).count();
  };

  double cold_seconds = 0.0;
  double warm_seconds = 0.0;
  run_pass(cold_seconds);
  run_pass(warm_seconds);

  std::cout << "PYBNCORE_POINT_ROWS: " << batch_size << "\n";
  std::cout << "PYBNCORE_TIME_SECONDS_COLD: " << cold_seconds << "\n";
  std::cout << "PYBNCORE_TIME_SECONDS_WARM: " << warm_seconds << "\n";
  return 0;
}
