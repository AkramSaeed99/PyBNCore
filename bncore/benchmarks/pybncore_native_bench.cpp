// pybncore_native_bench.cpp
// Native C++ benchmark for PyBNCore. Reads the same JSONL scenario files used
// by smile_bench_full so comparisons are apples-to-apples (no Python overhead).
// Each scenario uses its own query variable set (queries differ per row).
//
// Build (from bncore root, after cmake + make):
//   clang++ -O3 -std=c++17 \
//     -I include \
//     benchmarks/pybncore_native_bench.cpp \
//     build/libbncore_core.a \
//     -o benchmarks/pybncore_native_bench
//
// Output: pybncore_native\t<n_scen>\t<total_s>\t<avg_ms>

#include "bncore/graph/graph.hpp"
#include "bncore/inference/compiler.hpp"
#include "bncore/inference/engine.hpp"
#include "bncore/inference/junction_tree.hpp"
#include "bncore/io/xdsl_reader.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Minimal JSON string trimmer
// ---------------------------------------------------------------------------
static std::string json_strip(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n\"");
  size_t b = s.find_last_not_of(" \t\r\n\"");
  if (a == std::string::npos) return "";
  return s.substr(a, b - a + 1);
}

struct BenchScenario {
  std::vector<std::pair<std::string, std::string>> evidence; // node→state
  std::vector<std::string> queries;                          // query node names
};

static BenchScenario parse_scenario(const std::string &line) {
  BenchScenario sc;
  size_t ev_s = line.find("\"evidence\"");
  size_t q_s  = line.find("\"queries\"");
  if (ev_s == std::string::npos || q_s == std::string::npos) return sc;

  // Parse evidence object
  size_t brace   = line.find('{', ev_s);
  size_t brace_e = line.find('}', brace);
  std::string ev_str = line.substr(brace + 1, brace_e - brace - 1);
  for (size_t pos = 0; pos < ev_str.size();) {
    size_t colon = ev_str.find(':', pos);
    if (colon == std::string::npos) break;
    size_t comma = ev_str.find(',', colon);
    if (comma == std::string::npos) comma = ev_str.size();
    auto k = json_strip(ev_str.substr(pos, colon - pos));
    auto v = json_strip(ev_str.substr(colon + 1, comma - colon - 1));
    if (!k.empty() && !v.empty()) sc.evidence.push_back({k, v});
    pos = comma + 1;
  }

  // Parse queries array
  size_t arr_s = line.find('[', q_s);
  size_t arr_e = line.find(']', arr_s);
  std::string q_str = line.substr(arr_s + 1, arr_e - arr_s - 1);
  for (size_t pos = 0; pos < q_str.size();) {
    size_t comma = q_str.find(',', pos);
    if (comma == std::string::npos) comma = q_str.size();
    auto node = json_strip(q_str.substr(pos, comma - pos));
    if (!node.empty()) sc.queries.push_back(node);
    pos = comma + 1;
  }
  return sc;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <network.xdsl> <scenarios.jsonl>\n";
    return 1;
  }

  // Load + compile
  bncore::Graph graph;
  try { bncore::read_xdsl(argv[1], graph); }
  catch (const std::exception &e) {
    std::cerr << "Failed to load network: " << e.what() << "\n"; return 1;
  }
  auto jt = bncore::JunctionTreeCompiler::compile(graph, "min_fill");

  // Build name→id and name→state-index lookups
  std::size_t num_vars = graph.num_variables();
  std::unordered_map<std::string, bncore::NodeId> name_to_id;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> name_to_state;
  for (std::size_t i = 0; i < num_vars; ++i) {
    const auto &vm = graph.get_variable((bncore::NodeId)i);
    name_to_id[vm.name] = (bncore::NodeId)i;
    for (std::size_t s = 0; s < vm.states.size(); ++s)
      name_to_state[vm.name][vm.states[s]] = (int)s;
  }

  // Load all scenarios
  std::ifstream f(argv[2]);
  if (!f) { std::cerr << "Cannot open " << argv[2] << "\n"; return 1; }
  std::vector<BenchScenario> scenarios;
  std::string line;
  while (std::getline(f, line))
    if (!line.empty()) scenarios.push_back(parse_scenario(line));

  int N = (int)scenarios.size();
  if (N == 0) { std::cerr << "No scenarios\n"; return 1; }

  // Pre-build evidence int vectors (one per scenario)
  std::vector<std::vector<int>> ev_mats(N, std::vector<int>(num_vars, -1));
  for (int i = 0; i < N; ++i)
    for (const auto &kv : scenarios[i].evidence) {
      auto nit = name_to_id.find(kv.first);
      if (nit == name_to_id.end()) continue;
      auto sit = name_to_state[kv.first].find(kv.second);
      if (sit == name_to_state.at(kv.first).end()) continue;
      ev_mats[i][nit->second] = sit->second;
    }

  // Engine: batch_size=1, HCL point-mode equivalent
  bncore::BatchExecutionEngine engine(*jt, /*num_threads=*/1, /*chunk_size=*/1);

  // We keep a reusable output buffer — max possible size = num_vars * max_states.
  // For binary nets: num_vars * 2. Use a generous fixed size.
  std::vector<double> out_buf(num_vars * 16, 0.0);

  // Warmup: run first 5 scenarios (or N if fewer)
  for (int i = 0; i < std::min(5, N); ++i) {
    const auto &sc = scenarios[i];
    std::vector<bncore::NodeId> qvars;
    std::vector<std::size_t> offsets;
    offsets.push_back(0);
    for (const auto &qn : sc.queries) {
      auto it = name_to_id.find(qn);
      if (it == name_to_id.end()) continue;
      qvars.push_back(it->second);
      offsets.push_back(offsets.back() + graph.get_variable(it->second).states.size());
    }
    if (qvars.empty()) continue;
    try {
      engine.evaluate_multi(ev_mats[i].data(), 1, num_vars,
                            qvars.data(), qvars.size(),
                            offsets.data(), out_buf.data());
    } catch (...) {}
  }

  // Timed run: use each scenario's own query set
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    const auto &sc = scenarios[i];

    // Build per-scenario query vars and offsets
    std::vector<bncore::NodeId> qvars;
    std::vector<std::size_t> offsets;
    offsets.push_back(0);
    for (const auto &qn : sc.queries) {
      auto it = name_to_id.find(qn);
      if (it == name_to_id.end()) continue;
      qvars.push_back(it->second);
      offsets.push_back(offsets.back() + graph.get_variable(it->second).states.size());
    }
    if (qvars.empty()) continue;

    try {
      engine.evaluate_multi(ev_mats[i].data(), 1, num_vars,
                            qvars.data(), qvars.size(),
                            offsets.data(), out_buf.data());
    } catch (...) {}
  }
  auto t1 = std::chrono::high_resolution_clock::now();

  double total_s = std::chrono::duration<double>(t1 - t0).count();
  double avg_ms  = total_s * 1000.0 / N;
  std::cout << "pybncore_native\t" << N << "\t" << total_s << "\t" << avg_ms << "\n";
  return 0;
}
