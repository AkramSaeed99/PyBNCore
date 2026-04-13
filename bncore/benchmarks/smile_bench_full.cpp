// SMILE C++ Benchmark Runner
// Reads standardized JSONL scenario files and reports timing per scenario set.
// Build: clang++ -O3 -std=c++11 smile_bench_full.cpp -I../deps/smile_cpp
// -L../deps/smile_cpp -lsmile -o smile_bench_full Run:   ./smile_bench_full
// <network.xdsl> <scenarios.jsonl>

#include "../deps/smile_cpp/smile.h"
#include "../deps/smile_license/smile_license.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSON value extraction (no dependency on nlohmann or similar)
// We parse two things from each JSONL line:
//   "evidence": {"NodeA": "State1", ...}
//   "queries":  ["NodeX", ...]
// ---------------------------------------------------------------------------
struct Scenario {
  std::vector<std::pair<std::string, std::string>> evidence; // node -> state
  std::vector<std::string> queries;
};

static std::string strip(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r\n\"");
  size_t b = s.find_last_not_of(" \t\r\n\"");
  if (a == std::string::npos)
    return "";
  return s.substr(a, b - a + 1);
}

static Scenario parse_line(const std::string &line) {
  Scenario sc;
  // Evidence section
  size_t ev_start = line.find("\"evidence\"");
  size_t q_start = line.find("\"queries\"");
  if (ev_start != std::string::npos && q_start != std::string::npos) {
    size_t brace = line.find('{', ev_start);
    size_t brace_end = line.find('}', brace);
    std::string ev_str = line.substr(brace + 1, brace_end - brace - 1);
    // parse k:v pairs
    size_t pos = 0;
    while (pos < ev_str.size()) {
      size_t colon = ev_str.find(':', pos);
      if (colon == std::string::npos)
        break;
      size_t comma = ev_str.find(',', colon);
      if (comma == std::string::npos)
        comma = ev_str.size();
      std::string key = strip(ev_str.substr(pos, colon - pos));
      std::string val = strip(ev_str.substr(colon + 1, comma - colon - 1));
      if (!key.empty() && !val.empty())
        sc.evidence.push_back({key, val});
      pos = comma + 1;
    }
    // Queries array
    size_t arr_start = line.find('[', q_start);
    size_t arr_end = line.find(']', arr_start);
    std::string q_str = line.substr(arr_start + 1, arr_end - arr_start - 1);
    pos = 0;
    while (pos < q_str.size()) {
      size_t comma = q_str.find(',', pos);
      if (comma == std::string::npos)
        comma = q_str.size();
      std::string node = strip(q_str.substr(pos, comma - pos));
      if (!node.empty())
        sc.queries.push_back(node);
      pos = comma + 1;
    }
  }
  return sc;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <network.xdsl> <scenarios.jsonl>\n";
    return 1;
  }
  const char *net_path = argv[1];
  const char *scen_path = argv[2];

  DSL_network net;
  if (net.ReadFile(net_path) != DSL_OKAY) {
    std::cerr << "SMILE: failed to read " << net_path << "\n";
    return 1;
  }

  std::ifstream f(scen_path);
  if (!f.is_open()) {
    std::cerr << "Cannot open " << scen_path << "\n";
    return 1;
  }

  std::vector<Scenario> scenarios;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty())
      scenarios.push_back(parse_line(line));
  }

  int n_scen = (int)scenarios.size();
  if (n_scen == 0) {
    std::cerr << "No scenarios loaded\n";
    return 1;
  }

  // Warmup
  net.UpdateBeliefs();

  auto t0 = std::chrono::high_resolution_clock::now();

  for (const auto &sc : scenarios) {
    // Apply evidence
    for (const auto &kv : sc.evidence) {
      int handle = net.FindNode(kv.first.c_str());
      if (handle < 0)
        continue;
      DSL_node *node = net.GetNode(handle);
      const DSL_idArray *outcomes = node->Definition()->GetOutcomesNames();
      for (int s = 0; s < outcomes->NumItems(); ++s) {
        if (std::string((*outcomes)[s]) == kv.second) {
          node->Value()->SetEvidence(s);
          break;
        }
      }
    }
    // Run inference
    net.UpdateBeliefs();
    // Read queried marginals (forces belief retrieval)
    for (const auto &q : sc.queries) {
      int handle = net.FindNode(q.c_str());
      if (handle < 0)
        continue;
      const DSL_Dmatrix *beliefs = net.GetNode(handle)->Value()->GetMatrix();
      volatile double sum = 0.0;
      for (int s = 0; s < beliefs->GetSize(); ++s)
        sum += (*beliefs)[s];
      (void)sum;
    }
    // Clear evidence
    net.ClearAllEvidence();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  double avg_ms = total_s * 1000.0 / n_scen;

  std::cout << "SMILE"
            << "\t" << n_scen << "\t" << total_s << "\t" << avg_ms << "\n";
  return 0;
}
