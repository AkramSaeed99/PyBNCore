#include "../../smile_cpp/smile.h"
#include "../../smile_license (1)/smile_license.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./benchmark_smile_vector <xdsl_file> <evidence_csv>\n";
    return 1;
  }

  DSL_network net;
  if (net.ReadFile(argv[1]) != DSL_OKAY) {
    std::cerr << "SMILE failed to read file: " << argv[1] << std::endl;
    return 1;
  }

  // Read evidence CSV
  // Format: first row is variable IDs (e.g. L0_N1, L2_N3)
  // Subsequent rows are integers, -1 means no evidence
  std::ifstream file(argv[2]);
  std::string line;
  std::getline(file, line);
  std::stringstream ss(line);
  std::string token;

  std::vector<int> evidence_nodes;
  while (std::getline(ss, token, ',')) {
    int handle = net.FindNode(token.c_str());
    if (handle < 0) {
      std::cerr << "Error: Node " << token << " not found in network.\n";
      return 1;
    }
    evidence_nodes.push_back(handle);
  }

  std::vector<std::vector<int>> evidence_matrix;
  while (std::getline(file, line)) {
    std::vector<int> row;
    std::stringstream css(line);
    std::string val;
    while (std::getline(css, val, ',')) {
      row.push_back(std::stoi(val));
    }
    evidence_matrix.push_back(row);
  }

  int batch_size = evidence_matrix.size();
  int target_node = net.FindNode("L9_N9"); // Assuming 10x10 DAG target

  std::cout << "SMILE C++ Loaded " << batch_size << " evidence scenarios.\n";

  // Start benchmark
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < batch_size; ++i) {
    // Sequential SetEvidence
    for (size_t col = 0; col < evidence_nodes.size(); ++col) {
      if (evidence_matrix[i][col] != -1) {
        net.GetNode(evidence_nodes[col])
            ->Value()
            ->SetEvidence(evidence_matrix[i][col]);
      }
    }

    // Sequential Inference Update
    net.UpdateBeliefs();

    // (Optional) Retreive marginals here
    const DSL_Dmatrix *mat = net.GetNode(target_node)->Value()->GetMatrix();
    volatile double dummy = (*mat)[0];
    // Clear for next iteration
    net.ClearAllEvidence();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t1 - t0;

  std::cout << "SMILE_TIME_SECONDS: " << diff.count() << "\n";
  return 0;
}
