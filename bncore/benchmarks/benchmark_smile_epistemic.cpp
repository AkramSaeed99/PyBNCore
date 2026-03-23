#include "../../smile_cpp/smile.h"
#include "../../smile_license (1)/smile_license.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr
        << "Usage: ./benchmark_smile_epistemic <xdsl_file> <batch_size> [target_node]\n";
    return 1;
  }

  int batch_size = std::stoi(argv[2]);

  DSL_network net;
  if (net.ReadFile(argv[1]) != DSL_OKAY) {
    std::cerr << "SMILE failed to read file: " << argv[1] << std::endl;
    return 1;
  }

  std::cout << "SMILE C++ Loaded Graph. Entering Epistemic Uncertainty Loop ("
            << batch_size << " samples)...\n";

  // Pre-generate random parameter samples for all nodes to isolate raw
  // Inference Time from RNG
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.1, 0.9);

  std::vector<int> all_nodes;
  int h = net.GetFirstNode();
  while (h >= 0) {
    all_nodes.push_back(h);
    h = net.GetNextNode(h);
  }
  std::vector<std::vector<std::vector<double>>> all_batched_cpts(
      all_nodes.size());

  for (size_t i = 0; i < all_nodes.size(); ++i) {
    int node_handle = all_nodes[i];
    DSL_node *node = net.GetNode(node_handle);
    const DSL_Dmatrix *def = node->Definition()->GetMatrix();
    int size = def->GetSize();
    int states = node->Definition()->GetNumberOfOutcomes();

    all_batched_cpts[i].resize(batch_size);
    for (int b = 0; b < batch_size; ++b) {
      std::vector<double> cpt(size);
      for (int k = 0; k < size; k += states) {
        double sum = 0.0;
        for (int s = 0; s < states; ++s) {
          cpt[k + s] = dist(rng);
          sum += cpt[k + s];
        }
        for (int s = 0; s < states; ++s) {
          cpt[k + s] /= sum;
        }
      }
      all_batched_cpts[i][b] = cpt;
    }
  }

  const char *target_name = (argc >= 4) ? argv[3] : "L9_N9";
  int target_node = net.FindNode(target_name);
  if (target_node < 0) {
    std::cerr << "Error: target node '" << target_name << "' not found.\n";
    return 1;
  }

  // Start benchmark
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int b = 0; b < batch_size; ++b) {
    // 1. Sequential Parameter Assignment
    for (size_t i = 0; i < all_nodes.size(); ++i) {
      DSL_node *node = net.GetNode(all_nodes[i]);
      DSL_doubleArray new_def;
      for (double val : all_batched_cpts[i][b]) {
        new_def.Add(val);
      }
      node->Definition()->SetDefinition(new_def);
    }

    // 2. Sequential Compile and Inference (No Evidence)
    net.UpdateBeliefs();

    // 3. (Optional) Retreive marginals here
    const DSL_Dmatrix *mat = net.GetNode(target_node)->Value()->GetMatrix();
    volatile double dummy = (*mat)[0];
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t1 - t0;

  std::cout << "SMILE_TIME_SECONDS: " << diff.count() << "\n";
  return 0;
}
