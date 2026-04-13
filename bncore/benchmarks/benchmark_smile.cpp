#include "../deps/smile_cpp/smile.h"
#include "../deps/smile_license/smile_license.h"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  std::cout << "==================================================\n";
  std::cout << " SMILE (C++) Execution Benchmark\n";
  std::cout << "==================================================\n";

  DSL_network net;
  int res = net.ReadFile("data/benchmark_perf.xdsl");
  if (res != DSL_OKAY) {
    std::cerr << "SMILE failed to read file: data/benchmark_perf.xdsl"
              << std::endl;
    return 1;
  }

  std::cout << "Successfully loaded network into SMILE Core.\n";

  // Warmup
  net.UpdateBeliefs();

  int total_nodes = net.GetNumberOfNodes();
  std::cout << "Total Nodes: " << total_nodes << std::endl;

  // We try to benchmark equivalent heavy workloads.
  // PyBNcore ran 200,000 scenarios in 110s natively.
  // Let's run 5,000 scenarios dynamically clearing/updating in SMILE to project
  // its speed.
  int batch_size = 5000;

  std::cout << "\n[Baseline] Running SMILE UpdateBeliefs loop for "
            << batch_size << " queries..." << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < batch_size; ++i) {
    // We set and clear an evidence point to mark the network as dirty,
    // otherwise SMILE will cache the marginals and return instantly.
    int first_node = net.GetFirstNode();
    net.GetNode(first_node)->Value()->SetEvidence(0); // Arbitrary state
    net.UpdateBeliefs();
    net.GetNode(first_node)->Value()->ClearEvidence();
  }

  // One final update for the cleared evidence state
  net.UpdateBeliefs();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t1 - t0;

  double ext_time = diff.count() * (200000.0 / batch_size);

  std::cout << "SMILE Sequence computed: " << diff.count()
            << " s (Actual for 5k)\n";
  std::cout << "SMILE Projected 200,000 batch: ~" << ext_time << " s\n";
  return 0;
}
