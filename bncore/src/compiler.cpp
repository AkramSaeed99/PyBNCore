#include "bncore/inference/compiler.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

namespace {
void validate_commercial_license() {
  // Intentionally disabled for local/HCL integration builds.
}
} // namespace

namespace bncore {

std::vector<std::vector<NodeId>>
JunctionTreeCompiler::moralize(const Graph &graph) {
  std::size_t n = graph.num_variables();
  std::vector<std::set<NodeId>> adj(n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto &parents = graph.get_parents(i);
    const auto &children = graph.get_children(i);

    // Add directed edges as undirected
    for (NodeId p : parents) {
      adj[i].insert(p);
      adj[p].insert(i);
    }
    for (NodeId c : children) {
      adj[i].insert(c);
      adj[c].insert(i);
    }

    // Marry the parents
    for (std::size_t p1 = 0; p1 < parents.size(); ++p1) {
      for (std::size_t p2 = p1 + 1; p2 < parents.size(); ++p2) {
        adj[parents[p1]].insert(parents[p2]);
        adj[parents[p2]].insert(parents[p1]);
      }
    }
  }

  std::vector<std::vector<NodeId>> moral_graph(n);
  for (std::size_t i = 0; i < n; ++i) {
    moral_graph[i].assign(adj[i].begin(), adj[i].end());
  }
  return moral_graph;
}

std::unique_ptr<JunctionTree>
JunctionTreeCompiler::compile(const Graph &graph,
                              const std::string &heuristic) {
  validate_commercial_license();
  auto jt = std::make_unique<JunctionTree>(&graph);

  // Step 1: Moralize
  auto moral_graph = moralize(graph);

  // Step 2: Triangulate using min_fill
  std::size_t n = graph.num_variables();
  std::vector<bool> eliminated(n, false);
  std::vector<std::vector<NodeId>> adj = moral_graph;
  std::vector<std::vector<NodeId>> cliques;

  // Heuristic selection for elimination ordering:
  //   "min_weight"        — minimize product of state counts (default, best general)
  //   "min_fill"          — minimize number of fill edges added
  //   "min_degree"        — minimize number of active neighbours
  //   "weighted_min_fill" — minimize fill edges weighted by state count product
  enum class Heuristic { MinWeight, MinFill, MinDegree, WeightedMinFill };
  Heuristic h = Heuristic::MinWeight;
  if (heuristic == "min_fill")            h = Heuristic::MinFill;
  else if (heuristic == "min_degree")     h = Heuristic::MinDegree;
  else if (heuristic == "weighted_min_fill") h = Heuristic::WeightedMinFill;
  else if (heuristic != "min_weight" && heuristic != "min_fill")
    h = Heuristic::MinWeight;  // default fallback

  for (std::size_t step = 0; step < n; ++step) {
    std::size_t best_node = n;
    std::size_t best_score1 = std::numeric_limits<std::size_t>::max();
    std::size_t best_score2 = std::numeric_limits<std::size_t>::max();
    std::size_t best_score3 = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < n; ++i) {
      if (eliminated[i]) continue;

      std::vector<NodeId> neighbors;
      for (NodeId neighbor : adj[i])
        if (!eliminated[neighbor]) neighbors.push_back(neighbor);

      std::size_t fill_edges = 0;
      for (std::size_t u = 0; u < neighbors.size(); ++u)
        for (std::size_t v = u + 1; v < neighbors.size(); ++v)
          if (std::find(adj[neighbors[u]].begin(), adj[neighbors[u]].end(),
                        neighbors[v]) == adj[neighbors[u]].end())
            fill_edges++;

      // Compute clique weight = product of state counts
      std::size_t weight = graph.get_variable(i).states.size();
      for (NodeId nb : neighbors) {
        std::size_t ns = graph.get_variable(nb).states.size();
        if (weight > (std::numeric_limits<std::size_t>::max() / 2) / ns)
          weight = std::numeric_limits<std::size_t>::max() / 2;
        else
          weight *= ns;
      }
      std::size_t clique_sz = neighbors.size() + 1;

      // Compute weighted fill = sum of state_count products for each fill edge
      std::size_t weighted_fill = 0;
      if (h == Heuristic::WeightedMinFill) {
        for (std::size_t u = 0; u < neighbors.size(); ++u)
          for (std::size_t v = u + 1; v < neighbors.size(); ++v)
            if (std::find(adj[neighbors[u]].begin(), adj[neighbors[u]].end(),
                          neighbors[v]) == adj[neighbors[u]].end()) {
              std::size_t w = graph.get_variable(neighbors[u]).states.size() *
                              graph.get_variable(neighbors[v]).states.size();
              weighted_fill += w;
            }
      }

      // Score based on heuristic
      std::size_t s1, s2, s3;
      switch (h) {
        case Heuristic::MinWeight:
          s1 = weight; s2 = fill_edges; s3 = clique_sz; break;
        case Heuristic::MinFill:
          s1 = fill_edges; s2 = weight; s3 = clique_sz; break;
        case Heuristic::MinDegree:
          s1 = clique_sz; s2 = fill_edges; s3 = weight; break;
        case Heuristic::WeightedMinFill:
          s1 = weighted_fill; s2 = weight; s3 = clique_sz; break;
      }

      if (s1 < best_score1 ||
          (s1 == best_score1 && s2 < best_score2) ||
          (s1 == best_score1 && s2 == best_score2 && s3 < best_score3)) {
        best_score1 = s1;
        best_score2 = s2;
        best_score3 = s3;
        best_node   = i;
      }
    }

    eliminated[best_node] = true;
    std::vector<NodeId> neighbors;
    for (NodeId neighbor : adj[best_node]) {
      if (!eliminated[neighbor])
        neighbors.push_back(neighbor);
    }

    std::vector<NodeId> clique = neighbors;
    clique.push_back(best_node);
    std::sort(clique.begin(), clique.end());
    cliques.push_back(clique);

    for (std::size_t u = 0; u < neighbors.size(); ++u) {
      for (std::size_t v = u + 1; v < neighbors.size(); ++v) {
        auto nu = neighbors[u], nv = neighbors[v];
        if (std::find(adj[nu].begin(), adj[nu].end(), nv) == adj[nu].end()) {
          adj[nu].push_back(nv);
          adj[nv].push_back(nu);
        }
      }
    }
  }

  // Step 3: Extract maximum cliques
  std::vector<std::vector<NodeId>> max_cliques;
  std::sort(cliques.begin(), cliques.end(),
            [](const auto &a, const auto &b) { return a.size() > b.size(); });
  for (const auto &clique : cliques) {
    bool is_subset = false;
    for (const auto &max_clq : max_cliques) {
      if (std::includes(max_clq.begin(), max_clq.end(), clique.begin(),
                        clique.end())) {
        is_subset = true;
        break;
      }
    }
    if (!is_subset)
      max_cliques.push_back(clique);
  }

  for (const auto &clique : max_cliques) {
    std::vector<std::size_t> state_sizes;
    for (NodeId id : clique)
      state_sizes.push_back(graph.get_variable(id).states.size());
    jt->add_clique(clique, state_sizes);
  }

  // Step 4: Build Spanning Tree (Kruskal's)
  struct Edge {
    std::size_t u, v, weight;
    std::vector<NodeId> sepset;
  };
  std::vector<Edge> edges;
  for (std::size_t i = 0; i < max_cliques.size(); ++i) {
    for (std::size_t j = i + 1; j < max_cliques.size(); ++j) {
      std::vector<NodeId> intersection;
      std::set_intersection(max_cliques[i].begin(), max_cliques[i].end(),
                            max_cliques[j].begin(), max_cliques[j].end(),
                            std::back_inserter(intersection));
      if (!intersection.empty()) {
        edges.push_back({i, j, intersection.size(), intersection});
      }
    }
  }

  std::sort(edges.begin(), edges.end(),
            [](const Edge &a, const Edge &b) { return a.weight > b.weight; });

  std::vector<std::size_t> parent(max_cliques.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::size_t i) {
    std::size_t root = i;
    while (root != parent[root])
      root = parent[root];
    std::size_t curr = i;
    while (curr != root) {
      std::size_t nxt = parent[curr];
      parent[curr] = root;
      curr = nxt;
    }
    return root;
  };
  auto unite = [&](std::size_t i, std::size_t j) {
    std::size_t root_i = find(i);
    std::size_t root_j = find(j);
    if (root_i != root_j) {
      parent[root_i] = root_j;
      return true;
    }
    return false;
  };

  for (const auto &edge : edges) {
    if (unite(edge.u, edge.v))
      jt->add_separator(edge.u, edge.v, edge.sepset);
  }

  // Step 5: Initialize Base Potentials with CPTs
  for (auto &clq : jt->get_mutable_cliques()) {
    clq.base_potential.tensor().fill(1.0);
  }

  for (std::size_t i = 0; i < graph.num_variables(); ++i) {
    std::vector<NodeId> family;
    for (NodeId p : graph.get_parents(i))
      family.push_back(p);
    family.push_back(static_cast<NodeId>(i));

    for (auto &clq : jt->get_mutable_cliques()) {
      bool contains_family = true;
      for (NodeId n : family) {
        if (std::find(clq.scope.begin(), clq.scope.end(), n) ==
            clq.scope.end()) {
          contains_family = false;
          break;
        }
      }
      if (contains_family) {
        clq.assigned_cpts.push_back(i);
        break;
      }
    }
  }

  return jt;
}

} // namespace bncore
