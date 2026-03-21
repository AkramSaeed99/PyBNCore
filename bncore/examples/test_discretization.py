import pybncore as bn

def test_discretization():
    graph = bn.Graph()
    # Add a continuous-like variable with a single unbounded bin
    v1 = graph.add_variable("Temp", ["Bin0"])
    
    manager = bn.DiscretizationManager(max_bins_per_var=4)
    
    print(f"Initial states for Temp: {graph.get_variable('Temp').states}")
    
    # Simulate a loop that repeatedly splits the lowest bin until maximum caps are hit
    iteration = 1
    while manager.should_split(graph, v1):
        # We always split bin 0 for demonstration
        manager.split_bin(graph, v1, 0)
        print(f"Iteration {iteration} metadata states: {graph.get_variable('Temp').states}")
        iteration += 1

    print("Discretization loop finished correctly. Max constraint reached!")

if __name__ == "__main__":
    test_discretization()
