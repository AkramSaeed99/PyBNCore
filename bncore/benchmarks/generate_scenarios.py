import pybncore
import random
import os

def generate(xdsl_path, output_path, num_scenarios=2000):
    wrapper = pybncore.PyBNCoreWrapper(xdsl_path)
    nodes = wrapper.nodes()
    
    with open(output_path, "w") as f:
        f.write(f"{num_scenarios}\n")
        for i in range(num_scenarios):
            # 5-15 random evidence nodes
            num_ev = random.randint(5, 15)
            ev_nodes = random.sample(nodes, num_ev)
            
            f.write(f"SCENARIO {i}\n")
            for node in ev_nodes:
                outcomes = wrapper.get_outcomes(node)
                state_idx = random.randint(0, len(outcomes) - 1)
                f.write(f"E {node} {state_idx}\n")
            
            # 5 random query nodes
            num_queries = 5
            query_nodes = random.sample([n for n in nodes if n not in ev_nodes], num_queries)
            for node in query_nodes:
                f.write(f"Q {node}\n")
            f.write("END\n")

if __name__ == "__main__":
    generate("data/comprehensive_bench.xdsl", "data/scenarios.txt")
