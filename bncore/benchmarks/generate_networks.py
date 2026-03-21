import xml.etree.ElementTree as ET
import xml.dom.minidom
import random
import argparse
import os

def generate_cpt_data(num_parents):
    """
    Randomly generates probabilities for a binary boolean variable.
    Since it's binary, each parent configuration requires 2 probabilities summing to 1.
    """
    num_configs = 1 << num_parents
    probs = []
    for _ in range(num_configs):
        p0 = random.uniform(0.1, 0.9)
        p1 = 1.0 - p0
        probs.extend([p0, p1])
    return " ".join(f"{p:.6f}" for p in probs)

def generate_xdsl(filename, num_layers, nodes_per_layer, max_in_degree):
    """
    Generates a massive layered Directed Acyclic Graph saved in Smile .XDSL XML format.
    """
    smile = ET.Element("smile", version="1.0", id="BenchmarkNetwork", numsamples="10000", discsamples="10000")
    nodes = ET.SubElement(smile, "nodes")
    
    layer_nodes = []
    
    for l in range(num_layers):
        current_layer = []
        for n in range(nodes_per_layer):
            node_id = f"L{l}_N{n}"
            current_layer.append(node_id)
            
            cpt = ET.SubElement(nodes, "cpt", id=node_id)
            ET.SubElement(cpt, "state", id="State0")
            ET.SubElement(cpt, "state", id="State1")
            
            parents = []
            if l > 0:
                # Pick random parents from previous layer
                num_parents = random.randint(1, max_in_degree)
                num_parents = min(num_parents, len(layer_nodes[l-1]))
                parents = random.sample(layer_nodes[l-1], num_parents)
                
                if parents:
                    parents_elem = ET.SubElement(cpt, "parents")
                    parents_elem.text = " ".join(parents)
                    
            probs_elem = ET.SubElement(cpt, "probabilities")
            probs_elem.text = generate_cpt_data(len(parents))
            
        layer_nodes.append(current_layer)
        
    # Formatting into clean readable XML
    xml_str = ET.tostring(smile, encoding='utf-8')
    parsed = xml.dom.minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="    ")
    
    # Strip minidom's default XML declaration and enforce strictly correct ISO header
    pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
    final_xml = '<?xml version="1.0" encoding="ISO-8859-1"?>\n' + pretty_xml
    
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="ISO-8859-1") as f:
        f.write(final_xml)
        
    print(f"Generated {filename}")
    print(f"Total Nodes: {num_layers * nodes_per_layer}")
    print(f"Max In-Degree: {max_in_degree}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programmatic XDSL Network Generator")
    parser.add_argument("--filename", type=str, default="data/benchmark_100.xdsl", help="Output file name")
    parser.add_argument("--layers", type=int, default=10, help="Number of graph layers")
    parser.add_argument("--width", type=int, default=10, help="Nodes per layer")
    parser.add_argument("--degree", type=int, default=3, help="Max in-degree per node")
    
    args = parser.parse_args()
    generate_xdsl(args.filename, args.layers, args.width, args.degree)
