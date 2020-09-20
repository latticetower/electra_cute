import networkx as nx
import numpy as np

class NodeBuilder:
    def __call__(self, data, **kwargs):
        assert "nodes" in data
        nodes = data["nodes"]
        graph = nx.Graph()
        node_coords = []
        for name, coords in nodes.items():
            for x1, y1, x2, y2 in coords:
                graph.add_node(
                    len(node_coords), 
                    name=name,
                    coordinates =[x1, y1, x2, y2])
                node_coords.append([x1, y1, x2, y2])
        data['graph'] = graph
        data['node_coords'] = np.asarray(node_coords)
        return data


class EdgeBuilder:
    def __call__(self, data, **kwargs):
        pass