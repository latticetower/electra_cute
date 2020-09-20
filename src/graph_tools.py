import networkx as nx
import numpy as np

class NodeBuilder:
    def __call__(self, data, **kwargs):
        assert "nodes" in data
        nodes = data["nodes"]
        graph = nx.Graph()
        node_coords = []
        node_labels = []
        for name, coords in nodes.items():
            coords = coords
            for x0, y0, x1, y1 in coords:
                graph.add_node(
                    len(node_coords), 
                    name=name,
                    x0 =x0, y0=y0, x1=x1, y1=y1)
                node_coords.append([x0, y0, x1, y1])
                node_labels.append(name)
        data['graph'] = graph
        data['node_coords'] = np.asarray(node_coords)
        data['node_labels'] = np.asarray(node_labels)
        return data


class EdgeBuilder:
    def __call__(self, data, **kwargs):
        line_clusters = data["line_clusters"] #all_clusters
        graph = data["graph"] #all_points
        
        connection_matrix = data["connection_matrix"] #all_clusters
        for a, b in np.argwhere(connection_matrix==1):
            graph.add_edge(a, b)
