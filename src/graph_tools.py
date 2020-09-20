import networkx as nx
FILE2TYPE = os.listdir

class NodeNxExtractor:
    def __call__(self, data, **kwargs):
        nodes = data["nodes"]
        for filename, coords in nodes.items():
            pass
        
        pass