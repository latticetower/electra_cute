import numpy as np
import cv2
import os
from PIL import Image
import plotly.graph_objects as go
import json
import networkx as nx
import igviz as ig

class ImageRenderer:
    def __init__(self, objects=["nodes", "lines"]):
        self.objects = objects

    def __call__(self, data, image_path, **kwargs):
        img = data['raw_image'].copy()
        if 'nodes' in self.objects:
            predictions = data['nodes']
            #print(predictions)
            for name, rectangles in predictions.items():
                for (x0, y0, x1, y1) in rectangles:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if 'lines' in self.objects:
            lines = data['lines']
            for x0, y0, x1, y1 in lines:
                cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 4)
        name = os.path.basename(image_path)
        if name.find(".") >= 0:
            name = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join("saves", name + "_nodes.png"), img)


class PlotlyNodeRenderer:
    def __init__(self):
        pass
        
    def __call__(self, data, image_path, **kwargs):
        predictions = data["nodes"]
        H, W = data['raw_image'].shape[:2]
        bg_image = Image.open(image_path)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=[0, 0.5, 1, 2, 2.2], y=[1.23, 2.5, 0.42, 3, 1])
            )

        for name, coords in predictions.items():
            for y, x, h, w in coords:
                fig.add_shape(
                    # unfilled Rectangle
                    type="rect",
                    x0=y,
                    y0=x,
                    x1=h,
                    y1=w,
                    line=dict(
                        color="RoyalBlue",
                    )
                )
        # Add images
        fig.add_layout_image(
                dict(
                    source=bg_image, #image_path,
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=W,
                    sizey=H,
                    sizing="stretch",
                    opacity=0.5,
                    layer="below")
        )
        # Set templates
        fig.update_layout(
            width=W,
            height=H,
            autosize=True,
            template="plotly_white")
            
        fig.update_yaxes(automargin=True, autorange="reversed")
        name = os.path.basename(image_path)
        if name.find(".") >= 0:
            name = os.path.splitext(name)[0]
        fig.write_html(os.path.join("saves", name + "_nodes_plot.html"))
        

class NetworkxRenderer:
    def __call__(self, data, image_path, **kwargs):
        graph = data['graph']
        #jsdata = nx.readwrite.json_graph.cytoscape_data(graph)
        #print(jsdata)
        fig = ig.plot(graph,
            "Electra? Cute!",
            #color_method="name",
            node_text=["name"]
        )
        name = os.path.basename(image_path)
        if name.find(".") >= 0:
            name = os.path.splitext(name)[0]
        fig.write_html(os.path.join("saves", name + "_finished.html"))
        #with open(os.path.join("saves", name + "_graph.json"), 'w') as f:
        #    json.dump(jsdata, f)