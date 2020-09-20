"""Main module with image processing pipeline with several stages:

1. Processing: simplyfy image to get better results.
Here we do color clustering and similar image transformations.
2. Select shapes: find similar shapes on image
3. Classify shapes
4. Connect shapes: find lines on image and make connections between shapes
5. use networkx to store graph data
6. draw
"""
import sys
import os
import cv2

from processors import GMMProcessor, CannyProcessor
from detectors import SimpleTemplateDetector
from renderers import ImageRenderer, PlotlyNodeRenderer

class GraphPipeline:
    def __init__(self, tempdir="images"):
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)
        self.tempdir = tempdir
        self.stages = [
            SimpleTemplateDetector(),
            #CannyProcessor()
            ImageRenderer(),
            PlotlyNodeRenderer()
        ]

    def process(self, image_path, i=0):
        path = os.path.join(self.tempdir, str(i))
        if not os.path.exists(path):
            os.makedirs(path)
        image = cv2.imread(image_path)
        data = {
            "raw_image": image
        }
        for s in self.stages:
            s(data, image_path=image_path)
        # print({k: type(v) for k, v in data.items()})
        
    

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Please provide path to image as a parameter")
        exit(1)
    image_path = sys.argv[1]
    pipeline = GraphPipeline()
    pipeline.process(image_path)