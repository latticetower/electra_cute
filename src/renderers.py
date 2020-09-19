import numpy as np
import cv2
import os


class ImageRenderer:
    def __init__(self):
        pass
    def __call__(self, data, image_path, **kwargs):
        img = data['raw_image'].copy()
        predictions = data['nodes']
        #print(predictions)
        for name, rectangles in predictions.items():
            for (x0, y0, x1, y1) in rectangles:
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        name = os.path.basename(image_path)
        if name.find(".") >= 0:
            name = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join("saves", name + "_nodes.png"), img)


class PlotlyRenderer:
    def __init__(self):
        pass
    def __call__(self, data, **kwargs):
        pass