import cv2
import numpy as np
import os
from utils import *

class CircleDetector:
    def __init__(self):
        pass


class SimpleTemplateDetector:
    def __init__(self, symdir="data/data_SI/symbols_png"):
        self.symdir = symdir
        symbols = [
            os.path.join(symdir, x) for x in os.listdir(symdir)
            if x not in ["Pipe.png", "Terminal.png"]
        ]
        self.templates = [
            (os.path.basename(s), cv2.imread(s))
            for s in symbols]
        
    def __call__(self, data, **kwargs):
        img = data["raw_image"]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = img_gray.shape
        predictions = dict()
        for name, template_basic in self.templates:
            template_basic = shrink_template(template_basic)
            #if not np.any(template_basic == 0):
            #    continue
            transforms = [
                template_basic,
                #np.fliplr(template_basic),
                #np.flipud(template_basic)
            ]
            transforms = [
                np.rot90(t, i) for i in [0, 1, 2, 3] for t in transforms
            ]
            transforms = unique_transforms(transforms)
            for template in transforms:
                h, w = template.shape[:2]
                #print(template.shape, img_gray.shape)

                res = cv2.matchTemplate(img.copy(), template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where( res >= threshold)
                #print(res[loc])

                for pt in zip(*loc[::-1]):
                    if not name in predictions:
                        predictions[name] = []
                    #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
                    predictions[name].append(list(pt) + [pt[0] + w, pt[1] + h])
        predictions = {
            k: non_max_suppression_fast(np.asarray(v), 0.1)
            for k, v in predictions.items()
        }
        data['nodes'] = predictions
        return data



class ConflictResolver:
    "resolves conflict between nodes"
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        pass
