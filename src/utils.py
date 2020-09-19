import cv2
import numpy as np

def unique_transforms(transforms):
    ids = set(np.arange(len(transforms)))
    indices = []
    while len(ids) > 0:
        i = ids.pop()
        indices.append(i)
        for j in list(ids):
            if transforms[j].shape == transforms[i].shape and np.all(transforms[j] == transforms[i]):
                ids.remove(j)
    return [transforms[i] for i in indices]


def shrink_template(template_rgb):
    if len(template_rgb.shape)==3:
        template = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
    else:
        template = template_rgb
    zero_template = template==0
    hstd = np.argwhere(zero_template.std(1) > 0)
    wstd = np.argwhere(zero_template.std(0) > 0)
    if len(hstd)==0 or len(wstd) == 0:
        return template_rgb
    hmin, hmax, wmin, wmax = hstd.min(), hstd.max()+1, wstd.min(), wstd.max()+1
    return template_rgb[hmin:hmax, wmin: wmax]
    
    
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# # Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
    
