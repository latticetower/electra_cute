import cv2
import numpy as np
from scipy import spatial
import itertools
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster import dbscan

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
    

def get_line_clusters(lines):
    """Takes array with N lines [N, (x0, y0, x1, y1)], and clusters them.
    We need to connect lines if they have common points somewhere and we don't connect them otherwise.
    After that we do clustering to get sets of lines.
    """
    X = lines[:, :2]
    Y = lines[:, 2:]
    threshold = 10
    a = np.argwhere(cdist(X, Y) < threshold)
    b = np.argwhere(cdist(Y, X) < threshold)
    c = np.argwhere(cdist(X, X) < threshold)
    d = np.argwhere(cdist(Y, Y) < threshold)
    edges = np.concatenate([a,b, c, d])
    matrix = np.ones((X.shape[0], X.shape[0]))
    for k, v in edges:
        matrix[k, v] = 0
    core_samples, labels = dbscan(matrix, metric="precomputed", eps=0.5, min_samples=2)
    clusters = np.unique(labels[core_samples])
    for cluster in clusters:
        line_ids = np.argwhere(labels == cluster).flatten()
        yield lines[line_ids, :]
    #np.argwhere(labels==-1)
    for x in np.argwhere(labels==-1):
        yield lines[x, :]


def convert2centers(preds):
    XX = preds[:, :2]
    YY = preds[:, 2:]
    C = (XX + YY)/2.
    dif = XX - C
    
    return C, np.sqrt(dif[:, 0]**2+dif[:, 1]**2)


def get_nodes_connection_matrix(all_clusters, all_points, node_labels, node_rect):
    
    # 1. extract points and build KDTree
    X = all_points[:, :2]
    Y = all_points[:, 2:]
    
    point_ids = np.concatenate([np.arange(len(X))]*2)
    kdtree = spatial.KDTree(np.concatenate([X, Y]))
    # 2. get centers and radii for nodes 
    node_centers, node_radii = convert2centers(node_rect)
    # 3. query node centers with given radii against points in kdtree.
    all_connections = dict()
    for coords, r, node in zip(node_centers, node_radii, np.arange(len(node_labels))):
        ids = kdtree.query_ball_point(coords, r)

        ids_ = np.unique(point_ids[ids])
        for cluster in np.unique(all_clusters[ids_]):
            if not cluster in all_connections:
                all_connections[cluster] = set()
            all_connections[cluster].add(node)
    # 4. convert to connection matrix
    N = len(node_labels)
    
    connection_matrix = np.zeros((N, N))
    for nodes in all_connections.values():
        for i, j in itertools.combinations(nodes, 2):
            connection_matrix[i, j] = 1
            connection_matrix[j, i] = 1
    return connection_matrix
    #END
