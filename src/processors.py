from sklearn.mixture import GaussianMixture
import numpy as np
import cv2

from utils import *

class GMMProcessor:
    def __init__(self, nclusters=4):
        if isinstance(nclusters, int):
            nclusters = [ nclusters ]
        self.nclusters = nclusters

    def __call__(self, data, **kwargs):
        #print(data)
        img = data["raw_image"].copy()
        etec  # COLOR_BGR2HSV
        print(img.shape)
        if len(img.shape) == 3:
            h, w, c = img.shape[-1]
        else:
            h, w = img.shape
            c = 1
        colors = img.reshape(-1, c)
        #if isinstance(self.nclusters, list):
        models = [ GaussianMixture(n) for n in self.nclusters ]
        [m.fit(colors) for m in models]
        aics = [m.aic(colors) for m in models]
        print(aics)
        i = np.argmin(aics)
        gmm = models[i]
        #gmm = GaussianMixture(self.nclusters)
        clusters = gmm.fit_predict(colors)
        clusters = clusters.astype(int).reshape((h, w))
        processed_image = gmm.means_[clusters].astype(np.uint8)
        data["clusters"] = clusters
        data["processed_image"] = processed_image
        return data


class LineExtractor:
    def __init__(self, minLineLength=10, maxLineGap=10):
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.minT = 50
        self.maxT = 150

    def __call__(self, data, **kwargs):
        img = data['raw_image'].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(gray, 50, 150,apertureSize = 3)
        edges = ((gray > self.minT) & (gray < self.maxT)).astype(np.uint8)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200,
            self.minLineLength, self.maxLineGap)
        data["lines"] = lines[:, 0]
        return data


class LineClusterizer:
    def __call__(self, data, **kwargs):
        
        all_points = []
        all_clusters = []
        for i, points in enumerate(get_line_clusters(data['lines'])):
            all_points.append(points)
            all_clusters.extend([i]*points.shape[0])
        data['line_clusters'] = np.asarray(all_clusters)
        data['line_points'] = np.concatenate(all_points)
        return data


class NodeConnector:
    def __call__(self, data, **kwargs):
        line_clusters = data["line_clusters"] #all_clusters
        line_points = data["line_points"] #all_points
        
        node_labels = data["node_labels"] #all_clusters
        node_rect = data["node_coords"] #all_points
        connection_matrix = get_nodes_connection_matrix(
            line_clusters, line_points,
            node_labels, node_rect)
        data['connection_matrix'] = connection_matrix
        return data

