from sklearn.mixture import GaussianMixture
import numpy as np
import cv2


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


class CannyProcessor:
    def __init__(self):
        pass

    def __call__(self, data, **kwargs):
        #print(data)
        return data

