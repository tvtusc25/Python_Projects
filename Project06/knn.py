'''knn.py
K-Nearest Neighbors algorithm for classification
Trey Tuscai
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        # num_classes: int. The number of classes in the dataset.
        self.num_classes = num_classes
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        num_test_samps = data.shape[0]
        preds = np.zeros(num_test_samps, dtype=int)
        for i in range(num_test_samps):
            dists = np.sqrt(np.sum((self.exemplars - data[i])**2, axis=1))
            idxs = np.argsort(dists)[:k]
            knn_labels = self.classes[idxs]
            counts = np.bincount(knn_labels.astype(int))
            preds[i] = np.argmax(counts)
        return preds

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)

    def plot_predictions(self, k, n_sample_pts):
        cmap = ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'])
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        y_pred = self.predict(data, k).reshape(n_sample_pts, n_sample_pts)
        plt.pcolormesh(x, y, y_pred, cmap=cmap)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def confusion_matrix(self, y, y_pred):
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        for i in range(len(y)):
            conf_mat[y[i], y_pred[i]] += 1

        return conf_mat
