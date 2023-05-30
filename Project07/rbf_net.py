'''rbf_net.py
Radial Basis Function Neural Network
Trey Tuscai
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import kmeans


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        self.k = num_hidden_units  # number of hidden units
        self.num_classes = num_classes  # number of output units
        self.prototypes = None  # hidden unit prototypes
        self.sigmas = None  # hidden unit sigmas
        self.wts = None  # weights connecting hidden and output layer neurons

    def get_prototypes(self):
        return self.prototypes

    def get_num_hidden_units(self):
        return self.k

    def get_num_output_units(self):
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        avg_dists = np.zeros((centroids.shape[0],))
        for i in range(centroids.shape[0]):
            index = np.where(cluster_assignments == i)[0]
            cluster_data = data[index]
            dists = kmeans_obj.dist_pt_to_centroids(cluster_data, centroids[i])
            avg_dists[i] = np.mean(dists)
        avg_dists = np.clip(avg_dists, a_min=1e-6, a_max=None)
        return avg_dists

    def initialize(self, data):
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(k = self.k)
        self.prototypes = kmeans_obj.get_centroids()
        cluster_assignments = kmeans_obj.get_data_centroid_labels()
        self.sigmas = self.avg_cluster_dist(data, self.prototypes, cluster_assignments, kmeans_obj)
        pass

    def linear_regression(self, A, y):
        A = np.hstack((A, np.ones((A.shape[0], 1))))
        c, _, _, _ = scipy.linalg.lstsq( A, y )
        return c

    def hidden_act(self, data):
        dists = np.sqrt(np.sum((data[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2))
        acts = np.exp(-dists ** 2 / (2 * self.sigmas ** 2))
        return acts

    def output_act(self, hidden_acts):
        hidden_acts_bias = np.hstack((hidden_acts, np.ones((hidden_acts.shape[0], 1))))
        output_acts = hidden_acts_bias @ self.wts
        return output_acts

    def train(self, data, y):
        self.initialize(data)
        classes = np.unique(y)
        new_y = np.zeros((len(y), len(classes)))
        for i, c in enumerate(classes):
            new_y[y == c, i] = 1
        hidden_acts = self.hidden_act(data)
        self.wts = self.linear_regression(hidden_acts, new_y)
        pass

    def predict(self, data):
        hidden_acts = self.hidden_act(data)
        output_acts = self.output_act(hidden_acts)
        predicted_classes = np.argmax(output_acts, axis=1)
        return predicted_classes

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)
