'''kmeans.py
Performs K-Means clustering
Trey Tuscai
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Safe_10
from matplotlib.animation import FuncAnimation


class KMeans:
    def __init__(self, data=None):
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        self.data = data
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        return np.copy(self.data)

    def get_centroids(self):
        return self.centroids

    def get_data_centroid_labels(self):
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        return np.sqrt(np.sum(np.square(pt_1 - pt_2)))

    def dist_pt_to_centroids(self, pt, centroids):
        return np.sqrt(np.sum((centroids - pt) ** 2, axis=1))

    def initialize(self, k):
        centroid_indices = np.random.choice(self.num_samps, size=k, replace=False)
        centroids = self.data[centroid_indices]
        return centroids

    def cluster(self, k=3, tol=1e-2, max_iter=1000, verbose=False):
        self.k = k
        centroids = self.initialize(k)
        labels = self.update_labels(centroids)
        prev_centroids = None
        iters = 0
        converged = False
        
        while not converged and iters < max_iter:
            prev_centroids = np.copy(centroids)
            centroids, _ = self.update_centroids(k, labels, prev_centroids)
            labels = self.update_labels(centroids)
            
            if np.abs(prev_centroids - centroids).max() < tol:
                converged = True
            iters += 1
            
            if verbose:
                print(f"Iteration {iters}: Inertia={self.compute_inertia(labels, centroids):.4f}")
        
        self.centroids = centroids
        self.data_centroid_labels = labels
        self.inertia = self.compute_inertia()
        
        #print(f"K-means converged after {iters} iterations")
        
        return self.inertia, iters
    
    def animate(self, frames=20, interval=1000):
        fig, ax = plt.subplots()
        self.iters = 0
        def update(i):
            ax.clear()
            self.k = 5
            centroids = self.initialize(self.k)
            labels = self.update_labels(centroids)
            prev_centroids = None
            prev_centroids = np.copy(centroids)
            centroids, _ = self.update_centroids(self.k, labels, prev_centroids)
            labels = self.update_labels(centroids)
            ax.scatter(self.data[:, 0], self.data[:, 1], c=labels)
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
            ax.set_title(f"Iteration {self.iters}")
            self.centroids = centroids
            self.data_centroid_labels = labels
            self.inertia = self.compute_inertia()
            self.iters += 1 
            return ax
        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit = False)
        anim.save("animation.gif")


    def cluster_batch(self, k, n_iter=5, verbose=False):
        best_inertia = float('inf')
        for i in range(n_iter):
            inertia, _ = self.cluster(k=k, verbose=verbose)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy()
                best_data_centroid_labels = self.data_centroid_labels.copy()
                best_inertia = self.inertia
        self.centroids = best_centroids
        self.data_centroid_labels = best_data_centroid_labels
        self.inertia = best_inertia

    def update_labels(self, centroids):
        labels = np.zeros(self.data.shape[0], dtype=int)
        for i in range(self.data.shape[0]):
            dists = self.dist_pt_to_centroids(self.data[i], centroids)
            labels[i] = np.argmin(dists)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        new_centroids = np.zeros((k, self.num_features))
        counts = np.zeros((k,))
        for i in range(self.num_samps):
            cluster_index = data_centroid_labels[i]
            new_centroids[cluster_index] += self.data[i]
            counts[cluster_index] += 1
        for i in range(k):
            if counts[i] == 0:
                index = np.random.choice(self.num_samps)
                new_centroids[i] = self.data[index]
                counts[i] = 1
        new_centroids /= counts[:, np.newaxis]
        centroid_diff = new_centroids - prev_centroids
        return new_centroids, centroid_diff

    def compute_inertia(self):
        dists = np.sqrt(((self.data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        self.inertia = np.mean(np.min(dists, axis=0) ** 2)
        return self.inertia

    def plot_clusters(self):
        colors = Safe_10.mpl_colors
        for i in range(self.k):
            cluster_points = self.data[self.data_centroid_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i % len(colors)])
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', s=200, linewidths=2)
        plt.show()

    def elbow_plot(self, max_k, n_iter):
        inertias = []
        for k in range(1, max_k + 1):
            self.cluster_batch(k=k, n_iter = n_iter)
            inertias.append(self.inertia)
        plt.plot(range(1, max_k + 1), inertias, '-o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.xticks(range(1, max_k + 1))
        plt.show()

    def replace_color_with_centroid(self):
        self.data = self.centroids[self.data_centroid_labels].copy()
